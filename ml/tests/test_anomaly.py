"""
test_anomaly.py
===============
Phase 2 — ML Anomaly Detection Tests
Assignee: Shreeraksha M

Tests:
  1. Model trains without error on normal data
  2. Output schema is exactly correct
  3. is_anomaly flag works (injected known outliers get flagged)
  4. Model reloads from pkl correctly
  5. Cold-start fallback triggers when sample count < MIN_TRAIN_SAMPLES
  6. No null values in output

Run with:
    pytest ml/tests/test_anomaly.py -v
"""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Patch the DATA paths before importing the modules under test, so tests
# don't need real files in data/processed/
# ---------------------------------------------------------------------------
# We use monkeypatching inside each test where needed.

from ml.anomaly_detector import detect_anomalies, run as detector_run, FEATURE_COLS
from ml.trainer import AnomalyTrainer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_features_df(n: int = 200, add_outliers: bool = True) -> pd.DataFrame:
    """Generate a synthetic features DataFrame for testing.

    Args:
        n: Number of normal rows.
        add_outliers: If True, append 5 obviously anomalous rows.

    Returns:
        DataFrame with log_id and all FEATURE_COLS.
    """
    rng = np.random.default_rng(seed=0)

    df = pd.DataFrame({
        "log_id": [f"log_{i}" for i in range(n)],
        "frequency_score": rng.normal(loc=1.0, scale=0.3, size=n).clip(0),
        "severity_weight": rng.uniform(0.1, 0.5, size=n),
        "temporal_delta": rng.exponential(scale=2.0, size=n),
        "counter_proximity": rng.uniform(0, 0.2, size=n),
        "session_id": [f"sess_{i // 20}" for i in range(n)],
    })

    if add_outliers:
        # Inject 5 rows with extreme feature values — these should be flagged
        outliers = pd.DataFrame({
            "log_id": [f"outlier_{i}" for i in range(5)],
            "frequency_score": [50.0, 45.0, 60.0, 55.0, 48.0],
            "severity_weight": [1.0, 1.0, 1.0, 1.0, 1.0],
            "temporal_delta": [100.0, 90.0, 120.0, 110.0, 95.0],
            "counter_proximity": [1.0, 1.0, 1.0, 1.0, 1.0],
            "session_id": ["sess_outlier"] * 5,
        })
        df = pd.concat([df, outliers], ignore_index=True)

    return df


@pytest.fixture
def normal_features_df() -> pd.DataFrame:
    return make_features_df(n=200, add_outliers=True)


@pytest.fixture
def tiny_features_df() -> pd.DataFrame:
    """DataFrame below MIN_TRAIN_SAMPLES for cold-start testing."""
    # MIN_TRAIN_SAMPLES is typically 50; we use 10 which is always below that
    return make_features_df(n=10, add_outliers=False)


@pytest.fixture
def tmp_model_dir(tmp_path: Path) -> Path:
    """Temporary model store directory."""
    d = tmp_path / "model_store"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Test 1: Model trains without error
# ---------------------------------------------------------------------------

class TestModelTraining:
    def test_detect_anomalies_runs_without_error(self, normal_features_df):
        """detect_anomalies should complete without exceptions on valid input."""
        result = detect_anomalies(normal_features_df)
        assert result is not None
        assert len(result) == len(normal_features_df)

    def test_trainer_retrain_runs_without_error(self, normal_features_df, tmp_model_dir):
        """AnomalyTrainer.retrain() should complete and save a .pkl."""
        with patch("ml.trainer.MODEL_STORE_DIR", tmp_model_dir):
            with patch("ml.trainer.RETRAIN_STATE_FILE", tmp_model_dir / "retrain_state.json"):
                trainer = AnomalyTrainer()
                model = trainer.retrain(normal_features_df)

        assert model is not None
        # Should have written one pkl and one json sidecar
        pkls = list(tmp_model_dir.glob("isolation_forest_v*.pkl"))
        jsons = list(tmp_model_dir.glob("isolation_forest_v*.json"))
        assert len(pkls) == 1
        assert len(jsons) == 1


# ---------------------------------------------------------------------------
# Test 2: Output schema is correct
# ---------------------------------------------------------------------------

class TestOutputSchema:
    EXPECTED_COLUMNS = {"log_id", "isolation_score", "zscore", "combined_score", "is_anomaly"}
    EXPECTED_DTYPES = {
        "log_id": "object",           # str → object in pandas
        "isolation_score": "float64",
        "zscore": "float64",
        "combined_score": "float64",
        "is_anomaly": "bool",
    }

    def test_column_names_exact(self, normal_features_df):
        result = detect_anomalies(normal_features_df)
        assert set(result.columns) == self.EXPECTED_COLUMNS

    def test_dtypes_correct(self, normal_features_df):
        result = detect_anomalies(normal_features_df)
        for col, expected_dtype in self.EXPECTED_DTYPES.items():
            actual = str(result[col].dtype)
            if expected_dtype == "object":
                assert actual in ("object", "str"), (
                f"Column '{col}': expected string dtype, got '{actual}'"
            )
            else:
                assert actual == expected_dtype, (
                f"Column '{col}': expected dtype '{expected_dtype}', got '{actual}'"
            )

    def test_row_count_preserved(self, normal_features_df):
        result = detect_anomalies(normal_features_df)
        assert len(result) == len(normal_features_df)

    def test_log_id_values_preserved(self, normal_features_df):
        result = detect_anomalies(normal_features_df)
        assert set(result["log_id"]) == set(normal_features_df["log_id"].astype(str))


# ---------------------------------------------------------------------------
# Test 3: is_anomaly flag works
# ---------------------------------------------------------------------------

class TestAnomalyFlag:
    def test_injected_outliers_are_flagged(self, normal_features_df):
        """The 5 injected extreme outliers should be flagged as anomalies."""
        result = detect_anomalies(normal_features_df)
        outlier_ids = {f"outlier_{i}" for i in range(5)}
        flagged = set(result[result["is_anomaly"]]["log_id"])

        # All 5 outliers should be flagged (they have extreme feature values)
        assert outlier_ids.issubset(flagged), (
            f"Expected outliers {outlier_ids - flagged} to be flagged but were not."
        )

    def test_combined_score_range(self, normal_features_df):
        """combined_score should be in [0, 1] since both components are normalised."""
        result = detect_anomalies(normal_features_df)
        assert result["combined_score"].between(0.0, 1.0).all(), (
            "combined_score contains values outside [0, 1]"
        )

    def test_is_anomaly_consistent_with_combined_score(self, normal_features_df):
        """is_anomaly should be True iff combined_score > ANOMALY_THRESHOLD."""
        from common.config import ANOMALY_THRESHOLD
        result = detect_anomalies(normal_features_df)
        expected_flag = result["combined_score"] > ANOMALY_THRESHOLD
        pd.testing.assert_series_equal(
            result["is_anomaly"].reset_index(drop=True),
            expected_flag.reset_index(drop=True),
            check_names=False,
        )


# ---------------------------------------------------------------------------
# Test 4: Model reloads from pkl correctly
# ---------------------------------------------------------------------------

class TestModelPersistence:
    def test_model_save_and_reload(self, normal_features_df, tmp_model_dir):
        """Model saved by retrain() should be loadable and produce same results."""
        with patch("ml.trainer.MODEL_STORE_DIR", tmp_model_dir):
            with patch("ml.trainer.RETRAIN_STATE_FILE", tmp_model_dir / "state.json"):
                trainer = AnomalyTrainer()
                original_model = trainer.retrain(normal_features_df)
                loaded_model = trainer.load_latest_model()

        assert loaded_model is not None

        # Both models should produce identical predictions on the same input
        available_cols = [c for c in FEATURE_COLS if c in normal_features_df.columns]
        X = normal_features_df[available_cols].fillna(0).values
        pred_original = original_model.predict(X)
        pred_loaded = loaded_model.predict(X)
        np.testing.assert_array_equal(pred_original, pred_loaded)

    def test_sidecar_json_content(self, normal_features_df, tmp_model_dir):
        """JSON sidecar should contain all required metadata fields."""
        with patch("ml.trainer.MODEL_STORE_DIR", tmp_model_dir):
            with patch("ml.trainer.RETRAIN_STATE_FILE", tmp_model_dir / "state.json"):
                trainer = AnomalyTrainer()
                trainer.retrain(normal_features_df)

        json_files = list(tmp_model_dir.glob("*.json"))
        assert len(json_files) >= 1
        sidecar = json.loads(json_files[0].read_text())

        required_keys = {"timestamp", "n_samples", "contamination", "feature_columns", "w1", "w2"}
        assert required_keys.issubset(sidecar.keys()), (
            f"Sidecar missing keys: {required_keys - sidecar.keys()}"
        )

    def test_load_returns_none_when_no_model_exists(self, tmp_model_dir):
        """load_latest_model should return None gracefully if store is empty."""
        with patch("ml.trainer.MODEL_STORE_DIR", tmp_model_dir):
            with patch("ml.trainer.RETRAIN_STATE_FILE", tmp_model_dir / "state.json"):
                trainer = AnomalyTrainer()
                result = trainer.load_latest_model()
        assert result is None


# ---------------------------------------------------------------------------
# Test 5: Cold-start fallback
# ---------------------------------------------------------------------------

class TestColdStart:
    def test_cold_start_triggers_when_below_threshold(self, tiny_features_df):
        """When n_samples < MIN_TRAIN_SAMPLES, isolation_score should be 0.0."""
        result = detect_anomalies(tiny_features_df)
        # In cold-start, isolation_score is forced to 0.0 for all rows
        assert (result["isolation_score"] == 0.0).all(), (
            "Expected all isolation_scores to be 0.0 during cold-start fallback."
        )

    def test_cold_start_still_produces_valid_schema(self, tiny_features_df):
        """Cold-start output must still satisfy the P4 schema contract."""
        result = detect_anomalies(tiny_features_df)
        expected_cols = {"log_id", "isolation_score", "zscore", "combined_score", "is_anomaly"}
        assert set(result.columns) == expected_cols

    def test_trainer_retrain_returns_none_on_cold_start(self, tiny_features_df, tmp_model_dir):
        """AnomalyTrainer.retrain() should return None (no model) on cold-start."""
        with patch("ml.trainer.MODEL_STORE_DIR", tmp_model_dir):
            with patch("ml.trainer.RETRAIN_STATE_FILE", tmp_model_dir / "state.json"):
                trainer = AnomalyTrainer()
                model = trainer.retrain(tiny_features_df)
        assert model is None

    def test_cold_start_warning_is_logged(self, tiny_features_df, caplog):
        """Cold-start should emit a WARNING log."""
        import logging
        with caplog.at_level(logging.WARNING):
            detect_anomalies(tiny_features_df)
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("COLD-START" in msg or "cold" in msg.lower() for msg in warning_messages), (
            "Expected a cold-start WARNING log message."
        )


# ---------------------------------------------------------------------------
# Test 6: No null values in output
# ---------------------------------------------------------------------------

class TestNoNullValues:
    def test_no_nulls_normal_run(self, normal_features_df):
        result = detect_anomalies(normal_features_df)
        null_counts = result.isnull().sum()
        assert null_counts.sum() == 0, (
            f"anomaly_df contains nulls — P4 contract violation:\n{null_counts}"
        )

    def test_no_nulls_cold_start(self, tiny_features_df):
        result = detect_anomalies(tiny_features_df)
        null_counts = result.isnull().sum()
        assert null_counts.sum() == 0, (
            f"anomaly_df contains nulls in cold-start mode:\n{null_counts}"
        )

    def test_no_nulls_with_partial_features(self):
        """Even if P1 hasn't delivered all FEATURE_COLS, output should be null-free."""
        # Only provide one feature column
        df = pd.DataFrame({
            "log_id": [f"log_{i}" for i in range(100)],
            "frequency_score": np.random.rand(100),   # only one of the four cols
            "session_id": [f"sess_{i // 10}" for i in range(100)],
        })
        result = detect_anomalies(df)
        assert result.isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# Test 7: Periodic retrain trigger
# ---------------------------------------------------------------------------

class TestRetrainTrigger:
    def test_maybe_retrain_fires_after_k_logs(self, normal_features_df, tmp_model_dir):
        """maybe_retrain should trigger when new log count crosses RETRAIN_EVERY_K_LOGS."""
        from common.config import RETRAIN_EVERY_K_LOGS
        with patch("ml.trainer.MODEL_STORE_DIR", tmp_model_dir):
            with patch("ml.trainer.RETRAIN_STATE_FILE", tmp_model_dir / "state.json"):
                trainer = AnomalyTrainer()
                # Simulate that last retrain was at 0 logs
                trainer._logs_seen_at_last_retrain = 0
                # features_df has > RETRAIN_EVERY_K_LOGS rows → should trigger
                big_df = make_features_df(n=max(RETRAIN_EVERY_K_LOGS + 10, 210))
                model = trainer.maybe_retrain(big_df)
        # Should have trained a model
        assert model is not None

    def test_maybe_retrain_does_not_fire_below_k(self, tmp_model_dir):
        """maybe_retrain should NOT retrain when new log count is below K."""
        from common.config import RETRAIN_EVERY_K_LOGS
        with patch("ml.trainer.MODEL_STORE_DIR", tmp_model_dir):
            with patch("ml.trainer.RETRAIN_STATE_FILE", tmp_model_dir / "state.json"):
                trainer = AnomalyTrainer()
                trainer._logs_seen_at_last_retrain = 0
                # Only 5 new logs — far below K
                small_df = make_features_df(n=5, add_outliers=False)
                model = trainer.maybe_retrain(small_df)
        assert model is None