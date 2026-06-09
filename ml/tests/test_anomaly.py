"""
ml/tests/test_anomaly.py
========================
Phase 3 — ML Anomaly Detection Tests
Assignee: Shreeraksha M

Covers:
  anomaly_detector — output schema, score bounds, cold-start, full-confidence,
                     identical-scores edge case, NaN/inf row dropping,
                     parquet persistence
  trainer          — pkl + sidecar creation, sidecar keys, versioned filenames,
                     chronological sliding window, maybe_retrain trigger
  evaluator        — graceful fallback without ground truth, classification
                     metrics with ground truth, summary file written

Run with:
    pytest ml/tests/test_anomaly.py -v
"""

from __future__ import annotations

import json
import logging
import tempfile
from datetime import datetime as _real_datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from common.config import (
    ANOMALY_SCORE_THRESHOLD,
    ANOMALY_DYNAMIC_K,
    ANOMALY_FLAG_MODE,
    ANOMALY_CONTAMINATION,
    COLD_START_FULL_CONFIDENCE_THRESHOLD,
    IF_FEATURE_COLUMNS,
    RETRAINING_TRIGGER_EVERY_K,
)
from ml.anomaly_detector import OUTPUT_COLUMNS, detect, run
from ml.evaluator import run_evaluation
from ml.trainer import AnomalyTrainer


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def build_features_df(n_rows: int = 200, n_sessions: int = 5) -> pd.DataFrame:
    """Generate a valid synthetic features_df with all required columns.

    All float columns have finite, reasonable values. No NaN or inf.
    """
    rng = np.random.default_rng(0)
    templates = ["ROUTE", "PORT_DOWN", "LOGIN", "HEALTH"]
    return pd.DataFrame({
        "sequence_number":          np.arange(1, n_rows + 1),
        "session_id":               [f"sess_{i % n_sessions}" for i in range(n_rows)],
        "template_id":              [templates[i % len(templates)] for i in range(n_rows)],
        "host":                     ["host-01" if i % 2 == 0 else "host-02" for i in range(n_rows)],
        "timestamp":                pd.date_range("2026-01-01", periods=n_rows, freq="1s"),
        "event_weight":             rng.choice([0.1, 0.4, 0.7, 1.0], n_rows),
        "frequency_score":          rng.uniform(0.0, 1.0, n_rows),
        "burstiness_score":         rng.uniform(0.0, 5.0, n_rows),
        "zscore_base":              rng.uniform(-3.0, 3.0, n_rows),
        "time_delta_prev":          rng.exponential(5.0, n_rows),
        "time_delta_session_start": rng.uniform(0.0, 1800.0, n_rows),
        "inter_arrival_rate":       rng.exponential(5.0, n_rows),
        "counter_proximity":        rng.uniform(0.0, 1.0, n_rows),
        # Section-4 metric features (values + present flags)
        "metric_zscore":            rng.uniform(0.0, 4.0, n_rows),
        "metric_zscore_present":    rng.choice([0.0, 1.0], n_rows),
        "drop_rate":                rng.uniform(0.0, 100.0, n_rows),
        "drop_rate_present":        rng.choice([0.0, 1.0], n_rows),
        "utilization":              rng.uniform(0.0, 100.0, n_rows),
        "utilization_present":      rng.choice([0.0, 1.0], n_rows),
    })


def _make_anomaly_df(n_rows: int = 100) -> pd.DataFrame:
    """Minimal anomaly_df matching the AnomalyRow schema."""
    rng = np.random.default_rng(1)
    combined = rng.uniform(0.0, 1.0, n_rows)
    return pd.DataFrame({
        "sequence_number":  np.arange(1, n_rows + 1),
        "isolation_score":  rng.uniform(0.0, 1.0, n_rows),
        "zscore_norm":      rng.uniform(0.0, 1.0, n_rows),
        "combined_score":   combined,
        "is_anomaly":       combined > ANOMALY_SCORE_THRESHOLD,
        "model_confidence": 1.0,
    })


def _make_eval_features_df(n_rows: int = 100) -> pd.DataFrame:
    """Minimal features_df for evaluator tests (only needs sequence_number + template_id)."""
    templates = ["ROUTE", "PORT_DOWN", "LOGIN", "HEALTH"]
    return pd.DataFrame({
        "sequence_number": np.arange(1, n_rows + 1),
        "template_id":     [templates[i % len(templates)] for i in range(n_rows)],
    })


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def features_df() -> pd.DataFrame:
    return build_features_df(n_rows=200, n_sessions=5)


@pytest.fixture
def full_confidence_df() -> pd.DataFrame:
    """DataFrame large enough to trigger full model confidence (>= threshold)."""
    return build_features_df(n_rows=COLD_START_FULL_CONFIDENCE_THRESHOLD, n_sessions=10)


@pytest.fixture
def tmp_model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "model_store"
    d.mkdir()
    return d


# ===========================================================================
# anomaly_detector tests
# ===========================================================================

class TestDetectOutputSchema:
    """detect() produces the correct output schema."""

    def test_all_output_columns_present(self, features_df):
        result = detect(features_df)
        assert list(result.columns) == OUTPUT_COLUMNS, (
            f"Expected columns {OUTPUT_COLUMNS}, got {list(result.columns)}"
        )

    def test_row_count_preserved(self, features_df):
        result = detect(features_df)
        assert len(result) == len(features_df)

    def test_no_nulls_in_output(self, features_df):
        result = detect(features_df)
        null_counts = result.isnull().sum()
        assert null_counts.sum() == 0, f"Unexpected nulls:\n{null_counts}"

    def test_is_anomaly_bool_dtype(self, features_df):
        result = detect(features_df)
        assert result["is_anomaly"].dtype == bool, (
            f"is_anomaly dtype should be bool, got {result['is_anomaly'].dtype}"
        )


class TestDetectScoreBounds:
    """All score columns stay within [0.0, 1.0]."""

    def test_isolation_score_in_unit_interval(self, features_df):
        result = detect(features_df)
        assert result["isolation_score"].between(0.0, 1.0).all(), (
            "isolation_score contains values outside [0, 1]"
        )

    def test_zscore_norm_in_unit_interval(self, features_df):
        result = detect(features_df)
        assert result["zscore_norm"].between(0.0, 1.0).all(), (
            "zscore_norm contains values outside [0, 1]"
        )

    def test_combined_score_in_unit_interval(self, features_df):
        result = detect(features_df)
        assert result["combined_score"].between(0.0, 1.0).all(), (
            "combined_score contains values outside [0, 1]"
        )

    def test_model_confidence_in_unit_interval(self, features_df):
        result = detect(features_df)
        conf = result["model_confidence"].iloc[0]
        assert 0.0 <= conf <= 1.0, f"model_confidence {conf} outside [0, 1]"

    def test_is_anomaly_consistent_with_threshold(self, features_df):
        result = detect(features_df)
        scores = result["combined_score"].values
        score_std = float(scores.std())
        # Mirror the detector's configured flag strategy (see anomaly_detector Step 6).
        if score_std < 1e-6:
            expected = result["combined_score"] > ANOMALY_SCORE_THRESHOLD
        elif ANOMALY_FLAG_MODE == "quantile":
            threshold = float(np.quantile(scores, 1.0 - ANOMALY_CONTAMINATION))
            expected = result["combined_score"] >= threshold
        else:  # dynamic_k
            threshold = float(scores.mean()) + ANOMALY_DYNAMIC_K * score_std
            expected = result["combined_score"] > threshold
        pd.testing.assert_series_equal(
            result["is_anomaly"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )


class TestDetectColdStart:
    """Confidence scaling: n_samples < threshold → model_confidence < 1.0."""

    def _cold_df(self) -> pd.DataFrame:
        # 100 rows → confidence = 100 / COLD_START_FULL_CONFIDENCE_THRESHOLD < 1.0
        return build_features_df(n_rows=100, n_sessions=3)

    def test_cold_start_model_confidence_below_one(self):
        result = detect(self._cold_df())
        conf = result["model_confidence"].iloc[0]
        assert conf < 1.0, (
            f"Expected model_confidence < 1.0 for n < threshold, got {conf}"
        )

    def test_cold_start_confidence_value_correct(self):
        df = self._cold_df()
        result = detect(df)
        expected_conf = len(df) / COLD_START_FULL_CONFIDENCE_THRESHOLD
        assert result["model_confidence"].iloc[0] == pytest.approx(expected_conf, abs=1e-9)

    def test_cold_start_combined_closer_to_zscore_norm(self):
        """At low confidence the blend leans heavily on zscore_norm.

        Formally: mean|combined - zscore_norm| < mean|combined - isolation_score|.
        """
        result = detect(self._cold_df())
        diff_z   = (result["combined_score"] - result["zscore_norm"]).abs().mean()
        diff_iso = (result["combined_score"] - result["isolation_score"]).abs().mean()
        assert diff_z < diff_iso, (
            f"At low confidence combined should be closer to zscore_norm "
            f"(diff_z={diff_z:.4f}, diff_iso={diff_iso:.4f})"
        )


class TestDetectFullConfidence:
    """n_samples >= threshold → model_confidence == 1.0."""

    def test_full_confidence_value(self, full_confidence_df):
        result = detect(full_confidence_df)
        assert result["model_confidence"].iloc[0] == pytest.approx(1.0)


class TestDetectEdgeCases:
    """Edge cases: identical IF scores, NaN/inf rows."""

    def test_identical_if_scores_isolation_is_half(self, features_df):
        """When all decision_function values are equal, isolation_score == 0.5."""
        mock_pipeline = MagicMock()
        mock_pipeline.n_samples_seen_ = COLD_START_FULL_CONFIDENCE_THRESHOLD
        mock_pipeline.decision_function.return_value = np.ones(len(features_df))

        with patch("ml.anomaly_detector._train_model", return_value=mock_pipeline):
            result = detect(features_df)

        assert np.allclose(result["isolation_score"].values, 0.5), (
            "All identical raw scores should produce isolation_score == 0.5"
        )

    def test_nan_and_inf_rows_dropped_before_scoring(self, features_df, caplog):
        """Rows with NaN or inf in feature columns are excluded from output."""
        df = features_df.copy()
        df.loc[0, "frequency_score"] = np.nan
        df.loc[1, "burstiness_score"] = np.inf
        n_bad = 2

        # Mock _train_model so StandardScaler doesn't choke on NaN during fitting
        mock_pipeline = MagicMock()
        mock_pipeline.n_samples_seen_ = len(df)
        mock_pipeline.decision_function.return_value = np.zeros(len(df) - n_bad)

        with patch("ml.anomaly_detector._train_model", return_value=mock_pipeline):
            with caplog.at_level(logging.WARNING):
                result = detect(df)

        assert len(result) == len(df) - n_bad, (
            f"Expected {len(df) - n_bad} rows after dropping NaN/inf, "
            f"got {len(result)}"
        )
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("Dropping" in m for m in warning_msgs), (
            f"Expected a WARNING containing 'Dropping'. Got: {warning_msgs}"
        )


class TestDetectParquetPersistence:
    """run() saves anomaly_df as parquet, reloadable with correct schema."""

    def test_run_saves_parquet_and_schema_is_correct(self, features_df, tmp_path):
        feat_path = tmp_path / "features_df.parquet"
        out_path = tmp_path / "anomaly_df.parquet"
        features_df.to_parquet(feat_path, index=False)

        run(features_path=feat_path, output_path=out_path)

        assert out_path.exists(), "anomaly_df.parquet was not written"
        loaded = pd.read_parquet(out_path)
        assert list(loaded.columns) == OUTPUT_COLUMNS
        assert len(loaded) == len(features_df)
        assert loaded.isnull().sum().sum() == 0


# ===========================================================================
# trainer tests
# ===========================================================================

class TestTrainerRetrain:
    """retrain() persists model and sidecar correctly."""

    def test_retrain_creates_pkl(self, features_df, tmp_model_dir):
        with patch("ml.trainer.MODEL_STORE_DIR", tmp_model_dir), \
             patch("ml.trainer.RETRAIN_STATE_FILE", tmp_model_dir / "state.json"):
            trainer = AnomalyTrainer()
            trainer.retrain(features_df)

        pkls = list(tmp_model_dir.glob("isolation_forest_v*.pkl"))
        assert len(pkls) == 1, f"Expected 1 pkl, found {len(pkls)}"

    def test_retrain_creates_json_sidecar(self, features_df, tmp_model_dir):
        with patch("ml.trainer.MODEL_STORE_DIR", tmp_model_dir), \
             patch("ml.trainer.RETRAIN_STATE_FILE", tmp_model_dir / "state.json"):
            trainer = AnomalyTrainer()
            trainer.retrain(features_df)

        jsons = list(tmp_model_dir.glob("isolation_forest_v*.json"))
        assert len(jsons) == 1, f"Expected 1 JSON sidecar, found {len(jsons)}"

    def test_sidecar_has_all_required_keys(self, features_df, tmp_model_dir):
        with patch("ml.trainer.MODEL_STORE_DIR", tmp_model_dir), \
             patch("ml.trainer.RETRAIN_STATE_FILE", tmp_model_dir / "state.json"):
            trainer = AnomalyTrainer()
            trainer.retrain(features_df)

        sidecar = json.loads(
            next(tmp_model_dir.glob("isolation_forest_v*.json")).read_text()
        )
        required = {
            "timestamp", "n_samples", "contamination", "feature_columns",
            "isolation_weight", "zscore_weight", "cold_start_threshold",
        }
        missing = required - sidecar.keys()
        assert not missing, f"Sidecar missing keys: {missing}"

    def test_consecutive_retrains_produce_different_filenames(
        self, features_df, tmp_model_dir
    ):
        """Each retrain() call generates a uniquely versioned filename."""
        times = iter([
            _real_datetime(2026, 1, 1, 10, 0, 1),
            _real_datetime(2026, 1, 1, 10, 0, 2),
        ])
        with patch("ml.trainer.MODEL_STORE_DIR", tmp_model_dir), \
             patch("ml.trainer.RETRAIN_STATE_FILE", tmp_model_dir / "state.json"), \
             patch("ml.trainer.datetime") as mock_dt:
            mock_dt.now.side_effect = times
            trainer = AnomalyTrainer()
            trainer.retrain(features_df)
            trainer.retrain(features_df)

        pkls = sorted(tmp_model_dir.glob("isolation_forest_v*.pkl"))
        assert len(pkls) == 2, f"Expected 2 versioned pkl files, got {len(pkls)}"
        assert pkls[0].stem != pkls[1].stem


class TestTrainerSlidingWindow:
    """_apply_sliding_window() returns the chronologically last N sessions."""

    def _make_ordered_df(self) -> pd.DataFrame:
        """Three sessions; alphabetical name order is OPPOSITE of chronological order.

        sess_A → 2026-01-10  (most recent)
        sess_B → 2026-01-05  (middle)
        sess_C → 2026-01-01  (oldest)

        Alphabetically last 2 = {sess_B, sess_C}.
        Chronologically last 2 = {sess_A, sess_B}  ← correct.
        """
        session_starts = {
            "sess_A": "2026-01-10",
            "sess_B": "2026-01-05",
            "sess_C": "2026-01-01",
        }
        rows_per = 40
        dfs = []
        for i, (sid, start) in enumerate(session_starts.items()):
            chunk = build_features_df(n_rows=rows_per, n_sessions=1)
            chunk["session_id"] = sid
            chunk["timestamp"] = pd.date_range(start, periods=rows_per, freq="1min")
            chunk["sequence_number"] = np.arange(i * rows_per + 1, (i + 1) * rows_per + 1)
            dfs.append(chunk)
        # Shuffle to break insertion order
        return pd.concat(dfs).sample(frac=1, random_state=7).reset_index(drop=True)

    def test_window_is_chronologically_last_n(self, tmp_model_dir):
        df = self._make_ordered_df()

        with patch("ml.trainer.MODEL_STORE_DIR", tmp_model_dir), \
             patch("ml.trainer.RETRAIN_STATE_FILE", tmp_model_dir / "state.json"), \
             patch("ml.trainer.RETRAINING_SESSION_WINDOW", 2):
            trainer = AnomalyTrainer()
            windowed = trainer._apply_sliding_window(df)

        actual = set(windowed["session_id"].unique())
        expected = {"sess_A", "sess_B"}   # two most recent by timestamp
        wrong    = {"sess_B", "sess_C"}   # what alphabetical ordering would return
        assert actual == expected, (
            f"Expected chronologically last 2 sessions {expected}, got {actual}. "
            f"Alphabetical ordering would have returned {wrong}."
        )


class TestTrainerMaybeRetrain:
    """maybe_retrain() fires / suppresses retrain correctly."""

    def test_below_trigger_threshold_no_retrain(self, tmp_model_dir):
        """Below K new logs: no pkl saved, count updated."""
        small_df = build_features_df(n_rows=50, n_sessions=2)
        with patch("ml.trainer.MODEL_STORE_DIR", tmp_model_dir), \
             patch("ml.trainer.RETRAIN_STATE_FILE", tmp_model_dir / "state.json"):
            trainer = AnomalyTrainer()
            trainer._unprocessed_logs_count = 0
            result = trainer.maybe_retrain(small_df, new_logs_count=len(small_df))

        assert result is None, "maybe_retrain should return None below K threshold"
        pkls = list(tmp_model_dir.glob("isolation_forest_v*.pkl"))
        assert len(pkls) == 0, "No pkl should be saved below K threshold"

    def test_at_trigger_threshold_retrain_fires(self, tmp_model_dir):
        """At or above K new logs: retrain triggered, new pkl saved."""
        # Build df large enough to exceed RETRAINING_TRIGGER_EVERY_K
        n = RETRAINING_TRIGGER_EVERY_K + 10
        big_df = build_features_df(n_rows=n, n_sessions=10)
        with patch("ml.trainer.MODEL_STORE_DIR", tmp_model_dir), \
             patch("ml.trainer.RETRAIN_STATE_FILE", tmp_model_dir / "state.json"):
            trainer = AnomalyTrainer()
            trainer._unprocessed_logs_count = 0
            result = trainer.maybe_retrain(big_df, new_logs_count=n)

        assert result is not None, "maybe_retrain should return a pipeline at K threshold"
        pkls = list(tmp_model_dir.glob("isolation_forest_v*.pkl"))
        assert len(pkls) == 1, f"Expected 1 pkl after trigger, got {len(pkls)}"


# ===========================================================================
# evaluator tests
# ===========================================================================

class TestEvaluator:
    """run_evaluation() metrics and report file."""

    @staticmethod
    def _patch_summary(tmp_path: Path):
        """Return a context-manager patch that redirects summary writes to tmp_path."""
        return patch("ml.evaluator.SUMMARY_PATH", tmp_path / "anomaly_summary.txt")

    def test_no_ground_truth_no_crash(self, tmp_path, caplog):
        """run_evaluation() must not raise when data/synthetic/ has no ground truth."""
        anomaly_df  = _make_anomaly_df()
        features_df = _make_eval_features_df()

        with patch("ml.evaluator.SYNTHETIC_DIR", tmp_path / "empty_synthetic"), \
             self._patch_summary(tmp_path):
            # Must not raise
            metrics = run_evaluation(anomaly_df, features_df)

        assert metrics is not None

    def test_no_ground_truth_warning_logged(self, tmp_path, caplog):
        anomaly_df  = _make_anomaly_df()
        features_df = _make_eval_features_df()

        with patch("ml.evaluator.SYNTHETIC_DIR", tmp_path / "empty_synthetic"), \
             self._patch_summary(tmp_path):
            with caplog.at_level(logging.WARNING):
                run_evaluation(anomaly_df, features_df)

        assert any(
            "No ground truth" in r.message
            for r in caplog.records if r.levelno == logging.WARNING
        ), "Expected a WARNING about missing ground truth"

    def test_distribution_metrics_always_present(self, tmp_path):
        """Distribution metrics appear regardless of ground truth availability."""
        anomaly_df  = _make_anomaly_df()
        features_df = _make_eval_features_df()

        with patch("ml.evaluator.SYNTHETIC_DIR", tmp_path / "no_synthetic"), \
             self._patch_summary(tmp_path):
            metrics = run_evaluation(anomaly_df, features_df)

        for key in (
            "anomaly_rate", "mean_combined_score", "score_distribution",
            "top_anomalous_templates", "model_confidence",
        ):
            assert key in metrics, f"Expected '{key}' in metrics dict"

        assert 0.0 <= metrics["anomaly_rate"] <= 1.0
        assert 0.0 <= metrics["mean_combined_score"] <= 1.0
        assert 0.0 <= metrics["model_confidence"] <= 1.0
        assert isinstance(metrics["score_distribution"], dict)
        assert isinstance(metrics["top_anomalous_templates"], list)

    def test_no_classification_metrics_without_ground_truth(self, tmp_path):
        anomaly_df  = _make_anomaly_df()
        features_df = _make_eval_features_df()

        with patch("ml.evaluator.SYNTHETIC_DIR", tmp_path / "no_synthetic"), \
             self._patch_summary(tmp_path):
            metrics = run_evaluation(anomaly_df, features_df)

        for key in ("precision", "recall", "f1", "false_negative_rate",
                    "noise_suppression_ratio"):
            assert key not in metrics, (
                f"'{key}' should not appear without ground truth"
            )

    def test_with_ground_truth_classification_metrics_present(self, tmp_path):
        """All classification metrics present when ground truth is available."""
        n = 100
        anomaly_df  = _make_anomaly_df(n)
        features_df = _make_eval_features_df(n)

        syn_dir = tmp_path / "synthetic"
        syn_dir.mkdir()
        gt = pd.DataFrame({
            "sequence_number": np.arange(1, n + 1),
            "true_label": ["anomaly" if i % 3 == 0 else "normal" for i in range(n)],
        })
        gt.to_parquet(syn_dir / "ground_truth.parquet", index=False)

        with patch("ml.evaluator.SYNTHETIC_DIR", syn_dir), \
             self._patch_summary(tmp_path):
            metrics = run_evaluation(anomaly_df, features_df)

        for key in ("precision", "recall", "f1", "false_negative_rate",
                    "noise_suppression_ratio"):
            assert key in metrics, f"Expected '{key}' in metrics when GT available"
            assert 0.0 <= metrics[key] <= 1.0, (
                f"'{key}' = {metrics[key]} is outside [0, 1]"
            )

    def test_false_negative_rate_correct(self, tmp_path):
        """FNR = true anomalies scored below threshold / total true anomalies."""
        n = 100
        rng = np.random.default_rng(99)
        combined = rng.uniform(0.0, 1.0, n)
        anomaly_df = pd.DataFrame({
            "sequence_number":  np.arange(1, n + 1),
            "isolation_score":  rng.uniform(0, 1, n),
            "zscore_norm":      rng.uniform(0, 1, n),
            "combined_score":   combined,
            "is_anomaly":       combined > ANOMALY_SCORE_THRESHOLD,
            "model_confidence": 1.0,
        })
        features_df = _make_eval_features_df(n)

        # All rows labelled anomaly — FNR = fraction whose combined_score <= threshold
        syn_dir = tmp_path / "synthetic2"
        syn_dir.mkdir()
        gt = pd.DataFrame({
            "sequence_number": np.arange(1, n + 1),
            "true_label":      "anomaly",
        })
        gt.to_parquet(syn_dir / "ground_truth.parquet", index=False)

        with patch("ml.evaluator.SYNTHETIC_DIR", syn_dir), \
             self._patch_summary(tmp_path):
            metrics = run_evaluation(anomaly_df, features_df)

        expected_fnr = float((~anomaly_df["is_anomaly"]).mean())
        assert metrics["false_negative_rate"] == pytest.approx(expected_fnr, abs=1e-9)

    def test_noise_suppression_ratio_correct(self, tmp_path):
        """NSR = true normals correctly below threshold / total true normals."""
        n = 100
        rng = np.random.default_rng(77)
        combined = rng.uniform(0.0, 1.0, n)
        anomaly_df = pd.DataFrame({
            "sequence_number":  np.arange(1, n + 1),
            "isolation_score":  rng.uniform(0, 1, n),
            "zscore_norm":      rng.uniform(0, 1, n),
            "combined_score":   combined,
            "is_anomaly":       combined > ANOMALY_SCORE_THRESHOLD,
            "model_confidence": 1.0,
        })
        features_df = _make_eval_features_df(n)

        # All rows labelled normal — NSR = fraction whose is_anomaly == False
        syn_dir = tmp_path / "synthetic3"
        syn_dir.mkdir()
        gt = pd.DataFrame({
            "sequence_number": np.arange(1, n + 1),
            "true_label":      "normal",
        })
        gt.to_parquet(syn_dir / "ground_truth.parquet", index=False)

        with patch("ml.evaluator.SYNTHETIC_DIR", syn_dir), \
             self._patch_summary(tmp_path):
            metrics = run_evaluation(anomaly_df, features_df)

        expected_nsr = float((~anomaly_df["is_anomaly"]).mean())
        assert metrics["noise_suppression_ratio"] == pytest.approx(expected_nsr, abs=1e-9)

    def test_summary_file_written_to_results_dir(self, tmp_path):
        """anomaly_summary.txt is created under evaluation/results/."""
        anomaly_df  = _make_anomaly_df()
        features_df = _make_eval_features_df()
        summary_path = tmp_path / "results" / "anomaly_summary.txt"

        with patch("ml.evaluator.SYNTHETIC_DIR", tmp_path / "no_synthetic"), \
             patch("ml.evaluator.SUMMARY_PATH", summary_path):
            run_evaluation(anomaly_df, features_df)

        assert summary_path.exists(), "anomaly_summary.txt was not created"
        content = summary_path.read_text()
        assert "anomaly_rate" in content
        assert "model_confidence" in content
