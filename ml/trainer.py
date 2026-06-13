"""
trainer.py
==========
Phase 2 — ML Training Strategy
Assignee: Shreeraksha M

Manages IsolationForest lifecycle:
  - Sliding window retraining (last N sessions)
  - Periodic retraining trigger (every K new logs)
  - Model + JSON sidecar persistence to ml/model_store/
  - Model loading for inference

Usage:
    trainer = AnomalyTrainer()
    trainer.maybe_retrain(features_df)       # respects K-log trigger
    model = trainer.load_latest_model()      # for inference elsewhere
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from common.config import (
    IF_CONTAMINATION,
    IF_FEATURE_COLUMNS,
    IF_ISOLATION_WEIGHT,
    IF_N_ESTIMATORS,
    IF_RANDOM_STATE,
    IF_ZSCORE_WEIGHT,
    COLD_START_FULL_CONFIDENCE_THRESHOLD,
    MIN_TRAIN_SAMPLES,
    MODEL_STORE_PATH,
    RETRAINING_SESSION_WINDOW,
    RETRAINING_TRIGGER_EVERY_K,
)
from common.logger import get_logger
from ml.anomaly_detector import _train_model

logger = get_logger(__name__)

MODEL_STORE_DIR = Path(MODEL_STORE_PATH)
# State file tracks how many logs were seen at the last retrain
RETRAIN_STATE_FILE = MODEL_STORE_DIR / "retrain_state.json"


class AnomalyTrainer:
    """Manages training, persistence, and loading of the IsolationForest model.

    Design rationale:
        - Sliding window (last N sessions) prevents the model from drifting on
          stale historical patterns while keeping enough data to generalise.
        - Periodic trigger (every K logs) avoids retraining on every single log
          while keeping the model reasonably fresh during live ingestion.
        - JSON sidecar alongside each .pkl makes it easy to audit which config
          produced which model — critical for debugging anomaly rate changes.
    """

    def __init__(self) -> None:
        MODEL_STORE_DIR.mkdir(parents=True, exist_ok=True)
        self._unprocessed_logs_count = self._load_retrain_state()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def maybe_retrain(
        self, features_df: pd.DataFrame, new_logs_count: int = 0
    ) -> Optional[Pipeline]:
        """Retrain if the periodic K-log trigger fires.

        Call this every time new logs arrive. It tracks cumulative log counts
        and only triggers a full retrain when the delta exceeds K.

        Args:
            features_df: Rolling features DataFrame (last N sessions).
            new_logs_count: Number of new logs ingested in this pipeline run.

        Returns:
            The newly trained IsolationForest if retrain happened, else None.
        """
        self._unprocessed_logs_count += new_logs_count

        logger.info(
            f"maybe_retrain: {len(features_df)} rolling logs, "
            f"{self._unprocessed_logs_count} new since last retrain "
            f"(trigger at K={RETRAINING_TRIGGER_EVERY_K})."
        )

        if self._unprocessed_logs_count >= RETRAINING_TRIGGER_EVERY_K:
            logger.info("Periodic retrain trigger fired.")
            model = self.retrain(features_df)
            self._unprocessed_logs_count = 0
            self._save_retrain_state(0)
            return model

        self._save_retrain_state(self._unprocessed_logs_count)
        logger.info("No retrain needed yet.")
        return None

    def retrain(self, features_df: pd.DataFrame) -> Optional[Pipeline]:
        """Train on the sliding window (last N sessions) and persist the model.

        Args:
            features_df: Full features DataFrame with a 'session_id' column.

        Returns:
            Trained IsolationForest, or None if cold-start fallback triggered.
        """
        windowed_df = self._apply_sliding_window(features_df)
        n_samples = len(windowed_df)

        # -----------------------------------------------------------------------
        # Cold-start guard: can't train a reliable model on too few samples
        # -----------------------------------------------------------------------
        if n_samples < MIN_TRAIN_SAMPLES:
            logger.warning(
                f"[COLD-START] Windowed dataset has only {n_samples} samples "
                f"(min_train_samples={MIN_TRAIN_SAMPLES}). Skipping IsolationForest training. "
                "The anomaly_detector will fall back to z-score only."
            )
            return None

        # -----------------------------------------------------------------------
        # Train via the shared factory so pipeline construction is not duplicated
        # -----------------------------------------------------------------------
        logger.info(
            f"Training on {n_samples} samples, "
            f"{len(IF_FEATURE_COLUMNS)} features, contamination={IF_CONTAMINATION}."
        )
        pipeline = _train_model(windowed_df)   # sets pipeline.n_samples_seen_

        # -----------------------------------------------------------------------
        # Persist model + sidecar
        # -----------------------------------------------------------------------
        self._save_model(pipeline, n_samples)
        return pipeline

    def load_latest_model(self) -> Optional[Pipeline]:
        """Load the most recently saved model from model_store.

        Validates the JSON sidecar before loading to guard against sklearn
        version mismatches and feature column drift that would corrupt scores
        silently.

        Returns:
            The deserialized IsolationForest, or None if no model exists or
            sidecar validation fails.
        """
        pkl_files = sorted(MODEL_STORE_DIR.glob("isolation_forest_v*.pkl"))
        if not pkl_files:
            logger.warning("No saved model found in model_store. Run retrain() first.")
            return None

        latest = pkl_files[-1]   # sorted by name = sorted by timestamp
        sidecar = latest.with_suffix(".json")

        if not sidecar.exists():
            logger.warning(
                "No sidecar found for %s — skipping load to avoid silent corruption.",
                latest.name,
            )
            return None

        if not self._validate_sidecar(sidecar):
            return None

        logger.info("Loading model from %s.", latest)
        model = joblib.load(latest)
        logger.info("Model loaded successfully.")
        return model

    def load_model_by_timestamp(self, timestamp: str) -> Pipeline:
        """Load a specific model version by its timestamp string.

        Args:
            timestamp: e.g. '20260415_143022'

        Returns:
            Deserialized IsolationForest.
        """
        path = MODEL_STORE_DIR / f"isolation_forest_v{timestamp}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        return joblib.load(path)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _apply_sliding_window(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Return only the last N sessions from features_df.

        If 'session_id' is not present, fall back to using the full DataFrame
        (with a warning). This makes P2 robust to P1 delays.

        Args:
            features_df: Full features DataFrame.

        Returns:
            Subset containing only logs from the last N sessions.
        """
        if "session_id" not in features_df.columns:
            logger.warning(
                "session_id column not found — P1 may not have shipped it yet. "
                f"Training on all {len(features_df)} rows instead of last "
                f"{RETRAINING_SESSION_WINDOW} sessions."
            )
            return features_df

        # Sort sessions by their earliest timestamp so "last N" is chronological,
        # not insertion-order (unique() does not guarantee time ordering).
        session_order = (
            features_df.groupby("session_id")["timestamp"]
            .min()
            .sort_values()
            .index
        )
        ordered_sessions = session_order.tolist()
        all_sessions = features_df["session_id"].unique()
        window_sessions = ordered_sessions[-RETRAINING_SESSION_WINDOW:]
        windowed = features_df[features_df["session_id"].isin(window_sessions)].copy()

        logger.info(
            f"Sliding window: {len(window_sessions)} sessions "
            f"(of {len(all_sessions)} total) → {len(windowed)} log rows."
        )
        return windowed

    def _save_model(
        self,
        pipeline: Pipeline,
        n_samples: int,
    ) -> None:
        """Persist pipeline .pkl and JSON sidecar with training metadata.

        n_samples_seen_ is already set on the pipeline object by _train_model(),
        but is also written to the sidecar so detect() can recover it on load
        without requiring the attribute to survive pkl round-trips.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pkl_path = MODEL_STORE_DIR / f"isolation_forest_v{timestamp}.pkl"
        meta_path = MODEL_STORE_DIR / f"isolation_forest_v{timestamp}.json"

        # n_samples_seen_ is set by _train_model(); ensure it survives the dump
        pipeline.n_samples_seen_ = n_samples
        joblib.dump(pipeline, pkl_path)
        logger.info(f"Model saved to {pkl_path}.")

        metadata = {
            "timestamp": timestamp,
            "n_samples": n_samples,
            # Training-score calibration (informational copy — the live values
            # ride on pipeline.calibration_ through the pickle).
            "calibration": getattr(pipeline, "calibration_", None),
            "contamination": IF_CONTAMINATION,
            "feature_columns": IF_FEATURE_COLUMNS,
            "isolation_weight": IF_ISOLATION_WEIGHT,
            "zscore_weight": IF_ZSCORE_WEIGHT,
            "cold_start_threshold": COLD_START_FULL_CONFIDENCE_THRESHOLD,
            # Version info for load-time validation
            "sklearn_version": sklearn.__version__,
            "numpy_version": numpy.__version__,
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata sidecar saved to {meta_path}.")

    def _save_retrain_state(self, count: int) -> None:
        """Persist the unprocessed log count so it survives process restarts."""
        with open(RETRAIN_STATE_FILE, "w") as f:
            json.dump({"unprocessed_logs_count": count}, f)

    def _load_retrain_state(self) -> int:
        """Load persisted retrain state, or return 0 if no state exists yet."""
        if RETRAIN_STATE_FILE.exists():
            with open(RETRAIN_STATE_FILE) as f:
                data = json.load(f)
                return data.get("unprocessed_logs_count") or data.get("logs_seen_at_last_retrain", 0)
        return 0

    def _validate_sidecar(self, sidecar_path: Path) -> bool:
        """Validate sidecar metadata before loading a saved model.

        Checks:
        - sklearn version matches current environment (warns on minor mismatch,
          refuses on major mismatch)
        - feature_columns in sidecar exactly matches IF_FEATURE_COLUMNS from config
          (reordering or addition/removal corrupts StandardScaler silently)

        Returns:
            True if the model is safe to load, False otherwise.
        """
        try:
            with open(sidecar_path) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read sidecar %s: %s", sidecar_path, exc)
            return False

        saved_sklearn = meta.get("sklearn_version", "unknown")
        saved_features = meta.get("feature_columns", [])

        # Feature column check — must be identical (order matters for StandardScaler)
        if saved_features != IF_FEATURE_COLUMNS:
            logger.warning(
                "Model sidecar feature_columns mismatch. "
                "Saved: %s  Current: %s. Refusing to load — retraining from scratch.",
                saved_features,
                IF_FEATURE_COLUMNS,
            )
            return False

        # sklearn version check
        saved_major = saved_sklearn.split(".")[0] if saved_sklearn != "unknown" else None
        current_major = sklearn.__version__.split(".")[0]
        if saved_major and saved_major != current_major:
            logger.warning(
                "sklearn major version mismatch: saved=%s current=%s. "
                "Refusing to load — retraining from scratch.",
                saved_sklearn,
                sklearn.__version__,
            )
            return False

        if saved_sklearn != sklearn.__version__:
            logger.warning(
                "sklearn minor version differs: saved=%s current=%s. "
                "Scores may differ slightly — proceeding with load.",
                saved_sklearn,
                sklearn.__version__,
            )

        return True