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
import pandas as pd
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
        self._logs_seen_at_last_retrain = self._load_retrain_state()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def maybe_retrain(self, features_df: pd.DataFrame) -> Optional[Pipeline]:
        """Retrain if the periodic K-log trigger fires.

        Call this every time new logs arrive. It tracks cumulative log counts
        and only triggers a full retrain when the delta exceeds K.

        Args:
            features_df: Current full features DataFrame (all logs seen so far).

        Returns:
            The newly trained IsolationForest if retrain happened, else None.
        """
        current_count = len(features_df)
        logs_since_last = current_count - self._logs_seen_at_last_retrain

        logger.info(
            f"maybe_retrain: {current_count} total logs, "
            f"{logs_since_last} new since last retrain "
            f"(trigger at K={RETRAINING_TRIGGER_EVERY_K})."
        )

        if logs_since_last >= RETRAINING_TRIGGER_EVERY_K:
            logger.info("Periodic retrain trigger fired.")
            model = self.retrain(features_df)
            self._logs_seen_at_last_retrain = current_count
            self._save_retrain_state(current_count)
            return model

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

        Returns:
            The deserialized IsolationForest, or None if no model exists yet.
        """
        pkl_files = sorted(MODEL_STORE_DIR.glob("isolation_forest_v*.pkl"))
        if not pkl_files:
            logger.warning("No saved model found in model_store. Run retrain() first.")
            return None

        latest = pkl_files[-1]   # sorted by name = sorted by timestamp
        logger.info(f"Loading model from {latest}.")
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
            "contamination": IF_CONTAMINATION,
            "feature_columns": IF_FEATURE_COLUMNS,
            "isolation_weight": IF_ISOLATION_WEIGHT,
            "zscore_weight": IF_ZSCORE_WEIGHT,
            "cold_start_threshold": COLD_START_FULL_CONFIDENCE_THRESHOLD,
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata sidecar saved to {meta_path}.")

    def _save_retrain_state(self, log_count: int) -> None:
        """Persist the log count at last retrain so it survives process restarts."""
        with open(RETRAIN_STATE_FILE, "w") as f:
            json.dump({"logs_seen_at_last_retrain": log_count}, f)

    def _load_retrain_state(self) -> int:
        """Load persisted retrain state, or return 0 if no state exists yet."""
        if RETRAIN_STATE_FILE.exists():
            with open(RETRAIN_STATE_FILE) as f:
                return json.load(f).get("logs_seen_at_last_retrain", 0)
        return 0