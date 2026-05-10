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
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from common.config import (
    CONTAMINATION,
    WEIGHT_ISOLATION,
    WEIGHT_ZSCORE,
    MIN_TRAIN_SAMPLES,
    TRAINING_WINDOW_SESSIONS,
    RETRAIN_EVERY_K_LOGS,
)
from common.logger import get_logger
from ml.anomaly_detector import FEATURE_COLS, compute_zscore_base

logger = get_logger(__name__)

MODEL_STORE_DIR = Path("ml/model_store")
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

    def maybe_retrain(self, features_df: pd.DataFrame) -> Optional[IsolationForest]:
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
            f"(trigger at K={RETRAIN_EVERY_K_LOGS})."
        )

        if logs_since_last >= RETRAIN_EVERY_K_LOGS:
            logger.info("Periodic retrain trigger fired.")
            model = self.retrain(features_df)
            self._logs_seen_at_last_retrain = current_count
            self._save_retrain_state(current_count)
            return model

        logger.info("No retrain needed yet.")
        return None

    def retrain(self, features_df: pd.DataFrame) -> Optional[IsolationForest]:
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
        # Select features — same columns as anomaly_detector uses
        # -----------------------------------------------------------------------
        available_cols = [c for c in FEATURE_COLS if c in windowed_df.columns]
        if not available_cols:
            logger.error("No feature columns available in windowed data. Aborting retrain.")
            return None

        X = windowed_df[available_cols].fillna(0).values

        # -----------------------------------------------------------------------
        # Train
        # -----------------------------------------------------------------------
        logger.info(
            f"Training IsolationForest on {n_samples} samples, "
            f"{len(available_cols)} features, contamination={CONTAMINATION}."
        )
        start_time = time.time()
        model = IsolationForest(
            contamination=CONTAMINATION,
            random_state=42,
            n_estimators=100,
            n_jobs=-1,
        )
        model.fit(X)
        elapsed = time.time() - start_time
        logger.info(f"Training complete in {elapsed:.2f}s.")

        # -----------------------------------------------------------------------
        # Persist model + sidecar
        # -----------------------------------------------------------------------
        self._save_model(model, n_samples, available_cols)
        return model

    def load_latest_model(self) -> Optional[IsolationForest]:
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

    def load_model_by_timestamp(self, timestamp: str) -> IsolationForest:
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
                f"{TRAINING_WINDOW_SESSIONS} sessions."
            )
            return features_df

        all_sessions = features_df["session_id"].unique()
        # Take the last N session IDs (assumes sessions are ordered by creation time)
        window_sessions = all_sessions[-TRAINING_WINDOW_SESSIONS:]
        windowed = features_df[features_df["session_id"].isin(window_sessions)].copy()

        logger.info(
            f"Sliding window: {len(window_sessions)} sessions "
            f"(of {len(all_sessions)} total) → {len(windowed)} log rows."
        )
        return windowed

    def _save_model(
        self,
        model: IsolationForest,
        n_samples: int,
        feature_columns: list[str],
    ) -> None:
        """Persist model .pkl and JSON sidecar with training metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pkl_path = MODEL_STORE_DIR / f"isolation_forest_v{timestamp}.pkl"
        meta_path = MODEL_STORE_DIR / f"isolation_forest_v{timestamp}.json"

        # Save model
        joblib.dump(model, pkl_path)
        logger.info(f"Model saved to {pkl_path}.")

        # Save sidecar
        metadata = {
            "timestamp": timestamp,
            "n_samples": n_samples,
            "contamination": CONTAMINATION,
            "feature_columns": feature_columns,
            "w1": WEIGHT_ISOLATION,
            "w2": WEIGHT_ZSCORE,
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