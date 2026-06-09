"""
Main feature engineering pipeline.

Reads sessionized_logs.parquet, applies all feature modules, validates the output
schema, and saves features_df.parquet for the ML anomaly detection stage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pathlib import Path

from common.config import (
    FEATURE_COLUMNS,
    FEATURE_ROLLING_MAX_SESSIONS,
    FEATURE_ROLLING_STORE_PATH,
    FEATURES_OUTPUT_PATH,
    METRICS_DF_PATH,
    ZSCORE_BASELINE_N_SESSIONS,
    ZSCORE_BASELINE_STORE_PATH,
)
from common.logger import get_logger
from common.utils import load_parquet, save_parquet, validate_schema

from features.statistical_features import (
    burstiness_score,
    log_frequency_score,
    update_feature_rolling_store,
    zscore_base_persistent,
)
from features.temporal_features import time_delta_prev, time_delta_session_start, inter_arrival_rate
from features.counter_proximity import compute_counter_proximity
from features.metric_features import compute_metric_features, METRIC_FEATURE_COLUMNS

logger = get_logger(__name__)

_REQUIRED_INPUT_COLUMNS = [
    "sequence_number",
    "timestamp",
    "session_id",
    "template_id",
    "frequency",
    "host",
    "log_level",
    "event_weight",
]

_NO_NULL_COLUMNS = ["sequence_number", "session_id", "event_weight"]


def run_pipeline(input_path: str) -> pd.DataFrame:
    """Run the full feature engineering pipeline.

    Args:
        input_path: Path to sessionized_logs.parquet produced by parsing stage.

    Returns:
        DataFrame containing exactly FEATURE_COLUMNS, saved to FEATURES_OUTPUT_PATH.

    Raises:
        FileNotFoundError: If input_path does not exist.
        ValueError: If input schema is invalid, or output contains nulls/inf/NaN.
    """
    try:
        # 1. Load
        df = load_parquet(input_path)
        logger.info(f"Loaded {len(df):,} rows from {input_path}")

        # 2. Validate input schema
        validate_schema(df, _REQUIRED_INPUT_COLUMNS)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # 3. Statistical features
        df["frequency_score"] = (
            df.groupby("session_id", group_keys=False)
            .apply(log_frequency_score)
            .astype(float)
        )
        df["burstiness_score"] = (
            df.groupby("session_id", group_keys=False)
            .apply(burstiness_score)
            .astype(float)
        )
        df["zscore_base"] = zscore_base_persistent(
            df, ZSCORE_BASELINE_STORE_PATH
        )

        # 4. Temporal features
        df["time_delta_prev"] = (
            df.groupby("session_id", group_keys=False)
            .apply(time_delta_prev)
            .astype(float)
        )
        df["time_delta_session_start"] = (
            df.groupby("session_id", group_keys=False)
            .apply(time_delta_session_start)
            .astype(float)
        )
        df["inter_arrival_rate"] = (
            df.groupby("session_id", group_keys=False)
            .apply(inter_arrival_rate)
            .astype(float)
        )

        # 5. Counter proximity (host-scoped, not session-scoped)
        df["counter_proximity"] = compute_counter_proximity(df)

        # 5b. Section-4 numeric-metric features (joined from metrics_df by time
        # proximity). Absent metrics → neutral 0 with present=0, so the legacy
        # syslog path (no metrics_df) and metric-less scenarios stay valid.
        metrics_df = None
        if Path(METRICS_DF_PATH).exists():
            metrics_df = load_parquet(METRICS_DF_PATH)
        metric_feats = compute_metric_features(df, metrics_df)
        for col in METRIC_FEATURE_COLUMNS:
            df[col] = metric_feats[col].astype(float).values

        # 6. event_weight already present in parquet from sessionizer — not recomputed

        # 7. Select output columns only
        validate_schema(df, FEATURE_COLUMNS)
        out = df[FEATURE_COLUMNS].copy()

        # 8. Validate output — no nulls in required columns, no inf/NaN in floats
        for col in _NO_NULL_COLUMNS:
            if out[col].isnull().any():
                raise ValueError(
                    f"Null values found in required output column '{col}'. "
                    "Check upstream parsing stage."
                )
        float_cols = out.select_dtypes(include=[float]).columns.tolist()
        for col in float_cols:
            if not np.isfinite(out[col]).all():
                raise ValueError(
                    f"Infinite or NaN values in float column '{col}'. "
                    "Check feature computation for divide-by-zero or overflow."
                )

        # 9. Save
        save_parquet(out, FEATURES_OUTPUT_PATH)
        logger.info(
            f"Saved features_df → {FEATURES_OUTPUT_PATH} ({len(out):,} rows, "
            f"{len(FEATURE_COLUMNS)} columns)"
        )

        # 10. Log summary
        logger.info(f"Shape: {out.shape}")
        null_summary = out.isnull().sum()
        if null_summary.any():
            logger.warning(f"Null counts:\n{null_summary.to_string()}")
        else:
            logger.info("No nulls in output.")
        logger.info(f"Sample (first row):\n{out.head(1).to_string()}")

        # 10. Update rolling feature store for cross-run IF retraining.
        # Pass the full df (has session_id + timestamp) filtered to FEATURE_COLUMNS
        # plus session_id so AnomalyTrainer can apply the sliding-window logic.
        store_cols = list(dict.fromkeys(["session_id", "timestamp"] + FEATURE_COLUMNS))
        store_cols = [c for c in store_cols if c in df.columns]
        update_feature_rolling_store(
            df[store_cols],
            FEATURE_ROLLING_STORE_PATH,
            FEATURE_ROLLING_MAX_SESSIONS,
        )
        logger.info(
            "Rolling feature store updated: last %d sessions retained at %s",
            FEATURE_ROLLING_MAX_SESSIONS,
            FEATURE_ROLLING_STORE_PATH,
        )

        return out

    except Exception:
        logger.exception("Feature pipeline failed")
        raise


if __name__ == "__main__":
    run_pipeline("data/processed/sessionized_logs.parquet")
