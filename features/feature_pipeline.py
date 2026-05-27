"""
Main feature engineering pipeline.

Reads sessionized_logs.parquet, applies all feature modules, validates the output
schema, and saves features_df.parquet for the ML anomaly detection stage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.config import (
    FEATURE_COLUMNS,
    FEATURES_OUTPUT_PATH,
    ZSCORE_BASELINE_N_SESSIONS,
)
from common.logger import get_logger
from common.utils import load_parquet, save_parquet, validate_schema

from features.statistical_features import log_frequency_score, burstiness_score, zscore_base
from features.temporal_features import time_delta_prev, time_delta_session_start, inter_arrival_rate
from features.counter_proximity import compute_counter_proximity

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
        df["zscore_base"] = zscore_base(df, ZSCORE_BASELINE_N_SESSIONS)

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

        # 10. Print summary
        print(f"\nShape: {out.shape}")
        print("\nNull counts:")
        print(out.isnull().sum().to_string())
        print("\nSample (5 rows):")
        print(out.head().to_string())

        return out

    except Exception:
        logger.exception("Feature pipeline failed")
        raise


if __name__ == "__main__":
    run_pipeline("data/processed/sessionized_logs.parquet")
