"""
Main feature engineering pipeline.

Reads sessionized logs, applies all feature extraction modules,
validates the final schema, and saves features_df.parquet.

This output is used directly by the ML anomaly detection module.
"""


import pandas as pd

from common.config import (
    FEATURE_COLUMNS,
)

from common.logger import get_logger

from features.statistical_features import (
    log_frequency_score,
    burstiness_score,
    zscore_base,
)

from features.temporal_features import (
    add_temporal_features,
)

from features.severity_features import (
    add_severity_weight,
)

from features.counter_proximity import (
    add_counter_proximity,
)

logger = get_logger(__name__)


def validate_schema(df: pd.DataFrame):

    missing = [
        col for col in FEATURE_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def run_pipeline(input_path: str):

    try:

        df = pd.read_parquet(input_path)

        df = df.dropna(
            subset=["log_id", "session_id"]
        )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

        df = df.dropna(subset=["timestamp"])

        df = log_frequency_score(df)

        df = burstiness_score(df)

        df = zscore_base(df)

        df = add_temporal_features(df)

        df = add_severity_weight(df)

        df = add_counter_proximity(df)

        validate_schema(df)

        features_df = df[FEATURE_COLUMNS]

        print("\nShape:")
        print(features_df.shape)

        print("\nNull counts:")
        print(features_df.isnull().sum())

        print("\nSample rows:")
        print(features_df.head())

        features_df.to_parquet(
            "data/processed/features_df.parquet",
            index=False,
        )

        logger.info(
            "Feature pipeline completed successfully"
        )

        return features_df

    except Exception as e:

        logger.error(
            f"Pipeline failed: {str(e)}"
        )

        raise


if __name__ == "__main__":

    run_pipeline(
        "parsing/processed/sessionized_logs.parquet"
    )