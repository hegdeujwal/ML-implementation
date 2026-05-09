"""
Statistical feature calculations for log behavior analysis.

Includes:
- frequency-based features
- burstiness calculations
- rolling z-score baseline analysis

These features help identify unusual log activity patterns.
"""

import numpy as np
import pandas as pd

from common.config import (
    ZSCORE_ROLLING_WINDOW,
    BURSTINESS_MIN_EVENTS
)


def log_frequency_score(df: pd.DataFrame) -> pd.DataFrame:

    counts = (
        df.groupby(["session_id", "template_id"])
        ["template_id"]
        .transform("count")
    )

    max_count = counts.max()

    if max_count == 0:
        df["frequency_score"] = 0.0
    else:
        df["frequency_score"] = counts / max_count

    return df


def burstiness_score(df: pd.DataFrame) -> pd.DataFrame:

    df = df.sort_values(["session_id", "timestamp"])

    inter_arrival = (
        df.groupby("session_id")["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    def fano_factor(series):

        series = series.dropna()

        if len(series) < BURSTINESS_MIN_EVENTS:
            return 0.0

        mean = series.mean()

        if mean == 0:
            return 0.0

        variance = series.var()

        return variance / mean

    scores = (
        inter_arrival.groupby(df["session_id"])
        .transform(fano_factor)
    )

    df["burstiness_score"] = scores.fillna(0.0)

    return df


def zscore_base(df: pd.DataFrame) -> pd.DataFrame:

    df = df.sort_values("timestamp")

    template_counts = (
        df.groupby([
            pd.Grouper(key="timestamp", freq="1min"),
            "template_id"
        ])
        .size()
        .reset_index(name="count")
    )

    template_counts["rolling_mean"] = (
        template_counts
        .groupby("template_id")["count"]
        .transform(
            lambda x: x.rolling(
                ZSCORE_ROLLING_WINDOW,
                min_periods=1
            ).mean()
        )
    )

    template_counts["rolling_std"] = (
        template_counts
        .groupby("template_id")["count"]
        .transform(
            lambda x: x.rolling(
                ZSCORE_ROLLING_WINDOW,
                min_periods=1
            ).std()
        )
        .fillna(1.0)
    )

    template_counts["zscore_base"] = (
        (template_counts["count"] - template_counts["rolling_mean"])
        / template_counts["rolling_std"]
    )

    zmap = (
        template_counts
        .groupby("template_id")["zscore_base"]
        .mean()
    )

    df["zscore_base"] = (
        df["template_id"].map(zmap)
    ).fillna(0.0)

    return df