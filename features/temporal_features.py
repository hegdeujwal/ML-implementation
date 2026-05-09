"""
Temporal feature extraction for sessionized logs.

Calculates:
- time since previous log
- time since session start
- inter-arrival rate trends

Used to capture timing-based anomaly patterns.
"""




import numpy as np
import pandas as pd

from common.config import INTER_ARRIVAL_EMA_SPAN


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.dropna(subset=["timestamp"])

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values(["session_id", "timestamp"])

    df["time_delta_prev"] = (
        df.groupby("session_id")["timestamp"]
        .diff()
        .dt.total_seconds()
        .fillna(0.0)
    )

    session_start = (
        df.groupby("session_id")["timestamp"]
        .transform("min")
    )

    df["time_delta_session_start"] = (
        df["timestamp"] - session_start
    ).dt.total_seconds()

    df["inter_arrival_rate"] = (
        df.groupby("session_id")["time_delta_prev"]
        .transform(
            lambda x: x.ewm(
                span=INTER_ARRIVAL_EMA_SPAN,
                adjust=False
            ).mean()
        )
    )

    return df