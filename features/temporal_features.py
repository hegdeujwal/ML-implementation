"""
Temporal feature extraction for sessionized logs.

All three functions are per-session — call them via groupby("session_id").apply()
from feature_pipeline.py.  Each sorts by timestamp internally and returns a Series
aligned to the original session_df.index (original index labels are preserved through
the sort, so no explicit re-indexing is required).
"""

from __future__ import annotations

import pandas as pd

from common.config import IAR_EMA_ALPHA


def time_delta_prev(session_df: pd.DataFrame) -> pd.Series:
    """Seconds since the previous log in the session, sorted by timestamp.

    First log in the session → 0.0.  Handles out-of-order input by sorting
    internally without modifying the caller's DataFrame.
    """
    sorted_df = session_df.sort_values("timestamp")
    return (
        sorted_df["timestamp"]
        .diff()
        .dt.total_seconds()
        .fillna(0.0)
        .astype(float)
    )


def time_delta_session_start(session_df: pd.DataFrame) -> pd.Series:
    """Seconds from the first log in the session to each log.

    First log → 0.0.  Handles out-of-order input by sorting internally.
    """
    sorted_df = session_df.sort_values("timestamp")
    first_ts = sorted_df["timestamp"].iloc[0]
    return (sorted_df["timestamp"] - first_ts).dt.total_seconds().astype(float)


def inter_arrival_rate(session_df: pd.DataFrame) -> pd.Series:
    """EMA of inter-arrival times within the session (alpha from config IAR_EMA_ALPHA).

    First log → 0.0 (no previous arrival).  Single-log sessions → 0.0 for all rows.
    Handles out-of-order input by sorting internally.
    """
    if len(session_df) < 2:
        return pd.Series(0.0, index=session_df.index, dtype=float)

    sorted_df = session_df.sort_values("timestamp")
    diffs = (
        sorted_df["timestamp"]
        .diff()
        .dt.total_seconds()
        .fillna(0.0)
    )
    return (
        diffs
        .ewm(alpha=IAR_EMA_ALPHA, adjust=False)
        .mean()
        .astype(float)
    )
