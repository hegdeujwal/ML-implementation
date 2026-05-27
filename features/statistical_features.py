"""
Statistical feature calculations for log behaviour analysis.

log_frequency_score and burstiness_score are per-session functions — call them
via groupby("session_id").apply() from feature_pipeline.py.
zscore_base operates on the full DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_frequency_score(session_df: pd.DataFrame) -> pd.Series:
    """Normalise per-session template frequency to [0.0, 1.0].

    Uses the frequency column already present from sessionizer — does not
    recompute groupby counts.
    """
    max_freq = session_df["frequency"].max()
    if max_freq == 0:
        return pd.Series(0.0, index=session_df.index, dtype=float)
    return (session_df["frequency"] / max_freq).astype(float)


def burstiness_score(session_df: pd.DataFrame) -> pd.Series:
    """Fano factor (variance / mean) of inter-arrival times, clipped to [0.0, 10.0].

    The same scalar value is broadcast to every row in the session.
    """
    if len(session_df) < 2:
        return pd.Series(0.0, index=session_df.index, dtype=float)

    sorted_ts = session_df["timestamp"].sort_values()
    inter_arrivals = sorted_ts.diff().dt.total_seconds().dropna()

    mean = inter_arrivals.mean()
    if mean == 0:
        return pd.Series(0.0, index=session_df.index, dtype=float)

    fano = float(np.clip(inter_arrivals.var() / mean, 0.0, 10.0))
    return pd.Series(fano, index=session_df.index, dtype=float)


def zscore_base(df: pd.DataFrame, n_sessions: int) -> pd.Series:
    """Rolling z-score of template frequency, computed per (host, template_id).

    For each session, the baseline is the mean and std of that template's frequency
    across the previous n_sessions sessions on the same host only.  Returns 0.0 when
    fewer than 2 sessions of history exist or baseline std is zero.  Result clipped
    to [-5.0, 5.0] — extreme values are noise not signal.

    Args:
        df:         Full sessionized DataFrame with host, session_id, template_id,
                    frequency, and timestamp columns.
        n_sessions: Number of prior sessions to use as the rolling baseline.
    """
    # Session start times — used to order sessions chronologically per host
    session_starts = (
        df.groupby(["host", "session_id"])["timestamp"]
        .min()
        .reset_index()
        .rename(columns={"timestamp": "session_start"})
    )

    # One row per (host, session_id, template_id) with its session-level frequency
    stf = (
        df.groupby(["host", "session_id", "template_id"])["frequency"]
        .first()
        .reset_index()
    )
    stf = stf.merge(session_starts, on=["host", "session_id"])
    stf = stf.sort_values(["host", "template_id", "session_start"]).reset_index(drop=True)

    # Compute rolling z-score for each (host, template_id) pair independently
    zscore_lookup: dict[tuple, float] = {}
    for (host, template_id), grp in stf.groupby(["host", "template_id"], sort=False):
        grp = grp.reset_index(drop=True)
        freqs = grp["frequency"].values.astype(float)
        for i in range(len(grp)):
            key = (host, grp.loc[i, "session_id"], template_id)
            if i < 2:
                # Fewer than 2 sessions of history — no meaningful baseline
                zscore_lookup[key] = 0.0
                continue
            history = freqs[max(0, i - n_sessions):i]
            mean = float(history.mean())
            std = float(np.std(history, ddof=1)) if len(history) > 1 else 0.0
            if std == 0.0:
                zscore_lookup[key] = 0.0
            else:
                z = (freqs[i] - mean) / std
                zscore_lookup[key] = float(np.clip(z, -5.0, 5.0))

    # Map z-scores back to every row in the original DataFrame
    keys = list(zip(df["host"], df["session_id"], df["template_id"]))
    return pd.Series(
        [zscore_lookup.get(k, 0.0) for k in keys],
        index=df.index,
        dtype=float,
    )
