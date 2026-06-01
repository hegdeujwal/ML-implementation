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


# ---------------------------------------------------------------------------
# Persistent z-score baseline (Welford's online algorithm)
# ---------------------------------------------------------------------------

def zscore_base_persistent(
    df: pd.DataFrame,
    store_path: str,
) -> pd.Series:
    """Cross-run rolling z-score using Welford's online algorithm.

    Persists per-(host, template_id) running statistics to disk so that
    z-scores are computed against ALL historical sessions, not only the
    sessions present in the current batch.  This is the primary mechanism
    for detecting slow drift across pipeline runs.

    Algorithm (Welford's):
        n    += 1
        delta = freq - mean
        mean += delta / n
        M2   += delta * (freq - mean)   # uses updated mean
        std   = sqrt(M2 / (n-1))        # sample std, defined when n >= 2

    Args:
        df:         Full sessionized DataFrame with host, session_id,
                    template_id, frequency, and timestamp columns.
        store_path: Path to the Welford baseline parquet.  Created on first
                    run; updated in-place on every subsequent run.

    Returns:
        Float Series aligned to df.index, clipped to [-5.0, 5.0].
        Returns 0.0 for groups with fewer than 2 historical observations.
    """
    from pathlib import Path
    import json

    store_p = Path(store_path)

    # --- Load existing baseline store ---
    if store_p.exists():
        store = pd.read_parquet(store_p)
        # seen_session_ids is stored as JSON string per row
        store["_seen_ids"] = store["seen_session_ids"].apply(json.loads)
    else:
        store = pd.DataFrame(
            columns=["host", "template_id", "n", "welford_mean", "welford_M2",
                     "seen_session_ids", "_seen_ids"]
        )
        store["_seen_ids"] = pd.Series(dtype=object)

    # Build a lookup: (host, template_id) -> row index in store
    store = store.set_index(["host", "template_id"])

    # --- Compute per-(host, session, template) frequency summary ---
    session_starts = (
        df.groupby(["host", "session_id"])["timestamp"]
        .min().reset_index().rename(columns={"timestamp": "session_start"})
    )
    stf = (
        df.groupby(["host", "session_id", "template_id"])["frequency"]
        .first().reset_index()
    )
    stf = stf.merge(session_starts, on=["host", "session_id"])

    # --- Apply Welford updates for new sessions ---
    store_updates: dict = {}  # (host, template_id) -> updated dict

    for _, row in stf.iterrows():
        key = (row["host"], row["template_id"])
        freq = float(row["frequency"])
        sid = str(row["session_id"])

        if key in store.index:
            rec = store.loc[key].to_dict()
            seen = set(rec.get("_seen_ids") or [])
            n = float(rec["n"])
            mean = float(rec["welford_mean"])
            M2 = float(rec["welford_M2"])
        else:
            seen = set()
            n, mean, M2 = 0.0, 0.0, 0.0

        # Idempotent: skip sessions already incorporated
        if sid in seen:
            store_updates[key] = {"n": n, "welford_mean": mean,
                                  "welford_M2": M2, "_seen_ids": seen}
            continue

        # Welford online update
        n += 1.0
        delta = freq - mean
        mean += delta / n
        M2 += delta * (freq - mean)
        seen.add(sid)

        store_updates[key] = {"n": n, "welford_mean": mean,
                              "welford_M2": M2, "_seen_ids": seen}

    # Merge updates back into store
    for (host, template_id), upd in store_updates.items():
        key = (host, template_id)
        if key not in store.index:
            store.loc[key, :] = None
        store.at[key, "n"] = upd["n"]
        store.at[key, "welford_mean"] = upd["welford_mean"]
        store.at[key, "welford_M2"] = upd["welford_M2"]
        store.at[key, "_seen_ids"] = upd["_seen_ids"]

    # --- Compute z-scores for the current batch using post-update stats ---
    zscore_lookup: dict = {}
    for _, row in stf.iterrows():
        key = (row["host"], row["template_id"])
        sid = row["session_id"]
        freq = float(row["frequency"])

        if key in store.index:
            n = float(store.at[key, "n"])
            mean = float(store.at[key, "welford_mean"])
            M2 = float(store.at[key, "welford_M2"])
        else:
            n, mean, M2 = 1.0, freq, 0.0

        lookup_key = (row["host"], sid, row["template_id"])
        if n < 2 or M2 <= 0.0:
            zscore_lookup[lookup_key] = 0.0
        else:
            std = float(np.sqrt(M2 / (n - 1.0)))
            if std < 1e-9:
                zscore_lookup[lookup_key] = 0.0
            else:
                z = (freq - mean) / std
                zscore_lookup[lookup_key] = float(np.clip(z, -5.0, 5.0))

    # --- Persist updated store ---
    store = store.reset_index()
    store["seen_session_ids"] = store["_seen_ids"].apply(
        lambda s: json.dumps(sorted(s)) if isinstance(s, set) else json.dumps([])
    )
    store = store.drop(columns=["_seen_ids"])
    store_p.parent.mkdir(parents=True, exist_ok=True)
    store.to_parquet(store_p, index=False)

    # Map back to every row in df
    keys = list(zip(df["host"], df["session_id"], df["template_id"]))
    return pd.Series(
        [zscore_lookup.get(k, 0.0) for k in keys],
        index=df.index,
        dtype=float,
    )


def update_feature_rolling_store(
    features_df: pd.DataFrame,
    store_path: str,
    max_sessions: int,
) -> None:
    """Append current batch features to the rolling store for IF retraining.

    Keeps only the most recent max_sessions unique sessions (by session_start
    time) so the store does not grow unbounded.  Deduplicates by session_id
    so re-running the pipeline on the same data is idempotent.

    Args:
        features_df: Output of feature_pipeline.run_pipeline() — must contain
                     session_id and timestamp columns.
        store_path:  Path to the rolling store parquet.
        max_sessions: Maximum number of unique sessions to retain.
    """
    from pathlib import Path

    store_p = Path(store_path)

    if store_p.exists():
        existing = pd.read_parquet(store_p)
        combined = pd.concat([existing, features_df], ignore_index=True)
        # Deduplicate rows by session_id — keep the latest version (from new batch)
        combined = combined.drop_duplicates(
            subset=["session_id", "sequence_number"], keep="last"
        )
    else:
        combined = features_df.copy()

    # Determine the most recent max_sessions sessions chronologically
    session_starts = (
        combined.groupby("session_id")["timestamp"]
        .min()
        .sort_values()
    )
    keep_sessions = session_starts.index[-max_sessions:].tolist()
    combined = combined[combined["session_id"].isin(keep_sessions)]

    store_p.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(store_p, index=False)

