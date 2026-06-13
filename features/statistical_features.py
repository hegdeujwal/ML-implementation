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

    Edge cases:
    - Fewer than 2 rows: no inter-arrivals possible, returns 0.0.
    - Exactly 2 rows: one inter-arrival, variance is undefined (NaN), returns 0.0.
    - Mean of inter-arrivals == 0: all events simultaneous, returns 0.0.
    """
    if len(session_df) < 2:
        return pd.Series(0.0, index=session_df.index, dtype=float)

    sorted_ts = session_df["timestamp"].sort_values()
    inter_arrivals = sorted_ts.diff().dt.total_seconds().dropna()

    if len(inter_arrivals) < 2:
        # Only one interval — variance is undefined, no burstiness signal.
        return pd.Series(0.0, index=session_df.index, dtype=float)

    mean = inter_arrivals.mean()
    if mean == 0:
        return pd.Series(0.0, index=session_df.index, dtype=float)

    variance = inter_arrivals.var()
    if not np.isfinite(variance):
        return pd.Series(0.0, index=session_df.index, dtype=float)

    fano = float(np.clip(variance / mean, 0.0, 10.0))
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
    seen_cap: int | None = None,
) -> pd.Series:
    """Cross-run rolling z-score using Welford's online algorithm.

    Persists per-(host, template_id) running statistics to disk so that
    z-scores are computed against ALL historical sessions, not only the
    sessions present in the current batch.  This is the primary mechanism
    for detecting slow drift across pipeline runs.

    Each session is scored against the baseline accumulated BEFORE it:
    sessions are processed in chronological order, the z-score is computed
    from the pre-update statistics, and only then is the session folded into
    the baseline.  (Scoring against post-update stats lets an anomalous spike
    raise its own baseline mean and shrink its own z — self-contamination.)

    Re-runs are idempotent: the z-score computed for each session is stored
    alongside its ID, so a session already incorporated in the store replays
    its original score exactly (and is never double-counted). Legacy stores
    that recorded only session IDs fall back to leave-one-out removal
    (reverse Welford), which approximates the original score against all
    OTHER observations.

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
        seen_cap:   Maximum session IDs remembered per (host, template_id)
                    for re-run deduplication. Defaults to
                    ZSCORE_BASELINE_SEEN_CAP from config. Session IDs embed
                    the session start timestamp, so the cap evicts oldest
                    first; an evicted session re-run would be double-counted
                    (rare, accepted trade-off for a bounded store).

    Returns:
        Float Series aligned to df.index, clipped to [-5.0, 5.0].
        Returns 0.0 for groups with fewer than 2 prior observations.
    """
    from pathlib import Path
    import json

    if seen_cap is None:
        from common.config import ZSCORE_BASELINE_SEEN_CAP
        seen_cap = ZSCORE_BASELINE_SEEN_CAP

    store_p = Path(store_path)

    # --- Load existing baseline store into plain dicts ---
    # stats[(host, template_id)] = {"n", "mean", "M2", "seen"}
    # "seen" maps session_id -> the z-score originally computed for it
    # (None for legacy stores that recorded only a JSON list of IDs).
    stats: dict[tuple, dict] = {}
    if store_p.exists():
        existing = pd.read_parquet(store_p)
        for row in existing.itertuples(index=False):
            parsed = json.loads(row.seen_session_ids)
            seen = dict.fromkeys(parsed) if isinstance(parsed, list) else dict(parsed)
            stats[(row.host, row.template_id)] = {
                "n": float(row.n),
                "mean": float(row.welford_mean),
                "M2": float(row.welford_M2),
                "seen": seen,
            }

    # --- Per-(host, session, template) frequency summary, chronological ---
    session_starts = (
        df.groupby(["host", "session_id"])["timestamp"]
        .min().reset_index().rename(columns={"timestamp": "session_start"})
    )
    stf = (
        df.groupby(["host", "session_id", "template_id"])["frequency"]
        .first().reset_index()
    )
    stf = stf.merge(session_starts, on=["host", "session_id"])
    stf = stf.sort_values("session_start", kind="stable")

    # --- Single chronological pass: score against prior stats, then update ---
    zscore_lookup: dict = {}
    for row in stf.itertuples(index=False):
        key = (row.host, row.template_id)
        freq = float(row.frequency)
        sid = str(row.session_id)

        rec = stats.setdefault(
            key, {"n": 0.0, "mean": 0.0, "M2": 0.0, "seen": {}}
        )
        n, mean, M2 = rec["n"], rec["mean"], rec["M2"]

        if sid in rec["seen"]:
            stored_z = rec["seen"][sid]
            if stored_z is not None:
                # Re-run: replay the originally computed score exactly.
                z = float(stored_z)
            elif n >= 3.0:
                # Legacy store without per-session scores — approximate via
                # exact leave-one-out (reverse Welford). Removal needs
                # n-1 >= 2 remaining observations for a sample std.
                mean_wo = (n * mean - freq) / (n - 1.0)
                M2_wo = max(M2 - (freq - mean_wo) * (freq - mean), 0.0)
                std = float(np.sqrt(M2_wo / (n - 2.0)))
                z = (freq - mean_wo) / std if std > 1e-9 else 0.0
            else:
                z = 0.0
        else:
            # Score against the PRE-update baseline ...
            if n >= 2.0 and M2 > 0.0:
                std = float(np.sqrt(M2 / (n - 1.0)))
                z = (freq - mean) / std if std > 1e-9 else 0.0
            else:
                z = 0.0
            # ... then fold this session into the baseline.
            n += 1.0
            delta = freq - mean
            mean += delta / n
            M2 += delta * (freq - mean)
            rec["n"], rec["mean"], rec["M2"] = n, mean, M2
            rec["seen"][sid] = float(np.clip(z, -5.0, 5.0))

        zscore_lookup[(row.host, row.session_id, row.template_id)] = float(
            np.clip(z, -5.0, 5.0)
        )

    # --- Persist updated store (seen IDs capped, oldest evicted first) ---
    # seen_session_ids is a JSON object {session_id: original_z} so re-runs
    # replay scores exactly; sorting by ID is chronological (IDs embed the
    # session start timestamp).
    store_rows = [
        {
            "host": host,
            "template_id": template_id,
            "n": rec["n"],
            "welford_mean": rec["mean"],
            "welford_M2": rec["M2"],
            "seen_session_ids": json.dumps(
                dict(sorted(rec["seen"].items())[-seen_cap:])
            ),
        }
        for (host, template_id), rec in stats.items()
    ]
    store = pd.DataFrame(
        store_rows,
        columns=["host", "template_id", "n", "welford_mean", "welford_M2",
                 "seen_session_ids"],
    )
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

