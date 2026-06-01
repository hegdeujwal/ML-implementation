"""
Counter anomaly proximity scoring.

compute_counter_proximity scores each log row by its temporal proximity to
counter/interface anomaly events.  Proximity is host-scoped and time-scoped —
not session-scoped.  Uses np.searchsorted for O(n log m) per-host performance.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from common.config import (
    COUNTER_ANOMALY_PATTERNS,
    COUNTER_ANOMALY_HINT_KEYWORDS,
    COUNTER_PROXIMITY_WINDOW_SECONDS,
)
from common.logger import get_logger

logger = get_logger(__name__)


def compute_counter_proximity(df: pd.DataFrame) -> pd.Series:
    """Score each log row by its temporal proximity to counter anomaly events.

    Step 1 — identify counter anomaly rows via regex patterns from config.
    Step 2 — warn (once per template_id) about unmatched templates that contain
             hint keywords suggesting they may be counter events.
    Step 3 — for each host independently, use np.searchsorted to find the nearest
             counter event within ±COUNTER_PROXIMITY_WINDOW_SECONDS and compute
             proximity = 1 / (1 + distance_in_seconds).

    Args:
        df: Full features DataFrame; must have columns host, template_id, timestamp.

    Returns:
        Float Series aligned to df.index, range [0.0, 1.0].
    """
    # --- Step 1: identify counter anomaly rows ---
    # Match against unique template_ids to avoid re-running regex per row
    unique_templates = df["template_id"].unique()
    matched_templates = {
        t for t in unique_templates
        if any(re.search(p, t, re.IGNORECASE) for p in COUNTER_ANOMALY_PATTERNS)
    }
    counter_mask = df["template_id"].isin(matched_templates)
    counter_events_df = df[counter_mask]

    # --- Step 2: hint keyword warnings (one WARNING per unseen template_id) ---
    hint_re = re.compile(
        "|".join(re.escape(kw) for kw in COUNTER_ANOMALY_HINT_KEYWORDS),
        re.IGNORECASE,
    )
    warned: set[str] = set()
    for tid in df.loc[~counter_mask, "template_id"].unique():
        if hint_re.search(tid) and tid not in warned:
            warned.add(tid)
            logger.warning(
                f"Possible unmatched counter anomaly template: {tid}. "
                "Add to COUNTER_ANOMALY_PATTERNS in config if this is a counter event."
            )

    # --- Step 3: vectorised proximity per host ---
    result = pd.Series(0.0, index=df.index, dtype=float)

    if counter_events_df.empty:
        return result

    for host, host_df in df.groupby("host"):
        host_ce = counter_events_df[counter_events_df["host"] == host]
        if host_ce.empty:
            continue

        # Convert timestamps to float seconds since epoch for arithmetic
        ce_times = np.sort(
            pd.to_datetime(host_ce["timestamp"]).values.astype(np.int64)
        ) / 1e9
        row_times = pd.to_datetime(host_df["timestamp"]).values.astype(np.int64) / 1e9

        # searchsorted gives the insertion index; nearest event is at idx or idx-1
        idxs = np.searchsorted(ce_times, row_times)

        right_dist = np.where(
            idxs < len(ce_times),
            np.abs(ce_times[np.clip(idxs, 0, len(ce_times) - 1)] - row_times),
            np.inf,
        )
        left_dist = np.where(
            idxs > 0,
            np.abs(ce_times[np.clip(idxs - 1, 0, len(ce_times) - 1)] - row_times),
            np.inf,
        )

        min_dist = np.minimum(left_dist, right_dist)
        prox = np.where(
            min_dist <= COUNTER_PROXIMITY_WINDOW_SECONDS,
            1.0 / (1.0 + min_dist),
            0.0,
        )

        result.loc[host_df.index] = prox

    return result
