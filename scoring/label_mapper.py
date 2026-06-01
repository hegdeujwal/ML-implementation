"""
scoring/label_mapper.py

Map final_score to a human-readable severity label.

Public API
----------
map_labels(scored_df) -> pd.DataFrame
    Adds a "label" column; returns the updated DataFrame.

run() -> pd.DataFrame
    Thin wrapper: loads scored_logs_df.parquet and calls map_labels().
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.config import LABEL_IGNORE_MAX, LABEL_LOW_MAX, LABEL_MEDIUM_MAX
from common.logger import get_logger
from common.utils import load_parquet

logger = get_logger(__name__)

_SCORED_PATH = "data/processed/scored_logs_df.parquet"


def map_labels(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Add a label column derived from final_score.

    Thresholds (all from common/config.py):
        final_score <= LABEL_IGNORE_MAX  → "ignore"
        final_score <= LABEL_LOW_MAX     → "low"
        final_score <= LABEL_MEDIUM_MAX  → "medium"
        final_score >  LABEL_MEDIUM_MAX  → "critical"

    Parameters
    ----------
    scored_df : pd.DataFrame
        Must contain a "final_score" column with values in [0, 1].

    Returns
    -------
    pd.DataFrame
        Input df with "label" column added.
    """
    df = scored_df.copy()

    df["label"] = np.select(
        [
            df["final_score"] <= LABEL_IGNORE_MAX,
            df["final_score"] <= LABEL_LOW_MAX,
            df["final_score"] <= LABEL_MEDIUM_MAX,
        ],
        ["ignore", "low", "medium"],
        default="critical",
    )

    if df["label"].isna().any():
        raise ValueError("label column has NaN values after mapping — check final_score column")

    total = len(df)
    dist = df["label"].value_counts()
    for lbl, count in dist.items():
        logger.info("Label %-8s: %5d rows (%5.1f%%)", lbl, count, 100 * count / total)

    return df


def run() -> pd.DataFrame:
    """Thin wrapper: load scored_logs_df.parquet and call map_labels()."""
    df = load_parquet(_SCORED_PATH)
    return map_labels(df)
