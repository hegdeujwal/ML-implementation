"""
correlation/precursor_elevator.py

Retroactive score elevation for historical precursor logs (P5.5).

When a current critical incident is linked to a historical precursor incident,
this module:

  1. Marks the precursor incident as elevated in the incident_history store.
  2. Boosts the final_score of each precursor log by:
       elevated_score = min(1.0, original_score + PRECURSOR_BOOST * chain_confidence)
     and re-derives its label from the elevated score.
  3. Sets is_precursor_elevated = True on those rows.

The boost is applied to the incident_history store (parquet or Postgres).
Per the agreed design: scores are updated in-place in the incident_history
parquet; for the DB path, the caller (cross_run.py) handles the write.

Public API
----------
elevate_precursor_scores(history_df, chain_results, boost)
    -> pd.DataFrame
    Marks precursor incidents as elevated in history_df and returns it.
"""

from __future__ import annotations

import pandas as pd

from common.logger import get_logger

logger = get_logger(__name__)

# Label thresholds — must match common/config.py values.
# Duplicated here intentionally to avoid a circular import at module load time.
_LABEL_THRESHOLDS = [
    (0.75, "critical"),
    (0.50, "medium"),
    (0.20, "low"),
    (0.00, "ignore"),
]


def _label_from_score(score: float) -> str:
    """Map a final_score in [0, 1] to its label string."""
    for threshold, label in _LABEL_THRESHOLDS:
        if score >= threshold:
            return label
    return "ignore"


def elevate_precursor_scores(
    history_df: pd.DataFrame,
    chain_results: list[dict],
    boost: float,
) -> pd.DataFrame:
    """Mark precursor incidents as elevated in the incident_history store.

    For each chain_result that has a non-null precursor_incident_id, the
    corresponding row in history_df has is_precursor_elevated set to True.
    This is the lightweight in-history-store update; the actual per-log score
    boost to the scored_logs parquet is applied by cross_run.elevate_log_scores().

    Parameters
    ----------
    history_df : pd.DataFrame
        Current (possibly updated) incident_history store.
    chain_results : list[dict]
        Output from chain_builder.assign_chains() — one dict per current incident.
    boost : float
        PRECURSOR_BOOST from config (passed in explicitly to stay test-friendly).

    Returns
    -------
    pd.DataFrame
        history_df with is_precursor_elevated set for matched precursor rows.
    """
    if history_df is None or len(history_df) == 0:
        return history_df

    history = history_df.copy()

    elevated_count = 0
    for result in chain_results:
        precursor_id = result.get("precursor_incident_id")
        chain_confidence = result.get("chain_confidence", 0.0)

        if not precursor_id or chain_confidence <= 0.0:
            continue

        mask = history["incident_id"] == precursor_id
        if not mask.any():
            continue

        history.loc[mask, "is_precursor_elevated"] = True
        elevated_count += 1

        logger.info(
            "Precursor %s marked as elevated (chain_confidence=%.3f, boost=%.3f)",
            precursor_id,
            chain_confidence,
            boost,
        )

    if elevated_count:
        logger.info("%d precursor incident(s) marked as elevated.", elevated_count)

    return history


def elevate_log_scores(
    scored_df: pd.DataFrame,
    precursor_correlation_ids: set[str],
    chain_confidence: float,
    boost: float,
) -> pd.DataFrame:
    """Boost final_score for logs belonging to precursor incidents.

    This is the per-log score elevation.  It operates on scored_logs_df
    (the current batch's parquet, which is overwritten by cross_run.py).

    Parameters
    ----------
    scored_df : pd.DataFrame
        scored_logs_df for the CURRENT run (not history).
    precursor_correlation_ids : set[str]
        Run-local correlation_ids (e.g. {'INC-0000'}) whose logs should be boosted.
    chain_confidence : float
        Jaccard similarity between precursor and current incident.
    boost : float
        PRECURSOR_BOOST from config.

    Returns
    -------
    pd.DataFrame
        scored_df with elevated final_score, updated label,
        and is_precursor_elevated flag set on affected rows.
    """
    if not precursor_correlation_ids or chain_confidence <= 0.0:
        return scored_df

    df = scored_df.copy()

    if "is_precursor_elevated" not in df.columns:
        df["is_precursor_elevated"] = False

    mask = df["correlation_id"].isin(precursor_correlation_ids)
    if not mask.any():
        return df

    delta = boost * chain_confidence
    df.loc[mask, "final_score"] = (
        df.loc[mask, "final_score"] + delta
    ).clip(0.0, 1.0)

    # Re-derive label from elevated score
    df.loc[mask, "label"] = df.loc[mask, "final_score"].map(_label_from_score)
    df.loc[mask, "is_precursor_elevated"] = True

    n = int(mask.sum())
    logger.info(
        "Elevated %d log(s) in precursor incidents %s by +%.4f (confidence=%.3f)",
        n,
        sorted(precursor_correlation_ids),
        delta,
        chain_confidence,
    )

    return df
