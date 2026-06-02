"""
correlation/chain_builder.py

Chain ID assignment and precursor linking for cross-run incident correlation.

Accepts a list of current-run incidents (with fingerprints) and the
incident_history DataFrame from prior runs.  Returns:

1. An enriched list of incident dicts with chain_id, precursor_incident_id,
   chain_position, and chain_confidence assigned.
2. An updated incident_history DataFrame reflecting any chain merges that
   occurred (fan-out: when a current incident matches multiple historical
   chains, they are unified into one).

Fan-out resolution (per design decision): merge all matched chains into a
single unified chain by picking the lexicographically smallest chain_id and
re-labelling all others to that canonical ID.

Public API
----------
assign_chains(current_incidents, history_df, threshold, lookback_hours)
    -> (enriched_current_incidents: list[dict], updated_history_df: pd.DataFrame)
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

import common.config as cfg
from common.logger import get_logger
from correlation.fingerprint import fingerprint_from_list, jaccard, overlap_coefficient

logger = get_logger(__name__)

# Module-level counter for unique chain ID sequences within a process run.
_chain_seq: int = 0


def _new_chain_id() -> str:
    """Generate a globally unique chain ID: CHAIN-<unix_ts>-<seq:04d>."""
    global _chain_seq
    _chain_seq += 1
    return f"{cfg.CHAIN_ID_PREFIX}-{int(time.time())}-{_chain_seq:04d}"


def assign_chains(
    current_incidents: list[dict[str, Any]],
    history_df: pd.DataFrame,
    threshold: float,
    lookback_hours: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Assign chain linkage to each current incident.

    Parameters
    ----------
    current_incidents : list[dict]
        Each dict must contain:
          - global_incident_id  (str)  globally unique ID for this run's incident
          - template_fingerprint (frozenset[str])
          - start_time  (pd.Timestamp)
          - end_time    (pd.Timestamp)
    history_df : pd.DataFrame
        incident_history parquet loaded (may be empty on first run).
        Expected columns: incident_id, template_fingerprint, chain_id,
        chain_position, end_time.
    threshold : float
        Minimum Jaccard similarity to consider two incidents related.
    lookback_hours : int
        Only query history incidents whose end_time is within this window.

    Returns
    -------
    tuple[list[dict], pd.DataFrame]
        enriched_current_incidents: list with chain fields added.
        updated_history_df: history_df with chain merges applied in-place.
    """
    if history_df is None or len(history_df) == 0:
        # First run ever — no history to link against
        return [
            {
                **inc,
                "chain_id": None,
                "precursor_incident_id": None,
                "chain_position": 1,
                "chain_confidence": 0.0,
            }
            for inc in current_incidents
        ], pd.DataFrame(columns=_history_columns())

    history = history_df.copy()
    results: list[dict] = []

    for curr in current_incidents:
        curr_fp: frozenset = curr["template_fingerprint"]
        curr_start: pd.Timestamp = curr["start_time"]

        # Determine lookback cutoff
        if pd.isnull(curr_start):
            recent_hist = history
        else:
            cutoff = curr_start - pd.Timedelta(hours=lookback_hours)
            # Enforce temporal ordering: precursor must end before current starts
            recent_hist = history[
                (history["end_time"] >= cutoff) &
                (history["end_time"] < curr_start)
            ]

        if len(recent_hist) == 0:
            results.append(_no_chain(curr))
            continue

        # Score every historical incident against the current fingerprint.
        # Use overlap_coefficient for the threshold check (handles size asymmetry:
        # precursor incidents have fewer templates than downstream failures).
        # Report jaccard as chain_confidence for human interpretability.
        matches: list[dict] = []
        for _, h_row in recent_hist.iterrows():
            h_fp = fingerprint_from_list(h_row["template_fingerprint"])
            overlap = overlap_coefficient(curr_fp, h_fp)
            jacc = jaccard(curr_fp, h_fp)
            
            if overlap >= threshold and jacc >= getattr(cfg, "CROSS_RUN_MIN_JACCARD", 0.05):
                matches.append({
                    "incident_id": h_row["incident_id"],
                    "chain_id": h_row.get("chain_id"),
                    "chain_position": h_row.get("chain_position") or 1,
                    "end_time": h_row["end_time"],
                    "similarity": jaccard(curr_fp, h_fp),  # reported confidence
                    "overlap": overlap,
                })

        if not matches:
            results.append(_no_chain(curr))
            continue

        # Collect all distinct non-None chain_ids from matching incidents
        matched_chain_ids = {m["chain_id"] for m in matches if m["chain_id"]}

        if not matched_chain_ids:
            # All matching historical incidents are still unlinked → new chain
            chain_id = _new_chain_id()
            for m in matches:
                _update_history_chain(history, m["incident_id"], chain_id, position=1)
        elif len(matched_chain_ids) == 1:
            chain_id = next(iter(matched_chain_ids))
            # Link any unlinked matching incidents into the existing chain
            for m in matches:
                if not m["chain_id"]:
                    pos = _max_chain_pos(history, chain_id)
                    _update_history_chain(history, m["incident_id"], chain_id, pos)
        else:
            # Fan-out: multiple chains match — merge into the canonical one
            canonical = min(matched_chain_ids)
            for old_chain in matched_chain_ids - {canonical}:
                history.loc[history["chain_id"] == old_chain, "chain_id"] = canonical
                logger.info(
                    "Chain merge: %s absorbed into %s", old_chain, canonical
                )
            chain_id = canonical
            # Link any unlinked matching incidents too
            for m in matches:
                if not m["chain_id"]:
                    pos = _max_chain_pos(history, chain_id)
                    _update_history_chain(history, m["incident_id"], chain_id, pos)

        # Pick the most recent matching incident as the direct precursor
        best = max(matches, key=lambda m: m["end_time"])
        precursor_id = best["incident_id"]
        chain_confidence = float(best["similarity"])

        # chain_position = 1 + max position currently in this chain
        chain_position = _max_chain_pos(history, chain_id) + 1

        logger.info(
            "Incident %s → chain=%s, precursor=%s, "
            "jaccard=%.3f, overlap=%.3f, position=%d",
            curr["global_incident_id"],
            chain_id,
            precursor_id,
            chain_confidence,
            best.get("overlap", 0.0),
            chain_position,
        )

        results.append({
            **curr,
            "chain_id": chain_id,
            "precursor_incident_id": precursor_id,
            "chain_position": chain_position,
            "chain_confidence": chain_confidence,
        })

    return results, history


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _no_chain(inc: dict) -> dict:
    """Return an incident dict with no chain linkage (position=1)."""
    return {
        **inc,
        "chain_id": None,
        "precursor_incident_id": None,
        "chain_position": 1,
        "chain_confidence": 0.0,
    }


def _max_chain_pos(history: pd.DataFrame, chain_id: str) -> int:
    """Return the maximum chain_position currently assigned in this chain."""
    mask = history["chain_id"] == chain_id
    positions = history.loc[mask, "chain_position"].dropna()
    return int(positions.max()) if len(positions) > 0 else 1


def _update_history_chain(
    history: pd.DataFrame,
    incident_id: str,
    chain_id: str,
    position: int,
) -> None:
    """Set chain_id and chain_position for a historical incident row in-place."""
    mask = history["incident_id"] == incident_id
    history.loc[mask, "chain_id"] = chain_id
    history.loc[mask, "chain_position"] = position


def _history_columns() -> list[str]:
    """Canonical column list for an empty incident_history DataFrame."""
    return [
        "incident_id",
        "run_date",
        "run_timestamp",
        "start_time",
        "end_time",
        "template_fingerprint",
        "root_cause_templates",
        "severity",
        "log_count",
        "hosts",
        "is_cross_system",
        "chain_id",
        "precursor_incident_id",
        "chain_position",
        "is_precursor_elevated",
    ]
