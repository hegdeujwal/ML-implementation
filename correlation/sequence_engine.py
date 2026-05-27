"""
correlation/sequence_engine.py

Detect recurring ordered log sequences within sessions.

A "sequence" is an ordered list of log templates (length >= SEQUENCE_MIN_LENGTH)
that occurs in at least SEQUENCE_MIN_SUPPORT distinct sessions.  Two consecutive
templates in a sequence must appear within SEQUENCE_WINDOW_SECONDS of each other.

Algorithm
---------
1. Group events by session_id, sort by timestamp within each session.
2. For every session, generate all ordered sub-sequences of length
   [min_length, min_length + 2] using a sliding window capped at
   SEQUENCE_WINDOW_SECONDS.  We cap max length at min_length + 2 by default
   to keep the search space tractable; for most log analytics use cases
   3- to 5-event sequences are the most actionable.
3. Count sessions exhibiting each unique sequence tuple.
4. Retain sequences where session_count >= min_support.
5. Write sequences.json and return the set of log_ids that are part of any
   retained sequence.

Output file format (sequences.json)
-------------------------------------
[
  {
    "sequence": ["IF_DOWN", "BGP_PEER_RESET", "OSPF_ADJACENCY_LOST"],
    "support_count": 12,
    "session_ids": ["session_001", "session_007", ...]
  },
  ...
]

Public API
----------
detect_sequences(df, window_seconds, min_length, min_support, output_path)
    -> set[str]   -- log_ids that are part of at least one detected sequence
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Optional

import pandas as pd

import common.config as cfg


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_session_sequences(
    session_df: pd.DataFrame,
    window_seconds: int,
    min_length: int,
) -> dict[tuple, list]:
    """Return mapping of sequence_tuple -> [sequence_number, ...] for one session.

    Only sequence_numbers that are part of a detected candidate sequence are
    included. A candidate must fit within window_seconds end-to-end.

    Parameters
    ----------
    session_df : pd.DataFrame
        Single-session slice, sorted by timestamp.  Columns: sequence_number,
        timestamp, template_id.
    window_seconds : int
        Maximum elapsed time between first and last event in a sequence.
    min_length : int
        Minimum number of templates in a sequence.

    Returns
    -------
    dict mapping sequence tuple -> list of contributing sequence_numbers
    """
    rows = session_df[["sequence_number", "timestamp", "template_id"]].values.tolist()
    n = len(rows)
    found: dict[tuple, list] = {}

    # Sliding window: for each start index, grow right while within window
    for i in range(n):
        seq_templates: list[str] = [rows[i][2]]
        seq_ids: list = [rows[i][0]]
        ts_start: float = rows[i][1]

        for j in range(i + 1, n):
            ts_j = rows[j][1]
            if ts_j - ts_start > window_seconds:
                break
            seq_templates.append(rows[j][2])
            seq_ids.append(rows[j][0])

            if len(seq_templates) >= min_length:
                key = tuple(seq_templates[:min_length])
                if key not in found:
                    found[key] = list(seq_ids[:min_length])

    return found


# ---------------------------------------------------------------------------
# Public: detect_sequences
# ---------------------------------------------------------------------------

def detect_sequences(
    df: pd.DataFrame,
    window_seconds: Optional[int] = None,
    min_length: Optional[int] = None,
    min_support: Optional[int] = None,
    output_path: Optional[str] = None,
) -> set:
    """Detect recurring ordered log sequences across sessions.

    Parameters
    ----------
    df : pd.DataFrame
        Sessionized log DataFrame.  Required columns:
        sequence_number, session_id, timestamp, template_id.
    window_seconds : int, optional
        Time window within which consecutive templates must occur.
        Defaults to cfg.SEQUENCE_WINDOW_SECONDS (30 s).
    min_length : int, optional
        Minimum sequence length.  Defaults to cfg.SEQUENCE_MIN_LENGTH (3).
    min_support : int, optional
        Minimum number of distinct sessions exhibiting the sequence.
        Defaults to cfg.SEQUENCE_MIN_SUPPORT (5).
    output_path : str, optional
        Path to write sequences.json.
        Defaults to cfg.SEQUENCES_JSON_PATH.

    Returns
    -------
    set of int
        sequence_numbers that are part of at least one retained sequence.
        Caller updates graph_scores_df["in_sequence"] via:
            graph_scores_df["in_sequence"] = (
                graph_scores_df["sequence_number"].isin(sequence_number_set))
    """
    if window_seconds is None:
        window_seconds = cfg.SEQUENCE_WINDOW_SECONDS
    if min_length is None:
        min_length = cfg.SEQUENCE_MIN_LENGTH
    if min_support is None:
        min_support = cfg.SEQUENCE_MIN_SUPPORT
    if output_path is None:
        output_path = cfg.SEQUENCES_JSON_PATH

    required_cols = {"sequence_number", "session_id", "timestamp", "template_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"detect_sequences: missing columns: {missing}")

    # Normalise timestamp to float seconds (Unix epoch) so arithmetic works
    # regardless of whether the upstream parquet stored datetime64 or float.
    df = df.copy()
    if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = df["timestamp"].astype("int64") / 1e9

    df = df.sort_values(["session_id", "timestamp"]).reset_index(drop=True)

    # sequence_tuple -> {session_id -> [sequence_numbers in that session]}
    sequence_sessions: dict[tuple, dict[str, list]] = defaultdict(dict)

    for session_id, session_df in df.groupby("session_id"):
        session_seqs = _extract_session_sequences(
            session_df, window_seconds, min_length
        )
        for seq_key, seq_nums in session_seqs.items():
            sequence_sessions[seq_key][str(session_id)] = seq_nums

    # Filter by min_support
    retained: list[dict] = []
    in_sequence_ids: set = set()

    for seq_key, sessions_map in sequence_sessions.items():
        support = len(sessions_map)
        if support >= min_support:
            all_ids = [sid for sids in sessions_map.values() for sid in sids]
            in_sequence_ids.update(all_ids)
            retained.append({
                "sequence": list(seq_key),
                "support_count": support,
                "session_ids": sorted(sessions_map.keys()),
            })

    # Sort by support_count descending for readability
    retained.sort(key=lambda r: -r["support_count"])

    # Persist sequences.json
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(retained, fh, indent=2)

    return in_sequence_ids
