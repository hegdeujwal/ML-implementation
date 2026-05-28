"""
scoring/root_cause_engine.py

Identify root cause candidates within each incident cluster and save
the final scored_logs_df.parquet.

Public API
----------
identify_root_causes(scored_df) -> tuple[pd.DataFrame, pd.DataFrame]
    Returns (updated_scored_df, root_causes_df).
    Saves scored_logs_df.parquet and root_causes_df.parquet as side effects.

run() -> tuple[pd.DataFrame, pd.DataFrame]
    Thin wrapper: loads scored_logs_df.parquet and calls identify_root_causes().
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import common.config as cfg
from common.logger import get_logger
from common.utils import load_parquet, save_parquet

logger = get_logger(__name__)

_SCORED_PATH = "data/processed/scored_logs_df.parquet"
_ROOT_CAUSES_PATH = "data/processed/root_causes_df.parquet"

_SCORED_LOG_COLS = [
    "log_id",
    "sequence_number",
    "final_score",
    "label",
    "correlation_id",
    "is_root_cause",
    "root_cause_confidence",
    "is_cross_system",
]


def identify_root_causes(
    scored_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Identify root cause candidates and save final output parquets.

    Parameters
    ----------
    scored_df : pd.DataFrame
        Output of cluster_incidents(). Must contain: correlation_id,
        in_graph, centrality_score, sequence_number, cluster_id,
        is_cross_system, final_score, label.

    Returns
    -------
    (updated_scored_df, root_causes_df)
        updated_scored_df : full df with is_root_cause and
            root_cause_confidence columns added.
        root_causes_df : one row per root cause candidate.
    """
    df = scored_df.copy()

    # Step 1 — initialise output columns
    df["is_root_cause"] = False
    df["root_cause_confidence"] = 0.0

    root_cause_rows: list[dict] = []

    # Step 2 — process each incident
    valid_incidents = df["correlation_id"].dropna().unique()
    logger.info("Processing %d incidents for root cause identification", len(valid_incidents))

    for incident_id in valid_incidents:
        cluster_rows = df[df["correlation_id"] == incident_id]

        # Candidate selection: prefer in_graph=True logs
        in_graph_rows = cluster_rows[cluster_rows["in_graph"] == True]
        if len(in_graph_rows) > 0:
            candidates = in_graph_rows
        else:
            candidates = cluster_rows
            logger.warning(
                "Incident %s: no in-graph logs found, falling back to all %d "
                "logs for root cause selection",
                incident_id, len(cluster_rows),
            )

        # Ranking: centrality descending, capped at ROOT_CAUSE_TOP_N
        candidates_sorted = candidates.sort_values("centrality_score", ascending=False)
        top_candidates = candidates_sorted.head(cfg.ROOT_CAUSE_TOP_N)

        # Confidence
        max_centrality = float(candidates["centrality_score"].max())
        if max_centrality == 0.0:
            confidence_per_candidate = 1.0 / len(candidates)
            logger.warning(
                "Incident %s: max_centrality=0.0, distributing equal confidence "
                "(%.4f) across %d candidates",
                incident_id, confidence_per_candidate, len(candidates),
            )
            confidences = {idx: confidence_per_candidate for idx in top_candidates.index}
        else:
            confidences = {
                idx: float(row["centrality_score"]) / max_centrality
                for idx, row in top_candidates.iterrows()
            }

        # Mark is_root_cause and root_cause_confidence in df
        df.loc[top_candidates.index, "is_root_cause"] = True
        df.loc[top_candidates.index, "root_cause_confidence"] = [
            confidences[idx] for idx in top_candidates.index
        ]

        # Collect root_causes_df rows
        for idx, row in top_candidates.iterrows():
            root_cause_rows.append({
                "incident_id": incident_id,
                "root_cause_log_id": row.get("log_id", f"log_{int(row['sequence_number']):06d}"),
                "confidence_score": confidences[idx],
                "in_graph": bool(row["in_graph"]),
            })

    # Step 3 — assemble and save root_causes_df
    if root_cause_rows:
        root_causes_df = pd.DataFrame(root_cause_rows)
    else:
        root_causes_df = pd.DataFrame(
            columns=["incident_id", "root_cause_log_id", "confidence_score", "in_graph"]
        )
    save_parquet(root_causes_df, _ROOT_CAUSES_PATH)

    # Step 4 — assemble and save scored_logs_df
    # Drop temporal_proximity and all other processing-only columns
    scored_logs_df = df[[c for c in _SCORED_LOG_COLS if c in df.columns]].copy()

    # Validate
    for col in ("sequence_number", "final_score", "label"):
        if scored_logs_df[col].isna().any():
            raise ValueError(f"Column '{col}' has NaN values in scored_logs_df")
    for col in ("final_score", "root_cause_confidence"):
        if not np.isfinite(scored_logs_df[col].to_numpy()).all():
            raise ValueError(f"Column '{col}' has inf or NaN values in scored_logs_df")

    save_parquet(scored_logs_df, _SCORED_PATH)

    n_incidents = len(valid_incidents)
    n_cross = (
        int(
            df[df["correlation_id"].notna()]
            .groupby("correlation_id")["is_cross_system"]
            .first()
            .sum()
        )
        if n_incidents > 0 else 0
    )
    n_root = int(scored_logs_df["is_root_cause"].sum())
    label_dist = scored_logs_df["label"].value_counts().to_dict()
    print(
        f"scored_logs_df: shape={scored_logs_df.shape}, labels={label_dist}, "
        f"incidents={n_incidents}, cross_system_incidents={n_cross}, "
        f"root_cause_candidates={n_root}"
    )

    # Step 5 — return
    return df, root_causes_df


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Thin wrapper: load scored_logs_df.parquet and call identify_root_causes()."""
    df = load_parquet(_SCORED_PATH)
    return identify_root_causes(df)
