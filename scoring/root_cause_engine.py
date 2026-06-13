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
    "sequence_number",
    "final_score",
    "label",
    "incident_id",
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
        Output of cluster_incidents(). Must contain: correlation_id or incident_id,
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
    
    # Normalize: rename correlation_id → incident_id for internal processing
    if "correlation_id" in df.columns and "incident_id" not in df.columns:
        df = df.rename(columns={"correlation_id": "incident_id"})

    # Step 1 — initialise output columns
    df["is_root_cause"] = False
    df["root_cause_confidence"] = 0.0

    root_cause_rows: list[dict] = []

    # Step 2 — process each incident
    valid_incidents = df["incident_id"].dropna().unique()
    logger.info("Processing %d incidents for root cause identification", len(valid_incidents))

    for incident_id in valid_incidents:
        cluster_rows = df[df["incident_id"] == incident_id]

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
                "root_cause_log_id": f"log_{int(row['sequence_number']):06d}",
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
    # Drop temporal_proximity and all other processing-only columns.
    # Rename internal incident_id back to correlation_id (canonical schema name).
    _output_cols = ["sequence_number", "final_score", "label", "incident_id", "is_root_cause",
                    "root_cause_confidence", "is_cross_system"]
    # Audit flags from the scorer: True where the row's upstream score was
    # mean-filled rather than computed (absent on legacy callers).
    _output_cols += [c for c in ("anomaly_missing", "graph_missing") if c in df.columns]
    scored_logs_df = df[_output_cols].copy()
    scored_logs_df = scored_logs_df.rename(columns={"incident_id": "correlation_id"})

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
            df[df["incident_id"].notna()]
            .groupby("incident_id")["is_cross_system"]
            .first()
            .sum()
        )
        if n_incidents > 0 else 0
    )
    n_root = int(scored_logs_df["is_root_cause"].sum())
    label_dist = scored_logs_df["label"].value_counts().to_dict()
    logger.info(
        "scored_logs_df: shape=%s, labels=%s, incidents=%d, "
        "cross_system_incidents=%d, root_cause_candidates=%d",
        scored_logs_df.shape,
        label_dist,
        n_incidents,
        n_cross,
        n_root,
    )

    # Step 5 — rename internal incident_id back to correlation_id before returning
    # so callers can filter on correlation_id (the canonical AGENTS.md column name).
    df = df.rename(columns={"incident_id": "correlation_id"})

    return df, root_causes_df


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Thin wrapper: load scored_logs_df.parquet and call identify_root_causes()."""
    df = load_parquet(_SCORED_PATH)
    return identify_root_causes(df)
