"""
scoring/importance_scorer.py

Merge upstream signals (features, anomaly, graph) into a single DataFrame
and compute the per-log final importance score.

Public API
----------
score(features_df, anomaly_df, graph_scores_df) -> pd.DataFrame
    Returns the full merged DataFrame including temporal_proximity.
    Does NOT save to parquet — downstream modules (incident_clusterer)
    need temporal_proximity, and root_cause_engine saves the final output.

run(features_path, anomaly_path, graph_path) -> pd.DataFrame
    Thin wrapper: loads parquets via load_parquet() and calls score().
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import common.config as cfg
from common.logger import get_logger
from common.utils import load_parquet

logger = get_logger(__name__)

_BOOL_FILL_COLS = {"is_anomaly", "in_graph", "in_sequence"}


def score(
    features_df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    graph_scores_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge upstream signals and compute final_score per log.

    Parameters
    ----------
    features_df : pd.DataFrame
        P2 output. Must contain: sequence_number, session_id, timestamp.
    anomaly_df : pd.DataFrame
        P3-ML output. Must contain: sequence_number, combined_score.
    graph_scores_df : pd.DataFrame
        P3-Graph output. Must contain: sequence_number, centrality_score,
        in_graph, cluster_id.

    Returns
    -------
    pd.DataFrame
        Merged df with final_score and temporal_proximity added.
        temporal_proximity is a DBSCAN processing artifact — it is dropped
        before the final scored_logs_df.parquet is written.
    """
    # Step 1 — left join all three on sequence_number
    n_missing_anomaly = int(
        (~features_df["sequence_number"].isin(anomaly_df["sequence_number"])).sum()
    )
    n_missing_graph = int(
        (~features_df["sequence_number"].isin(graph_scores_df["sequence_number"])).sum()
    )

    df = (
        features_df
        .merge(anomaly_df, on="sequence_number", how="left")
        .merge(graph_scores_df, on="sequence_number", how="left")
    )

    logger.info(
        "Merged inputs: total_rows=%d, missing_from_anomaly_df=%d, "
        "missing_from_graph_scores_df=%d",
        len(df), n_missing_anomaly, n_missing_graph,
    )

    # Step 2 — fill missing values
    # Bool columns (is_anomaly, in_graph, in_sequence) → False
    for col in _BOOL_FILL_COLS:
        if col in df.columns:
            null_mask = df[col].isna()
            n = int(null_mask.sum())
            if n:
                df[col] = df[col].fillna(False)
                logger.warning(
                    "Column %s: %d rows filled with False due to missing upstream data",
                    col, n,
                )

    # cluster_id (str) → "UNCAPPED"
    if "cluster_id" in df.columns:
        null_mask = df["cluster_id"].isna()
        n = int(null_mask.sum())
        if n:
            df["cluster_id"] = df["cluster_id"].fillna("UNCAPPED")
            logger.warning(
                "Column cluster_id: %d rows filled with 'UNCAPPED' due to "
                "missing upstream data",
                n,
            )

    # correlated_log_ids (list) → []
    # Use .at to avoid pandas interpreting a list-of-lists as a 2D array.
    if "correlated_log_ids" in df.columns:
        for idx in df.index[df["correlated_log_ids"].isna()]:
            df.at[idx, "correlated_log_ids"] = []

    # Float/int columns from anomaly_df and graph_scores_df → column mean
    _special = (
        {"sequence_number"} | _BOOL_FILL_COLS | {"cluster_id", "correlated_log_ids"}
    )
    fill_cols = list(dict.fromkeys(
        c for c in list(anomaly_df.columns) + list(graph_scores_df.columns)
        if c not in _special
    ))
    for col in fill_cols:
        if col not in df.columns:
            continue
        null_mask = df[col].isna()
        n = int(null_mask.sum())
        if n:
            mean_val = float(df[col].mean())
            df[col] = df[col].fillna(mean_val)
            logger.warning(
                "Column %s: %d rows filled with mean (%.4f) due to missing upstream data",
                col, n, mean_val,
            )

    # Verify critical columns have no nulls after fill
    for req in ("combined_score", "centrality_score"):
        if req in df.columns and df[req].isna().any():
            raise ValueError(
                f"Column '{req}' still has NaN after fill — check upstream data"
            )

    # Step 3 — temporal_proximity per session (for DBSCAN; not saved to parquet)
    if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        ts_num = df["timestamp"].astype("int64")
    else:
        ts_num = df["timestamp"].astype(float)
    df["_ts_num"] = ts_num
    session_min = df.groupby("session_id")["_ts_num"].transform("min")
    session_max = df.groupby("session_id")["_ts_num"].transform("max")
    df["temporal_proximity"] = (df["_ts_num"] - session_min) / (
        session_max - session_min + 1e-10
    )
    df = df.drop(columns=["_ts_num"])

    # Step 4 — final_score (3-term formula, clipped to [0, 1])
    # Severity (event_weight) is an explicit term here, deliberately kept OUT of the
    # IsolationForest features so it contributes exactly once and does not leak the
    # severity label into the unsupervised model. event_weight is part of the P2
    # features contract; default to 0 if a legacy caller omits it.
    severity_term = df["event_weight"] if "event_weight" in df.columns else 0.0
    df["final_score"] = (
        cfg.SCORING_ML_WEIGHT * df["combined_score"]
        + cfg.SCORING_GRAPH_WEIGHT * df["centrality_score"]
        + cfg.SCORING_SEVERITY_WEIGHT * severity_term
    ).clip(0.0, 1.0)

    # Step 5 — validate
    if df["sequence_number"].isna().any():
        raise ValueError("sequence_number has NaN values after merge")
    if df["final_score"].isna().any():
        raise ValueError("final_score has NaN values")
    if not np.isfinite(df["final_score"].to_numpy()).all():
        raise ValueError("final_score has inf or NaN values")

    return df


def run(
    features_path: str = "data/processed/features_df.parquet",
    anomaly_path: str = "data/processed/anomaly_df.parquet",
    graph_path: str = "data/processed/graph_scores_df.parquet",
) -> pd.DataFrame:
    """Full scoring pipeline entry point (called by root pipeline.py).

    Delegates to scoring.pipeline.run_scoring_pipeline() which orchestrates
    all four steps: score → map_labels → cluster_incidents → identify_root_causes.
    The lazy import here breaks the potential circular-import cycle at module
    load time (scoring.pipeline imports score from this module at call time).
    """
    from scoring.pipeline import run_scoring_pipeline
    scored_df, _ = run_scoring_pipeline(features_path, anomaly_path, graph_path)
    return scored_df
