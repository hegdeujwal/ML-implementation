"""
scoring/incident_clusterer.py

Group related logs into incidents using DBSCAN on the merged scoring features.

Only non-ignore-labelled rows are clustered. The raw DBSCAN integer output is
stored in a temporary "_dbscan_label" column and is never written to cluster_id
(which must always remain the "C0000"-format string from the correlation graph).

Public API
----------
cluster_incidents(scored_df) -> pd.DataFrame
    Adds correlation_id and is_cross_system columns; returns updated df.

run() -> pd.DataFrame
    Thin wrapper: loads scored_logs_df.parquet and calls cluster_incidents().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

import common.config as cfg
from common.logger import get_logger
from common.utils import load_parquet

logger = get_logger(__name__)

_SCORED_PATH = "data/processed/scored_logs_df.parquet"


def cluster_incidents(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Group non-ignore logs into incidents via DBSCAN.

    Parameters
    ----------
    scored_df : pd.DataFrame
        Must contain: final_score, centrality_score, temporal_proximity,
        label, cluster_id.

    Returns
    -------
    pd.DataFrame
        Input df with correlation_id and is_cross_system columns added.
        cluster_id is unchanged (still "C0000"-format strings from P3).
    """
    df = scored_df.copy()

    # Initialise output columns — all rows default to no incident
    df["correlation_id"] = None
    df["is_cross_system"] = False

    # Phase 1 — prepare feature matrix for non-ignore rows only
    non_ignore_mask = df["label"] != "ignore"
    X_df = df.loc[
        non_ignore_mask, ["final_score", "centrality_score", "temporal_proximity"]
    ].copy()

    if len(X_df) == 0:
        logger.info("No non-ignore rows to cluster — skipping DBSCAN")
        return df

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_df)

    # Phase 2 — DBSCAN
    dbscan = DBSCAN(eps=cfg.DBSCAN_EPS, min_samples=cfg.DBSCAN_MIN_SAMPLES)
    raw_labels = dbscan.fit_predict(X_scaled)

    # Store raw DBSCAN integers in a temporary column — NEVER in "cluster_id"
    df["_dbscan_label"] = np.nan
    df.loc[non_ignore_mask, "_dbscan_label"] = raw_labels.astype(float)

    # Convert to correlation_id strings; noise (-1) → None
    corr_id_values = pd.Series(
        [None if lbl == -1 else f"INC-{int(lbl):04d}" for lbl in raw_labels],
        index=df.index[non_ignore_mask],
    )
    df.loc[non_ignore_mask, "correlation_id"] = corr_id_values

    n_incidents = len(df["correlation_id"].dropna().unique())
    n_noise = int((raw_labels == -1).sum())
    noise_ratio = n_noise / len(X_df)
    cluster_sizes = df[df["correlation_id"].notna()].groupby("correlation_id").size()
    largest = int(cluster_sizes.max()) if len(cluster_sizes) > 0 else 0
    logger.info(
        "Incidents found: %d, noise_ratio=%.2f, largest_incident_size=%d",
        n_incidents, noise_ratio, largest,
    )

    # Phase 3 — is_cross_system flag per incident
    valid_incidents = df["correlation_id"].dropna().unique()
    n_cross_system = 0
    for cid in valid_incidents:
        mask = df["correlation_id"] == cid
        n_unique_cluster_ids = df.loc[mask, "cluster_id"].nunique()
        is_cross = bool(n_unique_cluster_ids > 1)
        df.loc[mask, "is_cross_system"] = is_cross
        if is_cross:
            n_cross_system += 1

    logger.info("Cross-system incidents: %d", n_cross_system)

    # Phase 4 — clean up temporary column
    df = df.drop(columns=["_dbscan_label"])

    # Sanity check: "cluster_id" must still be string dtype — DBSCAN must not
    # have leaked its integer labels into it
    if "cluster_id" in df.columns and len(df) > 0:
        assert not pd.api.types.is_integer_dtype(df["cluster_id"]), (
            "cluster_id column is integer dtype — DBSCAN label leaked into cluster_id"
        )

    return df


def run() -> pd.DataFrame:
    """Thin wrapper: load scored_logs_df.parquet and call cluster_incidents()."""
    df = load_parquet(_SCORED_PATH)
    return cluster_incidents(df)
