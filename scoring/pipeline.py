"""
scoring/pipeline.py

Orchestrate all four scoring steps in sequence.

Public API
----------
run_scoring_pipeline(features_path, anomaly_path, graph_path)
    -> tuple[pd.DataFrame, pd.DataFrame]
    Runs score → map_labels → cluster_incidents → identify_root_causes.
    Returns (scored_df, root_causes_df).

This module is the authoritative entry point for the scoring stage.
The root pipeline.py calls scoring.importance_scorer.run(), which
lazy-imports and delegates here so the call chain stays intact.
"""

from __future__ import annotations

from common.logger import get_logger
from common.utils import load_parquet

logger = get_logger(__name__)


def run_scoring_pipeline(
    features_path: str = "data/processed/features_df.parquet",
    anomaly_path: str = "data/processed/anomaly_df.parquet",
    graph_path: str = "data/processed/graph_scores_df.parquet",
):
    """Run the full scoring stage end-to-end.

    Parameters
    ----------
    features_path : str
        Path to features_df.parquet (P2 output).
    anomaly_path : str
        Path to anomaly_df.parquet (P3-ML output).
    graph_path : str
        Path to graph_scores_df.parquet (P3-Graph output).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (scored_df, root_causes_df)
        scored_df contains all intermediate columns (including in_graph,
        cluster_id, temporal_proximity) for any downstream inspection.
        root_causes_df is the one-row-per-candidate summary.

    Side effects
    ------------
    Saves data/processed/scored_logs_df.parquet and
    data/processed/root_causes_df.parquet via identify_root_causes().
    """
    from scoring.importance_scorer import score
    from scoring.label_mapper import map_labels
    from scoring.incident_clusterer import cluster_incidents
    from scoring.root_cause_engine import identify_root_causes

    logger.info("Scoring step 1/4: loading inputs and computing final_score")
    features_df = load_parquet(features_path)
    anomaly_df = load_parquet(anomaly_path)
    graph_scores_df = load_parquet(graph_path)
    scored_df = score(features_df, anomaly_df, graph_scores_df)

    logger.info("Scoring step 2/4: mapping labels")
    scored_df = map_labels(scored_df)

    logger.info("Scoring step 3/4: clustering incidents")
    scored_df = cluster_incidents(scored_df)

    logger.info("Scoring step 4/4: identifying root causes")
    scored_df, root_causes_df = identify_root_causes(scored_df)

    return scored_df, root_causes_df
