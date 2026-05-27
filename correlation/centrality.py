"""
correlation/centrality.py

Compute per-node centrality scores for the co-occurrence graph and assemble
the per-log-row output DataFrame (P3 → P4 contract).

Public API
----------
compute_centrality(graph, sessionized_logs)
    Accept the nx.Graph from graph_builder and the sessionized log DataFrame.
    Compute betweenness and PageRank centrality; normalise to [0,1]; join back
    to individual log rows; compute correlated_log_ids; save graph_scores_df.

Centrality metrics
------------------
degree
    Raw node degree from graph.degree[v]. Stored as int.

betweenness_centrality
    k=50 approximation for performance; exact betweenness is O(n³).
    NetworkX normalised=True already divides by possible path count.

pagerank  (used as centrality_score)
    alpha=GRAPH_PAGERANK_ALPHA=0.85. Min-max normalised to [0,1].
    Degenerate case (single node or all-equal scores) → 0.5 instead of 0.0.

Capped templates (outside GRAPH_MAX_NODES)
    centrality_score = global mean of in-graph PageRank values
    betweenness      = global mean of in-graph betweenness values
    degree           = 0
    in_graph         = False
    cluster_id       = "UNCAPPED"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx

import common.config as cfg
from common.logger import get_logger
from common.utils import save_parquet, validate_schema

logger = get_logger(__name__)


def compute_centrality(
    graph: nx.Graph,
    sessionized_logs: pd.DataFrame,
) -> pd.DataFrame:
    """Compute centrality scores and assemble the P3 output DataFrame.

    Parameters
    ----------
    graph : nx.Graph
        Output of graph_builder.build_graph() or load_or_build_graph().
    sessionized_logs : pd.DataFrame
        Must contain: sequence_number, session_id, template_id.

    Returns
    -------
    pd.DataFrame
        One row per log. Columns match GraphScoreRow:
        sequence_number, centrality_score, degree, betweenness,
        in_graph, cluster_id, in_sequence, correlated_log_ids.
    """
    included_templates = graph.graph.get("included_templates", set())

    # Step 1 — compute centrality on in-graph nodes
    if len(graph) == 0:
        pr_norm: dict = {}
        bw_norm: dict = {}
        global_mean_centrality = 0.5
        global_mean_betweenness = 0.5
    else:
        # k approximation — exact betweenness is O(n³), k=50 keeps it fast
        bw_raw = nx.betweenness_centrality(
            graph,
            weight="weight",
            normalized=True,
            k=min(50, len(graph)),
        )

        pr_raw = nx.pagerank(
            graph, alpha=cfg.GRAPH_PAGERANK_ALPHA, weight="weight"
        )

        # Min-max normalise PageRank to [0,1]; degenerate → 0.5 not 0.0
        pr_vals = np.array(list(pr_raw.values()))
        if pr_vals.max() == pr_vals.min():
            pr_norm = {t: 0.5 for t in pr_raw}
        else:
            lo, hi = float(pr_vals.min()), float(pr_vals.max())
            pr_norm = {
                t: float((v - lo) / (hi - lo)) for t, v in pr_raw.items()
            }

        # Betweenness degenerate case → 0.5 not 0.0
        bw_vals = np.array(list(bw_raw.values()))
        if bw_vals.max() == bw_vals.min():
            bw_norm = {t: 0.5 for t in bw_raw}
        else:
            bw_norm = {t: float(v) for t, v in bw_raw.items()}

        global_mean_centrality = float(np.mean(list(pr_norm.values())))
        global_mean_betweenness = float(np.mean(list(bw_norm.values())))

    # Step 2 — build per-template score lookup
    # In-graph templates get computed scores; capped templates get global mean
    all_templates = sessionized_logs["template_id"].unique()
    score_lookup: dict = {}

    for template in included_templates:
        score_lookup[template] = {
            "centrality_score": pr_norm.get(template, global_mean_centrality),
            "degree": int(graph.degree[template]),
            "betweenness": bw_norm.get(template, global_mean_betweenness),
            "in_graph": True,
            "cluster_id": graph.nodes[template].get("cluster_id", "C0000"),
        }

    for template in all_templates:
        if template not in score_lookup:
            score_lookup[template] = {
                "centrality_score": global_mean_centrality,
                "degree": 0,
                "betweenness": global_mean_betweenness,
                "in_graph": False,
                "cluster_id": "UNCAPPED",
            }

    # Step 3 — join centrality values to log rows via template_id
    df = sessionized_logs[
        ["sequence_number", "session_id", "template_id"]
    ].copy()

    df["centrality_score"] = df["template_id"].map(
        lambda t: score_lookup[t]["centrality_score"]
    )
    df["degree"] = df["template_id"].map(
        lambda t: score_lookup[t]["degree"]
    )
    df["betweenness"] = df["template_id"].map(
        lambda t: score_lookup[t]["betweenness"]
    )
    df["in_graph"] = df["template_id"].map(
        lambda t: score_lookup[t]["in_graph"]
    )
    df["cluster_id"] = df["template_id"].map(
        lambda t: score_lookup[t]["cluster_id"]
    )

    # Step 4 — correlated_log_ids per row
    # For each row: find other logs in the same session whose template is a
    # graph-neighbour of this row's template; return their sequence_numbers
    # as strings. Empty list for capped templates or templates with no edges.
    session_tmpl_to_seqnums: dict = (
        df.groupby(["session_id", "template_id"])["sequence_number"]
        .apply(list)
        .to_dict()
    )

    def _correlated(row: pd.Series) -> list:
        if not score_lookup[row["template_id"]]["in_graph"]:
            return []
        neighbours = set(graph.neighbors(row["template_id"]))
        result: list = []
        for neighbour in neighbours:
            key = (row["session_id"], neighbour)
            result.extend(
                str(s) for s in session_tmpl_to_seqnums.get(key, [])
            )
        return result

    df["correlated_log_ids"] = df.apply(_correlated, axis=1)

    # Step 5 — in_sequence placeholder; sequence_engine updates this column
    df["in_sequence"] = False

    # Step 6 — validate, save, return
    output_cols = [
        "sequence_number", "centrality_score", "degree", "betweenness",
        "in_graph", "cluster_id", "in_sequence", "correlated_log_ids",
    ]
    graph_scores_df = df[output_cols].reset_index(drop=True)

    validate_schema(graph_scores_df, output_cols)
    save_parquet(graph_scores_df, cfg.GRAPH_SCORES_PATH)

    in_graph_rate = graph_scores_df["in_graph"].mean() * 100
    cluster_count = graph_scores_df["cluster_id"].nunique()
    mean_centrality = graph_scores_df["centrality_score"].mean()
    print(
        f"graph_scores_df: shape={graph_scores_df.shape}, "
        f"in_graph={in_graph_rate:.1f}%, "
        f"clusters={cluster_count}, "
        f"mean_centrality={mean_centrality:.4f}"
    )

    return graph_scores_df
