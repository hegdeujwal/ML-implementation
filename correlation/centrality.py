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
    Distance = 1/weight (co-occurrence weight is affinity; NetworkX expects
    path cost). Exact below BETWEENNESS_LARGE_GRAPH_THRESHOLD nodes, k-pivot
    approximation (k=BETWEENNESS_K) above. normalised=True divides by
    possible path count.

pagerank  (used as centrality_score)
    Weighted by PMI (frequency-corrected association) so centrality means
    "structurally implicated", not "frequent". alpha=GRAPH_PAGERANK_ALPHA.
    Min-max normalised to [0,1]; degenerate case (single node or all-equal
    scores) → 0.5 instead of 0.0.

Capped templates (outside GRAPH_MAX_NODES)
    centrality_score = global mean of in-graph PageRank values
    betweenness      = global mean of in-graph betweenness values
    degree           = 0
    in_graph         = False
    cluster_id       = "UNCAPPED"
"""

from __future__ import annotations

import gc
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
        # NetworkX interprets betweenness edge weight as *distance* (path
        # cost), so strongly co-occurring templates must be CLOSE, not far:
        # use 1/weight. Passing weight="weight" directly inverts the metric.
        for _u, _v, _d in graph.edges(data=True):
            _d["distance"] = 1.0 / (_d.get("weight", 0.0) + 1e-10)

        # Exact betweenness below the size threshold, k-pivot approximation
        # above it (exact is O(V·E); see config).
        _bw_k = (
            None
            if len(graph) <= cfg.BETWEENNESS_LARGE_GRAPH_THRESHOLD
            else min(cfg.BETWEENNESS_K, len(graph))
        )
        bw_raw = nx.betweenness_centrality(
            graph,
            weight="distance",
            normalized=True,
            k=_bw_k,
        )

        # PageRank weighted by PMI: frequency-corrected association, so hub
        # status means "structurally implicated", not merely "common chatter".
        # Raw co-occurrence weight made the most frequent routine templates
        # the most central. Fall back to raw weight when PMI is degenerate
        # (all zero — possible on tiny graphs).
        _has_pmi = any(d.get("pmi", 0.0) > 0.0 for _, _, d in graph.edges(data=True))
        pr_raw = nx.pagerank(
            graph,
            alpha=cfg.GRAPH_PAGERANK_ALPHA,
            weight="pmi" if _has_pmi else "weight",
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
    # All rows with the same (session_id, template_id) get the same result,
    # so compute at that level (~session*template entries) instead of per-row
    # (35K+ Series allocations via apply). Much lower peak memory.
    session_tmpl_to_seqnums: dict = (
        df.groupby(["session_id", "template_id"])["sequence_number"]
        .apply(list)
        .to_dict()
    )

    # Pre-compute neighbor sets once per in-graph template
    _neighbor_cache: dict = {}
    for tid in included_templates:
        if graph.has_node(tid):
            _neighbor_cache[tid] = set(graph.neighbors(tid))

    # Build lookup at (session_id, template_id) level
    # Cap at 20 entries per list — this column is explainability metadata,
    # not used in scoring. Uncapped lists cause multi-GB memory bloat.
    _MAX_CORRELATED = 20
    _corr_lookup: dict = {}
    _empty_list: list = []
    for (sid, tid), _seqnums in session_tmpl_to_seqnums.items():
        if tid not in _neighbor_cache:
            _corr_lookup[(sid, tid)] = _empty_list
            continue
        result: list = []
        for nbr in _neighbor_cache[tid]:
            result.extend(
                str(s) for s in session_tmpl_to_seqnums.get((sid, nbr), [])
            )
            if len(result) >= _MAX_CORRELATED:
                result = result[:_MAX_CORRELATED]
                break
        _corr_lookup[(sid, tid)] = result

    # Map back to rows via tuple key (vectorized dict lookup, no apply)
    _keys = list(zip(df["session_id"], df["template_id"]))
    df["correlated_log_ids"] = [_corr_lookup.get(k, _empty_list) for k in _keys]
    del _keys, _corr_lookup, _neighbor_cache, session_tmpl_to_seqnums
    gc.collect()

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
    logger.info(
        "graph_scores_df: shape=%s, in_graph=%.1f%%, clusters=%d, mean_centrality=%.4f",
        graph_scores_df.shape,
        in_graph_rate,
        cluster_count,
        mean_centrality,
    )

    return graph_scores_df
