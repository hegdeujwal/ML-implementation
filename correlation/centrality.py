"""
correlation/centrality.py

Compute per-node centrality scores for the correlation graph and assemble
the per-log-row output DataFrame handed to Phase 4 (Ujwal / scorer).

Public API
----------
compute_centrality(g)
    Accept a CorrelationGraph, convert it to a NetworkX graph, compute three
    centrality metrics, normalize, and return a node-level DataFrame.

build_graph_scores_df(centrality_df, raw_df, g, sequence_log_ids)
    Join centrality scores back to individual log rows, add cluster_id,
    in_sequence, and correlated_log_ids, then return the P4 output DataFrame.

Centrality metrics
------------------
degree_centrality
    NetworkX degree_centrality: deg(v) / (N-1).  Already in [0,1].

betweenness_centrality
    Fraction of shortest paths passing through each node.  Exact computation
    is O(V * E) — impractical for graphs > 200 nodes.  When
    len(graph) > BETWEENNESS_LARGE_GRAPH_THRESHOLD we use the Brandes
    k-approximation (k=BETWEENNESS_K=50 random pivot nodes).
    Reference: Brandes, 2001; Bader & Madduri, 2006.

pagerank_score  (used as centrality_score)
    Stationary distribution of a random walk with teleportation.
    alpha=PAGERANK_ALPHA=0.85 (industry standard; Google original value).
    Edge weights are used as transition probabilities, so highly co-occurring
    pairs contribute more to a node's PageRank.

Normalization
-------------
NetworkX already returns degree_centrality and pagerank in [0,1] for
connected graphs.  Betweenness_centrality is in [0,1] by default from
NetworkX (it divides by the number of possible pairs).  An explicit
min-max normalization pass is applied to all three to guarantee the
contract even for degenerate graphs (single node, isolated nodes, etc.).
"""

from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

import common.config as cfg
from correlation.graph_builder import CorrelationGraph, correlation_graph_to_nx


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _minmax_normalize(series: pd.Series) -> pd.Series:
    """Normalize a Series to [0, 1]; returns all-zeros if range is 0."""
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-12:
        return pd.Series(0.0, index=series.index)
    return (series - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Public: compute_centrality
# ---------------------------------------------------------------------------

def compute_centrality(g: CorrelationGraph) -> pd.DataFrame:
    """Compute degree, betweenness, and PageRank centrality for every node.

    Parameters
    ----------
    g : CorrelationGraph
        The built correlation graph (output of build_graph or
        build_graph_from_parquet).

    Returns
    -------
    pd.DataFrame
        One row per node.  Columns:
            node_id            : str
            degree_centrality  : float in [0, 1]
            betweenness        : float in [0, 1]
            pagerank_score     : float in [0, 1]
            centrality_score   : float in [0, 1]   (alias of pagerank_score)
    """
    nx_graph = correlation_graph_to_nx(g)

    n_nodes = nx_graph.number_of_nodes()

    if n_nodes == 0:
        return pd.DataFrame(columns=[
            "node_id", "degree_centrality", "betweenness",
            "pagerank_score", "centrality_score",
        ])

    # --- Degree centrality ---------------------------------------------------
    deg = nx.degree_centrality(nx_graph)

    # --- Betweenness centrality ----------------------------------------------
    # Use k-approximation when graph is large to avoid O(V*E) cost.
    # k=BETWEENNESS_K (default 50) means 50 random source nodes are sampled;
    # results are unbiased estimators of the true betweenness.
    if n_nodes > cfg.BETWEENNESS_LARGE_GRAPH_THRESHOLD:
        # k cannot exceed the graph size; cap it to avoid ValueError.
        effective_k = min(cfg.BETWEENNESS_K, n_nodes)
        bet = nx.betweenness_centrality(
            nx_graph,
            k=effective_k,
            weight="weight",
            normalized=True,
        )
    else:
        bet = nx.betweenness_centrality(
            nx_graph,
            weight="weight",
            normalized=True,
        )

    # --- PageRank ------------------------------------------------------------
    # alpha=PAGERANK_ALPHA=0.85 (damping factor; probability of following an
    # edge vs. teleporting to a random node).
    # Weight parameter tells NetworkX to use edge weight as transition
    # probability; heavier co-occurrence edges carry more flow.
    pr = nx.pagerank(nx_graph, alpha=cfg.PAGERANK_ALPHA, weight="weight")

    # --- Assemble and normalize ----------------------------------------------
    nodes = list(nx_graph.nodes())
    df = pd.DataFrame({
        "node_id": nodes,
        "degree_centrality": [deg[n] for n in nodes],
        "betweenness": [bet[n] for n in nodes],
        "pagerank_score": [pr[n] for n in nodes],
    })

    df["degree_centrality"] = _minmax_normalize(df["degree_centrality"])
    df["betweenness"] = _minmax_normalize(df["betweenness"])
    df["pagerank_score"] = _minmax_normalize(df["pagerank_score"])

    # Primary centrality score = PageRank
    df["centrality_score"] = df["pagerank_score"]

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public: build_graph_scores_df
# ---------------------------------------------------------------------------

def build_graph_scores_df(
    centrality_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    g: CorrelationGraph,
    sequence_log_ids: Optional[set] = None,
) -> pd.DataFrame:
    """Assemble the per-log-row DataFrame for P4 consumption.

    Parameters
    ----------
    centrality_df : pd.DataFrame
        Output of compute_centrality().
    raw_df : pd.DataFrame
        The original sessionized log DataFrame.  Must contain columns:
        log_id, session_id, template_id.  timestamp is optional.
    g : CorrelationGraph
        Used to derive connected-component cluster IDs.
    sequence_log_ids : set of str, optional
        Set of log_ids that are part of at least one detected sequence
        (output of sequence_engine.detect_sequences).  Defaults to empty set.

    Returns
    -------
    pd.DataFrame
        One row per log_id.  Columns:
            log_id             : str
            centrality_score   : float in [0, 1]
            degree             : int      (raw degree count in nx graph)
            betweenness        : float in [0, 1]
            cluster_id         : str      (connected-component label "cc_<int>")
            in_sequence        : bool
            correlated_log_ids : list[str]
    """
    if sequence_log_ids is None:
        sequence_log_ids = set()

    nx_graph = correlation_graph_to_nx(g)

    # --- Connected-component cluster IDs ------------------------------------
    # Assign each node an integer component label, zero-indexed.
    # We cast to string ("cc_0", "cc_1", ...) to future-proof against
    # DBSCAN labels that P4 will add later.
    component_map: dict[str, str] = {}
    for comp_idx, component in enumerate(nx.connected_components(nx_graph)):
        label = f"cc_{comp_idx}"
        for node in component:
            component_map[node] = label

    # Nodes that were excluded by max_nodes cap won't be in the nx graph.
    # Map them to "cc_unknown" so we still have a value for every log row.
    component_map_with_fallback = lambda node: component_map.get(node, "cc_unknown")

    # --- Raw degree per node -------------------------------------------------
    degree_map: dict[str, int] = dict(nx_graph.degree())

    # --- Build template -> centrality_score lookup --------------------------
    score_lookup = centrality_df.set_index("node_id")[
        ["centrality_score", "betweenness"]
    ].to_dict("index")

    # --- Correlated log_ids: per-session template co-occurrence -------------
    # For each log_id, find other log_ids in the same session that share any
    # edge in the graph with the current log's template.
    # This is a best-effort approximation; exact co-occurrence is tracked
    # per-window in graph_builder but is not back-propagated to log_ids.
    template_to_neighbors: dict[str, set] = {}
    for node in nx_graph.nodes():
        template_to_neighbors[node] = set(nx_graph.neighbors(node))

    # Group log_ids by (session_id, template_id) for fast lookup
    session_template_to_log_ids: dict[tuple, list] = (
        raw_df.groupby(["session_id", "template_id"])["log_id"]
        .apply(list)
        .to_dict()
    )

    def _correlated_log_ids(row: pd.Series) -> list:
        """Return log_ids in the same session whose templates are neighbors."""
        neighbors = template_to_neighbors.get(row["template_id"], set())
        result: list[str] = []
        for neighbor_template in neighbors:
            key = (row["session_id"], neighbor_template)
            result.extend(session_template_to_log_ids.get(key, []))
        return result

    # --- Assemble output DataFrame ------------------------------------------
    df_out = raw_df[["log_id", "session_id", "template_id"]].copy()

    df_out["centrality_score"] = df_out["template_id"].apply(
        lambda t: score_lookup.get(t, {}).get("centrality_score", 0.0)
    )
    df_out["degree"] = df_out["template_id"].apply(
        lambda t: degree_map.get(t, 0)
    )
    df_out["betweenness"] = df_out["template_id"].apply(
        lambda t: score_lookup.get(t, {}).get("betweenness", 0.0)
    )
    df_out["cluster_id"] = df_out["template_id"].apply(
        component_map_with_fallback
    )
    df_out["in_sequence"] = df_out["log_id"].isin(sequence_log_ids)
    df_out["correlated_log_ids"] = df_out.apply(_correlated_log_ids, axis=1)

    # Drop intermediate columns; keep only P4 contract columns
    df_out = df_out[[
        "log_id",
        "centrality_score",
        "degree",
        "betweenness",
        "cluster_id",
        "in_sequence",
        "correlated_log_ids",
    ]]

    return df_out.reset_index(drop=True)
