"""
correlation/graph_builder.py

Build a weighted undirected co-occurrence graph from sessionized log events.

Co-occurrence is a within-session concept: edges only form between templates
that appear in the same session within GRAPH_COOCCURRENCE_WINDOW_SECONDS.
A single global graph is assembled by merging per-session subgraphs.

Graph schema
------------
Node attributes
    frequency  : int   -- total occurrence count across all sessions
    cluster_id : str   -- connected component label "C0000", "C0001", ...
                          largest component = C0000

Edge attributes
    weight             : float -- session co-occurrence rate (sessions_together / n_sessions)
    pmi                : float -- positive pointwise mutual information (>= 0)
    cooccurrence_count : int   -- number of sessions where both templates co-occur

graph.graph["included_templates"] stores the set of template_ids that passed
the GRAPH_MAX_NODES cap; all other templates are excluded from graph construction.

Persistence
-----------
Python pickle is used (nx.write_gpickle was removed in NetworkX 3.x).
Cache path: data/processed/correlation_graph.pkl
"""

from __future__ import annotations

import math
import pickle
from collections import Counter
from pathlib import Path

import networkx as nx
import pandas as pd

import common.config as cfg
from common.logger import get_logger

logger = get_logger(__name__)

# Use the canonical path from config so all modules share the same cache file.
_GRAPH_PICKLE_PATH = cfg.GRAPH_PICKLE_PATH


def build_graph(sessionized_logs: pd.DataFrame) -> nx.Graph:
    """Build a weighted undirected co-occurrence graph from sessionized logs.

    Parameters
    ----------
    sessionized_logs : pd.DataFrame
        Must contain: sequence_number, session_id, template_id,
        timestamp, frequency.

    Returns
    -------
    nx.Graph
        Undirected weighted graph. Node attributes: frequency, cluster_id.
        Edge attributes: weight, pmi, cooccurrence_count.
    """
    # Step 1 — filter to top GRAPH_MAX_NODES templates by total frequency
    template_freq = sessionized_logs.groupby("template_id")["frequency"].sum()
    top = template_freq.nlargest(cfg.GRAPH_MAX_NODES)
    included_templates = set(top.index)

    working_df = sessionized_logs[
        sessionized_logs["template_id"].isin(included_templates)
    ].copy()

    if pd.api.types.is_datetime64_any_dtype(working_df["timestamp"]):
        working_df["timestamp"] = working_df["timestamp"].astype("int64") / 1e9
    else:
        working_df["timestamp"] = working_df["timestamp"].astype(float)

    # Step 2 — per-session co-occurrence counting
    # session_cooccurrence[(t_a, t_b)] = number of sessions where both appear
    # Keys are canonical: (min, max) lexicographically so each pair is unique
    session_cooccurrence: Counter = Counter()
    n_sessions = int(sessionized_logs["session_id"].nunique())

    for _, session_df in working_df.groupby("session_id"):
        session_df = session_df.sort_values("timestamp")
        rows = list(session_df[["timestamp", "template_id"]].itertuples(index=False))
        n = len(rows)

        seen_pairs: set = set()
        for i in range(n):
            ts_i, tmpl_i = rows[i]
            for j in range(i + 1, n):
                ts_j, tmpl_j = rows[j]
                if ts_j - ts_i > cfg.GRAPH_COOCCURRENCE_WINDOW_SECONDS:
                    break  # sorted — no further j can be within window
                if tmpl_i == tmpl_j:
                    continue
                pair = (tmpl_i, tmpl_j) if tmpl_i <= tmpl_j else (tmpl_j, tmpl_i)
                seen_pairs.add(pair)

        # Each unique pair counts once per session regardless of how many
        # times the two templates co-occurred within the session window
        for pair in seen_pairs:
            session_cooccurrence[pair] += 1

    # Step 3 — build nx.Graph with PMI-weighted edges
    graph = nx.Graph()
    graph.graph["included_templates"] = included_templates

    # Add all included templates as nodes (even isolated ones)
    for template in included_templates:
        graph.add_node(template, frequency=int(template_freq[template]))

    for (t_a, t_b), count in session_cooccurrence.items():
        freq_a = int(template_freq[t_a])
        freq_b = int(template_freq[t_b])
        session_weight = count / n_sessions
        # Smoothed log-PMI: log((P(A,B) * N) / (freq_A * freq_B) + epsilon)
        pmi_raw = math.log(
            (count * n_sessions) / (freq_a * freq_b) + 1e-10
        )
        positive_pmi = max(0.0, pmi_raw)
        graph.add_edge(
            t_a, t_b,
            weight=session_weight,
            pmi=positive_pmi,
            cooccurrence_count=count,
        )

    # Step 4 — connected components → deterministic cluster_id
    # Sort components by size descending: largest = C0000
    components = sorted(
        nx.connected_components(graph), key=len, reverse=True
    )
    for i, component in enumerate(components):
        cluster_id = f"C{i:04d}"
        for template in component:
            graph.nodes[template]["cluster_id"] = cluster_id

    # Step 5 — persist (path comes from cfg.GRAPH_PICKLE_PATH via module-level constant)
    path = Path(_GRAPH_PICKLE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(graph, fh, protocol=pickle.HIGHEST_PROTOCOL)

    n_components = nx.number_connected_components(graph)
    logger.info(
        "Graph built: %d nodes, %d edges, %d connected components",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        n_components,
    )
    return graph


def load_or_build_graph(sessionized_logs: pd.DataFrame) -> nx.Graph:
    """Return cached graph if available, otherwise build and cache it.

    Parameters
    ----------
    sessionized_logs : pd.DataFrame
        Passed to build_graph() on a cache miss.

    Returns
    -------
    nx.Graph
    """
    path = Path(_GRAPH_PICKLE_PATH)
    if path.exists():
        with open(path, "rb") as fh:
            graph = pickle.load(fh)
        logger.info(
            "Loaded graph from cache (%d nodes)", graph.number_of_nodes()
        )
        return graph
    return build_graph(sessionized_logs)
