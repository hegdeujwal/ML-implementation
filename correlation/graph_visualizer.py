"""
correlation/graph_visualizer.py

Export the correlation graph as JSON for downstream visualization.

Output format (correlation_graph.json)
---------------------------------------
{
  "nodes": [
    {
      "id": "IF_DOWN",
      "cluster_id": "C0001",
      "degree": 5,
      "centrality_score": 0.42
    },
    ...
  ],
  "edges": [
    {
      "source": "IF_DOWN",
      "target": "BGP_PEER_RESET",
      "weight": 0.15,
      "pmi": 0.83,
      "cooccurrence_count": 12
    },
    ...
  ],
  "metadata": {
    "n_nodes": int,
    "n_edges": int,
    "n_components": int,
    "max_nodes_cap": GRAPH_MAX_NODES,
    "cooccurrence_window_seconds": GRAPH_COOCCURRENCE_WINDOW_SECONDS
  }
}

Public API
----------
export_graph_json(graph, output_path) -> str
export_graph_png(graph, output_path)  -> None  (stub — not yet implemented)
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

import common.config as cfg
from common.logger import get_logger

logger = get_logger(__name__)


def export_graph_json(
    graph: nx.Graph,
    output_path: str = "data/processed/correlation_graph.json",
) -> str:
    """Export the correlation graph as JSON.

    Parameters
    ----------
    graph : nx.Graph
        Output of graph_builder.build_graph() or load_or_build_graph().
        Node attributes used: cluster_id, centrality_score (optional).
        Edge attributes used: weight, pmi, cooccurrence_count.
    output_path : str
        Destination file path.

    Returns
    -------
    str
        The path the file was written to.
    """
    nodes_list = []
    for node_id in graph.nodes:
        attrs = graph.nodes[node_id]
        nodes_list.append({
            "id": node_id,
            "cluster_id": attrs.get("cluster_id", ""),
            "degree": graph.degree[node_id],
            "centrality_score": round(float(attrs.get("centrality_score", 0.0)), 6),
        })

    edges_list = []
    for u, v, attrs in graph.edges(data=True):
        edges_list.append({
            "source": u,
            "target": v,
            "weight": round(float(attrs.get("weight", 0.0)), 6),
            "pmi": round(float(attrs.get("pmi", 0.0)), 6),
            "cooccurrence_count": int(attrs.get("cooccurrence_count", 0)),
        })

    n_components = nx.number_connected_components(graph)

    payload = {
        "nodes": nodes_list,
        "edges": edges_list,
        "metadata": {
            "n_nodes": graph.number_of_nodes(),
            "n_edges": graph.number_of_edges(),
            "n_components": n_components,
            "max_nodes_cap": cfg.GRAPH_MAX_NODES,
            "cooccurrence_window_seconds": cfg.GRAPH_COOCCURRENCE_WINDOW_SECONDS,
        },
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    logger.info("Graph exported to %s", output_path)
    return output_path


def export_graph_png(graph: nx.Graph, output_path: str) -> None:
    # TODO: implement matplotlib/graphviz PNG export for demo day
    logger.info("PNG export not yet implemented — skipped")
    return None
