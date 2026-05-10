"""
correlation/graph_visualizer.py

Export the correlation graph as a JSON document for downstream visualization.

Output format (correlation_graph.json)
---------------------------------------
{
  "nodes": [
    {
      "id": "IF_DOWN",
      "template": "IF_DOWN",
      "node_type": "log_template",
      "centrality_score": 0.723
    },
    ...
  ],
  "edges": [
    {
      "source": "IF_DOWN",
      "target": "BGP_PEER_RESET",
      "weight": 0.85
    },
    ...
  ]
}

PNG export is out of scope for Phase 3 (requires matplotlib + layout engine
which adds significant dependency weight).  The JSON export is the primary
deliverable; a separate visualization layer can consume it.

Public API
----------
export_graph_json(g, centrality_df, output_path)
    Write the JSON file and return the dict that was written.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import pandas as pd

import common.config as cfg
from correlation.graph_builder import CorrelationGraph


def export_graph_json(
    g: CorrelationGraph,
    centrality_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> dict:
    """Export the correlation graph as JSON.

    Parameters
    ----------
    g : CorrelationGraph
        The built correlation graph.
    centrality_df : pd.DataFrame
        Output of centrality.compute_centrality().  Provides centrality_score
        per node.  Must have columns: node_id, centrality_score.
    output_path : str, optional
        Destination file path.  Defaults to cfg.GRAPH_JSON_PATH.

    Returns
    -------
    dict
        The dict that was serialized to JSON (nodes + edges lists).
    """
    if output_path is None:
        output_path = cfg.GRAPH_JSON_PATH

    # Build a fast lookup: node_id -> centrality_score
    score_lookup: dict[str, float] = {}
    if not centrality_df.empty:
        score_lookup = (
            centrality_df.set_index("node_id")["centrality_score"]
            .to_dict()
        )

    nodes_list: list[dict] = []
    for node_id, node in g.nodes.items():
        nodes_list.append({
            "id": node_id,
            "template": node_id,
            "node_type": node.node_type,
            "centrality_score": round(score_lookup.get(node_id, 0.0), 6),
        })

    edges_list: list[dict] = []
    for (src, tgt), edge in g.edges.items():
        edges_list.append({
            "source": src,
            "target": tgt,
            "weight": round(edge.weight, 6),
        })

    payload = {"nodes": nodes_list, "edges": edges_list}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    return payload
