"""
dashboard/components/graph_view.py
====================================
pyvis network graph wrapper for the Incident Detail page.
Renders a correlation graph highlighting root cause nodes and anomalous templates.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


_GRAPH_JSON_PATH = "data/processed/correlation_graph.json"

_COLOUR = {
    "root_cause":  "#DC2626",  # red
    "anomalous":   "#F59E0B",  # amber
    "normal":      "#3B82F6",  # blue
    "grey":        "#94A3B8",  # slate — not in incident
}


def render_graph(
    correlation_id: str,
    incident_logs: pd.DataFrame,
    root_cause_ids: list[str],
    max_nodes: int = 30,
) -> None:
    """
    Render a pyvis force-directed graph for the given incident.

    Parameters
    ----------
    correlation_id  : used to derive a unique /tmp path for the HTML file
    incident_logs   : DataFrame with at least template_id and label columns
    root_cause_ids  : list of template_id or log_id strings marking root causes
    max_nodes       : if the incident has more than this many templates, show
                      only the top-N by centrality_score (performance guard)
    """
    try:
        from pyvis.network import Network
    except ImportError:
        st.warning("pyvis not installed. Run: pip install pyvis")
        return

    # Load correlation graph JSON
    graph_path = Path(_GRAPH_JSON_PATH)
    if not graph_path.exists():
        st.info("Correlation graph not found at data/processed/correlation_graph.json. "
                "Run the pipeline with graph correlation enabled.")
        return

    try:
        graph_data = json.loads(graph_path.read_text())
    except Exception as exc:
        st.error(f"Failed to parse correlation_graph.json: {exc}")
        return

    if incident_logs.empty or "template_id" not in incident_logs.columns:
        st.info("No template data available for this incident.")
        return

    incident_templates: set[str] = set(incident_logs["template_id"].dropna().unique())

    # Determine anomalous templates from labels
    anomalous_templates: set[str] = set()
    if "label" in incident_logs.columns:
        anom_df = incident_logs[incident_logs["label"].isin(["critical", "medium"])]
        anomalous_templates = set(anom_df["template_id"].dropna().unique())

    # Filter and optionally limit nodes
    all_nodes = graph_data.get("nodes", [])
    incident_nodes = [n for n in all_nodes if n.get("id") in incident_templates]

    if len(incident_nodes) > max_nodes:
        incident_nodes = sorted(
            incident_nodes,
            key=lambda n: n.get("centrality_score", 0),
            reverse=True,
        )[:max_nodes]
        visible_ids = {n["id"] for n in incident_nodes}
    else:
        visible_ids = {n["id"] for n in incident_nodes}

    if not incident_nodes:
        st.info("No graph nodes match this incident's templates.")
        return

    # Build pyvis network
    net = Network(
        height="380px",
        width="100%",
        bgcolor="#0f172a",
        font_color="#e2e8f0",
        directed=False,
    )
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -3000,
          "springLength": 80,
          "springConstant": 0.04
        }
      },
      "nodes": {
        "font": { "size": 11, "face": "IBM Plex Mono" }
      },
      "edges": {
        "smooth": { "type": "curvedCW", "roundness": 0.1 },
        "color": { "inherit": false, "color": "#334155" }
      }
    }
    """)

    for node in incident_nodes:
        nid = node["id"]
        is_rc = nid in root_cause_ids
        is_anom = nid in anomalous_templates
        centrality = node.get("centrality_score", 0)

        colour = _COLOUR["root_cause"] if is_rc else (
            _COLOUR["anomalous"] if is_anom else _COLOUR["normal"]
        )
        size = 14 + centrality * 28

        border = "#ffffff" if is_rc else colour
        title = (
            f"<b>{nid}</b><br>"
            f"Centrality: {centrality:.3f}<br>"
            f"{'🔴 ROOT CAUSE<br>' if is_rc else ''}"
            f"{'⚠️ Anomalous<br>' if is_anom else ''}"
        )

        net.add_node(
            nid,
            label=nid,
            color={
                "background": colour,
                "border": border,
                "highlight": {"background": "#F59E0B", "border": "#fff"},
            },
            size=size,
            title=title,
            borderWidth=3 if is_rc else 1,
        )

    # Add edges between visible nodes
    for edge in graph_data.get("edges", []):
        src, tgt = edge.get("source"), edge.get("target")
        if src in visible_ids and tgt in visible_ids:
            weight = edge.get("weight", 1)
            net.add_edge(src, tgt, value=max(1, weight), title=f"co-occurrence: {weight}")

    # Render
    tmp_path = f"/tmp/graph_{correlation_id.replace('/', '_')}.html"
    try:
        net.save_graph(tmp_path)
        html_content = Path(tmp_path).read_text()
        components.html(html_content, height=390, scrolling=False)
    except Exception as exc:
        st.error(f"Graph render failed: {exc}")
