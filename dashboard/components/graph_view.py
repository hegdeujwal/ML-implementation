"""
dashboard/components/graph_view.py
====================================
pyvis network graph wrapper for the Incident Detail page.
Renders a correlation graph highlighting root cause nodes and anomalous templates.

Fixes applied:
  1. Height increased from 380px → 600px so nodes have room to breathe.
  2. Physics tuned: gravitationalConstant -8000, springLength 220, springConstant 0.02,
     damping 0.18 — spreads nodes far apart and stabilises faster.
  3. Node labels shortened to max 18 chars with "…" truncation; full ID shown on hover.
  4. Font size raised to 13px; node sizes capped for readability.
  5. Edge colour set to semi-transparent slate so it doesn't dominate.
  6. "stabilization" options added so the graph settles quickly on load.
  7. Hierarchical layout disabled (was implicitly on for some pyvis versions).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


_GRAPH_JSON_PATH = "data/processed/correlation_graph.json"

_COLOUR = {
    "root_cause": "#DC2626",  # red
    "anomalous":  "#F59E0B",  # amber
    "normal":     "#3B82F6",  # blue
    "grey":       "#94A3B8",  # slate — not in incident
}

_MAX_LABEL_LEN = 18  # characters before truncation with "…"


def _short_label(template_id: str) -> str:
    """
    Return a display-friendly label for a template_id node.

    OSPF_ADJACENCY_ON_CHANGED_STATE  →  "Ospf Adjacency On…"
    CPU_UTILIZATION_EXCEEDED         →  "Cpu Utilization E…"
    """
    readable = template_id.replace("_", " ").title()
    if len(readable) > _MAX_LABEL_LEN:
        return readable[: _MAX_LABEL_LEN - 1] + "…"
    return readable


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

    # ── Load correlation graph JSON ─────────────────────────────────────────
    graph_path = Path(_GRAPH_JSON_PATH)
    if not graph_path.exists():
        st.info(
            "Correlation graph not found at data/processed/correlation_graph.json. "
            "Run the pipeline with graph correlation enabled."
        )
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

    # ── Filter nodes ────────────────────────────────────────────────────────
    all_nodes = graph_data.get("nodes", [])
    incident_nodes = [n for n in all_nodes if n.get("id") in incident_templates]

    if len(incident_nodes) > max_nodes:
        incident_nodes = sorted(
            incident_nodes,
            key=lambda n: n.get("centrality_score", 0),
            reverse=True,
        )[:max_nodes]

    visible_ids = {n["id"] for n in incident_nodes}

    if not incident_nodes:
        st.info("No graph nodes match this incident's templates.")
        return

    # ── Build pyvis network ─────────────────────────────────────────────────
    # FIX 1: height 600px gives nodes 60% more vertical room to spread out.
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#0f172a",
        font_color="#e2e8f0",
        directed=False,
    )

    # FIX 2: physics — much stronger repulsion + longer spring so 20+ nodes
    #         don't pile on top of each other.
    # FIX 6: stabilization iterations so the layout settles before the user
    #         sees it (avoids the "spinning balls" first impression).
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.25,
          "springLength": 220,
          "springConstant": 0.02,
          "damping": 0.18,
          "avoidOverlap": 0.6
        },
        "stabilization": {
          "enabled": true,
          "iterations": 300,
          "updateInterval": 25,
          "fit": true
        },
        "minVelocity": 0.75
      },
      "nodes": {
        "font": {
          "size": 13,
          "face": "IBM Plex Mono, monospace",
          "strokeWidth": 3,
          "strokeColor": "#0f172a"
        },
        "shape": "dot",
        "scaling": { "min": 12, "max": 36 }
      },
      "edges": {
        "smooth": { "type": "curvedCW", "roundness": 0.15 },
        "color": { "inherit": false, "color": "#334155", "opacity": 0.6 },
        "width": 1.5
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": false,
        "zoomView": true
      }
    }
    """)

    for node in incident_nodes:
        nid        = node["id"]
        is_rc      = nid in root_cause_ids
        is_anom    = nid in anomalous_templates
        centrality = node.get("centrality_score", 0)

        colour = (
            _COLOUR["root_cause"] if is_rc else
            _COLOUR["anomalous"]  if is_anom else
            _COLOUR["normal"]
        )

        # FIX 4: cap size so high-centrality nodes don't swallow their label.
        #         Range: 14 (min) → 36 (max).
        size   = min(14 + centrality * 28, 36)
        border = "#ffffff" if is_rc else colour

        # FIX 3: short label on the node canvas; full ID in the hover tooltip.
        # TOOLTIP FIX: pyvis passes `title` directly to vis.js which renders it
        # as an HTML div. Single-quoted inline styles (style='...') inside an
        # HTML attribute break the browser's parser and the raw string is shown
        # instead of rendered HTML. Solution: use a <div> with no inline styles
        # and rely on vis.js's own tooltip container for styling, OR — simplest
        # and most reliable — use plain text only (no HTML tags at all).
        # Plain text is always safe; vis.js wraps it in a <div> automatically.
        short  = _short_label(nid)
        status_parts = []
        if is_rc:   status_parts.append("ROOT CAUSE")
        if is_anom: status_parts.append("Anomalous")
        status_str = "  |  " + "  |  ".join(status_parts) if status_parts else ""
        tooltip = f"{nid}\nCentrality: {centrality:.3f}{status_str}"

        net.add_node(
            nid,
            label=short,          # FIX 3: truncated readable label
            title=tooltip,        # full ID + stats on hover
            color={
                "background":  colour,
                "border":      border,
                "highlight":   {"background": "#FBBF24", "border": "#ffffff"},
                "hover":       {"background": "#FBBF24", "border": "#ffffff"},
            },
            size=size,
            borderWidth=3 if is_rc else 1,
            shadow={"enabled": True, "color": "rgba(0,0,0,0.5)", "size": 8},
        )

    # ── Add edges between visible nodes ─────────────────────────────────────
    # FIX 5: edge opacity handled in set_options; weight capped at 8 so thick
    #         edges don't obscure node labels.
    for edge in graph_data.get("edges", []):
        src, tgt = edge.get("source"), edge.get("target")
        if src in visible_ids and tgt in visible_ids:
            weight = edge.get("weight", 1)
            net.add_edge(
                src, tgt,
                value=min(max(1, weight), 8),
                title=f"co-occurrence: {weight}",   # plain text — no HTML
            )

    # ── Render ──────────────────────────────────────────────────────────────
    try:
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tf:
            tmp_path = tf.name

        net.save_graph(tmp_path)
        html_content = Path(tmp_path).read_text(encoding="utf-8")

        # Patch 1: dark background fills the iframe without a white flash.
        html_content = html_content.replace(
            "<body>",
            "<body style='background:#0f172a; margin:0; padding:0;'>",
        )

        # Patch 2: style the vis.js tooltip <div class="vis-tooltip"> so it
        # looks polished instead of the default plain white box.
        # Injected just before </head> so it overrides vis.js defaults.
        tooltip_css = """
        <style>
        div.vis-tooltip {
            background: #1e293b !important;
            border: 1px solid #475569 !important;
            border-radius: 6px !important;
            color: #e2e8f0 !important;
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 12px !important;
            padding: 6px 10px !important;
            white-space: pre !important;   /* preserve \n line breaks */
            box-shadow: 0 4px 12px rgba(0,0,0,0.5) !important;
            max-width: 320px !important;
            pointer-events: none !important;
        }
        </style>
        """
        html_content = html_content.replace("</head>", tooltip_css + "</head>")

        components.html(html_content, height=615, scrolling=False)

        try:
            os.remove(tmp_path)
        except Exception:
            pass

    except Exception as exc:
        st.error(f"Graph render failed: {exc}")