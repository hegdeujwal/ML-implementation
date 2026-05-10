"""
correlation/graph_builder.py

Build a weighted undirected co-occurrence graph from structured log events.

Graph schema
------------
Node attributes
    id        : str   -- unique template string or anomaly marker identifier
    node_type : str   -- "log_template" | "anomaly"
    count     : int   -- raw occurrence frequency in the input window

Edge attributes
    co_occurrences : int   -- raw count of windows in which both nodes appear together
    weight         : float -- normalized co-occurrence in [0, 1]
                             weight = co_occurrences / max_co_occurrences_across_all_edges

Normalization note
    Dividing by the single global maximum (rather than per-node) preserves the
    relative importance of edges across the whole graph, which makes downstream
    centrality and scoring logic straightforward to interpret.

Node cap
    Only the MAX_GRAPH_NODES most-frequent templates are admitted.  Anomaly
    nodes are always admitted regardless of the cap because they are injected
    explicitly by the caller.  See common/config.py for the cap value and a
    longer discussion of the memory / CPU trade-off.

Real-data integration (Phase 3)
    build_graph_from_parquet -- load sessionized_logs.parquet and call build_graph
    correlation_graph_to_nx  -- convert CorrelationGraph to a networkx.Graph
    persist_graph / load_graph -- pickle cache to skip rebuilds on repeat runs
"""

from __future__ import annotations

import itertools
import os
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import pandas as pd

import common.config as cfg


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

@dataclass
class LogEvent:
    """A single parsed log event ready for graph construction.

    Attributes
    ----------
    timestamp : float
        Unix epoch seconds.  Sub-second precision is preserved as a float.
    template  : str
        The normalized log template produced by the parsing stage (e.g. Drain
        output).  All events that share the same template map to the same node.
    is_anomaly : bool
        True when this event has been flagged by the anomaly-detection stage
        (e.g. an interface counter anomaly).  Anomaly events get an additional
        dedicated anomaly node so that they appear twice in the graph: once as
        their template node and once as an anomaly marker node.
    anomaly_label : str, optional
        Human-readable label for the anomaly node (e.g. "anomaly:if_counter").
        Ignored when is_anomaly is False.
    """
    timestamp: float
    template: str
    is_anomaly: bool = False
    anomaly_label: Optional[str] = None


@dataclass
class GraphNode:
    """A node in the correlation graph."""
    id: str
    node_type: str          # "log_template" | "anomaly"
    count: int = 0


@dataclass
class GraphEdge:
    """An edge in the correlation graph."""
    source: str
    target: str
    co_occurrences: int = 0
    weight: float = 0.0


@dataclass
class CorrelationGraph:
    """Container for the complete correlation graph.

    Attributes
    ----------
    nodes : dict mapping node id -> GraphNode
    edges : dict mapping (source_id, target_id) -> GraphEdge
        Keys are always stored with source < target (lexicographic) so each
        undirected edge has exactly one canonical representation.
    time_window_seconds : int
        The window width that was used to build this graph.
    """
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: Dict[Tuple[str, str], GraphEdge] = field(default_factory=dict)
    time_window_seconds: int = cfg.CORRELATION_TIME_WINDOW_SECONDS


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _canonical_edge_key(a: str, b: str) -> Tuple[str, str]:
    """Return (min, max) so each undirected edge has one canonical key."""
    return (a, b) if a <= b else (b, a)


def _select_top_templates(
    events: Sequence[LogEvent],
    max_nodes: int,
) -> frozenset:
    """Return the set of the *max_nodes* most frequent templates."""
    freq: Counter = Counter(e.template for e in events)
    top = freq.most_common(max_nodes)
    return frozenset(t for t, _ in top)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_graph(
    events: Sequence[LogEvent],
    time_window_seconds: Optional[int] = None,
    max_nodes: Optional[int] = None,
) -> CorrelationGraph:
    """Build a weighted co-occurrence graph from a list of log events.

    Parameters
    ----------
    events : sequence of LogEvent
        Must be sorted by timestamp (ascending).  The function does not sort
        internally to avoid hiding caller bugs, but asserts the invariant in
        debug builds.
    time_window_seconds : int, optional
        Override the global CORRELATION_TIME_WINDOW_SECONDS from config.
    max_nodes : int, optional
        Override the global MAX_GRAPH_NODES from config.

    Returns
    -------
    CorrelationGraph
        Fully populated graph with normalized edge weights.

    Algorithm
    ---------
    A two-pointer sliding window over the sorted event list is used:

        left, right = 0, 0
        while right < len(events):
            advance right
            while events[right].timestamp - events[left].timestamp > window:
                advance left
            emit all pairs (i, j) with left <= i < j == right

    This is O(N * W) where W is the average number of events per window, which
    is efficient for typical log densities.
    """
    if time_window_seconds is None:
        time_window_seconds = cfg.CORRELATION_TIME_WINDOW_SECONDS
    if max_nodes is None:
        max_nodes = cfg.MAX_GRAPH_NODES

    # Sort defensively; callers are expected to pass sorted data.
    events = sorted(events, key=lambda e: e.timestamp)

    # Select the top-N templates; anomaly nodes are always admitted.
    allowed_templates = _select_top_templates(events, max_nodes)

    graph = CorrelationGraph(time_window_seconds=time_window_seconds)

    # --- Pass 1: build nodes and count raw occurrences -------------------
    for event in events:
        # Register the template node (if within the cap).
        if event.template in allowed_templates:
            if event.template not in graph.nodes:
                graph.nodes[event.template] = GraphNode(
                    id=event.template,
                    node_type="log_template",
                )
            graph.nodes[event.template].count += 1

        # Register an anomaly node for flagged events (always admitted).
        if event.is_anomaly:
            label = event.anomaly_label or f"anomaly:{event.template}"
            if label not in graph.nodes:
                graph.nodes[label] = GraphNode(
                    id=label,
                    node_type="anomaly",
                )
            graph.nodes[label].count += 1

    # --- Pass 2: sliding-window co-occurrence counting -------------------
    # Build a flat list of (timestamp, node_id) pairs for all node references.
    # An anomalous event contributes two references: its template node and its
    # anomaly node, so both are linked to every other event in the same window.
    references: List[Tuple[float, str]] = []
    for event in events:
        if event.template in allowed_templates:
            references.append((event.timestamp, event.template))
        if event.is_anomaly:
            label = event.anomaly_label or f"anomaly:{event.template}"
            references.append((event.timestamp, label))

    # References are already timestamp-sorted because we sorted events above
    # and appended template before anomaly label for each event.

    raw_co: Dict[Tuple[str, str], int] = defaultdict(int)

    left = 0
    for right in range(len(references)):
        ts_right, node_right = references[right]
        # Shrink window from the left.
        while references[left][0] < ts_right - time_window_seconds:
            left += 1
        # Pair node_right with every other node currently in the window.
        for i in range(left, right):
            _, node_left = references[i]
            if node_left == node_right:
                # Self-loops are not meaningful for co-occurrence.
                continue
            key = _canonical_edge_key(node_left, node_right)
            raw_co[key] += 1

    # --- Pass 3: populate edges and normalize weights --------------------
    if raw_co:
        max_co = max(raw_co.values())
        for (src, tgt), count in raw_co.items():
            edge = GraphEdge(
                source=src,
                target=tgt,
                co_occurrences=count,
                weight=round(count / max_co, 6),
            )
            graph.edges[(src, tgt)] = edge
    # If raw_co is empty the graph has nodes but no edges; that is valid.

    return graph


# ---------------------------------------------------------------------------
# Real-data integration helpers (Phase 3)
# ---------------------------------------------------------------------------

def build_graph_from_parquet(
    path: str,
    time_window_seconds: Optional[int] = None,
    max_nodes: Optional[int] = None,
) -> CorrelationGraph:
    """Build a CorrelationGraph from a sessionized-log Parquet file.

    Parameters
    ----------
    path : str
        Path to the Parquet file.  Expected columns:
            log_id        : str
            session_id    : str
            timestamp     : numeric (Unix epoch seconds) or datetime
            template_id   : str  -- normalized log template
            is_anomaly    : bool, optional  (defaults to False if absent)
            anomaly_label : str, optional  (defaults to "" if absent)
    time_window_seconds : int, optional
        Override cfg.CORRELATION_TIME_WINDOW_SECONDS.
    max_nodes : int, optional
        Override cfg.MAX_GRAPH_NODES.

    Returns
    -------
    CorrelationGraph
    """
    df = pd.read_parquet(path)

    # Normalise timestamp to float epoch seconds
    if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = df["timestamp"].astype("int64") / 1e9
    else:
        df["timestamp"] = df["timestamp"].astype(float)

    has_anomaly = "is_anomaly" in df.columns
    has_label = "anomaly_label" in df.columns

    events: List[LogEvent] = []
    for row in df.itertuples(index=False):
        is_anom = bool(getattr(row, "is_anomaly", False)) if has_anomaly else False
        label = str(getattr(row, "anomaly_label", "") or "") if has_label else ""
        events.append(LogEvent(
            timestamp=row.timestamp,
            template=row.template_id,
            is_anomaly=is_anom,
            anomaly_label=label if (is_anom and label) else None,
        ))

    return build_graph(events, time_window_seconds=time_window_seconds, max_nodes=max_nodes)


def correlation_graph_to_nx(g: CorrelationGraph) -> nx.Graph:
    """Convert a CorrelationGraph to a networkx.Graph.

    Node attributes: node_type (str), count (int).
    Edge attributes: co_occurrences (int), weight (float).

    Parameters
    ----------
    g : CorrelationGraph

    Returns
    -------
    nx.Graph
        Undirected weighted graph.  The 'weight' attribute is used by
        centrality.py for weighted betweenness and PageRank.
    """
    nx_graph = nx.Graph()

    for node_id, node in g.nodes.items():
        nx_graph.add_node(
            node_id,
            node_type=node.node_type,
            count=node.count,
        )

    for (src, tgt), edge in g.edges.items():
        nx_graph.add_edge(
            src,
            tgt,
            co_occurrences=edge.co_occurrences,
            weight=edge.weight,
        )

    return nx_graph


def persist_graph(g: CorrelationGraph, path: str) -> None:
    """Pickle a CorrelationGraph to disk.

    Using Python's built-in pickle (not nx.write_gpickle, which was removed
    in NetworkX 3.x) because CorrelationGraph is a pure-Python dataclass.

    Parameters
    ----------
    g : CorrelationGraph
    path : str
        Destination file path (e.g. data/processed/correlation_graph.gpickle).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(g, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_graph(path: str) -> CorrelationGraph:
    """Load a pickled CorrelationGraph from disk.

    Parameters
    ----------
    path : str
        Path to the pickle file written by persist_graph.

    Returns
    -------
    CorrelationGraph

    Raises
    ------
    FileNotFoundError if the pickle file does not exist.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Cached graph not found at '{path}'.  "
            "Run build_graph_from_parquet() first."
        )
    with open(path, "rb") as fh:
        return pickle.load(fh)
