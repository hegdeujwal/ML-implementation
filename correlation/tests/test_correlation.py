"""
correlation/tests/test_correlation.py

Unit tests for the full correlation pipeline:
    Phase 1 (retained): graph_builder.py -- 5-node synthetic scenario
    Phase 3 (new):      centrality.py, sequence_engine.py, graph_visualizer.py,
                        and the assembled graph_scores_df output contract.

Running
-------
From the project root:
    python -m pytest correlation/tests/test_correlation.py -v

Synthetic scenario (Phase 1, unchanged)
------------------------------------------
Five unique log templates (T1..T5) and one anomaly event placed on a timeline
so co-occurrence relationships are fully determined:

    t=0   T1
    t=10  T2, anomaly_label="anomaly:if_counter"
    t=30  T3
    t=50  T4   (within 60 s window of T1 through T3)
    t=70  T5   (outside 60 s window of T1, within window of T2..T4)
    t=75  T1   (second occurrence)

Expected: 6 nodes, 15 edges.  T1-T3 and T1-T4 both appear twice (weight=1.0);
all other edges have weight=0.5.
"""

import json
import math
import os
import sys
import tempfile

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from correlation.centrality import build_graph_scores_df, compute_centrality
from correlation.graph_builder import (
    CorrelationGraph,
    GraphEdge,
    GraphNode,
    LogEvent,
    build_graph,
    build_graph_from_parquet,
    correlation_graph_to_nx,
    load_graph,
    persist_graph,
)
from correlation.graph_visualizer import export_graph_json
from correlation.sequence_engine import detect_sequences


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

ANOMALY_LABEL = "anomaly:if_counter"

SYNTHETIC_EVENTS = [
    LogEvent(timestamp=0,  template="T1"),
    LogEvent(timestamp=10, template="T2", is_anomaly=True, anomaly_label=ANOMALY_LABEL),
    LogEvent(timestamp=30, template="T3"),
    LogEvent(timestamp=50, template="T4"),
    LogEvent(timestamp=70, template="T5"),
    LogEvent(timestamp=75, template="T1"),   # second occurrence of T1
]

EXPECTED_NODE_IDS = {"T1", "T2", "T3", "T4", "T5", ANOMALY_LABEL}
EXPECTED_EDGE_COUNT = 15


@pytest.fixture
def graph() -> CorrelationGraph:
    return build_graph(SYNTHETIC_EVENTS, time_window_seconds=60)


@pytest.fixture
def centrality_df(graph):
    return compute_centrality(graph)


@pytest.fixture
def simple_log_df() -> pd.DataFrame:
    """Minimal DataFrame matching sessionized_logs schema for score assembly tests."""
    rows = []
    for i, event in enumerate(SYNTHETIC_EVENTS):
        rows.append({
            "log_id": f"log_{i:03d}",
            "session_id": "s_0",
            "timestamp": event.timestamp,
            "template_id": event.template,
            "is_anomaly": event.is_anomaly,
            "anomaly_label": event.anomaly_label or "",
        })
    return pd.DataFrame(rows)


@pytest.fixture
def sequence_log_df() -> pd.DataFrame:
    """DataFrame with deliberate sequences across multiple sessions."""
    rows = []
    lid = [0]

    def add(session, ts, template):
        lid[0] += 1
        rows.append({
            "log_id": f"log_{lid[0]:04d}",
            "session_id": session,
            "timestamp": float(ts),
            "template_id": template,
            "is_anomaly": False,
            "anomaly_label": "",
        })

    # Three sessions each containing the sequence A->B->C within 30 s
    for s in range(5):
        sid = f"session_{s:02d}"
        base = s * 1000.0
        add(sid, base + 0, "A")
        add(sid, base + 5, "B")
        add(sid, base + 10, "C")
        # Extra noise events
        add(sid, base + 50, "D")
        add(sid, base + 60, "E")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Phase 1 tests (retained unchanged)
# ---------------------------------------------------------------------------

class TestNodes:
    def test_node_count(self, graph):
        assert len(graph.nodes) == 6

    def test_node_ids_present(self, graph):
        assert set(graph.nodes.keys()) == EXPECTED_NODE_IDS

    def test_node_types_log_template(self, graph):
        for tid in ("T1", "T2", "T3", "T4", "T5"):
            assert graph.nodes[tid].node_type == "log_template"

    def test_anomaly_node_type(self, graph):
        assert graph.nodes[ANOMALY_LABEL].node_type == "anomaly"

    def test_node_counts(self, graph):
        assert graph.nodes["T1"].count == 2
        for tid in ("T2", "T3", "T4", "T5"):
            assert graph.nodes[tid].count == 1
        assert graph.nodes[ANOMALY_LABEL].count == 1

    def test_node_schema(self, graph):
        for node in graph.nodes.values():
            assert isinstance(node, GraphNode)
            assert hasattr(node, "id")
            assert hasattr(node, "node_type")
            assert hasattr(node, "count")
            assert node.node_type in ("log_template", "anomaly")
            assert node.count > 0


class TestEdges:
    def test_edge_count(self, graph):
        assert len(graph.edges) == EXPECTED_EDGE_COUNT

    def test_edge_schema(self, graph):
        for edge in graph.edges.values():
            assert isinstance(edge, GraphEdge)
            assert isinstance(edge.source, str)
            assert isinstance(edge.target, str)
            assert isinstance(edge.co_occurrences, int)
            assert edge.co_occurrences >= 1
            assert isinstance(edge.weight, float)
            assert 0.0 < edge.weight <= 1.0

    def test_max_weight_is_one(self, graph):
        max_w = max(e.weight for e in graph.edges.values())
        assert math.isclose(max_w, 1.0)

    def test_weights_in_range(self, graph):
        for edge in graph.edges.values():
            assert 0.0 < edge.weight <= 1.0

    def test_highest_weight_edges(self, graph):
        def get_edge(a, b):
            key = (a, b) if a <= b else (b, a)
            assert key in graph.edges
            return graph.edges[key]

        e_t1_t3 = get_edge("T1", "T3")
        e_t1_t4 = get_edge("T1", "T4")
        assert e_t1_t3.co_occurrences == 2
        assert e_t1_t4.co_occurrences == 2
        assert math.isclose(e_t1_t3.weight, 1.0)
        assert math.isclose(e_t1_t4.weight, 1.0)

    def test_single_occurrence_edges_weight(self, graph):
        for edge in graph.edges.values():
            if edge.co_occurrences == 1:
                assert math.isclose(edge.weight, 0.5, rel_tol=1e-5)

    def test_anomaly_node_has_edges(self, graph):
        anomaly_edges = [
            e for e in graph.edges.values()
            if e.source == ANOMALY_LABEL or e.target == ANOMALY_LABEL
        ]
        assert len(anomaly_edges) > 0

    def test_anomaly_connected_to_template_nodes(self, graph):
        expected_neighbors = {"T1", "T2", "T3", "T4", "T5"}
        actual_neighbors = set()
        for (src, tgt), edge in graph.edges.items():
            if src == ANOMALY_LABEL:
                actual_neighbors.add(tgt)
            elif tgt == ANOMALY_LABEL:
                actual_neighbors.add(src)
        assert expected_neighbors == actual_neighbors

    def test_no_self_loops(self, graph):
        for src, tgt in graph.edges.keys():
            assert src != tgt

    def test_canonical_key_order(self, graph):
        for src, tgt in graph.edges.keys():
            assert src <= tgt


class TestGraphMetadata:
    def test_time_window_stored(self, graph):
        assert graph.time_window_seconds == 60

    def test_time_window_override(self):
        g = build_graph(SYNTHETIC_EVENTS, time_window_seconds=20)
        assert g.time_window_seconds == 20

    def test_narrow_window_fewer_edges(self):
        g_narrow = build_graph(SYNTHETIC_EVENTS, time_window_seconds=5)
        g_default = build_graph(SYNTHETIC_EVENTS, time_window_seconds=60)
        assert len(g_narrow.edges) < len(g_default.edges)

    def test_max_nodes_cap(self):
        g = build_graph(SYNTHETIC_EVENTS, time_window_seconds=60, max_nodes=3)
        template_nodes = [n for n in g.nodes.values() if n.node_type == "log_template"]
        assert len(template_nodes) <= 3

    def test_anomaly_node_admitted_regardless_of_cap(self):
        g = build_graph(SYNTHETIC_EVENTS, time_window_seconds=60, max_nodes=1)
        assert ANOMALY_LABEL in g.nodes

    def test_empty_input(self):
        g = build_graph([], time_window_seconds=60)
        assert len(g.nodes) == 0
        assert len(g.edges) == 0

    def test_single_event_no_edges(self):
        g = build_graph([LogEvent(timestamp=0, template="T1")], time_window_seconds=60)
        assert "T1" in g.nodes
        assert len(g.edges) == 0


# ---------------------------------------------------------------------------
# Phase 3 tests: graph_builder extensions
# ---------------------------------------------------------------------------

class TestGraphBuilderExtensions:
    def test_correlation_graph_to_nx(self, graph):
        import networkx as nx
        nx_g = correlation_graph_to_nx(graph)
        assert isinstance(nx_g, nx.Graph)
        assert nx_g.number_of_nodes() == len(graph.nodes)
        assert nx_g.number_of_edges() == len(graph.edges)

    def test_nx_node_attributes(self, graph):
        nx_g = correlation_graph_to_nx(graph)
        for node_id, attrs in nx_g.nodes(data=True):
            assert "node_type" in attrs
            assert "count" in attrs
            assert attrs["node_type"] in ("log_template", "anomaly")
            assert attrs["count"] > 0

    def test_nx_edge_weight_attribute(self, graph):
        nx_g = correlation_graph_to_nx(graph)
        for u, v, data in nx_g.edges(data=True):
            assert "weight" in data
            assert 0.0 < data["weight"] <= 1.0

    def test_persist_and_load_graph(self, graph, tmp_path):
        pickle_path = str(tmp_path / "test_graph.gpickle")
        persist_graph(graph, pickle_path)
        loaded = load_graph(pickle_path)
        assert set(loaded.nodes.keys()) == set(graph.nodes.keys())
        assert len(loaded.edges) == len(graph.edges)

    def test_load_graph_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_graph(str(tmp_path / "nonexistent.gpickle"))

    def test_build_graph_from_parquet(self, simple_log_df, tmp_path):
        parquet_path = str(tmp_path / "test_logs.parquet")
        simple_log_df.to_parquet(parquet_path, index=False)
        g = build_graph_from_parquet(parquet_path, time_window_seconds=60)
        assert len(g.nodes) > 0
        # At minimum all templates from the events should appear as nodes
        for tmpl in simple_log_df["template_id"].unique():
            assert tmpl in g.nodes

    def test_build_graph_from_parquet_datetime_timestamp(self, simple_log_df, tmp_path):
        """Parquet files with datetime timestamps must be converted correctly."""
        import pandas as pd
        df = simple_log_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        parquet_path = str(tmp_path / "test_logs_dt.parquet")
        df.to_parquet(parquet_path, index=False)
        g = build_graph_from_parquet(parquet_path, time_window_seconds=60)
        assert len(g.nodes) > 0


# ---------------------------------------------------------------------------
# Phase 3 tests: centrality
# ---------------------------------------------------------------------------

class TestCentrality:
    def test_returns_dataframe(self, centrality_df):
        assert isinstance(centrality_df, pd.DataFrame)

    def test_required_columns(self, centrality_df):
        required = {"node_id", "degree_centrality", "betweenness", "pagerank_score", "centrality_score"}
        assert required.issubset(set(centrality_df.columns))

    def test_one_row_per_node(self, graph, centrality_df):
        assert len(centrality_df) == len(graph.nodes)

    def test_no_nan_values(self, centrality_df):
        assert not centrality_df[["degree_centrality", "betweenness", "pagerank_score"]].isna().any().any()

    def test_degree_centrality_in_range(self, centrality_df):
        assert (centrality_df["degree_centrality"] >= 0.0).all()
        assert (centrality_df["degree_centrality"] <= 1.0).all()

    def test_betweenness_in_range(self, centrality_df):
        assert (centrality_df["betweenness"] >= 0.0).all()
        assert (centrality_df["betweenness"] <= 1.0).all()

    def test_pagerank_in_range(self, centrality_df):
        assert (centrality_df["pagerank_score"] >= 0.0).all()
        assert (centrality_df["pagerank_score"] <= 1.0).all()

    def test_centrality_score_equals_pagerank(self, centrality_df):
        assert (centrality_df["centrality_score"] == centrality_df["pagerank_score"]).all()

    def test_empty_graph_returns_empty_df(self):
        g = build_graph([], time_window_seconds=60)
        df = compute_centrality(g)
        assert len(df) == 0

    def test_betweenness_k_used_for_large_graph(self, monkeypatch):
        """When graph > BETWEENNESS_LARGE_GRAPH_THRESHOLD the k-approx path runs."""
        import common.config as cfg
        monkeypatch.setattr(cfg, "BETWEENNESS_LARGE_GRAPH_THRESHOLD", 0)
        g = build_graph(SYNTHETIC_EVENTS, time_window_seconds=60)
        df = compute_centrality(g)
        # Should not raise and should still return valid scores
        assert (df["betweenness"] >= 0.0).all()
        assert (df["betweenness"] <= 1.0).all()


# ---------------------------------------------------------------------------
# Phase 3 tests: graph_scores_df output contract
# ---------------------------------------------------------------------------

class TestGraphScoresDf:
    @pytest.fixture
    def scores_df(self, graph, centrality_df, simple_log_df):
        return build_graph_scores_df(centrality_df, simple_log_df, graph)

    def test_returns_dataframe(self, scores_df):
        assert isinstance(scores_df, pd.DataFrame)

    def test_one_row_per_log_id(self, simple_log_df, scores_df):
        assert len(scores_df) == len(simple_log_df)

    def test_schema_contract(self, scores_df):
        """P4 output contract: exact column set in exact order."""
        expected = [
            "log_id", "centrality_score", "degree",
            "betweenness", "cluster_id", "in_sequence", "correlated_log_ids",
        ]
        assert list(scores_df.columns) == expected

    def test_centrality_score_in_range(self, scores_df):
        assert (scores_df["centrality_score"] >= 0.0).all()
        assert (scores_df["centrality_score"] <= 1.0).all()

    def test_betweenness_in_range(self, scores_df):
        assert (scores_df["betweenness"] >= 0.0).all()
        assert (scores_df["betweenness"] <= 1.0).all()

    def test_degree_is_int(self, scores_df):
        assert scores_df["degree"].dtype in (int, "int64", "int32")

    def test_in_sequence_is_bool(self, scores_df):
        assert scores_df["in_sequence"].dtype == bool

    def test_cluster_id_is_string(self, scores_df):
        # pandas 2.x may use StringDtype instead of object; check values directly
        assert scores_df["cluster_id"].str.startswith("cc_").all()

    def test_correlated_log_ids_is_list_column(self, scores_df):
        assert scores_df["correlated_log_ids"].dtype == object
        for val in scores_df["correlated_log_ids"]:
            assert isinstance(val, list)

    def test_in_sequence_populated_from_set(self, graph, centrality_df, simple_log_df):
        first_log_id = simple_log_df["log_id"].iloc[0]
        df = build_graph_scores_df(
            centrality_df, simple_log_df, graph,
            sequence_log_ids={first_log_id}
        )
        assert df.loc[df["log_id"] == first_log_id, "in_sequence"].iloc[0] == True

    def test_no_missing_log_ids(self, simple_log_df, scores_df):
        assert set(scores_df["log_id"]) == set(simple_log_df["log_id"])

    def test_parquet_roundtrip(self, scores_df, tmp_path):
        path = str(tmp_path / "graph_scores_df.parquet")
        scores_df.to_parquet(path, index=False)
        reloaded = pd.read_parquet(path)
        assert list(reloaded.columns) == list(scores_df.columns)
        assert len(reloaded) == len(scores_df)


# ---------------------------------------------------------------------------
# Phase 3 tests: sequence_engine
# ---------------------------------------------------------------------------

class TestSequenceEngine:
    def test_returns_set(self, sequence_log_df, tmp_path):
        out = str(tmp_path / "sequences.json")
        result = detect_sequences(sequence_log_df, min_length=3, min_support=3, output_path=out)
        assert isinstance(result, set)

    def test_sequences_json_is_valid_json(self, sequence_log_df, tmp_path):
        out = str(tmp_path / "sequences.json")
        detect_sequences(sequence_log_df, min_length=3, min_support=3, output_path=out)
        with open(out) as fh:
            data = json.load(fh)
        assert isinstance(data, list)

    def test_sequence_entry_schema(self, sequence_log_df, tmp_path):
        out = str(tmp_path / "sequences.json")
        detect_sequences(sequence_log_df, min_length=3, min_support=3, output_path=out)
        with open(out) as fh:
            data = json.load(fh)
        for entry in data:
            assert "sequence" in entry
            assert "support_count" in entry
            assert "session_ids" in entry
            assert isinstance(entry["sequence"], list)
            assert isinstance(entry["support_count"], int)
            assert isinstance(entry["session_ids"], list)

    def test_sequence_min_length_respected(self, sequence_log_df, tmp_path):
        out = str(tmp_path / "sequences.json")
        detect_sequences(sequence_log_df, min_length=3, min_support=3, output_path=out)
        with open(out) as fh:
            data = json.load(fh)
        for entry in data:
            assert len(entry["sequence"]) >= 3

    def test_sequence_min_support_respected(self, sequence_log_df, tmp_path):
        out = str(tmp_path / "sequences.json")
        min_sup = 3
        detect_sequences(sequence_log_df, min_length=3, min_support=min_sup, output_path=out)
        with open(out) as fh:
            data = json.load(fh)
        for entry in data:
            assert entry["support_count"] >= min_sup

    def test_seeded_sequence_detected(self, sequence_log_df, tmp_path):
        """The deliberately seeded A->B->C sequence must be detected."""
        out = str(tmp_path / "sequences.json")
        detect_sequences(sequence_log_df, min_length=3, min_support=3, output_path=out)
        with open(out) as fh:
            data = json.load(fh)
        sequences = [tuple(e["sequence"]) for e in data]
        assert ("A", "B", "C") in sequences

    def test_in_sequence_log_ids_nonempty(self, sequence_log_df, tmp_path):
        out = str(tmp_path / "sequences.json")
        result = detect_sequences(sequence_log_df, min_length=3, min_support=3, output_path=out)
        assert len(result) > 0

    def test_missing_column_raises(self, tmp_path):
        bad_df = pd.DataFrame({"log_id": ["a"], "session_id": ["s0"]})
        with pytest.raises(ValueError, match="missing columns"):
            detect_sequences(bad_df, output_path=str(tmp_path / "out.json"))

    def test_high_min_support_yields_empty(self, sequence_log_df, tmp_path):
        out = str(tmp_path / "sequences.json")
        result = detect_sequences(
            sequence_log_df, min_length=3, min_support=9999, output_path=out
        )
        assert result == set()
        with open(out) as fh:
            data = json.load(fh)
        assert data == []


# ---------------------------------------------------------------------------
# Phase 3 tests: graph_visualizer
# ---------------------------------------------------------------------------

class TestGraphVisualizer:
    @pytest.fixture
    def json_payload(self, graph, centrality_df, tmp_path):
        out = str(tmp_path / "correlation_graph.json")
        payload = export_graph_json(graph, centrality_df, output_path=out)
        return payload, out

    def test_file_is_valid_json(self, json_payload):
        _, path = json_payload
        with open(path) as fh:
            data = json.load(fh)
        assert isinstance(data, dict)

    def test_top_level_keys(self, json_payload):
        payload, _ = json_payload
        assert "nodes" in payload
        assert "edges" in payload

    def test_nodes_nonempty(self, json_payload):
        payload, _ = json_payload
        assert len(payload["nodes"]) > 0

    def test_edges_nonempty(self, json_payload):
        payload, _ = json_payload
        assert len(payload["edges"]) > 0

    def test_node_schema(self, json_payload):
        payload, _ = json_payload
        for node in payload["nodes"]:
            assert "id" in node
            assert "template" in node
            assert "node_type" in node
            assert "centrality_score" in node
            assert isinstance(node["centrality_score"], float)
            assert 0.0 <= node["centrality_score"] <= 1.0

    def test_edge_schema(self, json_payload):
        payload, _ = json_payload
        for edge in payload["edges"]:
            assert "source" in edge
            assert "target" in edge
            assert "weight" in edge
            assert isinstance(edge["weight"], float)
            assert 0.0 < edge["weight"] <= 1.0

    def test_node_count_matches_graph(self, graph, centrality_df, tmp_path):
        out = str(tmp_path / "g.json")
        payload = export_graph_json(graph, centrality_df, output_path=out)
        assert len(payload["nodes"]) == len(graph.nodes)

    def test_edge_count_matches_graph(self, graph, centrality_df, tmp_path):
        out = str(tmp_path / "g.json")
        payload = export_graph_json(graph, centrality_df, output_path=out)
        assert len(payload["edges"]) == len(graph.edges)

    def test_empty_graph_produces_empty_lists(self, tmp_path):
        empty_g = build_graph([], time_window_seconds=60)
        empty_centrality = compute_centrality(empty_g)
        out = str(tmp_path / "empty.json")
        payload = export_graph_json(empty_g, empty_centrality, output_path=out)
        assert payload["nodes"] == []
        assert payload["edges"] == []
