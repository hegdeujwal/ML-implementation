"""
correlation/tests/test_correlation.py

Unit tests for the correlation pipeline (P3):
  graph_builder.py, centrality.py, sequence_engine.py, graph_visualizer.py

All synthetic DataFrames use sequence_number (not log_id) as the universal
join key and include: session_id, template_id, timestamp, host, frequency.

Running
-------
From the project root:
    pytest correlation/tests/test_correlation.py -v
"""

import dataclasses
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import common.config as cfg
from common.schema import GraphScoreRow
from correlation.centrality import compute_centrality
from correlation.graph_builder import build_graph, load_or_build_graph
from correlation.graph_visualizer import export_graph_json, export_graph_png
from correlation.sequence_engine import detect_sequences


# ---------------------------------------------------------------------------
# Helper: build_sessions_df
# ---------------------------------------------------------------------------

def build_sessions_df(n_sessions: int, templates_per_session: int) -> pd.DataFrame:
    """Generate synthetic sessionized_logs with deterministic template sequences.

    Templates are named T001..T{N}.
    Timestamps within a session are 10 s apart.
    Sessions are 10 minutes (600 s) apart.
    """
    rows = []
    seq_num = 0
    base = pd.Timestamp("2026-01-01 00:00:00")
    for s in range(n_sessions):
        session_id = f"session_{s:03d}"
        session_ts = base + pd.Timedelta(seconds=s * 600)
        for t in range(templates_per_session):
            template_id = f"T{t + 1:03d}"
            ts = session_ts + pd.Timedelta(seconds=t * 10)
            rows.append({
                "sequence_number": seq_num,
                "session_id": session_id,
                "template_id": template_id,
                "timestamp": ts,
                "host": "host-01",
                "frequency": 1,
            })
            seq_num += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Autouse fixture: redirect all file outputs to tmp_path
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_output_paths(monkeypatch, tmp_path):
    """Keep tests isolated — redirect all disk writes to tmp_path."""
    import correlation.graph_builder as gb
    monkeypatch.setattr(gb, "_GRAPH_PICKLE_PATH", str(tmp_path / "graph.pkl"))
    monkeypatch.setattr(cfg, "GRAPH_SCORES_PATH", str(tmp_path / "graph_scores.parquet"))
    monkeypatch.setattr(cfg, "SEQUENCES_JSON_PATH", str(tmp_path / "sequences.json"))


# ---------------------------------------------------------------------------
# TestGraphBuilder
# ---------------------------------------------------------------------------

class TestGraphBuilder:
    def test_node_count(self):
        """5 unique templates across 3 sessions → 5 nodes in graph."""
        df = build_sessions_df(n_sessions=3, templates_per_session=5)
        graph = build_graph(df)
        assert graph.number_of_nodes() == 5

    def test_no_cross_session_edges(self):
        """Templates from different sessions must not share edges even when
        their timestamps overlap within the co-occurrence time window."""
        base = pd.Timestamp("2026-01-01")
        rows = [
            # Session A: T001 at t=0s, T002 at t=10s
            {"sequence_number": 0, "session_id": "sA", "template_id": "T001",
             "timestamp": base, "host": "h", "frequency": 1},
            {"sequence_number": 1, "session_id": "sA", "template_id": "T002",
             "timestamp": base + pd.Timedelta(seconds=10), "host": "h", "frequency": 1},
            # Session B: T003 at t=5s, T004 at t=15s — wall-clock overlaps sA
            {"sequence_number": 2, "session_id": "sB", "template_id": "T003",
             "timestamp": base + pd.Timedelta(seconds=5), "host": "h", "frequency": 1},
            {"sequence_number": 3, "session_id": "sB", "template_id": "T004",
             "timestamp": base + pd.Timedelta(seconds=15), "host": "h", "frequency": 1},
        ]
        df = pd.DataFrame(rows)
        graph = build_graph(df)
        edge_pairs = {frozenset(e[:2]) for e in graph.edges}
        for pair in [
            frozenset({"T001", "T003"}), frozenset({"T001", "T004"}),
            frozenset({"T002", "T003"}), frozenset({"T002", "T004"}),
        ]:
            assert pair not in edge_pairs, f"Cross-session edge found: {pair}"

    def test_session_weight_in_range(self):
        df = build_sessions_df(n_sessions=5, templates_per_session=3)
        graph = build_graph(df)
        for u, v, data in graph.edges(data=True):
            assert 0.0 <= data["weight"] <= 1.0, f"weight out of range for {u}-{v}"

    def test_pmi_non_negative(self):
        df = build_sessions_df(n_sessions=5, templates_per_session=3)
        graph = build_graph(df)
        for u, v, data in graph.edges(data=True):
            assert data["pmi"] >= 0.0, f"pmi < 0 for edge {u}-{v}"

    def test_cluster_id_format_on_all_nodes(self):
        df = build_sessions_df(n_sessions=3, templates_per_session=3)
        graph = build_graph(df)
        pattern = re.compile(r"^C\d{4}$")
        for node in graph.nodes:
            cid = graph.nodes[node].get("cluster_id", "")
            assert pattern.match(cid), f"cluster_id {cid!r} does not match C{{n:04d}}"

    def test_largest_component_is_c0000(self):
        """The largest connected component must receive cluster_id 'C0000'.

        Large group: T001-T002-T003 (3-node component).
        Small group: T004-T005      (2-node component).
        3 > 2, so C0000 is deterministic for the large group.
        """
        base = pd.Timestamp("2026-01-01")
        rows = []
        seq = 0
        # Large group: T001, T002, T003 co-occur within 60 s in 5 sessions
        for s in range(5):
            for i, tmpl in enumerate(["T001", "T002", "T003"]):
                rows.append({
                    "sequence_number": seq, "session_id": f"sA_{s}",
                    "template_id": tmpl,
                    "timestamp": base + pd.Timedelta(seconds=s * 600 + i * 10),
                    "host": "h", "frequency": 1,
                })
                seq += 1
        # Small group: T004, T005 co-occur in 2 sessions, time-separated from A
        for s in range(2):
            for i, tmpl in enumerate(["T004", "T005"]):
                rows.append({
                    "sequence_number": seq, "session_id": f"sB_{s}",
                    "template_id": tmpl,
                    "timestamp": base + pd.Timedelta(seconds=(100 + s) * 600 + i * 10),
                    "host": "h", "frequency": 1,
                })
                seq += 1
        df = pd.DataFrame(rows)
        graph = build_graph(df)
        # 3-node component → C0000
        for tmpl in ("T001", "T002", "T003"):
            assert graph.nodes[tmpl]["cluster_id"] == "C0000", tmpl
        # 2-node component → C0001
        for tmpl in ("T004", "T005"):
            assert graph.nodes[tmpl]["cluster_id"] == "C0001", tmpl

    def test_templates_beyond_max_nodes_excluded(self, monkeypatch):
        monkeypatch.setattr(cfg, "GRAPH_MAX_NODES", 2)
        df = build_sessions_df(n_sessions=3, templates_per_session=5)
        graph = build_graph(df)
        assert graph.number_of_nodes() <= 2
        assert set(graph.nodes) == graph.graph["included_templates"]

    def test_load_or_build_graph_uses_cache(self, monkeypatch):
        """Second call must load from the pkl cache, not invoke build_graph."""
        import correlation.graph_builder as gb
        df = build_sessions_df(n_sessions=3, templates_per_session=3)
        graph1 = load_or_build_graph(df)  # builds and caches

        monkeypatch.setattr(
            gb, "build_graph",
            lambda _: pytest.fail("build_graph must not be called on a cache hit"),
        )
        graph2 = load_or_build_graph(df)  # must load from cache
        assert graph2.number_of_nodes() == graph1.number_of_nodes()


# ---------------------------------------------------------------------------
# TestCentrality
# ---------------------------------------------------------------------------

class TestCentrality:
    def test_centrality_score_in_range(self):
        df = build_sessions_df(n_sessions=5, templates_per_session=3)
        result = compute_centrality(build_graph(df), df)
        assert (result["centrality_score"] >= 0.0).all()
        assert (result["centrality_score"] <= 1.0).all()

    def test_betweenness_in_range(self):
        df = build_sessions_df(n_sessions=5, templates_per_session=3)
        result = compute_centrality(build_graph(df), df)
        assert (result["betweenness"] >= 0.0).all()
        assert (result["betweenness"] <= 1.0).all()

    def test_degree_non_negative(self):
        df = build_sessions_df(n_sessions=5, templates_per_session=3)
        result = compute_centrality(build_graph(df), df)
        assert (result["degree"] >= 0).all()

    def test_in_graph_true_for_included_templates(self):
        df = build_sessions_df(n_sessions=3, templates_per_session=3)
        graph = build_graph(df)
        result = compute_centrality(graph, df)
        included = graph.graph["included_templates"]
        tmpl_map = df.set_index("sequence_number")["template_id"]
        for _, row in result.iterrows():
            if tmpl_map[row["sequence_number"]] in included:
                assert row["in_graph"] == True

    def test_in_graph_false_for_capped_templates(self, monkeypatch):
        monkeypatch.setattr(cfg, "GRAPH_MAX_NODES", 2)
        df = build_sessions_df(n_sessions=3, templates_per_session=5)
        graph = build_graph(df)
        result = compute_centrality(graph, df)
        included = graph.graph["included_templates"]
        tmpl_map = df.set_index("sequence_number")["template_id"]
        capped = result[result.apply(
            lambda r: tmpl_map[r["sequence_number"]] not in included, axis=1
        )]
        assert len(capped) > 0
        assert (capped["in_graph"] == False).all()

    def test_capped_templates_get_global_mean_not_zero(self, monkeypatch):
        monkeypatch.setattr(cfg, "GRAPH_MAX_NODES", 2)
        df = build_sessions_df(n_sessions=5, templates_per_session=5)
        graph = build_graph(df)
        result = compute_centrality(graph, df)
        global_mean = float(result[result["in_graph"]]["centrality_score"].mean())
        capped_scores = result[~result["in_graph"]]["centrality_score"]
        assert len(capped_scores) > 0
        # Use abs diff — pytest.approx does not dispatch through pandas __eq__
        assert (capped_scores - global_mean).abs().max() < 1e-6

    def test_capped_templates_cluster_id_is_uncapped(self, monkeypatch):
        monkeypatch.setattr(cfg, "GRAPH_MAX_NODES", 2)
        df = build_sessions_df(n_sessions=3, templates_per_session=5)
        graph = build_graph(df)
        result = compute_centrality(graph, df)
        capped = result[~result["in_graph"]]
        assert len(capped) > 0
        assert (capped["cluster_id"] == "UNCAPPED").all()

    def test_degenerate_single_node_gives_half(self):
        """Single-template graph → all-equal PageRank → degenerate case → 0.5."""
        df = build_sessions_df(n_sessions=1, templates_per_session=1)
        graph = build_graph(df)
        result = compute_centrality(graph, df)
        # Use abs diff — pytest.approx does not dispatch through pandas __eq__
        assert (result["centrality_score"] - 0.5).abs().max() < 1e-6

    def test_correlated_log_ids_non_empty_for_neighbours(self):
        """A template with a graph neighbour in the same session must have a
        non-empty correlated_log_ids list."""
        df = build_sessions_df(n_sessions=3, templates_per_session=2)
        result = compute_centrality(build_graph(df), df)
        in_graph_rows = result[result["in_graph"]]
        total_correlated = in_graph_rows["correlated_log_ids"].apply(len).sum()
        assert total_correlated > 0

    def test_correlated_log_ids_are_strings(self):
        """correlated_log_ids must be a list of strings (sequence_number cast to str)."""
        df = build_sessions_df(n_sessions=3, templates_per_session=2)
        result = compute_centrality(build_graph(df), df)
        for vals in result["correlated_log_ids"]:
            assert isinstance(vals, list)
            for v in vals:
                assert isinstance(v, str), f"Expected str in correlated_log_ids, got {type(v)}"

    def test_output_schema_matches_graphscorerow(self):
        expected_cols = [f.name for f in dataclasses.fields(GraphScoreRow)]
        df = build_sessions_df(n_sessions=3, templates_per_session=3)
        result = compute_centrality(build_graph(df), df)
        assert list(result.columns) == expected_cols


# ---------------------------------------------------------------------------
# TestSequenceEngine
# ---------------------------------------------------------------------------

def _make_sequence_df(n_with_seq: int, n_noise: int = 0) -> pd.DataFrame:
    """Build a df where the first n_with_seq sessions contain T001→T002→T003
    within 30 s (fits SEQUENCE_WINDOW_SECONDS), plus n_noise single-log sessions."""
    rows = []
    seq_num = 0
    base = pd.Timestamp("2026-01-01")
    for s in range(n_with_seq):
        ts_base = base + pd.Timedelta(seconds=s * 600)
        for i, tmpl in enumerate(["T001", "T002", "T003"]):
            rows.append({
                "sequence_number": seq_num,
                "session_id": f"seq_{s:03d}",
                "template_id": tmpl,
                "timestamp": ts_base + pd.Timedelta(seconds=i * 5),
                "host": "h", "frequency": 1,
            })
            seq_num += 1
    for s in range(n_noise):
        ts_base = base + pd.Timedelta(seconds=(n_with_seq + s) * 600)
        rows.append({
            "sequence_number": seq_num,
            "session_id": f"noise_{s:03d}",
            "template_id": "T999",
            "timestamp": ts_base,
            "host": "h", "frequency": 1,
        })
        seq_num += 1
    return pd.DataFrame(rows)


class TestSequenceEngine:
    def test_sequence_above_support_is_known_pattern(self, tmp_path):
        """Sequence appearing in >= 5 sessions → returned in result set."""
        df = _make_sequence_df(n_with_seq=6)
        out = str(tmp_path / "seq.json")
        result = detect_sequences(df, min_length=3, min_support=5, output_path=out)
        assert len(result) > 0

    def test_sequence_below_support_is_not_detected(self, tmp_path):
        """Sequence appearing in < 5 sessions → empty result set."""
        df = _make_sequence_df(n_with_seq=3)
        out = str(tmp_path / "seq.json")
        result = detect_sequences(df, min_length=3, min_support=5, output_path=out)
        assert len(result) == 0

    def test_in_sequence_true_for_known_pattern_logs(self, tmp_path):
        """Logs belonging to a known sequence → in_sequence = True after isin()."""
        df = _make_sequence_df(n_with_seq=6)
        out = str(tmp_path / "seq.json")
        seq_set = detect_sequences(df, min_length=3, min_support=5, output_path=out)
        in_seq = df["sequence_number"].isin(seq_set)
        assert in_seq.any()

    def test_in_sequence_false_for_noise_logs(self, tmp_path):
        """Logs not part of any sequence (T999 noise) → not in result set."""
        df = _make_sequence_df(n_with_seq=6, n_noise=4)
        out = str(tmp_path / "seq.json")
        seq_set = detect_sequences(df, min_length=3, min_support=5, output_path=out)
        noise_seqnums = set(df[df["template_id"] == "T999"]["sequence_number"])
        assert noise_seqnums.isdisjoint(seq_set)

    def test_sequences_json_structure(self, tmp_path):
        """sequences.json must be a list where each entry has the correct keys."""
        df = _make_sequence_df(n_with_seq=6)
        out = str(tmp_path / "seq.json")
        detect_sequences(df, min_length=3, min_support=5, output_path=out)
        with open(out) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) > 0
        for entry in data:
            assert "sequence" in entry
            assert "support_count" in entry
            assert "session_ids" in entry
            assert isinstance(entry["sequence"], list)
            assert isinstance(entry["support_count"], int)
            assert isinstance(entry["session_ids"], list)

    def test_single_log_sessions_no_sequences_no_crash(self, tmp_path):
        """Single-log sessions cannot form sequences — must return empty set."""
        df = pd.DataFrame({
            "sequence_number": [0, 1, 2],
            "session_id":      ["s0", "s1", "s2"],
            "template_id":     ["T001", "T001", "T001"],
            "timestamp":       [pd.Timestamp("2026-01-01")] * 3,
            "host":            ["h"] * 3,
            "frequency":       [1] * 3,
        })
        out = str(tmp_path / "seq.json")
        result = detect_sequences(df, min_length=3, min_support=5, output_path=out)
        assert result == set()


# ---------------------------------------------------------------------------
# TestGraphVisualizer
# ---------------------------------------------------------------------------

class TestGraphVisualizer:
    @pytest.fixture
    def exported(self, tmp_path):
        """Build a small graph, export JSON, and return (graph, parsed_data)."""
        df = build_sessions_df(n_sessions=3, templates_per_session=3)
        graph = build_graph(df)
        out = str(tmp_path / "graph.json")
        export_graph_json(graph, out)
        with open(out) as f:
            data = json.load(f)
        return graph, data

    def test_returns_output_path(self, tmp_path):
        df = build_sessions_df(n_sessions=3, templates_per_session=3)
        graph = build_graph(df)
        out = str(tmp_path / "graph.json")
        assert export_graph_json(graph, out) == out

    def test_file_is_valid_json(self, exported):
        _, data = exported
        assert isinstance(data, dict)

    def test_top_level_keys(self, exported):
        _, data = exported
        assert "nodes" in data
        assert "edges" in data
        assert "metadata" in data

    def test_all_node_ids_exist_in_graph(self, exported):
        graph, data = exported
        graph_node_ids = set(graph.nodes)
        for node in data["nodes"]:
            assert node["id"] in graph_node_ids

    def test_edge_endpoints_exist_as_node_ids(self, exported):
        _, data = exported
        node_ids = {n["id"] for n in data["nodes"]}
        for edge in data["edges"]:
            assert edge["source"] in node_ids
            assert edge["target"] in node_ids

    def test_metadata_n_nodes_matches_graph(self, exported):
        graph, data = exported
        assert data["metadata"]["n_nodes"] == graph.number_of_nodes()

    def test_export_graph_png_returns_none(self, tmp_path):
        df = build_sessions_df(n_sessions=1, templates_per_session=2)
        graph = build_graph(df)
        result = export_graph_png(graph, str(tmp_path / "graph.png"))
        assert result is None
