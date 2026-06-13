"""
scoring/tests/test_scoring.py

Full test suite for the scoring/ module.

All inputs are synthetic DataFrames built here — no parquet files are read
from disk.  The autouse fixture patches root_cause_engine's output paths to
pytest's tmp_path so no production files are written during tests.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import common.config as cfg
from scoring.importance_scorer import score
from scoring.incident_clusterer import cluster_incidents
from scoring.label_mapper import map_labels
from scoring.root_cause_engine import identify_root_causes


# ---------------------------------------------------------------------------
# Synthetic data builder
# ---------------------------------------------------------------------------

def build_test_inputs(n_rows: int = 100, n_sessions: int = 5):
    """Return (features_df, anomaly_df, graph_scores_df) with consistent
    sequence_numbers, finite values, and a mix of in_graph True/False.
    """
    rng = np.random.default_rng(42)

    # Assign rows to sessions; sessions are far apart in time so there is
    # no cross-session proximity confusion in temporal_proximity tests.
    sequence_numbers: list[int] = []
    session_ids: list[str] = []
    timestamps: list[pd.Timestamp] = []

    base_ts = pd.Timestamp("2024-01-01")
    seq = 1
    rows_per_session, remainder = divmod(n_rows, n_sessions)

    for s in range(n_sessions):
        n = rows_per_session + (1 if s < remainder else 0)
        for r in range(n):
            sequence_numbers.append(seq)
            session_ids.append(f"S{s:03d}")
            timestamps.append(base_ts + pd.Timedelta(seconds=s * 10_000 + r * 10))
            seq += 1

    n = len(sequence_numbers)
    seq_arr = np.array(sequence_numbers)

    features_df = pd.DataFrame({
        "sequence_number": seq_arr,
        "session_id": session_ids,
        "timestamp": timestamps,
        "template_id": [f"T{i % 10:03d}" for i in range(n)],
        "host": [f"host{i % 3}" for i in range(n)],
        "frequency_score": rng.uniform(0, 1, n),
        "burstiness_score": rng.uniform(0, 1, n),
        "zscore_base": rng.uniform(0, 1, n),
        "time_delta_prev": rng.uniform(0, 100, n),
        "time_delta_session_start": rng.uniform(0, 1000, n),
        "inter_arrival_rate": rng.uniform(0, 1, n),
        "event_weight": rng.uniform(0.1, 1.0, n),
        "counter_proximity": rng.uniform(0, 1, n),
    })

    anomaly_df = pd.DataFrame({
        "sequence_number": seq_arr,
        "isolation_score": rng.uniform(0, 1, n),
        "zscore_norm": rng.uniform(0, 1, n),
        "combined_score": rng.uniform(0, 1, n),
        "is_anomaly": rng.choice([True, False], n),
        "model_confidence": rng.uniform(0, 1, n),
    })

    # ~2/3 in-graph, C0000/C0001/C0002 cluster_ids
    in_graph = np.array([i % 3 != 0 for i in range(n)])
    cluster_ids = [f"C{i % 3:04d}" for i in range(n)]

    graph_scores_df = pd.DataFrame({
        "sequence_number": seq_arr,
        "centrality_score": rng.uniform(0, 1, n),
        "degree": rng.integers(0, 10, n),
        "betweenness": rng.uniform(0, 1, n),
        "in_graph": in_graph,
        "cluster_id": cluster_ids,
        "in_sequence": rng.choice([True, False], n),
        "correlated_log_ids": [[] for _ in range(n)],
    })

    return features_df, anomaly_df, graph_scores_df


# ---------------------------------------------------------------------------
# Helpers for incident_clusterer and root_cause_engine tests
# ---------------------------------------------------------------------------

def _make_clusterable_df() -> pd.DataFrame:
    """Two tight DBSCAN clusters + 2 noise points + 5 ignore rows.

    Cluster A (10 rows, "critical"): all features ≈ 0.9, cluster_id spans
        C0000 and C0001 → is_cross_system should be True after clustering.
    Cluster B (10 rows, "low"):    all features ≈ 0.1, cluster_id = C0002
        → is_cross_system should be False.
    Noise (2 rows, "medium"):      features ≈ 0.5, only 2 rows < min_samples.
    Ignore (5 rows, "ignore"):     excluded from DBSCAN entirely.
    """
    rows: list[dict] = []
    seq = 1

    for i in range(10):  # Cluster A
        rows.append({
            "sequence_number": seq, "session_id": "S000",
            "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=i),
            "final_score": 0.9, "centrality_score": 0.9, "temporal_proximity": 0.9,
            "label": "critical",
            "cluster_id": "C0000" if i < 5 else "C0001",  # spans 2 graph clusters
            "in_graph": True, "correlation_id": None, "is_cross_system": False,
        })
        seq += 1

    for i in range(10):  # Cluster B
        rows.append({
            "sequence_number": seq, "session_id": "S001",
            "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=i),
            "final_score": 0.1, "centrality_score": 0.1, "temporal_proximity": 0.1,
            "label": "low", "cluster_id": "C0002",
            "in_graph": True, "correlation_id": None, "is_cross_system": False,
        })
        seq += 1

    for i in range(2):  # Noise — only 2 rows, below min_samples=5
        rows.append({
            "sequence_number": seq, "session_id": "S002",
            "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=i),
            "final_score": 0.5 + i * 0.01,
            "centrality_score": 0.5 + i * 0.01,
            "temporal_proximity": 0.5 + i * 0.01,
            "label": "medium", "cluster_id": "C0003",
            "in_graph": True, "correlation_id": None, "is_cross_system": False,
        })
        seq += 1

    for i in range(5):  # Ignore — excluded from clustering
        rows.append({
            "sequence_number": seq, "session_id": "S003",
            "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=i),
            "final_score": 0.15, "centrality_score": 0.15, "temporal_proximity": 0.0,
            "label": "ignore", "cluster_id": "C0000",
            "in_graph": False, "correlation_id": None, "is_cross_system": False,
        })
        seq += 1

    return pd.DataFrame(rows)


def _make_incident_df() -> pd.DataFrame:
    """Scored df with pre-assigned correlation_ids for root cause tests.

    INC-0000: 5 in_graph=True (centrality 0.1–0.5) + 3 in_graph=False
              (centrality 0.7–0.9). in_graph=False rows intentionally have
              higher centrality to test that in_graph=True is preferred.
    INC-0001: 10 in_graph=True rows with decreasing centrality 0.9→0.45.
    Noise:    3 rows with correlation_id=None.
    """
    rows: list[dict] = []
    seq = 1

    for cen, ig in [
        (0.5, True), (0.4, True), (0.3, True), (0.2, True), (0.1, True),
        (0.9, False), (0.8, False), (0.7, False),
    ]:
        rows.append({
            "sequence_number": seq, "session_id": "S000",
            "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=seq),
            "temporal_proximity": 0.5,
            "final_score": 0.8, "label": "critical",
            "centrality_score": cen, "cluster_id": "C0000",
            "in_graph": ig, "correlation_id": "INC-0000", "is_cross_system": False,
        })
        seq += 1

    for i in range(10):
        rows.append({
            "sequence_number": seq, "session_id": "S001",
            "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=seq),
            "temporal_proximity": 0.5,
            "final_score": 0.6, "label": "medium",
            "centrality_score": round(0.9 - i * 0.05, 4),
            "cluster_id": "C0001",
            "in_graph": True, "correlation_id": "INC-0001", "is_cross_system": False,
        })
        seq += 1

    for i in range(3):  # noise / non-incident rows
        rows.append({
            "sequence_number": seq, "session_id": "S002",
            "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=seq),
            "temporal_proximity": 0.0,
            "final_score": 0.3, "label": "low",
            "centrality_score": 0.3, "cluster_id": "C0002",
            "in_graph": True, "correlation_id": None, "is_cross_system": False,
        })
        seq += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Autouse fixture — redirect all parquet saves to tmp_path
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_output_paths(monkeypatch, tmp_path):
    import scoring.root_cause_engine as rce
    monkeypatch.setattr(rce, "_SCORED_PATH", str(tmp_path / "scored_logs_df.parquet"))
    monkeypatch.setattr(rce, "_ROOT_CAUSES_PATH", str(tmp_path / "root_causes_df.parquet"))


# ---------------------------------------------------------------------------
# TestImportanceScorer
# ---------------------------------------------------------------------------

class TestImportanceScorer:

    def test_final_score_in_range(self):
        features_df, anomaly_df, graph_scores_df = build_test_inputs()
        result = score(features_df, anomaly_df, graph_scores_df)
        assert (result["final_score"] >= 0.0).all()
        assert (result["final_score"] <= 1.0).all()

    def test_no_nulls_no_nan_in_outputs(self):
        features_df, anomaly_df, graph_scores_df = build_test_inputs()
        result = score(features_df, anomaly_df, graph_scores_df)
        assert not result["final_score"].isna().any()
        assert not result["sequence_number"].isna().any()
        assert np.isfinite(result["final_score"].to_numpy()).all()

    def test_missing_anomaly_rows_filled_with_mean(self, caplog):
        features_df, anomaly_df, graph_scores_df = build_test_inputs(n_rows=20, n_sessions=2)
        anomaly_partial = anomaly_df.iloc[:15].copy()
        # 25% missing exceeds the systematic-gap cap — raise it to test the fill path
        with patch.object(cfg, "SCORING_MAX_MISSING_FRACTION", 0.5), \
             caplog.at_level(logging.WARNING, logger="scoring.importance_scorer"):
            result = score(features_df, anomaly_partial, graph_scores_df)
        assert not result["combined_score"].isna().any()
        assert "filled with mean" in caplog.text

    def test_missing_graph_rows_filled_with_mean(self, caplog):
        features_df, anomaly_df, graph_scores_df = build_test_inputs(n_rows=20, n_sessions=2)
        graph_partial = graph_scores_df.iloc[:15].copy()
        with patch.object(cfg, "SCORING_MAX_MISSING_FRACTION", 0.5), \
             caplog.at_level(logging.WARNING, logger="scoring.importance_scorer"):
            result = score(features_df, anomaly_df, graph_partial)
        assert not result["centrality_score"].isna().any()
        assert "filled with mean" in caplog.text

    def test_boolean_cols_filled_with_false_not_mean(self):
        features_df, anomaly_df, graph_scores_df = build_test_inputs(n_rows=20, n_sessions=2)
        anomaly_partial = anomaly_df.iloc[:15].copy()
        graph_partial = graph_scores_df.iloc[:15].copy()
        with patch.object(cfg, "SCORING_MAX_MISSING_FRACTION", 0.5):
            result = score(features_df, anomaly_partial, graph_partial)
        missing_anomaly = ~features_df["sequence_number"].isin(anomaly_partial["sequence_number"])
        missing_graph = ~features_df["sequence_number"].isin(graph_partial["sequence_number"])
        assert (result.loc[missing_anomaly, "is_anomaly"] == False).all()
        assert (result.loc[missing_graph, "in_graph"] == False).all()
        assert (result.loc[missing_graph, "in_sequence"] == False).all()

    def test_missing_rows_flagged_in_audit_columns(self):
        features_df, anomaly_df, graph_scores_df = build_test_inputs(n_rows=100, n_sessions=5)
        # Drop 2% from anomaly, 3% from graph — below the 5% cap, no raise
        anomaly_partial = anomaly_df.iloc[2:].copy()
        graph_partial = graph_scores_df.iloc[3:].copy()
        result = score(features_df, anomaly_partial, graph_partial)
        assert int(result["anomaly_missing"].sum()) == 2
        assert int(result["graph_missing"].sum()) == 3
        dropped_anomaly_seqs = set(anomaly_df["sequence_number"].iloc[:2])
        assert set(result.loc[result["anomaly_missing"], "sequence_number"]) == dropped_anomaly_seqs

    def test_complete_inputs_have_no_missing_flags(self):
        features_df, anomaly_df, graph_scores_df = build_test_inputs()
        result = score(features_df, anomaly_df, graph_scores_df)
        assert not result["anomaly_missing"].any()
        assert not result["graph_missing"].any()

    def test_systematic_gap_raises(self):
        features_df, anomaly_df, graph_scores_df = build_test_inputs(n_rows=100, n_sessions=5)
        anomaly_partial = anomaly_df.iloc[50:].copy()   # 50% missing
        with pytest.raises(ValueError, match="missing from anomaly_df"):
            score(features_df, anomaly_partial, graph_scores_df)

    def test_temporal_proximity_in_range_per_session(self):
        features_df, anomaly_df, graph_scores_df = build_test_inputs()
        result = score(features_df, anomaly_df, graph_scores_df)
        assert (result["temporal_proximity"] >= 0.0).all()
        assert (result["temporal_proximity"] <= 1.0).all()
        for _, grp in result.groupby("session_id"):
            assert grp["temporal_proximity"].min() >= 0.0
            assert grp["temporal_proximity"].max() <= 1.0

    def test_single_log_session_temporal_proximity_is_zero(self):
        features_single = pd.DataFrame({
            "sequence_number": [1],
            "session_id": ["S000"],
            "timestamp": [pd.Timestamp("2024-01-01")],
            "template_id": ["T001"], "host": ["host0"],
            "frequency_score": [0.5], "burstiness_score": [0.5],
            "zscore_base": [0.5], "time_delta_prev": [0.0],
            "time_delta_session_start": [0.0], "inter_arrival_rate": [0.5],
            "event_weight": [0.5], "counter_proximity": [0.0],
        })
        anomaly_single = pd.DataFrame({
            "sequence_number": [1], "isolation_score": [0.5],
            "zscore_norm": [0.5], "combined_score": [0.5],
            "is_anomaly": [False], "model_confidence": [0.5],
        })
        graph_single = pd.DataFrame({
            "sequence_number": [1], "centrality_score": [0.5],
            "degree": [2], "betweenness": [0.5], "in_graph": [True],
            "cluster_id": ["C0000"], "in_sequence": [False],
            "correlated_log_ids": [[]],
        })
        result = score(features_single, anomaly_single, graph_single)
        assert result["temporal_proximity"].iloc[0] == pytest.approx(0.0)

    def test_temporal_proximity_column_present(self):
        features_df, anomaly_df, graph_scores_df = build_test_inputs()
        result = score(features_df, anomaly_df, graph_scores_df)
        assert "temporal_proximity" in result.columns


# ---------------------------------------------------------------------------
# TestLabelMapper
# ---------------------------------------------------------------------------

class TestLabelMapper:

    def _df(self, scores: list[float]) -> pd.DataFrame:
        return pd.DataFrame({"final_score": scores})

    def test_ignore_label(self):
        assert map_labels(self._df([0.1]))["label"].iloc[0] == "ignore"

    def test_low_label(self):
        # Midpoints of the configured bands, so threshold retunes don't break these tests
        mid_low = (cfg.LABEL_IGNORE_MAX + cfg.LABEL_LOW_MAX) / 2
        assert map_labels(self._df([mid_low]))["label"].iloc[0] == "low"

    def test_medium_label(self):
        mid_medium = (cfg.LABEL_LOW_MAX + cfg.LABEL_MEDIUM_MAX) / 2
        assert map_labels(self._df([mid_medium]))["label"].iloc[0] == "medium"

    def test_critical_label(self):
        assert map_labels(self._df([0.9]))["label"].iloc[0] == "critical"

    def test_boundary_values(self):
        scores = [cfg.LABEL_IGNORE_MAX, cfg.LABEL_LOW_MAX, cfg.LABEL_MEDIUM_MAX]
        result = map_labels(self._df(scores))
        labels = result["label"].tolist()
        assert labels[0] == "ignore"   # exactly at LABEL_IGNORE_MAX (<=)
        assert labels[1] == "low"      # exactly at LABEL_LOW_MAX (<=)
        assert labels[2] == "medium"   # exactly at LABEL_MEDIUM_MAX (<=)

    def test_no_nulls_after_mapping(self):
        features_df, anomaly_df, graph_scores_df = build_test_inputs()
        scored = score(features_df, anomaly_df, graph_scores_df)
        result = map_labels(scored)
        assert not result["label"].isna().any()


# ---------------------------------------------------------------------------
# TestIncidentClusterer
# ---------------------------------------------------------------------------

class TestIncidentClusterer:

    def test_ignore_rows_get_null_correlation_id(self):
        df = _make_clusterable_df()
        result = cluster_incidents(df)
        ignore_mask = df["label"] == "ignore"
        assert result.loc[ignore_mask, "correlation_id"].isna().all()

    def test_cluster_id_not_corrupted_by_dbscan(self):
        df = _make_clusterable_df()
        result = cluster_incidents(df)
        # cluster_id must remain "C0000"-format strings — DBSCAN integers must never leak in
        assert not pd.api.types.is_integer_dtype(result["cluster_id"])
        assert result["cluster_id"].str.startswith("C").all()

    def test_non_ignore_rows_with_clear_separation_get_incidents(self):
        df = _make_clusterable_df()
        result = cluster_incidents(df)
        non_ignore = result[result["label"] != "ignore"]
        assert non_ignore["correlation_id"].notna().any()

    def test_noise_rows_get_null_correlation_id(self):
        df = _make_clusterable_df()
        result = cluster_incidents(df)
        # The 2 isolated medium rows have fewer than min_samples=5 neighbors → noise
        medium_mask = df["label"] == "medium"
        assert result.loc[medium_mask, "correlation_id"].isna().all()

    def test_is_cross_system_true_for_multiple_cluster_ids(self):
        df = _make_clusterable_df()
        result = cluster_incidents(df)
        # Cluster A rows span C0000 and C0001 → cross-system
        critical_mask = df["label"] == "critical"
        assigned = result.loc[critical_mask, "correlation_id"].dropna().unique()
        assert len(assigned) > 0, "Cluster A rows must form at least one incident"
        incident_rows = result[result["correlation_id"] == assigned[0]]
        assert incident_rows["is_cross_system"].all()

    def test_is_cross_system_false_for_single_system(self):
        df = _make_clusterable_df()
        result = cluster_incidents(df)
        # Cluster B rows all have cluster_id="C0002" → not cross-system
        low_mask = df["label"] == "low"
        assigned = result.loc[low_mask, "correlation_id"].dropna().unique()
        assert len(assigned) > 0, "Cluster B rows must form at least one incident"
        incident_rows = result[result["correlation_id"] == assigned[0]]
        assert not incident_rows["is_cross_system"].any()

    def test_correlation_id_format(self):
        df = _make_clusterable_df()
        result = cluster_incidents(df)
        valid = result["correlation_id"].dropna()
        assert valid.str.match(r"^INC-\d{4}$").all()

    def test_dbscan_label_column_not_in_returned_df(self):
        df = _make_clusterable_df()
        result = cluster_incidents(df)
        assert "_dbscan_label" not in result.columns


# ---------------------------------------------------------------------------
# TestRootCauseEngine
# ---------------------------------------------------------------------------

class TestRootCauseEngine:

    def test_in_graph_preferred_over_not_in_graph(self):
        df = _make_incident_df()
        updated_df, _ = identify_root_causes(df)
        inc_0 = updated_df[updated_df["correlation_id"] == "INC-0000"]
        root_causes = inc_0[inc_0["is_root_cause"]]
        # All selected root causes must be in_graph=True despite in_graph=False
        # rows having higher centrality_score
        assert (root_causes["in_graph"] == True).all()
        not_in_graph = inc_0[~inc_0["in_graph"]]
        assert (not_in_graph["is_root_cause"] == False).all()

    def test_root_cause_top_n_per_cluster(self):
        df = _make_incident_df()
        updated_df, _ = identify_root_causes(df)
        inc_1 = updated_df[updated_df["correlation_id"] == "INC-0001"]
        n_root = int(inc_1["is_root_cause"].sum())
        assert n_root == min(cfg.ROOT_CAUSE_TOP_N, len(inc_1))

    def test_root_cause_confidence_in_range(self):
        df = _make_incident_df()
        updated_df, _ = identify_root_causes(df)
        conf = updated_df["root_cause_confidence"]
        assert (conf >= 0.0).all()
        assert (conf <= 1.0).all()

    def test_max_centrality_zero_equal_confidence_with_warning(self, caplog):
        df = _make_incident_df().copy()
        df.loc[df["correlation_id"] == "INC-0000", "centrality_score"] = 0.0
        with caplog.at_level(logging.WARNING, logger="scoring.root_cause_engine"):
            updated_df, _ = identify_root_causes(df)
        assert "max_centrality=0.0" in caplog.text
        inc_0 = updated_df[updated_df["correlation_id"] == "INC-0000"]
        root_causes = inc_0[inc_0["is_root_cause"]]
        # Equal confidence = 1.0 / n_in_graph_candidates
        n_in_graph = int((inc_0["in_graph"] == True).sum())
        expected = 1.0 / n_in_graph
        actual = root_causes["root_cause_confidence"].unique()
        assert len(actual) == 1
        assert abs(actual[0] - expected) < 1e-9

    def test_all_not_in_graph_fallback_with_warning(self, caplog):
        df = pd.DataFrame({
            "sequence_number": [1, 2, 3, 4, 5],
            "session_id": ["S000"] * 5,
            "timestamp": [pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=i) for i in range(5)],
            "temporal_proximity": [0.0, 0.25, 0.5, 0.75, 1.0],
            "final_score": [0.8] * 5,
            "label": ["critical"] * 5,
            "centrality_score": [0.5, 0.4, 0.3, 0.2, 0.1],
            "cluster_id": ["C0000"] * 5,
            "in_graph": [False] * 5,
            "correlation_id": ["INC-0000"] * 5,
            "is_cross_system": [False] * 5,
        })
        with caplog.at_level(logging.WARNING, logger="scoring.root_cause_engine"):
            updated_df, _ = identify_root_causes(df)
        assert "no in-graph logs found" in caplog.text
        n_root = int(updated_df["is_root_cause"].sum())
        assert n_root == min(cfg.ROOT_CAUSE_TOP_N, len(df))

    def test_non_incident_rows_not_root_cause(self):
        df = _make_incident_df()
        updated_df, _ = identify_root_causes(df)
        noise = updated_df[updated_df["correlation_id"].isna()]
        assert (noise["is_root_cause"] == False).all()
        assert (noise["root_cause_confidence"] == 0.0).all()

    def test_temporal_proximity_not_in_scored_logs_parquet(self, tmp_path):
        df = _make_incident_df()
        assert "temporal_proximity" in df.columns
        identify_root_causes(df)
        saved = pd.read_parquet(tmp_path / "scored_logs_df.parquet")
        assert "temporal_proximity" not in saved.columns

    def test_root_causes_df_has_correct_schema(self):
        df = _make_incident_df()
        _, root_causes_df = identify_root_causes(df)
        required = {"incident_id", "root_cause_log_id", "confidence_score", "in_graph"}
        assert required.issubset(set(root_causes_df.columns))

    def test_both_parquets_reload_with_correct_schema(self, tmp_path):
        df = _make_incident_df()
        identify_root_causes(df)
        scored = pd.read_parquet(tmp_path / "scored_logs_df.parquet")
        root = pd.read_parquet(tmp_path / "root_causes_df.parquet")
        for col in ("sequence_number", "final_score", "label", "correlation_id",
                    "is_root_cause", "root_cause_confidence", "is_cross_system"):
            assert col in scored.columns, f"scored_logs_df missing column: {col}"
        for col in ("incident_id", "root_cause_log_id", "confidence_score", "in_graph"):
            assert col in root.columns, f"root_causes_df missing column: {col}"
