"""
evaluation/tests/test_oracle_report.py
======================================
Oracle evaluation harness tests.

Uses tiny hand-computable parquet fixtures so every expected metric value
can be verified by inspection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.oracle_report import run_oracle_report


def _write_fixtures(
    tmp_path,
    *,
    with_scenario: bool = True,
    drop_anomaly_rows: int = 0,
):
    """Write a 10-row fixture set; returns the four parquet paths.

    Layout (sequence_number 1..10):
      rows 1-3   CRITICAL  (signal)   scenario S1
      rows 4-10  INFO      (noise)    rows 4-6 scenario S1, rows 7-10 S2
      is_anomaly True for rows 1, 2, 4   →  tp=2 fp=1 fn=1
      final_score descending with sequence_number (row 1 highest)
      label: rows 1-2 "critical", row 3 "low", rows 4-5 "ignore", rest "low"
      correlation_id: rows 1-2 "INC-0001", others None
    """
    n = 10
    session = pd.DataFrame({
        "sequence_number": np.arange(1, n + 1),
        "log_level": ["CRITICAL"] * 3 + ["INFO"] * 7,
    })
    if with_scenario:
        session["scenario_id"] = ["S1"] * 6 + ["S2"] * 4

    anomaly = pd.DataFrame({
        "sequence_number": np.arange(1, n + 1),
        "is_anomaly": [True, True, False, True] + [False] * 6,
        "combined_score": np.linspace(1.0, 0.1, n),
    })
    if drop_anomaly_rows:
        anomaly = anomaly.iloc[drop_anomaly_rows:]

    scored = pd.DataFrame({
        "sequence_number": np.arange(1, n + 1),
        "final_score": np.linspace(1.0, 0.1, n),
        "label": ["critical", "critical", "low", "ignore", "ignore"] + ["low"] * 5,
        "correlation_id": ["INC-0001", "INC-0001"] + [None] * 8,
    })

    labels = pd.DataFrame({
        "scenario_id": ["S1", "S2"],
        "source_file": ["s1.log", "s2.log"],
        "training_label": ["CRITICAL_X", "HIGH_Y"],
    })

    paths = {}
    for name, df in [
        ("session", session), ("anomaly", anomaly),
        ("scored", scored), ("labels", labels),
    ]:
        p = tmp_path / f"{name}.parquet"
        df.to_parquet(p, index=False)
        paths[name] = str(p)
    return paths


def _run(tmp_path, paths) -> dict:
    return run_oracle_report(
        scored_path=paths["scored"],
        anomaly_path=paths["anomaly"],
        sessionized_path=paths["session"],
        labels_path=paths["labels"],
        output_path=str(tmp_path / "results" / "oracle_report.txt"),
    )


class TestAnomalyStageMetrics:
    def test_truth_counts(self, tmp_path):
        metrics = _run(tmp_path, _write_fixtures(tmp_path))
        assert metrics["total_logs"] == 10
        assert metrics["truth_signal_count"] == 3
        assert metrics["truth_signal_rate"] == pytest.approx(0.3)

    def test_precision_recall_f1(self, tmp_path):
        metrics = _run(tmp_path, _write_fixtures(tmp_path))
        # tp=2 (rows 1,2), fp=1 (row 4), fn=1 (row 3)
        assert metrics["anomaly_tp"] == 2
        assert metrics["anomaly_fp"] == 1
        assert metrics["anomaly_fn"] == 1
        assert metrics["anomaly_precision"] == pytest.approx(2 / 3)
        assert metrics["anomaly_recall"] == pytest.approx(2 / 3)
        assert metrics["anomaly_f1"] == pytest.approx(2 / 3)

    def test_unscored_rows_count_as_misses(self, tmp_path):
        # Drop rows 1-2 from anomaly_df: their is_anomaly becomes NaN→False,
        # so tp falls to 0 and fn rises to 3.
        paths = _write_fixtures(tmp_path, drop_anomaly_rows=2)
        metrics = _run(tmp_path, paths)
        assert metrics["unscored_rows"] == 2
        assert metrics["anomaly_tp"] == 0
        assert metrics["anomaly_fn"] == 3


class TestRankingAndLabelMetrics:
    def test_recall_at_k(self, tmp_path):
        # k=3; top-3 by final_score are rows 1-3, all signal → 1.0
        metrics = _run(tmp_path, _write_fixtures(tmp_path))
        assert metrics["ranking_recall_at_k"] == pytest.approx(1.0)
        # top-3 by combined_score are also rows 1-3 in the fixture
        assert metrics["ranking_recall_at_k_ml"] == pytest.approx(1.0)

    def test_score_separation_positive(self, tmp_path):
        metrics = _run(tmp_path, _write_fixtures(tmp_path))
        assert metrics["score_separation"] == pytest.approx(
            metrics["mean_final_score_signal"] - metrics["mean_final_score_noise"]
        )
        assert metrics["score_separation"] > 0

    def test_capture_and_suppression(self, tmp_path):
        metrics = _run(tmp_path, _write_fixtures(tmp_path))
        # 2 of 3 signal rows labelled critical/medium
        assert metrics["critical_capture_rate"] == pytest.approx(2 / 3)
        # 2 of 7 noise rows labelled ignore
        assert metrics["noise_suppression_ratio"] == pytest.approx(2 / 7)

    def test_incident_coverage(self, tmp_path):
        metrics = _run(tmp_path, _write_fixtures(tmp_path))
        # 2 of 3 signal rows carry a correlation_id
        assert metrics["signal_incident_coverage"] == pytest.approx(2 / 3)


class TestScenarioBreakdown:
    def test_per_scenario_detection(self, tmp_path):
        metrics = _run(tmp_path, _write_fixtures(tmp_path))
        by_id = {s["scenario_id"]: s for s in metrics["per_scenario"]}
        assert by_id["S1"]["n_signal"] == 3
        assert by_id["S1"]["n_signal_flagged"] == 2
        assert by_id["S1"]["detected"] is True
        assert by_id["S1"]["training_label"] == "CRITICAL_X"
        # S2 has no signal rows → not counted as detectable
        assert by_id["S2"]["n_signal"] == 0
        assert metrics["scenario_detection_rate"] == pytest.approx(1.0)

    def test_legacy_path_without_scenario_column(self, tmp_path):
        paths = _write_fixtures(tmp_path, with_scenario=False)
        metrics = _run(tmp_path, paths)
        assert metrics["per_scenario"] == []
        assert "scenario_detection_rate" not in metrics
        # Log-level metrics still computed
        assert metrics["anomaly_tp"] == 2


class TestReportFile:
    def test_report_written_and_readable(self, tmp_path):
        _run(tmp_path, _write_fixtures(tmp_path))
        report = (tmp_path / "results" / "oracle_report.txt").read_text()
        assert "ORACLE EVALUATION REPORT" in report
        assert "anomaly_precision" in report
        assert "Per-scenario breakdown:" in report
        assert "CRITICAL_X" in report
