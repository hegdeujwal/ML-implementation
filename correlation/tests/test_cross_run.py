"""
correlation/tests/test_cross_run.py

Unit tests for the P5.5 cross-run incident correlation modules:
  - correlation.fingerprint
  - correlation.chain_builder
  - correlation.precursor_elevator

Run with:
    pytest correlation/tests/test_cross_run.py -v
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta

import pandas as pd
import pytest

from correlation.fingerprint import (
    fingerprint_from_df,
    fingerprint_from_list,
    fingerprint_to_json,
    jaccard,
)
from correlation.chain_builder import assign_chains, _history_columns
from correlation.precursor_elevator import elevate_precursor_scores, elevate_log_scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(offset_hours: int = 0) -> pd.Timestamp:
    base = datetime(2026, 3, 15, 6, 0, 0, tzinfo=timezone.utc)
    return pd.Timestamp(base + timedelta(hours=offset_hours))


def _make_history(**kwargs) -> pd.DataFrame:
    """Build a minimal incident_history DataFrame for testing."""
    defaults = {
        "incident_id":           "INC-20260314-0000",
        "run_date":              "20260314",
        "run_timestamp":         _ts(-24),
        "start_time":            _ts(-25),
        "end_time":              _ts(-24),
        "template_fingerprint":  json.dumps(["T001", "T002", "T003"]),
        "root_cause_templates":  json.dumps(["T001"]),
        "severity":              "medium",
        "log_count":             50,
        "hosts":                 json.dumps(["switch-01"]),
        "is_cross_system":       False,
        "chain_id":              None,
        "precursor_incident_id": None,
        "chain_position":        None,
        "is_precursor_elevated": False,
    }
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


def _make_incident(
    global_id: str = "INC-20260315-0000",
    local_id: str = "INC-0000",
    templates: list[str] | None = None,
    start_offset: int = 0,
    end_offset: int = 1,
) -> dict:
    if templates is None:
        templates = ["T001", "T002"]
    return {
        "global_incident_id":  global_id,
        "local_incident_id":   local_id,
        "run_date":            "20260315",
        "run_timestamp":       _ts(0),
        "start_time":          _ts(start_offset),
        "end_time":            _ts(end_offset),
        "template_fingerprint": frozenset(templates),
        "root_cause_templates": ["T001"],
        "severity":            "critical",
        "log_count":           100,
        "hosts":               ["switch-01"],
        "is_cross_system":     True,
    }


# ---------------------------------------------------------------------------
# fingerprint tests
# ---------------------------------------------------------------------------

class TestFingerprint:

    def test_fingerprint_from_df_returns_unique_templates(self):
        df = pd.DataFrame({"template_id": ["T001", "T002", "T001", "T003"]})
        fp = fingerprint_from_df(df)
        assert fp == frozenset({"T001", "T002", "T003"})

    def test_fingerprint_from_df_ignores_nulls(self):
        df = pd.DataFrame({"template_id": ["T001", None, "T002"]})
        fp = fingerprint_from_df(df)
        assert None not in fp
        assert "T001" in fp

    def test_fingerprint_from_list_json_string(self):
        fp = fingerprint_from_list('["T001", "T002"]')
        assert fp == frozenset({"T001", "T002"})

    def test_fingerprint_from_list_python_list(self):
        fp = fingerprint_from_list(["T001", "T002"])
        assert fp == frozenset({"T001", "T002"})

    def test_fingerprint_from_list_none(self):
        assert fingerprint_from_list(None) == frozenset()

    def test_fingerprint_from_list_empty_json(self):
        assert fingerprint_from_list("[]") == frozenset()

    def test_fingerprint_to_json_is_sorted(self):
        fp = frozenset({"T003", "T001", "T002"})
        result = json.loads(fingerprint_to_json(fp))
        assert result == sorted(result)

    def test_jaccard_identical_sets(self):
        a = frozenset({"T001", "T002"})
        assert jaccard(a, a) == pytest.approx(1.0)

    def test_jaccard_disjoint_sets(self):
        a = frozenset({"T001"})
        b = frozenset({"T002"})
        assert jaccard(a, b) == pytest.approx(0.0)

    def test_jaccard_partial_overlap(self):
        a = frozenset({"T001", "T002"})
        b = frozenset({"T002", "T003"})
        # intersection={T002}, union={T001,T002,T003} → 1/3
        assert jaccard(a, b) == pytest.approx(1 / 3)

    def test_jaccard_both_empty(self):
        assert jaccard(frozenset(), frozenset()) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# chain_builder tests
# ---------------------------------------------------------------------------

class TestChainBuilder:

    def test_no_history_returns_no_chain(self):
        inc = [_make_incident()]
        results, _ = assign_chains(inc, pd.DataFrame(), threshold=0.3, lookback_hours=72)
        assert results[0]["chain_id"] is None
        assert results[0]["chain_position"] == 1
        assert results[0]["chain_confidence"] == pytest.approx(0.0)

    def test_low_similarity_below_threshold_returns_no_chain(self):
        """When Jaccard similarity is below threshold, no chain is assigned."""
        history = _make_history(
            template_fingerprint=json.dumps(["X001", "X002", "X003"])
        )
        inc = [_make_incident(templates=["T001", "T002"])]  # no overlap
        results, _ = assign_chains(inc, history, threshold=0.3, lookback_hours=72)
        assert results[0]["chain_id"] is None

    def test_high_similarity_creates_chain(self):
        """Jaccard >= threshold links incident to a new or existing chain."""
        history = _make_history(
            template_fingerprint=json.dumps(["T001", "T002", "T003"])
        )
        inc = [_make_incident(templates=["T001", "T002", "T003"])]  # identical
        results, updated_hist = assign_chains(inc, history, threshold=0.3, lookback_hours=72)
        assert results[0]["chain_id"] is not None
        assert results[0]["precursor_incident_id"] == "INC-20260314-0000"
        assert results[0]["chain_position"] == 2
        assert results[0]["chain_confidence"] == pytest.approx(1.0)

    def test_chain_inherits_existing_chain_id(self):
        """If the matching historical incident already has a chain_id, inherit it."""
        history = _make_history(
            template_fingerprint=json.dumps(["T001", "T002"]),
            chain_id="CHAIN-999-0001",
            chain_position=1,
        )
        inc = [_make_incident(templates=["T001", "T002"])]
        results, _ = assign_chains(inc, history, threshold=0.3, lookback_hours=72)
        assert results[0]["chain_id"] == "CHAIN-999-0001"

    def test_chain_fan_out_merges_two_chains(self):
        """When incident matches two different historical chains, they are merged."""
        h1 = _make_history(
            incident_id="INC-A",
            template_fingerprint=json.dumps(["T001", "T002"]),
            chain_id="CHAIN-A",
            chain_position=1,
            end_time=_ts(-48),
        )
        h2 = _make_history(
            incident_id="INC-B",
            template_fingerprint=json.dumps(["T002", "T003"]),
            chain_id="CHAIN-B",
            chain_position=1,
            end_time=_ts(-24),
        )
        history = pd.concat([h1, h2], ignore_index=True)
        history["end_time"] = pd.to_datetime(history["end_time"], utc=True)

        inc = [_make_incident(templates=["T001", "T002", "T003"])]
        results, updated_hist = assign_chains(inc, history, threshold=0.1, lookback_hours=72)

        # Both CHAIN-A and CHAIN-B should now be the same canonical chain
        chain_ids = updated_hist["chain_id"].dropna().unique()
        assert len(chain_ids) == 1, f"Expected 1 chain after merge, got {chain_ids}"

    def test_lookback_window_excludes_old_incidents(self):
        """Incidents outside the lookback window should not be matched."""
        history = _make_history(
            template_fingerprint=json.dumps(["T001", "T002"]),
            end_time=_ts(-100),  # 100 hours ago — outside 72h window
        )
        history["end_time"] = pd.to_datetime(history["end_time"], utc=True)
        inc = [_make_incident(templates=["T001", "T002"])]
        results, _ = assign_chains(inc, history, threshold=0.3, lookback_hours=72)
        assert results[0]["chain_id"] is None

    def test_chain_position_increments(self):
        """chain_position should be max(existing)+1 for follow-on incidents."""
        history = _make_history(
            template_fingerprint=json.dumps(["T001", "T002"]),
            chain_id="CHAIN-X",
            chain_position=3,
        )
        history["end_time"] = pd.to_datetime(history["end_time"], utc=True)
        inc = [_make_incident(templates=["T001", "T002"])]
        results, _ = assign_chains(inc, history, threshold=0.3, lookback_hours=72)
        assert results[0]["chain_position"] == 4


# ---------------------------------------------------------------------------
# precursor_elevator tests
# ---------------------------------------------------------------------------

class TestPrecursorElevator:

    def test_elevate_marks_history_row(self):
        history = _make_history(incident_id="INC-20260314-0000")
        chain_results = [{
            "global_incident_id": "INC-20260315-0000",
            "precursor_incident_id": "INC-20260314-0000",
            "chain_confidence": 0.8,
        }]
        updated = elevate_precursor_scores(history, chain_results, boost=0.15)
        assert bool(updated.loc[
            updated["incident_id"] == "INC-20260314-0000", "is_precursor_elevated"
        ].iloc[0])

    def test_elevate_ignores_null_precursor(self):
        history = _make_history()
        chain_results = [{
            "global_incident_id": "INC-20260315-0000",
            "precursor_incident_id": None,
            "chain_confidence": 0.0,
        }]
        updated = elevate_precursor_scores(history, chain_results, boost=0.15)
        assert not updated["is_precursor_elevated"].any()

    def test_elevate_empty_history_returns_unchanged(self):
        result = elevate_precursor_scores(pd.DataFrame(), [], boost=0.15)
        assert len(result) == 0

    def test_elevate_log_scores_boosts_correct_rows(self):
        scored = pd.DataFrame({
            "sequence_number": [1, 2, 3, 4],
            "correlation_id": ["INC-0000", "INC-0000", "INC-0001", None],
            "final_score": [0.4, 0.3, 0.8, 0.1],
            "label": ["low", "low", "critical", "ignore"],
        })
        result = elevate_log_scores(
            scored,
            precursor_correlation_ids={"INC-0000"},
            chain_confidence=1.0,
            boost=0.15,
        )
        assert result.loc[0, "final_score"] == pytest.approx(0.4 + 0.15, abs=1e-6)
        assert result.loc[1, "final_score"] == pytest.approx(0.3 + 0.15, abs=1e-6)
        # INC-0001 and None rows unchanged
        assert result.loc[2, "final_score"] == pytest.approx(0.8)
        assert result.loc[3, "final_score"] == pytest.approx(0.1)

    def test_elevate_log_scores_clips_to_one(self):
        scored = pd.DataFrame({
            "sequence_number": [1],
            "correlation_id": ["INC-0000"],
            "final_score": [0.95],
            "label": ["critical"],
        })
        result = elevate_log_scores(scored, {"INC-0000"}, chain_confidence=1.0, boost=0.15)
        assert result.loc[0, "final_score"] == pytest.approx(1.0)

    def test_elevate_log_scores_updates_label(self):
        scored = pd.DataFrame({
            "sequence_number": [1],
            "correlation_id": ["INC-0000"],
            "final_score": [0.45],  # currently "low"
            "label": ["low"],
        })
        result = elevate_log_scores(scored, {"INC-0000"}, chain_confidence=1.0, boost=0.15)
        # 0.45 + 0.15 = 0.60 → medium
        assert result.loc[0, "label"] == "medium"

    def test_elevate_log_scores_sets_is_precursor_elevated(self):
        scored = pd.DataFrame({
            "sequence_number": [1, 2],
            "correlation_id": ["INC-0000", "INC-0001"],
            "final_score": [0.3, 0.8],
            "label": ["low", "critical"],
        })
        result = elevate_log_scores(scored, {"INC-0000"}, chain_confidence=0.5, boost=0.15)
        assert bool(result.loc[0, "is_precursor_elevated"])
        assert not bool(result.loc[1, "is_precursor_elevated"])

    def test_elevate_log_scores_zero_confidence_no_boost(self):
        scored = pd.DataFrame({
            "sequence_number": [1],
            "correlation_id": ["INC-0000"],
            "final_score": [0.3],
            "label": ["low"],
        })
        result = elevate_log_scores(scored, {"INC-0000"}, chain_confidence=0.0, boost=0.15)
        # chain_confidence=0.0 → no elevation
        assert result.loc[0, "final_score"] == pytest.approx(0.3)
