"""
Unit tests for feature engineering modules.

Covers statistical features, temporal features, counter proximity, and the
full feature pipeline.  All synthetic DataFrames use canonical schema columns.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from features.statistical_features import burstiness_score, log_frequency_score, zscore_base
from features.temporal_features import inter_arrival_rate, time_delta_prev, time_delta_session_start
from features.counter_proximity import compute_counter_proximity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(
    n: int,
    *,
    host: str = "sw-01",
    session_id: str = "s1",
    template_id: str = "TMPL_A",
    frequency: int = 1,
    log_level: str = "info",
    event_weight: float = 0.1,
    start: datetime | None = None,
    step_seconds: float = 5.0,
) -> pd.DataFrame:
    """Build a minimal per-session DataFrame with canonical columns."""
    if start is None:
        start = datetime(2024, 1, 1, 0, 0, 0)
    rows = []
    for i in range(n):
        rows.append({
            "sequence_number": i + 1,
            "timestamp": start + timedelta(seconds=step_seconds * i),
            "session_id": session_id,
            "template_id": template_id,
            "frequency": frequency,
            "host": host,
            "log_level": log_level,
            "event_weight": event_weight,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# statistical_features — log_frequency_score
# ---------------------------------------------------------------------------

class TestLogFrequencyScore:
    def test_half_max(self):
        df = _make_session(3, frequency=4)
        df.loc[df.index[2], "frequency"] = 2
        result = log_frequency_score(df)
        assert isinstance(result, pd.Series)
        assert result.max() == pytest.approx(1.0)
        assert result.iloc[2] == pytest.approx(0.5)

    def test_single_log_is_one(self):
        df = _make_session(1, frequency=7)
        result = log_frequency_score(df)
        assert result.iloc[0] == pytest.approx(1.0)

    def test_zero_max_returns_zeros(self):
        df = _make_session(3, frequency=0)
        result = log_frequency_score(df)
        assert (result == 0.0).all()


# ---------------------------------------------------------------------------
# statistical_features — burstiness_score
# ---------------------------------------------------------------------------

class TestBurstinessScore:
    def test_single_log_returns_zero(self):
        df = _make_session(1)
        result = burstiness_score(df)
        assert result.iloc[0] == pytest.approx(0.0)

    def test_uniform_arrivals_low_fano(self):
        # Equal spacing → variance=0 → Fano=0
        df = _make_session(5, step_seconds=10.0)
        result = burstiness_score(df)
        assert (result == 0.0).all()

    def test_bursty_arrivals_higher_than_uniform(self):
        start = datetime(2024, 1, 1)
        df = pd.DataFrame({
            "sequence_number": range(1, 6),
            "timestamp": [
                start,
                start + timedelta(seconds=1),
                start + timedelta(seconds=2),
                start + timedelta(seconds=100),
                start + timedelta(seconds=101),
            ],
            "session_id": "s1",
            "template_id": "T",
            "frequency": 1,
            "host": "sw-01",
            "log_level": "info",
            "event_weight": 0.1,
        })
        uniform_df = _make_session(5, step_seconds=20.0)
        bursty_val = burstiness_score(df).iloc[0]
        uniform_val = burstiness_score(uniform_df).iloc[0]
        assert bursty_val > uniform_val

    def test_clip_to_ten(self):
        # Extremely bursty → raw Fano > 10 → clipped to 10
        start = datetime(2024, 1, 1)
        ts = [start, start + timedelta(seconds=1)] + [
            start + timedelta(seconds=10000 * (i + 1)) for i in range(8)
        ]
        df = pd.DataFrame({
            "sequence_number": range(1, 11),
            "timestamp": ts,
            "session_id": "s1",
            "template_id": "T",
            "frequency": 1,
            "host": "sw-01",
            "log_level": "info",
            "event_weight": 0.1,
        })
        result = burstiness_score(df)
        assert result.iloc[0] <= 10.0


# ---------------------------------------------------------------------------
# statistical_features — zscore_base
# ---------------------------------------------------------------------------

class TestZscoreBase:
    def _multi_session_df(
        self, n_sessions: int, *, host: str = "sw-01", template_id: str = "T_A", freq: int = 5
    ) -> pd.DataFrame:
        frames = []
        base = datetime(2024, 1, 1)
        for i in range(n_sessions):
            start = base + timedelta(hours=i)
            frames.append(_make_session(
                3,
                host=host,
                session_id=f"s{i}",
                template_id=template_id,
                frequency=freq,
                start=start,
            ))
        return pd.concat(frames, ignore_index=True)

    def test_few_sessions_returns_zero(self):
        df = self._multi_session_df(1)
        result = zscore_base(df, n_sessions=20)
        assert (result == 0.0).all()

    def test_stable_baseline_near_zero(self):
        df = self._multi_session_df(25, freq=10)
        result = zscore_base(df, n_sessions=20)
        # Last session has same freq as all history → z ≈ 0
        last_session = df["session_id"].unique()[-1]
        last_mask = df["session_id"] == last_session
        assert result[last_mask].iloc[0] == pytest.approx(0.0, abs=0.1)

    def test_spike_session_positive_zscore(self):
        frames = []
        base = datetime(2024, 1, 1)
        for i in range(20):
            start = base + timedelta(hours=i)
            # Alternate freq so history has non-zero std (required for z-score to fire)
            freq = 3 if i % 2 == 0 else 7
            frames.append(_make_session(2, host="sw-01", session_id=f"s{i}",
                                        template_id="T_X", frequency=freq, start=start))
        # spike session
        frames.append(_make_session(2, host="sw-01", session_id="s_spike",
                                    template_id="T_X", frequency=500,
                                    start=base + timedelta(hours=20)))
        df = pd.concat(frames, ignore_index=True)
        result = zscore_base(df, n_sessions=20)
        spike_mask = df["session_id"] == "s_spike"
        assert result[spike_mask].iloc[0] > 1.0

    def test_host_isolation(self):
        base = datetime(2024, 1, 1)
        # Host A: stable template T_A at freq=5
        frames_a = [
            _make_session(2, host="sw-A", session_id=f"sA{i}", template_id="T_A",
                          frequency=5, start=base + timedelta(hours=i))
            for i in range(22)
        ]
        # Host B: same template T_A but spike — should not bleed into Host A
        frames_b = [
            _make_session(2, host="sw-B", session_id=f"sB{i}", template_id="T_A",
                          frequency=500, start=base + timedelta(hours=i))
            for i in range(22)
        ]
        df = pd.concat(frames_a + frames_b, ignore_index=True)
        result = zscore_base(df, n_sessions=20)
        # Host A last session should still have z ≈ 0
        last_a = df[(df["host"] == "sw-A") & (df["session_id"] == "sA21")]
        assert result[last_a.index].iloc[0] == pytest.approx(0.0, abs=0.1)


# ---------------------------------------------------------------------------
# statistical_features — zscore_base_persistent (Welford store)
# ---------------------------------------------------------------------------

class TestZscorePersistent:
    """Pre-update scoring, leave-one-out re-runs, and bounded seen-ID store.

    Fixture sequence (one host, one template, chronological sessions):
        freqs = [10, 12, 8, 30]
    Hand-computed expectations against the PRIOR baseline:
        s0: no history                        → z = 0
        s1: 1 prior obs, no std               → z = 0
        s2: baseline [10, 12]   mean=11 std≈1.414 → z = (8-11)/1.414 ≈ -2.121
        s3: baseline [10, 12, 8] mean=10 std=2    → z = (30-10)/2 = 10 → clip 5.0
    (The old post-update scoring folded the spike into its own baseline:
    mean=15 std≈10.1 → z≈1.48 — the spike shrank its own score.)
    """

    FREQS = [10, 12, 8, 30]

    def _df(self) -> pd.DataFrame:
        base = datetime(2024, 1, 1)
        frames = [
            _make_session(2, host="sw-01", session_id=f"s{i}", template_id="T_W",
                          frequency=f, start=base + timedelta(hours=i))
            for i, f in enumerate(self.FREQS)
        ]
        return pd.concat(frames, ignore_index=True)

    def _run(self, tmp_path, df=None):
        from features.statistical_features import zscore_base_persistent
        return zscore_base_persistent(
            df if df is not None else self._df(),
            str(tmp_path / "welford.parquet"),
        )

    def test_spike_scored_against_prior_baseline(self, tmp_path):
        df = self._df()
        result = self._run(tmp_path, df)
        z_by_session = {
            sid: result[df["session_id"] == sid].iloc[0]
            for sid in ("s0", "s1", "s2", "s3")
        }
        assert z_by_session["s0"] == pytest.approx(0.0)
        assert z_by_session["s1"] == pytest.approx(0.0)
        assert z_by_session["s2"] == pytest.approx(-2.1213, abs=1e-3)
        # Spike clipped at +5; post-update scoring would have given ~1.48
        assert z_by_session["s3"] == pytest.approx(5.0)

    def test_rerun_reproduces_scores_exactly(self, tmp_path):
        """Leave-one-out on a seen session must equal the original pre-update z."""
        df = self._df()
        first = self._run(tmp_path, df)
        second = self._run(tmp_path, df)
        pd.testing.assert_series_equal(first, second)

    def test_rerun_does_not_double_count(self, tmp_path):
        self._run(tmp_path)
        self._run(tmp_path)
        store = pd.read_parquet(tmp_path / "welford.parquet")
        assert store["n"].iloc[0] == len(self.FREQS)

    def test_seen_ids_capped(self, tmp_path):
        from features.statistical_features import zscore_base_persistent
        import json
        df = self._df()
        zscore_base_persistent(df, str(tmp_path / "welford.parquet"), seen_cap=2)
        store = pd.read_parquet(tmp_path / "welford.parquet")
        seen = json.loads(store["seen_session_ids"].iloc[0])
        assert len(seen) == 2
        # Oldest evicted first — the two newest session IDs remain,
        # each mapped to its originally computed z-score
        assert sorted(seen) == ["s2", "s3"]
        assert seen["s3"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# temporal_features
# ---------------------------------------------------------------------------

class TestTemporalFeatures:
    def test_time_delta_prev_first_row_zero(self):
        df = _make_session(3, step_seconds=10.0)
        result = time_delta_prev(df)
        assert result.iloc[0] == pytest.approx(0.0)
        assert result.iloc[1] == pytest.approx(10.0)
        assert result.iloc[2] == pytest.approx(10.0)

    def test_time_delta_session_start(self):
        df = _make_session(4, step_seconds=5.0)
        result = time_delta_session_start(df)
        assert result.iloc[0] == pytest.approx(0.0)
        assert result.iloc[3] == pytest.approx(15.0)

    def test_inter_arrival_rate_single_log(self):
        df = _make_session(1)
        result = inter_arrival_rate(df)
        assert result.iloc[0] == pytest.approx(0.0)

    def test_all_three_out_of_order(self):
        start = datetime(2024, 1, 1)
        df = pd.DataFrame({
            "sequence_number": [1, 2, 3],
            "timestamp": [
                start + timedelta(seconds=20),
                start,
                start + timedelta(seconds=10),
            ],
            "session_id": "s1",
            "template_id": "T",
            "frequency": 1,
            "host": "sw-01",
            "log_level": "info",
            "event_weight": 0.1,
        })
        # All three functions should sort internally; first chronological row → 0.0
        tdp = time_delta_prev(df)
        # After sorting: t=0, t=10, t=20 → deltas 0, 10, 10
        sorted_idx = df["timestamp"].sort_values().index
        assert tdp[sorted_idx[0]] == pytest.approx(0.0)
        assert tdp[sorted_idx[1]] == pytest.approx(10.0)

        tds = time_delta_session_start(df)
        assert tds[sorted_idx[0]] == pytest.approx(0.0)
        assert tds[sorted_idx[2]] == pytest.approx(20.0)

        iar = inter_arrival_rate(df)
        assert iar[sorted_idx[0]] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# counter_proximity
# ---------------------------------------------------------------------------

COUNTER_TMPL = "INTERFACE_BANDWIDTH_THRESHOLD"   # matches r"INTERFACE_.*THRESHOLD"
HINT_TMPL    = "PORT_ERROR_THRESHOLD"             # contains "THRESHOLD" but no full pattern match


def _make_full_df(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame with all required canonical columns."""
    defaults = {
        "sequence_number": 0,
        "timestamp": datetime(2024, 1, 1),
        "session_id": "s1",
        "template_id": "NORMAL_LOG",
        "frequency": 1,
        "host": "sw-01",
        "log_level": "info",
        "event_weight": 0.1,
    }
    records = []
    for i, row in enumerate(rows):
        r = {**defaults, "sequence_number": i + 1, **row}
        records.append(r)
    return pd.DataFrame(records)


class TestCounterProximity:
    def test_within_window_nonzero(self):
        base = datetime(2024, 1, 1, 0, 0, 0)
        df = _make_full_df([
            {"timestamp": base, "template_id": COUNTER_TMPL},
            {"timestamp": base + timedelta(seconds=5), "template_id": "NORMAL"},
        ])
        result = compute_counter_proximity(df)
        assert result.iloc[1] > 0.0
        assert result.iloc[1] <= 1.0

    def test_coincident_event_score_one(self):
        base = datetime(2024, 1, 1)
        df = _make_full_df([
            {"timestamp": base, "template_id": COUNTER_TMPL},
            {"timestamp": base, "template_id": "NORMAL"},
        ])
        result = compute_counter_proximity(df)
        # Distance = 0 → 1/(1+0) = 1.0
        assert result.iloc[1] == pytest.approx(1.0)

    def test_different_host_no_bleed(self):
        base = datetime(2024, 1, 1)
        df = _make_full_df([
            {"timestamp": base, "template_id": COUNTER_TMPL, "host": "sw-A"},
            {"timestamp": base + timedelta(seconds=1), "template_id": "NORMAL", "host": "sw-B"},
        ])
        result = compute_counter_proximity(df)
        # Counter event on sw-A should not affect sw-B row
        assert result.iloc[1] == pytest.approx(0.0)

    def test_no_counter_events_all_zero(self):
        df = _make_full_df([
            {"template_id": "NORMAL_A"},
            {"template_id": "NORMAL_B"},
        ])
        result = compute_counter_proximity(df)
        assert (result == 0.0).all()

    def test_hint_warning_logged_once(self, caplog):
        import logging
        df = _make_full_df([
            {"template_id": HINT_TMPL},
            {"template_id": HINT_TMPL},
            {"template_id": "OTHER"},
        ])
        with caplog.at_level(logging.WARNING, logger="features.counter_proximity"):
            compute_counter_proximity(df)
        # Only one WARNING for the hint template, not one per row
        warnings = [r for r in caplog.records if HINT_TMPL in r.message]
        assert len(warnings) == 1

    def test_performance_100k_rows(self):
        rng = np.random.default_rng(42)
        n = 100_000
        base = datetime(2024, 1, 1)
        hosts = [f"sw-{i:02d}" for i in range(5)]
        timestamps = [base + timedelta(seconds=float(s)) for s in rng.integers(0, 86400, n)]
        template_ids = np.where(
            rng.random(n) < 0.01,
            COUNTER_TMPL,
            "NORMAL_LOG",
        )
        df = pd.DataFrame({
            "sequence_number": np.arange(1, n + 1),
            "timestamp": timestamps,
            "session_id": "s1",
            "template_id": template_ids,
            "frequency": 1,
            "host": [hosts[i % 5] for i in range(n)],
            "log_level": "info",
            "event_weight": 0.1,
        })
        start = time.perf_counter()
        result = compute_counter_proximity(df)
        elapsed = time.perf_counter() - start
        assert len(result) == n
        assert elapsed < 30.0, f"counter_proximity took {elapsed:.1f}s on 100k rows (limit 30s)"


# ---------------------------------------------------------------------------
# feature_pipeline
# ---------------------------------------------------------------------------

class TestFeaturePipeline:
    def _make_pipeline_input(self, n_sessions: int = 5, rows_per: int = 10) -> pd.DataFrame:
        frames = []
        base = datetime(2024, 1, 1)
        seq = 1
        for i in range(n_sessions):
            start = base + timedelta(hours=i)
            for j in range(rows_per):
                frames.append({
                    "sequence_number": seq,
                    "timestamp": start + timedelta(seconds=j * 5),
                    "session_id": f"s{i}",
                    "template_id": f"TMPL_{j % 3}",
                    "frequency": j + 1,
                    "host": "sw-01",
                    "log_level": "info",
                    "event_weight": 0.1,
                })
                seq += 1
        return pd.DataFrame(frames)

    def test_valid_schema_returns_feature_columns(self, tmp_path):
        from unittest.mock import patch
        from features.feature_pipeline import run_pipeline
        from common.config import FEATURE_COLUMNS
        from common.utils import save_parquet

        input_df = self._make_pipeline_input()
        input_path = str(tmp_path / "sessionized_logs.parquet")
        input_df.to_parquet(input_path, index=False)

        with patch("features.feature_pipeline.save_parquet"):
            out = run_pipeline(input_path)

        assert list(out.columns) == FEATURE_COLUMNS
        assert len(out) == len(input_df)

    def test_missing_required_column_raises(self, tmp_path):
        from features.feature_pipeline import run_pipeline

        input_df = self._make_pipeline_input()
        input_df = input_df.drop(columns=["event_weight"])
        input_path = str(tmp_path / "sessionized_logs.parquet")
        input_df.to_parquet(input_path, index=False)

        with pytest.raises(ValueError):
            run_pipeline(input_path)

    def test_output_floats_are_finite(self, tmp_path):
        from features.feature_pipeline import run_pipeline

        input_df = self._make_pipeline_input()
        input_path = str(tmp_path / "sessionized_logs.parquet")
        input_df.to_parquet(input_path, index=False)

        with patch("features.feature_pipeline.save_parquet"):
            out = run_pipeline(input_path)

        float_cols = out.select_dtypes(include=[float]).columns
        for col in float_cols:
            assert np.isfinite(out[col]).all(), f"Non-finite values in {col}"
