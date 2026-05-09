"""
Unit tests for feature engineering modules.

Covers:
- statistical features
- temporal features
- severity mapping
- counter proximity logic
- edge-case handling
"""




import pandas as pd
from datetime import datetime, timedelta

from features.statistical_features import (
    log_frequency_score,
    burstiness_score,
    zscore_base
)

from features.temporal_features import (
    add_temporal_features
)

from features.severity_features import (
    add_severity_weight
)

from features.counter_proximity import (
    add_counter_proximity
)


def sample_df():

    base = datetime.now()

    return pd.DataFrame({
        "log_id": ["l1", "l2", "l3"],
        "session_id": ["s1", "s1", "s1"],
        "timestamp": [
            base,
            base + timedelta(seconds=5),
            base + timedelta(seconds=15)
        ],
        "template_id": [
            "IF_DOWN",
            "IF_DOWN",
            "CPU_HIGH"
        ],
        "log_level": [
            "CRITICAL",
            "ERROR",
            "WARN"
        ]
    })


def test_frequency_score():

    df = sample_df()

    result = log_frequency_score(df)

    assert "frequency_score" in result.columns
    assert result["frequency_score"].max() <= 1.0


def test_burstiness_score():

    df = sample_df()

    result = burstiness_score(df)

    assert "burstiness_score" in result.columns


def test_zscore_base():

    df = sample_df()

    result = zscore_base(df)

    assert "zscore_base" in result.columns


def test_temporal_features():

    df = sample_df()

    result = add_temporal_features(df)

    assert "time_delta_prev" in result.columns
    assert result["time_delta_prev"].iloc[1] == 5.0


def test_severity_weight():

    df = sample_df()

    result = add_severity_weight(df)

    assert "severity_weight" in result.columns
    assert result["severity_weight"].iloc[0] == 1.0


def test_counter_proximity():

    df = sample_df()

    result = add_counter_proximity(df)

    assert "counter_proximity" in result.columns
    assert result["counter_proximity"].max() <= 1.0

def test_missing_timestamp_handling():

    df = sample_df()

    df.loc[0, "timestamp"] = None

    result = add_temporal_features(df.dropna(subset=["timestamp"]))

    assert result.empty is False



def test_single_log_session():

    df = sample_df().iloc[:1]

    result = burstiness_score(df)

    assert result["burstiness_score"].iloc[0] == 0.0