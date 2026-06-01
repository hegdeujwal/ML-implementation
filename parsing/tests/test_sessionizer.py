"""
Tests for parsing/sessionizer.py — session assignment, frequency, derivations, schema.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

import pandas as pd
import pytest

from parsing.sessionizer import _assign_sessions, _derive_event_action, REQUIRED_OUTPUT_COLUMNS
from common.config import SESSION_GAP_SECONDS, DEFAULT_SOURCE_TYPE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal input DataFrame as sessionizer would before _assign_sessions."""
    return pd.DataFrame(rows)


def _base_ts() -> datetime:
    return datetime(2026, 3, 12, 10, 0, 0)


# ---------------------------------------------------------------------------
# Test 1 — Session boundary: same host, gap > threshold → different session_ids
# ---------------------------------------------------------------------------

def test_session_boundary_different_sessions():
    t0 = _base_ts()
    t1 = t0 + timedelta(seconds=SESSION_GAP_SECONDS + 1)
    df = _make_df([
        {"host": "sw-01", "timestamp": t0, "template_id": "PORT_DOWN", "service": "PORT"},
        {"host": "sw-01", "timestamp": t1, "template_id": "PORT_UP",   "service": "PORT"},
    ])
    result = _assign_sessions(df)
    assert result["session_id"].iloc[0] != result["session_id"].iloc[1]


# ---------------------------------------------------------------------------
# Test 2 — Session continuity: same host, gap < threshold → same session_id
# ---------------------------------------------------------------------------

def test_session_continuity_same_session():
    t0 = _base_ts()
    t1 = t0 + timedelta(seconds=SESSION_GAP_SECONDS - 1)
    df = _make_df([
        {"host": "sw-01", "timestamp": t0, "template_id": "PORT_DOWN", "service": "PORT"},
        {"host": "sw-01", "timestamp": t1, "template_id": "PORT_UP",   "service": "PORT"},
    ])
    result = _assign_sessions(df)
    assert result["session_id"].iloc[0] == result["session_id"].iloc[1]


# ---------------------------------------------------------------------------
# Test 3 — Frequency: template appearing 3× in a session → frequency=3 on all rows
# ---------------------------------------------------------------------------

def test_frequency_count():
    import pandas as pd
    from parsing.sessionizer import _assign_sessions
    from common.config import SESSION_GAP_SECONDS

    t0 = _base_ts()
    rows = [
        {"host": "sw-01", "timestamp": t0 + timedelta(seconds=i * 10),
         "template_id": "OSPF_NEIGHBOR_DOWN", "service": "OSPF",
         "sequence_number": i + 1, "source_type": "switch",
         "log_level": "ERROR", "event_weight": 0.7, "message": "msg", "_raw_text": "raw"}
        for i in range(3)
    ]
    df = pd.DataFrame(rows)
    df = _assign_sessions(df)
    df["event_type"] = df["service"]
    df["event_action"] = df.apply(
        lambda r: _derive_event_action(r["service"], r["template_id"]), axis=1
    )
    df["frequency"] = (
        df.groupby(["session_id", "template_id"])["template_id"]
        .transform("count")
        .astype(int)
    )
    assert (df["frequency"] == 3).all()


# ---------------------------------------------------------------------------
# Test 4 — event_type/event_action: OSPF service + OSPF_NEIGHBOR_STATE_CHANGE
# ---------------------------------------------------------------------------

def test_event_type_and_action_with_service_prefix():
    action = _derive_event_action("OSPF", "OSPF_NEIGHBOR_STATE_CHANGE")
    assert action == "NEIGHBOR_STATE_CHANGE"


# ---------------------------------------------------------------------------
# Test 5 — event_type fallback: missing service → split template_id on first _
# ---------------------------------------------------------------------------

def test_event_action_fallback_split():
    action = _derive_event_action("UNKNOWN", "PORT_CHANGED_STATE")
    assert action == "CHANGED_STATE"


# ---------------------------------------------------------------------------
# Test 6 — source_type: always DEFAULT_SOURCE_TYPE ('switch')
# ---------------------------------------------------------------------------

def test_source_type_is_switch():
    assert DEFAULT_SOURCE_TYPE == "switch"


# ---------------------------------------------------------------------------
# Test 7 — Output schema: correct columns, correct order, no extras
# ---------------------------------------------------------------------------

def test_output_schema_columns_and_order():
    expected = [
        "sequence_number", "timestamp", "source_type", "service", "host",
        "log_level", "event_type", "event_action", "template_id", "frequency",
        "event_weight", "session_id", "message", "metadata",
    ]
    assert REQUIRED_OUTPUT_COLUMNS == expected


# ---------------------------------------------------------------------------
# Test 8 — session_id format: matches "{host}_{YYYYMMDDTHHmmSS}"
# ---------------------------------------------------------------------------

def test_session_id_format():
    t0 = _base_ts()
    df = _make_df([
        {"host": "sw-access-01", "timestamp": t0, "template_id": "X", "service": "S"},
    ])
    result = _assign_sessions(df)
    sid = result["session_id"].iloc[0]
    pattern = re.compile(r"^.+_\d{8}T\d{6}$")
    assert pattern.match(sid), f"session_id '{sid}' does not match expected format"
    assert sid == f"sw-access-01_{t0.strftime('%Y%m%dT%H%M%S')}"
