"""
test_llm_summary.py
===================
Standalone tests for llm_summary.py (google-genai SDK version).
Runs WITHOUT Postgres or Docker.

Run from project ROOT:
    cd ML-implementation
    python -m pytest dashboard/test_llm_summary.py -v

Prerequisites:
    pip install google-genai pandas pytest
    GEMINI_API_KEY in .env (only needed for live test)
"""

import os
import sys
import json
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
env_path = Path(".env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

# ---------------------------------------------------------------------------
# Mock all external dependencies BEFORE importing llm_summary
# ---------------------------------------------------------------------------
db_mock = MagicMock()
db_mock.get_summary.return_value = None
db_mock.write_summaries_batch.return_value = None
db_mock.write_summary.return_value = None

sys.modules["data"] = MagicMock()
sys.modules["data.db"] = db_mock

storage_mock = MagicMock()
storage_mock.get_summary.return_value = None
storage_mock.write_summary.return_value = None
storage_mock.write_summaries_batch.return_value = None
sys.modules["storage"] = MagicMock()
sys.modules["storage.db_writer"] = storage_mock

config_mock = MagicMock()
config_mock.DB_HOST = "localhost"
config_mock.DB_PORT = "5432"
config_mock.DB_NAME = "hpe_logs"
config_mock.DB_PASSWORD = "postgres"
sys.modules["common"] = MagicMock()
sys.modules["common.config"] = config_mock
sys.modules["common.logger"] = MagicMock(get_logger=lambda x: MagicMock())


# ---------------------------------------------------------------------------
# Synthetic test data
# ---------------------------------------------------------------------------

def make_scored_df(n_incidents: int = 3, logs_per_incident: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_incidents):
        cid = f"INC-{i:04d}"
        for j in range(logs_per_incident):
            rows.append({
                "sequence_number": f"log_{i}_{j}",
                "correlation_id": cid,
                "incident_id": cid,
                "host": f"switch-0{i + 1}",
                "template_id": rng.choice([
                    "IF_DOWN", "OSPF_NBR_CHANGE", "PORT_ERR", "CPU_HIGH"
                ]),
                "label": rng.choice(["critical", "medium", "low", "ignore"]),
                "final_score": float(rng.uniform(0.1, 1.0)),
                "is_cross_system": bool(i % 2 == 0),
                "timestamp": (
                    pd.Timestamp("2026-05-28 10:00:00")
                    + pd.Timedelta(seconds=j * 5 + i * 300)
                ),
                "message": f"Sample log message {i}_{j}",
            })
    return pd.DataFrame(rows)


def make_root_causes_df(n_incidents: int = 3) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "incident_id": f"INC-{i:04d}",
            "root_cause_log_id": f"log_{i}_0",
            "confidence_score": round(0.85 - i * 0.05, 2),
        }
        for i in range(n_incidents)
    ])


# ---------------------------------------------------------------------------
# Tests — patch _generate() directly (SDK-independent)
# ---------------------------------------------------------------------------

class TestBuildContext(unittest.TestCase):
    def setUp(self):
        self.scored_df = make_scored_df()
        self.rc_df = make_root_causes_df()

    def test_has_all_required_keys(self):
        from dashboard.llm_summary import _build_context
        ctx = _build_context("INC-0000", self.scored_df, self.rc_df)
        required = {
            "correlation_id", "host", "start_time", "duration_seconds",
            "log_count", "is_cross_system", "template_sequence",
            "root_causes", "top3_logs",
        }
        self.assertTrue(required.issubset(ctx.keys()))

    def test_correlation_id_correct(self):
        from dashboard.llm_summary import _build_context
        ctx = _build_context("INC-0001", self.scored_df, self.rc_df)
        self.assertEqual(ctx["correlation_id"], "INC-0001")

    def test_log_count_correct(self):
        from dashboard.llm_summary import _build_context
        ctx = _build_context("INC-0000", self.scored_df, self.rc_df)
        expected = len(self.scored_df[self.scored_df["correlation_id"] == "INC-0000"])
        self.assertEqual(ctx["log_count"], expected)

    def test_template_sequence_truncated(self):
        from dashboard.llm_summary import _build_context
        big_df = make_scored_df(n_incidents=1, logs_per_incident=15)
        ctx = _build_context("INC-0000", big_df, self.rc_df)
        self.assertIn("more", ctx["template_sequence"])


class TestBuildSinglePrompt(unittest.TestCase):
    def test_prompt_contains_key_fields(self):
        from dashboard.llm_summary import _build_single_prompt
        ctx = {
            "correlation_id": "INC-TEST",
            "host": "switch-01",
            "duration_seconds": 45,
            "log_count": 12,
            "template_sequence": "IF_DOWN → OSPF_NBR_CHANGE",
            "root_causes": "log_0_0 (conf: 0.85)",
            "top3_logs": "IF_DOWN (0.95)",
            "is_cross_system": False,
        }
        prompt = _build_single_prompt(ctx)
        self.assertIn("INC-TEST", prompt)
        self.assertIn("switch-01", prompt)
        self.assertIn("IF_DOWN", prompt)


class TestGenerateAllSummaries(unittest.TestCase):
    def setUp(self):
        self.scored_df = make_scored_df()
        self.rc_df = make_root_causes_df()
        db_mock.get_summary.return_value = None

    @patch("dashboard.llm_summary._generate")
    def test_calls_write_summaries_batch(self, mock_generate):
        """Should call write_summaries_batch with generated summaries."""
        mock_generate.return_value = json.dumps([
            {"correlation_id": "INC-0000", "summary_text": "Interface flap on switch-01."},
            {"correlation_id": "INC-0001", "summary_text": "OSPF lost on switch-02."},
            {"correlation_id": "INC-0002", "summary_text": "CPU spike on switch-03."},
        ])

        from dashboard.llm_summary import generate_all_summaries
        generate_all_summaries(self.scored_df, self.rc_df, batch_size=20)

        db_mock.write_summaries_batch.assert_called_once()
        written = db_mock.write_summaries_batch.call_args[0][0]
        self.assertEqual(len(written), 3)

    @patch("dashboard.llm_summary._generate")
    def test_skips_cached_incidents(self, mock_generate):
        """Already cached incidents must not trigger Gemini calls."""
        db_mock.get_summary.return_value = "Already cached."

        from dashboard.llm_summary import generate_all_summaries
        generate_all_summaries(self.scored_df, self.rc_df, batch_size=20)

        mock_generate.assert_not_called()
        db_mock.get_summary.return_value = None  # reset

    @patch("dashboard.llm_summary._generate")
    def test_fallback_on_bad_json(self, mock_generate):
        """Invalid JSON triggers individual fallback — no crash."""
        # First call (batch) returns bad JSON, subsequent calls (individual) return plain text
        mock_generate.side_effect = [
            "NOT VALID JSON {{{{{",           # batch call fails
            "Plain summary for INC-0000.",    # individual fallback
            "Plain summary for INC-0001.",
            "Plain summary for INC-0002.",
        ]

        from dashboard.llm_summary import generate_all_summaries
        generate_all_summaries(self.scored_df, self.rc_df, batch_size=20)
        db_mock.write_summaries_batch.assert_called()

    def test_empty_scored_df_skips_gracefully(self):
        from dashboard.llm_summary import generate_all_summaries
        generate_all_summaries(pd.DataFrame(), self.rc_df, batch_size=20)

    def test_empty_rc_df_skips_gracefully(self):
        from dashboard.llm_summary import generate_all_summaries
        generate_all_summaries(self.scored_df, pd.DataFrame(), batch_size=20)


class TestRegenerateSummary(unittest.TestCase):

    @patch("dashboard.llm_summary._generate")
    def test_returns_text_and_writes_to_db(self, mock_generate):
        mock_generate.return_value = "Switch-01 experienced an interface flap at 10:00."

        from dashboard.llm_summary import regenerate_summary
        result = regenerate_summary("INC-0000", {
            "correlation_id": "INC-0000",
            "host": "switch-01",
            "template_sequence": "IF_DOWN → OSPF_NBR_CHANGE",
            "root_causes": "log_0_0 (conf: 0.85)",
            "duration_seconds": 45,
            "log_count": 12,
            "top3_logs": "IF_DOWN (0.95)",
            "is_cross_system": False,
        })
        # Case-insensitive — Gemini may capitalise host names
        self.assertIn("switch-01", result.lower())
        db_mock.write_summary.assert_called_with("INC-0000", result)

    @patch("dashboard.llm_summary._generate")
    def test_returns_error_string_on_api_failure(self, mock_generate):
        """API failure must return error string, never raise."""
        mock_generate.side_effect = Exception("API down")

        from dashboard.llm_summary import regenerate_summary
        result = regenerate_summary("INC-0000", {
            "correlation_id": "INC-0000",
            "host": "switch-01",
            "template_sequence": "IF_DOWN",
            "root_causes": "none",
            "duration_seconds": 0,
            "log_count": 1,
            "top3_logs": "N/A",
            "is_cross_system": False,
        })
        self.assertIn("unavailable", result.lower())


class TestLiveGeminiAPI(unittest.TestCase):
    def setUp(self):
        if not os.environ.get("GEMINI_API_KEY"):
            self.skipTest("GEMINI_API_KEY not set — skipping live API test.")

    def test_single_live_call(self):
        from dashboard.llm_summary import regenerate_summary
        result = regenerate_summary("INC-LIVE-TEST", {
            "correlation_id": "INC-LIVE-TEST",
            "host": "switch-01",
            "template_sequence": "IF_DOWN → OSPF_NBR_CHANGE → PORT_ERR",
            "root_causes": "IF_DOWN (conf: 0.92)",
            "duration_seconds": 47,
            "log_count": 15,
            "top3_logs": "IF_DOWN (0.95); OSPF_NBR_CHANGE (0.88)",
            "is_cross_system": False,
        })
        print(f"\nLive Gemini response:\n{result}\n")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 20)
        self.assertNotIn("unavailable", result.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)