"""
parsing/sessionizer.py
======================
Converts a raw syslog file into sessionized_logs.parquet.

Pipeline
--------
1. Read raw log lines from the input file.
2. Normalize each line via normalizer.normalize_line() →
   {raw_text, timestamp, host, service, log_level, message}.
3. Parse the message through DrainParser to obtain template_id.
4. Group events into sessions: same host + no gap > SESSION_GAP_SECONDS.
5. Derive event_type (= service) and event_action (= template_id with
   service prefix stripped, or first-underscore split as fallback).
6. Compute frequency: count of this template_id within its session.
7. Write data/processed/sessionized_logs.parquet.

Output schema
-------------
sequence_number  int       -- universal join key (1-based, monotonically increasing)
timestamp        datetime
source_type      str       -- always 'switch' for HPE CX logs
service          str       -- normalised subsystem name (OSPF, BGP, SYSTEM, ...)
host             str       -- device hostname
log_level        str       -- CRITICAL | ERROR | WARN | INFO
event_type       str       -- subsystem label (= service)
event_action     str       -- specific action (template_id minus service prefix)
template_id      str       -- Drain template slug
frequency        int       -- count of this template in the same session
event_weight     float     -- severity weight: CRITICAL=1.0, ERROR=0.7, WARN=0.4, INFO=0.1
message          str       -- log message content (severity tokens stripped for Drain)
metadata         str       -- JSON: {"raw_text": "<original line>"}
session_id       str       -- groups related events; not in canonical DB schema
                              but kept here for downstream feature engineering
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from parsing.normalizer import normalize_line
from parsing.log_parser import DrainParser
from common.config import (
    SESSION_GAP_SECONDS,
    SESSIONIZED_LOGS_PATH,
    SEVERITY_WEIGHTS,
    DEFAULT_SEVERITY_WEIGHT,
    DEFAULT_SOURCE_TYPE,
)
from common.logger import get_logger
from common.utils import save_parquet, validate_schema

logger = get_logger(__name__)

REQUIRED_OUTPUT_COLUMNS = [
    "sequence_number",
    "timestamp",
    "source_type",
    "service",
    "host",
    "log_level",
    "event_type",
    "event_action",
    "template_id",
    "frequency",
    "event_weight",
    "session_id",
    "message",
    "metadata",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assign_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Add session_id column by grouping on host + time gap."""
    df = df.sort_values(["host", "timestamp"]).reset_index(drop=True)

    ts_series = pd.to_datetime(df["timestamp"])
    gap_seconds = ts_series.diff().dt.total_seconds().fillna(float("inf"))
    host_changed = df["host"] != df["host"].shift(1)

    session_ids = []
    current_session_id: str | None = None
    _seen_bases: dict[str, int] = {}
    for i in range(len(df)):
        if host_changed.iloc[i] or gap_seconds.iloc[i] > SESSION_GAP_SECONDS:
            session_start_ts = ts_series.iloc[i].to_pydatetime()
            host = df["host"].iloc[i]
            base = f"{host}_{session_start_ts.strftime('%Y%m%dT%H%M%S')}"
            count = _seen_bases.get(base, 0) + 1
            _seen_bases[base] = count
            current_session_id = base if count == 1 else f"{base}_{count}"
        session_ids.append(current_session_id)

    df["session_id"] = session_ids
    return df


def _derive_event_action(service: str, template_id: str) -> str:
    """Return the action portion of template_id with the service prefix removed.

    e.g. service="OSPF", template_id="OSPF_NEIGHBOR_STATE_CHANGE"
         → "NEIGHBOR_STATE_CHANGE"

    Falls back to splitting on the first underscore when the template_id does
    not start with the service name.
    """
    prefix = service + "_"
    if template_id.startswith(prefix):
        return template_id[len(prefix):]
    parts = template_id.split("_", 1)
    return parts[1] if len(parts) > 1 else template_id


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    input_path: str,
    output_path: str = SESSIONIZED_LOGS_PATH,
) -> pd.DataFrame:
    """Parse a raw syslog file and write sessionized_logs.parquet.

    Args:
        input_path:  Path to a raw syslog text file.
        output_path: Destination parquet path (parent dirs created if needed).

    Returns:
        The sessionized DataFrame.

    Raises:
        FileNotFoundError: If input_path does not exist.
        ValueError:        If no parseable log lines are found.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input log file not found: {input_path}")

    logger.info(f"Reading raw logs from {input_path}")

    parser = DrainParser()
    rows = []
    seq_num = 0
    skipped = 0

    # Pass 1: feed every message through Drain and store the cluster object.
    # template_id() is NOT called here — Drain templates are still evolving.
    with open(input_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parsed = normalize_line(line)
            if parsed is None:
                skipped += 1
                continue

            seq_num += 1
            cluster = parser.add_log_message_cluster(
                parsed["message"], sequence_number=seq_num
            )

            rows.append({
                "sequence_number": seq_num,
                "timestamp": parsed["timestamp"],
                "source_type": DEFAULT_SOURCE_TYPE,
                "service": parsed["service"],
                "host": parsed["host"],
                "log_level": parsed["log_level"],
                "_cluster": cluster,
                "event_weight": SEVERITY_WEIGHTS.get(
                    parsed["log_level"], DEFAULT_SEVERITY_WEIGHT
                ),
                "message": parsed["message"],
                "_raw_text": parsed["raw_text"],
            })

    if not rows:
        raise ValueError(
            f"No parseable log lines found in {input_path}. "
            "Check that the file contains syslog-formatted entries."
        )

    logger.info(f"Parsed {len(rows):,} lines ({skipped} skipped) from {input_path}")

    # Pass 2: Drain templates are now stable — resolve final collision-safe slugs.
    # session_id assignment and frequency groupby both run after this point.
    for row in rows:
        row["template_id"] = parser.resolve_template_id(row.pop("_cluster"))

    df = pd.DataFrame(rows)
    df = _assign_sessions(df)

    df["event_type"] = df["service"]
    df["event_action"] = df.apply(
        lambda r: _derive_event_action(r["service"], r["template_id"]), axis=1
    )

    # frequency: count of this template_id within its session
    df["frequency"] = (
        df.groupby(["session_id", "template_id"])["template_id"]
        .transform("count")
        .astype(int)
    )

    df["metadata"] = df["_raw_text"].apply(lambda t: json.dumps({"raw_text": t}))
    df = df.drop(columns=["_raw_text"])

    validate_schema(df, REQUIRED_OUTPUT_COLUMNS)
    save_parquet(df[REQUIRED_OUTPUT_COLUMNS], output_path)

    logger.info(
        f"Wrote {len(df):,} rows → {output_path} "
        f"({df['session_id'].nunique()} sessions, "
        f"{df['template_id'].nunique()} templates)"
    )
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Sessionize a raw syslog file.")
    ap.add_argument(
        "input",
        nargs="?",
        default="data/raw/sample.log",
        help="Path to raw syslog file (default: data/raw/sample.log)",
    )
    ap.add_argument(
        "--output",
        default=SESSIONIZED_LOGS_PATH,
        help=f"Output parquet path (default: {SESSIONIZED_LOGS_PATH})",
    )
    args = ap.parse_args()

    df = run(args.input, args.output)
    print(f"Sessions : {df['session_id'].nunique()}")
    print(f"Templates: {df['template_id'].nunique()}")
    print(f"Rows     : {len(df):,}")
    print(f"\nSchema:\n{df.dtypes}")
    print(f"\nSample:\n{df.head(3).to_string()}")
