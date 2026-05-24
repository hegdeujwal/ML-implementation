"""
parsing/sessionizer.py
======================
Converts a raw syslog file into sessionized_logs.parquet.

Pipeline
--------
1. Read raw log lines from the input file.
2. Normalize each line via normalizer.normalize_line().
3. Parse the message through DrainParser to obtain a template_id.
4. Group events into sessions: same source + no gap > SESSION_GAP_SECONDS.
5. Write data/processed/sessionized_logs.parquet.

Output schema (sessionized_logs.parquet)
-----------------------------------------
log_id       str    -- unique per-row identifier, e.g. "log_000001"
raw_text     str    -- original unparsed line
timestamp    datetime -- UTC datetime (datetime64[us] in parquet)
source       str    -- hostname / device
session_id   str    -- groups related events, e.g. "session_001"
template_id  str    -- Drain template slug, e.g. "INTERFACE_DOWN"
severity     str    -- CRITICAL / ERROR / WARN / INFO
log_level    str    -- alias of severity (kept for backwards compatibility)
is_anomaly   bool   -- always False from parsing; set by anomaly_detector
anomaly_label str   -- empty string; populated by anomaly_detector

Known limitations
-----------------
- Sessions are defined purely by time gap within the same source.
  This is a first-order approximation; real session logic may factor in
  protocol resets or configuration events.
- The parser is stateless between calls to run(); each call trains a fresh
  DrainParser instance, so templates are not persisted across runs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from parsing.normalizer import normalize_line
from parsing.log_parser import DrainParser
from common.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SESSION_GAP_SECONDS: int = 1800   # 30-minute inactivity gap starts a new session
DEFAULT_OUTPUT_PATH: str = "data/processed/sessionized_logs.parquet"

REQUIRED_OUTPUT_COLUMNS = [
    "log_id",
    "raw_text",
    "timestamp",
    "source",
    "session_id",
    "template_id",
    "severity",
    "log_level",
    "is_anomaly",
    "anomaly_label",
]


# ---------------------------------------------------------------------------
# Sessionizer logic
# ---------------------------------------------------------------------------

def _assign_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Add a session_id column by grouping on source + time gap."""
    df = df.sort_values(["source", "timestamp"]).reset_index(drop=True)

    # Compute seconds between consecutive rows per source for gap detection
    ts_series = pd.to_datetime(df["timestamp"])
    gap_seconds = ts_series.diff().dt.total_seconds().fillna(float("inf"))
    src_changed = df["source"] != df["source"].shift(1)

    session_ids = []
    session_counter = 0

    for i in range(len(df)):
        if src_changed.iloc[i] or gap_seconds.iloc[i] > SESSION_GAP_SECONDS:
            session_counter += 1
        session_ids.append(f"session_{session_counter:04d}")

    df["session_id"] = session_ids
    return df


def run(
    input_path: str,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Parse a raw syslog file and write sessionized_logs.parquet.

    Args:
        input_path:  Path to a raw syslog text file.
        output_path: Destination parquet path (created with parent dirs).

    Returns:
        The sessionized DataFrame.

    Raises:
        FileNotFoundError: If input_path does not exist.
        ValueError: If no parseable log lines are found.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input log file not found: {input_path}")

    logger.info(f"Reading raw logs from {input_path}")

    parser = DrainParser()
    rows = []
    log_counter = 0
    skipped = 0

    with open(input_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parsed = normalize_line(line)
            if parsed is None:
                skipped += 1
                continue

            log_counter += 1
            log_id = f"log_{log_counter:06d}"

            template_id, _ = parser.add_log_message(
                parsed["message"], log_id=log_id
            )

            # normalizer now returns a datetime object directly
            ts_dt = parsed["timestamp"]
            if not isinstance(ts_dt, datetime):
                try:
                    ts_dt = datetime.strptime(str(ts_dt), "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    ts_dt = datetime.utcnow()

            rows.append({
                "log_id": log_id,
                "raw_text": parsed["raw_text"],
                "timestamp": ts_dt,
                "source": parsed["source"],
                "template_id": template_id,
                "severity": parsed["severity"],
                "is_anomaly": False,
                "anomaly_label": "",
            })

    if not rows:
        raise ValueError(
            f"No parseable log lines found in {input_path}. "
            "Check that the file contains syslog-formatted entries."
        )

    logger.info(
        f"Parsed {len(rows):,} lines ({skipped} skipped) from {input_path}"
    )

    df = pd.DataFrame(rows)
    df = _assign_sessions(df)

    # log_level is an alias for severity (backwards compat with features module)
    df["log_level"] = df["severity"]

    # Reorder to canonical schema
    df = df[REQUIRED_OUTPUT_COLUMNS]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    logger.info(
        f"Wrote {len(df):,} rows -> {output_path} "
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
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output parquet path (default: {DEFAULT_OUTPUT_PATH})",
    )
    args = ap.parse_args()

    df = run(args.input, args.output)
    print(f"Sessions : {df['session_id'].nunique()}")
    print(f"Templates: {df['template_id'].nunique()}")
    print(f"Rows     : {len(df):,}")
    print(f"\nSchema:\n{df.dtypes}")
    print(f"\nSample:\n{df.head(3).to_string()}")
