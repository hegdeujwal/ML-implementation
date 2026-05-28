"""
pipeline.py
===========
End-to-end pipeline orchestrator for the HPE CX log analysis system.

Execution order
---------------
  1. parsing       parsing/sessionizer.py        -> data/processed/sessionized_logs.parquet
  2. features      features/feature_pipeline.py  -> data/processed/features_df.parquet
  3. anomaly       ml/anomaly_detector.py        -> data/processed/anomaly_df.parquet
  4. correlation   correlation/run_correlation.py -> data/processed/graph_scores_df.parquet
  5. scoring       scoring/importance_scorer.py  -> data/processed/scored_logs_df.parquet
  6. storage       storage/db_writer.py          -> Postgres (skipped in --dry-run)

Usage
-----
  # Full run (writes to Postgres)
  python pipeline.py

  # Dry run — all steps except Postgres write
  python pipeline.py --dry-run

  # Restart from a specific step (reads existing parquets for earlier steps)
  python pipeline.py --from-step anomaly

  # Dry run starting from scoring
  python pipeline.py --dry-run --from-step scoring

  # Use a specific raw log file as input to the parsing step
  python pipeline.py --log-file data/raw/cx_switches.log

Step names (for --from-step)
-----------------------------
  parsing | features | anomaly | correlation | scoring | storage
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from common.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Step ordering
# ---------------------------------------------------------------------------

STEPS = ["parsing", "features", "anomaly", "correlation", "scoring", "storage"]

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

SESSIONIZED_PATH = "data/processed/sessionized_logs.parquet"
FEATURES_PATH    = "data/processed/features_df.parquet"
ANOMALY_PATH     = "data/processed/anomaly_df.parquet"
GRAPH_SCORES_PATH = "data/processed/graph_scores_df.parquet"
SCORED_LOGS_PATH  = "data/processed/scored_logs_df.parquet"

Path("data/processed").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------

def _step_parsing(log_file: str) -> int:
    """Run the parsing / sessionization step.

    Reads a raw syslog file and writes sessionized_logs.parquet.
    If no log file is provided, generates synthetic data instead.
    """
    p = Path(log_file)

    if p.exists():
        logger.info(f"Parsing raw log file: {log_file}")
        from parsing.sessionizer import run as sessionize
        df = sessionize(log_file, output_path=SESSIONIZED_PATH)
    else:
        logger.warning(
            f"Log file not found: {log_file}. "
            "Generating synthetic sessionized data instead."
        )
        import json
        import pandas as pd
        from scripts.generate_real_logs import generate_dataset
        from common.config import SEVERITY_WEIGHTS, DEFAULT_SEVERITY_WEIGHT, DEFAULT_SOURCE_TYPE
        from common.utils import save_parquet

        df = generate_dataset()

        # Map to canonical schema
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_localize(None)
        df["sequence_number"] = df.index.astype("int64")
        df["source_type"] = DEFAULT_SOURCE_TYPE
        # Derive host from session_id (session_000 → synth-host-000)
        df["host"] = "synth-host-" + df["session_id"].str.extract(r"(\d+)$")[0].fillna("000")
        # service = first token of template_id before the first "_"
        df["service"] = df["template_id"].str.split("_").str[0]
        df["event_type"] = df["service"]
        df["event_action"] = df.apply(
            lambda r: r["template_id"][len(r["service"]) + 1:]
            if r["template_id"].startswith(r["service"] + "_")
            else r["template_id"].split("_", 1)[-1],
            axis=1,
        )
        # frequency = count of same template within the same session
        df["frequency"] = df.groupby(["session_id", "template_id"])["template_id"].transform("count").astype(int)
        df["event_weight"] = df["log_level"].map(SEVERITY_WEIGHTS).fillna(DEFAULT_SEVERITY_WEIGHT)
        df["importance_score"] = 0.0
        df["correlation_id"] = None
        df["message"] = df["log_level"] + " " + df["template_id"] + " synthetic log entry"
        df["metadata"] = df["message"].apply(lambda m: json.dumps({"raw_text": m}))

        # Keep only canonical columns (plus session_id for downstream feature groupby)
        canonical_cols = [
            "log_id", "sequence_number", "timestamp", "source_type", "service", "host",
            "log_level", "event_type", "event_action", "template_id",
            "frequency", "event_weight", "importance_score", "correlation_id",
            "message", "metadata", "session_id",
        ]
        # Ensure a stable, non-null `log_id` for downstream modules and DB writes
        df["log_id"] = df["sequence_number"].apply(lambda i: f"log_{int(i):06d}")

        df = df[canonical_cols]

        Path(SESSIONIZED_PATH).parent.mkdir(parents=True, exist_ok=True)
        save_parquet(df, SESSIONIZED_PATH)

    return len(df)


def _step_features() -> int:
    """Run the feature engineering step."""
    from features.feature_pipeline import run_pipeline
    df = run_pipeline(SESSIONIZED_PATH)
    return len(df)


def _step_anomaly() -> int:
    """Run the ML anomaly detection step."""
    from pathlib import Path as _Path
    from ml.anomaly_detector import run
    df = run(
        features_path=_Path(FEATURES_PATH),
        output_path=_Path(ANOMALY_PATH),
    )
    return len(df)


def _step_correlation() -> int:
    """Run the graph correlation step."""
    import os

    # Correlation module reads SESSIONIZED_LOGS_PATH from config.
    # Point it at our canonical path.
    import common.config as cfg
    _orig = cfg.SESSIONIZED_LOGS_PATH

    # Monkey-patch so run_correlation reads from our pipeline path.
    # This is safe — it only affects this process for the duration of the call.
    cfg.SESSIONIZED_LOGS_PATH = SESSIONIZED_PATH

    try:
        from correlation.run_correlation import run
        run()
    finally:
        cfg.SESSIONIZED_LOGS_PATH = _orig

    import pandas as pd
    df = pd.read_parquet(GRAPH_SCORES_PATH)
    return len(df)


def _step_scoring() -> int:
    """Run the importance scoring step."""
    from scoring.importance_scorer import run
    df = run()
    return len(df)


def _step_storage(dry_run: bool) -> int:
    """Write all parquets to Postgres (skipped in dry-run mode)."""
    if dry_run:
        logger.info("--dry-run: skipping Postgres write.")
        return 0

    import pandas as pd
    from storage.db_writer import (
        apply_schema,
        get_connection,
        write_logs,
        write_features,
        write_anomalies,
        write_scores,
    )

    conn = get_connection()
    try:
        apply_schema(conn)

        counts = {}

        if Path(SESSIONIZED_PATH).exists():
            logs_df = pd.read_parquet(SESSIONIZED_PATH)
            counts["logs"] = write_logs(logs_df, conn)

        if Path(FEATURES_PATH).exists():
            feat_df = pd.read_parquet(FEATURES_PATH)
            counts["features"] = write_features(feat_df, conn)

        if Path(ANOMALY_PATH).exists():
            anom_df = pd.read_parquet(ANOMALY_PATH)
            counts["anomalies"] = write_anomalies(anom_df, conn)

        if Path(SCORED_LOGS_PATH).exists():
            scored_df = pd.read_parquet(SCORED_LOGS_PATH)
            counts["scores"] = write_scores(scored_df, conn)

        conn.commit()
        logger.info(f"Postgres write counts: {counts}")
        return sum(counts.values())

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _run_step(
    step: str,
    dry_run: bool,
    log_file: str,
) -> int:
    """Dispatch to the correct step function and return row count."""
    dispatch = {
        "parsing":     lambda: _step_parsing(log_file),
        "features":    _step_features,
        "anomaly":     _step_anomaly,
        "correlation": _step_correlation,
        "scoring":     _step_scoring,
        "storage":     lambda: _step_storage(dry_run),
    }
    return dispatch[step]()


def run_pipeline(
    dry_run: bool = False,
    from_step: Optional[str] = None,
    log_file: str = "data/raw/sample.log",
) -> None:
    """Run the full pipeline from from_step onwards.

    Args:
        dry_run:   Skip the Postgres write step.
        from_step: Name of the step to start from (skips earlier steps,
                   reading their parquet outputs from disk instead).
        log_file:  Path to the raw log file for the parsing step.
    """
    if from_step and from_step not in STEPS:
        logger.error(
            f"Unknown step '{from_step}'. Valid steps: {', '.join(STEPS)}"
        )
        sys.exit(1)

    start_idx = STEPS.index(from_step) if from_step else 0
    active_steps = STEPS[start_idx:]

    mode_tag = "[DRY-RUN] " if dry_run else ""
    logger.info(f"{mode_tag}Pipeline starting — steps: {', '.join(active_steps)}")

    total_start = time.perf_counter()

    for step in active_steps:
        t0 = time.perf_counter()
        logger.info(f"{'='*50}")
        logger.info(f"STEP: {step.upper()}")

        try:
            row_count = _run_step(step, dry_run=dry_run, log_file=log_file)
            elapsed = time.perf_counter() - t0
            logger.info(
                f"DONE: {step} — {row_count:,} rows — {elapsed:.2f}s"
            )

        except FileNotFoundError as exc:
            logger.error(
                f"FAILED at step '{step}': {exc}\n"
                f"Hint: use --from-step to restart from an earlier step, "
                f"or ensure the upstream parquet exists."
            )
            sys.exit(1)

        except Exception as exc:
            logger.error(f"FAILED at step '{step}': {type(exc).__name__}: {exc}")
            raise

    total_elapsed = time.perf_counter() - total_start
    logger.info("=" * 50)
    logger.info(
        f"{mode_tag}Pipeline complete — {len(active_steps)} steps in "
        f"{total_elapsed:.2f}s"
    )

    if dry_run:
        logger.info("Dry run finished. Postgres write was skipped.")

    _print_output_summary()


def _print_output_summary() -> None:
    """Log a summary of output files produced."""
    outputs = [
        ("sessionized_logs", SESSIONIZED_PATH),
        ("features_df", FEATURES_PATH),
        ("anomaly_df", ANOMALY_PATH),
        ("graph_scores_df", GRAPH_SCORES_PATH),
        ("scored_logs_df", SCORED_LOGS_PATH),
    ]
    logger.info("Output files:")
    for name, path in outputs:
        p = Path(path)
        if p.exists():
            size_kb = p.stat().st_size / 1024
            logger.info(f"  {name:<22} {path}  ({size_kb:.1f} KB)")
        else:
            logger.info(f"  {name:<22} {path}  (not produced)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Run the HPE CX log analysis pipeline end-to-end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all steps but skip the Postgres write (step 6).",
    )
    ap.add_argument(
        "--from-step",
        metavar="STEP",
        choices=STEPS,
        default=None,
        help=(
            f"Start from this step, reading upstream parquets from disk. "
            f"Choices: {', '.join(STEPS)}"
        ),
    )
    ap.add_argument(
        "--log-file",
        metavar="PATH",
        default="data/raw/sample.log",
        help=(
            "Path to the raw syslog file for the parsing step. "
            "If the file does not exist, synthetic data is generated. "
            "(default: data/raw/sample.log)"
        ),
    )
    args = ap.parse_args()

    run_pipeline(
        dry_run=args.dry_run,
        from_step=args.from_step,
        log_file=args.log_file,
    )
