"""
ingestion/batch_loader.py
=========================
Ingests raw log files from a source directory into data/raw/.

Responsibilities
----------------
- Accept a path to a single log file OR a directory of log files.
- Copy/move each file into data/raw/, preserving the filename.
- Emit a summary: files discovered, bytes transferred, any errors.

This module does NOT parse logs — it is a raw data staging step.
Parsing is handled by parsing/sessionizer.py in the next pipeline stage.

Known limitations / TODO
------------------------
- Does not deduplicate files between ingestion runs (same filename will
  overwrite an existing file in data/raw/).
- Compression (.gz, .bz2) is not decompressed here; sessionizer.py must
  handle compressed inputs if needed.
- fluent_bit.conf handles live-streaming ingestion; this module is for
  batch imports of historical log dumps only.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List

from common.logger import get_logger

logger = get_logger(__name__)

DEFAULT_OUTPUT_DIR = "data/raw"


def ingest(
    source: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    move: bool = False,
) -> List[str]:
    """Copy (or move) raw log files into data/raw/.

    Args:
        source:     Path to a single .log file or a directory containing log files.
        output_dir: Destination directory (created if it does not exist).
        move:       If True, move files instead of copying. Defaults to False (copy).

    Returns:
        List of destination file paths that were written.

    Raises:
        FileNotFoundError: If source does not exist.
        ValueError:        If source is a directory but contains no .log files.
    """
    src = Path(source)
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect candidate files
    if src.is_file():
        candidates = [src]
    else:
        candidates = sorted(src.glob("**/*.log")) + sorted(src.glob("**/*.txt"))

    if not candidates:
        raise ValueError(
            f"No .log or .txt files found under {source}. "
            "Check that the source directory contains log files."
        )

    op = shutil.move if move else shutil.copy2
    op_name = "Moved" if move else "Copied"

    written: List[str] = []
    total_bytes = 0

    for f in candidates:
        dest = out_dir / f.name
        op(str(f), str(dest))
        size = dest.stat().st_size
        total_bytes += size
        written.append(str(dest))
        logger.info(f"  {op_name}: {f.name} -> {dest} ({size:,} bytes)")

    logger.info(
        f"Ingestion complete: {len(written)} file(s), "
        f"{total_bytes / 1024:.1f} KB total -> {out_dir}"
    )
    return written


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Ingest raw log files into data/raw/."
    )
    ap.add_argument(
        "source",
        help="Path to a .log file or a directory of log files.",
    )
    ap.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    ap.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying.",
    )
    args = ap.parse_args()

    paths = ingest(args.source, args.output_dir, move=args.move)
    for p in paths:
        print(p)
