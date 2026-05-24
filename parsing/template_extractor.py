"""
parsing/template_extractor.py
==============================
Analyses and displays the template list from sessionized_logs.parquet.

Input:  data/processed/sessionized_logs.parquet (or any path passed on the CLI)
Output: Console summary of templates sorted by frequency.
        Optionally saves a templates.csv alongside the parquet.

Known limitations
-----------------
- Templates are derived from Drain clustering which is sensitive to the
  SIM_THRESHOLD setting.  If results look too coarse (few templates covering
  many messages) lower the threshold; if too granular (many one-off clusters)
  raise it.  Tune DrainParser.SIM_THRESHOLD in log_parser.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from common.logger import get_logger

logger = get_logger(__name__)

DEFAULT_INPUT_PATH = "data/processed/sessionized_logs.parquet"


def extract_templates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute template frequency statistics from a sessionized DataFrame.

    Args:
        df: DataFrame containing at minimum 'template_id' and 'severity' columns.

    Returns:
        DataFrame with columns: template_id, count, pct, top_severity.
        Sorted descending by count.
    """
    total = len(df)
    stats = (
        df.groupby("template_id")
        .agg(
            count=("template_id", "size"),
            top_severity=("severity", lambda s: s.mode().iloc[0] if not s.empty else "INFO"),
        )
        .reset_index()
    )
    stats["pct"] = (stats["count"] / total * 100).round(2)
    stats = stats.sort_values("count", ascending=False).reset_index(drop=True)
    return stats


def run(
    input_path: str = DEFAULT_INPUT_PATH,
    save_csv: bool = False,
) -> pd.DataFrame:
    """Load sessionized parquet, extract templates, and print summary.

    Args:
        input_path: Path to sessionized_logs.parquet.
        save_csv:   If True, save templates.csv alongside the parquet.

    Returns:
        Template statistics DataFrame.

    Raises:
        FileNotFoundError: If the parquet does not exist.
    """
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(
            f"sessionized_logs.parquet not found at {input_path}. "
            "Run parsing/sessionizer.py first."
        )

    df = pd.read_parquet(p)
    logger.info(f"Loaded {len(df):,} rows from {input_path}")

    templates = extract_templates(df)

    print(f"\n{'='*55}")
    print(f"  Template summary — {len(templates)} unique templates")
    print(f"{'='*55}")
    print(f"{'template_id':<35} {'count':>8}  {'%':>6}  severity")
    print("-" * 55)
    for _, row in templates.iterrows():
        print(
            f"{row['template_id']:<35} {int(row['count']):>8}  "
            f"{row['pct']:>5.1f}%  {row['top_severity']}"
        )
    print("=" * 55)

    n = len(templates)
    if n < 5:
        logger.warning(
            f"Only {n} templates found — result may be too coarse. "
            "Consider lowering DrainParser.SIM_THRESHOLD in log_parser.py."
        )
    elif n > 200:
        logger.warning(
            f"{n} templates found — result may be too granular. "
            "Consider raising DrainParser.SIM_THRESHOLD in log_parser.py."
        )
    else:
        logger.info(f"Template count {n} looks reasonable.")

    if save_csv:
        csv_path = p.parent / "templates.csv"
        templates.to_csv(csv_path, index=False)
        logger.info(f"Saved template list to {csv_path}")

    return templates


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Print template summary from sessionized parquet.")
    ap.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_INPUT_PATH,
        help=f"Path to sessionized_logs.parquet (default: {DEFAULT_INPUT_PATH})",
    )
    ap.add_argument(
        "--save-csv",
        action="store_true",
        help="Save templates.csv alongside the parquet file.",
    )
    args = ap.parse_args()
    run(args.input, save_csv=args.save_csv)
