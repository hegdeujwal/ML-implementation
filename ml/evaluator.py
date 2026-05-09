"""
evaluator.py
============
Phase 2 — Anomaly Detection Evaluator
Assignee: Shreeraksha M

Runs the trained anomaly detector against synthetic incident scenarios in
data/synthetic/, computes precision/recall/F1, and saves a summary report
to evaluation/results/anomaly_summary.txt.

Expected synthetic data format:
    Each file in data/synthetic/ is a parquet with the same schema as
    features_df.parquet PLUS a 'ground_truth_anomaly' bool column.
    This is P2's responsibility to generate if P1 hasn't yet.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

from common.logger import get_logger
from ml.anomaly_detector import detect_anomalies, FEATURE_COLS

logger = get_logger(__name__)

SYNTHETIC_DIR = Path("data/synthetic")
RESULTS_DIR = Path("evaluation/results")
SUMMARY_PATH = RESULTS_DIR / "anomaly_summary.txt"


def load_synthetic_scenarios() -> pd.DataFrame:
    """Load all synthetic parquet files from data/synthetic/.

    Each file should have:
        - All columns from FEATURE_COLS
        - log_id (str)
        - ground_truth_anomaly (bool)  ← label for evaluation

    Returns:
        Combined DataFrame across all scenario files.

    Raises:
        FileNotFoundError: If no synthetic files exist.
    """
    parquet_files = list(SYNTHETIC_DIR.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No synthetic parquet files found in {SYNTHETIC_DIR}. "
            "Generate synthetic data or ask P1 to provide scenario files."
        )

    dfs = []
    for path in parquet_files:
        df = pd.read_parquet(path)
        df["_source_file"] = path.name     # helps with debugging
        dfs.append(df)
        logger.info(f"Loaded synthetic scenario: {path.name} ({len(df)} rows)")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total synthetic rows: {len(combined)}")
    return combined


def evaluate() -> dict:
    """Run the full evaluation pipeline and return metrics dict.

    Returns:
        Dictionary with keys: precision, recall, f1, anomaly_rate,
        top_10_templates (list of template strings).
    """
    # -----------------------------------------------------------------------
    # Load synthetic data
    # -----------------------------------------------------------------------
    synthetic_df = load_synthetic_scenarios()

    if "ground_truth_anomaly" not in synthetic_df.columns:
        raise KeyError(
            "'ground_truth_anomaly' column missing from synthetic data. "
            "Each synthetic file must include a bool ground truth label."
        )

    # -----------------------------------------------------------------------
    # Run anomaly detection on synthetic data
    # -----------------------------------------------------------------------
    # detect_anomalies expects at least log_id + FEATURE_COLS; it handles
    # extra columns (like ground_truth_anomaly) gracefully.
    anomaly_df = detect_anomalies(synthetic_df)

    # -----------------------------------------------------------------------
    # Merge ground truth back in for evaluation
    # -----------------------------------------------------------------------
    eval_df = anomaly_df.merge(
        synthetic_df[["log_id", "ground_truth_anomaly"]].drop_duplicates("log_id"),
        on="log_id",
        how="left",
    )

    y_true = eval_df["ground_truth_anomaly"].astype(int).values
    y_pred = eval_df["is_anomaly"].astype(int).values

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    anomaly_rate = y_pred.mean()

    logger.info(
        f"Evaluation — Precision: {precision:.4f}, Recall: {recall:.4f}, "
        f"F1: {f1:.4f}, Anomaly rate: {anomaly_rate:.4f}"
    )

    # -----------------------------------------------------------------------
    # Top 10 anomalous templates
    # -----------------------------------------------------------------------
    # If synthetic data includes a 'template' or 'log_template' column, we
    # can surface which template types are most anomalous. Graceful fallback
    # if the column doesn't exist yet.
    template_col = next(
        (c for c in ["log_template", "template", "event_template"] if c in synthetic_df.columns),
        None,
    )
    top_10_templates = []
    if template_col:
        template_df = eval_df.merge(
            synthetic_df[["log_id", template_col]].drop_duplicates("log_id"),
            on="log_id",
            how="left",
        )
        top_10_templates = (
            template_df[template_df["is_anomaly"]][template_col]
            .value_counts()
            .head(10)
            .index.tolist()
        )
    else:
        logger.warning(
            "No template column found in synthetic data — top 10 templates unavailable. "
            "Add 'log_template', 'template', or 'event_template' to synthetic files."
        )

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "anomaly_rate": anomaly_rate,
        "top_10_templates": top_10_templates,
        "n_samples": len(eval_df),
        "n_true_positives": int((y_pred & y_true.astype(bool)).sum()),
        "n_false_positives": int((y_pred.astype(bool) & ~y_true.astype(bool)).sum()),
        "n_false_negatives": int((~y_pred.astype(bool) & y_true.astype(bool)).sum()),
    }

    # -----------------------------------------------------------------------
    # Print full sklearn report to console as well
    # -----------------------------------------------------------------------
    print(classification_report(y_true, y_pred, target_names=["normal", "anomaly"]))

    return metrics


def save_summary(metrics: dict, path: Path = SUMMARY_PATH) -> None:
    """Write evaluation metrics to a human-readable text file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 60,
        "ShreeRaksha — Phase 2 Anomaly Detection Evaluation",
        "=" * 60,
        f"Samples evaluated : {metrics['n_samples']}",
        f"Anomaly rate      : {metrics['anomaly_rate']:.4f} ({metrics['anomaly_rate']*100:.1f}%)",
        "",
        "Classification metrics:",
        f"  Precision        : {metrics['precision']:.4f}",
        f"  Recall           : {metrics['recall']:.4f}",
        f"  F1 Score         : {metrics['f1']:.4f}",
        "",
        "Confusion breakdown:",
        f"  True Positives   : {metrics['n_true_positives']}",
        f"  False Positives  : {metrics['n_false_positives']}",
        f"  False Negatives  : {metrics['n_false_negatives']}",
        "",
        "Top 10 most anomalous log templates:",
    ]

    if metrics["top_10_templates"]:
        for i, tmpl in enumerate(metrics["top_10_templates"], 1):
            lines.append(f"  {i:2d}. {tmpl}")
    else:
        lines.append("  (template column not available in synthetic data)")

    lines.append("=" * 60)

    summary_text = "\n".join(lines)
    print(summary_text)
    path.write_text(summary_text)
    logger.info(f"Evaluation summary saved to {path}.")


def run() -> dict:
    """End-to-end evaluator entry point."""
    metrics = evaluate()
    save_summary(metrics)
    return metrics


if __name__ == "__main__":
    run()