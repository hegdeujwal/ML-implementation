"""
evaluator.py
============
Phase 3 — Anomaly Detection Evaluator
Assignee: Shreeraksha M

Computes distribution metrics on anomaly_df (always) and classification
metrics against a ground-truth file (when available).

Ground truth file (optional):
    data/synthetic/ground_truth.parquet  or  data/synthetic/ground_truth.csv
    Required columns: [sequence_number, true_label]
    true_label values: "anomaly" | "normal"

If no ground truth is found a WARNING is logged and only distribution
metrics are reported — the function does not crash.

Report is saved to evaluation/results/anomaly_summary.txt.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from common.config import ANOMALY_SCORE_THRESHOLD
from common.logger import get_logger
from common.utils import load_parquet

logger = get_logger(__name__)

SYNTHETIC_DIR = Path("data/synthetic")
RESULTS_DIR = Path("evaluation/results")
SUMMARY_PATH = RESULTS_DIR / "anomaly_summary.txt"

_ANOMALY_OUTPUT_PATH = Path("data/processed/anomaly_df.parquet")
_FEATURES_OUTPUT_PATH = Path("data/processed/features_df.parquet")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_ground_truth() -> pd.DataFrame:
    """Load ground-truth labels from data/synthetic/.

    Tries ground_truth.parquet first, then ground_truth.csv.

    Returns:
        DataFrame with columns [sequence_number, true_label].

    Raises:
        FileNotFoundError: If neither file exists.
    """
    parquet_path = SYNTHETIC_DIR / "ground_truth.parquet"
    csv_path = SYNTHETIC_DIR / "ground_truth.csv"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        logger.info(f"Loaded ground truth from {parquet_path} ({len(df)} rows)")
        return df

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded ground truth from {csv_path} ({len(df)} rows)")
        return df

    raise FileNotFoundError(
        f"No ground truth file found in {SYNTHETIC_DIR}. "
        "Expected ground_truth.parquet or ground_truth.csv with columns "
        "[sequence_number, true_label]."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_evaluation(
    anomaly_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> dict:
    """Compute evaluation metrics against anomaly_df.

    Distribution metrics are always reported.  Classification metrics
    (precision, recall, F1, FNR, NSR) are added only when a ground-truth
    file is found in data/synthetic/.

    Args:
        anomaly_df:  Output of anomaly_detector.detect().  Must contain:
                     sequence_number, combined_score, is_anomaly,
                     model_confidence.
        features_df: Output of the features stage.  Must contain:
                     sequence_number, template_id.

    Returns:
        Metrics dict.  Keys always present:
            anomaly_rate, mean_combined_score, score_distribution,
            top_anomalous_templates, model_confidence.
        Keys present only when ground truth available:
            precision, recall, f1, false_negative_rate,
            noise_suppression_ratio.
    """
    # ------------------------------------------------------------------
    # Fix 1 — graceful fallback when no synthetic ground truth exists
    # ------------------------------------------------------------------
    gt_df: pd.DataFrame | None = None
    try:
        gt_df = _load_ground_truth()
    except FileNotFoundError:
        logger.warning(
            "No ground truth found in data/synthetic/ — "
            "reporting distribution metrics only."
        )

    # ------------------------------------------------------------------
    # Fix 4 — distribution metrics, always computed
    # ------------------------------------------------------------------
    anomaly_rate = float(anomaly_df["is_anomaly"].mean())
    mean_combined_score = float(anomaly_df["combined_score"].mean())
    score_distribution = anomaly_df["combined_score"].describe().to_dict()
    model_confidence = float(anomaly_df["model_confidence"].iloc[0])

    # Top 10 template_ids by mean combined_score (join on sequence_number)
    # Fix 2 — join on sequence_number, not log_id
    merged = anomaly_df.merge(
        features_df[["sequence_number", "template_id"]],
        on="sequence_number",
        how="left",
    )
    top_anomalous_templates: list[str] = (
        merged.groupby("template_id")["combined_score"]
        .mean()
        .nlargest(10)
        .index.tolist()
    )

    metrics: dict = {
        "anomaly_rate": anomaly_rate,
        "mean_combined_score": mean_combined_score,
        "score_distribution": score_distribution,
        "top_anomalous_templates": top_anomalous_templates,
        "model_confidence": model_confidence,
    }

    # ------------------------------------------------------------------
    # Classification metrics — only when ground truth is available
    # ------------------------------------------------------------------
    if gt_df is not None:
        # Fix 2 — join on sequence_number
        eval_df = anomaly_df.merge(
            gt_df[["sequence_number", "true_label"]].drop_duplicates("sequence_number"),
            on="sequence_number",
            how="left",
        )

        y_true = (eval_df["true_label"] == "anomaly").astype(int).values
        y_pred = eval_df["is_anomaly"].astype(int).values

        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        # Fix 3 — false_negative_rate and noise_suppression_ratio
        true_anomaly_mask = y_true.astype(bool)
        if true_anomaly_mask.any():
            false_negative_rate = float(
                (~eval_df["is_anomaly"].values[true_anomaly_mask]).mean()
            )
        else:
            false_negative_rate = 0.0

        true_normal_mask = ~true_anomaly_mask
        if true_normal_mask.any():
            noise_suppression_ratio = float(
                (~eval_df["is_anomaly"].values[true_normal_mask]).mean()
            )
        else:
            noise_suppression_ratio = 0.0

        metrics.update({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_negative_rate": false_negative_rate,
            "noise_suppression_ratio": noise_suppression_ratio,
        })

        logger.info(
            f"Classification — Precision: {precision:.4f}, Recall: {recall:.4f}, "
            f"F1: {f1:.4f}, FNR: {false_negative_rate:.4f}, "
            f"NSR: {noise_suppression_ratio:.4f}"
        )

    logger.info(
        f"Distribution — anomaly_rate: {anomaly_rate:.4f}, "
        f"mean_combined_score: {mean_combined_score:.4f}, "
        f"model_confidence: {model_confidence:.4f}"
    )

    save_summary(metrics)
    return metrics


def save_summary(metrics: dict, path: Path = None) -> None:
    """Write evaluation metrics to a human-readable text file."""
    if path is None:
        path = SUMMARY_PATH   # resolved at call time so patches take effect
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 60,
        "Phase 3 — Anomaly Detection Evaluation Summary",
        "=" * 60,
        "",
        "Distribution metrics:",
        f"  anomaly_rate          : {metrics['anomaly_rate']:.4f} "
        f"({metrics['anomaly_rate'] * 100:.1f}%)",
        f"  mean_combined_score   : {metrics['mean_combined_score']:.4f}",
        f"  model_confidence      : {metrics['model_confidence']:.4f}",
        "",
        "Score distribution (combined_score):",
    ]
    dist = metrics["score_distribution"]
    for stat in ("min", "25%", "50%", "75%", "max", "mean", "std"):
        if stat in dist:
            lines.append(f"  {stat:<6}: {dist[stat]:.4f}")

    lines += [
        "",
        "Top 10 templates by mean combined_score:",
    ]
    templates = metrics.get("top_anomalous_templates", [])
    if templates:
        for i, tmpl in enumerate(templates, 1):
            lines.append(f"  {i:2d}. {tmpl}")
    else:
        lines.append("  (template_id not available)")

    if "precision" in metrics:
        lines += [
            "",
            "Classification metrics (ground truth available):",
            f"  Precision             : {metrics['precision']:.4f}",
            f"  Recall                : {metrics['recall']:.4f}",
            f"  F1 Score              : {metrics['f1']:.4f}",
            f"  False negative rate   : {metrics['false_negative_rate']:.4f}",
            f"  Noise suppression     : {metrics['noise_suppression_ratio']:.4f}",
        ]
    else:
        lines += [
            "",
            "Classification metrics: not available (no ground truth in data/synthetic/)",
        ]

    lines.append("=" * 60)

    summary_text = "\n".join(lines)
    print(summary_text)
    path.write_text(summary_text)
    logger.info(f"Evaluation summary saved to {path}.")


def run() -> dict:
    """End-to-end entry point: load parquets → run_evaluation → return metrics."""
    anomaly_df = load_parquet(str(_ANOMALY_OUTPUT_PATH))
    features_df = load_parquet(str(_FEATURES_OUTPUT_PATH))
    return run_evaluation(anomaly_df, features_df)


if __name__ == "__main__":
    run()
