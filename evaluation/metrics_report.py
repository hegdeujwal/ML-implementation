import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

RESULTS_PATH = Path("evaluation/results/metrics_report.txt")


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def run_report(scored_logs_df: pd.DataFrame, ground_truth_df: Optional[pd.DataFrame] = None) -> dict:
    report = {}

    total_logs = len(scored_logs_df)
    ignore_logs = int((scored_logs_df.get("label") == "ignore").sum()) if total_logs else 0
    report["total_logs"] = int(total_logs)
    report["noise_suppression_ratio"] = _safe_ratio(ignore_logs, total_logs)

    if total_logs:
        critical_mask = scored_logs_df.get("label") == "critical"
        critical_total = int(critical_mask.sum())
        critical_misses = int((critical_mask & (scored_logs_df.get("final_score", 0) < 0.5)).sum())
        report["critical_logs_total"] = critical_total
        report["critical_miss_rate"] = _safe_ratio(critical_misses, critical_total)
    else:
        report["critical_logs_total"] = 0
        report["critical_miss_rate"] = 0.0

    if ground_truth_df is not None and not ground_truth_df.empty:
        merged = scored_logs_df.merge(ground_truth_df, on="log_id", how="inner", suffixes=("", "_gt"))
        if "is_anomaly_gt" in merged.columns and "is_anomaly" in merged.columns and len(merged) > 0:
            y_true = merged["is_anomaly_gt"].astype(bool)
            y_pred = merged["is_anomaly"].astype(bool)
            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())
            report["anomaly_precision"] = _safe_ratio(tp, tp + fp)
            report["anomaly_recall"] = _safe_ratio(tp, tp + fn)
            p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
            report["anomaly_f1"] = float(f1)
            report["anomaly_precision_sklearn"] = float(p)
            report["anomaly_recall_sklearn"] = float(r)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = ["METRICS REPORT", f"Generated at epoch: {int(time.time())}"]
    lines.extend([f"{k}: {v}" for k, v in report.items()])
    text = "\n".join(lines)
    print(text)
    RESULTS_PATH.write_text(text, encoding="utf-8")

    return report


if __name__ == "__main__":
    n = 200
    df = pd.DataFrame(
        {
            "log_id": [f"log_{i}" for i in range(n)],
            "label": np.random.choice(["ignore", "low", "medium", "critical"], size=n),
            "final_score": np.random.random(size=n),
            "is_anomaly": np.random.choice([True, False], size=n, p=[0.15, 0.85]),
        }
    )
    gt = pd.DataFrame(
        {
            "log_id": [f"log_{i}" for i in range(n)],
            "is_anomaly_gt": np.random.choice([True, False], size=n, p=[0.1, 0.9]),
        }
    )
    run_report(df, gt)
