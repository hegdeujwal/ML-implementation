"""
evaluation/oracle_report.py
===========================
Evaluate pipeline output against the synthetic dataset's ground truth —
this closes the loop that scenario_labels.parquet was created for.

Ground-truth definition
-----------------------
Log level   : a row is "signal" when its log_level is in
              ORACLE_TRUTH_SEVERITIES (CRITICAL / HIGH / ERROR).
              Severity is deliberately excluded from IF_FEATURE_COLUMNS,
              so the anomaly stage is judged fairly against it.
              final_score carries a SCORING_SEVERITY_WEIGHT term, so the
              ranking metrics are partially favoured by construction —
              treat them as upper bounds.
Scenario    : scenario_labels.parquet (Section 7 of each file) provides one
              training_label per scenario. A scenario counts as "detected"
              when at least one of its signal logs is flagged anomalous.

Metrics
-------
anomaly stage   precision / recall / F1 of is_anomaly vs signal truth,
                plus flagged-vs-truth rate comparison (exposes a mis-set
                ANOMALY_CONTAMINATION immediately).
ranking         recall@k of final_score where k = number of truth-signal
                rows (R-precision), and mean-score separation between
                signal and noise rows.
labels          critical_capture_rate — truth-signal rows labelled
                medium/critical; noise_suppression_ratio — non-signal rows
                labelled ignore.
incidents       fraction of truth-signal rows assigned a correlation_id.
per-scenario    signal recall and detection flag per scenario file.

Public API
----------
run_oracle_report(...) -> dict
    Loads the parquets, computes all metrics, writes
    evaluation/results/oracle_report.txt, and returns the metrics dict.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import common.config as cfg
from common.logger import get_logger
from common.utils import load_parquet

logger = get_logger(__name__)

# Pipeline labels that count as "captured" for a truth-signal row.
_CAPTURE_LABELS = {"medium", "critical"}


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _prf(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Precision / recall / F1 without sklearn (avoids zero-division warnings)."""
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = _safe_ratio(2 * precision * recall, precision + recall)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def run_oracle_report(
    scored_path: str = "data/processed/scored_logs_df.parquet",
    anomaly_path: str = "data/processed/anomaly_df.parquet",
    sessionized_path: str = "data/processed/sessionized_logs.parquet",
    labels_path: str = cfg.SCENARIO_LABELS_PATH,
    output_path: str = cfg.ORACLE_REPORT_PATH,
) -> dict:
    """Compute oracle metrics for the current pipeline run.

    Raises
    ------
    FileNotFoundError
        If any required parquet is missing (callers that want optional
        behaviour should check labels_path themselves before calling).
    """
    session_df = load_parquet(sessionized_path)
    anomaly_df = load_parquet(anomaly_path)
    scored_df = load_parquet(scored_path)
    labels_df = load_parquet(labels_path)

    # ------------------------------------------------------------------
    # Build the evaluation frame: one row per log with truth + predictions
    # ------------------------------------------------------------------
    truth_levels = {lv.upper() for lv in cfg.ORACLE_TRUTH_SEVERITIES}
    base_cols = ["sequence_number", "log_level"]
    if "scenario_id" in session_df.columns:
        base_cols.append("scenario_id")
    df = session_df[base_cols].copy()
    df["is_signal"] = df["log_level"].str.upper().isin(truth_levels)

    df = df.merge(
        anomaly_df[["sequence_number", "is_anomaly", "combined_score"]],
        on="sequence_number", how="left",
    )
    scored_cols = ["sequence_number", "final_score", "label", "correlation_id"]
    scored_cols = [c for c in scored_cols if c in scored_df.columns]
    df = df.merge(scored_df[scored_cols], on="sequence_number", how="left")

    # Rows dropped by the anomaly stage (NaN/inf features) arrive as NaN here.
    # They are counted as not-flagged: losing a signal row to a feature bug is
    # a miss, and the report should say so.
    n_unscored = int(df["is_anomaly"].isna().sum())
    df["is_anomaly"] = df["is_anomaly"].fillna(False).astype(bool)

    metrics: dict = {
        "total_logs": int(len(df)),
        "truth_signal_count": int(df["is_signal"].sum()),
        "truth_signal_rate": float(df["is_signal"].mean()),
        "unscored_rows": n_unscored,
    }

    # ------------------------------------------------------------------
    # 1. Anomaly stage vs truth
    # ------------------------------------------------------------------
    metrics["anomaly_flagged_count"] = int(df["is_anomaly"].sum())
    metrics["anomaly_flagged_rate"] = float(df["is_anomaly"].mean())
    metrics.update(
        {f"anomaly_{k}": v for k, v in _prf(df["is_signal"], df["is_anomaly"]).items()}
    )

    # ------------------------------------------------------------------
    # 2. Ranking quality of final_score (R-precision: recall@k, k = n_signal)
    # ------------------------------------------------------------------
    if "final_score" in df.columns and df["final_score"].notna().any():
        k = metrics["truth_signal_count"]
        if k > 0:
            top_k = df.nlargest(k, "final_score")
            metrics["ranking_recall_at_k"] = float(top_k["is_signal"].mean())
        else:
            metrics["ranking_recall_at_k"] = 0.0
        sig_scores = df.loc[df["is_signal"], "final_score"]
        noise_scores = df.loc[~df["is_signal"], "final_score"]
        metrics["mean_final_score_signal"] = float(sig_scores.mean()) if len(sig_scores) else 0.0
        metrics["mean_final_score_noise"] = float(noise_scores.mean()) if len(noise_scores) else 0.0
        metrics["score_separation"] = (
            metrics["mean_final_score_signal"] - metrics["mean_final_score_noise"]
        )

    # ------------------------------------------------------------------
    # 3. Label quality
    # ------------------------------------------------------------------
    if "label" in df.columns:
        sig = df[df["is_signal"]]
        noise = df[~df["is_signal"]]
        metrics["critical_capture_rate"] = _safe_ratio(
            int(sig["label"].isin(_CAPTURE_LABELS).sum()), len(sig)
        )
        metrics["noise_suppression_ratio"] = _safe_ratio(
            int((noise["label"] == "ignore").sum()), len(noise)
        )

    # ------------------------------------------------------------------
    # 4. Incident coverage of signal rows
    # ------------------------------------------------------------------
    if "correlation_id" in df.columns:
        sig = df[df["is_signal"]]
        metrics["signal_incident_coverage"] = _safe_ratio(
            int(sig["correlation_id"].notna().sum()), len(sig)
        )

    # ------------------------------------------------------------------
    # 5. Per-scenario breakdown (only when the section-aware loader ran)
    # ------------------------------------------------------------------
    per_scenario: list[dict] = []
    if "scenario_id" in df.columns:
        label_lookup = (
            labels_df.set_index("scenario_id")["training_label"].to_dict()
            if "scenario_id" in labels_df.columns else {}
        )
        for sid, grp in df.groupby("scenario_id"):
            sig = grp[grp["is_signal"]]
            flagged = int(sig["is_anomaly"].sum())
            per_scenario.append({
                "scenario_id": str(sid),
                "training_label": label_lookup.get(sid, "UNKNOWN"),
                "n_logs": int(len(grp)),
                "n_signal": int(len(sig)),
                "n_signal_flagged": flagged,
                "signal_recall": _safe_ratio(flagged, len(sig)),
                "detected": bool(flagged > 0),
            })
        n_detectable = sum(1 for s in per_scenario if s["n_signal"] > 0)
        n_detected = sum(1 for s in per_scenario if s["detected"])
        metrics["scenario_detection_rate"] = _safe_ratio(n_detected, n_detectable)
        metrics["scenarios_detected"] = n_detected
        metrics["scenarios_total"] = len(per_scenario)
    metrics["per_scenario"] = per_scenario

    _write_report(metrics, output_path)
    logger.info(
        "Oracle report: anomaly P=%.3f R=%.3f F1=%.3f | recall@k=%.3f | "
        "scenario detection %s/%s — full report at %s",
        metrics.get("anomaly_precision", 0.0),
        metrics.get("anomaly_recall", 0.0),
        metrics.get("anomaly_f1", 0.0),
        metrics.get("ranking_recall_at_k", 0.0),
        metrics.get("scenarios_detected", "-"),
        metrics.get("scenarios_total", "-"),
        output_path,
    )
    return metrics


def _write_report(metrics: dict, output_path: str) -> None:
    """Render the metrics dict as a readable text report."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "ORACLE EVALUATION REPORT",
        f"truth = log_level in {cfg.ORACLE_TRUTH_SEVERITIES} "
        "(severity is excluded from IF features; final_score ranking metrics "
        "are partially favoured by the severity term)",
        "",
    ]
    for key, value in metrics.items():
        if key == "per_scenario":
            continue
        if isinstance(value, float):
            lines.append(f"{key}: {value:.4f}")
        else:
            lines.append(f"{key}: {value}")

    if metrics["per_scenario"]:
        lines.append("")
        lines.append("Per-scenario breakdown:")
        scenario_df = pd.DataFrame(metrics["per_scenario"])
        lines.append(scenario_df.to_string(index=False))

    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_oracle_report()
