"""
anomaly_detector.py
===================
Phase 3 — ML Anomaly Detection
Assignee: Shreeraksha M

Loads features_df.parquet produced by P2 (features stage), trains a
StandardScaler + IsolationForest pipeline, computes a confidence-weighted
hybrid anomaly score, and writes anomaly_df.parquet for P4 (scoring) to consume.

Output schema (strict, no nulls):
    sequence_number   int
    isolation_score   float   [0.0, 1.0]
    zscore_norm       float   [0.0, 1.0]
    combined_score    float   [0.0, 1.0]
    is_anomaly        bool
    model_confidence  float   [0.0, 1.0]

Scoring formula:
    confidence       = min(1.0, n_training_samples / COLD_START_FULL_CONFIDENCE_THRESHOLD)
    combined_score   = confidence  * (IF_ISOLATION_WEIGHT * isolation_score
                                      + IF_ZSCORE_WEIGHT  * zscore_norm)
                     + (1-confidence) * zscore_norm

At full confidence (n >= threshold) this reduces to the standard hybrid score.
At zero confidence it degrades to pure zscore_norm, which requires no trained model.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common.config import (
    IF_CONTAMINATION,
    IF_FEATURE_COLUMNS,
    IF_ISOLATION_WEIGHT,
    IF_N_ESTIMATORS,
    IF_RANDOM_STATE,
    IF_ZSCORE_WEIGHT,
    COLD_START_FULL_CONFIDENCE_THRESHOLD,
    ANOMALY_SCORE_THRESHOLD,
    ANOMALY_DYNAMIC_K,
    ANOMALY_FLAG_MODE,
    ANOMALY_CONTAMINATION,
)
from common.logger import get_logger
from common.utils import load_parquet, save_parquet, validate_schema

DATA_DIR = Path("data/processed")
FEATURES_PATH = DATA_DIR / "features_df.parquet"
ANOMALY_OUTPUT_PATH = DATA_DIR / "anomaly_df.parquet"

logger = get_logger(__name__)

# Required input columns: all IF features + the join key + zscore_base explicitly.
# zscore_base is already in IF_FEATURE_COLUMNS but listed here for documentation clarity.
# dict.fromkeys preserves order and deduplicates.
_REQUIRED_INPUT_COLS: list[str] = list(
    dict.fromkeys(IF_FEATURE_COLUMNS + ["sequence_number", "zscore_base"])
)

OUTPUT_COLUMNS: list[str] = [
    "sequence_number",
    "isolation_score",
    "zscore_norm",
    "combined_score",
    "is_anomaly",
    "model_confidence",
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _train_model(features_df: pd.DataFrame) -> Pipeline:
    """Fit a StandardScaler → IsolationForest pipeline on features_df.

    The scaler ensures that features with different natural ranges
    (e.g. time_delta_session_start in [0, 18000] vs frequency_score in [0, 1])
    contribute equally to the isolation trees.

    Sets pipeline.n_samples_seen_ so callers can recover confidence scaling
    without a separate sidecar lookup.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("if", IsolationForest(
            contamination=IF_CONTAMINATION,
            n_estimators=IF_N_ESTIMATORS,
            random_state=IF_RANDOM_STATE,
            n_jobs=-1,
        )),
    ])
    pipeline.fit(features_df[IF_FEATURE_COLUMNS])
    pipeline.n_samples_seen_ = len(features_df)

    # Persist a score calibration from the TRAINING distribution so inference
    # batches are mapped through a fixed reference instead of their own
    # min-max. Per-batch min-max guarantees some row scores 1.0 even on a
    # fully healthy batch and makes scores incomparable across runs. Robust
    # percentiles so a single training outlier can't stretch the range.
    raw_train = pipeline.decision_function(features_df[IF_FEATURE_COLUMNS])
    pipeline.calibration_ = {
        "raw_min": float(np.quantile(raw_train, 0.005)),
        "raw_max": float(np.quantile(raw_train, 0.995)),
    }

    logger.info(
        f"Trained IsolationForest pipeline on {len(features_df):,} samples, "
        f"{len(IF_FEATURE_COLUMNS)} features "
        f"(calibration raw range [{pipeline.calibration_['raw_min']:.4f}, "
        f"{pipeline.calibration_['raw_max']:.4f}])."
    )
    return pipeline


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(
    features_df: pd.DataFrame,
    model_path: str = None,
) -> pd.DataFrame:
    """Detect anomalies on a features DataFrame.

    Args:
        features_df: Output of the features pipeline stage. Must contain all
            IF_FEATURE_COLUMNS, sequence_number, and zscore_base.
        model_path: Optional path to a saved Pipeline pkl. If provided and the
            file exists the model is loaded; its JSON sidecar (same stem, .json
            extension) is read for n_samples. Otherwise a fresh model is trained
            on features_df.

    Returns:
        anomaly_df with exactly OUTPUT_COLUMNS, zero nulls.

    Raises:
        ValueError: If required input columns are missing or output contains nulls.
    """
    # ------------------------------------------------------------------
    # Step 1 — load or train model
    # ------------------------------------------------------------------
    if model_path is not None and Path(model_path).exists():
        pipeline: Pipeline = joblib.load(model_path)
        sidecar = Path(model_path).with_suffix(".json")
        if sidecar.exists():
            with open(sidecar) as fh:
                meta = json.load(fh)
            n_samples = int(meta.get("n_samples", getattr(pipeline, "n_samples_seen_", len(features_df))))
        else:
            n_samples = int(getattr(pipeline, "n_samples_seen_", len(features_df)))
        logger.info(f"Loaded model from {model_path} (n_samples={n_samples:,})")
    else:
        pipeline = _train_model(features_df)
        n_samples = len(features_df)

    # ------------------------------------------------------------------
    # Step 2 — validate input; drop rows with NaN/inf in feature columns
    # ------------------------------------------------------------------
    validate_schema(features_df, _REQUIRED_INPUT_COLS)

    feature_values = features_df[IF_FEATURE_COLUMNS].values
    finite_mask = np.isfinite(feature_values).all(axis=1)
    n_dropped = int((~finite_mask).sum())
    if n_dropped > 0:
        logger.warning(
            f"Dropping {n_dropped} row(s) with NaN or inf in IF feature columns "
            "before scoring."
        )
    clean = features_df.loc[finite_mask].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Step 3 — isolation score normalised to [0, 1] (1 = most anomalous)
    # ------------------------------------------------------------------
    X = clean[IF_FEATURE_COLUMNS]
    raw = pipeline.decision_function(X)           # higher = more normal

    # Prefer the calibration captured at training time: it keeps scores
    # comparable across batches/runs and lets a healthy batch score uniformly
    # low. Per-batch min-max is only the fallback for models saved before
    # calibration existed (or a degenerate calibration range).
    cal = getattr(pipeline, "calibration_", None)
    if cal and cal["raw_max"] > cal["raw_min"]:
        isolation_score = np.clip(
            1.0 - (raw - cal["raw_min"]) / (cal["raw_max"] - cal["raw_min"]),
            0.0,
            1.0,
        )
        logger.info(
            "Isolation scores mapped via training calibration "
            f"[{cal['raw_min']:.4f}, {cal['raw_max']:.4f}]."
        )
    else:
        raw_min, raw_max = float(raw.min()), float(raw.max())
        if raw_max > raw_min:
            isolation_score = 1.0 - (raw - raw_min) / (raw_max - raw_min)
            logger.warning(
                "No training calibration on model — falling back to per-batch "
                "min-max normalisation (scores not comparable across runs)."
            )
        else:
            # All scores identical — no anomaly signal from IF; fall back to midpoint.
            isolation_score = np.full(len(clean), 0.5, dtype=float)

    # ------------------------------------------------------------------
    # Step 4 — normalise zscore_base to [0, 1], direction-agnostic
    # ------------------------------------------------------------------
    # |z| so a template that suddenly goes quiet (z < 0, e.g. a dying
    # heartbeat) scores as anomalous as one that spikes. The previous
    # (z+5)/10 mapping sent z=-5 to 0.0 — the least anomalous value.
    zscore_norm: np.ndarray = (
        clean["zscore_base"].clip(-5.0, 5.0).abs().values
    ) / 5.0

    # ------------------------------------------------------------------
    # Step 5 — confidence-scaled combined score
    # ------------------------------------------------------------------
    confidence = min(1.0, n_samples / COLD_START_FULL_CONFIDENCE_THRESHOLD)
    combined_score = np.clip(
        confidence * (IF_ISOLATION_WEIGHT * isolation_score + IF_ZSCORE_WEIGHT * zscore_norm)
        + (1.0 - confidence) * zscore_norm,
        0.0,
        1.0,
    )

    # ------------------------------------------------------------------
    # Step 6 — anomaly flag
    # ------------------------------------------------------------------
    _score_std = float(combined_score.std())
    if _score_std < 1e-6:
        # All scores identical — no signal; use the static fallback threshold.
        _threshold = ANOMALY_SCORE_THRESHOLD
        is_anomaly = combined_score > _threshold
        logger.info(
            f"Anomaly threshold: {_threshold:.4f} (static fallback — score std≈0)"
        )
    elif ANOMALY_FLAG_MODE == "absolute":
        # Fixed threshold on the calibrated combined_score. Unlike quantile
        # mode this can flag NOTHING on a healthy batch — which is the point.
        # Requires training-calibrated isolation scores to be meaningful
        # across runs.
        _threshold = ANOMALY_SCORE_THRESHOLD
        is_anomaly = combined_score > _threshold
        logger.info(
            f"Anomaly threshold: {_threshold:.4f} (absolute mode)"
        )
    elif ANOMALY_FLAG_MODE == "quantile":
        # Flag the top ANOMALY_CONTAMINATION fraction by combined_score. This
        # guarantees a stable, non-zero anomaly rate tied to the batch's own score
        # distribution rather than a mean+kσ assumption that can flag nothing.
        _threshold = float(np.quantile(combined_score, 1.0 - ANOMALY_CONTAMINATION))
        is_anomaly = combined_score >= _threshold
        logger.info(
            f"Anomaly threshold: {_threshold:.4f}  "
            f"(quantile mode, contamination={ANOMALY_CONTAMINATION:.2f})"
        )
    else:  # "dynamic_k" — legacy mean + k·std
        _threshold = float(combined_score.mean()) + ANOMALY_DYNAMIC_K * _score_std
        is_anomaly = combined_score > _threshold
        logger.info(
            f"Anomaly threshold: {_threshold:.4f}  "
            f"(dynamic_k mode, mean={combined_score.mean():.4f}, "
            f"std={_score_std:.4f}, k={ANOMALY_DYNAMIC_K})"
        )

    # ------------------------------------------------------------------
    # Step 7 — assemble and validate output
    # ------------------------------------------------------------------
    anomaly_df = pd.DataFrame({
        "sequence_number":  clean["sequence_number"].values,
        "isolation_score":  isolation_score.astype(float),
        "zscore_norm":      zscore_norm.astype(float),
        "combined_score":   combined_score.astype(float),
        "is_anomaly":       is_anomaly.astype(bool),
        "model_confidence": float(confidence),
    })

    null_counts = anomaly_df.isnull().sum()
    if null_counts.any():
        raise ValueError(
            f"anomaly_df contains nulls — schema contract violated:\n{null_counts}"
        )

    # Print summary to stdout
    logger.info(f"Shape: {anomaly_df.shape}")
    logger.info(
        f"Anomaly rate: {is_anomaly.mean():.4f}  ({int(is_anomaly.sum())} / {len(anomaly_df)})"
    )
    logger.info(f"Model confidence: {confidence:.4f}")
    top10 = anomaly_df.nlargest(10, "combined_score")[["sequence_number", "combined_score"]]
    logger.info(f"Top 10 by combined_score:\n{top10.to_string(index=False)}")

    return anomaly_df


def run(
    features_path: Path = FEATURES_PATH,
    output_path: Path = ANOMALY_OUTPUT_PATH,
) -> pd.DataFrame:
    """End-to-end entry point: load -> detect (with saved model) -> retrain if needed -> save.

    Model lifecycle (cross-run):
    1. Load rolling feature store (last N sessions across all runs) for training.
    2. Load + validate the latest saved IsolationForest via AnomalyTrainer.
    3. Run detect() -- uses saved model if valid, otherwise trains fresh.
    4. Call AnomalyTrainer.maybe_retrain() -- retrains on rolling store if K-log
       trigger fires, persisting the new model for subsequent runs.

    Falls back gracefully to fresh training on first run (cold start).

    Returns:
        anomaly_df for in-memory use by the next pipeline stage.
    """
    import os
    import tempfile

    import joblib

    import common.config as cfg
    from ml.trainer import AnomalyTrainer

    df = load_parquet(str(features_path))
    logger.info("Loaded %d rows from %s", len(df), features_path)

    trainer = AnomalyTrainer()

    # Load rolling feature store for training (cross-run history)
    rolling_path = Path(cfg.FEATURE_ROLLING_STORE_PATH)
    if rolling_path.exists():
        train_df = load_parquet(str(rolling_path))
        logger.info(
            "Loaded rolling feature store: %d rows for IF training.", len(train_df)
        )
    else:
        train_df = df
        logger.info(
            "No rolling feature store found -- cold start, training on current batch (%d rows).",
            len(df),
        )

    # Load latest saved model (validated against sidecar)
    latest_pipeline = trainer.load_latest_model()
    tmp_model_path: str | None = None
    if latest_pipeline is not None:
        tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        tmp.close()
        joblib.dump(latest_pipeline, tmp.name)
        tmp_model_path = tmp.name
        logger.info("Using validated saved model for inference.")
    else:
        logger.info("No valid saved model -- will train fresh for inference.")

    # Detect anomalies
    try:
        anomaly_df = detect(df, model_path=tmp_model_path)
    finally:
        if tmp_model_path:
            try:
                os.unlink(tmp_model_path)
            except OSError:
                pass

    # Retrain model if the periodic K-log trigger fires
    retrained = trainer.maybe_retrain(train_df, new_logs_count=len(df))
    if retrained is not None:
        logger.info(
            "AnomalyTrainer retrained on %d rows. New model saved to %s.",
            len(train_df),
            cfg.MODEL_STORE_PATH,
        )
    else:
        logger.info("No retraining triggered this run.")

    save_parquet(anomaly_df, str(output_path))
    logger.info("anomaly_df saved -> %s (%d rows)", output_path, len(anomaly_df))

    return anomaly_df


if __name__ == "__main__":
    run()
