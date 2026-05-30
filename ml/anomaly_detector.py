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
    logger.info(
        f"Trained IsolationForest pipeline on {len(features_df):,} samples, "
        f"{len(IF_FEATURE_COLUMNS)} features."
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
    raw_min, raw_max = float(raw.min()), float(raw.max())
    if raw_max > raw_min:
        isolation_score = 1.0 - (raw - raw_min) / (raw_max - raw_min)
    else:
        # All scores identical — no anomaly signal from IF; fall back to midpoint.
        isolation_score = np.full(len(clean), 0.5, dtype=float)

    # ------------------------------------------------------------------
    # Step 4 — normalise zscore_base to [0, 1]
    # ------------------------------------------------------------------
    zscore_norm: np.ndarray = (
        clean["zscore_base"].clip(-5.0, 5.0).values + 5.0
    ) / 10.0

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
    is_anomaly = combined_score > ANOMALY_SCORE_THRESHOLD

    # ------------------------------------------------------------------
    # Step 7 — assemble and validate output
    # ------------------------------------------------------------------
    anomaly_df = pd.DataFrame({
        "sequence_number":  clean["sequence_number"].values,
        "log_id":           clean["sequence_number"].map(
            lambda x: f"log_{int(x)}"
        ),
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
    print(f"\nShape: {anomaly_df.shape}")
    print(f"Anomaly rate: {is_anomaly.mean():.4f}  ({int(is_anomaly.sum())} / {len(anomaly_df)})")
    print(f"Model confidence: {confidence:.4f}")
    print("\nTop 10 by combined_score:")
    print(
        anomaly_df.nlargest(10, "combined_score")[["sequence_number", "combined_score"]]
        .to_string(index=False)
    )

    return anomaly_df


def run(
    features_path: Path = FEATURES_PATH,
    output_path: Path = ANOMALY_OUTPUT_PATH,
) -> pd.DataFrame:
    """End-to-end entry point: load → detect → save.

    Called by pipeline.py. Signature must remain stable.

    Returns:
        anomaly_df for in-memory use by the next pipeline stage.
    """
    df = load_parquet(str(features_path))
    logger.info(f"Loaded {len(df):,} rows from {features_path}")

    anomaly_df = detect(df)

    save_parquet(anomaly_df, str(output_path))
    logger.info(f"anomaly_df saved → {output_path} ({len(anomaly_df):,} rows)")

    return anomaly_df


if __name__ == "__main__":
    run()
