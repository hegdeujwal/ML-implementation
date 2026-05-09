"""
anomaly_detector.py
Phase 2 — ML Anomaly Detection Layer
Assignee: Shreeraksha M

Loads features_df.parquet produced by P1, trains an IsolationForest,
computes a hybrid anomaly score (isolation + z-score), and writes
anomaly_df.parquet for P4 (Ujwal) to consume.

Output schema (strict, no nulls):
    log_id          str
    isolation_score float   (0.0 if cold-start fallback)
    zscore          float
    combined_score  float
    is_anomaly      bool
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore as scipy_zscore

# Import shared config and logger
from common.config import (
    CONTAMINATION,
    WEIGHT_ISOLATION,
    WEIGHT_ZSCORE,
    ANOMALY_THRESHOLD,
    MIN_TRAIN_SAMPLES,
)
from common.logger import get_logger


# Paths

DATA_DIR = Path("data/processed")
FEATURES_PATH = DATA_DIR / "features_df.parquet"
ANOMALY_OUTPUT_PATH = DATA_DIR / "anomaly_df.parquet"

logger = get_logger(__name__)

# Feature column selection — rationale

# These five columns cover the three main axes of anomaly signal in network logs:
#
#   frequency_score   — unusually high or low event rates per session are strong
#                       indicators of anomalies (burst floods, silent drops).
#
#   severity_weight   — a high-severity log appearing in an otherwise quiet session
#                       is a clear outlier; IsolationForest will isolate it quickly.
#
#   temporal_delta    — abrupt changes in inter-arrival time (log storms or gaps)
#                       are classic precursors to network incidents.
#
#   counter_proximity — logs that co-occur with interface counter anomalies (drops,
#                       errors) are much more likely to be causally relevant.
#
#   session_id        — intentionally EXCLUDED here. It is a grouping key, not a
#                       numeric feature, and would only add noise.
#
# We start with these 5 because they are the minimum agreed P1→P2 contract.
# Add more as P1 ships them — just extend FEATURE_COLS below.
FEATURE_COLS = [
    "frequency_score",
    "burstiness_score",
    "zscore_base",
    "time_delta_prev",
    "time_delta_session_start",
    "inter_arrival_rate",
    "severity_weight",
    "counter_proximity",
]


def load_features(path: Path = FEATURES_PATH) -> pd.DataFrame:
    """Load the features parquet produced by P1.

    Args:
        path: Path to features_df.parquet.

    Returns:
        DataFrame with at minimum the columns in FEATURE_COLS plus 'log_id'.

    Raises:
        FileNotFoundError: If the parquet does not exist yet.
        KeyError: If required columns are missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"features_df.parquet not found at {path}. "
            "Wait for P1 to run the feature pipeline, or generate synthetic data."
        )

    df = pd.read_parquet(path)
    logger.info(f"Loaded features_df: {len(df)} rows, columns: {list(df.columns)}")

    # Verify required columns exist
    missing = [c for c in ["log_id"] + FEATURE_COLS if c not in df.columns]
    if missing:
        raise KeyError(
            f"features_df is missing required columns: {missing}. "
            "Check with P1 (Sharva) that the feature pipeline has run."
        )

    return df


def select_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the numeric feature columns used for IsolationForest.

    Handles the case where P1 hasn't delivered all columns yet — we silently
    drop missing optional columns and log a warning. The detector can train
    on as few as one feature column, but precision will be lower.
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    dropped = set(FEATURE_COLS) - set(available)
    if dropped:
        logger.warning(
            f"Feature columns not yet available (waiting on P1?): {dropped}. "
            f"Training on available subset: {available}"
        )
    return df[available].copy()


def compute_zscore_base(feature_matrix: pd.DataFrame) -> np.ndarray:
    """Compute a scalar z-score per row as the mean of per-column z-scores.

    This gives a single 'how unusual is this row overall' number using
    pure statistics — no training required. Used both as w2's signal in
    the hybrid score, and as the sole signal during cold-start.

    Returns:
        1-D numpy array of shape (n_rows,), one z-score per log entry.
    """
    # scipy_zscore is column-wise; we average across columns to get a row scalar.
    # nan_policy='omit' handles any stray NaNs from P1 without crashing.
    col_zscores = scipy_zscore(feature_matrix.fillna(0).values, axis=0, nan_policy='omit')
    # Take absolute value so rare lows and rare highs both score high,
    # then mean across feature columns → single anomaly signal per row.
    return np.abs(col_zscores).mean(axis=1)


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Full anomaly detection pipeline.

    Trains IsolationForest on feature_matrix, computes hybrid score, and
    returns anomaly_df with the exact schema required by P4 (Ujwal).

    Cold-start fallback:
        If len(df) < MIN_TRAIN_SAMPLES, IsolationForest cannot produce
        reliable scores (it would overfit a tiny sample). We fall back to
        z-score only and set isolation_score = 0.0. This is logged as a
        WARNING so the team knows the model isn't trained yet.

    Args:
        df: Output of load_features().

    Returns:
        anomaly_df with columns: log_id, isolation_score, zscore,
        combined_score, is_anomaly. Zero nulls guaranteed.
    """
    feature_matrix = select_feature_matrix(df)
    n_samples = len(feature_matrix)

    logger.info(f"Starting anomaly detection on {n_samples} samples.")

    # -----------------------------------------------------------------------
    # Compute z-score first — needed in both paths (normal + cold-start)
    # -----------------------------------------------------------------------
    zscore_arr = compute_zscore_base(feature_matrix)

    # -----------------------------------------------------------------------
    # Cold-start check
    # -----------------------------------------------------------------------
    if n_samples < MIN_TRAIN_SAMPLES:
        logger.warning(
            f"[COLD-START] Only {n_samples} samples available "
            f"(min_train_samples={MIN_TRAIN_SAMPLES}). "
            "IsolationForest skipped — falling back to z-score only. "
            "isolation_score will be 0.0 for all rows."
        )
        isolation_scores = np.zeros(n_samples, dtype=float)
    else:
        # -------------------------------------------------------------------
        # Train IsolationForest
        # -------------------------------------------------------------------
        # contamination: expected fraction of outliers in the dataset.
        # We pull this from config so the team can tune it without touching code.
        # Initial guideline: 0.05 (5%) is a reasonable starting point for
        # network logs — real-world incident rates in healthy networks are low.
        # If the anomaly rate on first run looks wrong (>20% or <1%), adjust
        # contamination in config.py and rerun.
        model = IsolationForest(
            contamination=CONTAMINATION,
            random_state=42,      # reproducibility
            n_estimators=100,     # default, good enough for tabular log features
            n_jobs=-1,            # use all cores — logs can be large
        )
        model.fit(feature_matrix.fillna(0))

        # decision_function returns higher = more normal (positive = inlier).
        # We negate and min-max normalise to get [0,1] where 1 = most anomalous.
        raw_scores = model.decision_function(feature_matrix.fillna(0))
        negated = -raw_scores
        score_min, score_max = negated.min(), negated.max()
        if score_max > score_min:
            isolation_scores = (negated - score_min) / (score_max - score_min)
        else:
            # Edge case: all scores identical → no anomalies distinguishable
            isolation_scores = np.zeros(n_samples, dtype=float)

        logger.info(
            f"IsolationForest trained. "
            f"Isolation score range: [{isolation_scores.min():.4f}, {isolation_scores.max():.4f}]"
        )

    # -----------------------------------------------------------------------
    # Normalise z-score to [0, 1] for fair weighting with isolation_score
    # -----------------------------------------------------------------------
    z_min, z_max = zscore_arr.min(), zscore_arr.max()
    if z_max > z_min:
        zscore_norm = (zscore_arr - z_min) / (z_max - z_min)
    else:
        zscore_norm = np.zeros(n_samples, dtype=float)

    # -----------------------------------------------------------------------
    # Hybrid combined score
    # -----------------------------------------------------------------------
    # combined_score = w1 * isolation_score + w2 * zscore_norm
    #
    # Weight rationale (initial values from config):
    #   w1 (isolation) = 0.65 — IsolationForest captures multi-dimensional
    #       interactions between features that z-score misses. Deserves
    #       higher weight once the model is trained.
    #   w2 (zscore)    = 0.35 — z-score is model-free, interpretable, and
    #       essential during cold-start. Lower weight because it treats each
    #       feature independently.
    #
    # These weights come from config so Ujwal (P4) can tune them without
    # touching this file. Document any change in the model JSON sidecar.
    combined_scores = (WEIGHT_ISOLATION * isolation_scores) + (WEIGHT_ZSCORE * zscore_norm)

    # -----------------------------------------------------------------------
    # Anomaly flag
    # -----------------------------------------------------------------------
    is_anomaly = combined_scores > ANOMALY_THRESHOLD

    logger.info(
        f"Anomaly detection complete. "
        f"Anomaly rate: {is_anomaly.sum()}/{n_samples} "
        f"({100 * is_anomaly.mean():.1f}%)"
    )

    if is_anomaly.mean() > 0.3:
        logger.warning(
            "Anomaly rate exceeds 30% — consider raising ANOMALY_THRESHOLD or "
            "lowering CONTAMINATION in config.py."
        )

    # -----------------------------------------------------------------------
    # Assemble output DataFrame — strict schema, no nulls
    # -----------------------------------------------------------------------
    anomaly_df = pd.DataFrame({
        "log_id": df["log_id"].astype(str).values,
        "isolation_score": isolation_scores.astype(float),
        "zscore": zscore_arr.astype(float),          # raw (not normalised) for interpretability
        "combined_score": combined_scores.astype(float),
        "is_anomaly": is_anomaly.astype(bool),
    })

    # Guarantee no nulls — fail fast rather than silently ship bad data to P4
    null_counts = anomaly_df.isnull().sum()
    if null_counts.any():
        raise ValueError(
            f"anomaly_df contains nulls — this violates the P4 contract:\n{null_counts}"
        )

    return anomaly_df


def save_anomaly_df(anomaly_df: pd.DataFrame, path: Path = ANOMALY_OUTPUT_PATH) -> None:
    """Persist anomaly_df as parquet for P4 (Ujwal) to consume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    anomaly_df.to_parquet(path, index=False)
    logger.info(f"anomaly_df saved to {path} ({len(anomaly_df)} rows).")


def run(features_path: Path = FEATURES_PATH, output_path: Path = ANOMALY_OUTPUT_PATH) -> pd.DataFrame:
    """End-to-end entry point: load → detect → save.

    Call this from trainer.py or directly via `python -m ml.anomaly_detector`.

    Returns:
        anomaly_df for in-memory use (e.g. by trainer.py or tests).
    """
    df = load_features(features_path)
    anomaly_df = detect_anomalies(df)
    save_anomaly_df(anomaly_df, output_path)
    return anomaly_df


if __name__ == "__main__":
    run()