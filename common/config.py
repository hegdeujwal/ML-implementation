from common.env_handler import get_env

# ---------------------------------------------------------------------------
# Correlation / Graph parameters
# ---------------------------------------------------------------------------

# Time window (seconds) within which two log events are considered co-occurring.
# Widening this produces a denser graph; narrowing it produces a sparser one.
CORRELATION_TIME_WINDOW_SECONDS: int = 60

# Hard cap on the number of template nodes admitted into the correlation graph.
# Only the MAX_GRAPH_NODES most-frequent templates are kept; the rest are
# silently dropped before edge construction begins.  Raising this limit
# increases memory and CPU cost roughly as O(N^2) in the worst case because
# the sliding-window join visits every pair of events that fall inside the
# same time window.  500 is a safe default for a single-host deployment;
# reduce to ~100 for memory-constrained environments or increase carefully
# after profiling.
MAX_GRAPH_NODES: int = 500


# ---------------------------------------------------------------------------
# Dynamic environment-variable access (credentials, service URLs, etc.)
# ---------------------------------------------------------------------------

def __getattr__(name: str):
    """
    Dynamically fetch environment variables when they are accessed.
    This allows lazy evaluation: variables are only checked when actually imported/used.
    """
    if name.startswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    # Fetch the requested variable (e.g., DB_URL, ELASTIC_URL) directly from the .env file
    return get_env(name)

ML_CONFIG = {
    "contamination": 0.05,   # % of anomalies expected
    "weight_isolation": 0.7,
    "weight_zscore": 0.3,
    "training_window_sessions": 50
}

# ----------------------------
# SCORING WEIGHTS
# Controls contribution of each signal to final score
# ----------------------------

# ----------------------------
# Weights for final importance score
# ----------------------------
ML_WEIGHT: float = 0.4
GRAPH_WEIGHT: float = 0.35
RULE_WEIGHT: float = 0.25


# ----------------------------
# Label thresholds
# ----------------------------
LABEL_THRESHOLDS = {
    "ignore": (0.0, 0.2),
    "low": (0.2, 0.5),
    "medium": (0.5, 0.75),
    "critical": (0.75, 1.0),
}

# ----------------------------
# DBSCAN parameters (for clustering)
# ----------------------------
DBSCAN_EPS: float = 0.5
DBSCAN_MIN_SAMPLES: int = 5

# ---------------------------------------------------------------------------
# P1 Feature Engineering Configuration
# ---------------------------------------------------------------------------

# Severity weights
SEVERITY_WEIGHTS = {
    "CRITICAL": 1.0,
    "ERROR": 0.7,
    "WARN": 0.4,
    "INFO": 0.1,
}

DEFAULT_SEVERITY_WEIGHT: float = 0.1


# Counter anomaly proximity
COUNTER_PROXIMITY_WINDOW_SECONDS: int = 30
COUNTER_PROXIMITY_DECAY_RATE: float = 0.05


# Statistical features
ZSCORE_ROLLING_WINDOW: int = 20
ZSCORE_MIN_STD: float = 1e-6
BURSTINESS_MIN_EVENTS: int = 2


# Temporal features
INTER_ARRIVAL_ROLLING_WINDOW: int = 5


# Feature pipeline paths
SESSIONIZED_LOGS_PATH: str = (
    "parsing/processed/sessionized_logs.parquet"
)

FEATURES_OUTPUT_PATH: str = (
    "data/processed/features_df.parquet"
)


# Feature dataframe schema contract
FEATURE_COLUMNS = [
    "log_id",
    "session_id",
    "frequency_score",
    "burstiness_score",
    "zscore_base",
    "time_delta_prev",
    "severity_weight",
    "counter_proximity",
]

# ---------------------------------------------------------------------------
# Phase 2 — ML Anomaly Detection
# ---------------------------------------------------------------------------
 
# IsolationForest contamination: expected fraction of anomalies in the dataset.
# 0.05 = 5% anomaly rate assumption. Adjust if first-run anomaly rate looks off.
# If you change this, document it in the JSON sidecar saved alongside the model.
CONTAMINATION: float = 0.05
 
# Hybrid score weights: combined_score = w1 * isolation_score + w2 * zscore_norm
# Must sum to 1.0. w1 > w2 because IsolationForest captures feature interactions
# that z-score misses; but z-score keeps the system interpretable.
WEIGHT_ISOLATION: float = 0.65   # w1
WEIGHT_ZSCORE: float = 0.35      # w2
 
# A log is flagged is_anomaly=True when combined_score > this threshold.
# 0.5 = middle of [0,1] range; tune upward to reduce false positives.
ANOMALY_THRESHOLD: float = 0.5
 
# Minimum number of log rows needed before IsolationForest training is attempted.
# Below this, the system falls back to z-score only (cold-start mode).
MIN_TRAIN_SAMPLES: int = 50
 
# Sliding window retraining: only use logs from the last N sessions.
# Prevents the model from memorising stale historical patterns.
TRAINING_WINDOW_SESSIONS: int = 10
 
# Periodic retraining trigger: retrain every K new log rows ingested.
# Lower K = fresher model but more compute. Start high, tune down if needed.
RETRAIN_EVERY_K_LOGS: int = 500
 
# ---------------------------------------------------------------------------
# Phase 4 — Importance Scoring (P4: Ujwal Hegde)
# ---------------------------------------------------------------------------
 
# Weights for the final importance score (ML + graph + rule-based signals)
SCORING_WEIGHT_ML: float = 0.40
SCORING_WEIGHT_GRAPH: float = 0.35
SCORING_WEIGHT_RULE: float = 0.25
 
# Label thresholds: ignore / low / medium / critical
LABEL_IGNORE_MAX: float = 0.2
LABEL_LOW_MAX: float = 0.5
LABEL_MEDIUM_MAX: float = 0.75
# Anything above LABEL_MEDIUM_MAX → critical
 
# DBSCAN clustering parameters (incident_clusterer.py)
DBSCAN_EPS: float = 0.3
DBSCAN_MIN_SAMPLES: int = 5