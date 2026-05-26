from common.env_handler import get_env

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

# Console log level for all modules using common/logger.py.
# The logger reads this via os.environ["LOG_LEVEL"] to avoid a circular import.
# Set LOG_LEVEL=DEBUG in your .env for verbose output.
LOG_LEVEL: str = "INFO"

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

# PageRank damping factor (standard value; literature range 0.8–0.9).
PAGERANK_ALPHA: float = 0.85

# Betweenness centrality approximation: number of pivot nodes sampled.
# Full exact computation is O(V*E) which is impractical for graphs > 200 nodes.
# k=50 gives a good bias-variance tradeoff for typical network-log graphs.
BETWEENNESS_K: int = 50
BETWEENNESS_LARGE_GRAPH_THRESHOLD: int = 200

# Sequence engine parameters.
# Window within which template B is considered a "follow-on" of template A.
SEQUENCE_WINDOW_SECONDS: int = 30
# A sequence must contain at least this many log templates.
SEQUENCE_MIN_LENGTH: int = 3
# A sequence must appear in at least this many distinct sessions.
SEQUENCE_MIN_SUPPORT: int = 3

# ---------------------------------------------------------------------------
# Phase 3 output paths
# ---------------------------------------------------------------------------
GRAPH_PICKLE_PATH: str = "data/processed/correlation_graph.gpickle"
GRAPH_JSON_PATH: str = "data/processed/correlation_graph.json"
SEQUENCES_JSON_PATH: str = "data/processed/sequences.json"
GRAPH_SCORES_PATH: str = "data/processed/graph_scores_df.parquet"


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

# ---------------------------------------------------------------------------
# Phase 1 — Parsing
# ---------------------------------------------------------------------------

SESSION_GAP_SECONDS: int = 1800  # 30-min inactivity gap within same host → new session

# Daemon/process name → canonical subsystem label.
# Only entries where the raw process name is ambiguous or non-standard need to be listed.
# All other names are upper-cased and used as-is (e.g. "OSPF" → "OSPF").
SERVICE_ALIAS_MAP: dict = {
    "eventmgr":    "SYSTEM",
    "hpe-routing": "ROUTING",
    "kernel":      "SYSTEM",
    "sshd":        "SYSTEM",
    "cron":        "SYSTEM",
    "sudo":        "SYSTEM",
    "snmpd":       "SNMP",
    "lldpd":       "LLDP",
    "cfgd":        "CONFIG",
}


# Statistical features
ZSCORE_ROLLING_WINDOW: int = 60
ZSCORE_MIN_STD: float = 1e-6
BURSTINESS_MIN_EVENTS: int = 2


# Temporal features
INTER_ARRIVAL_EMA_SPAN: int = 5


# Feature pipeline paths
SESSIONIZED_LOGS_PATH: str = (
    "data/processed/sessionized_logs.parquet"
)

FEATURES_OUTPUT_PATH: str = (
    "data/processed/features_df.parquet"
)


# Feature dataframe schema contract
FEATURE_COLUMNS = [
    "sequence_number",
    "session_id",
    "frequency",
    "event_weight",
    "burstiness_score",
    "zscore_base",
    "time_delta_prev",
    "time_delta_session_start",
    "inter_arrival_rate",
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
