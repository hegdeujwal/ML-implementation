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

# ---------------------------------------------------------------------------
# Graph construction — canonical names
# ---------------------------------------------------------------------------

# Time window (seconds) within which two log events are considered co-occurring.
# Widening this produces a denser graph; narrowing it produces a sparser one.
GRAPH_COOCCURRENCE_WINDOW_SECONDS: int = 60

# Hard cap on unique templates admitted into the co-occurrence graph.
# Only the top-N most-frequent templates are kept; the rest are excluded
# before edge construction.  500 is a safe default for a single-host
# deployment; reduce to ~100 for memory-constrained environments.
GRAPH_MAX_NODES: int = 500

# PageRank damping factor (standard value; literature range 0.8–0.9).
GRAPH_PAGERANK_ALPHA: float = 0.85

# Backward-compatible aliases — kept so existing imports don't break.
CORRELATION_TIME_WINDOW_SECONDS: int = GRAPH_COOCCURRENCE_WINDOW_SECONDS
MAX_GRAPH_NODES: int = GRAPH_MAX_NODES
PAGERANK_ALPHA: float = GRAPH_PAGERANK_ALPHA

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
SEQUENCE_MIN_SUPPORT: int = 5

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

# All current ingestion is HPE CX switch logs — revisit when multi-device support is added
DEFAULT_SOURCE_TYPE: str = "switch"

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

# Feature engineering — zscore baseline
ZSCORE_BASELINE_N_SESSIONS: int = 20  # rolling window: last N sessions per host

# Feature engineering — inter arrival rate
IAR_EMA_ALPHA: float = 0.3  # EMA smoothing factor, per session scope only

# Counter proximity — regex patterns that identify counter/interface anomaly templates
COUNTER_ANOMALY_PATTERNS: list = [
    r"INTERFACE_.*THRESHOLD",
    r"INTERFACE_.*ERROR.*EXCEED",
    r"INTERFACE_.*DROP.*EXCEED",
]
# Hint keywords — templates containing these but not matching COUNTER_ANOMALY_PATTERNS
# trigger a WARNING so the pattern list can be updated as new templates are discovered
COUNTER_ANOMALY_HINT_KEYWORDS: list = ["THRESHOLD", "ERROR", "DROP", "EXCEED"]


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
    "template_id",
    "host",
    "timestamp",
    "frequency_score",
    "burstiness_score",
    "zscore_base",
    "time_delta_prev",
    "time_delta_session_start",
    "inter_arrival_rate",
    "event_weight",
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
# Phase 3 — IsolationForest hyperparameters (P3: Shreeraksha M)
# ---------------------------------------------------------------------------

# "auto" lets sklearn set contamination to 1/n_estimators; avoids overfitting
# the anomaly fraction assumption on small real-data batches.
IF_CONTAMINATION: str = "auto"

IF_N_ESTIMATORS: int = 100
IF_RANDOM_STATE: int = 42

# Feature columns fed into IsolationForest.
# Identifiers (sequence_number, session_id, host, template_id, timestamp) and
# rule-based signals (event_weight) are excluded — they are not learned features.
IF_FEATURE_COLUMNS: list = [
    "frequency_score",
    "burstiness_score",
    "zscore_base",
    "time_delta_prev",
    "time_delta_session_start",
    "inter_arrival_rate",
    "counter_proximity",
]

# Hybrid score weights (IF weighted higher — it captures multi-feature interactions
# that the per-column zscore_base signal misses).
IF_ISOLATION_WEIGHT: float = 0.7
IF_ZSCORE_WEIGHT: float = 0.3

# Model confidence scales linearly 0.0 → 1.0 as training samples grow.
# Below this threshold the blend leans on zscore_base; at or above it the full
# hybrid score is used.
COLD_START_FULL_CONFIDENCE_THRESHOLD: int = 500

# Combined score above this value → is_anomaly = True.
ANOMALY_SCORE_THRESHOLD: float = 0.5

# Sliding window: retrain on the last N sessions only.
RETRAINING_SESSION_WINDOW: int = 50

# Periodic trigger: retrain every K new log rows ingested.
RETRAINING_TRIGGER_EVERY_K: int = 1000

# Directory where versioned model pkl files and JSON sidecars are stored.
MODEL_STORE_PATH: str = "ml/model_store"

# ---------------------------------------------------------------------------
# Phase 4 — Importance Scoring (P4: Ujwal Hegde)
# ---------------------------------------------------------------------------

# Weights for the final importance score — 2-term formula.
# event_weight flows through the ML model indirectly via combined_score.
# Weights sum to 1.0.
SCORING_ML_WEIGHT: float = 0.65
SCORING_GRAPH_WEIGHT: float = 0.35

# Label thresholds: ignore / low / medium / critical
LABEL_IGNORE_MAX: float = 0.2
LABEL_LOW_MAX: float = 0.5
LABEL_MEDIUM_MAX: float = 0.75
# Anything above LABEL_MEDIUM_MAX → critical

# DBSCAN clustering parameters (incident_clusterer.py)
DBSCAN_EPS: float = 0.3
DBSCAN_MIN_SAMPLES: int = 5

# Root cause candidates selected per incident cluster.
ROOT_CAUSE_TOP_N: int = 3

# Missing upstream input fill strategy.
# Rows absent from anomaly_df or graph_scores_df after the left join are
# filled with the column mean of the non-null rows. Boolean columns
# (is_anomaly, in_graph, in_sequence) are always filled with False.
MISSING_INPUT_FILL: str = "mean"
