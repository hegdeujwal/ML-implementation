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

# Severity weights.
# HIGH and MEDIUM were added for the multi-section synthetic dataset, whose events
# carry an explicit severity=HIGH|MEDIUM field. They are ordered between the
# pre-existing levels so the CRITICAL > HIGH > ERROR > MEDIUM > WARN > INFO ranking
# is preserved instead of collapsing unknown levels to DEFAULT_SEVERITY_WEIGHT.
SEVERITY_WEIGHTS = {
    "CRITICAL": 1.0,
    "HIGH": 0.85,
    "ERROR": 0.7,
    "MEDIUM": 0.55,
    "WARN": 0.4,
    "INFO": 0.1,
}

DEFAULT_SEVERITY_WEIGHT: float = 0.1


# Counter anomaly proximity
COUNTER_PROXIMITY_WINDOW_SECONDS: int = 30

# Time window (seconds) for joining Section-4 numeric metrics onto event rows
# (features/metric_features.py). A metric sample within ±this of an event is
# considered "near" it. Scoped per scenario so incidents never cross-contaminate.
# NOTE: metric samples are ~480s (8 min) apart, so a 60s window only matches
# events that happen to fall within 60s of a sample (~25% coverage). Widen toward
# ~240s (half the spacing) to lift coverage toward 100%.
METRIC_JOIN_WINDOW_SECONDS: int = 240

# Trailing-window sizes (in metric SAMPLES, not seconds) for the rolling-slope
# trend features. Samples are ~8 min apart, so short=4 ≈ 32 min, long=12 ≈ 96 min.
# Short reacts fast to drift onset but is noisier; long is smoother but laggier.
# Both are computed as backward-looking OLS slope normalised by the series' own
# std, so they share units and are comparable to each other.
METRIC_SLOPE_SHORT_WINDOW: int = 4
METRIC_SLOPE_LONG_WINDOW: int = 12

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

    # --- Generic vendor-neutral component names (mentor's synthetic dataset) ---
    "spanning_tree_daemon":  "STP",
    "redundancy_daemon":     "REDUNDANCY",
    "forwarding_engine":     "FORWARDING",
    "access_control_daemon": "ACL",
    "routing_daemon":        "ROUTING",
    "mac_learning":          "MAC",
    "qos_scheduler_daemon":  "QOS",
    "buffer_manager":        "BUFFER",
    "physical_monitor":      "PHYSICAL",
    "statistics_collector":  "STATS",
    "system_logger":         "SYSTEM",
    "process_monitor":       "SYSTEM",
    "network_monitor":       "NETWORK",

    # --- Routine/heartbeat services: the ~90% baseline noise. Collapsed to a
    #     single NOISE label so the feature stage can suppress/down-weight them. ---
    "monitoring":             "NOISE",
    "continuous_monitoring":  "NOISE",
    "routine_check":          "NOISE",
    "periodic_status":        "NOISE",
    "system_check":           "NOISE",
    "status_verification":    "NOISE",
    "health_check":           "NOISE",
    "metrics_update":         "NOISE",
    "frame_monitoring":       "NOISE",
}

# Canonical service label assigned to routine/heartbeat logs (see SERVICE_ALIAS_MAP).
# Feature/noise-suppression logic can key off this single value.
NOISE_SERVICE_LABEL: str = "NOISE"


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
    r"DROP",
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

# --- Multi-section synthetic dataset artifacts (parsing/synthetic_dataset_loader.py) ---
# Long/tidy numeric metrics extracted from Section 4 of each scenario file.
# Long format means "metric not applicable to a scenario" is simply an absent row,
# avoiding a wide sparse table full of structural NaNs.
METRICS_DF_PATH: str = "data/processed/metrics_df.parquet"

# Per-file ground-truth record from Section 7 (training_label, correlation_signals, …).
# Used ONLY by the evaluation harness as an oracle — never fed to the model.
SCENARIO_LABELS_PATH: str = "data/processed/scenario_labels.parquet"

# ---------------------------------------------------------------------------
# Oracle evaluation (evaluation/oracle_report.py)
# ---------------------------------------------------------------------------

# Log levels that define ground-truth "signal" rows for evaluation.
# Severity is deliberately excluded from IF_FEATURE_COLUMNS, so judging the
# anomaly stage against severity-derived truth is fair (not circular).
# final_score DOES carry a severity term (SCORING_SEVERITY_WEIGHT) — ranking
# metrics computed against this truth are partially favoured by construction.
ORACLE_TRUTH_SEVERITIES: list = ["CRITICAL", "HIGH", "ERROR"]

# Where the oracle evaluation report text file is written.
ORACLE_REPORT_PATH: str = "evaluation/results/oracle_report.txt"

# ---------------------------------------------------------------------------
# Persistent drift detection stores
# ---------------------------------------------------------------------------

# Welford online z-score baseline: one row per (host, template_id).
# Accumulates mean/variance across pipeline runs for cross-run drift detection.
ZSCORE_BASELINE_STORE_PATH: str = "data/processed/zscore_baseline_store.parquet"

# Rolling feature store for IsolationForest sliding-window retraining.
# Holds raw feature rows from the last FEATURE_ROLLING_MAX_SESSIONS sessions.
FEATURE_ROLLING_STORE_PATH: str = "data/processed/feature_rolling_store.parquet"

# Maximum unique sessions to retain in the rolling feature store.
# Matches RETRAINING_SESSION_WINDOW so the two stores stay in sync.
FEATURE_ROLLING_MAX_SESSIONS: int = 50


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
    # Section-4 numeric-metric features (features/metric_features.py).
    "metric_zscore",
    "metric_zscore_present",
    "drop_rate",
    "drop_rate_present",
    "utilization",
    "utilization_present",
    # Rolling-slope trend features (two windows) for gradual-drift detection.
    "metric_slope_short",
    "metric_slope_short_present",
    "metric_slope_long",
    "metric_slope_long_present",
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
    # Section-4 numeric telemetry — observed measurements (not label proxies),
    # so safe to learn from. Paired *_present flags let the model tell a real 0
    # from a neutrally-filled absent value.
    "metric_zscore",
    "metric_zscore_present",
    "drop_rate",
    "drop_rate_present",
    "utilization",
    "utilization_present",
    # Trend features: backward-looking OLS slope over two windows. These give the
    # model the rate-of-change signal that point-in-time metrics lack — the
    # signature of gradual drift (memory leaks, thermal/CPU creep, disk I/O decay).
    "metric_slope_short",
    "metric_slope_short_present",
    "metric_slope_long",
    "metric_slope_long_present",
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
# Used as fallback when score std < 1e-6 (all-identical edge case).
ANOMALY_SCORE_THRESHOLD: float = 0.5

# Multiplier for the dynamic threshold: threshold = mean(scores) + k × std(scores).
# k=2.0 flags scores more than 2 standard deviations above the batch mean (~top 2.3%).
ANOMALY_DYNAMIC_K: float = 2.0

# Anomaly-flag strategy:
#   "quantile"  — flag the top ANOMALY_CONTAMINATION fraction by combined_score.
#                 Self-adjusts to each batch and guarantees a stable, non-zero
#                 anomaly rate.
#   "dynamic_k" — legacy mean + k·std rule (kept for back-compat). Fragile: when
#                 combined_score is tightly clustered the threshold can exceed the
#                 max achievable score and flag nothing (observed: 0/935).
ANOMALY_FLAG_MODE: str = "quantile"

# Expected fraction of the batch that is anomalous (top-N flagged in quantile mode).
# Set to the measured signal rate of the synthetic dataset (~13% non-baseline logs).
ANOMALY_CONTAMINATION: float = 0.13

# Sliding window: retrain on the last N sessions only.
RETRAINING_SESSION_WINDOW: int = 50

# Periodic trigger: retrain every K new log rows ingested.
RETRAINING_TRIGGER_EVERY_K: int = 1000

# Directory where versioned model pkl files and JSON sidecars are stored.
MODEL_STORE_PATH: str = "ml/model_store"

# ---------------------------------------------------------------------------
# Phase 4 — Importance Scoring (P4: Ujwal Hegde)
# ---------------------------------------------------------------------------

# Weights for the final importance score — 3-term formula. Weights sum to 1.0.
#   final_score = ML_WEIGHT·combined_score        (behavioral anomaly, unsupervised IF)
#               + GRAPH_WEIGHT·centrality_score    (structural importance)
#               + SEVERITY_WEIGHT·event_weight     (declared severity)
# Severity is kept as its own explicit, tunable term here rather than baked into the
# IsolationForest features. This avoids (a) leaking the severity label into an
# unsupervised model that is then validated against severity-derived ground truth,
# and (b) double-counting severity once the model and the score both carry it.
SCORING_ML_WEIGHT: float = 0.5
SCORING_GRAPH_WEIGHT: float = 0.25
SCORING_SEVERITY_WEIGHT: float = 0.25

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

# ---------------------------------------------------------------------------
# Phase 5.5 — Cross-Run Incident Correlation
# ---------------------------------------------------------------------------

# Master switch: set False to skip P5.5 entirely (no history written or read).
CROSS_RUN_ENABLED: bool = True

# How far back (hours) to search the incident history for potential precursors.
# 72h = 3 days covers weekend-to-Monday drift and slow-burn memory leaks.
CROSS_RUN_LOOKBACK_HOURS: int = 72

# Jaccard similarity threshold for declaring two incidents "related".
# 0.3 = at least 30% of the combined template vocabulary must be shared.
# Intentionally low: precursors typically share a subset, not all, templates.
CROSS_RUN_SIMILARITY_THRESHOLD: float = 0.3

# Minimum Jaccard similarity floor. Even if overlap_coefficient is high,
# the link is rejected if Jaccard similarity is below this floor.
# Prevents a 1-template incident from linking to a 100-template incident.
CROSS_RUN_MIN_JACCARD: float = 0.05

# Score boost applied to precursor logs when a descendant critical incident
# is discovered. Capped to [0, 1] after application.
# elevated_score = min(1.0, original_score + PRECURSOR_BOOST * chain_confidence)
PRECURSOR_BOOST: float = 0.15

# Prefix for generated chain IDs (format: CHAIN-<unix_ts>-<seq>).
CHAIN_ID_PREFIX: str = "CHAIN"

# Parquet-based fallback store for incident history.
# Used in dry-run mode (no Postgres) and synced to the DB on live runs.
INCIDENT_HISTORY_PATH: str = "data/processed/incident_history.parquet"
