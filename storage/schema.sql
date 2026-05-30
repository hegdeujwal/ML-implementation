-- =========================
-- LOGS TABLE
-- =========================
CREATE TABLE IF NOT EXISTS logs (
    log_id TEXT PRIMARY KEY,
    sequence_number BIGINT,
    timestamp TIMESTAMPTZ,
    source_type VARCHAR(100),

    service VARCHAR(100),
    host VARCHAR(100),
    log_level VARCHAR(50),
    event_type VARCHAR(100),
    event_action VARCHAR(100),
    template_id VARCHAR(100),

    message TEXT,
    raw_text TEXT,

    -- IMPORTANT: now explicit columns (not only JSONB)
    incident_id TEXT,
    label TEXT,

    metadata JSONB,
    session_id TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =========================
-- FEATURES TABLE
-- =========================
CREATE TABLE IF NOT EXISTS features (
    log_id TEXT PRIMARY KEY REFERENCES logs(log_id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ,
    label TEXT,
    incident_id TEXT,

    frequency INT,
    event_weight DOUBLE PRECISION,
    frequency_score DOUBLE PRECISION,
    severity_weight DOUBLE PRECISION,
    counter_proximity DOUBLE PRECISION,

    feature_payload JSONB,
    in_sequence BOOLEAN,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =========================
-- ANOMALIES TABLE
-- =========================
CREATE TABLE IF NOT EXISTS anomalies (
    log_id TEXT PRIMARY KEY REFERENCES logs(log_id) ON DELETE CASCADE,
    incident_id TEXT,
    timestamp TIMESTAMPTZ,
    label TEXT,

    isolation_score DOUBLE PRECISION,
    zscore DOUBLE PRECISION,
    anomaly_score DOUBLE PRECISION,
    is_anomaly BOOLEAN,
    in_sequence BOOLEAN,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =========================
-- SCORES TABLE
-- =========================
CREATE TABLE IF NOT EXISTS scores (
    log_id TEXT PRIMARY KEY REFERENCES logs(log_id) ON DELETE CASCADE,

    importance_score DOUBLE PRECISION,
    final_score DOUBLE PRECISION,
    label TEXT,

    correlation_id VARCHAR(100),
    incident_id TEXT,

    is_root_cause BOOLEAN,
    root_cause_confidence DOUBLE PRECISION,
    in_sequence BOOLEAN,

    timestamp TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =========================
-- INCIDENTS TABLE
-- =========================
CREATE TABLE IF NOT EXISTS incidents (
    incident_id TEXT PRIMARY KEY,

    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,

    root_cause_log_id TEXT REFERENCES logs(log_id),

    severity TEXT,
    label TEXT,

    root_cause_confidence DOUBLE PRECISION,
    log_count INTEGER,

    status TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =========================
-- SUMMARIES TABLE
-- =========================
CREATE TABLE IF NOT EXISTS summaries (
    correlation_id TEXT PRIMARY KEY,
    summary_text TEXT NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_summaries_correlation_id
ON summaries (correlation_id);

-- =========================
-- INDEXES (LOGS)
-- =========================
CREATE INDEX IF NOT EXISTS idx_logs_log_id ON logs (log_id);
CREATE INDEX IF NOT EXISTS idx_logs_incident_id ON logs (incident_id);
CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs (timestamp);
CREATE INDEX IF NOT EXISTS idx_logs_label ON logs (label);

CREATE INDEX IF NOT EXISTS idx_logs_incident_time
ON logs (incident_id, timestamp);

-- =========================
-- INDEXES (FEATURES)
-- =========================
CREATE INDEX IF NOT EXISTS idx_features_log_id ON features (log_id);
CREATE INDEX IF NOT EXISTS idx_features_incident_id ON features (incident_id);
CREATE INDEX IF NOT EXISTS idx_features_timestamp ON features (timestamp);
CREATE INDEX IF NOT EXISTS idx_features_label ON features (label);

CREATE INDEX IF NOT EXISTS idx_features_incident_time
ON features (incident_id, timestamp);

-- =========================
-- INDEXES (ANOMALIES)
-- =========================
CREATE INDEX IF NOT EXISTS idx_anomalies_log_id ON anomalies (log_id);
CREATE INDEX IF NOT EXISTS idx_anomalies_incident_id ON anomalies (incident_id);
CREATE INDEX IF NOT EXISTS idx_anomalies_timestamp ON anomalies (timestamp);
CREATE INDEX IF NOT EXISTS idx_anomalies_label ON anomalies (label);

-- =========================
-- INDEXES (SCORES)
-- =========================
CREATE INDEX IF NOT EXISTS idx_scores_log_id ON scores (log_id);
CREATE INDEX IF NOT EXISTS idx_scores_incident_id ON scores (incident_id);
CREATE INDEX IF NOT EXISTS idx_scores_timestamp ON scores (timestamp);
CREATE INDEX IF NOT EXISTS idx_scores_label ON scores (label);

CREATE INDEX IF NOT EXISTS idx_scores_incident_time
ON scores (incident_id, timestamp);

-- =========================
-- INDEXES (INCIDENTS)
-- =========================
CREATE INDEX IF NOT EXISTS idx_incidents_incident_id ON incidents (incident_id);
CREATE INDEX IF NOT EXISTS idx_incidents_timestamp ON incidents (start_time);
CREATE INDEX IF NOT EXISTS idx_incidents_label ON incidents (label);