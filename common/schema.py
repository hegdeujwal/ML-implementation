"""
Canonical dataclass schemas for all inter-module DataFrames.

These models serve as the authoritative schema reference for the whole team.
Modules may use them for validation, documentation, or type hints.

Each class mirrors the parquet schema produced by its stage:
    LogEntry      -- parsing/sessionizer.py   -> data/processed/sessionized_logs.parquet
    FeaturesRow   -- features/feature_pipeline -> data/processed/features_df.parquet
    AnomalyRow    -- ml/anomaly_detector.py   -> data/processed/anomaly_df.parquet
    GraphScoreRow -- correlation/run_correlation -> data/processed/graph_scores_df.parquet
    ScoredLogRow  -- scoring/importance_scorer  -> data/processed/scored_logs_df.parquet
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LogEntry:
    """One row of sessionized_logs.parquet (parsing output, P0 handoff).

    Canonical schema — all downstream modules read from this.

    Attributes
    ----------
    log_id      : Unique log identifier, e.g. "log_000001".
    raw_text    : Original unparsed log line.
    timestamp   : Unix epoch seconds (float).
    source      : Hostname / device identifier.
    session_id  : Groups related events, e.g. "session_001".
    template_id : Normalized Drain template, e.g. "IF_DOWN".
    severity    : Log level — one of CRITICAL / ERROR / WARN / INFO.
    is_anomaly  : True when flagged by the anomaly-detection stage.
    anomaly_label : Non-empty only when is_anomaly=True.
    """
    log_id: str
    raw_text: str
    timestamp: float
    source: str
    session_id: str
    template_id: str
    severity: str
    is_anomaly: bool = False
    anomaly_label: str = ""

    # Alias kept for backwards compatibility with features/severity_features.py
    @property
    def log_level(self) -> str:
        return self.severity


@dataclass
class FeaturesRow:
    """One row of features_df.parquet (P1 output).

    Attributes
    ----------
    log_id                  : Foreign key into LogEntry.
    session_id              : Session grouping key.
    frequency_score         : Normalised template count within its session [0, 1].
    burstiness_score        : Fano factor of inter-arrival times in session.
    zscore_base             : Z-score of template count vs 1-hour rolling baseline.
    time_delta_prev         : Seconds since the previous log in this session.
    time_delta_session_start: Seconds from session start to this log.
    inter_arrival_rate      : EMA of per-session arrival rates.
    severity_weight         : Numeric encoding of severity (CRITICAL=1.0 .. INFO=0.1).
    counter_proximity       : Proximity score to known counter-anomaly events.
    """
    log_id: str
    session_id: str
    frequency_score: float
    burstiness_score: float
    zscore_base: float
    time_delta_prev: float
    time_delta_session_start: float
    inter_arrival_rate: float
    severity_weight: float
    counter_proximity: float


@dataclass
class AnomalyRow:
    """One row of anomaly_df.parquet (P2 output).

    Attributes
    ----------
    log_id          : Foreign key into LogEntry.
    isolation_score : IsolationForest score normalised to [0, 1]. 0.0 during cold-start.
    zscore          : Raw (unnormalised) per-row z-score.
    combined_score  : Hybrid score = w1*isolation + w2*zscore_norm.
    is_anomaly      : True when combined_score > ANOMALY_THRESHOLD.
    """
    log_id: str
    isolation_score: float
    zscore: float
    combined_score: float
    is_anomaly: bool


@dataclass
class GraphScoreRow:
    """One row of graph_scores_df.parquet (P3 output).

    Attributes
    ----------
    log_id             : Foreign key into LogEntry.
    centrality_score   : PageRank score [0, 1] — primary signal for P4.
    degree             : Raw edge count in the correlation graph.
    betweenness        : Normalised betweenness centrality [0, 1].
    cluster_id         : Connected-component label (cc_0, cc_1, ...).
    in_sequence        : True if this log is part of a detected recurring sequence.
    correlated_log_ids : Other log_ids in the same session whose template is a
                         graph neighbour of this log's template.
    """
    log_id: str
    centrality_score: float
    degree: int
    betweenness: float
    cluster_id: str
    in_sequence: bool
    correlated_log_ids: List[str] = field(default_factory=list)


@dataclass
class ScoredLogRow:
    """One row of scored_logs_df.parquet (P4 output).

    Attributes
    ----------
    log_id                : Foreign key into LogEntry.
    final_score           : Weighted sum of ML + graph + rule signals [0, 1].
    label                 : Human-readable tier: ignore / low / medium / critical.
    incident_id           : DBSCAN cluster label (e.g. "INC-001"). None = noise.
    is_root_cause         : True for the top-ranked log within an incident.
    root_cause_confidence : Confidence in the root-cause assignment [0, 1].
    """
    log_id: str
    final_score: float
    label: str
    incident_id: Optional[str]
    is_root_cause: bool
    root_cause_confidence: float
