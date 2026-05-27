"""
Canonical dataclass schemas for all inter-module DataFrames.

These models serve as the authoritative schema reference for the whole team.
Modules may use them for validation, documentation, or type hints.

Each class mirrors the parquet schema produced by its stage:
    LogEntry      -- parsing/sessionizer.py      -> data/processed/sessionized_logs.parquet
    FeaturesRow   -- features/feature_pipeline   -> data/processed/features_df.parquet
    AnomalyRow    -- ml/anomaly_detector.py      -> data/processed/anomaly_df.parquet
    GraphScoreRow -- correlation/run_correlation -> data/processed/graph_scores_df.parquet
    ScoredLogRow  -- scoring/importance_scorer   -> data/processed/scored_logs_df.parquet
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class LogEntry:
    """One row of sessionized_logs.parquet (P1 output).

    sequence_number is the universal join key across all downstream DataFrames.
    session_id is kept here for feature engineering groupby operations but is
    not part of the canonical Postgres/ES schema.
    """
    sequence_number: int
    timestamp: datetime
    source_type: str        # 'switch' for all HPE CX logs
    service: str            # normalised subsystem: OSPF, BGP, SYSTEM, ...
    host: str               # device hostname
    log_level: str          # CRITICAL | ERROR | WARN | INFO
    event_type: str         # subsystem label (= service)
    event_action: str       # specific action (template_id minus service prefix)
    template_id: str        # Drain template slug
    frequency: int          # count of this template_id in the same session
    event_weight: float     # CRITICAL=1.0, ERROR=0.7, WARN=0.4, INFO=0.1
    message: str            # log message content
    metadata: str           # JSON string: {"raw_text": "<original line>"}
    session_id: str         # groups related events; not in canonical DB schema


@dataclass
class FeaturesRow:
    """One row of features_df.parquet (P2 output).

    sequence_number, session_id, template_id, host, timestamp, and event_weight
    are carried from LogEntry (no recomputation).  frequency_score normalises the
    raw frequency column from parsing.  All other fields are computed here.
    """
    sequence_number: int
    session_id: str
    template_id: str
    host: str
    timestamp: datetime
    frequency_score: float
    burstiness_score: float
    zscore_base: float
    time_delta_prev: float
    time_delta_session_start: float
    inter_arrival_rate: float
    event_weight: float
    counter_proximity: float


@dataclass
class AnomalyRow:
    """One row of anomaly_df.parquet (P3 output).

    combined_score = confidence * (IF_ISOLATION_WEIGHT * isolation_score
                                   + IF_ZSCORE_WEIGHT * zscore_norm)
                   + (1 - confidence) * zscore_norm
    is_anomaly = combined_score > ANOMALY_SCORE_THRESHOLD.
    """
    sequence_number: int
    isolation_score: float   # IsolationForest score normalised to [0, 1]; 0.5 when all scores identical
    zscore_norm: float       # zscore_base from P2 clipped to [-5,5] then scaled to [0, 1]
    combined_score: float    # confidence-weighted hybrid score [0, 1]
    is_anomaly: bool
    model_confidence: float  # linear ramp 0.0 → 1.0 as n_training_samples grows


@dataclass
class GraphScoreRow:
    """One row of graph_scores_df.parquet (P4 output).

    centrality_score (PageRank) is the primary signal consumed by scoring.
    correlation_id groups logs into incident clusters; NULL for unclustered logs.
    """
    sequence_number: int
    centrality_score: float  # PageRank [0, 1] — primary signal for scoring
    degree: int              # edge count in co-occurrence graph
    betweenness: float       # normalised betweenness centrality [0, 1]
    correlation_id: Optional[str]  # incident cluster label; NULL if unclustered
    in_sequence: bool        # True if part of a detected recurring sequence


@dataclass
class ScoredLogRow:
    """One row of scored_logs_df.parquet (P5 output).

    importance_score = SCORING_WEIGHT_ML * combined_score
                     + SCORING_WEIGHT_GRAPH * centrality_score
                     + SCORING_WEIGHT_RULE * event_weight
    Clipped to [0.0, 1.0] before saving.
    """
    sequence_number: int
    importance_score: float  # [0.0, 1.0]
    label: str               # ignore | low | medium | critical
    correlation_id: Optional[str]   # propagated from GraphScoreRow
    is_root_cause: bool
