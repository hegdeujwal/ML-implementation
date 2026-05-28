"""
correlation/run_correlation.py

Phase 3 pipeline entry point.

Wires all correlation components together in the correct order:
  1. Generate / verify synthetic log data
  2. Build or load cached CorrelationGraph from sessionized_logs.parquet
  3. Compute per-node centrality scores
  4. Detect recurring log sequences
  5. Assemble graph_scores_df.parquet (P4 handoff)
  6. Export correlation_graph.json (visualization)

Usage
-----
From the project root:
    python -m correlation.run_correlation

Optional flags (edit constants below or pass via environment):
    REBUILD_GRAPH=1  -- force rebuild even if pickle cache exists
"""

from __future__ import annotations

import json
import os
import sys
import time

import pandas as pd

import common.config as cfg
from correlation.centrality import compute_centrality
from correlation.graph_builder import build_graph, load_or_build_graph
from correlation.graph_visualizer import export_graph_json
from correlation.sequence_engine import detect_sequences

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SESSIONIZED_LOGS_PATH = cfg.SESSIONIZED_LOGS_PATH
GRAPH_PICKLE_PATH = cfg.GRAPH_PICKLE_PATH
GRAPH_JSON_PATH = cfg.GRAPH_JSON_PATH
SEQUENCES_JSON_PATH = cfg.SEQUENCES_JSON_PATH
GRAPH_SCORES_PATH = cfg.GRAPH_SCORES_PATH

REBUILD_GRAPH = os.environ.get("REBUILD_GRAPH", "0") == "1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_data_dir() -> None:
    os.makedirs("data/processed", exist_ok=True)


def _ensure_logs_exist() -> None:
    """Generate synthetic data if the parquet file does not exist."""
    if not os.path.exists(SESSIONIZED_LOGS_PATH):
        print(f"  sessionized_logs.parquet not found at {SESSIONIZED_LOGS_PATH}")
        print("  Generating synthetic data via scripts/generate_real_logs.py ...")
        # Import and run inline to avoid subprocess dependency
        sys.path.insert(0, ".")
        from scripts.generate_real_logs import generate_dataset  # type: ignore
        os.makedirs(os.path.dirname(SESSIONIZED_LOGS_PATH), exist_ok=True)
        df = generate_dataset()
        df.to_parquet(SESSIONIZED_LOGS_PATH, index=False)
        print(f"  Generated {len(df):,} rows -> {SESSIONIZED_LOGS_PATH}")


def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run() -> None:
    total_start = time.perf_counter()

    _section("Phase 3 — Graph Correlation Pipeline")
    _ensure_data_dir()
    _ensure_logs_exist()

    # ------------------------------------------------------------------
    # Step 1: Build or load CorrelationGraph
    # ------------------------------------------------------------------
    _section("Step 1/5 — Build correlation graph")
    t0 = time.perf_counter()

    raw_df = pd.read_parquet(SESSIONIZED_LOGS_PATH)

    if REBUILD_GRAPH and os.path.exists(GRAPH_PICKLE_PATH):
        os.remove(GRAPH_PICKLE_PATH)
        print("  Forced rebuild: removed cached graph.")

    g = load_or_build_graph(raw_df)
    print(f"  Graph: {len(g.nodes)} nodes, {len(g.edges)} edges")
    print(f"  Elapsed: {time.perf_counter() - t0:.2f}s")

    # ------------------------------------------------------------------
    # Step 2: Compute centrality scores
    # ------------------------------------------------------------------
    _section("Step 2/5 — Compute centrality scores")
    t0 = time.perf_counter()

    graph_scores_df = compute_centrality(g, raw_df)
    print(f"  Computed centrality for {len(graph_scores_df)} log rows")
    print(f"  centrality_score range: "
          f"[{graph_scores_df['centrality_score'].min():.4f}, "
          f"{graph_scores_df['centrality_score'].max():.4f}]")
    print(f"  Elapsed: {time.perf_counter() - t0:.2f}s")

    # ------------------------------------------------------------------
    # Step 3: Detect sequences
    # ------------------------------------------------------------------
    _section("Step 3/5 — Detect recurring sequences")
    t0 = time.perf_counter()

    # sequence_engine expects float epoch seconds; normalise datetime if needed
    seq_df = raw_df
    if pd.api.types.is_datetime64_any_dtype(seq_df["timestamp"]):
        seq_df = seq_df.copy()
        seq_df["timestamp"] = seq_df["timestamp"].astype("int64") / 1e9
    in_sequence_log_ids = detect_sequences(seq_df, output_path=SEQUENCES_JSON_PATH)

    with open(SEQUENCES_JSON_PATH, "r") as fh:
        sequences = json.load(fh)

    print(f"  Detected {len(sequences)} recurring sequences")
    print(f"  Logs in sequence: {len(in_sequence_log_ids):,}")
    print(f"  Written to {SEQUENCES_JSON_PATH}")
    print(f"  Elapsed: {time.perf_counter() - t0:.2f}s")

    # ------------------------------------------------------------------
    # Step 4: Patch in_sequence and persist graph_scores_df
    # ------------------------------------------------------------------
    _section("Step 4/5 — Patch in_sequence and save graph_scores_df.parquet")
    t0 = time.perf_counter()

    graph_scores_df["in_sequence"] = graph_scores_df["sequence_number"].isin(
        in_sequence_log_ids
    )
    from common.utils import save_parquet
    save_parquet(graph_scores_df, GRAPH_SCORES_PATH)

    print(f"  Rows: {len(graph_scores_df):,}")
    print(f"  in_sequence=True: {graph_scores_df['in_sequence'].sum():,}")
    print(f"  Columns: {list(graph_scores_df.columns)}")
    print(f"  Written to {GRAPH_SCORES_PATH}")
    print(f"  Elapsed: {time.perf_counter() - t0:.2f}s")

    # ------------------------------------------------------------------
    # Step 5: Export JSON visualization
    # ------------------------------------------------------------------
    _section("Step 5/5 — Export graph JSON")
    t0 = time.perf_counter()

    export_graph_json(g, output_path=GRAPH_JSON_PATH)
    print(f"  Graph: {len(g.nodes)} nodes, {len(g.edges)} edges")
    print(f"  Written to {GRAPH_JSON_PATH}")
    print(f"  Elapsed: {time.perf_counter() - t0:.2f}s")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _section("Done")
    print(f"  Total elapsed: {time.perf_counter() - total_start:.2f}s")
    print()
    print("  Output files:")
    for path in [GRAPH_PICKLE_PATH, SEQUENCES_JSON_PATH, GRAPH_SCORES_PATH, GRAPH_JSON_PATH]:
        size_kb = os.path.getsize(path) / 1024 if os.path.exists(path) else 0
        print(f"    {path}  ({size_kb:.1f} KB)")
    print()
    print("  P4 handoff: data/processed/graph_scores_df.parquet is ready.")


if __name__ == "__main__":
    run()
