# Pipeline Bug Fix Report

## Results

- **Before fixes**: `in_graph=0.0%`, all logs UNCAPPED, only `low`/`medium` labels, stale graph cache silently used
- **After fixes**: `in_graph=100%`, centrality scores spanning `[0.0, 1.0]`, `critical`/`medium`/`low` labels, graph rebuilt fresh each run
- **Test suite**: 137/137 passing

---

## Bug 1 — Graph pickle path inconsistency (Critical)

**Files**: `correlation/graph_builder.py`, `common/config.py`

**Problem**: `graph_builder.py` hardcoded `_GRAPH_PICKLE_PATH = "data/processed/correlation_graph.pkl"` while `common/config.py` defines `GRAPH_PICKLE_PATH = "data/processed/correlation_graph.gpickle"`. The cache existence check in `run_correlation.py` used the config path, so it never found the builder's `.pkl` file and always loaded stale old graphs.

**Fix**: `graph_builder.py` now uses `cfg.GRAPH_PICKLE_PATH` so all modules share one canonical path.

---

## Bug 2 — Stale graph cache poisoning centrality scores (Critical)

**File**: `pipeline.py`

**Problem**: `_step_correlation()` was loading a cached graph built from an entirely different set of template IDs (old synthetic data). Since none of the current session's templates matched the cached graph nodes, `in_graph` was `False` for 100% of rows, `cluster_id` was `UNCAPPED` everywhere, and `centrality_score` was a constant `0.5116` for all rows. This completely broke the scoring stage.

**Fix**: `_step_correlation()` now deletes the cache file before calling `run_correlation.run()`, forcing a fresh graph rebuild on every pipeline execution. The rebuild takes < 0.2s for typical data sizes.

---

## Bug 3 — DB upsert key mismatch (Critical for storage)

**File**: `storage/db_writer.py`

**Problem**: `_upsert_key_for_table()` returned `"log_id"` for all non-incident tables. But the actual pipeline DataFrames (sessionized_logs, features_df, anomaly_df, scored_logs_df) never contain a `log_id` column — the universal key is `sequence_number`. Every write to Postgres would fail with a missing key error.

**Fix**: Replaced with `_TABLE_KEY_MAP` dict that correctly maps each table to its actual conflict key (`sequence_number` for all pipeline tables, `incident_id` for incidents).

---

## Bug 4 — Schema SQL used wrong primary key type (Critical for storage)

**File**: `storage/schema.sql`

**Problem**: All tables used `log_id TEXT PRIMARY KEY` with foreign key references. The actual pipeline never produces a `log_id` column. The schema was designed around a different (older) data model and was completely disconnected from the pipeline's DataFrame contracts.

**Fix**: Rewrote schema to use `sequence_number BIGINT PRIMARY KEY` throughout, matching the canonical schema in AGENTS.md. All table columns now exactly match what each pipeline stage produces.

---

## Bug 5 — Emojis in error output (Rule violation)

**File**: `common/env_handler.py`

**Problem**: Error messages contained emoji characters (`[ENV ERROR]` with emoji prefix), violating the project rule "Do not use emojis in the codebase or documentation."

**Fix**: Removed all emoji characters from error messages. Also fixed the misplaced `import sys` (was mid-file, now at top).

---

## Bug 6 — Stale manual_test.py importing removed API (Import error)

**File**: `correlation/manual_test.py`

**Problem**: The file imported `LogEvent` from `correlation.graph_builder`, a class that no longer exists (the API was redesigned to use DataFrames). This caused pytest collection to fail with an `ImportError`, preventing any tests from running.

**Fix**: Rewrote the script to use the current DataFrame-based API (`build_graph(df)`, `compute_centrality(g, df)`).

---

## Bug 7 — `print()` calls throughout pipeline (Convention violation)

**Files**: `features/feature_pipeline.py`, `ml/anomaly_detector.py`, `correlation/centrality.py`, `correlation/run_correlation.py`, `scoring/root_cause_engine.py`

**Problem**: Multiple modules used bare `print()` for warnings and informational output, violating the project rule "Logging via `get_logger(__name__)` — no bare `print()` for warnings/errors."

**Fix**: Replaced all `print()` calls with `logger.info()` / `logger.warning()` using the module's existing logger instance.

> [!NOTE]
> The `print()` calls in `generate_real_logs.py` and `sessionizer.py` `__main__` blocks are intentional CLI output and were left unchanged.
