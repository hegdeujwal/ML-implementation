# Persistent Drift Detection — Implementation Plan

## Goal

Make the pipeline detect **gradual degradation across multiple runs** by:
1. Persisting a z-score baseline across pipeline invocations (Welford's algorithm)
2. Accumulating a rolling feature store for sliding-window IF retraining
3. Wiring `AnomalyTrainer` into `anomaly_detector.run()` (currently disconnected dead code)
4. Validating saved models against sklearn version + feature column drift before loading

---

## Architecture

```
Run N-1                              Run N
─────────────────────────────        ──────────────────────────────────────
features_df.parquet                  features_df.parquet (new batch)
        │                                    │
        ▼                                    ▼
zscore_baseline_store.parquet ──────► zscore_base_persistent()
(n, mean, M2 per host+template)       computes z-scores using accumulated history
        │                             updates + re-saves store
        │                                    │
feature_rolling_store.parquet ──────► AnomalyTrainer.retrain()
(last 50 sessions of feature rows)    trains IF on rolling window
        │                             saves model + sidecar
        │                                    │
isolation_forest_vXXXX.pkl    ──────► anomaly_detector.detect()
(validated via sidecar)               scores current batch
                                             │
                                      anomaly_df.parquet
```

---

## Files to Change

### 1. `common/config.py` — Add 3 new constants

```python
# Persistent drift detection
ZSCORE_BASELINE_STORE_PATH: str = "data/processed/zscore_baseline_store.parquet"
FEATURE_ROLLING_STORE_PATH: str = "data/processed/feature_rolling_store.parquet"
# Cap raw feature rows to last N sessions per host for IF training
FEATURE_ROLLING_MAX_SESSIONS: int = 50  # matches RETRAINING_SESSION_WINDOW
```

### 2. `features/statistical_features.py` — Add `zscore_base_persistent()`

New function alongside existing `zscore_base()`. Uses Welford's algorithm:

**Welford update per new session:**
```
n    += 1
delta = freq - mean
mean += delta / n
M2   += delta * (freq - mean)   # uses updated mean
std   = sqrt(M2 / (n - 1))      # sample std, defined when n >= 2
```

**What gets persisted:** One row per `(host, template_id)`:
```
host | template_id | n | welford_mean | welford_M2 | seen_session_ids (JSON list)
```

**Logic:**
1. Load `zscore_baseline_store.parquet` if it exists.
2. For each new `(host, session_id, template_id, frequency)` in current batch:
   - Skip if `session_id` already in `seen_session_ids` (idempotent re-runs).
   - Apply Welford update to that group's `(n, mean, M2)`.
   - Add `session_id` to `seen_session_ids`.
3. Compute z-score for each row in current batch using the **post-update** stats.
4. Save updated store back to disk.
5. Return z-scores aligned to `df.index`.

> [!NOTE]
> Welford updates happen before z-score computation so the current session is
> included in its own baseline. This is intentional: it softens false positives
> on the very first session containing a pattern.

### 3. `features/feature_pipeline.py` — Use persistent zscore

Replace:
```python
df["zscore_base"] = zscore_base(df, ZSCORE_BASELINE_N_SESSIONS)
```
With:
```python
df["zscore_base"] = zscore_base_persistent(df, ZSCORE_BASELINE_STORE_PATH)
```

Also add a step to **append current batch features to the rolling store**:
```python
_update_feature_rolling_store(out, FEATURE_ROLLING_STORE_PATH, FEATURE_ROLLING_MAX_SESSIONS)
```

### 4. `features/statistical_features.py` — Add `_update_feature_rolling_store()`

Helper called by feature_pipeline.py:
1. Load existing rolling store if it exists.
2. Append current batch rows.
3. Keep only rows from the last `FEATURE_ROLLING_MAX_SESSIONS` unique sessions (by session_start).
4. Save back.

### 5. `ml/trainer.py` — Add sidecar validation

In `_save_model()`, add sklearn + numpy version to the JSON sidecar:
```python
import sklearn, numpy as np
metadata["sklearn_version"] = sklearn.__version__
metadata["numpy_version"] = numpy.__version__
metadata["feature_columns"] = IF_FEATURE_COLUMNS
```

Add `_validate_sidecar(sidecar_path)` that warns if versions mismatch and raises
`ValueError` if `feature_columns` in the sidecar does not match `IF_FEATURE_COLUMNS`
from config (silent score corruption otherwise).

### 6. `ml/anomaly_detector.py` — Wire `AnomalyTrainer` into `run()`

Replace the current `run()` that always retrains fresh:

```python
def run(features_path, output_path):
    df = load_parquet(str(features_path))

    trainer = AnomalyTrainer()

    # Load rolling feature store for IF training (cross-run history)
    rolling_store_path = Path(cfg.FEATURE_ROLLING_STORE_PATH)
    if rolling_store_path.exists():
        train_df = load_parquet(str(rolling_store_path))
        logger.info("Loaded rolling feature store: %d rows for IF training", len(train_df))
    else:
        train_df = df  # cold start: train on current batch only
        logger.info("No rolling store found — cold start, training on current batch.")

    # Load latest saved model (validated against sidecar)
    latest_model = trainer.load_latest_model()
    model_path = str(latest_model) if latest_model else None

    # Detect anomalies (uses saved model if valid, otherwise trains fresh)
    anomaly_df = detect(df, model_path=model_path)

    # Trigger retraining if K-log threshold crossed
    trainer.maybe_retrain(train_df)

    save_parquet(anomaly_df, str(output_path))
    return anomaly_df
```

---

## What Changes and Why

| Change | Reason |
|---|---|
| Welford baseline store | z-scores are now computed against ALL historical sessions, not just the current batch |
| Rolling feature store | IF is trained on last 50 sessions across runs, not just this run's data |
| AnomalyTrainer wired in | model lifecycle (load → score → retrain if triggered → save) is finally active |
| Sidecar validation | prevents silent score corruption from sklearn version drift or column reordering |

## What Does NOT Change

- The output contract of `anomaly_df.parquet` is identical — same 6 columns
- `feature_pipeline.py` output contract unchanged — same `FEATURE_COLUMNS`
- All existing tests pass without modification

## Rollback / Safety

- Both stores (`zscore_baseline_store.parquet`, `feature_rolling_store.parquet`) are
  append-only with deduplication — safe to delete and start fresh
- The pipeline degrades gracefully to the current behavior if either store is missing
  (cold-start path)
- Model loading is validated before use; if validation fails, falls back to fresh training
