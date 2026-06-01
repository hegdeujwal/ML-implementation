# Network Log Analysis ML Pipeline

This project is a comprehensive Machine Learning pipeline for network log analysis. It covers the full lifecycle from data ingestion to anomaly detection, correlation, log importance scoring, data storage, and visualization.

## Quickstart

```bash
# 1. Start infrastructure (Postgres, Elasticsearch, Grafana, Kibana)
docker compose up -d

# 2. Set up environment
cp .env.example .env          # fill in credentials
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Run the full pipeline (dry-run — skips Postgres write)
python pipeline.py --dry-run

# 4. Run with Postgres write (requires docker compose up)
python pipeline.py

# 5. Restart from a specific step (reads existing parquets for earlier steps)
python pipeline.py --dry-run --from-step scoring
```

If you have a real syslog file, pass it with `--log-file`:
```bash
python pipeline.py --dry-run --log-file data/raw/cx_switches.log
```

Without a log file, the pipeline generates 5 000 rows of synthetic CX switch data automatically.

## Module Overview

| Folder | Owner | Output file | Description |
|--------|-------|-------------|-------------|
| `ingestion/` | Sumukha Rao H | `data/raw/*.log` | Raw log ingestion (batch or Fluent Bit stream) |
| `parsing/` | Sumukha Rao H | `data/processed/sessionized_logs.parquet` | Drain-style parsing + sessionization |
| `features/` | Sharva | `data/processed/features_df.parquet` | Statistical + temporal + severity feature engineering |
| `ml/` | Shreeraksha M | `data/processed/anomaly_df.parquet` | IsolationForest + z-score hybrid anomaly detection |
| `correlation/` | Sumukha Rao H | `data/processed/graph_scores_df.parquet` | Co-occurrence graph, centrality scores, sequence detection |
| `scoring/` | Ujwal Hegde | `data/processed/scored_logs_df.parquet` | Final importance score, labels, incident clustering |
| `storage/` | — | Postgres / Elasticsearch | Persistence layer |
| `common/` | Sumukha Rao H | — | Shared config, logger, utils, schema definitions |
| `pipeline.py` | Sumukha Rao H | all of the above | End-to-end orchestrator |

### sessionized_logs.parquet schema (canonical — all downstream modules read from this)

| Column | Type | Description |
|--------|------|-------------|
| `log_id` | str | Unique per-row identifier, e.g. `log_000001` |
| `raw_text` | str | Original unparsed log line |
| `timestamp` | datetime64 | UTC datetime |
| `source` | str | Hostname / device |
| `session_id` | str | Session group, e.g. `session_0001` |
| `template_id` | str | Drain template slug, e.g. `IF_DOWN` |
| `severity` | str | `CRITICAL` / `ERROR` / `WARN` / `INFO` |
| `log_level` | str | Alias of `severity` (backwards compat) |
| `is_anomaly` | bool | Set by anomaly detector; always `False` from parsing |
| `anomaly_label` | str | Non-empty only when `is_anomaly=True` |

## Development Progress

| Module | Status | Notes |
| --------------------------------------- | ----------- | --------------------------------------------------------------- |
| `common/config.py` | Done | Lazy env-var access, graph + centrality + sequence constants |
| `common/env_handler.py` | Done | Fail-fast `.env` loader |
| `correlation/graph_builder.py` | Done | Co-occurrence graph, parquet loader, pickle cache, nx converter |
| `correlation/centrality.py` | Done | Degree, betweenness (k-approx), PageRank; `graph_scores_df` assembly |
| `correlation/sequence_engine.py` | Done | Session-scoped recurring sequence detection; `sequences.json` |
| `correlation/graph_visualizer.py` | Done | JSON export of nodes + edges with centrality scores |
| `correlation/run_correlation.py` | Done | End-to-end pipeline entry point |
| `correlation/tests/test_correlation.py` | Done | 70 unit tests, all passing |
| `scripts/generate_real_logs.py` | Done | Synthetic log generator (5 000 rows, 50 sessions, 20 templates) |
| `ingestion/` | Done | `batch_loader.py` + documented `fluent_bit.conf` |
| `parsing/` | Done | Drain parser, normalizer, sessionizer, template extractor |
| `features/` | Done | Statistical + temporal + severity + counter proximity features |
| `ml/` | Done | Isolation Forest + Z-score hybrid anomaly detection |
| `scoring/` | Done | Full importance scoring, incident clustering, root-cause analysis |
| `storage/` | Done | Postgres upsert writer; Elasticsearch writer |
| `common/` | Done | Config, logger, utils, Pydantic-style schema definitions |
| `pipeline.py` | Done | End-to-end orchestrator with `--dry-run` and `--from-step` |
| `visualization/` | Skeleton | Grafana + Kibana dashboards (see infra setup below) |
| `evaluation/` | Skeleton | Latency benchmark + metrics report |

## Architecture & Modules

The project is structured into logical modules reflecting the steps in the ML pipeline:

- **`data/`**: Directory for storing raw and processed datasets.
- **`ingestion/`**: Data ingestion processes to bring in network logs.
- **`parsing/`**: Parsers to convert raw log lines into structured data formats.
- **`features/`**: Feature engineering logic to transform parsed data into model-ready features.
- **`ml/`**: Machine Learning models specifically built for detecting anomalies in log data.
- **`correlation/`**: Identifies correlations between log events and anomalies. See details below.
- **`scoring/`**: Mechanisms to compute an "importance score" for log events to prioritize review.
- **`storage/`**: Interfaces for data persistence, interacting with PostgreSQL and Elasticsearch.
- **`visualization/`**: Configurations and instructions for dashboarding via Kibana and Grafana.
- **`evaluation/`**: Contains scripts and utilities for evaluating model and pipeline performance.
- **`pipeline.py`**: The central orchestrator script that ties the modules together.

## Correlation Module (Phase 3)

### Overview

Builds a weighted co-occurrence graph from sessionized log data, computes centrality
scores for every template node, detects recurring ordered log sequences, and ships
a per-log-row parquet file to the scoring stage (P4).

### Components

| File | Purpose |
|---|---|
| `graph_builder.py` | Build graph, load from parquet, pickle cache, NetworkX conversion |
| `centrality.py` | Degree / betweenness / PageRank centrality; assemble `graph_scores_df` |
| `sequence_engine.py` | Detect recurring ordered sequences across sessions |
| `graph_visualizer.py` | Export graph as JSON for downstream visualization |
| `run_correlation.py` | Pipeline entry point wiring all components |

### Graph Schema

The correlation graph is a weighted undirected graph where:

- **Node** = a unique log template string or an anomaly marker
  - `id`: template string or `anomaly:<label>`
  - `node_type`: `"log_template"` or `"anomaly"`
  - `count`: raw occurrence frequency
- **Edge** = co-occurrence within a configurable time window
  - `co_occurrences`: raw count of windows where both nodes appear together
  - `weight`: normalized value in `(0, 1]`, where `1.0` is the most frequent pair

### Configuration (`common/config.py`)

| Constant | Default | Description |
|---|---|---|
| `CORRELATION_TIME_WINDOW_SECONDS` | `60` | Window width for co-occurrence detection |
| `MAX_GRAPH_NODES` | `500` | Cap on template nodes; anomaly nodes always admitted |
| `PAGERANK_ALPHA` | `0.85` | PageRank damping factor |
| `BETWEENNESS_K` | `50` | Pivot samples for betweenness approximation |
| `BETWEENNESS_LARGE_GRAPH_THRESHOLD` | `200` | Use k-approx when node count exceeds this |
| `SEQUENCE_WINDOW_SECONDS` | `30` | Max elapsed time between events in a sequence |
| `SEQUENCE_MIN_LENGTH` | `3` | Minimum templates per sequence |
| `SEQUENCE_MIN_SUPPORT` | `3` | Minimum sessions exhibiting a sequence |

### Output Contract (`graph_scores_df.parquet`)

One row per log event. Consumed by the scoring module (P4).

| Column | Type | Description |
|---|---|---|
| `log_id` | str | Unique log identifier |
| `centrality_score` | float [0, 1] | PageRank score (primary signal for P4) |
| `degree` | int | Raw edge count in the correlation graph |
| `betweenness` | float [0, 1] | Normalized betweenness centrality |
| `cluster_id` | str | Connected-component label (`cc_0`, `cc_1`, ...) |
| `in_sequence` | bool | True if this log is part of a detected recurring sequence |
| `correlated_log_ids` | list[str] | Other log_ids in the same session whose template is a graph neighbor |

### Sequence Output (`sequences.json`)

```json
[
  {
    "sequence": ["IF_DOWN", "BGP_PEER_RESET", "OSPF_ADJACENCY_LOST"],
    "support_count": 12,
    "session_ids": ["session_001", "session_007"]
  }
]
```

### Running the Correlation Pipeline

**Generate synthetic data** (first run or when you want fresh data):

```bash
python scripts/generate_real_logs.py
```

**Run the full pipeline:**

```bash
python -m correlation.run_correlation
```

Force graph rebuild (skip pickle cache):

```bash
REBUILD_GRAPH=1 python -m correlation.run_correlation
```

**Unit tests (70 tests):**

```bash
python -m pytest correlation/tests/test_correlation.py -v
```

**Manual inspection (synthetic 5-node example):**

```bash
python -m correlation.manual_test
```

### Output Files

| File | Description |
|---|---|
| `data/processed/correlation_graph.gpickle` | Pickled graph (rebuild cache) |
| `data/processed/sequences.json` | Detected recurring sequences |
| `data/processed/graph_scores_df.parquet` | P4 handoff — one row per log |
| `data/processed/correlation_graph.json` | Graph JSON for visualization |
| `parsing/processed/sessionized_logs.parquet` | Sessionized log input |

## ML Module (Anomaly Detection)

### Overview

The ML module detects anomalies using:

- Isolation Forest (ML-based)
- Z-score (statistical)

### Model Design

- Isolation Forest → learns normal behavior
- Z-score → measures deviation

Hybrid:

```text
combined_score = w1 * isolation_score + w2 * zscore
```

where:

- w1 = weight for ML signal
- w2 = weight for statistical signal

### Configuration (common/config.py)

| Parameter                | Description                       |
| ------------------------ | --------------------------------- |
| contamination            | Expected proportion of anomalies  |
| weight_isolation         | Weight for Isolation Forest score |
| weight_zscore            | Weight for Z-score contribution   |
| training_window_sessions | Number of sessions for retraining |

### Output Contract

| Column          | Description            |
| --------------- | ---------------------- |
| log_id          | Unique identifier      |
| isolation_score | Isolation Forest score |
| zscore          | Statistical deviation  |
| combined_score  | Hybrid score           |
| is_anomaly      | Boolean anomaly flag   |

Saved to:

```text
data/processed/anomaly_df.parquet
```

### Model Training

Handled via `ml/trainer.py`:

- Trains Isolation Forest
- Saves model:

```text
ml/model_store/isolation_forest_v{timestamp}.pkl
```

### Cold-Start Strategy

If insufficient data:

- predictions unstable
- fallback:
  - rely on z-score
  - or mark non-anomalous

### Testing

```bash
pytest ml/tests/test_anomaly.py
```

## Infrastructure Setup

To run the pipeline infrastructure, we use Docker Compose to spin up the required databases and visualization tools.

### 1. Setup Virtual Environment and Dependencies

Create a virtual environment and install the required libraries:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Set up your local environment variables:

```bash
cp .env.example .env
```

### 3. Start the Services

Deploy the infrastructure stack:

```bash
docker compose up -d
```

### 4. Running Scripts

To avoid import errors, **always run scripts from the project root** using the `-m` (module) flag:

```bash
python3 -m storage.db_writer
```

## Environment Variable Handling

All environment variables must be accessed using the global environment handler via `common.config`.

### Usage

```python
# Variables are lazily evaluated. They are only checked when actually imported/used.
from common.config import DB_URL
```

### Behavior

- **Lazy Evaluation**: Fails only when a specific variable is accessed.
- **Clean Errors**: If a variable is missing, the app outputs a clear message and exits without a long traceback:
  `[ENV ERROR] Missing required variable: NAME`

## Rules

- Do not use `os.getenv()` or `dotenv` directly.
- Always import variables from `common.config`.
- Never print raw environment variables to the console.
- Always run scripts from the project root using `python3 -m <module>` to avoid import errors.

## Scoring Module (Importance Scoring)

The `scoring/` module integrates outputs from:
- feature engineering
- ML anomaly detection
- graph correlation

to generate:
- final importance scores
- severity labels
- incident clusters
- root-cause candidates

### Components

#### `importance_scorer.py`
- Loads:
  - `features_df.parquet`
  - `anomaly_df.parquet`
  - `graph_scores_df.parquet`
- Merges inputs on `log_id`
- Handles missing ML/graph outputs using fallback scores (`0.0`)
- Computes:

```text
final_score =
    (ML_WEIGHT * combined_score)
  + (GRAPH_WEIGHT * centrality_score)
  + (RULE_WEIGHT * severity_weight)
```

- Saves:
  - `data/processed/scored_logs_df.parquet`

---

#### `label_mapper.py`
Maps `final_score` into:
- `ignore`
- `low`
- `medium`
- `critical`

using thresholds defined in `common/config.py`.

---

#### `incident_clusterer.py`
- Uses DBSCAN clustering
- Clustering features:
  - `final_score`
  - `centrality_score`
  - `time_delta_session_start`
- Assigns incident IDs:
  - `INC-000`
  - `INC-001`
- Noise points receive:
  - `incident_id = None`

### DBSCAN Configuration

| Parameter | Default |
|-----------|----------|
| `DBSCAN_EPS` | `0.5` |
| `DBSCAN_MIN_SAMPLES` | `5` |

---

#### `root_cause_engine.py`
- Identifies top root-cause candidates per incident
- Ranks logs using `centrality_score`
- Computes:
  - `root_cause_confidence`
- Saves:
  - `data/processed/root_causes_df.parquet`

---

## Output Contracts

### `scored_logs_df.parquet`

Schema:
- `log_id`
- `final_score`
- `label`
- `incident_id`
- `is_root_cause`
- `root_cause_confidence`

### `root_causes_df.parquet`

Schema:
- `incident_id`
- `root_cause_log_id`
- `confidence_score`

---

## Tests

Implemented in:

```text
scoring/tests/test_scoring.py
```

Run using:

```bash
pytest scoring/tests/test_scoring.py -v
```
data/processed/scored_logs_df.parquet
```

## Visualizing Results (Grafana & Kibana)

After deploying the infrastructure using Docker Compose (`docker compose up -d`), you can visualize the log data and anomaly scores.

### 1. Seed Synthetic Data
If you haven't processed real logs yet, you can seed the databases with synthetic testing data (generated for **May 1, 2026**):
```bash
python3 -m storage.db_writer
python3 -m storage.es_writer
```

### 2. Grafana Dashboard
Grafana is used to visualize log importance scores and anomaly rates over time.
1. Open **[http://localhost:3000](http://localhost:3000)** (Default login: `admin` / `admin`).
2. **Add PostgreSQL Data Source**:
   - Go to **Connections** → **Data sources** → **Add data source** → **PostgreSQL**.
   - **Host URL**: `postgres:5432`
   - **Database name**: `log-postgres`
   - **Username**: `log-user`
   - **Password**: `[PASSWORD]`
   - **TLS/SSL Mode**: `disable`
   - Click **Save & test**.
3. **Import Dashboard**:
   - Go to **Dashboards** → **New** → **Import**.
   - Upload `visualization/grafana/dashboard.json`.
   - Select the PostgreSQL data source you just created and click **Import**.
4. **View Data**: 
   - Since the synthetic data is seeded for May 2026, **change the time filter** in the top-right corner to include **May 1, 2026**.

### 3. Kibana Dashboard
Kibana is used for deep-dive searches and incident drill-downs.
1. Open **[http://localhost:5601](http://localhost:5601)** (No login required).
2. **Import Dashboard**:
   - Go to the **Hamburger Menu** → **Stack Management** → **Saved Objects**.
   - Click **Import** and upload `visualization/kibana/dashboard.ndjson`.
3. **View Dashboards & Drill-down**:
   - Go to **Analytics** → **Dashboard** and open **`Logs_Dash`**.
   - Go to **Analytics** → **Discover** and open the **`All_Logs`** saved search.
   - You will see a data table with custom columns (`timestamp`, `incident_id`, `label`, `final_score`, `raw_text`).
   - Click the **`+` magnifying glass icon** next to any `incident_id` to instantly drill down and filter the entire view by that incident cluster.
4. **View Data**:
   - Just like Grafana, ensure your time filter in the top-right corner is set to include **May 1, 2026**.
