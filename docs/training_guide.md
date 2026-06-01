# ML Training Guide: Network Anomaly Detection

This document outlines how model training works in the pipeline. Because the architecture uses an **unsupervised, online learning approach**, there is no manual "fit" script to run. Training is fully integrated into the batch processing pipeline.

## 1. How Training Works (Architecture)

The pipeline employs a hybrid **Isolation Forest** (for multidimensional feature anomalies) + **Welford Z-Score** (for single-metric frequency drift).

*   **No Manual Triggers**: You do not run a separate training script. The `AnomalyTrainer` in `ml/trainer.py` handles model lifecycle automatically during standard pipeline runs.
*   **Cold Start**: On the first pipeline run, if no model exists in the `model_store`, the pipeline trains an initial Isolation Forest on the current batch of logs.
*   **Rolling Feature Store**: During each pipeline run, features are extracted and appended to a sliding window store (`data/processed/feature_rolling_store.parquet`).
*   **Automatic Retraining**: The trainer tracks how many new logs have been seen since the last training. When this counter crosses `RETRAINING_TRIGGER_EVERY_K` (default: 1,000), the model automatically retrains using the historical data in the rolling feature store.
*   **Safe Serialization**: Saved models (`.pkl`) are accompanied by a JSON sidecar. The system validates `sklearn` versions and feature column signatures before loading to prevent silent score corruption.

## 2. Instructions for the ML Engineer

While the system is automated, your job is to configure the hyperparameters, provide baseline data, and monitor the drift detection. Here is your workflow:

### Step 1: Configure Hyperparameters
All training hyperparameters are centralized in `common/config.py`. You should tune these based on network traffic volume:
*   `IF_CONTAMINATION` (default: `0.05`): The expected proportion of anomalies. Lower this if the model is too noisy.
*   `RETRAINING_TRIGGER_EVERY_K` (default: `1000`): How many new logs trigger a background retrain.
*   `FEATURE_ROLLING_MAX_SESSIONS` (default: `50`): How many historical sessions to keep in the rolling store. Increase this if you want the model to have a longer memory.
*   `IF_FEATURE_COLUMNS`: The specific features fed into the Isolation Forest.

### Step 2: Establish the Initial Baseline (Cold Start)
To train the first robust model, feed the pipeline a clean, representative dataset (e.g., 24 hours of normal network logs).
```bash
# Run the pipeline in dry-run mode (skips DB write)
python pipeline.py --dry-run --log-file data/raw/baseline_logs.txt
```
*Expected Result*: A new `.pkl` and `.json` model will be created in `ml/model_store/`. The Welford z-score persistent store (`zscore_baseline_store.parquet`) will be initialized.

### Step 3: Validate Model Safety & Drift
As new logs come in, the pipeline will load the model from the `model_store`.
```bash
python pipeline.py --dry-run --log-file data/raw/new_daily_logs.txt
```
*Expected Result*: The logs will report `Using validated saved model for inference.` If the new data crosses the `RETRAINING_TRIGGER` threshold, you will see `AnomalyTrainer retrained on X rows.`

### Step 4: Run the ML Tests
If you make changes to `ml/trainer.py` or `ml/anomaly_detector.py`, ensure the test suite passes:
```bash
pytest ml/tests/test_anomaly.py -v
```

## 3. Common Troubleshooting

*   **"Refusing to load — retraining from scratch"**: If you change `IF_FEATURE_COLUMNS` in `config.py` or upgrade `scikit-learn`, the JSON sidecar validation will fail intentionally. The system will discard the incompatible model and automatically train a fresh one.
*   **Model is flagging too much**: Increase the `ZSCORE_BASELINE_N_SESSIONS` or decrease `IF_CONTAMINATION`.
*   **Disk space concerns**: The `FEATURE_ROLLING_STORE_PATH` max size is bounded by `FEATURE_ROLLING_MAX_SESSIONS`. It automatically prunes old sessions to prevent unbounded memory growth.
