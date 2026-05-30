import pandas as pd
from pathlib import Path

# Load the scored logs data
scored_logs_df = pd.read_parquet(Path("data/processed/scored_logs_df.parquet"))
print(f"Scored logs shape: {scored_logs_df.shape}")
print(f"Columns: {scored_logs_df.columns.tolist()}")
print(f"\nFirst few rows:")
print(scored_logs_df.head(3))
print(f"\nUnique incident_ids: {scored_logs_df['incident_id'].unique()}")

# Check sessionized logs
sessionized_logs = pd.read_parquet(Path("data/processed/sessionized_logs.parquet"))
print(f"\nSessionized logs shape: {sessionized_logs.shape}")
print(f"Sessionized columns: {sessionized_logs.columns.tolist()}")

# Simulate what db_writer does
print("\n--- Simulating db_writer incident logic ---")
scores_with_ts = scored_logs_df.merge(
    sessionized_logs[["sequence_number", "timestamp"]],
    on="sequence_number",
    how="left"
)
print(f"After merge with timestamps: {scores_with_ts.shape}")
print(f"Columns: {scores_with_ts.columns.tolist()}")

incidents_df = (
    scores_with_ts.groupby("incident_id")
    .agg(
        start_time=("timestamp", "min"),
        end_time=("timestamp", "max"),
        log_count=("log_id", "count"),
        label=("label", "max"),
    )
    .reset_index()
)
print(f"\nIncidents after groupby: {incidents_df.shape}")
print(incidents_df)
