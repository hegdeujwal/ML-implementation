import json
from pathlib import Path

import pandas as pd
from elasticsearch import Elasticsearch, helpers
from elasticsearch.helpers import BulkIndexError

from common.config import ELASTIC_URL

INDEX_NAME = "scored-logs"

MAPPING_PATH = Path(
    "storage/es_index_mapping.json"
)


def get_client():
    return Elasticsearch(ELASTIC_URL)


def ensure_index(client):
    mapping = json.loads(
        MAPPING_PATH.read_text(
            encoding="utf-8"
        )
    )

    if not client.indices.exists(index=INDEX_NAME):
        client.indices.create(
            index=INDEX_NAME,
            body=mapping,
        )


def index_logs(df):
    client = get_client()

    ensure_index(client)

    actions = []

    for _, row in df.iterrows():
        log_id = row.get("log_id")
        
        # Skip documents without log_id (required for document _id)
        if pd.isna(log_id) or log_id is None:
            continue

        timestamp = row.get("timestamp")
        
        # Helper function to convert NaN to empty string for keyword fields
        def safe_str(val):
            if pd.isna(val) or val is None:
                return ""
            return str(val)
        
        doc = {
            "log_id": str(log_id),
            "timestamp": (
                timestamp.isoformat()
                if pd.notnull(timestamp)
                else None
            ),
            "raw_text": row.get("raw_text") or "",
            "label": safe_str(row.get("label")),
            "final_score": float(
                row.get("final_score", 0)
            ),
            "incident_id": safe_str(row.get("incident_id")),
            "is_root_cause": bool(
                row.get("is_root_cause", False)
            ),
            "root_cause_confidence": float(
                row.get(
                    "root_cause_confidence",
                    0
                )
            ),
        }

        # Only include document if it has valid data
        if doc["log_id"]:
            actions.append({
                "_index": INDEX_NAME,
                "_id": str(log_id),
                "_source": doc,
            })

    if not actions:
        print("[WARNING] No valid documents to index")
        return 0

    try:
        success, errors = helpers.bulk(
            client,
            actions,
            refresh="wait_for",
        )

        if errors:
            print(f"[ERROR] {len(errors)} document(s) failed to index:")
            for error in errors[:5]:  # Print first 5 errors
                print(f"  {error}")

        return success
    except BulkIndexError as e:
        print(f"[ERROR] Bulk indexing failed: {len(e.errors)} document(s) failed to index.")
        print("[ERROR] First few errors:")
        for error in e.errors[:5]:
            print(f"  {error}")
        raise


if __name__ == "__main__":

    scores_df = pd.read_parquet(
        "data/processed/scored_logs_df.parquet"
    )

    logs_df = pd.read_parquet(
        "data/processed/sessionized_logs.parquet"
    )

    # Ensure log_id exists in both DataFrames
    if "log_id" not in scores_df.columns and "sequence_number" in scores_df.columns:
        scores_df = scores_df.copy()
        scores_df["log_id"] = scores_df["sequence_number"].map(
            lambda x: f"log_{int(x)}" if pd.notnull(x) else None
        )

    if "log_id" not in logs_df.columns and "sequence_number" in logs_df.columns:
        logs_df = logs_df.copy()
        logs_df["log_id"] = logs_df["sequence_number"].map(
            lambda x: f"log_{int(x)}" if pd.notnull(x) else None
        )

    # Select available columns
    log_cols = ["log_id", "timestamp"]
    if "raw_text" in logs_df.columns:
        log_cols.append("raw_text")

    merged = scores_df.merge(
        logs_df[log_cols],
        on="log_id",
        how="left",
    )

    inserted = index_logs(merged)

    print({
        "indexed_docs": inserted
    })