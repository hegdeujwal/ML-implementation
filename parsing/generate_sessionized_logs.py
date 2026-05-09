"""
Generates synthetic sessionized log data for testing.

Creates realistic sample log sessions and stores them
as a parquet file for feature pipeline validation.
"""




import pandas as pd
from datetime import datetime, timedelta
import random
import os

rows = []

base = datetime.now()

for s in range(5):

    session_id = f"s_{s}"

    current = base

    for i in range(20):

        current += timedelta(
            seconds=random.randint(1, 10)
        )

        rows.append({
            "log_id": f"log_{s}_{i}",
            "session_id": session_id,
            "timestamp": current,
            "template_id": random.choice([
                "IF_DOWN",
                "CPU_HIGH",
                "LOGIN_FAIL"
            ]),
            "log_level": random.choice([
                "INFO",
                "WARN",
                "ERROR",
                "CRITICAL"
            ])
        })

df = pd.DataFrame(rows)

os.makedirs(
    "parsing/processed",
    exist_ok=True
)

df.to_parquet(
    "parsing/processed/sessionized_logs.parquet",
    index=False
)

print("done")