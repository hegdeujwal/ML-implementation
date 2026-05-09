"""
Counter anomaly proximity scoring.

Measures how close a log event is to known
counter/interface anomaly events within a time window.
"""




import pandas as pd

from common.config import (
    COUNTER_PROXIMITY_WINDOW_SECONDS
)


COUNTER_ANOMALY_TEMPLATES = [
    "IF_DOWN",
    "PORT_SCAN",
]


def add_counter_proximity(df: pd.DataFrame) -> pd.DataFrame:

    anomaly_rows = df[
        df["template_id"].isin(
            COUNTER_ANOMALY_TEMPLATES
        )
    ]

    if anomaly_rows.empty:
        df["counter_proximity"] = 0.0
        return df

    scores = []

    for _, row in df.iterrows():

        session = row["session_id"]
        timestamp = row["timestamp"]

        subset = anomaly_rows[
            anomaly_rows["session_id"] == session
        ]

        if subset.empty:
            scores.append(0.0)
            continue

        min_distance = (
            subset["timestamp"] - timestamp
        ).abs().dt.total_seconds().min()

        if min_distance <= COUNTER_PROXIMITY_WINDOW_SECONDS:
            score = 1 / (1 + min_distance)
        else:
            score = 0.0

        scores.append(score)

    df["counter_proximity"] = scores

    return df