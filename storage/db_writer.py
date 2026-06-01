import numpy as np
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json, execute_values

from common.config import DB_URL

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
SYSTEM_COLUMNS = {"created_at", "updated_at"}


def get_connection(conn: Optional[psycopg2.extensions.connection] = None):
    return conn or psycopg2.connect(DB_URL)


def apply_schema(conn: psycopg2.extensions.connection) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()


def _table_columns(conn, table_name: str) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            ORDER BY ordinal_position
            """,
            (table_name,),
        )
        return [r[0] for r in cur.fetchall()]


def _normalize_df(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    normalized = df.copy()

    for c in columns:
        if c not in normalized.columns:
            normalized[c] = None

    for c in normalized.columns:
        if normalized[c].dtype == "object":
            normalized[c] = normalized[c].map(
                lambda x: Json(x) if isinstance(x, (dict, list)) else x
            )

    ordered = [c for c in columns if c in normalized.columns]
    return normalized[ordered]


_TABLE_KEY_MAP: dict = {
    "logs": "sequence_number",
    "features": "sequence_number",
    "anomalies": "sequence_number",
    "scores": "sequence_number",
    "incidents": "incident_id",
}


def _upsert_key_for_table(table_name: str) -> str:
    return _TABLE_KEY_MAP.get(table_name, "sequence_number")


def write_dataframe(
    df: pd.DataFrame,
    table_name: str,
    conn: Optional[psycopg2.extensions.connection] = None,
) -> int:
    if df is None or df.empty:
        return 0

    owned_conn = conn is None
    conn = get_connection(conn)

    try:
        table_cols = [
            c for c in _table_columns(conn, table_name)
            if c not in SYSTEM_COLUMNS
        ]

        if not table_cols:
            raise ValueError(
                f"Table `{table_name}` not found or has no writable columns"
            )

        use_df = _normalize_df(df, table_cols)

        key_col = _upsert_key_for_table(table_name)

        if key_col not in use_df.columns:
            raise ValueError(
                f"Missing required key column `{key_col}` for table `{table_name}`"
            )

        insert_cols = [c for c in table_cols if c in use_df.columns]
        update_cols = [c for c in insert_cols if c != key_col]

        assignments = [
            sql.SQL("{} = EXCLUDED.{}").format(
                sql.Identifier(c),
                sql.Identifier(c)
            )
            for c in update_cols
        ]

        assignments.append(sql.SQL("updated_at = NOW()"))

        update_clause = sql.SQL(", ").join(assignments)

        query = sql.SQL(
            """
            INSERT INTO {table} ({columns}) VALUES %s
            ON CONFLICT ({key_col})
            DO UPDATE SET {updates}
            """
        ).format(
            table=sql.Identifier(table_name),
            columns=sql.SQL(", ").join(
                sql.Identifier(c) for c in insert_cols
            ),
            key_col=sql.Identifier(key_col),
            updates=update_clause,
        )

        rows = [
            tuple(row[c] for c in insert_cols)
            for _, row in use_df.iterrows()
        ]

        with conn.cursor() as cur:
            execute_values(
                cur,
                query.as_string(conn),
                rows,
                page_size=1000
            )

        conn.commit()
        return len(rows)

    except Exception:
        conn.rollback()
        raise

    finally:
        if owned_conn:
            conn.close()


def write_logs(df: pd.DataFrame, conn=None) -> int:
    return write_dataframe(df, "logs", conn)


def write_features(df: pd.DataFrame, conn=None) -> int:
    return write_dataframe(df, "features", conn)


def write_anomalies(df: pd.DataFrame, conn=None) -> int:
    return write_dataframe(df, "anomalies", conn)


def write_scores(df: pd.DataFrame, conn=None) -> int:
    return write_dataframe(df, "scores", conn)


def write_incidents(df: pd.DataFrame, conn=None) -> int:
    return write_dataframe(df, "incidents", conn)


def _synthetic_data(n: int = 200) -> dict[str, pd.DataFrame]:
    timestamps = pd.date_range("2026-05-01", periods=n, freq="min")

    log_ids = [f"log_{i:06d}" for i in range(n)]

    labels = ["ignore", "low", "medium", "critical"]

    logs = pd.DataFrame(
        {
            "log_id": log_ids,
            "sequence_number": range(n),
            "timestamp": timestamps,
            "source_type": "synthetic",

            "service": [
                ["auth", "payment", "network", "database"][i % 4]
                for i in range(n)
            ],

            "host": [
                ["server1", "server2", "server3"][i % 3]
                for i in range(n)
            ],

            "log_level": [
                ["INFO", "WARN", "ERROR", "CRITICAL"][i % 4]
                for i in range(n)
            ],

            "event_type": [
                ["login", "transaction", "api", "db_query"][i % 4]
                for i in range(n)
            ],

            "event_action": [
                ["success", "failed", "timeout"][i % 3]
                for i in range(n)
            ],

            "template_id": [f"tmpl_{i % 5}" for i in range(n)],

            "message": [
                [
                    "User login successful",
                    "Payment transaction failed",
                    "API timeout detected",
                    "Database connection error",
                ][i % 4]
                for i in range(n)
            ],

            "raw_text": [
                [
                    "INFO auth login successful",
                    "ERROR payment transaction failed",
                    "WARN api timeout detected",
                    "CRITICAL database connection error",
                ][i % 4]
                for i in range(n)
            ],

            "metadata": [
                {
                    "incident_id": f"inc_{i // 20:03d}",
                    "label": labels[i % 4],
                }
                for i in range(n)
            ],

            "session_id": [
                f"s_{i // 10:03d}"
                for i in range(n)
            ],
        }
    )

    features = pd.DataFrame(
        {
            "log_id": log_ids,
            "timestamp": timestamps,
            "label": [labels[i % 4] for i in range(n)],
            "incident_id": [f"inc_{i // 20:03d}" for i in range(n)],

            "frequency": np.random.randint(1, 50, size=n),

            "event_weight": np.random.uniform(0.1, 1.0, size=n),

            "frequency_score": np.random.uniform(0.1, 1.0, size=n),

            "severity_weight": np.random.uniform(0.1, 1.0, size=n),

            "counter_proximity": np.random.uniform(0.1, 1.0, size=n),

            "feature_payload": [
                {"source": "synthetic"}
            ] * n,

            "in_sequence": [
                i % 2 == 0
                for i in range(n)
            ],
        }
    )

    anomalies = pd.DataFrame(
        {
            "log_id": log_ids,

            "incident_id": [
                f"inc_{i // 20:03d}"
                for i in range(n)
            ],

            "timestamp": timestamps,

            "label": [
                labels[i % 4]
                for i in range(n)
            ],

            "isolation_score": np.random.uniform(
                0.1,
                1.0,
                size=n
            ),

            "zscore": np.random.uniform(
                0.1,
                5.0,
                size=n
            ),

            "anomaly_score": np.random.uniform(
                0.1,
                1.0,
                size=n
            ),

            "is_anomaly": np.random.choice(
                [True, False],
                size=n,
                p=[0.12, 0.88]
            ),

            "in_sequence": [
                i % 2 == 0
                for i in range(n)
            ],
        }
    )

    scores = pd.DataFrame(
        {
            "log_id": log_ids,

            "importance_score": np.random.uniform(
                0.1,
                1.0,
                size=n
            ),

            "final_score": np.random.uniform(
                0.1,
                1.0,
                size=n
            ),

            "label": [
                labels[i % 4]
                for i in range(n)
            ],

            "correlation_id": [
                f"corr_{i // 10:03d}"
                for i in range(n)
            ],

            "incident_id": [
                f"inc_{i // 20:03d}"
                for i in range(n)
            ],

            "is_root_cause": [
                i % 25 == 0
                for i in range(n)
            ],

            "root_cause_confidence": [
                0.85 if i % 25 == 0 else 0.15
                for i in range(n)
            ],

            "in_sequence": [
                i % 2 == 0
                for i in range(n)
            ],

            "timestamp": timestamps,
        }
    )

    incidents = pd.DataFrame(
        {
            "incident_id": [
                f"inc_{i:03d}"
                for i in range(max(1, n // 20))
            ],

            "start_time": [
                timestamps[i * 20]
                for i in range(max(1, n // 20))
            ],

            "end_time": [
                timestamps[min(n - 1, i * 20 + 19)]
                for i in range(max(1, n // 20))
            ],

            "root_cause_log_id": [
                f"log_{i * 20:06d}"
                for i in range(max(1, n // 20))
            ],

            "severity": np.random.choice(
                ["low", "medium", "high", "critical"],
                size=max(1, n // 20)
            ),

            "label": np.random.choice(
                ["low", "medium", "critical"],
                size=max(1, n // 20)
            ),

            "root_cause_confidence": np.random.uniform(
                0.5,
                1.0,
                size=max(1, n // 20)
            ),

            "log_count": np.random.randint(
                5,
                30,
                size=max(1, n // 20)
            ),

            "status": np.random.choice(
                ["open", "investigating", "resolved"],
                size=max(1, n // 20)
            ),
        }
    )

    return {
        "logs": logs,
        "features": features,
        "anomalies": anomalies,
        "scores": scores,
        "incidents": incidents,
    }


def seed_synthetic_data(conn=None, n: int = 200) -> dict[str, int]:
    owned_conn = conn is None

    conn = get_connection(conn)

    try:
        apply_schema(conn)

        data = _synthetic_data(n)

        out = {}

        for table in [
            "logs",
            "features",
            "anomalies",
            "scores",
            "incidents",
        ]:
            out[table] = write_dataframe(
                data[table],
                table,
                conn
            )

        return out

    finally:
        if owned_conn:
            conn.close()


if __name__ == "__main__":
    counts = seed_synthetic_data(n=250)

    print(
        {
            "seeded_rows": counts
        }
    )