from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json, execute_values

from common.config import DB_URL

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
PROCESSED_DIR = Path("data/processed")
SESSIONIZED_PATH = PROCESSED_DIR / "sessionized_logs.parquet"

SYSTEM_COLUMNS = {"created_at", "updated_at"}


def get_connection(conn: Optional[psycopg2.extensions.connection] = None):
    return conn or psycopg2.connect(DB_URL)


def _ensure_log_id(df: pd.DataFrame) -> pd.DataFrame:
    """Create a deterministic log_id from sequence_number when missing."""
    if "log_id" in df.columns:
        if df["log_id"].isna().any() and "sequence_number" in df.columns:
            df = df.copy()
            df["log_id"] = df["log_id"].fillna(
                df["sequence_number"].map(
                    lambda x: f"log_{int(x)}" if pd.notnull(x) else None
                )
            )
        return df

    if "sequence_number" not in df.columns:
        raise ValueError(
            "DataFrame must contain either 'log_id' or 'sequence_number' "
            "to write to a table keyed by log_id."
        )

    df = df.copy()
    df["log_id"] = df["sequence_number"].map(
        lambda x: f"log_{int(x)}" if pd.notnull(x) else None
    )
    return df


def apply_schema(conn: psycopg2.extensions.connection):
    with conn.cursor() as cur:
        cur.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()


def _table_columns(conn, table_name: str):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema='public'
              AND table_name=%s
            ORDER BY ordinal_position
            """,
            (table_name,),
        )

        return [r[0] for r in cur.fetchall()]


def _normalize_df(df: pd.DataFrame, columns: Iterable[str]):
    normalized = df.copy()

    for c in columns:
        if c not in normalized.columns:
            normalized[c] = None

    for c in normalized.columns:
        if normalized[c].dtype == "object":
            normalized[c] = normalized[c].map(
                lambda x: Json(x)
                if isinstance(x, (dict, list))
                else x
            )

    ordered = [c for c in columns if c in normalized.columns]

    # Convert pandas NaN/NaT to Python None so they are written as SQL NULL.
    # Without this, a float NaN sent to a nullable INTEGER column (e.g. an
    # unassigned chain_position) makes Postgres attempt NaN→int and raise
    # "integer out of range" (error 22003) rather than inserting NULL.
    for c in ordered:
        col = normalized[c]
        if col.isna().any():
            normalized[c] = col.astype(object).where(col.notna(), None)

    return normalized[ordered]


_TABLE_KEY_MAP: dict = {
    "logs": "sequence_number",
    "features": "sequence_number",
    "anomalies": "sequence_number",
    "scores": "sequence_number",
    "incidents": "incident_id",
    "incident_history": "incident_id",
}


def _upsert_key_for_table(table_name: str) -> str:
    return _TABLE_KEY_MAP.get(table_name, "sequence_number")


def write_dataframe(
    df: pd.DataFrame,
    table_name: str,
    conn=None,
):
    if df is None or df.empty:
        return 0

    owned_conn = conn is None

    conn = get_connection(conn)

    try:
        table_cols = [
            c
            for c in _table_columns(conn, table_name)
            if c not in SYSTEM_COLUMNS
        ]

        use_df = _normalize_df(df, table_cols)

        key_col = _upsert_key_for_table(table_name)

        if key_col == "log_id":
            use_df = _ensure_log_id(use_df)

        insert_cols = [
            c for c in table_cols
            if c in use_df.columns
        ]

        update_cols = [
            c for c in insert_cols
            if c != key_col
        ]

        assignments = [
            sql.SQL("{} = EXCLUDED.{}").format(
                sql.Identifier(c),
                sql.Identifier(c),
            )
            for c in update_cols
        ]

        assignments.append(
            sql.SQL("updated_at = NOW()")
        )

        query = sql.SQL(
            """
            INSERT INTO {table} ({columns})
            VALUES %s
            ON CONFLICT ({key_col})
            DO UPDATE SET {updates}
            """
        ).format(
            table=sql.Identifier(table_name),
            columns=sql.SQL(", ").join(
                sql.Identifier(c)
                for c in insert_cols
            ),
            key_col=sql.Identifier(key_col),
            updates=sql.SQL(", ").join(assignments),
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
                page_size=1000,
            )

        conn.commit()

        return len(rows)

    except Exception:
        conn.rollback()
        raise

    finally:
        if owned_conn:
            conn.close()


def write_logs(df, conn=None):
    return write_dataframe(df, "logs", conn)


def write_features(df, conn=None):
    return write_dataframe(df, "features", conn)


def write_anomalies(df, conn=None):
    return write_dataframe(df, "anomalies", conn)


def write_scores(df, conn=None):
    return write_dataframe(df, "scores", conn)


def write_incidents(df, conn=None):
    return write_dataframe(df, "incidents", conn)


def write_incident_history(df, conn=None):
    """Upsert incident_history rows (keyed on incident_id)."""
    return write_dataframe(df, "incident_history", conn)


def query_incident_history(
    lookback_hours: int = 72,
    conn=None,
) -> "pd.DataFrame":
    """Fetch recent incident_history rows from Postgres.

    Falls back to an empty DataFrame if the table does not exist yet.
    """
    import pandas as _pd

    owned_conn = conn is None
    conn = get_connection(conn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT *
                FROM incident_history
                WHERE end_time >= NOW() - INTERVAL '%s hours'
                ORDER BY end_time DESC
                """,
                (lookback_hours,),
            )
            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description]
        return _pd.DataFrame(rows, columns=cols)
    except Exception:
        return _pd.DataFrame()
    finally:
        if owned_conn:
            conn.close()


# =====================================
# summaries
# =====================================

def write_summary(correlation_id: str, summary_text: str):
    df = pd.DataFrame([
        {
            "correlation_id": correlation_id,
            "summary_text": summary_text,
        }
    ])

    return write_dataframe(df, "summaries")


def get_summary(correlation_id: str):
    conn = get_connection()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT summary_text
                FROM summaries
                WHERE correlation_id = %s
                """,
                (correlation_id,),
            )

            row = cur.fetchone()

            return row[0] if row else None

    finally:
        conn.close()


def write_summaries_batch(summaries: list[dict]):
    if not summaries:
        return 0

    df = pd.DataFrame(summaries)

    return write_dataframe(df, "summaries")


# =====================================
# LOAD REAL PIPELINE OUTPUTS
# =====================================

def load_pipeline_outputs():
    conn = get_connection()

    try:
        apply_schema(conn)

        # logs
        logs_df = pd.read_parquet(
            PROCESSED_DIR / "sessionized_logs.parquet"
        )

        logs_df["incident_id"] = None
        logs_df["label"] = None

        logs_df = logs_df.rename(
            columns={
                "source": "host",
                "severity": "log_level",
            }
        )

        write_logs(logs_df, conn)

        # features
        features_path = (
            PROCESSED_DIR / "features_df.parquet"
        )

        if features_path.exists():
            write_features(
                pd.read_parquet(features_path),
                conn,
            )

        # anomalies
        anomaly_path = (
            PROCESSED_DIR / "anomaly_df.parquet"
        )

        if anomaly_path.exists():
            write_anomalies(
                pd.read_parquet(anomaly_path),
                conn,
            )

        # scores
        scores_df = pd.read_parquet(
            PROCESSED_DIR / "scored_logs_df.parquet"
        )
        
        # Handle both correlation_id (old) and incident_id (new)
        if "incident_id" in scores_df.columns and "correlation_id" not in scores_df.columns:
            scores_df = scores_df.rename(columns={"incident_id": "correlation_id"})

        write_scores(scores_df, conn)

        # incidents
        rc_path = (
            PROCESSED_DIR / "root_causes_df.parquet"
        )
        if rc_path.exists():

            rc_df = pd.read_parquet(rc_path)
            
            # Merge scores_df with logs_df to get timestamp for groupby
            logs_df = pd.read_parquet(SESSIONIZED_PATH)
            scores_with_ts = scores_df.merge(
                logs_df[["sequence_number", "timestamp"]],
                on="sequence_number",
                how="left"
            )

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

            incidents_df["severity"] = incidents_df["label"]
            incidents_df["status"] = "open"

            # Keep only the highest-confidence root cause per incident
            rc_df_best = rc_df.sort_values("confidence_score", ascending=False).drop_duplicates("incident_id")

            incidents_df = incidents_df.merge(
                rc_df_best,
                on="incident_id",
                how="left",
            )

            incidents_df = incidents_df.rename(
                columns={
                    "confidence_score":
                        "root_cause_confidence",
                }
            )

            # Convert root_cause_log_id from integer to log_id string format
            # to match FK constraint on logs(log_id)
            incidents_df["root_cause_log_id"] = (
                incidents_df["root_cause_log_id"]
                .fillna(0)
                .astype(int)
                .apply(lambda x: f"log_{x}" if x > 0 else None)
            )

            write_incidents(
                incidents_df,
                conn,
            )

        print("[INFO] pipeline outputs loaded")

    finally:
        conn.close()


if __name__ == "__main__":
    load_pipeline_outputs()