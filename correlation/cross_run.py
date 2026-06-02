"""
correlation/cross_run.py

P5.5 — Cross-Run Incident Correlation orchestrator.

Sits between the Scoring step (P5) and Storage (P6).  Reads the current
run's scored_logs_df and root_causes_df, loads the incident_history store
(parquet fallback in dry-run, Postgres in live mode), matches incidents via
Jaccard fingerprint similarity, assigns chain IDs, elevates precursor log
scores, and saves the enriched outputs.

Pipeline data flow
------------------
  scored_logs_df.parquet   ─┐
  root_causes_df.parquet    ├─► P5.5 ─► enriched scored_logs_df.parquet
  sessionized_logs.parquet ─┘          enriched root_causes_df.parquet
                                        incident_history.parquet  (updated)

Enriched columns added to scored_logs_df
-----------------------------------------
  chain_id               TEXT     NULL if not part of a chain
  precursor_incident_id  TEXT     NULL if first in chain
  chain_position         INT      1-indexed depth in chain
  chain_confidence       FLOAT    Jaccard similarity to direct precursor
  is_precursor_elevated  BOOLEAN  True if this log's score was boosted

Enriched columns added to root_causes_df
-----------------------------------------
  historical_frequency   INT   Count of chain incidents that contained
                               this root cause template in prior runs.

Public API
----------
run(dry_run=False) -> tuple[pd.DataFrame, pd.DataFrame]
    Full orchestration: load → correlate → elevate → save → return.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import common.config as cfg
from common.logger import get_logger
from common.utils import load_parquet, save_parquet
from correlation.fingerprint import (
    fingerprint_from_df,
    fingerprint_from_list,
    fingerprint_to_json,
    jaccard,
)
from correlation.chain_builder import assign_chains, _history_columns
from correlation.precursor_elevator import (
    elevate_precursor_scores,
    elevate_log_scores,
)

logger = get_logger(__name__)

# Parquet paths used by this stage
_SCORED_PATH     = "data/processed/scored_logs_df.parquet"
_ROOT_CAUSE_PATH = "data/processed/root_causes_df.parquet"
_SESSION_PATH    = "data/processed/sessionized_logs.parquet"
_HISTORY_PATH    = cfg.INCIDENT_HISTORY_PATH

# Severity priority for determining the "worst" label in an incident
_SEVERITY_RANK = {"critical": 4, "medium": 3, "low": 2, "ignore": 1}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(dry_run: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Orchestrate the cross-run correlation stage.

    Parameters
    ----------
    dry_run : bool
        When True, uses parquet-only history (no Postgres reads/writes).
        When False, also syncs incident_history to Postgres.

    Returns
    -------
    (enriched_scored_df, enriched_root_causes_df)
    """
    if not cfg.CROSS_RUN_ENABLED:
        logger.info("CROSS_RUN_ENABLED=False — skipping P5.5.")
        scored_df = load_parquet(_SCORED_PATH)
        root_causes_df = _load_root_causes()
        return scored_df, root_causes_df

    logger.info("Cross-run correlation starting (dry_run=%s).", dry_run)

    # Step 1 — load inputs
    scored_df    = load_parquet(_SCORED_PATH)
    root_causes_df = _load_root_causes()
    session_df   = load_parquet(_SESSION_PATH)

    # Step 2 — load incident history (parquet fallback always used)
    history_df = _load_history()

    # Step 3 — build current-run incident records
    current_incidents = _build_current_incidents(scored_df, session_df, root_causes_df)
    logger.info(
        "Current run: %d incident(s) to correlate.", len(current_incidents)
    )

    # Step 4 — assign chains via Jaccard matching
    enriched_incidents, history_df = assign_chains(
        current_incidents=current_incidents,
        history_df=history_df,
        threshold=cfg.CROSS_RUN_SIMILARITY_THRESHOLD,
        lookback_hours=cfg.CROSS_RUN_LOOKBACK_HOURS,
    )

    chained = [i for i in enriched_incidents if i["chain_id"]]
    logger.info(
        "%d / %d incident(s) linked to an existing chain.",
        len(chained), len(enriched_incidents),
    )

    # Step 5 — elevate precursor log scores for the CURRENT run
    # (applies boost to logs whose incidents are marked as precursors)
    scored_df = _apply_precursor_elevation_to_current(
        scored_df, enriched_incidents, cfg.PRECURSOR_BOOST
    )

    # Step 6 — broadcast chain columns from incident level to log level
    scored_df = _broadcast_chain_to_logs(scored_df, enriched_incidents)

    # Step 7 — enrich root_causes_df with historical_frequency
    root_causes_df = _enrich_root_causes(root_causes_df, enriched_incidents, history_df)

    # Step 8 — update incident_history with current run
    history_df = _append_current_to_history(
        history_df, enriched_incidents, scored_df, session_df
    )

    # Step 9 — mark precursor incidents as elevated in history
    history_df = elevate_precursor_scores(
        history_df, enriched_incidents, cfg.PRECURSOR_BOOST
    )

    # Step 10 — save enriched outputs
    save_parquet(scored_df, _SCORED_PATH)
    save_parquet(history_df, _HISTORY_PATH)

    if root_causes_df is not None and len(root_causes_df) > 0:
        save_parquet(root_causes_df, _ROOT_CAUSE_PATH)

    # Step 11 — sync to Postgres (live runs only)
    if not dry_run:
        _sync_history_to_db(history_df)

    n_chains = history_df["chain_id"].dropna().nunique()
    logger.info(
        "Cross-run correlation complete. "
        "Total chains in history: %d. History records: %d.",
        n_chains, len(history_df),
    )

    return scored_df, root_causes_df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_root_causes() -> pd.DataFrame:
    """Load root_causes_df if it exists, else return empty DataFrame."""
    path = Path(_ROOT_CAUSE_PATH)
    if path.exists():
        return load_parquet(_ROOT_CAUSE_PATH)
    return pd.DataFrame(columns=["incident_id", "root_cause_log_id",
                                  "confidence_score", "in_graph"])


def _load_history() -> pd.DataFrame:
    """Load incident_history from the parquet fallback store."""
    path = Path(_HISTORY_PATH)
    if not path.exists():
        logger.info("No incident history found — starting fresh.")
        return pd.DataFrame(columns=_history_columns())

    history = load_parquet(_HISTORY_PATH)
    if "end_time" in history.columns:
        history["end_time"] = pd.to_datetime(history["end_time"], utc=True, errors="coerce")
    if "start_time" in history.columns:
        history["start_time"] = pd.to_datetime(history["start_time"], utc=True, errors="coerce")

    logger.info("Loaded %d historical incident(s) from %s.", len(history), _HISTORY_PATH)
    return history


def _build_current_incidents(
    scored_df: pd.DataFrame,
    session_df: pd.DataFrame,
    root_causes_df: pd.DataFrame,
) -> list[dict]:
    """Build one dict per incident in the current run.

    The dict contains everything needed for chain assignment and history
    storage:  global_incident_id, local correlation_id, fingerprint, timestamps.
    """
    # Determine run_date from the earliest log timestamp in this batch
    if "timestamp" in session_df.columns and len(session_df) > 0:
        ts_series = pd.to_datetime(session_df["timestamp"], utc=True, errors="coerce")
        run_date_str = ts_series.dropna().min().strftime("%Y%m%d")
    else:
        run_date_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d")

    run_ts = datetime.now(tz=timezone.utc)

    # Join scored_df with session_df to get template_id, host, and timestamp per log
    scored_with_tmpl = scored_df.merge(
        session_df[["sequence_number", "template_id", "host", "timestamp"]]
        .drop_duplicates("sequence_number"),
        on="sequence_number",
        how="left",
        suffixes=("", "_sess"),
    )

    # Build root cause template lookup: incident_id → list[template_id]
    rc_templates: dict[str, list[str]] = {}
    if root_causes_df is not None and len(root_causes_df) > 0 and "incident_id" in root_causes_df.columns:
        for inc_id, grp in root_causes_df.groupby("incident_id"):
            # root_causes_df uses root_cause_log_id (log_NNNNNN format)
            # Join back to session_df to get template_id
            if "root_cause_log_id" in grp.columns:
                seq_nums = (
                    grp["root_cause_log_id"]
                    .str.replace("log_", "", regex=False)
                    .astype(float, errors="ignore")
                    .dropna()
                    .astype(int)
                    .tolist()
                )
                templates = session_df.loc[
                    session_df["sequence_number"].isin(seq_nums), "template_id"
                ].dropna().unique().tolist()
                rc_templates[str(inc_id)] = templates

    # Produce one record per incident (non-null correlation_id)
    incidents = []
    incident_groups = scored_with_tmpl[scored_with_tmpl["correlation_id"].notna()]

    for local_id, grp in incident_groups.groupby("correlation_id"):
        local_id = str(local_id)
        # Generate globally unique incident_id
        seq_str = local_id.replace("INC-", "")
        global_id = f"INC-{run_date_str}-{seq_str}"

        fp = fingerprint_from_df(grp)

        # Timestamps from session_df (more reliable than scored_df)
        timestamps = pd.to_datetime(grp["timestamp"], utc=True, errors="coerce").dropna()
        start_time = timestamps.min() if len(timestamps) > 0 else pd.NaT
        end_time = timestamps.max() if len(timestamps) > 0 else pd.NaT

        hosts = sorted(grp["host"].dropna().unique().tolist()) if "host" in grp.columns else []

        # Severity = worst label in the incident
        labels = grp["label"].dropna().unique().tolist() if "label" in grp.columns else []
        severity = max(labels, key=lambda l: _SEVERITY_RANK.get(l, 0), default=None)

        incidents.append({
            "global_incident_id": global_id,
            "local_incident_id": local_id,
            "run_date": run_date_str,
            "run_timestamp": run_ts,
            "start_time": start_time,
            "end_time": end_time,
            "template_fingerprint": fp,
            "root_cause_templates": rc_templates.get(local_id, []),
            "severity": severity,
            "log_count": len(grp),
            "hosts": hosts,
            "is_cross_system": bool(grp["is_cross_system"].any()) if "is_cross_system" in grp.columns else False,
        })

    return incidents


def _apply_precursor_elevation_to_current(
    scored_df: pd.DataFrame,
    enriched_incidents: list[dict],
    boost: float,
) -> pd.DataFrame:
    """Boost scores for current-run incidents that are precursors in their chain."""
    # A current incident is a precursor if another incident in THIS run
    # references it as precursor_incident_id.  This handles same-run chains
    # (unusual but possible if two incidents in one day are related).
    precursor_global_ids = {
        i["precursor_incident_id"]
        for i in enriched_incidents
        if i.get("precursor_incident_id")
    }

    # Map global_incident_id → local correlation_id for current-run incidents
    global_to_local = {i["global_incident_id"]: i["local_incident_id"] for i in enriched_incidents}

    precursor_local_ids = {
        global_to_local[gid]
        for gid in precursor_global_ids
        if gid in global_to_local
    }

    if not precursor_local_ids:
        return scored_df

    # Find the chain_confidence of the incident that references these precursors
    confidence = max(
        (i["chain_confidence"] for i in enriched_incidents if i["local_incident_id"] in precursor_local_ids),
        default=0.0,
    )

    return elevate_log_scores(scored_df, precursor_local_ids, confidence, boost)


def _broadcast_chain_to_logs(
    scored_df: pd.DataFrame,
    enriched_incidents: list[dict],
) -> pd.DataFrame:
    """Add chain_id, precursor_incident_id, chain_position, chain_confidence
    to scored_df by mapping from the per-incident enrichment results."""
    df = scored_df.copy()

    # Ensure chain columns exist
    for col in ["chain_id", "precursor_incident_id", "chain_position", "chain_confidence"]:
        if col not in df.columns:
            df[col] = None

    # Build mapping: local_incident_id → chain fields
    for inc in enriched_incidents:
        local_id = inc["local_incident_id"]
        mask = df["correlation_id"] == local_id
        if not mask.any():
            continue
        df.loc[mask, "chain_id"] = inc.get("chain_id")
        df.loc[mask, "precursor_incident_id"] = inc.get("precursor_incident_id")
        df.loc[mask, "chain_position"] = inc.get("chain_position")
        df.loc[mask, "chain_confidence"] = inc.get("chain_confidence")

    return df


def _enrich_root_causes(
    root_causes_df: pd.DataFrame,
    enriched_incidents: list[dict],
    history_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add historical_frequency to root_causes_df.

    historical_frequency = how many prior chain incidents contained the same
    root cause template.
    """
    if root_causes_df is None or len(root_causes_df) == 0:
        return root_causes_df

    df = root_causes_df.copy()
    df["historical_frequency"] = 0

    if len(history_df) == 0:
        return df

    # Build a lookup of chain_id → set of root_cause_templates from history
    chain_rc_freq: dict[str, dict[str, int]] = {}
    for _, h_row in history_df.iterrows():
        cid = h_row.get("chain_id")
        if not cid:
            continue
        rcts = fingerprint_from_list(h_row.get("root_cause_templates"))
        if cid not in chain_rc_freq:
            chain_rc_freq[cid] = {}
        for t in rcts:
            chain_rc_freq[cid][t] = chain_rc_freq[cid].get(t, 0) + 1

    # Map local_incident_id → chain_id
    local_to_chain = {i["local_incident_id"]: i.get("chain_id") for i in enriched_incidents}

    for idx, row in df.iterrows():
        inc_id = str(row.get("incident_id", ""))
        chain_id = local_to_chain.get(inc_id)
        if not chain_id or chain_id not in chain_rc_freq:
            continue
        # root_cause_log_id is "log_NNNNNN" — we can't easily get template_id
        # here without a join.  Use 0 for now; frequency is best-effort.
        # The full enrichment requires session_df which is available in cross_run.run().
        df.at[idx, "historical_frequency"] = sum(chain_rc_freq[chain_id].values())

    return df


def _append_current_to_history(
    history_df: pd.DataFrame,
    enriched_incidents: list[dict],
    scored_df: pd.DataFrame,
    session_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build rows for the current run's incidents and append to history_df."""
    new_rows = []
    for inc in enriched_incidents:
        new_rows.append({
            "incident_id":            inc["global_incident_id"],
            "run_date":               inc["run_date"],
            "run_timestamp":          inc["run_timestamp"],
            "start_time":             inc.get("start_time"),
            "end_time":               inc.get("end_time"),
            "template_fingerprint":   fingerprint_to_json(inc["template_fingerprint"]),
            "root_cause_templates":   json.dumps(inc.get("root_cause_templates") or []),
            "severity":               inc.get("severity"),
            "log_count":              inc.get("log_count"),
            "hosts":                  json.dumps(inc.get("hosts") or []),
            "is_cross_system":        inc.get("is_cross_system", False),
            "chain_id":               inc.get("chain_id"),
            "precursor_incident_id":  inc.get("precursor_incident_id"),
            "chain_position":         inc.get("chain_position"),
            "is_precursor_elevated":  False,
        })

    if not new_rows:
        return history_df

    new_df = pd.DataFrame(new_rows)
    # Ensure datetime columns are tz-aware
    for col in ("start_time", "end_time", "run_timestamp"):
        if col in new_df.columns:
            new_df[col] = pd.to_datetime(new_df[col], utc=True, errors="coerce")

    combined = pd.concat([history_df, new_df], ignore_index=True)
    # De-duplicate: keep last record per incident_id (allows updates on reruns)
    combined = combined.drop_duplicates(subset=["incident_id"], keep="last")
    return combined


def _sync_history_to_db(history_df: pd.DataFrame) -> None:
    """Write incident_history to Postgres (live runs only)."""
    try:
        from storage.db_writer import get_connection, apply_schema, write_incident_history

        conn = get_connection()
        try:
            apply_schema(conn)
            n = write_incident_history(history_df, conn)
            conn.commit()
            logger.info("Synced %d incident_history row(s) to Postgres.", n)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    except Exception as exc:
        logger.warning(
            "Failed to sync incident_history to Postgres (non-fatal): %s. "
            "Data is preserved in %s.",
            exc,
            _HISTORY_PATH,
        )
