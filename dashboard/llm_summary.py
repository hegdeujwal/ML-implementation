"""
dashboard/llm_summary.py
========================
Gemini-powered batch LLM summary generation and cache management.

Two public entry points:
  generate_all_summaries()  — called at end of pipeline.py, batches all new incidents
  regenerate_summary()      — called only from the dashboard Regenerate button
"""

from __future__ import annotations

import json
import logging
import os
import re

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy Gemini initialisation — avoids import errors when GEMINI_API_KEY is absent
# ---------------------------------------------------------------------------

_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            import google.generativeai as genai

            api_key = os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                logger.warning("GEMINI_API_KEY not set — LLM summaries will be unavailable.")
                return None

            genai.configure(api_key=api_key)
            _model = genai.GenerativeModel("gemini-2.5-flash")
        except Exception as exc:
            logger.warning("Failed to initialise Gemini: %s", exc)
            return None
    return _model


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_context(cid: str, scored_df: pd.DataFrame, root_causes_df: pd.DataFrame) -> dict:
    cluster = scored_df[scored_df["correlation_id"] == cid]
    ordered = cluster.sort_values("timestamp") if "timestamp" in cluster.columns else cluster

    templates = (
        ordered["template_id"].dropna().tolist()
        if "template_id" in ordered.columns
        else []
    )
    seq = " → ".join(templates[:10])
    if len(templates) > 10:
        seq += f" ... (+{len(templates) - 10} more)"

    rc_col = "incident_id" if "incident_id" in root_causes_df.columns else None
    rc = pd.DataFrame()
    if rc_col:
        rc = root_causes_df[root_causes_df[rc_col] == cid]

    rc_str = "none identified"
    if not rc.empty and "root_cause_log_id" in rc.columns:
        rc_str = ", ".join(
            f"{r['root_cause_log_id']} (conf: {r.get('confidence_score', 0):.2f})"
            for _, r in rc.iterrows()
        )

    hosts = (
        ", ".join(cluster["host"].dropna().unique())
        if "host" in cluster.columns
        else "unknown"
    )
    log_count = len(cluster)
    is_cross = bool(cluster["is_cross_system"].any()) if "is_cross_system" in cluster.columns else False

    duration_s = 0
    if "timestamp" in ordered.columns and len(ordered) > 1:
        try:
            duration_s = int(
                (ordered["timestamp"].iloc[-1] - ordered["timestamp"].iloc[0]).total_seconds()
            )
        except Exception:
            pass

    worst_label = "low"
    if "label" in cluster.columns:
        label_order = {"critical": 3, "medium": 2, "low": 1, "ignore": 0}
        worst_label = max(
            cluster["label"].dropna().tolist() or ["low"],
            key=lambda x: label_order.get(x, 0),
        )

    return {
        "correlation_id": cid,
        "host": hosts,
        "start_time": str(ordered["timestamp"].iloc[0]) if "timestamp" in ordered.columns and len(ordered) else "unknown",
        "duration_seconds": duration_s,
        "log_count": log_count,
        "is_cross_system": is_cross,
        "template_sequence": seq or "(no templates)",
        "root_causes": rc_str,
        "worst_label": worst_label,
    }


# ---------------------------------------------------------------------------
# Gemini batch call
# ---------------------------------------------------------------------------

def _call_gemini_batch(batch: list[dict]) -> list[dict]:
    model = _get_model()
    if model is None:
        return [
            {"correlation_id": ctx["correlation_id"], "summary_text": "Summary unavailable — API not configured."}
            for ctx in batch
        ]

    incidents_text = ""
    for ctx in batch:
        incidents_text += f"""
---
{ctx['correlation_id']}
Host: {ctx['host']} | Severity: {ctx['worst_label']} | Duration: {ctx['duration_seconds']}s | Logs: {ctx['log_count']}
Templates: {ctx['template_sequence']}
Root causes: {ctx['root_causes']}
Cross-system: {ctx['is_cross_system']}
"""

    prompt = f"""You are a network operations assistant for HPE CX network switches.
For each incident below, write a 3-5 sentence plain English summary for a
network engineer. Identify the failure pattern by name if recognisable (e.g. STP loop,
OSPF flap, BGP reconvergence, interface CRC errors, CPU spike, memory exhaustion).
No jargon. No bullet points. Be specific about the impact and likely root cause.

Return ONLY a valid JSON array — no markdown, no explanation, no preamble:
[{{"correlation_id": "INC-0001", "summary_text": "..."}}, ...]

Incidents:
{incidents_text}"""

    try:
        response = model.generate_content(prompt)
        text = re.sub(r"```json|```", "", response.text).strip()
        parsed = json.loads(text)
        # Validate structure
        if isinstance(parsed, list) and all("correlation_id" in r for r in parsed):
            return parsed
        raise ValueError("Unexpected JSON structure from Gemini")
    except Exception as exc:
        logger.warning("Batch parse failed (%s) — falling back to individual calls", exc)
        return _fallback_individual(batch)


def _fallback_individual(batch: list[dict]) -> list[dict]:
    """Fallback when batch JSON parse fails — calls API once per incident."""
    model = _get_model()
    results = []
    for ctx in batch:
        try:
            if model is None:
                raise RuntimeError("Model not available")
            prompt = (
                f"Summarise this HPE CX switch incident in 3-5 plain English sentences "
                f"for a network engineer. Identify failure patterns by name. No bullet points.\n\n"
                f"Incident {ctx['correlation_id']}:\n"
                f"  Templates: {ctx['template_sequence']}\n"
                f"  Root causes: {ctx['root_causes']}\n"
                f"  Duration: {ctx['duration_seconds']}s | Logs: {ctx['log_count']}"
            )
            response = model.generate_content(prompt)
            results.append(
                {
                    "correlation_id": ctx["correlation_id"],
                    "summary_text": response.text.strip(),
                }
            )
        except Exception:
            results.append(
                {
                    "correlation_id": ctx["correlation_id"],
                    "summary_text": "Summary unavailable.",
                }
            )
    return results


# ---------------------------------------------------------------------------
# Public entry point 1: pipeline batch generation
# ---------------------------------------------------------------------------

def generate_all_summaries(
    scored_df: pd.DataFrame,
    root_causes_df: pd.DataFrame,
    batch_size: int = 20,
) -> None:
    """
    Batch-generates LLM summaries for all new incidents in scored_df.
    Skips incidents that already have a cached summary in Postgres.
    Writes all new summaries via write_summaries_batch() — single transaction.
    batch_size=20 is safe for Gemini 2.5 Flash; do not exceed 30.
    """
    from data.db import get_summary, write_summaries_batch

    if "correlation_id" not in scored_df.columns:
        logger.warning("scored_df has no correlation_id column — skipping summary generation")
        return

    incident_ids = scored_df["correlation_id"].dropna().unique().tolist()
    if not incident_ids:
        logger.info("No incidents in scored_df — skipping summary generation")
        return

    uncached = [cid for cid in incident_ids if get_summary(cid) is None]
    if not uncached:
        logger.info("All incident summaries already cached — skipping Gemini calls")
        return

    logger.info(
        "Generating summaries for %d incidents in batches of %d",
        len(uncached),
        batch_size,
    )

    contexts = [_build_context(cid, scored_df, root_causes_df) for cid in uncached]

    all_summaries: list[dict] = []
    for i in range(0, len(contexts), batch_size):
        batch = contexts[i: i + batch_size]
        summaries = _call_gemini_batch(batch)
        all_summaries.extend(summaries)
        logger.info(
            "Batch %d: %d summaries generated", i // batch_size + 1, len(summaries)
        )

    write_summaries_batch(all_summaries)
    logger.info("Cached %d summaries to Postgres", len(all_summaries))


# ---------------------------------------------------------------------------
# Public entry point 2: dashboard Regenerate button
# ---------------------------------------------------------------------------

def regenerate_summary(correlation_id: str, incident_data: dict) -> str:
    """
    Bypass cache, call Gemini, write new text back to summaries table.
    Called only from the dashboard Regenerate button.
    """
    from data.db import write_summary

    model = _get_model()
    if model is None:
        return "Summary unavailable — GEMINI_API_KEY not configured. Set it in your .env file."

    try:
        prompt = (
            f"Summarise this HPE CX switch incident in 3-5 plain English sentences "
            f"for a network engineer. Identify failure patterns by name. No bullet points.\n\n"
            f"Incident {incident_data.get('correlation_id', correlation_id)}:\n"
            f"  Templates: {incident_data.get('template_sequence', '(unknown)')}\n"
            f"  Root causes: {incident_data.get('root_causes', 'none')}"
        )
        response = model.generate_content(prompt)
        text = response.text.strip()
        write_summary(correlation_id, text)
        return text
    except Exception as exc:
        logger.warning("regenerate_summary failed: %s", exc)
        return "Summary unavailable — API unreachable. Click Regenerate to try again."
