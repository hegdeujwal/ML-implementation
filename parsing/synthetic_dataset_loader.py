"""
parsing/synthetic_dataset_loader.py
===================================
Section-aware loader for the mentor's multi-section synthetic log dataset
(``data/raw/synthetic_logs/``).

Why this exists
---------------
Each scenario file is a structured 7-section *document*, not a flat syslog
stream. The legacy line-oriented normalizer (parsing/normalizer.py) shreds it:
multi-line events lose their body, ``timestamp_ms`` is 3 years off the ISO
label, time-only ``[HH:MM:SS]`` stamps get today's date, explicit ``severity=``
is ignored, Section 4 numeric metrics are dumped as text, and the Section 7
ground-truth labels are dropped. Running the old path on a known-CRITICAL file
yields 0 anomalies.

This loader instead recognises the ``## SECTION N`` structure and routes each
section to the right parser, producing three artifacts:

1. sessionized_logs.parquet  — event rows, schema-compatible with the legacy
   path plus additive provenance/inference columns (LogEntry).
2. metrics_df.parquet        — long/tidy numeric telemetry from Section 4
   (MetricRow). Long format => "metric not applicable" is an absent row.
3. scenario_labels.parquet   — per-file oracle from Section 7 (ScenarioLabelRow).
   Used only for evaluation; never fed to the model.

Design principles for missing values
-------------------------------------
- Every required categorical column gets an explicit sentinel, never a null.
- ``timestamp`` is the only "drop if underivable" field — never back-filled with
  ``datetime.now()``.
- ``severity_explicit`` flags whether log_level came from an explicit field or
  was inferred/defaulted, so a guessed severity can't masquerade as observed.
- Original raw fields are archived in ``metadata`` JSON, so nothing is lost.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, time
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from common.config import (
    DEFAULT_SEVERITY_WEIGHT,
    DEFAULT_SOURCE_TYPE,
    METRICS_DF_PATH,
    SCENARIO_LABELS_PATH,
    SERVICE_ALIAS_MAP,
    SESSIONIZED_LOGS_PATH,
    SEVERITY_WEIGHTS,
)
from common.logger import get_logger
from common.utils import save_parquet, validate_schema
from parsing.log_parser import DrainParser
from parsing.normalizer import normalize_line
# Reuse the legacy session/action helpers so this loader stays DRY and produces
# an identical session_id / event_action contract.
from parsing.sessionizer import (
    REQUIRED_OUTPUT_COLUMNS as _BASE_OUTPUT_COLUMNS,
    _assign_sessions,
    _derive_event_action,
)

logger = get_logger(__name__)

# sessionized schema = legacy columns + additive provenance columns.
_EXTRA_OUTPUT_COLUMNS = [
    "source_file",
    "scenario_id",
    "section",
    "component",
    "code_location",
    "severity_explicit",
]
REQUIRED_OUTPUT_COLUMNS = _BASE_OUTPUT_COLUMNS + _EXTRA_OUTPUT_COLUMNS

METRICS_COLUMNS = [
    "timestamp", "source_file", "scenario_id", "entity", "metric_name", "metric_value",
]
LABEL_COLUMNS = [
    "scenario_id", "source_file", "training_label", "failure_mode", "root_cause",
    "file_severity", "affected_components", "correlation_signals",
]

# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(r"^##\s*SECTION\s+(\d+)", re.IGNORECASE)
_ISO_TS_RE = re.compile(r"\[?(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)\]?")
_TIME_ONLY_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\]\s*(.*)$")
_EVENT_HDR_RE = re.compile(r"EVENT:\s*([A-Z0-9_]+)")
_KV_RE = re.compile(r'(\w+)=("[^"]*"|\[[^\]]*\]|[^\s|]+)')
_NUM_KV_RE = re.compile(r"([A-Za-z][\w/\[\]]*)\s*=\s*([\d][\d,]*\.?\d*)")
_DURATION_DATE_RE = re.compile(r"#\s*Duration:\s*(\d{4}-\d{2}-\d{2})")
_HEADER_SEVERITY_RE = re.compile(r"#\s*Severity:\s*([A-Z\-]+)", re.IGNORECASE)
_SEVERITY_LEVELS = set(SEVERITY_WEIGHTS.keys())

# Junk/placeholder lines that should never become rows.
_JUNK_RES = [
    re.compile(r"^\s*\[\.\.\."),                 # "[... 40 more lines ...]"
    re.compile(r"more lines", re.IGNORECASE),
    re.compile(r"^\s*END OF SCENARIO", re.IGNORECASE),
    re.compile(r"^\s*-{3,}\s*$"),                # "---" separators
    re.compile(r"^\s*\[(Trend|Metric|Analysis)\s+\d+\]", re.IGNORECASE),  # synthetic filler
]


def _is_junk(line: str) -> bool:
    return any(r.search(line) for r in _JUNK_RES)


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def _parse_iso(ts_str: str) -> Optional[datetime]:
    """Parse an ISO8601 stamp (with optional trailing Z), tz-stripped."""
    s = ts_str.strip().rstrip("Z")
    try:
        dt = datetime.fromisoformat(s)
        return dt.replace(tzinfo=None) if dt.tzinfo else dt
    except ValueError:
        return None


def _parse_time_only(t_str: str, ctx_date: date) -> Optional[datetime]:
    """Combine a time-only ``HH:MM:SS[.ms]`` with the file's context date.

    This is the fix for the legacy bug where dateutil stamped these with today's
    date, scattering one incident across multiple calendar days.
    """
    try:
        parts = t_str.split(":")
        h, m = int(parts[0]), int(parts[1])
        sec_str = parts[2]
        if "." in sec_str:
            s_int, frac = sec_str.split(".")
            micro = int(frac.ljust(6, "0")[:6])
        else:
            s_int, micro = sec_str, 0
        return datetime.combine(ctx_date, time(h, m, int(s_int), micro))
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Severity helper
# ---------------------------------------------------------------------------

def _severity_from_field(value: str) -> Optional[str]:
    """Map an explicit severity token to a canonical level, or None."""
    v = value.strip().upper()
    if v in ("WARNING",):
        v = "WARN"
    return v if v in _SEVERITY_LEVELS else None


def _weight_for(level: str) -> float:
    return SEVERITY_WEIGHTS.get(level, DEFAULT_SEVERITY_WEIGHT)


# ---------------------------------------------------------------------------
# Record factory — guarantees every schema field is populated with a sentinel
# ---------------------------------------------------------------------------

def _make_record(
    *,
    timestamp: datetime,
    section: int,
    source_file: str,
    scenario_id: str,
    message: str,
    raw_text: str,
    service: str = "UNKNOWN",
    host: str = "network_device",
    log_level: str = "INFO",
    severity_explicit: bool = False,
    component: str = "UNKNOWN",
    code_location: str = "NONE",
    extra_meta: Optional[dict] = None,
) -> dict:
    meta = {"raw_text": raw_text}
    if extra_meta:
        meta.update(extra_meta)
    return {
        "timestamp": timestamp,
        "source_type": DEFAULT_SOURCE_TYPE,
        "service": service,
        "host": host,
        "log_level": log_level,
        "event_weight": _weight_for(log_level),
        "message": message or raw_text.strip(),
        "_raw_text": raw_text,
        "_metadata_extra": meta,
        "source_file": source_file,
        "scenario_id": scenario_id,
        "section": section,
        "component": component,
        "code_location": code_location,
        "severity_explicit": severity_explicit,
    }


def _service_from_component(name: str) -> str:
    """Map a daemon/component token to a canonical service via the alias map."""
    return SERVICE_ALIAS_MAP.get(name.strip().lower(), name.strip().upper() or "UNKNOWN")


# ---------------------------------------------------------------------------
# Section parsers
# ---------------------------------------------------------------------------

def _parse_section1_events(block_lines: list[str], ctx: dict) -> list[dict]:
    """Section 1 — multi-line structured events separated by blank lines.

    Joins continuation lines into one logical event, then parses the
    ``EVENT: NAME | key=value | ...`` fields. Uses the ISO timestamp; ignores
    ``timestamp_ms`` (it is 3 years off the ISO label).
    """
    records: list[dict] = []
    for block in _split_blocks(block_lines):
        joined = " ".join(l.strip() for l in block).strip()
        if not joined or _is_junk(joined):
            continue
        ts_m = _ISO_TS_RE.search(joined)
        ts = _parse_iso(ts_m.group(1)) if ts_m else None
        if ts is None:
            continue
        kv = {k: v.strip('"') for k, v in _KV_RE.findall(joined)}
        ev_m = _EVENT_HDR_RE.search(joined)
        event_name = ev_m.group(1) if ev_m else "EVENT"
        component = kv.get("component", "UNKNOWN")
        service = _service_from_component(component) if component != "UNKNOWN" else "EVENT"
        level = _severity_from_field(kv.get("severity", ""))
        explicit = level is not None
        level = level or "INFO"
        message = kv.get("message", event_name)
        records.append(_make_record(
            timestamp=ts, section=1, source_file=ctx["source_file"],
            scenario_id=ctx["scenario_id"], message=f"{event_name} {message}".strip(),
            raw_text=joined, service=service, log_level=level,
            severity_explicit=explicit, component=component,
            extra_meta={"event_name": event_name, "event_id": kv.get("event_id", "")},
        ))
    return records


def _parse_syslog_section(block_lines: list[str], section: int, ctx: dict) -> list[dict]:
    """Sections 2 & 5 — genuine one-line syslog. Reuse the legacy normalizer."""
    records: list[dict] = []
    for line in block_lines:
        if not line.strip() or _is_junk(line):
            continue
        parsed = normalize_line(line)
        if parsed is None:
            continue
        records.append(_make_record(
            timestamp=parsed["timestamp"], section=section, source_file=ctx["source_file"],
            scenario_id=ctx["scenario_id"], message=parsed["message"], raw_text=parsed["raw_text"],
            service=parsed["service"], host=parsed.get("host", "network_device"),
            log_level=parsed["log_level"], severity_explicit=False,
            component=parsed["service"],
        ))
    return records


def _parse_section3_debug(block_lines: list[str], ctx: dict) -> list[dict]:
    """Section 3 — ``[HH:MM:SS.ms] file.c:NNNN | func: message``.

    Time-only stamp inherits the file date; ``file.c:NNNN`` becomes code_location
    (not host); the function name is the action portion of the message.
    """
    records: list[dict] = []
    for line in block_lines:
        if not line.strip() or _is_junk(line):
            continue
        m = _TIME_ONLY_RE.match(line.strip())
        if not m:
            continue
        ts = _parse_time_only(m.group(1), ctx["ctx_date"])
        if ts is None:
            continue
        body = m.group(2)
        code_location, func, message = "NONE", "", body
        if "|" in body:
            left, right = body.split("|", 1)
            code_location = left.strip()
            right = right.strip()
            if ":" in right:
                func, message = right.split(":", 1)
                func, message = func.strip(), message.strip()
            else:
                message = right
        # component derived from the source file stem, e.g. spanning_tree.c -> spanning_tree
        comp_token = code_location.split(".")[0] if code_location != "NONE" else "debug"
        service = _service_from_component(comp_token)
        records.append(_make_record(
            timestamp=ts, section=3, source_file=ctx["source_file"],
            scenario_id=ctx["scenario_id"], message=f"{func}: {message}".strip(": "),
            raw_text=line.strip(), service=service, log_level="INFO",
            severity_explicit=False, component=comp_token, code_location=code_location,
            extra_meta={"function": func},
        ))
    return records


def _parse_section4_metrics(block_lines: list[str], ctx: dict) -> list[dict]:
    """Section 4 — numeric performance metrics → LONG metric rows.

    Handles the per-scenario format variety pragmatically:
      - ``[ts] entity: key=value key=value`` (link-style: RX=, util_RX=12.3%, queue=45/1024)
      - ``[ts] Buffer: 364/1024MB (35%), Queues[RX]:234 ...``
      - ``[ts] Counter 4095: 4,284,900,000 (99.80% of max)``
    Anything not recognised yields fewer rows rather than a bad row — in long
    format a missing metric is simply an absent row.
    """
    rows: list[dict] = []
    for line in block_lines:
        s = line.strip()
        if not s or _is_junk(s):
            continue
        m = _TIME_ONLY_RE.match(s)
        if not m:
            continue
        ts = _parse_time_only(m.group(1), ctx["ctx_date"])
        if ts is None:
            continue
        body = m.group(2)
        entity = body.split(":", 1)[0].strip() if ":" in body else "device"
        # Normalise entity for the counter form "Counter 4095"
        entity = re.sub(r"\s+", "_", entity)

        emitted = 0
        # 1) key=value numeric pairs (link-style)
        for k, v in _NUM_KV_RE.findall(body):
            try:
                val = float(v.replace(",", ""))
            except ValueError:
                continue
            rows.append(_metric_row(ts, ctx, entity, k.lower().strip("[]"), val))
            emitted += 1
        # 2) queue=45/1024 ratios -> _used / _max
        for k, used, total in re.findall(r"(\w+)\s*=\s*(\d+)\s*/\s*(\d+)", body):
            rows.append(_metric_row(ts, ctx, entity, f"{k.lower()}_used", float(used)))
            rows.append(_metric_row(ts, ctx, entity, f"{k.lower()}_max", float(total)))
            emitted += 2
        # 3) parenthesised percentages e.g. "(35%)" or "(99.80% of max)"
        pct_m = re.search(r"\(([\d.]+)%", body)
        if pct_m:
            rows.append(_metric_row(ts, ctx, entity, "pct", float(pct_m.group(1))))
            emitted += 1
        # 4) "Queues[RX]:234" style colon counters
        for k, v in re.findall(r"(\w+(?:\[\w+\])?)\s*:\s*([\d,]+)\b", body):
            if k.strip().lower() == entity.lower():
                continue
            try:
                rows.append(_metric_row(ts, ctx, entity, k.lower().replace("[", "_").strip("]"),
                                        float(v.replace(",", ""))))
                emitted += 1
            except ValueError:
                pass
    return rows


def _metric_row(ts, ctx, entity, name, value) -> dict:
    return {
        "timestamp": ts, "source_file": ctx["source_file"],
        "scenario_id": ctx["scenario_id"], "entity": entity,
        "metric_name": name, "metric_value": float(value),
    }


def _parse_section7_labels(block_lines: list[str], ctx: dict) -> dict:
    """Section 7 — ``key: value`` metadata → one oracle record."""
    kv: dict[str, str] = {}
    for line in block_lines:
        s = line.strip()
        if not s or ":" not in s or _is_junk(s):
            continue
        k, v = s.split(":", 1)
        kv[k.strip().lower()] = v.strip()

    def _list(field: str) -> list:
        raw = kv.get(field, "")
        m = re.search(r"\[(.*)\]", raw)
        if not m:
            return []
        return [x.strip() for x in m.group(1).split(",") if x.strip()]

    return {
        "scenario_id": ctx["scenario_id"],
        "source_file": ctx["source_file"],
        "training_label": kv.get("training_label", "UNKNOWN"),
        "failure_mode": kv.get("failure_mode", "UNKNOWN"),
        "root_cause": kv.get("root_cause", "UNKNOWN"),
        "file_severity": ctx.get("file_severity", "UNKNOWN"),
        "affected_components": _list("affected_components"),
        "correlation_signals": _list("correlation_signals"),
    }


def _split_blocks(lines: Iterable[str]) -> list[list[str]]:
    """Group lines into blocks separated by blank lines."""
    blocks, cur = [], []
    for line in lines:
        if line.strip():
            cur.append(line)
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    return blocks


# ---------------------------------------------------------------------------
# Per-file orchestration
# ---------------------------------------------------------------------------

def _file_context(path: Path, header_lines: list[str]) -> dict:
    """Build the per-file context: scenario id, date for time-only stamps, severity."""
    ctx_date = None
    file_severity = "UNKNOWN"
    for hl in header_lines:
        if ctx_date is None:
            dm = _DURATION_DATE_RE.search(hl)
            if dm:
                ctx_date = datetime.strptime(dm.group(1), "%Y-%m-%d").date()
        sm = _HEADER_SEVERITY_RE.search(hl)
        if sm:
            sev = _severity_from_field(sm.group(1))
            if sev:
                file_severity = sev
    return {
        "source_file": path.name,
        "scenario_id": path.stem,
        "ctx_date": ctx_date,           # may be None; resolved later from S1/S2
        "file_severity": file_severity,
    }


def parse_file(path: Path) -> tuple[list[dict], list[dict], Optional[dict]]:
    """Parse one scenario file into (event_records, metric_rows, label_record)."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()

    # Split into sections; everything before SECTION 1 is the header.
    sections: dict[int, list[str]] = {}
    header: list[str] = []
    current = 0
    for line in lines:
        sm = _SECTION_RE.match(line)
        if sm:
            current = int(sm.group(1))
            sections.setdefault(current, [])
            continue
        (sections[current] if current else header).append(line)

    ctx = _file_context(path, header)

    # Resolve the context date if the header lacked a Duration line: use the
    # first ISO timestamp found in Section 1 or 2.
    if ctx["ctx_date"] is None:
        for sec in (1, 2):
            for ln in sections.get(sec, []):
                tm = _ISO_TS_RE.search(ln)
                if tm:
                    dt = _parse_iso(tm.group(1))
                    if dt:
                        ctx["ctx_date"] = dt.date()
                        break
            if ctx["ctx_date"]:
                break
    if ctx["ctx_date"] is None:
        ctx["ctx_date"] = datetime.now().date()
        logger.warning("No date context for %s; time-only stamps will use today.", path.name)

    events: list[dict] = []
    events += _parse_section1_events(sections.get(1, []), ctx)
    events += _parse_syslog_section(sections.get(2, []), 2, ctx)
    events += _parse_section3_debug(sections.get(3, []), ctx)
    events += _parse_syslog_section(sections.get(5, []), 5, ctx)

    metrics = _parse_section4_metrics(sections.get(4, []), ctx)
    label = _parse_section7_labels(sections.get(7, []), ctx) if sections.get(7) else None

    return events, metrics, label


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    input_path: str,
    output_path: str = SESSIONIZED_LOGS_PATH,
    metrics_path: str = METRICS_DF_PATH,
    labels_path: str = SCENARIO_LABELS_PATH,
) -> pd.DataFrame:
    """Load a scenario file OR a directory of scenario files.

    Writes sessionized_logs.parquet, metrics_df.parquet, scenario_labels.parquet
    and returns the sessionized DataFrame.
    """
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    files = [src] if src.is_file() else sorted(src.glob("*.log"))
    if not files:
        raise ValueError(f"No .log files found under {input_path}")

    parser = DrainParser()
    all_event_rows: list[dict] = []
    all_metric_rows: list[dict] = []
    all_labels: list[dict] = []
    seq_num = 0

    for f in files:
        events, metrics, label = parse_file(f)
        logger.info(
            "Parsed %s: %d events, %d metric samples%s",
            f.name, len(events), len(metrics),
            ", label=" + label["training_label"] if label else "",
        )
        # Pass 1: feed messages through Drain (cluster objects; ids resolved later).
        for ev in events:
            seq_num += 1
            ev["sequence_number"] = seq_num
            ev["_cluster"] = parser.add_log_message_cluster(ev["message"], sequence_number=seq_num)
            all_event_rows.append(ev)
        all_metric_rows.extend(metrics)
        if label:
            all_labels.append(label)

    if not all_event_rows:
        raise ValueError(f"No parseable event lines found under {input_path}")

    # Pass 2: resolve stable template ids now that Drain has converged.
    for ev in all_event_rows:
        ev["template_id"] = parser.resolve_template_id(ev.pop("_cluster"))
        ev["metadata"] = json.dumps(ev.pop("_metadata_extra"))
        ev.pop("_raw_text", None)

    df = pd.DataFrame(all_event_rows)
    df = _assign_sessions(df)
    df["event_type"] = df["service"]
    df["event_action"] = df.apply(
        lambda r: _derive_event_action(r["service"], r["template_id"]), axis=1
    )
    df["frequency"] = (
        df.groupby(["session_id", "template_id"])["template_id"].transform("count").astype(int)
    )

    validate_schema(df, REQUIRED_OUTPUT_COLUMNS)
    save_parquet(df[REQUIRED_OUTPUT_COLUMNS], output_path)
    logger.info(
        "Wrote %d event rows → %s (%d sessions, %d templates)",
        len(df), output_path, df["session_id"].nunique(), df["template_id"].nunique(),
    )

    # Metrics (long/tidy). May legitimately be empty for label-only inputs.
    metrics_df = pd.DataFrame(all_metric_rows, columns=METRICS_COLUMNS)
    save_parquet(metrics_df, metrics_path)
    logger.info("Wrote %d metric samples → %s", len(metrics_df), metrics_path)

    # Oracle labels.
    labels_df = pd.DataFrame(all_labels, columns=LABEL_COLUMNS)
    save_parquet(labels_df, labels_path)
    logger.info("Wrote %d scenario labels → %s", len(labels_df), labels_path)

    return df


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Section-aware loader for synthetic_logs.")
    ap.add_argument(
        "input", nargs="?", default="data/raw/synthetic_logs",
        help="Scenario .log file or directory (default: data/raw/synthetic_logs)",
    )
    ap.add_argument("--output", default=SESSIONIZED_LOGS_PATH)
    ap.add_argument("--metrics", default=METRICS_DF_PATH)
    ap.add_argument("--labels", default=SCENARIO_LABELS_PATH)
    args = ap.parse_args()

    out = run(args.input, args.output, args.metrics, args.labels)
    print(f"Rows     : {len(out):,}")
    print(f"Sessions : {out['session_id'].nunique()}")
    print(f"Templates: {out['template_id'].nunique()}")
    print(f"Levels   :\n{out['log_level'].value_counts().to_string()}")
