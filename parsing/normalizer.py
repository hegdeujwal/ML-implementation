"""
parsing/normalizer.py
=====================
Pre-processes raw syslog lines into structured dicts before Drain parsing.

Supported formats
-----------------
1. RFC 3164 with PRI:   <191>Mar 12 10:00:00 hostname process: message
2. ISO 8601:            2024-01-15 10:23:45 hostname process: message
3. BSD syslog (bare):   Jan 15 10:23:45 hostname process: message

Input:  A single raw log line (str).
Output: dict with keys {raw_text, timestamp, host, service, log_level, message}
        or None if the line is blank or a comment.

Service extraction
------------------
The token immediately before the first ':' after the hostname is treated as the
process/service name.  PID suffixes like "[1234]" are stripped.  Names are then
normalised via SERVICE_ALIAS_MAP in common/config.py — daemon names (sshd, cfgd)
are mapped to canonical subsystem labels (SYSTEM, CONFIG); everything else is
upper-cased and used as-is.

Severity inference
------------------
The PRI facility/severity code is decoded first.  For datasets where all
messages share a single facility the code alone is insufficient; keyword-based
overrides are applied afterwards to map process-specific language to a
meaningful level.

Known limitations
-----------------
- Year defaults to the current year for BSD syslog timestamps (no year field).
- Sub-second timestamps are truncated to second precision.
- Multi-line log messages are treated as standalone entries.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from common.config import SERVICE_ALIAS_MAP

# ---------------------------------------------------------------------------
# PRI code handling (RFC 3164 / 5424)
# ---------------------------------------------------------------------------

_PRI_RE = re.compile(r"^<(\d{1,3})>")

_PRI_SEVERITY_MAP = {
    0: "CRITICAL",   # EMERG
    1: "CRITICAL",   # ALERT
    2: "CRITICAL",   # CRIT
    3: "ERROR",      # ERR
    4: "WARN",       # WARNING
    5: "INFO",       # NOTICE
    6: "INFO",       # INFO
    7: "INFO",       # DEBUG
}

# ---------------------------------------------------------------------------
# Timestamp patterns
# ---------------------------------------------------------------------------

_TS_PATTERNS = [
    re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})\s+(?P<rest>.+)$"),
    re.compile(r"^(?P<ts>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+(?P<rest>.+)$"),
]

def _parse_timestamp(ts_str: str) -> Optional[datetime]:
    ts_str = ts_str.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            pass
    try:
        dt = datetime.strptime(ts_str, "%b %d %H:%M:%S")
        return dt.replace(year=datetime.now().year)
    except ValueError:
        pass
    return None

# ---------------------------------------------------------------------------
# Service extraction
# ---------------------------------------------------------------------------

_PID_SUFFIX_RE = re.compile(r"\[\d+\]$")  # strips "[1234]" from "sshd[1234]"


def _extract_service(remaining: str) -> tuple[str, str]:
    """Split 'process: message text' into (service, message).

    Returns ("UNKNOWN", remaining) when no ':' separator is found.
    """
    if ":" not in remaining:
        return "UNKNOWN", remaining

    raw_service, message = remaining.split(":", 1)
    raw_service = raw_service.strip()
    # Take only the last whitespace-separated token (handles "Jan 15 hostname sshd[1]:")
    raw_service = raw_service.split()[-1] if raw_service else "UNKNOWN"
    raw_service = _PID_SUFFIX_RE.sub("", raw_service).strip() or "UNKNOWN"

    service = SERVICE_ALIAS_MAP.get(raw_service.lower(), raw_service.upper())
    return service, message.strip()

# ---------------------------------------------------------------------------
# Severity — inline token detection (fallback when PRI is uninformative)
# ---------------------------------------------------------------------------

_SEVERITY_TOKEN_RE = re.compile(
    r"\b(CRITICAL|CRIT|ERROR|ERR|WARNING|WARN|NOTICE|DEBUG|INFO)\b",
    re.IGNORECASE,
)
_SEVERITY_TOKEN_MAP = {
    "CRITICAL": "CRITICAL", "CRIT": "CRITICAL",
    "ERROR": "ERROR",       "ERR":  "ERROR",
    "WARNING": "WARN",      "WARN": "WARN",
    "NOTICE": "INFO",       "INFO": "INFO",  "DEBUG": "INFO",
}

_KEYWORD_OVERRIDES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"changed state from \S+ to DOWN",            re.I), "CRITICAL"),
    (re.compile(r"adjacency.*lost|adjacency.*down",           re.I), "CRITICAL"),
    (re.compile(r"port scan detected",                        re.I), "CRITICAL"),
    (re.compile(r"temperature.*threshold|thermal.*exceed",    re.I), "CRITICAL"),
    (re.compile(r"link.*flap|flap.*detect",                   re.I), "CRITICAL"),
    (re.compile(r"changed state to down",                     re.I), "ERROR"),
    (re.compile(r"authentication failure",                    re.I), "ERROR"),
    (re.compile(r"ACL error",                                 re.I), "ERROR"),
    (re.compile(r"session.*reset|reset.*session",             re.I), "ERROR"),
    (re.compile(r"disk.*latency|write.*latency",              re.I), "ERROR"),
    (re.compile(r"login failed|login failure",                re.I), "ERROR"),
    (re.compile(r"Drop DHCP|untrusted port",                  re.I), "WARN"),
    (re.compile(r"MAC .{5,30} blocked",                       re.I), "WARN"),
    (re.compile(r"cpu.*utilization.*exceed|cpu.*threshold",   re.I), "WARN"),
    (re.compile(r"memory.*usage|mem.*threshold",              re.I), "WARN"),
    (re.compile(r"packet drop.*high|drop rate.*high",         re.I), "WARN"),
    (re.compile(r"changed state to up",                       re.I), "INFO"),
    (re.compile(r"adjacency.*established|session established",re.I), "INFO"),
    (re.compile(r"configuration saved|config.*saved",         re.I), "INFO"),
    (re.compile(r"login successful|login success",            re.I), "INFO"),
    (re.compile(r"binding added|vlan.*added|vlan.*removed",   re.I), "INFO"),
]

def _infer_severity(message: str, pri_severity: Optional[str]) -> str:
    for pattern, level in _KEYWORD_OVERRIDES:
        if pattern.search(message):
            return level
    m = _SEVERITY_TOKEN_RE.search(message)
    if m:
        return _SEVERITY_TOKEN_MAP.get(m.group(1).upper(), "INFO")
    return pri_severity or "INFO"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_line(line: str) -> Optional[dict]:
    """Parse a single syslog line into a structured dict.

    Args:
        line: Raw log line (RFC 3164 with PRI, ISO 8601, or bare BSD syslog).

    Returns:
        dict with keys: raw_text, timestamp, host, service, log_level, message
        or None if the line is blank or a comment.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    pri_severity: Optional[str] = None
    m = _PRI_RE.match(line)
    if m:
        pri_val = int(m.group(1))
        pri_severity = _PRI_SEVERITY_MAP.get(pri_val % 8, "INFO")
        line_body = line[m.end():]
    else:
        line_body = line

    ts_dt: Optional[datetime] = None
    rest = line_body
    for pattern in _TS_PATTERNS:
        tm = pattern.match(line_body)
        if tm:
            ts_dt = _parse_timestamp(tm.group("ts"))
            if ts_dt:
                rest = tm.group("rest").strip()
                break

    if ts_dt is None:
        ts_dt = datetime.utcnow()

    # First token after timestamp is the hostname
    parts = rest.split(None, 1)
    host = parts[0] if parts else "unknown"
    remaining = parts[1].strip() if len(parts) > 1 else ""

    service, message_raw = _extract_service(remaining)

    # Strip leading severity token before passing to Drain (keeps templates clean)
    message = _SEVERITY_TOKEN_RE.sub("", message_raw, count=1).strip().lstrip(":").strip()
    message = message or message_raw

    log_level = _infer_severity(message_raw, pri_severity)

    return {
        "raw_text": line,
        "timestamp": ts_dt,
        "host": host,
        "service": service,
        "log_level": log_level,
        "message": message,
    }
