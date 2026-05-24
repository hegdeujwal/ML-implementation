"""
parsing/normalizer.py
=====================
Pre-processes raw syslog lines into structured dicts before Drain parsing.

Supported formats
-----------------
1. RFC 3164 with PRI:   <191>Mar 12 10:00:00 hostname process: message
2. ISO 8601:            2024-01-15 10:23:45 hostname severity: message
3. BSD syslog (bare):   Jan 15 10:23:45 hostname process: message

Input:  A single raw log line (str).
Output: dict with keys {raw_text, timestamp, source, severity, message}
        or None if the line is blank or a comment.

Severity inference
------------------
The PRI facility/severity code is decoded first.  For datasets where all
messages share a single facility (e.g. local7.notice/info/debug) the code
alone is insufficient; keyword-based overrides are applied afterwards to
map process-specific language (e.g. "changed state to down" → ERROR,
"port scan detected" → CRITICAL) to a meaningful level.

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

# ---------------------------------------------------------------------------
# PRI code handling (RFC 3164 / 5424)
# ---------------------------------------------------------------------------

_PRI_RE = re.compile(r"^<(\d{1,3})>")

# syslog severity (PRI % 8) → our level
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
    # ISO 8601: 2024-01-15T10:23:45 or 2024-01-15 10:23:45
    re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})\s+(?P<rest>.+)$"),
    # BSD syslog: Mar 12 10:00:00  (no year)
    re.compile(r"^(?P<ts>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+(?P<rest>.+)$"),
]

def _parse_timestamp(ts_str: str) -> Optional[datetime]:
    ts_str = ts_str.strip()
    # ISO
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            pass
    # BSD syslog — no year; inject current year
    try:
        dt = datetime.strptime(ts_str, "%b %d %H:%M:%S")
        return dt.replace(year=datetime.now().year)
    except ValueError:
        pass
    return None

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

# Keyword patterns applied to the full message text to infer severity when
# PRI and inline tokens both resolve to INFO.
_KEYWORD_OVERRIDES: list[tuple[re.Pattern, str]] = [
    # CRITICAL conditions
    (re.compile(r"changed state from \S+ to DOWN",            re.I), "CRITICAL"),
    (re.compile(r"adjacency.*lost|adjacency.*down",           re.I), "CRITICAL"),
    (re.compile(r"port scan detected",                        re.I), "CRITICAL"),
    (re.compile(r"temperature.*threshold|thermal.*exceed",    re.I), "CRITICAL"),
    (re.compile(r"link.*flap|flap.*detect",                   re.I), "CRITICAL"),
    # ERROR conditions
    (re.compile(r"changed state to down",                     re.I), "ERROR"),
    (re.compile(r"authentication failure",                    re.I), "ERROR"),
    (re.compile(r"ACL error",                                 re.I), "ERROR"),
    (re.compile(r"session.*reset|reset.*session",             re.I), "ERROR"),
    (re.compile(r"disk.*latency|write.*latency",              re.I), "ERROR"),
    (re.compile(r"login failed|login failure",                re.I), "ERROR"),
    # WARN conditions
    (re.compile(r"Drop DHCP|untrusted port",                  re.I), "WARN"),
    (re.compile(r"MAC .{5,30} blocked",                       re.I), "WARN"),
    (re.compile(r"cpu.*utilization.*exceed|cpu.*threshold",   re.I), "WARN"),
    (re.compile(r"memory.*usage|mem.*threshold",              re.I), "WARN"),
    (re.compile(r"packet drop.*high|drop rate.*high",         re.I), "WARN"),
    # INFO — explicit positive / informational
    (re.compile(r"changed state to up",                       re.I), "INFO"),
    (re.compile(r"adjacency.*established|session established",re.I), "INFO"),
    (re.compile(r"configuration saved|config.*saved",         re.I), "INFO"),
    (re.compile(r"login successful|login success",            re.I), "INFO"),
    (re.compile(r"binding added|vlan.*added|vlan.*removed",   re.I), "INFO"),
]

def _infer_severity(message: str, pri_severity: Optional[str]) -> str:
    """Determine severity from keyword patterns, falling back to PRI then INFO."""
    for pattern, level in _KEYWORD_OVERRIDES:
        if pattern.search(message):
            return level
    # Inline severity token in the message text (e.g. "ERROR:" prefix)
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
        dict with keys: raw_text, timestamp, source, severity, message
        or None if the line is blank or a comment.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    # Strip RFC 3164 PRI prefix <NNN> and decode severity
    pri_severity: Optional[str] = None
    m = _PRI_RE.match(line)
    if m:
        pri_val = int(m.group(1))
        pri_severity = _PRI_SEVERITY_MAP.get(pri_val % 8, "INFO")
        line_body = line[m.end():]
    else:
        line_body = line

    # Match timestamp
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
        # No recognisable timestamp — keep line but use current time
        ts_dt = datetime.utcnow()

    # Next token is hostname / source device
    parts = rest.split(None, 1)
    source = parts[0] if parts else "unknown"
    message = parts[1].strip() if len(parts) > 1 else rest

    # Strip leading severity / process token from message before Drain
    # e.g. "OSPF: Neighbor ..." -> "Neighbor ..."  (keep the process prefix
    # for template clustering, strip only bare severity words like "ERROR:")
    clean_message = _SEVERITY_TOKEN_RE.sub("", message, count=1).strip().lstrip(":").strip()

    severity = _infer_severity(message, pri_severity)

    return {
        "raw_text": line,           # original line (with PRI stripped already)
        "timestamp": ts_dt,
        "source": source,
        "severity": severity,
        "message": clean_message or message,
    }
