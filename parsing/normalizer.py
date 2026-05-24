"""
parsing/normalizer.py
=====================
Pre-processes raw syslog lines into structured dicts before Drain parsing.

Supported format (HPE CX / Aruba syslog):
    <priority>timestamp hostname process[pid]: severity message
    OR the common BSD syslog RFC 3164 variant:
    Jan 15 10:23:45 switch1 CRIT: Interface eth0/1 is down

Input:  A single raw log line (str).
Output: dict with keys {raw_text, timestamp (str), source, severity, message}
        or None if the line cannot be parsed.

Known limitations
-----------------
- Sub-second timestamps are truncated to second precision.
- Lines without a recognisable severity token are assigned severity "INFO".
- Multi-line log messages (continuation lines) are treated as standalone entries.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

# Matches ISO 8601 and common syslog timestamp prefixes
_TS_PATTERNS = [
    # ISO 8601 — 2024-01-15T10:23:45 or 2024-01-15 10:23:45
    (
        re.compile(
            r"^(?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})"
            r"\s+(?P<rest>.+)$"
        ),
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ),
    # BSD syslog — Jan 15 10:23:45
    (
        re.compile(
            r"^(?P<ts>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})"
            r"\s+(?P<rest>.+)$"
        ),
        "%b %d %H:%M:%S",
        None,
    ),
]

_SEVERITY_TOKENS = {
    "CRITICAL": "CRITICAL",
    "CRIT":     "CRITICAL",
    "ERROR":    "ERROR",
    "ERR":      "ERROR",
    "WARNING":  "WARN",
    "WARN":     "WARN",
    "INFO":     "INFO",
    "NOTICE":   "INFO",
    "DEBUG":    "INFO",
}

_SEVERITY_RE = re.compile(
    r"\b(CRITICAL|CRIT|ERROR|ERR|WARNING|WARN|NOTICE|DEBUG|INFO)\b",
    re.IGNORECASE,
)


def _parse_timestamp(ts_str: str, fmt1: str, fmt2: Optional[str]) -> Optional[str]:
    ts_str = ts_str.replace("T", " ")
    for fmt in filter(None, [fmt1, fmt2]):
        try:
            dt = datetime.strptime(ts_str, fmt)
            # Year defaults to 1900 for BSD syslog — patch to current year
            if dt.year == 1900:
                dt = dt.replace(year=datetime.now().year)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    return None


def _extract_severity(text: str) -> str:
    m = _SEVERITY_RE.search(text)
    if m:
        return _SEVERITY_TOKENS.get(m.group(1).upper(), "INFO")
    return "INFO"


def normalize_line(line: str) -> Optional[dict]:
    """Parse a single syslog line into a structured dict.

    Args:
        line: Raw log line from a syslog file.

    Returns:
        dict with keys: raw_text, timestamp, source, severity, message
        or None if the line is blank, a comment, or cannot be parsed.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    for ts_pattern, fmt1, fmt2 in _TS_PATTERNS:
        m = ts_pattern.match(line)
        if not m:
            continue

        ts_str = m.group("ts")
        rest = m.group("rest").strip()

        timestamp = _parse_timestamp(ts_str, fmt1, fmt2)
        if timestamp is None:
            continue

        # Next token is typically the hostname / source
        parts = rest.split(None, 1)
        source = parts[0] if parts else "unknown"
        message = parts[1].strip() if len(parts) > 1 else ""

        severity = _extract_severity(rest)

        return {
            "raw_text": line,
            "timestamp": timestamp,
            "source": source,
            "severity": severity,
            "message": message,
        }

    # Fallback: unparseable line — keep raw with minimal metadata
    return {
        "raw_text": line,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "unknown",
        "severity": "INFO",
        "message": line,
    }
