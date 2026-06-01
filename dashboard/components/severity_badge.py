"""
dashboard/components/severity_badge.py
=======================================
Reusable severity badge component for the HPE CX dashboard.
"""

from __future__ import annotations

import streamlit as st

# Palette: critical=red, medium=amber/orange, low=green, ignore=slate
_COLOURS: dict[str, tuple[str, str]] = {
    "critical": ("#DC2626", "#FEF2F2"),   # (text-bg, pill-bg)
    "medium":   ("#B45309", "#FFFBEB"),
    "low":      ("#15803D", "#F0FDF4"),
    "ignore":   ("#475569", "#F1F5F9"),
}

_DEFAULT = ("#475569", "#F1F5F9")


def severity_badge(label: str, size: str = "sm") -> str:
    """
    Return an HTML string for an inline severity badge.
    Suitable for use with st.markdown(..., unsafe_allow_html=True).
    """
    label_lower = (label or "ignore").lower()
    fg, bg = _COLOURS.get(label_lower, _DEFAULT)
    font_size = "11px" if size == "sm" else "13px"
    padding = "3px 9px" if size == "sm" else "4px 12px"
    return (
        f"<span style='"
        f"background:{bg}; color:{fg}; "
        f"font-size:{font_size}; font-weight:700; "
        f"padding:{padding}; border-radius:20px; "
        f"letter-spacing:0.06em; border:1px solid {fg}33;"
        f"font-family: \"IBM Plex Mono\", monospace;"
        f"'>{label_lower.upper()}</span>"
    )


def render_severity_badge(label: str, size: str = "sm") -> None:
    """Render a severity badge directly into Streamlit."""
    st.markdown(severity_badge(label, size), unsafe_allow_html=True)


def severity_dot(label: str) -> str:
    """Return a small coloured circle span — for compact lists."""
    label_lower = (label or "ignore").lower()
    fg, _ = _COLOURS.get(label_lower, _DEFAULT)
    return (
        f"<span style='display:inline-block; width:8px; height:8px; "
        f"border-radius:50%; background:{fg}; margin-right:6px;'></span>"
    )
