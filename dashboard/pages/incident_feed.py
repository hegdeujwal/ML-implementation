"""
dashboard/pages/incident_feed.py
==================================
Page 1 — Incident Feed

Displays all incidents in reverse-chronological order with filters,
metric tiles, and per-card previews of the LLM summary.
"""

import sys
from pathlib import Path

_DASHBOARD_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT  = _DASHBOARD_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_DASHBOARD_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

from data import db
from ui import apply_theme, render_time_window
from components.severity_badge import severity_badge, severity_dot

st.set_page_config(
    page_title="Incident Feed · HPE CX",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

# ── Sidebar filters ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-size:0.72rem; font-weight:700; text-transform:uppercase; "
        "letter-spacing:0.08em; color:#64748b; padding-bottom:0.4rem;'>Filters</div>",
        unsafe_allow_html=True,
    )

    # Time range
    start_dt, end_dt = render_time_window("feed")

    st.markdown("---")

    # Host filter
    all_hosts = db.get_host_list()
    host_filter = st.multiselect(
        "Host",
        options=all_hosts,
        default=[],
        placeholder="All hosts",
        key="feed_host_filter",
    )

    # Severity filter
    severity_options = ["critical", "medium", "low", "ignore"]
    severity_filter = st.multiselect(
        "Severity",
        options=severity_options,
        default=["critical", "medium", "low"],
        key="feed_severity_filter",
    )

    # Cross-system only
    cross_system_only = st.toggle("Cross-system only", value=False, key="feed_cross")

    st.markdown("---")
    st.caption("Showing up to 200 most recent incidents.")

# ── Fetch data ─────────────────────────────────────────────────────────────
with st.spinner("Loading incidents…"):
    incidents = db.get_incidents(
        host=host_filter[0] if len(host_filter) == 1 else None,
        severity=severity_filter if severity_filter else None,
        start_time=start_dt,
        end_time=end_dt,
        cross_system_only=cross_system_only,
    )

    # Client-side multi-host filter (if >1 host selected)
    if len(host_filter) > 1:
        incidents = [i for i in incidents if i.get("host") in host_filter]

# ── Page header ────────────────────────────────────────────────────────────
st.markdown(
    "<h1>📋 Incident Feed</h1>",
    unsafe_allow_html=True,
)

# ── Metric tiles ───────────────────────────────────────────────────────────
total = len(incidents)
critical_count = sum(1 for i in incidents if (i.get("label") or "").lower() == "critical")
cross_count = sum(1 for i in incidents if i.get("is_cross_system"))

affected_hosts: dict[str, int] = {}
for inc in incidents:
    h = inc.get("host", "")
    affected_hosts[h] = affected_hosts.get(h, 0) + 1
most_affected = max(affected_hosts, key=affected_hosts.get) if affected_hosts else "—"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Incidents", total)
col2.metric("Critical", critical_count, delta=None)
col3.metric("Cross-system", cross_count)
col4.metric("Most Affected Host", most_affected)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── Empty state ────────────────────────────────────────────────────────────
if not incidents:
    st.markdown(
        """
        <div style='background:#f8fafc; border:1px dashed #cbd5e1; border-radius:12px;
                    padding:3rem 2rem; text-align:center; margin-top:1rem;'>
          <div style='font-size:2.5rem; margin-bottom:0.75rem;'>🔍</div>
          <div style='font-weight:600; color:#334155; font-size:1rem;'>No incidents found</div>
          <div style='color:#64748b; font-size:0.85rem; margin-top:0.4rem;'>
            Adjust your filters or run the scoring pipeline to populate data.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Section label ──────────────────────────────────────────────────────────
st.markdown(
    f"<div style='font-size:0.78rem; font-weight:600; color:#64748b; "
    f"text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.6rem;'>"
    f"{total} incident{'s' if total != 1 else ''}</div>",
    unsafe_allow_html=True,
)

# ── Incident cards ─────────────────────────────────────────────────────────
_LABEL_ORDER = {"critical": 0, "medium": 1, "low": 2, "ignore": 3}
incidents_sorted = sorted(
    incidents, key=lambda i: _LABEL_ORDER.get((i.get("label") or "ignore").lower(), 99)
)

for incident in incidents_sorted:
    cid = incident.get("correlation_id", "—")
    label = (incident.get("label") or "ignore").lower()
    host = incident.get("host", "—")
    start = incident.get("start_time")
    log_count = incident.get("log_count", 0)
    duration = incident.get("duration", 0)
    is_cross = incident.get("is_cross_system", False)

    # Format timestamp
    start_str = ""
    if start:
        try:
            import pandas as pd
            ts = pd.to_datetime(start)
            start_str = ts.strftime("%d %b %Y, %H:%M")
        except Exception:
            start_str = str(start)[:16]

    # Duration string
    if duration and duration > 0:
        if duration < 60:
            dur_str = f"{duration}s"
        elif duration < 3600:
            dur_str = f"{duration // 60}m {duration % 60}s"
        else:
            dur_str = f"{duration // 3600}h {(duration % 3600) // 60}m"
    else:
        dur_str = "—"

    # Left accent colour
    accent = {"critical": "#DC2626", "medium": "#F59E0B", "low": "#22C55E"}.get(label, "#94A3B8")

    with st.container(border=True):
        # Card header row
        col_badge, col_info, col_actions = st.columns([1.5, 7, 1.5])

        with col_badge:
            st.markdown(
                f"<div style='padding-top:4px;'>{severity_badge(label)}</div>",
                unsafe_allow_html=True,
            )
            if is_cross:
                st.markdown(
                    "<div style='margin-top:5px;'>"
                    "<span class='cross-system-badge'>⚠ CROSS-SYS</span>"
                    "</div>",
                    unsafe_allow_html=True,
                )

        with col_info:
            # ID + host row
            st.markdown(
                f"<div style='display:flex; align-items:baseline; gap:10px; flex-wrap:wrap;'>"
                f"<span style='font-family:\"IBM Plex Mono\",monospace; font-weight:700; "
                f"font-size:0.92rem; color:#0f172a;'>{cid}</span>"
                f"<span style='color:#94a3b8; font-size:0.8rem;'>·</span>"
                f"<span style='font-size:0.82rem; color:#475569; font-weight:500;'>{host}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            # Meta row
            st.markdown(
                f"<div style='font-size:0.78rem; color:#94a3b8; margin-top:3px; "
                f"font-family:\"IBM Plex Mono\",monospace;'>"
                f"{start_str} &nbsp;·&nbsp; {log_count:,} logs &nbsp;·&nbsp; {dur_str}"
                f"</div>",
                unsafe_allow_html=True,
            )
            # LLM summary preview — reads cache, no API call
            summary = db.get_summary(cid) or ""
            if summary:
                preview = summary[:130] + ("…" if len(summary) > 130 else "")
                st.markdown(
                    f"<div style='font-size:0.82rem; color:#475569; margin-top:6px; "
                    f"line-height:1.5; font-style:italic;'>{preview}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div style='font-size:0.78rem; color:#cbd5e1; margin-top:4px; "
                    "font-style:italic;'>No summary cached — run pipeline to generate.</div>",
                    unsafe_allow_html=True,
                )

        with col_actions:
            if st.button(
                "View →",
                key=f"view_{cid}",
                type="primary",
                use_container_width=True,
            ):
                st.session_state["selected_incident"] = cid
                st.switch_page("pages/incident_detail.py")

    # Tiny gap between cards
    st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)
