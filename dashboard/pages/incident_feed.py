"""
dashboard/pages/incident_feed.py
==================================
Page 1 — Incident Feed

Displays all incidents in reverse-chronological order with filters,
metric tiles, and per-card previews of the LLM summary.
"""

import sys
from pathlib import Path

# ── sys.path bootstrap ──────────────────────────────────────────────────────
_DASHBOARD_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT  = _DASHBOARD_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_DASHBOARD_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st
from data import db
from ui import apply_theme, render_time_window
from components.severity_badge import severity_badge

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
        "<div style='font-size:0.75rem; font-weight:700; text-transform:uppercase; "
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

# Premium custom KPI cards
st.markdown(f"""
<div style="display: flex; gap: 1rem; width: 100%; margin-bottom: 1.5rem; flex-wrap: wrap;">
  <div class="kpi-card" style="flex: 1; min-width: 200px;">
    <div class="kpi-title">Total Incidents</div>
    <div class="kpi-value">{total}</div>
  </div>
  <div class="kpi-card" style="flex: 1; min-width: 200px;">
    <div class="kpi-title" style="color: #dc2626;">Critical Incidents</div>
    <div class="kpi-value" style="color: #dc2626;">{critical_count}</div>
  </div>
  <div class="kpi-card" style="flex: 1; min-width: 200px;">
    <div class="kpi-title">Cross-System</div>
    <div class="kpi-value">{cross_count}</div>
  </div>
  <div class="kpi-card" style="flex: 1; min-width: 200px;">
    <div class="kpi-title">Most Affected Host</div>
    <div class="kpi-value" style="font-size: 1.25rem; font-weight: 700; padding-top: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{most_affected}</div>
  </div>
</div>
""", unsafe_allow_html=True)

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
    f"text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.8rem;'>"
    f"Showing {total} incident{'s' if total != 1 else ''}</div>",
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
    end = incident.get("end_time")
    log_count = incident.get("log_count", 0)
    duration = incident.get("duration", 0)
    is_cross = incident.get("is_cross_system", False)
    final_score = incident.get("final_score", 0.0)
    rc_conf = incident.get("root_cause_confidence", 0.0)

    # Format timestamps
    start_str = "—"
    end_str = "—"
    if start:
        try:
            import pandas as pd
            start_str = pd.to_datetime(start).strftime("%d %b %Y, %H:%M")
        except Exception:
            start_str = str(start)[:16]
    if end:
        try:
            import pandas as pd
            end_str = pd.to_datetime(end).strftime("%H:%M")
        except Exception:
            end_str = str(end)[:16]

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

    # LLM summary preview — reads cache, no API call
    summary = db.get_summary(cid) or ""
    if summary:
        summary_preview = summary[:180] + ("…" if len(summary) > 180 else "")
    else:
        summary_preview = "No summary cached for this incident."

    # Horizontal row container
    with st.container():
        col_card, col_btn = st.columns([8.5, 1.5], vertical_alignment="top")
        with col_card:
            st.markdown(f"""
            <div class="incident-card incident-card-{label}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
                        {severity_badge(label)}
                        <span style="font-family:'IBM Plex Mono',monospace; font-weight:700; font-size:0.95rem; color:#0f172a;">{cid}</span>
                        {f"<span class='cross-system-badge'>⚠ CROSS-SYS</span>" if is_cross else ""}
                    </div>
                    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.78rem; color:#64748b;">
                        ⏱️ {dur_str} &nbsp;·&nbsp; 📋 {log_count:,} events
                    </div>
                </div>
                <div style="margin-top:6px; font-size:0.8rem; color:#475569; font-family:'IBM Plex Mono',monospace;">
                    📅 {start_str} → {end_str} &nbsp;·&nbsp; 🖥️ Host: <span style="font-weight:600; color:#0f172a;">{host}</span>
                </div>
                <div style="margin-top:8px; font-size:0.83rem; color:#334155; line-height:1.5; font-style:italic; border-left: 3px solid #e2e8f0; padding-left: 8px;">
                    {summary_preview}
                </div>
                <div style="margin-top:10px; display:flex; gap:20px; align-items:center; flex-wrap:wrap;">
                    <div style="font-size:0.75rem; color:#64748b; font-weight:600; text-transform:uppercase; letter-spacing:0.04em;">
                        Final Score: <span style="color:#0f172a; font-family:'IBM Plex Mono',monospace; font-size:0.8rem; font-weight:700;">{final_score:.3f}</span>
                    </div>
                    <div style="font-size:0.75rem; color:#64748b; font-weight:600; text-transform:uppercase; letter-spacing:0.04em;">
                        Root Cause Confidence: <span style="color:#0f172a; font-family:'IBM Plex Mono',monospace; font-size:0.8rem; font-weight:700;">{rc_conf:.0%}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_btn:
            st.markdown("<div style='min-height: 0.35rem;'></div>", unsafe_allow_html=True)
            if st.button("View details →", key=f"view_{cid}", use_container_width=True, type="primary"):
                st.session_state["selected_incident"] = cid
                st.switch_page("pages/incident_detail.py")
