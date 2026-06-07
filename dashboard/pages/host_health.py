"""
dashboard/pages/host_health.py
================================
Page 3 — Host Health

Per-host incident counts, anomaly rates, volume trends, and chart visualizations.
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
import pandas as pd
from data import db
from ui import apply_theme, render_time_window

st.set_page_config(
    page_title="Host Health · HPE CX",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

# ── Sidebar filters ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-size:0.75rem; font-weight:700; text-transform:uppercase; "
        "letter-spacing:0.08em; color:#64748b; padding-bottom:0.4rem;'>Filters</div>",
        unsafe_allow_html=True,
    )
    start_dt, end_dt = render_time_window("host_health")

# ── Page header ────────────────────────────────────────────────────────────
st.markdown("<h1>🖥️ Host Health & Anomalies</h1>", unsafe_allow_html=True)

# ── Fetch data ─────────────────────────────────────────────────────────────
with st.spinner("Loading host statistics…"):
    try:
        stats = db.get_host_stats(start_time=start_dt, end_time=end_dt)
    except Exception as e:
        st.error(f"Failed to load host stats: {e}")
        st.info("Ensure PostgreSQL is running and the scoring pipeline has written data.")
        st.stop()

if stats is None or stats.empty:
    st.markdown(
        """
        <div style='background:#f8fafc; border:1px dashed #cbd5e1; border-radius:12px;
                    padding:3rem 2rem; text-align:center; margin-top:1rem;'>
          <div style='font-size:2.5rem; margin-bottom:0.75rem;'>🖥️</div>
          <div style='font-weight:600; color:#334155; font-size:1rem;'>No host data for this time range</div>
          <div style='color:#64748b; font-size:0.85rem; margin-top:0.4rem;'>
            Run the scoring pipeline first, or expand the time window.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Metric tiles ────────────────────────────────────────────────────────────
total_hosts = len(stats)
total_inc   = int(stats["incident_count"].sum())   if "incident_count" in stats.columns else 0
total_crit  = int(stats["critical_count"].sum())   if "critical_count"  in stats.columns else 0
avg_anomaly = float(stats["anomaly_rate"].mean())  if "anomaly_rate"    in stats.columns else 0.0

worst_host = "—"
if "incident_count" in stats.columns and not stats.empty:
    worst_host = stats.sort_values("incident_count", ascending=False).iloc[0].get("host", "—")

# Premium custom KPI cards
st.markdown(f"""
<div style="display: flex; gap: 1rem; width: 100%; margin-bottom: 1.5rem; flex-wrap: wrap;">
  <div class="kpi-card" style="flex: 1; min-width: 200px;">
    <div class="kpi-title">Hosts Monitored</div>
    <div class="kpi-value">{total_hosts}</div>
  </div>
  <div class="kpi-card" style="flex: 1; min-width: 200px;">
    <div class="kpi-title">Total Incidents</div>
    <div class="kpi-value">{total_inc:,}</div>
  </div>
  <div class="kpi-card" style="flex: 1; min-width: 200px;">
    <div class="kpi-title" style="color: #dc2626;">Critical Incidents</div>
    <div class="kpi-value" style="color: #dc2626;">{total_crit:,}</div>
  </div>
  <div class="kpi-card" style="flex: 1; min-width: 200px;">
    <div class="kpi-title">Avg Anomaly Rate</div>
    <div class="kpi-value">{avg_anomaly:.1%}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Two Column visual section ──────────────────────────────────────────────
col_chart, col_trend = st.columns([1, 1])

with col_chart:
    st.markdown("<h2>Anomaly Rate by Host</h2>", unsafe_allow_html=True)

    if "host" in stats.columns and "anomaly_rate" in stats.columns:
        try:
            import plotly.express as px

            chart_df = stats.sort_values("anomaly_rate", ascending=False).head(15)
            # Colour bars by anomaly rate severity
            chart_df["_colour"] = chart_df["anomaly_rate"].apply(
                lambda r: "#DC2626" if r > 0.3 else ("#F59E0B" if r > 0.1 else "#22C55E")
            )

            fig = px.bar(
                chart_df,
                x="host",
                y="anomaly_rate",
                color="_colour",
                color_discrete_map="identity",
                labels={"host": "Host", "anomaly_rate": "Anomaly Rate"},
                text=chart_df["anomaly_rate"].apply(lambda r: f"{r:.1%}"),
            )
            fig.update_layout(
                height=250,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.01)",
                font=dict(family="IBM Plex Mono, monospace", size=10, color="#334155"),
                xaxis=dict(gridcolor="#f1f5f9", title=None),
                yaxis=dict(gridcolor="#e2e8f0", tickformat=".0%", title=None),
                showlegend=False,
                hoverlabel=dict(bgcolor="#0f172a", font_color="#f8fafc"),
            )
            fig.update_traces(textposition="outside", textfont_size=9)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        except ImportError:
            chart_data = stats.set_index("host")[["anomaly_rate"]]
            st.bar_chart(chart_data, color="#1d4ed8")
            
with col_trend:
    st.markdown("<h2>Incident Volume Over Time</h2>", unsafe_allow_html=True)
    try:
        hourly_df = db.get_incident_count_by_hour(start_time=start_dt, end_time=end_dt)
        if not hourly_df.empty:
            import plotly.graph_objects as go

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=hourly_df["hour"] if "hour" in hourly_df.columns else hourly_df.index,
                y=hourly_df.get("incident_count", hourly_df.iloc[:, 0]),
                mode="lines+markers",
                line=dict(color="#1d4ed8", width=2.5),
                marker=dict(size=6, color="#1d4ed8"),
                fill="tozeroy",
                fillcolor="rgba(29,78,216,0.06)",
                name="Incidents / hour",
            ))
            fig2.update_layout(
                height=250,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.01)",
                font=dict(family="IBM Plex Mono, monospace", size=10, color="#334155"),
                xaxis=dict(gridcolor="#f1f5f9", title=None),
                yaxis=dict(gridcolor="#e2e8f0", title=None),
                showlegend=False,
                hoverlabel=dict(bgcolor="#0f172a", font_color="#f8fafc"),
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No timeline data available for the selected range.")
    except Exception:
        st.info("No timeline data available.")

st.divider()

# ── Per-host table ─────────────────────────────────────────────────────────
st.markdown("<h2>Per-host Overview</h2>", unsafe_allow_html=True)

col_config: dict = {}
if "host" in stats.columns:
    col_config["host"] = st.column_config.TextColumn("Host")
if "incident_count" in stats.columns:
    col_config["incident_count"] = st.column_config.NumberColumn("Incidents", format="%d")
if "critical_count" in stats.columns:
    col_config["critical_count"] = st.column_config.NumberColumn("Critical 🔴", format="%d")
if "anomaly_rate" in stats.columns:
    col_config["anomaly_rate"] = st.column_config.ProgressColumn(
        "Anomaly Rate",
        min_value=0.0,
        max_value=1.0,
        format="%.1f%%",
    )
if "last_incident_at" in stats.columns:
    col_config["last_incident_at"] = st.column_config.DatetimeColumn(
        "Last Incident Timestamp", format="DD MMM YYYY, HH:mm"
    )

display_cols = [c for c in ["host", "incident_count", "critical_count", "anomaly_rate", "last_incident_at"] if c in stats.columns]

st.dataframe(
    stats[display_cols].sort_values("incident_count", ascending=False),
    use_container_width=True,
    hide_index=True,
    column_config=col_config,
)
