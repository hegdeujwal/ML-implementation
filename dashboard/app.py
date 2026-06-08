"""
dashboard/app.py
================
HPE CX Incident Intelligence Dashboard — Streamlit entry point.

Streamlit multi-page apps work by placing page scripts in pages/.
This file configures global layout and renders the landing / home view,
which redirects to the Incident Feed.
"""

import sys
from pathlib import Path

# ── sys.path bootstrap (each Streamlit page is its own process) ────────────
_DASHBOARD_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT  = _DASHBOARD_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_DASHBOARD_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st
from ui import apply_theme, service_status_dot
from data import db, es

st.set_page_config(
    page_title="HPE CX Incident Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()

# ── Sidebar branding ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style='padding: 0.5rem 0 1rem 0;'>
          <div style='font-size:1.15rem; font-weight:700; color:#0f172a; letter-spacing:-0.02em;'>
            ⚡ HPE CX Intelligence
          </div>
          <div style='font-size:0.7rem; color:#64748b; margin-top:2px; font-family:"IBM Plex Mono",monospace;'>
            Observability Platform
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    
    st.markdown("<h3>System Services</h3>", unsafe_allow_html=True)
    db_ok = db.is_db_healthy()
    es_ok = es.is_elasticsearch_healthy()
    
    st.markdown(
        f"""
        <div style='display:flex; flex-direction:column; gap:8px; margin-top:6px;'>
          {service_status_dot(db_ok, "PostgreSQL: " + ("Connected" if db_ok else "Disconnected"))}
          {service_status_dot(es_ok, "Elasticsearch: " + ("Online" if es_ok else "Offline"))}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

# ── Home display ──────────────────────────────────────────────────────────
hero_col, status_col = st.columns([2.2, 1], gap="large")

with hero_col:
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #1e293b 100%);
                    padding: 2rem 2rem 1.4rem 2rem; border-radius: 18px; margin-bottom: 1.2rem;
                    box-shadow: 0 10px 30px rgba(15,23,42,0.18); border: 1px solid rgba(148,163,184,0.18);'>
          <div style='display:flex; align-items:center; gap:10px; margin-bottom:0.5rem;'>
            <div style='font-size:1.8rem;'>⚡</div>
            <span style='font-size:0.82rem; text-transform:uppercase; letter-spacing:0.18em; color:#cbd5e1;'>HPE CX Intelligence Hub</span>
          </div>
          <h1 style='font-size:2.15rem; font-weight:800; color:#ffffff !important; margin:0 0 0.35rem 0; letter-spacing:-0.03em;'>
            Start from a clear incident view, not a noisy log dump.
          </h1>
          <p style='color:#cbd5e1; font-size:1rem; line-height:1.55; max-width:760px; margin:0 0 0.8rem 0;'>
            Monitor incidents, inspect root causes, and move from signal to response with one unified operations dashboard.
          </p>
          <div style='display:flex; flex-wrap:wrap; gap:0.45rem;'>
            <span style='background:rgba(59,130,246,0.14); color:#bfdbfe; border:1px solid rgba(147,197,253,0.25); border-radius:999px; padding:0.35rem 0.65rem; font-size:0.78rem;'>Anomaly detection</span>
            <span style='background:rgba(34,197,94,0.12); color:#bbf7d0; border:1px solid rgba(134,239,172,0.25); border-radius:999px; padding:0.35rem 0.65rem; font-size:0.78rem;'>Root-cause clues</span>
            <span style='background:rgba(251,191,36,0.12); color:#fde68a; border:1px solid rgba(253,224,71,0.25); border-radius:999px; padding:0.35rem 0.65rem; font-size:0.78rem;'>Live observability</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with status_col:
    st.markdown(
        """
        <div class="kpi-card" style='padding:1rem; min-height: 180px; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px;'>
          <div style='font-size:0.75rem; text-transform:uppercase; letter-spacing:0.18em; color:#64748b; margin-bottom:0.35rem;'>System status</div>
          <div style='font-size:1.2rem; font-weight:700; color:#0f172a; margin-bottom:0.45rem;'>Live services</div>
          <div style='color:#475569; font-size:0.9rem; line-height:1.45;'>PostgreSQL and Elasticsearch health are visible from the sidebar and this home view for quick triage.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<h2 style='margin-bottom: 0.8rem;'>Quick access</h2>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4, gap="medium")

with col1:
    st.markdown(
        """
        <div class="kpi-card" style="min-height: 170px; display: flex; flex-direction: column; justify-content: space-between; margin-bottom: 10px;">
          <div>
            <div style="font-size: 1.6rem; margin-bottom: 0.5rem;">📋</div>
            <div style="font-weight: 700; color: #0f172a; font-size: 0.95rem;">Incident Feed</div>
            <div style="color: #64748b; font-size: 0.78rem; margin-top: 4px; line-height: 1.45;">
              Real-time incident view with severity, host, and cross-system correlation filters.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open Feed →", key="nav_feed", use_container_width=True, type="primary"):
        st.switch_page("pages/incident_feed.py")

with col2:
    st.markdown(
        """
        <div class="kpi-card" style="min-height: 170px; display: flex; flex-direction: column; justify-content: space-between; margin-bottom: 10px;">
          <div>
            <div style="font-size: 1.6rem; margin-bottom: 0.5rem;">🔍</div>
            <div style="font-weight: 700; color: #0f172a; font-size: 0.95rem;">Incident Detail</div>
            <div style="color: #64748b; font-size: 0.78rem; margin-top: 4px; line-height: 1.45;">
              Force-directed correlation graphs, interactive timelines, and cached AI summaries.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open Details →", key="nav_detail", use_container_width=True, type="primary"):
        st.switch_page("pages/incident_detail.py")

with col3:
    st.markdown(
        """
        <div class="kpi-card" style="min-height: 170px; display: flex; flex-direction: column; justify-content: space-between; margin-bottom: 10px;">
          <div>
            <div style="font-size: 1.6rem; margin-bottom: 0.5rem;">🖥️</div>
            <div style="font-weight: 700; color: #0f172a; font-size: 0.95rem;">Host Health</div>
            <div style="color: #64748b; font-size: 0.78rem; margin-top: 4px; line-height: 1.45;">
              Per-host statistics, incident counts, anomaly rate visualizations, and trend graphs.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open Health →", key="nav_health", use_container_width=True, type="primary"):
        st.switch_page("pages/host_health.py")

with col4:
    st.markdown(
        """
        <div class="kpi-card" style="min-height: 170px; display: flex; flex-direction: column; justify-content: space-between; margin-bottom: 10px;">
          <div>
            <div style="font-size: 1.6rem; margin-bottom: 0.5rem;">🔎</div>
            <div style="font-weight: 700; color: #0f172a; font-size: 0.95rem;">Log Search</div>
            <div style="color: #64748b; font-size: 0.78rem; margin-top: 4px; line-height: 1.45;">
              Full-text search indexed via Elasticsearch with filters and CSV export capabilities.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open Search →", key="nav_search", use_container_width=True, type="primary"):
        st.switch_page("pages/log_search.py")

st.markdown("<br><br>", unsafe_allow_html=True)
st.info("👈 Use the sidebar navigation or click any buttons above to explore.", icon="ℹ️")
