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

st.set_page_config(
    page_title="HPE CX Incident Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui import apply_theme
apply_theme()

# ── Sidebar branding ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style='padding: 0.5rem 0 1.2rem 0;'>
          <div style='font-size:1.1rem; font-weight:700; color:#0f172a; letter-spacing:-0.02em;'>
            ⚡ HPE CX Intelligence
          </div>
          <div style='font-size:0.72rem; color:#64748b; margin-top:2px; font-family:"IBM Plex Mono",monospace;'>
            Network Incident Dashboard
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

# ── Home redirect ─────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='text-align:center; padding: 3rem 0 2rem 0;'>
      <div style='font-size:3rem;'>⚡</div>
      <h1 style='font-size:2rem; font-weight:700; color:#0f172a; margin:0.5rem 0;'>
        HPE CX Incident Intelligence
      </h1>
      <p style='color:#64748b; font-size:1rem; margin-bottom:2rem;'>
        Real-time network incident analysis powered by ML anomaly detection and LLM summarisation.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
        <div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;
                    padding:1.5rem; text-align:center; transition:all 0.15s;'>
          <div style='font-size:2rem; margin-bottom:0.5rem;'>📋</div>
          <div style='font-weight:700; color:#0f172a; font-size:0.95rem;'>Incident Feed</div>
          <div style='color:#64748b; font-size:0.8rem; margin-top:4px;'>
            All incidents with severity filters
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;
                    padding:1.5rem; text-align:center;'>
          <div style='font-size:2rem; margin-bottom:0.5rem;'>🔍</div>
          <div style='font-weight:700; color:#0f172a; font-size:0.95rem;'>Incident Detail</div>
          <div style='color:#64748b; font-size:0.8rem; margin-top:4px;'>
            Graph, timeline &amp; LLM summary
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;
                    padding:1.5rem; text-align:center;'>
          <div style='font-size:2rem; margin-bottom:0.5rem;'>🖥️</div>
          <div style='font-weight:700; color:#0f172a; font-size:0.95rem;'>Host Health</div>
          <div style='color:#64748b; font-size:0.8rem; margin-top:4px;'>
            Per-host anomaly rates
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        """
        <div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;
                    padding:1.5rem; text-align:center;'>
          <div style='font-size:2rem; margin-bottom:0.5rem;'>🔎</div>
          <div style='font-weight:700; color:#0f172a; font-size:0.95rem;'>Log Search</div>
          <div style='color:#64748b; font-size:0.8rem; margin-top:4px;'>
            Full-text Elasticsearch search
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
st.info("👈 Use the sidebar navigation to open a page.", icon="ℹ️")
