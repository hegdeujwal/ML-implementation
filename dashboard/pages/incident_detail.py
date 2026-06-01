"""
dashboard/pages/incident_detail.py
=====================================
Page 2 — Incident Detail  (the demo moment)

Three-column layout:
  Left   — Correlation graph (pyvis)
  Centre — Event timeline (Plotly) + raw log inspector
  Right  — LLM summary, root cause candidates, Regenerate button, deep links
"""

import sys
from pathlib import Path

_DASHBOARD_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT  = _DASHBOARD_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_DASHBOARD_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st
import pandas as pd

from data import db
from ui import apply_theme
from components.severity_badge import severity_badge, severity_dot
from components.graph_view import render_graph
from components.timeline_view import render_timeline
from llm_summary import regenerate_summary

st.set_page_config(
    page_title="Incident Detail · HPE CX",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_theme()

# ── Guard: need a selected incident ─────────────────────────────────────────
cid = st.session_state.get("selected_incident")

if not cid:
    st.markdown(
        """
        <div style='background:#f8fafc; border:1px dashed #cbd5e1; border-radius:12px;
                    padding:3rem 2rem; text-align:center; margin-top:3rem;'>
          <div style='font-size:2.5rem; margin-bottom:0.75rem;'>🔍</div>
          <div style='font-weight:600; color:#334155; font-size:1rem;'>No incident selected</div>
          <div style='color:#64748b; font-size:0.85rem; margin-top:0.4rem;'>
            Open the Incident Feed and click "View →" on any incident card.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("← Back to Incident Feed"):
        st.switch_page("pages/incident_feed.py")
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────
with st.spinner("Loading incident data…"):
    incident_logs = db.get_incident_logs(cid)
    root_causes   = db.get_root_causes(cid)
    summary       = db.get_summary(cid)

# ── Page header ────────────────────────────────────────────────────────────
worst_label = "low"
if not incident_logs.empty and "label" in incident_logs.columns:
    label_order = {"critical": 3, "medium": 2, "low": 1, "ignore": 0}
    worst_label = max(
        incident_logs["label"].dropna().tolist() or ["low"],
        key=lambda x: label_order.get(x, 0),
    )

# Back button + title row
header_col, back_col = st.columns([8, 1])
with header_col:
    st.markdown(
        f"<h1 style='margin-bottom:0;'>🔍 Incident Detail</h1>",
        unsafe_allow_html=True,
    )
with back_col:
    if st.button("← Feed", key="back_to_feed"):
        st.switch_page("pages/incident_feed.py")

# Meta bar
host_val = "—"
start_val = "—"
end_val = "—"
log_count = len(incident_logs)
is_cross = False

if not incident_logs.empty:
    if "host" in incident_logs.columns:
        hosts = incident_logs["host"].dropna().unique()
        host_val = ", ".join(hosts) if len(hosts) <= 3 else f"{len(hosts)} hosts"
        is_cross = len(hosts) > 1

    if "timestamp" in incident_logs.columns:
        ts = pd.to_datetime(incident_logs["timestamp"], errors="coerce").dropna()
        if len(ts):
            start_val = ts.min().strftime("%d %b %Y %H:%M")
            end_val   = ts.max().strftime("%d %b %Y %H:%M")

cross_tag = (
    " &nbsp;<span class='cross-system-badge'>⚠ CROSS-SYS</span>"
    if is_cross else ""
)

st.markdown(
    f"<div style='display:flex; align-items:center; gap:12px; flex-wrap:wrap; "
    f"margin:4px 0 1rem 0;'>"
    f"{severity_badge(worst_label, size='md')}"
    f"<span style='font-family:\"IBM Plex Mono\",monospace; font-weight:700; "
    f"font-size:1.05rem; color:#0f172a;'>{cid}</span>"
    f"<span style='color:#94a3b8;'>·</span>"
    f"<span style='font-size:0.85rem; color:#475569;'>{host_val}</span>"
    f"<span style='color:#cbd5e1;'>·</span>"
    f"<span style='font-size:0.8rem; color:#94a3b8; font-family:\"IBM Plex Mono\",monospace;'>"
    f"{start_val} → {end_val}</span>"
    f"<span style='color:#cbd5e1;'>·</span>"
    f"<span style='font-size:0.8rem; color:#94a3b8;'>{log_count:,} logs</span>"
    f"{cross_tag}"
    f"</div>",
    unsafe_allow_html=True,
)

st.divider()

# ── Three-column layout ────────────────────────────────────────────────────
col_graph, col_timeline, col_summary = st.columns([2, 3, 2])

# ── LEFT: Correlation Graph ────────────────────────────────────────────────
with col_graph:
    st.markdown(
        "<h3 style='margin-bottom:0.4rem;'>🕸️ Correlation Graph</h3>",
        unsafe_allow_html=True,
    )

    # Legend
    st.markdown(
        "<div style='font-size:0.73rem; color:#64748b; margin-bottom:8px; "
        "font-family:\"IBM Plex Mono\",monospace; display:flex; gap:12px;'>"
        "<span>🔴 Root cause</span><span>🟡 Anomalous</span><span>🔵 Normal</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    root_cause_ids: list[str] = []
    if not root_causes.empty and "root_cause_log_id" in root_causes.columns:
        root_cause_ids = root_causes["root_cause_log_id"].dropna().tolist()

    if incident_logs.empty:
        st.info("No log data available for graph rendering.")
    else:
        max_nodes = st.checkbox("Limit to top 20 nodes", value=False, key="graph_limit")
        render_graph(
            correlation_id=cid,
            incident_logs=incident_logs,
            root_cause_ids=root_cause_ids,
            max_nodes=20 if max_nodes else 60,
        )

# ── CENTRE: Event Timeline ─────────────────────────────────────────────────
with col_timeline:
    st.markdown(
        "<h3 style='margin-bottom:0.4rem;'>⏱️ Event Timeline</h3>",
        unsafe_allow_html=True,
    )

    if incident_logs.empty:
        st.info("No log data available for timeline.")
    else:
        render_timeline(incident_logs)

# ── RIGHT: LLM Summary + Root Causes ──────────────────────────────────────
with col_summary:
    st.markdown(
        "<h3 style='margin-bottom:0.4rem;'>🤖 AI Summary</h3>",
        unsafe_allow_html=True,
    )

    # Summary box — always reads from cache, NEVER calls Gemini here
    if summary:
        st.markdown(
            f"""
            <div style='background:#f0f9ff; border:1px solid #bae6fd; border-radius:10px;
                        padding:1rem 1.1rem; font-size:0.87rem; color:#0c4a6e;
                        line-height:1.65; margin-bottom:0.75rem;'>
              {summary}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style='background:#fafafa; border:1px dashed #e2e8f0; border-radius:10px;
                        padding:1rem; font-size:0.83rem; color:#94a3b8; text-align:center;
                        margin-bottom:0.75rem;'>
              No summary cached.<br>
              <span style='font-size:0.78rem;'>Run pipeline or click Regenerate below.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Regenerate button — the ONLY place Gemini is called from the dashboard
    if st.button("🔄 Regenerate summary", use_container_width=True):
        with st.spinner("Calling Gemini API…"):
            template_seq = ""
            if not incident_logs.empty and "template_id" in incident_logs.columns:
                sorted_logs = incident_logs.sort_values("timestamp") if "timestamp" in incident_logs.columns else incident_logs
                templates = sorted_logs["template_id"].dropna().tolist()
                template_seq = " → ".join(templates[:10])
                if len(templates) > 10:
                    template_seq += f" (+{len(templates) - 10} more)"

            rc_str = "none"
            if not root_causes.empty and "root_cause_log_id" in root_causes.columns:
                rc_str = ", ".join(root_causes["root_cause_log_id"].dropna().astype(str).tolist())

            incident_data = {
                "correlation_id": cid,
                "template_sequence": template_seq,
                "root_causes": rc_str,
            }
            new_summary = regenerate_summary(cid, incident_data)
        st.rerun()

    st.divider()

    # Root cause candidates
    st.markdown(
        "<h3 style='margin-bottom:0.5rem;'>🎯 Root Cause Candidates</h3>",
        unsafe_allow_html=True,
    )

    if root_causes.empty:
        st.caption("No root cause candidates identified for this incident.")
    else:
        for _, rc in root_causes.iterrows():
            rc_id = rc.get("root_cause_log_id", "—")
            conf  = float(rc.get("confidence_score", 0))
            in_graph = bool(rc.get("in_graph", False))
            template = rc.get("template_id", "")
            rc_host  = rc.get("host", "")

            graph_badge = (
                "<span style='background:#dcfce7; color:#15803d; font-size:10px; "
                "padding:2px 6px; border-radius:4px; font-weight:600; "
                "font-family:\"IBM Plex Mono\",monospace;'>IN-GRAPH</span>"
                if in_graph else
                "<span style='background:#f1f5f9; color:#64748b; font-size:10px; "
                "padding:2px 6px; border-radius:4px; font-weight:600; "
                "font-family:\"IBM Plex Mono\",monospace;'>OUT-OF-GRAPH</span>"
            )

            # Confidence bar colour
            bar_colour = "#DC2626" if conf >= 0.8 else ("#F59E0B" if conf >= 0.5 else "#22C55E")

            st.markdown(
                f"""
                <div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px;
                            padding:0.7rem 0.9rem; margin-bottom:0.5rem;'>
                  <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
                    <span style='font-family:\"IBM Plex Mono\",monospace; font-size:0.78rem;
                                 font-weight:700; color:#0f172a;'>{rc_id}</span>
                    {graph_badge}
                  </div>
                  {f'<div style="font-size:0.75rem; color:#64748b; margin-top:3px;">{template} · {rc_host}</div>' if template else ''}
                  <div style='margin-top:8px;'>
                    <div style='display:flex; justify-content:space-between; margin-bottom:3px;'>
                      <span style='font-size:0.72rem; color:#94a3b8; font-weight:600;
                                   text-transform:uppercase; letter-spacing:0.05em;'>Confidence</span>
                      <span style='font-family:\"IBM Plex Mono\",monospace; font-size:0.75rem;
                                   font-weight:700; color:{bar_colour};'>{conf:.0%}</span>
                    </div>
                    <div style='background:#e2e8f0; border-radius:4px; height:4px;'>
                      <div style='background:{bar_colour}; width:{conf*100:.0f}%; height:4px;
                                  border-radius:4px; transition:width 0.3s ease;'></div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # Deep links
    st.markdown(
        "<h3 style='margin-bottom:0.5rem;'>🔗 Deep Links</h3>",
        unsafe_allow_html=True,
    )
    grafana_url = f"http://localhost:3000/d/incidents?var-incident={cid}"
    kibana_url  = f"http://localhost:5601/app/discover#/?_g=()&_a=(query:(match_phrase:(correlation_id:'{cid}')))"

    link_col1, link_col2 = st.columns(2)
    with link_col1:
        st.link_button("📊 Grafana", grafana_url, use_container_width=True)
    with link_col2:
        st.link_button("🔎 Kibana", kibana_url, use_container_width=True)

# ── Raw log table (expandable) ─────────────────────────────────────────────
st.divider()
with st.expander("📄 All Logs in this Incident", expanded=False):
    if incident_logs.empty:
        st.info("No log data available.")
    else:
        display_cols = [
            c for c in [
                "sequence_number", "timestamp", "host", "template_id",
                "label", "importance_score", "is_root_cause", "message",
            ]
            if c in incident_logs.columns
        ]
        st.dataframe(
            incident_logs[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "importance_score": st.column_config.NumberColumn(
                    "Score", format="%.3f"
                ),
                "is_root_cause": st.column_config.CheckboxColumn("Root Cause"),
                "timestamp": st.column_config.DatetimeColumn(
                    "Timestamp", format="YYYY-MM-DD HH:mm:ss"
                ),
            },
        )
        st.download_button(
            "⬇️ Export CSV",
            data=incident_logs[display_cols].to_csv(index=False),
            file_name=f"incident_{cid}_logs.csv",
            mime="text/csv",
        )
