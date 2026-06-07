"""
dashboard/pages/incident_detail.py
=====================================
Page 2 — Incident Detail  (the demo moment)

Three-column layout:
  Left   — Correlation graph (pyvis) + Root Cause Detail card
  Centre — Event timeline (Plotly) + Event Flow Pathway + Raw log inspector
  Right  — LLM summary, Regenerate button, ML Diagnostics, Root cause candidates, Deep links
"""

import re
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
from data import db, es
from ui import apply_theme
from components.severity_badge import severity_badge
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


def format_transition_label(template_id: str) -> str:
    """Convert raw template IDs into human-readable labels for the UI."""
    text = str(template_id or "").strip()
    text = re.sub(r"[_-]+", " ", text)
    return text.title()


# ── Back button + dropdown header ───────────────────────────────────────────
col_back, col_title = st.columns([1.1, 8.9], vertical_alignment="top")
with col_back:
    if st.button("← Feed", key="back_to_feed", use_container_width=True):
        st.switch_page("pages/incident_feed.py")
with col_title:
    st.markdown("<h1 style='margin-top:0;'>🔍 Incident Detail & Diagnostics</h1>", unsafe_allow_html=True)

# ── Guard: need a selected incident ─────────────────────────────────────────
cid = st.session_state.get("selected_incident")

# Fetch recent incidents for fallback selector
recent_incidents = db.get_incidents(time_range_hours=720)

if not cid:
    if not recent_incidents:
        st.markdown(
            """
            <div style='background:#f8fafc; border:1px dashed #cbd5e1; border-radius:12px;
                        padding:3rem 2rem; text-align:center; margin-top:2rem;'>
              <div style='font-size:2.5rem; margin-bottom:0.75rem;'>🔍</div>
              <div style='font-weight:600; color:#334155; font-size:1rem;'>No incidents found in database</div>
              <div style='color:#64748b; font-size:0.85rem; margin-top:0.4rem;'>
                Run the scoring pipeline to ingest log data.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()
    else:
        st.markdown("### Select an incident from the list below to begin analysis:")
        options = {i["correlation_id"]: f"{i['correlation_id']} (Severity: {i['label'].upper()} · Host: {i['host']})" for i in recent_incidents}
        selected = st.selectbox("Active Incidents", list(options.keys()), format_func=options.get, key="incident_selector")
        if selected:
            st.session_state["selected_incident"] = selected
            st.rerun()
        st.stop()

# If selected, show details
with st.spinner("Loading incident data…"):
    incident_logs = db.get_incident_logs(cid)
    root_causes   = db.get_root_causes(cid)
    summary       = db.get_summary(cid)

# Handle empty state
if incident_logs.empty:
    st.warning(f"No log data found in database for incident ID {cid}.")
    st.info("Ensure the pipeline has run and loaded data to PostgreSQL.")
    if st.button("Select another incident"):
        st.session_state["selected_incident"] = None
        st.rerun()
    st.stop()

# ── Incident Metabar ────────────────────────────────────────────────────────
worst_label = "low"
if "label" in incident_logs.columns:
    label_order = {"critical": 3, "medium": 2, "low": 1, "ignore": 0}
    worst_label = max(
        incident_logs["label"].dropna().tolist() or ["low"],
        key=lambda x: label_order.get(x, 0),
    )

host_val = "—"
start_val = "—"
end_val = "—"
log_count = len(incident_logs)
is_cross = False

if "host" in incident_logs.columns:
    hosts = incident_logs["host"].dropna().unique()
    host_val = ", ".join(hosts) if len(hosts) <= 3 else f"{len(hosts)} hosts"
    is_cross = len(hosts) > 1

if "timestamp" in incident_logs.columns:
    ts = pd.to_datetime(incident_logs["timestamp"], errors="coerce").dropna()
    if len(ts):
        start_val = ts.min().strftime("%d %b %Y %H:%M:%S")
        end_val   = ts.max().strftime("%H:%M:%S")

cross_tag = " &nbsp;<span class='cross-system-badge'>⚠ CROSS-SYS</span>" if is_cross else ""

st.markdown(
    f"<div style='display:flex; align-items:center; gap:12px; flex-wrap:wrap; "
    f"margin:-8px 0 1rem 0; padding: 0.5rem 0.8rem; background:#f8fafc; border-radius:8px; border:1px solid #e2e8f0;'>"
    f"{severity_badge(worst_label, size='md')}"
    f"<span style='font-family:\"IBM Plex Mono\",monospace; font-weight:700; font-size:1.05rem; color:#0f172a;'>{cid}</span>"
    f"<span style='color:#cbd5e1;'>|</span>"
    f"<span style='font-size:0.85rem; color:#475569;'><b>Hosts:</b> {host_val}</span>"
    f"<span style='color:#cbd5e1;'>|</span>"
    f"<span style='font-size:0.8rem; color:#64748b; font-family:\"IBM Plex Mono\",monospace;'>"
    f"📅 {start_val} → {end_val}</span>"
    f"<span style='color:#cbd5e1;'>|</span>"
    f"<span style='font-size:0.8rem; color:#64748b;'><b>Volume:</b> {log_count:,} logs</span>"
    f"{cross_tag}"
    f"</div>",
    unsafe_allow_html=True,
)

# ── Incident Switcher bar ───────────────────────────────────────────────────
if recent_incidents:
    switcher_options = {i["correlation_id"]: f"{i['correlation_id']} ({i['label'].upper()} · {i['host']})" for i in recent_incidents}
    col_label, col_select = st.columns([2, 8])
    with col_label:
        st.write("<div style='padding-top:6px; font-size:0.82rem; font-weight:600; color:#64748b;'>SWITCH INCIDENT:</div>", unsafe_allow_html=True)
    with col_select:
        switched = st.selectbox(
            "Switch Incident",
            options=list(switcher_options.keys()),
            format_func=switcher_options.get,
            index=list(switcher_options.keys()).index(cid) if cid in switcher_options else 0,
            label_visibility="collapsed",
            key="incident_switcher"
        )
        if switched != cid:
            st.session_state["selected_incident"] = switched
            st.rerun()

# ── Three-column layout ────────────────────────────────────────────────────
col_graph, col_timeline, col_summary = st.columns([2.5, 3.5, 2.5])

# ── LEFT: Correlation Graph & Root Cause details ──────────────────────────
with col_graph:
    st.markdown(
        "<h3 style='margin-bottom:0.4rem; display:flex; align-items:center; gap:6px;'>🕸️ Correlation Graph</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:0.73rem; color:#64748b; margin-bottom:8px; "
        "font-family:\"IBM Plex Mono\",monospace; display:flex; gap:12px;'>"
        "<span>🔴 Root cause</span><span>🟡 Anomalous</span><span>🔵 Normal</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    root_cause_ids: list[str] = []
    if not root_causes.empty and "root_cause_log_id" in root_causes.columns:
        root_cause_templates = root_causes["template_id"].dropna().tolist()
        root_cause_ids = root_cause_templates

    max_nodes = st.checkbox("Limit to top 20 nodes", value=False, key="graph_limit")
    render_graph(
        correlation_id=cid,
        incident_logs=incident_logs,
        root_cause_ids=root_cause_ids,
        max_nodes=20 if max_nodes else 60,
    )

    st.markdown("<h3>🎯 Core Root Cause Event</h3>", unsafe_allow_html=True)
    if not root_causes.empty:
        best_rc = root_causes.iloc[0]
        rc_conf = float(best_rc.get("confidence_score", 0))
        rc_ts   = pd.to_datetime(best_rc.get("timestamp")).strftime("%d %b, %H:%M:%S")
        st.markdown(
            f"""
            <div style='background:#fff5f5; border:1px solid #fee2e2; border-radius:10px; padding:0.85rem; margin-top:0.5rem;'>
              <div style='display:flex; justify-content:space-between; align-items:center;'>
                <span style='font-family:"IBM Plex Mono",monospace; font-size:0.8rem; font-weight:700; color:#b91c1c;'>
                  {best_rc.get("root_cause_log_id")}
                </span>
                <span style='background:#fecaca; color:#991b1b; font-size:10px; font-weight:700; padding:2px 6px; border-radius:4px;'>
                  CONFIDENCE: {rc_conf:.0%}
                </span>
              </div>
              <div style='font-size:0.8rem; font-weight:600; color:#374151; margin-top:6px;'>
                Template: <code style='color:#b91c1c;'>{best_rc.get("template_id")}</code>
              </div>
              <div style='font-size:0.75rem; color:#6b7280; margin-top:2px;'>
                Host: <b>{best_rc.get("host")}</b> · Time: {rc_ts}
              </div>
              <div style='background:#ffffff; border:1px solid #f3f4f6; border-radius:6px; padding:6px; font-size:0.78rem; color:#4b5563; font-family:"IBM Plex Mono",monospace; margin-top:8px;'>
                {best_rc.get("message")}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.caption("No root cause log identified for this incident.")

# ── CENTRE: Event Timeline, Event Flow, and Log Inspector ─────────────────
with col_timeline:
    st.markdown(
        "<h3 style='margin-bottom:0.4rem;'>⏱️ Event Timeline</h3>",
        unsafe_allow_html=True,
    )
    render_timeline(incident_logs)

    # ── Event Flow Pathway ──────────────────────────────────────────────────
    # FIX 1: Merged the <style> block and the flow HTML into ONE st.markdown()
    # call. Streamlit sandboxes each markdown block independently, so a CSS
    # class injected in block A is never visible to block B. All styling is
    # now fully inline so no stylesheet dependency exists at all.
    st.markdown("<h3>🔄 Incident Event Flow Pathway</h3>", unsafe_allow_html=True)

    if not incident_logs.empty and "template_id" in incident_logs.columns:
        sorted_flow = incident_logs.sort_values("timestamp")

        flow_steps = []
        last_tpl = None
        for _, r in sorted_flow.iterrows():
            tpl   = r["template_id"]
            lbl   = r["label"]
            is_rc = r.get("is_root_cause", False)
            if tpl != last_tpl:
                flow_steps.append((tpl, lbl, is_rc))
                last_tpl = tpl

        steps_html = ""
        for tpl, lbl, is_rc in flow_steps[:8]:
            display_name = format_transition_label(tpl)

            if is_rc:
                dot_style = "background:#b91c1c; width:12px; height:12px; left:-21px; border-radius:2px;"
                label_tag = "<span style='color:#b91c1c; font-size:10px; font-weight:700;'>[ROOT CAUSE]</span>"
            elif lbl == "critical":
                dot_style = "background:#dc2626; border-radius:50%;"
                label_tag = "<span style='color:#dc2626; font-size:10px; font-weight:700;'>[CRITICAL]</span>"
            elif lbl == "medium":
                dot_style = "background:#f59e0b; border-radius:50%;"
                label_tag = "<span style='color:#b45309; font-size:10px; font-weight:700;'>[WARNING]</span>"
            else:
                dot_style = "background:#94a3b8; border-radius:50%;"
                label_tag = ""

            steps_html += (
                "<div style='position:relative; margin-bottom:12px;'>"
                "  <div style='position:absolute; left:-20px; top:5px; width:10px; height:10px;"
                "              border:2px solid white; " + dot_style + "'></div>"
                "  <code style='font-size:0.8rem; font-weight:600; color:#334155;'>" + display_name + "</code>"
                "  " + label_tag +
                "  <div style='font-size:0.72rem; color:#64748b; margin-top:2px;'>" + tpl + "</div>"
                "</div>"
            )

        if len(flow_steps) > 8:
            steps_html += (
                "<div style='position:relative; margin-bottom:12px;'>"
                "  <span style='font-size:0.78rem; color:#64748b; font-style:italic;'>"
                "    ... (+" + str(len(flow_steps) - 8) + " more transitions)"
                "  </span>"
                "</div>"
            )

        # Vertical guide line rendered as a real <div>, not a CSS ::before
        # pseudo-element (Streamlit's sanitizer strips pseudo-elements).
        flow_html = (
            "<div style='position:relative; padding-left:20px; margin-top:10px; margin-bottom:15px;'>"
            "  <div style='position:absolute; left:4px; top:8px; bottom:8px; width:2px; background:#cbd5e1;'></div>"
            + steps_html
            + "</div>"
        )
        st.markdown(flow_html, unsafe_allow_html=True)
    else:
        st.caption("No event flow data available.")

    # ── Raw log table ────────────────────────────────────────────────────────
    st.markdown("<h3>📄 Raw Log Inspector</h3>", unsafe_allow_html=True)
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
        height=280,
        column_config={
            "importance_score": st.column_config.NumberColumn("Score", format="%.3f"),
            "is_root_cause":    st.column_config.CheckboxColumn("Root Cause"),
            "timestamp":        st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm:ss"),
        },
    )
    st.download_button(
        "⬇️ Export CSV",
        data=incident_logs[display_cols].to_csv(index=False),
        file_name=f"incident_{cid}_logs.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ── RIGHT: AI Summary, ML Diagnostics, Root Cause Candidates ─────────────
with col_summary:
    st.markdown(
        "<h3 style='margin-bottom:0.4rem;'>🤖 AI Incident Summary</h3>",
        unsafe_allow_html=True,
    )

    if summary:
        st.markdown(
            f"""
            <div style='background:#f0f9ff; border:1px solid #bae6fd; border-radius:10px;
                        padding:1rem; font-size:0.87rem; color:#0c4a6e;
                        line-height:1.6; margin-bottom:0.6rem;'>
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
                        margin-bottom:0.6rem;'>
              No summary cached.<br>
              <span style='font-size:0.75rem;'>Click Regenerate below to call Gemini.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.button("🔄 Regenerate summary", use_container_width=True):
        with st.spinner("Calling Gemini API…"):
            template_seq = ""
            if not incident_logs.empty and "template_id" in incident_logs.columns:
                sorted_logs = incident_logs.sort_values("timestamp")
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

    # System Health Check
    st.markdown("<h3>🩺 System Health Check</h3>", unsafe_allow_html=True)
    db_ok    = db.is_db_healthy()
    es_ok    = es.is_elasticsearch_healthy()
    db_color = "#22C55E" if db_ok else "#DC2626"
    es_color = "#22C55E" if es_ok else "#DC2626"
    db_label = "Healthy" if db_ok else "Unavailable"
    es_label = "Online"  if es_ok else "Offline"
    st.markdown(
        "<div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:0.8rem; font-size:0.82rem; color:#334155;'>"
        "  <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;'>"
        "    <span>PostgreSQL</span>"
        "    <span style='color:" + db_color + "; font-weight:700;'>" + db_label + "</span>"
        "  </div>"
        "  <div style='display:flex; justify-content:space-between; align-items:center;'>"
        "    <span>Elasticsearch</span>"
        "    <span style='color:" + es_color + "; font-weight:700;'>" + es_label + "</span>"
        "  </div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── ML Diagnostics ───────────────────────────────────────────────────────
    st.markdown("<h3>📊 ML Diagnostics</h3>", unsafe_allow_html=True)

    max_final = float(incident_logs["final_score"].max())           if "final_score"           in incident_logs.columns else 0.0
    max_rc    = float(incident_logs["root_cause_confidence"].max()) if "root_cause_confidence" in incident_logs.columns else 0.0
    max_freq  = float(incident_logs["frequency_score"].max())       if "frequency_score"       in incident_logs.columns else 0.0
    max_sev   = float(incident_logs["severity_weight"].max())       if "severity_weight"       in incident_logs.columns else 0.0
    max_prox  = float(incident_logs["counter_proximity"].max())     if "counter_proximity"     in incident_logs.columns else 0.0

    # FIX 2: metric_bar() now builds HTML via plain string concatenation.
    # The original version used f-strings containing bare "%" characters
    # (e.g. "width:{fill_pct}%"). When metric_bar() was called *inside*
    # the outer diag_html f-string, Python re-parsed those "%" as format
    # specifiers and raised a ValueError / silently corrupted the string.
    # Using str concatenation + explicit "{}%".format(n) avoids this entirely.
    def metric_bar(label: str, value: float, max_val: float = 1.0, is_pct: bool = False) -> str:
        fill_pct  = min(100, int((value / max_val) * 100)) if max_val else 0
        color     = "#DC2626" if value >= (max_val * 0.75) else ("#F59E0B" if value >= (max_val * 0.4) else "#22C55E")
        val_str   = "{:.0%}".format(value)   if is_pct else "{:.3f}".format(value)
        width_str = "{}%".format(fill_pct)
        return (
            "<div style='margin-bottom:8px;'>"
            "  <div style='display:flex; justify-content:space-between; font-size:0.75rem; font-weight:600; color:#475569;'>"
            "    <span>" + label + "</span>"
            "    <span style='font-family:\"IBM Plex Mono\",monospace; color:" + color + ";'>" + val_str + "</span>"
            "  </div>"
            "  <div style='background:#e2e8f0; height:5px; border-radius:3px; margin-top:2px;'>"
            "    <div style='background:" + color + "; width:" + width_str + "; height:5px; border-radius:3px;'></div>"
            "  </div>"
            "</div>"
        )

    # FIX 2 (cont.): diag_html assembled via concatenation, not one giant
    # f-string, so "%" inside metric_bar() return values are never re-parsed.
    diag_html = (
        "<div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:0.85rem; margin-bottom:0.75rem;'>"
        + metric_bar("Max final_score (Severity Rank)",        max_final)
        + metric_bar("Max root_cause_confidence",              max_rc,   is_pct=True)
        + metric_bar("Max frequency_score (Anomaly Rate)",     max_freq, max_val=10.0)
        + metric_bar("Max severity_weight (Template Risk)",    max_sev)
        + metric_bar("Max counter_proximity (Drop Proximity)", max_prox)
        + "</div>"
    )
    st.markdown(diag_html, unsafe_allow_html=True)

    # Classification Rationale
    st.markdown("<h4>🚨 Classification Rationale</h4>", unsafe_allow_html=True)
    reasons = []
    if max_sev  >= 0.7: reasons.append("<li>High severity template match (CRITICAL / ERROR log level detected).</li>")
    if max_prox >= 0.5: reasons.append("<li>Direct statistical proximity to interface drop/packet drop templates.</li>")
    if max_freq >= 3.0: reasons.append("<li>Burst frequency anomaly: anomalous volume burst on templates.</li>")
    if is_cross:        reasons.append("<li>Cross-system propagation: incident affects multiple network switches.</li>")
    if not reasons:     reasons.append("<li>Standard low-level statistical warning or routing change sequence.</li>")

    st.markdown(
        "<div style='background:#fffbeb; border:1px solid #fef3c7; border-radius:8px; padding:0.6rem 0.8rem; font-size:0.78rem; color:#92400e;'>"
        "  <ul style='margin:0; padding-left:1.1rem; line-height:1.4;'>"
        + "".join(reasons)
        + "  </ul>"
          "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # Root cause candidates list
    st.markdown("<h3>🎯 Root Cause Candidates</h3>", unsafe_allow_html=True)
    if root_causes.empty:
        st.caption("No root cause candidates identified.")
    else:
        for _, rc in root_causes.iterrows():
            rc_id       = str(rc.get("root_cause_log_id", "—"))
            conf        = float(rc.get("confidence_score", 0))
            in_g        = bool(rc.get("in_graph", False))
            rc_host     = str(rc.get("host", ""))
            rc_template = str(rc.get("template_id", ""))

            graph_badge = (
                "<span style='background:#dcfce7; color:#15803d; font-size:9px; "
                "padding:2px 5px; border-radius:4px; font-weight:700; "
                "font-family:\"IBM Plex Mono\",monospace;'>IN-GRAPH</span>"
                if in_g else
                "<span style='background:#f1f5f9; color:#64748b; font-size:9px; "
                "padding:2px 5px; border-radius:4px; font-weight:700; "
                "font-family:\"IBM Plex Mono\",monospace;'>OUT-OF-GRAPH</span>"
            )

            bar_color = "#DC2626" if conf >= 0.8 else ("#F59E0B" if conf >= 0.5 else "#22C55E")
            conf_pct  = "{:.0%}".format(conf)
            bar_width = "{:.0f}%".format(conf * 100)

            st.markdown(
                "<div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:8px;"
                "            padding:0.6rem 0.8rem; margin-bottom:0.4rem;'>"
                "  <div style='display:flex; justify-content:space-between; align-items:center;'>"
                "    <span style='font-family:\"IBM Plex Mono\",monospace; font-size:0.75rem;"
                "                 font-weight:700; color:#0f172a;'>" + rc_id + "</span>"
                "    " + graph_badge +
                "  </div>"
                "  <div style='font-size:0.72rem; color:#64748b; margin-top:2px;'>"
                + rc_template + " · <b>" + rc_host + "</b>"
                + "  </div>"
                  "  <div style='margin-top:6px;'>"
                  "    <div style='display:flex; justify-content:space-between; margin-bottom:2px;'>"
                  "      <span style='font-size:0.68rem; color:#94a3b8; font-weight:600; text-transform:uppercase;'>Confidence</span>"
                  "      <span style='font-family:\"IBM Plex Mono\",monospace; font-size:0.72rem; font-weight:700; color:" + bar_color + ";'>" + conf_pct + "</span>"
                  "    </div>"
                  "    <div style='background:#e2e8f0; border-radius:2px; height:3px;'>"
                  "      <div style='background:" + bar_color + "; width:" + bar_width + "; height:3px; border-radius:2px;'></div>"
                  "    </div>"
                  "  </div>"
                  "</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # Deep links
    st.markdown("<h3>🔗 External Integrations</h3>", unsafe_allow_html=True)
    grafana_url = f"http://localhost:3000/d/incidents?var-incident={cid}"
    kibana_url  = f"http://localhost:5601/app/discover#/?_g=()&_a=(query:(match_phrase:(correlation_id:'{cid}')))"

    link_col1, link_col2 = st.columns(2)
    with link_col1:
        st.link_button("📊 Open Grafana", grafana_url, use_container_width=True)
    with link_col2:
        st.link_button("🔎 Open Kibana", kibana_url, use_container_width=True)