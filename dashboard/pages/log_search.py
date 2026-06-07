"""
dashboard/pages/log_search.py
================================
Page 4 — Log Search

Full-text Elasticsearch search with filters, a results table,
correlation ID jump links, and CSV export.
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
from data import db, es
from ui import apply_theme, service_status_dot

st.set_page_config(
    page_title="Log Search · HPE CX",
    page_icon="🔎",
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

    all_hosts = db.get_host_list()
    host_filter = st.selectbox(
        "Host",
        ["All"] + all_hosts,
        key="search_host",
    )
    label_filter = st.selectbox(
        "Label",
        ["All", "critical", "medium", "low", "ignore"],
        key="search_label",
    )
    time_range = st.selectbox(
        "Time range",
        [1, 6, 24, 48, 168],
        index=2,
        format_func=lambda h: f"Last {h}h" if h < 168 else "Last 7d",
        key="search_time",
    )
    result_limit = st.slider("Max results", 10, 500, 100, step=10, key="search_limit")

    st.divider()

    # ES health indicator
    es_ok = es.is_elasticsearch_healthy()
    st.markdown(
        "<h3>Service Health</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        service_status_dot(es_ok, "Elasticsearch: " + ("Online" if es_ok else "Offline")),
        unsafe_allow_html=True,
    )

# ── Page header ────────────────────────────────────────────────────────────
st.markdown("<h1>🔎 Log Search & Discovery</h1>", unsafe_allow_html=True)

# ── ES Offline Banner ───────────────────────────────────────────────────────
if not es_ok:
    st.markdown(
        """
        <div style='background:#fef2f2; border:1px solid #fee2e2; border-radius:12px;
                    padding:1.5rem; margin-bottom:1.5rem; color:#991b1b;'>
          <div style='font-size:1.15rem; font-weight:700; margin-bottom:0.4rem;'>⚠️ Elasticsearch is Unavailable</div>
          <div style='font-size:0.83rem; line-height:1.5; color:#7f1d1d;'>
            The log search capability is disabled because the Elasticsearch cluster is unreachable. 
            To activate text search, please run the following command in your terminal:<br>
            <code style='background:#fee2e2; color:#b91c1c; padding:2px 6px; border-radius:4px; font-weight:600;'>docker compose up -d elasticsearch</code>
            <br><br>
            Once the container starts, the status dot in the sidebar will turn green, and you can query the logs database directly.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Search bar ─────────────────────────────────────────────────────────────
search_col, btn_col = st.columns([8.5, 1.5])
with search_col:
    query = st.text_input(
        "Search query",
        placeholder="e.g.  OSPF neighbor state  ·  interface CRC error  ·  BGP session",
        label_visibility="collapsed",
        key="search_query",
        disabled=not es_ok,
    )
with btn_col:
    search_clicked = st.button("Search logs", type="primary", use_container_width=True, disabled=not es_ok)

# ── Execute search ─────────────────────────────────────────────────────────
results: list[dict] = []
searched = False

if (search_clicked or (query and st.session_state.get("_last_query") != query)) and es_ok:
    st.session_state["_last_query"] = query

    if not query.strip():
        st.warning("Enter a search query to begin.")
    else:
        with st.spinner(f"Searching for **{query}**…"):
            results = es.search_logs(
                query=query,
                host=None if host_filter == "All" else host_filter,
                label=None if label_filter == "All" else label_filter,
                time_range_hours=time_range,
                size=result_limit,
            )
        searched = True

# ── Results display ────────────────────────────────────────────────────────
if searched:
    if not results:
        st.markdown(
            f"""
            <div style='background:#fef9ec; border:1px solid #fde68a; border-radius:10px;
                        padding:1.5rem; text-align:center; margin-top:1rem;'>
              <div style='font-size:1.5rem; margin-bottom:0.5rem;'>🔍</div>
              <div style='font-weight:600; color:#92400e; font-size:0.95rem;'>No results found</div>
              <div style='font-size:0.83rem; color:#b45309; margin-top:4px;'>
                No logs matched "<b>{query}</b>" in the last {time_range}h.
                Try broadening your query or adjusting the time range.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        df = pd.DataFrame(results)
        n = len(df)

        label_counts = (
            df["label"].value_counts().to_dict()
            if "label" in df.columns
            else {}
        )

        colour_map = {
            "critical": "#DC2626",
            "medium": "#F59E0B",
            "low": "#22C55E",
            "ignore": "#64748b"
        }

        label_str = " · ".join(
            f"<span style='color:{colour_map.get(k, '#94a3b8')}; font-weight:700;'>{k.upper()}: {v}</span>"
            for k, v in sorted(label_counts.items(), key=lambda x: x[0])
        )

        st.markdown(
            f"<div style='font-size:0.8rem; color:#64748b; margin-bottom:0.8rem; "
            f"font-family:\"IBM Plex Mono\",monospace; background:#f8fafc; padding:8px 12px; border-radius:6px; border:1px solid #e2e8f0;'>"
            f"Found <b style='color:#0f172a;'>{n}</b> log match{'s' if n != 1 else ''} for "
            f"\"<b style='color:#1d4ed8;'>{query}</b>\" "
            f"&nbsp;·&nbsp; {label_str}</div>",
            unsafe_allow_html=True,
        )

        # ── Results table ──────────────────────────────────────────────────
        display_cols = [c for c in [
            "timestamp", "host", "template_id", "label",
            "importance_score", "correlation_id", "message",
        ] if c in df.columns]

        col_config: dict = {}
        if "importance_score" in df.columns:
            col_config["importance_score"] = st.column_config.NumberColumn(
                "Final Score", format="%.3f"
            )
        if "timestamp" in df.columns:
            col_config["timestamp"] = st.column_config.DatetimeColumn(
                "Timestamp", format="YYYY-MM-DD HH:mm:ss"
            )
        if "label" in df.columns:
            col_config["label"] = st.column_config.TextColumn("Label")
        if "correlation_id" in df.columns:
            col_config["correlation_id"] = st.column_config.TextColumn("Incident ID")

        st.dataframe(
            df[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config=col_config,
            height=400,
        )

        # ── Actions row ────────────────────────────────────────────────────
        action_col1, action_col2, action_col3 = st.columns([3, 3, 4])

        with action_col1:
            st.download_button(
                "⬇️ Export Results to CSV",
                data=df[display_cols].to_csv(index=False),
                file_name=f"log_search_{query[:30].replace(' ','_')}.csv",
                mime="text/csv",
                use_container_width=True,
                type="secondary"
            )

        with action_col2:
            # Jump to incident detail
            incident_ids = []
            if "correlation_id" in df.columns:
                incident_ids = [
                    cid for cid in df["correlation_id"].dropna().unique()
                    if cid and str(cid) != "None" and str(cid) != "nan"
                ]

            if incident_ids:
                col_sel, col_go = st.columns([7, 3])
                with col_sel:
                    selected_cid = st.selectbox(
                        "Jump to incident",
                        [None] + incident_ids,
                        format_func=lambda x: "Select incident…" if x is None else str(x),
                        label_visibility="collapsed",
                        key="search_jump_select",
                    )
                with col_go:
                    if st.button("Jump →", use_container_width=True, disabled=selected_cid is None):
                        st.session_state["selected_incident"] = selected_cid
                        st.switch_page("pages/incident_detail.py")
            else:
                st.caption("No incident IDs in search results")

        with action_col3:
            # Quick statistics
            if "host" in df.columns:
                top_hosts = df["host"].value_counts().head(3)
                hosts_str = ", ".join(f"{h} ({c})" for h, c in top_hosts.items())
                st.markdown(
                    f"<div style='font-size:0.75rem; color:#64748b; padding-top:4px;'>Top hosts: <b>{hosts_str}</b></div>",
                    unsafe_allow_html=True
                )

# ── Tips when idle ──────────────────────────────────────────────────────────
if not searched and not query:
    st.markdown(
        """
        <div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px;
                    padding:1.5rem 2rem; margin-top:1rem;'>
          <div style='font-weight:600; color:#334155; margin-bottom:0.6rem; font-size:0.9rem;'>
            💡 Search Tips & syntax
          </div>
          <ul style='color:#64748b; font-size:0.83rem; line-height:1.8; margin:0; padding-left:1.2rem;'>
            <li>Search by failure type: <code>OSPF neighbor</code>, <code>BGP session dropped</code></li>
            <li>Search by template: <code>INTERFACE_ERROR</code>, <code>STP_TOPOLOGY_CHANGE</code></li>
            <li>Use filters in the sidebar to narrow by host, severity label, or time range</li>
            <li>If a log row has an associated Incident ID, select it in the dropdown and click "Jump" to investigate its correlation graph and AI summary.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
