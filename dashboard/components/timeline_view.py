"""
dashboard/components/timeline_view.py
========================================
Plotly-based event timeline for the Incident Detail page.
Shows each log as a dot on a time axis, colour-coded by severity label.
Root-cause logs are shown as stars.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


_COLOUR_MAP = {
    "critical": "#DC2626",
    "medium":   "#F59E0B",
    "low":      "#22C55E",
    "ignore":   "#94A3B8",
}


def render_timeline(incident_logs: pd.DataFrame) -> None:
    """
    Render an interactive Plotly scatter timeline of incident logs.

    Expects columns: timestamp, host, label, template_id,
                     importance_score, message, sequence_number,
                     is_root_cause  (optional)
    """
    if incident_logs.empty:
        st.info("No logs to display.")
        return

    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("plotly not installed. Run: pip install plotly")
        return

    df = incident_logs.copy()

    # Ensure timestamp is datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])

    if df.empty:
        st.info("No valid timestamps in log data.")
        return

    # Defaults for optional columns
    for col, default in [
        ("label", "ignore"),
        ("host", "unknown"),
        ("template_id", ""),
        ("importance_score", 0.0),
        ("message", ""),
        ("sequence_number", ""),
        ("is_root_cause", False),
    ]:
        if col not in df.columns:
            df[col] = default

    df["is_root_cause"] = df["is_root_cause"].fillna(False).astype(bool)
    df["_colour"] = df["label"].map(_COLOUR_MAP).fillna(_COLOUR_MAP["ignore"])
    df["_size"] = df["is_root_cause"].map({True: 16, False: 8})
    df["_symbol"] = df["is_root_cause"].map({True: "star", False: "circle"})

    # Build separate traces per label for the legend
    fig = go.Figure()

    for label, colour in _COLOUR_MAP.items():
        subset = df[df["label"] == label]
        if subset.empty:
            continue

        for is_rc, symbol, size, opacity in [
            (True, "star", 18, 1.0),
            (False, "circle", 8, 0.85),
        ]:
            sub = subset[subset["is_root_cause"] == is_rc]
            if sub.empty:
                continue

            name = f"{label.upper()} {'★ RC' if is_rc else ''}"
            hover = (
                "<b>%{customdata[0]}</b><br>"
                "Host: %{y}<br>"
                "Time: %{x}<br>"
                "Score: %{customdata[1]:.3f}<br>"
                "Seq: %{customdata[2]}<br>"
                "<i>%{customdata[3]}</i>"
                "<extra></extra>"
            )

            fig.add_trace(
                go.Scatter(
                    x=sub["timestamp"],
                    y=sub["host"],
                    mode="markers",
                    name=name,
                    marker=dict(
                        symbol=symbol,
                        size=size,
                        color=colour,
                        opacity=opacity,
                        line=dict(color="#ffffff", width=1 if is_rc else 0),
                    ),
                    customdata=sub[
                        ["template_id", "importance_score", "sequence_number", "message"]
                    ].values,
                    hovertemplate=hover,
                    showlegend=True,
                )
            )

    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=24, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.03)",
        font=dict(family="IBM Plex Mono, monospace", size=11, color="#334155"),
        xaxis=dict(
            gridcolor="#e2e8f0",
            showgrid=True,
            title=None,
        ),
        yaxis=dict(
            gridcolor="#e2e8f0",
            title=None,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(size=10),
        ),
        hoverlabel=dict(
            bgcolor="#0f172a",
            font_color="#f8fafc",
            bordercolor="#334155",
        ),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Expandable raw log inspector
    st.caption("Select a log to inspect its raw message:")
    seq_options = df["sequence_number"].tolist()
    if seq_options:
        selected_seq = st.selectbox(
            "Sequence number",
            options=seq_options,
            format_func=lambda s: f"#{s} — {df[df['sequence_number'] == s]['template_id'].values[0] if len(df[df['sequence_number'] == s]) else ''}",
            label_visibility="collapsed",
            key="timeline_seq_select",
        )
        if selected_seq is not None:
            row = df[df["sequence_number"] == selected_seq]
            if not row.empty:
                r = row.iloc[0]
                label_html = (
                    f"<span style='color:#F59E0B; font-weight:700'>{r['label'].upper()}</span>"
                    if r["label"] in ("critical", "medium")
                    else f"<span style='color:#94A3B8'>{r['label'].upper()}</span>"
                )
                rc_tag = " 🔴 **ROOT CAUSE**" if r["is_root_cause"] else ""
                with st.expander(
                    f"#{selected_seq} — {r['template_id']} @ {r['timestamp']}{rc_tag}",
                    expanded=True,
                ):
                    st.markdown(
                        f"Host: **{r['host']}** · {label_html} · "
                        f"score: **{float(r['importance_score']):.3f}**",
                        unsafe_allow_html=True,
                    )
                    st.code(r["message"] or "(no message)", language="text")
