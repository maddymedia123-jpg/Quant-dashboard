"""Trap Intelligence - BTC market-surveillance terminal (Phase 1: deterministic layer)."""
from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from core.accuracy import majority_direction, record_report, score_due_calls, score_reports, summary
from core.agents.llm import LLMClient
from core.agents.report import to_markdown
from core.agents.runner import run_pipeline_sync
from core.agents.settings import load_settings
from core.anchors import anchor_step
from core.config import CATEGORIES, MACRO_EVENTS, TTL
from core.data.market import assemble, load_context, load_spot
from core.indicators.category import analyze_category
from core.indicators.early_warning import early_warnings
from core.store import Store
from core.indicators.trade_map import map_trade
from ui import panels
from ui.charts import render_chart
from ui.theme import inject_css, kpi_strip

st.set_page_config(page_title="Trap Intelligence | BTC", layout="wide", initial_sidebar_state="expanded")


@st.cache_data(ttl=TTL["spot"], show_spinner=False)
def _spot():
    return load_spot()


@st.cache_data(ttl=TTL["context"], show_spinner=False)
def _context():
    return load_context()


@st.cache_resource(show_spinner=False)
def _store():
    return Store()


# ---------- sidebar ----------
st.session_state.setdefault("dark", False)

st.sidebar.markdown("<p class='ti-title'>Trap Intelligence</p><p class='ti-sub'>BTC market surveillance</p>", unsafe_allow_html=True)
dark = st.sidebar.toggle("Dark mode", value=st.session_state["dark"])
st.session_state["dark"] = dark
inject_css(dark)

if st.sidebar.button("Refresh data", width="stretch"):
    _spot.clear()
    _context.clear()
    st.rerun()
settings = load_settings()
st.session_state.setdefault("trap_report", None)
run_clicked = st.sidebar.button(
    "Run Analysis", width="stretch", disabled=settings is None,
    help=(f"13-agent Trap Intelligence report via {settings.provider}: specialists on {settings.specialist_model}, Director on {settings.director_model}."
          if settings else "Add GEMINI_API_KEY or OPENROUTER_API_KEY to .streamlit/secrets.toml (or Streamlit Cloud Secrets) to enable."),
)

with st.spinner("Loading market data"):
    m = assemble(_spot(), _context())

st.sidebar.markdown("---")
st.sidebar.markdown(panels.data_status_html(m), unsafe_allow_html=True)
st.sidebar.caption(f"Updated {m.generated_at.astimezone(timezone.utc):%H:%M:%S} UTC · spot {TTL['spot']}s · context {TTL['context']}s")

# ---------- analyses ----------
analyses = {}
if m.spot.available:
    for key, cat in CATEGORIES.items():
        a = analyze_category(cat, m.spot.frames, m.futures)
        if a is not None:
            analyses[key] = a
store = _store()
now_ms = int(time.time() * 1000)

# ---------- run analysis ----------
if run_clicked and settings is not None and analyses:
    with st.status("Running Trap Intelligence pipeline...", expanded=True) as status:
        lines = st.empty()
        seen: dict[str, str] = {}

        def _progress(agent_id: str, state: str) -> None:
            seen[agent_id] = state
            lines.markdown("  \n".join(f"`{k}` {v}" for k, v in seen.items()))

        try:
            report = run_pipeline_sync(LLMClient(settings), m, analyses, settings,
                                       raw_metrics=panels.raw_metrics_rows(m), progress=_progress)
            st.session_state["trap_report"] = report
            intraday_a = analyses.get("intraday")
            record_report(store, report, m.spot.last or next(iter(analyses.values())).price,
                          intraday_a.vol.sigma_pct if intraday_a else None, majority_direction(analyses), now_ms)
            status.update(label=f"Report {report.report_id} ready in {report.wall_time_s:.0f}s", state="complete", expanded=False)
        except Exception as e:  # noqa: BLE001 - never crash the page on an LLM failure
            status.update(label=f"Pipeline failed: {e}", state="error")
    st.rerun()

report = st.session_state.get("trap_report")

# ---------- anchoring + accuracy ----------
verdicts = {}
classification = report.director.trap_classification if (report is not None and report.director is not None) else None
try:
    for key, a in analyses.items():
        verdicts[key] = anchor_step(store, key, a, m, classification, now_ms)
    if analyses:
        _px = m.spot.last or next(iter(analyses.values())).price
        score_due_calls(store, now_ms, _px)
        score_reports(store, now_ms, _px)
except Exception as e:  # noqa: BLE001 - ledger problems must never blank the page
    st.warning(f"Ledger unavailable this refresh: {e}")

st.sidebar.markdown(f"<span class='ti-chip'>ledger: {store.path}</span>", unsafe_allow_html=True)
st.sidebar.caption("Anchors and accuracy persist in SQLite. On Streamlit Cloud this resets on reboot until a hosted database is configured.")

# ---------- header ----------
st.markdown("<p class='ti-title'>BTC / USD · Trap Intelligence Terminal</p>"
            "<p class='ti-sub'>Deterministic market structure per timeframe. Director report and desk briefs arrive with Run Analysis (Phase 2).</p>",
            unsafe_allow_html=True)
if not m.spot.available:
    st.error(f"Spot data unavailable from {m.spot.source}: {m.spot.error}. Nothing to analyse.")

tab_labels = [c.label for c in CATEGORIES.values()] + ["Active Trade", "War Room"]
tabs = st.tabs(tab_labels)


def category_tab(tab, key):
    with tab:
        a = analyses.get(key)
        if a is None:
            st.info(f"Not enough {CATEGORIES[key].chart_tf} history to analyse this category.")
            return
        kpi_strip(panels.kpis_for(a, m))
        if key in ("weekly", "monthly"):
            last_sig = store.last_signal(key, before_ms=now_ms)
            banner = panels.early_warning_html(early_warnings(a, m.futures, last_sig["squeeze_score"] if last_sig else None))
            if banner:
                panels.render(banner)
        render_chart(a, dark)
        panels.render(panels.layman_html(a))
        c1, c2 = st.columns(2)
        with c1:
            v = verdicts.get(key)
            panels.render(panels.anchored_direction_html(a, v, now_ms) if v else panels.direction_html(a))
            panels.render(panels.volatility_html(a))
        with c2:
            panels.render(panels.divergence_html(a))
            panels.render(panels.agent_summary_html(a, report))
        if key in ("weekly", "monthly"):
            panels.render(panels.calendar_html(m, now_ms))


for tab, key in zip(tabs[:4], CATEGORIES.keys()):
    category_tab(tab, key)

# ---------- active trade ----------
with tabs[4]:
    live, intraday = analyses.get("live"), analyses.get("intraday")
    spot = m.spot.last or (live.price if live else 0.0)
    with st.form("trade"):
        c1, c2, c3, c4, c5 = st.columns(5)
        t_dir = c1.selectbox("Direction", ["LONG", "SHORT"])
        t_entry = c2.number_input("Entry ($)", value=float(round(spot, 2)), step=10.0)
        t_lev = c3.number_input("Leverage (x)", min_value=1.0, max_value=125.0, value=10.0, step=1.0)
        t_sl = c4.number_input("Stop loss ($)", value=float(round(spot * 0.98, 2)), step=10.0)
        t_tp = c5.number_input("Take profit ($)", value=float(round(spot * 1.04, 2)), step=10.0)
        go = st.form_submit_button("Run stress-test", width="stretch")
    if go:
        tm = map_trade({"dir": t_dir, "entry": t_entry, "lev": t_lev, "sl": t_sl, "tp": t_tp}, live, intraday, m.futures)
        panels.render(panels.trade_result_html(tm))
    else:
        st.caption("Enter the position and run the stress-test. Structures come from the 15m and 1h EMAs and swing levels.")

# ---------- war room ----------
with tabs[5]:
    if report is not None:
        panels.render(panels.report_stats_html(report))
        md = to_markdown(report)
        st.download_button("Download report (.md)", md, file_name=f"{report.report_id}.md", mime="text/markdown")
        st.markdown(md)
        with st.expander("Specialist briefs"):
            for b in report.briefs:
                st.markdown(f"**{b.agent_id} · {b.desk} · {b.domain} · conviction {b.conviction:.0f}/10**  \n{b.interpretation}")
                if b.traps_detected:
                    st.markdown("Traps: " + "; ".join(b.traps_detected))
                st.markdown(f"Key risk to thesis: {b.key_risk_to_thesis}")
                if b.data_gaps:
                    st.caption("Data gaps: " + ", ".join(b.data_gaps))
                st.markdown("---")
    else:
        hint = ("Click Run Analysis in the sidebar to generate the Director's report." if settings
                else "Add GEMINI_API_KEY or OPENROUTER_API_KEY to secrets to enable Run Analysis.")
        panels.render(f"<div class='ti-card'><h4>Trap Intelligence Report</h4><p class='muted'>{hint} "
                      "Section 1 of that report, the raw metric snapshot, is live below.</p></div>")
    rows = panels.raw_metrics_rows(m)
    st.dataframe(pd.DataFrame(rows, columns=["Metric", "Value", "Source"]), width="stretch", hide_index=True)
    try:
        panels.render(panels.accuracy_html(summary(store)))
        log_rows = store.anchor_log(50)
    except Exception as e:  # noqa: BLE001
        st.warning(f"Accuracy ledger unavailable: {e}")
        log_rows = []
    with st.expander("Audit ledger — anchor changes (persistent)"):
        if log_rows:
            df_log = pd.DataFrame([{
                "time_utc": datetime.fromtimestamp(r["ts_ms"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
                "category": r["category"], "from": r["from_dir"] or "—", "to": r["to_dir"],
                "price": r["price"], "triggers": ", ".join(r["triggers"]),
            } for r in log_rows])
            st.dataframe(df_log, width="stretch", hide_index=True)
        else:
            st.caption("No anchor changes recorded yet.")
