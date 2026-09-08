"""Trap Intelligence - BTC market-surveillance terminal (Phase 1: deterministic layer)."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from core.config import CATEGORIES, MACRO_EVENTS, TTL
from core.data.market import load_market
from core.indicators.category import analyze_category
from core.indicators.trade_map import map_trade
from ui import panels
from ui.charts import render_chart
from ui.theme import inject_css, kpi_strip

st.set_page_config(page_title="Trap Intelligence | BTC", layout="wide", initial_sidebar_state="expanded")


@st.cache_data(ttl=TTL["spot"], show_spinner=False)
def _market():
    return load_market()


# ---------- sidebar ----------
st.session_state.setdefault("dark", False)
st.session_state.setdefault("audit", [])
st.session_state.setdefault("last_dir", {})

st.sidebar.markdown("<p class='ti-title'>Trap Intelligence</p><p class='ti-sub'>BTC market surveillance</p>", unsafe_allow_html=True)
dark = st.sidebar.toggle("Dark mode", value=st.session_state["dark"])
st.session_state["dark"] = dark
inject_css(dark)

if st.sidebar.button("Refresh data", width="stretch"):
    _market.clear()
    st.rerun()
st.sidebar.button("Run Analysis", width="stretch", disabled=True,
                  help="The 13-agent Trap Intelligence report is delivered in Phase 2.")

with st.spinner("Loading market data"):
    m = _market()

st.sidebar.markdown("---")
st.sidebar.markdown(panels.data_status_html(m), unsafe_allow_html=True)
st.sidebar.caption(f"Updated {m.generated_at.astimezone(timezone.utc):%H:%M:%S} UTC · spot cache {TTL['spot']}s")

# ---------- analyses ----------
analyses = {}
if m.spot.available:
    for key, cat in CATEGORIES.items():
        a = analyze_category(cat, m.spot.frames, m.futures)
        if a is not None:
            analyses[key] = a
            prev = st.session_state["last_dir"].get(key)
            if prev and prev != a.direction.direction:
                st.session_state["audit"].append({
                    "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
                    "category": cat.label, "from": prev, "to": a.direction.direction,
                    "price": round(a.price, 2), "reason": "; ".join(a.direction.drivers[:2]),
                })
            st.session_state["last_dir"][key] = a.direction.direction

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
        banner = panels.squeeze_banner_html(a) if key in ("weekly", "monthly") else None
        if banner:
            panels.render(banner)
        render_chart(a, dark)
        c1, c2 = st.columns(2)
        with c1:
            panels.render(panels.direction_html(a))
            panels.render(panels.volatility_html(a))
        with c2:
            panels.render(panels.divergence_html(a))
            panels.render(panels.agent_placeholder_html(a))
        if key in ("weekly", "monthly"):
            panels.render(panels.macro_html(MACRO_EVENTS))


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
    panels.render("<div class='ti-card'><h4>Trap Intelligence Report</h4><p class='muted'>Phase 2 wires the Director, the Bullish and Bearish desks, "
                  "the volatility agent and the accuracy agent to the Run Analysis button. Section 1 of that report, the raw metric snapshot, is live below.</p></div>")
    rows = panels.raw_metrics_rows(m)
    st.dataframe(pd.DataFrame(rows, columns=["Metric", "Value", "Source"]), width="stretch", hide_index=True)
    with st.expander("Audit ledger — direction changes this session"):
        if st.session_state["audit"]:
            st.dataframe(pd.DataFrame(st.session_state["audit"]), width="stretch", hide_index=True)
        else:
            st.caption("No direction changes recorded yet in this session.")
