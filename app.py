"""Trap Intelligence - BTC market-surveillance terminal."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import html
import time
from datetime import datetime, timezone

import pathlib
import sys
import types

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


def _drop_stale_modules() -> None:
    """Streamlit Cloud re-runs a freshly pulled app.py inside the process that loaded the previous revision,
    so `core.*` and `ui.*` stay in memory at their old versions. A name added in the new revision then fails
    to import (ImportError on ui.theme.md_safe, AttributeError on panels.patterns_html, a Store missing new
    methods) until someone reboots the app. When any project source file has changed since this process
    loaded it, drop those modules and the caches built from them, so the imports below load the new code."""
    root = pathlib.Path(__file__).resolve().parent
    fingerprint = max(p.stat().st_mtime for pkg in ("core", "ui") for p in (root / pkg).rglob("*.py"))
    deploy = sys.modules.setdefault("_ti_deploy", types.ModuleType("_ti_deploy"))
    previous = getattr(deploy, "fingerprint", None)
    deploy.fingerprint = fingerprint
    if previous is None or previous == fingerprint:
        return                        # first run in this process, or nothing changed: nothing can be stale
    for name in [n for n in sys.modules if n.split(".")[0] in ("core", "ui")]:
        del sys.modules[name]
    st.cache_data.clear()             # cached objects were built by the old code too
    st.cache_resource.clear()


_drop_stale_modules()

from core.accuracy import majority_direction, record_report, score_due_calls, score_reports, summary  # noqa: E402
from core.agents.llm import LLMClient
from core.agents.report import to_markdown
from core.agents.runner import run_pipeline_sync
from core.agents.settings import load_settings
from core.agents.judge import Judge
from core.agents.deck import sub_agent_deck
from core.agents.trap_desk import judge_candidate, record, settle
from core.agents.trade_desk import judge_trade
from core.alerts import Dispatcher, Event, detect_events
from core.indicators.fib_time import fib_time_zones
from core import trades
from core.traps import find_candidates
from core.agents.live_recon import build_state, run_live_recon
from core.agents import handshake
from core.agents.recon_profiles import LIVE, PROFILES
from core.anchors import anchor_step
from core import live_accuracy, live_anchor
from core.config import CATEGORIES, MACRO_EVENTS, TTL
from core.data.market import assemble, load_context, load_spot
from core.derived import enrich_futures, record_sample
from core.indicators.category import analyze_category
from core.indicators.early_warning import early_warnings
from core.store import Store
from core.indicators.trade_map import map_trade
from ui import panels
from core.trades import fmt_when
from ui.charts import render_chart
from ui.nav import pick_from_strip
from ui.theme import inject_css, kpi_strip, md_safe

st.set_page_config(page_title="Trap Intelligence | BTC", layout="wide", initial_sidebar_state="auto")


@st.cache_data(ttl=TTL["spot"], show_spinner=False)
def _spot():
    return load_spot()


@st.cache_data(ttl=TTL["context"], show_spinner=False)
def _context():
    return load_context()


# Streamlit Cloud reloads the script on deploy but keeps cached resources, so a Store built from the
# previous revision survives and is missing whatever that revision did not have. Keying the cache on a
# version makes a deploy that changes core.store hand back a fresh Store instead of a stale one.
STORE_VERSION = 8   # bump whenever core.store gains tables or methods, or changes how it matches rows


@st.cache_resource(show_spinner=False)
def _store(version: int = STORE_VERSION):
    return Store()


def _live_store():
    """A Store that definitely understands the current schema, even behind a stale cache.

    The version has to be passed, not defaulted: Streamlit hashes the arguments a call actually makes, so
    `_store()` produced a cache key that never mentioned STORE_VERSION and a bump changed nothing. The
    hasattr probe stays as a second line of defence for a revision that forgets to bump."""
    s = _store(STORE_VERSION)
    if not hasattr(s, "open_trade"):           # cached from an older revision
        _drop_store(s)
        s = _store(STORE_VERSION)
    return s


def _drop_store(s) -> None:
    """Close the connection before dropping the cache entry, or each deploy leaks an open sqlite file."""
    try:
        s.close()
    except Exception:  # noqa: BLE001 - a store we are throwing away must not break the page
        pass
    _store.clear()


# ---------- sidebar ----------
st.session_state.setdefault("dark", False)
st.session_state.setdefault("signal", False)

st.sidebar.markdown("<p class='ti-title'>Trap Intelligence</p><p class='ti-sub'>BTC market-surveillance terminal</p>", unsafe_allow_html=True)
# filled once the desks have a read; created here so it sits at the top of the rail
rail_slot = st.sidebar.container()
dark = st.sidebar.toggle("Dark mode", value=st.session_state["dark"])
st.session_state["dark"] = dark
# the design is monochrome by default (hollow steel up, solid ink down); this restores green and red
signal = st.sidebar.toggle("Signal colours", value=st.session_state["signal"],
                           help="Green and red for up and down, instead of the monochrome steel and ink.")
st.session_state["signal"] = signal
inject_css(dark, signal)
top = st.container()
with top:
    _price_col, _auto_col, _refresh_col, _run_col = st.columns([4.2, 3, 1.5, 1.7], vertical_alignment="bottom")
price_slot = _price_col.empty()
# the dealer gamma regime: the market condition every desk below trades inside
gamma_slot = st.empty()
# phones get the screens as a strip under the header (the rail is hidden in the collapsed sidebar there)
nav_slot = top.container(key="ti_mobile_nav")

if _refresh_col.button("Refresh data", width="stretch"):
    _spot.clear()
    _context.clear()
    st.rerun()

AUTO_OPTIONS = (0, 30, 60, 120)
st.session_state.setdefault("auto_refresh", 0)
auto_every = _auto_col.segmented_control(
    "Auto-refresh", AUTO_OPTIONS, default=st.session_state["auto_refresh"], key="auto_refresh_ctl",
    format_func=lambda v: "Off" if v == 0 else f"{v}s",
    help="Reruns the page on a timer. Data caches still apply (spot 25s, context 120s), so this adds no API load.",
) or 0
st.session_state["auto_refresh"] = auto_every
st.session_state["last_full_run_ms"] = int(time.time() * 1000)

if auto_every:
    @st.fragment(run_every=f"{auto_every}s")
    def _auto_tick():
        elapsed = int(time.time() * 1000) - st.session_state.get("last_full_run_ms", 0)
        if elapsed >= auto_every * 1000 - 500:
            st.rerun(scope="app")
        st.caption(f"Auto-refresh every {auto_every}s · last full update {elapsed // 1000}s ago")

    with st.sidebar:
        _auto_tick()
settings = load_settings()
st.session_state.setdefault("trap_report", None)
run_clicked = _run_col.button(
    "Run Analysis", type="primary", width="stretch", disabled=settings is None,
    help=(f"13-agent Trap Intelligence report via {settings.provider}: specialists on {settings.specialist_model}, Director on {settings.director_model}."
          if settings else "Add GEMINI_API_KEY or OPENROUTER_API_KEY to .streamlit/secrets.toml (or Streamlit Cloud Secrets) to enable."),
)

with st.spinner("Loading market data"):
    m = assemble(_spot(), _context())
try:
    _store_early = _live_store()
    _now_early = int(time.time() * 1000)
    record_sample(_store_early, m.futures, _now_early)
    m = m.model_copy(update={"futures": enrich_futures(_store_early, m.futures, _now_early)})
except Exception as e:  # noqa: BLE001 - derived fields are optional
    st.sidebar.caption(f"Derived futures fields unavailable: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown(panels.data_status_html(m), unsafe_allow_html=True)
with st.sidebar:
    # a ticking component, not a rendered string: the page only redraws on a rerun
    components.html(panels.clock_html(dark), height=48)
st.sidebar.caption(f"Updated {fmt_when(int(m.generated_at.timestamp() * 1000), seconds=True)} · spot {TTL['spot']}s · context {TTL['context']}s")

# ---------- analyses ----------
analyses = {}
if m.spot.available:
    for key, cat in CATEGORIES.items():
        a = analyze_category(cat, m.spot.frames, m.futures)
        if a is not None:
            analyses[key] = a
store = _live_store()
now_ms = int(time.time() * 1000)

def _macro_event_now(m, now_ms: int, profile=LIVE) -> str | None:
    """A high-impact USD release landing inside this category's candle, for the side-note check."""
    if not m.calendar.available:
        return None
    due = m.calendar.upcoming(now_ms, profile.window_close(now_ms) - now_ms)
    return due[0].get("title") if due else None


def sweep_plan() -> list[tuple[str, str]]:
    """Which desk judges which timeframe, so a trap is scanned once rather than once per desk.

    The desks' timeframes overlap (Live reads 15m/1h/4h, Intraday 1h/4h/1d, Weekly 4h/1d/1w, Monthly
    1d/1w/1M), and scanning per desk found the same shape twice, paid for two judgements of it, and
    anchored two rows for one raid. A timeframe belongs to the first desk that reads it - the shortest
    horizon that can see the level, and so the desk that will act on it soonest - which gives 15m/1h/4h
    to Live, 1d to Intraday, 1w to Weekly and the monthly candle to Monthly. Every desk still shows every
    trap on the timeframes it reads."""
    plan, claimed = [], set()
    for cat, profile in PROFILES.items():
        for tf in profile.timeframes:
            if tf not in claimed:
                claimed.add(tf)
                plan.append((cat, tf))
    return plan


def run_trap_sweep() -> None:
    """Scan every timeframe once, then judge the strongest candidate on each desk that owns one.

    The scan is free arithmetic; only the strongest candidate per owning desk goes to the agents. Each
    desk's result is committed to session state as it completes, so a failure part-way through keeps the
    desks that already succeeded - their traps are in the database by then either way."""
    run_ms = int(time.time() * 1000)          # not the page-load time: the judgements take round-trips
    results: list[dict] = []
    st.session_state["trap_sweep"] = {"at": run_ms, "results": results, "complete": False}
    plan = sweep_plan()
    with st.status("TRAP desk: sweeping every timeframe...", expanded=False) as status:
        try:
            judge = Judge(settings, typesafe_key=settings.typesafe_key, llm=LLMClient(settings))
            for cat, timeframes in _grouped(plan):
                profile = PROFILES[cat]
                state = _desk_state(cat, _data_stamp())
                price = m.spot.last
                candidates = find_candidates(state, timeframes)
                entry = {"category": cat, "timeframes": timeframes, "candidates": candidates,
                         "verdict": None}
                results.append(entry)
                if not candidates:
                    continue
                verdict = asyncio.run(judge_candidate(judge, state, candidates[0], profile))
                entry["verdict"] = verdict
                trap_id = record(store, verdict, cat, int(time.time() * 1000))
                if trap_id is not None:
                    c = verdict.candidate
                    store.add_side_note(
                        profile.window_open(run_ms), run_ms, "TRAP", f"trap:{trap_id}",
                        f"TRAP desk: {c.label} anchored at {c.level:,.0f} on the {c.timeframe} chart.",
                        f"Invalidated at {c.invalidation:,.0f}; pays off at {c.plays_out:,.0f}. "
                        + "; ".join(c.evidence), price, category=cat)
            declared = sum(1 for r in results if r["verdict"] is not None and r["verdict"].declared)
            watching = sum(len(r["candidates"]) for r in results)
            st.session_state["trap_sweep"]["complete"] = True
            status.update(label=f"Sweep complete - {declared} declared, {watching} candidate(s) on watch",
                          state="complete")
        except Exception as e:  # noqa: BLE001 - a sweep failure must not crash the page
            done = sum(1 for r in results if r["verdict"] is not None)
            status.update(label=f"Trap sweep failed after {done} desk(s): {type(e).__name__}: {e}",
                          state="error")
            st.error(f"The trap sweep stopped after {done} desk(s): {e}. Any trap already declared is "
                     "in the ledger below; the desks that did not run are missing from it.")


def _grouped(plan: list[tuple[str, str]]) -> list[tuple[str, tuple[str, ...]]]:
    """The plan as one entry per desk, so each desk's state is built and judged once."""
    out: dict[str, list[str]] = {}
    for cat, tf in plan:
        out.setdefault(cat, []).append(tf)
    return [(cat, tuple(tfs)) for cat, tfs in out.items()]


def run_war_room(profile) -> None:
    """Both teams and the trap audit for one category, then anchor or append side notes."""
    with st.status(f"{profile.label} war room: both teams and the trap audit...", expanded=False) as status:
        try:
            judge = Judge(settings, typesafe_key=settings.typesafe_key, llm=LLMClient(settings))
            recon = asyncio.run(run_live_recon(judge, m, analyses, profile=profile))
            run_ms = int(time.time() * 1000)
            # the cross-desk handshake: ask the other three heads before conviction is final
            try:
                recon = handshake.apply(recon, handshake.consult(store, profile.category, recon.bias, run_ms))
            except Exception as e:  # noqa: BLE001 - an unreadable ledger leaves the war room's own conviction
                recon = replace(recon, notes=tuple(recon.notes) + (f"Cross-desk handshake unavailable: {e}",))
            published = live_anchor.publish(store, recon, run_ms, price=m.spot.last,
                                            macro_event=_macro_event_now(m, run_ms, profile), profile=profile)
            if published.anchored:
                # and the other direction: desks this verdict now opposes get a side note
                try:
                    handshake.notify(store, profile.category, recon.bias, run_ms, price=m.spot.last)
                except Exception as e:  # noqa: BLE001
                    st.warning(f"Could not notify the other desks: {e}")
            st.session_state["recon"][profile.category] = (recon, published)
            label = (f"Anchored the {profile.window_label} summary" if published.anchored else
                     f"Summary held; {len(published.appended)} side note(s) appended")
            status.update(label=f"{label} - {recon.bias} via {judge.provider}", state="complete")
        except Exception as e:  # noqa: BLE001 - never crash the page on a judge failure
            status.update(label=f"War room failed: {e}", state="error")


# ---------- run analysis ----------
if run_clicked and settings is not None and analyses:
    with st.status("Running Trap Intelligence pipeline...", expanded=True) as status:
        lines = st.empty()
        seen: dict[str, str] = {}

        def _progress(agent_id: str, state: str) -> None:
            seen[agent_id] = state
            done = sum(1 for v in seen.values() if v in ("ok", "done", "failed", "error"))
            lines.markdown(f"<p>{done} of {len(seen)} agents answered. Specialists run in parallel; the desks, "
                           "the accuracy agent and the Director follow.</p><div class='ti-agents'>"
                           + "".join(f"<div><strong>{html.escape(k)}</strong><span class='muted'>"
                                     f"{html.escape(str(v))}</span></div>" for k, v in seen.items())
                           + "</div>", unsafe_allow_html=True)

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

# ---------- war rooms ----------
if not isinstance(st.session_state.get("recon"), dict):
    st.session_state["recon"] = {}

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

# settle anchored traps the price has now answered, before anything is rendered
st.session_state.setdefault("trap_sweep", None)
try:
    _px = m.spot.last or (next(iter(analyses.values())).price if analyses else None)
    if _px:
        _settled = settle(store, float(_px), now_ms)
        for _t in _settled:
            log_line = f"{_t['side']} {_t['timeframe']} {_t['status'].replace('_', ' ')}"
            st.sidebar.caption(f"TRAP: {log_line}")
    elif store.traps(status="active", limit=1):
        # silence here would leave anchored traps rendering as live with no hint that nothing is
        # checking them against price any more
        st.warning("No price this refresh, so anchored traps were not checked for settlement. "
                   "The levels below are as of the last successful refresh.")
except Exception as e:  # noqa: BLE001 - trap bookkeeping must never blank the page
    st.warning(f"Anchored traps could not be checked against price this refresh ({e}); "
               "they may already have been invalidated or paid off.")

# score every war-room window whose candle has closed, once, against the price at that close
try:
    for _profile in PROFILES.values():
        live_accuracy.score_due(store, m.spot.frames, now_ms, _profile)
except Exception as e:  # noqa: BLE001 - scoring problems must never blank the page
    st.warning(f"Accuracy audit unavailable this refresh: {e}")

st.sidebar.markdown(f"<span class='ti-chip'>ledger: {store.path}</span>", unsafe_allow_html=True)
st.sidebar.caption("Anchors and accuracy persist in SQLite. On Streamlit Cloud this resets on reboot until a hosted database is configured.")

# ---------- top bar ----------
_live_a = analyses.get("live")
price_slot.markdown(panels.top_bar_html(
    m.spot.last if m.spot.last is not None else (_live_a.price if _live_a else None),
    _live_a.price_change_24h_pct if _live_a else None,
    f"Updated {fmt_when(int(m.generated_at.timestamp() * 1000), seconds=True)}"), unsafe_allow_html=True)
gamma_slot.markdown(panels.gamma_bar_html(m.options.gamma if m.options.available else None, m.spot.last),
                    unsafe_allow_html=True)
if not m.spot.available:
    st.error(f"Spot data unavailable from {m.spot.source}: {m.spot.error}. Nothing to analyse.")

# ---------- the rail: four desks, then four engines; only the open screen renders ----------
ENGINES = {"trap": "TRAP Intelligence", "trade": "Active Trade", "director": "Director report", "macro": "Macro & News"}
SCREENS = list(CATEGORIES) + list(ENGINES)
_labels = {**{k: c.label for k, c in CATEGORIES.items()}, **ENGINES}
_captions = ([panels.rail_read(verdicts.get(k), analyses.get(k)) for k in CATEGORIES]
             + ["Sweep, anchored traps, watch list", "Positions, stops and targets",
                f"Report {report.report_id}" if report is not None else "Run Analysis to generate",
                "Headlines, gold, rates, odds"])
st.session_state.setdefault("screen", SCREENS[0])


def _from_rail():
    st.session_state["screen_nav"] = st.session_state["screen"]


def _from_strip():
    st.session_state["screen"], st.session_state["screen_nav"] = pick_from_strip(
        st.session_state["screen"], st.session_state.get("screen_nav"))


st.session_state.setdefault("screen_nav", st.session_state["screen"])
with nav_slot:
    st.pills("Screen", SCREENS, format_func=_labels.get, key="screen_nav", on_change=_from_strip,
             label_visibility="collapsed")
with rail_slot:
    screen = st.radio("Desks", SCREENS, format_func=_labels.get, captions=_captions, key="screen",
                      on_change=_from_rail)
    panels.render(panels.rail_style_html(
        [panels.rail_tone(verdicts.get(k), analyses.get(k)) for k in CATEGORIES],
        [CATEGORIES[k].chart_tf for k in CATEGORIES], dark, signal))


def _data_stamp() -> str:
    """A fingerprint of the snapshot this run is rendering.

    `_desk_state` reads module globals that are rebound on every script run, so keying its cache on the
    clock alone handed back a state built from an older snapshot: after a manual Refresh the header
    showed the new price while the volatility, structure and liquidity panels - and the trap scan -
    still read the old one. Keying on the data means the cache follows the snapshot, not the minute."""
    last = getattr(m.spot, "last", None)
    stamps = [str(last)]
    for tf in ("15m", "1h", "4h", "1d", "1w", "2w", "1M", "3M"):
        df = m.spot.frames.get(tf)
        stamps.append(str(int(df["timestamp"].iloc[-1])) if df is not None and not df.empty else "-")
    return "|".join(stamps)


@st.cache_data(ttl=TTL["spot"], show_spinner=False)
def _desk_state(category: str, data_stamp: str) -> dict:
    """The very state the desk's judges are handed, so the screen and the scorecard cannot drift
    apart. Cached per desk per snapshot; a war-room run builds its own state at the moment it runs."""
    return build_state(m, analyses, now_ms, PROFILES[category])


PROTOCOLS = ("Volatility", "Structure", "Liquidity", "Quant matrix")


def _open_screen(screen: str) -> None:
    """Jump to a screen from inside another one; the rail and the phone strip follow."""
    st.session_state["screen"] = st.session_state["screen_nav"] = screen


@st.cache_data(ttl=TTL["spot"], show_spinner=False)
def _desk_cascade(category: str, data_stamp: str):
    """The desk's cascade, cached per snapshot like the judges' state, and built by the same function."""
    from core.indicators.cascade import classify
    return classify(m.spot.frames, *PROFILES[category].cascade)


@st.cache_data(ttl=TTL["spot"], show_spinner=False)
def _desk_breaks(category: str, data_stamp: str):
    """Qualified value-area breaks on the desk's own 1H/4H/1D, the same reads the judges get."""
    from core.indicators.vwap_break import read_all
    return read_all(m.spot.frames, PROFILES[category].timeframes)


@st.cache_data(ttl=TTL["spot"], show_spinner=False)
def _event_reads(data_stamp: str):
    """Positioning into and reactions to this week's high-impact USD prints, the same read the judges get."""
    from core.indicators.event_trap import read_all
    return read_all(m.spot.frames.get("1h"), m.calendar.events if m.calendar.available else [], now_ms)


def render_protocols(profile) -> None:
    """The deterministic reads - volatility, structure, liquidity - and each sub-agent's own deck, one at
    a time behind a switch, as the design lays them out."""
    try:
        state = _desk_state(profile.category, _data_stamp())
    except Exception as e:  # noqa: BLE001 - a protocol failure must not blank the tab
        st.warning(f"Protocols unavailable this refresh: {e}")
        return
    # every trap on a timeframe this desk reads, not only the ones it declared itself
    try:
        rows = store.traps(timeframes=profile.timeframes, status="active")
    except Exception as e:  # noqa: BLE001 - the ledger must not blank the tab
        rows = []
        st.warning(f"Trap ledger unavailable: {e}")
    if rows:
        # rendered outside the query's guard: a warning that vanishes silently is worse than none, so a
        # row that will not render is reported as a broken row rather than as no trap at all
        try:
            warning = panels.trap_warning_html(rows, now_ms)
            if warning:
                panels.render(warning)
                st.button("Open TRAP desk →", key=f"open_trap_{profile.category}", on_click=_open_screen,
                          args=("trap",))
        except Exception as e:  # noqa: BLE001
            st.error(f"{len(rows)} trap(s) are anchored on this desk but could not be drawn "
                     f"({type(e).__name__}: {e}). Treat the levels as live and check the TRAP tab.")
    panels.render(panels.section_head_html("Protocols"))
    head, pick = st.columns([3, 2], vertical_alignment="bottom")
    head.markdown("<h2 class='ti-h2'>Deterministic reads</h2>", unsafe_allow_html=True)
    key = f"proto_{profile.category}"
    st.session_state.setdefault(key, PROTOCOLS[0])
    choice = pick.segmented_control("Protocol", PROTOCOLS, key=key, label_visibility="collapsed") or PROTOCOLS[0]
    if choice == "Volatility":
        panels.render(panels.vol_matrix_html(state.get("volatility", {})))
    elif choice == "Structure":
        panels.render(panels.smc_html(state.get("smc", {})))
    elif choice == "Liquidity":
        panels.render(panels.liquidity_html(state.get("liquidity", {})))
    else:
        panels.render(panels.agent_matrix_html(sub_agent_deck(state, profile),
                                               f"Institutional quantitative matrix — {profile.label} sub-agents"))


def render_war_room(profile) -> None:
    """A category's anchored summary, both teams, the trap audit, the side notes and the accuracy report."""
    cat = profile.category
    panels.render(panels.section_head_html("War room"))
    head, btn = st.columns([3, 1], vertical_alignment="bottom")
    head.markdown(f"<h2 class='ti-h2'>{profile.label} war room</h2>", unsafe_allow_html=True)
    if btn.button(f"Run {profile.label} war room", key=f"war_room_{cat}", disabled=settings is None,
                  width="stretch",
                  help=f"Eleven agents on {', '.join(profile.timeframes)} over the next {profile.horizon}."):
        run_war_room(profile)
        st.rerun()
    st.caption("Five bullish and five bearish domain agents score a 100-point checklist; the Head consults a "
               "trap agent before publishing. A direction needs a 10-point lead.")

    w_open, w_close = profile.window_open(now_ms), profile.window_close(now_ms)
    try:
        row = store.get_live_anchor(w_open, category=cat)
        notes = store.side_notes(w_open, category=cat) if row else []
    except Exception as e:  # noqa: BLE001 - the ledger must never blank the tab
        st.warning(f"Anchor ledger unavailable: {e}")
        return
    try:
        accuracy = panels.accuracy_report_html(live_accuracy.diagnose(store, profile))
    except Exception as e:  # noqa: BLE001 - the audit must never blank the tab
        accuracy = None
        st.warning(f"Accuracy audit unavailable: {e}")

    pair = st.session_state["recon"].get(cat)
    recon = pair[0] if pair and pair[1].window_open_ms == w_open else None
    anchored_now = bool(recon is not None and pair[1].anchored)
    closes = datetime.fromtimestamp(w_close / 1000, timezone.utc)
    closes_txt = fmt_when(w_close)
    if row is None:
        panels.render(panels.card_html(
            f"{profile.window_title} anchored summary",
            f"<p class='muted'>Nothing anchored for the current {profile.window_label} candle yet. Run the "
            f"{profile.label} war room to publish the summary for the window closing {closes_txt}. "
            "It then stays pinned until that close; re-runs only add side notes.</p>"))
        if accuracy:
            panels.render(accuracy)
        return

    payload = row["payload"]
    panels.render(panels.anchored_summary_html(payload, w_close, now_ms, anchored_now, profile))
    # the teams as pinned; an interim re-run this session shows its own scorecards instead, labelled so
    c1, c2 = st.columns(2, gap="large")
    with c1:
        panels.render(panels.team_html(recon.bull) if recon else panels.payload_team_html("bullish", payload))
    with c2:
        panels.render(panels.team_html(recon.bear) if recon else panels.payload_team_html("bearish", payload))
    t1, t2, t3 = st.columns(3, gap="large")
    with t1:
        panels.render(panels.war_room_html(recon) if recon else panels.trap_audit_html(
            payload.get("trap"), payload.get("trap_probabilities") or {}, bool(payload.get("consulted")),
            payload.get("override"), None if payload.get("trap") or payload.get("trap_probabilities") else
            "the trap agent did not answer in the pinned run"))
    with t2:
        panels.render(panels.side_notes_html(notes))
    with t3:
        if accuracy:
            panels.render(accuracy)
    if recon is not None:
        st.caption(f"{'Pinned' if anchored_now else 'Interim'} run {fmt_when(int(recon.generated_at.timestamp() * 1000), seconds=True)} · "
                   f"{recon.bull.provider or 'n/a'} · {recon.prompt_tokens + recon.completion_tokens:,} tokens")


def category_tab(key):
    with st.container():
        a = analyses.get(key)
        if a is None:
            st.info(f"Not enough {CATEGORIES[key].chart_tf} history to analyse this category.")
            return
        panels.render(panels.desk_hero_html(SCREENS.index(key) + 1, PROFILES[key], CATEGORIES[key], now_ms))
        kpi_strip(panels.kpis_for(a, m))
        # the cascade sits in the quick read: bias above, the anchor, the engine below
        try:
            v_ = verdicts.get(key)
            panels.render(panels.cascade_html(
                _desk_cascade(key, _data_stamp()),
                invalidation=v_.anchored.invalidation if v_ is not None else None, price=a.price))
        except Exception as e:  # noqa: BLE001 - a cascade failure must not blank the desk
            st.warning(f"Timeframe cascade unavailable this refresh: {e}")
        try:
            _breaks = _desk_breaks(key, _data_stamp())
            if _breaks:
                panels.render(panels.value_breaks_html(_breaks))
        except Exception as e:  # noqa: BLE001 - nor may a break read
            st.warning(f"Value-area breaks unavailable this refresh: {e}")
        if key in ("weekly", "monthly"):
            last_sig = store.last_signal(key, before_ms=now_ms)
            banner = panels.early_warning_html(early_warnings(a, m.futures, last_sig["squeeze_score"] if last_sig else None))
            if banner:
                panels.render(banner)
        # the chart and the call it supports sit side by side, as the design draws them
        c_chart, c_dir = st.columns([2, 1], gap="large")
        with c_chart:
            with st.container(key="ti_chart"):
                panels.render(panels.chart_head_html(a, signal))
                render_chart(a, dark, height=480, signal=signal)
                panels.render(f"<p class='ti-chart-cap'>{panels.CHART_CAPTION}</p>")
        with c_dir:
            v = verdicts.get(key)
            panels.render(panels.anchored_direction_html(a, v, now_ms) if v else panels.direction_html(a))
        panels.render(panels.layman_html(a))
        c1, c2 = st.columns(2, gap="large")
        with c1:
            panels.render(panels.patterns_html(a))
            panels.render(panels.divergence_html(a))
        with c2:
            panels.render(panels.fib_html(a))
            panels.render(panels.volatility_html(a))
        profile = PROFILES.get(key)
        if profile is not None:
            render_protocols(profile)
            render_war_room(profile)
        panels.render(panels.agent_summary_html(a, report))
        if key in ("live", "intraday"):
            # a print inside these horizons is part of the trade: show it only when there is something to read
            try:
                _active = [r for r in _event_reads(_data_stamp()) if r.state in panels.ACTIVE_EVENT_STATES]
                if _active:
                    panels.render(panels.event_traps_html(_active, now_ms))
            except Exception as e:  # noqa: BLE001
                st.warning(f"Event traps unavailable: {e}")
        if key in ("weekly", "monthly"):
            panels.render(panels.calendar_html(m, now_ms))
            # weekly positioning belongs on the weekly and monthly view only: it is three to ten days
            # old in normal operation, which is meaningless against a four-hour or 24-hour horizon
            try:
                panels.render(panels.cot_html(m.cot, now_ms))
            except Exception as e:  # noqa: BLE001
                st.warning(f"Positioning report unavailable: {e}")


if screen in CATEGORIES:
    category_tab(screen)

# ---------- TRAP intelligence ----------
if screen == "trap":
    sweep = st.session_state.get("trap_sweep")
    hero, act = st.columns([3, 1], vertical_alignment="bottom")
    with hero:
        panels.render(panels.engine_hero_html(
            5, "TRAP Intelligence",
            "The watchdog for engineered moves: fakeouts, swing failures and stop runs. Candidates are found "
            "in code, then judged by ten sub-agents and a Head that consults the affected desk. A trap stays "
            "anchored until price settles it."))
    with act:
        if st.button("Sweep every timeframe", key="trap_sweep_btn", type="primary", width="stretch",
                     disabled=settings is None,
                     help="Deterministic scan of every timeframe once; the strongest candidate on each "
                          "owning desk goes to the agents."):
            # deliberately no st.rerun(): it discards the status box, so a sweep that failed on desk three
            # came back looking exactly like a page that had done nothing. The panels below render from
            # session state in this same run.
            run_trap_sweep()
            sweep = st.session_state.get("trap_sweep")
        st.caption((f"Last sweep {panels._trap_age(sweep['at'], now_ms)} ago" if sweep and sweep.get("at")
                    else "Not swept this session") + " · at most six requests")

    try:
        # anchored traps are what a trader acts on, so they are never pushed off the list by newer
        # settled ones: the two are queried separately rather than sliced out of one ordered list
        active = store.traps(status="active", limit=25)
        settled = [r for r in store.traps(limit=50) if r.get("status") != "active"][:6]
    except Exception as e:  # noqa: BLE001 - never crash the tab on ledger trouble
        active, settled = [], []
        st.error(f"The trap ledger could not be read ({type(e).__name__}: {e}). "
                 "Anchored traps are not shown below; do not read this as 'no traps'.")

    labels = {k: c.label for k, c in CATEGORIES.items()}
    watching = [c for r in (sweep or {}).get("results", []) for c in r["candidates"]]
    try:
        panels.render(panels.trap_console_html(active, watching, now_ms, labels, settled=settled,
                                              swept_ms=(sweep or {}).get("at"), plan=sweep_plan()))
    except Exception as e:  # noqa: BLE001
        st.error(f"{len(active)} anchored trap(s) exist but the console could not be drawn "
                 f"({type(e).__name__}: {e}).")

    panels.render(panels.section_head_html("Anchored"))
    if not active:
        st.caption("Nothing anchored.")
    _px = m.spot.last
    cols = st.columns(2, gap="large")
    for i, row in enumerate(active):
        # per row, so one malformed trap cannot take the others off the screen with it
        with cols[i % 2]:
            try:
                panels.render(panels.trap_card_html(row, now_ms, labels.get(row["category"], row["category"]), _px))
            except Exception as e:  # noqa: BLE001
                st.warning(f"Trap #{row.get('id', '?')} ({row.get('side', 'unknown')} on "
                           f"{row.get('timeframe', '?')}) could not be drawn: {type(e).__name__}: {e}")

    judged = []
    for r in (sweep or {}).get("results", []):
        v = r["verdict"]
        if v is not None and not v.declared:
            judged.append({"label": f"{labels[r['category']]} · {v.candidate.label} on {v.candidate.timeframe}",
                           "score": v.score, "status": v.status, "why": " · ".join(v.notes[:2])})
    w, se = st.columns(2, gap="large")
    with w:
        panels.render(panels.trap_watch_html(watching, now_ms, (sweep or {}).get("at"), judged))
    with se:
        try:
            panels.render(panels.trap_settled_html(settled, now_ms, labels))
        except Exception as e:  # noqa: BLE001
            st.warning(f"Settled traps could not be drawn: {type(e).__name__}: {e}")
    st.caption("A trap is declared when its five sub-agents score 60 or more, the Head calls it engineered, and "
               "the category desk does not read the move as an authentic break. It is invalidated when the "
               "break proves real and played out when price returns inside value. Nothing expires on a clock.")

# ---------- active trade ----------
def _secrets() -> dict:
    """Streamlit secrets, or an empty mapping when there is no secrets file.

    `hasattr(st, "secrets")` is always True - the attribute exists whether or not a file does - so the
    old guard was dead and `dict(st.secrets)` raised on any deploy or fresh clone without one, aborting
    the script mid-tab and taking every tab after it down with it. core/agents/settings.py already
    handled this correctly; this is the same pattern."""
    try:
        return dict(st.secrets)
    except Exception:  # noqa: BLE001 - not running under streamlit, or no secrets file
        return {}


def _trade_price() -> float | None:
    return m.spot.last or (next(iter(analyses.values())).price if analyses else None)


def build_trade_plan(asset: str, side: str, entry: float, given_stop: float | None, category: str) -> dict:
    """Everything the card needs, computed once at open: the stop, the targets, the guardrail, the plan.

    Deterministic first and agents second, so a card still carries its levels when the model is down."""
    profile = PROFILES[category]
    ltf, mtf, htf = profile.timeframes
    state = _desk_state(category, _data_stamp())
    frame = m.spot.frames.get(mtf)
    atr_value = (state.get("volatility", {}).get(mtf) or {}).get("atr")
    atr = float(atr_value) if atr_value else entry * trades.DEFAULT_ATR_PCT
    run_ms = int(time.time() * 1000)

    stop = trades.verify_stop(state, side, entry, given_stop, mtf, atr)
    fib = fib_time_zones(frame, atr_value=atr) if frame is not None and not frame.empty else None
    targets = trades.timed_targets(state, side, entry, stop.stop, htf, atr, fib, run_ms)
    pullback = trades.pullback_band(frame, side, timeframe=mtf)
    schedule = trades.monitoring_schedule(run_ms, profile.timeframes, targets)

    plan = {
        "asset": asset, "side": side, "entry": entry, "category": category, "timeframe": mtf,
        "horizon": profile.horizon, "atr": round(atr, 2), "built_ms": run_ms,
        "stop_verdict": stop.verdict, "stop_reasons": list(stop.reasons),
        "protects_at": stop.protects_at, "protects_kind": stop.protects_kind,
        "targets": [{"name": t.name, "price": t.price, "basis": t.basis, "due_ms": t.due_ms,
                     "due_from": t.due_from, "reward_r": t.reward_r} for t in targets],
        "pullback": {"available": pullback.available, "note": pullback.note,
                     "ordinary": pullback.ordinary, "edge": pullback.edge, "breaking": pullback.breaking,
                     "horizon_bars": pullback.horizon_bars, "timeframe": pullback.timeframe,
                     "levels": pullback.levels(side, entry) if pullback.available else {}},
        "schedule": [{"timeframe": c.timeframe, "at_ms": c.at_ms, "reason": c.reason} for c in schedule],
        "ok": False, "notes": [],
    }

    if settings is None:
        plan["notes"] = ["No model provider is configured, so the ten sub-agents were not run. The levels "
                         "above are deterministic and stand on their own."]
        return plan, stop, targets

    try:
        judge = Judge(settings, typesafe_key=settings.typesafe_key, llm=LLMClient(settings))
        verdict = asyncio.run(judge_trade(judge, state, side, entry, stop, targets, profile,
                                         store=store, now_ms=run_ms, atr=atr))
    except Exception as e:  # noqa: BLE001 - a card without a score is still a usable plan
        plan["notes"] = [f"The trade desk could not be reached ({type(e).__name__}: {e}). The levels "
                         "above are deterministic and stand on their own."]
        return plan, stop, targets

    plan.update({
        "ok": verdict.ok, "score": verdict.score, "probability": verdict.probability,
        "margin": verdict.margin, "for_points": verdict.for_points,
        "against_points": verdict.against_points,
        "for_domains": [{"key": d.key, "title": d.title, "points": d.points} for d in verdict.for_domains],
        "against_domains": [{"key": d.key, "title": d.title, "points": d.points}
                            for d in verdict.against_domains],
        "head_call": verdict.head_call, "head_confidence": verdict.head_confidence,
        "aligned": verdict.aligned,
        "alignment": [{"category": a.category, "label": a.label, "bias": a.bias, "verdict": a.verdict,
                       "note": a.note} for a in verdict.alignment],
        "traps": [{"trap_id": t.trap_id, "side": t.side, "timeframe": t.timeframe, "level": t.level,
                   "near": t.near, "distance": t.distance} for t in verdict.traps],
        "notes": list(verdict.notes), "provider": verdict.provider,
    })
    return plan, stop, targets


def trade_alert_pass(row: dict, prog) -> None:
    """Detect what has happened to a pinned trade, log it once, and push whatever a channel accepts."""
    plan = row.get("plan") or {}
    checkpoints = [trades.Checkpoint(c.get("timeframe", ""), int(c.get("at_ms") or 0),
                                     str(c.get("reason") or "")) for c in plan.get("schedule") or []]
    traps = [type("T", (), {"trap_id": t.get("trap_id"), "side": t.get("side", ""),
                            "timeframe": t.get("timeframe", ""), "near": t.get("near", ""),
                            "distance": t.get("distance", 0.0)})() for t in plan.get("traps") or []]
    events = detect_events(row, prog.price if prog else None, prog, traps, checkpoints, now_ms)
    for e in events:
        # the unique index decides what is new: the same target reached is one event, not one per refresh
        store.add_trade_event(row["id"], now_ms, e.kind, e.dedup_key, e.headline, e.price)
    # oldest first, with a limit well above one pass: the default hid the oldest pending alert once a
    # backlog passed it, and the oldest is the one most likely to be a stop
    pending = store.trade_events(row["id"], unnotified_only=True, limit=500)
    if not pending:
        return
    dispatcher = Dispatcher(_secrets())
    to_send = [Event(p["kind"], p["dedup_key"], p["headline"], p.get("price")) for p in pending]
    delivered, problems = dispatcher.send(to_send)
    if delivered:
        keys = {e.dedup_key for e in delivered}
        store.mark_notified([p["id"] for p in pending if p["dedup_key"] in keys])
    for note in problems[:1]:
        st.caption(f"Alerts: {note}")


@st.dialog("Kill this trade?")
def _kill_dialog(row: dict, price_now) -> None:
    st.markdown(f"<p>#{row['id']} {html.escape(str(row['side']))} {html.escape(str(row['asset']))} from "
                f"${float(row['entry']):,.0f}. The card is unpinned and every alert for this position stops. "
                "It moves to recently closed.</p>", unsafe_allow_html=True)
    k1, k2 = st.columns(2)
    if k1.button("Keep tracking", key=f"tr_keep_{row['id']}", width="stretch"):
        st.session_state.pop("kill_pending", None)
        st.rerun()
    if k2.button("KILL", key=f"tr_kill_confirm_{row['id']}", type="primary", width="stretch"):
        st.session_state.pop("kill_pending", None)
        store.close_trade(row["id"], "killed by the trader", price_now, now_ms)
        st.rerun()


if screen == "trade":
    panels.render(panels.engine_hero_html(
        6, "Active Trade",
        "Enter a position and it is evaluated, pinned and tracked until you kill it. Levels are computed in "
        "code first; ten sub-agents, the four category heads and the TRAP head are consulted second. Cards "
        "live in the database and survive a refresh and a redeploy."))

    spot = _trade_price()
    with st.form("new_trade", clear_on_submit=False, border=False):
        st.markdown("<span class='ti-kicker'>Open a position</span>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
        f_asset = c1.text_input("Asset", value="BTC/USDT")
        f_side = c2.segmented_control("Direction", ["LONG", "SHORT"], default="LONG", key="tr_side") or "LONG"
        f_entry = c3.number_input("Entry ($) · defaults to spot", value=float(round(spot or 0.0, 2)), step=10.0,
                                  help="Where you actually got in.")
        f_stop = c4.number_input("Stop loss ($) · 0 lets the desk place it", value=0.0, step=10.0,
                                 help="Leave at 0 and the engine derives a structural stop.")
        f_cat = st.segmented_control(
            "Judge it over", list(PROFILES), default="live", key="tr_cat",
            format_func=lambda k: f"{CATEGORIES[k].label} · {PROFILES[k].horizon}") or "live"
        b_col, n_col = st.columns([1, 2], vertical_alignment="center")
        opened = b_col.form_submit_button("Open and track this trade", type="primary", width="stretch",
                                          disabled=not spot)
        n_col.caption("Two batched requests plus one Head call.")
    if opened:
        if not f_entry:
            st.error("An entry price is needed.")
        else:
            with st.status("Active trade desk: ten sub-agents and two consultations...", expanded=False) as s_:
                try:
                    plan, stop_v, _t = build_trade_plan(f_asset.strip() or "BTC/USDT", f_side,
                                                        float(f_entry),
                                                        float(f_stop) if f_stop else None, f_cat)
                    tid = store.open_trade(opened_ms=int(time.time() * 1000), asset=plan["asset"],
                                           side=f_side, entry=float(f_entry), stop=stop_v.stop,
                                           given_stop=float(f_stop) if f_stop else None,
                                           stop_verdict=stop_v.verdict, category=f_cat, plan=plan)
                    s_.update(label=f"Trade #{tid} pinned", state="complete")
                except Exception as e:  # noqa: BLE001 - a failed open must not blank the tab
                    s_.update(label=f"Could not open the trade: {type(e).__name__}: {e}", state="error")
                    st.error(f"The trade was not pinned: {e}")

    try:
        open_trades = store.trades(status="open", limit=10)
        closed_trades = store.trades(status="closed", limit=3)
    except Exception as e:  # noqa: BLE001
        open_trades, closed_trades = [], []
        st.error(f"The trade ledger could not be read ({type(e).__name__}: {e}). "
                 "Do not read this as 'no open positions'.")

    if not open_trades:
        st.caption("No position is being tracked. Open one above and it stays pinned here.")

    price_now = _trade_price()
    for row in open_trades:
        plan = row.get("plan") or {}
        target_objs = [trades.TimedTarget(t.get("name", ""), float(t.get("price") or 0.0),
                                          str(t.get("basis") or ""), t.get("due_ms"),
                                          float(t.get("reward_r") or 0.0), str(t.get("due_from") or ""))
                       for t in plan.get("targets") or []]
        prog = None
        if price_now:
            try:
                prog = trades.progress(row["side"], float(row["entry"]), float(row["stop"]),
                                       target_objs, float(price_now))
            except Exception as e:  # noqa: BLE001
                st.warning(f"Trade #{row['id']}: live numbers unavailable ({e}).")
        try:
            panels.render(panels.trade_card_html(row, prog, now_ms, price_now))
        except Exception as e:  # noqa: BLE001 - one bad card must not remove the others
            st.error(f"Trade #{row['id']} ({row.get('asset')} {row.get('side')}) could not be drawn "
                     f"({type(e).__name__}: {e}). Its levels: entry {row.get('entry')}, "
                     f"stop {row.get('stop')}.")
        if prog is not None:
            try:
                trade_alert_pass(row, prog)
            except Exception as e:  # noqa: BLE001
                st.caption(f"Alerts unavailable for #{row['id']}: {e}")

        b1, b2, note, b3 = st.columns([1, 1, 2.2, 1], vertical_alignment="center")
        note.caption("Live block refreshed "
                     + (f"{panels._trap_age(row.get('refreshed_ms'), now_ms)} ago" if row.get("refreshed_ms")
                        else "never - the numbers above are live on every rerun"))
        if b1.button("Refresh", key=f"tr_refresh_{row['id']}", width="stretch",
                     help="Updates the live numbers only. The thesis and the plan are not rewritten."):
            store.touch_trade(row["id"], now_ms)
            st.rerun()
        if b2.button("Re-judge", key=f"tr_rejudge_{row['id']}", disabled=settings is None, width="stretch",
                     help="Runs the ten sub-agents again and writes a new plan for this position."):
            rejudged = False
            with st.status("Re-judging the position...", expanded=False) as s2:
                try:
                    fresh_plan, fresh_stop, _ = build_trade_plan(
                        row["asset"], row["side"], float(row["entry"]),
                        row.get("given_stop"), row.get("category") or "live")
                    # one transaction: opening then closing as two calls left two open rows for one
                    # position when the close failed
                    new_id = store.rejudge_trade(
                        row["id"], opened_ms=int(time.time() * 1000), asset=row["asset"],
                        side=row["side"], entry=float(row["entry"]), stop=fresh_stop.stop,
                        given_stop=row.get("given_stop"), stop_verdict=fresh_stop.verdict,
                        category=row.get("category") or "live", plan=fresh_plan)
                    s2.update(label=f"Re-judged as #{new_id}", state="complete")
                    rejudged = True
                except Exception as e:  # noqa: BLE001
                    s2.update(label=f"Could not re-judge: {type(e).__name__}: {e}", state="error")
                    st.error(f"Trade #{row['id']} was NOT re-judged ({type(e).__name__}: {e}). The card "
                             "below is still the plan it was opened with - the sub-agents did not run.")
            # only on success: an unconditional rerun discards the status box, which is what made a
            # failed re-judge look exactly like a successful one
            if rejudged:
                st.rerun()
        if b3.button("KILL", key=f"tr_kill_{row['id']}", width="stretch",
                     help="Unpins the card and stops every alert for this position."):
            st.session_state["kill_pending"] = row["id"]
        if st.session_state.get("kill_pending") == row["id"]:
            # the design asks before it kills, in a dialog: a misclick must not unpin a live position
            _kill_dialog(row, price_now)

    if closed_trades:
        with st.expander(f"Recently closed ({len(closed_trades)})", expanded=False):
            for row in closed_trades:
                try:
                    panels.render(panels.closed_trade_html(row, now_ms))
                except Exception as e:  # noqa: BLE001
                    st.caption(f"#{row['id']} could not be drawn: {e}")

    panels.render(panels.section_head_html("Leverage and liquidation stress-test"))
    with st.expander("How much room a given leverage leaves before liquidation, against the structures on "
                     "the 15m and 1h", expanded=False):
        live_a, intraday_a = analyses.get("live"), analyses.get("intraday")
        base = _trade_price() or (live_a.price if live_a else 0.0)
        with st.form("stress_test"):
            d1, d2, d3, d4, d5 = st.columns(5)
            s_dir = d1.segmented_control("Direction", ["LONG", "SHORT"], default="LONG", key="st_dir") or "LONG"
            s_entry = d2.number_input("Entry ($)", value=float(round(base, 2)), step=10.0, key="st_entry")
            s_lev = d3.number_input("Leverage (x)", min_value=1.0, max_value=125.0, value=10.0, step=1.0,
                                    key="st_lev")
            s_sl = d4.number_input("Stop loss ($)", value=float(round(base * 0.98, 2)), step=10.0,
                                   key="st_sl")
            s_tp = d5.number_input("Take profit ($)", value=float(round(base * 1.04, 2)), step=10.0,
                                   key="st_tp")
            run_stress = st.form_submit_button("Run stress-test", width="stretch")
        if run_stress:
            try:
                tm = map_trade({"dir": s_dir, "entry": s_entry, "lev": s_lev, "sl": s_sl, "tp": s_tp},
                               live_a, intraday_a, m.futures)
                panels.render(panels.trade_result_html(tm))
            except Exception as e:  # noqa: BLE001 - the stress-test must not blank the tab
                st.warning(f"Stress-test unavailable: {e}")

    channels = Dispatcher(_secrets()).channels
    ready = [c.name for c in channels if c.configured]
    missing = [f"{c.name} ({c.reason})" for c in channels if not c.configured]
    panels.render(panels.section_head_html("Alert channels"))
    panels.render(panels.channels_html(channels))
    st.caption(("Alerts go to: " + ", ".join(ready) + ". " if ready else "No alert channel is configured. ")
               + "Not set up: " + "; ".join(missing)
               + ". Alerts are detected and sent while this page is open - continuous background alerting "
                 "needs a scheduled worker, which Streamlit Cloud does not provide.")

# ---------- war room ----------
if screen == "director":
    labels = {k: c.label for k, c in CATEGORIES.items()}
    tfs = {k: c.chart_tf for k, c in CATEGORIES.items()}
    hero, act = st.columns([3, 1.2], vertical_alignment="bottom")
    if report is not None:
        age = panels._trap_age(int(report.generated_at.timestamp() * 1000), now_ms)
        with hero:
            panels.render(
                f"<div class='ti-hero'><div class='ti-hero-main'><span class='ti-kicker'>07 · Report "
                f"{html.escape(report.report_id)}</span><h1>Director report</h1><div class='ti-hero-meta'><span>"
                f"Ten specialists, a volatility agent, two desks, an accuracy agent and the Director. Generated "
                f"{fmt_when(int(report.generated_at.timestamp() * 1000))}, {age} ago.</span></div></div></div>")
        with act:
            md = to_markdown(report)
            st.download_button("Download .md", md, file_name=f"{report.report_id}.md", mime="text/markdown",
                               width="stretch")
            st.caption("Run Analysis in the top bar runs it again.")
        panels.render(panels.report_stats_html(report))
        panels.render(panels.director_html(report))
        panels.render(panels.director_lists_html(report))
        panels.render(panels.section_head_html("Per desk"))
        panels.render(panels.director_desks_html(report, labels, tfs))
        with st.expander(f"Specialist briefs ({len(report.briefs)})"):
            panels.render(panels.briefs_html(report))
    else:
        with hero:
            panels.render(panels.engine_hero_html(
                7, "Director report",
                "Ten specialists, a volatility agent, two desks, an accuracy agent and the Director."))
        hint = ("Run Analysis in the top bar generates the Director's report." if settings
                else "Add GEMINI_API_KEY or OPENROUTER_API_KEY to secrets to enable Run Analysis.")
        panels.render(panels.card_html("Trap Intelligence Report",
                                       f"<p class='muted'>{hint} Section 1 of that report, the raw metric "
                                       "snapshot, is live below.</p>"))

    left, right = st.columns(2, gap="large")
    with left:
        panels.render(panels.raw_metrics_html(panels.raw_metrics_rows(m)))
    with right:
        try:
            panels.render(panels.accuracy_html(summary(store)))
            log_rows = store.anchor_log(50)
        except Exception as e:  # noqa: BLE001
            st.warning(f"Accuracy ledger unavailable: {e}")
            log_rows = []
        with st.expander("Audit ledger — anchor changes (persistent)"):
            panels.render(panels.anchor_audit_html(log_rows))


# ---------- macro and news engines (spec section 2) ----------
if screen == "macro":
    panels.render(panels.engine_hero_html(
        8, "Macro & News",
        "The global engines: publisher headlines, gold, rates, the economic calendar and what the prediction "
        "markets are pricing. None of it is a read on one timeframe."))

    left, right = st.columns(2, gap="large")
    with left:
        try:
            panels.render(panels.gold_html(m.metals, now_ms))
        except Exception as e:  # noqa: BLE001 - one panel must not blank the tab
            st.warning(f"Gold unavailable: {e}")
    with right:
        try:
            panels.render(panels.calendar_html(m, now_ms))
        except Exception as e:  # noqa: BLE001
            st.warning(f"Calendar unavailable: {e}")

    try:
        panels.render(panels.rates_html(m.macro, now_ms))
    except Exception as e:  # noqa: BLE001
        st.warning(f"Rates unavailable: {e}")

    # the forecasting layer: positioning into each print, what the print did, and how fresh each story is
    ev_col, nar_col = st.columns(2, gap="large")
    with ev_col:
        try:
            panels.render(panels.event_traps_html(_event_reads(_data_stamp()), now_ms))
        except Exception as e:  # noqa: BLE001
            st.warning(f"Event traps unavailable: {e}")
    with nar_col:
        try:
            from core.indicators.narratives import read as read_narratives
            panels.render(panels.narratives_html(
                read_narratives(m.news.headlines, now_ms) if m.news.available else []))
        except Exception as e:  # noqa: BLE001
            st.warning(f"Narratives unavailable: {e}")

    n_news = len(getattr(m.news, "headlines", []) or []) if getattr(m.news, "available", False) else 0
    panels.render(panels.section_head_html(f"Headlines · {n_news}"))
    topics = panels.news_topics(m.news)
    topic = st.pills("Topic", ["All"] + topics, default="All", key="news_topic", label_visibility="collapsed") \
        if topics else "All"
    try:
        panels.render(panels.news_html(m.news, now_ms, limit=14, topic=None if topic in (None, "All") else topic))
    except Exception as e:  # noqa: BLE001
        st.error(f"The headline feed could not be drawn ({type(e).__name__}: {e}). "
                 "Do not read this as 'no news'.")

    panels.render(panels.section_head_html("Prediction markets"))
    odds_asset = st.segmented_control("Prediction markets for", ["BTC", "ETH", "SOL", "Fed & macro", "All"],
                                      default="BTC", key="odds_asset", label_visibility="collapsed") or "BTC"
    _odds_key = {"All": None, "Fed & macro": "MACRO"}.get(odds_asset, odds_asset)
    try:
        panels.render(panels.predictions_html(m.predictions, now_ms, asset=_odds_key, limit=10))
    except Exception as e:  # noqa: BLE001
        st.error(f"The odds feed could not be drawn ({type(e).__name__}: {e}).")
    st.caption("Data from Polymarket. Powered by VoxOdds, which scores whether each quote can be filled. A "
               "probability on a market flagged fragile is a printed number, not a forecast.")

# ---------- phone footer: the feed list, which lives in the folded-away sidebar on a desktop ----------
with st.container(key="ti_mobile_foot"):
    panels.render(panels.data_status_html(m))
    st.caption("The ledger persists in SQLite; on Streamlit Cloud it resets on reboot.")
