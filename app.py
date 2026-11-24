"""Trap Intelligence - BTC market-surveillance terminal."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import html
import json
import os
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

from core.accuracy import (  # noqa: E402
    majority_direction, mode_scorecard, record_report, summary,
)
from core.agents.one_agent import client_for as one_agent_client, run_one_agent_sync  # noqa: E402
from core.agents.llm import LLMClient
from core.agents.report import to_markdown
from core.agents.runner import run_pipeline_sync
from core.agents.settings import load_settings
from core.agents.judge import Judge
from core.agents.deck import sub_agent_deck
from core.agents.trap_desk import judge_candidate, record
from core.agents.trade_desk import judge_trade
from core.alerts import Dispatcher
from core import trades
from core.trade_plan import deterministic_plan
from core.traps import desk_timeframes, find_candidates, tf_label
from core.agents.live_recon import build_state, run_live_recon
from core.agents import handshake
from core.agents.recon_profiles import LIVE, PROFILES
from core import live_accuracy, live_anchor, pipeline
from core.config import CATEGORIES, CHART_VIEWS, MACRO_EVENTS, TTL
from core.data.market import assemble, load_context, load_spot
from core.derived import enrich_dominance, enrich_futures, record_dominance, record_sample
from core.indicators import desk_derivs
from core.indicators.category import analyze_category
from core.indicators.early_warning import early_warnings
from core.indicators.forecast import desk_forecast
from core.store import TURSO_KEYS, Store, find_turso
from core.indicators.trade_map import map_trade
from ui import panels
from core.trades import fmt_when
from ui.charts import render_projected_chart
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
STORE_VERSION = 10   # bump whenever core.store gains tables or methods, or changes how it matches rows


def _turso_secrets() -> dict[str, str]:
    """The Turso settings from Streamlit's secrets, wherever they were pasted (core.store.find_turso)."""
    try:
        return find_turso(dict(st.secrets))
    except Exception:  # noqa: BLE001 - no secrets file
        return {}


def _export_turso() -> None:
    """core.store reads Turso's settings from the environment, as the worker gets them from its .env."""
    for k, v in _turso_secrets().items():
        os.environ.setdefault(k, v)


@st.cache_resource(show_spinner=False)
def _store(version: int = STORE_VERSION):
    _export_turso()
    return Store()


def _live_store():
    """A Store that definitely understands the current schema, even behind a stale cache.

    The version has to be passed, not defaulted: Streamlit hashes the arguments a call actually makes, so
    `_store()` produced a cache key that never mentioned STORE_VERSION and a bump changed nothing. The
    hasattr probe stays as a second line of defence for a revision that forgets to bump."""
    s = _store(STORE_VERSION)
    if not hasattr(s, "put_qf_plan"):   # cached from an older revision
        _drop_store(s)
        s = _store(STORE_VERSION)
    # the store is built once per process: one opened before the Turso secrets were saved stays on the
    # local file until a reboot. Secrets present now but missing when it opened: open it again, once.
    if (getattr(s, "backend", "") == "sqlite" and str(getattr(s, "turso_note", "") or "").startswith("not set")
            and len(_turso_secrets()) == len(TURSO_KEYS)):
        _drop_store(s)
        s = _store(STORE_VERSION)
    try:
        s.sync(min_gap_s=30)                   # what the worker wrote; at most one pull per 30 s (Turso sync budget)
    except Exception as e:  # noqa: BLE001 - a missed sync shows slightly older rows, never a blank page
        st.sidebar.caption(f"Store sync skipped this refresh: {e}")
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
    st.session_state["last_full_run_ms"] = int(time.time() * 1000)
    st.rerun()

AUTO_OPTIONS = (0, 30, 60, 120)
st.session_state.setdefault("auto_refresh", 0)
st.session_state.setdefault("auto_refresh_ctl", st.session_state["auto_refresh"])
if st.session_state["auto_refresh_ctl"] not in AUTO_OPTIONS:
    st.session_state["auto_refresh_ctl"] = 0
auto_every = _auto_col.segmented_control(
    "Auto-refresh",
    AUTO_OPTIONS,
    key="auto_refresh_ctl",
    required=True,
    format_func=lambda v: "Off" if v == 0 else f"{v}s",
    help="Reruns the page on a timer. Data caches still apply (spot 25s, context 120s), so this adds no API load.",
)
st.session_state["auto_refresh"] = auto_every
st.session_state.setdefault("last_full_run_ms", int(time.time() * 1000))

if auto_every:
    @st.fragment(run_every=f"{auto_every}s")
    def _auto_tick():
        now_tick_ms = int(time.time() * 1000)
        elapsed = max(0, now_tick_ms - st.session_state.get("last_full_run_ms", now_tick_ms))
        if elapsed >= auto_every * 1000 - 500:
            st.session_state["last_full_run_ms"] = now_tick_ms
            st.rerun(scope="app")
        st.caption(f"Auto-refresh every {auto_every}s · last full update {elapsed // 1000}s ago")

    with st.sidebar:
        _auto_tick()
settings = load_settings()
st.session_state.setdefault("trap_report", None)
run_clicked = _run_col.button(
    "Run Analysis", type="primary", width="stretch", disabled=settings is None,
    help=(f"1-agent Director report: one call on {settings.director_model} reading the whole state. The "
          "Director report screen can also run the 15-agent pipeline, or both on the same snapshot."
          if settings else "Add GEMINI_API_KEY or OPENROUTER_API_KEY to .streamlit/secrets.toml (or Streamlit Cloud Secrets) to enable."),
)

with st.spinner("Loading market data"):
    m = assemble(_spot(), _context())
try:
    _store_early = _live_store()
    _now_early = int(time.time() * 1000)
    record_sample(_store_early, m.futures, _now_early)
    record_dominance(_store_early, m.dominance, _now_early)
    m = m.model_copy(update={"futures": enrich_futures(_store_early, m.futures, _now_early),
                             "dominance": enrich_dominance(_store_early, m.dominance, _now_early)})
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
                        f"TRAP desk: {c.label} anchored at {c.level:,.0f} on the {tf_label(c.timeframe)} chart.",
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
# Two pipelines write the same report, plan as numbers included: the 1-agent Director (one call) and the classic
# 15-agent chain. "both" runs them on this one snapshot, so the compare view sets like against like.
if not isinstance(st.session_state.get("director_reports"), dict):
    st.session_state["director_reports"] = {}
_director_mode = "1-agent" if run_clicked else st.session_state.pop("director_pending", None)


def _director_failed(label: str, status) -> None:
    """The run block ends in st.rerun(), which wipes the status widget and its error with it: a failed run
    looked like nothing happened (23 Nov). The message is kept and shown on the Director screen until the
    next run succeeds."""
    status.update(label=label, state="error")
    st.session_state["director_error"] = f"{label} ({fmt_when(now_ms)})"


def _keep_report(rep) -> None:
    st.session_state["director_reports"][rep.mode] = rep
    st.session_state["trap_report"] = rep
    intraday_a = analyses.get("intraday")
    record_report(store, rep, m.spot.last or next(iter(analyses.values())).price,
                  intraday_a.vol.sigma_pct if intraday_a else None, majority_direction(analyses), now_ms)


if _director_mode and settings is not None and analyses:
    snapshot_id = f"S-{m.generated_at:%Y%m%d-%H%M%S}"
    modes = ("1-agent", "15-agent") if _director_mode == "both" else (_director_mode,)
    for mode in modes:
        if mode == "1-agent":
            with st.status(f"1-agent Director on {settings.director_model}: one call reading the whole state...",
                           expanded=False) as status:
                try:
                    rep = run_one_agent_sync(one_agent_client(settings), m, analyses, settings.director_model,
                                             panels.raw_metrics_rows(m), snapshot_id)
                    if rep.director is None:
                        _director_failed(f"1-agent Director failed: {rep.runs[0].error}", status)
                    else:
                        _keep_report(rep)
                        st.session_state.pop("director_error", None)
                        status.update(label=f"1-agent report {rep.report_id} ready in {rep.wall_time_s:.0f}s "
                                            f"on {rep.runs[0].model}", state="complete")
                except Exception as e:  # noqa: BLE001 - never crash the page on an LLM failure
                    _director_failed(f"1-agent Director failed: {type(e).__name__}: {e}", status)
            continue
        with st.status("Running the 15-agent pipeline...", expanded=True) as status:
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
                rep = run_pipeline_sync(LLMClient(settings), m, analyses, settings,
                                        raw_metrics=panels.raw_metrics_rows(m), progress=_progress)
                rep.snapshot_id = snapshot_id
                _keep_report(rep)
                st.session_state.pop("director_error", None)
                status.update(label=f"15-agent report {rep.report_id} ready in {rep.wall_time_s:.0f}s",
                              state="complete", expanded=False)
            except Exception as e:  # noqa: BLE001 - never crash the page on an LLM failure
                _director_failed(f"15-agent pipeline failed: {type(e).__name__}: {e}", status)
    st.session_state["last_full_run_ms"] = int(time.time() * 1000)
    st.rerun()

# ---------- war rooms ----------
if not isinstance(st.session_state.get("recon"), dict):
    st.session_state["recon"] = {}

report = st.session_state.get("trap_report")

# ---------- anchoring + accuracy ----------
# The same pass the 24/7 worker runs each minute (core.pipeline): anchor each desk, score what is due,
# settle the traps price has answered, score the war-room windows whose candle closed.
classification = report.director.trap_classification if (report is not None and report.director is not None) else None
st.session_state.setdefault("trap_sweep", None)
_ledger = pipeline.ledger(store, m, analyses, now_ms, classification)
verdicts = _ledger.verdicts
for _t in _ledger.settled:
    st.sidebar.caption(f"TRAP: {_t['side']} {_t['timeframe']} {_t['status'].replace('_', ' ')}")
for _p in _ledger.problems:
    st.warning(f"Ledger step unavailable this refresh ({_p}).")
if not (m.spot.last or analyses) and store.traps(status="active", limit=1):
    # silence here would leave anchored traps rendering as live with no hint that nothing is
    # checking them against price any more
    st.warning("No price this refresh, so anchored traps were not checked for settlement. "
               "The levels below are as of the last successful refresh.")

st.sidebar.markdown(f"<span class='ti-chip'>ledger: {store.path}</span>", unsafe_allow_html=True)
if store.backend == "turso":
    st.sidebar.caption("Anchors, accuracy and alerts are shared with the 24/7 worker in Turso.")
else:
    _seen = _turso_secrets()
    st.sidebar.caption(
        "Anchors and accuracy persist in a local SQLite file, which resets on every deploy. Turso: "
        + html.escape(str(getattr(store, "turso_note", "") or "not configured")) + ". Secrets seen: "
        + ", ".join(f"{k} {'yes' if k in _seen else 'no'}" for k in TURSO_KEYS) + ".")

# ---------- top bar ----------
_live_a = analyses.get("live")
price_slot.markdown(panels.top_bar_html(
    m.spot.last if m.spot.last is not None else (_live_a.price if _live_a else None),
    _live_a.price_change_24h_pct if _live_a else None,
    f"Updated {fmt_when(int(m.generated_at.timestamp() * 1000), seconds=True)}", m.spot.symbol),
    unsafe_allow_html=True)
# on a desk the bar shows that desk's own gamma (its expiries only); on an engine screen, the 30-day chain
_bar_desk = st.session_state.get("screen", "live")
_bar_desk = _bar_desk if _bar_desk in CATEGORIES else None
gamma_slot.markdown(panels.gamma_bar_html(desk_derivs.gamma_for(_bar_desk, m), m.spot.last,
                                          CATEGORIES[_bar_desk].label if _bar_desk else ""),
                    unsafe_allow_html=True)
if not m.spot.available:
    st.error(f"Price data unavailable from {m.spot.source}: {m.spot.error}. Nothing to analyse.")

# ---------- the rail: four desks, then four engines; only the open screen renders ----------
ENGINES = {"trap": "TRAP Intelligence", "trade": "Active Trade", "director": "Director report", "macro": "Macro & News",
           "qf": "Quant Forecast"}
SCREENS = list(CATEGORIES) + list(ENGINES)
_labels = {**{k: c.label for k, c in CATEGORIES.items()}, **ENGINES}
_captions = ([panels.rail_read(verdicts.get(k), analyses.get(k)) for k in CATEGORIES]
             + ["Sweep, anchored traps, watch list", "Positions, stops and targets",
                f"Report {report.report_id}" if report is not None else "Run Analysis to generate",
                "Headlines, gold, rates, odds", "Today's 24h plan, audited live"])
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
    for tf in ("15m", "1h", "4h", "12h", "1d", "3d", "0.5w", "1w", "2w", "1M", "3M"):
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
def _desk_view(category: str, view_tf: str, data_stamp: str):
    """The desk's chart on the chosen candle (his Main Chart View), cached per snapshot like the rest."""
    from core.indicators.category import view_analysis
    return view_analysis(CATEGORIES[category], view_tf, m.spot.frames, m.futures, analyses[category])


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


@st.cache_data(ttl=900, show_spinner=False)
def _sine_read(quarter: int, data_stamp: str):
    """The experimental sine card: the warning off the regular 15m feed, the live record off a paged history
    reaching back to the go-live date (the feed keeps three weeks). Recomputed once per 15m candle."""
    from core.data.gate_candles import fetch_15m_since
    from core.data.market import _offline
    from core.indicators import sine_wave as sw
    feed = m.spot.frames.get("15m")
    hist = None if _offline() else fetch_15m_since(sw.GO_LIVE_MS - 2 * 86_400_000, now_ms)
    note = "" if hist is not None else "The full history did not load, so the record covers the last three weeks only."
    warning, ny_now = sw.current(feed, now_ms)
    return warning, ny_now, sw.live_record(hist if hist is not None else feed, now_ms), note


@st.cache_data(ttl=TTL["spot"], show_spinner=False)
def _desk_harmonics(category: str, data_stamp: str):
    """Harmonics on the desk's own 1H/4H/1D, with the gamma levels they may sit on."""
    from core.indicators.harmonics import find_all
    return find_all(m.spot.frames, PROFILES[category].timeframes, desk_derivs.gamma_for(category, m))


def _desk_evidence(category: str, data_stamp: str):
    """What the summary's confluence gate reads: the desk's harmonics, its break reads, its liquidity,
    this week's prints and the dollar and yields - each the cached read the cards above already show."""
    from core.agents.context import _rates
    from core.config import TF_MINUTES
    from core.confluence import HIGHER, Evidence
    from core.gravity_engine import evaluate as evaluate_gravity
    from core.indicators import smc
    # the higher-timeframe bar reads the levels of the desks above this one (Weekly, Monthly), never its own
    htf = tuple((f"{k} level", float(lv.price)) for k in HIGHER.get(category, ()) if analyses.get(k) is not None
                for lv in analyses[k].levels)
    try:
        own = desk_derivs.read(category, m, now_ms)
    except Exception:  # noqa: BLE001 - the gate then reads the derivatives the summary passes, or none
        own = None
    desk = analyses.get(category)
    frame = m.spot.frames.get(CATEGORIES[category].chart_tf)
    book = store.book_report() if hasattr(store, "book_report") else None
    absorption_drop = None
    absorption_side = None
    if book and book.get("available") and 0 <= now_ms - int(book.get("ts_ms", 0)) <= 10 * 60_000:
        for scoop in reversed(book.get("scoops") or []):
            wall = scoop.get("wall_btc")
            remaining = scoop.get("now_btc")
            if wall and remaining is not None and float(wall) > 0:
                drop = max(0.0, min(1.0, 1.0 - float(remaining) / float(wall)))
                if drop > 0:
                    absorption_drop = drop
                    absorption_side = scoop.get("side")
                    break
    futures = m.futures if m.futures.available else None
    liq_levels = m.liq_levels.to_state() if m.liq_levels.available else None
    gravity = evaluate_gravity(
        frame,
        price=desk.price if desk is not None else m.spot.last,
        atr=desk.vol.atr if desk is not None else None,
        oi_change_pct=futures.oi_change_24h_pct if futures is not None else None,
        funding_rate=futures.funding_rate if futures is not None else None,
        absorption_drop=absorption_drop,
        absorption_side=absorption_side,
        liq_levels=liq_levels,
        fvgs=smc.fair_value_gaps(frame) if frame is not None else None,
        bar_minutes=TF_MINUTES[CATEGORIES[category].chart_tf],
    )
    return Evidence(
        harmonics=tuple(h for found in _desk_harmonics(category, data_stamp).values() for h in found),
        gravity=gravity,
    )


st.session_state["last_full_run_ms"] = int(time.time() * 1000)
