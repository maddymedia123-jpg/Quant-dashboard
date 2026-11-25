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


def _data_status_html(snap) -> str:
    fn = getattr(panels, "data_status_html", None)
    if callable(fn):
        return fn(snap)
    rows = []
    for name in getattr(type(snap), "SOURCES", ()):
        s = getattr(snap, name, None)
        if s is None:
            continue
        source = html.escape(str(getattr(s, "source", "")))
        avail = bool(getattr(s, "available", False))
        err = getattr(s, "error", None)
        src_txt = source + ("" if avail else " · unavailable")
        details = html.escape(f"{source} — {err}" if err else source)
        rows.append(
            f"<div class='ti-feed{'' if avail else ' off'}'><span class='dot'></span>"
            f"<span>{html.escape(name)}</span>"
            f"<span class='src' title='{details}'>{src_txt}</span></div>"
        )
    return "<span class='ti-label'>Feeds</span><div class='ti-feeds'>" + "".join(rows) + "</div>"

from core.trades import fmt_when
from ui.charts import render_chart, render_forecast_chart
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
st.sidebar.markdown(_data_status_html(m), unsafe_allow_html=True)
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
                        status.update(label=f"1-agent Director failed: {rep.runs[0].error}", state="error")
                    else:
                        _keep_report(rep)
                        status.update(label=f"1-agent report {rep.report_id} ready in {rep.wall_time_s:.0f}s "
                                            f"on {rep.runs[0].model}", state="complete")
                except Exception as e:  # noqa: BLE001 - never crash the page on an LLM failure
                    status.update(label=f"1-agent Director failed: {e}", state="error")
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
                status.update(label=f"15-agent report {rep.report_id} ready in {rep.wall_time_s:.0f}s",
                              state="complete", expanded=False)
            except Exception as e:  # noqa: BLE001 - never crash the page on an LLM failure
                status.update(label=f"15-agent pipeline failed: {e}", state="error")
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
    from core.confluence import HIGHER, Evidence
    # the higher-timeframe bar reads the levels of the desks above this one (Weekly, Monthly), never its own
    htf = tuple((f"{k} level", float(lv.price)) for k in HIGHER.get(category, ()) if analyses.get(k) is not None
                for lv in analyses[k].levels)
    try:
        own = desk_derivs.read(category, m, now_ms)
    except Exception:  # noqa: BLE001 - the gate then reads the derivatives the summary passes, or none
        own = None
    desk = analyses.get(category)
    return Evidence(harmonics=tuple(h for found in _desk_harmonics(category, data_stamp).values() for h in found),
                    htf_levels=htf, derivs=own,
                    fear_greed=m.sentiment.fear_greed if m.sentiment.available else None,
                    price_change_24h_pct=desk.price_change_24h_pct if desk is not None else None,
                    oi_history=tuple(desk_derivs.oi_history(m)),
                    breaks=tuple(_desk_breaks(category, data_stamp).values()),
                    liquidity=_desk_state(category, data_stamp).get("liquidity"),
                    prints=tuple(_event_reads(data_stamp)),
                    rates=_rates(m),
                    futures=m.futures if m.futures.available else None,
                    liq_levels=m.liq_levels.to_state() if m.liq_levels.available else None)


def render_protocols(profile) -> None:
    """The deterministic reads - volatility, structure, liquidity - and each sub-agent's own deck, one at
    a time behind a switch, as the design lays them out."""
    try:
        state = _desk_state(profile.category, _data_stamp())
    except Exception as e:  # noqa: BLE001 - a protocol failure must not blank the tab
        st.warning(f"Protocols unavailable this refresh: {e}")
        return
    # every trap on a timeframe this desk reads, not only the ones it declared itself - except the
    # short-horizon (15m/1h) traps, which only the Live desk sees (v22 Section 1.4, owner 4 Nov)
    try:
        rows = store.traps(timeframes=desk_timeframes(profile.category, profile.timeframes), status="active")
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
        # the desk's own-window derivatives: the strip, the deck and the summary all read this one
        try:
            _own = desk_derivs.read(key, m, now_ms)
        except Exception:  # noqa: BLE001 - the strip falls back to the 24h snapshot, labelled 24h
            _own = None
        kpi_strip(panels.kpis_for(a, m, _own))
        # his weekend microstructure modifier: flagged on every desk while it applies
        try:
            from core.indicators.weekend import read as _weekend_read
            _wk_card = panels.weekend_html(_weekend_read(now_ms))
            if _wk_card:
                panels.render(_wk_card)
        except Exception as e:  # noqa: BLE001
            st.warning(f"Weekend flag unavailable: {e}")
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
        try:
            _harm = _desk_harmonics(key, _data_stamp())
            if _harm:
                panels.render(panels.harmonics_html(_harm))
        except Exception as e:  # noqa: BLE001
            st.warning(f"Harmonic scan unavailable this refresh: {e}")
        # his Section 5.3: the data-versus-candle anomaly trap, from the same evidence the summary's gate reads
        try:
            from core.indicators.anomaly_trap import from_evidence
            _trap = from_evidence(_desk_evidence(key, _data_stamp()), a.chart_tf, now_ms)
            if _trap is not None:
                panels.render(panels.anomaly_trap_html(_trap))
        except Exception as e:  # noqa: BLE001 - a trap read must not blank the desk
            st.warning(f"Anomaly trap read unavailable this refresh: {e}")
        if key == "live":
            # experimental and labelled so: tracked live from its go-live date, never fed to agents or alerts
            try:
                panels.render(panels.sine_experimental_html(*_sine_read(now_ms // 900_000, _data_stamp())))
            except Exception as e:  # noqa: BLE001
                st.warning(f"Sine wave card unavailable this refresh: {e}")
        if key in ("weekly", "monthly"):
            last_sig = store.last_signal(key, before_ms=now_ms)
            banner = panels.early_warning_html(early_warnings(a, m.futures, last_sig["squeeze_score"] if last_sig else None))
            if banner:
                panels.render(banner)
        # the chart and the call it supports sit side by side, as the design draws them
        c_chart, c_dir = st.columns([2, 1], gap="large")
        with c_chart:
            with st.container(key="ti_chart"):
                # his Section 3 Main Chart View: the desk's own candle by default, one coarser view a
                # click away; the call on the right and everything below stay on the desk's candle
                views = CHART_VIEWS[key]
                view_tf = st.segmented_control("Chart candle", list(views), default=views[0],
                                               key=f"chart_view_{key}", label_visibility="collapsed") or views[0]
                try:
                    view = _desk_view(key, view_tf, _data_stamp())
                except Exception as e:  # noqa: BLE001 - a view failure must not blank the chart
                    view = None
                    st.warning(f"{view_tf} view unavailable this refresh: {e}")
                if view is None:
                    if view_tf != a.chart_tf:
                        st.caption(f"Not enough {view_tf} candles for that view yet; showing the {a.chart_tf} chart.")
                    view = a
                panels.render(panels.chart_head_html(view, signal, m.spot.symbol, call_tf=a.chart_tf))
                render_chart(view, dark, height=480, signal=signal)
        with c_dir:
            v = verdicts.get(key)
            panels.render(panels.anchored_direction_html(a, v, now_ms) if v else panels.direction_html(a))
        # his Section 8: one clean forecast pane per desk - the median path, its range and its odds
        fcast, record = desk_forecast(a, desk_derivs.gamma_for(key, m),
                                      m.spot.frames.get(a.chart_tf))
        render_forecast_chart(a, fcast, dark, height=340, signal=signal)
        panels.render(panels.forecast_html(fcast, record, a.vol.horizon_label))
        if fcast.available and fcast.end_low and fcast.end_high and a.price:
            # one range on the desk: everything below (plain read, volatility card, summary) quotes the
            # forecast's 68% range - the one with a track record - instead of a second, simpler estimate
            a = replace(a, vol=replace(a.vol, exp_low=fcast.end_low, exp_high=fcast.end_high,
                                       sigma_pct=(fcast.end_high - fcast.end_low) / 2 / a.price * 100))
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
        # a print inside the five-day window is part of every desk's trade: shown when there is something to read
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
            # daily on-chain flows: like COT, too slow for the four-hour and 24-hour desks
            try:
                from core.indicators.exchange_flows import read as read_flows
                _d1 = m.spot.frames.get("1d")
                panels.render(panels.exchange_flows_html(read_flows(
                    m.exchange_flows, now_ms,
                    list(zip(_d1["timestamp"].astype("int64"), _d1["close"].astype(float))) if _d1 is not None else None)))
            except Exception as e:  # noqa: BLE001
                st.warning(f"Exchange flows unavailable: {e}")
        # his Section 9: every desk ends with the same standardized summary, gated by seven confluences
        try:
            _ev = _desk_evidence(key, _data_stamp())
        except Exception as e:  # noqa: BLE001 - a missing feed is a family that says nothing, not a blank desk
            st.warning(f"Confluence evidence incomplete this refresh: {e}")
            _ev = None
        panels.render(panels.layman_summary_html(a, CATEGORIES[key].label, _own,
                                                 m.sentiment.fear_greed if m.sentiment.available else None,
                                                 now_ms, _ev))


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
            st.warning(f"Settled traps could not be drawn: {e}")
    st.caption("A trap is declared when its five sub-agents score 60 or more, the Head calls it engineered, and "
               "the category desk does not read the move as an authentic break. It is invalidated when the "
               "break proves real and played out when price returns inside value. Nothing expires on a clock.")

# ---------- active trade ----------
def _secrets() -> dict:
    """Streamlit secrets, or an empty mapping when there is no secrets file.

    `hasattr(st, "secrets")` is always True - the attribute exists whether or not a file does - so the
    old guard was dead and `dict(st.secrets
