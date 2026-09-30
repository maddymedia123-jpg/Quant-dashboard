"""Trap Intelligence - BTC market-surveillance terminal."""
from __future__ import annotations

import asyncio
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
from ui.charts import render_chart
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
STORE_VERSION = 7   # bump whenever core.store gains tables or methods, or changes how it matches rows


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

st.sidebar.markdown("<p class='ti-title'>Trap Intelligence</p><p class='ti-sub'>BTC market surveillance</p>", unsafe_allow_html=True)
dark = st.sidebar.toggle("Dark mode", value=st.session_state["dark"])
st.session_state["dark"] = dark
inject_css(dark)

if st.sidebar.button("Refresh data", width="stretch"):
    _spot.clear()
    _context.clear()
    st.rerun()

AUTO_OPTIONS = (0, 30, 60, 120)
st.session_state.setdefault("auto_refresh", 0)
auto_every = st.sidebar.selectbox(
    "Auto-refresh", AUTO_OPTIONS, index=AUTO_OPTIONS.index(st.session_state["auto_refresh"]),
    format_func=lambda v: "Off" if v == 0 else f"every {v}s",
    help="Reruns the page on a timer. Data caches still apply (spot 25s, context 120s), so this adds no API load.",
)
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
run_clicked = st.sidebar.button(
    "Run Analysis", width="stretch", disabled=settings is None,
    help=(f"13-agent Trap Intelligence report via {settings.provider}: specialists on {settings.specialist_model}, Director on {settings.director_model}."
          if settings else "Add GEMINI_API_KEY or OPENROUTER_API_KEY to .streamlit/secrets.toml (or Streamlit Cloud Secrets) to enable."),
)

recon_clicked = st.sidebar.button(
    "Run Live Recon war room", width="stretch", disabled=settings is None,
    help=("Eleven Live Recon agents: five bullish and five bearish domain agents scored on the 100-point "
          "checklist, plus the war room trap audit. The 4h summary stays pinned; re-runs add side notes."
          if settings else "Add an API key to enable."),
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
st.sidebar.caption(f"Updated {m.generated_at.astimezone(timezone.utc):%H:%M:%S} UTC · spot {TTL['spot']}s · context {TTL['context']}s")

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

    The feed stops at the weekly candle, so four desks cannot have four distinct three-timeframe windows:
    Weekly and Monthly read exactly the same candles. Scanning per desk therefore found the same shape
    twice, paid for two judgements of it, and anchored two rows for one raid. A timeframe belongs to the
    first desk that reads it - the shortest horizon that can see the level, and so the desk that will act
    on it soonest - which gives 15m/1h/4h to Live, 1d to Intraday and 1w to Weekly. Monthly owns nothing
    of its own; it still shows every trap on the timeframes it reads."""
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
            published = live_anchor.publish(store, recon, run_ms, price=m.spot.last,
                                            macro_event=_macro_event_now(m, run_ms, profile), profile=profile)
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

# ---------- war rooms: the sidebar button is the Live Recon shortcut ----------
if not isinstance(st.session_state.get("recon"), dict):
    st.session_state["recon"] = {}
if recon_clicked and settings is not None and analyses:
    run_war_room(LIVE)
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

# ---------- header ----------
st.markdown("<p class='ti-title'>BTC / USD · Trap Intelligence Terminal</p>"
            "<p class='ti-sub'>Deterministic market structure, SMC and liquidity per timeframe, with a war room per category. "
            "Run Analysis adds the Director report and desk briefs.</p>",
            unsafe_allow_html=True)
if not m.spot.available:
    st.error(f"Spot data unavailable from {m.spot.source}: {m.spot.error}. Nothing to analyse.")

tab_labels = ([c.label for c in CATEGORIES.values()]
              + ["TRAP Intelligence", "Active Trade", "War Room", "Macro & News"])
tabs = st.tabs(tab_labels)


def _data_stamp() -> str:
    """A fingerprint of the snapshot this run is rendering.

    `_desk_state` reads module globals that are rebound on every script run, so keying its cache on the
    clock alone handed back a state built from an older snapshot: after a manual Refresh the header
    showed the new price while the volatility, structure and liquidity panels - and the trap scan -
    still read the old one. Keying on the data means the cache follows the snapshot, not the minute."""
    last = getattr(m.spot, "last", None)
    stamps = [str(last)]
    for tf in ("15m", "1h", "4h", "1d", "1w"):
        df = m.spot.frames.get(tf)
        stamps.append(str(int(df["timestamp"].iloc[-1])) if df is not None and not df.empty else "-")
    return "|".join(stamps)


@st.cache_data(ttl=TTL["spot"], show_spinner=False)
def _desk_state(category: str, data_stamp: str) -> dict:
    """The very state the desk's judges are handed, so the screen and the scorecard cannot drift
    apart. Cached per desk per snapshot; a war-room run builds its own state at the moment it runs."""
    return build_state(m, analyses, now_ms, PROFILES[category])


def render_protocols(profile) -> None:
    """The deterministic reads - volatility, structure, liquidity - and each sub-agent's own deck."""
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
        except Exception as e:  # noqa: BLE001
            st.error(f"{len(rows)} trap(s) are anchored on this desk but could not be drawn "
                     f"({type(e).__name__}: {e}). Treat the levels as live and check the TRAP tab.")
    panels.render(panels.vol_matrix_html(state.get("volatility", {})))
    panels.render(panels.smc_html(state.get("smc", {})))
    panels.render(panels.liquidity_html(state.get("liquidity", {})))
    with st.expander(f"Institutional quantitative matrix — {profile.label} sub-agents", expanded=False):
        panels.render(panels.agent_matrix_html(sub_agent_deck(state, profile)))


def render_war_room(profile) -> None:
    """A category's anchored summary, the side notes appended during its candle, and the scorecards."""
    st.markdown("---")
    cat = profile.category
    if st.button(f"Run {profile.label} war room", key=f"war_room_{cat}", disabled=settings is None,
                 help=f"Eleven agents on {', '.join(profile.timeframes)} over the next {profile.horizon}."):
        run_war_room(profile)
        st.rerun()

    w_open, w_close = profile.window_open(now_ms), profile.window_close(now_ms)
    try:
        row = store.get_live_anchor(w_open, category=cat)
        notes = store.side_notes(w_open, category=cat) if row else []
    except Exception as e:  # noqa: BLE001 - the ledger must never blank the tab
        st.warning(f"Anchor ledger unavailable: {e}")
        return

    try:
        panels.render(panels.accuracy_report_html(live_accuracy.diagnose(store, profile)))
    except Exception as e:  # noqa: BLE001 - the audit must never blank the tab
        st.warning(f"Accuracy audit unavailable: {e}")

    pair = st.session_state["recon"].get(cat)
    anchored_now = bool(pair and pair[1].anchored and pair[1].window_open_ms == w_open)
    closes = datetime.fromtimestamp(w_close / 1000, timezone.utc)
    closes_txt = f"{closes:%H:%M} UTC" if profile.window == "4h" else f"{closes:%a %d %b %H:%M} UTC"
    if row is None:
        panels.render(panels.card_html(
            f"{profile.window_title} anchored summary",
            f"<p class='muted'>Nothing anchored for the current {profile.window_label} candle yet. Run the "
            f"{profile.label} war room to publish the summary for the window closing {closes_txt}. "
            "It then stays pinned until that close; re-runs only add side notes.</p>"))
    else:
        panels.render(panels.anchored_summary_html(row["payload"], w_close, now_ms, anchored_now, profile))
        panels.render(panels.side_notes_html(notes))

    if not pair:
        return
    recon = pair[0]
    c1, c2 = st.columns(2)
    with c1:
        panels.render(panels.team_html(recon.bull))
    with c2:
        panels.render(panels.team_html(recon.bear))
    panels.render(panels.war_room_html(recon))
    st.caption(f"Last war-room run {recon.generated_at:%H:%M:%S} UTC · {recon.bull.provider or 'n/a'} · "
               f"{recon.prompt_tokens + recon.completion_tokens:,} tokens")


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
            panels.render(panels.patterns_html(a))
            panels.render(panels.volatility_html(a))
        with c2:
            panels.render(panels.fib_html(a))
            panels.render(panels.divergence_html(a))
            panels.render(panels.agent_summary_html(a, report))
        profile = PROFILES.get(key)
        if profile is not None:
            render_protocols(profile)
            render_war_room(profile)
        if key in ("weekly", "monthly"):
            panels.render(panels.calendar_html(m, now_ms))


for tab, key in zip(tabs[:4], CATEGORIES.keys()):
    category_tab(tab, key)

# ---------- TRAP intelligence ----------
with tabs[4]:
    st.markdown("### TRAP intelligence")
    st.caption("Ten trap sub-agents across the five domains, a Head that consults the affected desk, and "
               "traps anchored to that desk until price settles them.")
    if st.button("Sweep every timeframe for traps", key="trap_sweep_btn", disabled=settings is None,
                 help="Deterministic scan of every timeframe once; the strongest candidate on each "
                      "owning desk goes to the agents."):
        # deliberately no st.rerun(): it discards the status box, so a sweep that failed on desk three
        # came back looking exactly like a page that had done nothing. The console below renders from
        # session state in this same run.
        run_trap_sweep()

    sweep = st.session_state.get("trap_sweep")
    try:
        # anchored traps are what a trader acts on, so they are never pushed off the list by newer
        # settled ones: the two are queried separately rather than sliced out of one ordered list
        active = store.traps(status="active", limit=25)
        settled = [r for r in store.traps(limit=50) if r.get("status") != "active"][:6]
    except Exception as e:  # noqa: BLE001 - never crash the tab on ledger trouble
        active, settled = [], []
        st.error(f"The trap ledger could not be read ({type(e).__name__}: {e}). "
                 "Anchored traps are not shown below; do not read this as 'no traps'.")

    watching = [c for r in (sweep or {}).get("results", []) for c in r["candidates"]]
    try:
        panels.render(panels.trap_console_html(active, watching, now_ms,
                                              {k: c.label for k, c in CATEGORIES.items()},
                                              settled=settled, swept_ms=(sweep or {}).get("at")))
    except Exception as e:  # noqa: BLE001
        st.error(f"{len(active)} anchored trap(s) exist but the console could not be drawn "
                 f"({type(e).__name__}: {e}).")
    for row in active + settled:
        # per row, so one malformed trap cannot take the others off the screen with it
        try:
            label = CATEGORIES[row["category"]].label if row["category"] in CATEGORIES else row["category"]
            panels.render(panels.trap_card_html(row, now_ms, label))
        except Exception as e:  # noqa: BLE001
            st.warning(f"Trap #{row.get('id', '?')} ({row.get('side', 'unknown')} on "
                       f"{row.get('timeframe', '?')}) could not be drawn: {type(e).__name__}: {e}")

    if sweep:
        for r in sweep["results"]:
            v = r["verdict"]
            if v is None:
                continue
            with st.expander(f"{CATEGORIES[r['category']].label} · {v.candidate.label} on "
                             f"{v.candidate.timeframe} · {v.score:.0f}/100 · {v.status}", expanded=False):
                st.markdown(f"**Head call:** {v.head_call or '—'}"
                            + (f" ({v.head_confidence:.0%} confidence)" if v.head_confidence else "")
                            + f"  \n**Desk says:** {v.desk.reads}")
                if v.desk.reasons:
                    st.markdown("  \n".join(f"- {x}" for x in v.desk.reasons))
                st.markdown("  \n".join(f"- {d.title}: {d.points:.0f}/20" for d in v.domains))
                if v.notes:
                    st.caption(" · ".join(v.notes))

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


with tabs[5]:
    st.markdown("### Active trade desk")
    st.caption("Ten sub-agents on the position, the four category heads and the TRAP head consulted, and "
               "a card that stays pinned until you kill it. Cards live in the database, so they survive a "
               "refresh and a redeploy.")

    spot = _trade_price()
    with st.form("new_trade", clear_on_submit=False):
        c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
        f_asset = c1.text_input("Asset", value="BTC/USDT")
        f_side = c2.selectbox("Direction", ["LONG", "SHORT"])
        f_entry = c3.number_input("Entry ($)", value=float(round(spot or 0.0, 2)), step=10.0,
                                  help="Where you actually got in.")
        f_stop = c4.number_input("Stop loss ($) - 0 to let the desk place it", value=0.0, step=10.0,
                                 help="Leave at 0 and the engine derives a structural stop.")
        f_cat = st.selectbox("Judge it over", list(PROFILES),
                             format_func=lambda k: f"{CATEGORIES[k].label} ({PROFILES[k].horizon})", index=0)
        opened = st.form_submit_button("Open and track this trade", width="stretch",
                                       disabled=not spot)
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

        b1, b2, b3 = st.columns(3)
        if b1.button("Refresh", key=f"tr_refresh_{row['id']}",
                     help="Updates the live numbers only. The thesis and the plan are not rewritten."):
            store.touch_trade(row["id"], now_ms)
            st.rerun()
        if b2.button("Re-judge", key=f"tr_rejudge_{row['id']}", disabled=settings is None,
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
        if b3.button("KILL", key=f"tr_kill_{row['id']}", type="primary",
                     help="Unpins the card and stops every alert for this position."):
            store.close_trade(row["id"], "killed by the trader", price_now, now_ms)
            st.rerun()

    if closed_trades:
        with st.expander(f"Recently closed ({len(closed_trades)})", expanded=False):
            for row in closed_trades:
                try:
                    panels.render(panels.trade_card_html(row, None, now_ms))
                except Exception as e:  # noqa: BLE001
                    st.caption(f"#{row['id']} could not be drawn: {e}")

    with st.expander("Leverage and liquidation stress-test", expanded=False):
        st.caption("A separate question from the card above: how much room a given leverage leaves before "
                   "liquidation, against the structures on the 15m and 1h.")
        live_a, intraday_a = analyses.get("live"), analyses.get("intraday")
        base = _trade_price() or (live_a.price if live_a else 0.0)
        with st.form("stress_test"):
            d1, d2, d3, d4, d5 = st.columns(5)
            s_dir = d1.selectbox("Direction", ["LONG", "SHORT"], key="st_dir")
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
    st.caption(("Alerts go to: " + ", ".join(ready) + ". " if ready else "No alert channel is configured. ")
               + "Not set up: " + "; ".join(missing)
               + ". Alerts are detected and sent while this page is open - continuous background alerting "
                 "needs a scheduled worker, which Streamlit Cloud does not provide.")

# ---------- war room ----------
with tabs[6]:
    if report is not None:
        panels.render(panels.report_stats_html(report))
        md = to_markdown(report)
        st.download_button("Download report (.md)", md, file_name=f"{report.report_id}.md", mime="text/markdown")
        st.markdown(md_safe(md))
        with st.expander("Specialist briefs"):
            for b in report.briefs:
                st.markdown(f"**{b.agent_id} · {b.desk} · {b.domain} · conviction {b.conviction:.0f}/10**  \n{md_safe(b.interpretation)}")
                if b.traps_detected:
                    st.markdown("Traps: " + "; ".join(md_safe(t) for t in b.traps_detected))
                st.markdown(f"Key risk to thesis: {md_safe(b.key_risk_to_thesis)}")
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


# ---------- macro and news engines (spec section 2) ----------
with tabs[7]:
    st.markdown("### Macro and news")
    st.caption("The global engines: publisher headlines, gold, and what the prediction markets are "
               "pricing. Global rather than per-desk, because none of it is a read on one timeframe.")

    left, right = st.columns([1, 1])
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
        panels.render(panels.news_html(m.news, now_ms, limit=14))
    except Exception as e:  # noqa: BLE001
        st.error(f"The headline feed could not be drawn ({type(e).__name__}: {e}). "
                 "Do not read this as 'no news'.")

    odds_asset = st.selectbox("Prediction markets for", ["BTC", "ETH", "SOL", "everything"], index=0,
                              key="odds_asset")
    try:
        panels.render(panels.predictions_html(m.predictions, now_ms,
                                              asset=None if odds_asset == "everything" else odds_asset,
                                              limit=10))
    except Exception as e:  # noqa: BLE001
        st.error(f"The odds feed could not be drawn ({type(e).__name__}: {e}).")
    st.caption("Odds come from Polymarket through VoxOdds, which scores whether each quote is actually "
               "fillable. A probability on a market flagged fragile is a printed number, not a forecast.")
