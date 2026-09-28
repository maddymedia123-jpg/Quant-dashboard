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
STORE_VERSION = 6   # bump whenever core.store gains tables or methods, or changes how it matches rows


@st.cache_resource(show_spinner=False)
def _store(version: int = STORE_VERSION):
    return Store()


def _live_store():
    """A Store that definitely understands the current schema, even behind a stale cache.

    The version has to be passed, not defaulted: Streamlit hashes the arguments a call actually makes, so
    `_store()` produced a cache key that never mentioned STORE_VERSION and a bump changed nothing. The
    hasattr probe stays as a second line of defence for a revision that forgets to bump."""
    s = _store(STORE_VERSION)
    if not hasattr(s, "put_trap"):             # cached from an older revision
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

tab_labels = [c.label for c in CATEGORIES.values()] + ["TRAP Intelligence", "Active Trade", "War Room"]
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
with tabs[5]:
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
