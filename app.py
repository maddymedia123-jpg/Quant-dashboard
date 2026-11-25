"""Streamlit entrypoint for TRAP INTELLIGENCE."""
from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import os
import pkgutil
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from core.config import CATEGORIES, CHART_VIEWS, Category  # noqa: E402
from core.indicators import category as cat_ind  # noqa: E402
from core.indicators import desk_derivs  # noqa: E402
from core.trades import fmt_when  # noqa: E402
from ui import charts, panels, theme  # noqa: E402


def _opt_import(module_path: str):
    try:
        return importlib.import_module(module_path)
    except Exception:  # noqa: BLE001
        return None


core_data = _opt_import("core.data")
core_pipeline = _opt_import("core.pipeline")
accuracy = _opt_import("core.accuracy")
qf_alerts = _opt_import("core.qf_alerts")
quant_forecast = _opt_import("core.quant_forecast")
_store_mod = _opt_import("core.store")
traps = _opt_import("core.traps")
recon_profiles = _opt_import("core.agents.recon_profiles")
runner = _opt_import("core.agents.runner")
agents_types = _opt_import("core.agents.types")
LLMConfig = getattr(agents_types, "LLMConfig", None) if agents_types else None
trade_map = _opt_import("core.indicators.trade_map")

log = logging.getLogger("trap_intelligence")

st.set_page_config(page_title="TRAP INTELLIGENCE", page_icon="◆", layout="wide",
                   initial_sidebar_state="expanded")

CATEGORY_KEYS = tuple(CATEGORIES.keys())
CATEGORY_LABELS = {k: c.label for k, c in CATEGORIES.items()}
CATEGORY_TIMEFRAMES = {k: (c.chart_tf, c.context_tf) for k, c in CATEGORIES.items()}

MAX_TRADES = 20
NAV = (
    *[("desk", k, c.label, f"{CHART_VIEWS.get(k, (c.chart_tf, c.context_tf))[0]} chart · {c.context_tf} ctx")
      for k, c in CATEGORIES.items()],
    ("traps", "traps", "TRAP intelligence", "Sweep, anchored traps, watch list"),
    ("trade", "trade", "Active trade", "Positions, stops and targets"),
    ("director", "director", "Director report", "Run Analysis to generate"),
    ("macro", "macro", "Macro & news", "Headlines, gold, rates, odds"),
    ("qf", "qf", "Quant forecast", "Today's 24h plan, audited live"),
)
NAV_LABELS = [label for _kind, _key, label, _sub in NAV]
NAV_CAPTIONS = [sub for _kind, _key, _label, sub in NAV]
NAV_BY_LABEL = {label: (kind, key) for (kind, key, label, _sub) in NAV}
DESK_INDEX = {key: i + 1 for i, key in enumerate(CATEGORY_KEYS)}

_FALLBACK_CSS = """
<style>
.ti-kicker { font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.75; font-weight: 600; }
.ti-label { display: block; font-size: 0.72rem; letter-spacing: 0.06em; text-transform: uppercase; opacity: 0.7; margin: 0.35rem 0 0.2rem; }
.ti-card { border: 1px solid rgba(128,128,128,0.28); border-radius: 6px; padding: 0.85rem 1rem; margin-bottom: 0.8rem; position: relative; }
.ti-card.up { border-left: 3px solid #16a34a; }
.ti-card.down { border-left: 3px solid #dc2626; }
.ti-card.warn { border-left: 3px solid #d97706; }
.ti-gamma { display: flex; flex-wrap: wrap; gap: 0.6rem; align-items: center; padding: 0.65rem 0.9rem; border: 1px solid rgba(128,128,128,0.28); border-radius: 6px; margin-bottom: 0.85rem; }
.ti-hero { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; margin-bottom: 0.6rem; }
.ti-row-rule { padding: 0.55rem 0; border-bottom: 1px solid rgba(128,128,128,0.18); }
.ti-row-rule:last-child { border-bottom: none; }
.ti-split { display: flex; justify-content: space-between; align-items: center; gap: 0.5rem; }
.ti-chip { display: inline-block; padding: 0.12rem 0.45rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; background: rgba(128,128,128,0.16); }
.ti-chip.up { background: rgba(22,163,74,0.18); color: #16a34a; }
.ti-chip.down { background: rgba(220,38,38,0.18); color: #dc2626; }
.ti-chip.warn { background: rgba(217,119,6,0.18); color: #d97706; }
.ti-top { display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: baseline; }
.ti-top-price { font-size: 1.35rem; font-weight: 700; }
.ti-top-chg.up { color: #16a34a; font-weight: 600; }
.ti-top-chg.down { color: #dc2626; font-weight: 600; }
.ti-top-upd { font-size: 0.82rem; opacity: 0.7; }
.ti-feeds { display: flex; flex-direction: column; gap: 0.25rem; }
.ti-feed { display: flex; align-items: center; gap: 0.45rem; font-size: 0.82rem; }
.ti-feed .dot { width: 7px; height: 7px; border-radius: 50%; background: #16a34a; display: inline-block; }
.ti-feed.off .dot { background: #dc2626; }
.ti-feed .src { margin-left: auto; opacity: 0.7; font-size: 0.75rem; }
.muted { opacity: 0.72; }
</style>
"""


def _secrets_dict() -> dict:
    out = dict(os.environ)
    try:
        for k, v in st.secrets.items():
            if isinstance(v, (str, int, float, bool)):
                out[str(k)] = str(v)
            elif isinstance(v, dict):
                for sk, sv in v.items():
                    out[str(sk)] = str(sv)
    except Exception:  # noqa: BLE001
        pass
    return out


@st.cache_resource(show_spinner=False)
def _get_store():
    if _store_mod is None:
        return None
    for attr in ("get_store", "open_store", "connect", "Store"):
        factory = getattr(_store_mod, attr, None)
        if callable(factory):
            try:
                return factory()
            except TypeError:
                try:
                    secrets = _secrets_dict()
                    url = secrets.get("DATABASE_URL") or secrets.get("SUPABASE_DB_URL") or secrets.get("DB_PATH")
                    if url:
                        return factory(url)
                except Exception:  # noqa: BLE001
                    pass
            except Exception as exc:  # noqa: BLE001
                log.warning("store init via %s failed: %s", attr, exc)
    return _store_mod


def _render_html_frame(html_str: str, height: int = 54) -> None:
    if hasattr(st, "iframe"):
        try:
            st.iframe(html_str, height=height)
            return
        except Exception:  # noqa: BLE001
            pass
    components.html(html_str, height=height)


def _build_css(dark: bool, signal: bool) -> str:
    for fn_name in ("css", "build_css", "get_css", "stylesheet", "styles"):
        fn = getattr(theme, fn_name, None)
        if callable(fn):
            try:
                out = fn(dark, signal)
            except TypeError:
                try:
                    out = fn(dark=dark, signal=signal)
                except TypeError:
                    try:
                        out = fn(dark)
                    except TypeError:
                        out = fn()
            if isinstance(out, str) and out.strip():
                return out if "<style" in out else f"<style>{out}</style>"
    return _FALLBACK_CSS


def _run_coro_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    holder: dict[str, object] = {}
    err_holder: dict[str, BaseException] = {}

    def _worker():
        try:
            holder["val"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001
            err_holder["err"] = exc

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=60)
    if "err" in err_holder:
        raise err_holder["err"]
    return holder.get("val")


def _resolve_snapshot_loader():
    mods = []
    if core_data is not None:
        mods.append(core_data)
        pkg_path = getattr(core_data, "__path__", None)
        pkg_prefix = getattr(core_data, "__name__", "core.data") + "."
        if pkg_path is not None:
            for _finder, mod_name, _ispkg in pkgutil.walk_packages(pkg_path, pkg_prefix):
                m = _opt_import(mod_name)
                if m is not None:
                    mods.append(m)
    preferred = (
        "fetch_all",
        "fetch_snapshot",
        "load_snapshot",
        "get_snapshot",
        "build_snapshot",
        "collect_snapshot",
        "gather_snapshot",
        "fetch_market_snapshot",
        "fetch",
        "load_all",
        "collect",
        "snapshot",
    )
    for mod in mods:
        for name in preferred:
            fn = getattr(mod, name, None)
            if callable(fn):
                return fn
    return None


_SNAPSHOT_LOADER = _resolve_snapshot_loader()


def _fallback_snapshot(bucket_ms: int):
    def _feed(source="unavailable", **extra):
        base = {"available": False, "source": source, "error": "snapshot loader unavailable"}
        base.update(extra)
        return SimpleNamespace(**base)

    return SimpleNamespace(
        generated_at=datetime.fromtimestamp(bucket_ms / 1000.0, tz=timezone.utc),
        spot=_feed("spot", last=None, change_24h_pct=None, symbol="BTC/USDT", frames={}),
        futures=_feed("futures", funding_rate=None, funding_7d_mean=None, open_interest=None,
                     open_interest_usd=None, oi_change_24h_pct=None, long_short_ratio=None,
                     taker_buy_sell_ratio=None, oi_history=[]),
        options=_feed("options", max_pain=None, max_pain_expiry=None, put_call_oi_ratio=None,
                      iv_atm=None, iv_skew=None, expiries=[], gamma=None, gamma_by_desk={}),
        sentiment=_feed("sentiment", fear_greed=None, classification=None),
        hyperliquid=_feed("hyperliquid", funding_rate_1h=None, open_interest=None),
        liquidations=_feed("liquidations", total_usd=None, long_usd=None, short_usd=None,
                           btc_usd=None, btc_long_usd=None),
        whales=_feed("whales", btc_buy_usd=None, btc_sell_usd=None, window_minutes=None, btc_trades=None),
        stablecoins=_feed("stablecoins", total_usd=None, change_24h_pct=None),
        taker_flow=_feed("taker_flow", frames={}),
        calendar=_feed("calendar", events=[], upcoming=lambda _now, _win: []),
        macro=_feed("macro", series={}, spread_2s10s=None, delta_2y=None),
        fx=_feed("fx", dxy=None, dxy_as_of_ms=None, dxy_week_ago_ms=None, dxy_stale=False,
                 dxy_change_1w_pct=None, dxy_change_1m_pct=None, dxy_note=None, rate=None,
                 as_of_ms=None, stale=False, change_1w_pct=None, change_1m_pct=None, boj_rate_pct=None),
        metals=_feed("metals", xau_usd=None, unit="troy_ounce", bid=None, ask=None,
                     is_stale=False, computed_ms=None),
        dominance=_feed("dominance", btc_pct=None, eth_pct=None, stable_pct=None,
                        change_24h_pp=None, change_7d_pp=None, total_mcap_usd=None,
                        mcap_change_24h_pct=None, as_of_ms=None),
        news=_feed("news", headlines=[]),
    ), ["Fallback snapshot used"]


def _call_snapshot_loader(bucket_ms: int):
    if _SNAPSHOT_LOADER is None:
        return _fallback_snapshot(bucket_ms)
    sig = inspect.signature(_SNAPSHOT_LOADER)
    kwargs = {}
    if "now_ms" in sig.parameters:
        kwargs["now_ms"] = bucket_ms
    for p in sig.parameters.values():
        if p.name not in kwargs and p.default is inspect.Parameter.empty and p.kind not in (
            inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD
        ):
            pname = p.name.lower()
            ann = str(p.annotation).lower() if p.annotation is not inspect.Parameter.empty else ""
            if "dict" in ann or any(k in pname for k in ("cfg", "config", "settings", "secrets", "opts")):
                kwargs[p.name] = _secrets_dict()
            elif "int" in ann or "ms" in pname:
                kwargs[p.name] = bucket_ms
    try:
        res = _SNAPSHOT_LOADER(**kwargs)
    except Exception:  # noqa: BLE001
        res = _SNAPSHOT_LOADER()
    if inspect.isawaitable(res):
        res = _run_coro_sync(res)
    if isinstance(res, tuple) and len(res) == 2 and hasattr(res[0], "spot"):
        return res
    if hasattr(res, "spot"):
        return res, []
    return _fallback_snapshot(bucket_ms)


def _inject_css(dark: bool, signal: bool) -> None:
    st.markdown(_build_css(dark, signal), unsafe_allow_html=True)


def _now_ms() -> int:
    return int(time.time() * 1000)


@st.cache_resource(ttl=60, show_spinner=False)
def _load_snapshot_cached(bucket_ms: int):
    return _call_snapshot_loader(bucket_ms)


def _snapshot(force: bool = False) -> tuple:
    step = 60_000
    bucket = (_now_ms() // step) * step
    if force:
        _load_snapshot_cached.clear()
        bucket = _now_ms()
        st.session_state["snap_bucket"] = bucket
    elif "snap_bucket" not in st.session_state:
        st.session_state["snap_bucket"] = bucket
    else:
        bucket = st.session_state["snap_bucket"]
    return _load_snapshot_cached(bucket)


def _compute_analyses(m) -> dict[str, cat_ind.CategoryAnalysis | None]:
    frames = getattr(getattr(m, "spot", None), "frames", None) or {}
    futures = getattr(m, "futures", None)
    out: dict[str, cat_ind.CategoryAnalysis | None] = {}
    for key, cat in CATEGORIES.items():
        try:
            out[key] = cat_ind.analyze_category(cat, frames, futures)
        except Exception as exc:  # noqa: BLE001
            log.warning("analysis failed for %s: %s", key, exc)
            out[key] = None
    return out


def _run_pipeline_pass(store, m, analyses: dict, now_ms: int) -> None:
    if store is None or core_pipeline is None:
        return
    valid_analyses = {k: v for k, v in analyses.items() if v is not None}
    try:
        res = core_pipeline.ledger(store, m, valid_analyses, now_ms)
        for prob in res.problems:
            log.warning("pipeline %s", prob)
    except Exception as exc:  # noqa: BLE001
        log.warning("pipeline ledger failed: %s", exc)


def _llm_config_from_env():
    if LLMConfig is None:
        return None
    secrets = _secrets_dict()
    for provider, env_key, default_model in (
        ("openrouter", "OPENROUTER_API_KEY", "deepseek/deepseek-chat-v3-0324:free"),
        ("groq", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
        ("gemini", "GEMINI_API_KEY", "gemini-2.5-flash"),
    ):
        key = (secrets.get(env_key) or "").strip()
        if key:
            return LLMConfig(provider=provider, api_key=key, model=default_model)
    return None


def _render_sidebar(m, analyses: dict) -> tuple[bool, bool, str, bool]:
    if "dark_mode" not in st.session_state:
        st.session_state["dark_mode"] = False
    if "signal_colours" not in st.session_state:
        st.session_state["signal_colours"] = False

    st.sidebar.markdown("<span class='ti-kicker'>TRAP INTELLIGENCE</span>", unsafe_allow_html=True)
    st.sidebar.caption("BTC market-surveillance terminal")

    dark = st.sidebar.toggle("Dark mode", key="dark_mode")
    signal = st.sidebar.toggle("Signal colours", key="signal_colours",
                               help="Green/red for moves and bullish/bearish; steel and amber otherwise.")
    _inject_css(dark, signal)

    tones, tfs = [], []
    for key in CATEGORY_KEYS:
        a = analyses.get(key)
        tones.append(panels.rail_tone(None, a))
        tfs.append(CHART_VIEWS.get(key, CATEGORY_TIMEFRAMES[key])[0])
    extra = len(NAV) - len(CATEGORY_KEYS)
    tones.extend(["neutral"] * extra)
    tfs.extend([""] * extra)

    st.sidebar.markdown("<span class='ti-label'>Desks & Engines</span>", unsafe_allow_html=True)
    st.sidebar.markdown(panels.rail_style_html(tones, tfs, dark, signal), unsafe_allow_html=True)
    choice = st.sidebar.radio(
        "Navigate",
        NAV_LABELS,
        captions=NAV_CAPTIONS,
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    try:
        st.sidebar.markdown(panels.data_status_html(m), unsafe_allow_html=True)
    except Exception as exc:  # noqa: BLE001
        st.sidebar.caption(f"Feed status unavailable: {exc}")

    st.sidebar.markdown("---")
    with st.sidebar:
        _render_html_frame(panels.clock_html(dark), height=54)

    force_refresh = False
    if st.sidebar.button("Force fresh snapshot", use_container_width=True):
        force_refresh = True
    return dark, signal, choice, force_refresh


def _render_topbar(m, now_ms: int) -> bool:
    c1, c2, c3 = st.columns([5, 2, 2])
    price = getattr(m.spot, "last", None)
    chg = getattr(m.spot, "change_24h_pct", None)
    updated = f"Updated {fmt_when(now_ms)}"
    symbol = getattr(m.spot, "symbol", None)
    with c1:
        panels.render(panels.top_bar_html(price, chg, updated, symbol))
    refresh_clicked = False
    with c2:
        if st.button("REFRESH DATA", use_container_width=True):
            refresh_clicked = True
    with c3:
        run_clicked = st.button("RUN ANALYSIS", type="primary", use_container_width=True)
    if run_clicked:
        st.session_state["run_analysis_requested"] = True
    return refresh_clicked


def _render_desk(key: str, m, analyses: dict, now_ms: int, dark: bool, signal: bool) -> None:
    base_a = analyses.get(key)
    cat = CATEGORIES[key]
    if base_a is None:
        st.warning(f"No analysis available for {cat.label}.")
        return

    for_cat = getattr(recon_profiles, "FOR_CATEGORY", None) or getattr(recon_profiles, "PROFILES", None) or {}
    profile = for_cat.get(key) or getattr(recon_profiles, "LIVE", None)
    if profile is not None:
        panels.render(panels.desk_hero_html(DESK_INDEX[key], profile, base_a, now_ms))
    gamma = desk_derivs.gamma_for(key, m)
    panels.render(panels.gamma_bar_html(gamma, getattr(m.spot, "last", None), cat.label))

    views = CHART_VIEWS.get(key, (cat.chart_tf, cat.context_tf))
    view_tf = st.radio(
        "Chart candle",
        list(views),
        horizontal=True,
        key=f"chart_view_{key}",
    )
    frames = getattr(getattr(m, "spot", None), "frames", None) or {}
    chart_a = cat_ind.view_analysis(cat, view_tf, frames, getattr(m, "futures", None), base_a) or base_a

    fg = getattr(m.sentiment, "fear_greed", None)
    derivs = desk_derivs.read(key, m, now_ms)

    left, right = st.columns([7, 5], gap="large")
    with left:
        panels.render(panels.chart_head_html(chart_a, signal=signal, symbol=getattr(m.spot, "symbol", None)))
        st.plotly_chart(charts.price_chart(chart_a, dark=dark, signal=signal), use_container_width=True,
                        config={"displayModeBar": False})
        panels.render(panels.layman_html(chart_a))
        panels.render(panels.patterns_html(chart_a))
        panels.render(panels.divergence_html(chart_a))
    with right:
        panels.render(panels.direction_html(base_a))
        panels.render(panels.volatility_html(base_a))
        panels.render(panels.fib_html(chart_a))

    st.markdown("---")
    panels.render(panels.layman_summary_html(base_a, cat.label, derivs, fg, now_ms))


def _render_traps(store, m, analyses: dict, now_ms: int) -> None:
    panels.render(panels.engine_hero_html(
        5, "TRAP Intelligence",
        "Sweep timeframes for engineered bull/bear traps, monitor anchored traps, and review settled history."
    ))
    gamma = desk_derivs.gamma_for(None, m)
    panels.render(panels.gamma_bar_html(gamma, getattr(m.spot, "last", None)))

    if traps is None:
        st.info("TRAP intelligence module not present in this build.")
        return

    rows = []
    if store is not None and hasattr(store, "traps"):
        rows = store.traps(limit=200)
    elif hasattr(traps, "all_traps"):
        rows = traps.all_traps()

    candidates = st.session_state.get("trap_candidates") or []
    swept_ms = st.session_state.get("trap_swept_ms")
    judged = st.session_state.get("trap_judged") or []
    settled = [r for r in rows if r.get("status") != "active"][:12]
    plan = traps.sweep_plan() if hasattr(traps, "sweep_plan") else []

    if st.button("Sweep timeframes now", use_container_width=False):
        try:
            candidates, judged = traps.scan_and_judge(m, analyses, now_ms)
            swept_ms = now_ms
            st.session_state["trap_candidates"] = candidates
            st.session_state["trap_swept_ms"] = swept_ms
            st.session_state["trap_judged"] = judged
            if store is not None and hasattr(store, "traps"):
                rows = store.traps(limit=200)
            elif hasattr(traps, "all_traps"):
                rows = traps.all_traps()
            settled = [r for r in rows if r.get("status") != "active"][:12]
        except Exception as exc:  # noqa: BLE001
            st.error(f"Sweep failed: {exc}")

    panels.render(panels.trap_console_html(rows, candidates, now_ms, CATEGORY_LABELS, settled, swept_ms, plan))
    active = [r for r in rows if r.get("status") == "active"]
    price_now = getattr(m.spot, "last", None)
    if active:
        for r in active:
            panels.render(panels.trap_card_html(r, now_ms, CATEGORY_LABELS.get(r.get("category", ""), ""), price_now))
    panels.render(panels.trap_watch_html(candidates, now_ms, swept_ms, judged))
    panels.render(panels.trap_settled_html(settled, now_ms, CATEGORY_LABELS))


def _render_trade(store, m, analyses: dict, now_ms: int) -> None:
    panels.render(panels.engine_hero_html(
        6, "Active Trade",
        "Enter a position and it is evaluated, pinned and tracked until you kill it. Levels are computed in code first; "
        "ten sub-agents, the four category heads and the TRAP head are consulted second. Cards live in the database "
        "and survive a refresh and a redeploy."
    ))
    gamma = desk_derivs.gamma_for(None, m)
    panels.render(panels.gamma_bar_html(gamma, getattr(m.spot, "last", None)))
    if hasattr(panels, "master_lanes_html"):
        panels.render(panels.master_lanes_html(analyses, m, now_ms))

    st.subheader("Open a position")
    spot_px = float(getattr(m.spot, "last", None) or 60000.0)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _asset = st.text_input("Asset", value="BTC/USDT")
    with c2:
        side = st.selectbox("Direction", ["LONG", "SHORT"])
    with c3:
        entry = st.number_input("Entry ($) · defaults to spot", value=round(spot_px, 2), min_value=1.0, step=50.0)
    with c4:
        stop_in = st.number_input("Stop loss ($) · 0 lets the desk place it", value=0.0, min_value=0.0, step=50.0)

    horizon_key = st.radio(
        "Judge it over",
        list(CATEGORY_KEYS),
        format_func=lambda k: {
            "live": "Live Recon · four hours",
            "intraday": "Intraday · 24 hours",
            "weekly": "Weekly · seven days",
            "monthly": "Monthly · 30 days",
        }.get(k, k),
        horizontal=True,
    )

    if st.button("Open and track this trade", type="primary"):
        if trade_map is None or not hasattr(trade_map, "build_trade_map"):
            st.warning("Trade map builder not available in this build.")
        else:
            try:
                tm = trade_map.build_trade_map(
                    m=m,
                    analyses=analyses,
                    category=horizon_key,
                    side=side,
                    entry=float(entry),
                    stop=float(stop_in) if stop_in > 0 else None,
                    leverage=5.0,
                )
                panels.render(panels.trade_result_html(tm))
                st.session_state["last_trade_map"] = tm
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not evaluate trade: {exc}")

    if "last_trade_map" in st.session_state:
        panels.render(panels.trade_result_html(st.session_state["last_trade_map"]))

    open_trades = []
    if store is not None:
        if hasattr(store, "trades"):
            open_trades = store.trades(status="open", limit=MAX_TRADES)
        elif hasattr(store, "open_trades"):
            open_trades = store.open_trades()
    if not open_trades:
        st.caption("No position is being tracked. Open one above and it stays pinned here.")
    else:
        for t in open_trades[:MAX_TRADES]:
            panels.render(panels.trade_card_html(t, now_ms=now_ms, price_now=getattr(m.spot, "last", None)))


def _render_director(store, m, analyses: dict, now_ms: int) -> None:
    panels.render(panels.engine_hero_html(
        7, "Director Report",
        "Multi-agent synthesis across technicals, order flow, derivatives, and macro."
    ))
    report = st.session_state.get("latest_report")
    if report is None and store is not None:
        if hasattr(store, "reports"):
            reps = store.reports(limit=1)
            report = reps[0] if reps else None
        elif hasattr(store, "latest_report"):
            report = store.latest_report()

    if st.session_state.pop("run_analysis_requested", False):
        cfg = _llm_config_from_env()
        if cfg is None:
            st.warning("No LLM API key configured in environment (OPENROUTER_API_KEY, GROQ_API_KEY, or GEMINI_API_KEY).")
        elif runner is None or not hasattr(runner, "run_15_agent"):
            st.warning("Multi-agent runner not available in this build.")
        else:
            with st.spinner("Running multi-agent analysis..."):
                try:
                    report = runner.run_15_agent(cfg, m, analyses, now_ms=now_ms)
                    st.session_state["latest_report"] = report
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Analysis run failed: {exc}")

    if report is None:
        st.info("Run Analysis to generate a Director report.")
        return

    panels.render(panels.report_stats_html(report))
    panels.render(panels.director_html(report))
    panels.render(panels.director_lists_html(report))
    panels.render(panels.director_desks_html(report, CATEGORY_LABELS, {k: c.chart_tf for k, c in CATEGORIES.items()}))
    panels.render(panels.briefs_html(report))


def _render_macro(m, now_ms: int) -> None:
    panels.render(panels.engine_hero_html(
        8, "Macro & News",
        "High-impact USD releases, gold, US rates, dollar index, yen carry, and headlines."
    ))
    c1, c2 = st.columns(2, gap="large")
    with c1:
        panels.render(panels.calendar_html(m, now_ms))
        panels.render(panels.rates_html(getattr(m, "macro", None), now_ms))
        panels.render(panels.dollar_html(getattr(m, "fx", None), getattr(m, "spot", None), now_ms))
    with c2:
        panels.render(panels.gold_html(getattr(m, "metals", None), now_ms))
        panels.render(panels.yen_html(getattr(m, "fx", None), getattr(m, "macro", None), now_ms))
        panels.render(panels.dominance_html(getattr(m, "dominance", None), now_ms))
    panels.render(panels.news_html(getattr(m, "news", None), now_ms))


def _render_qf(store, m, analyses: dict, now_ms: int) -> None:
    panels.render(panels.engine_hero_html(
        9, "Quant Forecast",
        "24-hour quantitative plan, volatility windows, timeline, and live self-audit."
    ))
    if quant_forecast is None or qf_alerts is None:
        st.info("Quant forecast module not present in this build.")
        return
    try:
        try:
            plan = quant_forecast.ensure_plan(store, m, analyses, now_ms)
        except TypeError:
            plan = quant_forecast.ensure_plan(m, analyses, now_ms)
        live = quant_forecast.live_state(plan, m, now_ms)
        status, msgs = qf_alerts.evaluate(plan, live, m, now_ms)
        panels.render(panels.qf_wire_html(plan, status=status, live=live))
        panels.render(panels.qf_directive_html(plan))
        panels.render(panels.qf_schedule_html(plan, now_ms))
        panels.render(panels.qf_timeline_html(plan, now_ms))
        panels.render(panels.qf_status_html(status, msgs, [plan]))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Quant forecast unavailable: {exc}")


def main() -> None:
    now_ms = _now_ms()
    store = _get_store()
    m, _errs = _snapshot(force=False)
    analyses = _compute_analyses(m)
    _run_pipeline_pass(store, m, analyses, now_ms)

    dark, signal, choice, force_refresh = _render_sidebar(m, analyses)
    if force_refresh:
        m, _errs = _snapshot(force=True)
        analyses = _compute_analyses(m)
        now_ms = _now_ms()

    top_refresh = _render_topbar(m, now_ms)
    if top_refresh:
        m, _errs = _snapshot(force=True)
        analyses = _compute_analyses(m)
        now_ms = _now_ms()
        st.rerun()

    kind, key = NAV_BY_LABEL[choice]
    if kind == "desk":
        _render_desk(key, m, analyses, now_ms, dark, signal)
    elif kind == "traps":
        _render_traps(store, m, analyses, now_ms)
    elif kind == "trade":
        _render_trade(store, m, analyses, now_ms)
    elif kind == "director":
        _render_director(store, m, analyses, now_ms)
    elif kind == "macro":
        _render_macro(m, now_ms)
    elif kind == "qf":
        _render_qf(store, m, analyses, now_ms)


if __name__ == "__main__":
    main()
