
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

from core import pipeline as core_pipeline  # noqa: E402
from core import qf_alerts, quant_forecast  # noqa: E402
from core.agents import live_recon, recon_profiles  # noqa: E402
from core.config import CATEGORIES, CHART_VIEWS  # noqa: E402
from core.indicators import (  # noqa: E402
    anomaly_trap,
    cascade,
    desk_derivs,
    early_warning,
    forecast,
    harmonics,
    liquidity,
    smc,
    volatility,
    vwap_break,
    weekend,
)
from core.indicators import category as cat_ind  # noqa: E402
from core.trades import fmt_when  # noqa: E402
from ui import charts, panels, theme  # noqa: E402


def _opt_import(module_path: str):
    try:
        return importlib.import_module(module_path)
    except Exception:  # noqa: BLE001
        return None


core_data = _opt_import("core.data")
accuracy = _opt_import("core.accuracy")
_store_mod = _opt_import("core.store")
traps = _opt_import("core.traps")
runner = _opt_import("core.agents.runner")
judge_mod = _opt_import("core.agents.judge")
agents_types = _opt_import("core.agents.types")
LLMConfig = getattr(agents_types, "LLMConfig", None) if agents_types else None
trade_map = _opt_import("core.indicators.trade_map")
sine_mod = _opt_import("core.indicators.sine_wave")
handshake_mod = _opt_import("core.agents.handshake")

log = logging.getLogger("trap_intelligence")

st.set_page_config(
    page_title="TRAP INTELLIGENCE",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

CATEGORY_KEYS = tuple(CATEGORIES.keys())
CATEGORY_LABELS = {k: c.label for k, c in CATEGORIES.items()}
CATEGORY_TIMEFRAMES = {k: (c.chart_tf, c.context_tf) for k, c in CATEGORIES.items()}

# Master Trades are strictly two swing tiers: Intraday/Swing (12h/1d) and Weekly/Swing (1d/1w).
MASTER_TRADE_DESKS: tuple[tuple[str, str, str], ...] = (
    ("intraday", "Master Trade I · Intraday / Swing", "12h chart · 1d anchor · $1,000–$3,000+ BTC swing tier"),
    ("weekly", "Master Trade II · Weekly / Structural Swing", "1d chart · 1w anchor · multi-day structural swing tier"),
)

MAX_TRADES = 20
NAV = (
    *[
        ("desk", k, c.label, f"{CHART_VIEWS.get(k, (c.chart_tf, c.context_tf))[0]} chart · {c.context_tf} ctx")
        for k, c in CATEGORIES.items()
    ],
    ("traps", "traps", "TRAP intelligence", "Sweep, anchored traps, watch list"),
    ("trade", "trade", "Master trades", "Intraday & Weekly confluence swing setups"),
    ("director", "director", "Director report", "15-agent synthesis & desk table"),
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
    components.html(html_str, height=height, scrolling=False)


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
    t.join(timeout=90)
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
        futures=_feed(
            "futures",
            funding_rate=None,
            funding_7d_mean=None,
            open_interest=None,
            open_interest_usd=None,
            oi_change_24h_pct=None,
            long_short_ratio=None,
            taker_buy_sell_ratio=None,
            oi_history=[],
        ),
        options=_feed(
            "options",
            max_pain=None,
            max_pain_expiry=None,
            put_call_oi_ratio=None,
            iv_atm=None,
            iv_skew=None,
            expiries=[],
            gamma=None,
            gamma_by_desk={},
        ),
        sentiment=_feed("sentiment", fear_greed=None, classification=None),
        hyperliquid=_feed("hyperliquid", funding_rate_1h=None, open_interest=None),
        liquidations=_feed(
            "liquidations",
            total_usd=None,
            long_usd=None,
            short_usd=None,
            btc_usd=None,
            btc_long_usd=None,
        ),
        whales=_feed("whales", btc_buy_usd=None, btc_sell_usd=None, window_minutes=None, btc_trades=None),
        stablecoins=_feed("stablecoins", total_usd=None, change_24h_pct=None),
        taker_flow=_feed("taker_flow", frames={}),
        calendar=_feed("calendar", events=[], upcoming=lambda _now, _win: []),
        macro=_feed("macro", series={}, spread_2s10s=None, delta_2y=None),
        fx=_feed(
            "fx",
            dxy=None,
            dxy_as_of_ms=None,
            dxy_week_ago_ms=None,
            dxy_stale=False,
            dxy_change_1w_pct=None,
            dxy_change_1m_pct=None,
            dxy_note=None,
            rate=None,
            as_of_ms=None,
            stale=False,
            change_1w_pct=None,
            change_1m_pct=None,
            boj_rate_pct=None,
        ),
        metals=_feed("metals", xau_usd=None, unit="troy_ounce", bid=None, ask=None, is_stale=False, computed_ms=None),
        dominance=_feed(
            "dominance",
            btc_pct=None,
            eth_pct=None,
            stable_pct=None,
            change_24h_pp=None,
            change_7d_pp=None,
            total_mcap_usd=None,
            mcap_change_24h_pct=None,
            as_of_ms=None,
        ),
        news=_feed("news", headlines=[]),
        unavailable=lambda: [],
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
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
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


def _run_pipeline_pass(store, m, analyses: dict, now_ms: int) -> dict:
    if store is None:
        return {}
    valid_analyses = {k: v for k, v in analyses.items() if v is not None}
    secrets = _secrets_dict()
    may_send = bool(
        (secrets.get("DISCORD_WEBHOOK_URL") or "").strip()
        or (secrets.get("DISCORD_BOT_TOKEN") or "").strip()
    )
    try:
        res = core_pipeline.ledger(store, m, valid_analyses, now_ms)
        core_pipeline.market_alerts(store, m, valid_analyses, now_ms, secrets, may_send=may_send, result=res)
        px = getattr(m.spot, "last", None)
        watch_fn = core_pipeline.desk_watcher(m, valid_analyses, now_ms)
        core_pipeline.all_trade_alerts(store, px, now_ms, secrets, may_send=may_send, result=res, watch_for=watch_fn)
        core_pipeline.closed_trade_alerts(store, now_ms, secrets, may_send=may_send, result=res)
        for prob in res.problems:
            log.warning("pipeline %s", prob)
        return res.verdicts or {}
    except Exception as exc:  # noqa: BLE001
        log.warning("pipeline pass failed: %s", exc)
        return {}


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


def _build_judge(cfg):
    if judge_mod is None or cfg is None:
        return None
    JudgeCls = getattr(judge_mod, "Judge", None)
    if JudgeCls is None:
        return None
    try:
        return JudgeCls(cfg)
    except Exception:  # noqa: BLE001
        for factory_name in ("from_config", "build_judge", "create_judge"):
            fn = getattr(judge_mod, factory_name, None)
            if callable(fn):
                try:
                    return fn(cfg)
                except Exception:  # noqa: BLE001
                    pass
    return None


def _execute_full_ai_run(store, m, analyses: dict, now_ms: int) -> None:
    """Run the 11-agent war room across all 4 desks (anchored until invalidated) + 15-agent Director report."""
    cfg = _llm_config_from_env()
    if cfg is None:
        st.warning("No LLM API key found in environment (OPENROUTER_API_KEY, GROQ_API_KEY, or GEMINI_API_KEY).")
        return

    valid_analyses = {k: v for k, v in analyses.items() if v is not None}
    now_dt = datetime.fromtimestamp(now_ms / 1000.0, tz=timezone.utc)

    with st.spinner("Running 11-agent confluence war rooms across Live Recon, Intraday, Weekly & Monthly..."):
        judge = _build_judge(cfg)
        if judge is not None:
            async def _run_all_desks():
                tasks = [
                    live_recon.run_live_recon(judge, m, valid_analyses, now=now_dt, profile=recon_profiles.PROFILES[k])
                    for k in CATEGORY_KEYS
                ]
                return await asyncio.gather(*tasks, return_exceptions=True)

            results = _run_coro_sync(_run_all_desks()) or []
            war_rooms: dict[str, live_recon.LiveReconResult] = {}
            for k, res in zip(CATEGORY_KEYS, results):
                if isinstance(res, Exception):
                    log.warning("war room failed for %s: %s", k, res)
                elif res is not None:
                    war_rooms[k] = res
            if handshake_mod is not None and hasattr(handshake_mod, "apply"):
                try:
                    handshake_mod.apply(war_rooms)
                except Exception as exc:  # noqa: BLE001
                    log.warning("handshake failed: %s", exc)
            st.session_state["desk_war_rooms"] = war_rooms
            st.session_state["desk_war_rooms_ms"] = now_ms

        if runner is not None and hasattr(runner, "run_15_agent"):
            try:
                report = runner.run_15_agent(cfg, m, valid_analyses, now_ms=now_ms)
                st.session_state["latest_report"] = report
            except Exception as exc:  # noqa: BLE001
                log.warning("15-agent director failed: %s", exc)


def _safe_rail_tone(v, a) -> str:
    fn = getattr(panels, "rail_tone", None)
    if callable(fn):
        try:
            return fn(v, a)
        except Exception:  # noqa: BLE001
            pass
    d = ""
    if v is not None and getattr(v, "anchored", None) is not None:
        d = str(getattr(v.anchored, "direction", "") or "").upper()
    elif a is not None and getattr(a, "direction", None) is not None:
        d = str(getattr(a.direction, "direction", "") or "").upper()
    if "BULL" in d or "LONG" in d:
        return "up"
    if "BEAR" in d or "SHORT" in d:
        return "down"
    return "neutral"


def _render_sidebar(m, analyses: dict, verdicts: dict) -> tuple[bool, bool, str, bool]:
    if "dark_mode" not in st.session_state:
        st.session_state["dark_mode"] = False
    if "signal_colours" not in st.session_state:
        st.session_state["signal_colours"] = False

    st.sidebar.markdown("<span class='ti-kicker'>TRAP INTELLIGENCE</span>", unsafe_allow_html=True)
    st.sidebar.caption("BTC market-surveillance terminal")

    dark = st.sidebar.toggle("Dark mode", key="dark_mode")
    signal = st.sidebar.toggle(
        "Signal colours",
        key="signal_colours",
        help="Green/red for moves and bullish/bearish; steel and amber otherwise.",
    )
    _inject_css(dark, signal)

    tones, tfs = [], []
    for key in CATEGORY_KEYS:
        a = analyses.get(key)
        v = verdicts.get(key)
        tones.append(_safe_rail_tone(v, a))
        tfs.append(CHART_VIEWS.get(key, CATEGORY_TIMEFRAMES[key])[0])
    extra = len(NAV) - len(CATEGORY_KEYS)
    tones.extend(["neutral"] * extra)
    tfs.extend([""] * extra)

    st.sidebar.markdown("<span class='ti-label'>Desks & Engines</span>", unsafe_allow_html=True)
    if hasattr(panels, "rail_style_html"):
        try:
            st.sidebar.markdown(panels.rail_style_html(tones, tfs, dark, signal), unsafe_allow_html=True)
        except Exception:  # noqa: BLE001
            pass
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
        if st.button("RUN ANALYSIS", type="primary", use_container_width=True):
            st.session_state["run_analysis_requested"] = True
    return refresh_clicked


def _render_kpi_strip(kpis: list[dict]) -> None:
    if not kpis:
        return
    for start in range(0, len(kpis), 6):
        chunk = kpis[start : start + 6]
        cols = st.columns(len(chunk))
        for col, item in zip(cols, chunk):
            with col:
                st.metric(
                    label=str(item.get("label") or ""),
                    value=str(item.get("value") or "—"),
                    delta=item.get("delta"),
                )


def _render_desk(key: str, store, m, analyses: dict, verdicts: dict, now_ms: int, dark: bool, signal: bool) -> None:
    base_a = analyses.get(key)
    cat = CATEGORIES[key]
    if base_a is None:
        st.warning(f"No analysis available for {cat.label}.")
        return

    profile = recon_profiles.PROFILES[key]
    panels.render(panels.desk_hero_html(DESK_INDEX[key], profile, base_a, now_ms))

    # Top banners: weekend tape, active TRAP warnings, anomaly traps, early warnings, squeeze
    try:
        w_banner = panels.weekend_html(weekend.read(now_ms))
        if w_banner:
            panels.render(w_banner)
    except Exception:  # noqa: BLE001
        pass

    if store is not None and hasattr(store, "traps"):
        try:
            desk_tfs = set(profile.read_order)
            active_traps = [
                r
                for r in store.traps(limit=100)
                if r.get("status") == "active" and (not desk_tfs or r.get("timeframe") in desk_tfs)
            ]
            t_banner = panels.trap_warning_html(active_traps, now_ms)
            if t_banner:
                panels.render(t_banner)
        except Exception:  # noqa: BLE001
            pass

    derivs = desk_derivs.read(key, m, now_ms)
    frames = getattr(getattr(m, "spot", None), "frames", None) or {}
    oi_hist = desk_derivs.oi_history(m)
    flow_frames = (
        getattr(m.taker_flow, "frames", None)
        if getattr(getattr(m, "taker_flow", None), "available", False)
        else {}
    )

    # Compute multi-timeframe confluence blocks
    smc_by_tf, liq_by_tf, vol_by_tf, vb_by_tf, harm_by_tf = {}, {}, {}, {}, {}
    gamma = desk_derivs.gamma_for(key, m)
    for tf in profile.read_order:
        df = frames.get(tf)
        if df is None or df.empty:
            continue
        smc_by_tf[tf] = smc.summarise(df)
        liq_by_tf[tf] = liquidity.summarise(
            df,
            oi_hist,
            now_ms,
            windows=profile.timeframes,
            flow=(flow_frames or {}).get(tf),
        )
        vol_by_tf[tf] = volatility.matrix_summary(df, tf, getattr(m, "options", None))
        if tf in vwap_break.TIMEFRAMES:
            vb_by_tf[tf] = vwap_break.read(df, tf).to_state()
        if tf in harmonics.TIMEFRAMES:
            try:
                harm_by_tf[tf] = [h.to_state() for h in harmonics.find(df, tf, gamma)]
            except Exception:  # noqa: BLE001
                harm_by_tf[tf] = []

    try:
        fg = m.sentiment.fear_greed if getattr(getattr(m, "sentiment", None), "available", False) else None
        fut = m.futures if getattr(getattr(m, "futures", None), "available", False) else None
        at = anomaly_trap.read(
            fg,
            fut,
            derivs,
            liq_by_tf.get(cat.chart_tf),
            base_a.price_change_24h_pct,
            oi_hist,
            now_ms,
        )
        if at is not None:
            panels.render(panels.anomaly_trap_html(at))
    except Exception:  # noqa: BLE001
        pass

    try:
        if getattr(getattr(m, "futures", None), "available", False):
            last_sq = getattr(getattr(base_a, "squeeze", None), "score", None)
            ews = early_warning.early_warnings(base_a, m.futures, last_sq)
            ew_banner = panels.early_warning_html(ews)
            if ew_banner:
                panels.render(ew_banner)
    except Exception:  # noqa: BLE001
        pass

    sq_banner = panels.squeeze_banner_html(base_a)
    if sq_banner:
        panels.render(sq_banner)

    panels.render(panels.gamma_bar_html(gamma, getattr(m.spot, "last", None), cat.label))
    _render_kpi_strip(panels.kpis_for(base_a, m, derivs))

    # Desk Forecast (projected path + record)
    fc, fc_record = None, None
    try:
        fc, fc_record = forecast.desk_forecast(base_a, gamma, frames.get(cat.chart_tf))
    except Exception as exc:  # noqa: BLE001
        log.warning("desk_forecast failed for %s: %s", key, exc)

    views = CHART_VIEWS.get(key, (cat.chart_tf, cat.context_tf))
    view_tf = st.radio(
        "Chart candle",
        list(views),
        horizontal=True,
        key=f"chart_view_{key}",
    )
    chart_a = cat_ind.view_analysis(cat, view_tf, frames, getattr(m, "futures", None), base_a) or base_a

    left, right = st.columns([7, 5], gap="large")
    with left:
        panels.render(
            panels.chart_head_html(
                chart_a,
                signal=signal,
                symbol=getattr(m.spot, "symbol", None),
                call_tf=cat.chart_tf,
            )
        )
        if fc is not None and getattr(fc, "available", False):
            charts.render_projected_chart(
                chart_a,
                dark=dark,
                forecast=fc,
                gamma=gamma,
                forecast_tf=cat.chart_tf,
                height=560,
                signal=signal,
            )
        else:
            charts.render_chart(chart_a, dark=dark, height=560, signal=signal)
        panels.render(panels.layman_html(chart_a))
        panels.render(panels.patterns_html(chart_a))
        panels.render(panels.divergence_html(chart_a))
    with right:
        verdict = verdicts.get(key)
        if verdict is not None:
            panels.render(panels.anchored_direction_html(base_a, verdict, now_ms))
        else:
            panels.render(panels.direction_html(base_a))
        panels.render(panels.volatility_html(base_a))
        panels.render(panels.fib_html(chart_a))

    # 11-Agent War Room Anchored Partner Summary (persists until invalidated)
    war_rooms = st.session_state.get("desk_war_rooms") or {}
    wr = war_rooms.get(key)
    if wr is not None:
        st.markdown("---")
        verdict = verdicts.get(key)
        if verdict is not None and hasattr(panels, "anchored_summary_html"):
            panels.render(panels.anchored_summary_html(wr, verdict, now_ms))
        elif hasattr(panels, "agent_summary_html"):
            panels.render(panels.agent_summary_html(wr, now_ms))
        w1, w2 = st.columns(2, gap="large")
        with w1:
            if hasattr(panels, "team_html"):
                panels.render(panels.team_html(wr.bull, signal=signal))
        with w2:
            if hasattr(panels, "team_html"):
                panels.render(panels.team_html(wr.bear, signal=signal))
        if hasattr(panels, "trap_audit_html"):
            panels.render(panels.trap_audit_html(wr))
    else:
        st.info(
            f"Click **RUN ANALYSIS** in the top bar to run the 11-agent confluence partner verdict for {cat.label} (held until invalidated)."
        )

    # Timeframe cascade
    try:
        bias_tf, anchor_tf, engine_tfs = profile.cascade
        c_state = cascade.classify(frames, bias_tf, anchor_tf, engine_tfs)
        inval = verdicts[key].anchored.invalidation if verdicts.get(key) is not None else base_a.direction.invalidation
        panels.render(panels.cascade_html(c_state, invalidation=inval, price=base_a.price))
    except Exception as exc:  # noqa: BLE001
        log.warning("cascade panel failed for %s: %s", key, exc)

    # Forecast path & hit-rate table
    if fc is not None:
        try:
            panels.render(panels.forecast_html(fc, fc_record, cat.horizon_label))
        except Exception as exc:  # noqa: BLE001
            log.warning("forecast panel failed for %s: %s", key, exc)

    # Multi-timeframe confluence matrices (SMC, Liquidity, Harmonics, Value Breaks, Volatility Matrix)
    c_a, c_b = st.columns(2, gap="large")
    with c_a:
        if smc_by_tf:
            panels.render(panels.smc_html(smc_by_tf))
        if harm_by_tf:
            panels.render(panels.harmonics_html(harm_by_tf))
    with c_b:
        if liq_by_tf:
            panels.render(panels.liquidity_html(liq_by_tf))
        if vb_by_tf:
            panels.render(panels.value_breaks_html(vb_by_tf))

    if vol_by_tf:
        panels.render(panels.vol_matrix_html(vol_by_tf))

    if key == "live" and sine_mod is not None and hasattr(sine_mod, "evaluate"):
        try:
            warn, ny_now, rec = sine_mod.evaluate(store, frames, now_ms)
            panels.render(panels.sine_experimental_html(warn, ny_now, rec))
        except Exception:  # noqa: BLE001
            pass

    st.markdown("---")
    fg_val = getattr(m.sentiment, "fear_greed", None)
    if hasattr(panels, "layman_summary_html"):
        try:
            panels.render(panels.layman_summary_html(base_a, cat.label, derivs, fg_val, now_ms))
        except Exception as exc:  # noqa: BLE001
            log.warning("layman_summary_html failed for %s: %s", key, exc)


def _render_traps(store, m, analyses: dict, now_ms: int) -> None:
    panels.render(
        panels.engine_hero_html(
            5,
            "TRAP Intelligence",
            "Sweep timeframes for engineered bull/bear traps, monitor anchored traps, and review settled history.",
        )
    )
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


def _render_trade(store, m, analyses: dict, verdicts: dict, now_ms: int) -> None:
    panels.render(
        panels.engine_hero_html(
            6,
            "Master Swing Trades",
            "Confluence across all feeds distilled into two master swing setups only: (1) Intraday / Swing (12h/1d) and "
            "(2) Weekly / Structural Swing (1d/1w). Category desks provide directional confluence until invalidated; "
            "actual trade execution lives strictly in these two master lanes.",
        )
    )
    gamma = desk_derivs.gamma_for(None, m)
    panels.render(panels.gamma_bar_html(gamma, getattr(m.spot, "last", None)))

    if hasattr(panels, "master_lanes_html"):
        panels.render(panels.master_lanes_html(analyses, m, now_ms))

    spot_px = float(getattr(m.spot, "last", None) or 0.0)
    war_rooms = st.session_state.get("desk_war_rooms") or {}

    st.subheader("Confluence Master Setups (Intraday Swing & Weekly Swing)")
    cols = st.columns(2, gap="large")
    for col, (desk_key, title, subtitle) in zip(cols, MASTER_TRADE_DESKS):
        with col:
            a = analyses.get(desk_key)
            v = verdicts.get(desk_key)
            wr = war_rooms.get(desk_key)
            if a is None:
                st.info(f"{title}: waiting for candle data.")
                continue

            if wr is not None and wr.bias in ("BULL", "BEAR"):
                confluent_dir = "LONG" if wr.bias == "BULL" else "SHORT"
                basis = f"11-Agent War Room ({wr.bias}, {wr.confidence:.0%} conf)"
            elif wr is not None and "TRAP" in wr.bias:
                confluent_dir = None
                basis = f"STAND ASIDE — {wr.bias}"
            elif v is not None and v.anchored.direction in ("BULLISH", "BEARISH"):
                confluent_dir = "LONG" if v.anchored.direction == "BULLISH" else "SHORT"
                basis = f"Anchored Desk ({v.anchored.direction}, {v.anchored.confidence:.0%} conf)"
            elif a.direction.direction in ("BULLISH", "BEARISH"):
                confluent_dir = "LONG" if a.direction.direction == "BULLISH" else "SHORT"
                basis = f"Desk Direction ({a.direction.direction}, {a.direction.confidence:.0%} conf)"
            else:
                confluent_dir = None
                basis = "BALANCED / NO CONFLUENCE — STAND ASIDE"

            st.markdown(f"**{title}**  \n_{subtitle}_  \n**Confluence:** `{basis}`")
            if confluent_dir is None:
                st.caption("No directional master trade is active on this tier until confluence resolves.")
            elif trade_map is not None and hasattr(trade_map, "build_trade_map") and spot_px > 0:
                try:
                    tm = trade_map.build_trade_map(
                        m=m,
                        analyses=analyses,
                        category=desk_key,
                        side=confluent_dir,
                        entry=spot_px,
                        stop=None,
                        leverage=5.0,
                    )
                    panels.render(panels.trade_result_html(tm))
                except Exception as exc:  # noqa: BLE001
                    st.caption(f"Master setup calculation pending: {exc}")

    st.markdown("---")
    st.subheader("Pin a Master Swing Trade for Live 24/7 Monitoring")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        horizon_key = st.selectbox(
            "Master Tier",
            ["intraday", "weekly"],
            format_func=lambda k: "Intraday / Swing (12h · 1d)" if k == "intraday" else "Weekly / Structural Swing (1d · 1w)",
        )
    with c2:
        side = st.selectbox("Direction", ["LONG", "SHORT"])
    with c3:
        entry = st.number_input("Entry ($) · defaults to spot", value=round(spot_px or 60000.0, 2), min_value=1.0, step=50.0)
    with c4:
        stop_in = st.number_input("Stop loss ($) · 0 lets desk place it", value=0.0, min_value=0.0, step=50.0)

    if st.button("Open and pin Master Trade", type="primary"):
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
                st.session_state["last_pinned_master_trade"] = tm
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not evaluate master trade: {exc}")

    if "last_pinned_master_trade" in st.session_state:
        panels.render(panels.trade_result_html(st.session_state["last_pinned_master_trade"]))

    open_trades = []
    if store is not None:
        if hasattr(store, "trades"):
            open_trades = store.trades(status="open", limit=MAX_TRADES)
        elif hasattr(store, "open_trades"):
            open_trades = store.open_trades()
    if not open_trades:
        st.caption("No master swing position is currently pinned in the ledger.")
    else:
        for t in open_trades[:MAX_TRADES]:
            panels.render(panels.trade_card_html(t, now_ms=now_ms, price_now=getattr(m.spot, "last", None)))


def _render_director(store, m, analyses: dict, now_ms: int) -> None:
    panels.render(
        panels.engine_hero_html(
            7,
            "Director Report",
            "15-agent multi-desk synthesis across technicals, order flow, derivatives, and macro.",
        )
    )
    report = st.session_state.get("latest_report")
    if report is None and store is not None:
        if hasattr(store, "reports"):
            reps = store.reports(limit=1)
            report = reps[0] if reps else None
        elif hasattr(store, "latest_report"):
            report = store.latest_report()

    if report is None:
        st.info(
            "Click **RUN ANALYSIS** in the top bar to generate the 15-agent Director report and all 4 desk war-room summaries."
        )
        if accuracy is not None and hasattr(accuracy, "summary") and store is not None:
            try:
                panels.render(panels.accuracy_html(accuracy.summary(store)))
            except Exception:  # noqa: BLE001
                pass
        return

    panels.render(panels.report_stats_html(report))
    panels.render(panels.director_html(report))
    panels.render(panels.director_lists_html(report))
    panels.render(panels.director_desks_html(report, CATEGORY_LABELS, {k: c.chart_tf for k, c in CATEGORIES.items()}))
    panels.render(panels.briefs_html(report))
    if accuracy is not None and hasattr(accuracy, "summary") and store is not None:
        try:
            panels.render(panels.accuracy_html(accuracy.summary(store)))
        except Exception:  # noqa: BLE001
            pass


def _render_macro(m, now_ms: int) -> None:
    panels.render(
        panels.engine_hero_html(
            8,
            "Macro & News",
            "High-impact USD releases, gold, US rates, dollar index, yen carry, and headlines.",
        )
    )
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
    panels.render(
        panels.engine_hero_html(
            9,
            "Quant Forecast",
            "24-hour quantitative plan, volatility windows, timeline, and live self-audit.",
        )
    )
    h1 = (getattr(getattr(m, "spot", None), "frames", None) or {}).get("1h")
    flow = (
        getattr(m.taker_flow, "frames", None).get("1h")
        if getattr(getattr(m, "taker_flow", None), "available", False)
        else None
    )
    price = float(getattr(m.spot, "last", None) or 0.0)

    try:
        if store is not None and hasattr(qf_alerts, "run"):
            try:
                qf_alerts.run(
                    store,
                    m,
                    now_ms,
                    desk_derivs.gamma_for("intraday", m),
                    weekly=analyses.get("weekly"),
                    weekly_gamma=desk_derivs.gamma_for("weekly", m),
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("qf_alerts.run failed: %s", exc)

        plan = None
        if store is not None:
            for attr in ("latest_qf_plan", "qf_plan", "get_qf_plan"):
                fn = getattr(store, attr, None)
                if callable(fn):
                    try:
                        plan = fn(quant_forecast.day_start(now_ms))
                    except TypeError:
                        plan = fn()
                    if plan:
                        break

        if not plan or not plan.get("available"):
            if h1 is None or h1.empty or price <= 0:
                st.warning("Waiting for 1h candle history to build today's Quant Forecast.")
                return
            start_ms = quant_forecast.day_start(now_ms)
            end_ms = start_ms + quant_forecast.DAY_MS
            plan = quant_forecast.build_plan(
                h1,
                now_ms,
                end_ms,
                price,
                flow=flow,
                gamma=desk_derivs.gamma_for("intraday", m),
            )
            if not plan.get("backtest"):
                plan["backtest"] = quant_forecast.backtest(h1, now_ms)

        if not plan.get("available"):
            st.warning(f"Quant Forecast unavailable: {plan.get('reason', 'insufficient history')}")
            return

        live_outcomes = []
        if store is not None and hasattr(store, "qf_outcomes"):
            try:
                live_outcomes = store.qf_outcomes(limit=30)
            except Exception:  # noqa: BLE001
                pass
        live_rec = quant_forecast.record(live_outcomes) if live_outcomes else None

        panels.render(panels.qf_wire_html(plan, status="ACTIVE", live=live_rec))
        if h1 is not None and len(h1):
            charts.render_qf_chart(plan, h1, dark=True, height=520, signal=True, now_ms=now_ms)
        panels.render(panels.qf_directive_html(plan))
        panels.render(panels.qf_schedule_html(plan, now_ms))
        panels.render(panels.qf_timeline_html(plan, now_ms, h1))
        panels.render(panels.qf_status_html("ACTIVE", [], [plan]))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Quant forecast error: {exc}")


def main() -> None:
    now_ms = _now_ms()
    store = _get_store()
    m, _errs = _snapshot(force=False)
    analyses = _compute_analyses(m)
    verdicts = _run_pipeline_pass(store, m, analyses, now_ms)

    dark, signal, choice, force_refresh = _render_sidebar(m, analyses, verdicts)
    if force_refresh:
        m, _errs = _snapshot(force=True)
        analyses = _compute_analyses(m)
        now_ms = _now_ms()
        verdicts = _run_pipeline_pass(store, m, analyses, now_ms)

    top_refresh = _render_topbar(m, now_ms)
    if top_refresh:
        m, _errs = _snapshot(force=True)
        analyses = _compute_analyses(m)
        now_ms = _now_ms()
        verdicts = _run_pipeline_pass(store, m, analyses, now_ms)
        st.rerun()

    if st.session_state.pop("run_analysis_requested", False):
        _execute_full_ai_run(store, m, analyses, now_ms)

    kind, key = NAV_BY_LABEL[choice]
    if kind == "desk":
        _render_desk(key, store, m, analyses, verdicts, now_ms, dark, signal)
    elif kind == "traps":
        _render_traps(store, m, analyses, now_ms)
    elif kind == "trade":
        _render_trade(store, m, analyses, verdicts, now_ms)
    elif kind == "director":
        _render_director(store, m, analyses, now_ms)
    elif kind == "macro":
        _render_macro(m, now_ms)
    elif kind == "qf":
        _render_qf(store, m, analyses, now_ms)


if __name__ == "__main__":
    main()

```
