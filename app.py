"""Streamlit entrypoint for TRAP INTELLIGENCE."""
from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
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

from core.indicators import category as cat_ind  # noqa: E402
from core.indicators import desk_derivs  # noqa: E402
from core.trades import fmt_when  # noqa: E402
from ui import charts, panels, theme  # noqa: E402


def _opt_import(module_path: str):
    try:
        return importlib.import_module(module_path)
    except Exception:  # noqa: BLE001
        return None


_cfg = _opt_import("core.config")
_DEFAULT_CATEGORIES = ("live", "intraday", "weekly", "monthly")
_DEFAULT_LABELS = {
    "live": "Live Recon",
    "intraday": "Intraday",
    "weekly": "Weekly",
    "monthly": "Monthly",
}
_DEFAULT_TIMEFRAMES = {
    "live": ("1h", "4h"),
    "intraday": ("4h", "1d"),
    "weekly": ("1d", "1w"),
    "monthly": ("1w", "1M"),
}

CATEGORIES = tuple(getattr(_cfg, "CATEGORIES", None) or _DEFAULT_CATEGORIES)
_raw_labels = getattr(_cfg, "CATEGORY_LABELS", None)
CATEGORY_LABELS = dict(_raw_labels) if isinstance(_raw_labels, dict) and _raw_labels else dict(_DEFAULT_LABELS)
_raw_tfs = getattr(_cfg, "CATEGORY_TIMEFRAMES", None)
CATEGORY_TIMEFRAMES = dict(_raw_tfs) if isinstance(_raw_tfs, dict) and _raw_tfs else dict(_DEFAULT_TIMEFRAMES)

for _k in CATEGORIES:
    CATEGORY_LABELS.setdefault(_k, _k.replace("_", " ").title())
    CATEGORY_TIMEFRAMES.setdefault(_k, ("1h", "4h"))

core_data = _opt_import("core.data")
core_pipeline = _opt_import("core.pipeline")
accuracy = _opt_import("core.accuracy")
qf_alerts = _opt_import("core.qf_alerts")
quant_forecast = _opt_import("core.quant_forecast")
_raw_store = _opt_import("core.store")
traps = _opt_import("core.traps")
recon_profiles = _opt_import("core.agents.recon_profiles")
runner = _opt_import("core.agents.runner")
agents_types = _opt_import("core.agents.types")
LLMConfig = getattr(agents_types, "LLMConfig", None) if agents_types else None
trade_map = _opt_import("core.indicators.trade_map")


class _StoreProxy:
    """Forwards to core.store while supplying missing ledger/table attributes safely."""

    def __init__(self, inner):
        self._inner = inner
        self.calls: list = []

    def __getattr__(self, name: str):
        if self._inner is not None and hasattr(self._inner, name):
            return getattr(self._inner, name)
        if name in ("calls", "all_calls", "recent_calls", "open_calls"):
            return []
        if name.startswith(("list_", "get_", "load_", "fetch_", "all_")):
            return lambda *_a, **_kw: []
        if name.startswith(("save_", "record_", "insert_", "update_", "delete_", "close_", "prune_", "sync_")):
            return lambda *_a, **_kw: None
        raise AttributeError(f"'_StoreProxy' object has no attribute '{name}'")


store = _StoreProxy(_raw_store) if _raw_store is not None else None

log = logging.getLogger("trap_intelligence")

st.set_page_config(page_title="TRAP INTELLIGENCE", page_icon="◆", layout="wide",
                   initial_sidebar_state="expanded")

MAX_TRADES = 20
NAV = (
    *[("desk", k, CATEGORY_LABELS[k], f"{CATEGORY_TIMEFRAMES[k][0]} chart · {CATEGORY_TIMEFRAMES[k][1]} ctx")
      for k in CATEGORIES],
    ("traps", "traps", "TRAP intelligence", "Sweep, anchored traps, watch list"),
    ("trade", "trade", "Active trade", "Positions, stops and targets"),
    ("director", "director", "Director report", "Run Analysis to generate"),
    ("macro", "macro", "Macro & news", "Headlines, gold, rates, odds"),
    ("qf", "qf", "Quant forecast", "Today's 24h plan, audited live"),
)
NAV_LABELS = [label for _kind, _key, label, _sub in NAV]
NAV_CAPTIONS = [sub for _kind, _key, _label, sub in NAV]
NAV_BY_LABEL = {label: (kind, key) for (kind, key, label, _sub) in NAV}
DESK_INDEX = {key: i + 1 for i, key in enumerate(CATEGORIES)}

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


def _candidate_modules():
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
    if core_pipeline is not None:
        mods.append(core_pipeline)
    return mods


def _resolve_snapshot_loader():
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
    mods = _candidate_modules()
    for mod in mods:
        for name in preferred:
            fn = getattr(mod, name, None)
            if callable(fn):
                return fn
    for mod in mods:
        for name, obj in inspect.getmembers(mod, callable):
            if not name.startswith("_") and name.startswith(("fetch", "load", "get", "collect", "build", "gather")):
                if "snapshot" in name or "market" in name or "all" in name:
                    return obj
    return None


_SNAPSHOT_LOADER = _resolve_snapshot_loader()


def _fallback_snapshot(bucket_ms: int):
    snap_cls = getattr(core_data, "MarketSnapshot", None) if core_data else None
    if callable(snap_cls):
        try:
            obj = snap_cls()
            if getattr(obj, "generated_at", None) is None:
                obj.generated_at = datetime.fromtimestamp(bucket_ms / 1000.0, tz=timezone.utc)
            return obj, ["Using fallback snapshot: no snapshot loader discovered in core.data"]
        except Exception:  # noqa: BLE001
            pass

    def _feed(source="unavailable", **extra):
        base = {"available": False, "source": source, "error": "no loader in core.data"}
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
    ), ["Using fallback snapshot: no snapshot loader discovered in core.data"]


def _invoke_loader_smart(fn, bucket_ms: int):
    sig = inspect.signature(fn)
    params = sig.parameters
    kwargs = {}

    if "now_ms" in params:
        kwargs["now_ms"] = bucket_ms
    elif "ts_ms" in params:
        kwargs["ts_ms"] = bucket_ms
    elif "timestamp_ms" in params:
        kwargs["timestamp_ms"] = bucket_ms

    for p in params.values():
        if p.name in kwargs:
            continue
        if p.default is not inspect.Parameter.empty:
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        pname = p.name.lower()
        ann = str(p.annotation).lower() if p.annotation is not inspect.Parameter.empty else ""
        if "dict" in ann or "mapping" in ann or any(k in pname for k in ("cfg", "config", "settings", "opts", "options", "params", "secrets", "env")):
            kwargs[p.name] = {}
        elif "int" in ann or "ms" in pname or "time" in pname:
            kwargs[p.name] = bucket_ms
        elif "str" in ann or "symbol" in pname:
            kwargs[p.name] = "BTC/USDT"
        elif "bool" in ann:
            kwargs[p.name] = False

    return fn(**kwargs)


def _call_snapshot_loader(bucket_ms: int):
    if _SNAPSHOT_LOADER is None:
        return _fallback_snapshot(bucket_ms)
    try:
        res = _invoke_loader_smart(_SNAPSHOT_LOADER, bucket_ms)
    except Exception:  # noqa: BLE001
        try:
            res = _SNAPSHOT_LOADER()
        except Exception:  # noqa: BLE001
            res = _SNAPSHOT_LOADER(now_ms=bucket_ms)

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


def _load_snapshot(bucket_ms: int):
    return _load_snapshot_cached(bucket_ms)


_load_snapshot.clear = _load_snapshot_cached.clear  # type: ignore[attr-defined]


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
    return _load_snapshot(bucket)


def _resolve_category_analyzer():
    for name in (
        "analyze",
        "analyze_category",
        "build_category",
        "run_category",
        "compute_category",
        "evaluate_category",
        "compute",
        "evaluate",
        "run",
        "build",
    ):
        fn = getattr(cat_ind, name, None)
        if callable(fn):
            return fn
    for name, obj in inspect.getmembers(cat_ind, callable):
        if name.startswith("_"):
            continue
        if getattr(obj, "__module__", "") != getattr(cat_ind, "__name__", ""):
            continue
        try:
            params = [
                p for p in inspect.signature(obj).parameters.values()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            if len(params) >= 2:
                return obj
        except (TypeError, ValueError):
            continue
    return None


_CAT_ANALYZER = _resolve_category_analyzer()


def _map_param_value(p: inspect.Parameter, key: str, m, now_ms: int):
    pname = p.name.lower()
    ann = str(p.annotation).lower() if p.annotation is not inspect.Parameter.empty else ""
    frames = getattr(getattr(m, "spot", None), "frames", None) or {}
    tfs = CATEGORY_TIMEFRAMES.get(key, ("1h", "4h"))

    if pname in ("category", "cat", "key", "desk", "horizon"):
        return key
    if pname in ("m", "snap", "snapshot", "market", "data") or "snapshot" in ann:
        return m
    if pname in ("frames", "candles", "ohlc", "ohlcv"):
        return frames
    if pname in ("primary_tf", "tf", "timeframe", "entry_tf"):
        return tfs[0]
    if pname in ("context_tf", "ctx_tf", "higher_tf", "htf"):
        return tfs[1]
    if pname in ("tfs", "timeframes"):
        return tfs
    if pname in ("now_ms", "ts_ms", "timestamp_ms", "now") or "int" in ann:
        return now_ms
    if pname in ("store", "db"):
        return store
    if pname in ("cfg", "config", "settings", "opts", "options", "params") or "dict" in ann:
        return {}
    return None


def _invoke_category_analyzer(fn, key: str, m, now_ms: int):
    sig = inspect.signature(fn)
    pos_args = []
    kw_args = {}

    for p in sig.parameters.values():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        val = _map_param_value(p, key, m, now_ms)
        if val is None and p.default is not inspect.Parameter.empty:
            continue
        if p.kind == inspect.Parameter.KEYWORD_ONLY:
            kw_args[p.name] = val
        else:
            pos_args.append(val)

    try:
        res = fn(*pos_args, **kw_args)
        if res is not None:
            return res
    except Exception:  # noqa: BLE001
        pass

    for call in (
        lambda: fn(key, m, now_ms),
        lambda: fn(m, key, now_ms),
        lambda: fn(key, m),
        lambda: fn(m, key),
    ):
        try:
            res = call()
            if res is not None:
                return res
        except Exception:  # noqa: BLE001
            continue
    return None


def _compute_analyses(m, now_ms: int) -> dict:
    out = {}
    if _CAT_ANALYZER is None:
        log.warning("No analyzer callable found in core.indicators.category")
        return {k: None for k in CATEGORIES}
    for key in CATEGORIES:
        try:
            out[key] = _invoke_category_analyzer(_CAT_ANALYZER, key, m, now_ms)
        except Exception as exc:  # noqa: BLE001
            log.warning("analysis failed for %s: %s", key, exc)
            out[key] = None
    return out


def _sync_accuracy(m, analyses: dict, now_ms: int) -> dict:
    empty = {"total_scored": 0, "open_calls": 0, "by_category": {},
             "by_classification_24h": {}, "by_classification_7d": {}, "recent_calls": []}
    if accuracy is None:
        return empty
    try:
        if hasattr(accuracy, "record_calls"):
            try:
                accuracy.record_calls(m, analyses, now_ms, store)
            except TypeError:
                accuracy.record_calls(m, analyses, now_ms)
        if hasattr(accuracy, "score_due"):
            try:
                accuracy.score_due(m, now_ms, store)
            except TypeError:
                accuracy.score_due(m, now_ms)
        if hasattr(accuracy, "summary"):
            try:
                return accuracy.summary(store)
            except TypeError:
                return accuracy.summary()
    except Exception as exc:  # noqa: BLE001
        log.warning("accuracy ledger failed: %s", exc)
    return empty


def _llm_config_from_env():
    if LLMConfig is None:
        return None
    import os
    for provider, env_key, default_model in (
        ("openrouter", "OPENROUTER_API_KEY", "deepseek/deepseek-chat-v3-0324:free"),
        ("groq", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
        ("gemini", "GEMINI_API_KEY", "gemini-2.5-flash"),
    ):
        key = (os.environ.get(env_key) or "").strip()
        if key:
            return LLMConfig(provider=provider, api_key=key, model=default_model)
    return None


def _render_sidebar(m, analyses: dict, now_ms: int) -> tuple[bool, bool, str, bool]:
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
    for key in CATEGORIES:
        a = analyses.get(key)
        tones.append(panels.rail_tone(None, a))
        tfs.append(CATEGORY_TIMEFRAMES[key][0])
    extra = len(NAV) - len(CATEGORIES)
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
    a = analyses.get(key)
    if a is None:
        st.warning(f"No analysis available for {CATEGORY_LABELS[key]}.")
        return
    for_cat = getattr(recon_profiles, "FOR_CATEGORY", None) or {}
    profile = for_cat.get(key) or getattr(recon_profiles, "LIVE", None)
    if profile is not None:
        panels.render(panels.desk_hero_html(DESK_INDEX[key], profile, a, now_ms))
    gamma = desk_derivs.gamma_for(key, m)
    panels.render(panels.gamma_bar_html(gamma, getattr(m.spot, "last", None), CATEGORY_LABELS[key]))

    fg = getattr(m.sentiment, "fear_greed", None)
    derivs = desk_derivs.read(key, m, now_ms)

    left, right = st.columns([7, 5], gap="large")
    with left:
        panels.render(panels.chart_head_html(a, signal=signal, symbol=getattr(m.spot, "symbol", None)))
        st.plotly_chart(charts.price_chart(a, dark=dark, signal=signal), use_container_width=True, config={"displayModeBar": False})
        panels.render(panels.layman_html(a))
        panels.render(panels.patterns_html(a))
        panels.render(panels.divergence_html(a))
    with right:
        panels.render(panels.direction_html(a))
        panels.render(panels.volatility_html(a))
        panels.render(panels.fib_html(a))

    st.markdown("---")
    panels.render(panels.layman_summary_html(a, CATEGORY_LABELS[key], derivs, fg, now_ms))


def _render_traps(m, analyses: dict, now_ms: int) -> None:
    panels.render(panels.engine_hero_html(
        5, "TRAP Intelligence",
        "Sweep timeframes for engineered bull/bear traps, monitor anchored traps, and review settled history."
    ))
    gamma = desk_derivs.gamma_for(None, m)
    panels.render(panels.gamma_bar_html(gamma, getattr(m.spot, "last", None)))

    if traps is None:
        st.info("TRAP intelligence module not present in this build.")
        return

    rows = traps.all_traps() if hasattr(traps, "all_traps") else []
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
            rows = traps.all_traps() if hasattr(traps, "all_traps") else rows
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


def _render_trade(m, analyses: dict, now_ms: int) -> None:
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
        asset = st.text_input("Asset", value="BTC/USDT")
    with c2:
        side = st.selectbox("Direction", ["LONG", "SHORT"])
    with c3:
        entry = st.number_input("Entry ($) · defaults to spot", value=round(spot_px, 2), min_value=1.0, step=50.0)
    with c4:
        stop_in = st.number_input("Stop loss ($) · 0 lets the desk place it", value=0.0, min_value=0.0, step=50.0)

    horizon_key = st.radio(
        "Judge it over",
        ["live", "intraday", "weekly", "monthly"],
        format_func=lambda k: {
            "live": "Live Recon · four hours",
            "intraday": "Intraday · 24 hours",
            "weekly": "Weekly · seven days",
            "monthly": "Monthly · 30 days",
        }[k],
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

    open_trades = store.open_trades() if (store is not None and hasattr(store, "open_trades")) else []
    if not open_trades:
        st.caption("No position is being tracked. Open one above and it stays pinned here.")
    else:
        for t in open_trades[:MAX_TRADES]:
            panels.render(panels.trade_card_html(t, now_ms=now_ms, price_now=getattr(m.spot, "last", None)))


def _render_director(m, analyses: dict, now_ms: int) -> None:
    panels.render(panels.engine_hero_html(
        7, "Director Report",
        "Multi-agent synthesis across technicals, order flow, derivatives, and macro."
    ))
    latest_fn = getattr(store, "latest_report", None) if store is not None else None
    report = st.session_state.get("latest_report") or (latest_fn() if callable(latest_fn) else None)
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
    panels.render(panels.director_desks_html(report, CATEGORY_LABELS, {k: CATEGORY_TIMEFRAMES[k][0] for k in CATEGORIES}))
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


def _render_qf(m, analyses: dict, now_ms: int) -> None:
    panels.render(panels.engine_hero_html(
        9, "Quant Forecast",
        "24-hour quantitative plan, volatility windows, timeline, and live self-audit."
    ))
    if quant_forecast is None or qf_alerts is None:
        st.info("Quant forecast module not present in this build.")
        return
    try:
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
    m, _errs = _snapshot(force=False)
    analyses = _compute_analyses(m, now_ms)
    _sync_accuracy(m, analyses, now_ms)

    dark, signal, choice, force_refresh = _render_sidebar(m, analyses, now_ms)
    if force_refresh:
        m, _errs = _snapshot(force=True)
        analyses = _compute_analyses(m, now_ms)
        now_ms = _now_ms()

    top_refresh = _render_topbar(m, now_ms)
    if top_refresh:
        m, _errs = _snapshot(force=True)
        analyses = _compute_analyses(m, now_ms)
        now_ms = _now_ms()
        st.rerun()

    kind, key = NAV_BY_LABEL[choice]
    if kind == "desk":
        _render_desk(key, m, analyses, now_ms, dark, signal)
    elif kind == "traps":
        _render_traps(m, analyses, now_ms)
    elif kind == "trade":
        _render_trade(m, analyses, now_ms)
    elif kind == "director":
        _render_director(m, analyses, now_ms)
    elif kind == "macro":
        _render_macro(m, now_ms)
    elif kind == "qf":
        _render_qf(m, analyses, now_ms)


if __name__ == "__main__":
    main()
