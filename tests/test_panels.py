import html
import json
import pathlib

from core.config import CATEGORIES, MACRO_EVENTS
from core.data.kraken_spot import parse_ohlc
from core.data.types import FuturesSnapshot, LiquidationsSnapshot, MarketSnapshot, OptionsSnapshot, SentimentSnapshot, SpotSnapshot
from core.indicators.category import analyze_category
from core.indicators.trade_map import map_trade
from ui import panels

FX = pathlib.Path(__file__).parent / "fixtures"


def _market():
    frames = {"15m": parse_ohlc(json.loads((FX / "kraken_ohlc_15m.json").read_text())),
              "1h": parse_ohlc(json.loads((FX / "kraken_ohlc_1h.json").read_text()))}
    return MarketSnapshot(
        spot=SpotSnapshot(source="kraken", last=float(frames["1h"]["close"].iloc[-1]), frames=frames),
        futures=FuturesSnapshot(source="binance", funding_rate=0.0006, funding_7d_mean=0.0001, open_interest=100000, open_interest_usd=8e9,
                                oi_change_24h_pct=7.0, long_short_ratio=1.7, taker_buy_sell_ratio=0.9),
        options=OptionsSnapshot(source="deribit", max_pain=80000, max_pain_expiry="2026-09-25", put_call_oi_ratio=0.6, iv_atm=55.0, iv_skew=3.0),
        sentiment=SentimentSnapshot.unavailable("alternative.me", "timeout"),
        liquidations=LiquidationsSnapshot(source="coinlobster", total_usd=2.04e8, long_usd=1.54e8, short_usd=5.0e7),
    )


def test_kpis_cover_required_fields_with_dash_for_missing():
    m = _market()
    a = analyze_category(CATEGORIES["live"], m.spot.frames, m.futures)
    labels = [k["label"] for k in panels.kpis_for(a, m)]
    for req in ("Price", "Direction", "RSI 14", "ATR %", "Funding", "OI 24h", "Long/Short", "Liq 24h", "Fear & Greed"):
        assert req in labels
    liq = next(k for k in panels.kpis_for(a, m) if k["label"] == "Liq 24h")
    assert liq["value"] == "$204M" and liq["delta"] == "75% longs" and liq["tone"] == "down"
    fg = next(k for k in panels.kpis_for(a, m) if k["label"] == "Fear & Greed")
    assert fg["value"] == "—"


def test_panels_render_strings_and_squeeze_banner():
    m = _market()
    a = analyze_category(CATEGORIES["weekly"], {**m.spot.frames, "4h": m.spot.frames["1h"], "1d": m.spot.frames["1h"]}, m.futures)
    assert "ti-card" in panels.direction_html(a)
    lay = panels.layman_html(a)
    assert "BTC trades at" in lay and "2σ band" in lay and "flips if" in lay
    assert "2σ band (20-bar)" in panels.volatility_html(a)
    assert "RSI" in panels.divergence_html(a)
    assert a.vol.horizon_label in panels.volatility_html(a)
    assert "Run Analysis" in panels.agent_placeholder_html(a)
    banner = panels.squeeze_banner_html(a)
    assert banner and "SQUEEZE" in banner


def test_trade_result_macro_and_status():
    m = _market()
    live = analyze_category(CATEGORIES["live"], m.spot.frames, m.futures)
    tm = map_trade({"dir": "LONG", "entry": live.price, "lev": 10.0, "sl": live.price * 0.98, "tp": live.price * 1.04}, live, None, m.futures)
    h = panels.trade_result_html(tm)
    assert tm.classification in h and "Liquidation" in h
    assert html.escape(MACRO_EVENTS[0]["title"]) in panels.macro_html(MACRO_EVENTS)
    status = panels.data_status_html(m)
    assert "unavailable" in status and "kraken" in status
    rows = panels.raw_metrics_rows(m)
    assert any(r[0] == "Max pain" and "80,000" in r[1] for r in rows)
    assert any(r[0] == "Liquidations 24h" and "$204.0M" in r[1] for r in rows)
    assert "hyperliquid" in status and "stablecoins" in status


def test_agent_summary_and_report_stats():
    from datetime import datetime, timezone
    from core.agents.schemas import AgentRun, DirectorReport, Scenario, TradePlan, TrapReport
    m = _market()
    a = analyze_category(CATEGORIES["live"], m.spot.frames, m.futures)
    assert "Run Analysis" in panels.agent_summary_html(a, None)
    d = DirectorReport(executive_summary="e", trap_classification="BULL_TRAP", classification_rationale="r",
                       scenarios=[Scenario(name="a", probability=60, path="p", invalidation="i"), Scenario(name="b", probability=40, path="p", invalidation="i")],
                       trade_plan=TradePlan(existing_position_management="h", new_entry_conditions="n", position_sizing="s", stop_loss="l", max_leverage="3x"),
                       category_summaries={"live": "Live paragraph <b>"})
    r = TrapReport(report_id="TIR-1", generated_at=datetime.now(timezone.utc), window="w", provider="gemini", director=d,
                   runs=[AgentRun(agent_id="B1", ok=True, model="m", latency_s=1, prompt_tokens=10, completion_tokens=5),
                         AgentRun(agent_id="B2", ok=False, model="m", latency_s=1, error="x")])
    h = panels.agent_summary_html(a, r)
    assert "Live paragraph &lt;b&gt;" in h and "BULL TRAP" in h and "ti-card down" in h
    st = panels.report_stats_html(r)
    assert "agents 1/2 ok" in st and "tokens 15" in st and "failed: B2" in st and "free tier" in st


def test_anchored_direction_early_warning_and_accuracy_panels():
    from core.accuracy import HitRate
    from core.anchors import AnchorContext, AnchoredVerdict
    from core.indicators.early_warning import EarlyWarning
    m = _market()
    a = analyze_category(CATEGORIES["live"], m.spot.frames, m.futures)
    an = AnchorContext(direction="BULLISH", confidence=0.6, anchor=78000.0, invalidation=77800.0, price=78400.0, ts_ms=0,
                       funding_sign=1, oi_change_24h_pct=1.0, long_short_ratio=1.1, squeeze_score=20, classification=None,
                       macro_within_24h=False, sigma_pct=0.5, drivers=["EMA stack bullish"], flips_if="close below 77,800")
    live = AnchorContext(**{**an.__dict__, "direction": "BEARISH", "confidence": 0.3})
    v = AnchoredVerdict("live", an, live, since_ms=0, triggers_fired=[], changed=False)
    h = panels.anchored_direction_html(a, v, now_ms=90 * 60_000)
    assert "anchored 1h 30m ago" in h and "live read BEARISH" in h and "ti-card up" in h
    v2 = AnchoredVerdict("live", live, live, since_ms=0, triggers_fired=["funding_flip"], changed=True)
    assert "Re-anchored now on: funding_flip" in panels.anchored_direction_html(a, v2, now_ms=0)
    assert panels.early_warning_html([]) is None
    ew = panels.early_warning_html([EarlyWarning("BULL_TRAP_FORMING", "forming", ["pressing resistance"])])
    assert "BULL TRAP FORMING" in ew and "ti-card warn" in ew
    empty = panels.accuracy_html({"total_scored": 0, "open_calls": 3})
    assert "No scored calls yet" in empty and "waiting: 3" in empty
    full = panels.accuracy_html({"total_scored": 3, "open_calls": 0, "by_category": {"live": HitRate(3, 2)},
                                 "by_classification_24h": {"BULL_TRAP": HitRate(1, 1)}, "by_classification_7d": {},
                                 "recent_calls": [{"category": "live", "direction": "BULLISH", "price": 78000.0, "realised_pct": 0.4, "hit": 1}],
                                 "recent_reports": []})
    assert "67%" in full and "small sample" in full and "BULL_TRAP" in full and "hit)" in full
