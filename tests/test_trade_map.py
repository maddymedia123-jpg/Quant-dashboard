import json
import pathlib

from core.config import CATEGORIES
from core.data.kraken_spot import parse_ohlc
from core.data.types import FuturesSnapshot
from core.indicators.category import analyze_category
from core.indicators.trade_map import map_trade

FX = pathlib.Path(__file__).parent / "fixtures"


def _ctx():
    frames = {"15m": parse_ohlc(json.loads((FX / "kraken_ohlc_15m.json").read_text())),
              "1h": parse_ohlc(json.loads((FX / "kraken_ohlc_1h.json").read_text()))}
    fut = FuturesSnapshot(source="binance", funding_rate=0.0002, funding_7d_mean=0.0001, oi_change_24h_pct=2.0, long_short_ratio=1.2)
    return analyze_category(CATEGORIES["live"], frames, fut), analyze_category(CATEGORIES["intraday"], frames, fut), fut


def test_long_trade_math_and_structures():
    live, intra, fut = _ctx()
    p = live.price
    tm = map_trade({"dir": "LONG", "entry": p, "lev": 10.0, "sl": p * 0.98, "tp": p * 1.04}, live, intra, fut)
    assert abs(tm.liquidation_price - p * (1 - 0.1 + 0.005)) < 1e-6
    assert abs(tm.rr - 2.0) < 1e-6
    assert tm.classification in {"SCALP", "SWING"}
    assert tm.risk_label in {"OPTIMAL", "ELEVATED", "EXTREME"}
    assert isinstance(tm.structures_above, list) and isinstance(tm.structures_below, list)
    assert any("Funding" in t for t in tm.anchor_triggers)


def test_short_liquidation_and_extreme_risk():
    live, intra, fut = _ctx()
    p = live.price
    tm = map_trade({"dir": "SHORT", "entry": p, "lev": 100.0, "sl": p * 1.02, "tp": p * 0.98}, live, intra, fut)
    assert tm.liquidation_price > p and tm.risk_label == "EXTREME"
    assert any("leverage" in w.lower() for w in tm.warnings)


def test_no_analysis_still_returns_math():
    fut = FuturesSnapshot.unavailable("binance,bybit", "451")
    tm = map_trade({"dir": "LONG", "entry": 100.0, "lev": 5.0, "sl": 95.0, "tp": 110.0}, None, None, fut)
    assert tm.rr == 2.0 and tm.structures_above == [] and any("unavailable" in t for t in tm.anchor_triggers)
