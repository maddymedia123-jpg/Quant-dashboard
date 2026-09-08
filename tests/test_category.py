import json
import pathlib

from core.config import CATEGORIES
from core.data.kraken_spot import parse_ohlc
from core.data.types import FuturesSnapshot
from core.indicators.category import analyze_category, price_change_24h

FX = pathlib.Path(__file__).parent / "fixtures"


def _frames():
    return {"15m": parse_ohlc(json.loads((FX / "kraken_ohlc_15m.json").read_text())),
            "1h": parse_ohlc(json.loads((FX / "kraken_ohlc_1h.json").read_text()))}


def test_live_category_on_fixtures():
    fut = FuturesSnapshot(source="binance", funding_rate=0.0001, funding_7d_mean=0.0001, oi_change_24h_pct=1.0, long_short_ratio=1.1)
    a = analyze_category(CATEGORIES["live"], _frames(), fut)
    assert a is not None and a.chart_tf == "15m" and a.context_tf == "1h"
    assert len(a.chart_df) <= 300 and len(a.rsi) == len(a.chart_df)
    assert a.direction.direction in {"BULLISH", "BEARISH", "NEUTRAL"}
    assert a.vol.regime in {"LOW", "NORMAL", "HIGH", "UNKNOWN"}
    assert 0 <= len(a.levels) <= 6
    assert a.fib is not None and a.squeeze.score is not None
    assert a.ctx_ema is not None


def test_missing_frame_returns_none():
    assert analyze_category(CATEGORIES["weekly"], _frames(), FuturesSnapshot.unavailable("x", "y")) is None


def test_price_change_24h_uses_1h_frame():
    pc = price_change_24h(_frames())
    assert pc is not None and -50 < pc < 50
