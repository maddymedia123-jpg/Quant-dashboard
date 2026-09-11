import json
import pathlib

import pytest

from core.config import CATEGORIES
from core.data.kraken_spot import parse_ohlc
from core.data.types import FuturesSnapshot
from core.indicators.category import analyze_category
from core.indicators.fib_time import FibTime, FibZone
from core.indicators.levels import Level
from core.indicators.patterns import Pattern
from core.indicators.reversal import reversal_window

FX = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture()
def a():
    frames = {"15m": parse_ohlc(json.loads((FX / "kraken_ohlc_15m.json").read_text())),
              "1h": parse_ohlc(json.loads((FX / "kraken_ohlc_1h.json").read_text()))}
    x = analyze_category(CATEGORIES["live"], frames, FuturesSnapshot(source="x"))
    x.levels, x.divergences, x.trendlines, x.patterns, x.fib_retr = [], [], [], [], None
    x.rsi = x.rsi * 0 + 50.0
    return x


def _fib(bars_now: list[int], unit=6) -> FibTime:
    zones = [FibZone(k, 1_000_000 + b * 900_000, b > 0, b) for k, b in zip((3, 5, 8), bars_now)]
    return FibTime(0, "low", 1, "high", unit, zones, [z for z in zones if z.is_future])


def test_no_zone_nearby_is_upcoming(a):
    a.fib = _fib([-20, 10, 40])
    rw = reversal_window(a)
    assert rw.status == "UPCOMING" and rw.zone_k == 5 and rw.bars_away == 10 and rw.window_bars == 2


def test_zone_now_without_confluence_is_time_only(a):
    a.fib = _fib([-20, 1, 40])
    rw = reversal_window(a)
    assert rw.status == "TIME_ONLY" and rw.confluence == []


def test_zone_now_at_resistance_with_bearish_pattern_is_active_bearish(a):
    a.fib = _fib([-20, 0, 40])
    atr = a.vol.atr
    a.levels = [Level(a.price + 0.2 * atr, 4, "resistance", 0)]
    a.patterns = [Pattern("Double top", "bearish", "forming", 0.7, None, None, None, [], 0)]
    rw = reversal_window(a)
    assert rw.status == "ACTIVE" and rw.bias == "bearish"
    assert any("resistance" in c for c in rw.confluence) and any("Double top" in c for c in rw.confluence)


def test_oversold_at_support_is_active_bullish(a):
    a.fib = _fib([-20, -1, 40])
    a.levels = [Level(a.price - 0.1 * a.vol.atr, 3, "support", 0)]
    a.rsi = a.rsi * 0 + 25.0
    rw = reversal_window(a)
    assert rw.status == "ACTIVE" and rw.bias == "bullish" and rw.bars_away == -1


def test_analyze_category_attaches_reversal_patterns_and_retracements():
    frames = {"15m": parse_ohlc(json.loads((FX / "kraken_ohlc_15m.json").read_text())),
              "1h": parse_ohlc(json.loads((FX / "kraken_ohlc_1h.json").read_text()))}
    x = analyze_category(CATEGORIES["intraday"], frames, FuturesSnapshot(source="x"))
    assert x.reversal is not None and x.reversal.status in {"ACTIVE", "TIME_ONLY", "UPCOMING", "NONE"}
    assert x.fib_retr is not None and set(x.fib_retr.levels) >= {0.382, 0.5, 0.618}
    assert isinstance(x.patterns, list) and x.fib.anchor_price > 0 and x.fib.anchor2_price > 0
