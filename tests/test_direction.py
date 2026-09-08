import numpy as np
import pandas as pd

from core.indicators.direction import direction_call
from core.indicators.ema import ema_stack
from core.indicators.levels import Level
from core.indicators.rsi import rsi


def _trend(n=260, up=True):
    x = np.linspace(100, 160, n) if up else np.linspace(160, 100, n)
    rng = np.random.default_rng(0)
    close = x + rng.normal(0, 0.3, n)
    return pd.DataFrame({"timestamp": np.arange(n) * 60_000, "open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 10.0})


def test_uptrend_is_bullish_with_anchor_on_support():
    df = _trend(up=True)
    price = float(df["close"].iloc[-1])
    levels = [Level(price - 2, 3, "support", 0), Level(price + 6, 2, "resistance", 0)]
    d = direction_call(df, ema_stack(df), rsi(df["close"]), levels, atr_value=1.0)
    assert d.direction == "BULLISH" and d.confidence > 0.3
    assert d.anchor == price - 2 and d.invalidation < d.anchor
    assert "below" in d.flips_if


def test_downtrend_is_bearish():
    df = _trend(up=False)
    d = direction_call(df, ema_stack(df), rsi(df["close"]), [], atr_value=1.0)
    assert d.direction == "BEARISH" and d.anchor is None


def test_flat_is_neutral():
    n = 260
    close = np.full(n, 100.0)  # perfectly flat: EMAs equal price, RSI neutral, CVD flat
    df = pd.DataFrame({"timestamp": np.arange(n) * 60_000, "open": close, "high": close + 0.2, "low": close - 0.2, "close": close, "volume": 10.0})
    d = direction_call(df, ema_stack(df), rsi(df["close"]), [], atr_value=0.3)
    assert d.direction == "NEUTRAL"
