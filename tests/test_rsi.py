import numpy as np
import pandas as pd

from core.indicators.rsi import detect_divergences, rsi


def _df(highs, lows=None):
    n = len(highs)
    highs = np.asarray(highs, dtype=float)
    lows = highs - 1.0 if lows is None else np.asarray(lows, dtype=float)
    close = (highs + lows) / 2
    return pd.DataFrame({"timestamp": np.arange(n) * 60_000, "open": close, "high": highs, "low": lows, "close": close, "volume": 1.0})


def test_rsi_extremes():
    up = rsi(pd.Series(np.linspace(100, 200, 60)))
    assert up.iloc[-1] > 95
    down = rsi(pd.Series(np.linspace(200, 100, 60)))
    assert down.iloc[-1] < 5
    assert np.isnan(up.iloc[5])  # warm-up


def test_regular_bearish_divergence_confirmed():
    highs = np.full(50, 100.0)
    highs[10] = 110.0   # first peak
    highs[30] = 115.0   # higher high
    r = pd.Series(np.full(50, 50.0)); r[10] = 80.0; r[30] = 70.0  # lower RSI high
    d = detect_divergences(_df(highs), r)
    assert any(x.kind == "regular" and x.direction == "bearish" and x.status == "confirmed" and x.bars == (10, 30) for x in d)


def test_regular_bullish_divergence_forming_on_last_bar():
    lows = np.full(40, 100.0)
    lows[10] = 90.0      # confirmed pivot low
    lows[39] = 85.0      # current bar making a lower low, not yet confirmed
    r = pd.Series(np.full(40, 50.0)); r[10] = 20.0; r[39] = 30.0  # RSI higher low
    d = detect_divergences(_df(lows + 1.0, lows), r)
    assert any(x.kind == "regular" and x.direction == "bullish" and x.status == "forming" for x in d)


def test_no_divergence_on_flat():
    d = detect_divergences(_df(np.full(50, 100.0)), pd.Series(np.full(50, 50.0)))
    assert d == []
