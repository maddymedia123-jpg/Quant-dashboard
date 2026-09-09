import numpy as np
import pandas as pd

from core.indicators.levels import support_resistance


def _zigzag(n=200, lo=100.0, hi=110.0, period=20):
    x = np.arange(n)
    tri = np.abs(((x % period) / (period / 2)) - 1.0)  # 1 -> 0 -> 1 triangle
    mid = lo + (hi - lo) * (1 - tri)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.05, n)
    close = mid + noise
    return pd.DataFrame({"timestamp": x * 60_000, "open": close, "high": close + 0.2, "low": close - 0.2, "close": close, "volume": 1.0})


def test_zigzag_yields_two_strong_levels():
    df = _zigzag()
    levels = support_resistance(df, atr_value=1.0)
    prices = [l.price for l in levels]
    assert any(abs(p - 110.2) < 1.0 for p in prices)
    assert any(abs(p - 99.8) < 1.0 for p in prices)
    strong = [l for l in levels if l.touches >= 3]
    assert len(strong) >= 2
    assert levels == sorted(levels, key=lambda l: l.price, reverse=True)
    assert all(l.kind in ("support", "resistance") for l in levels)
    assert len(levels) <= 6


def test_short_history_returns_empty():
    df = _zigzag(n=8)
    assert support_resistance(df, atr_value=1.0) == []
