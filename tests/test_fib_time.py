import numpy as np
import pandas as pd

from core.indicators.fib_time import fib_time_zones


def _df(n=100, low_at=50, high_at=70):
    close = np.full(n, 100.0)
    low = close.copy(); high = close.copy()
    low[low_at] = 80.0; high[high_at] = 120.0
    return pd.DataFrame({"timestamp": np.arange(n) * 60_000, "open": close, "high": high, "low": low, "close": close, "volume": 1.0})


def test_zones_from_anchor_with_future_extrapolation():
    ft = fib_time_zones(_df(), max_future=3)
    assert ft.anchor_kind == "low" and ft.anchor_ms == 50 * 60_000
    past = [z for z in ft.zones if not z.is_future]
    assert [z.k for z in past] == [1, 2, 3, 5, 8, 13, 21, 34]
    assert [z.k for z in ft.upcoming] == [55, 89, 144]
    assert ft.upcoming[0].timestamp_ms == (50 + 55) * 60_000  # extrapolated with bar interval


def test_anchor_is_earlier_extreme():
    ft = fib_time_zones(_df(low_at=70, high_at=40))
    assert ft.anchor_kind == "high" and ft.anchor_ms == 40 * 60_000


def test_too_short_returns_none():
    assert fib_time_zones(_df(n=3, low_at=1, high_at=2)) is None
