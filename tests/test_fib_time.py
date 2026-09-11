import numpy as np
import pandas as pd

from core.indicators.fib_time import fib_retracements, fib_time_zones


def _df(n=100, low_at=50, high_at=70):
    close = np.full(n, 100.0)
    low = close.copy(); high = close.copy()
    low[low_at] = 80.0; high[high_at] = 120.0
    return pd.DataFrame({"timestamp": np.arange(n) * 60_000, "open": close, "high": high, "low": low, "close": close, "volume": 1.0})


def test_two_point_zones_use_the_swing_gap_as_unit():
    ft = fib_time_zones(_df(), max_future=3)
    assert ft.anchor_kind == "low" and ft.anchor_ms == 50 * 60_000
    assert ft.anchor2_kind == "high" and ft.anchor2_ms == 70 * 60_000
    assert ft.unit_bars == 20
    past = [z.k for z in ft.zones if not z.is_future]
    assert past == [0, 1, 2]                      # bars 50, 70, 90
    assert [z.k for z in ft.upcoming] == [3, 5, 8]
    assert ft.upcoming[0].timestamp_ms == (50 + 3 * 20) * 60_000
    assert ft.upcoming[0].bars_from_now == 50 + 60 - 99
    assert [z.bars_from_now for z in ft.zones if not z.is_future] == [-49, -29, -9]


def test_anchor_is_earlier_extreme_and_second_anchor_is_opposite():
    ft = fib_time_zones(_df(low_at=70, high_at=40))
    assert ft.anchor_kind == "high" and ft.anchor_ms == 40 * 60_000
    assert ft.anchor2_kind == "low" and ft.unit_bars == 30


def test_second_anchor_skips_counter_swings_smaller_than_one_atr():
    n = 100
    close = np.full(n, 100.0)
    high = close + 0.2
    low = close - 0.2
    low[20] = 90.0            # anchor 1: the major low
    high[26] = 100.9          # small bounce: 10.9 above the low, under one ATR (15)
    high[45] = 110.0          # the real counter-swing
    df = pd.DataFrame({"timestamp": np.arange(n) * 60_000, "open": close, "high": high, "low": low, "close": close, "volume": 1.0})
    ft = fib_time_zones(df, atr_value=15.0)   # ATR 15: the 10.9 bounce is noise, the 20-point swing counts
    assert ft.anchor_kind == "low" and ft.anchor2_ms == 45 * 60_000 and ft.unit_bars == 25


def test_minimum_unit_prevents_one_bar_crowding():
    ft = fib_time_zones(_df(low_at=50, high_at=51), max_future=3)
    assert ft.unit_bars == 2
    ks = [z.k for z in ft.zones]
    gaps = np.diff([z.timestamp_ms for z in ft.zones]) // 60_000
    assert ks[:3] == [0, 1, 2] and gaps.min() >= 2


def test_too_short_returns_none():
    assert fib_time_zones(_df(n=3, low_at=1, high_at=2)) is None
    assert fib_retracements(_df(n=3, low_at=1, high_at=2)) is None


def test_retracements_measure_back_from_the_end_of_the_leg():
    up = fib_retracements(_df(low_at=30, high_at=70))          # low first → up leg, retrace down from 120
    assert up.leg == "up" and up.high == 120 and up.low == 80
    assert abs(up.levels[0.5] - 100.0) < 1e-9 and abs(up.levels[0.618] - (120 - 0.618 * 40)) < 1e-9
    down = fib_retracements(_df(low_at=70, high_at=30))        # high first → down leg, retrace up from 80
    assert down.leg == "down" and abs(down.levels[0.618] - (80 + 0.618 * 40)) < 1e-9


def test_second_anchor_is_the_end_of_the_leg_not_a_pause_inside_it():
    n = 80
    xs = [0, 20, 25, 27, 40, 55, 79]
    ys = [100, 90, 100, 98, 110, 95, 97]           # rally 90→110 pauses at 100 (dip to 98), ends at 110, then drops
    close = np.interp(np.arange(n), xs, ys)
    df = pd.DataFrame({"timestamp": np.arange(n) * 60_000, "open": close, "high": close + 0.1,
                       "low": close - 0.1, "close": close, "volume": 1.0})
    ft = fib_time_zones(df, atr_value=3.0)          # the 2-point dip is under 1.5 ATR, the 15-point drop is not
    assert ft.anchor_kind == "low" and ft.anchor_ms == 20 * 60_000
    assert ft.anchor2_ms == 40 * 60_000 and ft.unit_bars == 20 and abs(ft.anchor2_price - 110.1) < 1e-9


def test_a_single_wide_bar_does_not_end_the_leg_on_its_own_range():
    n = 60
    close = np.full(n, 100.0)
    high = close + 0.2
    low = close - 0.2
    low[10] = 90.0                        # anchor 1
    high[11], low[11] = 104.0, 96.0       # one wide bar: its own low sits 8 below its high
    high[12], low[12] = 106.0, 103.5
    high[13], low[13] = 108.0, 105.5      # the leg ends here
    for j in range(14, n):
        high[j], low[j] = 103.0, 101.0     # retrace of 5 > 1.5 ATR
    df = pd.DataFrame({"timestamp": np.arange(n) * 60_000, "open": close, "high": high, "low": low, "close": close, "volume": 1.0})
    ft = fib_time_zones(df, atr_value=2.0)
    assert ft.anchor2_ms == 13 * 60_000 and ft.unit_bars == 3
