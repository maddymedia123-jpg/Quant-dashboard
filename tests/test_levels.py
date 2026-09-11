import numpy as np
import pandas as pd

from core.indicators.levels import count_reactions, support_resistance


def _zigzag(n=200, lo=100.0, hi=110.0, period=20):
    x = np.arange(n)
    tri = np.abs(((x % period) / (period / 2)) - 1.0)  # 1 -> 0 -> 1 triangle
    mid = lo + (hi - lo) * (1 - tri)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.05, n)
    close = mid + noise
    return pd.DataFrame({"timestamp": x * 60_000, "open": close, "high": close + 0.2, "low": close - 0.2, "close": close, "volume": 1.0})


def _flat_with_spikes(n, spikes: dict[int, float], base=100.0):
    """Flat market (high base+0.5, low base-0.5) with wicks reaching given highs at given bars."""
    close = np.full(n, base)
    high = close + 0.5
    low = close - 0.5
    for i, h in spikes.items():
        high[i] = h
    return pd.DataFrame({"timestamp": np.arange(n) * 60_000, "open": close, "high": high, "low": low, "close": close, "volume": 1.0})


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
    assert support_resistance(_zigzag(n=8), atr_value=1.0) == []


def test_touches_equal_visible_reactions_including_the_current_bar():
    # five confirmed wicks to ~105 plus one on the very last bar (cannot be a confirmed pivot yet)
    spikes = {20: 105.0, 40: 105.1, 60: 104.95, 80: 105.05, 100: 105.0, 119: 105.02}
    lv = support_resistance(_flat_with_spikes(120, spikes), atr_value=1.0)
    res = [l for l in lv if l.kind == "resistance"]
    assert len(res) == 1
    assert res[0].touches == 6
    assert abs(res[0].price - 105.01) < 0.05          # drawn where the wicks actually are
    assert res[0].last_touch_ms == 119 * 60_000
    assert len(res[0].reactions_ms) == 6


def test_group_width_is_capped_so_the_line_stays_on_its_reactions():
    # chain of wicks 0.6 apart: single-linkage would merge all four into one line at their mean
    spikes = {20: 104.0, 40: 104.6, 60: 105.2, 80: 105.8}
    lv = [l for l in support_resistance(_flat_with_spikes(110, spikes), atr_value=1.0) if l.kind == "resistance"]
    assert len(lv) == 2
    lo, hi = sorted(l.price for l in lv)
    assert abs(lo - 104.3) < 0.05 and abs(hi - 105.5) < 0.05
    assert all(l.touches == 2 for l in lv)


def test_consecutive_touching_bars_count_as_one_reaction():
    highs = np.array([100, 105, 105.1, 105, 100, 100, 100, 100, 100, 105, 100], dtype=float)
    lows = highs - 1.0
    starts, idx = count_reactions(highs, lows, level=105.0, tol=0.35)
    assert starts == [1, 9] and list(idx) == [1, 2, 3, 9]


def test_levels_use_the_whole_window_the_chart_draws():
    # the only wicks to 105 are 250 bars back: outside a 200-bar window, inside the 300 the chart shows
    spikes = {30: 105.0, 50: 105.0}
    df = _flat_with_spikes(300, spikes)
    assert [l for l in support_resistance(df, atr_value=1.0, lookback=200) if l.kind == "resistance"] == []
    assert [l.touches for l in support_resistance(df, atr_value=1.0, lookback=300) if l.kind == "resistance"] == [2]
