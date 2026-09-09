import numpy as np
import pandas as pd

from core.indicators.trendlines import fit_trendlines


def _rising_channel(n=120):
    x = np.arange(n, dtype=float)
    base = 100 + 0.5 * x                      # support line: 100 + 0.5x
    wave = 5 * np.abs(np.sin(x / 6.0))        # bounces off the line every ~19 bars
    low = base + wave
    high = low + 4.0
    close = (low + high) / 2
    return pd.DataFrame({"timestamp": (x * 60_000).astype("int64"), "open": close, "high": high, "low": low, "close": close, "volume": 1.0})


def test_support_trendline_matches_generating_line():
    tl = [t for t in fit_trendlines(_rising_channel()) if t.kind == "support"]
    assert tl, "expected a support trendline"
    t = tl[0]
    assert abs(t.slope - 0.5) < 0.05
    assert t.r2 > 0.95
    assert len(t.points) == 2 and t.points[1][0] > t.points[0][0]
    assert t.value_now > t.points[0][1]


def test_lines_below_r2_gate_are_dropped():
    rng = np.random.default_rng(1)
    n = 120
    close = np.full(n, 100.0)
    low = close - rng.uniform(0, 5, n)
    high = close + rng.uniform(0, 5, n)
    df = pd.DataFrame({"timestamp": np.arange(n) * 60_000, "open": close, "high": high, "low": low, "close": close, "volume": 1.0})
    lines = fit_trendlines(df, min_r2=0.99)
    assert all(t.r2 >= 0.99 for t in lines)
