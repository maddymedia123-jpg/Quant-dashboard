import numpy as np
import pandas as pd

from core.indicators.patterns import detect_patterns


def _path(points: list[tuple[int, float]], wick: float = 0.2) -> pd.DataFrame:
    """Piecewise-linear close through (bar, price) points; highs/lows ±wick."""
    xs, ys = zip(*points)
    n = xs[-1] + 1
    close = np.interp(np.arange(n), xs, ys)
    return pd.DataFrame({"timestamp": np.arange(n) * 60_000, "open": close, "high": close + wick,
                         "low": close - wick, "close": close, "volume": 1.0})


def _names(ps):
    return [p.name for p in ps]


def test_double_top_forming_then_confirmed():
    forming = detect_patterns(_path([(0, 100), (40, 110), (55, 104), (70, 110), (80, 106)]), atr_value=1.0)
    p = next(p for p in forming if p.name == "Double top")
    assert p.status == "forming" and p.bias == "bearish"
    assert abs(p.breakout - 103.8) < 1e-6 and abs(p.target - (103.8 - 6.4)) < 1e-6
    assert p.invalidation > 110.2 and len(p.lines) == 2

    confirmed = detect_patterns(_path([(0, 100), (40, 110), (55, 104), (70, 110), (82, 103.5), (85, 101)]), atr_value=1.0)
    assert next(p for p in confirmed if p.name == "Double top").status == "confirmed"


def test_double_bottom_mirror():
    ps = detect_patterns(_path([(0, 110), (40, 100), (55, 106), (70, 100), (80, 104)]), atr_value=1.0)
    p = next(p for p in ps if p.name == "Double bottom")
    assert p.bias == "bullish" and p.status == "forming" and abs(p.breakout - 106.2) < 1e-6


def test_second_top_being_tested_now():
    ps = detect_patterns(_path([(0, 100), (40, 110), (55, 104), (68, 110.1)]), atr_value=1.0)
    p = next(p for p in ps if p.name == "Double top")
    assert p.status == "forming" and "tested now" in p.note


def test_stale_breakout_is_dropped():
    ps = detect_patterns(_path([(0, 100), (40, 110), (55, 104), (70, 110), (80, 103), (100, 96)]), atr_value=1.0)
    assert "Double top" not in _names(ps)


def test_head_and_shoulders_forming():
    ps = detect_patterns(_path([(0, 100), (20, 105), (30, 102), (40, 109), (50, 102), (60, 105), (70, 103.5)]), atr_value=1.0)
    p = next(p for p in ps if p.name == "Head and shoulders")
    assert p.bias == "bearish" and p.status == "forming"
    assert abs(p.breakout - 101.8) < 1e-6 and abs(p.target - (101.8 - 7.4)) < 1e-6 and p.invalidation == 109.2
    assert "Double top" not in _names(ps)


def test_inverse_head_and_shoulders():
    ps = detect_patterns(_path([(0, 110), (20, 105), (30, 108), (40, 101), (50, 108), (60, 105), (70, 106.5)]), atr_value=1.0)
    p = next(p for p in ps if p.name == "Inverse head and shoulders")
    assert p.bias == "bullish" and p.status == "forming"


def test_ascending_triangle_forming_with_apex_and_no_duplicate_double_top():
    df = _path([(0, 102), (10, 100), (20, 110), (30, 103), (40, 110), (50, 105), (60, 110), (70, 108.5)])
    ps = detect_patterns(df, atr_value=1.0)
    p = next(p for p in ps if p.name == "Ascending triangle")
    assert p.bias == "bullish" and p.status == "forming"
    assert abs(p.breakout - 110.2) < 0.05 and p.apex_bars is not None and 10 < p.apex_bars < 40
    assert p.target > p.breakout and p.invalidation < 108.5
    assert "Double top" not in _names(ps)


def test_falling_wedge_is_bullish():
    df = _path([(0, 111), (10, 110), (20, 100), (30, 106), (40, 98.5), (50, 103.5), (60, 97.8), (65, 99.5)])
    p = next(p for p in detect_patterns(df, atr_value=1.0) if p.name == "Falling wedge")
    assert p.bias == "bullish" and p.status == "forming"


def test_breakout_confirms_and_sets_bias():
    df = _path([(0, 102), (10, 100), (20, 110), (30, 103), (40, 110), (50, 105), (60, 110), (68, 108.5), (71, 111.5)])
    p = next(p for p in detect_patterns(df, atr_value=1.0) if p.name == "Ascending triangle")
    assert p.status == "confirmed" and p.bias == "bullish"


def test_flat_market_has_no_patterns_and_lines_are_time_ordered():
    flat = pd.DataFrame({"timestamp": np.arange(120) * 60_000, "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1.0})
    assert detect_patterns(flat, atr_value=1.0) == []
    ps = detect_patterns(_path([(0, 100), (20, 105), (30, 102), (40, 109), (50, 102), (60, 105), (70, 103.5)]), atr_value=1.0)
    for p in ps:
        for line in p.lines:
            times = [t for t, _ in line]
            assert times == sorted(set(times))


def test_double_top_and_double_bottom_together_are_one_range():
    # tops at 110 (bars 20, 50) and bottoms at 100 (bars 35, 65), price now back mid-range
    df = _path([(0, 105), (20, 110), (35, 100), (50, 110), (65, 100), (72, 104)])
    ps = detect_patterns(df, atr_value=1.0)
    names = _names(ps)
    assert "Double top" not in names and "Double bottom" not in names
    r = next(p for p in ps if p.name.startswith("Range"))
    assert r.bias == "neutral" and r.status == "forming"
    assert abs(r.breakout - 110.2) < 0.3 and abs(r.invalidation - 99.8) < 0.3
