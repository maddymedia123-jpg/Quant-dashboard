import numpy as np

from core.indicators.pivots import find_pivots


def test_find_pivot_highs_and_lows():
    v = np.array([1, 2, 3, 2, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2], dtype=float)
    assert find_pivots(v, 2, 2, "high") == [2, 7]
    assert find_pivots(v, 2, 2, "low") == [4, 11]


def test_pivot_requires_full_window():
    v = np.array([5, 4, 3, 2, 1], dtype=float)
    assert find_pivots(v, 2, 2, "low") == []
