"""Swing pivot detection shared by RSI divergence, S/R, trendlines, fib time."""
from __future__ import annotations

from typing import Literal

import numpy as np


def find_pivots(values: np.ndarray, left: int, right: int, kind: Literal["high", "low"]) -> list[int]:
    out: list[int] = []
    n = len(values)
    for i in range(left, n - right):
        v = values[i]
        wl = values[i - left:i]
        wr = values[i + 1:i + 1 + right]
        if kind == "high":
            if v > wl.max() and v >= wr.max():
                out.append(i)
        else:
            if v < wl.min() and v <= wr.min():
                out.append(i)
    return out
