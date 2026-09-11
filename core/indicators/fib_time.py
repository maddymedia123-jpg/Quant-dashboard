"""Fibonacci time zones (TradingView two-point method) and Fibonacci price retracements.

Time zones: anchor 1 is the earlier of the major high and low in the lookback window; anchor 2 is the
end of the leg that follows it — the running extreme until price retraces 1.5 ATR from it — provided the
leg is at least one ATR long (otherwise the other major extreme).
The bars between them are the unit, and vertical zones sit at 0, 1, 2, 3, 5, 8, 13 ... units from
anchor 1 — the same geometry as drawing TradingView's Fib Time Zone tool on those two swings.

Retracements: the major leg between the window's high and low, with the standard ratios measured back
from the end of the leg."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.indicators.pivots import find_pivots

FIB = (0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377)
RETRACEMENTS = (0.236, 0.382, 0.5, 0.618, 0.786)


@dataclass
class FibZone:
    k: int
    timestamp_ms: int
    is_future: bool
    bars_from_now: int  # negative = already passed, 0 = current bar


@dataclass
class FibTime:
    anchor_ms: int
    anchor_kind: str      # low | high
    anchor2_ms: int
    anchor2_kind: str     # the opposite of anchor_kind
    unit_bars: int
    zones: list[FibZone]
    upcoming: list[FibZone]
    anchor_price: float = 0.0
    anchor2_price: float = 0.0


@dataclass
class FibRetracement:
    leg: str              # up (low then high) | down (high then low)
    high: float
    low: float
    high_ms: int
    low_ms: int
    levels: dict[float, float] = field(default_factory=dict)  # ratio -> price


def _leg_end(a_i: int, kind: str, highs: np.ndarray, lows: np.ndarray, atr: float, retrace_atr: float = 1.5) -> int | None:
    """Index of the end of the leg that starts at anchor 1: the running extreme after the anchor, fixed at
    the first bar that retraces from it by at least retrace_atr x ATR (or the extreme so far if the leg is
    still running). Pauses smaller than that inside the leg do not end it."""
    n = len(highs)
    if a_i >= n - 1:
        return None
    ext = a_i + 1
    for j in range(a_i + 1, n):
        if kind == "low":
            if highs[j] > highs[ext]:
                ext = j
            elif j > ext and highs[ext] - lows[j] >= retrace_atr * atr:
                return ext
        else:
            if lows[j] < lows[ext]:
                ext = j
            elif j > ext and highs[j] - lows[ext] >= retrace_atr * atr:
                return ext
    return ext


def fib_time_zones(df: pd.DataFrame, lookback: int = 100, max_future: int = 3, atr_value: float | None = None,
                   pivot_lr: int = 3, min_unit: int = 2) -> FibTime | None:
    d = df.tail(lookback).reset_index(drop=True)
    if len(d) < 10:
        return None
    ts = d["timestamp"].to_numpy(dtype="int64")
    highs = d["high"].to_numpy(dtype=float)
    lows = d["low"].to_numpy(dtype=float)
    interval = int(np.median(np.diff(ts)))
    last_i = len(d) - 1
    atr = atr_value if atr_value else float(np.median(highs - lows)) or 1e-9

    i_lo, i_hi = int(lows.argmin()), int(highs.argmax())
    if i_lo <= i_hi:
        a_i, kind, a_price, opp_kind, opp_arr, other_extreme = i_lo, "low", lows[i_lo], "high", highs, i_hi
    else:
        a_i, kind, a_price, opp_kind, opp_arr, other_extreme = i_hi, "high", highs[i_hi], "low", lows, i_lo

    b_i = _leg_end(a_i, kind, highs, lows, atr)
    if b_i is None or abs(opp_arr[b_i] - a_price) < atr:
        b_i = other_extreme
    unit = max(min_unit, b_i - a_i)

    zones: list[FibZone] = []
    future = 0
    for k in FIB:
        idx = a_i + k * unit
        if idx <= last_i:
            zones.append(FibZone(k, int(ts[idx]), False, idx - last_i))
        else:
            if future >= max_future:
                break
            zones.append(FibZone(k, int(ts[last_i] + (idx - last_i) * interval), True, idx - last_i))
            future += 1
    b_price = float(highs[b_i] if opp_kind == "high" else lows[b_i])
    return FibTime(anchor_ms=int(ts[a_i]), anchor_kind=kind, anchor2_ms=int(ts[min(b_i, last_i)]), anchor2_kind=opp_kind,
                   unit_bars=int(unit), zones=zones, upcoming=[z for z in zones if z.is_future],
                   anchor_price=float(a_price), anchor2_price=b_price)


def fib_retracements(df: pd.DataFrame, lookback: int = 100) -> FibRetracement | None:
    d = df.tail(lookback).reset_index(drop=True)
    if len(d) < 10:
        return None
    ts = d["timestamp"].to_numpy(dtype="int64")
    highs = d["high"].to_numpy(dtype=float)
    lows = d["low"].to_numpy(dtype=float)
    i_hi, i_lo = int(highs.argmax()), int(lows.argmin())
    hi, lo = float(highs[i_hi]), float(lows[i_lo])
    if hi <= lo:
        return None
    leg = "up" if i_lo < i_hi else "down"
    span = hi - lo
    levels = {r: (hi - r * span) if leg == "up" else (lo + r * span) for r in RETRACEMENTS}
    return FibRetracement(leg=leg, high=hi, low=lo, high_ms=int(ts[i_hi]), low_ms=int(ts[i_lo]), levels=levels)
