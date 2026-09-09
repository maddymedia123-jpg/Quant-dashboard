"""Fibonacci time zones projected from the earliest major swing in the lookback window."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FIB = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377)


@dataclass
class FibZone:
    k: int
    timestamp_ms: int
    is_future: bool


@dataclass
class FibTime:
    anchor_ms: int
    anchor_kind: str  # low | high
    zones: list[FibZone]
    upcoming: list[FibZone]


def fib_time_zones(df: pd.DataFrame, lookback: int = 100, max_future: int = 3) -> FibTime | None:
    d = df.tail(lookback).reset_index(drop=True)
    if len(d) < 5:
        return None
    ts = d["timestamp"].to_numpy(dtype="int64")
    interval = int(np.median(np.diff(ts)))
    i_lo = int(d["low"].to_numpy().argmin())
    i_hi = int(d["high"].to_numpy().argmax())
    anchor_i, kind = (i_lo, "low") if i_lo <= i_hi else (i_hi, "high")
    last_i = len(d) - 1
    zones: list[FibZone] = []
    future = 0
    for k in FIB:
        idx = anchor_i + k
        if idx <= last_i:
            zones.append(FibZone(k, int(ts[idx]), False))
        else:
            if future >= max_future:
                break
            zones.append(FibZone(k, int(ts[last_i] + (idx - last_i) * interval), True))
            future += 1
    return FibTime(anchor_ms=int(ts[anchor_i]), anchor_kind=kind, zones=zones, upcoming=[z for z in zones if z.is_future])
