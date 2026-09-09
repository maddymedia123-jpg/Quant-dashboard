"""Least-squares trendlines through the last N swing lows (support) / highs (resistance)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.indicators.pivots import find_pivots


@dataclass
class Trendline:
    kind: str                        # support | resistance
    slope: float                     # price units per bar
    r2: float
    points: list[tuple[int, float]]  # (ms, price) start and projected end
    value_now: float


def fit_trendlines(df: pd.DataFrame, left: int = 5, right: int = 5, lookback: int = 200, min_r2: float = 0.8,
                   n_points: int = 3, extend_bars: int = 10) -> list[Trendline]:
    d = df.tail(lookback).reset_index(drop=True)
    if len(d) < left + right + 2:
        return []
    ts = d["timestamp"].to_numpy(dtype="int64")
    interval = int(np.median(np.diff(ts))) if len(ts) > 1 else 60_000
    last = len(d) - 1
    out: list[Trendline] = []
    for kind, arr, pk in (("support", d["low"].to_numpy(dtype=float), "low"), ("resistance", d["high"].to_numpy(dtype=float), "high")):
        piv = find_pivots(arr, left, right, pk)[-n_points:]
        if len(piv) < n_points:
            continue
        x = np.array(piv, dtype=float)
        y = arr[piv]
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        if r2 < min_r2:
            continue
        x0, x1 = piv[0], last + extend_bars
        out.append(Trendline(
            kind=kind, slope=float(slope), r2=float(r2),
            points=[(int(ts[x0]), float(slope * x0 + intercept)), (int(ts[last] + interval * extend_bars), float(slope * x1 + intercept))],
            value_now=float(slope * last + intercept),
        ))
    return out
