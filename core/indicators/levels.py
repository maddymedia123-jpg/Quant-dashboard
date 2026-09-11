"""Support/resistance levels.

Candidate prices come from confirmed swing pivots. A level's strength is the number of distinct
reactions a trader would count on the chart: bars whose high or low reaches within the tolerance band
of the line, with consecutive touching bars grouped into one reaction. Counting runs over the same bars
the chart draws and includes the most recent bars, which can never be confirmed pivots yet.

Pivots are grouped with a capped width (every member within 2 x tolerance of the group's lowest price),
and the line is drawn at the median of the wicks that actually touched it, so it sits on the reactions
rather than between them."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.config import CHART_BARS
from core.indicators.pivots import find_pivots

REACTION_GAP = 3  # touching bars closer than this belong to the same reaction


@dataclass
class Level:
    price: float
    touches: int
    kind: str  # support | resistance (relative to current price)
    last_touch_ms: int
    reactions_ms: list[int] = field(default_factory=list)  # first bar of each reaction


def count_reactions(highs: np.ndarray, lows: np.ndarray, level: float, tol: float, gap: int = REACTION_GAP) -> tuple[list[int], np.ndarray]:
    """Return (index of the first bar of each reaction, indices of every touching bar)."""
    touch = ((highs >= level - tol) & (highs <= level + tol)) | ((lows >= level - tol) & (lows <= level + tol))
    idx = np.flatnonzero(touch)
    starts: list[int] = []
    prev = None
    for i in idx:
        if prev is None or i - prev > gap:
            starts.append(int(i))
        prev = int(i)
    return starts, idx


def _in_band_extremes(highs: np.ndarray, lows: np.ndarray, level: float, tol: float) -> np.ndarray:
    return np.concatenate([highs[(highs >= level - tol) & (highs <= level + tol)],
                           lows[(lows >= level - tol) & (lows <= level + tol)]])


def support_resistance(df: pd.DataFrame, atr_value: float | None, left: int = 5, right: int = 5,
                       lookback: int = CHART_BARS, tol_atr: float = 0.35, max_levels: int = 6) -> list[Level]:
    d = df.tail(lookback).reset_index(drop=True)
    if len(d) < left + right + 2:
        return []
    highs = d["high"].to_numpy(dtype=float)
    lows = d["low"].to_numpy(dtype=float)
    ts = d["timestamp"].to_numpy()
    price = float(d["close"].iloc[-1])
    tol = tol_atr * atr_value if atr_value else price * 0.002

    pts = sorted([highs[i] for i in find_pivots(highs, left, right, "high")] +
                 [lows[i] for i in find_pivots(lows, left, right, "low")])
    if not pts:
        return []
    groups: list[list[float]] = [[pts[0]]]
    for p in pts[1:]:
        if p - groups[-1][0] <= 2 * tol:
            groups[-1].append(p)
        else:
            groups.append([p])

    candidates: list[Level] = []
    for g in groups:
        centre = float(np.median(g))
        extremes = _in_band_extremes(highs, lows, centre, tol)
        if extremes.size:
            centre = float(np.median(extremes))
        starts, idx = count_reactions(highs, lows, centre, tol)
        if not starts:
            continue
        candidates.append(Level(price=centre, touches=len(starts), kind="resistance" if centre > price else "support",
                                last_touch_ms=int(ts[idx[-1]]), reactions_ms=[int(ts[i]) for i in starts]))

    # after re-centring two groups can land on the same price: keep the stronger one
    candidates.sort(key=lambda l: l.price)
    merged: list[Level] = []
    for lv in candidates:
        if merged and abs(lv.price - merged[-1].price) <= tol:
            if lv.touches > merged[-1].touches:
                merged[-1] = lv
        else:
            merged.append(lv)

    half = max(1, max_levels // 2)
    above = sorted([l for l in merged if l.kind == "resistance"], key=lambda l: l.price - price)[:half]
    below = sorted([l for l in merged if l.kind == "support"], key=lambda l: price - l.price)[:half]
    return sorted(above + below, key=lambda l: l.price, reverse=True)
