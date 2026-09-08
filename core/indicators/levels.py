"""Swing-pivot support/resistance clustered by an ATR tolerance."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.indicators.pivots import find_pivots


@dataclass
class Level:
    price: float
    touches: int
    kind: str  # support | resistance (relative to current price)
    last_touch_ms: int


def support_resistance(df: pd.DataFrame, atr_value: float | None, left: int = 5, right: int = 5,
                       lookback: int = 200, tol_atr: float = 0.35, max_levels: int = 6) -> list[Level]:
    d = df.tail(lookback).reset_index(drop=True)
    if len(d) < left + right + 2:
        return []
    highs = d["high"].to_numpy(dtype=float)
    lows = d["low"].to_numpy(dtype=float)
    ts = d["timestamp"].to_numpy()
    price = float(d["close"].iloc[-1])
    pts = [(highs[i], int(ts[i])) for i in find_pivots(highs, left, right, "high")]
    pts += [(lows[i], int(ts[i])) for i in find_pivots(lows, left, right, "low")]
    if not pts:
        return []
    pts.sort()
    tol = tol_atr * atr_value if atr_value else price * 0.002
    clusters: list[list[tuple[float, int]]] = [[pts[0]]]
    for p in pts[1:]:
        if p[0] - clusters[-1][-1][0] <= tol:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    levels = []
    for c in clusters:
        mean = sum(p for p, _ in c) / len(c)
        levels.append(Level(price=mean, touches=len(c), kind="resistance" if mean > price else "support", last_touch_ms=max(t for _, t in c)))
    half = max(1, max_levels // 2)
    above = sorted([l for l in levels if l.kind == "resistance"], key=lambda l: l.price - price)[:half]
    below = sorted([l for l in levels if l.kind == "support"], key=lambda l: price - l.price)[:half]
    return sorted(above + below, key=lambda l: l.price, reverse=True)
