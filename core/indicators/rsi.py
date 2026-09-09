"""Wilder RSI and pivot-based divergence detection (regular + hidden, confirmed + forming)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.indicators.pivots import find_pivots


@dataclass
class Divergence:
    kind: str        # regular | hidden
    direction: str   # bullish | bearish
    status: str      # confirmed | forming
    price_points: list[tuple[int, float]]
    rsi_points: list[tuple[int, float]]
    bars: tuple[int, int]

    @property
    def label(self) -> str:
        return f"{self.kind.title()} {self.direction} ({self.status})"


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.astype(float).diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    au = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    out = 100 - 100 / (1 + au / ad.replace(0.0, np.nan))
    out = out.mask((ad == 0) & (au > 0), 100.0)   # only gains: RSI 100
    out = out.mask((ad == 0) & (au == 0), 50.0)   # no movement at all: neutral
    return out


def detect_divergences(df: pd.DataFrame, rsi_series: pd.Series, left: int = 3, right: int = 3, lookback: int = 60) -> list[Divergence]:
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    ts = df["timestamp"].to_numpy()
    r = rsi_series.to_numpy(dtype=float)
    n = len(df)
    if n < left + right + 2:
        return []
    start = max(0, n - lookback)
    ph = [i for i in find_pivots(highs, left, right, "high") if i >= start]
    pl = [i for i in find_pivots(lows, left, right, "low") if i >= start]
    out: list[Divergence] = []

    def emit(kind, direction, status, p1, p2, arr):
        out.append(Divergence(kind, direction, status,
                              [(int(ts[p1]), float(arr[p1])), (int(ts[p2]), float(arr[p2]))],
                              [(int(ts[p1]), float(r[p1])), (int(ts[p2]), float(r[p2]))],
                              (p1, p2)))

    def check(p1, p2, arr, is_high, status):
        if np.isnan(r[p1]) or np.isnan(r[p2]) or arr[p1] == arr[p2] or r[p1] == r[p2]:
            return
        price_up = arr[p2] > arr[p1]
        rsi_up = r[p2] > r[p1]
        if is_high:
            if price_up and not rsi_up:
                emit("regular", "bearish", status, p1, p2, arr)
            elif not price_up and rsi_up:
                emit("hidden", "bearish", status, p1, p2, arr)
        else:
            if not price_up and rsi_up:
                emit("regular", "bullish", status, p1, p2, arr)
            elif price_up and not rsi_up:
                emit("hidden", "bullish", status, p1, p2, arr)

    if len(ph) >= 2:
        check(ph[-2], ph[-1], highs, True, "confirmed")
    if len(pl) >= 2:
        check(pl[-2], pl[-1], lows, False, "confirmed")

    last = n - 1
    if ph and last - ph[-1] > right and highs[last] >= highs[last - left:last].max():
        check(ph[-1], last, highs, True, "forming")
    if pl and last - pl[-1] > right and lows[last] <= lows[last - left:last].min():
        check(pl[-1], last, lows, False, "forming")
    return out
