"""The turn definition shared by the research scripts, rebuilt from core/indicators/turn_window.py's docstring.

A turn is a TOP or a BOTTOM: the bar's high (low) is the highest (lowest) within +-`half` hours, and price
then moves at least `k_atr` ATR(14) away from that extreme within the next `follow` hours, measured on
the extreme bar's ATR. The extreme bar's open hour (UTC) is the hour of the turn.

The earlier docstring does not say how long the follow-through may take; `follow` is the one free choice
here, reported with the counts (see kill_zones.py --calibrate).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

H_MS = 3_600_000


def load(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts = df["timestamp"].astype("int64")
    if ts.iloc[-1] > 10**14:            # microseconds (Binance dumps from 2025 on spot)
        ts = ts // 1000
    df["timestamp"] = ts
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df


def atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    h, l, c = df["high"].to_numpy(float), df["low"].to_numpy(float), df["close"].to_numpy(float)
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.c_[h - l, np.abs(h - pc), np.abs(l - pc)], axis=1)
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False, min_periods=n).mean().to_numpy()


def find_turns(df: pd.DataFrame, half: int = 6, k_atr: float = 3.0, follow: int = 24) -> pd.DataFrame:
    """Rows: idx, timestamp, kind ('top'|'bottom'), hour (UTC). Uses the future: a LABEL, never a signal."""
    h, l = df["high"].to_numpy(float), df["low"].to_numpy(float)
    a = atr(df)
    n = len(df)
    hi_win = pd.Series(h).rolling(2 * half + 1, center=True).max().to_numpy()
    lo_win = pd.Series(l).rolling(2 * half + 1, center=True).min().to_numpy()
    fut_lo = pd.Series(l[::-1]).rolling(follow, min_periods=follow).min().to_numpy()[::-1]   # min of l[i..i+follow-1]
    fut_hi = pd.Series(h[::-1]).rolling(follow, min_periods=follow).max().to_numpy()[::-1]
    out = []
    for i in range(half, n - max(half, follow) - 1):
        if np.isnan(a[i]):
            continue
        if h[i] == hi_win[i] and fut_lo[i + 1] <= h[i] - k_atr * a[i]:
            out.append((i, "top"))
        if l[i] == lo_win[i] and fut_hi[i + 1] >= l[i] + k_atr * a[i]:
            out.append((i, "bottom"))
    t = pd.DataFrame(out, columns=["idx", "kind"])
    t["timestamp"] = df["timestamp"].to_numpy()[t["idx"]]
    t["hour"] = (t["timestamp"] // H_MS) % 24
    return t


def near_turn_mask(n: int, turn_idx, within: int = 2) -> np.ndarray:
    """mask[i] = a turn lies within +-within bars of bar i."""
    m = np.zeros(n, bool)
    for i in turn_idx:
        m[max(0, i - within): i + within + 1] = True
    return m
