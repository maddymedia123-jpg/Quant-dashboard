"""EXPERIMENTAL, NOT PROVEN: the Ehlers Sine Wave as a 30-minute warning of a big turn, New York hours only.

His spec (v15, Section 3) asks for sine-wave oscillation to isolate turning extremes. Tested on 15m, 1h,
4h and daily candles, two to six years, alone and with the kill zones and Fibonacci counts (research/), it
did no better than a random hour - with one exception, which is what this module shows:

    Ehlers Sine Wave, 15-minute candles, New York hours (13:00-17:00 UTC), a BIG turn (the extreme of
    +-12 bars, followed by 5 ATR within 12 bars) starting in the next 2 bars (30 minutes):
        11 Jul - 23 Oct 2026   Gate 2.19x random NY bars, Binance 1.94x, Bybit 2.04x, Bitget 2.44x,
                               Coinbase spot 1.73x (about 104 signals and 9 hits on each)
        Binance by year        2023 0.92x, 2024 0.72x, 2025 1.06x, 2026 1.33x
        within 60 minutes      1.27-1.61x

Why it is not proven: the exchanges trade the same bitcoin, so five venues agreeing is close to one result
seen five times; it is about nine events in three and a half months; it was found by looking through
roughly a hundred combinations, and a few always look this good by luck; and before 2025 it was below
chance. So it is shown labelled EXPERIMENTAL, never fed to the agents or the alerts, and its LIVE record
from GO_LIVE_MS is replayed from candles on every page load - signals the method could not have been
picked for. That record decides whether it stays.

The indicator (Ehlers, Rocket Science for Traders, 2001, ch. 9): the dominant cycle period from a Hilbert
transform, its phase from one period of smoothed price, Sine = sin(phase), LeadSine = sin(phase + 45).
LeadSine crossing above Sine = a cycle bottom; below = a cycle top. Every value at a bar uses only that bar
and earlier ones, and only completed candles are read."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

BAR_MS = 900_000
NY_START_UTC, NY_END_UTC = 13, 17
AHEAD_BARS = 2                      # the warning: a big turn starting within the next 30 minutes
BIG_HALF, BIG_ATR = 12, 5.0         # a big turn: the extreme of +-12 bars, then 5 ATR within 12 bars
SETTLE_BARS = 2 * BIG_HALF + AHEAD_BARS   # bars after a signal before its outcome is known
GO_LIVE_MS = int(datetime(2026, 10, 24, tzinfo=timezone.utc).timestamp() * 1000)
MEASURED = ("15m candles, 11 Jul - 23 Oct 2026: 1.7-2.4x random New York bars on Gate, Binance, Bybit, Bitget "
            "and Coinbase (about 9 hits in 104 signals each). Binance 2023-2025: 0.72-1.06x. Not proven.")


def ehlers_sine(high, low) -> tuple[np.ndarray, np.ndarray]:
    """Sine and LeadSine per bar, transcribed from Ehlers' EasyLanguage (angles in degrees)."""
    n = len(high)
    price = (np.asarray(high, float) + np.asarray(low, float)) / 2
    smooth, det, q1, i1, ji, jq, i2, q2, re, im, per, sper = (np.zeros(n) for _ in range(12))
    sine, lead = np.full(n, np.nan), np.full(n, np.nan)
    phase = 0.0

    def hilbert(x, t, p):
        return (0.0962 * x[t] + 0.5769 * x[t - 2] - 0.5769 * x[t - 4] - 0.0962 * x[t - 6]) * (0.075 * p + 0.54)

    for t in range(n):
        if t < 6:
            per[t] = sper[t] = 0.0 if t == 0 else per[t - 1]
            continue
        smooth[t] = (4 * price[t] + 3 * price[t - 1] + 2 * price[t - 2] + price[t - 3]) / 10
        p1 = per[t - 1]
        det[t] = hilbert(smooth, t, p1)
        q1[t] = hilbert(det, t, p1)
        i1[t] = det[t - 3]
        ji[t] = hilbert(i1, t, p1)
        jq[t] = hilbert(q1, t, p1)
        i2[t] = 0.2 * (i1[t] - jq[t]) + 0.8 * i2[t - 1]
        q2[t] = 0.2 * (q1[t] + ji[t]) + 0.8 * q2[t - 1]
        re[t] = 0.2 * (i2[t] * i2[t - 1] + q2[t] * q2[t - 1]) + 0.8 * re[t - 1]
        im[t] = 0.2 * (i2[t] * q2[t - 1] - q2[t] * i2[t - 1]) + 0.8 * im[t - 1]
        p = p1
        if im[t] != 0 and re[t] != 0:
            ang = math.degrees(math.atan(im[t] / re[t]))
            if ang != 0:
                p = 360 / ang
        p = min(max(min(max(p, 0.67 * p1), 1.5 * p1), 6), 50)
        per[t] = 0.2 * p + 0.8 * p1
        sper[t] = 0.33 * per[t] + 0.67 * sper[t - 1]
        dc = int(sper[t] + 0.5)
        if t < 50 or dc < 1:
            continue
        real = sum(math.sin(math.radians(360 * k / dc)) * smooth[t - k] for k in range(dc))
        imag = sum(math.cos(math.radians(360 * k / dc)) * smooth[t - k] for k in range(dc))
        if abs(imag) > 0:
            phase = math.degrees(math.atan(real / imag))
        if abs(imag) <= 0.001:
            phase += 90 * np.sign(real)
        phase += 90 + 360 / sper[t]
        if imag < 0:
            phase += 180
        if phase > 315:
            phase -= 360
        sine[t] = math.sin(math.radians(phase))
        lead[t] = math.sin(math.radians(phase + 45))
    return sine, lead


def crossings(df: pd.DataFrame) -> np.ndarray:
    """+1 cycle bottom (LeadSine crosses above Sine), -1 cycle top, 0 none - per bar."""
    sine, lead = ehlers_sine(df["high"], df["low"])
    d = lead - sine
    s = np.zeros(len(df), int)
    prev, cur = d[:-1], d[1:]
    ok = ~np.isnan(prev) & ~np.isnan(cur)
    s[1:][ok & (prev <= 0) & (cur > 0)] = 1
    s[1:][ok & (prev >= 0) & (cur < 0)] = -1
    return s


def in_ny(ts_ms) -> np.ndarray:
    hour = (np.asarray(ts_ms, "int64") // 3_600_000) % 24
    return (hour >= NY_START_UTC) & (hour < NY_END_UTC)


def completed(df: pd.DataFrame | None, now_ms: int) -> pd.DataFrame | None:
    """Only candles that have closed: the one still forming would move the signal under the reader."""
    if df is None or not len(df):
        return None
    return df[df["timestamp"].astype("int64") + BAR_MS <= now_ms].reset_index(drop=True)


def _atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    h, l, c = df["high"].to_numpy(float), df["low"].to_numpy(float), df["close"].to_numpy(float)
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.c_[h - l, np.abs(h - pc), np.abs(l - pc)], axis=1)
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False, min_periods=n).mean().to_numpy()


def big_turns(df: pd.DataFrame) -> list[tuple[int, str]]:
    """(bar, 'top'|'bottom') for every big turn, judged with hindsight - a LABEL for scoring, never a signal."""
    h, l = df["high"].to_numpy(float), df["low"].to_numpy(float)
    a, n, w = _atr(df), len(df), 2 * BIG_HALF + 1
    hi = pd.Series(h).rolling(w, center=True).max().to_numpy()
    lo = pd.Series(l).rolling(w, center=True).min().to_numpy()
    out = []
    for i in range(BIG_HALF, n - BIG_HALF - 1):
        if np.isnan(a[i]):
            continue
        after = slice(i + 1, i + 1 + BIG_HALF)
        if h[i] == hi[i] and l[after].min() <= h[i] - BIG_ATR * a[i]:
            out.append((i, "top"))
        if l[i] == lo[i] and h[after].max() >= l[i] + BIG_ATR * a[i]:
            out.append((i, "bottom"))
    return out


@dataclass(frozen=True)
class Warning:
    side: str                # "bottom" (a turn up) or "top" (a turn down)
    at_ms: int               # the close of the signal candle
    price: float


def current(df15: pd.DataFrame | None, now_ms: int) -> tuple[Warning | None, bool]:
    """(the active warning or None, whether New York hours are on now). A warning is a crossing on one of
    the last AHEAD_BARS completed candles inside New York hours: after that its 30 minutes have passed."""
    df = completed(df15, now_ms)
    ny_now = bool(in_ny([now_ms])[0])
    if df is None or len(df) < 100:
        return None, ny_now
    s = crossings(df)
    ts = df["timestamp"].to_numpy("int64")
    for i in range(len(df) - 1, max(len(df) - 1 - AHEAD_BARS, 0), -1):
        if s[i] != 0 and in_ny([ts[i]])[0]:
            return Warning("bottom" if s[i] == 1 else "top", int(ts[i]) + BAR_MS, float(df["close"].iloc[i])), ny_now
    return None, ny_now


@dataclass(frozen=True)
class Record:
    since_ms: int
    signals: int
    hits: int                # a big turn started within the next 30 minutes
    right_way: int           # ... and it was the turn the signal named
    base: float | None       # the same, for any New York bar in the period

    @property
    def hit_rate(self) -> float | None:
        return self.hits / self.signals if self.signals else None

    @property
    def lift(self) -> float | None:
        return (self.hit_rate / self.base) if self.signals and self.base else None


def live_record(df15: pd.DataFrame | None, now_ms: int, since_ms: int = GO_LIVE_MS) -> Record:
    """Every New York signal since `since_ms` whose 30 minutes AND whose turn label have settled, scored.

    Replayed from candles each time, so it needs no database (Streamlit Cloud wipes one on every deploy)
    and it is the same however often the page loads."""
    df = completed(df15, now_ms)
    if df is None or len(df) < 100:
        return Record(since_ms, 0, 0, 0, None)
    n = len(df)
    ts = df["timestamp"].to_numpy("int64")
    s = crossings(df)
    ahead = np.zeros(n, bool)
    ahead_side = {"top": np.zeros(n, bool), "bottom": np.zeros(n, bool)}
    for i, kind in big_turns(df):
        lo = max(0, i - AHEAD_BARS)
        ahead[lo:i] = True
        ahead_side[kind][lo:i] = True
    scored = (ts >= since_ms) & (np.arange(n) < n - SETTLE_BARS) & (np.arange(n) >= 100) & in_ny(ts)
    on = scored & (s != 0)
    hits = int(ahead[on].sum())
    right = int((ahead_side["bottom"] & on & (s == 1)).sum() + (ahead_side["top"] & on & (s == -1)).sum())
    base = float(ahead[scored].mean()) if scored.any() else None
    return Record(since_ms, int(on.sum()), hits, right, base)
