"""Sine-wave follow-up: the three things the first test left out (23 Oct 2026).

1. Another sine indicator: Ehlers' Even Better Sinewave (Cycle Analytics for Traders, 2013), with its
   published defaults (high-pass 40, super smoother 10) - nothing tuned. Signals: the wave crossing back
   inside +-0.85 after being beyond it (the overbought / oversold reading), and the plain Ehlers Sine Wave
   from research/sine_wave.py for comparison.
2. Other timeframes: 1h, 4h and daily. The turn definition is in BARS (an extreme within +-6 bars,
   followed by 3 ATR within 6 bars; big: +-12 bars, 5 ATR), so it scales with the timeframe.
3. His combination (spec v15 s.3: cycle completion cross-referenced with the kill zones and the time
   cycles): sine signals kept only when they fall in the New York kill zone (13-17 UTC, the one zone that
   beat chance), only when the bar count from a recent major swing is a Fibonacci number (3, 5, 8, 13, 21),
   and both.

The fair baseline for a combination is random hours INSIDE the same filter: New York hours already hold
1.8x their share of turns, so a sine signal in New York must beat New York hours, not the whole day, to
add anything. Both baselines are printed.

All signals are causal (computed from bars <= t). Samples: Binance BTCUSDT perp 2020-01 .. 2026-09 (halves
reported separately), and Gate.io BTC_USDT perp's newest 10,000 hours.

    python research/sine_wave_more.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from sine_wave import ehlers_sine, sine_signals  # noqa: E402
from turns import find_turns, load, near_turn_mask  # noqa: E402

DATA = Path(__file__).parent / "data"
WITHIN, WARMUP = 2, 100
NORMAL = dict(half=6, k_atr=3.0, follow=6)
BIG = dict(half=12, k_atr=5.0, follow=12)
FIB = (3, 5, 8, 13, 21)
H_MS = 3_600_000


def even_better_sinewave(close, hp_period: int = 40, ss_period: int = 10) -> np.ndarray:
    """Ehlers 2013, as published: one-pole high-pass, two-pole super smoother, wave normalised by power."""
    c = np.asarray(close, float)
    n = len(c)
    ang = 2 * math.pi / hp_period
    a1 = (1 - math.sin(ang)) / math.cos(ang)
    a = math.exp(-1.414 * math.pi / ss_period)
    c2, c3 = 2 * a * math.cos(1.414 * math.pi / ss_period), -a * a
    c1 = 1 - c2 - c3
    hp, filt, wave = np.zeros(n), np.zeros(n), np.full(n, np.nan)
    for t in range(1, n):
        hp[t] = 0.5 * (1 + a1) * (c[t] - c[t - 1]) + a1 * hp[t - 1]
        if t >= 2:
            filt[t] = c1 * (hp[t] + hp[t - 1]) / 2 + c2 * filt[t - 1] + c3 * filt[t - 2]
            w = (filt[t] + filt[t - 1] + filt[t - 2]) / 3
            p = (filt[t] ** 2 + filt[t - 1] ** 2 + filt[t - 2] ** 2) / 3
            if p > 0:
                wave[t] = w / math.sqrt(p)
    return wave


def ebsw_signals(wave, level: float = 0.85) -> np.ndarray:
    """-1 top: the wave falls back below +level from above it; +1 bottom: rises back above -level."""
    s = np.zeros(len(wave), int)
    prev, cur = wave[:-1], wave[1:]
    ok = ~np.isnan(prev) & ~np.isnan(cur)
    s[1:][ok & (prev > level) & (cur <= level)] = -1
    s[1:][ok & (prev < -level) & (cur >= -level)] = 1
    return s


def ny_mask(df) -> np.ndarray:
    hour = (df["timestamp"].to_numpy() // H_MS) % 24
    return (hour >= 13) & (hour < 17)


def fib_cycle_mask(df, k: int = 6, last: int = 5) -> np.ndarray:
    """Bar t sits a Fibonacci number of bars after one of the last `last` MAJOR swings (a +-k-bar extreme),
    counting only swings already confirmed at t (k bars after them): causal, as a trader would count."""
    h, l = df["high"].to_numpy(float), df["low"].to_numpy(float)
    n = len(df)
    hi = pd.Series(h).rolling(2 * k + 1, center=True).max().to_numpy()
    lo = pd.Series(l).rolling(2 * k + 1, center=True).min().to_numpy()
    pivots = [i for i in range(k, n - k) if h[i] == hi[i] or l[i] == lo[i]]
    mask = np.zeros(n, bool)
    j, known = 0, []
    for t in range(n):
        while j < len(pivots) and pivots[j] + k <= t:       # confirmed by bar t
            known.append(pivots[j])
            j += 1
        if any((t - p) in FIB for p in known[-last:]):
            mask[t] = True
    return mask


def score(sig, turns, n, lo, hi, cond=None):
    near = near_turn_mask(n, turns["idx"], WITHIN)
    near_top = near_turn_mask(n, turns.loc[turns.kind == "top", "idx"], WITHIN)
    near_bot = near_turn_mask(n, turns.loc[turns.kind == "bottom", "idx"], WITHIN)
    r = slice(lo, hi)
    c = np.ones(n, bool)[r] if cond is None else cond[r]
    s = sig[r] * c
    k = int((s != 0).sum())
    if k < 1:
        return None
    hit = near[r][s != 0].mean()
    base_all = near[r].mean()
    base_cond = near[r][c].mean()
    hit_dir = (near_top[r][s == -1].sum() + near_bot[r][s == 1].sum()) / k
    z = (hit - base_cond) / math.sqrt(base_cond * (1 - base_cond) / k)
    return dict(n=k, hit=hit, base_all=base_all, base_cond=base_cond, lift_all=hit / base_all,
                lift_cond=hit / base_cond, z=z, hit_dir=hit_dir)


def line(name, r):
    if r is None:
        return f"  {name:40s} no signals"
    flag = "  (n<30, not significant)" if r["n"] < 30 else ""
    return (f"  {name:40s} n={r['n']:5d}  hit {r['hit']:5.1%} | random all {r['base_all']:5.1%} "
            f"({r['lift_all']:4.2f}x) | random same-filter {r['base_cond']:5.1%} ({r['lift_cond']:4.2f}x, "
            f"z {r['z']:4.1f}){flag}")


def evaluate(df, label, intraday: bool):
    n = len(df)
    sine, lead = ehlers_sine(df.high, df.low)
    sigs = {"Ehlers Sine Wave": sine_signals(sine, lead),
            "Even Better Sinewave (+-0.85)": ebsw_signals(even_better_sinewave(df.close))}
    fib = fib_cycle_mask(df)
    filters = {"alone": None, "+ Fibonacci bar count": fib}
    if intraday:
        ny = ny_mask(df)
        filters.update({"+ New York kill zone": ny, "+ New York + Fibonacci": ny & fib})
    for tname, kw in (("normal turns", NORMAL), ("big turns", BIG)):
        turns = find_turns(df, **kw)
        print(f"\n[{label}] {tname}: {len(turns)} turns in {n} bars")
        for part, (lo, hi) in (("all", (WARMUP, n - 13)), ("1st half", (WARMUP, n // 2)),
                               ("2nd half", (n // 2, n - 13))):
            for sname, s in sigs.items():
                for fname, f in filters.items():
                    print(line(f"{part:8s} {sname} {fname}", score(s, turns, n, lo, hi, f)))


def causality_check(df):
    full = even_better_sinewave(df.close)
    full_f = fib_cycle_mask(df)
    worst, fib_bad = 0.0, 0
    for cut in (800, 2500, len(df) - 9):
        w = even_better_sinewave(df.close[:cut])
        worst = max(worst, float(np.nanmax(np.abs(w - full[:cut]))))
        f = fib_cycle_mask(df.iloc[:cut].reset_index(drop=True))
        fib_bad += int((f != full_f[:cut]).sum())
    print(f"causality: Even Better Sinewave max change when the future is cut = {worst:.3g}; "
          f"Fibonacci-count bars that change = {fib_bad}")


if __name__ == "__main__":
    b1h = load(DATA / "binance_1h.csv")
    causality_check(b1h.iloc[:6000].reset_index(drop=True))
    evaluate(b1h, "BINANCE 1h 2020-2026", intraday=True)
    evaluate(load(DATA / "btc_1h.csv").iloc[:-1].reset_index(drop=True), "GATE 1h Sep 2025-Oct 2026", intraday=True)
    evaluate(load(DATA / "binance_4h.csv"), "BINANCE 4h 2020-2026", intraday=False)
    evaluate(load(DATA / "binance_1d.csv"), "BINANCE daily 2020-2026", intraday=False)
