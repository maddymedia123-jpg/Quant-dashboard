"""TASK B: does "sine wave oscillation" isolate BTC's turning extremes, with no lookahead?

Spec v15 s.3 claims a smooth periodic oscillation isolates turning extremes. Tested here:

1. Ehlers' Sine Wave indicator ("Rocket Science for Traders", 2001, ch. 9): Hilbert-transform homodyne
   discriminator for the dominant cycle period, then the dominant-cycle phase from one period of smoothed
   price; Sine = sin(phase), LeadSine = sin(phase + 45). LeadSine crossing ABOVE Sine = cycle bottom,
   crossing BELOW = cycle top. No free parameters (the period adapts), so nothing to tune.
2. A simpler variant: Ehlers' band-pass filter at a fixed period, signalling at its peaks/troughs
   (slope turns, seen one bar late, so causal) and at its zero crosses. Period and bandwidth are tuned on
   the FIRST half of the Gate sample only, then reported on the second half and on older Binance years.

Every value at bar t is computed from bars <= t only (recursive filters and trailing sums); the
causality check at the bottom recomputes on truncated histories and compares.

Score: a signal at bar t is a HIT when a real turn (research/turns.py: +-6 h extreme followed by >= 3 ATR,
or the big +-12 h / 5 ATR kind) lies within bars t-2..t+2. Baseline: the share of ALL hours that have a turn
within +-2 hours, i.e. what the same number of signals placed at random hours would score on average.
Direction-matched: a bottom signal must find a bottom, a top signal a top.

    python research/sine_wave.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from turns import find_turns, load, near_turn_mask  # noqa: E402

DATA = Path(__file__).parent / "data"
WITHIN = 2
WARMUP = 100
NORMAL = dict(half=6, k_atr=3.0, follow=6)
BIG = dict(half=12, k_atr=5.0, follow=12)


# ----------------------------------------------------------------------------------------------- indicators
def ehlers_sine(high, low):
    """Sine and LeadSine per bar, Ehlers 2001, transcribed from the EasyLanguage (angles in degrees)."""
    n = len(high)
    price = (np.asarray(high, float) + np.asarray(low, float)) / 2
    z = lambda: np.zeros(n)  # noqa: E731
    smooth, det, q1, i1, ji, jq, i2, q2, re, im, per, sper = (z() for _ in range(12))
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
        i2[t] = i1[t] - jq[t]
        q2[t] = q1[t] + ji[t]
        i2[t] = 0.2 * i2[t] + 0.8 * i2[t - 1]
        q2[t] = 0.2 * q2[t] + 0.8 * q2[t - 1]
        re[t] = i2[t] * i2[t - 1] + q2[t] * q2[t - 1]
        im[t] = i2[t] * q2[t - 1] - q2[t] * i2[t - 1]
        re[t] = 0.2 * re[t] + 0.8 * re[t - 1]
        im[t] = 0.2 * im[t] + 0.8 * im[t - 1]
        p = p1
        if im[t] != 0 and re[t] != 0:
            ang = math.degrees(math.atan(im[t] / re[t]))
            if ang != 0:
                p = 360 / ang
        if p > 1.5 * p1:
            p = 1.5 * p1
        if p < 0.67 * p1:
            p = 0.67 * p1
        p = min(max(p, 6), 50)
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
        phase += 90
        phase += 360 / sper[t]
        if imag < 0:
            phase += 180
        if phase > 315:
            phase -= 360
        sine[t] = math.sin(math.radians(phase))
        lead[t] = math.sin(math.radians(phase + 45))
    return sine, lead


def sine_signals(sine, lead):
    """+1 bottom (LeadSine crosses above Sine), -1 top (crosses below), at the bar of the cross."""
    s = np.zeros(len(sine), int)
    d = lead - sine
    prev, cur = d[:-1], d[1:]
    ok = ~np.isnan(prev) & ~np.isnan(cur)
    s[1:][ok & (prev <= 0) & (cur > 0)] = 1
    s[1:][ok & (prev >= 0) & (cur < 0)] = -1
    return s


def bandpass(high, low, period, delta):
    """Ehlers band-pass (Cycle Analytics for Traders, 2013) on the mid price; recursive, causal."""
    x = (np.asarray(high, float) + np.asarray(low, float)) / 2
    beta = math.cos(2 * math.pi / period)
    gamma = 1 / math.cos(4 * math.pi * delta / period)
    alpha = gamma - math.sqrt(gamma * gamma - 1)
    bp = np.zeros(len(x))
    for t in range(2, len(x)):
        bp[t] = 0.5 * (1 - alpha) * (x[t] - x[t - 2]) + beta * (1 + alpha) * bp[t - 1] - alpha * bp[t - 2]
    return bp


def bp_extreme_signals(bp):
    """Peak seen at bar t when bp[t-1] was above both neighbours: -1 top; trough: +1 bottom. Causal."""
    s = np.zeros(len(bp), int)
    a, b, c = bp[:-2], bp[1:-1], bp[2:]
    s[2:][(b > a) & (b > c)] = -1
    s[2:][(b < a) & (b < c)] = 1
    return s


def bp_zero_signals(bp):
    """Band-pass crossing zero: down-cross -1 (read as top), up-cross +1 (read as bottom)."""
    s = np.zeros(len(bp), int)
    s[1:][(bp[:-1] >= 0) & (bp[1:] < 0)] = -1
    s[1:][(bp[:-1] <= 0) & (bp[1:] > 0)] = 1
    return s


# ---------------------------------------------------------------------------------------------- scoring
def score(sig, turns, n, lo, hi):
    """Hit rate vs random-hour baseline for signals in bars [lo, hi)."""
    near_any = near_turn_mask(n, turns["idx"], WITHIN)
    near_top = near_turn_mask(n, turns.loc[turns.kind == "top", "idx"], WITHIN)
    near_bot = near_turn_mask(n, turns.loc[turns.kind == "bottom", "idx"], WITHIN)
    rng = slice(lo, hi)
    s = sig[rng]
    k = int((s != 0).sum())
    if k == 0:
        return dict(n=0)
    hit = near_any[rng][s != 0].mean()
    base = near_any[rng].mean()
    tops, bots = (s == -1), (s == 1)
    hit_dir = (near_top[rng][tops].sum() + near_bot[rng][bots].sum()) / k
    base_dir = (tops.sum() * near_top[rng].mean() + bots.sum() * near_bot[rng].mean()) / k
    z = (hit - base) / math.sqrt(base * (1 - base) / k)
    return dict(n=k, hit=hit, base=base, lift=hit / base, z=z, hit_dir=hit_dir, base_dir=base_dir,
                lift_dir=hit_dir / base_dir if base_dir else float("nan"))


def fmt(name, r):
    if not r.get("n"):
        return f"  {name:44s} no signals"
    return (f"  {name:44s} n={r['n']:5d}  hit {r['hit']:5.1%} vs {r['base']:5.1%} random ({r['lift']:4.2f}x, "
            f"z {r['z']:4.1f})  same-direction {r['hit_dir']:5.1%} vs {r['base_dir']:5.1%} ({r['lift_dir']:4.2f}x)")


def lookahead_canary(df, turns):
    """A signal that cheats (fires on every bar that is the +-6 h extreme, which needs 6 future bars):
    proves the scorer CAN report an edge when there is one."""
    h, l = df.high.to_numpy(), df.low.to_numpy()
    hi = pd.Series(h).rolling(13, center=True).max().to_numpy()
    lo_ = pd.Series(l).rolling(13, center=True).min().to_numpy()
    s = np.zeros(len(df), int)
    s[h == hi] = -1
    s[l == lo_] = 1
    return s


def naive_fractal(df, k):
    """No-skill causal benchmark: at bar t, bar t-k's high (low) was the highest (lowest) of bars t-2k..t.
    It is just 'price made a swing and has come off it for k bars' - what any filter turning also says."""
    h, l = df.high.to_numpy(), df.low.to_numpy()
    hi = pd.Series(h).rolling(2 * k + 1).max().to_numpy()
    lo = pd.Series(l).rolling(2 * k + 1).min().to_numpy()
    s = np.zeros(len(df), int)
    s[k:][h[:-k] == hi[k:]] = -1
    s[k:][l[:-k] == lo[k:]] = 1
    return s


def naive_direction(df, sig, lookback=6, seed=7):
    """Same number of signals at RANDOM hours, called 'bottom' if price fell over the last `lookback`
    hours and 'top' if it rose - the no-skill benchmark for the same-direction score."""
    rng = np.random.default_rng(seed)
    c = df.close.to_numpy()
    s = np.zeros(len(df), int)
    k = int((sig != 0).sum())
    idx = rng.choice(np.arange(WARMUP, len(df) - 13), size=k, replace=False)
    s[idx] = np.where(c[idx] < c[idx - lookback], 1, -1)
    return s


def evaluate(df, label, bp_params):
    n = len(df)
    mid = n // 2
    sine, lead = ehlers_sine(df.high, df.low)
    sine_sig = sine_signals(sine, lead)
    sigs = {"Ehlers sine (LeadSine x Sine)": sine_sig,
            "  benchmark: random hours, dir = 6h move": naive_direction(df, sine_sig),
            "  benchmark: raw 1-bar fractal (no filter)": naive_fractal(df, 1),
            "  benchmark: raw 2-bar fractal (no filter)": naive_fractal(df, 2),
            f"band-pass extreme P={bp_params[0]} d={bp_params[1]}": bp_extreme_signals(bandpass(df.high, df.low, *bp_params)),
            f"band-pass zero-cross P={bp_params[0]} d={bp_params[1]}": bp_zero_signals(bandpass(df.high, df.low, *bp_params))}
    for tname, kw in (("normal turns (+-6h, 3 ATR)", NORMAL), ("big turns (+-12h, 5 ATR)", BIG)):
        turns = find_turns(df, **kw)
        print(f"\n[{label}] {tname}: {len(turns)} turns in {n} bars")
        for part, (lo, hi) in (("all", (WARMUP, n - 13)), ("1st half", (WARMUP, mid)), ("2nd half", (mid, n - 13))):
            for name, s in sigs.items():
                print(fmt(f"{part:8s} {name}", score(s, turns, n, lo, hi)))
        print(fmt("all      CANARY lookahead +-6h extreme", score(lookahead_canary(df, turns), turns, n, WARMUP, n - 13)))


def tune_bandpass(df):
    """Pick (period, delta) by same-direction lift on the FIRST half of the Gate sample only."""
    n = len(df)
    turns = find_turns(df, **NORMAL)
    best = None
    print("\nband-pass tuning, Gate FIRST HALF only (normal turns), extreme signals:")
    for p in (10, 14, 20, 30, 40, 50):
        for d in (0.1, 0.3, 0.5):
            r = score(bp_extreme_signals(bandpass(df.high, df.low, p, d)), turns, n, WARMUP, n // 2)
            print(f"  P={p:2d} d={d}  n={r['n']:5d} lift {r['lift']:4.2f}x  same-dir {r['lift_dir']:4.2f}x")
            if best is None or r["lift_dir"] > best[0]:
                best = (r["lift_dir"], (p, d))
    print("  chosen on first half:", best[1])
    return best[1]


def causality_check(df):
    """Values at bar t must not change when bars after t are removed."""
    full_s, full_l = ehlers_sine(df.high, df.low)
    full_bp = bandpass(df.high, df.low, 20, 0.3)
    worst = 0.0
    for cut in (500, 1234, 3000, len(df) - 7):
        s, l = ehlers_sine(df.high[:cut], df.low[:cut])
        bp = bandpass(df.high[:cut], df.low[:cut], 20, 0.3)
        worst = max(worst, np.nanmax(np.abs(s - full_s[:cut])), np.nanmax(np.abs(l - full_l[:cut])),
                    np.max(np.abs(bp - full_bp[:cut])))
    print(f"\ncausality: max change at any bar when the future is cut off = {worst:.3g}")


if __name__ == "__main__":
    gate = load(DATA / "btc_1h.csv").iloc[:-1].reset_index(drop=True)
    causality_check(gate)
    params = tune_bandpass(gate)
    evaluate(gate, "GATE", params)
    bpath = DATA / "binance_1h.csv"
    if bpath.exists():
        evaluate(load(bpath), "BINANCE 2020-2025 (out-of-sample)", params)
