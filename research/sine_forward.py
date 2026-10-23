"""Forward-only: does the signal come BEFORE the turn? A hit = a real turn in bars t+1 .. t+k only.

The +-2-bar score credits a signal that fires just after the turn. Here a turn at or before the signal bar
counts for nothing: only warning ahead of time scores. Baseline: the same window after a random bar.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from sine_wave import ehlers_sine, naive_fractal, sine_signals  # noqa: E402
from sine_wave_more import BIG, DATA, NORMAL, WARMUP, ebsw_signals, even_better_sinewave, ny_mask  # noqa: E402
from turns import find_turns, load  # noqa: E402


def ahead_mask(n, idx, k):
    """m[t] = a turn lies in bars t+1..t+k."""
    m = np.zeros(n, bool)
    for i in idx:
        m[max(0, i - k): i] = True
    return m


def run(df, label, intraday, ks=(2, 4)):
    n = len(df)
    sine, lead = ehlers_sine(df.high, df.low)
    sigs = {"Ehlers Sine Wave": sine_signals(sine, lead),
            "Even Better Sinewave": ebsw_signals(even_better_sinewave(df.close)),
            "plain swing pivot (no indicator)": naive_fractal(df, 2)}
    conds = {"all hours": None}
    if intraday:
        conds["NY hours only"] = ny_mask(df)
    for tname, kw in (("normal", NORMAL), ("big", BIG)):
        turns = find_turns(df, **kw)
        for k in ks:
            ahead = ahead_mask(n, turns["idx"], k)
            same = {"top": ahead_mask(n, turns.loc[turns.kind == "top", "idx"], k),
                    "bottom": ahead_mask(n, turns.loc[turns.kind == "bottom", "idx"], k)}
            print(f"\n[{label}] {tname} turns, turn must come in the NEXT {k} bars")
            for cname, c in conds.items():
                cm = np.ones(n, bool) if c is None else c
                r = slice(WARMUP, n - 13)
                base = ahead[r][cm[r]].mean()
                print(f"  {cname:14s} random bar: {base:5.1%}")
                for sname, s in sigs.items():
                    s = s[r] * cm[r]
                    on = s != 0
                    hit = ahead[r][on].mean()
                    # right direction: a bottom signal followed by a bottom, a top by a top
                    right = (same["bottom"][r][s == 1].sum() + same["top"][r][s == -1].sum()) / on.sum()
                    z = (hit - base) / math.sqrt(base * (1 - base) / on.sum())
                    print(f"  {cname:14s} {sname:34s} n={on.sum():5d} turn ahead {hit:5.1%} "
                          f"({hit / base:4.2f}x, z {z:4.1f})  right direction {right:5.1%}")


if __name__ == "__main__" and sys.argv[1:] == ["15m"]:
    # 15-minute candles: the next 2, 4 and 8 bars are the next 30 minutes, 1 hour and 2 hours
    run(load(DATA / "binance_15m.csv"), "BINANCE 15m 2023-2026", True, ks=(2, 4, 8))
    run(load(DATA / "gate_15m.csv").iloc[:-1].reset_index(drop=True), "GATE 15m (newest 10,000)", True, ks=(2, 4, 8))
    sys.exit(0)
if __name__ == "__main__":
    run(load(DATA / "binance_1h.csv"), "BINANCE 1h", True)
    run(load(DATA / "btc_1h.csv").iloc[:-1].reset_index(drop=True), "GATE 1h", True)
    run(load(DATA / "binance_4h.csv"), "BINANCE 4h", False)
