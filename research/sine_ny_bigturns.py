"""The one promising cell, checked by period: Ehlers Sine Wave, BIG turns, New York hours, 15m candles.

Gate's newest 10,000 15m bars (about 3.5 months) showed 2.19x over the next 2 bars; Binance 2023-2026 as a
whole showed 0.95-0.99x. If the edge is real but recent, the whole-history number would bury it, so
Binance is split by year and also cut to the SAME months as Gate's sample. Forward-only, as
sine_forward.py: the turn must come in the next k bars.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from sine_forward import ahead_mask  # noqa: E402
from sine_wave import ehlers_sine, sine_signals  # noqa: E402
from sine_wave_more import BIG, DATA, ny_mask  # noqa: E402
from turns import find_turns, load  # noqa: E402

YEAR_MS = 365.25 * 86_400_000


def cell(df, lo_ms, hi_ms, label):
    n = len(df)
    sig = sine_signals(*ehlers_sine(df.high, df.low))
    turns = find_turns(df, **BIG)
    ts = df["timestamp"].to_numpy()
    ny = ny_mask(df)
    span = (ts >= lo_ms) & (ts < hi_ms) & (np.arange(n) > 100) & (np.arange(n) < n - 13)
    for k in (2, 4):
        ahead = ahead_mask(n, turns["idx"], k)
        same = (ahead_mask(n, turns.loc[turns.kind == "bottom", "idx"], k) & (sig == 1)) | \
               (ahead_mask(n, turns.loc[turns.kind == "top", "idx"], k) & (sig == -1))
        on = span & ny & (sig != 0)
        base = ahead[span & ny].mean()
        hit = ahead[on].mean() if on.any() else float("nan")
        z = (hit - base) / math.sqrt(base * (1 - base) / on.sum()) if on.sum() and 0 < base < 1 else float("nan")
        print(f"  {label:34s} next {k} bars: n={on.sum():4d}  hit {hit:5.1%} vs NY random {base:5.1%} "
              f"({hit / base:4.2f}x, z {z:4.1f})  right direction {same[on].mean():5.1%}")


if __name__ == "__main__":
    b = load(DATA / "binance_15m.csv")
    g = load(DATA / "gate_15m.csv").iloc[:-1].reset_index(drop=True)
    import datetime as dt
    ms = lambda y, m, d=1: int(dt.datetime(y, m, d, tzinfo=dt.timezone.utc).timestamp() * 1000)  # noqa: E731
    g0, g1 = int(g.timestamp.iloc[0]), int(g.timestamp.iloc[-1])
    print(f"Gate sample: {dt.datetime.fromtimestamp(g0/1000, dt.timezone.utc):%d %b %Y} -> "
          f"{dt.datetime.fromtimestamp(g1/1000, dt.timezone.utc):%d %b %Y}")
    cell(g, g0, g1 + 1, "GATE (the promising result)")
    cell(b, g0, ms(2026, 10), "BINANCE, same months as Gate")
    for y in (2023, 2024, 2025, 2026):
        cell(b, ms(y, 1), ms(y + 1, 1), f"BINANCE {y}")
    for a, z in ((ms(2023, 1), ms(2024, 7)), (ms(2024, 7), ms(2026, 10))):
        cell(b, a, z, f"BINANCE {dt.datetime.fromtimestamp(a/1000):%b %Y}-{dt.datetime.fromtimestamp(z/1000):%b %Y}")
