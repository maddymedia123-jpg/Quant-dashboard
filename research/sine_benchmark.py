"""Does the Even Better Sinewave beat a plain swing pivot (no indicator) under the same filters?

naive_fractal(df, 2): at bar t, bar t-2 was the highest (lowest) of the last 5 bars - "price made a swing
and came off it", what any oscillator turning also says. Same scoring as sine_wave_more.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sine_wave import naive_fractal  # noqa: E402
from sine_wave_more import (BIG, DATA, NORMAL, WARMUP, ebsw_signals, even_better_sinewave,  # noqa: E402
                            fib_cycle_mask, line, ny_mask, score)
from turns import find_turns, load  # noqa: E402

for label, path, intraday in (("BINANCE 1h", "binance_1h.csv", True), ("GATE 1h", "btc_1h.csv", True),
                              ("BINANCE 4h", "binance_4h.csv", False)):
    df = load(DATA / path)
    n = len(df)
    sigs = {"EBSW": ebsw_signals(even_better_sinewave(df.close)), "plain swing pivot": naive_fractal(df, 2)}
    fib = fib_cycle_mask(df)
    filters = {"alone": None, "+ Fib": fib}
    if intraday:
        ny = ny_mask(df)
        filters.update({"+ NY": ny, "+ NY + Fib": ny & fib})
    for tname, kw in (("normal", NORMAL), ("big", BIG)):
        turns = find_turns(df, **kw)
        print(f"\n[{label}] {tname} turns")
        for part, (lo, hi) in (("1st half", (WARMUP, n // 2)), ("2nd half", (n // 2, n - 13))):
            for fname, f in filters.items():
                for sname, s in sigs.items():
                    print(line(f"{part} {sname} {fname}", score(s, turns, n, lo, hi, f)))
