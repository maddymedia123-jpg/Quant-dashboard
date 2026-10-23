"""TASK A: do BTC's turns cluster in the spec's three "institutional kill zones"?

Spec v15 s.3: Asian 00-04 UTC, London 07-10 UTC, New York 13-17 UTC. Compared with the dashboard's own
measured window, 12-16 UTC. For each zone: share of all turns inside it, hours covered, and times chance
(share / (hours/24)). Normal turns (+-6 h, 3 ATR) and big turns (+-12 h, 5 ATR). Robustness: first vs
second half of the sample by date, and DST (US/UK summer time vs winter time).

    python research/kill_zones.py [--calibrate]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from turns import find_turns, load  # noqa: E402

DATA = Path(__file__).parent / "data"
ZONES = {"Asian 00-04": range(0, 4), "London 07-10": range(7, 10), "New York 13-17": range(13, 17),
         "Ours 12-16": range(12, 16)}
NORMAL = dict(half=6, k_atr=3.0, follow=6)     # the move completes inside the same +-6 h: 353 turns on Gate (the
#                                              earlier run counted 395; follow=7 gives 402, see --calibrate)
BIG = dict(half=12, k_atr=5.0, follow=12)


def _second_sunday_march(y):  # US DST start
    d = pd.Timestamp(y, 3, 1)
    return d + pd.Timedelta(days=(6 - d.weekday()) % 7 + 7)


def _first_sunday_nov(y):
    d = pd.Timestamp(y, 11, 1)
    return d + pd.Timedelta(days=(6 - d.weekday()) % 7)


def _last_sunday(y, m):
    d = pd.Timestamp(y, m + 1, 1) - pd.Timedelta(days=1)
    return d - pd.Timedelta(days=(d.weekday() - 6) % 7)


def us_summer(ts_ms: pd.Series) -> np.ndarray:
    t = pd.to_datetime(ts_ms, unit="ms")
    return np.array([_second_sunday_march(x.year) <= x < _first_sunday_nov(x.year) for x in t])


def uk_summer(ts_ms: pd.Series) -> np.ndarray:
    t = pd.to_datetime(ts_ms, unit="ms")
    return np.array([_last_sunday(x.year, 3) <= x < _last_sunday(x.year, 10) for x in t])


def zone_table(turns: pd.DataFrame) -> list[tuple]:
    n = len(turns)
    rows = []
    for name, hrs in ZONES.items():
        k = int(turns["hour"].isin(list(hrs)).sum())
        share = k / n if n else float("nan")
        exp = len(hrs) / 24
        se = np.sqrt(exp * (1 - exp) / n) if n else float("nan")     # binomial, turns treated as independent
        rows.append((name, len(hrs), k, share, share / exp, (share - exp) / se))
    return rows


def print_table(title: str, turns: pd.DataFrame):
    print(f"\n{title}: {len(turns)} turns")
    print(f"  {'zone':16s} {'hrs':>3s} {'turns':>5s} {'share':>6s} {'x chance':>8s} {'z':>5s}")
    for name, hrs, k, share, x, z in zone_table(turns):
        print(f"  {name:16s} {hrs:3d} {k:5d} {share:6.1%} {x:8.2f} {z:5.1f}")


def hist(turns: pd.DataFrame, title: str):
    n = len(turns)
    c = turns["hour"].value_counts().reindex(range(24), fill_value=0)
    print(f"\n{title} per-hour histogram ({n} turns; x chance = share*24)")
    for h, k in c.items():
        print(f"  {h:02d} UTC {k:4d} {k / n * 24:5.2f}x {'#' * int(round(k / n * 24 * 10))}")


def run(df: pd.DataFrame, label: str):
    print("=" * 78)
    t0, t1 = pd.to_datetime(df.timestamp.iloc[[0, -1]], unit="ms")
    print(f"{label}: {len(df)} hourly bars, {t0:%Y-%m-%d} -> {t1:%Y-%m-%d}")
    for tname, kw in (("NORMAL (+-6h, 3 ATR)", NORMAL), ("BIG (+-12h, 5 ATR)", BIG)):
        t = find_turns(df, **kw)
        print_table(f"[{label}] {tname} - all", t)
        hist(t, f"[{label}] {tname}")
        mid = df.timestamp.iloc[len(df) // 2]
        print_table(f"[{label}] {tname} - first half (< {pd.to_datetime(mid, unit='ms'):%Y-%m-%d})",
                    t[t.timestamp < mid])
        print_table(f"[{label}] {tname} - second half", t[t.timestamp >= mid])
        us = us_summer(t.timestamp)
        uk = uk_summer(t.timestamp)
        print_table(f"[{label}] {tname} - US summer time (EDT, NYSE opens 13:30 UTC)", t[us])
        print_table(f"[{label}] {tname} - US winter time (EST, NYSE opens 14:30 UTC)", t[~us])
        print_table(f"[{label}] {tname} - UK summer time (BST)", t[uk])
        print_table(f"[{label}] {tname} - UK winter time (GMT)", t[~uk])


def calibrate(df):
    for follow in (6, 12, 24, 48, 72):
        print("follow", follow, "normal", len(find_turns(df, 6, 3.0, follow)),
              "big", len(find_turns(df, 12, 5.0, follow)))


def sensitivity(df, label):
    """Zone x-chance across follow-through windows, so no conclusion hangs on that one free choice."""
    print(f"\n[{label}] sensitivity to the follow-through window (x chance per zone)")
    for name, base in (("normal", NORMAL), ("big", BIG)):
        for mult in (1, 2, 4):
            kw = dict(base, follow=base["half"] * mult)
            t = find_turns(df, **kw)
            cells = "  ".join(f"{z.split()[0]} {x:4.2f}" for z, _, _, _, x, _ in zone_table(t))
            print(f"  {name:6s} follow {kw['follow']:3d}h  n={len(t):5d}  {cells}")


if __name__ == "__main__":
    gate = load(DATA / "btc_1h.csv").iloc[:-1]            # newest candle is still in progress
    bpath = DATA / "binance_1h.csv"
    if "--calibrate" in sys.argv:
        calibrate(gate)
    elif "--sens" in sys.argv:
        sensitivity(gate, "GATE")
        if bpath.exists():
            sensitivity(load(bpath), "BINANCE")
    else:
        run(gate, "GATE BTC_USDT perp")
        if bpath.exists():
            run(load(bpath), "BINANCE BTCUSDT perp (older, out-of-sample)")
