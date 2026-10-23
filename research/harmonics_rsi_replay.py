"""Throwaway replay: does a regular RSI divergence at D improve harmonic setups? (spec v15, 4.2)

Rebuilds the replay behind harmonics.TRACK_RECORD on the Gate.io BTC_USDT perp - the contract the dashboard
reads first. Gate serves only the newest 10,000 candles per interval, so the history is: 1h ~14 months,
4h ~4.5 years, 1d back to Nov 2019. Candles are cached in research/.cache (gitignored).

Walk: for every completed candle t, `harmonics.find` is called on the candles up to and including t plus
one more row standing in for the forming candle (find never reads it), so nothing after t is seen. A setup
is one pattern (name, side, X-A-B-C candles) and enters the sample the first candle it shows as in_zone
or completed. Its outcome is read on the candles after that one: target 1 hit (high/low touches it) before
a close past the invalidation level; a candle doing both counts as invalidated (conservative). Setups still
open at the end of the data are dropped.

Two views:
* at entry  - the flags the dashboard showed on the candle the setup reached its zone.
* RSI wait  - a setup that shows the divergence on some candle while still open is re-entered on that
              candle (targets and invalidation as shown then) and resolved from there; the rest are
              resolved from entry. This is the trader who waits for the divergence before acting.

Run from the repo root:  .venv/Scripts/python research/harmonics_rsi_replay.py"""
from __future__ import annotations

import pathlib
import sys
import time

import httpx
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from core.data.gate_candles import BASE, CONTRACT, HEADERS, parse_candles  # noqa: E402
from core.indicators import harmonics as hm  # noqa: E402

CACHE = pathlib.Path(__file__).with_name(".cache")
STEP_S = {"1h": 3600, "4h": 14400, "1d": 86400}
WINDOW = 600            # rows passed to find: 300-candle pattern window + RSI warm-up; levels read <=200
MAX_POINTS = 10_000


def candles(tf: str) -> pd.DataFrame:
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"gate_{tf}.csv"
    if path.exists():
        return pd.read_csv(path)
    step = STEP_S[tf]
    now = int(time.time()) // step * step
    start = now - (MAX_POINTS - 10) * step
    parts = []
    with httpx.Client(timeout=30) as c:
        frm = start
        while frm < now:
            to = min(now, frm + 1999 * step)
            r = c.get(f"{BASE}/candlesticks", headers=HEADERS,
                      params={"contract": CONTRACT, "interval": tf, "from": frm, "to": to})
            r.raise_for_status()
            rows = r.json()
            if rows:
                parts.append(parse_candles(rows))
            frm = to + step
    df = pd.concat(parts).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


def resolve(df: pd.DataFrame, t: int, bull: bool, t1: float, inv: float) -> tuple[str, int] | None:
    """('hit'|'invalid', candle) for the first outcome after candle t, None while still open."""
    hi, lo, cl = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    for j in range(t + 1, len(df)):
        dead = cl[j] < inv if bull else cl[j] > inv
        if dead:
            return "invalid", j
        if (hi[j] >= t1) if bull else (lo[j] <= t1):
            return "hit", j
    return None


def replay(tf: str) -> list[dict]:
    df = candles(tf)
    ts = df["timestamp"].to_numpy()
    setups: dict[tuple, dict] = {}
    for t in range(hm.MIN_BARS + 1, len(df) - 1):
        s = max(0, t + 1 - WINDOW)
        window = df.iloc[s:t + 2]
        found = hm.find(window, tf)
        start = max(s, t - hm.LOOKBACK + 1)               # absolute row of closed.iloc[0] inside find
        for h in found:
            if h.stage == "forming":
                continue
            key = (h.name, h.side, *(int(ts[start + p.index]) for p in h.points[:4]))
            snap = {"t": t, "t1": h.targets[0], "inv": h.invalidation, "valid": h.valid,
                    "div": bool(h.rsi_divergence), "stage": h.stage}
            if key not in setups:
                setups[key] = {"tf": tf, "name": h.name, "bull": h.side == "bullish", "entry": snap,
                               "div_snap": snap if snap["div"] else None}
            elif setups[key]["div_snap"] is None and snap["div"]:
                setups[key]["div_snap"] = snap
    out = []
    for k, st in setups.items():
        e = st["entry"]
        res = resolve(df, e["t"], st["bull"], e["t1"], e["inv"])
        if res is None:
            continue
        rec = {"tf": tf, "name": st["name"], "bull": st["bull"], "valid": e["valid"], "div_entry": e["div"],
               "hit": res[0] == "hit", "entry_stage": e["stage"]}
        d = st["div_snap"]
        rec["div_later"] = False
        rec["hit_wait"] = rec["hit"]
        if d is not None and d["t"] < res[1]:             # divergence showed while the setup was still open
            r2 = resolve(df, d["t"], st["bull"], d["t1"], d["inv"])
            if r2 is None:
                continue                                  # its waited-for trade is still open
            rec["div_later"] = True
            rec["hit_wait"] = r2[0] == "hit"
            rec["valid_wait"] = d["valid"]
        out.append(rec)
    return out


def line(label: str, rows: pd.DataFrame, col: str = "hit") -> str:
    n = len(rows)
    if not n:
        return f"  {label:<44} n=0"
    rate = rows[col].mean() * 100
    flag = "" if n >= 20 else "   (n<20: not significant)"
    return f"  {label:<44} n={n:<4} T1 first {rate:5.1f}%{flag}"


def main() -> None:
    recs = []
    for tf in ("1h", "4h", "1d"):
        df = candles(tf)
        print(f"{tf}: {len(df)} candles {pd.to_datetime(df.timestamp.iloc[0], unit='ms'):%Y-%m-%d} -> "
              f"{pd.to_datetime(df.timestamp.iloc[-1], unit='ms'):%Y-%m-%d}", flush=True)
        recs += replay(tf)
    r = pd.DataFrame(recs)
    for scope, rows in [("ALL TIMEFRAMES", r), *[(tf, r[r.tf == tf]) for tf in ("1h", "4h", "1d")]]:
        print(f"\n{scope}  (resolved setups: {len(rows)})")
        print("  -- flags at zone entry --")
        print(line("all setups", rows))
        print(line("RSI divergence at entry", rows[rows.div_entry]))
        print(line("no RSI divergence at entry", rows[~rows.div_entry]))
        print(line("level-valid", rows[rows.valid]))
        print(line("level-unconfirmed", rows[~rows.valid]))
        print(line("level-valid + RSI divergence", rows[rows.valid & rows.div_entry]))
        print("  -- wait for the divergence (re-entered on its candle) --")
        print(line("divergence showed while open", rows[rows.div_later], "hit_wait"))
        print(line("divergence never showed", rows[~rows.div_later], "hit_wait"))
        lv = rows[rows.div_later & rows.valid_wait.fillna(False).astype(bool)] if "valid_wait" in rows else rows[:0]
        print(line("level-valid at the divergence candle", lv, "hit_wait"))
    print("\nentry stage counts:", r.entry_stage.value_counts().to_dict())
    print("by pattern (at entry):", r.groupby("name").hit.agg(["count", "mean"]).round(2).to_dict("index"))


if __name__ == "__main__":
    main()
