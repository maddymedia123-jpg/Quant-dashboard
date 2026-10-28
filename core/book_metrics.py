"""Scoop and sweep, read off the recorded order book (core.book_store) - Hammad's "Scoop & Sweep
Scaling: order book scoop metrics to measure liquidity absorption rates and sweep vectors".

His prompt names the two ideas; the measurements below are ours, each a documented constant, provisional
until a few weeks of book have been recorded and the thresholds can be tuned against what followed.
Nothing here gates a plan or sends an alert yet: the report goes to the drawer and the JSON.

* A SWEEP: within SWEEP_WINDOW_MS, mid moves at least SWEEP_MIN_BPS, and the resting size that stood in
  the range it moved through (asks on the way up, bids on the way down) at the start was at least
  SWEEP_MIN_BTC. That size is reported as cleared. Whether it was eaten or pulled needs the trade tape,
  which is not recorded; the report says "cleared", not "filled".
* A SCOOP (absorption): a wall - one level holding at least WALL_X times the median level on its side
  within WALL_SCAN_BPS of mid, and at least WALL_MIN_BTC - that price tests (mid within TEST_BPS of it)
  for at least SCOOP_MIN_MINUTES of the minute-by-minute record while the wall keeps at least WALL_HOLD of
  its largest size and price does not go through it. Its outcome is tracked: still holding, held (price
  left it by LEFT_BPS with the wall intact), pulled (the wall went before price reached through it), or
  swept (price went through).

Everything is computed on the minute-by-minute record (the first snapshot of each minute) over the last
WINDOW_MS, so a pass costs a few hundred rows, not thousands."""
from __future__ import annotations

import statistics

from core.book_store import BookStore

MINUTE_MS = 60_000
WINDOW_MS = 60 * MINUTE_MS
STALE_MS = 10 * MINUTE_MS         # no snapshot this recent: the collector is down, say so
SWEEP_WINDOW_MS = MINUTE_MS
SWEEP_MIN_BPS = 5.0
SWEEP_MIN_BTC = 5.0
WALL_SCAN_BPS = 30.0
WALL_X = 8.0
WALL_MIN_BTC = 10.0
WALL_HOLD = 0.6
TEST_BPS = 3.0
LEFT_BPS = 10.0
SCOOP_MIN_MINUTES = 2
PRICE_KEY = 1                     # prices matched to this many decimals across snapshots


def _bps(a: float, b: float) -> float:
    return (b - a) / a * 10_000


def _rows(rows) -> list[dict]:
    """Store rows as they are; BookSnapshot objects as the same four fields."""
    return [r if isinstance(r, dict) else {"ts_ms": r.ts_ms, "mid": r.mid, "bids": r.bids, "asks": r.asks} for r in rows]


def sweeps(rows: list) -> list[dict]:
    """Sweep events across consecutive minute rows (see the module docstring)."""
    rows = _rows(rows)
    out: list[dict] = []
    for a, b in zip(rows, rows[1:]):
        if b["ts_ms"] - a["ts_ms"] > 2 * SWEEP_WINDOW_MS:
            continue
        move = _bps(a["mid"], b["mid"])
        if abs(move) < SWEEP_MIN_BPS:
            continue
        lo, hi = sorted((a["mid"], b["mid"]))
        side = "asks" if move > 0 else "bids"
        cleared = sum(q for p, q in a[side] if lo <= p <= hi)
        if cleared < SWEEP_MIN_BTC:
            continue
        out.append({"t_ms": b["ts_ms"], "side": side, "direction": "up" if move > 0 else "down", "from": round(a["mid"], 1),
                    "to": round(b["mid"], 1), "move_bps": round(move, 1), "cleared_btc": round(cleared, 2),
                    "seconds": round((b["ts_ms"] - a["ts_ms"]) / 1000)})
    return out


def _walls(row: dict) -> list[dict]:
    """The walls standing in one snapshot."""
    mid = row["mid"]
    lim = mid * WALL_SCAN_BPS / 10_000
    out = []
    for side in ("bids", "asks"):
        near = [(p, q) for p, q in row[side] if abs(p - mid) <= lim]
        if len(near) < 3:
            continue
        med = statistics.median(q for _, q in near)
        for p, q in near:
            if q >= WALL_MIN_BTC and (med <= 0 or q >= WALL_X * med):
                out.append({"side": side, "price": round(p, PRICE_KEY), "btc": q})
    return out


def _size_at(row: dict, side: str, price: float) -> float:
    return sum(q for p, q in row[side] if round(p, PRICE_KEY) == price)


def scoops(rows: list) -> list[dict]:
    """Absorption events: walls tested by price, and how each one ended (see the module docstring)."""
    rows = _rows(rows)
    tracked: dict[tuple[str, float], dict] = {}
    done: list[dict] = []
    for row in rows:
        mid = row["mid"]
        for w in _walls(row):
            key = (w["side"], w["price"])
            if key not in tracked:
                tracked[key] = {"side": w["side"], "price": w["price"], "first_ms": row["ts_ms"], "max_btc": w["btc"],
                                "tested_minutes": 0, "tested": False, "outcome": "holding"}
            tracked[key]["max_btc"] = max(tracked[key]["max_btc"], w["btc"])
        for key, t in list(tracked.items()):
            size = _size_at(row, t["side"], t["price"])
            through = (mid < t["price"]) if t["side"] == "bids" else (mid > t["price"])
            testing = abs(_bps(t["price"], mid)) <= TEST_BPS
            t["btc"] = round(size, 2)
            t["last_ms"] = row["ts_ms"]
            if through:
                t["outcome"] = "swept"
            elif size < WALL_HOLD * t["max_btc"]:
                t["outcome"] = "pulled" if not t["tested"] else "pulled after test"
            elif testing:
                t["tested"] = True
                t["tested_minutes"] += 1
                t["outcome"] = "holding"
            elif t["tested"] and abs(_bps(t["price"], mid)) >= LEFT_BPS:
                t["outcome"] = "held"
            if t["outcome"] in ("swept", "pulled", "pulled after test", "held"):
                done.append(tracked.pop(key))
    events = [t for t in [*done, *tracked.values()] if t["tested"] and t["tested_minutes"] >= SCOOP_MIN_MINUTES]
    events.sort(key=lambda t: t["first_ms"])
    return [{"side": "bids" if e["side"] == "bids" else "asks", "price": e["price"], "wall_btc": round(e["max_btc"], 2),
             "now_btc": e["btc"], "tested_minutes": e["tested_minutes"], "outcome": e["outcome"],
             "first_ms": e["first_ms"], "last_ms": e["last_ms"]} for e in events]


def report(now_ms: int, book: BookStore | None = None) -> dict:
    """The last hour of the book as one JSON-safe dict for the main store: the depth now, the sweeps and
    the scoops, and how much book has been recorded. {"available": False, "reason": ...} with no
    collector running here."""
    try:
        bk = book or BookStore(readonly=True)
    except Exception as e:  # noqa: BLE001 - no file, no permission: no recorder runs here, which is normal off the VM
        return {"available": False, "recorder": "absent", "reason": f"order book store unavailable: {type(e).__name__}: {e}"}
    try:
        latest = bk.latest()
        if latest is None:
            return {"available": False, "recorder": "empty", "reason": "no order book recorded yet"}
        if now_ms - latest["ts_ms"] > STALE_MS:
            return {"available": False, "recorder": "stale", "last_ms": latest["ts_ms"],
                    "reason": f"last order book snapshot {int((now_ms - latest['ts_ms']) / MINUTE_MS)} min ago"}
        rows = bk.between(now_ms - WINDOW_MS, now_ms + 1, step_ms=MINUTE_MS)
        st = bk.stats()
    finally:
        if book is None:
            bk.close()
    depth = {k: latest[k] for k in latest if k.startswith(("bid_", "ask_")) or k in ("mid", "best_bid", "best_ask", "spread_bps", "imbalance_025")}
    return {"available": True, "as_of_ms": latest["ts_ms"], "depth": depth, "sweeps": sweeps(rows)[-12:],
            "scoops": scoops(rows)[-12:], "minutes_in_window": len(rows),
            "recorded": {"rows": st["rows"], "since_ms": st["first_ms"], "mb": round(st["bytes"] / 1e6, 1)},
            "method": "sweep: mid moved >= %.0f bps within a minute through >= %.0f BTC of resting size (cleared, "
                      "not necessarily filled); scoop: a wall >= %.0fx its side's median level and >= %.0f BTC, "
                      "tested within %.0f bps for >= %d minutes and still >= %.0f%% of its size. Provisional."
                      % (SWEEP_MIN_BPS, SWEEP_MIN_BTC, WALL_X, WALL_MIN_BTC, TEST_BPS, SCOOP_MIN_MINUTES, WALL_HOLD * 100)}
