"""Gate.io's BTC_USDT perpetual order book: one snapshot of resting bids and asks.

Hammad's "scoop and sweep" metrics (the Quant Forecast's master prompt, section 2) are statements about
the order book - resting liquidity absorbing a hit, or being run through. Exchanges serve the book only as
it stands now, never its history, so the book has to be recorded as it happens (book_collector.py, every
POLL_S seconds, into core.book_store) before anything can be measured on it.

Sizes arrive in CONTRACTS; a BTC_USDT contract is `quanto_multiplier` bitcoin (0.0001), read from the
contract itself once per collector start, so a size is never shown as bitcoin without that conversion.
`summary` turns a snapshot into the handful of numbers kept on every row: mid, best bid and ask, the
spread, and the bitcoin resting within DEPTH_BPS of mid on each side."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx

from core.data.taker_flow import BASE, CONTRACT, DEFAULT_MULTIPLIER, HEADERS, MAX_BYTES, _num

log = logging.getLogger(__name__)
LEVELS = 50                       # levels fetched per side
KEEP_LEVELS = 25                  # ...and kept on the row (the rest is summarised into depth)
DEPTH_BPS = (10, 25, 50, 100)     # resting size within this many basis points of mid, per side


@dataclass
class BookSnapshot:
    ts_ms: int
    book_id: int
    bids: list[tuple[float, float]] = field(default_factory=list)   # (price, btc), best first
    asks: list[tuple[float, float]] = field(default_factory=list)   # (price, btc), best first

    @property
    def mid(self) -> float | None:
        if not self.bids or not self.asks:
            return None
        return (self.bids[0][0] + self.asks[0][0]) / 2


def parse_book(payload: dict, multiplier: float, ts_ms: int | None = None) -> BookSnapshot:
    """Gate.io's `/order_book` payload (`asks`/`bids` as [{"p": price, "s": contracts}], best first) in
    bitcoin. `ts_ms` defaults to the exchange's own `current` time."""
    def side(rows) -> list[tuple[float, float]]:
        out = []
        for r in rows or []:
            p, s = _num(r.get("p")), _num(r.get("s"))
            if p and s is not None and s >= 0:
                out.append((float(p), float(s) * multiplier))
        return out
    bids, asks = side(payload.get("bids")), side(payload.get("asks"))
    bids.sort(key=lambda x: -x[0])
    asks.sort(key=lambda x: x[0])
    t = ts_ms if ts_ms is not None else int(float(payload.get("current") or 0) * 1000)
    return BookSnapshot(ts_ms=int(t), book_id=int(payload.get("id") or 0), bids=bids, asks=asks)


def depth_within(levels: list[tuple[float, float]], mid: float, bps: float) -> float:
    """Bitcoin resting within `bps` of mid on one side."""
    lim = mid * bps / 10_000
    return float(sum(s for p, s in levels if abs(p - mid) <= lim))


def summary(snap: BookSnapshot) -> dict:
    """The numbers kept on every stored row: mid, best bid/ask, spread in bps, depth per side per band, and
    the 25 bps imbalance (bids minus asks over their sum: +1 all bids, -1 all asks)."""
    mid = snap.mid
    if mid is None:
        return {"mid": None}
    out = {"mid": round(mid, 2), "best_bid": snap.bids[0][0], "best_ask": snap.asks[0][0],
           "spread_bps": round((snap.asks[0][0] - snap.bids[0][0]) / mid * 10_000, 3)}
    for b in DEPTH_BPS:
        out[f"bid_{b:03d}"] = round(depth_within(snap.bids, mid, b), 4)
        out[f"ask_{b:03d}"] = round(depth_within(snap.asks, mid, b), 4)
    tot = out["bid_025"] + out["ask_025"]
    out["imbalance_025"] = round((out["bid_025"] - out["ask_025"]) / tot, 4) if tot > 0 else 0.0
    return out


async def fetch_multiplier(client: httpx.AsyncClient) -> float:
    try:
        r = await client.get(f"{BASE}/contracts/{CONTRACT}", headers=HEADERS, follow_redirects=True)
        r.raise_for_status()
        mult = _num(r.json().get("quanto_multiplier"))
        if mult:
            return float(mult)
    except Exception as e:  # noqa: BLE001 - provider boundary
        log.warning("Gate.io contract size unavailable (%s); using %s", e, DEFAULT_MULTIPLIER)
    return DEFAULT_MULTIPLIER


async def fetch_order_book(client: httpx.AsyncClient, multiplier: float) -> BookSnapshot:
    """One snapshot, LEVELS deep each side. Raises on any failure: the collector decides how to retry."""
    r = await client.get(f"{BASE}/order_book", params={"contract": CONTRACT, "limit": LEVELS, "with_id": "true"},
                         headers=HEADERS, follow_redirects=True)
    r.raise_for_status()
    if len(r.content) > MAX_BYTES:
        raise ValueError(f"{len(r.content) / 1024:.0f} KB from /order_book")
    snap = parse_book(r.json(), multiplier)
    if snap.mid is None:
        raise ValueError("empty order book")
    return snap
