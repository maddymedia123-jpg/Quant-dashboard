"""BTC dominance: bitcoin's share of the whole crypto market, from CoinGecko's keyless /global endpoint.

Roadmap item 10 (owner's GO, 5 November): "BTC Dominance capital rotation". Dominance rising while the
market rises means money is going into bitcoin first; dominance falling while the market rises means it
is rotating out into the alternatives (risk appetite stretching); dominance rising while the market
falls means the alternatives are being sold harder. The level alone says little - the CHANGE is the
read, so the worker stores a sample every pass (core.derived.record_dominance) and the 24-hour and
7-day changes are computed from the dashboard's own history, labelled derived and never invented when
the history is too short.

Found by calling the endpoint: `data.market_cap_percentage` carries the top ten assets' shares in
percent (btc, eth, usdt, usdc, ...), `data.updated_at` is epoch SECONDS, and `data.total_market_cap`
is a dict per quote currency. The free tier allows a handful of calls a minute; one call every refresh
is far inside that."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from core.data.types import DominanceSnapshot

log = logging.getLogger(__name__)

URL = "https://api.coingecko.com/api/v3/global"
HEADERS = {"User-Agent": "trap-intel-dashboard/1.0"}
MAX_BYTES = 256 * 1024
STALE_MS = 2 * 3_600_000          # CoinGecko refreshes the global figures every few minutes
PLAUSIBLE = (20.0, 95.0)          # bitcoin's share has lived between 35% and 75% since 2017
STABLES = ("usdt", "usdc")
ROTATION_PP_24H = 0.5             # a day's share move that counts as rotation (percentage points)
ROTATION_PP_7D = 1.5              # a week's


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def parse_global(payload: dict) -> DominanceSnapshot:
    """The share figures from one /global response. Refuses a payload that is not CoinGecko's shape or a
    bitcoin share no market has ever printed (a decimal shift, a swapped field)."""
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        raise ValueError("coingecko: no 'data' object in the response")
    shares = data.get("market_cap_percentage") or {}
    if not isinstance(shares, dict) or shares.get("btc") is None:
        raise ValueError("coingecko: no btc share in market_cap_percentage")
    btc = float(shares["btc"])
    if not PLAUSIBLE[0] <= btc <= PLAUSIBLE[1]:
        raise ValueError(f"coingecko: btc share {btc:.1f}% is outside the plausible {PLAUSIBLE[0]:.0f}-{PLAUSIBLE[1]:.0f}% range")
    eth = shares.get("eth")
    stables = sum(float(shares.get(k) or 0.0) for k in STABLES)
    caps = data.get("total_market_cap") or {}
    total = caps.get("usd") if isinstance(caps, dict) else None
    change = data.get("market_cap_change_percentage_24h_usd")
    updated = data.get("updated_at")
    return DominanceSnapshot(
        source="coingecko",
        btc_pct=round(btc, 2),
        eth_pct=round(float(eth), 2) if eth is not None else None,
        stable_pct=round(stables, 2) if stables else None,
        total_mcap_usd=float(total) if total is not None else None,
        mcap_change_24h_pct=round(float(change), 2) if change is not None else None,
        as_of_ms=int(updated) * 1000 if updated else None,
    )


def is_stale(snap: DominanceSnapshot, now_ms: int) -> bool:
    return snap.as_of_ms is not None and now_ms - snap.as_of_ms > STALE_MS


async def fetch_dominance(client: httpx.AsyncClient) -> DominanceSnapshot:
    r = await client.get(URL, headers=HEADERS, follow_redirects=True)
    r.raise_for_status()
    if len(r.content) > MAX_BYTES:
        raise ValueError(f"coingecko: {len(r.content) / 1024:.0f} KB for the global figures")
    return parse_global(r.json())


def rotation(snap: DominanceSnapshot) -> dict:
    """Where capital is going, from the share's change: {"state", "note"}. The 24-hour change decides when
    the history has it; the 7-day change otherwise; with neither the state is unknown, not flat."""
    if not getattr(snap, "available", False) or snap.btc_pct is None:
        return {"state": "unknown", "note": "no dominance feed"}
    day, week = snap.change_24h_pp, snap.change_7d_pp
    if day is not None:
        move, span, bar = day, "24h", ROTATION_PP_24H
    elif week is not None:
        move, span, bar = week, "7d", ROTATION_PP_7D
    else:
        return {"state": "unknown",
                "note": f"bitcoin holds {snap.btc_pct:.1f}% of the market; no change yet (history too short)"}
    mcap = snap.mcap_change_24h_pct
    market = ("" if mcap is None else
              f" with the whole market {'up' if mcap >= 0 else 'down'} {abs(mcap):.1f}% on the day")
    if move >= bar:
        return {"state": "into_btc",
                "note": (f"bitcoin's share rose {move:+.1f} pp over {span} to {snap.btc_pct:.1f}%{market}: "
                         "capital is rotating into bitcoin, the alternatives are lagging")}
    if move <= -bar:
        return {"state": "into_alts",
                "note": (f"bitcoin's share fell {move:+.1f} pp over {span} to {snap.btc_pct:.1f}%{market}: "
                         "capital is rotating out to the alternatives"
                         + (" (risk appetite stretching)" if mcap is not None and mcap > 0 else ""))}
    return {"state": "flat",
            "note": f"bitcoin's share is steady at {snap.btc_pct:.1f}% ({move:+.1f} pp over {span}){market}: no rotation"}
