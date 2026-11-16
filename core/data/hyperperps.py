"""Resting liquidation clusters on Hyperliquid, from the HyperPerps public heatmap (no key, no account).

Why this feed: Hyperliquid positions are on-chain, so these liquidation prices are observed, not modelled
from open interest and an assumed leverage mix the way the heatmap vendors do it. The price of that is
coverage: one venue, a slice of the market, and every message that uses it says so by name.

What the engine does with it (17 Nov 2026): the clusters are hazards for the Active Trade stop
(core.trades._hazards: a stop just above $23M of longs is taken by the flush that forces them out) and the
positioning family's magnet read (core.confluence: price is drawn to the bigger cluster within reach of the
zone). Not a cascade predictor: the branching-ratio model Hammad's note cites was eliminated by its own
authors (arXiv 2608.03616), so nothing here claims to forecast a cascade.
"""
from __future__ import annotations

import httpx

from core.data.types import LiquidationCluster, LiquidationLevelsSnapshot

URL = "https://trade.hyperperps.app/api/public/heatmap/BTC"
VENUE = "Hyperliquid"


def _cluster(row: dict, spot: float | None, long: bool) -> LiquidationCluster | None:
    """One cluster, or None when it has no price, no size, or sits on the wrong side of the price (a long
    liquidates below the price, a short above it)."""
    try:
        price = float(row["price"])
        notional = float(row.get("notional_usd") or 0)
    except (KeyError, TypeError, ValueError):
        return None
    if price <= 0 or notional <= 0:
        return None
    if spot is not None and ((price >= spot) if long else (price <= spot)):
        return None
    dist = row.get("distance_pct")
    try:
        wallets = int(row.get("wallet_count") or 0)
    except (TypeError, ValueError):
        wallets = 0
    return LiquidationCluster(price=price, notional_usd=notional, wallets=wallets,
                              distance_pct=float(dist) if dist is not None else None)


def parse_heatmap(d: dict) -> LiquidationLevelsSnapshot:
    liq = d["liquidations"]            # a body without it is malformed: raise, and the market guard names the feed unavailable
    spot = d.get("spot_at_compute")
    spot = float(spot) if spot is not None else None
    longs = [c for c in (_cluster(r, spot, True) for r in liq.get("longs") or []) if c is not None]
    shorts = [c for c in (_cluster(r, spot, False) for r in liq.get("shorts") or []) if c is not None]
    return LiquidationLevelsSnapshot(
        source="hyperperps", venue=VENUE, spot=spot,
        as_of_ms=int(d["updated_at"]) if d.get("updated_at") is not None else None,
        sample_size=int(d["sample_size"]) if d.get("sample_size") is not None else None,
        longs=sorted(longs, key=lambda c: -c.price),       # nearest to the price first
        shorts=sorted(shorts, key=lambda c: c.price),
    )


async def fetch_hyperperps(client: httpx.AsyncClient) -> LiquidationLevelsSnapshot:
    r = await client.get(URL, timeout=20)
    r.raise_for_status()
    return parse_heatmap(r.json())
