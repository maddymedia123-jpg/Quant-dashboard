"""Hyperliquid public info: BTC perp funding (hourly), open interest, mark/oracle."""
from __future__ import annotations

import logging

import httpx

from core.data.types import HyperliquidSnapshot

log = logging.getLogger(__name__)
URL = "https://api.hyperliquid.xyz/info"


def parse_meta(payload: list) -> HyperliquidSnapshot:
    meta, ctxs = payload
    names = [u["name"] for u in meta["universe"]]
    i = names.index("BTC")
    c = ctxs[i]
    return HyperliquidSnapshot(source="hyperliquid", funding_rate_1h=float(c["funding"]), open_interest=float(c["openInterest"]),
                               mark_price=float(c["markPx"]), oracle_price=float(c["oraclePx"]))


async def fetch_hyperliquid(client: httpx.AsyncClient) -> HyperliquidSnapshot:
    try:
        r = await client.post(URL, json={"type": "metaAndAssetCtxs"})
        r.raise_for_status()
        return parse_meta(r.json())
    except Exception as e:  # noqa: BLE001 - provider boundary
        log.error("hyperliquid unavailable: %s", e)
        return HyperliquidSnapshot.unavailable("hyperliquid", str(e))
