"""DefiLlama stablecoin supply (dry powder)."""
from __future__ import annotations

import logging

import httpx

from core.data.types import StablecoinSnapshot

log = logging.getLogger(__name__)
URL = "https://stablecoins.llama.fi/stablecoins"


def _usd(a: dict, key: str) -> float:
    return float(((a.get(key) or {}).get("peggedUSD")) or 0.0)


def parse_stablecoins(d: dict) -> StablecoinSnapshot:
    assets = d.get("peggedAssets", [])
    total = sum(_usd(a, "circulating") for a in assets)
    prev = sum(_usd(a, "circulatingPrevDay") for a in assets)
    top = sorted(({"name": a.get("symbol") or a.get("name"), "usd": _usd(a, "circulating")} for a in assets), key=lambda x: -x["usd"])[:5]
    return StablecoinSnapshot(source="defillama", total_usd=total, prev_day_usd=prev,
                              change_24h_pct=((total - prev) / prev * 100.0) if prev else None, top=top)


async def fetch_stablecoins(client: httpx.AsyncClient) -> StablecoinSnapshot:
    try:
        r = await client.get(URL, params={"includePrices": "false"})
        r.raise_for_status()
        return parse_stablecoins(r.json())
    except Exception as e:  # noqa: BLE001 - provider boundary
        log.error("defillama unavailable: %s", e)
        return StablecoinSnapshot.unavailable("defillama", str(e))
