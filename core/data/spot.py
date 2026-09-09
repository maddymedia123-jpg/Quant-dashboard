"""Spot OHLCV chain: Kraken first, Gemini exchange second."""
from __future__ import annotations

import httpx

from core.data.gemini_spot import fetch_gemini_spot
from core.data.kraken_spot import fetch_spot as fetch_kraken
from core.data.types import SpotSnapshot


async def fetch_spot_chain(client: httpx.AsyncClient) -> SpotSnapshot:
    errors = []
    for fn in (fetch_kraken, fetch_gemini_spot):
        s = await fn(client)
        if s.available:
            return s
        errors.append(f"{s.source}: {s.error}")
    return SpotSnapshot.unavailable("kraken,gemini", " | ".join(errors))
