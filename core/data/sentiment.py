"""Crypto Fear & Greed index (alternative.me, keyless)."""
from __future__ import annotations

import logging

import httpx

from core.data.types import SentimentSnapshot

log = logging.getLogger(__name__)
URL = "https://api.alternative.me/fng/"


def parse_fng(payload: dict) -> SentimentSnapshot:
    data = payload["data"]
    cur = data[0]
    prev = data[1] if len(data) > 1 else None
    return SentimentSnapshot(
        source="alternative.me",
        fear_greed=int(cur["value"]),
        classification=cur["value_classification"],
        fear_greed_prev=int(prev["value"]) if prev else None,
    )


async def fetch_sentiment(client: httpx.AsyncClient) -> SentimentSnapshot:
    try:
        r = await client.get(URL, params={"limit": 2})
        r.raise_for_status()
        return parse_fng(r.json())
    except Exception as e:  # noqa: BLE001 - provider boundary
        log.error("fear&greed unavailable: %s", e)
        return SentimentSnapshot.unavailable("alternative.me", str(e))
