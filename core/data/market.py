"""Fetch every provider concurrently; a failing provider never takes the page down."""
from __future__ import annotations

import asyncio

import httpx

from core.data.binance_futures import fetch_futures
from core.data.deribit_options import fetch_options
from core.data.kraken_spot import fetch_spot
from core.data.sentiment import fetch_sentiment
from core.data.types import FuturesSnapshot, MarketSnapshot, OptionsSnapshot, SentimentSnapshot, SpotSnapshot

HEADERS = {"User-Agent": "trap-intel-dashboard/1.0"}
TIMEOUT = httpx.Timeout(12.0, connect=6.0)


async def _guard(coro, fallback_cls, source: str):
    try:
        return await coro
    except Exception as e:  # noqa: BLE001 - last line of defence
        return fallback_cls.unavailable(source, str(e))


async def fetch_all(client: httpx.AsyncClient | None = None) -> MarketSnapshot:
    own = client is None
    client = client or httpx.AsyncClient(timeout=TIMEOUT, headers=HEADERS)
    try:
        spot, fut, opt, sent = await asyncio.gather(
            _guard(fetch_spot(client), SpotSnapshot, "kraken"),
            _guard(fetch_futures(client), FuturesSnapshot, "binance,bybit"),
            _guard(fetch_options(client), OptionsSnapshot, "deribit"),
            _guard(fetch_sentiment(client), SentimentSnapshot, "alternative.me"),
        )
    finally:
        if own:
            await client.aclose()
    return MarketSnapshot(spot=spot, futures=fut, options=opt, sentiment=sent)


def load_market() -> MarketSnapshot:
    """Synchronous entry point for Streamlit (wrapped with st.cache_data in app.py)."""
    return asyncio.run(fetch_all())
