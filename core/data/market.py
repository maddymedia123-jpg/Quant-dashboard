"""Fetch providers concurrently; a failing provider never takes the page down.

Spot is fast (25 s cache); context feeds are slow (120 s cache) because several are IP-rate-limited."""
from __future__ import annotations

import asyncio
import os

import httpx

from core.data.binance_futures import fetch_futures
from core.data.coinlobster import fetch_coinlobster
from core.data.deribit_options import fetch_options
from core.data.hyperliquid import fetch_hyperliquid
from core.data.sentiment import fetch_sentiment
from core.data.spot import fetch_spot_chain as fetch_spot
from core.data.stablecoins import fetch_stablecoins
from core.data.types import (
    FuturesSnapshot, HyperliquidSnapshot, LiquidationsSnapshot, MarketSnapshot, OptionsSnapshot,
    SentimentSnapshot, SpotSnapshot, StablecoinSnapshot, WhalesSnapshot,
)

HEADERS = {"User-Agent": "trap-intel-dashboard/1.0"}
TIMEOUT = httpx.Timeout(12.0, connect=6.0)


def _offline() -> bool:
    return os.environ.get("TI_OFFLINE_FIXTURES") == "1"


async def _guard(coro, fallback_cls, source: str):
    try:
        return await coro
    except Exception as e:  # noqa: BLE001 - last line of defence
        return fallback_cls.unavailable(source, str(e))


async def _guard_cl(coro):
    try:
        return await coro
    except Exception as e:  # noqa: BLE001
        return LiquidationsSnapshot.unavailable("coinlobster", str(e)), WhalesSnapshot.unavailable("coinlobster", str(e))


async def _with_client(fn):
    async with httpx.AsyncClient(timeout=TIMEOUT, headers=HEADERS) as client:
        return await fn(client)


async def fetch_spot_only(client: httpx.AsyncClient | None = None) -> SpotSnapshot:
    if client is None:
        return await _with_client(fetch_spot_only)
    return await _guard(fetch_spot(client), SpotSnapshot, "kraken,gemini")


async def fetch_context(client: httpx.AsyncClient | None = None) -> dict:
    if client is None:
        return await _with_client(fetch_context)
    fut, opt, sent, hl, cl, stables = await asyncio.gather(
        _guard(fetch_futures(client), FuturesSnapshot, "binance,bybit"),
        _guard(fetch_options(client), OptionsSnapshot, "deribit"),
        _guard(fetch_sentiment(client), SentimentSnapshot, "alternative.me"),
        _guard(fetch_hyperliquid(client), HyperliquidSnapshot, "hyperliquid"),
        _guard_cl(fetch_coinlobster(client)),
        _guard(fetch_stablecoins(client), StablecoinSnapshot, "defillama"),
    )
    liqs, whales = cl
    return {"futures": fut, "options": opt, "sentiment": sent, "hyperliquid": hl,
            "liquidations": liqs, "whales": whales, "stablecoins": stables}


async def fetch_all(client: httpx.AsyncClient | None = None) -> MarketSnapshot:
    if client is None:
        return await _with_client(fetch_all)
    spot, ctx = await asyncio.gather(fetch_spot_only(client), fetch_context(client))
    return assemble(spot, ctx)


def assemble(spot: SpotSnapshot, ctx: dict) -> MarketSnapshot:
    return MarketSnapshot(spot=spot, **ctx)


def load_spot() -> SpotSnapshot:
    if _offline():
        from core.data.offline import fixture_spot
        return fixture_spot()
    return asyncio.run(fetch_spot_only())


def load_context() -> dict:
    if _offline():
        from core.data.offline import fixture_context
        return fixture_context()
    return asyncio.run(fetch_context())


def load_market() -> MarketSnapshot:
    """Synchronous entry point (scripts/tests). TI_OFFLINE_FIXTURES=1 serves recorded fixtures."""
    return assemble(load_spot(), load_context())
