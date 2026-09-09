import asyncio

import pandas as pd

from core.data import market
from core.data.types import FuturesSnapshot, OptionsSnapshot, SentimentSnapshot, SpotSnapshot


def test_fetch_all_isolates_provider_failures(monkeypatch):
    async def ok_spot(client):
        return SpotSnapshot(source="kraken", last=1.0, frames={"1h": pd.DataFrame()})

    async def boom(client):
        raise RuntimeError("network down")

    async def ok_opts(client):
        return OptionsSnapshot(source="deribit")

    async def ok_sent(client):
        return SentimentSnapshot(source="alternative.me", fear_greed=50)

    monkeypatch.setattr(market, "fetch_spot", ok_spot)
    monkeypatch.setattr(market, "fetch_futures", boom)
    monkeypatch.setattr(market, "fetch_options", ok_opts)
    monkeypatch.setattr(market, "fetch_sentiment", ok_sent)
    for name in ("fetch_hyperliquid", "fetch_coinlobster", "fetch_stablecoins", "fetch_calendar"):
        monkeypatch.setattr(market, name, boom)  # never touch the network in tests

    snap = asyncio.run(market.fetch_all())
    assert snap.spot.available
    assert snap.futures.available is False and "network down" in snap.futures.error
    assert snap.unavailable() == ["futures", "hyperliquid", "liquidations", "whales", "stablecoins", "calendar"]
