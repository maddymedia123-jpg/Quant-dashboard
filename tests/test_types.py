from datetime import datetime, timezone

import pandas as pd

from core.data.types import (
    FuturesSnapshot, MarketSnapshot, OptionsSnapshot, SentimentSnapshot, SpotSnapshot,
)


def test_unavailable_factory_sets_flags():
    f = FuturesSnapshot.unavailable("binance", "451 blocked")
    assert f.available is False
    assert f.source == "binance"
    assert f.error == "451 blocked"
    assert f.funding_rate is None


def test_spot_holds_dataframes():
    df = pd.DataFrame({"timestamp": [1, 2], "open": [1.0, 2.0], "high": [1, 2], "low": [1, 2], "close": [1, 2], "volume": [1, 1]})
    s = SpotSnapshot(source="kraken", last=2.0, frames={"1h": df}, fetched_at=datetime.now(timezone.utc))
    assert s.frames["1h"].shape == (2, 6)


def test_market_lists_unavailable_sources():
    m = MarketSnapshot(
        spot=SpotSnapshot(source="kraken"),
        futures=FuturesSnapshot.unavailable("binance", "x"),
        options=OptionsSnapshot(source="deribit"),
        sentiment=SentimentSnapshot.unavailable("alternative.me", "timeout"),
    )
    assert m.unavailable() == ["futures", "sentiment", "hyperliquid", "liquidations", "whales", "stablecoins", "calendar"]
