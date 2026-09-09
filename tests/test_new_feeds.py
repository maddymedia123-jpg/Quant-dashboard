import asyncio
import json
import pathlib

import numpy as np
import pandas as pd

from core.data.coinlobster import parse_liquidations, parse_radar, parse_whales
from core.data.gemini_spot import parse_candles, resample
from core.data.hyperliquid import parse_meta
from core.data.stablecoins import parse_stablecoins
from core.data.types import (
    FuturesSnapshot, MarketSnapshot, OptionsSnapshot, SentimentSnapshot, SpotSnapshot,
)

FX = pathlib.Path(__file__).parent / "fixtures"


def _j(n):
    return json.loads((FX / n).read_text(encoding="utf-8"))


def test_market_defaults_new_feeds_to_unavailable():
    m = MarketSnapshot(spot=SpotSnapshot(source="kraken"), futures=FuturesSnapshot(source="binance"),
                       options=OptionsSnapshot(source="deribit"), sentiment=SentimentSnapshot(source="alternative.me"))
    assert m.unavailable() == ["hyperliquid", "liquidations", "whales", "stablecoins", "calendar"]
    assert m.liquidations.error == "not fetched"


def test_gemini_parse_and_resample():
    df = parse_candles(_j("gemini_1hr.json"))
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert df["timestamp"].is_monotonic_increasing and df["timestamp"].dtype == np.int64
    h4 = resample(df, 240)
    assert len(h4) < len(df) and (h4["high"] >= h4["low"]).all()
    assert (h4["timestamp"] % (240 * 60_000) == 0).all()
    d = parse_candles(_j("gemini_1day.json"))
    w = resample(d, 10080)
    assert all(pd.Timestamp(int(t), unit="ms", tz="UTC").weekday() == 0 for t in w["timestamp"])
    assert w["volume"].iloc[-1] > 0


def test_spot_chain_falls_back_to_gemini(monkeypatch):
    from core.data import spot

    async def k(client):
        return SpotSnapshot.unavailable("kraken", "dns")

    async def g(client):
        return SpotSnapshot(source="gemini", last=1.0, frames={"1h": pd.DataFrame()})

    monkeypatch.setattr(spot, "fetch_kraken", k)
    monkeypatch.setattr(spot, "fetch_gemini_spot", g)
    s = asyncio.run(spot.fetch_spot_chain(None))
    assert s.available and s.source == "gemini"


def test_hyperliquid_parse():
    h = parse_meta(_j("hyperliquid_meta.json"))
    assert h.available and h.source == "hyperliquid"
    assert -0.01 < h.funding_rate_1h < 0.01 and h.open_interest > 100 and h.mark_price > 1000


def test_coinlobster_liquidations_parse():
    l = parse_liquidations(_j("coinlobster_liquidations.json"))
    assert l.available and l.window == "24h"
    assert l.total_usd > 1e6 and abs(l.long_usd + l.short_usd - l.total_usd) / l.total_usd < 0.02
    assert l.btc_usd and l.btc_long_usd is not None and len(l.hourly) == 24 and l.venues


def test_coinlobster_whales_parse_with_radar():
    w = parse_whales(_j("coinlobster_whales.json"), _j("coinlobster_radar.json"))
    assert w.available and w.btc_trades > 0 and w.btc_buy_usd >= 0 and w.btc_sell_usd >= 0
    assert "Binance Futures" in w.funding_by_exchange and w.oi_by_exchange_usd
    assert w.radar and {"coin", "direction", "unusual"} <= set(w.radar[0])
    assert parse_radar(_j("coinlobster_radar.json"))[0]["coin"]


def test_stablecoins_parse():
    s = parse_stablecoins(_j("defillama_stablecoins.json"))
    assert s.available and s.total_usd > 1e11 and s.prev_day_usd > 1e11
    assert s.change_24h_pct is not None and len(s.top) == 5 and s.top[0]["usd"] >= s.top[1]["usd"]


def test_fetch_context_isolates_and_assemble(monkeypatch):
    from core.data import market
    from core.data.types import LiquidationsSnapshot, WhalesSnapshot

    async def boom(client):
        raise RuntimeError("down")

    async def ok_cl(client):
        return (LiquidationsSnapshot(source="coinlobster", total_usd=1.0, long_usd=0.5, short_usd=0.5),
                WhalesSnapshot(source="coinlobster"))

    for name in ("fetch_futures", "fetch_options", "fetch_sentiment", "fetch_hyperliquid", "fetch_stablecoins", "fetch_calendar"):
        monkeypatch.setattr(market, name, boom)
    monkeypatch.setattr(market, "fetch_coinlobster", ok_cl)
    ctx = asyncio.run(market.fetch_context(client=object()))
    m = market.assemble(SpotSnapshot(source="kraken"), ctx)
    assert m.liquidations.available and m.whales.available
    assert set(m.unavailable()) == {"futures", "options", "sentiment", "hyperliquid", "stablecoins", "calendar"}


def test_offline_fixture_market_has_all_sources():
    from core.data.offline import fixture_market
    m = fixture_market()
    assert m.unavailable() == []
    assert all(getattr(m, n).source == "fixture" for n in MarketSnapshot.SOURCES)


def test_compose_futures_from_coinlobster_and_hyperliquid():
    from core.data.binance_futures import compose_futures
    from core.data.types import HyperliquidSnapshot
    w = parse_whales(_j("coinlobster_whales.json"), None)
    h = parse_meta(_j("hyperliquid_meta.json"))
    f = compose_futures(w, h, "451")
    assert f.available and f.source == "coinlobster+hyperliquid (Binance Futures)"
    assert f.funding_rate == w.funding_by_exchange["Binance Futures"] and f.open_interest_usd > 1e8
    assert f.open_interest > 1000 and f.mark_price == h.mark_price and f.oi_change_24h_pct is None
    only_hl = compose_futures(None, h, "451")
    assert only_hl.available and only_hl.source.startswith("hyperliquid") and abs(only_hl.funding_rate - h.funding_rate_1h * 8) < 1e-12
    none = compose_futures(None, HyperliquidSnapshot.unavailable("hyperliquid", "x"), "451")
    assert none.available is False


def test_fetch_context_composes_when_direct_futures_blocked(monkeypatch):
    from core.data import market
    from core.data.types import HyperliquidSnapshot, LiquidationsSnapshot, WhalesSnapshot

    async def blocked(client):
        return FuturesSnapshot.unavailable("binance,bybit", "451")

    async def boom(client):
        raise RuntimeError("down")

    async def ok_hl(client):
        return HyperliquidSnapshot(source="hyperliquid", funding_rate_1h=0.00001, open_interest=30000.0, mark_price=78000.0, oracle_price=78000.0)

    async def ok_cl(client):
        return (LiquidationsSnapshot(source="coinlobster", total_usd=1.0, long_usd=0.5, short_usd=0.5),
                WhalesSnapshot(source="coinlobster", funding_by_exchange={"Binance Futures": 0.0001}, oi_by_exchange_usd={"Binance Futures": 8e9}))

    monkeypatch.setattr(market, "fetch_futures", blocked)
    for name in ("fetch_options", "fetch_sentiment", "fetch_stablecoins", "fetch_calendar"):
        monkeypatch.setattr(market, name, boom)
    monkeypatch.setattr(market, "fetch_hyperliquid", ok_hl)
    monkeypatch.setattr(market, "fetch_coinlobster", ok_cl)
    ctx = asyncio.run(market.fetch_context(client=object()))
    assert ctx["futures"].available and ctx["futures"].funding_rate == 0.0001 and "coinlobster" in ctx["futures"].source
