"""Resting liquidation clusters from Hyperliquid (the HyperPerps public heatmap, no key).

Why this feed and not a heatmap vendor: Hyperliquid positions are on-chain, so these liquidation prices
are observed, not modelled from open interest and assumed leverage tiers; the price of that is coverage,
one venue. The fixture is the live response recorded on 16 Nov 2026 21:18 UTC.
"""
import asyncio
import json
import pathlib

import httpx
import pytest

from core.data import hyperperps
from core.data.types import LiquidationLevelsSnapshot, MarketSnapshot

FX = pathlib.Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FX / name).read_text(encoding="utf-8"))


def _run(coro):
    return asyncio.run(coro)


def test_the_recorded_heatmap_parses_into_clusters_on_the_right_side_of_spot():
    s = hyperperps.parse_heatmap(load("hyperperps_heatmap_btc.json"))
    assert s.available and s.source == "hyperperps" and s.venue == "Hyperliquid"
    assert s.spot == 64_940.5 and s.as_of_ms == 1_794_863_881_888 and s.sample_size == 18_207
    assert len(s.longs) == 10 and len(s.shorts) == 10
    assert all(c.price < s.spot for c in s.longs), "longs liquidate below the price"
    assert all(c.price > s.spot for c in s.shorts), "shorts liquidate above it"
    by_price = {c.price: c for c in s.longs}
    c = by_price[63_966.39]
    assert (c.notional_usd, c.wallets, c.distance_pct) == (23_113_326.0, 74, -1.5)
    assert s.longs[0].distance_pct == -1.0, "the feed's own order is not kept: nearest to the price first"
    assert [c.price for c in s.longs] == sorted((c.price for c in s.longs), reverse=True), "nearest to price first"
    assert [c.price for c in s.shorts] == sorted(c.price for c in s.shorts)
    assert round(sum(c.notional_usd for c in s.longs)) == 90_448_273


def test_a_cluster_on_the_wrong_side_of_spot_or_without_a_size_is_dropped():
    d = load("hyperperps_heatmap_btc.json")
    d["liquidations"]["longs"] = [{"price": 70_000.0, "notional_usd": 5_000_000},      # above spot: not a long liquidation
                                  {"price": 60_000.0, "notional_usd": 0},              # nothing there
                                  {"price": "junk", "notional_usd": 9_000_000},
                                  {"price": 61_000.0, "notional_usd": 4_000_000, "wallet_count": 3}]
    s = hyperperps.parse_heatmap(d)
    assert [(c.price, c.wallets) for c in s.longs] == [(61_000.0, 3)]
    assert len(s.shorts) == 10, "the other side is untouched"


def test_a_failing_or_malformed_feed_raises_so_the_market_guard_names_it_unavailable():
    with pytest.raises(httpx.HTTPStatusError):
        _run(hyperperps.fetch_hyperperps(httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500)))))
    with pytest.raises((KeyError, TypeError, ValueError)):
        _run(hyperperps.fetch_hyperperps(httpx.AsyncClient(transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"something": "else"})))))


def test_the_request_hits_the_public_btc_endpoint_without_a_key():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["auth"] = any(k.lower() in ("authorization", "x-api-key") for k in request.headers)
        return httpx.Response(200, json=load("hyperperps_heatmap_btc.json"))
    s = _run(hyperperps.fetch_hyperperps(httpx.AsyncClient(transport=httpx.MockTransport(handler))))
    assert seen["url"] == hyperperps.URL and "hyperperps.app" in seen["url"] and seen["url"].endswith("/BTC")
    assert not seen["auth"] and s.available


def test_the_snapshot_is_a_market_source_and_the_offline_fixture_serves_it():
    from core.data.offline import fixture_context
    assert "liq_levels" in MarketSnapshot.SOURCES
    ctx = fixture_context()
    assert ctx["liq_levels"].available and ctx["liq_levels"].venue == "Hyperliquid" and len(ctx["liq_levels"].longs) == 10
    assert LiquidationLevelsSnapshot.unavailable("hyperperps", "down").to_state() == {"available": False}


def test_to_state_is_plain_json_the_desk_state_and_the_stop_can_read():
    st = hyperperps.parse_heatmap(load("hyperperps_heatmap_btc.json")).to_state()
    json.dumps(st)
    assert st["available"] and st["venue"] == "Hyperliquid" and st["spot"] == 64_940.5
    assert {"price": 63_966.39, "notional_usd": 23_113_326.0, "wallets": 74, "distance_pct": -1.5} in st["longs"]
    assert len(st["shorts"]) == 10 and st["as_of_ms"] == 1_794_863_881_888
