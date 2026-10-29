"""A host that refuses this machine (403, 451) is parked for six hours and asked once, not every minute."""
import asyncio
import json
import pathlib

import httpx
import pytest

from core.data import binance_futures, blocklist, news
from core.data.types import FuturesSnapshot

FX = pathlib.Path(__file__).parent / "fixtures"
NOW = 1_790_000_000_000


@pytest.fixture(autouse=True)
def _clean():
    blocklist.reset()
    yield
    blocklist.reset()


def _err(code: int) -> httpx.HTTPStatusError:
    req = httpx.Request("GET", "https://x.invalid/")
    return httpx.HTTPStatusError("boom", request=req, response=httpx.Response(code, request=req))


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_a_refusal_parks_the_source_and_a_server_error_does_not():
    assert blocklist.note_failure("binance", _err(451), NOW) is True
    assert blocklist.blocked("binance", NOW + 1) and blocklist.blocked("binance", NOW + blocklist.BLOCK_MS - 1)
    assert not blocklist.blocked("binance", NOW + blocklist.BLOCK_MS), "asked again after six hours"
    assert blocklist.note_failure("bybit", _err(503), NOW) is False and not blocklist.blocked("bybit", NOW)
    assert blocklist.note_failure("bybit", httpx.ReadTimeout("slow"), NOW) is False
    assert blocklist.status_code(ValueError("x")) is None


def test_the_futures_chain_stops_asking_venues_that_refused(monkeypatch):
    """The VM saw Binance 451 and Bybit 403 every minute for days: two failed round trips and two warnings
    a pass before Gate answered. After one refusal each, the pass goes straight to Gate."""
    calls = []

    def handler(request: httpx.Request):
        calls.append(request.url.host)
        return httpx.Response(451 if "binance" in request.url.host else 403, request=request)

    gate = FuturesSnapshot(source="gate", funding_rate=0.0001, open_interest=1.0)

    async def fake_gate(client):
        return gate
    monkeypatch.setattr("core.data.gate_futures.fetch_gate_futures", fake_gate)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    first = _run(binance_futures.fetch_futures(client))
    assert first is gate and {"fapi.binance.com", "api.bybit.com"} <= set(calls)
    calls.clear()
    second = _run(binance_futures.fetch_futures(client))
    assert second is gate and calls == [], "neither refused venue was asked again"


def test_the_news_aggregator_that_refused_is_skipped_and_named():
    def load(name):
        return (FX / name).read_text(encoding="utf-8")
    asked = []

    def handler(request: httpx.Request):
        asked.append(request.url.host)
        if "cryptocurrency.cv" in request.url.host:
            return httpx.Response(403, request=request)
        return httpx.Response(200, content=load("coindesk_rss.xml").encode())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    first = _run(news.fetch_news(client))
    assert first.available and "cryptocurrency.cv" in (first.error or "")
    asked.clear()
    second = _run(news.fetch_news(client))
    assert second.available and "cryptocurrency.cv" not in asked, "not asked again"
    assert "cryptocurrency.cv: skipped (refused this host earlier)" in (second.error or "")
    assert asked, "the publisher feeds are still read"
