"""Price candles from the Gate.io BTC perpetual: parsing, Monday weeks, older history, and the fallback chain."""
import asyncio
from datetime import datetime, timezone

import httpx
import pandas as pd

from core.data import gate_candles as gc
from core.data import spot
from core.data.types import SpotSnapshot

DAY_S = 86_400
MONDAY_S = 1_791_158_400            # 2026-10-05, a Monday
THURSDAY_S = MONDAY_S + 3 * DAY_S


def row(t, price=80_000.0, v=10_000):
    return {"t": t, "o": str(price), "h": str(price + 50), "l": str(price - 50), "c": str(price + 10), "v": v,
            "sum": str(v * 0.0001 * price)}


def test_rows_become_ms_timestamps_and_btc_volume_oldest_first():
    df = gc.parse_candles([row(MONDAY_S + 3600, v=20_000), row(MONDAY_S)])
    assert list(df["timestamp"]) == [MONDAY_S * 1000, (MONDAY_S + 3600) * 1000]
    assert df["volume"].iloc[1] == 2.0, "20,000 contracts of 0.0001 BTC is 2 BTC, not 20,000"
    assert df["close"].iloc[0] == 80_010.0


def test_a_broken_row_is_dropped_not_read_as_zero():
    bad = {"t": MONDAY_S + 60, "o": None, "h": "1", "l": "1", "c": "1", "v": 1}
    df = gc.parse_candles([row(MONDAY_S), bad])
    assert len(df) == 1 and (df["low"] > 0).all()


def test_an_error_payload_is_not_candles():
    try:
        gc.parse_candles({"label": "INVALID_PARAM_VALUE"})
    except ValueError as e:
        assert "INVALID_PARAM_VALUE" in str(e)
    else:
        raise AssertionError("an error object must not parse")


def _transport(daily_from=None, ticker=None, fail=()):
    """A fake Gate.io: 1d candles from Thursday to Thursday so the weeks must be rebuilt on Monday."""
    seen = []

    def handler(request):
        q = dict(request.url.params)
        seen.append(q)
        if request.url.path.endswith("/tickers"):
            return httpx.Response(200, json=ticker if ticker is not None else [{"last": "81000.5"}])
        tf = q["interval"]
        if tf in fail:
            return httpx.Response(503, json={"label": "SERVER_ERROR"})
        if tf == "1d" and "from" in q:
            return httpx.Response(200, json=daily_from or [])
        step = {"15m": 900, "1h": 3600, "4h": 14_400, "1d": DAY_S}[tf]
        return httpx.Response(200, json=[row(THURSDAY_S + i * step) for i in range(10)])
    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), seen


def test_weeks_open_on_monday_not_on_gate_s_thursday():
    client, _ = _transport()
    s = asyncio.run(gc.fetch_gate_candles(client))
    assert s.available and s.source == gc.SOURCE and s.last == 81000.5
    weeks = s.frames["1w"]["timestamp"]
    assert all(datetime.fromtimestamp(t / 1000, timezone.utc).weekday() == 0 for t in weeks)
    assert s.frames["1w"]["volume"].sum() == s.frames["1d"]["volume"].sum(), "every day lands in one week"


def test_older_daily_history_is_joined_in_front():
    older = [row(THURSDAY_S - (i + 1) * DAY_S, price=70_000.0) for i in range(5)]
    client, seen = _transport(daily_from=older)
    d = asyncio.run(gc.fetch_gate_candles(client)).frames["1d"]
    assert len(d) == 15 and d["timestamp"].is_monotonic_increasing and d["timestamp"].is_unique
    asked = next(q for q in seen if "from" in q)
    assert int(asked["to"]) == THURSDAY_S - DAY_S, "the older request stops the day before the newest batch"


def test_a_failed_ticker_falls_back_to_the_last_close():
    client, _ = _transport(ticker=[])
    assert asyncio.run(gc.fetch_gate_candles(client)).last == 80_010.0


def test_without_hourly_candles_the_feed_is_unavailable():
    client, _ = _transport(fail=("1h",))
    s = asyncio.run(gc.fetch_gate_candles(client))
    assert not s.available and "1h" in s.error


def _snap(source, ok=True):
    async def fn(client):
        if not ok:
            return SpotSnapshot.unavailable(source, "down")
        return SpotSnapshot(source=source, last=1.0, frames={"1h": pd.DataFrame()})
    return fn


def test_the_perp_is_read_first(monkeypatch):
    monkeypatch.setattr(spot, "fetch_gate_candles", _snap(gc.SOURCE))
    monkeypatch.setattr(spot, "fetch_kraken", _snap("kraken"))
    assert asyncio.run(spot.fetch_spot_chain(None)).source == gc.SOURCE


def test_spot_is_the_fallback_and_says_so(monkeypatch):
    monkeypatch.setattr(spot, "fetch_gate_candles", _snap(gc.SOURCE, ok=False))
    monkeypatch.setattr(spot, "fetch_kraken", _snap("kraken"))
    assert asyncio.run(spot.fetch_spot_chain(None)).source == "kraken"


def test_the_screen_names_the_market_the_candles_came_from():
    from ui import panels
    assert "BTC perp" in panels.top_bar_html(81_000, 0.5, "now", "BTC_USDT perp")
    assert "BTC / USD spot" in panels.top_bar_html(81_000, 0.5, "now", "BTC/USD")
