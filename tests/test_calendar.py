import json
import pathlib
from datetime import datetime, timezone

from core.data.calendar import parse_calendar

FX = pathlib.Path(__file__).parent / "fixtures"


def _snap():
    return parse_calendar(json.loads((FX / "ff_calendar_thisweek.json").read_text(encoding="utf-8")))


def test_parse_calendar_shape_and_ordering():
    c = _snap()
    assert c.available and c.source == "forexfactory" and len(c.events) > 20
    e = c.events[0]
    assert {"title", "country", "impact", "time_ms", "date", "forecast", "previous"} <= set(e)
    times = [x["time_ms"] for x in c.events if x["time_ms"] is not None]
    assert times == sorted(times) and times[0] > 1_600_000_000_000
    assert any(x["impact"] == "High" and x["country"] == "USD" for x in c.events)


def test_upcoming_filters_by_window_impact_country():
    c = _snap()
    first_high = next(x for x in c.events if x["impact"] == "High" and x["country"] == "USD")
    now = first_high["time_ms"] - 3_600_000  # one hour before the print
    soon = c.upcoming(now, within_ms=86_400_000)
    assert first_high in soon and all(x["impact"] == "High" and x["country"] == "USD" for x in soon)
    assert c.upcoming(now - 30 * 86_400_000, within_ms=86_400_000) == []
    everything = c.upcoming(now, within_ms=7 * 86_400_000, min_impact="Low", countries=())
    assert len(everything) >= len(soon)


def test_iso_offset_dates_parse_to_utc_ms():
    c = parse_calendar([{"title": "CPI m/m", "country": "USD", "impact": "High", "date": "2026-09-11T08:30:00-04:00", "forecast": "0.4%", "previous": "0.1%"}])
    expect = int(datetime(2026, 9, 11, 12, 30, tzinfo=timezone.utc).timestamp() * 1000)
    assert c.events[0]["time_ms"] == expect and c.events[0]["forecast"] == "0.4%"


def test_the_feed_is_fetched_once_per_cache_window_and_the_last_good_snapshot_survives_a_429(monkeypatch):
    import asyncio
    import httpx
    from core.data import calendar as cal
    cal.reset_cache()
    clock = {"ms": 1_000_000_000_000}
    monkeypatch.setattr(cal, "_now_ms", lambda: clock["ms"])
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json=[{"title": "CPI m/m", "country": "USD", "impact": "High",
                                              "date": "2026-09-11T08:30:00-04:00"}])
        return httpx.Response(429, text="Too Many Requests")

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            a = await cal.fetch_calendar(c)
            clock["ms"] += 60_000
            b = await cal.fetch_calendar(c)                   # within the window: no request at all
            clock["ms"] += cal.CACHE_MS
            d = await cal.fetch_calendar(c)                   # window over: asks, gets 429, serves the old one
            clock["ms"] += 60_000
            e = await cal.fetch_calendar(c)                   # a failure is not retried every minute
            return a, b, d, e
    a, b, d, e = asyncio.run(go())
    assert a.available and a.events[0]["title"] == "CPI m/m"
    assert b is a and d is a and e is a
    assert calls["n"] == 2, "one fetch per window, one retry after it, none in between"
    cal.reset_cache()
    clock["ms"] += cal.RETRY_MS

    async def cold():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await cal.fetch_calendar(c)
    assert not asyncio.run(cold()).available, "with nothing cached, a 429 is reported as unavailable"
    cal.reset_cache()
