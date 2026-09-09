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
