"""Forex Factory economic calendar (keyless weekly JSON feed): high-impact macro prints with forecast/previous."""
from __future__ import annotations

import logging
from datetime import datetime

import httpx

from core.data.types import CalendarSnapshot

log = logging.getLogger(__name__)
URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (trap-intel-dashboard)"}


def _time_ms(s: str) -> int | None:
    try:
        return int(datetime.fromisoformat(s).timestamp() * 1000)
    except (TypeError, ValueError):
        return None


def parse_calendar(payload: list[dict]) -> CalendarSnapshot:
    events = []
    for e in payload:
        events.append({
            "title": e.get("title", ""), "country": e.get("country", ""), "impact": e.get("impact", ""),
            "time_ms": _time_ms(e.get("date", "")), "date": e.get("date", ""),
            "forecast": e.get("forecast") or None, "previous": e.get("previous") or None,
        })
    events.sort(key=lambda x: (x["time_ms"] is None, x["time_ms"] or 0))
    return CalendarSnapshot(source="forexfactory", events=events)


async def fetch_calendar(client: httpx.AsyncClient) -> CalendarSnapshot:
    try:
        r = await client.get(URL, headers=HEADERS)
        r.raise_for_status()
        return parse_calendar(r.json())
    except Exception as e:  # noqa: BLE001 - provider boundary
        log.error("forexfactory calendar unavailable: %s", e)
        return CalendarSnapshot.unavailable("forexfactory", str(e))
