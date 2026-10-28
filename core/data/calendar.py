"""Forex Factory economic calendar (keyless weekly JSON feed): high-impact macro prints with forecast/previous."""
from __future__ import annotations

import logging
import time
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


# The feed is a weekly file that changes a few times a day, and Forex Factory rate-limits a caller that asks
# for it every minute (the 24/7 worker was getting 429 on every pass by 28 Oct). One fetch per CACHE_MS is
# kept in the process; between fetches, and while the feed refuses us, the last good snapshot is served.
CACHE_MS = 30 * 60_000
RETRY_MS = 5 * 60_000            # after a failure, do not ask again sooner than this
_cache: dict = {"snapshot": None, "fetched_ms": 0, "failed_ms": 0}


def _now_ms() -> int:
    return int(time.time() * 1000)


def reset_cache() -> None:
    _cache.update(snapshot=None, fetched_ms=0, failed_ms=0)


async def fetch_calendar(client: httpx.AsyncClient) -> CalendarSnapshot:
    now = _now_ms()
    good = _cache["snapshot"]
    if good is not None and now - _cache["fetched_ms"] < CACHE_MS:
        return good
    if now - _cache["failed_ms"] < RETRY_MS:
        return good if good is not None else CalendarSnapshot.unavailable("forexfactory", "rate-limited; retry pending")
    try:
        r = await client.get(URL, headers=HEADERS)
        r.raise_for_status()
        snap = parse_calendar(r.json())
        _cache.update(snapshot=snap, fetched_ms=now, failed_ms=0)
        return snap
    except Exception as e:  # noqa: BLE001 - provider boundary
        _cache["failed_ms"] = now
        if good is not None:
            log.warning("forexfactory calendar fetch failed, serving the snapshot from %d min ago: %s",
                        (now - _cache["fetched_ms"]) // 60_000, e)
            return good
        log.error("forexfactory calendar unavailable: %s", e)
        return CalendarSnapshot.unavailable("forexfactory", str(e))
