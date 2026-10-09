"""US macro context from FRED (keyless CSV): the dollar, the 10-year, the 2-year, and the 2s10s spread.

Why these four. Bitcoin trades as a risk asset against the dollar and against real yields, and the
2s10s spread is the market's own read on the cycle. The feed already carries headlines like "Bitcoin
fails to sustain $85,000 breakout, and bond yield spikes" - this is what lets the screen say whether
yields actually moved, rather than leaving the headline unverified.

FRED's `fredgraph.csv` endpoint needs no key. Three things about it were found by calling it, and each
is handled below:

* Holidays are EMPTY values (`2025-12-25,`), and older exports used a dot. So the last row is not the
  last value - read naively on a holiday, a series reports nothing at all, or zero.
* The 10-year and the 2-year can publish on different days. A spread that subtracts one day's 2-year
  from another day's 10-year is a number about no day, so the spread uses the latest date both share.
* `cosd=` (a start date) is honoured, which takes a response from 104 KB of history since 2006 to a few
  hundred bytes. Only the recent window is ever needed.

Daily series publish with a business-day lag or two, so a value a couple of days old is normal. A week
old is not, and is flagged rather than shown as current."""
from __future__ import annotations

import asyncio
import csv
import io
import logging
from datetime import datetime, timedelta, timezone

import httpx

from core.data.types import MacroSnapshot

log = logging.getLogger(__name__)

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
HEADERS = {"User-Agent": "trap-intel-dashboard/1.0"}
WINDOW_DAYS = 45               # enough for a week-ago comparison across holidays, nothing more
# Staleness is per series, because the publication cadences differ. The Treasury yields are daily and
# lag a business day or two. The broad dollar index is the Fed's H.10, released WEEKLY on Mondays, so a
# week-plus lag is its normal state: one shared threshold would flag it stale every week before each
# release - a false alarm that teaches the reader to ignore the flag.
WEEK_BACK_OBSERVATIONS = 5     # five trading days, so "1w change" is a week of sessions, not calendar
MAX_BYTES = 1024 * 1024        # a 45-day window is under 1 KB; anything near this is not that

# (FRED id, label, unit, plausible low, plausible high, stale after N days). The bands refuse nothing
# real and catch a provider swapping a series or shifting a decimal - a 10-year of 52.8 is not a yield
# anybody has seen.
SERIES: tuple[tuple[str, str, str, float, float, int], ...] = (
    ("DTWEXBGS", "US dollar (broad index)", "index", 50.0, 250.0, 12),   # weekly H.10 release
    ("DGS10", "US 10-year yield", "%", -2.0, 25.0, 5),                    # daily, lags 1-2 days
    ("DGS2", "US 2-year yield", "%", -2.0, 25.0, 5),
)
MISSING = {"", ".", "NA", "N/A", "null"}


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _date_ms(text: str) -> int | None:
    try:
        d = datetime.strptime(text.strip()[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None
    return int(d.timestamp() * 1000)


def parse_series(csv_text: str, series_id: str) -> list[tuple[int, float]]:
    """(date_ms, value) for every observation that HAS a value, oldest first.

    The header is `observation_date,<ID>` today and was `DATE,<ID>` before, so columns are read by
    position. Rows with an empty or dotted value are holidays and are skipped, not read as zero."""
    reader = csv.reader(io.StringIO(csv_text))
    header = next(reader, None)
    if not header or len(header) < 2:
        raise ValueError(f"{series_id}: no header row")
    if header[1].strip().upper() != series_id.upper():
        # FRED answering with a different series is the same shape of bug as goldprice.dev answering
        # in a different currency: HTTP 200, plausible numbers, wrong thing
        raise ValueError(f"asked for {series_id}, the response is {header[1].strip()!r}")
    out: list[tuple[int, float]] = []
    for row in reader:
        if len(row) < 2:
            continue
        raw = row[1].strip()
        if raw in MISSING:
            continue
        when = _date_ms(row[0])
        if when is None:
            continue
        try:
            out.append((when, float(raw)))
        except ValueError:
            continue
    out.sort()
    return out


def summarise(series_id: str, label: str, unit: str, low: float, high: float,
              rows: list[tuple[int, float]], now_ms: int, stale_after_days: int = 5) -> dict:
    """The latest value, the value a week of sessions earlier, and whether it is too old to trust."""
    if not rows:
        return {"id": series_id, "label": label, "unit": unit, "available": False,
                "note": "no observations in the window"}
    as_of, value = rows[-1]
    if not low <= value <= high:
        return {"id": series_id, "label": label, "unit": unit, "available": False,
                "note": f"latest value {value} is outside the plausible {low}-{high} range"}
    back = rows[-1 - WEEK_BACK_OBSERVATIONS] if len(rows) > WEEK_BACK_OBSERVATIONS else None
    age_days = (now_ms - as_of) / 86_400_000
    return {"id": series_id, "label": label, "unit": unit, "available": True,
            "value": value, "as_of_ms": as_of,
            "week_ago": back[1] if back else None,
            "change": round(value - back[1], 4) if back else None,
            "stale": age_days > stale_after_days,
            "age_days": round(age_days, 1)}


def curve_spread(ten: list[tuple[int, float]], two: list[tuple[int, float]]) -> tuple[float | None, int | None]:
    """10-year minus 2-year, on the latest date BOTH published. Subtracting different days' values
    would describe no day at all."""
    twos = dict(two)
    for when, ten_value in reversed(ten):
        if when in twos:
            return round(ten_value - twos[when], 4), when
    return None, None


async def _fetch_csv(client: httpx.AsyncClient, series_id: str, since: str) -> str:
    r = await client.get(URL, params={"id": series_id, "cosd": since}, headers=HEADERS,
                         follow_redirects=True)
    r.raise_for_status()
    if len(r.content) > MAX_BYTES:
        raise ValueError(f"{series_id}: {len(r.content) / 1024:.0f} KB for a {WINDOW_DAYS}-day window")
    return r.text


def build(texts: dict[str, str | BaseException], now_ms: int) -> MacroSnapshot:
    """Assemble the snapshot from whatever arrived. One series failing is a degraded feed, named."""
    series: dict[str, dict] = {}
    parsed: dict[str, list[tuple[int, float]]] = {}
    problems: list[str] = []
    for series_id, label, unit, low, high, stale_after in SERIES:
        raw = texts.get(series_id)
        if isinstance(raw, BaseException) or raw is None:
            reason = f"{type(raw).__name__}: {raw}".strip(": ") if raw is not None else "not fetched"
            series[series_id] = {"id": series_id, "label": label, "unit": unit, "available": False,
                                 "note": reason}
            problems.append(f"{series_id}: {reason}")
            continue
        try:
            rows = parse_series(raw, series_id)
        except ValueError as e:
            series[series_id] = {"id": series_id, "label": label, "unit": unit, "available": False,
                                 "note": str(e)}
            problems.append(f"{series_id}: {e}")
            continue
        parsed[series_id] = rows
        series[series_id] = summarise(series_id, label, unit, low, high, rows, now_ms, stale_after)

    if not any(s.get("available") for s in series.values()):
        return MacroSnapshot.unavailable("fred", "; ".join(problems) or "no series available")

    spread, spread_ms = None, None
    if series.get("DGS10", {}).get("available") and series.get("DGS2", {}).get("available"):
        spread, spread_ms = curve_spread(parsed["DGS10"], parsed["DGS2"])
    return MacroSnapshot(source="fred", series=series, spread_2s10s=spread, spread_as_of_ms=spread_ms,
                         error="; ".join(problems) or None)


async def fetch_macro(client: httpx.AsyncClient, now_ms: int | None = None) -> MacroSnapshot:
    now = now_ms or _now_ms()
    since = (datetime.fromtimestamp(now / 1000, timezone.utc) - timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
    ids = [s[0] for s in SERIES]
    results = await asyncio.gather(*(_fetch_csv(client, i, since) for i in ids), return_exceptions=True)
    for series_id, result in zip(ids, results):
        if isinstance(result, BaseException):
            log.warning("FRED %s unavailable: %s", series_id, result)
    return build(dict(zip(ids, results)), now)
