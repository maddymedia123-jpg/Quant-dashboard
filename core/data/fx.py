"""USD/JPY and the yen carry trade, from the ECB reference rates via frankfurter.dev (keyless).

Roadmap item 10 (owner's GO, 5 November): "Japanese Yen Carry-Trade Unwind: monitors BoJ rate hikes vs.
US yields". The carry trade borrows yen at the Bank of Japan's near-zero rate and buys dollar assets,
bitcoin among them. It unwinds when the yen strengthens fast (USD/JPY falls) or the rate gap closes:
August 2024 took bitcoin from about 65,000 to 49,000 in four days on exactly that. So two things are read:

* the pace of USD/JPY: a fall of UNWIND_WEEK_PCT or more in a week of sessions is the unwind warning;
* the carry itself: the US funds rate (EFFR, from the macro feed) minus the Bank of Japan's policy rate.

The BoJ rate has no free feed worth trusting (it changes a few times a year, at scheduled meetings), so
it is HAND-SET by the owner in the environment as BOJ_POLICY_RATE_PCT (for example 0.75). Unset, the
carry differential is not shown and the read rests on the pace of USD/JPY alone; it is never guessed.

Found by calling the feed: frankfurter.app answers 301 (the host moved to api.frankfurter.dev/v1); the
range form `/v1/2026-08-25..?base=USD&symbols=JPY` returns {"rates": {"2026-10-20": {"JPY": 158.13},
...}} on ECB business days only (no weekends, no ECB holidays), so a reading three days old over a
weekend is normal and the staleness threshold allows for it."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

import httpx

from core.data.types import FxSnapshot

log = logging.getLogger(__name__)

URL = "https://api.frankfurter.dev/v1/{start}.."
HEADERS = {"User-Agent": "trap-intel-dashboard/1.0"}
MAX_BYTES = 256 * 1024
WINDOW_DAYS = 45
WEEK_BACK_OBSERVATIONS = 5        # a week of sessions, as the macro feed counts it
MONTH_BACK_OBSERVATIONS = 21
STALE_DAYS = 4                    # ECB business days: a Friday fix read on Monday is 3 days old
PLAUSIBLE = (50.0, 400.0)         # USD/JPY has lived between 75 and 162 since 1990
UNWIND_WEEK_PCT = 1.5             # the yen strengthening this fast in a week is the unwind warning
BOJ_ENV = "BOJ_POLICY_RATE_PCT"


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _date_ms(text: str) -> int | None:
    try:
        d = datetime.strptime(str(text).strip()[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None
    return int(d.timestamp() * 1000)


def parse_range(payload: dict) -> list[tuple[int, float]]:
    """(date_ms, USD/JPY) for every day in a range response, oldest first. Refuses a response for another
    base currency - the goldprice.dev wrong-currency bug, in another feed."""
    if not isinstance(payload, dict):
        raise ValueError("frankfurter: not an object")
    base = str(payload.get("base") or "").upper()
    if base != "USD":
        raise ValueError(f"frankfurter: asked for USD, the response is base {base or 'missing'!r}")
    rates = payload.get("rates")
    if not isinstance(rates, dict) or not rates:
        raise ValueError("frankfurter: no rates in the window")
    out: list[tuple[int, float]] = []
    for day, quotes in rates.items():
        when = _date_ms(day)
        value = quotes.get("JPY") if isinstance(quotes, dict) else None
        if when is None or value is None:
            continue
        try:
            out.append((when, float(value)))
        except (TypeError, ValueError):
            continue
    if not out:
        raise ValueError("frankfurter: no JPY quotes in the window")
    out.sort()
    return out


def boj_rate(env=None) -> float | None:
    """The hand-set Bank of Japan policy rate, or None when the owner has not set it."""
    raw = (env if env is not None else os.environ).get(BOJ_ENV, "")
    raw = str(raw).strip().rstrip("%")
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        log.warning("%s=%r is not a number; the BoJ rate is treated as unset", BOJ_ENV, raw)
        return None
    if not -1.0 <= value <= 10.0:
        log.warning("%s=%s is not a policy rate anybody has seen; treated as unset", BOJ_ENV, value)
        return None
    return value


def build(rows: list[tuple[int, float]], now_ms: int, boj_rate_pct: float | None = None,
          source: str = "frankfurter") -> FxSnapshot:
    if not rows:
        return FxSnapshot.unavailable(source, "no USD/JPY quotes in the window")
    as_of, rate = rows[-1]
    if not PLAUSIBLE[0] <= rate <= PLAUSIBLE[1]:
        return FxSnapshot.unavailable(source, f"USD/JPY {rate} is outside the plausible {PLAUSIBLE[0]:.0f}-{PLAUSIBLE[1]:.0f} range")
    week = rows[-1 - WEEK_BACK_OBSERVATIONS][1] if len(rows) > WEEK_BACK_OBSERVATIONS else None
    month = rows[-1 - MONTH_BACK_OBSERVATIONS][1] if len(rows) > MONTH_BACK_OBSERVATIONS else None
    age_days = (now_ms - as_of) / 86_400_000
    return FxSnapshot(
        source=source, rate=rate, as_of_ms=as_of,
        week_ago=week, change_1w_pct=round((rate / week - 1.0) * 100.0, 2) if week else None,
        month_ago=month, change_1m_pct=round((rate / month - 1.0) * 100.0, 2) if month else None,
        stale=age_days > STALE_DAYS, age_days=round(age_days, 1),
        boj_rate_pct=boj_rate_pct,
    )


async def fetch_fx(client: httpx.AsyncClient, now_ms: int | None = None) -> FxSnapshot:
    now = now_ms or _now_ms()
    start = (datetime.fromtimestamp(now / 1000, timezone.utc) - timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
    r = await client.get(URL.format(start=start), params={"base": "USD", "symbols": "JPY"}, headers=HEADERS,
                         follow_redirects=True)
    r.raise_for_status()
    if len(r.content) > MAX_BYTES:
        raise ValueError(f"frankfurter: {len(r.content) / 1024:.0f} KB for a {WINDOW_DAYS}-day window")
    return build(parse_range(r.json()), now, boj_rate())


def yen_read(fx: FxSnapshot, effr_pct: float | None = None) -> dict:
    """The carry-trade read: {"state": unwind_risk | carry_on | quiet | unknown, "carry_pct", "note"}."""
    if not getattr(fx, "available", False) or fx.rate is None:
        return {"state": "unknown", "carry_pct": None, "note": "no USD/JPY feed"}
    carry = None
    if effr_pct is not None and fx.boj_rate_pct is not None:
        carry = round(effr_pct - fx.boj_rate_pct, 2)
    carry_text = (f"; the carry (US funds rate {effr_pct:.2f}% minus BoJ {fx.boj_rate_pct:.2f}%) is {carry:+.2f} pp"
                  if carry is not None else "; BoJ rate not set, so the carry gap is not shown")
    week = fx.change_1w_pct
    if week is None:
        return {"state": "unknown", "carry_pct": carry,
                "note": f"USD/JPY {fx.rate:.2f}, no week-ago quote yet{carry_text}"}
    if week <= -UNWIND_WEEK_PCT:
        return {"state": "unwind_risk", "carry_pct": carry,
                "note": (f"USD/JPY {fx.rate:.2f}, down {abs(week):.1f}% in a week: the yen is strengthening fast, "
                         f"carry-trade unwind risk (yen-funded longs in bitcoin get cut){carry_text}")}
    if week >= UNWIND_WEEK_PCT:
        return {"state": "carry_on", "carry_pct": carry,
                "note": (f"USD/JPY {fx.rate:.2f}, up {week:.1f}% in a week: the yen is weakening, the carry trade "
                         f"is on and supportive of risk{carry_text}")}
    return {"state": "quiet", "carry_pct": carry,
            "note": f"USD/JPY {fx.rate:.2f}, {week:+.1f}% on the week: the yen is quiet, no carry signal{carry_text}"}
