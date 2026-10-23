"""Market Pulse Dispatch: his Discord template, filled in plain words.

Sent at two moments, one message each (the alert ledger's unique index keeps it to one even with several
pages open):

  * a report is run   - from the report's plan when it has one (direction, Target 1, the stop), and from
                        the Live desk otherwise; keyed on the report id
  * a 4h candle close - from the Live desk (next 4 hours); keyed on the close

Both are checked on every page load, so with no page open nothing is sent. A close or a report older than
SEND_WITHIN_MS is skipped rather than sent late: a three-hour-old pulse reads as news and is not.

The words are written here from the numbers, never copied from a model's text, and checked against the
same banned list as the plan alerts: the reader is not a trader."""
from __future__ import annotations

import json

from core.alerts import Event

HOUR_MS = 3_600_000
CLOSE_MS = 4 * HOUR_MS
SEND_WITHIN_MS = 30 * 60_000
NEWS_WITHIN_MS = 6 * HOUR_MS             # headlines from the last six hours
EVENTS_WITHIN_MS = 24 * HOUR_MS          # high-impact US prints in the next day
RULE = "=" * 40
DIRECTION_WORD = {"BULLISH": "BULLISH", "BEARISH": "BEARISH", "NEUTRAL": "RANGE-BOUND"}
CONF_WORD = lambda c: ("very strong" if c >= 0.85 else "strong" if c >= 0.60  # noqa: E731
                       else "moderate" if c >= 0.35 else "slight")


def _money(x: float | None) -> str:
    return "—" if x is None else f"${x:,.0f}"


def _when(ms: int) -> str:
    from core.trades import fmt_when
    return fmt_when(ms)


def _pkt(ms: int) -> str:
    from datetime import datetime, timezone
    return f"{datetime.fromtimestamp(ms / 1000 + 5 * 3600, timezone.utc):%H:%M} PKT"


def _floor_ceiling(a) -> tuple[float | None, float | None]:
    sups = sorted((lv.price for lv in a.levels if lv.kind == "support" and lv.price < a.price), reverse=True)
    ceils = sorted(lv.price for lv in a.levels if lv.kind == "resistance" and lv.price > a.price)
    return (sups[0] if sups else a.vol.exp_low), (ceils[0] if ceils else a.vol.exp_high)


def desk_read(a) -> dict:
    """Direction, the two sentences, target and safety line, from the Live desk's own analysis."""
    d = a.direction
    floor, ceiling = _floor_ceiling(a)
    direction = DIRECTION_WORD.get(d.direction, "RANGE-BOUND")
    if direction == "BULLISH":
        happening = (f"BTC is {_money(a.price)} and leaning higher ({CONF_WORD(d.confidence)}) for the next 4 hours. "
                     f"The nearest ceiling is {_money(ceiling)}; holding above {_money(floor)} keeps buyers in control.")
        target = _money(ceiling)
    elif direction == "BEARISH":
        happening = (f"BTC is {_money(a.price)} and leaning lower ({CONF_WORD(d.confidence)}) for the next 4 hours. "
                     f"The nearest floor is {_money(floor)}; staying below {_money(ceiling)} keeps sellers in control.")
        target = _money(floor)
    else:
        happening = (f"BTC is {_money(a.price)} with no clear side for the next 4 hours. "
                     f"It is stuck between {_money(floor)} and {_money(ceiling)} until one of them breaks.")
        target = f"{_money(floor)} or {_money(ceiling)}, whichever breaks first"
    if direction != "RANGE-BOUND" and d.invalidation is not None:
        safety = f"{_money(d.invalidation)} (a 1-hour close {'below' if direction == 'BULLISH' else 'above'} it ends this view)"
    elif floor and ceiling:
        safety = f"a 1-hour close below {_money(floor)} or above {_money(ceiling)} picks a side"
    else:
        safety = "no clear line yet"
    return {"direction": direction, "happening": happening, "target": target, "safety": safety}


def plan_read(report_row: dict, price: float | None) -> dict | None:
    """The same fields from a report's plan, when it has one with a side."""
    try:
        plan = json.loads(report_row.get("plan") or "null")
    except (TypeError, ValueError):
        return None
    if not plan or plan.get("side") not in ("LONG", "SHORT"):
        return None
    short = plan["side"] == "SHORT"
    targets = sorted((float(t) for t in plan.get("targets") or []), reverse=short)
    zone = f"{_money(float(plan['zone_low']))}–{_money(float(plan['zone_high']))}"
    hours = float(plan.get("expires_hours") or 24)
    happening = (f"BTC is {_money(price)}. The new report expects a move {'down' if short else 'up'}: it looks to "
                 f"{'sell' if short else 'buy'} in the {zone} zone, and only after a candle confirms it "
                 f"(the idea is good for {hours:g} hours).")
    return {"direction": "BEARISH" if short else "BULLISH", "happening": happening,
            "target": _money(targets[0]) if targets else "—", "safety": _money(float(plan["stop"]))}


def _news(m, now_ms: int) -> str:
    lines = []
    cal = getattr(m, "calendar", None)
    if getattr(cal, "available", False):
        for e in sorted(cal.upcoming(now_ms, EVENTS_WITHIN_MS), key=lambda e: e["time_ms"])[:3]:
            lines.append(f"US {e.get('title', 'event')} at {_when(e['time_ms'])}: expect sharp moves around it.")
    news = getattr(m, "news", None)
    if getattr(news, "available", False):
        heads = sorted(news.recent(now_ms, NEWS_WITHIN_MS, "high"), key=lambda h: -(h.get("published_ms") or 0))
        lines += [f"Headline: {h.get('title', '').strip()}" for h in heads[:2] if h.get("title")]
    return "\n".join(lines) or "No major news or US data releases in the next 24 hours."


def text(read: dict, m, now_ms: int, basis: str) -> str:
    from core.indicators.turn_window import next_window
    w = next_window(now_ms)
    turn = (f"NOW, until {_pkt(w.end_ms)} (the busiest turning hours of the day)" if w.active
            else f"Between {_pkt(w.start_ms)} and {_pkt(w.end_ms)}")
    return "\n".join([
        RULE, f"🤖 MARKET PULSE DISPATCH · {basis}", f"⏰ Time: {_when(now_ms)}", RULE, "",
        f"🎯 MARKET DIRECTION: {read['direction']}",
        "💡 WHAT IS HAPPENING:", read["happening"], "",
        "🕒 EXACT TIME TO WATCH:", f"Next Turning Window: {turn}", "",
        "📍 IMPORTANT PRICE LEVELS:", f"Target Level: {read['target']}",
        f"Invalidation (Safety Line): {read['safety']}", "",
        "⚠️ IMPORTANT NEWS / EVENT WARNINGS:", _news(m, now_ms), RULE])


def close_event(live, m, now_ms: int) -> Event | None:
    """The pulse for the 4h candle that just closed, while that close is still fresh."""
    if live is None:
        return None
    close = now_ms - now_ms % CLOSE_MS
    if now_ms - close > SEND_WITHIN_MS:
        return None
    return Event("pulse", f"pulse:4h:{close}", text(desk_read(live), m, now_ms, "4h candle close · Live desk"),
                 live.price)


def report_event(report_row: dict | None, live, m, now_ms: int) -> Event | None:
    """The pulse for the newest report, once, while it is fresh."""
    if not report_row or now_ms - int(report_row.get("ts_ms") or 0) > SEND_WITHIN_MS:
        return None
    price = getattr(m.spot, "last", None)
    read = plan_read(report_row, price)
    basis = f"new report {report_row.get('report_id', '')}".strip()
    if read is None:
        if live is None:
            return None
        read, basis = desk_read(live), basis + " · Live desk"
    return Event("pulse", f"pulse:report:{report_row.get('report_id')}", text(read, m, now_ms, basis), price)
