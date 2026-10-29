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
MARKET_WIDE = ("macro", "geopolitics", "etf")   # headline topics that move every market, BTC named or not
FED_WITHIN_MS = 7 * 24 * HOUR_MS
FED_WORDS = ("fomc", "federal funds rate", "fed chair")
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


def _target(a, up: bool, floor: float | None, ceiling: float | None) -> str:
    """The farther of the first level in the way and the 4-hour expected-range edge. The first level alone
    read as the target (29 Oct: a 'target' 0.4% away) when it is only the first obstacle."""
    edge = a.vol.exp_high if up else a.vol.exp_low
    first = ceiling if up else floor
    cands = [x for x in (first, edge) if x is not None]
    if not cands:
        return "—"
    return _money(max(cands) if up else min(cands))


def desk_read(a) -> dict:
    """Direction, the two sentences, target and safety line, from the Live desk's own analysis."""
    d = a.direction
    floor, ceiling = _floor_ceiling(a)
    direction = DIRECTION_WORD.get(d.direction, "RANGE-BOUND")
    if direction == "BULLISH":
        happening = (f"BTC is {_money(a.price)} and leaning higher ({CONF_WORD(d.confidence)}) for the next 4 hours. "
                     f"The nearest ceiling is {_money(ceiling)}; holding above {_money(floor)} keeps buyers in control.")
        target = _target(a, True, floor, ceiling)
    elif direction == "BEARISH":
        happening = (f"BTC is {_money(a.price)} and leaning lower ({CONF_WORD(d.confidence)}) for the next 4 hours. "
                     f"The nearest floor is {_money(floor)}; staying below {_money(ceiling)} keeps sellers in control.")
        target = _target(a, False, floor, ceiling)
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


def _plan(report_row: dict | None) -> dict | None:
    try:
        plan = json.loads((report_row or {}).get("plan") or "null")
    except (TypeError, ValueError):
        return None
    return plan if plan and plan.get("side") in ("LONG", "SHORT") else None


def plan_live(report_row: dict | None, now_ms: int) -> bool:
    """Whether a report's plan is still the one being traded: it has a side, its hours have not run out,
    and the ledger has not closed it (target, stop, not triggered). While it is, every pulse speaks from it,
    so the 4h-close message and the report message name the same target and the same safety line."""
    plan = _plan(report_row)
    if plan is None:
        return False
    if (report_row.get("plan_outcome") or "open") != "open":
        return False
    expires = int(report_row.get("ts_ms") or 0) + int(float(plan.get("expires_hours") or 24) * 3_600_000)
    return now_ms < expires


def plan_read(report_row: dict, price: float | None, now_ms: int | None = None) -> dict | None:
    """The same fields from a report's plan, when it has one with a side."""
    plan = _plan(report_row)
    if plan is None:
        return None
    from core.plan_alerts import TF_WORDS

    short = plan["side"] == "SHORT"
    targets = sorted((float(t) for t in plan.get("targets") or []), reverse=short)
    low, high = float(plan["zone_low"]), float(plan["zone_high"])
    zone = f"{_money(low)}–{_money(high)}"
    hours = float(plan.get("expires_hours") or 24)
    tf = plan.get("confirm_timeframe") or "1h"
    level = float(plan["confirm_close"]) if plan.get("confirm_close") is not None else (low if short else high)
    # the confirming close, spelled out from the plan's own numbers (the same rule the WATCH alert sends)
    confirm = f"a {TF_WORDS.get(tf, tf)} candle closes {'below' if short else 'above'} {_money(level)}"
    if plan.get("taker_ratio_limit") is not None:
        confirm += f", with {'sellers' if short else 'buyers'} doing more of the trading"
    act = "sell" if short else "buy"
    # where price stands against the zone decides the first words: a short entered above price must say
    # "wait for the rise", or a reader shorts at the market, the one place the plan says not to
    if price is not None and low <= price <= high:
        where = f"BTC is {_money(price)}, inside the {act} zone ({zone})."
        step = f"{act.capitalize()} only after {confirm}."
    elif price is not None and (price < low if short else price > high):
        where = f"BTC is {_money(price)}, {'below' if short else 'above'} the {act} zone ({zone})."
        step = (f"Do not {act} here: wait for a {'rise' if short else 'dip'} into the zone, then {act} only after "
                f"{confirm}.")
    else:
        where = f"BTC is {_money(price)}; the plan {act}s from the {zone} zone."
        step = f"{act.capitalize()} only after {confirm}."
    happening = (f"{where} The new report expects a move {'down' if short else 'up'}. {step} "
                 f"The idea is good for {hours:g} hours.")
    return {"direction": "BEARISH" if short else "BULLISH", "happening": happening,
            "target": _money(targets[0]) if targets else "—", "safety": _money(float(plan["stop"]))}


def _positioning(m, event: dict, now_ms: int) -> str:
    """The macro engine's read of BTC into a print (core.indicators.event_trap), in plain words, only when
    BTC has already moved ahead of it - the case a reader can do something about."""
    try:
        from core.indicators.event_trap import read_all
        reads = read_all(m.spot.frames.get("1h"), [event], now_ms)
    except Exception:  # noqa: BLE001 - the release line stands without it
        return ""
    r = reads[0] if reads else None
    if r is None or r.state != "positioned" or r.runup_pct is None:
        return ""
    up = r.runup_pct > 0
    return (f" BTC has already moved {r.runup_pct:+.1f}% ahead of it, so a release that only matches forecast "
            f"could reverse that ({'a drop' if up else 'a bounce'}).")


def _fed_meeting_soon(cal, now_ms: int) -> bool:
    return any(any(w in str(e.get("title", "")).lower() for w in FED_WORDS)
               for e in cal.upcoming(now_ms, FED_WITHIN_MS))


def _news(m, now_ms: int) -> str:
    """From the Macro & News screen's own data, in order: US releases in the next day (with the macro
    engine's read when BTC has run ahead of one), headlines that name Bitcoin or move every market, and
    the Fed odds when a Fed meeting is near. A hack on a small chain or a crypto-policy story abroad is not
    a BTC warning, so headlines are kept only when they name BTC or fall in a market-wide topic."""
    lines = []
    cal = getattr(m, "calendar", None)
    if getattr(cal, "available", False):
        for e in sorted(cal.upcoming(now_ms, EVENTS_WITHIN_MS), key=lambda e: e["time_ms"])[:3]:
            lines.append(f"US {e.get('title', 'release')} at {_when(e['time_ms'])}: expect sharp moves around it."
                         + _positioning(m, e, now_ms))
    news = getattr(m, "news", None)
    if getattr(news, "available", False):
        heads = [h for h in news.recent(now_ms, NEWS_WITHIN_MS, "high") if h.get("title")
                 and ("BTC" in (h.get("assets") or []) or h.get("topic") in MARKET_WIDE)]
        # Bitcoin named first, then newest: "sanctions" also tags crypto-policy stories as geopolitics
        heads.sort(key=lambda h: ("BTC" not in (h.get("assets") or []), -(h.get("published_ms") or 0)))
        lines += [f"Headline: {h['title'].strip()}" for h in heads[:2]]
    preds = getattr(m, "predictions", None)
    if getattr(cal, "available", False) and getattr(preds, "available", False) and _fed_meeting_soon(cal, now_ms):
        odds = [x for x in preds.tradeable("MACRO") if "fed" in x.get("question", "").lower()]
        if odds:
            top = max(odds, key=lambda x: x.get("volume_24h") or 0)
            lines.append(f"Fed meeting this week: markets give {top['probability']:.0%} to \"{top['question'].strip()}\"")
    return "\n".join(lines) or "No US data releases or Bitcoin-moving headlines in the next 24 hours."


def text(read: dict, m, now_ms: int, basis: str) -> str:
    from core.indicators.turn_window import PKT_S, next_window
    w = next_window(now_ms)
    # the window names a PKT day: once today's has passed, "tomorrow" has to be said, or a 21:00 PKT
    # message reading "between 17:00 and 21:00 PKT" names hours that have just ended (29 Oct)
    pkt_day = lambda ms: (ms + PKT_S * 1000) // 86_400_000  # noqa: E731
    when = "Tomorrow, between" if pkt_day(w.start_ms) > pkt_day(now_ms) else "Later today, between"
    turn = (f"NOW, until {_pkt(w.end_ms)} (the busiest turning hours of the day)" if w.active
            else f"{when} {_pkt(w.start_ms)} and {_pkt(w.end_ms)}")
    return "\n".join([
        RULE, f"🤖 MARKET PULSE DISPATCH · {basis}", f"⏰ Time: {_when(now_ms)}", RULE, "",
        f"🎯 MARKET DIRECTION: {read['direction']}",
        "💡 WHAT IS HAPPENING:", read["happening"], "",
        "🕒 EXACT TIME TO WATCH:", f"Next Turning Window: {turn}", "",
        "📍 IMPORTANT PRICE LEVELS:", f"Target Level: {read['target']}",
        f"Invalidation (Safety Line): {read['safety']}", "",
        "⚠️ IMPORTANT NEWS / EVENT WARNINGS:", _news(m, now_ms), RULE])


def close_event(live, m, now_ms: int, plan_row: dict | None = None) -> Event | None:
    """The pulse for the 4h candle that just closed, while that close is still fresh. With a live plan it
    speaks from the plan (the report's zone, target and stop); otherwise from the Live desk."""
    close = now_ms - now_ms % CLOSE_MS
    if now_ms - close > SEND_WITHIN_MS:
        return None
    price = getattr(m.spot, "last", None)
    if plan_live(plan_row, now_ms):
        read = plan_read(plan_row, price if price is not None else getattr(live, "price", None))
        if read is not None:
            basis = f"4h candle close · plan from report {plan_row.get('report_id', '')}".strip()
            return Event("pulse", f"pulse:4h:{close}", text(read, m, now_ms, basis), price)
    if live is None:
        return None
    return Event("pulse", f"pulse:4h:{close}", text(desk_read(live), m, now_ms, "4h candle close · Live desk"),
                 live.price)


def report_event(report_row: dict | None, live, m, now_ms: int) -> Event | None:
    """The pulse for a new report that carries a plan with a side, once, while it is fresh. A report
    without one (the 15-agent report, or a 1-agent 'stand aside') adds nothing the 4h-close pulse does not
    already say, so it sends nothing: a compare run is one dispatch, not two."""
    if not report_row or now_ms - int(report_row.get("ts_ms") or 0) > SEND_WITHIN_MS:
        return None
    price = getattr(m.spot, "last", None)
    read = plan_read(report_row, price)
    if read is None:
        return None
    basis = f"new report {report_row.get('report_id', '')}".strip()
    return Event("pulse", f"pulse:report:{report_row.get('report_id')}", text(read, m, now_ms, basis), price)
