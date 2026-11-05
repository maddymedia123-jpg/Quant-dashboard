"""Market Pulse Dispatch: his Discord template, filled in plain words.

Sent at two moments, one message each (the alert ledger's unique index keeps it to one even with several
pages open):

  * a report is run   - from the report's plan when it has one (direction, Target 1, the stop), and from
                        the Live desk otherwise; keyed on the report id
  * a candle close    - every hour boundary (his Section 10: the 1H, 2H and 4H closes), one message per
                        boundary that names which closes it covers ("1h + 2h + 4h candle closes" at
                        00:00, 04:00, ... UTC; "1h + 2h" on the other even hours; "1h" otherwise); from
                        the Live desk (next 4 hours), or a live plan; keyed on the close

Both are checked on every page load, so with no page open nothing is sent. A close or a report older than
SEND_WITHIN_MS is skipped rather than sent late: a three-hour-old pulse reads as news and is not.

The words are written here from the numbers, never copied from a model's text, and checked against the
same banned list as the plan alerts: the reader is not a trader.

Shape (wording batch 1, 6 Nov; the owner agreed the whole rewrite): six lines, no rule lines, no section
labels. The header carries the closes and the price; then the read (from the desk, or the plan's own line,
where price stands and the one thing to do); the busiest hours with UTC and PKT; the news, prints that
share a minute on one line and a headline dropped when its dollar figure is more than 2% from the price;
and the time."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from core.alerts import Event

HOUR_MS = 3_600_000
CLOSE_MS = HOUR_MS                       # the pulse cadence: every hour boundary (was every 4 hours)
CLOSE_NAMES = ((1, "1h"), (2, "2h"), (4, "4h"))
SEND_WITHIN_MS = 30 * 60_000
NEWS_WITHIN_MS = 6 * HOUR_MS             # headlines from the last six hours
EVENTS_WITHIN_MS = 24 * HOUR_MS          # US prints in the next day: High and Medium impact (fix 4, 5 Nov)
PRINTED_WITHIN_MS = 24 * HOUR_MS         # high-impact US prints of the last day, with the engine's read
PRINT_IMPACT = "Medium"                  # the least impact a scheduled print needs to be named
MARKET_WIDE = ("macro", "geopolitics", "etf")   # headline topics that move every market, BTC named or not
FED_WITHIN_MS = 7 * 24 * HOUR_MS
FED_WORDS = ("fomc", "federal funds rate", "fed chair")
STALE_HEADLINE_PCT = 0.02                # a headline's dollar figure this far from the price is old news
_DOLLARS = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s?([KkMmBb])?")
DIRECTION_WORD = {"BULLISH": "BULLISH", "BEARISH": "BEARISH", "NEUTRAL": "RANGE-BOUND"}
CONF_WORD = lambda c: ("very strong" if c >= 0.85 else "strong" if c >= 0.60  # noqa: E731
                       else "moderate" if c >= 0.35 else "slight")


def _money(x: float | None) -> str:
    return "—" if x is None else f"${x:,.0f}"


def _when(ms: int) -> str:
    from core.trades import fmt_when
    return fmt_when(ms)


def _hhmm(ms: int, shift_s: int = 0) -> str:
    return f"{datetime.fromtimestamp(ms / 1000 + shift_s, timezone.utc):%H:%M}"


def _join(names: list[str]) -> str:
    """'A', 'A and B', 'A, B and C'."""
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


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
        happening = (f"Leaning higher for the next 4 hours ({CONF_WORD(d.confidence)}). Nearest ceiling {_money(ceiling)}; "
                     f"holding above {_money(floor)} keeps buyers in control.")
        target = _target(a, True, floor, ceiling)
    elif direction == "BEARISH":
        happening = (f"Leaning lower for the next 4 hours ({CONF_WORD(d.confidence)}). Nearest floor {_money(floor)}; "
                     f"staying below {_money(ceiling)} keeps sellers in control.")
        target = _target(a, False, floor, ceiling)
    else:
        happening = (f"No clear side for the next 4 hours: stuck between {_money(floor)} and {_money(ceiling)} "
                     "until one of them breaks.")
        target = f"{_money(floor)} or {_money(ceiling)}, whichever breaks first"
    if direction != "RANGE-BOUND" and d.invalidation is not None:
        safety = f"{_money(d.invalidation)} (a 1-hour close {'below' if direction == 'BULLISH' else 'above'} it ends this view)"
        levels = f"Target {target} · stop line {safety}."
    elif floor and ceiling:
        safety = f"a 1-hour close below {_money(floor)} or above {_money(ceiling)} picks a side"
        levels = f"Target {target}; {safety}."
    else:
        safety = "no clear line yet"
        levels = f"Target {target} · stop line: {safety}."
    return {"direction": direction, "happening": happening, "target": target, "safety": safety,
            "lines": [happening, levels]}


def _plan(report_row: dict | None) -> dict | None:
    try:
        plan = json.loads((report_row or {}).get("plan") or "null")
    except (TypeError, ValueError):
        return None
    return plan if plan and plan.get("side") in ("LONG", "SHORT") else None


def plan_live(report_row: dict | None, now_ms: int) -> bool:
    """Whether a report's plan is still the one being traded: it has a side, its hours have not run out,
    and the ledger has not closed it (target, stop, cancelled, not triggered). While it is, every pulse
    speaks from it, so the close message and the report message name the same target and the same safety
    line; the minute the ledger closes it, the pulse speaks from the Live desk again."""
    plan = _plan(report_row)
    if plan is None:
        return False
    if (report_row.get("plan_outcome") or "open") != "open":
        return False
    expires = int(report_row.get("ts_ms") or 0) + int(float(plan.get("expires_hours") or 24) * 3_600_000)
    return now_ms < expires


def plan_read(report_row: dict, price: float | None, now_ms: int | None = None, frames: dict | None = None) -> dict | None:
    """The same fields from a report's plan, when it has one with a side: the plan's own line (made when,
    buy or sell where, target, stop, until when), where price stands, the one thing to do. With the candles
    (`frames`) and the time, it says whether the zone has been touched since the report: the candle high
    (a short) or low (a long), then where price is now - the WATCH alert's own words
    (core.plan_alerts.zone_touch)."""
    plan = _plan(report_row)
    if plan is None:
        return None
    from core.plan_alerts import TF_WORDS, regime_line, zone_touch

    short = plan["side"] == "SHORT"
    targets = sorted((float(t) for t in plan.get("targets") or []), reverse=short)
    low, high = float(plan["zone_low"]), float(plan["zone_high"])
    zone = f"{_money(low)}–{_money(high)}"
    hours = float(plan.get("expires_hours") or 24)
    made_ms = int(report_row.get("ts_ms") or 0)
    tf = plan.get("confirm_timeframe") or "1h"
    level = float(plan["confirm_close"]) if plan.get("confirm_close") is not None else (low if short else high)
    # the confirming close, spelled out from the plan's own numbers (the same rule the WATCH alert sends)
    confirm = f"a {TF_WORDS.get(tf, tf)} candle closes {'below' if short else 'above'} {_money(level)}"
    if plan.get("taker_ratio_limit") is not None:
        confirm += f" with {'sellers' if short else 'buyers'} doing more of the trading"
    act = "sell" if short else "buy"
    target = _money(targets[0]) if targets else "—"
    stop = _money(float(plan["stop"]))
    plan_line = (f"Plan from the {_hhmm(made_ms)} UTC report: {act} {zone}, target {target}, stop {stop}, "
                 f"until {_when(made_ms + int(hours * HOUR_MS))}.")
    touch = zone_touch(plan, made_ms, frames) if frames else None
    # where price stands against the zone decides the first words: a short entered above price must say
    # "wait for the rise", or a reader shorts at the market, the one place the plan says not to
    if price is not None and low <= price <= high:
        where = f"BTC is {_money(price)}, inside the {act} zone {zone}."
        step = f"{act.capitalize()} only if {confirm}. Not before."
        now_where = "inside the zone"
    elif price is not None and (price < low if short else price > high):
        where = f"BTC is {_money(price)}, {'below' if short else 'above'} the {act} zone {zone}."
        step = f"Do not {act} here: wait for a {'rise' if short else 'dip'} into the zone, then {act} only if {confirm}."
        now_where = f"back {'below' if short else 'above'} the zone"
    else:
        where = f"BTC is {_money(price)}; the plan {act}s from the {zone} zone."
        step = f"{act.capitalize()} only if {confirm}. Not before."
        now_where = f"{'above' if short else 'below'} the zone" if price is not None else ""
    if touch is not None and price is not None:
        # 4 Nov 21:26: "BTC is $80,088, below the sell zone" an hour after the candle high had touched it.
        # Touched, the candle's extreme, then the current price: the WATCH alert's words.
        where = (f"The {act} zone {zone} was touched at {_when(touch.at_ms)}: the candle "
                 f"{'high' if short else 'low'} reached {_money(touch.extreme)}. BTC is now {_money(price)}, {now_where}.")
    lines = [plan_line, where, step]
    if regime_line(plan):
        r, lev = plan.get("risk_pct"), plan.get("max_leverage")
        size = f" (risk {float(r):g}%, max {float(lev):g}x)" if r is not None and lev is not None else ""
        lines.append(f"This plan was made inside the {plan['event_regime']} event regime: size halved{size}. "
                     "The no-range-plan rule below is the day plan's, not this trade's.")
    return {"direction": "BEARISH" if short else "BULLISH", "happening": " ".join(lines[1:]),
            "target": target, "safety": stop, "lines": lines}


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


def _reaction(m, event: dict, now_ms: int) -> str:
    """The macro engine's read after a print, in its own words, or a plain line when it has none."""
    try:
        from core.indicators.event_trap import read_all
        reads = read_all(m.spot.frames.get("1h"), [event], now_ms)
    except Exception:  # noqa: BLE001 - the printed line stands without it
        reads = []
    r = reads[0] if reads else None
    if r is None or r.state in ("positioned", "flat", "scheduled", "no_data"):
        return "the reaction is still being traded."
    return r.headline


def _printed(cal, now_ms: int) -> list[dict]:
    """High-impact US prints of the last PRINTED_WITHIN_MS, newest first."""
    out = [e for e in getattr(cal, "events", None) or []
           if e.get("time_ms") is not None and now_ms - PRINTED_WITHIN_MS <= e["time_ms"] <= now_ms
           and e.get("impact") == "High" and e.get("country") == "USD"]
    return sorted(out, key=lambda e: -e["time_ms"])


def _fed_meeting_soon(cal, now_ms: int) -> bool:
    return any(any(w in str(e.get("title", "")).lower() for w in FED_WORDS)
               for e in cal.upcoming(now_ms, FED_WITHIN_MS))


def _stale(title: str, price: float | None) -> bool:
    """5 Nov 17:00: 'Bitcoin holds above $80,000' went out while BTC was $76,358. A headline whose dollar
    figure (a price, not a flow in millions or billions) is more than STALE_HEADLINE_PCT from the price is
    old news."""
    if price is None:
        return False
    for num, suffix in _DOLLARS.findall(title):
        if suffix in ("M", "m", "B", "b"):
            continue
        value = float(num.replace(",", "")) * (1000 if suffix else 1)
        if value >= 1000 and abs(value - price) / price > STALE_HEADLINE_PCT:
            return True
    return False


def _news(m, now_ms: int) -> str:
    """From the Macro & News screen's own data, in order: the event regime when one is active (an election,
    an FOMC: fix 3), the high-impact US prints of the last day with the macro engine's read of the
    reaction, the US prints of the next day, High and Medium (Medium labelled; fix 4: on 4 Nov ISM Services
    was Medium and went unnamed), with the engine's read when BTC has run ahead of one, headlines that name
    Bitcoin or move every market, and the Fed odds when a Fed meeting is near. A hack on a small chain or a
    crypto-policy story abroad is not a BTC warning, so headlines are kept only when they name BTC or fall
    in a market-wide topic, and not when their dollar figure is far from the price (_stale)."""
    from core.indicators.event_regime import read as regime_read
    lines = []
    price = getattr(getattr(m, "spot", None), "last", None)
    cal = getattr(m, "calendar", None)
    if getattr(cal, "available", False):
        regime = regime_read(getattr(cal, "events", None) or [], now_ms)
        if regime.active:
            # the rule is the Quant Forecast's (core.quant_forecast.build_plan, event=); a report's plan made
            # inside the regime carries its own note above (core.agents.one_agent.tag_plan)
            lines.append(f"🚨 EVENT REGIME: {regime.note}. Expect sharp moves either way: the day plan's rule is "
                         "no range plan, wider safety lines, smaller size. That rule is the Quant Forecast's; a "
                         "report's trade plan is sized by its own regime note.")
        for e in [p for p in _printed(cal, now_ms) if p.get("title") != regime.title][:2]:
            lines.append(f"US {e.get('title', 'release')} printed at {_when(e['time_ms'])}: {_reaction(m, e, now_ms)}")
        ahead = sorted([e for e in cal.upcoming(now_ms, EVENTS_WITHIN_MS, min_impact=PRINT_IMPACT) if e["time_ms"] > now_ms],
                       key=lambda e: e["time_ms"])
        # prints that share a minute (5 Nov: Non-Farm Employment Change, Unemployment Rate and Hourly Earnings
        # at 13:30) are one line, not three identical ones
        groups: list[list[dict]] = []
        for e in ahead:
            if groups and groups[-1][0]["time_ms"] == e["time_ms"]:
                groups[-1].append(e)
            else:
                groups.append([e])
        for g in groups[:3]:
            names = [f"{e.get('title', 'release')}{'' if e.get('impact') == 'High' else ' (medium impact)'}" for e in g]
            high = any(e.get("impact") == "High" for e in g)
            lines.append(f"US {_join(names)} at {_when(g[0]['time_ms'])}: expect {'sharp moves' if high else 'a reaction'} "
                         f"around {'them' if len(g) > 1 else 'it'}." + _positioning(m, g[0], now_ms))
    news = getattr(m, "news", None)
    if getattr(news, "available", False):
        heads = [h for h in news.recent(now_ms, NEWS_WITHIN_MS, "high") if h.get("title")
                 and ("BTC" in (h.get("assets") or []) or h.get("topic") in MARKET_WIDE)
                 and not _stale(h["title"], price)]
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


def busiest_line(now_ms: int) -> str:
    """The next turning window with UTC and PKT: 'we are inside them now', 'later today' or 'tomorrow'
    (29 Oct: a 21:00 PKT message read 'between 17:00 and 21:00 PKT', hours that had just ended)."""
    from core.indicators.turn_window import PKT_S, next_window
    w = next_window(now_ms)
    span = (f"{_hhmm(w.start_ms)}–{_hhmm(w.end_ms)} UTC "
            f"({_hhmm(w.start_ms, PKT_S)}–{_hhmm(w.end_ms, PKT_S)} PKT)")
    if w.active:
        return f"Busiest hours: {span}: we are inside them now."
    pkt_day = lambda ms: (ms + PKT_S * 1000) // 86_400_000  # noqa: E731
    return f"Busiest hours: {span}, {'tomorrow' if pkt_day(w.start_ms) > pkt_day(now_ms) else 'later today'}."


def text(read: dict, m, now_ms: int, basis: str, price: float | None) -> str:
    """Header (the closes, the price), the read's lines, the busiest hours, the news, the time."""
    return "\n".join([f"🤖 PULSE {basis} · BTC {_money(price)}", *read["lines"], busiest_line(now_ms),
                       _news(m, now_ms), _when(now_ms)])


def closes_at(close_ms: int) -> list[str]:
    """Which candles closed on this hour boundary: the 1h always; the 2h on even UTC hours; the 4h at 00:00,
    04:00, 08:00, 12:00, 16:00 and 20:00 UTC (the exchange's candles are UTC-aligned)."""
    hours = close_ms // HOUR_MS
    return [name for n, name in CLOSE_NAMES if hours % n == 0]


def close_label(close_ms: int) -> str:
    return " + ".join(closes_at(close_ms)) + " close"


def close_key(close_ms: int) -> str:
    """The ledger key: the coarsest candle that closed. A 4h boundary keeps its old key, so the deploy that
    added the hourly cadence could not send a second message for a close already sent."""
    return f"pulse:{closes_at(close_ms)[-1]}:{close_ms}"


def close_event(live, m, now_ms: int, plan_row: dict | None = None) -> Event | None:
    """The pulse for the candle(s) that just closed on the last hour boundary, while that close is still
    fresh. With a live plan it speaks from the plan (the report's zone, target and stop); otherwise from
    the Live desk. The basis line names every candle that closed, so the reader knows which closes it covers."""
    close = now_ms - now_ms % CLOSE_MS
    if now_ms - close > SEND_WITHIN_MS:
        return None
    price = getattr(m.spot, "last", None)
    if price is None:
        price = getattr(live, "price", None)
    label, key = close_label(close), close_key(close)
    if plan_live(plan_row, now_ms):
        read = plan_read(plan_row, price, now_ms, getattr(m.spot, "frames", None))
        if read is not None:
            return Event("pulse", key, text(read, m, now_ms, label, price), price)
    if live is None:
        return None
    return Event("pulse", key, text(desk_read(live), m, now_ms, label, price), price)


def report_event(report_row: dict | None, live, m, now_ms: int) -> Event | None:
    """The pulse for a new report that carries a plan with a side, once, while it is fresh. A report
    without one (a Director that wrote none, or a 'stand aside') adds nothing the 4h-close pulse does not
    already say, so it sends nothing: a compare run is one dispatch, not two."""
    if not report_row or now_ms - int(report_row.get("ts_ms") or 0) > SEND_WITHIN_MS:
        return None
    price = getattr(m.spot, "last", None)
    read = plan_read(report_row, price, now_ms, getattr(m.spot, "frames", None))
    if read is None:
        return None
    return Event("pulse", f"pulse:report:{report_row.get('report_id')}", text(read, m, now_ms, "new report", price), price)
