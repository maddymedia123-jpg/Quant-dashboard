"""Plan alerts: the latest report's plan, watched against real candles, told in plain words.

The alerts used to report the desks' internal state ("Strong setup down -> Pause in a downtrend"). A
trader wants something else: what to watch, when to look again, when the setup is on, and when it is
over. Those all follow from the plan's numbers, so they are derived here from candles:

  WATCH      price reached the entry zone - what confirms it, and when to look again
  CONFIRMED  a closed candle on the plan's timeframe closed beyond the confirm level (and, when the plan
             sets one, the taker ratio agreed)
  NO SETUP   before confirming, a candle closed through the far side of the zone, or the stop traded
  TARGET n   after confirming, each target as price reaches it
  STOPPED    after confirming, the stop
  EXPIRED    the plan's time ran out without a confirmed entry

The sequence is replayed from the report's time on every pass, so it is the same however often the page
loads, and each step has a fixed dedup key: the alert ledger's unique index makes each one a single
message. Only the newest plan is watched (either pipeline's since 5 November); a new report replaces it. A step older than
STALE_AFTER_MS is not announced as news - with no page open for hours, a five-hour-old WATCH helps no one.

Wording batch 4 (6 Nov, the owner agreed the rewrite): the same verbs on every step - "Buy only if" /
"Sell only if", "Forget it if", "Stop … · targets …" - the WATCH carries the targets and no desk context,
CONFIRMED is "✅ SELL NOW" with the percent distance to the stop and the targets and the expiry, NO SETUP
is "⚪ NO TRADE · short off" with a price and the next step, TARGET says "1 OF 3", STOPPED OUT names the
loss, and every end state says "Next: the hourly Pulse, or a new report". The report is named by its time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from core.alerts import Event

TF_MS = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
TF_WORDS = {"15m": "15-minute", "1h": "1-hour", "4h": "4-hour", "1d": "daily"}
# never in a bot message: the reader is not a trader (his Section 8)
BANNED_WORDS = ("gamma", "cvd", "rsi", "fibonacci", "delta", "option", "vwap", "amt", "taker", "ema",
                "liquidity", "sweep", "vah", "val ", "order flow", "funding", "open interest", "squeeze")
STALE_AFTER_MS = 2 * 3_600_000
LATE_AFTER_MS = 15 * 60_000


def _money(x: float | None) -> str:
    return "—" if x is None else f"${x:,.0f}"


def _when(ms: int) -> str:
    from core.trades import fmt_when
    return fmt_when(ms)


def _hhmm(ms: int) -> str:
    from datetime import datetime, timezone
    return f"{datetime.fromtimestamp(ms / 1000, timezone.utc):%H:%M} UTC"


def _pct(a: float, b: float) -> str:
    return f"{abs(a - b) / b * 100:.1f}%"


@dataclass
class Step:
    name: str            # watch, confirmed, cancelled, target1.., stopped, expired
    at_ms: int
    headline: str
    price: float | None


@dataclass(frozen=True)
class Touch:
    """The first 15-minute candle that reached the zone, and how far it went: the candle high for a short
    (price rose into the zone), the candle low for a long."""
    at_ms: int
    extreme: float
    side: str


def zone_touch(plan: dict, ts_ms: int, frames: dict) -> Touch | None:
    """The first 15m candle since the report that reached the zone before the plan expires (a forming
    candle counts), or None. Shared by the WATCH alert and the Pulse, so both name the same touch and the
    same candle extreme."""
    side = plan.get("side")
    if side not in ("LONG", "SHORT"):
        return None
    fine = (frames or {}).get("15m")
    if fine is None or not len(fine):
        return None
    low, high = float(plan["zone_low"]), float(plan["zone_high"])
    expires = ts_ms + int(float(plan.get("expires_hours") or 24) * 3_600_000)
    for r in fine.itertuples():
        t = int(r.timestamp)
        if t + 900_000 > ts_ms and t < expires and r.high >= low and r.low <= high:
            return Touch(t, float(r.high if side == "SHORT" else r.low), side)
    return None


def _stands(price_now: float | None, low: float, high: float, short: bool) -> str:
    """Where price is now against the zone, after the touch: inside, back on the side it came from, or
    through the zone toward the stop."""
    if price_now is None:
        return ""
    if low <= price_now <= high:
        where = "inside the zone"
    elif price_now < low:
        where = "back below the zone" if short else "below the zone"
    else:
        where = "above the zone" if short else "back above the zone"
    return f" BTC is now {_money(price_now)}, {where}."


def regime_line(plan: dict) -> str:
    """The line a plan tagged with an event regime carries (core.agents.one_agent.tag_plan): the Pulse's
    regime line says 'no range plan, smaller size', and on 4 November a report's short sent the same hour
    read as its contradiction. The rule is the day plan's; this plan was sized by the regime in code."""
    title = plan.get("event_regime")
    if not title:
        return ""
    r, lev = plan.get("risk_pct"), plan.get("max_leverage")
    size = f" (risk {float(r):g}%, max {float(lev):g}x)" if r is not None and lev is not None else ""
    return (f"\nEvent regime ({title}): the report made this plan inside it, with size halved{size}. "
            "The no-range-plan rule belongs to the day plan, not to this trade.")


def _taker_ratio(flow_frame, open_ms: int) -> float | None:
    if flow_frame is None or not len(flow_frame):
        return None
    row = flow_frame[flow_frame["timestamp"] == open_ms]
    if not len(row):
        return None
    buy, sell = float(row["buy"].iloc[0]), float(row["sell"].iloc[0])
    return buy / sell if sell > 0 else None


def replay(plan: dict, ts_ms: int, frames: dict, flows: dict | None, now_ms: int, price_now: float | None) -> list[Step]:
    """Every step this plan has reached so far, in order."""
    side = plan.get("side")
    if side not in ("LONG", "SHORT"):
        return []
    short = side == "SHORT"
    low, high, stop = float(plan["zone_low"]), float(plan["zone_high"]), float(plan["stop"])
    targets = sorted((float(t) for t in plan["targets"]), reverse=short)       # nearest first
    tf = plan.get("confirm_timeframe") or "1h"
    bar = TF_MS.get(tf, 3_600_000)
    confirm = plan.get("confirm_close")
    level = float(confirm) if confirm is not None else (low if short else high)
    ratio_limit = plan.get("taker_ratio_limit")
    expires = ts_ms + int(float(plan.get("expires_hours") or 24) * 3_600_000)
    fine, conf = frames.get("15m"), frames.get(tf)
    flow = (flows or {}).get(tf)
    if fine is None or conf is None or not len(fine):
        return []
    word, beyond, far = ("short", "below", "above") if short else ("long", "above", "below")
    act = "sell" if short else "buy"
    candle = TF_WORDS.get(tf, tf)
    zone = f"{_money(low)}–{_money(high)}"
    idea = f"The {act} idea from the {_hhmm(ts_ms)} report"
    next_step = "Next: the hourly Pulse, or a new report."
    steps: list[Step] = []

    # 1. the zone is touched (a forming candle counts: this is the real-time part)
    reached = zone_touch(plan, ts_ms, frames)
    touch = reached.at_ms if reached else None
    if touch is None:
        if now_ms >= expires:
            steps.append(Step("expired", expires, (
                f"⌛ PLAN EXPIRED · BTC {_money(price_now)}\n{idea} ran out at {_when(expires)}: price never reached "
                f"the zone {zone}. No trade.\n{next_step}"), price_now))
        return steps
    next_close = (touch // bar + 1) * bar
    # The rules are written here from the plan's numbers, not copied from the model's own wording: that
    # wording ("4h CVD remains net negative") is exactly the jargon the bot must not send.
    rules = f"a {candle} candle closes {beyond} {_money(level)}"
    if ratio_limit is not None:
        rules += f" with {'sellers' if short else 'buyers'} doing more of the trading"
    from core.indicators.turn_window import next_window
    tw = next_window(touch)
    # "Price reached the zone" under a headline price below it read as a contradiction (4 Nov, 21:00 UTC:
    # the candle high had touched the zone and price was back under it). So: touched, the candle's
    # extreme, then where price is now.
    steps.append(Step("watch", touch, (
        f"🟡 WATCH · BTC {_money(price_now)}\n"
        f"The {act} zone {zone} was touched: the candle {'high' if short else 'low'} reached "
        f"{_money(reached.extreme)} at {_when(touch)}.{_stands(price_now, low, high, short)}\n"
        f"{act.capitalize()} only if {rules}. Next close: {_when(next_close)}.\n"
        f"Forget it if a {candle} candle closes {far} {_money(high if short else low)}. "
        f"Stop {_money(stop)} · targets " + " / ".join(_money(t) for t in targets) + ".\n"
        + ("We are inside the busiest turning hours now." if tw.active else f"Busiest turning hours: {tw.label}.")
        + regime_line(plan)), price_now))

    # 2. the confirming close, or the setup breaking first
    entry_ms = entry_px = None
    for r in conf.itertuples():
        open_ms = int(r.timestamp)
        close_ms = open_ms + bar
        if close_ms <= touch or close_ms > now_ms or open_ms >= expires:
            continue
        if (short and r.high >= stop) or (not short and r.low <= stop):
            steps.append(Step("cancelled", close_ms, (
                f"⚪ NO TRADE · {word} off · BTC {_money(stop)}\nPrice went through the stop {_money(stop)} before any "
                f"confirmation. {idea} is over.\n{next_step}"), stop))
            return steps
        if (short and r.close > high) or (not short and r.close < low):
            steps.append(Step("cancelled", close_ms, (
                f"⚪ NO TRADE · {word} off · BTC {_money(float(r.close))}\nA {candle} candle closed at {_money(float(r.close))}, "
                f"{far} the zone {zone}, before any confirmation. {idea} is over.\n{next_step}"), float(r.close)))
            return steps
        if (short and r.close < level) or (not short and r.close > level):
            ratio = _taker_ratio(flow, open_ms)
            if ratio_limit is not None and ratio is not None and (
                    (short and ratio >= ratio_limit) or (not short and ratio <= ratio_limit)):
                continue                                   # the close is there, the flow is not: keep waiting
            entry_ms, entry_px = close_ms, float(r.close)
            who = ("sellers" if ratio < 1 else "buyers") if ratio is not None else None
            flow_note = (f", with {who} doing more of the trading" if who else
                         " (the buyer/seller check was not available for this candle)" if ratio_limit is not None
                         else "")
            past = entry_px < low if short else entry_px > high
            steps.append(Step("confirmed", close_ms, (
                f"✅ {act.upper()} NOW · BTC {_money(entry_px)}\n"
                f"A {candle} candle closed at {_money(entry_px)}, {beyond} {_money(level)}{flow_note}.\n"
                f"Stop {_money(stop)} ({_pct(stop, entry_px)} away) · targets " + " / ".join(_money(t) for t in targets)
                + " (" + " / ".join(_pct(t, entry_px) for t in targets) + ").\n"
                f"Entry was planned at {zone}; price is "
                + ("past it, so the risk is the distance to the stop from here." if past else "inside it.") + "\n"
                f"Good until {_when(expires)}."), entry_px))
            break
    if entry_ms is None:
        if now_ms >= expires + bar:
            steps.append(Step("expired", expires, (
                f"⌛ PLAN EXPIRED · BTC {_money(price_now)}\n{idea} ran out at {_when(expires)}: the zone was touched "
                f"but no {candle} candle confirmed. No trade.\n{next_step}"), price_now))
        return steps

    # 3. after the entry: targets in order, or the stop (a candle spanning both counts as the stop)
    hit = 0
    for r in fine.itertuples():
        t = int(r.timestamp)
        if t < entry_ms:
            continue
        if (short and r.high >= stop) or (not short and r.low <= stop):
            loss = abs(stop - entry_px) / entry_px * 100
            steps.append(Step("stopped", t, (
                f"🔴 STOPPED OUT · BTC {_money(stop)}\n{word.capitalize()} from {_money(entry_px)} hit the stop "
                f"{_money(stop)}: {loss:.2f}% loss. {idea} was wrong.\n{next_step} No re-entry from this plan."), stop))
            return steps
        while hit < len(targets) and ((short and r.low <= targets[hit]) or (not short and r.high >= targets[hit])):
            tgt = targets[hit]
            hit += 1
            move = (entry_px - tgt) / entry_px * 100 if short else (tgt - entry_px) / entry_px * 100
            nxt = f"Next target {_money(targets[hit])}." if hit < len(targets) else "That was the last target: the plan is done."
            breakeven = f" and move the stop to {_money(entry_px)} (break-even)" if hit == 1 else ""
            steps.append(Step(f"target{hit}", t, (
                f"🎯 TARGET {hit} OF {len(targets)} HIT · BTC {_money(tgt)}\n{word.capitalize()} from "
                f"{_money(entry_px)}: {move:+.2f}% in your favour.\n{nxt}\nSuggested: take some profit here{breakeven}."), tgt))
        if hit == len(targets):
            return steps
    return steps


def events(report_row: dict | None, frames: dict, flows: dict | None, now_ms: int, price_now: float | None) -> list[Event]:
    """The alert events for the newest plan; old steps are left out as stale."""
    if not report_row or not report_row.get("plan"):
        return []
    plan = json.loads(report_row["plan"])
    rid = report_row["report_id"]
    out = []
    for s in replay(plan, int(report_row["ts_ms"]), frames, flows, now_ms, price_now):
        if now_ms - s.at_ms > STALE_AFTER_MS and s.name in ("watch",):
            continue                     # a WATCH hours later is history, not a call to look
        kind = {"confirmed": "plan_confirmed", "stopped": "plan_stopped"}.get(s.name, "plan")
        late = now_ms - s.at_ms > LATE_AFTER_MS
        # with no page open the alert can go out late: then it says when it actually happened
        out.append(Event(kind, f"plan:{rid}:{s.name}",
                         s.headline + (f"\n(This happened at {_when(s.at_ms)}.)" if late else ""), s.price))
    return out


def latest_plan_report(store) -> dict | None:
    """The newest report that carries a plan, from either pipeline (the classic Director writes the same
    numbers plan since 5 November): a new report replaces the plan being watched, a stand-aside included.
    A report saved without one (a Director that failed, a report from before plans existed) leaves the
    plan before it alone."""
    for r in store.reports(limit=50):
        if r.get("plan"):
            return r
    return None
