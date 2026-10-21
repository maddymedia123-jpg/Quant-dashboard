"""Plan alerts: the latest 1-agent report's plan, watched against real candles, told in plain words.

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
message. Only the newest 1-agent plan is watched; a new report replaces it. A step older than
STALE_AFTER_MS is not announced as news - with no page open for hours, a five-hour-old WATCH helps no one.
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


@dataclass
class Step:
    name: str            # watch, confirmed, cancelled, target1.., stopped, expired
    at_ms: int
    headline: str
    price: float | None


def _taker_ratio(flow_frame, open_ms: int) -> float | None:
    if flow_frame is None or not len(flow_frame):
        return None
    row = flow_frame[flow_frame["timestamp"] == open_ms]
    if not len(row):
        return None
    buy, sell = float(row["buy"].iloc[0]), float(row["sell"].iloc[0])
    return buy / sell if sell > 0 else None


def replay(plan: dict, ts_ms: int, frames: dict, flows: dict | None, now_ms: int, price_now: float | None,
           classification: str = "", context: str = "") -> list[Step]:
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
    zone = f"{_money(low)}–{_money(high)}"
    steps: list[Step] = []

    # 1. the zone is reached (a forming candle counts: this is the real-time part)
    touch = next((int(r.timestamp) for r in fine.itertuples()
                  if int(r.timestamp) + 900_000 > ts_ms and int(r.timestamp) < expires
                  and r.high >= low and r.low <= high), None)
    if touch is None:
        if now_ms >= expires:
            steps.append(Step("expired", expires, f"⌛ PLAN EXPIRED · {word}\nPrice never reached the zone ({zone}) "
                                                  f"before {_when(expires)}. No trade.", price_now))
        return steps
    next_close = (touch // bar + 1) * bar
    # The rules are written here from the plan's numbers, not copied from the model's own wording: that
    # wording ("4h CVD remains net negative") is exactly the jargon the bot must not send.
    rules = f"a {TF_WORDS.get(tf, tf)} candle closes {beyond} {_money(level)}"
    if ratio_limit is not None:
        rules += f", with {'sellers' if short else 'buyers'} doing more of the trading on that candle"
    kind_word = {"BULL_TRAP": "fake breakout up", "BEAR_TRAP": "fake breakdown", "RANGE_TRAP": "range",
                 "NO_TRAP": "trend"}.get(classification, "latest")
    from core.indicators.turn_window import next_window
    tw = next_window(touch)
    steps.append(Step("watch", touch, (
        f"🟡 WATCH · BTC {_money(price_now)}\n"
        f"Price reached the {word} zone ({zone}) from the {kind_word} report.\n"
        f"Next check: {TF_WORDS.get(tf, tf)} close at {_when(next_close)}.\n"
        f"{word.capitalize()} only if: {rules}.\n"
        f"Cancelled if a {TF_WORDS.get(tf, tf)} candle closes {far} {_money(high if short else low)}. "
        f"Plan wrong past {_money(stop)}.\n"
        + ("We are inside the busiest turning hours of the day right now." if tw.active
           else f"Busiest turning hours: {tw.label}.")
        + (f"\nContext: {context}" if context else "")), price_now))

    # 2. the confirming close, or the setup breaking first
    entry_ms = entry_px = None
    for r in conf.itertuples():
        open_ms = int(r.timestamp)
        close_ms = open_ms + bar
        if close_ms <= touch or close_ms > now_ms or open_ms >= expires:
            continue
        if (short and r.high >= stop) or (not short and r.low <= stop):
            steps.append(Step("cancelled", close_ms, f"⚪ NO SETUP · {word} cancelled\nPrice traded through the "
                                                     f"stop level {_money(stop)} before any confirmation.", stop))
            return steps
        if (short and r.close > high) or (not short and r.close < low):
            steps.append(Step("cancelled", close_ms, f"⚪ NO SETUP · {word} cancelled\nThe {TF_WORDS.get(tf, tf)} candle closed at "
                                                     f"{_money(r.close)}, {far} the zone ({zone}): no rejection.",
                              float(r.close)))
            return steps
        if (short and r.close < level) or (not short and r.close > level):
            ratio = _taker_ratio(flow, open_ms)
            if ratio_limit is not None and ratio is not None and (
                    (short and ratio >= ratio_limit) or (not short and ratio <= ratio_limit)):
                continue                                   # the close is there, the flow is not: keep waiting
            entry_ms, entry_px = close_ms, float(r.close)
            who = ("sellers" if ratio < 1 else "buyers") if ratio is not None else None
            flow_note = (f" ({who} did more of the trading)" if who else
                         " (the buyer/seller check was not available for this candle)" if ratio_limit is not None
                         else "")
            steps.append(Step("confirmed", close_ms, (
                f"✅ SETUP CONFIRMED · {word} · BTC {_money(entry_px)}\n"
                f"The {TF_WORDS.get(tf, tf)} candle closed at {_money(entry_px)}, {beyond} {_money(level)}{flow_note}.\n"
                f"Entry zone {zone} · stop {_money(stop)} · targets " + " / ".join(_money(t) for t in targets)),
                entry_px))
            break
    if entry_ms is None:
        if now_ms >= expires + bar:
            steps.append(Step("expired", expires, f"⌛ PLAN EXPIRED · {word}\nPrice reached the zone but no {TF_WORDS.get(tf, tf)} "
                                                  f"candle confirmed before {_when(expires)}. No trade.", price_now))
        return steps

    # 3. after the entry: targets in order, or the stop (a candle spanning both counts as the stop)
    hit = 0
    for r in fine.itertuples():
        t = int(r.timestamp)
        if t < entry_ms:
            continue
        if (short and r.high >= stop) or (not short and r.low <= stop):
            move = (entry_px - stop) / entry_px * 100 if short else (stop - entry_px) / entry_px * 100
            steps.append(Step("stopped", t, (
                f"🔴 STOPPED · {word} from {_money(entry_px)}\nPrice reached the stop {_money(stop)} "
                f"({move:+.2f}%). The {classification.replace('_', ' ').lower() or 'plan'} call was wrong this time."),
                stop))
            return steps
        while hit < len(targets) and ((short and r.low <= targets[hit]) or (not short and r.high >= targets[hit])):
            tgt = targets[hit]
            hit += 1
            move = (entry_px - tgt) / entry_px * 100 if short else (tgt - entry_px) / entry_px * 100
            nxt = f" Next: {_money(targets[hit])}." if hit < len(targets) else " That was the last target."
            steps.append(Step(f"target{hit}", t, f"🎯 TARGET {hit} HIT · {_money(tgt)}\n{word.capitalize()} from "
                                                 f"{_money(entry_px)}: {move:+.2f}%.{nxt}", tgt))
        if hit == len(targets):
            return steps
    return steps


def events(report_row: dict | None, frames: dict, flows: dict | None, now_ms: int, price_now: float | None,
           context: str = "") -> list[Event]:
    """The alert events for the newest 1-agent plan; old steps are left out as stale."""
    if not report_row or not report_row.get("plan"):
        return []
    plan = json.loads(report_row["plan"])
    rid = report_row["report_id"]
    out = []
    for s in replay(plan, int(report_row["ts_ms"]), frames, flows, now_ms, price_now,
                    report_row.get("classification") or "", context):
        if now_ms - s.at_ms > STALE_AFTER_MS and s.name in ("watch",):
            continue                     # a WATCH hours later is history, not a call to look
        kind = {"confirmed": "plan_confirmed", "stopped": "plan_stopped"}.get(s.name, "plan")
        late = now_ms - s.at_ms > LATE_AFTER_MS
        # with no page open the alert can go out late: then it says when it actually happened
        out.append(Event(kind, f"plan:{rid}:{s.name}",
                         s.headline + (f"\n(This happened at {_when(s.at_ms)}.)" if late else ""), s.price))
    return out


def latest_plan_report(store) -> dict | None:
    """The newest 1-agent report that carries a plan: a new report replaces the plan being watched."""
    for r in store.reports(limit=50):
        if (r.get("mode") or "15-agent") == "1-agent":
            return r if r.get("plan") else None
    return None
