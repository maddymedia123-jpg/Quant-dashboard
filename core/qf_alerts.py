"""Quant Forecast audit and alerts: the day's plan watched on every pass (the 24/7 worker's, and the page's).

The tiers, as agreed with Hammad (28 Oct):

* CAUTION (tier 1) - a closed 15-minute candle moves more than CAUTION_SIGMA times that hour's expected
  15-minute move against the plan, or the last hour's taker delta turns against it at CVD_FLIP_X times
  its usual size, while invalidation holds. On a neutral plan any such move counts, either way. At most
  one caution an hour.
* MODEL HELD (tier 2B) - after a shock like that, price takes back at least half of it within
  HOLD_WINDOW_MS without breaking invalidation.
* INVALIDATED (tier 3) - a decisive breach of the whole-day invalidation level
  (core.quant_forecast.invalidation_hit: a 4h close or two hourly closes in a row beyond the level; the
  single-candle tests were retired 30 Oct). The plan is replaced by Scenario B - a new forecast for the rest of the day, drawn
  on a secondary chart under the voided one; the LOCK_AFTER_INVALIDATIONS-th failure locks the day instead.
* The magnitude scaler (core.quant_forecast.magnitude_scale) runs on every pass that finds the plan intact
  and widens its target once when open interest or taker flow shows more force than the simulation gave it.
  Every message carries the compact timeline (which window is active, which is next) - message 6.

Detection is read from the candles each pass, so it holds no state of its own; what is new is decided by
the alert ledger's unique keys, and who replaces a plan by the one-statement update in the store. A plan's
row is written before its AI write-up is asked for, so a day can never be left with no active plan by a slow
or failing model, and a pass that finds the day's last plan closed with nothing after it (a crash between
the two steps) builds the replacement then. An alert older than SEND_WITHIN_MS (the worker was down) is
skipped: an hour-old caution is not news."""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone

import pandas as pd

from core import qf_preflight
from core import quant_forecast as qf
from core.alerts import Event

log = logging.getLogger(__name__)

HOUR_MS = qf.HOUR_MS
M15_MS = 900_000
CAUTION_SIGMA = 2.0
CVD_BARS = 4                     # the last hour of 15-minute taker delta
CVD_TYPICAL_ROWS = 96            # its usual size, from the day before
CVD_FLIP_X = 1.5
HOLD_WINDOW_MS = HOUR_MS
SEND_WITHIN_MS = 30 * 60_000
REGIME_WORD = {"bullish": "BULLISH EXPANSION", "bearish": "BEARISH EXPANSION",
               "neutral": "NEUTRAL / RANGE-BOUND", "locked": "LOCKOUT MODE: HIGH CHOP / STAND ASIDE"}


def _money(x) -> str:
    return "—" if x is None else f"${x:,.0f}"


def _clock(ms: int) -> tuple[str, str]:
    u = datetime.fromtimestamp(ms / 1000, timezone.utc)
    k = datetime.fromtimestamp(ms / 1000 + 5 * 3600, timezone.utc)
    return f"{u:%H:%M}", f"{k:%H:%M}"


def at_label(ms: int) -> str:
    """'14:00 UTC (19:00 PKT)'"""
    u, k = _clock(ms)
    return f"{u} UTC ({k} PKT)"


def window_label(start_ms: int, end_ms: int) -> str:
    """'00:00–06:00 UTC (05:00–11:00 PKT)'"""
    (us, ks), (ue, ke) = _clock(start_ms), _clock(end_ms)
    return f"{us}–{ue} UTC ({ks}–{ke} PKT)"


def _against(regime: str) -> int:
    """Which way a move hurts the plan: -1 down hurts a bullish plan, +1 up a bearish one, 0 either way."""
    return {"bullish": -1, "bearish": 1}.get(regime, 0)


def _sigma_15m(plan: dict, t_ms: int) -> float | None:
    """The plan's expected 15-minute move (as a log return) for the hour holding t_ms: half the whole
    hour's sigma (a quarter of its variance). The whole hour's, not the first hour's remaining-minutes
    sigma, which is scaled down for the path and would make an ordinary candle look like a shock."""
    for h in plan.get("hours") or []:
        if t_ms <= h["t_ms"]:
            return h.get("sigma_full_pct", h["sigma_pct"]) / 100 / 2
    return None


def _closed_since(df: pd.DataFrame | None, bar_ms: int, since_ms: int, now_ms: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "close"])
    t = df["timestamp"].astype("int64")
    return df[(t + bar_ms > since_ms) & (t + bar_ms <= now_ms)].reset_index(drop=True)


def shocks(plan: dict, m15: pd.DataFrame | None, now_ms: int) -> list[dict]:
    """Closed 15-minute candles since the plan was made that moved CAUTION_SIGMA times the expected
    15-minute move against it (either way on a neutral plan)."""
    rows = _closed_since(m15, M15_MS, plan["created_ms"], min(now_ms, plan["end_ms"]))
    if m15 is None or rows.empty:
        return []
    allc = m15.set_index(m15["timestamp"].astype("int64"))["close"].astype(float)
    sign = _against(plan["regime"])
    out = []
    for t, close in zip(rows["timestamp"].astype("int64"), rows["close"].astype(float)):
        prev = allc.get(int(t) - M15_MS)
        s = _sigma_15m(plan, int(t) + M15_MS)
        if prev is None or not s or prev <= 0:
            continue
        r = math.log(close / prev)
        size = r * sign if sign else abs(r)
        if size > CAUTION_SIGMA * s:
            out.append({"t_ms": int(t) + M15_MS, "from": round(float(prev), 2), "to": round(close, 2),
                        "sigmas": round(abs(r) / s, 1)})
    return out


def cvd_flips(plan: dict, flow15: pd.DataFrame | None, now_ms: int) -> list[dict]:
    """The moments the last hour's taker delta turned against a directional plan at CVD_FLIP_X its usual
    size (the first 15 minutes of each such stretch only)."""
    sign = _against(plan["regime"])
    if not sign or flow15 is None or flow15.empty or "delta" not in flow15:
        return []
    f = flow15[flow15["timestamp"].astype("int64") + M15_MS <= now_ms].reset_index(drop=True)
    roll = f["delta"].astype(float).rolling(CVD_BARS).sum()
    typical = roll.abs().rolling(CVD_TYPICAL_ROWS, min_periods=24).mean().shift(CVD_BARS)
    against = (roll * sign > 0) & (roll.abs() > CVD_FLIP_X * typical)
    out = []
    for i in range(len(f)):
        t_close = int(f["timestamp"].iloc[i]) + M15_MS
        if t_close <= plan["created_ms"] or t_close > plan["end_ms"]:
            continue
        if bool(against.iloc[i]) and not (i > 0 and bool(against.iloc[i - 1])):
            out.append({"t_ms": t_close, "delta_btc": round(float(roll.iloc[i]), 1)})
    return out


def held_after(shock: dict, m15: pd.DataFrame | None, now_ms: int, invalidated_ms: int | None) -> int | None:
    """When price took back half the shock within HOLD_WINDOW_MS, invalidation intact; else None."""
    if m15 is None:
        return None
    rows = _closed_since(m15, M15_MS, shock["t_ms"], min(now_ms, shock["t_ms"] + HOLD_WINDOW_MS))
    half = shock["to"] + 0.5 * (shock["from"] - shock["to"])
    down = shock["to"] < shock["from"]
    for t, close in zip(rows["timestamp"].astype("int64"), rows["close"].astype(float)):
        t_close = int(t) + M15_MS
        if invalidated_ms is not None and t_close >= invalidated_ms:
            return None
        if (close >= half) if down else (close <= half):
            return t_close
    return None


def audit(plan: dict, frames: dict, flow15: pd.DataFrame | None, now_ms: int) -> dict:
    """Where the active plan stands now: status, the cautions and holds since it was made, and the
    invalidation if price has broken it."""
    if plan.get("regime") == "locked":
        return {"status": "locked", "cautions": [], "holds": [], "invalidation": None}
    h1, m15 = frames.get("1h"), frames.get("15m")
    inv = qf.invalidation_hit(plan, h1, now_ms) if h1 is not None else None
    inv_ms = inv["close_ms"] if inv else None
    cautions = []
    for s in shocks(plan, m15, now_ms):
        if inv_ms is None or s["t_ms"] < inv_ms:
            cautions.append({**s, "why": "move"})
    for c in cvd_flips(plan, flow15, now_ms):
        if inv_ms is None or c["t_ms"] < inv_ms:
            cautions.append({**c, "why": "flow"})
    cautions.sort(key=lambda c: c["t_ms"])
    holds = []
    for s in (c for c in cautions if c["why"] == "move"):
        t = held_after(s, m15, now_ms, inv_ms)
        if t is not None:
            holds.append({"t_ms": t, "shock_ms": s["t_ms"], "from": s["from"], "to": s["to"]})
    last_c = cautions[-1]["t_ms"] if cautions else None
    last_h = holds[-1]["t_ms"] if holds else None
    status = ("invalidated" if inv else
              "model held" if last_h and (last_c is None or last_h >= last_c) and now_ms - last_h <= 4 * HOUR_MS else
              "caution" if last_c and now_ms - last_c <= HOUR_MS else "on track")
    return {"status": status, "cautions": cautions, "holds": holds, "invalidation": inv}


# ---------- the messages ----------

def _inv_text(plan: dict) -> str:
    inv = plan.get("invalidation") or {}
    parts = [f"{'below' if side == 'below' else 'above'} {_money(v)}" for side, v in inv.items()]
    return " or ".join(parts) if parts else "none"


def record_words(plan: dict, live: dict | None) -> list[str]:
    """The record that matters for this plan's kind: direction calls for a directional plan, range days
    for a neutral one, from the backtest and the live days."""
    key = ("range_held", "range_days", "range held") if plan.get("regime") == "neutral" else         ("direction_right", "direction_days", "direction right")
    out = []
    for name, rec in (("backtest", plan.get("backtest") or {}), ("live", live or {})):
        if rec.get(key[1]):
            out.append(f"{name}: {key[2]} {rec[key[0]]} of {rec[key[1]]} days")
    return out


def trap_label(plan: dict) -> str | None:
    """'Trap Agent: Clear' or 'Trap Agent: BULL TRAP active (4H)', from the pre-flight the plan was built with."""
    return ((plan.get("preflight") or {}).get("trap") or {}).get("label")


def regime_line(plan: dict, trap: str | None = None, live: dict | None = None) -> str:
    """'BULLISH EXPANSION | 62% Confidence | live: 7 of last 10 days | Trap Agent: Clear' - the top line of
    the wire plan (Hammad, message 5). The hit rate is the last RECENT_DAYS, live once a day has finished."""
    p = plan.get("p_scenario_a")
    bits = [REGIME_WORD[plan["regime"]]]
    if p is not None:
        bits.append(f"{p:.0%} Confidence")
    rate = qf.hit_rate(plan, live)
    if rate:
        bits.append(rate)
    trap = trap or trap_label(plan)
    if trap:
        bits.append(trap)
    return " | ".join(bits)


def windows_line(plan: dict) -> str:
    """'Chop 00:00–06:00 UTC (05:00–11:00 PKT) · Expansion 13:00–16:00 UTC (…)' or ''."""
    parts = []
    for phase, word in (("chop", "Chop"), ("expansion", "Expansion")):
        ws = [window_label(w["start_ms"], w["end_ms"]) for w in plan.get("windows") or [] if w["phase"] == phase]
        if ws:
            parts.append(f"{word} " + ", ".join(ws[:3]))
    return " · ".join(parts)


def coordinates_line(plan: dict) -> str:
    pl, ph = plan["projected_low"], plan["projected_high"]
    return (f"Low {_money(pl['price'])} at {at_label(pl['t_ms'])} → High {_money(ph['price'])} at "
            f"{at_label(ph['t_ms'])}")


def _node_label(n: dict) -> str:
    return f"{n['emoji']} {window_label(n['start_ms'], n['end_ms'])} {n['phase']} {_money(n['low'])}–{_money(n['high'])}"


def timeline_line(plan: dict, now_ms: int | None = None) -> str:
    """The compact timeline tree for a message (Hammad, message 6): 'Now ⚡ 12:00–16:00 UTC (…) expansion
    $80,200–$81,000 · Next 🎯 16:00–20:00 UTC (…) normal $80,400–$81,200', or '' on a locked plan."""
    from core.qf_tree import current_and_next, timeline_tree
    nodes = timeline_tree(plan, plan["created_ms"] if now_ms is None else now_ms)
    active, nxt = current_and_next(nodes)
    bits = []
    if active:
        bits.append("Now " + _node_label(active))
    if nxt:
        bits.append("Next " + _node_label(nxt))
    done = sum(1 for n in nodes if n["state"] == "COMPLETED_FADED")
    if nodes and done:
        bits.append(f"{done} of {len(nodes)} windows passed")
    return " · ".join(bits)


def window_state(plan: dict, now_ms: int) -> str:
    """'12:00–16:00 UTC (17:00–21:00 PKT) expansion window' - the active window for a tier message's tag."""
    from core.qf_tree import current_and_next, timeline_tree
    active, nxt = current_and_next(timeline_tree(plan, now_ms))
    n = active or nxt
    return f"{window_label(n['start_ms'], n['end_ms'])} {n['phase']} window" if n else "no window"


def directive_line(plan: dict) -> str:
    """'Trade: LONG · entry $79,600–$79,900 · stop $79,000 · TP $80,200 / $80,500 / $80,800' or ''."""
    from core.qf_tree import trade_directive
    d = trade_directive(plan)
    if d["direction"] == "NONE":
        return ""
    return (f"Trade: {d['direction']} · entry {_money(d['entry_zone'][0])}–{_money(d['entry_zone'][1])} · "
            f"stop {_money(d['stop_loss_invalidation'])} · TP {_money(d['tp1'])} / {_money(d['tp2'])} / {_money(d['tp3'])}")


def plan_headline(plan: dict, live: dict | None = None, title: str = "QUANT FORECAST",
                  trap: str | None = None, now_ms: int | None = None) -> str:
    """The day's plan as the bot sends it: the layman wire plan - regime, the one-sentence directive, the
    coordinates, the windows, the hard invalidation, the trade figures and the compact timeline (message 6).
    The record stays on the dashboard's drawer."""
    d = datetime.fromtimestamp(plan["day_ms"] / 1000, timezone.utc)
    lines = [f"📊 {title} · {d:%a %d %b} · BTC {_money(plan['origin'])}", f"Regime: {regime_line(plan, trap, live)}"]
    if plan["regime"] == "locked":
        lines.append(plan["scenario_a"] + f" Until {at_label(plan['end_ms'])}.")
        if plan.get("day_range"):
            lines.append(f"Today's range so far: {_money(plan['day_range'][0])} – {_money(plan['day_range'][1])}")
        return "\n".join(lines)
    lines.append(f"Directive: {qf.wire_line(plan)}")
    lines.append(coordinates_line(plan))
    if windows_line(plan):
        lines.append(windows_line(plan))
    if directive_line(plan):
        lines.append(directive_line(plan))
    if timeline_line(plan, now_ms):
        lines.append(timeline_line(plan, now_ms))
    lines.append(f"Whole-day invalidation: {_inv_text(plan)}")
    return "\n".join(lines)


def caution_headline(plan: dict, c: dict) -> str:
    """Tier 1, in Hammad's words (message 6), tagged with the active window and carrying the timeline."""
    what, why = (("Volatility spike", f"a 15-minute candle moved {c['sigmas']}x its usual size "
                                      f"({_money(c['from'])} → {_money(c['to'])})") if c["why"] == "move" else
                 ("CVD divergence", f"the last hour's taker flow turned against the plan ({c['delta_btc']:+,.0f} BTC)"))
    return (f"⚠️ CAUTION [{window_state(plan, c['t_ms'])}]: Order flow metrics are moving differently than baseline "
            f"projection. {what} detected: {why}. Whole-Day Invalidation at {_inv_text(plan)} has NOT been breached. "
            f"Plan remains active.\n{timeline_line(plan, c['t_ms'])}")


def held_headline(plan: dict, h: dict) -> str:
    inv = plan.get("invalidation") or {}
    where = (f"above whole-day invalidation boundary {_money(inv['below'])}" if list(inv) == ["below"] else
             f"below whole-day invalidation boundary {_money(inv['above'])}" if list(inv) == ["above"] else
             f"inside whole-day invalidation boundary {_inv_text(plan)}")
    return (f"🟢 MODEL HELD: Intraday shock to {_money(h['to'])} absorbed successfully {where}. "
            f"Baseline 1-day forecast path remains active.\n{timeline_line(plan, h['t_ms'])}")


def invalidated_headline(plan: dict, hit: dict, new: dict | None) -> str:
    """Tier 3, in Hammad's words (message 6), then Scenario B - the plan that replaces it."""
    where = (f"{hit['basis']}, {hit['side']} the {_money(hit['level'])} line, {at_label(hit['close_ms'])}")
    head = (f"🔴 WHOLE-DAY PLAN INVALIDATED: Macro structure breached at {_money(hit['close'])} ({where}). "
            "Original roadmap voided.")
    if new is not None and new["regime"] == "locked":
        return (head + f" Invalidated {_TIMES.get(qf.LOCK_AFTER_INVALIDATIONS, qf.LOCK_AFTER_INVALIDATIONS)} today: "
                "LOCKOUT MODE: HIGH CHOP / STAND ASIDE for the remainder of the 24-hour cycle.\n"
                + plan_headline(new, title="LOCKOUT MODE"))
    head += " Scenario B secondary chart activated with new targets."
    if new is None:
        return head + "\nThe new targets follow on the next pass."
    return head + "\n" + plan_headline(new, title=f"SCENARIO B · FORECAST #{new['version']}", now_ms=new["created_ms"])


_TIMES = {2: "twice", 3: "three times"}


def _key(plan: dict, *parts) -> str:
    return ":".join(["qf", str(plan["day_ms"]), str(plan["version"]), *map(str, parts)])


def audit_events(plan: dict, a: dict, now_ms: int) -> list[Event]:
    """Cautions (one an hour) and holds as alert events; stale ones are dropped, not sent late.

    The hour is taken by the caution that is sent, not by a stale one that was not: the ledger's key per
    hour still keeps a second caution in the same hour from going out. A MODEL HELD goes out only for a
    shock whose caution went out, else it answers a message nobody got."""
    out, hours_seen, sent_shocks = [], set(), set()
    for c in a["cautions"]:
        bucket = c["t_ms"] // HOUR_MS
        if bucket in hours_seen or now_ms - c["t_ms"] > SEND_WITHIN_MS:
            continue
        hours_seen.add(bucket)
        sent_shocks.add(c["t_ms"])
        out.append(Event("qf_caution", _key(plan, "caution", bucket), caution_headline(plan, c), c.get("to")))
    for h in a["holds"]:
        if h["shock_ms"] in sent_shocks or _shock_was_sent(a, h, now_ms):
            if now_ms - h["t_ms"] <= SEND_WITHIN_MS:
                out.append(Event("qf_held", _key(plan, "held", h["shock_ms"]), held_headline(plan, h), h["to"]))
    return out


def _shock_was_sent(a: dict, h: dict, now_ms: int) -> bool:
    """Whether the shock behind this hold was the first caution of its hour (so an earlier pass sent it)."""
    bucket = h["shock_ms"] // HOUR_MS
    first = next((c for c in a["cautions"] if c["t_ms"] // HOUR_MS == bucket), None)
    return bool(first) and first["t_ms"] == h["shock_ms"]


# ---------- one pass ----------

def live_record(store, n: int = 10) -> dict:
    """The morning plans' live record over the last n scored days."""
    rows = [r for r in store.qf_plans(limit=200) if r["version"] == 0 and r.get("outcome")]
    return qf.record([r["outcome"] for r in rows[:n]])


def _write_up(plan: dict, store) -> None:
    """Step 5: the AI's plain-English plan, checked against the numbers; kept on the plan when it passes,
    the reason kept when it does not (the code's own text is used then)."""
    from core import qf_writeup
    try:
        w = qf_writeup.write(plan, live_record(store))
    except Exception as e:  # noqa: BLE001 - the plan goes out with the code's text
        w = {"rejected": f"{type(e).__name__}: {e}"}
    if w is not None:
        plan["writeup"] = w


def _publish(store, plan: dict) -> bool:
    """Write the plan's row, then - only as the process that won the row - ask for its write-up and keep
    it on the row. False when another process had already written this version."""
    pid = store.put_qf_plan(plan)
    if not pid:
        return False
    _write_up(plan, store)
    if plan.get("writeup"):
        store.set_qf_payload(pid, plan)
    return True


def _replacement(store, plans: list[dict], prev: dict, hit: dict, h1, price: float, now_ms: int,
                 flow: dict, gamma: dict | None, pre: dict) -> dict | None:
    """The plan that follows an invalidated one: a fresh forecast, or the locked rest of the day once
    LOCK_AFTER_INVALIDATIONS plans have failed. None when it could not be built or was built elsewhere."""
    day, end = prev["day_ms"], prev["end_ms"]
    failed = sum(1 for r in plans if r["status"] == "invalidated")
    version = max(r["version"] for r in plans) + 1
    if failed >= qf.LOCK_AFTER_INVALIDATIONS:
        h_since = h1[h1["timestamp"].astype("int64") >= day]
        new = qf.locked_plan(now_ms, end, float(price), version,
                             float(h_since["high"].max()) if len(h_since) else None,
                             float(h_since["low"].min()) if len(h_since) else None)
    else:
        new = qf.build_plan(h1, now_ms, end, float(price), version=version, flow=flow.get("1h"), gamma=gamma,
                            **qf_preflight.plan_kwargs(pre))
        if not new.get("available"):
            return None
        new["backtest"] = prev.get("backtest")
        new["preflight"] = pre
    new["replaces"] = {"version": prev["version"], "invalidation": hit}
    return new if _publish(store, new) else None


def run(store, m, now_ms: int, gamma: dict | None = None, weekly=None, weekly_gamma: dict | None = None) -> list[Event]:
    """Make today's plan if there is none, replace it when invalidated, audit it, score what has ended.
    Returns the alert events this pass found; the caller logs them in the ledger, which keeps them once.
    `weekly` (the Weekly desk's analysis) writes the Sticky Monday Master Plan on the week's first pass."""
    frames = m.spot.frames or {}
    h1 = frames.get("1h")
    price = m.spot.last
    if h1 is None or h1.empty or not price:
        return []
    if weekly is not None:
        from core import qf_weekly
        try:
            qf_weekly.ensure(store, weekly, weekly_gamma, now_ms, float(price))
        except Exception as e:  # noqa: BLE001 - the week's note must not stop the day's plan
            log.warning("quant forecast: weekly plan not written: %s", e)
    flow = m.taker_flow.frames if getattr(m.taker_flow, "available", False) else {}
    day = qf.day_start(now_ms)
    end = day + qf.DAY_MS
    events: list[Event] = []

    for row in store.qf_due(now_ms):
        outcome = qf.score_plan(row["payload"], h1, row.get("closed_ms"))
        if outcome is not None:
            store.score_qf_plan(row["id"], outcome, now_ms)

    plans = store.qf_plans(day)
    pre = qf_preflight.preflight(store, m, now_ms, end)     # Trap Head and Economic Data, before any plan is built
    if not plans:
        plan = qf.build_plan(h1, now_ms, end, float(price), version=0, flow=flow.get("1h"), gamma=gamma,
                             **qf_preflight.plan_kwargs(pre))
        if not plan.get("available"):
            return events
        plan["backtest"] = qf.backtest(h1, now_ms)
        plan["preflight"] = pre
        if _publish(store, plan):
            events.append(Event("qf_plan", _key(plan, "plan"), plan_headline(plan, live_record(store)), plan["origin"]))
        plans = store.qf_plans(day)

    active = next((r for r in reversed(plans) if r["status"] == "active"), None)
    if active is None:
        # the day's last plan was closed and nothing followed it (a crash between the two steps, or the
        # replacement could not be built then): build it now, and send the invalidation with it
        last = plans[-1] if plans else None
        if last and last["status"] == "invalidated" and now_ms < end:
            try:
                hit = json.loads(last.get("closed_reason") or "")
            except ValueError:
                hit = {"basis": "4-hour close", "close": last["payload"]["origin"], "side": "beyond",
                       "level": 0.0, "close_ms": last.get("closed_ms") or now_ms}
            new = _replacement(store, plans, last["payload"], hit, h1, float(price), now_ms, flow, gamma, pre)
            if new is not None:
                events.append(Event("qf_invalid", _key(last["payload"], "invalidated"),
                                    invalidated_headline(last["payload"], hit, new), hit.get("close")))
        return events
    plan = active["payload"]
    a = audit(plan, frames, flow.get("15m"), now_ms)
    hit = a["invalidation"]
    if hit is None:
        if qf.magnitude_scale(plan, h1, flow.get("1h"), now_ms):
            store.set_qf_payload(active["id"], plan)      # the target widened once; the chart's badge follows
        return events + audit_events(plan, a, now_ms)
    if not store.close_qf_plan(active["id"], "invalidated", json.dumps(hit), now_ms):
        return events                               # another process replaced it first, and sent the alert
    plans = store.qf_plans(day)
    new = _replacement(store, plans, plan, hit, h1, float(price), now_ms, flow, gamma, pre)
    events.append(Event("qf_invalid", _key(plan, "invalidated"), invalidated_headline(plan, hit, new), hit["close"]))
    return events
