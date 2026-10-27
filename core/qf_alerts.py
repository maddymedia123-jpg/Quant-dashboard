"""Quant Forecast audit and alerts: the day's plan watched on every pass (the 24/7 worker's, and the page's).

The tiers, as agreed with Hammad (28 Oct):

* CAUTION (tier 1) - a closed 15-minute candle moves more than CAUTION_SIGMA times that hour's expected
  15-minute move against the plan, or the last hour's taker delta turns against it at CVD_FLIP_X times
  its usual size, while invalidation holds. On a neutral plan any such move counts, either way. At most
  one caution an hour.
* MODEL HELD (tier 2B) - after a shock like that, price takes back at least half of it within
  HOLD_WINDOW_MS without breaking invalidation.
* INVALIDATED (tier 2A) - a 4-hour candle closes beyond the invalidation level, or an hourly candle
  closes more than one 1h ATR beyond it (core.quant_forecast.invalidation_hit). The plan is replaced by a
  new one for the rest of the day; the LOCK_AFTER_INVALIDATIONS-th failure locks the day instead.

Detection is read from the candles each pass, so it holds no state of its own; what is new is decided by
the alert ledger's unique keys, and who replaces a plan by the one-statement update in the store. An
alert older than SEND_WITHIN_MS (the worker was down) is skipped: an hour-old caution is not news."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import pandas as pd

from core import quant_forecast as qf
from core.alerts import Event

HOUR_MS = qf.HOUR_MS
M15_MS = 900_000
CAUTION_SIGMA = 2.0
CVD_BARS = 4                     # the last hour of 15-minute taker delta
CVD_TYPICAL_ROWS = 96            # its usual size, from the day before
CVD_FLIP_X = 1.5
HOLD_WINDOW_MS = HOUR_MS
SEND_WITHIN_MS = 30 * 60_000
REGIME_WORD = {"bullish": "BULLISH EXPANSION", "bearish": "BEARISH EXPANSION",
               "neutral": "NEUTRAL / RANGE-BOUND", "locked": "SYSTEM LOCKED: HIGH-CHOP / STAND ASIDE"}


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
    """The plan's expected 15-minute move (as a log return) for the hour holding t_ms."""
    for h in plan.get("hours") or []:
        if t_ms <= h["t_ms"]:
            return h["sigma_pct"] / 100 / 2          # a quarter of the hour's variance
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


def _odds_text(plan: dict, live: dict | None = None) -> str:
    p = plan.get("p_scenario_a")
    out = "" if p is None else f"{p:.0%} by the model"
    out += "".join(f" · {t}" for t in record_words(plan, live))
    return out


def record_words(plan: dict, live: dict | None) -> list[str]:
    """The record that matters for this plan's kind: direction calls for a directional plan, range days
    for a neutral one, from the backtest and the live days."""
    key = ("range_held", "range_days", "range held") if plan.get("regime") == "neutral" else         ("direction_right", "direction_days", "direction right")
    out = []
    for name, rec in (("backtest", plan.get("backtest") or {}), ("live", live or {})):
        if rec.get(key[1]):
            out.append(f"{name}: {key[2]} {rec[key[0]]} of {rec[key[1]]} days")
    return out


def plan_headline(plan: dict, live: dict | None = None, title: str = "QUANT FORECAST") -> str:
    """The day's plan, as the bot sends it: regime and odds, the plan in a line, the windows, the levels."""
    d = datetime.fromtimestamp(plan["day_ms"] / 1000, timezone.utc)
    lines = [f"📊 {title} · {d:%a %d %b} · BTC {_money(plan['origin'])}",
             f"Regime: {REGIME_WORD[plan['regime']]}" + (f" ({_odds_text(plan, live)})" if plan.get("p_scenario_a") else "")]
    if plan["regime"] == "locked":
        lines.append(plan["scenario_a"] + f" Until {at_label(plan['end_ms'])}.")
        return "\n".join(lines)
    w = plan.get("writeup") or {}
    lines.append(f"Game plan: {w['summary']}" if w.get("summary") else f"Plan: {plan['scenario_a']}")
    for phase, word in (("chop", "Quiet / chop"), ("expansion", "Expansion")):
        ws = [window_label(w["start_ms"], w["end_ms"]) for w in plan.get("windows") or [] if w["phase"] == phase]
        if ws:
            lines.append(f"{word}: " + ", ".join(ws[:3]))
    ph, pl = plan["projected_high"], plan["projected_low"]
    lines.append(f"High ≈ {_money(ph['price'])} around {at_label(ph['t_ms'])} · "
                 f"Low ≈ {_money(pl['price'])} around {at_label(pl['t_ms'])}")
    lines.append(f"Invalidation (4h close): {_inv_text(plan)}")
    return "\n".join(lines)


def caution_headline(plan: dict, c: dict) -> str:
    why = (f"a 15-minute candle moved {c['sigmas']}x its usual size ({_money(c['from'])} → {_money(c['to'])})"
           if c["why"] == "move" else
           f"the last hour's taker flow turned against the plan ({c['delta_btc']:+,.0f} BTC)")
    return (f"⚠️ CAUTION: market is moving differently from the forecast: {why}. Be careful. "
            f"Hard invalidation {_inv_text(plan)} has NOT been breached.")


def held_headline(plan: dict, h: dict) -> str:
    return (f"🟢 MODEL HELD: the shock to {_money(h['to'])} was absorbed without breaching invalidation "
            f"({_inv_text(plan)}). The original forecast stays active.")


def invalidated_headline(plan: dict, hit: dict, new: dict | None) -> str:
    head = (f"🔴 FORECAST INVALIDATED: {hit['basis']} at {_money(hit['close'])}, {hit['side']} the "
            f"{_money(hit['level'])} line ({at_label(hit['close_ms'])}). The plan is cancelled.")
    if new is None:
        return head
    if new["regime"] == "locked":
        return head + "\n" + plan_headline(new, title="SYSTEM LOCKED")
    return head + "\n" + plan_headline(new, title=f"NEW FORECAST #{new['version']}")


def _key(plan: dict, *parts) -> str:
    return ":".join(["qf", str(plan["day_ms"]), str(plan["version"]), *map(str, parts)])


def audit_events(plan: dict, a: dict, now_ms: int) -> list[Event]:
    """Cautions (one an hour) and holds as alert events; stale ones are dropped, not sent late."""
    out, hours_seen = [], set()
    for c in a["cautions"]:
        bucket = c["t_ms"] // HOUR_MS
        if bucket in hours_seen:
            continue
        hours_seen.add(bucket)
        if now_ms - c["t_ms"] <= SEND_WITHIN_MS:
            out.append(Event("qf_caution", _key(plan, "caution", bucket), caution_headline(plan, c), c.get("to")))
    for h in a["holds"]:
        if now_ms - h["t_ms"] <= SEND_WITHIN_MS:
            out.append(Event("qf_held", _key(plan, "held", h["shock_ms"]), held_headline(plan, h), h["to"]))
    return out


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


def run(store, m, now_ms: int, gamma: dict | None = None) -> list[Event]:
    """Make today's plan if there is none, replace it when invalidated, audit it, score what has ended.
    Returns the alert events this pass found; the caller logs them in the ledger, which keeps them once."""
    frames = m.spot.frames or {}
    h1 = frames.get("1h")
    price = m.spot.last
    if h1 is None or h1.empty or not price:
        return []
    flow = m.taker_flow.frames if getattr(m.taker_flow, "available", False) else {}
    day = qf.day_start(now_ms)
    end = day + qf.DAY_MS
    events: list[Event] = []

    for row in store.qf_due(now_ms):
        outcome = qf.score_plan(row["payload"], h1)
        if outcome is not None:
            store.score_qf_plan(row["id"], outcome, now_ms)

    plans = store.qf_plans(day)
    if not plans:
        plan = qf.build_plan(h1, now_ms, end, float(price), version=0, flow=flow.get("1h"), gamma=gamma)
        if not plan.get("available"):
            return events
        plan["backtest"] = qf.backtest(h1, now_ms)
        _write_up(plan, store)
        if store.put_qf_plan(plan):
            events.append(Event("qf_plan", _key(plan, "plan"), plan_headline(plan, live_record(store)), plan["origin"]))
        plans = store.qf_plans(day)

    active = next((r for r in reversed(plans) if r["status"] == "active"), None)
    if active is None:
        return events
    plan = active["payload"]
    a = audit(plan, frames, flow.get("15m"), now_ms)
    hit = a["invalidation"]
    if hit is None:
        return events + audit_events(plan, a, now_ms)
    if not store.close_qf_plan(active["id"], "invalidated", json.dumps(hit), now_ms):
        return events                               # another process replaced it first, and sent the alert
    failed = sum(1 for r in plans if r["status"] == "invalidated") + 1
    version = max(r["version"] for r in plans) + 1
    new = None
    if failed >= qf.LOCK_AFTER_INVALIDATIONS:
        h_since = h1[h1["timestamp"].astype("int64") >= day]
        new = qf.locked_plan(now_ms, end, float(price), version,
                             float(h_since["high"].max()) if len(h_since) else None,
                             float(h_since["low"].min()) if len(h_since) else None)
    else:
        built = qf.build_plan(h1, now_ms, end, float(price), version=version, flow=flow.get("1h"), gamma=gamma)
        if built.get("available"):
            built["backtest"] = plan.get("backtest")
            built["replaces"] = {"version": plan["version"], "invalidation": hit}
            _write_up(built, store)
            new = built
    if new is not None:
        new["replaces"] = {"version": plan["version"], "invalidation": hit}
        store.put_qf_plan(new)
    events.append(Event("qf_invalid", _key(plan, "invalidated"), invalidated_headline(plan, hit, new), hit["close"]))
    return events
