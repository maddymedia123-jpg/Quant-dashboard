"""The 4-hour tactical execution window against the 24-hour intraday horizon (roadmap item 12, owner's GO,
5 November): "evaluates 4-hour tactical execution zones against 24-hour intraday liquidity horizons".

Two desks own the two horizons. The Live desk executes inside the 4-hour candle (its anchor): its
expected range for the next four hours is the tactical zone. The Intraday desk owns the day: the day's
high and low so far, the prior day's high and low (the pools stops rest behind), the day's average price
(the anchored VWAP the cascade reads) and the desk's 24-hour expected range are the horizon the
tactical move has to live inside or break out of.

The framing is computed here, in code, from numbers already on the screen, and given to the Director
as a block it must not contradict; the Director then writes the sentences (horizon_framing) and the
plan it chose is checked against the same horizon (plan_check): a zone and targets inside the day's
edges are a tactical trade; a first target beyond them needs the 24-hour horizon taken first and says
so; a zone outside them is a trade at the day's edge, not inside it. Nothing here guesses a level: a
missing frame or forecast leaves its part of the block absent and named."""
from __future__ import annotations

from datetime import datetime, timezone

HOUR_MS = 3_600_000
DAY_MS = 86_400_000
EDGE_BAND_PCT = 20.0        # within the top or bottom fifth of the day's range counts as "at" that edge


def _utc(ms) -> str | None:
    if ms is None:
        return None
    return f"{datetime.fromtimestamp(int(ms) / 1000, timezone.utc):%H:%M UTC %a %d %b}"


def _candle(frames: dict, tf: str, back: int = 0) -> dict | None:
    """The candle `back` rows from the end of a frame (0 = the forming one, 1 = the last completed)."""
    try:
        df = frames.get(tf) if frames else None
        if df is None or len(df) <= back:
            return None
        row = df.iloc[-1 - back]
        return {"opened": _utc(row.get("timestamp")), "open": round(float(row["open"])), "high": round(float(row["high"])),
                "low": round(float(row["low"])), "close": round(float(row["close"]))}
    except Exception:  # noqa: BLE001 - a malformed frame is an absent part, not a crash
        return None


def _nearest_levels(a, price: float) -> dict:
    """The nearest floor below and ceiling above price from a desk's levels, with their touch counts."""
    out: dict = {}
    levels = getattr(a, "levels", None) or []
    floors = [lv for lv in levels if getattr(lv, "kind", "") == "support" and lv.price < price]
    ceilings = [lv for lv in levels if getattr(lv, "kind", "") == "resistance" and lv.price > price]
    if floors:
        lv = max(floors, key=lambda x: x.price)
        out["floor"] = {"price": round(lv.price), "touches": lv.touches}
    if ceilings:
        lv = min(ceilings, key=lambda x: x.price)
        out["ceiling"] = {"price": round(lv.price), "touches": lv.touches}
    return out


def _range(forecast: dict | None) -> list | None:
    if not forecast or not forecast.get("available"):
        return None
    r = forecast.get("range_68")
    if not r or len(r) != 2 or r[0] is None or r[1] is None:
        return None
    return [round(float(r[0])), round(float(r[1]))]


def read(m, analyses: dict, now_ms: int, forecasts: dict | None = None, cascade: dict | None = None,
         derivs: dict | None = None) -> dict:
    """The two horizons and the framing between them. `forecasts`, `cascade` and `derivs` are the
    Director state's own blocks, so nothing is computed twice."""
    frames = getattr(getattr(m, "spot", None), "frames", None) or {}
    live, intraday = analyses.get("live"), analyses.get("intraday")
    price = getattr(getattr(m, "spot", None), "last", None) or (live.price if live is not None else None)
    forecasts, cascade, derivs = forecasts or {}, cascade or {}, derivs or {}

    four: dict = {"desk": "live", "window": "the current 4-hour candle", "horizon": "next 4h"}
    four["candle"] = _candle(frames, "4h", 0)
    four["prior_candle"] = _candle(frames, "4h", 1)
    four["expected_range_68"] = _range(forecasts.get("live"))
    if live is not None and price:
        four["levels"] = _nearest_levels(live, price)

    day: dict = {"desk": "intraday", "window": "the current day (UTC)", "horizon": "next 24h"}
    day["day"] = _candle(frames, "1d", 0)
    day["prior_day"] = _candle(frames, "1d", 1)
    vwap = (cascade.get("live") or {}).get("vwap")
    day["day_vwap"] = round(float(vwap)) if vwap else None
    day["expected_range_68"] = _range(forecasts.get("intraday"))
    if intraday is not None and price:
        day["levels"] = _nearest_levels(intraday, price)
    g = (derivs.get("intraday") or {}).get("gamma") or {}
    walls = {k: g.get(k) for k in ("call_wall", "put_wall", "zero_gamma") if g.get(k) is not None}
    if walls:
        day["gamma_walls"] = walls
    liq = (derivs.get("intraday") or {}).get("liquidations")
    if liq is not None:
        day["liquidations"] = liq

    return {"four_hour": four, "twenty_four_hour": day, "framing": frame(price, four, day),
            "method": ("the 4-hour expected range (Live desk, next 4h) against the day's high and low so far, "
                       "the prior day's high and low, the day's average price and the Intraday desk's 24-hour "
                       "expected range; the pools named are the ones the 4-hour range reaches")}


def _pools(price: float, day: dict) -> tuple[list[dict], list[dict]]:
    """The day's edges above and below price, nearest first, each named."""
    cands = []
    d, p = day.get("day") or {}, day.get("prior_day") or {}
    r = day.get("expected_range_68")
    # the day's average price is a reference inside the range, not an edge where stops rest: it is named
    # in the note but never counted as a pool the 4-hour range "reaches"
    for name, value in (("the day's high", d.get("high")), ("the day's low", d.get("low")),
                        ("the prior day's high", p.get("high")), ("the prior day's low", p.get("low")),
                        ("the 24-hour expected high", r[1] if r else None), ("the 24-hour expected low", r[0] if r else None)):
        if value is not None:
            cands.append({"name": name, "price": float(value)})
    above = sorted((c for c in cands if c["price"] > price), key=lambda c: c["price"])
    below = sorted((c for c in cands if c["price"] < price), key=lambda c: -c["price"])
    return above, below


def frame(price: float | None, four: dict, day: dict) -> dict:
    """{"state": inside | probe_above | probe_below | expansion | unknown, "note", ...}: where price sits in
    the day's range and which of the day's edges the 4-hour expected range reaches."""
    r4 = four.get("expected_range_68")
    d = day.get("day") or {}
    if not price or not r4:
        return {"state": "unknown", "note": "no 4-hour expected range this run, so the tactical zone cannot be framed"}
    above, below = _pools(price, day)
    reach_up = [c for c in above if r4[1] >= c["price"]]
    reach_down = [c for c in below if r4[0] <= c["price"]]
    out: dict = {"four_hour_range": r4, "reaches_above": [c["name"] for c in reach_up],
                 "reaches_below": [c["name"] for c in reach_down]}
    where = "inside the day's range"
    if d.get("high") is not None and d.get("low") is not None and d["high"] > d["low"]:
        pct = (price - d["low"]) / (d["high"] - d["low"]) * 100.0
        out["position_in_day_pct"] = round(pct)
        out["day_range"] = [d["low"], d["high"]]
        out["four_hour_vs_day_width_pct"] = round((r4[1] - r4[0]) / (d["high"] - d["low"]) * 100.0)
        where = ("at the day's high" if pct >= 100 - EDGE_BAND_PCT else
                 "at the day's low" if pct <= EDGE_BAND_PCT else "mid-range in the day")
    if reach_up and reach_down:
        out["state"] = "expansion"
        verb = (f"reaches {', '.join(c['name'] for c in reach_up)} above and {', '.join(c['name'] for c in reach_down)} "
                "below: the next four hours can take the day's edges either way, an expansion window, not a range")
    elif reach_up:
        out["state"] = "probe_above"
        verb = (f"reaches {', '.join(c['name'] for c in reach_up)} above ({', '.join(f'{c['price']:,.0f}' for c in reach_up)}) "
                "and nothing below: the tactical move is a probe of the day's upper edge, where resting sells "
                "and stops sit; a break needs a close beyond it, a rejection there is the trap")
    elif reach_down:
        out["state"] = "probe_below"
        verb = (f"reaches {', '.join(c['name'] for c in reach_down)} below ({', '.join(f'{c['price']:,.0f}' for c in reach_down)}) "
                "and nothing above: the tactical move is a probe of the day's lower edge, where resting bids "
                "and stops sit; a break needs a close beyond it, a rejection there is the trap")
    else:
        out["state"] = "inside"
        nxt = []
        if above:
            nxt.append(f"{above[0]['name']} {above[0]['price']:,.0f} above")
        if below:
            nxt.append(f"{below[0]['name']} {below[0]['price']:,.0f} below")
        verb = ("stays inside the day's edges" + (f" ({' and '.join(nxt)})" if nxt else "")
                + ": the next four hours are a trade inside the day's range, not a break of it")
    out["note"] = f"Price {price:,.0f} is {where}; the 4-hour expected range {r4[0]:,.0f}-{r4[1]:,.0f} {verb}"
    vwap = day.get("day_vwap")
    if vwap:
        out["note"] += f"; the day's average price {vwap:,.0f} is {'below' if vwap < price else 'above'} price"
    return out


def plan_check(plan, horizons: dict | None) -> dict | None:
    """The Director's numeric plan against the 24-hour horizon: {"verdict": tactical | reaches_beyond |
    outside | no_trade, "note", ...}; None when there is no plan or no horizon to check it against."""
    if plan is None or horizons is None:
        return None
    framing = horizons.get("framing") or {}
    base = {"framing_state": framing.get("state"), "framing_note": framing.get("note")}
    if getattr(plan, "side", "NONE") == "NONE":
        return {**base, "verdict": "no_trade", "note": "no plan to frame"}
    day = horizons.get("twenty_four_hour") or {}
    edges = []
    d, p, r = day.get("day") or {}, day.get("prior_day") or {}, day.get("expected_range_68")
    for name, value in (("the day's high", d.get("high")), ("the prior day's high", p.get("high")),
                        ("the 24-hour expected high", r[1] if r else None)):
        if value is not None:
            edges.append(("upper", name, float(value)))
    for name, value in (("the day's low", d.get("low")), ("the prior day's low", p.get("low")),
                        ("the 24-hour expected low", r[0] if r else None)):
        if value is not None:
            edges.append(("lower", name, float(value)))
    uppers = [e for e in edges if e[0] == "upper"]
    lowers = [e for e in edges if e[0] == "lower"]
    if not uppers or not lowers:
        return {**base, "verdict": "unknown", "note": "no 24-hour edges this run, so the plan cannot be framed"}
    hi = max(uppers, key=lambda e: e[2])
    lo = min(lowers, key=lambda e: e[2])
    out = {**base, "horizon": [round(lo[2]), round(hi[2])], "horizon_edges": {"upper": hi[1], "lower": lo[1]}}
    zone_inside = lo[2] <= plan.zone_low and plan.zone_high <= hi[2]
    beyond = [t for t in plan.targets if t > hi[2] or t < lo[2]]
    out["targets_beyond_horizon"] = [round(t) for t in beyond]
    out["stop_beyond_horizon"] = bool(plan.stop is not None and (plan.stop > hi[2] or plan.stop < lo[2]))
    side = plan.side
    if not zone_inside:
        out["verdict"] = "outside"
        out["note"] = (f"the {side} zone {plan.zone_low:,.0f}-{plan.zone_high:,.0f} sits beyond the 24-hour horizon "
                       f"{lo[2]:,.0f}-{hi[2]:,.0f} ({lo[1]} to {hi[1]}): a trade at the day's edge, not inside it; "
                       "it needs price to get there first")
    elif beyond:
        edge = hi if side == "LONG" else lo
        out["verdict"] = "reaches_beyond"
        out["note"] = (f"the {side} zone {plan.zone_low:,.0f}-{plan.zone_high:,.0f} is inside the 24-hour horizon "
                       f"{lo[2]:,.0f}-{hi[2]:,.0f}, but target(s) {', '.join(f'{t:,.0f}' for t in beyond)} lie beyond "
                       f"{edge[1]} {edge[2]:,.0f}: the trade needs the 24-hour horizon taken, not just the 4-hour zone "
                       "held; size for the first target inside it")
    else:
        out["verdict"] = "tactical"
        out["note"] = (f"the {side} zone {plan.zone_low:,.0f}-{plan.zone_high:,.0f} and every target sit inside the "
                       f"24-hour horizon {lo[2]:,.0f}-{hi[2]:,.0f} ({lo[1]} to {hi[1]}): a tactical trade inside the day")
    if out["stop_beyond_horizon"]:
        out["note"] += f"; the stop {plan.stop:,.0f} is beyond the horizon, so a day-edge sweep is what stops it out"
    return out
