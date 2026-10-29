"""The Sticky Monday Master Plan (Hammad, message 8, panel B): the week's bias, the dealers' gamma walls
and the macro baseline levels, in plain words, anchored for the week.

Built once per week from the Weekly desk's own reading and the Deribit gamma profile, on the first pass
after Monday 00:00 UTC (05:00 PKT), and kept as written: the store's INSERT OR IGNORE makes the first
writer's plan the week's plan, whoever runs the pass. It does not move with the price during the week -
that is what "sticky" buys: a baseline the daily forecast is read against, not a second daily forecast.

Everything here is read off numbers already computed elsewhere; no model writes a word of it."""
from __future__ import annotations

from core.quant_forecast import DAY_MS

WEEK_MS = 7 * DAY_MS
_MONDAY_OFFSET_DAYS = 4          # 1970-01-01 was a Thursday; the first Monday is day 4


def week_start(ms: int) -> int:
    """Monday 00:00 UTC of the week holding `ms`."""
    days = int(ms) // DAY_MS
    return (days - (days - _MONDAY_OFFSET_DAYS) % 7) * DAY_MS


def _money(x) -> str:
    return "—" if x is None else f"${x:,.0f}"


def _conf_word(c: float | None) -> str:
    c = float(c or 0.0)
    return "a strong lean" if c >= 0.66 else "a moderate lean" if c >= 0.4 else "a slight lean"


def _floor_ceiling(levels, price: float) -> tuple[float | None, float | None]:
    sups = sorted((lv.price for lv in levels or [] if getattr(lv, "kind", "") == "support" and lv.price < price), reverse=True)
    ceils = sorted(lv.price for lv in levels or [] if getattr(lv, "kind", "") == "resistance" and lv.price > price)
    return (sups[0] if sups else None), (ceils[0] if ceils else None)


def build(weekly, gamma: dict | None, now_ms: int, price: float | None = None) -> dict:
    """The week's plan from the Weekly desk analysis (direction, levels, vol) and its gamma profile."""
    d = weekly.direction
    px = float(price or weekly.price)
    bias = d.direction if d.direction in ("BULLISH", "BEARISH") else "RANGE"
    floor, ceiling = _floor_ceiling(getattr(weekly, "levels", None), px)
    vol = getattr(weekly, "vol", None)
    if floor is None and vol is not None:
        floor = getattr(vol, "exp_low", None)
    if ceiling is None and vol is not None:
        ceiling = getattr(vol, "exp_high", None)
    g = gamma or {}
    put_wall = g.get("put_wall") if g.get("available", True) else None
    call_wall = g.get("call_wall") if g.get("available", True) else None
    put_wall = float(put_wall) if put_wall else None
    call_wall = float(call_wall) if call_wall else None

    parts = []
    if bias == "RANGE":
        parts.append("The weekly chart has no lean this week: a range week until one edge gives way.")
    else:
        parts.append(f"The weekly chart leans {bias.lower()} ({_conf_word(d.confidence)}).")
    if floor is not None or ceiling is not None:
        parts.append(f"Baseline floor {_money(floor)} and ceiling {_money(ceiling)} from the weekly structure.")
    if put_wall is not None or call_wall is not None:
        parts.append(f"Options dealers' walls sit at {_money(put_wall)} below and {_money(call_wall)} above; "
                     "price tends to slow into them and accelerate through them.")
    inv = getattr(d, "invalidation", None)
    if bias != "RANGE" and inv:
        parts.append(f"The weekly view flips on a weekly close {'below' if bias == 'BULLISH' else 'above'} {_money(inv)}.")
    return {"week_ms": week_start(now_ms), "created_ms": int(now_ms), "weekly_bias": bias,
            "confidence": round(float(d.confidence or 0.0), 3), "weekly_gex_walls": [put_wall, call_wall],
            "baseline": {"floor": floor, "ceiling": ceiling}, "invalidation": inv,
            "drivers": list(getattr(d, "drivers", []) or [])[:3], "macro_baseline_summary": " ".join(parts)}


def ensure(store, weekly, gamma: dict | None, now_ms: int, price: float | None = None) -> dict | None:
    """This week's plan from the store, written first if the week has none and a weekly reading is here."""
    row = store.qf_week(week_start(now_ms))
    if row is not None:
        return row["payload"]
    if weekly is None:
        return None
    store.put_qf_week(build(weekly, gamma, now_ms, price))
    row = store.qf_week(week_start(now_ms))
    return row["payload"] if row else None
