"""The Quant Forecast's timeline tree and trade directive (Hammad, message 6, panels B and C).

Both are read off the plan's own numbers, in code: no model writes a node or a level. The tree is the day
cut into its phases - each run of chop, normal or expansion hours is one node - with a state that follows
the clock (COMPLETED_FADED, ACTIVE_HIGHLIGHTED, PENDING), a one-line summary, the numbers behind it, and
the prices it spans. The directive is the plan as pure figures for execution: direction, stop (the
whole-day invalidation), an entry zone, three targets and the window they belong to.

How the figures are chosen (so a reader can check them):
* Entry zone: from the projected low (bullish; the projected high on a bearish plan) to the median path at
  the hour that low is most likely to print - the pullback the simulation expects before the move.
* Targets: the plan's levels between the entry zone and the projected high (low), in order; the projected
  high (low) itself is always the last. With fewer than three, the gap to the final target is split evenly.
* Setup type: the level the entry zone sits on, if one is within SETUP_TOL of it; else a plain pullback.
A neutral or locked day has no directive: stand aside."""
from __future__ import annotations

import math

import pandas as pd

from core import quant_forecast as qf

HOUR_MS = qf.HOUR_MS
MIN_NODE_HOURS = 2                # a shorter run of hours joins the node before it: 11 nodes a day read as noise
STATE_EMOJI = {"COMPLETED_FADED": "🕒", "ACTIVE_HIGHLIGHTED": "⚡", "PENDING": "🎯"}
# the "why" is plain trader English (message 8: zero raw math on the main screen); the numbers behind it
# stay on the node (bps) for the drawer and the JSON
TEMPO = {"COMPRESSED": ("quiet", "A quieter stretch than most of the day"),
         "NORMAL": ("normal", "An ordinary stretch for the day"),
         "HIGH VOLATILITY SPIKE": ("busy", "One of the day's busiest stretches")}
STATUS_WORD = {"COMPLETED_FADED": "🫥 Passed", "ACTIVE_HIGHLIGHTED": "🔥 Active Focus", "PENDING": "⏳ Pending"}
SETUP_TOL = 0.003                 # a level within 0.3% of the entry zone names the setup
SETUP_WORD = {"value low": "Value Area Low retest", "value high": "Value Area High retest", "POC": "POC retest",
              "air pocket": "Air-pocket retest", "est. liquidation zone": "Liquidation-zone retest",
              "put wall": "Put-wall retest", "call wall": "Call-wall retest", "gamma flip": "Gamma-flip retest"}


def _money(x) -> str:
    return "—" if x is None else f"${x:,.0f}"


def _state(start_ms: int, end_ms: int, now_ms: int) -> str:
    return "COMPLETED_FADED" if end_ms <= now_ms else "ACTIVE_HIGHLIGHTED" if start_ms <= now_ms else "PENDING"


def _actual(h1: pd.DataFrame | None, start_ms: int, end_ms: int) -> tuple[float, float] | None:
    """The low and high the market actually printed inside a window, from the closed hourly candles."""
    if h1 is None or h1.empty:
        return None
    t = h1["timestamp"].astype("int64")
    rows = h1[(t >= start_ms) & (t + HOUR_MS <= end_ms)]
    if rows.empty:
        return None
    return float(rows["low"].min()), float(rows["high"].max())


def _runs(hours: list[dict]) -> list[tuple[int, int, str]]:
    """(first index, last index, phase) for each run of same-phase hours, runs shorter than MIN_NODE_HOURS
    folded into the run before them (the first into the one after), neighbours of one phase then joined."""
    runs: list[list] = []
    for k, h in enumerate(hours):
        if runs and runs[-1][2] == h["phase"]:
            runs[-1][1] = k
        else:
            runs.append([k, k, h["phase"]])
    k = 0
    while len(runs) > 1 and k < len(runs):
        i, j, _ = runs[k]
        if j - i + 1 >= MIN_NODE_HOURS:
            k += 1
            continue
        if k == 0:
            runs[1][0] = i
        else:
            runs[k - 1][1] = j
        runs.pop(k)
        if 0 < k < len(runs) and runs[k - 1][2] == runs[k][2]:     # the fold made two neighbours one phase
            runs[k - 1][1] = runs[k][1]
            runs.pop(k)
    return [(i, j, ph) for i, j, ph in runs]


def timeline_tree(plan: dict, now_ms: int, h1: pd.DataFrame | None = None) -> list[dict]:
    """The day's nodes in time order. Each: start_ms, end_ms, phase, state, emoji, status, bps (the window's
    expected move, its hours' sigmas added in variance), low/high (the window's own range: the projected path
    where it starts, +- its hours' 68% move - the chart's box, not the day's cumulative band), levels inside
    that range, summary (one line), why (the numbers behind it), and - for a window that has passed, with
    candles - actual_low/actual_high. A locked plan has no nodes."""
    hours = plan.get("hours") or []
    if not hours or plan.get("regime") == "locked":
        return []
    up, down = plan["regime"] == "bullish", plan["regime"] == "bearish"
    ph, pl = plan["projected_high"], plan["projected_low"]
    levels = plan.get("levels") or []
    parts = plan.get("parts") or {}
    macro = plan.get("macro_windows") or []
    out: list[dict] = []
    for i, j, phase in _runs(hours):
        hs = hours[i:j + 1]
        start = plan["created_ms"] if i == 0 else hours[i - 1]["t_ms"]
        end = hs[-1]["t_ms"]
        centre = plan["origin"] if i == 0 else hours[i - 1]["median"]
        half = qf.Z84 * math.sqrt(sum((h["sigma_pct"] / 100) ** 2 for h in hs))
        lo, hi = centre * math.exp(-half), centre * math.exp(half)
        var = sum((h.get("sigma_full_pct", h["sigma_pct"]) / 100) ** 2 for h in hs)
        bps = qf._bps(math.sqrt(var) * 100)
        inside = [lv for lv in levels if lo <= lv["price"] <= hi]
        state = _state(start, end, now_ms)
        holds_low, holds_high = start <= pl["t_ms"] <= end, start <= ph["t_ms"] <= end
        # the one-liner
        if phase == "chop":
            summary = f"Quiet consolidation between {_money(lo)} and {_money(hi)}."
        elif phase == "expansion":
            summary = (f"Expansion window: the push toward {_money(ph['price'])}." if up else
                       f"Expansion window: the drive down toward {_money(pl['price'])}." if down else
                       f"Expansion window: the day's bigger swings, {_money(lo)} to {_money(hi)}.")
        else:
            summary = (f"Steady drift higher, {_money(lo)} to {_money(hi)}." if up else
                       f"Steady drift lower, {_money(lo)} to {_money(hi)}." if down else
                       f"Two-way trade between {_money(lo)} and {_money(hi)}.")
        if holds_low:
            summary += (f" Pullback floor near {_money(pl['price'])} most likely here." if up else
                        f" Day's low near {_money(pl['price'])} most likely here.")
        if holds_high:
            summary += (f" Rally cap near {_money(ph['price'])} most likely here." if down else
                        f" Day's high near {_money(ph['price'])} most likely here.")
        actual = _actual(h1, start, end) if state == "COMPLETED_FADED" else None
        if actual:
            summary = f"Traded {_money(actual[0])} to {_money(actual[1])}. " + summary.replace("most likely here", "was due here")
        # the why, from the numbers, said in plain words
        tempo, tempo_line = TEMPO[qf._character(sum(h.get('mult', 1.0) for h in hs) / len(hs))]
        why = [tempo_line]
        for w in macro:
            if w["start_ms"] < end and w["end_ms"] > start:
                why.append(f"{w['title']} is released inside this window")
        flow = parts.get("flow") or {}
        if flow.get("delta_btc") is not None:
            why.append("buyers have been paying up" if flow["delta_btc"] > 0 else
                       "sellers have been hitting bids" if flow["delta_btc"] < 0 else "buying and selling balanced")
        oi = parts.get("open_interest") or {}
        if oi.get("pct") is not None:
            why.append("more positions have been opened over the day" if oi["pct"] > 0.5 else
                       "positions have been closing out over the day" if oi["pct"] < -0.5 else
                       "open positions about unchanged")
        if inside:
            why.append("price meets " + ", ".join(f"the {lv['label']} at {_money(lv['price'])}" for lv in inside[:3]))
        node = {"start_ms": int(start), "end_ms": int(end), "phase": phase, "state": state, "emoji": STATE_EMOJI[state],
                "status": STATUS_WORD[state], "bps": bps, "tempo": tempo, "low": round(lo, 2), "high": round(hi, 2),
                "levels": inside[:4], "summary": summary, "why": "; ".join(why) + "."}
        if actual:
            node["actual_low"], node["actual_high"] = round(actual[0], 2), round(actual[1], 2)
        out.append(node)
    return out


def current_and_next(nodes: list[dict]) -> tuple[dict | None, dict | None]:
    """The active node and the first pending one (the morning: none active, the first node next)."""
    active = next((n for n in nodes if n["state"] == "ACTIVE_HIGHLIGHTED"), None)
    nxt = next((n for n in nodes if n["state"] == "PENDING"), None)
    return active, nxt


def trade_directive(plan: dict) -> dict:
    """The plan as pure figures. {"direction": "NONE", ...} on a neutral or locked day."""
    regime = plan.get("regime")
    if regime not in ("bullish", "bearish"):
        word = ("Stand aside: the day is locked." if regime == "locked" else
                "Stand aside: no direction today, a range is expected.")
        return {"asset": "BTC-PERP", "direction": "NONE", "setup_type": word, "stop_loss_invalidation": None,
                "entry_zone": None, "tp1": None, "tp2": None, "tp3": None, "active_start_ms": None,
                "active_end_ms": None, "risk_reward": None}
    up = regime == "bullish"
    ph, pl = plan["projected_high"], plan["projected_low"]
    anchor = pl if up else ph
    hour = next((h for h in plan.get("hours") or [] if h["t_ms"] >= anchor["t_ms"]), None) or (plan.get("hours") or [{}])[-1]
    median = hour.get("median", plan["origin"])
    zone = sorted([float(anchor["price"]), float(median)])
    stop = plan["invalidation"]["below" if up else "above"]
    final = ph["price"] if up else pl["price"]
    levels = plan.get("levels") or []
    between = sorted((lv["price"] for lv in levels
                      if ((zone[1] < lv["price"] < final) if up else (final < lv["price"] < zone[0]))),
                     reverse=not up)
    if len(between) >= 2:
        tps = [between[0], between[len(between) // 2], final]
    elif between:
        tps = [between[0], round((between[0] + final) / 2, 2), final]
    else:
        edge = zone[1] if up else zone[0]
        tps = [round(edge + (final - edge) * k / 3, 2) for k in (1, 2)] + [final]
    tps = [round(float(x), 2) for x in tps]
    near = [lv for lv in levels if lv["label"] in SETUP_WORD and min(abs(lv["price"] - z) / z for z in zone) <= SETUP_TOL]
    setup = (SETUP_WORD[near[0]["label"]] if near else
             "Pullback into the projected low" if up else "Rally into the projected high")
    if plan.get("magnitude"):
        setup += " · target widened"
    start = anchor["t_ms"]
    end = (ph if up else pl)["t_ms"]
    if end <= start:
        start, end = plan["created_ms"], end if end > plan["created_ms"] else plan["end_ms"]
    mid = sum(zone) / 2
    risk = abs(mid - stop)
    rr = round(abs(tps[-1] - mid) / risk, 1) if risk > 0 else None
    return {"asset": "BTC-PERP", "direction": "LONG" if up else "SHORT", "setup_type": setup,
            "stop_loss_invalidation": stop, "entry_zone": [round(zone[0], 2), round(zone[1], 2)],
            "tp1": tps[0], "tp2": tps[1], "tp3": tps[2], "active_start_ms": int(start), "active_end_ms": int(end),
            "risk_reward": rr}
