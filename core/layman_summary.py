"""His Section 6: the layman summary at the bottom of every desk, in his exact standardized format.

    📌 LAYMAN SUMMARY                      1-3 plain sentences: the state, the crowd's mood, who is paying
                                           and who is selling, and what must flip for a full move
    📊 Quantitative Swing Trade Framework  bias, entry zone, hard invalidation, Target 1 and Target 2
    ⏳ Time Reversal Windows (PKT)         Window 1 = the measured turning hours; Window 2 = cycle
                                           confluence, which we tested and do not trust (said so, not faked)
    ❌ Bias Invalidation Condition         the exact candle close that cancels the thesis

Everything comes from the desk's own deterministic analysis and its own-window derivatives
(core.indicators.desk_derivs) - never from a model's text - so it can be checked against the cards above.

The framework is geometry, not a forecast: the entry zone is the nearest level the bias would act from,
widened by a quarter ATR each side; the invalidation sits beyond that zone (the desk's own invalidation
when it does, otherwise half an ATR past the far edge); targets are the next two levels in the bias's
direction, nearest first. A sideways desk takes his example, "Range-Bound / Conditional Long on Floor
Retest" (or the short from the ceiling, whichever edge price is nearer)."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace as NS

ZONE_ATR = 0.25          # the entry zone: a quarter ATR either side of the level
STOP_ATR = 0.5           # the fallback invalidation: half an ATR past the zone's far edge
TF_WORDS = {"15m": "15-minute", "1h": "1-hour", "4h": "4-hour", "1d": "daily", "1w": "weekly", "1M": "monthly"}
HORIZON_WORDS = {"next 4h": "the next 4 hours", "next 24h": "the next 24 hours"}
PKT_OFFSET_MS = 5 * 3_600_000


def _money(x: float | None) -> str:
    return "—" if x is None else f"${x:,.0f}"


def _pkt(ms: int) -> str:
    from datetime import datetime, timezone
    return f"{datetime.fromtimestamp((ms + PKT_OFFSET_MS) / 1000, timezone.utc):%H:%M}"


def _conf(c: float) -> str:
    return "very strong" if c >= 0.85 else "strong" if c >= 0.60 else "moderate" if c >= 0.35 else "slight"


@dataclass(frozen=True)
class Framework:
    bias: str
    zone: tuple[float, float] | None
    invalidation: float | None
    targets: tuple[float | None, float | None]
    condition: str
    side: str                       # LONG, SHORT or NONE
    breakout: float | None = None   # a sideways desk: the far edge that turns the range into a trend


def _levels(a) -> tuple[list[float], list[float]]:
    below = sorted((lv.price for lv in a.levels if lv.kind == "support" and lv.price < a.price), reverse=True)
    above = sorted(lv.price for lv in a.levels if lv.kind == "resistance" and lv.price > a.price)
    return below, above


def framework(a) -> Framework:
    """Bias, entry zone, invalidation and two targets from the desk's levels; geometry always holds:
    a short's invalidation above its zone and targets below it, a long's the mirror image."""
    d, atr = a.direction, a.vol.atr
    tf = TF_WORDS.get(a.chart_tf, a.chart_tf)
    below, above = _levels(a)
    if atr is None or atr <= 0:
        return Framework("No clear bias: not enough history", None, None, (None, None), "No clear line yet.", "NONE")
    floor = below[0] if below else a.vol.exp_low
    ceiling = above[0] if above else a.vol.exp_high

    if d.direction == "BEARISH" or (d.direction == "NEUTRAL" and floor and ceiling
                                    and ceiling - a.price < a.price - floor):
        if ceiling is None:
            return Framework("Bearish, no level above to sell from", None, None, (None, None), "No clear line yet.", "NONE")
        zone = (ceiling - ZONE_ATR * atr, ceiling + ZONE_ATR * atr)
        inv = d.invalidation if (d.direction == "BEARISH" and d.invalidation and d.invalidation > zone[1]) \
            else zone[1] + STOP_ATR * atr
        lower = [p for p in below if p < zone[0]] + ([a.vol.exp_low] if a.vol.exp_low and a.vol.exp_low < zone[0] else [])
        lower = sorted(set(lower), reverse=True)
        bias = (f"Bearish ({_conf(d.confidence)}) / Sell a rally into {_money(ceiling)}" if d.direction == "BEARISH"
                else f"Range-Bound / Conditional Short on Ceiling Retest at {_money(ceiling)}")
        return Framework(bias, zone, inv, (lower[0] if lower else None, lower[1] if len(lower) > 1 else None),
                         f"A {tf} candle closing above {_money(inv)} cancels the short.", "SHORT",
                         floor if d.direction == "NEUTRAL" else None)

    if floor is None:
        return Framework("Bullish, no level below to buy from", None, None, (None, None), "No clear line yet.", "NONE")
    zone = (floor - ZONE_ATR * atr, floor + ZONE_ATR * atr)
    inv = d.invalidation if (d.direction == "BULLISH" and d.invalidation and d.invalidation < zone[0]) \
        else zone[0] - STOP_ATR * atr
    upper = [p for p in above if p > zone[1]] + ([a.vol.exp_high] if a.vol.exp_high and a.vol.exp_high > zone[1] else [])
    upper = sorted(set(upper))
    bias = (f"Bullish ({_conf(d.confidence)}) / Buy a dip into {_money(floor)}" if d.direction == "BULLISH"
            else f"Range-Bound / Conditional Long on Floor Retest at {_money(floor)}")
    return Framework(bias, zone, inv, (upper[0] if upper else None, upper[1] if len(upper) > 1 else None),
                     f"A {tf} candle closing below {_money(inv)} cancels the long.", "LONG",
                     ceiling if d.direction == "NEUTRAL" else None)


def _mood(fear_greed: int | None) -> str:
    if fear_greed is None:
        return ""
    word = ("extreme fear" if fear_greed < 25 else "fear" if fear_greed < 45 else "neutral" if fear_greed <= 55
            else "greed" if fear_greed < 75 else "extreme greed")
    return f"The crowd's mood is {word} ({fear_greed}/100)."


def _flow(derivs: dict | None) -> str:
    """Who is selling at market, who is paying to hold, and whether new bets are going on - over the
    desk's own window, in words."""
    if not derivs:
        return ""
    bits = []
    td = derivs.get("taker_delta") or {}
    if td.get("delta_btc") is not None:
        side = "selling" if td["delta_btc"] < 0 else "buying"
        bits.append(f"over the last {td['over']} there was more {side} than {'buying' if side == 'selling' else 'selling'} "
                    f"at market (net {abs(td['delta_btc']):,.0f} BTC {'sold' if side == 'selling' else 'bought'})")
    rate = (derivs.get("funding") or {}).get("latest_8h_rate")
    if rate is not None:
        bits.append("buyers are paying a high premium to stay in (a crowded long)" if rate >= 0.0003 else
                    "buyers pay a small premium to stay in" if rate > 0 else
                    "sellers are paying to stay in (a crowded short)" if rate < 0 else "neither side is paying to hold")
    ois = derivs.get("open_interest") or []
    if ois and ois[-1].get("change_pct") is not None:
        o = ois[-1]
        bits.append(f"open positions are {'growing' if o['change_pct'] > 0 else 'shrinking'} "
                    f"({o['change_pct']:+.1f}% in {o['over']})")
    return (bits[0][0].upper() + bits[0][1:] + ("; " + "; ".join(bits[1:]) if bits[1:] else "") + ".") if bits else ""


def lines(a, derivs: dict | None, fear_greed: int | None, now_ms: int) -> dict:
    """The four blocks, as data: the page renders them and tests read them."""
    from core.indicators.turn_window import next_window

    f = framework(a)
    d = a.direction
    horizon = HORIZON_WORDS.get(a.vol.horizon_label, a.vol.horizon_label)
    horizon = horizon if horizon.startswith("the ") else f"the {horizon}"
    state = {"BULLISH": f"Buyers have the edge for {horizon}: BTC is {_money(a.price)}, leaning higher ({_conf(d.confidence)}).",
             "BEARISH": f"Sellers have the edge for {horizon}: BTC is {_money(a.price)}, leaning lower ({_conf(d.confidence)}).",
             }.get(d.direction, f"No side has the edge for {horizon}: BTC is {_money(a.price)}, moving sideways.")
    tfw = TF_WORDS.get(a.chart_tf, a.chart_tf)
    if f.breakout is not None and f.side == "LONG":
        flip = (f"It picks a side on a {tfw} close above {_money(f.breakout)} (up) or below "
                f"{_money(f.invalidation)} (down).")
    elif f.breakout is not None and f.side == "SHORT":
        flip = (f"It picks a side on a {tfw} close above {_money(f.invalidation)} (up) or below "
                f"{_money(f.breakout)} (down).")
    elif f.side == "SHORT":
        flip = f"A full move up needs a {TF_WORDS.get(a.chart_tf, a.chart_tf)} close above {_money(f.invalidation)}."
    elif f.side == "LONG":
        flip = f"A full move down needs a {TF_WORDS.get(a.chart_tf, a.chart_tf)} close below {_money(f.invalidation)}."
    else:
        flip = ""
    summary = " ".join(s for s in (state, " ".join(x for x in (_mood(fear_greed), _flow(derivs)) if x), flip) if s)

    w = next_window(now_ms)
    from datetime import datetime, timezone
    day = datetime.fromtimestamp((w.start_ms + PKT_OFFSET_MS) / 1000, timezone.utc)
    utc = lambda ms: f"{datetime.fromtimestamp(ms / 1000, timezone.utc):%H:%M}"  # noqa: E731
    window1 = (f"{_pkt(w.start_ms)} – {_pkt(w.end_ms)} PKT {day:%a %d %b} ({utc(w.start_ms)}–{utc(w.end_ms)} UTC)"
               + (", happening now" if w.active else "") + ": about a third of BTC's turns happen in these hours")
    return {
        "summary": summary,
        "bias": f.bias,
        "entry": f"{_money(f.zone[0])} – {_money(f.zone[1])}" if f.zone else "—",
        "invalidation": _money(f.invalidation),
        "target1": _money(f.targets[0]), "target2": _money(f.targets[1]),
        "window1": window1,
        "window2": "None we trust: cycle timing was tested on a year of candles and timed turns no better than chance.",
        "condition": f.condition,
        "kill_zones": _kill_zones(now_ms),
    }


def _kill_zones(now_ms: int) -> str:
    """His three kill zones with what they measured: the ones near chance are said to be, not dropped."""
    from core.indicators.turn_window import ZONES, next_zones

    on = {w.zone.key: w.active for w in next_zones(now_ms)}
    edge, none = [], []
    for z in ZONES:                                   # the day's order, not soonest-first: it reads the same each time
        w = NS(active=on.get(z.key, False))
        pkt = z.hours_label.split("(")[1].rstrip(")")
        if z.near_chance:
            none.append(f"{z.name} ({pkt})")
        else:
            edge.append(f"{z.name} {pkt}{' (on now)' if w.active else ''}: {z.share_of_turns:.0%} of turns, "
                        f"{z.times_chance:.1f}x chance")
    text = "; ".join(edge)
    if none:
        text += (". " if text else "") + " and ".join(none) + ": no better than chance, not turning times"
    return text + "."
