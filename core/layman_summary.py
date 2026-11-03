"""His Section 9 (v22): the standardized summary at the bottom of every desk, in his layout.

    🧭 BIG PICTURE            1-3 plain sentences: the state, the crowd's mood, who is paying and who is selling
    🕒 TIMELINE               what is happening now and what to expect; the measured turning window (UTC + PKT)
    🎯 BOUNCE ZONE            the zone the desk acts from, with its plain reason
    📈 TARGETS                Target 1, 2 and 3 as the highs (or lows) to expect
    🛡 SAFETY LINE            the hard invalidation and the close that cancels the trade
    ✅ VERIFIED CONFLUENCES   the seven families (core.confluence): which agree, which oppose, which say nothing
    📣 DIRECTIVE              BUY / SELL the zone with three targets, or WAIT naming what is missing
    ⚠️ WARNING                the next news print, the dollar, the weekend

Everything comes from the desk's own deterministic analysis, its own-window derivatives
(core.indicators.desk_derivs) and the gate's evidence - never from a model's text - so it can be checked
against the cards above.

The framework is geometry, not a forecast: the entry zone is the nearest level the bias would act from,
widened by a quarter ATR each side; the invalidation sits beyond that zone (the desk's own invalidation
when it does, otherwise half an ATR past the far edge); targets are the next three levels in the bias's
direction, nearest first. A sideways desk takes his example, "Range-Bound / Conditional Long on Floor
Retest" (or the short from the ceiling, whichever edge price is nearer). The trade is cancelled by a
candle close beyond the safety line: the 4-hour candle on the four-hour and 24-hour desks (his rule), the
desk's own candle on the weekly and monthly ones."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace as NS

ZONE_ATR = 0.25          # the entry zone: a quarter ATR either side of the level
STOP_ATR = 0.5           # the fallback invalidation: half an ATR past the zone's far edge
TF_WORDS = {"15m": "15-minute", "1h": "1-hour", "4h": "4-hour", "1d": "daily", "1w": "weekly", "1M": "monthly"}
HORIZON_WORDS = {"next 4h": "the next 4 hours", "next 24h": "the next 24 hours"}
WARN_AHEAD_MS = {"next 4h": 24 * 3_600_000, "next 24h": 24 * 3_600_000}    # how far ahead the warning looks
WEEK_MS = 7 * 86_400_000
PKT_OFFSET_MS = 5 * 3_600_000


def _money(x: float | None) -> str:
    return "—" if x is None else f"${x:,.0f}"


def _pkt(ms: int) -> str:
    return f"{datetime.fromtimestamp((ms + PKT_OFFSET_MS) / 1000, timezone.utc):%H:%M}"


def _utc(ms: int) -> str:
    return f"{datetime.fromtimestamp(ms / 1000, timezone.utc):%H:%M}"


def _conf(c: float) -> str:
    return "very strong" if c >= 0.85 else "strong" if c >= 0.60 else "moderate" if c >= 0.35 else "slight"


def rule_tf(chart_tf: str) -> str:
    """The candle whose close cancels the trade: his 4-hour rule on the short desks, the desk's own
    candle where the chart is already a day or longer."""
    return chart_tf if chart_tf in ("1d", "1w", "1M") else "4h"


@dataclass(frozen=True)
class Framework:
    bias: str
    zone: tuple[float, float] | None
    invalidation: float | None
    targets: tuple[float | None, float | None]
    condition: str
    side: str                       # LONG, SHORT or NONE
    breakout: float | None = None   # a sideways desk: the far edge that turns the range into a trend
    target3: float | None = None
    level: float | None = None      # the level the zone is built on


def _levels(a) -> tuple[list[float], list[float]]:
    below = sorted((lv.price for lv in a.levels if lv.kind == "support" and lv.price < a.price), reverse=True)
    above = sorted(lv.price for lv in a.levels if lv.kind == "resistance" and lv.price > a.price)
    return below, above


def framework(a) -> Framework:
    """Bias, entry zone, invalidation and three targets from the desk's levels; geometry always holds:
    a short's invalidation above its zone and targets below it, a long's the mirror image."""
    d, atr = a.direction, a.vol.atr
    tf = TF_WORDS.get(rule_tf(a.chart_tf), a.chart_tf)
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
        lower = sorted(set(lower), reverse=True) + [None, None, None]
        bias = (f"Bearish ({_conf(d.confidence)}) / Sell a rally into {_money(ceiling)}" if d.direction == "BEARISH"
                else f"Range-Bound / Conditional Short on Ceiling Retest at {_money(ceiling)}")
        return Framework(bias, zone, inv, (lower[0], lower[1]),
                         f"A {tf} candle closing above {_money(inv)} cancels the short.", "SHORT",
                         floor if d.direction == "NEUTRAL" else None, lower[2], ceiling)

    if floor is None:
        return Framework("Bullish, no level below to buy from", None, None, (None, None), "No clear line yet.", "NONE")
    zone = (floor - ZONE_ATR * atr, floor + ZONE_ATR * atr)
    inv = d.invalidation if (d.direction == "BULLISH" and d.invalidation and d.invalidation < zone[0]) \
        else zone[0] - STOP_ATR * atr
    upper = [p for p in above if p > zone[1]] + ([a.vol.exp_high] if a.vol.exp_high and a.vol.exp_high > zone[1] else [])
    upper = sorted(set(upper)) + [None, None, None]
    bias = (f"Bullish ({_conf(d.confidence)}) / Buy a dip into {_money(floor)}" if d.direction == "BULLISH"
            else f"Range-Bound / Conditional Long on Floor Retest at {_money(floor)}")
    return Framework(bias, zone, inv, (upper[0], upper[1]),
                     f"A {tf} candle closing below {_money(inv)} cancels the long.", "LONG",
                     ceiling if d.direction == "NEUTRAL" else None, upper[2], floor)


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


def _warning(side: str, now_ms: int, ev, horizon_label: str, f=None, atr: float | None = None) -> str:
    """The next high-impact US print inside the window, a dollar move against the trade, the weekend, and any
    bigger level standing between the zone and Target 1 that does not bar the trade outright."""
    from core.confluence import levels_before_target
    bits = []
    if f is not None and f.zone and atr:
        for label, p in levels_before_target(side, f.zone, atr, f.targets[0], getattr(ev, "htf_levels", ()))[:2]:
            bits.append(f"a {label} at {_money(p)} stands between the zone and Target 1; expect a pause there")
    ahead = WARN_AHEAD_MS.get(horizon_label, WEEK_MS)
    prints = sorted((p for p in getattr(ev, "prints", ()) or ()
                     if getattr(p, "state", "") in ("scheduled", "flat", "positioned")
                     and now_ms < p.time_ms <= now_ms + ahead), key=lambda p: p.time_ms)
    if prints:
        p = prints[0]
        day = datetime.fromtimestamp(p.time_ms / 1000, timezone.utc)
        line = f"a high-impact US print, {p.title}, lands at {_utc(p.time_ms)} UTC ({_pkt(p.time_ms)} PKT) {day:%a}"
        if p.state == "positioned" and getattr(p, "runup_pct", None) is not None:
            line += f"; BTC has already moved {p.runup_pct:+.1f}% into it, so a print that only meets forecast can reverse that"
        bits.append(line)
    d = ((getattr(ev, "rates", None) or {}).get("DTWEXBGS") or {})
    if d.get("value") and d.get("change_1w") is not None:
        pct = d["change_1w"] / d["value"] * 100
        if side == "LONG" and pct >= 0.25:
            bits.append(f"the dollar has risen {pct:.1f}% this week, a headwind for a long")
        elif side == "SHORT" and pct <= -0.25:
            bits.append(f"the dollar has fallen {abs(pct):.1f}% this week, a headwind for a short")
    now = datetime.fromtimestamp(now_ms / 1000, timezone.utc)
    if now.weekday() >= 5:
        bits.append("weekend tape: thin volume, so moves run further and reverse faster than the levels suggest")
    elif now.weekday() == 4 and now.hour >= 18:
        bits.append("the weekend starts within hours: thin volume ahead")
    if not bits:
        return "Nothing scheduled: no high-impact US print, dollar move against the trade, weekend or bigger level inside this window."
    return "Watch: " + "; ".join(bits) + "."


def lines(a, derivs: dict | None, fear_greed: int | None, now_ms: int, evidence=None) -> dict:
    """The blocks, as data: the page renders them and tests read them."""
    from core import confluence
    from core.indicators.turn_window import next_window

    f = framework(a)
    d = a.direction
    horizon = HORIZON_WORDS.get(a.vol.horizon_label, a.vol.horizon_label)
    horizon = horizon if horizon.startswith("the ") else f"the {horizon}"
    lean = {"BULLISH": f"leaning higher ({_conf(d.confidence)})", "BEARISH": f"leaning lower ({_conf(d.confidence)})"
            }.get(d.direction, "moving sideways")
    state = {"BULLISH": f"Buyers have the edge for {horizon}: BTC is {_money(a.price)}, {lean}.",
             "BEARISH": f"Sellers have the edge for {horizon}: BTC is {_money(a.price)}, {lean}.",
             }.get(d.direction, f"No side has the edge for {horizon}: BTC is {_money(a.price)}, moving sideways.")
    tfw = TF_WORDS.get(rule_tf(a.chart_tf), a.chart_tf)
    # A sideways desk names the same floor and ceiling as the plain-English read above it; the setup's
    # stop is a separate line further out. Naming the stop as "where it turns down" gave the reader two
    # different down-levels on one screen.
    level = (f.zone[0] + f.zone[1]) / 2 if f.zone else None
    if f.breakout is not None and f.side == "LONG":
        flip = (f"It tilts lower on a {tfw} close below the {_money(level)} floor and higher above the "
                f"{_money(f.breakout)} ceiling; the long setup is cancelled below {_money(f.invalidation)}.")
    elif f.breakout is not None and f.side == "SHORT":
        flip = (f"It tilts higher on a {tfw} close above the {_money(level)} ceiling and lower below the "
                f"{_money(f.breakout)} floor; the short setup is cancelled above {_money(f.invalidation)}.")
    elif f.side == "SHORT":
        flip = f"A full move up needs a {tfw} close above {_money(f.invalidation)}."
    elif f.side == "LONG":
        flip = f"A full move down needs a {tfw} close below {_money(f.invalidation)}."
    else:
        flip = ""
    summary = " ".join(s for s in (state, " ".join(x for x in (_mood(fear_greed), _flow(derivs)) if x), flip) if s)

    # the gate: the desk's own derivatives and mood are part of the evidence even when the caller passes none
    ev = evidence or confluence.Evidence()
    if (ev.derivs is None and derivs is not None) or (ev.fear_greed is None and fear_greed is not None):
        ev = confluence.Evidence(**{**ev.__dict__, "derivs": ev.derivs if ev.derivs is not None else derivs,
                                    "fear_greed": ev.fear_greed if ev.fear_greed is not None else fear_greed})
    g = confluence.read(f, a, ev, now_ms)
    n = len(g.families) or len(confluence.FAMILIES)
    score = f"{len(g.agreeing)} of {n} agree"
    entry = f"{_money(f.zone[0])} – {_money(f.zone[1])}" if f.zone else "—"
    long = f.side == "LONG"
    beyond, kind = ("below", "floor") if long else ("above", "ceiling")
    targets = [t for t in (f.targets[0], f.targets[1], f.target3) if t is not None]

    if g.verdict == "TRADE":
        expect = (f"a {'bounce' if long else 'rejection'} from the zone toward {_money(targets[0]) if targets else 'the next level'}"
                  + (f", then {_money(targets[1])}" if len(targets) > 1 else "")
                  + f"; the trade is off on a {tfw} close {beyond} {_money(f.invalidation)}")
        directive = (f"{'BUY' if long else 'SELL'} {entry}. Targets {', '.join(_money(t) for t in targets)}. "
                     f"Out on a {tfw} close {beyond} {_money(f.invalidation)}.")
    elif g.verdict == "WAIT":
        expect = "no trade yet. " + g.reason
        directive = "WAIT. " + g.reason[len("WAIT: "):]
    else:
        expect = "no setup until this desk has a level to act from"
        directive = "NO SETUP. No level to trade from on this desk."
    phase_now = (f"What's happening: BTC is {_money(a.price)}, {lean}"
                 + (f"; the {kind} at {_money(f.level)} is the zone ({score})" if f.zone else "")
                 + f". What to expect: {expect}" + ("" if expect.endswith(".") else "."))
    zone_why = (f"the nearest {kind} {beyond} price, a quarter of a {TF_WORDS.get(a.chart_tf, a.chart_tf)} swing either side"
                + ("; backed by " + "; ".join(x.note for x in g.agreeing) if g.agreeing else "; nothing else backs it yet")
                if f.zone else "no level to build a zone on")
    safety = (f"{_money(f.invalidation)}. {f.condition} Until that close, a wick through the zone is noise, not a signal."
              if f.zone else f.condition)

    w = next_window(now_ms)
    day = datetime.fromtimestamp((w.start_ms + PKT_OFFSET_MS) / 1000, timezone.utc)
    window1 = (f"{_pkt(w.start_ms)} – {_pkt(w.end_ms)} PKT {day:%a %d %b} ({_utc(w.start_ms)}–{_utc(w.end_ms)} UTC)"
               + (", happening now" if w.active else "") + ": about a third of BTC's turns happen in these hours")
    return {
        "summary": summary,
        "bias": f.bias,
        "side": f.side,
        "verdict": g.verdict,
        "phase_now": phase_now,
        "entry": entry,
        "zone_why": zone_why,
        "invalidation": _money(f.invalidation),
        "safety": safety,
        "target1": _money(f.targets[0]), "target2": _money(f.targets[1]), "target3": _money(f.target3),
        "window1": window1,
        "window2": "None we trust: cycle timing was tested on a year of candles and timed turns no better than chance.",
        "condition": f.condition,
        "kill_zones": _kill_zones(now_ms),
        "score": score,
        "confluences": [{"key": x.key, "label": x.label, "side": x.side,
                         "agrees": None if x.side == "NONE" else x.side == f.side, "note": x.note}
                        for x in g.families],
        "directive": directive,
        "warning": _warning(f.side, now_ms, ev, a.vol.horizon_label, f, a.vol.atr),
        "bar": None if g.bar is None else {"label": g.bar.label, "price": g.bar.price, "note": g.bar.note},
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
    # two windows on one screen invite "which one?": they are the same afternoon, one hour apart
    return text + (". Window 1 (12:00-16:00 UTC) is the strongest four hours; the New York session "
                   "(13:00-17:00 UTC) overlaps it by three, so they are one afternoon, not two.")
