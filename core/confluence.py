"""His v22 Section 9: the confluence gate. A desk's trade box opens only when at least three of seven
independent families of evidence agree with the zone the desk wants to act from; fewer is a WAIT that names
what is missing, and two families pointing the other way is a WAIT that names the conflict.

The seven families, each judged on its own feed so no single read can open the gate on its own:

    pattern       a harmonic completing inside the zone, on the desk's own timeframes
    options       a put wall (for a long) or call wall (for a short) inside the zone
    flow          who is winning at market on the desk's chart: absorption at the level, or flow confirming
    positioning   a crowded side to force out - read the contrarian way (Hammad): extreme fear plus a
                  crowded short is a bullish trigger, extreme greed plus a crowded long a bearish one
    structure     a failed break (the trap it is), a sweep that was reclaimed, or a value-area edge in the zone
    macro         the dollar and the 2-year yield over the week, and which side the last news print trapped
    timing        the measured turning window, counting only while it is on

Everything here is code over the desk's own numbers; nothing comes from a model's text. Notes are written
for the layman summary, so they name no indicator."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

LONG, SHORT, NONE = "LONG", "SHORT", "NONE"
NEED = 3                          # families that must agree before the trade box opens
ZONE_PAD_ATR = 0.25               # a level counts as "at the zone" this far outside it
FEAR_EXTREME, GREED_EXTREME = 25, 75
PREMIUM_CROWDED_LONG = 0.0003     # 8-hour perpetual premium: a high price to stay long
PREMIUM_CROWDED_SHORT = -0.0001   # a negative premium: shorts are paying
LONG_ACCOUNTS_CROWDED = 0.60      # share of accounts long
SHORT_ACCOUNTS_CROWDED = 0.45
DOLLAR_WEEK_PCT = 0.25            # the broad dollar index, one-week change, in percent
YIELD_WEEK_PTS = 0.10             # the US 2-year yield, one-week change, in percentage points
PKT_MS = 5 * 3_600_000

FAMILIES = (("pattern", "a pattern completing at the zone"),
            ("options", "an options wall at the zone"),
            ("flow", "a turn in the order flow"),
            ("positioning", "a crowded side to force out"),
            ("structure", "a failed break, a reclaimed sweep or a value edge"),
            ("macro", "the dollar, yields and the news prints"),
            ("timing", "the turning window"))
LABEL = dict(FAMILIES)


@dataclass(frozen=True)
class Evidence:
    """What the desk screen hands the gate; every field may be missing, and a missing feed is a family that
    says nothing, never one that agrees."""
    harmonics: tuple = ()             # core.indicators.harmonics.Harmonic, the desk's own timeframes
    derivs: dict | None = None        # core.indicators.desk_derivs.read(...)
    fear_greed: int | None = None
    futures: object | None = None     # FuturesSnapshot: long_account_pct, long_short_ratio
    breaks: tuple = ()                # core.indicators.vwap_break.BreakRead per timeframe
    liquidity: dict | None = None     # timeframe -> core.indicators.liquidity.summarise(...)
    prints: tuple = ()                # core.indicators.event_trap.EventRead
    rates: dict | None = None         # core.agents.context._rates(m)


@dataclass(frozen=True)
class Family:
    key: str
    label: str
    side: str       # LONG, SHORT or NONE
    note: str       # one plain clause, no indicator words


@dataclass(frozen=True)
class Gate:
    side: str                      # the zone's side: LONG, SHORT or NONE
    verdict: str                   # TRADE, WAIT or NO SETUP
    families: tuple[Family, ...]
    reason: str
    agreeing: tuple[Family, ...] = field(default=())
    opposing: tuple[Family, ...] = field(default=())

    @property
    def missing(self) -> tuple[Family, ...]:
        return tuple(f for f in self.families if f.side == NONE)


def _money(x: float | None) -> str:
    return "—" if x is None else f"${x:,.0f}"


def _at(level: float | None, zone: tuple[float, float], atr: float) -> bool:
    return level is not None and zone[0] - ZONE_PAD_ATR * atr <= level <= zone[1] + ZONE_PAD_ATR * atr


def _other(side: str) -> str:
    return {LONG: SHORT, SHORT: LONG}.get(side, NONE)


def _vote(key: str, votes: list[tuple[str, str]], nothing: str, both: str) -> Family:
    """One family from several reads: all one way is that way; reads both ways cancel and say so."""
    sides = {s for s, _ in votes}
    if not votes:
        return Family(key, LABEL[key], NONE, nothing)
    if len(sides) > 1:
        return Family(key, LABEL[key], NONE, both + ": " + "; ".join(n for _, n in votes))
    return Family(key, LABEL[key], votes[0][0], "; ".join(n for _, n in votes))


# ---------- the seven families ----------
def pattern(harmonics, zone: tuple[float, float], atr: float) -> Family:
    found = []
    for h in harmonics or ():
        if getattr(h, "stage", "") not in ("in_zone", "completed"):
            continue
        lo, hi = h.prz
        if lo <= zone[1] + ZONE_PAD_ATR * atr and hi >= zone[0] - ZONE_PAD_ATR * atr:
            side = LONG if h.side == "bullish" else SHORT
            found.append((side, f"a {h.name} {'completing' if h.stage == 'completed' else 'in its zone'} at "
                                f"{_money(lo)} – {_money(hi)}"))
    return _vote("pattern", found, "no pattern completing at the zone", "patterns disagree at the zone")


def options(derivs: dict | None, zone: tuple[float, float], atr: float) -> Family:
    g = (derivs or {}).get("gamma") or {}
    votes = []
    if _at(g.get("put_wall"), zone, atr):
        votes.append((LONG, f"a put wall at {_money(g['put_wall'])} sits in the zone"))
    if _at(g.get("call_wall"), zone, atr):
        votes.append((SHORT, f"a call wall at {_money(g['call_wall'])} sits in the zone"))
    return _vote("options", votes, "no options wall at the zone", "walls on both sides of the zone")


def flow(chart_tf: str, liquidity: dict | None) -> Family:
    state = ((liquidity or {}).get(chart_tf) or {}).get("cumulative_delta", {}).get("state")
    word = {"bullish absorption": (LONG, "price was pushed down but buyers took everything offered"),
            "bearish absorption": (SHORT, "price was pushed up but sellers met every bid"),
            "confirming up": (LONG, "buyers are winning at market as price rises"),
            "confirming down": (SHORT, "sellers are winning at market as price falls")}.get(state)
    if word is None:
        return Family("flow", LABEL["flow"], NONE, "no turn in the order flow yet" if state else
                      "order flow not readable on this chart")
    return Family("flow", LABEL["flow"], word[0], word[1])


def positioning(fear_greed: int | None, futures, derivs: dict | None) -> Family:
    """Contrarian by design: the crowded side is the one that gets forced out."""
    votes = []
    if fear_greed is not None:
        if fear_greed < FEAR_EXTREME:
            votes.append((LONG, f"extreme fear ({fear_greed}/100)"))
        elif fear_greed > GREED_EXTREME:
            votes.append((SHORT, f"extreme greed ({fear_greed}/100)"))
    pct = getattr(futures, "long_account_pct", None)
    ratio = getattr(futures, "long_short_ratio", None)
    if pct is None and ratio is not None:
        pct = ratio / (1 + ratio)
    if pct is not None:
        if pct <= SHORT_ACCOUNTS_CROWDED:
            votes.append((LONG, f"a crowded short ({pct:.0%} of accounts long)"))
        elif pct >= LONG_ACCOUNTS_CROWDED:
            votes.append((SHORT, f"a crowded long ({pct:.0%} of accounts long)"))
    rate = ((derivs or {}).get("funding") or {}).get("latest_8h_rate")
    if rate is not None:
        if rate <= PREMIUM_CROWDED_SHORT:
            votes.append((LONG, "shorts are paying to hold"))
        elif rate >= PREMIUM_CROWDED_LONG:
            votes.append((SHORT, "longs are paying a high premium to hold"))
    fam = _vote("positioning", votes, "no crowded side to force out", "crowded both ways")
    if fam.side != NONE:
        who = "shorts" if fam.side == LONG else "longs"
        fam = Family(fam.key, fam.label, fam.side, f"{fam.note}: the {who} are the crowd that gets forced out")
    return fam


def structure(chart_tf: str, zone: tuple[float, float], atr: float, breaks, liquidity: dict | None) -> Family:
    """The first thing that fires, chart timeframe first: a break read, then a sweep, then a value edge."""
    reads = sorted((b for b in breaks or () if getattr(b, "available", False)),
                   key=lambda b: 0 if b.timeframe == chart_tf else 1)
    for b in reads:
        up = b.direction == "up"
        if b.state == "failed":
            return Family("structure", LABEL["structure"], SHORT if up else LONG,
                          f"the {b.timeframe} break {'above' if up else 'below'} {_money(b.edge)} failed: "
                          f"a trap for the {'buyers' if up else 'sellers'}")
        if b.state in ("qualified", "pullback"):
            return Family("structure", LABEL["structure"], LONG if up else SHORT,
                          f"a clean {b.timeframe} break {'above' if up else 'below'} {_money(b.edge)}"
                          + (", now pulling back to it" if b.state == "pullback" else ""))
    tfs = [chart_tf] + [tf for tf in (liquidity or {}) if tf != chart_tf]
    for tf in tfs:
        lq = (liquidity or {}).get(tf) or {}
        pools = lq.get("pools") or {}
        for side_key, side in (("sellside", LONG), ("buyside", SHORT)):
            p = pools.get(side_key) or {}
            if p.get("swept") and p.get("reclaimed") and _at(p.get("price"), zone, atr):
                return Family("structure", LABEL["structure"], side,
                              f"the {'lows' if side == LONG else 'highs'} at {_money(p['price'])} were swept and "
                              f"reclaimed on the {tf}")
        vp = lq.get("volume_profile") or {}
        if vp.get("available"):
            if _at(vp.get("value_area_low"), zone, atr):
                return Family("structure", LABEL["structure"], LONG,
                              f"the zone is the bottom edge of where the {tf} volume traded ({_money(vp['value_area_low'])})")
            if _at(vp.get("value_area_high"), zone, atr):
                return Family("structure", LABEL["structure"], SHORT,
                              f"the zone is the top edge of where the {tf} volume traded ({_money(vp['value_area_high'])})")
    return Family("structure", LABEL["structure"], NONE, "no failed break, reclaimed sweep or value edge at the zone")


def macro(rates: dict | None, prints) -> Family:
    votes = []
    r = rates or {}
    d = r.get("DTWEXBGS") or {}
    if d.get("value") and d.get("change_1w") is not None:
        pct = d["change_1w"] / d["value"] * 100
        if pct <= -DOLLAR_WEEK_PCT:
            votes.append((LONG, f"the dollar fell {abs(pct):.1f}% this week, a tailwind"))
        elif pct >= DOLLAR_WEEK_PCT:
            votes.append((SHORT, f"the dollar rose {pct:.1f}% this week, a headwind"))
    y = r.get("DGS2") or {}
    if y.get("change_1w") is not None:
        if y["change_1w"] <= -YIELD_WEEK_PTS:
            votes.append((LONG, f"the 2-year yield fell {abs(y['change_1w']):.2f} points this week"))
        elif y["change_1w"] >= YIELD_WEEK_PTS:
            votes.append((SHORT, f"the 2-year yield rose {y['change_1w']:.2f} points this week"))
    trapped = [(p, getattr(p, "trapped", None)) for p in prints or ()
               if getattr(p, "state", "") in ("sell_the_news", "spike_faded")]
    for p, who in trapped[-1:]:
        if who in ("longs", "shorts"):
            votes.append((SHORT if who == "longs" else LONG, f"the {p.title} print trapped the {who}"))
    return _vote("macro", votes, "no dollar, yield or news-print lean this week", "macro is mixed")


def timing(side: str, now_ms: int) -> Family:
    from core.indicators.turn_window import next_window
    w = next_window(now_ms)
    hm = lambda ms, off=0: f"{datetime.fromtimestamp((ms + off) / 1000, timezone.utc):%H:%M}"  # noqa: E731
    span = f"{hm(w.start_ms)}–{hm(w.end_ms)} UTC, {hm(w.start_ms, PKT_MS)}–{hm(w.end_ms, PKT_MS)} PKT"
    if w.active:
        return Family("timing", LABEL["timing"], side, f"the turning window is on now ({span})")
    day = datetime.fromtimestamp(w.start_ms / 1000, timezone.utc)
    return Family("timing", LABEL["timing"], NONE, f"next turning window {span} {day:%a %d %b}")


# ---------- the gate ----------
def judge(side: str, zone: tuple[float, float], atr: float, chart_tf: str, ev: Evidence, now_ms: int) -> tuple[Family, ...]:
    return (pattern(ev.harmonics, zone, atr),
            options(ev.derivs, zone, atr),
            flow(chart_tf, ev.liquidity),
            positioning(ev.fear_greed, ev.futures, ev.derivs),
            structure(chart_tf, zone, atr, ev.breaks, ev.liquidity),
            macro(ev.rates, ev.prints),
            timing(side, now_ms))


def gate(side: str, families: tuple[Family, ...]) -> Gate:
    if side not in (LONG, SHORT):
        return Gate(NONE, "NO SETUP", families, "NO SETUP: no level to trade from.")
    agreeing = tuple(f for f in families if f.side == side)
    opposing = tuple(f for f in families if f.side == _other(side))
    n = len(families)
    if len(opposing) >= 2:
        reason = (f"WAIT: {' and '.join(f.label for f in opposing)} point the other way: "
                  + "; ".join(f.note for f in opposing) + ".")
        return Gate(side, "WAIT", families, reason, agreeing, opposing)
    if len(agreeing) >= NEED:
        reason = f"TRADE: {len(agreeing)} of {n} confluences agree"
        if opposing:
            reason += f"; {opposing[0].label} points the other way: {opposing[0].note}"
        return Gate(side, "TRADE", families, reason + ".", agreeing, opposing)
    missing = [f.label for f in families if f.side != side]
    reason = (f"WAIT: {len(agreeing)} of {n} confluences agree, {NEED} are needed. Missing: "
              + ", ".join(missing) + ".")
    return Gate(side, "WAIT", families, reason, agreeing, opposing)


def read(f, a, ev: Evidence | None, now_ms: int) -> Gate:
    """The gate for a desk: `f` is its layman framework (side and zone), `a` its analysis."""
    ev = ev or Evidence()
    atr = getattr(getattr(a, "vol", None), "atr", None)
    if f.side not in (LONG, SHORT) or f.zone is None or not atr:
        return gate(NONE, ())
    return gate(f.side, judge(f.side, f.zone, atr, a.chart_tf, ev, now_ms))
