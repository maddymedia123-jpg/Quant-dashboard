"""Dealer gamma exposure (GEX) from the Deribit BTC options chain: the regime bar's evidence.

For every option: Black-Scholes gamma at its own implied volatility, times open interest (Deribit BTC
options are 1 BTC each), times S^2 x 1% - dollars of delta dealers must trade for a 1% move. Summed
across the chain at a range of hypothetical prices, that gives a gamma profile:

* Zero gamma level (ZGL / gamma flip): where the net profile crosses zero. Above it dealers are net long
  gamma and hedge AGAINST moves - they sell rallies and buy dips, which suppresses volatility (balance,
  chop). Below it they are net short gamma and hedge WITH moves (imbalance, expansion).
* Call wall: the strike carrying the most call gamma - a ceiling dealers sell into.
* Put wall: the strike carrying the most put gamma - a floor dealers buy into.
* Max gamma: the strike with the largest net gamma - where price is pulled into settlement.

The sign convention is the standard naive one: call gamma counts positive, put gamma negative - i.e.
dealers are assumed long the calls customers sold them and short the puts customers bought. Nobody
publishes real dealer positioning; this assumption is how every public GEX figure is built, and the panel
says so. Expiries within the next 30 days are used: gamma concentrates near expiry, and the weekly and
daily expiries are where his framework looks.

Each desk also gets its own profile, from only the options that settle inside its window (DESK_DAYS):
a 4-hour desk reading a put wall built from options three weeks out is reading the wrong clock, and the
monthly desk needs the quarterly that a 30-day cut leaves out (his v15, Section 2).

Deribit expiries settle at 08:00 UTC on the expiry date. Time is measured from the moment the chain was
quoted (the response's own server time), not from whenever the page happens to render."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from core.indicators.desk_derivs import WINDOWS

YEAR_MS = 365.0 * 86_400_000
HORIZON_DAYS = 30
GRID = 161                 # hypothetical prices from 80% to 120% of spot
GRID_LO, GRID_HI = 0.80, 1.20
WALL_RANGE = 0.25          # walls are looked for within +-25% of spot
# options settling inside each desk's window: the same expiry windows its max pain is read from
DESK_DAYS = {desk: w.expiry_days for desk, w in WINDOWS.items()}
MIN_T_YEARS = 1.0 / (365 * 24 * 4)     # 15 minutes: an option this close to settling has no meaningful gamma


@dataclass(frozen=True)
class GammaProfile:
    available: bool
    spot: float | None = None
    net_gex_usd: float | None = None          # at spot, dollars of hedging per 1% move
    zero_gamma: float | None = None
    call_wall: float | None = None
    put_wall: float | None = None
    max_gamma: float | None = None
    regime: str = "unknown"                    # positive, negative, unknown
    expiries: tuple[str, ...] = ()
    options_used: int = 0
    note: str = ""
    horizon_days: float = HORIZON_DAYS
    profile: tuple[tuple[float, float], ...] = field(default_factory=tuple)

    def to_state(self) -> dict:
        r = lambda v: None if v is None else round(v, 2)  # noqa: E731
        return {"available": self.available, "regime": self.regime, "spot": r(self.spot),
                "net_gex_usd_per_1pct": r(self.net_gex_usd), "zero_gamma_level": r(self.zero_gamma),
                "call_wall": r(self.call_wall), "put_wall": r(self.put_wall), "max_gamma_strike": r(self.max_gamma),
                "expiries": list(self.expiries), "options_used": self.options_used, "note": self.note,
                "horizon_days": self.horizon_days,
                "assumption": "naive dealer sign: calls positive, puts negative"}


def bs_gamma(spot: float, strike: float, t_years: float, vol: float) -> float:
    """Black-Scholes gamma per unit of the underlying, zero rates (crypto options price off the forward)."""
    if spot <= 0 or strike <= 0 or t_years <= 0 or vol <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + 0.5 * vol * vol * t_years) / (vol * math.sqrt(t_years))
    return math.exp(-0.5 * d1 * d1) / (math.sqrt(2 * math.pi) * spot * vol * math.sqrt(t_years))


def _expiry_ms(code: str) -> int | None:
    try:
        d = datetime.strptime(code, "%d%b%y")
    except ValueError:
        return None
    return int(datetime(d.year, d.month, d.day, 8, tzinfo=timezone.utc).timestamp() * 1000)


def chain_rows(items: list[dict], now_ms: int, horizon_days: float = HORIZON_DAYS) -> list[dict]:
    """Options within the horizon, with what the gamma needs. Rows missing a strike, an IV or open
    interest are left out rather than given a guessed value."""
    out = []
    for it in items:
        try:
            _, code, strike, kind = str(it["instrument_name"]).split("-")
            k = float(strike)
        except (KeyError, ValueError):
            continue
        exp = _expiry_ms(code)
        iv, oi = it.get("mark_iv"), it.get("open_interest")
        if exp is None or not iv or not oi or kind not in ("C", "P"):
            continue
        t = (exp - now_ms) / YEAR_MS
        if t < MIN_T_YEARS or t > horizon_days / 365.0:
            continue
        out.append({"strike": k, "kind": kind, "oi": float(oi), "vol": float(iv) / 100.0, "t": t, "expiry": code})
    return out


def _gex(rows: list[dict], s: float) -> tuple[float, dict[float, float], dict[float, float]]:
    """Net GEX at price s, and per-strike call and put GEX (both as positive magnitudes)."""
    net, calls, puts = 0.0, {}, {}
    for r in rows:
        g = bs_gamma(s, r["strike"], r["t"], r["vol"]) * r["oi"] * s * s * 0.01
        if r["kind"] == "C":
            net += g
            calls[r["strike"]] = calls.get(r["strike"], 0.0) + g
        else:
            net -= g
            puts[r["strike"]] = puts.get(r["strike"], 0.0) + g
    return net, calls, puts


def profile(items: list[dict], spot: float | None, now_ms: int, horizon_days: float = HORIZON_DAYS) -> GammaProfile:
    if not spot or spot <= 0:
        return GammaProfile(False, note="no index price in the options chain")
    rows = chain_rows(items, now_ms, horizon_days)
    if not rows:
        return GammaProfile(False, spot=spot, note=f"no options with IV and open interest in the next {horizon_days:g} days")

    net, calls, puts = _gex(rows, spot)
    near = lambda k: abs(k - spot) <= WALL_RANGE * spot  # noqa: E731
    call_wall = max((k for k in calls if near(k)), key=lambda k: calls[k], default=None)
    put_wall = max((k for k in puts if near(k)), key=lambda k: puts[k], default=None)
    strikes = {k for k in list(calls) + list(puts) if near(k)}
    max_gamma = max(strikes, key=lambda k: abs(calls.get(k, 0.0) - puts.get(k, 0.0)), default=None)

    # the profile across hypothetical prices, and the zero crossing nearest to spot
    grid = [spot * (GRID_LO + (GRID_HI - GRID_LO) * i / (GRID - 1)) for i in range(GRID)]
    curve = [(s, _gex(rows, s)[0]) for s in grid]
    flips = []
    for (s1, g1), (s2, g2) in zip(curve, curve[1:]):
        if g1 == 0:
            flips.append(s1)
        elif (g1 < 0) != (g2 < 0):
            flips.append(s1 + (s2 - s1) * (-g1) / (g2 - g1))      # linear interpolation of the crossing
    zero = min(flips, key=lambda s: abs(s - spot)) if flips else None

    return GammaProfile(
        True, horizon_days=horizon_days, spot=spot, net_gex_usd=net, zero_gamma=zero, call_wall=call_wall, put_wall=put_wall,
        max_gamma=max_gamma, regime="positive" if net > 0 else "negative" if net < 0 else "unknown",
        expiries=tuple(sorted({r["expiry"] for r in rows}, key=lambda c: _expiry_ms(c) or 0)),
        options_used=len(rows),
        note="" if zero is not None else "the profile does not cross zero within 20% of spot",
        profile=tuple((round(s, 2), round(g, 2)) for s, g in curve))


def desk_profiles(items: list[dict], spot: float | None, now_ms: int) -> dict[str, dict]:
    """One gamma profile per desk, each from its own expiries."""
    return {desk: profile(items, spot, now_ms, days).to_state() for desk, days in DESK_DAYS.items()}
