"""Fibonacci reversal window: a Fibonacci time zone at (or next to) the current bar, combined with price
confluence — support/resistance, a recent regular RSI divergence, an RSI extreme, a 0.5/0.618
retracement, a trendline, or a forming reversal pattern. Time alone is not a signal; time plus
confluence is."""
from __future__ import annotations

from dataclasses import dataclass, field

from core.config import TF_MINUTES

REVERSAL_PATTERNS = ("Double top", "Double bottom", "Head and shoulders", "Inverse head and shoulders",
                     "Rising wedge", "Falling wedge")


@dataclass
class ReversalWindow:
    status: str                 # ACTIVE | TIME_ONLY | UPCOMING | NONE
    bias: str                   # bullish | bearish | mixed | none
    zone_k: int | None
    zone_ms: int | None
    bars_away: int | None       # 0 = this bar, negative = just passed
    window_bars: int
    confluence: list[str] = field(default_factory=list)


def reversal_window(a) -> ReversalWindow:
    fib = a.fib
    if fib is None or not fib.zones:
        return ReversalWindow("NONE", "none", None, None, None, 0)
    window = min(3, max(1, fib.unit_bars // 3))
    nearest = min(fib.zones, key=lambda z: abs(z.bars_from_now))
    upcoming = next((z for z in fib.zones if z.bars_from_now > 0), None)
    in_window = abs(nearest.bars_from_now) <= window

    price = a.price
    atr = a.vol.atr or price * 0.005
    bull = bear = 0
    conf: list[str] = []

    near_levels = sorted((l for l in a.levels if l.touches >= 2 and abs(l.price - price) <= 0.5 * atr),
                         key=lambda l: abs(l.price - price))
    if near_levels:
        l = near_levels[0]
        if l.kind == "resistance":
            bear += 1
            conf.append(f"at resistance {l.price:,.0f} ({l.touches} touches)")
        else:
            bull += 1
            conf.append(f"at support {l.price:,.0f} ({l.touches} touches)")

    last_ms = int(a.chart_df["timestamp"].iloc[-1])
    horizon_ms = 10 * TF_MINUTES[a.chart_tf] * 60_000
    for d in a.divergences:
        if d.kind == "regular" and last_ms - d.price_points[1][0] <= horizon_ms:
            if d.direction == "bearish":
                bear += 1
            else:
                bull += 1
            conf.append(f"{d.label} RSI divergence")

    r = a.rsi.dropna()
    if len(r):
        rv = float(r.iloc[-1])
        if rv >= 70:
            bear += 1
            conf.append(f"RSI {rv:.0f} overbought")
        elif rv <= 30:
            bull += 1
            conf.append(f"RSI {rv:.0f} oversold")

    if a.fib_retr is not None:
        for ratio in (0.5, 0.618):
            lvl = a.fib_retr.levels[ratio]
            if abs(price - lvl) <= 0.3 * atr:
                if a.fib_retr.leg == "up":
                    bull += 1
                else:
                    bear += 1
                conf.append(f"at the {ratio} retracement {lvl:,.0f}")
                break

    for t in a.trendlines:
        if abs(t.value_now - price) <= 0.5 * atr:
            if t.kind == "resistance":
                bear += 1
            else:
                bull += 1
            conf.append(f"at {t.kind} trendline {t.value_now:,.0f}")

    for p in a.patterns:
        if p.status == "forming" and p.name in REVERSAL_PATTERNS:
            if p.bias == "bearish":
                bear += 1
            elif p.bias == "bullish":
                bull += 1
            conf.append(f"{p.name} forming")

    bias = "bearish" if bear > bull else "bullish" if bull > bear else ("mixed" if bear else "none")
    if in_window:
        status = "ACTIVE" if conf else "TIME_ONLY"
        zone = nearest
    elif upcoming is not None:
        status, zone = "UPCOMING", upcoming
    else:
        status, zone = "NONE", None
    return ReversalWindow(status=status, bias=bias, zone_k=zone.k if zone else None,
                          zone_ms=zone.timestamp_ms if zone else None, bars_away=zone.bars_from_now if zone else None,
                          window_bars=window, confluence=conf)
