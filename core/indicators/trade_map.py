"""Active-trade mapping: scalp vs swing, structure above/below, stop quality, liquidation, re-anchor triggers."""
from __future__ import annotations

from dataclasses import dataclass

from core.data.types import FuturesSnapshot
from core.indicators.category import CategoryAnalysis

MAINT_MARGIN = 0.005


@dataclass
class TradeMap:
    classification: str
    direction: str
    entry: float
    leverage: float
    stop: float
    target: float
    liquidation_price: float
    dist_to_liq_pct: float
    risk_label: str
    rr: float | None
    structures_above: list[str]
    structures_below: list[str]
    stop_structural: bool
    warnings: list[str]
    anchor_triggers: list[str]


def _structures(analysis: CategoryAnalysis | None, tag: str) -> list[tuple[float, str]]:
    if analysis is None:
        return []
    out = []
    for n, v in analysis.ema.values.items():
        if v is not None:
            out.append((v, f"EMA{n} {tag}"))
    for l in analysis.levels:
        out.append((l.price, f"{l.kind.title()} {tag} ({l.touches}x)"))
    return out


def _fmt(s: tuple[float, str]) -> str:
    return f"{s[1]} @ {s[0]:,.0f}"


def map_trade(form: dict, live: CategoryAnalysis | None, intraday: CategoryAnalysis | None, futures: FuturesSnapshot) -> TradeMap:
    d, entry, lev, sl, tp = form["dir"], float(form["entry"]), float(form["lev"]), float(form["sl"]), float(form["tp"])
    price = live.price if live else (intraday.price if intraday else entry)

    if d == "LONG":
        liq = entry * (1 - 1 / lev + MAINT_MARGIN)
        dist = (price - liq) / price * 100
    else:
        liq = entry * (1 + 1 / lev - MAINT_MARGIN)
        dist = (liq - price) / price * 100
    risk = "EXTREME" if dist < 2.0 else "ELEVATED" if dist < 5.0 else "OPTIMAL"
    risk_dist = abs(entry - sl)
    rr = (abs(tp - entry) / risk_dist) if risk_dist > 0 else None

    atr15 = live.vol.atr if live and live.vol.atr else None
    half_range_1h = (intraday.vol.exp_high - intraday.price) if intraday and intraday.vol.exp_high else None
    scalp = (atr15 is not None and abs(entry - price) < 3 * atr15) and (half_range_1h is not None and abs(tp - entry) <= half_range_1h)
    classification = "SCALP" if scalp else "SWING"

    structs = _structures(live, live.chart_tf if live else "") + _structures(intraday, intraday.chart_tf if intraday else "")
    above = sorted([s for s in structs if s[0] > entry], key=lambda s: s[0] - entry)[:4]
    below = sorted([s for s in structs if s[0] < entry], key=lambda s: entry - s[0])[:4]

    if d == "LONG":
        stop_structural = any(sl <= p < entry for p, _ in structs)
    else:
        stop_structural = any(entry < p <= sl for p, _ in structs)

    warnings: list[str] = []
    if atr15 is not None and risk_dist < atr15:
        warnings.append(f"Stop sits inside 15m noise (1xATR = {atr15:,.0f})")
    if lev > 20:
        warnings.append(f"Leverage {lev:.0f}x leaves {dist:.1f}% to liquidation")
    if not stop_structural:
        warnings.append("Stop is not protected by any EMA or swing level")
    if rr is not None and rr < 1.5:
        warnings.append(f"Reward:risk {rr:.2f} below 1.5")
    if (d == "LONG" and sl >= entry) or (d == "SHORT" and sl <= entry):
        warnings.append("Stop is on the wrong side of entry")

    if futures.available:
        fr = futures.funding_rate or 0.0
        oi = futures.oi_change_24h_pct
        ls = futures.long_short_ratio
        triggers = [
            f"Funding flips sign (now {fr:+.4%})",
            f"OI 24h change crosses +/-5% (now {oi:+.1f}%)" if oi is not None else "OI 24h change crosses +/-5%",
            f"Long/short ratio crosses 1.0 (now {ls:.2f})" if ls is not None else "Long/short ratio crosses 1.0",
        ]
    else:
        triggers = ["Funding / OI / long-short triggers inactive (futures data unavailable)"]
    sq = live.squeeze if live else None
    triggers.append(f"Squeeze score reaches 60 (now {sq.score})" if sq and sq.score is not None else "Squeeze score reaches 60")
    triggers.append("Trap classification changes (Director report, Phase 2)")
    triggers.append("Curated macro event inside 24h")

    return TradeMap(classification, d, entry, lev, sl, tp, liq, dist, risk, rr,
                    [_fmt(s) for s in above], [_fmt(s) for s in below], stop_structural, warnings, triggers)
