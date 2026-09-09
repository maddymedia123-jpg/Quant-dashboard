"""Build the compact, domain-scoped data payload each agent receives.

Nothing here invents a number: unavailable snapshots are omitted and listed under `unavailable`."""
from __future__ import annotations

import math
from typing import Any

from core.config import MACRO_EVENTS
from core.data.types import MarketSnapshot
from core.indicators.category import CategoryAnalysis
from core.indicators.direction import cvd_slope

DOMAIN_OF = {"B1": "options", "R1": "options", "B2": "leverage", "R2": "leverage", "B3": "liquidity", "R3": "liquidity",
             "B4": "flow", "R4": "flow", "B5": "sentiment", "R5": "sentiment"}
_EXCLUDE = {"available", "fetched_at", "error", "source"}


def _round(v: Any, nd: int = 6) -> Any:
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, nd)
    if isinstance(v, dict):
        return {k: _round(x, nd) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_round(x, nd) for x in v]
    return v


def _dump(snapshot, drop: tuple[str, ...] = ()) -> dict:
    d = snapshot.model_dump(exclude=_EXCLUDE | set(drop))
    return _round(d)


def category_brief(a: CategoryAnalysis) -> dict:
    rsi_last = float(a.rsi.dropna().iloc[-1]) if a.rsi.notna().any() else None
    return _round({
        "chart_tf": a.chart_tf, "context_tf": a.context_tf, "price": a.price,
        "direction": a.direction.direction, "confidence": a.direction.confidence, "drivers": a.direction.drivers,
        "anchor": a.direction.anchor, "invalidation": a.direction.invalidation, "flips_if": a.direction.flips_if,
        "ema_state": a.ema.state, "ema": {str(k): v for k, v in a.ema.values.items()},
        "context_ema_state": a.ctx_ema.state if a.ctx_ema else None,
        "rsi": rsi_last, "divergences": [d.label for d in a.divergences],
        "regime": a.vol.regime, "atr": a.vol.atr, "atr_pct": a.vol.atr_pct,
        "expected_range": {"horizon": a.vol.horizon_label, "low": a.vol.exp_low, "high": a.vol.exp_high, "sigma_pct": a.vol.sigma_pct},
        "sigma2_band": {"low": a.vol.sigma2_dn, "high": a.vol.sigma2_up, "mean20": a.vol.mean20},
        "levels": [{"price": l.price, "kind": l.kind, "touches": l.touches} for l in a.levels],
        "trendlines": [{"kind": t.kind, "slope_per_bar": t.slope, "value_now": t.value_now, "r2": t.r2} for t in a.trendlines],
        "fib_upcoming_zones": len(a.fib.upcoming) if a.fib else 0,
        "squeeze": {"score": a.squeeze.score, "direction": a.squeeze.direction, "drivers": a.squeeze.drivers},
        "cvd_slope_5bars": cvd_slope(a.chart_df),
        "price_change_24h_pct": a.price_change_24h_pct,
    })


def _calendar_events(m: MarketSnapshot) -> dict:
    if m.calendar.available and m.calendar.events:
        hi = [{"title": e["title"], "country": e["country"], "impact": e["impact"], "date": e["date"],
               "forecast": e["forecast"], "previous": e["previous"]}
              for e in m.calendar.events if e.get("impact") == "High" and e.get("country") in ("USD",)]
        return {"source": m.calendar.source, "usd_high_impact_this_week": hi[:12]}
    return {"source": "curated (calendar unavailable)", "events": MACRO_EVENTS}


def common_payload(m: MarketSnapshot, analyses: dict[str, CategoryAnalysis]) -> dict:
    live = analyses.get("live") or next(iter(analyses.values()), None)
    return {
        "asset": "BTC/USD",
        "spot": _round({"price": m.spot.last, "source": m.spot.source, "price_change_24h_pct": live.price_change_24h_pct if live else None}),
        "categories": {k: category_brief(a) for k, a in analyses.items()},
        "macro_calendar": _calendar_events(m),
        "generated_at_utc": m.generated_at.strftime("%Y-%m-%d %H:%M UTC"),
    }


def domain_payload(domain: str, m: MarketSnapshot, analyses: dict[str, CategoryAnalysis]) -> tuple[dict, list[str]]:
    data: dict[str, Any] = {}
    unavailable: list[str] = []

    def take(name: str, **kw):
        snap = getattr(m, name)
        if snap.available:
            data[name] = _dump(snap, **kw)
        else:
            unavailable.append(name)

    if domain == "options":
        take("options")
        if "options" in data:
            data["options"]["expiries"] = sorted(data["options"].get("expiries", []), key=lambda e: -(e.get("call_oi", 0) + e.get("put_oi", 0)))[:5]
    elif domain == "leverage":
        take("futures", drop=("oi_history",))
        take("hyperliquid")
        take("liquidations", drop=("hourly", "biggest", "venues"))
        data["squeeze_by_category"] = {k: category_brief(a)["squeeze"] for k, a in analyses.items()}
    elif domain == "liquidity":
        take("liquidations")
        take("whales", drop=("radar", "funding_by_exchange", "oi_by_exchange_usd"))
        data["structure_by_category"] = {k: {"levels": category_brief(a)["levels"], "trendlines": category_brief(a)["trendlines"],
                                             "anchor": a.direction.anchor, "invalidation": a.direction.invalidation} for k, a in analyses.items()}
    elif domain == "flow":
        take("whales", drop=("radar", "funding_by_exchange", "oi_by_exchange_usd"))
        if m.futures.available:
            data["taker_buy_sell_ratio"] = _round(m.futures.taker_buy_sell_ratio)
        else:
            unavailable.append("futures")
        data["flow_by_category"] = {k: {"cvd_slope_5bars": category_brief(a)["cvd_slope_5bars"], "direction": a.direction.direction,
                                        "drivers": a.direction.drivers} for k, a in analyses.items()}
    elif domain == "sentiment":
        take("futures", drop=("oi_history", "open_interest", "open_interest_usd", "oi_change_24h_pct", "mark_price"))
        take("sentiment")
        take("stablecoins")
        if m.whales.available:
            data["funding_by_exchange"] = _round(m.whales.funding_by_exchange)
            data["unusual_flow_radar"] = [r for r in m.whales.radar if r.get("unusual")][:10]
            data["btc_radar"] = m.whales.btc_radar
        else:
            unavailable.append("whales")
    else:
        raise KeyError(domain)
    return data, unavailable


def payload_for(agent_id: str, m: MarketSnapshot, analyses: dict[str, CategoryAnalysis]) -> dict:
    domain = DOMAIN_OF[agent_id]
    data, unavailable = domain_payload(domain, m, analyses)
    out = common_payload(m, analyses)
    out["domain"] = domain
    out["domain_data"] = data
    out["unavailable"] = unavailable
    return out


def desk_payload(desk: str, briefs: list, failed: list[str]) -> dict:
    return {"desk": desk, "briefs": [b.model_dump() for b in briefs], "failed_or_missing_agents": failed, "unavailable": []}


def volatility_payload(m: MarketSnapshot, analyses: dict[str, CategoryAnalysis]) -> dict:
    out = common_payload(m, analyses)
    out["unavailable"] = m.unavailable()
    return out


def accuracy_payload(desks: dict, volatility, analyses: dict[str, CategoryAnalysis], unavailable: list[str]) -> dict:
    return {
        "desk_reports": {k: v.model_dump() for k, v in desks.items()},
        "volatility": volatility.model_dump() if volatility else None,
        "deterministic_directions": {k: {"direction": a.direction.direction, "confidence": round(a.direction.confidence, 2),
                                         "ema_state": a.ema.state, "regime": a.vol.regime} for k, a in analyses.items()},
        "unavailable": unavailable,
    }


def director_payload(m: MarketSnapshot, analyses: dict[str, CategoryAnalysis], desks: dict, volatility, accuracy,
                     raw_metrics: list[tuple[str, str, str]], failed: list[str]) -> dict:
    out = common_payload(m, analyses)
    out.update({
        "desk_reports": {k: v.model_dump() for k, v in desks.items()},
        "volatility_agent": volatility.model_dump() if volatility else None,
        "accuracy_agent": accuracy.model_dump() if accuracy else None,
        "raw_metric_snapshot": [{"metric": r[0], "value": r[1], "source": r[2]} for r in raw_metrics],
        "failed_agents": failed,
        "unavailable": m.unavailable(),
    })
    return out
