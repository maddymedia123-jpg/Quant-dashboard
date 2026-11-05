"""Build the compact, domain-scoped data payload each agent receives.

Nothing here invents a number: unavailable snapshots are omitted and listed under `unavailable`."""
from __future__ import annotations

import math
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from core.config import MACRO_EVENTS
from core.data.types import MarketSnapshot
from core.indicators.category import CategoryAnalysis
from core.indicators.direction import cvd_slope

DOMAIN_OF = {"B1": "options", "R1": "options", "B2": "leverage", "R2": "leverage", "B3": "liquidity", "R3": "liquidity",
             "B4": "flow", "R4": "flow", "B5": "sentiment", "R5": "sentiment"}
_EXCLUDE = {"available", "fetched_at", "error", "source"}


def _round(v: Any, nd: int = 4) -> Any:
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        # a small number keeps its significant figures: four decimal places turned a funding rate of
        # 3.5e-05 into 0.0, and the specialists then reported "neutral funding (0.0%)"
        if v != 0 and abs(v) < 1:
            return float(f"{v:.{nd}g}")
        return round(v, nd)
    if isinstance(v, dict):
        return {k: _round(x, nd) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_round(x, nd) for x in v]
    return v


def _dump(snapshot, drop: tuple[str, ...] = ()) -> dict:
    d = snapshot.model_dump(exclude=_EXCLUDE | set(drop))
    return _round(d)


def _fib_brief(a: CategoryAnalysis) -> dict | None:
    f = a.fib
    if f is None:
        return None
    iso = lambda ms: datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")  # noqa: E731
    return {"anchor_1": {"kind": f.anchor_kind, "price": f.anchor_price, "time": iso(f.anchor_ms)},
            "anchor_2": {"kind": f.anchor2_kind, "price": f.anchor2_price, "time": iso(f.anchor2_ms)},
            "unit_bars": f.unit_bars,
            "upcoming": [{"zone": f"F{z.k}", "time": iso(z.timestamp_ms), "bars_away": z.bars_from_now} for z in f.upcoming]}


CVD_BARS = 5


def cvd_brief(a: CategoryAnalysis, m: MarketSnapshot | None = None) -> dict:
    """The last five bars of cumulative volume delta on the desk's chart timeframe.

    Real taker flow when the feed has it: market buys minus market sells on the Gate.io BTC perpetual, in
    BTC - the same contract the candles come from. Only without it does this fall back to an estimate
    from where each candle closed in its range, and the method says which one it is. The agents used to
    get the estimate under the name "CVD" and read it as spot CVD, while the real flow said the opposite."""
    flow = getattr(m, "taker_flow", None) if m is not None else None
    frame = flow.frames.get(a.chart_tf) if flow is not None and getattr(flow, "available", False) else None
    if frame is not None and len(frame) >= CVD_BARS:
        tail = frame.tail(CVD_BARS)
        buy, sell = float(tail["buy"].sum()), float(tail["sell"].sum())
        return {"delta_btc": round(buy - sell, 2), "taker_buy_btc": round(buy, 2), "taker_sell_btc": round(sell, 2),
                "bars": CVD_BARS, "timeframe": a.chart_tf,
                "method": "real taker flow: market buys minus market sells, Gate.io BTC_USDT perpetual "
                          "(the newest bar is still forming)"}
    return {"estimate": round(cvd_slope(a.chart_df, CVD_BARS), 2), "bars": CVD_BARS, "timeframe": a.chart_tf,
            "method": "ESTIMATE from where each candle closed in its range - no taker data for this "
                      "timeframe; not real CVD"}


def category_brief(a: CategoryAnalysis, m: MarketSnapshot | None = None) -> dict:
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
        "levels": [{"price": l.price, "kind": l.kind, "touches": l.touches, "reactions": "distinct wick tests over the chart window"} for l in a.levels][:6],
        "trendlines": [{"kind": t.kind, "slope_per_bar": t.slope, "value_now": t.value_now, "r2": t.r2} for t in a.trendlines],
        "fib_time": _fib_brief(a),
        "fib_retracement": ({"leg": a.fib_retr.leg, "low": a.fib_retr.low, "high": a.fib_retr.high,
                             "levels": {str(k): v for k, v in a.fib_retr.levels.items()}} if a.fib_retr else None),
        "reversal_window": (asdict(a.reversal) if a.reversal is not None else None),
        "patterns": [{"name": p.name, "bias": p.bias, "status": p.status, "confidence": p.confidence,
                      "breakout": p.breakout, "target": p.target, "invalidation": p.invalidation,
                      "apex_bars": p.apex_bars, "note": p.note} for p in a.patterns],
        "squeeze": {"score": a.squeeze.score, "direction": a.squeeze.direction, "drivers": a.squeeze.drivers},
        "cvd_5bars": cvd_brief(a, m),
        "price_change_24h_pct": a.price_change_24h_pct,
    })


def _calendar_events(m: MarketSnapshot) -> dict:
    if m.calendar.available and m.calendar.events:
        hi = [{"title": e["title"], "country": e["country"], "impact": e["impact"], "date": e["date"],
               "forecast": e["forecast"], "previous": e["previous"]}
              for e in m.calendar.events if e.get("impact") == "High" and e.get("country") in ("USD",)]
        return {"source": m.calendar.source, "usd_high_impact_this_week": hi[:16],
                "note": "the feed carries the current week only; the event window is five days before to five days after each print, as far as the feed reaches"}
    return {"source": "curated (calendar unavailable)", "events": MACRO_EVENTS}


def macro_read(m: MarketSnapshot, now_ms: int | None = None) -> dict:
    """The macro engine's read, computed in code: positioning into and reactions to each high-impact USD
    print, the headline narratives and how fresh they are, and what the Fed markets price. Each part fails
    on its own and says so."""
    from core.indicators import event_trap, narratives
    now = now_ms if now_ms is not None else int(m.generated_at.timestamp() * 1000)
    out: dict[str, Any] = {}
    try:
        out["prints"] = ([r.to_state() for r in event_trap.read_all(m.spot.frames.get("1h"), m.calendar.events, now)]
                         if m.calendar.available else {"available": False, "note": m.calendar.error or "calendar down"})
    except Exception as e:  # noqa: BLE001 - one part of the read must not take the others down
        out["prints"] = {"available": False, "note": f"{type(e).__name__}: {e}"}
    try:
        out["narratives"] = ([n.to_state() for n in narratives.read(m.news.headlines, now)]
                             if m.news.available else {"available": False, "note": m.news.error or "news down"})
    except Exception as e:  # noqa: BLE001
        out["narratives"] = {"available": False, "note": f"{type(e).__name__}: {e}"}
    try:
        from core.indicators.event_regime import read as regime_read
        out["event_regime"] = regime_read(m.calendar.events if m.calendar.available else [], now).to_state()
    except Exception as e:  # noqa: BLE001
        out["event_regime"] = {"active": False, "note": f"{type(e).__name__}: {e}"}
    if m.predictions.available:
        out["fed_and_macro_odds"] = [{"question": x["question"], "probability": round(x["probability"], 3),
                                      "band": x["band"]} for x in m.predictions.tradeable("MACRO")[:8]]
    out["rates_and_dollar"] = _rates(m)
    out["gold"] = _gold(m)
    out["btc_dominance"] = _dominance(m)
    out["yen_carry"] = _yen(m)
    out["dollar_index"] = _dollar(m)
    out["method"] = ("prints: BTC 1h moves against its own hourly volatility; narratives: headline attention by "
                     "topic (feed reaches back ~1.5 days); odds: Polymarket via VoxOdds, clean and watch bands only; "
                     "rates: US 2y and 10y yields, the funds rate and the broad dollar index with their one-week "
                     "change; delta_2y_read: 2y minus funds rate beyond +25 bp = liquidity contraction, below -25 bp "
                     "= expansion; btc_dominance: share change from own samples, +/-0.5 pp a day is rotation; "
                     "yen_carry: USD/JPY down 1.5% in a week is carry-unwind risk, the carry gap only when the BoJ "
                     "rate is hand-set; dollar_index: ICE's DXY formula rebuilt on the ECB's six daily rates (a day "
                     "behind, within a few tenths), a 1% week is a spike or slide, a spike with bitcoin up 3% over the "
                     "same dates is the false-breakout signal")
    return out


def _rates(m: MarketSnapshot) -> dict:
    """The US 2y and 10y and the dollar, as the US macro card shows them: value, one-week change, date,
    and whether the series is stale - so a judge never reads last month's yield as today's."""
    mac = getattr(m, "macro", None)
    if mac is None or not mac.available:
        return {"available": False, "note": getattr(mac, "error", None) or "macro feed down"}
    out: dict[str, Any] = {"available": True, "source": mac.source}
    for sid, s in mac.series.items():
        if not s.get("available"):
            out[sid] = {"available": False, "note": s.get("note")}
            continue
        out[sid] = {"label": s.get("label"), "value": s.get("value"), "unit": s.get("unit"),
                    "change_1w": s.get("change"), "age_days": s.get("age_days"), "stale": s.get("stale")}
    if mac.spread_2s10s is not None:
        out["spread_2s10s_pct"] = round(mac.spread_2s10s, 3)
    delta = getattr(mac, "delta_2y", None)
    if delta is not None:
        from core.data.macro import delta_2y_read
        read = delta_2y_read(delta)
        out["delta_2y_bp"] = round(delta * 100)
        out["delta_2y_read"] = {"state": read["state"], "note": read["note"]}
    return out


def _dominance(m: MarketSnapshot) -> dict:
    d = getattr(m, "dominance", None)
    if d is None or not d.available or d.btc_pct is None:
        return {"available": False, "note": getattr(d, "error", None) or "dominance feed down"}
    from core.data.dominance import rotation
    read = rotation(d)
    return {"available": True, "btc_pct": d.btc_pct, "eth_pct": d.eth_pct, "stable_pct": d.stable_pct,
            "change_24h_pp": d.change_24h_pp, "change_7d_pp": d.change_7d_pp,
            "market_cap_change_24h_pct": d.mcap_change_24h_pct, "rotation": read["state"], "note": read["note"]}


def _dollar(m: MarketSnapshot) -> dict:
    fx = getattr(m, "fx", None)
    if fx is None or not fx.available:
        return {"available": False, "note": getattr(fx, "error", None) or "USD/JPY feed down"}
    if fx.dxy is None:
        return {"available": False, "note": fx.dxy_note or "no day in the window has all six rates"}
    from core.data.fx import btc_change_between, dollar_read
    btc = btc_change_between(getattr(getattr(m, "spot", None), "frames", None) or {}, fx.dxy_week_ago_ms, fx.dxy_as_of_ms)
    read = dollar_read(fx, btc)
    return {"available": True, "dxy_reconstructed": fx.dxy, "change_1w_pct": fx.dxy_change_1w_pct,
            "change_1m_pct": fx.dxy_change_1m_pct, "btc_same_dates_pct": btc, "age_days": fx.dxy_age_days,
            "stale": fx.dxy_stale, "state": read["state"], "note": read["note"]}


def _yen(m: MarketSnapshot) -> dict:
    fx = getattr(m, "fx", None)
    if fx is None or not fx.available or fx.rate is None:
        return {"available": False, "note": getattr(fx, "error", None) or "USD/JPY feed down"}
    from core.data.fx import yen_read
    mac = getattr(m, "macro", None)
    effr = None
    if mac is not None and mac.available:
        s = mac.series.get("EFFR") or {}
        effr = s.get("value") if s.get("available") else None
    read = yen_read(fx, effr)
    return {"available": True, "usdjpy": fx.rate, "change_1w_pct": fx.change_1w_pct, "change_1m_pct": fx.change_1m_pct,
            "age_days": fx.age_days, "stale": fx.stale, "boj_rate_pct": fx.boj_rate_pct, "us_funds_rate_pct": effr,
            "carry_pct": read["carry_pct"], "state": read["state"], "note": read["note"]}


def _gold(m: MarketSnapshot) -> dict:
    g = getattr(m, "metals", None)
    if g is None or not g.available or not g.xau_usd:
        return {"available": False, "note": getattr(g, "error", None) or "gold feed down"}
    return {"available": True, "xau_usd": round(g.xau_usd, 2), "stale": bool(g.is_stale), "source": g.source}


def _cot(m: MarketSnapshot) -> dict:
    """CME Bitcoin futures positioning, weekly (Tuesday positions, Friday release), in contracts."""
    c = getattr(m, "cot", None)
    if c is None or not c.available:
        return {"available": False, "note": getattr(c, "error", None) or "COT feed down"}
    keys = ("market", "age_days", "stale", "contract_btc", "open_interest", "open_interest_change", "spec_long",
            "spec_short", "spec_net", "spec_net_change", "spec_net_pct_oi", "commercial_net", "small_trader_net",
            "cot_index", "weeks_in_index")
    out = {"available": True, **{k: getattr(c, k, None) for k in keys}}
    out["reading"] = ("cot_index is where the large speculators' net sits in its own 52-week range: 80+ crowded "
                      "long, 20 or below crowded short. Weekly data: context for the weekly and monthly desks, "
                      "not intraday timing.")
    return out


def _turning_window(m: MarketSnapshot) -> dict:
    """When BTC usually turns (measured), so no agent times a scenario off Fibonacci zones, which failed."""
    from core.indicators.turn_window import to_state
    return to_state(int(m.generated_at.timestamp() * 1000))


def common_payload(m: MarketSnapshot, analyses: dict[str, CategoryAnalysis]) -> dict:
    live = analyses.get("live") or next(iter(analyses.values()), None)
    return {
        "asset": "BTC perpetual (BTC_USDT)" if "perp" in (m.spot.symbol or "") else "BTC/USD spot",
        "spot": _round({"price": m.spot.last, "source": m.spot.source, "price_change_24h_pct": live.price_change_24h_pct if live else None}),
        "categories": {k: category_brief(a, m) for k, a in analyses.items()},
        "macro_calendar": _calendar_events(m),
        "macro_read": macro_read(m),
        "turning_window": _turning_window(m),
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
        data["flow_by_category"] = {k: {"cvd_5bars": cvd_brief(a, m), "direction": a.direction.direction,
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


def derivatives_payload(m: MarketSnapshot) -> tuple[dict, list[str]]:
    """Funding, open interest, positioning, liquidations, options and stablecoins for the desks.

    The On-Chain & Derivatives sub-agent is worth twenty points, so this has to reach the judges; an
    unavailable feed is named in the second return value rather than left as a silent gap."""
    data: dict[str, Any] = {}
    unavailable: list[str] = []
    for name, kw in (("futures", {"drop": ("oi_history",)}),
                     ("options", {"drop": ("expiries",)}),
                     ("liquidations", {"drop": ("hourly", "biggest", "venues")}),
                     ("stablecoins", {}),
                     ("hyperliquid", {})):
        snap = getattr(m, name)
        if snap.available:
            data[name] = _dump(snap, **kw)
        else:
            unavailable.append(name)
    # _dump rounds to four places, which turns a funding rate of 4.9e-05 into 0.0 and makes the
    # "overheated funding" item unjudgeable. Rates keep the precision the exchange reported: rounding
    # anything at 1e-5 to a fixed number of decimals throws away significant figures.
    # exchange flows go in as their read, not their 420 raw days
    try:
        from core.indicators.exchange_flows import read as read_flows
        d1 = m.spot.frames.get("1d")
        closes = (list(zip(d1["timestamp"].astype("int64"), d1["close"].astype(float))) if d1 is not None else None)
        flows = read_flows(m.exchange_flows, int(m.generated_at.timestamp() * 1000), closes)
    except Exception as e:  # noqa: BLE001 - one feed's read must not take the payload down
        flows = {"available": False, "note": f"{type(e).__name__}: {e}"}
    if flows.get("available"):
        data["exchange_flows"] = flows
    else:
        unavailable.append("exchange_flows")
    cot = _cot(m)
    if cot.get("available"):
        data["cot"] = cot
    else:
        unavailable.append("cot")
    if "futures" in data:
        for rate in ("funding_rate", "funding_7d_mean", "predicted_funding_rate"):
            value = getattr(m.futures, rate, None)
            if value is not None:
                data["futures"][rate] = float(value)
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


def desk_blocks(m: MarketSnapshot, analyses: dict[str, CategoryAnalysis]) -> dict:
    """What both Directors judge per desk: the derivatives over each desk's own window (roadmap item 9),
    the cascade state and the forecast per desk, and the 4-hour zone framed against the 24-hour horizon
    (roadmap item 12). Built once here so the 1-agent Director and the classic Director read the same
    blocks; a desk's failure is named in its block, never fatal."""
    from core.agents.recon_profiles import PROFILES
    from core.indicators import desk_derivs, horizons
    from core.indicators.cascade import classify
    from core.indicators.forecast import desk_forecast, to_state

    now_ms = int(m.generated_at.timestamp() * 1000)
    out: dict = {"desk_derivatives": {key: desk_derivs.read(key, m, now_ms) for key in PROFILES}, "cascade": {}, "forecast": {}}
    for key, profile in PROFILES.items():
        try:
            bias_tf, anchor_tf, engine_tfs = profile.cascade
            out["cascade"][key] = classify(m.spot.frames, bias_tf, anchor_tf, engine_tfs).to_state()
        except Exception as e:  # noqa: BLE001 - one desk's read must not take the payload down
            out["cascade"][key] = {"available": False, "note": f"{type(e).__name__}: {e}"}
        a = analyses.get(key)
        if a is not None:
            try:
                f, tr = desk_forecast(a, desk_derivs.gamma_for(key, m), m.spot.frames.get(a.chart_tf))
                out["forecast"][key] = to_state(f, tr)
            except Exception as e:  # noqa: BLE001
                out["forecast"][key] = {"available": False, "note": f"{type(e).__name__}: {e}"}
    try:
        out["horizons"] = horizons.read(m, analyses, now_ms, forecasts=out["forecast"], cascade=out["cascade"],
                                        derivs=out["desk_derivatives"])
    except Exception as e:  # noqa: BLE001 - the framing is an addition; its failure is named, not fatal
        out["horizons"] = {"available": False, "note": f"{type(e).__name__}: {e}"}
    return out


def director_payload(m: MarketSnapshot, analyses: dict[str, CategoryAnalysis], desks: dict, volatility, accuracy,
                     raw_metrics: list[tuple[str, str, str]], failed: list[str]) -> dict:
    out = common_payload(m, analyses)
    out.update(desk_blocks(m, analyses))
    out.update({
        "desk_reports": {k: v.model_dump() for k, v in desks.items()},
        "volatility_agent": volatility.model_dump() if volatility else None,
        "accuracy_agent": accuracy.model_dump() if accuracy else None,
        "raw_metric_snapshot": [{"metric": r[0], "value": r[1], "source": r[2]} for r in raw_metrics],
        "failed_agents": failed,
        "unavailable": m.unavailable(),
    })
    return out
