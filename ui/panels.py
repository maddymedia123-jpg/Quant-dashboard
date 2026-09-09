"""Panel builders: pure HTML functions plus thin Streamlit renderers."""
from __future__ import annotations

import html

import streamlit as st

from core.data.types import MarketSnapshot
from core.indicators.category import CategoryAnalysis
from core.indicators.trade_map import TradeMap
from ui.theme import card_html, chip, fmt_num, fmt_pct, tone_for_direction


def render(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def _li(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{html.escape(i)}</li>" for i in items) + "</ul>" if items else ""


def kpis_for(a: CategoryAnalysis, m: MarketSnapshot) -> list[dict]:
    f, s, o = m.futures, m.sentiment, m.options
    rsi_last = float(a.rsi.dropna().iloc[-1]) if a.rsi.notna().any() else None
    pc = a.price_change_24h_pct
    return [
        {"label": "Price", "value": fmt_num(a.price, 0, "$"), "delta": (fmt_pct(pc) + " 24h") if pc is not None else None,
         "tone": "up" if (pc or 0) >= 0 else "down"},
        {"label": "Direction", "value": a.direction.direction, "delta": f"confidence {a.direction.confidence:.0%}", "tone": tone_for_direction(a.direction.direction)},
        {"label": "RSI 14", "value": fmt_num(rsi_last, 0), "delta": a.ema.state + " EMA stack", "tone": {"BULL": "up", "BEAR": "down"}.get(a.ema.state)},
        {"label": "ATR %", "value": fmt_num(a.vol.atr_pct, 2, suffix="%"), "delta": a.vol.regime + " regime", "tone": "warn" if a.vol.regime == "HIGH" else None},
        {"label": "Funding", "value": fmt_pct(f.funding_rate * 100, 4) if f.funding_rate is not None else "—",
         "delta": (f"7d {fmt_pct(f.funding_7d_mean * 100, 4)}" if f.funding_7d_mean is not None else None), "tone": None},
        {"label": "OI 24h", "value": fmt_pct(f.oi_change_24h_pct, 1), "delta": (fmt_num(f.open_interest_usd / 1e9, 2, "$", "B") if f.open_interest_usd else None),
         "tone": "up" if (f.oi_change_24h_pct or 0) >= 0 else "down"},
        {"label": "Long/Short", "value": fmt_num(f.long_short_ratio, 2), "delta": (f"taker {fmt_num(f.taker_buy_sell_ratio, 2)}" if f.taker_buy_sell_ratio else None), "tone": None},
        {"label": "Max pain", "value": fmt_num(o.max_pain, 0, "$"), "delta": o.max_pain_expiry, "tone": None},
        {"label": "Fear & Greed", "value": fmt_num(s.fear_greed, 0), "delta": s.classification, "tone": None},
    ]


def direction_html(a: CategoryAnalysis) -> str:
    d = a.direction
    body = f"<p><strong>{d.direction}</strong> · confidence {d.confidence:.0%} · {a.chart_tf} chart, {a.context_tf} context"
    if a.ctx_ema:
        body += f" ({a.ctx_ema.state} on {a.context_tf})"
    body += "</p>"
    body += f"<p>Anchor {fmt_num(d.anchor, 0, '$')} · Invalidation {fmt_num(d.invalidation, 0, '$')}</p>"
    body += f"<p class='muted'>Flips if: {html.escape(d.flips_if)}</p>"
    body += _li(d.drivers)
    return card_html("Market direction", body, tone_for_direction(d.direction))


def _touches(n: int) -> str:
    return "1 touch" if n == 1 else f"{n} touches"


def layman_html(a: CategoryAnalysis) -> str:
    """Plain-English summary built only from the deterministic analysis (Phase 2 replaces it with the head agent)."""
    d, v, s = a.direction, a.vol, a.squeeze
    sup = [l for l in a.levels if l.kind == "support"]
    res = [l for l in a.levels if l.kind == "resistance"]
    nearest_sup = max(sup, key=lambda l: l.price) if sup else None
    nearest_res = min(res, key=lambda l: l.price) if res else None
    rsi_last = float(a.rsi.dropna().iloc[-1]) if a.rsi.notna().any() else None
    lean = {"BULLISH": "leaning up", "BEARISH": "leaning down", "NEUTRAL": "without a clear lean"}[d.direction]
    stack = {"BULL": "above all four EMAs, a bullish stack", "BEAR": "below all four EMAs, a bearish stack", "MIXED": "inside a mixed EMA stack"}[a.ema.state]
    parts = [f"On the {a.chart_tf} chart BTC trades at {fmt_num(a.price, 0, '$')}, {lean} with {d.confidence:.0%} confidence, {stack}."]
    if nearest_sup or nearest_res:
        bits = []
        if nearest_sup:
            bits.append(f"support at {fmt_num(nearest_sup.price, 0, '$')} ({_touches(nearest_sup.touches)})")
        if nearest_res:
            bits.append(f"resistance at {fmt_num(nearest_res.price, 0, '$')} ({_touches(nearest_res.touches)})")
        parts.append("Nearest " + " and ".join(bits) + ".")
    if v.exp_low and v.exp_high:
        parts.append(f"Volatility is {v.regime.lower()}; a one-sigma move over the {v.horizon_label} spans {fmt_num(v.exp_low, 0, '$')} to {fmt_num(v.exp_high, 0, '$')}, "
                     f"and the 20-bar 2σ band sits at {fmt_num(v.sigma2_dn, 0, '$')} to {fmt_num(v.sigma2_up, 0, '$')}.")
    if rsi_last is not None:
        zone = "overbought" if rsi_last >= 70 else "oversold" if rsi_last <= 30 else "neutral territory"
        parts.append(f"RSI is {rsi_last:.0f}, {zone}.")
    if a.divergences:
        parts.append("Divergence watch: " + "; ".join(x.label for x in a.divergences) + ".")
    if s.score is not None and s.score >= 40:
        parts.append(f"Positioning shows {s.direction.replace('_', ' ').lower()} risk at {s.score}/100.")
    if d.anchor is not None:
        parts.append(f"The call stays anchored at {fmt_num(d.anchor, 0, '$')} and flips if: {d.flips_if.lower()}.")
    else:
        parts.append(f"It flips if: {d.flips_if.lower()}.")
    return card_html("Summary", "<p>" + html.escape(" ".join(parts)) + "</p>", tone_for_direction(d.direction))


def divergence_html(a: CategoryAnalysis) -> str:
    if not a.divergences:
        return card_html("RSI divergence", "<p class='muted'>No RSI divergence in the last 60 bars.</p>")
    rows = []
    tone = "neutral"
    for d in a.divergences:
        p1, p2 = d.price_points
        rows.append(f"{d.label}: price {p1[1]:,.0f} → {p2[1]:,.0f}, RSI {d.rsi_points[0][1]:.0f} → {d.rsi_points[1][1]:.0f}")
        if d.status == "forming":
            tone = "warn"
        elif tone == "neutral":
            tone = "up" if d.direction == "bullish" else "down"
    return card_html("RSI divergence", _li(rows), tone)


def volatility_html(a: CategoryAnalysis) -> str:
    v, s = a.vol, a.squeeze
    body = f"<p>Regime <strong>{v.regime}</strong> · ATR {fmt_num(v.atr, 0, '$')} ({fmt_num(v.atr_pct, 2, suffix='%')})</p>"
    body += f"<p>Expected range {v.horizon_label}: {fmt_num(v.exp_low, 0, '$')} – {fmt_num(v.exp_high, 0, '$')} (±{fmt_num(v.sigma_pct, 2, suffix='%')})</p>"
    body += f"<p>2σ band (20-bar): {fmt_num(v.sigma2_dn, 0, '$')} – {fmt_num(v.sigma2_up, 0, '$')} · mean {fmt_num(v.mean20, 0, '$')}</p>"
    if s.score is None:
        body += "<p class='muted'>Squeeze risk: futures data unavailable.</p>"
    else:
        body += f"<p>Squeeze risk <strong>{s.score}/100</strong> · {s.direction.replace('_', ' ')}</p>" + _li(s.drivers)
    tone = "warn" if (s.score or 0) >= 60 or v.regime == "HIGH" else "neutral"
    return card_html("Volatility & positioning", body, tone)


def squeeze_banner_html(a: CategoryAnalysis) -> str | None:
    s = a.squeeze
    if s.score is None or s.score < 60:
        return None
    return card_html(f"Early warning: {s.direction.replace('_', ' ')} risk {s.score}/100", _li(s.drivers), "warn")


def agent_placeholder_html(a: CategoryAnalysis) -> str:
    return card_html("Head-agent summary", "<p class='muted'>Run Analysis (Phase 2) generates the Director's per-category summary here. "
                                            "Until then the deterministic direction, divergence and volatility panels above are the source of truth.</p>")


def trade_result_html(tm: TradeMap) -> str:
    tone = {"OPTIMAL": "up", "ELEVATED": "warn", "EXTREME": "down"}[tm.risk_label]
    body = (f"<p><strong>{tm.classification}</strong> {tm.direction} · entry {fmt_num(tm.entry, 0, '$')} · {tm.leverage:.0f}x · "
            f"stop {fmt_num(tm.stop, 0, '$')} · target {fmt_num(tm.target, 0, '$')} · R:R {fmt_num(tm.rr, 2)}</p>")
    body += f"<p>Liquidation {fmt_num(tm.liquidation_price, 0, '$')} ({tm.dist_to_liq_pct:.1f}% away) · risk <strong>{tm.risk_label}</strong> · "
    body += ("stop protected by structure" if tm.stop_structural else "stop not protected by structure") + "</p>"
    if tm.structures_above:
        body += "<p>Structure above</p>" + _li(tm.structures_above)
    if tm.structures_below:
        body += "<p>Structure below</p>" + _li(tm.structures_below)
    if tm.warnings:
        body += "<p>Warnings</p>" + _li(tm.warnings)
    body += "<p>Verdict stays anchored until</p>" + _li(tm.anchor_triggers)
    return card_html("Active trade stress-test", body, tone)


def macro_html(events: list[dict]) -> str:
    rows = [f"{e['date_utc']} · {e['title']} · {e['impact']} impact. {e['note']}" for e in events]
    return card_html("Macro calendar (curated)", _li(rows) + "<p class='muted'>Manually maintained until a news source is connected.</p>")


def data_status_html(m: MarketSnapshot) -> str:
    out = []
    for name in ("spot", "futures", "options", "sentiment"):
        s = getattr(m, name)
        label = f"{name}: {s.source}" + ("" if s.available else " unavailable")
        out.append(chip(label, "up" if s.available else "down"))
    return "".join(out)


def raw_metrics_rows(m: MarketSnapshot) -> list[tuple[str, str, str]]:
    f, o, s = m.futures, m.options, m.sentiment
    return [
        ("Spot", fmt_num(m.spot.last, 0, "$"), m.spot.source),
        ("Funding (8h)", fmt_pct(f.funding_rate * 100, 4) if f.funding_rate is not None else "—", f.source),
        ("Funding 7d mean", fmt_pct(f.funding_7d_mean * 100, 4) if f.funding_7d_mean is not None else "—", f.source),
        ("Open interest", fmt_num(f.open_interest, 0, suffix=" BTC"), f.source),
        ("OI 24h change", fmt_pct(f.oi_change_24h_pct, 1), f.source),
        ("Long/short ratio", fmt_num(f.long_short_ratio, 2), f.source),
        ("Taker buy/sell", fmt_num(f.taker_buy_sell_ratio, 2), f.source),
        ("Max pain", fmt_num(o.max_pain, 0, "$") + (f" ({o.max_pain_expiry})" if o.max_pain_expiry else ""), o.source),
        ("Put/call OI", fmt_num(o.put_call_oi_ratio, 2), o.source),
        ("ATM IV", fmt_num(o.iv_atm, 1, suffix="%"), o.source),
        ("IV skew (put − call)", fmt_num(o.iv_skew, 1, suffix=" pts"), o.source),
        ("Fear & Greed", (fmt_num(s.fear_greed, 0) + (f" {s.classification}" if s.classification else "")), s.source),
    ]
