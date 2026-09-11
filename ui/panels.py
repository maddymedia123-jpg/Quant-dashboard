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


def _liq_kpi(l) -> dict:
    if not l.available or not l.total_usd:
        return {"label": "Liq 24h", "value": "—", "delta": None, "tone": None}
    share = (l.long_usd or 0.0) / l.total_usd * 100.0
    tone = "down" if share >= 60 else "up" if share <= 40 else None
    return {"label": "Liq 24h", "value": fmt_num(l.total_usd / 1e6, 0, "$", "M"), "delta": f"{share:.0f}% longs", "tone": tone}


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
        _liq_kpi(m.liquidations),
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
    if a.patterns:
        p0 = a.patterns[0]
        extra = f", breaking {fmt_num(p0.breakout, 0, '$')} targets {fmt_num(p0.target, 0, '$')}" if (p0.breakout and p0.target) else ""
        parts.append(f"Pattern: {p0.name.lower()} {p0.status} ({p0.bias}){extra}.")
    rw = a.reversal
    if rw is not None and rw.status == "ACTIVE":
        parts.append(f"A Fibonacci reversal window is open now with a {rw.bias} lean: {'; '.join(rw.confluence[:3])}.")
    elif rw is not None and rw.status == "UPCOMING" and rw.bars_away is not None:
        parts.append(f"The next Fibonacci time zone arrives {_bars(rw.bars_away)} ({_utc(rw.zone_ms)}).")
    if s.score is not None and s.score >= 40:
        parts.append(f"Positioning shows {s.direction.replace('_', ' ').lower()} risk at {s.score}/100.")
    if d.anchor is not None:
        parts.append(f"The call stays anchored at {fmt_num(d.anchor, 0, '$')} and flips if: {d.flips_if.lower()}.")
    else:
        parts.append(f"It flips if: {d.flips_if.lower()}.")
    return card_html("Summary", "<p>" + html.escape(" ".join(parts)) + "</p>", tone_for_direction(d.direction))


def _ago(delta_ms: int) -> str:
    mins = max(0, int(delta_ms // 60_000))
    if mins < 60:
        return f"{mins}m"
    hours, mins = divmod(mins, 60)
    if hours < 48:
        return f"{hours}h {mins:02d}m"
    return f"{hours // 24}d {hours % 24}h"


TRIGGERS_WATCHED = ("funding flip · OI ±5% · long/short crosses 1.0 · squeeze ≥ 60 · close past invalidation · "
                    "Director trap change · macro event within 24h · stale anchor")


def anchored_direction_html(a: CategoryAnalysis, v, now_ms: int) -> str:
    """Anchored verdict card: the displayed direction holds until a trigger fires."""
    an = v.anchored
    body = (f"<p><strong>{an.direction}</strong> · confidence {an.confidence:.0%} · anchored {_ago(now_ms - v.since_ms)} ago "
            f"at {fmt_num(an.price, 0, '$')} · {a.chart_tf} chart</p>")
    if v.changed and v.triggers_fired and v.triggers_fired != ["initial"]:
        body += f"<p class='muted'>Re-anchored now on: {html.escape(', '.join(v.triggers_fired))}</p>"
    body += f"<p>Anchor {fmt_num(an.anchor, 0, '$')} · Invalidation {fmt_num(an.invalidation, 0, '$')}</p>"
    body += f"<p class='muted'>Flips if: {html.escape(an.flips_if)}</p>"
    if v.unconfirmed:
        body += (f"<p><span class='ti-chip warn'>live read {html.escape(v.live.direction)} ({v.live.confidence:.0%})</span> "
                 f"unconfirmed until a trigger fires</p>")
    body += _li(an.drivers)
    body += f"<p class='muted'>Triggers watched: {TRIGGERS_WATCHED}</p>"
    return card_html("Market direction (anchored)", body, tone_for_direction(an.direction))


def early_warning_html(ws) -> str | None:
    if not ws:
        return None
    body = ""
    for w in ws:
        body += f"<p><strong>{html.escape(w.kind.replace('_', ' '))}</strong> · {html.escape(w.level)}</p>" + _li(w.drivers)
    return card_html("Early warning", body, "warn")


def _rate_rows(d: dict) -> str:
    rows = ""
    for k, h in sorted(d.items()):
        rate = f"{h.rate:.0%}" if h.rate is not None else "—"
        flag = " (small sample)" if h.small else ""
        rows += f"<tr><td>{html.escape(str(k))}</td><td>{rate}</td><td>{h.hits}/{h.n}{html.escape(flag)}</td></tr>"
    return rows


def accuracy_html(s: dict) -> str:
    if not s.get("total_scored"):
        body = ("<p class='muted'>No scored calls yet. Direction calls score once their horizon elapses "
                "(Live 1h, Intraday 24h, Weekly 7d, Monthly 30d); Director reports score at 24h and 7d. "
                f"Open calls waiting: {s.get('open_calls', 0)}.</p>")
        return card_html("Accuracy ledger", body)
    body = "<p>Direction calls by category</p><table><tr><th>Category</th><th>Hit rate</th><th>Hits / n</th></tr>" + _rate_rows(s["by_category"]) + "</table>"
    if s["by_classification_24h"]:
        body += "<p>Director classification, 24h</p><table><tr><th>Class</th><th>Hit rate</th><th>Hits / n</th></tr>" + _rate_rows(s["by_classification_24h"]) + "</table>"
    if s["by_classification_7d"]:
        body += "<p>Director classification, 7d</p><table><tr><th>Class</th><th>Hit rate</th><th>Hits / n</th></tr>" + _rate_rows(s["by_classification_7d"]) + "</table>"
    recent = [f"{c['category']} {c['direction']} @ {c['price']:,.0f} → {c['realised_pct']:+.2f}% ({'hit' if c['hit'] else 'miss'})" for c in s["recent_calls"][:10]]
    if recent:
        body += "<p>Recent scored calls</p>" + _li(recent)
    body += f"<p class='muted'>Open calls waiting: {s.get('open_calls', 0)}. Samples under 10 are indicative only.</p>"
    return card_html("Accuracy ledger", body)


def _utc(ms: int | None) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%a %d %b %H:%M UTC") if ms else "—"


def _bars(n: int | None) -> str:
    if n is None:
        return ""
    if n == 0:
        return "this bar"
    return f"in {n} bar{'s' if n != 1 else ''}" if n > 0 else f"{-n} bar{'s' if n != -1 else ''} ago"


def fib_html(a: CategoryAnalysis) -> str:
    """Fibonacci time zones (two-point), the reversal window, and retracement levels."""
    f, rw, fr = a.fib, a.reversal, a.fib_retr
    if f is None:
        return card_html("Fibonacci time & reversal window", "<p class='muted'>Not enough history for Fibonacci time zones.</p>")
    body = (f"<p>Anchors: {f.anchor_kind} {fmt_num(f.anchor_price, 0, '$')} ({_utc(f.anchor_ms)}) → "
            f"{f.anchor2_kind} {fmt_num(f.anchor2_price, 0, '$')} ({_utc(f.anchor2_ms)}) · unit {f.unit_bars} bars</p>")
    if f.upcoming:
        body += "<p>Next zones</p>" + _li([f"F{z.k} · {_utc(z.timestamp_ms)} ({_bars(z.bars_from_now)})" for z in f.upcoming])
    tone = "neutral"
    if rw is not None:
        if rw.status == "ACTIVE":
            tone = "warn"
            body += (f"<p><strong>Reversal window open</strong> · F{rw.zone_k} {_bars(rw.bars_away)} (±{rw.window_bars} bar) · "
                     f"bias <strong>{html.escape(rw.bias)}</strong></p>") + _li(rw.confluence)
        elif rw.status == "TIME_ONLY":
            body += (f"<p>Fib time zone F{rw.zone_k} {_bars(rw.bars_away)}, but price has no confluence "
                     "(no level, divergence, retracement or pattern), so no reversal signal.</p>")
        elif rw.status == "UPCOMING":
            body += f"<p>Next reversal window: F{rw.zone_k} {_bars(rw.bars_away)}.</p>"
            if rw.confluence:
                body += "<p class='muted'>Confluence building now</p>" + _li(rw.confluence)
    if fr is not None:
        lv = " · ".join(f"{r} {fmt_num(fr.levels[r], 0, '$')}" for r in (0.382, 0.5, 0.618))
        body += (f"<p class='muted'>Retracement of the {fr.leg} leg {fmt_num(fr.low, 0, '$')} – {fmt_num(fr.high, 0, '$')}: {lv}</p>")
    return card_html("Fibonacci time & reversal window", body, tone)


def patterns_html(a: CategoryAnalysis) -> str:
    if not a.patterns:
        return card_html("Chart patterns", "<p class='muted'>No clean pattern in the last 120 bars.</p>")
    body = ""
    tone = "neutral"
    for p in a.patterns:
        head = (f"<p><strong>{html.escape(p.name)}</strong> · {p.status} · {p.bias} · confidence {p.confidence:.0%}"
                + (f" · apex {_bars(p.apex_bars)}" if p.apex_bars else "") + "</p>")
        keys = []
        if p.breakout is not None:
            keys.append(f"breakout {fmt_num(p.breakout, 0, '$')}")
        if p.target is not None:
            keys.append(f"target {fmt_num(p.target, 0, '$')}")
        if p.invalidation is not None:
            keys.append(f"invalidation {fmt_num(p.invalidation, 0, '$')}")
        body += head + (f"<p>{' · '.join(keys)}</p>" if keys else "") + (f"<p class='muted'>{html.escape(p.note)}</p>" if p.note else "")
        if tone == "neutral":
            tone = tone_for_direction({"bullish": "BULLISH", "bearish": "BEARISH"}.get(p.bias, ""))
    return card_html("Chart patterns", body, tone)


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


def agent_summary_html(a: CategoryAnalysis, report) -> str:
    """Director's per-category paragraph when a report exists; otherwise the Phase-1 placeholder."""
    if report is not None and report.director is not None:
        text = report.director.category_summaries.get(a.key)
        if text:
            tone = {"BULL_TRAP": "down", "BEAR_TRAP": "up", "RANGE_TRAP": "warn"}.get(report.director.trap_classification, "neutral")
            body = (f"<p>{html.escape(text)}</p><p class='muted'>Director verdict: {report.director.trap_classification.replace('_', ' ')} · "
                    f"report {html.escape(report.report_id)} · {report.generated_at:%H:%M} UTC</p>")
            return card_html("Head-agent summary", body, tone)
    return agent_placeholder_html(a)


def report_stats_html(report) -> str:
    ok = sum(1 for r in report.runs if r.ok)
    models = sorted({r.model for r in report.runs})
    cost = f"${report.total_cost_usd:.4f}" if report.total_cost_usd is not None else "not reported (free tier)"
    bits = [f"provider {report.provider}", f"models {', '.join(models)}", f"agents {ok}/{len(report.runs)} ok",
            f"tokens {report.total_tokens:,}", f"cost {cost}", f"wall {report.wall_time_s:.0f}s",
            f"generated {report.generated_at:%Y-%m-%d %H:%M} UTC"]
    if report.failed_agents:
        bits.append("failed: " + ", ".join(report.failed_agents))
    return "".join(chip(b, "down" if b.startswith("failed") else "") for b in bits)


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
    return card_html("Macro calendar (curated)", _li(rows) + "<p class='muted'>Fallback list; the live calendar feed was unavailable.</p>")


def calendar_html(m: MarketSnapshot, now_ms: int, limit: int = 8) -> str:
    """High-impact USD prints this week from the live calendar; falls back to the curated list."""
    from datetime import datetime, timezone
    from core.config import MACRO_EVENTS
    c = m.calendar
    if not (c.available and c.events):
        return macro_html(MACRO_EVENTS)
    hi = [e for e in c.events if e.get("impact") == "High" and e.get("country") == "USD" and e.get("time_ms")]
    upcoming = [e for e in hi if e["time_ms"] >= now_ms - 3_600_000][:limit]
    soon = c.upcoming(now_ms, 86_400_000)
    rows = []
    for e in upcoming:
        t = datetime.fromtimestamp(e["time_ms"] / 1000, tz=timezone.utc).strftime("%a %d %b %H:%M UTC")
        extra = " · ".join(x for x in (f"forecast {e['forecast']}" if e.get("forecast") else "", f"previous {e['previous']}" if e.get("previous") else "") if x)
        rows.append(f"{t} · {e['title']}" + (f" ({extra})" if extra else ""))
    body = _li(rows) if rows else "<p class='muted'>No further USD high-impact releases this week.</p>"
    if soon:
        body = f"<p><strong>Inside 24h:</strong> {html.escape(', '.join(e['title'] for e in soon))}. No new leveraged entries into the print.</p>" + body
    body += f"<p class='muted'>Source: Forex Factory weekly calendar via {html.escape(c.source)}.</p>"
    return card_html("Macro calendar", body, "warn" if soon else "neutral")


def data_status_html(m: MarketSnapshot) -> str:
    out = []
    for name in MarketSnapshot.SOURCES:
        s = getattr(m, name)
        label = f"{name}: {s.source}" + ("" if s.available else " unavailable")
        out.append(chip(label, "up" if s.available else "down"))
    return "".join(out)


def raw_metrics_rows(m: MarketSnapshot) -> list[tuple[str, str, str]]:
    f, o, s = m.futures, m.options, m.sentiment
    h, l, w, sc = m.hyperliquid, m.liquidations, m.whales, m.stablecoins
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
        ("Hyperliquid funding (1h)", fmt_pct(h.funding_rate_1h * 100, 4) if h.funding_rate_1h is not None else "—", h.source),
        ("Hyperliquid OI", fmt_num(h.open_interest, 0, suffix=" BTC"), h.source),
        ("Liquidations 24h", (f"{fmt_num(l.total_usd / 1e6, 1, '$', 'M')} (long {fmt_num(l.long_usd / 1e6, 1, '$', 'M')} / short {fmt_num(l.short_usd / 1e6, 1, '$', 'M')})" if l.total_usd else "—"), l.source),
        ("BTC liquidations 24h", (f"{fmt_num(l.btc_usd / 1e6, 1, '$', 'M')} (long {fmt_num(l.btc_long_usd / 1e6, 1, '$', 'M')})" if l.btc_usd else "—"), l.source),
        ("Whale BTC buy / sell", (f"{fmt_num(w.btc_buy_usd / 1e6, 1, '$', 'M')} / {fmt_num(w.btc_sell_usd / 1e6, 1, '$', 'M')} over {fmt_num(w.window_minutes, 0)} min" if w.available and w.btc_trades else "—"), w.source),
        ("Stablecoin supply", (f"{fmt_num(sc.total_usd / 1e9, 1, '$', 'B')} ({fmt_pct(sc.change_24h_pct, 2)} 24h)" if sc.total_usd else "—"), sc.source),
    ]
