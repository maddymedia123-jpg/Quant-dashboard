"""Panel builders: pure HTML functions plus thin Streamlit renderers."""
from __future__ import annotations

import html
import json

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
    """Plain-English summary built only from the deterministic analysis."""
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
    return card_html("Head-agent summary", "<p class='muted'>Run Analysis generates the Director's per-category summary here. "
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


# ---------- Live Recon war room ----------
TRAP_LABELS = {"BULL_TRAP": "bull trap", "BEAR_TRAP": "bear trap",
               "GENUINE_MOVE": "genuine move", "NO_TRAP": "no trap"}


def _team_rows(team) -> str:
    rows = ""
    for d in team.domains:
        rows += (f"<tr><td>{d.agent_no}. {html.escape(d.title)}</td>"
                 f"<td>{d.points:.1f} / {d.max_points}</td><td>{d.pct:.0f}%</td></tr>")
    return rows


def team_html(team) -> str:
    """One team's scorecard: five domain agents, twenty points each."""
    if not team.ok:
        body = f"<p class='muted'>Did not score: {html.escape(team.error or 'unknown error')}. "
        body += "This is a failure, not a reading of zero.</p>"
        return card_html(f"{team.side.title()} team", body, "warn")
    tone = "up" if team.side == "bullish" else "down"
    body = (f"<p><strong>{team.points:.0f} / 100</strong> · {html.escape(team.verdict)}"
            f"<span class='muted'> · {html.escape(team.provider or 'n/a')}"
            + (f" · {team.latency_s:.1f}s" if team.latency_s else "") + "</span></p>")
    body += ("<table><thead><tr><th>Domain agent</th><th>Points</th><th>Share</th></tr></thead>"
             f"<tbody>{_team_rows(team)}</tbody></table>")
    if team.missing:
        body += f"<p class='muted'>{len(team.missing)} checklist items unanswered and scored 0.</p>"
    return card_html(f"{team.side.title()} team", body, tone)


def war_room_html(res) -> str:
    """The trap agent's read, and whether the head had to consult it."""
    t = res.trap
    if not t.ok:
        return card_html("War room · trap audit",
                         f"<p class='muted'>No answer: {html.escape(t.error or 'unknown error')}. "
                         "No trap override was applied.</p>", "warn")
    label = TRAP_LABELS.get(t.choice or "", t.choice or "—")
    rows = "".join(f"<tr><td>{html.escape(TRAP_LABELS.get(k, k))}</td><td>{v:.0%}</td></tr>"
                   for k, v in sorted(t.probabilities.items(), key=lambda kv: -kv[1]))
    tone = "warn" if t.choice in ("BULL_TRAP", "BEAR_TRAP") else "neutral"
    body = f"<p><strong>{html.escape(label.title())}</strong></p>"
    body += f"<table><tbody>{rows}</tbody></table>"
    body += (f"<p class='muted'>Head consulted the war room: {'yes' if res.consulted else 'not required'} "
             "(consulted when both teams score above 60 or neither leads by 10).</p>")
    if res.override:
        body += f"<p><span class='ti-chip warn'>override</span> {html.escape(res.override)}</p>"
    return card_html("War room · trap audit", body, tone)


def _remaining(mins: int) -> str:
    if mins >= 24 * 60:
        return f"{mins // 1440}d {(mins % 1440) // 60:02d}h"
    return f"{mins // 60}h {mins % 60:02d}m"


def anchored_summary_html(anchor: dict, window_close_ms: int, now_ms: int, anchored_now: bool,
                          profile=None) -> str:
    """The pinned summary. It does not change until the category's candle closes."""
    from core.agents.recon_profiles import LIVE
    p = profile or LIVE
    mins = max(0, (window_close_ms - now_ms) // 60000)
    bias = str(anchor.get("bias", "—"))
    body = (f"<p><strong>{html.escape(bias)}</strong> · confidence {float(anchor.get('confidence') or 0):.0%} "
            f"· margin {float(anchor.get('margin') or 0):+.0f} points</p>")
    body += (f"<p>Bullish {float(anchor.get('bull_points') or 0):.0f} / 100 · "
             f"Bearish {float(anchor.get('bear_points') or 0):.0f} / 100 · "
             f"trap read {html.escape(TRAP_LABELS.get(anchor.get('trap') or '', anchor.get('trap') or '—'))}</p>")
    body += (f"<p class='muted'>Anchored {html.escape(str(anchor.get('published_at_utc', '')))} "
             + (f"at {fmt_num(anchor.get('price'), 0, '$')} " if anchor.get("price") else "")
             + f"· pinned for {_remaining(mins)} more, until the {p.window_label} candle closes.</p>")
    body += ("<p class='muted'>" + ("Published by this run." if anchored_now else
             "Re-runs during this candle append side notes; they never rewrite this summary.") + "</p>")
    if anchor.get("degraded"):
        body += "<p><span class='ti-chip warn'>degraded</span> part of the checklist did not score.</p>"
    return card_html(f"{p.window_title} anchored summary", body, tone_for_direction(bias.split()[0]))


def side_notes_html(rows: list[dict]) -> str:
    """Interim invalidations, appended alongside the anchor and never replacing it."""
    if not rows:
        return card_html("Interim side notes", "<p class='muted'>Nothing substantial has changed since the "
                                               "summary was anchored.</p>", "neutral")
    body = ""
    for r in rows:
        body += (f"<p><span class='ti-chip warn'>{html.escape(str(r['kind']).replace('_', ' ').lower())}</span> "
                 f"<strong>{html.escape(str(r['headline']))}</strong>")
        if r.get("price"):
            body += f" <span class='muted'>at {fmt_num(r['price'], 0, '$')}</span>"
        body += "</p>"
        if r.get("detail"):
            body += f"<p class='muted'>{html.escape(str(r['detail']))}</p>"
    return card_html(f"Interim side notes ({len(rows)})", body, "warn")


# ---------- SMC and liquidity protocols ----------
def _dash(v, digits=0, prefix="") -> str:
    return "—" if v is None else fmt_num(v, digits, prefix)


def _pct_or_dash(v, digits=1) -> str:
    """A dash carrying a percent sign reads as a measured zero, so the unit goes only with a number."""
    return "—" if v is None else f"{fmt_num(v, digits)}%"


def smc_html(by_tf: dict) -> str:
    """Structure, order blocks, gaps and premium/discount per timeframe."""
    rows = ""
    for tf, d in by_tf.items():
        st = (d or {}).get("structure", {})
        if not st.get("available"):
            rows += f"<tr><td>{html.escape(tf)}</td><td colspan='4' class='muted'>— not enough candles</td></tr>"
            continue
        ev = st.get("last_event") or {}
        event = ("—" if not ev else
                 f"{html.escape(str(ev['type']))} {html.escape(str(ev['direction']))[:4]} @ {_dash(ev.get('level'), 0, '$')}"
                 f" <span class='muted'>({ev.get('bars_ago')} bars)</span>")
        ob = (d.get("order_blocks") or {}).get("nearest_demand") or (d.get("order_blocks") or {}).get("nearest_supply")
        gap = (d.get("fair_value_gaps") or {}).get("unfilled_below") or (d.get("fair_value_gaps") or {}).get("unfilled_above")
        pdz = (d.get("premium_discount") or {}).get("zone", "—")
        rows += (f"<tr><td>{html.escape(tf)}</td><td>{html.escape(str(st.get('bias', '—')))}</td>"
                 f"<td>{event}</td>"
                 f"<td>{'—' if not ob else _dash(ob['low'], 0, '$') + '–' + _dash(ob['high'], 0, '$')}</td>"
                 f"<td>{'—' if not gap else _dash(gap['low'], 0, '$') + '–' + _dash(gap['high'], 0, '$')}</td>"
                 f"<td>{html.escape(str(pdz))}</td></tr>")
    body = ("<table><thead><tr><th>TF</th><th>Bias</th><th>Last break</th><th>Nearest block</th>"
            f"<th>Unfilled gap</th><th>Range</th></tr></thead><tbody>{rows}</tbody></table>")
    body += ("<p class='muted'>Break of structure and change of character are taken from confirmed swings; "
             "blocks and gaps are dropped once price mitigates or fills them.</p>")
    return card_html("Market structure (SMC)", body)


def liquidity_html(by_tf: dict) -> str:
    """Pools, delta, volume profile and open interest per timeframe."""
    rows = ""
    for tf, d in by_tf.items():
        pools = (d or {}).get("pools") or {}
        cvd = (d or {}).get("cumulative_delta") or {}
        vp = (d or {}).get("volume_profile") or {}
        if not cvd.get("available"):
            rows += f"<tr><td>{html.escape(tf)}</td><td colspan='4' class='muted'>— not enough candles</td></tr>"
            continue

        def pool_cell(side: str) -> str:
            p = pools.get(side)
            if not p:
                return "—"
            state = "swept" if p.get("swept") else "broken" if p.get("broken") else "intact"
            return (f"{_dash(p.get('price'), 0, '$')} <span class='muted'>{p.get('touches')} touches · "
                    f"{state}</span>")

        oi = (d or {}).get("open_interest") or {}
        oi_cell = " / ".join("—" if oi.get(k) is None else f"{oi[k]:+.1f}%" for k in ("15m", "1h", "4h"))
        rows += (f"<tr><td>{html.escape(tf)}</td><td>{pool_cell('buyside')}</td><td>{pool_cell('sellside')}</td>"
                 f"<td>{html.escape(str(cvd.get('state', '—')))}</td>"
                 f"<td>{'—' if not vp.get('available') else _dash(vp.get('point_of_control'), 0, '$')}</td>"
                 f"<td>{oi_cell}</td></tr>")
    body = ("<table><thead><tr><th>TF</th><th>Buyside pool</th><th>Sellside pool</th><th>Delta</th>"
            f"<th>POC</th><th>OI 15m/1h/4h</th></tr></thead><tbody>{rows}</tbody></table>")
    body += ("<p class='muted'>Delta is a candle close-position proxy, not tick tape; treat it as a lean, "
             "not a measurement. A pool is swept when price wicks through and closes back inside, and "
             "broken when it closes beyond.</p>")
    return card_html("Liquidity & order flow", body)


# ---------- war-room accuracy report ----------
def accuracy_report_html(rep) -> str:
    """The head agent's grade of both teams, and the protocols to correct when it is not optimal."""
    def pct(v):
        return "—" if v is None else f"{v:.0%}"

    def brier(v):
        return "—" if v is None else f"{v:.2f}"

    if rep.status == "collecting":
        body = (f"<p class='muted'>{html.escape(rep.headline)} Each pinned summary is scored once its candle "
                "closes, against the price at that close.</p>")
        return card_html("Accuracy report", body)

    body = f"<p>{html.escape(rep.headline)}</p>"
    body += ("<table><thead><tr><th>Direction right</th><th>Trap call right</th><th>Bullish team</th>"
             "<th>Bearish team</th></tr></thead>"
             f"<tbody><tr><td>{pct(rep.hit_rate)}</td><td>{pct(rep.trap_hit_rate)}</td>"
             f"<td>{brier(rep.bull_brier)}</td><td>{brier(rep.bear_brier)}</td></tr></tbody></table>")
    body += ("<p class='muted'>Team columns are Brier scores: 0 is perfect, 0.25 is a coin flip, higher is "
             f"worse. Over the last {rep.n} scored windows.</p>")
    if rep.status == "healthy":
        return card_html("Accuracy report", body, "up")

    for i in rep.issues:
        body += (f"<p><span class='ti-chip warn'>{html.escape(i.kind)}</span> "
                 f"{html.escape(i.finding)} <span class='muted'>({i.windows} windows)</span></p>"
                 f"<p class='muted'>Suggested change: {html.escape(i.suggestion)}</p>")
    if not rep.issues:
        body += "<p class='muted'>Accuracy is below the floor but no single protocol stands out yet.</p>"
    body += "<p class='muted'>These are suggestions for a person to review; nothing is changed automatically.</p>"
    return card_html("Accuracy report · calibration needed", body, "warn")


# ---------- volatility matrix (spec section 2) ----------
def vol_matrix_html(by_tf: dict) -> str:
    """Realized and implied volatility, ATR, the sigma channels and market memory, per timeframe."""
    rows = ""
    bands_line = ""
    for tf, m in by_tf.items():
        if not (m or {}).get("available"):
            rows += (f"<tr><td>{html.escape(tf)}</td><td colspan='6' class='muted'>"
                     f"— {html.escape(str((m or {}).get('note') or 'not enough candles'))}</td></tr>")
            continue
        prem = m.get("implied_minus_realized_pct")
        rows += (f"<tr><td>{html.escape(tf)}</td>"
                 f"<td>{_pct_or_dash(m.get('realized_vol_pct'), 1)}</td>"
                 f"<td>{_pct_or_dash(m.get('implied_vol_pct'), 1)}</td>"
                 f"<td>{'—' if prem is None else f'{prem:+.1f} pts'}</td>"
                 f"<td>{_pct_or_dash(m.get('atr_pct'), 2)}</td>"
                 f"<td>{_dash(m.get('hurst'), 3)} <span class='muted'>{html.escape(str(m.get('memory', '')))}</span></td>"
                 f"<td>{html.escape(str(m.get('regime', '—')))}</td></tr>")
        b = m.get("bands") or {}
        if not bands_line and b:
            bands_line = (f"<p>{html.escape(tf)} channel · 1&sigma; {_dash(b.get('sigma1_dn'), 0, '$')}–"
                          f"{_dash(b.get('sigma1_up'), 0, '$')} · 2&sigma; {_dash(b.get('sigma2_dn'), 0, '$')}–"
                          f"{_dash(b.get('sigma2_up'), 0, '$')} · 3&sigma; {_dash(b.get('sigma3_dn'), 0, '$')}–"
                          f"{_dash(b.get('sigma3_up'), 0, '$')}</p>")
    body = ("<table><thead><tr><th>TF</th><th>Realized</th><th>Implied</th><th>IV − RV</th><th>ATR</th>"
            f"<th>Hurst</th><th>Regime</th></tr></thead><tbody>{rows}</tbody></table>" + bands_line)
    body += ("<p class='muted'>Realized volatility is annualised from that timeframe's returns; implied "
             "comes from Deribit options or shows as — , never from realized. Hurst measures memory: 0.5 "
             "is a coin-flip walk, above 0.56 trends persist, below 0.44 moves mean-revert; readings "
             "inside that band are noise.</p>")
    return card_html("Volatility matrix", body)


# ---------- institutional quantitative matrix (spec section 5) ----------
def _deck_value(value, kind: str) -> str:
    if value is None or value == "":
        return "—"
    if kind == "usd":
        return fmt_num(value, 0, "$")
    if kind == "pct":
        return f"{float(value):+.2f}%"
    if kind == "rate":
        return f"{float(value) * 100:+.3f}%"
    if kind == "num":
        return fmt_num(value, 3) if abs(float(value)) < 100 else fmt_num(value, 0)
    return html.escape(str(value))


def agent_matrix_html(deck) -> str:
    """What each of the five sub-agents read, straight from the state the judges were given."""
    body = ""
    for a in deck:
        rows = "".join(f"<tr><td class='muted'>{html.escape(label)}</td>"
                       f"<td>{_deck_value(value, kind)}</td></tr>" for label, value, kind in a.rows)
        body += (f"<p><strong>{a.agent_no}. {html.escape(a.title)}</strong> "
                 f"<span class='muted'>· {html.escape(a.timeframe)}</span></p>"
                 f"<table><tbody>{rows}</tbody></table>")
        if a.note:
            body += f"<p class='muted'>{html.escape(a.note)}</p>"
    body += ("<p class='muted'>These are the exact readings handed to the ten domain agents, so the "
             "scorecard cannot disagree with what is shown here. A dash means no feed, not zero.</p>")
    return card_html("Institutional quantitative matrix", body)


# ---------- TRAP intelligence console (spec section 1) ----------
TRAP_LABEL = {"bull_trap": "bull trap", "bear_trap": "bear trap"}
TRAP_SIDES = ("bull_trap", "bear_trap")


def _trap_side(row) -> tuple[str, str]:
    """The side as stored and as written. An unrecognised side is named as unknown rather than described
    as a bear trap by falling through a `== "bull_trap"` branch, and never printed as the word None."""
    raw = row.get("side") if isinstance(row, dict) else getattr(row, "side", None)
    if raw in TRAP_SIDES:
        return raw, TRAP_LABEL[raw]
    return "", (str(raw) if raw else "trap of unknown side")


def _trap_age(ms, now_ms: int) -> str:
    """How long ago, or a dash. `.get(key, default)` returns the default only when the key is absent, so
    a stored NULL arrived here as None and raised - which the caller then reported as a missing trap."""
    if ms is None or now_ms is None:
        return "—"
    try:
        mins = max(0, (int(now_ms) - int(ms)) // 60000)
    except (TypeError, ValueError):
        return "—"
    return f"{mins // 1440}d {(mins % 1440) // 60}h" if mins >= 1440 else f"{mins // 60}h {mins % 60:02d}m"


def _evidence_list(raw) -> list[str]:
    """Evidence as stored (a JSON string) or already decoded. A trap card without its evidence would
    assert a trap with nothing behind it, so a blob that will not parse says so instead of going quiet."""
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(e) for e in raw]
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return ["(the stored evidence for this trap could not be read)"]
    if isinstance(parsed, (list, tuple)):
        return [str(e) for e in parsed]
    return [str(parsed)]


def trap_card_html(row: dict, now_ms: int, category_label: str = "") -> str:
    """One anchored trap: what it is, the level it is built on, and what settles it."""
    side_key, side = _trap_side(row)
    tone = "warn" if row.get("status") == "active" else "neutral"
    head = (f"<p><span class='ti-chip warn'>{html.escape(side)}</span> "
            f"<strong>{html.escape(str(row.get('timeframe') or '—'))} at "
            f"{fmt_num(row.get('level'), 0, '$')}</strong>")
    if category_label:
        head += f" <span class='muted'>· declared by the {html.escape(category_label)} desk</span>"
    # a dash means the score was never recorded, not that the sub-agents scored it zero
    score = row.get("score")
    head += (f" <span class='muted'>· declared {_trap_age(row.get('declared_ms'), now_ms)} ago"
             f" · sub-agents {fmt_num(score, 0) if score is not None else '—'}/100</span></p>")
    body = head
    if side_key == "bull_trap":
        body += f"<p>Invalidated above {fmt_num(row.get('invalidation'), 0, '$')}"
    elif side_key == "bear_trap":
        body += f"<p>Invalidated below {fmt_num(row.get('invalidation'), 0, '$')}"
    else:
        body += f"<p>Invalidated at {fmt_num(row.get('invalidation'), 0, '$')}"
    body += f" · pays off back at {fmt_num(row.get('plays_out'), 0, '$')}</p>"
    body += _li(_evidence_list(row.get("evidence")))
    if row.get("status") != "active":
        body += (f"<p class='muted'>{html.escape(str(row.get('status') or 'unknown').replace('_', ' '))} at "
                 f"{fmt_num(row.get('resolved_price'), 0, '$')}</p>")
    return card_html(f"Anchored {side}", body, tone)


def trap_console_html(rows: list[dict], candidates: list, now_ms: int, labels: dict | None = None,
                      settled: list[dict] | None = None, swept_ms: int | None = None) -> str:
    """The watchdog console: what is anchored, and what is being watched but not declared.

    `rows` are the anchored traps and `settled` the recently closed ones, queried separately so newer
    settled traps cannot push the live ones out of sight. The watch list is dated: candidates live in
    session state and survive every auto-refresh, so an undated one read as current long after the scan
    that found it."""
    labels = labels or {}
    settled = settled or []
    active = [r for r in rows if r.get("status") == "active"]
    body = (f"<p><strong>{len(active)} anchored</strong> · {len(candidates)} candidate(s) on watch · "
            f"{len(settled)} recently settled</p>")
    if not active:
        body += ("<p class='muted'>No trap is anchored. A trap is declared only when its five sub-agents "
                 "score 60 or more, the TRAP Head calls it engineered, and the category desk does not read "
                 "the move as an authentic break.</p>")
    if candidates:
        age = _trap_age(swept_ms, now_ms)
        body += (f"<p>On watch, from the deterministic scan of {age} ago:</p><ul>"
                 if swept_ms else "<p>On watch, from the deterministic scan:</p><ul>")
        for c in candidates[:6]:
            _key, label = _trap_side(c)
            body += (f"<li>{html.escape(label)} · {html.escape(str(c.timeframe))} at "
                     f"{fmt_num(c.level, 0, '$')} · {c.strength} factors: "
                     f"{html.escape('; '.join(c.evidence[:2]))}</li>")
        body += "</ul>"
        if swept_ms and (now_ms - swept_ms) > 30 * 60_000:
            body += ("<p class='muted'>That scan is over half an hour old. Sweep again before trading "
                     "from it - these levels were measured against a price that has since moved.</p>")
    if settled:
        parts = []
        for r in settled:
            _key, label = _trap_side(r)
            desk = labels.get(r.get("category"), r.get("category") or "")
            parts.append(f"{label} {r.get('timeframe') or '?'} "
                         f"{str(r.get('status') or '').replace('_', ' ')}"
                         + (f" ({desk})" if desk else ""))
        body += "<p class='muted'>Recently settled: " + html.escape(", ".join(parts)) + "</p>"
    body += ("<p class='muted'>Traps stay anchored until price settles them: invalidated when the break "
             "proves real, played out when price returns inside value. Nothing expires on a clock.</p>")
    return card_html("TRAP intelligence · watchdog", body, "warn" if active else "neutral")


def trap_warning_html(rows: list[dict], now_ms: int) -> str | None:
    """The outbound half of the handshake, on every desk that reads the trap's timeframe."""
    active = [r for r in rows if r.get("status") == "active"]
    if not active:
        return None
    body = ""
    for r in active[:3]:
        side_key, side = _trap_side(r)
        body += (f"<p><span class='ti-chip warn'>{html.escape(side)}</span> "
                 f"{html.escape(str(r.get('timeframe') or '—'))} at {fmt_num(r.get('level'), 0, '$')} · "
                 f"invalidated at {fmt_num(r.get('invalidation'), 0, '$')} · "
                 f"declared {_trap_age(r.get('declared_ms'), now_ms)} ago</p>")
    if len(active) > 3:
        body += f"<p class='muted'>and {len(active) - 3} more on this desk's timeframes.</p>"
    body += "<p class='muted'>From the TRAP desk. It stays here until price settles it.</p>"
    return card_html("TRAP warning", body, "warn")


# ---------- the pinned Active Trade card (spec sections 3 and 4) ----------
STOP_TONE = {"sound": "up", "supplied": "neutral", "would be hunted": "down",
             "inside the noise": "down", "on the wrong side": "down"}
ALIGN_WORD = {"supports": "supports this trade", "opposes": "against this trade",
              "neutral": "no directional call", "silent": "no live verdict"}


def _when(ms, now_ms: int, future: bool = True) -> str:
    """A date a trader can set an alarm by.

    `future=True` is for dates that are meant to be ahead of us - a target's due time, a monitoring
    checkpoint - where having gone past is news. It was applied to `opened_ms` and `refreshed_ms` too,
    which are past by definition, so every card permanently read "opened 19:37 UTC (passed)"."""
    if not ms:
        return "no date"
    from core.trades import fmt_when
    try:
        stamp = int(ms)
    except (TypeError, ValueError):
        return "no date"
    return fmt_when(stamp) + (" (passed)" if future and stamp <= now_ms else "")


def trade_card_html(trade: dict, prog=None, now_ms: int = 0, price_now: float | None = None) -> str:
    """One pinned position: the plan as it was written, and where it stands now.

    `price_now` lets the card tell "no price this refresh" apart from "there is a price but this
    position's arithmetic failed" - it used to blame the feed for both."""
    plan = trade.get("plan") or {}
    side = str(trade.get("side") or "")
    asset = str(trade.get("asset") or "")
    closed = trade.get("status") != "open"
    entry = trade.get("entry")
    stop = trade.get("stop")

    head = (f"<p><span class='ti-chip {'up' if side == 'LONG' else 'down'}'>{html.escape(side)}</span> "
            f"<strong>{html.escape(asset)} from {fmt_num(entry, 0, '$')}</strong>")
    if plan.get("ok"):
        head += (f" <span class='muted'>· score {fmt_num(plan.get('score'), 0)}/100 · "
                 f"{fmt_num(plan.get('probability'), 0, suffix='%')} confluence estimate</span>")
    else:
        head += " <span class='muted'>· unscored: the sub-agents did not answer</span>"
    head += f" <span class='muted'>· opened {_when(trade.get('opened_ms'), now_ms, future=False)}</span></p>"
    body = head

    # --- live block: the only part a Refresh changes ---
    if closed:
        body += (f"<p class='muted'>Closed {_when(trade.get('closed_ms'), now_ms, future=False)} at "
                 f"{fmt_num(trade.get('closed_price'), 0, '$')} - "
                 f"{html.escape(str(trade.get('closed_reason') or 'no reason recorded'))}.</p>")
    elif prog is not None:
        hit = ", ".join(prog.hit) if prog.hit else "none yet"
        # the price is recomputed on every rerun, so it is stamped with now, not with the last time
        # the Refresh button was pressed - which used to sit under a live number and make it look stale.
        # `refreshed_ms` is reported separately, and only when it really happened.
        body += (f"<p>Now {fmt_num(prog.price, 0, '$')} · <strong>{prog.pnl_pct:+.2f}%</strong> "
                 f"({prog.pnl_r:+.2f}R) · targets reached: {html.escape(hit)}"
                 + (" · <strong>the stop has been reached</strong>" if prog.stopped else "")
                 + f" <span class='muted'>· as of {_when(now_ms, now_ms, future=False)}"
                 + (f", last manual refresh {_when(trade.get('refreshed_ms'), now_ms, future=False)}"
                    if trade.get("refreshed_ms") else "")
                 + "</span></p>")
    elif price_now:
        body += (f"<p class='muted'>Price is {fmt_num(price_now, 0, '$')}, but this position's live "
                 "numbers could not be calculated - see the warning above. The plan below is unaffected."
                 "</p>")
    else:
        body += "<p class='muted'>No live price this refresh, so the numbers below are the plan only.</p>"

    # --- the stop ---
    verdict = str(plan.get("stop_verdict") or trade.get("stop_verdict") or "")
    given = trade.get("given_stop")
    body += (f"<p><span class='ti-chip {STOP_TONE.get(verdict, 'neutral')}'>stop "
             f"{html.escape(verdict)}</span> trading at {fmt_num(stop, 0, '$')}")
    if given is not None and verdict not in ("sound", "supplied"):
        body += f" <span class='muted'>· you gave {fmt_num(given, 0, '$')}, corrected here</span>"
    body += "</p>" + _li([str(r) for r in (plan.get("stop_reasons") or [])])

    # --- the timed targets ---
    targets = plan.get("targets") or []
    if targets:
        rows = "".join(
            f"<tr><td>{html.escape(str(t.get('name')))}</td>"
            f"<td>{fmt_num(t.get('price'), 0, '$')}</td>"
            f"<td>{fmt_num(t.get('reward_r'), 2, suffix='R')}</td>"
            f"<td>{html.escape(_when(t.get('due_ms'), now_ms))}</td>"
            f"<td class='muted'>{html.escape(str(t.get('due_from') or ''))}</td>"
            f"<td class='muted'>{html.escape(str(t.get('basis') or ''))}</td></tr>" for t in targets)
        body += ("<p>Timed take-profits</p><table><thead><tr><th>Target</th><th>Price</th><th>Reward</th>"
                 "<th>Due</th><th>Dated by</th><th>Price from</th></tr></thead>"
                 f"<tbody>{rows}</tbody></table>")
        first = targets[0]
        try:
            if float(first.get("reward_r") or 0) < 1.0:
                body += ("<p class='ti-chip down'>reward below risk</p><p class='muted'>The first target "
                         "is nearer than the stop, so this position risks more than it makes at TP1.</p>")
        except (TypeError, ValueError):
            pass

    # --- the pullback guardrail, in plain English ---
    pb = plan.get("pullback") or {}
    if pb.get("available"):
        levels = pb.get("levels") or {}
        body += (f"<p>Price elasticity and pullback guardrail</p>"
                 f"<p>Over the last {pb.get('horizon_bars')} {html.escape(str(pb.get('timeframe')))} "
                 f"candles this market has normally gone "
                 f"{fmt_num(pb.get('ordinary'), 0, '$')} against a position like this before continuing, "
                 f"and {fmt_num(pb.get('edge'), 0, '$')} in four cases out of five. So a move to "
                 f"{fmt_num(levels.get('ordinary'), 0, '$')} is ordinary and "
                 f"{fmt_num(levels.get('edge'), 0, '$')} is still normal - do not close there. "
                 f"Past {fmt_num(levels.get('breaking'), 0, '$')} this is no longer a pullback: that is "
                 f"the 95th percentile, and the move you were expecting is not happening.</p>")
    elif pb:
        body += (f"<p class='muted'>No pullback guardrail: {html.escape(str(pb.get('note') or 'not measurable'))}."
                 "</p>")

    # --- the monitoring schedule ---
    schedule = plan.get("schedule") or []
    if schedule:
        body += "<p>Chart monitoring schedule</p><ul>"
        for c in schedule:
            body += (f"<li>{html.escape(_when(c.get('at_ms'), now_ms))} - "
                     f"{html.escape(str(c.get('reason') or ''))}</li>")
        body += "</ul>"

    # --- the two handshakes ---
    alignment = plan.get("alignment") or []
    if alignment:
        body += (f"<p>Category consultation <span class='muted'>· "
                 f"{html.escape(str(plan.get('aligned') or ''))}</span></p><ul>")
        for a in alignment:
            word = ALIGN_WORD.get(str(a.get("verdict")), str(a.get("verdict")))
            body += (f"<li>{html.escape(str(a.get('label')))}: "
                     f"{html.escape(str(a.get('bias') or 'no verdict'))} - {html.escape(word)}"
                     + (f" <span class='muted'>({html.escape(str(a.get('note')))})</span>"
                        if a.get("note") else "") + "</li>")
        body += "</ul>"

    traps = plan.get("traps") or []
    if traps:
        body += "<p><span class='ti-chip warn'>TRAP consultation</span></p><ul>"
        for t in traps:
            body += (f"<li>An anchored {html.escape(str(t.get('side', '')).replace('_', ' '))} on the "
                     f"{html.escape(str(t.get('timeframe')))} sits {fmt_num(t.get('distance'), 0, '$')} "
                     f"from this trade's {html.escape(str(t.get('near')))} at "
                     f"{fmt_num(t.get('level'), 0, '$')}</li>")
        body += "</ul>"
    elif plan.get("ok"):
        body += "<p class='muted'>TRAP consultation: no anchored trap sits on this trade's levels.</p>"

    # --- the desks' own numbers and the head ---
    if plan.get("ok"):
        for label, key in (("for", "for_domains"), ("against", "against_domains")):
            doms = plan.get(key) or []
            if doms:
                body += (f"<p class='muted'>The case {label} "
                         f"({fmt_num(plan.get(f'{label}_points'), 0)}/100): "
                         + ", ".join(f"{html.escape(str(d.get('title')))} "
                                     f"{fmt_num(d.get('points'), 0)}/20" for d in doms) + "</p>")
        if plan.get("head_call"):
            conf = plan.get("head_confidence")
            body += (f"<p><strong>Head of the Active Trade desk: "
                     f"{html.escape(str(plan['head_call']).replace('_', ' ').title())}</strong>"
                     + (f" <span class='muted'>({float(conf):.0%} confidence)</span>" if conf else "")
                     + "</p>")

    body += _li([str(n) for n in (plan.get("notes") or [])])
    body += ("<p class='muted'>The score and the estimate come from the agents' own probabilities and the "
             "desks' alignment. They are a measure of confluence, not a backtested win rate - nothing "
             "here counts how often a setup like this has worked.</p>")

    tone = "neutral" if closed else ("down" if (prog is not None and prog.stopped) or traps else "warn")
    title = f"{asset} {side}" + ("" if not closed else " (closed)")
    return card_html(title, body, tone)


# ---------- macro and news engines (spec section 2) ----------
IMPACT_TONE = {"high": "down", "medium": "warn", "low": "neutral"}
BAND_TONE = {"clean": "up", "watch": "warn", "fragile": "down", "unknown": "neutral"}


def _age(ms, now_ms: int) -> str:
    if not ms:
        return "undated"
    mins = max(0, (int(now_ms) - int(ms)) // 60_000)
    if mins < 60:
        return f"{mins}m ago"
    return f"{mins // 60}h ago" if mins < 1440 else f"{mins // 1440}d ago"


def news_html(news, now_ms: int, limit: int = 12) -> str:
    """Headlines, newest first, with the keyword tags shown as tags.

    The topic and impact labels are keyword matches, not judgments - a headline can be mislabelled by a
    word used in passing - so the card says so rather than letting "high impact" read as a verdict."""
    if not getattr(news, "available", False):
        return card_html("News", f"<p class='muted'>No headline feed: "
                                 f"{html.escape(str(getattr(news, 'error', '') or 'unavailable'))}.</p>",
                         "neutral")
    headlines = list(getattr(news, "headlines", []) or [])
    high = [h for h in headlines if h.get("impact") == "high"]
    body = (f"<p><strong>{len(headlines)} headlines</strong> from "
            f"{html.escape(str(getattr(news, 'source', '')))}"
            + (f" · <span class='ti-chip down'>{len(high)} high-impact</span>" if high else "")
            + "</p>")
    if getattr(news, "error", None):
        body += (f"<p class='muted'>Partial feed: {html.escape(str(news.error))[:160]} - the count above "
                 "is what did arrive, not everything published.</p>")
    body += "<table><thead><tr><th>When</th><th>Topic</th><th>Headline</th></tr></thead><tbody>"
    for h in headlines[:limit]:
        assets = " ".join(str(a) for a in (h.get("assets") or [])[:3])
        tone = IMPACT_TONE.get(str(h.get("impact")), "neutral")
        title = html.escape(str(h.get("title") or ""))[:150]
        url = str(h.get("url") or "")
        # only an http(s) link is rendered as one: a feed controls this string
        linked = (f"<a href='{html.escape(url)}' target='_blank' rel='noopener noreferrer'>{title}</a>"
                  if url.startswith(("https://", "http://")) else title)
        body += (f"<tr><td class='muted'>{html.escape(_age(h.get('published_ms'), now_ms))}</td>"
                 f"<td><span class='ti-chip {tone}'>{html.escape(str(h.get('topic') or ''))}</span></td>"
                 f"<td>{linked}"
                 + (f" <span class='muted'>{html.escape(assets)}</span>" if assets else "")
                 + f" <span class='muted'>· {html.escape(str(h.get('source') or ''))}</span></td></tr>")
    body += "</tbody></table>"
    body += ("<p class='muted'>Topic and impact are keyword tags over the headline text, not a judgment "
             "of the story. Treat them as a way to sort the list, not as a read on the market.</p>")
    return card_html("News", body, "down" if high else "neutral")


def gold_html(metals, now_ms: int) -> str:
    """Gold spot, with the feed's own freshness carried through."""
    if not getattr(metals, "available", False):
        return card_html("Gold (XAU/USD)",
                         f"<p class='muted'>No gold feed: "
                         f"{html.escape(str(getattr(metals, 'error', '') or 'unavailable'))}.</p>", "neutral")
    price = getattr(metals, "xau_usd", None)
    body = (f"<p><strong>{fmt_num(price, 2, '$')}</strong> per "
            f"{html.escape(str(getattr(metals, 'unit', 'troy_ounce')).replace('_', ' '))}")
    bid, ask = getattr(metals, "bid", None), getattr(metals, "ask", None)
    if bid is not None and ask is not None:
        body += f" <span class='muted'>· bid {fmt_num(bid, 2, '$')} / ask {fmt_num(ask, 2, '$')}</span>"
    body += "</p>"
    if getattr(metals, "is_stale", False):
        body += ("<p><span class='ti-chip down'>stale</span> The feed reports this print as stale, so it "
                 "is not the current price.</p>")
    else:
        body += (f"<p class='muted'>Computed {html.escape(_age(getattr(metals, 'computed_ms', None), now_ms))}"
                 f" · {html.escape(str(getattr(metals, 'source', '')))}</p>")
    body += ("<p class='muted'>Gold only: silver and copper are gated behind a paid tier on this feed, so "
             "they are not shown rather than guessed at.</p>")
    return card_html("Gold (XAU/USD)", body, "warn" if getattr(metals, "is_stale", False) else "neutral")


def predictions_html(preds, now_ms: int, asset: str | None = None, limit: int = 8) -> str:
    """What the crowd is pricing, and whether the quote can actually be filled."""
    if not getattr(preds, "available", False):
        return card_html("Prediction markets",
                         f"<p class='muted'>No odds feed: "
                         f"{html.escape(str(getattr(preds, 'error', '') or 'unavailable'))}.</p>", "neutral")
    markets = list(getattr(preds, "markets", []) or [])
    if asset:
        markets = [m for m in markets if asset in (m.get("assets") or [])]
    tradeable = [m for m in markets if m.get("band") in ("clean", "watch")]
    body = (f"<p><strong>{len(markets)} market(s)</strong>"
            + (f" mentioning {html.escape(asset)}" if asset else "")
            + f" · {len(tradeable)} with a quote worth quoting</p>")
    body += ("<table><thead><tr><th>Probability</th><th>Execution</th><th>24h volume</th><th>Ends</th>"
             "<th>Market</th></tr></thead><tbody>")
    for m in markets[:limit]:
        band = str(m.get("band") or "unknown")
        tone = BAND_TONE.get(band, "neutral")
        question = html.escape(str(m.get("question") or ""))[:120]
        url = str(m.get("url") or "")
        linked = (f"<a href='{html.escape(url)}' target='_blank' rel='noopener noreferrer'>{question}</a>"
                  if url.startswith("https://") else question)
        days = m.get("days_left")
        body += (f"<tr><td><strong>{fmt_num((m.get('probability') or 0) * 100, 1, suffix='%')}</strong></td>"
                 f"<td><span class='ti-chip {tone}'>{html.escape(band)}</span>"
                 + (f" <span class='muted'>{fmt_num(m.get('band_score'), 0)}</span>"
                    if m.get("band_score") is not None else "")
                 + f"</td><td>{fmt_num(m.get('volume_24h'), 0, '$')}</td>"
                 f"<td class='muted'>{fmt_num(days, 0) if days is not None else '—'}d</td>"
                 f"<td>{linked}</td></tr>")
    body += "</tbody></table>"
    fragile = [m for m in markets[:limit] if m.get("band") == "fragile"]
    if fragile:
        reason = next((str(m.get("band_reason")) for m in fragile if m.get("band_reason")), "")
        body += (f"<p class='muted'><strong>{len(fragile)} of these are fragile.</strong> "
                 + html.escape(reason or "The quote can move or fill badly.")
                 + " A probability on a fragile market is a printed number, not a forecast.</p>")
    attribution = str(getattr(preds, "attribution", "") or "")
    if attribution:
        body += f"<p class='muted'>{html.escape(attribution)}</p>"
    return card_html("Prediction markets", body, "neutral")
