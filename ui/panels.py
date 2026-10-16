"""Panel builders: pure HTML functions plus thin Streamlit renderers."""
from __future__ import annotations

import html
import json

import streamlit as st

from core.data.types import MarketSnapshot
from core.indicators.category import CategoryAnalysis
from core.indicators.trade_map import TradeMap
from ui.theme import CORNERS, card_html, chip, fmt_num, fmt_pct, palette, tone_for_direction


def render(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def _li(items: list[str]) -> str:
    """A list the way the design sets one: a dash per line, no bullets."""
    return "".join(f"<p class='ti-dash'>— {html.escape(str(i))}</p>" for i in items) if items else ""


def _kv(label: str, value: str, cls: str = "") -> str:
    """A labelled figure: small uppercase label over a condensed value. `value` is already escaped."""
    return (f"<div class='ti-kv'><span class='ti-label'>{html.escape(label)}</span>"
            f"<span class='ti-kv-v {cls}'>{value}</span></div>")


def _bar(pct, tone: str = "") -> str:
    p = 0 if pct is None else max(0, min(100, round(float(pct))))
    return f"<div class='ti-bar {tone}'><div style='width:{p}%'></div></div>"


def section_head_html(kicker: str, title: str | None = None) -> str:
    """A section opener: a steel kicker on a hairline rule, then the section's own heading."""
    out = f"<div class='ti-section'><span class='ti-kicker'>{html.escape(kicker)}</span><div class='ti-rule'></div></div>"
    return out + (f"<h2 class='ti-h2'>{html.escape(title)}</h2>" if title else "")
def _liq_kpi(l) -> dict:
    if not l.available or not l.total_usd:
        return {"label": "Liq 24h", "value": "—", "delta": None, "tone": None}
    share = (l.long_usd or 0.0) / l.total_usd * 100.0
    tone = "down" if share >= 60 else "up" if share <= 40 else None
    return {"label": "Liq 24h", "value": fmt_num(l.total_usd / 1e6, 0, "$", "M"), "delta": f"{share:.0f}% longs", "tone": tone}


def _taker_kpi(flow) -> dict:
    """Net taker delta over the last 24 completed hours: who has been paying the spread."""
    df = (flow.frames.get("1h") if getattr(flow, "available", False) else None)
    if df is None or len(df) < 25:
        return {"label": "Taker delta 24h", "value": "—", "delta": "no taker feed", "tone": None}
    net = float(df["delta"].iloc[-25:-1].sum())           # the newest hour is still forming
    side = "buyers paying" if net > 0 else "sellers paying" if net < 0 else "balanced"
    return {"label": "Taker delta 24h", "value": f"{net:+,.0f} BTC", "delta": side,
            "tone": "up" if net > 0 else "down" if net < 0 else None}


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
        _taker_kpi(getattr(m, "taker_flow", None)),
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


CANDLE_WORD = {"1h": "1h", "4h": "4h", "1d": "daily", "1w": "weekly"}
HORIZON_WORD = {"next 4h": "next 4 hours", "next 24h": "next 24 hours"}


def _times(n: int) -> str:
    return "once" if n == 1 else "twice" if n == 2 else f"{n} times"


def _conf_word(c: float) -> str:
    # a directional call needs |score| > 0.15, so the bands start there
    return "Very strong" if c >= 0.85 else "Strong" if c >= 0.60 else "Moderate" if c >= 0.35 else "Slight"


def layman_lines(a: CategoryAnalysis) -> list[tuple[str, str]]:
    """The read in everyday words: what is happening, the levels that matter, what changes the view, and
    - only when one applies - what else is worth noting. Built only from the deterministic analysis; the
    jargon and the exact numbers (EMA stack, confidence %, 2-sigma band, upcoming Fibonacci zones) stay in
    the detail cards below."""
    d, v, s = a.direction, a.vol, a.squeeze
    money = lambda x: fmt_num(x, 0, "$")  # noqa: E731
    sups = sorted((l for l in a.levels if l.kind == "support"), key=lambda l: -l.price)
    res = sorted((l for l in a.levels if l.kind == "resistance"), key=lambda l: l.price)
    sup, nxt, ceil = (sups[0] if sups else None), (sups[1] if len(sups) > 1 else None), (res[0] if res else None)
    chart = {"1d": "daily chart", "1w": "weekly chart"}.get(a.chart_tf, f"{a.chart_tf} chart")
    candle = CANDLE_WORD.get(a.chart_tf, a.chart_tf)
    price = money(a.price)

    if d.direction in ("BULLISH", "BEARISH"):
        now = (f"{_conf_word(d.confidence)} lean {'higher' if d.direction == 'BULLISH' else 'lower'}. "
               f"BTC is {price} on the {chart}.")
    elif sup and ceil:
        now = f"No clear side. BTC is {price} on the {chart}, stuck between its floor and ceiling."
    else:
        now = f"No clear side. BTC is {price} on the {chart}, with mixed signals."

    if sup and ceil:
        levels = (f"Floor {money(sup.price)} (held {_times(sup.touches)}), "
                  f"ceiling {money(ceil.price)} (rejected {_times(ceil.touches)}).")
    elif sup:
        levels = f"Floor {money(sup.price)} (held {_times(sup.touches)}); no clear ceiling above."
    elif ceil:
        levels = f"No clear floor below; ceiling {money(ceil.price)} (rejected {_times(ceil.touches)})."
    else:
        levels = ""
    if v.exp_low and v.exp_high:
        horizon = HORIZON_WORD.get(v.horizon_label, v.horizon_label)
        span = f"{money(v.exp_low)} to {money(v.exp_high)}"
        levels += " " + {"HIGH": f"Range for the {horizon}, wider than usual: {span}.",
                         "LOW": f"Range for the {horizon}, tighter than usual: {span}."}.get(
            v.regime, f"Normal range for the {horizon}: {span}.")
    elif not levels:
        levels = "Not enough history yet for levels or a range."
    if not (sup or ceil) and v.exp_low and v.exp_high:
        levels = "No clear floor or ceiling nearby. " + levels.strip()
    levels = levels.strip()

    if d.direction == "BEARISH" and d.invalidation is not None:
        change = f"A {candle} close above {money(d.invalidation)}, or the moving averages turning up, flips this higher."
        if sup and nxt:
            change += f" Losing {money(sup.price)} opens the way toward {money(nxt.price)}."
        elif sup and v.exp_low:
            change += f" Losing {money(sup.price)} opens the way toward the range low, {money(v.exp_low)}."
        elif sup:
            change += f" Losing {money(sup.price)} leaves no clear floor beneath."
        else:
            change += " There is no clear floor beneath to lean on."
    elif d.direction == "BULLISH" and d.invalidation is not None:
        # the bullish invalidation sits half an ATR under the floor - a few hundred dollars on the 1h chart,
        # thousands on the weekly - so the floor is named rather than called "just under"
        under = f" (under the {money(sup.price)} floor)" if sup and sup.price > d.invalidation else ""
        change = (f"A {candle} close below {money(d.invalidation)}{under}, or the moving averages "
                  "turning down, flips this lower.")
    elif d.direction in ("BULLISH", "BEARISH"):
        change = (f"The moving averages turning the other way flips this. No clear "
                  f"{'floor' if d.direction == 'BULLISH' else 'ceiling'} nearby to anchor it.")
    elif sup and ceil:
        change = (f"A {candle} close below {money(sup.price)} tilts it lower, above {money(ceil.price)} higher, "
                  "if the moving averages agree.")
    elif sup:
        change = f"A {candle} close below {money(sup.price)} tilts it lower, if the moving averages agree."
    elif ceil:
        change = f"A {candle} close above {money(ceil.price)} tilts it higher, if the moving averages agree."
    else:
        change = "A clean break either way, backed by the moving averages, picks a side."

    notes: list[str] = []
    rsi_last = float(a.rsi.dropna().iloc[-1]) if a.rsi.notna().any() else None
    if rsi_last is not None and rsi_last >= 70:
        notes.append(f"RSI {rsi_last:.0f}, overbought, the move is stretched")
    elif rsi_last is not None and rsi_last <= 30:
        notes.append(f"RSI {rsi_last:.0f}, oversold, the drop is stretched")
    rw = a.reversal
    if rw is not None and rw.status == "ACTIVE":
        lean = "with mixed signals" if rw.bias == "mixed" else f"leaning {rw.bias}"
        # the confluence list is engine shorthand ("at resistance 82,506 (6 touches)"); the card below has it
        notes.append(f"a reversal window is open right now, {lean}")
    if a.patterns:
        p0 = a.patterns[0]
        name = p0.name.lower()
        if p0.status == "confirmed":
            notes.append(f"a {name} just confirmed, {p0.bias}" + (f", pointing to {money(p0.target)}" if p0.target else ""))
        elif p0.bias == "neutral":
            notes.append(f"a {name} is forming, neutral until it breaks")
        else:
            notes.append(f"a {name} is forming, {p0.bias}"
                         + (f", a break of {money(p0.breakout)} points to {money(p0.target)}"
                            if p0.breakout and p0.target else ""))
    if a.divergences:
        dv = next((x for x in a.divergences if x.status == "confirmed"), a.divergences[0])
        if dv.kind == "hidden":
            notes.append(f"a hidden {dv.direction} divergence favours the current trend continuing ({dv.status})")
        else:
            swing = "higher high" if dv.direction == "bearish" else "lower low"
            notes.append(f"price made a {swing} without momentum behind it ({dv.status} {dv.direction} divergence)")
    if s.score is not None and s.score >= 40 and s.direction in ("LONG_SQUEEZE", "SHORT_SQUEEZE"):
        crowd = "heavily crowded" if s.score >= 70 else "crowded"
        notes.append(f"{crowd} longs ({s.score}/100) raise the odds of a sharp shakeout lower"
                     if s.direction == "LONG_SQUEEZE" else
                     f"{crowd} shorts ({s.score}/100) raise the odds of a sharp pop higher")

    lines = [("What's happening", now), ("Levels that matter", levels), ("What changes the view", change)]
    if notes:
        text = "; ".join(notes)
        lines.append(("Also worth noting", text[0].upper() + text[1:] + "."))
    return lines


def layman_html(a: CategoryAnalysis) -> str:
    """Plain-English summary built only from the deterministic analysis."""
    # text nodes only, so quotes need no escaping ("What's happening" stays readable in the source)
    rows = "".join(f"<p><strong>{html.escape(k, quote=False)}:</strong> {html.escape(t, quote=False)}</p>"
                   for k, t in layman_lines(a))
    return f"<div class='ti-read'><span class='ti-kicker'>Plain-English read</span>{rows}</div>"


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
    tone = tone_for_direction(an.direction)
    body = (f"<div class='ti-dir {tone}'>{DIR_GLYPH.get(an.direction, '◆')} {an.direction}</div>"
            f"<p>Confidence {an.confidence:.0%} · anchored {_ago(now_ms - v.since_ms)} ago "
            f"at {fmt_num(an.price, 0, '$')} · {a.chart_tf} chart</p>"
            f"<div class='ti-bar {tone}'><div style='width:{round(an.confidence * 100)}%'></div></div>")
    if v.unconfirmed:
        body += (f"<p><span class='ti-chip warn'>live read {html.escape(v.live.direction)} ({v.live.confidence:.0%})</span> "
                 f"unconfirmed until a trigger fires</p>")
    if v.changed and v.triggers_fired and v.triggers_fired != ["initial"]:
        body += f"<p class='muted'>Re-anchored now on: {html.escape(', '.join(v.triggers_fired))}</p>"
    body += (f"<div class='ti-pair'><div><span class='ti-label'>Anchor</span><span class='ti-pair-v'>"
             f"{fmt_num(an.anchor, 0, '$')}</span></div><div><span class='ti-label'>Invalidation</span>"
             f"<span class='ti-pair-v down'>{fmt_num(an.invalidation, 0, '$')}</span></div></div>")
    body += f"<span class='ti-label'>Flips if</span><p>{html.escape(an.flips_if)}</p>"
    body += f"<span class='ti-label'>Drivers</span>" + _li(an.drivers)
    body += ("<details class='ti-triggers'><summary>Show the eight re-anchor triggers</summary><div class='ti-tags'>"
             + "".join(chip(t.strip(), "neutral-fill") for t in TRIGGERS_WATCHED.split("·"))
             + f"</div><p class='muted'>Triggers watched: {TRIGGERS_WATCHED}</p></details>")
    return card_html("Market direction · anchored", body, tone)


def early_warning_html(ws) -> str | None:
    if not ws:
        return None
    body = ""
    for w in ws:
        body += (f"<div class='ti-early-row'><span class='ti-early-title'>Early warning · "
                 f"{html.escape(w.kind.replace('_', ' '))}</span><strong>{html.escape(w.level)}</strong></div>"
                 + "<div class='ti-early-drivers'>" + "".join(f"<span>— {html.escape(d)}</span>" for d in w.drivers) + "</div>")
    return f"<div class='ti-early blueprint ti-card warn'>{CORNERS}{body}</div>"
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
    from core.trades import fmt_when
    return fmt_when(int(ms)) if ms else "—"


def _bars(n: int | None) -> str:
    if n is None:
        return ""
    if n == 0:
        return "this bar"
    return f"in {n} bar{'s' if n != 1 else ''}" if n > 0 else f"{-n} bar{'s' if n != -1 else ''} ago"


def fib_html(a: CategoryAnalysis) -> str:
    """Fibonacci time zones (two-point), the reversal window, and retracement levels."""
    f, rw, fr = a.fib, a.reversal, a.fib_retr
    title = "Fibonacci time · reversal window"
    if f is None:
        return card_html(title, "<p class='muted'>Not enough history for Fibonacci time zones.</p>")
    body, tone = "", "neutral"
    if rw is not None:
        label = {"ACTIVE": "Window open", "UPCOMING": "Upcoming", "TIME_ONLY": "Time only"}.get(rw.status, rw.status)
        body += f"<div class='ti-tags'>{chip(label, 'accent')}</div>"
        head = f"F{rw.zone_k} {_bars(rw.bars_away)}" + (f" · ±{rw.window_bars} bar" if rw.status == "ACTIVE" else "")
        body += f"<div class='ti-h3'>{html.escape(head)}</div>"
        if rw.status == "ACTIVE":
            tone = "warn"
            body += (f"<p>Reversal window open: a Fibonacci reversal window is open with a "
                     f"<strong>{html.escape(rw.bias)}</strong> lean.</p>")
            if rw.confluence:
                body += "<span class='ti-label'>Confluence</span>" + _li(rw.confluence)
        elif rw.status == "TIME_ONLY":
            body += ("<p>A Fibonacci time zone, but price has no confluence (no level, divergence, retracement "
                     "or pattern), so no reversal signal.</p>")
        else:
            body += f"<p>Next reversal window: {_utc(rw.zone_ms)}.</p>"
            if rw.confluence:
                body += "<span class='ti-label'>Confluence building now</span>" + _li(rw.confluence)
    if f.upcoming:
        body += "<div class='ti-rows'>" + "".join(
            f"<div class='ti-row3'><strong>F{z.k}</strong><span>{_utc(z.timestamp_ms)}</span>"
            f"<span class='muted'>{_bars(z.bars_from_now)}</span></div>" for z in f.upcoming) + "</div>"
    if fr is not None:
        body += (f"<span class='ti-label'>Retracement of the {fr.leg} leg, {fmt_num(fr.low, 0, '$')} – "
                 f"{fmt_num(fr.high, 0, '$')}</span><div class='ti-grid3'>"
                 + "".join(_kv(str(r), fmt_num(fr.levels[r], 0, '$')) for r in (0.382, 0.5, 0.618)) + "</div>")
    body += (f"<p class='muted ti-note'>Anchors: {f.anchor_kind} {fmt_num(f.anchor_price, 0, '$')} ({_utc(f.anchor_ms)}) → "
             f"{f.anchor2_kind} {fmt_num(f.anchor2_price, 0, '$')} ({_utc(f.anchor2_ms)}) · unit {f.unit_bars} bars</p>")
    return card_html(title, body, tone)
def patterns_html(a: CategoryAnalysis) -> str:
    if not a.patterns:
        return card_html("Chart patterns", "<p class='muted'>No clean pattern in the last 120 bars.</p>")
    body, tone = "", "neutral"
    for p in a.patterns:
        bias_tone = {"bullish": "up", "bearish": "down"}.get(p.bias, "")
        body += (f"<div class='ti-h3'>{html.escape(p.name)}</div>"
                 f"<div class='ti-tags'>{chip(p.status, 'outline')}{chip(p.bias, bias_tone)}</div>"
                 f"<p class='muted'>Confidence {p.confidence:.0%}"
                 + (f" · apex {_bars(p.apex_bars)}" if p.apex_bars else "") + "</p>")
        body += "<div class='ti-grid3'>" + "".join(
            _kv(label, fmt_num(v, 0, "$") if v is not None else "—")
            for label, v in (("Breakout", p.breakout), ("Target", p.target), ("Invalidation", p.invalidation))) + "</div>"
        if p.note:
            body += f"<p class='muted'>{html.escape(p.note)}</p>"
        if tone == "neutral":
            tone = tone_for_direction({"bullish": "BULLISH", "bearish": "BEARISH"}.get(p.bias, ""))
    return card_html("Chart patterns", body, tone)
def divergence_html(a: CategoryAnalysis) -> str:
    rsi_last = float(a.rsi.dropna().iloc[-1]) if a.rsi.notna().any() else None
    foot = (f"<div class='ti-foot'>{_kv('RSI 14', fmt_num(rsi_last, 0), 'big')}</div>")
    if not a.divergences:
        return card_html("RSI divergence", "<p class='muted'>No RSI divergence in the last 60 bars.</p>" + foot)
    body, tone = "", "neutral"
    for d in a.divergences:
        p1, p2 = d.price_points
        cls = "up" if d.direction == "bullish" else "down"
        body += (f"<div class='ti-h3 {cls}'>{html.escape(d.label)}</div>"
                 f"<div class='ti-tags'>{chip(d.status, 'outline')}</div>"
                 f"<p>Price {p1[1]:,.0f} → {p2[1]:,.0f} · RSI {d.rsi_points[0][1]:.0f} → {d.rsi_points[1][1]:.0f}</p>")
        if d.status == "forming":
            tone = "warn"
        elif tone == "neutral":
            tone = cls
    return card_html("RSI divergence", body + foot, tone)
def volatility_html(a: CategoryAnalysis) -> str:
    v, s = a.vol, a.squeeze
    body = (f"<div class='ti-split'><span class='ti-h3'>{html.escape(v.regime)} regime</span>"
            f"<span>ATR {fmt_num(v.atr, 0, '$')} · {fmt_num(v.atr_pct, 2, suffix='%')}</span></div>")
    body += "<div class='ti-rows'>"
    body += (f"<div class='ti-split'><span class='ti-label'>Expected range, {html.escape(v.horizon_label)}</span>"
             f"<span><strong>{fmt_num(v.exp_low, 0, '$')} – {fmt_num(v.exp_high, 0, '$')}</strong> "
             f"<span class='muted'>±{fmt_num(v.sigma_pct, 2, suffix='%')}</span></span></div>")
    body += (f"<div class='ti-split'><span class='ti-label'>2σ band, 20-bar</span>"
             f"<span><strong>{fmt_num(v.sigma2_dn, 0, '$')} – {fmt_num(v.sigma2_up, 0, '$')}</strong> "
             f"<span class='muted'>mean {fmt_num(v.mean20, 0, '$')}</span></span></div></div>")
    if s.score is None:
        body += "<p class='muted'>Squeeze risk: futures data unavailable.</p>"
    else:
        hot = s.score >= 60
        body += (f"<div class='ti-split'><span class='ti-label'>Squeeze risk · {html.escape(s.direction.replace('_', ' ').lower())}</span>"
                 f"<strong>{s.score}/100</strong></div>" + _bar(s.score, "warn" if hot else "") + _li(s.drivers))
    tone = "warn" if (s.score or 0) >= 60 or v.regime == "HIGH" else "neutral"
    return card_html("Volatility · positioning", body, tone)
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
            return (f"<div class='ti-head-summary'><span class='ti-kicker'>Head-agent summary</span>"
                    f"<p>{html.escape(text)}</p><span class='muted'>Director verdict: "
                    f"{report.director.trap_classification.replace('_', ' ').lower()} · report "
                    f"{html.escape(report.report_id)} · {_utc(int(report.generated_at.timestamp() * 1000))}</span></div>")
    return agent_placeholder_html(a)
def report_stats_html(report) -> str:
    ok = sum(1 for r in report.runs if r.ok)
    models = sorted({r.model for r in report.runs})
    cost = f"${report.total_cost_usd:.4f}" if report.total_cost_usd is not None else "not reported (free tier)"
    bits = [f"provider {report.provider}", f"models {', '.join(models)}", f"agents {ok}/{len(report.runs)} ok",
            f"tokens {report.total_tokens:,}", f"cost {cost}", f"wall {report.wall_time_s:.0f}s",
            f"generated {_utc(int(report.generated_at.timestamp() * 1000))}"]
    if report.failed_agents:
        bits.append("failed: " + ", ".join(report.failed_agents))
    return "<div class='ti-tags'>" + "".join(chip(b, "down" if b.startswith("failed") else "neutral-fill")
                                             for b in bits) + "</div>"


CLASS_WORDS = {"BULL_TRAP": "Bull trap", "BEAR_TRAP": "Bear trap", "NO_TRAP": "No trap", "RANGE_TRAP": "Range trap"}


def director_html(report) -> str:
    """The Director's call: the classification and why, the weighted scenarios, and the trade plan."""
    d = report.director
    if d is None:
        return card_html("Director report", "<p class='muted'>The Director did not answer in this run; the "
                                             "specialist briefs below are all there is.</p>", "warn")
    scen = "".join(
        f"<div class='ti-scen'><div class='ti-split'><span>{html.escape(x.name)}</span><strong>{x.probability:.0f}%</strong></div>"
        f"{_bar(x.probability)}<p class='muted'>{html.escape(x.path)} · invalidated if {html.escape(x.invalidation)}</p></div>"
        for x in sorted(d.scenarios, key=lambda x: -x.probability))
    tp = d.trade_plan
    plan = (f"<p>{html.escape(tp.new_entry_conditions)}</p>"
            "<div class='ti-grid4'>" + _kv("Stop loss", html.escape(tp.stop_loss)) + _kv("Sizing", html.escape(tp.position_sizing))
            + _kv("Max leverage", html.escape(tp.max_leverage))
            + _kv("Existing positions", html.escape(tp.existing_position_management)) + "</div>")
    if tp.take_profit_ladder:
        plan += "<span class='ti-label'>Take-profit ladder</span>" + _li(tp.take_profit_ladder)
    if tp.time_rules:
        plan += "<span class='ti-label'>Time rules</span>" + _li(tp.time_rules)
    tone = {"BULL_TRAP": "down", "BEAR_TRAP": "up", "RANGE_TRAP": "warn"}.get(d.trap_classification, "neutral")
    body = (f"<div class='ti-two'><div><span class='ti-label'>Trap classification</span>"
            f"<div class='ti-verdict xl'>{CLASS_WORDS.get(d.trap_classification, d.trap_classification)}</div>"
            f"<p class='ti-sum-p'>{html.escape(d.executive_summary)}</p><p class='muted'>{html.escape(d.classification_rationale)}</p></div>"
            f"<div><span class='ti-label'>Probability-weighted scenarios</span>{scen}</div></div>"
            f"<div class='ti-foot'><span class='ti-label'>Trade plan</span>{plan}</div>")
    return card_html(f"Report {report.report_id}", body, tone)


def director_lists_html(report) -> str:
    """Where the two desks disagree, where they agree, and what nobody could see."""
    d = report.director
    if d is None:
        return ""
    contra = [f"{c.metric}: the bulls read {c.bull_read}; the bears read {c.bear_read}. Stronger: {c.stronger_evidence}"
              for c in d.contradictions]
    cols = (("Contradictions", contra), ("Convergence", d.convergence), ("Blind spots", d.blind_spots))
    return "<div class='ti-three'>" + "".join(
        f"<div><span class='ti-h4'>{t}</span>" + (_li(items) or "<p class='muted'>None reported.</p>") + "</div>"
        for t, items in cols) + "</div>"


def director_desks_html(report, labels: dict, timeframes: dict) -> str:
    """The Director's paragraph for each desk."""
    d = report.director
    if d is None or not d.category_summaries:
        return "<p class='muted'>No per-desk paragraphs in this report.</p>"
    return "<div class='ti-two'>" + "".join(
        f"<div class='ti-desk-sum'><span class='ti-h3'>{html.escape(labels.get(k, k))} "
        f"<span class='muted'>{html.escape(timeframes.get(k, ''))}</span></span><p>{html.escape(v)}</p></div>"
        for k, v in d.category_summaries.items()) + "</div>"


def briefs_html(report) -> str:
    """Every specialist brief, with its conviction drawn to scale."""
    out = ""
    for b in report.briefs:
        out += (f"<div class='ti-brief'><div class='ti-split'><span class='ti-kicker'>{html.escape(b.agent_id)} · "
                f"{html.escape(b.desk)} · {html.escape(b.domain)}</span><span class='muted'>conviction {b.conviction:.0f}/10"
                f"</span></div>{_bar(b.conviction * 10, 'up' if b.desk.lower().startswith('bull') else 'down')}"
                f"<p>{html.escape(b.interpretation)}</p>"
                + (f"<p class='muted'>Traps: {html.escape('; '.join(b.traps_detected))}</p>" if b.traps_detected else "")
                + f"<p class='muted'>Key risk to thesis: {html.escape(b.key_risk_to_thesis)}</p>"
                + (f"<p class='muted'>Data gaps: {html.escape(', '.join(b.data_gaps))}</p>" if b.data_gaps else "")
                + "</div>")
    return out or "<p class='muted'>No specialist briefs in this report.</p>"


def raw_metrics_html(rows: list[tuple[str, str, str]]) -> str:
    """Section 1 of the report, live: every raw metric with the venue it came from."""
    body = "".join(f"<tr><td>{html.escape(a)}</td><td><strong>{html.escape(b)}</strong></td>"
                   f"<td class='muted'>{html.escape(c)}</td></tr>" for a, b, c in rows)
    return card_html("Section 1 · raw metric snapshot · live",
                     f"<table><thead><tr><th>Metric</th><th>Value</th><th>Source</th></tr></thead><tbody>{body}</tbody></table>")


def anchor_audit_html(rows: list[dict]) -> str:
    """Every anchor change, newest first, with what triggered it."""
    if not rows:
        return "<p class='muted'>No anchor changes recorded yet.</p>"
    body = "".join(
        f"<tr><td>{_utc(r.get('ts_ms'))}</td><td>{html.escape(str(r.get('category')))}</td>"
        f"<td>{html.escape(str(r.get('from_dir') or '—'))} → <strong>{html.escape(str(r.get('to_dir')))}</strong></td>"
        f"<td>{fmt_num(r.get('price'), 0)}</td><td class='muted'>{html.escape(', '.join(r.get('triggers') or []))}</td></tr>"
        for r in rows)
    return (f"<table><thead><tr><th>Time</th><th>Desk</th><th>Change</th><th>Price</th><th>Triggers</th></tr></thead>"
            f"<tbody>{body}</tbody></table>")
def agent_placeholder_html(a: CategoryAnalysis) -> str:
    return ("<div class='ti-head-summary'><span class='ti-kicker'>Head-agent summary</span>"
            "<p class='muted'>Run Analysis to add the Director's paragraph for this desk.</p></div>")
def trade_result_html(tm: TradeMap) -> str:
    tone = {"OPTIMAL": "up", "ELEVATED": "warn", "EXTREME": "down"}[tm.risk_label]
    body = (f"<div class='ti-h3'>{tm.classification} · {tm.direction} {tm.leverage:.0f}x from {fmt_num(tm.entry, 0, '$')}</div>"
            "<div class='ti-grid4 stress'>" + _kv("Liquidation", fmt_num(tm.liquidation_price, 0, "$"), "down")
            + _kv("Distance", f"{tm.dist_to_liq_pct:.1f}% away") + _kv("Reward : risk", fmt_num(tm.rr, 2))
            + _kv("Risk", tm.risk_label) + "</div>")
    body += (f"<p class='muted'>Stop {fmt_num(tm.stop, 0, '$')} · target {fmt_num(tm.target, 0, '$')} · "
             + ("stop protected by structure" if tm.stop_structural else "stop not protected by structure") + "</p>")
    body += ("<div class='ti-two'><div><span class='ti-label'>Structure above</span>"
             + (_li(tm.structures_above) or "<p class='muted'>none found</p>")
             + "</div><div><span class='ti-label'>Structure below</span>"
             + (_li(tm.structures_below) or "<p class='muted'>none found</p>") + "</div></div>")
    if tm.warnings:
        body += "<span class='ti-label'>Warnings</span>" + _li(tm.warnings)
    body += "<span class='ti-label'>Verdict stays anchored until</span>" + _li(tm.anchor_triggers)
    return card_html("Active trade stress-test", body, tone)
def macro_html(events: list[dict]) -> str:
    rows = [f"{e['date_utc']} · {e['title']} · {e['impact']} impact. {e['note']}" for e in events]
    return card_html("Macro calendar (curated)", _li(rows) + "<p class='muted'>Fallback list; the live calendar feed was unavailable.</p>")


def calendar_html(m: MarketSnapshot, now_ms: int, limit: int = 8) -> str:
    """High-impact USD prints this week from the live calendar; falls back to the curated list."""
    from core.config import MACRO_EVENTS
    from core.trades import fmt_when
    c = m.calendar
    if not (c.available and c.events):
        return macro_html(MACRO_EVENTS)
    hi = [e for e in c.events if e.get("impact") == "High" and e.get("country") == "USD" and e.get("time_ms")]
    upcoming = [e for e in hi if e["time_ms"] >= now_ms - 3_600_000][:limit]
    soon = c.upcoming(now_ms, 86_400_000)
    soon_titles = {e.get("title") for e in soon}
    body = ""
    if soon:
        body += (f"<p><strong>Inside 24h:</strong> {html.escape(', '.join(e['title'] for e in soon))}. "
                 "No new leveraged entries into the print.</p>")
    if upcoming:
        body += "<div class='ti-rows'>"
        for e in upcoming:
            state = ("released" if e["time_ms"] <= now_ms else
                     "inside 24h" if e.get("title") in soon_titles else "this week")
            extra = " · ".join(x for x in (f"forecast {e['forecast']}" if e.get("forecast") else "",
                                           f"previous {e['previous']}" if e.get("previous") else "") if x)
            body += (f"<div class='ti-cal'><div class='ti-split'><strong>{html.escape(str(e['title']))}</strong>"
                     f"{chip(state, 'warn' if state == 'inside 24h' else 'outline')}</div>"
                     f"<span>{html.escape(fmt_when(e['time_ms']))}</span>"
                     + (f"<span class='muted'>{html.escape(extra)}</span>" if extra else "") + "</div>")
        body += "</div>"
    else:
        body += "<p class='muted'>No further USD high-impact releases this week.</p>"
    body += (f"<p class='muted ti-note'>Source: Forex Factory weekly calendar via {html.escape(c.source)}. A release "
             "inside 24 hours fires the macro trigger.</p>")
    return card_html("Macro calendar · high-impact USD", body, "warn" if soon else "neutral")
FEED_NAMES = {"spot": "price"}   # the snapshot keeps its old name in code; it is the perp first now


def data_status_html(m: MarketSnapshot) -> str:
    """One row per feed: a square status mark, the feed, and where it came from - filled when the feed
    answered, hollow when it did not, so a dead feed is visible without relying on colour."""
    rows = []
    for name in MarketSnapshot.SOURCES:
        s = getattr(m, name)
        src = html.escape(str(s.source)) + ("" if s.available else " · unavailable")
        rows.append(f"<div class='ti-feed{'' if s.available else ' off'}'><span class='dot'></span>"
                    f"<span>{html.escape(FEED_NAMES.get(name, name))}</span><span class='src' title='{src}'>{src}</span></div>")
    return "<span class='ti-label'>Feeds</span><div class='ti-feeds'>" + "".join(rows) + "</div>"


def raw_metrics_rows(m: MarketSnapshot) -> list[tuple[str, str, str]]:
    f, o, s = m.futures, m.options, m.sentiment
    h, l, w, sc = m.hyperliquid, m.liquidations, m.whales, m.stablecoins
    return [
        ("Price", fmt_num(m.spot.last, 0, "$"), m.spot.source),
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


def _team_card(side: str, points, verdict: str, domains: list[tuple[int, str, float, float]], meta: str = "") -> str:
    """One team as the design draws it: the total, the verdict, and a bar per domain agent."""
    tone = "up" if side == "bullish" else "down"
    rows = "".join(
        f"<div class='ti-dom'><div class='ti-split'><span>{no}. {html.escape(title)}</span>"
        f"<span>{pts:.1f} / {mx:g}</span></div>{_bar(100 * pts / mx if mx else 0, tone)}</div>"
        for no, title, pts, mx in domains)
    head = f"<span class='ti-card-meta'>{html.escape(meta)}</span>" if meta else ""
    return (f"{head}<div class='ti-team-pts {tone}'>{float(points or 0):.0f}<span> / 100</span></div>"
            f"<p>{html.escape(verdict or '')}</p>{rows}")
def team_html(team) -> str:
    """One team's scorecard from a run: five domain agents, twenty points each."""
    if not team.ok:
        body = f"<p class='muted'>Did not score: {html.escape(team.error or 'unknown error')}. "
        body += "This is a failure, not a reading of zero.</p>"
        return card_html(f"{team.side.title()} team", body, "warn")
    meta = (team.provider or "n/a") + (f" · {team.latency_s:.1f}s" if team.latency_s else "")
    body = _team_card(team.side, team.points, team.verdict,
                      [(d.agent_no, d.title, d.points, d.max_points) for d in team.domains], meta)
    if team.missing:
        body += f"<p class='muted'>{len(team.missing)} checklist items unanswered and scored 0.</p>"
    return card_html(f"{team.side.title()} team", body, "up" if team.side == "bullish" else "down").replace("<h4>", "<h4 class='big'>", 1)


def payload_team_html(side: str, anchor: dict) -> str:
    """A team rebuilt from the pinned summary, so the scorecards stay on screen for the whole candle."""
    from core.agents.rubric import DOMAINS
    pts = (anchor.get("domains") or {}).get(side)
    if not pts:
        return card_html(f"{side.title()} team", "<p class='muted'>This team did not score in the pinned run.</p>", "warn")
    domains = [(d.agent_no, d.title, float(pts.get(d.key) or 0), d.max_points) for d in DOMAINS]
    body = _team_card(side, anchor.get(f"{'bull' if side == 'bullish' else 'bear'}_points"),
                      str(anchor.get(f"{'bull' if side == 'bullish' else 'bear'}_verdict") or ""), domains, "pinned run")
    return card_html(f"{side.title()} team", body, "up" if side == "bullish" else "down").replace("<h4>", "<h4 class='big'>", 1)
def trap_audit_html(choice, probabilities: dict, consulted: bool, override=None, error=None) -> str:
    """The trap agent's distribution, most likely first, and whether the Head had to consult it."""
    if error:
        return card_html("Trap audit", f"<p class='muted'>No answer: {html.escape(error)}. "
                                       "No trap override was applied.</p>", "warn")
    label = TRAP_LABELS.get(choice or "", choice or "—")
    rows = ""
    for i, (k, v) in enumerate(sorted((probabilities or {}).items(), key=lambda kv: -kv[1])):
        rows += (f"<div class='ti-prob'><span class='l'>{html.escape(TRAP_LABELS.get(k, k))}</span>"
                 f"{_bar(v * 100, '' if i == 0 else 'faint')}<span class='v'>{v:.0%}</span></div>")
    body = f"<div class='ti-h3 lg'>{html.escape(label)}</div>{rows}"
    body += ("<p class='muted'>Consulted: neither team cleared the bar on its own.</p>" if consulted else
             "<p class='muted'>Not required: consulted only when both teams clear 60 or neither leads by ten.</p>")
    if override:
        body += f"<p><span class='ti-chip warn'>override</span> {html.escape(override)}</p>"
    return card_html("Trap audit", body, "warn" if choice in ("BULL_TRAP", "BEAR_TRAP") else "neutral")


def war_room_html(res) -> str:
    """The trap audit from a run held in this session."""
    t = res.trap
    return trap_audit_html(t.choice, t.probabilities, res.consulted, res.override,
                           None if t.ok else (t.error or "unknown error"))
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
    from core.agents.handshake import lean
    bias = str(anchor.get("bias", "—"))
    # war-room biases are BULL / BEAR / BULL TRAP RISK, not BULLISH / BEARISH: looked up directly they all
    # came out neutral. A trap-risk call points against its own word, as everywhere else it is read.
    word = {"up": "BULLISH", "down": "BEARISH"}.get(lean(bias) or "", "")
    tone = tone_for_direction(word)
    glyph = DIR_GLYPH.get(word, "◆")
    bull, bear = float(anchor.get("bull_points") or 0), float(anchor.get("bear_points") or 0)
    trap = TRAP_LABELS.get(anchor.get("trap") or "", anchor.get("trap") or "—")
    published = (html.escape(str(anchor.get("published_at_utc", ""))) +
                 (f" · {fmt_num(anchor.get('price'), 0, '$')}" if anchor.get("price") else ""))
    body = (f"<div class='ti-split ti-sum-head'><span class='ti-tag-solid'>Pinned · {_remaining(mins)} left</span>"
            f"<span class='muted'>{published}</span></div>")
    body += (f"<div class='ti-sum'><div><div class='ti-dir xl {tone}'>{glyph} {html.escape(bias)}</div>"
             f"<p>Confidence {float(anchor.get('confidence') or 0):.0%} · margin "
             f"{float(anchor.get('margin') or 0):+.0f} points · trap read: {html.escape(trap)}</p></div>"
             f"<div class='ti-teams'>"
             f"<div class='ti-team-row'><span>Bullish</span>{_bar(bull, 'up tick60')}<span class='n'>{bull:.0f}</span></div>"
             f"<div class='ti-team-row'><span>Bearish</span>{_bar(bear, 'down tick60')}<span class='n'>{bear:.0f}</span></div>"
             f"<span class='muted'>Out of 100 per team. The tick marks 60, where the trap agent is consulted "
             f"if both teams clear it.</span></div></div>")
    body += (f"<p class='muted ti-note'>Anchored {html.escape(str(anchor.get('published_at_utc', '')))} "
             + (f"at {fmt_num(anchor.get('price'), 0, '$')} " if anchor.get("price") else "")
             + f"· pinned for {_remaining(mins)} more, until the {p.window_label} candle closes. "
             + ("Published by this run." if anchored_now else
                "Pinned to the candle and immutable for that window. Re-runs during this candle append "
                "side notes; they never rewrite this summary.") + "</p>")
    if anchor.get("degraded"):
        body += "<p><span class='ti-chip warn'>degraded</span> part of the checklist did not score.</p>"
    body += handshake_html(anchor)
    return card_html(f"{p.window_title} anchored summary", body, tone)


HANDSHAKE_TONE = {"agrees": "accent", "opposes": "warn", "neutral": "outline", "silent": "outline"}


def handshake_html(anchor: dict) -> str:
    """What the other three desks said when this one asked, and what it did to the conviction."""
    hs = anchor.get("handshake")
    if not hs:
        return ""
    rows = ""
    for a in hs.get("answers") or []:
        conf = a.get("confidence")
        rows += (f"<tr><td><strong>{html.escape(str(a.get('label', '')))}</strong></td>"
                 f"<td class='muted'>{html.escape(str(a.get('relation', '')))} desk</td>"
                 f"<td>{chip(str(a.get('answer', '')), HANDSHAKE_TONE.get(a.get('answer'), 'outline'))}</td>"
                 f"<td class='muted'>{html.escape(str(a.get('bias') or '—'))}"
                 + (f" · {float(conf):.0%}" if conf is not None else "")
                 + (f" · {html.escape(str(a['note']))}" if a.get("note") else "") + "</td></tr>")
    before, after = anchor.get("confidence_before_handshake"), anchor.get("confidence")
    line = ""
    if before is not None and after is not None:
        line = (f"<p>Conviction {float(before):.0%} from the war room, {float(after):.0%} after the handshake "
                f"(x{float(hs.get('factor', 1.0)):.2f}).</p>")
    return ("<span class='ti-label'>Cross-desk handshake</span>" + line
            + f"<table class='ti-cascade'><tbody>{rows}</tbody></table>"
            "<p class='muted ti-note'>Each desk answers with the verdict it has anchored for its current candle. "
            "A higher desk agreeing adds 10% to conviction and opposing takes 20% off; a lower desk opposing "
            "takes 5%. The direction itself is the war room's.</p>")
def side_notes_html(rows: list[dict]) -> str:
    """Interim invalidations, appended alongside the anchor and never replacing it."""
    if not rows:
        return card_html("Interim side notes · 0", "<p class='muted'>Nothing substantial has changed since the "
                                                   "summary was anchored.</p>", "neutral")
    body = ""
    for r in rows:
        when = _utc(r.get("ts_ms")) if r.get("ts_ms") else ""
        body += (f"<div class='ti-note-item'><div class='ti-split'>{chip(str(r['kind']).replace('_', ' ').lower(), 'accent')}"
                 f"<span class='muted'>{html.escape(when)}</span></div>"
                 f"<p><strong>{html.escape(str(r['headline']))}</strong>"
                 + (f" <span class='muted'>at {fmt_num(r['price'], 0, '$')}</span>" if r.get("price") else "")
                 + "</p>" + (f"<p class='muted'>{html.escape(str(r['detail']))}</p>" if r.get("detail") else "")
                 + "</div>")
    return card_html(f"Interim side notes · {len(rows)}", body, "warn")
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
    oi_windows = list(by_tf)
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
        # each desk measures open interest over its own timeframes, so the column follows them
        oi_windows = list(oi) or oi_windows
        oi_cell = " / ".join("—" if oi.get(k) is None else f"{oi[k]:+.1f}%" for k in oi_windows)
        rows += (f"<tr><td>{html.escape(tf)}</td><td>{pool_cell('buyside')}</td><td>{pool_cell('sellside')}</td>"
                 f"<td>{html.escape(str(cvd.get('state', '—')))}"
                 + (f"<br><span class='muted'>{cvd['velocity_per_bar']:+,.1f} BTC/bar"
                    + (f", {'accelerating' if (cvd.get('acceleration') or 0) * cvd['velocity_per_bar'] > 0 else 'slowing'}"
                       if cvd.get("acceleration") is not None and cvd['velocity_per_bar'] else "")
                    + "</span>" if cvd.get("velocity_per_bar") is not None else "")
                 + ("" if cvd.get("units") == "BTC" else " <span class='muted'>(proxy)</span>") + "</td>"
                 f"<td>{'—' if not vp.get('available') else _dash(vp.get('point_of_control'), 0, '$')}</td>"
                 f"<td>{oi_cell}</td></tr>")
    body = ("<table><thead><tr><th>TF</th><th>Buyside pool</th><th>Sellside pool</th><th>Delta</th>"
            f"<th>POC</th><th>OI {'/'.join(oi_windows)}</th></tr></thead><tbody>{rows}</tbody></table>")
    methods = {str(((d or {}).get("cumulative_delta") or {}).get("units", "")) for d in by_tf.values()}
    real = "BTC" in methods
    body += ("<p class='muted'>"
             + ("Delta is real taker volume - market buys minus market sells - on the Gate.io BTC_USDT perpetual, "
                "with its velocity: the average per bar over the last five, and whether that is speeding up. "
                if real else "")
             + ("Rows marked proxy have no taker data for that timeframe and use the candle close-position "
                "proxy, a lean rather than a measurement. " if methods - {"BTC"} else "")
             + "A pool is swept when price wicks through and closes back inside, and broken when it closes "
               "beyond.</p>")
    return card_html("Liquidity & order flow", body)


# ---------- war-room accuracy report ----------
def accuracy_report_html(rep) -> str:
    """The head agent's grade of both teams, and the protocols to correct when it is not optimal."""
    def pct(v):
        return "—" if v is None else f"{v:.0%}"

    def brier(v):
        return "—" if v is None else f"{v:.2f}"

    if rep.status == "collecting":
        body = (f"<div class='ti-tags'>{chip('Collecting', 'outline')}</div><p>{html.escape(rep.headline)}</p>"
                + _bar(min(100, rep.n * 10)) +
                "<p class='muted'>Each pinned summary is scored once its candle closes, against the price at "
                "that close. The report grades once ten candles have closed.</p>")
        return card_html("Accuracy report", body)

    label = "Healthy" if rep.status == "healthy" else "Calibration needed"
    body = (f"<div class='ti-tags'>{chip(label, 'outline' if rep.status == 'healthy' else 'warn')}</div>"
            f"<p>{html.escape(rep.headline)}</p><div class='ti-grid2 acc'>"
            + _kv("Direction right", pct(rep.hit_rate)) + _kv("Trap call right", pct(rep.trap_hit_rate))
            + _kv("Bullish Brier", brier(rep.bull_brier)) + _kv("Bearish Brier", brier(rep.bear_brier)) + "</div>")
    body += ("<p class='muted'>Team scores are Brier scores: 0 is perfect, 0.25 is a coin flip, higher is "
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


def agent_matrix_html(deck, title: str = "Institutional quantitative matrix") -> str:
    """What each of the five sub-agents read, straight from the state the judges were given - one framed
    card per agent, as the design lays the matrix out."""
    cards = ""
    for a in deck:
        rows = "".join(f"<div class='ti-split ti-mx-row'><span class='muted'>{html.escape(label)}</span>"
                       f"<span>{_deck_value(value, kind)}</span></div>" for label, value, kind in a.rows)
        cards += (f"<div class='ti-mx blueprint'>{CORNERS}<div class='ti-split'><span class='ti-kicker'>"
                  f"{a.agent_no}. {html.escape(a.title)}</span><span class='muted'>{html.escape(a.timeframe)}</span></div>"
                  f"{rows}" + (f"<p class='muted'>{html.escape(a.note)}</p>" if a.note else "") + "</div>")
    return (f"<div class='ti-mx-grid'>{cards}</div>"
            f"<p class='muted ti-note'>{html.escape(title)}: the exact readings handed to the ten domain agents, "
            "so the scorecard cannot disagree with what is shown here. A dash means no feed, not zero.</p>")


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


HEAD_WORDS = {"ENGINEERED_TRAP": "engineered", "GENUINE_MOVE": "genuine move", "UNCLEAR": "unclear"}


def _detail(row) -> dict:
    """The judgement stored with a trap (domain points, Head confidence, the desk's read). Older rows have
    none, and a blob that will not parse is treated the same way - the card simply omits that part."""
    raw = row.get("detail") if isinstance(row, dict) else None
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def trap_card_html(row: dict, now_ms: int, category_label: str = "", price_now: float | None = None) -> str:
    """One anchored trap: what it is, the level it is built on, what settles it, and the judgement behind it."""
    side_key, side = _trap_side(row)
    active = row.get("status") == "active"
    tone = "warn" if active else "neutral"
    detail = _detail(row)
    score = row.get("score")
    head = HEAD_WORDS.get(str(row.get("head_call") or ""), str(row.get("head_call") or "").replace("_", " ").lower())
    if detail.get("head_confidence") is not None and head:
        head += f", {float(detail['head_confidence']):.0%}"
    # a dash means the score was never recorded, not that the sub-agents scored it zero
    meta = ((f"{html.escape(category_label)} desk · " if category_label else "")
            + f"declared {_trap_age(row.get('declared_ms'), now_ms)} ago"
            + f" · sub-agents {fmt_num(score, 0) if score is not None else '—'}/100"
            + (f" · Head: {html.escape(head)}" if head else ""))
    body = (f"<span class='ti-card-meta'>#{html.escape(str(row.get('id') or '?'))}</span>"
            f"<div class='ti-h3'>{html.escape(str(row.get('timeframe') or '—'))} at {fmt_num(row.get('level'), 0, '$')}</div>"
            f"<p class='muted'>{meta}</p>")
    inval, plays = row.get("invalidation"), row.get("plays_out")
    if side_key == "bull_trap":
        lo = ("Pays off back at", plays)
        hi = ("Invalidated above", inval)
    elif side_key == "bear_trap":
        lo = ("Invalidated below", inval)
        hi = ("Pays off back at", plays)
    else:
        lo, hi = ("Invalidated at", inval), ("Pays off back at", plays)
    body += (f"<div class='ti-range'><div class='ti-split'>{_kv(lo[0], fmt_num(lo[1], 0, '$'))}"
             f"<div class='ti-kv right'><span class='ti-label'>{hi[0]}</span>"
             f"<span class='ti-kv-v'>{fmt_num(hi[1], 0, '$')}</span></div></div>")
    if side_key and price_now is not None and lo[1] is not None and hi[1] is not None and hi[1] != lo[1]:
        pct = max(0, min(100, round(100 * (price_now - lo[1]) / (hi[1] - lo[1]))))
        body += (f"<div class='ti-track'><div class='ti-track-now' style='left:{pct}%'></div></div>"
                 f"<span class='muted'>Now {fmt_num(price_now, 0, '$')}</span>")
    body += "</div>"
    evidence = _evidence_list(row.get("evidence"))
    if evidence:
        body += "<span class='ti-label'>Evidence</span>" + _li(evidence)
    domains = detail.get("domains") or []
    if domains:
        body += "<div class='ti-mini-doms'>" + "".join(
            f"<div title='{html.escape(str(d.get('title', '')))}'>{_bar(5 * float(d.get('points') or 0))}"
            f"<span class='muted'>{html.escape(str(d.get('title', '')))}</span>"
            f"<span>{float(d.get('points') or 0):.0f}/20</span></div>" for d in domains) + "</div>"
    if detail.get("desk_reads"):
        reasons = "; ".join(str(r) for r in detail.get("desk_reasons") or [])
        body += (f"<p><strong>Desk says:</strong> {html.escape(str(detail['desk_reads']))}"
                 + (f" - {html.escape(reasons)}" if reasons else "") + "</p>")
    if not active:
        body += (f"<p class='muted'>{html.escape(str(row.get('status') or 'unknown').replace('_', ' '))} at "
                 f"{fmt_num(row.get('resolved_price'), 0, '$')}</p>")
    return card_html(f"Anchored {side}", body, tone)


def sweep_plan_html(plan: list[tuple[str, str]], labels: dict) -> str:
    """Which desk scans which timeframes, so each raid is judged once."""
    by_desk: dict[str, list[str]] = {}
    for cat, tf in plan:
        by_desk.setdefault(cat, []).append(tf)
    lines = [f"<strong>{html.escape(' · '.join(tfs))}</strong> → {html.escape(labels.get(cat, cat))}"
             for cat, tfs in by_desk.items()]
    for cat, label in labels.items():
        if cat not in by_desk:
            lines.append(f"<strong>none</strong> → {html.escape(label)} reads the same candles as a desk above")
    return "<span class='ti-label'>Sweep plan</span>" + "".join(f"<p class='ti-dash'>{x}</p>" for x in lines)


def trap_console_html(rows: list[dict], candidates: list, now_ms: int, labels: dict | None = None,
                      settled: list[dict] | None = None, swept_ms: int | None = None,
                      plan: list[tuple[str, str]] | None = None) -> str:
    """The watchdog console: how many traps are anchored, watched and settled, and the sweep plan.

    `rows` are the anchored traps and `settled` the recently closed ones, queried separately so newer
    settled traps cannot push the live ones out of sight."""
    labels = labels or {}
    settled = settled or []
    active = [r for r in rows if r.get("status") == "active"]
    counts = "".join(f"<div class='ti-count'><span class='n'>{n}</span><span class='ti-label'>{label}</span></div>"
                     for n, label in ((len(active), "Anchored"), (len(candidates), "On watch"),
                                      (len(settled), "Recently settled")))
    body = (f"<div class='ti-split'><span class='muted'>{len(active)} anchored · {len(candidates)} candidate(s) on "
            f"watch · {len(settled)} recently settled</span><span class='muted'>Nothing expires on a clock</span></div>"
            f"<div class='ti-counts'>{counts}</div>")
    if not active:
        body += ("<p class='muted'>No trap is anchored. A trap is declared only when its five sub-agents "
                 "score 60 or more, the TRAP Head calls it engineered, and the category desk does not read "
                 "the move as an authentic break.</p>")
    if plan:
        body += sweep_plan_html(plan, labels)
    return card_html("TRAP intelligence · watchdog", body, "warn" if active else "neutral")


def trap_watch_html(candidates: list, now_ms: int, swept_ms: int | None = None, judged: list | None = None) -> str:
    """What the deterministic scan is watching but has not declared, dated so an old scan reads as old.

    Candidates live in session state and survive every auto-refresh, so an undated one read as current
    long after the scan that found it."""
    age = _trap_age(swept_ms, now_ms) if swept_ms else None
    title = f"On watch · deterministic scan {age} ago" if age else "On watch"
    if not candidates:
        body = "<p class='muted'>Nothing on watch. Sweep to scan every timeframe.</p>"
    else:
        body = "<div class='ti-rows'>"
        for c in candidates[:8]:
            _key, label = _trap_side(c)
            factors = f"{c.strength} factor{'s' if c.strength != 1 else ''}"
            body += (f"<div class='ti-watch'><span>{html.escape(label)}</span><strong>{html.escape(str(c.timeframe))}</strong>"
                     f"<span>{fmt_num(c.level, 0, '$')}</span><span class='muted'>{factors}: "
                     f"{html.escape('; '.join(c.evidence[:2]))}</span></div>")
        body += "</div>"
        if swept_ms and (now_ms - swept_ms) > 30 * 60_000:
            body += ("<p class='muted'>That scan is over half an hour old. Sweep again before trading "
                     "from it - these levels were measured against a price that has since moved.</p>")
    for v in judged or []:
        body += (f"<p class='muted'>Judged this sweep: {html.escape(v['label'])} · {v['score']:.0f}/100 · "
                 f"{html.escape(v['status'])}" + (f" - {html.escape(v['why'])}" if v.get("why") else "") + "</p>")
    return card_html(title, body)


def trap_settled_html(settled: list[dict], now_ms: int, labels: dict | None = None) -> str:
    """Recently settled traps: how each ended, at what price, and on whose desk."""
    labels = labels or {}
    if not settled:
        return card_html("Recently settled", "<p class='muted'>Nothing settled yet.</p>")
    body = "<div class='ti-rows'>"
    for r in settled:
        _key, label = _trap_side(r)
        desk = labels.get(r.get("category"), r.get("category") or "")
        status = str(r.get("status") or "").replace("_", " ")
        body += (f"<div class='ti-watch'><span>{html.escape(label)}</span><strong>{html.escape(str(r.get('timeframe') or '?'))}</strong>"
                 f"<span><strong>{html.escape(status)}</strong> at {fmt_num(r.get('resolved_price'), 0, '$')}"
                 + (f" <span class='muted'>· {html.escape(desk)}</span>" if desk else "")
                 + f"</span><span class='muted'>{_trap_age(r.get('resolved_ms') or r.get('declared_ms'), now_ms)} ago</span></div>")
    return card_html("Recently settled", body + "</div>")


def trap_warning_html(rows: list[dict], now_ms: int) -> str | None:
    """The outbound half of the handshake, on every desk that reads the trap's timeframe: a dark band."""
    active = [r for r in rows if r.get("status") == "active"]
    if not active:
        return None
    lines = ""
    for r in active[:3]:
        side_key, side = _trap_side(r)
        word = {"bull_trap": "above", "bear_trap": "below"}.get(side_key, "at")
        lines += (f"<p><strong>{html.escape(side)}</strong> · {html.escape(str(r.get('timeframe') or '—'))} at "
                  f"{fmt_num(r.get('level'), 0, '$')} · invalidated {word} {fmt_num(r.get('invalidation'), 0, '$')} · "
                  f"declared {_trap_age(r.get('declared_ms'), now_ms)} ago</p>")
    if len(active) > 3:
        lines += f"<p>and {len(active) - 3} more on this desk's timeframes.</p>"
    return (f"<div class='ti-band'><span class='ti-band-title'>TRAP warning</span><div class='ti-band-lines'>{lines}"
            "<p class='ti-band-note'>From the TRAP desk. It stays here until price settles it.</p></div></div>")


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


def _ladder_html(points: list[tuple[str, float, str]], risk: tuple[float, float] | None,
                 reward: tuple[float, float] | None) -> str:
    """The position on one line: stop, entry, now and the targets placed to scale, with the risk and the
    reward spans shaded - so a stop that is wider than the first target is visible at a glance."""
    vals = [p for _l, p, _c in points if p is not None]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0

    def pos(p):
        return round((p - lo) / span * 1000) / 10

    out = "<div class='ti-ladder'><div class='ti-ladder-track'></div>"
    for cls, seg in (("risk", risk), ("reward", reward)):
        if seg and None not in seg:
            a, b = sorted((pos(seg[0]), pos(seg[1])))
            out += f"<div class='ti-ladder-seg {cls}' style='left:{a}%;width:{b - a}%'></div>"
    for i, (label, p, cls) in enumerate(sorted((x for x in points if x[1] is not None), key=lambda x: x[1])):
        where = "top" if i % 2 == 0 else "bottom"
        out += (f"<div class='ti-ladder-mark {cls} {where}' style='left:{pos(p)}%'><div class='tick'></div>"
                f"<div class='lab'><span>{html.escape(label)}</span><span>{fmt_num(p, 0)}</span></div></div>")
    return out + "</div>"


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
    targets = plan.get("targets") or []

    head = (f"<div class='ti-trade-head'><span class='ti-chip {'up' if side == 'LONG' else 'down'}'>{html.escape(side)}</span>"
            f"<span class='ti-h3'>{html.escape(asset)} from {fmt_num(entry, 0, '$')}</span>"
            f"<span class='muted'>opened {_when(trade.get('opened_ms'), now_ms, future=False)}</span></div>")
    body = head

    # --- live block: the only part a Refresh changes ---
    if closed:
        body += (f"<p class='muted'>Closed {_when(trade.get('closed_ms'), now_ms, future=False)} at "
                 f"{fmt_num(trade.get('closed_price'), 0, '$')} - "
                 f"{html.escape(str(trade.get('closed_reason') or 'no reason recorded'))}.</p>")
    elif prog is not None:
        reached = f"{len(prog.hit)} / {len(targets)} · " + (", ".join(prog.hit) if prog.hit else "none yet")
        pnl_tone = "up" if prog.pnl_pct >= 0 else "down"
        body += ("<div class='ti-grid4 live'>" + _kv("Now", fmt_num(prog.price, 0, "$"))
                 + _kv("PnL", f"{prog.pnl_pct:+.2f}%", pnl_tone) + _kv("In R", f"{prog.pnl_r:+.2f}R", pnl_tone)
                 + _kv("Targets reached", html.escape(reached)) + "</div>")
        # the price is recomputed on every rerun, so it is stamped with now; `refreshed_ms` is reported
        # separately, and only when a manual refresh really happened
        body += ("<p class='muted'>" + ("<strong>the stop has been reached</strong> · " if prog.stopped else "")
                 + f"as of {_when(now_ms, now_ms, future=False)}"
                 + (f", last manual refresh {_when(trade.get('refreshed_ms'), now_ms, future=False)}"
                    if trade.get("refreshed_ms") else "") + "</p>")
    elif price_now:
        body += (f"<p class='muted'>Price is {fmt_num(price_now, 0, '$')}, but this position's live "
                 "numbers could not be calculated - see the warning above. The plan below is unaffected."
                 "</p>")
    else:
        body += "<p class='muted'>No live price this refresh, so the numbers below are the plan only.</p>"

    # --- the ladder ---
    given = trade.get("given_stop")
    now_px = prog.price if prog is not None else (trade.get("closed_price") if closed else None)
    pts = [("Stop", stop, "down")]
    if given is not None and given != stop:
        pts.append(("Given stop", given, "muted"))
    pts += [("Entry", entry, "ink"), ("Now", now_px, "now")]
    pts += [(str(t.get("name")), t.get("price"), "up") for t in targets]
    far = targets[-1].get("price") if targets else None
    body += _ladder_html(pts, (stop, entry) if stop is not None and entry is not None else None,
                         (entry, far) if far is not None and entry is not None else None)

    # --- verdict beside the stop ---
    verdict = str(plan.get("stop_verdict") or trade.get("stop_verdict") or "")
    left = "<span class='ti-label'>Desk verdict</span>"
    if plan.get("ok"):
        conf = plan.get("head_confidence")
        call = str(plan.get("head_call") or "—").replace("_", " ").title()
        left += (f"<div class='ti-verdict'>{html.escape(call)}</div>"
                 f"<p class='muted'>Head of the Active Trade desk: {html.escape(call)}"
                 + (f" ({float(conf):.0%} confidence)" if conf else "")
                 + f" · score {fmt_num(plan.get('score'), 0)}/100 · "
                 f"{fmt_num(plan.get('probability'), 0, suffix='%')} confluence estimate</p>")
        fors = {str(d.get("title")): d.get("points") for d in plan.get("for_domains") or []}
        againsts = {str(d.get("title")): d.get("points") for d in plan.get("against_domains") or []}
        left += (f"<div class='ti-split ti-fa'><span class='up'>For · {fmt_num(plan.get('for_points'), 0)}</span>"
                 f"<span class='down'>Against · {fmt_num(plan.get('against_points'), 0)}</span></div>")
        for title in list(dict.fromkeys(list(fors) + list(againsts))):
            f_, a_ = fors.get(title), againsts.get(title)
            left += (f"<div class='ti-twin'><span>{html.escape(title)}</span>"
                     f"<div class='ti-twin-bars'><div class='f'><div style='width:{5 * float(f_ or 0):.0f}%'></div></div>"
                     f"<div class='a'><div style='width:{5 * float(a_ or 0):.0f}%'></div></div></div>"
                     f"<span class='muted'>{fmt_num(f_, 0)}/20 · {fmt_num(a_, 0)}/20</span></div>")
    else:
        left += "<p class='muted'>unscored: the sub-agents did not answer</p>"
    left += ("<p class='muted ti-note'>The score and the estimate come from the agents' own probabilities and the "
             "desks' alignment. They are a measure of confluence, not a backtested win rate - nothing "
             "here counts how often a setup like this has worked.</p>")
    right = (f"<span class='ti-label'>Stop verification</span>"
             f"<div class='ti-stopv {STOP_TONE.get(verdict, 'neutral')}'>{html.escape(verdict)}</div>"
             f"<div class='ti-grid2'>{_kv('Given', fmt_num(given, 0, '$') if given is not None else 'none given')}"
             f"{_kv('Desk stop', fmt_num(stop, 0, '$'))}</div>"
             + _li([str(r) for r in (plan.get("stop_reasons") or [])]))
    if given is not None and verdict not in ("sound", "supplied"):
        right += "<p class='muted'>The stop you gave is corrected here; the desk trades the one above.</p>"
    body += f"<div class='ti-two'><div>{left}</div><div>{right}</div></div>"

    # --- the timed targets ---
    if targets:
        rows = "".join(
            f"<tr><td><strong>{html.escape(str(t.get('name')))}</strong></td>"
            f"<td>{fmt_num(t.get('price'), 0, '$')}</td>"
            f"<td class='muted'>{html.escape(str(t.get('basis') or ''))}</td>"
            f"<td>{html.escape(_when(t.get('due_ms'), now_ms))}<br><span class='muted'>"
            f"{html.escape(str(t.get('due_from') or ''))}</span></td>"
            f"<td>{fmt_num(t.get('reward_r'), 2, suffix='R')}</td></tr>" for t in targets)
        body += ("<table class='ti-targets'><thead><tr><th>Target</th><th>Price</th><th>Basis</th><th>Due</th>"
                 f"<th>Reward</th></tr></thead><tbody>{rows}</tbody></table>")
        try:
            if float(targets[0].get("reward_r") or 0) < 1.0:
                body += ("<p><span class='ti-chip down'>reward below risk</span> The first target is nearer "
                         "than the stop, so this position risks more than it makes at TP1.</p>")
        except (TypeError, ValueError):
            pass

    # --- the TRAP handshake ---
    traps = plan.get("traps") or []
    if traps:
        body += "<div class='ti-handshake'><span class='ti-label'>TRAP handshake</span>" + "".join(
            f"<p>An anchored {html.escape(str(t.get('side', '')).replace('_', ' '))} on the "
            f"{html.escape(str(t.get('timeframe')))} sits {fmt_num(t.get('distance'), 0, '$')} from this "
            f"trade's {html.escape(str(t.get('near')))} at {fmt_num(t.get('level'), 0, '$')}. Expect the level "
            "to be defended.</p>" for t in traps) + "</div>"
    elif plan.get("ok"):
        body += "<p class='muted'>TRAP handshake: no anchored trap sits on this trade's levels.</p>"

    # --- guardrail, alignment, schedule ---
    pb = plan.get("pullback") or {}
    col1 = "<span class='ti-label'>Pullback guardrail</span>"
    if pb.get("available"):
        levels = pb.get("levels") or {}
        col1 += ("<div class='ti-grid3'>" + _kv("Ordinary", fmt_num(levels.get("ordinary"), 0, "$"))
                 + _kv("Edge", fmt_num(levels.get("edge"), 0, "$"))
                 + _kv("Breaking", fmt_num(levels.get("breaking"), 0, "$"), "down") + "</div>"
                 f"<p class='muted'>Over the last {pb.get('horizon_bars')} {html.escape(str(pb.get('timeframe')))} "
                 f"candles this market has normally gone {fmt_num(pb.get('ordinary'), 0, '$')} against a position "
                 f"like this before continuing, and {fmt_num(pb.get('edge'), 0, '$')} in four cases out of five. "
                 f"So a move to {fmt_num(levels.get('ordinary'), 0, '$')} is ordinary and "
                 f"{fmt_num(levels.get('edge'), 0, '$')} is still normal - do not close there. "
                 f"Past {fmt_num(levels.get('breaking'), 0, '$')} this is no longer a pullback: that is the 95th "
                 "percentile, and the move you were expecting is not happening.</p>")
    elif pb:
        col1 += f"<p class='muted'>No pullback guardrail: {html.escape(str(pb.get('note') or 'not measurable'))}.</p>"
    else:
        col1 += "<p class='muted'>Not measured for this position.</p>"
    alignment = plan.get("alignment") or []
    col2 = ("<span class='ti-label'>Desk alignment</span>"
            + (f"<p class='muted'>{html.escape(str(plan.get('aligned') or ''))}</p>" if plan.get("aligned") else ""))
    for a in alignment:
        v = str(a.get("verdict"))
        word = ALIGN_WORD.get(v, v)
        cls = {"supports": "up", "opposes": "down"}.get(v, "muted")
        col2 += (f"<div class='ti-split ti-row-rule'><span><strong>{html.escape(str(a.get('label')))}</strong> "
                 f"{html.escape(str(a.get('bias') or 'no verdict'))}</span><span class='{cls}'>{html.escape(word)}</span></div>"
                 + (f"<p class='muted'>{html.escape(str(a.get('note')))}</p>" if a.get("note") else ""))
    schedule = plan.get("schedule") or []
    col3 = "<span class='ti-label'>Monitoring schedule</span>" + "".join(
        f"<div class='ti-row-rule'><strong>{html.escape(str(c.get('timeframe') or ''))}</strong> "
        f"{html.escape(_when(c.get('at_ms'), now_ms))}<br><span class='muted'>{html.escape(str(c.get('reason') or ''))}"
        "</span></div>" for c in schedule)
    body += f"<div class='ti-three'><div>{col1}</div><div>{col2}</div><div>{col3}</div></div>"

    body += _li([str(n) for n in (plan.get("notes") or [])])
    tone = "neutral" if closed else ("down" if (prog is not None and prog.stopped) or traps else "warn")
    title = f"Position #{trade.get('id') or '?'}" + ("" if not closed else " (closed)")
    return card_html(title, body, tone)


def closed_trade_html(trade: dict, now_ms: int) -> str:
    """A closed position as one line: what it was, how it ended, and what it made."""
    side = str(trade.get("side") or "")
    entry, px = trade.get("entry"), trade.get("closed_price")
    result = ""
    try:
        sign = 1 if side == "LONG" else -1
        risk = abs(float(entry) - float(trade.get("stop"))) or 1.0
        pnl = (float(px) - float(entry)) / float(entry) * 100 * sign
        result = f"{pnl:+.2f}% · {(float(px) - float(entry)) * sign / risk:+.2f}R"
    except (TypeError, ValueError, ZeroDivisionError):
        result = "no closing price recorded"
    return (f"<div class='ti-split ti-row-rule'><strong>#{html.escape(str(trade.get('id') or '?'))} {html.escape(side)} "
            f"{html.escape(str(trade.get('asset') or ''))} from {fmt_num(entry, 0, '$')}</strong>"
            f"<span class='muted'>Closed {_when(trade.get('closed_ms'), now_ms, future=False)} at {fmt_num(px, 0, '$')} · "
            f"{html.escape(str(trade.get('closed_reason') or 'no reason recorded'))}</span><span>{result}</span></div>")


def channels_html(channels) -> str:
    """Every alert channel, configured or not - an unconfigured one is named, never hidden."""
    rows = "".join(
        f"<div class='ti-feed{'' if c.configured else ' off'} ti-chan'><span class='dot'></span><span>{html.escape(c.name)}</span>"
        f"<span class='src'>{'configured' if c.configured else html.escape(c.reason or 'not configured')}</span></div>"
        for c in channels)
    return f"<div class='ti-feeds'>{rows}</div>"
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


def news_topics(news) -> list[str]:
    """The keyword topics present in the feed, most common first, for the filter above the list."""
    counts: dict[str, int] = {}
    for h in getattr(news, "headlines", []) or []:
        t = str(h.get("topic") or "")
        if t:
            counts[t] = counts.get(t, 0) + 1
    return sorted(counts, key=lambda t: -counts[t])


def news_html(news, now_ms: int, limit: int = 12, topic: str | None = None) -> str:
    """Headlines, newest first, with the keyword tags shown as tags.

    The topic and impact labels are keyword matches, not judgments - a headline can be mislabelled by a
    word used in passing - so the card says so rather than letting "high impact" read as a verdict."""
    if not getattr(news, "available", False):
        return card_html("News", f"<p class='muted'>No headline feed: "
                                 f"{html.escape(str(getattr(news, 'error', '') or 'unavailable'))}.</p>",
                         "neutral")
    headlines = list(getattr(news, "headlines", []) or [])
    high = [h for h in headlines if h.get("impact") == "high"]
    shown = [h for h in headlines if not topic or h.get("topic") == topic]
    body = (f"<p><strong>{len(headlines)} headlines</strong> from "
            f"{html.escape(str(getattr(news, 'source', '')))}"
            + (f" · <span class='ti-chip down'>{len(high)} high-impact</span>" if high else "")
            + (f" · showing {html.escape(topic)}" if topic else "") + "</p>")
    if getattr(news, "error", None):
        body += (f"<p class='muted'>Partial feed: {html.escape(str(news.error))[:160]} - the count above "
                 "is what did arrive, not everything published.</p>")
    # the high-impact headlines are pinned above the latest ones: counted over the whole feed but listed
    # newest first, they fell below the cut whenever the last hour was quiet, and the count named stories
    # the card did not show
    pinned = [h for h in shown if h.get("impact") == "high"][:limit]
    rest = [h for h in shown if h.get("impact") != "high"][:max(0, limit - len(pinned))]
    body += ((f"<span class='ti-label'>High impact</span><div class='ti-rows'>{_news_rows(pinned, now_ms)}</div>"
              "<span class='ti-label'>Latest</span>") if pinned else "")
    body += f"<div class='ti-rows'>{_news_rows(rest, now_ms)}</div>"
    body += ("<p class='muted ti-note'>Topic and impact are keyword tags over the headline text, not a judgment "
             "of the story. Treat them as a way to sort the list, not as a read on the market.</p>")
    return card_html("News", body, "down" if high else "neutral")


def _news_rows(rows: list[dict], now_ms: int) -> str:
    body = ""
    for h in rows:
        assets = " ".join(str(a) for a in (h.get("assets") or [])[:3])
        impact = str(h.get("impact") or "")
        title = html.escape(str(h.get("title") or ""))[:150]
        url = str(h.get("url") or "")
        # only an http(s) link is rendered as one: a feed controls this string
        linked = (f"<a href='{html.escape(url)}' target='_blank' rel='noopener noreferrer'>{title}</a>"
                  if url.startswith(("https://", "http://")) else title)
        body += (f"<div class='ti-news'><span class='muted'>{html.escape(_age(h.get('published_ms'), now_ms))}</span>"
                 f"<div><span class='ti-news-title'>{linked}</span><div class='ti-tags'>"
                 f"<span class='muted'>{html.escape(str(h.get('source') or ''))}</span>"
                 + (f"<span class='muted'>{html.escape(assets)}</span>" if assets else "")
                 + chip(str(h.get("topic") or ""), "neutral-fill")
                 + (chip(impact, IMPACT_TONE.get(impact, "neutral")) if impact else "")
                 + "</div></div></div>")
    return body
def gold_html(metals, now_ms: int) -> str:
    """Gold spot, with the feed's own freshness carried through."""
    if not getattr(metals, "available", False):
        return card_html("Gold · XAU/USD",
                         f"<p class='muted'>No gold feed: "
                         f"{html.escape(str(getattr(metals, 'error', '') or 'unavailable'))}.</p>", "neutral")
    price = getattr(metals, "xau_usd", None)
    unit = html.escape(str(getattr(metals, "unit", "troy_ounce")).replace("_", " "))
    body = (f"<div class='ti-big'>{fmt_num(price, 2, '$')}</div><p class='muted'>per {unit}")
    bid, ask = getattr(metals, "bid", None), getattr(metals, "ask", None)
    if bid is not None and ask is not None:
        body += f" · bid {fmt_num(bid, 2, '$')} / ask {fmt_num(ask, 2, '$')}"
    body += "</p>"
    if getattr(metals, "is_stale", False):
        body += ("<p><span class='ti-chip down'>stale</span> The feed reports this print as stale, so it "
                 "is not the current price.</p>")
    else:
        body += (f"<div class='ti-tags'>{chip('fresh · computed ' + _age(getattr(metals, 'computed_ms', None), now_ms), 'accent')}</div>"
                 f"<p class='muted'>Computed {html.escape(_age(getattr(metals, 'computed_ms', None), now_ms))}"
                 f" · {html.escape(str(getattr(metals, 'source', '')))}</p>")
    body += ("<div class='ti-rows'><div class='ti-split'><span>Silver</span><span class='muted'>plan-gated on the free tier</span></div>"
             "<div class='ti-split'><span>Copper</span><span class='muted'>plan-gated on the free tier</span></div></div>"
             "<p class='muted ti-note'>Gold only: silver and copper are gated behind a paid tier on this feed, so "
             "they are not shown rather than guessed at.</p>")
    return card_html("Gold · XAU/USD", body, "warn" if getattr(metals, "is_stale", False) else "neutral")
EVENT_TONE = {"sell_the_news": "warn", "spike_faded": "warn", "positioned": "warn", "confirmed": "", "absorbed": "",
              "flat": "", "reacting": "", "scheduled": "", "no_data": ""}
EVENT_LABEL = {"sell_the_news": "sell the news", "spike_faded": "spike faded", "positioned": "positioned",
               "confirmed": "confirmed", "absorbed": "absorbed", "flat": "flat into it", "reacting": "reacting",
               "scheduled": "scheduled", "no_data": "no candles"}
ACTIVE_EVENT_STATES = ("positioned", "reacting", "sell_the_news", "spike_faded", "confirmed")


def event_traps_html(reads, now_ms: int) -> str:
    """Buy the rumour, sell the news: positioning into each high-impact USD print and what it did after."""
    from core.trades import fmt_when
    if not reads:
        return card_html("Event traps", "<p class='muted'>No high-impact USD prints from 48 hours ago onward.</p>",
                         "neutral")
    body = "<div class='ti-rows'>"
    for r in reads:
        trapped = r.trapped
        side = f" · {trapped}{' trapped' if trapped in ('longs', 'shorts') else ''}" if trapped else ""
        label = EVENT_LABEL.get(r.state, r.state)
        if r.state == "sell_the_news" and (r.runup_pct or 0) < 0:
            label = "buy the news"
        nums = " · ".join(x for x in (
            f"into it {r.runup_pct:+.2f}% ({r.runup_z:+.1f}σ)" if r.runup_pct is not None else "",
            f"first hour {r.spike_pct:+.2f}% ({r.spike_z:+.1f}σ)" if r.spike_pct is not None else "",
            f"since {r.since_pct:+.2f}%" if r.since_pct is not None else "",
            f"normal 4h move ±{r.expected_move_pct:.2f}%" if r.expected_move_pct is not None else "") if x)
        body += (f"<div class='ti-cal'><div class='ti-split'><strong>{html.escape(r.title)}</strong>"
                 f"{chip(label + side, EVENT_TONE.get(r.state) or 'outline')}</div>"
                 f"<span>{html.escape(fmt_when(r.time_ms))}</span><span>{html.escape(r.headline)}</span>"
                 + (f"<span class='muted'>{html.escape(nums)}</span>" if nums else "") + "</div>")
    body += ("</div><p class='muted ti-note'>Read from BTC's 1h candles against its own hourly volatility over the "
             "prior week. The calendar has no actual figures, so the surprise is read from the price. Prints in "
             "the same minute count as one.</p>")
    tone = "warn" if any(r.state in ("sell_the_news", "spike_faded", "positioned") for r in reads) else "neutral"
    return card_html("Event traps · buy the rumour, sell the news", body, tone)


NARRATIVE_TONE = {"fresh": "warn", "building": "warn", "running": "outline", "fading": "outline"}


def narratives_html(items) -> str:
    """Which headline stories are rising or dying, and so how much of each is likely priced."""
    if not items:
        return card_html("Narrative shelf-life", "<p class='muted'>No tagged topic has two or more headlines in the "
                         "feed.</p>", "neutral")
    body = "<div class='ti-rows'>"
    for n in items:
        body += (f"<div class='ti-cal'><div class='ti-split'><strong>{html.escape(n.topic.title())}</strong>"
                 f"{chip(n.state, NARRATIVE_TONE.get(n.state, 'outline'))}</div>"
                 f"<span>{html.escape(n.shelf_life)}</span>"
                 f"<span class='muted'>{n.headlines} headlines, {n.recent} in the last 6h · latest: "
                 f"{html.escape(n.latest_title)}</span></div>")
    body += ("</div><p class='muted ti-note'>An attention tracker, not a forecast: topics come from keyword tags and "
             "the feed reaches back about a day and a half, so an age marked + is a floor.</p>")
    return card_html("Narrative shelf-life", body, "neutral")


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
            + (" on the Fed, rates and the economy" if asset == "MACRO" else f" mentioning {html.escape(asset)}" if asset else "")
            + f" · {len(tradeable)} with a quote worth quoting</p><div class='ti-rows'>")
    for m in markets[:limit]:
        band = str(m.get("band") or "unknown")
        question = html.escape(str(m.get("question") or ""))[:120]
        url = str(m.get("url") or "")
        linked = (f"<a href='{html.escape(url)}' target='_blank' rel='noopener noreferrer'>{question}</a>"
                  if url.startswith("https://") else question)
        prob = (m.get("probability") or 0) * 100
        days = m.get("days_left")
        assets = " ".join(str(a) for a in (m.get("assets") or [])[:2])
        body += (f"<div class='ti-odds'><div><span class='ti-news-title'>{linked}</span><br><span class='muted'>"
                 f"{html.escape(assets)}{' · ' if assets else ''}volume {fmt_num(m.get('volume_24h'), 0, '$')} · "
                 f"ends in {fmt_num(days, 0) if days is not None else '—'}d</span></div>"
                 f"<div class='ti-odds-p'>{_bar(prob)}<strong>{fmt_num(prob, 1, suffix='%')}</strong></div>"
                 f"<div>{chip(band, BAND_TONE.get(band, 'neutral'))}"
                 + (f"<span class='muted'>{fmt_num(m.get('band_score'), 0)}</span>" if m.get("band_score") is not None else "")
                 + (f"<br><span class='muted'>{html.escape(str(m.get('band_reason')))}</span>"
                    if m.get("band_reason") and band != "clean" else "")
                 + "</div></div>")
    body += "</div>"
    fragile = [m for m in markets[:limit] if m.get("band") == "fragile"]
    if fragile:
        reason = next((str(m.get("band_reason")) for m in fragile if m.get("band_reason")), "")
        body += (f"<p class='muted'><strong>{len(fragile)} of these are fragile.</strong> "
                 + html.escape(reason or "The quote can move or fill badly.")
                 + " A probability on a fragile market is a printed number, not a forecast.</p>")
    attribution = str(getattr(preds, "attribution", "") or "")
    if attribution:
        body += f"<p class='muted ti-note'>{html.escape(attribution)}</p>"
    return card_html("Prediction markets", body, "neutral")
# ---------- the sidebar clock ----------
# Rendered as a self-contained ticking component rather than a formatted string: Streamlit only redraws
# on a rerun, so a server-rendered clock sits frozen at the page-load time, which is worse than none.
CLOCK_HTML = """<link href="https://fonts.googleapis.com/css2?family=Barlow:wght@400;600&family=Barlow+Condensed:wght@600&display=swap" rel="stylesheet">
<div id="ti-clock" style="font:12px Barlow,system-ui,sans-serif;color:__TEXT__;line-height:1.35">
<div style="font-size:11px;letter-spacing:.08em;text-transform:uppercase;font-weight:600;opacity:.7">Clock</div>
<span id="ti-utc" style="font:600 24px 'Barlow Condensed',Barlow,system-ui,sans-serif"></span><br>
<span id="ti-pkt" style="opacity:.7"></span></div>
<script>
function pad(n){return String(n).padStart(2,'0')}
function tick(){
  var now = new Date();
  // PKT is UTC+5 with no daylight saving, so a fixed offset is exact for every date
  var pkt = new Date(now.getTime() + 5*3600*1000);
  var days = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
  var mons = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  document.getElementById('ti-utc').textContent =
    pad(now.getUTCHours())+':'+pad(now.getUTCMinutes())+':'+pad(now.getUTCSeconds())+' UTC';
  document.getElementById('ti-pkt').textContent =
    '('+pad(pkt.getUTCHours())+':'+pad(pkt.getUTCMinutes())+':'+pad(pkt.getUTCSeconds())+' PKT - '
    + days[pkt.getUTCDay()]+' '+pad(pkt.getUTCDate())+' '+mons[pkt.getUTCMonth()]+')  PKT is UTC+5';
}
tick(); setInterval(tick, 1000);
</script>"""


def clock_html(dark: bool) -> str:
    """Both zones, ticking: UTC first and large, because every candle boundary in this system is a UTC
    boundary; PKT after it in brackets, the zone the trader lives in."""
    return CLOCK_HTML.replace("__TEXT__", palette(dark)["text"])


def rates_html(macro, now_ms: int) -> str:
    """The dollar, the 10-year, the 2-year and the 2s10s spread, each with the date it is from.

    Every value carries its own as-of date because these are daily series published with a lag - a
    reading two days old is normal and is labelled so, rather than presented as live."""
    if not getattr(macro, "available", False):
        return card_html("US macro",
                         f"<p class='muted'>No macro feed: "
                         f"{html.escape(str(getattr(macro, 'error', '') or 'unavailable'))}.</p>", "neutral")
    from core.trades import fmt_date

    rows = ""
    for s in (getattr(macro, "series", {}) or {}).values():
        label = html.escape(str(s.get("label") or s.get("id") or ""))
        if not s.get("available"):
            rows += (f"<tr><td>{label}</td><td colspan='3' class='muted'>"
                     f"{html.escape(str(s.get('note') or 'unavailable'))}</td></tr>")
            continue
        unit = str(s.get("unit") or "")
        is_yield = unit == "%"
        value = fmt_num(s.get("value"), 2, suffix="%" if is_yield else "")
        change = s.get("change")
        if change is None:
            moved = "<span class='muted'>no week-ago value</span>"
        else:
            # yields move in basis points, the dollar index in points
            size = change * 100 if is_yield else change
            text = f"{size:+.0f} bp" if is_yield else f"{size:+.2f}"
            loud = abs(change) >= (0.15 if is_yield else 1.5)
            moved = f"<span class='ti-chip {'warn' if loud else 'neutral'}'>{html.escape(text)}</span>"
        when = html.escape(fmt_date(s.get("as_of_ms")))
        stale = (" <span class='ti-chip down'>stale</span>" if s.get("stale") else "")
        rows += (f"<tr><td>{label}</td><td><strong>{value}</strong></td><td>{moved}</td>"
                 f"<td class='muted'>as of {when}{stale}</td></tr>")

    body = ("<table><thead><tr><th>Series</th><th>Latest</th><th>1 week</th><th>Date</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>")
    spread = getattr(macro, "spread_2s10s", None)
    if spread is not None:
        state = "inverted - the market prices cuts ahead" if spread < 0 else "positively sloped"
        body += (f"<p>2s10s spread <strong>{spread * 100:+.0f} bp</strong> "
                 f"<span class='muted'>({html.escape(state)}; both yields from the same day)</span></p>")
    if getattr(macro, "error", None):
        body += (f"<p class='muted'>Partial: {html.escape(str(macro.error))[:200]}</p>")
    origin = {"fred": "FRED", "treasury.gov": "the US Treasury yield curve (FRED unreachable)",
              "fred+treasury.gov": "FRED, yields from the US Treasury yield curve"}.get(macro.source, macro.source)
    body += (f"<p class='muted'>Daily series from {html.escape(origin)}, published with a business-day lag or two - a "
             "reading a couple of days old is normal. A move of 15 bp in a yield or 1.5 points in the "
             "dollar over a week is highlighted.</p>")
    return card_html("US macro", body, "neutral")


def exchange_flows_html(f: dict) -> str:
    """Coins arriving on exchanges or leaving them, measured against the past year, with its track record."""
    from core.trades import fmt_date
    if not f.get("available"):
        return card_html("Exchange flows", f"<p class='muted'>Unavailable: {html.escape(str(f.get('note', '')))}.</p>",
                         "neutral")
    tone = {"distribution": "down", "accumulation": "up"}.get(f["state"], "neutral")
    z = f.get("net_7d_z")
    head = {"distribution": "coins arriving on exchanges", "accumulation": "coins leaving exchanges",
            "neutral": "exchange flows within their normal range"}[f["state"]]
    body = (f"<div class='ti-dir {tone}'>{html.escape(f['state'].title())}</div>"
            f"<p>{html.escape(head[0].upper() + head[1:])}: 7-day netflow <strong>{f['net_7d_btc']:+,.0f} BTC</strong>"
            + (f" ({z:+.1f}σ against the past year)" if z is not None else "") + "</p><div class='ti-grid3'>"
            + _kv("Net, last day", f"{f['net_1d_btc']:+,.0f} BTC")
            + _kv("Net, 30 days", f"{f['net_30d_btc']:+,.0f} BTC")
            + _kv("Reserve, 30d", "—" if f.get("reserve_30d_change_btc") is None else
                  f"{f['reserve_30d_change_btc']:+,.0f} BTC ({f['reserve_30d_change_pct']:+.2f}%)")
            + "</div>")
    if f.get("divergence"):
        body += f"<p><span class='ti-chip warn'>price divergence</span> {html.escape(f['divergence'])}.</p>"
    if f.get("inflow_spike"):
        body += (f"<p><span class='ti-chip warn'>inflow spike</span> {f['inflow_1d_btc']:,.0f} BTC sent to exchanges "
                 f"on the last day, {f['inflow_spike_z']:+.1f}σ against the past 90 days.</p>")
    flags = (" · preliminary (Coin Metrics flash figures)" if f.get("preliminary") else "") + \
            (f" · {f['age_days']:.0f} days old" if f.get("stale") else "")
    body += (f"<p class='muted ti-note'>As of {html.escape(fmt_date(f['as_of_ms']))}{html.escape(flags)}. "
             f"{html.escape(f['scope'][0].upper() + f['scope'][1:])}. {html.escape(f['track_record'])}</p>")
    return card_html("Exchange flows · on-chain", body, tone)


def cot_html(cot, now_ms: int) -> str:
    """Who is positioned which way in CME Bitcoin futures, from the CFTC's weekly report.

    The as-of date leads, because this is weekly data: positions as of Tuesday, released Friday, and by
    the following Thursday it describes positioning from before the current weekly candle opened. The
    headline is the COT index - where the large speculators' net sits within its own 52-week range -
    because that, not the raw number, is what says whether the trade is crowded."""
    if not getattr(cot, "available", False):
        return card_html("CME Bitcoin positioning (CFTC)",
                         f"<p class='muted'>No positioning report: "
                         f"{html.escape(str(getattr(cot, 'error', '') or 'unavailable'))}.</p>", "neutral")
    from core.data.cot import CROWDED_LONG, CROWDED_SHORT
    from core.trades import fmt_date

    when = html.escape(fmt_date(cot.as_of_ms)) if cot.as_of_ms else "unknown"
    age = cot.age_days
    # the weekday comes from the date itself: holiday weeks move the as-of day off its usual Tuesday
    body = (f"<p><strong>Positions as of {when}</strong> <span class='muted'>· released the following Friday"
            + (f" · {age:.0f} days old" if age is not None else "") + "</span>")
    if cot.stale:
        body += (" <span class='ti-chip down'>stale</span></p><p class='muted'>Older than this report "
                 "should ever be - a release was missed, as happens during a US government shutdown. "
                 "Do not read this as current positioning.</p>")
    else:
        body += "</p>"

    def contracts(n):
        if n is None:
            return "—"
        text = f"{n:+,} contracts"
        if cot.contract_btc:
            text += f" (~{n * cot.contract_btc:+,.0f} BTC)"
        return text

    index = cot.cot_index
    if index is None:
        body += (f"<p class='muted'>Not enough history for a positioning index "
                 f"({cot.weeks_in_index} weeks; needs 26).</p>")
    else:
        if index >= CROWDED_LONG:
            read, tone = "crowded long", "down"
        elif index <= CROWDED_SHORT:
            read, tone = "crowded short", "up"
        else:
            read, tone = "neither extreme", "neutral"
        body += (f"<p>Large speculators: <span class='ti-chip {tone}'>{html.escape(read)}</span> "
                 f"<strong>{index:.0f} of 100</strong> <span class='muted'>against their own "
                 f"{cot.weeks_in_index}-week range</span></p>")

    rows = [("Large speculators (net)", contracts(cot.spec_net),
             f"{cot.spec_net_change:+,} on the week" if cot.spec_net_change is not None else ""),
            ("Commercials (net)", contracts(cot.commercial_net), ""),
            ("Small traders (net)", contracts(cot.small_trader_net), ""),
            ("Open interest", f"{cot.open_interest:,} contracts" if cot.open_interest is not None else "—",
             f"{cot.open_interest_change:+,} on the week" if cot.open_interest_change is not None else "")]
    body += "<table><tbody>" + "".join(
        f"<tr><td>{html.escape(a)}</td><td><strong>{html.escape(b)}</strong></td>"
        f"<td class='muted'>{html.escape(c)}</td></tr>" for a, b, c in rows) + "</tbody></table>"

    traders = cot.traders or {}
    if traders.get("commercials_long") is not None and traders.get("commercials_short") is not None:
        body += (f"<p class='muted'>Commercials are only {traders['commercials_long']} long and "
                 f"{traders['commercials_short']} short traders here, so their net is a handful of "
                 "hedgers rather than a market read; the speculators are the signal.</p>")
    body += (f"<p class='muted'>{html.escape(cot.market or 'CME Bitcoin')}. Weekly data - context for "
             "the weekly and monthly view, never for an intraday call. A reading of "
             f"{CROWDED_LONG}+ is conventionally crowded long, {CROWDED_SHORT} or below crowded short.</p>")
    return card_html("CME Bitcoin positioning (CFTC)", body, "warn" if cot.stale else "neutral")


# ---------- Industry layout: hero, header bar, rail ----------

DIR_GLYPH = {"BULLISH": "▲", "BEARISH": "▼"}


def _left(ms: int) -> str:
    mins = max(0, int(ms // 60_000))
    if mins >= 1440:
        return f"{mins // 1440}d {(mins % 1440) // 60:02d}h"
    return f"{mins // 60}h {mins % 60:02d}m"


def desk_hero_html(number: int, profile, category, now_ms: int) -> str:
    """The desk's title block and the countdown to the candle its summary is pinned to.

    The countdown is the one number on the desk that changes the reading of everything else: a summary
    anchored three minutes before its candle closes is about to be scored, not about to be acted on."""
    from core.trades import fmt_when

    opened, closes = profile.window_open(now_ms), profile.window_close(now_ms)
    span = max(1, closes - opened)
    pct = min(100, max(0, round(100 * (now_ms - opened) / span)))
    tfs = " / ".join(profile.timeframes)
    meta = "".join(f"<span>{html.escape(x)}</span>" for x in (
        f"{category.chart_tf} chart · {category.context_tf} context", f"Reads {tfs}",
        f"Judges the next {profile.horizon}", f"Anchored to the {profile.window_label} candle"))
    return (f"<div class='ti-hero'><div class='ti-hero-main'>"
            f"<span class='ti-kicker'>{number:02d} · Desk</span><h1>{html.escape(profile.label)}</h1>"
            f"<div class='ti-hero-meta'>{meta}</div></div>"
            f"<div class='ti-countdown blueprint'>{CORNERS}"
            f"<div class='ti-countdown-top'><span>{html.escape(profile.window_title)} candle closes in</span>"
            f"<span>{pct}% elapsed</span></div>"
            f"<span class='ti-countdown-left'>{_left(closes - now_ms)}</span>"
            f"<div class='ti-bar'><div style='width:{pct}%'></div></div>"
            f"<span class='ti-countdown-at'>{html.escape(fmt_when(closes))}</span></div></div>")


def engine_hero_html(number: int, title: str, sub: str) -> str:
    return (f"<div class='ti-hero'><div class='ti-hero-main'><span class='ti-kicker'>{number:02d} · Engine</span>"
            f"<h1>{html.escape(title)}</h1><p class='ti-hero-p'>{html.escape(sub)}</p></div></div>")


def market_label(symbol: str | None) -> str:
    """The market the candles came from: the BTC perpetual, or spot when the perp feed is down."""
    return "BTC perp" if symbol and "perp" in symbol else "BTC / USD spot"


def top_bar_html(price, change_pct, updated: str, symbol: str | None = None) -> str:
    """BTC price, its 24h move and the snapshot time - the line every screen is read against."""
    tone = "" if change_pct is None else ("up" if change_pct >= 0 else "down")
    change = "" if change_pct is None else f"<span class='ti-top-chg {tone}'>{fmt_pct(change_pct)} 24h</span>"
    return (f"<div class='ti-top'><span class='ti-label'>{html.escape(market_label(symbol))}</span>"
            f"<span class='ti-top-price'>{fmt_num(price, 0, '$')}</span>{change}"
            f"<span class='ti-top-upd'>{html.escape(updated)}</span></div>")


GAMMA_READ = {
    "positive": ("POSITIVE GAMMA · BALANCE", "up",
                 "Dealers hedge against the move: they sell rallies and buy dips. Expect a range. "
                 "Fade the edges toward the walls, target the point of control, and do not trust breakouts."),
    "negative": ("NEGATIVE GAMMA · IMBALANCE", "down",
                 "Dealers hedge with the move: they chase it. Expect expansion. Ride momentum, enter on "
                 "VWAP pullbacks, and use the walls as targets rather than as ceilings."),
}


def gamma_bar_html(g: dict | None, price: float | None = None) -> str:
    """The regime every desk trades inside: dealer gamma from the Deribit chain, with its key levels."""
    if not g or not g.get("available"):
        why = (g or {}).get("note") or "options chain unavailable"
        return (f"<div class='ti-gamma neutral'><i class='ti-gamma-mark'></i><span class='ti-gamma-reg'>GAMMA REGIME UNKNOWN</span>"
                f"<span class='ti-gamma-play'>{html.escape(why)}</span></div>")
    label, tone, play = GAMMA_READ.get(g.get("regime"), ("GAMMA NEUTRAL", "neutral", "Net dealer gamma is flat at spot."))
    zgl = g.get("zero_gamma_level")
    side = ""
    if zgl and price:
        side = f" · price {'above' if price >= zgl else 'below'} the flip by {fmt_num(abs(price - zgl), 0, '$')}"
    net = g.get("net_gex_usd_per_1pct")
    net_txt = "—" if net is None else f"{'+' if net >= 0 else '-'}${abs(net) / 1e6:,.1f}M"
    levels = [("Put wall", g.get("put_wall")), ("Zero gamma", zgl), ("Max gamma", g.get("max_gamma_strike")),
              ("Call wall", g.get("call_wall"))]
    lv = "".join(f"<span><b>{n}</b> {fmt_num(v, 0, '$')}</span>" for n, v in levels)
    n_exp = len(g.get("expiries") or [])
    tip = (f"Net GEX {net_txt} per 1% move, from {g.get('options_used', 0)} options across {n_exp} expiries in the "
           "next 30 days. Assumes the naive dealer sign: calls positive, puts negative. Real dealer positioning is not published.")
    return (f"<div class='ti-gamma {tone}' title='{html.escape(tip)}'><i class='ti-gamma-mark'></i>"
            f"<span class='ti-gamma-reg'>{label}</span>"
            f"<span class='ti-gamma-net'>net {net_txt} / 1%{html.escape(side)}</span>"
            f"<span class='ti-gamma-lv'>{lv}</span>"
            f"<span class='ti-gamma-play'>{html.escape(play)}</span></div>")


def rail_tone(verdict, analysis) -> str:
    """up, down or neutral for the desk's current call, so the rail can colour its read."""
    d = (verdict.anchored.direction if verdict is not None else
         analysis.direction.direction if analysis is not None else "")
    return tone_for_direction(d)


def rail_style_html(tones: list[str], timeframes: list[str], dark: bool, signal: bool) -> str:
    """Per-desk rules for the rail: each read in its direction's colour (green and red when signal colours
    are on, steel and ink when not), and the chart timeframe set beside the desk's name. Generated per run
    because Streamlit gives no way to style one radio caption differently from the next."""
    pal = palette(dark, signal)
    colour = {"up": pal["up"], "down": pal["down"], "neutral": pal["muted"]}
    # the n-th option is the n-th child of the radio group, whether Streamlit renders each option as a
    # bare label or wraps it: label:nth-of-type matched every option as the first once it was wrapped
    opt = 'section[data-testid="stSidebar"] [role="radiogroup"] > :nth-child'
    rules = []
    for i, (tone, tf) in enumerate(zip(tones, timeframes), start=1):
        rules.append(f"{opt}({i}) [data-testid=\"stCaptionContainer\"] p {{ color: {colour.get(tone, pal['muted'])} !important; }}")
        rules.append(f"{opt}({i}) [data-testid=\"stMarkdownContainer\"] p::after {{ content: \"{html.escape(tf)}\"; "
                     f"font-family: 'Barlow', sans-serif; font-weight: 400; font-size: 0.75rem; "
                     f"letter-spacing: 0; text-transform: none; color: {pal['muted']}; margin-left: 10px; }}")
    return "<style>" + "\n".join(rules) + "</style>"


def rail_read(verdict, analysis) -> str:
    """The one-line read under a desk in the rail: the anchored call if there is one, else the live read."""
    if verdict is not None:
        d, c = verdict.anchored.direction, verdict.anchored.confidence
    elif analysis is not None:
        d, c = analysis.direction.direction, analysis.direction.confidence
    else:
        return "no data"
    return f"{DIR_GLYPH.get(d, '◆')} {d} · {c:.0%}"


def chart_head_html(a, signal: bool = False, symbol: str | None = None) -> str:
    """The chart's title and legend, as the design frames the figure."""
    legend = ("<span><i class='sw ema21'></i>EMA 21</span><span><i class='sw ema50'></i>EMA 50</span>"
              "<span><i class='sw up'></i>up</span><span><i class='sw down'></i>down</span>"
              "<span title='The volatility envelope drawn as bars: body ±0.5σ, wick ±1σ. Not predicted candles.'>"
              "<i class='sw env'></i>envelope</span>")
    # the series legend is drawn inside the chart itself (it also covers EMA 100/200 and the forecast),
    # so the frame carries the title only rather than a second, shorter legend
    del legend
    return (f"<div class='ti-chart-head'><span class='ti-chart-title'>{html.escape(market_label(symbol))} · {html.escape(a.chart_tf)}</span>"
            "</div>")




# ---------- the timeframe cascade (bias / anchor / engine) ----------
CASCADE_TONE = {"A": "up", "B": "warn", "C": "down", "N": "neutral", "?": "neutral"}
ROLE_WORDS = {"bias": "Bias · sets the vector", "anchor": "Anchor · the candle being called",
              "engine": "Engine · where change shows first"}


CHECK_WORDS = {"close_outside_value": "close outside value", "volume_expansion": "volume expansion",
               "vwap_acceptance": "VWAP acceptance"}


def value_breaks_html(reads: dict) -> str:
    """The desk's value-area breaks on 1H/4H/1D: the latest one on each, which of the three conditions it
    met, and what came after it. Computed from completed candles in code."""
    rows = ""
    tone = "neutral"
    for tf, r in reads.items():
        if not r.available:
            rows += f"<tr><td><strong>{html.escape(tf)}</strong></td><td class='muted' colspan='2'>{html.escape(r.note or r.headline)}</td></tr>"
            continue
        # a held break reads in its own direction; a failed one in the opposite one (a failed break down is
        # a bear trap, which is bullish); an unqualified poke has no direction yet
        cls = "muted"
        if r.state in ("qualified", "pullback"):
            cls = "up" if r.direction == "up" else "down"
        elif r.state == "failed":
            cls = "down" if r.direction == "up" else "up"
        if cls != "muted":
            tone = cls
        detail = ""
        if r.state != "none":
            ticks = " · ".join(f"<span style='white-space:nowrap'>{'■' if ok else '□'} {CHECK_WORDS[k]}</span>"
                               for k, ok in r.checks)
            when = "the last completed candle" if r.bars_ago == 0 else f"{r.bars_ago} candles ago"
            detail = (f"edge {fmt_num(r.edge, 0, '$')} · VWAP {fmt_num(r.vwap, 0, '$')} · volume "
                      f"{'—' if r.volume_ratio is None else f'{r.volume_ratio:.1f}x'} median · broke {when}<br>{ticks}")
            if r.pullback_bars_ago is not None:
                detail += f"<br>first pullback {'on the last completed candle' if r.pullback_bars_ago == 0 else f'{r.pullback_bars_ago} candles ago'}"
        rows += (f"<tr><td><strong>{html.escape(tf)}</strong></td><td class='{cls}'>{html.escape(r.headline)}</td>"
                 f"<td class='muted'>{detail}</td></tr>")
    body = (f"<table class='ti-cascade'><tbody>{rows}</tbody></table>"
            "<p class='muted ti-note'>A break counts only when a completed candle closes outside the value area "
            "with volume at least 1.5x its 20-candle median and on the far side of VWAP anchored at the open of "
            "the next timeframe up. Read on 1H, 4H and 1D only.</p>")
    return card_html("Value-area breaks", body, tone)


def harmonics_html(found: dict) -> str:
    """Harmonic patterns on the desk's 1H/4H/1D, valid only beside a VWAP, value-area or gamma level."""
    from core.indicators.harmonics import TRACK_RECORD
    rows, tone = "", "neutral"
    for tf, hs in found.items():
        for h in hs:
            cls = ("up" if h.side == "bullish" else "down") if h.valid else "muted"
            if h.valid and tone == "neutral":
                tone = cls
            stage = h.stage.replace("_", " ")
            extra = (f" · {h.distance_pct:+.2f}% from price" if h.distance_pct is not None else "")
            rows += (f"<tr><td><strong>{html.escape(tf)}</strong></td>"
                     f"<td class='{cls}'>{html.escape(h.side.title() + ' ' + h.name)}</td>"
                     f"<td>{chip(stage, 'warn' if h.stage == 'in_zone' else 'outline')} "
                     f"{chip('valid' if h.valid else 'unconfirmed', 'accent' if h.valid else 'outline')}</td>"
                     f"<td class='muted'>zone {fmt_num(h.prz[0], 0, '$')}–{fmt_num(h.prz[1], 0, '$')}{extra}"
                     f" · targets {fmt_num(h.targets[0], 0, '$')} / {fmt_num(h.targets[1], 0, '$')}"
                     f" · invalid past {fmt_num(h.invalidation, 0, '$')}"
                     + (f"<br>at the zone: {html.escape(', '.join(h.confluence))}" if h.confluence else "")
                     + "</td></tr>")
    if not rows:
        body = ("<p class='muted'>No Gartley, Bat, Butterfly, Crab or Cypher forming, in its zone or just "
                f"completed on {', '.join(found) or 'the 1H, 4H or 1D'}.</p>")
    else:
        body = f"<table class='ti-cascade'><tbody>{rows}</tbody></table>"
    body += ("<p class='muted ti-note'>Swings from an ATR zigzag on completed candles; ratios to the standard "
             "XABCD tables. A pattern is valid only when its reversal zone sits on a VWAP, value-area or gamma "
             f"level. {html.escape(TRACK_RECORD)}</p>")
    return card_html("Harmonic patterns", body, tone)


def cascade_html(c, invalidation: float | None = None, price: float | None = None) -> str:
    """The desk's cascade: which state it is in, what that means, and each timeframe's read.

    The state is computed in code from the candles; this card only reports it. A control shift that has
    also closed beyond the desk's own anchored invalidation says so, because that is the stronger case."""
    arrows = {"bullish": "▲ bullish", "bearish": "▼ bearish", "neutral": "◆ neutral"}
    tone = CASCADE_TONE.get(c.state, "neutral")
    if c.state == "A" and c.bias is not None:
        tone = "up" if c.bias.direction == "bullish" else "down"
    body = (f"<div class='ti-split'><div class='ti-dir {tone}'>State {html.escape(c.state)} · "
            f"{html.escape(c.title)}</div></div><p>{html.escape(c.playbook)}</p>")
    if c.state == "C" and invalidation is not None and price is not None and c.bias is not None:
        broke = price < invalidation if c.bias.direction == "bullish" else price > invalidation
        if broke:
            body += (f"<p><span class='ti-chip down'>beyond invalidation</span> Price {fmt_num(price, 0, '$')} "
                     f"is also past the anchored invalidation at {fmt_num(invalidation, 0, '$')}.</p>")
    rows = ""
    for r in (c.bias, c.anchor, *c.engine):
        if r is None:
            continue
        d = arrows.get(r.direction, r.direction) if r.available else "not enough candles"
        cls = {"bullish": "up", "bearish": "down"}.get(r.direction, "muted") if r.available else "muted"
        rows += (f"<tr><td><strong>{html.escape(r.timeframe)}</strong></td><td class='muted'>{ROLE_WORDS[r.role]}</td>"
                 f"<td class='{cls}'>{html.escape(d)}</td>"
                 f"<td class='muted'>EMA {html.escape(r.ema)} · structure {html.escape(r.structure)}</td></tr>")
    body += f"<table class='ti-cascade'><tbody>{rows}</tbody></table>"
    if c.vwap is not None:
        body += (f"<p class='muted'>VWAP from the open of the current {html.escape(c.bias.timeframe if c.bias else '')} "
                 f"candle: <strong>{fmt_num(c.vwap, 0, '$')}</strong>"
                 + (f" · {c.closes_beyond_vwap} completed {html.escape(c.engine[-1].timeframe)} closes against the bias"
                    if c.closes_beyond_vwap and c.engine else "") + "</p>")
    if c.pullback:
        pb = c.pullback
        body += ("<span class='ti-label'>Pullback bands · how deep is still normal on the "
                 f"{html.escape(pb['timeframe'])}</span><div class='ti-grid3'>"
                 + _kv("50th", fmt_num(pb["p50"], 0, "$")) + _kv("80th", fmt_num(pb["p80"], 0, "$"))
                 + _kv("95th", fmt_num(pb["p95"], 0, "$"), "down") + "</div>")
    body += ("<p class='muted ti-note'>Read from the candles in code, not by a model. The engine timeframes warn "
             "early; only the bias and the anchor set the call.</p>")
    return card_html("Timeframe cascade", body, tone)


# ---------- the forecast pane (his Section 8) ----------
FORECAST_HORIZON = {"next 4h": "next 4 hours", "next 24h": "next 24 hours"}


def forecast_html(f, tr, horizon_label: str) -> str:
    """The odds beside the forecast chart: where the median path ends, the 68% range, the chance of
    finishing higher, the chance of trading at each level, the path candle by candle in UTC and PKT, and
    how the same model has done on this desk's own history."""
    from core.trades import fmt_when
    from core.indicators.volatility import memory_label

    horizon = FORECAST_HORIZON.get(horizon_label, horizon_label)
    if not getattr(f, "available", False):
        return card_html(f"Forecast · {horizon}", f"<p class='muted'>No forecast: {html.escape(f.reason)}.</p>")
    money = lambda x: fmt_num(x, 0, "$")  # noqa: E731
    end_ms = f.steps[-1].time_ms + f.bar_ms if f.steps else None
    body = (f"<p>Most likely path ends near <strong>{money(f.end_median)}</strong> by "
            f"{html.escape(fmt_when(end_ms))}.</p>"
            f"<p>68% range <strong>{money(f.end_low)} to {money(f.end_high)}</strong> · "
            f"chance it ends higher <strong>{f.p_up:.0%}</strong></p>")
    reachable = [lv for lv in f.levels if lv.p_touch >= 0.01]   # a wall 10% away at 0% is not a level today
    if reachable:
        rows = "".join(f"<tr><td>{html.escape(lv.label)}</td><td>{money(lv.price)}</td>"
                       f"<td><strong>{lv.p_touch:.0%}</strong></td></tr>" for lv in reachable)
        body += ("<p class='muted'>Chance price trades at each level before the horizon ends:</p>"
                 "<table><thead><tr><th>Level</th><th>Price</th><th>Odds</th></tr></thead>"
                 f"<tbody>{rows}</tbody></table>")
    rows = "".join(f"<tr><td>{html.escape(fmt_when(s.time_ms + f.bar_ms))}</td><td>{money(s.close)}</td>"
                   f"<td class='muted'>{money(s.band_low)} to {money(s.band_high)}</td>"
                   f"<td>{s.p_above_now:.0%}</td></tr>" for s in f.steps)
    body += ("<table><thead><tr><th>Candle closes</th><th>Path</th><th>68% range</th><th>Above now</th></tr></thead>"
             f"<tbody>{rows}</tbody></table>")

    if tr is not None and tr.origins and tr.band_hit is not None:
        lean = ""
        if tr.direction_hit is not None and tr.base_up is not None:
            edge = tr.direction_hit - max(tr.base_up, 1 - tr.base_up)
            verdict = ("the lean beat simply betting on the usual direction"
                       if edge > 0.02 else "the lean has no proven edge here. Read the range, not the lean")
            lean = (f" The path's lean was right {tr.direction_hit:.0%} of the time, while price rose "
                    f"{tr.base_up:.0%} of the time anyway: {verdict}.")
        body += (f"<p class='muted'>Track record, replayed on this desk's last {tr.origins:,} candles: the 68% "
                 f"range held {tr.band_hit:.0%} of the time (aim: 68%).{html.escape(lean)}</p>")
    memory = memory_label(f.hurst)
    parts = [f"Hurst {f.hurst:.2f} ({memory})" if f.hurst is not None else "no Hurst reading (short history)",
             "volatility of the last 50 candles"]
    if f.gamma_regime:
        parts.append(f"{f.gamma_regime} dealer gamma ({'narrows' if f.gamma_scale < 1 else 'widens'} the range "
                     f"{abs(1 - f.gamma_scale):.0%}; not testable on history)")
    body += (f"<p class='muted ti-note'>The path is the median of 4,000 simulated paths, not a promise: built from "
             f"{html.escape(', '.join(parts))}. Hollow candles are the path; the dashed lines are the 68% range.</p>")
    return card_html(f"Forecast · {horizon}", body, "neutral")
