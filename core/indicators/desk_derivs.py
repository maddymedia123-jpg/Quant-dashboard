"""Derivatives measured over each desk's own window, not one 24-hour block for every desk."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS
BAR_MS = {"15m": 900_000, "1h": HOUR_MS, "4h": 4 * HOUR_MS, "1d": DAY_MS}
SPAN_MS = {"1h": HOUR_MS, "4h": 4 * HOUR_MS, "24h": DAY_MS, "7d": 7 * DAY_MS, "30d": 30 * DAY_MS}
ROWS_FOR = {"1h": "1h", "4h": "1h", "24h": "1h", "7d": "4h", "30d": "1d"}
EXPIRY_HOUR_UTC = 8


@dataclass(frozen=True)
class Window:
    label: str
    oi: tuple[str, ...]
    flow: str
    funding: str | None
    expiry_days: float
    expiry_label: str


WINDOWS: dict[str, Window] = {
    "live": Window("the next 4 hours", ("1h", "4h"), "4h", None, 1, "24 hours"),
    "intraday": Window("the next 24 hours", ("4h", "24h"), "24h", "24h", 3, "3 days"),
    "weekly": Window("the next 7 days", ("7d",), "7d", "7d", 8, "8 days"),
    "monthly": Window("the next 30 days", ("30d",), "30d", "30d", 100, "100 days (monthly and quarterly)"),
}
SOURCE = "Gate.io BTC_USDT perpetual (one venue: read the trend, not the market total)"


def _rows(flow, span: str):
    tf = ROWS_FOR[span]
    df = (getattr(flow, "frames", None) or {}).get(tf)
    if df is None or not len(df):
        return None, 0
    return df, max(1, SPAN_MS[span] // BAR_MS[tf])


def _oi_change(flow, span: str) -> dict | None:
    df, n = _rows(flow, span)
    if df is None or "oi_btc" not in df:
        return None
    oi = df["oi_btc"].dropna()
    if len(oi) < n + 1:
        return None
    now, then = float(oi.iloc[-1]), float(oi.iloc[-1 - n])
    if then <= 0:
        return None
    out = {
        "over": span,
        "now_btc": round(now),
        "change_btc": round(now - then),
        "change_pct": round((now / then - 1) * 100, 2),
    }
    if len(oi) >= 2 * n + 1 and float(oi.iloc[-1 - 2 * n]) > 0:
        out["prior_change_pct"] = round((then / float(oi.iloc[-1 - 2 * n]) - 1) * 100, 2)
    return out


def _flow_sums(flow, span: str) -> tuple[dict | None, dict | None]:
    df, n = _rows(flow, span)
    if df is None or len(df) < n:
        return None, None
    last = df.tail(n)
    delta = {
        "over": span,
        "delta_btc": round(float(last["delta"].sum()), 1),
        "taker_buy_btc": round(float(last["buy"].sum()), 1),
        "taker_sell_btc": round(float(last["sell"].sum()), 1),
    }
    liqs = {
        "over": span,
        "longs_usd": round(float(last["long_liq_usd"].sum())),
        "shorts_usd": round(float(last["short_liq_usd"].sum())),
    }
    return delta, liqs


def _funding(flow, span: str | None) -> dict | None:
    df, n = _rows(flow, span or "1h")
    if df is None or "funding_rate" not in df:
        return None
    rates = df["funding_rate"].dropna()
    if not len(rates):
        return None
    hourly, _ = _rows(flow, "1h")
    latest = hourly["funding_rate"].dropna() if hourly is not None and "funding_rate" in hourly else rates
    out = {"latest_8h_rate": float((latest if len(latest) else rates).iloc[-1])}
    if span is not None and len(rates) >= n:
        out.update({"over": span, "mean_8h_rate": float(rates.tail(n).mean())})
    return out


def _expiry_ms(expiry: str) -> int:
    s = str(expiry).strip()
    if len(s) == 8 and s.isdigit():
        # Handle YYMMDD fallback format if present (e.g., 261225 -> 2026-12-25)
        s = f"20{s[:2]}-{s[2:4]}-{s[4:6]}"
    d = datetime.fromisoformat(s).replace(hour=EXPIRY_HOUR_UTC, tzinfo=timezone.utc)
    return int(d.timestamp() * 1000)


def _expiries(options, days: float, now_ms: int, price: float | None, limit: int | None = 4) -> list[dict]:
    if not getattr(options, "available", False):
        return []
    out = []
    for e in options.expiries or []:
        if not isinstance(e, dict) or not e.get("expiry"):
            continue
        try:
            at = _expiry_ms(str(e["expiry"]))
        except (TypeError, ValueError):
            continue
        if now_ms < at <= now_ms + days * DAY_MS:
            oi_btc = float(e.get("open_interest_btc") or ((e.get("call_oi") or 0) + (e.get("put_oi") or 0)))
            row = {
                "expiry": e["expiry"],
                "hours_left": round((at - now_ms) / HOUR_MS, 1),
                "max_pain": e.get("max_pain"),
                "open_interest_btc": round(oi_btc),
            }
            if price and e.get("max_pain"):
                row["max_pain_vs_price_pct"] = round((float(e["max_pain"]) / price - 1) * 100, 2)
            out.append(row)
    return sorted(out, key=lambda r: -r["open_interest_btc"])[:limit]


def oi_history(m) -> list[tuple[int, float]]:
    flow = getattr(m, "taker_flow", None)
    frames = (getattr(flow, "frames", None) or {}) if getattr(flow, "available", False) else {}
    out: list[tuple[int, float]] = []
    for tf in ("1d", "4h", "1h"):
        df = frames.get(tf)
        if df is None or "oi_btc" not in df:
            continue
        rows = [(int(t), float(v)) for t, v in zip(df["timestamp"], df["oi_btc"]) if v == v and v is not None]
        if rows:
            out = [r for r in out if r[0] < rows[0][0]] + rows
    return out or list(getattr(m.futures, "oi_history", []) or [])


def gamma_for(category: str | None, m) -> dict | None:
    """Resolve gamma profile from active or fallback options feed."""
    opts = getattr(m, "options", None)
    if not getattr(opts, "available", False):
        return None
    own = (getattr(opts, "gamma_by_desk", None) or {}).get(category) if category else None
    g = own if (isinstance(own, dict) and own.get("available")) else getattr(opts, "gamma", None)
    if isinstance(g, dict) and g.get("available"):
        out = dict(g)
        if not out.get("source") and getattr(opts, "source", None):
            out["source"] = opts.source
        return out
    # Fallback synthesis if options feed is available with max_pain/walls but no prebuilt gamma dict
    mp = getattr(opts, "max_pain", None)
    if mp is not None:
        return {
            "available": True,
            "regime": "neutral",
            "zero_gamma_level": float(mp),
            "max_gamma_strike": float(mp),
            "call_wall": None,
            "put_wall": None,
            "net_gex_usd_per_1pct": None,
            "options_used": len(getattr(opts, "expiries", None) or []),
            "expiries": getattr(opts, "expiries", None) or [],
            "horizon_days": WINDOWS.get(category or "", WINDOWS["monthly"]).expiry_days,
            "source": getattr(opts, "source", "fallback options"),
            "note": "Fallback options max-pain anchor (full gamma surface unavailable).",
        }
    return None


def read(category: str, m, now_ms: int | None = None) -> dict:
    w = WINDOWS[category]
    now = now_ms if now_ms is not None else int(m.generated_at.timestamp() * 1000)
    flow = getattr(m, "taker_flow", None)
    flow = flow if getattr(flow, "available", False) else None
    out: dict = {"window": w.label, "source": SOURCE, "unavailable": []}

    ois = [c for c in (_oi_change(flow, s) for s in w.oi) if c is not None] if flow else []
    if ois:
        out["open_interest"] = ois
    else:
        out["unavailable"].append("open_interest")

    delta, liqs = _flow_sums(flow, w.flow) if flow else (None, None)
    for key, value in (("taker_delta", delta), ("liquidations", liqs)):
        if value is not None:
            out[key] = value
        else:
            out["unavailable"].append(key)

    fund = _funding(flow, w.funding) if flow else None
    if fund is not None:
        out["funding"] = fund
    else:
        out["unavailable"].append("funding")

    gamma = gamma_for(category, m)
    if gamma is not None:
        out["gamma"] = gamma

    exp = _expiries(m.options, w.expiry_days, now, getattr(m.spot, "last", None))
    if exp:
        out["options_expiries"] = exp
        if exp[0].get("max_pain") is not None:
            out["max_pain"] = {
                "expiry": exp[0]["expiry"],
                "price": exp[0]["max_pain"],
                "why": f"the biggest options expiry in the next {w.expiry_label}",
            }
    else:
        later = _expiries(m.options, 400, now, getattr(m.spot, "last", None), limit=None)
        nxt = min(later, key=lambda r: r["hours_left"]) if later else None
        if nxt and nxt.get("max_pain") is not None:
            out["max_pain"] = {
                "expiry": nxt["expiry"],
                "price": nxt["max_pain"],
                "why": f"no expiry in the next {w.expiry_label}; this is the next one, after it",
            }
        elif getattr(m.options, "available", False) and getattr(m.options, "max_pain", None) is not None:
            out["max_pain"] = {
                "expiry": getattr(m.options, "max_pain_expiry", None) or "fallback",
                "price": getattr(m.options, "max_pain", None),
                "why": f"fallback options chain ({getattr(m.options, 'source', 'options')})",
            }
        else:
            out["unavailable"].append("options_expiries")
    return out


def match_window(derivatives: dict, category: str, m, now_ms: int) -> None:
    derivatives["matched_window"] = read(category, m, now_ms)
    gamma = gamma_for(category, m)
    if "options" in derivatives and gamma is not None:
        derivatives["options"]["gamma"] = gamma
    if category != "intraday":
        (derivatives.get("futures") or {}).pop("oi_change_24h_pct", None)
        derivatives.pop("liquidations", None)
    if category != "monthly":
        for key in ("max_pain", "max_pain_expiry"):
            (derivatives.get("options") or {}).pop(key, None)
