"""Derivatives measured over each desk's own window, not one 24-hour block for every desk.

Every desk used to read the same snapshot: open interest change over 24 hours, liquidations over 24
hours, the expiry with the most open interest. A weekly forecast leaning on a 24-hour open interest
change is reading the wrong clock (his v15, Section 2). Each desk now reads:

  Live Recon (next 4h)    open interest over 1h and 4h, taker delta and liquidations over 4h, the
                          latest funding rate, options expiring within a day
  Intraday (next 24h)     open interest over 4h and 24h, taker delta and liquidations over 24h,
                          funding over 24h, expiries within three days
  Weekly (next 7 days)    open interest over 7 days and its pace against the 7 days before, taker
                          delta and liquidations over 7 days, funding over 7 days, expiries within 8 days
  Monthly (next 30 days)  the same over 30 days, expiries within 100 days (monthly and quarterly)

Dealer gamma (regime, flip, call and put walls) is built per desk from the same expiry windows.

Open interest, taker delta, liquidations and funding come from the Gate.io BTC_USDT perpetual's own period stats
(one venue: a trend, not the market total). Each window is counted in whole periods ending with the one
in progress, so a 7-day window on 4h rows is the last 42 rows."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS
BAR_MS = {"15m": 900_000, "1h": HOUR_MS, "4h": 4 * HOUR_MS, "1d": DAY_MS}
SPAN_MS = {"1h": HOUR_MS, "4h": 4 * HOUR_MS, "24h": DAY_MS, "7d": 7 * DAY_MS, "30d": 30 * DAY_MS}
# the finest rows that hold the window twice over (300 rows each: 12 days of 1h, 50 of 4h, 300 of 1d)
ROWS_FOR = {"1h": "1h", "4h": "1h", "24h": "1h", "7d": "4h", "30d": "1d"}
EXPIRY_HOUR_UTC = 8                         # Deribit options settle at 08:00 UTC


@dataclass(frozen=True)
class Window:
    label: str                              # how the window reads in a sentence
    oi: tuple[str, ...]                     # open interest changes, shortest first
    flow: str                               # taker delta and liquidations
    funding: str | None                     # None = the latest rate only
    expiry_days: float                      # options expiring within this many days
    expiry_label: str


WINDOWS: dict[str, Window] = {
    "live": Window("the next 4 hours", ("1h", "4h"), "4h", None, 1, "24 hours"),
    "intraday": Window("the next 24 hours", ("4h", "24h"), "24h", "24h", 3, "3 days"),
    "weekly": Window("the next 7 days", ("7d",), "7d", "7d", 8, "8 days"),
    "monthly": Window("the next 30 days", ("30d",), "30d", "30d", 100, "100 days (monthly and quarterly)"),
}
SOURCE = "Gate.io BTC_USDT perpetual (one venue: read the trend, not the market total)"


def _rows(flow, span: str):
    """The frame for this span and how many rows make the span, or (None, 0)."""
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
    out = {"over": span, "now_btc": round(now), "change_btc": round(now - then),
           "change_pct": round((now / then - 1) * 100, 2)}
    if len(oi) >= 2 * n + 1 and float(oi.iloc[-1 - 2 * n]) > 0:
        # pace: the same change one window earlier, so "building faster" or "unwinding" can be read
        out["prior_change_pct"] = round((then / float(oi.iloc[-1 - 2 * n]) - 1) * 100, 2)
    return out


def _flow_sums(flow, span: str) -> tuple[dict | None, dict | None]:
    df, n = _rows(flow, span)
    if df is None or len(df) < n:
        return None, None
    last = df.tail(n)
    delta = {"over": span, "delta_btc": round(float(last["delta"].sum()), 1),
             "taker_buy_btc": round(float(last["buy"].sum()), 1), "taker_sell_btc": round(float(last["sell"].sum()), 1)}
    liqs = {"over": span, "longs_usd": round(float(last["long_liq_usd"].sum())),
            "shorts_usd": round(float(last["short_liq_usd"].sum()))}
    return delta, liqs


def _funding(flow, span: str | None) -> dict | None:
    """The latest rate, and over the desk's window the mean of the rate each period carried (a time-
    weighted mean: an 8-hour rate repeats on every hourly row it covers)."""
    df, n = _rows(flow, span or "1h")
    if df is None or "funding_rate" not in df:
        return None
    rates = df["funding_rate"].dropna()
    if not len(rates):
        return None
    hourly, _ = _rows(flow, "1h")                # the latest rate is the newest stamp, whatever the window
    latest = hourly["funding_rate"].dropna() if hourly is not None and "funding_rate" in hourly else rates
    out = {"latest_8h_rate": float((latest if len(latest) else rates).iloc[-1])}
    if span is not None and len(rates) >= n:
        out.update({"over": span, "mean_8h_rate": float(rates.tail(n).mean())})
    return out


def _expiry_ms(expiry: str) -> int:
    d = datetime.fromisoformat(expiry).replace(hour=EXPIRY_HOUR_UTC, tzinfo=timezone.utc)
    return int(d.timestamp() * 1000)


def _expiries(options, days: float, now_ms: int, price: float | None, limit: int | None = 4) -> list[dict]:
    if not getattr(options, "available", False):
        return []
    out = []
    for e in options.expiries or []:
        try:
            at = _expiry_ms(e["expiry"])
        except (KeyError, TypeError, ValueError):
            continue
        if now_ms < at <= now_ms + days * DAY_MS:
            row = {"expiry": e["expiry"], "hours_left": round((at - now_ms) / HOUR_MS, 1),
                   "max_pain": e.get("max_pain"), "open_interest_btc": round(e.get("call_oi", 0) + e.get("put_oi", 0))}
            if price and e.get("max_pain"):
                row["max_pain_vs_price_pct"] = round((e["max_pain"] / price - 1) * 100, 2)
            out.append(row)
    return sorted(out, key=lambda r: -r["open_interest_btc"])[:limit]


def oi_history(m) -> list[tuple[int, float]]:
    """Open interest (ms, BTC) back ten months: daily rows, then 4h, then hourly for the recent days.

    The futures snapshot holds 24 hourly samples, so a 1w or 1M open interest change read from it had
    nothing to start from and came back empty. Falls back to that snapshot when the period stats are down."""
    flow = getattr(m, "taker_flow", None)
    frames = (getattr(flow, "frames", None) or {}) if getattr(flow, "available", False) else {}
    out: list[tuple[int, float]] = []
    for tf in ("1d", "4h", "1h"):                    # coarse first; each finer series takes over where it starts
        df = frames.get(tf)
        if df is None or "oi_btc" not in df:
            continue
        rows = [(int(t), float(v)) for t, v in zip(df["timestamp"], df["oi_btc"]) if v == v and v is not None]
        if rows:
            out = [r for r in out if r[0] < rows[0][0]] + rows
    return out or list(getattr(m.futures, "oi_history", []) or [])


def gamma_for(category: str | None, m) -> dict | None:
    """The desk's own gamma profile; the 30-day one when the desk has none (an old snapshot) or no desk."""
    opts = getattr(m, "options", None)
    if not getattr(opts, "available", False):
        return None
    own = (getattr(opts, "gamma_by_desk", None) or {}).get(category) if category else None
    return own if own else getattr(opts, "gamma", None)


def read(category: str, m, now_ms: int | None = None) -> dict:
    """The derivatives for this desk, each figure labelled with the window it covers."""
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
        out["max_pain"] = {"expiry": exp[0]["expiry"], "price": exp[0]["max_pain"],
                           "why": f"the biggest options expiry in the next {w.expiry_label}"}
    else:
        # nothing settles inside the window: the next expiry after it, said to be outside it
        later = _expiries(m.options, 400, now, getattr(m.spot, "last", None), limit=None)
        nxt = min(later, key=lambda r: r["hours_left"]) if later else None
        if nxt:
            out["max_pain"] = {"expiry": nxt["expiry"], "price": nxt["max_pain"],
                               "why": f"no expiry in the next {w.expiry_label}; this is the next one, after it"}
        else:
            out["unavailable"].append("options_expiries")
    return out


def match_window(derivatives: dict, category: str, m, now_ms: int) -> None:
    """Give a desk's derivatives payload its own window, in place.

    The matched figures go in as `matched_window`. The snapshot's own 24-hour figures (open interest
    change, liquidations) stay only on the 24-hour desk, and the all-expiries max pain (usually the
    quarterly) only on Monthly: anywhere else they are a different clock from the desk's question."""
    derivatives["matched_window"] = read(category, m, now_ms)
    if "options" in derivatives and gamma_for(category, m) is not None:
        derivatives["options"]["gamma"] = gamma_for(category, m)    # walls from this desk's expiries only
    if category != "intraday":
        (derivatives.get("futures") or {}).pop("oi_change_24h_pct", None)
        derivatives.pop("liquidations", None)
    if category != "monthly":
        for key in ("max_pain", "max_pain_expiry"):
            (derivatives.get("options") or {}).pop(key, None)
