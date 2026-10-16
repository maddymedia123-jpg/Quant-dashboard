"""Buy the rumour, sell the news: how BTC is positioned into each high-impact USD print, and what the print did.

Measured on the 1H candles against BTC's own recent volatility (the standard deviation of hourly log
returns over the week before), so "a big run-up" means big for this market this week, not a fixed percent.

Before a print (inside 48 hours):
* positioned - price has already run at least 1.5 24-hour standard deviations in the last 24 hours. The
  rumour has been traded; a print that only meets forecast tends to be sold (or bought) against the run.
* flat - no run-up to unwind. The print itself is the catalyst.

After a print (up to 48 hours on), once two hourly candles from the print have closed:
* sell_the_news - there was a run-up into it, and price has since given back at least half of it. The
  side that bought the rumour is the trapped side.
* spike_faded - the first-hour reaction was at least two hourly standard deviations, and price has since
  closed back through the pre-print price. The reaction was the trap.
* confirmed - the first-hour reaction was real and price has extended it at least as far again.
* absorbed - nothing lasting: the print was priced.

The calendar carries forecast and previous but no actual, so the surprise is read from the price, not
from the number. Prints at the same minute (CPI m/m and y/y) are one event."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

H = 3_600_000
LOOKBACK_H = 168            # a week of hourly returns sets "normal"
RUNUP_H = 24
PRE_WINDOW_MS = 48 * H      # prints further out are listed, not read
POST_WINDOW_MS = 48 * H
FOLLOW_H = 24               # how long after a print its outcome is judged over
STRETCH_Z = 1.5             # a run-up of 1.5 24h standard deviations counts as positioned (a random walk
                            # does that on about one day in eight; at 1.0 it was one in three)
SPIKE_Z = 2.0               # a first-hour reaction of two hourly standard deviations counts as real
GIVEBACK = 0.5


@dataclass(frozen=True)
class EventRead:
    title: str
    time_ms: int
    state: str                       # positioned, flat, scheduled, reacting, sell_the_news, spike_faded, confirmed, absorbed, no_data
    runup_pct: float | None = None
    runup_z: float | None = None
    spike_pct: float | None = None
    spike_z: float | None = None
    since_pct: float | None = None   # pre-print price to the latest (or 24h-later) close
    expected_move_pct: float | None = None   # one standard deviation over the four hours after the print
    forecast: str | None = None
    previous: str | None = None

    @property
    def trapped(self) -> str | None:
        """Which side the read says is trapped, if any: longs or shorts."""
        if self.state == "sell_the_news" and self.runup_pct is not None:
            return "longs" if self.runup_pct > 0 else "shorts"
        if self.state == "spike_faded" and self.spike_pct is not None:
            return "longs" if self.spike_pct > 0 else "shorts"
        if self.state == "positioned" and self.runup_pct is not None:
            return "longs at risk" if self.runup_pct > 0 else "shorts at risk"
        return None

    @property
    def headline(self) -> str:
        up = (self.runup_pct or 0) > 0
        sp_up = (self.spike_pct or 0) > 0
        return {
            "positioned": (f"Rumour {'bought' if up else 'sold'}: BTC moved {_pct(self.runup_pct)} into the print. "
                           f"A print that only meets forecast risks a sell-the-news {'drop' if up else 'squeeze'}."),
            "flat": "Flat into the print: no run-up to unwind, the print itself is the catalyst.",
            "scheduled": "Scheduled: too far out to read positioning.",
            "reacting": "Released: reacting now, read after two hourly closes.",
            "sell_the_news": (f"{'Sell' if up else 'Buy'} the news: the {_pct(self.runup_pct)} "
                              f"{'run-up' if up else 'sell-off'} into the print has been given back "
                              f"({_pct(self.since_pct)} since the print). A {'bull' if up else 'bear'} trap."),
            "spike_faded": (f"Spike faded: the {_pct(self.spike_pct)} first-hour reaction reversed through the "
                            f"pre-print price. A {'bull' if sp_up else 'bear'} trap."),
            "confirmed": (f"News confirmed: the {_pct(self.spike_pct)} reaction extended to "
                          f"{_pct(self.since_pct)}."),
            "absorbed": "Absorbed: no lasting reaction, the print was priced.",
            "no_data": "No hourly candles around this print.",
        }.get(self.state, self.state)

    def to_state(self) -> dict:
        r = lambda v: None if v is None else round(v, 2)  # noqa: E731
        return {"title": self.title, "time_ms": self.time_ms, "state": self.state, "read": self.headline,
                "trapped": self.trapped, "runup_24h_pct": r(self.runup_pct), "runup_z": r(self.runup_z),
                "first_hour_pct": r(self.spike_pct), "first_hour_z": r(self.spike_z),
                "since_print_pct": r(self.since_pct), "expected_4h_move_pct": r(self.expected_move_pct),
                "forecast": self.forecast, "previous": self.previous}


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{v:+.2f}%"


def clusters(events: list[dict], countries: tuple[str, ...] = ("USD",)) -> list[dict]:
    """High-impact prints grouped by their release minute: one event, its titles joined."""
    out: dict[int, dict] = {}
    for e in events:
        t = e.get("time_ms")
        if t is None or e.get("impact") != "High" or e.get("country") not in countries:
            continue
        c = out.setdefault(t, {"time_ms": t, "titles": [], "forecast": [], "previous": []})
        c["titles"].append(str(e.get("title") or ""))
        if e.get("forecast"):
            c["forecast"].append(f"{e['title']} {e['forecast']}")
        if e.get("previous"):
            c["previous"].append(f"{e['title']} {e['previous']}")
    return [out[t] for t in sorted(out)]


def _open_at(ts: np.ndarray, opens: np.ndarray, t: int) -> float | None:
    """Open of the hourly candle containing t (the price just before anything inside that hour)."""
    i = int(np.searchsorted(ts, t - t % H, "left"))
    return float(opens[i]) if i < len(ts) and ts[i] == t - t % H else None


def _sigma(closes: np.ndarray) -> float | None:
    if len(closes) < 24:
        return None
    r = np.diff(np.log(closes))
    s = float(np.std(r, ddof=1))
    return s if s > 0 else None


def read_event(df: pd.DataFrame | None, cluster: dict, now_ms: int) -> EventRead:
    t = int(cluster["time_ms"])
    base = dict(title=" · ".join(cluster["titles"]), time_ms=t,
                forecast="; ".join(cluster["forecast"]) or None, previous="; ".join(cluster["previous"]) or None)
    if t - now_ms > PRE_WINDOW_MS:
        return EventRead(state="scheduled", **base)
    if df is None or len(df) < 30:
        return EventRead(state="no_data", **base)
    # completed candles only: the last row is forming
    closed = df.iloc[:-1]
    ts = closed["timestamp"].to_numpy(dtype="int64")
    opens = closed["open"].to_numpy(dtype=float)
    closes = closed["close"].to_numpy(dtype=float)

    ref = min(t, now_ms)
    # "normal" is the week before the run-up window, so the run-up does not inflate its own yardstick
    before = closes[(ts + H) <= ref - RUNUP_H * H][-LOOKBACK_H:]
    sigma = _sigma(before)
    if sigma is None:
        return EventRead(state="no_data", **base)
    expected = sigma * math.sqrt(4) * 100.0

    if t > now_ms:
        p_then = _open_at(ts, opens, now_ms - RUNUP_H * H)
        p_now = float(closes[-1])
        if p_then is None:
            return EventRead(state="no_data", expected_move_pct=expected, **base)
        ru = (p_now / p_then - 1.0) * 100.0
        z = math.log(p_now / p_then) / (sigma * math.sqrt(RUNUP_H))
        return EventRead(state="positioned" if abs(z) >= STRETCH_Z else "flat", runup_pct=ru, runup_z=z,
                         expected_move_pct=expected, **base)

    if now_ms - t > POST_WINDOW_MS:
        return EventRead(state="no_data", **base)
    p0 = _open_at(ts, opens, t)
    pb = _open_at(ts, opens, t - RUNUP_H * H)
    if p0 is None or pb is None:
        return EventRead(state="no_data", expected_move_pct=expected, **base)
    after = (ts >= t - t % H) & (ts < t - t % H + FOLLOW_H * H)
    post = closes[after]
    ru = (p0 / pb - 1.0) * 100.0
    ru_z = math.log(p0 / pb) / (sigma * math.sqrt(RUNUP_H))
    common = dict(runup_pct=ru, runup_z=ru_z, expected_move_pct=expected, **base)
    if len(post) < 2:
        return EventRead(state="reacting", **common)
    sp = (post[0] / p0 - 1.0) * 100.0
    sp_z = math.log(post[0] / p0) / sigma
    since = (post[-1] / p0 - 1.0) * 100.0
    common.update(spike_pct=sp, spike_z=sp_z, since_pct=since)

    if abs(ru_z) >= STRETCH_Z and math.copysign(1, ru) * since <= -GIVEBACK * abs(ru):
        return EventRead(state="sell_the_news", **common)
    if abs(sp_z) >= SPIKE_Z and math.copysign(1, sp) * since < 0:
        return EventRead(state="spike_faded", **common)
    if abs(sp_z) >= SPIKE_Z and math.copysign(1, sp) * since >= 2 * abs(sp):
        return EventRead(state="confirmed", **common)
    return EventRead(state="absorbed", **common)


def read_all(df: pd.DataFrame | None, events: list[dict], now_ms: int) -> list[EventRead]:
    """Every high-impact USD print from 48 hours ago onward, oldest first."""
    return [read_event(df, c, now_ms) for c in clusters(events) if c["time_ms"] >= now_ms - POST_WINDOW_MS]
