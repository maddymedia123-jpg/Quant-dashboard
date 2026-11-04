"""The dollar against the majors from the ECB reference rates via frankfurter.dev (keyless): USD/JPY and the
yen carry trade (roadmap item 10), and the dollar index reconstructed from six of those rates (item 11).

USD/JPY and the carry (item 10, owner's GO, 5 November). "Japanese Yen Carry-Trade Unwind: monitors BoJ
rate hikes vs. US yields". The carry trade borrows yen at the Bank of Japan's near-zero rate and buys
dollar assets, bitcoin among them. It unwinds when the yen strengthens fast (USD/JPY falls) or the rate
gap closes: August 2024 took bitcoin from about 65,000 to 49,000 in four days on exactly that. So two
things are read:

* the pace of USD/JPY: a fall of UNWIND_WEEK_PCT or more in a week of sessions is the unwind warning;
* the carry itself: the US funds rate (EFFR, from the macro feed) minus the Bank of Japan's policy rate.

The BoJ rate has no free feed worth trusting (it changes a few times a year, at scheduled meetings), so
it is HAND-SET by the owner in the environment as BOJ_POLICY_RATE_PCT (for example 0.75). Unset, the
carry differential is not shown and the read rests on the pace of USD/JPY alone; it is never guessed.

The dollar index (item 11, owner's GO, 5 November). "Real-Time DXY ... DXY spike + BTC rally = false
breakout signal". ICE's DXY has no free feed, and FRED's broad dollar index is weekly and refused on the
dashboard host. But DXY is a published formula: a constant times six dollar rates raised to fixed weights
(euro 57.6%, yen 13.6%, sterling 11.9%, Canadian dollar 9.1%, krona 4.2%, franc 3.6%), and the ECB
publishes all six every business day. So the index is REBUILT here from the same one request: the same
formula on the ECB's 14:15 CET fixes rather than ICE's live spot, which puts it within a few tenths of
the real print and a day behind it by design. It is labelled reconstructed wherever it is shown. The
read the owner asked for is the divergence: the dollar spiking over a week of sessions while bitcoin
rallies over the same sessions is the false-breakout signal.

Found by calling the feed: frankfurter.app answers 301 (the host moved to api.frankfurter.dev/v1); the
range form `/v1/2026-08-25..?base=USD&symbols=EUR,JPY,...` returns {"rates": {"2026-10-20": {"EUR":
0.86, "JPY": 158.13, ...}, ...}} on ECB business days only (no weekends, no ECB holidays), so a reading
three days old over a weekend is normal and the staleness threshold allows for it. With base=USD every
quote is units of the currency per dollar, which is the orientation DXY's yen, CAD, krona and franc
terms want; the euro and sterling terms want dollars per unit, so their weights are applied to the
reciprocal, which is the same as applying the positive weight to the per-dollar quote."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

import httpx

from core.data.types import FxSnapshot

log = logging.getLogger(__name__)

URL = "https://api.frankfurter.dev/v1/{start}.."
HEADERS = {"User-Agent": "trap-intel-dashboard/1.0"}
MAX_BYTES = 256 * 1024
WINDOW_DAYS = 45
WEEK_BACK_OBSERVATIONS = 5        # a week of sessions, as the macro feed counts it
MONTH_BACK_OBSERVATIONS = 21
STALE_DAYS = 4                    # ECB business days: a Friday fix read on Monday is 3 days old
PLAUSIBLE = (50.0, 400.0)         # USD/JPY has lived between 75 and 162 since 1990
UNWIND_WEEK_PCT = 1.5             # the yen strengthening this fast in a week is the unwind warning
BOJ_ENV = "BOJ_POLICY_RATE_PCT"

# ICE US Dollar Index: 50.14348112 x EURUSD^-0.576 x USDJPY^0.136 x GBPUSD^-0.119 x USDCAD^0.091
# x USDSEK^0.042 x USDCHF^0.036. On per-dollar quotes every weight is positive (see the module note).
DXY_CONSTANT = 50.14348112
DXY_WEIGHTS = {"EUR": 0.576, "JPY": 0.136, "GBP": 0.119, "CAD": 0.091, "SEK": 0.042, "CHF": 0.036}
MAJORS: tuple[str, ...] = tuple(DXY_WEIGHTS)
DXY_PLAUSIBLE = (60.0, 160.0)     # DXY has lived between 70 and 165 since 1973
DXY_SPIKE_WEEK_PCT = 1.0          # a week's move in the index that counts as a spike (or a slide)
BTC_RALLY_WEEK_PCT = 3.0          # a bitcoin move over the same sessions that counts as a rally


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _date_ms(text: str) -> int | None:
    try:
        d = datetime.strptime(str(text).strip()[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None
    return int(d.timestamp() * 1000)


def parse_rates(payload: dict) -> dict[str, list[tuple[int, float]]]:
    """{currency: [(date_ms, units per dollar), ...] oldest first} for every currency in a range
    response. Refuses a response for another base currency - the goldprice.dev wrong-currency bug, in
    another feed."""
    if not isinstance(payload, dict):
        raise ValueError("frankfurter: not an object")
    base = str(payload.get("base") or "").upper()
    if base != "USD":
        raise ValueError(f"frankfurter: asked for USD, the response is base {base or 'missing'!r}")
    rates = payload.get("rates")
    if not isinstance(rates, dict) or not rates:
        raise ValueError("frankfurter: no rates in the window")
    out: dict[str, list[tuple[int, float]]] = {}
    for day, quotes in rates.items():
        when = _date_ms(day)
        if when is None or not isinstance(quotes, dict):
            continue
        for ccy, value in quotes.items():
            try:
                out.setdefault(str(ccy).upper(), []).append((when, float(value)))
            except (TypeError, ValueError):
                continue
    if not out:
        raise ValueError("frankfurter: no quotes in the window")
    for rows in out.values():
        rows.sort()
    return out


def parse_range(payload: dict) -> list[tuple[int, float]]:
    """(date_ms, USD/JPY) for every day in a range response, oldest first."""
    rates = parse_rates(payload)
    if "JPY" not in rates:
        raise ValueError("frankfurter: no JPY quotes in the window")
    return rates["JPY"]


def dxy_series(rates: dict[str, list[tuple[int, float]]]) -> list[tuple[int, float]]:
    """The reconstructed index for every day on which ALL six rates published, oldest first. A day with
    one rate missing is skipped rather than filled from the day before: the index would describe no
    day at all."""
    by_day: dict[int, dict[str, float]] = {}
    for ccy in MAJORS:
        for when, value in rates.get(ccy, []):
            by_day.setdefault(when, {})[ccy] = value
    out = []
    for when in sorted(by_day):
        quotes = by_day[when]
        if any(ccy not in quotes or quotes[ccy] <= 0 for ccy in MAJORS):
            continue
        index = DXY_CONSTANT
        for ccy, weight in DXY_WEIGHTS.items():
            index *= quotes[ccy] ** weight
        out.append((when, round(index, 3)))
    return out


def boj_rate(env=None) -> float | None:
    """The hand-set Bank of Japan policy rate, or None when the owner has not set it."""
    raw = (env if env is not None else os.environ).get(BOJ_ENV, "")
    raw = str(raw).strip().rstrip("%")
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        log.warning("%s=%r is not a number; the BoJ rate is treated as unset", BOJ_ENV, raw)
        return None
    if not -1.0 <= value <= 10.0:
        log.warning("%s=%s is not a policy rate anybody has seen; treated as unset", BOJ_ENV, value)
        return None
    return value


def _back(rows: list[tuple[int, float]], sessions: int) -> float | None:
    return rows[-1 - sessions][1] if len(rows) > sessions else None


def _pct(now: float, then: float | None) -> float | None:
    return round((now / then - 1.0) * 100.0, 2) if then else None


def build(rows: list[tuple[int, float]], now_ms: int, boj_rate_pct: float | None = None,
          source: str = "frankfurter", dxy_rows: list[tuple[int, float]] | None = None) -> FxSnapshot:
    if not rows:
        return FxSnapshot.unavailable(source, "no USD/JPY quotes in the window")
    as_of, rate = rows[-1]
    if not PLAUSIBLE[0] <= rate <= PLAUSIBLE[1]:
        return FxSnapshot.unavailable(source, f"USD/JPY {rate} is outside the plausible {PLAUSIBLE[0]:.0f}-{PLAUSIBLE[1]:.0f} range")
    week, month = _back(rows, WEEK_BACK_OBSERVATIONS), _back(rows, MONTH_BACK_OBSERVATIONS)
    age_days = (now_ms - as_of) / 86_400_000
    snap = FxSnapshot(
        source=source, rate=rate, as_of_ms=as_of,
        week_ago=week, change_1w_pct=_pct(rate, week),
        month_ago=month, change_1m_pct=_pct(rate, month),
        stale=age_days > STALE_DAYS, age_days=round(age_days, 1),
        boj_rate_pct=boj_rate_pct,
    )
    if dxy_rows is None:
        return snap
    if not dxy_rows:
        return snap.model_copy(update={"dxy_note": "no day in the window has all six rates"})
    dxy_as_of, dxy = dxy_rows[-1]
    if not DXY_PLAUSIBLE[0] <= dxy <= DXY_PLAUSIBLE[1]:
        return snap.model_copy(update={"dxy_note": f"reconstructed index {dxy} is outside the plausible "
                                                   f"{DXY_PLAUSIBLE[0]:.0f}-{DXY_PLAUSIBLE[1]:.0f} range"})
    dw, dm = _back(dxy_rows, WEEK_BACK_OBSERVATIONS), _back(dxy_rows, MONTH_BACK_OBSERVATIONS)
    return snap.model_copy(update={
        "dxy": dxy, "dxy_as_of_ms": dxy_as_of,
        "dxy_week_ago": dw, "dxy_change_1w_pct": _pct(dxy, dw),
        "dxy_month_ago": dm, "dxy_change_1m_pct": _pct(dxy, dm),
    })


async def fetch_fx(client: httpx.AsyncClient, now_ms: int | None = None) -> FxSnapshot:
    now = now_ms or _now_ms()
    start = (datetime.fromtimestamp(now / 1000, timezone.utc) - timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
    r = await client.get(URL.format(start=start), params={"base": "USD", "symbols": ",".join(MAJORS)},
                         headers=HEADERS, follow_redirects=True)
    r.raise_for_status()
    if len(r.content) > MAX_BYTES:
        raise ValueError(f"frankfurter: {len(r.content) / 1024:.0f} KB for a {WINDOW_DAYS}-day window")
    rates = parse_rates(r.json())
    if "JPY" not in rates:
        raise ValueError("frankfurter: no JPY quotes in the window")
    return build(rates["JPY"], now, boj_rate(), dxy_rows=dxy_series(rates))


def yen_read(fx: FxSnapshot, effr_pct: float | None = None) -> dict:
    """The carry-trade read: {"state": unwind_risk | carry_on | quiet | unknown, "carry_pct", "note"}."""
    if not getattr(fx, "available", False) or fx.rate is None:
        return {"state": "unknown", "carry_pct": None, "note": "no USD/JPY feed"}
    carry = None
    if effr_pct is not None and fx.boj_rate_pct is not None:
        carry = round(effr_pct - fx.boj_rate_pct, 2)
    carry_text = (f"; the carry (US funds rate {effr_pct:.2f}% minus BoJ {fx.boj_rate_pct:.2f}%) is {carry:+.2f} pp"
                  if carry is not None else "; BoJ rate not set, so the carry gap is not shown")
    week = fx.change_1w_pct
    if week is None:
        return {"state": "unknown", "carry_pct": carry,
                "note": f"USD/JPY {fx.rate:.2f}, no week-ago quote yet{carry_text}"}
    if week <= -UNWIND_WEEK_PCT:
        return {"state": "unwind_risk", "carry_pct": carry,
                "note": (f"USD/JPY {fx.rate:.2f}, down {abs(week):.1f}% in a week: the yen is strengthening fast, "
                         f"carry-trade unwind risk (yen-funded longs in bitcoin get cut){carry_text}")}
    if week >= UNWIND_WEEK_PCT:
        return {"state": "carry_on", "carry_pct": carry,
                "note": (f"USD/JPY {fx.rate:.2f}, up {week:.1f}% in a week: the yen is weakening, the carry trade "
                         f"is on and supportive of risk{carry_text}")}
    return {"state": "quiet", "carry_pct": carry,
            "note": f"USD/JPY {fx.rate:.2f}, {week:+.1f}% on the week: the yen is quiet, no carry signal{carry_text}"}


def btc_sessions_change_pct(frames: dict, sessions: int = WEEK_BACK_OBSERVATIONS) -> float | None:
    """Bitcoin's move over the last `sessions` completed daily candles, in percent, from the spot frames;
    None when the daily frame is missing or too short. The forming candle is the last row and is not
    counted as a session."""
    try:
        f = frames.get("1d") if frames else None
        if f is None or len(f) < sessions + 2:
            return None
        closes = [float(c) for c in f["close"].tolist()]
    except Exception:  # noqa: BLE001 - a malformed frame is no read, not a crash
        return None
    last, then = closes[-2], closes[-2 - sessions]
    return _pct(last, then) if then else None


def dollar_read(fx: FxSnapshot, btc_week_pct: float | None = None) -> dict:
    """The reconstructed dollar index against bitcoin over the same week of sessions:
    {"state": spike_divergence | spike | slide | quiet | unknown, "note"}."""
    if not getattr(fx, "available", False) or getattr(fx, "dxy", None) is None:
        note = getattr(fx, "dxy_note", None) or getattr(fx, "error", None) or "no dollar index"
        return {"state": "unknown", "note": note}
    week = fx.dxy_change_1w_pct
    if week is None:
        return {"state": "unknown", "note": f"dollar index {fx.dxy:.2f} (reconstructed), no week-ago value yet"}
    btc = ("" if btc_week_pct is None else
           f"; bitcoin {'+' if btc_week_pct >= 0 else ''}{btc_week_pct:.1f}% over the same sessions")
    if week >= DXY_SPIKE_WEEK_PCT:
        if btc_week_pct is not None and btc_week_pct >= BTC_RALLY_WEEK_PCT:
            return {"state": "spike_divergence",
                    "note": (f"dollar index {fx.dxy:.2f}, up {week:.1f}% in a week while bitcoin rallied "
                             f"{btc_week_pct:+.1f}% over the same sessions: a dollar spike with a bitcoin rally "
                             "is the false-breakout signal; treat the rally's breaks as suspect until the dollar turns")}
        return {"state": "spike",
                "note": f"dollar index {fx.dxy:.2f}, up {week:.1f}% in a week: a dollar spike, a headwind for bitcoin{btc}"}
    if week <= -DXY_SPIKE_WEEK_PCT:
        return {"state": "slide",
                "note": f"dollar index {fx.dxy:.2f}, down {abs(week):.1f}% in a week: the dollar is sliding, a tailwind for bitcoin{btc}"}
    return {"state": "quiet",
            "note": f"dollar index {fx.dxy:.2f}, {week:+.1f}% on the week: the dollar is quiet, no signal{btc}"}
