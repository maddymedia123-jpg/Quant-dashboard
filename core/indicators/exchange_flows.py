"""Reading exchange flows: is BTC arriving on exchanges to be sold, or leaving them to be held?

The 7-day netflow (inflow minus outflow) is measured against its own trailing year, as a z-score, so
"heavy" means heavy for the current market rather than a fixed number of coins. The exchange reserve's
30-day change says which way the stock of sellable coins is trending.

States:
* distribution - 7-day netflow at least 1 standard deviation above normal: coins arriving on exchanges.
* accumulation - at least 1 standard deviation below normal: coins being withdrawn.
* neutral      - in between.

And against price over the same 7 days:
* inflows into strength - distribution while price rose 2% or more: supply arriving into the rally.
* outflows into weakness - accumulation while price fell 2% or more: coins withdrawn on the dip.

What it has been worth, measured on Coin Metrics' own data 2019-01 to 2026-10 (2,845 days): weeks after
heavy inflows (z >= 1) closed up 45% of the time against 53% for all weeks, and averaged -0.21% against
+1.01%; heavy outflows 57% up. The correlation with the next week's return is about zero and the
heavy-inflow sample is ~40 independent weeks. It is context for the weekly and monthly reads, not a
timing signal, and the card says so."""
from __future__ import annotations

import numpy as np

DAY = 86_400_000
Z_HEAVY = 1.0
PRICE_MOVE = 2.0            # percent over 7 days that counts as "price rose / fell"
MIN_DAYS = 187              # half a year of 7-day sums before a z-score means anything
STALE_DAYS = 3
SPIKE_Z = 2.0
TRACK_RECORD = ("Measured 2019-2026: weeks after heavy inflows closed up 45% of the time vs 53% for all "
                "weeks; heavy outflows 57%. Correlation with the next week is about zero - context, not a "
                "timing signal.")


def read(snap, now_ms: int, daily_closes: list[tuple[int, float]] | None = None) -> dict:
    """The judges' and the card's read. `daily_closes` is [(open_ms, close)] for the price check."""
    if not getattr(snap, "available", False) or not snap.days:
        return {"available": False, "note": getattr(snap, "error", None) or "exchange flows unavailable"}
    days = snap.days
    if len(days) < MIN_DAYS:
        return {"available": False, "note": f"only {len(days)} days of exchange flows; {MIN_DAYS} needed"}

    net = np.array([d["net"] for d in days], dtype=float)
    inflow = np.array([d["inflow"] for d in days], dtype=float)
    n7 = np.convolve(net, np.ones(7), "valid")
    hist = n7[-366:-1]
    sd = float(hist.std())
    z7 = None if sd <= 0 else float((n7[-1] - hist.mean()) / sd)
    base = inflow[-91:-1]
    spike = None if base.std() <= 0 else float((inflow[-1] - base.mean()) / base.std())
    res_now, res_30 = days[-1]["reserve"], days[-31]["reserve"] if len(days) > 30 else None
    age = (now_ms - (days[-1]["time_ms"] + DAY)) / DAY        # the day's figures are complete at its close

    state = ("distribution" if z7 is not None and z7 >= Z_HEAVY else
             "accumulation" if z7 is not None and z7 <= -Z_HEAVY else "neutral")
    price_7d = None
    if daily_closes:
        end = days[-1]["time_ms"]
        px = {t: c for t, c in daily_closes}
        a, b = px.get(end - 6 * DAY), px.get(end)
        if a and b:
            price_7d = (b / a - 1) * 100
    divergence = None
    if price_7d is not None and state == "distribution" and price_7d >= PRICE_MOVE:
        divergence = "inflows into strength: supply is arriving into the rally"
    elif price_7d is not None and state == "accumulation" and price_7d <= -PRICE_MOVE:
        divergence = "outflows into weakness: coins are being withdrawn on the dip"

    r = lambda v, n=2: None if v is None else round(v, n)  # noqa: E731
    return {
        "available": True, "state": state, "as_of_ms": days[-1]["time_ms"], "age_days": r(age, 1),
        "stale": age > STALE_DAYS, "preliminary": snap.preliminary,
        "net_1d_btc": r(float(net[-1]), 0), "net_7d_btc": r(float(n7[-1]), 0),
        "net_30d_btc": r(float(net[-30:].sum()), 0), "net_7d_z": r(z7),
        "inflow_1d_btc": r(float(inflow[-1]), 0), "inflow_spike_z": r(spike),
        "inflow_spike": spike is not None and spike >= SPIKE_Z,
        "reserve_btc": r(res_now, 0),
        "reserve_30d_change_btc": None if res_30 is None else r(res_now - res_30, 0),
        "reserve_30d_change_pct": None if not res_30 else r((res_now / res_30 - 1) * 100),
        "price_7d_pct": r(price_7d), "divergence": divergence,
        "scope": ("all exchange flows (Coin Metrics attribution), not only whales; daily. Summed flows and the "
                  "reserve change need not agree: Coin Metrics re-tags exchange wallets over time, so the "
                  "reserve is the stock and the flows are the transfers"),
        "track_record": TRACK_RECORD,
    }
