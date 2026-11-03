"""His Section 5.3: the data-versus-candle anomaly trap. Positioning read against what price is doing.

    Bullish squeeze trap     extreme fear plus a crowded short: the crowd is selling in a panic; when the
                             order flow absorbs that selling (price pushed down, buyers taking everything
                             offered) the trap is SET, otherwise it is FORMING
    Liquidation flush trap   extreme greed plus over-leveraged longs (a crowded long with open interest
                             building or at a high): the crowd is euphoric; when the flow meets every bid
                             while price rises the trap is SET, otherwise FORMING

The read is contrarian by construction: the crowded side is the one that gets forced out, so a squeeze trap
implies a long and a flush trap a short. Everything is code over the desk's own numbers (mood, accounts
long, the premium to hold, open interest, taker flow on the desk's chart, the 24-hour move); nothing comes
from a model. Notes are written for the layman summary, so they name no indicator."""
from __future__ import annotations

from dataclasses import dataclass

LONG, SHORT = "LONG", "SHORT"
FEAR_EXTREME, GREED_EXTREME = 25, 75
PREMIUM_CROWDED_LONG = 0.0003      # 8-hour perpetual premium: a high price to stay long
PREMIUM_CROWDED_SHORT = -0.0001    # a negative premium: shorts are paying
LONG_ACCOUNTS_CROWDED = 0.60       # share of accounts long
SHORT_ACCOUNTS_CROWDED = 0.45
OI_HIGH_SHARE = 0.97               # open interest within 3% of its 30-day high counts as "at a high"
OI_HIGH_DAYS = 30
DAY_MS = 86_400_000


@dataclass(frozen=True)
class AnomalyTrap:
    name: str            # Bullish squeeze trap | Liquidation flush trap
    side: str            # LONG | SHORT: the trade the trap implies
    status: str          # set | forming
    trapped: str         # shorts | longs
    mood: str            # extreme fear (18/100)
    crowd: tuple[str, ...]
    candle: str          # what price did
    flow: str            # what the order flow did against it

    @property
    def headline(self) -> str:
        ing = " + ".join((self.mood,) + self.crowd)
        return (f"{self.name}, {self.status}: {ing}. {self.candle}; {self.flow}. "
                f"The {self.trapped} are the crowd that gets forced out.")

    def to_state(self) -> dict:
        return {"name": self.name, "side": self.side, "status": self.status, "trapped": self.trapped,
                "mood": self.mood, "crowd": list(self.crowd), "candle": self.candle, "flow": self.flow,
                "headline": self.headline}


def long_share(futures) -> float | None:
    """The share of accounts long, from the share itself or the long/short ratio."""
    pct = getattr(futures, "long_account_pct", None)
    ratio = getattr(futures, "long_short_ratio", None)
    if pct is None and ratio is not None:
        pct = ratio / (1 + ratio)
    return pct


def oi_at_high(oi_history, now_ms: int | None) -> bool:
    """Open interest within 3% of its 30-day high (the newest sample against the window's maximum)."""
    rows = [(t, v) for t, v in (oi_history or ()) if v]
    if len(rows) < 2:
        return False
    rows.sort()
    now = now_ms if now_ms is not None else rows[-1][0]
    window = [v for t, v in rows if t >= now - OI_HIGH_DAYS * DAY_MS]
    return bool(window) and rows[-1][1] >= OI_HIGH_SHARE * max(window)


def _candle(px: float | None) -> str:
    if px is None:
        return "the 24-hour move is not readable"
    way = "fell" if px < 0 else "rose"
    return f"BTC {way} {abs(px):.1f}% in 24 hours"


def read(fear_greed: int | None, futures, derivs: dict | None, liquidity_chart: dict | None,
         price_change_24h_pct: float | None, oi_history=None, now_ms: int | None = None) -> AnomalyTrap | None:
    """The trap on this desk, or None. `liquidity_chart` is core.indicators.liquidity.summarise(...) for the
    desk's own chart; `derivs` is desk_derivs.read(...)."""
    if fear_greed is None:
        return None
    share = long_share(futures)
    rate = ((derivs or {}).get("funding") or {}).get("latest_8h_rate")
    td = ((derivs or {}).get("taker_delta") or {}).get("delta_btc")
    cd = ((liquidity_chart or {}).get("cumulative_delta") or {}).get("state")
    px = price_change_24h_pct

    if fear_greed < FEAR_EXTREME:
        crowd = []
        if share is not None and share <= SHORT_ACCOUNTS_CROWDED:
            crowd.append(f"a crowded short ({share:.0%} of accounts long)")
        if rate is not None and rate <= PREMIUM_CROWDED_SHORT:
            crowd.append("shorts paying to hold")
        if not crowd:
            return None
        absorbs = cd == "bullish absorption" or (td is not None and td > 0 and px is not None and px < 0)
        flow = ("price was pushed down but buyers took everything offered" if cd == "bullish absorption" else
                "more buying than selling at market while price fell" if absorbs else
                "no absorption yet: the selling is not being taken")
        return AnomalyTrap("Bullish squeeze trap", LONG, "set" if absorbs else "forming", "shorts",
                           f"extreme fear ({fear_greed}/100)", tuple(crowd), _candle(px), flow)

    if fear_greed > GREED_EXTREME:
        crowd = []
        if share is not None and share >= LONG_ACCOUNTS_CROWDED:
            crowd.append(f"a crowded long ({share:.0%} of accounts long)")
        if rate is not None and rate >= PREMIUM_CROWDED_LONG:
            crowd.append("longs paying a high premium to hold")
        if not crowd:
            return None
        ois = (derivs or {}).get("open_interest") or []
        building = bool(ois) and (ois[-1].get("change_pct") or 0) > 0
        high = oi_at_high(oi_history, now_ms)
        if building:
            crowd.append(f"open positions growing ({ois[-1]['change_pct']:+.1f}% in {ois[-1]['over']})")
        if high:
            crowd.append("open positions at a 30-day high")
        if not (building or high):
            return None                       # a crowded long without leverage building is greed, not a flush
        absorbs = cd == "bearish absorption" or (td is not None and td < 0 and px is not None and px > 0)
        flow = ("price was pushed up but sellers met every bid" if cd == "bearish absorption" else
                "more selling than buying at market while price rose" if absorbs else
                "no absorption yet: the buying is still being taken")
        return AnomalyTrap("Liquidation flush trap", SHORT, "set" if absorbs else "forming", "longs",
                           f"extreme greed ({fear_greed}/100)", tuple(crowd), _candle(px), flow)
    return None


def from_evidence(ev, chart_tf: str, now_ms: int | None = None) -> AnomalyTrap | None:
    """The same read off a confluence Evidence bundle (duck-typed, so this module imports nothing of it)."""
    liq = (getattr(ev, "liquidity", None) or {}).get(chart_tf)
    return read(getattr(ev, "fear_greed", None), getattr(ev, "futures", None), getattr(ev, "derivs", None), liq,
                getattr(ev, "price_change_24h_pct", None), getattr(ev, "oi_history", ()), now_ms)
