"""Typed market snapshots. Every snapshot carries an availability flag so no
consumer can mistake a missing feed for a real value."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, ClassVar, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Availability(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    available: bool = True
    source: str = ""
    fetched_at: Optional[datetime] = Field(default_factory=_now)
    error: Optional[str] = None

    @classmethod
    def unavailable(cls, source: str, error: str):
        return cls(available=False, source=source, error=error)


class SpotSnapshot(Availability):
    symbol: str = "BTC/USD"
    last: Optional[float] = None
    frames: dict[str, pd.DataFrame] = Field(default_factory=dict)


class FuturesSnapshot(Availability):
    funding_rate: Optional[float] = None          # latest 8h rate, e.g. 0.0001 = 0.01%
    funding_7d_mean: Optional[float] = None
    open_interest: Optional[float] = None         # contracts (BTC)
    open_interest_usd: Optional[float] = None
    oi_change_24h_pct: Optional[float] = None
    long_short_ratio: Optional[float] = None      # accounts long / short
    long_account_pct: Optional[float] = None      # 0..1
    taker_buy_sell_ratio: Optional[float] = None
    mark_price: Optional[float] = None
    oi_history: list[tuple[int, float]] = Field(default_factory=list)  # (ms, contracts)


class OptionsSnapshot(Availability):
    underlying_price: Optional[float] = None
    max_pain: Optional[float] = None
    max_pain_expiry: Optional[str] = None
    put_call_oi_ratio: Optional[float] = None
    put_call_volume_ratio: Optional[float] = None
    iv_atm: Optional[float] = None                # percent
    iv_skew: Optional[float] = None               # put IV - call IV (percent points), OTM 5-15%
    total_call_oi: Optional[float] = None
    total_put_oi: Optional[float] = None
    expiries: list[dict[str, Any]] = Field(default_factory=list)


class SentimentSnapshot(Availability):
    fear_greed: Optional[int] = None
    classification: Optional[str] = None
    fear_greed_prev: Optional[int] = None


class HyperliquidSnapshot(Availability):
    funding_rate_1h: Optional[float] = None
    open_interest: Optional[float] = None
    mark_price: Optional[float] = None
    oracle_price: Optional[float] = None


class LiquidationsSnapshot(Availability):
    window: str = "24h"
    total_usd: Optional[float] = None
    long_usd: Optional[float] = None
    short_usd: Optional[float] = None
    long_count: Optional[int] = None
    short_count: Optional[int] = None
    last_hour_usd: Optional[float] = None
    btc_usd: Optional[float] = None
    btc_long_usd: Optional[float] = None
    btc_short_usd: Optional[float] = None
    hourly: list[dict[str, Any]] = Field(default_factory=list)
    biggest: Optional[dict[str, Any]] = None
    venues: list[str] = Field(default_factory=list)


class WhalesSnapshot(Availability):
    window_minutes: Optional[float] = None
    btc_trades: Optional[int] = None
    btc_buy_usd: Optional[float] = None
    btc_sell_usd: Optional[float] = None
    btc_largest: Optional[dict[str, Any]] = None
    funding_by_exchange: dict[str, float] = Field(default_factory=dict)
    oi_by_exchange_usd: dict[str, float] = Field(default_factory=dict)
    radar: list[dict[str, Any]] = Field(default_factory=list)
    btc_radar: Optional[dict[str, Any]] = None


class StablecoinSnapshot(Availability):
    total_usd: Optional[float] = None
    prev_day_usd: Optional[float] = None
    change_24h_pct: Optional[float] = None
    top: list[dict[str, Any]] = Field(default_factory=list)


class CalendarSnapshot(Availability):
    """Economic calendar events: each {title, country, impact, time_ms, forecast, previous}."""
    events: list[dict[str, Any]] = Field(default_factory=list)

    def upcoming(self, now_ms: int, within_ms: int, min_impact: str = "High", countries: tuple[str, ...] = ("USD",)) -> list[dict[str, Any]]:
        rank = {"Low": 1, "Medium": 2, "High": 3, "Holiday": 0}
        need = rank.get(min_impact, 3)
        return [e for e in self.events
                if e.get("time_ms") is not None and now_ms - 3_600_000 <= e["time_ms"] <= now_ms + within_ms
                and rank.get(e.get("impact"), 0) >= need and (not countries or e.get("country") in countries)]


class MarketSnapshot(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    SOURCES: ClassVar[tuple[str, ...]] = ("spot", "futures", "options", "sentiment", "hyperliquid", "liquidations", "whales", "stablecoins", "calendar")

    spot: SpotSnapshot
    futures: FuturesSnapshot
    options: OptionsSnapshot
    sentiment: SentimentSnapshot
    hyperliquid: HyperliquidSnapshot = Field(default_factory=lambda: HyperliquidSnapshot.unavailable("hyperliquid", "not fetched"))
    liquidations: LiquidationsSnapshot = Field(default_factory=lambda: LiquidationsSnapshot.unavailable("coinlobster", "not fetched"))
    whales: WhalesSnapshot = Field(default_factory=lambda: WhalesSnapshot.unavailable("coinlobster", "not fetched"))
    stablecoins: StablecoinSnapshot = Field(default_factory=lambda: StablecoinSnapshot.unavailable("defillama", "not fetched"))
    calendar: CalendarSnapshot = Field(default_factory=lambda: CalendarSnapshot.unavailable("forexfactory", "not fetched"))
    generated_at: datetime = Field(default_factory=_now)

    def unavailable(self) -> list[str]:
        return [n for n in self.SOURCES if not getattr(self, n).available]
