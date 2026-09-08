"""Typed market snapshots. Every snapshot carries an availability flag so no
consumer can mistake a missing feed for a real value."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

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


class MarketSnapshot(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    spot: SpotSnapshot
    futures: FuturesSnapshot
    options: OptionsSnapshot
    sentiment: SentimentSnapshot
    generated_at: datetime = Field(default_factory=_now)

    def unavailable(self) -> list[str]:
        out = []
        for name in ("spot", "futures", "options", "sentiment"):
            if not getattr(self, name).available:
                out.append(name)
        return out
