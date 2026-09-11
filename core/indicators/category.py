"""Compose all indicators into one CategoryAnalysis per timeframe category."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from core.config import CHART_BARS, TF_MINUTES, Category
from core.data.types import FuturesSnapshot
from core.indicators.direction import DirectionCall, direction_call
from core.indicators.ema import EmaStack, ema_stack
from core.indicators.fib_time import FibRetracement, FibTime, fib_retracements, fib_time_zones
from core.indicators.levels import Level, support_resistance
from core.indicators.patterns import Pattern, detect_patterns
from core.indicators.rsi import Divergence, detect_divergences, rsi
from core.indicators.squeeze import SqueezeRisk, squeeze_risk
from core.indicators.trendlines import Trendline, fit_trendlines
from core.indicators.volatility import VolProfile, volatility_profile


@dataclass
class CategoryAnalysis:
    key: str
    label: str
    chart_tf: str
    context_tf: str
    price: float
    chart_df: pd.DataFrame
    ema: EmaStack
    rsi: pd.Series
    divergences: list[Divergence]
    levels: list[Level]
    trendlines: list[Trendline]
    fib: FibTime | None
    vol: VolProfile
    direction: DirectionCall
    squeeze: SqueezeRisk
    price_change_24h_pct: float | None
    ctx_ema: EmaStack | None
    fib_retr: FibRetracement | None = None
    patterns: list[Pattern] = field(default_factory=list)
    reversal: object | None = None   # core.indicators.reversal.ReversalWindow


def price_change_24h(frames: dict[str, pd.DataFrame]) -> float | None:
    for tf in ("1h", "15m", "4h"):
        df = frames.get(tf)
        if df is None:
            continue
        bars = 1440 // TF_MINUTES[tf]
        if len(df) > bars:
            return float(df["close"].iloc[-1] / df["close"].iloc[-1 - bars] - 1.0) * 100.0
    return None


def analyze_category(cat: Category, frames: dict[str, pd.DataFrame], futures: FuturesSnapshot) -> CategoryAnalysis | None:
    df = frames.get(cat.chart_tf)
    if df is None or len(df) < 30:
        return None
    ctx = frames.get(cat.context_tf)
    price = float(df["close"].iloc[-1])
    stack = ema_stack(df)
    r = rsi(df["close"])
    vol = volatility_profile(df, cat.horizon_bars, cat.horizon_label)
    levels = support_resistance(df, vol.atr, lookback=CHART_BARS)
    pc = price_change_24h(frames)
    a = CategoryAnalysis(
        key=cat.key, label=cat.label, chart_tf=cat.chart_tf, context_tf=cat.context_tf, price=price,
        chart_df=df.tail(CHART_BARS).reset_index(drop=True),
        ema=stack,
        rsi=r.tail(CHART_BARS).reset_index(drop=True),
        divergences=detect_divergences(df, r),
        levels=levels,
        trendlines=fit_trendlines(df),
        fib=fib_time_zones(df, atr_value=vol.atr),
        vol=vol,
        direction=direction_call(df, stack, r, levels, vol.atr),
        squeeze=squeeze_risk(futures, pc),
        price_change_24h_pct=pc,
        ctx_ema=ema_stack(ctx) if ctx is not None and len(ctx) >= 30 else None,
        fib_retr=fib_retracements(df),
        patterns=detect_patterns(df, vol.atr),
    )
    from core.indicators.reversal import reversal_window  # local import: reversal reads CategoryAnalysis fields
    a.reversal = reversal_window(a)
    return a
