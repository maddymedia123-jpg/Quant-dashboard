"""ATR, relative volatility regime, and log-return expected range for a horizon."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VolProfile:
    atr: float | None
    atr_pct: float | None
    regime: str            # LOW | NORMAL | HIGH | UNKNOWN
    sigma_pct: float | None
    exp_low: float | None
    exp_high: float | None
    horizon_label: str


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    pc = c.shift()
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def volatility_profile(df: pd.DataFrame, horizon_bars: int, horizon_label: str, ret_window: int = 30) -> VolProfile:
    if len(df) < 20:
        return VolProfile(None, None, "UNKNOWN", None, None, None, horizon_label)
    close = df["close"].astype(float)
    price = float(close.iloc[-1])
    a = atr(df)
    atr_v = float(a.iloc[-1]) if not np.isnan(a.iloc[-1]) else None
    atr_pct_series = (a / close * 100.0)
    atr_pct = float(atr_pct_series.iloc[-1]) if atr_v is not None else None
    med = float(atr_pct_series.tail(100).median()) if atr_pct is not None else None
    if atr_pct is None or med is None or np.isnan(med) or med == 0:
        regime = "UNKNOWN"
    elif atr_pct < 0.75 * med:
        regime = "LOW"
    elif atr_pct > 1.5 * med:
        regime = "HIGH"
    else:
        regime = "NORMAL"
    lr = np.log(close).diff().tail(ret_window)
    sigma = float(lr.std()) * math.sqrt(horizon_bars) if lr.notna().sum() >= 5 else None
    exp_low = price * math.exp(-sigma) if sigma is not None else None
    exp_high = price * math.exp(sigma) if sigma is not None else None
    return VolProfile(atr_v, atr_pct, regime, sigma * 100 if sigma is not None else None, exp_low, exp_high, horizon_label)
