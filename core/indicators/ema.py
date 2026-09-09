"""EMA 21/50/100/200 and stack state."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

DEFAULT_PERIODS = (21, 50, 100, 200)


@dataclass
class EmaStack:
    values: dict[int, float | None]
    state: str  # BULL | BEAR | MIXED
    series: dict[int, pd.Series] = field(default_factory=dict)


def ema(close: pd.Series, n: int) -> pd.Series:
    return close.ewm(span=n, adjust=False).mean()


def ema_stack(df: pd.DataFrame, periods: tuple[int, ...] = DEFAULT_PERIODS) -> EmaStack:
    close = df["close"].astype(float)
    series: dict[int, pd.Series] = {}
    values: dict[int, float | None] = {}
    for n in periods:
        if len(close) >= n:
            s = ema(close, n)
            series[n] = s
            values[n] = float(s.iloc[-1])
        else:
            values[n] = None
    state = "MIXED"
    if all(v is not None for v in values.values()):
        seq = [float(close.iloc[-1])] + [values[n] for n in periods]
        if all(a > b for a, b in zip(seq, seq[1:])):
            state = "BULL"
        elif all(a < b for a, b in zip(seq, seq[1:])):
            state = "BEAR"
    return EmaStack(values=values, state=state, series=series)
