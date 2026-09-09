import numpy as np
import pandas as pd

from core.indicators.volatility import atr, volatility_profile


def _df(n=200, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    close = 100 + rng.normal(0, scale, n).cumsum()
    return pd.DataFrame({"timestamp": np.arange(n) * 60_000, "open": close, "high": close + scale, "low": close - scale, "close": close, "volume": 1.0})


def test_atr_positive_after_warmup():
    a = atr(_df())
    assert np.isnan(a.iloc[3]) and a.iloc[-1] > 0


def test_profile_brackets_price_and_has_regime():
    df = _df()
    vp = volatility_profile(df, horizon_bars=24, horizon_label="next 24h")
    price = float(df["close"].iloc[-1])
    assert vp.exp_low < price < vp.exp_high
    assert vp.regime in {"LOW", "NORMAL", "HIGH"}
    assert vp.atr_pct > 0 and vp.horizon_label == "next 24h"
    assert vp.sigma2_dn < vp.mean20 < vp.sigma2_up


def test_high_regime_when_recent_range_expands():
    df = _df()
    df.loc[df.index[-15:], "high"] += 15.0
    df.loc[df.index[-15:], "low"] -= 15.0
    assert volatility_profile(df, 4, "next hour").regime == "HIGH"
