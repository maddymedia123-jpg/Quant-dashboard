"""The Volatility Matrix the new spec injects into every desk and the TRAP head.

Realized vol, implied vol, ATR, multi-sigma bands and the Hurst exponent. Hurst is checked against
series whose behaviour is known by construction: a random walk must land near 0.5, a trending series
above it, a mean-reverting one below. The spec's own snippet estimates it with a mis-scaled formula, so
these tests exist to keep ours honest rather than to match that code."""
import numpy as np
import pandas as pd
import pytest

from core.data.types import OptionsSnapshot
from core.indicators import volatility as v


def frame(closes, highs=None, lows=None) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    highs = closes * 1.002 if highs is None else np.asarray(highs, dtype=float)
    lows = closes * 0.998 if lows is None else np.asarray(lows, dtype=float)
    return pd.DataFrame({
        "timestamp": [1_700_000_000_000 + i * 3_600_000 for i in range(len(closes))],
        "open": closes, "high": highs, "low": lows, "close": closes,
        "volume": np.ones(len(closes)),
    })


def prices_from(returns) -> np.ndarray:
    return 80_000.0 * np.exp(np.cumsum(np.asarray(returns, dtype=float)))


def ar1(phi: float, n: int = 800, sigma: float = 0.004, seed: int = 7) -> np.ndarray:
    """Returns with autocorrelation phi: positive persists (trends), negative mean-reverts."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, sigma, n)
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + eps[i]
    return out


# ---------- Hurst ----------
def test_a_random_walk_sits_near_one_half():
    rng = np.random.default_rng(11)
    h = v.hurst(prices_from(rng.normal(0.0, 0.004, 900)))
    assert h is not None and 0.40 <= h <= 0.60, h


def test_a_trending_series_reads_above_one_half():
    h = v.hurst(prices_from(ar1(0.6)))
    assert h is not None and h > 0.58, h


def test_a_mean_reverting_series_reads_below_one_half():
    h = v.hurst(prices_from(ar1(-0.6)))
    assert h is not None and h < 0.42, h


def test_hurst_declines_to_answer_rather_than_guessing():
    rng = np.random.default_rng(3)
    assert v.hurst(prices_from(rng.normal(0, 0.004, 20))) is None, "too few bars"
    assert v.hurst(np.full(500, 80_000.0)) is None, "a flat series has no memory to measure"


def test_memory_label_follows_the_exponent():
    assert v.memory_label(0.70) == "trending"
    assert v.memory_label(0.30) == "mean-reverting"
    assert v.memory_label(0.50) == "random walk"
    assert v.memory_label(None) == "unknown"


# ---------- realized volatility ----------
def test_realized_volatility_is_annualised_from_the_timeframe():
    rng = np.random.default_rng(5)
    df = frame(prices_from(rng.normal(0.0, 0.01, 600)))
    bv = v.realized_vol_annual(df, "1h")
    expected = 0.01 * np.sqrt(365 * 24) * 100.0          # crypto trades every hour of the year
    assert bv is not None and abs(bv - expected) / expected < 0.15, (bv, expected)
    assert v.realized_vol_annual(df, "1d") < bv, "the same returns annualise lower on a slower chart"


def test_realized_volatility_needs_enough_returns():
    assert v.realized_vol_annual(frame(prices_from([0.001] * 5)), "1h") is None


# ---------- sigma bands ----------
def test_bands_are_ordered_and_evenly_spaced():
    rng = np.random.default_rng(9)
    b = v.sigma_bands(frame(prices_from(rng.normal(0, 0.005, 200))))
    assert b is not None
    ordered = [b["sigma3_dn"], b["sigma2_dn"], b["sigma1_dn"], b["mean"],
               b["sigma1_up"], b["sigma2_up"], b["sigma3_up"]]
    assert ordered == sorted(ordered)
    step = b["sigma1_up"] - b["mean"]
    assert abs((b["sigma2_up"] - b["mean"]) - 2 * step) < 1e-6
    assert abs((b["sigma3_up"] - b["mean"]) - 3 * step) < 1e-6
    assert abs((b["mean"] - b["sigma3_dn"]) - 3 * step) < 1e-6


# ---------- the matrix ----------
def test_implied_volatility_comes_from_the_options_feed_never_from_realized():
    rng = np.random.default_rng(13)
    df = frame(prices_from(rng.normal(0, 0.006, 600)))
    opts = OptionsSnapshot(source="deribit", iv_atm=55.0, iv_skew=3.0)
    m = v.volatility_matrix(df, "1h", opts)
    assert m.available and m.bviv_pct == 55.0
    assert m.bv_pct is not None and m.iv_premium_pct == pytest.approx(55.0 - m.bv_pct)

    blind = v.volatility_matrix(df, "1h", OptionsSnapshot.unavailable("deribit", "timeout"))
    assert blind.bviv_pct is None and blind.iv_premium_pct is None
    assert "implied" in blind.note.lower()
    assert blind.bv_pct is not None, "realized vol still stands on its own"


def test_the_matrix_carries_atr_bands_and_memory():
    m = v.volatility_matrix(frame(prices_from(ar1(0.6))), "1h", None)
    assert m.atr is not None and m.atr_pct is not None
    assert m.hurst is not None and m.memory == "trending"
    assert set(m.bands) == {"mean", "sigma1_up", "sigma1_dn", "sigma2_up", "sigma2_dn",
                            "sigma3_up", "sigma3_dn"}
    assert m.regime in ("LOW", "NORMAL", "HIGH", "UNKNOWN")


def test_the_matrix_says_unavailable_on_thin_data_instead_of_zeros():
    m = v.volatility_matrix(frame(prices_from([0.001] * 6)), "1h", None)
    assert not m.available and m.bv_pct is None and m.hurst is None and m.bands == {}


def test_the_summary_handed_to_the_agents_is_json_safe():
    import json

    rng = np.random.default_rng(17)
    df = frame(prices_from(rng.normal(0, 0.005, 400)))
    out = v.matrix_summary(df, "1h", OptionsSnapshot(source="deribit", iv_atm=48.0))
    json.dumps(out)
    assert out["available"] is True and out["implied_vol_pct"] == 48.0
    assert "realized_vol_pct" in out and "hurst" in out and "memory" in out
    assert out["bands"]["sigma3_up"] > out["bands"]["sigma1_up"]


def test_the_neutral_band_is_as_wide_as_the_estimators_noise():
    """A random walk measures 0.50 give or take 0.03, so readings inside two sigma claim nothing."""
    assert v.memory_label(0.55) == "random walk" and v.memory_label(0.45) == "random walk"
    assert v.memory_label(0.57) == "trending" and v.memory_label(0.43) == "mean-reverting"
    assert v.TRENDING_AT - 0.5 == pytest.approx(0.5 - v.MEAN_REVERTING_AT), "the band is symmetric"


def test_the_estimator_separates_the_three_regimes_across_many_seeds():
    """Guards the discrimination the spec's own formula lacks: it reads a random walk and a
    mean-reverting series within 0.001 of each other."""
    reads = {"walk": [], "trend": [], "revert": []}
    for seed in range(12):
        rng = np.random.default_rng(seed)
        reads["walk"].append(v.hurst(prices_from(rng.normal(0, 0.004, 900))))
        reads["trend"].append(v.hurst(prices_from(ar1(0.6, seed=seed))))
        reads["revert"].append(v.hurst(prices_from(ar1(-0.6, seed=seed))))
    walk, trend, revert = (np.array(reads[k], dtype=float) for k in ("walk", "trend", "revert"))
    assert abs(walk.mean() - 0.5) < 0.03, walk.mean()
    assert trend.min() > walk.max(), "every trending path must read above every random walk"
    assert revert.max() < walk.min(), "every mean-reverting path must read below every random walk"


def test_the_unavailable_note_gives_the_real_reason():
    """A 500-bar flat series is not short of candles; saying so points the operator at the wrong thing."""
    m = v.volatility_matrix(frame(np.full(500, 80_000.0)), "1h", None)
    assert not m.available
    assert "variance" in m.note.lower() or "flat" in m.note.lower()
    assert "500" not in m.note


def test_one_degenerate_lag_does_not_discard_the_whole_estimate():
    """A flat stretch inside live data kills one aggregation lag; the remaining lags still answer."""
    rng = np.random.default_rng(23)
    live = np.diff(np.log(prices_from(rng.normal(0, 0.004, 700))))
    padded = np.concatenate([np.zeros(64), live])            # 64 identical bars, then real movement
    h = v.hurst(80_000.0 * np.exp(np.cumsum(padded)))
    assert h is not None, "one dead lag must not void the other lags"
