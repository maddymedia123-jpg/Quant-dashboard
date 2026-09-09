from core.data.types import FuturesSnapshot
from core.indicators.squeeze import squeeze_risk


def test_crowded_longs_flag_long_squeeze():
    fut = FuturesSnapshot(source="binance", funding_rate=0.0005, funding_7d_mean=0.0001, oi_change_24h_pct=6.0, long_short_ratio=1.6)
    r = squeeze_risk(fut, price_change_24h_pct=0.2)
    assert r.direction == "LONG_SQUEEZE" and r.score >= 60
    assert any("funding" in d.lower() for d in r.drivers)


def test_crowded_shorts_flag_short_squeeze():
    fut = FuturesSnapshot(source="binance", funding_rate=-0.0004, funding_7d_mean=0.0, oi_change_24h_pct=5.0, long_short_ratio=0.6)
    r = squeeze_risk(fut, price_change_24h_pct=-0.3)
    assert r.direction == "SHORT_SQUEEZE" and r.score >= 60


def test_calm_market_none():
    fut = FuturesSnapshot(source="binance", funding_rate=0.0001, funding_7d_mean=0.0001, oi_change_24h_pct=0.2, long_short_ratio=1.0)
    r = squeeze_risk(fut, price_change_24h_pct=3.0)
    assert r.direction == "NONE" and r.score < 40


def test_unavailable_futures():
    r = squeeze_risk(FuturesSnapshot.unavailable("binance,bybit", "451"), price_change_24h_pct=1.0)
    assert r.score is None and r.direction == "UNKNOWN"


def test_partial_futures_scores_with_missing_components_named():
    fut = FuturesSnapshot(source="coinlobster (Binance Futures)", funding_rate=0.0006, funding_7d_mean=None,
                          oi_change_24h_pct=None, long_short_ratio=None, open_interest_usd=8e9)
    r = squeeze_risk(fut, price_change_24h_pct=0.1)
    assert r.score is not None and r.direction == "LONG_SQUEEZE"
    assert any("OI 24h change unavailable" in d for d in r.drivers) and any("long/short" in d for d in r.drivers)
