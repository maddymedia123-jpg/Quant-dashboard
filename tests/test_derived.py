from core.data.types import FuturesSnapshot
from core.derived import DAY_MS, HOUR_MS, enrich_futures, record_sample
from core.store import Store


def _fb(**kw):
    base = dict(source="coinlobster+hyperliquid (Binance Futures)", funding_rate=0.0001, open_interest_usd=8.0e9)
    base.update(kw)
    return FuturesSnapshot(**base)


def test_no_history_leaves_fields_none():
    s = Store(":memory:")
    f = enrich_futures(s, _fb(), now_ms=DAY_MS)
    assert f.oi_change_24h_pct is None and f.funding_7d_mean is None and "derived" not in f.source


def test_oi_change_and_funding_mean_from_samples():
    s = Store(":memory:")
    t0 = 10 * DAY_MS
    for i, (fr, oi) in enumerate([(0.0001, 8.0e9), (0.0002, 8.1e9), (0.0003, 8.2e9)]):
        record_sample(s, _fb(funding_rate=fr, open_interest_usd=oi), t0 + i * 8 * HOUR_MS)
    now = t0 + 24 * HOUR_MS
    f = enrich_futures(s, _fb(funding_rate=0.0004, open_interest_usd=8.4e9), now)
    assert abs(f.oi_change_24h_pct - 5.0) < 1e-9          # 8.0e9 → 8.4e9 against the sample 24h ago
    assert abs(f.funding_7d_mean - 0.0002) < 1e-12        # mean of the three stored samples
    assert "derived" in f.source and "24h" in f.source


def test_young_history_uses_oldest_sample_only_when_old_enough():
    s = Store(":memory:")
    t0 = 10 * DAY_MS
    record_sample(s, _fb(open_interest_usd=8.0e9), t0)
    assert enrich_futures(s, _fb(open_interest_usd=8.8e9), t0 + 5 * HOUR_MS).oi_change_24h_pct is None
    f = enrich_futures(s, _fb(open_interest_usd=8.8e9), t0 + 21 * HOUR_MS)
    assert abs(f.oi_change_24h_pct - 10.0) < 1e-9 and "21h" in f.source


def test_feed_values_are_never_overwritten_and_sampling_is_throttled():
    s = Store(":memory:")
    direct = FuturesSnapshot(source="binance", funding_rate=0.0001, funding_7d_mean=0.00005, open_interest_usd=8e9, oi_change_24h_pct=-1.0)
    record_sample(s, direct, 1_000)
    record_sample(s, direct, 30_000)   # within a minute → dropped
    assert len(s.futures_samples_since(0)) == 1
    assert enrich_futures(s, direct, 2 * DAY_MS).oi_change_24h_pct == -1.0
    record_sample(s, FuturesSnapshot.unavailable("x", "y"), 5_000_000)
    assert len(s.futures_samples_since(0)) == 1
