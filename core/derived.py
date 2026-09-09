"""Derive futures fields the fallback feeds cannot provide, from the app's own stored samples.

When the direct exchanges are geo-blocked we still see funding and open interest on every refresh.
Storing those samples lets us compute the 24h OI change and a rolling funding mean ourselves; the
result is labelled "derived" and never invented when history is too short."""
from __future__ import annotations

from core.data.types import FuturesSnapshot

HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS
WEEK_MS = 7 * DAY_MS
MIN_OI_AGE_MS = 20 * HOUR_MS      # need a sample at least this old to call it a "24h" change
MIN_FUNDING_SAMPLES = 3


def record_sample(store, fut: FuturesSnapshot, now_ms: int) -> None:
    if fut.available and (fut.funding_rate is not None or fut.open_interest_usd is not None):
        store.add_futures_sample(now_ms, fut.funding_rate, fut.open_interest_usd)


def enrich_futures(store, fut: FuturesSnapshot, now_ms: int) -> FuturesSnapshot:
    """Fill oi_change_24h_pct and funding_7d_mean from stored samples when the feed lacks them."""
    if not fut.available:
        return fut
    updates: dict = {}
    notes: list[str] = []
    if fut.oi_change_24h_pct is None and fut.open_interest_usd:
        old = store.futures_sample_at_or_before(now_ms - DAY_MS)
        if old is None:
            # oldest sample we have, if it is old enough to be meaningful
            oldest = store.oldest_futures_sample()
            if oldest and now_ms - oldest["ts_ms"] >= MIN_OI_AGE_MS:
                old = oldest
        if old and old.get("open_interest_usd"):
            updates["oi_change_24h_pct"] = (fut.open_interest_usd / float(old["open_interest_usd"]) - 1.0) * 100.0
            hours = (now_ms - old["ts_ms"]) / HOUR_MS
            notes.append(f"OI change over {hours:.0f}h from own samples")
    if fut.funding_7d_mean is None and fut.funding_rate is not None:
        rates = [s["funding_rate"] for s in store.futures_samples_since(now_ms - WEEK_MS) if s.get("funding_rate") is not None]
        if len(rates) >= MIN_FUNDING_SAMPLES:
            updates["funding_7d_mean"] = sum(rates) / len(rates)
            span_h = (now_ms - store.futures_samples_since(now_ms - WEEK_MS)[0]["ts_ms"]) / HOUR_MS
            notes.append(f"funding mean over {span_h:.0f}h ({len(rates)} samples)")
    if not updates:
        return fut
    updates["source"] = f"{fut.source} · derived: {'; '.join(notes)}"
    return fut.model_copy(update=updates)
