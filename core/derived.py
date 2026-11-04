"""Derive futures fields the fallback feeds cannot provide, from the app's own stored samples.

When the direct exchanges are geo-blocked we still see funding and open interest on every refresh.
Storing those samples lets us compute the 24h OI change and a rolling funding mean ourselves; the
result is labelled "derived" and never invented when history is too short."""
from __future__ import annotations

from core.data.types import DominanceSnapshot, FuturesSnapshot

HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS
WEEK_MS = 7 * DAY_MS
MIN_OI_AGE_MS = 20 * HOUR_MS      # need a sample at least this old to call it a "24h" change
MIN_FUNDING_SAMPLES = 3
MIN_DOM_DAY_AGE_MS = 20 * HOUR_MS   # a sample at least this old before a "24h" share change is claimed
MIN_DOM_WEEK_AGE_MS = 6 * DAY_MS    # and this old for a "7d" one


def record_sample(store, fut: FuturesSnapshot, now_ms: int) -> None:
    if fut.available and (fut.funding_rate is not None or fut.open_interest_usd is not None):
        store.add_futures_sample(now_ms, fut.funding_rate, fut.open_interest_usd)


def record_dominance(store, dom: DominanceSnapshot, now_ms: int) -> None:
    if getattr(dom, "available", False) and dom.btc_pct is not None:
        store.add_dominance_sample(now_ms, dom.btc_pct, dom.total_mcap_usd)


def _dominance_change(store, dom: DominanceSnapshot, now_ms: int, back_ms: int, min_age_ms: int):
    """(change in percentage points, hours covered) against the sample at or before now - back_ms, or the
    oldest sample when it is old enough to mean something; (None, None) otherwise."""
    old = store.dominance_sample_at_or_before(now_ms - back_ms)
    if old is None:
        oldest = store.oldest_dominance_sample()
        if oldest and now_ms - oldest["ts_ms"] >= min_age_ms:
            old = oldest
    if not old or old.get("btc_pct") is None:
        return None, None
    return round(dom.btc_pct - float(old["btc_pct"]), 2), (now_ms - old["ts_ms"]) / HOUR_MS


def enrich_dominance(store, dom: DominanceSnapshot, now_ms: int) -> DominanceSnapshot:
    """Fill change_24h_pp and change_7d_pp from stored samples; the feed itself has no history."""
    if not getattr(dom, "available", False) or dom.btc_pct is None:
        return dom
    updates: dict = {}
    notes: list[str] = []
    day, hours = _dominance_change(store, dom, now_ms, DAY_MS, MIN_DOM_DAY_AGE_MS)
    if day is not None:
        updates["change_24h_pp"] = day
        notes.append(f"24h change over {hours:.0f}h")
    week, hours = _dominance_change(store, dom, now_ms, WEEK_MS, MIN_DOM_WEEK_AGE_MS)
    if week is not None:
        updates["change_7d_pp"] = week
        notes.append(f"7d change over {hours / 24:.1f}d")
    if not updates:
        return dom
    updates["derived_note"] = "from own samples: " + "; ".join(notes)
    return dom.model_copy(update=updates)


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
