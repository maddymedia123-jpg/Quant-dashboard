"""BTC dominance from CoinGecko /global (roadmap item 10), its change from the dashboard's own samples,
the rotation read, the card and the agents' read.

The level alone says little; the change is the read, and the feed has no history, so the samples the
worker stores are what make a change possible. Each guard here is a way the number could lie: a share
no market has printed, a change claimed from a sample too young to mean "24h", a stale print shown as
current."""
import asyncio
import json
import pathlib

import httpx
import pytest

from core.data import dominance
from core.data.offline import fixture_market
from core.data.types import DominanceSnapshot
from core.derived import DAY_MS, HOUR_MS, WEEK_MS, enrich_dominance, record_dominance
from core.store import Store
from ui import panels

FX = pathlib.Path(__file__).parent / "fixtures"
NOW = 1_793_824_000_000   # a couple of minutes after the recorded updated_at


def load():
    return json.loads((FX / "coingecko_global.json").read_text(encoding="utf-8"))


def snap(**kw) -> DominanceSnapshot:
    base = dict(source="coingecko", btc_pct=59.54, eth_pct=10.9, stable_pct=9.2, total_mcap_usd=2.708e12,
                mcap_change_24h_pct=0.39, as_of_ms=NOW - 120_000)
    base.update(kw)
    return DominanceSnapshot(**base)


# ---------- parsing ----------
def test_the_recorded_global_payload_parses():
    d = dominance.parse_global(load())
    assert d.available and d.source == "coingecko"
    assert d.btc_pct == 59.54 and d.eth_pct == 10.9
    assert d.stable_pct == pytest.approx(6.58 + 2.62, abs=0.01), "USDT plus USDC"
    assert d.total_mcap_usd == pytest.approx(2.708e12, rel=0.01)
    assert d.mcap_change_24h_pct == 0.39
    assert d.as_of_ms == 1793823873 * 1000, "updated_at is epoch seconds"
    assert d.change_24h_pp is None and d.change_7d_pp is None, "the feed has no history of its own"


@pytest.mark.parametrize("payload", [{}, {"data": []}, {"data": {}}, {"data": {"market_cap_percentage": {"eth": 10.0}}}])
def test_a_payload_without_the_share_is_refused(payload):
    with pytest.raises(ValueError):
        dominance.parse_global(payload)


@pytest.mark.parametrize("share", [0.5954, 5954.0, 12.0, 97.0])
def test_a_share_no_market_has_printed_is_refused(share):
    with pytest.raises(ValueError, match="plausible"):
        dominance.parse_global({"data": {"market_cap_percentage": {"btc": share}}})


def test_missing_optional_fields_leave_none_not_zero():
    d = dominance.parse_global({"data": {"market_cap_percentage": {"btc": 55.0}}})
    assert d.btc_pct == 55.0 and d.eth_pct is None and d.stable_pct is None
    assert d.total_mcap_usd is None and d.mcap_change_24h_pct is None and d.as_of_ms is None


def test_staleness_is_two_hours():
    assert not dominance.is_stale(snap(), NOW)
    assert dominance.is_stale(snap(as_of_ms=NOW - 3 * HOUR_MS), NOW)
    assert not dominance.is_stale(snap(as_of_ms=None), NOW), "an undated print is not called stale"


def test_the_fetch_hits_the_global_endpoint_and_refuses_an_oversized_body():
    seen = []

    def ok(request: httpx.Request):
        seen.append(str(request.url))
        return httpx.Response(200, json=load())

    d = asyncio.run(dominance.fetch_dominance(httpx.AsyncClient(transport=httpx.MockTransport(ok))))
    assert d.btc_pct == 59.54 and seen == [dominance.URL]

    def big(request: httpx.Request):
        return httpx.Response(200, text="[" + "1," * 200_000 + "1]")

    with pytest.raises(ValueError, match="KB"):
        asyncio.run(dominance.fetch_dominance(httpx.AsyncClient(transport=httpx.MockTransport(big))))


# ---------- the rotation read ----------
def test_no_history_is_unknown_not_flat():
    r = dominance.rotation(snap())
    assert r["state"] == "unknown" and "history too short" in r["note"] and "59.5%" in r["note"]


def test_a_rising_share_is_rotation_into_bitcoin():
    r = dominance.rotation(snap(change_24h_pp=0.8))
    assert r["state"] == "into_btc" and "+0.8 pp over 24h" in r["note"] and "into bitcoin" in r["note"]
    assert "whole market up 0.4%" in r["note"]


def test_a_falling_share_in_a_rising_market_is_risk_appetite_stretching():
    r = dominance.rotation(snap(change_24h_pp=-0.6))
    assert r["state"] == "into_alts" and "risk appetite stretching" in r["note"]
    r = dominance.rotation(snap(change_24h_pp=-0.6, mcap_change_24h_pct=-2.0))
    assert r["state"] == "into_alts" and "risk appetite" not in r["note"] and "down 2.0%" in r["note"]


def test_a_small_move_is_flat_and_the_day_decides_before_the_week():
    assert dominance.rotation(snap(change_24h_pp=0.3))["state"] == "flat"
    assert dominance.rotation(snap(change_24h_pp=0.3, change_7d_pp=4.0))["state"] == "flat", "24h read wins"
    r = dominance.rotation(snap(change_7d_pp=2.0))
    assert r["state"] == "into_btc" and "over 7d" in r["note"], "the week serves when the day has no sample"
    assert dominance.rotation(snap(change_7d_pp=1.0))["state"] == "flat", "a week needs 1.5 pp"


def test_an_unavailable_feed_reads_unknown():
    assert dominance.rotation(DominanceSnapshot.unavailable("coingecko", "429"))["state"] == "unknown"


# ---------- own samples ----------
def test_no_samples_leaves_the_changes_none():
    s = Store(":memory:")
    d = enrich_dominance(s, snap(), NOW)
    assert d.change_24h_pp is None and d.change_7d_pp is None and d.derived_note is None


def test_the_day_and_week_changes_come_from_samples_old_enough():
    s = Store(":memory:")
    record_dominance(s, snap(btc_pct=58.0), NOW - WEEK_MS - HOUR_MS)
    record_dominance(s, snap(btc_pct=59.0), NOW - DAY_MS - HOUR_MS)
    d = enrich_dominance(s, snap(btc_pct=59.54), NOW)
    assert d.change_24h_pp == pytest.approx(0.54) and d.change_7d_pp == pytest.approx(1.54)
    assert "24h change over 25h" in d.derived_note and "7d change over 7.0d" in d.derived_note


def test_a_young_history_uses_the_oldest_sample_only_once_it_is_old_enough():
    s = Store(":memory:")
    record_dominance(s, snap(btc_pct=59.0), NOW - 10 * HOUR_MS)
    assert enrich_dominance(s, snap(), NOW).change_24h_pp is None, "10 hours is not a day"
    s2 = Store(":memory:")
    record_dominance(s2, snap(btc_pct=59.0), NOW - 21 * HOUR_MS)
    d = enrich_dominance(s2, snap(btc_pct=59.54), NOW)
    assert d.change_24h_pp == pytest.approx(0.54) and d.change_7d_pp is None, "21 hours counts as a day, not a week"


def test_sampling_is_throttled_and_unavailable_feeds_are_not_sampled():
    s = Store(":memory:")
    record_dominance(s, snap(btc_pct=59.0), NOW)
    record_dominance(s, snap(btc_pct=60.0), NOW + 5 * 60_000)
    record_dominance(s, DominanceSnapshot.unavailable("coingecko", "429"), NOW + 20 * 60_000)
    assert s.oldest_dominance_sample()["btc_pct"] == 59.0
    assert s.dominance_sample_at_or_before(NOW + DAY_MS)["btc_pct"] == 59.0, "the 5-minute repeat was dropped"
    record_dominance(s, snap(btc_pct=61.0), NOW + 11 * 60_000)
    assert s.dominance_sample_at_or_before(NOW + DAY_MS)["btc_pct"] == 61.0


def test_an_unavailable_feed_passes_through_enrichment_untouched():
    bad = DominanceSnapshot.unavailable("coingecko", "429")
    assert enrich_dominance(Store(":memory:"), bad, NOW) is bad


# ---------- the card ----------
def test_the_card_shows_the_share_the_changes_and_the_read():
    h = panels.dominance_html(snap(change_24h_pp=0.8, change_7d_pp=1.2), NOW)
    assert "59.5%" in h and "ETH 10.9%" in h and "stablecoins 9.2%" in h
    assert "+0.80 pp" in h and "+1.20 pp" in h and "$2.71T" in h and "+0.4% 24h" in h
    assert "rotating into BTC" in h and "capital is rotating into bitcoin" in h
    assert "stale" not in h


def test_the_card_names_an_empty_history_and_a_stale_print():
    h = panels.dominance_html(snap(as_of_ms=NOW - 5 * HOUR_MS), NOW)
    assert "no sample yet" in h and "no history yet" in h and ">stale<" in h


def test_the_card_warns_on_rotation_to_alts_and_escapes():
    h = panels.dominance_html(snap(change_24h_pp=-0.9), NOW)
    assert "ti-card warn" in h and "rotating to alts" in h
    h = panels.dominance_html(DominanceSnapshot.unavailable("coingecko", "<429>"), NOW)
    assert "No dominance feed" in h and "&lt;429&gt;" in h


# ---------- the agents ----------
def test_the_fixture_market_carries_dominance_to_the_macro_read():
    from core.agents import context
    m = fixture_market()
    assert m.dominance.source == "fixture" and m.dominance.btc_pct == 59.54
    read = context.macro_read(m)["btc_dominance"]
    assert read["available"] and read["btc_pct"] == 59.54 and read["rotation"] == "unknown"
    assert "no change yet" in read["note"]
    m2 = m.model_copy(update={"dominance": m.dominance.model_copy(update={"change_24h_pp": 1.1})})
    assert context.macro_read(m2)["btc_dominance"]["rotation"] == "into_btc"
    m3 = m.model_copy(update={"dominance": DominanceSnapshot.unavailable("coingecko", "429")})
    assert context.macro_read(m3)["btc_dominance"] == {"available": False, "note": "429"}
