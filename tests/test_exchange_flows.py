"""Exchange flows: Coin Metrics parsing, the read measured against its own year, and where it is shown."""
import asyncio
import json
import pathlib

import httpx
import pytest

from core.data import exchange_flows as ef
from core.data.types import ExchangeFlowSnapshot
from core.indicators import exchange_flows as rf

FX = pathlib.Path(__file__).parent / "fixtures"
DAY = 86_400_000
T0 = 20_000 * DAY


def _raw():
    return json.loads((FX / "coinmetrics_exchange_flows.json").read_text(encoding="utf-8"))


def _snap(nets, inflow=20_000.0, reserve=2_700_000.0, flash=False):
    """A flow history from a list of daily netflows; inflow fixed so the outflow carries the net."""
    days, res = [], reserve
    for i, n in enumerate(nets):
        res += n
        days.append({"time_ms": T0 + i * DAY, "inflow": inflow + max(n, 0), "outflow": inflow - min(n, 0),
                     "net": float(n), "reserve": res})
    return ExchangeFlowSnapshot(source="coinmetrics", days=days, as_of_ms=days[-1]["time_ms"], preliminary=flash)


def _noise(n, seed=1, scale=1_000.0):
    import numpy as np
    return list(np.random.default_rng(seed).normal(0, scale, n))


NOW = T0 + 400 * DAY


# ---------- parsing ----------
def test_the_recorded_response_parses_to_complete_days_in_btc():
    s = ef.parse_flows(_raw())
    assert s.available and len(s.days) == 420 and s.preliminary
    d = s.days[-1]
    assert d["net"] == pytest.approx(d["inflow"] - d["outflow"]) and 2_000_000 < d["reserve"] < 3_500_000
    assert [x["time_ms"] for x in s.days] == sorted(x["time_ms"] for x in s.days)


def test_a_day_missing_a_series_is_dropped_not_zero_filled():
    raw = {"data": [{"time": "2026-10-01T00:00:00.000000000Z", "FlowInExNtv": "10", "FlowOutExNtv": "4", "SplyExNtv": "100"},
                    {"time": "2026-10-02T00:00:00.000000000Z", "FlowOutExNtv": "4", "SplyExNtv": "100"},
                    {"time": "2026-10-03T00:00:00.000000000Z", "FlowInExNtv": "x", "FlowOutExNtv": "4", "SplyExNtv": "100"}]}
    s = ef.parse_flows(raw)
    assert len(s.days) == 1 and s.days[0]["net"] == 6.0


def test_an_error_payload_is_unavailable_with_its_reason():
    s = ef.parse_flows({"error": {"type": "forbidden", "message": "nope"}})
    assert not s.available and "forbidden" in s.error


def test_the_request_asks_for_btc_daily_with_a_years_history():
    seen = []

    class T(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            seen.append(request.url.params)
            return httpx.Response(200, json=_raw(), request=request)

    async def go():
        async with httpx.AsyncClient(transport=T()) as c:
            return await ef.fetch_exchange_flows(c, now_ms=1_792_000_000_000)
    s = asyncio.run(go())
    p = seen[0]
    assert s.available and p["assets"] == "btc" and p["frequency"] == "1d"
    assert set(p["metrics"].split(",")) == {"FlowInExNtv", "FlowOutExNtv", "SplyExNtv"}
    from datetime import datetime, timedelta, timezone
    want = (datetime.fromtimestamp(1_792_000_000, timezone.utc) - timedelta(days=420)).strftime("%Y-%m-%d")
    assert p["start_time"] == want, "420 days back"


def test_a_dead_endpoint_is_unavailable_not_an_exception():
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(503))) as c:
            return await ef.fetch_exchange_flows(c)
    assert not asyncio.run(go()).available


# ---------- the read ----------
def test_a_heavy_week_of_inflows_against_a_quiet_year_is_distribution():
    f = rf.read(_snap(_noise(393) + [6_000.0] * 7), NOW)
    assert f["state"] == "distribution" and f["net_7d_z"] > 3 and f["net_7d_btc"] == pytest.approx(42_000, abs=1)


def test_the_yardstick_is_the_year_before_this_week_not_including_it():
    import numpy as np
    nets = _noise(393) + [6_000.0] * 7
    sums = np.convolve(np.array(nets), np.ones(7), "valid")
    prior = sums[-366:-1]
    want = (sums[-1] - prior.mean()) / prior.std()
    with_itself = (sums[-1] - sums[-366:].mean()) / sums[-366:].std()
    assert abs(want - with_itself) > 0.05, "the test can tell the two apart"
    assert rf.read(_snap(nets), NOW)["net_7d_z"] == pytest.approx(want, abs=0.006)


def test_a_heavy_week_of_withdrawals_is_accumulation():
    assert rf.read(_snap(_noise(393) + [-6_000.0] * 7), NOW)["state"] == "accumulation"


def test_an_ordinary_week_is_neutral():
    assert rf.read(_snap(_noise(400)), NOW)["state"] == "neutral"


def test_heavy_is_measured_against_the_year_not_a_fixed_size():
    """The same 42,000 BTC week is ordinary in a year of big swings."""
    f = rf.read(_snap(_noise(393, scale=40_000.0) + [6_000.0] * 7), NOW)
    assert f["state"] == "neutral" and abs(f["net_7d_z"]) < 1


def test_inflows_into_a_rally_are_flagged_as_divergence():
    snap = _snap(_noise(393) + [6_000.0] * 7)
    end = snap.days[-1]["time_ms"]
    closes = [(end - 6 * DAY, 100.0), (end, 104.0)]
    assert "into the rally" in rf.read(snap, NOW, closes)["divergence"]
    assert rf.read(snap, NOW, [(end - 6 * DAY, 100.0), (end, 101.0)])["divergence"] is None, "price barely moved"


def test_withdrawals_on_a_dip_are_flagged_as_divergence():
    snap = _snap(_noise(393) + [-6_000.0] * 7)
    end = snap.days[-1]["time_ms"]
    assert "on the dip" in rf.read(snap, NOW, [(end - 6 * DAY, 100.0), (end, 96.0)])["divergence"]


def test_a_one_day_inflow_spike_is_flagged():
    nets = _noise(399)
    snap = _snap(nets + [0.0])
    snap.days[-1]["inflow"] = 80_000.0
    f = rf.read(snap, NOW)
    assert f["inflow_spike"] and f["inflow_spike_z"] > 2


def test_the_reserve_trend_is_the_30_day_change():
    f = rf.read(_snap([0.0] * 370 + [-100.0] * 30), NOW)
    assert f["reserve_30d_change_btc"] == pytest.approx(-3_000)


def test_old_or_short_histories_say_so():
    assert rf.read(_snap(_noise(400)), T0 + 405 * DAY)["stale"]
    short = rf.read(_snap(_noise(100)), NOW)
    assert not short["available"] and "100 days" in short["note"]
    assert not rf.read(ExchangeFlowSnapshot.unavailable("coinmetrics", "down"), NOW)["available"]


# ---------- where it goes ----------
def _market():
    import os
    os.environ["TI_OFFLINE_FIXTURES"] = "1"
    from core.data.offline import fixture_market
    return fixture_market()


def test_the_on_chain_agent_gets_the_read_not_the_raw_days():
    from core.agents.context import derivatives_payload
    data, missing = derivatives_payload(_market())
    f = data["exchange_flows"]
    assert f["available"] and "days" not in f and "not only whales" in f["scope"] and "exchange_flows" not in missing
    gone = _market().model_copy(update={"exchange_flows": ExchangeFlowSnapshot.unavailable("coinmetrics", "x")})
    assert "exchange_flows" in derivatives_payload(gone)[1]


def test_the_deck_shows_the_weekly_netflow():
    from core.agents.deck import sub_agent_deck
    from core.agents.recon_profiles import PROFILES
    st = {"derivatives": {"exchange_flows": {"net_7d_btc": -12_345.0}}}
    derivs = next(a for a in sub_agent_deck(st, PROFILES["weekly"]) if a.key == "derivs")
    assert ("Exchange netflow 7d, BTC", -12_345.0, "num") in derivs.rows


def test_the_card_states_scope_and_track_record():
    from ui import panels
    snap = _snap(_noise(393) + [6_000.0] * 7, flash=True)
    end = snap.days[-1]["time_ms"]
    h = panels.exchange_flows_html(rf.read(snap, NOW, [(end - 6 * DAY, 100.0), (end, 104.0)]))
    assert "ti-dir down'>Distribution</div>" in h and "Coins arriving on exchanges: 7-day netflow" in h
    assert "+42,000 BTC" in h and "ti-card down" in h
    assert "price divergence" in h and "not only whales" in h and "45% of the time" in h and "preliminary" in h
    assert "Unavailable" in panels.exchange_flows_html({"available": False, "note": "down"})


def test_the_feed_is_listed_with_the_others():
    from core.data.types import MarketSnapshot
    assert "exchange_flows" in MarketSnapshot.SOURCES
