"""Real taker flow from the Gate.io perpetual: parsing, the delta it gives, and where it replaces the proxy."""
import asyncio
import json
import pathlib

import httpx
import numpy as np
import pandas as pd
import pytest

from core.data import taker_flow as tf
from core.indicators import liquidity as lq

FX = pathlib.Path(__file__).parent / "fixtures"
H = 3_600_000


def _raw():
    return json.loads((FX / "gate_contract_stats_btc.json").read_text(encoding="utf-8"))


def _row(t, buy, sell, **over):
    r = {"time": t, "long_taker_size": buy, "short_taker_size": sell, "long_liq_usd": 0, "short_liq_usd": 0}
    r.update(over)
    return r


# ---------- parsing ----------
def test_contract_sizes_are_converted_to_bitcoin():
    df = tf.parse_stats([_row(1000, 50_000, 20_000)], 0.0001)
    assert (df.buy[0], df.sell[0], df.delta[0]) == pytest.approx((5.0, 2.0, 3.0))
    assert df.timestamp[0] == 1_000_000, "seconds become milliseconds"


def test_a_row_missing_a_taker_size_is_dropped_not_read_as_balanced():
    df = tf.parse_stats([_row(1, 10, 5), {"time": 2, "long_taker_size": 7}, _row(3, None, 4), "junk"], 1.0)
    assert list(df.timestamp) == [1000]


def test_rows_are_sorted_and_deduplicated():
    df = tf.parse_stats([_row(3, 1, 0), _row(1, 1, 0), _row(3, 1, 0)], 1.0)
    assert list(df.timestamp) == [1000, 3000]


def test_a_payload_that_is_not_rows_is_refused():
    with pytest.raises(ValueError):
        tf.parse_stats({"label": "INVALID_PARAM"}, 1.0)


def test_the_recorded_stats_build_every_timeframe_the_desks_read():
    raw = _raw()
    s = tf.build({k: raw[k] for k in tf.FETCHED}, float(raw["contract"]["quanto_multiplier"]))
    assert s.available and s.btc_per_contract == 0.0001 and s.contract == "BTC_USDT"
    assert set(s.frames) == {"15m", "1h", "4h", "1d", "1w", "1M"}
    assert all(len(s.frames[t]) == 120 for t in tf.FETCHED)


def test_weekly_and_monthly_flow_are_sums_of_the_days():
    day = 86_400_000
    monday = tf.EPOCH_MONDAY_MS + 3000 * 7 * day
    daily = tf.parse_stats([_row((monday + i * day) // 1000, 10 + i, 4) for i in range(9)], 1.0)
    w = tf.weekly(daily)
    assert list(w.timestamp) == [monday, monday + 7 * day]
    assert w.buy[0] == sum(10 + i for i in range(7)) and w.delta[1] == (17 - 4) + (18 - 4)
    m = tf.monthly(daily)
    assert m.delta.sum() == pytest.approx(daily.delta.sum())


def test_one_interval_failing_is_a_degraded_feed_that_names_it():
    raw = _raw()
    s = tf.build({"15m": raw["15m"], "1h": httpx.ConnectError("down"), "4h": raw["4h"], "1d": raw["1d"]}, 0.0001)
    assert s.available and "1h" not in s.frames and "1h: ConnectError: down" in s.error


def test_every_interval_failing_is_unavailable():
    s = tf.build({k: ValueError("x") for k in tf.FETCHED}, 0.0001)
    assert not s.available and "15m" in s.error


# ---------- the delta it gives ----------
def _price(ts, closes):
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"timestamp": ts, "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1.0})


def _flow(ts, deltas):
    d = np.asarray(deltas, dtype=float)
    return pd.DataFrame({"timestamp": ts, "buy": np.maximum(d, 0), "sell": np.maximum(-d, 0), "delta": d,
                         "long_liq_usd": 0.0, "short_liq_usd": 0.0})


def test_price_rising_while_takers_sell_is_bearish_absorption():
    ts = [i * H for i in range(30)]
    r = lq.taker_delta(_price(ts, range(100, 130)), _flow(ts, [-5.0] * 30))
    assert r.state == "bearish absorption" and r.units == "BTC" and r.method == lq.TAKER_METHOD
    assert r.delta_change == pytest.approx(-100.0)


def test_velocity_is_the_recent_average_and_acceleration_its_change():
    ts = [i * H for i in range(30)]
    deltas = [1.0] * 20 + [2.0] * 5 + [6.0] * 5
    r = lq.taker_delta(_price(ts, range(100, 130)), _flow(ts, deltas))
    assert r.velocity == pytest.approx(6.0) and r.acceleration == pytest.approx(4.0)


def test_periods_are_matched_on_their_opening_time_not_their_position():
    ts = [i * H for i in range(30)]
    flow = _flow([t + 15 * 60_000 for t in ts], [5.0] * 30)        # offset by a quarter hour: no match
    assert lq.taker_delta(_price(ts, range(100, 130)), flow) is None


def test_without_enough_taker_history_the_proxy_is_used_and_says_so():
    ts = [i * H for i in range(30)]
    out = lq.summarise(_price(ts, range(100, 130)), [], 30 * H, flow=_flow(ts[-10:], [5.0] * 10))
    assert out["cumulative_delta"]["method"] == lq.PROXY_METHOD and out["cumulative_delta"]["units"] == "proxy units"
    real = lq.summarise(_price(ts, range(100, 130)), [], 30 * H, flow=_flow(ts, [5.0] * 30))
    assert real["cumulative_delta"]["method"] == lq.TAKER_METHOD and real["cumulative_delta"]["units"] == "BTC"
    assert real["cumulative_delta"]["velocity_per_bar"] == 5.0


def test_the_judges_get_real_delta_when_the_market_carries_taker_flow():
    import os
    os.environ["TI_OFFLINE_FIXTURES"] = "1"
    from core.agents.live_recon import build_state
    from core.agents.recon_profiles import PROFILES
    from core.data.offline import fixture_market
    from core.data.types import TakerFlowSnapshot

    m = fixture_market()
    frames = {k: _flow(list(df["timestamp"]), [3.0] * len(df)) for k, df in m.spot.frames.items()}
    m = m.model_copy(update={"taker_flow": TakerFlowSnapshot(source="gate.io", frames=frames)})
    st = build_state(m, {}, profile=PROFILES["live"])
    assert all(st["liquidity"][t]["cumulative_delta"]["units"] == "BTC" for t in ("15m", "1h", "4h"))
    gone = m.model_copy(update={"taker_flow": TakerFlowSnapshot.unavailable("gate.io", "down")})
    assert build_state(gone, {}, profile=PROFILES["live"])["liquidity"]["1h"]["cumulative_delta"]["units"] != "BTC"


# ---------- what the page shows ----------
def test_the_kpi_sums_the_last_24_completed_hours():
    from core.data.types import TakerFlowSnapshot
    from ui import panels

    ts = [i * H for i in range(30)]
    deltas = [0.0] * 5 + [10.0] * 24 + [-999.0]                    # the forming hour must not count
    k = panels._taker_kpi(TakerFlowSnapshot(source="gate.io", frames={"1h": _flow(ts, deltas)}))
    assert k["value"] == "+240 BTC" and k["delta"] == "buyers paying" and k["tone"] == "up"
    assert panels._taker_kpi(TakerFlowSnapshot.unavailable("gate.io", "x"))["value"] == "—"


def test_the_liquidity_panel_names_the_method_of_each_row():
    from ui import panels
    real = {"available": True, "state": "confirming up", "units": "BTC", "velocity_per_bar": 12.5,
            "acceleration": 3.0}
    proxy = {"available": True, "state": "flat", "units": "proxy units"}
    row = lambda c: {"pools": {}, "cumulative_delta": c, "volume_profile": {"available": False}, "open_interest": {}}  # noqa: E731
    h = panels.liquidity_html({"1h": row(real), "1M": row(proxy)})
    assert "+12.5 BTC/bar, accelerating" in h and "(proxy)" in h
    assert "real taker volume" in h and "Rows marked proxy" in h
    only_real = panels.liquidity_html({"1h": row(real)})
    assert "Rows marked proxy" not in only_real and "(proxy)" not in only_real


def test_the_order_flow_agent_says_which_delta_it_read():
    from core.agents.deck import sub_agent_deck
    from core.agents.recon_profiles import PROFILES
    state = {"liquidity": {"1h": {"cumulative_delta": {"units": "BTC", "state": "confirming up"}}}}
    deck = sub_agent_deck(state, PROFILES["live"])
    flow_agent = next(a for a in deck if a.agent_no == 3)
    assert "real taker volume" in flow_agent.note


# ---------- fetching ----------
class _Transport(httpx.AsyncBaseTransport):
    def __init__(self, contract_status=200, size=None):
        self.requests, self.contract_status, self.size = [], contract_status, size

    async def handle_async_request(self, request):
        self.requests.append(request)
        if request.url.path.endswith("/contracts/BTC_USDT"):
            return httpx.Response(self.contract_status, json={"quanto_multiplier": "0.001"}, request=request)
        body = b"[" + b" " * self.size + b"]" if self.size else json.dumps([_row(1, 100, 40)]).encode()
        return httpx.Response(200, content=body, request=request)


def _fetch(t):
    async def go():
        async with httpx.AsyncClient(transport=t) as c:
            return await tf.fetch_taker_flow(c)
    return asyncio.run(go())


def test_the_request_asks_for_the_btc_perpetual_on_every_interval_and_uses_its_contract_size():
    t = _Transport()
    s = _fetch(t)
    stats = [r for r in t.requests if r.url.path.endswith("/contract_stats")]
    assert sorted(r.url.params["interval"] for r in stats) == sorted(tf.FETCHED)
    assert all(r.url.params["contract"] == "BTC_USDT" for r in stats)
    assert s.btc_per_contract == 0.001 and s.frames["1h"].delta[0] == pytest.approx(0.06)


def test_a_failed_contract_lookup_falls_back_to_the_published_size():
    s = _fetch(_Transport(contract_status=503))
    assert s.available and s.btc_per_contract == tf.DEFAULT_MULTIPLIER


def test_an_oversized_response_is_refused():
    s = _fetch(_Transport(size=tf.MAX_BYTES + 1))
    assert not s.available and "KB" in s.error, "every interval refused, and why"
