"""The macro engine: positioning into and reactions to USD prints, narrative shelf-life, and the Fed odds."""
import asyncio
import json
import math
import pathlib

import httpx
import numpy as np
import pandas as pd
import pytest

from core.data import prediction as pr
from core.indicators import event_trap as et
from core.indicators import narratives as nv

FX = pathlib.Path(__file__).parent / "fixtures"
H = 3_600_000
T0 = 20_000 * 24 * H
EVENT = T0 + 250 * H + 30 * 60_000          # a print at hh:30, as US releases are


def _candles(now_ms, path):
    """Hourly candles from T0 up to the one containing now (which is forming). `path(h)` is the price at
    hour h, multiplied by a small fixed noise so the volatility yardstick is ~0.1% an hour."""
    rng = np.random.default_rng(3)
    n = int((now_ms - T0) // H) + 1
    noise = np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    closes = np.array([path(h) for h in range(n)]) * noise
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame({"timestamp": [T0 + h * H for h in range(n)], "open": opens,
                         "high": np.maximum(opens, closes) * 1.0005, "low": np.minimum(opens, closes) * 0.9995,
                         "close": closes, "volume": 10.0})


def _event(t=EVENT, title="CPI m/m", impact="High", country="USD"):
    return {"title": title, "time_ms": t, "impact": impact, "country": country, "forecast": "0.3%", "previous": "0.2%"}


def _ramp(start_h, end_h, pct):
    """Flat at 100, then a straight move of `pct` between two hours, then flat."""
    def f(h):
        if h <= start_h:
            return 100.0
        if h >= end_h:
            return 100.0 * (1 + pct / 100)
        return 100.0 * (1 + pct / 100 * (h - start_h) / (end_h - start_h))
    return f


def _read(now, path, t=EVENT):
    return et.read_event(_candles(now, path), et.clusters([_event(t)])[0], now)


# ---------- before a print ----------
def test_a_run_up_into_the_print_is_positioned_and_names_who_is_at_risk():
    now = EVENT - 5 * H
    r = _read(now, _ramp(221, 245, 3.0))
    assert r.state == "positioned" and r.runup_z > 3 and r.trapped == "longs at risk"
    assert "Rumour bought" in r.headline and "sell-the-news drop" in r.headline


def test_the_yardstick_is_the_week_before_the_run_up_not_the_run_up_itself():
    """A run-up measured against volatility that includes the run-up would shrink its own z-score."""
    now = EVENT - 5 * H
    r = _read(now, _ramp(221, 245, 10.0))
    flat_sigma = r.expected_move_pct / 100 / math.sqrt(4)
    assert flat_sigma < 0.0013, "the hourly yardstick is the quiet week, ~0.1%, not inflated by the 10% run"


def test_no_run_up_is_flat_into_the_print():
    r = _read(EVENT - 5 * H, lambda h: 100.0)
    assert r.state == "flat" and 1.0 < abs(r.runup_z) < et.STRETCH_Z, "noise alone drifted over 1 sigma here"


def test_a_random_walk_rarely_reads_as_positioned():
    """Pure noise, read by the real reader across 200 random markets: few should look positioned."""
    now = EVENT - 5 * H
    cluster = et.clusters([_event()])[0]
    states = []
    for seed in range(200):
        rng = np.random.default_rng(seed)
        n = int((now - T0) // H) + 1
        closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
        df = pd.DataFrame({"timestamp": [T0 + h * H for h in range(n)], "open": np.r_[closes[0], closes[:-1]],
                           "high": closes, "low": closes, "close": closes, "volume": 1.0})
        states.append(et.read_event(df, cluster, now).state)
    share = states.count("positioned") / len(states)
    assert 0.05 < share < 0.22, share


def test_a_print_more_than_two_days_out_is_listed_not_read():
    assert _read(EVENT - 72 * H, lambda h: 100.0).state == "scheduled"


# ---------- after a print ----------
def test_a_run_up_given_back_after_the_print_is_sell_the_news():
    path = lambda h: _ramp(226, 250, 3.0)(h) if h <= 250 else _ramp(250, 256, -2.6)(h) * 1.03  # noqa: E731
    r = _read(EVENT + 10 * H, path)
    assert r.state == "sell_the_news" and r.trapped == "longs" and "A bull trap" in r.headline


def test_a_sell_off_bought_back_is_buy_the_news_with_shorts_trapped():
    path = lambda h: _ramp(226, 250, -3.0)(h) if h <= 250 else 97.0 * (1 + 0.026 * min(h - 250, 6) / 6)  # noqa: E731
    r = _read(EVENT + 10 * H, path)
    assert r.state == "sell_the_news" and r.trapped == "shorts" and r.headline.startswith("Buy the news")
    assert "sell-off into the print" in r.headline


def test_a_first_hour_spike_that_reverses_through_the_pre_print_price_faded():
    path = lambda h: 100.0 if h < 250 else 101.0 if h == 250 else 99.5  # noqa: E731
    r = _read(EVENT + 6 * H, path)
    assert r.state == "spike_faded" and r.trapped == "longs" and r.spike_z > 5


def test_a_spike_that_extends_is_confirmed():
    path = lambda h: 100.0 if h < 250 else 101.0 if h == 250 else 102.5  # noqa: E731
    assert _read(EVENT + 6 * H, path).state == "confirmed"


def test_a_print_that_moved_nothing_was_absorbed():
    assert _read(EVENT + 6 * H, lambda h: 100.0).state == "absorbed"


def test_until_two_hourly_closes_the_print_is_still_reacting():
    r = _read(EVENT + 40 * 60_000, lambda h: 100.0)                  # the print's own candle is still forming
    assert r.state == "reacting"


def test_the_forming_candle_is_never_used():
    now = EVENT + 6 * H
    df = _candles(now, lambda h: 100.0 if h < 250 else 101.0 if h == 250 else 102.5)
    df.loc[df.index[-1], "close"] = 90.0                             # a forming crash must not fade the spike
    assert et.read_event(df, et.clusters([_event()])[0], now).state == "confirmed"


def test_prints_in_the_same_minute_are_one_event_and_minor_ones_are_ignored():
    cs = et.clusters([_event(title="CPI m/m"), _event(title="CPI y/y"), _event(title="Claims", impact="Medium"),
                      _event(title="ECB", country="EUR"), _event(t=EVENT + H, title="Fed Speaks")])
    assert [c["titles"] for c in cs] == [["CPI m/m", "CPI y/y"], ["Fed Speaks"]]
    assert et.read_event(_candles(EVENT - H, lambda h: 100.0), cs[0], EVENT - H).title == "CPI m/m · CPI y/y"


def test_prints_more_than_two_days_old_drop_out():
    reads = et.read_all(_candles(EVENT + 60 * H, lambda h: 100.0), [_event()], EVENT + 60 * H)
    assert reads == []


# ---------- narratives ----------
def _h(topic, hours_ago, now, title="t"):
    return {"topic": topic, "published_ms": now - int(hours_ago * H), "title": title}


NOW = T0 + 1000 * H


def _narr(*hs):
    feed = [_h("general", 30, NOW)] + list(hs)                         # the feed reaches back 30 hours
    return {n.topic: n for n in nv.read(feed, NOW)}


def test_a_story_first_seen_hours_ago_is_fresh():
    assert _narr(_h("flows", 1, NOW), _h("flows", 3, NOW), _h("flows", 5, NOW))["flows"].state == "fresh"


def test_steady_attention_over_a_day_is_running_and_probably_priced():
    n = _narr(*[_h("etf", a, NOW) for a in (1, 7, 13, 19, 25)])["etf"]
    assert n.state == "running" and "probably priced in" in n.shelf_life


def test_attention_accelerating_on_an_older_story_is_building():
    n = _narr(_h("hack", 20, NOW), _h("hack", 1, NOW), _h("hack", 2, NOW), _h("hack", 3, NOW))["hack"]
    assert n.state == "building"


def test_a_story_quiet_for_twelve_hours_is_fading():
    n = _narr(_h("regulation", 28, NOW), _h("regulation", 20, NOW), _h("regulation", 14, NOW))["regulation"]
    assert n.state == "fading" and "ones at risk" in n.shelf_life


def test_a_few_quiet_hours_on_a_steady_story_is_not_fading():
    n = _narr(*[_h("etf", a, NOW) for a in (8, 14, 20, 26)])["etf"]
    assert n.state == "running"


def test_a_story_still_covered_at_half_its_earlier_rate_is_fading():
    n = _narr(_h("etf", 2, NOW), *[_h("etf", a, NOW) for a in (8, 10, 12, 14, 16, 18, 20, 22)])["etf"]
    assert n.state == "fading"


def test_untagged_and_single_headlines_are_not_narratives():
    assert set(_narr(_h("etf", 2, NOW), _h("general", 1, NOW), _h("general", 2, NOW))) == set()


def test_an_age_as_old_as_the_feed_is_marked_a_floor():
    n = _narr(_h("etf", 29.5, NOW), _h("etf", 1, NOW))["etf"]
    assert n.age_is_floor and "30h+" in n.shelf_life


# ---------- Fed and macro odds ----------
def _econ():
    return json.loads((FX / "voxodds_economics.json").read_text(encoding="utf-8"))


def test_only_rates_and_macro_markets_are_kept_from_the_economics_category():
    s = pr.parse_markets(_econ(), macro=True)
    qs = [m["question"] for m in s.markets]
    assert qs and all("IPO" not in q for q in qs) and any("Fed interest rates" in q for q in qs)
    assert all(m["assets"] == ["MACRO"] for m in s.markets)


def test_one_half_of_the_odds_feed_failing_is_degraded_and_named():
    crypto = pr.parse_markets(json.loads((FX / "voxodds_crypto.json").read_text(encoding="utf-8")))
    down = pr.PredictionSnapshot.unavailable("voxodds", "ReadTimeout")
    m = pr.merge(crypto, down)
    assert m.available and m.error == "macro: ReadTimeout" and len(m.markets) == len(crypto.markets)
    assert not pr.merge(down, down).available


def test_both_categories_are_requested():
    seen = []

    class T(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            seen.append(request.url.params["category"])
            body = _econ() if request.url.params["category"] == "Economics" else \
                json.loads((FX / "voxodds_crypto.json").read_text(encoding="utf-8"))
            return httpx.Response(200, json=body, request=request)

    async def go():
        async with httpx.AsyncClient(transport=T()) as c:
            return await pr.fetch_predictions(c)
    s = asyncio.run(go())
    assert sorted(seen) == ["Crypto", "Economics"] and s.tradeable("MACRO") and s.tradeable("BTC")


# ---------- what the agents and the page get ----------
def _market():
    import os
    os.environ["TI_OFFLINE_FIXTURES"] = "1"
    from core.data.offline import fixture_market
    return fixture_market()


def test_every_agent_gets_the_macro_read():
    from core.agents.context import common_payload
    out = common_payload(_market(), {})["macro_read"]
    assert {"prints", "narratives", "fed_and_macro_odds", "method"} <= set(out)
    assert all(o["band"] in ("clean", "watch") for o in out["fed_and_macro_odds"]), "fragile quotes are not reasoned from"


def test_one_part_of_the_macro_read_failing_leaves_the_rest(monkeypatch):
    from core.agents.context import macro_read

    def boom(*a, **k):
        raise RuntimeError("bad candles")
    monkeypatch.setattr(et, "read_all", boom)
    out = macro_read(_market())
    assert out["prints"] == {"available": False, "note": "RuntimeError: bad candles"}
    assert isinstance(out["narratives"], list)


def test_the_cards_show_the_read_the_numbers_and_the_method():
    from ui import panels
    path = lambda h: _ramp(226, 250, 3.0)(h) if h <= 250 else _ramp(250, 256, -2.6)(h) * 1.03  # noqa: E731
    r = _read(EVENT + 10 * H, path)
    h = panels.event_traps_html([r], EVENT + 10 * H)
    assert "sell the news · longs trapped" in h and "A bull trap" in h and f"into it {r.runup_pct:+.2f}%" in h
    assert "no actual figures" in h and "ti-card warn" in h
    n = _narr(_h("flows", 1, NOW, "ETF inflows jump"), _h("flows", 3, NOW))
    nh = panels.narratives_html(list(n.values()))
    assert "Flows" in nh and "fresh" in nh and "ETF inflows jump" in nh and "not a forecast" in nh
    assert "No high-impact USD prints" in panels.event_traps_html([], 0)


def test_a_sell_off_bought_back_is_labelled_buy_the_news():
    from ui import panels
    path = lambda h: _ramp(226, 250, -3.0)(h) if h <= 250 else 97.0 * (1 + 0.026 * min(h - 250, 6) / 6)  # noqa: E731
    h = panels.event_traps_html([_read(EVENT + 10 * H, path)], EVENT + 10 * H)
    assert "buy the news · shorts trapped" in h and "sell the news" not in h.split("ti-chip")[1][:40]
    pos = panels.event_traps_html([_read(EVENT - 5 * H, _ramp(221, 245, 3.0))], EVENT - 5 * H)
    assert "positioned · longs at risk</span>" in pos


def test_only_live_reads_reach_the_desks():
    from ui import panels
    assert set(panels.ACTIVE_EVENT_STATES) == {"positioned", "reacting", "sell_the_news", "spike_faded", "confirmed"}


def test_the_odds_panel_names_the_macro_filter_in_words():
    from ui import panels
    h = panels.predictions_html(_market().predictions, 0, asset="MACRO")
    assert "on the Fed, rates and the economy" in h and "mentioning MACRO" not in h
