"""Everything the dashboard shows reaches the agents: COT, US yields and the dollar, gold, and each desk's
forecast pane joined the war-room state on 20 Oct (they were on screen but not in the payload)."""
import json
import os
import tempfile

import pytest

os.environ["TI_OFFLINE_FIXTURES"] = "1"
os.environ.setdefault("TI_DATA_DIR", tempfile.mkdtemp())

from core.agents import context  # noqa: E402
from core.agents.live_recon import build_state  # noqa: E402
from core.agents.recon_profiles import PROFILES  # noqa: E402
from core.config import CATEGORIES, TF_MINUTES  # noqa: E402
from core.data.offline import fixture_market  # noqa: E402
from core.data.types import COTSnapshot, MacroSnapshot, MetalsSnapshot  # noqa: E402
from core.indicators.category import analyze_category  # noqa: E402


@pytest.fixture(scope="module")
def market():
    m = fixture_market()
    m.cot = COTSnapshot(source="cftc", market="BITCOIN - CME", spec_net=-1200, spec_net_change=300, cot_index=18.0,
                        weeks_in_index=52, open_interest=24_000, contract_btc=5.0)
    m.metals = MetalsSnapshot(source="goldprice.dev", xau_usd=4148.55)
    m.macro = MacroSnapshot(source="treasury.gov", spread_2s10s=0.35, series={
        "DGS10": {"available": True, "label": "US 10-year yield", "unit": "%", "value": 5.09, "change": -0.11,
                  "age_days": 1.0, "stale": False},
        "DTWEXBGS": {"available": False, "note": "403"}})
    return m


@pytest.fixture(scope="module")
def analyses(market):
    return {k: analyze_category(c, market.spot.frames, market.futures) for k, c in CATEGORIES.items()}


def test_cot_reaches_the_derivatives_payload(market):
    data, missing = context.derivatives_payload(market)
    assert data["cot"]["cot_index"] == 18.0 and data["cot"]["spec_net"] == -1200
    assert "crowded short" in data["cot"]["reading"] and "cot" not in missing


def test_a_missing_cot_is_named_not_silent(market):
    m = market.model_copy(update={"cot": COTSnapshot.unavailable("cftc", "socrata 500")})
    data, missing = context.derivatives_payload(m)
    assert "cot" not in data and "cot" in missing


def test_yields_dollar_and_gold_reach_every_agent_through_the_macro_read(market):
    read = context.macro_read(market)
    rates = read["rates_and_dollar"]
    assert rates["source"] == "treasury.gov" and rates["DGS10"]["value"] == 5.09 and rates["DGS10"]["change_1w"] == -0.11
    assert rates["DTWEXBGS"] == {"available": False, "note": "403"}, "a missing series is named"
    assert rates["spread_2s10s_pct"] == 0.35
    assert read["gold"] == {"available": True, "xau_usd": 4148.55, "stale": False, "source": "goldprice.dev"}


def test_each_war_room_gets_its_own_desks_forecast(market, analyses):
    for key, profile in PROFILES.items():
        state = build_state(market, analyses, profile=profile)
        f = state["forecast"]
        assert f["available"], key
        desk = CATEGORIES[profile.category]
        assert f["candles_ahead"] == desk.horizon_bars, f"{key} judged another desk's forecast"
        assert f["candle_minutes"] == TF_MINUTES[desk.chart_tf], key
        assert f["range_68"][0] < f["path_end_median"] < f["range_68"][1]
        assert 0 <= f["p_close_higher"] <= 1 and all(0 <= lv["p_trades_there"] <= 1 for lv in f["level_touch_odds"])
        assert f["track_record"]["candles_replayed"] > 0 and "no edge" in f["track_record"]["note"]
        assert state["macro_read"]["gold"]["available"] and "cot" in state["derivatives"]
        json.dumps(state, default=str)


def test_a_broken_forecast_does_not_take_the_war_room_down(market, analyses, monkeypatch):
    import core.indicators.forecast as fc

    def boom(*a, **k):
        raise ValueError("no candles")
    monkeypatch.setattr(fc, "desk_forecast", boom)
    state = build_state(market, analyses, profile=PROFILES["live"])
    assert state["forecast"] == {"available": False, "note": "ValueError: no candles"}
