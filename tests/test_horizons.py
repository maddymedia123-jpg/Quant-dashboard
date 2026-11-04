"""The 4-hour tactical execution window against the 24-hour horizon (roadmap item 12).

The framing is arithmetic on numbers already on the screen, so every state has a synthetic day built
to produce it, and the plan check is tried on plans inside, beyond and outside the day's edges."""
from types import SimpleNamespace

import pandas as pd
import pytest

from core.agents.schemas import PlanLevels
from core.indicators import horizons
from core.indicators.levels import Level
from ui import panels

H = 3_600_000
DAY = 1_793_750_400_000   # 2026-11-04 00:00 UTC


def frame(rows):
    """rows: (timestamp, open, high, low, close)."""
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close"])


def market(price=80_000.0, day=(79_000, 81_000), prior=(78_000, 80_500), four=(79_800, 80_300)):
    frames = {
        "1d": frame([(DAY - 24 * H, 78_500, prior[1], prior[0], 79_200), (DAY, 79_200, day[1], day[0], price)]),
        "4h": frame([(DAY + 8 * H, 79_900, 80_400, 79_700, 80_100), (DAY + 12 * H, 80_100, four[1], four[0], price)]),
    }
    return SimpleNamespace(spot=SimpleNamespace(frames=frames, last=price))


def analyses(price=80_000.0):
    lv = lambda p, kind, n=3: Level(price=p, touches=n, kind=kind, last_touch_ms=DAY)  # noqa: E731
    live = SimpleNamespace(price=price, levels=[lv(79_600, "support"), lv(80_400, "resistance", 2), lv(78_000, "support")])
    intraday = SimpleNamespace(price=price, levels=[lv(78_900, "support"), lv(81_200, "resistance", 4)])
    return {"live": live, "intraday": intraday}


def state(live_range, intraday_range=(78_800, 81_300), vwap=79_900):
    return {"forecasts": {"live": {"available": True, "range_68": list(live_range)},
                          "intraday": {"available": True, "range_68": list(intraday_range)}},
            "cascade": {"live": {"vwap": vwap}},
            "derivs": {"intraday": {"gamma": {"call_wall": 82_000, "put_wall": 78_000}, "liquidations": {"over": "24h", "total_usd": 1.0e8}}}}


# ---------- the block ----------
def test_the_block_carries_both_horizons_with_their_sources():
    hz = horizons.read(market(), analyses(), DAY + 13 * H, **state((79_700, 80_300)))
    four, day = hz["four_hour"], hz["twenty_four_hour"]
    assert four["desk"] == "live" and four["candle"]["high"] == 80_300 and four["prior_candle"]["high"] == 80_400
    assert four["expected_range_68"] == [79_700, 80_300]
    assert four["levels"] == {"floor": {"price": 79_600, "touches": 3}, "ceiling": {"price": 80_400, "touches": 2}}
    assert day["desk"] == "intraday" and day["day"] == {"opened": "00:00 UTC Wed 04 Nov", "open": 79_200, "high": 81_000, "low": 79_000, "close": 80_000}
    assert day["prior_day"]["high"] == 80_500 and day["prior_day"]["low"] == 78_000
    assert day["day_vwap"] == 79_900 and day["expected_range_68"] == [78_800, 81_300]
    assert day["levels"] == {"floor": {"price": 78_900, "touches": 3}, "ceiling": {"price": 81_200, "touches": 4}}
    assert day["gamma_walls"] == {"call_wall": 82_000, "put_wall": 78_000} and day["liquidations"]["over"] == "24h"
    assert "method" in hz and hz["framing"]["state"] == "inside"


def test_missing_frames_and_forecasts_leave_parts_absent_not_invented():
    m = SimpleNamespace(spot=SimpleNamespace(frames={}, last=80_000.0))
    hz = horizons.read(m, {}, DAY)
    assert hz["four_hour"]["candle"] is None and hz["twenty_four_hour"]["day"] is None
    assert hz["four_hour"]["expected_range_68"] is None and hz["twenty_four_hour"]["day_vwap"] is None
    assert "levels" not in hz["four_hour"] and "gamma_walls" not in hz["twenty_four_hour"]
    assert hz["framing"]["state"] == "unknown" and "no 4-hour expected range" in hz["framing"]["note"]


# ---------- the framing ----------
def test_a_range_inside_the_day_names_the_next_edges_either_side():
    hz = horizons.read(market(), analyses(), DAY + 13 * H, **state((79_700, 80_300)))
    f = hz["framing"]
    assert f["state"] == "inside" and f["reaches_above"] == [] and f["reaches_below"] == []
    assert f["position_in_day_pct"] == 50 and f["day_range"] == [79_000, 81_000] and f["four_hour_vs_day_width_pct"] == 30
    assert "mid-range in the day" in f["note"] and "the prior day's high 80,500 above" in f["note"]
    assert "the day's low 79,000 below" in f["note"] and "inside the day's range, not a break" in f["note"]
    assert f["note"].endswith("the day's average price 79,900 is below price"), "the average price is named, never an edge"


def test_a_range_reaching_the_days_high_is_a_probe_above():
    hz = horizons.read(market(price=80_800, prior=(78_000, 80_200)), analyses(80_800), DAY + 13 * H, **state((80_400, 81_100)))
    f = hz["framing"]
    assert f["state"] == "probe_above" and f["reaches_above"] == ["the day's high"] and f["reaches_below"] == []
    assert f["position_in_day_pct"] == 90 and "at the day's high" in f["note"]
    assert "reaches the day's high above (81,000)" in f["note"] and "a rejection there is the trap" in f["note"]


def test_a_range_reaching_the_days_low_and_the_prior_low_is_a_probe_below():
    hz = horizons.read(market(price=79_200), analyses(79_200), DAY + 13 * H, **state((77_900, 79_500)))
    f = hz["framing"]
    assert f["state"] == "probe_below" and f["reaches_above"] == []
    assert f["reaches_below"] == ["the day's low", "the 24-hour expected low", "the prior day's low"]
    assert f["position_in_day_pct"] == 10 and "at the day's low" in f["note"]


def test_a_range_wider_than_the_day_is_an_expansion_window():
    hz = horizons.read(market(), analyses(), DAY + 13 * H, **state((78_700, 81_400)))
    f = hz["framing"]
    assert f["state"] == "expansion" and f["four_hour_vs_day_width_pct"] == 135
    assert "the day's high" in f["reaches_above"] and "the day's low" in f["reaches_below"]
    assert "an expansion window, not a range" in f["note"]


def test_the_edge_band_is_a_fifth_of_the_day():
    assert horizons.frame(80_600, {"expected_range_68": [80_500, 80_700]}, {"day": {"high": 81_000, "low": 79_000}})["note"].startswith("Price 80,600 is at the day's high")
    assert "mid-range" in horizons.frame(80_599, {"expected_range_68": [80_500, 80_700]}, {"day": {"high": 81_000, "low": 79_000}})["note"]


def test_no_price_or_no_range_cannot_be_framed():
    assert horizons.frame(None, {"expected_range_68": [1, 2]}, {})["state"] == "unknown"
    assert horizons.frame(80_000, {"expected_range_68": None}, {})["state"] == "unknown"


# ---------- the plan against the day's edges ----------
def plan(side="SHORT", zone=(80_200, 80_400), stop=80_700, targets=(79_600, 79_100)):
    return PlanLevels(side=side, zone_low=zone[0], zone_high=zone[1], stop=stop, targets=list(targets),
                      confirm_timeframe="1h", confirm_close=80_150)


def hz():
    return horizons.read(market(), analyses(), DAY + 13 * H, **state((79_700, 80_300)))


def test_a_plan_inside_the_days_edges_is_tactical():
    c = horizons.plan_check(plan(), hz())
    assert c["verdict"] == "tactical" and c["horizon"] == [78_000, 81_300]
    assert c["horizon_edges"] == {"upper": "the 24-hour expected high", "lower": "the prior day's low"}
    assert c["targets_beyond_horizon"] == [] and not c["stop_beyond_horizon"]
    assert "a tactical trade inside the day" in c["note"] and c["framing_state"] == "inside" and c["framing_note"]


def test_a_target_beyond_the_days_edge_needs_the_horizon_taken():
    c = horizons.plan_check(plan(targets=(79_600, 77_500)), hz())
    assert c["verdict"] == "reaches_beyond" and c["targets_beyond_horizon"] == [77_500]
    assert "77,500 lie beyond the prior day's low 78,000" in c["note"] and "needs the 24-hour horizon taken" in c["note"]
    c = horizons.plan_check(plan("LONG", (79_600, 79_800), 79_300, (81_000, 82_500)), hz())
    assert c["verdict"] == "reaches_beyond" and "beyond the 24-hour expected high 81,300" in c["note"]


def test_a_zone_outside_the_days_edges_is_a_trade_at_the_edge():
    c = horizons.plan_check(plan(zone=(81_400, 81_600), stop=81_900, targets=(80_000,)), hz())
    assert c["verdict"] == "outside" and "a trade at the day's edge, not inside it" in c["note"]


def test_a_stop_beyond_the_horizon_is_named():
    c = horizons.plan_check(plan(stop=81_500), hz())
    assert c["verdict"] == "tactical" and c["stop_beyond_horizon"] and "a day-edge sweep is what stops it out" in c["note"]


def test_no_plan_no_horizon_or_no_edges_give_no_verdict():
    c = horizons.plan_check(PlanLevels(side="NONE", why_none="no edge"), hz())
    assert c["verdict"] == "no_trade" and c["framing_state"] == "inside"
    assert horizons.plan_check(None, hz()) is None and horizons.plan_check(plan(), None) is None
    bare = {"framing": {"state": "unknown", "note": "n"}, "twenty_four_hour": {}}
    assert horizons.plan_check(plan(), bare)["verdict"] == "unknown"


# ---------- the card ----------
def test_the_director_card_shows_the_framing_the_sentences_and_the_verdict():
    d = SimpleNamespace(horizon_framing="The short needs the day's low taken.",
                        horizon_check=horizons.plan_check(plan(targets=(79_600, 77_500)), hz()))
    h = panels.horizon_block_html(d)
    assert "4-hour zone vs 24-hour horizon" in h and "Price 80,000 is mid-range in the day" in h
    assert "The short needs the day" in h and "needs the day&#x27;s edge taken" in h and "77,500 lie beyond" in h
    assert panels.horizon_block_html(SimpleNamespace(horizon_framing="", horizon_check=None)) == ""
    quiet = panels.horizon_block_html(SimpleNamespace(horizon_framing="", horizon_check={"verdict": "no_trade", "framing_note": "Price 1 is x"}))
    assert "Price 1 is x" in quiet and "no plan" not in quiet
    assert "&lt;b&gt;" in panels.horizon_block_html(SimpleNamespace(horizon_framing="<b>", horizon_check=None))
