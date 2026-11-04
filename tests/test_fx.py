"""USD/JPY from frankfurter.dev and the yen carry-trade read (roadmap item 10).

Found by calling the feed: the old host 301s, the range form answers ECB business days only, and the
payload names its base currency - so the parser refuses a response for another base (the
goldprice.dev wrong-currency bug, in another feed). The BoJ rate is hand-set in the environment and
never guessed: unset, the carry gap is absent and the read says so."""
import asyncio
import datetime as dt
import json
import pathlib

import httpx
import pytest

from core.data import fx
from core.data.offline import fixture_market
from core.data.types import FxSnapshot, MacroSnapshot
from ui import panels

FX = pathlib.Path(__file__).parent / "fixtures"


def day(y, m, d) -> int:
    return int(dt.datetime(y, m, d, tzinfo=dt.timezone.utc).timestamp() * 1000)


RECORDED = day(2026, 11, 4)


def load():
    return json.loads((FX / "frankfurter_usd_majors.json").read_text(encoding="utf-8"))


def rows(values: list[float], start=(2026, 10, 1)) -> list[tuple[int, float]]:
    first = day(*start)
    return [(first + i * 86_400_000, v) for i, v in enumerate(values)]


# ---------- parsing ----------
def test_the_recorded_window_parses_oldest_first_on_business_days_only():
    r = fx.parse_range(load())
    assert len(r) == 34 and r[0][0] < r[-1][0]
    assert r[-1] == (RECORDED, 158.25)
    assert all(dt.datetime.fromtimestamp(t / 1000, dt.timezone.utc).weekday() < 5 for t, _ in r), "no weekends"


@pytest.mark.parametrize("payload", [
    {"base": "EUR", "rates": {"2026-11-04": {"JPY": 170.0}}},
    {"rates": {"2026-11-04": {"JPY": 170.0}}},
    {"base": "USD", "rates": {}},
    {"base": "USD", "rates": {"2026-11-04": {"EUR": 0.9}}},
    [],
])
def test_a_response_that_is_not_usd_jpy_is_refused(payload):
    with pytest.raises(ValueError):
        fx.parse_range(payload)


def test_a_bad_day_is_skipped_not_fatal():
    r = fx.parse_range({"base": "USD", "rates": {"2026-11-04": {"JPY": 158.0}, "junk": {"JPY": 1.0},
                                                   "2026-11-03": {"JPY": "x"}}})
    assert r == [(RECORDED, 158.0)]


# ---------- the snapshot ----------
def test_the_recorded_window_builds_a_fresh_snapshot():
    s = fx.build(fx.parse_range(load()), RECORDED)
    assert s.available and s.rate == 158.25 and s.as_of_ms == RECORDED and not s.stale and s.age_days == 0.0
    assert s.week_ago == 157.96 and s.change_1w_pct == pytest.approx(0.18, abs=0.01)
    assert s.month_ago is not None and s.change_1m_pct is not None and s.boj_rate_pct is None


def test_the_week_and_month_are_sessions_back_and_absent_when_short():
    s = fx.build(rows([150.0, 151.0, 152.0, 153.0, 154.0, 155.0, 156.0]), day(2026, 10, 7))
    assert s.week_ago == 151.0 and s.change_1w_pct == pytest.approx(3.31, abs=0.01)
    assert s.month_ago is None and s.change_1m_pct is None
    short = fx.build(rows([150.0, 151.0]), day(2026, 10, 2))
    assert short.available and short.change_1w_pct is None


def test_a_weekend_old_fix_is_normal_but_a_week_old_one_is_stale():
    r = rows([150.0])
    assert not fx.build(r, day(2026, 10, 4)).stale, "three days over a weekend"
    assert fx.build(r, day(2026, 10, 7)).stale


@pytest.mark.parametrize("rate", [1.5825, 15825.0, 20.0])
def test_a_rate_no_market_has_printed_is_unavailable_with_a_reason(rate):
    s = fx.build(rows([rate]), day(2026, 10, 1))
    assert not s.available and "plausible" in s.error


def test_no_rows_is_unavailable():
    assert not fx.build([], RECORDED).available


# ---------- the hand-set BoJ rate ----------
@pytest.mark.parametrize("raw,expected", [
    ("", None), ("0.75", 0.75), ("0.75%", 0.75), (" 0.5 ", 0.5), ("abc", None), ("75", None), ("-5", None),
])
def test_the_boj_rate_is_read_from_the_environment_or_absent(raw, expected):
    assert fx.boj_rate({fx.BOJ_ENV: raw}) == expected
    assert fx.boj_rate({}) is None


def test_the_fetch_asks_for_the_window_in_usd_jpy_and_carries_the_boj_rate(monkeypatch):
    seen = []

    def handler(request: httpx.Request):
        seen.append(request)
        return httpx.Response(200, json=load())

    monkeypatch.setenv(fx.BOJ_ENV, "0.75")
    s = asyncio.run(fx.fetch_fx(httpx.AsyncClient(transport=httpx.MockTransport(handler)), RECORDED))
    assert s.rate == 158.25 and s.boj_rate_pct == 0.75
    u = seen[0].url
    assert u.host == "api.frankfurter.dev" and u.path == "/v1/2026-09-20.." and u.params["base"] == "USD" \
        and u.params["symbols"] == "EUR,JPY,GBP,CAD,SEK,CHF"
    assert s.dxy == pytest.approx(101.255, abs=0.001), "the index comes from the same request"

    def big(request: httpx.Request):
        return httpx.Response(200, text="[" + "1," * 200_000 + "1]")

    with pytest.raises(ValueError, match="KB"):
        asyncio.run(fx.fetch_fx(httpx.AsyncClient(transport=httpx.MockTransport(big)), RECORDED))


# ---------- the carry read ----------
def snap(**kw) -> FxSnapshot:
    base = dict(source="frankfurter", rate=150.0, as_of_ms=RECORDED, week_ago=152.0, change_1w_pct=-1.3,
                month_ago=155.0, change_1m_pct=-3.2, age_days=0.0)
    base.update(kw)
    return FxSnapshot(**base)


def test_the_yen_strengthening_fast_is_unwind_risk():
    r = fx.yen_read(snap(change_1w_pct=-1.5))
    assert r["state"] == "unwind_risk" and "down 1.5% in a week" in r["note"] and "unwind" in r["note"]
    assert r["carry_pct"] is None and "BoJ rate not set" in r["note"]


def test_the_yen_weakening_is_carry_on_and_a_small_move_is_quiet():
    assert fx.yen_read(snap(change_1w_pct=1.6))["state"] == "carry_on"
    r = fx.yen_read(snap(change_1w_pct=-1.4))
    assert r["state"] == "quiet" and "-1.4% on the week" in r["note"]


def test_the_carry_gap_appears_only_when_both_rates_are_known():
    r = fx.yen_read(snap(boj_rate_pct=0.75), effr_pct=3.88)
    assert r["carry_pct"] == pytest.approx(3.13) and "+3.13 pp" in r["note"] and "BoJ 0.75%" in r["note"]
    assert fx.yen_read(snap(boj_rate_pct=0.75), effr_pct=None)["carry_pct"] is None
    assert fx.yen_read(snap(), effr_pct=3.88)["carry_pct"] is None


def test_no_week_ago_quote_or_no_feed_is_unknown():
    assert fx.yen_read(snap(change_1w_pct=None))["state"] == "unknown"
    r = fx.yen_read(FxSnapshot.unavailable("frankfurter", "503"))
    assert r["state"] == "unknown" and r["carry_pct"] is None


# ---------- the card ----------
def macro(effr=3.88) -> MacroSnapshot:
    return MacroSnapshot(source="fred", series={"EFFR": {"id": "EFFR", "available": effr is not None, "value": effr}})


def test_the_card_shows_the_rate_the_changes_the_rates_and_the_read():
    h = panels.yen_html(snap(change_1w_pct=-2.0, boj_rate_pct=0.75), macro(), RECORDED)
    assert "150.00" in h and "Wed 04 Nov" in h and "-2.00%" in h and "-3.20%" in h
    assert "0.75% (hand-set)" in h and "3.88%" in h and "+3.13 pp" in h
    assert "unwind risk" in h and "ti-card warn" in h and "stale" not in h


def test_the_card_says_when_the_boj_rate_is_not_set_and_marks_a_stale_fix():
    h = panels.yen_html(snap(stale=True), macro(None), RECORDED)
    assert "not set" in h and ">stale<" in h and "Carry (US minus Japan)" not in h and "US funds rate" not in h
    assert "quiet" in h and "ti-card neutral" in h


def test_the_card_names_a_missing_feed_and_escapes():
    h = panels.yen_html(FxSnapshot.unavailable("frankfurter", "<503>"), macro(), RECORDED)
    assert "No USD/JPY feed" in h and "&lt;503&gt;" in h


# ---------- the agents ----------
def test_the_fixture_market_carries_the_yen_read_to_the_macro_read():
    from core.agents import context
    m = fixture_market()
    assert m.fx.source == "fixture" and m.fx.rate == 158.25 and not m.fx.stale
    read = context.macro_read(m)["yen_carry"]
    assert read["available"] and read["usdjpy"] == 158.25 and read["state"] == "quiet"
    assert read["us_funds_rate_pct"] == 3.88 and read["boj_rate_pct"] is None and read["carry_pct"] is None
    m2 = m.model_copy(update={"fx": m.fx.model_copy(update={"boj_rate_pct": 0.5, "change_1w_pct": -2.2})})
    r2 = context.macro_read(m2)["yen_carry"]
    assert r2["state"] == "unwind_risk" and r2["carry_pct"] == pytest.approx(3.38)
    m3 = m.model_copy(update={"fx": FxSnapshot.unavailable("frankfurter", "503")})
    assert context.macro_read(m3)["yen_carry"] == {"available": False, "note": "503"}


# ---------- roadmap item 11: the dollar index rebuilt from the six rates ----------
LAST_DAY = {"CAD": 1.3935, "CHF": 0.81548, "EUR": 0.88755, "GBP": 0.74075, "JPY": 158.25, "SEK": 9.738}


def test_the_recorded_window_carries_all_six_majors():
    rates = fx.parse_rates(load())
    assert set(rates) == set(fx.MAJORS) and all(len(rates[c]) == 34 for c in fx.MAJORS)
    assert {c: rates[c][-1][1] for c in fx.MAJORS} == LAST_DAY and rates["EUR"][-1][0] == RECORDED


def test_the_weights_are_ice_weights_and_sum_to_one():
    assert fx.DXY_WEIGHTS == {"EUR": 0.576, "JPY": 0.136, "GBP": 0.119, "CAD": 0.091, "SEK": 0.042, "CHF": 0.036}
    assert sum(fx.DXY_WEIGHTS.values()) == pytest.approx(1.0)


def test_the_index_is_the_published_formula_on_per_dollar_quotes():
    """Worked by hand from the 4 November fixes: EURUSD 1.12670, GBPUSD 1.34998, so the euro and sterling
    terms are 1.12670^-0.576 = 0.93360 and 1.34998^-0.119 = 0.96492; the yen, CAD, krona and franc terms
    are 158.25^0.136 = 1.99118, 1.3935^0.091 = 1.03066, 9.738^0.042 = 1.10031, 0.81548^0.036 = 0.99268;
    50.14348112 x the product = 101.255."""
    series = fx.dxy_series(fx.parse_rates(load()))
    assert len(series) == 34 and series[-1] == (RECORDED, pytest.approx(101.255, abs=0.001))
    assert series[-6][1] == pytest.approx(102.348, abs=0.001)
    one = fx.dxy_series({c: [(RECORDED, v)] for c, v in LAST_DAY.items()})
    assert one == [(RECORDED, pytest.approx(101.255, abs=0.001))]


def test_a_day_missing_one_rate_is_skipped_not_filled():
    rates = {c: [(RECORDED, v), (RECORDED + 86_400_000, v)] for c, v in LAST_DAY.items()}
    rates["SEK"] = [(RECORDED, 9.738)]
    assert [t for t, _ in fx.dxy_series(rates)] == [RECORDED]
    assert fx.dxy_series({"EUR": [(RECORDED, 0.9)]}) == []
    rates = {c: [(RECORDED, v)] for c, v in LAST_DAY.items()}
    rates["JPY"] = [(RECORDED, 0.0)]
    assert fx.dxy_series(rates) == []


def test_the_snapshot_carries_the_index_with_its_week_and_month():
    rates = fx.parse_rates(load())
    s = fx.build(rates["JPY"], RECORDED, dxy_rows=fx.dxy_series(rates))
    assert s.dxy == pytest.approx(101.255, abs=0.001) and s.dxy_as_of_ms == RECORDED
    assert s.dxy_week_ago == pytest.approx(102.348, abs=0.001) and s.dxy_change_1w_pct == pytest.approx(-1.07, abs=0.01)
    assert s.dxy_month_ago is not None and s.dxy_change_1m_pct is not None and s.dxy_note is None
    assert s.rate == 158.25, "the yen read is untouched"


def test_no_index_rows_or_an_implausible_index_leave_the_index_none_with_a_note():
    jpy = rows([158.0, 158.1, 158.2])
    s = fx.build(jpy, day(2026, 10, 3), dxy_rows=[])
    assert s.available and s.dxy is None and "all six rates" in s.dxy_note
    s = fx.build(jpy, day(2026, 10, 3), dxy_rows=[(day(2026, 10, 3), 1012.55)])
    assert s.available and s.dxy is None and "plausible" in s.dxy_note
    assert fx.build(jpy, day(2026, 10, 3)).dxy is None and fx.build(jpy, day(2026, 10, 3)).dxy_note is None


def dxy_snap(**kw) -> FxSnapshot:
    base = dict(source="frankfurter", rate=158.25, as_of_ms=RECORDED, change_1w_pct=0.2, dxy=101.26,
                dxy_as_of_ms=RECORDED, dxy_change_1w_pct=-1.07, dxy_change_1m_pct=-0.5, age_days=0.0)
    base.update(kw)
    return FxSnapshot(**base)


def test_a_dollar_spike_with_a_bitcoin_rally_is_the_false_breakout_signal():
    r = fx.dollar_read(dxy_snap(dxy_change_1w_pct=1.2), btc_week_pct=4.5)
    assert r["state"] == "spike_divergence" and "false-breakout" in r["note"] and "+4.5%" in r["note"]
    assert fx.dollar_read(dxy_snap(dxy_change_1w_pct=1.2), btc_week_pct=2.9)["state"] == "spike"
    assert fx.dollar_read(dxy_snap(dxy_change_1w_pct=0.99), btc_week_pct=4.5)["state"] == "quiet"
    r = fx.dollar_read(dxy_snap(dxy_change_1w_pct=1.2), btc_week_pct=None)
    assert r["state"] == "spike" and "same sessions" not in r["note"]


def test_a_sliding_dollar_is_a_tailwind_and_the_rest_is_quiet():
    r = fx.dollar_read(dxy_snap(), btc_week_pct=-2.0)
    assert r["state"] == "slide" and "tailwind" in r["note"] and "bitcoin -2.0%" in r["note"]
    assert fx.dollar_read(dxy_snap(dxy_change_1w_pct=0.4))["state"] == "quiet"


def test_no_index_or_no_week_ago_value_is_unknown_with_the_reason():
    assert fx.dollar_read(dxy_snap(dxy_change_1w_pct=None))["state"] == "unknown"
    r = fx.dollar_read(dxy_snap(dxy=None, dxy_note="no day in the window has all six rates"))
    assert r["state"] == "unknown" and r["note"] == "no day in the window has all six rates"
    r = fx.dollar_read(FxSnapshot.unavailable("frankfurter", "503"))
    assert r["state"] == "unknown" and r["note"] == "503"


def test_bitcoins_week_is_five_completed_daily_candles():
    import pandas as pd
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 110.0, 999.0]   # the last row is the forming candle
    frames = {"1d": pd.DataFrame({"close": closes})}
    assert fx.btc_sessions_change_pct(frames) == pytest.approx((110.0 / 101.0 - 1) * 100, abs=0.01)
    assert fx.btc_sessions_change_pct({"1d": pd.DataFrame({"close": closes[:6]})}) is None, "too short"
    assert fx.btc_sessions_change_pct({}) is None and fx.btc_sessions_change_pct(None) is None
    assert fx.btc_sessions_change_pct({"1d": pd.DataFrame({"open": closes})}) is None, "no close column"


def test_the_dollar_card_shows_the_index_the_changes_bitcoin_and_the_read():
    import pandas as pd
    spot = type("S", (), {"frames": {"1d": pd.DataFrame({"close": [100.0] * 3 + [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0]})}})()
    h = panels.dollar_html(dxy_snap(dxy_change_1w_pct=1.5), spot, RECORDED)
    assert "101.26" in h and "Wed 04 Nov" in h and "+1.50%" in h and "-0.50%" in h and "+5.0%" in h
    assert "false-breakout signal" in h and "ti-card warn" in h and "reconstructed" in h.lower()
    h = panels.dollar_html(dxy_snap(), type("S", (), {"frames": {}})(), RECORDED)
    assert "dollar sliding" in h and "no daily candles" in h and "ti-card neutral" in h


def test_the_dollar_card_names_a_missing_index_and_escapes():
    h = panels.dollar_html(FxSnapshot.unavailable("frankfurter", "<503>"), None, RECORDED)
    assert "No dollar index" in h and "&lt;503&gt;" in h
    h = panels.dollar_html(dxy_snap(dxy=None, dxy_note="no day in the window has all six rates"), None, RECORDED)
    assert "all six rates" in h


def test_the_fixture_market_carries_the_dollar_index_to_the_macro_read():
    from core.agents import context
    m = fixture_market()
    assert m.fx.dxy == pytest.approx(101.255, abs=0.001)
    read = context.macro_read(m)["dollar_index"]
    assert read["available"] and read["dxy_reconstructed"] == pytest.approx(101.255, abs=0.001)
    assert read["change_1w_pct"] == pytest.approx(-1.07, abs=0.01) and read["state"] == "slide"
    assert read["btc_same_sessions_pct"] is not None
    m2 = m.model_copy(update={"fx": m.fx.model_copy(update={"dxy": None, "dxy_note": "no day in the window has all six rates"})})
    assert context.macro_read(m2)["dollar_index"] == {"available": False, "note": "no day in the window has all six rates"}
    m3 = m.model_copy(update={"fx": FxSnapshot.unavailable("frankfurter", "503")})
    assert context.macro_read(m3)["dollar_index"] == {"available": False, "note": "503"}
