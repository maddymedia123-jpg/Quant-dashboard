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
    return json.loads((FX / "frankfurter_usdjpy.json").read_text(encoding="utf-8"))


def rows(values: list[float], start=(2026, 10, 1)) -> list[tuple[int, float]]:
    first = day(*start)
    return [(first + i * 86_400_000, v) for i, v in enumerate(values)]


# ---------- parsing ----------
def test_the_recorded_window_parses_oldest_first_on_business_days_only():
    r = fx.parse_range(load())
    assert len(r) == 52 and r[0][0] < r[-1][0]
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
        and u.params["symbols"] == "JPY"

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
