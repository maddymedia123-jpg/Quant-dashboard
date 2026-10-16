"""US macro from FRED: the dollar, the 10-year, the 2-year and the 2s10s spread.

Each trap found by calling the feed has a test here:
  * holidays are EMPTY values, so the last row is not the last value;
  * the 10-year and 2-year can publish on different days, so the spread must use a shared date;
  * the broad dollar index is released WEEKLY, so one staleness threshold for all series would flag it
    stale every week before each release - a false alarm that teaches the reader to ignore the flag;
  * a response naming a different series is the goldprice.dev wrong-currency bug in another field.
"""
import datetime as dt
import pathlib

import httpx
import pytest

from core.data import macro
from core.data.types import MacroSnapshot
from ui import panels

FX = pathlib.Path(__file__).parent / "fixtures"


def load(name: str) -> str:
    return (FX / name).read_text(encoding="utf-8")


def day(y, m, d) -> int:
    return int(dt.datetime(y, m, d, tzinfo=dt.timezone.utc).timestamp() * 1000)


def csv(series_id: str, rows: list[tuple[str, str]]) -> str:
    return "observation_date," + series_id + "\n" + "\n".join(f"{d},{v}" for d, v in rows) + "\n"


RECORDED = day(2026, 10, 9)


# ---------- parsing ----------
def test_the_recorded_windows_parse():
    for sid in ("DTWEXBGS", "DGS10", "DGS2"):
        rows = macro.parse_series(load(f"fred_{sid}.csv"), sid)
        assert len(rows) > 20, sid
        assert rows == sorted(rows), "oldest first"


def test_a_holiday_is_skipped_rather_than_read_as_zero():
    """FRED writes holidays as an empty value. Read naively, a series ending on Christmas reports
    nothing, or zero."""
    rows = macro.parse_series(load("fred_DGS10_holidays.csv"), "DGS10")
    values = [v for _, v in rows]
    assert 0.0 not in values, "a blank must never become zero"
    dates = {dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%m-%d") for ms, _ in rows}
    assert "12-25" not in dates and "01-01" not in dates


@pytest.mark.parametrize("blank", ["", ".", "NA"])
def test_every_missing_marker_is_skipped(blank):
    rows = macro.parse_series(csv("DGS10", [("2026-10-05", "5.31"), ("2026-10-06", blank)]), "DGS10")
    assert rows == [(day(2026, 10, 5), 5.31)], "the blank last row leaves the last REAL value last"


def test_the_older_date_header_still_parses():
    """The header was `DATE,<ID>` before it became `observation_date,<ID>`; columns are read by position."""
    rows = macro.parse_series("DATE,DGS10\n2026-10-06,5.27\n", "DGS10")
    assert rows == [(day(2026, 10, 6), 5.27)]


def test_a_response_for_a_different_series_is_refused():
    """HTTP 200, plausible numbers, wrong thing - the same shape as goldprice.dev answering in AUD."""
    with pytest.raises(ValueError, match="asked for DGS10"):
        macro.parse_series(csv("DGS2", [("2026-10-06", "4.79")]), "DGS10")


def test_a_body_with_no_header_is_refused():
    with pytest.raises(ValueError):
        macro.parse_series("", "DGS10")


# ---------- the summary ----------
def test_the_summary_carries_value_date_and_a_week_of_sessions_change():
    rows = [(day(2026, 9, 28 + i), 5.0 + i * 0.05) for i in range(3)] + \
           [(day(2026, 10, 1 + i), 5.15 + i * 0.05) for i in range(5)]
    s = macro.summarise("DGS10", "US 10-year yield", "%", -2.0, 25.0, rows, day(2026, 10, 6), 5)
    assert s["available"] and s["value"] == rows[-1][1]
    assert s["as_of_ms"] == rows[-1][0]
    assert s["week_ago"] == rows[-1 - macro.WEEK_BACK_OBSERVATIONS][1]
    assert s["change"] == pytest.approx(rows[-1][1] - rows[-6][1])


def test_too_short_a_history_reports_no_change_rather_inventing_one():
    s = macro.summarise("DGS10", "x", "%", -2.0, 25.0, [(day(2026, 10, 6), 5.27)], day(2026, 10, 7), 5)
    assert s["available"] and s["change"] is None and s["week_ago"] is None


@pytest.mark.parametrize("value", [52.8, -9.0, 999.0])
def test_an_implausible_yield_is_refused(value):
    s = macro.summarise("DGS10", "x", "%", -2.0, 25.0, [(day(2026, 10, 6), value)], day(2026, 10, 7), 5)
    assert not s["available"] and "plausible" in s["note"]


def test_no_observations_is_unavailable_with_a_reason():
    s = macro.summarise("DGS10", "x", "%", -2.0, 25.0, [], day(2026, 10, 7), 5)
    assert not s["available"] and s["note"]


# ---------- staleness is per series ----------
def test_staleness_follows_each_series_own_cadence():
    """The broad dollar index is the Fed's H.10, released weekly; the yields are daily. One threshold
    would flag the dollar stale every week before each release."""
    cadences = {sid: stale for sid, _, _, _, _, stale in macro.SERIES}
    assert cadences["DTWEXBGS"] > 7, "a weekly release must tolerate more than a week"
    assert cadences["DGS10"] <= 5 and cadences["DGS2"] <= 5, "daily yields should not"


def test_a_week_old_dollar_print_is_normal_but_a_week_old_yield_is_not():
    eight_days_ago = day(2026, 10, 1)
    now = day(2026, 10, 9)
    dollar = macro.summarise("DTWEXBGS", "d", "index", 50, 250, [(eight_days_ago, 121.4)], now, 12)
    yield10 = macro.summarise("DGS10", "y", "%", -2, 25, [(eight_days_ago, 5.2)], now, 5)
    assert not dollar["stale"], "eight days is ordinary for a weekly release"
    assert yield10["stale"], "eight days is not ordinary for a daily one"


# ---------- the spread ----------
def test_the_spread_uses_the_latest_date_both_yields_share():
    """Subtracting Monday's 2-year from Tuesday's 10-year describes no day at all."""
    ten = [(day(2026, 10, 5), 5.31), (day(2026, 10, 6), 5.27)]
    two = [(day(2026, 10, 5), 4.80)]                       # the 2-year has not printed Tuesday yet
    spread, when = macro.curve_spread(ten, two)
    assert when == day(2026, 10, 5), "the shared date, not the 10-year's latest"
    assert spread == pytest.approx(0.51)


def test_no_shared_date_means_no_spread():
    assert macro.curve_spread([(day(2026, 10, 6), 5.27)], [(day(2026, 10, 5), 4.8)]) == (None, None)


def test_an_inverted_curve_is_negative():
    spread, _ = macro.curve_spread([(day(2026, 10, 6), 4.0)], [(day(2026, 10, 6), 4.6)])
    assert spread == pytest.approx(-0.6)


# ---------- assembly ----------
def test_the_recorded_fixtures_assemble_into_a_snapshot():
    texts = {sid: load(f"fred_{sid}.csv") for sid in ("DTWEXBGS", "DGS10", "DGS2")}
    snap = macro.build(texts, RECORDED)
    assert snap.available and snap.error is None
    assert set(snap.series) == {"DTWEXBGS", "DGS10", "DGS2"}
    assert all(s["available"] and not s["stale"] for s in snap.series.values())
    assert snap.spread_2s10s is not None and snap.spread_as_of_ms is not None


def test_one_series_failing_is_a_degraded_feed_that_names_it():
    texts = {"DTWEXBGS": load("fred_DTWEXBGS.csv"), "DGS10": httpx.ReadTimeout(""),
             "DGS2": load("fred_DGS2.csv")}
    snap = macro.build(texts, RECORDED)
    assert snap.available, "two of three is still a feed"
    assert not snap.series["DGS10"]["available"]
    assert "DGS10" in (snap.error or "") and "ReadTimeout" in (snap.error or "")
    assert snap.spread_2s10s is None, "no spread without both yields"


def test_the_2_year_failing_alone_gives_no_spread_and_does_not_crash():
    """The other half of the case above: with the 10-year present and the 2-year missing, the spread
    must be skipped - a guard that only checked the 10-year reached for a series never parsed."""
    texts = {"DTWEXBGS": load("fred_DTWEXBGS.csv"), "DGS10": load("fred_DGS10.csv"),
             "DGS2": httpx.ReadTimeout("")}
    snap = macro.build(texts, RECORDED)
    assert snap.available and snap.series["DGS10"]["available"]
    assert not snap.series["DGS2"]["available"]
    assert snap.spread_2s10s is None and snap.spread_as_of_ms is None


def test_every_series_failing_is_unavailable():
    snap = macro.build({sid: httpx.ConnectError("") for sid in ("DTWEXBGS", "DGS10", "DGS2")}, RECORDED)
    assert not snap.available and "ConnectError" in (snap.error or "")


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def test_the_request_asks_for_a_recent_window_and_follows_redirects():
    """cosd= takes a response from 104 KB of history since 2006 to a few hundred bytes."""
    seen = []

    def handler(request: httpx.Request):
        seen.append(str(request.url))
        sid = request.url.params["id"]
        return httpx.Response(200, text=load(f"fred_{sid}.csv"))

    snap = _run(macro.fetch_macro(httpx.AsyncClient(transport=httpx.MockTransport(handler)), RECORDED))
    assert snap.available
    assert len(seen) == 3 and all("cosd=" in u for u in seen)
    assert {httpx.URL(u).params["id"] for u in seen} == {"DTWEXBGS", "DGS10", "DGS2"}


def test_an_oversized_response_is_refused():
    def handler(request: httpx.Request):
        return httpx.Response(200, text="observation_date,DGS10\n" + "2026-10-06,5.27\n" * 100_000)

    snap = _run(macro.fetch_macro(httpx.AsyncClient(transport=httpx.MockTransport(handler)), RECORDED))
    assert not snap.available and "KB" in (snap.error or "")


# ---------- the panel ----------
def snapshot() -> MacroSnapshot:
    texts = {sid: load(f"fred_{sid}.csv") for sid in ("DTWEXBGS", "DGS10", "DGS2")}
    return macro.build(texts, RECORDED)


def test_the_rates_panel_shows_each_value_its_date_and_the_spread():
    h = panels.rates_html(snapshot(), RECORDED)
    assert "US 10-year yield" in h and "US 2-year yield" in h and "US dollar" in h
    assert "5.28%" in h and "4.77%" in h
    assert "as of Wed 07 Oct" in h, "the 10-year's own observation date"
    assert "as of Fri 02 Oct" in h, "and the dollar's, which is older - weekly release"
    assert "2s10s spread" in h and "bp" in h and "from the same day" in h


def test_an_observation_date_is_not_given_a_clock_time():
    """FRED observations are calendar dates, not instants. Printing "05:00 PKT" beside one would invent
    a time of day the data does not have, so the panel shows the date alone."""
    import re

    h = panels.rates_html(snapshot(), RECORDED)
    dated = h.split("<tbody>")[1].split("</tbody>")[0]
    assert not re.search(r"\d{2}:\d{2}", dated), "no clock time beside a daily observation"
    assert "PKT" not in dated and "UTC" not in dated


def test_yields_move_in_basis_points_and_the_dollar_in_points():
    """The exact figures, not just the unit: asserting " bp" alone passed with the change printed in
    percent, because -0.11 percent renders as "-0 bp"."""
    snap = snapshot()
    assert snap.series["DGS2"]["change"] == pytest.approx(-0.11)
    h = panels.rates_html(snap, RECORDED)
    assert "-11 bp" in h, "an eleven basis point fall in the 2-year"
    assert "+1.05" in h, "the dollar index change, in points"


def test_a_stale_series_is_flagged_on_the_panel():
    snap = snapshot()
    snap.series["DGS10"]["stale"] = True
    assert "stale" in panels.rates_html(snap, RECORDED)


def test_an_inverted_curve_says_so():
    snap = snapshot()
    snap.spread_2s10s = -0.42
    h = panels.rates_html(snap, RECORDED)
    assert "-42 bp" in h and "inverted" in h


def test_an_unavailable_series_shows_its_reason_not_a_number():
    snap = snapshot()
    snap.series["DGS2"] = {"id": "DGS2", "label": "US 2-year yield", "unit": "%", "available": False,
                           "note": "ReadTimeout"}
    h = panels.rates_html(snap, RECORDED)
    assert "ReadTimeout" in h


def test_no_macro_feed_says_so():
    h = panels.rates_html(MacroSnapshot.unavailable("fred", "every series refused"), RECORDED)
    assert "No macro feed" in h and "every series refused" in h and "<table>" not in h


def test_the_rates_panel_escapes_feed_strings():
    snap = snapshot()
    snap.series["DGS10"]["label"] = "<script>x</script>"
    snap.error = "<img src=x onerror=alert(1)>"
    h = panels.rates_html(snap, RECORDED)
    assert "<script>" not in h and "<img" not in h


def test_the_curated_calendar_fallback_was_not_shadowed():
    """panels already had a macro_html - the curated-calendar fallback calendar_html uses when the
    Forex Factory feed is down. Naming the new panel macro_html would have rebound that name and broken
    the fallback only on the day the calendar feed failed."""
    from core.config import MACRO_EVENTS

    assert "Macro calendar" in panels.macro_html(MACRO_EVENTS)
    assert panels.rates_html is not panels.macro_html


# ---------- FRED refused: the Treasury's own curve ----------
TREASURY_CSV = ('Date,"1 Mo","2 Yr","3 Yr","10 Yr","30 Yr"\n'
                "10/09/2026,4.10,4.80,4.85,5.20,5.50\n"
                "10/08/2026,4.11,4.78,4.84,5.18,5.49\n"
                "10/02/2026,4.12,4.70,4.80,5.10,5.40\n")


def _fred_down(treasury=TREASURY_CSV, dollar_ok=False):
    seen = []

    def handler(request: httpx.Request):
        seen.append(request.url.host)
        if "treasury" in request.url.host:
            return httpx.Response(200, text=treasury) if treasury else httpx.Response(503)
        if dollar_ok and request.url.params["id"] == "DTWEXBGS":
            return httpx.Response(200, text=load("fred_DTWEXBGS.csv"))
        return httpx.Response(403)
    return _run(macro.fetch_macro(httpx.AsyncClient(transport=httpx.MockTransport(handler)), RECORDED)), seen


def test_the_treasury_column_becomes_a_fred_series_oldest_first():
    text = macro.treasury_as_fred(TREASURY_CSV, "DGS10")
    assert macro.parse_series(text, "DGS10")[-1][1] == 5.20
    assert [v for _, v in macro.parse_series(macro.treasury_as_fred(TREASURY_CSV, "DGS2"), "DGS2")] == [4.70, 4.78, 4.80]


def test_a_refused_fred_falls_back_to_the_treasury_yields():
    snap, seen = _fred_down()
    assert snap.available and snap.source == "treasury.gov"
    assert snap.series["DGS10"]["available"] and snap.series["DGS2"]["available"]
    assert not snap.series["DTWEXBGS"]["available"], "the dollar has no Treasury twin and stays a named gap"
    assert snap.spread_2s10s == pytest.approx(0.40)
    assert "fred.stlouisfed.org" in seen and "home.treasury.gov" in seen


def test_fred_answering_partly_is_named_as_both():
    snap, _ = _fred_down(dollar_ok=True)
    assert snap.source == "fred+treasury.gov" and snap.series["DTWEXBGS"]["available"]


def test_fred_up_never_calls_the_treasury():
    def handler(request: httpx.Request):
        assert "treasury" not in request.url.host
        return httpx.Response(200, text=load(f"fred_{request.url.params['id']}.csv"))
    assert _run(macro.fetch_macro(httpx.AsyncClient(transport=httpx.MockTransport(handler)), RECORDED)).source == "fred"


def test_both_down_names_both():
    snap, _ = _fred_down(treasury=None)
    assert not snap.available and "treasury fallback" in snap.error and "403" in snap.error


def test_a_treasury_file_without_the_column_is_refused():
    with pytest.raises(ValueError, match="10 Yr"):
        macro.treasury_as_fred('Date,"1 Mo"\n10/09/2026,4.1\n', "DGS10")
