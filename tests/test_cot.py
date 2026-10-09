"""CFTC Commitments of Traders for CME Bitcoin futures: the provider and the panel.

Every trap here was found by calling the live API: a name match that also returns Bitcoin Cash, a reused
contract code, misspelled columns, and a contract that is five bitcoin, not one."""
import asyncio
import json
import pathlib
from datetime import datetime, timezone

import httpx
import pytest

from core.data import cot
from core.data.cot import (CONTRACT_CODE, CROWDED_LONG, CROWDED_SHORT, MIN_WEEKS_FOR_INDEX,
                           contract_size, cot_index, fetch_cot, parse_cot)
from core.data.types import COTSnapshot
from ui.panels import cot_html

FX = pathlib.Path(__file__).parent / "fixtures"
RECORDED = int(datetime(2026, 10, 10, tzinfo=timezone.utc).timestamp() * 1000)
DAY = 86_400_000


def _rows():
    return json.loads((FX / "cftc_cot_btc_cme.json").read_text(encoding="utf-8"))


def _row(date="2026-10-06T00:00:00.000", long=100, short=40, **over):
    r = {"report_date_as_yyyy_mm_dd": date, "cftc_contract_market_code": CONTRACT_CODE,
         "market_and_exchange_names": "BITCOIN - CHICAGO MERCANTILE EXCHANGE",
         "contract_units": "(CONTRACTS OF 5 BITCOINS)", "open_interest_all": "1000",
         "noncomm_positions_long_all": str(long), "noncomm_positions_short_all": str(short),
         "comm_positions_long_all": "10", "comm_positions_short_all": "30",
         "nonrept_positions_long_all": "5", "nonrept_positions_short_all": "8"}
    r.update(over)
    return r


def _chip(out):
    """The read on the chip itself: the footer legend also says "crowded long" and "crowded short"."""
    return out.split("Large speculators: <span class='ti-chip ", 1)[1].split("</span>", 1)[0]


def _weeks(nets):
    """One row per week, newest first, with the given speculator nets."""
    start = datetime(2026, 10, 6, tzinfo=timezone.utc).timestamp()
    out = []
    for i, n in enumerate(nets):
        d = datetime.fromtimestamp(start - i * 7 * 86400, timezone.utc).strftime("%Y-%m-%dT00:00:00.000")
        out.append(_row(d, long=1000 + n, short=1000))
    return out


# ---------------------------------------------------------------- the recorded report

def test_the_recorded_report_parses_to_the_cme_contract():
    s = parse_cot(_rows(), RECORDED)
    assert s.available and s.source == "cftc"
    assert s.market == "BITCOIN - CHICAGO MERCANTILE EXCHANGE"
    assert s.as_of_ms == int(datetime(2026, 10, 6, tzinfo=timezone.utc).timestamp() * 1000)
    assert s.spec_net == 2540 and s.spec_net_change == 75
    assert s.commercial_net == -3006 and s.small_trader_net == 466
    assert s.open_interest == 21432 and s.contract_btc == 5.0
    assert s.weeks_in_index == 52 and s.cot_index == pytest.approx(70.6)
    assert s.age_days == 4.0 and not s.stale


def test_the_three_groups_net_to_zero_as_every_long_has_a_short():
    s = parse_cot(_rows(), RECORDED)
    assert s.spec_net + s.commercial_net + s.small_trader_net == 0


# ---------------------------------------------------------------- selecting the market

def test_bitcoin_cash_is_not_bitcoin():
    rows = [_row(cftc_contract_market_code="133LM1",
                 market_and_exchange_names="BITCOIN CASH PERP STYLE - COINBASE DERIVATIVES", long=9999)]
    s = parse_cot(rows + [_row()], RECORDED)
    assert s.spec_net == 60


def test_another_contract_on_the_same_exchange_is_not_mixed_in():
    # the exchange check alone would admit this newer row: only the exact code keeps it out
    other = _row(date="2026-10-13T00:00:00.000", cftc_contract_market_code="133742",
                 market_and_exchange_names="MICRO BITCOIN - CHICAGO MERCANTILE EXCHANGE", long=9999)
    s = parse_cot([other, _row()], RECORDED)
    assert s.spec_net == 60 and s.market == "BITCOIN - CHICAGO MERCANTILE EXCHANGE"


def test_a_reused_code_on_another_exchange_is_refused():
    s = parse_cot([_row(market_and_exchange_names="BITCOIN - CBOE FUTURES EXCHANGE")], RECORDED)
    assert not s.available and CONTRACT_CODE in s.error


def test_the_latest_week_is_read_whatever_order_the_rows_arrive_in():
    rows = _weeks([50, 10, 20])
    s = parse_cot(list(reversed(rows)), RECORDED)
    assert s.spec_net == 50 and s.as_of_ms == int(datetime(2026, 10, 6, tzinfo=timezone.utc).timestamp() * 1000)


@pytest.mark.parametrize("payload", [{"error": "x"}, "nope", None, []])
def test_a_payload_that_is_not_rows_is_unavailable(payload):
    assert not parse_cot(payload, RECORDED).available


# ---------------------------------------------------------------- missing and misspelled columns

@pytest.mark.parametrize("field", ["noncomm_positions_long_all", "open_interest_all",
                                   "comm_positions_short_all"])
def test_a_missing_column_is_named_not_read_as_zero(field):
    r = _row()
    del r[field]
    s = parse_cot([r], RECORDED)
    assert not s.available and field in s.error


def test_a_misspelled_column_is_not_accepted_in_place_of_the_real_one():
    r = _row()
    r["noncomm_postions_long_all"] = r.pop("noncomm_positions_long_all")
    s = parse_cot([r], RECORDED)
    assert not s.available and "noncomm_positions_long_all" in s.error


def test_an_unreadable_date_is_named():
    s = parse_cot([_row(date="not a date")], RECORDED)
    assert not s.available and "report_date_as_yyyy_mm_dd" in s.error


def test_without_the_change_columns_the_change_comes_from_the_prior_week():
    s = parse_cot(_weeks([50, 20]), RECORDED)
    assert s.spec_net_change == 30


def test_one_week_with_no_change_columns_reports_no_change():
    assert parse_cot([_row()], RECORDED).spec_net_change is None


# ---------------------------------------------------------------- contract size

@pytest.mark.parametrize("units, size", [("(CONTRACTS OF 5 BITCOINS)", 5.0), ("(5 Bitcoins)", 5.0),
                                         ("(0.1 bitcoin)", 0.1), ("(CONTRACTS OF $25)", None),
                                         ("", None), (None, None)])
def test_the_contract_size_comes_from_the_report_itself(units, size):
    assert contract_size(units) == size


def test_no_bitcoin_figure_is_shown_when_the_size_is_unknown():
    s = parse_cot([_row(contract_units="(CONTRACTS)")], RECORDED)
    assert s.contract_btc is None
    assert "BTC)" not in cot_html(s, RECORDED)


def test_contracts_are_converted_at_five_bitcoin_each():
    assert "+60 contracts (~+300 BTC)" in cot_html(parse_cot([_row()], RECORDED), RECORDED)


# ---------------------------------------------------------------- the COT index

def test_the_index_is_where_the_net_sits_in_its_range():
    assert cot_index(50, [50] + [0] * 24 + [100]) == 50.0
    assert cot_index(100, [100] + [0] * 25) == 100.0
    assert cot_index(0, [0] + [100] * 25) == 0.0


def test_the_index_needs_half_a_year_of_history():
    assert cot_index(5, list(range(MIN_WEEKS_FOR_INDEX - 1))) is None
    assert cot_index(5, list(range(MIN_WEEKS_FOR_INDEX))) is not None


def test_a_flat_range_has_no_index():
    assert cot_index(7, [7] * 30) is None


def test_only_the_last_52_weeks_count():
    # a giant net from 60 weeks ago must not compress today's reading
    assert cot_index(100, [100] + [0] * 51 + [10_000] * 10) == 100.0


def test_too_short_a_history_says_so_on_the_panel():
    s = parse_cot(_weeks(range(10)), RECORDED)
    assert s.cot_index is None and s.weeks_in_index == 10
    assert "Not enough history" in cot_html(s, RECORDED)


@pytest.mark.parametrize("top, read", [(True, "crowded long"), (False, "crowded short")])
def test_an_extreme_is_called_crowded(top, read):
    nets = ([100] + [0] * 30) if top else ([0] + [100] * 30)
    out = cot_html(parse_cot(_weeks(nets), RECORDED), RECORDED)
    assert read in _chip(out)


def test_the_thresholds_are_the_conventional_80_and_20():
    assert (CROWDED_LONG, CROWDED_SHORT) == (80, 20)
    edge = parse_cot(_weeks([80] + [0] * 29 + [100]), RECORDED)
    assert edge.cot_index == 80.0 and _chip(cot_html(edge, RECORDED)).endswith("crowded long")
    mid = parse_cot(_weeks([79] + [0] * 29 + [100]), RECORDED)
    assert _chip(cot_html(mid, RECORDED)).endswith("neither extreme")
    low = parse_cot(_weeks([20] + [0] * 29 + [100]), RECORDED)
    assert low.cot_index == 20.0 and _chip(cot_html(low, RECORDED)).endswith("crowded short")


# ---------------------------------------------------------------- staleness

def test_a_report_ten_days_old_is_normal_and_eleven_is_stale():
    as_of = int(datetime(2026, 10, 6, tzinfo=timezone.utc).timestamp() * 1000)
    assert not parse_cot([_row()], as_of + 10 * DAY).stale
    late = parse_cot([_row()], as_of + 11 * DAY)
    assert late.stale and "stale" in cot_html(late, as_of + 11 * DAY)


def test_a_fresh_report_is_not_flagged():
    assert "stale" not in cot_html(parse_cot(_rows(), RECORDED), RECORDED)


# ---------------------------------------------------------------- the panel

def test_the_panel_leads_with_the_as_of_date_and_no_clock_time():
    out = cot_html(parse_cot(_rows(), RECORDED), RECORDED)
    assert "Positions as of Tue 06 Oct" in out
    assert "PKT" not in out and "UTC" not in out


def test_a_holiday_week_shows_its_real_weekday():
    out = cot_html(parse_cot([_row(date="2026-12-28T00:00:00.000")], RECORDED), RECORDED)
    assert "Positions as of Mon 28 Dec" in out


def test_no_report_says_why():
    out = cot_html(COTSnapshot.unavailable("cftc", "ConnectError: <down>"), RECORDED)
    assert "No positioning report" in out and "&lt;down&gt;" in out and "<down>" not in out


def test_the_market_name_is_escaped():
    s = parse_cot([_row(market_and_exchange_names="BITCOIN <b>X</b> - CHICAGO MERCANTILE EXCHANGE")],
                  RECORDED)
    out = cot_html(s, RECORDED)
    assert "&lt;b&gt;" in out and "<b>X" not in out


def test_the_commercials_caveat_names_the_trader_counts():
    out = cot_html(parse_cot(_rows(), RECORDED), RECORDED)
    assert "only 2 long and 6 short traders" in out


# ---------------------------------------------------------------- fetching

class _Transport(httpx.AsyncBaseTransport):
    def __init__(self, body, status=200):
        self.body, self.status, self.request = body, status, None

    async def handle_async_request(self, request):
        self.request = request
        return httpx.Response(self.status, content=self.body, request=request)


def _fetch(transport):
    async def go():
        async with httpx.AsyncClient(transport=transport) as c:
            return await fetch_cot(c, RECORDED)
    return asyncio.run(go())


def test_the_request_asks_for_the_cme_contract_and_a_year_of_weeks():
    t = _Transport(json.dumps(_rows()).encode())
    s = _fetch(t)
    assert s.available
    q = t.request.url.params
    assert q["cftc_contract_market_code"] == CONTRACT_CODE and q["$limit"] == "53"
    assert "report_date_as_yyyy_mm_dd DESC" == q["$order"]
    assert "noncomm_positions_long_all" in q["$select"]


def test_an_http_error_is_unavailable_and_named():
    s = _fetch(_Transport(b"{}", status=503))
    assert not s.available and s.error.startswith("HTTPStatusError")


def test_an_oversized_response_is_refused():
    s = _fetch(_Transport(b"[" + b" " * (cot.MAX_BYTES + 1) + b"]"))
    assert not s.available and "KB" in s.error


# ---------------------------------------------------------------- placement

def test_positioning_shows_on_the_weekly_and_monthly_tabs_only(tmp_path, monkeypatch):
    monkeypatch.setenv("TI_OFFLINE_FIXTURES", "1")
    monkeypatch.setenv("TI_DATA_DIR", str(tmp_path))
    app_py = str(pathlib.Path(__file__).resolve().parent.parent / "app.py")
    at = pytest.importorskip("streamlit.testing.v1").AppTest.from_file(app_py, default_timeout=180).run()
    shown = {}
    for screen in ("live", "intraday", "weekly", "monthly", "trap", "trade", "director", "macro"):
        at = at.radio(key="screen").set_value(screen).run()
        assert not at.exception, (screen, [e.value for e in at.exception])
        assert not any("Positioning report unavailable" in w.value for w in at.warning)
        shown[screen] = "CME Bitcoin positioning (CFTC)" in " ".join(m.value for m in at.markdown)
    assert shown == {"live": False, "intraday": False, "weekly": True, "monthly": True,
                     "trap": False, "trade": False, "director": False, "macro": False}
