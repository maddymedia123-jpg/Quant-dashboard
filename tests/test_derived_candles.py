"""2W and 1M candles: no free spot venue serves them, so they are built from weekly and daily candles."""
from datetime import datetime, timezone

import pandas as pd

from core.data.spot import (EPOCH_MONDAY_MS, HALF_WEEK_MS, calendar_month, half_week, three_day, twelve_hour,
                            two_week, with_derived_frames)
from core.trades import next_close, nth_close

DAY = 86_400_000


def _ms(*a):
    return int(datetime(*a, tzinfo=timezone.utc).timestamp() * 1000)


def _candles(start_ms, step_ms, n, open0=100.0):
    rows = []
    for i in range(n):
        o = open0 + i
        rows.append([start_ms + i * step_ms, o, o + 5, o - 5, o + 1, 10.0])
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def test_a_month_is_its_calendar_days_with_first_open_last_close_and_extremes():
    daily = _candles(_ms(2026, 1, 30), DAY, 5)        # Jan 30, Jan 31, Feb 1, Feb 2, Feb 3
    m = calendar_month(daily)
    assert list(m["timestamp"]) == [_ms(2026, 1, 1), _ms(2026, 2, 1)]
    jan, feb = m.iloc[0], m.iloc[1]
    assert (jan.open, jan.close, jan.high, jan.low, jan.volume) == (100, 102, 106, 95, 20)
    assert (feb.open, feb.close, feb.high, feb.low, feb.volume) == (102, 105, 109, 97, 30)


def test_two_weeks_pair_the_same_weeks_whatever_the_first_candle():
    weekly = _candles(EPOCH_MONDAY_MS + 2001 * 7 * DAY, 7 * DAY, 4)    # starts on an odd week
    pairs = two_week(weekly)
    shifted = two_week(weekly.iloc[1:].reset_index(drop=True))
    # weeks 2001..2004 pair as (2000,2001) (2002,2003) (2004,2005): the odd first week stands alone
    assert list(pairs["timestamp"]) == [EPOCH_MONDAY_MS + w * 7 * DAY for w in (2000, 2002, 2004)]
    assert pairs.iloc[1].open == 101 and pairs.iloc[1].close == 103, "weeks 2002 and 2003 together"
    assert all((t - EPOCH_MONDAY_MS) % (14 * DAY) == 0 for t in pairs["timestamp"])
    # dropping the first week must not re-pair the rest
    assert pairs["timestamp"].iloc[-1] == shifted["timestamp"].iloc[-1]
    assert pairs.iloc[-1].close == shifted.iloc[-1].close


def test_derived_frames_are_added_and_never_replace_a_fetched_one():
    daily, weekly = _candles(_ms(2026, 1, 1), DAY, 40), _candles(EPOCH_MONDAY_MS, 7 * DAY, 6)
    out = with_derived_frames({"1d": daily, "1w": weekly})
    assert set(out) == {"1d", "3d", "1w", "2w", "1M", "3M"}, "no hourly or 4h candles, so no 12h or half-week"
    hourly, four_hour = _candles(_ms(2026, 1, 1), 3_600_000, 48), _candles(EPOCH_MONDAY_MS, 4 * 3_600_000, 84)
    full = with_derived_frames({"1h": hourly, "4h": four_hour, "1d": daily, "1w": weekly})
    assert set(full) == {"1h", "4h", "12h", "1d", "3d", "0.5w", "1w", "2w", "1M", "3M"}
    own = _candles(0, DAY, 3)
    assert with_derived_frames({"1d": daily, "1M": own})["1M"] is own
    assert "1M" not in with_derived_frames({"1w": weekly}), "no daily candles, no months"


def test_a_monthly_checkpoint_is_the_first_of_next_month():
    assert next_close(_ms(2026, 10, 14, 9), "1M") == _ms(2026, 11, 1)
    assert nth_close(_ms(2026, 12, 20), "1M", 2) == _ms(2027, 2, 1), "across the year end"


def test_a_two_week_checkpoint_lands_on_a_two_week_candle_close():
    now = _ms(2026, 10, 14, 9)
    close = next_close(now, "2w")
    assert close > now and (close - EPOCH_MONDAY_MS) % (14 * DAY) == 0 and close - now <= 14 * DAY


def test_a_quarter_is_its_three_calendar_months():
    from core.data.spot import calendar_quarter
    daily = _candles(_ms(2026, 3, 30), DAY, 4)        # Mar 30, Mar 31, Apr 1, Apr 2
    q = calendar_quarter(daily)
    assert list(q["timestamp"]) == [_ms(2026, 1, 1), _ms(2026, 4, 1)]
    assert q.iloc[0].close == 102 and q.iloc[1].open == 102


# ---------- his v22 Section 3 derived candles: 12H, 3D, 0.5W ----------
def test_twelve_hour_candles_open_at_midnight_and_noon_utc():
    hourly = _candles(_ms(2026, 3, 2, 7), 3_600_000, 30)        # 07:00 Mon 2 Mar to 12:00 Tue 3 Mar
    h12 = twelve_hour(hourly)
    assert list(h12["timestamp"]) == [_ms(2026, 3, 2, 0), _ms(2026, 3, 2, 12), _ms(2026, 3, 3, 0), _ms(2026, 3, 3, 12)]
    first = h12.iloc[0]
    assert (first.open, first.close, first.volume) == (100, 105, 50), "07:00 to 11:00: five hourly candles"
    assert h12.iloc[1].open == 105 and h12.iloc[1].close == 117 and h12.iloc[1].volume == 120, "the full noon half-day"


def test_three_day_candles_count_from_the_epoch_monday_whatever_the_first_candle():
    daily = _candles(EPOCH_MONDAY_MS + 3001 * DAY, DAY, 7)      # starts one day into a 3-day run
    d3 = three_day(daily)
    assert list(d3["timestamp"]) == [EPOCH_MONDAY_MS + n * DAY for n in (3000, 3003, 3006)]
    assert d3.iloc[1].open == 102 and d3.iloc[1].close == 105 and d3.iloc[1].volume == 30, "days 3003-3005 together"
    shifted = three_day(daily.iloc[2:].reset_index(drop=True))
    assert shifted["timestamp"].iloc[0] == EPOCH_MONDAY_MS + 3003 * DAY, "dropping days does not regroup the rest"
    assert all((t - EPOCH_MONDAY_MS) % (3 * DAY) == 0 for t in d3["timestamp"])


def test_half_week_candles_split_the_week_at_thursday_noon_so_two_make_the_weekly():
    four_hour = _candles(EPOCH_MONDAY_MS + 2000 * 7 * DAY, 4 * 3_600_000, 42)   # exactly one week of 4h candles
    hw = half_week(four_hour)
    monday = EPOCH_MONDAY_MS + 2000 * 7 * DAY
    assert list(hw["timestamp"]) == [monday, monday + HALF_WEEK_MS]
    assert datetime.fromtimestamp((monday + HALF_WEEK_MS) / 1000, timezone.utc).strftime("%a %H:%M") == "Thu 12:00"
    a, b = hw.iloc[0], hw.iloc[1]
    assert (a.volume, b.volume) == (210, 210), "21 four-hour candles in each half"
    assert a.open == 100 and b.close == 142 and max(a.high, b.high) == 146 and min(a.low, b.low) == 95
    week = two_week  # noqa: F841 - the weekly candle is the two halves end to end
    assert a.close == 121 and b.open == 121, "the halves meet at Thursday noon"
