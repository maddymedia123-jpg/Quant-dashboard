"""2W and 1M candles: no free spot venue serves them, so they are built from weekly and daily candles."""
from datetime import datetime, timezone

import pandas as pd

from core.data.spot import EPOCH_MONDAY_MS, calendar_month, two_week, with_derived_frames
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
    assert set(out) == {"1d", "1w", "2w", "1M"}
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
