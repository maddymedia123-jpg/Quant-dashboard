"""Each desk reads derivatives over its own window (his v15, Section 2): no 24h block for every desk."""
from datetime import datetime, timezone
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from core.data import taker_flow
from core.indicators import desk_derivs as dd

H = 3_600_000
NOW = int(datetime(2026, 10, 22, 12, tzinfo=timezone.utc).timestamp() * 1000)


def frame(bar_ms: int, n: int, oi_step: float = 10.0, delta: float = 1.0, funding: float = 0.0001):
    """n rows ending at NOW; open interest rises by oi_step a row, so a change over k rows is k*oi_step."""
    ts = [NOW - (n - 1 - i) * bar_ms for i in range(n)]
    return pd.DataFrame({"timestamp": ts, "buy": [2.0] * n, "sell": [2.0 - delta] * n, "delta": [delta] * n,
                         "long_liq_usd": [100.0] * n, "short_liq_usd": [50.0] * n,
                         "oi_btc": [1000.0 + i * oi_step for i in range(n)], "funding_rate": [funding] * n})


def market(frames=None, expiries=(), price=80_000.0):
    flow = NS(available=frames is not None, frames=frames or {})
    return NS(taker_flow=flow, futures=NS(available=True, funding_rate=0.0001, oi_history=[(1, 5.0)]),
              options=NS(available=True, expiries=list(expiries)), spot=NS(last=price),
              generated_at=datetime.fromtimestamp(NOW / 1000, timezone.utc))


FRAMES = {"1h": frame(H, 300), "4h": frame(4 * H, 300), "1d": frame(24 * H, 300)}


def test_each_desk_counts_its_own_window_in_whole_periods():
    live = dd.read("live", market(FRAMES), NOW)
    assert [o["over"] for o in live["open_interest"]] == ["1h", "4h"]
    assert live["open_interest"][1]["change_btc"] == 40           # 4 hourly rows x 10
    assert live["taker_delta"] == {"over": "4h", "delta_btc": 4.0, "taker_buy_btc": 8.0, "taker_sell_btc": 4.0}
    assert live["liquidations"] == {"over": "4h", "longs_usd": 400, "shorts_usd": 200}
    assert "over" not in live["funding"], "the 4h desk reads the latest rate, not a mean"

    weekly = dd.read("weekly", market(FRAMES), NOW)
    assert weekly["open_interest"][0]["over"] == "7d"
    assert weekly["open_interest"][0]["change_btc"] == 420        # 42 four-hour rows x 10
    assert weekly["liquidations"]["longs_usd"] == 42 * 100
    assert weekly["funding"]["over"] == "7d"

    monthly = dd.read("monthly", market(FRAMES), NOW)
    assert monthly["open_interest"][0]["change_btc"] == 300       # 30 daily rows x 10
    assert monthly["taker_delta"]["delta_btc"] == 30.0


def test_open_interest_pace_compares_the_window_before():
    oi = dd.read("intraday", market(FRAMES), NOW)["open_interest"][1]
    now, then, before = 1000 + 299 * 10, 1000 + 275 * 10, 1000 + 251 * 10
    assert oi["change_pct"] == round((now / then - 1) * 100, 2)
    assert oi["prior_change_pct"] == round((then / before - 1) * 100, 2)


def test_too_little_history_is_a_named_gap_not_a_short_window():
    short = {"1h": frame(H, 10), "4h": frame(4 * H, 10), "1d": frame(24 * H, 10)}
    w = dd.read("monthly", market(short), NOW)
    assert "open_interest" not in w and "taker_delta" not in w
    assert {"open_interest", "taker_delta", "liquidations"} <= set(w["unavailable"])


def test_funding_mean_is_over_the_window_and_latest_is_the_newest_hour():
    frames = dict(FRAMES, **{"4h": frame(4 * H, 300, funding=0.0002), "1h": frame(H, 300, funding=0.0003)})
    f = dd.read("weekly", market(frames), NOW)["funding"]
    assert f["mean_8h_rate"] == pytest.approx(0.0002) and f["latest_8h_rate"] == 0.0003


def test_max_pain_is_the_biggest_expiry_inside_the_window():
    def exp(days, oi, pain):
        d = datetime.fromtimestamp((NOW + days * 24 * H) / 1000, timezone.utc).date().isoformat()
        return {"expiry": d, "max_pain": pain, "call_oi": oi, "put_oi": 0}
    chain = [exp(1, 500, 81_000), exp(2, 900, 82_000), exp(8, 9_000, 78_000), exp(64, 20_000, 75_000)]
    m = market(FRAMES, chain)
    assert dd.read("live", m, NOW)["max_pain"]["price"] == 81_000      # tomorrow 08:00 is 20h away
    assert dd.read("intraday", m, NOW)["max_pain"]["price"] == 82_000
    assert dd.read("weekly", m, NOW)["max_pain"]["price"] == 78_000
    assert dd.read("monthly", m, NOW)["max_pain"]["price"] == 75_000   # the quarterly
    only_far = market(FRAMES, [exp(64, 20_000, 75_000)])
    mp = dd.read("live", only_far, NOW)["max_pain"]
    assert mp["price"] == 75_000 and "after it" in mp["why"]


def test_long_oi_history_reaches_back_months_and_prefers_the_finest_rows():
    h = dd.oi_history(market(FRAMES))
    assert h[0][0] == FRAMES["1d"]["timestamp"].iloc[0]
    assert h[-1][0] == NOW and [t for t, _ in h] == sorted(t for t, _ in h)
    assert sum(1 for t, _ in h if t >= FRAMES["1h"]["timestamp"].iloc[0]) == 300
    assert dd.oi_history(market(None)) == [(1, 5.0)], "falls back to the futures snapshot"


def test_only_the_matching_desks_keep_the_24h_and_quarterly_figures():
    def payload():
        return {"futures": {"oi_change_24h_pct": 1.0, "funding_rate": 1e-4}, "liquidations": {"long_usd": 1},
                "options": {"max_pain": 75_000, "max_pain_expiry": "x", "iv_skew": 1}}
    for key in dd.WINDOWS:
        d = payload()
        dd.match_window(d, key, market(FRAMES), NOW)
        assert d["matched_window"]["window"] == dd.WINDOWS[key].label
        assert ("liquidations" in d) == (key == "intraday") == ("oi_change_24h_pct" in d["futures"])
        assert ("max_pain" in d["options"]) == (key == "monthly")
        assert d["options"]["iv_skew"] == 1 and d["futures"]["funding_rate"] == 1e-4


def test_weekly_and_monthly_flow_rows_carry_open_interest_as_a_level_not_a_sum():
    daily = frame(24 * H, 14)
    wk = taker_flow.weekly(daily)
    last_week = wk.iloc[-1]
    in_it = daily[daily["timestamp"] >= last_week["timestamp"]]
    assert last_week["oi_btc"] == in_it["oi_btc"].iloc[-1]
    assert last_week["delta"] == in_it["delta"].sum()


def test_period_rows_parse_open_interest_and_funding_and_leave_missing_ones_empty():
    rows = [{"time": 1, "long_taker_size": 10, "short_taker_size": 4, "open_interest": 5000, "last_funding_rate": "0.0001"},
            {"time": 2, "long_taker_size": 10, "short_taker_size": 4}]
    df = taker_flow.parse_stats(rows, 0.0001)
    assert df["oi_btc"].iloc[0] == 0.5 and df["funding_rate"].iloc[0] == 0.0001
    assert pd.isna(df["oi_btc"].iloc[1]) and pd.isna(df["funding_rate"].iloc[1])


def test_the_trap_desk_scales_its_open_interest_bar_to_the_window():
    """5% a day is the bar; over 4h it is 5% x sqrt(1/6) = 2.04%, over 7 days 13.2%."""
    from core.agents.trap_desk import desk_view
    from core.traps import BULL_TRAP, TrapCandidate

    cand = TrapCandidate(BULL_TRAP, "1h", 80_000.0, 81_000.0, 79_000.0, 3)

    def reasons(change, over):
        state = {"derivatives": {"matched_window": {"open_interest": [{"over": over, "change_pct": change}]}}}
        return " ".join(desk_view(state, cand).reasons)
    assert "+2.5% over 4h" in reasons(2.5, "4h")
    assert reasons(1.9, "4h") == ""
    assert reasons(9.0, "7d") == "", "9% in a week is not the crowd piling in on a weekly clock"
    assert "+14.0% over 7d" in reasons(14.0, "7d")
