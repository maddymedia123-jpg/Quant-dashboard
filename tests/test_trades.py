"""The Active Trade desk's arithmetic: the stop, the timed targets, the pullback band, the schedule.

The spec's four deliverables are all measurements, so they are all testable. The fixtures here are built
to a coherent BTC tape: a 1h ATR of 400 on an 80,000 price is 0.5%, which is ordinary, and every level
sits where such a tape would put it.
"""
import numpy as np
import pandas as pd
import pytest

from core import trades
from core.trades import LONG, SHORT

ATR = 400.0
ENTRY = 80_000.0


def state(**over) -> dict:
    """What build_state hands over, with the keys the stop reads: pools, the value area, structure."""
    base = {
        "volatility": {"1h": {"available": True, "atr": ATR, "bands": {
            "mean": 80_000.0, "sigma1_up": 80_600.0, "sigma1_dn": 79_400.0,
            "sigma2_up": 81_200.0, "sigma2_dn": 78_800.0,
            "sigma3_up": 81_800.0, "sigma3_dn": 78_200.0}}},
        "liquidity": {"1h": {
            "pools": {"buyside": {"price": 80_900.0, "touches": 3},
                      "sellside": {"price": 79_150.0, "touches": 4}},
            "volume_profile": {"available": True, "point_of_control": 80_000.0,
                               "value_area_high": 80_600.0, "value_area_low": 79_400.0}}},
        "smc": {"1h": {"structure": {"available": True, "bias": "bullish", "swing_low": 79_050.0,
                                     "swing_high": 80_950.0,
                                     "last_event": {"type": "BOS", "level": 80_100.0}}}},
    }
    for path, value in over.items():
        node, keys = base, path.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return base


def candles(n=400, start=80_000.0, seed=7, drift=0.0, bar=200.0) -> pd.DataFrame:
    """A synthetic 1h tape with a realistic bar range, so pullbacks are measurable rather than invented."""
    rng = np.random.default_rng(seed)
    closes = start + np.cumsum(rng.normal(drift, bar / 2, n))
    highs = closes + rng.uniform(bar / 4, bar, n)
    lows = closes - rng.uniform(bar / 4, bar, n)
    ts = np.arange(n, dtype="int64") * 3_600_000 + 1_700_000_000_000
    return pd.DataFrame({"timestamp": ts, "open": closes, "high": highs, "low": lows,
                         "close": closes, "volume": rng.uniform(10, 100, n)})


# ---------- the stop ----------
def test_a_stop_beyond_the_structure_with_nothing_just_past_it_is_sound():
    v = trades.verify_stop(state(), LONG, ENTRY, 78_600.0, "1h", ATR)
    assert v.verdict == "sound" and v.sound and v.stop == 78_600.0
    assert "no resting liquidity within" in " ".join(v.reasons)


def test_a_stop_with_liquidity_just_beyond_it_would_be_hunted():
    """The sellside pool at 79,150 is 100 below a stop at 79,250: the wick that takes the pool takes the
    stop with it, which is the sweep that usually precedes the move going the trader's way."""
    v = trades.verify_stop(state(), LONG, ENTRY, 79_250.0, "1h", ATR)
    assert v.verdict == "would be hunted" and not v.sound
    assert "resting liquidity at 79,150" in " ".join(v.reasons)
    assert v.stop == v.proposed and v.stop < 79_150.0, "the corrected stop goes beyond that liquidity"


def test_liquidity_far_beyond_the_stop_is_not_a_stop_hunt():
    """A level well past the stop is simply further away. Treating any level beyond the stop as a hunt
    condemned every sane stop and pushed the correction five ATR from entry.

    The value area low is moved to 78,000 so that there really is a hazard 2.5 ATR beyond the stop - with
    it at 79,400 nothing sat below the stop at all and the bound was never tested."""
    deep = state(**{"liquidity.1h.volume_profile": {
        "available": True, "point_of_control": 80_000.0, "value_area_high": 80_600.0,
        "value_area_low": 78_000.0}})
    hazards = [lvl for lvl, _ in trades._hazards(deep, LONG, ENTRY, "1h")]
    assert 78_000.0 in hazards, "the fixture must put a level well beyond the stop"

    v = trades.verify_stop(deep, LONG, ENTRY, 79_000.0, "1h", ATR)
    assert v.verdict == "sound", v.reasons
    assert 79_000.0 - 78_000.0 > ATR * trades.HUNT_ATR, "and it is beyond the hunt window, not inside it"


def test_a_stop_inside_a_single_bar_range_is_rejected():
    v = trades.verify_stop(state(), LONG, ENTRY, ENTRY - ATR * 0.4, "1h", ATR)
    assert v.verdict == "inside the noise"
    assert "ordinary noise would take it out" in " ".join(v.reasons)


def test_a_stop_on_the_wrong_side_of_entry_is_named_as_such():
    assert trades.verify_stop(state(), LONG, ENTRY, ENTRY + 100, "1h", ATR).verdict == "on the wrong side"
    assert trades.verify_stop(state(), SHORT, ENTRY, ENTRY - 100, "1h", ATR).verdict == "on the wrong side"


def test_no_stop_given_means_the_engine_supplies_one_from_structure():
    v = trades.verify_stop(state(), LONG, ENTRY, None, "1h", ATR)
    assert v.verdict == "supplied" and v.given is None
    assert v.stop == v.proposed
    assert v.stop < 79_050.0, "beyond the swing low, not inside it"
    assert "No stop was given" in " ".join(v.reasons)


def test_the_proposed_stop_clears_the_whole_cluster_of_levels():
    """The swing at 79,050 and the pool at 79,150 are one cluster within an ATR: a stop between them
    would be taken by the single move that sweeps both."""
    v = trades.verify_stop(state(), LONG, ENTRY, None, "1h", ATR)
    assert v.stop < 79_050.0 - 1, "below the deeper of the two, plus a buffer"
    assert v.stop == pytest.approx(79_050.0 - ATR * trades.STOP_BUFFER_ATR)
    assert "cluster" in v.protects_kind


def test_the_proposed_stop_stays_a_tradeable_distance_from_entry():
    """Sitting beyond every level in range put the stop five ATR out, which leaves no trade worth
    taking. The correction has to be beyond the structure but still a risk a trader would accept."""
    v = trades.verify_stop(state(), LONG, ENTRY, None, "1h", ATR)
    risk = ENTRY - v.stop
    assert ATR * 0.75 <= risk <= ATR * 3, f"{risk / ATR:.1f} ATR of risk"


def test_the_short_mirror_reads_the_levels_above_entry():
    """The buyside pool is the outermost level here, so a mirror that read the sellside pool instead would
    place the stop 350 lower - with the pool at 80,900 it was redundant with the swing and the error was
    invisible."""
    s = state(**{"liquidity.1h.pools": {"buyside": {"price": 81_300.0, "touches": 3},
                                        "sellside": {"price": 79_150.0, "touches": 4}}})
    hazards = trades._hazards(s, SHORT, ENTRY, "1h")
    assert [lvl for lvl, _ in hazards][-1] == 81_300.0, "the pool is the outermost hazard above entry"

    v = trades.verify_stop(s, SHORT, ENTRY, None, "1h", ATR)
    assert v.stop == pytest.approx(81_300.0 + ATR * trades.STOP_BUFFER_ATR)
    assert v.stop > 81_300.0, "beyond the buyside pool, not merely beyond the swing high"
    assert trades.verify_stop(s, SHORT, ENTRY, 81_200.0, "1h", ATR).verdict == "would be hunted"


def test_a_stop_works_with_no_structure_at_all_and_says_so():
    bare = {"volatility": {"1h": {"atr": ATR}}, "liquidity": {"1h": {}}, "smc": {"1h": {}}}
    v = trades.verify_stop(bare, LONG, ENTRY, None, "1h", ATR)
    assert v.stop == pytest.approx(ENTRY - 2 * ATR)
    assert "default" in v.protects_kind, "and it does not pretend structure produced it"


# ---------- the targets ----------
class Zone:
    def __init__(self, ms):
        self.timestamp_ms, self.is_future = ms, True


class Fib:
    def __init__(self, *ms):
        self.upcoming = [Zone(m) for m in ms]


NOW = 1_700_000_000_000


def test_targets_come_from_the_volatility_bands_and_are_dated_by_the_fib_windows():
    fib = Fib(NOW + 3_600_000, NOW + 7_200_000, NOW + 10_800_000)
    ts = trades.timed_targets(state(), LONG, ENTRY, 79_000.0, "1h", ATR, fib, NOW)
    assert [t.name for t in ts] == ["TP1", "TP2", "TP3"]
    assert [t.price for t in ts] == [80_600.0, 81_200.0, 81_800.0], "the 1, 2 and 3 sigma bands"
    assert all(t.dated_by_fib for t in ts)
    assert [t.due_ms for t in ts] == [NOW + 3_600_000, NOW + 7_200_000, NOW + 10_800_000]
    assert "sigma band" in ts[0].basis


def test_the_short_targets_use_the_bands_below_entry():
    ts = trades.timed_targets(state(), SHORT, ENTRY, 81_000.0, "1h", ATR, None, NOW)
    assert [t.price for t in ts] == [79_400.0, 78_800.0, 78_200.0]


def test_reward_is_measured_against_the_actual_risk():
    ts = trades.timed_targets(state(), LONG, ENTRY, 79_600.0, "1h", ATR, None, NOW)   # 400 of risk
    assert [t.reward_r for t in ts] == [1.5, 3.0, 4.5]


def test_a_band_behind_the_entry_falls_back_to_a_range_multiple_and_says_so():
    """Entry above the 3-sigma band: the bands cannot produce a target, so the fallback is explicit
    rather than silently handing back a target behind the entry."""
    ts = trades.timed_targets(state(), LONG, 82_000.0, 81_000.0, "1h", ATR, None, NOW)
    assert all(t.price > 82_000.0 for t in ts), "a target behind the entry is not a target"
    assert all("sigma band is behind the entry" in t.basis for t in ts)
    assert [t.price for t in ts] == [82_400.0, 82_800.0, 83_200.0]


def test_every_target_is_dated_even_without_fibonacci_windows():
    """An undated target cannot be monitored, so it falls back to the candle a trader is watching."""
    ts = trades.timed_targets(state(), LONG, ENTRY, 79_000.0, "4h", ATR, None, NOW)
    assert all(t.due_ms and t.due_ms > NOW for t in ts)
    assert [t.due_from for t in ts] == ["the next 4h close", "the second 4h close", "the third 4h close"]
    assert ts[0].due_ms < ts[1].due_ms < ts[2].due_ms


def test_a_fibonacci_window_already_past_is_not_used_as_a_due_date():
    """On a stale frame every projected zone is behind us; a date in the past is worse than none."""
    ts = trades.timed_targets(state(), LONG, ENTRY, 79_000.0, "1h", ATR,
                              Fib(NOW - 7_200_000, NOW - 3_600_000, NOW + 3_600_000), NOW)
    assert ts[0].due_ms == NOW + 3_600_000 and ts[0].dated_by_fib, "the one window still ahead"
    assert all(t.due_ms > NOW for t in ts)
    assert not ts[1].dated_by_fib and "close" in ts[1].due_from


def test_targets_survive_a_missing_volatility_block():
    ts = trades.timed_targets({"volatility": {}}, LONG, ENTRY, 79_000.0, "1h", ATR, None, NOW)
    assert [t.price for t in ts] == [80_400.0, 80_800.0, 81_200.0]


# ---------- the pullback band ----------
def test_the_pullback_band_is_measured_from_the_candles():
    pb = trades.pullback_band(candles(), LONG, horizon_bars=12, timeframe="1h")
    assert pb.available
    assert 0 < pb.ordinary < pb.edge < pb.breaking, "the percentiles have to be ordered"
    assert pb.breaking < 10_000, "and a sane size for a 200-point bar range"
    levels = pb.levels(LONG, ENTRY)
    assert levels["ordinary"] > levels["edge"] > levels["breaking"], "a long's pullbacks are downward"


def test_the_short_pullback_band_points_the_other_way():
    pb = trades.pullback_band(candles(), SHORT, horizon_bars=12, timeframe="1h")
    levels = pb.levels(SHORT, ENTRY)
    assert levels["ordinary"] < levels["edge"] < levels["breaking"]
    assert all(v > ENTRY for v in levels.values())


def test_a_trending_tape_gives_a_long_a_shallower_pullback_than_a_short():
    """The measurement has to respond to the tape: in an uptrend a long is held through less adverse
    excursion than a short, and a band that could not tell them apart would be measuring nothing."""
    up = candles(seed=3, drift=60.0)
    assert trades.pullback_band(up, LONG, timeframe="1h").edge < \
           trades.pullback_band(up, SHORT, timeframe="1h").edge


def test_too_few_candles_says_unavailable_rather_than_guessing():
    pb = trades.pullback_band(candles(n=20), LONG, horizon_bars=12, timeframe="1h")
    assert not pb.available and pb.note and pb.ordinary == 0.0
    assert not trades.pullback_band(None, LONG, timeframe="1h").available


# ---------- the schedule ----------
def test_the_next_close_is_aligned_to_the_candle_not_to_now():
    at = 1_700_000_000_000                      # 2026-11-14 22:13:20 UTC
    assert trades.next_close(at, "1h") % 3_600_000 == 0
    assert trades.next_close(at, "4h") % (4 * 3_600_000) == 0
    assert trades.next_close(at, "1d") % 86_400_000 == 0
    assert trades.next_close(at, "15m") % 900_000 == 0
    for tf in ("15m", "1h", "4h", "1d", "1w"):
        assert trades.next_close(at, tf) > at


def test_the_weekly_close_lands_on_a_monday():
    from datetime import datetime, timezone
    close = trades.next_close(1_700_000_000_000, "1w")
    d = datetime.fromtimestamp(close / 1000, timezone.utc)
    assert (d.weekday(), d.hour, d.minute) == (0, 0, 0), f"{d} is not a Monday open"


def test_an_unknown_timeframe_raises_rather_than_inventing_a_close():
    with pytest.raises(ValueError):
        trades.next_close(NOW, "3d")


def test_the_nth_close_steps_one_candle_at_a_time():
    first = trades.nth_close(NOW, "4h", 1)
    assert first == trades.next_close(NOW, "4h")
    assert trades.nth_close(NOW, "4h", 3) == first + 2 * 4 * 3_600_000


def test_the_schedule_lists_the_candle_closes_soonest_first():
    # deliberately out of time order: passing them ascending made an unsorted schedule look sorted
    sched = trades.monitoring_schedule(NOW, ("4h", "15m", "1h"))
    assert [c.timeframe for c in sched] == ["15m", "1h", "4h"], "sorted by time, not by argument order"
    assert [c.at_ms for c in sched] == sorted(c.at_ms for c in sched)
    assert all("close" in c.reason for c in sched)
    assert "UTC" in sched[0].when


def test_the_schedule_names_what_dated_each_target():
    """It used to call every target window a Fibonacci one, including the ones dated by candle closes."""
    fib = Fib(NOW + 30 * 60_000)
    ts = trades.timed_targets(state(), LONG, ENTRY, 79_000.0, "1h", ATR, fib, NOW)
    sched = trades.monitoring_schedule(NOW, ("4h",), ts, limit=8)
    reasons = " | ".join(c.reason for c in sched)
    assert "TP1 is due (Fibonacci time window)" in reasons
    assert "Fibonacci" not in reasons.split("TP1 is due (Fibonacci time window)")[1], \
        "the candle-dated targets must not claim a Fibonacci window"


def test_the_schedule_does_not_repeat_a_time_the_candle_closes_already_cover():
    ts = trades.timed_targets(state(), LONG, ENTRY, 79_000.0, "4h", ATR, None, NOW)
    sched = trades.monitoring_schedule(NOW, ("4h",), ts, limit=8)
    assert len({c.at_ms for c in sched}) == len(sched), "no two checkpoints at the same moment"


# ---------- live progress ----------
def test_progress_reports_pnl_in_r_and_which_targets_are_hit():
    ts = trades.timed_targets(state(), LONG, ENTRY, 79_600.0, "1h", ATR, None, NOW)
    p = trades.progress(LONG, ENTRY, 79_600.0, ts, 80_700.0)
    assert p.pnl_r == 1.75 and p.pnl_pct == pytest.approx(0.875, abs=1e-3)
    assert p.hit == ("TP1",) and not p.stopped
    assert p.distances["TP2"] > 0


def test_progress_knows_when_the_stop_is_gone():
    ts = trades.timed_targets(state(), LONG, ENTRY, 79_600.0, "1h", ATR, None, NOW)
    assert trades.progress(LONG, ENTRY, 79_600.0, ts, 79_500.0).stopped
    assert trades.progress(LONG, ENTRY, 79_600.0, ts, 79_600.0).stopped, "touching the stop counts"
    assert not trades.progress(LONG, ENTRY, 79_600.0, ts, 79_700.0).stopped


def test_a_short_in_profit_is_not_reported_as_a_loss():
    ts = trades.timed_targets(state(), SHORT, ENTRY, 80_400.0, "1h", ATR, None, NOW)
    p = trades.progress(SHORT, ENTRY, 80_400.0, ts, 79_300.0)
    assert p.pnl_r > 0 and p.pnl_pct > 0
    assert "TP1" in p.hit and not p.stopped
    assert trades.progress(SHORT, ENTRY, 80_400.0, ts, 80_500.0).stopped


def test_the_time_format_is_always_utc():
    assert "UTC" in trades.fmt_when(NOW)
    assert trades.fmt_when(None) == "no date"
