"""The experimental sine card: causal, New York only, scored live from candles, and labelled on every line."""
import math
import pathlib

import numpy as np
import pandas as pd

from core.indicators import sine_wave as sw
from ui import panels

DAY = 86_400_000
T0 = 1_790_035_200_000 - 1_790_035_200_000 % DAY          # a midnight UTC


def candles(n=600, period=40, amp=400.0, start=T0):
    t = np.arange(n)
    mid = 80_000 + amp * np.sin(2 * math.pi * t / period)
    return pd.DataFrame({"timestamp": start + t * sw.BAR_MS, "open": mid, "high": mid + 30, "low": mid - 30,
                         "close": mid, "volume": 1.0})


def test_a_clean_cycle_gives_alternating_bottoms_and_tops():
    s = sw.crossings(candles())
    marks = s[s != 0]
    assert len(marks) >= 10
    assert all(a != b for a, b in zip(marks, marks[1:])), "bottom, top, bottom ..."


def test_no_value_moves_when_the_future_is_cut_off():
    df = candles(700)
    full = sw.crossings(df)
    for cut in (200, 431, 650):
        assert np.array_equal(sw.crossings(df.iloc[:cut].reset_index(drop=True)), full[:cut])


def test_only_new_york_hours_and_only_the_last_two_closed_candles_warn(monkeypatch):
    df = candles(800)
    ts = df["timestamp"].to_numpy()
    i = int(np.where((ts - T0) % DAY == 14 * 3_600_000)[0][-1])          # a 14:00 UTC candle, NY hours
    sig = np.zeros(len(df), int)
    sig[i] = 1
    monkeypatch.setattr(sw, "crossings", lambda d: sig[: len(d)])
    close = int(ts[i]) + sw.BAR_MS
    w, ny = sw.current(df, close + 60_000)                               # one minute after it closed
    assert w is not None and w.side == "bottom" and w.at_ms == close and ny
    assert sw.current(df, close + 2 * sw.BAR_MS)[0] is None, "30 minutes on, the warning has passed"
    assert sw.current(df, close - 60_000)[0] is None, "a candle still forming never signals"
    j = int(np.where((ts - T0) % DAY == 3 * 3_600_000)[0][-1])           # 03:00 UTC, outside New York
    sig[:] = 0
    sig[j] = -1
    assert sw.current(df, int(ts[j]) + sw.BAR_MS + 60_000)[0] is None


def test_the_live_record_scores_only_settled_new_york_signals_after_go_live(monkeypatch):
    df = candles(56 + 96 * 10 + 1)                    # the last candle opens at 14:00 UTC, inside New York
    ts = df["timestamp"].to_numpy()
    ny = np.where(sw.in_ny(ts))[0]
    a, b, late = ny[ny > 200][0], ny[ny > 400][0], len(df) - 3
    assert sw.in_ny([ts[late]])[0]
    sig = np.zeros(len(df), int)
    sig[[a, b, late]] = [1, -1, 1]
    monkeypatch.setattr(sw, "crossings", lambda d: sig[: len(d)])
    monkeypatch.setattr(sw, "big_turns", lambda d: [(a + 1, "bottom"), (b + 2, "bottom")])
    now = int(ts[-1]) + sw.BAR_MS
    r = sw.live_record(df, now, since_ms=int(ts[150]))
    assert (r.signals, r.hits, r.right_way) == (2, 2, 1), "the late signal has not settled; b named the wrong way"
    assert 0 < r.base < 1 and r.lift == r.hit_rate / r.base
    assert sw.live_record(df, now, since_ms=int(ts[300])).signals == 1, "nothing before go-live counts"


def test_the_card_says_experimental_and_never_reads_as_a_call():
    rec = sw.Record(sw.GO_LIVE_MS, 0, 0, 0, None)
    h = panels.sine_experimental_html(sw.Warning("top", T0 + 14 * 3_600_000, 81_000.0), True, rec)
    assert "EXPERIMENTAL, NOT PROVEN" in h and "do not trade on it alone" in h and "a turn DOWN" in h
    assert "no settled signals yet" in h
    off = panels.sine_experimental_html(None, False, sw.Record(sw.GO_LIVE_MS, 12, 3, 2, 0.04))
    assert "only reads New York hours" in off and "Too few to judge yet" in off and "6.2x" in off


def test_it_reaches_no_agent_plan_or_alert():
    root = pathlib.Path(__file__).resolve().parents[1]
    for f in ("core/agents/context.py", "core/agents/one_agent.py", "core/agents/live_recon.py",
              "core/plan_alerts.py", "core/market_alerts.py", "core/pulse.py", "core/layman_summary.py"):
        assert "sine_wave" not in (root / f).read_text(encoding="utf-8"), f
