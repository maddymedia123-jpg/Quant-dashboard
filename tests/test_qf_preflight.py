"""Quant Forecast pre-flight: the Trap Head override and the macro chop window, and how build_plan
honours them. Offline: fake store and snapshot objects, no network, nothing dispatched."""
import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("TI_OFFLINE_FIXTURES", "1")

from core import qf_preflight as pf  # noqa: E402
from core import quant_forecast as qf  # noqa: E402
from core.data.types import CalendarSnapshot  # noqa: E402
from core.store import Store  # noqa: E402

H = qf.HOUR_MS
DAY0 = 1_788_825_600_000            # a 00:00 UTC
END0 = DAY0 + qf.DAY_MS


def hourly(n: int, start_ms: int, sigma: float = 0.004, seed: int = 1, drift: float = 0.0,
           price: float = 80_000.0) -> pd.DataFrame:
    """The same synthetic hourly candles tests/test_quant_forecast.py builds plans from."""
    rng = np.random.default_rng(seed)
    t = start_ms + np.arange(n) * H
    close = price * np.exp(np.cumsum(rng.standard_normal(n) * sigma + drift))
    open_ = np.concatenate([[price], close[:-1]])
    wig = np.abs(rng.standard_normal(n)) * sigma * close * 0.5
    return pd.DataFrame({"timestamp": t, "open": open_, "high": np.maximum(open_, close) + wig,
                         "low": np.minimum(open_, close) - wig, "close": close,
                         "volume": rng.uniform(50, 150, n)})


class FakeStore:
    def __init__(self, rows=None, boom=False, loose=False):
        """`loose`: a store that ignores the filters it is asked for, so the check's own filtering shows."""
        self.rows, self.boom, self.loose, self.calls = list(rows or []), boom, loose, []

    def traps(self, category=None, status=None, limit=100, timeframes=None):
        self.calls.append({"status": status, "timeframes": tuple(timeframes or ())})
        if self.boom:
            raise RuntimeError("database is locked")
        if self.loose:
            return list(self.rows)
        return [r for r in self.rows if (status is None or r["status"] == status)
                and (not timeframes or r["timeframe"] in timeframes)]


class FakeMarket:
    def __init__(self, calendar):
        self.calendar = calendar


def trap(**kw):
    base = {"id": 1, "category": "intraday", "side": "bull_trap", "timeframe": "4h", "level": 81_000.0,
            "invalidation": 81_400.0, "plays_out": 80_200.0, "declared_ms": DAY0 - 3 * H, "score": 72.0,
            "head_call": "ENGINEERED_TRAP", "status": "active"}
    return {**base, **kw}


def event(title="CPI m/m", t_ms=DAY0 + 13 * H + 30 * 60_000, impact="High", country="USD"):
    return {"title": title, "country": country, "impact": impact, "time_ms": t_ms, "date": "",
            "forecast": None, "previous": None}


def cal(*events):
    return CalendarSnapshot(source="forexfactory", events=list(events))


def _plan(df, now=DAY0, **kw):
    return qf.build_plan(df, now, END0, float(df["close"].iloc[-1]), **kw)


# ---------- the checks ----------

def test_clear_path_when_no_trap_and_no_release():
    pre = pf.preflight(FakeStore(), FakeMarket(cal(event(t_ms=END0 + 5 * H))), DAY0, END0)
    assert pre == {"trap": {"active": False, "label": "Trap Agent: Clear", "traps": []}, "macro": [],
                   "forced_neutral": False, "errors": []}
    assert pf.plan_kwargs(pre) == {"force_neutral": False, "forced_windows": []}


def test_an_active_engineered_4h_trap_forces_neutral():
    store = FakeStore([trap()])
    pre = pf.preflight(store, FakeMarket(cal()), DAY0, END0)
    assert pre["forced_neutral"] is True and pre["trap"]["active"] is True
    assert pre["trap"]["label"] == "Trap Agent: BULL TRAP active (4H)"
    assert pre["trap"]["traps"][0]["type"] == "bull_trap" and pre["trap"]["traps"][0]["since_ms"] == DAY0 - 3 * H
    assert store.calls == [{"status": "active", "timeframes": ("4h",)}], "asked the ledger for live 4h traps only"
    assert pf.plan_kwargs(pre)["force_neutral"] is True
    # a legacy row with no head_call recorded was still a declared trap
    assert pf.preflight(FakeStore([trap(head_call=None)]), FakeMarket(cal()), DAY0, END0)["forced_neutral"]
    both = pf.preflight(FakeStore([trap(), trap(id=2, side="bear_trap")]), FakeMarket(cal()), DAY0, END0)
    assert both["trap"]["label"] == "Trap Agent: BEAR TRAP + BULL TRAP active (4H)"


def test_a_1h_trap_a_settled_trap_or_an_authentic_break_does_not_force_neutral():
    for row in (trap(timeframe="1h"), trap(status="played_out"), trap(status="invalidated"),
                trap(head_call="AUTHENTIC_BREAK")):
        for loose in (False, True):           # whether or not the store applied the filters itself
            pre = pf.preflight(FakeStore([row], loose=loose), FakeMarket(cal()), DAY0, END0)
            assert pre["forced_neutral"] is False and pre["trap"]["label"] == "Trap Agent: Clear", (row, loose)


def test_a_high_impact_usd_release_inside_the_day_is_one_chop_window():
    t = DAY0 + 13 * H + 30 * 60_000
    pre = pf.preflight(FakeStore(), FakeMarket(cal(event(t_ms=t), event("Core CPI m/m", t_ms=t))), DAY0, END0)
    assert pre["macro"] == [{"title": "CPI m/m + Core CPI m/m", "t_ms": t, "start_ms": t - 30 * 60_000,
                             "end_ms": t + 90 * 60_000, "impact": "High"}]
    assert pf.plan_kwargs(pre)["forced_windows"] == [(t - 30 * 60_000, t + 90 * 60_000, "CPI m/m + Core CPI m/m")]
    # clamped to the day
    late = pf.preflight(FakeStore(), FakeMarket(cal(event(t_ms=END0 - 10 * 60_000))), DAY0, END0)
    assert late["macro"][0]["end_ms"] == END0


def test_low_impact_foreign_or_out_of_window_releases_make_no_window():
    for e in (event(impact="Medium"), event(country="EUR"), event(t_ms=DAY0 - 60_000),
              event(t_ms=END0 + 60_000), event(t_ms=None)):
        pre = pf.preflight(FakeStore(), FakeMarket(cal(e)), DAY0, END0)
        assert pre["macro"] == [] and pre["errors"] == [], e


def test_a_broken_ledger_or_calendar_leaves_the_check_clear_with_a_note():
    pre = pf.preflight(FakeStore(boom=True), FakeMarket(CalendarSnapshot.unavailable("forexfactory", "451")), DAY0, END0)
    assert pre["forced_neutral"] is False and pre["trap"]["label"] == "Trap Agent: Clear" and pre["macro"] == []
    assert len(pre["errors"]) == 2 and "trap ledger unreadable" in pre["errors"][0] and "451" in pre["errors"][1]
    assert pf.plan_kwargs(pre) == {"force_neutral": False, "forced_windows": []}

    class NoCalendar:
        pass
    assert pf.preflight(FakeStore(), NoCalendar(), DAY0, END0)["errors"] == ["calendar not on the snapshot"]


# ---------- build_plan honours them ----------

@pytest.fixture(scope="module")
def trending():
    """Strongly drifting candles: the lean reads a direction on its own."""
    return hourly(800, DAY0 - 800 * H, drift=0.0012, seed=5)


def test_force_neutral_overrides_a_directional_lean(trending):
    free = _plan(trending)
    assert free["regime"] != "neutral", "the fixture must lean on its own for the override to mean anything"
    p = _plan(trending, force_neutral=True)
    assert p["regime"] == "neutral" and p["score"] == 0.0 and p["preflight_override"] == "trap"
    assert set(p["invalidation"]) == {"below", "above"}
    assert p["parts"] == free["parts"], "the reads are kept for the drawer"
    assert p["p_scenario_a"] != free["p_scenario_a"], "the odds are the range's, not the direction's"
    # no drift: the median path ends where it started, to within the simulation's noise
    assert abs(p["hours"][-1]["median"] / p["origin"] - 1) < 0.004
    assert p["scenario_a"].startswith("Range:")


def test_forced_windows_become_chop_hours_with_wider_sigma(trending):
    t = DAY0 + 13 * H + 30 * 60_000
    win = (t - 30 * 60_000, t + 90 * 60_000, "CPI m/m")
    free, p = _plan(trending), _plan(trending, forced_windows=[win])
    assert p["macro_windows"] == [{"title": "CPI m/m", "start_ms": win[0], "end_ms": win[1]}]
    touched = [h for h, h0 in zip(p["hours"], free["hours"]) if (h["t_ms"] - H) < win[1] and h["t_ms"] > win[0]]
    assert [h["t_ms"] for h in touched] == [DAY0 + 14 * H, DAY0 + 15 * H]
    for h, h0 in zip(touched, [h0 for h0 in free["hours"] if h0["t_ms"] in {DAY0 + 14 * H, DAY0 + 15 * H}]):
        assert h["phase"] == "chop" and h["mult"] == h0["mult"]
        assert abs(h["sigma_pct"] - h0["sigma_pct"] * qf.MACRO_SIGMA_X) <= 0.0015, "sigma_pct is rounded to 3 places"
    others = [(h, h0) for h, h0 in zip(p["hours"], free["hours"]) if h["t_ms"] not in {DAY0 + 14 * H, DAY0 + 15 * H}]
    assert all(h["phase"] == h0["phase"] and h["sigma_pct"] == h0["sigma_pct"] for h, h0 in others)
    boxes = [w for w in p["windows"] if w["phase"] == "chop" and w["start_ms"] <= DAY0 + 13 * H and w["end_ms"] >= DAY0 + 15 * H]
    assert len(boxes) == 1, p["windows"]
    assert boxes[0]["start_ms"] <= win[0] and boxes[0]["end_ms"] >= win[1]
    # a window outside the day is dropped, and one clamped to it is recorded clamped
    assert "macro_windows" not in _plan(trending, forced_windows=[(END0 + H, END0 + 2 * H, "x")])
    clamped = _plan(trending, forced_windows=[(END0 - 10 * 60_000, END0 + H, "NFP")])
    assert clamped["macro_windows"][0]["end_ms"] == END0 and clamped["hours"][-1]["phase"] == "chop"


def test_the_kwargs_round_trip_through_the_real_store(tmp_path):
    """The store's own traps() signature and rows, so the fake above cannot drift from it."""
    store = Store(tmp_path / "s.sqlite")
    store.put_trap(category="intraday", side="bear_trap", timeframe="4h", level=79_000.0, invalidation=78_600.0,
                   plays_out=79_800.0, declared_ms=DAY0 - H, score=70.0, head_call="ENGINEERED_TRAP",
                   evidence=["swept the low"])
    store.put_trap(category="live", side="bull_trap", timeframe="1h", level=81_000.0, invalidation=81_400.0,
                   plays_out=80_200.0, declared_ms=DAY0 - H, score=70.0, head_call="ENGINEERED_TRAP",
                   evidence=["swept the high"])
    pre = pf.preflight(store, FakeMarket(cal()), DAY0, END0)
    assert pre["forced_neutral"] and [t["type"] for t in pre["trap"]["traps"]] == ["bear_trap"]
    store.resolve_trap(pre["trap"]["traps"][0]["id"], "played_out", 79_800.0, DAY0)
    assert pf.preflight(store, FakeMarket(cal()), DAY0, END0)["forced_neutral"] is False
