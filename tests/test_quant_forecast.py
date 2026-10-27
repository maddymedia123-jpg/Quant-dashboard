"""Quant Forecast: GARCH, the hourly profile, air pockets, the plan, its invalidation, the audit tiers,
the store's once-only writes, scoring, and the pass that ties them together."""
import math
import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("TI_OFFLINE_FIXTURES", "1")

from core import qf_alerts as qa  # noqa: E402
from core import quant_forecast as qf  # noqa: E402
from core.indicators import garch  # noqa: E402
from core.indicators.liquidity import low_volume_nodes  # noqa: E402
from core.store import Store  # noqa: E402

H = qf.HOUR_MS
DAY0 = 1_788_825_600_000            # a 00:00 UTC


def hourly(n: int, start_ms: int, sigma: float = 0.004, seed: int = 1, loud_hour: int | None = None,
           drift: float = 0.0, price: float = 80_000.0) -> pd.DataFrame:
    """n hourly candles from start_ms, random-walk closes; `loud_hour` is three times as volatile."""
    rng = np.random.default_rng(seed)
    t = start_ms + np.arange(n) * H
    s = np.full(n, sigma)
    if loud_hour is not None:
        s[(t // H) % 24 == loud_hour] *= 3
    close = price * np.exp(np.cumsum(rng.standard_normal(n) * s + drift))
    open_ = np.concatenate([[price], close[:-1]])
    wig = np.abs(rng.standard_normal(n)) * s * close * 0.5
    return pd.DataFrame({"timestamp": t, "open": open_, "high": np.maximum(open_, close) + wig,
                         "low": np.minimum(open_, close) - wig, "close": close,
                         "volume": rng.uniform(50, 150, n)})


def test_garch_recovers_clustering_and_compression():
    rng = np.random.default_rng(3)
    a, b, lr = 0.10, 0.85, 1e-5
    var, r = lr, []
    for _ in range(3000):
        x = math.sqrt(var) * rng.standard_normal()
        r.append(x)
        var = lr * (1 - a - b) + a * x * x + b * var
    f = garch.fit(np.array(r), hours=3000)
    assert abs(f.alpha - a) <= 0.05 and abs(f.persistence - (a + b)) <= 0.04, (f.alpha, f.beta)
    assert garch.fit(np.zeros(500)) is None, "flat prices cannot be fitted"
    calm = garch.fit(np.concatenate([np.array(r), np.array(r[-50:]) * 0.05]), hours=3050)
    assert calm.compression < 0.8, "a quiet last stretch reads as compressed"
    assert abs(calm.var_ahead(500) - calm.long_run_var) / calm.long_run_var < 0.05, "and returns to its level"


def test_the_hourly_profile_finds_the_loud_hour():
    df = hourly(24 * 60, DAY0 - 60 * 24 * H, loud_hour=14)
    prof = garch.hourly_profile(df["timestamp"], df["close"])
    assert len(prof) == 24 and max(range(24), key=prof.__getitem__) == 14
    assert abs(sum(x * x for x in prof) / 24 - 1) < 1e-9, "averages one in variance"
    assert garch.hourly_profile([0, H], [1.0, 1.0]) == [1.0] * 24, "too little data: no pattern"


def test_an_air_pocket_between_two_value_areas_is_found_and_edges_are_not():
    rows = [(100, 101, 1000)] * 40 + [(101, 109, 30)] * 2 + [(109, 110, 1000)] * 40
    df = pd.DataFrame([{"timestamp": i, "open": lo, "high": hi, "low": lo, "close": hi, "volume": v}
                       for i, (lo, hi, v) in enumerate(rows)])
    nodes = low_volume_nodes(df, bins=20, lookback=len(df))
    assert nodes and all(101 <= n.low and n.high <= 109.5 for n in nodes), nodes


def _plan(df, now, **kw):
    return qf.build_plan(df, now, qf.day_start(now) + qf.DAY_MS, float(df["close"].iloc[-1]), **kw)


def test_a_plan_is_complete_and_its_numbers_hang_together():
    df = hourly(800, DAY0 - 800 * H, loud_hour=14)
    p = _plan(df, DAY0)
    assert p["available"] and p["day_ms"] == DAY0 and len(p["hours"]) == 24
    assert p["projected_low"]["price"] < p["origin"] < p["projected_high"]["price"]
    for h in p["hours"]:
        assert h["band_low"] <= h["median"] <= h["band_high"]
    if p["regime"] == "neutral":
        assert set(p["invalidation"]) == {"below", "above"}
    else:
        assert len(p["invalidation"]) == 1
    assert any(w["phase"] == "expansion" and w["start_ms"] <= DAY0 + 14 * H < w["end_ms"] + 1 for w in p["windows"]), \
        "the loud hour is an expansion window"
    assert 0 <= p["p_scenario_a"] <= 1 and p["scenario_a"] and p["scenario_b"]
    assert qf.directive(p, qa.at_label)


def test_chop_hours_carry_no_drift():
    df = hourly(800, DAY0 - 800 * H, drift=0.002)          # a strong trend: the plan leans
    p = _plan(df, DAY0)
    assert p["regime"] == "bullish", p["score"]
    hours = p["hours"]
    for prev, h in zip([{"median": p["origin"]}] + hours, hours):
        if h["phase"] == "chop":
            assert abs(h["median"] / prev["median"] - 1) < 0.0015, "a chop hour is drawn flat"


def test_too_little_history_is_said_not_guessed():
    p = _plan(hourly(100, DAY0 - 100 * H), DAY0)
    assert not p["available"] and "needs" in p["reason"]


def test_a_weak_lean_is_neutral_and_gets_a_box_both_ways(monkeypatch):
    monkeypatch.setattr(qf, "lean_parts", lambda *a, **k: {"momentum": {"value": 0.1, "note": ""}})
    p = _plan(hourly(800, DAY0 - 800 * H), DAY0)
    assert p["regime"] == "neutral" and set(p["invalidation"]) == {"below", "above"}
    assert all(abs(h["median"] / p["origin"] - 1) < 0.002 for h in p["hours"]), "no direction, no drift"


def _plan_fixed(regime="bullish", below=79_000.0, above=None, created=DAY0, atr=200.0):
    inv = {"below": below} if regime == "bullish" else {"above": above} if regime == "bearish" else \
        {"below": below, "above": above}
    return {"available": True, "version": 0, "day_ms": qf.day_start(created), "created_ms": created,
            "end_ms": qf.day_start(created) + qf.DAY_MS, "origin": 80_000.0, "regime": regime,
            "invalidation": inv, "atr_1h": atr, "p_scenario_a": 0.6,
            "hours": [{"t_ms": created + (k + 1) * H, "phase": "normal", "sigma_pct": 0.4, "median": 80_000.0,
                       "band_low": 79_500.0, "band_high": 80_500.0} for k in range(24)],
            "windows": [], "projected_high": {"price": 80_800.0, "t_ms": created + 14 * H},
            "projected_low": {"price": 79_600.0, "t_ms": created + 3 * H}, "scenario_a": "a", "scenario_b": "b"}


def candles(closes, start=DAY0, bar=H):
    c = np.array(closes, dtype=float)
    o = np.concatenate([[c[0]], c[:-1]])
    return pd.DataFrame({"timestamp": start + np.arange(len(c)) * bar, "open": o, "high": np.maximum(o, c) + 5,
                         "low": np.minimum(o, c) - 5, "close": c, "volume": 100.0})


def test_invalidation_needs_a_4h_close_or_an_hourly_close_one_atr_beyond():
    plan = _plan_fixed(below=79_000.0, atr=200.0)
    wick_and_shallow = candles([80_000, 78_950, 80_000, 80_000, 80_000, 80_000])  # 01:00 close below, not a 4h close
    assert qf.invalidation_hit(plan, wick_and_shallow, DAY0 + 6 * H) is None
    at_4h = candles([80_000, 80_000, 80_000, 78_990])                            # closes at 04:00
    hit = qf.invalidation_hit(plan, at_4h, DAY0 + 4 * H)
    assert hit and hit["basis"] == "4-hour close" and hit["close_ms"] == DAY0 + 4 * H
    deep = candles([80_000, 78_700])                                             # 300 below on an hourly close
    hit = qf.invalidation_hit(plan, deep, DAY0 + 2 * H)
    assert hit and hit["basis"].startswith("hourly close")
    assert qf.invalidation_hit(plan, at_4h, DAY0 + 3 * H) is None, "a candle still forming does not count"


def test_a_shock_against_the_plan_is_a_caution_and_taking_half_back_is_model_held():
    plan = _plan_fixed(below=78_000.0)          # 15m sigma = 0.2%; a 2-sigma 15m drop is 0.4%
    m15 = candles([80_000, 80_000, 79_500, 79_600, 79_800, 79_900], bar=qa.M15_MS)
    a = qa.audit(plan, {"1h": candles([80_000]), "15m": m15}, None, DAY0 + 6 * qa.M15_MS)
    assert [c["t_ms"] for c in a["cautions"]] == [DAY0 + 3 * qa.M15_MS]
    assert a["holds"] and a["holds"][0]["t_ms"] == DAY0 + 5 * qa.M15_MS and a["status"] == "model held"
    rally = candles([80_000, 80_000, 80_500], bar=qa.M15_MS)
    assert not qa.audit(plan, {"1h": candles([80_000]), "15m": rally}, None, DAY0 + 3 * qa.M15_MS)["cautions"], \
        "a move WITH a bullish plan is no caution"


def test_a_neutral_plan_is_cautioned_either_way_and_cvd_needs_a_direction():
    plan = _plan_fixed("neutral", below=78_000.0, above=82_000.0)
    m15 = candles([80_000, 80_000, 80_500], bar=qa.M15_MS)
    assert qa.audit(plan, {"1h": candles([80_000]), "15m": m15}, None, DAY0 + 3 * qa.M15_MS)["status"] == "caution"
    flow = pd.DataFrame({"timestamp": DAY0 + np.arange(-100, 4) * qa.M15_MS, "delta": [10.0] * 100 + [-500.0] * 4})
    assert qa.cvd_flips(plan, flow, DAY0 + 4 * qa.M15_MS) == []
    bullish = _plan_fixed()
    flips = qa.cvd_flips(bullish, flow, DAY0 + 4 * qa.M15_MS)
    assert len(flips) == 1 and flips[0]["delta_btc"] < 0, "one caution per stretch, not one per bar"


def test_cautions_are_one_an_hour_and_stale_ones_are_not_sent():
    plan = _plan_fixed()
    a = {"cautions": [{"t_ms": DAY0 + H + 900_000, "why": "move", "sigmas": 2.5, "from": 1, "to": 1},
                      {"t_ms": DAY0 + H + 1_800_000, "why": "move", "sigmas": 3, "from": 1, "to": 1}], "holds": []}
    assert len(qa.audit_events(plan, a, DAY0 + H + 1_900_000)) == 1
    assert qa.audit_events(plan, a, DAY0 + 5 * H) == [], "an hours-old caution is not news"
    text = qa.caution_headline(plan, a["cautions"][0])
    assert "NOT been breached" in text and "79,000" in text


def test_a_plan_is_written_once_per_day_and_version_and_closed_once(tmp_path):
    path = tmp_path / "s.sqlite"
    a, b = Store(path), Store(path)
    plan = _plan_fixed()
    assert a.put_qf_plan(plan) is not None and b.put_qf_plan(plan) is None, "the second writer gets nothing"
    row = b.qf_plans(DAY0)[0]
    assert row["payload"]["origin"] == 80_000.0 and row["status"] == "active"
    assert a.close_qf_plan(row["id"], "invalidated", "x", DAY0 + H) and not b.close_qf_plan(row["id"], "invalidated", "x", DAY0 + H)
    assert [r["id"] for r in a.qf_due(DAY0 + qf.DAY_MS)] == [row["id"]] and a.qf_due(DAY0 + H) == []
    assert a.score_qf_plan(row["id"], {"scored": True}, DAY0 + qf.DAY_MS) and not b.score_qf_plan(row["id"], {}, 0)
    a.add_market_alert(1, "qf_plan", f"qf:{DAY0}:0:plan", "p")
    a.add_market_alert(2, "plan", "other", "o")
    assert [r["dedup_key"] for r in a.market_alerts_keyed(f"qf:{DAY0}:")] == [f"qf:{DAY0}:0:plan"]
    a.close()
    b.close()


def test_scoring_waits_for_the_window_and_reads_the_day():
    plan = _plan_fixed(below=79_000.0)
    closes = [80_000 + 40 * k for k in range(24)]                # a steady rise to 80,920
    assert qf.score_plan(plan, candles(closes[:10])) is None, "not before the window ends"
    o = qf.score_plan(plan, candles(closes))
    assert o["scored"] and o["direction_hit"] and not o["invalidated"] and o["scenario_a_right"]
    assert o["high_hit"] and o["close"] == 80_920.0
    assert qf.record([o, {"scored": False}, None])["days"] == 1


def test_the_backtest_replays_past_days_price_only():
    df = hourly(24 * 28, DAY0 - 24 * 28 * H, seed=5)        # 560 warm-up hours, then three days
    bt = qf.backtest(df, DAY0, days=3, paths=200)
    assert bt["days"] == 3 and 0 <= bt["scenario_a_right"] <= 3 and "price only" in bt["method"]
    assert qf.backtest(df.head(10), DAY0)["days"] == 0


class _M:
    """A market snapshot with just what the forecast reads."""
    def __init__(self, h1, m15=None, price=None):
        self.spot = type("S", (), {"frames": {"1h": h1, "15m": m15}, "last": price or float(h1["close"].iloc[-1])})()
        self.taker_flow = type("F", (), {"available": False, "frames": {}})()


def test_one_pass_makes_the_day_plan_once_and_replaces_it_when_broken(tmp_path, monkeypatch):
    store = Store(tmp_path / "s.sqlite")
    hist = hourly(800, DAY0 - 800 * H)
    now = DAY0 + 60_000
    ev = qa.run(store, _M(hist), now)
    assert [e.kind for e in ev] == ["qf_plan"] and qa.run(store, _M(hist), now + 1) == []
    plan = store.qf_plans(DAY0)[0]["payload"]
    assert plan["backtest"]["days"] >= 1

    # force the regime and break the line on the 04:00 close
    level = plan["invalidation"].get("below") or plan["invalidation"]["above"]
    side = "below" if "below" in plan["invalidation"] else "above"
    away = level * (0.99 if side == "below" else 1.01)
    later = pd.concat([hist, candles([plan["origin"]] * 3 + [away], start=DAY0)], ignore_index=True)
    ev = qa.run(store, _M(later, price=away), DAY0 + 4 * H + 60_000)
    assert [e.kind for e in ev] == ["qf_invalid"] and "NEW FORECAST #1" in ev[0].headline
    rows = store.qf_plans(DAY0)
    assert [(r["version"], r["status"]) for r in rows] == [(0, "invalidated"), (1, "active")]
    assert rows[1]["payload"]["replaces"]["version"] == 0

    # the second failure locks the day
    p1 = rows[1]["payload"]
    lvl = p1["invalidation"].get("below") or p1["invalidation"]["above"]
    far = lvl * (0.98 if "below" in p1["invalidation"] else 1.02)
    later2 = pd.concat([later, candles([away] * 3 + [far], start=DAY0 + 4 * H)], ignore_index=True)
    ev = qa.run(store, _M(later2, price=far), DAY0 + 8 * H + 60_000)
    assert [e.kind for e in ev] == ["qf_invalid"] and "SYSTEM LOCKED" in ev[0].headline
    rows = store.qf_plans(DAY0)
    assert rows[-1]["regime"] == "locked" and rows[-1]["status"] == "active"
    assert qa.run(store, _M(later2, price=far), DAY0 + 9 * H) == [], "a locked day raises nothing more"
    store.close()


def test_the_chart_and_panels_build_from_a_plan():
    from ui import panels
    from ui.charts import build_qf_chart_html, qf_chart_payload
    df = hourly(800, DAY0 - 800 * H, loud_hour=14)
    p = _plan(df, DAY0)
    payload = qf_chart_payload(p, df, dark=False, now_ms=DAY0)
    assert len(payload["path"]) == 25 and payload["lines"] and payload["markers"]
    assert {b["kind"] for b in payload["boxes"]} & {"chop", "expansion"}
    assert "__PAYLOAD__" not in build_qf_chart_html(p, df, dark=True, now_ms=DAY0)
    live = {"days": 0}
    for markup in (panels.qf_gameplan_html(p, live), panels.qf_levels_html(p),
                   panels.qf_status_html("caution", "⚠️ CAUTION", [{"version": 0, "status": "active"}]),
                   panels.qf_record_html({"days": 2, "scenario_a_right": 1, "high_hit": 1, "low_hit": 0,
                                          "high_time_hit": 0, "low_time_hit": 0, "band_held": 1, "invalidated": 0},
                                         live, [{"day_ms": DAY0, "version": 0, "regime": "bullish",
                                                 "status": "active", "outcome": None}])):
        assert markup.startswith("<div class=\"ti-card")
    locked = qf.locked_plan(DAY0, DAY0 + qf.DAY_MS, 80_000.0, 2, 81_000.0, 79_000.0)
    assert "Stand aside" in panels.qf_gameplan_html(locked, live)


def test_range_days_are_counted_apart_from_direction_calls():
    days = [{"scored": True, "regime": "neutral", "scenario_a_right": True},
            {"scored": True, "regime": "bullish", "scenario_a_right": False},
            {"scored": True, "regime": "bearish", "scenario_a_right": True}]
    r = qf.record(days)
    assert (r["direction_days"], r["direction_right"], r["range_days"], r["range_held"]) == (2, 1, 1, 1)
    words = qa.record_words({"regime": "bullish", "backtest": r}, {"days": 0})
    assert words == ["backtest: direction right 1 of 2 days"], "a range day never props up a direction call"
    assert qa.record_words({"regime": "neutral", "backtest": r}, None) == ["backtest: range held 1 of 1 days"]
