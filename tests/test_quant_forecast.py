"""Quant Forecast: GARCH, the hourly profile, air pockets, the plan, its invalidation, the audit tiers,
the store's once-only writes, scoring, and the pass that ties them together."""
import json
import math
import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("TI_OFFLINE_FIXTURES", "1")

from core import qf_alerts as qa  # noqa: E402
from core import quant_forecast as qf  # noqa: E402
from core import qf_tree  # noqa: E402
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


def test_invalidation_is_a_decisive_breach_of_the_whole_day_level():
    """Hammad, message 5: a whole-day level, breached by sustained price action or a violent cascade - made
    exact as a 4h close, two hourly closes in a row, or one hourly close an ATR beyond."""
    plan = _plan_fixed(below=79_000.0, atr=200.0)
    one_shallow = candles([80_000, 78_950, 80_000, 80_000, 80_000, 80_000])      # one 01:00 close 50 below, back by 02:00
    assert qf.invalidation_hit(plan, one_shallow, DAY0 + 6 * H) is None, "a single shallow hourly close is a wick"
    at_4h = candles([80_000, 80_000, 80_000, 78_990])                            # closes at 04:00
    hit = qf.invalidation_hit(plan, at_4h, DAY0 + 4 * H)
    assert hit and hit["basis"] == "4-hour close" and hit["close_ms"] == DAY0 + 4 * H
    assert qf.invalidation_hit(plan, at_4h, DAY0 + 3 * H) is None, "a candle still forming does not count"
    sustained = candles([80_000, 78_950, 78_960, 80_000])                        # the 02:00 and 03:00 closes both below
    hit = qf.invalidation_hit(plan, sustained, DAY0 + 4 * H)
    assert hit and hit["basis"] == "2 hourly closes in a row" and hit["close_ms"] == DAY0 + 3 * H
    cascade = candles([80_000, 78_700, 80_000, 80_000])                          # 300 below on the 02:00 close
    hit = qf.invalidation_hit(plan, cascade, DAY0 + 4 * H)
    assert hit and hit["basis"].startswith("hourly close more than $200") and hit["close_ms"] == DAY0 + 2 * H
    assert "two hourly closes in a row" in qf.directive(plan, qa.at_label)
    # message 6: one firm hourly close (half an ATR) with the hour's taker flow behind the break
    firm = candles([80_000, 78_880, 80_000, 80_000])                             # 120 below on the 02:00 close
    assert qf.invalidation_hit(plan, firm, DAY0 + 4 * H) is None, "firm but no flow: a wick"
    sellers = pd.DataFrame({"timestamp": [DAY0, DAY0 + H], "delta": [10.0, -250.0]})
    hit = qf.invalidation_hit(plan, firm, DAY0 + 4 * H, sellers)
    assert hit and hit["close_ms"] == DAY0 + 2 * H and "250 BTC of taker flow behind it" in hit["basis"]
    buyers = pd.DataFrame({"timestamp": [DAY0, DAY0 + H], "delta": [10.0, 250.0]})
    assert qf.invalidation_hit(plan, firm, DAY0 + 4 * H, buyers) is None, "buyers on a break below is not opposing CVD"


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
    a = {"cautions": [{"t_ms": DAY0 + H, "why": "move", "sigmas": 2.5, "from": 1, "to": 1},
                      {"t_ms": DAY0 + H + 1_800_000, "why": "move", "sigmas": 3, "from": 1, "to": 1}], "holds": []}
    assert len(qa.audit_events(plan, a, DAY0 + H + 1_900_000)) == 1
    assert qa.audit_events(plan, a, DAY0 + 5 * H) == [], "an hours-old caution is not news"
    text = qa.caution_headline(plan, a["cautions"][0])
    assert "NOT been breached" in text and "79,000" in text
    # the worker was down for the first caution of the hour: the fresh one still goes out, and the hour's
    # ledger key is what keeps a third from following it
    late = qa.audit_events(plan, a, DAY0 + H + 1_800_000 + 5 * 60_000)
    assert [e.headline for e in late] == [qa.caution_headline(plan, a["cautions"][1])]
    # a MODEL HELD answers only a caution that was sent: the second shock of an hour never was
    held = {"cautions": [{"t_ms": DAY0 + H, "why": "move", "sigmas": 2.5, "from": 1, "to": 1},
                         {"t_ms": DAY0 + H + 900_000, "why": "move", "sigmas": 3, "from": 1, "to": 1}],
            "holds": [{"t_ms": DAY0 + H + 1_800_000, "shock_ms": DAY0 + H + 900_000, "from": 1, "to": 1}]}
    assert [e.kind for e in qa.audit_events(plan, held, DAY0 + H + 1_800_000)] == ["qf_caution"]
    held["holds"][0]["shock_ms"] = DAY0 + H
    assert [e.kind for e in qa.audit_events(plan, held, DAY0 + H + 1_800_000)] == ["qf_caution", "qf_held"]


def test_the_15m_caution_sigma_is_the_whole_hours_not_the_scaled_first_hour():
    plan = _plan_fixed()
    plan["hours"][0].update(sigma_pct=0.05, sigma_full_pct=0.4)      # made at :53, seven minutes of hour left
    m15 = candles([80_000, 80_000, 79_760], bar=qa.M15_MS)          # -0.3%: 1.5 sigma of the hour, not 12
    assert qa.shocks(plan, m15, DAY0 + 3 * qa.M15_MS) == []
    plan["hours"][0]["sigma_full_pct"] = 0.1
    assert len(qa.shocks(plan, m15, DAY0 + 3 * qa.M15_MS)) == 1
    p = _plan(hourly(800, DAY0 - 800 * H), DAY0 + 53 * 60_000)
    assert p["hours"][0]["sigma_full_pct"] > 2 * p["hours"][0]["sigma_pct"], "the plan carries both"


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
    # a candle the feed never delivered must not leave the plan unscored forever
    day = candles(closes)
    assert qf.score_plan(plan, day.drop(index=[5, 6]))["scored"], "two missing hours are tolerated"
    gap = qf.score_plan(plan, day.drop(index=[3, 4, 5, 6]))
    assert gap["scored"] is False and "missing" in gap["reason"]
    # an invalidated plan is judged on the hours it was live
    short = qf.score_plan(plan, day, until_ms=DAY0 + 8 * H)
    assert short["scored"] and short["close"] == closes[7] and short["high"] < 80_500


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
    # the morning post carries the compact timeline (message 6); this day is normal with lone chop and
    # expansion hours, which fold into one node, so the line names the active node and no next one
    assert ("Now ⚡" in ev[0].headline or "Next 🎯" in ev[0].headline), ev[0].headline
    assert len(qf_tree.timeline_tree(store.qf_plans(DAY0)[0]["payload"], now)) <= 6, "no fragmented day"
    plan = store.qf_plans(DAY0)[0]["payload"]
    assert plan["backtest"]["days"] >= 1
    if plan["regime"] != "neutral":
        assert "Trade: " in ev[0].headline and "stop " in ev[0].headline

    # force the regime and break the line on the 04:00 close
    level = plan["invalidation"].get("below") or plan["invalidation"]["above"]
    side = "below" if "below" in plan["invalidation"] else "above"
    away = level * (0.99 if side == "below" else 1.01)
    later = pd.concat([hist, candles([plan["origin"]] * 3 + [away], start=DAY0)], ignore_index=True)
    ev = qa.run(store, _M(later, price=away), DAY0 + 4 * H + 60_000)
    assert [e.kind for e in ev] == ["qf_invalid"] and "SCENARIO B · FORECAST #1" in ev[0].headline
    assert ev[0].headline.startswith("🔴 WHOLE-DAY PLAN INVALIDATED: Macro structure breached at"), "Hammad's words"
    assert "Original roadmap voided. Scenario B secondary chart activated with new targets." in ev[0].headline
    rows = store.qf_plans(DAY0)
    assert [(r["version"], r["status"]) for r in rows] == [(0, "invalidated"), (1, "active")]
    assert rows[1]["payload"]["replaces"]["version"] == 0

    # one re-forecast a day: the second failure locks the day (Hammad, message 5)
    p1 = rows[1]["payload"]
    lvl = p1["invalidation"].get("below") or p1["invalidation"]["above"]
    far = lvl * (0.98 if "below" in p1["invalidation"] else 1.02)
    later2 = pd.concat([later, candles([away] * 3 + [far], start=DAY0 + 4 * H)], ignore_index=True)
    ev = qa.run(store, _M(later2, price=far), DAY0 + 8 * H + 60_000)
    assert [e.kind for e in ev] == ["qf_invalid"] and "LOCKOUT MODE: HIGH CHOP / STAND ASIDE" in ev[0].headline
    assert "Invalidated twice today" in ev[0].headline and "Two plans failed" in ev[0].headline
    rows = store.qf_plans(DAY0)
    assert rows[-1]["regime"] == "locked" and rows[-1]["status"] == "active" and rows[-1]["version"] == 2
    assert qa.run(store, _M(later2, price=far), DAY0 + 9 * H) == [], "a locked day raises nothing more"
    store.close()


def test_a_day_left_with_no_active_plan_gets_its_replacement_on_the_next_pass(tmp_path):
    """A crash between closing a plan and writing its replacement (or a replacement that could not be built)
    used to leave the day dead and silent."""
    store = Store(tmp_path / "s.sqlite")
    hist = hourly(800, DAY0 - 800 * H)
    qa.run(store, _M(hist), DAY0 + 60_000)
    row = store.qf_plans(DAY0)[0]
    hit = {"side": "below", "level": 1.0, "close": 0.5, "close_ms": DAY0 + 4 * H, "basis": "4-hour close"}
    assert store.close_qf_plan(row["id"], "invalidated", json.dumps(hit), DAY0 + 4 * H)
    ev = qa.run(store, _M(hist), DAY0 + 4 * H + 120_000)
    assert [e.kind for e in ev] == ["qf_invalid"] and "SCENARIO B · FORECAST #1" in ev[0].headline
    assert [(r["version"], r["status"]) for r in store.qf_plans(DAY0)] == [(0, "invalidated"), (1, "active")]
    assert qa.run(store, _M(hist), DAY0 + 4 * H + 180_000) == []
    store.close()


def test_the_plan_row_is_written_before_the_ai_is_asked(tmp_path, monkeypatch):
    """Insert first, then the write-up: a slow or crashing model can never leave the day without a plan,
    and only the process that won the row makes the call."""
    store = Store(tmp_path / "s.sqlite")
    calls = []

    def write(plan, live=None):
        calls.append(plan["version"])
        assert store.qf_plans(DAY0), "the row exists before the model is asked"
        return {"summary": "Dips hold, then up.", "scenario_a": "Up.", "scenario_b": "Fails.", "model": "t"}
    monkeypatch.setattr(qw, "write", write)
    ev = qa.run(store, _M(hourly(800, DAY0 - 800 * H)), DAY0 + 60_000)
    assert calls == [0] and "Directive: Dips hold, then up." in ev[0].headline
    assert store.qf_plans(DAY0)[0]["payload"]["writeup"]["summary"] == "Dips hold, then up."
    assert qa.run(store, _M(hourly(800, DAY0 - 800 * H)), DAY0 + 60_000) == [] and calls == [0], \
        "a second process finds the row and asks nothing"
    store.close()


def test_the_other_detectors_failing_does_not_block_the_forecast_or_the_send_step(tmp_path, monkeypatch):
    from core import pipeline
    store = Store(tmp_path / "s.sqlite")

    def boom(*a, **k):
        raise RuntimeError("market events broke")
    monkeypatch.setattr(pipeline, "market_events", boom)
    r = pipeline.market_alerts(store, _M(hourly(800, DAY0 - 800 * H)), {}, DAY0 + 60_000, {}, may_send=False)
    assert any("market events broke" in p for p in r.problems)
    assert [a["kind"] for a in store.market_alerts_keyed(f"qf:{DAY0}:")] == ["qf_plan"], \
        "the day plan was still made and logged"
    store.close()


def test_the_chart_and_panels_build_from_a_plan():
    from ui import panels
    from ui.charts import build_qf_chart_html, qf_chart_payload
    df = hourly(800, DAY0 - 800 * H, loud_hour=14)
    p = _plan(df, DAY0)
    payload = qf_chart_payload(p, df, dark=False, now_ms=DAY0)
    assert len(payload["path"]) == 25 and payload["now"] == DAY0 // 1000
    assert {b["kind"] for b in payload["boxes"]} & {"chop", "expansion"}
    # right-axis badges only: the high, the low, the invalidation and the air pockets, no text on the chart
    kinds = [ln["kind"] for ln in payload["lines"]]
    assert kinds.count("high") == 1 and kinds.count("low") == 1 and kinds.count("invalidation") == len(p["invalidation"])
    assert "markers" not in payload and all("label" not in b for b in payload["boxes"])
    assert all("title" not in ln for ln in payload["lines"])
    html_dark = build_qf_chart_html(p, df, dark=False, now_ms=DAY0)
    assert "__PAYLOAD__" not in html_dark and "#16181A" in html_dark, "the chart is the dark terminal in light mode too"
    assert "fillText(b.label" not in html_dark and "title: ''" in html_dark
    live = {"days": 0}
    for markup in (panels.qf_wire_html(p, "on track"), panels.qf_gameplan_html(p, live), panels.qf_levels_html(p),
                   panels.qf_status_html("caution", ["⚠️ CAUTION", "📊 plan"], [{"version": 0, "status": "active"}]),
                   panels.qf_record_html({"days": 2, "scenario_a_right": 1, "high_hit": 1, "low_hit": 0,
                                          "high_time_hit": 0, "low_time_hit": 0, "band_held": 1, "invalidated": 0},
                                         live, [{"day_ms": DAY0, "version": 0, "regime": "bullish",
                                                 "status": "active", "outcome": None}])):
        assert markup.startswith("<div class=\"ti-card")
    assert "the 2nd locks the day" in panels.qf_status_html("on track", None, [])
    assert "Volatility schedule" in panels.qf_schedule_html(p, DAY0) and "bps" in panels.qf_schedule_html(p, DAY0)
    # message 6: the timeline tree fades what passed and glows on what is active; the directive is pure figures
    tl = panels.qf_timeline_html(p, DAY0 + 5 * H, df)
    assert "ti-tl-done" in tl and "ti-tl-active" in tl and "ti-tl-pending" in tl and "🔥 Active Focus" in tl
    assert "<em>Why:</em>" in tl and "GARCH volatility" in tl
    box = panels.qf_directive_html(p)
    assert "Intraday trade directive" in box and "Stop loss (whole-day invalidation)" in box and "TP3" in box
    assert any(f">{w}<" in box for w in ("LONG", "SHORT", "NONE"))
    assert ">NONE<" in panels.qf_directive_html(qf.locked_plan(DAY0, DAY0 + qf.DAY_MS, 1.0, 2, None, None))
    # every panel is one Markdown HTML block: a blank line or a <pre> would hand the rest to the Markdown parser
    for markup in (box, tl, panels.qf_wire_html(p, "on track"), panels.qf_gameplan_html(p, live)):
        assert "\n\n" not in markup and "<pre" not in markup
    locked = qf.locked_plan(DAY0, DAY0 + qf.DAY_MS, 80_000.0, 3, 81_000.0, 79_000.0)
    assert "Stand aside" in panels.qf_gameplan_html(locked, live) and "Stand aside" in panels.qf_wire_html(locked, "locked")
    lp = qf_chart_payload(locked, df, now_ms=DAY0 + 12 * H)
    assert [b["kind"] for b in lp["boxes"]] == ["chop"] and lp["boxes"][0]["p1"] == 79_000.0 and lp["path"] == [
        {"time": DAY0 // 1000, "value": 80_000.0}]
    assert lp["whitespace"] and lp["whitespace"][-1]["time"] == (DAY0 + qf.DAY_MS) // 1000


def test_the_wire_plan_carries_no_math_and_the_bot_message_matches_it():
    p = _plan_fixed()
    p["scenario_a"], p["scenario_b"] = qf.scenarios(p)
    p["backtest"] = {"days": 26, "direction_days": 26, "direction_right": 8}
    p["garch"] = {"alpha": 0.1, "beta": 0.8, "compression": 0.9}
    from ui import panels
    wire = panels.qf_wire_html(p, "on track")
    for banned in ("of 26", "alpha", "Hurst", "GARCH", "Written by", "0.6"):
        assert banned not in wire, banned
    assert "BULLISH EXPANSION | 60% Confidence" in wire and "Whole-day invalidation" in wire
    assert "$79,600" in wire and "03:00 UTC (08:00 PKT)" in wire and "$80,800" in wire
    assert "dips hold above $79,600" in wire.lower() and "Higher:" not in wire
    msg = qa.plan_headline(p)
    assert "direction right" not in msg and "Directive: Dips hold above" in msg
    assert "Low $79,600 at 03:00 UTC (08:00 PKT) → High $80,800 at 14:00 UTC (19:00 PKT)" in msg
    assert msg.splitlines()[-1] == "Whole-day invalidation: below $79,000"
    assert qa.regime_line(p, "Trap Agent: Clear") == "BULLISH EXPANSION | 60% Confidence | Trap Agent: Clear"
    # the hit rate (message 5): the backtest's last 10 days until a live day has finished, then the live ones
    p["backtest"]["recent"] = {"direction_days": 10, "direction_right": 7, "range_days": 0, "range_held": 0}
    assert qa.regime_line(p) == "BULLISH EXPANSION | 60% Confidence | backtest: 7 of last 10 days"
    live = {"days": 3, "direction_days": 3, "direction_right": 1, "range_days": 0, "range_held": 0}
    assert "live: 1 of last 3 days" in qa.regime_line(p, live=live) and "7 of last 10" not in qa.regime_line(p, live=live)
    assert "live: 1 of last 3 days" in qa.plan_headline(p, live)
    neutral = {**p, "regime": "neutral", "invalidation": {"below": 79_000.0, "above": 81_000.0}}
    assert qf.hit_rate(neutral, live) is None, "a range plan quotes range days only, and there are none"
    assert qf.wire_line({"regime": "neutral", "scenario_a": "Range: price stays between."}) == "Price stays between."


def test_range_days_are_counted_apart_from_direction_calls():
    days = [{"scored": True, "regime": "neutral", "scenario_a_right": True},
            {"scored": True, "regime": "bullish", "scenario_a_right": False},
            {"scored": True, "regime": "bearish", "scenario_a_right": True}]
    r = qf.record(days)
    assert (r["direction_days"], r["direction_right"], r["range_days"], r["range_held"]) == (2, 1, 1, 1)
    words = qa.record_words({"regime": "bullish", "backtest": r}, {"days": 0})
    assert words == ["backtest: direction right 1 of 2 days"], "a range day never props up a direction call"
    assert qa.record_words({"regime": "neutral", "backtest": r}, None) == ["backtest: range held 1 of 1 days"]


# ---------- step 5: the AI write-up ----------
from core import qf_writeup as qw  # noqa: E402


def _good_text(p):
    hi, lo = p["projected_high"], p["projected_low"]
    return {"summary": f"Today leans {'up' if p['regime'] == 'bullish' else 'down'}. The likely high is "
                       f"${hi['price']:,.0f} around {qa.at_label(hi['t_ms'])} and the low ${lo['price']:,.0f}; "
                       f"the main path has a {p['p_scenario_a']:.0%} chance and the plan remains in force.",
            "scenario_a": "Price works towards the high.", "scenario_b": "The plan is cancelled and replaced."}


def test_the_check_catches_invented_numbers_times_odds_direction_and_jargon():
    p = _plan_fixed()
    ok = " ".join(_good_text(p).values())
    assert qw.check(ok, p) == [], "its own numbers, rounded to the dollar, pass; 'remains' is not 'ema'"
    assert qw.check("Watch $81,000.", p) == ["price $81,000 is not in the plan"]
    assert qw.check("Watch $80.8k.", p) == [], "80.8k is the projected high"
    assert qw.check("Support at $80,000 keeps the plan alive.", p) == [], "'keeps' is not a thousands suffix"
    assert qw.check("Moves come at 07:00.", p) == ["time 07:00 is not in the plan"]
    assert qw.check("A 75% chance.", p) == ["75% is not one of the plan's odds"]
    assert qw.check("A bearish day.", p) == ["says 'bearish' on a bullish plan"]
    assert qw.check("Buyers paid up on taker flow and funding.", p) == ["jargon: 'taker'", "jargon: 'funding'"]


def test_a_good_reply_is_used_a_bad_or_failed_one_leaves_the_code_text():
    p = _plan_fixed()
    w = qw.write(p, ask=lambda s, u: {"data": _good_text(p), "model": "gemini-test"})
    assert w["summary"].startswith("Today leans up") and w["model"] == "gemini-test"
    bad = {**_good_text(p), "summary": "BTC will hit $90,000 for sure."}
    assert "not in the plan" in qw.write(p, ask=lambda s, u: {"data": bad})["rejected"]
    assert "missing" in qw.write(p, ask=lambda s, u: {"data": {"summary": "x"}})["rejected"]

    def down(s, u):
        raise TimeoutError("slow")
    assert "did not answer" in qw.write(p, ask=down)["rejected"]
    assert qw.write(p) is None, "never called on offline fixture data"
    assert qw.write(qf.locked_plan(DAY0, DAY0 + qf.DAY_MS, 80_000.0, 2, None, None), ask=down) is None


def test_the_model_is_shown_the_numbers_already_written_out():
    p = _plan_fixed()
    seen = {}
    qw.write(p, ask=lambda s, u: seen.update(system=s, user=u) or {"data": _good_text(p)})
    assert "$80,800 around" in seen["user"] and "price holds below $79,000" in seen["user"]
    assert "Never round" in seen["system"] and "ONE direct sentence" in seen["system"]
    # it is shown nothing it is forbidden to say, and no record to quote
    p["parts"] = {"funding": {"value": 0.2, "note": "funding 0.0100% per 8h"},
                  "flow": {"value": 0.1, "note": "24h taker delta +120 BTC"}}
    p["levels"] = [{"label": "gamma flip", "price": 80_500.0, "side": "above", "p_touch": 0.4}]
    p["backtest"] = {"days": 26, "direction_days": 26, "direction_right": 8}
    qw.write(p, live={"days": 3, "direction_days": 3, "direction_right": 1},
             ask=lambda s, u: seen.update(user=u) or {"data": _good_text(p)})
    low = seen["user"].lower()
    for w in ("funding", "taker", "delta", "gamma", "of 26", "of 3 days"):
        assert w not in low, w
    assert "hedging flip level $80,500" in seen["user"]
    long = {**_good_text(p), "summary": "x" * 300}
    assert "too long" in qw.write(p, ask=lambda s, u: {"data": long})["rejected"]


def test_the_plan_carries_the_write_up_into_the_message_and_the_screen(tmp_path, monkeypatch):
    from ui import panels
    store = Store(tmp_path / "s.sqlite")
    monkeypatch.setattr(qw, "write", lambda plan, live=None: {"summary": "A calm day, leaning up.",
                                                             "scenario_a": "Up.", "scenario_b": "Fails.",
                                                             "model": "gemini-test"})
    ev = qa.run(store, _M(hourly(800, DAY0 - 800 * H)), DAY0 + 60_000)
    assert "Directive: A calm day, leaning up." in ev[0].headline
    plan = store.qf_plans(DAY0)[0]["payload"]
    assert plan["writeup"]["model"] == "gemini-test"
    shown = panels.qf_gameplan_html(plan, {"days": 0})
    assert "A calm day, leaning up." in shown and "Written by gemini-test" in shown
    plan["writeup"] = {"rejected": "price $1 is not in the plan"}
    assert "was not used (price $1 is not in the plan)" in panels.qf_gameplan_html(plan, {"days": 0})
    store.close()


def test_an_active_4h_trap_forces_the_day_neutral_and_the_wire_says_so(tmp_path):
    """Pre-flight (Hammad, 28 Oct): an engineered trap active on the 4h overrides the lean; the top line of
    the plan and the bot's message carry the Trap Agent's word for it."""
    from ui import panels
    store = Store(tmp_path / "s.sqlite")
    hist = hourly(800, DAY0 - 800 * H, drift=0.0008, seed=4)        # a strong up-drift: a bullish lean
    assert _plan(hist, DAY0 + 60_000)["regime"] == "bullish", "the fixture leans without the trap"
    store.put_trap("live", "bull_trap", "4h", 80_500.0, 80_900.0, 79_000.0, DAY0 - H, 0.8, "ENGINEERED_TRAP", ["x"])
    ev = qa.run(store, _M(hist), DAY0 + 60_000)
    plan = store.qf_plans(DAY0)[0]["payload"]
    assert plan["regime"] == "neutral" and plan["preflight_override"] == "trap"
    assert plan["preflight"]["trap"]["label"] == "Trap Agent: BULL TRAP active (4H)"
    assert "NEUTRAL / RANGE-BOUND" in ev[0].headline and "| Trap Agent: BULL TRAP active (4H)" in ev[0].headline
    wire = panels.qf_wire_html(plan, "on track")
    assert "Trap Agent: BULL TRAP active (4H)" in wire
    assert "overridden to NEUTRAL" in panels.qf_gameplan_html(plan, {"days": 0})
    store.close()
    clear = Store(tmp_path / "c.sqlite")
    ev = qa.run(clear, _M(hist), DAY0 + 60_000)
    assert "| Trap Agent: Clear" in ev[0].headline and clear.qf_plans(DAY0)[0]["payload"]["regime"] == "bullish"
    clear.close()


def test_the_volatility_schedule_and_the_payload_follow_the_plan():
    from core import qf_payload
    df = hourly(800, DAY0 - 800 * H, loud_hour=14)
    p = _plan(df, DAY0 + 60_000)
    sched = qf.volatility_schedule(p, DAY0 + 5 * H)
    assert [h["t_ms"] for h in sched["next"]] == [DAY0 + 6 * H, DAY0 + 7 * H, DAY0 + 8 * H]
    assert all(h["bps"] > 0 and h["low"] < h["high"] for h in sched["next"])
    assert [b["start_ms"] for b in sched["blocks"]] == [DAY0 + 60_000] + [DAY0 + 4 * k * H for k in range(1, 6)]
    assert [b["done"] for b in sched["blocks"]] == [True, False, False, False, False, False]
    loud = sched["blocks"][3]                                          # 12:00-16:00 holds the loud hour
    assert loud["bps"] > sched["blocks"][1]["bps"] and loud["character"] in ("NORMAL", "HIGH VOLATILITY SPIKE")
    for b in sched["blocks"]:
        assert all(b["low"] <= lv["price"] <= b["high"] for lv in b["levels"])
    assert qf.volatility_schedule(qf.locked_plan(DAY0, DAY0 + qf.DAY_MS, 1.0, 2, None, None), DAY0) == {"next": [], "blocks": []}

    out = qf_payload.payload(p, "caution", [{"version": 0, "status": "active"}], "⚠️ x", {"days": 0}, DAY0 + 5 * H, df)
    assert set(out) == {"audit_and_alert_status", "layman_summary_and_timeline_tree", "tradingview_chart_payload",
                        "collapsible_raw_math_drawer"}, "Hammad's message-6 schema"
    a = out["audit_and_alert_status"]
    assert a["tier_status"] == "TIER-1 CAUTION" and a["re_forecast_count"] == 0 and a["bot_message"] == "⚠️ x"
    assert a["magnitude_scaler"] == {"scaled": False} and "Now ⚡" in a["timeline_status"]
    lay = out["layman_summary_and_timeline_tree"]
    assert lay["master_regime"] == qa.REGIME_WORD[p["regime"]] and lay["daily_coordinates"]["predicted_high"]["pkt_time"]
    assert lay["daily_coordinates"]["whole_day_macro_invalidation"] == p["invalidation"]
    nodes = lay["timeline_tree_nodes"]
    assert nodes and {n["state"] for n in nodes} <= {"COMPLETED_FADED", "ACTIVE_HIGHLIGHTED", "PENDING"}
    assert sum(1 for n in nodes if n["state"] == "ACTIVE_HIGHLIGHTED") == 1 and nodes[0]["time_window_utc"].startswith("00:01")
    assert all(n["summary_oneliner"] and n["rationale_why"] and n["price_levels"]["low"] < n["price_levels"]["high"] for n in nodes)
    box = lay["trade_directive_box"]
    assert set(box) >= {"direction", "stop_loss_invalidation", "entry_zone", "tp1", "tp2", "tp3"}
    traj = out["tradingview_chart_payload"]["hourly_trajectory_24h"]
    assert len(traj) == 24 and traj[0]["hour_index"] == 1 and traj[0]["projected_open"] == p["origin"]
    assert traj[1]["projected_open"] == traj[0]["projected_close"] and traj[0]["timestamp_pkt"] == "06:00"
    assert {t["regime_type"] for t in traj} <= {"CHOP_BOX", "EXPANSION_VECTOR", "NORMAL"}
    ov = out["tradingview_chart_payload"]["visual_overlays"]
    assert ov["scenario_b_secondary_chart_active"] is False
    badges = ov["right_axis_badges"]
    assert badges["predicted_high"] == p["projected_high"]["price"] and "lvn_air_pockets" in badges
    raw = out["collapsible_raw_math_drawer"]
    assert raw["garch_alpha"] == p["garch"]["alpha"] and "hurst_exponent" in raw and raw["backtest_summary_stats"] == {}
    assert len(raw["volatility_schedule_4h"]) == 6
    import json
    json.dumps(out)
    b = qf_payload.payload({**p, "version": 1}, "on track", [{"version": 0, "status": "invalidated"},
                                                               {"version": 1, "status": "active"}], "", {}, DAY0 + 5 * H)
    assert b["tradingview_chart_payload"]["visual_overlays"]["scenario_b_secondary_chart_active"] is True


def test_the_timeline_tree_and_the_trade_directive_follow_the_clock_and_the_plan():
    from core import qf_tree
    p = _plan_fixed()                                           # bullish, 24 normal hours: one node
    p["levels"] = [{"label": "value low", "price": 79_650.0, "side": "below", "p_touch": 0.5},
                   {"label": "POC", "price": 80_200.0, "side": "above", "p_touch": 0.6},
                   {"label": "call wall", "price": 80_500.0, "side": "above", "p_touch": 0.3}]
    nodes = qf_tree.timeline_tree(p, DAY0)
    assert len(nodes) == 1 and nodes[0]["state"] == "ACTIVE_HIGHLIGHTED" and nodes[0]["emoji"] == "⚡"
    assert "Pullback floor near $79,600" in nodes[0]["summary"] and "Day's high near $80,800" in nodes[0]["summary"]
    assert nodes[0]["why"].startswith("normal GARCH volatility") and "levels inside: value low $79,650" in nodes[0]["why"]
    assert qf_tree.timeline_tree(p, DAY0 - 1)[0]["state"] == "PENDING"
    done = qf_tree.timeline_tree(p, DAY0 + qf.DAY_MS, candles([80_000, 80_300, 79_700], start=DAY0))
    assert done[0]["state"] == "COMPLETED_FADED" and done[0]["actual_low"] == 79_695.0 and "Traded $79,695 to $80,305" in done[0]["summary"]
    # three phases make three nodes, in order, each spanning its own hours
    p3 = _plan_fixed()
    for h in p3["hours"][:6]:
        h["phase"] = "chop"
    for h in p3["hours"][13:16]:
        h["phase"] = "expansion"
    n3 = qf_tree.timeline_tree(p3, DAY0 + 7 * H)
    assert [n["phase"] for n in n3] == ["chop", "normal", "expansion", "normal"]
    assert [n["state"] for n in n3] == ["COMPLETED_FADED", "ACTIVE_HIGHLIGHTED", "PENDING", "PENDING"]
    assert n3[0]["start_ms"] == DAY0 and n3[0]["end_ms"] == DAY0 + 6 * H and n3[2]["start_ms"] == DAY0 + 13 * H
    assert n3[0]["summary"].startswith("Quiet consolidation") and "push toward $80,800" in n3[2]["summary"]
    assert qa.timeline_line(p3, DAY0 + 7 * H).startswith("Now ⚡ 06:00–13:00 UTC (11:00–18:00 PKT) normal")
    assert "Next 🎯 13:00–16:00 UTC" in qa.timeline_line(p3, DAY0 + 7 * H) and "1 of 4 windows passed" in qa.timeline_line(p3, DAY0 + 7 * H)
    assert qa.window_state(p3, DAY0 + 14 * H) == "13:00–16:00 UTC (18:00–21:00 PKT) expansion window"
    assert qf_tree.timeline_tree(qf.locked_plan(DAY0, DAY0 + qf.DAY_MS, 1.0, 2, None, None), DAY0) == []
    # a node's range is the window's own (the chart's box), not the day's cumulative band: 29 Oct the bot
    # read "chop $80,176–$82,912", a 3.4% "chop box", from the band at hour 16
    import math
    half = qf.Z84 * 0.004 * math.sqrt(6)
    assert (n3[0]["low"], n3[0]["high"]) == (round(80_000 * math.exp(-half), 2), round(80_000 * math.exp(half), 2))
    assert n3[0]["high"] - n3[0]["low"] < 1_600
    assert all(n["high"] - n["low"] < 0.05 * 80_000 for n in n3)
    # runs shorter than two hours fold into the node before them, so a lone normal hour between two chop
    # runs does not make three nodes of one
    p4 = _plan_fixed()
    for k, h in enumerate(p4["hours"]):
        h["phase"] = "chop" if k < 3 or 4 <= k < 8 else "normal"
    n4 = qf_tree.timeline_tree(p4, DAY0)
    assert [(n["phase"], n["start_ms"], n["end_ms"]) for n in n4] == [("chop", DAY0, DAY0 + 8 * H), ("normal", DAY0 + 8 * H, DAY0 + 24 * H)]
    p5 = _plan_fixed()
    p5["hours"][0]["phase"] = "expansion"                      # a lone first hour joins the node after it
    assert [n["phase"] for n in qf_tree.timeline_tree(p5, DAY0)] == ["normal"]

    d = qf_tree.trade_directive(p)
    assert d["direction"] == "LONG" and d["stop_loss_invalidation"] == 79_000.0
    assert d["entry_zone"] == [79_600.0, 80_000.0], "from the projected low to the median path at its hour"
    assert (d["tp1"], d["tp2"], d["tp3"]) == (80_200.0, 80_500.0, 80_800.0), "the levels between, then the projected high"
    assert d["setup_type"] == "Value Area Low retest" and d["active_start_ms"] == DAY0 + 3 * H and d["active_end_ms"] == DAY0 + 14 * H
    assert d["risk_reward"] == round((80_800 - 79_800) / (79_800 - 79_000), 1)
    p["levels"] = []
    d = qf_tree.trade_directive(p)
    assert (d["tp1"], d["tp2"], d["tp3"]) == (80_266.67, 80_533.33, 80_800.0) and d["setup_type"] == "Pullback into the projected low"
    bear = _plan_fixed(regime="bearish", above=81_000.0)
    bear["projected_high"], bear["projected_low"] = {"price": 80_400.0, "t_ms": DAY0 + 3 * H}, {"price": 79_200.0, "t_ms": DAY0 + 14 * H}
    d = qf_tree.trade_directive(bear)
    assert d["direction"] == "SHORT" and d["entry_zone"] == [80_000.0, 80_400.0] and d["tp3"] == 79_200.0 and d["tp1"] > d["tp2"] > d["tp3"]
    assert qf_tree.trade_directive({**p, "regime": "neutral"})["direction"] == "NONE"
    assert "Trade: LONG · entry $79,600–$80,000 · stop $79,000 · TP $80,267 / $80,533 / $80,800" == qa.directive_line(p)


def test_the_magnitude_scaler_widens_the_target_once_on_force_the_simulation_did_not_have():
    p = _plan_fixed()
    p["value_area"] = {"poc": 79_900.0, "high": 80_100.0, "low": 79_700.0}
    p["levels"] = [{"label": "call wall", "price": 81_200.0, "side": "above", "p_touch": 0.2},
                   {"label": "air pocket", "price": 81_500.0, "side": "above", "p_touch": 0.1},
                   {"label": "POC", "price": 81_000.0, "side": "above", "p_touch": 0.3}]
    quiet = pd.DataFrame({"timestamp": [DAY0 - (24 - k) * H for k in range(24)], "delta": [20.0, -20.0] * 12,
                          "oi_btc": [10_000.0 + k for k in range(24)]})
    h1 = candles([80_000, 80_050, 80_300], start=DAY0)                           # the 02:00 hour opens 80,050, closes 80,300: through VAH
    now = DAY0 + 3 * H                                                           # ...and has closed
    t2 = [DAY0 + H, DAY0 + 2 * H]
    # 1. a loud hour of taker flow the plan's way: 1.5x the usual hour's gross delta
    loud = pd.concat([quiet, pd.DataFrame({"timestamp": t2, "delta": [5.0, 40.0], "oi_btc": [10_024.0, 10_025.0]})])
    plan = {**p, "hours": [dict(h) for h in p["hours"]]}
    mg = qf.magnitude_scale(plan, h1, loud, now)
    assert mg and mg["to"] == 81_200.0 and mg["level"] == "call wall" and mg["from"] == 80_800.0 and mg["t_ms"] == now
    assert "2.0x the usual hour's" in mg["trigger"] and plan["projected_high"]["price"] == 81_200.0
    assert "81,200" in plan["scenario_a"] and plan["magnitude"] is mg
    assert qf.magnitude_scale(plan, h1, loud, now + H) is None, "once a day"
    # 2. no force: 1.4x is under the line; nothing moves
    mild = pd.concat([quiet, pd.DataFrame({"timestamp": t2, "delta": [5.0, 28.0], "oi_btc": [10_024.0, 10_025.0]})])
    plan = {**p, "hours": [dict(h) for h in p["hours"]]}
    assert qf.magnitude_scale(plan, h1, mild, now) is None and plan["projected_high"]["price"] == 80_800.0
    # 3. open interest surging on the hour that closed through the value area high
    jump = pd.concat([quiet, pd.DataFrame({"timestamp": t2, "delta": [5.0, 5.0], "oi_btc": [10_024.0, 10_399.0]})])
    plan = {**p, "hours": [dict(h) for h in p["hours"]]}
    mg = qf.magnitude_scale(plan, h1, jump, now)
    assert mg and mg["trigger"].startswith("open interest surged +375 BTC on the hourly close through the value area high $80,100")
    flat = candles([80_000, 80_050, 80_090], start=DAY0)                         # the same surge, no break of the VAH
    plan = {**p, "hours": [dict(h) for h in p["hours"]]}
    assert qf.magnitude_scale(plan, flat, jump, now) is None
    assert qf.magnitude_scale({**p, "hours": list(p["hours"])}, h1, jump, now - H) is None, "the break hour has not closed yet"
    # 4. no level beyond the target: nothing to widen to
    plan = {**p, "hours": [dict(h) for h in p["hours"]], "levels": []}
    assert qf.magnitude_scale(plan, h1, loud, now) is None
    assert qf.magnitude_scale({**p, "regime": "neutral"}, h1, loud, now) is None
    # the write-up is dropped when its numbers no longer hold, and the panel says what moved
    from ui import panels
    plan = {**p, "hours": [dict(h) for h in p["hours"]], "writeup": {"summary": "old words", "model": "x"}}
    qf.magnitude_scale(plan, h1, loud, now)
    assert "rejected" in plan["writeup"] and "Target widened $80,800 → $81,200 (call wall)" in panels.qf_wire_html(plan)


def test_the_tier_messages_use_hammads_words():
    p = _plan_fixed()
    c = {"t_ms": DAY0 + H, "why": "move", "sigmas": 2.5, "from": 80_000.0, "to": 79_500.0}
    text = qa.caution_headline(p, c)
    assert text.startswith("⚠️ CAUTION [00:00–00:00 UTC (05:00–05:00 PKT) normal window]: Order flow metrics are moving "
                           "differently than baseline projection. Volatility spike detected: a 15-minute candle moved 2.5x")
    assert "Whole-Day Invalidation at below $79,000 has NOT been breached. Plan remains active.\nNow ⚡" in text
    assert "CVD divergence detected" in qa.caution_headline(p, {**c, "why": "flow", "delta_btc": -300.0})
    held = qa.held_headline(p, {"t_ms": DAY0 + H, "shock_ms": 0, "from": 80_000.0, "to": 79_500.0})
    assert held.startswith("🟢 MODEL HELD: Intraday shock to $79,500 absorbed successfully above whole-day invalidation "
                           "boundary $79,000. Baseline 1-day forecast path remains active.\nNow ⚡")
    bear = _plan_fixed(regime="bearish", above=81_000.0)
    assert "below whole-day invalidation boundary $81,000" in qa.held_headline(bear, {"t_ms": DAY0 + H, "shock_ms": 0, "from": 1.0, "to": 2.0})
    hit = {"side": "below", "level": 79_000.0, "close": 78_900.0, "close_ms": DAY0 + 4 * H, "basis": "4-hour close"}
    assert qa.invalidated_headline(p, hit, None).startswith(
        "🔴 WHOLE-DAY PLAN INVALIDATED: Macro structure breached at $78,900 (4-hour close, below the $79,000 line, "
        "04:00 UTC (09:00 PKT)). Original roadmap voided. Scenario B secondary chart activated with new targets.")
    b = {**_plan_fixed(created=DAY0 + 4 * H), "version": 1}
    b["scenario_a"], b["scenario_b"] = qf.scenarios(b)
    assert "SCENARIO B · FORECAST #1" in qa.invalidated_headline(p, hit, b)
    locked = qf.locked_plan(DAY0 + 4 * H, DAY0 + qf.DAY_MS, 78_900.0, 2, 80_100.0, 78_800.0)
    assert "LOCKOUT MODE: HIGH CHOP / STAND ASIDE" in qa.invalidated_headline(p, hit, locked)
