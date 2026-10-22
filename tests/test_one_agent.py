"""The 1-agent Director: one call, the whole state, the same report, and the plan as numbers - plus the
store, the plan scoring and the side-by-side view that let it be judged against the 15-agent pipeline."""
import copy
import json
import os
import tempfile

import pandas as pd
import pytest

os.environ["TI_OFFLINE_FIXTURES"] = "1"
os.environ.setdefault("TI_DATA_DIR", tempfile.mkdtemp())

from core import accuracy  # noqa: E402
from core.agents import one_agent  # noqa: E402
from core.agents.llm import LLMResult  # noqa: E402
from core.agents.report import to_markdown  # noqa: E402
from core.agents.schemas import PlanLevels  # noqa: E402
from core.config import CATEGORIES  # noqa: E402
from core.data.offline import fixture_market  # noqa: E402
from core.indicators.category import analyze_category  # noqa: E402
from core.store import Store  # noqa: E402

_CASE = {"desk": "x", "summary": "s", "convergence": ["a"], "divergence": ["b"], "conviction": 7,
         "best_setup": {"direction": "SHORT", "entry": "83,500", "stop": "84,200", "targets": ["82,800"], "rationale": "r"},
         "acknowledged_risks": ["k"]}
PLAN = {"side": "SHORT", "zone_low": 83520, "zone_high": 83945, "confirm_timeframe": "1h", "confirm_close": 83077,
        "taker_ratio_limit": 0.95, "confirmations": ["1h close below 83,077"], "stop": 84200,
        "targets": [81432, 82820, 82237], "expires_hours": 24, "risk_pct": 1, "max_leverage": 5}
OUT = {"bull_case": dict(_CASE, conviction=4), "bear_case": dict(_CASE, conviction=8), "executive_summary": "e",
       "trap_classification": "BULL_TRAP", "classification_rationale": "r",
       "contradictions": [{"metric": "CVD", "bull_read": "b", "bear_read": "r", "stronger_evidence": "bear"}],
       "convergence": ["crowded longs"], "blind_spots": ["no realized price"],
       "scenarios": [{"name": "Bull trap", "probability": 55, "path": "p", "invalidation": "84,200"},
                     {"name": "Range", "probability": 30, "path": "p", "invalidation": "x"},
                     {"name": "Breakout", "probability": 15, "path": "p", "invalidation": "y"}],
       "trade_plan": {"existing_position_management": "trim", "new_entry_conditions": "short a rejection",
                      "position_sizing": "1%", "stop_loss": "84,200", "take_profit_ladder": ["82,820"],
                      "max_leverage": "5x", "time_rules": []},
       "plan": PLAN, "category_summaries": {"live": "L", "intraday": "I", "weekly": "W", "monthly": "M"}}


class Fake:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    async def complete_json(self, system, user, model, max_tokens=2048, temperature=0.2):
        self.calls.append((system, user, model, max_tokens))
        data = self.answers.pop(0)
        return LLMResult(json.dumps(data) if data else "", data, 9000, 3000, None, 1.0, model, "fake",
                         error=None if data else "busy")

    async def aclose(self):
        return None


@pytest.fixture(scope="module")
def ctx():
    m = fixture_market()
    return m, {k: analyze_category(c, m.spot.frames, m.futures) for k, c in CATEGORIES.items()}


# ---------- the call ----------
def test_one_call_reads_everything_and_returns_the_whole_report(ctx):
    m, an = ctx
    fake = Fake([OUT])
    r = one_agent.run_one_agent_sync(fake, m, an, "gemini-3.8-flash", [("Price", "$1", "x")], "S-1")
    assert len(fake.calls) == 1 and len(r.runs) == 1 and r.runs[0].ok and r.runs[0].model == "gemini-3.8-flash"
    assert r.mode == "1-agent" and r.snapshot_id == "S-1" and r.report_id.startswith("TIR1-")
    assert r.director.trap_classification == "BULL_TRAP" and set(r.desks) == {"bullish", "bearish"}
    assert r.desks["bearish"].desk == "bearish" and r.desks["bearish"].conviction == 8
    assert r.plan.side == "SHORT" and r.plan.targets == [82820, 82237, 81432], "targets nearest first"
    system, user, _model, max_tokens = fake.calls[0]
    assert system.startswith("AGENT_ID: DIRECTOR_1") and max_tokens == one_agent.MAX_TOKENS
    data = json.loads(user.split("DATA (JSON):\n", 1)[1].split("\n\nUNAVAILABLE", 1)[0])
    for key in ("categories", "derivatives", "macro_read", "cascade", "forecast", "options_expiries"):
        assert key in data, key
    assert set(data["cascade"]) == set(data["forecast"]) == {"live", "intraday", "weekly", "monthly"}
    assert "cvd_5bars" in data["categories"]["live"]


def test_a_broken_answer_gets_one_correction_round(ctx):
    m, an = ctx
    bad = copy.deepcopy(OUT)
    bad["plan"]["stop"] = 83000                                  # a SHORT stop below its zone
    fake = Fake([bad, OUT])
    r = one_agent.run_one_agent_sync(fake, m, an, "gemini-3.8-flash")
    assert len(fake.calls) == 2 and r.runs[0].ok
    assert "must be above the zone high" in fake.calls[1][1], "the validation error goes back to the model"
    assert r.runs[0].prompt_tokens == 18000, "both rounds are counted"


def test_still_broken_after_the_retry_is_a_failed_report_not_a_half_one(ctx):
    m, an = ctx
    bad = copy.deepcopy(OUT)
    bad["scenarios"] = bad["scenarios"][:1]
    r = one_agent.run_one_agent_sync(Fake([bad, bad]), m, an, "gemini-3.8-flash")
    assert r.director is None and r.plan is None and not r.desks and not r.runs[0].ok
    assert "scenarios" in r.runs[0].error


def test_no_answer_is_reported_as_failed(ctx):
    m, an = ctx
    r = one_agent.run_one_agent_sync(Fake([None]), m, an, "gemini-3.8-flash")
    assert r.director is None and r.runs[0].error == "busy"


# ---------- the plan as numbers ----------
def test_plan_geometry_is_enforced():
    with pytest.raises(ValueError, match="above the zone high"):
        PlanLevels(**dict(PLAN, stop=83600))
    with pytest.raises(ValueError, match="below the zone low"):
        PlanLevels(**dict(PLAN, targets=[84000]))
    with pytest.raises(ValueError, match="needs zone_low"):
        PlanLevels(side="LONG", stop=1.0)
    long = PlanLevels(side="LONG", zone_low=80000, zone_high=80500, stop=79500, targets=[82000, 81000])
    assert long.targets == [81000, 82000]
    with pytest.raises(ValueError, match="below the zone low"):
        PlanLevels(side="LONG", zone_low=80000, zone_high=80500, stop=80200, targets=[81000])
    assert PlanLevels(side="NONE", why_none="no edge").why_none == "no edge"
    swapped = PlanLevels(**dict(PLAN, zone_low=83945, zone_high=83520))
    assert (swapped.zone_low, swapped.zone_high) == (83520, 83945)


def test_numbers_written_as_text_are_read():
    p = PlanLevels(**dict(PLAN, zone_low="83,520", stop="$84,200", targets=["82,820", "81432"]))
    assert p.zone_low == 83520 and p.stop == 84200


# ---------- storage, scoring, compare ----------
H = 3_600_000
T0 = 1_790_000_000_000 - (1_790_000_000_000 % H)


def candles(path, step=900_000):
    """15m rows from (high, low, close) tuples starting at T0."""
    return pd.DataFrame([{"timestamp": T0 + i * step, "open": c, "high": h, "low": lo, "close": c}
                         for i, (h, lo, c) in enumerate(path)])


def hourly(fine):
    g = fine.assign(b=(fine["timestamp"] // H) * H).groupby("b")
    return pd.DataFrame({"timestamp": g["timestamp"].first().index, "open": g["open"].first().values,
                         "high": g["high"].max().values, "low": g["low"].min().values, "close": g["close"].last().values})


def score(path, plan=PLAN, now_offset=48 * H):
    fine = candles(path)
    return accuracy.plan_outcome(plan, T0, {"15m": fine, "1h": hourly(fine)}, T0 + now_offset)


def test_a_short_that_rejects_and_falls_is_a_target():
    up = [(83600, 83400, 83550)] * 4                    # hour 0: into the zone
    reject = [(83500, 83000, 83050)] * 4                # hour 1: closes below 83,077 - confirmed
    fall = [(83000, 82700, 82750)] * 4                  # hour 2: first target 82,820
    assert score(up + reject + fall) == "target"


def test_a_short_that_confirms_then_rips_is_a_stop():
    assert score([(83600, 83400, 83550)] * 4 + [(83500, 83000, 83050)] * 4 + [(84300, 83100, 84250)] * 4) == "stop"


def test_a_zone_never_reached_expires_untriggered():
    assert score([(83200, 83000, 83100)] * 8, now_offset=30 * H) == "not_triggered"
    assert score([(83200, 83000, 83100)] * 8, now_offset=3 * H) == "open", "not expired yet"


def test_a_touch_without_a_confirming_close_is_not_a_trade():
    assert score([(83600, 83400, 83550)] * 8 + [(83700, 83300, 83600)] * 100, now_offset=30 * H) == "not_triggered"


def test_no_trade_plans_are_scored_as_such():
    assert accuracy.plan_outcome({"side": "NONE"}, T0, {}, T0) == "no_trade"


def test_reports_keep_their_mode_and_plan_and_the_scorecard_splits_them(ctx, tmp_path):
    m, an = ctx
    s = Store(tmp_path / "s.sqlite")
    r1 = one_agent.run_one_agent_sync(Fake([OUT]), m, an, "x", snapshot_id="S-9")
    assert accuracy.record_report(s, r1, 83_000.0, 1.5, "BULLISH", T0)
    row = s.reports()[0]
    assert row["mode"] == "1-agent" and row["snapshot_id"] == "S-9" and json.loads(row["plan"])["side"] == "SHORT"
    fine = candles([(83600, 83400, 83550)] * 4 + [(83500, 83000, 83050)] * 4 + [(83000, 82700, 82750)] * 4)
    s.add_report("OLD-1", T0, "NO_TRAP", 83_000.0, 1.5, "BULLISH")         # a 15-agent report: no plan
    assert accuracy.score_plans(s, {"15m": fine, "1h": hourly(fine)}, T0 + 48 * H) == 1
    cards = accuracy.mode_scorecard(s)
    assert cards["1-agent"]["plans"]["target"] == 1 and cards["15-agent"]["reports"] == 1
    assert not cards["15-agent"]["plans"]
    s.close()


def test_an_old_reports_table_gains_the_new_columns(tmp_path):
    import sqlite3
    path = tmp_path / "old.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE reports (id INTEGER PRIMARY KEY AUTOINCREMENT, report_id TEXT UNIQUE NOT NULL, "
                "ts_ms INTEGER NOT NULL, classification TEXT NOT NULL, price REAL NOT NULL, sigma24_pct REAL, "
                "majority_dir TEXT, move_24h_pct REAL, hit_24h INTEGER, move_7d_pct REAL, hit_7d INTEGER)")
    con.execute("INSERT INTO reports (report_id, ts_ms, classification, price) VALUES ('A', 1, 'NO_TRAP', 1.0)")
    con.commit()
    con.close()
    s = Store(path)
    assert s.reports()[0]["mode"] == "15-agent", "old reports count as 15-agent"
    s.close()


def test_compare_view_and_markdown(ctx):
    from ui import panels
    m, an = ctx
    r1 = one_agent.run_one_agent_sync(Fake([OUT]), m, an, "x", snapshot_id="S-1")
    r15 = r1.model_copy(update={"mode": "15-agent", "plan": None, "report_id": "TIR-X"})
    h = panels.compare_html(r1, r15)
    assert "1-agent" in h and "15-agent" in h and "prose only" in h and "SHORT $83,520–$83,945" in h
    assert "same market snapshot" in h
    other = r15.model_copy(update={"snapshot_id": "S-2"})
    assert "not like for like" in panels.compare_html(r1, other)
    assert "watched by the alerts" in panels.plan_html(r1.plan) and "Bear case" in panels.cases_html(r1)
    md = to_markdown(r1)
    assert "## Pipeline: 1-agent · snapshot S-1" in md and "Plan as numbers" in md and "83,520 – 83,945" in md


def test_a_candle_that_spans_stop_and_target_counts_as_the_stop():
    """15m data cannot say which came first inside one candle: the cautious reading is the stop."""
    up, reject = [(83600, 83400, 83550)] * 4, [(83500, 83000, 83050)] * 4
    assert score(up + reject + [(84300, 82700, 83000)] * 4) == "stop"


def test_the_one_call_waits_longer_than_the_specialists():
    from core.agents.settings import LLMSettings
    s = LLMSettings(provider="gemini", api_key="k", specialist_model="a", director_model="b")
    c = one_agent.client_for(s)
    assert c.s.timeout_s == one_agent.TIMEOUT_S and s.timeout_s == 90.0, "the shared settings are not changed"


def test_the_1_agent_appendix_does_not_read_as_a_failure(ctx):
    m, an = ctx
    md = to_markdown(one_agent.run_one_agent_sync(Fake([OUT]), m, an, "x"))
    appendix = md.split("## APPENDIX")[1]
    assert "not produced this run" not in appendix and appendix.count("Not part of the 1-agent report") == 2


def test_the_director_reads_each_desk_over_its_own_window(ctx):
    """His v15 Section 2: a weekly read never leans on a 24-hour open interest change."""
    m, analyses = ctx[0], ctx[1]
    state = one_agent.director_state(m, analyses)
    windows = {k: v["window"] for k, v in state["desk_derivatives"].items()}
    assert windows == {"live": "the next 4 hours", "intraday": "the next 24 hours",
                       "weekly": "the next 7 days", "monthly": "the next 30 days"}
    assert state["desk_derivatives"]["weekly"]["liquidations"]["over"] == "7d"
    assert "desk_derivatives[desk]" in one_agent.ONE_AGENT_TEXT
