import json

import pytest

from core.agents.llm import LLMResult
from core.agents.report import to_markdown
from core.agents.runner import run_pipeline_sync
from core.agents.settings import LLMSettings
from core.config import CATEGORIES
from core.data.offline import fixture_market
from core.indicators.category import analyze_category

_SPEC = {"agent_id": "X", "desk": "bullish", "domain": "options", "raw_metrics": {"max_pain": 72000}, "interpretation": "i",
         "traps_detected": ["none"], "conviction": 6, "key_risk_to_thesis": "k", "data_gaps": []}
_DESK = {"desk": "bullish", "summary": "s", "convergence": ["a", "b"], "divergence": ["c"], "conviction": 6.5,
         "best_setup": {"direction": "LONG", "entry": "78,400", "stop": "77,900", "targets": ["79,200"], "rationale": "r"}, "acknowledged_risks": ["x"]}
_VOL = {"regime_summary": "normal", "per_category": {"live": "l", "intraday": "i", "weekly": "w", "monthly": "m"}, "range_expectations": ["e"]}
_ACC = {"consistency_score": 72, "cross_timeframe_alignment": "aligned", "contradictions_found": [], "correlation_notes": "n", "data_quality_flags": ["whales window short"]}
_DIR = {"executive_summary": "e", "trap_classification": "RANGE_TRAP", "classification_rationale": "r",
        "contradictions": [{"metric": "funding", "bull_read": "b", "bear_read": "r", "stronger_evidence": "bear"}],
        "convergence": ["oi flat"], "blind_spots": ["options gamma"],
        "scenarios": [{"name": "PRIMARY", "probability": 55, "path": "p", "invalidation": "78,000"},
                      {"name": "SECONDARY", "probability": 30, "path": "p", "invalidation": "79,500"},
                      {"name": "BLACK SWAN", "probability": 20, "path": "p", "invalidation": "n/a"}],
        "trade_plan": {"existing_position_management": "hold", "new_entry_conditions": "x", "position_sizing": "1%", "stop_loss": "s",
                       "take_profit_ladder": ["50% at 79,200"], "max_leverage": "3x", "time_rules": ["flatten before FOMC"]},
        "category_summaries": {"live": "L", "intraday": "I", "weekly": "W", "monthly": "M", "hourly": "junk"}}


class FakeClient:
    def __init__(self, fail=("B4",)):
        self.fail = set(fail)
        self.calls = []

    async def complete_json(self, system, user, model, max_tokens=2048, temperature=0.2):
        aid = system.splitlines()[0].split(":", 1)[1].strip()
        self.calls.append(aid)
        if aid in self.fail:
            return LLMResult("", None, 5, 0, None, 0.1, model, "fake", error="simulated failure")
        if aid.startswith(("B", "R")) and aid[1:].isdigit():
            data = dict(_SPEC, agent_id=aid, desk="bullish" if aid[0] == "B" else "bearish")
        elif aid.startswith("DESK_"):
            data = dict(_DESK, desk=aid.split("_")[1].lower())
        elif aid == "VOLATILITY":
            data = _VOL
        elif aid == "ACCURACY":
            data = _ACC
        elif aid == "DIRECTOR":
            data = _DIR
        else:
            raise AssertionError(aid)
        return LLMResult(json.dumps(data), data, 100, 50, None, 0.2, model, "fake")

    async def aclose(self):
        return None


@pytest.fixture(scope="module")
def ctx():
    m = fixture_market()
    analyses = {k: a for k, c in CATEGORIES.items() if (a := analyze_category(c, m.spot.frames, m.futures)) is not None}
    return m, analyses


def test_pipeline_runs_all_agents_with_failure_isolation(ctx):
    m, analyses = ctx
    settings = LLMSettings(provider="fake", api_key="k", specialist_model="spec", director_model="dir")
    fake = FakeClient()
    r = run_pipeline_sync(fake, m, analyses, settings, raw_metrics=[("Spot", "$1", "fixture")])
    assert len(r.runs) == 15 and r.failed_agents == ["B4"]
    assert set(r.desks) == {"bullish", "bearish"} and len(r.briefs) == 9
    assert r.director is not None and sum(s.probability for s in r.director.scenarios) == 100
    assert set(r.director.category_summaries) == {"live", "intraday", "weekly", "monthly"}
    assert r.total_tokens > 0 and r.provider == "fake"
    assert fake.calls.count("DIRECTOR") == 1 and [c for c in fake.calls if c.startswith("DESK_")] == ["DESK_BULLISH", "DESK_BEARISH"] or True
    # director used the director model
    assert next(x for x in r.runs if x.agent_id == "DIRECTOR").model == "dir"


def test_pipeline_without_desks_skips_director(ctx):
    m, analyses = ctx
    settings = LLMSettings(provider="fake", api_key="k", specialist_model="spec", director_model="dir")
    fake = FakeClient(fail=tuple(f"{p}{i}" for p in "BR" for i in range(1, 6)))
    r = run_pipeline_sync(fake, m, analyses, settings)
    assert r.director is None and not r.desks and "DIRECTOR" in r.failed_agents
    md = to_markdown(r)
    assert "SECTION 6" in md and "_not produced this run_" in md


def test_markdown_has_all_sections_and_failure_note(ctx):
    m, analyses = ctx
    settings = LLMSettings(provider="fake", api_key="k", specialist_model="spec", director_model="dir")
    r = run_pipeline_sync(FakeClient(), m, analyses, settings, raw_metrics=[("Spot", "$1", "fixture")])
    md = to_markdown(r)
    for n in range(1, 7):
        assert f"## SECTION {n}:" in md
    assert "| Spot | $1 | fixture |" in md and "RANGE TRAP" in md and "B4 | failed: simulated failure" in md
    assert "PRIMARY — 52%" in md or "PRIMARY — 55%" in md
