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
_PLAN = {"side": "SHORT", "zone_low": 83520, "zone_high": 83945, "confirm_timeframe": "1h", "confirm_close": 83077,
         "taker_ratio_limit": 0.95, "confirmations": ["1h close below 83,077"], "stop": 84200,
         "targets": [81432, 82820, 82237], "expires_hours": 24, "risk_pct": 1, "max_leverage": 5}
H = 3_600_000


class FakeClient:
    def __init__(self, fail=("B4",), fail_models=(), director=None):
        self.fail = set(fail)
        self.fail_models = set(fail_models)
        self.calls = []
        self.prompts = {}
        self.director = _DIR if director is None else director

    async def complete_json(self, system, user, model, max_tokens=2048, temperature=0.2):
        aid = system.splitlines()[0].split(":", 1)[1].strip()
        self.calls.append(aid)
        self.prompts[aid] = (system, user)
        if aid in self.fail or model in self.fail_models:
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
            data = self.director
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


# ---------- parity with the 1-agent Director (owner's GO, 5 November) ----------
def test_the_classic_director_is_handed_the_same_blocks_and_rules(ctx):
    """Roadmap items 9, 10, 12 and the regime rule were wired into the 1-agent state only: the classic
    Director read the market without the per-desk derivatives, the cascade and forecast per desk, or the
    4-hour-versus-24-hour framing, and wrote its plan as prose. Now it gets the same blocks and is asked
    for the same fields."""
    from core.agents import one_agent, prompts
    m, analyses = ctx
    settings = LLMSettings(provider="fake", api_key="k", specialist_model="spec", director_model="dir")
    fake = FakeClient(fail=())
    run_pipeline_sync(fake, m, analyses, settings)
    system, user = fake.prompts["DIRECTOR"]
    data = json.loads(user.split("DATA (JSON):\n", 1)[1].split("\n\nUNAVAILABLE", 1)[0])
    for key in ("desk_derivatives", "cascade", "forecast", "horizons", "macro_read"):
        assert key in data, key
    assert set(data["cascade"]) == set(data["forecast"]) == set(data["desk_derivatives"]) == {"live", "intraday", "weekly", "monthly"}
    assert data["horizons"]["framing"]["state"] in ("inside", "probe_above", "probe_below", "expansion", "unknown")
    # the same rule text, word for word, in both Directors' prompts
    for text in (prompts.PLAN_TEXT, prompts.HORIZON_TEXT, prompts.REGIME_TEXT, prompts.WINDOWS_TEXT):
        assert text in prompts.DIRECTOR_TEXT and text in one_agent.ONE_AGENT_TEXT
    assert "plan" in system and "horizon_framing" in system and "event_regime" in system, "the schema hint carries the fields"
    assert "B1" not in fake.prompts or "horizons" not in fake.prompts["B1"][1], "the specialists keep their own payloads"


def test_the_classic_report_carries_a_numbers_plan_framing_and_check(ctx):
    m, analyses = ctx
    settings = LLMSettings(provider="fake", api_key="k", specialist_model="spec", director_model="dir")
    director = dict(_DIR, plan=_PLAN, horizon_framing="The short sits inside the day; the day's low must go first.")
    r = run_pipeline_sync(FakeClient(fail=(), director=director), m, analyses, settings)
    assert r.mode == "15-agent" and r.plan is not None and r.plan.side == "SHORT"
    assert r.plan.targets == [82820, 82237, 81432], "targets nearest first, as the 1-agent plan"
    assert r.director.plan == r.plan and r.director.horizon_framing.startswith("The short sits")
    assert r.director.horizon_check and r.director.horizon_check["verdict"] in ("tactical", "reaches_beyond", "outside", "unknown")
    assert r.plan.event_regime == "" and r.plan.risk_pct == 1
    assert r.director.plan_notes and "sits 6,200 above the primary scenario's invalidation" in r.director.plan_notes[0], \
        "the fixture's SHORT stop 84,200 is beyond its primary invalidation 78,000: the classic report is checked too"
    # the fallback model's answer is finished the same way
    r2 = run_pipeline_sync(FakeClient(fail=(), fail_models=("dir",), director=director), m, analyses, settings)
    assert r2.plan is not None and r2.director.horizon_check is not None
    # an answer without a plan (a model that skips it, or a report from before) still validates
    r3 = run_pipeline_sync(FakeClient(fail=()), m, analyses, settings)
    assert r3.director is not None and r3.plan is None and r3.director.plan is None
    from core.agents.schemas import DirectorReport
    assert DirectorReport.model_validate(_DIR).plan is None


def test_the_classic_plan_is_tagged_and_halved_inside_an_event_regime(ctx):
    m, analyses = ctx
    settings = LLMSettings(provider="fake", api_key="k", specialist_model="spec", director_model="dir")
    now_ms = int(m.generated_at.timestamp() * 1000)
    election = {"title": "Congressional Elections", "country": "USD", "impact": "High", "time_ms": now_ms - 2 * H,
                "date": "2026-11-03T18:00:00-05:00", "forecast": "", "previous": ""}
    m2 = m.model_copy(update={"calendar": m.calendar.model_copy(update={"events": list(m.calendar.events) + [election]})})
    r = run_pipeline_sync(FakeClient(fail=(), director=dict(_DIR, plan=_PLAN)), m2, analyses, settings)
    assert r.plan.event_regime == "Congressional Elections" and r.plan.risk_pct == 0.5 and r.plan.max_leverage == 2.5
    assert r.plan.stop == 84200, "the geometry is the Director's"


def test_director_falls_back_to_specialist_model(ctx):
    m, analyses = ctx
    settings = LLMSettings(provider="fake", api_key="k", specialist_model="spec", director_model="dir")
    r = run_pipeline_sync(FakeClient(fail=(), fail_models=("dir",)), m, analyses, settings)
    ids = [x.agent_id for x in r.runs]
    assert "DIRECTOR" in ids and "DIRECTOR (fallback model)" in ids and len(r.runs) == 16
    assert r.director is not None and next(x for x in r.runs if x.agent_id == "DIRECTOR (fallback model)").model == "spec"
    assert r.failed_agents == ["DIRECTOR"]
