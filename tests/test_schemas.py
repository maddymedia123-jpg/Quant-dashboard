from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from core.agents.schemas import (
    AgentRun, DirectorReport, Scenario, SpecialistBrief, TradePlan, TrapReport, schema_hint,
)

_PLAN = TradePlan(existing_position_management="hold", new_entry_conditions="x", position_sizing="1%", stop_loss="s",
                  take_profit_ladder=["a"], max_leverage="3x", time_rules=["flatten before FOMC"])


def _director(probs, extra_cat=None):
    cats = {"live": "l", "weekly": "w"}
    if extra_cat:
        cats[extra_cat] = "junk"
    return DirectorReport(
        executive_summary="e", trap_classification="RANGE_TRAP", classification_rationale="r",
        scenarios=[Scenario(name=f"s{i}", probability=p, path="p", invalidation="i") for i, p in enumerate(probs)],
        trade_plan=_PLAN, category_summaries=cats,
    )


def test_probabilities_normalise_within_tolerance():
    d = _director([60, 30, 20])  # sums to 110 → normalised to 100
    assert sum(s.probability for s in d.scenarios) == 100
    assert d.scenarios[0].probability == 55


def test_probabilities_far_off_are_rejected():
    with pytest.raises(ValidationError):
        _director([80, 70])


def test_percent_strings_are_coerced():
    s = Scenario(name="a", probability="60%", path="p", invalidation="i")
    assert s.probability == 60.0
    b = SpecialistBrief(agent_id="B1", desk="bullish", domain="options", interpretation="x", conviction="7/10", key_risk_to_thesis="y")
    assert b.conviction == 7.0


def test_conviction_bounds_and_category_filter():
    with pytest.raises(ValidationError):
        SpecialistBrief(agent_id="B1", desk="bullish", domain="options", interpretation="x", conviction=11, key_risk_to_thesis="y")
    d = _director([50, 50], extra_cat="hourly")
    assert set(d.category_summaries) == {"live", "weekly"}


def test_trap_report_totals_and_hint():
    r = TrapReport(report_id="TIR-1", generated_at=datetime.now(timezone.utc), window="w", provider="gemini",
                   runs=[AgentRun(agent_id="B1", ok=True, model="m", latency_s=1.0, prompt_tokens=10, completion_tokens=5),
                         AgentRun(agent_id="B2", ok=False, model="m", latency_s=1.0, error="boom")])
    assert r.total_tokens == 15 and r.total_cost_usd is None and r.failed_agents == ["B2"]
    hint = schema_hint(SpecialistBrief)
    assert '"conviction"' in hint and '"title"' not in hint
