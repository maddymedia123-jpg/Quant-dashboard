"""Pydantic output contracts for every agent, plus the assembled TrapReport."""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

DESKS = ("bullish", "bearish")
TRAPS = ("BULL_TRAP", "BEAR_TRAP", "NO_TRAP", "RANGE_TRAP")
CATEGORY_KEYS = ("live", "intraday", "weekly", "monthly")


def _num(v):
    """Coerce '7', '7.5', '60%' → float; leave numbers alone."""
    if isinstance(v, str):
        m = re.search(r"-?\d+(?:\.\d+)?", v)
        if not m:
            raise ValueError(f"not a number: {v!r}")
        return float(m.group(0))
    return v


class SpecialistBrief(BaseModel):
    agent_id: str
    desk: str
    domain: str
    raw_metrics: dict[str, Any] = Field(default_factory=dict)
    interpretation: str
    traps_detected: list[str] = Field(default_factory=list)
    conviction: float = Field(ge=1, le=10)
    key_risk_to_thesis: str
    data_gaps: list[str] = Field(default_factory=list)

    @field_validator("conviction", mode="before")
    @classmethod
    def _c(cls, v):
        return _num(v)


class TradeSetup(BaseModel):
    direction: Literal["LONG", "SHORT", "NONE"]
    entry: str
    stop: str
    targets: list[str] = Field(default_factory=list)
    rationale: str


class DeskReport(BaseModel):
    desk: str
    summary: str
    convergence: list[str] = Field(default_factory=list)
    divergence: list[str] = Field(default_factory=list)
    conviction: float = Field(ge=1, le=10)
    best_setup: TradeSetup
    acknowledged_risks: list[str] = Field(default_factory=list)

    @field_validator("conviction", mode="before")
    @classmethod
    def _c(cls, v):
        return _num(v)


class VolatilityReport(BaseModel):
    regime_summary: str
    per_category: dict[str, str] = Field(default_factory=dict)
    range_expectations: list[str] = Field(default_factory=list)


class AccuracyReport(BaseModel):
    consistency_score: float = Field(ge=0, le=100)
    cross_timeframe_alignment: str
    contradictions_found: list[str] = Field(default_factory=list)
    correlation_notes: str
    data_quality_flags: list[str] = Field(default_factory=list)

    @field_validator("consistency_score", mode="before")
    @classmethod
    def _c(cls, v):
        return _num(v)


class Scenario(BaseModel):
    name: str
    probability: float = Field(ge=0, le=100)
    path: str
    invalidation: str

    @field_validator("probability", mode="before")
    @classmethod
    def _p(cls, v):
        return _num(v)


class TradePlan(BaseModel):
    existing_position_management: str
    new_entry_conditions: str
    position_sizing: str
    stop_loss: str
    take_profit_ladder: list[str] = Field(default_factory=list)
    max_leverage: str
    time_rules: list[str] = Field(default_factory=list)


class Contradiction(BaseModel):
    metric: str
    bull_read: str
    bear_read: str
    stronger_evidence: str


class DirectorReport(BaseModel):
    executive_summary: str
    trap_classification: Literal["BULL_TRAP", "BEAR_TRAP", "NO_TRAP", "RANGE_TRAP"]
    classification_rationale: str
    contradictions: list[Contradiction] = Field(default_factory=list)
    convergence: list[str] = Field(default_factory=list)
    blind_spots: list[str] = Field(default_factory=list)
    scenarios: list[Scenario]
    trade_plan: TradePlan
    category_summaries: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalise(self):
        if not 2 <= len(self.scenarios) <= 4:
            raise ValueError("2-4 scenarios required")
        total = sum(s.probability for s in self.scenarios)
        if not 90 <= total <= 110:
            raise ValueError(f"scenario probabilities sum to {total:.0f}, expected about 100")
        if abs(total - 100) > 1e-9:
            scale = 100.0 / total
            acc = 0.0
            for s in self.scenarios[:-1]:
                s.probability = round(s.probability * scale)
                acc += s.probability
            self.scenarios[-1].probability = round(100.0 - acc)
        self.category_summaries = {k: v for k, v in self.category_summaries.items() if k in CATEGORY_KEYS}
        return self


class AgentRun(BaseModel):
    agent_id: str
    ok: bool
    model: str
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: Optional[float] = None
    error: Optional[str] = None


class TrapReport(BaseModel):
    report_id: str
    generated_at: datetime
    asset: str = "BTC/USD"
    window: str
    provider: str
    briefs: list[SpecialistBrief] = Field(default_factory=list)
    desks: dict[str, DeskReport] = Field(default_factory=dict)
    volatility: Optional[VolatilityReport] = None
    accuracy: Optional[AccuracyReport] = None
    director: Optional[DirectorReport] = None
    runs: list[AgentRun] = Field(default_factory=list)
    wall_time_s: float = 0.0
    unavailable_domains: list[str] = Field(default_factory=list)
    raw_metrics: list[tuple[str, str, str]] = Field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(r.prompt_tokens + r.completion_tokens for r in self.runs)

    @property
    def total_cost_usd(self) -> Optional[float]:
        costs = [r.cost_usd for r in self.runs if r.cost_usd is not None]
        return sum(costs) if costs else None

    @property
    def failed_agents(self) -> list[str]:
        return [r.agent_id for r in self.runs if not r.ok]


def _strip_titles(node):
    if isinstance(node, dict):
        return {k: _strip_titles(v) for k, v in node.items() if k != "title"}
    if isinstance(node, list):
        return [_strip_titles(x) for x in node]
    return node


def schema_hint(model_cls: type[BaseModel]) -> str:
    """Compact JSON schema (titles stripped) for embedding in prompts."""
    return json.dumps(_strip_titles(model_cls.model_json_schema()), separators=(",", ":"))
