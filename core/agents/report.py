"""Render a TrapReport as the client's markdown Trap Intelligence Report (Sections 1-6 + appendix)."""
from __future__ import annotations

from core.agents.schemas import DeskReport, TrapReport

MISSING = "_not produced this run_"


def _ul(items: list[str]) -> str:
    return "\n".join(f"- {x}" for x in items) if items else MISSING


def _ol(items: list[str]) -> str:
    return "\n".join(f"{i}. {x}" for i, x in enumerate(items, 1)) if items else MISSING


def _desk_section(n: int, title: str, d: DeskReport | None) -> str:
    if d is None:
        return f"## SECTION {n}: {title}\n{MISSING}\n"
    setup = d.best_setup
    trade = (f"**{setup.direction}** · entry: {setup.entry} · stop: {setup.stop} · targets: {', '.join(setup.targets) or '—'}\n\n{setup.rationale}")
    return (f"## SECTION {n}: {title}\n"
            f"### Conviction Score: {d.conviction:.0f}/10\n"
            f"{d.summary}\n\n"
            f"### Key Arguments\n{_ol(d.convergence)}\n\n"
            f"### Internal Divergence\n{_ul(d.divergence)}\n\n"
            f"### Proposed Trade\n{trade}\n\n"
            f"### Acknowledged Risks\n{_ul(d.acknowledged_risks)}\n")


def to_markdown(r: TrapReport) -> str:
    parts = [
        "# TRAP INTELLIGENCE REPORT",
        f"## Report ID: {r.report_id}",
        f"## Timestamp: {r.generated_at:%Y-%m-%d %H:%M:%S} UTC",
        f"## Asset: {r.asset}",
        f"## Analysis Window: {r.window}",
        "",
        "---",
        "",
        "## SECTION 1: RAW METRIC SNAPSHOT",
        "| Metric | Value | Source |", "|---|---|---|",
    ]
    parts += [f"| {a} | {b} | {c} |" for a, b, c in r.raw_metrics] or ["| — | — | — |"]
    if r.unavailable_domains:
        parts.append(f"\nUnavailable this run: {', '.join(r.unavailable_domains)}")
    parts += ["", "---", "", _desk_section(2, "BULLISH DESK SUMMARY", r.desks.get("bullish")), "---", "",
              _desk_section(3, "BEARISH DESK SUMMARY", r.desks.get("bearish")), "---", ""]

    d = r.director
    parts.append("## SECTION 4: DIRECTOR'S CROSS-EXAMINATION")
    if d is None:
        parts.append(MISSING)
    else:
        parts += [f"{d.executive_summary}", "", "### Contradiction Analysis", "| Metric | Bull read | Bear read | Stronger evidence |", "|---|---|---|---|"]
        parts += [f"| {c.metric} | {c.bull_read} | {c.bear_read} | {c.stronger_evidence} |" for c in d.contradictions] or ["| — | — | — | — |"]
        parts += ["", "### Convergence (highest-confidence signals)", _ul(d.convergence), "", "### Blind Spots", _ul(d.blind_spots)]
    parts += ["", "---", "", "## SECTION 5: SCENARIO PROBABILITIES & TRAP CLASSIFICATION"]
    if d is None:
        parts.append(MISSING)
    else:
        for s in d.scenarios:
            parts.append(f"- **{s.name} — {s.probability:.0f}%**: {s.path} _Invalidation: {s.invalidation}_")
        parts += ["", f"### Final Classification: **{d.trap_classification.replace('_', ' ')}**", d.classification_rationale]
    parts += ["", "---", "", "## SECTION 6: ACTIONABLE TRADE PLAN"]
    if d is None:
        parts.append(MISSING)
    else:
        p = d.trade_plan
        parts += [
            f"1. **Existing Position Management:** {p.existing_position_management}",
            f"2. **New Entry Conditions:** {p.new_entry_conditions}",
            f"3. **Position Sizing:** {p.position_sizing}",
            f"4. **Stop Loss Placement:** {p.stop_loss}",
            "5. **Take Profit Ladder:**", _ul(p.take_profit_ladder),
            f"6. **Leverage Maximum:** {p.max_leverage}",
            "7. **Time-Based Rules:**", _ul(p.time_rules),
        ]

    parts += ["", "---", "", "## APPENDIX"]
    parts.append("### Volatility Agent")
    if r.volatility:
        parts += [r.volatility.regime_summary, ""]
        parts += [f"- **{k}**: {v}" for k, v in r.volatility.per_category.items()]
        parts += ["", _ul(r.volatility.range_expectations)]
    else:
        parts.append(MISSING)
    parts.append("\n### Accuracy & Correlation Agent")
    if r.accuracy:
        a = r.accuracy
        parts += [f"Consistency score: **{a.consistency_score:.0f}/100**", a.cross_timeframe_alignment, "",
                  "Contradictions found:", _ul(a.contradictions_found), "", a.correlation_notes, "", "Data quality flags:", _ul(a.data_quality_flags)]
    else:
        parts.append(MISSING)
    parts += ["\n### Agent Runs", "| Agent | Status | Model | Latency s | Tokens | Cost USD |", "|---|---|---|---|---|---|"]
    for run in r.runs:
        status = "ok" if run.ok else f"failed: {run.error}"
        cost = f"{run.cost_usd:.4f}" if run.cost_usd is not None else "—"
        parts.append(f"| {run.agent_id} | {status} | {run.model} | {run.latency_s:.1f} | {run.prompt_tokens + run.completion_tokens} | {cost} |")
    total_cost = f"${r.total_cost_usd:.4f}" if r.total_cost_usd is not None else "not reported by provider"
    parts += ["", f"Provider: {r.provider} · total tokens: {r.total_tokens:,} · total cost: {total_cost} · wall time: {r.wall_time_s:.1f}s",
              "", "_Probabilities are model-weighted scenarios built from the quoted data, not predictions. Not investment advice._"]
    return "\n".join(parts)
