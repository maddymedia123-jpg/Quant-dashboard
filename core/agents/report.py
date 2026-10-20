"""Render a TrapReport as the client's markdown Trap Intelligence Report (Sections 1-6 + appendix)."""
from __future__ import annotations

from core.agents.schemas import DeskReport, TrapReport

MISSING = "_not produced this run_"


def _dual(dt) -> str:
    from core.trades import fmt_when
    return fmt_when(int(dt.timestamp() * 1000), seconds=True)


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
        f"## Timestamp: {r.generated_at:%Y-%m-%d} {_dual(r.generated_at)}",
        f"## Asset: {r.asset}",
        f"## Analysis Window: {r.window}",
        f"## Pipeline: {r.mode}" + (f" · snapshot {r.snapshot_id}" if r.snapshot_id else ""),
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
    if r.plan is not None:
        pl = r.plan
        parts += ["", "### Plan as numbers (watched by the alerts)"]
        if pl.side == "NONE":
            parts.append(f"No trade: {pl.why_none}")
        else:
            beyond = "below" if pl.side == "SHORT" else "above"
            parts += [f"- **Side:** {pl.side}", f"- **Entry zone:** {pl.zone_low:,.0f} – {pl.zone_high:,.0f}",
                      f"- **Confirms on:** {pl.confirm_timeframe} close {beyond} "
                      + (f"{pl.confirm_close:,.0f}" if pl.confirm_close else "the zone")
                      + (f"; taker buy/sell {beyond} {pl.taker_ratio_limit:.2f}" if pl.taker_ratio_limit else ""),
                      f"- **Stop:** {pl.stop:,.0f}", f"- **Targets:** {', '.join(f'{t:,.0f}' for t in pl.targets)}",
                      f"- **Valid for:** {pl.expires_hours:.0f}h"]

    parts += ["", "---", "", "## APPENDIX"]
    one = r.mode == "1-agent"
    # the 1-agent report has no separate volatility or accuracy agent: "not produced this run" read as a
    # failure when nothing failed
    parts.append("### Volatility Agent")
    if one:
        parts.append("_Not part of the 1-agent report: the single model reads the volatility data directly._")
    elif r.volatility:
        parts += [r.volatility.regime_summary, ""]
        parts += [f"- **{k}**: {v}" for k, v in r.volatility.per_category.items()]
        parts += ["", _ul(r.volatility.range_expectations)]
    else:
        parts.append(MISSING)
    parts.append("\n### Accuracy & Correlation Agent")
    if one:
        parts.append("_Not part of the 1-agent report: its contradictions section does this job, and the "
                     "scorecard checks the calls against price._")
    elif r.accuracy:
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
