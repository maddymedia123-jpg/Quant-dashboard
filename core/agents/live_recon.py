"""Live Recon war room: two five-agent teams, one trap consult, one head verdict,
plus a 4-hour predictive Head-Agent Partner & self-auditing memory loop.

The client's model is eleven agents per timeframe: five bullish domain agents, five bearish domain
agents and a trap agent the Head of the War Room consults. Each domain agent is worth 20 points, so a
team carries 100.

Here the ten domain agents are one batched judgment per side (`core.agents.judge`) against the same
market state, because ten separate prompts would answer the same eighteen questions ten times over.
The scoring, the bias, the consult policy and the override live in this module as arithmetic, so every
number in the report can be recomputed from the probabilities without asking a model again."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.agents.context import category_brief, common_payload, derivatives_payload
from core.agents.judge import ChoiceResult, Judge, JudgeResult
from core.agents.recon_profiles import LIVE, ReconProfile
from core.agents.rubric import (
    BEARISH, BULLISH, DOMAINS, MAX_TEAM_POINTS, Domain,
    domain_score, missing_items, team_score, verdict,
)
from core.indicators import desk_derivs
from core.indicators import harmonics as hm
from core.indicators import liquidity as lq
from core.indicators import smc
from core.indicators import volatility as vol
from core.indicators import vwap_break as vb

log = logging.getLogger(__name__)

# A side needs this much more than the other before the head calls a direction at all.
DECISIVE_MARGIN = 10.0
# Both teams scoring this high means both cases are well supported: the classic trap setup.
CONFLICT_POINTS = 60.0
# How sure the trap agent must be before the head overrides a direction.
TRAP_OVERRIDE_P = 0.60

DEFAULT_MEMORY_PATH = Path("data/recon_memory.jsonl")

TRAP_OPTIONS: dict[str, str] = {
    "BULL_TRAP": "Upside strength is being manufactured: buyside liquidity is being taken to fill "
                 "sell orders, and the move up is likely to fail and reverse down.",
    "BEAR_TRAP": "Downside weakness is being manufactured: sellside liquidity is being taken to fill "
                 "buy orders, and the move down is likely to fail and reverse up.",
    "GENUINE_MOVE": "The current move is real continuation, supported by order flow and structure "
                    "rather than a liquidity raid.",
    "NO_TRAP": "There is no trap and no decisive move: the market is ranging or waiting on a catalyst.",
}


def trap_instructions(profile: ReconProfile | None = None) -> str:
    p = profile or LIVE
    return ("Both a bullish and a bearish team have argued their case on this BTC market state. Judge what "
            f"the setup over the next {p.horizon} actually is, paying attention to liquidity sweeps, CVD "
            "absorption, open interest and whether structure confirms the move or only price does.")


TRAP_INSTRUCTIONS = trap_instructions(LIVE)


@dataclass
class DomainBreakdown:
    key: str
    agent_no: int
    title: str
    points: float
    max_points: int
    items: tuple[tuple[str, int, float], ...] = ()   # (item id, weight, probability)

    @property
    def pct(self) -> float:
        return 100.0 * self.points / self.max_points if self.max_points else 0.0


@dataclass
class TeamScore:
    side: str
    points: float = 0.0
    domains: tuple[DomainBreakdown, ...] = ()
    provider: str = ""
    latency_s: float = 0.0
    missing: tuple[str, ...] = ()
    error: str | None = None

    @property
    def verdict(self) -> str:
        return verdict(self.points)

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class HeadBrief:
    """Actionable 4-hour predictive synthesis produced by the Head-Agent Partner."""
    horizon: str
    window_open_utc: str
    window_close_utc: str
    spot_price: float | None
    stance: str                          # e.g., "LONG BIAS", "SHORT BIAS", "FADE RALLY (BULL TRAP)", "WAIT / RANGE"
    confluence_thesis: str               # Plain-English synthesis of 1h shift + 4h structure + order flow
    buy_zone: tuple[float, float] | None
    sell_zone: tuple[float, float] | None
    target_price: float | None
    invalidation_level: float | None
    invalidation_reason: str
    inflection_window: str               # Estimated top/bottom or breakout timing inside the 4h candle
    one_hour_shifts: tuple[str, ...] = ()
    key_drivers: tuple[str, ...] = ()
    risk_warnings: tuple[str, ...] = ()
    memory_calibration: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LiveReconResult:
    generated_at: datetime
    bull: TeamScore
    bear: TeamScore
    trap: ChoiceResult
    bias: str = "BALANCED"
    margin: float = 0.0
    confidence: float = 0.0
    consulted: bool = False
    override: str | None = None
    notes: tuple[str, ...] = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0
    unavailable: tuple[str, ...] = field(default_factory=tuple)
    category: str = "live"
    # set by the cross-desk handshake (core.agents.handshake) before the verdict is published
    handshake: dict | None = None
    confidence_before_handshake: float | None = None
    # 4-hour predictive Head-Agent Partner brief + self-auditing calibration snapshot
    head_brief: HeadBrief | None = None

    @property
    def leader(self) -> TeamScore | None:
        if self.bias.startswith("BULL"):
            return self.bull
        if self.bias.startswith("BEAR"):
            return self.bear
        return None

    @property
    def degraded(self) -> bool:
        return not (self.bull.ok and self.bear.ok) or bool(self.bull.missing or self.bear.missing)


LIVE_TIMEFRAMES = LIVE.timeframes


# ---------------------------------------------------------------------------
# Self-Auditing Memory & Recursive Domain Weighting
# ---------------------------------------------------------------------------

class ReconMemoryStore:
    """Append-only JSONL ledger that audits expired 4h calls against realized OHLC candles
    and computes empirical domain reliability weights for future Head-Agent calibration."""

    def __init__(self, path: Path = DEFAULT_MEMORY_PATH) -> None:
        self.path = path

    def _read_records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        except Exception as exc:  # noqa: BLE001
            log.debug("recon memory read skipped: %s", exc)
        return records

    def _write_records(self, records: list[dict[str, Any]]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = "\n".join(json.dumps(r, sort_keys=True) for r in records)
            self.path.write_text((payload + "\n") if payload else "", encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            log.debug("recon memory write skipped: %s", exc)

    def audit_and_calibrate(self, m, category: str = "live", now_ms: int | None = None) -> dict[str, Any]:
        """Resolves any open predictions whose window_close_ms has passed using 1h/4h spot frames,
        persists updated outcomes, and returns empirical hit-rate + per-domain multipliers."""
        records = self._read_records()
        now = now_ms if now_ms is not None else int(m.generated_at.timestamp() * 1000)
        df_1h = getattr(getattr(m, "spot", None), "frames", {}).get("1h")
        changed = False

        for rec in records:
            if rec.get("category") != category or rec.get("status") != "OPEN":
                continue
            close_ms = int(rec.get("window_close_ms", 0))
            if close_ms <= 0 or now < close_ms:
                continue
            outcome = self._grade_record(rec, df_1h)
            if outcome is not None:
                rec.update(outcome)
                changed = True

        if changed:
            self._write_records(records)

        resolved = [r for r in records if r.get("category") == category and r.get("status") in ("WIN", "LOSS", "NEUTRAL")]
        directional = [r for r in resolved if r.get("status") in ("WIN", "LOSS")]
        wins = sum(1 for r in directional if r["status"] == "WIN")
        hit_rate = (wins / len(directional)) if directional else None

        # Compute per-domain calibration weights (bounded 0.75x to 1.25x) from resolved history
        domain_stats: dict[str, list[float]] = {d.key: [] for d in DOMAINS}
        for r in directional[-40:]:
            won = 1.0 if r["status"] == "WIN" else 0.0
            for d_key, pct in (r.get("leader_domains") or {}).items():
                if d_key in domain_stats and pct >= 60.0:
                    domain_stats[d_key].append(won)

        domain_weights: dict[str, float] = {}
        for d_key, samples in domain_stats.items():
            if len(samples) >= 3:
                acc = sum(samples) / len(samples)
                # Center around 0.55 baseline accuracy -> multiplier in [0.75, 1.25]
                mult = max(0.75, min(1.25, 1.0 + (acc - 0.55) * 0.8))
                domain_weights[d_key] = round(mult, 3)
            else:
                domain_weights[d_key] = 1.0

        return {
            "total_logged": len([r for r in records if r.get("category") == category]),
            "resolved_directional": len(directional),
            "wins": wins,
            "hit_rate_pct": round(hit_rate * 100.0, 1) if hit_rate is not None else None,
            "domain_weights": domain_weights,
            "recent_lessons": [r["lesson"] for r in resolved[-3:] if r.get("lesson")],
        }

    @staticmethod
    def _grade_record(rec: dict[str, Any], df_1h) -> dict[str, Any] | None:
        if df_1h is None or df_1h.empty:
            return None
        open_ms = int(rec.get("window_open_ms", 0))
        close_ms = int(rec.get("window_close_ms", 0))
        try:
            idx = df_1h.index
            if hasattr(idx, "tz") and idx.tz is not None:
                ts_ms = (idx.astype("int64") // 10**6)
            else:
                ts_ms = (idx.astype("int64") // 10**6)
            window_df = df_1h[(ts_ms >= open_ms) & (ts_ms <= close_ms)]
            if window_df.empty:
                return None
            hi = float(window_df["high"].max())
            lo = float(window_df["low"].min())
            end_close = float(window_df["close"].iloc[-1])
        except Exception:  # noqa: BLE001
            return None

        entry = float(rec.get("spot_price") or end_close)
        bias = rec.get("bias", "BALANCED")
        inv = rec.get("invalidation_level")
        target = rec.get("target_price")

        if bias in ("BULL", "BEAR TRAP RISK"):
            if inv is not None and lo <= float(inv):
                return {"status": "LOSS", "realized_close": end_close,
                        "lesson": f"Long bias invalidated at {inv:.0f} (window low {lo:.0f})."}
            hit_target = target is not None and hi >= float(target)
            won = hit_target or end_close > entry
            return {"status": "WIN" if won else "LOSS", "realized_close": end_close,
                    "lesson": f"Long {'hit target' if hit_target else 'closed ' + ('higher' if won else 'lower')} ({entry:.0f} -> {end_close:.0f})."}

        if bias in ("BEAR", "BULL TRAP RISK"):
            if inv is not None and hi >= float(inv):
                return {"status": "LOSS", "realized_close": end_close,
                        "lesson": f"Short bias invalidated at {inv:.0f} (window high {hi:.0f})."}
            hit_target = target is not None and lo <= float(target)
            won = hit_target or end_close < entry
            return {"status": "WIN" if won else "LOSS", "realized_close": end_close,
                    "lesson": f"Short {'hit target' if hit_target else 'closed ' + ('lower' if won else 'higher')} ({entry:.0f} -> {end_close:.0f})."}

        return {"status": "NEUTRAL", "realized_close": end_close,
                "lesson": f"Balanced/no-trade window closed at {end_close:.0f} vs {entry:.0f}."}

    def record_prediction(self, res: LiveReconResult, state: dict[str, Any], profile: ReconProfile,
                          now_ms: int, persist: bool = True) -> None:
        if not persist or res.degraded or res.head_brief is None:
            return
        open_ms = profile.window_open(now_ms)
        close_ms = profile.window_close(now_ms)
        records = self._read_records()
        # Deduplicate per category + 4h candle window so repeated UI refreshes don't spam the log
        if any(r.get("category") == profile.category and r.get("window_open_ms") == open_ms for r in records):
            return
        leader = res.leader
        leader_domains = {d.key: round(d.pct, 1) for d in leader.domains} if leader else {}
        records.append({
            "recorded_at": res.generated_at.isoformat(),
            "category": profile.category,
            "window_open_ms": open_ms,
            "window_close_ms": close_ms,
            "spot_price": res.head_brief.spot_price,
            "bias": res.bias,
            "confidence": round(res.confidence, 4),
            "bull_points": round(res.bull.points, 2),
            "bear_points": round(res.bear.points, 2),
            "trap_choice": res.trap.choice if res.trap.ok else None,
            "buy_zone": list(res.head_brief.buy_zone) if res.head_brief.buy_zone else None,
            "sell_zone": list(res.head_brief.sell_zone) if res.head_brief.sell_zone else None,
            "target_price": res.head_brief.target_price,
            "invalidation_level": res.head_brief.invalidation_level,
            "leader_domains": leader_domains,
            "status": "OPEN",
        })
        self._write_records(records[-500:])


# ---------------------------------------------------------------------------
# State Assembly & Indicator Extraction
# ---------------------------------------------------------------------------

def timeframe_brief(m, tf: str, context_tf: str) -> dict | None:
    """The same brief a desk's analysis gives, for a timeframe no desk charts."""
    from core.config import Category
    from core.indicators.category import analyze_category

    a = analyze_category(Category(tf, f"{tf} read", tf, context_tf, 4, f"next 4 x {tf}"),
                         m.spot.frames, m.futures)
    return category_brief(a, m) if a is not None else None


def cascade_reads(m, profile: ReconProfile) -> dict:
    """His v22 Section 3 read order: the derived candles between a desk's own three (Intraday's 12h,
    Weekly's 3d and half-week, Monthly's two-week), each as the same brief a desk's analysis gives, keyed
    by timeframe. A frame that could not be built is named as missing rather than left out, so a judge
    sees the gap. The desk's three-timeframe questions are unchanged; these are evidence beside them."""
    out: dict = {"order": list(profile.read_order), "briefs": {}}
    for tf in profile.cascade_reads:
        brief = timeframe_brief(m, tf, profile.timeframes[2])
        out["briefs"][tf] = brief if brief is not None else {"available": False, "note": f"no {tf} candles, or too few to read"}
    return out


def build_state(m, analyses: dict, now_ms: int | None = None, profile: ReconProfile | None = None) -> dict:
    """The shared state both teams and the trap agent judge.

    Live Recon reads 15m, 1h and 4h: the live and intraday analyses, and a 15m brief built here. The rest of the
    snapshot (spot, derivatives, calendar) comes along in the common payload, availability flags and
    all, so an agent can see that a field is missing instead of guessing at it.

    The SMC and liquidity blocks matter most: without them the structure, order block, gap, sweep,
    delta and volume-profile questions in the rubric have nothing to read, and a judge that cannot see
    the evidence correctly scores near zero. Every one of them is computed here from candles."""
    profile = profile or LIVE
    state = common_payload(m, analyses)
    wanted = profile.analysis_keys
    state["categories"] = {k: v for k, v in state.get("categories", {}).items() if k in wanted}
    state["timeframes_read"] = dict(wanted)
    # A timeframe the war room reads that no desk charts (Live's 15m, Monthly's 1M) still needs its own
    # brief - trend, EMA stack, RSI, patterns - or "all three timeframes point up" has one timeframe
    # with nothing behind it. It is built here, keyed by the timeframe itself.
    covered = set(wanted.values())
    for i, tf in enumerate(profile.timeframes):
        if tf in covered:
            continue
        brief = timeframe_brief(m, tf, profile.timeframes[i + 1] if i + 1 < len(profile.timeframes) else tf)
        if brief is not None:
            state["categories"][tf] = brief
            state["timeframes_read"][tf] = tf
    state["war_room"] = profile.label
    state["cascade_reads"] = cascade_reads(m, profile)
    if profile.bias_tf:
        # the timeframe cascade, the same object the desk shows: bias above, anchor, engine below
        from core.indicators.cascade import classify
        bias_tf, anchor_tf, engine_tfs = profile.cascade
        state["cascade"] = classify(m.spot.frames, bias_tf, anchor_tf, engine_tfs).to_state()
    state["horizon"] = profile.horizon
    # the desk's forecast pane, as the screen shows it: the range, the level odds and the record behind them
    desk = analyses.get(profile.category)
    if desk is not None:
        try:
            from core.indicators.forecast import desk_forecast, to_state as forecast_state
            f, tr = desk_forecast(desk, desk_derivs.gamma_for(profile.category, m),
                                  m.spot.frames.get(desk.chart_tf))
            state["forecast"] = forecast_state(f, tr)
        except Exception as e:  # noqa: BLE001 - the forecast must not take the war room down
            state["forecast"] = {"available": False, "note": f"{type(e).__name__}: {e}"}

    now = now_ms if now_ms is not None else int(m.generated_at.timestamp() * 1000)
    state["derivatives"], derivs_missing = derivatives_payload(m)
    state["derivatives"]["unavailable"] = derivs_missing
    # every feed that failed, not only the derivatives ones: a judge reading an empty list assumes
    # sentiment and the calendar are present
    state["unavailable"] = sorted(set(m.unavailable()) | set(derivs_missing))
    desk_derivs.match_window(state["derivatives"], profile.category, m, now)

    oi_history = desk_derivs.oi_history(m)
    state["smc"], state["liquidity"], state["volatility"] = {}, {}, {}
    for tf in profile.timeframes:
        df = m.spot.frames.get(tf)
        if df is None or df.empty:
            state["smc"][tf] = {"available": False, "note": f"no {tf} candles"}
            state["liquidity"][tf] = {"available": False, "note": f"no {tf} candles"}
            state["volatility"][tf] = {"available": False, "note": f"no {tf} candles"}
            continue
        state["smc"][tf] = smc.summarise(df)
        state["liquidity"][tf] = lq.summarise(df, oi_history, now, windows=profile.timeframes,
                                              flow=(getattr(m, "taker_flow", None).frames.get(tf)
                                                    if getattr(getattr(m, "taker_flow", None), "available", False)
                                                    else None))
        state["volatility"][tf] = vol.matrix_summary(df, tf, m.options)
        if tf in vb.TIMEFRAMES:
            # his qualified-break rule: close outside value + volume expansion + VWAP acceptance, 1H/4H/1D only
            state["liquidity"][tf]["qualified_break"] = vb.read(df, tf).to_state()
        if tf in hm.TIMEFRAMES:
            # harmonics, valid only beside a VWAP, value-area or gamma level
            try:
                found = hm.find(df, tf, desk_derivs.gamma_for(profile.category, m))
                state["smc"][tf]["harmonics"] = [h.to_state() for h in found]
            except Exception as e:  # noqa: BLE001 - a pattern scan must not take the state down
                state["smc"][tf]["harmonics"] = {"available": False, "note": f"{type(e).__name__}: {e}"}
    # his weekend modifier: the session the judges are reading, with the playbook that goes with it
    from core.indicators.weekend import read as weekend_read
    state["session"] = weekend_read(now).to_state()
    # election-day fix 3: the judges always know whether a major print's window is open
    try:
        from core.indicators.event_regime import read as regime_read
        cal = getattr(m, "calendar", None)
        state["event_regime"] = regime_read(getattr(cal, "events", None) or [] if getattr(cal, "available", False)
                                            else [], now).to_state()
    except Exception as e:  # noqa: BLE001 - the war room must not wait on the calendar
        state["event_regime"] = {"active": False, "note": f"{type(e).__name__}: {e}"}
    # resting liquidation clusters (Hyperliquid, on-chain): the stop's hazards and the positioning magnet
    ll = getattr(m, "liq_levels", None)
    state["liquidation_levels"] = ll.to_state() if ll is not None and getattr(ll, "available", False) else {"available": False}
    # deterministic early warnings (forming traps, squeezes, reversal windows, fading confluences)
    try:
        from core.indicators.early_warning import early_warnings
        if desk is not None and getattr(m, "futures", None) is not None:
            last_sq = getattr(getattr(desk, "squeeze", None), "score", None)
            ews = early_warnings(desk, m.futures, last_sq)
            state["early_warnings"] = [
                {"kind": w.kind, "level": w.level, "drivers": w.drivers, "anti_panic": w.anti_panic}
                for w in ews
            ]
        else:
            state["early_warnings"] = []
    except Exception as e:  # noqa: BLE001 - early warnings must not take the state down
        state["early_warnings"] = [{"kind": "ERROR", "level": "forming", "drivers": [f"{type(e).__name__}: {e}"], "anti_panic": ""}]
    # his Section 5.3: the named anomaly trap on this desk, so the judges and the Head read it by name
    try:
        from core.indicators import anomaly_trap
        trap = anomaly_trap.read(m.sentiment.fear_greed if getattr(m.sentiment, "available", False) else None,
                                 m.futures if getattr(m.futures, "available", False) else None,
                                 desk_derivs.read(profile.category, m, now),
                                 state["liquidity"].get(desk.chart_tf) if desk is not None else None,
                                 desk.price_change_24h_pct if desk is not None else None, oi_history, now)
        state["anomaly_trap"] = (trap.to_state() if trap is not None else
                                 {"name": None, "note": "no anomaly trap: the crowd is not extreme against the candle"})
    except Exception as e:  # noqa: BLE001 - the trap read must not take the war room down
        state["anomaly_trap"] = {"name": None, "note": f"{type(e).__name__}: {e}"}
    return state


# ---------------------------------------------------------------------------
# Scoring, Policy & Head-Agent Partner Synthesis
# ---------------------------------------------------------------------------

def score_team(side: str, res: JudgeResult) -> TeamScore:
    probs = res.probabilities
    breakdown = tuple(
        DomainBreakdown(
            key=d.key, agent_no=d.agent_no, title=d.title,
            points=domain_score(d, probs), max_points=d.max_points,
            items=tuple((i.id, i.weight, float(probs.get(i.id, 0.0))) for i in d.items),
        )
        for d in DOMAINS
    )
    return TeamScore(side=side, points=team_score(probs), domains=breakdown, provider=res.provider,
                     latency_s=res.latency_s, missing=tuple(missing_items(probs)) if res.ok else (),
                     error=res.error)


def bias_for(bull: float, bear: float) -> tuple[str, float]:
    """Direction and its margin. Below DECISIVE_MARGIN the head does not call a side."""
    margin = bull - bear
    if margin >= DECISIVE_MARGIN:
        return "BULL", margin
    if -margin >= DECISIVE_MARGIN:
        return "BEAR", margin
    return "BALANCED", margin


def needs_consult(bull: float, bear: float) -> bool:
    """The head consults the trap agent when the teams conflict or neither is decisive."""
    return (bull >= CONFLICT_POINTS and bear >= CONFLICT_POINTS) or abs(bull - bear) < DECISIVE_MARGIN


def _domain(key: str) -> Domain:
    return next(d for d in DOMAINS if d.key == key)


def resolve(bull: TeamScore, bear: TeamScore, trap: ChoiceResult) -> tuple[str, float, float, bool, str | None, tuple[str, ...]]:
    """Turn two team scores and the trap call into the head's verdict. Pure arithmetic and policy."""
    base, margin = bias_for(bull.points, bear.points)
    consulted = needs_consult(bull.points, bear.points)
    notes: list[str] = []
    override: str | None = None

    p = trap.probabilities if trap.ok else {}
    bull_trap, bear_trap = p.get("BULL_TRAP", 0.0), p.get("BEAR_TRAP", 0.0)
    bias = base

    if base == "BULL" and bull_trap >= TRAP_OVERRIDE_P:
        bias, override = "BULL TRAP RISK", f"bull trap probability {bull_trap:.0%} overrides the long bias"
        notes.append("High-danger bull-trap warning: avoid chasing upside spikes into resistance; watch for rejection or a relief bounce inside a broader drop.")
    elif base == "BEAR" and bear_trap >= TRAP_OVERRIDE_P:
        bias, override = "BEAR TRAP RISK", f"bear trap probability {bear_trap:.0%} overrides the short bias"
        notes.append("High-danger bear-trap / sweep warning: do not panic-sell into support or sweeps; wait for confirmed breakdown vs. rebound.")
    elif base == "BALANCED" and trap.ok and trap.choice in ("BULL_TRAP", "BEAR_TRAP"):
        notes.append(f"No decisive team score; trap agent reads the setup as {trap.choice.replace('_', ' ').lower()}.")

    winner = max(bull.points, bear.points)
    confidence = 0.0 if bias == "BALANCED" else min(1.0, (winner / MAX_TEAM_POINTS) * (0.5 + min(abs(margin), 40.0) / 80.0))

    if bull.points >= CONFLICT_POINTS and bear.points >= CONFLICT_POINTS:
        notes.append(f"Both teams score above {CONFLICT_POINTS:.0f}: the two cases are equally well supported, "
                     "which is the condition a trap is built in.")
    if abs(margin) < DECISIVE_MARGIN:
        notes.append(f"Margin {abs(margin):.1f} points is under the {DECISIVE_MARGIN:.0f}-point threshold, "
                     "so no direction is called.")
    for team in (bull, bear):
        if team.error:
            notes.append(f"The {team.side} team did not score ({team.error}); its 0 is a failure, not a reading.")
        elif team.missing:
            missing_titles = ", ".join(sorted({_domain(k.split('_')[0]).title for k in team.missing
                                               if k.split('_')[0] in {d.key for d in DOMAINS}}))
            notes.append(f"The {team.side} team is missing {len(team.missing)} checklist answers"
                         + (f" ({missing_titles})" if missing_titles else "") + "; those items scored 0.")
    if not trap.ok:
        notes.append("The trap agent did not answer, so no trap override was applied.")

    return bias, margin, confidence, consulted, override, tuple(notes)


def build_head_partner_brief(
    state: dict[str, Any],
    bull: TeamScore,
    bear: TeamScore,
    trap: ChoiceResult,
    bias: str,
    confidence: float,
    profile: ReconProfile,
    now_ms: int,
    calibration: dict[str, Any] | None = None,
) -> HeadBrief:
    """Synthesizes 1-hour shifts, 4-hour volatility/structure levels, trap risks, and historical
    memory weights into an actionable 4-hour Head-Agent Partner brief."""
    spot = state.get("spot", {}).get("price")
    spot_f = float(spot) if spot is not None else None

    open_ms = profile.window_open(now_ms)
    close_ms = profile.window_close(now_ms)
    open_dt = datetime.fromtimestamp(open_ms / 1000, tz=timezone.utc)
    close_dt = datetime.fromtimestamp(close_ms / 1000, tz=timezone.utc)
    rem_mins = max(0, int((close_ms - now_ms) // 60_000))

    mtf = profile.timeframes[1]   # "1h" for Live Recon
    htf = profile.timeframes[2]   # "4h" for Live Recon

    vol_htf = state.get("volatility", {}).get(htf, {})
    bands = vol_htf.get("bands") or {}
    liq_mtf = state.get("liquidity", {}).get(mtf, {})
    vp_mtf = liq_mtf.get("volume_profile") or {}
    smc_mtf = state.get("smc", {}).get(mtf, {})
    smc_htf = state.get("smc", {}).get(htf, {})

    # Extract 1-hour shifts (structure, qualified break, CVD, volume profile)
    shifts: list[str] = []
    struct_1h = (smc_mtf.get("structure") or {}).get("trend") or (smc_mtf.get("structure") or {}).get("bias")
    if struct_1h:
        shifts.append(f"{mtf} SMC structure: {struct_1h}")
    qb = liq_mtf.get("qualified_break") or {}
    if qb.get("active") or qb.get("direction"):
        shifts.append(f"{mtf} qualified break: {qb.get('direction') or qb.get('state') or 'active'}")
    cvd = (liq_mtf.get("cumulative_delta") or {}).get("trend") or (liq_mtf.get("cumulative_delta") or {}).get("bias")
    if cvd:
        shifts.append(f"{mtf} CVD flow: {cvd}")
    vp_pos = vp_mtf.get("price_position")
    if vp_pos:
        shifts.append(f"{mtf} auction position: {vp_pos}")

    # Compute Buy Zone, Sell Zone, Target, and Invalidation from 4h sigma bands + 1h value area
    val = vp_mtf.get("value_area_low")
    vah = vp_mtf.get("value_area_high")
    poc = vp_mtf.get("point_of_control")
    s1_dn, s2_dn = bands.get("sigma1_down"), bands.get("sigma2_down")
    s1_up, s2_up = bands.get("sigma1_up"), bands.get("sigma2_up")

    buy_low = s2_dn or val or (spot_f * 0.988 if spot_f else None)
    buy_high = s1_dn or poc or (spot_f * 0.995 if spot_f else None)
    if buy_low is not None and buy_high is not None and buy_low > buy_high:
        buy_low, buy_high = buy_high, buy_low
    buy_zone = (round(float(buy_low), 2), round(float(buy_high), 2)) if (buy_low and buy_high) else None

    sell_low = s1_up or vah or (spot_f * 1.005 if spot_f else None)
    sell_high = s2_up or (spot_f * 1.012 if spot_f else None)
    if sell_low is not None and sell_high is not None and sell_low > sell_high:
        sell_low, sell_high = sell_high, sell_low
    sell_zone = (round(float(sell_low), 2), round(float(sell_high), 2)) if (sell_low and sell_high) else None

    # Stance, Target, Invalidation & Top/Bottom Inflection Timeline
    memory_regime = vol_htf.get("memory", "random walk")
    if bias == "BULL":
        stance = "LONG BIAS — BUY PULLBACKS"
        target_price = round(float(s1_up or vah or (spot_f * 1.012)), 2) if spot_f else None
        invalidation = round(float(s2_dn or val or (spot_f * 0.985)), 2) if spot_f else None
        inv_reason = f"1h close below {htf} -2σ / Value Area Low breaks bullish 4h continuation."
        inflection = (
            f"Look for a local bottom within the next 1–2 hourly closes ({rem_mins}m left in current {htf} candle) "
            f"before expansion toward {target_price}."
        )
    elif bias == "BEAR":
        stance = "SHORT BIAS — SELL RALLIES"
        target_price = round(float(s1_dn or val or (spot_f * 0.988)), 2) if spot_f else None
        invalidation = round(float(s2_up or vah or (spot_f * 1.015)), 2) if spot_f else None
        inv_reason = f"1h close above {htf} +2σ / Value Area High invalidates bearish 4h structure."
        inflection = (
            f"Look for a lower-high rejection inside the next 1–2 hourly closes ({rem_mins}m left in current {htf} candle) "
            f"targeting {target_price}."
        )
    elif bias == "BULL TRAP RISK":
        stance = "DEFENSIVE / FADE UPSIDE SPIKES (BULL TRAP)"
        target_price = round(float(poc or s1_dn or (spot_f * 0.99)), 2) if spot_f else None
        invalidation = round(float(s2_up or (spot_f * 1.015)), 2) if spot_f else None
        inv_reason = "Sustained 1h acceptance with positive CVD above upper liquidity pool invalidates trap."
        inflection = f"Top/reversal window active now through the next 1h close ({rem_mins}m to {htf} close)."
    elif bias == "BEAR TRAP RISK":
        stance = "DEFENSIVE / FADE DOWNSIDE SWEEPS (BEAR TRAP)"
        target_price = round(float(poc or s1_up or (spot_f * 1.01)), 2) if spot_f else None
        invalidation = round(float(s2_dn or (spot_f * 0.985)), 2) if spot_f else None
        inv_reason = "Sustained 1h breakdown below lower liquidity pool invalidates rebound."
        inflection = f"Bottom/rebound window active now through the next 1h close ({rem_mins}m to {htf} close)."
    else:
        stance = "WAIT / STAND ASIDE (NO DECISIVE EDGE)"
        target_price = round(float(poc), 2) if poc is not None else None
        invalidation = None
        inv_reason = "No active directional exposure until Bull/Bear spread exceeds 10 points."
        inflection = (
            f"Range-bound ({memory_regime}); wait for the current {htf} candle close ({rem_mins}m remaining) "
            "or a qualified 1h break outside value."
        )

    # Top domain drivers weighted by empirical memory calibration
    cal = calibration or {}
    weights = cal.get("domain_weights") or {}
    leader = bull if bull.points >= bear.points else bear
    ranked_domains = sorted(
        leader.domains,
        key=lambda d: d.points * float(weights.get(d.key, 1.0)),
        reverse=True,
    )
    key_drivers = tuple(
        f"Agent {d.agent_no} ({d.title}): {d.points:.1f}/{d.max_points} pts (mem weight {weights.get(d.key, 1.0):.2f}x)"
        for d in ranked_domains[:3]
    )

    # Risk warnings from anomaly traps, event regime, and early warnings
    warnings: list[str] = []
    anom = state.get("anomaly_trap") or {}
    if anom.get("name"):
        warnings.append(f"Anomaly trap active: {anom['name']}")
    if (state.get("event_regime") or {}).get("active"):
        warnings.append("Macro event window open: expect spread widening and stop hunts.")
    for ew in (state.get("early_warnings") or [])[:2]:
        if ew.get("kind") and ew.get("kind") != "ERROR":
            warnings.append(f"{ew['kind']} ({ew.get('level', 'watch')})")

    htf_pd = (smc_htf.get("premium_discount") or {}).get("zone", "equilibrium")
    hit_txt = (
        f" | Self-audit accuracy: {cal['hit_rate_pct']}% ({cal['wins']}/{cal['resolved_directional']})"
        if cal.get("hit_rate_pct") is not None
        else ""
    )
    thesis = (
        f"{profile.label} projects the next {profile.horizon} ({open_dt:%H:%M}–{close_dt:%H:%M} UTC) as {stance}. "
        f"Bull ({bull.points:.1f}) vs Bear ({bear.points:.1f}) in {htf} {htf_pd} ({memory_regime} regime), "
        f"calibrated confidence {confidence:.0%}.{hit_txt}"
    )

    return HeadBrief(
        horizon=profile.horizon,
        window_open_utc=open_dt.strftime("%Y-%m-%d %H:%M UTC"),
        window_close_utc=close_dt.strftime("%Y-%m-%d %H:%M UTC"),
        spot_price=spot_f,
        stance=stance,
        confluence_thesis=thesis,
        buy_zone=buy_zone,
        sell_zone=sell_zone,
        target_price=target_price,
        invalidation_level=invalidation,
        invalidation_reason=inv_reason,
        inflection_window=inflection,
        one_hour_shifts=tuple(shifts),
        key_drivers=key_drivers,
        risk_warnings=tuple(warnings),
        memory_calibration=cal,
    )


async def run_live_recon(
    judge: Judge,
    m,
    analyses: dict,
    now: datetime | None = None,
    profile: ReconProfile | None = None,
    memory_store: ReconMemoryStore | None = None,
    persist_memory: bool = False,
) -> LiveReconResult:
    """One war-room pass: both teams and the trap consult, in parallel, on one shared state.

    Live Recon by default; any category's profile runs the same eleven agents on its own timeframes.
    Automatically audits past calls via ReconMemoryStore and builds the 4-hour HeadBrief."""
    profile = profile or LIVE
    gen_at = now or datetime.now(timezone.utc)
    now_ms = int(gen_at.timestamp() * 1000)

    store = memory_store or ReconMemoryStore()
    calibration = store.audit_and_calibrate(m, category=profile.category, now_ms=now_ms)

    state = build_state(m, analyses, now_ms=now_ms, profile=profile)
    state["memory_calibration"] = calibration

    bull_res, bear_res, trap = await asyncio.gather(
        judge.score(state, BULLISH, profile=profile),
        judge.score(state, BEARISH, profile=profile),
        judge.classify(state, "trap", trap_instructions(profile), TRAP_OPTIONS),
    )
    bull, bear = score_team(BULLISH, bull_res), score_team(BEARISH, bear_res)
    bias, margin, confidence, consulted, override, notes = resolve(bull, bear, trap)

    # Calibrate confidence modestly using historical hit rate when enough resolved samples exist
    if calibration.get("resolved_directional", 0) >= 5 and calibration.get("hit_rate_pct") is not None and confidence > 0.0:
        acc_factor = max(0.75, min(1.20, 0.65 + (calibration["hit_rate_pct"] / 100.0) * 0.65))
        confidence = round(min(1.0, confidence * acc_factor), 4)

    head_brief = build_head_partner_brief(
        state=state,
        bull=bull,
        bear=bear,
        trap=trap,
        bias=bias,
        confidence=confidence,
        profile=profile,
        now_ms=now_ms,
        calibration=calibration,
    )

    result = LiveReconResult(
        generated_at=gen_at,
        bull=bull, bear=bear, trap=trap,
        bias=bias, margin=margin, confidence=confidence, consulted=consulted, override=override, notes=notes,
        prompt_tokens=bull_res.prompt_tokens + bear_res.prompt_tokens,
        completion_tokens=bull_res.completion_tokens + bear_res.completion_tokens,
        unavailable=tuple(state.get("unavailable", ())),
        category=profile.category,
        head_brief=head_brief,
    )
    store.record_prediction(result, state, profile, now_ms=now_ms, persist=persist_memory)
    return result
