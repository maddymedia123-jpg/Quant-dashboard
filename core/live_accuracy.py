"""The head agent's accuracy audit: grade both teams, and name the protocols that failed.

The client's spec asks the head agent to gauge the accuracy of both teams and, when it is not optimal,
publish a separate accuracy report that pinpoints protocol failures, misread liquidity, false sentiment
signals, structural misalignment and missed traps, so the framework can be corrected.

Every pinned anchor is a prediction. When its candle closes it is scored once, against the price at
that exact candle boundary; if no completed candle sits on the boundary the window waits rather than
borrowing a neighbour's price. Scores are arithmetic, so the audit can always be recomputed.

"Optimal" is not defined in the spec, so the thresholds here are explicit and tunable:

* a move inside the category's flat band counts as flat, so a BALANCED call can be right;
* a team's 0-100 score is read as its forecast probability for its own direction and graded with a
  Brier score, where 0.25 is what always answering 50% would earn;
* the report stays in "collecting" until MIN_SAMPLES windows are scored, then reports "calibration
  needed" when the direction hit rate is under HIT_FLOOR or either team does worse than a coin flip.

Findings are pointers for a person to act on. Nothing here changes weights or prompts by itself."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from core.agents.recon_profiles import LIVE, PROFILES, ReconProfile
from core.agents.rubric import ALL_ITEMS, BEARISH, BULLISH

UP, DOWN, FLAT = "UP", "DOWN", "FLAT"
TF_MS = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000, "1w": 604_800_000}

# A move smaller than this, in percent, is noise for the horizon rather than a direction.
FLAT_BAND_PCT = {"live": 0.25, "intraday": 0.6, "weekly": 1.5, "monthly": 3.0}

MIN_SAMPLES = 10          # windows scored before the report says anything about calibration
HIT_FLOOR = 0.55          # direction hit rate below this is not optimal
COIN_FLIP = 0.25          # Brier score of always answering 50%
ITEM_MIN_WINDOWS = 5      # an item needs this many graded windows before it can be named
ITEM_ERROR = 0.25         # an item worse than a coin flip...
ITEM_GAP = 0.20           # ...and this far from what actually happened is a finding
MAX_ITEM_ISSUES = 8
TRAP_MIN_MISSES = 3

ITEM_KIND = {"smc": "structural misalignment", "mtf": "structural misalignment",
             "liq": "misread liquidity", "macro": "false sentiment", "quant": "volatility misread"}

SUGGESTIONS = {
    "structural misalignment": "Require a break to hold for a full {ltf} candle and agree with the {htf} bias "
                               "before this item scores high.",
    "misread liquidity": "Count a sweep as confirmation only after a reclaim close, and score sweeps against "
                         "the {htf} bias lower.",
    "false sentiment": "Down-weight sentiment when positioning (funding, open interest) disagrees with it "
                       "instead of treating it as a lead signal.",
    "volatility misread": "Treat band pushes as exhaustion unless ATR is expanding on the same timeframe.",
    "missed signal": "The team discounted a condition that kept preceding moves in its favour; check the "
                     "evidence for it is reaching the state.",
    "missed traps": "Lower the bar for the trap agent to flag a trap when liquidity was swept against the "
                    "anchored bias.",
    "false trap alarm": "Raise the trap override above 0.60 for this category, or require a confirming delta "
                        "divergence before overriding.",
}


@dataclass(frozen=True)
class WindowScore:
    move_pct: float
    realised: str
    direction_hit: bool
    trap_hit: bool | None
    bull_brier: float
    bear_brier: float


@dataclass
class Issue:
    kind: str
    finding: str
    suggestion: str
    windows: int
    side: str | None = None
    item: str | None = None
    mean_probability: float | None = None
    realised_rate: float | None = None


@dataclass
class AccuracyReport:
    category: str
    status: str                        # collecting, healthy, calibration needed
    n: int
    headline: str
    hit_rate: float | None = None
    trap_hit_rate: float | None = None
    bull_brier: float | None = None
    bear_brier: float | None = None
    issues: list[Issue] = field(default_factory=list)


def price_at(frames: dict[str, pd.DataFrame], boundary_ms: int) -> float | None:
    """Close of the completed candle that ends exactly on the boundary, finest timeframe first."""
    for tf, span in TF_MS.items():
        df = frames.get(tf)
        if df is None or df.empty:
            continue
        hit = df[df["timestamp"].astype("int64") + span == boundary_ms]
        if not hit.empty:
            return float(hit["close"].iloc[-1])
    return None


def realised_direction(move_pct: float, profile: ReconProfile) -> str:
    band = FLAT_BAND_PCT.get(profile.category, 0.25)
    return UP if move_pct > band else DOWN if move_pct < -band else FLAT


def score_window(bias: str, trap: str | None, bull_points: float, bear_points: float, open_px: float,
                 close_px: float, profile: ReconProfile = LIVE) -> WindowScore:
    move = (close_px / open_px - 1.0) * 100.0
    real = realised_direction(move, profile)

    if bias.startswith("BULL TRAP"):
        hit = real in (DOWN, FLAT)                   # the long it doubted did not work
    elif bias.startswith("BEAR TRAP"):
        hit = real in (UP, FLAT)
    elif bias.startswith("BULL"):
        hit = real == UP
    elif bias.startswith("BEAR"):
        hit = real == DOWN
    else:
        hit = real == FLAT

    base = "BULL" if bias.startswith("BULL") else "BEAR" if bias.startswith("BEAR") else "BALANCED"
    if trap is None:
        trap_hit = None
    elif trap == "BULL_TRAP":
        trap_hit = real == DOWN
    elif trap == "BEAR_TRAP":
        trap_hit = real == UP
    elif trap == "NO_TRAP":
        trap_hit = real == FLAT
    elif trap == "GENUINE_MOVE":
        trap_hit = None if base == "BALANCED" else real == (UP if base == "BULL" else DOWN)
    else:
        trap_hit = None

    p_up, p_down = bull_points / 100.0, bear_points / 100.0
    return WindowScore(move, real, hit, trap_hit,
                       (p_up - (1.0 if real == UP else 0.0)) ** 2,
                       (p_down - (1.0 if real == DOWN else 0.0)) ** 2)


def score_due(store, frames: dict[str, pd.DataFrame], now_ms: int,
              profile: ReconProfile = LIVE) -> list[WindowScore]:
    """Score every pinned window whose candle has closed. Windows without a closing price wait."""
    out: list[WindowScore] = []
    for row in store.anchors_due(now_ms, category=profile.category):
        open_px = row.get("price")
        if not open_px:
            continue
        close_px = price_at(frames, row["window_close_ms"])
        if close_px is None:
            continue
        p = row["payload"]
        s = score_window(row["bias"], row.get("trap"), float(p.get("bull_points") or 0.0),
                         float(p.get("bear_points") or 0.0), float(open_px), close_px, profile)
        if store.put_recon_score(profile.category, row["window_open_ms"], now_ms, float(open_px), close_px,
                                 s.move_pct, s.realised, row["bias"], s.direction_hit, row.get("trap"),
                                 s.trap_hit, s.bull_brier, s.bear_brier):
            out.append(s)
    return out


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _item_issues(rows: list[dict], profile: ReconProfile) -> list[Issue]:
    found: list[tuple[float, Issue]] = []
    for side, target in ((BULLISH, UP), (BEARISH, DOWN)):
        for item in ALL_ITEMS:
            pairs = [(float(r["payload"]["items"][side][item.id]), 1.0 if r["realised"] == target else 0.0)
                     for r in rows
                     if item.id in ((r["payload"].get("items") or {}).get(side) or {})]
            if len(pairs) < ITEM_MIN_WINDOWS:
                continue
            err = _mean([(p - o) ** 2 for p, o in pairs])
            mean_p, mean_o = _mean([p for p, _ in pairs]), _mean([o for _, o in pairs])
            if err <= ITEM_ERROR or abs(mean_p - mean_o) <= ITEM_GAP:
                continue
            over = mean_p > mean_o
            kind = ITEM_KIND[item.id.split("_")[0]] if over else "missed signal"
            finding = (f"The {side} team gave “{item.question(side, profile)}” an average "
                       f"{mean_p:.0%} across {len(pairs)} windows; its case played out in {mean_o:.0%} of them.")
            found.append((err, Issue(kind, finding, SUGGESTIONS[kind].format(**profile.fmt), len(pairs),
                                     side, item.id, mean_p, mean_o)))
    found.sort(key=lambda t: -t[0])
    return [i for _, i in found[:MAX_ITEM_ISSUES]]


def _trap_issues(rows: list[dict], profile: ReconProfile) -> list[Issue]:
    issues: list[Issue] = []
    graded = [r for r in rows if r.get("trap_hit") is not None]
    for call in ("GENUINE_MOVE", "NO_TRAP"):
        misses = [r for r in graded if r["trap"] == call and not r["trap_hit"]]
        if len(misses) >= TRAP_MIN_MISSES:
            total = sum(1 for r in graded if r["trap"] == call)
            issues.append(Issue("missed traps",
                                f"The war room called {call} in {len(misses)} of {total} windows that went the "
                                "other way; the setup was a trap or a reversal it did not flag.",
                                SUGGESTIONS["missed traps"].format(**profile.fmt), len(misses)))
    for call in ("BULL_TRAP", "BEAR_TRAP"):
        misses = [r for r in graded if r["trap"] == call and not r["trap_hit"]]
        if len(misses) >= TRAP_MIN_MISSES:
            total = sum(1 for r in graded if r["trap"] == call)
            issues.append(Issue("false trap alarm",
                                f"The war room called {call} in {len(misses)} of {total} windows where the move "
                                "it doubted carried on.",
                                SUGGESTIONS["false trap alarm"].format(**profile.fmt), len(misses)))
    return issues


def diagnose(store, profile: ReconProfile = LIVE, limit: int = 50) -> AccuracyReport:
    """Grade both teams over the most recent scored windows and name what failed."""
    rows = store.scored_windows(category=profile.category, limit=limit)
    n = len(rows)
    if n < MIN_SAMPLES:
        return AccuracyReport(profile.category, "collecting", n,
                              f"{n} of {MIN_SAMPLES} {profile.window_label} windows scored; the accuracy report "
                              f"activates at {MIN_SAMPLES}.")

    hit = _mean([float(r["direction_hit"]) for r in rows])
    traps = [float(r["trap_hit"]) for r in rows if r.get("trap_hit") is not None]
    bull_b = _mean([r["bull_brier"] for r in rows if r.get("bull_brier") is not None])
    bear_b = _mean([r["bear_brier"] for r in rows if r.get("bear_brier") is not None])
    below = hit < HIT_FLOOR or (bull_b or 0.0) > COIN_FLIP or (bear_b or 0.0) > COIN_FLIP

    rep = AccuracyReport(profile.category, "calibration needed" if below else "healthy", n, "",
                         hit, _mean(traps), bull_b, bear_b)
    if not below:
        rep.headline = (f"Direction right in {hit:.0%} of {n} windows; both teams beat a coin flip. "
                        "No calibration needed.")
        return rep

    rep.issues = _item_issues(rows, profile) + _trap_issues(rows, profile)
    worse = [name for name, b in (("bullish", bull_b), ("bearish", bear_b)) if b is not None and b > COIN_FLIP]
    rep.headline = (f"Direction right in {hit:.0%} of {n} windows (floor {HIT_FLOOR:.0%})"
                    + (f"; the {' and '.join(worse)} team{'s' if len(worse) > 1 else ''} scored worse than a "
                       "coin flip" if worse else "")
                    + f". {len(rep.issues)} protocol finding{'s' if len(rep.issues) != 1 else ''} below.")
    return rep


def profile_for(category: str) -> ReconProfile:
    return PROFILES.get(category, LIVE)
