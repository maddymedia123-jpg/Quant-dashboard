"""The Active Trade desk's arithmetic: the stop, the timed targets, the pullback band, the schedule.

The spec asks for four things a trader can act on, and every one of them is a measurement rather than an
opinion, so all of it lives here in code and none of it is asked of a model:

* *Stop-loss verification.* A stop is not judged by its distance but by what sits between it and price.
  Resting liquidity gets swept before a level gives way, so a stop placed nearer than the pool below a
  long is a stop that will be hunted on the way to a move that then goes the trader's way. When the given
  stop is unsafe the engine proposes one beyond the protective structure instead.
* *Timed take-profits.* The price of a target comes from the volatility bands - where price can actually
  reach inside the horizon - and its *time* comes from the Fibonacci time zones already computed for the
  charts. A target with no date is a wish, so a target that cannot be dated says so.
* *The pullback band.* Measured, not assumed: the distribution of adverse excursions over the lookback,
  so the card can say which drawdown is ordinary and which one breaks the thesis. This is the number that
  stops a trader closing a good position during a normal retracement.
* *The monitoring schedule.* The exact candle closes to look at, in UTC, from the same window arithmetic
  the war rooms anchor to.

Nothing here decides whether a trade is good; that is the trade desk's ten sub-agents and their Head."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from core.agents.recon_profiles import DAY_MS, HOUR_MS, MONDAY_OFFSET_MS, WEEK_MS
# one definition, imported rather than repeated: app.py reached for trades.DEFAULT_ATR_PCT on the
# no-ATR path and got an AttributeError, because the constant only existed in core.traps
from core.traps import DEFAULT_ATR_PCT

LONG, SHORT = "LONG", "SHORT"
# A stop has to sit beyond the structure that protects it, by enough that the wick which takes the
# structure does not also take the stop. Half an ATR is the same buffer the trap desk invalidates on.
STOP_BUFFER_ATR = 0.5
# Closer than this to price and the stop is inside the noise: normal bar range alone would trigger it.
MIN_STOP_ATR = 0.75
# Levels within this of one another are one cluster, and a stop belongs beyond the whole cluster rather
# than between two levels that price will take in a single move.
CLUSTER_ATR = 1.0
# A stop is hunted when liquidity sits this close beyond it: near enough that the wick which takes the
# liquidity also takes the stop. Anything further beyond is just a level further away.
HUNT_ATR = 1.0
TARGET_SIGMAS = (1.0, 2.0, 3.0)
# Percentiles of the measured adverse excursion: the first is an ordinary pullback, the second is the
# edge of ordinary, and past the third the move that was expected is not happening.
PULLBACK_PCTILES = (50, 80, 95)

TF_MS = {"15m": 15 * 60_000, "1h": HOUR_MS, "4h": 4 * HOUR_MS, "1d": DAY_MS, "1w": WEEK_MS}


def _utc(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, timezone.utc)


def fmt_when(ms: int | None) -> str:
    """A time a trader can set an alarm by, always in UTC because every candle close here is."""
    if ms is None:
        return "no date"
    d = _utc(ms)
    return f"{d:%H:%M} UTC {d:%a %d %b}"


# ---------- the stop ----------
@dataclass(frozen=True)
class StopVerdict:
    side: str
    given: float | None
    proposed: float
    verdict: str                 # sound | would be hunted | inside the noise | on the wrong side | supplied
    protects_at: float | None    # the structure the proposed stop sits beyond
    protects_kind: str
    reasons: tuple[str, ...] = ()

    @property
    def sound(self) -> bool:
        return self.verdict == "sound"

    @property
    def stop(self) -> float:
        """What to actually trade: the given stop when it is sound, otherwise the proposed one."""
        return self.given if (self.given is not None and self.sound) else self.proposed


def _hazards(state: dict, side: str, entry: float, timeframe: str) -> list[tuple[float, str]]:
    """Everything between entry and a stop that price is likely to reach for first, nearest last.

    For a long these are the levels *below* entry: resting sellside liquidity, the last swing low, the
    value-area low. Each one is a place price can trade to and recover from."""
    long = side == LONG
    out: list[tuple[float, str]] = []

    def add(level, kind):
        if level is None:
            return
        try:
            value = float(level)
        except (TypeError, ValueError):
            return
        if value <= 0:
            return
        if (value < entry) if long else (value > entry):
            out.append((value, kind))

    liq = ((state.get("liquidity") or {}).get(timeframe) or {})
    pools = liq.get("pools") or {}
    pool = pools.get("sellside" if long else "buyside") or {}
    add(pool.get("price"), "resting liquidity")

    vp = liq.get("volume_profile") or {}
    add(vp.get("value_area_low" if long else "value_area_high"), "the value area edge")

    # swing_low / swing_high live on the structure block, which is also where the last BOS or CHoCH
    # carries the level it broke
    structure = ((state.get("smc") or {}).get(timeframe) or {}).get("structure") or {}
    add(structure.get("swing_low" if long else "swing_high"), "the last swing")
    add((structure.get("last_event") or {}).get("level"), "the structural level")

    # nearest to entry first, so the first hazard is the one price meets soonest
    return sorted(out, key=lambda pair: -pair[0] if long else pair[0])


def _protective(hazards: list[tuple[float, str]], long: bool, entry: float,
                atr: float) -> tuple[float | None, str]:
    """The structure a stop should sit beyond: the nearest hazard far enough from entry to be outside the
    noise, extended through any cluster of levels within CLUSTER_ATR of it.

    Taking the deepest hazard instead put the stop five ATR from entry on real candles, which leaves no
    trade worth taking; taking the nearest put it inside a single bar's range."""
    usable = [(lvl, kind) for lvl, kind in hazards if abs(entry - lvl) >= atr * MIN_STOP_ATR]
    if not usable:
        return None, ""
    anchor, kind = usable[0]
    kinds = [kind]
    changed = True
    while changed:                      # walk outwards through levels that price would take in one move
        changed = False
        for lvl, k in usable:
            beyond = (lvl < anchor) if long else (lvl > anchor)
            if beyond and abs(lvl - anchor) <= atr * CLUSTER_ATR:
                anchor, changed = lvl, True
                kinds.append(k)
    label = kinds[0] if len(kinds) == 1 else f"a cluster of {len(kinds)} levels ending at {kinds[-1]}"
    return anchor, label


def verify_stop(state: dict, side: str, entry: float, given: float | None, timeframe: str,
                atr: float) -> StopVerdict:
    """Is this stop structurally sound, or will it be hunted on the way to the trade working?"""
    long = side == LONG
    hazards = _hazards(state, side, entry, timeframe)
    buffer = atr * STOP_BUFFER_ATR
    anchor, protects_kind = _protective(hazards, long, entry, atr)
    if anchor is not None:
        proposed = anchor - buffer if long else anchor + buffer
    else:
        # nothing to hide behind: fall back to a multiple of range, which is honest about being a default
        proposed = entry - 2 * atr if long else entry + 2 * atr
        protects_kind = "no structure far enough from entry, so a two-ATR default"
    proposed = round(proposed, 2)

    reasons: list[str] = []
    if anchor is not None:
        reasons.append(f"{protects_kind} sits at {anchor:,.0f}; a stop belongs "
                       f"{'below' if long else 'above'} it, not inside it")

    if given is None:
        return StopVerdict(side, None, proposed, "supplied", anchor, protects_kind,
                           tuple(reasons + ["No stop was given, so the engine placed one from structure."]))

    if (given >= entry) if long else (given <= entry):
        return StopVerdict(side, given, proposed, "on the wrong side", anchor, protects_kind,
                           tuple(reasons + [f"A {side.lower()} stop has to sit "
                                            f"{'below' if long else 'above'} the entry."]))

    if abs(entry - given) < atr * MIN_STOP_ATR:
        reasons.append(f"The stop is {abs(entry - given):,.0f} from entry, inside a single {timeframe} "
                       f"range of {atr:,.0f}: ordinary noise would take it out.")
        return StopVerdict(side, given, proposed, "inside the noise", anchor, protects_kind, tuple(reasons))

    hunted = [(lvl, kind) for lvl, kind in hazards
              if ((lvl < given) if long else (lvl > given)) and abs(given - lvl) <= atr * HUNT_ATR]
    if hunted:
        lvl, kind = max(hunted, key=lambda p: p[0]) if long else min(hunted, key=lambda p: p[0])
        reasons.append(f"{kind} at {lvl:,.0f} sits just {abs(given - lvl):,.0f} beyond the stop, so the "
                       f"wick that takes that liquidity takes the stop with it - the sweep that usually "
                       f"precedes the move.")
        return StopVerdict(side, given, proposed, "would be hunted", anchor, protects_kind, tuple(reasons))

    reasons.append(f"The stop sits beyond the levels price reaches for first on the {timeframe}, with no "
                   f"resting liquidity within {atr * HUNT_ATR:,.0f} beyond it.")
    return StopVerdict(side, given, proposed, "sound", anchor, protects_kind, tuple(reasons))


# ---------- the targets ----------
@dataclass(frozen=True)
class TimedTarget:
    name: str
    price: float
    basis: str                   # where the price came from
    due_ms: int | None
    reward_r: float
    due_from: str = ""           # where the time came from

    @property
    def when(self) -> str:
        return fmt_when(self.due_ms)

    @property
    def dated_by_fib(self) -> bool:
        return self.due_from.startswith("Fibonacci")


def timed_targets(state: dict, side: str, entry: float, stop: float, timeframe: str, atr: float,
                  fib=None, now_ms: int | None = None) -> list[TimedTarget]:
    """Three targets: the price from the volatility bands, the time from the Fibonacci time zones.

    A band that sits behind the entry is no target at all, so it falls back to a multiple of range; the
    fallback says so in its basis rather than pretending the band produced it."""
    long = side == LONG
    bands = ((state.get("volatility") or {}).get(timeframe) or {}).get("bands") or {}
    risk = abs(entry - stop) or atr
    # only windows still ahead of us: a target dated in the past is worse than an undated one
    zones = [z for z in (list(getattr(fib, "upcoming", []) or []) if fib is not None else [])
             if now_ms is None or z.timestamp_ms > now_ms]

    out: list[TimedTarget] = []
    for i, sigma in enumerate(TARGET_SIGMAS):
        key = f"sigma{int(sigma)}_{'up' if long else 'dn'}"
        band = bands.get(key)
        price, basis = None, ""
        try:
            if band is not None and ((float(band) > entry) if long else (float(band) < entry)):
                price, basis = float(band), f"the {int(sigma)} sigma band on the {timeframe}"
        except (TypeError, ValueError):
            price = None
        if price is None:
            price = entry + sigma * atr if long else entry - sigma * atr
            basis = f"{sigma:g} x the {timeframe} range (the {int(sigma)} sigma band is behind the entry)"
        if i < len(zones):
            due, due_from = zones[i].timestamp_ms, "Fibonacci time window"
        elif now_ms is not None:
            # the candle a trader would be watching anyway, so the target is still monitorable
            due, due_from = nth_close(now_ms, timeframe, i + 1), f"the {_ordinal(i + 1)} {timeframe} close"
        else:
            due, due_from = None, ""
        out.append(TimedTarget(f"TP{i + 1}", round(price, 2), basis, due,
                               round(abs(price - entry) / risk, 2) if risk else 0.0, due_from))
    return out


# ---------- the pullback band ----------
@dataclass(frozen=True)
class Pullback:
    ordinary: float          # the median adverse excursion, in price
    edge: float              # the 80th percentile: still normal
    breaking: float          # the 95th: past this the thesis is in question
    horizon_bars: int
    timeframe: str
    available: bool = True
    note: str = ""

    def levels(self, side: str, entry: float) -> dict[str, float]:
        sign = -1.0 if side == LONG else 1.0
        return {"ordinary": round(entry + sign * self.ordinary, 2),
                "edge": round(entry + sign * self.edge, 2),
                "breaking": round(entry + sign * self.breaking, 2)}


def pullback_band(df: pd.DataFrame, side: str, horizon_bars: int = 12, lookback: int = 300,
                  timeframe: str = "") -> Pullback:
    """How far this market normally goes against a position before continuing.

    Measured from the candles: for every bar in the lookback, the deepest adverse excursion over the next
    `horizon_bars`. The percentiles of that distribution are what "a normal pullback" means here, rather
    than a number chosen because it sounds reasonable."""
    if df is None or len(df) < horizon_bars + 20:
        return Pullback(0.0, 0.0, 0.0, horizon_bars, timeframe, False,
                        f"fewer than {horizon_bars + 20} {timeframe} candles")
    d = df.tail(lookback).reset_index(drop=True)
    closes = d["close"].to_numpy(dtype=float)
    lows, highs = d["low"].to_numpy(dtype=float), d["high"].to_numpy(dtype=float)
    excursions = []
    for i in range(len(closes) - horizon_bars):
        window_low, window_high = lows[i + 1:i + 1 + horizon_bars], highs[i + 1:i + 1 + horizon_bars]
        adverse = closes[i] - float(window_low.min()) if side == LONG else float(window_high.max()) - closes[i]
        excursions.append(max(0.0, adverse))
    if not excursions:
        return Pullback(0.0, 0.0, 0.0, horizon_bars, timeframe, False, "no complete windows")
    p50, p80, p95 = (float(np.percentile(excursions, p)) for p in PULLBACK_PCTILES)
    return Pullback(round(p50, 2), round(p80, 2), round(p95, 2), horizon_bars, timeframe)


# ---------- the schedule ----------
def _ordinal(n: int) -> str:
    return {1: "next", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}.get(n, f"{n}th")


def nth_close(now_ms: int, timeframe: str, n: int) -> int:
    """The n-th close of that candle from now, n=1 being the next one."""
    step = WEEK_MS if timeframe == "1w" else TF_MS.get(timeframe)
    if step is None:
        raise ValueError(f"unknown timeframe {timeframe!r}")
    return next_close(now_ms, timeframe) + (max(1, n) - 1) * step


def next_close(now_ms: int, timeframe: str) -> int:
    """The next close of that candle, UTC-aligned the same way the war rooms anchor their windows."""
    if timeframe == "1w":
        return now_ms - (now_ms - MONDAY_OFFSET_MS) % WEEK_MS + WEEK_MS
    step = TF_MS.get(timeframe)
    if step is None:
        raise ValueError(f"unknown timeframe {timeframe!r}")
    return now_ms - now_ms % step + step


@dataclass(frozen=True)
class Checkpoint:
    timeframe: str
    at_ms: int
    reason: str

    @property
    def when(self) -> str:
        return fmt_when(self.at_ms)


def monitoring_schedule(now_ms: int, timeframes, targets: list[TimedTarget] | None = None,
                        limit: int = 5) -> list[Checkpoint]:
    """The exact candle closes to look at, soonest first, plus any target's own due time."""
    out = [Checkpoint(tf, next_close(now_ms, tf), f"the {tf} close") for tf in timeframes if tf in TF_MS]
    seen = {c.at_ms for c in out}
    for t in targets or []:
        # only the windows that add a time the candle closes do not already cover, and named after what
        # actually dated them rather than after the Fibonacci zones in every case
        if t.due_ms and t.due_ms > now_ms and t.due_ms not in seen:
            seen.add(t.due_ms)
            out.append(Checkpoint("", t.due_ms, f"{t.name} is due ({t.due_from})" if t.due_from
                                  else f"{t.name} is due"))
    return sorted(out, key=lambda c: c.at_ms)[:limit]


# ---------- live position arithmetic ----------
@dataclass(frozen=True)
class Progress:
    price: float
    pnl_pct: float
    pnl_r: float
    to_stop_pct: float
    hit: tuple[str, ...] = ()        # targets price has already reached
    stopped: bool = False
    distances: dict[str, float] = field(default_factory=dict)


def progress(side: str, entry: float, stop: float, targets: list[TimedTarget], price: float) -> Progress:
    """Where the position stands now. Refresh updates this and nothing else."""
    long = side == LONG
    risk = abs(entry - stop) or 1e-9
    move = (price - entry) if long else (entry - price)
    hit = tuple(t.name for t in targets if ((price >= t.price) if long else (price <= t.price)))
    stopped = (price <= stop) if long else (price >= stop)
    return Progress(price=price,
                    pnl_pct=round(move / entry * 100, 3) if entry else 0.0,
                    pnl_r=round(move / risk, 2),
                    to_stop_pct=round(abs(price - stop) / price * 100, 3) if price else 0.0,
                    hit=hit, stopped=stopped,
                    distances={t.name: round(abs(t.price - price) / price * 100, 3) for t in targets})
