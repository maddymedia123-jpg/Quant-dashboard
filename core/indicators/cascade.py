"""The 3-tier timeframe cascade (Hammad's model): bias above, anchor in the middle, an engine below.

    [ BIAS ]    one candle size above the anchor - the vector the anchor candle tends to follow
    [ ANCHOR ]  the candle the desk's summary is pinned to - the battleground
    [ ENGINE ]  the two timeframes below it - where a change shows first

A desk reads three states off the cascade, mirrored for a bearish bias:

    A  Full confluence   bias bullish, engine bullish        -> high-conviction expansion
    B  Pullback          bias bullish, engine bearish/mixed  -> a retracement inside the trend: do not
                                                                short; the anchor's pullback bands say
                                                                how deep is still normal
    C  Control shift     bias bullish, engine bearish AND    -> the anchor candle is breaking down
                         the slower engine timeframe has        against its bias
                         closed twice below VWAP

and a fourth when there is no vector to align with: N, no bias.

Everything here is computed from candles, deterministically, so the panel a person reads and the state
the war-room judges are handed cannot disagree. It describes conditions; it does not promise outcomes -
State A is "high conviction", never "guaranteed".

VWAP is anchored to the open of the current bias candle and built from the finest engine timeframe: for
Live Recon that is today's VWAP from 15m candles, the standard intraday reference. Closes are taken from
completed candles only - the newest row of every frame is the candle still forming."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from core.indicators import smc
from core.indicators.ema import ema_stack
from core.trades import LONG, SHORT, pullback_band

BULLISH, BEARISH, NEUTRAL = "bullish", "bearish", "neutral"
OPPOSITE = {BULLISH: BEARISH, BEARISH: BULLISH}
ACCEPTANCE_CLOSES = 2      # consecutive completed closes beyond VWAP that count as acceptance
# Fewer candles than this and a timeframe has no read. Low enough for monthly and quarterly candles, which
# the free feeds hold only a couple of years of: market structure needs a dozen candles, and the EMA
# stack simply reports MIXED when its longer averages cannot be formed.
MIN_CANDLES = 12

STATE_TITLES = {
    "A": "Full confluence",
    "B": "Pullback",
    "C": "Control shift",
    "N": "No higher-timeframe bias",
    "?": "Not enough data",
}


@dataclass(frozen=True)
class TimeframeRead:
    timeframe: str
    role: str                  # bias, anchor, engine
    direction: str             # bullish, bearish, neutral
    ema: str                   # BULL, BEAR, MIXED
    structure: str             # bullish, bearish, unclear
    available: bool = True


@dataclass(frozen=True)
class Cascade:
    state: str
    title: str
    bias: TimeframeRead | None
    anchor: TimeframeRead | None
    engine: tuple[TimeframeRead, ...]
    engine_direction: str
    vwap: float | None
    vwap_anchor_ms: int | None
    closes_beyond_vwap: int
    playbook: str
    pullback: dict = field(default_factory=dict)

    def to_state(self) -> dict:
        """The JSON-safe form handed to the war-room judges."""
        def tf(r):
            return None if r is None else {"timeframe": r.timeframe, "role": r.role, "direction": r.direction,
                                           "ema_stack": r.ema, "structure": r.structure}
        return {"state": self.state, "title": self.title, "bias": tf(self.bias), "anchor": tf(self.anchor),
                "engine": [tf(r) for r in self.engine], "engine_direction": self.engine_direction,
                "vwap": None if self.vwap is None else round(self.vwap, 2),
                "closes_beyond_vwap": self.closes_beyond_vwap, "playbook": self.playbook,
                "pullback_bands": self.pullback}


def direction(df: pd.DataFrame | None) -> tuple[str, str, str]:
    """(direction, EMA stack, structure bias) for one timeframe.

    The EMA stack and market structure each vote. Agreement, or one voting while the other abstains, gives
    a direction; disagreement gives neutral - a timeframe whose trend and structure point opposite ways
    has no vector."""
    if df is None or len(df) < MIN_CANDLES:
        return NEUTRAL, "MIXED", "unclear"
    # the averages the history can form: a full 21/50/100/200 stack needs 200 candles, which monthly and
    # quarterly series never have - so a short series is read on the averages it does have
    periods = tuple(n for n in (21, 50, 100, 200) if len(df) >= n)
    ema = ema_stack(df, periods).state if periods else "MIXED"
    st = smc.structure(df)
    structure = st.bias if st.available else "unclear"
    votes = {BULLISH: 0, BEARISH: 0}
    if ema == "BULL":
        votes[BULLISH] += 1
    elif ema == "BEAR":
        votes[BEARISH] += 1
    if structure in votes:
        votes[structure] += 1
    if votes[BULLISH] and not votes[BEARISH]:
        return BULLISH, ema, structure
    if votes[BEARISH] and not votes[BULLISH]:
        return BEARISH, ema, structure
    return NEUTRAL, ema, structure


def anchored_vwap(df: pd.DataFrame | None, start_ms: int | None) -> float | None:
    """Volume-weighted average of the typical price from `start_ms` onward. None without volume."""
    if df is None or start_ms is None or df.empty:
        return None
    d = df[df["timestamp"] >= start_ms]
    vol = d["volume"].astype(float)
    if d.empty or float(vol.sum()) <= 0:
        return None
    typical = (d["high"].astype(float) + d["low"].astype(float) + d["close"].astype(float)) / 3.0
    return float((typical * vol).sum() / vol.sum())


def _closes_beyond(df: pd.DataFrame | None, vwap: float | None, side: str, since_ms: int | None) -> int:
    """Consecutive completed closes on the far side of VWAP, counting back from the latest completed one,
    and only among candles since VWAP's own start - a close from before it was not measured against it."""
    if df is None or vwap is None or since_ms is None or len(df) < 2:
        return 0
    closed = df.iloc[:-1]                                           # the last row is still forming
    closed = closed[closed["timestamp"] >= since_ms]
    n = 0
    for c in reversed(closed["close"].astype(float).to_numpy()):
        if (side == BEARISH and c < vwap) or (side == BULLISH and c > vwap):
            n += 1
        else:
            break
    return n


def classify(frames: dict[str, pd.DataFrame], bias_tf: str, anchor_tf: str,
             engine_tfs: tuple[str, ...]) -> Cascade:
    """Read the cascade for one desk."""
    def read(tf: str, role: str) -> TimeframeRead:
        df = frames.get(tf)
        d, ema, st = direction(df)
        return TimeframeRead(tf, role, d, ema, st, available=df is not None and len(df) >= MIN_CANDLES)

    bias, anchor = read(bias_tf, "bias"), read(anchor_tf, "anchor")
    engine = tuple(read(tf, "engine") for tf in engine_tfs)
    dirs = {r.direction for r in engine}
    engine_dir = dirs.pop() if len(dirs) == 1 else "mixed"

    bias_df = frames.get(bias_tf)
    fine = frames.get(engine_tfs[0]) if engine_tfs else None
    start = int(bias_df["timestamp"].iloc[-1]) if bias_df is not None and len(bias_df) else None
    vwap = anchored_vwap(fine, start)

    if not bias.available or not anchor.available:
        short = [r.timeframe for r in (bias, anchor) if not r.available]
        return Cascade("?", STATE_TITLES["?"], bias, anchor, engine, engine_dir, vwap, start, 0,
                       f"Fewer than {MIN_CANDLES} {' and '.join(short)} candles in the feed, so the cascade "
                       "cannot be read yet.")
    if bias.direction == NEUTRAL:
        return Cascade("N", STATE_TITLES["N"], bias, anchor, engine, engine_dir, vwap, start, 0,
                       f"The {bias_tf} has no vector to align with, so the {anchor_tf} is read on its own "
                       "levels: fade its range edges rather than trading breaks.")

    against = OPPOSITE[bias.direction]
    slow = frames.get(engine_tfs[-1]) if engine_tfs else None
    beyond = _closes_beyond(slow, vwap, against, start)
    side = LONG if bias.direction == BULLISH else SHORT
    word = "up" if bias.direction == BULLISH else "down"

    if engine_dir == bias.direction:
        return Cascade("A", STATE_TITLES["A"], bias, anchor, engine, engine_dir, vwap, start, beyond,
                       f"The {bias_tf} vector and the {' and '.join(engine_tfs)} engine both point {word}: "
                       f"high-conviction expansion of the {anchor_tf} toward the {bias_tf} targets.")

    bands = {}
    anchor_df = frames.get(anchor_tf)
    pb = pullback_band(anchor_df, side, horizon_bars=6, timeframe=anchor_tf)
    if pb.available and anchor_df is not None and len(anchor_df):
        px = float(anchor_df["close"].iloc[-1])
        sign = -1 if side == LONG else 1
        bands = {"timeframe": anchor_tf, "p50": round(px + sign * pb.ordinary, 2),
                 "p80": round(px + sign * pb.edge, 2), "p95": round(px + sign * pb.breaking, 2)}

    if engine_dir == against and beyond >= ACCEPTANCE_CLOSES:
        return Cascade("C", STATE_TITLES["C"], bias, anchor, engine, engine_dir, vwap, start, beyond,
                       f"The {engine_tfs[-1]} has closed {beyond} times {'below' if against == BEARISH else 'above'} "
                       f"VWAP against the {bias_tf} vector: control is shifting. The {anchor_tf} candle is "
                       f"breaking {'down' if against == BEARISH else 'up'}, and the {bias_tf} trend begins to decay "
                       "if it holds.", bands)
    return Cascade("B", STATE_TITLES["B"], bias, anchor, engine, engine_dir, vwap, start, beyond,
                   f"The {bias_tf} vector points {word} but the engine is {engine_dir}: a {anchor_tf} pullback "
                   f"inside the trend. Do not trade against the {bias_tf}; wait for the engine to absorb the move "
                   "at VWAP or the point of control. The bands are how deep a pullback normally runs.", bands)
