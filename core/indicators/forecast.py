"""The forecast path: one most-likely candle path per desk, with the odds that come with it.

His Section 8 asks for "a single, highly accurate institutional path" of simulated candles, built from
Hurst momentum, volatility envelopes and the gamma regime, "backed by probability metrics, exact price
levels, and dual timestamps". Agreed with Ezzam (17 Oct): no path is "zero guesswork", so the path is
drawn and labelled as the median of simulated paths, and every number beside it is a probability.

The model, kept small enough to test:

* Per-bar log returns are normal with a drift and a volatility.
* Volatility is the standard deviation of the last VOL_WINDOW closes' returns.
* Drift is the Hurst momentum term: the average return over the last horizon's worth of bars, times
  (2H - 1). A persistent series (H > 0.5) carries its recent move forward, a mean-reverting one
  (H < 0.5) leans against it, and a random walk (H = 0.5) gets no drift at all. It is capped at
  MAX_DRIFT_SIGMA of one bar's volatility, so a single hot hour cannot draw a runaway path.
* The gamma regime scales the volatility on the desks whose horizon the listed options cover (1h and
  4h): dealers long gamma sell rallies and buy dips (narrower), short gamma chase moves (wider).
  There is no gamma history to test that against, and the card says so.

From that, PATHS paths are simulated, each bar in SUB_STEPS pieces so the bar has a real high and low.
The drawn candle for each step is the median open, close, high and low across the paths. Beside it:
the chance of finishing above the current price, the 16th-84th percentile range (68%), and for each
nearby level the chance that price trades there at all before the horizon ends.

`track_record` replays the same model (without gamma) over the desk's own history and reports how often
the 68% range held and how often the drift's sign was right, against the base rate - so the reader can
see whether the path has earned any trust on this timeframe."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from core.indicators.volatility import hurst

PATHS = 4000
SUB_STEPS = 6
VOL_WINDOW = 50
HURST_WINDOW = 200
MAX_DRIFT_SIGMA = 0.25          # the drift never exceeds a quarter of one bar's sigma
GAMMA_SCALE = {"positive": 0.85, "negative": 1.15}
GAMMA_TIMEFRAMES = ("1h", "4h")  # the listed expiries span days, not weeks
BAND_LOW, BAND_HIGH = 16.0, 84.0
SEED = 7                         # a fixed seed: the same inputs draw the same path on every refresh


@dataclass
class ForecastStep:
    time_ms: int
    open: float
    high: float
    low: float
    close: float
    band_low: float
    band_high: float
    p_above_now: float


@dataclass
class LevelOdds:
    label: str
    price: float
    side: str            # above | below
    p_touch: float


@dataclass
class Forecast:
    available: bool
    reason: str = ""
    price: float | None = None
    steps: list[ForecastStep] = field(default_factory=list)
    end_median: float | None = None
    end_low: float | None = None
    end_high: float | None = None
    p_up: float | None = None
    levels: list[LevelOdds] = field(default_factory=list)
    hurst: float | None = None
    drift_per_bar: float | None = None
    sigma_per_bar: float | None = None
    gamma_regime: str | None = None
    gamma_scale: float = 1.0
    bar_ms: int = 0


def _returns(closes) -> np.ndarray:
    c = np.asarray(closes, dtype=float)
    c = c[np.isfinite(c) & (c > 0)]
    return np.diff(np.log(c))


def drift_and_sigma(closes, horizon: int) -> tuple[float, float, float | None] | None:
    """(drift per bar, sigma per bar, Hurst) from the closes up to now, or None without enough history."""
    r = _returns(closes)
    if len(r) < max(VOL_WINDOW, horizon) + 1:
        return None
    sigma = float(np.std(r[-VOL_WINDOW:], ddof=1))
    if not math.isfinite(sigma) or sigma <= 0:
        return None
    h = hurst(np.asarray(closes, dtype=float)[-HURST_WINDOW - 1:])
    momentum = float(np.mean(r[-max(horizon, 5):]))
    drift = 0.0 if h is None else (2.0 * h - 1.0) * momentum
    cap = MAX_DRIFT_SIGMA * sigma
    return max(-cap, min(cap, drift)), sigma, h


def simulate(price: float, drift: float, sigma: float, steps: int, paths: int = PATHS,
             sub: int = SUB_STEPS, seed: int = SEED) -> np.ndarray:
    """Prices along every path at every sub-step, shape (paths, steps * sub + 1), starting at `price`."""
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((paths, steps * sub)) * (sigma / math.sqrt(sub)) + drift / sub
    log_path = np.concatenate([np.zeros((paths, 1)), np.cumsum(shocks, axis=1)], axis=1)
    return price * np.exp(log_path)


def build(closes, price: float, last_ms: int, bar_ms: int, horizon: int, timeframe: str,
          levels: list[tuple[str, float]] | None = None, gamma_regime: str | None = None) -> Forecast:
    """The path, its range and the odds at each level, for `horizon` bars after the last candle."""
    if not price or horizon < 1:
        return Forecast(False, "no price")
    est = drift_and_sigma(closes, horizon)
    if est is None:
        return Forecast(False, f"needs at least {max(VOL_WINDOW, horizon) + 2} candles")
    drift, sigma, h = est
    scale = GAMMA_SCALE.get(gamma_regime or "", 1.0) if timeframe in GAMMA_TIMEFRAMES else 1.0
    sigma_used = sigma * scale
    prices = simulate(price, drift, sigma_used, horizon)

    steps: list[ForecastStep] = []
    for k in range(horizon):
        seg = prices[:, k * SUB_STEPS:(k + 1) * SUB_STEPS + 1]
        opens, closes_k = seg[:, 0], seg[:, -1]
        o, c = float(np.median(opens)), float(np.median(closes_k))
        hi, lo = float(np.median(seg.max(axis=1))), float(np.median(seg.min(axis=1)))
        steps.append(ForecastStep(
            time_ms=last_ms + (k + 1) * bar_ms, open=o, close=c,
            high=max(hi, o, c), low=min(lo, o, c),
            band_low=float(np.percentile(closes_k, BAND_LOW)), band_high=float(np.percentile(closes_k, BAND_HIGH)),
            p_above_now=float(np.mean(closes_k > price))))

    end = prices[:, -1]
    running_max, running_min = prices.max(axis=1), prices.min(axis=1)
    odds = []
    for label, lvl in levels or []:
        if lvl is None or not math.isfinite(lvl) or lvl <= 0 or abs(lvl / price - 1) < 1e-6:
            continue
        side = "above" if lvl > price else "below"
        hit = running_max >= lvl if side == "above" else running_min <= lvl
        odds.append(LevelOdds(label, float(lvl), side, float(np.mean(hit))))
    # one line per price: a floor that is also the put wall is one level with both names
    merged: dict[float, LevelOdds] = {}
    for o in odds:
        key = round(o.price, 2)
        if key in merged:
            merged[key].label += f" · {o.label}"
        else:
            merged[key] = o
    odds = sorted(merged.values(), key=lambda o: o.price, reverse=True)

    return Forecast(True, price=float(price), steps=steps, end_median=float(np.median(end)),
                    end_low=float(np.percentile(end, BAND_LOW)), end_high=float(np.percentile(end, BAND_HIGH)),
                    p_up=float(np.mean(end > price)), levels=odds, hurst=h, drift_per_bar=drift,
                    sigma_per_bar=sigma_used, gamma_regime=gamma_regime if scale != 1.0 else None, gamma_scale=scale,
                    bar_ms=bar_ms)


@dataclass
class TrackRecord:
    origins: int
    band_hit: float | None          # share of horizons that closed inside the 68% range (aim: 0.68)
    called: int                     # origins where the drift picked a side
    direction_hit: float | None     # of those, the share where price went that way
    base_up: float | None           # share of all horizons that closed higher, for comparison


def track_record(closes, horizon: int, origins: int = 300) -> TrackRecord:
    """Replay the model (volatility and Hurst drift; no gamma, which has no history) at each of the last
    `origins` closes that have a full horizon after them, and score it against what happened."""
    c = np.asarray(closes, dtype=float)
    first = max(HURST_WINDOW + 1, VOL_WINDOW + horizon + 1)
    last = len(c) - horizon - 1
    idx = range(max(first, last - origins + 1), last + 1)
    inside = up = called = right = n = 0
    z = 0.994457883                  # the normal quantile at 84%: the 68% band is +/- z sigma
    for i in idx:
        est = drift_and_sigma(c[: i + 1], horizon)
        if est is None:
            continue
        drift, sigma, _h = est
        actual = math.log(c[i + horizon] / c[i])
        centre, half = drift * horizon, z * sigma * math.sqrt(horizon)
        n += 1
        inside += centre - half <= actual <= centre + half
        up += actual > 0
        if abs(drift) > 1e-12:
            called += 1
            right += (actual > 0) == (drift > 0)
    if n == 0:
        return TrackRecord(0, None, 0, None, None)
    return TrackRecord(n, inside / n, called, right / called if called else None, up / n)


def desk_forecast(a, gamma: dict | None = None, history=None) -> tuple[Forecast, TrackRecord]:
    """The forecast and its track record for one desk's CategoryAnalysis, on its own chart timeframe and
    horizon. Levels with odds: the two nearest floors and ceilings, and on the 1h/4h desks the gamma walls
    and the zero-gamma flip. `history` is the desk timeframe's full frame (the chart keeps only its last
    few hundred candles, too few to replay); without it the chart's own candles are used."""
    from core.config import CATEGORIES, TF_MINUTES

    df = a.chart_df
    horizon = CATEGORIES[a.key].horizon_bars
    full = history if history is not None and len(history) >= len(df) else df
    closes = full["close"].astype(float).to_numpy()
    supports = sorted((lv for lv in a.levels if lv.kind == "support"), key=lambda lv: -lv.price)[:2]
    ceilings = sorted((lv for lv in a.levels if lv.kind == "resistance"), key=lambda lv: lv.price)[:2]
    levels = [(f"floor ({lv.touches}x)", lv.price) for lv in supports]
    levels += [(f"ceiling ({lv.touches}x)", lv.price) for lv in ceilings]
    regime = None
    if gamma and gamma.get("available") and a.chart_tf in GAMMA_TIMEFRAMES:
        regime = gamma.get("regime")
        for key, label in (("call_wall", "call wall"), ("put_wall", "put wall"), ("zero_gamma_level", "gamma flip")):
            if gamma.get(key):
                levels.append((label, float(gamma[key])))
    f = build(closes, float(a.price), int(full["timestamp"].iloc[-1]), TF_MINUTES[a.chart_tf] * 60_000, horizon,
              a.chart_tf, levels, regime)
    # the record replays closed candles only: the last one is still forming
    return f, track_record(closes[:-1], horizon, origins=500)
