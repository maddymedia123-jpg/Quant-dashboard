"""Quant Forecast: the day's plan for BTC, computed (Hammad's AI Quant Master Agent, steps 1-4).

Every number on the plan comes from here, never from a model: the AI's later job (step 5) is to pick
between these scenarios and put them in words.

The day runs from DAY_START_HOUR_UTC (00:00 UTC, 05:00 PKT) for 24 hours. The morning plan is built at
the start of the day; a plan that is invalidated is replaced by a new one for the rest of the day, at
most until LOCK_AFTER_INVALIDATIONS have failed, after which the day is locked: stand aside.

How the plan is made
--------------------
* Volatility, hour by hour: GARCH(1,1) on hourly returns with the hour-of-day pattern taken out
  (core.indicators.garch), so each hour of the plan has its own expected size of move.
* Chop and expansion windows: the hour-of-day pattern itself, measured on the last 60 days - hours at
  under CHOP_X of an average hour's volatility are chop, over EXPAND_X expansion. Measured, not guessed.
* The lean, a score from -1 to +1 built from five reads, each also -1 to +1: the last 24 hours' move
  scaled by its volatility and by Hurst memory (does a move here tend to carry on?), where price sits
  against the 3-day value area, the 24-hour taker imbalance, open interest rising or falling with the
  move, and funding stretched past its usual carry (read against the crowd).
* The regime: NEUTRAL / RANGE-BOUND when the lean is weak (or volatility is compressed and the lean only
  moderate); otherwise a bullish or bearish expansion. A neutral day gets no direction, only a box.
* The path: PATHS simulated days with each hour's volatility, the lean's drift spread over the normal
  and expansion hours only - never through a chop window, which is drawn flat.
* From the simulation: the median path and its 68% envelope, the projected high and low (the median of
  each path's high and low) with the hour they most likely print, the odds of trading at each level,
  and the invalidation level: the price only 10% of simulated days reach against the lean.
* Estimated liquidation zones: low-volume nodes (air pockets) on the 1h volume profile where open
  interest surged while price traded through them. An estimate from one venue's open interest, not
  exchange liquidation data, and labelled so.

Scoring: every plan is checked against the candles when its window ends (`score_plan`), and the same
rules replayed on past days (`backtest`, price-only: taker flow and open interest have no history this
deep) give a record from the first day, kept apart from the live one."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from core.indicators import garch
from core.indicators.liquidity import low_volume_nodes, volume_profile
from core.indicators.volatility import atr, hurst

HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS
FOUR_H_MS = 4 * HOUR_MS
DAY_START_HOUR_UTC = 0
LOCK_AFTER_INVALIDATIONS = 3     # two re-forecasts a day; the third failed plan locks the rest of it (Hammad, 28 Oct)
MISSING_HOURS_TOLERATED = 2      # a day is still scored with this many hourly candles missing from the feed

CHOP_X, EXPAND_X = 0.85, 1.15
NEUTRAL_SCORE = 0.20             # a lean weaker than this is no lean
COMPRESSED_SCORE = 0.35          # ...and when volatility is compressed, neither is one weaker than this
COMPRESSION_X = 0.80             # next hour's volatility under 80% of its long-run level
DRIFT_K = 0.5                    # a full-strength lean moves the median by half a day's sigma
INVALIDATION_PCT = 10.0          # invalidation: the price only 10% of simulated days reach against the lean
BAND_LOW, BAND_HIGH = 16.0, 84.0
Z84 = 0.994457883
PATHS, SUB, SEED = 3000, 4, 11
VALUE_HOURS = 72                 # the value area of the last three days
LVN_HOURS = 192                  # air pockets on the last eight days
OI_SURGE_Z = 2.0                 # an hourly open interest rise two standard deviations above the usual
HIGH_LOW_TOL = 0.25              # a projected high or low is right within a quarter of the projected range
TIME_TOL_MS = 2 * HOUR_MS
BACKTEST_DAYS = 60
MACRO_SIGMA_X = 1.5              # an hour inside a high-impact release window: a chop box, but a loud one

WEIGHTS = {"momentum": 0.35, "value": 0.20, "flow": 0.25, "open_interest": 0.10, "funding": 0.10}
PRICE_ONLY = ("momentum", "value")
NORMAL_FUNDING = 0.0001          # the usual 8h carry; only the stretch past it is read
FUNDING_SCALE = 0.0003


def day_start(ms: int) -> int:
    return (ms - DAY_START_HOUR_UTC * HOUR_MS) // DAY_MS * DAY_MS + DAY_START_HOUR_UTC * HOUR_MS


def closed_hours(h1: pd.DataFrame, now_ms: int) -> pd.DataFrame:
    """The 1h candles that have finished by now."""
    return h1[h1["timestamp"].astype("int64") + HOUR_MS <= now_ms].reset_index(drop=True)


def _clip(x: float) -> float:
    return max(-1.0, min(1.0, float(x)))


def lean_parts(closed: pd.DataFrame, price: float, flow: pd.DataFrame | None, now_ms: int,
               price_only: bool = False) -> dict[str, dict]:
    """Each read that goes into the lean: its value (-1 to +1) and one line saying what it saw."""
    c = closed["close"].astype(float).to_numpy()
    r = garch.log_returns(c)
    out: dict[str, dict] = {}
    if len(c) >= 25 and len(r) >= 168:
        sd = float(np.std(r[-168:], ddof=1))
        r24 = math.log(price / c[-25])       # the close 24 full hours before the last closed hour
        z = r24 / (sd * math.sqrt(24)) if sd > 0 else 0.0
        h = hurst(c[-201:])
        carry = _clip(0.5 + ((h or 0.5) - 0.5) * 5)
        out["momentum"] = {"value": round(math.tanh(z) * max(carry, 0.0), 3),
                           "note": f"24h move {r24 * 100:+.2f}% ({z:+.1f} sigma), Hurst "
                                   f"{'n/a' if h is None else f'{h:.2f}'}"}
    vp = volume_profile(closed.tail(VALUE_HOURS), lookback=VALUE_HOURS)
    if vp.available and vp.value_area_high > vp.value_area_low:
        pos = (price - vp.poc) / (vp.value_area_high - vp.value_area_low)
        out["value"] = {"value": round(_clip(pos), 3),
                        "note": f"price {vp.position} (POC ${vp.poc:,.0f}, value ${vp.value_area_low:,.0f}-"
                                f"${vp.value_area_high:,.0f})"}
    if price_only or flow is None or flow.empty:
        return out
    f = flow[flow["timestamp"].astype("int64") + HOUR_MS <= now_ms].tail(24)
    if len(f) >= 12:
        d = f["delta"].astype(float)
        gross = float(d.abs().sum())
        if gross > 0:
            imb = float(d.sum()) / gross
            out["flow"] = {"value": round(_clip(imb * 2), 3),
                           "note": f"24h taker delta {float(d.sum()):+,.0f} BTC "
                                   f"({'buyers' if imb > 0 else 'sellers'} paying)"}
        oi = f["oi_btc"].astype(float).dropna() if "oi_btc" in f else pd.Series(dtype=float)
        if len(oi) >= 12 and float(oi.iloc[0]) > 0 and "momentum" in out:
            oi_pct = (float(oi.iloc[-1]) / float(oi.iloc[0]) - 1) * 100
            move = math.log(price / c[-25]) if len(c) >= 25 else 0.0
            out["open_interest"] = {"value": round(float(np.sign(move)) * math.tanh(oi_pct / 3), 3),
                                    "note": f"open interest {oi_pct:+.1f}% over 24h "
                                            f"{'with' if (oi_pct > 0) else 'against'} the move"}
        fr = f["funding_rate"].astype(float).dropna() if "funding_rate" in f else pd.Series(dtype=float)
        if len(fr):
            rate = float(fr.iloc[-1])
            out["funding"] = {"value": round(-math.tanh((rate - NORMAL_FUNDING) / FUNDING_SCALE), 3),
                              "note": f"funding {rate * 100:.4f}% per 8h (read against the crowd)"}
    return out


def lean_score(parts: dict[str, dict]) -> float:
    used = {k: WEIGHTS[k] for k in parts if k in WEIGHTS}
    total = sum(used.values())
    return 0.0 if not total else sum(parts[k]["value"] * w for k, w in used.items()) / total


def _phase(mult: float) -> str:
    return "chop" if mult < CHOP_X else "expansion" if mult > EXPAND_X else "normal"


def estimated_liquidation_zones(closed: pd.DataFrame, flow: pd.DataFrame | None) -> list[dict]:
    """Air pockets on the 1h profile, each marked when open interest surged on an hour that traded
    through it (Hammad: "when OI surges into an LVN, that is where the liquidations sit")."""
    window = closed.tail(LVN_HOURS)
    nodes = low_volume_nodes(window, lookback=LVN_HOURS)
    surges: list[tuple[float, float]] = []
    if flow is not None and not flow.empty and "oi_btc" in flow:
        f = flow[["timestamp", "oi_btc"]].dropna()
        d = f["oi_btc"].astype(float).diff()
        sd = float(d.std()) if len(d) > 10 else 0.0
        if sd > 0:
            hot = f.loc[d > OI_SURGE_Z * sd, "timestamp"].astype("int64")
            bars = window.set_index(window["timestamp"].astype("int64"))
            surges = [(float(bars.at[t, "low"]), float(bars.at[t, "high"])) for t in hot if t in bars.index]
    return [{"low": round(n.low, 2), "high": round(n.high, 2),
             "oi_surge": any(lo <= n.high and hi >= n.low for lo, hi in surges)} for n in nodes]


def _windows(times: list[int], phases: list[str], sigmas: list[float], path: list[float], start_ms: int,
             origin: float) -> list[dict]:
    """Contiguous chop and expansion hours, each with its box: the median path where it starts, +- the
    68% move of that window's hours alone."""
    out: list[dict] = []
    i = 0
    while i < len(times):
        if phases[i] == "normal":
            i += 1
            continue
        j = i
        while j + 1 < len(times) and phases[j + 1] == phases[i]:
            j += 1
        begin = start_ms if i == 0 else times[i - 1]
        centre = origin if i == 0 else path[i - 1]
        half = Z84 * math.sqrt(sum(s * s for s in sigmas[i:j + 1]))
        out.append({"phase": phases[i], "start_ms": int(begin), "end_ms": int(times[j]),
                    "upper": round(centre * math.exp(half), 2), "lower": round(centre * math.exp(-half), 2)})
        i = j + 1
    return out


def build_plan(h1: pd.DataFrame, now_ms: int, end_ms: int, price: float, *, version: int = 0,
               flow: pd.DataFrame | None = None, gamma: dict | None = None, price_only: bool = False,
               paths: int = PATHS, force_neutral: bool = False, forced_windows: list | None = None) -> dict:
    """The plan from now to end_ms, as a JSON-safe dict. {"available": False, "reason": ...} without
    enough history.

    The pre-flight (core.qf_preflight) can override two things. `force_neutral` (an engineered trap is
    active on the 4h) keeps the lean's parts for the record but scores the day 0.0: the regime is neutral,
    there is no drift, and invalidation has both sides. `forced_windows` ([(start_ms, end_ms, title)], a
    high-impact release inside the day) marks the hours they touch as chop, so no drift is carried through
    them and the chart draws one grey box with the plan's own chop hours; their sigma is scaled by
    MACRO_SIGMA_X, because a print is a chop box in direction only - the range it trades is wider, not
    narrower, than the hour's usual, and the band and invalidation should say so."""
    closed = closed_hours(h1, now_ms)
    need = garch.FIT_HOURS + 50
    if len(closed) < need or not price or end_ms <= now_ms:
        return {"available": False, "reason": f"needs {need} closed hourly candles (has {len(closed)})"
                if len(closed) < need else "no price or no time left in the day"}
    ts = closed["timestamp"].astype("int64").to_numpy()
    closes = closed["close"].astype(float).to_numpy()
    profile = garch.hourly_profile(ts, closes)
    fit = garch.fit(garch.deseasonalise(ts, closes, profile))
    if fit is None:
        return {"available": False, "reason": "volatility could not be fitted (flat or missing prices)"}

    hour0 = now_ms - now_ms % HOUR_MS
    n = max(1, math.ceil((end_ms - hour0) / HOUR_MS))
    times = [min(hour0 + (k + 1) * HOUR_MS, end_ms) for k in range(n)]
    sigmas = garch.hour_sigmas(fit, profile, hour0, n)
    full_sigmas = list(sigmas)                                               # whole hours, for the 15m caution test
    sigmas[0] *= math.sqrt(max((times[0] - now_ms) / HOUR_MS, 1 / 60))     # what is left of this hour
    mults = [profile[int((hour0 + k * HOUR_MS) // HOUR_MS % 24)] for k in range(n)]
    phases = [_phase(x) for x in mults]
    macro_windows = []
    for ws, we, title in forced_windows or []:
        ws, we = max(int(ws), int(now_ms)), min(int(we), int(end_ms))
        if we <= ws:
            continue
        macro_windows.append({"title": str(title), "start_ms": ws, "end_ms": we})
        for k in range(n):
            begin = now_ms if k == 0 else times[k - 1]
            if begin < we and times[k] > ws and phases[k] != "macro":
                phases[k] = "macro"                     # marked once, scaled once, renamed below
                sigmas[k] *= MACRO_SIGMA_X
                full_sigmas[k] *= MACRO_SIGMA_X         # the caution test is as wide as the hour it judges
    phases = ["chop" if ph == "macro" else ph for ph in phases]

    parts = lean_parts(closed, price, flow, now_ms, price_only)
    score = 0.0 if force_neutral else lean_score(parts)
    compressed = fit.compression < COMPRESSION_X
    neutral = abs(score) < NEUTRAL_SCORE or (compressed and abs(score) < COMPRESSED_SCORE)
    regime = "neutral" if neutral else "bullish" if score > 0 else "bearish"

    var = np.array(sigmas) ** 2
    carry = np.where(np.array(phases) == "chop", 0.0, var)
    drift = np.zeros(n)
    if regime != "neutral" and carry.sum() > 0:
        drift = score * DRIFT_K * math.sqrt(var.sum()) * carry / carry.sum()

    rng = np.random.default_rng(SEED)
    steps = np.repeat(np.array(sigmas) / math.sqrt(SUB), SUB)
    mu = np.repeat(drift / SUB, SUB)
    logp = np.concatenate([np.zeros((paths, 1)),
                           np.cumsum(rng.standard_normal((paths, n * SUB)) * steps + mu, axis=1)], axis=1)
    prices = price * np.exp(logp)
    at_hours = prices[:, SUB::SUB]
    median = np.median(at_hours, axis=0)
    band_lo, band_hi = np.percentile(at_hours, BAND_LOW, axis=0), np.percentile(at_hours, BAND_HIGH, axis=0)
    highs, lows = prices.max(axis=1), prices.min(axis=1)
    sub_times = np.array([now_ms] + [t for k in range(n) for t in
                                     np.linspace(times[k - 1] if k else now_ms, times[k], SUB + 1)[1:]])
    hi_t = int(np.median(sub_times[prices.argmax(axis=1)]))
    lo_t = int(np.median(sub_times[prices.argmin(axis=1)]))
    end = prices[:, -1]

    inv_low = float(np.percentile(lows, INVALIDATION_PCT))
    inv_high = float(np.percentile(highs, 100 - INVALIDATION_PCT))
    invalidation = ({"below": round(inv_low, 2)} if regime == "bullish" else
                    {"above": round(inv_high, 2)} if regime == "bearish" else
                    {"below": round(inv_low, 2), "above": round(inv_high, 2)})
    holds = np.ones(paths, dtype=bool)
    if "below" in invalidation:
        holds &= lows > invalidation["below"]
    if "above" in invalidation:
        holds &= highs < invalidation["above"]
    p_up = float(np.mean(end > price))
    p_a = (p_up if regime == "bullish" else 1 - p_up if regime == "bearish" else float(np.mean(holds)))

    a_atr = atr(closed.tail(200))
    atr_1h = float(a_atr.iloc[-1]) if len(a_atr) and np.isfinite(a_atr.iloc[-1]) else None
    vp = volume_profile(closed.tail(VALUE_HOURS), lookback=VALUE_HOURS)
    liq = estimated_liquidation_zones(closed, None if price_only else flow)
    levels: list[tuple[str, float]] = []
    if vp.available:
        levels += [("POC", vp.poc), ("value high", vp.value_area_high), ("value low", vp.value_area_low)]
    if gamma and gamma.get("available", True) and not price_only:
        for key, label in (("call_wall", "call wall"), ("put_wall", "put wall"), ("zero_gamma_level", "gamma flip")):
            if gamma.get(key):
                levels.append((label, float(gamma[key])))
    for z in liq:
        levels.append(("est. liquidation zone" if z["oi_surge"] else "air pocket", (z["low"] + z["high"]) / 2))
    level_odds = []
    for label, lv in levels:
        if not lv or not math.isfinite(lv) or abs(lv / price - 1) < 1e-6:
            continue
        hit = highs >= lv if lv > price else lows <= lv
        level_odds.append({"label": label, "price": round(float(lv), 2), "side": "above" if lv > price else "below",
                           "p_touch": round(float(np.mean(hit)), 3)})
    level_odds.sort(key=lambda d: -d["price"])

    path = [round(float(x), 2) for x in median]
    plan = {
        "available": True, "version": version, "day_ms": day_start(now_ms), "created_ms": int(now_ms),
        "end_ms": int(end_ms), "origin": round(float(price), 2), "regime": regime,
        "score": round(score, 3), "parts": parts, "price_only": price_only,
        "p_scenario_a": round(p_a, 3), "p_up": round(p_up, 3),
        "hours": [{"t_ms": int(t), "phase": ph, "mult": round(mu_, 2), "sigma_pct": round(s * 100, 3),
                   "sigma_full_pct": round(fs * 100, 3),
                   "median": p, "band_low": round(float(lo), 2), "band_high": round(float(hi), 2)}
                  for t, ph, mu_, s, fs, p, lo, hi in zip(times, phases, mults, sigmas, full_sigmas, path, band_lo, band_hi)],
        "windows": _windows(times, phases, sigmas, path, now_ms, price),
        "projected_high": {"price": round(float(np.median(highs)), 2), "t_ms": hi_t},
        "projected_low": {"price": round(float(np.median(lows)), 2), "t_ms": lo_t},
        "range_68": [round(float(np.percentile(end, BAND_LOW)), 2), round(float(np.percentile(end, BAND_HIGH)), 2)],
        "invalidation": invalidation, "atr_1h": None if atr_1h is None else round(atr_1h, 2),
        "levels": level_odds, "liquidation_zones": liq,
        "value_area": None if not vp.available else {"poc": round(vp.poc, 2), "high": round(vp.value_area_high, 2),
                                                     "low": round(vp.value_area_low, 2)},
        "garch": {"alpha": fit.alpha, "beta": fit.beta, "compression": round(fit.compression, 3),
                  "compressed": compressed, "fit_hours": fit.n},
    }
    if force_neutral:
        plan["preflight_override"] = "trap"
    if macro_windows:
        plan["macro_windows"] = macro_windows
    plan["scenario_a"], plan["scenario_b"] = scenarios(plan)
    return plan


def locked_plan(now_ms: int, end_ms: int, price: float, version: int, high: float | None,
                low: float | None) -> dict:
    """The rest of the day after the last allowed invalidation: no path, no direction, stand aside."""
    return {"available": True, "version": version, "day_ms": day_start(now_ms), "created_ms": int(now_ms),
            "end_ms": int(end_ms), "origin": round(float(price), 2), "regime": "locked", "score": 0.0,
            "parts": {}, "p_scenario_a": None, "hours": [], "windows": [], "levels": [],
            "liquidation_zones": [], "invalidation": {},
            "day_range": None if high is None or low is None else [round(low, 2), round(high, 2)],
            "scenario_a": f"Stand aside. {_COUNT_WORD.get(LOCK_AFTER_INVALIDATIONS, LOCK_AFTER_INVALIDATIONS)} "
                          "plans failed today: the market is chopping both ways.",
            "scenario_b": ""}


_COUNT_WORD = {2: "Two", 3: "Three", 4: "Four"}


def wire_line(plan: dict) -> str:
    """The one-sentence directive for the top panel and the bot: the AI's checked sentence when there is
    one, else the code's Scenario A without its 'Higher:' / 'Lower:' / 'Range:' tag."""
    w = plan.get("writeup") or {}
    if w.get("summary"):
        return w["summary"]
    a = plan.get("scenario_a") or ""
    if plan.get("regime") == "locked" or ": " not in a:
        return a
    rest = a.split(": ", 1)[1]
    return rest[:1].upper() + rest[1:]


def _money(x) -> str:
    return "—" if x is None else f"${x:,.0f}"


def scenarios(plan: dict) -> tuple[str, str]:
    """Scenario A, the primary path, and B, the contingency, in plain words from the plan's numbers."""
    hi, lo = plan["projected_high"]["price"], plan["projected_low"]["price"]
    inv = plan["invalidation"]
    if plan["regime"] == "bullish":
        return (f"Higher: dips hold above {_money(lo)} and price works up towards {_money(hi)}.",
                f"The rise fails: price loses {_money(inv['below'])} on a 4-hour close and the plan is replaced.")
    if plan["regime"] == "bearish":
        return (f"Lower: bounces stall under {_money(hi)} and price works down towards {_money(lo)}.",
                f"The fall fails: price closes a 4-hour candle above {_money(inv['above'])} and the plan is replaced.")
    return (f"Range: price stays between {_money(inv['below'])} and {_money(inv['above'])}, "
            f"most likely {_money(lo)} to {_money(hi)}.",
            "A breakout: a 4-hour close outside the range starts a new plan in that direction.")


def directive(plan: dict, at) -> str:
    """The plain-English game plan, written from the plan's numbers. `at(ms)` formats a time."""
    if plan["regime"] == "locked":
        return plan["scenario_a"] + f" Nothing new until the next plan at {at(plan['end_ms'])}."
    chop = [w for w in plan.get("windows") or [] if w["phase"] == "chop"]
    expansion = [w for w in plan.get("windows") or [] if w["phase"] == "expansion"]
    ph, pl = plan["projected_high"], plan["projected_low"]
    lean = {"bullish": "The plan leans UP.", "bearish": "The plan leans DOWN.",
            "neutral": "No clear direction today: expect a range, not a trend."}[plan["regime"]]
    out = [lean]
    if chop:
        out.append(f"Expect quiet, sideways trade from {at(chop[0]['start_ms'])} to {at(chop[0]['end_ms'])}"
                   + (" (and in the other quiet windows below)" if len(chop) > 1 else "") + ".")
    if expansion:
        out.append(f"The bigger moves usually come from {at(expansion[0]['start_ms'])} to {at(expansion[0]['end_ms'])}.")
    out.append(f"Most likely high about {_money(ph['price'])} around {at(ph['t_ms'])}; most likely low about "
               f"{_money(pl['price'])} around {at(pl['t_ms'])}.")
    inv = plan["invalidation"]
    if plan["regime"] == "neutral":
        out.append(f"The range breaks if a 4-hour candle closes above {_money(inv['above'])} or below "
                   f"{_money(inv['below'])}.")
    else:
        side, level = next(iter(inv.items()))
        out.append(f"The plan is wrong if a 4-hour candle closes {side} {_money(level)}.")
    return " ".join(out)


# ---------- invalidation, read from the candles ----------

def invalidation_hit(plan: dict, h1: pd.DataFrame, now_ms: int) -> dict | None:
    """The first 4-hour candle close beyond the plan's invalidation since it was made; None while it holds.

    Strictly the 4h close (Hammad, 28 Oct): an hourly close beyond the line, however far, is a wick on the
    4h candle until that candle closes. The CAUTION tier covers the hours in between."""
    inv = plan.get("invalidation") or {}
    if not inv:
        return None
    rows = h1[(h1["timestamp"].astype("int64") >= plan["created_ms"] - plan["created_ms"] % HOUR_MS)
              & (h1["timestamp"].astype("int64") + HOUR_MS <= min(now_ms, plan["end_ms"]))]
    for t, close in zip(rows["timestamp"].astype("int64"), rows["close"].astype(float)):
        close_ms = int(t) + HOUR_MS
        if close_ms <= plan["created_ms"] or close_ms % FOUR_H_MS != 0:
            continue
        for side, level in inv.items():
            beyond = close - level if side == "above" else level - close
            if beyond > 0:
                return {"side": side, "level": level, "close": round(close, 2), "close_ms": close_ms,
                        "basis": "4-hour close"}
    return None


# ---------- scoring ----------

def score_plan(plan: dict, h1: pd.DataFrame, until_ms: int | None = None) -> dict | None:
    """How the plan did over its window, from the hourly candles. None until the window has closed on the
    feed; {"scored": False} when more than MISSING_HOURS_TOLERATED of its candles never arrived, so a gap
    in the feed is recorded once instead of retried forever. `until_ms` shortens the window to when the
    plan was replaced, so an invalidated plan is judged on the hours it was live."""
    if plan.get("regime") == "locked" or not plan.get("available"):
        return {"scored": False, "reason": "no forecast to score"}
    start = plan["created_ms"] - plan["created_ms"] % HOUR_MS
    end = min(plan["end_ms"], until_ms) if until_ms else plan["end_ms"]
    end = max(end, start + HOUR_MS)
    ts = h1["timestamp"].astype("int64")
    if ts.empty or int(ts.max()) + HOUR_MS < end:
        return None
    rows = h1[(ts >= start) & (ts + HOUR_MS <= end)]
    expected = math.ceil((end - start) / HOUR_MS)
    if len(rows) < expected - MISSING_HOURS_TOLERATED or rows.empty:
        return {"scored": False, "reason": f"{expected - len(rows)} of {expected} hourly candles missing"}
    i_hi, i_lo = int(rows["high"].astype(float).values.argmax()), int(rows["low"].astype(float).values.argmin())
    high, low = float(rows["high"].iloc[i_hi]), float(rows["low"].iloc[i_lo])
    hi_t = int(rows["timestamp"].iloc[i_hi]) + HOUR_MS // 2
    lo_t = int(rows["timestamp"].iloc[i_lo]) + HOUR_MS // 2
    close = float(rows["close"].iloc[-1])
    inv = invalidation_hit(plan, h1, end)
    regime = plan["regime"]
    direction = (close > plan["origin"] if regime == "bullish" else close < plan["origin"] if regime == "bearish"
                 else inv is None)
    ph, pl = plan["projected_high"], plan["projected_low"]
    tol = HIGH_LOW_TOL * (ph["price"] - pl["price"])
    hours = plan.get("hours") or []
    last = next((h for h in hours if h["t_ms"] >= end), hours[-1] if hours else None)
    return {"scored": True, "regime": regime, "high": round(high, 2), "high_ms": hi_t, "low": round(low, 2), "low_ms": lo_t,
            "close": round(close, 2), "direction_hit": bool(direction), "invalidated": inv is not None,
            "scenario_a_right": bool(direction) and inv is None,
            "high_hit": abs(high - ph["price"]) <= tol, "low_hit": abs(low - pl["price"]) <= tol,
            "high_time_hit": abs(hi_t - ph["t_ms"]) <= TIME_TOL_MS, "low_time_hit": abs(lo_t - pl["t_ms"]) <= TIME_TOL_MS,
            "band_held": bool(last and last["band_low"] <= close <= last["band_high"])}


SCORE_KEYS = ("scenario_a_right", "high_hit", "low_hit", "high_time_hit", "low_time_hit", "band_held", "invalidated")


def record(outcomes: list[dict]) -> dict:
    """Counts over scored outcomes, newest first: {"days": n, "scenario_a_right": k, ...}.

    Direction days and range days are also counted apart: a range day is right whenever its box holds,
    which its width makes likely, so a total mixing the two flatters the directional calls."""
    done = [o for o in outcomes if o and o.get("scored")]
    out = {"days": len(done)}
    for k in SCORE_KEYS:
        out[k] = sum(1 for o in done if o.get(k))
    directional = [o for o in done if o.get("regime") in ("bullish", "bearish")]
    ranges = [o for o in done if o.get("regime") == "neutral"]
    out.update({"direction_days": len(directional),
                "direction_right": sum(1 for o in directional if o.get("scenario_a_right")),
                "range_days": len(ranges), "range_held": sum(1 for o in ranges if o.get("scenario_a_right"))})
    return out


def backtest(h1: pd.DataFrame, now_ms: int, days: int = BACKTEST_DAYS, paths: int = 800) -> dict:
    """The morning plan's rules replayed on each past day the history allows, price-only, and scored the
    same way as the live plans. {"days": 0} with too little history."""
    closed = closed_hours(h1, now_ms)
    if closed.empty:
        return record([])
    first_ts = int(closed["timestamp"].iloc[0])
    last_close = int(closed["timestamp"].iloc[-1]) + HOUR_MS
    outcomes: list[dict] = []
    d = day_start(last_close) - DAY_MS
    while len(outcomes) < days and d - first_ts >= (garch.FIT_HOURS + 60) * HOUR_MS:
        hist = closed[closed["timestamp"].astype("int64") + HOUR_MS <= d]
        price = float(hist["close"].iloc[-1])
        plan = build_plan(hist, d, d + DAY_MS, price, price_only=True, paths=paths)
        if plan.get("available"):
            o = score_plan(plan, closed)
            if o and o.get("scored"):
                outcomes.append(o)
        d -= DAY_MS
    out = record(outcomes)
    out["method"] = "the morning plan replayed on past days with price only (no taker flow, open interest or gamma)"
    return out
