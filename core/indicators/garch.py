"""GARCH(1,1) volatility on hourly BTC returns, with the hour-of-day pattern taken out first.

Volatility clusters: a loud hour is usually followed by more loud hours, and a quiet stretch by more quiet
ones. GARCH(1,1) says next hour's variance is a long-run level plus a share of the last surprise (alpha)
plus a share of the last variance (beta):

    var[t] = omega + alpha * r[t-1]**2 + beta * var[t-1],   omega = long_run * (1 - alpha - beta)

BTC is also louder at some hours than others (New York's open, the US data prints), every day. A GARCH fit
on raw hourly returns reads that daily rhythm as clustering, so the hour-of-day factor is divided out
first (`hourly_profile`), the model is fitted on what is left, and the factor is put back on the forecast.

Fitted by maximum likelihood over a fixed grid of (alpha, beta), with omega pinned to the sample variance
(variance targeting). No optimiser and no extra package: the grid is fine enough for a volatility forecast,
the same inputs always give the same fit, and it runs in a few milliseconds on a month of hours."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

HOUR_MS = 3_600_000
FIT_HOURS = 500                  # about three weeks: enough for a stable fit, recent enough to be today's market
PROFILE_DAYS = 60                # the hour-of-day pattern, from the last two months
ALPHAS = np.round(np.arange(0.01, 0.31, 0.01), 2)
BETAS = np.round(np.arange(0.50, 0.99, 0.01), 2)
MAX_PERSISTENCE = 0.995          # alpha + beta below this: the variance must return to its long-run level


@dataclass(frozen=True)
class GarchFit:
    alpha: float
    beta: float
    long_run_var: float          # per hour, of the de-seasonalised return
    next_var: float              # the next hour's, same units
    loglik: float
    n: int

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    def var_ahead(self, k: int) -> float:
        """The variance k hours after the last return (k = 1 is the next hour)."""
        return self.long_run_var + self.persistence ** (k - 1) * (self.next_var - self.long_run_var)

    @property
    def compression(self) -> float:
        """Next hour's volatility against the long-run level: under 1 is quieter than usual."""
        return math.sqrt(self.next_var / self.long_run_var) if self.long_run_var > 0 else 1.0


def log_returns(closes) -> np.ndarray:
    c = np.asarray(closes, dtype=float)
    return np.diff(np.log(c))


def hourly_profile(timestamps_ms, closes, days: int = PROFILE_DAYS) -> list[float]:
    """24 multipliers, one per UTC hour: how loud that hour usually is against the day's average, as the
    RMS of its returns over the overall RMS (so a plain 1.0 is an average hour). Each return is labelled
    by the hour its candle opened. An hour with no data gets 1.0."""
    t = np.asarray(timestamps_ms, dtype=np.int64)[1:]
    r = log_returns(closes)
    keep = t >= t[-1] - days * 24 * HOUR_MS if len(t) else np.array([], dtype=bool)
    t, r = t[keep], r[keep]
    if len(r) < 48:
        return [1.0] * 24
    hours = (t // HOUR_MS) % 24
    overall = float(np.mean(r ** 2))
    if overall <= 0:
        return [1.0] * 24
    out = []
    for h in range(24):
        sel = r[hours == h]
        out.append(math.sqrt(float(np.mean(sel ** 2)) / overall) if len(sel) >= 5 else 1.0)
    # re-normalise so the 24 multipliers average to one in variance: the fit's long-run level stays the day's
    scale = math.sqrt(sum(x * x for x in out) / 24)
    return [x / scale for x in out]


def deseasonalise(timestamps_ms, closes, profile: list[float]) -> np.ndarray:
    t = np.asarray(timestamps_ms, dtype=np.int64)[1:]
    hours = (t // HOUR_MS) % 24
    return log_returns(closes) / np.asarray(profile)[hours]


def fit(returns, hours: int = FIT_HOURS) -> GarchFit | None:
    """The best (alpha, beta) on the grid for the last `hours` returns, or None with too few or flat ones."""
    r = np.asarray(returns, dtype=float)[-hours:]
    r = r[np.isfinite(r)]
    if len(r) < 100:
        return None
    r = r - r.mean()
    v = float(np.var(r))
    if v <= 0:
        return None
    a, b = np.meshgrid(ALPHAS, BETAS, indexing="ij")
    a, b = a.ravel(), b.ravel()
    ok = a + b < MAX_PERSISTENCE
    a, b = a[ok], b[ok]
    omega = v * (1 - a - b)
    var = np.full_like(a, v)
    ll = np.zeros_like(a)
    r2 = r ** 2
    for x2 in r2:
        ll -= 0.5 * (np.log(var) + x2 / var)
        var = omega + a * x2 + b * var
    best = int(np.argmax(ll))
    # `var` now holds, for each candidate, the variance of the hour after the last return
    return GarchFit(float(a[best]), float(b[best]), v, float(var[best]), float(ll[best]), len(r))


def hour_sigmas(fit_: GarchFit, profile: list[float], first_hour_ms: int, n: int) -> list[float]:
    """The forecast standard deviation of each of the next `n` hourly returns, the first being the hour
    that opens at `first_hour_ms`, with that hour's usual loudness put back."""
    out = []
    for k in range(n):
        h = ((first_hour_ms + k * HOUR_MS) // HOUR_MS) % 24
        out.append(math.sqrt(max(fit_.var_ahead(k + 1), 0.0)) * profile[int(h)])
    return out
