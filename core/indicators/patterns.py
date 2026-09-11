"""Chart patterns forming (or freshly confirmed) on the chart timeframe, built from swing pivots.

Detected: double top / double bottom (including a second top or bottom being tested right now),
head and shoulders / inverse, and the line-fit family — ascending, descending and symmetrical
triangles, rising and falling wedges, rising and falling channels, and flat ranges.

Every pattern carries a status (forming | confirmed), a breakout level, a measured-move target and an
invalidation level, plus the polylines to draw. A breakout older than FRESH bars is dropped as stale:
the move has already happened."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.indicators.pivots import find_pivots

LOOKBACK = 120   # bars scanned for patterns
RECENT = 40      # the pattern's last swing must sit inside the last RECENT bars
FRESH = 5        # a confirmed breakout older than this is stale
PIVOT_LR = 3
FLAT = 0.03      # |slope| below this many ATR per bar counts as flat
FIT_TOL = 0.75   # every pivot must sit within this many ATR of its fitted line


@dataclass
class Pattern:
    name: str
    bias: str                 # bullish | bearish | neutral
    status: str               # forming | confirmed
    confidence: float         # 0..1
    breakout: float | None
    target: float | None
    invalidation: float | None
    lines: list[list[tuple[int, float]]]
    start_ms: int
    note: str = ""
    apex_bars: int | None = None
    pivots: tuple[int, ...] = field(default=(), repr=False)   # indices used, for de-duplication


class _Ctx:
    def __init__(self, df: pd.DataFrame, atr_value: float | None):
        self.d = df.tail(LOOKBACK).reset_index(drop=True)
        self.n = len(self.d)
        self.highs = self.d["high"].to_numpy(dtype=float)
        self.lows = self.d["low"].to_numpy(dtype=float)
        self.closes = self.d["close"].to_numpy(dtype=float)
        self.ts = self.d["timestamp"].to_numpy(dtype="int64")
        self.interval = int(np.median(np.diff(self.ts))) if self.n > 1 else 60_000
        self.atr = atr_value if atr_value else (float(np.median(self.highs - self.lows)) or 1e-9)
        self.last = self.n - 1

    def t(self, i: float) -> int:
        i = int(round(i))
        return int(self.ts[i]) if i <= self.last else int(self.ts[self.last] + (i - self.last) * self.interval)

    def line(self, pts: list[tuple[float, float]]) -> list[tuple[int, float]]:
        out: list[tuple[int, float]] = []
        for i, p in pts:
            tt = self.t(i)
            if not out or tt > out[-1][0]:
                out.append((tt, float(p)))
        return out


def _clamp(x: float) -> float:
    return max(0.05, min(0.95, x))


def _double(c: _Ctx, top: bool) -> Pattern | None:
    arr = c.highs if top else c.lows
    piv = find_pivots(arr, PIVOT_LR, PIVOT_LR, "high" if top else "low")
    cands: list[tuple[int, int, bool]] = []
    if len(piv) >= 2:
        cands.append((piv[-2], piv[-1], False))
    if piv:
        cands.append((piv[-1], c.last, True))   # the current bar is retesting the last swing
    for p1, p2, in_progress in cands:
        if p2 - p1 < 5 or p2 < c.n - RECENT:
            continue
        e1, e2 = arr[p1], arr[p2]
        if abs(e1 - e2) > 0.6 * c.atr:
            continue
        mid = slice(p1 + 1, p2)
        if top:
            tr = p1 + 1 + int(np.argmin(c.lows[mid]))
            neck = c.lows[tr]
            depth = min(e1, e2) - neck
            extreme = max(e1, e2)
        else:
            tr = p1 + 1 + int(np.argmax(c.highs[mid]))
            neck = c.highs[tr]
            depth = neck - max(e1, e2)
            extreme = min(e1, e2)
        if depth < 1.5 * c.atr:
            continue
        if in_progress:
            # the retest must be the most extreme bar since the neckline, i.e. the second peak is printing now
            seg = arr[tr:c.last + 1]
            if (top and arr[c.last] < seg.max()) or (not top and arr[c.last] > seg.min()):
                continue
        after = c.closes[p2:]
        if (top and (after > extreme + 0.25 * c.atr).any()) or (not top and (after < extreme - 0.25 * c.atr).any()):
            continue  # invalidated
        broke = np.flatnonzero(after < neck - 0.1 * c.atr) if top else np.flatnonzero(after > neck + 0.1 * c.atr)
        status = "forming"
        if broke.size:
            if c.last - (p2 + int(broke[0])) > FRESH:
                continue
            status = "confirmed"
        conf = 0.5 + (0.15 if abs(e1 - e2) <= 0.3 * c.atr else 0) + (0.15 if depth >= 2.5 * c.atr else 0)
        conf += 0.15 if status == "confirmed" else 0
        conf -= 0.1 if in_progress else 0
        name = "Double top" if top else "Double bottom"
        return Pattern(
            name=name, bias="bearish" if top else "bullish", status=status, confidence=_clamp(conf),
            breakout=float(neck), target=float(neck - depth if top else neck + depth),
            invalidation=float(extreme + 0.25 * c.atr if top else extreme - 0.25 * c.atr),
            lines=[c.line([(p1, e1), (tr, neck), (p2, e2)]), c.line([(tr, neck), (c.last + 5, neck)])],
            start_ms=c.t(p1), note=("second " + ("top" if top else "bottom") + " being tested now") if in_progress else "",
            pivots=(p1, p2),
        )
    return None


def _head_shoulders(c: _Ctx, top: bool) -> Pattern | None:
    arr = c.highs if top else c.lows
    piv = find_pivots(arr, PIVOT_LR, PIVOT_LR, "high" if top else "low")
    if len(piv) < 3:
        return None
    a, b, s = piv[-3:]
    if s < c.n - RECENT or min(b - a, s - b) < 3:
        return None
    ea, eb, es = arr[a], arr[b], arr[s]
    head_ok = eb > max(ea, es) + 0.75 * c.atr if top else eb < min(ea, es) - 0.75 * c.atr
    if not head_ok or abs(ea - es) > 1.0 * c.atr:
        return None
    if top:
        t1 = a + 1 + int(np.argmin(c.lows[a + 1:b]))
        t2 = b + 1 + int(np.argmin(c.lows[b + 1:s]))
        n1, n2 = c.lows[t1], c.lows[t2]
    else:
        t1 = a + 1 + int(np.argmax(c.highs[a + 1:b]))
        t2 = b + 1 + int(np.argmax(c.highs[b + 1:s]))
        n1, n2 = c.highs[t1], c.highs[t2]
    slope = (n2 - n1) / (t2 - t1) if t2 != t1 else 0.0

    def neck(i: float) -> float:
        return n1 + slope * (i - t1)

    height = (eb - neck(b)) if top else (neck(b) - eb)
    if height <= 0:
        return None
    idx = np.arange(s, c.last + 1)
    if (top and (c.closes[s:] > eb).any()) or (not top and (c.closes[s:] < eb).any()):
        return None  # invalidated: price went through the head
    broke = np.flatnonzero(c.closes[s:] < neck(idx) - 0.1 * c.atr) if top else np.flatnonzero(c.closes[s:] > neck(idx) + 0.1 * c.atr)
    status = "forming"
    if broke.size:
        if c.last - (s + int(broke[0])) > FRESH:
            return None
        status = "confirmed"
    now_neck = neck(c.last)
    conf = 0.55 + (0.15 if abs(ea - es) <= 0.5 * c.atr else 0) + (0.15 if status == "confirmed" else 0)
    return Pattern(
        name="Head and shoulders" if top else "Inverse head and shoulders",
        bias="bearish" if top else "bullish", status=status, confidence=_clamp(conf),
        breakout=float(now_neck), target=float(now_neck - height if top else now_neck + height),
        invalidation=float(eb), start_ms=c.t(a),
        lines=[c.line([(a, ea), (t1, n1), (b, eb), (t2, n2), (s, es)]), c.line([(t1, n1), (c.last + 5, neck(c.last + 5))])],
        pivots=(a, b, s, t1, t2),
    )


def _line_family(c: _Ctx) -> Pattern | None:
    ph = find_pivots(c.highs, PIVOT_LR, PIVOT_LR, "high")
    pl = find_pivots(c.lows, PIVOT_LR, PIVOT_LR, "low")
    for k in (4, 3):
        H, L = ph[-k:], pl[-k:]
        if len(H) < 2 or len(L) < 2 or len(H) + len(L) < 5:
            continue
        if max(H[-1], L[-1]) < c.n - RECENT:
            continue
        hx, lx = np.array(H, dtype=float), np.array(L, dtype=float)
        su, bu = np.polyfit(hx, c.highs[H], 1)
        sl, bl = np.polyfit(lx, c.lows[L], 1)
        res_u = float(np.max(np.abs(c.highs[H] - (su * hx + bu))))
        res_l = float(np.max(np.abs(c.lows[L] - (sl * lx + bl))))
        if max(res_u, res_l) > FIT_TOL * c.atr:
            continue
        x0 = min(H[0], L[0])
        span = c.last - x0
        if span < 15:
            continue

        def up(x: float) -> float:
            return su * x + bu

        def lo(x: float) -> float:
            return sl * x + bl

        gap0, gapn = up(x0) - lo(x0), up(c.last) - lo(c.last)
        if gap0 <= 0 or gapn <= 0:
            continue
        nu, nl = su / c.atr, sl / c.atr
        fu, fl = abs(nu) < FLAT, abs(nl) < FLAT
        ratio = gapn / gap0
        name = bias = None
        converging = ratio < 0.75
        if converging:
            if fu and nl >= FLAT:
                name, bias = "Ascending triangle", "bullish"
            elif fl and nu <= -FLAT:
                name, bias = "Descending triangle", "bearish"
            elif nu <= -FLAT and nl >= FLAT:
                name, bias = "Symmetrical triangle", "neutral"
            elif nu >= FLAT and nl >= FLAT and nl > nu:
                name, bias = "Rising wedge", "bearish"
            elif nu <= -FLAT and nl <= -FLAT and nu < nl:
                name, bias = "Falling wedge", "bullish"
        elif 0.8 <= ratio <= 1.25:
            if nu >= FLAT and nl >= FLAT:
                name, bias = "Rising channel", "bullish"
            elif nu <= -FLAT and nl <= -FLAT:
                name, bias = "Falling channel", "bearish"
            elif fu and fl:
                name, bias = "Range", "neutral"
        if not name:
            continue

        pad = 0.25 * c.atr
        close = c.closes[c.last]
        status = "forming"
        if close > up(c.last) + pad or close < lo(c.last) - pad:
            direction_up = close > up(c.last) + pad
            recent = np.arange(max(x0, c.last - 2 * FRESH), c.last + 1)
            outside = (c.closes[recent] > up(recent) + pad) if direction_up else (c.closes[recent] < lo(recent) - pad)
            run = 0
            for flag in outside[::-1]:
                if not flag:
                    break
                run += 1
            if run > FRESH:
                return None  # breakout already played out
            status = "confirmed"
            bias = "bullish" if direction_up else "bearish"

        apex_bars = None
        if converging and su != sl:
            xa = (bl - bu) / (su - sl)
            if xa > c.last:
                apex_bars = int(round(xa - c.last))
        height = gap0
        channel = name.endswith("channel") or name == "Range"
        if channel and status == "forming":
            breakout = None
            target = up(c.last) if bias == "bullish" else lo(c.last) if bias == "bearish" else None
            invalidation = lo(c.last) - pad if bias == "bullish" else up(c.last) + pad if bias == "bearish" else None
            note = f"range {lo(c.last):,.0f} – {up(c.last):,.0f}" if bias == "neutral" else ""
        elif bias == "bullish":
            breakout, target, invalidation, note = up(c.last), up(c.last) + height, lo(c.last) - pad, ""
        elif bias == "bearish":
            breakout, target, invalidation, note = lo(c.last), lo(c.last) - height, up(c.last) + pad, ""
        else:
            breakout, target, invalidation = up(c.last), up(c.last) + height, lo(c.last)
            note = f"break below {lo(c.last):,.0f} targets {lo(c.last) - height:,.0f}"
        end_x = c.last + (min(apex_bars, 30) if apex_bars else 10)
        touches = len(H) + len(L)
        conf = 0.45 + 0.1 * (touches - 4) + (0.15 if max(res_u, res_l) < 0.3 * c.atr else -0.1 if max(res_u, res_l) > 0.5 * c.atr else 0)
        conf += (0.1 if span >= 30 else 0) + (0.1 if status == "confirmed" else 0)
        return Pattern(
            name=name, bias=bias, status=status, confidence=_clamp(conf),
            breakout=float(breakout) if breakout is not None else None,
            target=float(target) if target is not None else None,
            invalidation=float(invalidation) if invalidation is not None else None,
            lines=[c.line([(H[0], up(H[0])), (end_x, up(end_x))]), c.line([(L[0], lo(L[0])), (end_x, lo(end_x))])],
            start_ms=c.t(x0), note=note, apex_bars=apex_bars, pivots=tuple(H) + tuple(L),
        )
    return None


def detect_patterns(df: pd.DataFrame, atr_value: float | None, max_patterns: int = 3) -> list[Pattern]:
    c = _Ctx(df, atr_value)
    if c.n < 30:
        return []
    found = [p for p in (_line_family(c), _head_shoulders(c, True), _head_shoulders(c, False),
                         _double(c, True), _double(c, False)) if p is not None]
    # a double top/bottom whose swings are already part of a fitted triangle/wedge/channel is the same structure
    families = [p for p in found if p.name.endswith(("triangle", "wedge", "channel")) or p.name == "Range"]
    hs = [p for p in found if "shoulders" in p.name]
    kept = []
    for p in found:
        if p.name.startswith("Double") and any(set(p.pivots) <= set(f.pivots) for f in families + hs):
            continue
        kept.append(p)
    dt = next((p for p in kept if p.name == "Double top" and p.status == "forming"), None)
    db = next((p for p in kept if p.name == "Double bottom" and p.status == "forming"), None)
    if dt and db:
        top = dt.invalidation - 0.25 * c.atr
        bottom = db.invalidation + 0.25 * c.atr
        if top > bottom:
            start = min(dt.start_ms, db.start_ms)
            first = int(np.searchsorted(c.ts, start))
            kept = [p for p in kept if p is not dt and p is not db]
            kept.append(Pattern(
                name="Range (double top / double bottom)", bias="neutral", status="forming",
                confidence=max(dt.confidence, db.confidence), breakout=float(top), target=float(top + (top - bottom)),
                invalidation=float(bottom), start_ms=start,
                lines=[c.line([(first, top), (c.last + 5, top)]), c.line([(first, bottom), (c.last + 5, bottom)])],
                note=f"double top near {top:,.0f} and double bottom near {bottom:,.0f}: range until one side breaks; "
                     f"below {bottom:,.0f} targets {bottom - (top - bottom):,.0f}",
                pivots=dt.pivots + db.pivots,
            ))
    kept.sort(key=lambda p: p.confidence, reverse=True)
    return kept[:max_patterns]
