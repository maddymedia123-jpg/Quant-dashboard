"""Serialise a CategoryAnalysis into a self-contained TradingView lightweight-charts document."""
from __future__ import annotations

import json
import math

import streamlit.components.v1 as components

from core.config import CATEGORIES, LIGHTWEIGHT_CHARTS_URL, TF_MINUTES
from core.indicators.category import CategoryAnalysis
from ui.theme import palette

# The design draws the fast EMA in deep steel and the slow one in pale steel; the long averages step
# down the neutral ramp so four lines stay tellable apart without adding a second hue.
EMA_COLORS = {21: "#416180", 50: "#94BCE3", 100: "#98989B", 200: "#5D5D60"}


def _f(v) -> float | None:
    if v is None:
        return None
    v = float(v)
    return None if math.isnan(v) or math.isinf(v) else v


def chart_payload(a: CategoryAnalysis, dark: bool, signal: bool = False) -> dict:
    p = palette(dark, signal)
    df = a.chart_df
    n = len(df)
    t = (df["timestamp"] // 1000).astype("int64").tolist()
    last_t = t[-1]
    interval = TF_MINUTES[a.chart_tf] * 60
    horizon = CATEGORIES[a.key].horizon_bars

    candles = [{"time": t[i], "open": _f(df["open"].iloc[i]), "high": _f(df["high"].iloc[i]),
                "low": _f(df["low"].iloc[i]), "close": _f(df["close"].iloc[i])} for i in range(n)]

    # how far into the future the time axis must extend
    future_targets = [last_t + max(horizon, 12) * interval]
    if a.fib:
        future_targets += [z.timestamp_ms // 1000 for z in a.fib.upcoming]
    for tl in a.trendlines:
        future_targets.append(tl.points[1][0] // 1000)
    for pat in a.patterns:
        for ln in pat.lines:
            if ln:
                future_targets.append(ln[-1][0] // 1000)
    future_n = min(90, max(12, math.ceil((max(future_targets) - last_t) / interval)))
    whitespace = [{"time": last_t + k * interval} for k in range(1, future_n + 1)]

    emas = []
    for period, series in a.ema.series.items():
        vals = series.tail(n).tolist()
        data = [{"time": t[i], "value": _f(vals[i])} for i in range(n) if _f(vals[i]) is not None]
        emas.append({"period": period, "color": EMA_COLORS.get(period, p["muted"]), "data": data})

    levels = [{"price": _f(l.price), "title": f"{'R' if l.kind == 'resistance' else 'S'} {l.touches}x",
               "color": p["down"] if l.kind == "resistance" else p["up"]} for l in a.levels]

    trendlines = [{"kind": tl.kind, "color": p["down"] if tl.kind == "resistance" else p["up"],
                   "data": [{"time": tl.points[0][0] // 1000, "value": _f(tl.points[0][1])},
                            {"time": min(tl.points[1][0] // 1000, last_t + future_n * interval), "value": _f(tl.points[1][1])}]}
                  for tl in a.trendlines]

    fibs = []
    if a.fib:
        first_t = t[0]
        for z in a.fib.zones:
            zt = z.timestamp_ms // 1000
            if zt >= first_t:
                fibs.append({"time": int(zt), "label": f"F{z.k}", "future": z.is_future})

    retracements = []
    if a.fib_retr is not None:
        for ratio in (0.382, 0.5, 0.618):
            retracements.append({"price": _f(a.fib_retr.levels[ratio]), "title": f"Fib {ratio}"})

    pattern_color = {"bullish": p["up"], "bearish": p["down"], "neutral": p["accent"]}
    max_t = last_t + future_n * interval
    patterns = []
    for pat in a.patterns:
        lines = []
        for ln in pat.lines:
            pts = [{"time": int(ms // 1000), "value": _f(v)} for ms, v in ln if ms // 1000 <= max_t and _f(v) is not None]
            if len(pts) >= 2:
                lines.append(pts)
        if lines:
            patterns.append({"name": f"{pat.name} ({pat.status})", "color": pattern_color.get(pat.bias, p["accent"]), "lines": lines})

    # The forecast lives in its own pane now (forecast_payload below): one median path with its odds,
    # instead of a cone and an envelope here that answered a different question from it.

    rvals = a.rsi.tolist()
    rsi_data = [{"time": t[i], "value": _f(rvals[i])} for i in range(n) if _f(rvals[i]) is not None]
    markers = []
    for d in a.divergences:
        color = p["up"] if d.direction == "bullish" else p["down"]
        for (ts_ms, _v) in d.rsi_points:
            markers.append({"time": ts_ms // 1000, "position": "belowBar" if d.direction == "bullish" else "aboveBar",
                            "color": color, "shape": "circle", "text": ("~" if d.status == "forming" else "") + d.kind[0].upper() + "D"})
    markers.sort(key=lambda m: m["time"])

    return {
        "theme": {k: p[k] for k in ("panel", "border", "grid", "text", "muted", "accent", "up", "down", "fib", "fibFuture", "now")},
        "title": f"{a.label} · {a.chart_tf}",
        "candles": candles, "whitespace": whitespace, "emas": emas, "levels": levels,
        "trendlines": trendlines, "fibs": fibs, "now": last_t,
        "retracements": retracements, "patterns": patterns,
        "rsi": {"data": rsi_data, "markers": markers},
        "visibleBars": 140,
    }


_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&display=swap" rel="stylesheet">
<style>html,body{margin:0;padding:0;background:__PANEL__;font-family:Barlow,system-ui,sans-serif}#wrap{position:relative}#chart{width:100%;height:__HEIGHT__px}
#legend{position:absolute;left:10px;top:8px;z-index:5;font-size:11px;color:__TEXT__;display:flex;gap:10px;flex-wrap:wrap}
#legend span::before{content:"";display:inline-block;width:10px;height:2px;margin-right:4px;vertical-align:middle;background:var(--c)}</style>
<script src="__LIB__"></script></head><body><div id="wrap"><div id="legend"></div><div id="chart"></div></div>
<script>
const PAYLOAD = __PAYLOAD__;
(function(){
const P = PAYLOAD, T = P.theme, LW = LightweightCharts;
const el = document.getElementById('chart');
// On a phone the price-line titles make the right axis wider than the candles; keep just the price there.
const NARROW = el.clientWidth < 640;
const chart = LW.createChart(el, {
  width: el.clientWidth, height: __HEIGHT__,
  layout: { background: { type: 'solid', color: T.panel }, textColor: T.text, fontFamily: 'Barlow, system-ui, sans-serif',
            panes: { separatorColor: T.border, separatorHoverColor: T.border, enableResize: false } },
  grid: { vertLines: { color: T.grid }, horzLines: { color: T.grid } },
  rightPriceScale: { borderColor: T.border },
  timeScale: { borderColor: T.border, timeVisible: true, secondsVisible: false, rightOffset: 2 },
  crosshair: { mode: 0 },
  // a vertical swipe over the chart scrolls the page instead of panning the price axis
  handleScroll: { vertTouchDrag: false },
});
const candles = chart.addSeries(LW.CandlestickSeries, { upColor: T.panel, downColor: T.down, borderVisible: true, borderUpColor: T.up, borderDownColor: T.down, wickUpColor: T.up, wickDownColor: T.down });
candles.setData(P.candles.concat(P.whitespace));
const legend = document.getElementById('legend');
P.emas.forEach(e => {
  const s = chart.addSeries(LW.LineSeries, { color: e.color, lineWidth: 1, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
  s.setData(e.data);
  const tag = document.createElement('span'); tag.style.setProperty('--c', e.color); tag.textContent = 'EMA' + e.period; legend.appendChild(tag);
});
P.levels.forEach(l => candles.createPriceLine({ price: l.price, color: l.color, lineWidth: 1, lineStyle: LW.LineStyle.Dashed, axisLabelVisible: true, title: NARROW ? '' : l.title }));
(P.retracements || []).forEach(r => candles.createPriceLine({ price: r.price, color: T.fib, lineWidth: 1, lineStyle: LW.LineStyle.SparseDotted, axisLabelVisible: true, title: NARROW ? '' : r.title }));
(P.patterns || []).forEach(pt => {
  pt.lines.forEach((ln, j) => {
    const s = chart.addSeries(LW.LineSeries, { color: pt.color, lineWidth: 2, lineStyle: LW.LineStyle.LargeDashed, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
    s.setData(ln);
    if (j === 0) LW.createSeriesMarkers(s, [{ time: ln[ln.length - 1].time, position: 'aboveBar', color: pt.color, shape: 'square', text: pt.name }]);
  });
});
P.trendlines.forEach(t => {
  const s = chart.addSeries(LW.LineSeries, { color: t.color, lineWidth: 1, lineStyle: LW.LineStyle.Dotted, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
  s.setData(t.data);
});
const rsi = chart.addSeries(LW.LineSeries, { color: T.accent, lineWidth: 1, priceLineVisible: false, lastValueVisible: true, title: 'RSI 14' }, 1);
rsi.setData(P.rsi.data);
[70, 30].forEach(v => rsi.createPriceLine({ price: v, color: T.muted, lineWidth: 1, lineStyle: LW.LineStyle.Dotted, axisLabelVisible: false }));
if (P.rsi.markers.length) LW.createSeriesMarkers(rsi, P.rsi.markers);
const panes = chart.panes(); if (panes[1]) panes[1].setHeight(Math.round(__HEIGHT__ * 0.24));

class VLineRenderer {
  constructor(x, color, dash, label) { this._x = x; this._color = color; this._dash = dash; this._label = label; }
  draw(target) {
    target.useBitmapCoordinateSpace(s => {
      if (this._x === null || this._x === undefined) return;
      const ctx = s.context, hr = s.horizontalPixelRatio, vr = s.verticalPixelRatio;
      const x = Math.round(this._x * hr);
      ctx.save();
      ctx.strokeStyle = this._color; ctx.lineWidth = Math.max(1, Math.floor(hr));
      if (this._dash) ctx.setLineDash([4 * hr, 4 * hr]);
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, s.bitmapSize.height); ctx.stroke();
      if (this._label) {
        ctx.setLineDash([]); ctx.fillStyle = this._color; ctx.font = `${11 * vr}px Barlow, sans-serif`;
        const y = this._dash ? s.bitmapSize.height - 6 * vr : 12 * vr;  // fib labels at the foot, now-label at the head
        ctx.fillText(this._label, x + 4 * hr, y);
      }
      ctx.restore();
    });
  }
}
class VLineView {
  constructor(src) { this._src = src; this._x = null; }
  update() { this._x = this._src._chart.timeScale().timeToCoordinate(this._src._time); }
  renderer() { return new VLineRenderer(this._x, this._src._color, this._src._dash, this._src._label); }
}
class VLine {
  constructor(chart, time, color, dash, label) { this._chart = chart; this._time = time; this._color = color; this._dash = dash; this._label = label; this._views = [new VLineView(this)]; }
  updateAllViews() { this._views.forEach(v => v.update()); }
  paneViews() { return this._views; }
}
P.fibs.forEach(f => candles.attachPrimitive(new VLine(chart, f.time, f.future ? T.fibFuture : T.fib, true, f.label)));
candles.attachPrimitive(new VLine(chart, P.now, T.now, false, 'now'));

const total = P.candles.length + P.whitespace.length;
chart.timeScale().setVisibleLogicalRange({ from: Math.max(0, P.candles.length - P.visibleBars), to: total });
new ResizeObserver(() => chart.applyOptions({ width: el.clientWidth })).observe(el);
})();
</script></body></html>"""


def build_chart_html(a: CategoryAnalysis, dark: bool, height: int = 560, signal: bool = False) -> str:
    p = palette(dark, signal)
    payload = json.dumps(chart_payload(a, dark, signal), separators=(",", ":"))
    return (_TEMPLATE.replace("__LIB__", LIGHTWEIGHT_CHARTS_URL).replace("__HEIGHT__", str(height))
            .replace("__PANEL__", p["panel"]).replace("__TEXT__", p["text"]).replace("__PAYLOAD__", payload))


def render_chart(a: CategoryAnalysis, dark: bool, height: int = 560, signal: bool = False) -> None:
    components.html(build_chart_html(a, dark, height, signal), height=height + 6, scrolling=False)


# ---------- the forecast pane (his Section 8) ----------
FORECAST_CONTEXT_BARS = 40   # real candles shown before the path: enough to read it against, no more
CHART_MIN_TOUCH = 0.05       # a level price has under a 5% chance of reaching only squashes the price scale


def forecast_payload(a: CategoryAnalysis, f, dark: bool, signal: bool = False) -> dict:
    """The clean forecast chart: the last few real candles, the median path as hollow candles, the 68%
    range around it, and the levels with the odds of trading there. No indicators, by design."""
    p = palette(dark, signal)
    df = a.chart_df.tail(FORECAST_CONTEXT_BARS)
    candles = [{"time": int(df["timestamp"].iloc[i]) // 1000, "open": _f(df["open"].iloc[i]),
                "high": _f(df["high"].iloc[i]), "low": _f(df["low"].iloc[i]), "close": _f(df["close"].iloc[i])}
               for i in range(len(df))]
    now = candles[-1]["time"] if candles else 0
    path = [{"time": s.time_ms // 1000, "open": round(s.open, 2), "high": round(s.high, 2),
             "low": round(s.low, 2), "close": round(s.close, 2)} for s in f.steps]
    start = [{"time": now, "value": round(f.price, 2)}]
    band_hi = start + [{"time": s.time_ms // 1000, "value": round(s.band_high, 2)} for s in f.steps]
    band_lo = start + [{"time": s.time_ms // 1000, "value": round(s.band_low, 2)} for s in f.steps]
    levels = [{"price": round(lv.price, 2), "title": f"{lv.label} {lv.p_touch:.0%}",
               "color": p["down"] if lv.side == "above" else p["up"]} for lv in f.levels
              if lv.p_touch >= CHART_MIN_TOUCH]
    return {"theme": {k: p[k] for k in ("panel", "border", "grid", "text", "muted", "accent", "up", "down", "now")},
            "candles": candles, "path": path, "bandHigh": band_hi, "bandLow": band_lo, "levels": levels,
            "now": now, "intraday": TF_MINUTES[a.chart_tf] < 1440}


_FORECAST_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&display=swap" rel="stylesheet">
<style>html,body{margin:0;padding:0;background:__PANEL__;font-family:Barlow,system-ui,sans-serif}#wrap{position:relative}#chart{width:100%;height:__HEIGHT__px}
#axis{position:absolute;left:10px;top:8px;z-index:5;font-size:11px;color:__TEXT__}</style>
<script src="__LIB__"></script></head><body><div id="wrap"><div id="axis"></div><div id="chart"></div></div>
<script>
const P = __PAYLOAD__;
(function(){
const T = P.theme, LW = LightweightCharts, el = document.getElementById('chart');
// every time on this chart reads UTC first, PKT (UTC+5, no daylight saving) after it
const pad = n => String(n).padStart(2, '0');
const DAYS = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'], MONS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const hm = d => pad(d.getUTCHours()) + ':' + pad(d.getUTCMinutes());
const full = t => { const u = new Date(t * 1000), k = new Date(t * 1000 + 5 * 3600 * 1000);
  return hm(u) + ' UTC (' + hm(k) + ' PKT - ' + DAYS[k.getUTCDay()] + ' ' + pad(k.getUTCDate()) + ' ' + MONS[k.getUTCMonth()] + ')'; };
const tick = t => { const u = new Date(t * 1000), k = new Date(t * 1000 + 5 * 3600 * 1000);
  return P.intraday ? hm(u) + ' · ' + hm(k) : pad(u.getUTCDate()) + ' ' + MONS[u.getUTCMonth()]; };
// the axis ticks are short, so the legend names the two clocks; hover gives the full time and date
document.getElementById('axis').textContent = P.intraday ? 'Time axis: UTC · PKT · hollow candles = median path, dashed = 68% range'
                                                          : 'Dates in UTC (hover for PKT) · hollow candles = median path, dashed = 68% range';
const chart = LW.createChart(el, {
  width: el.clientWidth, height: __HEIGHT__,
  layout: { background: { type: 'solid', color: T.panel }, textColor: T.text, fontFamily: 'Barlow, system-ui, sans-serif' },
  grid: { vertLines: { visible: false }, horzLines: { color: T.grid } },
  rightPriceScale: { borderColor: T.border },
  timeScale: { borderColor: T.border, timeVisible: true, secondsVisible: false, rightOffset: 1, tickMarkFormatter: tick },
  localization: { timeFormatter: full },
  crosshair: { mode: 0 }, handleScroll: false, handleScale: false,
});
const real = chart.addSeries(LW.CandlestickSeries, { upColor: T.panel, downColor: T.down, borderVisible: true, borderUpColor: T.up, borderDownColor: T.down, wickUpColor: T.up, wickDownColor: T.down, priceLineVisible: false });
real.setData(P.candles);
// the path: hollow outlines in the accent, so it can never be mistaken for candles that printed
const path = chart.addSeries(LW.CandlestickSeries, { upColor: 'rgba(0,0,0,0)', downColor: 'rgba(0,0,0,0)', borderVisible: true,
  borderUpColor: T.accent, borderDownColor: T.accent, wickUpColor: T.accent, wickDownColor: T.accent,
  priceLineVisible: false, lastValueVisible: true, title: 'median path' });
path.setData(P.path);
[['bandHigh', '84%'], ['bandLow', '16%']].forEach(([k, label]) => {
  const s = chart.addSeries(LW.LineSeries, { color: T.muted, lineWidth: 1, lineStyle: LW.LineStyle.Dashed, priceLineVisible: false, lastValueVisible: true, crosshairMarkerVisible: false, title: label });
  s.setData(P[k]);
});
P.levels.forEach(l => real.createPriceLine({ price: l.price, color: l.color, lineWidth: 1, lineStyle: LW.LineStyle.Dotted, axisLabelVisible: true, title: el.clientWidth < 640 ? '' : l.title }));
chart.timeScale().fitContent();
new ResizeObserver(() => { chart.applyOptions({ width: el.clientWidth }); chart.timeScale().fitContent(); }).observe(el);
})();
</script></body></html>"""


def build_forecast_html(a: CategoryAnalysis, f, dark: bool, height: int = 360, signal: bool = False) -> str:
    p = palette(dark, signal)
    payload = json.dumps(forecast_payload(a, f, dark, signal), separators=(",", ":"))
    return (_FORECAST_TEMPLATE.replace("__LIB__", LIGHTWEIGHT_CHARTS_URL).replace("__HEIGHT__", str(height))
            .replace("__PANEL__", p["panel"]).replace("__TEXT__", p["muted"]).replace("__PAYLOAD__", payload))


def render_forecast_chart(a: CategoryAnalysis, f, dark: bool, height: int = 360, signal: bool = False) -> None:
    components.html(build_forecast_html(a, f, dark, height, signal), height=height + 6, scrolling=False)
