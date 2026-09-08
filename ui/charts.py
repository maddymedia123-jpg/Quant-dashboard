"""Serialise a CategoryAnalysis into a self-contained TradingView lightweight-charts document."""
from __future__ import annotations

import json
import math

import streamlit.components.v1 as components

from core.config import CATEGORIES, LIGHTWEIGHT_CHARTS_URL, TF_MINUTES
from core.indicators.category import CategoryAnalysis
from ui.theme import palette

EMA_COLORS = {21: "#F59E0B", 50: "#3B82F6", 100: "#8B5CF6", 200: "#EF4444"}


def _f(v) -> float | None:
    if v is None:
        return None
    v = float(v)
    return None if math.isnan(v) or math.isinf(v) else v


def chart_payload(a: CategoryAnalysis, dark: bool) -> dict:
    p = palette(dark)
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

    price = a.price
    proj_end = last_t + min(future_n, max(horizon, 1)) * interval
    hi = _f(a.vol.exp_high) or price
    lo = _f(a.vol.exp_low) or price
    sign = {"BULLISH": 1.0, "BEARISH": -1.0}.get(a.direction.direction, 0.0)
    mid_end = price + sign * a.direction.confidence * ((hi - price) if sign >= 0 else (price - lo))
    projection = {
        "color": p["accent"],
        "upper": [{"time": last_t, "value": price}, {"time": proj_end, "value": hi}],
        "lower": [{"time": last_t, "value": price}, {"time": proj_end, "value": lo}],
        "mid": [{"time": last_t, "value": price}, {"time": proj_end, "value": mid_end}],
    }

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
        "trendlines": trendlines, "fibs": fibs, "now": last_t, "projection": projection,
        "rsi": {"data": rsi_data, "markers": markers},
        "visibleBars": 140,
    }


_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>html,body{margin:0;padding:0;background:__PANEL__;font-family:Inter,system-ui,sans-serif}#wrap{position:relative}#chart{width:100%;height:__HEIGHT__px}
#legend{position:absolute;left:10px;top:8px;z-index:5;font-size:11px;color:__TEXT__;display:flex;gap:10px;flex-wrap:wrap}
#legend span::before{content:"";display:inline-block;width:10px;height:2px;margin-right:4px;vertical-align:middle;background:var(--c)}</style>
<script src="__LIB__"></script></head><body><div id="wrap"><div id="legend"></div><div id="chart"></div></div>
<script>
const PAYLOAD = __PAYLOAD__;
(function(){
const P = PAYLOAD, T = P.theme, LW = LightweightCharts;
const el = document.getElementById('chart');
const chart = LW.createChart(el, {
  width: el.clientWidth, height: __HEIGHT__,
  layout: { background: { type: 'solid', color: T.panel }, textColor: T.text, fontFamily: 'Inter, system-ui, sans-serif',
            panes: { separatorColor: T.border, separatorHoverColor: T.border, enableResize: false } },
  grid: { vertLines: { color: T.grid }, horzLines: { color: T.grid } },
  rightPriceScale: { borderColor: T.border },
  timeScale: { borderColor: T.border, timeVisible: true, secondsVisible: false, rightOffset: 2 },
  crosshair: { mode: 0 },
});
const candles = chart.addSeries(LW.CandlestickSeries, { upColor: T.up, downColor: T.down, borderVisible: false, wickUpColor: T.up, wickDownColor: T.down });
candles.setData(P.candles.concat(P.whitespace));
const legend = document.getElementById('legend');
P.emas.forEach(e => {
  const s = chart.addSeries(LW.LineSeries, { color: e.color, lineWidth: 1, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
  s.setData(e.data);
  const tag = document.createElement('span'); tag.style.setProperty('--c', e.color); tag.textContent = 'EMA' + e.period; legend.appendChild(tag);
});
P.levels.forEach(l => candles.createPriceLine({ price: l.price, color: l.color, lineWidth: 1, lineStyle: LW.LineStyle.Dashed, axisLabelVisible: true, title: l.title }));
P.trendlines.forEach(t => {
  const s = chart.addSeries(LW.LineSeries, { color: t.color, lineWidth: 1, lineStyle: LW.LineStyle.Dotted, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
  s.setData(t.data);
});
['upper', 'lower', 'mid'].forEach(k => {
  const s = chart.addSeries(LW.LineSeries, { color: P.projection.color, lineWidth: k === 'mid' ? 2 : 1, lineStyle: LW.LineStyle.Dashed,
    priceLineVisible: false, lastValueVisible: k !== 'mid', crosshairMarkerVisible: false, title: k === 'mid' ? 'projection' : '' });
  s.setData(P.projection[k]);
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
      if (this._label) { ctx.setLineDash([]); ctx.fillStyle = this._color; ctx.font = `${11 * vr}px Inter, sans-serif`; ctx.fillText(this._label, x + 4 * hr, 12 * vr); }
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


def build_chart_html(a: CategoryAnalysis, dark: bool, height: int = 560) -> str:
    p = palette(dark)
    payload = json.dumps(chart_payload(a, dark), separators=(",", ":"))
    return (_TEMPLATE.replace("__LIB__", LIGHTWEIGHT_CHARTS_URL).replace("__HEIGHT__", str(height))
            .replace("__PANEL__", p["panel"]).replace("__TEXT__", p["text"]).replace("__PAYLOAD__", payload))


def render_chart(a: CategoryAnalysis, dark: bool, height: int = 560) -> None:
    components.html(build_chart_html(a, dark, height), height=height + 6, scrolling=False)
