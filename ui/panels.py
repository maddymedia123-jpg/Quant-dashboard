"""TradingView Lightweight Charts v5 renderer for Streamlit iframes."""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd
import streamlit.components.v1 as components

from core.config import CHART_BARS, LIGHTWEIGHT_CHARTS_URL


def _palette(dark: bool, signal: bool) -> dict[str, str]:
    up = "#16a34a" if signal else "#0284c7"
    down = "#dc2626" if signal else "#d97706"
    if dark:
        return {
            "bg": "#0b0f17",
            "text": "#cbd5e1",
            "grid": "rgba(148, 163, 184, 0.08)",
            "border": "rgba(148, 163, 184, 0.22)",
            "crosshair": "rgba(148, 163, 184, 0.45)",
            "crossLabel": "#1e293b",
            "up": up,
            "down": down,
            "volUp": "rgba(22, 163, 74, 0.35)" if signal else "rgba(2, 132, 199, 0.35)",
            "volDown": "rgba(220, 38, 38, 0.35)" if signal else "rgba(217, 119, 6, 0.35)",
            "ema21": "#38bdf8",
            "ema50": "#f59e0b",
            "ema200": "#a855f7",
            "vwap": "#e2e8f0",
            "bb": "rgba(148, 163, 184, 0.35)",
            "bandFill": "rgba(56, 189, 248, 0.14)",
            "bandLine": "rgba(56, 189, 248, 0.55)",
            "median": "#38bdf8",
        }
    return {
        "bg": "#ffffff",
        "text": "#334155",
        "grid": "rgba(15, 23, 42, 0.06)",
        "border": "rgba(15, 23, 42, 0.16)",
        "crosshair": "rgba(15, 23, 42, 0.35)",
        "crossLabel": "#334155",
        "up": up,
        "down": down,
        "volUp": "rgba(22, 163, 74, 0.28)" if signal else "rgba(2, 132, 199, 0.28)",
        "volDown": "rgba(220, 38, 38, 0.28)" if signal else "rgba(217, 119, 6, 0.28)",
        "ema21": "#0284c7",
        "ema50": "#d97706",
        "ema200": "#7c3aed",
        "vwap": "#0f172a",
        "bb": "rgba(100, 116, 139, 0.35)",
        "bandFill": "rgba(2, 132, 199, 0.12)",
        "bandLine": "rgba(2, 132, 199, 0.50)",
        "median": "#0284c7",
    }


def _to_sec(ts: Any) -> int:
    v = int(ts)
    return v // 1000 if v > 10_000_000_000 else v


def _clean_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _extract_bars(df: pd.DataFrame | None, limit: int = CHART_BARS) -> tuple[list[dict], list[dict], list[dict]]:
    if df is None or df.empty:
        return [], [], []
    sub = df.tail(limit).copy()
    candles: list[dict] = []
    volumes: list[dict] = []
    rsi_pts: list[dict] = []

    closes = sub["close"].astype(float)
    delta = closes.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0.0, np.nan)
    rsi_series = 100.0 - (100.0 / (1.0 + rs))

    for idx, row in sub.iterrows():
        t = _to_sec(row["timestamp"])
        o = _clean_float(row.get("open"))
        h = _clean_float(row.get("high"))
        l = _clean_float(row.get("low"))
        c = _clean_float(row.get("close"))
        if None in (o, h, l, c):
            continue
        is_up = c >= o
        candles.append({"time": t, "open": o, "high": h, "low": l, "close": c})
        vol = _clean_float(row.get("volume"))
        if vol is not None:
            volumes.append({"time": t, "value": vol, "up": is_up})
        rv = _clean_float(rsi_series.loc[idx])
        if rv is not None:
            rsi_pts.append({"time": t, "value": round(rv, 2)})

    return candles, volumes, rsi_pts


def _ema_line(df: pd.DataFrame, span: int, limit: int = CHART_BARS) -> list[dict]:
    if df is None or df.empty or len(df) < 5:
        return []
    sub = df.tail(limit + span)
    ema = sub["close"].astype(float).ewm(span=span, adjust=False).mean().tail(limit)
    ts = sub["timestamp"].tail(limit)
    out = []
    for t, v in zip(ts, ema):
        fv = _clean_float(v)
        if fv is not None:
            out.append({"time": _to_sec(t), "value": round(fv, 2)})
    return out


def _vwap_line(df: pd.DataFrame, limit: int = CHART_BARS) -> list[dict]:
    if df is None or df.empty or "volume" not in df:
        return []
    sub = df.tail(limit).copy()
    tp = (sub["high"].astype(float) + sub["low"].astype(float) + sub["close"].astype(float)) / 3.0
    vol = sub["volume"].astype(float).clip(lower=0.0)
    # Anchor daily VWAP at UTC midnight
    days = (sub["timestamp"].astype("int64") // 86_400_000)
    cum_pv = (tp * vol).groupby(days).cumsum()
    cum_v = vol.groupby(days).cumsum().replace(0.0, np.nan)
    vwap = cum_pv / cum_v
    out = []
    for t, v in zip(sub["timestamp"], vwap):
        fv = _clean_float(v)
        if fv is not None:
            out.append({"time": _to_sec(t), "value": round(fv, 2)})
    return out


def _price_lines_from_analysis(a: Any, gamma: dict | None = None) -> list[dict]:
    lines: list[dict] = []
    if a is None:
        return lines

    direction = getattr(a, "direction", None)
    if direction is not None:
        inv = _clean_float(getattr(direction, "invalidation", None))
        if inv:
            lines.append({"price": inv, "color": "#ef4444", "style": 2, "title": "Invalidation"})
        t1 = _clean_float(getattr(direction, "target_1", None))
        if t1:
            lines.append({"price": t1, "color": "#22c55e", "style": 2, "title": "Target 1"})
        t2 = _clean_float(getattr(direction, "target_2", None))
        if t2:
            lines.append({"price": t2, "color": "#16a34a", "style": 3, "title": "Target 2"})

    vp = getattr(a, "volume_profile", None)
    if vp is not None and getattr(vp, "available", False):
        poc = _clean_float(getattr(vp, "poc", None))
        vah = _clean_float(getattr(vp, "value_area_high", None))
        val = _clean_float(getattr(vp, "value_area_low", None))
        if poc:
            lines.append({"price": poc, "color": "#eab308", "style": 0, "title": "POC"})
        if vah:
            lines.append({"price": vah, "color": "#64748b", "style": 2, "title": "VAH"})
        if val:
            lines.append({"price": val, "color": "#64748b", "style": 2, "title": "VAL"})

    if isinstance(gamma, dict) and gamma.get("available"):
        cw = _clean_float(gamma.get("call_wall"))
        pw = _clean_float(gamma.get("put_wall"))
        zg = _clean_float(gamma.get("zero_gamma_level"))
        if cw:
            lines.append({"price": cw, "color": "#3b82f6", "style": 1, "title": "Call Wall"})
        if pw:
            lines.append({"price": pw, "color": "#f97316", "style": 1, "title": "Put Wall"})
        if zg:
            lines.append({"price": zg, "color": "#a855f7", "style": 2, "title": "Zero Gamma"})

    return lines


def _render_lw_html(payload: dict, height: int) -> None:
    cfg_json = json.dumps(payload)
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<style>
  html, body {{
    margin: 0;
    padding: 0;
    width: 100%;
    height: {height}px;
    overflow: hidden;
    background: {payload['palette']['bg']};
  }}
  #chart-root {{
    width: 100%;
    height: {height}px;
    position: relative;
  }}
</style>
<script src="{LIGHTWEIGHT_CHARTS_URL}"></script>
</head>
<body>
<div id="chart-root"></div>
<script>
(function() {{
  const CFG = {cfg_json};
  const P = CFG.palette;
  const el = document.getElementById("chart-root");

  function getWidth() {{
    return el.clientWidth || document.documentElement.clientWidth || window.innerWidth || 920;
  }}

  function boot() {{
    if (!window.LightweightCharts) {{
      el.innerHTML = '<div style="padding:16px;color:' + P.text + ';font-family:sans-serif;">Chart library failed to load from CDN.</div>';
      return;
    }}
    const LW = window.LightweightCharts;
    const chart = LW.createChart(el, {{
      width: getWidth(),
      height: CFG.height,
      autoSize: true,
      layout: {{
        background: {{ type: LW.ColorType.Solid, color: P.bg }},
        textColor: P.text,
        fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif",
        fontSize: 11,
      }},
      grid: {{
        vertLines: {{ color: P.grid }},
        horzLines: {{ color: P.grid }},
      }},
      crosshair: {{
        mode: LW.CrosshairMode.Normal,
        vertLine: {{ color: P.crosshair, labelBackgroundColor: P.crossLabel }},
        horzLine: {{ color: P.crosshair, labelBackgroundColor: P.crossLabel }},
      }},
      rightPriceScale: {{
        borderColor: P.border,
        scaleMargins: {{ top: 0.08, bottom: 0.28 }},
      }},
      timeScale: {{
        borderColor: P.border,
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 12,
      }},
    }});

    const candleSeries = chart.addSeries(LW.CandlestickSeries, {{
      upColor: P.up,
      downColor: P.down,
      borderUpColor: P.up,
      borderDownColor: P.down,
      wickUpColor: P.up,
      wickDownColor: P.down,
    }});
    candleSeries.setData(CFG.candles || []);

    if (CFG.projCandles && CFG.projCandles.length) {{
      const projSeries = chart.addSeries(LW.CandlestickSeries, {{
        upColor: "rgba(0,0,0,0)",
        downColor: "rgba(0,0,0,0)",
        borderUpColor: P.up,
        borderDownColor: P.down,
        wickUpColor: P.up,
        wickDownColor: P.down,
      }});
      projSeries.setData(CFG.projCandles);
    }}

    if (CFG.bandHigh && CFG.bandHigh.length) {{
      const hiLine = chart.addSeries(LW.LineSeries, {{
        color: P.bandLine,
        lineWidth: 1,
        lineStyle: 2,
        priceLineVisible: false,
        lastValueVisible: false,
      }});
      hiLine.setData(CFG.bandHigh);
    }}

    if (CFG.bandLow && CFG.bandLow.length) {{
      const loLine = chart.addSeries(LW.LineSeries, {{
        color: P.bandLine,
        lineWidth: 1,
        lineStyle: 2,
        priceLineVisible: false,
        lastValueVisible: false,
      }});
      loLine.setData(CFG.bandLow);
    }}

    if (CFG.medianLine && CFG.medianLine.length) {{
      const medLine = chart.addSeries(LW.LineSeries, {{
        color: P.median,
        lineWidth: 2,
        priceLineVisible: false,
      }});
      medLine.setData(CFG.medianLine);
    }}

    if (CFG.ema21 && CFG.ema21.length) {{
      const s21 = chart.addSeries(LW.LineSeries, {{ color: P.ema21, lineWidth: 1, priceLineVisible: false, lastValueVisible: false }});
      s21.setData(CFG.ema21);
    }}
    if (CFG.ema50 && CFG.ema50.length) {{
      const s50 = chart.addSeries(LW.LineSeries, {{ color: P.ema50, lineWidth: 1, priceLineVisible: false, lastValueVisible: false }});
      s50.setData(CFG.ema50);
    }}
    if (CFG.ema200 && CFG.ema200.length) {{
      const s200 = chart.addSeries(LW.LineSeries, {{ color: P.ema200, lineWidth: 1, priceLineVisible: false, lastValueVisible: false }});
      s200.setData(CFG.ema200);
    }}
    if (CFG.vwap && CFG.vwap.length) {{
      const sv = chart.addSeries(LW.LineSeries, {{ color: P.vwap, lineWidth: 1, lineStyle: 1, priceLineVisible: false, lastValueVisible: false }});
      sv.setData(CFG.vwap);
    }}

    (CFG.priceLines || []).forEach(pl => {{
      candleSeries.createPriceLine({{
        price: pl.price,
        color: pl.color,
        lineWidth: 1,
        lineStyle: pl.style || 0,
        axisLabelVisible: true,
        title: pl.title || "",
      }});
    }});

    if (CFG.volumes && CFG.volumes.length) {{
      const volSeries = chart.addSeries(LW.HistogramSeries, {{
        priceFormat: {{ type: "volume" }},
        priceScaleId: "vol",
      }});
      chart.priceScale("vol").applyOptions({{
        scaleMargins: {{ top: 0.78, bottom: 0.0 }},
      }});
      volSeries.setData(CFG.volumes.map(v => ({{
        time: v.time,
        value: v.value,
        color: v.up ? P.volUp : P.volDown,
      }})));
    }}

    chart.timeScale().fitContent();

    if (window.ResizeObserver) {{
      const ro = new ResizeObserver(() => {{
        const w = getWidth();
        if (w > 0) chart.applyOptions({{ width: w, height: CFG.height }});
      }});
      ro.observe(el);
    }}
    window.addEventListener("load", () => {{
      const w = getWidth();
      if (w > 0) chart.applyOptions({{ width: w, height: CFG.height }});
    }});
    setTimeout(() => {{
      const w = getWidth();
      if (w > 0) chart.applyOptions({{ width: w, height: CFG.height }});
    }}, 150);
  }}

  boot();
}})();
</script>
</body>
</html>"""
    components.html(html, height=height, scrolling=False)


def render_chart(a: Any, dark: bool = False, height: int = 560, signal: bool = False) -> None:
    df = getattr(a, "df", None)
    candles, volumes, _rsi = _extract_bars(df, CHART_BARS)
    payload = {
        "height": height,
        "palette": _palette(dark, signal),
        "candles": candles,
        "volumes": volumes,
        "ema21": _ema_line(df, 21, CHART_BARS),
        "ema50": _ema_line(df, 50, CHART_BARS),
        "ema200": _ema_line(df, 200, CHART_BARS),
        "vwap": _vwap_line(df, CHART_BARS),
        "priceLines": _price_lines_from_analysis(a),
    }
    _render_lw_html(payload, height)


def render_projected_chart(
    a: Any,
    dark: bool = False,
    forecast: Any = None,
    gamma: dict | None = None,
    forecast_tf: str | None = None,
    height: int = 560,
    signal: bool = False,
) -> None:
    df = getattr(a, "df", None)
    candles, volumes, _rsi = _extract_bars(df, CHART_BARS)

    proj_candles: list[dict] = []
    band_high: list[dict] = []
    band_low: list[dict] = []
    median_line: list[dict] = []

    if forecast is not None and getattr(forecast, "available", False):
        steps = getattr(forecast, "steps", None) or getattr(forecast, "bars", None) or []
        prev_close = candles[-1]["close"] if candles else None
        for stp in steps:
            t_raw = getattr(stp, "t_ms", None) or getattr(stp, "timestamp", None)
            if t_raw is None:
                continue
            t = _to_sec(t_raw)
            med = _clean_float(getattr(stp, "median", None) or getattr(stp, "price", None))
            hi = _clean_float(getattr(stp, "band_high", None) or getattr(stp, "high", None))
            lo = _clean_float(getattr(stp, "band_low", None) or getattr(stp, "low", None))
            if med is not None:
                median_line.append({"time": t, "value": round(med, 2)})
                if prev_close is not None:
                    proj_candles.append({
                        "time": t,
                        "open": round(prev_close, 2),
                        "high": round(max(prev_close, med, hi or med), 2),
                        "low": round(min(prev_close, med, lo or med), 2),
                        "close": round(med, 2),
                    })
                prev_close = med
            if hi is not None:
                band_high.append({"time": t, "value": round(hi, 2)})
            if lo is not None:
                band_low.append({"time": t, "value": round(lo, 2)})

    payload = {
        "height": height,
        "palette": _palette(dark, signal),
        "candles": candles,
        "volumes": volumes,
        "projCandles": proj_candles,
        "bandHigh": band_high,
        "bandLow": band_low,
        "medianLine": median_line,
        "ema21": _ema_line(df, 21, CHART_BARS),
        "ema50": _ema_line(df, 50, CHART_BARS),
        "ema200": _ema_line(df, 200, CHART_BARS),
        "vwap": _vwap_line(df, CHART_BARS),
        "priceLines": _price_lines_from_analysis(a, gamma),
    }
    _render_lw_html(payload, height)


def render_qf_chart(
    plan: dict,
    h1: pd.DataFrame,
    dark: bool = True,
    height: int = 520,
    signal: bool = True,
    now_ms: int | None = None,
) -> None:
    candles, volumes, _rsi = _extract_bars(h1, 168)
    proj_candles: list[dict] = []
    band_high: list[dict] = []
    band_low: list[dict] = []
    median_line: list[dict] = []

    prev = _clean_float(plan.get("origin"))
    for h in plan.get("hours") or []:
        t = _to_sec(h["t_ms"])
        med = _clean_float(h.get("median"))
        hi = _clean_float(h.get("band_high"))
        lo = _clean_float(h.get("band_low"))
        if med is not None:
            median_line.append({"time": t, "value": round(med, 2)})
            if prev is not None:
                proj_candles.append({
                    "time": t,
                    "open": round(prev, 2),
                    "high": round(max(prev, med, hi or med), 2),
                    "low": round(min(prev, med, lo or med), 2),
                    "close": round(med, 2),
                })
            prev = med
        if hi is not None:
            band_high.append({"time": t, "value": round(hi, 2)})
        if lo is not None:
            band_low.append({"time": t, "value": round(lo, 2)})

    price_lines: list[dict] = []
    inv = plan.get("invalidation") or {}
    if inv.get("below"):
        price_lines.append({"price": float(inv["below"]), "color": "#ef4444", "style": 2, "title": "Invalidation Low"})
    if inv.get("above"):
        price_lines.append({"price": float(inv["above"]), "color": "#ef4444", "style": 2, "title": "Invalidation High"})
    ph = (plan.get("projected_high") or {}).get("price")
    pl = (plan.get("projected_low") or {}).get("price")
    if ph:
        price_lines.append({"price": float(ph), "color": "#22c55e", "style": 1, "title": "Proj High"})
    if pl:
        price_lines.append({"price": float(pl), "color": "#f97316", "style": 1, "title": "Proj Low"})

    payload = {
        "height": height,
        "palette": _palette(dark, signal),
        "candles": candles,
        "volumes": volumes,
        "projCandles": proj_candles,
        "bandHigh": band_high,
        "bandLow": band_low,
        "medianLine": median_line,
        "priceLines": price_lines,
    }
    _render_lw_html(payload, height)
