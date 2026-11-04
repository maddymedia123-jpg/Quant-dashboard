"""The per-desk chart switch (his v22 Section 3 Main Chart View, owner 4 Nov): the desk's own candle by
default, one coarser view a click away. Only the candles and the overlays read on them change; the levels
the call stands on, the forecast and the summary stay on the desk's candle."""
import json
import pathlib

from core.config import CATEGORIES, CHART_VIEWS, TF_MINUTES, TF_LADDER
from core.data.gemini_spot import resample
from core.data.kraken_spot import parse_ohlc
from core.data.types import FuturesSnapshot
from core.indicators.category import analyze_category, view_analysis
from ui import panels
from ui.charts import chart_payload

FX = pathlib.Path(__file__).parent / "fixtures"
FUT = FuturesSnapshot(source="binance", oi_change_24h_pct=1.0)


def frames():
    h1 = parse_ohlc(json.loads((FX / "kraken_ohlc_1h.json").read_text()))
    return {"15m": parse_ohlc(json.loads((FX / "kraken_ohlc_15m.json").read_text())), "1h": h1,
            "4h": resample(h1, 240), "1d": resample(h1, 1440)}


def test_every_desk_defaults_to_its_own_candle_with_one_coarser_view():
    for key, cat in CATEGORIES.items():
        default, coarser = CHART_VIEWS[key]
        assert default == cat.chart_tf, f"{key}: the default is the desk's own candle, unchanged"
        assert TF_MINUTES[coarser] > TF_MINUTES[default] and coarser in TF_LADDER
    assert CHART_VIEWS == {"live": ("1h", "4h"), "intraday": ("4h", "1d"), "weekly": ("1d", "3d"), "monthly": ("1w", "2w")}


def test_the_coarser_view_redraws_the_candles_but_keeps_the_desks_levels():
    fr = frames()
    base = analyze_category(CATEGORIES["live"], fr, FUT)
    assert view_analysis(CATEGORIES["live"], "1h", fr, FUT, base) is base, "the default view is the analysis itself"
    v = view_analysis(CATEGORIES["live"], "4h", fr, FUT, base)
    assert v is not None and v.chart_tf == "4h" and v.key == "live"
    steps = set(v.chart_df["timestamp"].diff().dropna().astype(int))
    assert steps == {4 * 3_600_000}, "4-hour candles"
    assert v.levels == base.levels, "the call's levels are drawn where the call was made"
    assert len(v.chart_df) < len(base.chart_df) and v.ema is not base.ema, "the overlays are read on the view's candles"
    assert view_analysis(CATEGORIES["live"], "3d", fr, FUT, base) is None, "no 3d frame here: the caller falls back"


def test_the_chart_names_the_view_and_keeps_the_desks_horizon_in_time():
    fr = frames()
    base = analyze_category(CATEGORIES["live"], fr, FUT)
    v = view_analysis(CATEGORIES["live"], "4h", fr, FUT, base)
    p = chart_payload(v, dark=True)
    assert p["title"] == "Live Recon · 4h view · call on 1h"
    assert p["whitespace"][0]["time"] - p["now"] == 4 * 3600, "the future axis steps in the view's candles"
    assert chart_payload(base, dark=True)["title"] == "Live Recon · 1h"
    head = panels.chart_head_html(v, False, "BTC/USDT", call_tf="1h")
    assert "4h view · call on 1h" in head
    assert "view" not in panels.chart_head_html(base, False, "BTC/USDT", call_tf="1h")
