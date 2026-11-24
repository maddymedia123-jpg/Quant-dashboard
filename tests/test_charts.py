import json
import pathlib
import re

from core.config import LIGHTWEIGHT_CHARTS_URL, CATEGORIES
from core.data.kraken_spot import parse_ohlc
from core.data.types import FuturesSnapshot
from core.indicators.category import analyze_category
from ui.charts import build_chart_html, chart_payload

FX = pathlib.Path(__file__).parent / "fixtures"


def _analysis():
    frames = {"15m": parse_ohlc(json.loads((FX / "kraken_ohlc_15m.json").read_text())),
              "1h": parse_ohlc(json.loads((FX / "kraken_ohlc_1h.json").read_text()))}
    return analyze_category(CATEGORIES["live"], frames, FuturesSnapshot(source="binance", oi_change_24h_pct=1.0))


def test_payload_is_json_clean_and_consistent():
    a = _analysis()
    p = chart_payload(a, dark=False)
    json.dumps(p)  # no NaN / numpy types
    times = [c["time"] for c in p["candles"]]
    assert len(times) <= 300 and times == sorted(times) and times[-1] < 10_000_000_000  # seconds
    assert p["whitespace"] and p["whitespace"][0]["time"] > times[-1]
    assert {e["period"] for e in p["emas"]} <= {21, 50, 100, 200}
    assert all(0 <= r["value"] <= 100 for r in p["rsi"]["data"])
    assert p["now"] == times[-1]
    assert not p["projection"] and not p["bandHigh"] and not p["bandLow"]
    assert all(len(lv["title"]) <= 40 for lv in p["levels"])
    assert not {"fibs", "retracements", "trendlines", "patterns", "reversalZone"} & p.keys()


def test_html_embeds_pinned_library_and_payload():
    h = build_chart_html(_analysis(), dark=True, height=500)
    assert LIGHTWEIGHT_CHARTS_URL in h
    m = re.search(r"const PAYLOAD = (\{.*?\});\n", h, re.S)
    assert m, "payload not embedded"
    payload = json.loads(m.group(1))
    assert payload["theme"]["panel"].startswith("#")
    assert "createChart" in h and "attachPrimitive" in h and "createSeriesMarkers" in h


def test_chart_omits_fibonacci_and_pattern_overlays():
    from core.indicators.patterns import Pattern
    a = _analysis()
    t0 = int(a.chart_df["timestamp"].iloc[-40])
    t1 = int(a.chart_df["timestamp"].iloc[-10])
    a.patterns = [Pattern("Double top", "bearish", "forming", 0.8, 1.0, 0.5, 2.0, [[(t0, a.price), (t1, a.price * 1.01)]], t0)]
    p = chart_payload(a, dark=False)
    assert not {"fibs", "retracements", "patterns", "trendlines"} & p.keys()
    h = build_chart_html(a, dark=False)
    assert "SparseDotted" not in h and "LargeDashed" not in h
    assert "P.projection" in h and "Hollow: median path" in h


def test_chart_snapshot_sits_outside_the_canvas_and_preserves_missing_values():
    from ui.panels import chart_snapshot_html

    snapshot = chart_snapshot_html(
        {"available": True, "call_wall": 102_000.0, "zero_gamma_level": 100_000.0, "put_wall": 98_000.0},
        101_000.0, "next 4 hours",
    )
    assert snapshot.index("Ceiling Wall") < snapshot.index("Gamma Flip") < snapshot.index("Median Path") < snapshot.index("Floor Wall")
    assert all(value in snapshot for value in ("$102,000", "$100,000", "$101,000", "$98,000"))
    missing = chart_snapshot_html(None, None)
    assert missing.count("—") == 4
