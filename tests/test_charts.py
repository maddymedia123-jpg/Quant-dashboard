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
    assert all(len(t["data"]) == 2 for t in p["trendlines"])
    assert all(len(p["projection"][k]) == 2 for k in ("upper", "lower", "mid"))
    assert all(f["time"] % 1 == 0 for f in p["fibs"])


def test_html_embeds_pinned_library_and_payload():
    h = build_chart_html(_analysis(), dark=True, height=500)
    assert LIGHTWEIGHT_CHARTS_URL in h
    m = re.search(r"const PAYLOAD = (\{.*?\});\n", h, re.S)
    assert m, "payload not embedded"
    payload = json.loads(m.group(1))
    assert payload["theme"]["panel"].startswith("#")
    assert "createChart" in h and "attachPrimitive" in h and "createSeriesMarkers" in h
