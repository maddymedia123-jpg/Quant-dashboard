"""Static configuration: categories, cache TTLs, symbols, curated macro events."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Category:
    key: str            # machine key
    label: str          # UI label
    chart_tf: str       # timeframe drawn on the chart
    context_tf: str     # higher timeframe used for context
    horizon_bars: int   # bars of chart_tf used for expected-range horizon
    horizon_label: str  # human label for that horizon


CATEGORIES: dict[str, Category] = {
    # 16 bars of 15m is four hours, which is the window the Live war room judges and its summary
    # anchors to. At 4 bars the chart forecast and expected range covered one hour while the
    # verdict beside them covered four.
    "live": Category("live", "Live Recon", "15m", "1h", 16, "next 4h"),
    "intraday": Category("intraday", "Intraday", "1h", "4h", 24, "next 24h"),
    "weekly": Category("weekly", "Weekly", "4h", "1d", 42, "next 7 days"),
    "monthly": Category("monthly", "Monthly", "1d", "1w", 30, "next 30 days"),
}

TIMEFRAMES: tuple[str, ...] = ("15m", "1h", "4h", "1d", "1w")

TF_MINUTES: dict[str, int] = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440, "1w": 10080}

# Cache TTLs (seconds)
TTL = {"spot": 25, "context": 120, "futures": 60, "options": 120, "sentiment": 1800}

SYMBOLS = {"kraken_pair": "XBTUSD", "binance": "BTCUSDT", "bybit": "BTCUSDT", "deribit": "BTC"}

CHART_BARS = 300  # bars drawn on each chart

LIGHTWEIGHT_CHARTS_URL = (
    "https://cdn.jsdelivr.net/npm/lightweight-charts@5.2.1/dist/lightweight-charts.standalone.production.js"
)

# Manually curated until a news source is chosen (spec §11). Shown with a "curated" label.
MACRO_EVENTS = [
    {
        "title": "FOMC Rate Decision & Statement",
        "date_utc": "2026-09-17",
        "impact": "HIGH",
        "note": "No new entries into the release; flatten or hedge leveraged positions before 18:00 UTC.",
    }
]
