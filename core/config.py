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


# The master framework's desks. Each charts one candle size below the candle it is anchored to, and
# forecasts that many chart candles ahead, so the forecast always spans exactly the anchored window:
#   Live Recon  - 1h chart, 4 x 1h  = the 4-hour candle it is pinned to
#   Intraday    - 4h chart, 6 x 4h  = the daily candle
#   Weekly      - 1D chart, 7 x 1D  = the weekly candle (Mon 00:00 UTC to Sunday close)
#   Monthly     - 1W chart, 4 x 1W  = the monthly candle
CATEGORIES: dict[str, Category] = {
    "live": Category("live", "Live Recon", "1h", "4h", 4, "next 4h"),
    "intraday": Category("intraday", "Intraday", "4h", "1d", 6, "next 24h"),
    "weekly": Category("weekly", "Weekly", "1d", "1w", 7, "next 7 days"),
    "monthly": Category("monthly", "Monthly", "1w", "1M", 4, "next month"),
}

# Fetched from the exchange. 15m is still fetched: the 24h change, the trade desk's first checkpoint
# and the TRAP sweep's finest look all read it.
TIMEFRAMES: tuple[str, ...] = ("15m", "1h", "4h", "1d", "1w")
# Built from those: no free spot venue serves two-week or calendar-month candles, so 2W is two
# Monday-aligned weeks and 1M is a calendar month of daily candles (see core.data.spot).
DERIVED_TIMEFRAMES: tuple[str, ...] = ("2w", "1M")

# 1M is a nominal 30 days, used only where a duration is needed (annualising volatility, horizons);
# month boundaries themselves are always taken from the calendar.
TF_MINUTES: dict[str, int] = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440, "1w": 10080, "2w": 20160, "1M": 43200}

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
