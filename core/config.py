"""Static configuration: categories, cache TTLs, symbols, curated macro events."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


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
#   Live Recon  - 1h chart, 4 x 1h   = the 4-hour candle it is pinned to
#   Intraday    - 12h chart, 2 x 12h = the daily candle (24h horizon)
#   Weekly      - 1D chart, 7 x 1D   = the weekly candle (Mon 00:00 UTC to Sunday close)
#   Monthly     - 1W chart, 4 x 1W   = the monthly candle
CATEGORIES: dict[str, Category] = {
    "live": Category("live", "Live Recon", "1h", "4h", 4, "next 4h"),
    "intraday": Category("intraday", "Intraday", "12h", "1d", 2, "next 24h"),
    "weekly": Category("weekly", "Weekly", "1d", "1w", 7, "next 7 days"),
    "monthly": Category("monthly", "Monthly", "1w", "1M", 4, "next month"),
}

# His v22 Section 3 "Main Chart View" (owner 4 Nov): each desk draws its own candle by default, with one
# coarser view a switch away. The call, the forecast, the summary and every read stay on the default;
# only the candles drawn, and the overlays read on them, change (core.indicators.category.view_analysis).
# Live Recon draws the 4-hour chart by default, as his table reads ("Main Chart View: 4-Hour Chart"); the
# 1h, which its call is made on, is the other view. Intraday defaults to 12h with 1d a switch away.
CHART_VIEWS: dict[str, tuple[str, str]] = {"live": ("4h", "1h"), "intraday": ("12h", "1d"),
                                          "weekly": ("1d", "3d"), "monthly": ("1w", "2w")}
# Every candle size the snapshot can carry, shortest first: the next one up is a view's context.
TF_LADDER: tuple[str, ...] = ("15m", "1h", "4h", "12h", "1d", "3d", "0.5w", "1w", "2w", "1M", "3M")

# Fetched from the exchange. 15m is still fetched: the 24h change, the trade desk's first checkpoint
# and the TRAP sweep's finest look all read it.
TIMEFRAMES: tuple[str, ...] = ("15m", "1h", "4h", "1d", "1w")
# Built from those (see core.data.spot): 12h is two UTC half-days of hourly candles; 3d is three days
# counted from a fixed epoch Monday; 0.5w is a half-week of 4h candles (Monday 00:00 and Thursday
# 12:00 UTC); 2W is two Monday-aligned weeks; 1M and 3M are calendar months and quarters of daily
# candles. His v22 Section 3 cascade (4H/12H/24H -> 1D, 4H -> 1D -> 0.5W -> 1W, 1D -> 1W -> 2W -> 1M)
# reads them; 24H is the UTC daily candle, which the 1d frame already is.
DERIVED_TIMEFRAMES: tuple[str, ...] = ("12h", "3d", "0.5w", "2w", "1M", "3M")

# 1M is a nominal 30 days, used only where a duration is needed (annualising volatility, horizons);
# month boundaries themselves are always taken from the calendar.
TF_MINUTES: dict[str, int] = {"15m": 15, "1h": 60, "4h": 240, "12h": 720, "1d": 1440, "3d": 4320, "0.5w": 5040,
                              "1w": 10080, "2w": 20160, "1M": 43200, "3M": 129600}

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

PKT_TZ = timezone(timedelta(hours=5))


def format_dual_time(dt_utc: datetime | None = None) -> str:
    """Render one instant in UTC and fixed UTC+5 Pakistan time."""
    dt_utc = dt_utc or datetime.now(timezone.utc)
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    else:
        dt_utc = dt_utc.astimezone(timezone.utc)
    dt_pkt = dt_utc.astimezone(PKT_TZ)
    return f"{dt_utc.strftime('%H:%M')} UTC ({dt_pkt.strftime('%H:%M PKT - %a %d %b')})"


# Dispatch context is distinct from the ordinary per-desk candle horizon.
DESKS = {
    "live_recon": {"label": "LIVE RECON", "primary_anchor": "4h", "monitor_tf": "1h",
                   "cascade_order": ["15m", "1h", "4h"], "horizon_label": "Next 4 Hours", "window_hours": 4},
    "live": {"label": "LIVE RECON", "horizon_label": "Next 4 Hours", "window_hours": 4},
    "intraday": {"label": "INTRADAY", "primary_anchor": "1d", "monitor_tf": "12h",
                 "cascade_order": ["4h", "12h", "24h", "1d"], "horizon_label": "Next 24 Hours",
                 "window_hours": 24, "master_tier": "INTRADAY / SWING MASTER",
                 "move_context": "BTC price move: $1,000-$3,000+"},
    "weekly": {"label": "WEEKLY SWING", "primary_anchor": "1w", "monitor_tf": "1d",
               "cascade_order": ["4h", "1d", "0.5w", "1w"], "horizon_label": "Next 7 Days",
               "window_hours": 168, "master_tier": "INTRADAY / SWING MASTER",
               "move_context": "BTC price move: $1,000-$3,000+"},
    "monthly": {"label": "MONTHLY", "primary_anchor": "1M", "monitor_tf": "2w",
                "cascade_order": ["1d", "1w", "2w", "1M"], "horizon_label": "Next 4 Weeks",
                "window_hours": 672, "master_tier": "LONG / MONTHLY MASTER",
                "move_context": "monthly structural range, set by its own levels"},
}
