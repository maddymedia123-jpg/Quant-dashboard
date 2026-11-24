from datetime import datetime, timezone, timedelta

PKT_TZ = timezone(timedelta(hours=5))

def format_dual_time(dt_utc: datetime | None = None) -> str:
    """Strict v27.0 Dual-Time Standard: 08:00 UTC (13:00 PKT - Tue 03 Nov)"""
    dt_utc = dt_utc or datetime.now(timezone.utc)
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    dt_pkt = dt_utc.astimezone(PKT_TZ)
    return f"{dt_utc.strftime('%H:%M')} UTC ({dt_pkt.strftime('%H:%M PKT - %a %d %b')})"

DESKS = {
    "live_recon": {
        "label": "LIVE RECON",
        "primary_anchor": "4h",
        "monitor_tf": "1h",                  # Checks 1h windows for major shifts affecting 4h levels
        "cascade_order": ["15m", "1h", "4h"],
        "horizon_label": "Next 4 Hours",
        "window_hours": 4,
    },
    "intraday": {
        "label": "INTRADAY",
        "primary_anchor": "1d",
        "monitor_tf": "4h",
        "cascade_order": ["4h", "12h", "24h", "1d"],
        "horizon_label": "Next 24 Hours",
        "window_hours": 24,
    },
    "weekly": {
        "label": "WEEKLY SWING",
        "primary_anchor": "1w",
        "monitor_tf": "1d",
        "cascade_order": ["4h", "1d", "0.5w", "1w"],
        "horizon_label": "Next 7 Days",
        "window_hours": 168,
    },
    "monthly": {
        "label": "MONTHLY",
        "primary_anchor": "1M",
        "monitor_tf": "2w",                  # 2-week midpoint structural check
        "cascade_order": ["1d", "1w", "2w", "1M"],
        "horizon_label": "Next 4 Weeks",
        "window_hours": 672,
    },
}
