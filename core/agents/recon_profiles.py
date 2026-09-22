"""One war room per category.

The client's spec: "In Live Recon, we have 11 agents, same like intraday, weekly, and monthly." Each
category runs the same eleven-agent model and the same rubric, but reads its own three timeframes,
judges over its own horizon, and pins its summary to its own candle.

Live Recon's timeframes, horizon and 4h window are exactly what the client specified. The other three
follow the same shape one step up the ladder. We fetch nothing above the weekly candle, so Weekly and
Monthly read the same 4h/1d/1w charts and differ in horizon and in the candle they anchor to."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS
WEEK_MS = 7 * DAY_MS
# The Unix epoch fell on a Thursday; weekly candles open on Monday 00:00 UTC, four days later.
MONDAY_OFFSET_MS = 4 * DAY_MS

# Which category analysis is drawn on which chart timeframe.
ANALYSIS_FOR_TF = {"15m": "live", "1h": "intraday", "4h": "weekly", "1d": "monthly"}


@dataclass(frozen=True)
class ReconProfile:
    category: str
    label: str
    timeframes: tuple[str, str, str]    # lower, middle, higher
    horizon: str                        # spelled out, as it reads inside a question
    window: str                         # "4h", "1d", "1w" or "1M"
    window_title: str                   # "4-hour anchored summary"
    window_label: str                   # "the current 4h candle"

    @property
    def fmt(self) -> dict[str, str]:
        ltf, mtf, htf = self.timeframes
        return {"ltf": ltf, "mtf": mtf, "htf": htf, "horizon": self.horizon}

    @property
    def analysis_keys(self) -> dict[str, str]:
        """Category analyses whose chart sits on one of this profile's timeframes."""
        return {ANALYSIS_FOR_TF[tf]: tf for tf in self.timeframes if tf in ANALYSIS_FOR_TF}

    def window_open(self, now_ms: int) -> int:
        if self.window == "4h":
            return now_ms - now_ms % (4 * HOUR_MS)
        if self.window == "1d":
            return now_ms - now_ms % DAY_MS
        if self.window == "1w":
            return now_ms - (now_ms - MONDAY_OFFSET_MS) % WEEK_MS
        if self.window == "1M":
            d = datetime.fromtimestamp(now_ms / 1000, timezone.utc)
            return int(datetime(d.year, d.month, 1, tzinfo=timezone.utc).timestamp() * 1000)
        raise ValueError(f"unknown window {self.window!r}")

    def window_close(self, now_ms: int) -> int:
        start = self.window_open(now_ms)
        if self.window == "4h":
            return start + 4 * HOUR_MS
        if self.window == "1d":
            return start + DAY_MS
        if self.window == "1w":
            return start + WEEK_MS
        d = datetime.fromtimestamp(start / 1000, timezone.utc)
        year, month = (d.year + 1, 1) if d.month == 12 else (d.year, d.month + 1)
        return int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp() * 1000)


PROFILES: dict[str, ReconProfile] = {
    "live": ReconProfile("live", "Live Recon", ("15m", "1h", "4h"), "four hours", "4h", "4-hour", "4h"),
    "intraday": ReconProfile("intraday", "Intraday", ("1h", "4h", "1d"), "24 hours", "1d", "Daily", "daily"),
    "weekly": ReconProfile("weekly", "Weekly", ("4h", "1d", "1w"), "seven days", "1w", "Weekly", "weekly"),
    "monthly": ReconProfile("monthly", "Monthly", ("4h", "1d", "1w"), "30 days", "1M", "Monthly", "monthly"),
}
LIVE = PROFILES["live"]
