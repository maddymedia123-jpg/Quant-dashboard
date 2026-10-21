"""When BTC tends to turn: the measured US-session window, in place of Fibonacci time zones.

Measured 21 Oct 2026 on 10,000 hourly Gate.io BTC_USDT perp candles (Aug 2025 - Oct 2026). A turn was a
top or bottom: the extreme within +-6 hours, followed by a move of at least 3 ATR. 395 of them.

* 35% of all turns fell between 12:00 and 16:00 UTC (17:00-21:00 PKT) - four hours, a sixth of the day -
  1.6 to 2.9 times what chance would put there. US data prints at 12:30 UTC and New York opens at 13:30
  (14:30 in winter), which is why. Bigger turns (+-12 h, 5 ATR) showed the same hours at 1.8-2.8x.
* 09:00 UTC, claimed as the institutional turning hour, came out at 1.09x: nothing special.
* Fibonacci time zones, run the way the dashboard draws them with only the data of the moment, landed a
  turn within +-2 hours 17.3% of the time against 18.7% for any random hour (6.2% vs 8.2% for big turns).
  They do not time turns, so this window replaces them wherever a turning time is told to a reader.

The window is a base rate, not a call: most turns happen in it, most of the time in it nothing turns.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

START_HOUR_UTC, END_HOUR_UTC = 12, 16
SHARE_OF_TURNS = 0.35            # of all turns, inside the window
TIMES_CHANCE = (1.6, 2.9)        # per hour, against an even spread
MEASURED = "10,000 hourly BTC perp candles, Aug 2025 - Oct 2026"
HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS


@dataclass(frozen=True)
class TurnWindow:
    start_ms: int
    end_ms: int
    active: bool                 # now is inside it

    @property
    def label(self) -> str:
        """'12:00-16:00 UTC (17:00-21:00 PKT - Wed 21 Oct)' - UTC first, PKT after, as everywhere else."""
        s = datetime.fromtimestamp(self.start_ms / 1000, timezone.utc)
        e = datetime.fromtimestamp(self.end_ms / 1000, timezone.utc)
        ps = datetime.fromtimestamp(self.start_ms / 1000 + 5 * 3600, timezone.utc)
        pe = datetime.fromtimestamp(self.end_ms / 1000 + 5 * 3600, timezone.utc)
        return f"{s:%H:%M}–{e:%H:%M} UTC ({ps:%H:%M}–{pe:%H:%M} PKT - {ps:%a %d %b})"

    @property
    def ends(self) -> str:
        """'16:00 UTC (21:00 PKT)'"""
        e = datetime.fromtimestamp(self.end_ms / 1000, timezone.utc)
        pe = datetime.fromtimestamp(self.end_ms / 1000 + 5 * 3600, timezone.utc)
        return f"{e:%H:%M} UTC ({pe:%H:%M} PKT)"

    @property
    def odds(self) -> str:
        return (f"about {SHARE_OF_TURNS:.0%} of BTC's turns happen in these four hours "
                f"(measured on {MEASURED})")


def next_window(now_ms: int) -> TurnWindow:
    """Today's window if it has not ended yet, otherwise tomorrow's."""
    day = now_ms - now_ms % DAY_MS
    start, end = day + START_HOUR_UTC * HOUR_MS, day + END_HOUR_UTC * HOUR_MS
    if now_ms >= end:
        start, end = start + DAY_MS, end + DAY_MS
    return TurnWindow(start, end, start <= now_ms < end)


def to_state(now_ms: int) -> dict:
    """For the agents: the window, how it was measured, and that Fibonacci timing was tested and failed."""
    w = next_window(now_ms)
    return {"next_window": w.label, "active_now": w.active, "share_of_turns": SHARE_OF_TURNS,
            "times_chance_per_hour": list(TIMES_CHANCE), "measured_on": MEASURED,
            "note": "A base rate for WHEN turns happen, not a direction. Fibonacci time zones were tested on "
                    "the same data and showed no edge (17.3% vs 18.7% random), so do not use them to time "
                    "turns; 09:00 UTC showed no edge either (1.09x)."}
