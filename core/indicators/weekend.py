"""His weekend microstructure modifier: spot volume falls to a fraction of the week's on weekends, so the
playbook changes. Breakout continuation is not trusted; the trade is the liquidity sweep and rejection at the
range edges, and moves run further and reverse faster than the levels suggest.

The weekend here is Friday 21:00 UTC (the US cash close, when volume thins) to Sunday 22:00 UTC (the CME
open, when the week's flow returns). Everything downstream reads this one definition: the desk flag, the
summary's wording, the confluence gate's structure family and the war-room state."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

START = (4, 21)     # Friday 21:00 UTC
END = (6, 22)       # Sunday 22:00 UTC
H = 3_600_000
PKT_MS = 5 * H

PLAYBOOK = ("sweep and rejection at the range edges, not breakout continuation: expect the zone to be run "
            "through and rejected rather than held cleanly, and a break to fail more often than it follows through")


@dataclass(frozen=True)
class Weekend:
    active: bool
    phase: str            # weekend | starting | weekday
    note: str
    until_ms: int         # when the current phase ends (the weekend's end, or its start)

    def to_state(self) -> dict:
        return {"weekend": self.active, "phase": self.phase, "note": self.note,
                "playbook": PLAYBOOK if self.active else "weekday: breakout continuation can be trusted when it qualifies",
                "ends_utc" if self.active else "starts_utc":
                    datetime.fromtimestamp(self.until_ms / 1000, timezone.utc).strftime("%a %H:%M UTC")}


def _next(now: datetime, weekday: int, hour: int) -> datetime:
    """The next occurrence of weekday at hour UTC, strictly after now."""
    d = now.replace(minute=0, second=0, microsecond=0, hour=hour)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    if d <= now:
        d += timedelta(days=7)
    return d


def read(now_ms: int) -> Weekend:
    now = datetime.fromtimestamp(now_ms / 1000, timezone.utc)
    end = _next(now, *END)
    start = _next(now, *START)
    hm = lambda d: f"{d:%a %H:%M} UTC ({(d + timedelta(hours=5)):%H:%M} PKT)"  # noqa: E731
    if end < start:                                    # inside Friday 21:00 - Sunday 22:00
        return Weekend(True, "weekend",
                       f"Weekend tape until {hm(end)}: spot volume runs far below the week's, so moves run further "
                       f"and reverse faster than the levels suggest. Playbook: {PLAYBOOK}.", int(end.timestamp() * 1000))
    hours = (start - now).total_seconds() / 3600
    if hours <= 6:
        return Weekend(False, "starting",
                       f"The weekend tape starts at {hm(start)}, "
                       f"{'in under an hour' if hours < 1 else f'in {hours:.0f} hour' + ('s' if hours >= 1.5 else '')}: "
                       "volume thins from there, so a late-Friday break is suspect.", int(start.timestamp() * 1000))
    return Weekend(False, "weekday", "", int(start.timestamp() * 1000))


def is_weekend(now_ms: int) -> bool:
    return read(now_ms).active
