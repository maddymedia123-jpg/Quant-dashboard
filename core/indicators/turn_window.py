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

The spec's three "institutional kill zones" (v15 s.3), measured 23 Oct 2026 (research/kill_zones.py)
--------------------------------------------------------------------------------------------------------
Asian 00:00-04:00, London 07:00-10:00 and New York 13:00-17:00 UTC, scored the same way: a turn is the
high or low extreme within +-6 h with a 3 ATR(14) move completed inside the next 6 h (big: +-12 h, 5 ATR,
inside 12 h); "times chance" is the zone's share of turns divided by its share of the day (hours / 24).
Main sample: 9,999 hourly Gate.io BTC_USDT perp candles, 2 Sep 2025 - 23 Oct 2026 (Gate serves no older
hourly points), 353 turns / 136 big. Second, independent sample: Binance BTCUSDT perp, 49,680 hours,
Jan 2020 - Aug 2025, 1,673 turns / 586 big.

    zone             hours   Gate turns    x chance  big   |  Binance 2020-25  x chance  big
    Asian 00-04        4     56  15.9%      0.95    0.62   |  190  11.4%       0.68    0.69
    London 07-10       3     42  11.9%      0.95    1.00   |  203  12.1%       0.97    0.86
    New York 13-17     4    106  30.0%      1.80    1.59   |  486  29.0%       1.74    1.79
    (ours 12-16)       4    127  36.0%      2.16    2.03   |  536  32.0%       1.92    2.02

* Only New York holds. Asian is at or BELOW chance in every split (0.53-0.99x on Gate, 0.48-0.80x on
  Binance); London never rises above 1.20x (that one a 40-turn cell). Neither may be told to a reader as
  a turning time.
* New York and our 12-16 window hold in both halves of both samples (Gate 2.04x / 1.58x; Binance
  2020-Oct 2022 1.23x, Nov 2022-2025 2.18x) and across summer and winter time (New York 1.47-2.52x on
  Gate, 1.60-2.02x on Binance; Gate's big turns in UK summer time were the low, 1.19x). Daylight saving moves the
  peak by the hour it should: on Binance in US winter (NYSE opens 14:30 UTC) 13-17 scored 2.02x against
  1.98x for 12-16, in US summer (13:30 UTC) 12-16 led, 1.90x against 1.60x. On Gate 12-16 led in both.
* Weakest cell: Gate's second half, big turns (68 of them): New York 1.32x, ours 1.41x - still above
  chance, not by much. The follow-through window is the one free choice; at two and four times it
  (12 / 24 h for normal turns, 24 / 48 h for big) New York stayed 1.45-1.75x and Asian / London stayed
  0.62-1.08x on both samples.
* Turns cluster (a top and the next bottom are hours apart), so the samples are not fully independent;
  the cross-venue, cross-year repeat is the evidence, not any single ratio.
* Rebuilt from this docstring the 12-16 window gives 36.0% (the first run said 35%); the constants below
  keep the first run's numbers, which the rebuild confirms.

The spec's "sine wave oscillation" was tested the same day (research/sine_wave.py) and is NOT in the
product: Ehlers' Sine Wave (Hilbert dominant cycle, LeadSine crossing Sine), computed with no lookahead,
put a turn within +-2 hours of 18.1% of its 646 signals on Gate against 17.2% for random hours (1.05x;
second half 1.04x; Binance 16.9% vs 16.2% on 3,246 signals). Its better score at calling the DIRECTION of
the nearby turn was matched by random hours labelled from the last 6 hours' move.

A zone at or below 1.2x (NEAR_CHANCE_X) is flagged `near_chance`: it is shown so the spec's claim is
answered, never as a reason to expect a turn.
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
PKT_S = 5 * 3600                 # Pakistan time, UTC+5 all year (no daylight saving)

NEAR_CHANCE_X = 1.2              # at or below this a zone is no better than any other hours of the day
ZONES_MEASURED = ("9,999 hourly Gate.io BTC perp candles, Sep 2025 - Oct 2026 (353 turns); checked on "
                  "49,680 Binance BTC perp hours, 2020 - Aug 2025 (1,673 turns)")


@dataclass(frozen=True)
class Zone:
    """One of the spec's kill zones with what it scored (research/kill_zones.py, 23 Oct 2026)."""
    key: str
    name: str
    start_hour_utc: int
    end_hour_utc: int
    turns: int                   # Gate turns inside it, of GATE_TURNS
    share_of_turns: float        # Gate, normal turns (+-6 h, 3 ATR)
    times_chance: float          # share / (hours / 24), Gate normal turns
    big_times_chance: float      # Gate, big turns (+-12 h, 5 ATR)
    older_times_chance: float    # Binance 2020 - Aug 2025, normal turns

    @property
    def hours(self) -> int:
        return self.end_hour_utc - self.start_hour_utc

    @property
    def near_chance(self) -> bool:
        return self.times_chance <= NEAR_CHANCE_X

    @property
    def hours_label(self) -> str:
        """'13:00–17:00 UTC (18:00–22:00 PKT)'"""
        s, e = self.start_hour_utc, self.end_hour_utc
        return f"{s:02d}:00–{e:02d}:00 UTC ({(s + 5) % 24:02d}:00–{(e + 5) % 24:02d}:00 PKT)"

    @property
    def reading(self) -> str:
        """Plain English, one line, the measured odds in it."""
        if self.near_chance:
            return (f"{self.name} {self.hours_label}: {self.share_of_turns:.0%} of BTC's turns, about what any "
                    f"{self.hours} hours of the day get ({self.times_chance:.2f}x chance) - no edge, "
                    f"not a turning time")
        return (f"{self.name} {self.hours_label}: {self.share_of_turns:.0%} of BTC's turns in {self.hours} hours, "
                f"{self.times_chance:.1f}x what chance would put there")


GATE_TURNS = 353
ZONES: tuple[Zone, ...] = (
    Zone("asian", "Asian session", 0, 4, 56, 0.159, 0.95, 0.62, 0.68),
    Zone("london", "London open", 7, 10, 42, 0.119, 0.95, 1.00, 0.97),
    Zone("new_york", "New York session", 13, 17, 106, 0.300, 1.80, 1.59, 1.74),
)


def zones() -> tuple[Zone, ...]:
    return ZONES


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
        ps = datetime.fromtimestamp(self.start_ms / 1000 + PKT_S, timezone.utc)
        pe = datetime.fromtimestamp(self.end_ms / 1000 + PKT_S, timezone.utc)
        return f"{s:%H:%M}–{e:%H:%M} UTC ({ps:%H:%M}–{pe:%H:%M} PKT - {ps:%a %d %b})"

    @property
    def ends(self) -> str:
        """'16:00 UTC (21:00 PKT)'"""
        e = datetime.fromtimestamp(self.end_ms / 1000, timezone.utc)
        pe = datetime.fromtimestamp(self.end_ms / 1000 + PKT_S, timezone.utc)
        return f"{e:%H:%M} UTC ({pe:%H:%M} PKT)"

    @property
    def odds(self) -> str:
        return (f"about {SHARE_OF_TURNS:.0%} of BTC's turns happen in these four hours "
                f"(measured on {MEASURED})")


@dataclass(frozen=True)
class ZoneWindow(TurnWindow):
    """The next sitting of one kill zone: its times, whether it is on now, and what it measured."""
    zone: Zone = ZONES[0]

    @property
    def odds(self) -> str:
        return self.zone.reading


def _next(now_ms: int, start_hour: int, end_hour: int) -> tuple[int, int, bool]:
    """Today's sitting if it has not ended yet, otherwise tomorrow's."""
    day = now_ms - now_ms % DAY_MS
    start, end = day + start_hour * HOUR_MS, day + end_hour * HOUR_MS
    if now_ms >= end:
        start, end = start + DAY_MS, end + DAY_MS
    return start, end, start <= now_ms < end


def next_window(now_ms: int) -> TurnWindow:
    """Today's window if it has not ended yet, otherwise tomorrow's."""
    return TurnWindow(*_next(now_ms, START_HOUR_UTC, END_HOUR_UTC))


def next_zones(now_ms: int) -> list[ZoneWindow]:
    """Each kill zone's next sitting (or the one on now), soonest first."""
    out = [ZoneWindow(*_next(now_ms, z.start_hour_utc, z.end_hour_utc), zone=z) for z in ZONES]
    return sorted(out, key=lambda w: w.start_ms)


def to_state(now_ms: int) -> dict:
    """For the agents: the window, how it was measured, that Fibonacci timing was tested and failed, and the
    spec's three kill zones with their measured odds (only New York beat chance)."""
    w = next_window(now_ms)
    return {"next_window": w.label, "active_now": w.active, "share_of_turns": SHARE_OF_TURNS,
            "times_chance_per_hour": list(TIMES_CHANCE), "measured_on": MEASURED,
            "note": "A base rate for WHEN turns happen, not a direction. Fibonacci time zones were tested on "
                    "the same data and showed no edge (17.3% vs 18.7% random), so do not use them to time "
                    "turns; 09:00 UTC showed no edge either (1.09x).",
            "kill_zones": [{"zone": z.zone.name, "hours": z.zone.hours_label, "next": z.label,
                            "active_now": z.active, "share_of_turns": z.zone.share_of_turns,
                            "times_chance": z.zone.times_chance, "big_turns_times_chance": z.zone.big_times_chance,
                            "older_sample_times_chance": z.zone.older_times_chance,
                            "near_chance": z.zone.near_chance, "reading": z.zone.reading}
                           for z in next_zones(now_ms)],
            "kill_zones_measured_on": ZONES_MEASURED,
            "kill_zones_note": "The spec's three kill zones, measured: only New York (13:00-17:00 UTC) beats "
                               "chance, and it overlaps the window above. Asian and London came out at or "
                               "below chance (near_chance true): never give them as a time BTC turns."}
