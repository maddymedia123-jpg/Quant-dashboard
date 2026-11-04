"""Event regime: the market condition around a major scheduled event (an election, an FOMC decision, a
Fed chair press conference), when a range plan is the wrong plan.

His Section 6.2: "The Head Agent queries the calendar for high-impact events (FOMC, Elections). If active
within the trade window, volatility parameters widen and trap warnings elevate." On 4 November 2026 the
calendar carried "Congressional Elections" as a High-impact print and nothing elevated it: the Quant
Forecast said NEUTRAL / RANGE-BOUND at 80% and was invalidated when BTC ran 78,336 to 81,194 the afternoon
after the vote (election-day fix 3, owner's GO ALL, 5 November).

A major event is a High-impact USD calendar entry whose title carries one of MAJOR_WORDS. The regime is
active from AHEAD_MS before the print (the positioning into it) to AFTER_MS after it (the reaction: an
election's result lands overnight and is traded the day after). CPI, NFP and the other prints stay what
they were, a chop window around the release (core.qf_preflight), not a regime.

While the regime is active:
* the Quant Forecast builds no range plan: the lean is scored 0, every hour's sigma is widened, the
  invalidation sits at the 5th percentile on both sides and the directive names the event
  (core.quant_forecast.build_plan, `event=`);
* the Pulse's event section leads with it (core.pulse);
* every desk summary's warning leads with it and its directive carries it (core.layman_summary);
* the war rooms and the Director see it in the state they judge (`event_regime` in the Live Recon state
  and in the macro read).

`read` accepts the calendar's event dicts or the macro engine's EventRead objects (title, time_ms), so
the summary can read it from the prints it already holds."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

H = 3_600_000
AHEAD_MS = 24 * H                 # the positioning into a major print
AFTER_MS = 48 * H                 # the reaction after it: an election result is traded the next day
MAJOR_WORDS = ("election", "fomc", "federal funds rate", "fed chair")
PKT_MS = 5 * H


@dataclass(frozen=True)
class Regime:
    active: bool
    title: str | None = None
    time_ms: int | None = None
    phase: str | None = None      # "ahead": inside AHEAD_MS before the print; "reaction": inside AFTER_MS after it
    until_ms: int | None = None   # when the regime ends (the print plus AFTER_MS)
    note: str | None = None       # the one plain-words line every message carries

    def to_state(self) -> dict:
        return {"active": False} if not self.active else asdict(self)


NONE = Regime(active=False)


def _field(e, key: str):
    return e.get(key) if isinstance(e, dict) else getattr(e, key, None)


def is_major(e) -> bool:
    """A High-impact USD print whose title names an election or the Fed. An EventRead (no impact field)
    is already High USD, so only its title is read."""
    title = str(_field(e, "title") or "").lower()
    if not any(w in title for w in MAJOR_WORDS):
        return False
    if isinstance(e, dict):
        return e.get("impact") == "High" and e.get("country") in ("USD", None)
    return True


def _utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%H:%M")


def _pkt(ms: int) -> str:
    return datetime.fromtimestamp((ms + PKT_MS) / 1000, timezone.utc).strftime("%H:%M")


def _day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%a %d %b")


def _in_words(ms: int) -> str:
    h = ms / H
    if h < 1:
        return "under an hour"
    if h < 36:
        return f"{h:.0f} hours"
    return f"{h / 24:.0f} days"


def read(events, now_ms: int) -> Regime:
    """The regime now: the major print whose window holds now_ms, and when several do, the one whose
    window ends last (an FOMC statement and its press conference are one regime)."""
    best = None
    for e in events or ():
        t = _field(e, "time_ms")
        if t is None or not is_major(e):
            continue
        t = int(t)
        if not (t - AHEAD_MS <= now_ms < t + AFTER_MS):
            continue
        if best is None or t > best[0]:
            best = (t, str(_field(e, "title") or "major event").strip())
    if best is None:
        return NONE
    t, title = best
    until = t + AFTER_MS
    if now_ms < t:
        note = (f"{title} prints at {_utc(t)} UTC ({_pkt(t)} PKT) {_day(t)}, {_in_words(t - now_ms)} away; "
                f"the event regime runs until {_utc(until)} UTC {_day(until)}")
        return Regime(True, title, t, "ahead", until, note)
    note = (f"{title} printed {_in_words(now_ms - t)} ago ({_utc(t)} UTC {_day(t)}); the reaction is being "
            f"traded until {_utc(until)} UTC ({_pkt(until)} PKT) {_day(until)}")
    return Regime(True, title, t, "reaction", until, note)
