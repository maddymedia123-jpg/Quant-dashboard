"""The Live Recon 4-hour anchored summary and its interim side notes.

The client's rule, in his words: the anchored summary aligns with the UTC 4-hour candle close, stays
pinned and immutable for that window, and a re-run mid-window must never reload or replace it. When an
interim 15m/1h scan finds something substantial - the bias flipped, a trap appeared, the case moved
hard, a high-impact release landed - the head agent appends a side note alongside the summary saying
what changed and what it does to the previous read.

So: publishing is insert-once per window (enforced in the store), and everything else in the window is
append-only. A side note carries a dedup key, because an interim scan that keeps seeing the same flip
should not keep announcing it."""
from __future__ import annotations

from dataclasses import dataclass

from core.agents.live_recon import DECISIVE_MARGIN, TRAP_OVERRIDE_P, LiveReconResult

WINDOW_MS = 4 * 60 * 60 * 1000          # the 4h candle, aligned to UTC
SWING_POINTS = 15.0                      # team-score move that counts as substantial
SWING_BUCKET = 10.0                      # so a drifting score does not file a note per point

BIAS_FLIP, TRAP_ALERT, SCORE_SWING, MACRO_ALERT, DEGRADED = (
    "BIAS_FLIP", "TRAP", "SCORE_SWING", "MACRO", "DEGRADED")


@dataclass(frozen=True)
class SideNote:
    kind: str
    dedup_key: str
    headline: str
    detail: str = ""


def window_open(now_ms: int) -> int:
    """Start of the UTC 4h candle containing now. Epoch ms are UTC, so this is just floor division."""
    return now_ms - now_ms % WINDOW_MS


def window_close(now_ms: int) -> int:
    return window_open(now_ms) + WINDOW_MS


def direction(bias: str) -> str:
    """The side a bias takes, ignoring the trap qualifier: BULL, BEAR or BALANCED."""
    if bias.startswith("BULL"):
        return "BULL"
    if bias.startswith("BEAR"):
        return "BEAR"
    return "BALANCED"


def anchor_payload(res: LiveReconResult, price: float | None = None) -> dict:
    """What gets pinned. Enough to rebuild the summary and to compare an interim run against."""
    return {
        "bias": res.bias,
        "direction": direction(res.bias),
        "margin": round(res.margin, 2),
        "confidence": round(res.confidence, 3),
        "bull_points": round(res.bull.points, 2),
        "bear_points": round(res.bear.points, 2),
        "bull_verdict": res.bull.verdict,
        "bear_verdict": res.bear.verdict,
        "trap": res.trap.choice if res.trap.ok else None,
        "trap_probabilities": {k: round(v, 3) for k, v in res.trap.probabilities.items()},
        "consulted": res.consulted,
        "override": res.override,
        "notes": list(res.notes),
        "degraded": res.degraded,
        "price": price,
        "domains": {
            side.side: {d.key: round(d.points, 2) for d in side.domains}
            for side in (res.bull, res.bear)
        },
        "published_at_utc": res.generated_at.strftime("%Y-%m-%d %H:%M UTC"),
    }


def detect_side_notes(anchor: dict, res: LiveReconResult, macro_event: str | None = None) -> list[SideNote]:
    """What an interim scan found that the pinned summary does not say."""
    notes: list[SideNote] = []
    old_dir, new_dir = anchor.get("direction", "BALANCED"), direction(res.bias)

    if new_dir != old_dir:
        notes.append(SideNote(
            BIAS_FLIP, f"bias:{old_dir}->{new_dir}",
            f"Bias has changed from {old_dir} to {new_dir} since the anchored summary.",
            f"Bullish team {res.bull.points:.0f}/100, bearish team {res.bear.points:.0f}/100 "
            f"(anchored at {anchor.get('bull_points', 0):.0f} and {anchor.get('bear_points', 0):.0f}). "
            "The anchored summary stands as the structural read for this candle; treat this as the live "
            "invalidation."))

    old_trap, new_trap = anchor.get("trap"), (res.trap.choice if res.trap.ok else None)
    trap_p = max(res.trap.probabilities.get("BULL_TRAP", 0.0), res.trap.probabilities.get("BEAR_TRAP", 0.0))
    if new_trap in ("BULL_TRAP", "BEAR_TRAP") and new_trap != old_trap and trap_p >= TRAP_OVERRIDE_P:
        notes.append(SideNote(
            TRAP_ALERT, f"trap:{new_trap}",
            f"War room now reads a {new_trap.replace('_', ' ').lower()} at {trap_p:.0%}.",
            f"The anchored summary was published with {old_trap or 'no trap call'}."))

    swing = res.margin - float(anchor.get("margin", 0.0))
    if abs(swing) >= SWING_POINTS and new_dir == old_dir:
        bucket = int(abs(swing) // SWING_BUCKET) * int(SWING_BUCKET)
        notes.append(SideNote(
            SCORE_SWING, f"swing:{'+' if swing > 0 else '-'}{bucket}",
            f"The case has moved {abs(swing):.0f} points {'towards bulls' if swing > 0 else 'towards bears'} "
            "without flipping the bias.",
            f"Margin {anchor.get('margin', 0.0):+.0f} at the anchor, {res.margin:+.0f} now; "
            f"{DECISIVE_MARGIN:.0f} points is the threshold for calling a direction."))

    if macro_event:
        notes.append(SideNote(
            MACRO_ALERT, f"macro:{macro_event}",
            f"High-impact event inside this candle: {macro_event}.",
            "Positioning taken on the anchored summary should account for the release."))

    if res.degraded and not anchor.get("degraded"):
        notes.append(SideNote(
            DEGRADED, "degraded",
            "This interim scan ran degraded: part of the checklist did not score.",
            "; ".join(res.notes) or "No detail returned."))

    return notes


@dataclass
class PublishResult:
    window_open_ms: int
    window_close_ms: int
    anchored: bool                       # True when this run published the anchor for the window
    anchor: dict                         # the pinned payload now in force
    appended: tuple[SideNote, ...] = ()  # side notes written by this run
    skipped: tuple[SideNote, ...] = ()   # detected but already noted in this window

    @property
    def ms_to_close(self) -> int:
        return self.window_close_ms - self.window_open_ms


def publish(store, res: LiveReconResult, now_ms: int, price: float | None = None,
            macro_event: str | None = None) -> PublishResult:
    """Anchor the summary if this window has none; otherwise append what changed.

    A run never rewrites a pinned summary, so an interim re-run cannot lose the structural context the
    client trades from - the new read arrives as a note beside it."""
    w_open, w_close = window_open(now_ms), window_close(now_ms)
    payload = anchor_payload(res, price)
    existing = store.get_live_anchor(w_open)

    if existing is None:
        published = store.put_live_anchor(w_open, w_close, now_ms, res.bias, res.margin, res.confidence,
                                          res.trap.choice if res.trap.ok else None, price, payload)
        if published:
            return PublishResult(w_open, w_close, True, payload)
        existing = store.get_live_anchor(w_open)   # a concurrent run won the race; fall through to notes

    anchor = existing["payload"]
    appended, skipped = [], []
    for note in detect_side_notes(anchor, res, macro_event):
        wrote = store.add_side_note(w_open, now_ms, note.kind, note.dedup_key, note.headline, note.detail, price)
        (appended if wrote else skipped).append(note)
    return PublishResult(w_open, w_close, False, anchor, tuple(appended), tuple(skipped))
