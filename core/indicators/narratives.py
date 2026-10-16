"""Narrative shelf-life: which stories the headline feed is talking about, and whether that is rising or dying.

A narrative is a headline topic (the keyword tags in core.data.news; untagged "general" headlines are not
one). For each, the attention in the last 6 hours is compared with its rate across the rest of the feed:

* fresh    - first seen in the last 12 hours. Not yet priced; reactions to it are largest now.
* building - headline rate in the last 6 hours at least 1.5x the earlier rate. Attention is still arriving.
* running  - steady attention for longer than 12 hours. Established, and probably priced in.
* fading   - nothing new for 12 hours, or still covered but at half the earlier rate. The trade built on it is
             the one at risk.

This is an attention tracker, not a forecast. The feed only reaches back as far as its publishers do
(about a day and a half), so an "age" equal to the feed's own span is a floor: the story may be older."""
from __future__ import annotations

from dataclasses import dataclass

H = 3_600_000
RECENT_H = 6
FRESH_H = 12
QUIET_H = 12
MIN_HEADLINES = 2


@dataclass(frozen=True)
class Narrative:
    topic: str
    state: str
    headlines: int
    recent: int
    age_h: float
    last_h: float
    age_is_floor: bool
    latest_title: str

    @property
    def shelf_life(self) -> str:
        age = f"{self.age_h:.0f}h{'+' if self.age_is_floor else ''}"
        return {"fresh": f"new ({age}): not yet priced, reactions are largest now",
                "building": f"{age} old and gaining attention",
                "running": f"established for {age}: probably priced in",
                "fading": f"{age} old and losing attention: positions built on it are the ones at risk"}[self.state]

    def to_state(self) -> dict:
        return {"topic": self.topic, "state": self.state, "shelf_life": self.shelf_life,
                "headlines": self.headlines, "last_6h": self.recent, "age_hours": round(self.age_h, 1),
                "age_is_a_floor": self.age_is_floor, "hours_since_last": round(self.last_h, 1),
                "latest": self.latest_title}


def read(headlines: list[dict], now_ms: int) -> list[Narrative]:
    """Topics with at least two headlines, most recently active first."""
    dated = [h for h in headlines if h.get("published_ms") and h["published_ms"] <= now_ms + H]
    if not dated:
        return []
    span_h = max((now_ms - min(h["published_ms"] for h in dated)) / H, RECENT_H + 1)
    by_topic: dict[str, list[dict]] = {}
    for h in dated:
        if h.get("topic") and h["topic"] != "general":
            by_topic.setdefault(h["topic"], []).append(h)

    out = []
    for topic, hs in by_topic.items():
        if len(hs) < MIN_HEADLINES:
            continue
        hs = sorted(hs, key=lambda h: h["published_ms"])
        age = (now_ms - hs[0]["published_ms"]) / H
        last = max(0.0, (now_ms - hs[-1]["published_ms"]) / H)
        recent = sum(1 for h in hs if now_ms - h["published_ms"] <= RECENT_H * H)
        rate_recent = recent / RECENT_H
        rate_prior = (len(hs) - recent) / max(span_h - RECENT_H, 1.0)
        # a gap of a few hours is normal for a sparse feed: quiet means twelve hours with nothing, and a
        # halved rate only counts while the story is still being written about at all
        if last >= QUIET_H or (recent and rate_prior > 0 and rate_recent <= 0.5 * rate_prior):
            state = "fading"
        elif age <= FRESH_H:
            state = "fresh"
        elif rate_recent >= 1.5 * rate_prior:
            state = "building"
        else:
            state = "running"
        out.append(Narrative(topic, state, len(hs), recent, age, last,
                             age_is_floor=age >= span_h - 1.0, latest_title=str(hs[-1].get("title") or "")))
    return sorted(out, key=lambda n: n.last_h)
