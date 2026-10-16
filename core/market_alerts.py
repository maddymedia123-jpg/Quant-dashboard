"""Market alerts: a desk's timeframe cascade changing state, and TRAP calls being declared or settled.

His ask: push the A -> B -> C transitions and the TRAP warnings to Discord. Detection is pure and every
event carries a dedup key, so the store's unique index makes each change one message however often the
page refreshes - the same pattern as the trade alerts in core.alerts, and through the same Dispatcher.

Cascade changes are found by comparing each desk's state with the last state recorded for it. The first
time a desk is seen there is nothing to compare with, so that reading is stored as the baseline and not
announced: a fresh deploy (Streamlit Cloud wipes the ledger on reboot) must not open with four "changes"
that are not changes. A change into or out of "?" (too few candles) is a data gap, not a market event,
and is recorded silently.

TRAP calls are read from the trap ledger: one alert when a trap is declared and one when price settles it
(played out, or invalidated - the break was real). Only the last 24 hours are announced, so an old ledger
is not replayed as news.

Alerts go out while a page is open; there is no background worker on Streamlit Cloud. A scheduled job
(the "Head Bot") would close that gap."""
from __future__ import annotations

from core.alerts import Event

DAY_MS = 86_400_000
ANNOUNCE_WINDOW_MS = DAY_MS
SILENT_STATES = ("?",)
STATE_WORDS = {"A": "A · full confluence", "B": "B · pullback", "C": "C · control shift",
               "N": "N · no higher-timeframe bias", "?": "? · not enough data"}


def cascade_event(desk: str, label: str, previous: str | None, cascade, candle_ms: int,
                  price: float | None = None) -> Event | None:
    """The alert for a desk whose cascade state differs from the last one recorded, or None."""
    state = cascade.state
    if previous is None or previous == state or state in SILENT_STATES or previous in SILENT_STATES:
        return None
    lean = ""
    if cascade.bias is not None and cascade.bias.direction in ("bullish", "bearish"):
        lean = f" ({cascade.bias.timeframe} bias {cascade.bias.direction})"
    play = cascade.playbook.split(". ")[0].rstrip(".") + "."
    head = (f"{label}: cascade {STATE_WORDS.get(previous, previous)} -> {STATE_WORDS.get(state, state)}{lean}. "
            f"{play}" + (f" BTC ${price:,.0f}." if price else ""))
    return Event("shift" if state == "C" else "cascade", f"cascade:{desk}:{previous}->{state}:{candle_ms}",
                 head, price)


def trap_events(traps: list[dict], now_ms: int) -> list[Event]:
    """A declared trap, and a settled one, each announced once - only those from the last 24 hours."""
    out: list[Event] = []
    for t in traps:
        side = "bull trap" if str(t.get("side", "")).upper().startswith("BULL") else "bear trap"
        where = f"{t.get('timeframe', '')} {side} at ${float(t.get('level') or 0):,.0f}"
        declared = int(t.get("declared_ms") or 0)
        if now_ms - declared <= ANNOUNCE_WINDOW_MS:
            out.append(Event("trap", f"trap:{t['id']}:declared",
                             f"TRAP declared: {where} ({t.get('category', '')} desk). Plays out at "
                             f"${float(t.get('plays_out') or 0):,.0f}; invalid past "
                             f"${float(t.get('invalidation') or 0):,.0f}.", t.get("level")))
        status, resolved = str(t.get("status") or ""), t.get("resolved_ms")
        if status in ("played_out", "invalidated") and resolved and now_ms - int(resolved) <= ANNOUNCE_WINDOW_MS:
            what = "played out" if status == "played_out" else "invalidated: the break was real"
            out.append(Event("trap_settled", f"trap:{t['id']}:{status}",
                             f"TRAP {what}: {where} at ${float(t.get('resolved_price') or 0):,.0f}.",
                             t.get("resolved_price")))
    return out
