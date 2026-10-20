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
# Plain names for the timeframes, so an alert reads "daily" rather than "1d"
TF_WORDS = {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "daily", "1w": "weekly", "2w": "2-week", "1M": "monthly",
            "3M": "quarterly"}
DESK_TF = {"live": "1h", "intraday": "4h", "weekly": "1d", "monthly": "1w"}
DESK_LABEL = {"live": "Live Recon", "intraday": "Intraday", "weekly": "Weekly", "monthly": "Monthly"}


def _tf(tf: str | None) -> str:
    return TF_WORDS.get(tf or "", tf or "")


def state_name(state: str, trend: str | None) -> str:
    """The cascade state in a trader's words: A strong setup, B a pause, C a trend losing control, N no trend."""
    up = trend == "bullish"
    if state == "A" and trend in ("bullish", "bearish"):
        return f"Strong setup {'up' if up else 'down'}"
    if state == "B" and trend in ("bullish", "bearish"):
        return f"Pause in {'an uptrend' if up else 'a downtrend'}"
    if state == "C" and trend in ("bullish", "bearish"):
        return f"{'Uptrend' if up else 'Downtrend'} losing control"
    return {"A": "Strong setup", "B": "Pause in the trend", "C": "Trend losing control",
            "N": "No bigger trend", "?": "Not enough data"}.get(state, state)


def explain(previous: str, cascade) -> str:
    """One sentence on what changed and what it means, built from the cascade's own reads."""
    bias, anchor = cascade.bias, cascade.anchor
    big = _tf(bias.timeframe if bias else None)
    engines = "/".join(reversed([_tf(r.timeframe) for r in cascade.engine])) or "shorter timeframes"
    trend = bias.direction if bias else None
    word = "up" if trend == "bullish" else "down"
    side = "upside" if trend == "bullish" else "downside"
    state = cascade.state
    if state == "N":
        return (f"The {big} has no clear trend, so there is no bigger direction to follow. "
                f"Trade the edges of the {_tf(anchor.timeframe if anchor else None)} range, not breakouts.")
    if state == "A":
        return f"{big.capitalize()} trend is {word} and the {engines} now agree. Best conditions for {side}."
    if state == "C":
        return (f"The {engines} turned against the {big} {word}trend and are holding there. "
                "Watch for a reversal.")
    # B: the trend stands; the shorter timeframes are mixed or leaning against it without having taken over
    if previous == "A":
        return f"{big.capitalize()} trend is still {word}, but the {engines} lost their push. A breather, not a reversal."
    if previous == "C":
        return f"{big.capitalize()} trend is {word} and the {engines} stopped fighting it. Less danger than before."
    if previous == "N":
        return f"The {big} picked a direction again: {word}. The {engines} are mixed, so this is a pause inside it."
    return f"{big.capitalize()} trend is {word}; the {engines} are mixed. A pause inside the trend."


def cascade_event(desk: str, label: str, previous: str | None, cascade, candle_ms: int,
                  price: float | None = None) -> Event | None:
    """The alert for a desk whose cascade state differs from the last one recorded, or None.

    Three lines: the desk and price; the old state and the new one in plain words; what changed."""
    state = cascade.state
    if previous is None or previous == state or state in SILENT_STATES or previous in SILENT_STATES:
        return None
    trend = cascade.bias.direction if cascade.bias is not None else None
    tf = DESK_TF.get(desk)
    head = (f"{label}{f' ({tf})' if tf else ''}" + (f" · BTC ${price:,.0f}" if price else "") + "\n"
            # the ledger keeps only the letter, so the old state is named with the trend as it reads now
            f"{state_name(previous, trend)} -> {state_name(state, trend)}\n"
            f"{explain(previous, cascade)}")
    return Event("shift" if state == "C" else "cascade", f"cascade:{desk}:{previous}->{state}:{candle_ms}",
                 head, price)


def trap_events(traps: list[dict], now_ms: int) -> list[Event]:
    """A declared trap, and a settled one, each announced once - only those from the last 24 hours."""
    out: list[Event] = []
    for t in traps:
        bull = str(t.get("side", "")).upper().startswith("BULL")
        level, target = float(t.get("level") or 0), float(t.get("plays_out") or 0)
        invalid = float(t.get("invalidation") or 0)
        desk = DESK_LABEL.get(str(t.get("category") or ""), str(t.get("category") or "").capitalize())
        tf = str(t.get("timeframe") or "")
        where = f"{desk} ({tf})" if tf else desk
        move = f"pop above ${level:,.0f}" if bull else f"dip below ${level:,.0f}"
        declared = int(t.get("declared_ms") or 0)
        if now_ms - declared <= ANNOUNCE_WINDOW_MS:
            out.append(Event("trap", f"trap:{t['id']}:declared",
                             f"TRAP · {where}\n"
                             f"The {move} should fail and {'drop' if bull else 'bounce'}.\n"
                             f"Target ${target:,.0f} · Wrong if price goes past ${invalid:,.0f}.", t.get("level")))
        status, resolved = str(t.get("status") or ""), t.get("resolved_ms")
        if status in ("played_out", "invalidated") and resolved and now_ms - int(resolved) <= ANNOUNCE_WINDOW_MS:
            at = float(t.get("resolved_price") or 0)
            if status == "played_out":
                text = f"TRAP worked · {where}\nThe {move} failed as called: price reached ${at:,.0f}."
            else:
                text = (f"TRAP failed · {where}\nThe {move} was a real move, not a trap: price went past "
                        f"${invalid:,.0f} (now ${at:,.0f}).")
            out.append(Event("trap_settled", f"trap:{t['id']}:{status}", text, t.get("resolved_price")))
    return out


# ---------- the cascade alerts that still go out on their own ----------
# Since the plan alerts (core.plan_alerts) took over, a desk's state change is a message of its own only
# when it matters with no report run: a desk lining up strongly (A) or its trend losing control (C).
# "Pause" moves stay quiet. And the new state must hold for HOLD_MS first: read from candles still
# forming, a state can flip and flip back within minutes (Intraday went A at 02:13 and back at 02:17 on
# 20 Oct), and a flicker is not news.
IMPORTANT_STATES = ("A", "C")
HOLD_MS = {"live": 5 * 60_000, "intraday": 15 * 60_000, "weekly": 60 * 60_000, "monthly": 4 * 60 * 60_000}


def important_cascade_event(desk: str, label: str, track: dict, cascade, now_ms: int,
                            price: float | None = None) -> Event | None:
    """The alert for a desk that moved into A or C and has held it, once per episode, or None."""
    state, prior = track.get("state"), track.get("prior_state")
    if track.get("first") or state not in IMPORTANT_STATES or prior is None or prior == state:
        return None
    since = int(track.get("since_ms") or now_ms)
    if now_ms - since < HOLD_MS.get(desk, 15 * 60_000):
        return None
    # keyed on when the state began: the same episode is one message, a new one later is another
    return cascade_event(desk, label, prior, cascade, since, price)
