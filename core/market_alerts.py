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
from core.traps import SHORT_HORIZON_LABEL, is_short_horizon

DAY_MS = 86_400_000
ANNOUNCE_WINDOW_MS = DAY_MS
SILENT_STATES = ("?",)
# Plain names for the timeframes, so an alert reads "daily" rather than "1d"
TF_WORDS = {"15m": "15m", "1h": "1h", "4h": "4h", "12h": "12h", "1d": "daily", "3d": "3-day", "0.5w": "half-week",
            "1w": "weekly", "2w": "2-week", "1M": "monthly", "3M": "quarterly"}
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
        # election-day fix 5: the level, the count and what confirms it, not "watch for a reversal"
        # the anchored VWAP in plain words: the average price since the bias candle opened
        beyond = "above" if trend == "bearish" else "below"
        level = f"${cascade.vwap:,.0f}" if cascade.vwap else f"the {big} average price"
        first = f"the {big} average price {level}" if cascade.vwap else level
        n = cascade.closes_beyond_vwap or 0
        closes = f"{n} times" if n != 1 else "once"
        at = f"the {_tf(anchor.timeframe)}" if anchor else "the anchor candle"
        return (f"The {engines} have closed {closes} {beyond} {first} against the {big} {word}trend. "
                f"Confirmation is {at} closing {beyond} {level}; until then this is a warning, not a trade.")
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
        if tf and is_short_horizon(tf):
            tf = f"{tf}, {SHORT_HORIZON_LABEL}"
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
# A desk that leaves A or C and comes back within this long is in the same episode, not a new one: on
# 29 Oct the Live desk went C at 16:00, 16:17 and 18:55, flapping through B and N between, and each return
# outlasted the 5-minute hold - three identical "Downtrend losing control" messages in three hours.
# 4 Nov: the Live desk's "Downtrend losing control" went out at 07:28, 10:29, 14:37 and 19:12 UTC, each just
# outside the 3-hour window. Eight hours now (fix 5), and a confirmed reversal counts as the episode too.
REPEAT_MS = {"live": 8 * 3_600_000, "intraday": 12 * 3_600_000, "weekly": 24 * 3_600_000, "monthly": 72 * 3_600_000}
CONFIRMED = "confirmed"               # the dedup state word of a REVERSAL CONFIRMED alert


def same_episode(recent: list[dict] | None, desk: str, state: str, now_ms: int) -> bool:
    """Whether this desk already alerted into `state` within REPEAT_MS. `recent`: market_alerts rows (ts_ms,
    dedup_key) whose key starts with 'cascade:{desk}:'."""
    limit = REPEAT_MS.get(desk, 6 * 3_600_000)
    return any(str(r.get("dedup_key", "")).startswith(f"cascade:{desk}:") and f"->{state}:" in str(r.get("dedup_key", ""))
               and 0 <= now_ms - int(r.get("ts_ms") or 0) < limit for r in recent or [])


def important_cascade_event(desk: str, label: str, track: dict, cascade, now_ms: int,
                            price: float | None = None, recent: list[dict] | None = None) -> Event | None:
    """The alert for a desk that moved into A or C and has held it, once per episode, or None. `recent`
    (the desk's past cascade alerts) makes a return to the same state within REPEAT_MS part of the last
    episode: silent."""
    state, prior = track.get("state"), track.get("prior_state")
    if track.get("first") or state not in IMPORTANT_STATES or prior is None or prior == state:
        return None
    since = int(track.get("since_ms") or now_ms)
    if now_ms - since < HOLD_MS.get(desk, 15 * 60_000):
        return None
    if same_episode(recent, desk, state, now_ms):
        return None
    if state == "C" and same_episode(recent, desk, CONFIRMED, now_ms):
        return None                            # the reversal was already confirmed: no warning after the fact
    # keyed on when the state began: the same episode is one message, a new one later is another
    return cascade_event(desk, label, prior, cascade, since, price)


def confirming_close(cascade, frames: dict, since_ms: int, now_ms: int) -> dict | None:
    """The first COMPLETED anchor candle that closed after the control shift began and closed beyond the
    anchored VWAP against the bias trend: {"t_ms", "close_ms", "close"}, or None. The candle still forming
    never counts: a close is a close."""
    from core.config import TF_MINUTES
    bias, anchor, vwap = cascade.bias, cascade.anchor, cascade.vwap
    if bias is None or anchor is None or vwap is None or bias.direction not in ("bullish", "bearish"):
        return None
    df = (frames or {}).get(anchor.timeframe)
    bar = TF_MINUTES.get(anchor.timeframe, 0) * 60_000
    if df is None or not len(df) or not bar:
        return None
    for t, close in zip(df["timestamp"].astype("int64"), df["close"].astype(float)):
        end = int(t) + bar
        if end <= int(since_ms) or end > int(now_ms):
            continue                           # closed before the shift began, or still forming
        if (bias.direction == "bearish" and close > vwap) or (bias.direction == "bullish" and close < vwap):
            return {"t_ms": int(t), "close_ms": end, "close": float(close)}
    return None


def reversal_event(desk: str, label: str, track: dict, cascade, frames: dict, now_ms: int,
                   price: float | None = None, recent: list[dict] | None = None) -> Event | None:
    """REVERSAL CONFIRMED (election-day fix 5): a desk in C whose anchor candle has closed beyond the anchored
    VWAP against the bias trend, once per episode. 4 Nov replayed: the 04:00-08:00 4h candle closed 78,140
    above the day's VWAP against the daily downtrend; this goes out at 08:00, six hours before the spike."""
    if track.get("state") != "C" or cascade.state != "C":
        return None
    since = int(track.get("since_ms") or now_ms)
    hit = confirming_close(cascade, frames, since, now_ms)
    if hit is None or same_episode(recent, desk, CONFIRMED, now_ms):
        return None
    trend = cascade.bias.direction
    big, anchor_tf = _tf(cascade.bias.timeframe), _tf(cascade.anchor.timeframe)
    beyond, word = ("above", "down") if trend == "bearish" else ("below", "up")
    off, side = ("Shorts", "upside") if trend == "bearish" else ("Longs", "downside")
    back = "below" if trend == "bearish" else "above"
    from core.qf_alerts import at_label
    tf = DESK_TF.get(desk)
    head = (f"{label}{f' ({tf})' if tf else ''}" + (f" · BTC ${price:,.0f}" if price else "") + "\n"
            f"REVERSAL CONFIRMED: the {anchor_tf} closed {beyond} the {big} average price ${cascade.vwap:,.0f} at ${hit['close']:,.0f} "
            f"({at_label(hit['close_ms'])}) against the {big} {word}trend.\n"
            f"{off} are off: the {big} {word}trend has lost control. Expect {side} expansion. "
            f"Wrong only on a {anchor_tf} close back {back} ${cascade.vwap:,.0f}.")
    return Event("reversal", f"cascade:{desk}:C->{CONFIRMED}:{hit['t_ms']}", head, price)
