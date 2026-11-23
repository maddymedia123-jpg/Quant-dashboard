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
(the "Head Bot") would close that gap.

Wording batch 3 (6 Nov, the owner agreed the rewrite): every message opens with an emoji, what happened,
the desk's horizon in plain words ("next 4 hours", "today", "this week", "this month") and the price;
"📈 UP SETUP", "⚠️ DOWNTREND WEAKENING", "🔄 REVERSAL CONFIRMED", "🪤 TRAP CALL", "✅ TRAP WORKED",
"❌ TRAP FAILED". A setup says "no trade from this alone"; a weakening trend says "not a trade yet" and
names the close that makes it a reversal. No desk names, no state letters, no "old state -> new state"."""
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
# the desk's horizon in plain words: the header says when the message matters, not which desk wrote it
DESK_HORIZON = {"live": "next 4 hours", "intraday": "today", "weekly": "this week", "monthly": "this month"}
CANDLE_WORDS = {"15m": "15-minute", "1h": "1-hour", "4h": "4-hour", "12h": "12-hour", "1d": "daily", "3d": "3-day",
                "0.5w": "half-week", "1w": "weekly", "2w": "2-week", "1M": "monthly", "3M": "quarterly"}
COUNT_WORDS = {1: "once", 2: "twice", 3: "three times"}
STATE_HEAD = {"A": ("📈 UP SETUP", "📉 DOWN SETUP", "📈 SETUP"),
              "B": ("⏸️ UPTREND PAUSING", "⏸️ DOWNTREND PAUSING", "⏸️ TREND PAUSING"),
              "C": ("⚠️ UPTREND WEAKENING", "⚠️ DOWNTREND WEAKENING", "⚠️ TREND WEAKENING"),
              "N": ("↔️ NO TREND", "↔️ NO TREND", "↔️ NO TREND")}


def _tf(tf: str | None) -> str:
    return TF_WORDS.get(tf or "", tf or "")


def _candle(tf: str | None) -> str:
    """'4-hour', 'daily': the candle's name in a sentence ('a 4-hour candle closes above …')."""
    return CANDLE_WORDS.get(tf or "", tf or "")


def _horizon(desk: str) -> str:
    return DESK_HORIZON.get(desk, DESK_LABEL.get(desk, str(desk).capitalize()))


def _header(what: str, where: str, price: float | None) -> str:
    """'📉 DOWN SETUP · next 4 hours · BTC $80,180' - one shape for every message."""
    return f"{what} · {where}" + (f" · BTC ${price:,.0f}" if price else "")


def _count(n: int) -> str:
    return COUNT_WORDS.get(n, f"{n} times")


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


def explain(previous: str, cascade, desk: str = "live") -> str:
    """What changed and what it means, one or two lines, built from the cascade's own reads."""
    bias, anchor = cascade.bias, cascade.anchor
    big = _tf(bias.timeframe if bias else None)
    engines = "the shorter candles (" + " and ".join(reversed([_tf(r.timeframe) for r in cascade.engine])) + ")" \
        if cascade.engine else "the shorter candles"
    trend = bias.direction if bias else None
    word = "up" if trend == "bullish" else "down"
    state = cascade.state
    anchor_candle = _candle(anchor.timeframe) if anchor else "anchor"
    if state == "N":
        return (f"The {big} has no clear trend, so there is no bigger direction to follow. "
                f"Trade the edges of the {_tf(anchor.timeframe if anchor else None)} range, not breakouts.")
    if state == "A":
        line = (f"The {big} trend is {word} and {engines} now agree: the best conditions for a move "
                f"{'higher' if trend == 'bullish' else 'lower'} {_horizon(desk)}.\n"
                "No trade from this alone: wait for the report's plan or the Pulse.")
        if cascade.vwap:
            line += (f" The trend's average price is ${cascade.vwap:,.0f}; a {anchor_candle} candle closing "
                     f"{'below' if trend == 'bullish' else 'above'} it ends this.")
        return line
    if state == "C":
        # election-day fix 5: the level, the count and what confirms it, not "watch for a reversal"
        # the anchored VWAP in plain words: the average price since the bias candle opened
        beyond = "above" if trend == "bearish" else "below"
        n = cascade.closes_beyond_vwap or 0
        if cascade.vwap:
            level = f"${cascade.vwap:,.0f}"
            first = (f"{engines.capitalize()} have closed {beyond} the {big} trend's average price {level} {_count(n)}."
                     if n else f"{engines.capitalize()} have started to turn against the {big} {word}trend but have not "
                               f"yet closed {beyond} its average price {level}.")
        else:
            level = f"the {big} average price"
            first = (f"{engines.capitalize()} have closed {beyond} {level} {_count(n)} against the {big} {word}trend."
                     if n else f"{engines.capitalize()} have started to turn against the {big} {word}trend but have not "
                               f"yet closed {beyond} its average price.")
        return (f"{first}\nNot a trade yet. It becomes a reversal if a {anchor_candle} candle closes {beyond} {level}; "
                "a REVERSAL CONFIRMED alert will say so.")
    # B: the trend stands; the shorter timeframes are mixed or leaning against it without having taken over
    if previous == "A":
        return f"The {big} trend is still {word}, but {engines} lost their push. A breather, not a reversal."
    if previous == "C":
        return f"The {big} trend is {word} and {engines} stopped fighting it. Less danger than before."
    if previous == "N":
        return f"The {big} picked a direction again: {word}. {engines.capitalize()} are mixed, so this is a pause inside it."
    return f"The {big} trend is {word}; {engines} are mixed. A pause inside the trend."


def cascade_event(desk: str, label: str, previous: str | None, cascade, candle_ms: int,
                  price: float | None = None) -> Event | None:
    """The alert for a desk whose cascade state differs from the last one recorded, or None.

    The header (what happened, the desk's horizon, the price), then what changed and what to do."""
    state = cascade.state
    if previous is None or previous == state or state in SILENT_STATES or previous in SILENT_STATES:
        return None
    trend = cascade.bias.direction if cascade.bias is not None else None
    heads = STATE_HEAD.get(state, (state, state, state))
    what = heads[0] if trend == "bullish" else heads[1] if trend == "bearish" else heads[2]
    head = _header(what, _horizon(desk), price) + "\n" + explain(previous, cascade, desk)
    return Event("shift" if state == "C" else "cascade", f"cascade:{desk}:{previous}->{state}:{candle_ms}",
                 head, price)


def trap_events(traps: list[dict], now_ms: int) -> list[Event]:
    """A declared trap, and a settled one, each announced once - only those from the last 24 hours."""
    out: list[Event] = []
    for t in traps:
        bull = str(t.get("side", "")).upper().startswith("BULL")
        level, target = float(t.get("level") or 0), float(t.get("plays_out") or 0)
        invalid = float(t.get("invalidation") or 0)
        horizon = _horizon(str(t.get("category") or ""))
        tf = str(t.get("timeframe") or "")
        if tf and is_short_horizon(tf):
            tf = f"{tf}, {SHORT_HORIZON_LABEL}"
        where = f"{horizon} ({tf})" if tf else horizon
        move = f"pop above ${level:,.0f}" if bull else f"dip below ${level:,.0f}"
        declared = int(t.get("declared_ms") or 0)
        if now_ms - declared <= ANNOUNCE_WINDOW_MS:
            out.append(Event("trap", f"trap:{t['id']}:declared",
                             f"{_header('🪤 TRAP CALL', where, level)}\n"
                             f"The {move} should fail and {'drop' if bull else 'bounce'}.\n"
                             f"Target ${target:,.0f} · wrong if price goes past ${invalid:,.0f}.", t.get("level")))
        status, resolved = str(t.get("status") or ""), t.get("resolved_ms")
        if status in ("played_out", "invalidated") and resolved and now_ms - int(resolved) <= ANNOUNCE_WINDOW_MS:
            at = float(t.get("resolved_price") or 0)
            if status == "played_out":
                text = f"{_header('✅ TRAP WORKED', where, at)}\nThe {move} failed as called: price reached the target ${target:,.0f}."
            else:
                text = (f"{_header('❌ TRAP FAILED', where, at)}\nThe {move} was real: price went past "
                        f"${invalid:,.0f}. The call was wrong.")
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
    big, candle = _tf(cascade.bias.timeframe), _candle(cascade.anchor.timeframe)
    beyond, word = ("above", "down") if trend == "bearish" else ("below", "up")
    held, move = ("short", "up") if trend == "bearish" else ("long", "down")
    back = "below" if trend == "bearish" else "above"
    from core.qf_alerts import _clock
    utc, pkt = _clock(hit["close_ms"])
    head = (_header("🔄 REVERSAL CONFIRMED", _horizon(desk), price) + "\n"
            f"A {candle} candle closed at ${hit['close']:,.0f} ({utc} UTC, {pkt} PKT), {beyond} the {big} trend's average "
            f"price ${cascade.vwap:,.0f}: the {big} {word}trend has lost control.\n"
            f"If you are {held} from the trend, it is over. Expect a move {move}. "
            f"Wrong only if a {candle} candle closes back {back} ${cascade.vwap:,.0f}.")
    return Event("reversal", f"cascade:{desk}:C->{CONFIRMED}:{hit['t_ms']}", head, price)
