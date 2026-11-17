"""The Active Trade plan's deterministic half, out of app.py so the worker can open a card without Streamlit.

Everything a card needs, computed once at open: the verified stop, the layout, the timed targets, the
pullback band, the monitoring schedule and what the entry rests on. The agents' verdict (ten sub-agents,
the Head, the handshakes) is added by the caller on top; a card still carries its levels when the model
is down, or when nobody asked the model at all."""
from __future__ import annotations

from dataclasses import replace

from core import trades
from core.agents.recon_profiles import PROFILES
from core.agents.trade_desk import category_alignment, plan_layout
from core.indicators.fib_time import fib_time_zones


def deterministic_plan(store, state: dict, frame, side: str, entry: float, given_stop: float | None,
                       category: str, now_ms: int, gamma: dict | None = None,
                       asset: str = "BTC/USDT") -> tuple[dict, trades.StopVerdict, list[trades.TimedTarget]]:
    """(plan, stop verdict, targets) for a position judged over `category`'s desk. `state` is that desk's own
    state (core.agents.live_recon.build_state), `frame` its middle timeframe's candles, `gamma` its walls."""
    profile = PROFILES[category]
    ltf, mtf, htf = profile.timeframes
    atr_value = (state.get("volatility", {}).get(mtf) or {}).get("atr")
    atr = float(atr_value) if atr_value else entry * trades.DEFAULT_ATR_PCT

    stop = trades.verify_stop(state, side, entry, given_stop, mtf, atr)
    fib = fib_time_zones(frame, atr_value=atr) if frame is not None and not frame.empty else None
    # his v28 counter-trend rule: the layout is decided from the desks' anchored verdicts (ledger, no model)
    try:
        layout = plan_layout(category_alignment(store, side, now_ms), category)
    except Exception as e:  # noqa: BLE001 - an unreadable ledger leaves the trend layout
        layout = plan_layout((), category)
        layout = replace(layout, note=f"{layout.note} (desks unreadable: {e})")
    targets = trades.timed_targets(state, side, entry, stop.stop, htf, atr, fib, now_ms, layout=layout.kind)
    pullback = trades.pullback_band(frame, side, timeframe=mtf)
    schedule = trades.monitoring_schedule(now_ms, profile.timeframes, targets)

    plan = {
        "asset": asset, "side": side, "entry": entry, "category": category, "timeframe": mtf,
        "horizon": profile.horizon, "atr": round(atr, 2), "built_ms": now_ms,
        "layout": layout.kind, "layout_note": layout.note,
        # his Section 6: what the entry rests on, so a later pass can see it shift before the stop is hit
        "basis": trades.entry_basis(trades.flow_and_walls(state, mtf, gamma), side),
        "stop_verdict": stop.verdict, "stop_reasons": list(stop.reasons),
        "protects_at": stop.protects_at, "protects_kind": stop.protects_kind,
        "targets": [{"name": t.name, "price": t.price, "basis": t.basis, "due_ms": t.due_ms,
                     "due_from": t.due_from, "reward_r": t.reward_r} for t in targets],
        "pullback": {"available": pullback.available, "note": pullback.note,
                     "ordinary": pullback.ordinary, "edge": pullback.edge, "breaking": pullback.breaking,
                     "horizon_bars": pullback.horizon_bars, "timeframe": pullback.timeframe,
                     "levels": pullback.levels(side, entry) if pullback.available else {}},
        "schedule": [{"timeframe": c.timeframe, "at_ms": c.at_ms, "reason": c.reason} for c in schedule],
        "ok": False, "notes": [],
    }
    return plan, stop, targets
