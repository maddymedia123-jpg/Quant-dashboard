"""The Quant Forecast as the JSON payload Hammad specified (message 9, section 4): the plan's status with
the self-audit note and the bot's compact message, the Sticky Monday Master Plan, the layman summary with
the timeline tree and the trade directive box, the chart payload as 1-hour projected candles with its
overlays, and - kept from message 5 as the one extra block - the raw math for the drawer.

Every number is the plan's own; nothing is computed here. The projected candles are the median path drawn
hour by hour (open = the path an hour before, close = the path at the hour, high and low = that hour's
68% box) - the simulation does not make candles, and the `note` field says so in the payload itself.
Nothing consumes this yet: the dashboard offers it as a download from the drawer."""
from __future__ import annotations

from datetime import datetime, timezone

from core import quant_forecast as qf
from core.qf_tree import timeline_tree, trade_directive

REGIME_TYPE = {"chop": "CHOP_BOX", "expansion": "EXPANSION_VECTOR", "normal": "NORMAL"}


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _window(start_ms: int, end_ms: int) -> tuple[str, str]:
    from core.qf_alerts import _clock
    (us, ks), (ue, ke) = _clock(start_ms), _clock(end_ms)
    return f"{us} - {ue}", f"{ks} - {ke}"


def payload(plan: dict, status: str, plans_today: list[dict], bot_message: str, live: dict | None,
            now_ms: int, h1=None, book: dict | None = None, week: dict | None = None,
            self_audit: dict | None = None) -> dict:
    from core.qf_alerts import REGIME_WORD, _clock, timeline_line, trap_label
    from ui.panels import QF_STATUS
    parts = plan.get("parts") or {}
    ph, pl = plan.get("projected_high") or {}, plan.get("projected_low") or {}
    inv = plan.get("invalidation") or {}

    candles = []
    for c in qf.projected_candles(plan):
        u, k = _clock(c["t_ms"])
        candles.append({"hour_index": c["hour_index"], "timestamp_utc": u, "timestamp_pkt": k, "open": c["open"],
                        "high": c["high"], "low": c["low"], "close": c["close"], "color": c["color"],
                        "regime_type": REGIME_TYPE.get(c["phase"], "NORMAL")})
    nodes = []
    for n in timeline_tree(plan, now_ms, h1):
        wu, wk = _window(n["start_ms"], n["end_ms"])
        node = {"time_window_utc": wu, "time_window_pkt": wk, "state": n["state"],
                "summary_oneliner": n["summary"], "rationale_why": n["why"],
                "price_levels": {"low": n["low"], "high": n["high"]}}
        if "actual_low" in n:
            node["actual_range"] = {"low": n["actual_low"], "high": n["actual_high"]}
        nodes.append(node)
    d = trade_directive(plan)
    box = {"direction": d["direction"], "setup_type": d["setup_type"], "stop_loss_invalidation": d["stop_loss_invalidation"],
           "entry_zone": d["entry_zone"], "tp1": d["tp1"], "tp2": d["tp2"], "tp3": d["tp3"],
           "active_window_utc": None, "active_window_pkt": None}
    if d["active_start_ms"]:
        box["active_window_utc"], box["active_window_pkt"] = _window(d["active_start_ms"], d["active_end_ms"])
    chop = [{"start_utc": _clock(w["start_ms"])[0], "end_utc": _clock(w["end_ms"])[0], "high": w["upper"], "low": w["lower"]}
            for w in plan.get("windows") or [] if w["phase"] == "chop"]
    rate = qf.hit_rate(plan, live)
    p = plan.get("p_scenario_a")
    voided = [r for r in plans_today if r.get("status") == "invalidated"]
    # his schema names one invalidation level; a range day has one each side and carries both
    inv_level = next(iter(inv.values())) if len(inv) == 1 else (inv or None)
    walls = (week or {}).get("weekly_gex_walls") or [None, None]
    return {
        "audit_and_alert_status": {
            "plan_id": plan["version"], "tier_status": QF_STATUS.get(status, (status.upper(),))[0],
            "self_audit_correction_note": (self_audit or {}).get("note"),
            "bot_message": bot_message or ""},
        "monday_master_plan_sticky": {
            "weekly_bias": (week or {}).get("weekly_bias"),
            "weekly_gex_walls": list(walls[:2]),
            "macro_baseline_summary": (week or {}).get("macro_baseline_summary")},
        "layman_summary_and_timeline_tree": {
            "master_regime": REGIME_WORD[plan["regime"]],
            "confidence_odds": (f"{p:.0%}" if p is not None else "") + (f" (Hit rate: {rate})" if rate else ""),
            "timeline_tree_nodes": nodes,
            "trade_directive_box": box},
        "tradingview_chart_payload": {
            "macro_anchor": "1-Day", "forecast_origin_timestamp": _iso(plan["created_ms"]),
            "note": "projected candles are the median path drawn hour by hour; high/low are that hour's 68% box, not a simulated candle range",
            "projected_candles_24h": candles,
            "visual_overlays": {
                "transparent_chop_bands": chop,
                "right_axis_badges": {"predicted_high": ph.get("price"), "predicted_low": pl.get("price"),
                                      "whole_day_invalidation_level": inv_level,
                                      "liquidation_cluster_levels": [round((z["low"] + z["high"]) / 2, 2)
                                                                     for z in plan.get("liquidation_zones") or []]}}},
        "collapsible_raw_math_drawer": {
            "garch_alpha": (plan.get("garch") or {}).get("alpha"), "garch_beta": (plan.get("garch") or {}).get("beta"),
            "garch_compression": (plan.get("garch") or {}).get("compression"),
            "hurst_exponent": (parts.get("momentum") or {}).get("hurst"),
            "cvd_24h_delta": (parts.get("flow") or {}).get("delta_btc"),
            "open_interest_24h_change_pct": (parts.get("open_interest") or {}).get("pct"),
            "funding_rate_8h": (parts.get("funding") or {}).get("rate"),
            "lean_score": plan.get("score"), "lean_parts": {k: v.get("value") for k, v in parts.items()},
            "plain_english_plan": qf.wire_line(plan),
            "daily_coordinates": {"predicted_low": {"price": pl.get("price"), "t_ms": pl.get("t_ms")},
                                  "predicted_high": {"price": ph.get("price"), "t_ms": ph.get("t_ms")},
                                  "whole_day_macro_invalidation": inv},
            "re_forecast_count": len(voided), "scenario_b_secondary_chart_active": plan["version"] >= 1 and bool(voided),
            "trap_agent": trap_label(plan), "magnitude_scaler": plan.get("magnitude") or {"scaled": False},
            "timeline_status": timeline_line(plan, now_ms),
            "timeline_expected_move_bps": [{"time_window_utc": _window(n["start_ms"], n["end_ms"])[0], "bps": n["bps"],
                                            "key_levels_inside": [lv["price"] for lv in n["levels"]]}
                                           for n in timeline_tree(plan, now_ms, h1)],
            "volatility_schedule_4h": qf.volatility_schedule(plan, now_ms)["blocks"],
            "order_book_scoop_and_sweep": book or {"available": False, "reason": "no report"},
            "self_audit": self_audit, "monday_plan": week,
            "backtest_summary_stats": plan.get("backtest") or {}},
    }
