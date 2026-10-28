"""The Quant Forecast as the JSON payload Hammad specified (message 6, section 5): the plan's status and the
bot's compact message, the layman summary with the timeline tree and the trade directive box, the chart
payload hour by hour with its overlays, and - kept from message 5 - the raw math for the drawer.

Every number is the plan's own; nothing is computed here. The hourly trajectory's projected_high and
projected_low are the 68% band of the simulated paths at that hour (the envelope the chart draws), not an
hourly candle's range - the simulation does not make hourly candles, and saying otherwise would overstate
it; the `note` field says so in the payload itself. Nothing consumes this yet: the dashboard offers it as a
download from the drawer."""
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
            now_ms: int, h1=None) -> dict:
    from core.qf_alerts import REGIME_WORD, _clock, timeline_line, trap_label
    from ui.panels import QF_STATUS
    hours = plan.get("hours") or []
    parts = plan.get("parts") or {}
    ph, pl = plan.get("projected_high") or {}, plan.get("projected_low") or {}

    def coord(d: dict) -> dict:
        if not d:
            return {"price": None, "utc_time": None, "pkt_time": None}
        u, k = _clock(d["t_ms"])
        return {"price": d["price"], "utc_time": u, "pkt_time": k}

    traj, prev = [], plan["origin"]
    for i, h in enumerate(hours):
        u, k = _clock(h["t_ms"])
        traj.append({"hour_index": i + 1, "timestamp_utc": u, "timestamp_pkt": k, "projected_open": prev,
                     "projected_high": h["band_high"], "projected_low": h["band_low"], "projected_close": h["median"],
                     "hourly_volatility_garch_bps": qf._bps(h.get("sigma_full_pct", h["sigma_pct"])),
                     "regime_type": REGIME_TYPE.get(h["phase"], "NORMAL")})
        prev = h["median"]
    nodes = []
    for n in timeline_tree(plan, now_ms, h1):
        wu, wk = _window(n["start_ms"], n["end_ms"])
        node = {"time_window_utc": wu, "time_window_pkt": wk, "state": n["state"], "phase": n["phase"],
                "summary_oneliner": n["summary"], "rationale_why": n["why"],
                "price_levels": {"low": n["low"], "high": n["high"]}, "expected_move_bps": n["bps"],
                "key_levels_inside": [lv["price"] for lv in n["levels"]]}
        if "actual_low" in n:
            node["actual_range"] = {"low": n["actual_low"], "high": n["actual_high"]}
        nodes.append(node)
    d = trade_directive(plan)
    box = {"asset": d["asset"], "direction": d["direction"], "setup_type": d["setup_type"],
           "stop_loss_invalidation": d["stop_loss_invalidation"], "entry_zone": d["entry_zone"],
           "tp1": d["tp1"], "tp2": d["tp2"], "tp3": d["tp3"], "risk_reward": d["risk_reward"],
           "active_window_utc": None, "active_window_pkt": None}
    if d["active_start_ms"]:
        box["active_window_utc"], box["active_window_pkt"] = _window(d["active_start_ms"], d["active_end_ms"])
    chop = [{"start_utc": _clock(w["start_ms"])[0], "end_utc": _clock(w["end_ms"])[0], "high": w["upper"], "low": w["lower"]}
            for w in plan.get("windows") or [] if w["phase"] == "chop"]
    rate = qf.hit_rate(plan, live)
    p = plan.get("p_scenario_a")
    voided = [r for r in plans_today if r.get("status") == "invalidated"]
    return {
        "audit_and_alert_status": {
            "plan_id": plan["version"], "tier_status": QF_STATUS.get(status, (status.upper(),))[0],
            "re_forecast_count": len(voided), "trap_agent": trap_label(plan),
            "magnitude_scaler": plan.get("magnitude") or {"scaled": False},
            "timeline_status": timeline_line(plan, now_ms), "bot_message": bot_message or ""},
        "layman_summary_and_timeline_tree": {
            "master_regime": REGIME_WORD[plan["regime"]],
            "confidence_odds": (f"{p:.0%}" if p is not None else "") + (f" (Hit rate: {rate})" if rate else ""),
            "plain_english_plan": qf.wire_line(plan),
            "daily_coordinates": {"predicted_low": coord(pl), "predicted_high": coord(ph),
                                  "whole_day_macro_invalidation": plan.get("invalidation") or {}},
            "timeline_tree_nodes": nodes,
            "trade_directive_box": box},
        "tradingview_chart_payload": {
            "macro_anchor": "1-Day", "forecast_origin_timestamp": _iso(plan["created_ms"]),
            "forecast_end_timestamp": _iso(plan["end_ms"]),
            "note": "projected_high/low per hour are the 68% band of the simulated paths, not an hourly candle range",
            "hourly_trajectory_24h": traj,
            "visual_overlays": {
                "chop_boxes": chop,
                "scenario_b_secondary_chart_active": plan["version"] >= 1 and bool(voided),
                "right_axis_badges": {"predicted_high": ph.get("price"), "predicted_low": pl.get("price"),
                                      "whole_day_invalidation_level": plan.get("invalidation") or {},
                                      "lvn_air_pockets": [round((z["low"] + z["high"]) / 2, 2)
                                                          for z in plan.get("liquidation_zones") or []]}}},
        "collapsible_raw_math_drawer": {
            "garch_alpha": (plan.get("garch") or {}).get("alpha"), "garch_beta": (plan.get("garch") or {}).get("beta"),
            "garch_compression": (plan.get("garch") or {}).get("compression"),
            "hurst_exponent": (parts.get("momentum") or {}).get("hurst"),
            "cvd_24h_delta": (parts.get("flow") or {}).get("delta_btc"),
            "open_interest_24h_change_pct": (parts.get("open_interest") or {}).get("pct"),
            "funding_rate_8h": (parts.get("funding") or {}).get("rate"),
            "lean_score": plan.get("score"), "lean_parts": {k: v.get("value") for k, v in parts.items()},
            "volatility_schedule_4h": qf.volatility_schedule(plan, now_ms)["blocks"],
            "backtest_summary_stats": plan.get("backtest") or {}},
    }
