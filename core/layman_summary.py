from core.config import format_dual_time

def render_dispatch_text(
    category_name: str,
    current_price: float,
    decision: dict,
    preemptive: PreemptiveSignal,
    window_utc_pkt: str,
) -> str:
    ts = format_dual_time()
    is_emergency = decision["verdict"] == "DEFCON_1_PREEMPTIVE"

    header = (
        f"🚨 EMERGENCY PREEMPTIVE ALERT ({category_name.upper()})"
        if is_emergency
        else f"📡 WAR ROOM DIRECTIVE ({category_name.upper()})"
    )

    if decision["action"] in ("BUY", "SELL"):
        t_str = " – ".join(f"${t:,.0f}" for t in decision["targets"])
        move_label = "Upside Squeeze Target" if decision["action"] == "BUY" else "Downside Liquidation Target"
        warning_line = (
            f"Preemptive Notice: {preemptive.plain_english_reason} "
            f"{'Build short hedges / close longs' if decision['action'] == 'SELL' else 'Position for upside squeeze'} immediately."
            if is_emergency
            else f"One-Sided Execution: {decision['action']} at {decision['entry']}. {decision['reason']}"
        )
        return (
            f"========================================\n"
            f"{header}\n"
            f"⏰ Time: {ts}\n"
            f"========================================\n\n"
            f"📈 CURRENT MARKET STATE & PRICE:\n"
            f"- Current Price: ${current_price:,.0f} (Live Tracking)\n"
            f"- Trend Phase: {decision['reason']}\n\n"
            f"🕒 EXPECTED TIMELINE & REVERSAL WINDOW:\n"
            f"- Preemptive Window: {window_utc_pkt}\n\n"
            f"📍 KEY LEVELS & TARGETS:\n"
            f"- {move_label}: {t_str}\n"
            f"- Safety Line (Hard Invalidation): ${decision['safety_line']:,.0f}\n\n"
            f"⚠️ ACTION DIRECTIVE:\n"
            f"- {warning_line}\n\n"
            f"Trade Wisely! ⚠️\n"
            f"========================================"
        )

    return (
        f"========================================\n"
        f"{header}\n"
        f"⏰ Time: {ts}\n"
        f"========================================\n"
        f"- Current Price: ${current_price:,.0f}\n"
        f"- Directive: WAIT / SITTING OUT IS AN EDGE ({decision['reason']})\n"
        f"========================================"
    )
