from dataclasses import dataclass

@dataclass
class MasterTradeCard:
    tier: str               # "INTRADAY SWING MASTER" | "MONTHLY MACRO MASTER"
    action: str             # "BUY" | "SELL" | "WAIT"
    entry_zone: str
    targets: list[float]
    safety_line: float
    aligned_count: int
    pre_invalidation_warning: str | None
    summary_line: str

def solve_one_sided_puzzle(
    desk_key: str,
    families: list,
    preemptive: PreemptiveSignal,
    entry_zone: str,
    targets: list[float],
    safety_line: float,
) -> dict:
    """
    Combines all data feeds into one decisive direction.
    1. DEFCON-1 preemptive trap overrides lagging candle closes.
    2. Otherwise, requires >= 3 core confluences aligned on one side.
    """
    if preemptive.defcon_1 and preemptive.direction in ("LONG", "SHORT"):
        action = "BUY" if preemptive.direction == "LONG" else "SELL"
        return {
            "verdict": "DEFCON_1_PREEMPTIVE",
            "side": preemptive.direction,
            "action": action,
            "entry": entry_zone,
            "targets": [preemptive.target_low, preemptive.target_high],
            "safety_line": preemptive.safety_line,
            "aligned_count": 4,
            "reason": preemptive.plain_english_reason,
        }

    bullish = [f for f in families if f.side == "LONG"]
    bearish = [f for f in families if f.side == "SHORT"]

    if len(bullish) >= 3 and len(bullish) > len(bearish):
        return {
            "verdict": "TRADE",
            "side": "LONG",
            "action": "BUY",
            "entry": entry_zone,
            "targets": targets,
            "safety_line": safety_line,
            "aligned_count": len(bullish),
            "reason": f"{len(bullish)} major structural confluences aligned bullish.",
        }

    if len(bearish) >= 3 and len(bearish) > len(bullish):
        return {
            "verdict": "TRADE",
            "side": "SHORT",
            "action": "SELL",
            "entry": entry_zone,
            "targets": targets,
            "safety_line": safety_line,
            "aligned_count": len(bearish),
            "reason": f"{len(bearish)} major structural confluences aligned bearish.",
        }

    return {
        "verdict": "WAIT",
        "side": "NONE",
        "action": "WAIT / SITTING OUT IS AN EDGE",
        "entry": "None",
        "targets": [],
        "safety_line": 0.0,
        "aligned_count": max(len(bullish), len(bearish)),
        "reason": "Order flow and options walls are balanced; fewer than 3 core confluences align.",
    }
