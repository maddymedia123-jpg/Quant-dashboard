"""Short-horizon traps (owner's decision 4 Nov on his v22 Section 1.4): a 15m or 1h trap stays on the Live
desk only, is labelled short-horizon wherever it is written, and never reaches a desk summary or the Pulse."""
import pathlib
import re

from core import traps
from core.agents import trade_desk as td
from core.agents.recon_profiles import PROFILES
from core import market_alerts as ma
from core import mcp_tools
from core.store import Store
from ui import panels

NOW = 1_760_000_000_000
H = 3_600_000
ROOT = pathlib.Path(__file__).resolve().parents[1]


def text(h: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", h or "")).strip()


def row(**over) -> dict:
    base = {"id": 1, "category": "live", "side": "bull_trap", "timeframe": "1h", "level": 78_600.0,
            "invalidation": 78_900.0, "plays_out": 78_000.0, "declared_ms": NOW - H, "score": 72.0,
            "head_call": "ENGINEERED_TRAP", "evidence": "[]", "notes": "[]", "status": "active",
            "resolved_ms": None, "resolved_price": None}
    base.update(over)
    return base


def test_the_rule_names_15m_and_1h_and_nothing_longer():
    assert traps.SHORT_HORIZON == ("15m", "1h")
    assert [traps.horizon(tf) for tf in ("15m", "1h", "4h", "1d", "1w", "1M")] == ["short", "short", "swing", "swing", "swing", "swing"]
    assert traps.tf_label("1h") == "1h · short-horizon" and traps.tf_label("4h") == "4h"


def test_only_the_live_desk_sees_short_horizon_traps():
    seen = {k: traps.desk_timeframes(k, p.timeframes) for k, p in PROFILES.items()}
    assert seen["live"] == ("15m", "1h", "4h"), "Live keeps all three"
    assert seen["intraday"] == ("4h", "1d"), "Intraday reads the 1h candle but does not trade 1h traps"
    assert seen["weekly"] == ("4h", "1d", "1w") and seen["monthly"] == ("1d", "1w", "1M")
    rows = [row(id=1, timeframe="15m"), row(id=2, timeframe="1h"), row(id=3, timeframe="4h")]
    assert [r["id"] for r in traps.visible_to(rows, "live")] == [1, 2, 3]
    assert [r["id"] for r in traps.visible_to(rows, "intraday")] == [3]


def test_every_place_a_trap_is_written_carries_the_label():
    assert "1h · short-horizon at $78,600" in text(panels.trap_warning_html([row()], NOW))
    assert "1h · short-horizon at $78,600" in text(panels.trap_card_html(row(), NOW, "Live Recon"))
    assert "4h at $78,600" in text(panels.trap_card_html(row(timeframe="4h"), NOW, "Live Recon"))
    settled = text(panels.trap_settled_html([row(status="played_out", resolved_ms=NOW, resolved_price=78_000.0)], NOW))
    assert "1h · short-horizon" in settled
    cand = type("C", (), {"side": "bull_trap", "timeframe": "15m", "level": 78_400.0, "strength": 2, "evidence": ("a", "b")})()
    assert "15m · short-horizon" in text(panels.trap_watch_html([cand], NOW, NOW))
    e = ma.trap_events([{**row(), "declared_ms": NOW - 60_000}], NOW)
    assert e[0].headline.startswith("🪤 TRAP CALL · next 4 hours (1h, short-horizon) · BTC $")
    e4 = ma.trap_events([{**row(timeframe="4h"), "declared_ms": NOW - 60_000}], NOW)
    assert e4[0].headline.startswith("🪤 TRAP CALL · next 4 hours (4h) · BTC $")


def test_a_swing_desk_trade_is_not_warned_off_by_a_short_horizon_trap(tmp_path):
    store = Store(":memory:")
    store.put_trap(declared_ms=NOW - H, category="live", side="bull_trap", timeframe="1h", level=80_000.0,
                   invalidation=80_400.0, plays_out=79_000.0, score=70.0, head_call="ENGINEERED_TRAP",
                   evidence=[], notes=[], detail={})
    targets = []
    assert td.trap_proximity(store, "LONG", 80_050.0, 79_000.0, targets, 400.0, "live"), "Live sees it"
    assert td.trap_proximity(store, "LONG", 80_050.0, 79_000.0, targets, 400.0, "weekly") == (), "Weekly does not"
    assert td.trap_proximity(store, "LONG", 80_050.0, 79_000.0, targets, 400.0), "the default is the Live desk"


def test_the_mcp_tool_names_the_horizon(tmp_path):
    store = Store(":memory:")
    store.put_trap(declared_ms=NOW - H, category="live", side="bull_trap", timeframe="15m", level=80_000.0,
                   invalidation=80_400.0, plays_out=79_000.0, score=70.0, head_call="ENGINEERED_TRAP",
                   evidence=[], notes=[], detail={})
    store.put_trap(declared_ms=NOW - 2 * H, category="weekly", side="bear_trap", timeframe="1d", level=70_000.0,
                   invalidation=69_600.0, plays_out=71_000.0, score=70.0, head_call="ENGINEERED_TRAP",
                   evidence=[], notes=[], detail={})
    out = mcp_tools.traps(store, now_ms=NOW)
    assert [(t["timeframe"], t["horizon"]) for t in out["traps"]] == [("15m", "short"), ("1d", "swing")]
    assert "Live desk only" in out["how_to_read"]


def test_the_summaries_and_the_pulse_never_read_the_trap_ledger():
    """The ban is structural: neither module imports the ledger or the trap scanner, so a 15m trap cannot
    reach a desk summary or a Pulse dispatch by any path."""
    for rel in ("core/layman_summary.py", "core/pulse.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert "store.traps" not in src and "from core.traps" not in src and "import core.traps" not in src, rel
