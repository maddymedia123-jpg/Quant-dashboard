"""Hammad's Section 6: ONE desk-owned Master Swing Trade card, opened by a higher desk's confluence gate and
held until its targets, its hard invalidation, or the desk turning.

The deterministic plan builder is the one the Active Trade screen uses, moved into core so the worker can
open a card without Streamlit. The desk reader is injected: these tests hand in fake desks, the worker
hands in the real ones."""
from types import SimpleNamespace as NS

import pytest

from core import master_card as mc
from core import trade_plan
from core.agents.recon_profiles import PROFILES
from core.confluence import Gate
from core.layman_summary import Framework
from core.store import Store
from core.trades import LONG, SHORT, progress
from tests.test_trades import candles, state

NOW = 1_760_000_000_000
ATR = 200.0


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


def framework(side=LONG, zone=(79_900.0, 80_100.0), invalidation=79_000.0, targets=(81_000.0, 82_000.0)):
    if side == SHORT:
        return Framework("Bearish", zone, invalidation, targets, "", SHORT, target3=None, level=sum(zone) / 2)
    return Framework("Bullish", zone, invalidation, targets, "", LONG, target3=83_000.0, level=sum(zone) / 2)


def gate(side=LONG, verdict="TRADE", reason="TRADE: pattern, options and flow agree with the floor."):
    return Gate(side, verdict, (), reason)


class FakeDesks:
    """What the worker's reader gives master_card: price, and per desk its state, chart, gamma and gate."""

    def __init__(self, price=80_050.0, gates=None):
        self.price = price
        self.gates = gates or {}
        self.asked = []

    def state(self, category):
        return state()

    def frame(self, category):
        return candles()

    def gamma(self, category):
        return {"put_wall": 79_500.0, "call_wall": 83_000.0}

    def gate(self, category):
        self.asked.append(category)
        return self.gates.get(category, (Framework("No clear bias", None, None, (None, None), "", "NONE"),
                                         Gate("NONE", "NO SETUP", (), "NO SETUP: no level to trade from.")))


# ---------- the plan builder, out of app.py ----------
def test_the_deterministic_plan_carries_every_level_the_card_needs(store):
    plan, stop, targets = trade_plan.deterministic_plan(
        store, state(), candles(), LONG, 80_000.0, None, "live", NOW, gamma={"put_wall": 79_500.0, "call_wall": 83_000.0})
    assert plan["asset"] == "BTC/USDT" and plan["side"] == LONG and plan["category"] == "live"
    assert plan["timeframe"] == PROFILES["live"].timeframes[1] and plan["horizon"] == PROFILES["live"].horizon
    assert plan["basis"] == {"timeframe": plan["timeframe"], "flow_state": None, "wall": 79_500.0, "wall_kind": "put wall"}
    assert stop.stop < 80_000.0 and plan["stop_verdict"] == stop.verdict and plan["stop_reasons"]
    assert targets and plan["targets"][0]["name"] == targets[0].name and plan["targets"][0]["price"] == targets[0].price
    assert plan["layout"] in ("trend", "mean-reversion") and plan["layout_note"]
    assert plan["schedule"] and plan["pullback"]["timeframe"] == plan["timeframe"]
    assert plan["ok"] is False and plan["notes"] == [], "the agents' verdict is added by the caller, not here"
    given = trade_plan.deterministic_plan(store, state(), candles(), LONG, 80_000.0, 79_000.0, "live", NOW)[0]
    assert given["basis"]["wall"] is None, "no gamma handed in: no wall recorded"


# ---------- opening ----------
def test_a_higher_desk_gate_saying_trade_with_price_in_the_zone_opens_the_one_master_card(store):
    desks = FakeDesks(gates={"weekly": (framework(), gate())})
    tid = mc.open_master(store, desks, NOW)
    row = store.trade(tid)
    assert row["category"] == "weekly" and row["side"] == LONG and row["entry"] == 80_000.0, "the zone's middle"
    assert row["given_stop"] == 79_000.0, "the desk's invalidation is the hard stop handed to the verifier"
    plan = row["plan"]
    assert plan["master"] is True and plan["desk"] == "weekly"
    assert plan["gate"] == {"verdict": "TRADE", "reason": "TRADE: pattern, options and flow agree with the floor.",
                            "zone": [79_900.0, 80_100.0], "invalidation": 79_000.0}
    assert plan["targets"], "the plan builder's timed targets, not the framework's bare levels"
    assert any("Master swing" in n for n in plan["notes"])
    ev = store.trade_events(tid)
    assert [e["kind"] for e in ev] == ["master"] and "opened" in ev[0]["headline"] and "weekly desk" in ev[0]["headline"]
    assert mc.open_master(store, desks, NOW + 60_000) is None, "one master card at a time"
    assert len([t for t in store.trades(status="open") if (t.get("plan") or {}).get("master")]) == 1


def test_nothing_opens_when_the_gate_waits_or_price_is_away_from_the_zone(store):
    assert mc.open_master(store, FakeDesks(gates={"weekly": (framework(), gate(verdict="WAIT"))}), NOW) is None
    assert mc.open_master(store, FakeDesks(price=83_000.0, gates={"weekly": (framework(), gate())}), NOW) is None
    assert mc.open_master(store, FakeDesks(gates={"live": (framework(), gate())}), NOW) is None, "only the swing desks"
    assert store.trades(status="open") == []


def test_the_monthly_desk_outranks_the_weekly_when_both_trade(store):
    desks = FakeDesks(gates={"weekly": (framework(), gate()), "monthly": (framework(), gate())})
    tid = mc.open_master(store, desks, NOW)
    assert store.trade(tid)["category"] == "monthly" and desks.asked[0] == "monthly"


def test_a_user_opened_card_does_not_block_the_master_and_is_never_reviewed_by_it(store):
    store.open_trade(opened_ms=NOW, asset="BTC/USDT", side=SHORT, entry=80_000.0, stop=81_000.0, given_stop=None,
                     stop_verdict="supplied", category="live", plan={})
    desks = FakeDesks(gates={"weekly": (framework(), gate())})
    assert mc.open_master(store, desks, NOW) is not None
    mc.review_masters(store, FakeDesks(price=70_000.0), NOW + 1)
    user = [t for t in store.trades(status="open") if not (t.get("plan") or {}).get("master")]
    assert len(user) == 1, "the trader's own position is theirs to kill"


# ---------- closing ----------
def _opened(store, price=80_050.0):
    desks = FakeDesks(price=price, gates={"weekly": (framework(), gate())})
    return mc.open_master(store, desks, NOW)


def test_the_master_closes_on_its_hard_invalidation(store):
    tid = _opened(store)
    stop = store.trade(tid)["stop"]
    closed = mc.review_masters(store, FakeDesks(price=stop - 50.0, gates={"weekly": (framework(), gate())}), NOW + 1)
    row = store.trade(tid)
    assert closed == [tid] and row["status"] == "closed" and "hard invalidation" in row["closed_reason"]
    assert row["closed_price"] == stop - 50.0
    assert any(e["kind"] == "master" and "closed" in e["headline"] for e in store.trade_events(tid))


def test_the_master_closes_when_its_last_target_is_reached(store):
    tid = _opened(store)
    last = max(float(t["price"]) for t in store.trade(tid)["plan"]["targets"])
    mc.review_masters(store, FakeDesks(price=last + 1.0, gates={"weekly": (framework(), gate())}), NOW + 1)
    row = store.trade(tid)
    assert row["status"] == "closed" and "targets reached" in row["closed_reason"]


def test_the_master_closes_when_its_desk_turns_the_other_way(store):
    tid = _opened(store)
    turned = FakeDesks(gates={"weekly": (framework(SHORT, zone=(80_000.0, 80_200.0), invalidation=81_000.0,
                                                   targets=(79_000.0, 78_000.0)), gate(SHORT))})
    mc.review_masters(store, turned, NOW + 1)
    row = store.trade(tid)
    assert row["status"] == "closed" and "weekly desk" in row["closed_reason"] and "other way" in row["closed_reason"]
    still = FakeDesks(gates={"weekly": (framework(), gate(verdict="WAIT"))})
    tid2 = _opened(store)
    mc.review_masters(store, still, NOW + 2)
    assert store.trade(tid2)["status"] == "open", "the desk waiting is not the desk turning: the card holds"


def test_one_pass_opens_then_holds_then_closes(store):
    r = mc.master_pass(store, FakeDesks(gates={"weekly": (framework(), gate())}), NOW)
    assert r.opened == 1 and r.closed == [] and not r.problems
    r = mc.master_pass(store, FakeDesks(gates={"weekly": (framework(), gate())}), NOW + 1)
    assert r.opened == 0 and r.closed == []
    tid = [t for t in store.trades(status="open") if t["plan"].get("master")][0]["id"]
    r = mc.master_pass(store, FakeDesks(price=78_000.0, gates={"weekly": (framework(), gate())}), NOW + 2)
    assert r.closed == [tid] and r.opened == 0, "closed this pass; a new one may open on the next"


def test_a_desk_that_cannot_be_read_is_a_problem_on_the_pass_not_a_crash(store):
    class Broken(FakeDesks):
        def gate(self, category):
            raise RuntimeError("no candles")
    r = mc.master_pass(store, Broken(), NOW)
    assert r.opened == 0 and any("no candles" in p for p in r.problems)


# ---------- the card ----------
def test_the_card_wears_the_master_chip_and_names_its_desk():
    from ui import panels
    plan = {"master": True, "desk": "weekly", "gate": {"verdict": "TRADE", "reason": "TRADE: three families agree."},
            "targets": [{"name": "TP1", "price": 81_000.0, "basis": "b", "due_ms": NOW, "reward_r": 1.0}],
            "stop_reasons": [], "notes": ["Master swing card: opened by the weekly desk's gate."]}
    row = {"id": 9, "asset": "BTC/USDT", "side": LONG, "entry": 80_000.0, "stop": 79_000.0, "status": "open",
           "plan": plan, "opened_ms": NOW}
    h = panels.trade_card_html(row, progress(LONG, 80_000.0, 79_000.0, [], 80_100.0), NOW, 80_100.0)
    assert "Master swing" in h and "Weekly" in h and "three families agree" in h
    assert "Master swing ·" not in panels.trade_card_html({**row, "plan": {**plan, "master": False}}, None, NOW, 80_100.0)


# ---------- the real reader on the recorded market ----------
def test_the_real_desk_reader_builds_the_gate_from_the_fixture_market():
    """The fakes above test the rules; this proves the worker's reader wires the real desks: state, chart,
    gamma and a gate with evidence built from core alone (no Streamlit)."""
    from core.config import CATEGORIES
    from core.data.offline import fixture_market
    from core.indicators.category import analyze_category
    m = fixture_market()
    analyses = {k: a for k, c in CATEGORIES.items() if (a := analyze_category(c, m.spot.frames, m.futures))}
    desks = mc.DeskReader(m, analyses, NOW)
    assert desks.price == m.spot.last
    f, g = desks.gate("weekly")
    assert isinstance(f, Framework) and isinstance(g, Gate) and g.verdict in ("TRADE", "WAIT", "NO SETUP")
    assert len(g.families) == 7 and g.reason
    ev = mc.evidence_for("weekly", m, analyses, desks.state("weekly"), NOW)
    assert ev.liquidity is not None and ev.rates is not None and ev.liq_levels is not None
    assert desks.frame("weekly") is not None and isinstance(desks.gamma("weekly") or {}, dict)
    assert desks.gate("monthly")[1].verdict in ("TRADE", "WAIT", "NO SETUP")
