import json

import pytest

from core.agents import context
from core.agents.prompts import SPECIALISTS, system_prompt, user_prompt
from core.config import CATEGORIES
from core.data.offline import fixture_market
from core.data.types import FuturesSnapshot
from core.indicators.category import analyze_category


@pytest.fixture(scope="module")
def market_and_analyses():
    m = fixture_market()
    analyses = {k: a for k, c in CATEGORIES.items() if (a := analyze_category(c, m.spot.frames, m.futures)) is not None}
    return m, analyses


def test_every_specialist_payload_is_json_and_scoped(market_and_analyses):
    m, analyses = market_and_analyses
    for aid in SPECIALISTS:
        p = context.payload_for(aid, m, analyses)
        json.dumps(p)
        assert p["domain"] == context.DOMAIN_OF[aid] and "categories" in p and "unavailable" in p
    assert context.payload_for("B1", m, analyses)["domain_data"]["options"]["max_pain"] > 0
    assert "liquidations" in context.payload_for("R3", m, analyses)["domain_data"]
    assert "stablecoins" in context.payload_for("B5", m, analyses)["domain_data"]


def test_unavailable_snapshot_is_omitted_and_listed(market_and_analyses):
    m, analyses = market_and_analyses
    m2 = m.model_copy(update={"futures": FuturesSnapshot.unavailable("binance,bybit", "451")})
    p = context.payload_for("B2", m2, analyses)
    assert "futures" not in p["domain_data"] and "futures" in p["unavailable"]
    assert "hyperliquid" in p["domain_data"]


def test_prompts_carry_role_text_and_schema():
    r3 = system_prompt("R3")
    assert r3.startswith("AGENT_ID: R3\n") and "SWEPT AND REJECTED" in r3 and '"conviction"' in r3
    d = system_prompt("DIRECTOR")
    assert "BULL_TRAP" in d and "probabilit" in d.lower() and "category_summaries" in d
    up = user_prompt("B1", {"a": 1, "unavailable": ["futures"]})
    assert up.startswith("DATA (JSON):") and '"a":1' in up and 'UNAVAILABLE: ["futures"]' in up
