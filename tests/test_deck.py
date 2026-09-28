"""The Institutional Quantitative Matrix: what each of the five sub-agents actually read.

Built from the same state handed to the judges, so what the trader sees and what the agents scored can
never drift apart. A reading with no feed behind it is None, never a placeholder number."""
import pytest

from core.agents.deck import sub_agent_deck
from core.agents.live_recon import build_state
from core.agents.recon_profiles import PROFILES
from core.agents.rubric import DOMAINS
from core.config import CATEGORIES
from core.data.offline import fixture_market
from core.data.types import FuturesSnapshot, OptionsSnapshot
from core.indicators.category import analyze_category


@pytest.fixture(scope="module")
def market_and_analyses():
    m = fixture_market()
    return m, {k: a for k, c in CATEGORIES.items() if (a := analyze_category(c, m.spot.frames, m.futures))}


def deck_for(m, analyses, profile=PROFILES["live"]):
    return sub_agent_deck(build_state(m, analyses, profile=profile), profile)


def test_the_deck_has_the_five_sub_agents_the_spec_names(market_and_analyses):
    m, analyses = market_and_analyses
    deck = deck_for(m, analyses)
    assert [a.agent_no for a in deck] == [1, 2, 3, 4, 5]
    assert [a.key for a in deck] == [d.key for d in DOMAINS]
    assert [a.title for a in deck] == [d.title for d in DOMAINS]
    assert all(len(a.rows) >= 3 for a in deck)


def test_every_reading_comes_from_the_state_and_is_filled_on_live_data(market_and_analyses):
    m, analyses = market_and_analyses
    by_key = {a.key: a for a in deck_for(m, analyses)}

    quant = dict((label, value) for label, value, _ in by_key["quant"].rows)
    assert quant["Hurst"] is not None and quant["Regime"] and quant["Realized vol"] > 0

    auction = dict((label, value) for label, value, _ in by_key["auction"].rows)
    assert auction["Point of control"] > 0 and auction["Value area high"] > auction["Value area low"]

    delta = dict((label, value) for label, value, _ in by_key["delta"].rows)
    assert delta["Delta read"] in ("confirming up", "confirming down", "bearish absorption",
                                   "bullish absorption", "flat")
    assert delta["Taker buy/sell"] is not None

    ict = dict((label, value) for label, value, _ in by_key["ict"].rows)
    assert ict["Structure"] in ("bullish", "bearish", "unclear")
    assert ict["Range position"] in ("premium", "discount", "equilibrium")

    derivs = dict((label, value) for label, value, _ in by_key["derivs"].rows)
    assert derivs["Funding"] is not None and derivs["Long/short ratio"] is not None
    assert derivs["Options max pain"] > 0


def test_a_missing_feed_leaves_the_reading_empty_and_says_so(market_and_analyses):
    m, analyses = market_and_analyses
    blind = m.model_copy(update={"futures": FuturesSnapshot.unavailable("binance,bybit,gate", "451"),
                                 "options": OptionsSnapshot.unavailable("deribit", "timeout")})
    by_key = {a.key: a for a in deck_for(blind, analyses)}
    derivs = dict((label, value) for label, value, _ in by_key["derivs"].rows)
    assert derivs["Funding"] is None and derivs["Options max pain"] is None
    assert "futures" in by_key["derivs"].note and "options" in by_key["derivs"].note


def test_each_reading_declares_how_it_should_be_shown(market_and_analyses):
    m, analyses = market_and_analyses
    for agent in deck_for(m, analyses):
        for label, _value, kind in agent.rows:
            assert label and kind in ("pct", "rate", "usd", "num", "text"), (label, kind)


def test_the_deck_follows_the_category_it_is_asked_for(market_and_analyses):
    m, analyses = market_and_analyses
    live = deck_for(m, analyses, PROFILES["live"])
    weekly = deck_for(m, analyses, PROFILES["weekly"])
    assert live[0].timeframe == "1h" and weekly[0].timeframe == "1d", "the desk's middle timeframe"
    assert any("4h" in a.note or a.timeframe for a in weekly)
