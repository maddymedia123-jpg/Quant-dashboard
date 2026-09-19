from core.agents.rubric import (
    ALL_ITEMS, BEARISH, BULLISH, DOMAINS, DOMAIN_BY_KEY, MAX_TEAM_POINTS,
    domain_score, missing_items, team_score, verdict,
)


def test_rubric_matches_the_client_spec_weights():
    assert [d.max_points for d in DOMAINS] == [20, 20, 20, 20, 20]
    assert MAX_TEAM_POINTS == 100
    assert [d.agent_no for d in DOMAINS] == [1, 2, 3, 4, 5]
    assert [d.key for d in DOMAINS] == ["smc", "liquidity", "mtf", "quant", "macro"]
    assert [i.weight for i in DOMAIN_BY_KEY["mtf"].items] == [8, 6, 6]
    assert [i.weight for i in DOMAIN_BY_KEY["macro"].items] == [8, 6, 6]
    assert all(len(d.checklist) == 4 for d in DOMAINS)
    assert len({i.id for i in ALL_ITEMS}) == len(ALL_ITEMS) == 18


def test_every_item_is_phrased_for_both_sides_and_differs():
    for i in ALL_ITEMS:
        assert i.question(BULLISH) and i.question(BEARISH)
        assert i.question(BULLISH) != i.question(BEARISH)


def test_scores_are_weighted_probabilities_computed_in_code():
    assert team_score({}) == 0.0
    assert team_score({i.id: 1.0 for i in ALL_ITEMS}) == 100.0
    half = team_score({i.id: 0.5 for i in ALL_ITEMS})
    assert abs(half - 50.0) < 1e-9
    smc = DOMAIN_BY_KEY["smc"]
    assert domain_score(smc, {"smc_bos": 1.0, "smc_ob": 0.5}) == 7.5


def test_out_of_range_probabilities_are_clamped_and_gaps_reported():
    assert domain_score(DOMAIN_BY_KEY["smc"], {"smc_bos": 3.0, "smc_ob": -2.0}) == 5.0
    probs = {i.id: 0.5 for i in ALL_ITEMS}
    probs.pop("macro_clear")
    assert missing_items(probs) == ["macro_clear"]
    assert missing_items({i.id: 0.1 for i in ALL_ITEMS}) == []


def test_verdict_bands_follow_the_spec():
    assert verdict(71) == "high confluence"
    assert verdict(70) == "moderate"
    assert verdict(50) == "moderate"
    assert verdict(49.9) == "neutral or conflicting"
