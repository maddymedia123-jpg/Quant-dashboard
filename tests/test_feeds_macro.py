"""The three keyless feeds: publisher RSS, gold spot, and Polymarket odds.

Each of these had a trap that only a live call revealed, and each trap has a test here:

* CoinDesk 308-redirects from the trailing-slash feed URL and httpx does not follow redirects by
  default, so one publisher silently dropped out.
* goldprice.dev accepts `symbols=` (plural) with HTTP 200 and answers in a *different currency*.
* Silver is plan-gated on the free tier and returns an error object, not a price.

The fixtures are recorded from the live feeds rather than hand-written, because a hand-made payload only
tests my idea of the payload.
"""
import json
import pathlib

import httpx
import pytest

from core.data import metals, news, prediction
from core.data.types import MetalsSnapshot, NewsSnapshot, PredictionSnapshot

FX = pathlib.Path(__file__).parent / "fixtures"
NOW = 1_759_000_000_000


def load(name):
    raw = (FX / name).read_text(encoding="utf-8")
    return json.loads(raw) if name.endswith(".json") else raw


# ---------- news ----------
def test_markup_and_entities_are_stripped_from_a_headline():
    """Real publisher titles carry neither, so the recorded fixtures cannot show that this happens."""
    feed = ('<?xml version="1.0"?><rss><channel><item>'
            '<title>Bitcoin &amp; Ether rally &lt;b&gt;hard&lt;/b&gt;</title>'
            '<link>https://example.test/a</link>'
            '<description>&lt;p&gt;Markets&amp;nbsp;moved &lt;em&gt;sharply&lt;/em&gt;.&lt;/p&gt;</description>'
            '<pubDate>Wed, 30 Sep 2026 12:00:00 +0000</pubDate>'
            '</item></channel></rss>')
    item = news.parse_feed("Test", feed)[0]
    # exact, with no `or` fallback: an assertion satisfiable two ways cannot fail one way
    assert item["title"] == "Bitcoin & Ether rally  hard"
    assert item["summary"] == "Markets moved  sharply ."
    assert "\xa0" not in item["summary"], "non-breaking spaces are normalised"


def test_a_recorded_feed_parses_into_dated_headlines():
    items = news.parse_feed("CoinDesk", load("coindesk_rss.xml"))
    assert len(items) >= 20
    first = items[0]
    assert first["title"] and first["source"] == "CoinDesk"
    assert first["url"].startswith("https://")
    assert isinstance(first["published_ms"], int) and first["published_ms"] > 1_700_000_000_000
    assert "<" not in first["title"] and "&amp;" not in first["title"], "markup and entities are cleaned"


def test_both_publishers_merge_newest_first():
    merged = news.merge([news.parse_feed("CoinDesk", load("coindesk_rss.xml")),
                         news.parse_feed("Cointelegraph", load("cointelegraph_rss.xml"))])
    assert len(merged) > 40
    assert {h["source"] for h in merged} == {"CoinDesk", "Cointelegraph"}
    dated = [h["published_ms"] for h in merged if h["published_ms"]]
    assert dated == sorted(dated, reverse=True), "newest first"


def test_the_same_story_from_two_publishers_is_one_entry():
    """A duplicated headline would read as two independent confirmations of the same story."""
    a = [{"title": "Bitcoin ETF sees record inflow", "url": "https://a", "source": "A",
          "published_ms": NOW, "summary": "", "topic": "etf", "impact": "high", "assets": ["BTC"]}]
    b = [{"title": "Bitcoin ETF Sees Record Inflow!", "url": "https://b", "source": "B",
          "published_ms": NOW - 60_000, "summary": "", "topic": "etf", "impact": "high",
          "assets": ["BTC"]}]
    assert len(news.merge([a, b])) == 1
    assert news.merge([a, b])[0]["source"] == "A", "the first publisher to carry it keeps the entry"


@pytest.mark.parametrize("title,topic,impact", [
    ("Exchange hacked for $40M in overnight exploit", "security", "high"),
    ("SEC sues major exchange over unregistered offering", "regulation", "high"),
    ("Spot Bitcoin ETF sees record inflow from BlackRock", "etf", "high"),
    ("Federal Reserve holds rates as CPI comes in hot", "macro", "high"),
    ("Major exchange halts withdrawals during outage", "exchange", "medium"),
    ("Whale accumulates 4,000 BTC as treasury holdings grow", "flows", "medium"),
    ("Payments firm launches custody integration", "adoption", "low"),
    ("Developer publishes a new wallet interface", "general", "low"),
])
def test_headlines_are_tagged_by_topic_and_impact(title, topic, impact):
    got_topic, got_impact, _ = news.classify(title)
    assert (got_topic, got_impact) == (topic, impact), title


def test_the_assets_named_in_a_headline_are_picked_up():
    _, _, assets = news.classify("Bitcoin and Ethereum rally as gold hits a record")
    assert set(assets) >= {"BTC", "ETH", "XAU"}
    _, _, none = news.classify("Regulator publishes new guidance")
    assert none == []


def test_a_body_mention_does_not_retag_the_story_when_the_title_names_an_asset():
    """A Solana story whose body mentions Bitcoin in passing is not a Bitcoin story."""
    _, _, assets = news.classify("Solana network upgrade goes live",
                                 "The upgrade follows similar work on Bitcoin and Ethereum.")
    assert assets == ["SOL"]


def test_recent_filters_by_age_impact_and_asset():
    snap = NewsSnapshot(source="t", headlines=[
        {"title": "old high", "published_ms": NOW - 86_400_000 * 3, "impact": "high", "assets": ["BTC"]},
        {"title": "fresh high", "published_ms": NOW - 600_000, "impact": "high", "assets": ["BTC"]},
        {"title": "fresh low", "published_ms": NOW - 600_000, "impact": "low", "assets": ["ETH"]},
        {"title": "undated", "published_ms": None, "impact": "high", "assets": ["BTC"]},
    ])
    recent = snap.recent(NOW, 3_600_000)
    assert [h["title"] for h in recent] == ["fresh high", "fresh low"], "undated and stale are excluded"
    assert [h["title"] for h in snap.recent(NOW, 3_600_000, min_impact="high")] == ["fresh high"]
    assert [h["title"] for h in snap.recent(NOW, 3_600_000, asset="ETH")] == ["fresh low"]


# ---------- the XML guard ----------
def test_a_feed_declaring_entities_is_refused():
    """The runtime here already refuses both attacks, but that depends on the host's expat build, which
    is not ours to choose. A news feed has no need of a document type declaration."""
    bomb = ('<?xml version="1.0"?><!DOCTYPE rss [<!ENTITY a "aaaaaaaaaa">'
            '<!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">]><rss><channel><item>'
            '<title>&b;</title></item></channel></rss>')
    with pytest.raises(ValueError, match="DOCTYPE or ENTITY"):
        news.safe_xml(bomb)
    with pytest.raises(ValueError, match="DOCTYPE or ENTITY"):
        news.safe_xml('<!DOCTYPE rss SYSTEM "http://evil/x.dtd"><rss/>')


def test_an_external_entity_cannot_read_a_local_file():
    xxe = ('<?xml version="1.0"?><!DOCTYPE t [<!ENTITY x SYSTEM "file:///etc/passwd">]>'
           '<rss><channel><item><title>&x;</title></item></channel></rss>')
    with pytest.raises(ValueError):
        news.safe_xml(xxe)


def test_an_oversized_feed_is_refused_before_parsing():
    with pytest.raises(ValueError, match="over the"):
        news.safe_xml(b"<rss>" + b"x" * (news.MAX_FEED_BYTES + 1) + b"</rss>")


def test_a_normal_feed_still_parses():
    assert news.safe_xml(load("coindesk_rss.xml")).findall(".//item")


# ---------- news over the wire ----------
def test_a_redirected_feed_is_followed(monkeypatch):
    """CoinDesk 308s from the trailing-slash URL; httpx does not follow redirects unless told to, which
    cost one publisher entirely until a live call exposed it.

    Only one feed is configured here, and only the redirect target serves a body, so a client that does
    not follow the redirect ends up with no headlines at all."""
    monkeypatch.setattr(news, "FEEDS", (("CoinDesk", "https://www.coindesk.com/feed/"),))
    seen = []

    def handler(request: httpx.Request):
        seen.append(str(request.url))
        if "cryptocurrency.cv" in str(request.url):
            return httpx.Response(200, json=load("cryptocv_breaking.json"))
        if request.url.path == "/feed/":
            return httpx.Response(308, headers={"Location": "https://www.coindesk.com/feed"})
        return httpx.Response(200, content=load("coindesk_rss.xml").encode())

    snap = _run(news.fetch_news(httpx.AsyncClient(transport=httpx.MockTransport(handler))))
    assert snap.available and len(snap.headlines) >= 20, "the redirect target was never read"
    # not seen[-1]: the aggregator is called after the feeds, so it is the last request either way
    assert "https://www.coindesk.com/feed" in seen, "the request followed through to the new URL"
    assert snap.error is None


def test_one_publisher_failing_is_a_degraded_feed_not_a_missing_one():
    def handler(request: httpx.Request):
        if "coindesk" in str(request.url):
            return httpx.Response(503)
        return httpx.Response(200, content=load("cointelegraph_rss.xml").encode())

    snap = _run(news.fetch_news(httpx.AsyncClient(transport=httpx.MockTransport(handler))))
    assert snap.available, "one publisher is still a feed"
    assert snap.headlines and {h["source"] for h in snap.headlines} == {"Cointelegraph"}
    assert snap.error and "CoinDesk" in snap.error, "and the gap is named rather than hidden"


def test_both_publishers_failing_is_unavailable():
    snap = _run(news.fetch_news(httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: httpx.Response(500)))))
    assert not snap.available and snap.headlines == []
    assert "CoinDesk" in (snap.error or "") and "Cointelegraph" in (snap.error or "")


def _run(coro):
    import asyncio
    return asyncio.run(coro)


# ---------- gold ----------
def test_the_recorded_gold_payload_parses():
    snap = metals.parse_gold(load("goldprice_xau.json"), NOW)
    assert snap.available and snap.xau_usd and snap.xau_usd > 100
    assert snap.unit == "troy_ounce" and snap.source == "goldprice.dev"


def test_gold_quoted_in_the_wrong_currency_is_refused():
    """The plural-parameter trap: `symbols=` returns HTTP 200 with XAU in AUD. A silently wrong gold
    price is worse than a missing one, because a dash cannot be traded on by mistake."""
    snap = metals.parse_gold({"symbols": [{"symbol": "XAU", "quote_currency": "AUD", "price": "5970.97"}]})
    assert not snap.available and "AUD" in (snap.error or "")
    assert snap.xau_usd is None


def test_a_plan_gated_symbol_is_reported_as_such():
    snap = metals.parse_gold({"error": "plan_gated", "tier": "free",
                              "message": "Symbol 'XAG-USD-SPOT' is not available on the 'free' tier."})
    assert not snap.available and "plan_gated" in (snap.error or "")
    assert "free" in (snap.error or "")


def test_bid_and_ask_are_optional():
    """The same endpoint returned them on one call and omitted them on the next."""
    with_quotes = metals.parse_gold({"symbols": [{"symbol": "XAU", "quote_currency": "USD",
                                                  "price": "4148.98", "bid": "4148.93", "ask": "4149.47"}]})
    assert with_quotes.bid == 4148.93 and with_quotes.ask == 4149.47
    without = metals.parse_gold({"symbols": [{"symbol": "XAU", "quote_currency": "USD",
                                              "price": "4148.98"}]})
    assert without.available and without.bid is None and without.ask is None


def test_a_stale_print_is_not_usable_even_when_priced():
    snap = metals.parse_gold({"symbols": [{"symbol": "XAU", "quote_currency": "USD", "price": "4148.98",
                                           "is_stale": True}]})
    assert snap.available and snap.xau_usd == 4148.98
    assert snap.is_stale and not snap.usable, "a stale print is not the current price"


def test_a_print_older_than_the_clock_allows_is_treated_as_stale():
    """The feed says fresh; the timestamp says otherwise. The clock wins."""
    old = NOW - metals.STALE_AFTER_MS - 60_000
    import datetime as dt
    stamp = dt.datetime.fromtimestamp(old / 1000, dt.timezone.utc).isoformat()
    snap = metals.parse_gold({"symbols": [{"symbol": "XAU", "quote_currency": "USD", "price": "4148.98",
                                           "is_stale": False, "computed_at": stamp}]}, NOW)
    assert snap.is_stale and not snap.usable


@pytest.mark.parametrize("payload,reason", [
    ({}, "no XAU row"),
    ({"symbols": []}, "no XAU row"),
    ({"symbols": [{"symbol": "XAG", "quote_currency": "USD", "price": "48"}]}, "no XAU row"),
    ({"symbols": [{"symbol": "XAU", "quote_currency": "USD", "price": "not a number"}]}, "not a number"),
    ({"symbols": [{"symbol": "XAU", "quote_currency": "USD", "price": "0"}]}, "is 0"),
    ([], "unexpected payload"),
])
def test_a_malformed_gold_payload_is_unavailable_with_a_reason(payload, reason):
    snap = metals.parse_gold(payload)
    assert not snap.available and reason in (snap.error or "")


def test_a_dead_gold_endpoint_is_unavailable_not_an_exception():
    snap = _run(metals.fetch_gold(httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: httpx.Response(503)))))
    assert not snap.available and snap.xau_usd is None


# ---------- prediction markets ----------
def test_the_recorded_odds_payload_parses_with_execution_bands():
    snap = prediction.parse_markets(load("voxodds_crypto.json"))
    assert snap.available and len(snap.markets) > 10
    assert snap.attribution and "Polymarket" in snap.attribution, "attribution is carried, not dropped"
    first = snap.markets[0]
    assert 0.0 <= first["probability"] <= 1.0
    assert first["band"] in prediction.BANDS or first["band"] == "unknown"
    assert snap.markets == sorted(snap.markets, key=lambda m: -(m["volume_24h"] or 0)), "busiest first"


def test_markets_are_reordered_by_volume_not_left_as_received():
    """The recorded payload already arrives volume-sorted, so it cannot show that this code sorts."""
    snap = prediction.parse_markets({"markets": [
        {"id": "quiet", "question": "Will BTC reach $1m?", "prices": [0.01], "volume_24h": 10.0,
         "execution": {"band": "clean"}},
        {"id": "busy", "question": "Will BTC reach $100k?", "prices": [0.4], "volume_24h": 900_000.0,
         "execution": {"band": "clean"}},
        {"id": "middling", "question": "Will BTC dip to $70k?", "prices": [0.3], "volume_24h": 5_000.0,
         "execution": {"band": "watch"}},
    ]})
    assert [m["id"] for m in snap.markets] == ["busy", "middling", "quiet"]


def test_only_clean_and_watch_markets_count_as_tradeable():
    snap = prediction.parse_markets(load("voxodds_crypto.json"))
    assert snap.tradeable(), "the fixture has some"
    assert all(m["band"] in ("clean", "watch") for m in snap.tradeable())
    assert any(m["band"] == "fragile" for m in snap.markets), "and fragile ones are kept, just not counted"


def test_markets_can_be_filtered_to_one_asset():
    snap = prediction.parse_markets(load("voxodds_crypto.json"))
    btc = snap.tradeable(asset="BTC")
    assert btc and all("BTC" in m["assets"] for m in btc)
    assert all("bitcoin" in m["question"].lower() or "btc" in m["question"].lower() for m in btc)


def test_an_unknown_execution_band_is_not_promoted_to_tradeable():
    snap = prediction.parse_markets({"markets": [
        {"id": "1", "question": "Will BTC reach $100k?", "prices": [0.4], "volume_24h": 10.0,
         "execution": {"band": "extremely-fillable-trust-me"}}]})
    assert snap.markets[0]["band"] == "unknown"
    assert snap.tradeable() == []


def test_a_market_without_a_usable_probability_is_dropped():
    snap = prediction.parse_markets({"markets": [
        {"id": "1", "question": "no prices", "prices": []},
        {"id": "2", "question": "out of range", "prices": [1.4]},
        {"id": "3", "question": "not a number", "prices": ["soon"]},
        {"id": "4", "question": "", "prices": [0.5]},
        {"id": "5", "question": "Will BTC reach $100k?", "prices": [0.42], "volume_24h": 5.0,
         "execution": {"band": "clean"}},
    ]})
    assert [m["id"] for m in snap.markets] == ["5"]


def test_a_nan_volume_does_not_become_a_number():
    snap = prediction.parse_markets({"markets": [
        {"id": "1", "question": "Will BTC reach $100k?", "prices": [0.4],
         "volume_24h": float("nan"), "execution": {"band": "clean"}}]})
    assert snap.markets[0]["volume_24h"] is None


@pytest.mark.parametrize("payload,reason", [
    ({}, "no markets array"),
    ({"markets": "lots"}, "no markets array"),
    ({"markets": []}, "no usable crypto markets"),
    ([], "unexpected payload"),
])
def test_a_malformed_odds_payload_is_unavailable_with_a_reason(payload, reason):
    snap = prediction.parse_markets(payload)
    assert not snap.available and reason in (snap.error or "")


def test_a_dead_odds_endpoint_is_unavailable_not_an_exception():
    snap = _run(prediction.fetch_predictions(httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: httpx.Response(500)))))
    assert not snap.available and snap.markets == []


def test_the_request_asks_for_the_crypto_category_only():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        return httpx.Response(200, json=load("voxodds_crypto.json"))

    _run(prediction.fetch_predictions(httpx.AsyncClient(transport=httpx.MockTransport(handler))))
    assert "category=Crypto" in seen["url"]


def test_the_gold_request_uses_the_singular_parameter():
    """`symbols=` is the trap; `symbol=` is the parameter."""
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        return httpx.Response(200, json=load("goldprice_xau.json"))

    _run(metals.fetch_gold(httpx.AsyncClient(transport=httpx.MockTransport(handler))))
    assert "symbol=XAU-USD-SPOT" in seen["url"]
    assert "symbols=" not in seen["url"], "the plural parameter answers in a different currency"


# ---------- assembled snapshot ----------
def test_a_market_snapshot_defaults_all_three_to_unavailable():
    from core.data.types import MarketSnapshot, OptionsSnapshot, SentimentSnapshot, SpotSnapshot

    from core.data.types import FuturesSnapshot
    m = MarketSnapshot(spot=SpotSnapshot(), futures=FuturesSnapshot(), options=OptionsSnapshot(),
                       sentiment=SentimentSnapshot())
    for name in ("news", "metals", "predictions"):
        assert name in m.unavailable()
    assert isinstance(m.news, NewsSnapshot) and isinstance(m.metals, MetalsSnapshot)
    assert isinstance(m.predictions, PredictionSnapshot)


# ---------- the aggregator leg ----------
def test_the_recorded_aggregator_payload_parses_into_headlines():
    items = news.parse_aggregator(load("cryptocv_breaking.json"))
    assert len(items) >= 15
    first = items[0]
    assert first["title"] and first["url"].startswith("https://")
    assert isinstance(first["published_ms"], int) and first["published_ms"] > 1_700_000_000_000
    assert first["via"] == "cryptocurrency.cv"


def test_the_aggregator_reaches_publishers_the_direct_feeds_do_not():
    """The whole reason for adding it. Two feeds read directly reach two publishers; this reaches many."""
    items = news.parse_aggregator(load("cryptocv_breaking.json"))
    publishers = {i["source"] for i in items}
    assert len(publishers) >= 8, publishers
    assert publishers - {"CoinDesk", "Cointelegraph"}, "it must add publishers, not repeat ours"


def test_a_headline_is_credited_to_its_publisher_not_to_the_aggregator():
    """Decrypt is the source; cryptocurrency.cv is only the route it arrived by. Crediting the relay
    would misreport where a story came from."""
    items = news.parse_aggregator({"articles": [
        {"title": "CFTC sends the White House new rules", "link": "https://decrypt.co/x",
         "source": "Decrypt", "pubDate": "2026-09-30T18:00:00.000Z", "contentType": "news"}]})
    assert items[0]["source"] == "Decrypt"
    assert items[0]["via"] == "cryptocurrency.cv"


def test_the_aggregator_dates_parse_from_iso_8601():
    """The feeds date in RFC 822 and the aggregator in ISO 8601; both have to land in epoch ms."""
    import datetime as dt

    items = news.parse_aggregator({"articles": [
        {"title": "A story", "link": "https://x.test", "source": "S",
         "pubDate": "2026-09-30T18:00:00.000Z"}]})
    # computed independently rather than pasted as a literal, which is how the first version got it wrong
    expected = int(dt.datetime(2026, 9, 30, 18, 0, tzinfo=dt.timezone.utc).timestamp() * 1000)
    assert items[0]["published_ms"] == expected


def test_a_bot_ticker_line_is_not_a_headline():
    """Its wider feed carries gas-price readouts among real stories."""
    items = news.parse_aggregator({"articles": [
        {"title": "\u26fd ETH Gas: 0.42 | 0.43 | 0.48 Gwei", "link": "https://etherscan.io/gastracker",
         "source": "Etherscan", "pubDate": "2026-09-30T19:01:00.000Z"},
        {"title": "Bitcoin ETFs extend win streak to nine days", "link": "https://decrypt.co/y",
         "source": "Decrypt", "pubDate": "2026-09-30T18:30:00.000Z"}]})
    assert [i["source"] for i in items] == ["Decrypt"]


def test_a_gas_story_that_is_actually_news_is_kept():
    """The filter needs two markers, so a genuine story that merely mentions gas fees survives it."""
    items = news.parse_aggregator({"articles": [
        {"title": "Ethereum gas: how the Cobalt upgrade changes fee markets", "link": "https://x.test",
         "source": "The Block", "pubDate": "2026-09-30T18:00:00.000Z"}]})
    assert len(items) == 1


def test_aggregator_headlines_are_tagged_with_our_own_vocabulary():
    """Its `category` is a different taxonomy and its `credibility` was a flat 0.6 on every article
    sampled, so neither is carried - one vocabulary across all three sources."""
    items = news.parse_aggregator({"articles": [
        {"title": "SEC sues a major exchange over unregistered offerings", "link": "https://x.test",
         "source": "The Block", "pubDate": "2026-09-30T18:00:00.000Z",
         "category": "policy", "credibility": 0.6, "reputation": 50}]})
    assert (items[0]["topic"], items[0]["impact"]) == ("regulation", "high")
    assert "credibility" not in items[0] and "reputation" not in items[0]


@pytest.mark.parametrize("payload", [{}, {"articles": "lots"}, [], "text"])
def test_a_malformed_aggregator_payload_raises_for_the_caller_to_handle(payload):
    with pytest.raises(ValueError):
        news.parse_aggregator(payload)


def test_articles_without_a_title_are_skipped():
    items = news.parse_aggregator({"articles": [
        {"title": "", "link": "https://a"}, {"link": "https://b"}, "not a dict",
        {"title": "A real story", "link": "https://c", "source": "S",
         "pubDate": "2026-09-30T18:00:00.000Z"}]})
    assert [i["title"] for i in items] == ["A real story"]


# ---------- all three merged ----------
def test_all_three_sources_merge_with_the_direct_copy_winning():
    """A story both a publisher feed and the aggregator carry should keep the copy that came straight
    from the publisher, because the relay can go down and the publisher's own name should be on it."""
    direct = [{"title": "Bitcoin ETFs extend win streak", "url": "https://coindesk.com/a",
               "source": "CoinDesk", "published_ms": 1_759_255_200_000, "summary": "",
               "topic": "etf", "impact": "high", "assets": ["BTC"]}]
    relayed = news.parse_aggregator({"articles": [
        {"title": "Bitcoin ETFs Extend Win Streak!", "link": "https://decrypt.co/a", "source": "Decrypt",
         "pubDate": "2026-09-30T18:00:00.000Z"},
        {"title": "A story only the aggregator has", "link": "https://theblock.co/b",
         "source": "The Block", "pubDate": "2026-09-30T18:05:00.000Z"}]})
    merged = news.merge([direct, relayed])
    assert len(merged) == 2, "the duplicate collapses"
    kept = next(h for h in merged if "win streak" in h["title"].lower())
    assert kept["source"] == "CoinDesk" and "via" not in kept, "the direct copy survived"


def test_the_aggregator_being_down_still_leaves_the_publisher_feeds():
    def handler(request: httpx.Request):
        if "cryptocurrency.cv" in str(request.url):
            return httpx.Response(503)
        body = ("coindesk_rss.xml" if "coindesk" in str(request.url) else "cointelegraph_rss.xml")
        return httpx.Response(200, content=load(body).encode())

    snap = _run(news.fetch_news(httpx.AsyncClient(transport=httpx.MockTransport(handler))))
    assert snap.available and len(snap.headlines) >= 40
    assert snap.error and "cryptocurrency.cv" in snap.error, "the gap is named"
    assert not any(h.get("via") for h in snap.headlines)
    assert "cryptocurrency.cv" not in snap.source, "and it is not credited as a live source"


def test_the_publisher_feeds_being_down_still_leaves_the_aggregator():
    def handler(request: httpx.Request):
        if "cryptocurrency.cv" in str(request.url):
            return httpx.Response(200, json=load("cryptocv_breaking.json"))
        return httpx.Response(503)

    snap = _run(news.fetch_news(httpx.AsyncClient(transport=httpx.MockTransport(handler))))
    assert snap.available and snap.headlines
    assert all(h.get("via") == "cryptocurrency.cv" for h in snap.headlines)
    assert snap.source == "cryptocurrency.cv"
    assert "CoinDesk" in (snap.error or "") and "Cointelegraph" in (snap.error or "")


def test_an_aggregator_that_answers_with_nothing_usable_is_reported():
    def handler(request: httpx.Request):
        if "cryptocurrency.cv" in str(request.url):
            return httpx.Response(200, json={"articles": []})     # what /api/news actually does
        return httpx.Response(200, content=load("coindesk_rss.xml").encode())

    snap = _run(news.fetch_news(httpx.AsyncClient(transport=httpx.MockTransport(handler))))
    assert snap.available and snap.headlines
    assert "no usable articles" in (snap.error or "")


def test_every_source_failing_is_unavailable():
    snap = _run(news.fetch_news(httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: httpx.Response(500)))))
    assert not snap.available and snap.headlines == []
    for name in ("CoinDesk", "Cointelegraph", "cryptocurrency.cv"):
        assert name in (snap.error or "")


def test_the_aggregator_request_asks_for_the_breaking_endpoint_with_a_limit():
    """Not /api/news: measured, that endpoint served three items from a near-empty cache."""
    seen = {}

    def handler(request: httpx.Request):
        if "cryptocurrency.cv" in str(request.url):
            seen["url"] = str(request.url)
            return httpx.Response(200, json=load("cryptocv_breaking.json"))
        return httpx.Response(200, content=load("coindesk_rss.xml").encode())

    _run(news.fetch_news(httpx.AsyncClient(transport=httpx.MockTransport(handler))))
    assert "/api/breaking" in seen["url"] and "/api/news" not in seen["url"]
    assert f"limit={news.AGGREGATOR_LIMIT}" in seen["url"]
