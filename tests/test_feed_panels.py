"""The macro/news panels: headlines, gold and prediction-market odds, as HTML.

These three had no test of any kind. Escaping was correct, but nothing defended it — stripping every
`html.escape` from all three left the whole suite green. They render strings from three third-party
feeds (headline titles, summaries, publisher names, URLs, Polymarket question text, execution-band
labels and reasons, the attribution line) straight into `st.markdown(..., unsafe_allow_html=True)`, so
a hostile or compromised feed controls those strings.
"""
import pytest

from core.data.types import MetalsSnapshot, NewsSnapshot, PredictionSnapshot
from ui import panels

NOW = 1_760_000_000_000
HOUR = 3_600_000
XSS = "<script>alert(1)</script>"


def headline(**over) -> dict:
    base = {"title": "Bitcoin ETFs extend win streak to nine days", "url": "https://decrypt.co/a",
            "source": "Decrypt", "published_ms": NOW - 30 * 60_000, "summary": "Inflows continued.",
            "topic": "etf", "impact": "high", "assets": ["BTC"], "via": "cryptocurrency.cv"}
    base.update(over)
    return base


def news(headlines=None, **over) -> NewsSnapshot:
    return NewsSnapshot(source=over.pop("source", "CoinDesk,Cointelegraph"),
                        headlines=headlines if headlines is not None else [headline()], **over)


def market(**over) -> dict:
    base = {"id": "1", "question": "Will Bitcoin reach $100,000 by December 31?", "probability": 0.275,
            "outcome": "Yes", "volume_24h": 92_631.0, "liquidity": 40_000.0, "end_ms": NOW + 40 * 24 * HOUR,
            "days_left": 40.0, "uncertainty": 8.0, "band": "clean", "band_label": "Clean",
            "band_score": 81.0, "flags": [], "band_reason": "", "url": "https://polymarket.com/event/x",
            "assets": ["BTC"]}
    base.update(over)
    return base


def preds(markets=None, **over) -> PredictionSnapshot:
    return PredictionSnapshot(source="voxodds",
                              markets=markets if markets is not None else [market()],
                              attribution=over.pop("attribution",
                                                   "Data from Polymarket. Powered by VoxOdds."),
                              retrieved_ms=NOW - 60_000, **over)


# ---------- news ----------
def test_the_news_panel_lists_headlines_with_their_age_topic_and_publisher():
    h = panels.news_html(news(), NOW)
    assert "1 headlines" in h and "CoinDesk,Cointelegraph" in h
    assert "Bitcoin ETFs extend win streak to nine days" in h
    assert "30m ago" in h and "etf" in h and "Decrypt" in h and "BTC" in h


def test_the_high_impact_count_is_shown_and_tones_the_card():
    quiet = panels.news_html(news([headline(impact="low", topic="general")]), NOW)
    assert "high-impact" not in quiet and "ti-card neutral" in quiet
    loud = panels.news_html(news([headline(), headline(title="SEC sues an exchange", impact="high")]), NOW)
    assert "2 high-impact" in loud and "ti-card down" in loud


def test_the_panel_says_the_tags_are_keyword_tags_not_a_verdict():
    assert "keyword tags" in panels.news_html(news(), NOW)
    assert "not as a read on the market" in panels.news_html(news(), NOW)


def test_a_partial_feed_says_which_publisher_is_missing():
    h = panels.news_html(news(error="CoinDesk: ReadTimeout"), NOW)
    assert "Partial feed" in h and "CoinDesk: ReadTimeout" in h
    assert "not everything published" in h


def test_an_unavailable_feed_says_so_rather_than_showing_an_empty_table():
    h = panels.news_html(NewsSnapshot.unavailable("rss", "every publisher refused"), NOW)
    assert "No headline feed" in h and "every publisher refused" in h
    assert "<table>" not in h


def test_an_undated_headline_says_undated_rather_than_now():
    assert "undated" in panels.news_html(news([headline(published_ms=None)]), NOW)


def test_the_headline_limit_is_honoured():
    many = [headline(title=f"Story number {i}", url=f"https://x.test/{i}") for i in range(40)]
    h = panels.news_html(news(many), NOW, limit=5)
    assert h.count("<tr><td") == 5, "the header row is a <tr> too"
    assert "40 headlines" in h, "the count is of everything received, not of what is shown"


# ---------- gold ----------
def test_the_gold_panel_shows_the_price_and_its_unit():
    h = panels.gold_html(MetalsSnapshot(source="goldprice.dev", xau_usd=4148.55, unit="troy_ounce",
                                        computed_ms=NOW - 5 * 60_000), NOW)
    assert "$4,148.55" in h and "troy ounce" in h
    assert "Computed 5m ago" in h and "goldprice.dev" in h


def test_a_bid_and_ask_are_shown_when_the_feed_sends_them():
    with_quotes = panels.gold_html(MetalsSnapshot(source="s", xau_usd=4148.98, bid=4148.93, ask=4149.47),
                                   NOW)
    assert "bid $4,148.93" in with_quotes and "ask $4,149.47" in with_quotes
    assert "bid" not in panels.gold_html(MetalsSnapshot(source="s", xau_usd=4148.98), NOW)


def test_a_stale_gold_print_is_flagged_and_not_presented_as_the_price():
    h = panels.gold_html(MetalsSnapshot(source="s", xau_usd=4148.55, is_stale=True), NOW)
    assert "stale" in h and "is not the current price" in h
    assert "ti-card warn" in h, "and the card is toned to match"


def test_the_gold_panel_says_only_gold_is_available():
    assert "silver and copper are gated" in panels.gold_html(
        MetalsSnapshot(source="s", xau_usd=4148.55), NOW)


def test_an_unavailable_gold_feed_says_why():
    h = panels.gold_html(MetalsSnapshot.unavailable("goldprice.dev", "plan_gated: free tier"), NOW)
    assert "No gold feed" in h and "plan_gated" in h
    assert "$" not in h.split("No gold feed")[1][:40], "no price is implied"


# ---------- prediction markets ----------
def test_the_odds_panel_shows_the_probability_band_and_volume():
    h = panels.predictions_html(preds(), NOW)
    assert "27.5%" in h and "clean" in h and "$92,631" in h and "40d" in h
    assert "Will Bitcoin reach $100,000 by December 31?" in h
    assert "1 with a quote worth quoting" in h


def test_a_fragile_market_is_called_out_with_its_reason():
    h = panels.predictions_html(preds([market(
        band="fragile", band_label="Fragile", band_score=44.4, flags=["extreme price"],
        band_reason="Quote can move or fill badly; verify depth before any sizing.")]), NOW)
    assert "fragile" in h and "0 with a quote worth quoting" in h
    assert "1 of these are fragile" in h
    assert "Quote can move or fill badly" in h
    assert "a printed number, not a forecast" in h


def test_the_attribution_the_feed_asks_for_is_rendered():
    assert "Data from Polymarket. Powered by VoxOdds." in panels.predictions_html(preds(), NOW)


def test_markets_can_be_filtered_to_one_asset():
    both = preds([market(), market(id="2", question="Will Ethereum reach $3,000?", assets=["ETH"])])
    h = panels.predictions_html(both, NOW, asset="BTC")
    assert "1 market(s)" in h and "mentioning BTC" in h
    assert "Ethereum" not in h


def test_an_unavailable_odds_feed_says_why():
    h = panels.predictions_html(PredictionSnapshot.unavailable("voxodds", "503 upstream"), NOW)
    assert "No odds feed" in h and "503 upstream" in h and "<table>" not in h


def test_an_unknown_band_is_not_counted_as_quotable():
    h = panels.predictions_html(preds([market(band="unknown")]), NOW)
    assert "0 with a quote worth quoting" in h and "unknown" in h


# ---------- escaping: the reason this file exists ----------
@pytest.mark.parametrize("field", ["title", "source", "topic", "summary"])
def test_every_feed_controlled_headline_field_is_escaped(field):
    h = panels.news_html(news([headline(**{field: XSS})]), NOW)
    assert "<script>" not in h, f"a feed's {field} reached the page as markup"


def test_a_headline_asset_tag_is_escaped():
    assert "<script>" not in panels.news_html(news([headline(assets=[XSS])]), NOW)


@pytest.mark.parametrize("bad_url", [
    "javascript:alert(1)",
    "JaVaScRiPt:alert(1)",
    "data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==",
    "vbscript:msgbox(1)",
    "  javascript:alert(1)",
])
def test_a_non_http_headline_url_is_never_turned_into_a_link(bad_url):
    h = panels.news_html(news([headline(url=bad_url)]), NOW)
    assert "javascript:" not in h.lower() and "vbscript:" not in h.lower()
    assert "data:text/html" not in h.lower()
    assert "<a href" not in h, "an unusable scheme is rendered as text, not as a link"


def test_a_headline_url_cannot_break_out_of_its_attribute():
    h = panels.news_html(news([headline(url="https://x.test/a' onmouseover='alert(1)")]), NOW)
    assert "onmouseover" not in h or "&#x27;" in h
    assert "onmouseover='alert" not in h
    h2 = panels.news_html(news([headline(url='https://x.test/a" onfocus="alert(1)')]), NOW)
    assert 'onfocus="alert' not in h2


@pytest.mark.parametrize("field", ["question", "band", "band_label", "band_reason"])
def test_every_feed_controlled_market_field_is_escaped(field):
    h = panels.predictions_html(preds([market(**{field: XSS, "band_reason": XSS})]), NOW)
    assert "<script>" not in h, f"a market's {field} reached the page as markup"


def test_a_market_url_that_is_not_https_is_not_linked():
    h = panels.predictions_html(preds([market(url="javascript:alert(1)")]), NOW)
    assert "javascript:" not in h.lower() and "<a href" not in h


def test_the_attribution_line_is_escaped():
    assert "<script>" not in panels.predictions_html(preds(attribution=XSS), NOW)


@pytest.mark.parametrize("field", ["unit", "source"])
def test_the_gold_panel_escapes_its_feed_strings(field):
    fields = {"source": "goldprice.dev", "xau_usd": 4148.55, field: XSS}
    assert "<script>" not in panels.gold_html(MetalsSnapshot(**fields), NOW)


def test_an_unavailable_feed_escapes_its_error_text():
    """The error string carries provider text, which is feed-controlled too."""
    for fn, snap in ((panels.news_html, NewsSnapshot.unavailable("rss", XSS)),
                     (panels.gold_html, MetalsSnapshot.unavailable("gold", XSS)),
                     (panels.predictions_html, PredictionSnapshot.unavailable("voxodds", XSS))):
        assert "<script>" not in fn(snap, NOW), fn.__name__


def test_a_partial_feed_error_is_escaped():
    assert "<script>" not in panels.news_html(news(error=XSS), NOW)


# ---------- robustness ----------
def test_the_panels_render_with_nothing_in_them():
    assert panels.news_html(news([]), NOW)
    assert panels.predictions_html(preds([]), NOW)


def test_the_panels_survive_missing_fields():
    assert panels.news_html(news([{"title": "Only a title"}]), NOW)
    assert panels.predictions_html(preds([{"question": "Only a question"}]), NOW)
    assert panels.gold_html(MetalsSnapshot(source="s"), NOW)
