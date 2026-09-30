"""Crypto headlines, keyless: the news half of the spec's macro/news engine.

Two publisher RSS feeds read directly, plus the cryptocurrency.cv aggregator for breadth - fourteen more
publishers than the direct feeds alone reach. The direct feeds are kept rather than replaced, because the
aggregator is one operator's hosted service: measuring it found `/api/news` serving three items from a
near-empty cache and every AI endpoint returning 429 from an exhausted upstream quota. RSS comes straight
from the publisher and cannot go down with one operator, so it is merged first and a story carried by both
keeps the direct copy.

Why not a conventional news API: every one in the public directories needs a key, and the two that do not
are an Indian news aggregator and a dead host. RSS needs no key, no account and no quota, and it is what
the publishers themselves maintain. Parsed with the standard library, so this adds no dependency.

Headlines are *tagged*, not judged. The topic and impact below come from keyword matching, which is a
crude instrument: it cannot tell a rumour from a confirmation, and it will mislabel a headline that uses
a loaded word in passing. That is why the tags are presented as tags on screen rather than as a verdict,
and why the desk agents get the headline text itself rather than only my label for it."""
from __future__ import annotations

import asyncio
import html
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx

from core.data.types import NewsSnapshot

log = logging.getLogger(__name__)

FEEDS = (("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss"),
         ("Cointelegraph", "https://cointelegraph.com/rss"))
HEADERS = {"User-Agent": "Mozilla/5.0 (trap-intel-dashboard)"}
MAX_PER_FEED = 30
# Feeds measure 29 KB and 51 KB; a megabyte is already an order of magnitude past anything legitimate.
MAX_FEED_BYTES = 4 * 1024 * 1024
DOCTYPE_RE = re.compile(rb"<!\s*(DOCTYPE|ENTITY)", re.IGNORECASE)

# Keyword tagging. Ordered: the first topic whose words appear wins, so the more market-moving
# categories are tested before the general ones.
TOPICS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("security", "high", ("hack", "hacked", "exploit", "stolen", "breach", "drained", "rug pull",
                          "vulnerability", "attack")),
    ("regulation", "high", ("sec ", "cftc", "lawsuit", "sues", "subpoena", "indict", "ban ", "banned",
                            "regulator", "congress", "senate", "court", "settlement", "fined")),
    ("etf", "high", ("etf", "inflow", "outflow", "blackrock", "ibit", "fidelity", "grayscale")),
    ("macro", "high", ("federal reserve", "fed ", "fomc", "powell", "cpi", "inflation", "rate cut",
                       "rate hike", "jobs report", "tariff", "treasury yield", "recession")),
    ("exchange", "medium", ("exchange", "delist", "halt", "withdrawal", "insolven", "proof of reserves",
                            "bankrupt", "outage")),
    ("flows", "medium", ("whale", "liquidation", "treasury", "holdings", "accumulat", "sell-off",
                         "selloff", "reserve")),
    ("adoption", "low", ("partnership", "launch", "integrat", "adopt", "custody", "tokeniz")),
)
ASSETS = (("BTC", ("bitcoin", "btc")), ("ETH", ("ethereum", "eth ", "ether ")),
          ("SOL", ("solana", "sol ")), ("XAU", ("gold",)))
TAG_RE = re.compile(r"<[^>]+>")


def _clean(text: str | None) -> str:
    """RSS descriptions carry markup and entities; a headline with &amp; in it is not a headline."""
    if not text:
        return ""
    return html.unescape(TAG_RE.sub(" ", text)).replace("\xa0", " ").strip()[:400]


def _published_ms(item: ET.Element) -> int | None:
    for tag in ("pubDate", "{http://purl.org/dc/elements/1.1/}date", "published", "updated"):
        raw = item.findtext(tag)
        if not raw:
            continue
        for parse in (parsedate_to_datetime, datetime.fromisoformat):
            try:
                d = parse(raw.strip())
                if d.tzinfo is None:
                    d = d.replace(tzinfo=timezone.utc)
                return int(d.timestamp() * 1000)
            except (TypeError, ValueError):
                continue
    return None


def _matches(text: str, words: tuple[str, ...]) -> bool:
    """Whole words only. Matched as bare substrings, "ban " found "Mark Cuban" and "Urban Institute",
    "ether " found "Tether", "eth " found "Elizabeth Warren", "gold" found "Goldman Sachs" and "court"
    found "courting" - so ordinary headlines were tagged regulation/high and shown as market-moving."""
    return any(re.search(rf"\b{re.escape(w.strip())}\b", text) for w in words)


def classify(title: str, summary: str = "") -> tuple[str, str, list[str]]:
    """Topic, impact and the assets named, from keywords. The title carries most of the signal, so the
    summary is only consulted for the asset names - a body that mentions Bitcoin in passing should not
    make a Solana story a Bitcoin story."""
    low = f" {title.lower()} "
    topic, impact = "general", "low"
    for name, weight, words in TOPICS:
        if _matches(low, words):
            topic, impact = name, weight
            break
    body = f"{low} {summary.lower()}"
    assets = [code for code, words in ASSETS if _matches(low, words)]
    if not assets:
        assets = [code for code, words in ASSETS if _matches(body, words)]
    return topic, impact, assets


def safe_xml(raw: str | bytes) -> ET.Element:
    """Parse remote XML without trusting the host's parser build.

    Measured on this runtime, the standard library already refuses both attacks that matter here: an
    external entity fails with "undefined entity", and expat 2.7.3 refuses an entity bomb with "limit on
    input amplification factor breached". The second of those comes from expat 2.4.1 or newer and the
    deploy host's build is not ours to choose, so the guard below is what has to hold on its own.

    It scans the entire buffer. An earlier version scanned only the first 4096 bytes, which a hostile
    feed defeats with a legal comment in the prolog: measured, a 5 KB comment put the DOCTYPE at byte
    5028, this function accepted the document, and the expanded entity reached a rendered headline. A
    feed needs no document type declaration, so refusing one outright removes the entity-expansion
    class, and the size cap bounds the rest."""
    data = raw.encode("utf-8", errors="replace") if isinstance(raw, str) else raw
    if len(data) > MAX_FEED_BYTES:
        raise ValueError(f"feed is {len(data) / 1_048_576:.1f} MB, over the {MAX_FEED_BYTES // 1_048_576} MB cap")
    # the WHOLE buffer, not a prefix: XML allows comments and processing instructions in the prolog,
    # so a hostile feed can pad past any window and still have its entities honoured. Measured: a
    # 5 KB comment put the DOCTYPE at byte 5028, the guard missed it, and the entity expanded into a
    # headline. Scanning 4 MB of bytes for this regex costs microseconds.
    if DOCTYPE_RE.search(data):
        raise ValueError("feed declares a DOCTYPE or ENTITY, which a news feed has no need of")
    return ET.fromstring(data)


def parse_feed(source: str, xml_text: str | bytes) -> list[dict]:
    """One feed's items. A feed that will not parse raises, so the caller can mark it unavailable."""
    root = safe_xml(xml_text)
    out: list[dict] = []
    for item in root.findall(".//item")[:MAX_PER_FEED] or root.findall(
            ".//{http://www.w3.org/2005/Atom}entry")[:MAX_PER_FEED]:
        title = _clean(item.findtext("title") or item.findtext("{http://www.w3.org/2005/Atom}title"))
        if not title:
            continue
        link = (item.findtext("link") or "").strip()
        if not link:
            anchor = item.find("{http://www.w3.org/2005/Atom}link")
            link = (anchor.get("href") if anchor is not None else "") or ""
        summary = _clean(item.findtext("description")
                         or item.findtext("{http://www.w3.org/2005/Atom}summary"))
        topic, impact, assets = classify(title, summary)
        out.append({"title": title, "url": link, "source": source, "published_ms": _published_ms(item),
                    "summary": summary, "topic": topic, "impact": impact, "assets": assets})
    return out


def merge(feeds: list[list[dict]]) -> list[dict]:
    """Newest first, one entry per story: the same story is carried by more than one publisher, and a
    duplicated headline would read as two independent confirmations of it."""
    seen: set[str] = set()
    merged: list[dict] = []
    for items in feeds:
        for item in items:
            key = re.sub(r"[^a-z0-9]+", "", item["title"].lower())[:60]
            if key and key in seen:
                continue
            seen.add(key)
            merged.append(item)
    merged.sort(key=lambda i: (i["published_ms"] is None, -(i["published_ms"] or 0)))
    return merged


# ---------- aggregator: cryptocurrency.cv (keyless) ----------
# Its /api/breaking endpoint, not /api/news: measured, /api/news returned three items from a near-empty
# cache (a 7 ms response) while /api/breaking returned twenty from fourteen publishers. The AI and
# sentiment endpoints are not used - they answered 429 from an exhausted upstream quota, and judgment
# here belongs to the desk agents anyway.
AGGREGATOR = ("cryptocurrency.cv", "https://cryptocurrency.cv/api/breaking")
AGGREGATOR_LIMIT = 40
# Its wider feed carries bot-generated ticker lines ("ETH Gas: ... Gwei") among the real stories. They do
# not appear on /api/breaking, but the filter costs nothing and a gas readout is not a headline.
TICKER_MARKERS = ("gas:", "gwei", "⛽")


def _is_ticker(title: str) -> bool:
    low = title.lower()
    return sum(marker in low for marker in TICKER_MARKERS) >= 2


def _published_ms_from(raw) -> int | None:
    """The aggregator dates in ISO 8601, the feeds in RFC 822. Both end up in epoch milliseconds."""
    if not raw:
        return None
    text = str(raw).strip()
    for parse in (lambda t: datetime.fromisoformat(t.replace("Z", "+00:00")), parsedate_to_datetime):
        try:
            d = parse(text)
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            return int(d.timestamp() * 1000)
        except (TypeError, ValueError):
            continue
    return None


def parse_aggregator(payload: dict) -> list[dict]:
    """The aggregator's articles, in the same shape the publisher feeds produce.

    Its own `category` is a different taxonomy from ours and its `credibility` was a flat 0.6 on every
    article sampled, so neither is carried: topic and impact come from the same `classify` the publisher
    feeds use, which keeps one vocabulary across all three sources."""
    if not isinstance(payload, dict):
        raise ValueError(f"unexpected payload {type(payload).__name__}")
    articles = payload.get("articles")
    if not isinstance(articles, list):
        raise ValueError("no articles array in the response")
    out: list[dict] = []
    for row in articles[:AGGREGATOR_LIMIT]:
        if not isinstance(row, dict):
            continue
        title = _clean(row.get("title"))
        if not title or _is_ticker(title):
            continue
        summary = _clean(row.get("description"))
        topic, impact, assets = classify(title, summary)
        out.append({"title": title, "url": str(row.get("link") or "").strip(),
                    # the publisher, not the aggregator: Decrypt is the source, cryptocurrency.cv is
                    # only the route it arrived by
                    "source": str(row.get("source") or AGGREGATOR[0]).strip() or AGGREGATOR[0],
                    "published_ms": _published_ms_from(row.get("pubDate")),
                    "summary": summary, "topic": topic, "impact": impact, "assets": assets,
                    "via": AGGREGATOR[0], "kind": str(row.get("contentType") or "").strip()})
    return out


async def _one_feed(client: httpx.AsyncClient, source: str, url: str) -> list[dict]:
    # publishers move feeds: CoinDesk 308s from the trailing-slash form, and httpx does not follow
    # redirects unless told to, which cost one publisher entirely until a live call showed it
    r = await client.get(url, headers=HEADERS, follow_redirects=True)
    r.raise_for_status()
    return parse_feed(source, r.content)


async def _aggregator(client: httpx.AsyncClient, url: str) -> list[dict]:
    r = await client.get(url, params={"limit": AGGREGATOR_LIMIT}, headers=HEADERS,
                         follow_redirects=True)
    r.raise_for_status()
    return parse_aggregator(r.json())


def _reason(e: BaseException) -> str:
    """str(ReadTimeout()) is empty - the most likely failure produced a blank reason on screen."""
    return f"{type(e).__name__}: {e}".strip(": ")


async def fetch_news(client: httpx.AsyncClient) -> NewsSnapshot:
    """Publisher feeds plus the aggregator, fetched concurrently and merged. Any one source being down is
    a degraded feed rather than a missing one, so whatever failed is named and the rest still render.

    Concurrent, not sequential: awaited one after another, this single leg could hold the page for three
    read timeouts in a row - 36 s measured - while every other provider in fetch_context caps at one."""
    legs = [(source, _one_feed(client, source, url)) for source, url in FEEDS]
    legs.append((AGGREGATOR[0], _aggregator(client, AGGREGATOR[1])))
    results = await asyncio.gather(*(coro for _, coro in legs), return_exceptions=True)

    collected: list[list[dict]] = []
    problems: list[str] = []
    ok_sources: list[str] = []
    for (name, _), outcome in zip(legs, results):
        if isinstance(outcome, BaseException):
            log.warning("news source %s unavailable: %s", name, outcome)
            problems.append(f"{name}: {_reason(outcome)}")
        elif outcome:
            collected.append(outcome)
            ok_sources.append(name)
        else:
            problems.append(f"{name}: returned no usable articles")

    if not collected:
        return NewsSnapshot.unavailable(",".join([s for s, _ in FEEDS] + [AGGREGATOR[0]]),
                                        "; ".join(problems))
    # the publisher feeds are merged first, so a story carried by both keeps the copy that came straight
    # from the publisher rather than the relayed one
    return NewsSnapshot(source=",".join(ok_sources), headlines=merge(collected),
                        error="; ".join(problems) or None)
