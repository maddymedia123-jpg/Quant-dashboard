"""Crypto headlines from publisher RSS (keyless): the news half of the spec's macro/news engine.

Why RSS rather than a news API: every news API in the public directories needs a key, and the two that
do not are an Indian news aggregator and a dead host. RSS needs no key, no account and no quota, and it
is what the publishers themselves maintain. Parsed with the standard library, so this adds no dependency.

Headlines are *tagged*, not judged. The topic and impact below come from keyword matching, which is a
crude instrument: it cannot tell a rumour from a confirmation, and it will mislabel a headline that uses
a loaded word in passing. That is why the tags are presented as tags on screen rather than as a verdict,
and why the desk agents get the headline text itself rather than only my label for it."""
from __future__ import annotations

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


def classify(title: str, summary: str = "") -> tuple[str, str, list[str]]:
    """Topic, impact and the assets named, from keywords. The title carries most of the signal, so the
    summary is only consulted for the asset names - a body that mentions Bitcoin in passing should not
    make a Solana story a Bitcoin story."""
    low = f" {title.lower()} "
    topic, impact = "general", "low"
    for name, weight, words in TOPICS:
        if any(w in low for w in words):
            topic, impact = name, weight
            break
    body = f"{low} {summary.lower()}"
    assets = [code for code, words in ASSETS if any(w in low for w in words)]
    if not assets:
        assets = [code for code, words in ASSETS if any(w in body for w in words)]
    return topic, impact, assets


def safe_xml(raw: str | bytes) -> ET.Element:
    """Parse remote XML without trusting the host's parser build.

    Measured on this runtime, the standard library already refuses both attacks that matter here: an
    external entity fails with "undefined entity", and expat 2.7.3 refuses an entity bomb with "limit on
    input amplification factor breached". But that second guarantee comes from expat 2.4.1 or newer, and
    the deploy host's build is not ours to choose - so the guard below does not depend on it. A feed needs
    no document type declaration, so refusing one removes the whole entity-expansion class outright, and
    the size cap bounds the rest. Cheaper and more verifiable than taking on a dependency for it."""
    data = raw.encode("utf-8", errors="replace") if isinstance(raw, str) else raw
    if len(data) > MAX_FEED_BYTES:
        raise ValueError(f"feed is {len(data) / 1_048_576:.1f} MB, over the {MAX_FEED_BYTES // 1_048_576} MB cap")
    if DOCTYPE_RE.search(data[:4096]):
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


async def fetch_news(client: httpx.AsyncClient) -> NewsSnapshot:
    """Both feeds. One publisher being down is a degraded feed, not a missing one, so it is named."""
    collected: list[list[dict]] = []
    problems: list[str] = []
    for source, url in FEEDS:
        try:
            # publishers move feeds: CoinDesk 308s from the trailing-slash form, and httpx does not
            # follow redirects unless told to, which cost one publisher until a live call showed it
            r = await client.get(url, headers=HEADERS, follow_redirects=True)
            r.raise_for_status()
            collected.append(parse_feed(source, r.content))
        except Exception as e:  # noqa: BLE001 - provider boundary, per feed
            log.warning("news feed %s unavailable: %s", source, e)
            problems.append(f"{source}: {e}")
    if not collected:
        return NewsSnapshot.unavailable(",".join(s for s, _ in FEEDS), "; ".join(problems))
    return NewsSnapshot(source=",".join(s for s, _ in FEEDS), headlines=merge(collected),
                        error="; ".join(problems) or None)
