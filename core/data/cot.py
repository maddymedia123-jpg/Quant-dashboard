"""CFTC Commitments of Traders for CME Bitcoin futures (keyless): who is positioned which way.

This is the institutional answer to the question the derivatives sub-agents keep asking - is the crowd
on one side? Funding and open interest say how crowded the market is; COT says WHO is crowded. The useful
read is the standard COT index: where the large speculators' net position sits within its own 52-week
range. Near the top is a crowded long, near the bottom a crowded short.

It is WEEKLY. Positions are as of Tuesday and released on Friday, so a report is three to ten days old
in normal operation. That is why it belongs on the Weekly and Monthly desks only, and why its date is
on screen: by the following Thursday it describes positioning from before the current weekly candle
opened. It must never feed the fast desks or the trap detector.

Found by calling it, and handled below:

* Matching the commodity name "Bitcoin" also returns BITCOIN CASH PERP STYLE - a different coin - and a
  CBOE contract that stopped reporting in 2019. The market is selected by its exact contract code.
* Contract codes get reused: 133LM1 has belonged to two different exchanges. So the code is not trusted
  alone, and the exchange name is checked as well.
* The CFTC misspells its own columns (`noncomm_postions_spread_all`, `change_in_noncomm_spead_all`).
  Only correctly spelled fields are read here, and every one is checked for presence, so a renamed or
  misspelled column surfaces as missing rather than as a silent zero.
* A CME Bitcoin contract is FIVE bitcoin. Positions are reported in contracts; presenting them as
  bitcoin would be five times wrong, so the multiplier is parsed from the report's own units string and
  no conversion is shown if it cannot be.
* Publication stops during a US government shutdown - it did in 2018-19 - so a report older than ten
  days means a release was missed, and is flagged rather than shown as this week's positioning."""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import httpx

from core.data.types import COTSnapshot

log = logging.getLogger(__name__)

URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"   # legacy futures-only report
CONTRACT_CODE = "133741"                                           # BITCOIN - CHICAGO MERCANTILE EXCHANGE
EXPECTED_EXCHANGE = "CHICAGO MERCANTILE EXCHANGE"
HEADERS = {"User-Agent": "trap-intel-dashboard/1.0"}
LOOKBACK_WEEKS = 52
MIN_WEEKS_FOR_INDEX = 26        # half a year of history before a range means anything
STALE_AFTER_DAYS = 10           # Tuesday positions, Friday release: 3-10 days old in normal operation
CROWDED_LONG, CROWDED_SHORT = 80, 20     # the conventional COT-index extremes
MAX_BYTES = 512 * 1024
FIELDS = (
    "report_date_as_yyyy_mm_dd", "market_and_exchange_names", "cftc_contract_market_code",
    "contract_units", "open_interest_all", "change_in_open_interest_all",
    "noncomm_positions_long_all", "noncomm_positions_short_all",
    "change_in_noncomm_long_all", "change_in_noncomm_short_all",
    "comm_positions_long_all", "comm_positions_short_all",
    "nonrept_positions_long_all", "nonrept_positions_short_all",
    "traders_noncomm_long_all", "traders_noncomm_short_all",
    "traders_comm_long_all", "traders_comm_short_all", "traders_tot_all",
)
REQUIRED = ("report_date_as_yyyy_mm_dd", "open_interest_all", "noncomm_positions_long_all",
            "noncomm_positions_short_all", "comm_positions_long_all", "comm_positions_short_all")


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _int(value) -> int | None:
    try:
        return int(float(str(value).replace(",", "").strip()))
    except (TypeError, ValueError):
        return None


def _date_ms(text) -> int | None:
    try:
        d = datetime.fromisoformat(str(text).replace("Z", "+00:00")[:19])
    except (TypeError, ValueError):
        return None
    return int(d.replace(tzinfo=timezone.utc).timestamp() * 1000)


def contract_size(units) -> float | None:
    """Bitcoin per contract from the report's own units string, e.g. "(5 Bitcoins)". None if it does not
    say, in which case no bitcoin figure is shown at all rather than a guessed one."""
    m = re.search(r"([\d.]+)\s*bitcoin", str(units or ""), re.IGNORECASE)
    try:
        return float(m.group(1)) if m else None
    except ValueError:
        return None


def cot_index(current: int, history: list[int]) -> float | None:
    """Where the current net position sits in its own range, 0 at the low and 100 at the high, including
    the current week so the result is bounded. None when the range is empty or the history too short."""
    window = history[:LOOKBACK_WEEKS]
    if len(window) < MIN_WEEKS_FOR_INDEX:
        return None
    lo, hi = min(window), max(window)
    if hi == lo:
        return None
    return round(100.0 * (current - lo) / (hi - lo), 1)


def parse_cot(rows, now_ms: int | None = None) -> COTSnapshot:
    now = now_ms or _now_ms()
    if not isinstance(rows, list):
        return COTSnapshot.unavailable("cftc", f"unexpected payload {type(rows).__name__}")

    # the exact contract, and the right exchange: codes have been reused across exchanges
    ours = [r for r in rows if isinstance(r, dict)
            and str(r.get("cftc_contract_market_code") or "").strip() == CONTRACT_CODE
            and EXPECTED_EXCHANGE in str(r.get("market_and_exchange_names") or "").upper()]
    if not ours:
        return COTSnapshot.unavailable("cftc", f"no rows for CME Bitcoin (contract {CONTRACT_CODE})")

    ours.sort(key=lambda r: str(r.get("report_date_as_yyyy_mm_dd") or ""), reverse=True)
    latest = ours[0]
    missing = [k for k in REQUIRED if _int(latest.get(k)) is None and k != "report_date_as_yyyy_mm_dd"]
    as_of = _date_ms(latest.get("report_date_as_yyyy_mm_dd"))
    if as_of is None:
        missing.append("report_date_as_yyyy_mm_dd")
    if missing:
        # a renamed or misspelled column surfaces here, not as a silent zero
        return COTSnapshot.unavailable("cftc", f"report is missing {', '.join(missing)}")

    def net(r, side):
        lo, sh = _int(r.get(f"{side}_positions_long_all")), _int(r.get(f"{side}_positions_short_all"))
        return None if lo is None or sh is None else lo - sh

    spec_net = net(latest, "noncomm")
    history = [n for n in (net(r, "noncomm") for r in ours) if n is not None]
    oi = _int(latest.get("open_interest_all"))

    change_long = _int(latest.get("change_in_noncomm_long_all"))
    change_short = _int(latest.get("change_in_noncomm_short_all"))
    if change_long is not None and change_short is not None:
        spec_change = change_long - change_short
    elif len(history) > 1:
        spec_change = history[0] - history[1]
    else:
        spec_change = None

    age_days = (now - as_of) / 86_400_000
    return COTSnapshot(
        source="cftc", market=str(latest.get("market_and_exchange_names") or "").strip(),
        as_of_ms=as_of, age_days=round(age_days, 1), stale=age_days > STALE_AFTER_DAYS,
        contract_btc=contract_size(latest.get("contract_units")),
        open_interest=oi, open_interest_change=_int(latest.get("change_in_open_interest_all")),
        spec_long=_int(latest.get("noncomm_positions_long_all")),
        spec_short=_int(latest.get("noncomm_positions_short_all")),
        spec_net=spec_net, spec_net_change=spec_change,
        spec_net_pct_oi=round(100.0 * spec_net / oi, 1) if oi else None,
        commercial_net=net(latest, "comm"), small_trader_net=net(latest, "nonrept"),
        traders={"speculators_long": _int(latest.get("traders_noncomm_long_all")),
                 "speculators_short": _int(latest.get("traders_noncomm_short_all")),
                 "commercials_long": _int(latest.get("traders_comm_long_all")),
                 "commercials_short": _int(latest.get("traders_comm_short_all")),
                 "total": _int(latest.get("traders_tot_all"))},
        cot_index=cot_index(spec_net, history), weeks_in_index=min(len(history), LOOKBACK_WEEKS),
    )


async def fetch_cot(client: httpx.AsyncClient, now_ms: int | None = None) -> COTSnapshot:
    params = {"cftc_contract_market_code": CONTRACT_CODE,
              "$order": "report_date_as_yyyy_mm_dd DESC",
              "$limit": str(LOOKBACK_WEEKS + 1), "$select": ",".join(FIELDS)}
    try:
        r = await client.get(URL, params=params, headers=HEADERS, follow_redirects=True)
        r.raise_for_status()
        if len(r.content) > MAX_BYTES:
            raise ValueError(f"{len(r.content) / 1024:.0f} KB for {LOOKBACK_WEEKS + 1} weekly rows")
        return parse_cot(r.json(), now_ms)
    except Exception as e:  # noqa: BLE001 - provider boundary
        log.error("CFTC COT unavailable: %s", e)
        return COTSnapshot.unavailable("cftc", f"{type(e).__name__}: {e}".strip(": "))
