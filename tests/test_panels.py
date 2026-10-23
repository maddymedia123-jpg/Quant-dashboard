import re
import html
import json
import pathlib

from core.config import CATEGORIES, MACRO_EVENTS
from core.data.kraken_spot import parse_ohlc
from core.data.types import FuturesSnapshot, LiquidationsSnapshot, MarketSnapshot, OptionsSnapshot, SentimentSnapshot, SpotSnapshot
from core.indicators.category import analyze_category
from core.indicators.trade_map import map_trade
from ui import panels

FX = pathlib.Path(__file__).parent / "fixtures"


def _market():
    frames = {"15m": parse_ohlc(json.loads((FX / "kraken_ohlc_15m.json").read_text())),
              "1h": parse_ohlc(json.loads((FX / "kraken_ohlc_1h.json").read_text()))}
    return MarketSnapshot(
        spot=SpotSnapshot(source="kraken", last=float(frames["1h"]["close"].iloc[-1]), frames=frames),
        futures=FuturesSnapshot(source="binance", funding_rate=0.0006, funding_7d_mean=0.0001, open_interest=100000, open_interest_usd=8e9,
                                oi_change_24h_pct=7.0, long_short_ratio=1.7, taker_buy_sell_ratio=0.9),
        options=OptionsSnapshot(source="deribit", max_pain=80000, max_pain_expiry="2026-09-25", put_call_oi_ratio=0.6, iv_atm=55.0, iv_skew=3.0),
        sentiment=SentimentSnapshot.unavailable("alternative.me", "timeout"),
        liquidations=LiquidationsSnapshot(source="coinlobster", total_usd=2.04e8, long_usd=1.54e8, short_usd=5.0e7),
    )


def test_kpis_cover_required_fields_with_dash_for_missing():
    m = _market()
    a = analyze_category(CATEGORIES["live"], m.spot.frames, m.futures)
    labels = [k["label"] for k in panels.kpis_for(a, m)]
    for req in ("Price", "Direction now", "RSI 14", "ATR %", "Funding", "OI 24h", "Long/Short", "Liq 24h", "Fear & Greed"):
        assert req in labels
    liq = next(k for k in panels.kpis_for(a, m) if k["label"] == "Liq 24h")
    assert liq["value"] == "$204M" and liq["delta"] == "75% longs" and liq["tone"] == "down"
    fg = next(k for k in panels.kpis_for(a, m) if k["label"] == "Fear & Greed")
    assert fg["value"] == "—"


def test_panels_render_strings_and_squeeze_banner():
    m = _market()
    a = analyze_category(CATEGORIES["weekly"], {**m.spot.frames, "4h": m.spot.frames["1h"], "1d": m.spot.frames["1h"]}, m.futures)
    assert "ti-card" in panels.direction_html(a)
    lay = panels.layman_html(a)
    assert "What's happening:" in lay and "Levels that matter:" in lay and "What changes the view:" in lay
    assert "2σ" not in lay and "EMA" not in lay and "%" not in lay, "the jargon stays in the detail cards"
    assert "2σ band, 20-bar" in panels.volatility_html(a)
    assert "RSI" in panels.divergence_html(a)
    assert a.vol.horizon_label in panels.volatility_html(a)
    assert "Run Analysis" in panels.agent_placeholder_html(a)
    banner = panels.squeeze_banner_html(a)
    assert banner and "SQUEEZE" in banner


def test_trade_result_macro_and_status():
    m = _market()
    live = analyze_category(CATEGORIES["live"], m.spot.frames, m.futures)
    tm = map_trade({"dir": "LONG", "entry": live.price, "lev": 10.0, "sl": live.price * 0.98, "tp": live.price * 1.04}, live, None, m.futures)
    h = panels.trade_result_html(tm)
    assert tm.classification in h and "Liquidation" in h
    assert html.escape(MACRO_EVENTS[0]["title"]) in panels.macro_html(MACRO_EVENTS)
    status = panels.data_status_html(m)
    assert "unavailable" in status and "kraken" in status
    rows = panels.raw_metrics_rows(m)
    assert any(r[0] == "Max pain" and "80,000" in r[1] for r in rows)
    assert any(r[0] == "Liquidations 24h" and "$204.0M" in r[1] for r in rows)
    assert "hyperliquid" in status and "stablecoins" in status


def test_agent_summary_and_report_stats():
    from datetime import datetime, timezone
    from core.agents.schemas import AgentRun, DirectorReport, Scenario, TradePlan, TrapReport
    m = _market()
    a = analyze_category(CATEGORIES["live"], m.spot.frames, m.futures)
    assert "Run Analysis" in panels.agent_summary_html(a, None)
    d = DirectorReport(executive_summary="e", trap_classification="BULL_TRAP", classification_rationale="r",
                       scenarios=[Scenario(name="a", probability=60, path="p", invalidation="i"), Scenario(name="b", probability=40, path="p", invalidation="i")],
                       trade_plan=TradePlan(existing_position_management="h", new_entry_conditions="n", position_sizing="s", stop_loss="l", max_leverage="3x"),
                       category_summaries={"live": "Live paragraph <b>"})
    r = TrapReport(report_id="TIR-1", generated_at=datetime.now(timezone.utc), window="w", provider="gemini", director=d,
                   runs=[AgentRun(agent_id="B1", ok=True, model="m", latency_s=1, prompt_tokens=10, completion_tokens=5),
                         AgentRun(agent_id="B2", ok=False, model="m", latency_s=1, error="x")])
    h = panels.agent_summary_html(a, r)
    assert "Live paragraph &lt;b&gt;" in h and "Director verdict: bull trap" in h and "Head-agent summary" in h
    st = panels.report_stats_html(r)
    assert "agents 1/2 ok" in st and "tokens 15" in st and "failed: B2" in st and "free tier" in st


def test_anchored_direction_early_warning_and_accuracy_panels():
    from core.accuracy import HitRate
    from core.anchors import AnchorContext, AnchoredVerdict
    from core.indicators.early_warning import EarlyWarning
    m = _market()
    a = analyze_category(CATEGORIES["live"], m.spot.frames, m.futures)
    an = AnchorContext(direction="BULLISH", confidence=0.6, anchor=78000.0, invalidation=77800.0, price=78400.0, ts_ms=0,
                       funding_sign=1, oi_change_24h_pct=1.0, long_short_ratio=1.1, squeeze_score=20, classification=None,
                       macro_within_24h=False, sigma_pct=0.5, drivers=["EMA stack bullish"], flips_if="close below 77,800")
    live = AnchorContext(**{**an.__dict__, "direction": "BEARISH", "confidence": 0.3})
    v = AnchoredVerdict("live", an, live, since_ms=0, triggers_fired=[], changed=False)
    h = panels.anchored_direction_html(a, v, now_ms=90 * 60_000)
    assert "anchored 1h 30m ago" in h and "live read BEARISH" in h and "ti-card up" in h
    v2 = AnchoredVerdict("live", live, live, since_ms=0, triggers_fired=["funding_flip"], changed=True)
    assert "Re-anchored now on: funding_flip" in panels.anchored_direction_html(a, v2, now_ms=0)
    assert panels.early_warning_html([]) is None
    ew = panels.early_warning_html([EarlyWarning("BULL_TRAP_FORMING", "forming", ["pressing resistance"])])
    assert "BULL TRAP FORMING" in ew and "ti-card warn" in ew
    empty = panels.accuracy_html({"total_scored": 0, "open_calls": 3})
    assert "No scored calls yet" in empty and "waiting: 3" in empty
    full = panels.accuracy_html({"total_scored": 3, "open_calls": 0, "by_category": {"live": HitRate(3, 2)},
                                 "by_classification_24h": {"BULL_TRAP": HitRate(1, 1)}, "by_classification_7d": {},
                                 "recent_calls": [{"category": "live", "direction": "BULLISH", "price": 78000.0, "realised_pct": 0.4, "hit": 1}],
                                 "recent_reports": []})
    assert "67%" in full and "small sample" in full and "BULL_TRAP" in full and "hit)" in full


def test_calendar_panel_live_and_fallback():
    import json, pathlib
    from core.data.calendar import parse_calendar
    m = _market()
    assert "Fallback list" in panels.calendar_html(m, now_ms=0)  # calendar unavailable → curated
    cal = parse_calendar(json.loads((pathlib.Path(__file__).parent / "fixtures" / "ff_calendar_thisweek.json").read_text(encoding="utf-8")))
    m2 = m.model_copy(update={"calendar": cal})
    first_high = next(e for e in cal.events if e["impact"] == "High" and e["country"] == "USD")
    h = panels.calendar_html(m2, now_ms=first_high["time_ms"] - 3_600_000)
    assert "Inside 24h" in h and first_high["title"] in h and "Forex Factory" in h and "ti-card warn" in h


def test_calendar_panel_escapes_feed_text():
    from core.data.types import CalendarSnapshot
    m = _market()
    evil = CalendarSnapshot(source="forexfactory", events=[{"title": "<img src=x onerror=alert(1)>", "country": "USD", "impact": "High",
                                                              "time_ms": 10_000_000, "date": "x", "forecast": "<b>1</b>", "previous": None}])
    h = panels.calendar_html(m.model_copy(update={"calendar": evil}), now_ms=10_000_000 - 60_000)
    assert "<img" not in h and "&lt;img" in h and "<b>1</b>" not in h and "&lt;b&gt;1&lt;/b&gt;" in h


def test_fib_and_pattern_cards_render_and_escape():
    from core.indicators.patterns import Pattern
    from core.indicators.reversal import ReversalWindow
    m = _market()
    a = analyze_category(CATEGORIES["live"], m.spot.frames, m.futures)
    h = panels.fib_html(a)
    assert "Fibonacci time · drawing only" in h and "count from the last short swing" in h
    assert "retracement of the larger" in h
    a.reversal = ReversalWindow("ACTIVE", "bearish", 5, a.fib.upcoming[0].timestamp_ms, 0, 2, ["at resistance 78,600 (4 touches)"])
    h = panels.fib_html(a)
    for call in ("Window open", "Reversal window open", "Next reversal window", "Confluence building"):
        assert call not in h, "tested and found no edge: never worded as a timing call"
    assert "ti-card warn" not in h, "tested and found no edge: not highlighted as a warning"
    assert "did not time turns" in h and "17.3%" in h
    a.patterns = [Pattern("<b>Double top</b>", "bearish", "forming", 0.8, 77000.0, 75000.0, 79000.0, [], 0, note="second top being tested now")]
    p = panels.patterns_html(a)
    assert "&lt;b&gt;Double top" in p
    # each level bound to its own label, so a swapped target and invalidation cannot pass
    for label, value in (("Breakout", "$77,000"), ("Target", "$75,000"), ("Invalidation", "$79,000")):
        assert f"<span class='ti-label'>{label}</span><span class='ti-kv-v '>{value}</span>" in p, label
    lay = panels.layman_html(a)
    assert "Also worth noting:" in lay and "&lt;b&gt;double top&lt;/b&gt; is forming, bearish" in lay
    assert "reversal window" not in lay, "Fibonacci timing is no longer told to the reader"
    a.patterns = []
    assert "No clean pattern" in panels.patterns_html(a)


# ---------- Live Recon war room panels ----------
def _recon(bull_p=0.8, bear_p=0.3, trap=None):
    from datetime import datetime, timezone

    from core.agents.judge import ChoiceResult, JudgeResult
    from core.agents.live_recon import LiveReconResult, resolve, score_team
    from core.agents.rubric import ALL_ITEMS, BEARISH, BULLISH

    trap = trap or ChoiceResult(choice="BULL_TRAP", probabilities={"BULL_TRAP": 0.7, "NO_TRAP": 0.3},
                                provider="typesafe")
    bull = score_team(BULLISH, JudgeResult(probabilities={i.id: bull_p for i in ALL_ITEMS},
                                          provider="typesafe", latency_s=1.4))
    bear = score_team(BEARISH, JudgeResult(probabilities={i.id: bear_p for i in ALL_ITEMS}, provider="typesafe"))
    bias, margin, conf, consulted, override, notes = resolve(bull, bear, trap)
    return LiveReconResult(generated_at=datetime(2026, 9, 19, 13, 5, tzinfo=timezone.utc), bull=bull, bear=bear,
                           trap=trap, bias=bias, margin=margin, confidence=conf, consulted=consulted,
                           override=override, notes=notes)


def test_team_scorecard_shows_five_domain_agents_and_their_points():
    res = _recon()
    h = panels.team_html(res.bull)
    assert "80<span> / 100</span>" in h and "high confluence" in h and "typesafe" in h and "1.4s" in h
    for n, title in ((1, "Quant &amp; Statistics"), (2, "Auction Market &amp; Volume Profile"),
                     (3, "Order Flow &amp; Delta"), (4, "ICT &amp; Liquidity"),
                     (5, "On-Chain &amp; Derivatives")):
        assert f"{n}. {title}" in h
    assert "16.0 / 20" in h and "ti-card up" in h
    assert "ti-card down" in panels.team_html(res.bear)


def test_a_failed_team_says_it_failed_rather_than_showing_zero():
    from core.agents.judge import JudgeResult
    from core.agents.live_recon import score_team

    h = panels.team_html(score_team("bullish", JudgeResult(probabilities={}, error="503 upstream")))
    assert "Did not score: 503 upstream" in h and "failure, not a reading of zero" in h
    assert "0 / 100" not in h


def _prob(h: str, label: str) -> str | None:
    """The percentage printed on the same row as a trap label."""
    m = re.search(rf"<span class='l'>{re.escape(label)}</span>.*?<span class='v'>([^<]*)</span>", h)
    return m.group(1) if m else None


def test_war_room_panel_shows_the_trap_distribution_and_any_override():
    h = panels.war_room_html(_recon())
    # each number bound to its own row: asserting the labels and the percentages separately passed
    # just as happily with the distribution rendered backwards
    assert _prob(h, "bull trap") == "70%", "70% belongs to the bull trap, not merely present"
    assert _prob(h, "no trap") == "30%"
    assert h.index("<span class='l'>bull trap<") < h.index("<span class='l'>no trap<"), "most likely first"
    assert "ti-card warn" in h
    assert "override" in h and "overrides the long bias" in h
    assert "Not required" in h  # a 50-point margin needs no consult


def test_the_war_room_distribution_cannot_render_backwards():
    from core.agents.judge import ChoiceResult

    inverted = panels.war_room_html(_recon(trap=ChoiceResult(
        choice="NO_TRAP", probabilities={"BULL_TRAP": 0.3, "NO_TRAP": 0.7}, provider="typesafe")))
    assert _prob(inverted, "no trap") == "70%"
    assert _prob(inverted, "bull trap") == "30%"
    assert inverted.index("<span class='l'>no trap<") < inverted.index("<span class='l'>bull trap<")

    from core.agents.judge import ChoiceResult
    silent = panels.war_room_html(_recon(trap=ChoiceResult(error="timeout")))
    assert "No answer: timeout" in silent and "No trap override was applied" in silent


def test_anchored_summary_panel_states_how_long_it_stays_pinned():
    from core import live_anchor

    res = _recon()
    payload = live_anchor.anchor_payload(res, price=78_000.0)
    now = 1_726_750_000_000
    h = panels.anchored_summary_html(payload, now + 95 * 60_000, now, anchored_now=True)
    assert "BULL TRAP RISK" in h and "margin +50" in h
    # each total bound to its own row, so the two teams cannot be swapped
    assert re.search(r"<span>Bullish</span>.*?<span class='n'>80</span>", h)
    assert re.search(r"<span>Bearish</span>.*?<span class='n'>30</span>", h) and "bull trap" in h
    assert "$78,000" in h and "pinned for 1h 35m more" in h and "Published by this run." in h
    held = panels.anchored_summary_html(payload, now + 60_000, now, anchored_now=False)
    assert "never rewrite this summary" in held


def test_side_note_panel_lists_appended_notes_and_says_when_there_are_none():
    assert "Nothing substantial has changed" in panels.side_notes_html([])
    rows = [{"kind": "BIAS_FLIP", "headline": "Bias has changed from BULL to BEAR since the anchored summary.",
             "detail": "Bullish team 20/100, bearish team 90/100.", "price": 76_500.0}]
    h = panels.side_notes_html(rows)
    assert "Interim side notes · 1" in h and "bias flip" in h and "changed from BULL to BEAR" in h
    assert "$76,500" in h and "bearish team 90/100" in h


# ---------- SMC and liquidity panels ----------
def _evidence():
    import os
    os.environ["TI_OFFLINE_FIXTURES"] = "1"
    from core.agents.live_recon import build_state
    from core.config import CATEGORIES
    from core.data.offline import fixture_market
    from core.indicators.category import analyze_category

    m = fixture_market()
    a = {k: an for k, c in CATEGORIES.items() if (an := analyze_category(c, m.spot.frames, m.futures))}
    st = build_state(m, a)
    return st["smc"], st["liquidity"]


def test_smc_panel_names_the_structure_event_and_the_nearest_zones():
    smc_by_tf, _ = _evidence()
    h = panels.smc_html(smc_by_tf)
    assert "Market structure (SMC)" in h
    for tf in ("15m", "1h", "4h"):
        assert f"<td>{tf}</td>" in h
    assert "CHOCH" in h or "BOS" in h
    assert "premium" in h or "discount" in h or "equilibrium" in h


def test_liquidity_panel_labels_delta_as_a_proxy_and_shows_pools():
    _, liq_by_tf = _evidence()
    h = panels.liquidity_html(liq_by_tf)
    assert "Liquidity &amp; order flow" in h
    assert "proxy" in h, "delta must never be presented as real tape"
    assert "touches" in h and ("swept" in h or "intact" in h)


def test_the_panels_say_unavailable_instead_of_inventing_numbers():
    blank = {tf: {"available": False, "note": "no candles"} for tf in ("15m", "1h", "4h")}
    assert "—" in panels.smc_html(blank) or "unavailable" in panels.smc_html(blank).lower()
    assert "—" in panels.liquidity_html(blank) or "unavailable" in panels.liquidity_html(blank).lower()


# ---------- accuracy report ----------
def test_accuracy_panel_says_it_is_collecting_before_ten_windows():
    from core.live_accuracy import AccuracyReport

    h = panels.accuracy_report_html(AccuracyReport("live", "collecting", 3, "3 of 10 4h windows scored"))
    assert "Accuracy report" in h and "3 of 10" in h and "ti-card warn" not in h


def test_accuracy_panel_shows_the_team_grades_when_healthy():
    from core.live_accuracy import AccuracyReport

    rep = AccuracyReport("live", "healthy", 12, "No calibration needed.", 0.75, 0.6, 0.12, 0.18)
    h = panels.accuracy_report_html(rep)
    assert "75%" in h and "0.12" in h and "0.18" in h and "No calibration needed" in h
    assert "coin flip" in h, "the Brier scale must be explained, not just printed"


def test_accuracy_panel_lists_each_protocol_finding_with_its_fix():
    from core.live_accuracy import AccuracyReport, Issue

    rep = AccuracyReport("live", "calibration needed", 12, "Direction right in 25% of 12 windows", 0.25, 0.0,
                         0.41, 0.2, [Issue("misread liquidity", "The bullish team gave <b>sweeps</b> 95%",
                                           "Count a sweep only after a reclaim close.", 12, "bullish",
                                           "liq_sweep", 0.95, 0.0)])
    h = panels.accuracy_report_html(rep)
    assert "Accuracy report \u00b7 calibration needed" in h and "ti-card warn" in h
    assert "misread liquidity" in h and "reclaim close" in h and "12 windows" in h
    assert "&lt;b&gt;sweeps" in h, "findings quote model-facing text, so they are escaped"


# ---------- volatility matrix panel ----------
def test_volatility_matrix_panel_shows_each_timeframe_with_its_memory():
    from core.data.types import OptionsSnapshot
    from core.indicators.volatility import matrix_summary

    m = _market()
    by_tf = {tf: matrix_summary(m.spot.frames[tf], tf, m.options) for tf in ("15m", "1h")}
    h = panels.vol_matrix_html(by_tf)
    assert "Volatility matrix" in h
    for tf in ("15m", "1h"):
        assert f"<td>{tf}</td>" in h
    assert "Hurst" in h and ("trending" in h or "random walk" in h or "mean-reverting" in h)
    assert "%" in h and "1&sigma;" in h and "3&sigma;" in h, "the sigma channel must be spelled out"

    blind = {"1h": matrix_summary(m.spot.frames["1h"], "1h", OptionsSnapshot.unavailable("deribit", "x"))}
    out = panels.vol_matrix_html(blind)
    assert "—" in out, "implied vol renders as a dash, never as a guess"


def test_volatility_matrix_panel_explains_what_hurst_means():
    from core.indicators.volatility import matrix_summary

    m = _market()
    h = panels.vol_matrix_html({"1h": matrix_summary(m.spot.frames["1h"], "1h", m.options)})
    assert "0.5" in h and ("mean-revert" in h.lower() or "trend" in h.lower())


def test_volatility_matrix_row_spans_the_whole_table_when_a_timeframe_is_thin():
    h = panels.vol_matrix_html({"1w": {"available": False, "note": "no candles"}})
    header_cells = h.count("<th>")
    assert f"colspan='{header_cells - 1}'" in h, f"row must fill all {header_cells} columns"


def test_a_missing_number_renders_as_a_dash_not_as_a_dash_with_a_unit():
    partial = {"1h": {"available": True, "realized_vol_pct": None, "implied_vol_pct": None,
                      "atr_pct": None, "hurst": None, "memory": "unknown", "regime": "UNKNOWN",
                      "bands": {}, "note": ""}}
    h = panels.vol_matrix_html(partial)
    assert "—%" not in h, "a dash carrying a percent sign reads as a measured zero"


# ---------- Director report screen ----------
def _report(**over):
    from datetime import datetime, timezone

    from core.agents.schemas import Contradiction, DirectorReport, Scenario, TradePlan, TrapReport
    d = DirectorReport(
        executive_summary="Value holds <b>between</b> the edges.", trap_classification="RANGE_TRAP",
        classification_rationale="Neither desk sees acceptance.",
        contradictions=[Contradiction(metric="funding", bull_read="shorts pay", bear_read="crowded",
                                      stronger_evidence="bear")],
        convergence=["value between 111,420 and 112,900"], blind_spots=[],
        scenarios=[Scenario(name="Upside break", probability=30, path="p1", invalidation="i1"),
                   Scenario(name="Range holds", probability=45, path="p2", invalidation="i2"),
                   Scenario(name="Downside sweep", probability=25, path="p3", invalidation="i3")],
        trade_plan=TradePlan(existing_position_management="hold", new_entry_conditions="No longs into 113,420.",
                             position_sizing="half", stop_loss="113,980", max_leverage="3x",
                             take_profit_ladder=["112,150"]),
        category_summaries={"live": "Live paragraph", "weekly": "Weekly paragraph"}, **over)
    return TrapReport(report_id="TIR-9", generated_at=datetime.now(timezone.utc), window="w", provider="p",
                      director=d)


def test_the_director_card_leads_with_the_classification_and_orders_the_scenarios():
    h = panels.director_html(_report())
    assert "Range trap" in h and "Value holds &lt;b&gt;between" in h and "Report TIR-9" in h
    assert h.index("Range holds") < h.index("Upside break") < h.index("Downside sweep"), "most likely first"
    assert "width:45%" in h, "the bar is drawn to the probability"
    assert "No longs into 113,420." in h and "113,980" in h and "3x" in h and "112,150" in h


def test_the_lists_say_none_rather_than_leaving_a_blank_column():
    h = panels.director_lists_html(_report())
    assert "funding: the bulls read shorts pay; the bears read crowded. Stronger: bear" in h
    assert "value between 111,420 and 112,900" in h
    assert "Blind spots" in h and "None reported." in h


def test_each_desk_paragraph_is_labelled_with_its_desk_and_timeframe():
    h = panels.director_desks_html(_report(), {"live": "Live Recon", "weekly": "Weekly"}, {"live": "15m", "weekly": "4h"})
    assert "Live Recon <span class='muted'>15m</span>" in h and "Weekly paragraph" in h


def test_a_report_without_a_director_says_so():
    from datetime import datetime, timezone

    from core.agents.schemas import TrapReport
    r = TrapReport(report_id="TIR-0", generated_at=datetime.now(timezone.utc), window="w", provider="p")
    assert "did not answer" in panels.director_html(r) and panels.director_lists_html(r) == ""


def test_the_raw_metric_table_and_the_audit_escape_their_strings():
    h = panels.raw_metrics_html([("Spot <b>", "$1", "kraken <i>")])
    assert "Spot &lt;b&gt;" in h and "kraken &lt;i&gt;" in h and "<b>" not in h and "<i>" not in h
    a = panels.anchor_audit_html([{"ts_ms": 0, "category": "<x>", "from_dir": None, "to_dir": "BULLISH",
                                   "price": 1.0, "triggers": ["<y>"]}])
    assert "&lt;x&gt;" in a and "&lt;y&gt;" in a and "— → <strong>BULLISH" in a
    assert "No anchor changes" in panels.anchor_audit_html([])


def test_the_pinned_teams_are_rebuilt_from_the_anchor_on_their_own_side():
    from core import live_anchor

    payload = live_anchor.anchor_payload(_recon(bull_p=0.8, bear_p=0.3), price=78_000.0)
    bull, bear = panels.payload_team_html("bullish", payload), panels.payload_team_html("bearish", payload)
    assert "80<span> / 100</span>" in bull and "16.0 / 20" in bull and "ti-card up" in bull
    assert "30<span> / 100</span>" in bear and "6.0 / 20" in bear and "ti-card down" in bear
    assert "1. Quant &amp; Statistics" in bull and "5. On-Chain &amp; Derivatives" in bear
    assert "did not score" in panels.payload_team_html("bearish", {**payload, "domains": {"bullish": {}}})


def test_the_rail_reads_take_the_signal_colours_when_they_are_on():
    from ui.theme import palette
    mono = panels.rail_style_html(["up", "down", "neutral"], ["15m", "1h", "4h"], False, False)
    sig = panels.rail_style_html(["up", "down", "neutral"], ["15m", "1h", "4h"], False, True)
    p, q = palette(False), palette(False, signal=True)
    assert f"> :nth-child(1) [data-testid=\"stCaptionContainer\"] p {{ color: {q['up']}" in sig
    assert f"> :nth-child(2) [data-testid=\"stCaptionContainer\"] p {{ color: {q['down']}" in sig
    assert f"> :nth-child(1) [data-testid=\"stCaptionContainer\"] p {{ color: {p['up']}" in mono
    assert f"> :nth-child(3) [data-testid=\"stCaptionContainer\"] p {{ color: {p['muted']}" in sig, "no call, no colour"
    assert 'content: "1h"' in sig, "the timeframe sits beside the desk's name"


def test_the_open_interest_column_follows_the_desks_own_timeframes():
    row = {"available": True, "pools": {}, "cumulative_delta": {"available": True, "state": "rising"},
           "volume_profile": {"available": False},
           "open_interest": {"1d": 1.5, "1w": -2.0, "1M": None}}
    h = panels.liquidity_html({"1d": row})
    assert "<th>OI 1d/1w/1M</th>" in h and "+1.5% / -2.0% / —" in h, "not dashes for 15m/1h/4h"


def test_the_number_strip_reads_the_desks_own_window():
    m = _market()
    a = analyze_category(CATEGORIES["live"], m.spot.frames, m.futures)
    w = {"open_interest": [{"over": "1h", "change_pct": 0.1}, {"over": "4h", "change_pct": -1.2, "now_btc": 58_400}],
         "liquidations": {"over": "4h", "longs_usd": 3e6, "shorts_usd": 1e6},
         "taker_delta": {"over": "4h", "delta_btc": -38.0},
         "max_pain": {"expiry": "2026-10-24", "price": 82_000.0}}
    k = {x["label"]: x for x in panels.kpis_for(a, m, w)}
    assert k["OI 4h"]["value"] == "-1.2%" and "Liq 4h" in k and k["Liq 4h"]["delta"] == "75% longs"
    assert k["Taker delta 4h"]["value"] == "-38 BTC" and k["Max pain"]["value"] == "$82,000"
    assert not {"OI 24h", "Liq 24h", "Taker delta 24h"} & set(k), "no 24h block on a 4h desk"
    assert "OI 24h" in {x["label"] for x in panels.kpis_for(a, m)}, "without a window it says 24h, honestly"
