# Quant Dashboard — System Handbook

*What exists, how it fits together, what is still open. Written for Hammad. Kept in the repo at
`docs/handbook.md` and updated with every merge; last revised 5 November 2026.*

Everything below is in the code on `main` and live unless a line says otherwise. Nothing here is planned
work dressed up as done.

---

## 1. What runs where

| Piece | What it does | Where |
| --- | --- | --- |
| **Dashboard** | The Streamlit terminal: nine screens, TradingView-style charts, the war rooms, the reports, the Quant Forecast | Streamlit Cloud, redeploys on every merge to `main` |
| **24/7 worker** | One pass a minute, whether or not a page is open: fetches the market, builds the four desks, anchors and scores, settles traps, sends every alert and the Market Pulse. Its only model calls are the scheduled agent runs (fix 6, 5 Nov): the Director and the Live war room after a High-impact print settles and after a day-plan invalidation, six a day at most. | Google Cloud VM, `quant-worker.service` |
| **Order-book recorder** | One snapshot of the Gate.io BTC perp book every 5 seconds, kept 35 days locally | Same VM, `quant-book.service` (recording since 28 Oct 2026) |
| **MCP server** | Read-only access for AI assistants (Claude, Gemini, ChatGPT, Cursor): 14 tools over the shared store | Same VM, behind HTTPS, token in the address |
| **Shared store** | Turso (hosted SQLite): every anchor, call, score, trap, trade, alert, report, plan and book report. The dashboard and the worker read and write the same database. Turso meters every page an embedded replica pulls back (3 GB a month on the free plan). Since 9 Nov the worker records a desk's squeeze score only when it moves 3 points or more, re-fires or an hour has passed, leaves an unchanged cascade state alone, rewrites the order-book report every 5 minutes, and the dashboard pulls at most once per 30 s: a page a minute or less instead of 22. | Turso, US-East |
| **Discord** | Where every alert lands, through one webhook | Hammad's server, channel of his choice |
| **Deploy** | A merge to `main` reaches Streamlit Cloud and the VM within two minutes, no hands | GitHub → Streamlit Cloud; VM timer pulls, installs, restarts |

Time on screen is UTC first with Pakistan time beside it. Every candle boundary, window, score and stored
timestamp is UTC, because exchange candles are UTC and the ledger scores at exact candle closes.

## 2. Data in

All feeds are keyless public APIs. Every feed carries an availability flag; a missing value shows as a
dash, never a number. Whichever provider answered is named in the sidebar.

| Data | Source | Used for |
| --- | --- | --- |
| BTC candles | Gate.io BTC_USDT perpetual (Kraken, then Gemini exchange spot as fallback) | Every chart, indicator, desk and the Quant Forecast |
| Funding, open interest, long/short, taker buy and sell volume | Binance → Bybit → Gate.io, then CoinLobster + Hyperliquid composite, Kraken Futures, finally our own stored samples | Derivatives domain, cascade, CVD, triggers. Binance and Bybit refuse US servers, so Gate.io answers in production |
| Real taker delta and CVD | Gate.io contract stats per timeframe | Order-flow domain, magnitude scaler, cautions |
| Liquidations, whale trades, unusual flow | CoinLobster | Derivatives domain |
| Options: max pain, put/call, IV skew, gamma by expiry, call and put walls | Deribit | Gamma regime bar, per-desk gamma, Monday plan, Quant Forecast levels |
| Stablecoin supply | DefiLlama | Derivatives domain |
| Fear & Greed | alternative.me | Sentiment |
| Economic calendar (high-impact USD prints) | Forex Factory, the current week's file (Sunday to Saturday; no next-week file is published), cached 30 minutes | Event window on every desk, event regime (election / FOMC), macro window trigger, Quant Forecast chop windows, Pulse (High and Medium prints) |
| Headlines | CoinDesk and Cointelegraph RSS plus the cryptocurrency.cv aggregator (14 publishers) | Macro & News screen, narratives in the agents' context, Pulse |
| Gold | goldprice.dev | Macro & News |
| Prediction markets | Polymarket via VoxOdds, with execution-quality bands | Macro & News, Fed odds in the Pulse |
| US yields and the funds rate | FRED, falling back to the US Treasury daily curve for the yields and the New York Fed's own JSON for the funds rate (FRED refuses Streamlit Cloud) | Macro & News, agents' macro read. FRED's weekly broad dollar index shows when FRED answers; the daily index is the reconstructed one below |
| BTC dominance (roadmap item 10, 5 Nov) | CoinGecko `/global`; the share's 24-hour and 7-day change from the dashboard's own samples, one per worker pass | Macro & News, agents' macro read |
| USD/JPY and the yen carry (roadmap item 10, 5 Nov) | ECB reference rates via frankfurter.dev; the Bank of Japan policy rate is hand-set (`BOJ_POLICY_RATE_PCT`), never guessed | Macro & News, agents' macro read |
| Dollar index, reconstructed (roadmap item 11, 5 Nov) | ICE's DXY formula on the ECB's six daily per-dollar rates from the same frankfurter.dev request; a day behind the real index, within a few tenths | Macro & News, agents' macro read |
| CFTC Commitments of Traders | CFTC, weekly | Weekly and Monthly desks only |
| Exchange net flows | CoinMetrics community (all flows, not whales) | Weekly and Monthly desks |
| Order book | Our own recorder on the VM | Scoop and sweep report, order_book tool |

A provider that answers 403 or 451 is parked for six hours and logged once, so a geo-block does not spam
the log every minute.

## 3. The four desks

| Desk | Reads | Charts | Anchored to | Scored over |
| --- | --- | --- | --- | --- |
| Live Recon | 15m / 1h / 4h | 4h view by default (call made on the 1h; 1h view optional), next 4 hours | the UTC 4h candle | 4 hours |
| Intraday | 1h / 4h / 1D, plus a 12H brief | 4h (or a 1D view), next 24 hours | the UTC day | 24 hours |
| Weekly | 4h / 1D / 1W, plus 3D and 0.5W briefs | 1D (or a 3D view), next 7 days | Monday 00:00 UTC | 7 days |
| Monthly | 1D / 1W / 1M, plus a 2W brief | 1W (or a 2W view), next month | the calendar month | 30 days |

Each desk screen carries: the hero and candle countdown, the chart with the anchored call, the plain-English
read (what is happening, levels that matter, what changes the view), the KPI strip for its own window, the
turning-window summary, the protocol cards (Volatility, Structure, Liquidity, Quant matrix), the forecast
pane, and the war room.

**Deterministic layer, computed from candles, no model:** EMAs 21/50/100/200, RSI with divergence,
reaction-counted support and resistance, trendlines, two-point Fibonacci with retracements, chart
patterns, harmonics (Gartley, Bat, Butterfly, Crab, Cypher; valid only beside VWAP, value or gamma),
volatility matrix (realised, implied, ATR regime, sigma channels, Hurst), Smart Money Concepts (structure,
break versus change of character, fair value gaps, order blocks, premium/discount), liquidity (equal
highs and lows, sweep versus break, volume profile, value area, real CVD), VWAP qualified break (1H/4H/1D),
the timeframe cascade (states A strong setup, B pause, C trend losing control, N none), per-desk
derivatives (OI, funding, liquidations, max pain and gamma over the desk's own window), kill zones (NY
13–17 UTC measured at 1.8× normal turn rate; Asian and London no better than chance), and the forecast
pane (median of 4,000 simulated paths, 68% band, measured to hold 72–74% on 1h to 1d).

**Desk summary and the confluence gate (his v22 Section 9, live 4 November).** Every desk ends with the
same summary in his layout: Big Picture; Timeline (what is happening, what to expect, the turning window in
UTC and PKT); the bounce zone with its plain reason; Targets 1, 2 and 3 as the highs (or lows) to expect;
the safety line; the verified confluences; the directive; the warning line. The zone is the nearest level
the bias acts from, a quarter ATR either side. The trade is cancelled by a 4-hour close beyond the safety
line on Live Recon and Intraday, and by the desk's own candle (daily, weekly) on Weekly and Monthly. The
trade box opens only when at least three of seven independent families agree with the zone; fewer is WAIT
naming what is missing; two families pointing the other way is WAIT naming the conflict. The seven, each
read from its own feed so no single read can open the gate: a harmonic completing in the zone; a put wall
(for a long) or call wall (for a short) in the zone; order flow on the desk's chart (absorption at the
level, or flow confirming); positioning read the contrarian way (extreme fear under 25 or greed over 75,
under 45% or over 60% of accounts long, a negative or a high premium to hold: the crowded side is the one
that gets forced out, so extreme fear plus a crowded short is a bullish trigger); structure (a failed break
is the trap it is, a swept-and-reclaimed pool, a value-area edge in the zone); macro (the broad dollar
±0.25% on the week, the 2-year yield ±0.10 points, which side the last news print trapped); and the measured
turning window, counting only while it is on. A missing feed is a family that says nothing, never one that
agrees. The warning names the next high-impact US print inside the event window (five days ahead, on every
desk, with the time to it), a dollar move against the trade, and the weekend.

**The higher-timeframe bar (live 4 November).** Over the gate: a Weekly or Monthly level within half an ATR
beyond the zone, on the path to Target 1, bars the trade even when three families agree (WAIT, naming the
level). A level inside the zone is the entry, not a wall, so it never bars (your note). A level farther
out but still before Target 1 is named in the warning as a pause, not a bar. Live Recon and Intraday read
the Weekly and Monthly desks' levels, Weekly reads Monthly's, Monthly has none above it.

**Named anomaly traps (your Section 5.3, live 4 November).** Positioning read against the candle, on every
desk. Extreme fear (under 25) plus a crowded short (45% or fewer accounts long, or shorts paying to hold)
is the **Bullish squeeze trap**. Extreme greed (over 75) plus over-leveraged longs (60% or more accounts
long or a high premium to hold, with open positions growing over the desk's window or at a 30-day high) is
the **Liquidation flush trap**. Each is SET when the order flow on the desk's chart already disagrees with
the crowd (buyers absorbing a falling tape, or sellers meeting every bid on a rising one) and FORMING when
the crowd is there but the flow has not turned. Contrarian by design: the trapped crowd is the one that
gets forced out, so a squeeze trap leans long and a flush trap leans short. It shows as a card on the desk,
as its own line under the Big Picture in the summary, speaks for the positioning family in the confluence
gate, and goes to the desk's judges and Head in their state by name. Greed without leverage building is
greed, not a flush, and is not named.

**4 November, election day: what went wrong and the fixes (owner's GO, 5 November).** BTC ran from 78,336
to 81,194 between 14:00 and 15:00 UTC on 4 November, the day after the US congressional elections, and the
bot was useless through it. Four causes, in order of damage. (1) The 00:05 UTC report's short plan was
cancelled by the plan tracker at 08:00 (price traded through the stop before any confirmation), but the
report ledger had no "cancelled" outcome, left the row open, and the Pulse sold from the dead plan in nine
messages from 09:00 to 17:00. (2) The head agents run only when a person clicks Run Analysis; the worker
makes no model calls, so nothing with judgment looked at the result for 14 hours. (3) The calendar carried
"Congressional Elections" as a High-impact print inside the five-day window, but no rule elevated it: the
Quant Forecast still said NEUTRAL, RANGE-BOUND, 80% confidence, and was invalidated at 16:00. (4) The
cascade said "downtrend losing control, watch for a reversal" three times, softly, and the Intraday "strong
setup up" arrived after the first big candle. The fixes, each merged live in this order:

- *Fix 1, live 5 November.* The report scorer now applies the plan tracker's two cancellation rules (through
  the stop before a confirming close; the confirming candle closes beyond the far edge of the zone) and
  records "cancelled". The Pulse speaks from a plan only while it is open, so a cancelled plan hands the
  Pulse back to the Live desk the same minute. The accuracy card and the MCP tools show the new outcome.
- *Fix 2, live 5 November.* Live Recon's chart opens on the 4-hour candle, as your table reads ("Main Chart
  View: 4-Hour Chart"); the 1h, which its call is made on, is the other view.
- *Fix 3, event regime, live 5 November.* Your Section 6.2 ("The Head Agent queries the calendar for
  high-impact events (FOMC, Elections). If active within the trade window, volatility parameters widen and
  trap warnings elevate"). A major print is a High-impact USD calendar entry that names an election, the
  FOMC, the federal funds rate or the Fed chair; its regime runs from 24 hours before the print to 48 hours
  after it (an election's result is traded the day after). While it is open: the Quant Forecast builds no
  range plan (the lean is scored 0, every hour's expected range is 1.5× wider, the whole-day invalidation
  sits at the 5th percentile on both sides instead of the 10th, and the regime line, the directive and every
  caution, hold and invalidation message name the event); the Pulse's event section leads with it; every
  desk summary's Watch line leads with it and its directive ends "Event regime (…): wider stop, smaller
  size, no range trade until …"; and the war rooms and the Director see `event_regime` in the state they
  judge. CPI, NFP and the other prints are unchanged: a chop window around the release, not a regime.
  *Carried into the Director's plan, live 5 November.* On 4 November at 21:26 UTC a report's short went
  out while the Pulse's regime line said "no range plan, smaller size": the regime governed only the Quant
  Forecast. Now a sided plan the Director makes while a regime is active is tagged with the event
  (`plan.event_regime`), its `risk_pct` and `max_leverage` are halved in code (the Director is told to give
  its normal size, and to place the stop beyond the next structural level so an event sweep of the day's
  edge does not stop it out), the plan card, the WATCH alert and the Pulse say the plan was made inside the
  regime with its halved size, and the Pulse's regime line says whose rule "no range plan" is: the Quant
  Forecast's. A plan made before the regime began carries no tag; the Pulse line still says the rule is
  the day plan's.
**Report review, 5 November (TIR1-20261105-153911, owner's FIX IT).** The report built its bear-trap case
on "$124.5M of long liquidations", the all-coins figure (BTC's was $43.1M); set its stop at 75,350 under a
primary scenario that died on a close below 75,700; called targets 79,448 and 80,219 "inside the 24-hour
range" when the 24-hour expected high was 78,752 (the code check said tactical because it keeps the
widest edge, the prior day's high); cited a Fibonacci reversal window and the spec's 13:00–17:00 session
instead of the measured 12:00–16:00 window; and relabelled a +70 bp liquidity contraction an expansion.
The download left out the framing, the verdict, the risk and leverage and any regime tag. Fixed: the plan
check names targets beyond the 24-hour expected range even when the verdict stays tactical
(`targets_beyond_expected_range`); code reads the first price in the primary scenario's invalidation and
names a stop placed beyond it (`director.plan_notes`, on the card and in the download, both pipelines);
the download carries the framing, the check, risk and leverage, the regime tag and the notes; the raw
table labels the all-coins liquidation row; and a discipline block in both Directors' rules (and two
shared rules for every agent) fixes the liquidation rows, the timing hours, the stop-versus-invalidation
rule, macro labels as given, and bans the reversal window as evidence.
- *Fix 4, the Pulse's event section names the scheduled prints, live 5 November.* In order: the event
  regime when one is open; the High-impact US prints of the last 24 hours with the macro engine's read of
  the reaction (sell-the-news, spike faded, confirmed, absorbed, or "the reaction is still being traded");
  the US prints of the next 24 hours, High and Medium (Medium labelled "(medium impact)": on 4 November
  ISM Services was Medium and went unnamed); then the headlines and the Fed odds as before.
- *Fix 5, cascade reversal alerts: a confirmation rule, firmer words, no flapping, live 5 November.* The
  "losing control" warning now names the level: how many times the engine candles have closed beyond the
  day's average price (the anchored VWAP, in plain words), the price itself, and what confirms it ("the 4h
  closing above $77,512; until then this is a warning, not a trade"). A new message, REVERSAL CONFIRMED
  (urgent), goes out once per episode when a completed anchor candle (Live 4h, Intraday 1D) that closed
  after the shift began closed beyond that average price against the bigger trend: "Shorts are off: the
  daily downtrend has lost control. Expect upside expansion. Wrong only on a 4h close back below $77,512."
  Replayed on 4 November it fires at 08:00 UTC at 78,140, six hours before the spike. The Live repeat
  window is 8 hours (was 3; the four repeats that day were 3 to 4½ hours apart), Intraday 12 (was 6),
  and a confirmed reversal silences the warning for the same window.
- *Fix 6, scheduled agent runs, live 5 November.* The worker runs the 1-agent Director and the Live war
  room on its own, twice over: 15 minutes after a High-impact USD print (the first 15-minute candle of the
  reaction has closed) and when a Quant Forecast day plan is invalidated. Each trigger runs once (its key is
  a row in the alert ledger, written before the run, so two processes cannot both run it), never within 30
  minutes of the last run, and at most six times a day; a throttled trigger is recorded as skipped, not
  retried, and a print or invalidation older than two hours (a reboot) is not re-run. A run that produced a
  report sends no line of its own: the report's own Market Pulse and plan alerts follow as they do for a
  report run by hand. A run the model refused sends "🤖 NO NEW REPORT · BTC $… / Triggered by …, but the
  model was unavailable. The last plan (report …) stays in force. The next automatic run comes with the next
  release or plan change, not before …" (wording batch 5, 6 Nov). Nothing runs on fixture data or
  without a model key.

**Roadmap item 10, four macro feeds (your Section 10; three of the four live 5 November).** Each is a
card on the Macro & News screen and a field in every agent's macro read:

- *2-year minus the funds rate.* The effective federal funds rate (EFFR) joins the US macro card as a
  fourth series, from FRED with the New York Fed's own JSON as the fallback. The card and the agents carry
  Δ2Y = DGS2 − EFFR on a date both series share, read by your rule: above +25 bp the 2-year prices
  tighter money ahead (liquidity contraction, a headwind for bitcoin); below −25 bp it prices cuts
  (liquidity expansion, a tailwind); between them, no signal. The recorded 7 October window reads
  +89 bp, contraction.
- *BTC dominance.* Bitcoin's share of the whole crypto market from CoinGecko, with the ETH and stablecoin
  shares and the market's 24-hour change. The feed has no history, so the worker stores a sample every
  pass (one per ten minutes at most, a month kept) and the card reads the share's 24-hour change (the
  7-day change when the day has no sample yet) from the dashboard's own samples: a rise of 0.5 pp in a
  day is capital rotating into bitcoin, a fall of 0.5 pp is rotation out to the alternatives ("risk
  appetite stretching" when the whole market is up). The change starts empty and fills in over the
  first day and week after the merge; until then the card says "no history yet".
- *Yen carry.* USD/JPY from the ECB's daily fix, with the week and month change. A fall of 1.5% or more
  in a week of sessions is the carry-unwind warning (August 2024: USD/JPY fell about 12% in four weeks
  and bitcoin lost a quarter in days). The Bank of Japan's policy rate has no free feed worth trusting
  and changes only at scheduled meetings, so it is hand-set: `BOJ_POLICY_RATE_PCT=0.75` (for example) in
  the worker's `.env` on the VM and as a top-level Streamlit secret. Until it is set the card says "not
  set" and the carry gap (US funds rate minus BoJ) is left out rather than guessed; the read then rests
  on the pace of USD/JPY alone.
- *US election odds.* Parked: VoxOdds answers 404 for its Politics markets and Polymarket's own API is
  not reachable from the hosts, so there is no feed to build on. The Prediction markets row above is
  unchanged.

**Roadmap item 11, the dollar index reconstructed (your Section 10 "Real-Time DXY", live 5 November).** ICE's
DXY has no free feed and FRED's broad dollar index is weekly and refused on the dashboard host. But DXY is a
published formula (a constant times six dollar rates at fixed weights: euro 57.6%, yen 13.6%, sterling
11.9%, Canadian dollar 9.1%, krona 4.2%, franc 3.6%) and the ECB publishes all six every business day, so
the index is rebuilt from the same request that fetches USD/JPY. It is the same formula on the ECB's 14:15
CET fixes rather than ICE's live spot, which puts it within a few tenths of the real print and a day behind
it by design; it is labelled reconstructed wherever it appears and a day with one rate missing is skipped
rather than filled, and the index carries its own age and stale flag. The card shows the index, its week and month change, and bitcoin's move between the same
two dates (the week-ago fix and the latest fix, read on bitcoin's completed daily candles of those dates, so
the two legs cover the same days); the read is your rule: a rise of 1% in a week is a dollar spike (a
headwind), a spike while bitcoin is up 3% between the same dates is the false-breakout signal (the card turns amber
and the agents' macro read says so), a fall of 1% is a slide (a tailwind). The recorded 4 November window
reads 101.26, down 1.1% on the week, a slide. The tech correlation named in the same line of your plan is
not built: the Nasdaq has no keyless intraday feed worth trusting.

**Roadmap item 12, the Director frames the 4-hour zone against the 24-hour horizon (your Section 10, live 5
November).** "Evaluates 4-hour tactical execution zones against 24-hour intraday liquidity horizons." Two desks
own the two horizons: Live executes inside the 4-hour candle (its expected range for the next four hours is
the tactical zone); Intraday owns the day (the day's high and low so far, the prior day's high and low, the
day's average price and the desk's 24-hour expected range are the edges where resting orders and stops sit).
The framing is computed in code from numbers already on the screen and handed to the Director as a block it
must not contradict: where price sits in the day's range, which edges the 4-hour range reaches (a probe
above, a probe below, an expansion window reaching both, or inside), and the width of the 4-hour range
against the day's. The Director then writes two or three sentences of judgment (horizon_framing), and the
plan it chose is checked against the same horizon in code: zone and every target inside the day's edges is a
tactical trade; a first target beyond them needs the 24-hour horizon taken first and the card says so; a zone
outside them is a trade at the day's edge, not inside it. The block sits under the trade plan on the Director
card and in the MCP's trap_report; reports made before 5 November have neither. Nothing is guessed: a
missing frame or forecast leaves its part absent and named.

**Per-desk chart switch (your Section 3 "Main Chart View", live 4 November; Live default changed 5 November).**
Each desk's chart has a two-way switch above it: Live opens on the 4-hour chart (your Main Chart View) with
the 1h as the other view; Intraday 4h or 1D, Weekly 1D or 3D, Monthly 1W or 2W, each opening on its own candle. The coarser view redraws the candles, and the
EMAs, RSI, divergences, trendlines, patterns and Fibonacci zones are re-read on those candles; the levels the
desk's call stands on are kept, so the call is drawn where it was made, and the title reads, for example,
"4h view · call on 1h". The call on the right, the forecast pane, the summary, the confluence gate and every
alert stay on the desk's own candle; nothing is judged on the coarser one. If the coarser frame is too short
to read, the desk says so and shows its own candle.

**Derived candles and the cascade read order (your Section 3, live 4 November).** No free venue serves
12-hour, 3-day, half-week or two-week candles, so they are built from the fetched ones on every refresh:
12H is two UTC half-days of hourly candles (opening 00:00 and 12:00); 3D is three days counted from a fixed
epoch Monday, so the same days always group together; 0.5W is a half-week of 4-hour candles, Monday 00:00
to Thursday 12:00 UTC and Thursday 12:00 to Monday, so two of them make exactly the weekly candle; 2W is
two Monday-aligned weeks (already built); 24H is the UTC daily candle, which the 1D frame already is. Each
war room now reads your cascade order as briefs beside its own three timeframes (the three-timeframe
questions are unchanged): Intraday 1h, 4h, 12h, 1D; Weekly 4h, 1D, 3D, 0.5W, 1W; Monthly 1D, 1W, 2W, 1M; Live
15m, 1h, 4h as before. A derived candle that could not be built is named as missing in the state, not left
out. The per-desk chart switch that would draw a 3D or 4h chart is the next item.

**Short-horizon traps (your Section 1.4, live 4 November).** A trap on the 15-minute or 1-hour candle is
short-horizon. It is still found, judged and anchored (the Live desk owns those candles in the sweep), but
it is shown and traded on the Live desk only: the Intraday desk reads the 1-hour candle and no longer
carries 1-hour traps in its TRAP warning band, and a swing desk's pinned trade is not warned off by a
15-minute or 1-hour trap sitting on its entry, stop or target. Wherever a short-horizon trap is written
(the Live desk's band, the TRAP desk's cards, the on-watch and settled lists, the Discord alert, the MCP
`traps` tool with a `horizon` field of short or swing) it carries the label "short-horizon". The desk
summaries and the Market Pulse never read the trap ledger, by construction: a test fails if either module
ever imports it. Traps on the 4-hour candle and longer are unchanged and read by every desk on that candle.

**Weekend modifier (your Section 4, live 4 November).** The weekend is Friday 21:00 UTC (the US cash close,
when volume thins) to Sunday 22:00 UTC (the CME open), one definition read everywhere. While it applies:
every desk carries a "Weekend tape" card at the top, and from six hours before, a "Weekend tape ahead" card;
the summary adds a Weekend tape line under the Big Picture, the directive ends "Weekend: trade the sweep and
rejection at the zone, not breakout continuation", and "what to expect" is worded as a sweep through the zone
and a rejection back inside the range with the first target as the realistic one; in the confluence gate a
clean break on weekend tape does not count as structure (a failed break, a reclaimed sweep or a value edge
still does, that is the weekend playbook); and the war-room state carries the session and its playbook, so
the judges and the Head read it. The Quant Forecast daily plan is unchanged: its own regime handles the day.

**Event window: five days either side (your Section 5.3, live 4 November).** Buy the rumour, sell the news
is read from five days before each high-impact US print to five days after (it was 48 hours). Inside the
window the print shows as flat or positioned (the run-up itself is still measured over the last 24 hours
against that week's volatility); after it, sell-the-news, spike-faded, confirmed or absorbed stays readable
for five days. The event-trap card now sits on every desk, not only Live Recon and Intraday, whenever a
print is positioned, reacting or has trapped a side; the summary's warning names the next print and how far
away it is on every desk. One limit, said plainly: Forex Factory publishes only the current week's file
(Sunday to Saturday; its next-week file does not exist, checked 4 November), so late in the week the
forward view is shorter than five days, and a print drops off the feed when the week rolls over. Keeping
prints in the store so the full five days hold across the week boundary is a small follow-up build that
needs your go. Unchanged on purpose: the Quant
Forecast chop window around a print (30 minutes before to 90 after) and the desks' 24-hour macro re-anchor
trigger, which are about the print's minute, not its week.

**The war room, per desk:** five bullish and five bearish domain agents (Quant, Auction, Order Flow, ICT,
Derivatives) each score a 100-point checklist of 22 weighted items. The model only answers "does this
condition hold" with a probability; weights and arithmetic are code. Judged by TypeSafe Jev, Gemini as
fallback, three batched requests per run. The Head needs a 10-point margin to call a direction, consults
the trap agent when both teams clear 60 or neither leads, and a trap probability of 0.60 or more
overrides the direction (BULL TRAP RISK). Before publishing, the desk consults the other desks' current
anchors (the handshake): a higher desk agreeing adds 10% conviction, opposing takes 20%, a lower desk
opposing takes 5%. Conviction moves, direction never.

**Anchoring.** The verdict is pinned to the desk's candle and cannot be overwritten for that window.
Interim runs add side notes only: a bias flip, a new trap at 0.60 or more, a 15-point swing, a high-impact
release inside the candle, a degraded scan. A desk's direction call re-anchors only on a trigger: funding
flips sign past a 0.003%/8h dead band, open interest moves 5 points, long/short crosses 1.0 by 0.03,
squeeze score reaches 60, price closes past the invalidation, the Director's classification changes, a
macro event is within 24 hours, or the anchor is stale.

**Scoring.** Every pinned window is scored once at its candle close against the price at that close.
Direction hit, flat band per desk (0.25% for 4h, 0.6% daily, 1.5% weekly, 3% monthly), Brier score per
team. A direction call is scored once per horizon unless the direction changes, and the ledger shows the
chance rate beside every hit rate. Under 10 samples is labelled indicative.

## 4. TRAP desk and Trap Intelligence reports

**TRAP desk (its own screen).** Candidates are found in code: liquidity taken and immediately reclaimed, or
price holding beyond the value-area edge while CVD drains the other way, with at least one independent
supporting factor. Three gates: the reclaim has happened, the raid is within 5 bars, the candidate is not
already settled. The strongest candidate goes to five domain questions, then the Head calls engineered,
authentic or unclear, and the affected desk vetoes if structure plus acceptance says the break is real. A
trap is declared at 60 or more, anchored until price settles it (played out at the point of control, or
invalidated half an ATR past the raid). Each timeframe is scanned once; a trap is one market event
whichever desk found it.

**Director reports (Director screen, "Run Analysis").** Two pipelines over the same snapshot:

- **1-agent**: one frontier call with the whole desk state (about 9,000 tokens in), returning the
  classification, scenarios, contradictions, blind spots and a geometric trade plan (side, zone, confirm
  candle and level, taker ratio, stop, targets, expiry). This is the one the top-bar button runs and the
  one the plan alerts follow.
- **15-agent**: ten specialists (Bullish B1–B5, Bearish R1–R5), a volatility agent, two desk syntheses,
  an accuracy agent and the Director. Richer, slower, more quota. Since 5 November its Director reads the
  same per-desk blocks as the 1-agent one (derivatives over each desk's window, cascade and forecast per
  desk, the 4-hour-versus-24-hour framing) under the same rules, word for word, and writes the same plan as
  numbers, the framing sentences and the event-regime tag; so a classic report drives the plan alerts and
  the Pulse exactly as a 1-agent one does. Before that its plan was prose only and nothing watched it.

Both are scored at 24 hours and 7 days, and either pipeline's plan is scored to its outcome (zone touch,
confirm, then stop or target). The plan being watched is the newest report's, whichever pipeline wrote it;
a report saved without a plan (a Director that failed) leaves the plan before it alone. Report bodies are stored in full since 30 October, so one report serves every
reader and the MCP.

Reports are run by a person, never by the worker, so the Gemini quota is spent only when someone asks.

## 5. Quant Forecast (its own screen, the daily plan)

Built from candles and the derivatives feeds at 00:00 UTC each day, audited every minute by the worker.

- **Engine:** GARCH(1,1) hourly sigmas, 3,000 Monte Carlo paths, a lean from momentum, value, flow, open
  interest and funding (weights 0.35 / 0.20 / 0.25 / 0.10 / 0.10). Phases per hour: chop, normal,
  expansion. The projected high and low with their hours, the 24 projected hourly candles (the median
  path drawn as candles, not simulated candles), the whole-day invalidation at the price only 10% of
  simulated days reach against the lean.
- **Levels:** value area of the last three days, air pockets (low-volume nodes) over eight days,
  estimated liquidation zones (air pockets where open interest surged), call wall, put wall, gamma flip.
- **Timeline tree and trade directive:** windows with plain-English reasons (quiet / normal / busy),
  then the plan as pure figures: direction, stop (the whole-day invalidation), an entry zone from the
  projected low (high on a bearish day) to the median path at the hour that low most likely prints, and
  three targets at the plan's levels in order, the projected high (low) always last. These rules are
  ours and await your confirmation.
- **Tiers (agreed 28 Oct):** CAUTION when a closed 15-minute candle moves more than twice its expected
  size against the plan, or the hour's taker delta turns against it at 1.5× usual, at most one an hour.
  MODEL HELD when price takes back half the shock within the hour. INVALIDATED on a 4-hour close or two
  hourly closes in a row beyond the level, and only then; the single-candle tests were retired 30 Oct.
  Scenario B is a fresh forecast for the rest of the day on a second chart; the second failure locks the
  day (LOCKOUT).
- **Magnitude scaler:** once a day, an open-interest surge through the value edge or taker delta at 1.5×
  usual widens the target to the next level.
- **Pre-flight:** an active 4h engineered trap forces a neutral plan; a high-impact USD release forces a
  chop window from 30 minutes before to 90 minutes after; a major print (election, FOMC, Fed chair) inside
  its 24h-before / 48h-after window puts the plan in event mode: no range plan, 1.5× sigma every hour,
  invalidation at the 5th percentile both sides, the event named in every message (fix 3, 5 Nov).
- **Sigma calibration (live since 30 Oct):** realised day range against the engine's own expected range
  over the last ten days, bounded 0.8–1.3. Neutral on the 60-day backtest; kept because it is honest.
- **Liquidation gravity (built, OFF):** pulling paths toward liquidation zones and dealer walls made the
  60-day level-touch Brier worse at every strength (0.141 at k 0.5 against 0.105 off). It runs in shadow
  at k 0.5; its hits sit on their own labelled card (Liquidation gravity · shadow ledger) under the track
  record, never inside the live line. The shadow record over live weeks decides whether it ever steers.
- **Sticky Monday Master Plan:** written once on the first pass after Monday 00:00 UTC from the Weekly
  desk and the Deribit gamma profile. Weekly bias, put and call walls, baseline floor and ceiling, weekly
  flip. Does not move during the week.
- **Self-audit card:** miss percentage and a rule-based root cause (gravity, volatility, lean, timing).
  Multipliers are never changed by the audit; recalibration awaits approval.
- **Record:** 60-day backtest counts, live record over the last 10 and 30 scored days, each scored day
  with projected against actual high and low. Backtest and live are kept apart on screen and in the MCP.
- **JSON payload (schema v7):** your key names, downloadable from the drawer and served by the MCP.

## 6. Alerts to Discord

One webhook, one channel. Every alert carries a dedup key behind a unique index, and the dashboard and
the worker claim an alert with one database statement before sending, so nothing is sent twice however
many pages are open. An alert older than 30 minutes (the worker was down) is skipped rather than sent late.
Every message is scanned against a banned-jargon list before it goes out.

| Alert | When | Words |
| --- | --- | --- |
| Quant Forecast plan | 00:00 UTC, and Plan B after an invalidation | Since 6 Nov (wording batch 2): "📊 DAY PLAN · Thu 05 Nov · BTC $80,467"; "Leaning higher · 60% confidence · record 1 of last 3 days."; a trap holding the day to a range, when one is; the event's note on an event day; the directive; "Expected range … until 00:00 UTC (05:00 PKT), low near …, high near …. Plan is off below … or above …"; "Now (window): phase, range. Next (…)"; "Busiest hours today: …" (the expansion windows, at most two); the trade figures on a directional day. The tiers: "⚠️ SHARP MOVE · BTC …" / "⚠️ SELLING PRESSURE AGAINST THE PLAN · BTC …" (what moved, in which window; the plan still stands and where it is off; what to do), "🟢 PLAN HELD · BTC …" (sent only 30 minutes or more after its own caution), "🔴 DAY PLAN OFF · BTC …" (the close that ended the plan, then "📊 PLAN B …") and "🔴 DAY PLAN OFF AGAIN · BTC …" (the lockout: stand aside until 00:00 UTC). The dashboard's wire panel keeps the stat line |
| CAUTION / MODEL HELD / INVALIDATED / LOCKOUT | As the tiers fire | Your tier wording, with the timeline line |
| Plan alerts (1-agent report) | WATCH when a candle touches the zone, SELL NOW / BUY NOW on the confirm close, NO TRADE, TARGET n OF N HIT, STOPPED OUT, PLAN EXPIRED | Since 6 Nov (wording batch 4): "🟡 WATCH · BTC $83,000 / The sell zone $83,075–$83,535 was touched: the candle high reached $83,100 at 15:30 UTC (20:30 PKT - Thu 05 Nov). BTC is now $83,000, back below the zone. / Sell only if a 1-hour candle closes below $83,077 with sellers doing more of the trading. Next close: … / Forget it if a 1-hour candle closes above $83,535. Stop $85,238 · targets $82,237 / $80,600 / $78,000."; "✅ SELL NOW · BTC $83,050" with the stop and targets as percent distances and "Good until …"; "⚪ NO TRADE · short off · BTC $83,650" with the close that ended it; "🎯 TARGET 1 OF 3 HIT · BTC $82,237"; "🔴 STOPPED OUT · BTC $85,238 / Short from $83,050 hit the stop $85,238: 2.63% loss. The sell idea from the 15:39 report was wrong."; "⌛ PLAN EXPIRED". The report is named by its time, every end state says "Next: the hourly Pulse, or a new report", a plan made inside an event regime still carries its regime line, and the Context line is gone |
| Cascade | Only moves into A (strong setup) or C (trend losing control), after the state holds (Live 5 min, Intraday 15 min, Weekly 1 h, Monthly 4 h); a return to the same state within the repeat window (8 h / 12 h / 24 h / 72 h, fix 5) stays silent. The C warning names the engine closes beyond the day's average price and the confirming close | Since 6 Nov (wording batch 3): "📈 UP SETUP · today · BTC $82,909" / "📉 DOWN SETUP", the trend and the shorter candles agreeing, "No trade from this alone: wait for the report's plan or the Pulse" and the average price whose close ends it; "⚠️ DOWNTREND WEAKENING · next 4 hours · BTC $80,317", how many times the shorter candles closed beyond the trend's average price, "Not a trade yet. It becomes a reversal if a 4-hour candle closes above $X; a REVERSAL CONFIRMED alert will say so" |
| Reversal confirmed (fix 5, 5 Nov) | A desk in C whose completed anchor candle (Live 4h, Intraday 1D, Weekly 1W, Monthly 1M) closed beyond the day's average price against the bigger trend, once per repeat window; urgent | "🔄 REVERSAL CONFIRMED · next 4 hours · BTC $78,140"; the candle that closed beyond the trend's average price, with its time; "If you are short from the trend, it is over. Expect a move up. Wrong only if a 4-hour candle closes back below $X" |
| TRAP | Declared, and settled | "🪤 TRAP CALL · today (4h) · BTC $80,000 / The pop above $80,000 should fail and drop. / Target $78,000 · wrong if price goes past $81,000"; "✅ TRAP WORKED" and "❌ TRAP FAILED … The call was wrong" when it settles |
| Market Pulse | Every hour boundary, one message naming the closes it covers: "1h + 2h + 4h candle closes" at 00:00, 04:00, 08:00, 12:00, 16:00 and 20:00 UTC, "1h + 2h" on the other even hours, "1h" otherwise (from the Live desk, or a live plan if one is open); and when a report with a sided plan is run. Live 4 Nov; it was the 4-hour close only. Speaks from the open plan while there is one; a plan the tracker cancels hands the Pulse back to the Live desk the same minute (fix 1, 5 Nov). The event section leads with the event regime, then the last day's High prints with their read, then the next day's High and Medium prints (fixes 3 and 4, 5 Nov) | Six lines since 6 Nov (wording batch 1): "🤖 PULSE 1h + 2h + 4h close · BTC $76,065"; the read (the desk's lean with the nearest floor or ceiling, or the plan's own line: made when, buy or sell where, target, stop, until when, then where price stands and the one thing to do); "Target … · stop line …"; "Busiest hours: 12:00–16:00 UTC (17:00–21:00 PKT), later today"; the news (prints at the same minute on one line; a headline whose dollar figure is more than 2% from the price is dropped); the time. No rule lines, no section labels, no report id |
| Trade alerts | A pinned trade's target, stop, trap on its levels, monitoring checkpoint | Since 6 Nov (wording batch 5): "🎯 BTC/USDT LONG · target TP1 hit · BTC $80,600 (+0.7R: 0.7 times your risk)", "🔴 BTC/USDT LONG · stopped out at $79,000 (-1.1R)", "🪤 BTC/USDT LONG · a bull trap call on the 1h chart sits $50 from your TP1: expect price to turn there.", "⏰ BTC/USDT LONG · the 4h close has passed: review the position." |
| Scheduled agent run | After a High-impact US release has settled or the day plan is invalidated; at most six a day, half an hour apart (core.scheduled_runs) | Since 6 Nov (wording batch 5): a run that produced a report sends nothing of its own, the report's Pulse carries it (the ledger keeps "🤖 NEW REPORT · BTC $80,000 / Triggered by the US CPI m/m release at 13:30 UTC (18:30 PKT). Report TIR1-…: engineered trap, plan sell. Details follow in the Pulse." for the record). A run the model refused sends "🤖 NO NEW REPORT · BTC $76,020 / Triggered by the day plan being invalidated at 16:00 UTC (21:00 PKT), but the model was unavailable. The last plan (report TIR1-20261105-153911, buy $76,000–$76,500) stays in force. The next automatic run comes with the next release or plan change, not before 16:30 UTC (21:30 PKT)." The daily cap sends the same header with "6 automatic runs already ran today" and when the next is allowed. No model names or error codes: those go to the pass's problems and the log |
| Test alert | On request, from the dashboard's alerts panel | "✅ Alerts connected. This channel gets the hourly Pulse, the day plan and its warnings, trade plans (watch, confirm, targets, stop) and trend changes." |

Telegram is implemented and switches on the moment a bot token and chat id are added to the VM's
environment. Email and SMS are not built: they need an SMTP sender and a paid gateway.

## 7. Active Trade desk

Enter a position and it is evaluated, pinned and tracked until you kill it. Stop verification by what sits
between the stop and price (a stop nearer than the pool below a long is one that gets hunted); timed
take-profits from the volatility bands and Fibonacci time zones, each with its R; the measured pullback
band; the monitoring schedule in UTC. Ten sub-agents and a Head call take / wait / stand aside; the two
handshakes (desk alignment and the trap ledger) are deterministic. The probability on the card is a
confluence estimate clamped to 15–85% and labelled as such: nothing counts how often a setup like this
worked. The plan is written once at open; Refresh updates the live block only; Re-judge opens a new card;
KILL unpins it.

No trade is pinned today.

## 8. AI access: the MCP server

Any assistant that speaks MCP can read the whole system and talk about it in your framework. The server
is read-only: no tool writes, fetches an exchange or runs a model. The connect address contains the
access token and is shared privately, never written down in the repo.

| Tool | Answers |
| --- | --- |
| `how_to_read` | The reader's guide: every field, the house rules, the glossary. Assistants are told to call it first |
| `todays_plan` | The day's Quant Forecast plan, full JSON; a past day by date |
| `plan_status` | Where the plan stands now: tier, directive, today's alerts, each plan row |
| `forecast_record` | Backtest, live record, scored days, the desks' ledger with chance rates |
| `desk_anchors` | The four desks' anchored directions and recent re-anchors with triggers |
| `recon_windows` | A desk's anchored windows with the full rubric reasoning, handshake, side notes, score |
| `desk_calls` | The desks' direction calls with each one's score |
| `trap_reports` / `trap_report` | The Director reports: the list, and one in full with every brief, scenario and plan |
| `traps` | Declared traps and how each settled |
| `trade_journal` | Pinned trades, plans, events |
| `order_book` | Depth, imbalance, sweeps and scoops from the recorder |
| `recent_alerts` | The last alerts sent, newest first |
| `monday_plan` | This week's Sticky Monday Master Plan |

**Claude** (web or desktop): Settings → Connectors → Add custom connector, paste the address, no sign-in.
Verified: Claude read today's plan correctly and in your terms.

**Gemini** (web app): Settings → Connected Apps → Custom apps → Add. Google limits this to US users, 18+,
on a personal account with Keep Activity on. Gemini caches the tool list when it connects, so after a new
tool is added it must be re-synced or re-added. Elsewhere, the Gemini CLI connects with one command.

**ChatGPT**: Developer-mode connectors accept the same address.

What the MCP cannot serve yet: news headlines. They are fetched live into the agents' context when a
report runs and are not stored, so there is nothing to read back. See section 9.

## 9. Not built, parked, or waiting on a decision

**Your earlier asks, status:**

| Ask | Status |
| --- | --- |
| Election-day fix 1: cancelled plans reach the report ledger | Fixed 5 Nov: the scorer applies the tracker's two cancel rules; the Pulse stops speaking from a cancelled plan the same minute (section 3) |
| Election-day fix 2: Live Recon opens on the 4-hour chart | Done 5 Nov: your Main Chart View; 1h is the other view (section 3) |
| Election-day fix 3: event regime (Section 6.2) | Live 5 Nov: an election / FOMC / Fed-chair print puts the Quant Forecast, the Pulse, the desk summaries and the war-room state into event mode from 24h before to 48h after (section 3) |
| Election-day fix 4: the Pulse names the scheduled prints | Live 5 Nov: regime first, last day's High prints with their read, next day's High and Medium prints (section 3) |
| Election-day fix 5: cascade reversal confirmation | Live 5 Nov: the warning names the level and the confirming close; REVERSAL CONFIRMED on the anchor candle's close beyond the day's average price; 8h / 12h repeat windows (section 3) |
| Election-day fix 6: scheduled agent runs | Live 5 Nov: the worker runs the Director + Live war room 15 min after a High-impact USD print and after a day-plan invalidation; once per trigger, 30 min apart, six a day at most (section 3) |
| Roadmap item 10: four macro feeds | Live 5 Nov: 2-year minus funds rate (FRED, New York Fed fallback) with the ±25 bp read; BTC dominance (CoinGecko, change from own samples); USD/JPY yen carry (ECB via frankfurter.dev, BoJ rate hand-set); election odds parked, VoxOdds 404 (section 3) |
| Roadmap item 11: dollar index reconstructed | Live 5 Nov: ICE's DXY formula on the ECB's six daily rates, a day behind and within a few tenths; 1% week = spike or slide; spike + bitcoin up 3% between the same dates = false-breakout signal (section 3) |
| Roadmap item 12: Director 4-hour zone vs 24-hour horizon | Live 5 Nov: framing computed in code (price in the day's range, which day edges the 4-hour range reaches), the Director's sentences on it, the plan checked against the day's edges (tactical / needs the edge taken / at the edge) (section 3) |
| Turso sync budget | Live 9 Nov: Turso warned at 75% of the free plan's 3 GB Monthly Syncs. Cause: four signals rows and four cascade-state rewrites every pass, about 22 pages (4 KB each) pulled back by the worker's replica every minute (~130 MB a day) and again by the dashboard's. Fix: a signals row only when the score moves 3 points or more, the anchor re-fires or an hour has passed (core.anchors; the score flickered 38 / 39 every minute), cascade state written only on change (core.store.track_cascade), the order-book report rewritten every 5 minutes instead of every pass (worker.BOOK_REPORT_GAP_MS), dashboard sync throttled to one pull per 30 s (Store.sync min_gap_s). Measured after the first deploy: 22 → 6 pages a minute; the second deploy targets 1 or less. Storage (5.9 MB of 5 GB) and writes (~6,000 a day of 10 M) are far from their limits. If the budget is still short: Turso Developer, $4.99 a month, 10 GB of syncs |
| Discord wording batch 5: run line, trade alerts, test alert | Live 6 Nov: a scheduled run that produced a report sends nothing (the Pulse carries it); a refused run sends NO NEW REPORT with the plan still in force and when the next run can come; the pinned-trade alerts carry an emoji, dollar figures and the R-multiple explained; the test alert names what the channel gets (core.scheduled_runs, core.alerts, core.store) |
| Discord wording batch 4: the plan alerts | Live 6 Nov: WATCH, SELL NOW / BUY NOW, NO TRADE, TARGET n OF N HIT, STOPPED OUT, PLAN EXPIRED in one header shape with the price; the report named by its time, percent distances to the stop and the targets, "Next: the hourly Pulse, or a new report" on every end state; the Context line dropped (core.plan_alerts) |
| Discord wording batch 3: cascade, reversal and TRAP | Live 6 Nov: UP / DOWN SETUP, UPTREND / DOWNTREND WEAKENING, REVERSAL CONFIRMED, TRAP CALL / WORKED / FAILED; one header shape with the desk's horizon in plain words and the price; no desk names, no state letters (core.market_alerts) |
| Discord wording batch 2: the Quant Forecast messages | Live 6 Nov: DAY PLAN / PLAN B / SHARP MOVE / SELLING or BUYING PRESSURE AGAINST THE PLAN / PLAN HELD / DAY PLAN OFF / DAY PLAN OFF AGAIN, plain sentences, no stat line, no chop list, no bracketed tags; a PLAN HELD within 30 minutes of its caution is not sent; the flow caution carries the spot price (core.qf_alerts) |
| Discord wording batch 1: the Pulse | Live 6 Nov: six lines, header with the closes and the price, no rule lines or section labels, busiest hours in UTC and PKT, same-minute prints grouped, stale-price headlines dropped (core.pulse). Batches 2–5 (Quant Forecast, cascade and trap, plan alerts, run / trade / test lines) follow |
| Report review 5 Nov (TIR1-20261105-153911) | Live 5 Nov: targets beyond the 24-hour expected range named; a stop beyond the primary scenario's invalidation noted (plan_notes); the .md download carries framing, check, risk/leverage, regime tag and notes; all-coins liquidation row labelled; discipline rules in both Directors (section 3) |
| Classic 15-agent Director parity | Live 5 Nov: the classic Director gets the per-desk derivatives, cascade, forecast and horizons blocks and the same plan, framing and regime rules as the 1-agent Director, and its numbers plan is watched by the alerts and the Pulse (section 5) |
| WATCH / Pulse "touched" wording and the event regime in the Director's plan | Live 5 Nov: the WATCH and the Pulse name the touch (the candle high or low, with its time) and then the current price, instead of "Price reached the zone" under a headline price below it; a plan made inside an event regime is tagged, its size halved in code, the Director asked for the wider stop, and the Pulse says the no-range rule is the day plan's (section 3, fix 3) |
| Per-desk chart switch (Section 3, Main Chart View) | Built 4 Nov: the desk's own candle by default, one coarser view a click away (4h / 1D / 3D / 2W); the call, forecast, summary and alerts stay on the desk's candle (section 3) |
| Derived candles 12H / 3D / 0.5W / 2W and the cascade read order (Section 3) | Built 4 Nov: built from the fetched candles each refresh; each war room reads your cascade order as briefs beside its three timeframes; 24H is the daily candle (section 3) |
| Short-horizon trap labels (Section 1.4) | Built 4 Nov: 15m and 1h traps labelled short-horizon, shown and traded on the Live desk only, never in a summary or the Pulse; 4h and longer unchanged (section 3) |
| Market Pulse on every 1H, 2H and 4H close (Section 10) | Built 4 Nov: one message per hour boundary naming which closes it covers; a message older than 30 minutes is skipped, not sent late; the 4-hour close keeps its old ledger key so nothing double-sent at the switch (section 6) |
| Weekend modifier (flag, sweep-and-rejection wording) | Built 4 Nov: Friday 21:00 to Sunday 22:00 UTC; desk card, summary wording and directive, a clean weekend break does not count in the gate, session in the judges' state (section 3) |
| Event window five days either side of each high-impact print | Built 4 Nov: event-trap reads, the card on every desk and the summary warning span five days before to five days after, as far as the feed reaches. Open: the feed is the current week only (no next-week file exists); storing prints to hold the full five days across the week boundary needs your go (section 3) |
| Named anomaly traps (Section 5.3: Bullish squeeze trap, Liquidation flush trap) | Built 4 Nov: on the desk, in the summary, in the gate's positioning family and in the judges' state; SET when the flow already disagrees with the crowd, FORMING before (section 3) |
| Higher-timeframe bar (Weekly/Monthly level within half an ATR on the path to Target 1 → WAIT) | Built 4 Nov: a level inside the zone is the entry and never bars; farther levels before Target 1 go in the warning (section 3) |
| Confluence gate and v22 summary layout on every desk | Built 4 Nov: three of seven families must agree before a desk's trade box opens; WAIT names what is missing or what conflicts (section 3) |
| Desks disagree in one batch (one message when the four desks conflict) | Not built. Needs your rule for what counts as disagreement and when to send |
| Cautions too trigger-happy inside chop windows | Not built. Proposal: raise the caution threshold inside a chop window, or suppress cautions there and keep invalidation only |
| Lockout after two failed plans | Built as specified (message 5). Open: whether a third plan should ever be allowed on a very busy day |
| HMM regime model versus the lean | Not built. Would need a labelled history to beat the current lean; no evidence yet it would |
| Telegram | Code ready; needs a bot token and chat id |
| Email / SMS | Not built; needs accounts |
| JSON consumer | Payload v7 is served by the MCP and the drawer; nothing consumes it yet besides assistants |
| Plan time 00:00 UTC | Current. Say if you want a different anchor hour |
| Pulse target = nearest level rather than plan target | Built 30 Oct: desk target is the farther of the first level and the 4h range edge |
| Pulse re-dispatch on every human report | Built 30 Oct: only reports with a sided plan send a Pulse |
| Order-book scoop and sweep | Recording since 28 Oct; metrics are provisional constants until weeks of book exist. They gate nothing yet |
| Kill zones | Measured on 10,000 hourly candles and shown with odds: New York 13–17 UTC turns at 1.8× chance; Asian and London are no better than chance and the screen says so |
| X (Twitter) feed | Not built; every usable X API is paid |
| Crash memory | Not built |
| Ehlers sine wave | No out-of-sample edge on any venue or timeframe. An EXPERIMENTAL card on Live Recon shadow-logs the one promising cell (big turns, NY, 15m). Keep or drop after about 100 live signals |
| Multi-asset (ETH, SOL, gold as desks) | Not built; a large parameterisation job. Gold is context only |
| Liquidation gravity steering the plan | Shadow only, by evidence (section 5) |
| News over the MCP | Needs the worker to store the context snapshot with each report; about an hour |

**Infrastructure:**

- Gemini API is on the free tier: 20 requests a day per model, so the 15-agent pipeline can fail by
  afternoon. Enabling billing on Google AI Studio fixes it. The fallback chain keeps the 1-agent report
  working on a lighter model.
- Google Cloud trial credit ends 26 January 2027. After that the VM needs billing enabled (about $4 a
  month for the public IP; the machine itself is in the free tier) or the worker stops.
- Oracle Cloud's free ARM machine (Dubai) has had no capacity on every attempt. A launch poller can keep
  trying from the VM; it has not been started.
- If the VM's public IP ever changes, the DuckDNS name for the MCP needs a manual update.

## 10. Decisions that need your answer

1. **Entry and target rules in the trade directive.** Ours today: entry zone from the projected low to
   the median path at its likeliest hour, stop at the whole-day invalidation, targets at the plan's
   levels with the projected high last. Confirm or replace.
2. **Invalidation definition.** 4-hour close or two hourly closes in a row. Confirm.
3. **Cautions in chop windows.** Suppress, raise the bar, or leave.
4. **Desks-disagree alert.** Define it, or drop it.
5. **Which desk speaks in the Pulse.** Live Recon today, by our choice.
6. **Gemini billing** (fixes the 15-agent failures) and **Telegram credentials** (switches it on).
7. **Sine wave card:** keep collecting, or remove.
8. **Liquidation gravity:** review the shadow record together after four live weeks.
9. **Anything on this page you want added.** Mark it, and it becomes the next build queue.
