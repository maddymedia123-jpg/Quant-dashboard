# Quant Dashboard — System Handbook

*What exists, how it fits together, what is still open. Written for Hammad. Kept in the repo at
`docs/handbook.md` and updated with every merge; last revised 2 November 2026.*

Everything below is in the code on `main` and live unless a line says otherwise. Nothing here is planned
work dressed up as done.

---

## 1. What runs where

| Piece | What it does | Where |
| --- | --- | --- |
| **Dashboard** | The Streamlit terminal: nine screens, TradingView-style charts, the war rooms, the reports, the Quant Forecast | Streamlit Cloud, redeploys on every merge to `main` |
| **24/7 worker** | One pass a minute, whether or not a page is open: fetches the market, builds the four desks, anchors and scores, settles traps, sends every alert and the Market Pulse. Makes no model calls. | Google Cloud VM, `quant-worker.service` |
| **Order-book recorder** | One snapshot of the Gate.io BTC perp book every 5 seconds, kept 35 days locally | Same VM, `quant-book.service` (recording since 28 Oct 2026) |
| **MCP server** | Read-only access for AI assistants (Claude, Gemini, ChatGPT, Cursor): 14 tools over the shared store | Same VM, behind HTTPS, token in the address |
| **Shared store** | Turso (hosted SQLite): every anchor, call, score, trap, trade, alert, report, plan and book report. The dashboard and the worker read and write the same database. | Turso, US-East |
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
| Economic calendar (high-impact USD prints) | Forex Factory, cached 30 minutes | Macro window trigger, Quant Forecast chop windows, Pulse |
| Headlines | CoinDesk and Cointelegraph RSS plus the cryptocurrency.cv aggregator (14 publishers) | Macro & News screen, narratives in the agents' context, Pulse |
| Gold | goldprice.dev | Macro & News |
| Prediction markets | Polymarket via VoxOdds, with execution-quality bands | Macro & News, Fed odds in the Pulse |
| US yields | FRED, falling back to the US Treasury daily curve (FRED refuses Streamlit Cloud) | Macro & News, agents' macro read. The dollar index is a named gap |
| CFTC Commitments of Traders | CFTC, weekly | Weekly and Monthly desks only |
| Exchange net flows | CoinMetrics community (all flows, not whales) | Weekly and Monthly desks |
| Order book | Our own recorder on the VM | Scoop and sweep report, order_book tool |

A provider that answers 403 or 451 is parked for six hours and logged once, so a geo-block does not spam
the log every minute.

## 3. The four desks

| Desk | Reads | Charts | Anchored to | Scored over |
| --- | --- | --- | --- | --- |
| Live Recon | 15m / 1h / 4h | 1h, next 4 hours | the UTC 4h candle | 4 hours |
| Intraday | 1h / 4h / 1D | 4h, next 24 hours | the UTC day | 24 hours |
| Weekly | 4h / 1D / 1W | 1D, next 7 days | Monday 00:00 UTC | 7 days |
| Monthly | 1D / 1W / 1M | 1W, next month | the calendar month | 30 days |

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
  an accuracy agent and the Director. Richer, slower, more quota.

Both are scored at 24 hours and 7 days, and the 1-agent plan is scored to its outcome (zone touch, confirm,
then stop or target). Report bodies are stored in full since 30 October, so one report serves every
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
  chop window from 30 minutes before to 90 minutes after.
- **Sigma calibration (live since 30 Oct):** realised day range against the engine's own expected range
  over the last ten days, bounded 0.8–1.3. Neutral on the 60-day backtest; kept because it is honest.
- **Liquidation gravity (built, OFF):** pulling paths toward liquidation zones and dealer walls made the
  60-day level-touch Brier worse at every strength (0.141 at k 0.5 against 0.105 off). It runs in shadow
  at k 0.5; the record panel shows shadow hits beside live hits. The shadow record over live weeks
  decides whether it ever steers.
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
| Quant Forecast plan | 00:00 UTC, and Scenario B after an invalidation | Regime, directive, timeline line |
| CAUTION / MODEL HELD / INVALIDATED / LOCKOUT | As the tiers fire | Your tier wording, with the timeline line |
| Plan alerts (1-agent report) | WATCH when price reaches the zone, CONFIRMED on the confirm close, NO SETUP, TARGET 1–3, STOPPED, EXPIRED | Plain words, from the plan's numbers |
| Cascade | Only moves into A (strong setup) or C (trend losing control), after the state holds (Live 5 min, Intraday 15 min, Weekly 1 h, Monthly 4 h); a return to the same state within the repeat window (3 h / 6 h / 24 h / 72 h) stays silent | Desk and price, old state → new state, what changed |
| TRAP | Declared, and settled | "A dip below X should fail and bounce" |
| Market Pulse | Each 4-hour close (from the Live desk, or a live plan if one is open) and when a report with a sided plan is run | Your Pulse template: direction, Target 1, stop, turning window, US releases, BTC headlines, Fed odds within 7 days |
| Trade alerts | A pinned trade's target, stop, trap on its levels, monitoring checkpoint | From the card |

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
5. **Which desk speaks in the 4-hour Pulse.** Live Recon today, by our choice.
6. **Gemini billing** (fixes the 15-agent failures) and **Telegram credentials** (switches it on).
7. **Sine wave card:** keep collecting, or remove.
8. **Liquidation gravity:** review the shadow record together after four live weeks.
9. **Anything on this page you want added.** Mark it, and it becomes the next build queue.
