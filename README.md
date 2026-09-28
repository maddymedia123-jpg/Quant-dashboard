# Trap Intelligence — BTC market-surveillance terminal

Streamlit dashboard with TradingView-style charts. Three layers:

1. **Deterministic analysis** per timeframe (Live 15m, Intraday 1h, Weekly 4h, Monthly 1d): EMA
   21/50/100/200, RSI with divergence detection (confirmed and forming), swing support/resistance,
   trendlines, Fibonacci time zones and retracements, chart patterns, volatility regime and expected
   range, squeeze risk, an anchored direction call with invalidation, plus Smart Money Concepts and
   liquidity protocols (below) and an active-trade stress-test.
2. **A war room per category** — eleven agents scoring a 100-point checklist, a trap audit, and a
   summary pinned to that category's candle.
3. **The 13-agent Director report** (`Run Analysis`) — ten specialists, two desks, an accuracy agent
   and the Director.

Nothing on the page is invented. Every feed carries an availability flag, missing values render as `—`,
and every score can be recomputed from the probabilities and the weights in the code.

## Run

    python -m venv .venv
    .venv/Scripts/pip install -r requirements-dev.txt
    .venv/Scripts/python -m pytest          # full suite, no network
    .venv/Scripts/python -m streamlit run app.py

Offline UI check without network: `TI_OFFLINE_FIXTURES=1 .venv/Scripts/python -m streamlit run app.py`
(serves the recorded fixtures under `tests/fixtures`; sources are labelled "fixture").

## Data (keyless public APIs)

Kraken spot OHLCV (Gemini exchange fallback) · USD-M futures for funding, open interest, long/short and
taker flow, tried in order **Binance → Bybit → Gate.io**, then thinner fallbacks (CoinLobster relay +
Hyperliquid composite, Kraken Futures, and finally values derived from our own stored samples) ·
Hyperliquid funding and OI · CoinLobster 24h liquidations, $100K+ whale trades and unusual-flow radar ·
Deribit options for max pain, put/call and IV skew · DefiLlama stablecoin supply · alternative.me Fear &
Greed · Forex Factory weekly economic calendar (high-impact USD prints with forecast/previous; drives the
macro-window trigger).

Binance and Bybit geo-block US egress, and Streamlit Cloud is US-hosted, which is why Gate.io sits in the
chain — the live site sources futures from it. Whichever provider answered is labelled in the sidebar.

## The war room (eleven agents per category)

Each category tab has a **Run … war room** button. Five bullish and five bearish domain agents score the
same market state against a 100-point checklist, and the Head of the War Room consults a trap agent
before publishing:

| Agent | Domain | Points |
| --- | --- | --- |
| 1 | Quant & Statistics | 20 |
| 2 | Auction Market & Volume Profile | 20 |
| 3 | Order Flow & Delta | 20 |
| 4 | ICT & Liquidity | 20 |
| 5 | On-Chain & Derivatives | 20 |

Twenty-two weighted items make up those 100 points (`core/agents/rubric.py`), each phrased once for the
bullish case and once for the bearish one. Macro and news are a global engine rather than a desk
sub-agent, following the enhanced spec. `RUBRIC_VERSION` records which checklist scored each window, so
the accuracy report never cites an old score as evidence about an item that did not exist yet.

An expander on each tab shows the **Institutional Quantitative Matrix**: every reading each sub-agent
was given, built from the same state dictionary the judges receive (`core/agents/deck.py`), so the
screen cannot disagree with the scorecard. A missing feed shows a dash, never a number. **The model only answers "does this condition hold" and
returns a probability; the weights and all arithmetic stay in code**, so a team score is auditable item by
item and a weight can change without re-running any inference.

The head's policy, also in code (`core/agents/live_recon.py`):

- a direction needs a **10-point margin** between the teams, otherwise no side is called;
- the trap agent is consulted when **both teams clear 60** or neither leads by ten;
- a trap probability of **0.60 or more overrides** the direction (`BULL TRAP RISK`);
- a team that failed to answer scores zero and the notes say it was a failure, not a reading of zero.

Each category reads its own timeframes, judges over its own horizon and pins to its own candle:

| Tab | Timeframes | Horizon | Anchored to |
| --- | --- | --- | --- |
| Live Recon | 15m / 1h / 4h | next four hours | UTC 4h candle |
| Intraday | 1h / 4h / 1d | next 24 hours | UTC day |
| Weekly | 4h / 1d / 1w | next seven days | week opening Monday 00:00 UTC |
| Monthly | 4h / 1d / 1w | next 30 days | calendar month |

Nothing above the weekly candle is fetched, so Weekly and Monthly read the same charts and differ in
horizon and anchor. Profiles and the window maths live in `core/agents/recon_profiles.py`; the Unix epoch
fell on a Thursday, so weeks are offset to Monday and months are calendar months of any length.

### Judgments: TypeSafe (Jev), with Gemini as fallback

`core/agents/judge.py` sends all eighteen questions for one side as a single batched request to TypeSafe
System One (`jev-latest`), which returns a calibrated probability per question — that batching is what
makes an eighteen-item checklist affordable, at three requests per war-room run rather than eleven. A
`Choice` judgment backs the trap audit. Without `TYPESAFE_API_KEY` the same questions run through Gemini
JSON mode instead, so the rubric works either way; the scorecard names the provider that answered. A
failure returns an empty result rather than raising, so a dead provider degrades the report instead of
killing the run.

## The anchored summary and interim side notes

The summary is pinned to the category's candle and **stays immutable for that window** — a re-run
mid-candle never reloads or replaces it. Insert-once is enforced in SQLite, not in the caller, so no code
path can overwrite the context being traded from.

Interim re-runs are **append-only side notes** when something substantial changed: the bias flipped, a new
trap call at or above 0.60, a 15-point swing that does not flip the bias, a high-impact release landing
inside the candle, or a scan that ran degraded. Each note carries a dedup key, so an interim scan that
keeps seeing the same flip announces it once (`core/live_anchor.py`).

## Volatility matrix

`core/indicators/volatility.py` computes, per timeframe: realized volatility annualised from that
timeframe's own returns (365 days, since crypto never closes), implied volatility taken from the Deribit
options feed or shown as a dash, ATR with its percentile regime, the ±1/2/3σ channels, and the **Hurst
exponent** with a plain-English memory label.

Hurst is estimated from how return variance scales with aggregation, using overlapping windows because
that measurably cut the estimator's spread by about a third against disjoint blocks. A random walk
measures 0.50 give or take 0.03 on the history we hold, so the neutral band is two standard deviations
wide: above 0.56 trends persist, below 0.44 moves mean-revert, and inside that band the reading claims
nothing. Tests hold it to series whose behaviour is known by construction.

## SMC and liquidity protocols

The rubric asks about structure and order flow, so both are computed from candles rather than left to a
model to eyeball, and shown on every category tab beside the agents' scores.

`core/indicators/smc.py` — swing structure; break of structure versus change of character (the
distinction is the prior bias, tracked while walking the series); fair value gaps with fill state; order
blocks taken from the last opposite candle before a break, with mitigation state; premium/discount against
the dealing range. Thin data returns "unclear", never a guess.

`core/indicators/liquidity.py` — pools of equal highs and lows, distinguishing a **sweep** (wicks through,
closes back inside) from a **break** (closes beyond), because they are opposite conclusions; cumulative
delta; volume profile point of control and value area; open interest change per timeframe window.

Two honesty rules worth knowing:

- **Delta is a candle close-position proxy, not tick tape.** Without trade-level data there is no true
  CVD, and it is labelled as a proxy everywhere it appears, in the agent state and on screen.
- **An open-interest window reports nothing unless a stored sample brackets it.** A reading from the wrong
  window is not an approximation of the right one.

## TRAP Intelligence (its own tab)

The watchdog for engineered moves: fakeouts, swing failures and stop runs. Candidates are found in code
and only then judged, so a trap is never asserted without the evidence that produced it and every level
on a card comes from the data.

**Two triggers, from the spec.** Liquidity taken and immediately reclaimed (the swing failure), or price
holding beyond the value-area edge while cumulative delta drains the other way. A trigger needs at least
one *independent* supporting factor from funding, open interest, the 2-sigma channel, premium/discount or
counter-trend structure — a factor that merely restates the reading which fired the trigger is not a
second signal.

**Three gates keep a candidate live**, each set from measurements over the fixture candles rather than by
taste:

- **The reclaim must have happened.** A fake break up is only a fake once price is back below the high it
  took; while price is still above it, the break is in progress.
- **The raid must be recent** — within `MAX_SWEEP_BARS` (5). Unbounded, this carded raids up to 189 bars
  old while still calling them "immediately reclaimed".
- **The candidate must not already be settled** by the price that found it, checked with the same
  `resolved_by()` the settle pass uses, so detection and settlement cannot disagree. Without these gates
  35% of declared traps were resolved seconds later, most recorded as "the break was real after all"
  against a trap declared at that very price.

The invalidation sits half an ATR beyond the raid (a quarter ATR is a hair trigger — ordinary noise would
resolve a live trap within minutes), and the payoff target is the point of control, pushed out when that
sits nearer than the invalidation so a card never advertises a trade that risks more than it makes.

**Ten sub-agents, then the Head, then the desk.** The strongest candidate goes to five domain questions in
one batched request, judged by the side whose case the trap would prove — the bearish specialists hunt
bull traps and the bullish ones hunt bear traps — out of 100. The Head then makes one categorical call:
engineered trap, authentic break, or unclear. Before anything is declared the affected desk is asked, in
code, whether this is a real higher-timeframe break: **structure in the direction of the move together
with acceptance beyond value** vetoes the declaration and the trap stays a watch. Open interest building
is context only, never a veto — a crowded move is exactly what a trap looks like.

A trap is declared when the sub-agents score 60 or more, the Head calls it engineered, and the desk does
not read an authentic break.

**A trap is a market event, not a desk's opinion of one.** Each timeframe is scanned once, by the first
desk that reads it — 15m/1h/4h by Live Recon, 1d by Intraday, 1w by Weekly — and a trap then appears on
every desk that reads its timeframe. Monthly owns no timeframe of its own, because the feed stops at the
weekly candle and so Weekly and Monthly read identical candles; scanning per desk instead judged the same
shape twice and anchored two rows for one raid. Identity is side, timeframe and level, matched within a
quarter ATR rather than to the cent, because a recomputed value-area edge drifts by dollars between
refreshes.

**Cost:** a full sweep is at most six requests — three desks own timeframes, two requests each. The scan
itself is free arithmetic.

**Anchored until price settles it.** Invalidated when the break proves real, played out when price returns
inside value; settlement happens once, so a refresh cannot double-report, and nothing expires on a clock.
Declared traps warn on every affected desk and append a side note to its anchored summary.

## Active Trade desk (spec sections 3 and 4)

Enter a position and it is evaluated, pinned and tracked until you kill it. Everything the card asserts is
either measured in code or judged by an agent that was shown the evidence; nothing on it is a number a
model was simply asked to produce.

**The deterministic half** (`core/trades.py`) runs first, so a card still carries usable levels when the
model is unreachable:

- **Stop verification.** A stop is judged by what sits between it and price, not by its distance. Resting
  liquidity is swept before a level gives way, so a stop nearer than the pool below a long is one that
  gets hunted on the way to a move that then works. When the given stop is unsafe — hunted, inside a
  single bar's range, or on the wrong side of entry — the engine proposes one beyond the protective
  structure and the cluster of levels around it, and the card shows both.
- **Timed take-profits.** The price comes from the volatility bands, the time from the Fibonacci time
  zones. A band behind the entry is not a target, so it falls back to a multiple of range and says so; a
  Fibonacci window already past is not a date, so it falls back to the k-th candle close. Every target
  carries its reward in R, and a first target nearer than the stop is called out.
- **The pullback guardrail**, measured rather than assumed: the distribution of adverse excursions over
  the lookback, stated in plain English so an ordinary retracement is not mistaken for a broken thesis.
- **The monitoring schedule**: the exact candle closes to look at, in UTC.

**The agents** (`core/agents/trade_desk.py`): five bullish and five bearish sub-agents across the same
five domains, both sides asked about the same position, because a trade has a case for it and a case
against it. The score is the difference, so a position whose opposite is equally well supported reads 50 —
which is also the condition a trap is built in, and the card says so. The Head then calls it take, wait or
stand aside.

**Both handshakes are deterministic.** The four category heads are consulted from the verdicts they have
already published rather than by asking a model again, which would cost more and could contradict what is
on screen; a desk whose anchored candle has closed is silent rather than counted, and a `BULL TRAP RISK`
verdict does not count as support for a long. The TRAP head is consulted against the anchored ledger:
whether a live trap sits on the entry, the stop or any target, within half an ATR.

**The probability is a confluence estimate, clamped to 15–85%, and labelled as such on the card.** Nothing
in this system counts how often a setup like this has worked, so a number at the extremes would be a lie
about what is known.

**Cost:** two batched requests plus one Head call per judgement.

**The card is pinned in the database**, not in session state, so it survives navigation, a refresh and a
redeploy. The plan is written once at open: `Refresh` updates the live block only — current price, PnL in
percent and in R, which targets have been reached — and cannot rewrite the thesis. `Re-judge` runs the
desk again and opens a new card rather than editing the old one. `KILL` unpins it and stops its alerts.
Several positions are tracked at once, each in its own card.

## Alerts

`core/alerts.py` detects what has happened to a pinned trade — a target reached, the stop reached, a trap
developing on its levels, a monitoring checkpoint passing — and pushes it. Every event carries a dedup key
behind a unique index, which is what makes an alert fire once rather than on every 30-second auto-refresh,
and nothing is marked as sent unless a channel confirms delivery, so a refused send is retried rather than
lost.

**Telegram** and **Discord** are implemented and work as soon as `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`
or `DISCORD_WEBHOOK_URL` exist in the secrets. **Email and SMS are not wired up** — they need an SMTP
account with a verified sender, and a paid gateway, neither of which exists here; they report themselves
unconfigured rather than pretending to send.

**Alerting only runs while the page is open.** Streamlit Cloud provides no scheduled worker, so continuous
background alerting needs a cron host. That is a hosting decision, not something the code can supply for
itself, and the tab says so rather than implying the alerts are always on.

## Accuracy report

Every pinned summary is a prediction. When its candle closes it is scored once, against the price at that
exact candle boundary; a window with no completed candle on the boundary waits rather than borrowing a
neighbour's price. Scoring runs on page load for all four categories (`core/live_accuracy.py`).

The spec leaves "optimal" undefined, so the thresholds are explicit and tunable:

- a move inside the category's flat band counts as flat, so a `BALANCED` call can be right — 0.25% for 4h,
  0.6% daily, 1.5% weekly, 3% monthly; a `TRAP RISK` call is right when the move it doubted failed;
- each team's 0-100 score is read as its forecast for its own direction and graded with a **Brier score**,
  where 0.25 is what always answering 50% would earn;
- the report stays **collecting** until 10 windows are scored, then flags **calibration needed** below a
  55% direction hit rate or when either team does worse than a coin flip.

When it fires, it grades every rubric item against what actually happened and names the worst in the
spec's own categories — structural misalignment, misread liquidity, false sentiment, volatility misread,
missed traps, false trap alarms — each with a suggested change. **Nothing is changed automatically.**

## Anchored direction & the Director report

Each timeframe's displayed direction is anchored: it holds until a trigger fires (funding flips sign, open
interest moves 5 points, long/short crosses 1.0, squeeze score reaches 60, price closes past the
invalidation level, the Director's trap classification changes, a macro event is within 24h, or the anchor
is stale). When the fresh read disagrees without a trigger, the card shows it as an unconfirmed live read.
Weekly and Monthly raise early-warning banners for traps and squeezes as they form.

`Run Analysis` runs the Director pipeline on demand: ten specialists (Bullish B1-B5, Bearish R1-R5 across
options, leverage, liquidity, flow, sentiment) plus a volatility agent in parallel, each desk synthesises
its briefs, an accuracy/correlation agent cross-checks both, and the Director issues contradictions,
convergence, blind spots, probability-weighted scenarios, a trap classification and a trade plan, plus a
plain-English paragraph per timeframe. One run is 15 LLM calls; results are cached for the session.

Anchor changes log direction calls and Director reports log their classification; calls score once their
horizon elapses (Live 1h, Intraday 24h, Weekly 7d, Monthly 30d) and reports at 24h and 7d. The War Room
tab shows hit rates with sample sizes; under 10 samples is labelled indicative.

## Secrets

`.streamlit/secrets.toml` locally, Streamlit Cloud Secrets in production:

    TYPESAFE_API_KEY = "..."        # war-room rubric and trap audit (Jev); absent = Gemini fallback
    TELEGRAM_BOT_TOKEN = "..."      # optional: trade alerts
    TELEGRAM_CHAT_ID = "..."        # optional: trade alerts
    DISCORD_WEBHOOK_URL = "..."     # optional: trade alerts
    GEMINI_API_KEY = "..."          # Director pipeline, and the rubric fallback
    OPENROUTER_API_KEY = "..."      # optional alternative to Gemini
    # optional overrides
    LLM_PROVIDER = "gemini"         # or "openrouter"
    SPECIALIST_MODEL = "gemini-3.5-flash-lite"
    DIRECTOR_MODEL = "gemini-3.7-flash"
    LLM_MAX_CONCURRENCY = "4"

## Storage

SQLite at `TI_DATA_DIR` (default `./data`, gitignored): anchored verdicts and their audit log, direction
calls and report scores, per-category war-room anchors (`recon_anchors`), side notes
(`recon_side_notes`), scored windows (`recon_scores`), anchored traps (`traps`), pinned positions
(`trades`) and what happened to them (`trade_events`), squeeze signals and futures samples. `STORE_VERSION` in `app.py` is passed to the cached `_store()` call, not defaulted:
Streamlit hashes the arguments a call actually makes, so a version left as a default never reaches the
cache key and a bump does nothing.

**On Streamlit Cloud this file resets on reboot or redeploy**, so the accuracy report rarely passes
"collecting" there. Point `TI_DATA_DIR` at a mounted volume, or move to a hosted database.

## Deploying

Streamlit Cloud re-runs a freshly pulled `app.py` inside the process that loaded the previous revision, so
`core.*` and `ui.*` can stay in memory at their old versions and a newly added name fails to import until
someone reboots. `app.py` fingerprints the project sources on each run and drops those modules, plus the
caches built from them, when they have changed. `STORE_VERSION` in `app.py` should still be bumped
whenever `core/store.py` gains tables or methods.

## Mobile

Verified at 390px: the sidebar opens collapsed on phones and the reopen control stays visible (hiding
Streamlit's header removes the only way back to it), tables and the tab strip scroll inside themselves
rather than pushing the page sideways, nothing is set below 12px, buttons meet a 44px tap target, a
vertical swipe over a chart scrolls the page, and price-line titles drop on narrow screens so the axis
stops eating the candles. Dollar amounts in Markdown are escaped, because Streamlit reads `$…$` as LaTeX.

## Layout

`core/data` providers → `core/indicators` pure functions (including `smc`, `liquidity`, `fib_time`) →
`core/traps` and `core/trades` (deterministic trap and trade arithmetic) →
`core/agents` (rubric, judge, war room, profiles, trap desk, trade desk, LLM client, prompts, context,
runner, report) → `core/store`, `core/anchors`, `core/accuracy`, `core/live_anchor`,
`core/live_accuracy`, `core/alerts` (persistence, anchoring, scoring, notification) → `ui/` renderers
→ `app.py` shell.

Tabs: Live Recon, Intraday, Weekly, Monthly, TRAP Intelligence, Active Trade, War Room.
Design spec and implementation plans live under `docs/superpowers/`.
