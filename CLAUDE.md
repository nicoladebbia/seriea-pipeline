# seriea-pipeline — Serie A Betting Intelligence

32-step ML pipeline for Serie A football predictions. 6-method ensemble, XGBoost/LightGBM/CatBoost, Flask dashboard, Odds API integration.

## Architecture
- **`cli.py`** — Main CLI entry point for running pipeline steps
- **`config/`** — Configuration files
- **`features/`** — Feature engineering modules
- **`ml/`** — Model training, ensemble logic, prediction generation
- **`models/`** — Model definitions and utilities
- **`scraper/`** — Web scraping (FBRef, Transfermarkt, Sofascore)
- **`parser/`** — Data parsing utilities
- **`pipeline/`** — Pipeline step orchestration
- **`scripts/`** — Standalone scripts (scraping, data processing)
- **`storage/`** — Data storage abstractions
- **`web/`** — Flask dashboard
- **`tests/`** — Test suite
- **`data/`** — Parquet files, trained models, cache (in .claudeignore — 5GB+)

## Code Navigation — ARCHITECTURE_MAP.md — grep before code-reading

**`ARCHITECTURE_MAP.md` at project root is the per-file navigability map for all CODE** (the data equivalent is DATA_CATALOG.md). Before you Grep/Read your way through the tree to find where something lives, **grep ARCHITECTURE_MAP.md first** — it has one entry per file with: what it does, what it imports / is imported by (which function does the talking), liveness (🟢 live / 🔧 one-shot / 🧪 test), a quality grade, and "to change X, the file is Y".

- **When to consult it:** "where is X handled?", "what calls Y?", "is this file dead?", "what's the entry point for Z?", any orientation or refactor question.
- **Companion `CLEANUP_PLAN.md`** holds the kill-list, keep-list (valid one-shots — do NOT delete), merge surface, and the rule that **zero importers ≠ dead** here (scripts/ are launchd/cron/subprocess-invoked).
- **Entry points** are the 15 launchd plists + `web/app.py` + `cli.py` — listed in ARCHITECTURE_MAP.md's "Entry points" table. That table IS the command/subscription surface.
- The import/liveness facts were derived mechanically (AST + plist scan), not narrated — trust them. If the map contradicts memory, trust the map. If a file moved/was deleted after the map was generated (2026-06-01), regenerate the relevant section.

## Commands
```bash
python3 cli.py                # Main CLI
python3 -m pytest tests/      # Run tests
ruff check .                  # Lint
mypy .                        # Type check
```

## Key Facts
- **Model performance:** see `MODEL_STATUS.md` — read live from `data/models/universal/over_under/ou_*_catboost_metadata.json` (the models that place bets) and `catboost_no_odds_metadata.json` (1X2, display only). NEVER quote a hard-coded accuracy here or anywhere else.
- Per-league model separation (not one model for all leagues)
- Time-decay weighting, 2017+ training window
- Betting leaks patched (odds NOT used as input features)
- Odds backfill via historical API
- Sofascore scraper for EPL data

## Model performance — ALWAYS read metadata, never quote markdown

**Which model is "the model" (settled 2026-09-04):** the betting layer bets ONLY O/U Over
(lines 1.5 / 2.5) + Alt O/U — see the market config in `scripts/betting/betting_unified.py`.
The 1X2 ensemble feeds the dashboard, Telegram and the fantacalcio fixture card, not a
single enabled bet (1X2 betting off since 2026-04 at −20% ROI, DC dead since 2026-06).
So "how is the model performing right now?" is answered in this order:

1. Run `python3 scripts/diagnostics/print_model_status.py` — it reads the O/U metadata
   (`over_under/ou_{1_5,2_5}_catboost_metadata.json`) AND `catboost_no_odds_metadata.json`.
2. **PRIMARY — the O/U section**: holdout log-loss vs naive, calibration gap, the last
   promotion decision (`promotion.promoted` / `reason`), and realised CLV + ROI on settled
   O/U bets from the journal. An O/U model that fails its gate is the headline, not a footnote.
3. **SECONDARY — 1X2**: `cv_summary.last3_accuracy` (walk-forward, last 3 eval seasons).
   Ceiling 53–55%; anything above ~56% is leakage or fiction. It is proven at ceiling
   (2026-06-04) — do not re-run accuracy hunts.
4. If you see a markdown file claiming a number, the doc is wrong — fix or delete it.

## Dashboard match table — column rules (settled 2026-09-04)

The `/` match table (`web/templates/dashboard.html`, fed by `/api/dashboard`) is a
**betting surface for the market that is actually bet** — O/U Over 1.5 / 2.5. It is NOT a
1X2 display. Until 2026-09-04 every one of its ten columns (PRED, H-D-A, xG, EDGE, CONF,
FLAGS) described the display-only 1X2 ensemble, EDGE was 1X2 model-minus-market on the
pick, and the footer counted "+3% model edge" for a market switched off since April.
Nothing on the row drove a bet.

**The row is five columns, ranked by how often a value in it flips a decision:**

| # | Column | What it is | Source |
|---|---|---|---|
| 1 | **Bet** | The gate's verdict: `selected` > `candidate` > `near_miss` > `none` / `gated` / `no_odds` / `no_model` | `unified_bet_slip.json` (selected_bets, near_misses) + `betting_candidates.json` — **read, never recomputed** |
| 2 | **KO** | Kickoff. Timing IS the edge (>24h-early bets −5%, <24h +63%). Also disambiguates a blank pill: hours out = not scanned yet, minutes out = rejected | odds `commence_time` |
| 3 | **O/U Over** | `O2.5 63% vs 47% +15.3` — model Over prob at the bettable line vs de-vigged market (Pinnacle, else consensus, ≥3 books), raw gap | `goal_predictions.json` + totals; `web/app.py::_ou_signal` |
| 4 | **STEAM / XI badges** | Conditional, zero cost when absent | market intelligence / lineup source |
| 5 | **Match** | Row key, not a signal. Row click opens `/prediction/<slug>` which has everything else | — |

**Rules:**
- **Every column must be justified against the market that is bet.** A 1X2 quantity
  (pick, H-D-A, 1X2 edge, confidence bucket, home/away xG split) never returns to the
  main row. The prediction detail page owns them. Don't add a hidden/expandable 1X2 row
  either — it was tried the same day and cut: the row click already goes there.
- **The Bet column shows the gate's OWN output.** Never recompute an edge in the dashboard
  and label it a verdict — shrinkage, per-line bands and situational adjustments live in
  `betting_unified.py`; a recomputation drifts silently. The O/U column's raw gap is the
  INPUT and is labelled as such; the pill is the DECISION.
- **A blank pill is not "no edge".** The slip is a T-30 artifact by design (candidate mode
  does not write it), so a match >30 min out legitimately has no verdict. Label it
  "no signal in the latest scan", never "rejected". The footer shows scan age for this
  reason — keep it.
- **Never merge two orthogonal signals into one badge.** CONF (max 1X2 prob, bucketed)
  and EDGE (model − market on the pick) looked like the same thing and were not: Man City
  vs Coventry was VERY HIGH with −10.3% edge. CONF was cut because it restated the H-D-A
  bar, not because it duplicated EDGE.
- **Stale-odds muting targets classes, not column positions** (`.ou-cell`, `.bet-cell`,
  `.dash-row__badges`). `nth-last-child` selectors broke every time a column moved.
- **The table is scanned for the exception, then that row is read alone — design for
  triage, not comparison and not isolated reading.** Within a day group rows sort by
  verdict tier (selected > candidate > near miss > none > no odds/model), then kickoff;
  verdict rows are highlighted, no-signal rows recede at 55% opacity (full on hover),
  gated-league rows collapse behind a one-line "N gated · show" toggle (EPL was 7 of 10
  rows on a Saturday, all saying GATED). The O/U cell is fixed-width right-aligned
  sub-columns so the eye can run down the column. Never re-sort by kickoff alone: the
  row that matters would be buried under seven that don't.
- **Staleness is per row and per artifact, not one global age.** Header dots stay small:
  the loud states are (a) the age ON the Bet pill when the verdict predates the current
  odds fetch by >30 min ("NEAR O2.5 +6.6% · 8h", amber border), (b) the O/U cell going
  amber when kickoff is inside 2h and odds are >30 min old, (c) the 24h DO-NOT-BET banner
  that mutes the numbers. The quick strip reads the SAME ages/thresholds as the header
  dots (60m / 6h) — it used to be a third clock with a 12h bar that said "Data fresh"
  under an amber dot. The footer names BOTH verdict clocks (slip vs candidates) when they
  differ by >30 min, so a pill saying 8h never sits under a footer saying "scan 3h".
- **The page is live, not a screenshot.** Until 2026-09-04 the table rendered once and
  every "Xm ago" was a string frozen at load — a tab left open an hour still said
  "Odds: 7m ago" and the stale banner never fired. Now `renderDashboard(data)` is
  idempotent and re-runs every 30s from the cached payload (ages, pill ages, per-row
  odds check, banner all re-derive); `pollDashboard()` + `loadQuickStats()` re-fetch
  every 60s while visible and on tab focus (local JSON reads — `record_predictions` on
  the endpoint dedups by match+date, so polling is safe); changed cells and stat values
  flash amber 2s; a verdict tier change toasts; a change that would re-sort rows is HELD
  while the cursor is in the table (amber chip in the header, applied on mouse-leave or
  click). Keep it that way: any new number on the page must be re-derived on the tick
  or diffed on the poll — a static string with a timestamp in it is a lie in 30 minutes.
- **Any new column needs a rank in the table above and a reason it beats #5.** If it
  cannot flip a decision, it belongs on the detail page.
- Tests: `tests/test_dashboard_ou_signal.py` (every verdict tier, line choice, thin
  markets, gating, Pinnacle fallback, null lists in the slip).

## In-play paper engine (built 2026-09-05) — measured first, money never

Nicola's ask: "Atalanta went up, put €10 on Roma winning." Measured on the Roma–Atalanta
price path before building: the Odds API in-play feed lags the pitch by minutes (the first
snapshot after a goal already carries the repriced line), Pinnacle does not reprice in-play
through it, the in-play overround is ~7% vs 5% pre-match, and there is no execution API. So
`scripts/betting/inplay.py` is a PAPER engine: `goal_process.simulate_from_state` prices the
match from the score + minute on the board (baseline = the pre-match MARKET via
`market_profile`, deliberately not model xG, so the question is whether the conditioning
beats the books' repricing given the same information), `live_monitor.poll_once` prices
every live snapshot (`snapshot.fair` / `fair_totals` / `best_edge`), takes one paper pick per
selection per match after a score change (1X2 only; edges above the journal's 12% cap are
counted, never journaled) into `data/betting/inplay_journal.json`, and settles at the whistle
with CLV against the NEXT snapshot — the price a human could actually have taken.

- **No hand-set thresholds (2026-09-05).** A pick must clear the snapshot's OWN overround
  (`sum(1/odds) − 1`, the price of betting into that book at that moment; measured median
  6.9%, p10–p90 4.8–8.0% on 1,536 in-play snapshots) plus 1.96 × the Monte Carlo standard
  error of the fair probability (`sqrt(p(1−p)/n_sims)`). That replaces `EDGE_MIN` / `FAIR_MIN`
  / `MAX_MINUTE`: a late state needs no minute cut-off because the fair price goes to 0/1
  and the edge vanishes; a rare outcome needs no probability floor because its interval is
  what it is. `pick.floor` / `pick.margin` on every pick say what it had to beat. The fair
  price is first shrunk toward the market by `shrink.w_latest` in the backtest file — a
  weight fitted WALK-FORWARD (Brier on earlier matchdays only, never the one it is applied
  to, None below 200 rows); `skill_blend_vs_market_walk_forward` next to
  `skill_raw_vs_market_same_rows` says whether shrinking helped (first run: w stayed 1.0
  through the season, the two numbers are equal). The backtest re-runs after every
  settlement (`auto_settle`, next to `settle_picks`), so the live hook always reads the
  latest weight and gate.
- **The stand-in baseline window is measured too.** A match with no pre-match line (122 of
  156 stored entries before the same-day fix) takes its first 0-0 snapshot as the line only
  inside `baseline_fallback.window_min` in the backtest file: `measure_baseline_window`
  tracks mean |in-play − pre-match| by minute at 0-0 on the matches that have both, and
  the window ends the minute before the drift first exceeds the simulator's own Monte
  Carlo noise on a coin-flip (`mc_se(0.5, N_SIMS_LIVE)` = 0.0065). First measurement, 34
  Serie A matches: drift ≤ 0.005 through 10', 0.007 at 11' → window 10 (the old literal,
  now re-derived every backtest; `BASELINE_WINDOW_SEED` is only the no-file fallback).
- **Only Serie A is priced (`PROFILE_LEAGUE`).** EPL has no fitted goal-process profile; until
  2026-09-05 live entries carried no `league` and 33 EPL matches were scored with the Serie A
  hazard, red-card multipliers and calibration (skill 0.008 mixed → 0.018 Serie A-only on 44
  matches). The monitor now stamps `league` on every entry (Odds API sport key, else
  `infer_league`); an EPL snapshot gets `inplay_note` and no fair price.

- **Read the verdict from `data/models/inplay/backtest.json`, never from a doc.** The gate is
  skill vs the in-play market's own 1X2 probabilities (≥ 0.02 on ≥ 200 snapshots) AND paper
  ROI at the NEXT snapshot's price. The first two runs of 2026-09-05 FAILED it (skill went
  −0.001 → +0.008 once red cards and the closing-line baseline landed; still short), so
  `inplay_pings` defaults OFF (the /live "In-play" select turns the Telegram ping on) and
  the paper record keeps growing regardless. The Roma case itself was negative value: fair
  P(Roma) from 0-1 at 81' = 3.4% vs market 10.0. The file also carries `model_variant` —
  the same engine on our archived pre-kickoff xG instead of the market split — and on the
  first run it was WORSE than the market baseline on every number; the market baseline
  stays primary until that flips.
- **The gate number carries a confidence interval — read it before arguing with the point
  estimate.** `skill_ci95_by_match_bootstrap` (matches are the clusters; snapshots of one
  match are not independent) spanned roughly −0.05..+0.07 on the first 64 matches: the
  sample cannot tell the simulator from the market in either direction. More matchdays,
  not more modelling, is what narrows it. `skill_pickable_window_le85` is the same number
  on the minutes the engine can act in (the after-85' bucket is where the old wall-clock
  minute was most wrong); the gate stays on the full window.
- **Snapshots carry the ESPN clock and score while the fast tick is fresh** (`score_source:
  espn_fast`, `clock`, `added_time`; `live_monitor.fast_state_for_snapshot`). Before this the
  minute was wall-clock time since the listed kickoff with a fixed 15-min interval, and the
  score lagged the pitch by minutes — a fair price for the wrong state. Stored snapshots
  from before 2026-09-05 still have the estimated minute; the backtest inherits that noise.
- **Red cards ARE modelled, from a measurement, not a guess.** `goal_process
  --measure-red` writes `red_mult` into `profile.json`: goals after the FIRST red of a match
  over what the league's per-side per-bin rate expected in the remaining bins (Serie A
  2017-26, n=606, ten-man side ×0.62, opponent ×1.53, split-half 0.64/1.71 vs 0.67/1.52).
  `simulate_from_state(red=(h, a))` applies it per card; the live engine reads the cards
  off `live_events` (`reds_at`). A refit (`--fit-profile`) keeps the measured `red_mult`.
- **The totals lines in a snapshot are NOT live prices** — they are the pre-match lines the
  feed carries along (the first backtest "found" Over 0.5 @ 2.32 in the 84th minute at 1-0).
  Totals fair prices are computed for the card and never picked.
- Baseline order: `pre_match_odds`, else the last pre-kickoff line in `data/odds_snapshots/`
  (`live_monitor._closing_line_from_snapshots`), else the first ≤10' 0-0 snapshot; a match
  with none of those is skipped and counted (`matches_without_baseline`).
- Re-run after more matchdays: `python3 -m scripts.betting.inplay --backtest`. Promotion to
  real stakes goes through the same bar as every other paper market (`market_promotion.py`),
  never by hand. Tests: `tests/test_inplay.py`.

## Goal-process simulator (settled 2026-09-05) — what it is and is not

`scripts/models/goal_process.py` samples minute-resolved goal paths (92 bins, league hazard,
score-state multipliers) and prices every market that depends on WHEN a goal happens or on a
LEAD existing at any moment: Vince o quasi, 1° tempo, Primo tempo / Finale, Prima squadra a
segnare, Minuti, 2° tempo under/over, Under/over 5.5 / 6.5. Rows reach the page through
`/api/match-markets/<slug>` (`served_rows` → `web/match_markets.py`, where a simulator row
REPLACES the independent-Poisson artifact row for the same bet).

- **The total is NOT the feature-frame xG sum.** Measured on 2023-26 (n=1,135):
  corr(xg_home+xg_away, goals) = 0.06, corr(xg_home−xg_away, goal diff) = 0.37. The
  Poisson xG carries the split between the sides and nothing about the total; fed raw it
  lost to the base rate on every totals market (over 2.5 skill −0.068). The profile
  therefore takes the SPLIT from xG and the TOTAL from a one-slope regression (≈ base rate),
  and in serving the O/U blend's P(over 2.5) rescales the total (`calibration_k`). Don't
  "fix" a totals market here by touching the xG: the O/U CatBoost is the total model.
- **Tiers come from `data/models/goal_process/backtest.json`, read at request time**
  (walk-forward: profile fitted on seasons before the test season, 2023-24 → 2025-26,
  gate skill ≥ 0.02 with ≥ 200 events). Re-run with
  `python3 -m scripts.models.goal_process --backtest` after a timeline refresh; never quote
  a skill number from a doc. At the first run, lead-based markets passed (home/away win,
  all four Vince o quasi, first team to score, 1° tempo 1, H/H) and everything timing-only
  (goal 0-15', 76-90', stoppage, 1° tempo under/over, both halves) sat within ±0.01 of
  zero: the hazard is league-level, so a timing market has no per-match information
  beyond the total. That is a tier B by measurement, not a bug.
- **First-half Goal and first-half double chance (added 2026-09-05)** are simulator rows
  (`1h_btts`, `ht_dc_1x/x2/12`) priced by the pick engine against the `btts_h1` /
  `double_chance_h1` feed that was fetched and unread until then. Measured on the same
  walk-forward (n=1,140): `ht_dc_x2` passes (skill +0.035, it is the complement of
  `ht_home`), `ht_dc_1x` / `ht_dc_12` sit at +0.009 / +0.001 (tier B), `1h_btts` at
  −0.0003, the timing-only pattern again. Paper only, like every LEAN.
- **1x2 finale and Over 2.5 are never served from the simulator** (`NOT_SERVED`): the
  ensemble owns 1x2 (the simulator's draw FAILS the gate, skill −0.022, the same
  independent-Poisson draw deficit the World Cup Dixon-Coles attempt could not fix) and
  the O/U model owns 2.5.
- The timeline holds BOTH leagues; `load_universe("serie_a")` scopes the fit. The first
  fitted profile was on the mix (caught the same session). EPL has no served profile.
- **`calibration_k` saturation is loud, not silent** (`K_BOUNDS = (0.25, 4.0)`): a served
  P(over 2.5) outside the reachable range pins k to the bound, logs a warning and stamps
  `calibration_saturated: true` on every row and on `/api/projections` `goal_process_meta`.
- **VAR markets are priced from Sofascore `varDecision` rows** (Speciali match, tier C).
  The incidents parser dropped them until 2026-09-05; `python3 -m scraper.sofascore_events
  --var-backfill 4` re-fetches the last four Serie A seasons (resumable: a match gets a
  `var_checked` marker row when it has no VAR incident, so VAR rates divide by CHECKED
  matches only). Semantics, verified on disk: `incident_class` is the ON-FIELD decision
  under review and `confirmed=False` means it was OVERTURNED (goalAwarded+False → 0/18
  had the goal in the goal list; penaltyNotAwarded+False → 10/12 followed by a penalty
  goal). So "Gol annullato" = goalAwarded+False, "Rigore VAR" = penaltyNotAwarded+False.
- **Rare events are tier C because conditioning was MEASURED dead, not because it was
  skipped.** `rare_event_conditioning` (reruns inside every `--rare-events`) walks forward a
  shrunk referee / home / away rate for own goal, red card, first-minute goal, bench goal,
  penalty, and the VAR markets: on Serie A 2023-26 every event, every conditioning scored
  ≤ +0.002 skill (referee on penalties), most negative. More matches tighten the base rate;
  nothing per-match moves it. The rebuild logs a WARNING the day a conditioning passes the
  gate — that is the signal to build a per-match row, not a hunch. Don't re-run the hunt
  by hand; read `rare_events.json["conditioning"]` after the next backfill instead.
- **Predicted starters are priced through a MEASURED start calibration (2026-09-05).**
  `lineup_predictor`'s `start_pct` is overconfident in every bucket: on 3,807 archived
  predictions (Feb–Sep 2026, `data/lineup_history/predictions_*.json` joined to the next
  played fixture) a 'certain' 99.7% started 82%, a 'likely' 75% started 51%, the predicted
  XI as a whole 92% → 74%. `data/models/player_floors/start_calibration.json` holds isotonic
  knots (Brier 0.213 → 0.170, base 0.249); `calibrated_start_prob` reads them and every
  predicted-XI starter is priced as P(starts)·P(market|starts) + (1−P)·P(market|20' sub)
  (`_mix_start_sub`, stamped `start_mix`/`start_prob`; the /picks card shows this number as
  `XI prob.`). A confirmed sheet bypasses all of it. Refit after a few matchweeks:
  `python3 -m scripts.betting.player_predictions calibrate-start` (needs ≥500 joined rows;
  the archive is written by every lineup_predictor run). The same overconfidence applies to
  every other consumer of `start_pct` / `status: certain` (fantacalcio p_play) — not yet
  corrected there.
- **Player per-half split (E3, `scripts/betting/player_predictions.py`)**: expected events
  in a half = per-90 rate × minutes on the pitch in that half (starter 45/rest, sub
  late) × the league's timing share. Measured before building: a player's OWN 1st-half
  shot share has std 0.058 across 322 players with ≥80 shots vs 0.056 binomial noise —
  no per-player timing term, it would be fitted noise. Shots / SoT halves are backtested
  (`validate-halves` → `data/models/player_floors/halves_backtest.json`, walk-forward
  2023-26, n=20,964 starter-matches; the served tier reads it). Fouls, tackles, duels,
  passes, interceptions have NO minute stamp in any catalog file: their split uses a
  flat 50/50 timing share, is stamped `timing: "flat"` and served as tier C. Never
  promote a flat split to A/B without a minute-stamped source.

## Pick engine (settled 2026-09-05) — every match gets a line, money only on VALUE

Nicola's decision, after the journal said "most probable" is not "mispriced" (1X2 −79 on 23
bets, BTTS −28 on 5, O/U 1.5 +47 on 47): **"Pick every match, real money only on VALUE."**
`scripts/betting/picks.py` writes `data/upcoming/picks.json` with one line per upcoming
Serie A match; `/picks` on Telegram, the `pick` field of `/api/dashboard` (a dashed LEAN
pill only where the gate said nothing) and the banner + `@odds (edge)` chips on
`/prediction/<slug>` all read it.

- **VALUE** is the engine's own verdict (slip `selected_bets`, or a morning candidate
  that commits at T-30), copied from the slip. **Never recompute it here.**
- **LEAN** is the best positive-edge angle across every row `build_match_markets` serves
  (ensemble 1x2, O/U blend, Poisson artifact, goal-process simulator, player floors incl.
  goalscorer/assists) against a REAL price. Ranking is `in band > tier A/B/C > multi-book
  > edge`; rows with model probability < 20% (a "+9%" on a 3% event is inside the model's
  own error) and edges above the 12% cap (>10% ran 38% WR live) are flagged and sink.
  With 40+ priced rows per match the biggest edge is a max over noise: the headline is
  the most MEASURED angle, not the largest number. Paper-journaled at a flat €10 in
  `data/betting/picks_journal.json` ONLY on the T-30 path (`save_bet_slip` non-dry) and
  only for kickoffs inside 3h. When the LEAN is a bet the real engine priced and
  rejected, the line carries the engine's reason (`engine_note`): its edge (shrunk,
  Pinnacle de-vigged, per-line band) is the one that counts for money.
- **NO EDGE** shows the most probable priced outcome with its price and why it is not a
  bet. A match with no per-event prices yet says so.
- **Insolite (exotic slot)**: per match, up to three positive-edge angles OUTSIDE the
  mainstream markets (1x2 finale, Under/over, Doppia chance, Goal): player props, first
  half, HT/FT, first team to score, exact score. When none beats its price, the most
  probable priced player prop is shown with the price that beats it (`exotic_fallback`).
  The headline LEAN and the best exotic are both paper-journaled. The slate feeds match
  rows AND player rows to the pick: the first live slate priced only `markets` and never
  saw a player prop — the fallback showing a first-half row instead of a player is what
  exposed it (2026-09-05).
- **Prices**: `odds_full.json` (h2h, totals) + `odds_extra_markets.json` (btts, DC, DNB,
  alt totals) + `data/upcoming/pick_markets_raw.json` from
  `odds_fetcher.fetch_pick_markets` (per-event `h2h_h1`, `totals_h1`, `btts_h1`,
  `halftime_fulltime`, `double_chance_h1`, corners/cards totals, `correct_score` and the
  eight `player_*` props; specimen-verified 2026-09-05 on Juventus vs AC Milan, region eu;
  `alternate_team_totals` / `h2h_h2` / `totals_h2` are NOT served and are deliberately
  absent — a 422 on the whole request still costs credits). 10 credits per event (was 16; six unread markets dropped 2026-09-05), gated by
  `check_budget_pacing(PRIORITY_EXTRAS)`, 45-min refresh per event; called at the T-6h /
  T-3h odds stages and at every pre-kickoff cycle (only the first of each window pays).
  `team_totals` / `alternate_team_totals` were probed eu+uk on 2026-09-05: zero books, so
  the 96 "Gol Casa / Gol Ospite" rows stay unpriced. Every key in `PICK_EVENT_MARKETS`
  must have a consumer in `price_key_for_row` (test enforces the set): a market fetched
  for a row nothing maps to is a credit burnt per event per cycle.
  Player names are joined per market with an accent-folded token match that returns None
  on ambiguity (`_match_player`): a wrong player is worse than no price. Card props are
  fetched but never priced or journaled — `player_match_stats.parquet` has no card column,
  so they could not be graded.
- **Grading** (`settle_picks`, called by auto_settle after the paper track): full-time
  markets from the results dict, first-half markets from `goal_timeline.parquet` (canonical
  `{date}_{Home}_{Away}` id, a match present with no 1H goal is 0-0), player props from
  `player_match_stats.parquet` (date + team + accent-folded name). Anything else stays
  pending and is COUNTED once per run, never warned per bet. `picks_record()` is the
  per-market bar: a market earns real stakes the way O/U 1.5 and the EPL gate do — a
  settled paper record with CLV, not a good week.
- Journal entries carry an `extra` dict (bet_type, player, team, source, tier, side,
  line, **lineup, start_pct** — the XI basis at journal time, so the paper record can be
  split confirmed vs predicted before any market earns real stakes) — `bet_journal.add_bet`
  stores it; real bets have `extra: null`.
- **A player who never entered is VOID, not pending** (books void the prop): once both
  squads' stats are on disk (≥ 22 rows for that date, `MIN_STAT_ROWS_FOR_DNP`) a journaled
  player with no row, or a row with 0 minutes, settles `voided`. Until the stats land the
  pick stays pending — and `health_check.check_player_stats_coverage` goes CRITICAL when a
  Serie A match finished > 14h ago has no player stats on disk (the Sofascore API was
  challenged on 2026-09-05 afternoon after working at 08:00; the evening ingest fails
  silently in that state).
- **Promotion gate (2026-09-05, `scripts/betting/market_promotion.py`) — a paper market earns
  real stakes by its settled record, never by hand.** Nicola wanted the props (shots, SoT,
  goalscorer, first-half angles) bet for real; the measured record said props −54% at real money
  and zero settled paper picks. So the gate decides: ≥ 50 settled paper bets, ROI > 0, z ≥ 1.0,
  CLV > 0 once ≥ 20 closing prices exist (`PROMOTION_BAR`). A promoted market's LEAN is mirrored
  into the REAL journal at Kelly × 0.5, cap 1.5% (`journal_promoted`, `pipeline_status:
  pick:promoted`, `extra.picks_ref`) and settled by `settle_picks` with the paper entry
  (`settle_linked`) — `results_fetcher.settle_bets` SKIPS `picks_ref` entries because its
  full-time grader defaults an unknown market to "lost". Demotion: ≥ 30 real bets at ROI < −10%
  or z < −1 → paper, and the paper count restarts (`record_from`). State:
  `data/betting/market_promotion.json`, rewritten after every settle; `/record` on Telegram and
  the Monday 09:00 digest render it. Paper CLV: `journal_lean` no longer writes 1/odds as the
  sharp prob (every paper CLV was a fake 0.0); the grader passes the feed's last price as
  `closing_odds` (`closing_price_for`), so CLV exists only where a closing price does.
  **Do not lower the bar to hit an income target and do not promote a market by editing the
  state file** — the gate is the product.
- **The T-30 run re-reads the code every time — no kickstart needed for `picks.py` /
  `betting_unified.py`.** The pre-kickoff monitor is a launchd `StartInterval 900` job (not
  a long-lived process; `launchctl list` shows PID `-` between cycles) and the T-30 itself is
  a child `run_full_pipeline.py --pre-kickoff` with `capture_output=True`, so its log lines
  live in `logs/pipeline.log`, NOT in `launchd-pre-kickoff-monitor-err.log`. The earlier
  claim that the first T-30 of 2026-09-05 "needed a kickstart" was wrong: measured from
  `pipeline.log`, the Roma–Atalanta T-30 ran at 14:10 EDT and the pick-engine hook was
  committed at 14:35. `health_check.check_picks_journal_activity` (WARNING) now says when a
  Serie A kickoff passed with nothing paper-journaled.
- **Every in-band angle is paper-journaled at T-30, not just the headline** (2026-09-05
  evening): `_journal_candidates` = headline, best exotic, then every alternative / exotic
  with `in_band`; the promotion bar is per MARKET and a market that only ever took the second
  slot could never reach 50 settled. The journal dedup is scoped to the market for picks
  (`add_bet(dedup_by_market=True)`): "1" is both the full-time and first-half home win,
  "Dimarco Over 0.5" is shots and shots on target — the fan-out test caught the full-time 1x2
  being blocked as a duplicate of the first-half one.
- Tests: `tests/test_picks.py` (specimen naming, row→price map, name join incl.
  ambiguity, ranking, VALUE-from-slip, engine note, journal dedup across players, every
  grading family, ungradable stays pending, fetch refresh window — the last one caught a
  real bug: the shared due-selector still read the scorer constant).

## Cleanup discipline — CRITICAL

This project accumulates abandoned experiments (`*_v2.py`, `*_hotfix.py`, `_phase3a_*`, scratch JSONs, stale baselines). The rule:

- **Edit existing files. Don't create new ones.** If a fix needs `predict.py`, edit `predict.py` — don't make `predict_v2.py`.
- **If you must branch (e.g. trying an alternative approach), delete the abandoned branch in the same session.** No leaving `*_v2`, `*_old`, `*_test`, `*_scratch`, `*_draft`, `*_tmp`, `*_hotfix` files in the tree.
- **Files with experimental qualifiers in their names must be either renamed to production names (qualifier removed, code wired up) OR deleted.** No third option.
- **Heuristic:** if a file would not pass code review tomorrow as production code, delete it tonight.
- **One-shot scripts (migrations, backfills, ablations) should be deleted after they run successfully.** The git log preserves them if you ever need to re-derive.

## Conventions
- Strict typing (mypy enforced)
- DuckDB for data processing, Parquet for storage
- Config-driven pipeline steps
- All data transformations in features/ directory
- ruff + black for formatting

## Data Reference — DATA_CATALOG.md — MANDATORY

**`DATA_CATALOG.md` at project root is the AUTHORITATIVE reference for every data file in this repo.** It is the single source of truth — check it before guessing, before reading code, before searching.

### When you MUST consult DATA_CATALOG.md (not optional)

Any question or task involving:
- **A specific file or parquet** (`matches.parquet`, `features_serie_a.parquet`, `player_stats.parquet`, `shotmap_stats.parquet`, `understat/*`, `sofascore/*`, etc.)
- **A specific column** (`poisson_home_xg`, `home_elo`, `ref_strictness_score`, any `ss_roll_*`, `fb_roll_*`, `odds_*`, `weather_*`, etc.)
- **Where data comes from** / which scraper writes it / how it's refreshed / what API feeds it
- **How data is joined across sources** (Sofascore shot events → canonical match, Understat xG → matches.parquet, FBref hash → date-based match_id)
- **What Plan A/B/C exists** if a source fails / what fallbacks are wired
- **The auto-refresh schedule** (daily morning pipeline, weekly Monday 04:00 plist, per-step)
- **What the 38 feature-pipeline steps produce** (which step writes `elo_diff`, `poisson_prob_H`, `ref_avg_yellows`, etc.)
- **NaN rates, fill percentages, known gaps** (Pinnacle odds 65%, weather 75% 2025-26, pre-2017 ref data, etc.)
- **Deprecated or broken files** (`shots.parquet` FBref-only through 2024-25, legacy Understat files in `_deprecated/`)

### How to consult it

1. **Open DATA_CATALOG.md first.** Do not skim — grep for the specific column/file/concept.
2. **Quote the relevant section** in your response so the user can verify.
3. If DATA_CATALOG.md contradicts something you remember, **trust the catalog over memory** — it was generated from the actual data.
4. If the catalog doesn't answer the question, **say so explicitly** before falling back to code-reading or web search.

### What's in it (16 top-level sections)

- **Data flow architecture** — ASCII diagram of scrapers → parsed → features → predictions → bets
- **Dataset inventory** — 17 canonical files with rows, cols, status, refresh timestamps, 2025-26 coverage
- **Ground truth** — matches.parquet deep description
- **Features** — features_serie_a.parquet (1,059 cols from 38 pipeline steps)
- **FBref / Sofascore / Understat / Weather / Referees** — per-source deep docs
- **Cross-source mapping** — match_id_mapping.parquet (FBref hash ↔ Sofascore id ↔ Understat id ↔ canonical)
- **Auto-refresh infrastructure** — what runs when, which plists, which scripts
- **Fallback matrix** — Plan A/B/C per source with rating
- **What's broken or partial** — known gaps and their mitigations
- **Column glossary** — 20 feature families explained (elo, poisson, rolling, h2h, odds, weather, ref, ss_roll_*, fb_roll_*, understat, etc.)
- **Join recipes** — 8 concrete code snippets for cross-source joins
- **Feature provenance** — every one of 38 pipeline steps mapped to the columns it writes
- **Per-file column audit** — 55+ files with per-column table: dtype, filled%, NaN%, unique, sample, min, max

### After making data changes

Every time you **refresh, scrape, restructure, or backfill data**, update DATA_CATALOG.md so the catalog stays authoritative. The file is deliberately at project root (not `.plans/`) because it's permanent reference, not a plan artifact.

**If I ask a data question and you don't cite DATA_CATALOG.md in your answer, you're breaking this rule.**

## Operational Bug Catalogue (2026-05-01 deep-fix session)

This section captures every real bug we found, why it existed, and the rule
that prevents it from recurring. **Future Claude: when you see a symptom that
matches one of these, the fix is documented; do not waste time re-diagnosing.**

Organised by *symptom-first* so you can grep for what you're seeing:

### Symptom: "All launchd plists in `~/Library/LaunchAgents/` look stripped (bare arrays, no schedule)"

- **What you'll see**: `cat ~/Library/LaunchAgents/com.seriea-pipeline.X.plist` returns one line `["python3", "...path..."]` — no `<plist>`, no `Label`, no `StartInterval`.
- **Why it happens**: Some macOS tool / linter / plutil-write rewrote them to compact form. Runtime keeps the schedule in launchd memory until reboot, then they're lost.
- **Fix**: regenerate from a healthy XML template, preserving `ProgramArguments`. See `/tmp/regen_plists.py` history (it's been run; pattern is in this CLAUDE.md).
- **Detection**: `for p in ~/Library/LaunchAgents/com.seriea-pipeline.*.plist; do head -c 1 "$p"; echo " $p"; done` — first char `<` = OK, `[` = stripped.
- **Prevention rule**: **never `plutil -convert json` a launchd plist**. Always edit XML form, atomic-write via tmpfile.

### Symptom: "Cron job exit code 1 with NameError"

- **What you'll see**: `launchctl list | grep seriea-pipeline` shows a job with exit 1, log has `NameError: 'today_matches' is not defined` (or `train_results`, etc.).
- **Why**: Variable scope mistake — function references a name that was defined in a sibling function (copy-paste rot).
- **Fix**: rename to the local-scope variable that exists, or guard with `try/except NameError`.
- **Affected this session**: `scheduler.run_pre_kickoff_monitor` (today_matches → horizon_matches), `weekly_retrain.full_retrain` (train_results → fallback to selected_features).
- **Prevention rule**: when copying a function body, run `python3 -c "from X import Y; Y()"` to catch NameErrors before the cron does.

### Symptom: "Telegram bot stopped 8 days ago, won't restart"

- **What you'll see**: `ImportError: cannot import name '_load_json' from 'web.advisor'`.
- **Why**: `advisor.py` was refactored to use `load_json_safe` from `scripts.utils.json_utils`, but `telegram_bot.py` still imported the old `_load_json` name.
- **Fix**: alias the new function as the old name in `telegram_bot.py`:
  ```python
  from scripts.utils.json_utils import load_json_safe as _load_json
  ```
- **Prevention rule**: **when you remove or rename a function exported from a module, grep for its name across the project** before deleting:
  ```
  grep -rln "from web.advisor import.*_load_json" .
  ```

### Symptom: "`/api/data-freshness` says odds_fetch_staleness 179h but odds_full.json was just refreshed"

- **What you'll see**: monitor reports odds stale, but `data/upcoming/odds_full.json` mtime is recent.
- **Why**: `_iso_age_hours()` in `scripts/pipeline/monitor.py` mixed naive/aware datetimes:
  - `datetime.fromisoformat("2026-05-01T04:30+00:00")` → tz-aware
  - `datetime.now()` → tz-naive
  - subtraction raised TypeError, caught, returned -1
- **Fix**: normalize both to UTC-aware:
  ```python
  if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
  return (datetime.now(timezone.utc) - dt).total_seconds() / 3600
  ```
- **Prevention rule**: **all timestamps in this repo must be written as UTC-aware ISO strings**. Any reader that parses them must use `datetime.now(timezone.utc)` not `datetime.now()`.

### Symptom: "`fetch_and_save_odds()` succeeds but health-monitor still reports staleness"

- **Why**: the fetcher writes the cache files but doesn't update `data/pipeline_state.json:last_odds_fetch`. Monitor reads that field.
- **Fix**: after `save_odds()`, write `state["last_odds_fetch"] = datetime.now(timezone.utc).isoformat()`.
- **Prevention rule**: **whenever a write succeeds, bump the state field that tracks freshness**. State files exist precisely to drive monitors; not updating them means silent rot.

### Symptom: "Bankroll says X, journal-derived says Y, drift > 0"

- **What you'll see**: `data/monitoring/health_status.json` reports `ledger_invariants CRITICAL: Ledger drift detected`.
- **Source of truth ranking**: `bet_journal.json` (immutable append-log) > `history.json` (settled-log cache) > `bankroll.json` (live snapshot).
- **Fix**: recompute snapshot from journal:
  ```python
  d = json.load(open('bet_journal.json'))
  settled = [b for b in d['bets'].values() if b['status'] in ('won','lost','push','void')]
  total_profit = sum(float(b.get('profit', 0) or 0) for b in settled)
  current_balance = 1000.0 + total_profit
  ```
  Then update `data/betting/bankroll.json` and (separately) append the new settlements to `data/betting/history.json` if they're missing.
- **Prevention rule**: **never edit `bankroll.json` or `history.json` directly when settling bets**. Only edit `bet_journal.json`; the snapshots derive from it. If a snapshot drifts, recompute, don't patch the snapshot in place.

### Symptom: "Daily Odds API spend spikes after Mac wake/launchctl reload"

- **What you'll see**: `logs/launchd-morning-err.log` and `logs/launchd-evening-err.log` both show `STARTING SCHEDULED PIPELINE RUN` at the same second, at a non-scheduled time (e.g. 00:24:18 right after a wake event). Each duplicate fires a full `fetch_and_save_odds` for SA + EPL costing ~226 cr — burned for nothing because the data hasn't moved.
- **Why it happens**: Both `morning.plist` and `evening.plist` have `RunAtLoad: true`. When launchd re-loads jobs (Mac wake from sleep, `launchctl reload`, login), every `RunAtLoad` job fires immediately regardless of `StartCalendarInterval`. Two of them = duplicate pipeline run.
- **Fix layered, in priority order**:
  1. **`CACHE_DURATION_MINUTES = 60`** in `scripts/data/odds_fetcher.py` (was 10). Per-event extras and bulk markets don't move enough pre-kickoff to need a 10-min refresh. T-5min closing snapshots bypass the cache because `fetch_tagged_snapshot()` passes `use_cache=False` — the cache is gated on `use_cache`, **not** on `critical`.
  2. **`use_cache=True`** on the non-critical `fetch_and_save_odds` callers in `run_full_pipeline.py` (the parallel path at ~988 and the sequential path at ~1282), and on `fetch_league_odds` at ~1590. The `run_incremental` path at line 616 already gates by `needs_odds_refresh(max_age_hours=4.0)` so leave that `use_cache=False`.
  3. **Optional plist tweak** (only if (1)+(2) prove insufficient): drop `<key>RunAtLoad</key><true/>` from `morning.plist` + `evening.plist`. `StartCalendarInterval` still catches up on next wake if the scheduled time was missed during sleep.
- **Status (verified 2026-08-01):** layer 2 is live (`run_full_pipeline` lines ~992/1286/1594 pass `use_cache=True`); **layer 1 was NOT live** — `CACHE_DURATION_MINUTES` was still `10`, the change having been left in `stash@{1}` and never applied. Now applied. Safe for closing lines: the cache is gated on `use_cache`, not `critical`, and `fetch_tagged_snapshot()` passes `use_cache=False`.
- **Prevention rule**: **all `fetch_and_save_odds` callers default to `use_cache=True` unless they have an upstream freshness check.** Same for `fetch_league_odds`. The cache exists precisely to absorb wake-storm duplicates.

### Symptom: "Auto-poll burning credits with no live matches"

- **What you'll see**: `Auto-poll: no live matches (N/12)` in `launchd-web-dashboard-err.log`, but `is_match_day()` returned True hours before kickoff. Each poll = 2 Odds API credits.
- **Why**: `is_match_day()` is too lax — returns True for the entire calendar day. Page visit triggered `_ensure_auto_poll()`.
- **Fix**: only auto-start when a match is **imminent** (within 30 min of kickoff or already live). Bail out after 4 empty polls, not 12.
- **Prevention rule**: **never auto-poll based on calendar day alone**. Always require a kickoff-time check. Default bail-out for empty polls = 4 (20 min), not 12 (60 min).
- **Arming (2026-09-05)**: `web/app.py::_live_window_open` is the ONE gate (T-5 to T+150 per
  fixture, minus fixtures the loop already saw finished via `_live_stopped_at`), read by the
  `/api/live` visit path, and by `_live_arm_loop`, a boot thread that checks it every 60s.
  Before that the loop armed only at boot (`is_match_day()`, calendar day: 4 wasted polls at
  04:00, then nothing) or on a page visit: the Roma–Atalanta pings of 2026-09-05 existed only
  because a /live tab happened to be open. The ESPN fast tick (`refresh_live_fast`, free) runs
  inside the same thread, so goal pings depend on this arming.

### Symptom: "Sofascore API blocks (HTTP 403) but I need fresh data"

- **FIRST: re-measure the ban before believing it.** A recorded ban is a *snapshot*, not a
  standing fact — they lift in hours-to-days. Probe with `curl_cffi` + `impersonate="chrome124"`,
  **not plain `curl`**: plain curl gets 403 from Sofascore *even when nothing is banned*
  (that's their normal always-on TLS-fingerprint protection), so a plain-curl 403 proves
  nothing. Measured 2026-07-16: api + www both **200** with live data via curl_cffi while
  plain curl still 403 — the June ban was gone, and two modules deferred as "403, do not
  build" (`live_sofascore`, `sofascore_watcher`) were buildable all along. One cheap probe.
- **TWO different 403s — tell them apart before doing anything, they have opposite fixes.**
  Measured 2026-08-25 from a university network (egress `131.94.x.x`, an institutional
  NAT): **every** Sofascore path 403'd — `www` HTML, `www` API, `api.sofascore`, and
  also `/robots.txt` and `/favicon.ico`. A static asset 403 cannot be a rate limit you
  earned, so that is the tell. Body is `{"error": {"code": 403, "reason": "Forbidden" }}`
  with **`server: Varnish`** — Sofascore's own edge denying the IP wholesale, NOT the
  Cloudflare fingerprint ban described below.
  - **Blanket-IP deny** (robots.txt 403, `server: Varnish`): the egress IP is denied.
    The HTML fallback DOES NOT help — it 403s too, so the page-tier map below is moot.
    Not a school firewall either: verify by curling any Cloudflare-fronted host, which
    returns 200. **Fix: change network** (hotspot/VPN). Waiting does nothing.
  - **Cloudflare fingerprint ban** (api tier 403, `www` HTML still 200): the documented
    case below — burn the cooldown, scrape the HTML.
  - Detection one-liner:
    `python3 -c "from curl_cffi import requests as r; x=r.get('https://www.sofascore.com/robots.txt',impersonate='chrome124'); print(x.status_code, x.headers.get('server'))"`
    → `403 Varnish` = blanket IP deny; `200` = IP is fine, problem is elsewhere.
  - While blanket-denied, `football-data.co.uk` still serves results (verified 200 the
    same minute) — that is the working third source for scores, not Sofascore HTML.
  - **Third shape, measured 2026-09-05 (Roma–Atalanta, T-39 lineup fetch):** `robots.txt`
    **200** (so not the blanket deny) while BOTH `api.sofascore.com/api/v1/event/<id>/lineups`
    and `www.sofascore.com/api/v1/...` answer `403 {"error":{"code":403,"reason":"challenge"}}`,
    `server: Varnish`. A JS challenge on the API tier only; every impersonation profile gets it.
    The lineup chain then produced nothing — and the first diagnosis of why was WRONG
    (`FOOTBALLDATA_KEY` IS set; the check had grepped other names). The real state: the
    football-data.org **free tier carries no `lineup` field at all**, API-Football's free plan
    refuses the 2026 season, and the scheduler ran the fetcher in a subprocess with
    `capture_output=True`, so none of it was ever logged. Fixed the same day: **ESPN is the
    second link** (`scraper/espn_lineups.py`, key-free, full XI + bench ~T-60, verified live on
    Roma–Atalanta with Pašalić on the bench), `lineup_chain_status.json` names each source's
    outcome every run, the scheduler logs the child's WARNING lines and pushes the reason to
    Telegram once per match. Player rows carry `lineup: confirmed|predicted|recent`; the /picks
    card marks non-confirmed players `XI prob. NN%` and an uncertain predicted starter is priced
    as the start%/bench mixture (`_mix_start_sub`). ESPN quirk: the DEFAULT python-requests
    agent gets 200, a browser UA gets 403 (curl_cffi is the third rung). **Self-checking, not
    "always works":** `health_check.check_lineup_sources` probes ESPN every cycle when a Serie A
    kickoff is inside 30h and goes CRITICAL when a match inside T-45 (or kicked off <3h ago) has
    no sheet, carrying the chain's own reason; the lineup stage retries every cycle down to T-5
    and a sheet that lands AFTER the T-30 run re-fires `prediction_update` so the pricing is
    redone on the XI. If a player line still says `XI prob.` at kickoff, the health JSON and
    `lineup_chain_status.json` say why — read those, don't re-diagnose the sources.
    **Same shape hit the live monitor mid-match (all three of `/incidents`, `/statistics`,
    `/lineups`), from a HOME IP.** Cookies from a prior www page visit, Referer/Origin headers
    and every impersonation profile still 403; rapid retries add `curl (7)` refusals. No
    request-level trick found. **Live stats/events therefore come from ESPN**
    (`scripts/data/live_espn.py`, unauthenticated `site.api.espn.com` scoreboard + summary,
    specimen-verified on the live match: possession, shots, SoT, blocked, corners, fouls,
    saves, tackles, clearances, cards + goals/cards/subs; NO per-player stats).
    `live_sofascore.fetch_live_data_for_matches` trips a 10-min breaker after a cycle where
    every Sofascore endpoint 403'd (one blocked cycle burns ~2 min of backoff, longer than
    the poll interval) and goes straight to ESPN; a match no source answered is OMITTED
    (last good data kept) and stamped `live_fetch_error`; `live_source` names the feed and
    the /live card shows it.
- **Throttling ≠ ban.** Rapid successive requests return `CurlError (7) Failed to connect
  ... port 443` — a *connection* error, not a 403. Back off ~20s; it recovers. Don't read it
  as the ban returning.
- **What you'll see** (a real ban): `api.sofascore.com/api/v1/...` returns 403 across all curl-cffi profiles, all domain variants, all timing.
- **Why**: Cloudflare IP-fingerprint ban, often after heavy scraping. Lasts hours to days.
- **Fix**: `www.sofascore.com/tournament/...` HTML pages return 200. Parse the embedded `<script id="__NEXT_DATA__">...</script>` JSON blob. Standings + match incidents + venue + referee + stoppage time + attendance live directly under `props.pageProps` (since ~2026-06; previously nested in `props.pageProps.initialProps` — the web/app.py parsers support both paths).
- **Match pages DO now carry incidents (re-measured 2026-09-05)**: `__NEXT_DATA__` on
  `/football/match/<slug>/<customId>` holds an `incidents` array (goals/cards/subs/periods
  with `homeScore`/`awayScore`) — the 2026-06-11 "i18n strings only" finding is stale for
  events. It still has NO `statistics`, so the HTML tier cannot feed live team stats.
  `/event/<id>` 302s to the slugged URL. Not wired (ESPN covers events too); if ESPN ever
  goes away, this is the Sofascore path for events, statistics stay ESPN-or-nothing.
- **Page-tier map (measured 2026-06-11, mid-ban)**: NOT all www pages are equal. **Tournament hub pages** are ISR-rendered FRESH (live scores within minutes — use these; WC: `scripts/worldcup/sofascore_fetch.WC_TOURNAMENT_PAGE`). **Daily-schedule pages** (`/football/{date}`) are stale prerenders (opener showed `notstarted` 75 min after kickoff) — last resort only. **Match pages** carried i18n strings only in June 2026 (NO event/lineups/statistics payloads); by 2026-09-05 they carry `incidents` but still no statistics/lineups — an HTML fallback for lineups or player stats is still IMPOSSIBLE; during bans, lineups degrade to caps-fallback XIs and the stats parquet catches up on the first healthy API run (`events/last` re-serves history).
- **Sentinels**: SA standings page must contain `Inter`; EPL must contain `Arsenal`. If sentinel missing → schema break, log and trip breaker.
- **Prevention rule**: **HTML scraping with breaker is the canonical fallback for Sofascore**. Never just retry the API in a loop when you get 403 — burn the cooldown, scrape the HTML. And before writing any NEW page parser, fetch one specimen and confirm the data is present AND fresh (see the global "never write a parser against an unverified source" rule — this project paid for it).

### Symptom: "EPL data missing where SA has it"

Common causes and where they live:

1. **Helper reads only the SA file**: e.g. `_load_match_team_stats` opened `match_team_stats.parquet`, missing `match_team_stats_premier_league.parquet`. **Rule**: every loader that takes a `match_id` must try BOTH parquet variants in fallback order.
2. **Helper reads only the SA dir**: e.g. `_load_match_lineup` scanned `data/external/sofascore/matches/` only, missing `matches_premier_league/`. **Rule**: scan both directories.
3. **Scraper iterates only SA match_ids**: `get_match_ids()` in `scraper/sofascore_events.py` was pulling from `player_match_stats.parquet` only. **Rule**: loaders that derive a master ID list must concat both league parquets.
4. **Lookup table is SA-only**: `TEAM_TO_CITY` in `scraper/weather.py` had no EPL teams → 0 EPL weather rows. **Rule**: all team-keyed lookups (cities, venues, normaliser maps) must include both leagues.
5. **Endpoint is single-league hardcoded**: `api_team_match_history` was hardcoded to SA parquet. **Rule**: any handler taking a team name must infer or accept league, then read the right source.

6. **A shared CACHE, not a shared parquet** (found 2026-08-25): `matchday_updater._fixtures_cache_path(season)` took no league, so both leagues read/wrote one `fixtures_{season}.json`. `run_matchday_update` loops serie_a → premier_league, so Serie A refreshed the cache and EPL then found it FRESH (6h window), loaded **Serie A's** fixtures, diffed them against **Serie A's** match ids (`_get_existing_sofascore_match_ids()` was also SA-only) and detected nothing. EPL starved on every run — no error, no warning, just `No new matches detected`, which is also what a healthy run logs. Net effect: `matches.parquet` took **zero** EPL rows after 2026-03-22 while the EPL Sofascore stat parquets stayed fresh, and `features_premier_league.parquet` faithfully mirrored the frozen ground truth. **Rule**: a per-league *cache* or *diff basis* is as load-bearing as a per-league output file — if a function takes `league`, every path it derives must consume it. The working sibling `scripts/data/scrape_sofascore.py` (`_league_suffix()`, `_get_output_paths()`) had it right all along; `matchday_updater` was a second implementation that skipped it. **Diff the sibling before theorising.**
7. **A failed refresh must fall back to the cache, not to `[]`** (same fix): when the Sofascore round endpoint 403s (routine, hours-to-days), `_refresh_fixtures_cache` logs at debug and returns `[]`, and detection returned `[]` — blind for the whole ban despite a usable cache on disk. Already-ingested matches get filtered out anyway, so a stale fixtures list is free. **Rule**: never let a failed refresh return empty when a cache exists.

**Residue RESOLVED (2026-08-27)**: `matchday_updater --backfill` diffs finished fixtures against **matches.parquet** (not the stat parquets) and rebuilds missing rows from the per-match JSONs already on disk — no network, idempotent (dedup keep="first"). The 80-match EPL gap (71 of 2025-26 + 9 of 2026-27) was backfilled with it: EPL now 380/380 + 10/10, score cross-check vs the stat parquets 0 mismatches. Note: backfilled rows carry 100% xG/possession while ALL older EPL rows carry none — EPL ground truth never had team stats before this. `results.json` is still SA-only, so `_fallback_ingest_from_results` still can't cover EPL; --backfill is the EPL recovery path.

**Prevention rule (umbrella)**: **whenever you write `data/external/sofascore/X.parquet`, immediately also handle `X_premier_league.parquet`** — and same for any other ACTIVE_LEAGUES file convention. Use this idiom:
```python
for fname in (f"{base}.parquet", f"{base}_premier_league.parquet"):
    p = DATA_DIR / "external" / "sofascore" / fname
    if p.exists():
        ...
```

### Symptom: "a derived cache under data/parsed/ stopped tracking its source, and nobody noticed for months"

- **What you'll see**: a feature family that is *partially* filled — high for old seasons, ~87% for last season, 0% for this one — with no error anywhere. `data/parsed/<x>.parquet` has an mtime months old while the raw files it derives from are current.
- **Why it happens**: the build-once cache idiom
  ```python
  def _ensure_cache():
      if CACHE_PATH.exists():
          return pd.read_parquet(CACHE_PATH)   # <-- frozen forever
      ...build...
  ```
  Its docstring will often *claim* "cached for incremental updates". There is no incremental update: the first successful build is the last one. Found 2026-08-25 in `features/missing_players.py`, frozen since 2026-04-22 — 3,329 rows against 6,776 match JSONs on disk, and **1,866 of the rows it did hold came from JSONs Sofascore has since rewritten** (a match JSON is rewritten after kickoff as lineups and absences are confirmed, so "cached" ≠ "accurate").
- **Fix**: keep the watermark **in the data**, one `source_mtime` column per row, and re-parse a file when `stat().st_mtime > row.source_mtime`. **Do NOT compare against the cache file's own mtime** — writing the cache stamps it `now`, so every source rewritten between two cache writes becomes invisible permanently.
- **Verify with idempotence, not a diff**: call the refresh twice; the second call must parse **zero** files and return a frame equal to the first. A test that only asserts "the frame didn't change" passes with the file-mtime bug still in place.
- **Same walk, same trap — check league parity too**: these walkers iterate `data/external/sofascore/matches/` and carry a dead `if "premier_league" in season_dir.name: continue` guard. The EPL lives in the **sibling top-level dir** `matches_premier_league/`, never in a subdir, so that guard matched nothing and the EPL was simply never scanned — `features_premier_league.parquet` had *none* of these columns. Sofascore ids are disjoint across the two dirs (verified 3,389 vs 3,387, overlap 0), so one cache keyed on `sofascore_id` covers both.
- **Second instance, fixed 2026-08-25**: `features/first_half_splits.py` had this defect verbatim — cache frozen at 1,465 rows (22% of the 6,775 JSONs carrying a 1ST period), EPL never scanned. Fixed the same way, but note the difference that made it a separate decision: `missing_players` is a match-level signal so a refresh only fills nulls, whereas first-half splits feed `_rolling_per_team`, so refreshing **changes existing non-null rolling values**. Measured on Serie A: 53,661 cells newly filled, **0 lost**, and 2,041 of 47,502 previously-filled cells changed (4.3%; median relative change 12%, p90 50%) — the rolling window now spans the real last-N matches instead of whichever N happened to be cached. **Always measure this delta before refreshing a cache that feeds a rolling window**, and treat it as a training-set change.
- **Prevention rule**: **a cache keyed on "does the file exist" is not a cache, it is a snapshot.** Any derived artifact under `data/parsed/` must record what it derived from (`source_mtime`) and re-derive when the source moves. If you write `_ensure_cache`, write the twice-in-a-row idempotence test in the same commit.

### Symptom: "a whole feature family is 100% NULL for the CURRENT season only, and a fresh rebuild doesn't fix it"

- **What you'll see**: `features_serie_a.parquet` was rebuilt hours ago, yet e.g. `home_squad_size`, `home_avg_player_value`, `home_max_player_value` are 100% NaN for the in-progress season while ~90% filled for the previous one. No error, no warning — the build logs success.
- **Why it happens**: `config/settings.py:SEASONS` is a **hand-maintained literal list** and it lags the calendar. On 2026-08-25 it still ended at `"2025-2026"` while `get_current_season()` returned `"2026-2027"`. Any per-season enrichment written as `seasons = [season] if season else SEASONS` therefore **never iterates the season being played**. Two such loops existed in `features/build.py` (`_add_odds`, `_add_market_data`), costing 10 squad-value columns + 58 odds/disagreement columns (3.4% fill vs 74.5% the season before) — all of them live model features.
- **Fix**: `features/build.py::_seasons_to_enrich(feature_df, season)` — iterate the seasons **actually present in the frame**, not the config list. Each of these loops is already masked per season and no-ops on a season it has no rows for, so the present-seasons list is a free superset that never needs a rollover edit.
- **Do NOT fix it by appending to `SEASONS`.** 23 non-test modules import that constant, including `scripts/analysis/backtest_unified.py` and `data_quality_report.py`. Pushing a 10-match partial season into all of them is a much larger, unaudited blast radius than the enrichment bug it would fix — and it re-arms the same trap next August.
- **Detection**: `python3 -c "import pandas as pd; d=pd.read_parquet('data/features/features_serie_a.parquet'); cur=d[d.season==d.season.max()]; print(sorted(c for c in d.columns if cur[c].isna().all() and d[c].notna().mean()>0.5))"` — any column filled historically but wholly null in the newest season.
- **Measured outcome (2026-08-25)**: after the fix AND a step-cache bust, Serie A 2026-2027
  squad value went 0.0% -> 100.0% and odds 3.4% -> 54.1%. The fix alone was NOT enough:
  `odds` and `market_data` are in the `never_cache` set so they recomputed every run, but a
  later `[CACHE HIT]` step restored a stale frame and discarded them. See the
  "I fixed a feature module, the rebuild logs success, and the parquet is byte-identical"
  section below — you almost certainly need to read it before this fix will land.
- **Prevention rule**: **never gate a per-season loop on `config.SEASONS`.** Derive the season list from the data in hand. A hand-maintained calendar constant is a time bomb with an annual fuse, and it fails *silently* — the skipped season looks exactly like a season with no data.

### Symptom: "I fixed a feature module, the rebuild logs success, and the parquet is byte-identical"

The single most expensive trap in this repo. Three commits sat inert for a full
session because of it. **Two independent defects in `features/build.py`'s step cache,
either of which alone silently discards your work.**

**Defect A — a cache hit REPLACES THE WHOLE FRAME, so it undoes earlier steps in the
same run.** Each step's cache parquet stores the *entire cumulative feature frame* as
of that step (~16 MB, same size as the output), and `build()` applies it as:
```python
cached_df = self._load_cache(plugin)
state.feature_df = cached_df          # wholesale replacement, NOT a merge
```
So a step that recomputes at position 46 has its output thrown away by the very next
`[CACHE HIT]` at 47, which restores a snapshot taken weeks earlier. **The final file is
whatever the LAST cached step snapshotted** — everything before it is decorative.
This makes the `never_cache` set (`odds`, `market_data`, `pivot_to_match_level`,
`backfill_managers`, `backfill_referees`, `manager_h2h_noop`) actively misleading:
those steps really do recompute every run, and their work is really discarded every
run, unless every step after them also recomputes. Measured 2026-08-25: `_add_market_data`
demonstrably turned 2026-27 squad value from 0% → 100% when called directly on the saved
frame, while the full build left it at 0%, because steps 47–58 were all cache hits.

**Defect B — the fingerprint is blind to the code you actually edited.** `_source_fingerprint`
hashes `inspect.getsource(plugin.apply)` plus the `data_inputs` mtime+size manifest.
But **46 of 59 plugins are two-line wrappers that delegate to a `features/*.py` module**,
and the callee's source is not hashed. Editing `features/missing_players.py` changes
nothing the fingerprint can see. Worse, `data_inputs` is hand-declared and frequently
names the wrong files: `missing_players_match_time` and `first_half_splits` both declared
`player_match_stats.parquet` / `match_team_stats.parquet` while the modules actually read
the raw `data/external/sofascore/matches*/` JSON trees and their own `data/parsed/*.parquet`
caches — none of it declared. 32 of those 46 delegating plugins declare **no** `data_inputs`
at all. Net effect: a feature-module edit invalidates nothing, and the cache only ever
refreshes by luck, when an unrelated scrape happens to touch a declared file.

**Do NOT "fix" this by declaring the derived `data/parsed/*.parquet` as a `data_input`** —
that deadlocks. The module's own self-refresh runs *inside* `apply()`, which the cache
check short-circuits *before* it runs, so the parquet never updates and the fingerprint
never changes.

**What to do when a feature-module fix must land:**
1. Delete the step's cache pair by hand — both `<step>_v<ver>.parquet` **and**
   `<step>_v<ver>.fingerprint` — under `data/cache/features/<league>/`, for **every**
   league. Bumping the plugin's `version` string works too and is more honest.
2. Because of Defect A, deleting one step is not enough if any *later* step is cached.
   Either delete every step from yours to the end of the order, or rebuild with
   `use_cache=False`.
3. **Verify by measuring the output file, never by reading the build log.** `[computed]`
   in the log does not mean the value survived to disk. Diff the parquet before/after
   and assert the specific columns moved.

**Detection**: compare a module's git mtime against its step's `.fingerprint` mtime, and
sanity-check that the fill% you expect is actually in the parquet:
```bash
for f in data/cache/features/serie_a/*.fingerprint; do
  echo "$(stat -f '%Sm' -t '%m-%d %H:%M' "$f")  $(basename "$f" .fingerprint)"
done | sort
```
Any fingerprint older than your edit to the module it wraps is a stale step.

**Prevention rule**: **a step cache that stores the whole frame and replaces it on hit is
not a cache, it is a checkpoint — and checkpoints must be all-or-nothing.** Until the
cache stores only the columns its step adds and *merges* them, treat a partial rebuild as
unsound: the only trustworthy full rebuild is `use_cache=False`. And **never trust
`[computed]` in the log as evidence a value reached the parquet** — the diff is the evidence.

### Symptom: "the staleness banner is red but the dashboard data is demonstrably correct"

- **What you'll see**: the global banner dumps a raw Python dict across the bottom of every
  page — `Live HTML scrape failing — serving cached parquet data. Parquet ages: {'serie_a':
  {'html_ok': False, ...}}` — while `/api/standings/<league>` serves a complete, correct
  20-row table for both leagues.
- **Why it happened**: `degraded_parquet_only` fired on `not any_html_ok` alone. That asks
  "is the preferred SOURCE reachable?" when the only question a reader cares about is "is the
  DATA I'm being shown behind?" On a denied egress IP those two answers disagree permanently:
  Sofascore 403s every tier while the parquet fallback is complete and current.
- **Fix (2026-08-26)**: `_league_ingest_lag()` in `web/app.py` compares the fixture calendar
  against the standings payload the dashboard actually serves — `missing = played fixtures
  not ingested`, with a 24h grace matching the daily ingest cadence. HTML down **and**
  `missing > 0` → `degraded_parquet_only` (red, `ok: False`). HTML down and everything played
  is ingested → `html_blocked_data_current` (`ok: True`, banner hidden, one-line message).
- **Two traps inside that fix, both paid for**:
  1. **Fail CLOSED on unknown.** `missing is None` (calendar unreadable) counts as behind. The
     opposite already happened here: `_club_leagues_dormant` read fixtures from a results-only
     store, got zero every time, and pinned the banner green over a live Serie A failure.
  2. **A cancelled fixture keeps its original past timestamp for the rest of the season.**
     Counting it as should-have-been-played pins `missing > 0` forever and re-arms the same
     false banner by a new mechanism. Exclude `canceled`/`postponed` — the sibling
     `_next_fixture_for_team` already did, and disagreeing with the sibling is how two readers
     of one file drift apart. Do **not** narrow to `status == "finished"`: a fixture file whose
     statuses haven't flipped yet would undercount and fail OPEN.
- **Prevention rule**: **a health check must assert on the artifact the user is served, not on
  the reachability of the source you would have preferred.** And never interpolate a raw dict
  into a user-facing string — if the detail matters, it belongs in the JSON payload
  (`leagues_health`), not in the banner.

### Symptom: "a season-stamped filename is hardcoded and silently addresses last season"

Third instance of this trap in this file (see also `config/settings.py:SEASONS` above). Found
2026-08-26 in `web/app.py`, **twice in the same module**:

- `_LEAGUE_FIXTURES_FILE` and the `fixtures_paths` dict inside `/api/data-freshness` both named
  `fixtures_2025_2026*.json`. After the 2026-08-01 rollover they addressed a file frozen on
  Jun 1 while the live files (`fixtures_2026_2027*.json`) were hours old.
- **Two silent failures, no error either time**: `_next_fixture_for_team` searched a *finished*
  season for an *upcoming* fixture — 0 of 385 rows qualified, so it returned `None` for every
  team, forever. And `fixtures_age_hours` reported **2067h** instead of ~9h, permanently arming
  the `max_file_hours >= 36` branch — which is why this file's own restart-procedure note used
  to name `fixtures_stale_html_ok` as the healthy signal. **That note was written against a
  false reading.**
- **Fix**: `_league_fixtures_path(league)` delegates to `scripts.utils.match_timing.
  _sofascore_fixture_files()`, which derives the season from `get_current_season()`. One
  definition cannot drift from itself.
- **Detection**: `grep -rn "20[0-9][0-9]_20[0-9][0-9]" --include='*.py' .` — any season literal
  in a path is a time bomb with an annual fuse.
- **Prevention rule**: **never write a season into a filename literal.** Derive it, and derive
  it by calling the one existing helper rather than re-implementing the naming convention —
  the repo had a correct, well-documented deriver the whole time and three call sites ignored it.

### Symptom: "predictions.json contains the OTHER league's matches" / "Data Validation Warning repeats twice a day forever"

- **What you'll see** (found 2026-08-27, the day before SA go-live): `predictions.json` had 32
  rows — 14 Serie A + 18 EPL fixtures, ALL tagged `league: serie_a`. The validator caught it
  within hours as `odds_full.json: missing 18/32 prediction matches` and said so twice a day
  for 3 days; the repetition trained the reader to ignore it.
- **Why — three stacked defects**:
  1. `predict_unified.load_upcoming_matches()` absorbed the Sofascore fixture files (which
     carry BOTH leagues) with a date filter but **no league filter** — a regression from the
     2026-08-24 stale-fixtures fix. Its non-SA sibling `_load_league_matches` filtered; the
     SA branch didn't. **Diff the sibling.**
  2. The engine stamps every prediction with the RUN's league → EPL rows became `serie_a`.
  3. `betting_unified.load_predictions()` gates **per FILE**: predictions.json passes as
     serie_a wholesale, and `load_odds_full()` merges ALL leagues' odds — so gated-league
     EPL matches were priceable and bettable as Serie A, at SA's looser Kelly.
- **Fix**: league guard in `_absorb` (predict_unified), foreign-league drop + warning at the
  engine choke point (`run_ensemble_predictions`), per-league pairs in
  `_check_cross_coverage`. Regression test: `test_epl_fixture_never_enters_the_serie_a_loader`.
- **Second instance (found 2026-08-31, four days after the first)**: the fix above gated
  `predictions*.json` only. `goal_predictions.json` (and btts/cards/corners/margin) are
  merged both-league files with **no league field** (23 rows = 10 SA + 10 EPL + 3 stale),
  and `scan_ou_market` — the ONLY enabled market — priced them against the merged odds
  with no gate. An EPL O/U bet would have been journaled as Serie A the moment its edge
  landed in band. Caught only because near-miss logging (`near_misses` in the slip, top-5
  in the run log) listed Arsenal–Chelsea. Fix: `_gate_aux_predictions` in `run()` +
  `scan_ou_market` skips matches outside `pred_by_match`. **When you audit a gate, enumerate
  every input the priced path iterates, not the one the gate was written for.**
- **Prevention rules**: **a per-file betting gate is only as safe as the file's homogeneity —
  any multi-league source feeding a per-league gate needs a row-level league check at the
  choke point.** And **a validation warning that fires unchanged twice a day is not noise to
  silence, it is a finding to READ** — dedup the alert (now signature-gated in
  run_full_pipeline: notifies only when the critical-issue SET changes, state cleared on
  recovery), never mute the check.

### Symptom: "Telegram is ~11 messages/day of machine status" (fixed 2026-08-27)

- **Measured** (notification_history.jsonl, 18 days): 200 sends — 77 Health (count-churn
  re-alerts: `missing 18/32` → `17/31` minted a fresh issue key every 30-min cycle), 37
  routine "✅ pipeline done" cards, 36 repeated validation warnings, 35 "N parlay picks
  ready" (against the Jun-15 parlay guardrail).
- **Fix**: `_issue_key` now collapses EVERY number (catch-all `\d+` → N); transient window
  45→120 min (monitor cycle is 30); routine-success scheduler cards persist state but don't
  send; dry-day pipeline cards don't send; parlay push removed (user decision — /parlays
  still answers); 6 dead notify builders deleted; quiet hours 23:00–07:00 enabled (alert+live
  bypass; overnight routine messages are DROPPED on Telegram not queued — the 09:00 digest
  covers them; macOS unaffected). Tests: `tests/test_notify_dedup.py`.
- **Prevention rule**: **the history file logs the macOS fallback `message`, NOT the
  `tg_html` Telegram actually renders** — judge Telegram content from the builder code, and
  judge VOLUME from the history. And **an alert channel averaging 4+ machine-status pings a
  day is a broken alert channel** — every new notify call site must say what changed and fire
  only on change.

### Symptom: "health-monitor flags 64 sparse columns CRITICAL but they're known-empty by design"

- **Why**: `features_quality` check flags any column >90% NaN unless its prefix is in `SPARSE_PREFIXES`. New feature families (e.g. `home_fh_*` first-half rollups, `home_xg_share_*` zone xG) weren't allowlisted.
- **Fix**: add the prefix to `SPARSE_PREFIXES` in `scripts/pipeline/health_check.py`.
- **Prevention rule**: **when adding a feature family that's known to be partially populated** (recent seasons only, sub-set of leagues), add the prefix to `SPARSE_PREFIXES` in the same commit.

### Symptom: "`/historical/` Odds API returns 422 INVALID_MARKET"

- **What you'll see**: backfill script burns credits but writes 0 rows.
- **Why**: `/historical/sports/<sport>/odds` only supports `h2h`, `totals`, `spreads`. **Not** `btts`, `double_chance`, `team_totals`, `draw_no_bet`, player props.
- **Fix**: never request `btts` etc. on the historical endpoint. Use per-event `/events/{id}/odds/` for those — and only on live future events, never historical.
- **Reference table** (memorise this):
  | Endpoint | Markets allowed |
  |---|---|
  | `/odds/` (bulk) | h2h, totals, spreads |
  | `/historical/sports/<s>/odds/` | h2h, totals, spreads |
  | `/events/{id}/odds/` | btts, double_chance, draw_no_bet, team_totals, alternate_totals, alternate_spreads, all player_* |
- **Prevention rule**: **invalid-market 422s STILL COST CREDITS**. Validate market×endpoint compatibility before sending.

### Symptom: "Groq bill is hundreds of dollars" (FIXED 2026-05-06)

- **What you'll see**: Monthly Groq spend creeping into 3-figures. Audit dashboard shows `groq/compound` (compound-beta) line dominating costs — was $73/mo on its own across May 2026.
- **Why**: `scripts/prediction/sentiment_analyzer.py` ran on every full pipeline AND every incremental refresh, making 5-15 web-search-augmented compound-beta calls per match × 41 matches × 3-5 builds/day. Default `GROQ_DAILY_LIMIT=800` was a per-call counter not a $-cap, so worst-case was $1000+/mo.
- **Fix**: Two-layer:
  1. **Default OFF**. Both pipeline call sites (`run_full_pipeline.py` Step 16 and the incremental path) gate behind `RUN_SENTIMENT=1` env var. Default is skip. Sentiment is a soft signal not used by any betting decision in this codebase, so the absence has zero downstream impact.
  2. **Hard $/day cap** if re-enabled. `GROQ_DAILY_BUDGET_USD` env var (default `$1.00`) is converted to a call cap at client init: `GROQ_DAILY_LIMIT = budget / cost_per_call`. Even if `RUN_SENTIMENT=1`, monthly worst-case ≈ `$1 × 30 = $30`.
- **Re-enabling sentiment**: requires evidence it's worth the cost. Backtest `sentiment_edge` as a binary feature against 1X2 outcomes, require skill_score > 0.02 over 200+ matches, before flipping `RUN_SENTIMENT=1`. Same standard as the corners/cards models we ripped out the day before.
- **API key**: this project uses a dedicated key (`gsk_CNw...02yQ`, labeled "SerieA-Pipeline" in Groq console) separate from Pulse's keys. Bills are isolated.
- **Prevention rule**: **any external-API caller must declare a $-budget cap in env, default-low.** A per-call counter is not a budget. Models that take a billable action per fixture × per build × per day need their cost computed against the call frequency before shipping.

### Symptom: "Match Intelligence shows corners/cards" (REMOVED 2026-05-06)

- **History**: As of 2026-05-04 the dashboard Match Intelligence card showed `λ ≈ 10.00` corners and `λ ≈ 4.50` cards on every match — silent fallback to constants because the 2026-04-27 cleanup deleted `data/models/markets/*.cbm`. First fix wired the walkforward predictor (`predict_walkforward_markets.py`) to overlay real per-match values.
- **Then 2026-05-06 we backtested** the walkforward corners + cards models against held-out 2024-25 SA (380 matches). Result: **all six lines (corners 8.5/9.5/10.5, cards 3.5/4.5/5.5) had skill score ≤ 0** — predictions were base-rate ± noise. AUC 0.51-0.60. Cards models also miscalibrated (over-predicted "Over" by 11-14pp, calibration_gap > 0.09 post-isotonic). The trainer's own `summary.json` confirms these numbers.
- **Decision**: removed corners + cards predictions from every consumer:
  - `web/templates/prediction_detail.html` — `renderMatchIntel` now shows scorers + AI reasoning only
  - `web/app.py:api_match_intel` — returns scorer + reasoning only; corner/yc fields dropped
  - `scripts/pipeline/telegram_bot.py:_handle_today` — corners + cards lines removed from digest; PNG attachment block removed
  - `scripts/betting/parlay_generator.py:887` — corners + cards leg insertion blocks disabled
  - `scripts/prediction/generate_unified_report.py` — `cards`/`corners` keys removed from per-match output
  - `scripts/pipeline/run_full_pipeline.py` — Step 22b-2 walkforward overlay removed (was added 2026-05-04, removed 2026-05-06)
- **What we kept**:
  - `predict_walkforward_markets.py` itself (on disk, not wired into pipeline) — keep for future re-enablement when models earn it
  - `data/models/walkforward/serie_a/{corners,cards}_over_*/` artifacts — same reason
  - `cards_predictions.json` / `corners_predictions.json` files — `ml_market_predictions.py` still writes constants there. Nothing reads them anymore. Could be deleted but low priority.
- **Re-enabling the predictions requires**:
  1. A held-out backtest with skill_score > 0.02 AND ECE < 0.05 on a recent season the models didn't train on
  2. Real bookmaker corners/cards odds from the per-event endpoint (currently we only have h2h/totals/spreads bulk)
  3. Verification that the new predictions don't suffer the same systematic over-prediction bias seen in cards (post-isotonic calibration_gap > 0.09)
- **Prevention rule**: **a model is not "production" because the trainer ran successfully — it's production after a held-out backtest beats the always-predict-base-rate baseline.** Skill score, not log-loss, is the right metric: `1 - brier/baseline_brier > 0` is the floor. Anything below should not be wired into a UI or a bet generator.

### Symptom: "ML classifier probabilities look arbitrary / the ensemble's ML leg disagrees with everything on upcoming matches"

- **What you'll see** (measured 2026-08-31): on upcoming fixtures the CatBoost 1X2 model saw
  **8 of 126 features reproduced exactly** vs the training rows for the same matches (ML L1
  0.22 vs market), while the same model scores 51% on walk-forward CV. Nothing errored.
- **Why**: `FeatureBuilder` in `ensemble_prediction_engine.py` approximated every upcoming row
  from a **team cache** — each team's most recent played row, whose `home_*`/`away_*` values
  are the PRE-match state of that PREVIOUS game (one match stale), with matchup/derived
  features borrowed from the wrong opponent. The real serving path,
  `features.build.build_upcoming_features` (fixtures through the 58-step pipeline), existed
  but nothing called it — and it was broken: fixture rows had `match_id=NaN`, so downstream
  merges joined NaN to NaN (10 fixtures → 1,008,000 rows, OOM), and it wrote the poisoned
  frames into the PRODUCTION step cache mid-build.
- **Fix (2026-09-01)**: `_fixture_frame()` gives fixtures the canonical
  `"{date}_{home}_{away}"` id, NaN scores and `matchweek = last played + 1`;
  `FeaturePipeline.build(write_cache=False)` for ad-hoc frames; daily
  `build_upcoming_feature_rows()` → `data/features/upcoming_features_{league}.parquet`
  (run_full_pipeline Step 10d + after the incremental feature rebuild, 24h gate, never on the
  T-30 path); `FeatureBuilder._load_prebuilt()` serves that row when (home, away, date)
  matches and the derived/interaction mirrors become fill-only; `ML_CACHE_FALLBACK_SCALE=0.5`
  halves the ML weight on any match still served from the cache (EPL, missing file).
  Blind MW3 rebuild: **94/126 exact**. The remaining 23 inexact features are one cluster
  (attack/defense strength → poisson_*, league/matchweek avg goals, position momentum) whose
  TRAINING values contain same-matchweek lookahead — the serving row is now the honest one;
  making training as-of is open work ("P1b" in `.plans/p1-ml-serving-skew-findings.md`).
- **Same session, same retrain**: the Aug-25 matchweek retrain shipped a **1-tree draw
  detector** (P(draw)=0.26 on every row) because the final fit early-stopped on the 10-match
  current season. Restored from archive; trainer now early-stops on a ≥200-match season and
  refuses to save a <20-tree / flat model (`tests/test_draw_detector_final_fit.py`).
- **Prevention rules**: **a serving feature builder is correct only if it is the training
  pipeline run on the unplayed row — an approximation from cached rows must be measured
  against the training parquet (exact-match rate per feature) before it is trusted, and any
  fallback path must announce itself (`last_feature_source`) and cost the model weight.**
  A fixture row entering the pipeline needs the same identity key as a played row. Ad-hoc
  pipeline runs must never write the production step cache. And a retrain's early-stopping
  set must meet the same n-floor as its gate folds.

### Symptom: "the T-30 log says `Journal: recorded 1 bets` but bet_journal.json has nothing new" (FIXED 2026-09-05)

- **What you'll see**: `Selected: 1 bets`, `Saved bet slip`, `Journal: recorded 1 bets` in
  `logs/pipeline.log`, and one line above them
  `Duplicate blocked: Lazio vs Milan OVER 1.5 already exists as 2026-03-15_Lazio_vs_Milan_OU_1.5_OVER_1.5`.
  The real journal took **zero** bets from go-live (2026-08-27) to 2026-09-05.
- **Why**: `bet_journal.add_bet` has a second dedup after the id check — same match + same
  selection, any market name — meant to catch market-name variants (`O/U 1.5` vs `OU_1.5`)
  on the same fixture. It was **date-blind**. Serie A fixtures repeat every season, the bet
  is always O/U Over on the same pair, so a settled bet from last season swallowed this
  season's and returned the OLD id. The caller then logged `len(slip.bets)` as "recorded".
- **Fix**: the guard requires the same `date`; picks additionally require the same market
  (`dedup_by_market=True`); `save_bet_slip` counts ids that start with the bet's own date and
  logs `Journal: N of M bets NOT recorded: …` at WARNING when any were blocked. Test:
  `test_last_seasons_settled_bet_never_blocks_this_seasons`.
- **Prevention rule**: **a "recorded N" log must count what the store accepted, never what
  the caller offered** — and any dedup key on a recurring event (fixtures, matchweeks,
  rounds) must include the date. The real-journal count since go-live is the check:
  `python3 -c "import json;d=json.load(open('data/betting/bet_journal.json'))['bets'];print(sum(1 for b in d.values() if (b.get('placed_at') or '')>='2026-08-27'))"`.

### Symptom: "every `ref_*` feature is NaN for the current season" / "referee_assignments_<league>.parquet has 0 rows" (FIXED 2026-09-05)

- **What you'll see**: `features_serie_a.parquet` current-season rows with `referee` "filled" but
  every `ref_*` column NaN; `matches.parquet` current-season `referee` = `""` (an empty string
  passes every `notna()` coverage check); the weekly refresh log `Saved 0 referee assignments
  to referee_assignments_serie_a.parquet` and a `✗ referees` step; the 1X2 ensemble (three
  `ref_*` inputs) silently loses the referee. The O/U money models do NOT use `ref_*`.
- **Why, three stacked**: (1) the Sofascore fixture list carries no `referee` (0 of 21
  finished 2026-27 fixtures), and `matchday_updater` wrote `referee_info.get("name", "")` —
  `""`, not None; (2) worldfootball publishes a season weeks late, and on 2026-08-31 the
  scraper wrote the EMPTY frame as the per-league cache, which the `exists()` short-circuit
  then served forever (the master `referee_assignments.parquet`, 3,368 rows, was untouched);
  (3) nothing checked referee coverage.
- **Fix**: ESPN names the referee once a match is `post` (`summary.gameInfo.officials`,
  position "Referee", full names in the same space as nine seasons of history — specimen
  Fiorentina–Torino 2026-09-05 → "Davide Massa"; EMPTY pre-kickoff, so this fills ground
  truth, never the upcoming row). `live_espn.match_referee` → `matchday_updater
  _referee_from_espn` on every new row, `backfill_referees` (`--backfill-referees`) on every
  matchday run for played rows still unnamed, `""` normalised to None (196 rows).
  `scraper/referee.py` never writes an empty cache; both 0-row caches deleted.
  `health_check.check_referee_coverage` (WARNING) reads matches.parquet against the fixture
  calendar. Tests: `test_live_espn.py`, `test_matchday_updater.py`, `test_referee_cache_guard.py`.
- **Prevention rules**: **`""` is not a value — write None for a missing field, or every
  coverage number lies.** **A cache written from a failed fetch is a poisoned cache — guard
  the WRITE (`if df.empty: return`), not only the read.** And when a source publishes late
  (worldfootball, FBref), name the second source in the code, not a comment.

### Symptom: "I want to make the betting go live" / "flip the dry-run flag"

- **`BETTING_DRY_RUN_FROM_MORNING=true` (morning+evening plists) is NOT paper mode — it is
  candidate-deferral, and it IS the live design.** Morning/evening Step 24 generates
  candidates only (no journal write); the commit happens at T-30 via the pre-kickoff-monitor
  (every 15 min) → `run_pre_kickoff()` → `generate_unified_report` + `save_report` →
  `save_bet_slip` journals the bets (and `_archive_bets` journals with supersede). That
  timing IS the edge: journal analysis 2026-04-25 showed >24h-early bets ran −5% ROI,
  <24h bets +63%.
- **Flipping the flag to false makes bets commit at MORNING against stale odds — the −5% path.
  It is an anti-live switch. Never touch it to "go live".**
- The system is de-facto paper exactly when the Odds API key is dead (engine finds no odds →
  0 bets). With a live key the chain is armed end-to-end and nothing needs flipping.
- **Stake size is Nicola's 2026-09-05 decision: Kelly 0.15 (was 0.05), cap 2.5%.** Three
  places must agree or `_make_bet` silently rescales: `BettingConfig.kelly_fraction`,
  `_LEAGUE_KELLY_DEFAULTS["serie_a"]` (the per-league scaler divides by the cfg value) and
  `market_rules["O/U_Over"]["kelly_fraction"]` (the 1.5 line's own fraction). A test pins all
  three. Dry run on the MW3 slate: EUR 5–8 a bet → EUR 18–22 (the 1.5 line sits at the cap ×
  the 0.85 marginal-edge multiplier = 2.1%). Same day, the cold_home / away_fav_ref veto was
  scoped to 1X2/DC/DNB (`_veto_applies`): O/U candidates are judged on edge alone, 1 → 3
  selected on the same inputs. Both changes reach the T-30 run at its next cycle (the T-30
  is a child process that re-reads the code; see the pick-engine section).
- **Go-live checklist that actually matters** (all verified 2026-08-27): key alive; ledger
  invariants green (journal-derived == bankroll.json); `pre-kickoff-monitor` + `telegram-bot`
  + `settlement` jobs loaded; `betting_unified --dry-run` produces a sane slate (enabled
  markets: O/U_Over 1.5/2.5 + Alt_OU only); per-league gate correct.
- **Per-league gate**: `_league_betting_enabled()` — Serie A always on; any other league needs
  `data/models/<league>/deployment_state.json` with `betting_enabled: true`. EPL was set true
  on 2026-04-29 (predates the June "EPL stays gated" decision) and would have silently taken
  real EPL bets; set false 2026-08-27 with `gated_reason`. Lift only via
  `scripts/models/validate_league_deployment.py` after EPL earns the bar.

---

## Restart procedure (when needed)

```bash
# 1. Snapshot current state
launchctl list | grep "com.seriea-pipeline" | sort -k3
ps aux | grep -E "scheduler.py|sofascore_watcher|telegram_bot|web/app.py" | grep -v grep

# 2. Stop everything
for plist in ~/Library/LaunchAgents/com.seriea-pipeline.*.plist; do
  launchctl unload "$plist"
done

# 3. Verify nothing left
ps aux | grep -E "scheduler.py|sofascore_watcher|telegram_bot|web/app.py" | grep "Projects/seriea-pipeline" | grep -v grep

# 4. Reload
for plist in ~/Library/LaunchAgents/com.seriea-pipeline.*.plist; do
  launchctl load "$plist"
done

# 5. Wait + verify health
sleep 8
curl -s http://localhost:5001/api/data-freshness | python3 -m json.tool
launchctl list | grep "com.seriea-pipeline" | awk '$2 != 0 && $2 != "-" {print}'
```

Healthy signal: `ok=True` with `severity` in {`ok`, `fixtures_stale_html_ok`, `offseason_dormant`, `html_blocked_data_current`}. **`live_standings_ok=false` is NOT a failure on its own** — under `html_blocked_data_current` it just means the live scraper is blocked while the served table is complete through the latest played matchweek (check `leagues_health[*].missing == 0`). Exit codes other than `0` or `-15`/`-9` (running) on any job indicate a real failure to investigate.

