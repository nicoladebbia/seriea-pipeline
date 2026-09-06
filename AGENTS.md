# seriea-pipeline — substrate briefing for a cold reader

Written 2026-09-06. This file replaced a verbatim copy of the maintainer's `CLAUDE.md`,
which carried every conclusion the previous reader had reached about this system. This
file carries none of them, on purpose. It exists so that a reader with no history can
(a) avoid the environment traps that would make a fresh read wrong, and (b) re-derive
every load-bearing claim from primary sources instead of inheriting it.

## 0. Posture

- Every doc, code comment, test name and commit message in this repo was written by an
  AI working with the owner. Prose is a CLAIM. A parquet, a journal row, a live HTTP
  response, a walk-forward you ran yourself is EVIDENCE.
- This file states what EXISTS, what the CURRENT CONFIGURATION is, and HOW THINGS FAIL
  SILENTLY. It does not say whether any model is good, whether any market has an edge,
  what any performance number is, or why any decision was taken. Code comments in the
  betting modules DO carry such claims (ROI figures next to thresholds); read them as
  the hypotheses that produced the threshold, not as facts.
- Do not read a number from a JSON file the pipeline wrote and call it verified. The
  pipeline writes gate verdicts, backtest summaries and metadata into `data/models/`;
  those files are excluded from the handover pack (§10). Re-derive from the parquet.
- The owner wants disagreement when it is warranted, with the evidence. Do not soften.

## 1. Shape of the system (no verdicts)

Serie A football. A feature pipeline builds ~1,000 pre-match features per fixture; several
model families price markets; a betting layer selects and journals bets against Odds API
prices; a Flask dashboard and a Telegram bot render the outputs; launchd runs everything.
There is NO bookmaker execution integration anywhere in the code: a "real" bet is a
journal row at the sized stake; placing it is a human act.

- Python ≥ 3.11 (`pyproject.toml`; launchd runs 3.13). pandas, pyarrow, numpy, scipy,
  scikit-learn, xgboost, lightgbm, catboost, optuna, flask, requests, curl_cffi, duckdb.
- Env var NAMES (`.env.example`, values never in the repo): `ODDS_API_KEY`, `GROQ_API_KEY`,
  `OPENAI_API_KEY`, `GOOGLE_GEMINI_KEY`, `ANTHROPIC_API_KEY`, `APIFOOTBALL_KEY`,
  `FOOTBALLDATA_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `PORT`.
- League keys `serie_a`, `premier_league` (`config/leagues.py`). Season strings
  `"YYYY-YYYY"`. Odds API sport keys `soccer_italy_serie_a`, `soccer_epl`; region `eu`.
- Canonical team names: `config/team_names.py` (`TEAM_NAME_MAP`, `normalize_team`,
  accent-folded, case-insensitive). Every join across sources goes through it.

Layout:
- `cli.py` — CLI entry for pipeline steps
- `config/` — settings (note `config/settings.py:SEASONS`, §4)
- `features/` — feature engineering; `features/build.py` is the step-cached pipeline
  (52 plugins, topologically ordered by declared dependencies; `never_cache` =
  `{pivot_to_match_level, backfill_managers, backfill_referees, odds, market_data,
  manager_h2h_noop}`; `build_upcoming_features()` runs fixture rows through it)
- `ml/` — training config (`ml/config.py`: `time_decay_per_season = 0.90`, fold
  recency weighting), correction layer, walk-forward helpers
- `scripts/models/` — `train_walkforward.py` (1X2), `train_over_under.py` (O/U:
  `TimeSeriesSplitter` season folds, XGBoost-importance feature selection, CatBoost
  early stop 150 rounds, isotonic calibration, artifacts in
  `data/models/universal/over_under/`), `goal_process.py` (minute-resolved simulator)
- `scripts/prediction/ensemble_prediction_engine.py` — 1X2 ensemble. Default weights
  `factor 0.035, xg 0.124, ml 0.605, player_xg 0.032, market 0.205`; overridable by
  `data/models/ensemble_weights.json` / `optimized_weights.json` written by the
  component ledger's gated refit. Appends every live prediction to
  `data/predictions/prediction_ledger.jsonl` (§6).
- `scripts/betting/` — `betting_unified.py` (real-money engine), `picks.py` (per-match
  line + paper journal), `inplay.py` (paper in-play engine), `market_promotion.py`
  (paper→real gate), `bet_journal.py` (the ledger), `fair_odds_tracker.py`
  (`record_predictions`), `player_predictions.py`, `betfair_*` (exchange odds, read-only)
- `scripts/data/` — `odds_fetcher.py`, `odds_tracker.py`, `matchday_updater.py`,
  `live_espn.py`, `live_sofascore.py`, `results_fetcher.py`
- `scripts/pipeline/` — `run_full_pipeline.py` (the driver), `scheduler.py` (match-clock
  stages + settlement), `health_check.py`, `telegram_bot.py`
- `web/app.py` — Flask app (~11k lines, two background threads; never import it to
  borrow a helper). Page routes: `/`, `/live`, `/projections`, `/matches`,
  `/track-record`, `/value-bets`, `/prediction/<slug>`, `/rosters`, `/fantacalcio`,
  `/worldcup`. API: `/api/dashboard`, `/api/live`, `/api/data-freshness`,
  `/api/standings/<league>`, `/api/projections`, `/api/match-markets/<slug>`,
  `/api/match/<date>/<home>/<away>`, `/api/best-bets`, `/api/players`.
- `tests/` — 109 files. `conftest.py` autouse `_promotion_state_isolated` redirects the
  promotion state file to tmp; there is no autouse redirect for the journals or
  bankroll — a test that imports a module-level PATH constant and writes through it
  hits the real file (it happened once with `bankroll.json`).
- `data/` — 4+ GB, gitignored

`ARCHITECTURE_MAP.md` is a per-file map with import graph and liveness derived by AST and
plist scan (generated 2026-06-01; files added since are not in it). Its import/liveness
facts are mechanical. Its per-file quality GRADES are opinion; ignore them.
`DATA_CATALOG.md` is the per-file, per-column data reference with fill rates, dtypes,
join recipes and provenance per pipeline step. Its column audit is mechanical. Its
"broken / partial" section and "fallback rating" are dated claims; check the file dates.

## 2. Run surface

```bash
python3 -m pytest tests/                 # tests
ruff check . && mypy .                   # lint, types
python3 scripts/pipeline/run_full_pipeline.py --help
python3 -m scripts.betting.betting_unified --dry-run   # a slate without journaling
python3 -m scripts.betting.inplay --backtest          # rewrites data/models/inplay/backtest.json
python3 -m scripts.models.goal_process --backtest     # rewrites data/models/goal_process/backtest.json
python3 -m scripts.betting.market_promotion           # gate state from the journals, no write
python3 -m scripts.betting.market_promotion --null-sim   # the bar's behaviour on a zero-edge market (§3)
python3 scripts/analysis/backtest_unified.py --mode {accuracy,betting,value,ml-walkforward,all} --season 2024-2025
```

Entry points are 21 launchd plists in `~/Library/LaunchAgents/com.seriea-pipeline.*.plist`
(morning, evening, pre-kickoff-monitor, settlement, matchweek-retrain, health-monitor,
telegram-bot, web-dashboard, odds-edge-scanner, odds-line-movement, sofascore-watcher,
weekly-data-refresh, weekly-monitor, daily-digest, refresh-understat, scrape-epl-current,
transfer-refresh, friendlies-refresh, fanta-tracker, state-backup, wc-refresh) plus
`web/app.py` on port 5001 and `cli.py`. `launchctl list | grep seriea-pipeline` shows what
is loaded; `-` as PID means an interval job between runs, not a dead job. Morning and
evening plists have `RunAtLoad: true`: a Mac wake or `launchctl reload` fires both at once,
which is why odds fetches are cache-gated (§3).

**Match clock** (`scheduler.py:MATCH_CLOCK_STAGES`, driven by the pre-kickoff monitor,
`StartInterval 900`): `odds_T72h`, `odds_T24h`, `odds_T6h`, `odds_T3h`, `player_props_T60`,
`lineup_fetch` (T-55, retried every cycle to T-5), `prediction_update` (T-30: re-predict on
the XI, commit bets, build picks), `fill_nudge_T10`, `odds_T5m` (closing snapshot,
`use_cache=False`), `fill_verify_close`, `settlement_check` (T+120). The monitor spawns
`run_full_pipeline.py --pre-kickoff` as a child with `capture_output=True`; the child's
lines are in `logs/pipeline.log`, not in the monitor's err log. The child re-imports the
code each run, so an edit to the betting modules takes effect at the next cycle.

**Settlement** (`scheduler.py settle`, `StartInterval 300`): fetch scores → settle the real
journal → settle the gated-league paper journal → `settle_picks` (paper picks; then
`evaluate_promotions`) → in-play settle + backtest rerun → P&L snapshot → bankroll from
journal → parlays → 1X2 prediction scoring → correction-layer rolling update → rolling
metrics → drift check. Note the order: the promotion gate is re-evaluated on EVERY
settlement run.

Logs: `logs/pipeline.log`, `logs/launchd-<job>-err.log`. Tests also log to
`pipeline.log` when run from the repo, so a burst of identical lines within one second
is a test, not a pipeline run.

Plist trap: a plist whose first byte is `[` instead of `<` has been rewritten to compact
JSON by some tool and has lost its schedule (survives in launchd memory until reboot).
Check: `for p in ~/Library/LaunchAgents/com.seriea-pipeline.*.plist; do head -c 1 "$p"; echo " $p"; done`.

## 3. Current configuration state (what IS, as of 2026-09-06)

**Real-money engine** (`scripts/betting/betting_unified.py`, `BettingConfig`):
- Enabled markets: `O/U_Over` and `Alt_OU`, `allowed_lines [1.5, 2.5]`. Everything else
  in `market_rules` is `enabled: False` (1X2 and variants, O/U_Under, AH, DC, DNB, BTTS,
  Corners, Cards).
- Edge band `O/U_Over`: `min_edge_pct 5.0`, `max_edge_pct 7.0`; line 2.5 overrides to
  `[7.0, 10.0]`; line 1.5 `line_shrinkage 0.6` (model 60% / market 40% before the edge
  is computed), line 2.5 unshrunk. `Alt_OU` same bands, no shrinkage.
- Selection rules applied in order: situational veto (`_veto_applies`, 1X2/DC/DNB only);
  odds dead zone, best price in [1.5, 2.0) rejected; `confidence_edge_tiers` adjust the
  minimum edge by model probability (floor 2%); reject below min edge, above max edge,
  raw edge < 2%; `min_odds 1.10`, `max_odds 500`; a Pinnacle price is REQUIRED and the
  best price must be ≤ Pinnacle × 1.05; best price in [2.0, 2.5] lowers the minimum edge
  by 0.5. Situational adjustments keyed `tag__market__selection` shift the threshold.
- Stake: `kelly_fraction 0.15`, `max_stake_pct 2.5`, bankroll from the journal
  (start 1000.0). Three places must agree or `_make_bet` silently rescales:
  `BettingConfig.kelly_fraction`, `_LEAGUE_KELLY_DEFAULTS["serie_a"]`,
  `market_rules["O/U_Over"]["kelly_fraction"]`. A test pins all three.
- League gate: `_league_betting_enabled()`. Serie A always on. Any other league needs
  `data/models/<league>/deployment_state.json` with `betting_enabled: true`; EPL is false.
- `BETTING_DRY_RUN_FROM_MORNING=true` in the morning and evening plists is NOT a paper
  switch. It makes those runs emit candidates only (no journal write); the commit happens
  at T-30 (`run_pre_kickoff` → `generate_unified_report` → `save_bet_slip`). Setting it
  false commits at the morning run against morning odds. Whether either timing is better
  is a question for the journal (§8).
- Journal dedup (`bet_journal.add_bet`): by `bet_id` (date + match + market + selection),
  then same date + match + selection (picks additionally same market). `MAX_EDGE_PCT 12`.

**Ledger**: `data/betting/bet_journal.json` is the append-only source of truth; `history.json`
and `bankroll.json` are derived snapshots; recompute rather than patch. Entry fields:
`bet_id, match, date, league, market, selection, model_prob, sharp_implied_prob, edge_pct,
odds, bookmaker, avg_odds, pinnacle_odds, stake, confidence, factors, status, result_score,
profit, closing_odds, clv_pct, placed_at, settled_at, match_kickoff_at, pipeline_status,
extra`. Status ∈ {pending, superseded, won, lost, push, voided}. CLV
(`_compute_clv`): `1/closing_odds − 1/placed_odds` as a percentage when a closing price
exists, else `sharp_implied_prob − 1/placed_odds`. Paper entries carry `extra`
(bet_type, player, team, source, tier, side, line, lineup, start_pct); real bets `extra: null`.

**Paper→real promotion** (`scripts/betting/market_promotion.py`): per market key, a paper
market is promoted when ≥ 50 settled paper bets (won/lost/push; voids excluded), ROI > 0,
z ≥ 1.0 where z = mean(unit return) / (pstdev / √n), and CLV > 0 once ≥ 20 closing prices
exist. **The check runs after every settlement run and promotes at the FIRST crossing**;
there is no fixed evaluation point. Promoted markets are mirrored into the real journal
at Kelly × 0.5, cap 1.5%, floor 0.2%, tagged `extra.picks_ref`, `pipeline_status
pick:promoted`. Demotion at ≥ 30 real bets with ROI < −10% or z < −1, checked the same
way; the paper count restarts. 14 market keys share the bar (`MARKET_NAMES_IT`). State
`data/betting/market_promotion.json` is derived; re-derive from journals.
`--null-sim` simulates exactly this evaluation on a market with a chosen true edge.

**Pick engine** (`scripts/betting/picks.py` → `data/upcoming/picks.json`): one line per
upcoming Serie A match. VALUE is copied from the engine's slip; LEAN is the best
positive-edge row across every served market against a real price, ranked
`in band > tier > multi-book > edge`, 12% edge cap, 20% model-probability floor; NO EDGE
otherwise. Paper-journaled at a flat €10 only on the T-30 path, kickoffs inside 3h:
headline + best exotic + every in-band alternative. `PICK_EVENT_MARKETS` = `h2h_h1,
totals_h1, btts_h1, halftime_fulltime, double_chance_h1, correct_score,
player_goal_scorer_anytime, player_shots_on_target, player_shots, player_assists`; a test
pins that every fetched key has a consumer in `price_key_for_row`. Journals:
`data/betting/picks_journal.json`, `data/betting/inplay_journal.json` — the hook that
writes them landed 2026-09-05 14:35 local, after that day's last T-30; the first entries
are expected from the 2026-09-06 slate.

**In-play** (`scripts/betting/inplay.py`): paper only, Serie A only (`PROFILE_LEAGUE`),
Telegram pings default off. Power de-vig (`p_i = q_i^k`, Σp = 1). A pick must beat the
snapshot's own overround plus 1.96 × the Monte Carlo standard error of its fair
probability; no fixed edge threshold. Fair prices are shrunk toward the market by a
walk-forward weight stored in the backtest file. Baseline order: `pre_match_odds`, else the
last pre-kickoff snapshot line, else the first 0-0 snapshot inside a measured window.
Settlement CLV is against the NEXT snapshot's price.

**Goal simulator** (`scripts/models/goal_process.py`): home/away split from Poisson xG,
total from a regression on the base rate, rescaled to the O/U model's P(over 2.5) via
`calibration_k` ∈ (0.25, 4.0), saturation stamped on rows. Tiers read from its backtest
file at request time. Never serves full-time 1X2 or Over 2.5 (`NOT_SERVED`). Red-card
multipliers in `profile.json` are measured from incidents, refit with `--measure-red`.

**Lineups**: Sofascore API → ESPN (`scraper/espn_lineups.py`) → predicted XI.
`lineup_chain_status.json` records each source's outcome per run. Rows carry
`lineup: confirmed|predicted|recent`; a predicted starter is priced as a mixture of start
and 20-minute sub via isotonic knots in `data/models/player_floors/start_calibration.json`
(fitted from `data/lineup_history/`, refit with `player_predictions calibrate-start`).

**Odds API** (`scripts/data/odds_fetcher.py`): `CACHE_DURATION_MINUTES 60`, gated on
`use_cache` not `critical`; T-5 closing passes `use_cache=False`. Budget pacing tiers:
`PRIORITY_CRITICAL 1` (T-5, never drops), `EXTRAS 2` (per-event, drops at 95% pacing),
`PROPS 3` (90%), `T24H 4` (80%), `T72H 5` (70%), `BACKFILL 6` (60%). Per-event pick
markets cost 10 credits per event, 45-minute refresh, `PRIORITY_EXTRAS`; the per-cycle
result is archived to `data/odds_snapshots/pick_markets_<ts>.json.gz` (since 2026-09-06).

**Other**: sentiment off unless `RUN_SENTIMENT=1`, `GROQ_DAILY_BUDGET_USD` caps calls.
Telegram dedups by issue signature with numbers collapsed; quiet hours 23:00–07:00 drop
routine messages. `data/notification_history.jsonl` logs the macOS fallback text, not what
Telegram rendered. Bot commands: `/today /picks /record /bets /bankroll /live /parlays
/worldcup /xi /formazioni /sfide /help`.

## 4. Identity keys and things that fail silently

- **Canonical match id** is `{YYYY-MM-DD}_{Home}_{Away}`. Some derived files were, or
  still are, keyed on an FBref hash or a Sofascore numeric id. A wrong-format join
  produces NaN columns, never an error. `data/parsed/match_id_mapping.parquet` maps hash ↔
  sofascore ↔ understat ↔ canonical. `data/external/sofascore/*` is natively Sofascore-keyed
  by design; do not "fix" those keys.
- **`config/settings.py:SEASONS`** is a hand-maintained literal and lags the calendar.
  `get_current_season()` is the live value. A loop gated on `SEASONS` skips the season
  being played and looks identical to a season with no data. `features/build.py::
  _seasons_to_enrich` derives seasons from the frame instead. Do not append to `SEASONS`;
  23 modules import it.
- **Season-stamped filename literals** (`fixtures_2025_2026*.json`) silently address last
  season after August. Derive via `scripts.utils.match_timing._sofascore_fixture_files()`
  (returns `(path, league)` tuples). Detection:
  `grep -rn "20[0-9][0-9]_20[0-9][0-9]" --include='*.py' .`
- **`""` is not a value.** An empty string passes `notna()`; the referee column was `""`
  for a whole season and every coverage check said full. Write None.
- **League parity**: Serie A and EPL live in sibling files (`X.parquet` and
  `X_premier_league.parquet`; `matches/` and `matches_premier_league/` top-level dirs, not
  subdirs) but several JSON outputs are ONE merged file with both leagues
  (`goal_predictions.json`, `odds_full.json`, fixture files). A per-FILE league gate over a
  merged file lets the other league through; a per-league cache that ignores its league
  argument starves the second league (`fixtures_{season}.json` did). Any loader taking a
  match id must try both parquet variants.
- **Step cache in `features/build.py`** (the most expensive trap in the repo):
  (A) each step caches the WHOLE cumulative frame and a cache hit REPLACES the frame, so
  a step that recomputed earlier in the run is undone by the next cached step; the
  final parquet is whatever the last cached step snapshotted. (B) the fingerprint hashes
  `plugin.apply` source plus a hand-declared `data_inputs` manifest; most plugins
  delegate to `features/*.py` modules whose source is NOT hashed and whose real inputs are
  often undeclared. Net: editing a feature module and rebuilding yields a byte-identical
  parquet, and `[computed]` in the log is not evidence a value reached disk. The only
  trustworthy full rebuild is `use_cache=False`. Never write the production cache from an
  ad-hoc frame (`build(write_cache=False)`). Verify a feature change by diffing the
  parquet, never by reading the build log. Cache dir: `data/cache/features/<league>/`
  (`<step>_v<ver>.parquet` + `.fingerprint`).
- **Build-once caches under `data/parsed/`**: the idiom `if CACHE.exists(): return read()`
  freezes at first build. The correct pattern keeps a `source_mtime` column per row and
  re-parses when the source file is newer; verify with two consecutive calls (second must
  parse zero files). Sofascore match JSONs are REWRITTEN after kickoff, so a cache of
  pre-kickoff parses is wrong, not just stale.
- **Journal dedup on recurring fixtures**: fixtures repeat every season; a date-blind
  dedup swallows this season's bet with last season's id and the caller logs it as
  recorded. Count what the store accepted, not what was offered.
- **Ingest completeness**: `matchday_updater` detects new matches by diffing fixture ids
  against what is on disk. A row that exists with NaN stats is "present" to that diff.
  `heal_from_espn` refills NaN cells from ESPN and stamps `source="espn"` /
  `data_source="espn"`; such rows are stand-ins, replaced when Sofascore answers.
- **Serving skew**: `ensemble_prediction_engine.FeatureBuilder` used to build upcoming rows
  from a per-team cache (one match stale, wrong opponent). Now `build_upcoming_features`
  runs fixtures through the pipeline daily into `data/features/upcoming_features_{league}.
  parquet` and `_load_prebuilt` serves that row; any fallback to the cache is stamped
  `last_feature_source` and scales the ML weight by `ML_CACHE_FALLBACK_SCALE`. The parity
  test is the exact-match rate per feature between the served row and the training row
  for the same match, after the match is played. A set of standings-derived training
  features still carry same-matchweek values.
- **Retrain early-stopping**: the trainer early-stops on a ≥ 200-match season and refuses
  to save a model with < 20 trees or a flat output, because a 10-match current season once
  produced a 1-tree draw detector.
- **Timestamps**: stored timestamps are meant to be UTC-aware ISO strings and readers
  compare against `datetime.now(timezone.utc)`. Exceptions exist: `prediction_ledger.jsonl`
  timestamps are naive local time. A naive/aware subtraction is caught and returned as −1
  in at least one age helper.
- **Freshness state**: `data/pipeline_state.json` fields like `last_odds_fetch` drive the
  monitors; a writer that saves a file without bumping its state field looks stale forever.
- **Health checks assert on the served artifact, not on source reachability**: the
  standings banner compares the fixture calendar to what is served (excluding
  canceled/postponed, failing closed on an unreadable calendar). A source being blocked
  is not, by itself, a data failure here.

## 5. External sources — access facts

**Sofascore.** Plain `curl` gets 403 from Sofascore at all times (TLS fingerprint); a plain
curl 403 proves nothing. Probe with `curl_cffi`, `impersonate="chrome124"`. Three distinct
403 shapes with different fixes:

| Shape | Tell | What helps |
|---|---|---|
| Blanket IP deny | `/robots.txt` also 403, `server: Varnish` | Change network. Waiting does nothing. HTML tier is also 403. |
| Cloudflare fingerprint ban | API tier 403, `www` HTML 200 | Back off; parse `__NEXT_DATA__` from `www.sofascore.com` pages |
| API challenge | `robots.txt` 200, API answers `{"error":{"code":403,"reason":"challenge"}}` on both `api.` and `www./api/` | No request-level fix found. Use ESPN. |

Detection: `python3 -c "from curl_cffi import requests as r; x=r.get('https://www.sofascore.com/robots.txt',impersonate='chrome124'); print(x.status_code, x.headers.get('server'))"`.
Rapid retries give `CurlError (7)` connection refusals: throttling, not a ban; back off
~20 s. Page tiers: tournament hub pages are ISR-rendered fresh; daily-schedule pages are
stale prerenders; match pages carry `incidents` in `__NEXT_DATA__` (measured 2026-09-05)
but no statistics and no lineups. `live_sofascore` trips a 10-minute breaker after a
cycle where every endpoint 403'd. Sentinels: the Serie A standings page must contain
`Inter`, the EPL page `Arsenal`.

**ESPN** (`site.api.espn.com`, key-free). The DEFAULT python-requests user agent gets 200; a
browser UA gets 403. Scoreboard + summary carry team stats (possession, shots, SoT, blocked,
corners, fouls, saves, tackles, clearances, cards), key events (goals, cards, subs),
lineups (XI + bench ~T-60), the referee once a match is `post` (empty pre-kickoff),
half-time linescores. No per-player stats, no xG. `scripts/data/live_espn.py`,
`scraper/espn_lineups.py`.

**Odds API** (the-odds-api). Endpoint × market compatibility; a 422 for an invalid market
STILL costs credits:

| Endpoint | Markets |
|---|---|
| `/odds/` bulk | h2h, totals, spreads |
| `/historical/sports/<s>/odds/` | h2h, totals, spreads only |
| `/events/{id}/odds/` | btts, double_chance, draw_no_bet, alternate_totals, h1 markets, halftime_fulltime, correct_score, player_* |

`team_totals` / `alternate_team_totals` returned zero bookmakers in eu+uk on 2026-09-05.
In-play feed semantics, measured on one match: the feed lags the pitch by minutes (the
first snapshot after a goal already carries the repriced line); Pinnacle does not reprice
in-play through it; the `totals` lines in an in-play snapshot are the PRE-MATCH lines
carried along, not live prices. In-play snapshots are stored under `data/live/`.

**Others.** `football-data.co.uk` serves results as CSV (a working third source when
Sofascore is denied). worldfootball publishes referee assignments weeks late; an empty
frame must never be written as a cache. FBref: a visible browser gets the page, headless
and `curl_cffi` get 403 (Turnstile); `shots_all` tables are gone from 2025-26 HTMLs.
football-data.org free tier carries no lineup field; API-Football free plan refuses the
2026 season. Betfair exchange odds: session token dies within 24h, fetch is manual.

## 6. Data substrate (where the evidence is)

| What | Path | Notes |
|---|---|---|
| Ground truth | `data/parsed/matches.parquet` | both leagues, canonical id, score, HT, team stats, referee, `data_source` |
| Features | `data/features/features_{serie_a,premier_league}.parquet` | ~1,000 cols; provenance per step in DATA_CATALOG §13 |
| Upcoming rows | `data/features/upcoming_features_{league}.parquet` | pipeline-built serving rows |
| Goal timeline | `data/parsed/goal_timeline.parquet` | minute-stamped goals, both leagues |
| Incidents | `data/external/sofascore/match_incidents*.parquet` | goals/cards/subs/VAR; `source` column |
| Player match stats | `data/external/sofascore/player_match_stats*.parquet` | no card column; no minute stamps for fouls/tackles/passes |
| Odds snapshots | `data/odds_snapshots/{bookmakers,odds,extra}_<ts>.json` | one triple per fetch (§6a) |
| Pick-market snapshots | `data/odds_snapshots/pick_markets_<ts>.json.gz` | per-event props/1H/HT-FT/CS prices, per fetch cycle, from 2026-09-06 |
| Current odds | `data/upcoming/odds_full.json`, `odds_extra_markets.json`, `pick_markets_raw.json` | merged both leagues; overwritten |
| Predictions (current) | `data/upcoming/predictions*.json`, `goal_predictions.json`, `picks.json` | goal_predictions has NO league field; overwritten |
| 1X2 prediction ledger | `data/predictions/prediction_ledger.jsonl` | append-only, one row per live ensemble call: home, away, match_date, prob_H/D/A, predicted, confidence, timestamp (naive local); ~4% of rows have empty match_date |
| Fair-odds ledger | `data/betting/fair_odds_ledger.json` | one row per fixture keyed (match, date): model probs, market h2h at prediction time, actual outcome once settled |
| Real journal | `data/betting/bet_journal.json` | §3 schema |
| Paper journals | `data/betting/paper_journal.json` (gated leagues), `picks_journal.json`, `inplay_journal.json` | same schema + `extra` |
| CLV history | `data/betting/clv_history.json` | per settled bet: bet_odds, closing_odds, clv, market, placed_at |
| Lineup archive | `data/lineup_history/predictions_*.json` | every predicted-XI run |
| Models | `data/models/universal/*.cbm`, `over_under/*.cbm`, `walkforward/`, `goal_process/profile.json` | metadata/backtest JSONs withheld |
| Fixture calendar | `data/external/sofascore/fixtures_<season>*.json` | list of Sofascore event dicts: `startTimestamp` (epoch UTC), `homeTeam.name`, `status.type` |
| Pipeline state | `data/pipeline_state.json` | derived |

**6a. Snapshot schemas.** `bookmakers_<ts>.json` = `{timestamp, matches: {"Home vs Away":
{h2h: [{bookmaker, home, draw, away}], "totals_2.5": [...over/under per book...],
"totals_2.75": ..., "spreads_0.5": ...}}}` — per-book prices, every totals line the books
offered. `odds_<ts>.json` = consensus h2h per match. `extra_<ts>.json` = per-event btts,
double_chance, draw_no_bet, alternate totals with `commence_time`. Archive: 1,428 triples
from 2026-03-29, median two per day, more on match days (the T-stage fetches). The T-5
closing snapshot is a normal triple; nothing in the file marks it — identify it by
timestamp vs `commence_time`.

**6b. Record sizes at the time of writing** (counts, not verdicts). Real journal: 198
rows placed 2026-02-06 → 2026-05-11, statuses won 101 / lost 82 / superseded 12 / push 2 /
voided 1; by market O/U 2.5 49, O/U 1.5 49, DC 45, 1X2 29, BTTS 5, O/U 3.5 5, others ≤ 3;
`match_kickoff_at`, `placed_at`, `closing_odds` present, so hours-to-kickoff is
derivable per bet. Zero real rows placed since 2026-08-27. `paper_journal.json`: 1 row.
`picks_journal.json` / `inplay_journal.json`: not yet on disk. Prediction ledger: 23,197
rows from 2026-03-03.

**6c. Grading semantics.** VAR incidents: `incident_class` is the ON-FIELD decision under
review and `confirmed=False` means it was overturned (goalAwarded+False = goal
disallowed; penaltyNotAwarded+False = penalty given). Own goals are credited to the
beneficiary. ESPN player ids are `espn:<accent-folded name>`. `settle_picks`: full-time
markets from the results dict; first-half markets from `goal_timeline.parquet` (a match
present with no 1H goal is 0-0), else ESPN key events; player props from
`player_match_stats.parquet` joined on date + team + accent-folded name; a journaled
player with no row or 0 minutes settles `voided` once ≥ 22 stat rows exist for that
date; anything ungradable stays pending. `results_fetcher.settle_bets` defaults an unknown
market to lost and therefore skips `picks_ref` entries. Paper CLV: the grader passes the
feed's last price as `closing_odds`, so CLV exists only where a closing price does.

## 7. Health checks that exist (what they assert, not whether they are right)

`scripts/pipeline/health_check.py`: ledger invariants (journal-derived vs bankroll),
features quality (columns > 90% NaN outside `SPARSE_PREFIXES`), lineup sources (probes
ESPN inside T-30h), player stats coverage (> 14h after a finished match), picks journal
activity (a Serie A kickoff passed with nothing paper-journaled), referee coverage,
match record completeness, data freshness. `/api/data-freshness` reports severities;
`live_standings_ok=false` under `html_blocked_data_current` is not a failure by itself.
State: `data/monitoring/health_status.json` (withheld from the pack as a derived report;
regenerate by running the check).

## 8. Questions for the cold read (framed without the hypothesis)

1. From `matches.parquet`, the snapshot archive and its closing snapshots alone, is there
   a bettable edge in any bulk market (h2h, totals lines), and at what timing relative to
   kickoff? Build your own check before reading any of the repo's.
2. From the real journal (198 rows, kickoff + placement + closing per bet), what is the
   settled record per market, per line, per hours-to-kickoff bucket, with an interval? Is
   any of it distinguishable from zero? For the paper markets the record starts
   2026-09-06; the per-event price archive (§6) is what makes it replayable at other
   timings or under other rules once it has accumulated.
3. Is the walk-forward in `scripts/models/` leak-free? Enumerate the as-of guarantee per
   feature family in `features/`, including the standings-derived ones.
4. The promotion bar (§3) is evaluated after every settlement across 14 markets and
   promotes at the first crossing. Run `--null-sim` and decide whether the bar, the
   evaluation cadence and the demotion bar are adequate controls, and what real-money
   exposure a zero-edge market gets before demotion.
5. Does the O/U selection path (Pinnacle reference, shrinkage, per-line band, dead zone,
   confidence tiers) select the bets its own settled record supports, when the record is
   re-derived by you?
6. Is the in-play fair price compared to the in-play market on equal information, and
   would the comparison survive a different baseline choice?
7. Pick three feature modules, edit one value, rebuild, and confirm the parquet changed.

## 9. What was withheld

Withheld because it carries conclusions: `CLAUDE.md` (project; also a stray copy under
`data/upcoming/`), `README.md` (its opening paragraph asserts the result),
`MODEL_STATUS.md`, `CLEANUP_PLAN.md`, `AUGUST_RUNBOOK.md`, `.plans/`, `docs/*.md`,
`.aider*`, `.claude/`, `.codesight/`, `logs/`, `scripts/analysis/FEATURE_AUDIT_REPORT.md`,
`data/models/**/draw_specialist_report.md`, the git history (commit subjects are
arguments), the maintainer's memory directory (per-path, not in the repo), and under
`data/`: every `*_metadata.json`, `backtest.json`, `rare_events.json`,
`halves_backtest.json`, `start_calibration.json`, `walkforward/**/summary.json`,
`market_promotion.json`, `data/betting/*report*.json`, `*summary*.json`, `*stats.json`,
`pnl_history.json`, `data/upcoming/backtest_results.json`, `data/monitoring/`, and the
directories of stored analyses and LLM narratives: `data/ai_reasoning_cache/`,
`data/analysis/`, `data/diagnostics/`, `data/experiments/`, `data/feedback/`,
`data/optimization/`, `data/quality/`, `data/pipeline/proof_of_edge_state.json`,
`data/upcoming/{match_reasoning,sentiment_analysis,bookmaker_analysis,player_analysis}.json`.

Cannot be withheld: the code's structure encodes the decisions; test names, docstrings and
the ROI comments beside thresholds carry rationale. A repo-level read can re-verify
measurements, not re-design from scratch. For the strongest read, hand over only the data
rows in §6 and question 1.

## 10. Handover

Build the pack from the repo root. It also excludes `.env` (secrets), `logs/`, tool caches
and the model archive. Verify the listing before handing it over.

```bash
tar -czf ../seriea-cold-pack.tgz \
  --exclude=.git --exclude=.env --exclude='.aider*' --exclude=.claude --exclude=.codesight \
  --exclude=CLAUDE.md --exclude=README.md --exclude=MODEL_STATUS.md \
  --exclude=CLEANUP_PLAN.md --exclude=AUGUST_RUNBOOK.md --exclude=.plans --exclude=docs \
  --exclude=logs --exclude='__pycache__' --exclude='.*_cache' --exclude=catboost_info \
  --exclude=data/cache --exclude=data/monitoring --exclude=data/models/universal/archive \
  --exclude=data/ai_reasoning_cache --exclude=data/analysis --exclude=data/diagnostics \
  --exclude=data/experiments --exclude=data/feedback --exclude=data/optimization --exclude=data/quality \
  --exclude=proof_of_edge_state.json --exclude=match_reasoning.json --exclude=sentiment_analysis.json \
  --exclude=bookmaker_analysis.json --exclude=player_analysis.json \
  --exclude='*_metadata.json' --exclude='backtest.json' --exclude='backtest_results.json' \
  --exclude='rare_events.json' --exclude='halves_backtest.json' --exclude='start_calibration.json' \
  --exclude='summary.json' --exclude='market_promotion.json' --exclude='*report*.json' \
  --exclude='*summary*.json' --exclude='*stats.json' --exclude='pnl_history.json' \
  --exclude='*_report.md' --exclude='*_REPORT.md' \
  .
tar -tzf ../seriea-cold-pack.tgz | grep -E '/\.env$|CLAUDE\.md|README\.md|_metadata\.json$|/backtest\.json$|/logs/|ai_reasoning|/diagnostics/|/optimization/|reasoning|sentiment' ; echo "(must print nothing)"
```

If the reader is a Claude Code session, run it from the extracted directory (a new path
gets an empty memory directory) and leave `~/.claude/CLAUDE.md` in place: it holds the
owner's working preferences, not project verdicts.
