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
- **`tools/`** — Developer tools and utilities
- **`web/`** — Flask dashboard
- **`monitoring/`** — Pipeline monitoring
- **`tests/`** — Test suite
- **`data/`** — Parquet files, trained models, cache (in .claudeignore — 5GB+)

## Commands
```bash
python3 cli.py                # Main CLI
python3 -m pytest tests/      # Run tests
ruff check .                  # Lint
mypy .                        # Type check
```

## Key Facts
- **Model performance:** see `MODEL_STATUS.md` — read live from `data/models/universal/catboost_no_odds_metadata.json`. NEVER quote a hard-coded accuracy here or anywhere else.
- Per-league model separation (not one model for all leagues)
- Time-decay weighting, 2017+ training window
- Betting leaks patched (odds NOT used as input features)
- Odds backfill via historical API
- Sofascore scraper for EPL data

## Model performance — ALWAYS read metadata, never quote markdown

When asked "how is the model performing right now?":

1. Run `python3 scripts/diagnostics/print_model_status.py` — it reads `catboost_no_odds_metadata.json` and prints the honest numbers.
2. Cite `cv_summary.last3_accuracy` as the primary metric (walk-forward 1X2 accuracy on last 3 eval seasons).
3. Realistic ceiling for 1X2 is 53–55% (Pinnacle close / academic SOTA). Anything above ~56% is leakage or fiction.
4. If you see a markdown file claiming a higher number, the doc is wrong — fix or delete it. Do not propagate.

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
  1. **`CACHE_DURATION_MINUTES = 60`** in `scripts/data/odds_fetcher.py` (was 10). Per-event extras and bulk markets don't move enough pre-kickoff to need a 10-min refresh. T-5min closing snapshots bypass cache via `critical=True` anyway.
  2. **`use_cache=True`** on the non-critical `fetch_and_save_odds` callers in `run_full_pipeline.py` (the parallel path at ~988 and the sequential path at ~1282), and on `fetch_league_odds` at ~1590. The `run_incremental` path at line 616 already gates by `needs_odds_refresh(max_age_hours=4.0)` so leave that `use_cache=False`.
  3. **Optional plist tweak** (only if (1)+(2) prove insufficient): drop `<key>RunAtLoad</key><true/>` from `morning.plist` + `evening.plist`. `StartCalendarInterval` still catches up on next wake if the scheduled time was missed during sleep.
- **Prevention rule**: **all `fetch_and_save_odds` callers default to `use_cache=True` unless they have an upstream freshness check.** Same for `fetch_league_odds`. The cache exists precisely to absorb wake-storm duplicates.

### Symptom: "Auto-poll burning credits with no live matches"

- **What you'll see**: `Auto-poll: no live matches (N/12)` in `launchd-web-dashboard-err.log`, but `is_match_day()` returned True hours before kickoff. Each poll = 2 Odds API credits.
- **Why**: `is_match_day()` is too lax — returns True for the entire calendar day. Page visit triggered `_ensure_auto_poll()`.
- **Fix**: only auto-start when a match is **imminent** (within 30 min of kickoff or already live). Bail out after 4 empty polls, not 12.
- **Prevention rule**: **never auto-poll based on calendar day alone**. Always require a kickoff-time check. Default bail-out for empty polls = 4 (20 min), not 12 (60 min).

### Symptom: "Sofascore API blocks (HTTP 403) but I need fresh data"

- **What you'll see**: `api.sofascore.com/api/v1/...` returns 403 across all curl-cffi profiles, all domain variants, all timing.
- **Why**: Cloudflare IP-fingerprint ban, often after heavy scraping. Lasts hours to days.
- **Fix**: `www.sofascore.com/tournament/...` HTML pages return 200. Parse the embedded `<script id="__NEXT_DATA__">...</script>` JSON blob. Standings + match incidents + venue + referee + stoppage time + attendance are all in `props.pageProps.initialProps`.
- **Sentinels**: SA standings page must contain `Inter`; EPL must contain `Arsenal`. If sentinel missing → schema break, log and trip breaker.
- **Prevention rule**: **HTML scraping with breaker is the canonical fallback for Sofascore**. Never just retry the API in a loop when you get 403 — burn the cooldown, scrape the HTML.

### Symptom: "EPL data missing where SA has it"

Common causes and where they live:

1. **Helper reads only the SA file**: e.g. `_load_match_team_stats` opened `match_team_stats.parquet`, missing `match_team_stats_premier_league.parquet`. **Rule**: every loader that takes a `match_id` must try BOTH parquet variants in fallback order.
2. **Helper reads only the SA dir**: e.g. `_load_match_lineup` scanned `data/external/sofascore/matches/` only, missing `matches_premier_league/`. **Rule**: scan both directories.
3. **Scraper iterates only SA match_ids**: `get_match_ids()` in `scraper/sofascore_events.py` was pulling from `player_match_stats.parquet` only. **Rule**: loaders that derive a master ID list must concat both league parquets.
4. **Lookup table is SA-only**: `TEAM_TO_CITY` in `scraper/weather.py` had no EPL teams → 0 EPL weather rows. **Rule**: all team-keyed lookups (cities, venues, normaliser maps) must include both leagues.
5. **Endpoint is single-league hardcoded**: `api_team_match_history` was hardcoded to SA parquet. **Rule**: any handler taking a team name must infer or accept league, then read the right source.

**Prevention rule (umbrella)**: **whenever you write `data/external/sofascore/X.parquet`, immediately also handle `X_premier_league.parquet`** — and same for any other ACTIVE_LEAGUES file convention. Use this idiom:
```python
for fname in (f"{base}.parquet", f"{base}_premier_league.parquet"):
    p = DATA_DIR / "external" / "sofascore" / fname
    if p.exists():
        ...
```

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

### Symptom: "Match Intelligence shows corners=10.00 / cards=4.50 on every match"

- **What you'll see**: Every Serie A and EPL match in the Match Intelligence card on the prediction-detail page reports identical `λ ≈ 10.00` corners and `λ ≈ 4.50` cards. Yellow-card eagerness shows N/A. Telegram digest shows nothing in the corners/cards lines.
- **Why it happens**: The 2026-04-27 cleanup deleted `data/models/markets/*.cbm` (96 legacy market models). `scripts/models/comprehensive_markets.py:predict_all_markets()` wraps every model load in `try/except → log.debug` and falls back to literal constants `(2.1, 2.4)` cards = 4.5 total and `(5.4, 4.5)` corners ≈ 10. Step 22b (`ml_market_predictions.py`) writes those constants to `cards_predictions.json` and `corners_predictions.json` with `source: "CatBoost ML"`. The Flask API gates real-vs-fake on `source == "walkforward_catboost"`.
- **Fix (already in place)**: Step 22b-2 in `run_full_pipeline.py` calls `predict_walkforward_markets.predict_walkforward_markets(league, fast_only=True)` AFTER Step 22b and overlays per-match real values for any Serie A fixture present in `features_serie_a.parquet`. Web API `web/app.py:api_match_intel` zeroes out values when `source != "walkforward_catboost"` so EPL and uncached SA matches show N/A instead of fake constants.
- **What still doesn't work**:
  - **EPL corners/cards** — no walkforward models exist under `data/models/walkforward/premier_league/{corners,cards}_over_*`. EPL match intel shows N/A until those are trained.
  - **Cache miss for far-out SA fixtures** — `features_serie_a.parquet` is only refreshed nightly. Fixtures injected after the build (e.g. matchweek 38 fixtures landing 14 days out) won't be in the cache, so `fast_only=True` skips them. They show N/A until the next nightly build.
  - **BTTS predictions are still constant** — `btts_predictions.json` is written by Step 22b which now produces `home_xg=1.4, away_xg=1.15, btts_yes=0.5148` for every match (same root cause as corners/cards). Walkforward HAS a BTTS model at `data/models/walkforward/serie_a/btts/season_2024-2025.cbm` but the walkforward script's `_load_market_models` hardcodes `{market}_over_{line}` directory naming and would need extending. Downstream consumers `scripts/betting/betting_unified.py:720` and `scripts/betting/parlay_generator.py:854` are sizing real bets on this constant. **Address in next session.**
  - **Slow-path on-demand feature build is broken** — `features/build.py → features/derived.py:_add_opposition_adjusted_xg` raises `ValueError: Length of values (16300) does not match length of index (16012)` due to duplicate matches in the merge. Triggered when `predict_walkforward_markets` is called WITHOUT `fast_only=True` and the cache is incomplete. Production must always use `fast_only=True`.
- **Prevention rule**: **never trust a `source` string alone — always have a fallback flag the API can check.** When a writer falls back to constants on model-load failure, it must mark the entry `_unavailable=True` or use a distinguishable source string. Silent fallback to literal numbers is the worst of both worlds.

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

Healthy signal: `ok=True`, `severity=fixtures_stale_html_ok` (or `ok`), `live_standings_ok=true`. Exit codes other than `0` or `-15`/`-9` (running) on any job indicate a real failure to investigate.

