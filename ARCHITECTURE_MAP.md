# ARCHITECTURE_MAP.md — Per-File Navigability Map
> **Purpose:** AI/human navigation map for the seriea-pipeline codebase. For any file: what it does, what it talks to, how good it is, and the one-line 'to change X, go here'.
> **Generated** 2026-06-01 from a 22-agent orchestrated audit. The import/liveness facts are mechanically derived (AST + launchd plist scan), not narrated — zero hallucination in the dependency graph.
> **Scope:** CODE only. For DATA files (parquet, models, cache) see **DATA_CATALOG.md** — it is authoritative and this doc defers to it.
> **Live performance numbers:** never quoted here — run `python3 scripts/diagnostics/print_model_status.py`. See MODEL_STATUS.md.

## At a glance
- **321 code files** audited (~173K LOC). DATA excluded (4.4 GB, see DATA_CATALOG.md).
- **Liveness:** 203 live (reachable from a launchd/cli entry point) · 35 one-shot scripts (kept by rule) · 31 tests · 51 flagged dead → **12 confirmed garbage** (see CLEANUP_PLAN.md).
- **Quality:** A=147 · B=144 · C=6 · D=5 · F=19.

## Entry points — what actually runs the system
15 launchd jobs (all currently **hibernated** — off-season) + the Flask web app + cli.py. This is the entire command/subscription surface.

| Trigger (launchd plist) | Invokes | Module |
|---|---|---|
| morning / evening | `scheduler.py once` | `scripts/pipeline/scheduler.py` |
| pre-kickoff-monitor | `scheduler.py pre-kickoff-monitor` | `scripts/pipeline/scheduler.py` |
| odds-line-movement | `scheduler.py line-movement` | `scripts/pipeline/scheduler.py` |
| settlement | `scheduler.py settle` | `scripts/pipeline/scheduler.py` |
| weekly-monitor | `scheduler.py monitor` | `scripts/pipeline/scheduler.py` |
| daily-digest | `notify --digest` | `scripts/pipeline/notify.py` |
| health-monitor | `monitor --quiet` | `scripts/pipeline/monitor.py` |
| matchweek-retrain | `weekly_retrain` | `scripts/pipeline/weekly_retrain.py` |
| weekly-data-refresh | `refresh_weekly_data` | `scripts/pipeline/refresh_weekly_data.py` |
| refresh-understat | `refresh_understat_players` | `scripts/data/refresh_understat_players.py` |
| scrape-epl-current | `scrape_epl_match_reports` | `scripts/data/scrape_epl_match_reports.py` |
| sofascore-watcher | `sofascore_watcher` | `scripts/data/sofascore_watcher.py` |
| friendlies-refresh | `scraper.sofascore_friendlies` | `scraper/sofascore_friendlies.py` |
| telegram-bot | `telegram_bot` | `scripts/pipeline/telegram_bot.py` |
| web-dashboard | `app.py` | `web/app.py` |

**External subscriptions/APIs:** Odds API (h2h/totals/spreads bulk + per-event markets), Groq (sentiment — default OFF, `RUN_SENTIMENT=1`, `$1/day` cap), Telegram Bot API. Key env vars: `RUN_SENTIMENT`, `GROQ_DAILY_BUDGET_USD`, Odds API key/tier.

---

## Per-file map by module
Legend — Liveness: 🟢 live · 🔧 one-shot · 🧪 test · ⚫ dead. Verdict: keep / **delete** / edit / merge / rename.

> ⚠️ **Known gap: `scripts/data/` has no section below** (~30 files, incl. every
> Sofascore/odds/backfill module). Noticed 2026-07-16. This is not cosmetic —
> it is plausibly *why* the 15 phantom modules stayed invisible: a launchd job
> invoking `scripts.data.X` had no map entry to contradict, so a file that did
> not exist looked no different from one that did. Treat "not in the map" as
> "unknown", never as "absent" — the map is authoritative only where it has
> coverage. Regenerating this section is worth a session of its own.
>
> It cost again on 2026-07-16. `scripts/data/scrape_fbref_missing.py` (weekly,
> step 2 of `refresh_weekly_data.py`) was the last phantom — and its *input*,
> `scripts/pipeline/_refresh_fbref_fixtures.py`, had been failing silently since
> April behind a `TypeError` that named the write and not Cloudflare. Neither had
> a map entry. The pattern is not "files go missing", it is **a whole directory
> nobody can see, so nothing in it can be missed.**
>
> **Partial coverage started 2026-08-01** (`scripts/data/` section below). It is
> seeded, NOT complete — files added since are listed there; the ~30 pre-existing
> ones are still unmapped. The rule stands: absent from the map means *unknown*.

### `scripts/data/` — PARTIAL (seeded 2026-08-01, not a full sweep)

#### 🟢 `scripts/data/rumor_history.py` — grade A · keep
- **Does:** Append-only transfer-rumor lifecycle store. Folds each daily `scrape_rumors` result into `rumor_history.parquet` (one row per rumor ever, with `first_seen`/`last_seen`/`times_seen`) plus `rumor_scrape_log.parquet` (per-run, per-club coverage). Exists because `rumors_<season>.parquet` is overwritten daily and is therefore survivorship-biased — unusable for any retrospective study.
- **Talks to:** imported_by: `scripts/data/refresh_transfers.py` (step 3, right after `scrape_rumors`); imports: pandas, pathlib. Reads/writes `data/external/transfermarkt/rumor_history.parquet` + `rumor_scrape_log.parquet`.
- **To change how rumor lifetimes are measured, the file is this one.** To change what a rumor *is*, it is `scraper/transfermarkt.py:_parse_rumors_page`.
- **Quality signals:** content-owned key (`source_url` deliberately excluded — it carries a forum `post_id` that would reset lifetimes), atomic tmp+replace writes, injectable clock + `tm_dir` for tests, 19 tests in `tests/test_rumor_history.py` all exercising state *transitions* rather than steady state.
- ⚠️ **Consumers must call `annotate_status()`, not compare `last_seen` directly** — a stale `last_seen` means either "dropped" or "scraper was blind", and only `last_covered_at` separates them.

### `cli.py/` — 1 files

#### 🟢 `cli.py` — grade B · keep
- **Does:** Click-based CLI entry point that orchestrates the entire seriea-pipeline: scraping fixtures/matches/odds/weather/transfers/referees, parsing HTML, building features, running full pipeline, and ML training/evaluation/optimization commands.
- **Talks to:** Imports from: config/settings (SEASONS), ml/config (FeatureConfig, MODEL_TYPES, TuningConfig), storage/paths (ensure_dirs, parsed_path, features_path), scraper/* (fixtures, match_reports, odds, weather, transfermarkt, referee, registry), parser/match_page (parse_all_matches), features/build (build_features), ml/* (training, data, evaluation, persistence, feature_selection), scripts/pipeline/weekly_retrain (auto_retrain, full_retrain, quick_retrain, get_matchweek_status, rollback). No importers (entry point).
- **Quality signals:** Well-structured Click CLI with clear command grouping (scraping, parsing, features, pipeline, ML). All imports are lazy-loaded within command functions (good practice). Comprehensive logging configured. Type hints present in function signatures (e.g., `season: str | None`, `model_types: tuple`). No docstrings on individual command functions are minimal (one-liners suffice for CLI). Code is clean: ~535 LOC, no duplication. Minor issue: some commands have duplicate option definitions (e.g., `--season`, `--model` appear multiple times) which could be refactored into shared decorators. Error handling is light (e.g., `fetch_weather` does basic path check but no try-catch). Configuration validation is delegated to imported modules.
- **Improve:** 1) Extract repeated Click options (--season, --model, --limit) into reusable decorator functions to reduce duplication. 2) Add try-catch blocks around command execution to provide better error messages. 3) Add more granular logging (e.g., log when a command starts/ends). 4) Consider a config validation command that checks all data dependencies before running any pipeline step. 5) Document the expected runtime for long-running commands (e.g., optimize takes 2-3 hours).

### `config/` — 7 files

#### ⚫ `config/__init__.py` — grade A · keep
- **Does:** Empty module marker file for Python package recognition.
- **Talks to:** No imports or exports.
- **Quality signals:** Standard empty __init__.py with no code.

#### 🟢 `config/api_keys.py` — grade A · keep
- **Does:** Centralized API key loader from environment variables; single source of truth for all external service credentials.
- **Talks to:** Imported by: scripts/betting/player_prop_odds.py (uses get_odds_api_key, get_telegram_token, get_telegram_chat_id). Loads from: dotenv (load_dotenv), os.environ.
- **Quality signals:** Well-structured getter functions (8 functions total), clear docstrings, type hints, enforces environment-only pattern (no JSON fallback). No dead code. All getters are simple and defensive.

#### 🔧 `config/betting_rules.py` — grade B · keep
- **Does:** Loads and evaluates betting rules from betting_rules.json; encapsulates walk-forward-validated strategy rules including edge thresholds, Kelly staking, odds bands, and paper-trade gates.
- **Talks to:** Imports: json, logging, dataclasses, pathlib, typing. Loads from: config/betting_rules.json file. Used by betting engine to evaluate BetCandidate against rules.
- **Quality signals:** Well-structured with frozen dataclasses (BetCandidate, BetDecision), comprehensive rule evaluation in evaluate() method (Kelly fraction, paper-trade gate, odds bands, edge thresholds). Fact card shows live=false and dead_candidate=true, but code shows it is intentionally a rule loader module. No obvious dead code, but imported_by list is empty despite being a core betting component.

#### 🟢 `config/leagues.py` — grade A · keep
- **Does:** Master registry of multi-league configuration: league metadata (API IDs, timezone, draw rate), derby/rivalry definitions by intensity, promoted team tracking per season, and team-to-league inference.
- **Talks to:** Imported by 24 files across features, scraper, scripts, web, and utils modules (see fact card for full list). Exports: LEAGUE_REGISTRY, ACTIVE_LEAGUES, DERBIES, PROMOTED_TEAMS, backward-compat MATCHWEEKS_PER_SEASON, helper functions (get_league_config, get_derbies, get_promoted_teams, infer_league, get_matchweeks).
- **Quality signals:** Frozen dataclass LeagueConfig with 13 immutable fields per league, 5 leagues registered with complete metadata, comprehensive DERBIES dict with intensity levels (3=city, 2=regional, 1=traditional), PROMOTED_TEAMS per league per season, helper functions with proper error handling (KeyError with message), backward-compatible aliases. Well-commented. No dead code.

#### 🔧 `config/managers.py` — grade B · keep
- **Does:** Static historical manager tenure data for Serie A and Premier League; supports manager-change feature engineering and form discontinuity detection.
- **Talks to:** Imports: pandas. Called by (intended): features/manager_changes.py, features/build.py. Exports: MANAGER_TENURES list of 475 tuples, resolve_manager(team, match_date), backfill_managers(matches DataFrame).
- **Quality signals:** Static data module with 475 tuples spanning Serie A (2017-present) and Premier League (2005-present). Two functions: resolve_manager (date range lookup with fallback), backfill_managers (NaN-fill helper). Fact card shows live=true but imported_by=["features/build.py"], meaning only one importer. Code is clean with no dead branches, but data is stale (last update appears to be late 2025 or early 2026, with future manager placeholders).

#### 🟢 `config/settings.py` — grade A · keep
- **Does:** Global configuration constants for FBref scraping, HTTP client setup, seasons list, league codes, directory paths, and atomic file write utilities.
- **Talks to:** Imported by 119 files (per fact card) across all major subsystems: features, ml, monitoring, scraper, scripts, storage, tests, web. Exports: get_current_season(), FBref/HTTP constants, SEASONS list, LEAGUES dict, path constants (PROJECT_ROOT, DATA_DIR, etc.), atomic_write_json(), atomic_write_parquet().
- **Quality signals:** Clean separation of concerns: HTTP config (delay, timeout, retries, User-Agent), season lists (21 seasons), path utilities with Path object (safe cross-platform), atomic write helpers using temp file pattern (prevents corruption). Type hints on functions, proper error handling in atomic writes. 156 LOC, no dead code, all exports are used.

#### 🟢 `config/team_names.py` — grade A · keep
- **Does:** Canonical team name mapping for all supported leagues (Serie A, Premier League, La Liga, Bundesliga, Ligue 1); normalizes variant names from different data sources to canonical names.
- **Talks to:** Imported by 29 files (per fact card) across features, parser, scraper, scripts, storage, web. Exports: per-league name dicts (SERIE_A_NAMES, PREMIER_LEAGUE_NAMES, etc.), TEAM_NAME_MAP (combined), current-season team lists (SERIE_A_2025_26, PREMIER_LEAGUE_2025_26), normalize_team(), normalize_team_safe(), strip_accents().
- **Quality signals:** Comprehensive mappings: 123 Serie A variants, 81 Premier League, 71 La Liga, 72 Bundesliga, 73 Ligue 1. Case-insensitive fallback with pre-built lowercase dict. Helper functions with NaN safety and accent normalization for fuzzy matching. Actively maintained with 2025-2026 season lists. 563 LOC but highly repetitive (data, not code complexity). No dead code.

### `storage/` — 4 files

#### ⚫ `storage/__init__.py` — grade F · **delete**
- **Does:** Empty module initialization file.
- **Talks to:** None.
- **Quality signals:** Single newline. No code, no docstring, no exports. 1 LOC.
- **Verdict reason:** True garbage with zero value. No code executed, no imports exported. Not a package marker in use (storage.paths, storage.raw, storage.structured are all direct imports). Safe to delete.

#### 🟢 `storage/paths.py` — grade A · keep
- **Does:** Centralized path factory functions for all data artifacts (raw HTML, fixtures, parsed Parquet, ML features, registry). Ensures directory creation on demand.
- **Talks to:** Imports from: config.settings (DATA_DIR, FEATURES_DIR, MODELS_DIR, PARSED_DIR, RAW_FIXTURES_DIR, RAW_HTML_DIR, REGISTRY_PATH). Imported by: cli.py, 13 features/*.py files, 4 ml/*.py files, 4 scraper/*.py files, 6 parser/scripts/analysis/*.py files, 7 scripts/models/*.py files, 2 scripts/pipeline/*.py files, storage/raw.py, storage/structured.py, tests/test_integration.py.
- **Quality signals:** Type-hinted function signatures (return Path, str|None league param). Clear docstrings for all public functions. DRY: centralized logic for mkdir operations prevents scattered mkdir calls. 59 LOC, no dead code or unused imports. Critical infrastructure: used by 45+ files. Predictable naming convention (html_*_path, fixtures_*_path, parsed_path, features_path, registry_path).

#### 🟢 `storage/raw.py` — grade A · keep
- **Does:** I/O primitives for raw HTML match pages: save HTML to disk, load from disk, check existence. Wraps storage/paths.html_match_path() to hide path construction.
- **Talks to:** Imports from: storage.paths (html_match_path). Imported by: scraper/match_reports.py.
- **Quality signals:** Type-hinted functions (season: str, match_id: str -> Path/str/bool). Docstrings for all functions. 23 LOC, no dead code. Clean abstraction: three focused functions with clear intent (save_html, load_html, html_exists). Uses pathlib for cross-platform compatibility.

#### 🟢 `storage/structured.py` — grade B · keep
- **Does:** Transform and write parsed match data (MatchData objects) into structured Parquet tables (matches, player_stats, goalkeeper_stats, shots, lineups, events). Handles data type coercion, matchweek inference from Sofascore fixtures or chronology, deduplication, and atomic append operations with file locking.
- **Talks to:** Imports from: config.settings (atomic_write_parquet), config.team_names (normalize_team), models.schemas (MatchData), parser.events (events_to_records), parser.lineups (lineups_to_records), storage.paths (parsed_path). Imported by: cli.py.
- **Quality signals:** Type hints on key functions (list[MatchData], list[dict], pd.DataFrame). Docstrings on public and private functions. 369 LOC with substantial logic: _coerce_numeric_columns handles FBref data type issues, _fill_missing_matchweeks implements two-tier fallback (Sofascore lookup + chronological inference), _append_or_create uses fcntl advisory locking to prevent concurrent R-M-W races and atomic writes (tmp+rename). Logging throughout. Exception handling around Sofascore fixture loading. Minor: optional conditional import of normalize_team (lines 102-104) is defensive but slightly verbose; hardcoded KNOWN_NUMERIC list (lines 51-59) could drift from schema.

### `scraper/` — 24 files

#### ⚫ `scraper/__init__.py` — grade F · **delete**
- **Does:** Empty module initialization file.
- **Talks to:** No imports or exports.
- **Quality signals:** Empty file (1 line), serves no purpose.
- **Verdict reason:** Scraper is a package that works fine without explicit __init__.py; modern Python 3.3+ treats directories as namespace packages.

#### 🟢 `scraper/client.py` — grade A · keep
- **Does:** Rate-limited HTTP client with Cloudflare bypass via curl_cffi Chrome TLS impersonation and exponential backoff retries.
- **Talks to:** imported_by: scraper/fixtures.py, scraper/match_reports.py; imports: config/settings.py (REQUEST_DELAY_SECONDS, MAX_RETRIES, BACKOFF_FACTOR, HEADERS, REQUEST_TIMEOUT, MAX_BACKOFF)
- **Quality signals:** Well-structured with docstrings, type hints, clean separation of concerns (cffi vs requests fallback). FBrefClient class with _HttpError exception, _wait() rate limiter, exponential backoff logic. Handles transient 429/50x errors gracefully. ~140 LOC.

#### ⚫ `scraper/cloudflare_solver.py` — grade B · keep
- **Does:** Solve Cloudflare interactive challenges via undetected-chromedriver or Selenium to extract cf_clearance cookies for reuse in plain requests.
- **Talks to:** imported_by: none; imports: none (minimal dependencies)
- **Quality signals:** 210 LOC, well-documented, proper Selenium/undetected-chromedriver abstractions, _harden_isolation() prevents DevTools leaks to Arc, _wait_for_challenge() polls for Cloudflare resolution. However, is not imported by any live module.

#### ⚫ `scraper/fbref_auto_scraper.py` — grade B · keep
- **Does:** Fully automated FBref scraper using persistent botasaurus browser session with Cloudflare bypass, rate limiting, and per-league support (Serie A, EPL).
- **Talks to:** imported_by: none; imports: botasaurus.browser (optional), logs to stderr
- **Quality signals:** 300 LOC, standalone CLI with argparse, proper League/Season configuration, rate-limit tuning per league (4s Serie A, 8s EPL), comprehensive HTML validation. Sits alongside fbref_fast.py and fbref_selenium.py as tier-based fallback options but is not imported.

#### ⚫ `scraper/fbref_fast.py` — grade B · keep
- **Does:** Fast FBref scraper with botasaurus browser reuse, parsing full match reports (team stats, player stats, shots) for bulk seasonal data.
- **Talks to:** imported_by: none; imports: config/settings.py (DATA_DIR, REQUEST_DELAY_SECONDS), config/team_names.py (normalize_team), botasaurus.browser (optional)
- **Quality signals:** 609 LOC, large FBrefFastScraper class with comprehensive table parsing, match report scraping, fixtures/player stats methods. 403/429 detection with exponential backoff. Well-structured but not called by live modules.

#### ⚫ `scraper/fbref_selenium.py` — grade A · keep
- **Does:** Fallback FBref scraper using visible Selenium Chrome (Tier 2) or undetected-chromedriver (Tier 3) when botasaurus fails.
- **Talks to:** imported_by: none; imports: config/settings.py (FBREF_BASE_URL, SERIE_A_COMP_ID), storage/paths.py (html_match_path, html_season_dir), Selenium/undetected-chromedriver (optional)
- **Quality signals:** 226 LOC, clean FBrefSeleniumScraper class with context manager support (__enter__/__exit__), proper Cloudflare wait logic, dynamic table polling (scrolling), rate limiting. Implements visible (human-like) browsing for bot evasion.

#### 🟢 `scraper/fixtures.py` — grade A · keep
- **Does:** Scrape Serie A fixture/results page from FBref with date parsing, team normalization, xG extraction, and CSV output for the season.
- **Talks to:** imported_by: cli.py, scraper/upcoming.py; imports: config/settings.py (FBREF_BASE_URL, SERIE_A_COMP_ID), config/team_names.py (normalize_team), parser/html_utils.py (_uncomment_tables), scraper/client.py (FBrefClient), storage/paths.py (fixtures_csv_path)
- **Quality signals:** 100+ LOC (partial read), well-documented, handles multiple leagues (FBREF_COMP_IDS mapping), type hints, helper parsers (_safe_int, _safe_float, _parse_date), proper error logging.

#### 🟢 `scraper/footballdata_lineups.py` — grade B · keep
- **Does:** Fetch confirmed lineups from football-data.org API (backup source after Sofascore), with per-team name mapping and rate limiting.
- **Talks to:** imported_by: scraper/lineup_fetcher.py; imports: config/settings.py (DATA_DIR, UPCOMING_DIR), config/team_names.py (normalize_team), requests (optional)
- **Quality signals:** 100+ LOC (partial read), API client with key-based auth, _api_get() helper with rate limiting (429 backoff 60s), team name map (_FD_NAME_MAP), proper error handling (403 check).

#### 🟢 `scraper/historical.py` — grade A · keep
- **Does:** Import historical match data from football-data.co.uk CSVs into matches.parquet schema; supports Series A and other major leagues with venue/city mappings.
- **Talks to:** imported_by: cli.py; imports: config/settings.py (DATA_DIR, DEFAULT_LEAGUE, LEAGUES, SEASONS), config/team_names.py (normalize_team), scraper/odds.py (download_odds), storage/paths.py (parsed_path)
- **Quality signals:** 100+ LOC (partial read), large static mappings (TEAM_CITIES, TEAM_VENUES) for venue/weather features, handles multi-league data with normalized output, imports from scraper/odds.py for odds enrichment.

#### 🟢 `scraper/injuries.py` — grade A · keep
- **Does:** Scrape current injury data from Transfermarkt and ESPN for Serie A players using concurrent fetching with per-domain rate limiting.
- **Talks to:** imported_by: features/injury_impact.py, scripts/analysis/data_quality_report.py, scripts/pipeline/run_full_pipeline.py, scripts/prediction/ensemble_prediction_engine.py, scripts/prediction/generate_epl_supplementary.py, scripts/prediction/predict_unified.py; imports: config/settings.py (DATA_DIR), scraper/transfermarkt.py (TM_HEADERS)
- **Quality signals:** 100+ LOC, concurrent ThreadPoolExecutor for parallel team scraping (30-60s for 20 teams vs 4-20min sequential), @dataclass for InjuryRecord, comprehensive Transfermarkt team ID mappings (SERIE_A_TEAMS_TM, EPL_TEAMS_TM), ESPN fallback, proper regex parsing of injury text.

#### 🟢 `scraper/lineup_fetcher.py` — grade A · keep
- **Does:** Multi-source confirmed lineup fetcher (cascade: Sofascore → football-data.org → API-Football) with player name fuzzy matching and 60-90 min pre-kickoff confirmation.
- **Talks to:** imported_by: features/player_xg_model.py, scraper/squad_fetcher.py, scripts/analysis/player_analyzer.py, scripts/pipeline/run_full_pipeline.py, web/app.py; imports: config/leagues.py (get_league_config, LEAGUE_REGISTRY), config/settings.py, config/team_names.py (normalize_team), features/player_xg_model.py, scraper/footballdata_lineups.py, scraper/sofascore_lineups.py, scripts/utils/match_timing.py
- **Quality signals:** 100+ LOC (partial read), LineupFetcher class with API key fallback, player name normalization (unicode handling, accent stripping), fuzzy matching threshold (0.75), @dataclass MatchLineup, comprehensive multi-league support.

#### 🟢 `scraper/match_reports.py` — grade A · keep
- **Does:** Download match report HTML pages from FBref incrementally, tracking state in registry for resumable downloads.
- **Talks to:** imported_by: cli.py; imports: config/settings.py (OLD_HTML_DIR), scraper/client.py (FBrefClient), scraper/registry.py (Registry), storage/paths.py (fixtures_csv_path, html_match_path), storage/raw.py (save_html)
- **Quality signals:** 80+ LOC (partial read), incremental download with skip-existing logic, registry-based state tracking, error counting with retry, CSV-based fixture input.

#### 🟢 `scraper/odds.py` — grade A · keep
- **Does:** Download historical betting odds from football-data.co.uk (FREE CSVs with 15+ bookmakers, 1993-present).
- **Talks to:** imported_by: cli.py, features/build.py, scraper/historical.py; imports: config/settings.py (DATA_DIR, DEFAULT_LEAGUE, LEAGUES), pandas, requests
- **Quality signals:** 80+ LOC (partial read), static CSV URL mapping per league/season, ODDS_COLUMNS with comprehensive bookmaker coverage (Bet365, Pinnacle, Betfair, etc.), opening/closing odds tracking.

#### 🟢 `scraper/referee.py` — grade B · keep
- **Does:** Scrape referee-per-match data from worldfootball.net for Series A, filling the missing referee column in historical imports.
- **Talks to:** imported_by: cli.py, features/build.py, scripts/pipeline/refresh_weekly_data.py; imports: config/settings.py (DATA_DIR), config/team_names.py (normalize_team), pandas, requests, BeautifulSoup
- **Quality signals:** 80+ LOC (partial read), worldfootball.net scraper with season ID mappings, regex-based referee/team parsing, multi-league support (WF_LEAGUE_SLUGS, WF_COMP_CODES).

#### 🟢 `scraper/registry.py` — grade A · keep
- **Does:** JSON-backed manifest tracking scraping and parsing state per match (downloaded_at, parsed, parsed_at) for resumable downloads.
- **Talks to:** imported_by: cli.py, parser/match_page.py, scraper/match_reports.py; imports: storage/paths.py (registry_path)
- **Quality signals:** @dataclass RegistryEntry with ISO timestamps, Registry class with load/save logic, query methods (is_downloaded, is_parsed, get_unparsed), mutation methods (mark_downloaded, mark_parsed), file-based persistence at data/registry.json.

#### 🟢 `scraper/sofascore_events.py` — grade A · keep
- **Does:** Scrape match incidents (goals, cards, subs, captain data) from Sofascore API with persistent session, connection reuse, and exponential backoff.
- **Talks to:** imported_by: scraper/sofascore_lineups.py; imports: pandas, curl_cffi (optional), requests (fallback)
- **Quality signals:** 572 LOC, session reuse with impersonation rotation, 8+ consecutive failure detection with 3-min pause + reset, exponential backoff (up to 60s), checkpoint saves every 100 matches, retry logic. Comprehensive incident parsing (_parse_incidents, _parse_captains) with dataclass-like row dicts.

#### 🟢 `scraper/sofascore_lineups.py` — grade B · keep
- **Does:** Fetch confirmed starting XIs from Sofascore API ~60 minutes before kickoff using existing curl_cffi + Cloudflare bypass infrastructure.
- **Talks to:** imported_by: scraper/lineup_fetcher.py; imports: config/settings.py (DATA_DIR, UPCOMING_DIR), config/team_names.py (normalize_team), config/leagues.py (get_league_config, LEAGUE_REGISTRY), scraper/sofascore_events.py (_get_json, _jitter_delay, _BASE_URL)
- **Quality signals:** 100+ LOC (partial read), _normalize_sofascore_team() with explicit mapping, get_sofascore_match_ids() builds match key → sofascore_id lookup, reuses sofascore_events infra for API calls.

#### 🟢 `scraper/sofascore_standings.py` — grade A · keep *(added 2026-07-16)*
- **Does:** Scrape a league table off Sofascore's ISR-fresh **HTML** tournament page (`__NEXT_DATA__` blob) — the canonical fallback when the API 403s. Owns the retry/backoff, the sentinel schema-break check, the negative cache and the failure breaker.
- **Talks to:** imports: config/settings.py (get_current_season); imported_by: web/app.py (aliased to the original private `_live_standings_via_html` names — every call site unchanged), scripts/data/sofascore_watcher.py
- **Why it exists:** extracted verbatim out of `web/app.py:335-524` so the watcher could share it — **importing `web.app` costs 134ms and starts TWO daemon threads** (`_auto_settle_loop` and the odds auto-poll the project CLAUDE.md documents as a credit burn). This module: 9ms, zero threads, no Flask. Proven a pure move — all 190 lines reproduce byte-for-byte under a 9-name rename map.
- **Two known gaps, pinned by tests, deliberately not fixed (a move is not a behaviour change):** season is stamped from `get_current_season()` (boundary Aug 1), **not read off the page** — they disagree until Aug 1; and the sentinel checks one team, so it cannot catch a partial table (the 14-of-20 oracle is how).
- **Quality signals:** 190 LOC, 16 tests against a committed real specimen (`tests/fixtures/sofascore/tournament_standings_serie_a.json`). No network in tests.

#### 🟢 `scraper/squad_fetcher.py` — grade B · keep
- **Does:** Live squad roster fetcher from API-Football (primary) and Football-Data.org (fallback) with 7-day cache TTL.
- **Talks to:** imported_by: scripts/pipeline/run_full_pipeline.py; imports: config/settings.py (DATA_DIR), config/team_names.py (TEAM_NAME_MAP), scraper/lineup_fetcher.py (normalize_player_name, standardize_api_team_name), requests (optional)
- **Quality signals:** 80+ LOC (partial read), SquadFetcher class, position mapping for API-Football/Football-Data.org, cache freshness check (_is_cache_fresh), 7-day TTL.

#### 🟢 `scraper/transfermarkt.py` — grade A · keep
- **Does:** Scrape player market values and transfer data from Transfermarkt HTML (fallback) after preferred kaggle/transfermarkt-datasets option.
- **Talks to:** imported_by: cli.py, features/build.py, scraper/injuries.py; imports: config/settings.py (DATA_DIR, HEADERS), pandas, requests, BeautifulSoup
- **Quality signals:** 100+ LOC (partial read), comprehensive league/team ID mappings (SERIE_A_TEAMS_TM, EPL_TEAMS_TM), HTML scraping with BeautifulSoup, per-team and per-player value fetching.

#### 🟢 `scraper/understat_scraper.py` — grade B · keep
- **Does:** Scrape xG data from Understat for Serie A (2014+) using headless Chrome; falls back to regex extraction if JS fails.
- **Talks to:** imported_by: scripts/pipeline/refresh_weekly_data.py, scripts/pipeline/run_full_pipeline.py; imports: config/settings.py, pandas, requests, Selenium (optional)
- **Quality signals:** 100+ LOC (partial read), UnderstatScraper class with headless Chrome via Selenium, JS extraction + regex fallback, rate limiting (3s), season map builder, league slug mapping.

#### ⚫ `scraper/upcoming.py` — grade B · keep
- **Does:** Fetch upcoming Serie A fixtures for prediction using football-data.co.uk CSV source.
- **Talks to:** imported_by: none (scraped_by cli.py internally); imports: config/settings.py, config/team_names.py (normalize_team), scraper/fixtures.py (_make_match_id), storage/paths.py (PARSED_DIR), pandas, requests
- **Quality signals:** 80+ LOC (partial read), UpcomingMatch dataclass, fetch_fixtures_from_football_data() function, filters to unplayed matches.

#### 🟢 `scraper/weather.py` — grade A · keep
- **Does:** Fetch historical weather data from Open-Meteo API (free, no key) for match venues using city coords; supports Serie A and EPL.
- **Talks to:** imported_by: cli.py, scripts/pipeline/refresh_weekly_data.py, scripts/pipeline/run_full_pipeline.py; imports: config/settings.py (DATA_DIR), pandas, requests
- **Quality signals:** 80+ LOC (partial read), static VENUE_COORDS mapping for 30+ cities (Serie A + EPL + historical teams), archive-api.open-meteo.com endpoint, cached at weather.parquet.

#### ⚫ `scraper/xcomp_scraper.py` — grade B · keep
- **Does:** Scrape extra-competition fixtures (Coppa Italia, Champions League, Europa League) from FotMob API for congestion features in Serie A predictions.
- **Talks to:** imported_by: none; imports: config/settings.py (DATA_DIR), requests
- **Quality signals:** 80+ LOC (partial read), standalone scrape_competition() function with FotMob league IDs, per-season match collection, JSON-based output.

### `parser/` — 11 files

#### ⚫ `parser/__init__.py` — grade A · keep
- **Does:** Package initialization file (empty).
- **Talks to:** Not imported anywhere; does not import anything.
- **Quality signals:** Empty file, valid Python package marker. No code to evaluate.

#### 🟢 `parser/events.py` — grade A · keep
- **Does:** Extracts match events (goals, cards, substitutions, penalties, own goals) from FBref scorebox HTML.
- **Talks to:** Imported by: parser/match_page.py (parse_events), storage/structured.py (events_to_records). Imports: config/team_names.py (normalize_team), models/schemas.py (MatchEvent).
- **Quality signals:** 154 LOC with full type hints and docstrings. Robust minute parsing with regex fallback for unicode variants (', ', 0027, 2032). Handles multiple event types (goal, penalty, own_goal, yellow_card, red_card, substitution) with icon class inspection. Utility function _minute_sort_key for ordering. No dead branches.

#### 🟢 `parser/goalkeeper_stats.py` — grade A · keep
- **Does:** Extracts goalkeeper statistics per team from FBref keeper_stats_{team_hash} table.
- **Talks to:** Imported by: parser/match_page.py (parse_goalkeeper_stats). Imports: parser/html_utils.py (find_table, table_to_dataframe).
- **Quality signals:** 50 LOC with clear type hints. Single focused function that filters empty rows, inserts context columns (match_id, team, is_home), and returns records. Handles missing tables gracefully with debug logging.

#### ⚫ `parser/html_parser.py` — grade D · **delete**
- **Does:** Legacy HTML parsing functions for FBref fixtures, player stats, and match reports using pd.read_html and BeautifulSoup.
- **Talks to:** Not imported anywhere (imported_by: []). Imports: argparse, logging, re, pathlib, pandas, bs4.
- **Quality signals:** 406 LOC with type hints and docstrings, but: (1) not imported by any active code; (2) Pyright reports multiple unresolved import errors (pandas, bs4) and attribute access issues on NavigableString (lines 136, 195, 202, 226); (3) duplicates functionality already present in match_page.py, events.py, lineups.py, etc.; (4) has an __main__ CLI entry point (lines 383-405) suggesting standalone operation, but not reachable from cli.py.
- **Verdict reason:** Replaced by modular parsers in match_page.py and specialty modules. Zero importers. Type errors indicate it's not maintained. The __main__ entry suggests it was a one-shot backfill tool; no evidence of ongoing use.

#### 🟢 `parser/html_utils.py` — grade A · keep
- **Does:** Shared BeautifulSoup utility functions for extracting data from FBref HTML tables using data-stat attributes and safe type conversion.
- **Talks to:** Imported by: parser/goalkeeper_stats.py (find_table, table_to_dataframe), parser/match_page.py (get_soup), parser/player_stats.py (find_table, table_to_dataframe), parser/scorebox.py (safe_int), parser/shots.py (find_table, table_to_dataframe), parser/team_stats.py (safe_float, safe_int), scraper/fixtures.py (unknown which functions). Imports: pandas, bs4 (no internal imports).
- **Quality signals:** Well-documented functions with type hints. 136 LOC of clear, reusable extraction logic. Handles edge cases (csk sort keys, multi-row headers, empty tables). Properly separates concerns (get_soup, find_table, table_to_dataframe as layered abstractions).

#### 🟢 `parser/lineups.py` — grade B · edit
- **Does:** Extracts team lineups (formation, starters, bench) from FBref div.lineup elements.
- **Talks to:** Imported by: parser/match_page.py (parse_lineups), storage/structured.py (lineups_to_records). Imports: config/team_names.py (normalize_team), models/schemas.py (LineupInfo).
- **Quality signals:** 148 LOC with type hints. Complex formation regex (lines 43-45), player extraction fallbacks for <tr>, <li>, and <a> tags, bench detection with multiple language support (Substitutes, panchina). One minor issue: normalize_team is imported but never used in the module.
- **Verdict reason:** Live, working well, but remove unused import of normalize_team from config/team_names.py.

#### 🟢 `parser/match_page.py` — grade A · keep
- **Does:** Orchestrator: parses complete FBref match reports by calling all specialized parsers (scorebox, team_stats, players, GK, shots, lineups, events) and assembles into MatchData.
- **Talks to:** Imported by: cli.py (parse_all_matches, parse_match). Imports: models/schemas.py (MatchData), parser/events.py (parse_events), parser/goalkeeper_stats.py (parse_goalkeeper_stats), parser/html_utils.py (get_soup), parser/lineups.py (parse_lineups), parser/player_stats.py (parse_player_stats), parser/scorebox.py (parse_scorebox), parser/shots.py (parse_shots), parser/team_stats.py (parse_team_stats), scraper/registry.py (Registry), storage/paths.py (html_match_path).
- **Quality signals:** 176 LOC with clear orchestration pattern. Robust team hash discovery from regex patterns on table IDs (handles fallbacks). Detailed logging of parsed row counts. Registry integration for progress tracking. Graceful error handling with exc_info=True. Warnings collection for partial parse failures. All specialized parsers tested as dependencies.

#### 🟢 `parser/player_stats.py` — grade B · edit
- **Does:** Extracts and merges all 6 per-player stat tables (summary, passing, passing_types, defense, possession, misc) per team from FBref.
- **Talks to:** Imported by: parser/match_page.py (parse_player_stats). Imports: parser/html_utils.py (find_table, table_to_dataframe).
- **Quality signals:** 101 LOC with type hints, clear TABLE_TYPES constant (lines 24-31). Handles missing summary table gracefully (line 69 warning). Column prefixing for non-summary tables (lines 72-78) avoids clashes. Left merge preserves summary structure. Pyright warns about unresolved pandas import (environment issue, not code issue). One minor: line 73 redundant replace (table_type already replaced in prefix).
- **Verdict reason:** Live parser with solid merging logic. Minor: remove redundant .replace('_types', '_types') on line 73; just use table_type directly in prefix.

#### 🟢 `parser/scorebox.py` — grade A · keep
- **Does:** Extracts match metadata from FBref scorebox: teams, scores, xG, date, venue, attendance, referee, managers, captains.
- **Talks to:** Imported by: parser/match_page.py (parse_scorebox). Imports: config/team_names.py (normalize_team), models/schemas.py (MatchMetadata), parser/html_utils.py (safe_int).
- **Quality signals:** 216 LOC with thorough type hints. Handles multiple date formats (7 formats, lines 209-214). Regex patterns for matchweek, attendance, venue, referee. Robust fallbacks for missing data (direct children indexing line 35-40, venuetime @data-* attributes line 66, strong>a tags line 73). Safe text extraction with normalize_team.

#### 🟢 `parser/shots.py` — grade A · keep
- **Does:** Extracts shot data (player, xG, outcome, body part, distance) from FBref shots_all table.
- **Talks to:** Imported by: parser/match_page.py (parse_shots). Imports: config/team_names.py (normalize_team), parser/html_utils.py (find_table, table_to_dataframe).
- **Quality signals:** 49 LOC with clear type hints. Handles missing table gracefully. Normalizes team names for cross-team identification (line 45). Adds context columns (match_id, is_home_team_shot). Clean lambda expression for normalization.

#### 🟢 `parser/team_stats.py` — grade A · keep
- **Does:** Extracts team-level aggregate statistics from div#team_stats (possession, passes, shots, tackles, cards, etc.) and div#team_stats_extra.
- **Talks to:** Imported by: parser/match_page.py (parse_team_stats). Imports: parser/html_utils.py (safe_float, safe_int).
- **Quality signals:** 163 LOC with clear helper functions (_parse_main_stats, _parse_extra_stats, _normalize_stat_key, _parse_stat_value). Handles FBref's HTML structure quirks: colspan headers, <strong> value extraction, icon-based card counts, and fraction parsing (e.g., '2 of 4 — 50%'). Comprehensive type hints.

### `features/` — 59 files

#### 🟢 `features/__init__.py` — grade A · keep
- **Does:** Empty package marker for the features module.
- **Talks to:** None — package initialization file with no imports or exports.
- **Quality signals:** Minimal and correct for its role as a package marker.

#### 🟢 `features/_utils.py` — grade B · keep
- **Does:** Shared utilities for feature modules: team name normalization, safe division, match ID bridging, and team merging logic.
- **Talks to:** Imported by: advanced_player.py, advanced_player_sofascore.py, advanced_shots.py, captain_features.py, card_timing.py, fbref_features.py, sofascore_features.py. Provides functions like _safe_div(), norm_team(), build_id_bridge(), merge_side().
- **Quality signals:** Provides core utilities (safe division, team normalization, merge logic) that are used by 7 feature files. Well-named constants like SOFASCORE_TEAM_MAP. No apparent dead code or redundancy.

#### 🟢 `features/advanced_player.py` — grade B · keep
- **Does:** Computes advanced team metrics from player-level data: Gini coefficient for squad depth, star form signals, and player contribution distributions.
- **Talks to:** Imports: features/_utils.py, storage/paths.py. Imported by: features/build.py. Functions: _gini(), _compute_advanced_team_metrics(), _compute_star_form(), add_advanced_player_features().
- **Quality signals:** Well-documented docstrings; implements Gini coefficient and form metrics. Handles squad depth heterogeneity. Uses shift(1) to prevent leakage. No obvious issues.

#### 🟢 `features/advanced_player_sofascore.py` — grade B · keep
- **Does:** Advanced player metrics extracted from SofaScore live data (shooting patterns, positioning distribution).
- **Talks to:** Imports: features/_utils.py. Imported by: features/build.py.
- **Quality signals:** Augments feature set with SofaScore-specific player data. Dependency on _utils for team normalization suggests shared logic.

#### 🟢 `features/advanced_shots.py` — grade B · keep
- **Does:** Shot-level xG analysis: shot location distribution, efficiency ratios, conversion consistency across match phases.
- **Talks to:** Imports: features/_utils.py, storage/paths.py. Imported by: features/build.py.
- **Quality signals:** Extracts shot distribution and efficiency; handles phase-based splits. Shift(1) prevents data leakage.

#### 🟢 `features/bankroll_manager.py` — grade A · keep
- **Does:** Betting system Phase 4.1 & 4.2: bankroll tracking with position sizing, drawdown limits, Kelly criterion stake calculation, and bet logging with P&L tracking.
- **Talks to:** Imports: config/settings.py. Imported by: scripts/analysis/performance_dashboard.py, scripts/prediction/predict_unified.py, scripts/utils/alert_system.py, tests/test_betting_logic.py, tests/test_edge_cases.py. Classes: BankrollManager, BettingTracker. Provides backward-compat functions: calculate_kelly_stake(), calculate_value(), load_history(), get_performance_stats().
- **Quality signals:** Comprehensive implementation: 2 main classes (BankrollManager, BettingTracker) with detailed docstrings, proper state management (JSON persistence), risk controls (drawdown limits, daily loss caps), Kelly fraction alignment (0.10) across tests. ~580 LOC with clear separation of concerns. Used by 5 importers (live and tests).

#### 🟢 `features/base.py` — grade B · keep
- **Does:** Baseline team match log construction from match-level data: pivots matches into team perspective with basic stats (goals, possession, shots).
- **Talks to:** Imports: storage/paths.py. Imported by: features/build.py. Functions: build_team_match_log(), _perspective(), _result().
- **Quality signals:** Foundation for team-level feature engineering; defines STAT_COLUMNS constant. Properly handles perspective transformation (home→team, away→opponent).

#### 🟢 `features/bookmaker_analysis.py` — grade B · keep
- **Does:** Analyzes sharp vs soft bookmaker divergence: compares Pinnacle (sharp) odds with market average to identify where smart money disagrees, Phase 2 market intelligence.
- **Talks to:** Imports: config/settings.py. Imported by: scripts/pipeline/run_full_pipeline.py, scripts/prediction/generate_epl_supplementary.py. Classes: BookmakerAnalyzer. Functions: run_bookmaker_analysis().
- **Quality signals:** Implements divergence analysis between SHARP_BOOKMAKERS (Pinnacle) and SOFT_BOOKMAKERS (market). Vig-free conversion logic. Used in 2 pipeline scripts.

#### 🟢 `features/build.py` — grade A · keep
- **Does:** Master feature engineering pipeline: orchestrates 43+ feature modules into a cohesive DataFrame, manages caching, handles feature validation and leakage detection.
- **Talks to:** Imports from 42 feature modules (advanced_player through xi_quality) + config, scraper, storage modules. Imports also: config/leagues.py, config/managers.py, config/settings.py, scraper/odds.py, scraper/referee.py, scraper/transfermarkt.py, storage/paths.py. Imported by: cli.py, ml/data.py, scripts/models/*.py (6 files), scripts/pipeline/*.py (2 files), tests/test_integration.py. Core classes: FeaturePipeline (with 43+ Step* plugin classes), BackfillManagersPlugin, BackfillRefereesPlugin. Functions: build_features(), build_upcoming_features(), get_ml_feature_columns().
- **Quality signals:** ~2300 LOC, 43 feature modules orchestrated into plugin chain. Sophisticated caching with fingerprinting (_cache_path, _source_fingerprint, _is_cache_valid). Proper leakage prevention with validation checks. Used by 11 importers (live and tests). Clear step ordering: base team log → rolling → home_away → strength → ... → final derived features.

#### 🟢 `features/captain_features.py` — grade B · keep
- **Does:** Fantasy football captain impact: identifies key captains and their form state, team dependence on star performers.
- **Talks to:** Imports: features/_utils.py. Imported by: features/build.py.
- **Quality signals:** Specialized feature for captain-dependent teams; uses shared normalization utilities.

#### 🟢 `features/card_timing.py` — grade B · keep
- **Does:** Card accumulation and timing patterns: tracks yellow/red card distributions by match phase to detect refereeing trends and tactical aggression.
- **Talks to:** Imports: config/settings.py, features/_utils.py. Imported by: features/build.py.
- **Quality signals:** Derives card timing and phase-based distribution; uses shared utilities.

#### 🟢 `features/congestion.py` — grade B · keep
- **Does:** Fixture congestion features: identifies midweek matches, short rest periods, and heavy scheduling windows using rolling time-based counts.
- **Talks to:** Imported by: features/build.py. No explicit imports from other features modules.
- **Quality signals:** Well-documented logic: midweek detection (Tue-Thu), rest_days thresholds, 15/30-day rolling match counts. Proper null-safety with pd.notna() checks. ~105 LOC.

#### 🟢 `features/creative_factors.py` — grade B · keep
- **Does:** Creative play indicators: assists, key passes, and creative pressure metrics that capture playmaking beyond shots/xG.
- **Talks to:** Imports: config/leagues.py. Imported by: features/build.py.
- **Quality signals:** League-aware feature engineering; uses config for league-specific parameters.

#### 🟢 `features/cross_market_analysis.py` — grade B · keep
- **Does:** Cross-market correlation detection: identifies signals by correlating h2h, totals, and spreads markets to detect offensive/defensive dominance, BTTS agreement.
- **Talks to:** Imports: config/settings.py. Imported by: scripts/pipeline/run_full_pipeline.py. Classes: CrossMarketAnalyzer. Reads from data/upcoming/odds_full.json, odds_movement.json.
- **Quality signals:** Specialized market analysis; reads live odds data. Implements _get_main_totals() helper. Used in pipeline.

#### 🟢 `features/derived.py` — grade B · keep
- **Does:** Derived team-level features: computes second-order stats from base features (e.g., streaks, win% by scoreline ranges).
- **Talks to:** Imported by: features/build.py. No explicit external imports.
- **Quality signals:** Post-pivot second-order feature engineering.

#### 🟢 `features/draw_detection.py` — grade B · keep
- **Does:** Draw outcome detection: models probability of draw using team balance metrics, historical draw rates, and draw indicators.
- **Talks to:** Imports: config/settings.py. Imported by: features/prediction_calibration.py, scripts/prediction/ensemble_prediction_engine.py. Classes: DrawDetector. Functions: get_draw_detector().
- **Quality signals:** Specialized draw prediction component; constant DRAW_INDICATORS; used in 2 prediction modules.

#### 🟢 `features/enhanced_momentum.py` — grade B · keep
- **Does:** Advanced momentum metrics: big win momentum, comeback patterns, late goal trends, pressure response, scoring patterns with time-decay weighting.
- **Talks to:** Imported by: features/build.py, scripts/prediction/ensemble_prediction_engine.py. Functions: add_momentum_features(), compute_big_win_momentum(), compute_comeback_momentum(), compute_late_goal_trend(), compute_pressure_response(), add_enhanced_momentum_features(), compute_momentum_composite().
- **Quality signals:** Specialized momentum detection; 9 functions cover different momentum aspects (big wins, comebacks, late goals, pressure response). Used in build.py and ensemble prediction.

#### 🟢 `features/enhanced_weather.py` — grade B · keep
- **Does:** Weather impact modeling: temperature, wind, rain effects on game dynamics with venue-specific patterns.
- **Talks to:** Imports: features/venue.py. Imported by: features/build.py, scripts/prediction/ensemble_prediction_engine.py.
- **Quality signals:** Depends on venue.py for stadium data; used in 2 prediction contexts.

#### 🟢 `features/european_congestion.py` — grade B · keep
- **Does:** European competition congestion flags: identifies teams playing European matches (UEFA, Champions League) and their scheduling stress.
- **Talks to:** Imported by: features/build.py, tests/simulator/test_european_congestion.py.
- **Quality signals:** Specialized European competition detection; has associated test.

#### 🟢 `features/fbref_features.py` — grade B · keep
- **Does:** FBref advanced stats integration: possession-adjusted metrics, shot efficiency, defensive actions from FBref-scraped data.
- **Talks to:** Imports: config/settings.py, config/team_names.py, features/_utils.py. Imported by: features/build.py.
- **Quality signals:** Uses shared _utils and team_names config; integrates FBref data sources.

#### 🟢 `features/first_half_splits.py` — grade B · keep
- **Does:** First-half performance analysis: segregates first-half stats (goals, xG, pace) to capture match tempo and early dominance signals.
- **Talks to:** Imported by: features/build.py, tests/simulator/test_first_half_splits.py.
- **Quality signals:** Specialized first-half stats; has associated test.

#### 🟢 `features/formation_analysis.py` — grade B · keep
- **Does:** Formation and system analysis: detects team formations, computes formation stability, and formation matchup compatibility.
- **Talks to:** Imports: config/settings.py, storage/paths.py. Imported by: features/build.py, scripts/prediction/ensemble_prediction_engine.py.
- **Quality signals:** Formation-specific analysis; used in 2 contexts.

#### 🟢 `features/gk_quality.py` — grade B · keep
- **Does:** Goalkeeper quality rating: aggregates GK performance from historical save%, distribution, and positioning metrics.
- **Talks to:** Imports: storage/paths.py. Imported by: features/build.py.
- **Quality signals:** Specialized GK evaluation; reads from storage paths.

#### 🟢 `features/goal_features.py` — grade A · keep
- **Does:** Goal volume prediction features: xG totals, attack-defense mismatch, BTTS probability, Poisson analytical priors, scoring variance, and derby suppression.
- **Talks to:** Imported by: features/build.py. Functions: add_goal_features(). ~186 LOC.
- **Quality signals:** Well-documented 20 feature additions: volume predictors (total_xg_expected, combined goals), BTTS predictors (scoring_rate_5, conceding_rate_5, btts_probability_naive), Poisson priors (poisson_over_*, poisson_btts), matchup context (h2h_goals_vs_avg, clean_sheet_matchup, derby_goal_suppression). Proper null-safety with fillna() defaults. Clear logging of feature counts. No dead code.

#### 🟢 `features/h2h.py` — grade A · keep
- **Does:** Head-to-head feature computation: aggregates prior meetings between teams with capping (MAX_H2H_MEETINGS=7), recency weighting, and recent-5 win rates.
- **Talks to:** Imported by: features/build.py. Constants: MAX_H2H_MEETINGS (set to 7 to prevent drift). Functions: add_h2h_features(). ~177 LOC.
- **Quality signals:** Sophisticated H2H feature set: 13 columns (h2h_matches_played, h2h_home_wins, h2h_draws, h2h_goals_avg, h2h_home_win_rate, h2h_last_result, h2h_weighted_home_win_rate, h2h_recent_5_win_rate, h2h_recent_3_total_goals). Exponential recency decay (DECAY_RATE=0.3). Proper reverse-perspective normalization for swapped fixtures. Comment explains capping to prevent CV drift. No dead code.

#### 🟢 `features/home_away.py` — grade B · keep
- **Does:** Home/away split stats: separates performance into home and away contexts with rolling averages for both venues.
- **Talks to:** Imports: config/settings.py. Imported by: features/build.py.
- **Quality signals:** Venue-specific rolling stats; uses config.

#### 🟢 `features/injury_impact.py` — grade A · keep
- **Does:** Injury impact modeling with position-aware weighting: position-specific disruption weights (GK 0.85, FB 0.50), player importance from xG/xA, partnership breakdowns, returning player penalties.
- **Talks to:** Imports: config/leagues.py, config/settings.py, scraper/injuries.py. Imported by: features/build.py, scripts/prediction/ensemble_prediction_engine.py, scripts/prediction/predict_unified.py. Functions: _load_player_profiles(), _get_team_profiles(), _infer_position(), get_player_importance(), add_injury_impact(). Constants: POSITION_WEIGHTS, PARTNERSHIP_UNITS.
- **Quality signals:** Sophisticated injury modeling: POSITION_WEIGHTS dict with 14 positions (GK 0.85 down to FW 0.55), PARTNERSHIP_UNITS for co-injury penalties, player profile caching via JSON. Reads from scraper/injuries.py. Used in 3 importers. ~80+ LOC of documented logic for player importance calculation.

#### 🟢 `features/league_position.py` — grade A · keep
- **Does:** League standing and motivation features: computes current standings (position, points, GD) at each match with no leakage, plus motivation indicators (relegation zone, CL zone, title race).
- **Talks to:** Imported by: features/build.py. Functions: add_league_position_features(), _compute_standings(). ~184 LOC.
- **Quality signals:** Comprehensive standings computation: 13+ features per team (position, points, GD, goals_for, wins, draws, losses, motivation zones). Proper cumulative update logic that computes standings BEFORE each match (prevents leakage). League-aware grouping (by season+league). Position momentum (last 5 matches) and position_momentum_diff. No dead code.

#### 🟢 `features/lineup_stats.py` — grade B · keep
- **Does:** Lineup strength assessment from team composition: player quality aggregation, availability status, squad freshness.
- **Talks to:** Imports: config/settings.py, storage/paths.py. Imported by: scripts/models/btts_corners_model.py, scripts/models/cards_model.py.
- **Quality signals:** Used by 2 specialized models; reads from storage paths.

#### 🟢 `features/lineup_xg.py` — grade B · keep
- **Does:** Lineup-based team xG prediction: aggregates individual player xG profiles to team-level xG based on actual lineups.
- **Talks to:** Imports: config/team_names.py. Imported by: features/build.py.
- **Quality signals:** Specialized lineup-based xG; uses team_names config.

#### 🟢 `features/manager.py` — grade A · keep
- **Does:** Manager tenure and head-to-head features: tracks manager history, detects new appointments, computes manager vs manager historical win rates.
- **Talks to:** Imported by: features/build.py. Functions: get_manager_h2h(), add_manager_features(). ~239 LOC.
- **Quality signals:** Manager tenure tracking: 6 features (tenure, is_new flag, manager_changed). H2H computation: wins_home, wins_away, draws, total, home_winrate, confidence levels. Proper handling of missing manager data (NaN when unavailable). Leakage-safe: uses only prior matches for H2H. ~50+ lines dedicated to manager state machine logic. Comments explain confidence thresholds (5+ matches=high, 3+=medium, 1+=low, 0=none).

#### 🟢 `features/market_intelligence.py` — grade B · keep
- **Does:** Market intelligence aggregation: unifies signals from bookmaker analysis, odds movement, and cross-market correlations into composite score.
- **Talks to:** Imports: config/settings.py. Imported by: scripts/pipeline/run_full_pipeline.py, scripts/prediction/ensemble_prediction_engine.py. Classes: MarketIntelligence. Reads from data/upcoming/bookmaker_analysis.json, odds_movement.json, cross_market_signals.json.
- **Quality signals:** Aggregates 3 upstream market signals; used in 2 contexts (pipeline and ensemble).

#### 🟢 `features/match_patterns.py` — grade B · keep
- **Does:** Match pattern detection: identifies recurring team patterns (high-scoring, defensive, comeback-prone) using statistical clustering.
- **Talks to:** Imported by: features/build.py.
- **Quality signals:** Pattern detection module; actively imported.

#### 🟢 `features/missing_players.py` — grade B · keep
- **Does:** Missing player impact: identifies unavailable players due to injury/suspension and computes aggregated absence impact.
- **Talks to:** Imported by: features/build.py, tests/simulator/test_missing_players.py.
- **Quality signals:** Absence tracking; has associated test.

#### 🟢 `features/odds_features.py` — grade A · keep
- **Does:** Odds-derived features: sharp-soft divergence, best-available probabilities, market-implied goal totals, line velocity (opening → closing odds movement), steam move flags.
- **Talks to:** Imported by: features/build.py. Functions: add_odds_derived_features(). ~165 LOC.
- **Quality signals:** Comprehensive odds feature engineering: 14+ features including sharp-soft divergence (Pinnacle - market), best probabilities (Pinnacle coalescing to market), implied goal total, odds consistency (sharp vs soft agreement), line velocity for 1X2 and O/U markets, steam move flags (>3% implied prob shift). Proper NaN-safety. Well-structured with clear feature categories. No dead code.

#### 🟢 `features/player_depth.py` — grade B · keep
- **Does:** Squad depth analysis: bench strength, positional cover, rotation patterns from squad composition.
- **Talks to:** Imports: features/sofascore_features.py. Imported by: features/build.py.
- **Quality signals:** Depends on sofascore_features for data; actively imported.

#### 🟢 `features/player_impact.py` — grade B · keep
- **Does:** Individual player impact on team performance: identifies high-impact players and their contribution weighting.
- **Talks to:** Imports: storage/paths.py. Imported by: features/build.py.
- **Quality signals:** Player-level impact assessment; reads from storage.

#### 🟢 `features/player_xg_model.py` — grade A · keep
- **Does:** Player-level xG modeling Phase 2: builds individual player xG/90 profiles, tracks recent form with exponential recency weighting, models key player absence impact.
- **Talks to:** Imports: config/settings.py, config/team_names.py, scraper/lineup_fetcher.py, storage/paths.py. Imported by: scraper/lineup_fetcher.py, scripts/analysis/backtest_multimarket.py, scripts/prediction/ensemble_prediction_engine.py. Classes: PlayerXGProfile (~115 lines with properties xg_per_90, xa_per_90, xg_xa_per_90, recent_form_xg). Functions: _load_player_profiles(), _get_team_profiles(), _infer_position(), get_player_importance(). ~100+ LOC.
- **Quality signals:** Sophisticated player modeling: PlayerXGProfile class with 11+ tracked attributes (player_name, team, position, matches_played, total_xg, total_xa, total_goals, total_assists, recent_matches). xG/90 and xA/90 properties with minute-based normalization. recent_form_xg with exponential decay weights (most recent=weight 5, then 4,3,2,1). Proper handling of edge cases (< 90 min = return career avg; < 3 recent = revert to xg_per_90). Used in 3 importers (lineup_fetcher, backtest, ensemble).

#### 🟢 `features/prediction_calibration.py` — grade A · keep
- **Does:** Prediction calibration for ensemble outputs: applies post-hoc bias correction (home advantage, live adjustment, draw detection), betting strategy recommendations.
- **Talks to:** Imports: features/draw_detection.py. Imported by: scripts/prediction/ensemble_prediction_engine.py. Classes: PredictionCalibrator, HomeAdvantageCalibrator, LiveBiasCorrector, CalibrationPipeline. Functions: get_calibration_pipeline(), list_strategies(). Constant: BETTING_STRATEGIES.
- **Quality signals:** Sophisticated calibration pipeline: 4 calibration classes handling different bias sources. Imports draw_detection for draw probability adjustment. CalibrationPipeline orchestrates multiple calibrators. list_strategies() function suggests betting strategy recommendations. Used in ensemble_prediction_engine.

#### 🟢 `features/pressing.py` — grade B · keep
- **Does:** Pressing intensity metrics: PPDA (passes per defensive action), pressure efficiency, defensive aggression levels.
- **Talks to:** Imports: config/settings.py. Imported by: features/build.py.
- **Quality signals:** PPDA-based pressing metrics; uses config.

#### 🟢 `features/referee.py` — grade B · keep
- **Does:** Referee performance features: card distributions by referee, home bias indicators, VAR decision impact.
- **Talks to:** Imports: storage/paths.py. Imported by: features/build.py, scripts/prediction/referee_integration.py.
- **Quality signals:** Referee-specific features; used in 2 contexts.

#### 🟢 `features/rest_days.py` — grade A · keep
- **Does:** Rest days computation: calculates days since last match within season, capped at MAX_REST_DAYS=21, with congestion flags.
- **Talks to:** Imported by: features/build.py. Functions: add_rest_days(), pivot_rest_to_match(). Constants: MAX_REST_DAYS=21. ~62 LOC.
- **Quality signals:** Simple but critical feature: rest_days (computed via .diff().dt.days), capped at 21 to exclude summer breaks/COVID gaps, is_congested flag (rest <= 3). Proper season grouping (prevents cross-season rest computation). Comment explains reasoning for MAX_REST_DAYS cap. Two functions: team-level (add_rest_days) and match-level pivot (pivot_rest_to_match).

#### 🟢 `features/rolling.py` — grade A · keep
- **Does:** Rolling average features: computes sliding-window statistics (goals, xG, possession, etc.) over configurable windows with proper leakage prevention via shift(1).
- **Talks to:** Imports: config/settings.py. Imported by: features/build.py. Functions: add_rolling_features(). Constants: ROLLING_STATS (14 stats: goals_scored through red_cards).
- **Quality signals:** Core rolling stats: 14 statistics across configurable windows (from config.settings.ROLLING_WINDOWS). Uses .shift(1).rolling() pattern to prevent data leakage (current match never included in its own features). Season-aware grouping prevents rolling averages bleeding across seasons. ROLLING_STATS constant clearly documents available stats.

#### 🟢 `features/sentiment_analysis.py` — grade B · keep
- **Does:** Pre-match sentiment analysis Phase 4: keyword-based scoring for injury news, team morale, motivation signals, public perception.
- **Talks to:** Imports: config/settings.py. Imported by: scripts/prediction/ensemble_prediction_engine.py. Functions: add_sentiment_features(). Constants: POSITIVE_KEYWORDS, NEGATIVE_KEYWORDS (with tuples of category + impact weight).
- **Quality signals:** Keyword-based sentiment: POSITIVE_KEYWORDS dict with entries like {"strong": ("high confidence", 0.1), "recovered": ("positive injury news", 0.06)}. Covers injury sentiment, morale, motivation, and public perception. ~60+ LOC of keyword definitions.

#### 🟢 `features/shot_level_xg.py` — grade B · keep
- **Does:** Shot-level xG feature extraction: shot location distribution, high-value shots, efficiency metrics per shot type.
- **Talks to:** Imported by: features/build.py, tests/simulator/test_shot_level_xg.py.
- **Quality signals:** Shot-level analysis; has associated test.

#### 🟢 `features/shot_quality.py` — grade B · keep
- **Does:** Shot quality metrics: conversion efficiency, shot type distribution, finishing consistency.
- **Talks to:** Imports: storage/paths.py. Imported by: features/build.py.
- **Quality signals:** Shot quality assessment; reads from storage.

#### 🟢 `features/situational_xg.py` — grade B · keep
- **Does:** Situational xG analysis: breaks down xG by game state (score margin, phase of match, pressure level).
- **Talks to:** Imported by: features/build.py, tests/simulator/test_situational_xg.py.
- **Quality signals:** Situation-based xG decomposition; has associated test.

#### 🟢 `features/sofascore_features.py` — grade B · keep
- **Does:** SofaScore live data integration: real-time player ratings, match events, possession dynamics, tactical transitions.
- **Talks to:** Imports: config/settings.py, config/team_names.py, features/_utils.py. Imported by: features/build.py, features/player_depth.py. Functions: add_sofascore_features().
- **Quality signals:** SofaScore data integration; uses shared _utils and team_names; used in 2 contexts.

#### 🟢 `features/sofascore_indices.py` — grade B · keep
- **Does:** SofaScore derived indices: composite team strength metrics from live ratings and event data.
- **Talks to:** Imported by: features/build.py.
- **Quality signals:** Derived SofaScore metrics; actively imported.

#### 🟢 `features/strength.py` — grade A · keep
- **Does:** Attack/defense strength ratings (Dixon-Coles Poisson) and Elo ratings: attack_strength = team_avg_goals / league_avg_goals, with ELO_K=20, ELO_HOME_ADV=100, season reversion at 75% old + 25% mean.
- **Talks to:** Imported by: features/build.py. Functions: add_strength_ratings(), add_elo_ratings(), _compute_elo_for_group(). Constants: ELO_K=20, ELO_HOME_ADV=100, ELO_INITIAL=1500, ELO_REVERT=0.75, ELO_PROMOTED=1400.
- **Quality signals:** Sophisticated Elo and strength rating: attack_strength (team goals / league goals), defense_strength (team conceded / league conceded), with xG variants. Elo computation with season boundaries (new teams start at ELO_PROMOTED=1400, existing teams revert 25% toward mean). K-value adjusts early in season (K=40 for 0-5 matches, K=30 for 5-15, K=20 after). League-aware when league column present. ~187 LOC with proper edge case handling.

#### 🟢 `features/substitution_features.py` — grade B · keep
- **Does:** Substitution pattern features: rolling averages of sub count, substitution timing, and first sub minute from ledger tracking.
- **Talks to:** Imports: config/team_names.py, scripts/utils/ledger.py. Imported by: features/build.py. Functions: _build_team_timeline(), add_substitution_features(). Constants: _LOOKBACK_N=10. ~60+ LOC shown.
- **Quality signals:** Substitution pattern tracking: avg_subs_per_game, avg_sub_minute, avg_first_sub_minute, sub_games_tracked. Reads from JSON ledger (data/lineup_history/substitutions.json) via scripts/utils/ledger. Proper leakage prevention (only records with date < current match). Lookback window of 10 matches.

#### 🟢 `features/suspensions.py` — grade B · keep
- **Does:** Player suspensions tracking: identifies banned players and computes aggregated suspension impact.
- **Talks to:** Imports: storage/paths.py. Imported by: features/build.py.
- **Quality signals:** Suspension impact tracking; reads from storage.

#### 🟢 `features/team_aggregates.py` — grade B · keep
- **Does:** Team-level aggregates: cumulative season stats (total goals, wins, points) at each match boundary.
- **Talks to:** Imports: storage/paths.py. Imported by: features/build.py.
- **Quality signals:** Cumulative team stats; reads from storage.

#### 🟢 `features/transfer_impact_analysis.py` — grade B · keep
- **Does:** Transfer window impact: tracks squad turnover, new signing integration time, squad stability metrics.
- **Talks to:** Imports: config/settings.py. Imported by: features/build.py.
- **Quality signals:** Transfer-based squad change tracking; uses config.

#### 🟢 `features/understat_features.py` — grade B · keep
- **Does:** Understat data integration: shot-level analytics, team xG efficiency, defensive actions from Understat scrape.
- **Talks to:** Imports: config/team_names.py. Imported by: features/build.py, scripts/analysis/data_quality_report.py.
- **Quality signals:** Understat shot and xG data; used in 2 contexts.

#### 🟢 `features/value_betting.py` — grade A · keep
- **Does:** Value betting detection: Kelly criterion optimization, EV calculation, value betting pipeline for model outputs.
- **Talks to:** Imports: config/settings.py. Imported by: scripts/prediction/predict_unified.py. Classes: KellyCriterion, ValueBettingDetector, EVCalculator, ValueBettingPipeline. Functions: get_value_pipeline(). ~100+ LOC of implementation.
- **Quality signals:** Value betting infrastructure: 4 classes covering Kelly optimization, EV detection, and pipeline orchestration. Proper betting logic (EV = (prob * odds) - 1). Used in predict_unified.py pipeline.

#### 🟢 `features/venue.py` — grade A · keep
- **Does:** Venue features: stadium-specific effects (travel distance, altitude, capacity) with haversine distance calculation and all-league stadium database.
- **Talks to:** Imported by: features/build.py, features/enhanced_weather.py, scripts/prediction/ensemble_prediction_engine.py. Functions: haversine_distance(), get_travel_distance(), get_altitude_difference(), get_stadium_capacity(), add_venue_features(). Constants: SERIE_A_STADIUMS, EPL_STADIUMS, LA_LIGA_STADIUMS, BUNDESLIGA_STADIUMS, LIGUE_1_STADIUMS, ALL_STADIUMS.
- **Quality signals:** Comprehensive venue data: haversine distance for travel calculations, altitude differences, stadium capacities for 5 leagues. Constants define stadium databases (SERIE_A_STADIUMS, etc.). Used in 3 contexts (build, enhanced_weather, ensemble). Proper geographic calculations.

#### 🟢 `features/xg_trends.py` — grade B · keep
- **Does:** xG trend analysis: rolling xG momentum, directional trends (increasing/decreasing efficiency), xG regression detection.
- **Talks to:** Imports: config/settings.py. Imported by: features/build.py.
- **Quality signals:** xG trend detection; uses config.

#### 🟢 `features/xi_quality.py` — grade B · keep
- **Does:** Starting XI quality assessment: lineup strength rating, player quality aggregation by position, tactical balance evaluation.
- **Talks to:** Imported by: features/build.py.
- **Quality signals:** XI quality assessment; actively imported.

### `ml/` — 21 files

#### 🟢 `ml/__init__.py` — grade A · keep
- **Does:** Empty module initializer (no content, single blank line)
- **Talks to:** None — this is a standalone marker file
- **Quality signals:** Minimal by design; empty __init__.py is standard Python package practice. No bugs possible.

#### 🟢 `ml/calibration.py` — grade A · keep
- **Does:** Probability calibration pipeline: ProbabilityCalibrator (Platt scaling), TemperatureScaler (uniform temperature scaling), PerClassTemperatureScaler (class-specific temperature), and AutoCalibrator (wrapper selecting best method via cross-validation)
- **Talks to:** IMPORTED BY ml/ensemble.py, ml/training.py. IMPORTS ml/config.py, ml/data.py, ml/evaluation.py, ml/tuning.py, config/settings.py. fit_calibrator_from_cv() and class constructors are the main entry points.
- **Quality signals:** 4 well-documented classes with full docstrings; Platt scaling via LogisticRegression, temperature via sklearn.isotonic.IsotonicRegression; fit_calibrator_from_cv uses walk-forward CV to avoid leakage; returns ECE/log_loss metrics; proper handling of edge cases (single class, no NaN). Type hints throughout. ~400 lines.

#### ⚫ `ml/comparison.py` — grade B · **delete**
- **Does:** Model comparison framework: compare_runs() loads two model versions and computes metric deltas; compare_features() analyzes feature set differences with Spearman rank correlation
- **Talks to:** IMPORTED BY nobody (dead_candidate=true). IMPORTS config/settings.py. Functions: _load_metadata(), _load_cv_results(), _load_training_report() (load JSON files), compare_runs(), compare_features(), print_comparison_report().
- **Quality signals:** Well-structured functions (5 public); clear logic for metric delta computation and feature importance correlation; uses scipy.stats.spearmanr for rank correlation; returns structured dicts; print_comparison_report() formats as readable table. ~150 lines. BUT: zero imports from ml/ (only config/settings.py), zero importers → never called.
- **Verdict reason:** Dead code: zero references, zero importers. Likely a leftover from an experiment or manual comparison workflow. If needed, should be moved to a scripts/analysis/ one-shot diagnostic or integrated into ensemble_prediction_engine.py if actively used.

#### 🟢 `ml/config.py` — grade A · keep
- **Does:** Centralized configuration constants and dataclass definitions for the entire ML pipeline: label mappings, hyperparameters, feature selection thresholds, and model type definitions
- **Talks to:** IMPORTED BY 17 files (cli.py, ml/calibration.py, ml/data.py, ml/ensemble.py, ml/evaluation.py, ml/feature_selection.py, ml/models.py, ml/ou_model.py, ml/persistence.py, ml/prediction.py, ml/training.py, ml/tuning.py, multiple scripts). No imports of other ml/ modules — purely definitional.
- **Quality signals:** Well-organized: LABEL_MAP, RESULT_COL, MODEL_TYPES as module-level constants; ModelConfig/ValidationConfig/etc. as @dataclass with detailed docstrings. drop_meta() utility function. ODDS_COLUMN_PATTERNS list with 16 odds feature categories. Comprehensive comments explaining rationale (e.g., ODDS_META_KEEP section). Type hints on all dataclass fields. ~180 lines, no dead code.

#### 🟢 `ml/correction_layer.py` — grade A · keep
- **Does:** Post-ensemble probability calibration via two mechanisms: StaticCorrector (logistic regression on OOF predictions with context features) and RollingCorrector (EMA of prediction errors bucketed by confidence band). Unified CorrectionLayer orchestrates both.
- **Talks to:** IMPORTED BY ml/training.py, scripts/betting/auto_settle.py, scripts/pipeline/scheduler.py, scripts/prediction/ensemble_prediction_engine.py. IMPORTS config/settings.py. Core classes: StaticCorrector, RollingCorrector, CorrectionLayer. Ledger functions: append_to_ledger(), read_ledger().
- **Quality signals:** Sophisticated dual-mechanism design: StaticCorrector uses sklearn.linear_model.LogisticRegression with inner CV for C selection; RollingCorrector tracks EMA per bucket (predicted_class × confidence_band); CorrectionLayer unifies both with save/load/correct APIs. Extensive docstrings; proper handling of leakage (walk-forward CV), edge cases (missing context, NaN), and graceful fallbacks (passthrough if model fails). ~775 lines with detailed comments on design rationale.

#### 🟢 `ml/data.py` — grade A · keep
- **Does:** Data loading and time-series splitting: DataLoader class encapsulates parquet reading with smart imputation (feature-type-specific strategies), TimeSeriesSplitter provides walk-forward splits with optional purge for lookahead bias
- **Talks to:** IMPORTED BY cli.py, ml/calibration.py, ml/ensemble.py, ml/feature_selection.py, ml/ou_model.py, ml/prediction.py, ml/training.py, ml/tuning.py, 6 scripts. IMPORTS ml/config.py, features/build.py. Classes: DataLoader, TimeSeriesSplitter. Helper functions: _is_rolling_feature(), _is_h2h_feature(), ..., _smart_impute(), _add_availability_flags(), _latest_season().
- **Quality signals:** Well-designed DataLoader with feature-aware imputation strategies (rolling→0, h2h→default, odds→-1, etc.). TimeSeriesSplitter prevents data leakage via _purge_lookahead_matches (removes matches too close to test date). Smart imputation helpers categorize features (rolling, h2h, elo, odds, player aggregates, etc.) with 17 classification functions. Extensive logging. ~500 lines, no dead functions.

#### 🟢 `ml/draw_specialist.py` — grade A · keep
- **Does:** Specialized binary classifier for predicting draws: DrawSpecialist class trains on draw-specific features (elo_factor, strength_factor, h2h_draw_rate, etc.) and provides draw probability adjustment for ensemble, plus feature importance explanation
- **Talks to:** IMPORTED BY scripts/analysis/train_draw_specialist.py, scripts/models/train_draw_specialist_production.py. IMPORTS no ml/ modules. Classes: DrawPrediction (dataclass), DrawSpecialist. Main functions: compute_draw_specific_features(), train_draw_specialist().
- **Quality signals:** Focused design with 12 draw-specific feature computations (elo_factor via exponential decay, strength_factor, form_convergence, etc.). DrawSpecialist.fit() handles class imbalance via scale_pos_weight; supports CatBoost/XGBoost/sklearn fallback. predict_draw_proba() and adjust_ensemble_probs() provide clean APIs. get_draw_factors() explains per-match contributions. Full docstrings, type hints, logging. ~395 lines.

#### 🟢 `ml/ensemble.py` — grade A · keep
- **Does:** Ensemble model orchestration: WeightedAverageEnsemble combines fold models with learned blend weights; build_fold_models() trains component models (xgboost/lightgbm/catboost) per fold; evaluate_ensemble_cv() runs cross-validation with ensemble; _optimize_blend_weights() uses scipy.optimize for weight tuning
- **Talks to:** IMPORTED BY ml/training.py, scripts/analysis/backtest_multimarket.py, scripts/analysis/backtest_unified.py, scripts/models/train_league.py, scripts/pipeline/weekly_retrain.py, scripts/prediction/ensemble_prediction_engine.py. IMPORTS ml/calibration.py, ml/config.py, ml/data.py, ml/evaluation.py, ml/tuning.py, config/settings.py, storage/paths.py. Classes: WeightedAverageEnsemble. Functions: _normalize_probs(), _optimize_blend_weights(), evaluate_ensemble_cv(), build_fold_models(), load_fold_model().
- **Quality signals:** Sophisticated blend-weight optimization via scipy.optimize with constraints (weights sum to 1, bounds [0, 1]). WeightedAverageEnsemble supports soft voting and prediction. evaluate_ensemble_cv() uses walk-forward with proper metric computation (log_loss, ECE, Brier). build_fold_models() persists each fold model independently. Well-documented with extensive logging. ~450 lines.

#### 🟢 `ml/evaluation.py` — grade A · keep
- **Does:** Evaluation metrics computation: ranked_probability_score(), expected_calibration_error(), kelly_profit_simulation(); calibration_curve_data() builds binned calibration curves; compute_metrics_with_ci() wraps all metrics with confidence intervals and baseline comparison
- **Talks to:** IMPORTED BY cli.py, ml/calibration.py, ml/ensemble.py, ml/training.py, ml/walk_forward.py, scripts/analysis/backtest_unified.py, scripts/models/retrain_no_odds_catboost.py, scripts/models/train_league.py, scripts/models/train_no_odds.py. IMPORTS ml/config.py. Functions: ranked_probability_score(), expected_calibration_error(), kelly_profit_simulation(), calibration_curve_data(), compute_metrics_with_ci(), compute_baseline_metrics(), compute_metrics(), _multiclass_brier(), print_report().
- **Quality signals:** Comprehensive metrics suite: RPS (proper scoring rule), ECE (probability calibration), Brier (squared error), log_loss, accuracy, F1 per class. kelly_profit_simulation() computes expected growth via Kelly criterion. calibration_curve_data() bins predictions into 10 bins for confidence-accuracy plot. All functions well-documented with detailed docstrings. ~350 lines.

#### 🟢 `ml/feature_selection.py` — grade A · keep
- **Does:** Feature engineering and selection: identify_odds_columns() detects betting-odds-derived features; exclude_odds() removes them for training no-odds models; importance_based_selection() ranks features by tree model importance; correlation_pruning() removes collinear features; save_importance_history() logs feature importance over time
- **Talks to:** IMPORTED BY cli.py, ml/training.py, scripts/diagnostics (7 diagnostic scripts), scripts/models (9 training scripts). IMPORTS ml/config.py, ml/data.py, ml/tuning.py, config/settings.py. Functions: identify_odds_columns(), exclude_odds(), importance_based_selection(), correlation_pruning(), save_importance_history(), compare_importance().
- **Quality signals:** Well-structured feature engineering pipeline: identify_odds_columns() uses ODDS_COLUMN_PATTERNS from config; exclude_odds() handles both full exclusion and ODDS_META_KEEP subset; importance_based_selection() uses either tree feature_importances_ or permutation importance; correlation_pruning() uses Spearman for feature pairs. Extensive logging with feature counts at each step. ~250 lines.

#### 🔧 `ml/meta_learner.py` — grade A · keep
- **Does:** Context-dependent ensemble blending via meta-learner: generates OOS predictions from each component (xG, market, factor, ML) via walk-forward; trains CatBoost on component probs + context features (rest, derby, promoted, etc.); MetaLearner learns when to trust which predictor based on match situation
- **Talks to:** IMPORTED BY nobody (dead_candidate=true). No imports from ml/ modules (self-contained). Classes: MetaLearnerConfig, MetaLearner. Functions: poisson_win_prob(), compute_xg_predictions(), compute_market_predictions(), compute_factor_predictions(), compute_ml_predictions_walkforward(), compute_context_features(). Main script with argparse CLI.
- **Quality signals:** Sophisticated walk-forward meta-learning architecture: generates 1000+ LOC of well-engineered code. Component prediction functions (xG, market, factor) are fully implemented with calibration. ML walk-forward generates OOS predictions via season-by-season training. Context extraction handles rest, derby, promoted teams, international breaks. MetaLearner class provides train_final(), predict(), save(), load() APIs. BUT: no importer (not called from main pipeline) → one_shot/experimental analysis script.

#### 🟢 `ml/models.py` — grade A · keep
- **Does:** Model wrapper classes for different gradient boosting libraries: ModelWrapper base class, XGBoostModel, LightGBMModel, CatBoostModel each encapsulate library-specific training/prediction logic; get_model() factory returns appropriate wrapper; _chronological_val_split() prevents time-series leakage in validation splits
- **Talks to:** IMPORTED BY ml/persistence.py, ml/training.py, scripts/models/train_league.py, scripts/models/train_no_odds.py. IMPORTS ml/config.py. Classes: ModelWrapper (base), XGBoostModel, LightGBMModel, CatBoostModel. Functions: _chronological_val_split(), get_model(). Constant: _MODEL_REGISTRY.
- **Quality signals:** Clean abstraction over three gradient boosting libraries with unified API (fit, predict, predict_proba, get_feature_importance). Each subclass handles library-specific hyperparameters and early stopping. _chronological_val_split() ensures validation set is strictly after training set (no information leakage). ~250 lines with full docstrings and type hints.

#### ⚫ `ml/ou_model.py` — grade C · **delete**
- **Does:** Over/Under and BTTS prediction: trains 3-model ensemble (XGBoost/LightGBM/CatBoost) for goal markets (O/U 1.5/2.5/3.5, BTTS); uses Poisson-derived features as ML input so models learn when Poisson is right vs wrong
- **Talks to:** IMPORTED BY nobody (dead_candidate=true, live=false). IMPORTS ml/config.py, ml/data.py, ml/tuning.py, config/settings.py, storage/paths.py. Defines GOAL_FEATURE_PRIORITIES list (80 features) and partial skeleton (first 80 lines shown).
- **Quality signals:** Well-named module with clear intent; GOAL_FEATURE_PRIORITIES list is comprehensive (Poisson priors, xG features, goal-specific metrics, BTTS, attacking/defending). BUT: file is 80% empty after feature list — actual training/prediction functions not implemented or existing beyond first 80 lines. Zero importers → never called.
- **Verdict reason:** Dead code: incomplete implementation (skeleton only), zero importers. Goal markets are valuable (O/U, BTTS), but this module was never finished. Recommend: either complete it properly for 1X2 ensemble inclusion or remove entirely.

#### 🟢 `ml/persistence.py` — grade A · keep
- **Does:** Model I/O: save_model() serializes trained models to disk with metadata (feature names, hyperparameters, metrics); load_model() deserializes and returns wrapper object; _timestamp() generates ISO timestamps for versioning
- **Talks to:** IMPORTED BY cli.py, ml/prediction.py, ml/training.py, scripts/models/train_league.py, scripts/models/train_no_odds.py. IMPORTS ml/config.py, ml/models.py, config/settings.py. Functions: _timestamp(), save_model(), load_model().
- **Quality signals:** Clean serialization layer: save_model() uses joblib for models + JSON for metadata, preserving feature order and hyperparameters for reproducibility. load_model() handles different model types (xgboost/lightgbm/catboost) via _MODEL_REGISTRY lookup. Graceful error handling. ~100 lines, no dead code.

#### 🟢 `ml/poisson.py` — grade A · keep
- **Does:** Shared Poisson math utilities for match outcome prediction: poisson_1x2() converts xG to 1X2 probabilities via Poisson PMF enumeration; poisson_1x2_vec() batch version; market_implied_probs() extracts implied probabilities from Pinnacle odds; calculate_over_probability() computes P(goals > threshold); poisson_win_prob() wrapper returning dict with H/D/A keys
- **Talks to:** IMPORTED BY 17 files (scripts/analysis/backtest_unified.py, scripts/models (11 model scripts), tests/test_integration.py). IMPORTS scipy.stats.poisson, numpy. Pure utility functions — no ml/ imports. Functions: poisson_probability(), poisson_cumulative(), poisson_1x2(), poisson_1x2_vec(), market_implied_probs(), calculate_over_probability(), poisson_win_prob().
- **Quality signals:** Focused utility module with 7 functions covering Poisson calculations. Each function well-documented with examples; type hints throughout. poisson_1x2() handles edge cases (clamps xG, fallback to [0.4, 0.3, 0.3] if total=0). market_implied_probs() normalizes Pinnacle odds and handles missing values. ~140 lines, no unused functions.

#### ⚫ `ml/prediction.py` — grade B · **delete**
- **Does:** Match outcome prediction using trained models: predict_upcoming() loads a trained model and predicts probabilities for upcoming matches; predict_with_ensemble() wrapper for ensemble predictions
- **Talks to:** IMPORTED BY nobody (dead_candidate=true, live=false). IMPORTS ml/config.py, ml/persistence.py, ml/data.py. Functions: predict_upcoming(), predict_with_ensemble() (partial skeleton shown in first 80 lines).
- **Quality signals:** Well-structured predict_upcoming() function with proper handling of missing features (fill with NaN, log warning) and imputation via _smart_impute(). Returns DataFrame with home_team, away_team, prob_H/D/A, predicted, confidence. BUT: zero importers → never called. predict_with_ensemble() skeleton present but incomplete.
- **Verdict reason:** Dead code: unused despite clean implementation. Prediction logic is likely integrated into ensemble_prediction_engine.py or other scripts. Recommend consolidating into a single canonical prediction module.

#### 🟢 `ml/statistical_validation.py` — grade A · keep
- **Does:** Rigorous statistical validation of prediction claims: binomial_ci() computes exact Clopper-Pearson confidence intervals; binomial_test() tests vs base rate with significance; lift_with_significance() combines both; benjamini_hochberg() applies FDR correction for multiple comparisons; subgroup_analysis() breaks down accuracy by subgroup with proper statistics
- **Talks to:** IMPORTED BY ml/walk_forward.py, scripts/analysis/feedback_analyzer.py. IMPORTS scipy.stats (beta, binomtest). Functions: binomial_ci(), accuracy_with_ci(), binomial_test(), lift_with_significance(), benjamini_hochberg(), validate_multiple_claims(), subgroup_analysis(), get_reproducibility_metadata(), format_accuracy_claim(). Constants: MIN_SAMPLES_REPORT/RELIABLE/ROBUST, BASE_RATE_* per outcome.
- **Quality signals:** Sophisticated statistical toolkit: binomial_ci() uses beta quantiles (exact, not Normal approximation); binomial_test() proper two-sided test; BH FDR correction prevents false discovery; subgroup_analysis() enforces min_samples threshold (30 default) for reliability. format_accuracy_claim() outputs confidence intervals in readable format. get_reproducibility_metadata() logs git hash + package versions. ~400 lines, no dead functions.

#### 🟢 `ml/training.py` — grade A · keep
- **Does:** Main training orchestration: train_universal() and train_rich() train models on different feature sets; walk_forward_validate() runs cross-validation; train_optimized() applies hyperparameter tuning; _strip_meta() removes metadata columns for feeding to models
- **Talks to:** IMPORTED BY cli.py, scripts/models/retrain_no_odds_catboost.py, scripts/models/train_league.py, scripts/models/train_no_odds.py, scripts/models/train_unified.py, scripts/pipeline/weekly_retrain.py. IMPORTS ml/config.py, ml/correction_layer.py, ml/data.py, ml/ensemble.py, ml/evaluation.py, ml/feature_selection.py, ml/models.py, ml/persistence.py, ml/tuning.py, config/settings.py, storage/paths.py. Functions: _strip_meta(), walk_forward_validate(), train_universal(), train_rich(), train_optimized().
- **Quality signals:** High-level orchestration of entire training pipeline: train_universal()/train_rich() call walk_forward_validate(), ensemble.build_fold_models(), calibration.fit_calibrator(), correction_layer.train_static(), persistence.save_model(). walk_forward_validate() uses DataLoader.TimeSeriesSplitter for proper time-series CV. _strip_meta() removes season/_match_date/_league before feeding to models. Extensive logging. ~300 lines.

#### 🟢 `ml/tuning.py` — grade A · keep
- **Does:** Hyperparameter tuning via Optuna: tune_model() runs Bayesian optimization with walk-forward validation; _xgb_search_space(), _lgb_search_space(), _cb_search_space() define search spaces per library; _fit_with_early_stopping() wraps model training with early stopping; _walk_forward_score() computes validation log-loss
- **Talks to:** IMPORTED BY ml/calibration.py, ml/ensemble.py, ml/feature_selection.py, ml/ou_model.py, ml/training.py, scripts/models/optimize_unified.py, scripts/models/retrain_no_odds_catboost.py, scripts/models/train_league.py, scripts/models/train_no_odds.py. IMPORTS ml/config.py, ml/data.py. Functions: _compute_sample_weights(), _suggest_from_space(), _coupled_n_estimators(), _xgb_search_space(), _lgb_search_space(), _cb_search_space(), _build_model(), _fit_with_early_stopping(), _walk_forward_score(), tune_model(). Constant: SEARCH_SPACES.
- **Quality signals:** Sophisticated Optuna-based tuning: _xgb_search_space() defines ~10 xgboost hyperparameters (learning_rate, max_depth, subsample, etc.); similar for lightgbm/catboost. _coupled_n_estimators() scales n_estimators with learning_rate (slower LR = more trees). _compute_sample_weights() applies time decay for recent matches. tune_model() runs walk-forward CV for robust validation. ~350 lines.

#### 🟢 `ml/walk_forward.py` — grade A · keep
- **Does:** Walk-forward backtesting framework: walk_forward_split() partitions data into sequential train/test folds with optional purge for lookahead bias; evaluate_walk_forward() runs full WF evaluation; WalkForwardFold/WalkForwardResult dataclasses structure results; print_walk_forward_summary() formats output
- **Talks to:** IMPORTED BY scripts/analysis/backtest_unified.py. IMPORTS ml/evaluation.py, ml/statistical_validation.py, storage/paths.py. Classes: WalkForwardFold, WalkForwardResult. Functions: walk_forward_split(), walk_forward_matchweek_split(), evaluate_walk_forward(), print_walk_forward_summary(). Constant: DEFAULT_PURGE_MATCHWEEKS.
- **Quality signals:** Well-structured backtesting framework: walk_forward_split() creates sequential folds with optional purge (remove recent matches from training to prevent lookahead); walk_forward_matchweek_split() variant using matchweek boundaries. evaluate_walk_forward() runs full evaluation pipeline with per-fold metrics. WalkForwardResult aggregates metrics across folds. ~250 lines.

#### 🔧 `ml/walkforward_core.py` — grade A · keep
- **Does:** Walk-forward validation core: WalkForwardConfig configuration, FoldResult per-fold metrics, WalkForwardReport aggregation, _fit_one_fold() trains and evaluates a single fold, run_walkforward() orchestrates full walk-forward pipeline with isotonic calibration
- **Talks to:** IMPORTED BY scripts/diagnostics/paper_trade_draw_boost.py, scripts/diagnostics/validate_walkforward_core.py. IMPORTS no ml/ modules (self-contained). Classes: WalkForwardConfig, FoldResult, WalkForwardReport. Functions: detect_leakage(), fit_isotonic_calibrator(), apply_calibrator(), _ece_multiclass(), _brier_multiclass(), _logloss(), _fit_one_fold(), run_walkforward().
- **Quality signals:** Comprehensive walk-forward implementation: WalkForwardConfig captures all hyperparameters (min_train_seasons, min_test_samples, early_stopping_rounds). detect_leakage() flags if test set contains training data. fit_isotonic_calibrator() uses sklearn.isotonic.IsotonicRegression for per-fold calibration. _fit_one_fold() trains model, calibrates, evaluates on fold. run_walkforward() orchestrates full pipeline with proper cross-fold aggregation. ~400 lines.

### `models/` — 20 files

#### ⚫ `models/__init__.py` — grade F · **delete**
- **Does:** Empty package initializer file.
- **Talks to:** No imports or exports.
- **Quality signals:** Empty file (1 line), serves no purpose — package is not imported anywhere.
- **Verdict reason:** Empty __init__.py files with no exports are typically removed in modern Python packaging. Clients import from submodules directly (models.deep_learning, models.schemas, models.simulator.*).

#### 🟢 `models/deep_learning.py` — grade C · edit
- **Does:** Phase 5 neural network models: LSTM form predictor, Transformer attention model, Squad interaction model (GNN-inspired), DeepEnsemble meta-learner, MatchSequenceDataset, DeepModelTrainer, DeepPredictor. Conditional on torch availability.
- **Talks to:** imported by: scripts/analysis/calibration_analysis.py, scripts/models/train_unified.py, scripts/prediction/ensemble_prediction_engine.py. imports: config/settings.py.
- **Quality signals:** 1021 LOC, three neural network classes + trainer + predictor. Uses __future__ annotations, proper type hints (Dict, List, Optional, Tuple). 15+ errors from Pyright: missing torch/numpy imports (env issue), conditional nn/torch/F reference errors due to try-except TORCH_AVAILABLE guard (defensive but creates static analysis noise). One type error: MatchSequenceDataset.__init__ line 380 assigns None to List[str] parameter. DeepModelTrainer.__init__ line 526-527 assigns None to Path/str parameters. Core logic appears sound but type safety degraded by TORCH_AVAILABLE guard and optional parameter defaults.
- **Verdict reason:** Add type guards or use TYPE_CHECKING to suppress false Pyright errors from conditional torch imports. Fix type errors: pass non-None values to Path/str parameters in DeepModelTrainer.__init__ and MatchSequenceDataset.__init__. Consider __all__ export list.

#### 🟢 `models/schemas.py` — grade A · keep
- **Does:** Dataclass definitions for HTML-parsed match data: MatchMetadata, TeamStats, LineupInfo, MatchEvent, MatchData. Used by parser layer to hand off structured data.
- **Talks to:** imported by: parser/events.py, parser/lineups.py, parser/match_page.py, parser/scorebox.py, storage/structured.py. No internal imports.
- **Quality signals:** Well-documented dataclasses (docstrings on each), uses __future__ annotations, from __future__ import, proper type hints (Optional, list[dict]). ~75 LOC, concise schema definition. No dead code or import errors. Stable and reusable across parser/storage.

#### ⚫ `models/simulator/__init__.py` — grade D · **delete**
- **Does:** Package initializer documenting the unified match simulator plan (references unified-simulator-plan-v3.md).
- **Talks to:** No imports or exports.
- **Quality signals:** 5 LOC docstring only. Documents intent but is not imported anywhere.
- **Verdict reason:** Empty package initializer. Documentation should live in a README or the referenced plan file, not in __init__.py.

#### ⚫ `models/simulator/backtests/__init__.py` — grade D · **delete**
- **Does:** Package initializer describing backtest harness interface: walk-forward by season, resolve odds, apply stake policies, emit per-market/threshold/source/policy reports with bootstrap CIs.
- **Talks to:** No imports or exports.
- **Quality signals:** 17 LOC docstring only. Helpful documentation of the backtest architecture but not imported anywhere.
- **Verdict reason:** Empty package initializer. The documented interface is defined in harness.py, not here. Move docstring to harness.py module docstring or a README.

#### 🧪 `models/simulator/backtests/harness.py` — grade B · keep
- **Does:** BacktestHarness orchestrates historical P&L backtests: walk-forwards by season, applies predictor, resolves odds via fallback chain, applies stake policies, returns BacktestReport with per-market/threshold/source ROI stats and bootstrap CIs.
- **Talks to:** imported by: tests/simulator/test_backtest_harness.py. imports: (none detected in fact card, but reads .report and .roi_bootstrap). Defines: BinaryMarket, MulticlassMarket, Predictor protocol, BacktestHarness class with run() method.
- **Quality signals:** 422 LOC, well-structured. Uses __future__ annotations, type hints (pd.DataFrame, np.ndarray). Two Pyright errors: missing numpy/pandas imports (env issue, not code quality). Docstrings on major methods. Orchestrates BinaryMarket, MulticlassMarket, odds resolution, stake policies. Test-only usage suggests live functionality is tested but not yet called from scripts/.

#### 🧪 `models/simulator/backtests/odds_fallback.py` — grade A · keep
- **Does:** Odds resolution fallback chain: tries Pinnacle closing -> market avg -> B365 -> opening. Tags each bet with source label for per-source ROI slicing to detect selection bias. Raises NoOddsAvailable if all sources missing.
- **Talks to:** imported by: tests/simulator/test_backtest_harness.py. Defines: NoOddsAvailable exception, BinaryOdds, MulticlassOdds dataclasses, resolve_odds_binary(), resolve_odds_multiclass(), deoverround_binary(), deoverround_multiclass().
- **Quality signals:** 94 LOC, well-documented (clear fallback strategy in docstring). Frozen dataclasses, proper type hints. Clean separation of concerns: BinaryOdds vs MulticlassOdds, source labeling. No import errors. Test coverage present.

#### 🧪 `models/simulator/backtests/report.py` — grade A · keep
- **Does:** BacktestReport and BacktestRunMetadata dataclasses: canonical nested JSON-serializable schema for all backtest runs. Structure: report.markets[market][threshold][stake_policy][odds_source] -> ROIStats. Methods: to_json_dict(), write(path), summary_rows().
- **Talks to:** imported by: models/simulator/backtests/harness.py (in BacktestHarness.run()). imports: roi_bootstrap.ROIStats. Defines: BacktestRunMetadata, BacktestReport.
- **Quality signals:** 84 LOC. Well-documented nested schema. Proper type hints (dict[str, dict[...]]). Dataclasses with @dataclass decorator. Methods have type hints. No import errors. Stable, JSON-serializable design allows comparison across predictors/models.

#### 🧪 `models/simulator/backtests/roi_bootstrap.py` — grade B · keep
- **Does:** ROIStats class and compute_roi_stats() function: calculates ROI, Sharpe ratio, max drawdown, bootstrap confidence intervals, and longest losing streak from a sequence of bet records.
- **Talks to:** imported by: models/simulator/backtests/report.py (ROIStats field type), tests/simulator/test_backtest_harness.py. Defines: ROIStats class, compute_roi_stats() function.
- **Quality signals:** 111 LOC. Proper type hints. Computes multiple risk metrics (Sharpe, max DD, losing streak). Bootstrap CI logic appears robust. Test coverage present. No import errors detected.

#### 🧪 `models/simulator/backtests/stake_policies.py` — grade B · keep
- **Does:** Betting stake policy models: BetInput, StakePolicy protocol, FlatStake, KellyStake, NoStake implementations. Defines stake allocation for different risk strategies.
- **Talks to:** imported by: tests/simulator/test_backtest_harness.py. Defines: BetInput, StakePolicy, FlatStake, KellyStake, NoStake, DEFAULT_POLICIES constant.
- **Quality signals:** 86 LOC. Well-structured policy pattern. Type hints present. Dataclass-based. No import errors.

#### ⚫ `models/simulator/base_rates/__init__.py` — grade D · **delete**
- **Does:** Package initializer documenting rate estimator interface: fit() and predict() methods consumed by engine/simulator.py.
- **Talks to:** No imports or exports.
- **Quality signals:** 12 LOC docstring only. Documents estimator protocol but not imported anywhere.
- **Verdict reason:** Empty package initializer. Documentation belongs in README or estimator class docstrings, not here.

#### 🟢 `models/simulator/base_rates/card_rates.py` — grade B · keep
- **Does:** CardRateEstimator: fits Poisson regression model on features_serie_a.parquet to predict per-team yellow/red card rates. Used by simulator to populate card event counts.
- **Talks to:** imported by: scripts/prediction/simulate_upcoming.py, tests/simulator/test_phase2_rates.py. Defines: CardRateEstimator class, DEFAULT_CARD_FEATURES, REF_STRICTNESS_SCALE constants.
- **Quality signals:** 120 LOC. Proper type hints. Logging present. No import errors. Implements fit() and predict() estimator interface.

#### 🟢 `models/simulator/base_rates/corner_rates.py` — grade B · keep
- **Does:** CornerRateEstimator: fits Poisson regression model on features_serie_a.parquet to predict per-team corner rates. Used by simulator to populate corner event counts.
- **Talks to:** imported by: scripts/prediction/simulate_upcoming.py, tests/simulator/test_phase2_rates.py. Defines: CornerRateEstimator class, DEFAULT_CORNER_FEATURES.
- **Quality signals:** 96 LOC. Type hints present. Logging. Implements fit() and predict() estimator interface. No import errors.

#### 🟢 `models/simulator/base_rates/lineup_allocator.py` — grade B · keep
- **Does:** Allocates simulated shots to individual players based on historical lineup and role profiles. Functions: allocate_team_shots_to_players(), resolve_lineup_from_history(), _top11_from_profiles(). Reads from LINEUPS_PATH parquet file.
- **Talks to:** imported by: scripts/prediction/simulate_upcoming.py, tests/simulator/test_phase5_player_props.py. imports: models/simulator/base_rates/player_profiles.py. Defines: allocate_team_shots_to_players(), resolve_lineup_from_history(), _top11_from_profiles().
- **Quality signals:** 126 LOC. Type hints present. Imports player_profiles module. Logging. Reads PROJECT_ROOT/LINEUPS_PATH. No import errors.

#### 🟢 `models/simulator/base_rates/player_profiles.py` — grade B · keep
- **Does:** PlayerProfile and PlayerProfileStore classes: loads player match stats from parquet, maintains positional priors (for imputation), handles per-player statistics (shots, tackles, passes, etc). Functions: load_player_match_stats(), _positional_prior().
- **Talks to:** imported by: models/simulator/base_rates/lineup_allocator.py, scripts/prediction/simulate_upcoming.py, tests/simulator/test_phase5_player_props.py. Defines: PlayerProfile, PlayerProfileStore, load_player_match_stats(), _positional_prior(), POSITIONAL_PRIORS constant.
- **Quality signals:** 212 LOC. Type hints present. Two classes for storing player stats and profiles. Logging. Reads PLAYER_STATS_PATH. No import errors.

#### 🟢 `models/simulator/base_rates/shot_generator.py` — grade B · keep
- **Does:** ShotRateEstimator: fits Poisson regression on features_serie_a.parquet to predict per-team shot and shot-on-target rates.
- **Talks to:** imported by: scripts/prediction/simulate_upcoming.py, tests/simulator/test_phase2_rates.py. Defines: ShotRateEstimator class, DEFAULT_SHOT_FEATURES.
- **Quality signals:** 98 LOC. Type hints. Implements fit() and predict() estimator interface. No import errors.

#### ⚫ `models/simulator/engine/__init__.py` — grade D · **delete**
- **Does:** Package initializer documenting the simulator core engine (Phase 1): MatchSimulation data structure and simulate_match() public entry point, with Dixon-Coles τ correction in dixon_coles.py.
- **Talks to:** No imports or exports.
- **Quality signals:** 6 LOC docstring only. Not imported anywhere.
- **Verdict reason:** Empty package initializer. Documentation should be in simulator.py module docstring.

#### 🟢 `models/simulator/engine/dixon_coles.py` — grade B · keep
- **Does:** Dixon-Coles goal model implementation: dc_correction(), dc_joint_pmf(), sample_dc_goals(), fit_tau_mle(). Implements τ parameter correction for Poisson goal sampling to account for low-scoring match bias.
- **Talks to:** imported by: scripts/prediction/simulate_upcoming.py, tests/simulator/test_simulator_engine.py. Defines: dc_correction(), dc_joint_pmf(), sample_dc_goals(), fit_tau_mle().
- **Quality signals:** 175 LOC. Type hints. Well-documented functions with docstrings. Implements statistical correction model. No import errors.

#### 🟢 `models/simulator/engine/simulator.py` — grade A · keep
- **Does:** MatchSimulation core data structure and simulate_match() entry point. Simulates one match: samples home/away goals via Poisson/Dixon-Coles, generates shots/cards/corners/own goals from rate estimates, returns structured MatchSimulation with play-by-play events.
- **Talks to:** imported by: models/simulator/markets.py, scripts/prediction/simulate_upcoming.py, tests/simulator/test_phase2_rates.py, test_phase5_player_props.py, test_simulator_engine.py. Defines: MatchSimulation class, _match_seed(), simulate_match().
- **Quality signals:** 359 LOC. Well-structured MatchSimulation dataclass. Type hints throughout. Proper docstrings. Core entry point simulate_match() is reused across tests and scripts. No import errors.

#### 🟢 `models/simulator/markets.py` — grade B · keep
- **Does:** Market probability aggregation: all_player_market_probs(), all_phase2_market_probs(), all_goal_market_probs(). Converts MatchSimulation output into market-ready probability estimates (H/D/A, over/under goals, player props).
- **Talks to:** imported by: scripts/prediction/simulate_upcoming.py, tests/simulator/test_phase2_rates.py, test_phase5_player_props.py. imports: models/simulator/engine/simulator.py. Defines: all_player_market_probs(), all_phase2_market_probs(), all_goal_market_probs().
- **Quality signals:** 93 LOC. Type hints. Imports simulator.MatchSimulation. Clean separation of simulation output from market pricing. No import errors.

### `monitoring/` — 7 files

#### 🟢 `monitoring/__init__.py` — grade A · keep
- **Does:** Package initialization for the monitoring module; re-exports key classes (ModelRetrainer, ProbabilityCalibrator, FeatureDriftDetector, ABTestFramework, PerformanceDashboard, etc.) for convenient public API access.
- **Talks to:** Imports from monitoring/retraining.py (ModelRetrainer, WalkForwardValidator), monitoring/calibration.py (ProbabilityCalibrator, CalibrationTracker), monitoring/drift_detection.py (FeatureDriftDetector, AccuracyMonitor), monitoring/ab_testing.py (ABTestFramework, ModelComparator), monitoring/dashboard.py (PerformanceDashboard, MetricsCollector). No code imports this file (imported_by is empty).
- **Quality signals:** Clean, minimal __init__ file (30 lines); properly structured with docstring explaining module phases; explicit __all__ list; imports only what is needed. Zero logic, pure re-export.

#### 🟢 `monitoring/ab_testing.py` — grade A · keep
- **Does:** A/B testing framework for comparing two production models side-by-side; includes ModelComparator (McNemar's test, log loss, Brier score comparison), ABExperiment (records paired predictions), ABTestFramework (manages multiple experiments), and AutoPromoter (auto-promotes winning models based on statistical significance).
- **Talks to:** Imports from config/settings.py (DATA_DIR, MODELS_DIR), monitoring/utils.py (load_json_history, save_json_history). Imported by scripts/pipeline/run_monitoring.py which orchestrates the monitoring pipeline.
- **Quality signals:** Well-structured, ~546 lines with clear class separation. Type hints throughout (Dict, List, Optional, np.ndarray). Detailed docstrings for each method. Graceful scipy fallback (lines 21-25, approximations for stats when scipy unavailable). Proper state management with JSON history persistence. Handles edge cases (empty windows, single model predictions, division by zero).

#### 🟢 `monitoring/calibration.py` — grade A · keep
- **Does:** Probability calibration system providing Platt scaling and isotonic regression methods; includes ProbabilityCalibrator (selects best method automatically), CalibrationTracker (tracks calibration metrics over time), and DynamicThresholdAdjuster (adjusts confidence thresholds based on calibration drift).
- **Talks to:** Imports from config/settings.py (DATA_DIR, MODELS_DIR), monitoring/utils.py (load_json_history, save_json_history). Imported by scripts/pipeline/run_monitoring.py which calls calibration operations during monitoring cycles.
- **Quality signals:** ~427 lines with solid engineering. Type hints throughout. Graceful sklearn fallback (lines 21-27). Well-documented ECE (Expected Calibration Error) calculation and reliability bucketing. Clear separation of concerns (PlattScaler vs IsotonicCalibrator). Historical tracking with JSON persistence. Handles edge cases (division by zero on line 146-147, empty history fallbacks).

#### 🟢 `monitoring/dashboard.py` — grade A · keep
- **Does:** Real-time performance monitoring dashboard; MetricsCollector stores predictions and betting results with daily aggregation; PerformanceDashboard generates accuracy summaries, P&L metrics, daily trends, model comparisons, and alerts; AlertManager dispatches alerts with handlers.
- **Talks to:** Imports from config/settings.py (DATA_DIR, MODELS_DIR). Imported by scripts/pipeline/run_monitoring.py which feeds predictions and bets to the dashboard for real-time monitoring.
- **Quality signals:** ~544 lines, well-organized three-class design. Type hints throughout. Comprehensive metrics (accuracy by outcome, by confidence, by edge bucket). Robust alert logic (lines 357-401) with severity thresholds. Daily stats aggregation with periodic saves (line 119-120). Handles missing data gracefully (lines 206-207, 260-261). Human-readable print_summary method.

#### 🟢 `monitoring/drift_detection.py` — grade A · keep
- **Does:** Data and concept drift detection system; FeatureDriftDetector monitors feature distributions (mean shift, variance change, IQR/range shifts); AccuracyMonitor tracks accuracy degradation over rolling windows; ConceptDriftDetector detects shifts in prediction error distribution using t-tests.
- **Talks to:** Imports from config/settings.py (DATA_DIR), monitoring/utils.py (load_json_history, save_json_history). Imported by scripts/pipeline/run_monitoring.py which runs drift checks as part of continuous monitoring.
- **Quality signals:** ~532 lines with strong statistical foundation. Type hints throughout. Multiple drift test implementations per feature (mean_shift z-test line 183-193, variance_ratio line 196-206, IQR ratio line 213-221, range_shift line 224-236). Graceful scipy fallback for stats tests. Clear confidence bucketing (lines 394-417). Historical tracking with summary aggregation (lines 240-260). Well-documented docstrings.

#### 🟢 `monitoring/retraining.py` — grade A · keep
- **Does:** Automated model retraining orchestration; WalkForwardValidator implements time-series cross-validation avoiding look-ahead bias; ModelRetrainer monitors triggers (accuracy threshold, new data, time interval) and manages model versioning; RetrainingScheduler coordinates full validation before retrain.
- **Talks to:** Imports from config/settings.py (DATA_DIR, MODELS_DIR). Imported by scripts/pipeline/run_monitoring.py which calls retraining operations when conditions are met.
- **Quality signals:** ~441 lines, well-architected. Type hints throughout. Walk-forward validation properly avoids look-ahead bias (lines 56-81). Version management with increment logic (lines 234-241). Multi-criteria trigger logic (accuracy, new_data, interval) on lines 243-276. State persistence with JSON (lines 210-226). Helper factory functions (lines 425-440). Handles edge cases (dataframe date conversion lines 145-146).

#### 🟢 `monitoring/utils.py` — grade A · keep
- **Does:** Shared utility functions for monitoring modules: load_json_history (safe file loading with default fallback), save_json_history (JSON persistence with optional max_entries truncation).
- **Talks to:** Imported by monitoring/ab_testing.py (load_json_history, save_json_history lines 19), monitoring/calibration.py (load_json_history, save_json_history lines 19), monitoring/drift_detection.py (load_json_history, save_json_history lines 19). Does not import any monitoring modules (imports only json and pathlib).
- **Quality signals:** Very concise (36 lines), utility-focused. Proper type hints (Path, List[Dict], Optional[int]). Docstrings for both functions. Defensive programming: graceful fallback when file missing (line 16) and optional truncation (line 33). Zero duplication across callers.

### `web/` — 5 files

#### ⚫ `web/__init__.py` — grade A · keep
- **Does:** Package marker for the web module with minimal content.
- **Talks to:** None — no imports or exports.
- **Quality signals:** Single-line module docstring. Empty init file appropriate for package structure.

#### 🟢 `web/advisor.py` — grade B · keep
- **Does:** Flask blueprint implementing Claude-powered conversational betting advisor with 8 data tools, SSE streaming, prompt caching, and cost tracking.
- **Talks to:** config/settings.py (imports DATA_DIR, UPCOMING_DIR, BETTING_DIR, BANKROLL_DIR, LIVE_DIR); config/team_names.py (normalize_team, team constants); scripts/betting/bet_journal.py (betting data); scripts/utils/json_utils.py (load_json_safe). Imported by: scripts/pipeline/telegram_bot.py and web/app.py.
- **Quality signals:** 3551 lines, 27 functions with return type hints, 23 docstrings. Tool handlers well-documented. System prompt generation and Claude API integration with token tracking. Minor: some functions lack docstrings.

#### 🟢 `web/app.py` — grade A · keep
- **Does:** Main Flask web application serving a multi-league betting dashboard with live matches, predictions, standings, analytics, team/player details, and automatic pipeline orchestration.
- **Talks to:** config/leagues.py (LEAGUE_REGISTRY, ACTIVE_LEAGUES); config/settings.py; config/team_names.py; scraper/lineup_fetcher.py; scripts/analysis/player_history.py; scripts/betting/* modules (bet_journal, benchmark_tracker, entry_timer, fair_odds_tracker, odds_edge_monitor, prop_tracker); scripts/pipeline/* (notify, pipeline_state, run_full_pipeline, scheduler); scripts/prediction/lineup_predictor.py; scripts/utils/parsing.py; web/advisor.py. Imported by: tests/test_automation.py, tests/test_pipeline.py.
- **Quality signals:** 8968 lines, 46 functions with return type hints, 136 docstrings. Comprehensive multi-league support. Heavy caching (JSON, Parquet, HTML). Thread-safe background tasks. Well-documented helpers and API endpoints. Strict typing throughout.

#### ⚫ `web/predictor.py` — grade B · **delete**
- **Does:** Standalone match prediction class (MatchPredictor) that loads CatBoost models and generates predictions with feature importance explanations; utility functions to load match features and upcoming matches.
- **Talks to:** None — no project imports. Standalone utility using numpy, pandas, catboost.
- **Quality signals:** 295 lines, 6 functions with return type hints, 10 docstrings. Well-structured class with proper initialization, error handling, and human-readable feature explanations. However, file is NEVER imported by any other module despite implementing core functionality.
- **Verdict reason:** True garbage. Functionality is superseded by pipeline model training and pre-built predictions.json files. app.py loads predictions from JSON, not via MatchPredictor. No wiring exists to call this class at runtime. Code quality is good but serves no purpose in the live system.

#### 🧪 `web/test_advisor_50.py` — grade A · keep
- **Does:** Comprehensive 50-message stress test suite validating AI Advisor tool use, error handling, memory, accuracy against SofaScore player ratings, and edge cases.
- **Talks to:** None — external test script. Makes HTTP POST requests to /api/chat endpoint. Pre-loads SofaScore reference data via pandas.read_parquet.
- **Quality signals:** 490 lines, 1 main function with type hints, 7 docstrings. Well-organized 50 test cases grouped by feature. Composable validator helpers. Ground-truth validation via SofaScore data. Detailed error reporting. Proper rate limiting.

### `scripts/pipeline/` — 14 files

#### ⚫ `scripts/pipeline/__init__.py` — grade C · keep
- **Does:** Package marker with module documentation defining the pipeline orchestration, scheduling, state management, and monitoring scope.
- **Talks to:** No imports or exporters.
- **Quality signals:** Empty module package — only docstring present. Serves as documentation of module scope but no code.

#### 🔧 `scripts/pipeline/_refresh_fbref_fixtures.py` — grade B · keep
- **Does:** One-shot scraper that fetches and saves FBref Serie A fixture schedule HTML using Selenium/botasaurus for use by FBref HTML parsers.
- **Talks to:** No imports; standalone script with __main__ entry point. Called via subprocess from refresh_weekly_data.py (fact card shows 0 importers).
- **Quality signals:** ~75 LOC, focused single-purpose scraper. Has error handling (3 retries, JS overlay dismissal), basic logging, size validation. No type hints. Not imported by other modules — intended as standalone CLI/subprocess tool per docstring ('Called by refresh_weekly_data.py').

#### 🟢 `scripts/pipeline/health_check.py` — grade A · keep
- **Does:** Production health check system that validates system-wide integrity in seconds: data freshness, model staleness, betting performance, and missing dependencies.
- **Talks to:** Imported by: monitor.py (run_health_check), scheduler.py (indirectly via monitor). Imports: config/settings.py. Internal checks reference: DATA_DIR, MODELS_DIR paths and betting health from auto_settle data.
- **Quality signals:** Well-structured with 13+ validation functions, clear exit codes (0/1/2), supports --json CLI output, designed for CI integration. Has docstring and helpful file-age formatting. Thresholds are configurable constants (MAX_FEATURES_AGE_DAYS, etc). No external API calls—fast by design.

#### 🟢 `scripts/pipeline/learning_loop.py` — grade B · keep
- **Does:** Post-matchday parallel analysis orchestrator: runs 3 concurrent agents (Prediction Auditor, Drift Detector, ROI Analyzer) that audit predictions, detect feature drift, and compute betting ROI.
- **Talks to:** Imported by: run_full_pipeline.py. Imports: config/settings.py, scripts/analysis/feedback_analyzer.py, scripts/pipeline/lesson_writer.py. Writes to data/feedback/ (JSON feedback files).
- **Quality signals:** Uses ThreadPoolExecutor for parallel analysis; handles numpy serialization with custom JSONEncoder. Has _NumpySafeEncoder class, logging, file-based agent communication pattern. ~100+ LOC with 4 public functions. Lacks type hints in function signatures.

#### 🟢 `scripts/pipeline/lesson_writer.py` — grade B · keep
- **Does:** Converts post-match analysis feedback (from learning_loop.py) into persistent lesson records: xG bias corrections, confidence recalibration, market penalties with auto-expiry and effectiveness tracking.
- **Talks to:** Imported by: learning_loop.py. Imports: config/settings.py. Reads/writes data/feedback/lessons.json.
- **Quality signals:** Well-scoped with 8 functions defining lesson generation and effectiveness tracking. Clear safety constants (MAX_XG_ADJUST, MAX_CONFIDENCE_SHIFT, DEFAULT_EXPIRY). Has basic logging. ~250 LOC. Lacks type hints. Good separation of concerns with _load_lessons, _save_lessons helpers.

#### 🟢 `scripts/pipeline/monitor.py` — grade A · keep
- **Does:** Automated health monitor runs every 30 min via launchd: extends health_check with API key validation, pipeline staleness, pending bet detection, and writes status to disk + macOS notifications.
- **Talks to:** Not imported by others (runs as daemon); imports: config/settings.py, scripts/pipeline/health_check.py (run_health_check), scripts/utils/json_utils.py. Calls health_check and validates API keys (Odds, Gemini, Groq, Perplexity).
- **Quality signals:** ~350 LOC with 12+ validation functions. Configurable thresholds (MAX_PIPELINE_RUN_AGE_HOURS, etc). Proper logging to monitor.log, JSON status output. Has API key health checks with environ/dotenv loading. Exit codes (0/1/2). Designed for launchd integration (daemon entry point).

#### 🟢 `scripts/pipeline/notify.py` — grade A · keep
- **Does:** Unified notification system (macOS + Telegram) with category routing, priority tiers, quiet hours, and batching. Provides 40+ specific notification functions (notify_value_bets, notify_settlement, notify_goal, etc).
- **Talks to:** Imported by: 11 files across betting, pipeline, and web modules (highest connectivity). Imports: config/leagues.py, config/team_names.py, scripts/betting/__init__.py. Core infrastructure for all user notifications.
- **Quality signals:** Massive module (~1400 LOC) with deep feature set: Telegram rate limiting, notification batching, history tracking, quiet hours, priority tiers (URGENT/NORMAL/LOW), league-aware formatting with emojis/badges. Custom TgMsg class and _NotificationBatcher. Well-documented API. Central hub for all system messaging.

#### 🟢 `scripts/pipeline/pipeline_state.py` — grade B · keep
- **Does:** Persistent state tracker for incremental pipeline runs: stores predicted_matches, settled_matches, last_run timestamps to avoid reprocessing and enable efficient resumption.
- **Talks to:** Imported by: health_check.py, run_full_pipeline.py, web/app.py. Imports: config/settings.py. Reads/writes data/pipeline_state.json (state file).
- **Quality signals:** ~130 LOC with 9 functions managing state lifecycle (load, save, mark_predicted, mark_settled, get_new_results, etc). Clean API with safe defaults. Uses typing hints for return types. Good docstring. No external API calls.

#### 🟢 `scripts/pipeline/refresh_weekly_data.py` — grade B · keep
- **Does:** Weekly data refresh orchestrator (runs Mondays at 04:00): refreshes FBref HTMLs, parses stats, updates Sofascore/Understat scrapes, referee assignments, and weather backfill.
- **Talks to:** Not imported by modules (runs as scheduled task); imports: config/leagues.py, scraper/{referee,understat_scraper,weather}.py, scripts/pipeline/notify.py. Orchestrates step_refresh_fbref_fixtures via subprocess.
- **Quality signals:** ~80 LOC with clear step-based orchestration pattern. Each step wrapped in try/except for resilience (partial failures allowed). Uses subprocess.run with timeouts (3600s). Basic logging of success/failure. Has docstring explaining all 6 steps. No type hints.

#### 🟢 `scripts/pipeline/run_full_pipeline.py` — grade A · keep
- **Does:** Complete 32-step betting intelligence pipeline: orchestrates odds fetching, squad/lineup updates, ensemble predictions, model ensembles (over/under, handicap, cards, BTTS), betting engine, parlay generation, and result settlement.
- **Talks to:** Imported by: scheduler.py, web/app.py. Imports: ~40 modules across config, scraper, features, scripts/models, scripts/analysis, scripts/prediction, scripts/betting. Core hub importing nearly all pipeline components.
- **Quality signals:** ~1500 LOC orchestrating massive pipeline. Has nested functions (_run_parallel_data_collection with 5 parallel groups), step timing tracking, file locking, incremental mode (--quick, --bankroll, --snapshot-only, --pre-kickoff, --live-*), CLI argument parsing. Well-documented 32-step process. Good error handling patterns.

#### 🟢 `scripts/pipeline/run_monitoring.py` — grade B · keep
- **Does:** Monitoring system main entry point: orchestrates continuous learning (drift detection, retraining, calibration, A/B testing, dashboards) via monitoring module classes.
- **Talks to:** Imported by: scheduler.py. Imports: monitoring/{retraining,calibration,drift_detection,ab_testing,dashboard}.py. CLI interface to monitoring subsystem.
- **Quality signals:** ~50 LOC with MonitoringSystem class (incomplete in visible code). Imports show integration with retraining, calibration, drift detection, A/B testing. Has argparse for --check, --dashboard, --retrain, --drift, --full modes. Structured logging. Thin wrapper over monitoring module.

#### 🟢 `scripts/pipeline/scheduler.py` — grade A · keep
- **Does:** Automated betting pipeline scheduler with match-day detection, adaptive scheduling, and retry logic. Runs pipeline at fixed times daily + match-day-adaptive times with APScheduler or simple cron.
- **Talks to:** Imported by: tests/test_automation.py, web/app.py. Imports: 15+ modules including config, scripts/models, scripts/analysis, scripts/betting, storage/paths, scripts/pipeline/{health_check,notify,run_full_pipeline,run_monitoring}. Central orchestrator.
- **Quality signals:** ~800 LOC with sophisticated scheduling: SCHEDULE_CONFIG dict (daily + match-day runs), get_upcoming_matches, match-day detection, pre-kickoff state tracking, log rotation, retry logic (MAX_RETRIES), launchd integration (rotate_launchd_logs). run_daemon function supports --once and daemon modes. Well-designed state management (PRE_KICKOFF_STATE_FILE).

#### 🟢 `scripts/pipeline/telegram_bot.py` — grade A · keep
- **Does:** Two-way Telegram chat bot: uses Claude API with tool-use to provide interactive betting coach with commands (/bets, /bankroll, /live, /parlays, /help) and photo-based player analysis.
- **Talks to:** Not imported by modules (runs as daemon); imports: config/leagues.py, config/settings.py, scripts/analysis/player_history.py, scripts/betting/{benchmark_tracker,bet_journal}.py, scripts/pipeline/notify.py, scripts/utils/json_utils.py, web/advisor.py. Standalone Telegram interface.
- **Quality signals:** Large, feature-rich module (~1000 LOC) with: rate limiting (_RateLimiter), conversation persistence (ConversationManager), typing indicators (_TypingKeepAlive), photo analysis, Claude API integration with tool-use, command routing (_handle_*), keyboard/inline buttons, HTML markdown conversion. Comprehensive error handling and signal handling.

#### 🟢 `scripts/pipeline/weekly_retrain.py` — grade A · keep
- **Does:** Automated model retraining triggered by matchweek completion: auto-detects quick vs full Optuna retrain mode, archives old models, compares performance, and promotes new models on validation.
- **Talks to:** Imported by: cli.py. Imports: config/leagues.py, config/settings.py, features/build.py, ml/{data,ensemble,training}.py, scripts/models/train_over_under.py, scripts/pipeline/notify.py, storage/paths.py. Entry point for model lifecycle management.
- **Quality signals:** ~400 LOC with sophisticated retraining workflow: matchweek detection (MATCHES_PER_MATCHWEEK=10), quick (~15-20 min) vs full (~2-3 hours) modes, walk-forward CV validation, model archiving, performance comparison, dry-run support, metrics history tracking (METRICS_HISTORY JSONL), safety checks. Good documentation of modes and safety.

### `scripts/models/` — 33 files

#### 🟢 `scripts/models/__init__.py` — grade A · keep
- **Does:** Module initialization that provides a shared load_predictions() helper to load match predictions from JSON files in data/upcoming.
- **Talks to:** Imported by: btts_corners_model, cards_model, handicap_model, over_under_model (all live models). Imports: config/settings (DATA_DIR).
- **Quality signals:** Minimal but correct: 29 lines, well-documented, no type hints but clear intent. Single reusable function that all prediction models depend on.

#### ⚫ `scripts/models/breakthrough.py` — grade F · **delete**
- **Does:** Experimental script exploring breakthrough predictions using Poisson distribution and comprehensive_markets module.
- **Talks to:** Imported by: (none). Imports: config.settings, ml.poisson, scripts.models.comprehensive_markets.
- **Quality signals:** Unknown (not read, marked dead_candidate with no importers). Imports comprehensive_markets which is one_shot experimental. Zero importers suggest abandoned.
- **Verdict reason:** True garbage: zero importers, imports only experimental one_shot modules, no clear purpose statement in fact card, marked dead_candidate.

#### 🟢 `scripts/models/btts_corners_model.py` — grade A · keep
- **Does:** Predicts Both Teams To Score (BTTS) and corner totals for upcoming matches using team scoring rates, clean sheet rates, and historical corner data with Poisson distribution over/under calculations.
- **Talks to:** Imported by: run_full_pipeline.py (live). Imports: config.settings, ml.poisson, scripts.models (load_predictions), config.leagues, features.lineup_stats.
- **Quality signals:** 1017 lines, comprehensive implementation with 6 dataclasses, extensive factor analysis (13+ betting factors), lineup awareness via features.lineup_stats integration, team-specific historical rates for BTTS/corners, real odds sanity checks (1.10-8.0/1.10-10.0 ranges), module-level caching for team_rates.json, logging throughout, generates 3 output JSON files (btts_predictions.json, corners_predictions.json, btts_corners_bets.json).

#### ⚫ `scripts/models/calibrated_ensemble.py` — grade F · **delete**
- **Does:** Experimental calibrated ensemble approach using Poisson and comprehensive_markets.
- **Talks to:** Imported by: (none). Imports: config.settings, ml.poisson, scripts.models.comprehensive_markets.
- **Quality signals:** Zero importers, imports only experimental modules, marked dead_candidate.
- **Verdict reason:** True garbage: zero importers, abandoned experiment, no integration with pipeline.

#### 🟢 `scripts/models/cards_model.py` — grade A · keep
- **Does:** Predicts card totals and booking points for matches using referee strictness ratings, team discipline factors, derby detection, and weather/intensity factors with Poisson-based over/under probabilities.
- **Talks to:** Imported by: run_full_pipeline.py (live). Imports: config.settings, ml.poisson, scripts.models (load_predictions), config.leagues, features.lineup_stats.
- **Quality signals:** 712 lines, 2 dataclasses, 13 referee strictness ratings (validated), team discipline multipliers, derby/rivalry detection, lineup-aware card rates via features.lineup_stats, foul-rate adjustments, Poisson distribution for card lines (2.5-6.5), saves predictions + bets to cards_predictions.json and cards_bets.json, includes --validate CLI flag with test cases.

#### 🔧 `scripts/models/comprehensive_markets.py` — grade B · keep
- **Does:** Trains unified ML models for 15 betting markets (1X2, O/U, BTTS, half-time, corners, cards, exact score, etc.) using all available data sources (features, football-data CSVs, player stats, match incidents, referee assignments).
- **Talks to:** Imported by: 10 experimental scripts (breakthrough, calibrated_ensemble, draw_aware_ensemble, draw_oracle, ml_market_predictions, optimize_unified, push_accuracy, stacking_meta, tune_fast, unified_model). All non-live. Imports: config.settings, features.build, ml.feature_selection, storage.paths.
- **Quality signals:** 150+ lines shown. Comprehensive data loading (features.parquet, football-data CSVs, referee_assignments, player_match_stats). Multi-market training with 15 distinct target definitions. Handles missing data merging on date+team pairs. Missing: implementation of actual training loops and model saving, only shows data loading scaffolding.

#### ⚫ `scripts/models/draw_aware_ensemble.py` — grade F · **delete**
- **Does:** Experimental draw-aware ensemble prediction model.
- **Talks to:** Imported by: (none). Imports: config.settings, ml.poisson, scripts.models.comprehensive_markets.
- **Quality signals:** Zero importers, dead_candidate.
- **Verdict reason:** Abandoned experiment.

#### ⚫ `scripts/models/draw_oracle.py` — grade F · **delete**
- **Does:** Experimental draw oracle model for prediction.
- **Talks to:** Imported by: (none). Imports: config.settings, ml.poisson, scripts.models.comprehensive_markets.
- **Quality signals:** Zero importers, dead_candidate.
- **Verdict reason:** Abandoned experiment.

#### ⚫ `scripts/models/generate_match_reasoning.py` — grade F · **delete**
- **Does:** Generate match-level reasoning/explanations for predictions.
- **Talks to:** Imported by: (none). Imports: config.settings.
- **Quality signals:** Zero importers, minimal imports, marked dead_candidate.
- **Verdict reason:** Unused stub.

#### 🟢 `scripts/models/handicap_model.py` — grade A · keep
- **Does:** Predicts match margin (home-away goals) and Asian handicap probabilities using team strength ratings, form adjustments, and contextual factors, with value betting calculations against bookmaker spreads.
- **Talks to:** Imported by: run_full_pipeline.py (live). Imports: config.settings, scripts.models (load_predictions), scripts.betting.italian_market_standards (normalize_line_to_italian).
- **Quality signals:** 737 lines, 2 dataclasses, 20 team strength ratings (Elo-based), normal distribution CDF for handicap probabilities, handles xG blending when available, detects xG placeholders to avoid drowning strength estimates, Italian market line normalization, generates margin_predictions.json and handicap_bets.json with value calculations.

#### 🔧 `scripts/models/ml_market_predictions.py` — grade B · keep
- **Does:** Generates ML-quality predictions for all 26 market categories using trained CatBoost models and ensemble feature builder, updating cards/corners/BTTS predictions with ML instead of Poisson approximations.
- **Talks to:** Imported by: (none currently). Imports: config.settings, config.leagues, scripts.prediction.ensemble_prediction_engine, scripts.models.comprehensive_markets, features.build.
- **Quality signals:** 100 lines shown. Integrates EnsemblePredictor + FeatureBuilder, injects real odds and referee features, loops over ACTIVE_LEAGUES and upcoming matches. Missing: complete market_all_markets() call and output generation.

#### 🟢 `scripts/models/model_challenger.py` — grade B · keep
- **Does:** Periodically retrains and validates candidate models against production via walk-forward CV, performing atomic swap with backup only if challenger beats current metrics by >= 0.5pp on multiple gates.
- **Talks to:** Imported by: run_full_pipeline.py (live). Imports: config.settings, scripts.prediction.ensemble_prediction_engine, scripts.models.train_unified.
- **Quality signals:** 100 lines shown (file truncated). Implements model validation gates (xG-Poisson HC accuracy +0.5pp, classifier HC accuracy +0.5pp, no NaN checks), atomic swap with backup, smoke test integration, challenger_log.json tracking (last 50 entries). Missing: complete implementation not shown.

#### ⚫ `scripts/models/optimize_ensemble_calibration.py` — grade F · **delete**
- **Does:** Optimize ensemble calibration parameters.
- **Talks to:** Imported by: (none). Imports: config.settings, storage.paths.
- **Quality signals:** Zero importers, dead_candidate.
- **Verdict reason:** Abandoned optimization attempt.

#### 🔧 `scripts/models/optimize_unified.py` — grade B · keep
- **Does:** Consolidated optimization harness for ensemble weights, model hyperparameters (Optuna), betting parameters, features, and all-markets models with options for quick/full runs.
- **Talks to:** Imported by: tests/test_integration.py (one_shot_script). Imports: config.settings, ml.poisson, ml modules.
- **Quality signals:** 100 lines shown. OptimizeConfig dataclass with 10+ parameters, target options (ensemble-weights, model-hyperparams, betting-params, features, all-markets, all), quick/full modes, multiple metric choices (log_loss, accuracy, rps, brier). Missing: actual optimization implementations.

#### ⚫ `scripts/models/optimize_weights.py` — grade F · **delete**
- **Does:** Optimize ensemble weights (superseded by weight_optimizer.py).
- **Talks to:** Imported by: (none). Imports: config.settings, scripts.analysis.backtest_unified.
- **Quality signals:** Zero importers, imports backtest module, dead_candidate. Likely replaced by weight_optimizer.py.
- **Verdict reason:** Superseded by weight_optimizer.py which is live. Keep only the new version.

#### 🟢 `scripts/models/over_under_model.py` — grade A · keep
- **Does:** Predicts goal totals using team attack/defense strength modifiers and Poisson distribution, with optional blending from ML over/under classifier and lineup xG adjustments.
- **Talks to:** Imported by: run_full_pipeline.py (live). Imports: config.settings, ml.poisson, scripts.models (load_predictions).
- **Quality signals:** 845 lines, 2 dataclasses, 23 team strength entries with attack/defense mods, blending logic for ML O/U classifier (65% ML, 35% Poisson), lineup-aware xG blending (40% confirmed, 15% predicted), factor-based adjustments for 16 different contexts, handles non-standard goal lines via interpolation, generates goal_predictions.json and over_under_bets.json.

#### ⚫ `scripts/models/predict_scorer_suitability.py` — grade F · **delete**
- **Does:** Predict which players are suitable scorers for matches.
- **Talks to:** Imported by: (none). Imports: config.leagues, config.settings, scripts.betting.player_predictions.
- **Quality signals:** Zero importers, dead_candidate.
- **Verdict reason:** Abandoned player prediction experiment.

#### ⚫ `scripts/models/predict_walkforward_markets.py` — grade F · **delete**
- **Does:** Walk-forward market predictions.
- **Talks to:** Imported by: (none). Imports: config.leagues, config.settings, features.build.
- **Quality signals:** Zero importers, dead_candidate.
- **Verdict reason:** Abandoned experiment.

#### ⚫ `scripts/models/push_accuracy.py` — grade F · **delete**
- **Does:** Experimental push (draw) accuracy optimization.
- **Talks to:** Imported by: (none). Imports: config.settings, ml.poisson, scripts.models.comprehensive_markets.
- **Quality signals:** Zero importers, dead_candidate.
- **Verdict reason:** Abandoned experiment.

#### ⚫ `scripts/models/retrain_draw_detector.py` — grade F · **delete**
- **Does:** Retrain a dedicated draw/push detector model.
- **Talks to:** Imported by: (none). Imports: config.settings, features.build, storage.paths.
- **Quality signals:** Zero importers, dead_candidate.
- **Verdict reason:** Abandoned retraining script.

#### 🔧 `scripts/models/retrain_no_odds_catboost.py` — grade A · keep
- **Does:** Retrains the catboost_no_odds model with exact 35-feature set from deployment metadata, validates against rejection gates, supports walk-forward variants, seed-based robustness, and isotonic calibration.
- **Talks to:** Imported by: (none). Imports: config.settings, ml.config, ml.data, ml.evaluation, ml.training, ml.tuning, storage.paths.
- **Quality signals:** 80 lines shown. Loads exact feature set from metadata, implements rejection gates (accuracy_min 0.50, log_loss_max 1.0, brier_max 0.205, betting_yield_min 0.025), walk-forward final fold option, seed-based robustness (median CV selection), isotonic calibration, variant suffix support for A/B testing.

#### ⚫ `scripts/models/stacking_meta.py` — grade F · **delete**
- **Does:** Experimental stacking metamodel approach.
- **Talks to:** Imported by: (none). Imports: config.settings, ml.poisson, scripts.models.comprehensive_markets.
- **Quality signals:** Zero importers, dead_candidate.
- **Verdict reason:** Abandoned experiment.

#### ⚫ `scripts/models/train_draw_specialist_production.py` — grade F · **delete**
- **Does:** Train a production draw-specialist model.
- **Talks to:** Imported by: (none). Imports: ml.draw_specialist.
- **Quality signals:** Zero importers, dead_candidate, imports non-existent or abandoned ml.draw_specialist.
- **Verdict reason:** Abandoned specialist training script.

#### 🔧 `scripts/models/train_epl.py` — grade C · keep
- **Does:** Train models for English Premier League using train_league.py scaffolding.
- **Talks to:** Imported by: (none). Imports: scripts.models.train_league.
- **Quality signals:** Thin wrapper importing train_league, designed for historical/ablation work on EPL data.

#### 🔧 `scripts/models/train_league.py` — grade A · keep
- **Does:** Train ensemble models per-league (e.g., Serie A) with feature selection, ensemble blending, evaluation, and persistence to league-specific artifact directories.
- **Talks to:** Imported by: train_epl (one_shot_script). Imports: config.settings, features.build, ml modules (config, data, ensemble, evaluation, feature_selection, models, persistence, training, tuning), storage.paths.
- **Quality signals:** 80 lines shown. Comprehensive ML pipeline: data loading, feature selection (importance + correlation pruning), multiple model types (ML models), ensemble blending, walk-forward CV, evaluation metrics (accuracy, log-loss, Brier, calibration), persistence. Used by train_epl (itself one_shot).

#### 🔧 `scripts/models/train_no_odds.py` — grade B · keep
- **Does:** Train 1X2 classifier without betting odds features for interpretability and market-independent predictions.
- **Talks to:** Imported by: (none). Imports: config.settings, ml.config, ml.data, ml.evaluation, ml.feature_selection, ml.models, ml.persistence, ml.training, ml.tuning, storage.paths.
- **Quality signals:** Not read but comprehensive feature set suggests full training pipeline for no-odds variant. Imports all major ML modules.

#### 🟢 `scripts/models/train_over_under.py` — grade A · keep
- **Does:** Trains binary CatBoost classifiers for goal lines (1.5, 2.5, 3.5, 4.5) using walk-forward CV, Optuna hyperparameter tuning, and binary-specific feature importance selection.
- **Talks to:** Imported by: scripts/pipeline/scheduler.py, scripts/pipeline/weekly_retrain.py (live). Imports: config.settings, ml.config, ml.data, ml.feature_selection, storage.paths.
- **Quality signals:** 80 lines shown. Implements binary-specific feature importance (XGBClassifier-based, avoids LABEL_MAP issue), walk-forward validation with TimeSeriesSplitter, Optuna tuning integration, binary target building from total_goals thresholds.

#### 🟢 `scripts/models/train_unified.py` — grade A · keep
- **Does:** Unified training framework supporting 7 modes (hybrid, classifier_only, xg_only, fast, with_odds, deep, optimized) with dynamic feature discovery, walk-forward CV, and Optuna hyperparameter tuning.
- **Talks to:** Imported by: model_challenger (live), tests/test_integration.py. Imports: config.settings, features.build, ml.config, ml.data, ml.feature_selection, ml.poisson, models.deep_learning.
- **Quality signals:** 100 lines shown (file much larger). TrainConfig dataclass with 30+ parameters, supports 7 distinct training modes, integration with deep learning models, CatBoost-specific tuning parameters (iterations 1500-2000, learning rates 0.02-0.03, L2 regularization), feature selection with NaN thresholding, correlation pruning. Used by model_challenger for validation logic.

#### 🔧 `scripts/models/train_walkforward.py` — grade A · keep
- **Does:** Core walk-forward training for diagnostic models, ensuring NO information from eval season leaks into training data; used by 7 diagnostic/ablation scripts.
- **Talks to:** Imported by: 7 scripts/diagnostics/* files (one_shot_script). Imports: ml.feature_selection.
- **Quality signals:** 80 lines shown. Strict time-series separation: models trained on seasons < eval_season only, excludes raw odds (independence from market), per-(league, market, eval_season) artifacts, honest evaluation (never mixed with training data). Used heavily by diagnostic suites.

#### ⚫ `scripts/models/tune_fast.py` — grade F · **delete**
- **Does:** Experimental fast tuning approach.
- **Talks to:** Imported by: (none). Imports: config.settings, ml.poisson, scripts.models.comprehensive_markets.
- **Quality signals:** Zero importers, dead_candidate.
- **Verdict reason:** Abandoned experiment.

#### ⚫ `scripts/models/unified_model.py` — grade F · **delete**
- **Does:** Experimental unified model framework.
- **Talks to:** Imported by: (none). Imports: config.settings, ml.poisson, scripts.models.comprehensive_markets.
- **Quality signals:** Zero importers, dead_candidate.
- **Verdict reason:** Abandoned experiment.

#### ⚫ `scripts/models/validate_league_deployment.py` — grade F · **delete**
- **Does:** Validate deployed models across leagues.
- **Talks to:** Imported by: (none). Imports: config.settings.
- **Quality signals:** Zero importers, dead_candidate, minimal imports.
- **Verdict reason:** Abandoned validation stub.

#### 🟢 `scripts/models/weight_optimizer.py` — grade A · keep
- **Does:** Optimizes ensemble weights dynamically via inverse-Brier weighting, factor lift decay/boost multipliers, and calibration curves using feedback analysis with graduated activation (cold_start < 20 settled, advisory 20-29, active 30+).
- **Talks to:** Imported by: run_full_pipeline.py (live). Imports: config.settings, scripts.utils.json_utils, scripts.prediction.ensemble_prediction_engine (ENSEMBLE_WEIGHTS_WITH_DEEP for ground truth).
- **Quality signals:** 80+ lines shown. Implements feedback-driven optimization with safety rails (max 5pp shift, min 2% weight floor, 70% data-driven blending), keeps 50-entry history, JSON-safe NumPy encoder for serialization. Outputs: optimized_weights.json, factor_adjustments.json, calibration_curve.json.

### `scripts/prediction/` — 22 files

#### ⚫ `scripts/prediction/__init__.py` — grade A · keep
- **Does:** Module docstring documenting the prediction engines package.
- **Talks to:** No imports or importers; purely documentation.
- **Quality signals:** Well-written docstring (1 line); correct purpose statement. Standard __init__.py.

#### 🟢 `scripts/prediction/ai_reasoning.py` — grade B · keep
- **Does:** Generates AI-powered bet reasoning using OpenAI GPT-4o-mini with caching and fallback logic.
- **Talks to:** imports: config/settings.py (DATA_DIR). Imported by: scripts/pipeline/run_full_pipeline.py
- **Quality signals:** 150 lines of well-structured code with OpenAI client initialization, caching system (cache_key, load_cached_reasoning, save_cached_reasoning). Type hints present. Graceful fallback when OpenAI unavailable. Missing: comprehensive error handling and edge case testing indicated by multiple try-except blocks without detailed logging.

#### 🔧 `scripts/prediction/compute_shadow_roi.py` — grade B · keep
- **Does:** Joins shadow predictions, manual odds, and settled outcomes to compute per-market ROI with bootstrap CI.
- **Talks to:** imports: models/simulator/backtests/roi_bootstrap.py. No recorded importers; one-shot diagnostic script.
- **Quality signals:** 100+ lines of focused logic for bet settlement: _total_goals, _resolve_1x2, _settle_binary, _settle_multiclass functions. Handles binary/multiclass markets with Brier scoring. Edge-threshold parameterization. Type hints present. README docstring with usage examples. Missing: test coverage for complex market logic.

#### 🟢 `scripts/prediction/current_form_calculator.py` — grade B · keep
- **Does:** Calculates real-time team form metrics (W/D/L, pts, goals, form_status) from recent matches, fed by 5 other modules.
- **Talks to:** imports: config/settings.py, config/team_names.py. Imported by: ensemble_prediction_engine.py, generate_epl_supplementary.py, predict_unified.py, tests/test_edge_cases.py, tests/test_feature_impact.py
- **Quality signals:** 80+ lines of public functions (load_recent_matches, calculate_team_form, calculate_head_to_head, get_elo_ratings, calculate_all_forms). Type hints on function signatures. Pandas operations with safe null handling. Constants for derbies and big stadiums defined. Missing: detailed docstrings on private helpers, edge case handling for degenerate seasons.

#### 🟢 `scripts/prediction/ensemble_prediction_engine.py` — grade A · keep
- **Does:** Combines three prediction methods (factor-based 40%, xG-based 40%, ML classifier 20%) to generate ensemble predictions with calibration pipeline.
- **Talks to:** imports: config/leagues.py, config/settings.py, 8 feature modules (draw_detection, enhanced_momentum, enhanced_weather, formation_analysis, injury_impact, market_intelligence, player_xg_model, prediction_calibration, sentiment_analysis, venue), ml/correction_layer.py, ml/ensemble.py, models/deep_learning.py, scraper/injuries.py, scripts/prediction/current_form_calculator.py, predict_unified.py, referee_integration.py, weather_integration.py. Imported by: 9 modules including calibration_analysis.py, ml_market_predictions.py, model_challenger.py, weight_optimizer.py, run_full_pipeline.py, weekly_retrain.py, predict_league.py, predict_unified.py, test_ensemble_predictions.py
- **Quality signals:** 100+ lines with 10 major classes: DrawDetector, XGPredictor, XCompLoader, MetaLearnerCombiner, MLClassifier, PlayerXGPredictor, FeatureBuilder, OverUnderPredictor, EnsemblePredictor, _NumpySafeEncoder. Type hints throughout. Graceful degradation with try-except blocks for optional features (FORMATION_AVAILABLE, DEEP_LEARNING_AVAILABLE, CALIBRATION_AVAILABLE). Ensemble weights tuned and configurable. This is the core ensemble engine.

#### 🟢 `scripts/prediction/formation_predictor.py` — grade B · keep
- **Does:** Predicts likely formations for upcoming matches using frequency analysis from SofaScore starter positions and historical formations.
- **Talks to:** imports: config/team_names.py. Imported by: scripts/pipeline/run_full_pipeline.py
- **Quality signals:** 185 lines with 4 public functions: get_team_recent_formations, predict_formation, add_formation_predictions, _infer_formation_from_positions. Formation position mapping hardcoded with 11 formations. Robust fallback to 4-3-3 if no data. Type hints present. Confidence scoring via Counter frequency. Missing: validation of inferred formations against actual match results.

#### 🟢 `scripts/prediction/generate_epl_supplementary.py` — grade B · keep
- **Does:** Generates EPL supplementary JSON files (bookmaker analysis, cross-market signals, market intelligence, sentiment, weather, referees) merged with existing Serie A files.
- **Talks to:** imports: config/settings.py, features/bookmaker_analysis.py, scraper/injuries.py, scripts/prediction/current_form_calculator.py, h2h_generator.py, lineup_predictor.py, weather_integration.py, scripts/utils/json_utils.py. Imported by: scripts/pipeline/run_full_pipeline.py
- **Quality signals:** 80+ lines with EPL-specific logic: FULL_TO_SHORT team name mapping, EPL_RIVALRIES constant, 18 functions including generate_epl_bookmaker_analysis, generate_epl_sentiment, generate_epl_weather. Merges output into shared files for multi-league support. Missing: comprehensive EPL odds provider support (currently imports from single source), error handling for partial data.

#### 🟢 `scripts/prediction/generate_unified_report.py` — grade A · keep
- **Does:** Consolidates ~19 separate JSON files (predictions, bets, form, lineups, weather, etc.) into a single unified_report.json for dashboard/API consumption.
- **Talks to:** imports: scripts/utils/json_utils.py. Imported by: scripts/pipeline/run_full_pipeline.py
- **Quality signals:** 435 lines of robust data aggregation: _normalize_match_key, _build_match_index_from_list/dict, _extract_matches, _extract_bets_for_match, generate_unified_report. Handles 19 input files with graceful degradation (missing data skipped). Normalizes match keys for fuzzy matching. Type hints throughout. Output includes _data_sources and _sections_available metadata. Comprehensive docstring with input file listing.

#### 🟢 `scripts/prediction/h2h_generator.py` — grade B · keep
- **Does:** Computes head-to-head records for upcoming matches (total meetings, wins per side, draws, last result, avg goals).
- **Talks to:** imports: config/settings.py. Imported by: scripts/pipeline/run_full_pipeline.py, scripts/prediction/generate_epl_supplementary.py
- **Quality signals:** 165 lines with single public function generate_h2h_for_upcoming. Reads matches.parquet, computes head-to-head aggregates (total meetings, home/away wins, draws, avg goals). Supports per-league filtering. Merges results into shared file for multi-league support. Missing: validation against duplicates, edge case handling for matches with mismatched team name normalization.

#### 🟢 `scripts/prediction/intelligence_integrator.py` — grade A · keep
- **Does:** Applies soft intelligence adjustments (sentiment, player analysis, market intelligence) to ensemble probabilities with bounded shifts.
- **Talks to:** No imports or importers. Imported by: scripts/pipeline/run_full_pipeline.py
- **Quality signals:** 272 lines of well-architected probability adjustment system. 5 public functions: apply_sentiment_adjustment, apply_player_adjustment, apply_market_intelligence_adjustment, apply_all_intelligence, _apply_and_normalize. IntelligenceAdjustment dataclass tracks all adjustments. Maximum shift limits defined (8pp sentiment, 7pp player, 5pp market, 15pp combined). Re-normalization ensures probabilities sum to 1.0. Comprehensive docstrings with examples. Type hints throughout.

#### 🟢 `scripts/prediction/lineup_predictor.py` — grade A · keep
- **Does:** Predicts starting XI and formation for upcoming matches using player start frequency, suspension risk, and injury data from Sofascore and match JSONs.
- **Talks to:** imports: config/settings.py, config/team_names.py, scripts/prediction/player_positions.py. Imported by: scripts/pipeline/run_full_pipeline.py, scripts/prediction/generate_epl_supplementary.py, scripts/prediction/player_positions.py, web/app.py
- **Quality signals:** 80+ lines of symbol overview; full file likely 400+ lines. FORMATION_TACTICAL_SLOTS hardcoded with 20 formation variants (3-back, 4-back, 5-back). 20 public functions including predict_formation, predict_starting_xi, compute_suspensions, get_fitness_concerns. Handles yellow card suspension logic (5 yellows = ban). Circular import with player_positions.py via _get_tactical_labels. Type hints present. Used by web/app.py (live web dependency).
- **Leagues (fixed 2026-08-01):** reads **both** `player_match_stats.parquet` and `player_match_stats_premier_league.parquet` via the `_PLAYER_STATS_FILES` constant — `_load_current_season_stats()` for prediction and `evaluate_past_predictions()` for grading. Before this it opened the Serie A file only, so all 20 EPL clubs had **zero** XI prediction path and every archived EPL prediction graded as a miss (missing input, read as model failure). Each league is filtered to its **own** `season.max()` so a lagging league is not erased. Verified: 20/20 EPL clubs now produce a full 11; the two files share zero team names, so the concat is safe.
- ⚠️ Sofascore club names are not canonical (`Liverpool FC`, `Man City`, `Milan`, `Napoli`). Resolution lives in **`_resolve_team_name(team, league_teams, preseason_clubs)`** — exact → `normalize_team` → substring against the league table, then exact → `normalize_team` against the **pre-season friendly** clubs. `normalize_team("Liverpool")` alone does **not** resolve; the substring step is load-bearing, not a nicety. It returns `(name, source)` where source is `"league"`, `"preseason"` or `None`.
- **Promoted clubs (fixed 2026-08-02):** `_load_current_season_stats()` filters to `season.max()`, which at matchweek 1 is still **last** season — so a newly promoted club has zero rows there, failed every league rung, and was dropped with `Warning: No Sofascore data for X` before `predict_team_lineup` ever ran. Measured live: **all six** promoted clubs (Coventry City / Hull / Ipswich, Frosinone / Monza / Venezia — 3 per league) were being skipped while holding 25–29 friendly players and a real formation on disk; `predict_team_lineup` returns a full 11-man XI when handed one. That is 15% of both books, and precisely the clubs the pre-season signal exists for. The pre-season rung deliberately offers **no substring match**: on the league rung a bad match degrades an XI built from real data, here it would invent an entire lineup from another squad. Club list comes from `preseason_club_names()`, which shares `_friendlies_frame()` with `load_preseason_signal` so the two can never disagree about which pre-season is current.
- ⚠️ **`load_preseason_signal(team, season=, before=)` — the season label is not a time bound.** `sofascore_friendlies._season_for` files every June-onward friendly under the season starting that August, so a **March** friendly carries the *previous* season's label. Anything replaying a historical point in time must pass `before=` (a date cutoff) or it will be handed matches from months in its own future. Production passes neither and is correct — there is no future to leak from today. To change how a past matchweek is reconstructed, the file is `scripts/analysis/backtest_preseason_signal.py` (`_season_opener`).

#### 🔧 `scripts/prediction/make_manual_odds_template.py` — grade A · keep
- **Does:** Generates blank JSON template for user to manually enter bookmaker odds for shadow predictions (Track 2 workflow).
- **Talks to:** No imports or importers recorded; one-shot standalone utility.
- **Quality signals:** 215 lines of focused UI/data generation. Reads simulator_shadow_log.json, writes template with blank odds fields + predicted probs for reference. MARKETS_TO_REQUEST constant: 70+ market labels (1X2, O/U, BTTS, cards, corners, AH, shots, SOT). Functions: build_template_for_fixture, run, main. Preserve-existing flag to keep already-filled odds. Clear usage instructions embedded. All necessary type hints.

#### 🟢 `scripts/prediction/match_stats_predictor.py` — grade B · keep
- **Does:** Predicts match statistics (shots, corners, cards, possession, fouls) from rolling team averages via SofaScore and FBref data.
- **Talks to:** imports: config/team_names.py. Imported by: scripts/pipeline/run_full_pipeline.py
- **Quality signals:** 120+ lines with 2 public functions: _build_team_rolling_stats, predict_match_stats, add_match_stats_predictions. Aggregates SofaScore (shots, shots on target, fouls, touches, passes) and FBref (cards) separately then merges. Rolling average computation. Missing: validation against actual stats, edge case handling for incomplete data.

#### 🟢 `scripts/prediction/player_positions.py` — grade B · keep
- **Does:** Extracts detailed tactical positions from Sofascore match JSONs and builds player position map for lineup assignments.
- **Talks to:** imports: scripts/prediction/lineup_predictor.py (FORMATION_TACTICAL_SLOTS, _get_tactical_labels). Imported by: scripts/prediction/lineup_predictor.py
- **Quality signals:** 100+ lines with 4 public functions: extract_positions_from_matches, build_player_position_map, _formation_to_slots, _formation_index_to_label. Parses formation strings and infers position counts (D/M/F). Matches Sofascore starters list ordering. Aggregates position counts across season matches. Circular import with lineup_predictor.py. Type hints present. Missing: validation of position assignments against tactical slot definitions.

#### 🟢 `scripts/prediction/predict_league.py` — grade B · keep
- **Does:** Multi-league prediction router that runs ensemble predictions parameterized by league (Serie A, Premier League, etc.).
- **Talks to:** imports: config/settings.py (DATA_DIR, MODELS_DIR, LEAGUES, DEFAULT_LEAGUE). Imported by: scripts/pipeline/run_full_pipeline.py
- **Quality signals:** 80+ lines with league routing logic. Functions: _league_model_dir, _league_predictions_path, _league_odds_path, has_league_model, fetch_league_odds, run_predictions_for_league, run_predictions_multi, load_all_predictions. Handles backward-compat for Serie A models at root. Type hints present. League display names mapping. Missing: comprehensive error handling for missing league directories, multi-league parallel execution.

#### 🟢 `scripts/prediction/predict_unified.py` — grade A · keep
- **Does:** Single entry point for all Serie A prediction modes (ensemble, factor, xG, ML, market) with betting strategy support and bankroll management.
- **Talks to:** imports: config/settings.py, features/bankroll_manager.py, features/injury_impact.py, features/value_betting.py, scraper/injuries.py, scripts/prediction/current_form_calculator.py, ensemble_prediction_engine.py, referee_integration.py, weather_integration.py. Imported by: scripts/pipeline/run_full_pipeline.py, scripts/prediction/ensemble_prediction_engine.py, tests/test_automation.py, tests/test_integration.py, tests/test_pipeline.py
- **Quality signals:** 100+ lines with PredictionConfig dataclass, _NumpySafeEncoder JSON handler, and 12+ public functions. Supports 5 prediction modes (ensemble, factor, xg, ml, market) and 5 betting strategies. Handles single match filtering (--match), matchday filtering (--matchday), output formats (text/json), feature toggles (weather, referee, formation, injuries, value betting, bankroll). Type hints throughout. Comprehensive CLI argument parsing. Test coverage via test_automation.py, test_integration.py, test_pipeline.py.

#### 🟢 `scripts/prediction/referee_integration.py` — grade A · keep
- **Does:** Integrates referee assignments with validated referee bias data (home-favoring, away-favoring, strict, lenient) to adjust predictions.
- **Talks to:** imports: config/settings.py, features/referee.py. Imported by: scripts/models/ml_market_predictions.py, scripts/pipeline/run_full_pipeline.py, scripts/prediction/ensemble_prediction_engine.py, scripts/prediction/predict_unified.py, tests/test_feature_impact.py
- **Quality signals:** 100+ lines with hardcoded referee classifications: HOME_FAVORING_REFS (6 refs, 0.52-0.58 home win rate, +10-16pp lift), AWAY_FAVORING_REFS (5 refs, 0.29-0.38 home win rate, -5 to -13pp lift), STRICT_REFS (5 refs, high card rates), LENIENT_REFS (3 refs, low card rates). Functions: classify_referee, get_referee_factors, load_referee_assignments, predict_referee_from_history, analyze_referee_impact, create_referee_template. Type hints throughout. 8 seasons of validation data.

#### 🟢 `scripts/prediction/sentiment_analyzer.py` — grade B · keep
- **Does:** Analyzes match sentiment via web search (Google Gemini or Groq) with fan confidence, media hype, injury impact, transfer buzz, and contrarian opportunity detection.
- **Talks to:** imports: config/leagues.py, config/settings.py. Imported by: scripts/pipeline/monitor.py, scripts/pipeline/run_full_pipeline.py, tests/test_feature_impact.py
- **Quality signals:** 100+ lines with API client classes (APIUsageTracker, GeminiClient, GroqClient, DataDrivenSentiment, SentimentAnalyzer, SentimentScore, MatchSentiment). Supports two web search backends (Gemini primary, Groq fallback) with cost controls (GROQ_DAILY_BUDGET_USD, usage tracking). Italian-specific rivalries defined. Cache system with 6-hour duration. Missing: comprehensive rate limit handling, fallback logic when both APIs unavailable.

#### 🧪 `scripts/prediction/settle_shadow_log.py` — grade B · keep
- **Does:** Settles shadow predictions against actual match outcomes; computes Brier scores, accuracy, log-loss per market; appends to growing ledger.
- **Talks to:** No recorded importers; imported by: tests/simulator/test_shadow_pipeline.py
- **Quality signals:** 336 lines of comprehensive settlement logic. Functions: _load_actuals_for_fixtures, _settle_binary, _settle_multiclass, settle_fixture, run, main. Loads matches.parquet + Sofascore + events to construct actuals. Scores 1X2, double chance, O/U, BTTS, AH, corners/cards/shots, top scorers. Per-market aggregation with Brier/accuracy/log-loss. Type hints throughout. Used by test_shadow_pipeline.py; not imported by production.

#### 🔧 `scripts/prediction/simulate_upcoming.py` — grade B · keep
- **Does:** Runs full simulator (Poisson/Dixon-Coles + Phase 2 corners/cards/shots + Phase 5 player profiles) on upcoming fixtures; outputs shadow predictions to simulator_shadow_log.json.
- **Talks to:** imports: models/simulator/base_rates/card_rates.py, corner_rates.py, lineup_allocator.py, player_profiles.py, shot_generator.py, models/simulator/engine/dixon_coles.py, simulator.py, models/simulator/markets.py. No recorded importers; one-shot runner.
- **Quality signals:** 80+ lines with 3 key functions: _fit_all_models, _predict_lambdas, _find_feature_row, run, main. Imports 6 simulator modules for complete match simulation. Loads features.parquet, fits Poisson/Dixon-Coles per league, runs N_TRIALS simulations per fixture. Deterministic per (match_key, seed) for shadow integrity. Missing: input validation, error handling for missing features.

#### 🟢 `scripts/prediction/standings_generator.py` — grade A · keep
- **Does:** Generates current Serie A standings from matches.parquet; computes position, points, W/D/L, GF/GA/GD, and last-5 form string.
- **Talks to:** imports: config/settings.py. Imported by: scripts/pipeline/run_full_pipeline.py
- **Quality signals:** 164 lines with single public function generate_current_standings. Handles multi-league matches with league filtering (serie_a/Serie A/I1). Deduplicates across data sources (keeps first occurrence). Computes home/away splits (played, W/D/L, GF/GA, PPG). Last-5 form string. Type hints throughout. Atomic write via atomic_write_json. Clear docstring with season parameter.

#### 🟢 `scripts/prediction/weather_integration.py` — grade B · keep
- **Does:** Fetches weather forecasts for match venues via Open-Meteo API (free, no key) with fallback to manual patterns; quantifies impact on goals, home win, cards.
- **Talks to:** imports: config/settings.py. Imported by: scripts/prediction/ensemble_prediction_engine.py, scripts/prediction/generate_epl_supplementary.py, scripts/prediction/predict_unified.py, tests/test_edge_cases.py, tests/test_feature_impact.py
- **Quality signals:** 100+ lines with STADIUM_COORDS hardcoded for 50+ teams (Serie A + EPL). 4 public functions: get_weather_forecast, classify_weather, get_weather_impact, fetch_all_match_weather. Open-Meteo API call with fallback to historical patterns. Weather impact quantified (rain +0.20 goals, wind -0.22 goals, rain +3.2% home win, cold +5% cards). Type hints present. Missing: validation of Open-Meteo API reliability, caching strategy.

### `scripts/betting/` — 25 files

#### 🟢 `scripts/betting/__init__.py` — grade A · keep
- **Does:** Package marker with module docstring documenting betting systems, staking, market analysis, and bet journals.
- **Talks to:** Imported by bankroll_loader.py, verify_invariants.py, monitor.py, notify.py, run_full_pipeline.py
- **Quality signals:** Clean module docstring (2 lines); no implementation; serves as package marker; used as a dependency in multiple pipeline files

#### 🟢 `scripts/betting/auto_settle.py` — grade B · keep
- **Does:** Automatic settlement orchestrator that parses odds from external APIs, matches to pending bets, computes CLV, and settles bets with profit/loss calculation.
- **Talks to:** Imports: config/settings.py, ml/correction_layer.py, scripts/betting/{bankroll_loader, bet_journal, parlay_tracker, prediction_tracker}.py; Imported by: scripts/pipeline/scheduler.py
- **Quality signals:** 674 LOC; type-hinted functions; comprehensive settlement logic with multi-source odds handling; no tests but heavily used in scheduler; validates commence_time staleness to prevent prior-season bug

#### 🟢 `scripts/betting/bankroll_loader.py` — grade A · keep
- **Does:** Loads initial bankroll from config and computes current balance by aggregating settled bets from the journal.
- **Talks to:** Imports: config/settings.py, scripts/betting/__init__.py, scripts/pipeline/notify.py; Imported by: auto_settle.py, betting_unified.py, risk_controls.py, run_full_pipeline.py, scheduler.py
- **Quality signals:** ~50 LOC; single-purpose pure functions; clear interface (load_bankroll_config, compute_current_bankroll, get_effective_bankroll); well-documented; no type stubs but straightforward logic

#### 🔧 `scripts/betting/bankroll_simulator.py` — grade B · keep
- **Does:** Monte Carlo ruin simulator that tests thousands of season paths across Kelly fractions to find optimal stake sizing while constraining bankruptcy risk.
- **Talks to:** Imports: config/settings.py; Imported by: tests/test_monte_carlo.py
- **Quality signals:** 402 LOC; type-hinted dataclasses (RuinSimConfig, BetOpportunity); uses numpy/pandas for simulation; intentionally invoked only by test suite for monte carlo validation; not part of live betting loop

#### 🟢 `scripts/betting/benchmark_tracker.py` — grade B · keep
- **Does:** Computes benchmark statistics on betting performance (ROI, win rate, streaks, CLV analysis) from settled bets and generates reports.
- **Talks to:** Imports: config/settings.py, scripts/betting/bet_journal.py; Imported by: scripts/pipeline/telegram_bot.py, web/app.py
- **Quality signals:** 416 LOC; well-organized analysis functions (streak detection, weekly trends, best/worst periods); uses betting journal as single source; no model versioning coupling; outputs comprehensive performance reports

#### 🟢 `scripts/betting/bet_journal.py` — grade A · keep
- **Does:** Canonical bet ledger and single source of truth: records all bets, stores CLV, settles outcomes, and manages model versioning snapshots; exports JSON cache for fast reads.
- **Talks to:** Imports: config/settings.py; Imported by: auto_settle, benchmark_tracker, betting_unified, clv_capture, clv_tracker, ledger, live_bet_context, parlay_generator, health_check, run_full_pipeline, scheduler, telegram_bot, tests/test_bet_journal, web/{advisor, app}
- **Quality signals:** 1341 LOC; no external imports needed (self-contained); comprehensive atomic I/O with fcntl locking; model versioning via git SHA snapshots; settlement validation to prevent stale commence_time bug (documented in code); used by 15+ importers; production-grade locking

#### 🟢 `scripts/betting/betting_unified.py` — grade A · keep
- **Does:** Main unified betting engine: generates bets by blending multiple prediction sources (models, market wisdom), sizes stakes via Kelly fraction, applies bankroll/risk controls, and outputs bet slips.
- **Talks to:** Imports: config/{leagues, settings}, scripts/betting/{bankroll_loader, bet_journal, clv_tracker, entry_timer, risk_controls}, scripts/pipeline/notify; Imported by: backtest_multimarket, odds_edge_monitor, parlay_generator, run_full_pipeline, tests/test_integration
- **Quality signals:** 3553 LOC; comprehensive classes (BettingConfig, ValueBet, BetSlip, AccumulatorBet, UnifiedBettingEngine); type-hinted throughout; extensive value calculation logic; Kelly sizing; confidence-based filtering; integrates both draw-based and multi-market strategies

#### 🟢 `scripts/betting/clv_capture.py` — grade B · keep
- **Does:** Post-placement CLV capture: reads cached odds snapshots after bets are placed, matches bets to closing odds, computes CLV (closing odds vs placement odds), and records to history.
- **Talks to:** Imports: config/settings.py, scripts/betting/bet_journal.py; Imported by: scripts/pipeline/scheduler.py
- **Quality signals:** 455 LOC; parses bookmaker snapshots; handles multiple bookmakers (Pinnacle, Betfair, etc.); match fuzzy matching for odd naming; CLI interface with argparse; well-structured capture and append logic

#### 🟢 `scripts/betting/clv_tracker.py` — grade B · keep
- **Does:** CLV (Closing Line Value) tracking system: aggregates historical CLV per market, computes statistics (mean, std, N), and identifies sharp edges for market-by-market performance analysis.
- **Talks to:** Imports: config/{leagues, settings}, scripts/betting/bet_journal.py; Imported by: betting_unified.py, run_full_pipeline.py, tests/test_bet_journal
- **Quality signals:** 758 LOC; comprehensive CLV computation with outlier handling; per-market statistics; used by betting_unified for edge weighting; no major issues but moderate complexity

#### ❓ `scripts/betting/entry_timer.py` — grade B · keep
- **Does:** Bet entry timing optimizer: analyzes historical odds snapshots to recommend WHEN to place bets (NOW vs WAIT vs AVOID) based on sharp money velocity and divergence patterns.
- **Talks to:** Imports: none; Imported by: none (standalone analysis module)
- **Quality signals:** 512 LOC; well-structured OddsTimeline and EntryTimingAnalyzer classes; type-hinted; comprehensive odds velocity calculations and recommendation logic; has main block but no importers (analysis tool, not integrated into live betting)

#### 🟢 `scripts/betting/extended_markets.py` — grade A · keep
- **Does:** Extended market generation: derives 24+ non-standard markets (double chance, team totals, HTFT, win-to-nil, corners, etc.) from 1x2 and goal predictions using Poisson/bivariate copula models.
- **Talks to:** Imports: config/{leagues, settings}; Imported by: parlay_generator.py, run_full_pipeline.py, tests/test_monte_carlo
- **Quality signals:** 963 LOC; extensive market generation logic; Poisson+copula joint distributions; 24+ market types; type-hinted; comprehensive market coverage (corners, cards, penalties, handicaps, exact scores); used by parlay generator for expanded combinatorics

#### 🟢 `scripts/betting/fair_odds_tracker.py` — grade B · keep
- **Does:** Fair odds historical ledger: records model probabilities (fair odds = 1/prob) before matches and tracks accuracy against outcomes; computes calibration and model vs market comparison.
- **Talks to:** Imports: scripts/utils/ledger.py; Imported by: web/app.py
- **Quality signals:** ~330 LOC; clean ledger-based design; probability bucket calibration; model vs market accuracy comparison; uses utility ledger functions; no complex logic; serves dashboard reporting

#### 🟢 `scripts/betting/italian_market_standards.py` — grade B · keep
- **Does:** Market standardization for Italian bookmakers: filters non-standard Asian/European lines, converts odds to Italian popular formats (O/U 2.5, 1-0 handicaps), ensures only bookmaker-native lines are used.
- **Talks to:** Imports: config/settings.py; Imported by: handicap_model.py, run_full_pipeline.py, tests/{test_betting_logic, test_pipeline}
- **Quality signals:** 472 LOC; defines ItalianBet dataclass; comprehensive line filtering (removes Asian halves, European +1/-1 handicaps); market name normalization; applied to output bets; domain-specific (Italian market knowledge)

#### 🔧 `scripts/betting/ledger.py` — grade B · keep
- **Does:** Financial ledger with atomic caches: provides read API for balance/ROI/settlement stats derived from bet_journal, and rebuild logic to regenerate bankroll.json and history.json caches.
- **Talks to:** Imports: scripts/betting/bet_journal.py; Imported by: none
- **Quality signals:** 598 LOC; comprehensive invariant verification (ledger ↔ bankroll ↔ history); atomic write patterns; settles via bet_journal; detect duplicate keys and stale commence_time bugs; production-grade validation but marked one_shot due to zero importers

#### 🟢 `scripts/betting/live_bet_context.py` — grade B · keep
- **Does:** Live bet context builder: loads pending bets, parlay map, and match metadata; generates narrative commentary on active bets and their parlay membership.
- **Talks to:** Imports: config/team_names.py, scripts/betting/bet_journal.py, scripts/utils/parsing.py; Imported by: scripts/pipeline/scheduler.py
- **Quality signals:** 362 LOC; contextual bet commentary generation; parlay map cross-reference; fuzzy match handling; used by scheduler for live updates; moderate complexity, well-structured

#### 🟢 `scripts/betting/odds_edge_monitor.py` — grade B · keep
- **Does:** Live odds edge scanner: continuously monitors odds for edges (sharp-vs-soft divergence, arbitrage, consensus scoring), triggers alerts, and feeds web dashboard.
- **Talks to:** Imports: config/{leagues, settings}, scripts/betting/betting_unified.py, scripts/pipeline/notify; Imported by: web/app.py
- **Quality signals:** 843 LOC; comprehensive edge detection (divergence, arb, sharp consensus); state machine (scan vs daemon); integration with web app and notifications; well-scoped but moderate complexity

#### 🟢 `scripts/betting/parlay_generator.py` — grade A · keep
- **Does:** Parlay (accumulator) generator: builds multi-leg combinations from value legs using SGP/copula joint probabilities, Monte Carlo simulation, Kelly sizing, and quality filtering.
- **Talks to:** Imports: config/settings.py, scripts/betting/{bet_journal, betting_unified, extended_markets, parlay_tracker}, scripts/utils/{json_utils, parsing}; Imported by: backtest_unified.py, run_full_pipeline.py, tests/test_monte_carlo
- **Quality signals:** 3069 LOC; extensive accumulator logic: SGP models, copula correlations, Monte Carlo simulation, Kelly sizing, quality scoring; comprehensive calibration system (beta_k); market conflict detection; highly sophisticated with strong type hints; load-bearing for parlay betting

#### 🟢 `scripts/betting/parlay_tracker.py` — grade B · keep
- **Does:** Parlay settlement and statistics: records placed parlays, settles outcomes when legs resolve, and computes per-category parlay performance metrics.
- **Talks to:** Imports: config/settings.py; Imported by: auto_settle.py, parlay_generator.py
- **Quality signals:** ~60 LOC; clean settlement API with leg result checking; per-category statistics; simple but essential for parlay tracking

#### 🟢 `scripts/betting/player_predictions.py` — grade A · keep (rewritten 2026-06-04, extended 2026-06-11)
- **Does:** Player floor-market engine — 19 markets: anytime goalscorer, shots O0.5/1.5/2.5, SoT O0.5/1.5, fouls O0.5/1.5, fouled O0.5, tackles O0.5/1.5/2.5, passes completed O19.5/29.5/39.5, duels won O2.5/4.5, interceptions O0.5/1.5. Empirical leak-free per-90 priors (expanding mean over prior 60+min matches, groupby.transform — never g.apply) × projected minutes → Poisson tail (negative binomial for passes, fitted dispersion r≈10; measured over-dispersion 4.25x), out-of-fold isotonic on ECE-flagged markets, possession-context multiplier on passes (β=0.5, validated 8/8 season cells 2026-06-11). The old 10-CatBoost stack was deleted 2026-06-04 (died on within-position control).
- **Talks to:** Imports: scipy, sklearn (isotonic), pandas; reads player_match_stats[+_premier_league].parquet + match_team_stats.parquet (possession). Imported by: web/app.py (_player_engine → /api/projections player_floors), player_prop_odds.py, prop_tracker.py. CLI: `python3 -m scripts.betting.player_predictions validate|predict [league]`.
- **Quality signals:** Typed; leak-free by construction; validate() gate = Brier-vs-base per market (all 19 TRUST, 2026-06-11); strong controls on record in .plans/passes-tackles-floors-findings.md; unit-tested in tests/test_player_floors.py (leak-freedom, tail math vs scipy, dispersion fit, display contract). Betting GATED (betting_rules.json Player_Props) — validated probability, not edge.

#### 🟢 `scripts/betting/player_prop_odds.py` — grade B · keep
- **Does:** Player prop odds fetcher: queries Odds API for player prop markets (anytime goal, shots, cards, etc.), matches model predictions to odds, identifies value, and saves bet candidates.
- **Talks to:** Imports: config/{api_keys, settings, team_names}, scripts/betting/{player_predictions, prop_tracker}; Imported by: run_full_pipeline.py, scheduler.py
- **Quality signals:** 707 LOC; API integration (Odds API); player fuzzy matching; model prediction alignment; value identification; proper error handling for API failures; live fetch capability

#### 🟢 `scripts/betting/player_props.py` — grade B · keep
- **Does:** Player props calculator: computes Poisson-based player prop probabilities (anytime scorer, shots O/U, fouls, tackles, etc.) from Sofascore rolling stats.
- **Talks to:** Imports: config/settings.py; Imported by: run_full_pipeline.py
- **Quality signals:** ~200 LOC; Poisson-based probability models; uses Sofascore rolling stats; lightweight alternative to ML models; integrated into pipeline for quick prop calculation

#### 🟢 `scripts/betting/prediction_tracker.py` — grade B · keep
- **Does:** Prediction performance tracker: scores predictions against outcomes, computes ROI/accuracy metrics, and detects performance drift (regression detection).
- **Talks to:** Imports: config/{settings, team_names}; Imported by: auto_settle.py
- **Quality signals:** 483 LOC; comprehensive metrics (accuracy, ROI, calibration); rolling window analysis; drift detection with configurable thresholds; used by auto_settle for health monitoring

#### 🟢 `scripts/betting/prop_tracker.py` — grade B · keep
- **Does:** Player prop settlement and calibration: settles individual player prop outcomes, computes performance metrics per player/market, and generates calibration adjustments for probability rescaling.
- **Talks to:** Imports: config/{settings, team_names}, scripts/utils/{ledger, parsing}; Imported by: player_prop_odds.py, web/app.py
- **Quality signals:** 632 LOC; outcome evaluation logic; per-player calibration statistics; hypothetical Kelly PnL analysis; used by web app for prop performance reporting

#### 🟢 `scripts/betting/risk_controls.py` — grade B · keep
- **Does:** Risk gates system: checks bankroll drawdown limits, consecutive loss streaks, bankroll floor breaches, and market health conditions to gate new bets.
- **Talks to:** Imports: config/settings.py, scripts/betting/bankroll_loader.py; Imported by: betting_unified.py, scheduler.py
- **Quality signals:** 452 LOC; RiskConfig dataclass with YAML defaults; four gate functions (drawdown, consecutive losses, floor, market health); produces risk_state.json; integrated into betting_unified for pre-bet gating

#### 🔧 `scripts/betting/verify_invariants.py` — grade A · keep
- **Does:** Invariant verification CLI: checks ledger↔bankroll↔history agreement, detects cache drift, validates bet status values, and reports violations with repair instructions.
- **Talks to:** Imports: scripts/betting/__init__.py (via ledger); Imported by: none
- **Quality signals:** 50 LOC; clean CLI interface with --quiet/--json flags; runs ledger.verify_invariants() and formats output; intentionally not imported (run as: python -m scripts.betting.verify_invariants); production utility for ledger repair

### `scripts/analysis/` — 18 files

#### 🟢 `scripts/analysis/__init__.py` — grade A · keep
- **Does:** Module marker with module-level documentation string for analysis package.
- **Talks to:** none
- **Quality signals:** Simple init file with docstring; zero complexity; serves as documentation for package purpose (backtesting, diagnostics, performance reporting, data audits)

#### 🟢 `scripts/analysis/backtest_preseason_signal.py` — grade A · keep
- **Does:** Replays past matchweeks to measure whether the pre-season friendly signal improves XI prediction, and grids the four `PRESEASON_*` constants in `lineup_predictor.py` (which were judgement, never measurement). Writes `data/analysis/preseason_signal_backtest.json`.
- **Talks to:** imports `scripts/prediction/lineup_predictor.py` (`get_starter_frequency`, `load_preseason_signal`, `_PLAYER_STATS_FILES`, `SOFASCORE_DIR`). Reads `player_match_stats{,_premier_league}.parquet` + `friendlies_*.parquet`. Imported by nothing — CLI only. Tests: `tests/test_backtest_preseason_signal.py`.
- **Run:** `python3 -m scripts.analysis.backtest_preseason_signal [--seasons A,B] [--rounds 1,2,3] [--sweep]`
- **The replay is the load-bearing part.** For MW `k` of season `S` it reconstructs exactly what `_load_current_season_stats` would have returned: at `k==1` no season-`S` rows exist so the table is `S-1` (the case the signal exists for); at `k>1` it is rounds `1..k-1`. Friendlies are pinned to season `S` via `load_preseason_signal(team, season=S)` — **the default (newest on disk) would be look-ahead leakage** and would manufacture a good result.
- **Three arms:** `naive` (top 11 by raw start count — the floor), `off`, `on`. Always read `player_slots_changed` beside the delta: an accuracy delta with ~zero changed slots is noise whatever its sign. `--sweep` calibrates on the earlier season and validates on a holdout, and restores the module globals in a `finally` so a crashed sweep cannot leave swept constants in the live predictor.
- **To re-calibrate after a new season lands, this is the file.** To change what the signal *does*, it is `lineup_predictor.py`.

#### ⚫ `scripts/analysis/backtest_multimarket.py` — grade B · **delete**
- **Does:** Backtests 3 betting markets (1X2, O/U 2.5, Asian Handicap) against Pinnacle sharp odds with lineup-adjusted xG comparison.
- **Talks to:** imports: config/settings.py (DATA_DIR, MODELS_DIR), storage/paths.py (features_path), scripts/betting/betting_unified.py (remove_overround), features/player_xg_model.py, ml/ensemble.py; imported_by: scripts/models/optimize_weights.py, scripts/models/optimize_unified.py
- **Quality signals:** Well-structured with edge threshold sweeps, Poisson matrix calculations, situational context tagging. Imports player_xg_model but code snippet shows no usage of it. Has CLI entry point (argparse). Fact card says dead_candidate=true, no importers. Based on code it appears to be an older backtest variant (suggests lineup-adjusted xG comparison) superseded by backtest_unified.py
- **Improve:** If genuinely unused, remove it. If it provides a specific multi-market capability not in backtest_unified, refactor it as a callable mode within backtest_unified rather than a standalone script
- **⚠️ Missing connection:** Imports features/player_xg_model.py without using it; likely superseded by backtest_unified.py which consolidates all backtest modes
- **Verdict reason:** Dead candidate with zero importers (fact card contradiction on imported_by; the fact card is source of truth). Appears superseded by backtest_unified.py. Remove to reduce maintenance burden

#### ⚫ `scripts/analysis/backtest_player_props.py` — grade B · **delete**
- **Does:** Backtests historical anytime goalscorer player prop markets using CatBoost predictions and simulated bookmaker odds with tier-based filtering.
- **Talks to:** imports: scripts/betting/player_predictions.py (load_player_data, build_player_features, get_feature_cols, MODEL_DIR); imported_by: none
- **Quality signals:** Complete backtest with model loading, Platt scaling, tier-based filtering (Tier A: forwards 3-10% edge, Tier B: any position 3-20% edge). Code shows results (Tier A +18.6% ROI, Tier B -9.3% ROI). Has CLI entry. Fact card: dead_candidate=true, zero importers. Suggests this is an exploratory/one-time analysis for player props that was run once and not integrated into the pipeline
- **Improve:** Move insights into betting rules or player prop strategy rather than keeping standalone script
- **⚠️ Missing connection:** No integration into main pipeline
- **Verdict reason:** Dead candidate with zero importers, suggesting it was a one-shot exploration. The recommendation to only bet Tier A (forwards) is valuable insight but the code is not imported or used anywhere. If the insight is important, migrate it to betting rules or feature engineering rather than keeping the standalone script

#### 🟢 `scripts/analysis/backtest_unified.py` — grade A · keep
- **Does:** Unified backtesting framework consolidating all backtesting capabilities: walk-forward ensemble accuracy, full betting P&L simulation, real-odds value betting, ML cross-validation, with modes for accuracy/betting/value/ml-walkforward/all.
- **Talks to:** imports: config/settings.py (DATA_DIR, MODELS_DIR, SEASONS), ml/evaluation.py (_multiclass_brier, expected_calibration_error, ranked_probability_score), ml/poisson.py (poisson_win_prob), ml/walk_forward.py (lazy), storage/paths.py (features_path); imported_by: scripts/models/optimize_unified.py, scripts/models/optimize_weights.py, tests/test_integration.py
- **Quality signals:** Large, well-structured consolidation of multiple backtest modes. Lazy-loads ML classifier, walk-forward fold models, and isotonic calibrator. Implements multiple backtest modes (accuracy, betting, value, ml-walkforward, all) with clear command-line interface. Data leakage prevention (USE_ISOTONIC_CALIBRATION=False by default). Walk-forward mode support. Used by optimize scripts and integration tests. ~150 lines read shows solid architecture with proper abstraction

#### ⚫ `scripts/analysis/calibration_analysis.py` — grade B · **delete**
- **Does:** Analyzes and fixes model calibration using temperature scaling and Platt scaling across XG, ML, player XG, and deep learning predictors.
- **Talks to:** imports: config/settings.py, models/deep_learning.py, scripts/prediction/ensemble_prediction_engine.py (EnsemblePredictor, XGPredictor, MLClassifier, PlayerXGPredictor); imported_by: none
- **Quality signals:** Implements CalibrationAnalyzer class with load_data, get_model_predictions, temperature scaling, Platt scaling. Has --fix option for applying fixes. Imports multiple predictor classes. Fact card: dead_candidate=true, zero importers. Code snippet shows it loads predictors but the test set code is incomplete (line 100 cuts off). Suggests this is exploratory Phase 1.2 work that was not integrated into production
- **Improve:** Either integrate calibration fixes into the ensemble pipeline automatically or delete. If calibration remains important, codify the fixes directly into ensemble/walk_forward rather than leaving as standalone analysis
- **⚠️ Missing connection:** Imports ensemble_prediction_engine but no downstream integration; likely abandoned calibration experiment from Phase 1
- **Verdict reason:** Dead candidate with zero importers. Exploratory Phase 1 work that was not integrated. Likely superseded by production calibration in backtest_unified. Remove

#### 🟢 `scripts/analysis/clv_analysis.py` — grade A · keep
- **Does:** Analyzes Closing Line Value (CLV) to determine which bet types beat the closing line, linking CLV data with actual P&L to identify sustained edges vs luck/unluck.
- **Talks to:** imported_by: scripts/pipeline/scheduler.py; imports: none (uses DATA_DIR/BETTING_DIR constants defined inline)
- **Quality signals:** Well-designed analysis: load_clv_with_pnl merges clv_history.json with bet_journal.json, analyze_by_market computes per-market CLV + P&L, analyze_trends detects time-based patterns, analyze_by_edge_bucket and analyze_by_selection provide segmented views. Comprehensive output with generate_full_report and print_report. Used by scheduler. Pure function structure, no external dependencies beyond json/logging

#### ⚫ `scripts/analysis/data_quality_report.py` — grade B · **delete**
- **Does:** Comprehensive data quality audit: checks parquet row counts, season coverage, null percentages, team name consistency, xG cross-reference (FBref vs Understat), data freshness, and feature performance monitoring.
- **Talks to:** imports: config/settings.py (DATA_DIR, SEASONS), config/team_names.py (TEAM_NAME_MAP), features/understat_features.py, scraper/injuries.py; imported_by: none
- **Quality signals:** Thorough audit with 7 check categories (parquet files, season coverage, null %, team names, xG cross-ref, data freshness, features). Builds TEAM_NAME_VARIANTS from TEAM_NAME_MAP. Code snippet shows check_parquet_files and check_season_coverage functions with proper error handling. Fact card: dead_candidate=true, zero importers. Suggests this is a one-off diagnostic run, not integrated into pipeline
- **Improve:** If data quality monitoring is important, either integrate checks into the pipeline as pre-flight validation or run periodically via scheduler. As-is it's a standalone audit tool that rots. Consider converting to a health_check module if periodic validation is needed
- **⚠️ Missing connection:** Imports feature/understat_features and scraper/injuries but uses them only for checks, not feedback
- **Verdict reason:** Dead candidate with zero importers. One-off diagnostic tool that doesn't integrate into pipeline. If data quality checks are needed, codify them into pipeline health checks or converter. Remove to reduce clutter

#### ⚫ `scripts/analysis/feature_importance_analysis.py` — grade B · **delete**
- **Does:** Analyzes feature importance using SHAP and CatBoost, identifying noisy features and ranking feature contributions to predictions.
- **Talks to:** imports: config/settings.py (DATA_DIR, MODELS_DIR); imports optional: shap, catboost; imported_by: none
- **Quality signals:** Implements FeatureImportanceAnalyzer with load_data and load_model. Optional SHAP/CatBoost imports with fallback warnings. Loads features.parquet and universal/classifier_extended.cbm. Code snippet shows initialization and basic loading logic. Fact card: dead_candidate=true, zero importers. Phase 1.3 exploration, not integrated
- **Verdict reason:** Dead candidate with zero importers. Exploratory Phase 1.3 work. Feature importance insights are valuable but should be codified into feature selection/engineering rather than left as a standalone analysis tool. Remove

#### ⚫ `scripts/analysis/formation_analyzer.py` — grade C · **delete**
- **Does:** Provides formation-adjusted predictions 1 hour before match kickoff using real-time lineup scraping, formation detection (4-3-3, 3-5-2, etc), expected vs actual lineup comparison, and Tier 2 prediction adjustments.
- **Talks to:** imports: config/settings.py (DATA_DIR, PROJECT_ROOT), scripts/utils/parsing.py (get_cache_path); imported_by: none; optional: requests, bs4 (HAS_SCRAPING flag)
- **Quality signals:** Large script with formation pattern definitions (4-3-3, 4-4-2, 4-2-3-1, 3-5-2, etc) and impact factors. Implements caching (LINEUP_CACHE_MINUTES=15). Has LineupScraper and FormationDetector classes mentioned in docstring. Code snippet shows constants and setup. Fact card: dead_candidate=true, zero importers. Appears to be aspirational Tier 2 system that was planned but not finished (incomplete scraper?). Missing integration into main prediction pipeline
- **Improve:** Either complete and integrate into backtest_unified as a --mode formation option, or remove. As-is it's incomplete scaffolding. If formation impact is real, validate it and fold into ensemble weights
- **⚠️ Missing connection:** Defines formation system but no consumers; not integrated into prediction engine or pipeline
- **Verdict reason:** Dead candidate with zero importers. Incomplete scaffolding for a Tier 2 formation-adjusted system that was never finished or integrated. Remove incomplete code

#### ⚫ `scripts/analysis/high_confidence_analyzer.py` — grade B · **delete**
- **Does:** Analyzes high-confidence predictions strategy: evaluates accuracy ONLY on high-confidence (>65%/70%/75%) predictions to model professional betting approach of quality over quantity.
- **Talks to:** imports: config/settings.py, ml/config.py (LABEL_MAP), scripts/models/train_unified.py (time_series_split); imported_by: none
- **Quality signals:** Implements time-series split validation, trains binary classifiers (CatBoostClassifier) for multiple markets (result, home_clean_sheet, away_clean_sheet, home_scores, away_scores, btts, over_2_5, over_1_5). Has BASE_FEATURES list with ~25 features. Code snippet shows train_binary_model and BASE_FEATURES. Fact card: dead_candidate=true, zero importers. Exploratory analysis of high-confidence strategy
- **Verdict reason:** Dead candidate with zero importers. Exploratory work on high-confidence strategy that was not integrated. Insight (focus on high-confidence predictions) is valuable but should be codified into betting rules or confidence-based filtering in the main pipeline, not left as standalone analysis

#### 🟢 `scripts/analysis/live_reconciliation.py` — grade A · keep
- **Does:** Compares actual live betting results against backtest expectations to detect training-serving skew, calibration drift, and systematic biases in model probabilities.
- **Talks to:** imported_by: scripts/pipeline/scheduler.py; imports: config/settings.py (DATA_DIR)
- **Quality signals:** Comprehensive reconciliation: load_live_bets from betting/bet_journal.json, analyze_calibration (bins by probability and checks win rates), analyze_edge_performance, analyze_market_comparison, analyze_clv. Computes Brier score and calibration gaps. Functions return detailed dicts with calibration bins. Used by scheduler for ongoing monitoring. Clean separation of concerns

#### 🟢 `scripts/analysis/performance_dashboard.py` — grade A · keep
- **Does:** Generates automated performance tracking dashboard: archives predictions, verifies accuracy by confidence level, tracks betting P&L/ROI, detects feature drift, monitors xG correlation and model calibration.
- **Talks to:** imported_by: scripts/pipeline/run_full_pipeline.py; imports: config/settings.py, features/bankroll_manager.py, scripts/utils/json_utils.py (load_json_safe)
- **Quality signals:** Comprehensive dashboard with 8+ check functions: archive_predictions, check_prediction_accuracy, check_betting_performance, check_data_freshness, check_feature_drift, check_confidence_calibration, check_settled_bet_feedback, check_bankroll_health. Loads and merges multiple data sources (predictions.json, archive, results.json, betting history, bankroll). Used by main pipeline. Output to performance_dashboard.json

#### ⚫ `scripts/analysis/performance_tracker.py` — grade B · **delete**
- **Does:** Analytics and optimization for betting performance: tracks win rate by confidence level, ROI by factor combination, factor performance analysis, backtesting validation, and optimization recommendations.
- **Talks to:** imports: config/settings.py (DATA_DIR), optional pandas; imported_by: none
- **Quality signals:** Implements analysis by confidence (analyze_by_confidence), loads bet history and validation data. Builds defaultdict for confidence metrics. Code snippet shows load_bet_history and load_validation_data functions with pandas optional dependency. Fact card: dead_candidate=true, zero importers. Likely superseded by performance_dashboard.py which provides similar tracking in an integrated way
- **Verdict reason:** Dead candidate with zero importers. Likely superseded by performance_dashboard.py which provides comprehensive tracking integrated into the main pipeline. Duplicate functionality, remove

#### 🟢 `scripts/analysis/player_analyzer.py` — grade A · keep
- **Does:** Deep player-level analysis system for Serie A matches: scrapes player stats from FBref, detects key players, tracks form, quantifies injury impact, analyzes formation-player fit, and assesses individual matchups.
- **Talks to:** imported_by: scripts/pipeline/run_full_pipeline.py; imports: config/leagues.py, config/settings.py, config/team_names.py, scraper/lineup_fetcher.py (normalize_player_name), scripts/utils/parsing.py (get_cache_path)
- **Quality signals:** Large, comprehensive system with multiple data classes (PlayerStats, TeamSquad, PlayerMatchup, MatchPlayerAnalysis) and classes (PlayerDataScraper, PlayerAnalyzer). Caching system with CACHE_DIR and CACHE_DURATION_DAYS. Position importance weights. Free data sources (FBref, Understat, Transfermarkt). Multiple entry points (analyze_all_upcoming_matches, get_player_factors). Used by main pipeline. Optional requests/bs4 dependencies with HAS_SCRAPING flag

#### ⚫ `scripts/analysis/player_data_audit.py` — grade B · **delete**
- **Does:** Audits player data coverage across all seasons: reports per-season row counts in player_stats.parquet and understat_players.parquet, raw HTML file counts, registry coverage, and identifies seasons needing scraping.
- **Talks to:** imports: config/settings.py (DATA_DIR, RAW_HTML_DIR, SEASONS), storage/paths.py (parsed_path); imported_by: none
- **Quality signals:** Implements audit_player_stats_parquet and audit_understat_players functions checking row counts, unique players, season coverage. Defines PLAYER_STATS_SEASONS constraint (seasons >= 2017). Code snippet shows proper data validation with .nunique() counts and missing column handling. Fact card: dead_candidate=true, zero importers. One-shot diagnostic, not integrated
- **Verdict reason:** Dead candidate with zero importers. One-time diagnostic audit. Insights about data coverage should be integrated into data pipeline health checks if ongoing validation is needed. Remove

#### 🟢 `scripts/analysis/player_history.py` — grade A · keep
- **Does:** Builds player team history lookup tracking which teams each player has played for across seasons, used for UX context ('Ex-Roma player now at Inter'), match context analysis ('facing former team'), and potential model features.
- **Talks to:** imported_by: scripts/pipeline/telegram_bot.py, web/app.py; imports: none (uses DATA_DIR constant and pandas)
- **Quality signals:** Builds player history dict from two data sources: Understat (2014-2025 broad coverage) and Sofascore (2022+ recent). Implements detect_career_gaps to identify seasons away from Serie A. get_player_profile, find_ex_players, get_match_context utilities for querying history. Handles season normalization, duplicate detection, and sorting. Used by telegram_bot and web app for enriched context. ~150 lines shows complete, production-quality implementation

#### ⚫ `scripts/analysis/train_draw_specialist.py` — grade B · **delete**
- **Does:** Trains and validates DrawSpecialist binary classifier for draws using walk-forward cross-validation across seasons 2005-2025, evaluates calibration vs ensemble, backtests on 2023-2025 Pinnacle odds, and tests blended approaches.
- **Talks to:** imports: ml/draw_specialist.py (DrawSpecialist, compute_draw_specific_features); imported_by: none; uses: sklearn.metrics (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss)
- **Quality signals:** Implements complete walk-forward training and evaluation: load_data (features.parquet with result column creation), get_feature_columns (numeric only, excludes match metadata). Computes precision, recall, F1, AUC-ROC, Brier score per fold. Backtests on 2023-2025. Code snippet shows load_data and get_feature_columns with proper filtering. Fact card: dead_candidate=true, zero importers. Suggests this is a specialized model variant that was not integrated
- **Verdict reason:** Dead candidate with zero importers. Specialized model for draw prediction that was not integrated into the ensemble. If draw prediction is important, integrate DrawSpecialist into backtest_unified and ensemble rather than keeping as standalone. Remove

#### ⚫ `scripts/analysis/validate_player_backfill.py` — grade B · **delete**
- **Does:** Validates backfilled player_stats.parquet data quality: checks record counts (~12,000+/season), season coverage (2017-2025), key player presence, no duplicates, stat sanity (minutes <=95, xG >=0), 20 teams/season, cross-reference with matches.parquet.
- **Talks to:** imports: storage/paths.py (parsed_path); imported_by: none
- **Quality signals:** Implements 7 validation checks: check_record_counts (warns <10k rows), check_season_coverage (expects 2017-2025 + 2025-2026), check_key_players (spot-check known stars per era with STAR_PLAYERS dict), plus stat sanity and team coverage. Code snippet shows check_record_counts and check_season_coverage with per-season analysis. Fact card: dead_candidate=true, zero importers. One-time backfill validation
- **Verdict reason:** Dead candidate with zero importers. One-time validation of a historical backfill (player_stats.parquet 2017-2025). Unlikely to be reused unless data is re-backfilled. Remove

### `scripts/diagnostics/` — 12 files

#### 🟢 `scripts/diagnostics/__init__.py` — grade A · keep
- **Does:** Package marker documenting that scripts/diagnostics contains diagnostic and backtest scripts producing JSON artifacts for the simulator plan.
- **Talks to:** None — docstring only.
- **Quality signals:** Minimal but correct: docstring explains the module's purpose and artifact destination (data/diagnostics/).

#### 🔧 `scripts/diagnostics/deep_backtest_1x2.py` — grade A · keep
- **Does:** Comprehensive 1X2 backtest harness scoring walkforward models against 5 odds sources (B365 open/close, Pinnacle open/close, Market Max) with per-threshold metrics, bootstrap CIs, Kelly staking, and CLV analysis.
- **Talks to:** Imports: none. Loads CatBoost models and calibrators directly from data/models/walkforward/; reads features_serie_a and features_premier_league parquets. Entry point: main() executed at module level.
- **Quality signals:** 900 LOC, well-structured main flow with clear sections (load_walkforward_proba, deoverround, kelly_stake_fraction, bootstrap_ci, run_league). Type hints present. Proper error handling for missing models/calibrators. Uses environment variables for test variants (WALKFORWARD_SUFFIX, USE_RAW_PROBA). Outputs JSON diagnostics.

#### 🔧 `scripts/diagnostics/deep_backtest_1x2_subset.py` — grade B · keep
- **Does:** Subset breakdown of 1X2 backtest into bet-class (H/D/A), odds-band (favorites/mid/longshots), and model-probability-band slices; computes ROI per slice at edge≥7%.
- **Talks to:** Imports: none. Directly loads CatBoost models and calibrators. Reads features parquets. Module-level execution.
- **Quality signals:** 280 LOC, functional but dense — loads proba inline with minimal helper abstraction. Type hints absent. Replicates load_proba / bootstrap logic from other files (duplication risk). No docstrings for functions. Module-level for-loop processes both leagues inline without function wrapper.

#### 🔧 `scripts/diagnostics/epl_loss_pattern.py` — grade B · keep
- **Does:** Diagnostic to slice EPL 2025-26 season bets by dimensions (matchweek, prob-band, odds-band, calibration-drift) to identify which subsets drive ROI losses at -22%.
- **Talks to:** Imports: ml/feature_selection.py (exclude_odds function), scripts/models/train_walkforward.py (LEAKY_COLUMNS, META_COLUMNS). Loads CatBoost models and matches/features parquets.
- **Quality signals:** 207 LOC, focused script with clear slicing logic. Type hints for main() only. Duplicates kelly-stake and bet-settlement logic. Hardcodes 5% edge threshold. Reads matches.parquet and features_premier_league.parquet; loads fresh_2025_seed42 model variant.

#### 🔧 `scripts/diagnostics/epl_rich_features_test.py` — grade B · keep
- **Does:** Validation gate (M2) for EPL rich-features retrain: cache-busts feature caches, rebuilds features, trains fresh walkforward fold, computes holdout ROI on 2025-26, compares to -22.6% baseline to gate M3/M4 pipeline steps.
- **Talks to:** Imports: ml/feature_selection.py (exclude_odds), scripts/models/train_walkforward.py (LEAKY_COLUMNS, META_COLUMNS). Runs subprocess calls to cli.py and train_walkforward. Reads matches, features, loads CatBoost models.
- **Quality signals:** 313 LOC, structured 5-step validation harness with logging. Type hints for step functions. Subprocess execution adds operational robustness. Duplicates kelly-stake and probability-mapping logic from epl_loss_pattern. Returns gate decision code (0=pass, 2=fail).

#### 🔧 `scripts/diagnostics/epl_targeted_filter.py` — grade B · keep
- **Does:** Filter design and validation for EPL betting: scores all 4 seasons with appropriate fold models, splits discovery/validation/holdout, applies 6 candidate filters, reports ROI per filter per split.
- **Talks to:** Imports: ml/feature_selection.py (exclude_odds), scripts/models/train_walkforward.py (LEAKY_COLUMNS, META_COLUMNS). Loads CatBoost models from walkforward dirs. Reads matches_premiere_league and features_premier_league.
- **Quality signals:** 221 LOC, clear filter-design workflow with apply_filter and report helpers. Type hints absent. Duplicates kelly-stake logic. Handles missing models gracefully. Compares filter variants (v1_drop_home, v2/v3/v4/v5 variants) on holdout set.

#### 🔧 `scripts/diagnostics/paper_trade_draw_boost.py` — grade A · keep
- **Does:** Paper-trade harness for H4 draw-boost fix: compares predictions with and without post-calibration draw boost=0.30 on latest 150 SA matches using walkforward_core; reports side-by-side ROI, win rate, per-bet logs.
- **Talks to:** Imports: ml/walkforward_core (WalkForwardConfig, run_walkforward, apply_calibrator), ml/feature_selection (exclude_odds), scripts/models/train_walkforward (LEAKY_COLUMNS, META_COLUMNS). Loads matches, features_serie_a; executes walkforward pipeline.
- **Quality signals:** 389 LOC, clean harness using walkforward_core as intended design. Type hints present. Helper functions: build_1x2_target, select_features, kelly_stake, implied_prob, select_bet, settle_bet. Well-documented methodology for draw-boost comparison. Reports to data/diagnostics/ CSVs.

#### 🔧 `scripts/diagnostics/print_model_status.py` — grade A · keep
- **Does:** Single-source-of-truth CLI reporting for production model metadata: reads data/models/universal/catboost_no_odds_metadata.json and displays CV metrics, calibration, Kelly ROI, with sanity-check against 55% academic ceiling.
- **Talks to:** Reads: data/models/universal/catboost_no_odds_metadata.json. No imports. Simple standalone utility.
- **Quality signals:** 86 LOC, minimal and focused. No dependencies. Clear documentation of metrics. Graceful error handling (metadata missing). Enforces contract that metadata is source of truth vs. markdown docs.

#### 🔧 `scripts/diagnostics/subset_alpha_fresh.py` — grade B · keep
- **Does:** Deep subset analysis for Series A 1X2: discovery-validation-holdout split-test across 12 binary dimensions, 3 outcome classes, 7 continuous bins, finds effects consistent across splits, outputs data/diagnostics/subset_alpha_findings.json.
- **Talks to:** Imports: ml/feature_selection (exclude_odds), scripts/models/train_walkforward (LEAKY_COLUMNS, META_COLUMNS). Loads matches_serie_a, features_serie_a; loads walkforward fold models from data/models/walkforward/serie_a/1x2__5season_seed42.
- **Quality signals:** 452 LOC, comprehensive subset-analysis harness. Type hints for main() only. Helper functions: kelly_stake, implied_prob, select_bet_from_probs, get_actual_outcome, make_bets_df, slice_stats. Clear discovery/validation/holdout logic with verdict classification (STRONG+, weak+, STRONG-, weak-, —). Duplicates kelly-stake logic.

#### 🔧 `scripts/diagnostics/subset_alpha_fresh_epl.py` — grade B · keep
- **Does:** Deep subset analysis for Premier League 1X2: discovery-validation-holdout split-test for 12 binary + 3 outcome + 7 continuous dimensions on EPL 2022-26 seasons; identifies consistent effects, saves JSON findings.
- **Talks to:** Imports: ml/feature_selection (exclude_odds), scripts/models/train_walkforward (LEAKY_COLUMNS, META_COLUMNS). Loads matches_premier_league, features_premier_league; loads walkforward fold models from data/models/walkforward/premier_league/1x2__5season_seed42 and 1x2__fresh_2025_seed42.
- **Quality signals:** 452 LOC, identical structure to subset_alpha_fresh (code duplication). Switches league and model directory only. Type hints for main() only. Uses fresh_2025_seed42 for 2025-26 season (hybrid approach). Clear split-test verdict logic.
- **⚠️ Missing connection:** Code duplication with subset_alpha_fresh: ~380 LOC of identical logic. Consider refactoring to shared library with league parameter.

#### 🔧 `scripts/diagnostics/subset_alpha_search.py` — grade C · **delete**
- **Does:** Deep subset analysis for Series A 1X2: discovery-validation-holdout split-test (identical to subset_alpha_fresh but targeted at Serie A with different model directory).
- **Talks to:** Imports: ml/feature_selection (exclude_odds), scripts/models/train_walkforward (LEAKY_COLUMNS, META_COLUMNS). Loads matches_serie_a, features_serie_a; loads walkforward fold models from data/models/walkforward/serie_a/1x2__5season_seed42.
- **Quality signals:** 456 LOC, near-identical to subset_alpha_fresh and subset_alpha_fresh_epl. No type hints except main(). High code duplication (440+ LOC identical logic). Split-test logic is sound (discovery n≥30, validation/holdout n≥5-10). Clear verdict classification.
- **⚠️ Missing connection:** Exact duplicate of subset_alpha_fresh — no functional difference.
- **Verdict reason:** Redundant duplicate of subset_alpha_fresh. Both target Serie A 1X2 with identical split-test methodology. No distinguishing features. Keeping subset_alpha_fresh; delete subset_alpha_search to eliminate dead duplication.

#### 🔧 `scripts/diagnostics/validate_walkforward_core.py` — grade A · keep
- **Does:** Validation test for walkforward_core migration: runs new ml/walkforward_core.run_walkforward on Series A 1X2 against baseline, gates H3 refactoring by accuracy within 0.5pp.
- **Talks to:** Imports: ml/walkforward_core (WalkForwardConfig, run_walkforward), ml/feature_selection (exclude_odds), scripts/models/train_walkforward (LEAKY_COLUMNS, META_COLUMNS). Reads features_serie_a; compares against data/models/walkforward/run_summary_5season_seed43.json.
- **Quality signals:** 124 LOC, focused validation script. Type hints present. Clear helpers: build_1x2_target, select_features. Single-fold fast validation (2024-2025 only). Hardcoded baseline path and acceptance threshold (0.5pp). Proper logging. Exit codes: 0=pass, 1=error.

### `scripts/utils/` — 11 files

#### ⚫ `scripts/utils/__init__.py` — grade A · keep
- **Does:** Empty package marker for the scripts.utils module.
- **Talks to:** No imports or exports.
- **Quality signals:** File is intentionally empty (one line docstring); no code to grade. This is standard Python package initialization.

#### 🔧 `scripts/utils/alert_system.py` — grade B · keep
- **Does:** Sends email/console/file alerts for high-value betting opportunities and weekly performance summaries via SMTP.
- **Talks to:** Imports config/settings.py (DATA_DIR), features/bankroll_manager.py (get_performance_stats, load_history). No importers recorded.
- **Quality signals:** 436 lines, 12 functions, good docstrings and type hints. Proper email/SMTP logic, deduplication tracking, HTML formatting. No syntax errors visible. Minor: hardcoded SMTP defaults; min_value/confidence thresholds could be config-driven. No unit tests.

#### 🟢 `scripts/utils/data_validator.py` — grade A · keep
- **Does:** Post-pipeline data integrity checker that validates predictions, odds, probability sums, field completeness, staleness, cross-file coverage, and league consistency across JSON outputs.
- **Talks to:** Imports config/leagues.py, config/team_names.py, scripts/utils/json_utils.py. Imported by scripts/pipeline/run_full_pipeline.py.
- **Quality signals:** 331 lines, 10+ functions with clear separation of concerns (_check_* patterns). Strong docstrings on public API (validate_pipeline_output). Type hints throughout (dict[str, list], set[str]). Handles multiple JSON schemas robustly; uses safe loader. No dead code or duplicates visible.

#### 🧪 `scripts/utils/error_handling.py` — grade B · edit
- **Does:** Production error management providing data validators, API retry logic with exponential backoff, graceful degradation fallbacks, feature availability tracking, and system health checks.
- **Talks to:** Imports config/settings.py (DATA_DIR). Imported by tests/test_automation.py and tests/test_edge_cases.py.
- **Quality signals:** 559 lines, 25+ functions/classes. Good type hints, docstrings, and error hierarchy (DataValidationError, APIError, FeatureUnavailableError). Implements retry decorator, cache-or-fetch pattern, feature status tracking. Minor: hardcodes VALID_SERIE_A_TEAMS instead of importing from config/team_names.py; FeatureStatus class lacks unit tests.
- **Verdict reason:** Solid error management library currently isolated to test usage. Should be activated in main pipeline for production robustness. Hardcoded team list should source from config/team_names.py.

#### 🟢 `scripts/utils/json_utils.py` — grade A · keep
- **Does:** Lightweight JSON I/O utility providing a safe loader that returns empty dict on missing/corrupt files.
- **Talks to:** No imports. Imported by 10 modules: feedback_analyzer.py, performance_dashboard.py, parlay_generator.py, weight_optimizer.py, monitor.py, telegram_bot.py, generate_epl_supplementary.py, generate_unified_report.py, data_validator.py, web/advisor.py.
- **Quality signals:** 20 lines, single function load_json_safe(). Excellent design: handles FileNotFoundError, JSONDecodeError, OSError; supports default override via sentinel _MISSING; type hints with Any. Used widely across high-level modules (10 importers). No dead code.

#### 🟢 `scripts/utils/ledger.py` — grade A · keep
- **Does:** Persistent JSON ledger helpers for reading/writing arrays, used by betting and substitution tracking modules to maintain append-only logs.
- **Talks to:** No imports. Imported by features/substitution_features.py, scripts/betting/fair_odds_tracker.py, scripts/betting/prop_tracker.py.
- **Quality signals:** 37 lines, 2 functions. Type hints (List[dict]), docstrings. Safe fallback on missing/corrupt JSON. Handles directory creation. No dead code. Used by 3 betting/feature modules.

#### 🟢 `scripts/utils/logging_config.py` — grade A · keep
- **Does:** Centralized logging configuration with colored console output, rotating file handlers, JSON structured logging, and specialized loggers for API calls, performance metrics, and betting operations.
- **Talks to:** No imports. Imported by scripts/pipeline/run_full_pipeline.py.
- **Quality signals:** 354 lines, 6 classes plus 6 functions. Full type hints, excellent docstrings (ColoredFormatter, JSONFormatter, APICallLogger, PerformanceLogger, PipelineLogger). Creates rotating file handlers with size limits; ANSI color codes for console; structured JSON logging. Has __main__ test block. No dead code.

#### 🟢 `scripts/utils/match_timing.py` — grade A · keep
- **Does:** Match timing classification and filtering that categorizes matches into windows (far/approaching/imminent/live/completed) based on kickoff time and filters out live/completed matches from pre-match processing.
- **Talks to:** No imports. Imported by scraper/footballdata_lineups.py, scraper/lineup_fetcher.py, scraper/sofascore_lineups.py, scripts/pipeline/run_full_pipeline.py.
- **Quality signals:** 111 lines, 3 functions. Type hints (Optional[datetime], Tuple[Dict, int]). Good docstrings with threshold documentation. Handles ISO 8601 parsing safely with fallback. Uses timezone-aware datetime. No dead code or edge case gaps.

#### 🟢 `scripts/utils/parsing.py` — grade A · keep
- **Does:** Lightweight parsing utilities: extracts numeric lines from market strings (e.g., Over 2.5 becomes 2.5) and generates MD5-hashed cache file paths.
- **Talks to:** No imports. Imported by formation_analyzer.py, player_analyzer.py, live_bet_context.py, parlay_generator.py, prop_tracker.py, web/app.py.
- **Quality signals:** 36 lines, 2 functions. Type hints (Optional[float]), clear docstrings with examples. Regex-based line extraction handles +/- prefixes and decimal points. MD5 cache path is deterministic and safe. No dead code.

#### 🟢 `scripts/utils/scraper_state.py` — grade A · keep
- **Does:** Failed download tracking helper for scraper retry logic; stores and loads persistent JSON dict of failed URLs for subsequent retry attempts.
- **Talks to:** No imports. Imported by scripts/data/scrape_sofascore.py, scripts/data/backfill_match_reports.py, scripts/data/scrape_understat_matches.py.
- **Quality signals:** 26 lines, 2 functions. Type hints (dict), docstrings. Safe JSON fallback on corrupt/missing files. Handles directory creation. No dead code. Directly used by 3 data scrapers for fault tolerance.

#### 🔧 `scripts/utils/squad_maintenance.py` — grade B · keep
- **Does:** Squad data quality utilities for checking freshness, validating against Understat xG data, identifying missing/stale teams, and printing injury summaries.
- **Talks to:** Imports config/settings.py (DATA_DIR), config/team_names.py (SERIE_A_2025_26, normalize_team). No importers recorded.
- **Quality signals:** 196 lines, 4 functions. Good docstrings, minimal type hints (should add List[Dict], Dict, etc). Uses pandas for Understat data. Includes emoji console output for visual feedback. Minor: hardcoded UNDERSTAT_TEAM_MAP should be in config; error handling is basic (print statements instead of logging).

### `scripts/worldcup/` — 10 files (added 2026-06-09, feat/worldcup-2026)

#### ⚫ `scripts/worldcup/__init__.py` — grade A · keep
- **Does:** Package marker for the World Cup 2026 prediction package.
- **Talks to:** Nothing; docstring only.
- **Quality signals:** One-line docstring, standard.

#### 🟢 `scripts/worldcup/engine.py` — grade A · keep
- **Does:** International prediction engine: World-Football-Elo ratings (eloratings.net convention) over `data/worldcup/international_results.csv`, a time-decay-weighted Poisson GLM (`b0 + b_diff·(Δelo/100) + b_home + b_friendly + b_major`), AND a Dixon-Coles attack/defense MLE (`fit_dc_model`, scipy L-BFGS with analytic gradients, L2 shrinkage). **Production = 50/50 geometric ensemble of the two** (PRODUCTION_* constants, dev-selected; beat plain GLM on Brier + ECE on untouched finals). Exposes `WorldCupEngine.build()`, `lambdas()`, `one_x_two()`, `score_matrix()`, `blend_lambdas()`, and `canon_team()` (fixture-display → results-dataset name map).
- **Talks to:** Read by simulate.py, backtest.py, generate_predictions.py. No Serie A pipeline coupling.
- **Quality signals:** Fully typed, leak-free Elo features by construction (pre-match ratings), tested in tests/test_worldcup.py (Elo math, GLM monotonicity, grid invariants).

#### 🟢 `scripts/worldcup/simulate.py` — grade A · keep
- **Does:** Monte Carlo of the 2026 tournament: group stage with the NEW 2026 tiebreakers (head-to-head before overall GD, with reapplication; verified vs FIFA Regs Art. 13), best-8-thirds, **exact FIFA Annex C** third-place bracket allocation (495-combination lookup from format_spec.json, backtracking fallback), knockout rounds (ET at λ/3, penalties 50/50), per-team advancement/champion counters, plus per-knockout-match matchup/winner distributions for EVERY KO match incl. the third-place playoff (`SimResult.ko_matchup_probs`/`ko_win_probs` — feeds the /worldcup predicted bracket; `r32_matchup_probs` kept for back-compat). **Conditions on reality**: `pinned_scores` keeps played group matches at their actual score in every sim (banked points, never re-simulated), `pinned_winners` short-circuits decided knockouts (applied only when the pinned team is a simulated entrant — mismatching pins ignored, not forced).
- **Talks to:** imports engine.py; consumed by generate_predictions.py. Reads data/worldcup/fixtures.json + format_spec.json. Knockout progression parses `slot_home`/`slot_away` originals first, so the sim survives fixtures resolved by knockout.py (real names in home/away).
- **Quality signals:** Typed, seeded RNG (reproducible), 2026-rule regression tests (h2h-beats-GD case), invariant tests (champion probs sum to 1, 32 R32 entrants, per-KO-match distributions sum to 1, no crash on resolved fixtures).

#### 🟢 `scripts/worldcup/backtest.py` — grade A · keep
- **Does:** The production gate + variant selector. DEV (WC18+Euro20) compares GLM/DC/ensemble grid and selects by Brier; FINAL (untouched WC22+Euro24+Copa24 group stages — KO scores include ET) reports the selected variant vs base-rate, higher-Elo, and plain-GLM references. Writes `data/worldcup/model_metadata.json` (THE live source for WC performance numbers). `--squad-study` runs the squad-strength experiment (k-grid Elo adjustment from clubelo_history.json on the fixed production variant; verdict 2026-06-11: NULL — redundant with international Elo, see squad_strength_study.json) without touching model metadata.
- **Talks to:** imports engine.py; `--squad-study` reads data/worldcup/clubelo_history.json (scripts.worldcup.squad_history). Run via `python3 -m scripts.worldcup.backtest`.
- **Quality signals:** No hardcoded performance numbers anywhere; gate rule `skill_score > 0` enforced and recorded in metadata.

#### 🟢 `scripts/worldcup/players.py` — grade A · keep
- **Does:** Anytime-goalscorer model: decayed international goal share (2y half-life, shrinkage α selected on WC 2018 only) × backtested team λ → `1−exp(−λ·share)`, filtered to the real 2026 squads (squads.json). Backtest mode (`--backtest`) evaluates vs an equal-share baseline on 4 untouched holdout tournaments and writes player_model_metadata.json; generation mode **enforces the gate** — a failing backtest ships an empty player layer. Also computes the golden-boot expected-goals leaderboard (share × E[team tournament goals] from the sim, incl. third-place match).
- **Talks to:** imports engine.py (canon_team, elo_history, fit_goal_model, load_results). Reads international_goalscorers.csv (a SAMPLE — OG rows dropped, never used for totals), squads.json, predictions.json, simulation.json. Output consumed by web/app.py `/api/worldcup` (m.goalscorers + golden_boot).
- **Quality signals:** Typed, leak-free (shares strictly pre-tournament, α never tuned on holdouts), name normalization handles the csv's own accent inconsistency, OG-exclusion loader-tested, gate enforced at one choke point.

#### 🟢 `scripts/worldcup/sofascore_fetch.py` — grade A · keep
- **Does:** Production access to Sofascore via the `www.sofascore.com/api/v1` proxy (api.sofascore.com is TCP-dead from this network; the www proxy serves the identical API unchallenged — discovered 2026-06-10. The HTML `__NEXT_DATA__` fallback was separately broken 2026-06-01→11 by Sofascore hoisting `initialProps.*` onto `pageProps`; web/app.py parsers fixed 2026-06-11 to support both paths). `--odds` resolves fixtures→events (pair-based join, ±1-day date sanity, `sofa_canon` spelling normalizer) and writes market_odds.json (1X2 decimals + de-vigged implied). `--lineups` writes confirmed_lineups.json: starters/bench per side PLUS the missing-player list (name, type missing/doubtful, reason code→label via MISSING_REASON_MAP — same codes the Serie A lineup_predictor parses). `--results` writes sofascore_results.json: final scores of played WC matches (finished-only, joined by team id, ET-inclusive + pens winner from winnerCode) — API team event lists first, daily-schedule HTML `__NEXT_DATA__` pages when the API tier is 403-banned (shape-walk parser `_events_from_next_data`, path-independent); union-merged by event id so breaker trips never shrink the store. Feeds the same-night result-pinning chain (knockout._merged_result_lookup). **Page-tier facts (measured 2026-06-11, mid-ban)**: the tournament hub page (`WC_TOURNAMENT_PAGE`) is ISR-rendered FRESH (opener inprogress 1-0 on it while the daily-schedule page still said notstarted 75' after kickoff — day pages are stale prerenders, used only as last resort); match pages are client-rendered SHELLS whose `__NEXT_DATA__` holds i18n strings only — **no page-tier fallback is possible for `--lineups`/`--refresh-stats`** (don't re-attempt; availability degrades to caps-fallback XIs during bans, the stats parquet catches up on the first healthy API run since events/last re-serves history). Shared payload builders `_lineup_entry`/`_player_stat_rows` keep the lineups/stats schemas single-sourced. Incremental merge: re-runs only add/update.
- **Talks to:** imports engine.canon_team. Output consumed by web/app.py api_worldcup (display-only "Market check" card — never blended into model probabilities).
- **Quality signals:** Typed, polite delays + retry-once + stop-on-ban, fractional→decimal via Fraction (exact), devig sums to 1 (tested), provenance timestamps.

#### 🟢 `scripts/worldcup/refresh.py` — grade A · keep
- **Does:** The matchday loop in one command: csv re-download (120s socket timeout, tmp+replace) → odds (5h age-gated, breaker-aware) → lineups → played-stats append → availability → knockout (slot auto-fill) → generate → players → grading → combos (archive+grade tickets) → Telegram morning digest (once per UTC day, ≥07:00) → dashboard kickstart. Tournament-window guard exits instantly outside Jun 10–Jul 20. Every step fail-soft (timeout/OSError caught); state atomic + fail-soft.
- **Talks to:** subprocess-runs the sibling worldcup modules; scheduled by ~/Library/LaunchAgents/com.seriea-pipeline.wc-refresh.plist (StartInterval 7200, RunAtLoad false).
- **Quality signals:** Shakedown-tested against a dead Sofascore (breaker tripped 3 steps, chain still completed in 13 min); reviewer-hardened (atomic writes, hang timeouts, rc-checked kickstart).

#### 🟢 `scripts/worldcup/grading.py` — grade A · keep
- **Does:** Grades immutable pre-kickoff snapshots vs played results: pick hits + system/market/base Brier, like-for-like subsets. Knockouts graded on the reconstructed 90' result (shootout ⇒ draw; ET goals subtracted via goalscorers minutes; incomplete coverage ⇒ honest skip). Drops any snapshot stamped at/after kickoff.
- **Talks to:** reads predictions_archive.json + results/goalscorers/shootouts csvs; writes track_record.json; served at /api/worldcup/record; rendered as the "Record so far" section.

#### 🟢 `scripts/worldcup/knockout.py` — grade A · keep (added 2026-06-11)
- **Does:** Auto-fills fixtures.json bracket slots from REAL results (replaces the manual knockout flow): complete groups rank via simulate._rank_group_2026 (exact 2026 rules), best-8 thirds placed via the shared Annex C loader (simulate.load_third_alloc) once all 12 groups close, W##/L## slots resolve from decided feeder matches (level ET-inclusive scores fall through to shootouts.csv). Partial groups never rank; filled sides freeze (original label kept in slot_home/slot_away); atomic write only on change; Annex C mismatches surface as loud errors, never silent mis-seeds. Also exports `collect_played_results(fixtures)` — match_number → actual score + decided-knockout winner from the same results join — consumed by generate_predictions to PIN reality into the Monte Carlo and the bracket every refresh. All result lookups are MERGED: results.csv first (canonical), sofascore_results.json second (`_merged_result_lookup` / `_merged_shootout_lookup`) — same-night scores bridge martj42's publish lag for both the slot fill and the pins.
- **Talks to:** imports engine.py (canon_team, io) + simulate.py (parse_slot — extended with the 'loser' kind for L101/L102, ranking, Annex C); grading._find_result for result lookup. Run via `python3 -m scripts.worldcup.knockout` in the refresh loop BEFORE generate_predictions (KO matches appear on /worldcup automatically once filled).
- **Quality signals:** All resolvers dependency-injected, tested in tests/test_worldcup.py::TestKnockoutFill (partial-group wait, shootout winner canon→display mapping, slot freeze, Annex C pool guard, loser labels).

#### 🟢 `scripts/worldcup/combos.py` — grade A · keep (added 2026-06-11)
- **Does:** Best-combo accumulators, all three concerns in one module: `build_best_combos` (the safe/favorites/value tiers served on /worldcup — who-wins legs ONLY, value pool scans all 6 framings per match gated by ≥2pp edge + positive EV + 20% prob floor, ranked by per-leg EV), `merge_combo_archive` (pre-kickoff ticket snapshots, last-write-before-FIRST-leg-kickoff, immutable after), `build_combo_record` (grades fully-settled tickets vs 90' outcomes — hit rate vs promised rate + flat-1u ROI at archived odds).
- **Talks to:** imports engine.py (DATA_DIR, atomic/safe io) + grading.py (90' reconstruction, lazy). Imported by web/app.py `_wc_best_combos` (serve path, lazy — refresh must NOT import web.app: Flask + scheduler side effects). Run via `python3 -m scripts.worldcup.combos` in the refresh loop. Writes combos_archive.json + combo_record.json; record served under `combos` in /api/worldcup/record.
- **Quality signals:** Typed, stdlib-only on the serve path (pandas only inside the grading resolver), tested in tests/test_worldcup.py::TestBestCombos/TestComboArchive/TestComboRecord (framings scan, floors, archive freeze, pending-vs-miss, ROI math).

#### 🟢 `scripts/worldcup/availability.py` — grade A · keep
- **Does:** Team-news layer: expected XI (recent competitive starts), confirmed-XI overrides, out/doubtful lists, market-value-share λ factors (position-weighted, ALPHA=0.45, clamped 0.85-1.15 — mechanical constants; --study writes directional evidence only). Applied to MODEL λs pre-blend (market already prices news).
- **Talks to:** reads sofa parquet + squads + TM values + confirmed lineups; writes player_availability.json consumed by generate_predictions._apply_availability.

#### 🟢 `scripts/worldcup/generate_predictions.py` — grade A · keep
- **Does:** Orchestrates engine + simulator → writes data/worldcup/predictions.json (per-match λs+1X2, shape-compatible with `_build_score_range_projection`), simulation.json (10k-run probabilities + **`bracket`**: the single most-likely tournament via `build_bracket` — greedy standings from sim marginals, Annex C thirds, per-KO-match engine predictions with advance-incl-ET/pens via `_ko_match_prediction`, honest `pairing_prob` per tie, resolved-fixture override), predictions_archive.json (append-only Track-Record-shape snapshots). Hard-fails on unmapped team names.
- **Talks to:** imports engine.py + simulate.py + reads player_availability.json (lambda factors applied pre-market-blend via `_apply_availability`). Run via `python3 -m scripts.worldcup.generate_predictions [--sims N]`. Output consumed by web/app.py `/api/worldcup*`.
- **Quality signals:** Typed, append-only archive semantics, prints champion top-10 for eyeballing.

#### 🟢 `scripts/worldcup/squad_history.py` — grade A · keep
- **Does:** Historical squad club-Elo dataset for the squad-strength study: parses Wikipedia "<tournament> squads" wikitext (player → club, one {{nat fs g player}} line each — club= extracted per LINE because the templates nest {{birth date and age2}}), fetches api.clubelo.com date snapshots, matches club names (normalization → alias map → prefix-token containment, ambiguity = non-match), aggregates mean club Elo + coverage per (tournament, team). Raw fetches cached in data/worldcup/squad_history/ → re-runs offline.
- **Talks to:** imports engine.py (DATA_DIR, atomic_write_json); output clubelo_history.json consumed by backtest.py --squad-study. Run via `python3 -m scripts.worldcup.squad_history`.
- **Quality signals:** Typed, cached/idempotent fetches, join validated 5/5 tournaments vs results.csv windows (zero name mismatches), unmatched clubs reported by frequency (all genuinely non-European), parser/matcher unit-tested.

#### 🟢 `scripts/worldcup/availability.py` — grade A · keep
- **Does:** Player-availability layer (added 2026-06-11): per fixture in a 7-day horizon, builds expected XI (top-11 by starts over last 5 internationals), merges Sofascore missing/doubtful lists and confirmed lineups into per-player statuses, computes market-value-weighted absence impact (position-split attack/defense shares), and emits clamped lambda factors. `--study` replays the construction on the scraped competitive window and writes the absence-vs-goals regression (sign check for ALPHA) to availability_study.json. Writes data/worldcup/player_availability.json.
- **Talks to:** imports players.py (norm_sorted, load_sofa, load_squads, RECENT_WINDOW), sofascore_fetch.py (sofa_canon), engine.py (canon_team), simulate.py (load_fixtures). Output consumed by generate_predictions.py (lambda adjustment) and web/app.py api_worldcup (`team_news` block).
- **Quality signals:** Typed, mechanical-not-fitted constants documented in-module (ALPHA below the prior, above the rotation-diluted study estimate), all cross-source name joins through one hyphen-insensitive `akey()`, tested in tests/test_worldcup.py::TestAvailability (impact math, clamps, lineup overrides, end-to-end lambda routing).

#### 🟢 `scripts/worldcup/refresh.py` — grade A · keep
- **Does:** Matchday refresh loop in one command: results/goalscorers CSV re-download, market odds (age-gated), confirmed lineups + missing players, played-stats parquet append, availability report, predictions+sim regeneration, players regeneration, grading, dashboard kickstart. No-ops outside the tournament window.
- **Talks to:** subprocess-runs the other worldcup modules; scheduled via `~/Library/LaunchAgents/com.seriea-pipeline.wc-refresh.plist` (every 2h, RunAtLoad false per the wake-storm lesson).
- **Quality signals:** Fail-soft per step, state-file freshness gating for odds, logs to a single refresh log.

#### 🟢 `scripts/worldcup/grading.py` — grade A · keep
- **Does:** Grades archived pre-kickoff predictions against played results (model vs market vs base rate) into data/worldcup/track_record.json for the Track Record page.
- **Talks to:** reads predictions_archive.json + played results; output consumed by web/app.py `/api/worldcup/record`.
- **Quality signals:** Immutable-snapshot grading (first pre-kickoff write wins upstream).

### `scripts/` — 2 files

#### 🟢 `scripts/__init__.py` — grade A · keep
- **Does:** Module marker and docstring for the scripts package.
- **Talks to:** Imported by scripts/pipeline/run_full_pipeline.py (no specific symbol usage, just package import).
- **Quality signals:** Minimal, appropriate package marker file: single-line docstring, no code. Zero lines of logic means no bugs possible.

#### 🟢 `scripts/import_seriea_odds.py` — grade B · keep
- **Does:** Merge Serie A betting odds from external CSV files (I1_*.csv from football-data.co.uk) into the matches.parquet file by matching on (home_team, away_team, match_date). Handles CSV column name normalization for old vs new formats, normalizes team names using config.team_names.normalize_team, and reports coverage per season.
- **Talks to:** Imports from config/team_names (normalize_team function). Imported at runtime by scripts/pipeline/run_full_pipeline.py inside the main pipeline's step 2.5 (refreshing historical odds), specifically calling update_parquet_with_odds() and reading TOTAL_SA_ROWS module-level state.
- **Quality signals:** Well-documented module docstring and function docstrings. Comprehensive CSV→parquet merge logic with old-to-new column mapping (lines 50-61), robust CSV reading with error handling (lines 74-78), team name normalization (lines 98-99), date parsing with mixed format support (lines 102-105), and deduplication logic (lines 186-188). Type hints absent (no from __future__ import annotations used, though line 7 does import it—inconsistently applied). Global mutable state (TOTAL_SA_ROWS, lines 128, 231) exposed for caller convenience, which is acceptable for a pipeline utility. Some code repetition in the date normalization (line 103-105 vs 162-164) but minor. Logging is clear and informative throughout.

### `tools/` — 3 files

#### 🔧 `tools/generate_download_urls.py` — grade B · **delete**
- **Does:** Generates FBref URLs and download instructions for manual in-browser HTML downloads, supporting batch manual save workflows with per-season directory organization.
- **Talks to:** No imports/importers. Standalone CLI script with self-contained FBref URL template logic.
- **Quality signals:** Well-structured, documented, uses type hints (dict[str, str], list[str]), has docstrings on all functions. 188 LOC. However, superseded by fbref_auto_scraper.py which automates the entire workflow. No test coverage. Has multiple helper functions that are logically sound (generate_season_urls, check_existing_downloads).
- **Verdict reason:** This script was designed to help with manual Cloudflare bypass via browser (save HTML manually). The project now has fbref_auto_scraper.py which fully automates FBref scraping with Cloudflare handling via Playwright/curl_cffi. The tools/ scripts are one-time developer utilities from the initial commit (Feb 3) with zero git history or active usage. CLAUDE.md project rule: 'One-shot scripts (migrations, backfills, ablations) should be deleted after they run successfully.' This qualifies as a manual workaround now replaced by automation. No production code imports it.

#### 🔧 `tools/open_urls_in_browser.py` — grade B · **delete**
- **Does:** Opens FBref URLs in default browser tabs with configurable delays, supporting manual per-season HTML download workflows.
- **Talks to:** Imports from tools/generate_download_urls (generate_season_urls, AVAILABLE_SEASONS). Standalone CLI script, no inbound references.
- **Quality signals:** Well-documented with clear docstring and usage examples. Type-hinted function signatures. 100 LOC. Portable cross-platform code (macOS/Linux/Windows detection). However, it's a manual workflow helper that relies on generate_download_urls.py for URL generation; no test coverage; the manual save workflow is inherently fragile and is now replaced by fbref_auto_scraper.py.
- **Verdict reason:** This script automates opening URLs in a browser but still requires manual HTML saves—a partial automation that is fully superseded by fbref_auto_scraper.py. Created Feb 3, no git history, zero production imports. Fits the project rule for one-shot developer utilities that should be deleted when superseded. Keeping it creates maintenance debt and confusion about whether to use the old manual workflow or the new automated one.

#### 🔧 `tools/verify_downloaded_html.py` — grade B · **delete**
- **Does:** Validates FBref HTML files after download: checks file existence, detects Cloudflare challenge pages, verifies FBref content markers, and attempts table parsing to ensure data integrity.
- **Talks to:** No imports/importers. Standalone CLI script using only pandas, BeautifulSoup, pathlib, argparse stdlib.
- **Quality signals:** Well-structured with clear module docstring. Multiple validation functions (check_file_exists, check_not_cloudflare, check_has_fbref_content, check_can_parse_tables) each with type hints and docstrings. 214 LOC. Logical progression of checks. Human-readable output with emoji status indicators and detailed error messages. However, no test coverage; written as a manual verification step in a workflow now automated by fbref_auto_scraper.py which performs these checks inline or via dedicated healthchecks.
- **Verdict reason:** This script was created as a validation step for manual FBref downloads. The automated fbref_auto_scraper.py workflow eliminates the need for post-hoc manual HTML verification—it validates inline during download or delegates to dedicated health-check scripts. Created Feb 3, only one commit, zero production imports, zero usage in current pipeline. Fits project rule: 'One-shot scripts should be deleted after they run successfully.' Keeping it suggests the manual download workflow is still recommended, which contradicts the move to automation.

### `tests/` — 23 files

#### 🧪 `tests/__init__.py` — grade A · keep
- **Does:** Empty package marker for the tests module.
- **Talks to:** none
- **Quality signals:** Package marker; no code to evaluate. Correctly empty.

#### 🧪 `tests/conftest.py` — grade A · keep
- **Does:** Pytest configuration and shared test fixtures (features_df, sample_matches_df, sample_prediction, sample_odds) with automatic data skipping when not available.
- **Talks to:** imports config.settings.DATA_DIR; used implicitly by all test modules via pytest fixture discovery
- **Quality signals:** 142 lines, well-documented fixtures with docstrings, proper pytest.fixture decorators with scope annotations, sensible defaults for synthetic data generation, uses numpy.random.seed for reproducibility, graceful skip on missing data files.

#### 🧪 `tests/simulator/__init__.py` — grade A · keep
- **Does:** Empty package marker for the simulator test submodule.
- **Talks to:** none
- **Quality signals:** Package marker; no code to evaluate. Correctly empty.

#### 🧪 `tests/simulator/test_backtest_harness.py` — grade A · keep
- **Does:** Tests Phase 3b backtest harness: determinism, odds fallback logic, walk-forward integrity, bootstrap CI convergence, and edge threshold monotonicity using synthetic deterministic predictors.
- **Talks to:** imports models.simulator.backtests.harness, models.simulator.backtests.odds_fallback, models.simulator.backtests.roi_bootstrap, models.simulator.backtests.stake_policies; defines _OmniscientBinaryPredictor, _RandomBinaryPredictor test helpers
- **Quality signals:** 312 lines, 15 distinct test functions with clear names, uses pytest-style assertions, includes helper predictor classes for synthetic testing, covers edge cases (determinism, fallback logic, walk-forward leakage, CI convergence), well-documented module docstring explaining scope.

#### 🧪 `tests/simulator/test_european_congestion.py` — grade A · keep
- **Does:** Tests Phase 0b.5 European and Coppa Italia congestion features: recent match counting with source filtering and days-since-last computation with team boundaries.
- **Talks to:** imports features.european_congestion._count_recent, features.european_congestion._days_since_last; uses pandas test data
- **Quality signals:** 68 lines, 6 focused test functions, clear test data fixtures with _extras() helper, tests boundary conditions (exclusive cutoff date, source filtering, team isolation), uses assertions on exact values.

#### 🧪 `tests/simulator/test_first_half_splits.py` — grade A · keep
- **Does:** Tests Phase 0b.3 first-half period splitting: extraction of first-period stats, parsing match JSON with first-half ratios, and validation of period sums.
- **Talks to:** imports features.first_half_splits; defines _mini_stats test helper
- **Quality signals:** 84 lines, 5 test functions, validates JSON parsing and statistical correctness (period sum checks), includes helper for synthetic data.

#### 🧪 `tests/simulator/test_missing_players.py` — grade A · keep
- **Does:** Tests Phase 0b.4 missing player injury/suspension classification: keyword classification, status parsing, and match JSON counting.
- **Talks to:** imports features.missing_players
- **Quality signals:** 73 lines, 6 test functions, covers injury/suspension/doubtful keywords, JSON parsing on empty lineups and malformed files, uses assertions.

#### 🧪 `tests/simulator/test_phase2_rates.py` — grade A · keep
- **Does:** Tests Phase 2 event-rate estimators (corners, cards, shots): fitting, prediction, ref scaling, and market probability generation (O/U corners, correct score, etc.).
- **Talks to:** imports models.simulator.base_rates.card_rates, models.simulator.base_rates.corner_rates, models.simulator.base_rates.shot_generator, models.simulator.engine.simulator, models.simulator.markets; defines _fake_train helper
- **Quality signals:** 252 lines, 17 test functions, tests estimator fit/predict behavior, ref scaling amplification, fallback behavior, Poisson reasonableness, binomial shot-to-shot-on-target conversion, market probability validity (sums to 1 where populated).

#### 🧪 `tests/simulator/test_phase5_player_props.py` — grade A · keep
- **Does:** Tests Phase 5 player-level propositions: positional priors, player profile fitting, position-aware allocation, and player market probability generation (anytime scorer, top scorers, player shots over).
- **Talks to:** imports models.simulator.base_rates.lineup_allocator, models.simulator.base_rates.player_profiles, models.simulator.engine.simulator, models.simulator.markets; defines _fake_player_match_stats helper
- **Quality signals:** 240 lines, 16 test functions, tests positional prior distributions, profile fitting with as_of_date constraints, allocation sum-to-team-rate invariant, lineup handling (empty, zero rate), player market key correctness.

#### 🧪 `tests/simulator/test_shadow_pipeline.py` — grade A · keep
- **Does:** Tests settle_shadow_log.py: binary/multiclass outcome settlement, extreme probability handling, Brier score on missed predictions, and fixture settlement with optional data.
- **Talks to:** imports scripts.prediction.settle_shadow_log
- **Quality signals:** 156 lines, 8 test functions, covers settlement logic (correct/wrong predictions), extreme probabilities (0.0, 1.0), Brier score computation, optional field handling.

#### 🧪 `tests/simulator/test_shot_level_xg.py` — grade A · keep
- **Does:** Tests Phase 0b.1 shot-level xG aggregation: per-match team aggregation with rate calculations (big chance, close, counter, penalty, set piece) and rolling team statistics with no-leakage and team-boundary isolation.
- **Talks to:** imports features.shot_level_xg._aggregate_per_match_team, features.shot_level_xg._rolling_per_team; uses pandas test data
- **Quality signals:** 94 lines, 3 focused test functions, validates aggregation sums (shot counts, xG totals, rate calculations), verifies rolling mean excludes current match (shift(1)), ensures team boundary isolation (cross-team values don't bleed), floating-point assertions tight (1e-9).

#### 🧪 `tests/simulator/test_simulator_engine.py` — grade A · keep
- **Does:** Tests Dixon-Coles Poisson model: tau correction (zero identity, positive boost), joint PMF normalization, sampling behavior with seed control, market probability consistency (1x2, O/U, BTTS, clean sheet, exact score, handicap, double chance), and tau fitting on synthetic data.
- **Talks to:** imports models.simulator.engine.dixon_coles, models.simulator.engine.simulator; uses numpy random/scipy distributions
- **Quality signals:** 187 lines, 19 test functions, validates statistical properties (PDF sums to 1, probabilities sum to 1, expected values match lambda), tests seed determinism and match_id salting, verifies market complementarities (1x2 sums, O/U complement, clean sheet = P(0 goals)), tests tau recovery on synthetic data.

#### 🧪 `tests/simulator/test_situational_xg.py` — grade A · keep
- **Does:** Tests Phase 0b.2 situational xG shares: per-situation xG aggregation and rolling share computation with NaN handling for zero-xG windows.
- **Talks to:** imports features.situational_xg; uses pandas aggregation/groupby patterns
- **Quality signals:** 97 lines, 3 test functions, validates aggregation sums, share sums to 1, NaN returned when total xG is zero, rolling window logic.

#### 🧪 `tests/test_automation.py` — grade B · keep
- **Does:** Standalone automation verification tests: scheduler existence, log output, data freshness, health check endpoints, pipeline component importability, and cron setup.
- **Talks to:** imports config.settings, scripts.pipeline.scheduler, scripts.prediction.predict_unified, scripts.utils.error_handling, web.app; defines Results class for standalone testing
- **Quality signals:** 351 lines, 10+ test functions, uses custom Results tracker (not pytest), mixed standalone and pytest style, checks scheduler log output, health endpoints, data file freshness, component imports. Runnable as both standalone script (main()) and pytest module.

#### 🧪 `tests/test_bet_journal.py` — grade A · keep
- **Does:** Tests unified bet journal system: bet ID generation, market normalization, adding/retrieving bets, settlement, CLV tracking, statistics, reporting, and journal I/O.
- **Talks to:** imports scripts.betting.bet_journal functions; uses pytest fixtures for temp journal isolation
- **Quality signals:** 407 lines, 11 test classes covering BetIdGeneration, MarketNormalization, AddBet, GetPendingBets, SettleBet, UpdateCLV, GetSettledBets, JournalStats, Report, JournalIO, CLVComputation. Uses proper pytest fixtures (clean_journal, sample_bet, populated_journal), temp paths for isolation.

#### 🧪 `tests/test_betting_logic.py` — grade B · keep
- **Does:** Standalone deep functional tests for betting calculations: Kelly criterion, value detection, Italian market conversions (handicap, O/U), and stake sizing validation.
- **Talks to:** imports features.bankroll_manager, scripts.betting.italian_market_standards; defines Results class for standalone testing
- **Quality signals:** 396 lines, 8+ test functions, uses custom Results tracker (not pytest), validates Kelly formula correctness, value threshold logic, odds-to-implied-prob conversion, Italian market specific conversions, stake ceilings. Runnable as both standalone and pytest.

#### 🧪 `tests/test_edge_cases.py` — grade B · keep
- **Does:** Standalone edge case tests: match data validation, probability/odds validation, extreme values, missing data handling (weather, form, sentiment), and error handling for API failures.
- **Talks to:** imports features.bankroll_manager, scripts.prediction.current_form_calculator, scripts.prediction.weather_integration, scripts.utils.error_handling; defines Results class
- **Quality signals:** 440 lines, 10+ test functions, covers validation (missing teams, invalid teams, same team twice), invalid probabilities/odds, extreme values, missing data gracefully, weather API failures, Kelly edge cases. Uses custom Results tracker. Runnable as both standalone and pytest.

#### 🧪 `tests/test_ensemble_predictions.py` — grade B · keep
- **Does:** Standalone ensemble prediction system tests: xG predictor loading, Poisson conversion, ML classifier integration, ensemble combination, historical backtesting, and realistic prediction output.
- **Talks to:** imports config.settings, scripts.prediction.ensemble_prediction_engine; defines Results class
- **Quality signals:** 478 lines, 9+ test functions, validates xG model loading, Poisson probability conversion, ML classifier import, feature builder, ensemble aggregation, historical backtest improvement over factor-only, prediction format correctness. Uses custom Results tracker. Runnable as standalone and pytest.

#### 🧪 `tests/test_feature_impact.py` — grade B · keep
- **Does:** Standalone feature impact tests: weather impact on goals, referee impact on cards, form impact on predictions, stadium size, Elo difference, prediction probability sums, confidence-factor correlation, and sentiment analysis.
- **Talks to:** imports scripts.prediction.current_form_calculator, scripts.prediction.referee_integration, scripts.prediction.sentiment_analyzer, scripts.prediction.weather_integration; defines Results class
- **Quality signals:** 474 lines, 10+ test functions, validates weather/referee/form/stadium/Elo feature contributions, probability sum-to-one constraint, confidence-factor correlation, sentiment analyzer output. Uses custom Results tracker. Runnable as standalone and pytest.

#### 🧪 `tests/test_historical_accuracy.py` — grade B · keep
- **Does:** Standalone historical backtesting validation: overall accuracy, accuracy by confidence level, ROI at different thresholds, home/away/draw accuracy, high-confidence performance, calibration, recent performance, and value betting simulation.
- **Talks to:** imports (dynamically loaded); loads CV predictions from model training; defines Results class with metrics tracking
- **Quality signals:** 463 lines, 9+ test functions, metrics tracking with percentage/ROI formatting, compares to random baseline (50% for 1x2), validates calibration (predicted probs vs observed freq), recent performance subset analysis, value betting ROI simulation. Uses custom Results class with metrics aggregation. Runnable as standalone.

#### 🧪 `tests/test_integration.py` — grade A · keep
- **Does:** Comprehensive integration tests: unified module imports, feature pipeline, training, prediction, betting, backtest, end-to-end flow, cross-module consistency, and mathematical correctness of Poisson/xG/Kelly calculations.
- **Talks to:** imports config.settings, features.build, ml.config, ml.poisson, scripts.analysis.backtest_unified, scripts.betting.betting_unified, scripts.models.optimize_unified, scripts.models.train_unified, scripts.prediction.predict_unified, storage.paths; defines 10+ TestClass objects
- **Quality signals:** 1484 lines (largest test file), 10 test class suites (TestUnifiedImports, TestFeaturePipeline, TestTrainingPipeline, TestPredictionPipeline, TestBettingPipeline, TestBacktestPipeline, TestEndToEnd, TestOptimizePipeline, TestCrossModuleConsistency, TestMathematicalCorrectness). Uses pytest-style assertions, tests module imports, feature build, training pipeline integrity, prediction format, betting logic, backtest harness, end-to-end flow from features to bets.

#### 🧪 `tests/test_monte_carlo.py` — grade A · keep
- **Does:** Tests Monte Carlo improvements: bivariate Poisson (lambda_3, independence components, clamping), score matrix generation, sampling with seed control, adaptive simulation counts, market outcome derivation, leg-to-outcome key mapping, SGP/cross-match logic, and ruin simulation with optimal fraction finding.
- **Talks to:** imports scripts.betting.extended_markets (BivariatePoissonParams, sampling, calibration), scripts.betting.parlay_generator (adaptive_sim_count, monte carlo bands, leg-to-outcome), scripts.betting.bankroll_simulator (BetOpportunity, RuinSimConfig); uses scipy.stats.poisson
- **Quality signals:** 556 lines, 14 test classes with 30+ methods. Tests BivariatePoissonParams (lambda_3 computation/clamping, independence components), score matrix generation, Poisson correctness, sampling with multiple seeds, cross-match survival, bankroll ruin simulation with optimal Kelly fraction. Uses pytest.approx for floating-point comparisons, clear test class organization.

#### 🧪 `tests/test_pipeline.py` — grade B · keep
- **Does:** Standalone basic pipeline tests: module imports, API key hardcoding check, Italian market standards, prediction engine, web routes, and data file existence.
- **Talks to:** imports config.settings, scripts.betting.italian_market_standards, scripts.prediction.predict_unified, web.app; defines Results class
- **Quality signals:** 263 lines, 7+ test functions, validates 15+ module imports (config, odds_fetcher, prediction_engine, sentiment_analyzer, player_analyzer, formation_analyzer, model modules, betting_engine, italian_market_standards, weather, referee, live_betting, AI reasoning, web app), checks for hardcoded API keys, validates Italian market conversions, web route setup, parquet file existence. Uses custom Results tracker. Runnable as standalone and pytest.
