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
- **Throttling ≠ ban.** Rapid successive requests return `CurlError (7) Failed to connect
  ... port 443` — a *connection* error, not a 403. Back off ~20s; it recovers. Don't read it
  as the ban returning.
- **What you'll see** (a real ban): `api.sofascore.com/api/v1/...` returns 403 across all curl-cffi profiles, all domain variants, all timing.
- **Why**: Cloudflare IP-fingerprint ban, often after heavy scraping. Lasts hours to days.
- **Fix**: `www.sofascore.com/tournament/...` HTML pages return 200. Parse the embedded `<script id="__NEXT_DATA__">...</script>` JSON blob. Standings + match incidents + venue + referee + stoppage time + attendance live directly under `props.pageProps` (since ~2026-06; previously nested in `props.pageProps.initialProps` — the web/app.py parsers support both paths).
- **Page-tier map (measured 2026-06-11, mid-ban)**: NOT all www pages are equal. **Tournament hub pages** are ISR-rendered FRESH (live scores within minutes — use these; WC: `scripts/worldcup/sofascore_fetch.WC_TOURNAMENT_PAGE`). **Daily-schedule pages** (`/football/{date}`) are stale prerenders (opener showed `notstarted` 75 min after kickoff) — last resort only. **Match pages** are client-rendered shells: `__NEXT_DATA__` carries i18n strings only, NO event/lineups/statistics payloads — an HTML fallback for lineups or player stats is IMPOSSIBLE; during bans, lineups degrade to caps-fallback XIs and the stats parquet catches up on the first healthy API run (`events/last` re-serves history).
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

**Known residue (2026-08-25)**: the fix restores EPL flow *going forward* only. Matches already present in the EPL stat parquets are invisible to detection (`0 new`), so the existing gap — 2026-2027 = 0/10, 2025-2026 = 309/380 — needs a **backfill keyed on `matches.parquet`**, not on the stat parquets. `results.json` is also SA-only (0 EPL completed), so `_fallback_ingest_from_results` can never cover EPL.

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

