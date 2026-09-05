# Serie A Pipeline — DATA CATALOG

*Generated: 2026-04-21, after full refresh + Sofascore fallback integration*
*Updated: 2026-04-25 — full EPL match-report backfill (87,326 player-match rows, 9 seasons), lineups.parquet now covers both leagues (266k rows), understat_players refreshed for both leagues*

**Purpose:** Canonical AI-readable reference for every data file in the pipeline — what it contains, where it comes from, how it's fetched, what the failure modes are, and which downstream files depend on it.

**Scope:** Serie A focus. EPL data is now backfilled at per-match granularity for 8 of 9 seasons (2017-18 through 2025-26, excluding partial 2019-20 due to FBref pandemic gap).

---

## TABLE OF CONTENTS

1. [Data flow architecture](#data-flow-architecture)
2. [Dataset inventory (quick reference)](#dataset-inventory-quick-reference)
3. [Ground truth — matches.parquet](#1-ground-truth--matchesparquet)
4. [Features — features_serie_a.parquet](#2-features--features_serie_aparquet)
5. [FBref parsed (5 files)](#3-fbref-parsed-5-files)
6. [Sofascore (6 files)](#4-sofascore-6-files)
7. [Understat](#5-understat)
8. [Weather / Referees / External context](#6-weather-referees-external-context)
8b. [World Cup 2026 — data/worldcup/](#6b-world-cup-2026--dataworldcup)
9. [Cross-source mapping — match_id_mapping.parquet](#7-cross-source-mapping--match_id_mappingparquet)
10. [Auto-refresh infrastructure](#8-auto-refresh-infrastructure)
11. [Fallback matrix (Plan A / B / C)](#9-fallback-matrix-plan-a--b--c)
12. [What's broken or partial](#10-whats-broken-or-partial)
13. [**Column glossary — what each feature family means**](#11-column-glossary--what-each-feature-family-means)
14. [**Join recipes — how to link data across sources**](#12-join-recipes--how-to-link-data-across-sources)
15. [**Feature provenance — which pipeline step writes what**](#13-feature-provenance--which-pipeline-step-writes-what)
16. [Per-file deep column audit](#per-file-deep-column-audit) (the rest of this document)

---

## Data flow architecture

```
╔════════════════════════════════════════════════════════════════════════════════╗
║                            DATA FLOW (Serie A)                                 ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║  ┌─── Scrapers (no API cost) ───┐  ┌─── Paid / limited APIs ───┐             ║
║  │  FBref (botasaurus)           │  │  The Odds API (100k/mo)  │             ║
║  │  Sofascore (aiohttp)          │  │  football-data.org (free) │             ║
║  │  Understat (selenium+req)     │  │  API-Football (100/day)   │             ║
║  │  Transfermarkt                │  │  Open-Meteo (unlimited)   │             ║
║  │  Whoscored (refs)             │  └────────────────────────────┘             ║
║  └────────────┬──────────────────┘              │                              ║
║               │                                 │                              ║
║               ▼                                 ▼                              ║
║  ┌──────────────────────────┐  ┌──────────────────────────┐                   ║
║  │  data/external/*         │  │  data/upcoming/odds_*    │                   ║
║  │  (sofascore, understat,  │  │  (live odds every 2h)    │                   ║
║  │   weather, referee,      │  └────────────┬─────────────┘                   ║
║  │   transfermarkt)         │               │                                 ║
║  └─────┬────────────────────┘               │                                 ║
║        │                                    │                                 ║
║        ▼                                    ▼                                 ║
║  ┌──────────────────────────────────────────────────────────┐                 ║
║  │  data/parsed/matches.parquet  ← FOUNDATION               │                 ║
║  │    (scores, basic stats, odds; 7,930 SA rows, 21 seasons)│                 ║
║  └─────────────────────────┬────────────────────────────────┘                 ║
║                            │                                                   ║
║  ┌─────────────────────────▼────────────────────────────────┐                 ║
║  │  features/build.py — 38-step feature pipeline            │                 ║
║  │  consumes: matches.parquet + ALL external + parsed files │                 ║
║  │                                                           │                 ║
║  │  outputs: features_serie_a.parquet (1,059 cols)          │                 ║
║  └─────────────────────────┬────────────────────────────────┘                 ║
║                            │                                                   ║
║                            ▼                                                   ║
║  ┌──────────────────────────────────────────────────────────┐                 ║
║  │  scripts/prediction/ensemble_prediction_engine.py        │                 ║
║  │  consumes: features + trained models                     │                 ║
║  │  outputs: data/upcoming/predictions.json                 │                 ║
║  └─────────────────────────┬────────────────────────────────┘                 ║
║                            │                                                   ║
║                            ▼                                                   ║
║  ┌──────────────────────────────────────────────────────────┐                 ║
║  │  scripts/betting/betting_unified.py                      │                 ║
║  │  picks bets, writes:                                     │                 ║
║  │    data/betting/history.json (settled)                   │                 ║
║  │    data/upcoming/bet_history.json (pending)              │                 ║
║  │    data/bankroll.json                                    │                 ║
║  └──────────────────────────────────────────────────────────┘                 ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
```

---

## Dataset inventory (quick reference)

**Status key:** ✅ = fresh + complete | ⚠ = has gaps | ❌ = stale / broken

| # | File | Rows | 2025-26 | Status | Refreshed via | Last update |
|---|------|------|---------|--------|---------------|-------------|
| 1 | `data/parsed/matches.parquet` | 15,980 (7,990 SA + 7,990 EPL) | **380/380 SA + 380/380 EPL** (EPL 80-match gap backfilled 2026-08-27 via `matchday_updater --backfill`; 2026-27: 10/10 both leagues) | ✅ | Daily morning/evening pipeline + `--backfill` for recovery | 2026-08-27 |
| 2 | `data/features/features_serie_a.parquet` | 7,980 | **380/380 canonical** (0 numeric, verified 2026-07-17 — see callout) | ✅ (was ⚠ until FBref backfill) | Daily `features/build.py` (24h gate) | 2026-07-17 |
| 3 | `data/parsed/player_stats.parquet` | **103,111** | **380/380** | ✅ | FBref HTMLs (`fbref_match`) | 2026-07-16 |
| 4 | `data/parsed/lineups.parquet` | **289,305** (SA + EPL) | 380/380 union | ⚠ dual-keyed — see callout | FBref HTMLs (`parse_all_lineups --include-epl`) + Sofascore fallback | 2026-07-17 |
| 5 | `data/parsed/events.parquet` | **12,095** | 373/380 union | ⚠ dual-keyed — see callout | FBref scorebox + Sofascore incidents | 2026-07-17 |
| 6 | `data/parsed/goalkeeper_stats.parquet` | **6,897** | **380/380** | ✅ | FBref HTMLs (no fallback yet) | 2026-07-16 |
| 7 | `data/parsed/shots.parquet` | 9,213 | **0** | ❌ FBref stopped serving `shots_all` | **Sofascore shotmap does NOT replace it** — see §`shots.parquet` | 64 days ago |
| 8 | `data/parsed/match_id_mapping.parquet` | 15,889 | **380/380** hash + sofascore + understat | ✅ rebuilt 2026-07-17 | `build_match_id_mapping.py` | 2026-07-17 |

> ⚠️ **"330" was never the season — a Serie A season is 380.** Until 2026-07-16 this
> table read `330/330 ✅` in several rows, which looks like *complete* and was not:
> 330 is how many matches the **frozen `fixtures.html`** had published back on
> 2026-04-21, and `_refresh_fbref_fixtures.py` had been failing silently since (it
> fetched headless, which Cloudflare Turnstile refuses, and reported it as a
> `TypeError` about the write). Every count derived from that page inherited the
> cap, and a stale *input* shrinks a coverage figure instead of failing it — so
> nothing looked wrong. Fixed 2026-07-16; the 119 absent FBref reports were fetched
> (380/380 on disk) and re-parsed. `matches.parquet` itself was always complete at
> 380 — the gap was in the FBref-derived parquets and in this table. **If you see a
> `/330` anywhere, it is a stale number, not a target.**

> ✅ **GROUND TRUTH `match_id` is now one format (fixed 2026-07-17, durable).**
> `matches.parquet` held 94 rows keyed on Sofascore's numeric fixture id
> (`13981687`) beside 15,795 canonical `{date}_{home}_{away}` ones — two
> incompatible formats in the ground-truth id column, the thing everything joins
> **to**. Now 0 numeric, 380/380 canonical for SA 2025-26, and it stays that way:
> the real defect was a live writer, `matchday_updater.py:454`, which did
> `match_id = str(fixture.get("id"))` every matchday. It now mints the canonical
> key (dedup was always on `(home,away,date,season)`, so the id was never
> load-bearing), pinned by `tests/test_matchday_updater.py`. `match_id_mapping`
> is likewise clean (380/380). The formula was proven by reconstructing the 286
> known-good ids byte-for-byte before writing. `data/external/sofascore/*` was
> deliberately **left alone** — natively Sofascore-keyed, so those keys are
> correct; renaming them would be the actual bug.
>
> ✅ **The DERIVED-layer re-mint is RESOLVED by the FBref backfill (verified
> 2026-07-17).** It used to come back numeric for these 94 after each
> `features/build.py` run **because** until the backfill those 94 late-season
> matches had **only** a Sofascore source, so multiple feature writers re-derived
> a numeric key for them independently. The backfill gave all 380 matches a
> canonical-keyed FBref source, removing the root cause. Verified 0 numeric across
> the whole derived layer, N>1 under fixed conditions (different code paths, built
> at different times today): `features_serie_a.parquet` (0/380, built 14:19), all
> **54** `data/cache/features/serie_a/*` caches (0 total), and
> `data/external/weather.parquet` (0, built 10:36). **Residual risk (a guard, not
> a bug):** a *new* matchday match that arrives Sofascore-only — before that
> week's FBref report lands — could transiently re-mint a numeric key until the
> weekly backfill catches it. This matters because the shot-level features join
> **canonical-only** (see `all_shots_with_xg` §, and `_map_to_canonical` in
> `features/shot_level_xg.py`): a numeric-keyed feature row silently receives
> **zero** shot columns, not partial NaN. **This is now guarded (2026-07-17):**
> `health_check.py:check_data_quality` emits `feature_id_keying` — for each
> ACTIVE_LEAGUES `features_{league}.parquet` it flags any current-season
> `match_id` absent from `matches.parquet` (classified by membership, not id
> shape). >1 matchweek mis-keyed = **CRITICAL** (systemic revert); a handful =
> WARNING (the transient new-match window above). Runs every 30 min via
> `monitor.py`, so a re-mint fails loud instead of silently emptying the shot
> features; pinned by `tests/test_feature_id_keying.py`. Note `match_id` is in
> `build.py:get_ml_feature_columns`'s EXCLUDE set, so a transient numeric id does
> **not** affect model training/prediction — only the feature/prediction ↔
> `matches.parquet` join (and, per the coupling above, the shot features) for the
> affected rows. Do **not** "fix" a transient re-mint by rewriting the parquet.
>
> ⚠️ Note the id **shape trap**: an 8-char Sofascore id and an 8-hex FBref hash
> are indistinguishable (`13980098` is valid hex; `02493616` is an FBref hash
> that `.isdigit()` calls numeric). **Classify ids via `match_id_mapping.parquet`,
> never by shape** — shape-guessing produced four wrong readings in one session.

> ⚠️ **`lineups` and `events` are DUAL-KEYED — the same match appears under two
> keys** (found 2026-07-17, **not fixed**). 370 lineups / 305 events matches exist
> both under the canonical id and under their FBref hash. For events the two sets
> are **complementary, not duplicates**: canonical-keyed rows carry `yellow_card`
> (1,264) and `second_yellow` (17) but **no** own-goals; hash-keyed rows carry
> `own_goal` (22) but **no** yellows — the same 79′ dismissal is `second_yellow`
> under one key and `red_card` under the other. **Dropping either side destroys
> real data.** A correct fix is a union-merge with dedup, complicated by name
> spellings differing across sources (`Ismael Koné` vs `Ismaël Koné`), so it needs
> its own decision rather than a rename. Until then: a consumer joining on the
> canonical id gets yellows but no own-goals and 330/380 coverage; joining on the
> hash gets the reverse. Union coverage is 380/380 (lineups) and 373/380 (events).
| 9 | `data/external/sofascore/player_match_stats.parquet` | 101,875 | 330/330 | ✅ | `scrape_sofascore.py` weekly | 6h ago |
| 10 | `data/external/sofascore/match_team_stats.parquet` | 8,790 | 330/330 | ✅ | `scrape_sofascore.py` weekly | 6h ago |
| 11 | `data/external/sofascore/shotmap_stats.parquet` | 6,684 | **378/380** (2 rows/match — per-**team aggregate**, not shot-level) | ✅ | `scrape_sofascore.py` weekly | 6h ago |
| 12 | `data/external/sofascore/all_shots_with_xg.parquet` | 86,628 | **380/380** | ✅ shot-level (per-shot xg/xgot/coords) | `write_shot_level_xg.py` (weekly Step 4b) | 2026-07-17 |
| 13 | `data/external/sofascore/match_incidents.parquet` | 90,041 (6,770 matches, both leagues) | — | ✅ | `scrape_sofascore.py` weekly | 2026-09-05 |
| 13b | `data/parsed/goal_timeline.parquet` (+ `goal_timeline_universe.parquet`) | 18,809 goals / 6,767 matches (SA 3,388 · EPL 3,379) | — | ✅ derived | `scripts/models/goal_process.py --build-timeline` (source-mtime watermark; rebuilt on demand by `build_goal_timeline()`) | 2026-09-05 |
| 14 | `data/external/sofascore/captains.parquet` | 6,650 | — | ✅ | `scrape_sofascore.py` weekly | 23h ago |
| 15 | `data/external/understat/matches_xg.parquet` | 3,370 | 330/330 | ✅ | `scrape_understat_xg()` + `parse_all_understat` | 5h ago |
| 16 | `data/external/weather.parquet` | 11,433 | 330/330 | ✅ | `scraper/weather.py` weekly (Open-Meteo) | 4h ago |
| 17 | `data/external/referee/referee_assignments.parquet` | 3,368 | 328/330 | ✅ | `scraper/referee.py` weekly (whoscored) | 6h ago |
| 18 | `data/parsed/player_stats_epl.parquet` | 87,326 | 309/309 EPL matches | ✅ | `scrape_epl_match_reports.py --season 2025-2026` | 2026-04-25 |
| 19 | `data/parsed/understat_players.parquet` | 11,669 | 1,086 (SA + EPL) | ✅ | `scripts.data.refresh_understat_players` (manual; no scheduler yet) | 2026-04-24 |
| 20 | `data/parsed/missing_players.parquet` | **6,776** (SA 3,389 + EPL 3,387) | **380/380** SA, 10/10 SA 2026-27 | ✅ self-refreshing | derived from `data/external/sofascore/matches{,_premier_league}/*/*.json` by `features/missing_players.py` | 2026-08-25 |
| 21 | `data/parsed/first_half_splits.parquet` | **6,775** (SA 3,388 + EPL 3,387) | 380/380 SA, 10/10 SA 2026-27 | ✅ self-refreshing | derived from `data/external/sofascore/matches{,_premier_league}/*/*.json` by `features/first_half_splits.py` (1ST-period `team_stats`) | 2026-08-25 |

### 2025-26 feature completeness (what the ML model sees)

| Column | Fill rate 2025-26 | Quality |
|--------|------|---------|
| home_elo, elo_diff | **100%** | ✅ |
| poisson_home_xg, poisson_away_xg | **100%** | ✅ |
| home_league_pos, away_league_pos | **100%** | ✅ |
| home_shots_total, home_corners, home_fouls | **100%** | ✅ (was 79% before today's fix) |
| home/away yellow_cards, red_cards | **100%** | ✅ (was 98% before fix) |
| ref_matches_officiated | **99%** | ✅ |
| home_us_team_xg (Understat) | **85%** | ⚠ |
| weather_temperature_2m_mean | **75%** | ⚠ (Open-Meteo timeouts on some matches) |
| odds_B365H | **79%** | ⚠ (70 post-Feb matches missing B365 CSV) |

### Historical coverage (ALL Serie A, all seasons)

| Column | Fill rate |
|--------|-----------|
| Result (H/D/A) | **100%** |
| Shot data (home/away_shots_total) | **100%** (after today's Sofascore backfill) |
| B365 odds | **99%** |
| Pinnacle closing odds | **65%** (only 2011+ seasons had PS data) |
| Referee | **52%** (pre-2017 matches lack external ref sources) |

---

## 1. Ground truth — matches.parquet

**Role:** The foundation table. Every match ever played, with basic stats and odds. Every other file joins back to this via `match_id`.

**Path:** `data/parsed/matches.parquet`
**Rows:** 15,839 (7,930 Serie A, 7,909 Premier League)
**Columns:** 109
**Date range:** 2005-08-13 → 2026-04-20
**Seasons:** 21

### Critical columns

| Column | Description | Fill rate (SA) | Source |
|--------|-------------|----------------|--------|
| `match_id` | Primary key (`YYYY-MM-DD_Home_Away`) | 100% | derived |
| `home_team`, `away_team` | Team names (normalized) | 100% | FBref + Sofascore |
| `match_date` | Match date | 100% | FBref + Sofascore |
| `home_score`, `away_score` | Final score | 100% | FBref + Sofascore |
| `result` | H/D/A | 100% (64 NaN fixed today) | derived |
| `season` | e.g. '2025-2026' | 100% | derived |
| `league` | 'serie_a' | 100% | derived |
| `home_shots_total`, `away_shots_total` | Basic match stats | 100% (70 backfilled today) | FBref CSV + Sofascore |
| `home_corners`, `away_corners` | | 100% (70 backfilled) | same |
| `home_yellow_cards`, `away_yellow_cards` | | 100% (6 backfilled) | same |
| `home_red_cards`, `away_red_cards` | | 100% (6 backfilled) | same |
| `referee` | Ref name | 52% all-time, 100% 2025-26 | whoscored + FBref scorebox |
| `odds_B365H/D/A` | Bet365 1X2 | 99% | football-data.co.uk CSV |
| `odds_AvgH/D/A` | Bookmaker average | 99% | football-data.co.uk |
| `odds_PSH/D/A` | Pinnacle opening | 65% (2011+ only) | football-data.co.uk |
| `odds_PS_close_H/D/A` | Pinnacle closing | 65% | Odds API historical backfill |
| `odds_AH_line` | Asian handicap line | 28% | limited |
| `matchweek` | Matchweek number | 99% | FBref |

### How it's fetched

1. **Daily (morning/evening pipelines):** `matchday_updater.py` fetches new match results from Sofascore API, appends rows
2. **Odds backfill:** `backfill_historical_odds.py` pulls football-data.co.uk CSV (weekly drop, sometimes stale)
3. **Historical parse:** `parser/match_page.py` reads FBref match reports for pre-existing data

### Failure modes + fallbacks
- **Primary:** Sofascore API daily → if it fails, falls back to...
- **Plan B:** football-data.org API (basic home/away/score) → if fails...
- **Plan C:** FBref fixtures scraper
- 3-level failover → very robust

---

## 2. Features — features_serie_a.parquet

**Role:** The 1,059-column merged table the ML model consumes. Rebuilt by `features/build.py` (38-step plugin pipeline) from matches + all external sources.

**Path:** `data/features/features_serie_a.parquet`
**Rows:** 7,930 (matches matched 1:1)
**Columns:** 1,059
**Date range:** 2005-08-27 → 2026-04-20

### The 38-step feature pipeline

Steps in `features/build.py::_create_pipeline()`:

| Step | Purpose | Input data |
|------|---------|------------|
| 01 | Team match log | matches.parquet |
| 02 | Rolling stats (5/10 game) | team log |
| 03 | Home/away splits | team log |
| 04 | xG trends | team log + player_stats |
| 05 | Strength ratings | matches |
| 06 | Rest days | matches |
| 07 | Momentum streaks | team log |
| 08-09 | Derived team features + pivot | team log |
| 10 | H2H | matches (leakage-safe via date cutoff) |
| 11 | Elo ratings | matches (time-ordered update) |
| 12 | Player impact | player_stats + lineups |
| 13 | Referee | referee_assignments + matches.referee |
| 14 | Team aggregates | team log |
| 15-16 | GK quality + shot quality | goalkeeper_stats + shots |
| 17-18 | Advanced player / shots | player_stats + shots |
| 19 | Understat xG | understat/matches_xg |
| 19b-c | Sofascore, player depth | sofascore/ |
| 20 | FBref season stats | fbref_stats_* parquets |
| 21 | League position | matches |
| 22 | Manager | matches |
| 23 | Congestion | matches (rest days + fixture density) |
| 24 | Suspensions | lineups (yellow card accumulation) |
| 25 | Formations | lineups |
| 26-27 | Derived match-level | composite |
| 28-29 | Venue + weather | weather.parquet |
| 30-31 | Odds + market data | matches.parquet odds cols |
| 32-38 | Injuries, PPDA, transfer, interactions, contextual | various |

### Refresh
- **Daily:** Morning pipeline rebuilds if >24h stale (gate in `run_full_pipeline.py`)
- **Staleness gate lowered from 72h→24h** during today's fixes
- Runtime: ~12 minutes

### Failure modes
- **No last-known-good fallback currently.** If `build_features()` crashes, the old parquet is overwritten anyway. Gap identified but not yet fixed.

---

## 3. FBref parsed (5 files)

All 5 parsed from HTML match reports in `data/raw/html/{season}/{fbref_hash}.html`. **Refresher:** `scripts/pipeline/refresh_weekly_data.py` weekly Monday 04:00.

### `data/parsed/player_stats.parquet`
- **Per-player per-match stats** (141 cols: goals, assists, shots, SOT, cards, fouls, tackles, etc.)
- **100,441 rows, 9 seasons**
- **Writer:** `scripts/data/parse_all_player_stats.py` (walks HTML, runs `parser/player_stats.py`)
- **Fallback (NEW today):** `scripts/data/fallback_sofascore_to_fbref.py` adds Sofascore `player_match_stats` rows when FBref HTMLs missing (+2,020 rows added today)
- **Keying (FIXED 2026-07-17):** `match_id` is the canonical `{date}_{home}_{away}` from each report's own date+teams, so it joins matches.parquet regardless of how the FBref report is named on disk. Previously the writer keyed by `html_path.stem`; FBref 2025-26 reports are saved as `{hash}.html`, so the entire current season was **hash-keyed and invisible to every canonical join** (player_impact / team_aggregates / advanced_player) — key-player tracking silently froze at the prior season, and promoted teams (no history) went null for all 38 matches. The 2026-07-17 migration re-keyed the 380 existing 2025-26 matches hash→canonical (pure relabel, content unchanged); the writer fix prevents recurrence. Verified: player_impact 2025-26 fill 80.0% → 99.7%.

### `data/parsed/lineups.parquet`
- **Starting XI + formation + subs**
- **266,386 rows (SA + EPL, both leagues all seasons)** — columns: match_id, team, is_home, formation, player_name, shirt_number, role, season
- **Writer:** `scripts/data/parse_all_lineups.py` — flags: `--season` (**required**), `--append`, `--replace-all`, `--dry-run`
- **Keying (FIXED 2026-07-17):** the FBref parser now builds the canonical `{date}_{home}_{away}` match_id from the report's own date+teams, not `html_path.stem`. Before the fix, SA 2025-26 was broken **two ways at once**: the FBref rows (correct team names) were hash-keyed, while a Sofascore lineup path wrote a **canonical-keyed copy with `team=''`** (empty — formation only). `features/player_impact.py` filters lineups by team on the canonical id, so it saw only the empty-team rows → `home/away_key_players_available`, `top_scorer_played`, `squad_rotation` were 0/null for the whole current season. The migration dropped the empty-team canonical SA 2025-26 rows and re-keyed the correct FBref copy hash→canonical (verified: player_impact starter∩player_stats overlap 11/11). `formation` is unaffected (present in both copies).
- ⚠️ **`--include-epl` / `--epl-only` no longer exist** (noted 2026-07-16). The original module was one of the 15 phantoms (never git-added, swept after 2026-06-01) and the rebuild reconstructs only the Serie A path that `refresh_weekly_data.py` actually invokes. The EPL rows already in this parquet are **preserved** — the merge replaces only the match_ids it parsed (verified: 271,530 rows across 6,427 other-convention matches survive a 2025-26 run). What is gone is the ability to *refresh* EPL lineups from `{season}_epl/` HTML; that path is unbuilt, and calling the old flag now errors loudly rather than silently doing nothing. See AUGUST_RUNBOOK §3b.
- **Fallback:** Sofascore JSON dumps in `data/external/sofascore/matches/{season}/*.json` have `home_lineup`/`away_lineup` objects
- **2025-26:** 569 matches (260 SA + 309 EPL), 40 teams, formation 100% populated for both leagues
- **Append safety:** `--append` keys on `(season, team)` so partial-league input does NOT wipe other-league rows for the same season (fixed 2026-04-25 after data-loss incident)

### `data/parsed/player_stats_epl.parquet`
- **EPL per-match player stats** (30 cols: shirtnumber, position, minutes, goals, assists, shots, cards, fouls, tackles, etc.)
- **87,326 rows across 9 seasons** — analogous schema to Serie A `player_stats.parquet`, distinct file because the merge logic and team aliases differ
- **Writer:** `scripts/data/scrape_epl_match_reports.py [--season YYYY-YYYY]` (downloads FBref match HTMLs to `data/raw/html/{season}_epl/`, then parses)
- **Per-season coverage:** 2017-18=380, 2018-19=380, 2019-20=68 (FBref pandemic gap, not fixable), 2020-21=380, 2021-22=380, 2022-23=380, 2023-24=380, 2024-25=380, 2025-26=309 (current)
- **Type safety:** Numeric columns (goals, assists, minutes, shots, cards, etc.) coerced to int/float on write — older runs stored them as strings; on-disk file is now properly typed
- **Append safety:** scraper merges by season instead of overwriting, so `--season X` runs do NOT wipe other seasons (fixed 2026-04-24)
- **Missing columns vs SA:** No xg/npxg/sca/gca/xg_assist/progressive_passes — FBref provides these in the EPL match report HTML but the parser doesn't extract them yet (TODO)
- **Cloudflare:** Selenium scrape, ~17s/match, ~3 hrs per full season. Chrome driver crashes ~once per ~120 matches (botasaurus reuse_driver leak), requires manual `kill` + restart on the same season (downloads skip already-existing files)

### `data/parsed/understat_players.parquet`
- **Season-aggregate player xG / xA / xGChain / xGBuildup**
- **11,706 rows** across 12 seasons of Serie A (2014-15..2025-26) + 9 of Premier
  League (2017-18..2025-26) — recounted from the file 2026-07-16
- **Schema:** league, season, team, player, league_id, season_id, team_id, player_id, position, matches, minutes, goals, xg, np_goals, np_xg, assists, xa, shots, key_passes, yellow_cards, red_cards, xg_chain, xg_buildup
- **Writer:** `scripts/data/refresh_understat_players.py [--season YYYY-YYYY] [--leagues serie_a,premier_league] [--all-seasons]`
- **Source:** `playersData` JS variable on Understat league pages (extracted via Selenium)
  - ⚠️ **Selenium is mandatory, and it is NOT an IP block.** Measured 2026-07-16:
    plain HTTP GET returns **200** but `playersData` occurrences = **0** (the page
    renders client-side). Understat is reachable — unlike Sofascore (403) and
    FBref (Cloudflare). Don't "fix" this by retrying with requests.
- **Cadence:** `com.seriea-pipeline.refresh-understat` plist (Tue 04:30) is
  INSTALLED and invokes `-m scripts.data.refresh_understat_players --season 2025-2026`.
  The draft path this section used to cite never existed; the real copy is
  `scripts/pipeline/com.seriea-pipeline.refresh-understat.plist`. Auto-refresh in
  `run_full_pipeline.py` only triggers when stale, and uses `scrape_understat_xg()`
  which only fetches Serie A team-level data (not player-level)
- **Placeholders:** `team_id` is always 0 and `league_id` always null — `playersData`
  carries neither. Don't build a join on them.
- **Team naming quirk:** Players who transferred mid-season have comma-joined team strings (e.g. "Atalanta,Fiorentina"). Downstream `_team_match` fuzzy lookup handles this via substring match
- **2025-26:** 1,123 player rows (586 SA, 537 EPL), stored 2026-05-31. Re-fetched
  live 2026-07-16 and both leagues came back **identical to the stored rows**
  (assert_frame_equal, same values and dtypes) — the season is complete, so the
  file is final, not stale.

### `data/parsed/events.parquet`
- **Goal + card events with minutes** (from FBref scorebox — only goals/reds/own-goals; full timelines aren't in match report HTML)
- **11,781 rows** — columns: match_id, season, minute, event_type, team, player, detail
- **Writer:** `scripts/data/parse_all_events.py` (NEW today)
- **Fallback:** Sofascore `match_incidents.parquet` has full event timeline (cards, goals, subs, VAR)
- **2025-26:** 569 records (FBref goals + Sofascore cards & goals combined)

### `data/parsed/goalkeeper_stats.parquet`
- **Per-match GK stats** (28 cols: saves, PSxG, launches, pass completion)
- **6,651 rows**
- **Writer:** `scripts/data/parse_all_goalkeeper_stats.py`
- **Keying (FIXED 2026-07-17):** `match_id` is the canonical `{date}_{home}_{away}` from each report's own date+teams. Same bug/fix as `player_stats.parquet` — the writer keyed by `html_path.stem`, so 2025-26 (FBref `{hash}.html` reports) was 100% hash-keyed and invisible to `features/gk_quality.py` (which merges on canonical match_id, `gk_quality.py:84`) → GK features null for all of 2025-26. The 380 existing 2025-26 matches were re-keyed hash→canonical (pure relabel); the writer fix prevents recurrence.
- **Fallback:** None yet (could be added from Sofascore player_match_stats where position='G')
- **2025-26:** only FBref source, suffers the same Cloudflare gap

### `data/parsed/shots.parquet`
- **Individual shot events w/ xG** (15 cols: minute, player, xg_shot, psxg_shot, outcome, distance)
- **9,213 rows — ONLY 2024-2025 season**
- **Writer:** `scripts/data/parse_all_shots.py` (NEW today but produces 0 rows for 2025-26)
- **Status:** ❌ **FBref removed `shots_all` from 2025-26 reports — but this file has NO
  live reader, so that removal has zero downstream impact.** The only code that touches
  `parsed/shots.parquet` is its own writer (`parse_all_shots.py`). Every shot-level
  *feature* reads the Sofascore file below, not this one. (Removal confirmed source-side
  2026-07-16: `shots_all` absent from all 380 cached 2025-26 reports, present in 2024-25,
  and a scrolled re-fetch came back larger and still lacked it.) What is genuinely lost
  for 2025-26 is FBref's shot-creating-action chain (`sca_1_player`, `sca_1_type`) and
  `psxg_shot` — neither is consumed by any current feature.
- ⚠️ **Earlier versions of this section were WRONG** (corrected 2026-07-17): they said the
  64 shot features read `shots.parquet` and that the Sofascore shotmap could only
  "approximate" them. Both false. The features read `all_shots_with_xg.parquet`, which is
  true shot-level (per-shot xg/xgot/coordinates), and it has now been fully restored — see
  its entry under §4.
- **The real cause was a dead writer, now fixed.** The 64 `{home,away}_shot_*` features
  (`shot_xg_mean_roll_*`, `openplay_xg_roll_*`, `setpiece_xg_roll_*`, `counter_xg_roll_*`,
  `penalty_xg_roll_*`) were **45.8% NaN in 2025-26** because their source,
  `all_shots_with_xg.parquet`, stopped updating at 2026-02-08 (206/380) when its one-shot
  writer died. `scripts/data/write_shot_level_xg.py` (built 2026-07-17) re-derives the
  file from the Sofascore shotmap cache — restored to **380/380**, and the 64 features
  dropped to **2.8% NaN** overall (post-Feb-8 window: 100% → 0.3%; residual is legitimate
  rolling-window warmup, not a gap).
- **Scope: Serie A only.** `all_shots_with_xg.parquet` is SA across all 9 seasons; the shot
  plugins read that single file, so **EPL shot features are empty and always have been** —
  giving EPL shot features is a separate, un-built feature (it would mean adding EPL shots to
  this file and validating cross-league rolling). The weekly rebuild (Step 4b) therefore runs
  `serie_a` only; `write_shot_level_xg.cache_dir` is league-aware so a future EPL run reads
  the EPL cache, but nothing is wired to consume an EPL shot file today.
- ⚠️ **The 2.8% is verified on the real, canonical-keyed feature table — and it depends on
  that keying.** The shot plugins (`features/shot_level_xg.py`, `features/situational_xg.py`)
  bridge the natively-Sofascore-keyed shot file to canonical ids via
  `match_id_mapping.parquet`, then `_map_to_canonical` **inner-joins on the canonical
  `match_id`**. A feature row keyed by a numeric Sofascore id therefore receives **zero**
  shot columns (measured: an all-sofascore-keyed frame produced "no matches matched" — total
  failure, not partial NaN), whereas the current all-canonical table fills at 2.8%. This is
  why the derived-layer re-mint callout (features row 2) is coupled to this fix: the fix
  lands today only because the FBref backfill made the feature table 0-numeric.

---

## 4. Sofascore (6 files)

All refreshed weekly via `scripts/data/scrape_sofascore.py`. Raw JSON dumps cached in `data/external/sofascore/matches/{season}/{sofa_id}.json` (one per match, ~330 per season × 9 seasons = ~3,000 files).

| File | Rows | Cols | What |
|------|------|------|------|
| `player_match_stats.parquet` | 101,875 | 80 | Per-player per-match (xG, shots, passes, tackles, duels, etc.). Feeds the 19-market player floor engine (passes/tackles/duels/interceptions validated 2026-06-11, NB tail for passes). **EPL twin: `player_match_stats_premier_league.parquet` (97,003 rows, 33 clubs, 2017-18→2025-26). Read BOTH — see below.** |
| `match_team_stats.parquet` | 8,790 | 54 | Per-team per-match (possession, shots, xG, corners, passes, fouls) |
| `shotmap_stats.parquet` | 6,684 | 30 | Per-**team** shot aggregate, 2 rows/match (totals, xG, xGOT, situation counts, distance stats) — **not** shot-level |
| `all_shots_with_xg.parquet` | 82,432 | 27 | **Legacy shot events** (9 seasons, 2017-2024 strong, partial 2025-26) |

> ⚠️ **`player_match_stats` is TWO files, one per league.** Any consumer reading only
> `player_match_stats.parquet` is silently EPL-blind. Fixed in `lineup_predictor.py`
> 2026-08-01 (`_PLAYER_STATS_FILES`), where the single-file read left all 20 Premier
> League clubs with no XI prediction and graded every archived EPL prediction as a miss.
> Safe to concatenate: **zero** team-name overlap between the files. But `player_id`
> DOES overlap (229 players, correctly — it is a global Sofascore id), so never key
> across leagues on `player_id` alone. Filter each file to its **own** `season.max()`;
> a global max erases whichever league lags. This is failure mode #1 of the
> "EPL data missing where SA has it" catalogue in `CLAUDE.md` — check any new loader
> against it.
| `match_incidents.parquet` | 90,041 + VAR rows | 14 | Event timeline: goal, card, substitution, and since 2026-09-05 `varDecision` / `inGamePenalty` (+ `confirmed` column; `var_checked` marker rows mark matches re-fetched with no VAR incident). `incident_class` on a varDecision is the on-field decision, `confirmed=False` = overturned. `is_home` on a goal row is the SIDE CREDITED (own goals included on the beneficiary): agrees with final scores on 99.8% of 6,330 matches (measured 2026-09-05) |

#### Derived: `data/parsed/goal_timeline.parquet` — one row per goal, canonical keys (goal-process simulator input)
Columns: `match_id` (canonical `{date}_{home}_{away}`, via `match_id_mapping.parquet`), `sofascore_id`, `season`, `league`, `side` (home/away as credited), `minute`, `added_time`, `bin` (0..91: 0-44 = 1H minutes 1-45, 45 = 1H stoppage, 46-90 = 2H minutes, 91 = 2H stoppage), `half`, `goal_type`, `source_mtime` (watermark = incidents parquet mtime; the builder re-derives when the source moves, never compares against its own mtime). Sibling `goal_timeline_universe.parquet` lists every match that has ANY incident row (goalless matches included, 6,767 = 6,770 minus 3 unmapped) so base rates are computed over the right population. Stoppage mass on Serie A: 2.1% of goals in 1H stoppage, 6.3% in 2H stoppage. Consumers: `scripts/models/goal_process.py` (`fit_profile`, `backtest`); the served profile and gate live in `data/models/goal_process/{profile,backtest}.json` and are read at request time by `web/app.py::api_match_markets`. Never quote a skill number for this engine from a doc: read `backtest.json`. Sibling for the player per-half split: `data/models/player_floors/halves_backtest.json` (written by `python3 -m scripts.betting.player_predictions validate-halves`; shots / SoT only). |
| `captains.parquet` | 6,650 | 5 | Team captain per match |

### Current season coverage
- **330/330 matches** for all the active-maintained files (player_match_stats, match_team_stats, shotmap_stats)
- Match IDs are **Sofascore numeric IDs** (e.g., `13981421`) — NOT our canonical `YYYY-MM-DD_Home_Away` format
- Use `match_id_mapping.parquet` to join back

### Failure modes
- **Primary:** Sofascore API via `aiohttp` (free, no quota)
- **Plan B:** Cached JSON files on disk (always available)
- **Plan C:** Last saved parquet (stale but valid)
- 3-level caching → very robust

### `data/external/sofascore/friendlies_{season}.parquet` (PRE-SEASON — availability signal, NOT performance)

Written by `scraper/sofascore_friendlies.py` from Sofascore **unique-tournament 853 ("Club Friendly Games")**. One row per named player per friendly, **both sides**.

| Season file | Rows | Matches | Tracked clubs |
|---|---:|---:|---:|
| `friendlies_2026_2027.parquet` | 4,500 | 98 | 38 |
| `friendlies_2025_2026.parquet` | 7,461 | 165 | 40 |
| `friendlies_2024_2025.parquet` | 7,705 | 172 | 40 |
| `friendlies_2023_2024.parquet` | 6,799 | 153 | 39 |
| `friendlies_2022_2023.parquet` | 8,471 | 191 | 38 |
| `friendlies_2021_2022.parquet` | 3,778 | 90 | 34 |
| `friendlies_2020_2021.parquet` | 1,624 | 41 | 21 |
| `friendlies_2019_2020.parquet` | 638 | 17 | 9 |

Backfilled 2026-08-02 with `--pages 12 --force`: **40,976 rows / 927 matches / 3,575 tracked-club players**. The `--pages 4` backfill of the day before reached only 2024-25 and left 2023-24 as a 3-match fragment, which is why the constants could be tuned on just **two** informative seasons. Pagination in fact reaches **2020-21** (probed on Juventus: 4 pages → Jul 2024, 8 → Jul 2022, 12 → Sep 2020), and the old friendlies still carry real lineups — 941 of 1,214 events parsed, ~22% having no lineup payload at all.

**Usable backtest seasons, by club coverage** (out of 20 per league):

| season | SA clubs | EPL clubs | lineup (`is_starter`) | participation (`minutes`, `was_used`, rating) |
|---|---|---|---|---|
| 2023-24 … 2026-27 | 19–20 | 18–20 | ✅ | ✅ **the only re-validation set** |
| 2022-23 | 18 | 20 | ✅ | ❌ all-zero |
| 2021-22 | 17 | 17 | ✅ | ❌ all-zero |
| 2020-21 | 13 | 8 | partial | ❌ all-zero |
| 2019-20 | 6 | 3 | fragment | ❌ all-zero |

**Sofascore keeps the LINEUP forever but drops the STATISTICS after ~3 years, and the boundary is wall-clock, so it MOVES.** Measured 2026-08-02 the cliff is at **2023-07-05**: every quarter before it has *exactly* 0.0 rows with `was_used`, every quarter after holds a steady ~0.25. Pre-cliff rows carry the named XI and `is_starter` (11.00 per club, clean) but `minutes_played` is identically 0 and `was_used` identically False for all 14,511 of them.

This is why the deep seasons do **not** give five re-validation seasons, only four (2023-24 … 2026-27, still 2× the two the constants were tuned on). The `(16, −30, fade 3.0)` window was solved around an invariant about *named-but-unused substitutes* — and pre-cliff every player looks unused, so pooling those seasons would introduce a season-correlated measurement difference in the exact variable under test. They remain valid for anything keyed on `is_starter` alone.

**Consequence for the writer, and it is a data-loss one:** `_save` merges on `(sofascore_event_id, player_id)`, and blind `keep="last"` against a source that degrades over time would let a future `--pages 12` run overwrite the real minutes with zeros — for exactly the seasons we can never re-acquire them for, since the parquet is now the only surviving copy. `_drop_duplicates_without_downgrading()` prefers whichever row still carries participation data and falls back to arrival order only on a tie, so "a corrected re-scrape updates in place" still holds and a same-day re-run that fills in stats still upgrades. Pinned by mutation-killed tests in `tests/test_sofascore_friendlies.py`.

Two properties to hold in mind before backtesting on the deep seasons. The club set is the **current** one, so a club has friendly rows for seasons in which it was in a different division (Cremonese 2021-22) and simply has no league rows to grade against — the join drops them, it does not corrupt them. And the shipped `(SHRINK_PRIOR_STRENGTH=16, PRESEASON_ABSENT_PENALTY=-30, PRESEASON_FADE_MATCHES=3.0)` window was solved on the **old, two-season** corpus; re-tuning against this larger one is a cross-condition comparison unless the eval set is held fixed.

**Columns:** `sofascore_event_id`, `match_date`, `season`, `club`, `club_id`, `opponent`, `is_home`, `is_our_club`, `club_league`, `formation`, `player`, `player_id`, `shirt_number`, `position`, `is_starter`, `minutes_played`, `was_used`, `rating_low_trust`, plus opponent provenance: `opponent_id`, `opponent_country`, `opponent_league`, `opponent_league_id`, `opponent_country_priority`, `opponent_is_national`, `opponent_is_youth`, `opponent_tier`.

**Opponent provenance — who did we actually play?** A friendly result is unreadable without the opposition's level: "Juventus 2-0 Nice" and "Torino 3-0 ACD Pinzolo" are identical in the payload. Each distinct opponent is resolved once via `/team/{id}` → `primaryUniqueTournament`, cached in `data/external/sofascore/friendly_opponent_profiles.json`. The **raw opponent name is always kept verbatim**; these columns only add context beside it, so a wrong bucket can be re-derived from the stored facts. `opponent_tier` ∈ `top5_league` / `other_professional` / `youth_or_reserve` / `national_team` / `lower_or_unknown`.

2026 pre-season distribution (90 matches): `other_professional` 72, `top5_league` 15, `youth_or_reserve` 2, `lower_or_unknown` 1. Note `other_professional` spans a wide range (Championship → League Two → Scottish Premiership), so use `opponent_league` / `opponent_country_priority` rather than the tier alone when strength matters.

**The profile cache is SEASON-STAMPED and self-invalidates (fixed 2026-08-02).** `friendly_opponent_profiles.json` is keyed by team id and never re-fetched within a season, so an opponent promoted or relegated after its first resolution would keep its old `opponent_league` / `opponent_tier` forever. This used to carry a "**delete the JSON on season rollover**" instruction here — a manual step nobody performs, which is why it was a latent bug rather than a documented safeguard. The file now stores a `__season__` key and `_load_opp_cache()` discards the whole cache when that stamp is not the current season, so re-resolution is automatic. An **unstamped** file is treated as current, not stale — the file on disk when the stamp was introduced had just been fully rebuilt by the Aug 1 backfill (verified: all six 26/27-promoted clubs already carried their NEW league, e.g. Frosinone → Serie A rather than Serie B), so discarding it would have re-fetched 261 correct profiles for nothing. Nothing is lost by re-deriving in any case: the parquet keeps the raw `opponent` name verbatim.

**Consumed by:** `scripts/prediction/lineup_predictor.load_preseason_signal()` → `get_starter_frequency(..., preseason=...)`. Three effects, all inert without friendly data: an informed Bayesian prior (itself shrunk by friendly count), injection of players with **zero** league rows, and a downgrade for last-season regulars absent from ≥3 club friendlies. Also supplies the **formation** for promoted clubs, which have no league history to infer one from.

**The signal expires in-season.** It is scaled by a fade that keys on whether the league table has caught up to the friendlies' own season, then decays over `PRESEASON_FADE_MATCHES` (3.0) league matches and is fully retired by MW3. This is not optional: un-faded, at 8 matchweeks it injected 32 players at 80% who had not played a league minute all season. Fading on match *count* alone would be wrong — during pre-season the 10-match window is full of *last* season's matches. Measured 2026-08-01 across 17 Serie A clubs: **194 players** who featured in a friendly had no rows in last season's league table and could not be predicted at all; **72 players** with ≥5/10 starts last season appeared in no friendly. This affects the **displayed** XI and the advisor's grounding only — `lineup_predictions.json` is not consumed by the match-outcome models.

**What it is FOR — read this before using it.** Friendly *performance* is close to noise: no stakes, wildly uneven opposition (a Serie A side vs a fourth-tier local club), rolling 11-man half-time substitutions, trialists in the XI, and squads deliberately short of match fitness. What it measures reliably is **availability and trust**: who is fit enough to be named, which signing is *not* getting minutes, who has been dropped, what shape is being trialled, and whose minutes are ramping toward matchday 1. That is a legitimate prior for MW1 XI prediction. The Sofascore rating is carried as **`rating_low_trust`** — the column name is deliberate, so no downstream join treats a July friendly rating as comparable to a league rating.

**Unused substitutes are kept with `minutes_played = 0`** (`was_used = False`). "Named but did not play" is the signal; dropping those rows would leave only the players who featured.

**Isolation — important.** Friendlies are **never** written to `lineups.parquet`, `player_match_stats.parquet` or any league table; a friendly row there would silently contaminate the training set. The module keeps its own tournament id and does **not** touch `scraper.sofascore_lineups._SUPPORTED_TOURNAMENT_IDS` (the league ingest gate). Verified: zero `sofascore_event_id` overlap with the league tables.

**Naming:** `club` is the **canonical** repo name for tracked clubs (`Milan`, not `AC Milan`); untracked friendly opponents keep their raw Sofascore name. Both sides are tagged `is_our_club` when two tracked clubs meet (e.g. Sassuolo v Parma) — 4 such fixtures in 2026 pre-season.

**Refresh:** automated — `com.seriea-pipeline.friendlies-refresh` plist, **daily 06:30** (template in `scripts/pipeline/`, installed to `~/Library/LaunchAgents/`, logs to `logs/launchd-friendlies-refresh{,-err}.log`). Appears in the dashboard's `/api/scheduler/status` active/inactive list like every other job. Deliberately *not* 05:00–05:30: the Monday `weekly-data-refresh` job runs at 05:00 and this scrape takes ~5 minutes. Writes are atomic (tmp + replace) so an overlapping reader can never see a torn parquet.

The script **self-gates on `FRIENDLY_WINDOWS`** (1 Jun–5 Sep, 15 Dec–10 Jan) and exits 0 immediately outside them, so it is cheap to leave loaded year-round rather than remembering to unload it in September. Manual run: `python3 -m scraper.sofascore_friendlies --leagues serie_a,premier_league` (`--dry-run` to fetch without writing, `--force` to override the window, `--pages 2` to reach the *previous* pre-season). Re-running is idempotent — rows merge on `(sofascore_event_id, player_id)`, last write wins, so a corrected re-scrape updates in place.

**Coverage limit:** the default single page of match history covers the whole *current* pre-season (measured: Milan page 0 spans 2025-12-04 → 2026-07-25) but not earlier ones. Use `--pages 4` to reach back three pre-seasons; page depth runs out during 2023-24.

**✅ The signal IS backtested (2026-08-01).** `scripts/analysis/backtest_preseason_signal.py` replays past matchweeks with the signal on and off, against a `naive` raw-start-count floor. Result over 578 fixtures, both leagues:

| Matchweek | n | naive | model, no signal | model + signal | delta |
|---|---:|---:|---:|---:|---:|
| **1** | 120 | 48.8% | 47.6% | **55.6%** | **+8.0pp** |
| 2 | 104 | 83.5% | 83.5% | 83.5% | +0.0pp |
| 3 | 116 | 78.2% | 81.2% | 81.1% | −0.1pp |
| 4 | 120 | 73.2% | 77.9% | 77.6% | −0.2pp |
| 5 | 118 | 72.8% | 76.9% | 76.7% | −0.2pp |

**The signal is real, and it is a matchweek-1 effect.** Paired t=4.80 at MW1, and of the 40 fixtures it moved, **39 improved and 1 worsened**. It is inert at MW2 and mildly *negative* at MW3–5 (3 better vs 10 worse combined) — so `PRESEASON_FADE_MATCHES` was cut 5 -> **3.0** (2026-08-02) to stop keeping it alive through matchweeks where it only costs accuracy. Note `_fade` is only computed when the table and the friendlies share a season, which is never true at MW1, so **the fade constant does not affect the matchweek the signal actually works on**.

**Coverage is monitored, because the gap is otherwise invisible.** `scripts/pipeline/health_check.check_preseason_coverage()` compares the live club list (`fetch_club_ids`, same normalisation as the writer) against the clubs actually present in the parquet, and raises a **WARNING** naming every club with no friendly rows. Nothing is broken when it fires — Sofascore may not list a club's friendlies, or they may have been played and cancelled (measured 2026-08-02: **Brentford** has none listed, **Brighton**'s only friendly was `status: canceled` with zero players, correctly refused by the writer). What the line buys is that those clubs' fallback to a summer-stale table on opening day is *stated* rather than silently costing ~11pp of MW1 XI accuracy. The roster it compares against is **read from a file, never fetched**: `data/external/sofascore/friendly_club_roster.json`, written by the daily scrape (`_save_club_roster`, skipped on `--dry-run`) and read by `load_club_roster()`. The health monitor runs every 30 minutes, so a live `fetch_club_ids` from there would be ~190 Sofascore requests a day for three months — against a source this repo is periodically banned from — to re-fetch a list that changes once a year. The sidecar is **season-stamped and fetch-timestamped**, and `load_club_roster()` returns `{}` (never a partial answer) when it is missing, malformed, stamped to another season, or older than 48 h; the caller must read `{}` as *not computable*, never as *no clubs*. Three traps are guarded and all are pinned by mutation-killed tests in `tests/test_preseason_coverage.py` and `tests/test_sofascore_friendlies.py`: the check resolves its season via `sofascore_friendlies.current_friendly_season()`, **not** `config.settings.get_current_season()` (the calendar helper rolls 1 Aug while the window opens 1 Jun, so through June and July it names the season that just ENDED and would open the wrong parquet); it reports `UNKNOWN` rather than `OK` when a league's list comes back with <18 clubs, because `fetch_club_ids` logs-and-continues on a 403 and an empty `expected` set would otherwise read as perfect coverage exactly when we are blocked; and only `is_our_club` rows count, so a club is never marked covered by having been *played against*. Outside `FRIENDLY_WINDOWS` and once the new season has ≥20 played matches the check is a no-op.

**Promoted clubs are the extreme case:** with no prior league table the model returns literally nothing, so the friendlies are the only evidence that exists.

**Looks like a bug, is not: the friendly club list will not match the live player-stats table.** Between May and the first matchweek, `player_match_stats` holds LAST season's clubs while the friendly scrape resolves the NEW season's, live from `/unique-tournament/{tid}/seasons` → `seasons[0]` (verified 2026-08-01 to be the 26/27 id for both leagues). The difference is promotion/relegation, and it should be exactly **3-up-3-down per league**:

| | out of the friendly list (relegated) | into it (promoted) |
|---|---|---|
| Serie A 2026-27 | Cremonese, Pisa, Verona | Frosinone, Monza, Venezia |
| Premier League 2026-27 | Burnley, West Ham, Wolves | Coventry City, Hull, Ipswich |

So a relegated club having **no** pre-season signal is correct — it is not in the league. A promoted club having one is the whole point, since it has no top-flight history at all. **Check the 3-up-3-down shape before concluding the club list is stale**; a mismatch that is *not* 3-and-3 is the real bug signal. A current-league club with zero rows usually just has not played a tracked friendly yet (Brentford and Brighton on 2026-08-01) and fills in as pre-season runs.

**Season attribution — and the leak it hides.** `_season_for` files **everything from June onward** under the season that *starts* that August, not the ordinary Aug-1 boundary in `config.settings.get_current_season()`. That is right for pre-season friendlies, which is what the column is for.

⚠️ **But the season label is therefore NOT a time bound.** A friendly played in **March 2025** is stamped `2024-2025` — the same label as the genuine July-2024 pre-season. Anything replaying a historical matchweek that filters by `season` alone will be handed matches from *months in its own future*. Two such friendlies exist in the 2024-08→2026-08 backfill.

**Rule for any consumer that reasons about a past point in time:** pass `load_preseason_signal(team, season=..., before=<cutoff>)`. `before` drops friendlies played on/after the cutoff; the backtest passes the club's first league fixture of that season, which is also what "pre-season" literally means. Production leaves `before=None` — there is no future to leak from *today*. Filtering by season alone is safe **only** for the current, in-progress pre-season.

---

## 5. Understat

**Single canonical file:** `data/external/understat/matches_xg.parquet` — 3,370 matches, 9 seasons (2017-2026)

### How it's refreshed
1. `scraper/understat_scraper.py::scrape_understat_xg()` scrapes via selenium (often times out)
2. Falls back to requests+HTML parsing
3. Writes per-season JSONs: `understat_2017_2018.json` ... `understat_2025_2026.json`
4. `scripts/data/parse_all_understat.py` merges all JSONs into canonical parquet

### Schema
`match_id`, `datetime`, `season`, `home_team`, `away_team`, `home_id`, `away_id`, `home_goals`, `away_goals`, `home_xg`, `away_xg`, `forecast_home/draw/away`, `is_result`

### Deprecated files (archived today)
- `matches_xg_2025_2026.parquet` (moved to `_deprecated/`)
- `matches_xg_normalized.parquet` (moved to `_deprecated/`)
- `matches_xg.parquet` is now the single source of truth

### Failure modes
- **Plan A:** Selenium scrape (times out 30-40% of the time)
- **Plan B:** Requests-based HTML parse (fallback in same scraper, activates on selenium fail)
- **Plan C:** Use existing per-season JSONs on disk (already 9 years of data cached)
- Very robust — worst case, current season is stale but 8 historical seasons stay intact

---

## 6. Weather / Referees / External context

### `data/external/weather.parquet`
- **Source:** Open-Meteo archive API (`archive-api.open-meteo.com` — free, unlimited)
- **11,433 rows** — one per match_id
- **13 columns:** temperature (min/max/mean), apparent_temp, precipitation, rain, snow, wind_speed, wind_gusts, wind_direction, humidity
- **Refresh:** Weekly, only fetches NEW match_ids (cache-based)
- **Failure fix shipped today:** when `venue` column is NaN (matches added via Sofascore without venue data), falls back to `home_team → city` mapping via new `TEAM_TO_CITY` dict in `scraper/weather.py`
- **Known issue:** Open-Meteo has 30s read timeout, sometimes all 3 retries fail. Affects ~25% of 2025-26 matches currently.

### `data/external/referee/referee_assignments.parquet`
- **Source:** whoscored.com scraper (`scraper/referee.py::scrape_all_referee_assignments`)
- **3,368 rows** — 9 seasons
- **Columns:** match_date, home_team, away_team, referee, matchweek, ref_yellows, ref_second_yellows, ref_reds, season
- **Refresh:** Weekly (cache file deleted before rescrape)
- **Failure modes:**
  - Plan A: whoscored (sometimes rate-limited)
  - Plan B: **MISSING** — football-data.org has ref data but not integrated
  - Plan C: FBref match report scorebox (already feeds `matches.parquet.referee`)

### `data/external/transfermarkt/market_values_*.parquet`
- **Per-season player market values (EUR)** from Transfermarkt scrape
- **~860 rows per season** × 9 seasons
- Used as team/player-strength proxy in features

### `data/external/transfermarkt/transfers_2026_2027.parquet` (ins/outs — squad tracker; delta GATED out of the live model)
- **Confirmed 2026-27 transfers per Serie A club** (arrivals + departures). 592 rows / 20 clubs (re-scraped 2026-07-14). Schema (16 cols): `team, transfer_type (in/out), player_name, position, nationality, age, from_club, to_club, market_value_at_transfer, fee_text, fee_eur, is_loan, window` (from TM) + `transfer_date, date_window, n_sources` (from the Wikipedia merge, below).
- **Writer:** `scraper/transfermarkt.scrape_transfers(season, league, only_teams)`. `current_league_teams()` resolves the ACTUAL 20 member clubs from TM's competition page (the static map is a historical superset). NOTE 2026-07-14: TM's 26/27 SA competition page correctly lists the promoted set **Venezia/Frosinone/Monza** (Cremonese/Hellas Verona/Pisa were relegated) — verified against Wikipedia's promoted/relegated table.
- **✅ Fee-vs-market-value bug FIXED in the 26/27 file 2026-07-14 (the 9 HISTORICAL transfer files 2017-2025 still carry the buggy fee=MV — re-scrape needed if ever used for fee sums).** On the `/plus/1` layout a row shows BOTH a market value and a fee as `td.rechts`; the old `select_one("td.rechts")` grabbed the FIRST (market value) and stored it as the fee — e.g. Højlund's €44m fee was recorded as his €60m market value. Now the LAST `td.rechts` is the fee and the first is captured separately as `market_value_at_transfer`. **Blast radius: `jan_spend` (sums fees) was corrupted; `net_squad_delta` was essentially UNAFFECTED** — it weights by the separate `market_values_*` file, and `_transfer_materiality` classes any €-fee as `("paid", 1.0)` regardless of magnitude, so a wrong fee number didn't change its weight or class. Neither is in the deployed model, so live-prediction impact was zero regardless. Locked by `tests/test_transfer_multisource.py`.
- **Richer detail added 2026-07-14 (dashboard + data-quality):** `position`, `nationality`, `from_club`/`to_club` (the club selector was scoped too tightly — `td.zentriert a` found 0 on `/plus/1`; now row-level `a[href*='/verein/']`, preferring the crest img alt for the full club name), `market_value_at_transfer`. All 100% filled on the 26/27 file.
- **Consumed by:** `features/transfer_impact_analysis.compute_net_squad_delta()` → `home/away_net_squad_delta` + `net_squad_delta_diff` (Step 35). Player weight blends TM market value with last-season minutes/rating from `sofascore/player_match_stats.parquet`.
- **⚠ NOT a live model feature yet — GATED OUT of `get_ml_feature_columns`** (`features/build.py`) pending a leak-free held-out backtest with skill > 0 (project cardinal rule). Both pre-conditions for the backtest are now MET: (1) the winter-window LEAK IS FIXED — `compute_net_squad_delta` excludes `window=="winter"` rows (transfer files are now window-tagged for all 10 seasons 2017-2027 via `backfill_transfer_windows.py`, 2026-07-14), so winter signings no longer leak onto August matches; (2) `market_values_2026_2027.parquet` now exists, so live-signing talent weight is real, not the 0.15 floor. Backtest harness: `scripts/analysis/backtest_net_squad_delta.py` → `net_squad_delta_backtest.json`. Until it passes skill > 0, the columns drive ONLY the `/transfers` dashboard.
- **NOTE — the `jan_*` / `squad_disruption` / `signing_integration` transfer features are NOT in the deployed 126-feature model** (verified against `catboost_no_odds_metadata.json:feature_names` 2026-07-14; only `us_squad_depth` from Understat is). The window-filter + loan-dedup fixes correct the feature VALUES (for the dashboard and the backtest input); they do not change that model.

  ⚠️ **Corrected 2026-08-01 — "never affected live 1X2 predictions" was too strong.** It holds for `catboost_no_odds`, but that is not the only live model. **`draw_detector`** (697 features, `blend_enabled=true`, `blend_alpha=0.32`, blended into the ensemble at `ensemble_prediction_engine.py:2481`) carries all six `home/away_{jan_arrivals,squad_disruption,signing_integration}`, and **`catboost_upcoming_v2`** carries `away_jan_arrivals`. So transfer features DO reach live output, via the draw blend. The CV-ceiling claim is unaffected — that is measured on `catboost_no_odds`.

  ⚠️ **`data/models/serie_a/` is NOT the Serie A production model directory.** It is written by `ml.persistence.save_model` and never read: `_league_model_dir()` returns `MODELS_DIR` *root* for Serie A, and `_load_league_model()` only runs for `league != "serie_a"`. Anything measured there describes a trained-but-undeployed model. `premier_league/` **is** read and live.

- **`signing_integration` is excluded from ML training as of 2026-08-01 — measured worthless.** Over 7,980 matches it takes **three** distinct values (1.00 / 0.30 / 0.65) with **93.3% of rows at 1.00**, and it ranks **62/68 at 0.0–0.1% importance** in all three CatBoost/LightGBM/XGBoost models under `models/serie_a/`. Its source, `INTEGRATION_CURVES`, is 32 hand-invented constants whose position-specificity never surfaces because nearly every row sits at the fully-integrated plateau. It is **still computed** (so already-trained models keep loading — prediction selects by model metadata, never by `get_ml_feature_columns`) but is withheld from the next retrain. Its sibling **`squad_disruption` is KEPT** — same module, but rank 10/68, 24/68 and 19/68 in those same models. Do not collapse the two in a cleanup.

  **Counter-evidence, recorded honestly.** Measured directly off `draw_detector.cbm` (its metadata stores no importances): `home_signing_integration` is rank **98/697 at 0.24%** — not zero. But `away_signing_integration` is rank 684/697 at **0.00%**, and the identical one-side-alive/mirror-exactly-zero pattern holds for `squad_disruption` (away 0.21% / home 0.00%) and `jan_arrivals` (away 0.08% / home 0.00%). A feature carrying real squad-integration signal should not matter for the home side and be exactly nil for the away side; that asymmetry is the signature of arbitrary selection among noisy correlated columns. **294 of 697 features sit at exactly zero**, the top five are all odds/market features (3.4–5.0% each), and the whole draw blend is worth `avg_ll_improvement=0.00278`. The exclusion is a judgement call resting on the model-independent resolution argument, not a slam dunk.
- **✅ Loan-to-permanent double-count FIXED 2026-07-14** — `_loan_to_permanent_outs()` drops the phantom "End of loan" OUT for a bought-back loanee (real IN + End-of-loan OUT same player) from BOTH `compute_net_squad_delta` and the live `squad_disruption` departure loop. A bought-back loanee counts as an arrival, not a departure. 12 tests in `tests/test_transfer_delta.py`.
- **✅ January temporal leak FIXED 2026-07-14** — `compute_january_window_features` filters on `window=="winter"` (was: dead date branch → `else: assume all January` ingesting the whole season). Untagged files → jan_*=0 (leak-free), not keep-all.
- **⚠ 60% of rows are "End of loan" returns** (not real squad changes) — the feature discounts them 0.3× and guards against double-counting a loanee who returns AND leaves.
- **✅ Reconciled against TM's own rosters 2026-07-14** (`scripts/analysis/roster_reconciliation.py` → `data/analysis/roster_reconciliation_2026_2027.json`). Method: per club, set-diff the 25/26 kader (`saison_id/2025`) vs the 26/27 kader (`saison_id/2026`) to get reality's arrivals/departures, compare to our IN/OUT list. All 20 clubs checked. **Result: our IN list captured every real arrival for 19 of 20 clubs — only miss = Lazio/Danilho Doekhi (genuinely absent, not a name-match miss).**
- **Scrape artifact — loan-to-permanent double-listing (32 players, now handled in code):** a player loaned in 25/26 then bought permanently in summer 2026 gets BOTH an "End of loan" OUT row AND a paid IN row in the raw parquet (e.g. Napoli/Højlund, Inter/Akanji, Juventus/Openda, Napoli/Vanja Milinković-Savić — the actual 26/27 keeper). The raw rows are LEFT as-is (they mirror TM); the phantom OUT is dropped at compute time by `_loan_to_permanent_outs()` (bought-back loanee = arrival, not departure). Decision resolved: arrival.

### `data/external/transfermarkt/market_values_2026_2027.parquet` (the CURRENT ROSA per club — squad pages)
- **The actual current squad per Serie A club** (distinct from the transfer-FLOW view): `team, player_name, position, age, market_value_eur, nationality`. **616 players / 20 clubs (re-scraped 2026-08-25).** This is TM's live squad page — last season's stayers + this window's arrivals − departures, already resolved.
- **Writer:** `scraper/transfermarkt.scrape_squad_market_values(season, league, only_teams)` → `_parse_squad_page`. Squad lists include primavera/youth on the senior page (counts run 25–43). Refreshed by `com.seriea-pipeline.transfer-refresh` (06:00 + 18:00 daily) via `scripts/data/refresh_transfers.py`; `SQUAD_CACHE_MAX_AGE_HOURS = 24` means one real re-scrape per day, the second run finding the cache fresh.
- **✅ FROZEN-CACHE + GHOST-CLUB bug FIXED 2026-08-25 (both visible on /rosters).** Two defects compounded. (1) **No TTL:** the writer returned the cache whenever every current club was already present, so the file stopped updating the day it was first completed — it sat unchanged through the whole summer window while the twice-daily job dutifully ran and did nothing. (2) **Append-only merge against a historical superset team map:** the file accumulated every club ever scraped, so a 20-club league had **29 clubs / 919 rows** — Chievo (last in Serie A 2018-19), SPAL, Crotone, Benevento, Brescia, Sampdoria, Salernitana, Empoli and Verona, **256 players rendered on /rosters as current squad members**. This entry previously read "663 players / 20 clubs" because that is the current-team SUBSET; the ghosts were never counted. Fixes: 24h TTL, replace-not-append on re-scrape (a bare concat would have doubled every squad on the first refresh), and `_prune_to_league()` which drops non-member clubs **and persists the prune to the parquet** — every consumer reads the file directly, so pruning only the return value would have left the ghosts on screen. Locked by `tests/test_transfermarkt_squad_cache.py`.
- **Consumed by:** `net_squad_delta` talent-weighting (season-matched market value per player) AND the `/rosters` dashboard page (`/api/rosters` → `web/templates/rosters.html`), which groups players GK→DEF→MID→ATT sorted by value and flags confirmed 26/27 arrivals as "new".
- **✅ Age bug FIXED 2026-07-14 (was a REAL defect, visible on /rosters).** The old age extractor took the first `^\d{1,2}$` cell — which is the SHIRT NUMBER (`td.rueckennummer`), not the age (e.g. Scamacca shirt 9 → age 9; 162 players under 15). The age is the parenthesized number in the DOB cell "01/07/2000 (26)". Fixed to parse `\((\d{1,2})\)`, with a shirt-number-skipping fallback bounded to [15,45]. Re-scraped: age range now 17–40, zero impossible ages.

### `data/external/transfermarkt/salaries_2026_2027.parquet` (Capology ESTIMATED wages — DISPLAY-ONLY, never a model feature)
- **Per-player estimated salary for all 20 Serie A clubs:** `player_name, team, annual_gross_eur, monthly_gross_eur, weekly_gross_eur, verified, capology_position, capology_age, contract_expiration, years_remaining, capology_slug, name_norm`. 561 players / 20 clubs (2026-07-14).
- **⚠ ESTIMATES, NOT OFFICIAL FIGURES.** Capology's own header: *"All amounts are estimates and do not represent official figures."* The number is the **Est. Fixed** guaranteed gross salary — it **EXCLUDES bonuses / image rights / commercial deals** (Capology reports Fixed / Bonus / Total separately; we store Fixed only, because bonuses are conditional and Capology flags them "may be incomplete"). So media "total comp" quotes read HIGHER than this fixed figure (e.g. Lautaro fixed €16.67m here vs a media "€20m+" incl. bonuses — both correct, different tiers). `monthly = annual/12`, `weekly = annual/52`. `verified` = Capology's green-check (239/561 = 43%); rest are estimated.
- **Writer:** `scraper/capology_salaries.py` (`save_salaries(season)` / `scrape_all_salaries`). Fetches `capology.com/club/{slug}/salaries/` with the project's `TM_HEADERS` (Capology 403s a bare fetcher, 200s with those). **Specimen-verified (cardinal rule):** the visible `<tbody>` is EMPTY (bootstrap-table renders client-side), but the row data IS in the static HTML in a `var data = [{...}]` JS array. Each row parsed **per-object** (name + salary + verified + age + contract from the SAME `{...}` block — never two zipped lists, so a name can't drift onto the wrong salary). **Per-club checksum:** the sum of parsed fixed salaries must equal Capology's own stated `#annual-fixed` total to the euro, else the club is DROPPED (breaker), never shipped mis-aligned. Range sentinel €30k–€35m. `CLUB_SLUGS` maps each of the 20 `team` names (matching market_values) to its canonical Capology slug — verified against Capology's Serie A index.
- **Consumed by:** ONLY the `/rosters` dashboard (`/api/rosters` joins it at READ-TIME by `(team, normalized_name)`, ~79% of the roster matches; unmatched → no figure shown, never a wrong one). **NEVER materialized into `market_values_*.parquet`** — co-mingling an estimate unlabeled beside TM's real market value is forbidden. UI labels it "Capology estimate · fixed gross" with a per-row ✓/≈ verified flag.
- **⚠ NEVER a model feature.** Salary estimates predicting match outcomes is exactly the fiction this pipeline gates (`net_squad_delta` itself is still GATED). This parquet feeds display only.
- **Cadence:** NOT on the twice-daily TM cron — Capology 429-throttles a fast sweep (empirical: 429 at ~club 15 of a 4s-paced run; the scraper now has exponential backoff on 429). Salaries change per signing, not per day → refresh **weekly / on-demand** via `python3 -m scraper.capology_salaries`.

### `data/external/transfermarkt/wiki_transfers_2026_2027.parquet` (SECOND SOURCE — cross-check + exact date)
- **Independent Wikipedia transfer feed for Serie A**, the complement to Transfermarkt: `player_name, from_club, to_club, fee_text, fee_eur, transfer_date (exact), window (derived from date), source`. 323 Serie A-touching rows (2026-07-14).
- **Writer:** `scraper/wiki_transfers.scrape_wiki_transfers(season)`. Parses the `List_of_Italian_football_transfers_{summer,winter}_YYYY` wikitables. **Verified specimen-first** (cardinal rule): header `[Date, Name, Moving from, Moving to, Fee]`, single wikitable per page.
- **Why a 2nd source (NOT redundant with TM):** TM has money/position but NO per-row date (only a summer/winter window from a URL param); Wikipedia has the EXACT date + independent from/to club. They cross-validate. `football-data.org` was rejected — it has no transfer data (fixtures/standings/odds only).
- **Rowspan handling:** Wikipedia rowspan-merges the Date cell (several transfers share one date → later rows have 4 cells, not 5). The parser carries the last-seen date forward — without this, ~330 real rows/page were silently dropped (e.g. Manor Solomon→Fiorentina). Bad-shape skips are logged, never silent.
- **Merge → TM spine:** `enrich_transfers_with_wiki(season)` joins Wikipedia's `transfer_date` onto the TM file by normalized name, adding `transfer_date, date_window, n_sources` (2 = confirmed by both feeds; 1 = TM only). 240/592 TM rows dual-confirmed; 95 non-loan rows carry an exact date; window agreement 79/95 (83%) on real transfers. **Loan-return rows are cross-confirmed but get NO date** — a TM "End of loan" (summer return) and a Wikipedia row (original Jan loan-out) are DIFFERENT events, so Wikipedia's date would be the wrong event's.
- **NEVER a direct model input** — display + data-quality only. The model feature (`net_squad_delta`) reads the TM spine columns, not `transfer_date`/`n_sources`.

### `data/external/transfermarkt/rumors_2026_2027.parquet` (DISPLAY-ONLY — never in the model)
- **Unconfirmed transfer rumors per club:** `team, player_name, age, current_club, market_value_text, market_value_eur, source_date, source_url, confirmed(=False), scraped_at`. ~399 rows / 18 clubs (2026-07-14).
- **Writer:** `scraper/transfermarkt.scrape_rumors()`. Overwritten each run (rumors expire); no incremental cache.
- **NEVER read by the feature layer** — `compute_net_squad_delta` reads `transfers_*` only. Surfaced on `/transfers` dashboard as speculation with source date for traceability. TM's per-rumor "assessment" is usually blank, so NO fabricated probability is stored.
- ⚠️ **Survivorship-biased — do NOT use this file for any retrospective study.** It shows only the rumors alive on the day you read it. Use `rumor_history.parquet` below.

### `data/external/transfermarkt/rumor_history.parquet` + `rumor_scrape_log.parquet` (APPEND-ONLY — the studyable record)

**What it is.** Every rumor ever observed, with a measurable lifetime. Seeded 2026-08-01 from the live snapshot (459 rumors / 20 Serie A clubs); grows daily thereafter. This is the file to read for *any* question of the form "do rumors predict transfers", "how long does a real link survive", "does market value predict completion".

**Key** (`rumor_history.KEY`): `league, season, team, player_name, current_club`. **`source_url` is deliberately NOT in the key** — it embeds a Transfermarkt forum `post_id`, so a fresh post about the same rumor would mint a new row and reset `first_seen`, destroying the lifetime that is the whole point. URL is a latest-wins attribute.

**Columns.** Key + latest-wins attributes (`age`, `market_value_eur`, `market_value_text`, `source_date`, `source_url`) + lifecycle: `first_seen`, `last_seen`, `last_covered_at`, `times_seen`, `first_run_id`, `last_run_id`.

**⚠️ How to read it — use `annotate_status()`, never a bare `last_seen` comparison.**
A stale `last_seen` has two opposite meanings: *the rumor was dropped*, or *the scraper was blind* (Transfermarkt 403s per club, and `refresh_transfers` swallows the whole step's exception). Those are **opposite labels** for the supervised question, so the store records per-club coverage separately and `last_covered_at` marks the last time a **successful** run looked at that club.

```python
from scripts.data.rumor_history import annotate_status
df = annotate_status()          # adds days_alive, is_dropped, is_live, days_dark
real_drops = df[df.is_dropped]  # a covering run ran AFTER last_seen
unreliable = df[df.days_dark > 3]   # scraper blind here — trust neither verdict
```
- `is_dropped` — `last_covered_at > last_seen`. The rumor genuinely disappeared.
- `is_live` — still listed as of the latest covering run.
- `days_alive` — `last_seen − first_seen`. The lifetime feature.
- `days_dark` — days since a run covered this club. Large ⇒ both verdicts unreliable.

**`rumor_scrape_log.parquet`** — one row per run: `run_id, league, season, status (ok/partial/failed), teams_expected, teams_covered, n_rows, covered_teams, failed_teams`. Read this before trusting any window of history; a `failed`/`partial` streak is a hole in the record, not a burst of dropped rumors.

**Writer:** `scripts/data/rumor_history.record_run()`, called by `scripts/data/refresh_transfers.py` step 3 right after `scrape_rumors` (which now fills a `coverage` out-param). Atomic tmp+replace. Additive and fail-soft: a dead scrape logs `status=failed` and **never** erases history.

**Still NEVER a model feature.** This makes rumors *studyable*, which is the precondition for ever deciding whether they earn a feature slot — not a promotion of rumors to one.

### Auto-refresh (transfers)
- **`com.seriea-pipeline.transfer-refresh` plist** (`deploy/launchagents/`, daily 06:00, `RunAtLoad: false`) runs `scripts/data/refresh_transfers.py` → scrapes confirmed + market values + rumors. Window-gated (summer 06-01→09-05, winter 01-01→02-05); exits instantly off-window. NOT auto-loaded — load with `launchctl load ~/Library/LaunchAgents/...` when wanted.

### `data/external/injuries/injuries_YYYY-MM-DD.parquet`
- **Weekly snapshots** (usually Friday)
- ~63 players per snapshot
- Used as match-day context (key player out, squad thinness)

---

## 6b. World Cup 2026 — data/worldcup/

Standalone international engine (added 2026-06-09, branch `feat/worldcup-2026`).
**Completely separate from the Serie A club model** — own data, own model, own
artifacts. Code: `scripts/worldcup/` (engine, simulate, backtest,
generate_predictions). UI: `/worldcup` page, `/api/worldcup` +
`/api/worldcup/simulation` endpoints.

| File | What it is | Source / writer | Refresh |
|------|-----------|-----------------|---------|
| `international_results.csv` | ~49.4k A-internationals 1872 → present (date, teams, score, tournament, city, country, neutral). **Contains future fixture rows with NaN scores — filter `home_score.notna()`** (engine.load_results does) | [martj42/international_results](https://github.com/martj42/international_results) raw CSV, master branch | manual `curl` re-download (updated upstream within days of matches) |
| `international_shootouts.csv` | 678 shootout outcomes (winner, first shooter) | same repo | manual, with results.csv |
| `fixtures.json` | All 104 WC2026 fixtures: match_number, stage, group, date_utc, real team names for 72 group games, slot labels (`1A`, `3ABCDF`, `W74`, `L101`) for knockouts. **Slots auto-fill from real results** (`scripts/worldcup/knockout.py` in the refresh loop): complete groups rank via the 2026 rules, thirds via Annex C once all 12 groups close, W/L slots once the feeder match is decided (level ET-inclusive score ⇒ shootouts.csv). A filled side keeps its original label in `slot_home`/`slot_away` and is never touched again | fixturedownload.com feed + Wikipedia bracket, cross-checked | automatic during the tournament; manual edits only if the filler reports an Annex C error |
| `format_spec.json` | Official 2026 format: advancement rules, group + third-place tiebreakers (verified vs FIFA Regulations Art. 13 — **head-to-head BEFORE overall GD, new in 2026**), R32 pairings, later-round progression, **exact Annex C third-place allocation (all 495 combinations)**, knockout rules, hosts | FIFA regs PDF + Wikipedia, agent-extracted + cross-verified | static for the tournament |
| `predictions.json` | Per-match λs + 1X2 for every fixture with known teams (72 now). Shape mirrors `data/upcoming/predictions.json` so `_build_score_range_projection` derives the full market family at serve time | `python3 -m scripts.worldcup.generate_predictions` | re-run after results.csv refresh or fixture updates |
| `simulation.json` | 10k-run Monte Carlo **conditioned on reality**: played group matches keep their ACTUAL score in every sim (banked points, not re-simulated), decided knockouts keep their real winner — `conditioned_on` counts the pins, `results` maps match_number → actual score (knockouts ET-inclusive). Per-team group win/runner-up/best-third/advance, reach-R32→final, champion probabilities; most-likely R32 ties; **`bracket`** = the single most-likely tournament feeding the /worldcup bracket view — greedy group standings + Annex C thirds, then per-KO-match engine predictions (90' 1X2, top scorelines, advance incl. ET/pens), each match carrying `pairing_prob`, top-3 alt ties and win-slot marginals (third place included; `resolved: true` once knockout.py fills names; `played: true` + `actual_score` once the match is real — the walk continues from the actual winner) | same generator (`--sims N`) | with predictions.json |
| `sofascore_results.json` | Same-night final scores of played WC matches (Sofascore): finished-only, team-id joined, oriented as Sofascore reports, ET-inclusive scores + penalties winner via winnerCode. **The bridge for martj42's publish lag** — `knockout._merged_result_lookup` reads results.csv FIRST, this file second, so the result-pinned simulation and the bracket fill see reality the same night; the canonical CSV silently takes over as it catches up. API route when healthy; during API-tier 403 bans, the tournament hub page's `__NEXT_DATA__` (ISR-rendered fresh — the daily-schedule pages are stale prerenders, last-resort only; match pages are data-free shells). Union-merged by event id | `python3 -m scripts.worldcup.sofascore_fetch --results` (in the 2h refresh loop, before knockout fill) | every refresh during the tournament |
| `clubelo_history.json` | Historical squad club-Elo per backtest tournament (WC18/Euro20/WC22/Euro24/Copa24): team → mean club Elo of the squad + coverage_pct, built from Wikipedia squad pages + api.clubelo.com date snapshots (raw cache in `squad_history/`). ClubElo is Europe-only → Copa squads ~45% coverage, Qatar 2022 = 0% (by design, recorded honestly) | `python3 -m scripts.worldcup.squad_history` | static (historical tournaments don't change) |
| `squad_strength_study.json` | **Squad-strength experiment verdict (2026-06-11): NULL.** k-grid over Elo-adjustment strength (z of squad club Elo × coverage) on the fixed production variant: dev picked k=20 over k=0 by Brier 0.0004 (noise), untouched finals said k=20 is WORSE (Brier +0.0005, ECE +0.012). Club-based squad strength is REDUNDANT with international Elo — do not re-run this hunt without a materially different signal (e.g. global-coverage Transfermarkt history) | `python3 -m scripts.worldcup.backtest --squad-study` | re-run only with new signal design |
| `model_metadata.json` | **The live source for every WC model performance number.** Variant-selection backtest: dev = WC18+Euro20 (selection ONLY), final = untouched WC22/Euro24/Copa24 group stages. Holds the dev grid (GLM / DC / ensembles), selected variant (GLM+Dixon-Coles geometric ensemble), final accuracy/Brier/skill/ECE vs base-rate + higher-Elo + plain-GLM references, gate verdict | `python3 -m scripts.worldcup.backtest` | after any engine/data change — **never quote WC numbers from docs, read this file** |
| `predictions_archive.json` | Pre-kickoff snapshots (Track Record join shape): **last write BEFORE kickoff wins** (graded probabilities = closing deployment incl. odds/lineups/news), immutable from kickoff (writer refuses + grading double-checks archived_at < kickoff). Carries deployed AND pure-model probabilities + frozen market view. Atomic writes; corrupt file quarantined, never silently rebuilt | generator (every refresh until kickoff) | automatic |
| `international_goalscorers.csv` | ~47.6k goal events (date, teams, **team = beneficiary**, scorer, minute, own_goal, penalty). **A SAMPLE, not a ledger**: whole matches lack scorer rows in 2024-26 (~55-80% goal coverage) and it lags results.csv by ~2 months. Use for who-scores propensity (shares); NEVER for goal totals. OG rows name the opponent's player — `players.load_goalscorers` drops them | same martj42 repo | manual, with results.csv |
| `squads.json` | Final 2026 squads: 48 teams × 23-26 players (1,246 total) with position/club/caps/goals, **keyed by fixture display names** | Wikipedia '2026 FIFA World Cup squads' wikitext, agent-parsed | manual if squads change (injury replacements) |
| `player_predictions.json` | Per-match anytime-goalscorer candidates (top 5/side, squad-filtered) + golden-boot expected-goals top 20. **Gate-enforced at generation**: a failing backtest writes an empty layer | `python3 -m scripts.worldcup.players` | after results/squads refresh |
| `player_model_metadata.json` | **Live source for player-model numbers.** Brier lift vs equal-share baseline on 5,280 player-matches (WC22/Euro20/Euro24/Copa24 holdouts; α selected on WC18 only), calibration, gate verdict | `python3 -m scripts.worldcup.players --backtest` | after any player-model change |
| `sofascore_intl_player_stats.parquet` | Per-player-match stats, **kept fresh by the WC refresh loop** (`sofascore_fetch --refresh-stats` appends played WC matches; scraped_at through the current matchday): as of 2026-07-13, **34,583 rows × 22 cols, 1,121 unique matches**, dates 2023-09 → live. Of these, **5,142 rows / 101 matches are WC-2026** (date ≥ 2026-06-11) with real per-match `goals` — this is the source for the LIVE golden-boot leaderboard (`players.actual_scorers`, deduped on (match_id, norm_name_sorted)). Cols: team, match_id, date, tournament, opponent, scores, player_name, `norm_name` (NFKD+lower), `norm_name_sorted` (token-order-proof — Korean names are reversed on Sofascore), position, **started, minutes, shots, shots_on_target, goals, rating**, scraped_at. shots/SoT/goals coalesced to 0 for played players (Sofascore omits zero stats); NaN = unused sub | Sofascore `www.sofascore.com/api/v1` proxy (`team/{id}/events/last/0` + `event/{id}/lineups`; api.sofascore.com unreachable from this network); seeded by a 2026-06-10 one-shot scrape, then live-appended | refreshed every WC matchday by the refresh loop |
| `sofascore_intl/` (dir) | Raw per-team JSONs behind the parquet: 48 files keyed by ASCII team name, each with resolved Sofascore team_id, scraped_at, source_url, 10 matches with full lineup player lists (raw nulls preserved) | same scrape | with the parquet |
| `transfermarkt_values.json` (+ `transfermarkt/` dir) | Current market value for 1,247 squad players, 48/48 teams, 94.4% name-match to squads.json (worst: Jordan 58%, IR Iran 69% — transliteration). `_meta` key holds provenance — its `players` field is a COUNT, skip it when iterating teams. Tournament total €17.26bn; France top €1.52bn | Transfermarkt national-team squad pages (2026-06-10 scrape) | one-shot; re-scrape near roster changes |
| `clubelo_raw.csv` + `clubelo_squad_strength.json` | Club-strength Elo for each squad player's club: 825/1,246 matched (66.2% — **European clubs only**; Saudi/Qatar/Jordan ≈4%). Per team: mean/median club Elo, coverage_pct, per-player elo (null if unmatched). **Gate any use on coverage_pct** — low-coverage means are 1-3 players | api.clubelo.com one-call CSV (2026-06-10) | manual re-pull |
| `scrape_model_metadata.json` | **Live source for scrape-derived gate verdicts.** Walk-forward inside the scraped window: starter-dampening (presence mode deployed, +2.5%, PASSED; minutes mode +3.3% but warm-up-rotation domain shift documented) and shots-floor (−7.4%, FAILED — not shipped) | `python3 -m scripts.worldcup.players --validate-scrape` | after a re-scrape |
| `team_context.json` | Display-only per-team context for the UI: squad_value_eur + mean_club_elo (only where clubelo coverage ≥50%) | written by the player generator | with player_predictions.json |
| `market_odds.json` | Bookmaker 1X2 per fixture (decimal odds + de-vigged implied probs), keyed by match_number. Join is pair-based with ±1-day date sanity. **Feeds the VALIDATED market blend** (see model_metadata.json market_blend) | `python3 -m scripts.worldcup.sofascore_fetch --odds` (www.sofascore.com/api/v1 proxy) | re-run before each matchday — odds sharpen toward kickoff and the blend inherits that |
| `historical_odds.json` | Closing 1X2 odds for the 5 backtest tournaments' group stages: 190/192 matches (48+36+48+36+22). Powers the blend validation: w=0.4 model / 0.6 market selected on DEV, beat the pure model on untouched finals (Brier 0.5773 vs 0.5884); **pure market alone was 0.5758 — the model adds picks, not pricing**. Validation used CLOSING odds; deployment uses current odds (sharper near kickoff) | `sofascore_fetch --historical-odds` | static |
| `track_record.json` | Live predicted-vs-actual scoreboard: pick hits, system/market/base Brier (like-for-like on the odds-covered subset), per-match grades. **KO matches graded on the reconstructed 90' result** (pens ⇒ draw; ET goals subtracted via goalscorers minute column; incomplete scorer coverage ⇒ skipped, counted in n_skipped_ko_coverage). Snapshots stamped at/after kickoff are never graded | `python3 -m scripts.worldcup.grading` (in the refresh loop) | every refresh |
| `player_availability.json` (+ `availability_study.json`) | Team news per fixture in a 7-day horizon: expected XI, out/doubtful (Sofascore missing-players), value-weighted λ factors (ALPHA=0.45, bounds 0.85-1.15 — mechanical, study file is directional evidence only, never read by production) | `python3 -m scripts.worldcup.availability` | every refresh |
| `combos_archive.json` | Pre-kickoff **combo ticket** snapshots (safe/favorites/value accumulators shown on /worldcup), keyed by `tier\|first_leg_kickoff`. Same protocol as predictions_archive: **last write BEFORE the first leg kicks off wins**, immutable after (writer refuses + grader double-checks archived_at < first kickoff). Legs are self-contained for cold grading (teams, stage, picks, prob, quoted odds, edge, EV) | `python3 -m scripts.worldcup.combos` (in the refresh loop) | every refresh until each ticket's first kickoff |
| `combo_record.json` | Settled combo tickets graded vs 90' outcomes (KO via grading.py reconstruction; a ticket grades only when EVERY leg settles, otherwise pending — never as a miss). Per tier: hit rate vs **expected_hit_rate** (mean promised combined prob — the calibration bar) + flat-1u ROI at archived quoted odds. Served under `combos` in /api/worldcup/record | same module, after archiving | every refresh |
| `refresh_state.json` | Matchday-loop state (odds fetch age gating, last run). Atomic writes, fail-soft reads — corruption self-heals to {} | scripts/worldcup/refresh.py | every run |
| `confirmed_lineups.json` | Per-fixture lineup feed: starters + bench (published ~1h pre-kickoff, `confirmed` flag) PLUS `missing` per side — Sofascore missing-player list with type (`missing`/`doubtful`) and reason (Injury/Suspension/Other via MISSING_REASON_MAP). When confirmed: scorer shares override presence-damping (starter ×1.0, bench ×0.35, absent ×0.05 — mechanical minute expectations) | `sofascore_fetch --lineups` on match days, then re-run availability + players generators | every matchday (refresh loop runs it 2-hourly in-window) |
| `player_availability.json` | **Team-news layer, keyed by match_number (7-day horizon)**: per side an expected XI ("who people think plays" — top-11 by starts over last 5 internationals, flips to the confirmed XI when published), out/doubtful lists with reasons + Transfermarkt values, and clamped lambda factors (position-split market-value share of absent expected starters × ALPHA). generate_predictions applies the factors to MODEL lambdas BEFORE the market blend (books already price news — no double counting); web/app.py serves the block as `team_news` | `python3 -m scripts.worldcup.availability` (after `--lineups`) | every refresh-loop run; rebuild before regenerating predictions |
| `availability_study.json` | **Live source for the availability adjustment's honesty numbers**: absence-value-share vs goals-shortfall OLS on competitive internationals in the scrape window (n, slope, SE, sign verdict). The slope is rotation-diluted — it bounds ALPHA from below; never quote adjustment strength from docs, read this file | `python3 -m scripts.worldcup.availability --study` | after a re-scrape |

**FBref note:** club-season per-90s were attempted (2026-06-10) and are NOT available — Cloudflare JS challenge blocks all non-browser access, mirrors dead. No fbref artifacts exist; do not trust any claim of FBref-derived WC player data.

**Team-name trap** — fixtures use FIFA display names, results.csv uses its own:
`USA→United States`, `Korea Republic→South Korea`, `IR Iran→Iran`,
`Côte d'Ivoire→Ivory Coast`, `Cabo Verde→Cape Verde`, `Türkiye→Turkey`,
`Czechia→Czech Republic`, `Congo DR→DR Congo`. Mapping lives in
`scripts/worldcup/engine.py:FIXTURE_TO_CANON` (`canon_team()`); plain
"Congo" in the dataset is Congo-Brazzaville — a different team, never merge.
The generator hard-fails if any of the 48 teams doesn't resolve.

**Model in one line:** World-Football-Elo (eloratings.net convention) over the
full history → Poisson GLM `log λ = b0 + b_diff·(Δelo/100) + b_home + b_friendly`
(time-decay weighted, 2010+) → score grid → markets via
`scripts.betting.extended_markets`. Host advantage applies only when a team
plays in its own country (city→country map in `simulate.country_of_city`).

---

## 7. Cross-source mapping — match_id_mapping.parquet

**File:** `data/parsed/match_id_mapping.parquet` (NEW today)
**Rows:** 7,930 (all SA matches)
**Columns:** `match_id` (canonical), `fbref_hash`, `sofascore_id`, `understat_id`, `match_date`, `home_team`, `away_team`, `season`, `league`

### Why it exists
- `matches.parquet` uses `2025-08-23_Genoa_Lecce` format (date+teams)
- FBref files use 8-char hex hash (`0078428e`)
- Sofascore uses numeric ID (`13981421`)
- Understat uses its own numeric ID (`29844`)

**Without this mapping, you can't join FBref player_stats + Sofascore shotmap + Understat xG onto the same match.**

### Coverage
- `fbref_hash` found: 3,370/7,930 (42.5%) — post-2017 matches
- `sofascore_id` found: 3,370/7,930 (42.5%)
- `understat_id` found: 3,365/7,930 (42.4%)
- **Pre-2017 matches:** no external source has them indexed, only exist in matches.parquet

### How it's built
`scripts/data/build_match_id_mapping.py`:
1. Reads all `data/raw/html/*/fixtures.html` → (home, away, date) → fbref_hash
2. Reads all `data/external/sofascore/fixtures_YYYY_YYYY.json` → sofascore_id
3. Reads `data/external/understat/matches_xg.parquet` → understat_id
4. Joins all three against matches.parquet on (home, away, date)

### Auto-refresh
Now runs as Step 8 in the weekly orchestrator (added today).

---

## 8. Auto-refresh infrastructure

### Level 1 — Daily (morning/evening pipelines)
- `morning.plist` @ 08:00, `evening.plist` @ 20:00
- Calls `scheduler.py once` → `run_full_pipeline.py`
- Does: matches.parquet update + injuries + features rebuild (if >24h stale) + predictions
- **Bottleneck:** Odds API quota — if exhausted, pipeline aborts early

### Level 2 — Weekly (NEW)
**`com.seriea-pipeline.weekly-data-refresh.plist`** (Monday 04:00)

Runs `scripts/pipeline/refresh_weekly_data.py` which does 13 steps:

| # | Step | Source | Output |
|---|------|--------|--------|
| 1 | FBref fixtures.html refresh | botasaurus → fbref.com | `data/raw/html/2025_2026/fixtures.html` |
| 2 | FBref missing match HTMLs | `scrape_fbref_missing.py` — **run it visible; `--headless` cannot pass Cloudflare Turnstile** (measured 2026-07-16: headless = a 27 KB wall, visible = a 426 KB report in ~6s). The weekly job passes `--headless`, probes one page, reports, and exits 0. | `data/raw/html/2025-2026/{fbref_8hex}.html` |
| 3 | Parse player_stats | `parse_all_player_stats --season 2025-2026 --append` | player_stats.parquet |
| 4 | Parse lineups | `parse_all_lineups --season 2025-2026 --append` | lineups.parquet |
| 5 | Parse events | `parse_all_events --season 2025-2026 --append` | events.parquet |
| 6 | Parse goalkeeper_stats | `parse_all_goalkeeper_stats --season 2025-2026 --append` | goalkeeper_stats.parquet |
| 7 | Parse shots — **WATCH, not a required step** (2026-08-25) | `parse_all_shots --season CURRENT_SEASON --append` | shots.parquet. Still exits 1 — FBref removed `shots_all` in 2025-26 and it is still absent from all 8 cached 2026-27 reports (re-measured 2026-08-25). Its result is now recorded in `watches`, NOT `results`, so it no longer gates the job's exit code; it ran in the required loop before, which made `refresh_weekly_data` return 1 every single week on a dead upstream it cannot act on. Still executed, because a True here is the only signal that FBref restored the table. |

> ⚠️ `--season` is **required** on all five parsers — there is no all-seasons
> default. `refresh_weekly_data.py` passes `--season CURRENT_SEASON`; the bare
> form shown here before 2026-07-16 was never what the job ran, and running it
> would rewrite the legacy 2024-25 `player_stats` slice (see that file's
> `--season` comment). Only the named season is parsed.
| 8 | Sofascore refresh | `scrape_sofascore.py --season 2025-2026` | 4 sofascore parquets |
| 9 | Understat refresh | `scrape_understat_xg()` + `parse_all_understat` | understat/matches_xg.parquet |
| 10 | Referee refresh | `scrape_all_referee_assignments` | referee_assignments.parquet |
| 11 | Weather backfill | `fetch_weather_for_matches` w/ new home_team fallback | weather.parquet |
| 12 | match_id mapping rebuild | `build_match_id_mapping.py` | match_id_mapping.parquet |
| 13 | **Sofascore→FBref fallback** | `fallback_sofascore_to_fbref.py` (NEW) | fills player_stats, lineups, events from Sofascore when FBref fails |

**All 13 steps are independent try/except** — one failure doesn't kill the rest.

Logs: `logs/launchd-weekly-data-refresh.log` + `logs/weekly-data-refresh.log`

Manual trigger:
```
launchctl kickstart gui/$(id -u)/com.seriea-pipeline.weekly-data-refresh
```

---

## 9. Fallback matrix (Plan A / B / C)

| Source | Plan A | Plan B | Plan C | Rating |
|--------|--------|--------|--------|--------|
| matches.parquet | Sofascore daily | football-data.org | FBref fixtures | ✅ Robust |
| features_serie_a | build_features | ❌ none | ❌ none | ⚠ Single-source |
| player_stats | FBref | **Sofascore fallback (NEW)** | ❌ | ✅ 2 sources |
| lineups | FBref | **Sofascore fallback (NEW)** | ❌ | ✅ 2 sources |
| events | FBref scorebox (goals/reds only) | **Sofascore incidents (full timeline)** | ❌ | ✅ 2 sources |
| goalkeeper_stats | FBref | ❌ missing | ❌ | ⚠ Single-source |
| shots (FBref) | ❌ deprecated | Sofascore shotmap_stats | Understat player-xG | ⚠ Plan A gone |
| sofascore/* | API | JSON dumps cache | Last parquet | ✅ 3 levels |
| understat | Selenium | Requests HTML | Per-season JSONs | ✅ 3 levels |
| weather | Open-Meteo + venue | **home_team→city fallback (NEW)** | ❌ | ⚠ Partial |
| referee | whoscored | ❌ missing | FBref scorebox (indirect) | ⚠ 1.5 sources |
| match_id_mapping | derived | N/A | N/A | ✅ |
| Odds API | The Odds API | ❌ missing | football-data.org partial | ❌ Single-source |

**Score:** 6 of 13 sources have ≥2 working fallbacks. 4 sources are single-source. 1 (Odds API) is broken with no Plan B.

---

## 10. What's broken or partial

### ❌ BROKEN
- **Odds API quota:** 0/100,000 until May 1 reset. No bets will fire until quota resets.
- **shots.parquet (FBref):** FBref removed `shots_all` tables from 2025-26 HTMLs. Use `sofascore/shotmap_stats.parquet` instead. Downstream code may still read shots.parquet — needs migration.

### ⚠ PARTIAL / KNOWN GAPS
- **FBref HTMLs:** 70 of 330 matches post-Feb-13 couldn't be scraped (Cloudflare). Sofascore fallback covers these.
- **goalkeeper_stats:** No Sofascore fallback yet. 79% 2025-26 coverage.
- **Referee pre-2017:** No scraper source exists. 48% of ALL SA matches have NaN referee.
- **weather 25%:** Open-Meteo timeouts leave ~25% of 2025-26 matches with NaN temp despite retries.
- **features_serie_a.parquet:** No last-known-good fallback if `build_features()` crashes. A single bug in any of 38 plugins can wipe the parquet.
- **No failure alerts:** If weekly refresh fails Monday 04:00, nothing tells you. Data rotted for 50 days before detected in the April audit.
- **Corners odds not fetched:** A real Pinnacle corners market exists (`alternate_totals_corners`, lines 8.0–11.5) but `odds_fetcher.py` does not pull it — corners predictions are display-only, never bet. Cards have no market at all. See §11.10 "Odds API market availability" + [[project_jun01_betting_audit]].

### ✅ STRONG
- **matches.parquet** (3-source failover)
- **Sofascore** (3-level caching)
- **Understat** (3-level fallback including JSON dumps)
- **match_id_mapping** (derived, auto-rebuilt)
- **Current-season basic stats** (100% fill rate after today's fixes)

---

*The rest of this document is the per-file column-level audit. Use Ctrl-F to search.*

---

---

---

## 11. Column glossary — what each feature family MEANS

The `features_serie_a.parquet` table has 1,059 columns. Grouped by semantic family below. **All rolling features use `shift(1)` before aggregation — values are from matches BEFORE the current match only, no leakage.**

### 11.1 Primary keys (11 cols)

| Column | Type | Meaning |
|--------|------|---------|
| `match_id` | str | Primary key: `YYYY-MM-DD_Home_Away` (or FBref 8-char hash for some rows) |
| `home_team`, `away_team` | str | Normalized team names (via `config.team_names.normalize_team`) |
| `match_date` | datetime | Match date (no time component usually) |
| `home_score`, `away_score` | float | Final score (int semantically) |
| `result` | str | 'H' / 'D' / 'A' (home win / draw / away win) |
| `season` | str | e.g. '2025-2026' (hyphen-separated, Aug-May) |
| `league` | str | 'serie_a' or 'premier_league' |
| `matchweek` | float | Round number within the season (1-38) |
| `ht_result` | str | Half-time result (same H/D/A coding) |

### 11.2 Elo ratings (15 cols)

| Column | Meaning |
|--------|---------|
| `home_elo`, `away_elo` | **PRE-MATCH** Elo rating. Updated chronologically after each match. Season-boundary regression: `0.75 * old_elo + 0.25 * ELO_INITIAL`. Newly promoted teams start at `ELO_PROMOTED` (below average). |
| `elo_diff` | `home_elo - away_elo` (positive = home favored) |
| `elo_diff_log` | `sign(elo_diff) * log(1 + abs(elo_diff))` — log-scaled for non-linear models |
| `home_elo_momentum`, `away_elo_momentum` | Change in Elo over last 5 matches (positive = improving) |
| `elo_momentum_diff` | `home_elo_momentum - away_elo_momentum` |
| `home_form_elo_blend`, `away_form_elo_blend` | Weighted combination of Elo + recent form points (captures "hot team overperforming rating") |
| `elo_form_blend_diff`, `elo_form_disagreement`, `form_elo_signal` | Derived signals from form vs. Elo divergence |

**Source:** `features/strength.py::add_elo_ratings()` called via `Step11Elo` in the feature pipeline.
**Leakage:** Safe — Elo at row N uses only matches 0..N-1.

### 11.3 Poisson xG model (9 cols)

| Column | Meaning |
|--------|---------|
| `poisson_home_xg`, `poisson_away_xg` | Expected goals from a simple Poisson model: `team_attack × opponent_defense × league_avg_goals`. NOT from shot-level xG — this is a team-rate estimate. |
| `poisson_prob_H`, `poisson_prob_D`, `poisson_prob_A` | P(outcome) derived from bivariate Poisson(home_xg, away_xg) |
| `poisson_over_1_5`, `poisson_over_2_5`, `poisson_over_3_5` | P(total goals > N) from the Poisson model |
| `poisson_btts` | P(both teams score) |

**Source:** `features/derived.py` via `Step08DerivedTeamFeatures` or `Step26DerivedMatchLevel`.

### 11.4 Rolling team stats (76 cols)

Pattern: `{home|away}_roll_{N}_{stat}` where N ∈ {3, 5, 10} and stat ∈ {goals_scored, goals_conceded, shots_on_target, corners, fouls, yellow_cards, red_cards, points, clean_sheet, win_rate}.

Also variance versions: `_std` suffix (e.g., `home_roll_5_goals_scored_std`).

| Pattern | Meaning |
|---------|---------|
| `home_roll_5_goals_scored` | Mean goals scored by home team over last 5 matches (excluding current) |
| `home_roll_10_points` | Total league points earned in last 10 matches |
| `home_roll_5_points_std` | Standard deviation of points over last 5 matches (measures consistency) |
| `home_goals_scored_trend`, `home_points_trend` | Slope of the last-N trend (positive = improving) |

**Source:** `features/rolling.py::add_rolling_features()` — uses `.shift(1).rolling(N).mean()` for leakage safety.

### 11.5 Venue-specific rolling (24 cols)

Same as rolling above, but limited to matches played at home / away: `home_venue_roll_{N}_{stat}`. Captures home/away advantage specific to each team.

### 11.6 Sofascore rolling (327 cols — the big one)

Pattern: `home_ss_roll_{stat}`, `away_ss_roll_{stat}`, `ss_diff_ss_roll_{stat}`.

Stats include: `xg`, `xgot`, `xa`, `goals`, `total_shots`, `shots_on_target`, `possession`, `accurate_passes`, `tackles`, `interceptions`, `duels_won`, `aerial_won`, `big_chances_created`, `key_passes`, `ball_recoveries`, `dispossessed`, `errors_to_shot`, `progressive_carries`, `final_third_entries`, `counter_xg_pct`, `set_piece_xg_pct`, `first_half_xg_share`, `corner_xg_share`, `penalty_xg_share`, ~100 more.

**Source:** `features/sofascore_features.py` via `Step19bSofascore`.
**Normalization:** Some stats are pct (0-1), some are counts. Check `Step19b2SofascoreIndices` for `ss_idx_*` normalized indices.
**Leakage:** Rolling aggregation uses `.shift(1)` — safe.
**Gap:** Pre-2022 seasons have 82% NaN for these (Sofascore historical coverage starts 2022-2023).

### 11.7 FBref rolling (99 cols)

Pattern: `home_fb_roll_{stat}`, `away_fb_roll_{stat}`, `fb_diff_{stat}`.

Similar to Sofascore rolling but from FBref match reports: `goals`, `xg`, `npxg`, `xg_assist`, `sca` (shot-creating actions), `gca` (goal-creating), `pass_accuracy`, `progressive_passes`, `touches`, `carries`, `defense_tackles_won`, `misc_ball_recoveries`, `aerial_win_rate`, etc.

**Source:** `features/fbref_features.py` via `Step20Fbref`.

### 11.8 Understat (49 cols)

Pattern: `home_us_team_{stat}`, `away_us_team_{stat}`, `us_xg_diff`, `us_xa_diff`.

Understat-specific xG metrics: `us_team_xg`, `us_team_npxg`, `us_team_xa`, `us_team_xg_chain`, `us_team_xg_buildup`.

**Source:** `features/understat_features.py` via `Step19UnderstatXg`.
**Independence:** Understat uses a different xG model than Sofascore — having both enables cross-validation.

### 11.9 H2H (22 cols)

| Pattern | Meaning |
|---------|---------|
| `h2h_matches_played` | Total head-to-head meetings (capped at 7 recent per config) |
| `h2h_home_wins`, `h2h_away_wins`, `h2h_draws` | Historical outcome counts |
| `h2h_draw_rate`, `h2h_home_win_rate`, `h2h_away_win_rate` | Uninformative prior = 1/3 each when no history |
| `h2h_goals_avg`, `h2h_btts_rate`, `h2h_over_2_5_rate` | Goal patterns in head-to-head |
| `h2h_current_venue_*` | Same stats but limited to matches at this specific venue |
| `manager_h2h_home_winrate`, `manager_h2h_matches` | Head-to-head between current managers (not just clubs) |

**Source:** `features/h2h.py` via `Step10H2H`.
**Leakage safety:** Uses `match_date` cutoff — only H2H meetings BEFORE current match are counted.

### 11.10 Odds + market (69 cols)

| Pattern | Meaning |
|---------|---------|
| `odds_B365H/D/A` | Bet365 pre-match odds for home/draw/away |
| `odds_AvgH/D/A` | Bookmaker average (roughly 20 books) |
| `odds_MaxH/D/A` | Highest odds across bookmakers |
| `odds_PSH/D/A` | Pinnacle opening odds (sharpest book) |
| `odds_PS_close_H/D/A` | Pinnacle closing odds (most accurate market price) |
| `odds_Avg_over25/under25`, `odds_B365_over25/under25` | Over/Under 2.5 goals |
| `odds_AH_line`, `odds_AH_close_line` | Asian handicap main line |
| `market_home_prob`, `market_draw_prob`, `market_away_prob` | Implied probabilities (overround-adjusted) |
| `pinnacle_home_prob/draw_prob/away_prob` | Pinnacle-only implied probs (sharpest) |
| `sharp_soft_home_div` | Divergence between sharp (Pinnacle) and soft books — positive = value opportunity |
| `line_vel_*` | Line movement velocity (from snapshots; 0 for historical) |
| `overround` | Total implied probability sum — 1.0; measures book margin |

**Source:** `features/odds.py` + `features/market.py` via `Step30Odds`, `Step30bDerivedOdds`, `Step31MarketData`.
**Gap:** `odds_PS_close_*` and `odds_AH_line` are ~65% and ~28% filled all-time (only recent seasons).

#### Odds API market availability (probed 2026-06-01 against historical SA events)

What The Odds API actually offers for Serie A, by endpoint — **memorise this; invalid market×endpoint combos STILL cost credits (422 INVALID_MARKET)**:

| Market | Endpoint | Books available | Status in this repo |
|---|---|---|---|
| `h2h` (1X2) | `/odds/` bulk + `/historical/.../odds/` | many (incl. Pinnacle) | Fetched, used |
| `totals` (O/U goals) | bulk + historical | many | Fetched, used |
| `spreads` (AH goals) | bulk + historical | many | Fetched, used |
| `btts`, `double_chance`, `team_totals`, `draw_no_bet`, `alternate_totals/spreads` | `/events/{id}/odds/` per-event only | varies | Fetched (in `PER_EVENT_MARKETS`); only DC real-line is bet |
| **`alternate_totals_corners`** | per-event + historical per-event | **Pinnacle ONLY (6/6 SA events), full line ladder 8.0–11.5** | **NOT fetched** by `odds_fetcher.py`. Real sharp market EXISTS. |
| `totals_corners` (standard corner line) | — | — | **INVALID_MARKET (422)** — only the `alternate_` form works |
| **`alternate_totals_cards` / `totals_cards`** | — | **0 books / INVALID_MARKET** | No card market exists for SA at all |

**Implication for corners/cards models** (see [[project_jun01_betting_audit]] + CLAUDE.md "Match Intelligence shows corners/cards"): a real, *sharp* (Pinnacle) corners market exists and could be wired into `odds_fetcher.py` per-event, but a 2026-06-01 viability probe showed Pinnacle's corners line open→close move is ~half systematic drift (mean +0.065 toward Under) / half match-specific (std 0.066, ratio 1.02) — too thin an exploitable residual for the existing zero-skill corners model to beat. Corners build is **parked** (plan in `.plans/corners-model-plan.md`). **Cards are dead — no market to bet against.** The corners/cards prediction files (`data/upcoming/{corners,cards}_predictions.json`) are still written and feed the dashboard as *informational display only* — they are NOT bet on.

### 11.11 Weather (11 cols)

| Column | Unit | Meaning |
|--------|------|---------|
| `weather_temperature_2m_max/min/mean` | °C | Match-day temperature |
| `weather_apparent_temperature_max/min` | °C | "Feels like" temperature |
| `weather_precipitation_sum`, `weather_rain_sum`, `weather_snowfall_sum` | mm | Total precipitation |
| `weather_wind_speed_10m_max`, `weather_wind_gusts_10m_max` | km/h | Wind |
| `weather_wind_direction_10m_dominant` | degrees | 0-360° |
| `weather_relative_humidity_2m_mean` | % | |

**Source:** `scraper/weather.py` → Open-Meteo archive API. Joined on `match_id` via `Step29Weather`.
**Fallback (added today):** If `venue` column is NaN, maps `home_team → city` via `TEAM_TO_CITY` dict.

### 11.12 Referee (18 cols)

| Column | Meaning |
|--------|---------|
| `referee` | Referee name (string) |
| `ref_matches_officiated` | Total matches this ref has officiated (in our data) |
| `ref_avg_yellows`, `ref_avg_reds`, `ref_avg_fouls` | Career averages — approximates strictness |
| `ref_strictness_score` | Composite 0-1 score (higher = stricter) |
| `ref_strictness_trend` | Is this ref getting stricter over time? |
| `ref_home_bias`, `ref_home_cards_bias` | Historical card rate home vs away for this ref |
| `ref_home_team_cards`, `ref_away_team_cards` | Cards this ref has given to THIS home/away team historically |
| `ref_vs_home_team_bias`, `ref_vs_away_team_bias` | Ref-vs-team interaction |
| `ref_last_match_reds`, `ref_last_match_cards` | Signal of current ref "temperament" |
| `ref_big_match_cards` | Card tendency in high-pressure matches |
| `ref_regression_signal` | Mean-reversion signal (was this ref recently unusual?) |

**Source:** `features/referee.py` via `Step13Referee`.
**Fallback:** Ref name backfilled into `matches.parquet` from `referee_assignments.parquet`.

### 11.13 League position + zone flags (17 cols)

| Column | Meaning |
|--------|---------|
| `home_league_pos`, `away_league_pos` | Table position at start of match (1-20) |
| `home_league_points`, `_gd`, `_goals_for`, `_wins`, `_draws`, `_losses` | Cumulative season stats |
| `home_in_relegation_zone` | 1 if position ≥ 18 |
| `home_in_cl_zone` | 1 if position ≤ 4 (Champions League spots) |
| `home_in_el_zone` | 1 if position in [5, 6] (Europa League spots) |
| `home_in_title_race` | 1 if position ≤ 2 |
| `league_position_diff` | home_pos - away_pos (negative = home higher) |
| `home_position_momentum`, `away_position_momentum` | Places gained/lost over last 5 matches |
| `home_points_to_cl_zone`, `home_points_to_relegation` | Distance to qualification / relegation |

**Source:** `features/league_position.py` via `Step21LeaguePosition`.
**Leakage safety:** Position computed from matches played BEFORE current — uses `shift()` pattern.

### 11.14 Strength ratings (13 cols)

| Column | Meaning |
|--------|---------|
| `home_attack_strength` | `team_goals_scored_per_game / league_avg_goals_scored` — league-relative attack rating (1.0 = league average) |
| `home_defense_strength` | `league_avg_goals_conceded / team_goals_conceded_per_game` — higher = better defense |
| `home_xg_attack_strength`, `home_xg_defense_strength` | Same but using xG instead of goals (less noise) |
| `attack_strength_diff`, `defense_strength_diff` | home - away versions |
| `home_attack_vs_away_def`, `away_attack_vs_home_def` | Matchup-specific (attack × opponent defense weakness) |
| `attack_defense_mismatch` | Higher when one team dominates in both directions |

**Source:** `features/strength.py::add_strength_ratings()` via `Step05StrengthRatings`.

### 11.15 Squad + injury + suspensions (16 cols)

| Column | Meaning |
|--------|---------|
| `home_key_players_available`, `away_key_players_available` | 1 if top scorers + regular starters all present, 0 otherwise (92% NaN — requires lineup data) |
| `home_top_scorer_played` | 1 if this season's top scorer is in the starting XI |
| `home_squad_rotation` | Number of changes from last match's XI |
| `home_suspended_count`, `away_suspended_count` | Players unavailable due to yellow-card accumulation |
| `home_at_risk_count` | Players on 4 yellow cards (one more = suspension) |
| `home_total_yellows` | Team's season yellow-card total |

**Source:** `features/suspensions.py`, `features/lineup_xg.py` via `Step24Suspensions`, `Step19c2LineupXg`.
**Gap:** 92% NaN all-time because lineup data is only strong 2024+ and requires FBref or Sofascore scrape.

### 11.16 Formation (9 cols)

| Column | Meaning |
|--------|---------|
| `home_formation`, `away_formation` | e.g., '4-3-3', '3-5-2' |
| `home_formation_flexibility` | How often this team changes formation (high = adaptable) |
| `formation_matchup_home_rate` | Historical home-win rate for THIS formation-vs-that-formation combination |
| `formation_matchup_draw_rate`, `formation_total_advantage` | From `formation_database.json` |
| `formation_confidence` | How much data supports this matchup estimate |
| `formation_width_mismatch` | Quantifies tactical width mismatch (3-at-back vs 5-at-back) |

**Source:** `features/formation_analysis.py` via `Step25Formations`.

### 11.17 Manager (8 cols)

| Column | Meaning |
|--------|---------|
| `home_manager`, `away_manager` | Current manager name |
| `home_manager_tenure` | Days since appointment |
| `home_manager_is_new` | 1 if tenure < 30 days |
| `home_manager_changed` | 1 if different from last match |
| `manager_h2h_home_winrate`, `manager_h2h_matches`, `manager_h2h_confidence` | Head-to-head between these specific managers |

**Source:** `features/manager.py` via `Step22Manager`.

### 11.18 Rest + congestion (10 cols)

| Column | Meaning |
|--------|---------|
| `home_rest_days`, `away_rest_days` | Days since previous match |
| `rest_advantage` | `home_rest_days - away_rest_days` (capped at ±5) |
| `congestion_asymmetry` | One team mid-week, other not? |
| `home_short_rest` | 1 if home had <4 days rest |
| `home_congestion_3`, `home_congestion_5` | Matches in last 3/5 days (including international) |

**Source:** `features/congestion.py` via `Step23Congestion`.

### 11.19 Draw-specific signals (7 cols)

| Column | Meaning |
|--------|---------|
| `league_draw_rate` | Expanding-window mean draw rate in the league |
| `home_draw_tendency_10`, `home_draw_tendency_5` | Rolling draw frequency for home team |
| `combined_draw_tendency` | Avg of both teams' draw tendency |
| `defense_similarity` | `1 / (1 + |home_def - away_def|)` — similar defenses → higher draw probability |
| `low_scoring_signal` | Avg recent goals conceded (low = defensive matchup) |
| `both_defenses_strong` | min(home_def, away_def) — mutual impermeability |

**Source:** `_add_league_draw_features` in `features/build.py` via `Step27LeagueDrawFeatures`.

### 11.20 Other / derived (~230 cols)

Catch-all for plugin outputs: `_gk_` (GK quality), `psxg` (post-shot xG), `_has_*` (coverage flags), interaction features (Step36 — e.g., `elo_x_form`, `shot_x_xg`), contextual (Step37 — e.g., `is_run_in`, `is_derby`, `is_midweek`).

---

## 12. Join recipes — how to link data across sources

### Recipe 1: Load full match features for a SA match

```python
import pandas as pd
features = pd.read_parquet('data/features/features_serie_a.parquet')
row = features[features['match_id'] == '2025-08-23_Genoa_Lecce'].iloc[0]
# 1,059 columns for that one match, ready to feed any model
```

### Recipe 2: Join Sofascore shot-level events to a Serie A match

Sofascore uses numeric IDs, we use canonical — go through `match_id_mapping.parquet`:

```python
import pandas as pd
matches = pd.read_parquet('data/parsed/matches.parquet')
mapping = pd.read_parquet('data/parsed/match_id_mapping.parquet')
shots = pd.read_parquet('data/external/sofascore/all_shots_with_xg.parquet')

# Join: our match → sofascore_id → sofascore shot rows
m_with_sofa = matches.merge(
    mapping[['match_id','sofascore_id']].dropna(),
    on='match_id'
)
# Sofascore match_id column is numeric; cast for join
shots['match_id'] = shots['match_id'].astype(str)
m_shots = m_with_sofa.merge(
    shots.rename(columns={'match_id':'sofascore_id'}),
    on='sofascore_id'
)
# Now each row has our match info + one shot. Group by match to aggregate.
```

### Recipe 3: Get Understat xG for a match

```python
mapping = pd.read_parquet('data/parsed/match_id_mapping.parquet')
understat = pd.read_parquet('data/external/understat/matches_xg.parquet')
understat['match_id'] = understat['match_id'].astype(str)

match_us = mapping.merge(
    understat.rename(columns={'match_id':'understat_id'}),
    on='understat_id'
)
# match_us has canonical match_id + understat home_xg / away_xg
```

### Recipe 4: Per-player stats for a specific match

As of the **2026-08-26** re-key, `player_stats.parquet` and `goalkeeper_stats.parquet`
are canonical-keyed for every season. Join on the canonical `match_id` — no hash dance
needed:

> ⚠️ This section previously claimed "canonical-keyed for every season" as of the
> 2026-07-17 re-key. That was **false**: 2026-07-17 only re-keyed 2025-26, the season
> that was visibly broken at the time. Seasons **2017-18 → 2023-24 stayed 100%
> hash-keyed** for another 13 months. Nothing raised, because a hash-vs-canonical
> mismatch does not error — `merge(on="match_id")` returns no rows and the feature
> columns come out silently NaN. Cost: `adv_roll5_*` (76 cols), `tagg_roll5_*` (52) and
> player_impact's key-player block (8) were **0% filled for all seven seasons**, i.e.
> every season of the 2017+ training window except the last two. The families that
> aggregate to (season, team) instead of joining per match — `fb_roll_*`, `lineup_*`,
> suspensions — were unaffected, which is exactly why the gap looked like "sparse
> historical data" rather than a bug. Re-keyed by
> `scripts/data/rekey_legacy_hash_ids.py` (idempotent, safe to re-run).

```python
players = pd.read_parquet('data/parsed/player_stats.parquet')
target_match = '2025-08-23_Genoa_Lecce'
stats = players[players['match_id'] == target_match]
```

Only reach for `match_id_mapping.parquet` (`fbref_hash` column) if you are reading an
OLD parquet from before the re-key, or joining to a source still keyed by hash.

### Recipe 5: All matches + odds + results for modeling

```python
matches = pd.read_parquet('data/parsed/matches.parquet')
sa = matches[matches['league']=='serie_a']
training_set = sa[sa['home_score'].notna()].copy()
# Target: result (H/D/A). Features: everything starting with odds_, home_, away_
X = training_set[[c for c in training_set.columns if c.startswith(('odds_','home_','away_'))]]
y = training_set['result']
```

### Recipe 6: Enrich match with weather + referee + injuries

```python
matches = pd.read_parquet('data/parsed/matches.parquet')
weather = pd.read_parquet('data/external/weather.parquet')
referee = pd.read_parquet('data/external/referee/referee_assignments.parquet')

m_w = matches.merge(weather, on='match_id', how='left')
m_w_r = m_w.merge(
    referee[['match_date','home_team','away_team','referee','ref_yellows','ref_reds']],
    on=['match_date','home_team','away_team'],
    how='left', suffixes=('', '_assigned')
)
# If matches.referee is NaN, m_w_r.referee_assigned may have it.
```

### Recipe 7: Find matches with xG from BOTH Sofascore and Understat (for cross-validation)

```python
mapping = pd.read_parquet('data/parsed/match_id_mapping.parquet')
both = mapping[mapping['sofascore_id'].notna() & mapping['understat_id'].notna()]
# 3,365 matches (42.4% of SA all-time, essentially all 2017+)
```

### Recipe 8: Backfill a missing column from a secondary source

Pattern used in today's fixes. Example: `matches.parquet.home_yellow_cards` filled from Sofascore incidents:

```python
import pandas as pd
m = pd.read_parquet('data/parsed/matches.parquet')
inc = pd.read_parquet('data/external/sofascore/match_incidents.parquet')
mim = pd.read_parquet('data/parsed/match_id_mapping.parquet')

# Aggregate cards per Sofascore match
cards = inc[inc['incident_type']=='card']
agg = cards.groupby(['match_id','is_home','incident_class']).size().unstack(fill_value=0).reset_index()

# sofa_id → canonical
sofa_map = dict(zip(mim['sofascore_id'].astype(str), mim['match_id']))

# Fill missing
missing = m[m['home_yellow_cards'].isna() & m['home_score'].notna()]
for idx, row in missing.iterrows():
    # Find sofascore_id via mapping, look up card count
    ...
```

---

## 13. Feature provenance — which pipeline step writes what

Run the feature pipeline: `python3 -m features.build` (entry point: `build_features()`).

| Plugin | Step class | Writes columns matching pattern |
|--------|-----------|----------------------------------|
| pre | `BackfillManagersPlugin` | `home_manager`, `away_manager` (fill if missing) |
| pre | `BackfillRefereesPlugin` | `referee` (fill if missing) |
| 01 | `Step01TeamMatchLog` | team_log dataframe (intermediate) |
| 02 | `Step02RollingStats` | `home_roll_*`, `away_roll_*` (76 cols) |
| 03 | `Step03HomeAwaySplits` | `home_venue_roll_*`, `away_venue_roll_*` (24 cols) |
| 04 | `Step04XgTrends` | xG rolling metrics |
| 05 | `Step05StrengthRatings` | `home_attack_strength`, `home_defense_strength`, `*_xg_*_strength` |
| 06 | `Step06RestDays` | `home_rest_days`, `away_rest_days` |
| 07 | `Step07MomentumStreaks` | momentum streak features |
| 08 | `Step08DerivedTeamFeatures` | team-derived aggregates, form volatility |
| 09 | `Step09PivotToMatchLevel` | pivots team_log back to match-level |
| 10 | `Step10H2H` | `h2h_*` (22 cols) |
| 11 | `Step11Elo` | `home_elo`, `away_elo`, `elo_diff` |
| 12 | `Step12PlayerImpact` | key player features from player_stats |
| 13 | `Step13Referee` | `ref_*` (18 cols) |
| 14 | `Step14TeamAggregates` | season-to-date team summaries |
| 15 | `Step15GkQuality` | `*_gk_*` columns (goalkeeper quality) |
| 16 | `Step16ShotQuality` | shot-derived features |
| 17 | `Step17AdvancedPlayer` | advanced per-player aggregates |
| 18 | `Step18AdvancedShots` | `*_advshot_*` columns |
| 19 | `Step19UnderstatXg` | `home_us_team_*`, `away_us_team_*` (49 cols) |
| 19b | `Step19bSofascore` | `home_ss_roll_*`, `away_ss_roll_*` (327 cols) |
| 19b2 | `Step19b2SofascoreIndices` | `ss_idx_*` normalized 0-1 indices |
| 19c | `Step19cPlayerDepth` | squad depth metrics |
| 19c2 | `Step19c2LineupXg` | `lineup_xg_sum`, `top2_xg_share` |
| 19d | `Step19dMatchPatterns` | match pattern classifications |
| 19e | `Step19eCreativeFactors` | `is_run_in`, `promoted_vs_established`, `matchweek_avg_goals` |
| 19f | `Step19fCaptain` | captain-based features |
| 19g | `Step19gCardTiming` | card-timing features |
| 20 | `Step20Fbref` | `home_fb_roll_*`, `away_fb_roll_*`, `fb_diff_*` (99 cols) |
| 21 | `Step21LeaguePosition` | `home_league_pos`, zone flags |
| 21b | `Step21bLeaguePositionContext` | `points_to_cl_zone`, `points_to_relegation` |
| 22 | `Step22Manager` | `*_manager_*`, `manager_h2h_*` |
| 23 | `Step23Congestion` | congestion features |
| 24 | `Step24Suspensions` | `home_suspended_count`, `at_risk_count` |
| 25 | `Step25Formations` | `formation_*` |
| 26 | `Step26DerivedMatchLevel` | Poisson outputs, `elo_form_blend_diff`, etc. |
| 26b | `Step26bDerby` | `is_derby` |
| 26c | `Step26cGoalFeatures` | `league_avg_goals`, `rolling_goals_diff` |
| 27 | `Step27LeagueDrawFeatures` | `league_draw_rate`, `home_draw_tendency_*`, `matchup_competitiveness` |
| 28 | `Step28Venue` | `travel_distance`, `altitude_diff` |
| 29 | `Step29Weather` | `weather_*` (11 cols) |
| 30 | `Step30Odds` | `odds_*`, `overround` |
| 30b | `Step30bDerivedOdds` | `market_*_prob`, `pinnacle_*_prob` |
| 31 | `Step31MarketData` | `sharp_soft_*_div`, `line_vel_*` |
| 32 | `Step32InjuryImpact` | injury-impact composite |
| 33 | `Step33PPDA` | pressing intensity (passes per defensive action) |
| 34 | `Step34ManagerH2HNoop` | (no-op placeholder for future) |
| 35 | `Step35TransferImpact` | post-window market-value changes |
| 36 | `Step36Interactions` | `elo_x_form`, `disruption_x_elo`, `tenure_x_form`, ~13 interaction features |
| 37 | `Step37Contextual` | `is_midweek`, `is_weekend`, `weather_impact` |
| 38 | `Step38SubstitutionPatterns` | substitution-pattern features |

**Total:** 38 plugins + 2 pre-steps → produces 1,059 columns in `features_serie_a.parquet`.

**Cleanup after plugins:**
1. Drop `Unnamed:` / `level_0_` artifacts from HTML parsing
2. Drop constant (zero-variance) columns
3. Drop columns with >95% null
4. Drop exact-duplicate columns (|r| ≥ 0.999)
5. Sanitize column names (XGBoost rejects `<`, `>`, `[`, `]` → replaced with `_lt_`, `_gt_`, etc.)

---

## PER-FILE DEEP COLUMN AUDIT

*Below is the auto-generated per-file column-level audit. Use Ctrl-F to search for specific columns.*

---

## 1. GROUND TRUTH — matches + results

### `data/parsed/matches.parquet`

_**The foundation.** Every match, final score, basic stats (shots, corners, cards, fouls). Used to settle bets and train models._

- **Format:** Parquet  
- **Size:** 832.9KB  
- **Modified:** 2026-04-21 18:43  
- **Rows:** 15,839  
- **Columns:** 109  
- **Date column:** `match_date` — range 2005-08-13 → 2026-04-20  
- **League distribution:** {'serie_a': 7930, 'premier_league': 7909} (7930/15839 = 50.1% Serie A)  
- **Seasons:** 21 covered — 2005-2006 → 2025-2026  

**Columns (109):**

| # | Column | Dtype | Filled % | NaN % | Unique | Sample | Min | Max |
|---|--------|-------|----------|-------|--------|--------|-----|-----|
| 1 | `home_team` | object | 100.0% | 0.0% | 89 | 'Fiorentina' | — | — |
| 2 | `away_team` | object | 100.0% | 0.0% | 89 | 'Sampdoria' | — | — |
| 3 | `match_date` | datetime64[ns] | 100.0% | 0.0% | 2828 | — | 2005-08-13 | 2026-04-20 |
| 4 | `home_score` | float64 | 100.0% | 0.0% | 10 | median=1 | 0 | 9 |
| 5 | `away_score` | float64 | 100.0% | 0.0% | 10 | median=1 | 0 | 9 |
| 6 | `result` | object | 100.0% | 0.0% | 3 | 'H' | — | — |
| 7 | `season` | object | 100.0% | 0.0% | 21 | '2005-2006' | — | — |
| 8 | `league` | object | 100.0% | 0.0% | 2 | 'serie_a' | — | — |
| 9 | `league_name` | object | 100.0% | 0.0% | 2 | 'Serie A' | — | — |
| 10 | `home_shots_total` | float64 | 99.7% | 0.3% | 43 | median=13 | 0 | 46 |
| 11 | `away_shots_total` | float64 | 99.7% | 0.3% | 34 | median=11 | 0 | 37 |
| 12 | `home_shots_on_target_count` | float64 | 99.7% | 0.3% | 24 | median=5 | 0 | 24 |
| 13 | `away_shots_on_target_count` | float64 | 99.7% | 0.3% | 21 | median=4 | 0 | 20 |
| 14 | `home_fouls` | float64 | 99.7% | 0.3% | 40 | median=12 | 0 | 48 |
| 15 | `away_fouls` | float64 | 99.7% | 0.3% | 39 | median=13 | 1 | 41 |
| 16 | `home_corners` | float64 | 99.7% | 0.3% | 22 | median=5 | 0 | 21 |
| 17 | `away_corners` | float64 | 99.7% | 0.3% | 20 | median=4 | 0 | 19 |
| 18 | `home_yellow_cards` | float64 | 99.7% | 0.3% | 8 | median=2 | 0 | 7 |
| 19 | `away_yellow_cards` | float64 | 99.7% | 0.3% | 10 | median=2 | 0 | 9 |
| 20 | `home_red_cards` | float64 | 99.8% | 0.2% | 4 | median=0 | 0 | 3 |
| 21 | `away_red_cards` | float64 | 99.7% | 0.3% | 4 | median=0 | 0 | 3 |
| 22 | `home_ht_goals` | float64 | 99.3% | 0.7% | 6 | median=0 | 0 | 5 |
| 23 | `away_ht_goals` | float64 | 99.3% | 0.7% | 6 | median=0 | 0 | 5 |
| 24 | `ht_result` | object | 99.3% | 0.7% | 3 | 'H' | — | — |
| 25 | `referee` | object | 76.0% | 24.0% | 212 | 'G. Paparesta' | — | — |
| 26 | `odds_B365H` | float64 | 99.3% | 0.7% | 141 | median=2.2 | 1.06 | 23 |
| 27 | `odds_B365D` | float64 | 99.3% | 0.7% | 83 | median=3.5 | 1.4 | 17 |
| 28 | `odds_B365A` | float64 | 99.3% | 0.7% | 136 | median=3.5 | 1.1 | 41 |
| 29 | `odds_AvgH` | float64 | 99.3% | 0.7% | 938 | median=2.18 | 1.05 | 20.8 |
| 30 | `odds_AvgD` | float64 | 99.3% | 0.7% | 702 | median=3.49 | 1.29 | 14.4 |
| 31 | `odds_AvgA` | float64 | 99.3% | 0.7% | 1586 | median=3.42 | 1.1 | 40.6 |
| 32 | `odds_MaxH` | float64 | 99.3% | 0.7% | 710 | median=2.28 | 1.06 | 35 |
| 33 | `odds_MaxD` | float64 | 99.3% | 0.7% | 603 | median=3.7 | 1.33 | 19 |
| 34 | `odds_MaxA` | float64 | 99.3% | 0.7% | 1140 | median=3.7 | 1.13 | 61 |
| 35 | `odds_Avg_over25` | float64 | 99.3% | 0.7% | 163 | median=1.91 | 1.18 | 2.91 |
| 36 | `odds_Avg_under25` | float64 | 99.3% | 0.7% | 256 | median=1.88 | 1.4 | 4.67 |
| 37 | `match_id` | object | 100.0% | 0.0% | 15839 | '2005-08-27_Fiorentina_Sampdor | — | — |
| 38 | `odds_PSH` | float64 | 64.9% | 35.1% | 906 | median=2.26 | 1.06 | 23.6 |
| 39 | `odds_PSD` | float64 | 64.9% | 35.1% | 657 | median=3.73 | 2.2 | 19 |
| 40 | `odds_PSA` | float64 | 64.9% | 35.1% | 1505 | median=3.47 | 1.13 | 42.9 |
| 41 | `odds_PS_close_H` | float64 | 64.9% | 35.1% | 903 | median=2.27 | 1.04 | 25.8 |
| 42 | `odds_PS_close_D` | float64 | 64.9% | 35.1% | 635 | median=3.7 | 2.2 | 18.5 |
| 43 | `odds_PS_close_A` | float64 | 64.9% | 35.1% | 1405 | median=3.49 | 1.09 | 50 |
| 44 | `odds_B365_over25` | float64 | 32.1% | 67.9% | 75 | median=1.8 | 1.2 | 3 |
| 45 | `odds_B365_under25` | float64 | 32.1% | 67.9% | 76 | median=2.03 | 1.4 | 4.5 |
| 46 | `odds_B365_close_H` | float64 | 32.1% | 67.9% | 119 | median=2.25 | 1.03 | 23 |
| 47 | `odds_B365_close_D` | float64 | 32.1% | 67.9% | 47 | median=3.75 | 2.63 | 19 |
| 48 | `odds_B365_close_A` | float64 | 32.1% | 67.9% | 115 | median=3.2 | 1.08 | 41 |
| 49 | `odds_B365_close_over25` | float64 | 32.1% | 67.9% | 83 | median=1.8 | 1.12 | 3.2 |
| 50 | `odds_B365_close_under25` | float64 | 32.1% | 67.9% | 82 | median=2 | 1.36 | 6 |
| 51 | `odds_B365_AH_H` | float64 | 32.1% | 67.9% | 48 | median=1.95 | 1.7 | 3.55 |
| 52 | `odds_B365_AH_A` | float64 | 32.1% | 67.9% | 50 | median=1.95 | 1.27 | 2.17 |
| 53 | `odds_AH_line` | float64 | 28.5% | 71.5% | 22 | median=-0.25 | -3 | 2.5 |
| 54 | `odds_AH_close_line` | float64 | 28.9% | 71.1% | 25 | median=-0.25 | -3.75 | 3 |
| 55 | `matchweek` | float64 | 100.0% | 0.0% | 38 | median=19 | 1 | 38 |
| 56 | `kickoff_time` | object | 0.4% | 99.6% | 12 | '19:45' | — | — |
| 57 | `venue` | object | 0.4% | 99.6% | 1 | '' | — | — |
| 58 | `attendance` | object | 0.0% | 100.0% | 0 | — | — | — |
| 59 | `home_formation` | object | 0.4% | 99.6% | 9 | '3-1-4-2' | — | — |
| 60 | `away_formation` | object | 0.4% | 99.6% | 10 | '3-5-2' | — | — |
| 61 | `home_possession` | float64 | 0.4% | 99.6% | 38 | median=53 | 24 | 73 |
| 62 | `away_possession` | float64 | 0.4% | 99.6% | 38 | median=47 | 27 | 76 |
| 63 | `home_passing_accuracy` | float64 | 0.4% | 99.6% | 51 | median=84.7 | 61.7 | 93.1 |
| 64 | `away_passing_accuracy` | float64 | 0.4% | 99.6% | 53 | median=82.7 | 61.5 | 91.9 |
| 65 | `home_shots_on_target` | float64 | 0.4% | 99.6% | 10 | median=4 | 0 | 9 |
| 66 | `away_shots_on_target` | float64 | 0.4% | 99.6% | 9 | median=3 | 0 | 8 |
| 67 | `home_saves` | float64 | 0.4% | 99.6% | 8 | median=2 | 0 | 7 |
| 68 | `away_saves` | float64 | 0.4% | 99.6% | 8 | median=2 | 0 | 8 |
| 69 | `home_cards` | float64 | 0.4% | 99.6% | 6 | median=1 | 0 | 5 |
| 70 | `away_cards` | float64 | 0.4% | 99.6% | 7 | median=2 | 0 | 7 |
| 71 | `home_crosses` | float64 | 0.4% | 99.6% | 10 | median=4 | 1 | 10 |
| 72 | `away_crosses` | float64 | 0.4% | 99.6% | 13 | median=3 | 0 | 12 |
| 73 | `home_touches` | float64 | 0.4% | 99.6% | 33 | median=21.5 | 7 | 61 |
| 74 | `away_touches` | float64 | 0.4% | 99.6% | 25 | median=17 | 5 | 64 |
| 75 | `home_tackles` | float64 | 0.4% | 99.6% | 20 | median=14 | 4 | 24 |
| 76 | `away_tackles` | float64 | 0.4% | 99.6% | 22 | median=15 | 4 | 33 |
| 77 | `home_interceptions` | float64 | 0.4% | 99.6% | 14 | median=8 | 2 | 18 |
| 78 | `away_interceptions` | float64 | 0.4% | 99.6% | 15 | median=8 | 1 | 15 |
| 79 | `home_aerials_won` | float64 | 0.4% | 99.6% | 22 | median=13.5 | 4 | 30 |
| 80 | `away_aerials_won` | float64 | 0.4% | 99.6% | 21 | median=13 | 5 | 34 |
| 81 | `home_clearances` | float64 | 0.4% | 99.6% | 29 | median=22.5 | 7 | 61 |
| 82 | `away_clearances` | float64 | 0.4% | 99.6% | 30 | median=23 | 7 | 56 |
| 83 | `home_offsides` | float64 | 0.4% | 99.6% | 7 | median=1.5 | 0 | 7 |
| 84 | `away_offsides` | float64 | 0.4% | 99.6% | 6 | median=1.5 | 0 | 5 |
| 85 | `home_goal_kicks` | float64 | 0.4% | 99.6% | 13 | median=6.5 | 1 | 14 |
| 86 | `away_goal_kicks` | float64 | 0.4% | 99.6% | 16 | median=8 | 1 | 18 |
| 87 | `home_throw_ins` | float64 | 0.4% | 99.6% | 21 | median=18 | 8 | 32 |
| 88 | `away_throw_ins` | float64 | 0.4% | 99.6% | 21 | median=18 | 9 | 37 |
| 89 | `home_long_balls` | float64 | 0.4% | 99.6% | 22 | median=23 | 11 | 44 |
| 90 | `away_long_balls` | float64 | 0.4% | 99.6% | 23 | median=22 | 8 | 47 |
| 91 | `home_passing_accuracy_count` | float64 | 0.4% | 99.6% | 57 | median=382 | 144 | 633 |
| 92 | `home_passing_accuracy_total` | float64 | 0.4% | 99.6% | 59 | median=446 | 208 | 680 |
| 93 | `away_passing_accuracy_count` | float64 | 0.4% | 99.6% | 59 | median=342 | 136 | 657 |
| 94 | `away_passing_accuracy_total` | float64 | 0.4% | 99.6% | 60 | median=412 | 221 | 715 |
| 95 | `home_shots_on_target_total` | float64 | 0.4% | 99.6% | 18 | median=13 | 3 | 25 |
| 96 | `away_shots_on_target_total` | float64 | 0.4% | 99.6% | 20 | median=9.5 | 3 | 26 |
| 97 | `home_saves_count` | float64 | 0.4% | 99.6% | 8 | median=2 | 0 | 7 |
| 98 | `home_saves_total` | float64 | 0.4% | 99.6% | 8 | median=2 | 0 | 7 |
| 99 | `away_saves_count` | float64 | 0.4% | 99.6% | 8 | median=2 | 0 | 8 |
| 100 | `away_saves_total` | float64 | 0.4% | 99.6% | 8 | median=2 | 0 | 8 |
| 101 | `home_xg` | float64 | 0.4% | 99.6% | 52 | median=1.28 | 0.19 | 3.29 |
| 102 | `away_xg` | float64 | 0.4% | 99.6% | 54 | median=0.915 | 0.16 | 3.64 |
| 103 | `home_manager` | object | 0.4% | 99.6% | 1 | '' | — | — |
| 104 | `away_manager` | object | 0.4% | 99.6% | 1 | '' | — | — |
| 105 | `home_captain` | object | 0.4% | 99.6% | 1 | '' | — | — |
| 106 | `away_captain` | object | 0.4% | 99.6% | 1 | '' | — | — |
| 107 | `data_source` | object | 0.4% | 99.6% | 1 | 'sofascore' | — | — |
| 108 | `home_ht_score` | float64 | 0.4% | 99.6% | 4 | median=0 | 0 | 3 |
| 109 | `away_ht_score` | float64 | 0.4% | 99.6% | 3 | median=0 | 0 | 2 |

---

## 2. EVENT DATA — FBref shots

### `data/parsed/shots.parquet`

_Individual shot events with xG, body part, assist chain. 2024-2025 only._

- **Format:** Parquet  
- **Size:** 119.6KB  
- **Modified:** 2026-02-17 12:47  
- **Rows:** 9,213  
- **Columns:** 15  
- **Seasons:** 1 covered — 2024-2025 → 2024-2025  

**Columns (15):**

| # | Column | Dtype | Filled % | NaN % | Unique | Sample | Min | Max |
|---|--------|-------|----------|-------|--------|--------|-----|-----|
| 1 | `match_id` | object | 100.0% | 0.0% | 380 | '2025-04-13_Atalanta_Bologna' | — | — |
| 2 | `minute` | object | 100.0% | 0.0% | 111 | '3' | — | — |
| 3 | `player` | object | 100.0% | 0.0% | 552 | 'Mateo Retegui' | — | — |
| 4 | `team` | object | 100.0% | 0.0% | 20 | 'Atalanta' | — | — |
| 5 | `xg_shot` | object | 100.0% | 0.0% | 99 | '0.39' | — | — |
| 6 | `psxg_shot` | object | 100.0% | 0.0% | 102 | '0.94' | — | — |
| 7 | `outcome` | object | 100.0% | 0.0% | 6 | 'Goal' | — | — |
| 8 | `distance` | object | 100.0% | 0.0% | 62 | '5' | — | — |
| 9 | `body_part` | object | 100.0% | 0.0% | 4 | 'Left Foot' | — | — |
| 10 | `notes` | object | 100.0% | 0.0% | 6 | '' | — | — |
| 11 | `sca_1_player` | object | 100.0% | 0.0% | 507 | 'Raoul Bellanova' | — | — |
| 12 | `sca_1_type` | object | 100.0% | 0.0% | 8 | 'Pass (Live)' | — | — |
| 13 | `sca_2_player` | object | 100.0% | 0.0% | 515 | 'Mario Pašalić' | — | — |
| 14 | `sca_2_type` | object | 100.0% | 0.0% | 8 | 'Pass (Live)' | — | — |
| 15 | `season` | object | 100.0% | 0.0% | 1 | '2024-2025' | — | — |

---

## 2. EVENT DATA — Sofascore shots (base)

### `data/external/sofascore/all_shots.parquet`

_**Biggest shot dataset.** 82k shots 2017-2026, with shot location (x,y,z) and player._

- **Format:** Parquet  
- **Size:** 1.2MB  
- **Modified:** 2026-02-09 14:16  
- **Rows:** 82,432  
- **Columns:** 17  
- **Seasons:** 9 covered — 2017-2018 → 2025-2026  

**Columns (17):**

| # | Column | Dtype | Filled % | NaN % | Unique | Sample | Min | Max |
|---|--------|-------|----------|-------|--------|--------|-----|-----|
| 1 | `season` | object | 100.0% | 0.0% | 9 | '2017-2018' | — | — |
| 2 | `match_id` | object | 100.0% | 0.0% | 3167 | '7497939' | — | — |
| 3 | `is_home` | bool | 100.0% | 0.0% | 2 | np.True_ | — | — |
| 4 | `player_id` | int64 | 100.0% | 0.0% | 1576 | median=3.11e+05 | 536 | 2.21e+06 |
| 5 | `player_name` | object | 100.0% | 0.0% | 1573 | 'Lucas Castro' | — | — |
| 6 | `shot_x` | float64 | 100.0% | 0.0% | 534 | median=13.5 | 0.3 | 90.5 |
| 7 | `shot_y` | float64 | 100.0% | 0.0% | 819 | median=49.9 | 0.2 | 99.7 |
| 8 | `gm_x` | float64 | 91.8% | 8.2% | 1 | median=0 | 0 | 0 |
| 9 | `gm_y` | float64 | 91.8% | 8.2% | 652 | median=50.2 | 0.2 | 100 |
| 10 | `gm_z` | float64 | 91.8% | 8.2% | 135 | median=19 | 0 | 100 |
| 11 | `situation` | object | 100.0% | 0.0% | 8 | 'assisted' | — | — |
| 12 | `body_part` | object | 100.0% | 0.0% | 4 | 'head' | — | — |
| 13 | `shot_type` | object | 100.0% | 0.0% | 5 | 'miss' | — | — |
| 14 | `xg` | float64 | 45.5% | 54.5% | 33462 | median=0.0506 | 0 | 0.998 |
| 15 | `xgot` | float64 | 26.4% | 73.6% | 8917 | median=0.0335 | 0 | 0.999 |
| 16 | `is_goal` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 17 | `time` | int64 | 100.0% | 0.0% | 90 | median=50 | 1 | 90 |

---

## 2. EVENT DATA — Sofascore shots + xG

### `data/external/sofascore/all_shots_with_xg.parquet`

_**Best shot dataset.** Same as base + xG, xGoT, distance — already enriched._

- **Format:** Parquet  
- **Size:** 2.6MB  
- **Modified:** 2026-02-09 14:18  
- **Rows:** 82,432  
- **Columns:** 27  
- **Seasons:** 9 covered — 2017-2018 → 2025-2026  

**Columns (27):**

| # | Column | Dtype | Filled % | NaN % | Unique | Sample | Min | Max |
|---|--------|-------|----------|-------|--------|--------|-----|-----|
| 1 | `season` | object | 100.0% | 0.0% | 9 | '2017-2018' | — | — |
| 2 | `match_id` | object | 100.0% | 0.0% | 3167 | '7497939' | — | — |
| 3 | `is_home` | bool | 100.0% | 0.0% | 2 | np.True_ | — | — |
| 4 | `player_id` | int64 | 100.0% | 0.0% | 1576 | median=3.11e+05 | 536 | 2.21e+06 |
| 5 | `player_name` | object | 100.0% | 0.0% | 1573 | 'Lucas Castro' | — | — |
| 6 | `shot_x` | float64 | 100.0% | 0.0% | 534 | median=13.5 | 0.3 | 90.5 |
| 7 | `shot_y` | float64 | 100.0% | 0.0% | 819 | median=49.9 | 0.2 | 99.7 |
| 8 | `gm_x` | float64 | 91.8% | 8.2% | 1 | median=0 | 0 | 0 |
| 9 | `gm_y` | float64 | 91.8% | 8.2% | 652 | median=50.2 | 0.2 | 100 |
| 10 | `gm_z` | float64 | 91.8% | 8.2% | 135 | median=19 | 0 | 100 |
| 11 | `situation` | object | 100.0% | 0.0% | 8 | 'assisted' | — | — |
| 12 | `body_part` | object | 100.0% | 0.0% | 4 | 'head' | — | — |
| 13 | `shot_type` | object | 100.0% | 0.0% | 5 | 'miss' | — | — |
| 14 | `xg` | float64 | 45.5% | 54.5% | 33462 | median=0.0506 | 0 | 0.998 |
| 15 | `xgot` | float64 | 26.4% | 73.6% | 8917 | median=0.0335 | 0 | 0.999 |
| 16 | `is_goal` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 17 | `time` | int64 | 100.0% | 0.0% | 90 | median=50 | 1 | 90 |
| 18 | `distance` | float64 | 100.0% | 0.0% | 32791 | median=19.1 | 0.583 | 97 |
| 19 | `angle` | float64 | 100.0% | 0.0% | 34811 | median=32.9 | 0 | 89.5 |
| 20 | `is_header` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 21 | `is_right` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 22 | `is_left` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 23 | `is_penalty` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 24 | `is_freekick` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 25 | `is_set_piece` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 26 | `is_fast_break` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 27 | `xg_predicted` | float64 | 100.0% | 0.0% | 43659 | median=0.0501 | 0.00757 | 0.98 |

---

## 2. EVENT DATA — goals/cards/subs timing

### `data/parsed/events.parquet`

_Event minutes (goal scored at min 45', yellow at min 72'). 2024-2025 only._

- **Format:** Parquet  
- **Size:** 102.9KB  
- **Modified:** 2026-04-21 18:55  
- **Rows:** 11,781  
- **Columns:** 7  
- **Seasons:** 9 covered — 2017-2018 → 2025-2026  

**Columns (7):**

| # | Column | Dtype | Filled % | NaN % | Unique | Sample | Min | Max |
|---|--------|-------|----------|-------|--------|--------|-----|-----|
| 1 | `match_id` | object | 100.0% | 0.0% | 3418 | '0005cd5f' | — | — |
| 2 | `season` | object | 100.0% | 0.0% | 9 | '2017-2018' | — | — |
| 3 | `minute` | float64 | 92.2% | 7.8% | 91 | median=52 | -5 | 90 |
| 4 | `event_type` | object | 100.0% | 0.0% | 5 | 'goal' | — | — |
| 5 | `team` | object | 100.0% | 0.0% | 32 | 'Juventus' | — | — |
| 6 | `player` | object | 100.0% | 0.0% | 1284 | 'Gonzalo Higuaín' | — | — |
| 7 | `detail` | object | 100.0% | 0.0% | 7 | '' | — | — |

---

## 2. EVENT DATA — starting XI

### `data/parsed/lineups.parquet`

_Formation + starting + subs per match. 2024-2025 only._

- **Format:** Parquet  
- **Size:** 487.3KB  
- **Modified:** 2026-04-21 18:55  
- **Rows:** 164,334  
- **Columns:** 8  
- **Seasons:** 9 covered — 2017-2018 → 2025-2026  

**Columns (8):**

| # | Column | Dtype | Filled % | NaN % | Unique | Sample | Min | Max |
|---|--------|-------|----------|-------|--------|--------|-----|-----|
| 1 | `match_id` | object | 100.0% | 0.0% | 3610 | '0005cd5f' | — | — |
| 2 | `team` | object | 100.0% | 0.0% | 33 | 'Napoli' | — | — |
| 3 | `is_home` | bool | 100.0% | 0.0% | 2 | np.True_ | — | — |
| 4 | `formation` | object | 100.0% | 0.0% | 19 | '4-3-3' | — | — |
| 5 | `player_name` | object | 100.0% | 0.0% | 2830 | 'Pepe Reina' | — | — |
| 6 | `shirt_number` | int64 | 100.0% | 0.0% | 100 | median=20 | 0 | 99 |
| 7 | `role` | object | 100.0% | 0.0% | 2 | 'starter' | — | — |
| 8 | `season` | object | 100.0% | 0.0% | 9 | '2017-2018' | — | — |

---

## 2. EVENT DATA — captains

### `data/external/sofascore/captains.parquet`

_Team captain per match._

- **Format:** Parquet  
- **Size:** 48.7KB  
- **Modified:** 2026-04-20 23:30  
- **Rows:** 6,650  
- **Columns:** 5  

**Columns (5):**

| # | Column | Dtype | Filled % | NaN % | Unique | Sample | Min | Max |
|---|--------|-------|----------|-------|--------|--------|-----|-----|
| 1 | `match_id` | int64 | 100.0% | 0.0% | 3327 | median=9.65e+06 | 7.5e+06 | 1.55e+07 |
| 2 | `is_home` | bool | 100.0% | 0.0% | 2 | np.True_ | — | — |
| 3 | `captain_name` | object | 100.0% | 0.0% | 338 | 'Giovanni Di Lorenzo' | — | — |
| 4 | `captain_id` | int64 | 100.0% | 0.0% | 335 | median=1.23e+05 | 1.05e+03 | 1.16e+06 |
| 5 | `position` | object | 100.0% | 0.0% | 4 | 'D' | — | — |

---

## 3. PLAYER STATS — per-match

### `data/parsed/player_stats.parquet`

_**Player prop source.** 98k rows. Goals, assists, shots, SOT, cards, fouls per player per match. 2017-2026._

- **Format:** Parquet  
- **Size:** 1.9MB  
- **Modified:** 2026-04-21 18:55  
- **Rows:** 100,441  
- **Columns:** 141  
- **Date column:** `match_date` — range 2017-08-19 → 2026-02-06  
- **Seasons:** 9 covered — 2017-2018 → 2025-2026  

**Columns (141):**

| # | Column | Dtype | Filled % | NaN % | Unique | Sample | Min | Max |
|---|--------|-------|----------|-------|--------|--------|-----|-----|
| 1 | `match_id` | object | 100.0% | 0.0% | 3315 | '0005cd5f' | — | — |
| 2 | `team` | object | 100.0% | 0.0% | 32 | 'Napoli' | — | — |
| 3 | `is_home` | bool | 100.0% | 0.0% | 2 | np.True_ | — | — |
| 4 | `data_source` | object | 100.0% | 0.0% | 2 | 'fbref_match' | — | — |
| 5 | `player` | object | 100.0% | 0.0% | 2028 | 'Lorenzo Insigne' | — | — |
| 6 | `shirtnumber` | int64 | 100.0% | 0.0% | 100 | median=18 | 0 | 99 |
| 7 | `nationality` | object | 98.0% | 2.0% | 102 | 'itITA' | — | — |
| 8 | `position` | object | 100.0% | 0.0% | 21 | 'FW' | — | — |
| 9 | `age` | object | 98.0% | 2.0% | 7820 | '26-180' | — | — |
| 10 | `minutes` | float64 | 100.0% | 0.0% | 91 | median=83 | 1 | 95 |
| 11 | `goals` | int64 | 100.0% | 0.0% | 5 | median=0 | 0 | 4 |
| 12 | `assists` | int64 | 100.0% | 0.0% | 4 | median=0 | 0 | 3 |
| 13 | `pens_made` | float64 | 98.0% | 2.0% | 3 | median=0 | 0 | 2 |
| 14 | `pens_att` | float64 | 98.0% | 2.0% | 3 | median=0 | 0 | 2 |
| 15 | `shots` | int64 | 100.0% | 0.0% | 15 | median=0 | 0 | 14 |
| 16 | `shots_on_target` | int64 | 100.0% | 0.0% | 10 | median=0 | 0 | 10 |
| 17 | `cards_yellow` | float64 | 98.0% | 2.0% | 3 | median=0 | 0 | 2 |
| 18 | `cards_red` | float64 | 98.0% | 2.0% | 2 | median=0 | 0 | 1 |
| 19 | `fouls` | float64 | 87.5% | 12.5% | 9 | median=1 | 0 | 8 |
| 20 | `fouled` | float64 | 87.5% | 12.5% | 11 | median=0 | 0 | 11 |
| 21 | `offsides` | float64 | 85.5% | 14.5% | 8 | median=0 | 0 | 7 |
| 22 | `crosses` | float64 | 85.5% | 14.5% | 28 | median=0 | 0 | 30 |
| 23 | `tackles_won` | float64 | 85.5% | 14.5% | 10 | median=0 | 0 | 9 |
| 24 | `interceptions` | int64 | 100.0% | 0.0% | 14 | median=0 | 0 | 13 |
| 25 | `own_goals` | float64 | 85.3% | 14.7% | 2 | median=0 | 0 | 1 |
| 26 | `pens_won` | object | 85.5% | 14.5% | 4 | '0' | — | — |
| 27 | `pens_conceded` | object | 85.5% | 14.5% | 3 | '0' | — | — |
| 28 | `season` | object | 100.0% | 0.0% | 9 | '2017-2018' | — | — |
| 29 | `match_date` | object | 98.0% | 2.0% | 1018 | '2017-12-01' | — | — |
| 30 | `touches` | float64 | 12.5% | 87.5% | 132 | median=33 | 0 | 145 |
| 31 | `tackles` | float64 | 14.5% | 85.5% | 11 | median=1 | 0 | 10 |
| 32 | `blocks` | float64 | 12.5% | 87.5% | 9 | median=0 | 0 | 8 |
| 33 | `xg` | float64 | 12.5% | 87.5% | 22 | median=0 | 0 | 2.1 |
| 34 | `npxg` | float64 | 12.5% | 87.5% | 19 | median=0 | 0 | 2 |
| 35 | `xg_assist` | float64 | 12.5% | 87.5% | 15 | median=0 | 0 | 1.5 |
| 36 | `sca` | float64 | 12.5% | 87.5% | 14 | median=1 | 0 | 16 |
| 37 | `gca` | float64 | 12.5% | 87.5% | 4 | median=0 | 0 | 3 |
| 38 | `passes_completed` | float64 | 12.5% | 87.5% | 115 | median=19 | 0 | 132 |
| 39 | `passes` | float64 | 12.5% | 87.5% | 124 | median=26 | 0 | 138 |
| 40 | `passes_pct` | float64 | 12.3% | 87.7% | 477 | median=80 | 0 | 100 |
| 41 | `progressive_passes` | float64 | 12.5% | 87.5% | 23 | median=1 | 0 | 23 |
| 42 | `carries` | float64 | 12.5% | 87.5% | 99 | median=18 | 0 | 104 |
| 43 | `progressive_carries` | float64 | 12.5% | 87.5% | 13 | median=1 | 0 | 13 |
| 44 | `take_ons` | float64 | 12.5% | 87.5% | 12 | median=0 | 0 | 11 |
| 45 | `take_ons_won` | float64 | 12.5% | 87.5% | 8 | median=0 | 0 | 7 |
| 46 | `passing_minutes` | float64 | 12.5% | 87.5% | 88 | median=75 | 1 | 90 |
| 47 | `passing_passes_completed` | float64 | 12.5% | 87.5% | 115 | median=19 | 0 | 132 |
| 48 | `passing_passes` | float64 | 12.5% | 87.5% | 124 | median=26 | 0 | 138 |
| 49 | `passing_passes_pct` | float64 | 12.3% | 87.7% | 477 | median=80 | 0 | 100 |
| 50 | `passing_passes_total_distance` | float64 | 12.5% | 87.5% | 1518 | median=332 | 0 | 2.78e+03 |
| 51 | `passing_passes_progressive_distance` | float64 | 12.5% | 87.5% | 757 | median=92.5 | 0 | 1.21e+03 |
| 52 | `passing_passes_completed_short` | float64 | 12.5% | 87.5% | 61 | median=8 | 0 | 65 |
| 53 | `passing_passes_short` | float64 | 12.5% | 87.5% | 63 | median=9 | 0 | 66 |
| 54 | `passing_passes_pct_short` | float64 | 12.0% | 88.0% | 180 | median=90.9 | 0 | 100 |
| 55 | `passing_passes_completed_medium` | float64 | 12.5% | 87.5% | 69 | median=8 | 0 | 90 |
| 56 | `passing_passes_medium` | float64 | 12.5% | 87.5% | 73 | median=9 | 0 | 93 |
| 57 | `passing_passes_pct_medium` | float64 | 11.8% | 88.2% | 228 | median=89.1 | 0 | 100 |
| 58 | `passing_passes_completed_long` | float64 | 12.5% | 87.5% | 23 | median=1 | 0 | 22 |
| 59 | `passing_passes_long` | float64 | 12.5% | 87.5% | 41 | median=3 | 0 | 41 |
| 60 | `passing_passes_pct_long` | object | 12.5% | 87.5% | 198 | '' | — | — |
| 61 | `passing_assists` | float64 | 12.5% | 87.5% | 3 | median=0 | 0 | 2 |
| 62 | `passing_xg_assist` | float64 | 12.5% | 87.5% | 15 | median=0 | 0 | 1.5 |
| 63 | `passing_pass_xa` | float64 | 12.5% | 87.5% | 12 | median=0 | 0 | 1.2 |
| 64 | `passing_assisted_shots` | float64 | 12.5% | 87.5% | 9 | median=0 | 0 | 8 |
| 65 | `passing_passes_into_final_third` | float64 | 12.5% | 87.5% | 23 | median=1 | 0 | 23 |
| 66 | `passing_passes_into_penalty_area` | float64 | 12.5% | 87.5% | 8 | median=0 | 0 | 7 |
| 67 | `passing_crosses_into_penalty_area` | float64 | 12.5% | 87.5% | 7 | median=0 | 0 | 6 |
| 68 | `passing_progressive_passes` | float64 | 12.5% | 87.5% | 23 | median=1 | 0 | 23 |
| 69 | `passing_types_minutes` | float64 | 12.5% | 87.5% | 88 | median=75 | 1 | 90 |
| 70 | `passing_types_passes` | float64 | 12.5% | 87.5% | 124 | median=26 | 0 | 138 |
| 71 | `passing_types_passes_live` | float64 | 12.5% | 87.5% | 117 | median=22 | 0 | 135 |
| 72 | `passing_types_passes_dead` | float64 | 12.5% | 87.5% | 28 | median=2 | 0 | 32 |
| 73 | `passing_types_passes_free_kicks` | float64 | 12.5% | 87.5% | 12 | median=0 | 0 | 11 |
| 74 | `passing_types_through_balls` | float64 | 12.5% | 87.5% | 5 | median=0 | 0 | 4 |
| 75 | `passing_types_passes_switches` | float64 | 12.5% | 87.5% | 6 | median=0 | 0 | 5 |
| 76 | `passing_types_crosses` | float64 | 12.5% | 87.5% | 19 | median=0 | 0 | 22 |
| 77 | `passing_types_throw_ins` | float64 | 12.5% | 87.5% | 22 | median=0 | 0 | 21 |
| 78 | `passing_types_corner_kicks` | float64 | 12.5% | 87.5% | 12 | median=0 | 0 | 12 |
| 79 | `passing_types_corner_kicks_in` | float64 | 12.5% | 87.5% | 8 | median=0 | 0 | 7 |
| 80 | `passing_types_corner_kicks_out` | float64 | 12.5% | 87.5% | 8 | median=0 | 0 | 8 |
| 81 | `passing_types_corner_kicks_straight` | float64 | 12.5% | 87.5% | 2 | median=0 | 0 | 1 |
| 82 | `passing_types_passes_completed` | float64 | 12.5% | 87.5% | 115 | median=19 | 0 | 132 |
| 83 | `passing_types_passes_offsides` | float64 | 12.5% | 87.5% | 4 | median=0 | 0 | 3 |
| 84 | `passing_types_passes_blocked` | float64 | 12.5% | 87.5% | 7 | median=0 | 0 | 6 |
| 85 | `defense_minutes` | float64 | 12.5% | 87.5% | 88 | median=75 | 1 | 90 |
| 86 | `defense_tackles` | float64 | 12.5% | 87.5% | 11 | median=1 | 0 | 10 |
| 87 | `defense_tackles_won` | float64 | 12.5% | 87.5% | 8 | median=0 | 0 | 7 |
| 88 | `defense_tackles_def_3rd` | float64 | 12.5% | 87.5% | 9 | median=0 | 0 | 8 |
| 89 | `defense_tackles_mid_3rd` | float64 | 12.5% | 87.5% | 8 | median=0 | 0 | 7 |
| 90 | `defense_tackles_att_3rd` | float64 | 12.5% | 87.5% | 4 | median=0 | 0 | 3 |
| 91 | `defense_challenge_tackles` | float64 | 12.5% | 87.5% | 8 | median=0 | 0 | 8 |
| 92 | `defense_challenges` | float64 | 12.5% | 87.5% | 11 | median=0 | 0 | 10 |
| 93 | `defense_challenge_tackles_pct` | object | 12.5% | 87.5% | 23 | '' | — | — |
| 94 | `defense_challenges_lost` | float64 | 12.5% | 87.5% | 8 | median=0 | 0 | 7 |
| 95 | `defense_blocks` | float64 | 12.5% | 87.5% | 9 | median=0 | 0 | 8 |
| 96 | `defense_blocked_shots` | float64 | 12.5% | 87.5% | 6 | median=0 | 0 | 5 |
| 97 | `defense_blocked_passes` | float64 | 12.5% | 87.5% | 8 | median=0 | 0 | 7 |
| 98 | `defense_interceptions` | float64 | 12.5% | 87.5% | 7 | median=0 | 0 | 6 |
| 99 | `defense_tackles_interceptions` | float64 | 12.5% | 87.5% | 13 | median=1 | 0 | 12 |
| 100 | `defense_clearances` | float64 | 12.5% | 87.5% | 20 | median=1 | 0 | 20 |
| 101 | `defense_errors` | float64 | 12.5% | 87.5% | 3 | median=0 | 0 | 2 |
| 102 | `possession_minutes` | float64 | 12.5% | 87.5% | 88 | median=75 | 1 | 90 |
| 103 | `possession_touches` | float64 | 12.5% | 87.5% | 132 | median=33 | 0 | 145 |
| 104 | `possession_touches_def_pen_area` | float64 | 12.5% | 87.5% | 57 | median=1 | 0 | 61 |
| 105 | `possession_touches_def_3rd` | float64 | 12.5% | 87.5% | 70 | median=6 | 0 | 71 |
| 106 | `possession_touches_mid_3rd` | float64 | 12.5% | 87.5% | 90 | median=14 | 0 | 102 |
| 107 | `possession_touches_att_3rd` | float64 | 12.5% | 87.5% | 57 | median=6 | 0 | 63 |
| 108 | `possession_touches_att_pen_area` | float64 | 12.5% | 87.5% | 16 | median=1 | 0 | 16 |
| 109 | `possession_touches_live_ball` | float64 | 12.5% | 87.5% | 132 | median=33 | 0 | 145 |
| 110 | `possession_take_ons` | float64 | 12.5% | 87.5% | 12 | median=0 | 0 | 11 |
| 111 | `possession_take_ons_won` | float64 | 12.5% | 87.5% | 8 | median=0 | 0 | 7 |
| 112 | `possession_take_ons_won_pct` | object | 12.5% | 87.5% | 32 | '0.0' | — | — |
| 113 | `possession_take_ons_tackled` | float64 | 12.5% | 87.5% | 9 | median=0 | 0 | 8 |
| 114 | `possession_take_ons_tackled_pct` | object | 12.5% | 87.5% | 31 | '100.0' | — | — |
| 115 | `possession_carries` | float64 | 12.5% | 87.5% | 99 | median=18 | 0 | 104 |
| 116 | `possession_carries_distance` | float64 | 12.5% | 87.5% | 478 | median=90 | 0 | 705 |
| 117 | `possession_carries_progressive_distance` | float64 | 12.5% | 87.5% | 316 | median=40 | 0 | 441 |
| 118 | `possession_progressive_carries` | float64 | 12.5% | 87.5% | 13 | median=1 | 0 | 13 |
| 119 | `possession_carries_into_final_third` | float64 | 12.5% | 87.5% | 12 | median=0 | 0 | 16 |
| 120 | `possession_carries_into_penalty_area` | float64 | 12.5% | 87.5% | 9 | median=0 | 0 | 9 |
| 121 | `possession_miscontrols` | float64 | 12.5% | 87.5% | 11 | median=1 | 0 | 10 |
| 122 | `possession_dispossessed` | float64 | 12.5% | 87.5% | 8 | median=0 | 0 | 7 |
| 123 | `possession_passes_received` | float64 | 12.5% | 87.5% | 110 | median=20 | 0 | 128 |
| 124 | `possession_progressive_passes_received` | float64 | 12.5% | 87.5% | 24 | median=1 | 0 | 25 |
| 125 | `misc_minutes` | float64 | 12.5% | 87.5% | 88 | median=75 | 1 | 90 |
| 126 | `misc_cards_yellow` | float64 | 12.5% | 87.5% | 3 | median=0 | 0 | 2 |
| 127 | `misc_cards_red` | float64 | 12.5% | 87.5% | 2 | median=0 | 0 | 1 |
| 128 | `misc_cards_yellow_red` | float64 | 12.5% | 87.5% | 2 | median=0 | 0 | 1 |
| 129 | `misc_fouls` | float64 | 12.5% | 87.5% | 9 | median=1 | 0 | 8 |
| 130 | `misc_fouled` | float64 | 12.5% | 87.5% | 10 | median=0 | 0 | 10 |
| 131 | `misc_offsides` | float64 | 12.5% | 87.5% | 6 | median=0 | 0 | 5 |
| 132 | `misc_crosses` | float64 | 12.5% | 87.5% | 19 | median=0 | 0 | 22 |
| 133 | `misc_interceptions` | float64 | 12.5% | 87.5% | 7 | median=0 | 0 | 6 |
| 134 | `misc_tackles_won` | float64 | 12.5% | 87.5% | 8 | median=0 | 0 | 7 |
| 135 | `misc_pens_won` | float64 | 12.5% | 87.5% | 2 | median=0 | 0 | 1 |
| 136 | `misc_pens_conceded` | float64 | 12.5% | 87.5% | 2 | median=0 | 0 | 1 |
| 137 | `misc_own_goals` | float64 | 12.5% | 87.5% | 2 | median=0 | 0 | 1 |
| 138 | `misc_ball_recoveries` | float64 | 12.5% | 87.5% | 14 | median=2 | 0 | 13 |
| 139 | `misc_aerials_won` | float64 | 12.5% | 87.5% | 15 | median=0 | 0 | 15 |
| 140 | `misc_aerials_lost` | float64 | 12.5% | 87.5% | 14 | median=0 | 0 | 14 |
| 141 | `misc_aerials_won_pct` | object | 12.5% | 87.5% | 62 | '25.0' | — | — |

---

## 3. PLAYER STATS — GK per-match

### `data/parsed/goalkeeper_stats.parquet`

_PSxG, save %, launch stats. 2024-2025 only._

- **Format:** Parquet  
- **Size:** 137.1KB  
- **Modified:** 2026-04-21 17:11  
- **Rows:** 6,651  
- **Columns:** 28  
- **Seasons:** 9 covered — 2017-2018 → 2025-2026  

**Columns (28):**

| # | Column | Dtype | Filled % | NaN % | Unique | Sample | Min | Max |
|---|--------|-------|----------|-------|--------|--------|-----|-----|
| 1 | `match_id` | object | 100.0% | 0.0% | 3280 | '0005cd5f' | — | — |
| 2 | `team` | object | 100.0% | 0.0% | 32 | 'Napoli' | — | — |
| 3 | `is_home` | bool | 100.0% | 0.0% | 2 | np.True_ | — | — |
| 4 | `player` | object | 100.0% | 0.0% | 147 | 'Pepe Reina' | — | — |
| 5 | `nationality` | object | 100.0% | 0.0% | 31 | 'esESP' | — | — |
| 6 | `age` | object | 100.0% | 0.0% | 4231 | '35-092' | — | — |
| 7 | `minutes` | int64 | 100.0% | 0.0% | 69 | median=90 | 0 | 90 |
| 8 | `gk_shots_on_target_against` | float64 | 100.0% | 0.0% | 17 | median=4 | 0 | 17 |
| 9 | `gk_goals_against` | float64 | 99.9% | 0.1% | 9 | median=1 | 0 | 8 |
| 10 | `gk_saves` | float64 | 100.0% | 0.0% | 16 | median=3 | 0 | 17 |
| 11 | `gk_save_pct` | float64 | 96.3% | 3.7% | 42 | median=75 | -50 | 100 |
| 12 | `season` | object | 100.0% | 0.0% | 9 | '2017-2018' | — | — |
| 13 | `gk_psxg` | float64 | 11.6% | 88.4% | 43 | median=1.1 | 0 | 4.6 |
| 14 | `gk_passes_completed_launched` | float64 | 11.5% | 88.5% | 18 | median=4 | 0 | 19 |
| 15 | `gk_passes_launched` | float64 | 11.6% | 88.4% | 37 | median=12 | 0 | 37 |
| 16 | `gk_passes_pct_launched` | float64 | 11.4% | 88.6% | 130 | median=33.3 | 0 | 100 |
| 17 | `gk_passes` | float64 | 11.5% | 88.5% | 54 | median=29 | 1 | 65 |
| 18 | `gk_passes_throws` | float64 | 11.6% | 88.4% | 15 | median=4 | 0 | 14 |
| 19 | `gk_pct_passes_launched` | float64 | 11.5% | 88.5% | 324 | median=32.1 | 0 | 100 |
| 20 | `gk_passes_length_avg` | float64 | 11.5% | 88.5% | 269 | median=32.8 | 6 | 62 |
| 21 | `gk_goal_kicks` | float64 | 11.5% | 88.5% | 19 | median=5 | 0 | 21 |
| 22 | `gk_pct_goal_kicks_launched` | float64 | 11.1% | 88.9% | 48 | median=50 | 0 | 100 |
| 23 | `gk_goal_kick_length_avg` | float64 | 11.1% | 88.9% | 389 | median=43 | 9 | 94 |
| 24 | `gk_crosses` | float64 | 11.6% | 88.4% | 36 | median=12 | 0 | 53 |
| 25 | `gk_crosses_stopped` | float64 | 11.5% | 88.5% | 8 | median=1 | 0 | 7 |
| 26 | `gk_crosses_stopped_pct` | float64 | 11.5% | 88.5% | 67 | median=4.4 | 0 | 100 |
| 27 | `gk_def_actions_outside_pen_area` | float64 | 11.5% | 88.5% | 8 | median=1 | 0 | 12 |
| 28 | `gk_avg_distance_def_actions` | float64 | 11.1% | 88.9% | 192 | median=12.5 | 1 | 97 |

---

## 4. TEAM STATS — FBref season-level

### `data/parsed/fbref_stats_standard.parquet`

_Season aggregates per player. Goals, assists, xG, xA. 2017-2018+._

- **Format:** Parquet  
- **Size:** 169.2KB  
- **Modified:** 2026-03-02 08:20  
- **Rows:** 5,611  
- **Columns:** 28  
- **Seasons:** 9 covered — 2017-2018 → 2025-2026  

**Columns (28):**

| # | Column | Dtype | Filled % | NaN % | Unique | Sample | Min | Max |
|---|--------|-------|----------|-------|--------|--------|-----|-----|
| 1 | `Rk` | object | 100.0% | 0.0% | 635 | '1' | — | — |
| 2 | `Unnamed: 1_level_0_Player` | object | 100.0% | 0.0% | 1961 | 'Rolando Aarons' | — | — |
| 3 | `Unnamed: 2_level_0_Nation` | object | 100.0% | 0.0% | 104 | 'eng ENG' | — | — |
| 4 | `Unnamed: 3_level_0_Pos` | object | 100.0% | 0.0% | 11 | 'MF,FW' | — | — |
| 5 | `Unnamed: 4_level_0_Squad` | object | 100.0% | 0.0% | 33 | 'Hellas Verona' | — | — |
| 6 | `Unnamed: 5_level_0_Age` | object | 99.9% | 0.1% | 553 | '21' | — | — |
| 7 | `Unnamed: 6_level_0_Born` | object | 99.9% | 0.1% | 34 | '1995' | — | — |
| 8 | `Playing Time_MP` | object | 100.0% | 0.0% | 39 | '11' | — | — |
| 9 | `Playing Time_Starts` | object | 100.0% | 0.0% | 40 | '6' | — | — |
| 10 | `Playing Time_Min` | object | 100.0% | 0.0% | 2399 | '517' | — | — |
| 11 | `Playing Time_90s` | object | 100.0% | 0.0% | 375 | '5.7' | — | — |
| 12 | `Performance_Gls` | object | 100.0% | 0.0% | 32 | '0' | — | — |
| 13 | `Performance_Ast` | object | 100.0% | 0.0% | 18 | '0' | — | — |
| 14 | `Performance_G+A` | object | 100.0% | 0.0% | 38 | '0' | — | — |
| 15 | `Performance_G-PK` | object | 100.0% | 0.0% | 26 | '0' | — | — |
| 16 | `Performance_PK` | object | 100.0% | 0.0% | 15 | '0' | — | — |
| 17 | `Performance_PKatt` | object | 100.0% | 0.0% | 15 | '0' | — | — |
| 18 | `Performance_CrdY` | object | 100.0% | 0.0% | 18 | '0' | — | — |
| 19 | `Performance_CrdR` | object | 100.0% | 0.0% | 5 | '0' | — | — |
| 20 | `Per 90 Minutes_Gls` | object | 100.0% | 0.0% | 118 | '0.00' | — | — |
| 21 | `Per 90 Minutes_Ast` | object | 100.0% | 0.0% | 95 | '0.00' | — | — |
| 22 | `Per 90 Minutes_G+A` | object | 100.0% | 0.0% | 146 | '0.00' | — | — |
| 23 | `Per 90 Minutes_G-PK` | object | 100.0% | 0.0% | 107 | '0.00' | — | — |
| 24 | `Per 90 Minutes_G+A-PK` | object | 100.0% | 0.0% | 139 | '0.00' | — | — |
| 25 | `Unnamed: 24_level_0_Matches` | object | 100.0% | 0.0% | 1 | 'Matches' | — | — |
| 26 | `stat_type` | object | 100.0% | 0.0% | 1 | 'stats_standard' | — | — |
| 27 | `source_file` | object | 100.0% | 0.0% | 1 | 'stats_standard.html' | — | — |
| 28 | `season` | object | 100.0% | 0.0% | 9 | '2017-2018' | — | — |

---

## 4. TEAM STATS — FBref shooting

### `data/parsed/fbref_stats_shooting.parquet`

_Season shots, SOT, G/Sh, G/SoT._

- **Format:** Parquet  
- **Size:** 137.6KB  
- **Modified:** 2026-03-02 08:20  
- **Rows:** 5,611  
- **Columns:** 22  
- **Seasons:** 9 covered — 2017-2018 → 2025-2026  

**Columns (22):**

| # | Column | Dtype | Filled % | NaN % | Unique | Sample | Min | Max |
|---|--------|-------|----------|-------|--------|--------|-----|-----|
| 1 | `Rk` | object | 100.0% | 0.0% | 635 | '1' | — | — |
| 2 | `Unnamed: 1_level_0_Player` | object | 100.0% | 0.0% | 1961 | 'Rolando Aarons' | — | — |
| 3 | `Unnamed: 2_level_0_Nation` | object | 100.0% | 0.0% | 104 | 'eng ENG' | — | — |
| 4 | `Unnamed: 3_level_0_Pos` | object | 100.0% | 0.0% | 11 | 'MF,FW' | — | — |
| 5 | `Unnamed: 4_level_0_Squad` | object | 100.0% | 0.0% | 33 | 'Hellas Verona' | — | — |
| 6 | `Unnamed: 5_level_0_Age` | object | 99.9% | 0.1% | 553 | '21' | — | — |
| 7 | `Unnamed: 6_level_0_Born` | object | 99.9% | 0.1% | 34 | '1995' | — | — |
| 8 | `Unnamed: 7_level_0_90s` | object | 100.0% | 0.0% | 375 | '5.7' | — | — |
| 9 | `Standard_Gls` | object | 100.0% | 0.0% | 32 | '0' | — | — |
| 10 | `Standard_Sh` | object | 99.9% | 0.1% | 133 | '2' | — | — |
| 11 | `Standard_SoT` | object | 100.0% | 0.0% | 61 | '2' | — | — |
| 12 | `Standard_SoT%` | object | 82.2% | 17.8% | 358 | '100.0' | — | — |
| 13 | `Standard_Sh/90` | object | 99.9% | 0.1% | 493 | '0.35' | — | — |
| 14 | `Standard_SoT/90` | object | 100.0% | 0.0% | 244 | '0.35' | — | — |
| 15 | `Standard_G/Sh` | object | 82.2% | 17.8% | 44 | '0.00' | — | — |
| 16 | `Standard_G/SoT` | object | 70.4% | 29.6% | 70 | '0.00' | — | — |
| 17 | `Standard_PK` | object | 100.0% | 0.0% | 15 | '0' | — | — |
| 18 | `Standard_PKatt` | object | 100.0% | 0.0% | 15 | '0' | — | — |
| 19 | `Unnamed: 18_level_0_Matches` | object | 100.0% | 0.0% | 1 | 'Matches' | — | — |
| 20 | `stat_type` | object | 100.0% | 0.0% | 1 | 'stats_shooting' | — | — |
| 21 | `source_file` | object | 100.0% | 0.0% | 1 | 'stats_shooting.html' | — | — |
| 22 | `season` | object | 100.0% | 0.0% | 9 | '2017-2018' | — | — |

---

## 4. TEAM STATS — FBref defense

### `data/parsed/fbref_stats_defense.parquet`

_Tackles, interceptions, blocks._

- **Format:** Parquet  
- **Size:** 104.8KB  
- **Modified:** 2026-03-02 08:20  
- **Rows:** 5,611  
- **Columns:** 28  
- **Seasons:** 9 covered — 2017-2018 → 2025-2026  

**Columns (28):**

| # | Column | Dtype | Filled % | NaN % | Unique | Sample | Min | Max |
|---|--------|-------|----------|-------|--------|--------|-----|-----|
| 1 | `Rk` | object | 100.0% | 0.0% | 635 | '1' | — | — |
| 2 | `Unnamed: 1_level_0_Player` | object | 100.0% | 0.0% | 1961 | 'Rolando Aarons' | — | — |
| 3 | `Unnamed: 2_level_0_Nation` | object | 100.0% | 0.0% | 104 | 'eng ENG' | — | — |
| 4 | `Unnamed: 3_level_0_Pos` | object | 100.0% | 0.0% | 11 | 'MF,FW' | — | — |
| 5 | `Unnamed: 4_level_0_Squad` | object | 100.0% | 0.0% | 33 | 'Hellas Verona' | — | — |
| 6 | `Unnamed: 5_level_0_Age` | object | 99.9% | 0.1% | 553 | '21' | — | — |
| 7 | `Unnamed: 6_level_0_Born` | object | 99.9% | 0.1% | 34 | '1995' | — | — |
| 8 | `Unnamed: 7_level_0_90s` | object | 100.0% | 0.0% | 375 | '5.7' | — | — |
| 9 | `Tackles_Tkl` | object | 3.8% | 96.2% | 1 | 'Tkl' | — | — |
| 10 | `Tackles_TklW` | object | 99.9% | 0.1% | 72 | '7' | — | — |
| 11 | `Tackles_Def 3rd` | object | 3.8% | 96.2% | 1 | 'Def 3rd' | — | — |
| 12 | `Tackles_Mid 3rd` | object | 3.8% | 96.2% | 1 | 'Mid 3rd' | — | — |
| 13 | `Tackles_Att 3rd` | object | 3.8% | 96.2% | 1 | 'Att 3rd' | — | — |
| 14 | `Challenges_Tkl` | object | 3.8% | 96.2% | 1 | 'Tkl' | — | — |
| 15 | `Challenges_Att` | object | 3.8% | 96.2% | 1 | 'Att' | — | — |
| 16 | `Challenges_Tkl%` | object | 3.8% | 96.2% | 1 | 'Tkl%' | — | — |
| 17 | `Challenges_Lost` | object | 3.8% | 96.2% | 1 | 'Lost' | — | — |
| 18 | `Blocks_Blocks` | object | 3.8% | 96.2% | 1 | 'Blocks' | — | — |
| 19 | `Blocks_Sh` | object | 3.8% | 96.2% | 1 | 'Sh' | — | — |
| 20 | `Blocks_Pass` | object | 3.8% | 96.2% | 1 | 'Pass' | — | — |
| 21 | `Unnamed: 20_level_0_Int` | object | 99.9% | 0.1% | 90 | '1' | — | — |
| 22 | `Unnamed: 21_level_0_Tkl+Int` | object | 3.8% | 96.2% | 1 | 'Tkl+Int' | — | — |
| 23 | `Unnamed: 22_level_0_Clr` | object | 3.8% | 96.2% | 1 | 'Clr' | — | — |
| 24 | `Unnamed: 23_level_0_Err` | object | 3.8% | 96.2% | 1 | 'Err' | — | — |
| 25 | `Unnamed: 24_level_0_Matches` | object | 100.0% | 0.0% | 1 | 'Matches' | — | — |
| 26 | `stat_type` | object | 100.0% | 0.0% | 1 | 'stats_defense' | — | — |
| 27 | `source_file` | object | 100.0% | 0.0% | 1 | 'stats_defense.html' | — | — |
| 28 | `season` | object | 100.0% | 0.0% | 9 | '2017-2018' | — | — |

---

## 4. TEAM STATS — FBref passing

### `data/parsed/fbref_stats_passing.parquet`

_Pass completion, distance, progression._

- **Format:** Parquet  
- **Size:** 100.1KB  
- **Modified:** 2026-03-02 08:20  
- **Rows:** 5,611  
- **Columns:** 32  
- **Seasons:** 9 covered — 2017-2018 → 2025-2026  

**Columns (32):**

| # | Column | Dtype | Filled % | NaN % | Unique | Sample | Min | Max |
|---|--------|-------|----------|-------|--------|--------|-----|-----|
| 1 | `Rk` | object | 100.0% | 0.0% | 635 | '1' | — | — |
| 2 | `Unnamed: 1_level_0_Player` | object | 100.0% | 0.0% | 1961 | 'Rolando Aarons' | — | — |
| 3 | `Unnamed: 2_level_0_Nation` | object | 100.0% | 0.0% | 104 | 'eng ENG' | — | — |
| 4 | `Unnamed: 3_level_0_Pos` | object | 100.0% | 0.0% | 11 | 'MF,FW' | — | — |
| 5 | `Unnamed: 4_level_0_Squad` | object | 100.0% | 0.0% | 33 | 'Hellas Verona' | — | — |
| 6 | `Unnamed: 5_level_0_Age` | object | 99.9% | 0.1% | 553 | '21' | — | — |
| 7 | `Unnamed: 6_level_0_Born` | object | 99.9% | 0.1% | 34 | '1995' | — | — |
| 8 | `Unnamed: 7_level_0_90s` | object | 100.0% | 0.0% | 375 | '5.7' | — | — |
| 9 | `Total_Cmp` | object | 3.8% | 96.2% | 1 | 'Cmp' | — | — |
| 10 | `Total_Att` | object | 3.8% | 96.2% | 1 | 'Att' | — | — |
| 11 | `Total_Cmp%` | object | 3.8% | 96.2% | 1 | 'Cmp%' | — | — |
| 12 | `Total_TotDist` | object | 3.8% | 96.2% | 1 | 'TotDist' | — | — |
| 13 | `Total_PrgDist` | object | 3.8% | 96.2% | 1 | 'PrgDist' | — | — |
| 14 | `Short_Cmp` | object | 3.8% | 96.2% | 1 | 'Cmp' | — | — |
| 15 | `Short_Att` | object | 3.8% | 96.2% | 1 | 'Att' | — | — |
| 16 | `Short_Cmp%` | object | 3.8% | 96.2% | 1 | 'Cmp%' | — | — |
| 17 | `Medium_Cmp` | object | 3.8% | 96.2% | 1 | 'Cmp' | — | — |
| 18 | `Medium_Att` | object | 3.8% | 96.2% | 1 | 'Att' | — | — |
| 19 | `Medium_Cmp%` | object | 3.8% | 96.2% | 1 | 'Cmp%' | — | — |
| 20 | `Long_Cmp` | object | 3.8% | 96.2% | 1 | 'Cmp' | — | — |
| 21 | `Long_Att` | object | 3.8% | 96.2% | 1 | 'Att' | — | — |
| 22 | `Long_Cmp%` | object | 3.8% | 96.2% | 1 | 'Cmp%' | — | — |
| 23 | `Unnamed: 22_level_0_Ast` | object | 100.0% | 0.0% | 18 | '0' | — | — |
| 24 | `Unnamed: 23_level_0_A-xAG` | object | 3.8% | 96.2% | 1 | 'A-xAG' | — | — |
| 25 | `Unnamed: 24_level_0_KP` | object | 3.8% | 96.2% | 1 | 'KP' | — | — |
| 26 | `Unnamed: 25_level_0_1/3` | object | 3.8% | 96.2% | 1 | '1/3' | — | — |
| 27 | `Unnamed: 26_level_0_PPA` | object | 3.8% | 96.2% | 1 | 'PPA' | — | — |
| 28 | `Unnamed: 27_level_0_CrsPA` | object | 3.8% | 96.2% | 1 | 'CrsPA' | — | — |
| 29 | `Unnamed: 28_level_0_Matches` | object | 100.0% | 0.0% | 1 | 'Matches' | — | — |
| 30 | `stat_type` | object | 100.0% | 0.0% | 1 | 'stats_passing' | — | — |
| 31 | `source_file` | object | 100.0% | 0.0% | 1 | 'stats_passing.html' | — | — |
| 32 | `season` | object | 100.0% | 0.0% | 9 | '2017-2018' | — | — |

---

## 4. TEAM STATS — FBref possession

### `data/parsed/fbref_stats_possession.parquet`

_Touches, take-ons, carries._

- **Format:** Parquet  
- **Size:** 97.0KB  
- **Modified:** 2026-03-02 08:20  
- **Rows:** 5,611  
- **Columns:** 32  
- **Seasons:** 9 covered — 2017-2018 → 2025-2026  

**Columns (32):**

| # | Column | Dtype | Filled % | NaN % | Unique | Sample | Min | Max |
|---|--------|-------|----------|-------|--------|--------|-----|-----|
| 1 | `Rk` | object | 100.0% | 0.0% | 635 | '1' | — | — |
| 2 | `Unnamed: 1_level_0_Player` | object | 100.0% | 0.0% | 1961 | 'Rolando Aarons' | — | — |
| 3 | `Unnamed: 2_level_0_Nation` | object | 100.0% | 0.0% | 104 | 'eng ENG' | — | — |
| 4 | `Unnamed: 3_level_0_Pos` | object | 100.0% | 0.0% | 11 | 'MF,FW' | — | — |
| 5 | `Unnamed: 4_level_0_Squad` | object | 100.0% | 0.0% | 33 | 'Hellas Verona' | — | — |
| 6 | `Unnamed: 5_level_0_Age` | object | 99.9% | 0.1% | 553 | '21' | — | — |
| 7 | `Unnamed: 6_level_0_Born` | object | 99.9% | 0.1% | 34 | '1995' | — | — |
| 8 | `Unnamed: 7_level_0_90s` | object | 100.0% | 0.0% | 375 | '5.7' | — | — |
| 9 | `Touches_Touches` | object | 3.8% | 96.2% | 1 | 'Touches' | — | — |
| 10 | `Touches_Def Pen` | object | 3.8% | 96.2% | 1 | 'Def Pen' | — | — |
| 11 | `Touches_Def 3rd` | object | 3.8% | 96.2% | 1 | 'Def 3rd' | — | — |
| 12 | `Touches_Mid 3rd` | object | 3.8% | 96.2% | 1 | 'Mid 3rd' | — | — |
| 13 | `Touches_Att 3rd` | object | 3.8% | 96.2% | 1 | 'Att 3rd' | — | — |
| 14 | `Touches_Att Pen` | object | 3.8% | 96.2% | 1 | 'Att Pen' | — | — |
| 15 | `Touches_Live` | object | 3.8% | 96.2% | 1 | 'Live' | — | — |
| 16 | `Take-Ons_Att` | object | 3.8% | 96.2% | 1 | 'Att' | — | — |
| 17 | `Take-Ons_Succ` | object | 3.8% | 96.2% | 1 | 'Succ' | — | — |
| 18 | `Take-Ons_Succ%` | object | 3.8% | 96.2% | 1 | 'Succ%' | — | — |
| 19 | `Take-Ons_Tkld` | object | 3.8% | 96.2% | 1 | 'Tkld' | — | — |
| 20 | `Take-Ons_Tkld%` | object | 3.8% | 96.2% | 1 | 'Tkld%' | — | — |
| 21 | `Carries_Carries` | object | 3.8% | 96.2% | 1 | 'Carries' | — | — |
| 22 | `Carries_TotDist` | object | 3.8% | 96.2% | 1 | 'TotDist' | — | — |
| 23 | `Carries_PrgDist` | object | 3.8% | 96.2% | 1 | 'PrgDist' | — | — |
| 24 | `Carries_1/3` | object | 3.8% | 96.2% | 1 | '1/3' | — | — |
| 25 | `Carries_CPA` | object | 3.8% | 96.2% | 1 | 'CPA' | — | — |
| 26 | `Carries_Mis` | object | 3.8% | 96.2% | 1 | 'Mis' | — | — |
| 27 | `Carries_Dis` | object | 3.8% | 96.2% | 1 | 'Dis' | — | — |
| 28 | `Unnamed: 27_level_0_Rec` | object | 3.8% | 96.2% | 1 | 'Rec' | — | — |
| 29 | `Unnamed: 28_level_0_Matches` | object | 100.0% | 0.0% | 1 | 'Matches' | — | — |
| 30 | `stat_type` | object | 100.0% | 0.0% | 1 | 'stats_possession' | — | — |
| 31 | `source_file` | object | 100.0% | 0.0% | 1 | 'stats_possession.html' | — | — |
| 32 | `season` | object | 100.0% | 0.0% | 9 | '2017-2018' | — | — |

---

## 4. TEAM STATS — FBref GCA (goal-creating actions)

### `data/parsed/fbref_stats_gca.parquet`

_Shot-creating actions and goal-creating actions by type._

- **Format:** Parquet  
- **Size:** 94.2KB  
- **Modified:** 2026-03-02 08:20  
- **Rows:** 5,611  
- **Columns:** 28  
- **Seasons:** 9 covered — 2017-2018 → 2025-2026  

**Columns (28):**

| # | Column | Dtype | Filled % | NaN % | Unique | Sample | Min | Max |
|---|--------|-------|----------|-------|--------|--------|-----|-----|
| 1 | `Rk` | object | 100.0% | 0.0% | 635 | '1' | — | — |
| 2 | `Unnamed: 1_level_0_Player` | object | 100.0% | 0.0% | 1961 | 'Rolando Aarons' | — | — |
| 3 | `Unnamed: 2_level_0_Nation` | object | 100.0% | 0.0% | 104 | 'eng ENG' | — | — |
| 4 | `Unnamed: 3_level_0_Pos` | object | 100.0% | 0.0% | 11 | 'MF,FW' | — | — |
| 5 | `Unnamed: 4_level_0_Squad` | object | 100.0% | 0.0% | 33 | 'Hellas Verona' | — | — |
| 6 | `Unnamed: 5_level_0_Age` | object | 99.9% | 0.1% | 553 | '21' | — | — |
| 7 | `Unnamed: 6_level_0_Born` | object | 99.9% | 0.1% | 34 | '1995' | — | — |
| 8 | `Unnamed: 7_level_0_90s` | object | 100.0% | 0.0% | 375 | '5.7' | — | — |
| 9 | `SCA_SCA` | object | 3.8% | 96.2% | 1 | 'SCA' | — | — |
| 10 | `SCA_SCA90` | object | 3.8% | 96.2% | 1 | 'SCA90' | — | — |
| 11 | `SCA Types_PassLive` | object | 3.8% | 96.2% | 1 | 'PassLive' | — | — |
| 12 | `SCA Types_PassDead` | object | 3.8% | 96.2% | 1 | 'PassDead' | — | — |
| 13 | `SCA Types_TO` | object | 3.8% | 96.2% | 1 | 'TO' | — | — |
| 14 | `SCA Types_Sh` | object | 3.8% | 96.2% | 1 | 'Sh' | — | — |
| 15 | `SCA Types_Fld` | object | 3.8% | 96.2% | 1 | 'Fld' | — | — |
| 16 | `SCA Types_Def` | object | 3.8% | 96.2% | 1 | 'Def' | — | — |
| 17 | `GCA_GCA` | object | 3.8% | 96.2% | 1 | 'GCA' | — | — |
| 18 | `GCA_GCA90` | object | 3.8% | 96.2% | 1 | 'GCA90' | — | — |
| 19 | `GCA Types_PassLive` | object | 3.8% | 96.2% | 1 | 'PassLive' | — | — |
| 20 | `GCA Types_PassDead` | object | 3.8% | 96.2% | 1 | 'PassDead' | — | — |
| 21 | `GCA Types_TO` | object | 3.8% | 96.2% | 1 | 'TO' | — | — |
| 22 | `GCA Types_Sh` | object | 3.8% | 96.2% | 1 | 'Sh' | — | — |
| 23 | `GCA Types_Fld` | object | 3.8% | 96.2% | 1 | 'Fld' | — | — |
| 24 | `GCA Types_Def` | object | 3.8% | 96.2% | 1 | 'Def' | — | — |
| 25 | `Unnamed: 24_level_0_Matches` | object | 100.0% | 0.0% | 1 | 'Matches' | — | — |
| 26 | `stat_type` | object | 100.0% | 0.0% | 1 | 'stats_gca' | — | — |
| 27 | `source_file` | object | 100.0% | 0.0% | 1 | 'stats_gca.html' | — | — |
| 28 | `season` | object | 100.0% | 0.0% | 9 | '2017-2018' | — | — |

---

## 4. TEAM STATS — FBref keepers

### `data/parsed/fbref_stats_keepers.parquet`

_Season GK stats._

- **Format:** Parquet  
- **Size:** 35.2KB  
- **Modified:** 2026-03-02 08:20  
- **Rows:** 435  
- **Columns:** 30  
- **Seasons:** 9 covered — 2017-2018 → 2025-2026  

**Columns (30):**

| # | Column | Dtype | Filled % | NaN % | Unique | Sample | Min | Max |
|---|--------|-------|----------|-------|--------|--------|-----|-----|
| 1 | `Rk` | object | 100.0% | 0.0% | 52 | '1' | — | — |
| 2 | `Unnamed: 1_level_0_Player` | object | 100.0% | 0.0% | 149 | 'Alisson' | — | — |
| 3 | `Unnamed: 2_level_0_Nation` | object | 100.0% | 0.0% | 32 | 'br BRA' | — | — |
| 4 | `Unnamed: 3_level_0_Pos` | object | 100.0% | 0.0% | 2 | 'GK' | — | — |
| 5 | `Unnamed: 4_level_0_Squad` | object | 100.0% | 0.0% | 33 | 'Roma' | — | — |
| 6 | `Unnamed: 5_level_0_Age` | object | 100.0% | 0.0% | 65 | '24' | — | — |
| 7 | `Unnamed: 6_level_0_Born` | object | 100.0% | 0.0% | 31 | '1992' | — | — |
| 8 | `Playing Time_MP` | object | 100.0% | 0.0% | 39 | '37' | — | — |
| 9 | `Playing Time_Starts` | object | 100.0% | 0.0% | 40 | '37' | — | — |
| 10 | `Playing Time_Min` | object | 99.8% | 0.2% | 172 | '3330' | — | — |
| 11 | `Playing Time_90s` | object | 100.0% | 0.0% | 128 | '37.0' | — | — |
| 12 | `Performance_GA` | object | 100.0% | 0.0% | 71 | '28' | — | — |
| 13 | `Performance_GA90` | object | 99.8% | 0.2% | 157 | '0.76' | — | — |
| 14 | `Performance_SoTA` | object | 100.0% | 0.0% | 148 | '109' | — | — |
| 15 | `Performance_Saves` | object | 100.0% | 0.0% | 130 | '81' | — | — |
| 16 | `Performance_Save%` | object | 96.3% | 3.7% | 176 | '77.1' | — | — |
| 17 | `Performance_W` | object | 100.0% | 0.0% | 29 | '22' | — | — |
| 18 | `Performance_D` | object | 100.0% | 0.0% | 18 | '8' | — | — |
| 19 | `Performance_L` | object | 100.0% | 0.0% | 26 | '7' | — | — |
| 20 | `Performance_CS` | object | 100.0% | 0.0% | 21 | '17' | — | — |
| 21 | `Performance_CS%` | object | 91.5% | 8.5% | 126 | '45.9' | — | — |
| 22 | `Penalty Kicks_PKatt` | object | 100.0% | 0.0% | 15 | '5' | — | — |
| 23 | `Penalty Kicks_PKA` | object | 100.0% | 0.0% | 13 | '3' | — | — |
| 24 | `Penalty Kicks_PKsv` | object | 100.0% | 0.0% | 6 | '2' | — | — |
| 25 | `Penalty Kicks_PKm` | object | 100.0% | 0.0% | 5 | '0' | — | — |
| 26 | `Penalty Kicks_Save%` | object | 67.1% | 32.9% | 19 | '40.0' | — | — |
| 27 | `Unnamed: 26_level_0_Matches` | object | 100.0% | 0.0% | 1 | 'Matches' | — | — |
| 28 | `stat_type` | object | 100.0% | 0.0% | 1 | 'stats_keepers' | — | — |
| 29 | `source_file` | object | 100.0% | 0.0% | 1 | 'stats_keepers.html' | — | — |
| 30 | `season` | object | 100.0% | 0.0% | 9 | '2017-2018' | — | — |

---

## 4. TEAM STATS — FBref keepers advanced

### `data/parsed/fbref_stats_keepers_adv.parquet`

_Advanced GK: PSxG, launches._

- **Format:** Parquet  
- **Size:** 29.4KB  
- **Modified:** 2026-03-02 08:20  
- **Rows:** 435  
- **Columns:** 37  
- **Seasons:** 9 covered — 2017-2018 → 2025-2026  

**Columns (37):**

| # | Column | Dtype | Filled % | NaN % | Unique | Sample | Min | Max |
|---|--------|-------|----------|-------|--------|--------|-----|-----|
| 1 | `Rk` | object | 100.0% | 0.0% | 52 | '1' | — | — |
| 2 | `Unnamed: 1_level_0_Player` | object | 100.0% | 0.0% | 149 | 'Alisson' | — | — |
| 3 | `Unnamed: 2_level_0_Nation` | object | 100.0% | 0.0% | 32 | 'br BRA' | — | — |
| 4 | `Unnamed: 3_level_0_Pos` | object | 100.0% | 0.0% | 2 | 'GK' | — | — |
| 5 | `Unnamed: 4_level_0_Squad` | object | 100.0% | 0.0% | 33 | 'Roma' | — | — |
| 6 | `Unnamed: 5_level_0_Age` | object | 100.0% | 0.0% | 65 | '24' | — | — |
| 7 | `Unnamed: 6_level_0_Born` | object | 100.0% | 0.0% | 31 | '1992' | — | — |
| 8 | `Unnamed: 7_level_0_90s` | object | 100.0% | 0.0% | 128 | '37.0' | — | — |
| 9 | `Goals_GA` | object | 100.0% | 0.0% | 71 | '28' | — | — |
| 10 | `Goals_PKA` | object | 100.0% | 0.0% | 13 | '3' | — | — |
| 11 | `Goals_FK` | object | 2.5% | 97.5% | 1 | 'FK' | — | — |
| 12 | `Goals_CK` | object | 2.5% | 97.5% | 1 | 'CK' | — | — |
| 13 | `Goals_OG` | object | 2.5% | 97.5% | 1 | 'OG' | — | — |
| 14 | `Expected_PSxG` | object | 2.5% | 97.5% | 1 | 'PSxG' | — | — |
| 15 | `Expected_PSxG/SoT` | object | 2.5% | 97.5% | 1 | 'PSxG/SoT' | — | — |
| 16 | `Expected_PSxG+/-` | object | 2.5% | 97.5% | 1 | 'PSxG+/-' | — | — |
| 17 | `Expected_/90` | object | 2.5% | 97.5% | 1 | '/90' | — | — |
| 18 | `Launched_Cmp` | object | 2.5% | 97.5% | 1 | 'Cmp' | — | — |
| 19 | `Launched_Att` | object | 2.5% | 97.5% | 1 | 'Att' | — | — |
| 20 | `Launched_Cmp%` | object | 2.5% | 97.5% | 1 | 'Cmp%' | — | — |
| 21 | `Passes_Att (GK)` | object | 2.5% | 97.5% | 1 | 'Att (GK)' | — | — |
| 22 | `Passes_Thr` | object | 2.5% | 97.5% | 1 | 'Thr' | — | — |
| 23 | `Passes_Launch%` | object | 2.5% | 97.5% | 1 | 'Launch%' | — | — |
| 24 | `Passes_AvgLen` | object | 2.5% | 97.5% | 1 | 'AvgLen' | — | — |
| 25 | `Goal Kicks_Att` | object | 2.5% | 97.5% | 1 | 'Att' | — | — |
| 26 | `Goal Kicks_Launch%` | object | 2.5% | 97.5% | 1 | 'Launch%' | — | — |
| 27 | `Goal Kicks_AvgLen` | object | 2.5% | 97.5% | 1 | 'AvgLen' | — | — |
| 28 | `Crosses_Opp` | object | 2.5% | 97.5% | 1 | 'Opp' | — | — |
| 29 | `Crosses_Stp` | object | 2.5% | 97.5% | 1 | 'Stp' | — | — |
| 30 | `Crosses_Stp%` | object | 2.5% | 97.5% | 1 | 'Stp%' | — | — |
| 31 | `Sweeper_#OPA` | object | 2.5% | 97.5% | 1 | '#OPA' | — | — |
| 32 | `Sweeper_#OPA/90` | object | 2.5% | 97.5% | 1 | '#OPA/90' | — | — |
| 33 | `Sweeper_AvgDist` | object | 2.5% | 97.5% | 1 | 'AvgDist' | — | — |
| 34 | `Unnamed: 33_level_0_Matches` | object | 100.0% | 0.0% | 1 | 'Matches' | — | — |
| 35 | `stat_type` | object | 100.0% | 0.0% | 1 | 'stats_keepers_adv' | — | — |
| 36 | `source_file` | object | 100.0% | 0.0% | 1 | 'stats_keepers_adv.html' | — | — |
| 37 | `season` | object | 100.0% | 0.0% | 9 | '2017-2018' | — | — |

---

## 4. TEAM STATS — FBref misc (aerials, fouls)

### `data/parsed/fbref_stats_misc.parquet`

_Cards, fouls, fouled, offsides._

- **Format:** Parquet  
- **Size:** 129.9KB  
- **Modified:** 2026-03-02 08:20  
- **Rows:** 5,611  
- **Columns:** 24  
- **Seasons:** 9 covered — 2017-2018 → 2025-2026  

**Columns (24):**

| # | Column | Dtype | Filled % | NaN % | Unique | Sample | Min | Max |
|---|--------|-------|----------|-------|--------|--------|-----|-----|
| 1 | `Rk` | object | 100.0% | 0.0% | 635 | '1' | — | — |
| 2 | `Unnamed: 1_level_0_Player` | object | 100.0% | 0.0% | 1961 | 'Rolando Aarons' | — | — |
| 3 | `Unnamed: 2_level_0_Nation` | object | 100.0% | 0.0% | 104 | 'eng ENG' | — | — |
| 4 | `Unnamed: 3_level_0_Pos` | object | 100.0% | 0.0% | 11 | 'MF,FW' | — | — |
| 5 | `Unnamed: 4_level_0_Squad` | object | 100.0% | 0.0% | 33 | 'Hellas Verona' | — | — |
| 6 | `Unnamed: 5_level_0_Age` | object | 99.9% | 0.1% | 553 | '21' | — | — |
| 7 | `Unnamed: 6_level_0_Born` | object | 99.9% | 0.1% | 34 | '1995' | — | — |
| 8 | `Unnamed: 7_level_0_90s` | object | 100.0% | 0.0% | 375 | '5.7' | — | — |
| 9 | `Performance_CrdY` | object | 100.0% | 0.0% | 18 | '0' | — | — |
| 10 | `Performance_CrdR` | object | 100.0% | 0.0% | 5 | '0' | — | — |
| 11 | `Performance_2CrdY` | object | 99.9% | 0.1% | 4 | '0' | — | — |
| 12 | `Performance_Fls` | object | 100.0% | 0.0% | 82 | '8' | — | — |
| 13 | `Performance_Fld` | object | 99.9% | 0.1% | 101 | '11' | — | — |
| 14 | `Performance_Off` | object | 99.9% | 0.1% | 44 | '0' | — | — |
| 15 | `Performance_Crs` | object | 99.9% | 0.1% | 208 | '11' | — | — |
| 16 | `Performance_Int` | object | 99.9% | 0.1% | 90 | '1' | — | — |
| 17 | `Performance_TklW` | object | 99.9% | 0.1% | 72 | '7' | — | — |
| 18 | `Performance_PKwon` | object | 23.6% | 76.4% | 6 | '0' | — | — |
| 19 | `Performance_PKcon` | object | 23.6% | 76.4% | 5 | '0' | — | — |
| 20 | `Performance_OG` | object | 99.9% | 0.1% | 5 | '0' | — | — |
| 21 | `Unnamed: 20_level_0_Matches` | object | 100.0% | 0.0% | 1 | 'Matches' | — | — |
| 22 | `stat_type` | object | 100.0% | 0.0% | 1 | 'stats_misc' | — | — |
| 23 | `source_file` | object | 100.0% | 0.0% | 1 | 'stats_misc.html' | — | — |
| 24 | `season` | object | 100.0% | 0.0% | 9 | '2017-2018' | — | — |

---

## 4. TEAM STATS — FBref passing types

### `data/parsed/fbref_stats_passing_types.parquet`

_Pass types, corner kicks, throw-ins._

- **Format:** Parquet  
- **Size:** 100.0KB  
- **Modified:** 2026-03-02 08:20  
- **Rows:** 5,611  
- **Columns:** 27  
- **Seasons:** 9 covered — 2017-2018 → 2025-2026  

**Columns (27):**

| # | Column | Dtype | Filled % | NaN % | Unique | Sample | Min | Max |
|---|--------|-------|----------|-------|--------|--------|-----|-----|
| 1 | `Rk` | object | 100.0% | 0.0% | 635 | '1' | — | — |
| 2 | `Unnamed: 1_level_0_Player` | object | 100.0% | 0.0% | 1961 | 'Rolando Aarons' | — | — |
| 3 | `Unnamed: 2_level_0_Nation` | object | 100.0% | 0.0% | 104 | 'eng ENG' | — | — |
| 4 | `Unnamed: 3_level_0_Pos` | object | 100.0% | 0.0% | 11 | 'MF,FW' | — | — |
| 5 | `Unnamed: 4_level_0_Squad` | object | 100.0% | 0.0% | 33 | 'Hellas Verona' | — | — |
| 6 | `Unnamed: 5_level_0_Age` | object | 99.9% | 0.1% | 553 | '21' | — | — |
| 7 | `Unnamed: 6_level_0_Born` | object | 99.9% | 0.1% | 34 | '1995' | — | — |
| 8 | `Unnamed: 7_level_0_90s` | object | 100.0% | 0.0% | 375 | '5.7' | — | — |
| 9 | `Unnamed: 8_level_0_Att` | object | 3.8% | 96.2% | 1 | 'Att' | — | — |
| 10 | `Pass Types_Live` | object | 3.8% | 96.2% | 1 | 'Live' | — | — |
| 11 | `Pass Types_Dead` | object | 3.8% | 96.2% | 1 | 'Dead' | — | — |
| 12 | `Pass Types_FK` | object | 3.8% | 96.2% | 1 | 'FK' | — | — |
| 13 | `Pass Types_TB` | object | 3.8% | 96.2% | 1 | 'TB' | — | — |
| 14 | `Pass Types_Sw` | object | 3.8% | 96.2% | 1 | 'Sw' | — | — |
| 15 | `Pass Types_Crs` | object | 99.9% | 0.1% | 208 | '11' | — | — |
| 16 | `Pass Types_TI` | object | 3.8% | 96.2% | 1 | 'TI' | — | — |
| 17 | `Pass Types_CK` | object | 3.8% | 96.2% | 1 | 'CK' | — | — |
| 18 | `Corner Kicks_In` | object | 3.8% | 96.2% | 1 | 'In' | — | — |
| 19 | `Corner Kicks_Out` | object | 3.8% | 96.2% | 1 | 'Out' | — | — |
| 20 | `Corner Kicks_Str` | object | 3.8% | 96.2% | 1 | 'Str' | — | — |
| 21 | `Outcomes_Cmp` | object | 3.8% | 96.2% | 1 | 'Cmp' | — | — |
| 22 | `Outcomes_Off` | object | 3.8% | 96.2% | 1 | 'Off' | — | — |
| 23 | `Outcomes_Blocks` | object | 3.8% | 96.2% | 1 | 'Blocks' | — | — |
| 24 | `Unnamed: 23_level_0_Matches` | object | 100.0% | 0.0% | 1 | 'Matches' | — | — |
| 25 | `stat_type` | object | 100.0% | 0.0% | 1 | 'stats_passing_types' | — | — |
| 26 | `source_file` | object | 100.0% | 0.0% | 1 | 'stats_passing_types.html' | — | — |
| 27 | `season` | object | 100.0% | 0.0% | 9 | '2017-2018' | — | — |

---

## 5. XG — Understat match-level (all seasons)

### `data/external/understat/matches_xg.parquet`

_Understat's match-level xG for home/away. Independent xG model._

- **Format:** Parquet  
- **Size:** 181.0KB  
- **Modified:** 2026-04-21 17:54  
- **Rows:** 3,370  
- **Columns:** 17  
- **Date column:** `datetime` — range 2017-08-19 → 2026-04-20  
- **Seasons:** 9 covered — 2017-2018 → 2025-2026  

**Columns (17):**

| # | Column | Dtype | Filled % | NaN % | Unique | Sample | Min | Max |
|---|--------|-------|----------|-------|--------|--------|-----|-----|
| 1 | `match_id` | object | 100.0% | 0.0% | 3370 | '7499' | — | — |
| 2 | `datetime` | object | 100.0% | 0.0% | 2586 | '2017-08-19 17:00:00' | — | — |
| 3 | `season` | object | 100.0% | 0.0% | 9 | '2017-2018' | — | — |
| 4 | `home_team` | object | 100.0% | 0.0% | 32 | 'Juventus' | — | — |
| 5 | `away_team` | object | 100.0% | 0.0% | 32 | 'Cagliari' | — | — |
| 6 | `home_short` | object | 100.0% | 0.0% | 32 | 'JUV' | — | — |
| 7 | `away_short` | object | 100.0% | 0.0% | 32 | 'CAG' | — | — |
| 8 | `home_id` | object | 100.0% | 0.0% | 32 | '98' | — | — |
| 9 | `away_id` | object | 100.0% | 0.0% | 32 | '116' | — | — |
| 10 | `home_goals` | float64 | 100.0% | 0.0% | 9 | median=1 | 0 | 8 |
| 11 | `away_goals` | float64 | 100.0% | 0.0% | 8 | median=1 | 0 | 7 |
| 12 | `home_xg` | float64 | 100.0% | 0.0% | 3356 | median=1.37 | 0.0129 | 6.24 |
| 13 | `away_xg` | float64 | 100.0% | 0.0% | 3353 | median=1.13 | 0 | 5.41 |
| 14 | `forecast_home` | float64 | 100.0% | 0.0% | 2814 | median=0.402 | 0.0004 | 0.999 |
| 15 | `forecast_draw` | float64 | 100.0% | 0.0% | 2259 | median=0.24 | 0.0011 | 0.844 |
| 16 | `forecast_away` | float64 | 100.0% | 0.0% | 2695 | median=0.274 | 0 | 0.998 |
| 17 | `is_result` | bool | 100.0% | 0.0% | 1 | np.True_ | — | — |

---

## 5. XG — Understat current season

### `data/external/understat/matches_xg_2025_2026.parquet`

**(NOT FOUND)**

_2025-2026 match xG. Uses different format (home_id, away_id)._

---

## 5. XG — Understat normalized

### `data/external/understat/matches_xg_normalized.parquet`

**(NOT FOUND)**

_Harmonized schema for Understat match xG._

---

## 5. XG — Understat player-level

### `data/external/understat/players_xg_2025_2026.parquet`

_Per-player xG in the current season._

- **Format:** Parquet  
- **Size:** 59.4KB  
- **Modified:** 2026-03-01 23:31  
- **Rows:** 551  
- **Columns:** 19  
- **Seasons:** 1 covered — 2025-2026 → 2025-2026  

**Columns (19):**

| # | Column | Dtype | Filled % | NaN % | Unique | Sample | Min | Max |
|---|--------|-------|----------|-------|--------|--------|-----|-----|
| 1 | `assists` | int64 | 100.0% | 0.0% | 9 | median=0 | 0 | 13 |
| 2 | `games` | int64 | 100.0% | 0.0% | 27 | median=16 | 1 | 27 |
| 3 | `goals` | int64 | 100.0% | 0.0% | 11 | median=0 | 0 | 14 |
| 4 | `id` | object | 100.0% | 0.0% | 551 | '7006' | — | — |
| 5 | `key_passes` | int64 | 100.0% | 0.0% | 48 | median=5 | 0 | 73 |
| 6 | `npg` | int64 | 100.0% | 0.0% | 11 | median=0 | 0 | 14 |
| 7 | `npxG` | float64 | 100.0% | 0.0% | 461 | median=0.655 | 0 | 14.6 |
| 8 | `player_name` | object | 100.0% | 0.0% | 551 | 'Lautaro Martínez' | — | — |
| 9 | `position` | object | 100.0% | 0.0% | 14 | 'F S' | — | — |
| 10 | `red_cards` | int64 | 100.0% | 0.0% | 3 | median=0 | 0 | 2 |
| 11 | `shots` | int64 | 100.0% | 0.0% | 58 | median=8 | 0 | 94 |
| 12 | `team_title` | object | 100.0% | 0.0% | 39 | 'Inter' | — | — |
| 13 | `time` | object | 100.0% | 0.0% | 474 | '1928' | — | — |
| 14 | `xA` | float64 | 100.0% | 0.0% | 466 | median=0.499 | 0 | 12.3 |
| 15 | `xG` | float64 | 100.0% | 0.0% | 461 | median=0.674 | 0 | 14.8 |
| 16 | `xGBuildup` | object | 100.0% | 0.0% | 512 | '8.480899337679148' | — | — |
| 17 | `xGChain` | object | 100.0% | 0.0% | 532 | '23.09404458105564' | — | — |
| 18 | `yellow_cards` | int64 | 100.0% | 0.0% | 10 | median=1 | 0 | 9 |
| 19 | `season` | object | 100.0% | 0.0% | 1 | '2025-2026' | — | — |

---

## 6. CONTEXT — weather per match

### `data/external/weather.parquet`

_Temperature, wind, precipitation, humidity per match_id._

- **Format:** Parquet  
- **Size:** 262.3KB  
- **Modified:** 2026-04-21 18:52  
- **Rows:** 11,433  
- **Columns:** 13  

**Columns (13):**

| # | Column | Dtype | Filled % | NaN % | Unique | Sample | Min | Max |
|---|--------|-------|----------|-------|--------|--------|-----|-----|
| 1 | `match_id` | object | 100.0% | 0.0% | 11433 | '2025-04-13_Atalanta_Bologna' | — | — |
| 2 | `weather_temperature_2m_max` | float64 | 82.6% | 17.4% | 362 | median=14.9 | -6.1 | 38.1 |
| 3 | `weather_temperature_2m_min` | float64 | 82.6% | 17.4% | 318 | median=7.6 | -18.4 | 25.4 |
| 4 | `weather_temperature_2m_mean` | float64 | 82.6% | 17.4% | 337 | median=11.1 | -11.3 | 30.3 |
| 5 | `weather_apparent_temperature_max` | float64 | 82.6% | 17.4% | 433 | median=13 | -9.7 | 40.2 |
| 6 | `weather_apparent_temperature_min` | float64 | 82.6% | 17.4% | 392 | median=5 | -22.6 | 28.8 |
| 7 | `weather_precipitation_sum` | float64 | 82.6% | 17.4% | 404 | median=0.2 | 0 | 100 |
| 8 | `weather_rain_sum` | float64 | 82.6% | 17.4% | 397 | median=0.2 | 0 | 100 |
| 9 | `weather_snowfall_sum` | float64 | 82.6% | 17.4% | 81 | median=0 | 0 | 17.3 |
| 10 | `weather_wind_speed_10m_max` | float64 | 82.6% | 17.4% | 443 | median=13.8 | 3.1 | 66.6 |
| 11 | `weather_wind_gusts_10m_max` | float64 | 82.6% | 17.4% | 263 | median=30.2 | 6.8 | 122 |
| 12 | `weather_wind_direction_10m_dominant` | float64 | 82.6% | 17.4% | 361 | median=171 | 0 | 360 |
| 13 | `weather_relative_humidity_2m_mean` | float64 | 82.6% | 17.4% | 68 | median=79 | 28 | 99 |

---

## 6. CONTEXT — referees

### `data/external/referee/referee_assignments.parquet`

_Ref name, yellows/reds given, per match._

- **Format:** Parquet  
- **Size:** 27.0KB  
- **Modified:** 2026-04-21 16:40  
- **Rows:** 3,368  
- **Columns:** 9  
- **Date column:** `match_date` — range 2017-08-19 → 2026-04-20  
- **Seasons:** 9 covered — 2017-2018 → 2025-2026  

**Columns (9):**

| # | Column | Dtype | Filled % | NaN % | Unique | Sample | Min | Max |
|---|--------|-------|----------|-------|--------|--------|-----|-----|
| 1 | `match_date` | object | 100.0% | 0.0% | 1059 | '2017-08-20' | — | — |
| 2 | `home_team` | object | 100.0% | 0.0% | 32 | 'Bologna' | — | — |
| 3 | `away_team` | object | 100.0% | 0.0% | 32 | 'Torino' | — | — |
| 4 | `referee` | object | 100.0% | 0.0% | 77 | 'Davide Massa' | — | — |
| 5 | `matchweek` | int64 | 100.0% | 0.0% | 38 | median=19 | 1 | 38 |
| 6 | `ref_yellows` | int64 | 100.0% | 0.0% | 14 | median=4 | 0 | 13 |
| 7 | `ref_second_yellows` | int64 | 100.0% | 0.0% | 4 | median=0 | 0 | 3 |
| 8 | `ref_reds` | int64 | 100.0% | 0.0% | 4 | median=0 | 0 | 3 |
| 9 | `season` | object | 100.0% | 0.0% | 9 | '2017-2018' | — | — |

---

## 6. CONTEXT — player market values (current season)

### `data/external/transfermarkt/market_values_2024_2025.parquet`

_Transfermarkt market value per player (EUR)._

- **Format:** Parquet  
- **Size:** 20.1KB  
- **Modified:** 2026-02-13 00:02  
- **Rows:** 861  
- **Columns:** 6  

**Columns (6):**

| # | Column | Dtype | Filled % | NaN % | Unique | Sample | Min | Max |
|---|--------|-------|----------|-------|--------|--------|-----|-----|
| 1 | `team` | object | 100.0% | 0.0% | 20 | 'Atalanta' | — | — |
| 2 | `player_name` | object | 100.0% | 0.0% | 811 | 'Marco Carnesecchi' | — | — |
| 3 | `position` | object | 97.7% | 2.3% | 4 | 'GK' | — | — |
| 4 | `age` | float64 | 66.4% | 33.6% | 86 | median=21 | 1 | 99 |
| 5 | `market_value_eur` | float64 | 100.0% | 0.0% | 93 | median=2.5e+06 | 0 | 9.5e+07 |
| 6 | `nationality` | object | 100.0% | 0.0% | 83 | 'Italy' | — | — |

---

## 6. CONTEXT — injuries (latest snapshot)

### `data/external/injuries/injuries_2026-04-17.parquet`

_Weekly injury snapshots. Latest = most recent._

- **Format:** Parquet  
- **Size:** 7.2KB  
- **Modified:** 2026-04-17 03:05  
- **Rows:** 63  
- **Columns:** 8  

**Columns (8):**

| # | Column | Dtype | Filled % | NaN % | Unique | Sample | Min | Max |
|---|--------|-------|----------|-------|--------|--------|-----|-----|
| 1 | `player_name` | object | 100.0% | 0.0% | 63 | 'Francesco Rossi' | — | — |
| 2 | `team` | object | 100.0% | 0.0% | 19 | 'Atalanta' | — | — |
| 3 | `injury_type` | object | 100.0% | 0.0% | 32 | 'unknown injury' | — | — |
| 4 | `start_date` | object | 0.0% | 100.0% | 0 | — | — | — |
| 5 | `expected_return` | object | 55.6% | 44.4% | 21 | datetime.date(2026, 6, 30) | — | — |
| 6 | `is_currently_out` | bool | 100.0% | 0.0% | 1 | np.True_ | — | — |
| 7 | `source` | object | 100.0% | 0.0% | 1 | 'transfermarkt' | — | — |
| 8 | `scraped_at` | datetime64[ns] | 100.0% | 0.0% | 63 | — | 2026-04-17 | 2026-04-17 |

---

## 7. FEATURES — final ML feature table (Serie A)

### `data/features/features_serie_a.parquet`

_**The input to models.** 1066 cols, merged from all sources above._

- **Format:** Parquet  
- **Size:** 9.8MB  
- **Modified:** 2026-04-21 18:26  
- **Rows:** 7,930  
- **Columns:** 1059  
- **Date column:** `match_date` — range 2005-08-27 → 2026-04-20  
- **League distribution:** {'serie_a': 7930} (7930/7930 = 100.0% Serie A)  
- **Seasons:** 21 covered — 2005-2006 → 2025-2026  

**Columns (1059):**

| # | Column | Dtype | Filled % | NaN % | Unique | Sample | Min | Max |
|---|--------|-------|----------|-------|--------|--------|-----|-----|
| 1 | `home_team` | object | 100.0% | 0.0% | 45 | 'Fiorentina' | — | — |
| 2 | `away_team` | object | 100.0% | 0.0% | 45 | 'Sampdoria' | — | — |
| 3 | `match_date` | datetime64[ns] | 100.0% | 0.0% | 2098 | — | 2005-08-27 | 2026-04-20 |
| 4 | `home_score` | float64 | 100.0% | 0.0% | 9 | median=1 | 0 | 8 |
| 5 | `away_score` | float64 | 100.0% | 0.0% | 8 | median=1 | 0 | 7 |
| 6 | `result` | object | 100.0% | 0.0% | 3 | 'H' | — | — |
| 7 | `season` | object | 100.0% | 0.0% | 21 | '2005-2006' | — | — |
| 8 | `league` | object | 100.0% | 0.0% | 1 | 'serie_a' | — | — |
| 9 | `home_shots_total` | float64 | 99.8% | 0.2% | 41 | median=13 | 1 | 46 |
| 10 | `away_shots_total` | float64 | 99.8% | 0.2% | 33 | median=11 | 0 | 34 |
| 11 | `home_shots_on_target_count` | float64 | 99.8% | 0.2% | 18 | median=5 | 0 | 18 |
| 12 | `away_shots_on_target_count` | float64 | 99.8% | 0.2% | 17 | median=4 | 0 | 16 |
| 13 | `home_fouls` | float64 | 99.9% | 0.1% | 39 | median=14 | 1 | 48 |
| 14 | `away_fouls` | float64 | 99.9% | 0.1% | 39 | median=15 | 1 | 41 |
| 15 | `home_corners` | float64 | 99.9% | 0.1% | 22 | median=5 | 0 | 21 |
| 16 | `away_corners` | float64 | 99.9% | 0.1% | 20 | median=4 | 0 | 19 |
| 17 | `home_yellow_cards` | float64 | 99.9% | 0.1% | 8 | median=2 | 0 | 7 |
| 18 | `away_yellow_cards` | float64 | 99.9% | 0.1% | 9 | median=2 | 0 | 8 |
| 19 | `home_red_cards` | float64 | 99.9% | 0.1% | 4 | median=0 | 0 | 3 |
| 20 | `away_red_cards` | float64 | 99.9% | 0.1% | 4 | median=0 | 0 | 3 |
| 21 | `home_ht_goals` | float64 | 99.1% | 0.9% | 6 | median=0 | 0 | 5 |
| 22 | `away_ht_goals` | float64 | 99.1% | 0.9% | 6 | median=0 | 0 | 5 |
| 23 | `ht_result` | object | 99.1% | 0.9% | 3 | 'H' | — | — |
| 24 | `referee` | object | 52.1% | 47.9% | 125 | 'G. Paparesta' | — | — |
| 25 | `odds_B365H` | float64 | 99.0% | 1.0% | 125 | median=2.2 | 1.06 | 21 |
| 26 | `odds_B365D` | float64 | 99.0% | 1.0% | 77 | median=3.4 | 1.4 | 15 |
| 27 | `odds_B365A` | float64 | 99.0% | 1.0% | 124 | median=3.5 | 1.1 | 34 |
| 28 | `odds_AvgH` | float64 | 99.7% | 0.3% | 735 | median=2.18 | 1.05 | 18.6 |
| 29 | `odds_AvgD` | float64 | 99.7% | 0.3% | 553 | median=3.43 | 1.29 | 14.4 |
| 30 | `odds_AvgA` | float64 | 99.7% | 0.3% | 1246 | median=3.47 | 1.1 | 37.9 |
| 31 | `odds_MaxH` | float64 | 99.7% | 0.3% | 563 | median=2.27 | 1.06 | 35 |
| 32 | `odds_MaxD` | float64 | 99.7% | 0.3% | 489 | median=3.6 | 1.33 | 19 |
| 33 | `odds_MaxA` | float64 | 99.7% | 0.3% | 881 | median=3.76 | 1.13 | 61 |
| 34 | `odds_Avg_over25` | float64 | 99.1% | 0.9% | 157 | median=1.94 | 1.18 | 2.86 |
| 35 | `odds_Avg_under25` | float64 | 99.1% | 0.9% | 209 | median=1.84 | 1.41 | 4.67 |
| 36 | `match_id` | object | 100.0% | 0.0% | 7930 | '2005-08-27_Fiorentina_Sampdor | — | — |
| 37 | `odds_PSD` | float64 | 66.0% | 34.0% | 519 | median=3.67 | 2.2 | 19 |
| 38 | `odds_PSA` | float64 | 66.0% | 34.0% | 1178 | median=3.5 | 1.16 | 41 |
| 39 | `odds_PS_close_H` | float64 | 64.8% | 35.2% | 734 | median=2.28 | 1.06 | 22 |
| 40 | `odds_PS_close_D` | float64 | 64.8% | 35.2% | 527 | median=3.63 | 2.2 | 17 |
| 41 | `odds_PS_close_A` | float64 | 64.8% | 35.2% | 1154 | median=3.5 | 1.14 | 50 |
| 42 | `odds_B365_over25` | float64 | 32.0% | 68.0% | 66 | median=1.88 | 1.25 | 3 |
| 43 | `odds_B365_under25` | float64 | 32.0% | 68.0% | 71 | median=2 | 1.4 | 4 |
| 44 | `odds_B365_close_H` | float64 | 32.0% | 68.0% | 108 | median=2.3 | 1.07 | 17 |
| 45 | `odds_B365_close_D` | float64 | 32.0% | 68.0% | 42 | median=3.6 | 2.63 | 12 |
| 46 | `odds_B365_close_A` | float64 | 32.0% | 68.0% | 105 | median=3.25 | 1.14 | 26 |
| 47 | `odds_B365_close_over25` | float64 | 32.0% | 68.0% | 76 | median=1.9 | 1.16 | 3.2 |
| 48 | `odds_B365_close_under25` | float64 | 32.0% | 68.0% | 73 | median=1.98 | 1.36 | 5 |
| 49 | `odds_B365_AH_H` | float64 | 32.0% | 68.0% | 47 | median=1.95 | 1.7 | 3.55 |
| 50 | `odds_B365_AH_A` | float64 | 32.0% | 68.0% | 47 | median=1.95 | 1.27 | 2.17 |
| 51 | `odds_AH_line` | float64 | 28.4% | 71.6% | 19 | median=-0.25 | -2.5 | 2.25 |
| 52 | `odds_AH_close_line` | float64 | 28.6% | 71.4% | 21 | median=-0.25 | -2.75 | 2.5 |
| 53 | `matchweek` | float64 | 99.9% | 0.1% | 38 | median=19 | 1 | 38 |
| 54 | `home_formation` | object | 100.0% | 0.0% | 5 | '4-2-3-1' | — | — |
| 55 | `away_formation` | object | 100.0% | 0.0% | 5 | '4-2-3-1' | — | — |
| 56 | `home_manager` | object | 44.5% | 55.5% | 81 | 'Roberto Donadoni' | — | — |
| 57 | `away_manager` | object | 44.5% | 55.5% | 81 | 'Roberto Donadoni' | — | — |
| 58 | `home_roll_3_goals_scored` | float64 | 97.4% | 2.6% | 21 | median=1.33 | 0 | 6 |
| 59 | `home_roll_3_goals_conceded` | float64 | 97.4% | 2.6% | 20 | median=1.33 | 0 | 5 |
| 60 | `home_roll_3_shots_on_target` | float64 | 97.4% | 2.6% | 49 | median=4.33 | 0 | 15 |
| 61 | `home_roll_3_corners` | float64 | 97.4% | 2.6% | 46 | median=4.67 | 0 | 12.7 |
| 62 | `home_roll_3_fouls` | float64 | 97.4% | 2.6% | 99 | median=14.7 | 4 | 33.3 |
| 63 | `home_roll_3_yellow_cards` | float64 | 97.4% | 2.6% | 25 | median=2.33 | 0 | 7 |
| 64 | `home_roll_3_red_cards` | float64 | 97.4% | 2.6% | 8 | median=0 | 0 | 1.67 |
| 65 | `home_roll_3_points` | float64 | 97.4% | 2.6% | 11 | median=1.33 | 0 | 3 |
| 66 | `home_roll_3_clean_sheet` | float64 | 97.4% | 2.6% | 5 | median=0.333 | 0 | 1 |
| 67 | `home_roll_3_win_rate` | float64 | 97.4% | 2.6% | 5 | median=0.333 | 0 | 1 |
| 68 | `home_roll_5_goals_scored` | float64 | 97.4% | 2.6% | 41 | median=1.2 | 0 | 6 |
| 69 | `home_roll_5_goals_conceded` | float64 | 97.4% | 2.6% | 42 | median=1.33 | 0 | 5 |
| 70 | `home_roll_5_shots_on_target` | float64 | 97.4% | 2.6% | 94 | median=4.4 | 0 | 15 |
| 71 | `home_roll_5_corners` | float64 | 97.4% | 2.6% | 92 | median=5 | 0 | 12 |
| 72 | `home_roll_5_fouls` | float64 | 97.4% | 2.6% | 190 | median=14.8 | 4 | 32 |
| 73 | `home_roll_5_yellow_cards` | float64 | 97.4% | 2.6% | 49 | median=2.2 | 0 | 7 |
| 74 | `home_roll_5_red_cards` | float64 | 97.4% | 2.6% | 13 | median=0 | 0 | 1.5 |
| 75 | `home_roll_5_points` | float64 | 97.4% | 2.6% | 28 | median=1.33 | 0 | 3 |
| 76 | `home_roll_5_clean_sheet` | float64 | 97.4% | 2.6% | 11 | median=0.2 | 0 | 1 |
| 77 | `home_roll_5_win_rate` | float64 | 97.4% | 2.6% | 11 | median=0.4 | 0 | 1 |
| 78 | `home_roll_10_goals_scored` | float64 | 97.4% | 2.6% | 102 | median=1.22 | 0 | 6 |
| 79 | `home_roll_10_goals_conceded` | float64 | 97.4% | 2.6% | 97 | median=1.3 | 0 | 5 |
| 80 | `home_roll_10_shots_on_target` | float64 | 97.4% | 2.6% | 227 | median=4.33 | 0 | 15 |
| 81 | `home_roll_10_corners` | float64 | 97.4% | 2.6% | 211 | median=5 | 0 | 12 |
| 82 | `home_roll_10_fouls` | float64 | 97.4% | 2.6% | 458 | median=14.7 | 4 | 32 |
| 83 | `home_roll_10_yellow_cards` | float64 | 97.4% | 2.6% | 113 | median=2.2 | 0 | 7 |
| 84 | `home_roll_10_red_cards` | float64 | 97.4% | 2.6% | 26 | median=0.1 | 0 | 1.5 |
| 85 | `home_roll_10_points` | float64 | 97.4% | 2.6% | 89 | median=1.3 | 0 | 3 |
| 86 | `home_roll_10_clean_sheet` | float64 | 97.4% | 2.6% | 32 | median=0.286 | 0 | 1 |
| 87 | `home_roll_10_win_rate` | float64 | 97.4% | 2.6% | 33 | median=0.3 | 0 | 1 |
| 88 | `home_roll_5_goals_scored_std` | float64 | 92.1% | 7.9% | 310 | median=1 | 0 | 3.79 |
| 89 | `home_roll_5_goals_conceded_std` | float64 | 92.1% | 7.9% | 311 | median=1 | 0 | 3.37 |
| 90 | `home_roll_5_shots_on_target_std` | float64 | 92.1% | 7.9% | 989 | median=2.07 | 0 | 6.06 |
| 91 | `home_roll_5_points_std` | float64 | 92.1% | 7.9% | 135 | median=1.3 | 0 | 1.73 |
| 92 | `home_roll_10_goals_scored_std` | float64 | 92.1% | 7.9% | 914 | median=1.03 | 0 | 3.79 |
| 93 | `home_roll_10_goals_conceded_std` | float64 | 92.1% | 7.9% | 904 | median=1.06 | 0 | 3.37 |
| 94 | `home_roll_10_shots_on_target_std` | float64 | 92.1% | 7.9% | 2404 | median=2.16 | 0 | 5.57 |
| 95 | `home_roll_10_points_std` | float64 | 92.1% | 7.9% | 396 | median=1.25 | 0 | 1.73 |
| 96 | `home_goals_scored_trend` | float64 | 97.4% | 2.6% | 366 | median=0 | -1.87 | 2.17 |
| 97 | `home_goals_conceded_trend` | float64 | 97.4% | 2.6% | 356 | median=0 | -1.67 | 2.2 |
| 98 | `home_shots_on_target_trend` | float64 | 97.4% | 2.6% | 691 | median=0 | -4.88 | 4.43 |
| 99 | `home_points_trend` | float64 | 97.4% | 2.6% | 335 | median=0 | -1.9 | 1.9 |
| 100 | `home_venue_roll_3_goals_scored` | float64 | 99.4% | 0.6% | 19 | median=1.33 | 0 | 5 |
| 101 | `home_venue_roll_3_goals_conceded` | float64 | 99.4% | 0.6% | 18 | median=1 | 0 | 5.5 |
| 102 | `home_venue_roll_3_points` | float64 | 99.4% | 0.6% | 11 | median=1.67 | 0 | 3 |
| 103 | `home_venue_roll_3_clean_sheet` | float64 | 99.4% | 0.6% | 5 | median=0.333 | 0 | 1 |
| 104 | `home_venue_roll_5_goals_scored` | float64 | 99.4% | 0.6% | 37 | median=1.4 | 0 | 4.6 |
| 105 | `home_venue_roll_5_goals_conceded` | float64 | 99.4% | 0.6% | 36 | median=1.2 | 0 | 5.5 |
| 106 | `home_venue_roll_5_points` | float64 | 99.4% | 0.6% | 28 | median=1.6 | 0 | 3 |
| 107 | `home_venue_roll_5_clean_sheet` | float64 | 99.4% | 0.6% | 11 | median=0.4 | 0 | 1 |
| 108 | `home_venue_roll_10_goals_scored` | float64 | 99.4% | 0.6% | 88 | median=1.4 | 0 | 3.4 |
| 109 | `home_venue_roll_10_goals_conceded` | float64 | 99.4% | 0.6% | 86 | median=1.1 | 0 | 5.5 |
| 110 | `home_venue_roll_10_points` | float64 | 99.4% | 0.6% | 78 | median=1.6 | 0 | 3 |
| 111 | `home_venue_roll_10_clean_sheet` | float64 | 99.4% | 0.6% | 30 | median=0.3 | 0 | 1 |
| 112 | `home_attack_strength` | float64 | 100.0% | 0.0% | 7127 | median=1 | 0 | 4.72 |
| 113 | `home_defense_strength` | float64 | 100.0% | 0.0% | 7138 | median=1 | 0 | 4.3 |
| 114 | `home_xg_attack_strength` | float64 | 100.0% | 0.0% | 58 | median=1 | 0.168 | 2.64 |
| 115 | `home_xg_defense_strength` | float64 | 100.0% | 0.0% | 58 | median=1 | 0.271 | 2.91 |
| 116 | `home_rest_days` | float64 | 97.4% | 2.6% | 19 | median=7 | 3 | 21 |
| 117 | `home_is_congested` | bool | 100.0% | 0.0% | 2 | np.False_ | — | — |
| 118 | `home_win_streak` | float64 | 99.8% | 0.2% | 17 | median=0 | 0 | 17 |
| 119 | `home_unbeaten_run` | float64 | 99.8% | 0.2% | 41 | median=1 | 0 | 49 |
| 120 | `home_winless_run` | float64 | 99.8% | 0.2% | 26 | median=1 | 0 | 26 |
| 121 | `home_loss_streak` | float64 | 99.8% | 0.2% | 13 | median=0 | 0 | 14 |
| 122 | `home_scoring_streak` | float64 | 99.8% | 0.2% | 44 | median=2 | 0 | 44 |
| 123 | `home_clean_sheet_streak` | float64 | 99.8% | 0.2% | 11 | median=0 | 0 | 10 |
| 124 | `home_form_points_5` | float64 | 99.8% | 0.2% | 15 | median=7 | 0 | 15 |
| 125 | `home_adj_attack_5` | float64 | 97.4% | 2.6% | 7153 | median=1.27 | 0 | 22 |
| 126 | `home_adj_defense_5` | float64 | 97.4% | 2.6% | 7165 | median=1.3 | 0 | 16.1 |
| 127 | `home_adj_attack_10` | float64 | 97.4% | 2.6% | 7176 | median=1.29 | 0 | 20.9 |
| 128 | `home_adj_defense_10` | float64 | 97.4% | 2.6% | 7194 | median=1.3 | 0 | 15.2 |
| 129 | `home_opp_difficulty_roll_5` | float64 | 99.8% | 0.2% | 7879 | median=1.01 | 0.426 | 2.3 |
| 130 | `home_form_overperformance` | float64 | 99.8% | 0.2% | 7471 | median=5.88 | -1.54 | 15 |
| 131 | `home_gd_roll_3` | float64 | 99.8% | 0.2% | 32 | median=0 | -4 | 4.33 |
| 132 | `home_gd_roll_5` | float64 | 99.8% | 0.2% | 50 | median=0 | -3 | 3.2 |
| 133 | `home_gd_per_match` | float64 | 97.4% | 2.6% | 1230 | median=-0.0741 | -5 | 5 |
| 134 | `home_momentum_gradient` | float64 | 98.6% | 1.4% | 119 | median=-8.11e-17 | -0.9 | 0.9 |
| 135 | `home_ewma_form` | float64 | 99.2% | 0.8% | 7844 | median=1.28 | 0 | 3 |
| 136 | `home_last3_vs_prev3` | float64 | 98.6% | 1.4% | 19 | median=0 | -9 | 9 |
| 137 | `away_roll_3_goals_scored` | float64 | 97.3% | 2.7% | 22 | median=1.33 | 0 | 5.33 |
| 138 | `away_roll_3_goals_conceded` | float64 | 97.3% | 2.7% | 21 | median=1.33 | 0 | 5 |
| 139 | `away_roll_3_shots_on_target` | float64 | 97.3% | 2.7% | 48 | median=4.33 | 0 | 13 |
| 140 | `away_roll_3_corners` | float64 | 97.3% | 2.7% | 49 | median=5 | 0 | 12.7 |
| 141 | `away_roll_3_fouls` | float64 | 97.3% | 2.7% | 102 | median=14.7 | 5 | 36 |
| 142 | `away_roll_3_yellow_cards` | float64 | 97.3% | 2.7% | 25 | median=2.33 | 0 | 7 |
| 143 | `away_roll_3_red_cards` | float64 | 97.3% | 2.7% | 8 | median=0 | 0 | 2 |
| 144 | `away_roll_3_points` | float64 | 97.3% | 2.7% | 11 | median=1.33 | 0 | 3 |
| 145 | `away_roll_3_clean_sheet` | float64 | 97.3% | 2.7% | 5 | median=0.333 | 0 | 1 |
| 146 | `away_roll_3_win_rate` | float64 | 97.3% | 2.7% | 5 | median=0.333 | 0 | 1 |
| 147 | `away_roll_5_goals_scored` | float64 | 97.3% | 2.7% | 44 | median=1.2 | 0 | 5 |
| 148 | `away_roll_5_goals_conceded` | float64 | 97.3% | 2.7% | 40 | median=1.2 | 0 | 5 |
| 149 | `away_roll_5_shots_on_target` | float64 | 97.3% | 2.7% | 95 | median=4.4 | 0 | 13 |
| 150 | `away_roll_5_corners` | float64 | 97.3% | 2.7% | 96 | median=5 | 0 | 12 |
| 151 | `away_roll_5_fouls` | float64 | 97.3% | 2.7% | 201 | median=14.8 | 5 | 36 |
| 152 | `away_roll_5_yellow_cards` | float64 | 97.3% | 2.7% | 49 | median=2.2 | 0 | 7 |
| 153 | `away_roll_5_red_cards` | float64 | 97.3% | 2.7% | 12 | median=0 | 0 | 2 |
| 154 | `away_roll_5_points` | float64 | 97.3% | 2.7% | 28 | median=1.4 | 0 | 3 |
| 155 | `away_roll_5_clean_sheet` | float64 | 97.3% | 2.7% | 11 | median=0.2 | 0 | 1 |
| 156 | `away_roll_5_win_rate` | float64 | 97.3% | 2.7% | 11 | median=0.4 | 0 | 1 |
| 157 | `away_roll_10_goals_scored` | float64 | 97.3% | 2.7% | 108 | median=1.3 | 0 | 5 |
| 158 | `away_roll_10_goals_conceded` | float64 | 97.3% | 2.7% | 95 | median=1.3 | 0 | 5 |
| 159 | `away_roll_10_shots_on_target` | float64 | 97.3% | 2.7% | 224 | median=4.4 | 0 | 13 |
| 160 | `away_roll_10_corners` | float64 | 97.3% | 2.7% | 211 | median=5 | 0 | 12 |
| 161 | `away_roll_10_fouls` | float64 | 97.3% | 2.7% | 467 | median=14.7 | 5 | 36 |
| 162 | `away_roll_10_yellow_cards` | float64 | 97.3% | 2.7% | 110 | median=2.2 | 0 | 7 |
| 163 | `away_roll_10_red_cards` | float64 | 97.3% | 2.7% | 26 | median=0.1 | 0 | 2 |
| 164 | `away_roll_10_points` | float64 | 97.3% | 2.7% | 91 | median=1.3 | 0 | 3 |
| 165 | `away_roll_10_clean_sheet` | float64 | 97.3% | 2.7% | 32 | median=0.3 | 0 | 1 |
| 166 | `away_roll_10_win_rate` | float64 | 97.3% | 2.7% | 33 | median=0.333 | 0 | 1 |
| 167 | `away_roll_5_goals_scored_std` | float64 | 92.0% | 8.0% | 308 | median=1 | 0 | 3.21 |
| 168 | `away_roll_5_goals_conceded_std` | float64 | 92.0% | 8.0% | 299 | median=1 | 0 | 3.79 |
| 169 | `away_roll_5_shots_on_target_std` | float64 | 92.0% | 8.0% | 997 | median=2.07 | 0 | 6.66 |
| 170 | `away_roll_5_points_std` | float64 | 92.0% | 8.0% | 130 | median=1.3 | 0 | 1.73 |
| 171 | `away_roll_10_goals_scored_std` | float64 | 92.0% | 8.0% | 906 | median=1.05 | 0 | 3.21 |
| 172 | `away_roll_10_goals_conceded_std` | float64 | 92.0% | 8.0% | 888 | median=1.06 | 0 | 3.79 |
| 173 | `away_roll_10_shots_on_target_std` | float64 | 92.0% | 8.0% | 2405 | median=2.17 | 0 | 6.66 |
| 174 | `away_roll_10_points_std` | float64 | 92.0% | 8.0% | 386 | median=1.25 | 0 | 1.73 |
| 175 | `away_goals_scored_trend` | float64 | 97.3% | 2.7% | 367 | median=0 | -2.07 | 2.73 |
| 176 | `away_goals_conceded_trend` | float64 | 97.3% | 2.7% | 356 | median=0 | -2.07 | 2.5 |
| 177 | `away_shots_on_target_trend` | float64 | 97.3% | 2.7% | 706 | median=0 | -3.9 | 4.83 |
| 178 | `away_points_trend` | float64 | 97.3% | 2.7% | 340 | median=0 | -1.87 | 1.9 |
| 179 | `away_venue_roll_3_goals_scored` | float64 | 99.4% | 0.6% | 20 | median=1 | 0 | 5.33 |
| 180 | `away_venue_roll_3_goals_conceded` | float64 | 99.4% | 0.6% | 20 | median=1.33 | 0 | 5 |
| 181 | `away_venue_roll_3_points` | float64 | 99.4% | 0.6% | 11 | median=1 | 0 | 3 |
| 182 | `away_venue_roll_3_clean_sheet` | float64 | 99.4% | 0.6% | 5 | median=0.333 | 0 | 1 |
| 183 | `away_venue_roll_5_goals_scored` | float64 | 99.4% | 0.6% | 34 | median=1 | 0 | 5 |
| 184 | `away_venue_roll_5_goals_conceded` | float64 | 99.4% | 0.6% | 41 | median=1.4 | 0 | 5 |
| 185 | `away_venue_roll_5_points` | float64 | 99.4% | 0.6% | 25 | median=1 | 0 | 3 |
| 186 | `away_venue_roll_5_clean_sheet` | float64 | 99.4% | 0.6% | 11 | median=0.2 | 0 | 1 |
| 187 | `away_venue_roll_10_goals_scored` | float64 | 99.4% | 0.6% | 76 | median=1.1 | 0 | 5 |
| 188 | `away_venue_roll_10_goals_conceded` | float64 | 99.4% | 0.6% | 85 | median=1.5 | 0 | 5 |
| 189 | `away_venue_roll_10_points` | float64 | 99.4% | 0.6% | 73 | median=1.1 | 0 | 3 |
| 190 | `away_venue_roll_10_clean_sheet` | float64 | 99.4% | 0.6% | 24 | median=0.2 | 0 | 1 |
| 191 | `away_attack_strength` | float64 | 100.0% | 0.0% | 7159 | median=1 | 0 | 4.08 |
| 192 | `away_defense_strength` | float64 | 100.0% | 0.0% | 7119 | median=1 | 0 | 3.93 |
| 193 | `away_xg_attack_strength` | float64 | 100.0% | 0.0% | 56 | median=1 | 0.258 | 2.47 |
| 194 | `away_xg_defense_strength` | float64 | 100.0% | 0.0% | 56 | median=1 | 0.193 | 3.27 |
| 195 | `away_rest_days` | float64 | 97.3% | 2.7% | 19 | median=7 | 3 | 21 |
| 196 | `away_is_congested` | bool | 100.0% | 0.0% | 2 | np.False_ | — | — |
| 197 | `away_win_streak` | float64 | 99.7% | 0.3% | 17 | median=0 | 0 | 16 |
| 198 | `away_unbeaten_run` | float64 | 99.7% | 0.3% | 39 | median=1 | 0 | 47 |
| 199 | `away_winless_run` | float64 | 99.7% | 0.3% | 25 | median=1 | 0 | 25 |
| 200 | `away_loss_streak` | float64 | 99.7% | 0.3% | 13 | median=0 | 0 | 13 |
| 201 | `away_scoring_streak` | float64 | 99.7% | 0.3% | 44 | median=2 | 0 | 43 |
| 202 | `away_clean_sheet_streak` | float64 | 99.7% | 0.3% | 11 | median=0 | 0 | 10 |
| 203 | `away_form_points_5` | float64 | 99.7% | 0.3% | 15 | median=7 | 0 | 15 |
| 204 | `away_adj_attack_5` | float64 | 97.3% | 2.7% | 7166 | median=1.28 | 0 | 20.1 |
| 205 | `away_adj_defense_5` | float64 | 97.3% | 2.7% | 7129 | median=1.26 | 0 | 16.2 |
| 206 | `away_adj_attack_10` | float64 | 97.3% | 2.7% | 7205 | median=1.3 | 0 | 23.5 |
| 207 | `away_adj_defense_10` | float64 | 97.3% | 2.7% | 7177 | median=1.3 | 0 | 16.8 |
| 208 | `away_opp_difficulty_roll_5` | float64 | 99.7% | 0.3% | 7877 | median=1.01 | 0.285 | 2 |
| 209 | `away_form_overperformance` | float64 | 99.7% | 0.3% | 7456 | median=6.19 | -1.46 | 15 |
| 210 | `away_gd_roll_3` | float64 | 99.7% | 0.3% | 30 | median=0 | -4.33 | 4 |
| 211 | `away_gd_roll_5` | float64 | 99.7% | 0.3% | 53 | median=0 | -3.5 | 3.6 |
| 212 | `away_gd_per_match` | float64 | 97.3% | 2.7% | 1236 | median=-0.0286 | -4 | 4 |
| 213 | `away_momentum_gradient` | float64 | 98.6% | 1.4% | 119 | median=0 | -0.9 | 0.9 |
| 214 | `away_ewma_form` | float64 | 99.1% | 0.9% | 7834 | median=1.37 | 0 | 3 |
| 215 | `away_last3_vs_prev3` | float64 | 98.6% | 1.4% | 19 | median=0 | -9 | 9 |
| 216 | `h2h_matches_played` | int64 | 100.0% | 0.0% | 8 | median=7 | 0 | 7 |
| 217 | `h2h_home_wins` | float64 | 91.0% | 9.0% | 8 | median=2 | 0 | 7 |
| 218 | `h2h_away_wins` | float64 | 91.0% | 9.0% | 8 | median=2 | 0 | 7 |
| 219 | `h2h_draws` | float64 | 91.0% | 9.0% | 7 | median=1 | 0 | 6 |
| 220 | `h2h_draw_rate` | float64 | 91.0% | 9.0% | 19 | median=0.286 | 0 | 1 |
| 221 | `h2h_goals_avg` | float64 | 91.0% | 9.0% | 90 | median=2.67 | 0 | 8 |
| 222 | `h2h_home_goals_avg` | float64 | 91.0% | 9.0% | 72 | median=1.29 | 0 | 7 |
| 223 | `h2h_away_goals_avg` | float64 | 91.0% | 9.0% | 77 | median=1.29 | 0 | 7 |
| 224 | `h2h_home_win_rate` | float64 | 91.0% | 9.0% | 19 | median=0.286 | 0 | 1 |
| 225 | `h2h_last_result` | float64 | 91.0% | 9.0% | 3 | median=0 | -1 | 1 |
| 226 | `h2h_weighted_home_win_rate` | float64 | 91.0% | 9.0% | 3661 | median=0.313 | 0 | 1 |
| 227 | `h2h_recent_5_win_rate` | float64 | 91.0% | 9.0% | 11 | median=0.4 | 0 | 1 |
| 228 | `h2h_recent_3_total_goals` | float64 | 91.0% | 9.0% | 31 | median=2.67 | 0 | 8 |
| 229 | `home_elo` | float64 | 100.0% | 0.0% | 7886 | median=1.48e+03 | 1.25e+03 | 1.81e+03 |
| 230 | `away_elo` | float64 | 100.0% | 0.0% | 7879 | median=1.48e+03 | 1.26e+03 | 1.81e+03 |
| 231 | `elo_diff` | float64 | 100.0% | 0.0% | 7913 | median=-0.165 | -495 | 457 |
| 232 | `home_key_players_available` | float64 | 7.8% | 92.2% | 6 | median=3 | 0 | 5 |
| 233 | `away_key_players_available` | float64 | 7.8% | 92.2% | 6 | median=3 | 0 | 5 |
| 234 | `home_top_scorer_played` | float64 | 7.8% | 92.2% | 2 | median=0 | 0 | 1 |
| 235 | `away_top_scorer_played` | float64 | 7.8% | 92.2% | 2 | median=0 | 0 | 1 |
| 236 | `home_squad_rotation` | float64 | 7.8% | 92.2% | 9 | median=1 | 0 | 8 |
| 237 | `away_squad_rotation` | float64 | 7.8% | 92.2% | 11 | median=2 | 0 | 10 |
| 238 | `ref_avg_yellows` | float64 | 50.5% | 49.5% | 316 | median=4.45 | 0 | 8 |
| 239 | `ref_avg_reds` | float64 | 50.5% | 49.5% | 432 | median=0.211 | 0 | 1.5 |
| 240 | `ref_avg_fouls` | float64 | 50.5% | 49.5% | 271 | median=26.6 | 0 | 67 |
| 241 | `ref_matches_officiated` | float64 | 52.1% | 47.9% | 153 | median=27 | 0 | 152 |
| 242 | `ref_strictness_score` | float64 | 50.5% | 49.5% | 635 | median=0.319 | 0 | 1 |
| 243 | `ref_home_bias` | float64 | 50.5% | 49.5% | 552 | median=-0.011 | -0.469 | 0.545 |
| 244 | `ref_home_cards_bias` | float64 | 50.5% | 49.5% | 487 | median=0.064 | -1 | 1 |
| 245 | `ref_avg_total_goals` | float64 | 50.5% | 49.5% | 211 | median=2.74 | 0 | 8 |
| 246 | `ref_strictness_trend` | float64 | 50.5% | 49.5% | 1376 | median=0 | -0.137 | 0.161 |
| 247 | `ref_big_match_cards` | float64 | 50.5% | 49.5% | 740 | median=1 | 0.393 | 1.72 |
| 248 | `ref_home_team_cards` | float64 | 50.5% | 49.5% | 203 | median=2.07 | 0 | 6 |
| 249 | `ref_away_team_cards` | float64 | 50.5% | 49.5% | 209 | median=2.36 | 0 | 7 |
| 250 | `ref_vs_home_team_bias` | float64 | 50.5% | 49.5% | 1407 | median=0 | -2.69 | 2.76 |
| 251 | `ref_vs_away_team_bias` | float64 | 50.5% | 49.5% | 1420 | median=0 | -2.15 | 2.61 |
| 252 | `ref_last_match_reds` | float64 | 50.5% | 49.5% | 5 | median=0 | 0 | 4 |
| 253 | `ref_last_match_cards` | float64 | 50.5% | 49.5% | 18 | median=5 | 0 | 17 |
| 254 | `ref_regression_signal` | float64 | 50.5% | 49.5% | 957 | median=-0.05 | -6.16 | 12.1 |
| 255 | `home_us_team_xg` | Float64 | 44.3% | 55.7% | 187 | median=52.7 | 29.6 | 94.5 |
| 256 | `home_us_team_npxg` | Float64 | 44.3% | 55.7% | 187 | median=47.7 | 26.6 | 86.1 |
| 257 | `home_us_team_xa` | Float64 | 44.3% | 55.7% | 187 | median=37.1 | 20.2 | 63.9 |
| 258 | `home_us_team_shots` | Int64 | 44.3% | 55.7% | 145 | median=504 | 327 | 780 |
| 259 | `home_us_team_key_passes` | Int64 | 44.3% | 55.7% | 141 | median=378 | 235 | 569 |
| 260 | `home_us_team_xg_chain` | Float64 | 44.3% | 55.7% | 187 | median=148 | 63.1 | 298 |
| 261 | `home_us_team_xg_buildup` | Float64 | 44.3% | 55.7% | 187 | median=86.7 | 33.8 | 204 |
| 262 | `home_us_team_minutes` | Int64 | 44.3% | 55.7% | 183 | median=3.76e+04 | 3.13e+04 | 4.27e+04 |
| 263 | `home_us_team_goals` | Int64 | 44.3% | 55.7% | 60 | median=52 | 23 | 105 |
| 264 | `home_us_team_assists` | Int64 | 44.3% | 55.7% | 50 | median=36 | 14 | 71 |
| 265 | `home_us_player_count` | float64 | 44.3% | 55.7% | 18 | median=28 | 22 | 42 |
| 266 | `home_us_team_xg_per_90` | Float64 | 44.3% | 55.7% | 187 | median=0.127 | 0.0826 | 0.222 |
| 267 | `home_us_team_xa_per_90` | Float64 | 44.3% | 55.7% | 187 | median=0.0889 | 0.0507 | 0.15 |
| 268 | `home_us_team_xg_per_shot` | Float64 | 44.3% | 55.7% | 187 | median=0.107 | 0.0797 | 0.149 |
| 269 | `home_us_team_goals_minus_xg` | Float64 | 44.3% | 55.7% | 187 | median=-2.1 | -17 | 22.2 |
| 270 | `home_us_top3_xg_share` | float64 | 44.3% | 55.7% | 187 | median=0.501 | 0.282 | 0.671 |
| 271 | `away_us_team_xg` | Float64 | 44.3% | 55.7% | 187 | median=52.7 | 29.6 | 94.5 |
| 272 | `away_us_team_npxg` | Float64 | 44.3% | 55.7% | 187 | median=47.7 | 26.6 | 86.1 |
| 273 | `away_us_team_xa` | Float64 | 44.3% | 55.7% | 187 | median=37.1 | 20.2 | 63.9 |
| 274 | `away_us_team_shots` | Int64 | 44.3% | 55.7% | 145 | median=504 | 327 | 780 |
| 275 | `away_us_team_key_passes` | Int64 | 44.3% | 55.7% | 141 | median=378 | 235 | 569 |
| 276 | `away_us_team_xg_chain` | Float64 | 44.3% | 55.7% | 187 | median=148 | 63.1 | 298 |
| 277 | `away_us_team_xg_buildup` | Float64 | 44.3% | 55.7% | 187 | median=86.7 | 33.8 | 204 |
| 278 | `away_us_team_minutes` | Int64 | 44.3% | 55.7% | 183 | median=3.76e+04 | 3.13e+04 | 4.27e+04 |
| 279 | `away_us_team_goals` | Int64 | 44.3% | 55.7% | 60 | median=52 | 23 | 105 |
| 280 | `away_us_team_assists` | Int64 | 44.3% | 55.7% | 50 | median=36 | 14 | 71 |
| 281 | `away_us_player_count` | float64 | 44.3% | 55.7% | 18 | median=28 | 22 | 42 |
| 282 | `away_us_team_xg_per_90` | Float64 | 44.3% | 55.7% | 187 | median=0.127 | 0.0826 | 0.222 |
| 283 | `away_us_team_xa_per_90` | Float64 | 44.3% | 55.7% | 187 | median=0.0889 | 0.0507 | 0.15 |
| 284 | `away_us_team_xg_per_shot` | Float64 | 44.3% | 55.7% | 187 | median=0.107 | 0.0797 | 0.149 |
| 285 | `away_us_team_goals_minus_xg` | Float64 | 44.3% | 55.7% | 187 | median=-2.1 | -17 | 22.2 |
| 286 | `away_us_top3_xg_share` | float64 | 44.3% | 55.7% | 187 | median=0.501 | 0.282 | 0.671 |
| 287 | `us_xg_diff` | Float64 | 100.0% | 0.0% | 3331 | median=0 | -94.5 | 94.5 |
| 288 | `us_xa_diff` | Float64 | 100.0% | 0.0% | 3331 | median=0 | -63.9 | 63.9 |
| 289 | `us_coverage` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 290 | `home_us_xg_rolling_avg` | float64 | 40.9% | 59.1% | 173 | median=50.3 | 34.6 | 82.4 |
| 291 | `home_us_xg_rolling_std` | float64 | 40.9% | 59.1% | 173 | median=5.6 | 0.164 | 20.4 |
| 292 | `home_us_xg_trend` | float64 | 40.9% | 59.1% | 173 | median=-0.131 | -33.2 | 31.2 |
| 293 | `away_us_xg_rolling_avg` | float64 | 40.9% | 59.1% | 173 | median=50.3 | 34.6 | 82.4 |
| 294 | `away_us_xg_rolling_std` | float64 | 40.9% | 59.1% | 173 | median=5.6 | 0.164 | 20.4 |
| 295 | `away_us_xg_trend` | float64 | 40.9% | 59.1% | 173 | median=-0.131 | -33.2 | 31.2 |
| 296 | `has_xg_data` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 297 | `home_ss_roll_xg` | float64 | 18.2% | 81.8% | 1434 | median=1.21 | 0.323 | 2.77 |
| 298 | `home_ss_roll_xgot` | float64 | 18.2% | 81.8% | 1436 | median=1.13 | 0.118 | 3.26 |
| 299 | `home_ss_roll_xa` | float64 | 18.2% | 81.8% | 1442 | median=0.741 | 0.253 | 2.27 |
| 300 | `home_ss_roll_goals` | float64 | 18.2% | 81.8% | 29 | median=1.2 | 0 | 3.6 |
| 301 | `home_ss_roll_total_shots` | float64 | 18.2% | 81.8% | 96 | median=12.2 | 5.33 | 23.5 |
| 302 | `home_ss_roll_shots_on_target` | float64 | 18.2% | 81.8% | 52 | median=4 | 0.8 | 9.4 |
| 303 | `home_ss_roll_big_chances_created` | float64 | 18.2% | 81.8% | 35 | median=1.2 | 0 | 4.8 |
| 304 | `home_ss_roll_key_passes` | float64 | 18.2% | 81.8% | 87 | median=9.2 | 3.5 | 18 |
| 305 | `home_ss_roll_pass_accuracy` | float64 | 18.2% | 81.8% | 1442 | median=0.816 | 0.653 | 0.915 |
| 306 | `home_ss_roll_xg_per_shot` | float64 | 18.2% | 81.8% | 1442 | median=0.1 | 0.0466 | 0.224 |
| 307 | `home_ss_roll_duel_win_rate` | float64 | 18.2% | 81.8% | 1440 | median=0.5 | 0.401 | 0.597 |
| 308 | `home_ss_roll_aerial_win_rate` | float64 | 18.2% | 81.8% | 1438 | median=0.498 | 0.278 | 0.708 |
| 309 | `home_ss_roll_tackles_won` | float64 | 18.2% | 81.8% | 71 | median=9 | 4.2 | 15.6 |
| 310 | `home_ss_roll_interceptions` | float64 | 18.2% | 81.8% | 66 | median=7.6 | 3.2 | 13.6 |
| 311 | `home_ss_roll_ball_recoveries` | float64 | 18.2% | 81.8% | 206 | median=48.8 | 34.6 | 76.2 |
| 312 | `home_ss_roll_blocks` | float64 | 18.2% | 81.8% | 56 | median=3.2 | 0 | 9 |
| 313 | `home_ss_roll_clearances` | float64 | 18.2% | 81.8% | 160 | median=19 | 6 | 39.6 |
| 314 | `home_ss_roll_fouls` | float64 | 18.2% | 81.8% | 79 | median=12.4 | 7 | 20 |
| 315 | `home_ss_roll_was_fouled` | float64 | 18.2% | 81.8% | 86 | median=11.6 | 6.2 | 18.6 |
| 316 | `home_ss_roll_progressive_carries` | float64 | 18.2% | 81.8% | 103 | median=0 | 0 | 26.4 |
| 317 | `home_ss_roll_rating` | float64 | 18.2% | 81.8% | 1345 | median=6.84 | 6.52 | 7.24 |
| 318 | `home_ss_roll_carry_distance_per_carry` | float64 | 18.2% | 81.8% | 272 | median=0 | 0 | 9.77 |
| 319 | `home_ss_roll_shot_blocked_rate` | float64 | 18.2% | 81.8% | 1406 | median=0.263 | 0.0364 | 0.482 |
| 320 | `home_ss_roll_last_man_tackle` | float64 | 18.2% | 81.8% | 10 | median=0 | 0 | 1.2 |
| 321 | `home_ss_roll_cross_accuracy` | float64 | 18.2% | 81.8% | 1432 | median=0.254 | 0.0654 | 0.506 |
| 322 | `home_ss_roll_error_to_goal` | float64 | 18.2% | 81.8% | 8 | median=0 | 0 | 0.8 |
| 323 | `home_ss_roll_error_to_shot` | float64 | 18.2% | 81.8% | 15 | median=0.2 | 0 | 1.8 |
| 324 | `home_ss_roll_contest_win_rate` | float64 | 18.2% | 81.8% | 1414 | median=0.48 | 0.24 | 0.719 |
| 325 | `home_ss_roll_unsuccessful_touch` | float64 | 18.2% | 81.8% | 85 | median=14.8 | 6.6 | 23 |
| 326 | `home_ss_roll_opp_half_pass_ratio` | float64 | 18.2% | 81.8% | 1442 | median=0.524 | 0.352 | 0.685 |
| 327 | `home_ss_roll_att_xg` | float64 | 18.2% | 81.8% | 1431 | median=0.622 | 0.11 | 1.94 |
| 328 | `home_ss_roll_att_xa` | float64 | 18.2% | 81.8% | 1442 | median=0.198 | 0.00536 | 1.02 |
| 329 | `home_ss_roll_att_shots` | float64 | 18.2% | 81.8% | 70 | median=5.2 | 1 | 12.6 |
| 330 | `home_ss_roll_att_key_passes` | float64 | 18.2% | 81.8% | 58 | median=2.6 | 0 | 9.8 |
| 331 | `home_ss_roll_att_rating` | float64 | 18.2% | 81.8% | 959 | median=6.82 | 6.21 | 7.67 |
| 332 | `home_ss_roll_mid_pass_accuracy` | float64 | 18.2% | 81.8% | 1442 | median=0.825 | 0.651 | 0.926 |
| 333 | `home_ss_roll_mid_duel_win_rate` | float64 | 18.2% | 81.8% | 1439 | median=0.493 | 0.371 | 0.617 |
| 334 | `home_ss_roll_mid_tackles` | float64 | 18.2% | 81.8% | 84 | median=7.4 | 2.33 | 14.8 |
| 335 | `home_ss_roll_mid_interceptions` | float64 | 18.2% | 81.8% | 46 | median=3 | 0.2 | 8 |
| 336 | `home_ss_roll_mid_rating` | float64 | 18.2% | 81.8% | 1310 | median=6.84 | 6.5 | 7.32 |
| 337 | `home_ss_roll_def_aerial_win_rate` | float64 | 18.2% | 81.8% | 1385 | median=0.573 | 0.236 | 0.865 |
| 338 | `home_ss_roll_def_clearances` | float64 | 18.2% | 81.8% | 120 | median=12 | 3.5 | 28.4 |
| 339 | `home_ss_roll_def_blocks` | float64 | 18.2% | 81.8% | 40 | median=2 | 0 | 7 |
| 340 | `home_ss_roll_def_tackles_won` | float64 | 18.2% | 81.8% | 47 | median=3.6 | 0.6 | 8 |
| 341 | `home_ss_roll_def_rating` | float64 | 18.2% | 81.8% | 1068 | median=6.84 | 6.31 | 7.5 |
| 342 | `home_ss_roll_gk_goals_prevented` | float64 | 18.2% | 81.8% | 1437 | median=-0.0537 | -1.28 | 1.13 |
| 343 | `home_ss_roll_gk_rating` | float64 | 18.2% | 81.8% | 202 | median=6.98 | 5.95 | 8.58 |
| 344 | `home_ss_roll_territory_ratio` | float64 | 18.2% | 81.8% | 1442 | median=0.462 | 0.284 | 0.648 |
| 345 | `home_ss_roll_press_intensity` | float64 | 18.2% | 81.8% | 1442 | median=0.208 | 0.124 | 0.315 |
| 346 | `home_ss_roll_dispossess_rate` | float64 | 18.2% | 81.8% | 1442 | median=0.0137 | 0.00515 | 0.0258 |
| 347 | `home_ss_roll_total_poss_lost` | float64 | 18.2% | 81.8% | 310 | median=124 | 92.8 | 173 |
| 348 | `home_ss_roll_xi_changes` | float64 | 18.0% | 82.0% | 46 | median=2.6 | 0 | 6.8 |
| 349 | `home_ss_roll_mins_concentration` | float64 | 18.2% | 81.8% | 1440 | median=0.902 | 0.848 | 0.967 |
| 350 | `home_ss_roll_counter_xg_pct` | float64 | 17.5% | 82.5% | 1134 | median=0.0611 | 0 | 0.45 |
| 351 | `home_ss_roll_set_piece_xg_pct` | float64 | 17.5% | 82.5% | 1379 | median=0.185 | 0.0109 | 0.627 |
| 352 | `home_ss_roll_open_play_xg_pct` | float64 | 17.5% | 82.5% | 1381 | median=0.733 | 0.184 | 0.989 |
| 353 | `home_ss_roll_header_shot_pct` | float64 | 17.5% | 82.5% | 1341 | median=0.181 | 0 | 0.433 |
| 354 | `home_ss_roll_first_half_xg_share` | float64 | 17.5% | 82.5% | 1382 | median=0.449 | 0.138 | 0.851 |
| 355 | `home_ss_roll_last_15_xg_share` | float64 | 17.5% | 82.5% | 1382 | median=0.222 | 0 | 0.588 |
| 356 | `home_ss_roll_corner_xg_share` | float64 | 17.5% | 82.5% | 1371 | median=0.132 | 0 | 0.627 |
| 357 | `home_ss_roll_free_kick_xg_share` | float64 | 17.5% | 82.5% | 991 | median=0.0174 | 0 | 0.236 |
| 358 | `home_ss_roll_penalty_xg_share` | float64 | 17.5% | 82.5% | 467 | median=0.0485 | 0 | 0.514 |
| 359 | `home_ss_roll_penalties_taken` | float64 | 17.5% | 82.5% | 11 | median=0.2 | 0 | 1 |
| 360 | `home_ss_roll_penalties_scored` | float64 | 17.5% | 82.5% | 10 | median=0 | 0 | 1 |
| 361 | `home_ss_roll_possession` | float64 | 18.2% | 81.8% | 193 | median=50 | 32.2 | 70.4 |
| 362 | `home_ss_roll_corners` | float64 | 18.2% | 81.8% | 55 | median=4.4 | 1.2 | 10 |
| 363 | `home_ss_roll_throw_ins` | float64 | 18.2% | 81.8% | 108 | median=19 | 11 | 30.2 |
| 364 | `home_ss_roll_shots_inside_box` | float64 | 18.2% | 81.8% | 81 | median=7.8 | 2.6 | 17.8 |
| 365 | `home_ss_roll_shots_outside_box` | float64 | 18.2% | 81.8% | 55 | median=4.4 | 1.2 | 9 |
| 366 | `home_ss_roll_shots_inside_box_pct` | float64 | 18.2% | 81.8% | 1408 | median=0.632 | 0.292 | 0.89 |
| 367 | `home_ss_roll_hit_woodwork` | float64 | 18.2% | 81.8% | 16 | median=0.2 | 0 | 1.6 |
| 368 | `home_ss_roll_big_chances_scored` | float64 | 18.2% | 81.8% | 22 | median=0.6 | 0 | 2.6 |
| 369 | `home_ss_roll_touches_in_opp_box` | float64 | 18.2% | 81.8% | 173 | median=10.8 | 0 | 45.8 |
| 370 | `home_ss_roll_fouled_final_third` | float64 | 18.2% | 81.8% | 37 | median=2 | 0.4 | 5.4 |
| 371 | `home_ss_roll_final_third_entries` | float64 | 18.2% | 81.8% | 204 | median=50.4 | 31.6 | 74.2 |
| 372 | `home_ss_roll_final_third_phases` | float64 | 18.2% | 81.8% | 441 | median=39.2 | 0 | 191 |
| 373 | `home_ss_roll_duel_won_pct` | float64 | 18.2% | 81.8% | 97 | median=50 | 40 | 59.8 |
| 374 | `home_ss_roll_ground_duels_pct` | float64 | 18.2% | 81.8% | 123 | median=33.4 | 23.2 | 46.8 |
| 375 | `home_ss_roll_aerial_duels_pct` | float64 | 18.2% | 81.8% | 117 | median=14.2 | 4.5 | 29.6 |
| 376 | `home_ss_roll_dribbles_pct` | float64 | 18.2% | 81.8% | 84 | median=6.6 | 2 | 16.4 |
| 377 | `home_ss_roll_dive_saves` | float64 | 18.2% | 81.8% | 17 | median=0.2 | 0 | 2.6 |
| 378 | `home_ss_roll_high_claims` | float64 | 18.2% | 81.8% | 23 | median=0.2 | 0 | 3.6 |
| 379 | `home_ss_roll_dispossessed` | float64 | 18.2% | 81.8% | 69 | median=8.2 | 3 | 14.6 |
| 380 | `home_ss_roll_avg_shot_xg` | float64 | 18.2% | 81.8% | 1310 | median=0.1 | 0.0475 | 0.223 |
| 381 | `home_ss_roll_max_shot_xg` | float64 | 18.2% | 81.8% | 1412 | median=0.406 | 0.115 | 0.803 |
| 382 | `home_ss_roll_total_xgot` | float64 | 18.2% | 81.8% | 1433 | median=1.13 | 0.234 | 3.26 |
| 383 | `home_ss_roll_sm_inside_box_pct` | float64 | 18.2% | 81.8% | 1405 | median=0.638 | 0.292 | 0.893 |
| 384 | `home_ss_roll_sm_header_pct` | float64 | 18.2% | 81.8% | 1397 | median=0.183 | 0.0154 | 0.448 |
| 385 | `home_ss_roll_sm_open_play_pct` | float64 | 18.2% | 81.8% | 1407 | median=0.662 | 0.363 | 0.88 |
| 386 | `home_ss_roll_sm_set_piece_pct` | float64 | 18.2% | 81.8% | 1398 | median=0.252 | 0.0775 | 0.565 |
| 387 | `home_ss_roll_sm_counter_pct` | float64 | 18.2% | 81.8% | 849 | median=0.0508 | 0 | 0.24 |
| 388 | `home_ss_roll_sm_conversion_rate` | float64 | 18.2% | 81.8% | 1295 | median=0.105 | 0 | 0.303 |
| 389 | `home_ss_roll_sm_big_chance_pct` | float64 | 18.2% | 81.8% | 1237 | median=0.0913 | 0 | 0.336 |
| 390 | `home_ss_roll_sm_avg_shot_distance` | float64 | 18.2% | 81.8% | 1225 | median=19.1 | 14.6 | 24.5 |
| 391 | `home_ss_roll_sm_median_shot_distance` | float64 | 18.2% | 81.8% | 1281 | median=18.8 | 13 | 27.4 |
| 392 | `home_ss_roll_sm_shot_distance_std` | float64 | 18.2% | 81.8% | 1061 | median=7.45 | 4.64 | 10.9 |
| 393 | `home_ss_roll_sm_close_range_pct` | float64 | 18.2% | 81.8% | 1102 | median=0.0658 | 0 | 0.272 |
| 394 | `home_ss_roll_xg_std` | float64 | 18.2% | 81.8% | 1442 | median=0.605 | 0.00481 | 2.09 |
| 395 | `home_ss_roll_rating_std` | float64 | 18.2% | 81.8% | 1440 | median=0.171 | 0.0255 | 0.485 |
| 396 | `home_top2_xg_share` | float64 | 18.5% | 81.5% | 1466 | median=0.638 | 0.27 | 1 |
| 397 | `away_ss_roll_xg` | float64 | 18.2% | 81.8% | 1438 | median=1.23 | 0.334 | 2.85 |
| 398 | `away_ss_roll_xgot` | float64 | 18.2% | 81.8% | 1438 | median=1.18 | 0.143 | 3.86 |
| 399 | `away_ss_roll_xa` | float64 | 18.2% | 81.8% | 1443 | median=0.757 | 0.226 | 2.12 |
| 400 | `away_ss_roll_goals` | float64 | 18.2% | 81.8% | 31 | median=1.2 | 0 | 4.5 |
| 401 | `away_ss_roll_total_shots` | float64 | 18.2% | 81.8% | 97 | median=12.4 | 4.5 | 23.5 |
| 402 | `away_ss_roll_shots_on_target` | float64 | 18.2% | 81.8% | 58 | median=4 | 1 | 9.4 |
| 403 | `away_ss_roll_big_chances_created` | float64 | 18.2% | 81.8% | 36 | median=1.4 | 0 | 5.5 |
| 404 | `away_ss_roll_key_passes` | float64 | 18.2% | 81.8% | 90 | median=9.2 | 3 | 19.5 |
| 405 | `away_ss_roll_pass_accuracy` | float64 | 18.2% | 81.8% | 1443 | median=0.817 | 0.653 | 0.916 |
| 406 | `away_ss_roll_xg_per_shot` | float64 | 18.2% | 81.8% | 1443 | median=0.1 | 0.0449 | 0.215 |
| 407 | `away_ss_roll_duel_win_rate` | float64 | 18.2% | 81.8% | 1443 | median=0.501 | 0.401 | 0.62 |
| 408 | `away_ss_roll_aerial_win_rate` | float64 | 18.2% | 81.8% | 1438 | median=0.498 | 0.344 | 0.711 |
| 409 | `away_ss_roll_tackles_won` | float64 | 18.2% | 81.8% | 68 | median=9 | 4.2 | 16.2 |
| 410 | `away_ss_roll_interceptions` | float64 | 18.2% | 81.8% | 66 | median=7.6 | 3.2 | 13.4 |
| 411 | `away_ss_roll_ball_recoveries` | float64 | 18.2% | 81.8% | 204 | median=48.8 | 34 | 75 |
| 412 | `away_ss_roll_blocks` | float64 | 18.2% | 81.8% | 59 | median=3.2 | 0 | 7.67 |
| 413 | `away_ss_roll_clearances` | float64 | 18.2% | 81.8% | 166 | median=18.6 | 4.67 | 39 |
| 414 | `away_ss_roll_fouls` | float64 | 18.2% | 81.8% | 77 | median=12.2 | 6.4 | 19.6 |
| 415 | `away_ss_roll_was_fouled` | float64 | 18.2% | 81.8% | 85 | median=11.6 | 6.2 | 18.6 |
| 416 | `away_ss_roll_progressive_carries` | float64 | 18.2% | 81.8% | 107 | median=0 | 0 | 26.6 |
| 417 | `away_ss_roll_rating` | float64 | 18.2% | 81.8% | 1351 | median=6.85 | 6.51 | 7.24 |
| 418 | `away_ss_roll_carry_distance_per_carry` | float64 | 18.2% | 81.8% | 273 | median=0 | 0 | 9.93 |
| 419 | `away_ss_roll_shot_blocked_rate` | float64 | 18.2% | 81.8% | 1420 | median=0.265 | 0.0827 | 0.548 |
| 420 | `away_ss_roll_last_man_tackle` | float64 | 18.2% | 81.8% | 9 | median=0 | 0 | 1.2 |
| 421 | `away_ss_roll_cross_accuracy` | float64 | 18.2% | 81.8% | 1427 | median=0.254 | 0.0455 | 0.473 |
| 422 | `away_ss_roll_error_to_goal` | float64 | 18.2% | 81.8% | 9 | median=0 | 0 | 0.8 |
| 423 | `away_ss_roll_error_to_shot` | float64 | 18.2% | 81.8% | 14 | median=0.2 | 0 | 1.6 |
| 424 | `away_ss_roll_contest_win_rate` | float64 | 18.2% | 81.8% | 1408 | median=0.484 | 0.251 | 0.727 |
| 425 | `away_ss_roll_unsuccessful_touch` | float64 | 18.2% | 81.8% | 89 | median=14.6 | 7.8 | 23.4 |
| 426 | `away_ss_roll_opp_half_pass_ratio` | float64 | 18.2% | 81.8% | 1443 | median=0.526 | 0.342 | 0.671 |
| 427 | `away_ss_roll_att_xg` | float64 | 18.2% | 81.8% | 1435 | median=0.637 | 0.0521 | 1.81 |
| 428 | `away_ss_roll_att_xa` | float64 | 18.2% | 81.8% | 1443 | median=0.202 | 0.0102 | 0.867 |
| 429 | `away_ss_roll_att_shots` | float64 | 18.2% | 81.8% | 76 | median=5.2 | 1.5 | 14.5 |
| 430 | `away_ss_roll_att_key_passes` | float64 | 18.2% | 81.8% | 62 | median=2.6 | 0 | 10.2 |
| 431 | `away_ss_roll_att_rating` | float64 | 18.2% | 81.8% | 974 | median=6.83 | 6.17 | 7.72 |
| 432 | `away_ss_roll_mid_pass_accuracy` | float64 | 18.2% | 81.8% | 1443 | median=0.827 | 0.649 | 0.927 |
| 433 | `away_ss_roll_mid_duel_win_rate` | float64 | 18.2% | 81.8% | 1442 | median=0.495 | 0.373 | 0.664 |
| 434 | `away_ss_roll_mid_tackles` | float64 | 18.2% | 81.8% | 79 | median=7.2 | 1.5 | 16 |
| 435 | `away_ss_roll_mid_interceptions` | float64 | 18.2% | 81.8% | 45 | median=3.2 | 0.6 | 7 |
| 436 | `away_ss_roll_mid_rating` | float64 | 18.2% | 81.8% | 1310 | median=6.84 | 6.47 | 7.3 |
| 437 | `away_ss_roll_def_aerial_win_rate` | float64 | 18.2% | 81.8% | 1381 | median=0.574 | 0.275 | 0.857 |
| 438 | `away_ss_roll_def_clearances` | float64 | 18.2% | 81.8% | 124 | median=11.6 | 3 | 30.8 |
| 439 | `away_ss_roll_def_blocks` | float64 | 18.2% | 81.8% | 43 | median=2 | 0 | 6 |
| 440 | `away_ss_roll_def_tackles_won` | float64 | 18.2% | 81.8% | 48 | median=3.4 | 0.5 | 7.2 |
| 441 | `away_ss_roll_def_rating` | float64 | 18.2% | 81.8% | 1084 | median=6.85 | 6.3 | 7.35 |
| 442 | `away_ss_roll_gk_goals_prevented` | float64 | 18.2% | 81.8% | 1441 | median=-0.0566 | -1.26 | 1.15 |
| 443 | `away_ss_roll_gk_rating` | float64 | 18.2% | 81.8% | 204 | median=6.98 | 6.02 | 8.3 |
| 444 | `away_ss_roll_territory_ratio` | float64 | 18.2% | 81.8% | 1443 | median=0.467 | 0.278 | 0.642 |
| 445 | `away_ss_roll_press_intensity` | float64 | 18.2% | 81.8% | 1443 | median=0.208 | 0.122 | 0.315 |
| 446 | `away_ss_roll_dispossess_rate` | float64 | 18.2% | 81.8% | 1443 | median=0.0136 | 0.00481 | 0.026 |
| 447 | `away_ss_roll_total_poss_lost` | float64 | 18.2% | 81.8% | 314 | median=124 | 93 | 172 |
| 448 | `away_ss_roll_xi_changes` | float64 | 18.0% | 82.0% | 46 | median=2.6 | 0.2 | 6.8 |
| 449 | `away_ss_roll_mins_concentration` | float64 | 18.2% | 81.8% | 1437 | median=0.903 | 0.85 | 0.965 |
| 450 | `away_ss_roll_counter_xg_pct` | float64 | 17.4% | 82.6% | 1112 | median=0.0591 | 0 | 0.456 |
| 451 | `away_ss_roll_set_piece_xg_pct` | float64 | 17.4% | 82.6% | 1379 | median=0.187 | 0.0121 | 0.534 |
| 452 | `away_ss_roll_open_play_xg_pct` | float64 | 17.4% | 82.6% | 1381 | median=0.731 | 0.256 | 0.988 |
| 453 | `away_ss_roll_header_shot_pct` | float64 | 17.4% | 82.6% | 1328 | median=0.18 | 0 | 0.575 |
| 454 | `away_ss_roll_first_half_xg_share` | float64 | 17.4% | 82.6% | 1381 | median=0.447 | 0.134 | 0.9 |
| 455 | `away_ss_roll_last_15_xg_share` | float64 | 17.4% | 82.6% | 1379 | median=0.229 | 0 | 0.552 |
| 456 | `away_ss_roll_corner_xg_share` | float64 | 17.4% | 82.6% | 1370 | median=0.133 | 0 | 0.474 |
| 457 | `away_ss_roll_free_kick_xg_share` | float64 | 17.4% | 82.6% | 999 | median=0.0171 | 0 | 0.279 |
| 458 | `away_ss_roll_penalty_xg_share` | float64 | 17.4% | 82.6% | 455 | median=0.0457 | 0 | 0.442 |
| 459 | `away_ss_roll_penalties_taken` | float64 | 17.4% | 82.6% | 11 | median=0.2 | 0 | 1 |
| 460 | `away_ss_roll_penalties_scored` | float64 | 17.4% | 82.6% | 9 | median=0 | 0 | 0.8 |
| 461 | `away_ss_roll_possession` | float64 | 18.2% | 81.8% | 195 | median=50.2 | 31 | 70.2 |
| 462 | `away_ss_roll_corners` | float64 | 18.2% | 81.8% | 53 | median=4.6 | 1.4 | 9.2 |
| 463 | `away_ss_roll_throw_ins` | float64 | 18.2% | 81.8% | 108 | median=19 | 11 | 32.2 |
| 464 | `away_ss_roll_shots_inside_box` | float64 | 18.2% | 81.8% | 82 | median=7.8 | 1.8 | 17.5 |
| 465 | `away_ss_roll_shots_outside_box` | float64 | 18.2% | 81.8% | 56 | median=4.6 | 1 | 10.5 |
| 466 | `away_ss_roll_shots_inside_box_pct` | float64 | 18.2% | 81.8% | 1414 | median=0.631 | 0.34 | 0.896 |
| 467 | `away_ss_roll_hit_woodwork` | float64 | 18.2% | 81.8% | 16 | median=0.2 | 0 | 2 |
| 468 | `away_ss_roll_big_chances_scored` | float64 | 18.2% | 81.8% | 24 | median=0.6 | 0 | 3 |
| 469 | `away_ss_roll_touches_in_opp_box` | float64 | 18.2% | 81.8% | 186 | median=10.6 | 0 | 44 |
| 470 | `away_ss_roll_fouled_final_third` | float64 | 18.2% | 81.8% | 37 | median=2 | 0.4 | 5.4 |
| 471 | `away_ss_roll_final_third_entries` | float64 | 18.2% | 81.8% | 209 | median=50.8 | 31 | 74.8 |
| 472 | `away_ss_roll_final_third_phases` | float64 | 18.2% | 81.8% | 451 | median=40.2 | 0 | 184 |
| 473 | `away_ss_roll_ground_duels_pct` | float64 | 18.2% | 81.8% | 128 | median=33.6 | 22.8 | 46.6 |
| 474 | `away_ss_roll_aerial_duels_pct` | float64 | 18.2% | 81.8% | 113 | median=14 | 5 | 31.6 |
| 475 | `away_ss_roll_dribbles_pct` | float64 | 18.2% | 81.8% | 78 | median=6.6 | 2 | 16.6 |
| 476 | `away_ss_roll_dive_saves` | float64 | 18.2% | 81.8% | 18 | median=0.2 | 0 | 2.4 |
| 477 | `away_ss_roll_high_claims` | float64 | 18.2% | 81.8% | 22 | median=0.2 | 0 | 3.4 |
| 478 | `away_ss_roll_dispossessed` | float64 | 18.2% | 81.8% | 71 | median=8.2 | 3.4 | 15.2 |
| 479 | `away_ss_roll_avg_shot_xg` | float64 | 18.2% | 81.8% | 1303 | median=0.1 | 0.0461 | 0.213 |
| 480 | `away_ss_roll_max_shot_xg` | float64 | 18.2% | 81.8% | 1404 | median=0.413 | 0.12 | 0.831 |
| 481 | `away_ss_roll_total_xgot` | float64 | 18.2% | 81.8% | 1434 | median=1.18 | 0.212 | 3.86 |
| 482 | `away_ss_roll_sm_inside_box_pct` | float64 | 18.2% | 81.8% | 1418 | median=0.638 | 0.34 | 0.896 |
| 483 | `away_ss_roll_sm_header_pct` | float64 | 18.2% | 81.8% | 1399 | median=0.18 | 0 | 0.575 |
| 484 | `away_ss_roll_sm_open_play_pct` | float64 | 18.2% | 81.8% | 1414 | median=0.662 | 0.365 | 0.889 |
| 485 | `away_ss_roll_sm_set_piece_pct` | float64 | 18.2% | 81.8% | 1410 | median=0.252 | 0.0835 | 0.525 |
| 486 | `away_ss_roll_sm_counter_pct` | float64 | 18.2% | 81.8% | 845 | median=0.0508 | 0 | 0.268 |
| 487 | `away_ss_roll_sm_conversion_rate` | float64 | 18.2% | 81.8% | 1316 | median=0.105 | 0 | 0.316 |
| 488 | `away_ss_roll_sm_big_chance_pct` | float64 | 18.2% | 81.8% | 1259 | median=0.0915 | 0 | 0.3 |
| 489 | `away_ss_roll_sm_avg_shot_distance` | float64 | 18.2% | 81.8% | 1218 | median=19.1 | 14.1 | 24 |
| 490 | `away_ss_roll_sm_median_shot_distance` | float64 | 18.2% | 81.8% | 1270 | median=18.8 | 13.1 | 26.4 |
| 491 | `away_ss_roll_sm_shot_distance_std` | float64 | 18.2% | 81.8% | 1056 | median=7.45 | 4.64 | 11.2 |
| 492 | `away_ss_roll_sm_close_range_pct` | float64 | 18.2% | 81.8% | 1124 | median=0.067 | 0 | 0.274 |
| 493 | `away_ss_roll_xg_std` | float64 | 18.2% | 81.8% | 1443 | median=0.605 | 0.0548 | 2.17 |
| 494 | `away_ss_roll_rating_std` | float64 | 18.2% | 81.8% | 1443 | median=0.17 | 0.0132 | 0.506 |
| 495 | `away_top2_xg_share` | float64 | 18.5% | 81.5% | 1448 | median=0.682 | 0.321 | 1 |
| 496 | `ss_diff_ss_roll_xg` | float64 | 18.1% | 81.9% | 1432 | median=-0.0425 | -1.81 | 1.56 |
| 497 | `ss_diff_ss_roll_xgot` | float64 | 18.1% | 81.9% | 1436 | median=-0.0296 | -2.59 | 2.38 |
| 498 | `ss_diff_ss_roll_xa` | float64 | 18.1% | 81.9% | 1436 | median=-0.0205 | -1.45 | 1.56 |
| 499 | `ss_diff_ss_roll_goals` | float64 | 18.1% | 81.9% | 99 | median=0 | -3 | 3 |
| 500 | `ss_diff_ss_roll_total_shots` | float64 | 18.1% | 81.9% | 265 | median=-0.2 | -14.3 | 12.2 |
| 501 | `ss_diff_ss_roll_shots_on_target` | float64 | 18.1% | 81.9% | 158 | median=0 | -6 | 4.6 |
| 502 | `ss_diff_ss_roll_big_chances_created` | float64 | 18.1% | 81.9% | 115 | median=0 | -4.5 | 4 |
| 503 | `ss_diff_ss_roll_key_passes` | float64 | 18.1% | 81.9% | 252 | median=-0.2 | -11.4 | 10.6 |
| 504 | `ss_diff_ss_roll_pass_accuracy` | float64 | 18.1% | 81.9% | 1436 | median=-0.000486 | -0.225 | 0.193 |
| 505 | `ss_diff_ss_roll_xg_per_shot` | float64 | 18.1% | 81.9% | 1436 | median=-0.00103 | -0.126 | 0.137 |
| 506 | `ss_diff_ss_roll_duel_win_rate` | float64 | 18.1% | 81.9% | 1436 | median=-0.000733 | -0.115 | 0.131 |
| 507 | `ss_diff_ss_roll_aerial_win_rate` | float64 | 18.1% | 81.9% | 1436 | median=-0.000183 | -0.273 | 0.275 |
| 508 | `ss_diff_ss_roll_tackles_won` | float64 | 18.1% | 81.9% | 199 | median=0 | -7.2 | 7.6 |
| 509 | `ss_diff_ss_roll_interceptions` | float64 | 18.1% | 81.9% | 193 | median=0 | -7.8 | 8.4 |
| 510 | `ss_diff_ss_roll_ball_recoveries` | float64 | 18.1% | 81.9% | 310 | median=-0.2 | -19.2 | 21.4 |
| 511 | `ss_diff_ss_roll_blocks` | float64 | 18.1% | 81.9% | 158 | median=0 | -6.33 | 6 |
| 512 | `ss_diff_ss_roll_clearances` | float64 | 18.1% | 81.9% | 396 | median=0 | -21.2 | 24.2 |
| 513 | `ss_diff_ss_roll_fouls` | float64 | 18.1% | 81.9% | 186 | median=0 | -9.8 | 9 |
| 514 | `ss_diff_ss_roll_was_fouled` | float64 | 18.1% | 81.9% | 202 | median=0 | -9.5 | 10.4 |
| 515 | `ss_diff_ss_roll_progressive_carries` | float64 | 18.1% | 81.9% | 208 | median=0 | -17 | 16.8 |
| 516 | `ss_diff_ss_roll_rating` | float64 | 18.1% | 81.9% | 1430 | median=-0.0129 | -0.635 | 0.479 |
| 517 | `ss_diff_ss_roll_carry_distance_per_carry` | float64 | 18.1% | 81.9% | 354 | median=0 | -9.85 | 9.75 |
| 518 | `ss_diff_ss_roll_shot_blocked_rate` | float64 | 18.1% | 81.9% | 1436 | median=-0.00313 | -0.34 | 0.282 |
| 519 | `ss_diff_ss_roll_last_man_tackle` | float64 | 18.1% | 81.9% | 25 | median=0 | -1.2 | 1.2 |
| 520 | `ss_diff_ss_roll_cross_accuracy` | float64 | 18.1% | 81.9% | 1436 | median=0.00194 | -0.29 | 0.29 |
| 521 | `ss_diff_ss_roll_error_to_goal` | float64 | 18.1% | 81.9% | 26 | median=0 | -0.8 | 0.8 |
| 522 | `ss_diff_ss_roll_error_to_shot` | float64 | 18.1% | 81.9% | 46 | median=0 | -1.6 | 1.5 |
| 523 | `ss_diff_ss_roll_contest_win_rate` | float64 | 18.1% | 81.9% | 1436 | median=-0.00367 | -0.408 | 0.366 |
| 524 | `ss_diff_ss_roll_unsuccessful_touch` | float64 | 18.1% | 81.9% | 223 | median=0 | -10.5 | 9.8 |
| 525 | `ss_diff_ss_roll_opp_half_pass_ratio` | float64 | 18.1% | 81.9% | 1436 | median=-0.00185 | -0.2 | 0.208 |
| 526 | `ss_diff_ss_roll_att_xg` | float64 | 18.1% | 81.9% | 1435 | median=-0.0205 | -1.47 | 1.63 |
| 527 | `ss_diff_ss_roll_att_xa` | float64 | 18.1% | 81.9% | 1436 | median=-0.0028 | -0.748 | 0.898 |
| 528 | `ss_diff_ss_roll_att_shots` | float64 | 18.1% | 81.9% | 225 | median=-0.2 | -9.67 | 9.2 |
| 529 | `ss_diff_ss_roll_att_key_passes` | float64 | 18.1% | 81.9% | 185 | median=0 | -7.6 | 8.4 |
| 530 | `ss_diff_ss_roll_att_rating` | float64 | 18.1% | 81.9% | 1285 | median=-0.00707 | -0.801 | 0.889 |
| 531 | `ss_diff_ss_roll_mid_pass_accuracy` | float64 | 18.1% | 81.9% | 1436 | median=-6.11e-05 | -0.233 | 0.175 |
| 532 | `ss_diff_ss_roll_mid_duel_win_rate` | float64 | 18.1% | 81.9% | 1436 | median=-0.000966 | -0.292 | 0.197 |
| 533 | `ss_diff_ss_roll_mid_tackles` | float64 | 18.1% | 81.9% | 230 | median=0 | -9 | 11 |
| 534 | `ss_diff_ss_roll_mid_interceptions` | float64 | 18.1% | 81.9% | 155 | median=0 | -4.6 | 6.5 |
| 535 | `ss_diff_ss_roll_mid_rating` | float64 | 18.1% | 81.9% | 1419 | median=-0.00738 | -0.659 | 0.533 |
| 536 | `ss_diff_ss_roll_def_aerial_win_rate` | float64 | 18.1% | 81.9% | 1436 | median=0.00349 | -0.472 | 0.381 |
| 537 | `ss_diff_ss_roll_def_clearances` | float64 | 18.1% | 81.9% | 342 | median=0.2 | -17.8 | 19.6 |
| 538 | `ss_diff_ss_roll_def_blocks` | float64 | 18.1% | 81.9% | 125 | median=0 | -5 | 5.5 |
| 539 | `ss_diff_ss_roll_def_tackles_won` | float64 | 18.1% | 81.9% | 146 | median=0 | -4.6 | 4.2 |
| 540 | `ss_diff_ss_roll_def_rating` | float64 | 18.1% | 81.9% | 1319 | median=-0.00769 | -0.864 | 0.876 |
| 541 | `ss_diff_ss_roll_gk_goals_prevented` | float64 | 18.1% | 81.9% | 1434 | median=0.0037 | -1.7 | 1.62 |
| 542 | `ss_diff_ss_roll_gk_rating` | float64 | 18.1% | 81.9% | 453 | median=0 | -1.58 | 2.1 |
| 543 | `ss_diff_ss_roll_territory_ratio` | float64 | 18.1% | 81.9% | 1436 | median=-0.00143 | -0.23 | 0.23 |
| 544 | `ss_diff_ss_roll_press_intensity` | float64 | 18.1% | 81.9% | 1436 | median=0.000652 | -0.135 | 0.172 |
| 545 | `ss_diff_ss_roll_dispossess_rate` | float64 | 18.1% | 81.9% | 1436 | median=-7.54e-05 | -0.0145 | 0.0163 |
| 546 | `ss_diff_ss_roll_total_poss_lost` | float64 | 18.1% | 81.9% | 666 | median=-0.9 | -59.4 | 50.6 |
| 547 | `ss_diff_ss_roll_xi_changes` | float64 | 17.9% | 82.1% | 153 | median=0 | -4.6 | 4.6 |
| 548 | `ss_diff_ss_roll_mins_concentration` | float64 | 18.1% | 81.9% | 1436 | median=-0.00049 | -0.0962 | 0.0798 |
| 549 | `ss_diff_ss_roll_counter_xg_pct` | float64 | 17.3% | 82.7% | 1348 | median=0.00234 | -0.438 | 0.382 |
| 550 | `ss_diff_ss_roll_set_piece_xg_pct` | float64 | 17.3% | 82.7% | 1370 | median=-0.000298 | -0.38 | 0.576 |
| 551 | `ss_diff_ss_roll_open_play_xg_pct` | float64 | 17.3% | 82.7% | 1370 | median=-0.00697 | -0.638 | 0.493 |
| 552 | `ss_diff_ss_roll_header_shot_pct` | float64 | 17.3% | 82.7% | 1370 | median=-0.000401 | -0.433 | 0.313 |
| 553 | `ss_diff_ss_roll_first_half_xg_share` | float64 | 17.3% | 82.7% | 1370 | median=0.00438 | -0.5 | 0.492 |
| 554 | `ss_diff_ss_roll_last_15_xg_share` | float64 | 17.3% | 82.7% | 1370 | median=-0.00497 | -0.454 | 0.393 |
| 555 | `ss_diff_ss_roll_corner_xg_share` | float64 | 17.3% | 82.7% | 1370 | median=-0.00333 | -0.361 | 0.576 |
| 556 | `ss_diff_ss_roll_free_kick_xg_share` | float64 | 17.3% | 82.7% | 1326 | median=0 | -0.231 | 0.227 |
| 557 | `ss_diff_ss_roll_penalty_xg_share` | float64 | 17.3% | 82.7% | 917 | median=0 | -0.442 | 0.441 |
| 558 | `ss_diff_ss_roll_penalties_taken` | float64 | 17.3% | 82.7% | 35 | median=0 | -1 | 1 |
| 559 | `ss_diff_ss_roll_penalties_scored` | float64 | 17.3% | 82.7% | 27 | median=0 | -0.8 | 1 |
| 560 | `ss_diff_ss_roll_possession` | float64 | 18.1% | 81.9% | 399 | median=0 | -30.8 | 27 |
| 561 | `ss_diff_ss_roll_corners` | float64 | 18.1% | 81.9% | 165 | median=0 | -6.07 | 6.2 |
| 562 | `ss_diff_ss_roll_throw_ins` | float64 | 18.1% | 81.9% | 280 | median=0 | -15.4 | 13.4 |
| 563 | `ss_diff_ss_roll_shots_inside_box` | float64 | 18.1% | 81.9% | 230 | median=0 | -11.4 | 9.6 |
| 564 | `ss_diff_ss_roll_shots_outside_box` | float64 | 18.1% | 81.9% | 155 | median=0 | -8 | 5 |
| 565 | `ss_diff_ss_roll_shots_inside_box_pct` | float64 | 18.1% | 81.9% | 1436 | median=-0.000755 | -0.359 | 0.484 |
| 566 | `ss_diff_ss_roll_hit_woodwork` | float64 | 18.1% | 81.9% | 49 | median=0 | -2 | 1.5 |
| 567 | `ss_diff_ss_roll_big_chances_scored` | float64 | 18.1% | 81.9% | 68 | median=0 | -3 | 2.2 |
| 568 | `ss_diff_ss_roll_touches_in_opp_box` | float64 | 18.1% | 81.9% | 439 | median=0 | -29 | 28.8 |
| 569 | `ss_diff_ss_roll_fouled_final_third` | float64 | 18.1% | 81.9% | 118 | median=0 | -3.4 | 4.2 |
| 570 | `ss_diff_ss_roll_final_third_entries` | float64 | 18.1% | 81.9% | 418 | median=-0.2 | -31 | 36.6 |
| 571 | `ss_diff_ss_roll_final_third_phases` | float64 | 18.1% | 81.9% | 734 | median=0 | -124 | 111 |
| 572 | `ss_diff_ss_roll_duel_won_pct` | float64 | 18.1% | 81.9% | 200 | median=0 | -11.5 | 13 |
| 573 | `ss_diff_ss_roll_ground_duels_pct` | float64 | 18.1% | 81.9% | 345 | median=0 | -17.8 | 18.6 |
| 574 | `ss_diff_ss_roll_aerial_duels_pct` | float64 | 18.1% | 81.9% | 317 | median=0 | -21.2 | 16.8 |
| 575 | `ss_diff_ss_roll_dribbles_pct` | float64 | 18.1% | 81.9% | 234 | median=-0.2 | -9.8 | 8 |
| 576 | `ss_diff_ss_roll_dive_saves` | float64 | 18.1% | 81.9% | 58 | median=0 | -2 | 2.4 |
| 577 | `ss_diff_ss_roll_high_claims` | float64 | 18.1% | 81.9% | 79 | median=0 | -3 | 2.8 |
| 578 | `ss_diff_ss_roll_dispossessed` | float64 | 18.1% | 81.9% | 205 | median=0 | -7.4 | 8.4 |
| 579 | `ss_diff_ss_roll_avg_shot_xg` | float64 | 18.1% | 81.9% | 1388 | median=-0.0006 | -0.11 | 0.141 |
| 580 | `ss_diff_ss_roll_max_shot_xg` | float64 | 18.1% | 81.9% | 1424 | median=0.00028 | -0.513 | 0.548 |
| 581 | `ss_diff_ss_roll_total_xgot` | float64 | 18.1% | 81.9% | 1432 | median=-0.0233 | -2.58 | 2.38 |
| 582 | `ss_diff_ss_roll_sm_inside_box_pct` | float64 | 18.1% | 81.9% | 1436 | median=-0.00107 | -0.359 | 0.474 |
| 583 | `ss_diff_ss_roll_sm_header_pct` | float64 | 18.1% | 81.9% | 1436 | median=7.7e-07 | -0.433 | 0.313 |
| 584 | `ss_diff_ss_roll_sm_open_play_pct` | float64 | 18.1% | 81.9% | 1436 | median=-0.00278 | -0.379 | 0.347 |
| 585 | `ss_diff_ss_roll_sm_set_piece_pct` | float64 | 18.1% | 81.9% | 1436 | median=-0.0015 | -0.371 | 0.363 |
| 586 | `ss_diff_ss_roll_sm_counter_pct` | float64 | 18.1% | 81.9% | 1361 | median=0.00198 | -0.237 | 0.2 |
| 587 | `ss_diff_ss_roll_sm_conversion_rate` | float64 | 18.1% | 81.9% | 1433 | median=-0.00175 | -0.245 | 0.227 |
| 588 | `ss_diff_ss_roll_sm_big_chance_pct` | float64 | 18.1% | 81.9% | 1432 | median=-0.001 | -0.265 | 0.294 |
| 589 | `ss_diff_ss_roll_sm_avg_shot_distance` | float64 | 18.1% | 81.9% | 1341 | median=-0.035 | -8.22 | 7.92 |
| 590 | `ss_diff_ss_roll_sm_median_shot_distance` | float64 | 18.1% | 81.9% | 1369 | median=-0.005 | -10.1 | 9.81 |
| 591 | `ss_diff_ss_roll_sm_shot_distance_std` | float64 | 18.1% | 81.9% | 1330 | median=0.054 | -4.36 | 4.77 |
| 592 | `ss_diff_ss_roll_sm_close_range_pct` | float64 | 18.1% | 81.9% | 1418 | median=-0.00192 | -0.236 | 0.224 |
| 593 | `ss_diff_ss_roll_xg_std` | float64 | 18.1% | 81.9% | 1436 | median=0.0016 | -1.41 | 1.46 |
| 594 | `ss_diff_ss_roll_rating_std` | float64 | 18.1% | 81.9% | 1436 | median=-0.00143 | -0.361 | 0.349 |
| 595 | `ss_diff_top2_xg_share` | float64 | 18.5% | 81.5% | 1470 | median=-0.0436 | -0.614 | 0.577 |
| 596 | `ss_coverage` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 597 | `ss_xg_coverage` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 598 | `home_ss_idx_attacking` | float64 | 18.2% | 81.8% | 1440 | median=0.479 | 0.0328 | 1 |
| 599 | `home_ss_idx_chance_creation` | float64 | 18.2% | 81.8% | 1435 | median=0.504 | 0.0976 | 1 |
| 600 | `home_ss_idx_shot_quality` | float64 | 18.2% | 81.8% | 1440 | median=0.525 | 0.0133 | 1 |
| 601 | `home_ss_idx_defense` | float64 | 18.2% | 81.8% | 1439 | median=0.519 | 0.0917 | 1 |
| 602 | `home_ss_idx_pressing` | float64 | 18.2% | 81.8% | 1431 | median=0.443 | 0.0507 | 1 |
| 603 | `home_ss_idx_passing` | float64 | 18.2% | 81.8% | 1437 | median=0.539 | 0.133 | 1 |
| 604 | `home_ss_idx_gk` | float64 | 18.2% | 81.8% | 1429 | median=0.587 | 0.219 | 1 |
| 605 | `home_ss_idx_duels` | float64 | 18.2% | 81.8% | 1442 | median=0.503 | 0.0499 | 1 |
| 606 | `home_ss_idx_match_control` | float64 | 18.2% | 81.8% | 1438 | median=0.534 | 0.135 | 1 |
| 607 | `home_ss_idx_set_pieces` | float64 | 18.2% | 81.8% | 1432 | median=0.5 | 0.0125 | 1 |
| 608 | `away_ss_idx_attacking` | float64 | 18.2% | 81.8% | 1439 | median=0.482 | 0.0341 | 1 |
| 609 | `away_ss_idx_chance_creation` | float64 | 18.2% | 81.8% | 1437 | median=0.492 | 0.0958 | 1 |
| 610 | `away_ss_idx_shot_quality` | float64 | 18.2% | 81.8% | 1439 | median=0.524 | 0.0245 | 1 |
| 611 | `away_ss_idx_defense` | float64 | 18.2% | 81.8% | 1436 | median=0.517 | 0.115 | 1 |
| 612 | `away_ss_idx_pressing` | float64 | 18.2% | 81.8% | 1440 | median=0.441 | 0.0355 | 1 |
| 613 | `away_ss_idx_passing` | float64 | 18.2% | 81.8% | 1433 | median=0.539 | 0.127 | 1 |
| 614 | `away_ss_idx_gk` | float64 | 18.2% | 81.8% | 1434 | median=0.585 | 0.212 | 1 |
| 615 | `away_ss_idx_duels` | float64 | 18.2% | 81.8% | 1440 | median=0.505 | 0.0469 | 1 |
| 616 | `away_ss_idx_match_control` | float64 | 18.2% | 81.8% | 1437 | median=0.525 | 0.12 | 1 |
| 617 | `away_ss_idx_set_pieces` | float64 | 18.2% | 81.8% | 1434 | median=0.504 | 0.0454 | 1 |
| 618 | `ss_idx_diff_attacking` | float64 | 18.1% | 81.9% | 1434 | median=-0.00857 | -0.85 | 0.85 |
| 619 | `ss_idx_diff_chance_creation` | float64 | 18.1% | 81.9% | 1434 | median=0.00515 | -0.731 | 0.688 |
| 620 | `ss_idx_diff_shot_quality` | float64 | 18.1% | 81.9% | 1436 | median=0.00256 | -0.857 | 0.89 |
| 621 | `ss_idx_diff_defense` | float64 | 18.1% | 81.9% | 1435 | median=-0.00721 | -0.655 | 0.635 |
| 622 | `ss_idx_diff_pressing` | float64 | 18.1% | 81.9% | 1434 | median=0.00221 | -0.711 | 0.836 |
| 623 | `ss_idx_diff_passing` | float64 | 18.1% | 81.9% | 1434 | median=0.00441 | -0.715 | 0.747 |
| 624 | `ss_idx_diff_gk` | float64 | 18.1% | 81.9% | 1431 | median=0.00677 | -0.627 | 0.686 |
| 625 | `ss_idx_diff_duels` | float64 | 18.1% | 81.9% | 1436 | median=-0.00444 | -0.712 | 0.693 |
| 626 | `ss_idx_diff_match_control` | float64 | 18.1% | 81.9% | 1430 | median=-0.00104 | -0.69 | 0.651 |
| 627 | `ss_idx_diff_set_pieces` | float64 | 18.1% | 81.9% | 1434 | median=-0.0146 | -0.725 | 0.909 |
| 628 | `home_lineup_xg_sum` | float64 | 18.5% | 81.5% | 1434 | median=0.997 | 0.0489 | 5.44 |
| 629 | `home_lineup_xa_sum` | float64 | 18.5% | 81.5% | 1465 | median=0.648 | 0.05 | 3.65 |
| 630 | `home_lineup_rating_mean` | float64 | 18.5% | 81.5% | 181 | median=6.96 | 6.12 | 7.86 |
| 631 | `home_lineup_rotation` | float64 | 18.3% | 81.7% | 10 | median=3 | 0 | 9 |
| 632 | `away_lineup_xg_sum` | float64 | 18.5% | 81.5% | 1423 | median=0.806 | 0 | 4.31 |
| 633 | `away_lineup_xa_sum` | float64 | 18.5% | 81.5% | 1465 | median=0.528 | 0.0158 | 3.03 |
| 634 | `away_lineup_rating_mean` | float64 | 18.5% | 81.5% | 167 | median=6.89 | 6.03 | 7.68 |
| 635 | `away_lineup_rotation` | float64 | 18.3% | 81.7% | 12 | median=3 | 0 | 11 |
| 636 | `lineup_xg_sum_diff` | float64 | 18.5% | 81.5% | 1452 | median=0.163 | -3.38 | 5.1 |
| 637 | `lineup_xa_sum_diff` | float64 | 18.5% | 81.5% | 1465 | median=0.105 | -2.56 | 3.48 |
| 638 | `lineup_rating_mean_diff` | float64 | 18.5% | 81.5% | 545 | median=0.0636 | -1.54 | 1.79 |
| 639 | `home_us_top11_xg90_sum` | float64 | 56.9% | 43.1% | 240 | median=3.06 | 1.39 | 7.53 |
| 640 | `home_us_squad_depth` | float64 | 56.9% | 43.1% | 15 | median=22 | 17 | 33 |
| 641 | `home_us_xg_concentration` | float64 | 56.9% | 43.1% | 240 | median=0.208 | 0.128 | 0.516 |
| 642 | `away_us_top11_xg90_sum` | float64 | 56.9% | 43.1% | 240 | median=3.06 | 1.39 | 7.53 |
| 643 | `away_us_squad_depth` | float64 | 56.9% | 43.1% | 15 | median=22 | 17 | 33 |
| 644 | `away_us_xg_concentration` | float64 | 56.9% | 43.1% | 240 | median=0.208 | 0.128 | 0.516 |
| 645 | `us_top11_xg90_sum_diff` | float64 | 56.9% | 43.1% | 4510 | median=0 | -5.51 | 5.51 |
| 646 | `us_squad_depth_diff` | float64 | 56.9% | 43.1% | 33 | median=0 | -16 | 16 |
| 647 | `h2h_home_dominance` | float64 | 82.1% | 17.9% | 65 | median=0.5 | 0 | 1 |
| 648 | `h2h_goals_diff` | float64 | 82.1% | 17.9% | 171 | median=0 | -5 | 5 |
| 649 | `h2h_btts_rate` | float64 | 82.1% | 17.9% | 33 | median=0.5 | 0 | 1 |
| 650 | `h2h_over25_rate` | float64 | 82.1% | 17.9% | 33 | median=0.5 | 0 | 1 |
| 651 | `h2h_meetings` | float64 | 82.1% | 17.9% | 9 | median=10 | 2 | 10 |
| 652 | `kickoff_hour` | float64 | 100.0% | 0.0% | 9 | median=15 | 10 | 19 |
| 653 | `is_night_match` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 654 | `day_of_week` | int32 | 100.0% | 0.0% | 7 | median=6 | 0 | 6 |
| 655 | `is_weekend` | int64 | 100.0% | 0.0% | 2 | median=1 | 0 | 1 |
| 656 | `is_midweek` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 657 | `season_phase` | float64 | 100.0% | 0.0% | 5 | median=3 | 1 | 5 |
| 658 | `home_post_intl_break` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 659 | `away_post_intl_break` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 660 | `is_no_crowd_match` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 661 | `five_sub_era` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 662 | `var_era` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 663 | `congestion_asymmetry` | float64 | 100.0% | 0.0% | 26 | median=0 | -18 | 18 |
| 664 | `any_team_fatigued` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 665 | `matchweek_avg_goals` | float64 | 99.7% | 0.3% | 5517 | median=2.6 | 0 | 6 |
| 666 | `home_is_promoted` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 667 | `away_is_promoted` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 668 | `promoted_derby` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 669 | `promoted_vs_established` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 670 | `home_seasons_since_topflight` | int64 | 100.0% | 0.0% | 8 | median=0 | 0 | 10 |
| 671 | `away_seasons_since_topflight` | int64 | 100.0% | 0.0% | 8 | median=0 | 0 | 10 |
| 672 | `home_previously_in_league` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 673 | `away_previously_in_league` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 674 | `home_captain_played` | float64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 675 | `away_captain_played` | float64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 676 | `home_captain_consistency` | float64 | 100.0% | 0.0% | 6 | median=0 | 0 | 1 |
| 677 | `away_captain_consistency` | float64 | 100.0% | 0.0% | 6 | median=0 | 0 | 1 |
| 678 | `home_captain_effect` | float64 | 100.0% | 0.0% | 457 | median=0 | -0.5 | 0.5 |
| 679 | `away_captain_effect` | float64 | 100.0% | 0.0% | 455 | median=0 | -0.5 | 0.667 |
| 680 | `home_ct_first_card_min_r5` | float64 | 100.0% | 0.0% | 331 | median=0 | 0 | 84.2 |
| 681 | `home_ct_cards_before_30_r5` | float64 | 100.0% | 0.0% | 14 | median=0 | 0 | 1.6 |
| 682 | `home_ct_cards_after_75_r5` | float64 | 100.0% | 0.0% | 18 | median=0 | 0 | 2.4 |
| 683 | `home_ct_card_timing_spread_r5` | float64 | 100.0% | 0.0% | 1290 | median=0 | 0 | 31.5 |
| 684 | `home_ct_first_goal_min_r5` | float64 | 100.0% | 0.0% | 336 | median=0 | 0 | 90 |
| 685 | `home_ct_goals_first_15_r5` | float64 | 100.0% | 0.0% | 10 | median=0 | 0 | 1 |
| 686 | `home_ct_goals_last_15_r5` | float64 | 100.0% | 0.0% | 11 | median=0 | 0 | 1.2 |
| 687 | `home_ct_total_goals_scored_r5` | float64 | 100.0% | 0.0% | 28 | median=0 | 0 | 3.6 |
| 688 | `home_ct_conceded_last_15_r5` | float64 | 100.0% | 0.0% | 12 | median=0 | 0 | 1.2 |
| 689 | `home_ct_total_goals_conceded_r5` | float64 | 100.0% | 0.0% | 28 | median=0 | 0 | 3.2 |
| 690 | `home_ct_first_sub_min_r5` | float64 | 100.0% | 0.0% | 195 | median=0 | 0 | 77.6 |
| 691 | `home_ct_early_sub_r5` | float64 | 100.0% | 0.0% | 11 | median=0 | 0 | 1 |
| 692 | `home_ct_goals_first_15_rate` | float64 | 100.0% | 0.0% | 35 | median=0 | 0 | 1 |
| 693 | `home_ct_goals_last_15_rate` | float64 | 100.0% | 0.0% | 49 | median=0 | 0 | 1 |
| 694 | `home_ct_conceded_last_15_rate` | float64 | 100.0% | 0.0% | 45 | median=0 | 0 | 1 |
| 695 | `away_ct_first_card_min_r5` | float64 | 100.0% | 0.0% | 332 | median=0 | 0 | 84 |
| 696 | `away_ct_cards_before_30_r5` | float64 | 100.0% | 0.0% | 14 | median=0 | 0 | 1.6 |
| 697 | `away_ct_cards_after_75_r5` | float64 | 100.0% | 0.0% | 19 | median=0 | 0 | 2.2 |
| 698 | `away_ct_card_timing_spread_r5` | float64 | 100.0% | 0.0% | 1282 | median=0 | 0 | 30.3 |
| 699 | `away_ct_first_goal_min_r5` | float64 | 100.0% | 0.0% | 329 | median=0 | 0 | 90 |
| 700 | `away_ct_goals_first_15_r5` | float64 | 100.0% | 0.0% | 10 | median=0 | 0 | 1 |
| 701 | `away_ct_goals_last_15_r5` | float64 | 100.0% | 0.0% | 13 | median=0 | 0 | 1.4 |
| 702 | `away_ct_total_goals_scored_r5` | float64 | 100.0% | 0.0% | 32 | median=0 | 0 | 3.8 |
| 703 | `away_ct_conceded_last_15_r5` | float64 | 100.0% | 0.0% | 13 | median=0 | 0 | 1.4 |
| 704 | `away_ct_total_goals_conceded_r5` | float64 | 100.0% | 0.0% | 30 | median=0 | 0 | 3.4 |
| 705 | `away_ct_first_sub_min_r5` | float64 | 100.0% | 0.0% | 192 | median=0 | 0 | 77 |
| 706 | `away_ct_early_sub_r5` | float64 | 100.0% | 0.0% | 11 | median=0 | 0 | 1 |
| 707 | `away_ct_goals_first_15_rate` | float64 | 100.0% | 0.0% | 39 | median=0 | 0 | 1 |
| 708 | `away_ct_goals_last_15_rate` | float64 | 100.0% | 0.0% | 54 | median=0 | 0 | 1 |
| 709 | `away_ct_conceded_last_15_rate` | float64 | 100.0% | 0.0% | 49 | median=0 | 0 | 1 |
| 710 | `home_fb_roll_goals` | float64 | 42.1% | 57.9% | 32 | median=1.2 | 0 | 3.8 |
| 711 | `home_fb_roll_assists` | float64 | 42.1% | 57.9% | 28 | median=0.8 | 0 | 13.8 |
| 712 | `home_fb_roll_shots` | float64 | 42.1% | 57.9% | 121 | median=12.6 | 4.2 | 65.2 |
| 713 | `home_fb_roll_shots_on_target` | float64 | 42.1% | 57.9% | 66 | median=4.2 | 0.8 | 17.6 |
| 714 | `home_fb_roll_shot_accuracy` | float64 | 42.1% | 57.9% | 3120 | median=0.337 | 0.0836 | 0.618 |
| 715 | `home_fb_roll_cards_yellow` | float64 | 42.1% | 57.9% | 36 | median=2.2 | 0.4 | 5.4 |
| 716 | `home_fb_roll_cards_red` | float64 | 42.1% | 57.9% | 9 | median=0 | 0 | 1 |
| 717 | `home_fb_roll_fouls` | float64 | 42.1% | 57.9% | 112 | median=12.8 | 0 | 20.8 |
| 718 | `home_fb_roll_fouled` | float64 | 42.1% | 57.9% | 109 | median=11.8 | 0 | 20.8 |
| 719 | `home_fb_roll_interceptions` | float64 | 42.1% | 57.9% | 107 | median=9.2 | 3.2 | 44 |
| 720 | `home_fb_roll_tackles_won` | float64 | 42.1% | 57.9% | 100 | median=9 | 0 | 18 |
| 721 | `home_fb_roll_blocks` | float64 | 42.1% | 57.9% | 73 | median=0 | 0 | 50.2 |
| 722 | `home_fb_roll_squad_players` | float64 | 42.1% | 57.9% | 23 | median=15.4 | 12.6 | 16.4 |
| 723 | `home_fb_roll_avg_minutes_per_player` | float64 | 42.1% | 57.9% | 785 | median=64.5 | 60.4 | 238 |
| 724 | `home_fb_roll_age_years` | float64 | 42.1% | 57.9% | 2378 | median=26.4 | 22.9 | 31 |
| 725 | `home_fb_roll_xg` | float64 | 42.1% | 57.9% | 148 | median=0 | 0 | 2.86 |
| 726 | `home_fb_roll_npxg` | float64 | 42.1% | 57.9% | 143 | median=0 | 0 | 2.7 |
| 727 | `home_fb_roll_xg_assist` | float64 | 42.1% | 57.9% | 124 | median=0 | 0 | 3.12 |
| 728 | `home_fb_roll_sca` | float64 | 42.1% | 57.9% | 135 | median=0 | 0 | 65.4 |
| 729 | `home_fb_roll_gca` | float64 | 42.1% | 57.9% | 32 | median=0 | 0 | 15.4 |
| 730 | `home_fb_roll_pass_accuracy` | float64 | 42.1% | 57.9% | 413 | median=0 | 0 | 0.893 |
| 731 | `home_fb_roll_progressive_passes` | float64 | 42.1% | 57.9% | 195 | median=0 | 0 | 160 |
| 732 | `home_fb_roll_xg_per_shot` | float64 | 42.1% | 57.9% | 404 | median=0 | 0 | 0.185 |
| 733 | `home_fb_roll_tackles` | float64 | 42.1% | 57.9% | 98 | median=0 | 0 | 27.2 |
| 734 | `home_fb_roll_touches` | float64 | 42.1% | 57.9% | 366 | median=0 | 0 | 2.58e+03 |
| 735 | `home_fb_roll_carries` | float64 | 42.1% | 57.9% | 367 | median=0 | 0 | 1.77e+03 |
| 736 | `home_fb_roll_progressive_carries` | float64 | 42.1% | 57.9% | 116 | median=0 | 0 | 98.4 |
| 737 | `home_fb_roll_defense_tackles_won` | float64 | 42.1% | 57.9% | 66 | median=0 | 0 | 14.8 |
| 738 | `home_fb_roll_defense_interceptions` | float64 | 42.1% | 57.9% | 57 | median=0 | 0 | 44 |
| 739 | `home_fb_roll_defense_clearances` | float64 | 42.1% | 57.9% | 135 | median=0 | 0 | 108 |
| 740 | `home_fb_roll_misc_ball_recoveries` | float64 | 42.1% | 57.9% | 144 | median=0 | 0 | 159 |
| 741 | `home_fb_roll_aerial_win_rate` | float64 | 42.1% | 57.9% | 412 | median=0 | 0 | 0.651 |
| 742 | `home_fb_roll_goals_std` | float64 | 42.1% | 57.9% | 462 | median=0.894 | 0 | 3 |
| 743 | `away_fb_roll_goals` | float64 | 42.0% | 58.0% | 33 | median=1.2 | 0 | 4 |
| 744 | `away_fb_roll_assists` | float64 | 42.0% | 58.0% | 30 | median=0.8 | 0 | 13.8 |
| 745 | `away_fb_roll_shots` | float64 | 42.0% | 58.0% | 113 | median=12.8 | 4.6 | 67.2 |
| 746 | `away_fb_roll_shots_on_target` | float64 | 42.0% | 58.0% | 68 | median=4.2 | 1 | 18 |
| 747 | `away_fb_roll_shot_accuracy` | float64 | 42.0% | 58.0% | 3128 | median=0.337 | 0.1 | 0.627 |
| 748 | `away_fb_roll_cards_yellow` | float64 | 42.0% | 58.0% | 35 | median=2.2 | 0.4 | 5.2 |
| 749 | `away_fb_roll_cards_red` | float64 | 42.0% | 58.0% | 8 | median=0 | 0 | 0.8 |
| 750 | `away_fb_roll_fouls` | float64 | 42.0% | 58.0% | 110 | median=12.6 | 0 | 21.4 |
| 751 | `away_fb_roll_fouled` | float64 | 42.0% | 58.0% | 111 | median=11.8 | 0 | 22 |
| 752 | `away_fb_roll_interceptions` | float64 | 42.0% | 58.0% | 105 | median=9.2 | 2.8 | 44.2 |
| 753 | `away_fb_roll_tackles_won` | float64 | 42.0% | 58.0% | 100 | median=9 | 0 | 19.6 |
| 754 | `away_fb_roll_blocks` | float64 | 42.0% | 58.0% | 76 | median=0 | 0 | 50.2 |
| 755 | `away_fb_roll_squad_players` | float64 | 42.0% | 58.0% | 22 | median=15.4 | 12.6 | 16.6 |
| 756 | `away_fb_roll_avg_minutes_per_player` | float64 | 42.0% | 58.0% | 784 | median=64.5 | 59.7 | 238 |
| 757 | `away_fb_roll_age_years` | float64 | 42.0% | 58.0% | 2366 | median=26.4 | 22.8 | 31.2 |
| 758 | `away_fb_roll_xg` | float64 | 42.0% | 58.0% | 157 | median=0 | 0 | 2.88 |
| 759 | `away_fb_roll_npxg` | float64 | 42.0% | 58.0% | 145 | median=0 | 0 | 2.76 |
| 760 | `away_fb_roll_xg_assist` | float64 | 42.0% | 58.0% | 127 | median=0 | 0 | 3.2 |
| 761 | `away_fb_roll_sca` | float64 | 42.0% | 58.0% | 127 | median=0 | 0 | 68.6 |
| 762 | `away_fb_roll_gca` | float64 | 42.0% | 58.0% | 35 | median=0 | 0 | 15.6 |
| 763 | `away_fb_roll_pass_accuracy` | float64 | 42.0% | 58.0% | 413 | median=0 | 0 | 0.895 |
| 764 | `away_fb_roll_progressive_passes` | float64 | 42.0% | 58.0% | 197 | median=0 | 0 | 165 |
| 765 | `away_fb_roll_xg_per_shot` | float64 | 42.0% | 58.0% | 412 | median=0 | 0 | 0.178 |
| 766 | `away_fb_roll_tackles` | float64 | 42.0% | 58.0% | 98 | median=0 | 0 | 26.8 |
| 767 | `away_fb_roll_touches` | float64 | 42.0% | 58.0% | 373 | median=0 | 0 | 2.61e+03 |
| 768 | `away_fb_roll_carries` | float64 | 42.0% | 58.0% | 362 | median=0 | 0 | 1.8e+03 |
| 769 | `away_fb_roll_progressive_carries` | float64 | 42.0% | 58.0% | 134 | median=0 | 0 | 98.8 |
| 770 | `away_fb_roll_defense_tackles_won` | float64 | 42.0% | 58.0% | 65 | median=0 | 0 | 14.6 |
| 771 | `away_fb_roll_defense_interceptions` | float64 | 42.0% | 58.0% | 59 | median=0 | 0 | 44.2 |
| 772 | `away_fb_roll_defense_clearances` | float64 | 42.0% | 58.0% | 140 | median=0 | 0 | 107 |
| 773 | `away_fb_roll_misc_ball_recoveries` | float64 | 42.0% | 58.0% | 150 | median=0 | 0 | 162 |
| 774 | `away_fb_roll_aerial_win_rate` | float64 | 42.0% | 58.0% | 413 | median=0 | 0 | 0.644 |
| 775 | `away_fb_roll_goals_std` | float64 | 42.0% | 58.0% | 471 | median=0.894 | 0 | 3 |
| 776 | `fb_diff_goals` | float64 | 41.9% | 58.1% | 119 | median=0 | -3 | 3 |
| 777 | `fb_diff_assists` | float64 | 41.9% | 58.1% | 99 | median=0 | -13.6 | 12.8 |
| 778 | `fb_diff_shots` | float64 | 41.9% | 58.1% | 392 | median=-0.2 | -56.4 | 58.6 |
| 779 | `fb_diff_shots_on_target` | float64 | 41.9% | 58.1% | 224 | median=0 | -14.2 | 14.2 |
| 780 | `fb_diff_shot_accuracy` | float64 | 41.9% | 58.1% | 3325 | median=-0.00152 | -0.36 | 0.365 |
| 781 | `fb_diff_cards_yellow` | float64 | 41.9% | 58.1% | 118 | median=0 | -3.35 | 3.2 |
| 782 | `fb_diff_cards_red` | float64 | 41.9% | 58.1% | 25 | median=0 | -0.8 | 1 |
| 783 | `fb_diff_fouls` | float64 | 41.9% | 58.1% | 276 | median=0 | -10 | 12.2 |
| 784 | `fb_diff_fouled` | float64 | 41.9% | 58.1% | 272 | median=0 | -11.6 | 11 |
| 785 | `fb_diff_interceptions` | float64 | 41.9% | 58.1% | 295 | median=0 | -39 | 36.4 |
| 786 | `fb_diff_tackles_won` | float64 | 41.9% | 58.1% | 253 | median=0 | -11 | 9.2 |
| 787 | `fb_diff_blocks` | float64 | 41.9% | 58.1% | 153 | median=0 | -41.4 | 40 |
| 788 | `fb_diff_squad_players` | float64 | 41.9% | 58.1% | 43 | median=0 | -2.4 | 2.2 |
| 789 | `fb_diff_avg_minutes_per_player` | float64 | 41.9% | 58.1% | 1838 | median=0 | -173 | 176 |
| 790 | `fb_diff_age_years` | float64 | 41.9% | 58.1% | 3133 | median=-0.0161 | -5.78 | 5.82 |
| 791 | `fb_diff_xg` | float64 | 41.9% | 58.1% | 278 | median=0 | -1.78 | 2 |
| 792 | `fb_diff_npxg` | float64 | 41.9% | 58.1% | 249 | median=0 | -1.92 | 2.16 |
| 793 | `fb_diff_xg_assist` | float64 | 41.9% | 58.1% | 250 | median=0 | -2.56 | 2.54 |
| 794 | `fb_diff_sca` | float64 | 41.9% | 58.1% | 233 | median=0 | -48.8 | 53.6 |
| 795 | `fb_diff_gca` | float64 | 41.9% | 58.1% | 102 | median=0 | -15 | 13.4 |
| 796 | `fb_diff_pass_accuracy` | float64 | 41.9% | 58.1% | 417 | median=0 | -0.856 | 0.876 |
| 797 | `fb_diff_progressive_passes` | float64 | 41.9% | 58.1% | 290 | median=0 | -132 | 128 |
| 798 | `fb_diff_xg_per_shot` | float64 | 41.9% | 58.1% | 417 | median=0 | -0.103 | 0.117 |
| 799 | `fb_diff_tackles` | float64 | 41.9% | 58.1% | 185 | median=0 | -12.8 | 16.2 |
| 800 | `fb_diff_touches` | float64 | 41.9% | 58.1% | 390 | median=0 | -2.02e+03 | 2.03e+03 |
| 801 | `fb_diff_carries` | float64 | 41.9% | 58.1% | 392 | median=0 | -1.45e+03 | 1.5e+03 |
| 802 | `fb_diff_progressive_carries` | float64 | 41.9% | 58.1% | 240 | median=0 | -83 | 83.6 |
| 803 | `fb_diff_defense_tackles_won` | float64 | 41.9% | 58.1% | 146 | median=0 | -7.2 | 9 |
| 804 | `fb_diff_defense_interceptions` | float64 | 41.9% | 58.1% | 133 | median=0 | -39 | 36.4 |
| 805 | `fb_diff_defense_clearances` | float64 | 41.9% | 58.1% | 235 | median=0 | -79.4 | 91 |
| 806 | `fb_diff_misc_ball_recoveries` | float64 | 41.9% | 58.1% | 209 | median=0 | -124 | 123 |
| 807 | `fb_diff_aerial_win_rate` | float64 | 41.9% | 58.1% | 417 | median=0 | -0.491 | 0.589 |
| 808 | `fb_diff_goals_std` | float64 | 41.9% | 58.1% | 2443 | median=0 | -2.49 | 2.33 |
| 809 | `fb_coverage` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 810 | `home_league_pos` | int64 | 100.0% | 0.0% | 20 | median=11 | 1 | 20 |
| 811 | `home_league_points` | int64 | 100.0% | 0.0% | 95 | median=22 | 0 | 99 |
| 812 | `home_league_gd` | int64 | 100.0% | 0.0% | 114 | median=-1 | -53 | 67 |
| 813 | `home_league_goals_for` | int64 | 100.0% | 0.0% | 90 | median=22 | 0 | 98 |
| 814 | `home_league_wins` | int64 | 100.0% | 0.0% | 32 | median=6 | 0 | 32 |
| 815 | `home_league_draws` | int64 | 100.0% | 0.0% | 19 | median=4 | 0 | 18 |
| 816 | `home_league_losses` | int64 | 100.0% | 0.0% | 29 | median=6 | 0 | 28 |
| 817 | `away_league_pos` | int64 | 100.0% | 0.0% | 20 | median=11 | 1 | 20 |
| 818 | `away_league_points` | int64 | 100.0% | 0.0% | 94 | median=23 | 0 | 96 |
| 819 | `away_league_gd` | int64 | 100.0% | 0.0% | 113 | median=0 | -52 | 67 |
| 820 | `away_league_goals_for` | int64 | 100.0% | 0.0% | 92 | median=22 | 0 | 96 |
| 821 | `away_league_wins` | int64 | 100.0% | 0.0% | 31 | median=6 | 0 | 31 |
| 822 | `away_league_draws` | int64 | 100.0% | 0.0% | 20 | median=4 | 0 | 19 |
| 823 | `away_league_losses` | int64 | 100.0% | 0.0% | 29 | median=6 | 0 | 28 |
| 824 | `home_in_relegation_zone` | float64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 825 | `home_in_cl_zone` | float64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 826 | `home_in_el_zone` | float64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 827 | `home_in_title_race` | float64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 828 | `away_in_relegation_zone` | float64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 829 | `away_in_cl_zone` | float64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 830 | `away_in_el_zone` | float64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 831 | `away_in_title_race` | float64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 832 | `league_position_diff` | int64 | 100.0% | 0.0% | 38 | median=1 | -19 | 19 |
| 833 | `home_position_momentum` | float64 | 97.2% | 2.8% | 39 | median=0 | -19 | 19 |
| 834 | `away_position_momentum` | float64 | 97.2% | 2.8% | 34 | median=0 | -17 | 16 |
| 835 | `position_momentum_diff` | float64 | 100.0% | 0.0% | 53 | median=0 | -28 | 28 |
| 836 | `home_position_zone` | int64 | 100.0% | 0.0% | 5 | median=4 | 1 | 5 |
| 837 | `away_position_zone` | int64 | 100.0% | 0.0% | 5 | median=4 | 1 | 5 |
| 838 | `home_points_to_cl_zone` | float64 | 100.0% | 0.0% | 615 | median=-6.2 | -192 | 48 |
| 839 | `home_points_to_relegation` | float64 | 100.0% | 0.0% | 655 | median=6 | -24 | 272 |
| 840 | `away_points_to_cl_zone` | float64 | 100.0% | 0.0% | 607 | median=-6 | -180 | 42 |
| 841 | `away_points_to_relegation` | float64 | 100.0% | 0.0% | 657 | median=6.4 | -20 | 256 |
| 842 | `home_manager_tenure` | float64 | 44.5% | 55.5% | 255 | median=25 | 1 | 342 |
| 843 | `home_manager_is_new` | float64 | 44.5% | 55.5% | 2 | median=0 | 0 | 1 |
| 844 | `home_manager_changed` | float64 | 44.3% | 55.7% | 2 | median=0 | 0 | 1 |
| 845 | `away_manager_tenure` | float64 | 44.5% | 55.5% | 257 | median=25 | 1 | 341 |
| 846 | `away_manager_is_new` | float64 | 44.5% | 55.5% | 2 | median=0 | 0 | 1 |
| 847 | `away_manager_changed` | float64 | 44.3% | 55.7% | 2 | median=0 | 0 | 1 |
| 848 | `manager_h2h_home_winrate` | float64 | 100.0% | 0.0% | 5 | median=0.5 | 0 | 1 |
| 849 | `manager_h2h_matches` | int64 | 100.0% | 0.0% | 4 | median=0 | 0 | 3 |
| 850 | `manager_h2h_confidence` | float64 | 100.0% | 0.0% | 3 | median=0 | 0 | 0.5 |
| 851 | `home_short_rest` | float64 | 97.4% | 2.6% | 2 | median=0 | 0 | 1 |
| 852 | `away_short_rest` | float64 | 97.3% | 2.7% | 2 | median=0 | 0 | 1 |
| 853 | `home_congestion_3` | int64 | 100.0% | 0.0% | 6 | median=2 | 0 | 5 |
| 854 | `away_congestion_3` | int64 | 100.0% | 0.0% | 5 | median=2 | 0 | 4 |
| 855 | `home_congestion_5` | int64 | 100.0% | 0.0% | 10 | median=4 | 0 | 9 |
| 856 | `away_congestion_5` | int64 | 100.0% | 0.0% | 10 | median=4 | 0 | 9 |
| 857 | `rest_advantage` | float64 | 100.0% | 0.0% | 11 | median=0 | -5 | 5 |
| 858 | `home_suspended_count` | int64 | 100.0% | 0.0% | 8 | median=0 | 0 | 7 |
| 859 | `away_suspended_count` | int64 | 100.0% | 0.0% | 8 | median=0 | 0 | 7 |
| 860 | `home_at_risk_count` | int64 | 100.0% | 0.0% | 9 | median=0 | 0 | 8 |
| 861 | `away_at_risk_count` | int64 | 100.0% | 0.0% | 9 | median=0 | 0 | 8 |
| 862 | `home_total_yellows` | int64 | 100.0% | 0.0% | 110 | median=10 | 0 | 111 |
| 863 | `away_total_yellows` | int64 | 100.0% | 0.0% | 110 | median=10 | 0 | 110 |
| 864 | `home_formation_flexibility` | float64 | 100.0% | 0.0% | 20 | median=0.611 | 0.462 | 0.949 |
| 865 | `away_formation_flexibility` | float64 | 100.0% | 0.0% | 20 | median=0.611 | 0.462 | 0.949 |
| 866 | `formation_matchup_home_rate` | float64 | 100.0% | 0.0% | 18 | median=0.436 | 0 | 0.714 |
| 867 | `formation_matchup_draw_rate` | float64 | 100.0% | 0.0% | 20 | median=0.359 | 0.1 | 0.474 |
| 868 | `formation_total_advantage` | float64 | 100.0% | 0.0% | 21 | median=-0.014 | -0.45 | 0.264 |
| 869 | `formation_confidence` | object | 100.0% | 0.0% | 2 | 'high' | — | — |
| 870 | `formation_width_mismatch` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 871 | `home_ppg_pace` | float64 | 99.9% | 0.1% | 905 | median=1.18 | 0 | 3 |
| 872 | `away_ppg_pace` | float64 | 99.9% | 0.1% | 921 | median=1.21 | 0 | 3 |
| 873 | `form_diff` | float64 | 99.6% | 0.4% | 31 | median=0 | -15 | 15 |
| 874 | `momentum_diff` | float64 | 99.6% | 0.4% | 30 | median=0 | -16 | 17 |
| 875 | `attack_strength_diff` | float64 | 100.0% | 0.0% | 7696 | median=0 | -3.67 | 3.61 |
| 876 | `defense_strength_diff` | float64 | 100.0% | 0.0% | 7697 | median=0 | -2.76 | 3.97 |
| 877 | `rolling_gd_diff` | float64 | 99.6% | 0.4% | 157 | median=0 | -5.4 | 4.9 |
| 878 | `rolling_goals_diff` | float64 | 97.3% | 2.7% | 147 | median=0 | -4 | 4.5 |
| 879 | `elo_diff_log` | float64 | 100.0% | 0.0% | 4646 | median=-0.151 | -6.21 | 6.13 |
| 880 | `attack_strength_diff_log` | float64 | 100.0% | 0.0% | 5901 | median=0 | -1.54 | 1.53 |
| 881 | `home_adj_attack_5_sqrt` | float64 | 100.0% | 0.0% | 5445 | median=1.12 | 0 | 4.69 |
| 882 | `home_adj_attack_10_sqrt` | float64 | 100.0% | 0.0% | 5338 | median=1.12 | 0 | 4.58 |
| 883 | `away_adj_attack_5_sqrt` | float64 | 100.0% | 0.0% | 5474 | median=1.12 | 0 | 4.48 |
| 884 | `away_adj_attack_10_sqrt` | float64 | 100.0% | 0.0% | 5305 | median=1.13 | 0 | 4.85 |
| 885 | `home_form_points_5_sq` | float64 | 100.0% | 0.0% | 15 | median=0.218 | 0 | 1 |
| 886 | `away_form_points_5_sq` | float64 | 100.0% | 0.0% | 15 | median=0.218 | 0 | 1 |
| 887 | `home_elo_momentum` | float64 | 94.3% | 5.7% | 1918 | median=-1.7 | -276 | 177 |
| 888 | `away_elo_momentum` | float64 | 94.3% | 5.7% | 1946 | median=-1.4 | -261 | 166 |
| 889 | `elo_momentum_diff` | float64 | 100.0% | 0.0% | 2519 | median=0 | -259 | 294 |
| 890 | `elo_form_blend_diff` | float64 | 100.0% | 0.0% | 3724 | median=-0.4 | -404 | 369 |
| 891 | `elo_form_disagreement` | float64 | 100.0% | 0.0% | 1263 | median=-1.2 | -97.4 | 90.3 |
| 892 | `poisson_home_xg` | float64 | 100.0% | 0.0% | 2339 | median=1.26 | 0.1 | 5 |
| 893 | `poisson_away_xg` | float64 | 100.0% | 0.0% | 2387 | median=1.29 | 0.1 | 5 |
| 894 | `poisson_prob_H` | float64 | 100.0% | 0.0% | 4858 | median=0.361 | 0.0008 | 0.932 |
| 895 | `poisson_prob_D` | float64 | 100.0% | 0.0% | 2505 | median=0.242 | 0.0095 | 0.827 |
| 896 | `poisson_prob_A` | float64 | 100.0% | 0.0% | 4858 | median=0.371 | 0.0008 | 0.932 |
| 897 | `is_derby` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 898 | `derby_intensity` | int64 | 100.0% | 0.0% | 4 | median=0 | 0 | 3 |
| 899 | `derby_home_advantage_boost` | float64 | 100.0% | 0.0% | 3 | median=0 | -0.05 | 0 |
| 900 | `total_xg_expected` | float64 | 100.0% | 0.0% | 2661 | median=2.67 | 0.2 | 9.95 |
| 901 | `combined_goals_per_game` | float64 | 100.0% | 0.0% | 79 | median=1.3 | 0 | 4.17 |
| 902 | `combined_goals_conceded` | float64 | 100.0% | 0.0% | 74 | median=1.3 | 0 | 3.5 |
| 903 | `attack_mismatch_score` | float64 | 100.0% | 0.0% | 2171 | median=1 | 0 | 28.9 |
| 904 | `defense_mismatch_score` | float64 | 100.0% | 0.0% | 2210 | median=1 | 0 | 40.8 |
| 905 | `scoring_variance` | float64 | 100.0% | 0.0% | 472 | median=1 | 0 | 2.87 |
| 906 | `home_scoring_rate_5` | float64 | 100.0% | 0.0% | 41 | median=0.699 | 0 | 0.998 |
| 907 | `away_scoring_rate_5` | float64 | 100.0% | 0.0% | 44 | median=0.699 | 0 | 0.993 |
| 908 | `btts_probability_naive` | float64 | 100.0% | 0.0% | 286 | median=0.46 | 0 | 0.969 |
| 909 | `poisson_over_1_5` | float64 | 100.0% | 0.0% | 2623 | median=0.746 | 0.0175 | 1 |
| 910 | `poisson_over_2_5` | float64 | 100.0% | 0.0% | 2657 | median=0.499 | 0.0011 | 0.997 |
| 911 | `poisson_over_3_5` | float64 | 100.0% | 0.0% | 2650 | median=0.28 | 0.0001 | 0.989 |
| 912 | `poisson_btts` | float64 | 100.0% | 0.0% | 3960 | median=0.48 | 0.0091 | 0.986 |
| 913 | `clean_sheet_matchup` | float64 | 100.0% | 0.0% | 27 | median=0.3 | 0 | 1 |
| 914 | `league_home_win_rate` | float64 | 100.0% | 0.0% | 7535 | median=0.463 | 0.443 | 1 |
| 915 | `league_draw_rate` | float64 | 100.0% | 0.0% | 7617 | median=0.271 | 0 | 0.307 |
| 916 | `league_avg_goals` | float64 | 100.0% | 0.0% | 7731 | median=2.61 | 2.2 | 3 |
| 917 | `home_draw_tendency_10` | float64 | 99.4% | 0.6% | 28 | median=0.3 | 0 | 1 |
| 918 | `home_draw_tendency_5` | float64 | 99.4% | 0.6% | 11 | median=0.2 | 0 | 1 |
| 919 | `away_draw_tendency_10` | float64 | 99.4% | 0.6% | 26 | median=0.3 | 0 | 1 |
| 920 | `away_draw_tendency_5` | float64 | 99.4% | 0.6% | 11 | median=0.2 | 0 | 1 |
| 921 | `matchup_competitiveness` | float64 | 100.0% | 0.0% | 7913 | median=0.815 | 0.447 | 1 |
| 922 | `both_defenses_strong` | float64 | 100.0% | 0.0% | 7075 | median=0.843 | 0 | 2.32 |
| 923 | `both_attacks_weak` | float64 | 100.0% | 0.0% | 1927 | median=0 | 0 | 1 |
| 924 | `combined_draw_tendency` | float64 | 100.0% | 0.0% | 119 | median=0.25 | 0 | 1 |
| 925 | `defense_similarity` | float64 | 100.0% | 0.0% | 7695 | median=0.792 | 0.201 | 1 |
| 926 | `travel_distance_km` | float64 | 100.0% | 0.0% | 170 | median=138 | 0 | 1.01e+03 |
| 927 | `altitude_diff` | float64 | 100.0% | 0.0% | 217 | median=0 | -351 | 351 |
| 928 | `home_stadium_capacity` | float64 | 79.9% | 20.1% | 19 | median=3.66e+04 | 1.12e+04 | 7.59e+04 |
| 929 | `long_travel` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 930 | `altitude_advantage` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 931 | `weather_temperature_2m_max` | float64 | 86.6% | 13.4% | 359 | median=16 | -6.1 | 38.1 |
| 932 | `weather_temperature_2m_min` | float64 | 86.6% | 13.4% | 318 | median=8.2 | -18.4 | 25.4 |
| 933 | `weather_temperature_2m_mean` | float64 | 86.6% | 13.4% | 332 | median=11.9 | -11.3 | 30.3 |
| 934 | `weather_apparent_temperature_max` | float64 | 86.6% | 13.4% | 423 | median=14.6 | -9.7 | 40.2 |
| 935 | `weather_apparent_temperature_min` | float64 | 86.6% | 13.4% | 389 | median=6.1 | -22.6 | 28.8 |
| 936 | `weather_precipitation_sum` | float64 | 86.6% | 13.4% | 392 | median=0 | 0 | 100 |
| 937 | `weather_snowfall_sum` | float64 | 86.6% | 13.4% | 73 | median=0 | 0 | 17.3 |
| 938 | `weather_wind_speed_10m_max` | float64 | 86.6% | 13.4% | 353 | median=11.7 | 3.1 | 52.9 |
| 939 | `weather_wind_gusts_10m_max` | float64 | 86.6% | 13.4% | 214 | median=27 | 6.8 | 122 |
| 940 | `weather_wind_direction_10m_dominant` | float64 | 86.6% | 13.4% | 361 | median=139 | 0 | 360 |
| 941 | `weather_relative_humidity_2m_mean` | float64 | 86.6% | 13.4% | 68 | median=78 | 28 | 99 |
| 942 | `odds_PCAHH` | float64 | 31.2% | 68.8% | 56 | median=1.96 | 1.68 | 2.29 |
| 943 | `odds_PC_lt_2.5` | float64 | 31.6% | 68.4% | 211 | median=2.01 | 1.37 | 5.53 |
| 944 | `odds_PC_gt_2.5` | float64 | 31.6% | 68.4% | 174 | median=1.89 | 1.17 | 3.26 |
| 945 | `odds_PAHH` | float64 | 31.2% | 68.8% | 51 | median=1.96 | 1.74 | 3.63 |
| 946 | `odds_PCAHA` | float64 | 31.2% | 68.8% | 58 | median=1.96 | 1.7 | 2.33 |
| 947 | `odds_MaxC_lt_2.5` | float64 | 32.0% | 68.0% | 199 | median=2.07 | 1.4 | 5.61 |
| 948 | `odds_MaxC_gt_2.5` | float64 | 32.0% | 68.0% | 172 | median=1.95 | 1.19 | 3.27 |
| 949 | `odds_MaxAHH` | float64 | 32.0% | 68.0% | 51 | median=1.98 | 1.76 | 3.75 |
| 950 | `odds_AvgAHA` | float64 | 32.0% | 68.0% | 54 | median=1.92 | 1.29 | 5.37 |
| 951 | `odds_B365CAHH` | float64 | 32.0% | 68.0% | 50 | median=1.95 | 1.63 | 2.2 |
| 952 | `odds_MaxAHA` | float64 | 32.0% | 68.0% | 51 | median=1.98 | 1.33 | 22 |
| 953 | `odds_Max_gt_2.5` | float64 | 32.0% | 68.0% | 148 | median=1.91 | 1.29 | 3.04 |
| 954 | `odds_Max_lt_2.5` | float64 | 32.0% | 68.0% | 172 | median=2.05 | 1.45 | 4.1 |
| 955 | `odds_AvgC_gt_2.5` | float64 | 32.6% | 67.4% | 169 | median=1.86 | 1.16 | 3.05 |
| 956 | `odds_PAHA` | float64 | 31.2% | 68.8% | 50 | median=1.95 | 1.33 | 2.21 |
| 957 | `odds_MaxCD` | float64 | 32.6% | 67.4% | 350 | median=3.75 | 1.75 | 14.3 |
| 958 | `odds_AvgAHH` | float64 | 32.0% | 68.0% | 47 | median=1.92 | 1.72 | 3.61 |
| 959 | `odds_AvgC_lt_2.5` | float64 | 32.6% | 67.4% | 208 | median=1.96 | 1.36 | 5.04 |
| 960 | `odds_P_lt_2.5` | float64 | 31.6% | 68.4% | 191 | median=2.02 | 1.41 | 3.86 |
| 961 | `odds_P_gt_2.5` | float64 | 31.6% | 68.4% | 151 | median=1.88 | 1.28 | 3.04 |
| 962 | `odds_AvgCD` | float64 | 32.6% | 67.4% | 402 | median=3.59 | 1.72 | 11.6 |
| 963 | `odds_MaxCH` | float64 | 32.6% | 67.4% | 531 | median=2.43 | 1.12 | 29 |
| 964 | `odds_B365CAHA` | float64 | 32.0% | 68.0% | 51 | median=1.96 | 1.65 | 2.25 |
| 965 | `odds_AvgCH` | float64 | 32.6% | 67.4% | 572 | median=2.31 | 1.08 | 20.8 |
| 966 | `odds_AvgCA` | float64 | 32.6% | 67.4% | 822 | median=3.25 | 1.14 | 24.7 |
| 967 | `odds_MaxCA` | float64 | 32.6% | 67.4% | 622 | median=3.46 | 1.17 | 34 |
| 968 | `pinnacle_home_prob` | float64 | 66.0% | 34.0% | 5167 | median=0.43 | 0.0413 | 0.925 |
| 969 | `pinnacle_draw_prob` | float64 | 66.0% | 34.0% | 5137 | median=0.266 | 0.0516 | 0.443 |
| 970 | `pinnacle_away_prob` | float64 | 66.0% | 34.0% | 5173 | median=0.279 | 0.0239 | 0.842 |
| 971 | `pinnacle_overround` | float64 | 66.0% | 34.0% | 5103 | median=1.02 | 1.02 | 1.06 |
| 972 | `market_draw_prob` | float64 | 99.7% | 0.3% | 7844 | median=0.276 | 0.0661 | 0.663 |
| 973 | `market_overround` | float64 | 99.7% | 0.3% | 7820 | median=1.06 | 0.994 | 1.17 |
| 974 | `pinnacle_ou_over_prob` | float64 | 31.6% | 68.4% | 664 | median=0.518 | 0.317 | 0.751 |
| 975 | `pinnacle_ou_overround` | float64 | 31.6% | 68.4% | 455 | median=1.03 | 1.02 | 1.06 |
| 976 | `market_ou_over_prob` | float64 | 32.6% | 67.4% | 749 | median=0.516 | 0.33 | 0.75 |
| 977 | `market_ou_overround` | float64 | 32.6% | 67.4% | 520 | median=1.05 | 1.04 | 1.08 |
| 978 | `sharp_soft_home_div` | float64 | 66.0% | 34.0% | 454 | median=0.0022 | -0.0513 | 0.0333 |
| 979 | `sharp_soft_draw_div` | float64 | 66.0% | 34.0% | 332 | median=-0.0011 | -0.0235 | 0.0265 |
| 980 | `sharp_soft_away_div` | float64 | 66.0% | 34.0% | 437 | median=-0.0023 | -0.0327 | 0.0748 |
| 981 | `sharp_soft_ou_div` | float64 | 31.6% | 68.4% | 322 | median=-0.0001 | -0.0627 | 0.0355 |
| 982 | `ah_line_abs` | float64 | 32.0% | 68.0% | 11 | median=0.5 | 0 | 2.5 |
| 983 | `odds_home_fav` | float64 | 99.7% | 0.3% | 2 | median=1 | 0 | 1 |
| 984 | `odds_consistency` | float64 | 66.0% | 34.0% | 662 | median=0.986 | 0.67 | 1 |
| 985 | `ou_consistency` | float64 | 31.6% | 68.4% | 328 | median=0.992 | 0.909 | 1 |
| 986 | `line_vel_pin_home` | float64 | 66.0% | 34.0% | 1257 | median=-0.0025 | -0.124 | 0.144 |
| 987 | `line_vel_pin_draw` | float64 | 66.0% | 34.0% | 721 | median=0.0019 | -0.0674 | 0.0812 |
| 988 | `line_vel_pin_away` | float64 | 66.0% | 34.0% | 1219 | median=-0.0012 | -0.165 | 0.159 |
| 989 | `line_vel_mkt_home` | float64 | 32.6% | 67.4% | 968 | median=-0.0028 | -0.117 | 0.139 |
| 990 | `line_vel_mkt_draw` | float64 | 32.6% | 67.4% | 561 | median=0.0031 | -0.0761 | 0.276 |
| 991 | `line_vel_mkt_away` | float64 | 32.6% | 67.4% | 945 | median=-0.0018 | -0.184 | 0.173 |
| 992 | `line_vel_ou_over` | float64 | 31.5% | 68.5% | 999 | median=-0.0039 | -0.131 | 0.11 |
| 993 | `line_vel_ou_under` | float64 | 31.5% | 68.5% | 1007 | median=0.0023 | -0.109 | 0.128 |
| 994 | `ah_line_movement` | float64 | 32.0% | 68.0% | 8 | median=0 | -0.75 | 1.5 |
| 995 | `steam_move_flag` | float64 | 66.0% | 34.0% | 2 | median=0 | 0 | 1 |
| 996 | `home_squad_total_value` | float64 | 41.1% | 58.9% | 174 | median=1.99e+08 | 1e+05 | 8.71e+08 |
| 997 | `home_avg_player_value` | float64 | 41.1% | 58.9% | 174 | median=4.88e+06 | 5.56e+03 | 2.78e+07 |
| 998 | `home_median_player_value` | float64 | 41.1% | 58.9% | 70 | median=2.5e+06 | 0 | 2.4e+07 |
| 999 | `home_max_player_value` | float64 | 41.1% | 58.9% | 53 | median=3e+07 | 1e+05 | 1.2e+08 |
| 1000 | `home_squad_size` | float64 | 41.1% | 58.9% | 41 | median=41 | 18 | 88 |
| 1001 | `away_squad_total_value` | float64 | 41.1% | 58.9% | 174 | median=1.99e+08 | 1e+05 | 8.71e+08 |
| 1002 | `away_avg_player_value` | float64 | 41.1% | 58.9% | 174 | median=4.88e+06 | 5.56e+03 | 2.78e+07 |
| 1003 | `away_median_player_value` | float64 | 41.1% | 58.9% | 70 | median=2.5e+06 | 0 | 2.4e+07 |
| 1004 | `away_max_player_value` | float64 | 41.1% | 58.9% | 53 | median=3e+07 | 1e+05 | 1.2e+08 |
| 1005 | `away_squad_size` | float64 | 41.1% | 58.9% | 41 | median=41 | 18 | 88 |
| 1006 | `squad_value_ratio` | float64 | 39.8% | 60.2% | 3155 | median=1 | 0.000151 | 6.64e+03 |
| 1007 | `home_transfer_spend` | float64 | 41.1% | 58.9% | 165 | median=4.58e+07 | 0 | 2.65e+08 |
| 1008 | `home_transfers_in` | float64 | 41.1% | 58.9% | 24 | median=13 | 3 | 29 |
| 1009 | `home_transfer_income` | float64 | 41.1% | 58.9% | 168 | median=4.27e+07 | 0 | 2.08e+08 |
| 1010 | `home_transfers_out` | float64 | 41.1% | 58.9% | 28 | median=15 | 1 | 38 |
| 1011 | `away_transfer_spend` | float64 | 41.1% | 58.9% | 165 | median=4.58e+07 | 0 | 2.65e+08 |
| 1012 | `away_transfers_in` | float64 | 41.1% | 58.9% | 24 | median=13 | 3 | 29 |
| 1013 | `away_transfer_income` | float64 | 41.1% | 58.9% | 168 | median=4.27e+07 | 0 | 2.08e+08 |
| 1014 | `away_transfers_out` | float64 | 41.1% | 58.9% | 28 | median=15 | 1 | 38 |
| 1015 | `home_net_spend` | float64 | 42.5% | 57.5% | 171 | median=4.26e+06 | -1.66e+08 | 1.68e+08 |
| 1016 | `away_net_spend` | float64 | 42.5% | 57.5% | 171 | median=4.26e+06 | -1.66e+08 | 1.68e+08 |
| 1017 | `home_jan_arrivals` | int64 | 100.0% | 0.0% | 49 | median=0 | 0 | 79 |
| 1018 | `home_squad_disruption` | float64 | 100.0% | 0.0% | 34 | median=0 | 0 | 1 |
| 1019 | `home_signing_integration` | float64 | 100.0% | 0.0% | 2 | median=1 | 0.3 | 1 |
| 1020 | `home_squad_avg_age` | float64 | 41.1% | 58.9% | 64 | median=25.5 | 0 | 33.2 |
| 1021 | `away_jan_arrivals` | int64 | 100.0% | 0.0% | 49 | median=0 | 0 | 79 |
| 1022 | `away_squad_disruption` | float64 | 100.0% | 0.0% | 34 | median=0 | 0 | 1 |
| 1023 | `away_signing_integration` | float64 | 100.0% | 0.0% | 2 | median=1 | 0.3 | 1 |
| 1024 | `away_squad_avg_age` | float64 | 41.1% | 58.9% | 64 | median=25.5 | 0 | 33.2 |
| 1025 | `squad_value_diff` | float64 | 39.8% | 60.2% | 3063 | median=2.25e+05 | -8.67e+08 | 8.67e+08 |
| 1026 | `elo_x_form` | float64 | 100.0% | 0.0% | 6247 | median=3.07 | -41.1 | 117 |
| 1027 | `attack_defense_mismatch` | float64 | 100.0% | 0.0% | 2497 | median=0 | -39.1 | 28.3 |
| 1028 | `rest_x_close_elo` | float64 | 100.0% | 0.0% | 20 | median=0 | -18 | 8 |
| 1029 | `defensive_form_diff` | float64 | 100.0% | 0.0% | 75 | median=0 | -5 | 4 |
| 1030 | `elo_xg_signal` | Float64 | 100.0% | 0.0% | 3963 | median=0 | -63.3 | 364 |
| 1031 | `attack_form_alignment` | float64 | 100.0% | 0.0% | 1233 | median=0.043 | -1.41 | 4.32 |
| 1032 | `h2h_competitiveness_signal` | float64 | 100.0% | 0.0% | 865 | median=-0.111 | -0.5 | 0.499 |
| 1033 | `form_elo_signal` | float64 | 100.0% | 0.0% | 5789 | median=1.7 | -14.6 | 58.3 |
| 1034 | `sharp_soft_x_elo` | float64 | 100.0% | 0.0% | 4619 | median=0 | -4.39 | 9.66 |
| 1035 | `market_elo_disagreement` | float64 | 100.0% | 0.0% | 2696 | median=-0.0572 | -0.441 | 0.46 |
| 1036 | `ah_x_form` | float64 | 100.0% | 0.0% | 118 | median=0 | -0.63 | 0.3 |
| 1037 | `draw_convergence_x_competitiveness` | float64 | 100.0% | 0.0% | 3215 | median=0.203 | 0 | 0.857 |
| 1038 | `is_early_kickoff` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 1039 | `is_evening_kickoff` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 1040 | `is_friday_night` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 1041 | `is_monday_night` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 1042 | `is_early_season` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 1043 | `is_mid_season` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 1044 | `is_late_season` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 1045 | `is_run_in` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 1046 | `is_august` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 1047 | `is_december` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 1048 | `is_january` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 1049 | `is_may` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 1050 | `home_travel_fatigue` | float64 | 100.0% | 0.0% | 1229 | median=19.3 | 0 | 337 |
| 1051 | `away_travel_fatigue` | float64 | 100.0% | 0.0% | 1209 | median=19.3 | 0 | 337 |
| 1052 | `home_avg_subs_per_game` | float64 | 42.2% | 57.8% | 51 | median=2.9 | 1 | 5 |
| 1053 | `home_avg_sub_minute` | float64 | 42.2% | 57.8% | 193 | median=69.5 | 45.3 | 81.7 |
| 1054 | `home_avg_first_sub_minute` | float64 | 42.2% | 57.8% | 289 | median=56.4 | 21 | 76.2 |
| 1055 | `home_sub_games_tracked` | float64 | 42.2% | 57.8% | 10 | median=10 | 1 | 10 |
| 1056 | `away_avg_subs_per_game` | float64 | 42.1% | 57.9% | 48 | median=2.9 | 1.9 | 4.5 |
| 1057 | `away_avg_sub_minute` | float64 | 42.1% | 57.9% | 197 | median=69.5 | 47 | 81.3 |
| 1058 | `away_avg_first_sub_minute` | float64 | 42.1% | 57.9% | 300 | median=56.3 | 20 | 75.7 |
| 1059 | `away_sub_games_tracked` | float64 | 42.1% | 57.9% | 10 | median=10 | 1 | 10 |

---

## 7. FEATURES — combined (all leagues)

### `data/features/features.parquet`

_Same schema, Serie A + EPL combined._

- **Format:** Parquet  
- **Size:** 20.1MB  
- **Modified:** 2026-04-20 23:56  
- **Rows:** 15,839  
- **Columns:** 1080  
- **Date column:** `match_date` — range 2005-08-13 → 2026-04-20  
- **League distribution:** {'serie_a': 7930, 'premier_league': 7909} (7930/15839 = 50.1% Serie A)  
- **Seasons:** 21 covered — 2005-2006 → 2025-2026  

**Columns (1080):**

| # | Column | Dtype | Filled % | NaN % | Unique | Sample | Min | Max |
|---|--------|-------|----------|-------|--------|--------|-----|-----|
| 1 | `home_team` | object | 100.0% | 0.0% | 89 | 'Everton' | — | — |
| 2 | `away_team` | object | 100.0% | 0.0% | 89 | 'Man United' | — | — |
| 3 | `match_date` | datetime64[ns] | 100.0% | 0.0% | 2828 | — | 2005-08-13 | 2026-04-20 |
| 4 | `home_score` | float64 | 100.0% | 0.0% | 10 | median=1 | 0 | 9 |
| 5 | `away_score` | float64 | 100.0% | 0.0% | 10 | median=1 | 0 | 9 |
| 6 | `result` | object | 100.0% | 0.0% | 3 | 'A' | — | — |
| 7 | `season` | object | 100.0% | 0.0% | 21 | '2005-2006' | — | — |
| 8 | `league` | object | 100.0% | 0.0% | 2 | 'premier_league' | — | — |
| 9 | `home_shots_total` | float64 | 99.2% | 0.8% | 43 | median=13 | 0 | 46 |
| 10 | `away_shots_total` | float64 | 99.2% | 0.8% | 34 | median=11 | 0 | 37 |
| 11 | `home_shots_on_target_count` | float64 | 99.6% | 0.4% | 24 | median=5 | 0 | 24 |
| 12 | `away_shots_on_target_count` | float64 | 99.6% | 0.4% | 21 | median=4 | 0 | 20 |
| 13 | `home_fouls` | float64 | 99.7% | 0.3% | 40 | median=12 | 0 | 48 |
| 14 | `away_fouls` | float64 | 99.7% | 0.3% | 39 | median=13 | 1 | 41 |
| 15 | `home_corners` | float64 | 99.7% | 0.3% | 22 | median=5 | 0 | 21 |
| 16 | `away_corners` | float64 | 99.7% | 0.3% | 20 | median=4 | 0 | 19 |
| 17 | `home_yellow_cards` | float64 | 99.7% | 0.3% | 8 | median=2 | 0 | 7 |
| 18 | `away_yellow_cards` | float64 | 99.7% | 0.3% | 10 | median=2 | 0 | 9 |
| 19 | `home_red_cards` | float64 | 99.7% | 0.3% | 4 | median=0 | 0 | 3 |
| 20 | `away_red_cards` | float64 | 99.7% | 0.3% | 4 | median=0 | 0 | 3 |
| 21 | `home_ht_goals` | float64 | 99.3% | 0.7% | 6 | median=0 | 0 | 5 |
| 22 | `away_ht_goals` | float64 | 99.3% | 0.7% | 6 | median=0 | 0 | 5 |
| 23 | `ht_result` | object | 99.3% | 0.7% | 3 | 'A' | — | — |
| 24 | `referee` | object | 76.0% | 24.0% | 212 | 'G Poll' | — | — |
| 25 | `odds_B365H` | float64 | 99.3% | 0.7% | 141 | median=2.2 | 1.06 | 23 |
| 26 | `odds_B365D` | float64 | 99.3% | 0.7% | 83 | median=3.5 | 1.4 | 17 |
| 27 | `odds_B365A` | float64 | 99.3% | 0.7% | 136 | median=3.5 | 1.1 | 41 |
| 28 | `odds_AvgH` | float64 | 99.8% | 0.2% | 938 | median=2.18 | 1.05 | 20.8 |
| 29 | `odds_AvgD` | float64 | 99.8% | 0.2% | 702 | median=3.49 | 1.29 | 14.4 |
| 30 | `odds_AvgA` | float64 | 99.8% | 0.2% | 1587 | median=3.42 | 1.1 | 40.6 |
| 31 | `odds_MaxH` | float64 | 99.8% | 0.2% | 710 | median=2.28 | 1.06 | 35 |
| 32 | `odds_MaxD` | float64 | 99.8% | 0.2% | 603 | median=3.7 | 1.33 | 19 |
| 33 | `odds_MaxA` | float64 | 99.8% | 0.2% | 1140 | median=3.7 | 1.13 | 61 |
| 34 | `odds_Avg_over25` | float64 | 99.3% | 0.7% | 163 | median=1.91 | 1.18 | 2.91 |
| 35 | `odds_Avg_under25` | float64 | 99.3% | 0.7% | 256 | median=1.88 | 1.4 | 4.67 |
| 36 | `match_id` | object | 100.0% | 0.0% | 15839 | '2005-08-13_Everton_Man United | — | — |
| 37 | `odds_PSD` | float64 | 66.2% | 33.8% | 663 | median=3.73 | 2.2 | 19 |
| 38 | `odds_PSA` | float64 | 66.2% | 33.8% | 1510 | median=3.47 | 1.13 | 42.9 |
| 39 | `odds_PS_close_H` | float64 | 64.9% | 35.1% | 903 | median=2.27 | 1.04 | 25.8 |
| 40 | `odds_PS_close_D` | float64 | 64.9% | 35.1% | 635 | median=3.7 | 2.2 | 18.5 |
| 41 | `odds_PS_close_A` | float64 | 64.9% | 35.1% | 1405 | median=3.49 | 1.09 | 50 |
| 42 | `odds_B365_over25` | float64 | 32.1% | 67.9% | 75 | median=1.8 | 1.2 | 3 |
| 43 | `odds_B365_under25` | float64 | 32.1% | 67.9% | 76 | median=2.03 | 1.4 | 4.5 |
| 44 | `odds_B365_close_H` | float64 | 32.1% | 67.9% | 119 | median=2.25 | 1.03 | 23 |
| 45 | `odds_B365_close_D` | float64 | 32.1% | 67.9% | 47 | median=3.75 | 2.63 | 19 |
| 46 | `odds_B365_close_A` | float64 | 32.1% | 67.9% | 115 | median=3.2 | 1.08 | 41 |
| 47 | `odds_B365_close_over25` | float64 | 32.1% | 67.9% | 83 | median=1.8 | 1.12 | 3.2 |
| 48 | `odds_B365_close_under25` | float64 | 32.1% | 67.9% | 82 | median=2 | 1.36 | 6 |
| 49 | `odds_B365_AH_H` | float64 | 32.1% | 67.9% | 48 | median=1.95 | 1.7 | 3.55 |
| 50 | `odds_B365_AH_A` | float64 | 32.1% | 67.9% | 50 | median=1.95 | 1.27 | 2.17 |
| 51 | `odds_AH_line` | float64 | 28.5% | 71.5% | 22 | median=-0.25 | -3 | 2.5 |
| 52 | `odds_AH_close_line` | float64 | 28.9% | 71.1% | 25 | median=-0.25 | -3.75 | 3 |
| 53 | `matchweek` | float64 | 100.0% | 0.0% | 38 | median=19 | 1 | 38 |
| 54 | `home_formation` | object | 50.1% | 49.9% | 5 | '4-2-3-1' | — | — |
| 55 | `away_formation` | object | 50.1% | 49.9% | 5 | '4-2-3-1' | — | — |
| 56 | `home_manager` | object | 43.2% | 56.8% | 161 | 'Roberto Donadoni' | — | — |
| 57 | `away_manager` | object | 43.2% | 56.8% | 161 | 'Roberto Donadoni' | — | — |
| 58 | `home_roll_3_goals_scored` | float64 | 97.4% | 2.6% | 22 | median=1.33 | 0 | 6 |
| 59 | `home_roll_3_goals_conceded` | float64 | 97.4% | 2.6% | 23 | median=1.33 | 0 | 6 |
| 60 | `home_roll_3_shots_on_target` | float64 | 97.3% | 2.7% | 58 | median=4.33 | 0 | 15 |
| 61 | `home_roll_3_corners` | float64 | 97.3% | 2.7% | 53 | median=5 | 0 | 16 |
| 62 | `home_roll_3_fouls` | float64 | 97.3% | 2.7% | 107 | median=12.7 | 2 | 33.3 |
| 63 | `home_roll_3_yellow_cards` | float64 | 97.3% | 2.7% | 26 | median=2 | 0 | 7 |
| 64 | `home_roll_3_red_cards` | float64 | 97.3% | 2.7% | 9 | median=0 | 0 | 2 |
| 65 | `home_roll_3_points` | float64 | 97.4% | 2.6% | 11 | median=1.33 | 0 | 3 |
| 66 | `home_roll_3_clean_sheet` | float64 | 97.4% | 2.6% | 5 | median=0.333 | 0 | 1 |
| 67 | `home_roll_3_win_rate` | float64 | 97.4% | 2.6% | 5 | median=0.333 | 0 | 1 |
| 68 | `home_roll_5_goals_scored` | float64 | 97.4% | 2.6% | 46 | median=1.2 | 0 | 6 |
| 69 | `home_roll_5_goals_conceded` | float64 | 97.4% | 2.6% | 44 | median=1.4 | 0 | 6 |
| 70 | `home_roll_5_shots_on_target` | float64 | 97.4% | 2.6% | 120 | median=4.6 | 0 | 15 |
| 71 | `home_roll_5_corners` | float64 | 97.4% | 2.6% | 107 | median=5 | 0 | 16 |
| 72 | `home_roll_5_fouls` | float64 | 97.4% | 2.6% | 211 | median=12.6 | 2 | 32 |
| 73 | `home_roll_5_yellow_cards` | float64 | 97.4% | 2.6% | 52 | median=2 | 0 | 7 |
| 74 | `home_roll_5_red_cards` | float64 | 97.4% | 2.6% | 14 | median=0 | 0 | 2 |
| 75 | `home_roll_5_points` | float64 | 97.4% | 2.6% | 28 | median=1.4 | 0 | 3 |
| 76 | `home_roll_5_clean_sheet` | float64 | 97.4% | 2.6% | 11 | median=0.2 | 0 | 1 |
| 77 | `home_roll_5_win_rate` | float64 | 97.4% | 2.6% | 11 | median=0.4 | 0 | 1 |
| 78 | `home_roll_10_goals_scored` | float64 | 97.4% | 2.6% | 116 | median=1.29 | 0 | 6 |
| 79 | `home_roll_10_goals_conceded` | float64 | 97.4% | 2.6% | 105 | median=1.3 | 0 | 6 |
| 80 | `home_roll_10_shots_on_target` | float64 | 97.4% | 2.6% | 303 | median=4.5 | 0 | 15 |
| 81 | `home_roll_10_corners` | float64 | 97.4% | 2.6% | 257 | median=5.1 | 0 | 16 |
| 82 | `home_roll_10_fouls` | float64 | 97.4% | 2.6% | 525 | median=12.6 | 2 | 32 |
| 83 | `home_roll_10_yellow_cards` | float64 | 97.4% | 2.6% | 123 | median=2 | 0 | 7 |
| 84 | `home_roll_10_red_cards` | float64 | 97.4% | 2.6% | 27 | median=0.1 | 0 | 2 |
| 85 | `home_roll_10_points` | float64 | 97.4% | 2.6% | 92 | median=1.3 | 0 | 3 |
| 86 | `home_roll_10_clean_sheet` | float64 | 97.4% | 2.6% | 32 | median=0.3 | 0 | 1 |
| 87 | `home_roll_10_win_rate` | float64 | 97.4% | 2.6% | 33 | median=0.333 | 0 | 1 |
| 88 | `home_roll_5_goals_scored_std` | float64 | 92.0% | 8.0% | 409 | median=1 | 0 | 3.86 |
| 89 | `home_roll_5_goals_conceded_std` | float64 | 92.0% | 8.0% | 378 | median=1.1 | 0 | 4.16 |
| 90 | `home_roll_5_shots_on_target_std` | float64 | 92.0% | 8.0% | 1488 | median=2.17 | 0 | 8.04 |
| 91 | `home_roll_5_points_std` | float64 | 92.0% | 8.0% | 156 | median=1.3 | 0 | 1.73 |
| 92 | `home_roll_10_goals_scored_std` | float64 | 92.0% | 8.0% | 1197 | median=1.06 | 0 | 3.86 |
| 93 | `home_roll_10_goals_conceded_std` | float64 | 92.0% | 8.0% | 1200 | median=1.08 | 0 | 4.16 |
| 94 | `home_roll_10_shots_on_target_std` | float64 | 92.0% | 8.0% | 3817 | median=2.25 | 0 | 7.55 |
| 95 | `home_roll_10_points_std` | float64 | 92.0% | 8.0% | 459 | median=1.26 | 0 | 1.73 |
| 96 | `home_goals_scored_trend` | float64 | 97.4% | 2.6% | 440 | median=0 | -2.21 | 2.33 |
| 97 | `home_goals_conceded_trend` | float64 | 97.4% | 2.6% | 423 | median=0 | -1.87 | 2.57 |
| 98 | `home_shots_on_target_trend` | float64 | 97.3% | 2.7% | 949 | median=0 | -5.57 | 5.37 |
| 99 | `home_points_trend` | float64 | 97.4% | 2.6% | 354 | median=0 | -1.9 | 2 |
| 100 | `home_venue_roll_3_goals_scored` | float64 | 99.4% | 0.6% | 22 | median=1.33 | 0 | 7 |
| 101 | `home_venue_roll_3_goals_conceded` | float64 | 99.4% | 0.6% | 21 | median=1 | 0 | 5.5 |
| 102 | `home_venue_roll_3_points` | float64 | 99.4% | 0.6% | 11 | median=1.67 | 0 | 3 |
| 103 | `home_venue_roll_3_clean_sheet` | float64 | 99.4% | 0.6% | 5 | median=0.333 | 0 | 1 |
| 104 | `home_venue_roll_5_goals_scored` | float64 | 99.4% | 0.6% | 43 | median=1.4 | 0 | 5.8 |
| 105 | `home_venue_roll_5_goals_conceded` | float64 | 99.4% | 0.6% | 41 | median=1.2 | 0 | 5.5 |
| 106 | `home_venue_roll_5_points` | float64 | 99.4% | 0.6% | 28 | median=1.6 | 0 | 3 |
| 107 | `home_venue_roll_5_clean_sheet` | float64 | 99.4% | 0.6% | 11 | median=0.4 | 0 | 1 |
| 108 | `home_venue_roll_10_goals_scored` | float64 | 99.4% | 0.6% | 101 | median=1.4 | 0 | 4.7 |
| 109 | `home_venue_roll_10_goals_conceded` | float64 | 99.4% | 0.6% | 92 | median=1.1 | 0 | 5.5 |
| 110 | `home_venue_roll_10_points` | float64 | 99.4% | 0.6% | 88 | median=1.6 | 0 | 3 |
| 111 | `home_venue_roll_10_clean_sheet` | float64 | 99.4% | 0.6% | 33 | median=0.3 | 0 | 1 |
| 112 | `home_attack_strength` | float64 | 100.0% | 0.0% | 14098 | median=0.947 | 0 | 4.72 |
| 113 | `home_defense_strength` | float64 | 100.0% | 0.0% | 14028 | median=1 | 0 | 5.74 |
| 114 | `home_xg_attack_strength` | float64 | 50.1% | 49.9% | 58 | median=1 | 0.168 | 2.64 |
| 115 | `home_xg_defense_strength` | float64 | 50.1% | 49.9% | 58 | median=1 | 0.271 | 2.91 |
| 116 | `home_rest_days` | float64 | 97.4% | 2.6% | 20 | median=7 | 2 | 21 |
| 117 | `home_is_congested` | bool | 100.0% | 0.0% | 2 | np.False_ | — | — |
| 118 | `home_win_streak` | float64 | 99.7% | 0.3% | 18 | median=0 | 0 | 17 |
| 119 | `home_unbeaten_run` | float64 | 99.7% | 0.3% | 45 | median=1 | 0 | 49 |
| 120 | `home_winless_run` | float64 | 99.7% | 0.3% | 30 | median=1 | 0 | 31 |
| 121 | `home_loss_streak` | float64 | 99.7% | 0.3% | 14 | median=0 | 0 | 14 |
| 122 | `home_scoring_streak` | float64 | 99.7% | 0.3% | 45 | median=2 | 0 | 44 |
| 123 | `home_clean_sheet_streak` | float64 | 99.7% | 0.3% | 14 | median=0 | 0 | 14 |
| 124 | `home_form_points_5` | float64 | 99.7% | 0.3% | 15 | median=7 | 0 | 15 |
| 125 | `home_adj_attack_5` | float64 | 97.4% | 2.6% | 14181 | median=1.25 | 0 | 22 |
| 126 | `home_adj_defense_5` | float64 | 97.4% | 2.6% | 14171 | median=1.38 | 0 | 16.1 |
| 127 | `home_adj_attack_10` | float64 | 97.4% | 2.6% | 14225 | median=1.27 | 0 | 20.9 |
| 128 | `home_adj_defense_10` | float64 | 97.4% | 2.6% | 14252 | median=1.38 | 0 | 15.2 |
| 129 | `home_opp_difficulty_roll_5` | float64 | 99.7% | 0.3% | 15732 | median=0.969 | 0.327 | 2.3 |
| 130 | `home_form_overperformance` | float64 | 99.7% | 0.3% | 14786 | median=5.95 | -1.72 | 15 |
| 131 | `home_gd_roll_3` | float64 | 99.7% | 0.3% | 37 | median=0 | -5.33 | 6.67 |
| 132 | `home_gd_roll_5` | float64 | 99.7% | 0.3% | 66 | median=0 | -4 | 5.8 |
| 133 | `home_gd_per_match` | float64 | 97.4% | 2.6% | 1465 | median=-0.0909 | -6 | 6 |
| 134 | `home_momentum_gradient` | float64 | 98.6% | 1.4% | 119 | median=-8.11e-17 | -0.9 | 0.9 |
| 135 | `home_ewma_form` | float64 | 99.2% | 0.8% | 15658 | median=1.28 | 0 | 3 |
| 136 | `home_last3_vs_prev3` | float64 | 98.6% | 1.4% | 19 | median=0 | -9 | 9 |
| 137 | `away_roll_3_goals_scored` | float64 | 97.3% | 2.7% | 24 | median=1.33 | 0 | 6 |
| 138 | `away_roll_3_goals_conceded` | float64 | 97.3% | 2.7% | 22 | median=1.33 | 0 | 6 |
| 139 | `away_roll_3_shots_on_target` | float64 | 97.3% | 2.7% | 66 | median=4.67 | 0 | 19 |
| 140 | `away_roll_3_corners` | float64 | 97.3% | 2.7% | 56 | median=5 | 0 | 18 |
| 141 | `away_roll_3_fouls` | float64 | 97.3% | 2.7% | 107 | median=12.7 | 3 | 36 |
| 142 | `away_roll_3_yellow_cards` | float64 | 97.3% | 2.7% | 25 | median=2 | 0 | 7 |
| 143 | `away_roll_3_red_cards` | float64 | 97.3% | 2.7% | 8 | median=0 | 0 | 2 |
| 144 | `away_roll_3_points` | float64 | 97.3% | 2.7% | 11 | median=1.33 | 0 | 3 |
| 145 | `away_roll_3_clean_sheet` | float64 | 97.3% | 2.7% | 5 | median=0.333 | 0 | 1 |
| 146 | `away_roll_3_win_rate` | float64 | 97.3% | 2.7% | 5 | median=0.333 | 0 | 1 |
| 147 | `away_roll_5_goals_scored` | float64 | 97.3% | 2.7% | 49 | median=1.2 | 0 | 6 |
| 148 | `away_roll_5_goals_conceded` | float64 | 97.3% | 2.7% | 45 | median=1.2 | 0 | 6 |
| 149 | `away_roll_5_shots_on_target` | float64 | 97.3% | 2.7% | 129 | median=4.6 | 0 | 19 |
| 150 | `away_roll_5_corners` | float64 | 97.3% | 2.7% | 110 | median=5.2 | 0 | 18 |
| 151 | `away_roll_5_fouls` | float64 | 97.3% | 2.7% | 217 | median=12.6 | 3 | 36 |
| 152 | `away_roll_5_yellow_cards` | float64 | 97.3% | 2.7% | 53 | median=2 | 0 | 7 |
| 153 | `away_roll_5_red_cards` | float64 | 97.3% | 2.7% | 13 | median=0 | 0 | 2 |
| 154 | `away_roll_5_points` | float64 | 97.3% | 2.7% | 28 | median=1.4 | 0 | 3 |
| 155 | `away_roll_5_clean_sheet` | float64 | 97.3% | 2.7% | 11 | median=0.2 | 0 | 1 |
| 156 | `away_roll_5_win_rate` | float64 | 97.3% | 2.7% | 11 | median=0.4 | 0 | 1 |
| 157 | `away_roll_10_goals_scored` | float64 | 97.3% | 2.7% | 121 | median=1.3 | 0 | 6 |
| 158 | `away_roll_10_goals_conceded` | float64 | 97.3% | 2.7% | 107 | median=1.3 | 0 | 6 |
| 159 | `away_roll_10_shots_on_target` | float64 | 97.3% | 2.7% | 306 | median=4.6 | 0 | 19 |
| 160 | `away_roll_10_corners` | float64 | 97.3% | 2.7% | 257 | median=5.11 | 0 | 18 |
| 161 | `away_roll_10_fouls` | float64 | 97.3% | 2.7% | 533 | median=12.5 | 3 | 36 |
| 162 | `away_roll_10_yellow_cards` | float64 | 97.3% | 2.7% | 121 | median=1.9 | 0 | 7 |
| 163 | `away_roll_10_red_cards` | float64 | 97.3% | 2.7% | 27 | median=0.1 | 0 | 2 |
| 164 | `away_roll_10_points` | float64 | 97.3% | 2.7% | 92 | median=1.33 | 0 | 3 |
| 165 | `away_roll_10_clean_sheet` | float64 | 97.3% | 2.7% | 32 | median=0.3 | 0 | 1 |
| 166 | `away_roll_10_win_rate` | float64 | 97.3% | 2.7% | 33 | median=0.375 | 0 | 1 |
| 167 | `away_roll_5_goals_scored_std` | float64 | 92.0% | 8.0% | 390 | median=1 | 0 | 3.78 |
| 168 | `away_roll_5_goals_conceded_std` | float64 | 92.0% | 8.0% | 377 | median=1 | 0 | 3.79 |
| 169 | `away_roll_5_shots_on_target_std` | float64 | 92.0% | 8.0% | 1490 | median=2.17 | 0 | 8.08 |
| 170 | `away_roll_5_points_std` | float64 | 92.0% | 8.0% | 148 | median=1.3 | 0 | 1.73 |
| 171 | `away_roll_10_goals_scored_std` | float64 | 92.0% | 8.0% | 1173 | median=1.06 | 0 | 3.39 |
| 172 | `away_roll_10_goals_conceded_std` | float64 | 92.0% | 8.0% | 1216 | median=1.07 | 0 | 3.79 |
| 173 | `away_roll_10_shots_on_target_std` | float64 | 92.0% | 8.0% | 3793 | median=2.26 | 0 | 8.08 |
| 174 | `away_roll_10_points_std` | float64 | 92.0% | 8.0% | 454 | median=1.26 | 0 | 1.73 |
| 175 | `away_goals_scored_trend` | float64 | 97.3% | 2.7% | 449 | median=0 | -2.17 | 2.73 |
| 176 | `away_goals_conceded_trend` | float64 | 97.3% | 2.7% | 413 | median=0 | -2.07 | 2.63 |
| 177 | `away_shots_on_target_trend` | float64 | 97.3% | 2.7% | 969 | median=0 | -5.2 | 5.93 |
| 178 | `away_points_trend` | float64 | 97.3% | 2.7% | 363 | median=0 | -1.9 | 2.1 |
| 179 | `away_venue_roll_3_goals_scored` | float64 | 99.4% | 0.6% | 21 | median=1 | 0 | 5.67 |
| 180 | `away_venue_roll_3_goals_conceded` | float64 | 99.4% | 0.6% | 21 | median=1.33 | 0 | 5 |
| 181 | `away_venue_roll_3_points` | float64 | 99.4% | 0.6% | 11 | median=1 | 0 | 3 |
| 182 | `away_venue_roll_3_clean_sheet` | float64 | 99.4% | 0.6% | 5 | median=0.333 | 0 | 1 |
| 183 | `away_venue_roll_5_goals_scored` | float64 | 99.4% | 0.6% | 36 | median=1.2 | 0 | 5 |
| 184 | `away_venue_roll_5_goals_conceded` | float64 | 99.4% | 0.6% | 45 | median=1.4 | 0 | 5 |
| 185 | `away_venue_roll_5_points` | float64 | 99.4% | 0.6% | 28 | median=1 | 0 | 3 |
| 186 | `away_venue_roll_5_clean_sheet` | float64 | 99.4% | 0.6% | 11 | median=0.2 | 0 | 1 |
| 187 | `away_venue_roll_10_goals_scored` | float64 | 99.4% | 0.6% | 85 | median=1.1 | 0 | 5 |
| 188 | `away_venue_roll_10_goals_conceded` | float64 | 99.4% | 0.6% | 102 | median=1.5 | 0 | 5 |
| 189 | `away_venue_roll_10_points` | float64 | 99.4% | 0.6% | 80 | median=1.1 | 0 | 3 |
| 190 | `away_venue_roll_10_clean_sheet` | float64 | 99.4% | 0.6% | 28 | median=0.2 | 0 | 1 |
| 191 | `away_attack_strength` | float64 | 100.0% | 0.0% | 14112 | median=0.957 | 0 | 4.35 |
| 192 | `away_defense_strength` | float64 | 100.0% | 0.0% | 14068 | median=1 | 0 | 4.34 |
| 193 | `away_xg_attack_strength` | float64 | 50.1% | 49.9% | 56 | median=1 | 0.258 | 2.47 |
| 194 | `away_xg_defense_strength` | float64 | 50.1% | 49.9% | 56 | median=1 | 0.193 | 3.27 |
| 195 | `away_rest_days` | float64 | 97.3% | 2.7% | 20 | median=7 | 2 | 21 |
| 196 | `away_is_congested` | bool | 100.0% | 0.0% | 2 | np.False_ | — | — |
| 197 | `away_win_streak` | float64 | 99.7% | 0.3% | 19 | median=0 | 0 | 18 |
| 198 | `away_unbeaten_run` | float64 | 99.7% | 0.3% | 44 | median=1 | 0 | 47 |
| 199 | `away_winless_run` | float64 | 99.7% | 0.3% | 29 | median=1 | 0 | 30 |
| 200 | `away_loss_streak` | float64 | 99.7% | 0.3% | 15 | median=0 | 0 | 16 |
| 201 | `away_scoring_streak` | float64 | 99.7% | 0.3% | 45 | median=2 | 0 | 44 |
| 202 | `away_clean_sheet_streak` | float64 | 99.7% | 0.3% | 12 | median=0 | 0 | 12 |
| 203 | `away_form_points_5` | float64 | 99.7% | 0.3% | 15 | median=7 | 0 | 15 |
| 204 | `away_adj_attack_5` | float64 | 97.3% | 2.7% | 14201 | median=1.28 | 0 | 20.1 |
| 205 | `away_adj_defense_5` | float64 | 97.3% | 2.7% | 14165 | median=1.34 | 0 | 16.2 |
| 206 | `away_adj_attack_10` | float64 | 97.3% | 2.7% | 14258 | median=1.28 | 0 | 23.5 |
| 207 | `away_adj_defense_10` | float64 | 97.3% | 2.7% | 14231 | median=1.37 | 0 | 16.8 |
| 208 | `away_opp_difficulty_roll_5` | float64 | 99.7% | 0.3% | 15725 | median=0.971 | 0.285 | 2.17 |
| 209 | `away_form_overperformance` | float64 | 99.7% | 0.3% | 14763 | median=6.19 | -1.58 | 15 |
| 210 | `away_gd_roll_3` | float64 | 99.7% | 0.3% | 36 | median=0 | -6 | 5.33 |
| 211 | `away_gd_roll_5` | float64 | 99.7% | 0.3% | 64 | median=0 | -4 | 4.8 |
| 212 | `away_gd_per_match` | float64 | 97.3% | 2.7% | 1460 | median=-0.0556 | -5 | 6 |
| 213 | `away_momentum_gradient` | float64 | 98.6% | 1.4% | 119 | median=0 | -0.9 | 0.9 |
| 214 | `away_ewma_form` | float64 | 99.1% | 0.9% | 15644 | median=1.39 | 0 | 3 |
| 215 | `away_last3_vs_prev3` | float64 | 98.6% | 1.4% | 19 | median=0 | -9 | 9 |
| 216 | `h2h_matches_played` | int64 | 100.0% | 0.0% | 8 | median=7 | 0 | 7 |
| 217 | `h2h_home_wins` | float64 | 90.9% | 9.1% | 8 | median=2 | 0 | 7 |
| 218 | `h2h_away_wins` | float64 | 90.9% | 9.1% | 8 | median=2 | 0 | 7 |
| 219 | `h2h_draws` | float64 | 90.9% | 9.1% | 8 | median=1 | 0 | 7 |
| 220 | `h2h_draw_rate` | float64 | 90.9% | 9.1% | 19 | median=0.25 | 0 | 1 |
| 221 | `h2h_goals_avg` | float64 | 90.9% | 9.1% | 97 | median=2.67 | 0 | 8 |
| 222 | `h2h_home_goals_avg` | float64 | 90.9% | 9.1% | 77 | median=1.29 | 0 | 7 |
| 223 | `h2h_away_goals_avg` | float64 | 90.9% | 9.1% | 82 | median=1.29 | 0 | 7 |
| 224 | `h2h_home_win_rate` | float64 | 90.9% | 9.1% | 19 | median=0.286 | 0 | 1 |
| 225 | `h2h_last_result` | float64 | 90.9% | 9.1% | 3 | median=0 | -1 | 1 |
| 226 | `h2h_weighted_home_win_rate` | float64 | 90.9% | 9.1% | 5679 | median=0.316 | 0 | 1 |
| 227 | `h2h_recent_5_win_rate` | float64 | 90.9% | 9.1% | 11 | median=0.4 | 0 | 1 |
| 228 | `h2h_recent_3_total_goals` | float64 | 90.9% | 9.1% | 31 | median=2.67 | 0 | 8 |
| 229 | `home_elo` | float64 | 100.0% | 0.0% | 15738 | median=1.48e+03 | 1.25e+03 | 1.82e+03 |
| 230 | `away_elo` | float64 | 100.0% | 0.0% | 15739 | median=1.48e+03 | 1.26e+03 | 1.82e+03 |
| 231 | `elo_diff` | float64 | 100.0% | 0.0% | 15804 | median=0.363 | -495 | 477 |
| 232 | `home_key_players_available` | float64 | 3.9% | 96.1% | 6 | median=3 | 0 | 5 |
| 233 | `away_key_players_available` | float64 | 3.9% | 96.1% | 6 | median=3 | 0 | 5 |
| 234 | `home_top_scorer_played` | float64 | 3.9% | 96.1% | 2 | median=0 | 0 | 1 |
| 235 | `away_top_scorer_played` | float64 | 3.9% | 96.1% | 2 | median=0 | 0 | 1 |
| 236 | `home_squad_rotation` | float64 | 3.9% | 96.1% | 9 | median=1 | 0 | 8 |
| 237 | `away_squad_rotation` | float64 | 3.9% | 96.1% | 11 | median=2 | 0 | 10 |
| 238 | `ref_avg_yellows` | float64 | 74.3% | 25.7% | 415 | median=3.46 | 0 | 8 |
| 239 | `ref_avg_reds` | float64 | 74.3% | 25.7% | 460 | median=0.158 | 0 | 3 |
| 240 | `ref_avg_fouls` | float64 | 74.3% | 25.7% | 292 | median=23.6 | 0 | 67 |
| 241 | `ref_matches_officiated` | float64 | 75.6% | 24.4% | 476 | median=62 | 0 | 475 |
| 242 | `ref_strictness_score` | float64 | 74.3% | 25.7% | 652 | median=0.248 | 0 | 1 |
| 243 | `ref_home_bias` | float64 | 74.3% | 25.7% | 608 | median=-0.005 | -0.469 | 0.545 |
| 244 | `ref_home_cards_bias` | float64 | 74.3% | 25.7% | 563 | median=0.097 | -1 | 1 |
| 245 | `ref_avg_total_goals` | float64 | 74.3% | 25.7% | 232 | median=2.7 | 0 | 8 |
| 246 | `ref_strictness_trend` | float64 | 74.3% | 25.7% | 1943 | median=0.0006 | -0.137 | 0.187 |
| 247 | `ref_big_match_cards` | float64 | 74.3% | 25.7% | 1208 | median=1 | 0 | 3 |
| 248 | `ref_home_team_cards` | float64 | 74.3% | 25.7% | 247 | median=1.57 | 0 | 6 |
| 249 | `ref_away_team_cards` | float64 | 74.3% | 25.7% | 252 | median=1.91 | 0 | 7 |
| 250 | `ref_vs_home_team_bias` | float64 | 74.3% | 25.7% | 2350 | median=0 | -2.69 | 3.21 |
| 251 | `ref_vs_away_team_bias` | float64 | 74.3% | 25.7% | 2322 | median=0 | -2.22 | 3.21 |
| 252 | `ref_last_match_reds` | float64 | 74.3% | 25.7% | 5 | median=0 | 0 | 4 |
| 253 | `ref_last_match_cards` | float64 | 74.3% | 25.7% | 18 | median=4 | 0 | 17 |
| 254 | `ref_regression_signal` | float64 | 74.3% | 25.7% | 1129 | median=-0.08 | -6.16 | 12.1 |
| 255 | `home_us_team_xg` | Float64 | 38.1% | 61.9% | 323 | median=54.6 | 29.6 | 104 |
| 256 | `home_us_team_npxg` | Float64 | 38.1% | 61.9% | 323 | median=50.2 | 26.6 | 95.5 |
| 257 | `home_us_team_xa` | Float64 | 38.1% | 61.9% | 323 | median=38.2 | 20.2 | 78.5 |
| 258 | `home_us_team_shots` | Int64 | 38.1% | 61.9% | 220 | median=493 | 327 | 790 |
| 259 | `home_us_team_key_passes` | Int64 | 38.1% | 61.9% | 192 | median=370 | 235 | 594 |
| 260 | `home_us_team_xg_chain` | Float64 | 38.1% | 61.9% | 323 | median=151 | 63.1 | 353 |
| 261 | `home_us_team_xg_buildup` | Float64 | 38.1% | 61.9% | 323 | median=88.9 | 32.7 | 231 |
| 262 | `home_us_team_minutes` | Int64 | 38.1% | 61.9% | 296 | median=3.76e+04 | 3.13e+04 | 4.28e+04 |
| 263 | `home_us_team_goals` | Int64 | 38.1% | 61.9% | 73 | median=53 | 23 | 105 |
| 264 | `home_us_team_assists` | Int64 | 38.1% | 61.9% | 58 | median=37 | 12 | 85 |
| 265 | `home_us_player_count` | float64 | 38.1% | 61.9% | 20 | median=27 | 20 | 42 |
| 266 | `home_us_team_xg_per_90` | Float64 | 38.1% | 61.9% | 323 | median=0.13 | 0.0778 | 0.249 |
| 267 | `home_us_team_xa_per_90` | Float64 | 38.1% | 61.9% | 323 | median=0.0909 | 0.0507 | 0.188 |
| 268 | `home_us_team_xg_per_shot` | Float64 | 38.1% | 61.9% | 323 | median=0.112 | 0.0797 | 0.157 |
| 269 | `home_us_team_goals_minus_xg` | Float64 | 38.1% | 61.9% | 323 | median=-2.68 | -22 | 22.2 |
| 270 | `home_us_top3_xg_share` | float64 | 38.1% | 61.9% | 323 | median=0.504 | 0.282 | 0.687 |
| 271 | `away_us_team_xg` | Float64 | 38.1% | 61.9% | 323 | median=54.6 | 29.6 | 104 |
| 272 | `away_us_team_npxg` | Float64 | 38.1% | 61.9% | 323 | median=50.2 | 26.6 | 95.5 |
| 273 | `away_us_team_xa` | Float64 | 38.1% | 61.9% | 323 | median=38.2 | 20.2 | 78.5 |
| 274 | `away_us_team_shots` | Int64 | 38.1% | 61.9% | 220 | median=493 | 327 | 790 |
| 275 | `away_us_team_key_passes` | Int64 | 38.1% | 61.9% | 192 | median=370 | 235 | 594 |
| 276 | `away_us_team_xg_chain` | Float64 | 38.1% | 61.9% | 323 | median=151 | 63.1 | 353 |
| 277 | `away_us_team_xg_buildup` | Float64 | 38.1% | 61.9% | 323 | median=88.9 | 32.7 | 231 |
| 278 | `away_us_team_minutes` | Int64 | 38.1% | 61.9% | 296 | median=3.76e+04 | 3.13e+04 | 4.28e+04 |
| 279 | `away_us_team_goals` | Int64 | 38.1% | 61.9% | 73 | median=53 | 23 | 105 |
| 280 | `away_us_team_assists` | Int64 | 38.1% | 61.9% | 58 | median=37 | 12 | 85 |
| 281 | `away_us_player_count` | float64 | 38.1% | 61.9% | 20 | median=27 | 20 | 42 |
| 282 | `away_us_team_xg_per_90` | Float64 | 38.1% | 61.9% | 323 | median=0.13 | 0.0778 | 0.249 |
| 283 | `away_us_team_xa_per_90` | Float64 | 38.1% | 61.9% | 323 | median=0.0909 | 0.0507 | 0.188 |
| 284 | `away_us_team_xg_per_shot` | Float64 | 38.1% | 61.9% | 323 | median=0.112 | 0.0797 | 0.157 |
| 285 | `away_us_team_goals_minus_xg` | Float64 | 38.1% | 61.9% | 323 | median=-2.68 | -22 | 22.2 |
| 286 | `away_us_top3_xg_share` | float64 | 38.1% | 61.9% | 323 | median=0.504 | 0.282 | 0.687 |
| 287 | `us_xg_diff` | Float64 | 100.0% | 0.0% | 5728 | median=0 | -104 | 104 |
| 288 | `us_xa_diff` | Float64 | 100.0% | 0.0% | 5728 | median=0 | -78.5 | 78.5 |
| 289 | `us_coverage` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 290 | `home_us_xg_rolling_avg` | float64 | 34.7% | 65.3% | 295 | median=52.9 | 34.6 | 97.3 |
| 291 | `home_us_xg_rolling_std` | float64 | 34.7% | 65.3% | 295 | median=6.4 | 0.122 | 24.2 |
| 292 | `home_us_xg_trend` | float64 | 34.7% | 65.3% | 295 | median=0.191 | -33.2 | 39.6 |
| 293 | `away_us_xg_rolling_avg` | float64 | 34.7% | 65.3% | 295 | median=52.9 | 34.6 | 97.3 |
| 294 | `away_us_xg_rolling_std` | float64 | 34.7% | 65.3% | 295 | median=6.4 | 0.122 | 24.2 |
| 295 | `away_us_xg_trend` | float64 | 34.7% | 65.3% | 295 | median=0.148 | -33.2 | 39.6 |
| 296 | `has_xg_data` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 297 | `home_ss_roll_xg` | float64 | 30.0% | 70.0% | 2853 | median=0.91 | 0 | 3.42 |
| 298 | `home_ss_roll_xgot` | float64 | 30.0% | 70.0% | 2848 | median=0.812 | 0 | 3.43 |
| 299 | `home_ss_roll_xa` | float64 | 30.0% | 70.0% | 2874 | median=0.553 | 0 | 2.27 |
| 300 | `home_ss_roll_goals` | float64 | 30.0% | 70.0% | 38 | median=1.2 | 0 | 4.8 |
| 301 | `home_ss_roll_total_shots` | float64 | 30.0% | 70.0% | 138 | median=12.2 | 4.5 | 25.6 |
| 302 | `home_ss_roll_shots_on_target` | float64 | 30.0% | 70.0% | 71 | median=4.2 | 0 | 10.4 |
| 303 | `home_ss_roll_big_chances_created` | float64 | 30.0% | 70.0% | 40 | median=1.4 | 0 | 4.8 |
| 304 | `home_ss_roll_key_passes` | float64 | 30.0% | 70.0% | 121 | median=9.2 | 2.5 | 19.4 |
| 305 | `home_ss_roll_pass_accuracy` | float64 | 30.0% | 70.0% | 4750 | median=0.803 | 0.608 | 0.92 |
| 306 | `home_ss_roll_xg_per_shot` | float64 | 30.0% | 70.0% | 2874 | median=0.0838 | 0 | 0.225 |
| 307 | `home_ss_roll_duel_win_rate` | float64 | 30.0% | 70.0% | 4747 | median=0.5 | 0.401 | 0.597 |
| 308 | `home_ss_roll_aerial_win_rate` | float64 | 30.0% | 70.0% | 4736 | median=0.499 | 0.278 | 0.714 |
| 309 | `home_ss_roll_tackles_won` | float64 | 30.0% | 70.0% | 107 | median=9.4 | 0 | 17.8 |
| 310 | `home_ss_roll_interceptions` | float64 | 30.0% | 70.0% | 111 | median=9 | 3.2 | 24 |
| 311 | `home_ss_roll_ball_recoveries` | float64 | 30.0% | 70.0% | 242 | median=56.2 | 34.6 | 83.2 |
| 312 | `home_ss_roll_blocks` | float64 | 30.0% | 70.0% | 75 | median=3.4 | 0 | 9.25 |
| 313 | `home_ss_roll_clearances` | float64 | 30.0% | 70.0% | 214 | median=20.4 | 6 | 43.8 |
| 314 | `home_ss_roll_fouls` | float64 | 30.0% | 70.0% | 106 | median=11.2 | 4.4 | 20 |
| 315 | `home_ss_roll_was_fouled` | float64 | 30.0% | 70.0% | 102 | median=10.6 | 4.4 | 18.6 |
| 316 | `home_ss_roll_progressive_carries` | float64 | 30.0% | 70.0% | 121 | median=0 | 0 | 29.2 |
| 317 | `home_ss_roll_rating` | float64 | 30.0% | 70.0% | 4392 | median=6.86 | 6.43 | 7.47 |
| 318 | `home_ss_roll_carry_distance_per_carry` | float64 | 30.0% | 70.0% | 480 | median=0 | 0 | 9.77 |
| 319 | `home_ss_roll_shot_blocked_rate` | float64 | 30.0% | 70.0% | 4501 | median=0.266 | 0.0364 | 0.553 |
| 320 | `home_ss_roll_last_man_tackle` | float64 | 30.0% | 70.0% | 12 | median=0 | 0 | 1.2 |
| 321 | `home_ss_roll_cross_accuracy` | float64 | 30.0% | 70.0% | 4677 | median=0.238 | 0.0417 | 0.534 |
| 322 | `home_ss_roll_error_to_goal` | float64 | 30.0% | 70.0% | 12 | median=0.2 | 0 | 1.2 |
| 323 | `home_ss_roll_error_to_shot` | float64 | 30.0% | 70.0% | 18 | median=0.2 | 0 | 2 |
| 324 | `home_ss_roll_contest_win_rate` | float64 | 30.0% | 70.0% | 4624 | median=0.525 | 0.183 | 0.833 |
| 325 | `home_ss_roll_unsuccessful_touch` | float64 | 30.0% | 70.0% | 112 | median=14.6 | 6.6 | 24 |
| 326 | `home_ss_roll_opp_half_pass_ratio` | float64 | 30.0% | 70.0% | 4750 | median=0.546 | 0.349 | 0.745 |
| 327 | `home_ss_roll_att_xg` | float64 | 30.0% | 70.0% | 2837 | median=0.362 | 0 | 2.15 |
| 328 | `home_ss_roll_att_xa` | float64 | 30.0% | 70.0% | 2874 | median=0.0859 | 0 | 1.02 |
| 329 | `home_ss_roll_att_shots` | float64 | 30.0% | 70.0% | 96 | median=4.6 | 0.8 | 15 |
| 330 | `home_ss_roll_att_key_passes` | float64 | 30.0% | 70.0% | 71 | median=2.4 | 0 | 9.8 |
| 331 | `home_ss_roll_att_rating` | float64 | 30.0% | 70.0% | 1934 | median=6.85 | 6.21 | 8.4 |
| 332 | `home_ss_roll_mid_pass_accuracy` | float64 | 30.0% | 70.0% | 4750 | median=0.822 | 0.645 | 0.926 |
| 333 | `home_ss_roll_mid_duel_win_rate` | float64 | 30.0% | 70.0% | 4742 | median=0.488 | 0.331 | 0.697 |
| 334 | `home_ss_roll_mid_tackles` | float64 | 30.0% | 70.0% | 104 | median=8 | 2.33 | 16.6 |
| 335 | `home_ss_roll_mid_interceptions` | float64 | 30.0% | 70.0% | 72 | median=3.8 | 0.2 | 11.5 |
| 336 | `home_ss_roll_mid_rating` | float64 | 30.0% | 70.0% | 3529 | median=6.85 | 6.4 | 7.69 |
| 337 | `home_ss_roll_def_aerial_win_rate` | float64 | 30.0% | 70.0% | 4456 | median=0.587 | 0.236 | 0.865 |
| 338 | `home_ss_roll_def_clearances` | float64 | 30.0% | 70.0% | 170 | median=13.4 | 3.5 | 34.6 |
| 339 | `home_ss_roll_def_blocks` | float64 | 30.0% | 70.0% | 55 | median=2.2 | 0 | 7 |
| 340 | `home_ss_roll_def_tackles_won` | float64 | 30.0% | 70.0% | 65 | median=3.8 | 0 | 9.8 |
| 341 | `home_ss_roll_def_rating` | float64 | 30.0% | 70.0% | 2262 | median=6.86 | 6.26 | 7.59 |
| 342 | `home_ss_roll_gk_goals_prevented` | float64 | 30.0% | 70.0% | 3030 | median=0 | -1.34 | 1.24 |
| 343 | `home_ss_roll_gk_rating` | float64 | 30.0% | 70.0% | 305 | median=6.92 | 5.74 | 8.58 |
| 344 | `home_ss_roll_territory_ratio` | float64 | 30.0% | 70.0% | 4750 | median=0.486 | 0.279 | 0.723 |
| 345 | `home_ss_roll_press_intensity` | float64 | 30.0% | 70.0% | 4750 | median=0.216 | 0.114 | 0.34 |
| 346 | `home_ss_roll_dispossess_rate` | float64 | 30.0% | 70.0% | 4750 | median=0.0145 | 0.00515 | 0.0368 |
| 347 | `home_ss_roll_total_poss_lost` | float64 | 30.0% | 70.0% | 440 | median=133 | 92.8 | 192 |
| 348 | `home_ss_roll_xi_changes` | float64 | 29.8% | 70.2% | 54 | median=2.4 | 0 | 7.2 |
| 349 | `home_ss_roll_mins_concentration` | float64 | 30.0% | 70.0% | 4471 | median=0.93 | 0.848 | 0.992 |
| 350 | `home_ss_roll_counter_xg_pct` | float64 | 8.7% | 91.3% | 1134 | median=0.0611 | 0 | 0.45 |
| 351 | `home_ss_roll_set_piece_xg_pct` | float64 | 8.7% | 91.3% | 1379 | median=0.185 | 0.0109 | 0.627 |
| 352 | `home_ss_roll_open_play_xg_pct` | float64 | 8.7% | 91.3% | 1381 | median=0.733 | 0.184 | 0.989 |
| 353 | `home_ss_roll_header_shot_pct` | float64 | 8.7% | 91.3% | 1341 | median=0.181 | 0 | 0.433 |
| 354 | `home_ss_roll_first_half_xg_share` | float64 | 8.7% | 91.3% | 1382 | median=0.449 | 0.138 | 0.851 |
| 355 | `home_ss_roll_last_15_xg_share` | float64 | 8.7% | 91.3% | 1382 | median=0.222 | 0 | 0.588 |
| 356 | `home_ss_roll_corner_xg_share` | float64 | 8.7% | 91.3% | 1371 | median=0.132 | 0 | 0.627 |
| 357 | `home_ss_roll_free_kick_xg_share` | float64 | 8.7% | 91.3% | 991 | median=0.0174 | 0 | 0.236 |
| 358 | `home_ss_roll_penalty_xg_share` | float64 | 8.7% | 91.3% | 467 | median=0.0485 | 0 | 0.514 |
| 359 | `home_ss_roll_penalties_taken` | float64 | 8.7% | 91.3% | 11 | median=0.2 | 0 | 1 |
| 360 | `home_ss_roll_penalties_scored` | float64 | 8.7% | 91.3% | 10 | median=0 | 0 | 1 |
| 361 | `home_ss_roll_possession` | float64 | 30.0% | 70.0% | 278 | median=49.6 | 25.6 | 75 |
| 362 | `home_ss_roll_corners` | float64 | 30.0% | 70.0% | 79 | median=4.8 | 0.8 | 12 |
| 363 | `home_ss_roll_throw_ins` | float64 | 30.0% | 70.0% | 149 | median=19.2 | 9.4 | 32 |
| 364 | `home_ss_roll_shots_inside_box` | float64 | 30.0% | 70.0% | 104 | median=7.8 | 2.4 | 18.6 |
| 365 | `home_ss_roll_shots_outside_box` | float64 | 30.0% | 70.0% | 77 | median=4.4 | 0.6 | 10.8 |
| 366 | `home_ss_roll_shots_inside_box_pct` | float64 | 30.0% | 70.0% | 4474 | median=0.648 | 0.285 | 0.906 |
| 367 | `home_ss_roll_hit_woodwork` | float64 | 30.0% | 70.0% | 16 | median=0.2 | 0 | 1.6 |
| 368 | `home_ss_roll_big_chances_scored` | float64 | 30.0% | 70.0% | 24 | median=0.8 | 0 | 3 |
| 369 | `home_ss_roll_touches_in_opp_box` | float64 | 30.0% | 70.0% | 218 | median=0 | 0 | 57.6 |
| 370 | `home_ss_roll_fouled_final_third` | float64 | 30.0% | 70.0% | 41 | median=2 | 0 | 5.4 |
| 371 | `home_ss_roll_final_third_entries` | float64 | 30.0% | 70.0% | 351 | median=52.4 | 0 | 92 |
| 372 | `home_ss_roll_final_third_phases` | float64 | 30.0% | 70.0% | 594 | median=0 | 0 | 294 |
| 373 | `home_ss_roll_duel_won_pct` | float64 | 30.0% | 70.0% | 125 | median=50 | 40 | 59.8 |
| 374 | `home_ss_roll_ground_duels_pct` | float64 | 30.0% | 70.0% | 170 | median=34.8 | 22.8 | 52 |
| 375 | `home_ss_roll_aerial_duels_pct` | float64 | 30.0% | 70.0% | 180 | median=15.2 | 4.5 | 34.2 |
| 376 | `home_ss_roll_dribbles_pct` | float64 | 29.0% | 71.0% | 121 | median=7.8 | 1.5 | 19.2 |
| 377 | `home_ss_roll_dive_saves` | float64 | 30.0% | 70.0% | 18 | median=0 | 0 | 2.6 |
| 378 | `home_ss_roll_high_claims` | float64 | 30.0% | 70.0% | 25 | median=0 | 0 | 3.6 |
| 379 | `home_ss_roll_dispossessed` | float64 | 30.0% | 70.0% | 108 | median=8.8 | 0 | 17.8 |
| 380 | `home_ss_roll_avg_shot_xg` | float64 | 30.0% | 70.0% | 2465 | median=0.0871 | 0 | 0.223 |
| 381 | `home_ss_roll_max_shot_xg` | float64 | 30.0% | 70.0% | 2913 | median=0.322 | 0 | 0.814 |
| 382 | `home_ss_roll_total_xgot` | float64 | 30.0% | 70.0% | 3008 | median=0.878 | 0 | 3.43 |
| 383 | `home_ss_roll_sm_inside_box_pct` | float64 | 30.0% | 70.0% | 4453 | median=0.653 | 0.285 | 0.906 |
| 384 | `home_ss_roll_sm_header_pct` | float64 | 30.0% | 70.0% | 4403 | median=0.174 | 0.0133 | 0.448 |
| 385 | `home_ss_roll_sm_open_play_pct` | float64 | 30.0% | 70.0% | 4477 | median=0.668 | 0.246 | 0.955 |
| 386 | `home_ss_roll_sm_set_piece_pct` | float64 | 30.0% | 70.0% | 4458 | median=0.255 | 0.0455 | 0.583 |
| 387 | `home_ss_roll_sm_counter_pct` | float64 | 30.0% | 70.0% | 2042 | median=0.0411 | 0 | 0.28 |
| 388 | `home_ss_roll_sm_conversion_rate` | float64 | 30.0% | 70.0% | 4031 | median=0.11 | 0 | 0.338 |
| 389 | `home_ss_roll_sm_big_chance_pct` | float64 | 30.0% | 70.0% | 2492 | median=0.0633 | 0 | 0.336 |
| 390 | `home_ss_roll_sm_avg_shot_distance` | float64 | 30.0% | 70.0% | 2991 | median=18.7 | 13 | 24.8 |
| 391 | `home_ss_roll_sm_median_shot_distance` | float64 | 30.0% | 70.0% | 3226 | median=18.5 | 11 | 27.4 |
| 392 | `home_ss_roll_sm_shot_distance_std` | float64 | 30.0% | 70.0% | 2294 | median=7.28 | 4.38 | 11.3 |
| 393 | `home_ss_roll_sm_close_range_pct` | float64 | 30.0% | 70.0% | 3283 | median=0.0738 | 0 | 0.299 |
| 394 | `home_ss_roll_xg_std` | float64 | 30.0% | 70.0% | 2874 | median=0.405 | 0 | 2.72 |
| 395 | `home_ss_roll_rating_std` | float64 | 30.0% | 70.0% | 4743 | median=0.198 | 0.0237 | 0.563 |
| 396 | `home_top2_xg_share` | float64 | 30.4% | 69.6% | 2901 | median=0.511 | 0 | 1 |
| 397 | `away_ss_roll_xg` | float64 | 30.0% | 70.0% | 2854 | median=0.926 | 0 | 3.26 |
| 398 | `away_ss_roll_xgot` | float64 | 30.0% | 70.0% | 2854 | median=0.831 | 0 | 3.86 |
| 399 | `away_ss_roll_xa` | float64 | 30.0% | 70.0% | 2877 | median=0.566 | 0 | 2.24 |
| 400 | `away_ss_roll_goals` | float64 | 30.0% | 70.0% | 40 | median=1.2 | 0 | 4.5 |
| 401 | `away_ss_roll_total_shots` | float64 | 30.0% | 70.0% | 136 | median=12.4 | 4.4 | 25.8 |
| 402 | `away_ss_roll_shots_on_target` | float64 | 30.0% | 70.0% | 76 | median=4.2 | 0 | 10.8 |
| 403 | `away_ss_roll_big_chances_created` | float64 | 30.0% | 70.0% | 41 | median=1.4 | 0 | 5.5 |
| 404 | `away_ss_roll_key_passes` | float64 | 30.0% | 70.0% | 120 | median=9.2 | 2.5 | 19.5 |
| 405 | `away_ss_roll_pass_accuracy` | float64 | 30.0% | 70.0% | 4751 | median=0.805 | 0.603 | 0.919 |
| 406 | `away_ss_roll_xg_per_shot` | float64 | 30.0% | 70.0% | 2877 | median=0.0838 | 0 | 0.23 |
| 407 | `away_ss_roll_duel_win_rate` | float64 | 30.0% | 70.0% | 4750 | median=0.501 | 0.401 | 0.62 |
| 408 | `away_ss_roll_aerial_win_rate` | float64 | 30.0% | 70.0% | 4730 | median=0.499 | 0.338 | 0.711 |
| 409 | `away_ss_roll_tackles_won` | float64 | 30.0% | 70.0% | 106 | median=9.4 | 0 | 18.5 |
| 410 | `away_ss_roll_interceptions` | float64 | 30.0% | 70.0% | 101 | median=9 | 3.2 | 24 |
| 411 | `away_ss_roll_ball_recoveries` | float64 | 30.0% | 70.0% | 240 | median=56.2 | 34 | 78.6 |
| 412 | `away_ss_roll_blocks` | float64 | 30.0% | 70.0% | 73 | median=3.4 | 0 | 9.6 |
| 413 | `away_ss_roll_clearances` | float64 | 30.0% | 70.0% | 215 | median=20.2 | 4.67 | 44.4 |
| 414 | `away_ss_roll_fouls` | float64 | 30.0% | 70.0% | 103 | median=11 | 4.6 | 19.6 |
| 415 | `away_ss_roll_was_fouled` | float64 | 30.0% | 70.0% | 103 | median=10.6 | 4.2 | 18.6 |
| 416 | `away_ss_roll_progressive_carries` | float64 | 30.0% | 70.0% | 128 | median=0 | 0 | 28.8 |
| 417 | `away_ss_roll_rating` | float64 | 30.0% | 70.0% | 4395 | median=6.87 | 6.39 | 7.43 |
| 418 | `away_ss_roll_carry_distance_per_carry` | float64 | 30.0% | 70.0% | 485 | median=0 | 0 | 10.1 |
| 419 | `away_ss_roll_shot_blocked_rate` | float64 | 30.0% | 70.0% | 4519 | median=0.265 | 0.0827 | 0.548 |
| 420 | `away_ss_roll_last_man_tackle` | float64 | 30.0% | 70.0% | 10 | median=0 | 0 | 1.2 |
| 421 | `away_ss_roll_cross_accuracy` | float64 | 30.0% | 70.0% | 4680 | median=0.237 | 0.0455 | 0.489 |
| 422 | `away_ss_roll_error_to_goal` | float64 | 30.0% | 70.0% | 11 | median=0.2 | 0 | 1.2 |
| 423 | `away_ss_roll_error_to_shot` | float64 | 30.0% | 70.0% | 18 | median=0.2 | 0 | 2 |
| 424 | `away_ss_roll_contest_win_rate` | float64 | 30.0% | 70.0% | 4632 | median=0.526 | 0.247 | 0.836 |
| 425 | `away_ss_roll_unsuccessful_touch` | float64 | 30.0% | 70.0% | 111 | median=14.6 | 7 | 23.4 |
| 426 | `away_ss_roll_opp_half_pass_ratio` | float64 | 30.0% | 70.0% | 4751 | median=0.548 | 0.342 | 0.769 |
| 427 | `away_ss_roll_att_xg` | float64 | 30.0% | 70.0% | 2839 | median=0.38 | 0 | 2.25 |
| 428 | `away_ss_roll_att_xa` | float64 | 30.0% | 70.0% | 2877 | median=0.0892 | 0 | 0.933 |
| 429 | `away_ss_roll_att_shots` | float64 | 30.0% | 70.0% | 95 | median=4.8 | 0.5 | 16.2 |
| 430 | `away_ss_roll_att_key_passes` | float64 | 30.0% | 70.0% | 72 | median=2.4 | 0 | 10.2 |
| 431 | `away_ss_roll_att_rating` | float64 | 30.0% | 70.0% | 1929 | median=6.85 | 6.17 | 8.35 |
| 432 | `away_ss_roll_mid_pass_accuracy` | float64 | 30.0% | 70.0% | 4751 | median=0.822 | 0.645 | 0.927 |
| 433 | `away_ss_roll_mid_duel_win_rate` | float64 | 30.0% | 70.0% | 4746 | median=0.489 | 0.346 | 0.664 |
| 434 | `away_ss_roll_mid_tackles` | float64 | 30.0% | 70.0% | 96 | median=7.8 | 1.5 | 17.6 |
| 435 | `away_ss_roll_mid_interceptions` | float64 | 30.0% | 70.0% | 72 | median=3.8 | 0.5 | 11.5 |
| 436 | `away_ss_roll_mid_rating` | float64 | 30.0% | 70.0% | 3576 | median=6.86 | 6.36 | 7.6 |
| 437 | `away_ss_roll_def_aerial_win_rate` | float64 | 30.0% | 70.0% | 4436 | median=0.586 | 0.275 | 0.879 |
| 438 | `away_ss_roll_def_clearances` | float64 | 30.0% | 70.0% | 176 | median=13.2 | 3 | 33.6 |
| 439 | `away_ss_roll_def_blocks` | float64 | 30.0% | 70.0% | 53 | median=2.2 | 0 | 6 |
| 440 | `away_ss_roll_def_tackles_won` | float64 | 30.0% | 70.0% | 71 | median=3.6 | 0 | 10.2 |
| 441 | `away_ss_roll_def_rating` | float64 | 30.0% | 70.0% | 2249 | median=6.87 | 6.22 | 7.57 |
| 442 | `away_ss_roll_gk_goals_prevented` | float64 | 30.0% | 70.0% | 3029 | median=0 | -1.59 | 1.24 |
| 443 | `away_ss_roll_gk_rating` | float64 | 30.0% | 70.0% | 316 | median=6.92 | 5.62 | 8.3 |
| 444 | `away_ss_roll_territory_ratio` | float64 | 30.0% | 70.0% | 4751 | median=0.488 | 0.278 | 0.756 |
| 445 | `away_ss_roll_press_intensity` | float64 | 30.0% | 70.0% | 4751 | median=0.216 | 0.116 | 0.349 |
| 446 | `away_ss_roll_dispossess_rate` | float64 | 30.0% | 70.0% | 4751 | median=0.0144 | 0.00481 | 0.0368 |
| 447 | `away_ss_roll_total_poss_lost` | float64 | 30.0% | 70.0% | 443 | median=134 | 93 | 189 |
| 448 | `away_ss_roll_xi_changes` | float64 | 29.8% | 70.2% | 55 | median=2.4 | 0 | 8.6 |
| 449 | `away_ss_roll_mins_concentration` | float64 | 30.0% | 70.0% | 4481 | median=0.93 | 0.85 | 0.993 |
| 450 | `away_ss_roll_counter_xg_pct` | float64 | 8.7% | 91.3% | 1112 | median=0.0591 | 0 | 0.456 |
| 451 | `away_ss_roll_set_piece_xg_pct` | float64 | 8.7% | 91.3% | 1379 | median=0.187 | 0.0121 | 0.534 |
| 452 | `away_ss_roll_open_play_xg_pct` | float64 | 8.7% | 91.3% | 1381 | median=0.731 | 0.256 | 0.988 |
| 453 | `away_ss_roll_header_shot_pct` | float64 | 8.7% | 91.3% | 1328 | median=0.18 | 0 | 0.575 |
| 454 | `away_ss_roll_first_half_xg_share` | float64 | 8.7% | 91.3% | 1381 | median=0.447 | 0.134 | 0.9 |
| 455 | `away_ss_roll_last_15_xg_share` | float64 | 8.7% | 91.3% | 1379 | median=0.229 | 0 | 0.552 |
| 456 | `away_ss_roll_corner_xg_share` | float64 | 8.7% | 91.3% | 1370 | median=0.133 | 0 | 0.474 |
| 457 | `away_ss_roll_free_kick_xg_share` | float64 | 8.7% | 91.3% | 999 | median=0.0171 | 0 | 0.279 |
| 458 | `away_ss_roll_penalty_xg_share` | float64 | 8.7% | 91.3% | 455 | median=0.0457 | 0 | 0.442 |
| 459 | `away_ss_roll_penalties_taken` | float64 | 8.7% | 91.3% | 11 | median=0.2 | 0 | 1 |
| 460 | `away_ss_roll_penalties_scored` | float64 | 8.7% | 91.3% | 9 | median=0 | 0 | 0.8 |
| 461 | `away_ss_roll_possession` | float64 | 30.0% | 70.0% | 283 | median=49.8 | 24.8 | 76.2 |
| 462 | `away_ss_roll_corners` | float64 | 30.0% | 70.0% | 82 | median=5 | 1.33 | 12.8 |
| 463 | `away_ss_roll_throw_ins` | float64 | 30.0% | 70.0% | 144 | median=19.4 | 9.6 | 32.2 |
| 464 | `away_ss_roll_shots_inside_box` | float64 | 30.0% | 70.0% | 111 | median=8 | 1.8 | 18.6 |
| 465 | `away_ss_roll_shots_outside_box` | float64 | 30.0% | 70.0% | 79 | median=4.4 | 0.6 | 11 |
| 466 | `away_ss_roll_shots_inside_box_pct` | float64 | 30.0% | 70.0% | 4502 | median=0.648 | 0.308 | 0.913 |
| 467 | `away_ss_roll_hit_woodwork` | float64 | 30.0% | 70.0% | 16 | median=0.4 | 0 | 2 |
| 468 | `away_ss_roll_big_chances_scored` | float64 | 30.0% | 70.0% | 27 | median=0.8 | 0 | 3.2 |
| 469 | `away_ss_roll_touches_in_opp_box` | float64 | 30.0% | 70.0% | 223 | median=0 | 0 | 56.8 |
| 470 | `away_ss_roll_fouled_final_third` | float64 | 30.0% | 70.0% | 43 | median=2 | 0 | 5.6 |
| 471 | `away_ss_roll_final_third_entries` | float64 | 30.0% | 70.0% | 342 | median=52.6 | 0 | 86 |
| 472 | `away_ss_roll_final_third_phases` | float64 | 30.0% | 70.0% | 587 | median=0 | 0 | 315 |
| 473 | `away_ss_roll_ground_duels_pct` | float64 | 30.0% | 70.0% | 169 | median=34.8 | 22.4 | 52 |
| 474 | `away_ss_roll_aerial_duels_pct` | float64 | 30.0% | 70.0% | 172 | median=15.2 | 5 | 41.4 |
| 475 | `away_ss_roll_dribbles_pct` | float64 | 28.9% | 71.1% | 114 | median=8 | 2 | 18.6 |
| 476 | `away_ss_roll_dive_saves` | float64 | 30.0% | 70.0% | 19 | median=0 | 0 | 2.4 |
| 477 | `away_ss_roll_high_claims` | float64 | 30.0% | 70.0% | 23 | median=0 | 0 | 3.4 |
| 478 | `away_ss_roll_dispossessed` | float64 | 30.0% | 70.0% | 107 | median=8.8 | 0 | 17.8 |
| 479 | `away_ss_roll_avg_shot_xg` | float64 | 30.0% | 70.0% | 2493 | median=0.0869 | 0 | 0.225 |
| 480 | `away_ss_roll_max_shot_xg` | float64 | 30.0% | 70.0% | 2918 | median=0.322 | 0 | 0.842 |
| 481 | `away_ss_roll_total_xgot` | float64 | 30.0% | 70.0% | 3009 | median=0.891 | 0 | 3.86 |
| 482 | `away_ss_roll_sm_inside_box_pct` | float64 | 30.0% | 70.0% | 4498 | median=0.654 | 0.308 | 0.913 |
| 483 | `away_ss_roll_sm_header_pct` | float64 | 30.0% | 70.0% | 4445 | median=0.174 | 0 | 0.583 |
| 484 | `away_ss_roll_sm_open_play_pct` | float64 | 30.0% | 70.0% | 4504 | median=0.669 | 0.199 | 0.919 |
| 485 | `away_ss_roll_sm_set_piece_pct` | float64 | 30.0% | 70.0% | 4496 | median=0.254 | 0.0417 | 0.585 |
| 486 | `away_ss_roll_sm_counter_pct` | float64 | 30.0% | 70.0% | 2050 | median=0.0404 | 0 | 0.28 |
| 487 | `away_ss_roll_sm_conversion_rate` | float64 | 30.0% | 70.0% | 4100 | median=0.11 | 0 | 0.5 |
| 488 | `away_ss_roll_sm_big_chance_pct` | float64 | 30.0% | 70.0% | 2553 | median=0.0632 | 0 | 0.358 |
| 489 | `away_ss_roll_sm_avg_shot_distance` | float64 | 30.0% | 70.0% | 2923 | median=18.7 | 13.5 | 25 |
| 490 | `away_ss_roll_sm_median_shot_distance` | float64 | 30.0% | 70.0% | 3206 | median=18.4 | 11.1 | 26.4 |
| 491 | `away_ss_roll_sm_shot_distance_std` | float64 | 30.0% | 70.0% | 2298 | median=7.3 | 3.93 | 11.5 |
| 492 | `away_ss_roll_sm_close_range_pct` | float64 | 30.0% | 70.0% | 3296 | median=0.0741 | 0 | 0.365 |
| 493 | `away_ss_roll_xg_std` | float64 | 30.0% | 70.0% | 2877 | median=0.408 | 0 | 2.66 |
| 494 | `away_ss_roll_rating_std` | float64 | 30.0% | 70.0% | 4749 | median=0.199 | 0.0132 | 0.568 |
| 495 | `away_top2_xg_share` | float64 | 30.4% | 69.6% | 2865 | median=0.536 | 0 | 1 |
| 496 | `ss_diff_ss_roll_xg` | float64 | 29.9% | 70.1% | 2858 | median=0 | -2.24 | 2.41 |
| 497 | `ss_diff_ss_roll_xgot` | float64 | 29.9% | 70.1% | 2865 | median=0 | -2.75 | 2.38 |
| 498 | `ss_diff_ss_roll_xa` | float64 | 29.9% | 70.1% | 2868 | median=0 | -1.59 | 1.81 |
| 499 | `ss_diff_ss_roll_goals` | float64 | 29.9% | 70.1% | 148 | median=0 | -3.8 | 4 |
| 500 | `ss_diff_ss_roll_total_shots` | float64 | 29.9% | 70.1% | 440 | median=-0.2 | -15.6 | 17.6 |
| 501 | `ss_diff_ss_roll_shots_on_target` | float64 | 29.9% | 70.1% | 248 | median=0 | -7.6 | 6.2 |
| 502 | `ss_diff_ss_roll_big_chances_created` | float64 | 29.9% | 70.1% | 160 | median=0 | -4.5 | 4.2 |
| 503 | `ss_diff_ss_roll_key_passes` | float64 | 29.9% | 70.1% | 388 | median=-0.2 | -13.2 | 15 |
| 504 | `ss_diff_ss_roll_pass_accuracy` | float64 | 29.9% | 70.1% | 4735 | median=-0.000789 | -0.26 | 0.283 |
| 505 | `ss_diff_ss_roll_xg_per_shot` | float64 | 29.9% | 70.1% | 2868 | median=0 | -0.181 | 0.149 |
| 506 | `ss_diff_ss_roll_duel_win_rate` | float64 | 29.9% | 70.1% | 4735 | median=-0.000733 | -0.122 | 0.131 |
| 507 | `ss_diff_ss_roll_aerial_win_rate` | float64 | 29.9% | 70.1% | 4735 | median=0.000824 | -0.273 | 0.275 |
| 508 | `ss_diff_ss_roll_tackles_won` | float64 | 29.9% | 70.1% | 308 | median=0 | -10.3 | 8.8 |
| 509 | `ss_diff_ss_roll_interceptions` | float64 | 29.9% | 70.1% | 318 | median=0 | -14 | 15 |
| 510 | `ss_diff_ss_roll_ball_recoveries` | float64 | 29.9% | 70.1% | 484 | median=-0.2 | -28.5 | 22 |
| 511 | `ss_diff_ss_roll_blocks` | float64 | 29.9% | 70.1% | 233 | median=0 | -8 | 7 |
| 512 | `ss_diff_ss_roll_clearances` | float64 | 29.9% | 70.1% | 637 | median=0.2 | -25.2 | 26 |
| 513 | `ss_diff_ss_roll_fouls` | float64 | 29.9% | 70.1% | 299 | median=0 | -9.8 | 9 |
| 514 | `ss_diff_ss_roll_was_fouled` | float64 | 29.9% | 70.1% | 318 | median=0 | -9.8 | 11 |
| 515 | `ss_diff_ss_roll_progressive_carries` | float64 | 29.9% | 70.1% | 264 | median=0 | -17 | 16.8 |
| 516 | `ss_diff_ss_roll_rating` | float64 | 29.9% | 70.1% | 4700 | median=-0.0109 | -0.756 | 0.839 |
| 517 | `ss_diff_ss_roll_carry_distance_per_carry` | float64 | 29.9% | 70.1% | 595 | median=0 | -9.85 | 9.75 |
| 518 | `ss_diff_ss_roll_shot_blocked_rate` | float64 | 29.9% | 70.1% | 4733 | median=0.000116 | -0.34 | 0.326 |
| 519 | `ss_diff_ss_roll_last_man_tackle` | float64 | 29.9% | 70.1% | 32 | median=0 | -1.2 | 1.2 |
| 520 | `ss_diff_ss_roll_cross_accuracy` | float64 | 29.9% | 70.1% | 4733 | median=1.12e-05 | -0.29 | 0.337 |
| 521 | `ss_diff_ss_roll_error_to_goal` | float64 | 29.9% | 70.1% | 41 | median=0 | -1.2 | 1.2 |
| 522 | `ss_diff_ss_roll_error_to_shot` | float64 | 29.9% | 70.1% | 71 | median=0 | -1.6 | 1.6 |
| 523 | `ss_diff_ss_roll_contest_win_rate` | float64 | 29.9% | 70.1% | 4735 | median=-0.00204 | -0.408 | 0.366 |
| 524 | `ss_diff_ss_roll_unsuccessful_touch` | float64 | 29.9% | 70.1% | 311 | median=0 | -10.6 | 10.6 |
| 525 | `ss_diff_ss_roll_opp_half_pass_ratio` | float64 | 29.9% | 70.1% | 4735 | median=-0.00178 | -0.235 | 0.286 |
| 526 | `ss_diff_ss_roll_att_xg` | float64 | 29.9% | 70.1% | 2861 | median=0 | -1.81 | 2.06 |
| 527 | `ss_diff_ss_roll_att_xa` | float64 | 29.9% | 70.1% | 2868 | median=0 | -0.829 | 0.898 |
| 528 | `ss_diff_ss_roll_att_shots` | float64 | 29.9% | 70.1% | 348 | median=-0.2 | -15.2 | 11.4 |
| 529 | `ss_diff_ss_roll_att_key_passes` | float64 | 29.9% | 70.1% | 262 | median=0 | -7.8 | 8.4 |
| 530 | `ss_diff_ss_roll_att_rating` | float64 | 29.9% | 70.1% | 3355 | median=-0.00733 | -1.59 | 1.57 |
| 531 | `ss_diff_ss_roll_mid_pass_accuracy` | float64 | 29.9% | 70.1% | 4735 | median=0.000355 | -0.235 | 0.214 |
| 532 | `ss_diff_ss_roll_mid_duel_win_rate` | float64 | 29.9% | 70.1% | 4735 | median=0.000255 | -0.292 | 0.25 |
| 533 | `ss_diff_ss_roll_mid_tackles` | float64 | 29.9% | 70.1% | 320 | median=0 | -10.2 | 11 |
| 534 | `ss_diff_ss_roll_mid_interceptions` | float64 | 29.9% | 70.1% | 238 | median=0 | -6 | 8 |
| 535 | `ss_diff_ss_roll_mid_rating` | float64 | 29.9% | 70.1% | 4518 | median=-0.011 | -0.898 | 0.959 |
| 536 | `ss_diff_ss_roll_def_aerial_win_rate` | float64 | 29.9% | 70.1% | 4730 | median=7.22e-06 | -0.497 | 0.381 |
| 537 | `ss_diff_ss_roll_def_clearances` | float64 | 29.9% | 70.1% | 533 | median=0.2 | -17.8 | 23 |
| 538 | `ss_diff_ss_roll_def_blocks` | float64 | 29.9% | 70.1% | 204 | median=0 | -5.67 | 5.5 |
| 539 | `ss_diff_ss_roll_def_tackles_won` | float64 | 29.9% | 70.1% | 220 | median=0 | -6.6 | 6.4 |
| 540 | `ss_diff_ss_roll_def_rating` | float64 | 29.9% | 70.1% | 3670 | median=-0.00519 | -1.03 | 1.18 |
| 541 | `ss_diff_ss_roll_gk_goals_prevented` | float64 | 29.9% | 70.1% | 3027 | median=0 | -1.83 | 2.03 |
| 542 | `ss_diff_ss_roll_gk_rating` | float64 | 29.9% | 70.1% | 706 | median=0 | -1.58 | 2.1 |
| 543 | `ss_diff_ss_roll_territory_ratio` | float64 | 29.9% | 70.1% | 4735 | median=-0.00249 | -0.275 | 0.312 |
| 544 | `ss_diff_ss_roll_press_intensity` | float64 | 29.9% | 70.1% | 4735 | median=0.000706 | -0.202 | 0.185 |
| 545 | `ss_diff_ss_roll_dispossess_rate` | float64 | 29.9% | 70.1% | 4735 | median=1.5e-05 | -0.0209 | 0.017 |
| 546 | `ss_diff_ss_roll_total_poss_lost` | float64 | 29.9% | 70.1% | 1076 | median=-0.2 | -62.4 | 61.2 |
| 547 | `ss_diff_ss_roll_xi_changes` | float64 | 29.6% | 70.4% | 213 | median=0 | -5.6 | 5.2 |
| 548 | `ss_diff_ss_roll_mins_concentration` | float64 | 29.9% | 70.1% | 4686 | median=-0.000265 | -0.0962 | 0.0914 |
| 549 | `ss_diff_ss_roll_counter_xg_pct` | float64 | 8.6% | 91.4% | 1348 | median=0.00234 | -0.438 | 0.382 |
| 550 | `ss_diff_ss_roll_set_piece_xg_pct` | float64 | 8.6% | 91.4% | 1370 | median=-0.000298 | -0.38 | 0.576 |
| 551 | `ss_diff_ss_roll_open_play_xg_pct` | float64 | 8.6% | 91.4% | 1370 | median=-0.00697 | -0.638 | 0.493 |
| 552 | `ss_diff_ss_roll_header_shot_pct` | float64 | 8.6% | 91.4% | 1370 | median=-0.000401 | -0.433 | 0.313 |
| 553 | `ss_diff_ss_roll_first_half_xg_share` | float64 | 8.6% | 91.4% | 1370 | median=0.00438 | -0.5 | 0.492 |
| 554 | `ss_diff_ss_roll_last_15_xg_share` | float64 | 8.6% | 91.4% | 1370 | median=-0.00497 | -0.454 | 0.393 |
| 555 | `ss_diff_ss_roll_corner_xg_share` | float64 | 8.6% | 91.4% | 1370 | median=-0.00333 | -0.361 | 0.576 |
| 556 | `ss_diff_ss_roll_free_kick_xg_share` | float64 | 8.6% | 91.4% | 1326 | median=0 | -0.231 | 0.227 |
| 557 | `ss_diff_ss_roll_penalty_xg_share` | float64 | 8.6% | 91.4% | 917 | median=0 | -0.442 | 0.441 |
| 558 | `ss_diff_ss_roll_penalties_taken` | float64 | 8.6% | 91.4% | 35 | median=0 | -1 | 1 |
| 559 | `ss_diff_ss_roll_penalties_scored` | float64 | 8.6% | 91.4% | 27 | median=0 | -0.8 | 1 |
| 560 | `ss_diff_ss_roll_possession` | float64 | 29.9% | 70.1% | 691 | median=-0.2 | -35.8 | 39.2 |
| 561 | `ss_diff_ss_roll_corners` | float64 | 29.9% | 70.1% | 274 | median=-0.2 | -8.4 | 7.8 |
| 562 | `ss_diff_ss_roll_throw_ins` | float64 | 29.9% | 70.1% | 398 | median=0 | -15.4 | 15 |
| 563 | `ss_diff_ss_roll_shots_inside_box` | float64 | 29.9% | 70.1% | 337 | median=-0.2 | -11.4 | 13.4 |
| 564 | `ss_diff_ss_roll_shots_outside_box` | float64 | 29.9% | 70.1% | 255 | median=0 | -8.73 | 7.2 |
| 565 | `ss_diff_ss_roll_shots_inside_box_pct` | float64 | 29.9% | 70.1% | 4732 | median=0.000971 | -0.382 | 0.484 |
| 566 | `ss_diff_ss_roll_hit_woodwork` | float64 | 29.9% | 70.1% | 67 | median=0 | -2 | 1.6 |
| 567 | `ss_diff_ss_roll_big_chances_scored` | float64 | 29.9% | 70.1% | 100 | median=0 | -3 | 2.4 |
| 568 | `ss_diff_ss_roll_touches_in_opp_box` | float64 | 29.9% | 70.1% | 581 | median=0 | -30.2 | 38.8 |
| 569 | `ss_diff_ss_roll_fouled_final_third` | float64 | 29.9% | 70.1% | 163 | median=0 | -4.4 | 4.2 |
| 570 | `ss_diff_ss_roll_final_third_entries` | float64 | 29.9% | 70.1% | 778 | median=-0.2 | -71 | 71 |
| 571 | `ss_diff_ss_roll_final_third_phases` | float64 | 29.9% | 70.1% | 1127 | median=0 | -236 | 216 |
| 572 | `ss_diff_ss_roll_duel_won_pct` | float64 | 29.9% | 70.1% | 260 | median=0 | -12.4 | 13 |
| 573 | `ss_diff_ss_roll_ground_duels_pct` | float64 | 29.9% | 70.1% | 506 | median=0 | -26 | 21.5 |
| 574 | `ss_diff_ss_roll_aerial_duels_pct` | float64 | 29.9% | 70.1% | 526 | median=0 | -21.6 | 23.6 |
| 575 | `ss_diff_ss_roll_dribbles_pct` | float64 | 28.6% | 71.4% | 366 | median=0 | -10.8 | 12.5 |
| 576 | `ss_diff_ss_roll_dive_saves` | float64 | 29.9% | 70.1% | 61 | median=0 | -2 | 2.4 |
| 577 | `ss_diff_ss_roll_high_claims` | float64 | 29.9% | 70.1% | 86 | median=0 | -3 | 2.8 |
| 578 | `ss_diff_ss_roll_dispossessed` | float64 | 29.9% | 70.1% | 320 | median=0 | -9.6 | 10.2 |
| 579 | `ss_diff_ss_roll_avg_shot_xg` | float64 | 29.9% | 70.1% | 2852 | median=0 | -0.144 | 0.141 |
| 580 | `ss_diff_ss_roll_max_shot_xg` | float64 | 29.9% | 70.1% | 3002 | median=0 | -0.525 | 0.719 |
| 581 | `ss_diff_ss_roll_total_xgot` | float64 | 29.9% | 70.1% | 3025 | median=0 | -2.75 | 2.76 |
| 582 | `ss_diff_ss_roll_sm_inside_box_pct` | float64 | 29.9% | 70.1% | 4728 | median=0.000987 | -0.382 | 0.474 |
| 583 | `ss_diff_ss_roll_sm_header_pct` | float64 | 29.9% | 70.1% | 4725 | median=-0.00186 | -0.433 | 0.313 |
| 584 | `ss_diff_ss_roll_sm_open_play_pct` | float64 | 29.9% | 70.1% | 4725 | median=-0.00183 | -0.596 | 0.462 |
| 585 | `ss_diff_ss_roll_sm_set_piece_pct` | float64 | 29.9% | 70.1% | 4730 | median=0.00146 | -0.377 | 0.471 |
| 586 | `ss_diff_ss_roll_sm_counter_pct` | float64 | 29.9% | 70.1% | 4110 | median=0 | -0.267 | 0.258 |
| 587 | `ss_diff_ss_roll_sm_conversion_rate` | float64 | 29.9% | 70.1% | 4711 | median=-0.000437 | -0.41 | 0.271 |
| 588 | `ss_diff_ss_roll_sm_big_chance_pct` | float64 | 29.9% | 70.1% | 3022 | median=0 | -0.301 | 0.294 |
| 589 | `ss_diff_ss_roll_sm_avg_shot_distance` | float64 | 29.9% | 70.1% | 3759 | median=0.006 | -8.22 | 8.24 |
| 590 | `ss_diff_ss_roll_sm_median_shot_distance` | float64 | 29.9% | 70.1% | 3996 | median=0.004 | -10.8 | 10.3 |
| 591 | `ss_diff_ss_roll_sm_shot_distance_std` | float64 | 29.9% | 70.1% | 3659 | median=0 | -5.14 | 5.95 |
| 592 | `ss_diff_ss_roll_sm_close_range_pct` | float64 | 29.9% | 70.1% | 4592 | median=-0.00056 | -0.275 | 0.26 |
| 593 | `ss_diff_ss_roll_xg_std` | float64 | 29.9% | 70.1% | 2868 | median=0 | -2.48 | 2.34 |
| 594 | `ss_diff_ss_roll_rating_std` | float64 | 29.9% | 70.1% | 4735 | median=-0.00295 | -0.362 | 0.451 |
| 595 | `ss_diff_top2_xg_share` | float64 | 30.4% | 69.6% | 2919 | median=0 | -0.614 | 0.63 |
| 596 | `ss_coverage` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 597 | `ss_xg_coverage` | float64 | 50.1% | 49.9% | 2 | median=0 | 0 | 1 |
| 598 | `home_ss_idx_attacking` | float64 | 30.0% | 70.0% | 4741 | median=0.553 | 0.0328 | 1 |
| 599 | `home_ss_idx_chance_creation` | float64 | 30.0% | 70.0% | 4726 | median=0.543 | 0.0976 | 1 |
| 600 | `home_ss_idx_shot_quality` | float64 | 30.0% | 70.0% | 4740 | median=0.582 | 0.0133 | 1 |
| 601 | `home_ss_idx_defense` | float64 | 30.0% | 70.0% | 4731 | median=0.51 | 0.068 | 1 |
| 602 | `home_ss_idx_pressing` | float64 | 30.0% | 70.0% | 4715 | median=0.412 | 0.0119 | 1 |
| 603 | `home_ss_idx_passing` | float64 | 30.0% | 70.0% | 4730 | median=0.563 | 0.133 | 1 |
| 604 | `home_ss_idx_gk` | float64 | 30.0% | 70.0% | 4711 | median=0.545 | 0.219 | 1 |
| 605 | `home_ss_idx_duels` | float64 | 30.0% | 70.0% | 4742 | median=0.477 | 0.0193 | 1 |
| 606 | `home_ss_idx_match_control` | float64 | 30.0% | 70.0% | 4732 | median=0.541 | 0.135 | 1 |
| 607 | `home_ss_idx_set_pieces` | float64 | 30.0% | 70.0% | 4685 | median=0.498 | 0.000329 | 1 |
| 608 | `away_ss_idx_attacking` | float64 | 30.0% | 70.0% | 4734 | median=0.554 | 0.0341 | 1 |
| 609 | `away_ss_idx_chance_creation` | float64 | 30.0% | 70.0% | 4731 | median=0.546 | 0.0958 | 1 |
| 610 | `away_ss_idx_shot_quality` | float64 | 30.0% | 70.0% | 4741 | median=0.582 | 0.0245 | 1 |
| 611 | `away_ss_idx_defense` | float64 | 30.0% | 70.0% | 4731 | median=0.508 | 0.0664 | 1 |
| 612 | `away_ss_idx_pressing` | float64 | 30.0% | 70.0% | 4732 | median=0.415 | 0.0169 | 1 |
| 613 | `away_ss_idx_passing` | float64 | 30.0% | 70.0% | 4726 | median=0.565 | 0.127 | 1 |
| 614 | `away_ss_idx_gk` | float64 | 30.0% | 70.0% | 4723 | median=0.549 | 0.212 | 1 |
| 615 | `away_ss_idx_duels` | float64 | 30.0% | 70.0% | 4740 | median=0.481 | 0.024 | 1 |
| 616 | `away_ss_idx_match_control` | float64 | 30.0% | 70.0% | 4726 | median=0.541 | 0.12 | 1 |
| 617 | `away_ss_idx_set_pieces` | float64 | 30.0% | 70.0% | 4697 | median=0.504 | 0.000701 | 1 |
| 618 | `ss_idx_diff_attacking` | float64 | 29.9% | 70.1% | 4728 | median=-0.000884 | -0.85 | 0.85 |
| 619 | `ss_idx_diff_chance_creation` | float64 | 29.9% | 70.1% | 4730 | median=-0.00157 | -0.731 | 0.688 |
| 620 | `ss_idx_diff_shot_quality` | float64 | 29.9% | 70.1% | 4727 | median=0.0044 | -0.857 | 0.89 |
| 621 | `ss_idx_diff_defense` | float64 | 29.9% | 70.1% | 4727 | median=-0.0025 | -0.681 | 0.68 |
| 622 | `ss_idx_diff_pressing` | float64 | 29.9% | 70.1% | 4730 | median=0.000709 | -0.758 | 0.836 |
| 623 | `ss_idx_diff_passing` | float64 | 29.9% | 70.1% | 4724 | median=0.00257 | -0.715 | 0.747 |
| 624 | `ss_idx_diff_gk` | float64 | 29.9% | 70.1% | 4715 | median=0.00155 | -0.735 | 0.686 |
| 625 | `ss_idx_diff_duels` | float64 | 29.9% | 70.1% | 4734 | median=-0.00177 | -0.712 | 0.795 |
| 626 | `ss_idx_diff_match_control` | float64 | 29.9% | 70.1% | 4719 | median=-0.00148 | -0.69 | 0.685 |
| 627 | `ss_idx_diff_set_pieces` | float64 | 29.9% | 70.1% | 4717 | median=-0.0118 | -0.981 | 0.98 |
| 628 | `home_lineup_xg_sum` | float64 | 9.2% | 90.8% | 1434 | median=0.997 | 0.0489 | 5.44 |
| 629 | `home_lineup_xa_sum` | float64 | 9.2% | 90.8% | 1465 | median=0.648 | 0.05 | 3.65 |
| 630 | `home_lineup_rating_mean` | float64 | 9.2% | 90.8% | 181 | median=6.96 | 6.12 | 7.86 |
| 631 | `home_lineup_rotation` | float64 | 9.2% | 90.8% | 10 | median=3 | 0 | 9 |
| 632 | `away_lineup_xg_sum` | float64 | 9.2% | 90.8% | 1423 | median=0.806 | 0 | 4.31 |
| 633 | `away_lineup_xa_sum` | float64 | 9.2% | 90.8% | 1465 | median=0.528 | 0.0158 | 3.03 |
| 634 | `away_lineup_rating_mean` | float64 | 9.2% | 90.8% | 167 | median=6.89 | 6.03 | 7.68 |
| 635 | `away_lineup_rotation` | float64 | 9.2% | 90.8% | 12 | median=3 | 0 | 11 |
| 636 | `lineup_xg_sum_diff` | float64 | 9.2% | 90.8% | 1452 | median=0.163 | -3.38 | 5.1 |
| 637 | `lineup_xa_sum_diff` | float64 | 9.2% | 90.8% | 1465 | median=0.105 | -2.56 | 3.48 |
| 638 | `lineup_rating_mean_diff` | float64 | 9.2% | 90.8% | 545 | median=0.0636 | -1.54 | 1.79 |
| 639 | `home_us_top11_xg90_sum` | float64 | 49.6% | 50.4% | 420 | median=3.17 | 1.39 | 7.81 |
| 640 | `home_us_squad_depth` | float64 | 49.6% | 50.4% | 16 | median=22 | 16 | 33 |
| 641 | `home_us_xg_concentration` | float64 | 49.6% | 50.4% | 420 | median=0.212 | 0.127 | 0.522 |
| 642 | `away_us_top11_xg90_sum` | float64 | 49.6% | 50.4% | 420 | median=3.17 | 1.39 | 7.81 |
| 643 | `away_us_squad_depth` | float64 | 49.6% | 50.4% | 16 | median=22 | 16 | 33 |
| 644 | `away_us_xg_concentration` | float64 | 49.6% | 50.4% | 420 | median=0.212 | 0.127 | 0.522 |
| 645 | `us_top11_xg90_sum_diff` | float64 | 49.6% | 50.4% | 7859 | median=-0.000917 | -6.15 | 6.15 |
| 646 | `us_squad_depth_diff` | float64 | 49.6% | 50.4% | 33 | median=0 | -16 | 16 |
| 647 | `h2h_home_dominance` | float64 | 81.9% | 18.1% | 65 | median=0.5 | 0 | 1 |
| 648 | `h2h_goals_diff` | float64 | 81.9% | 18.1% | 197 | median=0 | -5 | 5 |
| 649 | `h2h_btts_rate` | float64 | 81.9% | 18.1% | 33 | median=0.5 | 0 | 1 |
| 650 | `h2h_over25_rate` | float64 | 81.9% | 18.1% | 33 | median=0.5 | 0 | 1 |
| 651 | `h2h_meetings` | float64 | 81.9% | 18.1% | 9 | median=10 | 2 | 10 |
| 652 | `kickoff_hour` | float64 | 50.1% | 49.9% | 9 | median=15 | 10 | 19 |
| 653 | `is_night_match` | float64 | 50.1% | 49.9% | 2 | median=0 | 0 | 1 |
| 654 | `day_of_week` | int32 | 100.0% | 0.0% | 7 | median=5 | 0 | 6 |
| 655 | `is_weekend` | int64 | 100.0% | 0.0% | 2 | median=1 | 0 | 1 |
| 656 | `is_midweek` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 657 | `season_phase` | float64 | 100.0% | 0.0% | 5 | median=3 | 1 | 5 |
| 658 | `home_post_intl_break` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 659 | `away_post_intl_break` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 660 | `is_no_crowd_match` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 661 | `five_sub_era` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 662 | `var_era` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 663 | `congestion_asymmetry` | float64 | 100.0% | 0.0% | 37 | median=0 | -18 | 18 |
| 664 | `any_team_fatigued` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 665 | `matchweek_avg_goals` | float64 | 99.7% | 0.3% | 9507 | median=2.68 | 0 | 7 |
| 666 | `home_is_promoted` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 667 | `away_is_promoted` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 668 | `promoted_derby` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 669 | `promoted_vs_established` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 670 | `home_seasons_since_topflight` | int64 | 100.0% | 0.0% | 10 | median=0 | 0 | 12 |
| 671 | `away_seasons_since_topflight` | int64 | 100.0% | 0.0% | 10 | median=0 | 0 | 12 |
| 672 | `home_previously_in_league` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 673 | `away_previously_in_league` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 674 | `home_captain_played` | float64 | 50.1% | 49.9% | 2 | median=0 | 0 | 1 |
| 675 | `away_captain_played` | float64 | 50.1% | 49.9% | 2 | median=0 | 0 | 1 |
| 676 | `home_captain_consistency` | float64 | 50.1% | 49.9% | 6 | median=0 | 0 | 1 |
| 677 | `away_captain_consistency` | float64 | 50.1% | 49.9% | 6 | median=0 | 0 | 1 |
| 678 | `home_captain_effect` | float64 | 50.1% | 49.9% | 457 | median=0 | -0.5 | 0.5 |
| 679 | `away_captain_effect` | float64 | 50.1% | 49.9% | 455 | median=0 | -0.5 | 0.667 |
| 680 | `home_ct_first_card_min_r5` | float64 | 50.1% | 49.9% | 331 | median=0 | 0 | 84.2 |
| 681 | `home_ct_cards_before_30_r5` | float64 | 50.1% | 49.9% | 14 | median=0 | 0 | 1.6 |
| 682 | `home_ct_cards_after_75_r5` | float64 | 50.1% | 49.9% | 18 | median=0 | 0 | 2.4 |
| 683 | `home_ct_card_timing_spread_r5` | float64 | 50.1% | 49.9% | 1290 | median=0 | 0 | 31.5 |
| 684 | `home_ct_first_goal_min_r5` | float64 | 50.1% | 49.9% | 336 | median=0 | 0 | 90 |
| 685 | `home_ct_goals_first_15_r5` | float64 | 50.1% | 49.9% | 10 | median=0 | 0 | 1 |
| 686 | `home_ct_goals_last_15_r5` | float64 | 50.1% | 49.9% | 11 | median=0 | 0 | 1.2 |
| 687 | `home_ct_total_goals_scored_r5` | float64 | 50.1% | 49.9% | 28 | median=0 | 0 | 3.6 |
| 688 | `home_ct_conceded_last_15_r5` | float64 | 50.1% | 49.9% | 12 | median=0 | 0 | 1.2 |
| 689 | `home_ct_total_goals_conceded_r5` | float64 | 50.1% | 49.9% | 28 | median=0 | 0 | 3.2 |
| 690 | `home_ct_first_sub_min_r5` | float64 | 50.1% | 49.9% | 195 | median=0 | 0 | 77.6 |
| 691 | `home_ct_early_sub_r5` | float64 | 50.1% | 49.9% | 11 | median=0 | 0 | 1 |
| 692 | `home_ct_goals_first_15_rate` | float64 | 50.1% | 49.9% | 35 | median=0 | 0 | 1 |
| 693 | `home_ct_goals_last_15_rate` | float64 | 50.1% | 49.9% | 49 | median=0 | 0 | 1 |
| 694 | `home_ct_conceded_last_15_rate` | float64 | 50.1% | 49.9% | 45 | median=0 | 0 | 1 |
| 695 | `away_ct_first_card_min_r5` | float64 | 50.1% | 49.9% | 332 | median=0 | 0 | 84 |
| 696 | `away_ct_cards_before_30_r5` | float64 | 50.1% | 49.9% | 14 | median=0 | 0 | 1.6 |
| 697 | `away_ct_cards_after_75_r5` | float64 | 50.1% | 49.9% | 19 | median=0 | 0 | 2.2 |
| 698 | `away_ct_card_timing_spread_r5` | float64 | 50.1% | 49.9% | 1282 | median=0 | 0 | 30.3 |
| 699 | `away_ct_first_goal_min_r5` | float64 | 50.1% | 49.9% | 329 | median=0 | 0 | 90 |
| 700 | `away_ct_goals_first_15_r5` | float64 | 50.1% | 49.9% | 10 | median=0 | 0 | 1 |
| 701 | `away_ct_goals_last_15_r5` | float64 | 50.1% | 49.9% | 13 | median=0 | 0 | 1.4 |
| 702 | `away_ct_total_goals_scored_r5` | float64 | 50.1% | 49.9% | 32 | median=0 | 0 | 3.8 |
| 703 | `away_ct_conceded_last_15_r5` | float64 | 50.1% | 49.9% | 13 | median=0 | 0 | 1.4 |
| 704 | `away_ct_total_goals_conceded_r5` | float64 | 50.1% | 49.9% | 30 | median=0 | 0 | 3.4 |
| 705 | `away_ct_first_sub_min_r5` | float64 | 50.1% | 49.9% | 192 | median=0 | 0 | 77 |
| 706 | `away_ct_early_sub_r5` | float64 | 50.1% | 49.9% | 11 | median=0 | 0 | 1 |
| 707 | `away_ct_goals_first_15_rate` | float64 | 50.1% | 49.9% | 39 | median=0 | 0 | 1 |
| 708 | `away_ct_goals_last_15_rate` | float64 | 50.1% | 49.9% | 54 | median=0 | 0 | 1 |
| 709 | `away_ct_conceded_last_15_rate` | float64 | 50.1% | 49.9% | 49 | median=0 | 0 | 1 |
| 710 | `home_fb_roll_goals` | float64 | 25.0% | 75.0% | 35 | median=1.2 | 0 | 3.8 |
| 711 | `home_fb_roll_assists` | float64 | 25.0% | 75.0% | 30 | median=0.8 | 0 | 13.8 |
| 712 | `home_fb_roll_shots` | float64 | 25.0% | 75.0% | 131 | median=12.8 | 4 | 65.2 |
| 713 | `home_fb_roll_shots_on_target` | float64 | 25.0% | 75.0% | 72 | median=4.2 | 0.8 | 17.6 |
| 714 | `home_fb_roll_shot_accuracy` | float64 | 25.0% | 75.0% | 3458 | median=0.339 | 0.0836 | 0.65 |
| 715 | `home_fb_roll_cards_yellow` | float64 | 25.0% | 75.0% | 37 | median=2.2 | 0.4 | 5.4 |
| 716 | `home_fb_roll_cards_red` | float64 | 25.0% | 75.0% | 9 | median=0 | 0 | 1 |
| 717 | `home_fb_roll_fouls` | float64 | 25.0% | 75.0% | 122 | median=12.4 | 0 | 20.8 |
| 718 | `home_fb_roll_fouled` | float64 | 25.0% | 75.0% | 115 | median=11.6 | 0 | 20.8 |
| 719 | `home_fb_roll_interceptions` | float64 | 25.0% | 75.0% | 115 | median=8.8 | 3.2 | 44 |
| 720 | `home_fb_roll_tackles_won` | float64 | 25.0% | 75.0% | 104 | median=9.2 | 0 | 18 |
| 721 | `home_fb_roll_blocks` | float64 | 21.1% | 78.9% | 73 | median=0 | 0 | 50.2 |
| 722 | `home_fb_roll_squad_players` | float64 | 25.0% | 75.0% | 31 | median=15.4 | 12.6 | 16.4 |
| 723 | `home_fb_roll_avg_minutes_per_player` | float64 | 25.0% | 75.0% | 884 | median=64.5 | 60.4 | 238 |
| 724 | `home_fb_roll_age_years` | float64 | 25.0% | 75.0% | 2698 | median=26.4 | 22.9 | 31 |
| 725 | `home_fb_roll_xg` | float64 | 21.1% | 78.9% | 148 | median=0 | 0 | 2.86 |
| 726 | `home_fb_roll_npxg` | float64 | 21.1% | 78.9% | 143 | median=0 | 0 | 2.7 |
| 727 | `home_fb_roll_xg_assist` | float64 | 21.1% | 78.9% | 124 | median=0 | 0 | 3.12 |
| 728 | `home_fb_roll_sca` | float64 | 21.1% | 78.9% | 135 | median=0 | 0 | 65.4 |
| 729 | `home_fb_roll_gca` | float64 | 21.1% | 78.9% | 32 | median=0 | 0 | 15.4 |
| 730 | `home_fb_roll_pass_accuracy` | float64 | 21.1% | 78.9% | 413 | median=0 | 0 | 0.893 |
| 731 | `home_fb_roll_progressive_passes` | float64 | 21.1% | 78.9% | 195 | median=0 | 0 | 160 |
| 732 | `home_fb_roll_xg_per_shot` | float64 | 21.1% | 78.9% | 404 | median=0 | 0 | 0.185 |
| 733 | `home_fb_roll_tackles` | float64 | 21.1% | 78.9% | 98 | median=0 | 0 | 27.2 |
| 734 | `home_fb_roll_touches` | float64 | 21.1% | 78.9% | 366 | median=0 | 0 | 2.58e+03 |
| 735 | `home_fb_roll_carries` | float64 | 21.1% | 78.9% | 367 | median=0 | 0 | 1.77e+03 |
| 736 | `home_fb_roll_progressive_carries` | float64 | 21.1% | 78.9% | 116 | median=0 | 0 | 98.4 |
| 737 | `home_fb_roll_defense_tackles_won` | float64 | 21.1% | 78.9% | 66 | median=0 | 0 | 14.8 |
| 738 | `home_fb_roll_defense_interceptions` | float64 | 21.1% | 78.9% | 57 | median=0 | 0 | 44 |
| 739 | `home_fb_roll_defense_clearances` | float64 | 21.1% | 78.9% | 135 | median=0 | 0 | 108 |
| 740 | `home_fb_roll_misc_ball_recoveries` | float64 | 21.1% | 78.9% | 144 | median=0 | 0 | 159 |
| 741 | `home_fb_roll_aerial_win_rate` | float64 | 21.1% | 78.9% | 412 | median=0 | 0 | 0.651 |
| 742 | `home_fb_roll_goals_std` | float64 | 25.0% | 75.0% | 469 | median=0.894 | 0 | 4.24 |
| 743 | `away_fb_roll_goals` | float64 | 25.0% | 75.0% | 34 | median=1.4 | 0 | 4 |
| 744 | `away_fb_roll_assists` | float64 | 25.0% | 75.0% | 32 | median=0.8 | 0 | 13.8 |
| 745 | `away_fb_roll_shots` | float64 | 25.0% | 75.0% | 128 | median=13 | 4.6 | 67.2 |
| 746 | `away_fb_roll_shots_on_target` | float64 | 25.0% | 75.0% | 73 | median=4.2 | 1 | 18 |
| 747 | `away_fb_roll_shot_accuracy` | float64 | 25.0% | 75.0% | 3473 | median=0.34 | 0.1 | 0.627 |
| 748 | `away_fb_roll_cards_yellow` | float64 | 25.0% | 75.0% | 39 | median=2.2 | 0.4 | 5.2 |
| 749 | `away_fb_roll_cards_red` | float64 | 25.0% | 75.0% | 8 | median=0 | 0 | 0.8 |
| 750 | `away_fb_roll_fouls` | float64 | 25.0% | 75.0% | 120 | median=12.4 | 0 | 21.4 |
| 751 | `away_fb_roll_fouled` | float64 | 25.0% | 75.0% | 121 | median=11.6 | 0 | 22 |
| 752 | `away_fb_roll_interceptions` | float64 | 25.0% | 75.0% | 113 | median=8.8 | 2.5 | 44.2 |
| 753 | `away_fb_roll_tackles_won` | float64 | 25.0% | 75.0% | 105 | median=9.2 | 0 | 19.6 |
| 754 | `away_fb_roll_blocks` | float64 | 21.0% | 79.0% | 76 | median=0 | 0 | 50.2 |
| 755 | `away_fb_roll_squad_players` | float64 | 25.0% | 75.0% | 29 | median=15.4 | 12.6 | 16.6 |
| 756 | `away_fb_roll_avg_minutes_per_player` | float64 | 25.0% | 75.0% | 874 | median=64.5 | 59.7 | 238 |
| 757 | `away_fb_roll_age_years` | float64 | 25.0% | 75.0% | 2670 | median=26.4 | 22.8 | 31.2 |
| 758 | `away_fb_roll_xg` | float64 | 21.0% | 79.0% | 157 | median=0 | 0 | 2.88 |
| 759 | `away_fb_roll_npxg` | float64 | 21.0% | 79.0% | 145 | median=0 | 0 | 2.76 |
| 760 | `away_fb_roll_xg_assist` | float64 | 21.0% | 79.0% | 127 | median=0 | 0 | 3.2 |
| 761 | `away_fb_roll_sca` | float64 | 21.0% | 79.0% | 127 | median=0 | 0 | 68.6 |
| 762 | `away_fb_roll_gca` | float64 | 21.0% | 79.0% | 35 | median=0 | 0 | 15.6 |
| 763 | `away_fb_roll_pass_accuracy` | float64 | 21.0% | 79.0% | 413 | median=0 | 0 | 0.895 |
| 764 | `away_fb_roll_progressive_passes` | float64 | 21.0% | 79.0% | 197 | median=0 | 0 | 165 |
| 765 | `away_fb_roll_xg_per_shot` | float64 | 21.0% | 79.0% | 412 | median=0 | 0 | 0.178 |
| 766 | `away_fb_roll_tackles` | float64 | 21.0% | 79.0% | 98 | median=0 | 0 | 26.8 |
| 767 | `away_fb_roll_touches` | float64 | 21.0% | 79.0% | 373 | median=0 | 0 | 2.61e+03 |
| 768 | `away_fb_roll_carries` | float64 | 21.0% | 79.0% | 362 | median=0 | 0 | 1.8e+03 |
| 769 | `away_fb_roll_progressive_carries` | float64 | 21.0% | 79.0% | 134 | median=0 | 0 | 98.8 |
| 770 | `away_fb_roll_defense_tackles_won` | float64 | 21.0% | 79.0% | 65 | median=0 | 0 | 14.6 |
| 771 | `away_fb_roll_defense_interceptions` | float64 | 21.0% | 79.0% | 59 | median=0 | 0 | 44.2 |
| 772 | `away_fb_roll_defense_clearances` | float64 | 21.0% | 79.0% | 140 | median=0 | 0 | 107 |
| 773 | `away_fb_roll_misc_ball_recoveries` | float64 | 21.0% | 79.0% | 150 | median=0 | 0 | 162 |
| 774 | `away_fb_roll_aerial_win_rate` | float64 | 21.0% | 79.0% | 413 | median=0 | 0 | 0.644 |
| 775 | `away_fb_roll_goals_std` | float64 | 25.0% | 75.0% | 479 | median=1 | 0 | 3.21 |
| 776 | `fb_diff_goals` | float64 | 24.7% | 75.3% | 124 | median=0 | -3 | 3 |
| 777 | `fb_diff_assists` | float64 | 24.7% | 75.3% | 109 | median=0 | -13.6 | 12.8 |
| 778 | `fb_diff_shots` | float64 | 24.7% | 75.3% | 414 | median=-0.2 | -56.4 | 58.6 |
| 779 | `fb_diff_shots_on_target` | float64 | 24.7% | 75.3% | 234 | median=0 | -14.2 | 14.2 |
| 780 | `fb_diff_shot_accuracy` | float64 | 24.7% | 75.3% | 3901 | median=-0.00136 | -0.363 | 0.376 |
| 781 | `fb_diff_cards_yellow` | float64 | 24.7% | 75.3% | 128 | median=0 | -3.4 | 3.4 |
| 782 | `fb_diff_cards_red` | float64 | 24.7% | 75.3% | 25 | median=0 | -0.8 | 1 |
| 783 | `fb_diff_fouls` | float64 | 24.7% | 75.3% | 309 | median=0 | -10 | 12.2 |
| 784 | `fb_diff_fouled` | float64 | 24.7% | 75.3% | 302 | median=0 | -11.6 | 11 |
| 785 | `fb_diff_interceptions` | float64 | 24.7% | 75.3% | 310 | median=0 | -39 | 36.4 |
| 786 | `fb_diff_tackles_won` | float64 | 24.7% | 75.3% | 269 | median=0 | -11 | 9.2 |
| 787 | `fb_diff_blocks` | float64 | 21.0% | 79.0% | 153 | median=0 | -41.4 | 40 |
| 788 | `fb_diff_squad_players` | float64 | 24.7% | 75.3% | 52 | median=0 | -2.4 | 2.4 |
| 789 | `fb_diff_avg_minutes_per_player` | float64 | 24.7% | 75.3% | 2239 | median=0 | -173 | 176 |
| 790 | `fb_diff_age_years` | float64 | 24.7% | 75.3% | 3705 | median=-0.0167 | -5.78 | 5.82 |
| 791 | `fb_diff_xg` | float64 | 21.0% | 79.0% | 278 | median=0 | -1.78 | 2 |
| 792 | `fb_diff_npxg` | float64 | 21.0% | 79.0% | 249 | median=0 | -1.92 | 2.16 |
| 793 | `fb_diff_xg_assist` | float64 | 21.0% | 79.0% | 250 | median=0 | -2.56 | 2.54 |
| 794 | `fb_diff_sca` | float64 | 21.0% | 79.0% | 233 | median=0 | -48.8 | 53.6 |
| 795 | `fb_diff_gca` | float64 | 21.0% | 79.0% | 102 | median=0 | -15 | 13.4 |
| 796 | `fb_diff_pass_accuracy` | float64 | 21.0% | 79.0% | 417 | median=0 | -0.856 | 0.876 |
| 797 | `fb_diff_progressive_passes` | float64 | 21.0% | 79.0% | 290 | median=0 | -132 | 128 |
| 798 | `fb_diff_xg_per_shot` | float64 | 21.0% | 79.0% | 417 | median=0 | -0.103 | 0.117 |
| 799 | `fb_diff_tackles` | float64 | 21.0% | 79.0% | 185 | median=0 | -12.8 | 16.2 |
| 800 | `fb_diff_touches` | float64 | 21.0% | 79.0% | 390 | median=0 | -2.02e+03 | 2.03e+03 |
| 801 | `fb_diff_carries` | float64 | 21.0% | 79.0% | 392 | median=0 | -1.45e+03 | 1.5e+03 |
| 802 | `fb_diff_progressive_carries` | float64 | 21.0% | 79.0% | 240 | median=0 | -83 | 83.6 |
| 803 | `fb_diff_defense_tackles_won` | float64 | 21.0% | 79.0% | 146 | median=0 | -7.2 | 9 |
| 804 | `fb_diff_defense_interceptions` | float64 | 21.0% | 79.0% | 133 | median=0 | -39 | 36.4 |
| 805 | `fb_diff_defense_clearances` | float64 | 21.0% | 79.0% | 235 | median=0 | -79.4 | 91 |
| 806 | `fb_diff_misc_ball_recoveries` | float64 | 21.0% | 79.0% | 209 | median=0 | -124 | 123 |
| 807 | `fb_diff_aerial_win_rate` | float64 | 21.0% | 79.0% | 417 | median=0 | -0.491 | 0.589 |
| 808 | `fb_diff_goals_std` | float64 | 24.7% | 75.3% | 2652 | median=0 | -2.49 | 4.24 |
| 809 | `fb_coverage` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 810 | `home_league_pos` | int64 | 100.0% | 0.0% | 20 | median=11 | 1 | 20 |
| 811 | `home_league_points` | int64 | 100.0% | 0.0% | 95 | median=22 | 0 | 99 |
| 812 | `home_league_gd` | int64 | 100.0% | 0.0% | 130 | median=-1 | -66 | 76 |
| 813 | `home_league_goals_for` | int64 | 100.0% | 0.0% | 102 | median=22 | 0 | 102 |
| 814 | `home_league_wins` | int64 | 100.0% | 0.0% | 32 | median=6 | 0 | 32 |
| 815 | `home_league_draws` | int64 | 100.0% | 0.0% | 19 | median=4 | 0 | 18 |
| 816 | `home_league_losses` | int64 | 100.0% | 0.0% | 30 | median=6 | 0 | 29 |
| 817 | `away_league_pos` | int64 | 100.0% | 0.0% | 20 | median=11 | 1 | 20 |
| 818 | `away_league_points` | int64 | 100.0% | 0.0% | 96 | median=23 | 0 | 97 |
| 819 | `away_league_gd` | int64 | 100.0% | 0.0% | 134 | median=0 | -65 | 78 |
| 820 | `away_league_goals_for` | int64 | 100.0% | 0.0% | 98 | median=23 | 0 | 105 |
| 821 | `away_league_wins` | int64 | 100.0% | 0.0% | 32 | median=6 | 0 | 31 |
| 822 | `away_league_draws` | int64 | 100.0% | 0.0% | 20 | median=4 | 0 | 19 |
| 823 | `away_league_losses` | int64 | 100.0% | 0.0% | 29 | median=6 | 0 | 28 |
| 824 | `home_in_relegation_zone` | float64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 825 | `home_in_cl_zone` | float64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 826 | `home_in_el_zone` | float64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 827 | `home_in_title_race` | float64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 828 | `away_in_relegation_zone` | float64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 829 | `away_in_cl_zone` | float64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 830 | `away_in_el_zone` | float64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 831 | `away_in_title_race` | float64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 832 | `league_position_diff` | int64 | 100.0% | 0.0% | 38 | median=-1 | -19 | 19 |
| 833 | `home_position_momentum` | float64 | 97.2% | 2.8% | 39 | median=0 | -19 | 19 |
| 834 | `away_position_momentum` | float64 | 97.2% | 2.8% | 37 | median=0 | -18 | 18 |
| 835 | `position_momentum_diff` | float64 | 100.0% | 0.0% | 57 | median=0 | -28 | 28 |
| 836 | `home_position_zone` | int64 | 100.0% | 0.0% | 5 | median=4 | 1 | 5 |
| 837 | `away_position_zone` | int64 | 100.0% | 0.0% | 5 | median=4 | 1 | 5 |
| 838 | `home_points_to_cl_zone` | float64 | 100.0% | 0.0% | 690 | median=-6 | -352 | 48 |
| 839 | `home_points_to_relegation` | float64 | 100.0% | 0.0% | 769 | median=6 | -44 | 368 |
| 840 | `away_points_to_cl_zone` | float64 | 100.0% | 0.0% | 683 | median=-6 | -330 | 69 |
| 841 | `away_points_to_relegation` | float64 | 100.0% | 0.0% | 775 | median=6.2 | -22 | 391 |
| 842 | `home_manager_tenure` | float64 | 43.2% | 56.8% | 314 | median=30 | 1 | 342 |
| 843 | `home_manager_is_new` | float64 | 43.2% | 56.8% | 2 | median=0 | 0 | 1 |
| 844 | `home_manager_changed` | float64 | 43.0% | 57.0% | 2 | median=0 | 0 | 1 |
| 845 | `away_manager_tenure` | float64 | 43.2% | 56.8% | 313 | median=30 | 1 | 341 |
| 846 | `away_manager_is_new` | float64 | 43.2% | 56.8% | 2 | median=0 | 0 | 1 |
| 847 | `away_manager_changed` | float64 | 42.9% | 57.1% | 2 | median=0 | 0 | 1 |
| 848 | `manager_h2h_home_winrate` | float64 | 100.0% | 0.0% | 5 | median=0.5 | 0 | 1 |
| 849 | `manager_h2h_matches` | int64 | 100.0% | 0.0% | 4 | median=0 | 0 | 3 |
| 850 | `manager_h2h_confidence` | float64 | 100.0% | 0.0% | 3 | median=0 | 0 | 0.5 |
| 851 | `home_short_rest` | float64 | 97.4% | 2.6% | 2 | median=0 | 0 | 1 |
| 852 | `away_short_rest` | float64 | 97.3% | 2.7% | 2 | median=0 | 0 | 1 |
| 853 | `home_congestion_3` | int64 | 100.0% | 0.0% | 6 | median=2 | 0 | 5 |
| 854 | `away_congestion_3` | int64 | 100.0% | 0.0% | 5 | median=2 | 0 | 4 |
| 855 | `home_congestion_5` | int64 | 100.0% | 0.0% | 10 | median=4 | 0 | 9 |
| 856 | `away_congestion_5` | int64 | 100.0% | 0.0% | 10 | median=4 | 0 | 9 |
| 857 | `rest_advantage` | float64 | 100.0% | 0.0% | 11 | median=0 | -5 | 5 |
| 858 | `home_suspended_count` | float64 | 50.1% | 49.9% | 8 | median=0 | 0 | 7 |
| 859 | `away_suspended_count` | float64 | 50.1% | 49.9% | 8 | median=0 | 0 | 7 |
| 860 | `home_at_risk_count` | float64 | 50.1% | 49.9% | 9 | median=0 | 0 | 8 |
| 861 | `away_at_risk_count` | float64 | 50.1% | 49.9% | 9 | median=0 | 0 | 8 |
| 862 | `home_total_yellows` | int64 | 100.0% | 0.0% | 110 | median=24 | 0 | 111 |
| 863 | `away_total_yellows` | int64 | 100.0% | 0.0% | 110 | median=23 | 0 | 110 |
| 864 | `home_formation_flexibility` | float64 | 50.1% | 49.9% | 20 | median=0.611 | 0.462 | 0.949 |
| 865 | `away_formation_flexibility` | float64 | 50.1% | 49.9% | 20 | median=0.611 | 0.462 | 0.949 |
| 866 | `formation_matchup_home_rate` | float64 | 50.1% | 49.9% | 18 | median=0.436 | 0 | 0.714 |
| 867 | `formation_matchup_draw_rate` | float64 | 50.1% | 49.9% | 20 | median=0.359 | 0.1 | 0.474 |
| 868 | `formation_total_advantage` | float64 | 50.1% | 49.9% | 21 | median=-0.014 | -0.45 | 0.264 |
| 869 | `formation_confidence` | object | 50.1% | 49.9% | 2 | 'high' | — | — |
| 870 | `formation_width_mismatch` | float64 | 50.1% | 49.9% | 2 | median=0 | 0 | 1 |
| 871 | `home_ppg_pace` | float64 | 100.0% | 0.0% | 1000 | median=1.18 | 0 | 3 |
| 872 | `away_ppg_pace` | float64 | 100.0% | 0.0% | 1014 | median=1.2 | 0 | 3 |
| 873 | `form_diff` | float64 | 99.6% | 0.4% | 31 | median=0 | -15 | 15 |
| 874 | `momentum_diff` | float64 | 99.6% | 0.4% | 35 | median=0 | -18 | 17 |
| 875 | `attack_strength_diff` | float64 | 100.0% | 0.0% | 15365 | median=0 | -4.35 | 3.61 |
| 876 | `defense_strength_diff` | float64 | 100.0% | 0.0% | 15358 | median=0 | -3.66 | 5.11 |
| 877 | `rolling_gd_diff` | float64 | 99.6% | 0.4% | 216 | median=0 | -6.8 | 6.8 |
| 878 | `rolling_goals_diff` | float64 | 97.3% | 2.7% | 193 | median=0 | -6 | 6 |
| 879 | `elo_diff_log` | float64 | 100.0% | 0.0% | 6495 | median=0.31 | -6.21 | 6.17 |
| 880 | `attack_strength_diff_log` | float64 | 100.0% | 0.0% | 9414 | median=0 | -1.68 | 1.53 |
| 881 | `home_adj_attack_5_sqrt` | float64 | 100.0% | 0.0% | 8675 | median=1.11 | 0 | 4.69 |
| 882 | `home_adj_attack_10_sqrt` | float64 | 100.0% | 0.0% | 8236 | median=1.12 | 0 | 4.58 |
| 883 | `away_adj_attack_5_sqrt` | float64 | 100.0% | 0.0% | 8657 | median=1.12 | 0 | 4.48 |
| 884 | `away_adj_attack_10_sqrt` | float64 | 100.0% | 0.0% | 8203 | median=1.12 | 0 | 4.85 |
| 885 | `home_form_points_5_sq` | float64 | 100.0% | 0.0% | 15 | median=0.218 | 0 | 1 |
| 886 | `away_form_points_5_sq` | float64 | 100.0% | 0.0% | 15 | median=0.218 | 0 | 1 |
| 887 | `home_elo_momentum` | float64 | 94.4% | 5.6% | 2250 | median=-1.4 | -276 | 179 |
| 888 | `away_elo_momentum` | float64 | 94.4% | 5.6% | 2246 | median=-1.3 | -261 | 194 |
| 889 | `elo_momentum_diff` | float64 | 100.0% | 0.0% | 2995 | median=0 | -325 | 369 |
| 890 | `elo_form_blend_diff` | float64 | 100.0% | 0.0% | 4726 | median=0 | -404 | 408 |
| 891 | `elo_form_disagreement` | float64 | 100.0% | 0.0% | 1390 | median=-1.2 | -97.4 | 90.3 |
| 892 | `poisson_home_xg` | float64 | 100.0% | 0.0% | 2834 | median=1.24 | 0.1 | 5 |
| 893 | `poisson_away_xg` | float64 | 100.0% | 0.0% | 2845 | median=1.26 | 0.1 | 5 |
| 894 | `poisson_prob_H` | float64 | 100.0% | 0.0% | 6825 | median=0.361 | 0.0008 | 0.932 |
| 895 | `poisson_prob_D` | float64 | 100.0% | 0.0% | 3246 | median=0.243 | 0.0095 | 0.827 |
| 896 | `poisson_prob_A` | float64 | 100.0% | 0.0% | 6912 | median=0.37 | 0.0008 | 0.932 |
| 897 | `is_derby` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 898 | `derby_intensity` | int64 | 100.0% | 0.0% | 4 | median=0 | 0 | 3 |
| 899 | `derby_home_advantage_boost` | float64 | 100.0% | 0.0% | 3 | median=0 | -0.05 | 0 |
| 900 | `total_xg_expected` | float64 | 100.0% | 0.0% | 3361 | median=2.65 | 0.2 | 9.95 |
| 901 | `combined_goals_per_game` | float64 | 100.0% | 0.0% | 107 | median=1.3 | 0 | 4.17 |
| 902 | `combined_goals_conceded` | float64 | 100.0% | 0.0% | 105 | median=1.3 | 0 | 5 |
| 903 | `attack_mismatch_score` | float64 | 100.0% | 0.0% | 2642 | median=0.956 | 0 | 35.2 |
| 904 | `defense_mismatch_score` | float64 | 100.0% | 0.0% | 2685 | median=0.956 | 0 | 40.8 |
| 905 | `scoring_variance` | float64 | 100.0% | 0.0% | 636 | median=1 | 0 | 2.87 |
| 906 | `home_scoring_rate_5` | float64 | 100.0% | 0.0% | 46 | median=0.699 | 0 | 0.998 |
| 907 | `away_scoring_rate_5` | float64 | 100.0% | 0.0% | 49 | median=0.699 | 0 | 0.998 |
| 908 | `btts_probability_naive` | float64 | 100.0% | 0.0% | 370 | median=0.476 | 0 | 0.969 |
| 909 | `poisson_over_1_5` | float64 | 100.0% | 0.0% | 3237 | median=0.741 | 0.0175 | 1 |
| 910 | `poisson_over_2_5` | float64 | 100.0% | 0.0% | 3349 | median=0.493 | 0.0011 | 0.997 |
| 911 | `poisson_over_3_5` | float64 | 100.0% | 0.0% | 3313 | median=0.274 | 0.0001 | 0.989 |
| 912 | `poisson_btts` | float64 | 100.0% | 0.0% | 5245 | median=0.472 | 0.0091 | 0.986 |
| 913 | `clean_sheet_matchup` | float64 | 100.0% | 0.0% | 33 | median=0.3 | 0 | 1 |
| 914 | `league_home_win_rate` | float64 | 100.0% | 0.0% | 14694 | median=0.465 | 0 | 1 |
| 915 | `league_draw_rate` | float64 | 100.0% | 0.0% | 15051 | median=0.256 | 0 | 1 |
| 916 | `league_avg_goals` | float64 | 100.0% | 0.0% | 15463 | median=2.64 | 1.2 | 4 |
| 917 | `home_draw_tendency_10` | float64 | 99.4% | 0.6% | 28 | median=0.2 | 0 | 1 |
| 918 | `home_draw_tendency_5` | float64 | 99.4% | 0.6% | 11 | median=0.2 | 0 | 1 |
| 919 | `away_draw_tendency_10` | float64 | 99.4% | 0.6% | 26 | median=0.2 | 0 | 1 |
| 920 | `away_draw_tendency_5` | float64 | 99.4% | 0.6% | 11 | median=0.2 | 0 | 1 |
| 921 | `matchup_competitiveness` | float64 | 100.0% | 0.0% | 15803 | median=0.814 | 0.447 | 1 |
| 922 | `both_defenses_strong` | float64 | 100.0% | 0.0% | 13868 | median=0.867 | 0 | 3.84 |
| 923 | `both_attacks_weak` | float64 | 100.0% | 0.0% | 4763 | median=0 | 0 | 1 |
| 924 | `combined_draw_tendency` | float64 | 100.0% | 0.0% | 147 | median=0.25 | 0 | 1 |
| 925 | `defense_similarity` | float64 | 100.0% | 0.0% | 15347 | median=0.787 | 0.164 | 1 |
| 926 | `travel_distance_km` | float64 | 100.0% | 0.0% | 807 | median=168 | 0 | 1.01e+03 |
| 927 | `altitude_diff` | float64 | 100.0% | 0.0% | 349 | median=0 | -351 | 351 |
| 928 | `home_stadium_capacity` | float64 | 90.0% | 10.0% | 62 | median=3.92e+04 | 1.04e+04 | 7.59e+04 |
| 929 | `long_travel` | float64 | 50.1% | 49.9% | 2 | median=0 | 0 | 1 |
| 930 | `altitude_advantage` | float64 | 50.1% | 49.9% | 2 | median=0 | 0 | 1 |
| 931 | `weather_temperature_2m_max` | float64 | 58.9% | 41.1% | 362 | median=14.9 | -6.1 | 38.1 |
| 932 | `weather_temperature_2m_min` | float64 | 58.9% | 41.1% | 318 | median=7.6 | -18.4 | 25.4 |
| 933 | `weather_temperature_2m_mean` | float64 | 58.9% | 41.1% | 337 | median=11 | -11.3 | 30.3 |
| 934 | `weather_apparent_temperature_max` | float64 | 58.9% | 41.1% | 433 | median=13 | -9.7 | 40.2 |
| 935 | `weather_apparent_temperature_min` | float64 | 58.9% | 41.1% | 392 | median=5 | -22.6 | 28.8 |
| 936 | `weather_precipitation_sum` | float64 | 58.9% | 41.1% | 401 | median=0.2 | 0 | 100 |
| 937 | `weather_snowfall_sum` | float64 | 58.9% | 41.1% | 81 | median=0 | 0 | 17.3 |
| 938 | `weather_wind_speed_10m_max` | float64 | 58.9% | 41.1% | 443 | median=13.9 | 3.1 | 66.6 |
| 939 | `weather_wind_gusts_10m_max` | float64 | 58.9% | 41.1% | 263 | median=30.2 | 6.8 | 122 |
| 940 | `weather_wind_direction_10m_dominant` | float64 | 58.9% | 41.1% | 361 | median=172 | 0 | 360 |
| 941 | `weather_relative_humidity_2m_mean` | float64 | 58.9% | 41.1% | 68 | median=79 | 28 | 99 |
| 942 | `odds_MaxCH` | float64 | 32.7% | 67.3% | 648 | median=2.4 | 1.07 | 35 |
| 943 | `odds_P_lt_2.5` | float64 | 31.6% | 68.4% | 219 | median=2.07 | 1.41 | 5.06 |
| 944 | `odds_AvgC_lt_2.5` | float64 | 32.7% | 67.3% | 257 | median=2.02 | 1.36 | 5.77 |
| 945 | `odds_MaxAHA` | float64 | 32.1% | 67.9% | 53 | median=1.98 | 1.33 | 22 |
| 946 | `odds_MaxCA` | float64 | 32.7% | 67.3% | 771 | median=3.43 | 1.12 | 58 |
| 947 | `odds_B365CAHA` | float64 | 32.1% | 67.9% | 54 | median=1.96 | 1.27 | 2.25 |
| 948 | `odds_Max_gt_2.5` | float64 | 32.1% | 67.9% | 160 | median=1.85 | 1.21 | 3.04 |
| 949 | `odds_B365CAHH` | float64 | 32.1% | 67.9% | 53 | median=1.95 | 1.63 | 3.55 |
| 950 | `odds_AvgAHH` | float64 | 32.1% | 67.9% | 51 | median=1.92 | 1.71 | 3.61 |
| 951 | `odds_PC_lt_2.5` | float64 | 31.7% | 68.3% | 239 | median=2.06 | 1.37 | 6.72 |
| 952 | `odds_MaxC_gt_2.5` | float64 | 32.1% | 67.9% | 180 | median=1.89 | 1.16 | 3.27 |
| 953 | `odds_Max_lt_2.5` | float64 | 32.1% | 67.9% | 218 | median=2.11 | 1.45 | 5.06 |
| 954 | `odds_P_gt_2.5` | float64 | 31.6% | 68.4% | 158 | median=1.82 | 1.2 | 3.04 |
| 955 | `odds_PC_gt_2.5` | float64 | 31.7% | 68.3% | 183 | median=1.85 | 1.12 | 3.26 |
| 956 | `odds_PCAHA` | float64 | 31.4% | 68.6% | 65 | median=1.96 | 1.3 | 2.38 |
| 957 | `odds_PAHA` | float64 | 31.4% | 68.6% | 50 | median=1.95 | 1.33 | 2.21 |
| 958 | `odds_MaxCD` | float64 | 32.7% | 67.3% | 436 | median=3.87 | 1.75 | 26 |
| 959 | `odds_MaxC_lt_2.5` | float64 | 32.1% | 67.9% | 252 | median=2.14 | 1.4 | 6.82 |
| 960 | `odds_AvgAHA` | float64 | 32.1% | 67.9% | 63 | median=1.92 | 1.29 | 5.37 |
| 961 | `odds_AvgC_gt_2.5` | float64 | 32.7% | 67.3% | 177 | median=1.81 | 1.13 | 3.05 |
| 962 | `odds_AvgCA` | float64 | 32.7% | 67.3% | 1088 | median=3.24 | 1.09 | 40.9 |
| 963 | `odds_PAHH` | float64 | 31.4% | 68.6% | 52 | median=1.96 | 1.74 | 3.65 |
| 964 | `odds_PCAHH` | float64 | 31.4% | 68.6% | 64 | median=1.96 | 1.66 | 3.93 |
| 965 | `odds_AvgCH` | float64 | 32.7% | 67.3% | 731 | median=2.3 | 1.04 | 26 |
| 966 | `odds_AvgCD` | float64 | 32.7% | 67.3% | 548 | median=3.7 | 1.72 | 19.4 |
| 967 | `odds_MaxAHH` | float64 | 32.1% | 67.9% | 54 | median=1.98 | 1.75 | 3.75 |
| 968 | `pinnacle_home_prob` | float64 | 66.2% | 33.8% | 10245 | median=0.431 | 0.0413 | 0.925 |
| 969 | `pinnacle_draw_prob` | float64 | 66.2% | 33.8% | 10145 | median=0.261 | 0.0516 | 0.443 |
| 970 | `pinnacle_away_prob` | float64 | 66.2% | 33.8% | 10264 | median=0.282 | 0.0227 | 0.853 |
| 971 | `pinnacle_overround` | float64 | 66.2% | 33.8% | 10046 | median=1.02 | 1.01 | 1.06 |
| 972 | `market_draw_prob` | float64 | 99.8% | 0.2% | 15577 | median=0.271 | 0.0661 | 0.663 |
| 973 | `market_overround` | float64 | 99.8% | 0.2% | 15514 | median=1.05 | 0.994 | 1.17 |
| 974 | `pinnacle_ou_over_prob` | float64 | 31.6% | 68.4% | 998 | median=0.532 | 0.317 | 0.808 |
| 975 | `pinnacle_ou_overround` | float64 | 31.6% | 68.4% | 712 | median=1.03 | 1.02 | 1.07 |
| 976 | `market_ou_over_prob` | float64 | 32.7% | 67.3% | 1094 | median=0.531 | 0.33 | 0.791 |
| 977 | `market_ou_overround` | float64 | 32.7% | 67.3% | 760 | median=1.05 | 1.04 | 1.1 |
| 978 | `sharp_soft_home_div` | float64 | 66.2% | 33.8% | 475 | median=0.0021 | -0.0513 | 0.0485 |
| 979 | `sharp_soft_draw_div` | float64 | 66.2% | 33.8% | 344 | median=-0.0014 | -0.0235 | 0.0265 |
| 980 | `sharp_soft_away_div` | float64 | 66.2% | 33.8% | 460 | median=-0.0017 | -0.0422 | 0.0748 |
| 981 | `sharp_soft_ou_div` | float64 | 31.6% | 68.4% | 371 | median=0.0001 | -0.0627 | 0.0355 |
| 982 | `ah_line_abs` | float64 | 32.1% | 67.9% | 13 | median=0.5 | 0 | 3 |
| 983 | `odds_home_fav` | float64 | 99.8% | 0.2% | 2 | median=1 | 0 | 1 |
| 984 | `odds_consistency` | float64 | 66.2% | 33.8% | 745 | median=0.988 | 0.67 | 1 |
| 985 | `ou_consistency` | float64 | 31.6% | 68.4% | 366 | median=0.992 | 0.909 | 1 |
| 986 | `line_vel_pin_home` | float64 | 66.2% | 33.8% | 1442 | median=-0.002 | -0.169 | 0.15 |
| 987 | `line_vel_pin_draw` | float64 | 66.2% | 33.8% | 797 | median=0.0011 | -0.0674 | 0.0816 |
| 988 | `line_vel_pin_away` | float64 | 66.2% | 33.8% | 1407 | median=-0.0001 | -0.165 | 0.159 |
| 989 | `line_vel_mkt_home` | float64 | 32.7% | 67.3% | 1192 | median=-0.0024 | -0.176 | 0.144 |
| 990 | `line_vel_mkt_draw` | float64 | 32.7% | 67.3% | 642 | median=0.0023 | -0.0761 | 0.276 |
| 991 | `line_vel_mkt_away` | float64 | 32.7% | 67.3% | 1144 | median=-0.001 | -0.184 | 0.173 |
| 992 | `line_vel_ou_over` | float64 | 31.5% | 68.5% | 1208 | median=-0.0036 | -0.18 | 0.11 |
| 993 | `line_vel_ou_under` | float64 | 31.5% | 68.5% | 1228 | median=0.0019 | -0.109 | 0.178 |
| 994 | `ah_line_movement` | float64 | 32.1% | 67.9% | 10 | median=0 | -1 | 1.5 |
| 995 | `steam_move_flag` | float64 | 66.2% | 33.8% | 2 | median=0 | 0 | 1 |
| 996 | `home_squad_total_value` | float64 | 41.7% | 58.3% | 353 | median=3.13e+08 | 1e+05 | 1.46e+09 |
| 997 | `home_avg_player_value` | float64 | 41.7% | 58.3% | 353 | median=8.21e+06 | 5.56e+03 | 5.12e+07 |
| 998 | `home_median_player_value` | float64 | 41.7% | 58.3% | 99 | median=4e+06 | 0 | 5e+07 |
| 999 | `home_max_player_value` | float64 | 41.7% | 58.3% | 65 | median=4e+07 | 1e+05 | 2e+08 |
| 1000 | `home_squad_size` | float64 | 41.7% | 58.3% | 42 | median=40 | 18 | 88 |
| 1001 | `away_squad_total_value` | float64 | 41.7% | 58.3% | 353 | median=3.13e+08 | 1e+05 | 1.46e+09 |
| 1002 | `away_avg_player_value` | float64 | 41.7% | 58.3% | 353 | median=8.21e+06 | 5.56e+03 | 5.12e+07 |
| 1003 | `away_median_player_value` | float64 | 41.7% | 58.3% | 99 | median=4e+06 | 0 | 5e+07 |
| 1004 | `away_max_player_value` | float64 | 41.7% | 58.3% | 65 | median=4e+07 | 1e+05 | 2e+08 |
| 1005 | `away_squad_size` | float64 | 41.7% | 58.3% | 42 | median=40 | 18 | 88 |
| 1006 | `squad_value_ratio` | float64 | 41.1% | 58.9% | 6504 | median=0.999 | 0.000151 | 6.64e+03 |
| 1007 | `home_transfer_spend` | float64 | 20.6% | 79.4% | 165 | median=4.58e+07 | 0 | 2.65e+08 |
| 1008 | `home_transfers_in` | float64 | 20.6% | 79.4% | 24 | median=13 | 3 | 29 |
| 1009 | `home_transfer_income` | float64 | 20.6% | 79.4% | 168 | median=4.27e+07 | 0 | 2.08e+08 |
| 1010 | `home_transfers_out` | float64 | 20.6% | 79.4% | 28 | median=15 | 1 | 38 |
| 1011 | `away_transfer_spend` | float64 | 20.6% | 79.4% | 165 | median=4.58e+07 | 0 | 2.65e+08 |
| 1012 | `away_transfers_in` | float64 | 20.6% | 79.4% | 24 | median=13 | 3 | 29 |
| 1013 | `away_transfer_income` | float64 | 20.6% | 79.4% | 168 | median=4.27e+07 | 0 | 2.08e+08 |
| 1014 | `away_transfers_out` | float64 | 20.6% | 79.4% | 28 | median=15 | 1 | 38 |
| 1015 | `home_net_spend` | float64 | 21.3% | 78.7% | 171 | median=4.26e+06 | -1.66e+08 | 1.68e+08 |
| 1016 | `away_net_spend` | float64 | 21.3% | 78.7% | 171 | median=4.26e+06 | -1.66e+08 | 1.68e+08 |
| 1017 | `home_injuries_count` | float64 | 50.1% | 49.9% | 2 | median=0 | 0 | 3 |
| 1018 | `home_injury_impact` | float64 | 50.1% | 49.9% | 2 | median=0 | 0 | 0.282 |
| 1019 | `home_chemistry_disruption` | float64 | 50.1% | 49.9% | 2 | median=0 | 0 | 0.15 |
| 1020 | `away_injuries_count` | float64 | 50.1% | 49.9% | 2 | median=0 | 0 | 4 |
| 1021 | `away_key_attacker_out` | float64 | 50.1% | 49.9% | 2 | median=0 | 0 | 1 |
| 1022 | `away_injury_impact` | float64 | 50.1% | 49.9% | 2 | median=0 | 0 | 0.408 |
| 1023 | `away_chemistry_disruption` | float64 | 50.1% | 49.9% | 2 | median=0 | 0 | 0.15 |
| 1024 | `home_jan_arrivals` | float64 | 50.1% | 49.9% | 49 | median=0 | 0 | 79 |
| 1025 | `home_squad_disruption` | float64 | 50.1% | 49.9% | 34 | median=0 | 0 | 1 |
| 1026 | `home_signing_integration` | float64 | 50.1% | 49.9% | 2 | median=1 | 0.3 | 1 |
| 1027 | `home_squad_avg_age` | float64 | 41.7% | 58.3% | 78 | median=25.5 | 0 | 33.2 |
| 1028 | `away_jan_arrivals` | float64 | 50.1% | 49.9% | 49 | median=0 | 0 | 79 |
| 1029 | `away_squad_disruption` | float64 | 50.1% | 49.9% | 34 | median=0 | 0 | 1 |
| 1030 | `away_signing_integration` | float64 | 50.1% | 49.9% | 2 | median=1 | 0.3 | 1 |
| 1031 | `away_squad_avg_age` | float64 | 41.7% | 58.3% | 78 | median=25.5 | 0 | 33.2 |
| 1032 | `squad_value_diff` | float64 | 41.1% | 58.9% | 6124 | median=-2.32e+05 | -1.32e+09 | 1.32e+09 |
| 1033 | `elo_x_form` | float64 | 100.0% | 0.0% | 11399 | median=3.16 | -49.9 | 143 |
| 1034 | `attack_defense_mismatch` | float64 | 100.0% | 0.0% | 3222 | median=0 | -39.1 | 28.3 |
| 1035 | `rest_x_close_elo` | float64 | 100.0% | 0.0% | 32 | median=0 | -18 | 17 |
| 1036 | `defensive_form_diff` | float64 | 100.0% | 0.0% | 104 | median=0 | -5 | 5 |
| 1037 | `elo_xg_signal` | Float64 | 100.0% | 0.0% | 6670 | median=0 | -63.3 | 434 |
| 1038 | `attack_form_alignment` | float64 | 100.0% | 0.0% | 1572 | median=0.041 | -1.53 | 4.33 |
| 1039 | `h2h_competitiveness_signal` | float64 | 100.0% | 0.0% | 922 | median=-0.095 | -0.5 | 0.499 |
| 1040 | `form_elo_signal` | float64 | 100.0% | 0.0% | 9789 | median=1.74 | -16.1 | 71.6 |
| 1041 | `sharp_soft_x_elo` | float64 | 100.0% | 0.0% | 8312 | median=0 | -5.05 | 9.66 |
| 1042 | `market_elo_disagreement` | float64 | 100.0% | 0.0% | 3255 | median=-0.0563 | -0.441 | 0.46 |
| 1043 | `ah_x_form` | float64 | 100.0% | 0.0% | 149 | median=0 | -0.825 | 0.45 |
| 1044 | `draw_convergence_x_competitiveness` | float64 | 100.0% | 0.0% | 3789 | median=0.194 | 0 | 0.857 |
| 1045 | `is_early_kickoff` | float64 | 50.1% | 49.9% | 2 | median=0 | 0 | 1 |
| 1046 | `is_evening_kickoff` | float64 | 50.1% | 49.9% | 2 | median=0 | 0 | 1 |
| 1047 | `is_friday_night` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 1048 | `is_monday_night` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 1049 | `is_early_season` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 1050 | `is_mid_season` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 1051 | `is_late_season` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 1052 | `is_run_in` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 1053 | `is_august` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 1054 | `is_december` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 1055 | `is_january` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 1056 | `is_may` | int64 | 100.0% | 0.0% | 2 | median=0 | 0 | 1 |
| 1057 | `home_travel_fatigue` | float64 | 100.0% | 0.0% | 4578 | median=23.2 | 0 | 337 |
| 1058 | `away_travel_fatigue` | float64 | 100.0% | 0.0% | 4540 | median=23.1 | 0 | 337 |
| 1059 | `home_avg_subs_per_game` | float64 | 21.1% | 78.9% | 51 | median=2.9 | 1 | 5 |
| 1060 | `home_avg_sub_minute` | float64 | 21.1% | 78.9% | 193 | median=69.5 | 45.3 | 81.7 |
| 1061 | `home_avg_first_sub_minute` | float64 | 21.1% | 78.9% | 289 | median=56.4 | 21 | 76.2 |
| 1062 | `home_sub_games_tracked` | float64 | 21.1% | 78.9% | 10 | median=10 | 1 | 10 |
| 1063 | `away_avg_subs_per_game` | float64 | 21.1% | 78.9% | 48 | median=2.9 | 1.9 | 4.5 |
| 1064 | `away_avg_sub_minute` | float64 | 21.1% | 78.9% | 197 | median=69.5 | 47 | 81.3 |
| 1065 | `away_avg_first_sub_minute` | float64 | 21.1% | 78.9% | 300 | median=56.3 | 20 | 75.7 |
| 1066 | `away_sub_games_tracked` | float64 | 21.1% | 78.9% | 10 | median=10 | 1 | 10 |
| 1067 | `odds_PSH` | float64 | 33.1% | 66.9% | 762 | median=2.26 | 1.07 | 21.5 |
| 1068 | `home_form_points_3` | float64 | 49.8% | 50.2% | 9 | median=4 | 0 | 9 |
| 1069 | `away_form_points_3` | float64 | 49.8% | 50.2% | 9 | median=4 | 0 | 9 |
| 1070 | `home_ss_roll_errors_to_shot` | float64 | 20.9% | 79.1% | 13 | median=0 | 0 | 2 |
| 1071 | `home_ss_roll_errors_to_goal` | float64 | 20.9% | 79.1% | 7 | median=0 | 0 | 1.2 |
| 1072 | `away_ss_roll_duel_won_pct` | float64 | 20.9% | 79.1% | 106 | median=50 | 41.4 | 58.4 |
| 1073 | `away_ss_roll_errors_to_shot` | float64 | 20.9% | 79.1% | 12 | median=0 | 0 | 2 |
| 1074 | `away_ss_roll_errors_to_goal` | float64 | 20.9% | 79.1% | 7 | median=0 | 0 | 1.2 |
| 1075 | `ss_diff_ss_roll_errors_to_shot` | float64 | 20.8% | 79.2% | 42 | median=0 | -1.6 | 1.4 |
| 1076 | `ss_diff_ss_roll_errors_to_goal` | float64 | 20.8% | 79.2% | 22 | median=0 | -1 | 1.2 |
| 1077 | `home_very_short_rest` | float64 | 48.6% | 51.4% | 2 | median=0 | 0 | 1 |
| 1078 | `away_very_short_rest` | float64 | 48.6% | 51.4% | 2 | median=0 | 0 | 1 |
| 1079 | `low_scoring_signal` | float64 | 49.9% | 50.1% | 115 | median=1.38 | 0 | 5 |
| 1080 | `weather_rain_sum` | float64 | 15.5% | 84.5% | 192 | median=0.7 | 0 | 33.9 |

---

## 7. FEATURES — player metadata

### `data/features/player_metadata.json`

_Per-player bio: date of birth, height, nationality, market value, position._

- **Format:** JSON, `{sofascore_player_id: {...}}`
- **Size:** 1055.0KB
- **Players:** 4,933 (both leagues, 2017-18 → current)
- **Written by:** `scripts/data/build_player_metadata.py` — walks the Sofascore
  match JSONs (`data/external/sofascore/matches{,_premier_league}/*/*.json`)
  and collapses every appearance into one record per player. No network access.
- **Refreshed by:** `scripts/data/matchday_updater.run_matchday_update` Step 6d,
  whenever new matches are ingested (~6s).
- **Read by:** `web/app.py:api_player_detail` + `api_players` → the bio block on
  `/player/<team>/<name>` (`player.html`).
- **Fields:** `date_of_birth` (ISO, 100%), `height` (cm, 97.0%), `nationality` +
  `country_code` (100%), `market_value` + `market_value_currency` +
  `market_value_as_of` (86.0%), `name`, `position`, `age`.

**`age` is a convenience copy — derive from `date_of_birth` instead.** An age int
is correct only on the day it is written. Readers use `web/app.py:_player_age`,
which computes it live.

**`market_value` is display-only and must never become a training feature.** It
comes from Sofascore's `proposedMarketValueRaw`, which is the valuation served
when that match JSON was last *fetched*, not the valuation as of that kickoff —
match JSONs are rewritten after the fact, so a backfilled 2019 match would carry
today's number. `market_value_as_of` records the match date it was taken from.

**Conflict rule: last non-null wins, walking oldest match first.** Sofascore
corrects records over time (measured: `Honest Ahanor` served as Italy in older
JSONs and Nigeria in newer; `Lorenzo Palmisani` height corrected 196 → 185), so
the newest match carries the truth — but a later match that merely *omits* a
field must never erase it. Pinned by `tests/test_player_metadata_build.py`.

**History — this file had NO WRITER until 2026-08-26.** It was a one-shot
artifact dated 2026-02-17 that `web/app.py` read and nothing maintained. At the
time it was regenerated it held 2,701 players and covered **53.3%** of
current-era players (Serie A only — the EPL was entirely absent); **1,432 of its
ages were wrong**, 408 market values stale, and 41 Serie A 2026-27 players had
no bio at all and rendered blank. Post-fix: 4,933 players, **100%** current-era
coverage, 0 players lost.

---

## 7. FEATURES — player xG profiles

### `data/features/player_xg_profiles.json`

_Per-player shot/goal distributions._

- **Format:** JSON  
- **Size:** 535.8KB  
- **Modified:** 2026-04-21 19:55  
- **Type:** dict  
- **Top-level keys:** 1583  
- **Keys:** `['Denzel Dumfries|Inter', 'Lautaro Martínez|Inter', 'Romelu Lukaku|Inter', 'Robin Gosens|Inter', 'Hakan Çalhanoğlu|Inter', 'Marcelo Brozović|Inter', 'Nicolò Barella|Inter', 'Matteo Darmian|Inter', 'Federico Dimarco|Inter', 'Stefan de Vrij|Inter', 'Milan Škriniar|Inter', 'Samir Handanović|Inter', 'Marcin Listkowski|Lecce', 'Lameck Banda|Lecce', 'Joaquín Correa|Inter', 'Þórir Jóhann Helgason|Lecce', 'Alexis Blin|Lecce', 'Federico Di Francesco|Lecce', 'Assan Ceesay|Lecce', 'Gabriel Strefezza|Lecce']` ...  

---

## 7. FEATURES — formation matchup history

### `data/features/formation_database.json`

_Historical outcomes by formation matchup._

- **Format:** JSON  
- **Size:** 6.2KB  
- **Modified:** 2026-02-05 13:11  
- **Type:** dict  
- **Top-level keys:** 20  
- **Keys:** `['Empoli', 'Monza', 'Genoa', 'Inter', 'Milan', 'Torino', 'Fiorentina', 'Parma', 'Bologna', 'Udinese', 'Cagliari', 'Roma', 'Napoli', 'Verona', 'Lazio', 'Venezia', 'Como', 'Juventus', 'Atalanta', 'Lecce']`  

---

## 7. FEATURES — formation matchup pre-computed

### `data/features/formation_matchups.json`

_Pre-computed formation advantages._

- **Format:** JSON  
- **Size:** 10.0KB  
- **Modified:** 2026-02-05 13:11  
- **Type:** dict  
- **Top-level keys:** 2  
- **Keys:** `['formations', 'styles']`  

---

## 8. PREDICTIONS — main ensemble output (SA)

### `data/upcoming/predictions.json`

_**Top-level prediction file** consumed by the betting + web layers._

- **Format:** JSON  
- **Size:** 106.3KB  
- **Modified:** 2026-04-18 10:31  
- **Type:** dict  
- **Top-level keys:** 8  
- **Keys:** `['generated_at', 'league', 'model_version', 'ensemble_enabled', 'methods_available', 'phase4_features', 'predictions', 'summary']`  

**Nested `predictions` list (20 entries):**

Fields (41):

| # | Field | Type | Filled % | Sample |
|---|-------|------|----------|--------|
| 1 | `away_factors` | list | 90.0% | ['away_fav_ref'] |
| 2 | `away_form` | dict | 100.0% | {'team': 'Cagliari', 'last_n': 5, 'total_poin |
| 3 | `away_team` | str | 100.0% | Cagliari |
| 4 | `away_xg` | float | 100.0% | 0.81 |
| 5 | `betting_probabilities` | dict | 100.0% | {'home': 0.743, 'draw': 0.146, 'away': 0.111} |
| 6 | `betting_recommendation` | str | 100.0% | STRONG BET: Home Win |
| 7 | `component_predictions` | dict | 100.0% | {'market': {'prob_H': 0.7876, 'prob_D': 0.146 |
| 8 | `confidence` | float | 100.0% | 0.743 |
| 9 | `confidence_level` | str | 100.0% | VERY HIGH |
| 10 | `correction_deltas` | dict | 100.0% | {'delta_H': 0.0, 'delta_D': 0.0, 'delta_A': 0 |
| 11 | `date` | str | 100.0% | 2026-04-17 |
| 12 | `draw_analysis` | dict | 100.0% | {'draw_score': 0.0, 'is_draw_candidate': Fals |
| 13 | `expected_goals` | float | 100.0% | 2.77 |
| 14 | `formation_analysis` | dict | 100.0% | {'home_formation': '3-5-2', 'away_formation': |
| 15 | `home_factors` | list | 70.0% | ['big_stadium', 'big_home_favorite', 'home_fa |
| 16 | `home_form` | dict | 100.0% | {'team': 'Inter', 'last_n': 5, 'total_points' |
| 17 | `home_team` | str | 100.0% | Inter |
| 18 | `home_xg` | float | 100.0% | 1.96 |
| 19 | `injury_adjustments` | dict | 100.0% | {'home_injured': ['Alessandro Bastoni', 'Yann |
| 20 | `league` | str | 100.0% | serie_a |
| 21 | `lineup_source` | str | 100.0% | predicted |
| 22 | `market_anomaly` | dict | 5.0% | {'is_anomalous': True, 'reasons': ['Odds rati |
| 23 | `market_edge` | float | 100.0% | -0.045 |
| 24 | `market_implied` | dict | 100.0% | {'home': 0.788, 'draw': 0.146, 'away': 0.066, |
| 25 | `market_intelligence` | dict | 100.0% | {'sharp_direction': 'neutral', 'divergence':  |
| 26 | `match` | str | 100.0% | Inter vs Cagliari |
| 27 | `methods_used` | list | 100.0% | ['market', 'factor', 'xg', 'ml', 'player_xg'] |
| 28 | `momentum_analysis` | dict | 100.0% | {'home_big_win_recency': 2.04, 'away_big_win_ |
| 29 | `n_factors` | int | 100.0% | 6 |
| 30 | `neutral_factors` | list | 5.0% | ['strict_ref'] |
| 31 | `over_25` | bool | 100.0% | True |
| 32 | `predicted_outcome` | str | 100.0% | HOME |
| 33 | `probabilities` | dict | 100.0% | {'home': 0.743, 'draw': 0.146, 'away': 0.111} |
| 34 | `referee` | str | 100.0% | Daniele Doveri |
| 35 | `referee_bias` | str | 100.0% | away_favoring |
| 36 | `sentiment_analysis` | dict | 100.0% | {'home_motivation': 0.5, 'away_motivation': 0 |
| 37 | `situational_context` | dict | 100.0% | {'home_rest_days': 5.0, 'away_rest_days': 6.0 |
| 38 | `strategy` | str | 100.0% | default |
| 39 | `time` | str | 100.0% | 14:45 |
| 40 | `venue` | str | 100.0% | Inter Stadium |
| 41 | `weights_applied` | dict | 100.0% | {'market': 0.2047952047952048, 'ml': 0.604395 |

---

## 8. PREDICTIONS — goal markets (O/U any line)

### `data/upcoming/goal_predictions.json`

_Per-match expected goals and O/U 0.5, 1.5, 2.5, 3.5, 4.5._

- **Format:** JSON  
- **Size:** 17.4KB  
- **Modified:** 2026-04-16 20:52  
- **Type:** dict  
- **Top-level keys:** 3  
- **Keys:** `['generated_at', 'model', 'predictions']`  

**Nested `predictions` list (40 entries):**

Fields (12):

| # | Field | Type | Filled % | Sample |
|---|-------|------|----------|--------|
| 1 | `confidence` | str | 100.0% | HIGH |
| 2 | `date` | str | 100.0% | 2026-04-17 |
| 3 | `expected_away_goals` | float | 100.0% | 0.7158499999999999 |
| 4 | `expected_home_goals` | float | 100.0% | 2.49525 |
| 5 | `expected_total_goals` | float | 100.0% | 3.21 |
| 6 | `factors` | list | 95.0% | ['big_stadium', 'big_home_favorite', 'home_fa |
| 7 | `match` | str | 100.0% | Inter vs Cagliari |
| 8 | `over_0_5` | float | 100.0% | 0.96 |
| 9 | `over_1_5` | float | 100.0% | 0.785 |
| 10 | `over_2_5` | float | 100.0% | 0.682 |
| 11 | `over_3_5` | float | 100.0% | 0.33 |
| 12 | `over_4_5` | float | 100.0% | 0.221 |
| 13 | `ou_ml` | dict | SA rows only | `{"2.5": 0.6721, "1.5": 0.8178, "3.5": 0.3382}` — the O/U CatBoost leg per line |
| 14 | `ou_poisson` | dict | SA rows only | `{"2.5": 0.537, ...}` — the Poisson leg per line (pre-blend) |
| 15 | `ou_blend_weight` | dict | SA rows only | `{"2.5": 0.65, ...}` — the ML share that produced `over_X_Y` |

**Writers (2026-09-05):** pipeline Step 19 (`run_full_pipeline`) on every run, AND
`weekly_retrain._refresh_goal_predictions` right after a retrain promotes anything — both go
through `over_under_model.refresh_from_predictions` / `ml_probs_from_predictions`, one code
path. Before this the engine's post-retrain refresh rewrote only `predictions.json`, so a newly
promoted O/U classifier reached the file the scanner prices only at the next morning pipeline.

**How `over_1_5` / `over_2_5` / `over_3_5` are made (2026-09-04 audit trail):** when the ensemble
engine's `component_predictions.over_under_ml` is present for the match, `over_under_model`
blends `w_ml × ML + (1 − w_ml) × Poisson` and writes the blend INTO `over_X_Y` — these are the
probabilities `betting_unified.scan_ou_market` prices (the only enabled markets). Fields 13–15
record the legs so `component_ledger` can grade each one; `w_ml` defaults to 0.65
(`DEFAULT_ML_BLEND_WEIGHT`) unless `data/models/ou_blend_weights.json` (gated refit) overrides
the line. Rows without an ML leg (EPL, ML unavailable) are pure Poisson and carry empty dicts.

---

## 8. PREDICTIONS — BTTS

### `data/upcoming/btts_predictions.json`

_Per-match BTTS Yes/No._

- **Format:** JSON  
- **Size:** 4.1KB  
- **Modified:** 2026-04-16 20:52  
- **Type:** list  
- **Length:** 20  

**Fields (7):**

| # | Field | Type | Filled % | Sample |
|---|-------|------|----------|--------|
| 1 | `btts_no` | float | 100.0% | 0.5236 |
| 2 | `btts_yes` | float | 100.0% | 0.4764 |
| 3 | `date` | str | 100.0% | 2026-04-17 |
| 4 | `expected_away_goals` | float | 100.0% | 1.71 |
| 5 | `expected_home_goals` | float | 100.0% | 0.872 |
| 6 | `match` | str | 100.0% | Sassuolo vs Como |
| 7 | `source` | str | 100.0% | CatBoost ML |

---

## 8. PREDICTIONS — cards

### `data/upcoming/cards_predictions.json`

_Per-match expected cards + O/U._

- **Format:** JSON  
- **Size:** 5.1KB  
- **Modified:** 2026-04-16 20:52  
- **Type:** list  
- **Length:** 20  

**Fields (9):**

| # | Field | Type | Filled % | Sample |
|---|-------|------|----------|--------|
| 1 | `date` | str | 100.0% | 2026-04-17 |
| 2 | `expected_away_cards` | float | 100.0% | 1.81 |
| 3 | `expected_cards` | float | 100.0% | 3.93 |
| 4 | `expected_home_cards` | float | 100.0% | 2.12 |
| 5 | `match` | str | 100.0% | Sassuolo vs Como |
| 6 | `over_3_5` | float | 100.0% | 0.5529 |
| 7 | `over_4_5` | float | 100.0% | 0.3152 |
| 8 | `over_5_5` | float | 100.0% | 0.2042 |
| 9 | `source` | str | 100.0% | CatBoost ML |

---

## 8. PREDICTIONS — corners

### `data/upcoming/corners_predictions.json`

_Per-match expected corners + O/U._

- **Format:** JSON  
- **Size:** 5.2KB  
- **Modified:** 2026-04-16 20:52  
- **Type:** list  
- **Length:** 20  

**Fields (9):**

| # | Field | Type | Filled % | Sample |
|---|-------|------|----------|--------|
| 1 | `date` | str | 100.0% | 2026-04-17 |
| 2 | `expected_away_corners` | float | 100.0% | 5.0 |
| 3 | `expected_corners` | float | 100.0% | 10.0 |
| 4 | `expected_home_corners` | float | 100.0% | 5.0 |
| 5 | `match` | str | 100.0% | Sassuolo vs Como |
| 6 | `over_10_5` | float | 100.0% | 0.2622 |
| 7 | `over_8_5` | float | 100.0% | 0.5079 |
| 8 | `over_9_5` | float | 100.0% | 0.3767 |
| 9 | `source` | str | 100.0% | CatBoost ML |

---

## 8. PREDICTIONS — margin/handicap

### `data/upcoming/margin_predictions.json`

_Per-match expected margin + handicap probabilities._

- **Format:** JSON  
- **Size:** 36.6KB  
- **Modified:** 2026-04-16 20:52  
- **Type:** dict  
- **Top-level keys:** 3  
- **Keys:** `['generated_at', 'model', 'predictions']`  

**Nested `predictions` list (40 entries):**

Fields (10):

| # | Field | Type | Filled % | Sample |
|---|-------|------|----------|--------|
| 1 | `away_rating` | float | 100.0% | 0.78 |
| 2 | `confidence` | str | 100.0% | HIGH |
| 3 | `date` | str | 100.0% | 2026-04-17 |
| 4 | `expected_margin` | float | 100.0% | 1.8009999999999997 |
| 5 | `factors` | list | 95.0% | ['big_stadium', 'big_home_favorite', 'home_fa |
| 6 | `handicap_probs` | dict | 100.0% | {'-2.5': {'home': 0.355, 'away': 0.645}, '-2. |
| 7 | `home_rating` | float | 100.0% | 1.4 |
| 8 | `margin_std_dev` | float | 100.0% | 1.88 |
| 9 | `match` | str | 100.0% | Inter vs Cagliari |
| 10 | `rating_diff` | float | 100.0% | 0.62 |

---

## 8. PREDICTIONS — ML ensemble (per-match)

### `data/upcoming/ml_predictions.json`

_Detailed per-match ML breakdown by component._

- **Format:** JSON  
- **Size:** 273.3KB  
- **Modified:** 2026-04-16 20:52  
- **Type:** dict  
- **Top-level keys:** 20  
- **Keys:** `['Sassuolo vs Como', 'Inter vs Cagliari', 'Udinese vs Parma', 'Napoli vs Lazio', 'Roma vs Atalanta', 'Cremonese vs Torino', 'Verona vs Milan', 'Pisa vs Genoa', 'Juventus vs Bologna', 'Lecce vs Fiorentina', 'Napoli vs Cremonese', 'Parma vs Pisa', 'Bologna vs Roma', 'Verona vs Lecce', 'Fiorentina vs Sassuolo', 'Genoa vs Como', 'Torino vs Inter', 'Milan vs Juventus', 'Cagliari vs Atalanta', 'Lazio vs Udinese']`  

---

## 8. PREDICTIONS — predicted lineups

### `data/models/player_floors/start_calibration.json`

**Writer:** `python3 -m scripts.betting.player_predictions calibrate-start`
(`measure_start_calibration`): joins every `data/lineup_history/predictions_*.json` (the
lineup predictor's daily archive, no dates inside — matched to the next played fixture between
the two clubs within 10 days) to `player_match_stats.parquet` `is_starter` (absent = did not
start; names accent-folded; a name the club never fielded is a join miss, 0.8%). Fits isotonic
regression start%/100 → P(started). `{n, n_archives, fitted_at, dates, brier_raw,
brier_calibrated, brier_base, buckets[{lo,hi,n,pred,obs}], knots[[x,y]×41]}`.
**Reader:** `player_predictions.calibrated_start_prob` (mtime-cached) — every predicted-XI
starter's price is the start/bench mixture at this probability. First fit 2026-09-05:
n=3,807, Brier 0.213 → 0.170 (base 0.249); the predictor is overconfident in every bucket.

### `data/upcoming/confirmed_lineups.json` and `lineup_chain_status.json`

**Writer:** `scraper/lineup_fetcher.py::fetch_and_save_lineups` (the scheduler's `lineup_fetch`
stage at T-60/T-55, run as a subprocess). Chain order since 2026-09-05: **Sofascore → ESPN →
football-data.org → API-Football**. `confirmed_lineups.json` is written only when at least one
match has both XIs (`{fetched_at, sources, matches: {"Home vs Away": {home_lineup, home_bench,
home_formation, away_*, lineup_source, source_api}}}`); `source_api` is `sofascore` or `espn`.
`lineup_chain_status.json` is written **every run**: `{checked_at, n_matches, confirmed: [...],
sources: {sofascore: {n, last_failure_status}, espn: {n}, football_data: {key_set, n,
no_lineup_field}, api_football: {key_set, n, error}}, reason}` — `reason` is the one-line Italian
explanation the scheduler pushes to Telegram (once per match) when a match has no team sheet.

**Readers:** `scripts/betting/player_predictions.py::match_player_floors` (the official sheet wins
over the predicted XI; names accent-folded to the pms spelling — ESPN writes `Pasalic`),
`scripts/fantacalcio/lineup_check.py`, `scripts/pipeline/scheduler.py` (confirmation + notice).

**Measured state of the sources (2026-09-05, university network):** Sofascore answers 403
`challenge` on EVERY api endpoint (robots.txt 200, so not the blanket deny); ESPN 200 with full
XIs at half-time and at ~T-60 (WC verification 2026-07-13); football-data.org free tier: no
`lineup` field even on finished matches; API-Football free plan: `Free plans do not have access
to this season, try from 2022 to 2024`. Before ESPN was wired, the chain failed silently and
player props were priced off a predicted XI containing a benched player.

### `data/upcoming/lineup_predictions.json`

_Predicted starting XI per match._

- **Format:** JSON  
- **Size:** 979.5KB  
- **Modified:** 2026-04-21 16:18  
- **Type:** dict  
- **Top-level keys:** 5  
- **Keys:** `['generated_at', 'match_count', 'team_count', 'matches', 'teams']`  

---

## 8. PREDICTIONS — player prop value bets

### `data/upcoming/player_prop_value_bets.json`

_Player-level value bets detected (SOT, shots, goalscorer)._

- **Format:** JSON  
- **Size:** 75.3KB  
- **Modified:** 2026-03-17 20:39  
- **Type:** dict  
- **Top-level keys:** 4  
- **Keys:** `['generated_at', 'min_edge_pct', 'total_value_bets', 'bets']`  

**Nested `bets` list (171 entries):**

Fields (15):

| # | Field | Type | Filled % | Sample |
|---|-------|------|----------|--------|
| 1 | `avg_odds` | float | 100.0% | 3.25 |
| 2 | `best_bookmaker` | str | 100.0% | 1xBet |
| 3 | `best_odds` | float | 100.0% | 3.25 |
| 4 | `commence_time` | str | 100.0% | 2026-03-21T17:00:00Z |
| 5 | `edge_pct` | float | 100.0% | 49.8 |
| 6 | `implied_prob` | float | 100.0% | 0.3077 |
| 7 | `kelly_fraction` | float | 100.0% | 0.1 |
| 8 | `market` | str | 100.0% | Shots Over 0.5 |
| 9 | `match` | str | 100.0% | Milan vs Torino |
| 10 | `model_prob` | float | 100.0% | 0.4608 |
| 11 | `n_bookmakers` | int | 100.0% | 1 |
| 12 | `player` | str | 100.0% | Guillermo Maripan |
| 13 | `position` | str | 100.0% | D |
| 14 | `team` | str | 100.0% | Torino |
| 15 | `tier` | str | 100.0% | C |

---

## 8. PREDICTIONS — O/U bet recommendations

### `data/upcoming/over_under_bets.json`

_'recommended' vs 'consider' O/U bets._

- **Format:** JSON  
- **Size:** 11.1KB  
- **Modified:** 2026-04-16 20:53  
- **Type:** dict  
- **Top-level keys:** 5  
- **Keys:** `['generated_at', 'summary', 'recommended', 'consider', 'italian_market_standards']`  

**Nested `recommended` list (5 entries):**

Fields (11):

| # | Field | Type | Filled % | Sample |
|---|-------|------|----------|--------|
| 1 | `bet` | str | 100.0% | OVER 2.5 |
| 2 | `date` | str | 100.0% | 2026-04-18 |
| 3 | `expected_goals` | float | 100.0% | 2.72 |
| 4 | `factors` | list | 100.0% | ['big_stadium', 'hot_home', 'away_fav_ref'] |
| 5 | `implied_probability` | float | 100.0% | 0.503 |
| 6 | `italian_format` | bool | 100.0% | True |
| 7 | `match` | str | 100.0% | Roma vs Atalanta |
| 8 | `odds` | float | 100.0% | 1.99 |
| 9 | `our_probability` | float | 100.0% | 0.607 |
| 10 | `stake_pct` | float | 100.0% | 2.0 |
| 11 | `value_pct` | float | 100.0% | 10.4 |

---

## 8. PREDICTIONS — BTTS + corners recs

### `data/upcoming/btts_corners_bets.json`

_Recommended BTTS and corners bets._

- **Format:** JSON  
- **Size:** 38.8KB  
- **Modified:** 2026-04-16 20:52  
- **Type:** dict  
- **Top-level keys:** 4  
- **Keys:** `['generated_at', 'summary', 'recommended', 'consider']`  

**Nested `recommended` list (125 entries):**

Fields (7):

| # | Field | Type | Filled % | Sample |
|---|-------|------|----------|--------|
| 1 | `bet` | str | 100.0% | NO |
| 2 | `date` | str | 100.0% | 2026-04-17 |
| 3 | `factors` | list | 93.6% | ['big_stadium', 'big_home_favorite', 'home_fa |
| 4 | `fair_odds` | float | 100.0% | 1.63 |
| 5 | `market` | str | 100.0% | btts |
| 6 | `match` | str | 100.0% | Inter vs Cagliari |
| 7 | `our_probability` | float | 100.0% | 0.613 |

---

## 8. PREDICTIONS — cards recs

### `data/upcoming/cards_bets.json`

_Recommended cards bets._

- **Format:** JSON  
- **Size:** 8.8KB  
- **Modified:** 2026-04-16 20:52  
- **Type:** dict  
- **Top-level keys:** 5  
- **Keys:** `['generated_at', 'note', 'summary', 'recommended', 'consider']`  

**Nested `recommended` list (28 entries):**

Fields (7):

| # | Field | Type | Filled % | Sample |
|---|-------|------|----------|--------|
| 1 | `bet` | str | 100.0% | UNDER 5.5 |
| 2 | `date` | str | 100.0% | 2026-04-17 |
| 3 | `expected_cards` | float | 100.0% | 3.95 |
| 4 | `factors` | list | 17.9% | ['strict_ref', 'strict_ref'] |
| 5 | `fair_odds` | float | 100.0% | 1.26 |
| 6 | `match` | str | 100.0% | Inter vs Cagliari |
| 7 | `our_probability` | float | 100.0% | 0.793 |

---

## 8. PREDICTIONS — handicap recs

### `data/upcoming/handicap_bets.json`

_Asian handicap recommendations._

- **Format:** JSON  
- **Size:** 28.9KB  
- **Modified:** 2026-04-16 20:53  
- **Type:** dict  
- **Top-level keys:** 5  
- **Keys:** `['generated_at', 'summary', 'recommended', 'consider', 'italian_market_standards']`  

**Nested `recommended` list (34 entries):**

Fields (13):

| # | Field | Type | Filled % | Sample |
|---|-------|------|----------|--------|
| 1 | `bet` | str | 100.0% | AWAY +1 |
| 2 | `converted_to_italian` | bool | 100.0% | False |
| 3 | `date` | str | 100.0% | 2026-04-19 |
| 4 | `expected_margin` | float | 100.0% | 0.5860000000000001 |
| 5 | `factors` | list | 94.1% | ['big_stadium', 'home_favorite', 'hot_home',  |
| 6 | `implied_probability` | float | 100.0% | 0.478 |
| 7 | `italian_format` | bool | 100.0% | True |
| 8 | `match` | str | 100.0% | Juventus vs Bologna |
| 9 | `odds` | float | 100.0% | 2.09 |
| 10 | `original_line` | float | 55.9% | -1.5 |
| 11 | `our_probability` | float | 100.0% | 0.5778 |
| 12 | `stake_pct` | float | 100.0% | 2.0 |
| 13 | `value_pct` | float | 100.0% | 9.9 |

---

## 8. PREDICTIONS — unified strategy output

### `data/upcoming/predictions_unified.json`

_Top-level strategy recommendation per match._

- **Format:** JSON  
- **Size:** 16.0KB  
- **Modified:** 2026-02-05 17:58  
- **Type:** dict  
- **Top-level keys:** 4  
- **Keys:** `['generated_at', 'strategy', 'predictions', 'bankroll_status']`  

**Nested `predictions` list (8 entries):**

Fields (11):

| # | Field | Type | Filled % | Sample |
|---|-------|------|----------|--------|
| 1 | `bankroll_status` | dict | 100.0% | {'current': 1000.0, 'drawdown': 0.0, 'can_bet |
| 2 | `betting_recommendation` | dict | 100.0% | {'bet': 'draw', 'stake': 46.33, 'odds': 4.96, |
| 3 | `date` | str | 100.0% | 2026-02-09 |
| 4 | `draw_analysis` | dict | 100.0% | {'draw_score': 0.701, 'is_draw_candidate': 'T |
| 5 | `match` | str | 100.0% | Atalanta vs Cremonese |
| 6 | `prediction` | dict | 100.0% | {'outcome': 'DRAW', 'probabilities': {'home': |
| 7 | `strategy` | str | 100.0% | selective |
| 8 | `time` | str | 100.0% | 18:00 |
| 9 | `timestamp` | str | 100.0% | 2026-02-05T17:58:15.981039 |
| 10 | `value_analysis` | dict | 100.0% | {'has_value': True, 'best_bet': 'draw', 'edge |
| 11 | `venue` | str | 100.0% | Gewiss Stadium |

---

## 9. BETS — settled history

### `data/betting/history.json`

_**Source of ROI numbers.** All settled bets with profit/loss._

- **Format:** JSON  
- **Size:** 59.1KB  
- **Modified:** 2026-04-21 15:24  
- **Type:** list  
- **Length:** 186  

**Fields (21):**

| # | Field | Type | Filled % | Sample |
|---|-------|------|----------|--------|
| 1 | `bet_id` | str | 0.5% | 2026-03-01_Roma_vs_Juventus_OU_1.5_OVER_1.5 |
| 2 | `confidence` | str | 80.1% | MEDIUM |
| 3 | `date` | str | 100.0% | 2026-02-06 |
| 4 | `id` | int | 1.6% | 1 |
| 5 | `market` | str | 100.0% | totals |
| 6 | `match` | str | 100.0% | Verona vs Pisa |
| 7 | `notes` | str | 1.6% | Odds of 12.0 for OVER 1.5 are almost certainl |
| 8 | `odds` | float | 100.0% | 12.0 |
| 9 | `outcome` | str | 0.5% | won |
| 10 | `profit` | float | 100.0% | -30.0 |
| 11 | `profit_loss` | float | 0.5% | 5.85 |
| 12 | `result` | str | 99.5% | 0-0 |
| 13 | `return` | float | 1.6% | 0.0 |
| 14 | `score` | str | 0.5% | 3-3 |
| 15 | `selection` | str | 100.0% | OVER 1.5 |
| 16 | `settled_at` | str | 100.0% | 2026-02-07 |
| 17 | `source` | str | 4.3% | journal_backfill |
| 18 | `stake` | float | 100.0% | 30.0 |
| 19 | `status` | str | 100.0% | lost |
| 20 | `total_goals` | int | 1.1% | 0 |
| 21 | `value_pct` | float | 97.8% | 14.8 |

---

## 9. BETS — pending bet log

### `data/upcoming/bet_history.json`

_Pending + unresolved bets._

- **Format:** JSON  
- **Size:** 20.2KB  
- **Modified:** 2026-04-15 20:38  
- **Type:** list  
- **Length:** 49  

**Fields (14):**

| # | Field | Type | Filled % | Sample |
|---|-------|------|----------|--------|
| 1 | `best_bookmaker` | str | 100.0% | Unibet (NL) |
| 2 | `best_odds` | float | 100.0% | 3.15 |
| 3 | `date` | str | 100.0% | 2026-02-22 |
| 4 | `edge_pct` | float | 100.0% | 18.85 |
| 5 | `ev_per_unit` | float | 100.0% | 0.2411 |
| 6 | `id` | str | 100.0% | 2026-02-22_Genoa_vs_Torino_1X2_Draw |
| 7 | `market` | str | 100.0% | 1X2 |
| 8 | `match` | str | 100.0% | Genoa vs Torino |
| 9 | `model_prob` | float | 100.0% | 0.394 |
| 10 | `placed_at` | str | 100.0% | 2026-02-11T14:59:29.802632 |
| 11 | `profit` | — | 0.0% | — |
| 12 | `result` | — | 0.0% | — |
| 13 | `selection` | str | 100.0% | Draw |
| 14 | `stake` | float | 100.0% | 28.03 |

---

## 9. BETS — CLV tracker

### `data/betting/clv_history.json`

_Closing line value per bet._

- **Format:** JSON  
- **Size:** 56.4KB  
- **Modified:** 2026-04-17 12:03  
- **Type:** dict  
- **Top-level keys:** 8  
- **Keys:** `['bets', 'running_clv', 'total_bets_tracked', 'positive_clv_count', 'negative_clv_count', 'updated_at', 'running_clv_pct', 'captures']`  

**Nested `bets` list (175 entries):**

Fields (11):

| # | Field | Type | Filled % | Sample |
|---|-------|------|----------|--------|
| 1 | `bet_odds` | float | 100.0% | 3.15 |
| 2 | `closing_odds` | float | 100.0% | 3.15 |
| 3 | `clv` | float | 100.0% | 0.0 |
| 4 | `clv_pct` | float | 100.0% | 0.0 |
| 5 | `league` | str | 11.4% | serie_a |
| 6 | `market` | str | 100.0% | 1X2 |
| 7 | `match` | str | 100.0% | Genoa vs Torino |
| 8 | `placed_at` | str | 100.0% | 2026-02-11T15:16:51.467280 |
| 9 | `result` | str | 100.0% | unknown |
| 10 | `selection` | str | 100.0% | Draw |
| 11 | `tracked_at` | str | 100.0% | 2026-02-11T15:19:17.136701 |

---

## 9. BETS — P&L history

### `data/betting/pnl_history.json`

_Daily/weekly P&L summary._

- **Format:** JSON  
- **Size:** 10.6KB  
- **Modified:** 2026-04-06 23:30  
- **Type:** list  
- **Length:** 27  

**Fields (23):**

| # | Field | Type | Filled % | Sample |
|---|-------|------|----------|--------|
| 1 | `backfilled` | bool | 25.9% | True |
| 2 | `balance` | float | 100.0% | 1278.06 |
| 3 | `clv_avg_pct` | float | 74.1% | 6.64 |
| 4 | `cumulative_profit` | float | 74.1% | 278.06 |
| 5 | `date` | str | 100.0% | 2026-02-20 |
| 6 | `lost` | int | 25.9% | 0 |
| 7 | `lost_this_run` | int | 74.1% | 1 |
| 8 | `profit` | float | 25.9% | 6.25 |
| 9 | `profit_this_run` | float | 74.1% | 17.47 |
| 10 | `push_this_run` | int | 74.1% | 0 |
| 11 | `roi_pct` | float | 74.1% | 30.45 |
| 12 | `settled` | int | 25.9% | 1 |
| 13 | `settled_this_run` | int | 74.1% | 3 |
| 14 | `timestamp` | str | 74.1% | 2026-02-20T16:50:15.745873 |
| 15 | `total_bets` | int | 74.1% | 87 |
| 16 | `total_lost` | int | 74.1% | 17 |
| 17 | `total_pending` | int | 74.1% | 13 |
| 18 | `total_pushes` | int | 74.1% | 2 |
| 19 | `total_settled` | int | 100.0% | 45 |
| 20 | `total_staked` | float | 74.1% | 913.28 |
| 21 | `total_won` | int | 74.1% | 26 |
| 22 | `won` | int | 25.9% | 1 |
| 23 | `won_this_run` | int | 74.1% | 2 |

---

## 9. BETS — current bankroll

### `data/bankroll.json`

_Current balance + state._

- **Format:** JSON  
- **Size:** 353.0B  
- **Modified:** 2026-02-04 17:31  
- **Type:** dict  
- **Top-level keys:** 9  
- **Keys:** `['initial_bankroll', 'current_bankroll', 'peak_bankroll', 'allocated', 'pending_bets', 'bet_history', 'daily_stats', 'created_at', 'updated_at']`  

---

## 10. ODDS — live odds (full multi-market)

### `data/upcoming/odds_full.json`

_Current odds across all markets from all bookmakers._

- **Format:** JSON  
- **Size:** 252.9KB  
- **Modified:** 2026-04-16 20:45  
- **Type:** dict  
- **Top-level keys:** 5  
- **Keys:** `['fetched_at', 'source', 'league', 'markets', 'matches']`  

---

## 11. MODEL METADATA — SA CatBoost

### `data/models/serie_a/catboost_metadata.json`

_Serie A main 1X2 CatBoost classifier metadata._

- **Format:** JSON  
- **Size:** 5.6KB  
- **Modified:** 2026-04-13 13:11  
- **Type:** dict  
- **Top-level keys:** 8  
- **Keys:** `['model_type', 'variant', 'n_features', 'feature_names', 'saved_at', 'version', 'metrics', 'feature_importance']`  

---

## 11. MODEL METADATA — SA deployment state

### `data/models/serie_a/deployment_state.json`

_Which model version is live._

- **Format:** JSON  
- **Size:** 615.0B  
- **Modified:** 2026-04-04 10:10  
- **Type:** dict  
- **Top-level keys:** 11  
- **Keys:** `['league', 'current_model', 'model_version', 'deployed_at', 'validated_at', 'metrics', 'rejection_thresholds', 'validation_status', 'betting_enabled', 'force_enabled', 'per_fold']`  

---

## 11. MODEL METADATA — SA training report

### `data/models/serie_a/training_report.json`

_Latest retraining stats._

- **Format:** JSON  
- **Size:** 21.2KB  
- **Modified:** 2026-04-13 13:16  
- **Type:** dict  
- **Top-level keys:** 11  
- **Keys:** `['timestamp', 'model_type', 'note', 'n_features_original', 'n_features_selected', 'selected_features', 'exclude_odds', 'tuned_params', 'per_model_cv', 'ensemble_cv', 'top_20_features']`  

---

## 11. MODEL METADATA — SA calibration

### `data/models/serie_a/rolling_calibration.json`

_Rolling calibration for SA model._

- **Format:** JSON  
- **Size:** 89.0B  
- **Modified:** 2026-04-13 13:16  
- **Type:** dict  
- **Top-level keys:** 3  
- **Keys:** `['buckets', 'processed_count', 'updated_at']`  

---

## 11. MODEL METADATA — retrain log

### `data/models/retrain_history.jsonl`

_Jsonl history of all retraining attempts._

- **Format:** JSONL  
- **Size:** 6.1KB  
- **Modified:** 2026-04-18 10:31  
- **Lines:** 12  
- **Keys in first entry:** `['mode', 'timestamp', 'promoted', 'current_metrics', 'error']`
- **O/U entries (since 2026-09-04):** one line per O/U classifier per retrain, `mode: ou_1.5` / `ou_2.5`,
  with `promoted`, `reason`/`comparison` (the gate's verdict vs the incumbent), `holdout`
  (n, dates, candidate + incumbent metrics, naive log-loss), `cv_log_loss`, `cv_calibration_gap`,
  and `dry_run: true` on a preview run (the verdict is then in the reason as "DRY RUN — would
  PROMOTE/HOLD"). Written by `weekly_retrain._retrain_ou_classifiers`, which runs as an
  **auxiliary model on every retrain, whether or not the 1X2 ensemble was promoted** — until
  2026-09-05 it only ran inside the ensemble's promote branch, so a held ensemble froze the
  money model too. Read with `cli.py retrain_history` or `weekly_retrain --history`.

---


## 12. COLLAPSED DIRECTORIES — per-item raw dumps

These dirs contain many files (often one per match, day, or experiment). Summarized here, not audited individually.

### `data/external/fotmob/matches/`

- **Description:** Per-match Fotmob JSON dumps (one file per match_id)  
- **File count:** 3,775  
- **Total size:** 205.5MB  
- **Extensions:** {'.json': 3775}  
- **Newest file:** `data/external/fotmob/matches/2025-2026/4813664.json` (256.0B, 2026-03-28)  
- **Schema:** dict with 1 keys: `['basic']`

---

### `data/external/sofascore/matches/`

- **Description:** Per-match Sofascore JSON dumps (one file per match_id)  
- **File count:** 3,329  
- **Total size:** 380.2MB  
- **Extensions:** {'.json': 3329}  
- **Newest file:** `data/external/sofascore/matches/2025-2026/13980099.json` (147.8KB, 2026-04-20)  
- **Schema:** dict with 5 keys: `['match_id', 'home_lineup', 'away_lineup', 'team_stats', 'shotmap']`

---

### `data/odds_snapshots/`

- **Description:** Per-day odds snapshots (one file per fetch)  
- **File count:** 225  
- **Total size:** 15.2MB  
- **Extensions:** {'.json': 225}  
- **Newest file:** `data/odds_snapshots/extra_20260416_204734.json` (108.0KB, 2026-04-16)  
- **Schema:** dict with 2 keys: `['timestamp', 'matches']`

---

### `data/lineup_history/`

- **Description:** Per-match lineup history dumps  
- **File count:** 57  
- **Total size:** 61.1MB  
- **Extensions:** {'.json': 57}  
- **Newest file:** `data/lineup_history/predictions_20260421_161857.json` (979.5KB, 2026-04-21)  
- **Schema:** dict with 5 keys: `['generated_at', 'match_count', 'team_count', 'matches', 'teams']`

---

### `data/optimization/`

- **Description:** Per-run optimization experiment outputs  
- **File count:** 87  
- **Total size:** 6.5MB  
- **Extensions:** {'.json': 86, '.pkl': 1}  
- **Newest file:** `data/optimization/live_reconciliation.json` (4.1KB, 2026-03-15)  
- **Schema:** dict with 6 keys: `['run_date', 'n_bets', 'calibration', 'edge_performance', 'market_comparison', 'clv']`

---

### `data/external/injuries/`

- **Description:** Weekly injury snapshots (one file per Friday)  
- **File count:** 12  
- **Total size:** 87.1KB  
- **Extensions:** {'.parquet': 12}  
- **Newest file:** `data/external/injuries/injuries_2026-04-17.parquet` (7.2KB, 2026-04-17)  
- **Schema:** 8 cols, 63 rows. First cols: `['player_name', 'team', 'injury_type', 'start_date', 'expected_return', 'is_currently_out', 'source', 'scraped_at']`

---

### `data/external/transfermarkt/`

- **Description:** Per-season market values  
- **File count:** 36  
- **Total size:** 930.5KB  
- **Extensions:** {'.parquet': 36}  
- **Newest file:** `data/external/transfermarkt/premier_league_market_values_2025_2026.parquet` (21.9KB, 2026-03-27)  
- **Schema:** 6 cols, 920 rows. First cols: `['team', 'player_name', 'position', 'age', 'market_value_eur', 'nationality']`

---

### `data/external/understat/`

- **Description:** Per-season Understat season JSONs  
- **File count:** 13  
- **Total size:** 1.4MB  
- **Extensions:** {'.parquet': 4, '.json': 9}  
- **Newest file:** `data/external/understat/matches_xg.parquet` (181.0KB, 2026-04-21)  
- **Schema:** 17 cols, 3370 rows. First cols: `['match_id', 'datetime', 'season', 'home_team', 'away_team', 'home_short', 'away_short', 'home_id', 'away_id', 'home_goals', 'away_goals', 'home_xg']`

---

### `data/external/api_football/`

- **Description:** API-Football raw caches (mostly EPL)  
- **File count:** 105  
- **Total size:** 3.8MB  
- **Extensions:** {'.json': 104, '.parquet': 1}  
- **Newest file:** `data/external/api_football/epl_stats_2024.parquet` (34.1KB, 2026-03-28)  
- **Schema:** 42 cols, 97 rows. First cols: `['fixture_id', 'date', 'home_team', 'away_team', 'home_goals', 'away_goals', 'home_shots_on_goal', 'home_shots_off_goal', 'home_total_shots', 'home_blocked_shots', 'home_shots_insidebox', 'home_shots_outsidebox']`

---

### `data/external/fotmob/`

- **Description:** Fotmob season-level EPL stats  
- **File count:** 3,776  
- **Total size:** 205.6MB  
- **Extensions:** {'.parquet': 1, '.json': 3775}  
- **Newest file:** `data/external/fotmob/epl_match_stats.parquet` (148.3KB, 2026-03-28)  
- **Schema:** 97 cols, 3350 rows. First cols: `['match_id', 'season', 'home_team', 'away_team', 'home_team_id', 'away_team_id', 'home_score', 'away_score', 'match_date', 'data_source', 'league_round', 'home_possession']`

---

### `data/models/universal/`

- **Description:** Universal (cross-league) model binaries + configs  
- **File count:** 833  
- **Total size:** 1.6GB  
- **Extensions:** {'.json': 494, '.pkl': 38, '.cbm': 177, '.md': 5, '.txt': 104, '.bak': 10, '.parquet': 5}  
- **Newest file:** `data/models/universal/draw_detector_metadata.json` (19.1KB, 2026-04-18)  
- **Schema:** dict with 8 keys: `['trained_at', 'n_train', 'n_features', 'feature_names', 'blend_alpha', 'ablation_results', 'avg_ll_improvement', 'blend_enabled']`

---

### `data/models/markets/`

- **Description:** Per-market model binaries (BTTS, corners, cards, O/U)  
- **File count:** 96  
- **Total size:** 45.8MB  
- **Extensions:** {'.cbm': 79, '.json': 17}  
- **Newest file:** `data/models/markets/tuned_hyperparams.json` (214.0B, 2026-04-15)  
- **Schema read error:** Expecting value: line 14 column 14 (char 214)

---

### `data/models/player/`

- **Description:** Per-player-prop model binaries  
- **File count:** 31  
- **Total size:** 2.5MB  
- **Extensions:** {'.pkl': 20, '.cbm': 10, '.json': 1}  
- **Newest file:** `data/models/player/player_metadata.json` (10.8KB, 2026-02-11)  
- **Schema:** dict with 6 keys: `['n_features', 'feature_names', 'val_season', 'test_seasons', 'calibration', 'results']`

---

### `data/models/rich/`

- **Description:** 'Rich' experimental model variant  
- **File count:** 12  
- **Total size:** 2.4MB  
- **Extensions:** {'.txt': 2, '.json': 8, '.cbm': 2}  
- **Newest file:** `data/models/rich/catboost_metadata.json` (48.1KB, 2026-04-15)  
- **Schema:** dict with 8 keys: `['model_type', 'variant', 'n_features', 'feature_names', 'saved_at', 'version', 'metrics', 'feature_importance']`

---

### `data/models/deep/`

- **Description:** Deep learning experimental models  
- **File count:** 5  
- **Total size:** 2.0MB  
- **Extensions:** {'.pt': 4, '.json': 1}  
- **Newest file:** `data/models/deep/training_results.json` (526.0B, 2026-02-05)  
- **Schema:** dict with 3 keys: `['timestamp', 'lstm', 'transformer']`

---

### `data/monitoring/` state-backup + awake-hold artifacts (2026-09-02)

- **`state_backup.json`** — heartbeat of the daily 04:45 off-disk backup
  (`com.seriea-pipeline.state-backup` plist → `scripts/utils/state_backup.py`): tars
  `data/betting` + `data/fantacalcio` + `data/monitoring` + `pipeline_state.json` (~6 MB)
  into iCloud Drive `~/Library/Mobile Documents/com~apple~CloudDocs/seriea-backups/`,
  keeps newest 14. `monitor.check_state_backup` warns when >50h old. Restore:
  `tar -xzf <newest archive>` from the repo root.
- **`caffeinate.pid`** — singleton pidfile for the match-day awake hold
  (`scheduler._ensure_awake_hold`): kickoff within 2h → `/usr/bin/caffeinate -s` (AC
  power only) through the last such kickoff +45 min, so idle sleep cannot swallow the
  T-30 bet-commit window. Cannot WAKE a sleeping Mac (that needs a sudo pmset schedule).

### `data/monitoring/reports/`

- **Description:** Monitoring cycle reports  
- **File count:** 8  
- **Total size:** 574.1KB  
- **Extensions:** {'.json': 8}  
- **Newest file:** `data/monitoring/reports/cycle_20260419_090004.json` (13.5KB, 2026-04-19)  
- **Schema:** dict with 6 keys: `['timestamp', 'drift_check', 'calibration_check', 'retrain_check', 'ab_check', 'alerts']`

---

### `data/predictions/`

- **Description:** Legacy prediction dumps + the live component ledger  
- **File count:** 10  
- **Total size:** 3.4MB  
- **Extensions:** {'.json': 9, '.jsonl': 1}  
- **Newest file:** `data/predictions/component_ledger.json` (live, 2026-09-04+)  

| File | Writer | What it is |
|---|---|---|
| `component_ledger.json` | `scripts/prediction/component_ledger.py` (3 fail-soft scheduler hooks: pre-kickoff snapshot, settle sweep, post-pipeline sweep) | **The ensemble's calibration flywheel** (2026-09-04). Per SA match: each core component's 1X2 probs (`ml/market/xg/player_xg/factor`) + the ensemble's betting probs + `weights_applied` + `ml_reasons` (SHAP top-5) + `ml_drift`, upserted freely until THAT match's kickoff then frozen — ex-ante by construction, post-hoc rows refused. `settle()` grades vs `matches.parquet` (multiclass Brier / log-loss / pick-correct per component). `rot_alarm()` (recent-20 vs trailing-100 Brier, Δ>0.04) and `drift_alarm()` (≥15 serving features outside training bands) are change-gated notifies. `refit_weights()` needs ≥100 all-core settled rows, shrinks 0.5 toward production, deploys `data/models/ensemble_weights.json` ONLY on a time-ordered holdout log-loss win — which the engine loads at init (precedence: ledger > legacy `data/feedback/optimized_weights.json` > hardcoded). NOTE: the legacy feedback loop (`feedback_analyzer` → `weight_optimizer`) is starved at n_settled≈2 because its `results.json` side holds ~1 match; the ledger settles against matches.parquet instead. **O/U leg (2026-09-04, the model that places bets):** each row also carries `ou: {"1.5"|"2.5": {ml, poisson, served, w}}` from `goal_predictions.json` (only lines where the ML leg fired), `total_goals` after settlement, and `ou_grades: {line: {ml|poisson|served: {brier, log_loss, correct}}}` — binary grades vs the real goal total. `summary()["ou"]` rolls them up; `refit_ou_blend()` (≥60 rows per line, 1-D grid, shrink 0.5, same holdout gate) deploys `data/models/ou_blend_weights.json`. |
| `data/models/ensemble_weights.json` | `component_ledger.refit_weights` (gate-pass only) | Deployed ensemble weight override with provenance (`n_settled`, holdout LLs, `fitted_at`). Engine validates keys/sum/range and fail-softs to constants. Does not exist until the first gate pass. |
| `data/models/ou_blend_weights.json` | `component_ledger.refit_ou_blend` (gate-pass only) | `{"weights": {"2.5": w_ml, "1.5": w_ml}, "fitted_at", "lines": {per-line fit report}, "provenance"}` — the ML share of the served O/U probability per bet line. `over_under_model.load_blend_weights` validates [0, 1] per line and fail-softs to `DEFAULT_ML_BLEND_WEIGHT` = 0.65. Lines not refit keep their deployed value. Does not exist until the first gate pass. |
| `data/models/universal/feature_quantiles.json` | `MLClassifier._get_train_quantiles` | Per-feature training [q0.005, q0.995] bands for the serving-drift tripwire; keyed on `features.parquet` `source_mtime` (rebuilds when the source moves, never trusts its own mtime). |

---

### `data/live/`

- **Description:** Live-poll artifacts  
- **File count:** 18  
- **Total size:** 3.8MB  
- **Extensions:** {'.json': 18}  
- **Newest file:** `data/live/2026-04-14.json` (3.2KB, 2026-04-13)  
- **Schema:** dict with 5 keys: `['date', 'polls', 'api_calls', 'matches', 'bet_tracking']`
- **Per-match live enrichment keys (2026-09-05):** `live_events` (newest first), `live_stats`
  (`{key: {home, away}}`), `live_player_stats`, `sofascore_id`, `sofascore_fetched_at`, plus
  `live_source` (`sofascore` | `espn` — which feed answered the last cycle) and
  `live_player_source` (which feed wrote `live_player_stats`: ESPN's roster gives shots,
  on target, goals, assists, fouls committed/drawn, offsides, cards, saves, goals conceded,
  own goals and a `minutes_played` DERIVED from the substitution events — no passes,
  tackles, duels or rating; a Sofascore read <180s old is not replaced by the fast tick)
  and `live_fetch_error` (present only when NO source answered; the
  previous cycle's events/stats are kept, never blanked). Writer:
  `scripts/data/live_monitor.poll_once` via `live_sofascore.fetch_live_data_for_matches`
  (Sofascore first, `live_espn` fallback, 10-min 403 breaker). A field is overwritten only
  when its source answered that cycle (`fetched` flags), so a 403 on `/statistics` cannot
  blank good stats. Reader: `/api/live` (pass-through), `/live` card, `prop_tracker`,
  `substitution_tracker`, `live_reconciliation`.

---

### `data/feedback/`

- **Description:** Feedback loop outputs (drift, calibration curves)  
- **File count:** 9  
- **Total size:** 55.6KB  
- **Extensions:** {'.json': 9}  
- **Newest file:** `data/feedback/lessons.json` (3.2KB, 2026-04-16)  
- **Schema:** dict with 2 keys: `['lessons', 'metadata']`

---

### `data/analysis/`

- **Description:** Ad-hoc analysis dumps  
- **File count:** 5  
- **Total size:** 53.5KB  
- **Extensions:** {'.json': 5}  
- **Newest file:** `data/analysis/decay_sweep_premier_league_catboost.json` (1.4KB, 2026-04-13)  
- **Schema:** list of 6

---

### `data/experiments/`

- **Description:** Experiment results  
- **File count:** 2  
- **Total size:** 69.0KB  
- **Extensions:** {'.jsonl': 1, '.json': 1}  
- **Newest file:** `data/experiments/drift_report.json` (67.9KB, 2026-02-16)  
- **Schema:** dict with 4 keys: `['critical', 'warning', 'ok', 'timestamp']`

---

### `data/squads/`

- **Description:** Squad snapshots  
- **File count:** 2  
- **Total size:** 152.0KB  
- **Extensions:** {'.json': 2}  
- **Newest file:** `data/squads/current_squads.json` (120.5KB, 2026-04-21)  
- **Schema:** dict with 4 keys: `['fetched_at', 'source', 'season', 'teams']`

---

### `data/understat/`

- **Description:** Historic per-season understat dumps  
- **File count:** 18  
- **Total size:** 922.8KB  
- **Extensions:** {'.parquet': 18}  
- **Newest file:** `data/understat/all_player_stats.parquet` (235.7KB, 2026-02-03)  
- **Schema:** 20 cols, 4616 rows. First cols: `['league_id', 'season_id', 'team_id', 'player_id', 'position', 'matches', 'minutes', 'goals', 'xg', 'np_goals', 'np_xg', 'assists']`

---


---

## 17. LEAGUE FEATURE PARITY — the EPL table is 427 columns narrower than Serie A

**Measured 2026-08-02.** `features_serie_a.parquet` is `(7980, 1334)`;
`features_premier_league.parquet` is `(7909, 909)`. 907 columns are common, 2 are
EPL-only, and **427 exist in Serie A and not in the EPL table at all**.

### The gap is structural, not a fill-rate problem

For the 907 shared columns, EPL coverage is level with Serie A over the last three
seasons (`ss_roll_*` 99.7% vs 99.8%, shot/xG-zone 97.7% vs 98.1%, referee 97.5% vs
98.8%, elo 99.8% both). Nothing shared is thinly populated for the EPL. The
asymmetry is entirely *which columns exist*.

All 427 missing columns are well populated on the Serie A side — 170 family stems,
most at 94–100% filled over recent seasons. They are real signal in Serie A, not
dead columns that happen to be absent for the EPL.

### Severity: unused width, NOT a live inference bug

The production model `data/models/universal/catboost_no_odds_metadata.json`
(variant `universal/no_odds__phase5_v1`) uses **126 features, of which zero come
from any of the 427 missing families.** So these columns are not arriving as NaN at
EPL inference — they are not requested at all, for either league. Building them for
the EPL changes nothing until a retrain selects them. **Do not read this section as
"the EPL model is missing a third of its features."**

### Input data is at full parity — this is not a scraping gap

| file | Serie A | Premier League |
|---|---|---|
| `match_team_stats` | 20,270 rows × 54 cols | 20,268 rows × 54 cols |
| `shotmap_stats` | 6,688 rows × 30 cols | 6,698 rows × 30 cols |
| `player_match_stats` | 101,875 rows × 80 cols | 97,003 rows × 80 cols |

Same columns, same three recent seasons, near-identical row counts. The EPL data has
been scraped all along. Both feature tables were also rebuilt within 18 minutes of
each other on 2026-08-02, so **staleness is ruled out** — the current build produces
these families for Serie A and not for the EPL.

### Attribution — confirmed

- **`fh_*` first-half splits (28 cols)** — `features/first_half_splits.py:122`
  filters season directories with `or "premier_league" in season_dir.name`, an
  explicit in-code exclusion of the EPL. This one is deliberate, whatever the
  original reason.
- **`features/player_depth.py:42`** and **`features/player_xg_model.py:336`** each
  hardcode `player_match_stats.parquet` with no `_premier_league` sibling and no
  glob — the documented "helper reads only the SA file" bug class from
  `CLAUDE.md`. Candidate source of the `adv_*` (76) / `tagg_*` (52) / `gk_*` (8)
  block, **not yet confirmed end-to-end** to be the producer of those exact columns.

### Attribution — ruled out

- **`features/sofascore_features.py` is NOT a cause.** It globs
  `match_team_stats_*.parquet` and `shotmap_stats_*.parquet` (lines 660, 713) and
  loads league-specific player files (line 59) — fully dual-league. A filename grep
  flags it, which is why the grep is a hypothesis and not the finding.
- **`features/_utils.py`'s Serie A-only `_PMS_PATH`** feeds a Sofascore-id bridge
  with **zero callers**. Dead code, not a live gap.

### Correctly absent — do not "fix" these

- **`coppa_matches_last_7d` / `coppa_matches_last_14d` (4 cols)** — Coppa Italia.
  There is no EPL equivalent; absence is correct.
- `altitude_advantage`, `long_travel` (`features/venue.py`) are Serie A geography
  features and are *likely* correctly absent — **not verified**, flagged here so the
  next pass checks rather than assumes.

### The remaining ~250 columns are unattributed

`ct_*` card timing (26), `captain_*` (6), `formation_*` (8), transfers (8), missing
players (8), subs (8), squad/spend (12), `fb_roll_*`/`fb_diff_*` (~60), the
`ss_roll_*` shot-type-share subset and `shot_*`/`xg_share_*` derived layer (~100).
Their builders have not been traced to a cause. Given the severity finding above,
tracing them is only worth doing as part of a decision to retrain the EPL model on
the wider feature set — which is a deliberate call, not a cleanup.

---

## Fantacalcio (`data/fantacalcio/`)

**2026-08-26: `data/parsed/understat_players.parquet` now carries five leagues**, not two —
La Liga, Bundesliga and Ligue 1 (2017-2025) were added beside Serie A and the EPL, ~9,600
new rows, scraped via `scripts/data/refresh_understat_players.py` (its `LEAGUES` dict is
the extension point; Selenium required — plain GETs still return a data-free shell,
re-measured 2026-08-26). Understat `player_id` is stable across leagues, so cross-league
careers join without name matching. Sole consumer of the foreign rows: the auction board's
foreign-informed tier (conversion factors ×0.93 goals / ×1.00 assists / ×1.05 yellows,
measured on 164 league movers; minutes weighted ×0.85). The betting pipeline reads only
the Serie A rows and is unaffected.

Auction and season-scoring data for the private Fantacalcio league. Independent of the
betting pipeline — nothing here feeds `features_*.parquet` or any model.

| File | Written by | What it is |
|---|---|---|
| `listone_2026_2027.xlsx` | manual download | Official listone. `Id` is the fantacalcio player id and is the join key for everything below. |
| `auction_board.json` | `scripts/fantacalcio/build_auction_board.py` | Projections, prices, walk-away caps, fallback chains. Served by `/fantacalcio`. |
| `backtest_*.csv` | `scripts/fantacalcio/backtest.py` | Walk-forward validation of the projection over eight past auctions. |
| `voti/stats_{season}.parquet` | `scripts/fantacalcio/voti.py` | Season-aggregate media voto / fantamedia per player. |
| `voti/round_{season}_{rr}.parquet` | `scripts/fantacalcio/live_scores.py` | **Per-round voti.** One row per player who appeared: `pid, slug, team, role, voto, cards, bonus, fantavoto, played`. |
| `my_team.json` | `scripts/fantacalcio/import_rosters.py` (since 2026-09-02; previously `POST /api/fantacalcio/my-team`) | The squad actually won at auction (budget 500). Derived from the league export below; re-import overwrites it. |
| `league_rosters_source.xlsx` | manual download (Leghe "Rose" export) | All 10 league squads, 3-column blocks. Drop a fresh one + re-run the importer after any trade. |
| `league_rosters.json` | `scripts/fantacalcio/import_rosters.py` | All 10 squads matched to board ids, spent/unmatched per team. Served by `/api/fantacalcio/league`. |
| `tracker.json` | `scripts/fantacalcio/tracker.py` | Per-round scores for that squad. Rebuilt on demand by `/api/fantacalcio/tracker` when the roster moves or the file is >6h old. |
| `xi_advice.json` | `scripts/fantacalcio/xi_advisor.py` | Who to field next giornata: module + XI + bench (the league's 9 ordered slots: 1P/3D/3C/2A) + tribuna, from live levels x fixture terms x p_play. Priors for vote-less players follow `_board_priors`: auction model (season_points/mv_hat) > real latest-season record shrunk by LEVEL_K (e.g. David 6.33/30g) > 6.0 floor — never a bare 6.0 when history exists. Rebuilt on demand by `/api/fantacalcio/xi-advisor` when roster/tracker move or >6h old. |
| `indisponibili.json` | `scripts/fantacalcio/probabili.fetch_indisponibili` | Per-club injured/suspended lists from fantacalcio.it/indisponibili-serie-a (names only, no pids — matched by accent-folded surname WITHIN the club, ambiguity fails open). 6h TTL, stale-on-failure, sentinel >=15 clubs incl. Inter. Default for an injured row is OUT (specimen 2026-09-02: 42/43 were hard outs); doubt needs an explicit this-round-hope marker. Drives p_play for players the probabili page does not list: squalificato 0.02, infortunato 0.05, dubbio cap 0.35; title-bound news risk hits are the weakest tier (cap 0.60, never zeroes). Hierarchy: probabili listing (pid-exact fresh) always wins, the injury then rides along as `avail_note`. Every tier labels `p_play_src`, so pred_ledger grades each source against who actually got a voto (the refit path). |
| `tracker_heartbeat.json` | `scripts/fantacalcio/tracker._write_heartbeat` | Liveness stamp written on EVERY tracker run (`ran_at`, `ok`, `error` — a push failure lands in `error` with `ok: true`). `monitor.check_fantacalcio_health` alarms on absence (>30h CRITICAL — job runs 4×/day), on `ok: false`, and — only when `xi_advice.first_kickoff` is within 10 days — on probabili/indisponibili `fetched_at` older than 30h (the caches serve stale-forever on failure by design, so this check is their only alarm; off-season staleness is legitimate and ignored). |
| `probabili.json` | `scripts/fantacalcio/probabili.py` | Probable lineups from fantacalcio.it (starters/reserves/ballot pcts per pid + since 2026-09-02 the page's own per-player titolarità bar, `pct` — verified live: 479/479 players, starters 55–90, reserves 1–60). 6h-TTL cache; on fetch/schema failure the last good cache is served. p_play ladder: ballottaggio pct > titolarità bar (src `titolarita`) > flat P_STARTER/P_RESERVE fallback > model. NOTE: the bar measures P(starts), which understates a regular sub's P(voto) — the ledger's per-src buckets are the refit path. |
| *(consumes)* `data/upcoming/confirmed_lineups.json` | `scripts/fantacalcio/lineup_check.py` (reader; writer is the betting pipeline's `lineup_fetch` stage, T-55) | Official XIs at T-60: the scheduler's lineup_fetch hook calls `run_official_lineup_check()` — accent-folded token-suffix name matching (initial-aware: "Gaspar K." ↔ "Kialonda Gaspar") scoped per club, uniqueness both sides, ambiguity fails open; p_play overrides 0.97 titolare / 0.15 panchina / 0.03 escluso (`official_out` only when ≥70% of the club's board rows matched — a rename must not read as an exclusion). Sources labeled `official_xi/official_bench/official_out` for pred_ledger. Rebuilds xi_advice and pushes a 🚨 diff only while the league deadline (round's first kickoff) is still open; feed older than 3h is refused. |
| `calendar_coppa_del_nonno.xlsx` / `calendar_hunger_games.xlsx` | manual download (Leghe calendar exports) | Both competitions' full schedules. Re-import: `python3 -m scripts.fantacalcio.import_rosters --calendars`. |
| `league_schedule.json` | `scripts/fantacalcio/import_rosters.py --calendars` | Parsed calendars: CDN 10 group rounds (gironi A/B, SA 3..29, Riposa rows), HG 36 rounds (SA 3..38). Name-validated parse (never positional — score cells fill in as rounds play). |
| `club_congestion.json` | `scripts/fantacalcio/xi_advisor._club_congestion` (tracker job) | Per Serie A club: last competitive match (ALL competitions via Sofascore team events — sees Coppa Italia/Europe) + rest days before its next SA fixture. 12h TTL, stale-on-failure, 3-strike breaker. CONTEXT ONLY: measured 2025-26 within-player fantavoto cost of short rest = −0.06 ± 0.08 (zero) — no coefficient, just the 😴 flag. |
| `team_pulse.json` | `scripts/fantacalcio/team_pulse.py` (tracker job) | Press pulse per club: Italian-lexicon headline scores with 7d half-life decay + a Naive Bayes that trains on each settled round's result (headlines parked per (club, sa_round), win/loss-labeled; NB blended in by evidence volume). Zero paid API (the anti-Groq design). Served by `/api/fantacalcio/team-pulse`. |
| board `mercato_synced_at` + `sync_mercato` | `scripts/fantacalcio/import_rosters.sync_mercato` (tracker job, weekly TTL) | Reconciles auction_board.json against the LIVE fantacalcio.it quotazioni page: placeholder-pid adoption (auction-time 99xxx ids -> real pids, auction priors kept), intra-SA club moves (`team_listone` preserves the original), status verification, new-arrival rows. Deliberately NEVER marks departures: the quotazioni page keeps rows of players who left Serie A (measured 2026-09-02 — Di Gregorio still listed under JUV a week after Bournemouth), so DEPARTED belongs to the wiki-transfers pass alone: `_mark_departures` (same job) name-matches wiki_transfers rows (last-name token + from_club[:5], latest move) whose to_club is outside the live SA club set — BUT a pid on the current probabili page refutes ANY departure mark (pid-exact + fresh beats name-matched; restored 8 false marks incl. MW3 starters Mandas/Oyono A. on 2026-09-02). |
| `trades.json` | `scripts/fantacalcio/trades.build_trades` (tracker job) | Trade windows: best swap per (rival, my-role -> their-role) where the EXACT best-XI delta is positive on BOTH sides (cheap starter-line prefilter, then team_strength both ways), plus a full team-strength table. Offer evaluator: `python3 -m scripts.fantacalcio.trades --offer --give "X" --get "Y,Z"`. Served by `/api/fantacalcio/trades`. |
| `league_standings.json` | `scripts/fantacalcio/tracker._standings_from_schedule` (tracker job) | Real H2H tables per competition from the calendar exports' score cells (the Leghe page 404s anonymously — probed 2026-09-02). A fixture counts only when its score cell matches N-N, so unplayed/unknown shapes yield an empty table, never a wrong one. Refresh path: re-drop the two calendar xlsx + `--calendars` after each giornata. Served by `/api/fantacalcio/standings`. |
| `svincolati.json` | `scripts/fantacalcio/xi_advisor.build_svincolati` (tracker job) | Free-agent radar: the unowned half of the listone enriched with the same machinery as any roster (live levels, probabili, discipline), ranked by p_play x level per role; `upgrade_over` names my weakest same-role player on the SAME score basis (never raw level vs prior). Served by `/api/fantacalcio/svincolati`. |
| `rivals.json` | `scripts/fantacalcio/xi_advisor.build_rivals` (tracker job) | + since 2026-09-02: `next_opponents` per competition from league_schedule.json, per-rival future `meetings`, each rival's predicted `xi`, congestion flags both sides. Rival matrix: every league team's expected XI total (their roster through the SAME advise machinery, incl. their defense modifier; since 2026-09-02 restricted to that rival's OBSERVED module repertoire when rival_modules.json holds any — `module_src`: osservato/stimato) + my P(win) per opponent (normal approx, per-player sd measured on 2 seasons of voti, split-half r=0.47) + risk-tilted alternative XI when it gains ≥1pp. Calendar-free: covers both competitions (Coppa Del Nonno groups + Hunger Games). Served by `/api/fantacalcio/rivals`. |
| `rival_modules.json` | `scripts/fantacalcio/xi_advisor.record_fielded` | Per-rival ledger of ACTUALLY fielded modules `{team: [{round, module, at}]}`. Fed three ways: Telegram screenshot of the Leghe formation page (bot vision auto-records via the `Modulo avversario osservato: <squadra> <modulo>` reply convention), `python3 -m scripts.fantacalcio.xi_advisor --fielded "Team=4-4-2" [--round N]`, or telling Claude in-session. `_observed_modules` returns the repertoire most-recent-first; build_rivals restricts each rival's advise() to it (full-MODULES fallback if the observed set can't field an XI). |
| `pred_ledger.json` | `scripts/fantacalcio/pred_ledger.py` | Predicted-vs-actual loop: per-giornata forecast frozen EX-ANTE at first kickoff (post-kickoff writes refused), reconciled vs the round voti parquet after a 4-day grace, per-player err_fv + play Brier + per-source p_play calibration. Since 2026-09-04 the loop CLOSES: `frozen_entry` serves the stored forecast as the advice mid-round (no remaining-fixture churn), and `exp_bias()` writes the shrunk per-role error correction (n/(n+60), cap ±0.5) back into the next round's exp via `_apply_exp_bias` — self-arming, no manual refit. Served by `/api/fantacalcio/pred-ledger`. |
| `news.json` | `scripts/fantacalcio/news.py` | Player headlines from Gazzetta/CorSport/Tuttosport RSS, surname-matched to the 25-man roster. 14-day accumulator, dedup by link. Display-only (`/api/fantacalcio/news`); refreshed by the tracker job. |
| `player_rates.json` | `scripts/fantacalcio/player_rates.py` | Per-player xg90/ast90 from Sofascore `player_match_stats` — **xG-era rows only (2022-23+)**: pre-era xG is 0%-filled, and mixing it in as zeros deflated every prior 4× (measured 2026-09-04); within the era NaN xg == "no shots" (0 goals across 18,496 NaN rows) so it imputes 0. Shrunk 900 pseudo-min toward league mean + era moment-match multipliers (c≈0.98/0.99). Gated BEFORE wiring: held-out 2025-26+ walk-forward P(goal) skill +0.073, P(assist) +0.020. Watermark cache (`source_mtime`), rebuilds when the parquet moves. Feeds `_apply_rates`: ⚽/🅰 percentages everywhere + the role-mean-relative rate tilt (fades LEVEL_K/(LEVEL_K+n_level), cap ±0.35, never on market-priced players). Cards were measured the same day and REJECTED (skill +0.018 < 0.02, tails miscalibrated) — don't re-run. |
| `round_context.json` | `scripts/fantacalcio/xi_advisor.build_round_context` (tracker job) | Hub→fanta bridge: for every fixture of the advice round (played included), the main prediction system's card from `data/upcoming/predictions.json` — 1X2 probs, xG, draw score, confidence, market presence — plus which of MY players are exposed. Row-level league gate (the 2026-08-27 leak class). Attached to the `/api/fantacalcio/xi-advisor` payload when rounds match; rendered as the per-game strip on `/fantacalcio`. |

### Per-round voti — parsing facts that are not on the rules page

These were established by reconciling 936 of 942 published fantavoti in round 1 of
2025-26, not by reading documentation. Each one is a silent wrong answer if missed:

- **`data-value="55"` is the senza-voto sentinel**, not a rating. Whole ratings render bare
  (`6` = 6.0) and halves carry a comma (`6,5`). The count matches the sv total exactly in
  every round checked (33/33, 36/36, 33/33). The parser rejects anything outside `[0,10]`
  rather than special-casing the literal.
- **"Player of the match" is a bonus column worth ZERO.** Weighting it +1 is wrong for the
  nine players a round who get it.
- **Cards are not bonus columns.** They live in the grade span's class
  (`player-grade yellow-card`), so a `player-bonus`-only parser misses every booking.
  Yellow −0.5, red −1.0.
- **The fantavoto must not be range-guarded** — it legitimately exceeds 10 and can go
  negative. The `[0,10]` guard applies to the *voto* alone.
- **Three `.pill` blocks per player are three independent voti providers** and they
  genuinely disagree (143 of 314 rows in round 1). `LIST_DEFAULT = 0` is Fantacalcio
  Italia, which is what FantaLeghe uses.
- **A round that parses <100 rows is treated as NOT PLAYED.** An unplayed round and a
  broken selector both return HTTP 200 with zero rows; only one is benign, so
  `played_rounds()` stops at the first empty round rather than scanning past it.

### Season scoring

`tracker.py` reports two numbers per round and the gap between them is the point:
`settable` is the eleven that could actually have been fielded (module and XI chosen from
the projection *before* kickoff, then up to three same-role substitutions for starters with
no voto), and `hindsight` is the best eleven the roster could have produced knowing every
result. The ceiling is unattainable by construction; reporting it as "my score" would be a
lie, which is why both are computed and the page labels them.

The modificatore di difesa is scored on the **raw voto** of the keeper plus the best three
fielded defenders, using the same table and voto d'ufficio the auction board priced with. A
three-defender module forfeits it entirely, so the module search compares four- and
three-defender shapes on total points *including* the modifier.

### Pick engine artifacts (2026-09-05)

| File | Writer | Shape | Notes |
|---|---|---|---|
| `data/upcoming/picks.json` | `scripts/betting/picks.build_picks` (via `save_bet_slip`, morning dry + T-30) | `{generated_at, league, band, counts{VALUE,LEAN,NO_EDGE}, n_journaled, picks[{match, date, kickoff_utc, label, stage?, pick, lean?, alternatives[3], most_probable?, reason, n_rows, n_priced, n_positive, n_overconfident, n_longshot_edges, journaled_bet_id?, prices_fetched_at}]}` | Read by Telegram `/picks`, `/api/dashboard` (`pick`), `/api/match-markets` (`pick`). VALUE copied from the slip, never recomputed. |
| `data/upcoming/pick_markets_raw.json` | `odds_fetcher.fetch_pick_markets` (T-6h/T-3h stages + pre-kickoff, 45-min gate/event) | `{fetched_at, league, events{<odds_api_event_id>: {home, away, home_raw, away_raw, commence, fetched_at, bookmakers[{title, markets[{key, outcomes[{name, description, point, price}]}]}]}}}` | RAW outcome naming kept verbatim (team names in `h2h_h1`, `Juventus/Draw` in HT/FT, `Juventus:1\|AC Milan:0` in correct score, full player names in `description`). 16 markets, 16 credits/event. |
| `data/betting/picks_journal.json` | `picks.journal_lean` (T-30 only, kickoff ≤ 3h) | bet_journal schema + `extra{bet_type, player, team, source, tier, side, line}`; `market` = Odds API key, `selection` embeds the player | Flat €10 paper. Settled by `picks.settle_picks` from auto_settle; `picks.picks_record()` = per-market ROI/CLV bar. Never touches bankroll/history. |
