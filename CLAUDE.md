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
