# Serie A / Premier League Prediction Pipeline

A football match-prediction and betting-intelligence pipeline for Italian Serie A and the English Premier League. It scrapes historical and live data, engineers a large feature table, trains calibrated ML models with walk-forward validation, blends multiple prediction methods into an ensemble, and serves results through a Flask dashboard. A separate module forecasts the 2026 World Cup.

## What it does

- Scrapes match data, fixtures, odds, weather, referees, and market values from public sources (FBref, Football-data.co.uk, Open-Meteo, Understat, Sofascore, Transfermarkt).
- Parses raw HTML into structured Parquet tables and builds a multi-step feature-engineering pipeline (form, xG trends, Elo, head-to-head, referee and weather context, market intelligence, and more).
- Trains per-league models (XGBoost / LightGBM / CatBoost) with walk-forward cross-validation, Optuna tuning, isotonic calibration, and a draw-aware weighted ensemble.
- Generates 1X2 predictions plus extended-market outputs and value-bet / bankroll signals.
- Serves a Flask web dashboard for predictions, match context, and betting intelligence.
- Forecasts the 2026 World Cup via a dedicated simulation/knockout module.

Model performance is read live from model metadata, never from documentation. See `MODEL_STATUS.md` and run:

```bash
python3 scripts/diagnostics/print_model_status.py
```

## Tech stack

- **Python 3.11+**
- **Data:** pandas, pyarrow (Parquet), numpy, scipy
- **ML:** scikit-learn, xgboost, lightgbm, catboost, optuna
- **Scraping:** requests, beautifulsoup4, lxml (optional: selenium, botasaurus)
- **CLI:** click
- **Web:** Flask (optional extra)
- **LLM (optional):** google-genai, groq
- **Tooling:** ruff, mypy, pytest

## Coverage

- **Leagues:** Serie A and Premier League (`config/settings.py:LEAGUES`)
- **Seasons:** 21 seasons, 2005-2006 through 2025-2026 (`config/settings.py:SEASONS`)
- **Feature tables:** `data/features/features_serie_a.parquet`, `features_premier_league.parquet`, and a combined `features.parquet` (shapes and column inventory are documented in `DATA_CATALOG.md`).

## Install

```bash
pip install -e .

# Optional extras
pip install -e ".[web]"        # Flask dashboard
pip install -e ".[scraping]"   # selenium + botasaurus
```

### Environment

```bash
cp .env.example .env
# Edit .env with API keys (only the sources you use are required):
#   ODDS_API_KEY        - The Odds API (live odds)
#   GEMINI / GROQ keys  - optional LLM features
#   APIFOOTBALL_KEY     - confirmed lineups (optional)
```

## Run

### Full pipeline

```bash
# Full prediction pipeline (fetch odds, predictions, betting, dashboard data)
python3 scripts/pipeline/run_full_pipeline.py

# Use cached data, skip API calls
python3 scripts/pipeline/run_full_pipeline.py --quick

# Pre-kickoff mode (confirmed lineups)
python3 scripts/pipeline/run_full_pipeline.py --pre-kickoff

# Snapshot only (odds + bookmaker data)
python3 scripts/pipeline/run_full_pipeline.py --snapshot-only

# Choose leagues
python3 scripts/pipeline/run_full_pipeline.py --leagues serie_a,premier_league
```

### Web dashboard

```bash
python3 web/app.py
# Open http://localhost:5001   (override with PORT=...)
```

### CLI

The `seriea` CLI (entry point `cli:main`, also runnable as `python3 cli.py`) provides granular control:

```bash
# Data collection
seriea scrape-fixtures --season 2025-2026
seriea scrape-matches --season 2025-2026 --limit 10
seriea fetch-odds --season 2025-2026

# Processing
seriea parse --season 2025-2026
seriea features --season 2025-2026

# ML
seriea ml train                 # universal models (walk-forward CV)
seriea ml train-rich            # rich models (single season)
seriea ml optimize --trials 50  # feature selection + tuning + ensemble
seriea ml evaluate
seriea ml importance --top-n 30
seriea ml backtest
```

Run `seriea --help` and `seriea ml --help` for the full command list.

## Project layout

```
cli.py                       # Click CLI entry point
config/                      # Settings (leagues, seasons, paths, model config)
scraper/                     # Data collection (FBref, odds, weather, referees, ...)
parser/                      # HTML -> structured data
storage/                     # Parquet I/O and path config
features/                    # Feature engineering modules
ml/                          # Data loading, models, ensemble, tuning, calibration, eval
scripts/
  pipeline/                  # run_full_pipeline.py, scheduler, monitoring, retrain
  betting/                   # value-bet / parlay / bankroll logic
  prediction/                # prediction generation
  worldcup/                  # 2026 World Cup simulation + knockout module
  diagnostics/               # print_model_status.py and other probes
web/                         # Flask dashboard (app.py, templates, static)
data/                        # Parquet tables, trained models, caches (gitignored)
tests/                       # pytest suite
docs/                        # Architecture, user guide, data dictionary, dev, troubleshooting
```

## Key reference docs

- `MODEL_STATUS.md` — how to read live model performance (always from metadata, never markdown)
- `DATA_CATALOG.md` — authoritative reference for every data file and column
- `ARCHITECTURE_MAP.md` — per-file navigability map for the codebase
- `docs/ARCHITECTURE.md`, `docs/USER_GUIDE.md`, `docs/DATA_DICTIONARY.md`, `docs/DEVELOPMENT.md`, `docs/TROUBLESHOOTING.md`

## Development

```bash
python3 -m pytest tests/      # run tests
ruff check .                  # lint
mypy .                        # type check
```

## Status

Active personal project. Multi-league (Serie A + Premier League) with a World Cup 2026 module on the `feat/worldcup-2026` branch. Private — not for redistribution.
