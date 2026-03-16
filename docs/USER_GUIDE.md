# User Guide

## Installation

### Requirements

- Python 3.11+
- macOS or Linux (tested on Apple M2, 16GB RAM)
- ~2GB disk space for historical data

### Setup

```bash
cd seriea_pipeline

# Install core dependencies
pip install -e .

# Install additional dependencies not in pyproject.toml
pip install catboost tensorflow flask python-dotenv selenium botasaurus
```

### API Keys

Copy the environment template and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Required | Source | Purpose |
|----------|----------|--------|---------|
| `ODDS_API_KEY` | Yes | [the-odds-api.com](https://the-odds-api.com/) | Live betting odds |
| `OPENAI_API_KEY` | No | [platform.openai.com](https://platform.openai.com/) | AI reasoning narratives |
| `PERPLEXITY_API_KEY` | No | [perplexity.ai](https://www.perplexity.ai/) | Sentiment analysis |
| `APIFOOTBALL_KEY` | No | [rapidapi.com](https://rapidapi.com/api-sports/api/api-football) | Confirmed lineups |
| `FOOTBALLDATA_KEY` | No | [football-data.org](https://www.football-data.org/) | Squad roster fallback |
| `FLASK_SECRET_KEY` | No | Generate locally | Web app session security |

The pipeline works without optional keys but produces richer predictions with them.

## Common Workflows

### Run the Full Prediction Pipeline

```bash
# Standard run (fetches odds, runs all 30 steps)
python3 scripts/run_full_pipeline.py

# Quick mode (skip API calls, use cached data)
python3 scripts/run_full_pipeline.py --quick

# Pre-kickoff mode (fetch confirmed lineups, ~25 seconds)
python3 scripts/run_full_pipeline.py --pre-kickoff

# Snapshot only (fetch + save odds data, ~6 API credits)
python3 scripts/run_full_pipeline.py --snapshot-only

# Custom bankroll amount
python3 scripts/run_full_pipeline.py --bankroll 2000

# Live monitoring (polls every 15 minutes)
python3 scripts/run_full_pipeline.py --live-monitor

# Single live poll
python3 scripts/run_full_pipeline.py --live-once
```

### View the Web Dashboard

```bash
python3 web/app.py
# Open http://localhost:5001
```

The dashboard shows:
- **Main page** (`/`): Pipeline status, predictions, performance metrics
- **Betting page** (`/betting`): Match cards with value indicators, extended markets, player props

### Train ML Models

```bash
# Full optimized pipeline (recommended):
# Feature selection + Optuna tuning + calibration + ensemble
seriea ml optimize --trials 50

# Train universal models with walk-forward CV
seriea ml train

# Train rich model (all features, single season)
seriea ml train-rich --season 2025-2026

# Run walk-forward backtest
seriea ml backtest

# Compare with/without odds features
seriea ml ablation
```

### Inspect Models

```bash
# Evaluate model on last season
seriea ml evaluate --model xgboost

# View feature importances
seriea ml importance --top-n 30

# Check pipeline status
seriea status
```

## CLI Reference

The CLI is installed as `seriea` via the package entry point. All commands can also be run via `python3 cli.py <command>`.

### Data Collection Commands

| Command | Description | Options |
|---------|-------------|---------|
| `seriea scrape-fixtures` | Scrape fixture list from FBref | `--season` |
| `seriea scrape-matches` | Download match report HTML pages | `--season`, `--limit` |
| `seriea import-existing` | Import HTML from old project structure | - |
| `seriea fetch-odds` | Download odds from football-data.co.uk | `--season` |
| `seriea fetch-weather` | Fetch weather data for parsed matches | - |
| `seriea fetch-transfers` | Scrape Transfermarkt data | `--season` |
| `seriea import-historical` | Import football-data.co.uk results | - |
| `seriea fetch-referees` | Scrape referee assignments | `--season` |

### Processing Commands

| Command | Description | Options |
|---------|-------------|---------|
| `seriea parse` | Parse raw HTML into Parquet tables | `--season`, `--reparse` |
| `seriea features` | Build ML-ready feature table | `--season` |
| `seriea run-all` | Full pipeline: scrape + parse + features | `--season`, `--limit` |
| `seriea status` | Show pipeline status and data counts | - |

### ML Commands

| Command | Description | Options |
|---------|-------------|---------|
| `seriea ml train` | Train universal models | `--model`, `--no-validate` |
| `seriea ml train-rich` | Train rich single-season model | `--season`, `--model` |
| `seriea ml optimize` | Full optimization pipeline | `--trials`, `--top-k`, `--corr-threshold` |
| `seriea ml evaluate` | Evaluate model on last season | `--variant`, `--model` |
| `seriea ml importance` | Show feature importances | `--variant`, `--model`, `--top-n` |
| `seriea ml backtest` | Walk-forward backtest | - |
| `seriea ml ablation` | With/without odds comparison | - |

### Pipeline Runner Options

| Flag | Description |
|------|-------------|
| `--quick` | Skip API calls, use cached data |
| `--pre-kickoff` | Fetch confirmed lineups (~25s) |
| `--snapshot-only` | Odds snapshot only (steps 1-6, ~6 credits) |
| `--live-monitor` | Continuous polling every 15 minutes |
| `--live-once` | Single live poll (2 API credits) |
| `--live-status` | Show live monitoring status |
| `--bankroll N` | Set custom bankroll amount |

## Configuration

### `config/settings.py`

Core settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `SEASONS` | 2005-2026 | Seasons to process (21 total) |
| `REQUEST_DELAY_SECONDS` | 12 | Delay between scraping requests |
| `ROLLING_WINDOWS` | [3, 5, 10] | Game windows for rolling stats |
| `SERIE_A_COMP_ID` | 11 | FBref competition ID |

### `ml/config.py`

ML pipeline configuration:

| Config Class | Key Settings |
|-------------|-------------|
| `ModelConfig` | Default XGB/LGB/CatBoost hyperparameters, early stopping (50 rounds), draw weight multiplier (1.5x) |
| `ValidationConfig` | Walk-forward CV: min 3 training seasons, 1 test season, expanding window |
| `FeatureConfig` | Universal NaN threshold (20%), correlation pruning (0.70), max 60 features |
| `TuningConfig` | 80 Optuna trials, LR-estimator coupling (base: lr=0.03, n=600) |
| `CalibrationConfig` | Isotonic bounds [0.01, 0.99] |
| `EnsembleConfig` | Meta-learner C candidates [0.01, 0.1, 1.0, 10.0] |

### Ensemble Weights

Configured in `scripts/ensemble_prediction_engine.py`:

| Method | Weight | Notes |
|--------|--------|-------|
| Factor | 0.27 | 21-season validated |
| xG Poisson | 0.27 | Bookmaker-style |
| ML Classifier | 0.14 | Gradient boosting |
| Player xG | 0.05 | Boosted to 0.12 with confirmed lineups |
| Deep Learning | 0.07 | Neural network |
| Market | 0.20 | Sharp bookmaker consensus |

### Bankroll Settings

In `features/bankroll_manager.py`:

| Setting | Value | Description |
|---------|-------|-------------|
| Kelly fraction | 0.15 | Quarter Kelly for safety |
| Max per bet | 5% | Maximum single bet stake |
| Drawdown limit | 20% | Reduces stakes when breached |

## API Credit Management

### The Odds API

| Tier | Credits/Month | Cost |
|------|--------------|------|
| Free | 500 | $0 |
| Starter | 100,000 | $50 |

Pipeline usage per run: ~6 credits (odds) + ~2 credits (scores for settlement).

Check remaining credits: https://the-odds-api.com/account/

### API-Football

Free tier: 100 requests/day. The pipeline uses ~1 request per run for confirmed lineups.

## Output Files

After a pipeline run, key outputs are:

| File | Contents |
|------|----------|
| `data/upcoming/predictions.json` | Match predictions with probabilities |
| `data/upcoming/predictions_archive.json` | Historical prediction archive |
| `data/upcoming/odds.json` | Current bookmaker odds |
| `data/betting/betting_slip.json` | Recommended bets with stakes |
| `data/bankroll/state.json` | Bankroll tracking state |
| `data/upcoming/bookmaker_analysis.json` | Sharp/soft divergence data |

## Scheduling

### Cron Setup (Recommended)

```bash
crontab -e

# Daily pipeline run at 8 AM
0 8 * * * cd /path/to/seriea_pipeline && python3 scripts/run_full_pipeline.py >> logs/pipeline.log 2>&1

# Pre-match check 2 hours before Saturday/Sunday kickoffs
0 12 * * 6,0 cd /path/to/seriea_pipeline && python3 scripts/run_full_pipeline.py --pre-kickoff >> logs/pre_kickoff.log 2>&1
```

### Daily Workflow

1. **Morning**: Run full pipeline (fetches latest odds)
2. **Pre-match** (2h before kickoff): Run with `--pre-kickoff` for confirmed lineups
3. **Post-match**: Results auto-settled if Odds API scores available
4. **Weekly**: Review performance dashboard, adjust bankroll if needed
