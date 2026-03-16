# Serie A Prediction Pipeline

A production-grade football betting intelligence system for Italian Serie A. Combines 21 seasons of historical data (7,829 matches), real-time odds tracking, 6-method ensemble predictions, and automated bankroll management.

## Features

- **6-Method Ensemble**: Factor analysis, xG Poisson, ML classifier (XGBoost/LightGBM/CatBoost), player xG, deep learning, market-based
- **484-Column Feature Table**: 36-step feature engineering pipeline covering form, xG trends, Elo ratings, referee bias, weather, player impact, tactical metrics, and more
- **Real-Time Odds**: Multi-bookmaker tracking via The Odds API with sharp/soft divergence detection
- **ML Pipeline**: Walk-forward cross-validation, Optuna hyperparameter tuning, isotonic calibration, weighted average ensemble with draw-aware optimization
- **Extended Markets**: Over/Under, Handicap, Cards, BTTS, Corners, Double Chance, Team Totals, Exact Score, HT/FT
- **Bankroll Management**: Kelly criterion staking, drawdown limits, streak adjustments, closing line value tracking
- **Web Dashboard**: Flask app with match context strips, dual probability bars, player props, and value highlighting

## Quick Start

### Prerequisites

- Python 3.11+
- macOS / Linux (tested on Apple M2)

### Installation

```bash
cd seriea_pipeline
pip install -e .

# Additional dependencies not in pyproject.toml:
pip install catboost tensorflow flask python-dotenv selenium botasaurus
```

### Environment Setup

```bash
cp .env.example .env
# Edit .env with your API keys:
#   ODDS_API_KEY       - The Odds API (required for live odds)
#   OPENAI_API_KEY     - GPT-4o for AI reasoning (optional)
#   PERPLEXITY_API_KEY - Sentiment analysis (optional)
#   APIFOOTBALL_KEY    - Confirmed lineups (optional)
#   FOOTBALLDATA_KEY   - Squad rosters fallback (optional)
```

### Run the Full Pipeline

```bash
# Full 30-step pipeline (fetches odds, predictions, betting, dashboard)
python3 scripts/run_full_pipeline.py

# Quick mode (use cached data, skip API calls)
python3 scripts/run_full_pipeline.py --quick

# Pre-kickoff mode (confirmed lineups, ~25s)
python3 scripts/run_full_pipeline.py --pre-kickoff

# Snapshot only (odds + bookmaker data, ~6 API credits)
python3 scripts/run_full_pipeline.py --snapshot-only
```

### Web Dashboard

```bash
python3 web/app.py
# Open http://localhost:5001
```

## CLI Reference

The `seriea` CLI provides granular control over individual pipeline stages:

```bash
# Data collection
seriea scrape-fixtures --season 2025-2026
seriea scrape-matches --season 2025-2026 --limit 10
seriea fetch-odds --season 2025-2026
seriea fetch-weather
seriea fetch-transfers --season 2025-2026
seriea fetch-referees

# Processing
seriea parse --season 2025-2026
seriea features --season 2025-2026
seriea run-all --season 2025-2026

# ML training
seriea ml train                   # Universal models (walk-forward CV)
seriea ml train-rich              # Rich models (single season, all features)
seriea ml optimize --trials 50    # Full pipeline: feature selection + tuning + ensemble
seriea ml evaluate                # Evaluate on last season
seriea ml importance --top-n 30   # Feature importance ranking
seriea ml backtest                # Walk-forward backtest
seriea ml ablation                # Compare with/without odds features

# Status
seriea status                     # Pipeline status overview
```

## Project Structure

```
seriea_pipeline/
├── cli.py                          # Click CLI entry point
├── config/settings.py              # Global configuration
├── .env.example                    # API keys template
│
├── scraper/                        # Data collection
│   ├── fixtures.py                 # FBref fixture lists
│   ├── match_reports.py            # FBref match report HTML
│   ├── odds.py                     # Football-data.co.uk odds
│   ├── weather.py                  # Open-Meteo weather API
│   ├── referee.py                  # Worldfootball.net referees
│   ├── transfermarkt.py            # Market values & transfers
│   ├── lineup_fetcher.py           # API-Football lineups
│   └── historical.py              # Football-data.co.uk backfill
│
├── parser/                         # HTML -> structured data
│   └── match_page.py              # FBref match page parser
│
├── storage/                        # Data persistence
│   ├── paths.py                    # Path configuration
│   └── structured.py              # Parquet I/O
│
├── features/                       # Feature engineering (36 steps)
│   ├── build.py                    # Orchestrator
│   ├── base.py, rolling.py         # Core team stats
│   ├── h2h.py, strength.py         # Matchup features
│   ├── referee.py, weather.py      # Context features
│   ├── injury_impact.py            # Injury analysis
│   ├── pressing.py                 # PPDA from Understat
│   ├── bankroll_manager.py         # Kelly criterion engine
│   ├── market_intelligence.py      # Sharp/soft aggregation
│   └── bookmaker_analysis.py       # Bookmaker classification
│
├── ml/                             # Machine learning pipeline
│   ├── config.py                   # Model/tuning/feature configs
│   ├── data.py                     # DataLoader + time-series splits
│   ├── models.py                   # XGBoost/LightGBM/CatBoost wrappers
│   ├── training.py                 # train_universal, train_rich, train_optimized
│   ├── ensemble.py                 # WeightedAverageEnsemble (draw-aware)
│   ├── tuning.py                   # Optuna hyperparameter search
│   ├── calibration.py              # Isotonic regression calibrator
│   ├── evaluation.py               # RPS, ECE, Kelly, bootstrap CI
│   ├── feature_selection.py        # Importance + correlation pruning
│   ├── comparison.py               # Cross-run model comparison
│   └── persistence.py              # Model serialization
│
├── scripts/                        # Pipeline runners & models
│   ├── run_full_pipeline.py        # 30-step master pipeline
│   ├── run_betting_system.py       # Betting-focused runner
│   ├── ensemble_prediction_engine.py  # 6-method ensemble (~1700 lines)
│   ├── odds_fetcher.py             # Live odds via The Odds API
│   ├── odds_tracker.py             # Line movement tracking
│   ├── betting_engine.py           # Value bet identification
│   ├── advanced_betting_engine.py  # Multi-market engine
│   ├── over_under_model.py         # O/U predictions
│   ├── handicap_model.py           # Handicap predictions
│   ├── cards_model.py              # Cards predictions
│   ├── btts_corners_model.py       # BTTS & Corners
│   ├── extended_markets.py         # Exact score, HT/FT, etc.
│   ├── intelligence_integrator.py  # Post-ensemble adjustments
│   ├── sentiment_analyzer.py       # Perplexity AI sentiment
│   ├── ai_reasoning.py             # GPT-4o bet narratives
│   ├── results_fetcher.py          # Auto-settle via Odds API
│   ├── clv_tracker.py              # Closing line value
│   ├── performance_dashboard.py    # Accuracy & P/L tracking
│   └── standings_generator.py      # League table generation
│
├── web/                            # Flask web application
│   ├── app.py                      # Routes and data loading
│   ├── templates/                  # Jinja2 templates
│   │   ├── dashboard.html          # Main dashboard
│   │   └── betting.html            # Betting intelligence page
│   └── static/                     # CSS/JS assets
│
├── data/                           # All data artifacts
│   ├── raw/                        # Scraped HTML & fixtures
│   ├── parsed/                     # Parquet tables
│   ├── features/                   # features.parquet (7,829 x 484)
│   ├── upcoming/                   # Predictions & odds
│   ├── models/                     # Trained model artifacts
│   ├── bankroll/                   # Bankroll state
│   └── betting/                    # Bet history & slips
│
├── docs/                           # Documentation
├── tests/                          # Test suite
└── pyproject.toml                  # Package definition
```

## Pipeline Architecture

The system runs in two modes:

**Historical Training** (offline):
```
FBref HTML -> Parser -> Parquet Tables -> Feature Build (36 steps) -> ML Training -> Models
```

**Prediction Pipeline** (30 steps, live):
```
Odds API -> Match Sync -> Bookmaker Analysis -> Market Intelligence
    -> Ensemble Predictions (6 methods) -> Intelligence Adjustments
    -> Betting Markets (O/U, Handicap, Cards, BTTS)
    -> Value Detection -> Bankroll Sizing -> Dashboard
```

## Ensemble Weights

| Method | Weight | Description |
|--------|--------|-------------|
| Factor Analysis | 0.27 | 21-season validated factors (form, venue, Elo) |
| xG Poisson | 0.27 | Expected goals with Poisson distribution |
| ML Classifier | 0.14 | XGBoost/LightGBM/CatBoost ensemble |
| Player xG | 0.05 | Per-player expected goals aggregation |
| Deep Learning | 0.07 | Neural network classifier |
| Market-Based | 0.20 | Sharp bookmaker consensus odds |

## API Usage

| API | Credits/Month | Per Pipeline Run |
|-----|--------------|-----------------|
| The Odds API | 100K ($50 plan) | ~6 credits |
| Odds API /scores | (included) | ~2 credits |
| API-Football | 100/day (free) | ~1 call |
| Open-Meteo | Unlimited (free) | ~10 calls |
| Perplexity | Per-key pricing | ~10 calls |
| OpenAI | Per-key pricing | ~10 calls |

## Documentation

- [Architecture Guide](docs/ARCHITECTURE.md) - System design, data flow, ensemble internals
- [User Guide](docs/USER_GUIDE.md) - CLI reference, configuration, common workflows
- [Data Dictionary](docs/DATA_DICTIONARY.md) - Feature table columns, data sources
- [Development Guide](docs/DEVELOPMENT.md) - Contributing, testing, adding features
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common errors and fixes

## License

Private project. Not for redistribution.
