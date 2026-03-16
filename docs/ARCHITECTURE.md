# Architecture Guide

## System Overview

The Serie A Prediction Pipeline is a multi-stage system that collects football data, engineers features, trains ML models, and generates betting recommendations. It operates in two distinct modes:

1. **Historical Training** (offline) - Scrape, parse, engineer features, train models
2. **Prediction Pipeline** (live) - Fetch odds, generate predictions, identify value bets

## High-Level Data Flow

```
                          HISTORICAL TRAINING
                          ===================

FBref HTML ──> Parser ──> Parquet Tables ──> Feature Build (36 steps)
                              │                      │
                              │                      ▼
                              │               features.parquet
                              │               (7,829 x 863)
                              │                      │
                              ▼                      ▼
                    matches.parquet           ML Training Pipeline
                    player_stats.parquet      (walk-forward CV + Optuna)
                    shots.parquet                    │
                    goalkeeper_stats.parquet          ▼
                    lineups.parquet           Trained Models + Ensemble
                    events.parquet           (XGB + LGB + CatBoost)


                          PREDICTION PIPELINE (30 steps)
                          ==============================

 Odds API ──────────────┐
 API-Football (lineups) ─┤
 Perplexity (sentiment) ─┤
 OpenAI (reasoning) ─────┤
                         ▼
              ┌─────────────────────┐
              │  run_full_pipeline  │
              │    (orchestrator)   │
              └─────────┬───────────┘
                        │
    ┌───────────────────┼───────────────────┐
    ▼                   ▼                   ▼
 Market Data        Ensemble           Betting Models
 ───────────     ──────────────       ──────────────
 odds_fetcher    6-method blend:      over_under_model
 odds_tracker    - factor (0.035)     handicap_model
 bookmaker_      - xG (0.124)         cards_model
   analysis      - ML (0.605)         btts_corners_model
 market_intel    - player_xg (0.032)  extended_markets
                 - deep (0.00)
                 - market (0.205)
    │                   │                   │
    └───────────────────┼───────────────────┘
                        ▼
              intelligence_integrator
              (sentiment ±8pp, player ±7pp, market ±5pp)
                        │
                        ▼
              betting_engine + bankroll_manager
              (Kelly criterion, drawdown limits)
                        │
                        ▼
              predictions.json + betting_slip.json
                        │
                        ▼
              Web Dashboard (Flask)
```

## Module Architecture

### Data Collection (`scraper/`)

| Module | Source | Data |
|--------|--------|------|
| `fixtures.py` | FBref | Season fixture lists |
| `match_reports.py` | FBref | Match report HTML pages |
| `odds.py` | football-data.co.uk | Historical betting odds |
| `weather.py` | Open-Meteo API | Temperature, precipitation, wind |
| `referee.py` | worldfootball.net | Referee assignments |
| `transfermarkt.py` | Transfermarkt | Market values, transfers |
| `lineup_fetcher.py` | API-Football | Confirmed pre-match lineups |
| `historical.py` | football-data.co.uk | Historical match results |

All scrapers write raw data to `data/raw/` or directly to JSON in `data/upcoming/`. Rate limiting is configured globally in `config/settings.py` (12s delay for backfill, 6s normal).

### Parsing (`parser/`)

`match_page.py` converts raw FBref HTML into structured Parquet tables:
- `matches.parquet` - Match metadata (date, teams, score, venue)
- `player_stats.parquet` - Per-player performance stats
- `goalkeeper_stats.parquet` - GK-specific metrics
- `shots.parquet` - Shot-level data (xG per shot)
- `lineups.parquet` - Starting XI and substitutions
- `events.parquet` - Goals, cards, substitutions with timestamps

### Feature Engineering (`features/build.py`)

The 36-step pipeline is orchestrated by `build_features()`:

**Steps 1-8: Team-Level Features**
- Match log construction, rolling averages (3/5/10 game windows), home/away splits, xG trend analysis, team strength ratings, rest days, momentum scoring, derived ratios

**Steps 9-20: Match-Level Features**
- H2H records, Elo ratings, player impact scores, referee tendencies, team aggregates, GK quality (PSxG), shot quality metrics, advanced player ratios, Understat xG, FBref advanced stats

**Steps 21-26: Contextual Features**
- League position, manager tenure/changes (with H2H), fixture congestion, suspension risk, formation analysis, match-level derived features

**Steps 27-31: External Data**
- League draw tendency, venue capacity effects, weather impact, historical betting odds, market intelligence signals

**Steps 32-36: Advanced Features**
- Injury impact (positional weighting), PPDA/pressing intensity, manager head-to-head, transfer window impact, interaction features (16 cross-signal terms)

**Column Cleanup**: Drops constant/all-NaN columns, validates dtype consistency. Output: `features.parquet` (7,829 rows x 863 columns, 449 ML-safe).

### ML Pipeline (`ml/`)

#### Data Loading (`data.py`)

`DataLoader` partitions features into tiers based on NaN rates:
- **Universal features**: Available across all 21 seasons (NaN < 20% after smart imputation)
- **Rich features**: All features for a single season (FBref-era)

Smart imputation applies domain-aware strategies:
- H2H: Neutral priors (1/3 win rates, 0 counts)
- Rolling stats: 0.0 (season start = no history)
- Elo: Forward-fill, then 1500 (league mean)
- Odds: NOT imputed (genuinely missing for old seasons)
- Player/GK/Shot: Per-season median

Availability flags (`_has_player_agg`, `_has_odds`, etc.) let models learn to handle missing feature groups.

#### Model Training (`training.py`)

Three training modes:
1. **`train_universal()`** - All seasons, universal features, walk-forward CV
2. **`train_rich()`** - Single season, all features, chronological 80/20 split
3. **`train_optimized()`** - Full pipeline: feature selection → Optuna tuning → calibration → ensemble

Walk-forward CV (`walk_forward_validate()`):
- Expanding window: train on seasons 0..N-1, test on season N
- Minimum 3 training seasons before first test fold
- Sample weights: inverse class frequency with 1.5x draw multiplier

#### Hyperparameter Tuning (`tuning.py`)

- **Optuna** with MedianPruner (80 trials default)
- Search spaces defined in `ml/config.py` per model type
- LR-estimator coupling: `n_estimators = 600 * (0.03 / lr)` clamped to [300, 1200]
- Fold-level pruning: reports per-fold log_loss, prunes if worse than median trial

#### Ensemble (`ensemble.py`)

`WeightedAverageEnsemble` blends XGBoost + LightGBM + CatBoost:
1. Each model generates out-of-fold (OOF) predictions via walk-forward CV
2. Blend weights optimized via `scipy.optimize.minimize` on log_loss
3. **Draw-aware penalty**: If draw recall < 15%, adds penalty term `max(0, 0.15 - recall) * 2.0`
4. Post-blend isotonic calibration on OOF predictions
5. Graceful failure handling: if a model fails, it's excluded from blending

#### Calibration (`calibration.py`)

Per-class isotonic regression:
1. Fit on 70% of OOF predictions (chronological split)
2. Validate on held-out 30%: must improve ECE by at least 5%
3. If validation fails, falls back to identity (no calibration)
4. Minimum 200 OOF samples required; skips otherwise
5. Renormalizes calibrated probabilities to sum to 1.0

#### Evaluation (`evaluation.py`)

Comprehensive metrics:
- **Accuracy**: Top-1 prediction accuracy
- **Log Loss**: Probabilistic calibration quality
- **Brier Score**: Mean squared probability error
- **RPS**: Ranked Probability Score (ordinal metric for H/D/A)
- **ECE**: Expected Calibration Error (10-bin)
- **Kelly ROI**: Simulated betting profit vs base-rate market (7% overround)
- **Per-class F1**: Precision/recall/F1 for Home, Draw, Away
- **Bootstrap CI**: 500-iteration confidence intervals for all core metrics

### Scripts Organization (`scripts/`)

The scripts directory is organized into 7 thematic sub-directories:

| Sub-directory | Count | Purpose |
|---------------|-------|---------|
| `prediction/` | 13 | Ensemble engine, unified predictor, form/weather/referee calculators |
| `betting/` | 10 | Bet journal, CLV tracking, Italian market standards, parlays |
| `data/` | 15 | Odds/results fetching, FBref/Sofascore/Understat scrapers |
| `models/` | 20 | Model training, tuning, optimization, specialized models (O/U, handicap) |
| `analysis/` | 15 | Backtesting, calibration diagnostics, performance tracking |
| `pipeline/` | 7 | `run_full_pipeline` orchestrator, scheduler, monitoring |
| `utils/` | 5 | Error handling, logging config, alerts, match timing |
| `legacy/` | 46 | Deprecated/superseded scripts (kept for reference) |

### Ensemble Prediction Engine (`scripts/prediction/ensemble_prediction_engine.py`)

The production prediction engine (~1,700 lines) combines 6 methods:

1. **Factor-Based** (0.035): 21-season validated factors (form, venue size, Elo gap, referee bias)
2. **xG Poisson** (0.124): Predicts home/away xG, converts via Poisson distribution
3. **ML Classifier** (0.605): CatBoost gradient boosting on feature table — dominant signal
4. **Player xG** (0.032): Per-player expected goals aggregation
5. **Deep Learning** (0.00): Disabled — LSTM/Transformer loaded but zero-weighted
6. **Market-Based** (0.205): Sharp bookmaker consensus (Pinnacle, Betfair Exchange, Matchbook)

Post-ensemble adjustments via `scripts/prediction/intelligence_integrator.py`:
- Sentiment: ±8 percentage points (Perplexity AI)
- Player analysis: ±7 percentage points
- Market intelligence: ±5 percentage points
- Combined cap: ±15 percentage points

### Betting System

#### Value Detection (`scripts/betting/betting_unified.py`)

Compares model probabilities against bookmaker-implied probabilities. A bet has value when:
```
model_probability * decimal_odds > 1.0
```

Sharp/soft bookmaker divergence > 3% flags additional signals.

#### Bankroll Management (`features/bankroll_manager.py`)

- Kelly fraction: 0.10 (tenth Kelly for safety)
- Maximum stake: 2.5% of bankroll per bet
- Drawdown limit: 25% hard block, 15% warning (reduces stakes by 50%)
- Streak adjustments: Reduces after consecutive losses
- State persisted to `data/bankroll/state.json`

#### Market Intelligence (`features/market_intelligence.py`)

Aggregates signals from multiple sources:
- Sharp vs soft bookmaker consensus (classified in `bookmaker_analysis.py`)
- Cross-market correlations (O/U vs 1X2 consistency)
- Temporal odds movement (steam moves, reverse line movement)
- Line value vs closing price (CLV tracking post-settlement)

### Web Application (`web/`)

Flask app serving two pages:
- **Dashboard** (`/`): Pipeline status, recent predictions, performance metrics
- **Betting** (`/betting`): Match cards with context strips, dual probability bars, extended markets, player props, value highlighting (green = high value, amber = medium)

## Data Storage

All data lives under `data/`:

```
data/
├── raw/
│   ├── html/          # FBref match report HTML files
│   └── fixtures/      # Season fixture CSV files
├── parsed/            # Structured Parquet tables
│   ├── matches.parquet
│   ├── player_stats.parquet
│   ├── goalkeeper_stats.parquet
│   ├── shots.parquet
│   ├── lineups.parquet
│   └── events.parquet
├── features/
│   └── features.parquet    # ML-ready feature table
├── models/
│   └── universal/          # Trained model artifacts
│       ├── xgboost.json
│       ├── lightgbm.txt
│       ├── catboost.cbm
│       ├── ensemble/
│       ├── training_report.json
│       └── feature_importance_history.json
├── upcoming/
│   ├── predictions.json    # Current match predictions
│   ├── predictions_archive.json
│   ├── odds.json           # Live bookmaker odds
│   ├── bookmaker_odds.json
│   └── referees.json
├── bankroll/
│   └── state.json          # Bankroll tracking state
└── betting/
    ├── betting_slip.json   # Current recommendations
    └── history.json        # Settled bet history
```

## External API Dependencies

| API | Purpose | Auth | Rate Limit |
|-----|---------|------|------------|
| The Odds API | Live odds, scores | `ODDS_API_KEY` | 100K credits/month |
| API-Football | Confirmed lineups | `APIFOOTBALL_KEY` | 100 req/day (free) |
| Football-Data.org | Squad rosters | `FOOTBALLDATA_KEY` | 10 req/min (free) |
| Open-Meteo | Weather data | None | Unlimited |
| Perplexity | Sentiment analysis | `PERPLEXITY_API_KEY` | Per-key |
| OpenAI | AI reasoning | `OPENAI_API_KEY` | Per-key |
| FBref | Match reports | None (scraping) | 12s delay enforced |
| Worldfootball.net | Referee data | None (scraping) | Rate limited |
| Transfermarkt | Market values | None (scraping) | Rate limited |
