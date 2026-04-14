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
- Serie A accuracy: 53.4%, EPL accuracy: 55.3%
- Per-league model separation (not one model for all leagues)
- Time-decay weighting, 2017+ training window
- Betting leaks patched (odds NOT used as input features)
- Odds backfill via historical API
- Sofascore scraper for EPL data

## Conventions
- Strict typing (mypy enforced)
- DuckDB for data processing, Parquet for storage
- Config-driven pipeline steps
- All data transformations in features/ directory
- ruff + black for formatting
