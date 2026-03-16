# Development Guide

## Environment Setup

### Python Environment

```bash
# Recommended: use a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install core package
pip install -e .

# Install all dependencies (including optional ones)
pip install catboost tensorflow flask python-dotenv selenium botasaurus
pip install pytest pytest-cov
```

### Dependencies

Core (in `pyproject.toml`):
- `requests`, `beautifulsoup4`, `lxml` - Web scraping
- `pandas`, `pyarrow` - Data processing
- `numpy`, `scikit-learn` - ML foundation
- `xgboost`, `lightgbm`, `optuna` - Model training
- `click` - CLI framework

Additional (install separately):
- `catboost` - CatBoost model support
- `tensorflow` - Deep learning models
- `flask`, `python-dotenv` - Web dashboard
- `selenium`, `botasaurus` - Dynamic scraping
- `scipy` - Statistical functions (Poisson, optimization)

### Project Layout

```
seriea_pipeline/
├── cli.py              # CLI entry point (installed as 'seriea')
├── config/             # Configuration
├── scraper/            # Data collection modules
├── parser/             # HTML parsing
├── storage/            # Data persistence (paths, Parquet I/O)
├── features/           # Feature engineering (36 modules)
├── ml/                 # ML pipeline (training, ensemble, calibration)
├── scripts/            # Pipeline runners and prediction engines
├── web/                # Flask web application
├── models/             # Deep learning model definitions
├── tools/              # Utility scripts
├── tests/              # Test suite
└── data/               # All data artifacts (gitignored)
```

## Adding a New Feature

### 1. Create the feature module

Add a new file in `features/`, e.g. `features/my_feature.py`:

```python
"""My new feature description."""

import logging
import pandas as pd

log = logging.getLogger(__name__)

def add_my_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add my features to the match-level DataFrame.

    Args:
        df: DataFrame with columns from previous steps (at minimum:
            home_team, away_team, season, match_date, and prior features).

    Returns:
        DataFrame with new columns added.
    """
    # Compute features using only data available BEFORE the match
    # NEVER use home_score, away_score, or result (leakage!)
    df["my_new_feature"] = ...
    log.info("Added my_feature: %d non-null", df["my_new_feature"].notna().sum())
    return df
```

### 2. Register in the build pipeline

Edit `features/build.py`:

```python
# Add import at top
from features.my_feature import add_my_features

# Add step in build_features() function (after appropriate step)
log.info("Step XX: My features")
df = add_my_features(df)
```

### 3. Update feature exclusion if needed

If your feature has columns that could cause data leakage, add them to the exclusion list in `get_ml_feature_columns()` in `features/build.py`.

### 4. Rebuild features and retrain

```bash
seriea features
seriea ml optimize --trials 50
```

## Adding a New Betting Model

### 1. Create the model script

Add a file in `scripts/`, e.g. `scripts/my_market_model.py`. Follow the pattern of existing models (`over_under_model.py`, `cards_model.py`):

```python
def run_my_market(predictions, odds_data, output_path):
    """Generate my market predictions.

    Args:
        predictions: List of dicts from predictions.json
        odds_data: Dict from odds.json
        output_path: Path to write results JSON
    """
    results = []
    for pred in predictions:
        # Blend 70% ensemble xG / 30% own Poisson estimate
        ...
        results.append({...})

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=_NumpySafeEncoder)
```

### 2. Register in the pipeline runner

Add a new step in `scripts/run_full_pipeline.py`:

```python
log.info("Step XX: My market model")
from scripts.my_market_model import run_my_market
run_my_market(predictions, odds_data, output_path)
```

## Adding a New ML Model Type

### 1. Add model wrapper

Edit `ml/models.py` to add a new model class implementing `fit()`, `predict_proba()`, and `get_feature_importance()`.

### 2. Register in config

Add to `MODEL_TYPES` in `ml/config.py` and add default hyperparameters to `ModelConfig`.

### 3. Add search space

Add tuning search space to `TuningConfig` in `ml/config.py` and a corresponding `_suggest_*_params()` function in `ml/tuning.py`.

## Testing

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run with coverage
python3 -m pytest tests/ --cov=ml --cov=features --cov-report=term-missing

# Run specific test file
python3 -m pytest tests/test_evaluation.py -v
```

### Test conventions

- Test files go in `tests/` with `test_` prefix
- Use fixtures for data loading to avoid redundant I/O
- ML tests should use small synthetic datasets for speed
- Integration tests can use the real features.parquet if available

## Code Conventions

### Imports

- Standard library first, then third-party, then local
- Lazy imports for heavy dependencies (tensorflow, catboost) to avoid import-time failures
- Use `try/except ImportError` with `HAS_*` flags for optional dependencies

### Logging

- Use `logging.getLogger(__name__)` in every module
- INFO for step completions and summaries
- WARNING for fallbacks and degraded functionality
- DEBUG for detailed diagnostics

### Data handling

- Use `pandas` DataFrames throughout the pipeline
- Persist intermediate data as Parquet (not CSV)
- Use `_NumpySafeEncoder` for JSON serialization of numpy types
- Always cast `numpy.bool_` to `bool()` before JSON serialization

### Feature engineering rules

- **No data leakage**: Never use match-day stats (score, cards, etc.) as features
- **Rolling features**: Use only past matches, not the current match
- **Column naming**: `{side}_roll_{window}_{stat}` format (e.g. `home_roll_5_goals_scored`)
- **NaN handling**: Use domain-aware imputation strategies in `ml/data.py`

### ML conventions

- Walk-forward CV for all evaluations (no random splits)
- Sample weights with 1.5x draw multiplier
- Always report: accuracy, log_loss, brier_score, RPS, ECE, Kelly ROI
- Calibrate after blending, not before

## Debugging Tips

### Pipeline failures

Check logs in `logs/` directory. The pipeline logger writes timestamped entries for each step.

### Feature build issues

```python
# Quick diagnostic
import pandas as pd
df = pd.read_parquet("data/features/features.parquet")
print(f"Shape: {df.shape}")
print(f"NaN %: {df.isna().mean().sort_values(ascending=False).head(20)}")
print(f"Seasons: {df['season'].unique()}")
```

### Model performance

```python
# Load training report
import json
with open("data/models/universal/training_report.json") as f:
    report = json.load(f)
print(f"Features: {report['n_features_selected']}")
print(f"Top features: {report['top_20_features'][:10]}")
```

### Comparing model versions

```python
from ml.comparison import compare_runs, print_comparison_report
result = compare_runs("universal", "v1", "v2")
print(print_comparison_report(result))
```
