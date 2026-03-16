# Troubleshooting Guide

## Common Errors

### `python: command not found`

macOS requires `python3` explicitly. Use `python3` for all commands:

```bash
python3 scripts/run_full_pipeline.py   # correct
python scripts/run_full_pipeline.py    # may fail on macOS
```

### `ODDS_API_KEY not set`

```bash
# Set for current session
export ODDS_API_KEY=your_key_here

# Verify
echo $ODDS_API_KEY

# Or add to .env file
echo "ODDS_API_KEY=your_key_here" >> seriea_pipeline/.env
```

The pipeline loads `.env` automatically via `python-dotenv`. Make sure the `.env` file is in the `seriea_pipeline/` directory (not the parent).

### `No odds data received`

- Verify API key: https://the-odds-api.com/account/
- Check remaining credits (each call costs ~6 credits)
- Serie A may not have upcoming matches during off-season
- Live matches return in-play odds (Over 1.5 at 12.0 is normal for in-play, not corrupted data)

### `FBref 403 Forbidden`

FBref blocks aggressive scraping. The system has built-in rate limiting (12s delay), but if blocked:

1. Wait 30-60 minutes before retrying
2. The pipeline falls back to `current_squads.json` (covers ~7 teams) and xG profiles (all 20 teams)
3. Historical data already downloaded is not affected

### `ModuleNotFoundError: No module named 'catboost'`

CatBoost is not in `pyproject.toml`. Install it separately:

```bash
pip install catboost
```

Same applies for: `tensorflow`, `flask`, `python-dotenv`, `selenium`, `botasaurus`.

### `numpy.bool_ is not JSON serializable`

This occurs when numpy boolean values are passed to `json.dump()`. The codebase uses `_NumpySafeEncoder`:

```python
json.dump(data, f, cls=_NumpySafeEncoder)
```

If you encounter this in new code, either:
- Use the encoder class, or
- Cast explicitly: `bool(numpy_value)`, `int(numpy_int)`, `float(numpy_float)`

### `results.json iteration error`

`results.json` is a dict keyed by match name, not a list:

```python
# Wrong
for result in data.get("results", []):

# Correct
for match_name, result in data.get("results", {}).items():
```

### `KeyError: 'home_team'` in predictions.json

Predictions must include `home_team` and `away_team` as separate fields, not just a combined `match` string:

```json
{
  "match": "Inter vs Milan",
  "home_team": "Inter",
  "away_team": "Milan",
  "home_xg": 1.8,
  "away_xg": 1.2
}
```

### `predictions.json empty after pipeline run`

The pipeline overwrites `predictions.json` on each run. Predictions are archived automatically in step 1 to `predictions_archive.json`. If archiving was skipped:

```bash
# Check archive
cat data/upcoming/predictions_archive.json | python3 -m json.tool | head -20
```

### Feature build produces wrong column count

Expected: ~484 columns, ~449 ML-safe. If significantly different:

```python
import pandas as pd
df = pd.read_parquet("data/features/features.parquet")
print(f"Shape: {df.shape}")

# Check for unexpected NaN patterns
print(df.isna().mean().sort_values(ascending=False).head(20))
```

Common causes:
- Missing Understat data (no xG for some seasons)
- FBref scraping incomplete (partial player stats)
- New columns added without updating `get_ml_feature_columns()`

### `Rolling column name mismatch`

Rolling feature columns follow the format: `{side}_roll_{window}_{stat}`

```
home_roll_5_goals_scored     # correct
home_roll_goals_scored_5     # wrong (old format)
```

If you see the wrong format, re-run feature building: `seriea features`.

## ML Pipeline Issues

### `Walk-forward CV has only 1-2 folds`

Walk-forward requires at least 3 training seasons before the first test fold. With 21 seasons, you should get ~18 folds. If fewer:

- Check that `features.parquet` has data for all seasons
- Verify `_season` column is correctly populated
- Ensure `ValidationConfig.min_train_seasons` is 3 (default)

### `Calibration worsened ECE`

The calibrator validates on held-out data and falls back to identity if ECE worsens. This is logged as a WARNING. Causes:

- Too few OOF samples (< 200): calibration skipped entirely
- Model already well-calibrated: isotonic regression adds noise
- This is expected behavior, not an error

### `Optuna trial pruned`

Normal behavior. Fold-level pruning stops unpromising trials early. If too many trials are pruned:

- Increase `n_trials` (try 100-150)
- Check if search spaces are too wide
- Verify data is clean (no inf/NaN in features)

### `Kelly ROI is 0.0`

Kelly simulation uses base-rate class frequencies + 7% overround to simulate market odds. If ROI is exactly 0.0, the model's predicted probabilities don't exceed the simulated market's implied edge for any match. This indicates the model is at or below market accuracy for the test set.

### `Ensemble model failure during CV`

The ensemble handles partial model failures gracefully:
- Failed models are excluded from blending for affected folds
- Blend weights are reoptimized with remaining models
- A WARNING is logged: "Model X failed on fold Y"
- Results are still valid if at least 1 model succeeds

## Web Dashboard Issues

### `Port 5001 already in use`

```bash
# Find what's using the port
lsof -i :5001

# Kill the process
kill -9 <PID>

# Or use a different port
PORT=5002 python3 web/app.py
```

### Dashboard shows stale data

The dashboard reads JSON files from `data/upcoming/` and `data/betting/`. Re-run the pipeline to refresh:

```bash
python3 scripts/run_full_pipeline.py --quick
```

### `Template not found` error

Ensure you're running from the correct directory:

```bash
cd seriea_pipeline
python3 web/app.py
```

The Flask app expects `web/templates/` and `web/static/` relative to `web/app.py`.

## Data Issues

### `player_xg_profiles.json positions are all None`

All 634 profiles have `position: null`. This is a known limitation. The system:
- Cross-references squad data when available
- Infers position from xG patterns (high xG = forward, etc.)
- Falls back to generic weights if position unknown

### `current_squads.json has only 7 teams`

FBref scraping was blocked before completing all 20 teams. The system falls back to xG profiles (which cover all 20 teams) for player-level features.

### Parquet file corruption

If a `.parquet` file fails to load:

```bash
# Check file integrity
python3 -c "import pandas as pd; df = pd.read_parquet('data/features/features.parquet'); print(df.shape)"

# If corrupted, rebuild from scratch
seriea parse --reparse
seriea features
```

## Performance Issues

### Pipeline takes too long

- Use `--quick` to skip API calls and use cached data
- Use `--snapshot-only` for just odds data (~6 credits, ~30s)
- ML training (especially `ml optimize`) can take 30-60 minutes with 80 Optuna trials; reduce with `--trials 20`

### High memory usage during feature build

The full feature table (7,829 x 484) uses ~30MB in memory. If running on constrained hardware:

```bash
# Build features for a single season
seriea features --season 2025-2026
```

### FBref rate limiting

The scraper enforces a 12-second delay between requests (`config/settings.py: REQUEST_DELAY_SECONDS`). This is intentionally conservative for historical backfill. For current-season scraping, you can temporarily reduce to 6 seconds, but risk being blocked.
