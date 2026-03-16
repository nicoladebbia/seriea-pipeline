# Legacy Scripts

This directory contains scripts that have been deprecated or superseded by more comprehensive implementations.

## Training Scripts (15 files)
These training scripts were consolidated into the main trainers:
- `train_advanced_predictions.py` - Most comprehensive trainer (KEPT IN MAIN)
- `train_no_odds.py` - For upcoming matches without odds (KEPT IN MAIN)
- `train_with_odds.py` - For historical validation (KEPT IN MAIN)

### Moved to Legacy:
- `train_baseline.py` - Basic implementation
- `train_contextual.py` - Contextual factors only
- `train_hierarchical.py` - Hierarchical model
- `train_hierarchical_metrics.py` - Metrics version
- `train_high_base_rate.py` - High base rate approach
- `train_improved.py` - Incremental improvements
- `train_match_metrics.py` - Match metrics focus
- `train_mega_combinations.py` - Feature combinations
- `train_multi_market.py` - Multi-market training
- `train_optimized.py` - Optimization experiments
- `train_ultimate_stacking.py` - Stacking approach
- `train_upcoming_model.py` - Upcoming-specific
- `train_with_players.py` - Player-level features
- `train_xg_based.py` - xG-based model
- `train_advanced_storylines.py` - Storyline features

## Prediction Scripts (5 files)
These were superseded by `realtime_prediction_engine.py` which is used by the main pipeline.

### Moved to Legacy:
- `predict_combined_markets.py` - Combined market predictions
- `predict_high_confidence.py` - High confidence only
- `predict_matchday.py` - Single matchday
- `predict_matchweek.py` - Full matchweek
- `predict_upcoming_full.py` - Full upcoming

## Bankroll Manager (1 file)
- `bankroll_manager_advanced.py` - Not imported by any other script

## How to Use Legacy Scripts
These scripts can still be run manually if needed:
```bash
python scripts/legacy/train_xg_based.py
```

However, for normal operation, use the main pipeline:
```bash
python scripts/run_full_pipeline.py
```
