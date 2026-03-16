# Phase 2 Complete: Player-Level xG Modeling

## Summary

Successfully integrated player-level xG predictions as the 4th ensemble method.

## Results

| Metric | Phase 1 | Phase 2 | Change |
|--------|---------|---------|--------|
| Ensemble Methods | 3 | 4 | +1 |
| Model Version | v3.1 | v3.2-player-xg | - |
| Player Profiles | 0 | 634 | +634 |
| Teams Covered | 0 | 20 | +20 |

## New Ensemble Weights

```python
ENSEMBLE_WEIGHTS = {
    "factor": 0.35,      # Factor-based (validated 21 seasons)
    "xg": 0.35,          # xG regression + Poisson
    "ml": 0.20,          # CatBoost classifier
    "player_xg": 0.10,   # Player lineup-based xG
}
```

## Key Components Added

### 1. PlayerXGProfile Class
- Tracks individual player xG/90, xA/90
- Career totals and recent form (last 5 matches)
- Top contributors identification

### 2. PlayerXGDatabase
- 634 player profiles from match-level data
- Per-team player aggregation
- JSON persistence for fast loading

### 3. LineupXGPredictor
- Predicts team xG based on expected lineup
- Handles missing players (injury impact)
- Scales individual xG to team total

### 4. PlayerXGPredictor (Ensemble Integration)
- Converts lineup xG to match probabilities
- Integrated as 4th ensemble method

## Top Players by xG/90 (min 500 mins)

| Rank | Player | Team | xG/90 | xA/90 |
|------|--------|------|-------|-------|
| 1 | Mateo Retegui | Atalanta | 0.714 | 0.178 |
| 2 | Moise Kean | Fiorentina | 0.649 | 0.063 |
| 3 | Santiago Giménez | Milan | 0.646 | 0.121 |
| 4 | Dušan Vlahović | Juventus | 0.626 | 0.061 |
| 5 | Lautaro Martínez | Inter | 0.477 | 0.151 |

## Sample Predictions

```
Roma vs Cagliari: HOME (VERY HIGH)
  Probs: H=61.5% D=20.2% A=18.3%
  Methods: ['factor', 'xg', 'ml', 'player_xg']
  Player xG: Home=1.63 Away=1.58

Sassuolo vs Inter: AWAY (VERY HIGH)
  Probs: H=21.7% D=21.0% A=57.3%
  Player xG: Home=1.40 Away=1.82
```

## Files Created/Modified

### New Files
- `features/player_xg_model.py` - Player xG modeling system
- `data/features/player_xg_profiles.json` - 634 player profiles

### Modified Files
- `scripts/ensemble_prediction_engine.py`:
  - Added `PlayerXGPredictor` class
  - Updated `EnsemblePredictor` to use 4 methods
  - Updated weight distribution
  - Model version bumped to v3.2-player-xg

## Validation

All 29 ensemble tests pass:
- Player xG predictor loads correctly
- Lineup-based predictions working
- 4-method ensemble combining correctly
- Probabilities sum to 1.0

## Use Cases

### 1. Injury Impact Analysis
```python
from features.player_xg_model import PlayerXGDatabase, LineupXGPredictor

db = PlayerXGDatabase()
db.load()
predictor = LineupXGPredictor(db)

# Impact of losing top scorer
impact = predictor.estimate_absence_impact("Inter", ["Lautaro Martínez"])
# xG reduction: 0.24, Impact level: critical
```

### 2. Lineup-Based Predictions
```python
pred = predictor.predict_team_xg("Inter")
# predicted_xg: 1.63, confidence: high
# top_contributors: Lautaro Martínez (0.52 xG/90), Marcus Thuram (0.31)
```

## Next Steps (Phase 3)

1. **Formation Analysis**
   - Detect team formations from lineup data
   - Build formation matchup matrix
   - Track tactical shifts

2. **Advanced Shot Quality**
   - Integrate shot location data
   - Model shot conversion by zone
   - Big chance tracking

---
*Phase 2 completed: 2026-02-05*
