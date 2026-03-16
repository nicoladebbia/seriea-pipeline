# Phase 1: Ensemble Validation & Optimization - Complete Summary

**Date**: 2026-02-05
**Status**: COMPLETE

---

## 1. Weight Optimization Analysis (Step 1.1)

### Current Weights
| Method | Weight |
|--------|--------|
| Factor | 30% |
| xG | 30% |
| ML | 15% |
| Player xG | 10% |
| Deep | 15% |

### Tested Configurations (10 variants)
Best performing: **deep_focus** configuration
- Accuracy: 54.7% on recent test set
- Weights: factor=0.2, xg=0.25, ml=0.15, player_xg=0.1, deep=0.3

### Key Finding
Deep learning models deserve higher weight when properly calibrated.

---

## 2. Calibration Analysis (Step 1.2)

### Deep Model Issues (FIXED)
- **Problem**: NaN predictions due to missing data handling
- **Solution**: Implemented `_safe_get()` helper method
- **Problem**: Overconfident predictions (61.4% avg confidence, 52.5% accuracy)
- **Solution**: Temperature scaling T=2.4 (reduces ECE by 43%)

### Calibration Metrics
| Model | ECE (Before) | ECE (After T=2.4) |
|-------|--------------|-------------------|
| Deep | 8.9% | 5.1% |

### Temperature Scaling Applied
```python
def _apply_temperature_scaling(probs, temperature=2.4):
    logits = np.log(probs + 1e-10)
    scaled_logits = logits / temperature
    scaled_probs = np.exp(scaled_logits)
    return scaled_probs / scaled_probs.sum()
```

---

## 3. Feature Importance Analysis (Step 1.3)

### Top 10 Most Important Features
| Rank | Feature | CatBoost Importance |
|------|---------|---------------------|
| 1 | elo_diff | 10.03 |
| 2 | matchup_competitiveness | 3.50 |
| 3 | home_stadium_capacity | 2.58 |
| 4 | home_elo | 2.45 |
| 5 | away_roll_10_shots_on_target | 2.38 |
| 6 | away_elo | 2.33 |
| 7 | attack_strength_diff | 2.29 |
| 8 | us_xg_diff | 2.18 |
| 9 | us_xa_diff | 2.05 |
| 10 | h2h_away_goals_avg | 1.98 |

### Feature Groups (by total importance)
1. **ELO/Strength**: Highest predictive power
2. **Form/Rolling**: Second most important
3. **Understat xG**: Strong signal
4. **H2H**: Moderate value
5. **Rest/Congestion**: Lower importance

### Low Importance Features (28 identified)
Features in bottom 20% to consider removing for model simplification.

---

## 4. Walk-Forward Backtesting (Step 1.4)

### Test Configuration
- **Training data**: Seasons 2005-2019
- **Test seasons**: 2020-2021 through 2024-2025
- **Total test matches**: 1,900

### Season-by-Season Results
| Season | Overall Accuracy | High-Conf Accuracy |
|--------|------------------|-------------------|
| 2020-2021 | 51.1% (194/380) | 74.5% (70/94) |
| 2021-2022 | 51.3% (195/380) | 67.0% (63/94) |
| 2022-2023 | 49.5% (188/380) | 69.2% (72/104) |
| 2023-2024 | 51.8% (197/380) | 66.3% (63/95) |
| 2024-2025 | 50.8% (193/380) | 68.4% (67/98) |

### Aggregate Results
- **Overall Accuracy**: 50.9% (967/1900)
- **High-Confidence Accuracy**: 69.1% (335/485)

### Accuracy by Confidence Level
| Confidence | Accuracy | Count |
|------------|----------|-------|
| LOW (33-45%) | 40.4% | 941 |
| MEDIUM (45-55%) | 58.6% | 812 |
| HIGH (55-65%) | 75.5% | 147 |
| VERY_HIGH (65%+) | N/A | 0 |

### Accuracy by Predicted Outcome
| Outcome | Accuracy | Count |
|---------|----------|-------|
| HOME | 49.0% | 1362 |
| DRAW | N/A | 0 |
| AWAY | 55.8% | 538 |

### Betting Simulation (High-Conf Only)
- **Bets Placed**: 147
- **ROI**: +35.9%
- **Profit**: +52.80 units

---

## Critical Issues Identified

### Issue 1: No DRAW Predictions
The ensemble NEVER predicts DRAW outcomes with high confidence. This is a significant bias that loses potential value on ~25% of matches.

**Root Cause**: Factor model and xG model inherently favor HOME/AWAY outcomes.

**Recommendation for Phase 2**: Implement DRAW-specific calibration or separate DRAW detection model.

### Issue 2: Low-Confidence Predictions Below Random
Predictions with <45% confidence have 40.4% accuracy, which is worse than random chance (33%).

**Recommendation**: Filter out or reduce weight of predictions below 45% confidence.

### Issue 3: HOME Bias with Lower Accuracy
72% of predictions are HOME (1362/1900), but accuracy is only 49%.

**Recommendation**: Calibrate home advantage factor or adjust threshold for HOME predictions.

---

## Phase 1 Deliverables

1. `/scripts/optimize_ensemble.py` - Weight optimization framework
2. `/scripts/calibration_analysis.py` - Calibration analysis tools
3. `/scripts/feature_importance_analysis.py` - Feature importance analysis
4. `/scripts/backtest_ensemble.py` - Walk-forward backtesting
5. `/models/deep_learning.py` - Fixed NaN handling + temperature scaling
6. `/data/models/reduced_features.json` - Reduced feature set (111 features)

---

## Recommendations for Phase 2

1. **Implement DRAW detection**: Add specialized draw predictor or calibration
2. **Confidence filtering**: Reject predictions below 45% confidence
3. **Weight rebalancing**: Apply deep_focus weights (factor=0.2, xg=0.25, ml=0.15, player_xg=0.1, deep=0.3)
4. **Feature reduction**: Train new models with reduced 111-feature set
5. **HOME calibration**: Reduce home advantage inflation in factor model
