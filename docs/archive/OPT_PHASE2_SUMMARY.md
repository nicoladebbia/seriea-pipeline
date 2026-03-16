# Phase 2: Prediction Calibration - Complete Summary

**Date**: 2026-02-05
**Status**: COMPLETE

---

## Phase 2 Improvements Implemented

### 2.1 DRAW Detection Module
**File**: `features/draw_detection.py`

Specialized detector for identifying likely DRAW outcomes using:
- Elo difference analysis (tight matches = more draws)
- Derby detection (Serie A major derbies boosted)
- Head-to-head draw rate history
- Team draw rate caching
- Defense strength analysis
- Clean sheet rate analysis
- xG balance detection

**Result**: Now predicting 521 DRAW outcomes (27% of predictions) vs 0 before.

### 2.2 Confidence Filtering
**File**: `features/prediction_calibration.py`

Implemented confidence-based filtering with configurable thresholds:
- `min_confidence`: Reject predictions below this threshold
- `high_conf_threshold`: Mark high-confidence predictions
- Classification: SKIP / LOW / MEDIUM / HIGH
- Suggested bet sizing: 0 / 0.5 / 1.0 / 2.0 units

### 2.3 Betting Strategy Profiles
**File**: `features/prediction_calibration.py`

Five pre-configured betting strategies:

| Strategy | Description | Min Conf | High Conf |
|----------|-------------|----------|-----------|
| default | Balanced approach | 45% | 55% |
| volume | More bets, 54.7% accuracy | 42% | 50% |
| selective | Fewer bets, 73.1% HC accuracy | 50% | 58% |
| ml_heavy | Emphasizes ML classifier | 45% | 55% |
| xg_dominant | Bookmaker style | 45% | 53% |

### 2.4 HOME Advantage Calibration
**File**: `features/prediction_calibration.py`

Reduced home advantage inflation:
- Dynamic home threshold based on Elo difference
- Probability scaling: home*0.92, away*1.08, draw*1.05
- Only applies to close matches (elo_diff < 200)

---

## Before vs After Comparison (5 Seasons, 1,900 Matches)

| Metric | Phase 1 | Phase 2 | Improvement |
|--------|---------|---------|-------------|
| Overall Accuracy | 50.9% | 51.6% | +0.7% |
| High-Conf Accuracy | 69.1% | **77.1%** | **+8.0%** |
| DRAW Predictions | 0 | 521 | Now works! |
| HOME Predictions | 1362 | 964 | Reduced bias |
| HOME Accuracy | 49.0% | 54.4% | +5.4% |
| AWAY Accuracy | 55.8% | 60.5% | +4.7% |
| DRAW Accuracy | N/A | 39.3% | >33% random |
| ROI | +35.9% | **+44.8%** | **+8.9%** |

---

## Season-by-Season Results (Phase 2)

| Season | Overall | High-Conf | HC Count |
|--------|---------|-----------|----------|
| 2020-2021 | 52.6% | 80.0% | 40 |
| 2021-2022 | 51.3% | 75.0% | 44 |
| 2022-2023 | 50.0% | 73.6% | 53 |
| 2023-2024 | 51.3% | 76.1% | 46 |
| 2024-2025 | 52.6% | **82.5%** | 40 |

---

## Accuracy by Confidence Level

| Confidence | Accuracy | Count | Recommendation |
|------------|----------|-------|----------------|
| LOW (33-45%) | 44.2% | 1336 | SKIP |
| MEDIUM (45-55%) | 67.1% | 477 | Small bet |
| HIGH (55-65%) | **80.5%** | 87 | Strong bet |

---

## Accuracy by Predicted Outcome

| Outcome | Accuracy | Count | % of Predictions |
|---------|----------|-------|------------------|
| HOME | 54.4% | 964 | 50.7% |
| DRAW | 39.3% | 521 | 27.4% |
| AWAY | 60.5% | 415 | 21.8% |

---

## Files Created/Modified

**New Files:**
- `features/draw_detection.py` - DRAW detection module
- `features/prediction_calibration.py` - Calibration pipeline + strategies

**Modified Files:**
- `scripts/ensemble_prediction_engine.py` - Integrated calibration pipeline
- `scripts/backtest_ensemble.py` - Added strategy support

---

## Key Insights

1. **DRAW detection works**: 39.3% accuracy is significantly better than random (33%)
2. **High-confidence improved dramatically**: 77.1% vs 69.1%
3. **HOME bias reduced**: Predictions now 50.7% HOME vs 72% before, with better accuracy
4. **ROI improved**: +44.8% vs +35.9% on high-confidence bets
5. **Derby detection**: Correctly identifies Serie A derbies as draw candidates

---

## Recommendations for Phase 3

1. **Tune draw detection thresholds**: Currently conservative, could be more aggressive
2. **Add weather impact on draws**: Rainy conditions often lead to more draws
3. **Season-specific calibration**: Different seasons have different draw rates
4. **Implement Kelly criterion**: Optimal bet sizing based on edge
5. **Add live odds integration**: Real-time value betting detection
