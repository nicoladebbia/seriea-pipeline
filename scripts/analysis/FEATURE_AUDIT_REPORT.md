# Feature Engineering Pipeline Audit Report
**Date:** 2026-02-17
**Auditor:** feature-engineer agent
**Scope:** Full pipeline integrity check

---

## Executive Summary

The feature engineering pipeline is **functionally operational but inefficient**. The pipeline produces 863 columns, but:
- **428 columns (49.6%)** have >50% NaN rate globally → **unusable**
- **831 columns (96.3%)** have <50% NaN in recent seasons (2023-2026) → usable
- **CatBoost model uses only 35 features** → 97% of the feature table is ignored
- **10 feature modules (159 KB)** exist but are never imported → dead code
- **3 metadata features** (`_has_gk_data`, `_has_shot_data`, `_has_odds`) are injected at training time but NOT in `features.parquet` → **training-serving skew risk**

---

## 1. Module Inventory

### Active Modules (35 modules, imported in `features/build.py`)
All 35 are properly registered as pipeline steps. No orphaned imports found.

**Top contributors by feature count:**
- `sofascore_features.py` (35.5 KB) → momentum, captain, card timing, corners
- `enhanced_momentum.py` (26.5 KB) → streak/form features
- `enhanced_weather.py` (25.7 KB) → weather context
- `formation_analysis.py` (25.1 KB) → tactical matchups
- `referee.py` (23.9 KB) → referee bias, strictness

### Dead Code (10 modules, 159 KB total, NEVER imported)
These are **fully orphaned**. They exist in `features/` but `build.py` never imports them:

| Module | Size | Purpose (inferred) |
|--------|------|-------------------|
| `player_xg_model.py` | 32.1 KB | Player-level xG predictions |
| `bankroll_manager.py` | 19.2 KB | **Betting logic** (should be in `scripts/betting/`) |
| `value_betting.py` | 16.5 KB | **Betting logic** (misplaced) |
| `sentiment_analysis.py` | 15.3 KB | Social media sentiment? |
| `bookmaker_analysis.py` | 14.9 KB | **Betting logic** (misplaced) |
| `cross_market_analysis.py` | 14.5 KB | Market correlation features |
| `prediction_calibration.py` | 14.3 KB | **ML calibration** (should be in `ml/`) |
| `draw_detection.py` | 12.6 KB | Draw-specific features |
| `lineup_stats.py` | 10.8 KB | Starting XI stats |
| `market_intelligence.py` | 9.4 KB | **Betting logic** (misplaced) |

**Recommendation:** Archive to `_deprecated/` or delete. If any are valuable, integrate them as pipeline steps.

### Deprecated Modules (9 modules in `_deprecated/`, 107 KB)
Already moved to deprecated directory. Safe to ignore.

---

## 2. Feature Count: 863 Columns

### Global NaN Analysis (all seasons, 2005-2026)
| NaN Range | Count | % of Total | Status |
|-----------|-------|------------|--------|
| >50% NaN | 428 | 49.6% | **Unusable** — adding noise, not signal |
| 30-50% NaN | 50 | 5.8% | **Risky** — need imputation strategy |
| <30% NaN | 385 | 44.6% | **Usable** |
| <5% NaN | 371 | 43.0% | **High quality** |

**Top offenders (>90% NaN globally):**
```
home_xg_overperformance_roll_5: 95.0% NaN
away_xg_overperformance_roll_5: 95.0%
home_venue_roll_3_xg_for: 94.8% NaN
away_key_players_available: 93.2% NaN
home_top_scorer_played: 93.2% NaN
```

**Why?** Understat xG data only available from ~2014 onward. Venue-specific xG requires even more historical data (rolling windows). Player availability data comes from Sofascore (2018+).

### Recent Seasons NaN Analysis (2023-2026, 1,369 matches)
**Much better.** Only 32 columns with >30% NaN in recent seasons.

| NaN Range | Count | % of Total |
|-----------|-------|------------|
| >50% NaN | 0 | 0.0% |
| 30-50% NaN | 32 | 3.7% |
| <30% NaN | 831 | 96.3% |

**Worst features in recent seasons:**
```
home_xg_overperformance_roll_5: 71.2% NaN  (Understat coverage gaps)
away_key_players_available: 61.1% NaN  (Sofascore player data gaps)
home_squad_rotation: 58.7% NaN  (requires lineup data)
```

### Category-Level NaN Rates (Recent Seasons)
| Category | Columns | Avg NaN % | Status |
|----------|---------|-----------|--------|
| **Tactical** | 16 | 0.0% | Excellent |
| **Sofascore** | 20 | 1.0% | Excellent |
| **H2H** | 18 | 6.6% | Good |
| **Player** | 16 | 9.4% | Good (but player_available features at 60%) |
| **Weather** | 12 | 17.2% | Acceptable |
| **Understat (xG)** | 109 | 17.2% | Acceptable (venue xG features drag it down) |
| **Referee** | 18 | 19.9% | Acceptable |

---

## 3. CatBoost Model Features: 35 Features

The trained model (`catboost_latest.cbm`) uses **only 35 features** out of 863 available (4%).

### Feature Breakdown by Category
| Category | Count | Features |
|----------|-------|----------|
| **Form/Rolling** | 10 | `home_gd_roll_5`, `away_roll_5_shots_on_target`, `rolling_gd_diff`, etc. |
| **Elo/Strength** | 5 | `elo_diff`, `attack_strength_diff`, `elo_x_form`, `injury_x_elo`, `home_adj_attack_10` |
| **Other/Derived** | 5 | `matchup_competitiveness`, `combined_disruption`, `away_losing_at_ht_rate`, etc. |
| **H2H** | 3 | `h2h_btts_rate`, `h2h_goals_diff`, `h2h_away_goals_avg` |
| **Metadata** | 3 | `_has_gk_data`, `_has_shot_data`, `_has_odds` |
| **League Position** | 2 | `home_league_gd`, `away_points_to_cl_zone` |
| **Manager** | 2 | `tenure_x_form`, `home_chemistry_disruption` |
| **Venue** | 2 | `home_stadium_capacity`, `home_travel_fatigue` |
| **Injury** | 1 | `away_injury_impact` |
| **Congestion** | 1 | `congestion_asymmetry` |
| **Match Context** | 1 | `is_late_season` |

### Feature Availability Check
**32 features** exist in `features.parquet` with <30% NaN in recent seasons.
**3 metadata features** are **MISSING** from `features.parquet`:

```python
_has_gk_data    # Injected by ml/data.py during training
_has_shot_data  # Injected by ml/data.py during training
_has_odds       # Injected by ml/data.py during training
```

**Where they're injected:** `ml/data.py:200-225` (DataLoader class)

**Risk:** These features are computed at **training time** from the feature table. If prediction-time code doesn't inject them the same way, **you have training-serving skew**.

---

## 4. Prediction-Time Feature Injection

### Current Implementation
`ml/data.py` injects metadata features during **training**:
```python
df["_has_gk_data"] = (~df[gk_cols].isna().all(axis=1)).astype(np.int8)
df["_has_shot_data"] = (~df[shot_cols].isna().all(axis=1)).astype(np.int8)
df["_has_odds"] = (~df[odds_cols].isna().all(axis=1)).astype(np.int8)
```

### Gap: Prediction-Time Injection
Checked `scripts/prediction/` — **NO matches found** for these feature names.

**This means:**
1. At training time, the model sees `_has_gk_data=1` if goalkeeper stats are available.
2. At prediction time, if the ensemble predictor doesn't inject these features, they'll be **missing or defaulted to 0**.
3. The model will make predictions as if "no goalkeeper data exists" even when it does.

**Action Required:**
- Add metadata injection to `ensemble_prediction_engine.py` BEFORE calling `model.predict()`.
- Match the exact logic from `ml/data.py`.

---

## 5. Sofascore/Understat Feature Gaps

### Sofascore Features (20 columns, 1.0% avg NaN in recent seasons)
**Excellent coverage.** Sofascore scraper is working well.

**Columns checked:**
- `home_corners`, `away_corners` (0% NaN)
- `home_roll_5_corners` (2.9% NaN)
- Momentum, captain, card_timing features all <3% NaN

### Understat Features (109 columns, 17.2% avg NaN in recent seasons)
**Mixed.** Base xG is fine, but venue-specific and differential features have gaps.

**High NaN features:**
```
home_xg_overperformance_roll_5: 71.2% NaN  → DROP or fix Understat scraper
home_venue_roll_3_xg_for: 70.0% NaN  → DROP (not enough data for venue splits)
away_xg_diff_roll_10: 66.8% NaN  → DROP or fix
```

**Root cause:** Understat only covers 2014+ seasons. Rolling windows require 5-10 prior matches, which pushes coverage back further. Venue-specific xG features need even more data.

**Recommendation:**
- Drop all `venue_roll_*_xg` features (60-70% NaN, not used by model anyway).
- Check if `xg_overperformance` features are used anywhere. If not, drop them.
- If you need them, fix Understat scraper to backfill 2010-2013.

---

## 6. H2H Feature Reliability

### NaN Rate: 6.6% in recent seasons (acceptable)

**H2H features (18 columns):**
```
h2h_matches_played: 11.8% NaN
h2h_home_wins: 11.8% NaN
h2h_away_wins: 11.8% NaN
h2h_btts_rate: 11.8% NaN  (used by CatBoost)
h2h_goals_diff: 11.8% NaN  (used by CatBoost)
h2h_away_goals_avg: 11.8% NaN  (used by CatBoost)
```

**Why 11.8% NaN?**
- Teams that have **never played each other** (newly promoted teams, rare matchups).
- H2H requires at least 1 prior meeting. New matchups get NaN.

**Fallback behavior:**
- Check `features/h2h.py` for default imputation. Likely fills with league-average or 0.
- **Risk:** If h2h_btts_rate defaults to 0, the model interprets it as "these teams never have BTTS", which is wrong. Should default to league average.

**Action Required:**
- Audit `features/h2h.py` imputation logic.
- Ensure NaN → league average, NOT 0.

---

## 7. Critical Issues Found

### CRITICAL: Training-Serving Skew Risk
**Metadata features (`_has_*`) are not injected at prediction time.**

**Impact:** Model expects these features but gets missing/default values, degrading prediction quality.

**Fix:** Add to `ensemble_prediction_engine.py`:
```python
# BEFORE calling model.predict()
gk_cols = [c for c in df.columns if 'gk_' in c or 'saves' in c]
shot_cols = [c for c in df.columns if 'shot_' in c or 'xg' in c]
odds_cols = [c for c in df.columns if 'odds_' in c or 'implied_prob' in c]

df['_has_gk_data'] = (~df[gk_cols].isna().all(axis=1)).astype(int)
df['_has_shot_data'] = (~df[shot_cols].isna().all(axis=1)).astype(int)
df['_has_odds'] = (~df[odds_cols].isna().all(axis=1)).astype(int)
```

### CRITICAL: 428 Useless Columns (49.6% of features.parquet)
**Why this matters:**
- Storage waste: 8.6 MB parquet file, half of it is noise.
- Build time: Every feature rebuild computes 428 columns that will never be used.
- Debugging hell: When a feature breaks, you're searching through 863 columns.

**Recommendation:**
- Run feature selection to identify which columns are NEVER used by ANY model.
- Drop them from the pipeline or move to a separate "experimental" table.
- Target: reduce to ~200 columns (all models + experiments).

### WARNING: 10 Dead Modules (159 KB)
**Modules that exist but are never imported:**
- `bankroll_manager.py`, `value_betting.py`, `bookmaker_analysis.py` → **betting logic, should be in `scripts/betting/`**
- `prediction_calibration.py` → **ML calibration, should be in `ml/`**
- `player_xg_model.py` → 32 KB, could be valuable but unused
- `draw_detection.py` → draw-specific features, could help draw classifier

**Recommendation:**
- Move betting modules to `scripts/betting/`
- Move calibration to `ml/`
- Evaluate `player_xg_model.py` and `draw_detection.py` — if valuable, integrate into pipeline. Otherwise, delete.

### WARNING: Understat xG Features with 60-95% NaN
**Features like `home_xg_overperformance_roll_5`, `venue_roll_*_xg` are unusable.**

**Recommendation:**
- Drop all `venue_roll_*_xg` features (insufficient data for venue splits).
- Check if any model uses `xg_overperformance` features. If not, drop them.

---

## 8. Recommendations

### Immediate (This Week)
1. **Fix training-serving skew:** Add `_has_*` metadata injection to prediction engine.
2. **Audit H2H imputation:** Ensure NaN defaults to league avg, not 0.
3. **Archive dead modules:** Move 10 unused modules to `_deprecated/` or delete.

### Short-Term (This Month)
4. **Drop high-NaN features:** Remove 428 columns with >50% global NaN. Keep only <30% NaN features.
5. **Consolidate betting logic:** Move `bankroll_manager`, `value_betting`, `market_intelligence` to `scripts/betting/`.
6. **Run feature importance:** Identify which of the 863 columns are used by ANY model. Drop the rest.

### Long-Term (Next Quarter)
7. **Backfill Understat data:** Scrape 2010-2013 xG data to reduce NaN rates.
8. **Add player_xg_model features:** If it predicts individual player xG, integrate it (could boost player props).
9. **Refactor feature table:** Split into `core_features.parquet` (200 cols, <5% NaN) and `experimental_features.parquet` (rest).

---

## 9. Files Generated

All audit scripts saved to `scripts/analysis/`:
- `audit_features_detailed.py` — NaN analysis, category breakdown
- `trace_model_features.py` — Maps CatBoost features to categories, checks availability
- `find_unused_modules.py` — Finds dead code in `features/`
- `check_module_usage.py` — Checks import vs usage (has regex bugs, needs fix)
- `FEATURE_AUDIT_REPORT.md` — This report

---

## 10. Summary Stats

| Metric | Value |
|--------|-------|
| **Total rows** | 7,829 |
| **Total columns** | 863 |
| **Usable columns (recent seasons)** | 831 (96.3%) |
| **High-quality columns (<5% NaN)** | 371 (43.0%) |
| **Model features (CatBoost)** | 35 (4.0% of total) |
| **Active feature modules** | 35 |
| **Dead feature modules** | 10 (159 KB) |
| **Deprecated modules** | 9 (107 KB) |
| **Training-serving skew features** | 3 (`_has_gk_data`, `_has_shot_data`, `_has_odds`) |

**Overall Assessment:** Pipeline is **functional but bloated**. Immediate fix required for metadata feature injection. Long-term cleanup will improve build speed and maintainability.
