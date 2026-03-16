# Baselines & Rejection Thresholds

Read this before running backtests, validating models, or shipping changes.

## Current Baselines (NEVER regress beyond these)
| Metric | 4-Season Backtest | 2024-2025 OOS | Reject If |
|--------|-------------------|---------------|-----------|
| Accuracy | 54.3% | 53.2% | < 50.0% |
| Log-loss | 0.9576 | 0.9595 | > 0.99 |
| Brier | 0.1899 | 0.1910 | > 0.20 |
| RPS | 0.1255 | — | > 0.14 |
| ECE | 0.0196 | — | > 0.05 |
| Betting yield (1X2) | +10.5% | +10.7% | < +5% |

Note: Baselines updated Feb 17 2026 after fixing 3 critical bugs: (1) ENSEMBLE_WEIGHTS_WITH_DEEP synced to production weights, (2) FALLBACK_WEIGHTS recomputed proportionally, (3) raw_prob leak fixed (probs were saved AFTER draw boost 1.28x, inflating draw edges ~10-13%). Also fixed backtest parity: kelly_fraction 0.25→0.15, Poisson max_goals 8→10, added xG draw inflation to match production. 4-season backtest (2021-2025, 1520 matches): 1,038 bets, €3,940 profit, +6.8% flat ROI, +29.6% Kelly ROI. Season trend: +6.0% (2021-22) → +5.6% (2022-23) → +5.3% (2023-24) → +10.7% (2024-25). ECE improved 47% (0.0368→0.0196). Note: 2021-22 is an outlier (57.6% accuracy); conservative 3-season avg is 53.3%.

## Key Constants (updated Feb 22 2026)
| Parameter | Value | Context |
|-----------|-------|---------|
| Ensemble weights | F=0.035, xG=0.124, ML=0.605, PxG=0.032, D=0.00, M=0.205 | `ensemble_prediction_engine.py` |
| ML model | catboost_no_odds.cbm (45 features, retrained 2026-02-23) | Leakage-free selection, correlation-pruned |
| ML temperature | T=0.75 | Pre-ensemble sharpening (was 0.40; raised to preserve draw signal) |
| Post-ensemble T | T=0.90 | Sharpens underconfident predictions (was 1.04; ECE 0.0587→0.0329) |
| Draw boost | 1.12 | Draw compensation (was 1.28; reduced because T=0.75 compresses draws less) |
| Staking mode | **kelly** | Kelly criterion with fractional sizing |
| Kelly fraction | **0.10** | 10% Kelly — balances growth vs variance |
| Max stake | **2.5% of bankroll** | Per bet (was 5%) |
| Min bookmakers | **3** | O/U bets require 3+ bookmakers for reliable edge benchmark |
| 1X2 | **DISABLED** | Live: 5W/13L -17.6% ROI, 20% CLV beat. Backtest unreliable (data leakage). |
| 1X2_Draw | **DISABLED** | Same as 1X2 — model can't predict draws accurately (28% WR, need 31%) |
| DC min edge | 5% | Double chance: live +10.6% ROI (7W/3L), 78% CLV beat |
| O/U Over min edge | 5% | Lines 0.5, 1.5, 2.5, 3.5, 4.5 via alternate totals merge |
| O/U Under | DISABLED | Multi-market backtest: -1% to -14% ROI at all thresholds |
| AH | DISABLED | Multi-market backtest: -20% to -42% ROI |
| Edge cap | 12% (all markets) | Max edge — anything higher is likely bookmaker error |

## Live Performance (as of Feb 22 2026, 64 settled bets)
| Market | Bets | W/L | ROI | CLV Beat% | Status |
|--------|------|-----|-----|-----------|--------|
| O/U | 20 | 12W/8L | +3.6% | 95% | ACTIVE — primary edge source |
| DC | 10 | 7W/3L | +10.6% | 78% | ACTIVE — strong but small sample |
| AH | 6 | 4W/2L | +41.4% | — | DISABLED (small sample, backtest negative) |
| 1X2 | 18 | 5W/13L | -17.6% | 20% | DISABLED — confirmed unprofitable |
| **Total** | **64** | **28W/36L** | **+1.3%** | **70%** | EUR 17.16 profit on EUR 1,294 staked |
| **Without 1X2** | **46** | **23W/13L** | **+9.2%** | **~85%** | EUR 84 profit — real edge signal |

**Key insight**: CLV beat rate is the strongest signal for long-term profitability. O/U at 95% and DC at 78% indicate genuine edge. 1X2 at 20% confirms no edge exists there.

## Model Registry
```
No-odds CatBoost (catboost_no_odds.cbm) → 35 features, leakage-free selection, ACTIVE ML component (Feb 17)
CatBoost latest (catboost_latest.cbm) → 47 features, Feb 21 retrain, NOT DEPLOYED (51.55% WF acc, marginal vs 35-feat)
LightGBM latest (lightgbm_latest.txt) → 47 features, Feb 20, DEAD END (43% accuracy)
XGBoost latest (xgboost_latest.json) → 47 features, Feb 20, DEAD END (40% accuracy)
Meta-learner (ml/meta_learner.py) → 27 meta-features, Feb 20, NOT DEPLOYED (53.3% vs 53.8% fixed weights)
O/U 2.5 CatBoost (ou_2_5_catboost_latest.cbm) → 45 features, Optuna-tuned, ACTIVE (LL=0.6923, acc=52.2%)
O/U 1.5 CatBoost (ou_1_5_catboost_latest.cbm) → 50 features, Optuna-tuned, ACTIVE (LL=0.5669, acc=74.6%)
O/U 3.5 CatBoost (ou_3_5_catboost_latest.cbm) → 47 features, Optuna-tuned, ACTIVE (LL=0.5834, acc=73.1%)
Market models (prod_*.cbm) → 367 or 456 features (hybrid strategy)
Player models (player_*.cbm) → 153 features
Draw specialist (draw_specialist.pkl) → Side model for draw detection
```

## Weight Re-optimization with Market Cap (Feb 16 2026)

### Problem
The previous weights (ML=68.9%, market=10.4%) were not optimized for betting edge-finding. Optuna optimization without market caps converges to ~83% market weight because Pinnacle odds are the most accurate single predictor — but that copies the market instead of beating it.

### Solution
Re-optimized with normalized market cap ≤ 15% (3000 trials, TPE sampler). This forces the ensemble to derive 85% of its signal from independent methods (ML, xG, factor, player_xG) while using just enough market signal for calibration.

### Results (760 matches, 2023-2025)
| Config | ML% | Mkt% | xG% | Fac% | PxG% | Acc | LL | Bets | Profit | ROI |
|--------|-----|------|-----|------|------|-----|-----|------|--------|-----|
| Previous (ML-heavy) | 68.9 | 10.4 | 10.4 | 9.8 | 0.6 | 53.6% | 0.969 | 436 | €870 | 10.0% |
| **New (market ≤15%)** | **51.5** | **15.0** | **12.1** | **16.5** | **4.8** | **53.8%** | **0.966** | **410** | **€1,045** | **12.7%** |

Key improvements:
- +€175 profit (+20%)
- +2.7pp ROI (10.0% → 12.7%)
- All 5 methods now contribute meaningfully (previously 3 were near-zero)
- ML temperature sharpened (0.54 → 0.43) for more decisive ML predictions
- Mild draw boost (1.08) compensates for slight draw under-prediction

### Optimization Landscape (all strategies tested)
| Strategy | Acc | LL | Bets | Profit | ROI |
|----------|-----|-----|------|--------|-----|
| Market = 0% | 53.0% | 0.971 | 417 | €786 | 9.4% |
| Market ≤ 15% | 53.8% | 0.966 | 410 | €1,045 | 12.7% |
| Market ≤ 30% | 54.2% | 0.961 | 395 | €972 | 12.3% |
| No cap | 55.4% | 0.951 | 265 | €859 | 16.2% |

Trade-off: each 15% of market weight buys ~0.5pp accuracy but reduces bet volume and total profit.

## Fix 2+3: No-Odds ML + Recalibration (Feb 16 2026)

### Fix 2: Independent ML Signal
The ML classifier was dominated by bookmaker odds features (odds_B365H, odds_PSCH, etc.), making it redundant with the market predictor (together 79.3% of ensemble was odds-based). Trained a new CatBoost model using only team-performance features (44 features, zero odds).

### Fix 3: Ensemble Recalibration
The draw_boost (1.321x) and post-ensemble temperature (T=1.289) were calibrated for the old odds-dominated model. With the no-odds ML that predicts draws well, these were counterproductive — over-predicting draws (35.7% vs 28.9% actual) and under-predicting home wins (34.5% vs 40.8%). Removed both (set to 1.0).

### Combined Results (760 matches, 2023-2026)
| Stage | Accuracy | Log-loss | Brier | ECE |
|-------|----------|----------|-------|-----|
| Before (odds ML, boost=1.321, T=1.289) | 49.74% | 0.9917 | 0.1994 | 0.0435 |
| After Fix 2 (no-odds ML, old calibration) | 52.37% | 0.9822 | 0.1963 | 0.0438 |
| After Fix 2+3 (no-odds ML, no boost/T) | 53.55% | 0.9686 | 0.1930 | 0.0305 |
| **After weight re-opt (market ≤15%)** | **53.82%** | **0.9658** | — | — |

Top no-odds features: elo_diff (14.9%), matchup_competitiveness (5.4%), home_comeback_rate (3.6%), away_losing_at_ht_rate (3.6%), home_stadium_capacity (3.5%), attack_strength_diff (3.4%).

### Key ML Insight
The no-odds model has lower standalone accuracy (52.26% vs 53.20%) but improves the ensemble because it provides genuinely independent signal. This is the **diversity principle**: independent weaker signals beat correlated stronger ones in an ensemble.

## Calibration Analysis (Feb 17 2026)

### Measured ECE: 0.031 (760 matches, baseline weights)

With the baseline ensemble weights (ML=51.5%, market=15%, factor=16.5%, xG=12.1%, player_xG=4.8%), the model is well-calibrated: ECE=0.031, well below the 0.05 threshold.

**Important:** The formation weights path (used when `use_formation=True`) has ECE=0.065 — much worse because it excludes the ML classifier which is the strongest calibration signal. The formation path uses only factor/xG/market/formation with no ML. This path should only be used for formation-analysis research, not for production predictions.

**Per-class breakdown (baseline weights):**
| Class | Direction | Notes |
|-------|-----------|-------|
| Home | Slightly underconfident at high p | Normal for ensemble |
| Draw | Well-calibrated | ECE ~0.01 |
| Away | Overconfident at low p (<0.25) | Longshot away bets risky |

**Key finding:** Away predictions at low probability (0.15-0.25) are overconfident even with baseline weights. The no-odds ML model helps correct this but doesn't fully eliminate it. Longshot Away bets require extra caution.

**Post-hoc calibration analysis (formation weights, ECE=0.065):**
Isotonic, Platt, and temperature scaling all failed OOS — post-hoc calibration overfits with 380 samples/season. Not needed with baseline weights where ECE is already 0.031.

**Fixes applied:**
- Probability clipping in factor predictor (backtest) and ensemble output (negative prob bug)
- ECE now measured in baselines (was TBD)

## Multi-Market Backtest (Feb 17 2026)

760 matches (2023-2025), flat €20 stakes, Pinnacle closing odds as sharp benchmark.

### Market Profitability Summary
| Market | Best Source | Best Thresh | ROI | Bets | Status |
|--------|------------|-------------|-----|------|--------|
| 1X2 Draw | ensemble | 4% | **+24.0%** | 341 | ENABLED |
| O/U Over | blended_40 lineup xG | 5% | **+25.2%** | 51 | ENABLED |
| 1X2 Away | ensemble | 5% | +3.0% | 178 | Marginal, monitor |
| 1X2 Home | ensemble | 8% | +19.9% | 14 | Small sample |
| O/U Under | any | any | -1% to -14% | 130-300 | **DISABLED** |
| AH (both) | any | any | -20% to -42% | 200-400 | **DISABLED** |

### Lineup xG Impact on O/U Over (at 5% edge)
| xG Source | ROI | Win Rate | Bets | CLV |
|-----------|-----|----------|------|-----|
| team_xg (baseline) | -0.5% | 47.1% | 187 | -0.009 |
| lineup_xg (100%) | +12.5% | 62.2% | 37 | +0.003 |
| **blended_40** (60/40) | **+25.2%** | **64.7%** | **51** | **+0.003** |

Lineup xG uses player-level per-90 xG rates from confirmed starting XI. The 40% blend weight was optimal in backtest — pure lineup xG has too few bets, team-only has no edge.

### Season Consistency (1X2 Draw at 5% edge)
| Season | Bets | ROI |
|--------|------|-----|
| 2023-24 | 147 | +13.5% |
| 2024-25 | 142 | +20.1% |

### Situational Patterns (≥20 bets at 5% edge, updated Feb 20 2026)
| Situation | Market | ROI | Bets | Action |
|-----------|--------|-----|------|--------|
| Mgr change home | 1X2 Draw | +107.5% | 21 | Lower threshold (-1.5pp) |
| Post-intl home | 1X2 Draw | +85.6% | 28 | Lower threshold (-1.0pp) |
| Mgr change away | 1X2 Draw | +64.4% | 18 | Lower threshold (-1.5pp) |
| Derby | 1X2 Draw | +54.4% | 35 | Lower threshold (-2.0pp) |
| Rest advantage | 1X2 Draw | +52.8% | 61 | Lower threshold (-1.0pp) |
| Promoted away | 1X2 Draw | +33.9% | 86 | Lower threshold (-1.5pp) |
| Post-intl away | O/U Over | +39.1% | 28 | Lower threshold (-1.0pp) |
| Rest disadvantage | O/U Over | +37.1% | 42 | Lower threshold (-1.0pp) |
| Promoted home | O/U Over | +22.1% | 49 | Lower threshold (-1.0pp) |
| Derby | O/U Over | -56.0% | 30 | **Block** (+3.0pp) |
| Midweek | O/U Over | -25.3% | 19 | Raise threshold (+2.0pp) |

Note: All 9 situational adjustments are now FULLY ACTIVE in `betting_unified.py` as of Feb 20. Previously, 3 tag types (rest_advantage, rest_disadvantage, post_intl) were dead code.

### Why AH Fails
The Poisson model predicts total goals and 1X2 outcomes. Asian Handicap requires predicting the margin of victory — a fundamentally different problem. The model has no edge in margin prediction.

## Retrain with Substitution Features (Feb 20 2026)

### What Was Done
Backfilled substitution data from `data/external/sofascore/match_incidents.parquet` (19,281 sub events → 3,246 matches, 2017-2026). Added 6 features: `home/away_avg_subs_per_game`, `home/away_avg_sub_minute`, `home/away_sub_games_tracked`. Rebuilt features.parquet (7829 × 969). Ran full `train_optimized(exclude_odds=True, n_tune_trials=20)` — 3 models × 20 Optuna trials + 16-fold walk-forward CV.

### Results
| Model | All-folds Acc | All-folds LL | Test 2025-26 Acc | Test 2025-26 LL |
|-------|:---:|:---:|:---:|:---:|
| CatBoost (new) | 50.91% | 0.9973 | 51.97% | 1.0005 |
| CatBoost (old, production) | 50.75% | 1.0027 | 51.97% | 0.9950 |
| XGBoost | 40.52% | 1.0660 | — | — |
| LightGBM | 43.05% | 1.0682 | — | — |

### Outcome: NOT DEPLOYED
- Substitution features: **0 of 6 selected** by feature importance pruning. They don't carry signal.
- CatBoost improvement is within noise (+0.16pp all-folds accuracy, test fold slightly worse on LL).
- XGBoost/LightGBM confirmed as dead ends (40-43% vs 51%).
- Post-hoc calibration (Platt) worsened ECE on all 5 calibration folds — identity passthrough used.
- Production `catboost_no_odds.cbm` remains the Feb 17 version.
- New models saved to `catboost_latest.cbm` / `lightgbm_latest.txt` / `xgboost_latest.json` for reference.

### Conclusion
The no-odds model ceiling is ~51-52% accuracy. Adding more team-level features doesn't move this. The constraint is not feature quantity — it's information content. Future improvements should focus on ensemble architecture, meta-learner design, or better use of market signal, not feature engineering.

## Draw-Aware Optimization (Feb 2026)
Added 8 draw-convergence features (`both_defenses_strong`, `both_attacks_weak`, `combined_draw_tendency`, `defense_similarity`, `low_scoring_signal`, `home/away_draw_tendency_5`, `draw_convergence_x_competitiveness`). Retrained ML blend with 30% draw F1 objective, re-optimized ensemble weights via 300-trial Optuna with draw-aware scoring. Result: draw predictions in 55-65% bucket went from 0/2 correct to 4/8 correct. 2-season log-loss improved (0.8987→0.8958). NOTE: Draw boost was subsequently removed (Fix 3) since no-odds ML handles draws without needing a boost; a mild 1.08x boost was re-introduced during weight re-optimization.
