# Baselines & Rejection Thresholds

Read this before running backtests, validating models, or shipping changes.

## Current Baselines (NEVER regress beyond these)
| Metric | Walk-Forward CV (2017+) | Production 2025-2026 | Reject If |
|--------|------------------------|---------------------|-----------|
| Accuracy | 60.0% | 69.3% | < 55.0% |
| Log-loss | 0.860 | 0.709 | > 0.92 |
| Brier | 0.170 | — | > 0.18 |
| ECE | 0.031 (production) | 0.031 | > 0.05 |
| F1 Draw | 0.328 | — | < 0.20 |
| Walk-fwd backtest ROI | +12.3% | — | < +5% |

Note: Baselines updated Mar 23 2026 after major model overhaul:
- Training data: 2017+ only (3,340 matches) — drops low-signal pre-2017 data
- NaN threshold: 0.45 (was 0.20) — unlocks xG, lineup, odds velocity, pressing features
- Time-decay: 0.85/season (Dixon-Coles) — recent matches dominate
- Auto draw weights: 38% target effective share (was hardcoded 2.0x multiplier)
- 35 features selected (including xG diff, lineup rating, odds velocity, squad value)
- Walk-forward backtest (2023-2025): 643 bets, €3,192 profit, +12.3% ROI, €1000→€4192
- Betting rules: max_edge 8%, steam rejection, odds 1.5-2.0 dead zone (9% min), Kelly min 0.3%

## Key Constants (updated Mar 23 2026)
| Parameter | Value | Context |
|-----------|-------|---------|
| Ensemble weights | F=0.035, xG=0.124, ML=0.605, PxG=0.032, D=0.00, M=0.205 | `ensemble_prediction_engine.py` |
| ML model | catboost_no_odds.cbm + 3-model ensemble (35 features, 2017+ data) | Time-decay 0.85, auto draw weights |
| ML temperature | T=0.75 | Pre-ensemble sharpening |
| Post-ensemble T | T=0.90 | Sharpens underconfident predictions |
| Draw boost | 1.12 | Draw compensation |
| Staking mode | **kelly** | Kelly criterion with fractional sizing |
| Kelly fraction | **0.10** | 10% Kelly — balances growth vs variance |
| Kelly min reject | **0.30%** | Bets below 0.3% Kelly are rejected (edge too thin) |
| Max stake | **2.5% of bankroll** | Per bet |
| Max edge | **8%** (all markets) | Live data: >10% = 38% WR, -EUR133 (was 12%) |
| Odds dead zone | **1.5-2.0: 9% min edge** | Live data: 40% WR, -EUR121 in this range |
| Steam rejection | **Hard reject** | Steam moves against = bet killed entirely |
| DC min edge | 5% | Double chance |
| O/U Over min edge | 6% | Over/Under |
| 1X2 | **DISABLED** | Live: negative ROI |
| O/U Under | **DISABLED** | Backtest: negative ROI at all thresholds |
| AH | **DISABLED** | Backtest: negative ROI |

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

## Model Registry (updated Mar 23 2026)
```
ACTIVE PRODUCTION:
  catboost_no_odds.cbm → 35 features, 2017+ data, time-decay 0.85, CV acc=61.6% (Mar 23)
  Ensemble (XGB+LGB+CB) → 35 features, blend 30/36/34, CV acc=60.0% (Mar 23)
  O/U CatBoost models → ou_{1.5,2.5,3.5}_catboost_latest.cbm, Optuna-tuned

AUXILIARY (not in main ensemble):
  xG regressors → xg_home.cbm, xg_away.cbm (for Poisson predictions)
  Draw detector → draw_detector.cbm (auxiliary signal)

ARCHIVED:
  Old catboost_no_odds.cbm.bak → 45 features, Feb 23 (rollback if needed)
  catboost_latest.cbm → from train_optimized, auto-updated on retrain
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
| CatBoost (Feb 21, all-data) | 50.91% | 0.9973 | 51.97% | 1.0005 |
| **CatBoost (Mar 23, 2017+)** | **61.55%** | **0.8589** | **69.3%** | **0.709** |
| XGBoost (Mar 23, 2017+) | 59.2% | 0.866 | — | — |
| LightGBM (Mar 23, 2017+) | 59.1% | 0.866 | — | — |

### Outcome: NOT DEPLOYED (Feb 21)
- Substitution features: **0 of 6 selected** by feature importance pruning. They don't carry signal.
- CatBoost improvement is within noise (+0.16pp all-folds accuracy, test fold slightly worse on LL).
- XGBoost/LightGBM confirmed as dead ends (40-43% vs 51% on all-data training).
- Post-hoc calibration (Platt) worsened ECE on all 5 calibration folds — identity passthrough used.

### Conclusion (UPDATED Mar 23)
The old no-odds model ceiling of ~51-52% was broken by switching to **2017+ data only** (min_train_season=2017-2018, NaN threshold 0.45) + **time-decay 0.85/season** + **auto draw weights**. This unlocked xG, lineup, odds velocity, and squad value features. New model: **61.55% CV accuracy**, log-loss 0.8589, F1 Draw 0.328. Walk-forward backtest: +12.3% ROI. The constraint was not feature quantity — it was that pre-2017 data with zero xG coverage was drowning out the signal.

## Draw-Aware Optimization (Feb 2026)
Added 8 draw-convergence features (`both_defenses_strong`, `both_attacks_weak`, `combined_draw_tendency`, `defense_similarity`, `low_scoring_signal`, `home/away_draw_tendency_5`, `draw_convergence_x_competitiveness`). Retrained ML blend with 30% draw F1 objective, re-optimized ensemble weights via 300-trial Optuna with draw-aware scoring. Result: draw predictions in 55-65% bucket went from 0/2 correct to 4/8 correct. 2-season log-loss improved (0.8987→0.8958). NOTE: Draw boost was subsequently removed (Fix 3) since no-odds ML handles draws without needing a boost; a mild 1.08x boost was re-introduced during weight re-optimization.
