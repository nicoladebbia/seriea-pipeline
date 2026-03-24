# Roadmap

## Completed Phases

### Phase 1: Extended Feature Integration
- Expanded from 37 to 139 features
- Integrated advanced player and shot features into ensemble
- xG-Poisson CV accuracy: 50% -> 53.1%

### Phase 2: Market Intelligence
- Sharp/soft bookmaker classification
- Cross-market correlation analysis
- Odds movement tracking and steam detection

### Phase 4: Enhanced Momentum & Weather
- Big win momentum, comeback tracking, late goal trends
- Enhanced weather features (precipitation, wind, temperature)
- 8 new interaction features

### Phase 5: Deep Learning
- Neural network classifier added to ensemble
- TensorFlow-based deep predictor
- Ensemble weight: 0.07 (currently 0.00 — absorbed by ML)

### Phase 6: Calibration & Strategy
- Draw detection module
- Prediction calibration pipeline
- Strategy-based betting system (conservative, balanced, aggressive)

### ML Engineering Overhaul (Feb 16-17 2026)
- No-odds CatBoost model (35 features) replacing odds-dominated ML
- Ensemble weight re-optimization with market cap ≤15%
- Draw boost recalibration (1.321 → 1.28)
- ECE validation (0.031 — well-calibrated)
- Multi-market backtest: 1X2 Draw +24% ROI, O/U Over +25.2% with lineup xG
- O/U Under DISABLED (-1% to -14% ROI), AH DISABLED (-20% to -42% ROI)

### Situational Edge Exploitation (Feb 20 2026)
- 9 situational adjustments fully activated in betting engine
- Rest advantage/disadvantage, post-international break, derby, manager change tags
- Lineup fetcher verified end-to-end (Sofascore → fd.org → API-Football cascade)

### O/U Multi-Line ML Models (Feb 21 2026)
- CatBoost classifiers for lines 1.5, 2.5, 3.5 (was 2.5 only)
- Optuna-tuned hyperparameters, 65% ML + 35% Poisson blend
- Features rebuild: 7829 × 970

### Betting System Hardening (Feb 22 2026)
- 1X2 DISABLED (live: 5W/13L -17.6% ROI, 20% CLV beat rate)
- O/U expanded to lines 0.5-4.5 via alternate totals merge
- 3-bookmaker minimum for all O/U bets (prevents phantom edges)
- Duplicate scanner removed, staking mode switched to Kelly 10%
- Journal naming consistency fixes

### Negative Results (documented, don't retry)
- Meta-learner stacking: 53.3% vs 53.8% fixed weights — NOT deployed
- Substitution features: 0/6 survived feature importance pruning (41.4% NaN)
- XGBoost/LightGBM for 1X2: 40-43% accuracy — dead ends
- Post-hoc calibration (Platt, isotonic, temperature): overfits with 380 samples/fold

## Current State

- **Data**: 21 seasons (7,829 matches), 970 features in features.parquet
- **Ensemble**: 6-method blend — ML (60.5%), market (20.5%), factor (3.5%), xG (12.4%), player_xg (3.2%), deep (0%)
- **ML model**: CatBoost no-odds, 35 features, 54.3% accuracy, leakage-free
- **O/U models**: CatBoost classifiers for lines 1.5, 2.5, 3.5 (Optuna-tuned)
- **Pipeline**: 33-step prediction pipeline with live odds integration (~10 min)
- **Staking**: Kelly criterion at 10% fraction, 2.5% max stake per bet
- **Active markets**: O/U Over (lines 0.5-4.5), DC (home/away), 1X2 Away (marginal)
- **Disabled markets**: 1X2, 1X2 Draw, O/U Under, AH
- **Bookmaker minimum**: 3 bookmakers required for all O/U bets
- **Live performance** (64 bets): +1.3% ROI overall, +9.2% without 1X2, 70% CLV beat rate
- **Web dashboard**: Flask app with predictions, betting intelligence, live odds

## Planned Improvements

> **See also:** `.claude/system_guide.md` Section 13 (Research Frontier) for detailed
> implementation notes, risks, and prerequisites for each item below.

### Tier 0: Fix Known Bugs Degrading Current Performance

These are not improvements — they are documented bugs actively hurting results.

- ~~**Feature selection fold-0 bias** (KB #37)~~: **FIXED (2026-03-22).** `ml/feature_selection.py` now uses recency-weighted importance averaging (exponential decay, base=1.5) + supplementary recent-folds pass (last 4 folds). Modern Sofascore/FBref features are no longer penalized by early-fold imputation. Next: retrain to see how many new features are recovered.
- ~~**Walk-forward backtest leakage**~~: **FIXED (2026-03-22).** `ml/ensemble.py` has `build_fold_models()` / `load_fold_model()` for per-fold CatBoost persistence. `backtest_multimarket.py --walk-forward` flag. Results now include `evaluation_type` and leakage warnings. Next: build fold models and re-run production backtest for honest ROI.
- ~~**Deployment state uncertainty** (KB #19)~~: **FIXED (2026-03-22).** Two models exist by design: catboost_no_odds (`catboost_no_odds_metadata.json`) and ensemble (`training_report.json`). `deployment_state.json` has `active_ml_model` field. `health_check.py` validates consistency at runtime.

### Tier 1: Change the Optimization Target

The system optimizes for 1X2 classification accuracy. But money comes from calibrated probabilities in specific market contexts. Accuracy is the wrong metric.

- **Train for calibration, not accuracy**: Custom eval metric = Brier score or ECE instead of log-loss/accuracy. A 50% accurate model with perfect calibration makes more money via Kelly than a 54% accurate model with poor calibration.
- **Closing Line Value (CLV) as training signal**: Target = `(model_fair_odds / closing_odds) - 1`. `fair_odds_ledger.json` has 1,261 predictions. `clv_history.json` has per-bet CLV. A model that beats the closing line prints money regardless of accuracy.
- **Situation-weighted loss**: Weight training samples by situational tag profitability. Derbies, promoted teams, manager changes — these are where ROI lives. `Pool()` sample weights in CatBoost.

### Tier 2: Unlock Underutilized Market Signals

The system collects rich market microstructure data but uses it only as multiplicative confidence adjustments (0.3x-1.15x). These signals could be fundamental.

- **Promote odds velocity features to ODDS_META_KEEP**: 14 `line_vel_*` features are excluded despite capturing sharp money direction. Add `line_vel_pin_home/draw/away` and `steam_move_flag` to `ml/config.py:ODDS_META_KEEP`. 15-minute fix, then retrain.
- **Market microstructure edge modifier**: Replace flat 0.70x-1.15x adjustments with a learned function of (raw_edge, steam_direction, sharp_soft_divergence, odds_consistency, overround, bookmaker_count). A 6% edge with 20 bookmakers and sharps agreeing ≠ 6% edge in a thin market with sharps moving against you.
- **Bookmaker-specific CLV analysis**: `clv_history.json` records which bookmaker gave best price. Which books systematically offer better CLV per market type? Pure analysis, no model changes.
- **Activate StaticCorrector**: `correction_layer.py` logistic regression corrector was waiting for 30+ settled predictions. `fair_odds_ledger.json` now has 1,261 entries. Check activation condition and deploy.

### Tier 3: Situation-Specific Models

One universal model + post-hoc adjustments leaves money on the table for subpopulations that behave fundamentally differently.

- **Specialist ensemble**: Separate CatBoost models for derbies, promoted teams, post-international break, manager change matches. Use specialist when tag matches, universal otherwise. Derby specialist MVP: pool all derbies across 21 seasons (~150 matches), train, compare calibration.
- **Regime-aware ensemble weights**: Rule-based market regime detector → dynamic weight overrides. Early season → boost factor model. Late season → boost ML. Volatile odds day → boost market weight. Similar to situational edge adjustments but applied to ensemble blending.

### Tier 4: Scale Data via Multi-League Expansion

Multiple research paths are blocked by sample size (meta-learner needs 5K+, specialists need more per-situation data). This is the structural unlock.

- **Premier League first**: Infrastructure exists in `config/settings.py` LEAGUES dict. FBref + Sofascore coverage is excellent. 5 leagues × 380 matches/season × 21 seasons = ~40K matches. Meta-learner becomes viable. Specialists become viable. Fold-0 bias disappears.
- **La Liga, Bundesliga, Ligue 1**: After EPL proves the architecture generalizes. Need league indicator features or separate per-league models.

### Tier 5: Post-Prediction Edge

- **Live odds monitoring for entry timing**: Don't bet immediately. Set target odds, wait for market movement. If odds lengthen, bet at better value. `scheduler.py` already has time-based triggers.
- **Scale bankroll**: +9.2% ROI without 1X2, 70% CLV beat rate. EUR 140/week → EUR 500-1,000.

### Deprioritized (moved from previous roadmap)

- **Live in-play predictions**: Real-time probability updates during matches (complex, unclear ROI)
- **API endpoint**: REST API for predictions (not needed until multi-user)
- **Transfer window impact**: Better modeling of window effects (feature already exists in `features/transfer_window.py`, low importance)

## Historical Documentation

Archived phase completion reports and optimization summaries are preserved in `docs/archive/`.
