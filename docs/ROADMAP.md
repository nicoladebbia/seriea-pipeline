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

### High Priority (clear ROI, should do next)

- **Scale bankroll**: Increase weekly staking from ~EUR 140 to EUR 500-1,000 — +9.2% ROI is proven, bet bigger
- **Premier League expansion**: Add EPL data, scrapers, models — doubles bet volume, diversifies risk. Infrastructure exists in `config/settings.py` LEAGUES dict.
- **Walk-forward backtest fix**: Production CatBoost is trained on ALL data — backtest ROI is inflated by data leakage. Need proper walk-forward retraining per backtest fold for reliable historical analysis.
- **Confirmed lineups integration**: Sofascore fetcher works, lineup data flows to ensemble (player_xg 5%→12%, O/U lineup_xg 15%→40%). Need to verify on live matchday that scheduler T-60min fetch triggers correctly.

### Medium Priority (good ideas, need validation)

- **Specialized draw model**: Current 1X2 disabled because model can't predict draws accurately enough (28% WR, need 31%). A dedicated draw classifier trained on draw-specific features could re-enable 1X2 draws profitably.
- **Learn from mistakes feedback loop**: Track prediction errors by game conditions (form, injuries, weather) and feed back into next prediction. Currently `learning_loop.py` exists but doesn't track errors by condition.
- **Feature drift monitoring**: Alert when feature importance shifts significantly between training runs
- **LiveBiasCorrector activation**: `features/prediction_calibration.py` will auto-activate once 30+ predictions settle with fair_odds_ledger data
- **Test coverage expansion**: Increase test coverage for `ml/` and `features/` modules

### Low Priority (nice-to-have, future)

- **Multi-league support**: La Liga, Bundesliga (after EPL proves the architecture generalizes)
- **Live in-play predictions**: Real-time probability updates during matches
- **Automated retraining**: Scheduled model retraining as new season data accumulates
- **API endpoint**: REST API for predictions (replacing Flask dashboard)
- **Historical backtest dashboard**: Interactive visualization of historical prediction accuracy
- **Transfer window impact**: Better modeling of January/summer window effects on team strength

## Historical Documentation

Archived phase completion reports and optimization summaries are preserved in `docs/archive/`.
