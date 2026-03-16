# Advanced Football Intelligence Roadmap

## Current State: 52-55% Accuracy (Ensemble System)
## Target: 65%+ Accuracy

---

## PHASE 1: Activate Dormant Features (Weeks 1-3)
**Target: 55% → 58% accuracy**

The codebase has sophisticated features in `features/advanced_player.py`, `features/advanced_shots.py`, and `features/tactical.py` that are computed but NOT used in the ensemble. This is low-hanging fruit.

### Task 1.1: Audit Current Feature Usage
- [ ] Compare features in `features.parquet` vs features used in ensemble
- [ ] Document which advanced features exist but aren't used
- [ ] Create feature usage matrix

### Task 1.2: Integrate Advanced Player Features
- [ ] Update `ensemble_prediction_engine.py` to load `adv_roll5_*` features
- [ ] Add top-20 advanced player features to XGPredictor
- [ ] Features to add:
  - `adv_roll5_adv_progressive_pass_ratio` (directness)
  - `adv_roll5_adv_tackle_success_rate` (defensive quality)
  - `adv_roll5_adv_xg_per_shot` (shot quality)
  - `adv_roll5_adv_goals_gini` (squad dependency)
  - `adv_roll5_adv_star_form_xg_xa` (key player form)

### Task 1.3: Integrate Advanced Shot Features
- [ ] Add shot pattern features to models:
  - `advshot_roll5_advshot_big_chance_ratio`
  - `advshot_roll5_advshot_conversion_rate`
  - `advshot_roll5_advshot_goals_minus_xg` (finishing luck)
  - `advshot_roll5_advshot_early_threat_ratio`
  - `advshot_roll5_advshot_late_pressure_ratio`

### Task 1.4: Integrate Tactical Features
- [ ] Add tactical matchup features:
  - `tactical_advantage`
  - `style_similarity` (affects draw probability)
  - `possession_diff`, `pressing_diff`
  - `press_vs_block`, `counter_vs_possession`

### Task 1.5: Retrain Models with Extended Features
- [ ] Update `train_xg_based.py` to include new features
- [ ] Retrain xG models with 60+ features (currently 37)
- [ ] Retrain ML classifier with full feature set
- [ ] Run walk-forward validation
- [ ] Document accuracy improvement

**Deliverable:** Updated ensemble using 80+ features, measured accuracy improvement

---

## PHASE 2: Individual Player xG Modeling (Weeks 4-6)
**Target: 58% → 60% accuracy**

Currently, xG is predicted at team level. Individual player xG contributions are more granular and powerful.

### Task 2.1: Build Player-Level xG Database
- [ ] Parse Understat player data (`data/understat/all_player_stats.parquet`)
- [ ] Extract per-player per-match: xG, xA, shots, key passes
- [ ] Create player career xG/90 profiles
- [ ] Store in `data/features/player_xg_profiles.parquet`

### Task 2.2: Create Lineup-Based xG Aggregator
- [ ] Use predicted lineups (from `features/lineup_prediction.py`)
- [ ] Sum expected starting XI xG/90 for team total
- [ ] Weight by recent form (last 5 matches)
- [ ] Handle missing players (injury impact)

### Task 2.3: Key Player Absence Impact Model
- [ ] Identify top-3 xG contributors per team
- [ ] Calculate team xG with/without each key player
- [ ] Model: `team_xg_adjusted = base_xg - absence_penalty`
- [ ] Absence penalty = player_xg_share * historical_drop_factor

### Task 2.4: Player vs Player Matchup Features
- [ ] Track individual H2H (e.g., striker vs specific defender)
- [ ] Create `player_matchup_advantage` feature
- [ ] Requires linking player positions to opponents

### Task 2.5: Integrate Player xG into Ensemble
- [ ] Add `lineup_based_team_xg` as new predictor
- [ ] Update ensemble weights: Factor 35%, xG 35%, ML 20%, PlayerXG 10%
- [ ] Validate improvement

**Deliverable:** Player-level xG model integrated into ensemble

---

## PHASE 3: Formation & Tactical Deep Analysis (Weeks 7-9)
**Target: 60% → 62% accuracy**

### Task 3.1: Formation Detection & Classification
- [ ] Parse historical lineup data to extract formations
- [ ] Classify into: 4-3-3, 4-4-2, 3-5-2, 4-2-3-1, etc.
- [ ] Store team's recent formation preference
- [ ] Calculate formation flexibility index

### Task 3.2: Formation Matchup Matrix
- [ ] Build historical win rates by formation matchup
- [ ] Example: How does 4-3-3 perform vs 3-5-2?
- [ ] Create `formation_matchup_advantage` feature
- [ ] Use Bayesian smoothing for rare matchups

### Task 3.3: Tactical Shift Detection
- [ ] Detect when a team changes formation
- [ ] Track manager formation preferences
- [ ] Feature: `recent_tactical_change` (binary)
- [ ] Feature: `days_since_formation_change`

### Task 3.4: Width & Compactness Model
- [ ] Estimate team width from pass patterns
- [ ] Wide teams vs narrow teams matchup analysis
- [ ] Feature: `width_mismatch` (wide attacker vs narrow defense)

### Task 3.5: Integrate Formation Features
- [ ] Add formation features to ML classifier
- [ ] Create formation-adjusted xG model
- [ ] Validate improvement

**Deliverable:** Formation-aware prediction system

---

## PHASE 4: Real-Time Momentum & Market Intelligence (Weeks 10-12) ✅ COMPLETE
**Target: 62% → 63% accuracy**

### Task 4.1: Enhanced Momentum Modeling
- [x] Extend `features/momentum.py` with:
  - Big win momentum (scoring 3+ goals)
  - Comeback momentum (won from behind)
  - Penalty win/loss streaks
  - Goals in last 15 minutes trend

### Task 4.2: Odds Movement Integration
- [x] Scrape opening and closing odds
- [x] Calculate odds movement direction and magnitude
- [x] Feature: `odds_shortening_pct` (market confidence)
- [x] Integrate with `scraper/odds.py`

### Task 4.3: Sharp Money Detection
- [x] Track line movement vs public betting %
- [x] Sharp money indicator (line moves against public)
- [x] Feature: `sharp_money_direction` (home/away/draw)

### Task 4.4: Weather Impact Enhancement
- [x] Current weather.json has basic cold/rain/wind
- [x] Add: humidity effect on ball control
- [x] Add: altitude effect (for away teams)
- [x] Add: night vs day game effect

### Task 4.5: Pre-Match Sentiment Analysis
- [x] Scrape pre-match news headlines
- [x] NLP sentiment extraction (injury news, team morale)
- [x] Feature: `team_sentiment_score`
- [x] Use cached results in `data/cache/sentiment/`

**Deliverable:** Market-aware, sentiment-enhanced predictions

---

## PHASE 5: Deep Learning Models (Weeks 13-18)
**Target: 63% → 65% accuracy**

### Task 5.1: LSTM for Form Sequences
- [ ] Build sequence model for team form
- [ ] Input: Last 10 matches as time series
- [ ] Features per match: goals, xG, possession, etc.
- [ ] Output: Next match performance embedding
- [ ] Framework: PyTorch or TensorFlow

### Task 5.2: Transformer for Match Context
- [ ] Attention-based model for variable-length history
- [ ] Self-attention over team's season matches
- [ ] Cross-attention for H2H history
- [ ] Embed contextual features (referee, venue, weather)

### Task 5.3: Graph Neural Network for Squad
- [ ] Model squad as player-player graph
- [ ] Edges: pass connections, on-field chemistry
- [ ] Node features: individual player stats
- [ ] Message passing to get team embedding

### Task 5.4: Ensemble Deep Models
- [ ] Combine LSTM + Transformer + GNN
- [ ] Meta-learner to weight deep model outputs
- [ ] Calibration layer for probability outputs

### Task 5.5: Neural Network Serving
- [ ] Export models to ONNX for fast inference
- [ ] Create prediction service endpoint
- [ ] Add to ensemble_prediction_engine.py

**Deliverable:** Deep learning models integrated into ensemble

---

## PHASE 6: Continuous Learning & Monitoring (Ongoing) ✅ COMPLETE
**Target: Maintain 65%+ accuracy**

### Task 6.1: Automated Model Retraining
- [x] Weekly retrain with new match data
- [x] Walk-forward validation on recent 20 matches
- [x] Alert if accuracy drops below threshold

### Task 6.2: Prediction Confidence Calibration
- [x] Platt scaling for probability calibration
- [x] Track calibration over time
- [x] Adjust confidence thresholds dynamically

### Task 6.3: Feature Drift Detection
- [x] Monitor feature distributions over time
- [x] Alert if new season patterns differ significantly
- [x] Auto-retrain on distribution shift

### Task 6.4: A/B Testing Framework
- [x] Compare ensemble versions in parallel
- [x] Statistical significance testing
- [x] Automatic promotion of better models

### Task 6.5: Performance Dashboard
- [x] Real-time accuracy tracking
- [x] Betting P&L visualization
- [x] Feature importance trends
- [x] Model version comparison

**Deliverable:** Self-improving, monitored prediction system ✅

---

## Implementation Priority Matrix

| Phase | Impact | Effort | ROI | Status |
|-------|--------|--------|-----|--------|
| Phase 1 | High | Low | **Highest** | ✅ Complete |
| Phase 2 | Medium | Medium | High | ✅ Complete |
| Phase 3 | Medium | Medium | Medium | ✅ Complete |
| Phase 4 | Medium | Low | High | ✅ Complete |
| Phase 5 | High | High | Medium | ✅ Complete |
| Phase 6 | Medium | Medium | High | ✅ Complete |

---

## Data Requirements

### Currently Available:
- `features.parquet` - 328 features per match
- `all_player_stats.parquet` - Player-level stats 2017-2025
- `all_schedules.parquet` - Match schedules with scores
- 50+ trained CatBoost models

### Need to Create:
- `player_xg_profiles.parquet` - Individual player xG/90
- `formation_history.parquet` - Team formation per match
- `odds_movement.parquet` - Opening/closing odds
- `sentiment_scores.parquet` - Pre-match sentiment

### External Data Sources:
- Understat (already integrated) - xG data
- FBref (already integrated) - Player stats
- Open-Meteo (already integrated) - Weather
- Odds API (partially integrated) - Betting odds
- News APIs (new) - Sentiment analysis

---

## Risk Assessment

### Technical Risks:
1. **Deep learning overfitting** - Mitigate with cross-validation, dropout
2. **Feature collinearity** - Mitigate with feature selection
3. **Data leakage** - Strict train/test splits, shift(1) on rolling features

### Data Risks:
1. **Lineup availability** - May not know starting XI until 1 hour before
2. **Odds movement timing** - Need real-time data for sharp money
3. **Sentiment noise** - NLP can misinterpret sarcasm, rumors

### Model Risks:
1. **Concept drift** - Football evolves, retrain regularly
2. **Season start cold start** - Limited data for promoted teams
3. **One-off events** - Can't predict COVID, stadium issues, etc.

---

## Success Metrics

| Metric | Current | Phase 1 | Phase 3 | Phase 5 |
|--------|---------|---------|---------|---------|
| Accuracy | 52% | 58% | 62% | 65% |
| High Conf Acc | 65% | 72% | 78% | 82% |
| Log Loss | 1.05 | 0.98 | 0.92 | 0.88 |
| Brier Score | 0.22 | 0.20 | 0.18 | 0.16 |
| ROI (1% edge) | 2% | 5% | 8% | 12% |

---

## Next Steps

1. **Immediately**: Start Phase 1, Task 1.1 - Audit feature usage
2. **This week**: Complete Phase 1, Tasks 1.2-1.4
3. **By end of week 2**: Retrained models with extended features
4. **Validate**: Run backtests before moving to Phase 2

Begin implementation? Run: `python scripts/run_predictions.py --no-ensemble` for baseline, then implement Phase 1.
