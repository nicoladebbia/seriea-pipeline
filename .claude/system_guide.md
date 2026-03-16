# Serie A Prediction System — Complete Guide

Last updated: 2026-02-20. Read this to understand how everything works.

---

## 1. HOW PREDICTIONS ARE MADE

### The Pipeline (30-second version)
```
Scrapers → features.parquet → ML model → Ensemble blend → Betting system
```

### Full Flow
```
1. SCRAPE: 23 scrapers pull data from FBref, Sofascore, Understat, Odds API, etc.
   → data/external/ (raw) → data/parsed/ (cleaned parquet files)

2. BUILD FEATURES: features/build.py orchestrates 49 plugins
   → data/features/features.parquet (7829 matches × 969 columns)

3. TRAIN MODEL: ml/training.py → train_optimized(exclude_odds=True)
   → Feature selection (969 → 35) → Optuna tuning → Walk-forward CV
   → data/models/universal/catboost_no_odds.cbm

4. PREDICT: scripts/prediction/ensemble_prediction_engine.py
   Blends 5 predictors:
   - ML classifier (CatBoost, 60.5% weight)
   - Market odds (Pinnacle, 20.5%)
   - xG Poisson model (12.4%)
   - Factor model (3.5%)
   - Player xG (3.2%)
   → data/upcoming/predictions.json

5. BET: scripts/betting/betting_unified.py
   Compares ensemble probabilities to bookmaker odds, finds edges
   → data/upcoming/unified_bet_slip.json
   → data/betting/bet_journal.json

6. SCHEDULE: scripts/pipeline/scheduler.py
   T-60 min before kickoff: fetches confirmed lineups (Sofascore)
   T-0: runs predictions with lineup data
   Post-match: settles bets, updates results
```

---

## 2. KEY FILES — WHAT EACH ONE DOES

### Pipeline Orchestration
| File | Purpose | When to read |
|------|---------|-------------|
| `scripts/pipeline/run_full_pipeline.py` | Main pipeline entry point. 32 steps. | Running predictions |
| `scripts/pipeline/scheduler.py` | Cron-like scheduler. Pre-kickoff lineups, predictions, post-match settlement. | Matchday automation |
| `scripts/pipeline/pipeline_config.py` | Pipeline step configuration | Changing pipeline behavior |

### Prediction Engine (157K — NEVER read fully)
| File | Key sections |
|------|-------------|
| `scripts/prediction/ensemble_prediction_engine.py` | `ENSEMBLE_WEIGHTS` (~line 100), `predict()` method, `_blend_predictions()` |
| `scripts/prediction/predict_unified.py` | Loads features, calls ensemble, saves predictions.json |
| `scripts/prediction/over_under_model.py` | O/U goals Poisson model. `lineup_xg` at line ~609 |
| `scripts/prediction/cards_model.py` | Yellow/red card predictions |
| `scripts/prediction/btts_corners_model.py` | BTTS + corners |

### Feature Engineering
| File | Lines | Purpose |
|------|-------|---------|
| `features/build.py` | ~1200 | ORCHESTRATOR. Runs 49 plugins. Cleanup: drops constants, high-null, exact-dups. |
| `features/derived.py` | ~400 | Core derived features (Elo, form, strength) |
| `features/rolling.py` | ~300 | Rolling averages (5/10 match windows) |
| `features/h2h.py` | ~200 | Head-to-head historical stats |
| `features/injury_impact.py` | ~150 | Injury impact scoring |
| `features/strength.py` | ~200 | Attack/defense strength ratings |
| `features/substitution_features.py` | ~100 | Substitution patterns (NEW Feb 20, but features not selected by model) |
| `features/prediction_calibration.py` | ~350 | Calibration pipeline + LiveBiasCorrector (NEW Feb 20) |

### ML Training
| File | Purpose |
|------|---------|
| `ml/training.py` | `train_optimized()` — the production training pipeline. Feature selection → Optuna → WF-CV → save. |
| `ml/tuning.py` | Optuna hyperparameter tuning |
| `ml/feature_selection.py` | Walk-forward importance-based selection |
| `ml/walk_forward.py` | Walk-forward cross-validation (expanding window, 1 season per fold) |
| `ml/data.py` | `TimeSeriesSplitter` — generates train/test splits by season |
| `ml/ensemble.py` | Ensemble weight optimization |
| `ml/calibration.py` | Platt/isotonic calibration (currently disabled — worsens ECE) |

### Betting
| File | Purpose |
|------|---------|
| `scripts/betting/betting_unified.py` (82K) | Edge calculation, Kelly sizing, bet selection. `BettingConfig` class. |
| `scripts/betting/bet_journal.py` | Bet tracking, settlement, P&L |
| `scripts/betting/bankroll_manager.py` | Bankroll tracking, risk limits |

### Scrapers
| Scraper | Data Source | Data |
|---------|-----------|------|
| `scraper/sofascore_events.py` | Sofascore API | Match stats, shots, xG |
| `scraper/sofascore_lineups.py` | Sofascore API | **Confirmed lineups** (T-60min) |
| `scraper/footballdata_lineups.py` | football-data.org | Backup lineups (KEY NOT SET) |
| `scraper/lineup_fetcher.py` | Cascade orchestrator | Sofascore → fd.org → API-Football |
| `scraper/odds.py` | The Odds API | Live odds from 20+ bookmakers |
| `scraper/fbref_auto_scraper.py` | FBref | Player stats, match reports |
| `scraper/understat_scraper.py` | Understat | Player xG, team xG |
| `scraper/transfermarkt.py` | Transfermarkt | Squad values, transfers |
| `scraper/injuries.py` | Various | Current injuries |
| `scraper/weather.py` | Open-Meteo | Match weather |
| `scraper/referee.py` | FBref | Referee assignments |
| `scraper/fixtures.py` | Sofascore | Fixture IDs + schedule |

### Configuration
| File | Purpose |
|------|---------|
| `config/team_names.py` | Team name normalization (CRITICAL — every scraper uses this) |
| `config/settings.py` | Global settings |
| `.env` | API keys (ODDS_API_KEY, FOOTBALLDATA_KEY) |

### Web Interface
| File | Purpose |
|------|---------|
| `web/app.py` | Flask app. Shows predictions, lineups, bet slips at `/predictions` |

---

## 3. MODELS — WHAT EXISTS

### Production (ACTIVE)
| Model | File | Features | Accuracy | Purpose |
|-------|------|----------|----------|---------|
| CatBoost no-odds | `catboost_no_odds.cbm` | 35 | 51.97% | **THE** ML component in ensemble |

### Non-production (reference only)
| Model | File | Status |
|-------|------|--------|
| CatBoost latest (Feb 20) | `catboost_latest.cbm` | NOT deployed, marginal improvement |
| LightGBM latest | `lightgbm_latest.txt` | 43% accuracy — much worse than CatBoost |
| XGBoost latest | `xgboost_latest.json` | 40% accuracy — much worse than CatBoost |
| Market models | `models/markets/*.cbm` | O/U, BTTS, corners, cards |
| Player models | `models/player/*.cbm` | Player props |
| Deep models | `models/deep/*` | LSTM/Transformer (weight=0, disabled) |
| Draw specialist | `draw_specialist.pkl` | Draw detection side model |

### 35 Production Features (catboost_no_odds)
```
elo_diff, home_stadium_capacity, attack_strength_diff, away_losing_at_ht_rate,
matchup_competitiveness, is_late_season, away_gd_per_match, home_losing_at_ht_rate,
away_roll_5_shots_on_target, home_league_gd, away_roll_5_yellow_cards,
away_injury_impact, elo_x_form, congestion_asymmetry, tenure_x_form,
home_gd_roll_5, home_opp_difficulty_roll_5, home_chemistry_disruption,
combined_disruption, away_roll_5_clean_sheet, away_roll_5_red_cards,
injury_x_elo, home_venue_roll_10_points, home_adj_attack_10,
away_points_to_cl_zone, away_roll_5_corners, h2h_btts_rate,
home_travel_fatigue, away_ht_lead_hold, h2h_goals_diff,
rolling_gd_diff, h2h_away_goals_avg, _has_gk_data, _has_shot_data, _has_odds
```

---

## 4. DATA FLOW — WHERE EVERYTHING LIVES

### Input Data
```
data/external/          ← Raw scraped data (Sofascore, FBref, etc.)
data/parsed/            ← Cleaned parquet files (matches, players, shots)
data/external/sofascore/fixtures_2025_2026.json  ← Fixture IDs for current season
data/lineup_history/substitutions.json  ← Backfilled sub data (3246 matches)
```

### Feature Table
```
data/features/features.parquet  ← THE feature table (7829 × 969)
  - 21 seasons (2005-2026)
  - 49 feature plugins
  - After cleanup: constants, >95% null, exact duplicates removed
```

### Models
```
data/models/universal/catboost_no_odds.cbm  ← PRODUCTION MODEL
data/models/universal/catboost_latest.cbm   ← Latest retrain (NOT production)
data/models/universal/training_report.json  ← Full training results
data/models/deployment_state.json           ← Current deployment info
```

### Predictions (refreshed each pipeline run)
```
data/upcoming/predictions.json        ← Main ensemble predictions (1X2 + markets)
data/upcoming/ml_predictions.json     ← Raw ML classifier output
data/upcoming/lineup_predictions.json ← Lineup-adjusted predictions
data/upcoming/odds.json               ← Current odds
data/upcoming/odds_full.json          ← Full odds with all bookmakers
data/upcoming/confirmed_lineups.json  ← Confirmed starting XIs (from Sofascore)
```

### Betting
```
data/upcoming/unified_bet_slip.json   ← Current bet recommendations
data/betting/bet_journal.json         ← Master bet history + P&L
data/betting/bankroll.json            ← Current balance
data/betting/fair_odds_ledger.json    ← For LiveBiasCorrector (currently empty)
```

---

## 5. HOW TO RUN THINGS

### Full Pipeline (all predictions)
```bash
PYTHONPATH=. python3 scripts/pipeline/run_full_pipeline.py
```

### Pre-kickoff Mode (lineup-adjusted predictions)
```bash
PYTHONPATH=. python3 scripts/pipeline/run_full_pipeline.py --pre-kickoff
```

### Train Model
```bash
# CORRECT way — production pipeline:
PYTHONPATH=. python3 -c "from ml.training import train_optimized; train_optimized(exclude_odds=True, n_tune_trials=20)"

# WRONG — don't use:
# train_universal(model_types=['catboost_no_odds'])  ← doesn't exist
# train_universal(model_types=['catboost'])  ← uses ALL features, overfits
```

### Build Features
```bash
PYTHONPATH=. python3 -c "from features.build import build_features; build_features()"
```

### Fetch Confirmed Lineups
```bash
PYTHONPATH=. python3 -c "from scraper.lineup_fetcher import fetch_and_save_lineups; fetch_and_save_lineups()"
```

### Web Interface
```bash
PYTHONPATH=. python3 web/app.py
# → http://localhost:5000/predictions
```

---

## 6. KNOWN LIMITATIONS & DEAD ENDS

### Don't Retry
| What | Why | Evidence |
|------|-----|----------|
| Substitution features | Don't survive feature selection at 41.4% coverage | Feb 20 retrain: 0/6 selected |
| XGBoost for 1X2 | 40% accuracy (barely above random 33%) | Feb 20 training report |
| LightGBM for 1X2 | 43% accuracy | Feb 20 training report |
| Post-hoc calibration (Platt/isotonic) | Worsens ECE on every WF fold | 5/5 folds: identity passthrough was better |
| O/U Under bets | -1% to -14% ROI at all thresholds | Multi-market backtest Feb 17 |
| Asian Handicap bets | -20% to -42% ROI | Multi-market backtest Feb 17 |
| Adding more team-level features | Model ceiling is ~51-52%. More features ≠ better. | 47 features vs 35 features → same accuracy |

### Known Bugs / Gotchas
| Issue | Status | Details |
|-------|--------|---------|
| Sofascore match_id is int, pipeline match_id is string | KNOWN | Bridge required in lineup fetcher |
| `ENSEMBLE_WEIGHTS_WITH_DEEP` can override `ENSEMBLE_WEIGHTS` | KNOWN | Keep in sync |
| `normalize_team()` must be called in every scraper | KNOWN | Missing it causes silent join failures |
| Pinnacle closing odds (PSC prefix) must be derived from opening | KNOWN | Not direct API field |
| FOOTBALLDATA_KEY not set | TODO | Register at football-data.org, add to .env |
| `fair_odds_ledger.json` is empty | EXPECTED | LiveBiasCorrector activates after 30+ settled predictions |

---

## 7. ENSEMBLE DEEP DIVE

### How the 5 Predictors Work
1. **ML Classifier** (60.5%): CatBoost trained on 35 features (Elo, form, H2H, injuries). No odds. Temperature T=0.40 sharpens soft probabilities.
2. **Market Odds** (20.5%): Pinnacle implied probabilities with margin removed. Most accurate single predictor, but using too much copies the market.
3. **xG Poisson** (12.4%): Expected goals → Poisson distribution → P(home goals), P(away goals) → 1X2. Uses team-level xG.
4. **Factor Model** (3.5%): Home advantage + team strength ratings. Simplest predictor.
5. **Player xG** (3.2%): Per-player xG rates × confirmed lineup. Boosts to 12% when confirmed lineups available.

### Blending
```
final_prob = Σ(weight_i × predictor_i_prob)
→ draw_boost (1.28x on draw probability)
→ post_temperature (T=1.08, slight softening)
→ renormalize to sum=1.0
```

### Lineup Impact
When confirmed lineups are available (T-60min from Sofascore):
- Player xG weight: 3.2% → 12%
- O/U xG blend: team_xg only → 60% team + 40% lineup_xg
- Cards/corners: use per-player rates
- This was BROKEN before Feb 20 (scheduler had wrong import). Now FIXED.

---

## 8. TRAINING PIPELINE DEEP DIVE

### `train_optimized(exclude_odds=True)` — The Production Pipeline
```
Step 1: Load features.parquet (7829 × 969)
Step 2: Feature selection (walk-forward importance on FIRST fold only)
        → Top-K by importance → correlation pruning (r > 0.70)
        → 969 → ~35-47 features
Step 3: Optuna tuning (20 trials × 3 models)
        → CatBoost, LightGBM, XGBoost each get 20 Optuna trials
        → Objective: minimize walk-forward log loss
Step 4: Walk-forward CV (16 folds, expanding window)
        → min_train_seasons=5, 1 season per test fold
        → Fold 0: train 2005-2010, test 2010-2011
        → Fold 15: train 2005-2025, test 2025-2026
Step 5: Final fit on all data, save models
Step 6: Report to training_report.json
```

### Key Numbers
- Walk-forward CV: 16 folds (21 seasons, min 5 training)
- 380 matches per season (20 teams × 38 matchdays / 2)
- Current season (2025-2026): 229 matches played through ~MW24
- Full training run: ~5-6 hours on M-series Mac (3 models × 20 Optuna trials + 16 WF folds)

---

## 9. SITUATIONAL EDGE EXPLOITATION

### How It Works
The betting system dynamically adjusts min_edge thresholds based on match context tags. Tags are computed in `_get_situational_tags()` using data from the prediction dict's `situational_context` (injected by `ensemble_prediction_engine.py` from features.parquet).

### Data Flow
```
features.parquet → match_features dict → ensemble predict() → pred["situational_context"]
    ↓                                                                    ↓
home_rest_days, away_rest_days,                        _get_situational_tags(pred)
rest_advantage, home_post_intl_break,                          ↓
away_post_intl_break, congestion_asymmetry            [tags: "rest_advantage", "promoted_home", ...]
                                                               ↓
                                                    _apply_situational_adjustment()
                                                               ↓
                                                    min_edge 6% → 5% (or 6% → 9%)
```

### Tags Implemented
| Tag | Trigger | Source |
|-----|---------|--------|
| `derby` | `"derby" in neutral_factors` | identify_all_factors() |
| `midweek` | Match day is Tue/Wed/Thu | Match date |
| `promoted_home` | Home team in `_PROMOTED_TEAMS[season]` | Hardcoded per season |
| `promoted_away` | Away team in `_PROMOTED_TEAMS[season]` | Hardcoded per season |
| `rest_advantage` | `rest_advantage >= +2 days` | features.parquet |
| `rest_disadvantage` | `rest_advantage <= -2 days` | features.parquet |
| `post_intl` | Either team returning from intl break (12+ day gap) | features.parquet |
| `mgr_change` | `"mgr_change" in home_factors/away_factors/neutral_factors` | identify_all_factors() |

### Adjustments (from 760-match multi-market backtest)
| Situation | Market | Adj | Effect | Backtest ROI |
|-----------|--------|-----|--------|-------------|
| Derby | 1X2 Draw | -2.0pp | Threshold 6%→4% | +96% (7 bets) |
| Promoted away | 1X2 Draw | -1.5pp | Threshold 6%→4.5% | +28% (64 bets) |
| Manager change | 1X2 Draw | -1.5pp | Threshold 6%→4.5% | +87-134% |
| Rest advantage | 1X2 Draw | -1.0pp | Threshold 6%→5% | +42% (33 bets) |
| Post-intl break | O/U Over | -1.0pp | Threshold 6%→5% | +39% (28 bets) |
| Rest disadvantage | O/U Over | -1.0pp | Threshold 6%→5% | +37% (42 bets) |
| Promoted home | O/U Over | -1.0pp | Threshold 6%→5% | +31% (49 bets) |
| Derby | O/U Over | +3.0pp | Threshold 6%→9% | -56% (30 bets) |
| Midweek | O/U Over | +2.0pp | Threshold 6%→8% | -25% (19 bets) |

### Key Files
- `scripts/prediction/ensemble_prediction_engine.py:~2524` — Injects `situational_context` into prediction result
- `scripts/betting/betting_unified.py:705` — `_get_situational_tags()` computes tags
- `scripts/betting/betting_unified.py:758` — `_apply_situational_adjustment()` adjusts thresholds
- `scripts/betting/betting_unified.py:200` — `situational_edge_adjustments` config dict
- `scripts/analysis/backtest_multimarket.py:273` — `tag_match()` backtest equivalent

## 10. META-LEARNER EXPERIMENT (NEGATIVE RESULT — DO NOT REDEPLOY)

### What Was Tested
A CatBoost second-stage model (stacking/meta-learner) that replaces fixed ensemble weights with context-dependent blending. Walk-forward CV with 8 folds (2018-2026).

### Architecture
```
features.parquet → 5 component predictors generate H/D/A probs
                   ↓
                   12 probabilities + 4 agreement features + 8 context features = ~27 meta-features
                   ↓
                   CatBoost classifier (depth=3, l2=10, iter=300, lr=0.05)
                   ↓
                   Learned H/D/A probabilities
```

### Results (2026-02-20)
| Metric | Fixed Weights | Meta-Learner | Delta |
|--------|--------------|--------------|-------|
| Accuracy | 53.8% | 53.3% | -0.49pp |
| Log-loss | 0.972 | 0.978 | +0.006 (worse) |

**Verdict:** Fixed weights + situational edge rules outperform learned meta-learner at current data scale (~3K matches).

### Why It Failed
1. ~3,000 training matches too few for second-stage model to beat Optuna-optimized fixed weights
2. Situational rules (Section 9) already capture context-dependent value more robustly
3. Aggressive regularization prevents overfitting but also prevents learning useful context patterns

### Key File
- `ml/meta_learner.py` — Full implementation, runnable via `python -m ml.meta_learner --evaluate`
- `data/models/universal/meta_learner_report.json` — Saved evaluation results

### What NOT to Retry
- CatBoost/XGBoost/LightGBM stacking with <5K matches
- Neural net meta-learner (even worse sample efficiency)
- Adding more meta-features (regularization already at limit)

### What Could Work Later
- Linear meta-learner (logistic regression, fewer parameters)
- Per-situation weight lookup table (manually tuned from situation breakdown)
- Revisit when dataset reaches 5K+ matches (multi-league expansion)

## 11. SCRAPER PERFORMANCE — KNOWN FIXES

### Injury Scraper (`scraper/injuries.py`)
- **Concurrent TM fetching:** `ThreadPoolExecutor(max_workers=3)` with `_RateLimiter(2.5s)`
- **ESPN as deferred fallback:** Only tried for teams where TM returned 0 results (was tried for all 20)
- **Performance:** 49s full scrape (20 teams, 60 injuries), <1s when today's cache exists
- **Rate limits:** 2.5s between TM requests is safe. No 403/429 errors observed.

### football-data.org Lineup Fetcher (`scraper/footballdata_lineups.py`)
- **Critical fix:** Early return when no matches are imminent. Without this, the code fetched lineup details for ALL 129 scheduled matches at 6.5s each (~14 min blocking).
- **Root cause:** `if imminent_teams and ...` short-circuits when `imminent_teams` is empty set.
- **Impact:** Pipeline Group B: 1.0s (was 14+ min causing full pipeline hang)

### Pipeline Group Timing (typical)
```
Group A (odds chain):     ~136s  (19 events × extra markets — legitimate)
Group B (squads/lineups/injuries): ~1s  (all cached or early-return)
Group C (cross-market):   ~120s  (waits for Group A, then analysis)
Total pipeline:           ~10 min
```

## 12. KNOWN BUG FIXES (Feb 20 2026)

### Referee Import (FIXED)
- `run_full_pipeline.py:1263`: `get_referee_assignments` → `load_referee_assignments`
- Impact: zero — CatBoost model has 0 referee features (all pruned in feature selection)

### Promoted Teams (FIXED)
- `creative_factors.py:52`: `{"Bari", "Catanzaro", "Cesena"}` → `{"Sassuolo", "Pisa", "Cremonese"}`
- Impact: corrects `home_is_promoted`/`away_is_promoted` features + situational betting tags

### Backtest Validation (Feb 20 2026)
Post-fix backtest (760 matches, 2023-2025) confirmed no regressions:
- 1X2 Draw at 6% edge: **+18.0% ROI** (469 bets)
- OU Over blended_40 at 5%: **+25.2% ROI** (51 bets)
- All 9 situational tags show positive ROI (derby draws +54%, mgr_change draws +108%, rest_advantage draws +53%)
- Pipeline completes 33 steps in ~10 min without hanging
