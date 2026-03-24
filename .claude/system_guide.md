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
| CatBoost no-odds + Ensemble | `catboost_no_odds.cbm` + `ensemble/` | 35 | 61.55% CV / 69.3% prod | **THE** ML component (2017+ data, time-decay 0.85) |

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
| Adding more team-level features | Old ceiling was ~51-52% with all-data training. New ceiling ~61%+ with 2017+ data + time-decay. | 2017+ data unlocked xG, lineup, odds velocity features |

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
1. **ML Classifier** (60.5%): CatBoost no-odds model (35 features, 2017+ data, time-decay 0.85). CV acc=0.6155, ll=0.8589. Key features: xG diff, lineup rating, odds velocity, squad value, Elo, form, H2H. Temperature T=0.75.
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

---

## 13. RESEARCH FRONTIER — OPEN PROBLEMS (Read this BEFORE deciding what to work on)

> **Philosophy:** Sections 6 and 10 document what failed *within the current paradigm*.
> This section documents what hasn't been tried yet — paths that require **changing the
> problem formulation**, not tuning harder within it. The old 51% accuracy ceiling was
> broken by switching to 2017+ data + time-decay (now 61.55% CV). The new ceiling
> may be pushed further with multi-league data, deeper player models, or live odds timing.

---

### 13.1 KNOWN BUGS ACTIVELY HURTING PERFORMANCE (Fix these first)

These are not research — they are documented issues that are degrading current results.

#### A. Feature Selection Fold-0 Bias (KB #37 — UNRESOLVED)
- **Bug:** `ml/feature_selection.py` runs importance ranking on fold-0 only (train 2005-2010, test 2010-2011)
- **Impact:** Features from Sofascore (2022+, 271 cols at 98% recent coverage) and FBref advanced (2017+, 99 cols) get zero importance because they don't exist in fold-0's training window. They are then pruned before any recent fold sees them.
- **Fix:** Average importance across the last 4-6 folds (2020+), or use a union of top-K features from each fold. This could recover 50-100 modern features that are currently invisible to the model.
- **Expected impact:** Moderate. Modern features may not all survive correlation pruning, but some likely carry signal the 2005-era features can't.

#### B. Walk-Forward Backtest Leakage
- **Bug:** Production CatBoost is trained on ALL data (2005-2026), but the multi-market backtest (Feb 17, 760 matches) reports ROI as if it used walk-forward retraining per fold.
- **Impact:** Reported ROI numbers (+25.2% O/U Over, +18% 1X2 Draw) are overstated by an unknown amount (estimated 2-5pp).
- **Fix:** Implement proper per-fold retraining in `backtest_multimarket.py`. Each test fold should use only a model trained on data up to that fold's cutoff.
- **Priority:** HIGH — without this, all ROI claims are suspect.

#### C. Deployment State Uncertainty
- **RESOLVED (KB#19):** Two separate model pipelines exist: (1) catboost_no_odds.cbm with its own metadata (catboost_no_odds_metadata.json), and (2) the ensemble (training_report.json). Feature counts differ by design. `deployment_state.json` now tracks the active production model. `check_model_metadata_consistency()` in health_check.py validates alignment at runtime.
- **Fix:** Verify which model `ensemble_prediction_engine.py` actually loads. Check if `deployment_state.json` weights match `ENSEMBLE_WEIGHTS` constant (KB #19 flagged this, still unresolved).

---

### 13.2 CHANGE THE OBJECTIVE FUNCTION

The system optimizes for **1X2 classification accuracy**. But money is made through calibrated probability estimates in specific market contexts. Accuracy is the wrong target.

#### A. Train for Calibration, Not Accuracy
- **Idea:** Custom CatBoost loss function (or post-training objective) that minimizes ECE or Brier score instead of log-loss/accuracy.
- **Why:** A model at 50% accuracy with perfect calibration makes more money than one at 54% accuracy with poor calibration. Kelly criterion sizing depends entirely on probability quality.
- **What to try:** Train with `Logloss` (current) vs `MultiClass` with custom eval metric = Brier score. Compare betting ROI on walk-forward folds, not accuracy.

#### B. Train for Closing Line Value (CLV)
- **Idea:** Instead of predicting H/D/A outcomes, predict where the market misprices. Target = `(model_fair_odds / closing_odds) - 1`. Positive = model was right to bet, negative = model was wrong.
- **Why:** CLV is the single best predictor of long-term profitability. A model that consistently beats the closing line prints money regardless of individual match accuracy.
- **Data available:** `fair_odds_ledger.json` has 1,261 predictions with fair odds + outcomes. `clv_history.json` has per-bet CLV data. Growing daily.
- **Prerequisite:** Need 500+ settled CLV observations per market type. May already have enough for O/U.

#### C. Situation-Weighted Loss
- **Idea:** Weight training samples by how often they appear in profitable betting situations. Derby matches, promoted teams, manager changes — these are where ROI lives. The model should get these right even at the cost of getting "normal" matches slightly wrong.
- **Implementation:** In `train_optimized()`, pass sample weights to CatBoost's `Pool()` based on situational tag membership. Tags with positive backtest ROI get weight > 1.0.

---

### 13.3 SITUATION-SPECIFIC MODELS

The current system uses one universal model + post-hoc situational adjustments. But subpopulations behave fundamentally differently.

#### A. Ensemble of Specialist Models
- **Idea:** Train separate CatBoost models for identified subpopulations: derbies, promoted teams, post-international break, manager change matches. Use the specialist when the situation matches, universal otherwise.
- **Why:** Your situational tags already prove these subpopulations have different base rates (derby draws +96% ROI = market systematically misprices them). A model trained *only* on derbies could learn patterns invisible to the universal model.
- **Risk:** Small sample sizes per situation (derbies: ~7 per season, ~150 across 21 seasons). May need to pool across leagues.
- **Minimum viable test:** Derby specialist — pool all derbies from 21 seasons, train CatBoost, compare calibration vs universal model on derby subset.

#### B. Regime-Aware Predictions
- **Idea:** Detect "market regime" (volatile day, stable day, end-of-season, opening matchweek) and adjust ensemble weights dynamically rather than using fixed weights.
- **Why:** The meta-learner failed because it tried to learn this from data with too few samples. But a **rule-based regime detector** with manually tuned weight tables could work — similar to how situational edge adjustments work but applied to ensemble blending, not just edge thresholds.
- **Implementation:** Add `_get_market_regime()` to ensemble engine. Map regime → weight overrides (e.g., early season → boost factor model weight, late season → boost ML weight).

---

### 13.4 UNLOCK UNDERUTILIZED MARKET SIGNALS

The system collects rich market microstructure data but uses it only for multiplicative confidence adjustments (0.3x-1.15x on Kelly stake). These signals could be much more powerful.

#### A. Promote Odds Velocity Features to ODDS_META_KEEP
- **Current state:** 14 `line_vel_*` features are in `ODDS_COLUMN_PATTERNS` (excluded) but NOT in `ODDS_META_KEEP` (included). They capture opening→closing movement magnitude and direction.
- **Fix:** Add `line_vel_pin_home`, `line_vel_pin_draw`, `line_vel_pin_away`, `steam_move_flag` to `ODDS_META_KEEP` in `ml/config.py`.
- **Expected impact:** These features tell the model WHERE sharps are moving money. CatBoost can learn non-linear interactions (e.g., "steam move toward home + large Elo gap = value on away draw") that the current linear confidence adjustments can't express.
- **Effort:** 15 minutes to add to config, then retrain.

#### B. Market Microstructure as Edge Modifier
- **Idea:** Replace the flat multiplicative adjustments (0.70x-1.15x) with a learned function of market state. Input: (raw_edge, steam_direction, sharp_soft_divergence, odds_consistency, overround, bookmaker_count). Output: adjusted edge or confidence interval.
- **Why:** A 6% edge in a market with 20 bookmakers, low overround, and sharps agreeing with you is much more reliable than a 6% edge in a thin market with 3 bookmakers and sharps moving against you. Currently both get similar treatment.
- **Implementation:** Could be as simple as a decision tree fitted on CLV outcomes, or as complex as a small calibration model.

#### C. Bookmaker-Specific CLV Analysis
- **Data available:** `clv_history.json` records which bookmaker gave the best price for each bet.
- **Question:** Do certain bookmakers (SNAI, Lottomatica, bet365) systematically offer better CLV for certain market types? If yes, route bets to those books.
- **Effort:** Pure analysis — no model changes needed. Run a groupby on clv_history by bookmaker × market_type.

---

### 13.5 SCALE THE DATA, NOT THE MODEL

The meta-learner failed because 3K matches wasn't enough. Several ideas from earlier sections become viable with more data.

#### A. Multi-League Expansion (Documented in ROADMAP — Not Started)
- **Infrastructure exists:** `config/settings.py` has LEAGUES dict ready for Premier League, La Liga, Bundesliga, Ligue 1, Eredivisie.
- **Impact:** 5 leagues × 380 matches/season × 21 seasons = ~40K matches vs current 7.8K. Meta-learner becomes viable. Situation-specific models become viable. Feature selection fold-0 bias disappears (enough data in every fold).
- **Risk:** League-specific dynamics (Serie A home advantage ≠ EPL). Need league indicator features or separate per-league models.
- **Effort:** 3-4 week build. Scrapers need adaptation. Feature plugins should generalize.
- **Starting point:** Premier League has the most available data. FBref + Sofascore coverage is excellent.

#### B. Synthetic Data Augmentation
- **Idea:** Use the xG Poisson model to simulate 10-100 possible match outcomes for each historical match. Train on simulated outcomes weighted by their probability.
- **Why:** Converts 7.8K matches into 78K-780K training samples. Addresses the fundamental sample size constraint.
- **Risk:** Simulated outcomes carry the biases of the xG model. Could amplify systematic errors.

---

### 13.6 POST-PREDICTION EDGE (Live & Settlement)

Current system makes predictions pre-match and holds. There are opportunities after prediction.

#### A. Live Odds Monitoring for Entry Timing
- **Idea:** Don't bet immediately when the pipeline runs. Instead, set target odds and wait for market movement. If odds drift in your favor (lengthening), bet at better value. If odds shorten past your target, skip.
- **Why:** CLV data shows some bets placed early get worse value than waiting. Entry timing matters.
- **Implementation:** `scheduler.py` already has time-based triggers. Add an odds alert trigger: "bet X if odds reach Y before kickoff."

#### B. Activate the StaticCorrector
- **Current state:** `correction_layer.py` has a working logistic regression corrector. It's NOT deployed because it was waiting for 30+ settled predictions in `fair_odds_ledger.json`. The ledger now has 1,261 entries.
- **Action:** Check if the activation condition is met and deploy. Even small ECE improvements compound across hundreds of bets.

---

### 13.7 HOW TO DOCUMENT NEW EXPERIMENTS

When exploring any frontier path above:

1. **Before starting:** State the hypothesis, metric to evaluate, and success threshold
2. **After completing:** Add results to `docs/knowledge_base.jsonl` with severity and status
3. **If positive:** Update this section to move the path from "frontier" to "proven"
4. **If negative:** Add to Section 6 ("Don't Retry") with evidence and the specific conditions under which it failed — but note what *variant* might still work
5. **Key principle:** A negative result that narrows the search space is still progress. Document why it failed, not just that it failed
