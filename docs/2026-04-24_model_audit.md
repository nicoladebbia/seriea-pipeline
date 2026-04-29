## 2026-04-24 — Honest audit of the model + backtest pipeline

Generated after 4 walk-forward backtests showed the "+3.58% ROI" claim was
leakage-inflated. Real held-out (2024-25) performance is **negative across
every market that could be validated**. This document is the inventory of
what is actually broken and what would unblock each item.

It is NOT a plan — it is a finding.

---

### 1. Verified held-out performance (2024-25 Serie A, walk-forward)

| Market | Threshold | Bets | ROI point | ROI CI |
|---|---|---:|---:|---|
| O/U 2.5 | 0% edge | 380 | **-4.86%** | [-15.2, +6.4] |
| O/U 2.5 | 3% edge | 297 | -6.36% | [-19.1, +6.2] |
| O/U 2.5 | 5% edge | 243 | -7.02% | [-20.9, +7.5] |
| O/U 2.5 | 7% edge | 205 | -9.31% | [-24.6, +5.7] |
| O/U 2.5 | 10% edge | 156 | **-12.41%** | [-30.9, +5.1] |

Observation: higher edge threshold → worse ROI. This is the signature of
miscalibration: the model's confident picks are systematically wrong.

### 2. Broken: `data/models/markets/prod_1x2.cbm`

- Trained: 2026-02-11 with 456-feature vocabulary
- Current `features_serie_a.parquet` provides only 320 of those 456 features
- **136 features are missing (~30% of input vector)**
- `_CatboostWrapper.align` zero-fills missing features → probabilities corrupted
- Result: 0 bets meet edge threshold in walk-forward (corrupted probs never
  show real edge vs closing line)

Missing feature categories:
- **Renamed** (fixable by rename map): `odds_PSCH` → `odds_PS_close_H`,
  `ref_matches` → `ref_matches_officiated`, `ref_avg_yellows_given` →
  `ref_avg_yellows`
- **Deleted from feature pipeline** (fixable only by rebuilding features):
  `home_ppda`, `home_injuries_count`, `home_comeback_rate`, `capacity_ratio`,
  `home_ht_lead_hold`, `home_losing_at_ht_rate`, `home_ht_scoring_pct_roll_5`,
  `pressing_mismatch`, and 120+ others.

### 3. Blocked: `prod_btts.cbm`, `prod_corners_over_*.cbm`, `prod_cards_over_*.cbm`

- Models exist, load successfully.
- **Closing odds for these markets are not in `features_serie_a.parquet`.**
  - BTTS: no `odds_*_btts_yes` columns
  - Corners: no `odds_*_corners_over_*` columns
  - Cards: no `odds_*_cards_over_*` columns
- Harness requires closing odds to compute edge and simulate bets → skips
  these markets entirely.
- Unblock path: backfill via Odds API historical endpoint. Script exists:
  `scripts/betting/backfill_historical_odds.py` (previously used for 1X2 + O/U).
  Needs extension to additional markets.

### 4. Blocked: simulator λ-source fix (Stage 1.1)

- Proposed fix: replace simulator's `PoissonRegressor` with existing
  `data/models/universal/xg_home.cbm` and `xg_away.cbm`.
- **Those xG regressors were trained 2026-04-18 on data including
  2022-23, 2023-24, 2024-25.** These are the walk-forward eval seasons.
- Using them in a backtest = train-on-test leakage. ROI would be inflated.
- Unblock path: retrain xG regressors walk-forward. Each fold's model uses
  only data strictly before that fold's eval season. No existing script
  does this — would need `scripts/models/train_xg_walkforward.py`.
- Additional concern: current xG models use odds columns as input features
  (`odds_AvgA`, `odds_B365H`, etc.). This violates the "no live odds API"
  architectural constraint.

### 5. Dual training pipelines

Two independent trainers produce market models with different feature
vocabularies and different conventions:

| Aspect | `data/models/markets/prod_*.cbm` | `data/models/universal/over_under/` |
|---|---|---|
| Trainer | unknown / legacy | `scripts/models/train_over_under.py` |
| Trained at | 2026-02-11 | Apr 2026 |
| Features | 456 (Strategy B, 2017+ data) | current vocabulary, walk-forward |
| Leakage protection | train=2017-2023, val=2023-24, test=2024-25 | walk-forward CV |
| Output path | `data/models/markets/prod_*.cbm` | `data/models/universal/over_under/` |

The production `EnsemblePredictor` at
`scripts/prediction/ensemble_prediction_engine.py` loads from a mix of
paths. It is unclear which trainer is "canonical."

### 6. Inflated claims in source comments

- `scripts/prediction/ensemble_prediction_engine.py:149`: "Production
  acc=69.3% on 2025-2026"
- `docs/baselines.md`: "Accuracy 60.0% walk-forward CV, 69.3% production"
- `catboost_no_odds_metadata.json`: top-level `metrics.accuracy: 0.9667`
  (in-sample, clearly overfit), while `cv_summary.all_folds_accuracy: 0.4902`
  (real walk-forward)

The 60-69% numbers do not correspond to any reproducible walk-forward run
this repo can produce today. The real walk-forward numbers are in the
47-55% range depending on the model.

### 7. Accuracy ceiling — what's physically achievable

| Model class | Realistic walk-forward accuracy on 1X2 |
|---|---:|
| Pinnacle closing line | 53-55% |
| Academic SOTA (Dixon-Coles state-space) | 52-56% |
| Your xG-Poisson standalone | 54.3% |
| Your CatBoost/XGB/LightGBM ensemble | 47-49% |

**60-70% accuracy on 1X2 is not achievable.** Football is irreducibly
variable. Any claim above 56% on 1X2 walk-forward is leakage, overfitting,
or an easier sub-problem (e.g., "predict home win vs not-home-win" is ~75%
accuracy because most matches ARE home wins — but doesn't make money).

### 8. What IS within reach

- Per-market calibration error (ECE) < 0.05 on binary markets (O/U, BTTS).
  Current `lightgbm` ECE is 0.008, so this is achievable.
- Matching Pinnacle closing-line log-loss on 3-5 markets. Difficult but
  not impossible on O/U 2.5, BTTS, DC, DNB.
- Per-market Kelly ROI of +3% to +7% over a season (with variance).
- Portfolio assembled across 2 leagues × 5 markets = 10 bet streams for
  diversification.

### 9. What would need to happen to trust a future backtest

Minimum requirements:

1. Every model trained walk-forward (fold's model uses only data strictly
   before fold's eval season).
2. No odds-as-features in the predictor (or accept odds as a market anchor
   only, never as training signal).
3. Feature vocabulary consistent between training and inference (no
   zero-fill for 30% of inputs).
4. Closing odds available for every market being backtested.
5. Approval criterion: walk-forward ROI lower-CI > 0% at some edge
   threshold with ≥300 bets. Not point estimate > 0.
6. Out-of-sample confirmation: backtest-approved markets paper-traded for
   ≥100 live matches before real money.
