# Production / Walkforward Model Divergence — 2026-04-24

**Finding:** The project has TWO parallel 1X2 model systems. They are
trained by different scripts, live in different directories, and are
loaded by different consumers. Today's session improved one. Production
runs the other.

This is the most important finding of the day.

## The two systems

### System A — Walkforward (this session's focus)

- **Training script:** `scripts/models/train_walkforward.py`
- **Model artifacts:** `data/models/walkforward/{league}/1x2/season_{YYYY-YYYY}.cbm`
  + `season_{YYYY-YYYY}_calibrators.pkl`
- **Loaded by:**
  - `scripts/diagnostics/deep_backtest_1x2.py` — diagnostic backtest
  - `scripts/diagnostics/backtest_rules_engine.py` — rules-engine backtest
  - `models/simulator/backtests/walkforward_predictor.py` — simulator harness
  - `models/simulator/backtests/stacked_predictor.py` — parked stacked predictor
- **Loaded by NOTHING in `scripts/prediction/`, `scripts/betting/`, or `scripts/pipeline/`.**
- Produces probabilities the live betting flow does not consume.

### System B — Production (live betting flow)

- **Training script:** `scripts/models/retrain_no_odds_catboost.py` (and
  `train_unified.py`, `train_no_odds.py` variants)
- **Model artifacts:** `data/models/universal/catboost_no_odds.cbm`
  + `data/models/universal/lean_calibrators.pkl`
  + `data/models/universal/draw_detector.cbm`
  + `data/models/universal/catboost_upcoming.cbm` / `_v2.cbm`
- **Loaded by:** `scripts/prediction/ensemble_prediction_engine.py`
  (the actual prediction engine the dashboard, telegram bot, and bet
  flow run through)

`catboost_no_odds.cbm` was last trained 2026-04-18 by the weekly retrain
pipeline. `lean_calibrators.pkl` was last fit 2026-04-09. The live
inference applies isotonic calibration from `lean_calibrators.pkl` after
the `catboost_no_odds` raw probabilities.

## Why this matters

Every quantitative finding in today's session was measured on **System A**:

- Phase 1 walk-forward log-loss numbers
- Phase 1 rules-engine Kelly ROI numbers (SA +9.95% / EPL +7.95% etc.)
- The "+17.5% EPL B+raw" headline (and its retraction)
- The 89% vs 72% CLV+ delta when dropping calibration
- Phase 3a null result (xi_quality features)
- All seed-robustness measurements

**None of these have been re-measured on the production catboost_no_odds
model.** They might or might not transfer. Without a measurement on
System B, today's quantitative findings are diagnostic-only.

## What today's changes DID change about production

The feature pipeline is shared between System A and System B (both train
on `data/features/features_*.parquet`). So today's feature-pipeline edits
DO affect production over time:

- `features/lineup_xg.py` league-aware routing → next time `catboost_no_odds`
  retrains, it will see EPL lineup features (currently leaky-blacklisted in
  walkforward but System B may not blacklist them).
- `features/xi_quality.py` plugin → next retrain, the no-odds model would see
  the 5 SA + 4 EPL XI columns.
- The current production `catboost_no_odds.cbm` (April 18) was trained
  before today's edits and does not see these features. It needs a retrain
  to include them.

Whether to retrain System B with these features is a decision for next
session. Worth doing only after measuring whether the current production
model is well- or poorly-calibrated — same kind of A/B as the walkforward
17pp finding, but on the right model.

## Open questions for next session

1. **Calibration on System B**: does `lean_calibrators.pkl` help or hurt
   `catboost_no_odds`? Same A/B as walkforward, different model.
2. **Should the project unify on one system?** Two parallel 1X2 trainers
   + two parallel calibrators is duplication. Walkforward is methodologically
   cleaner (per-season holdout with strict no-leakage); catboost_no_odds is
   simpler and is what production actually serves. Unification is a multi-hour
   design decision, not an end-of-session call.
3. **Multi-seed mandate on System B**: the methodology mandate (≥3 seeds,
   thresholds above noise floor) applies to System A. System B currently
   trains with one seed. Apply same mandate to weekly_retrain pipeline?
4. **In-sample contamination on System B:** `catboost_no_odds.cbm` (April 18)
   was trained on data through April 18, INCLUDING the 2022-2025 seasons we
   use as eval in System A's walk-forward tests. Any backtest of this model
   on those seasons is in-sample, not walk-forward — its 96.7% in-sample
   accuracy (vs ~49% CV) confirms it saw the data. To measure cal-drop on
   production properly, the model itself needs to be retrained per-season
   under the walkforward protocol. This is really the same question as #2:
   *the production model's training pipeline needs the same walk-forward
   discipline the diagnostic models already have.*

## What this changes about the session's other docs

`docs/2026-04-24_training_window_audit.md` and
`docs/2026-04-24_phase3a_xi_quality.md` now have one-line PRODUCTION
DIVERGENCE NOTICE banners pointing here. Anyone reading those docs
without the banner would correctly assume the metrics applied to
production — and be wrong.

---

**Verdict:** STRONG (this is the day's most important finding)
**Confidence:** high
**If I'm wrong:** I missed a code path that DOES load walkforward in
production. Worth one more 5-min grep tomorrow:
`grep -rn "data/models/walkforward" scripts/{prediction,betting,pipeline}/`
should return nothing. If it returns something, the divergence claim
needs revisiting.
**What I'm NOT saying:** I'm not saying System A work was wasted. The
multi-seed methodology, the leakage-clean walkforward training, the
diagnostic infrastructure (env-var variant testing) are all real and
reusable. I'm saying the *quantitative ROI/CLV findings* from today
don't apply to live betting until measured on System B.
