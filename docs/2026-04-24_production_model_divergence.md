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

## Update — 2026-04-24 21:49 UTC: System B retrained under walkforward

Open question #4 is resolved by action. `scripts/models/retrain_no_odds_catboost.py`
gained five flags (`--walkforward-final`, `--include-new-features`, `--n-seeds`,
`--fit-calibrator`, `--variant-suffix`) and was run with all of them. The
phase5_v1 artifact saves the **last-fold model** (trained strictly before the
held-out final season — no in-sample contamination on the most recent season)
and a calibrator fit on the true held-out fold (ECE 0.043 vs prior in-sample
0.31). Multi-seed mandate satisfied: 3 seeds tried, selected by median last-3
log-loss.

**Promoted to production 2026-04-24 21:49.** The previous model is archived at
`data/models/universal/_archive_20260424T214905/`.

CV-vs-CV comparison (the only honest one):

| Metric (last-3 folds) | Prior prod (Apr 18) | Phase 5 v1 |
|---|---:|---:|
| Log-loss | 1.0066 | 1.0040 |
| Accuracy | 49.65% | 51.38% |
| Held-out ECE | 0.31 (in-sample) | 0.043 (clean) |

Δlog-loss is below the methodology mandate threshold (−0.01), so the swap
is **not** a measurable accuracy upgrade — it's a methodology / calibration
upgrade. Live Kelly sizing should be more honest now that the calibrator
isn't fit on memorized data.

## Remaining open questions

1. **Calibration on System B**: does `lean_calibrators.pkl` help or hurt
   `catboost_no_odds`? Same A/B as walkforward, different model. (Now testable
   with a clean held-out fold.)
2. **Should the project unify on one system?** Two parallel 1X2 trainers
   + two parallel calibrators is duplication. Walkforward is methodologically
   cleaner (per-season holdout with strict no-leakage); catboost_no_odds is
   simpler and is what production actually serves. Unification is a multi-hour
   design decision, not an end-of-session call.
3. **Multi-seed mandate on weekly retrain**: the new flags exist, but
   `scripts/pipeline/weekly_retrain.py` doesn't pass them yet. Next weekly
   retrain will revert to single-seed in-sample contamination unless wired.

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
