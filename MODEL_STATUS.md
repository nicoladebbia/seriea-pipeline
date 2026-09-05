# Model Status — How To Read Performance Live

**Read numbers from metadata, not from this file or any other markdown.**

There are two models to ask about, and they are not equally important.

| | Files | Bets money? | Written by |
|---|---|---|---|
| **O/U classifiers (PRIMARY)** | `data/models/universal/over_under/ou_1_5_catboost_metadata.json`, `ou_2_5_catboost_metadata.json` | **Yes** — O/U Over + Alt O/U are the only enabled markets | `scripts/models/train_over_under.py` (weekly via `weekly_retrain` as an auxiliary model — self-gated vs the incumbent, runs whether or not the 1X2 ensemble is promoted — or via the scheduler) |
| 1X2 CatBoost (secondary) | `data/models/universal/catboost_no_odds_metadata.json` | No — feeds dashboard, Telegram, fantacalcio | `scripts/models/retrain_no_odds_catboost.py` |

Neither file is ever edited by hand. Markdown numbers go stale within hours of a retrain — JSON metadata does not.

## How to answer "how is the model performing right now?"

```bash
python3 scripts/diagnostics/print_model_status.py
```

It prints the O/U section first (per line: holdout vs naive, calibration gap, the last
promotion decision, realised CLV/ROI on settled O/U bets from the journal), then the 1X2
section. Quote from that output.

## O/U metadata fields

| Field | Meaning |
|---|---|
| `cv_metrics.overall_log_loss` / `overall_brier` | Walk-forward CV (last 5 season folds). Compare to the naive baseline computed from `base_rate` — the script does this |
| `cv_metrics.overall_calibration_gap` | Mean absolute gap between predicted and realised over-rate per decile bin. Legacy fixed gate was 0.03 |
| `eval_metrics.*` | The final model on the newest 15% of rows (time-ordered holdout) — the slice the promotion gate judges on |
| `promotion.promoted` / `promotion.reason` | The gate decision when this model was trained. `false` never reaches this file: a refused candidate is written to `over_under/candidate/` and `_latest` is untouched, so the live metadata always describes the live model |
| `promotion.holdout.candidate` / `.incumbent` / `.naive_log_loss` | The numbers the decision was made on, same holdout for both |
| `quality_gates.*` | The three legacy fixed checks (log-loss < 0.693, Brier < naive, cal gap < 0.03). Informational since 2026-09-04: promotion is relative to the incumbent, never to a fixed bar the incumbent itself fails |
| `trained_at` / `n_training_rows` / `n_features` | Provenance |

A metadata file **without** a `promotion` block predates the gate: that model was saved unconditionally.

`over_under/prev/` holds the last incumbent that was replaced (retain 1). `over_under/candidate/` holds the last refused candidate. The engine loads only the top-level `ou_*_catboost_metadata.json` files, so both subdirectories are inert.

## 1X2 metadata fields

| Field | Meaning |
|---|---|
| `cv_summary.last3_accuracy` | Walk-forward 1X2 accuracy on the most recent 3 eval seasons |
| `cv_summary.all_folds_accuracy` | Walk-forward 1X2 accuracy averaged across all CV folds |
| `cv_summary.last3_logloss` | Log-loss on the most recent 3 folds (lower better, random = 1.099) |
| `cv_summary.last3_brier` | Brier score (lower better) |
| `metrics.ece` | Expected calibration error on held-out fold (lower better) |
| `metrics.kelly_roi` | Kelly ROI on held-out fold (point estimate, noisy at small N) |
| `n_features` | Feature count in the trained model |
| `saved_at` | When the model was last retrained |

## What the realistic 1X2 ceiling actually is

| Class | Walk-forward 1X2 accuracy ceiling |
|---|---:|
| Pinnacle closing line (global SOTA) | 53–55% |
| Academic Dixon-Coles state-space | 52–56% |
| This project's xG-Poisson standalone | ~54% |
| This project's CatBoost no-odds (current) | 50–52% |

**Anything above ~56% on 1X2 is leakage, overfitting, or a different sub-problem.** If you see a markdown file claiming 60%, 69%, or 72% — it is fiction. The 1X2 side was proven at ceiling on 2026-06-04; the only lever left is edge (price), which is why the O/U section comes first.

## Rules going forward

1. **Never quote a model performance number from markdown.** Read the metadata.
2. **Never hard-code an accuracy threshold in production code.** If a gate is needed, make it relative to the incumbent or read it from config.
3. **When the model retrains, the metadata updates automatically.** No manual file edits.
4. **If a doc disagrees with the metadata, the doc is wrong.** Fix or delete the doc.
5. **A model is not production because the trainer ran.** It is production because it beat the incumbent on the shared holdout — for O/U that decision is in `promotion.reason`.
