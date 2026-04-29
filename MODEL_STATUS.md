# Model Status — How To Read Performance Live

**Read numbers from metadata, not from this file or any other markdown.**

The honest source of truth for model performance is:

```
data/models/universal/catboost_no_odds_metadata.json
```

This file is written by `scripts/models/retrain_no_odds_catboost.py` whenever the production model retrains, and is never edited by hand. Markdown numbers go stale within hours of a retrain — JSON metadata does not.

## How to answer "how is the model performing right now?"

Run the print script:

```bash
python3 scripts/diagnostics/print_model_status.py
```

Or the one-liner if the script is unavailable:

```bash
python3 -c "import json; d=json.load(open('data/models/universal/catboost_no_odds_metadata.json')); s=d['cv_summary']; print(f\"Last 3 folds: {s['last3_accuracy']:.1%} acc, {s['last3_logloss']:.4f} logloss, brier {s['last3_brier']:.4f}\"); print(f\"All folds: {s['all_folds_accuracy']:.1%} acc, {s['all_folds_logloss']:.4f} logloss\"); print(f\"Saved at: {d.get('saved_at','unknown')}\")"
```

## What the metadata fields mean

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

## What the realistic ceiling actually is

| Class | Walk-forward 1X2 accuracy ceiling |
|---|---:|
| Pinnacle closing line (global SOTA) | 53–55% |
| Academic Dixon-Coles state-space | 52–56% |
| This project's xG-Poisson standalone | ~54% |
| This project's CatBoost no-odds (current) | 50–52% |

**Anything above ~56% on 1X2 is leakage, overfitting, or a different sub-problem.** If you see a markdown file claiming 60%, 69%, or 72% — it is fiction.

## Rules going forward

1. **Never quote a model performance number from markdown.** Read the metadata.
2. **Never hard-code an accuracy threshold in production code.** If a gate is needed, read it from a config or compute it from the metadata.
3. **When the model retrains, the metadata updates automatically.** No manual file edits.
4. **If a doc disagrees with the metadata, the doc is wrong.** Fix or delete the doc.
