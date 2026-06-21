# Football Match-Prediction Pipeline

**A leakage-aware, walk-forward ML system that predicts football match outcomes (1X2) from ~1,000 engineered features, blends three gradient-boosting models with a draw-aware weight optimizer, and calibrates the output — then proves it beats the bookmaker baseline on held-out seasons it never trained on.**

The hard part of this domain is not training a classifier. It is **not lying to yourself**. Football outcomes are ~50% home / 27% draw / 23% away, the realistic accuracy ceiling on 1X2 is ~53–55% (Pinnacle's closing line, academic state-space SOTA), and almost every naive pipeline beats that number — by leaking future information, miscalibrating, or quietly feeding the market odds back in as a feature. This project is built end-to-end to make that impossible, and to report numbers that survive contact with reality.

> **Live performance is read from model metadata, never from this file.**
> Markdown goes stale within hours of a retrain; JSON written by the trainer does not.
> ```bash
> python3 scripts/diagnostics/print_model_status.py
> ```
> At time of writing: **walk-forward 1X2 accuracy 51.4% (last-3 seasons), log-loss 1.004, ECE 0.043, 126 selected features.** Those numbers sit *inside* the honest ceiling — see [`MODEL_STATUS.md`](MODEL_STATUS.md). Anything claiming >56% is leakage or fiction, and the repo treats it as a bug.

---

## The problem, precisely

Three failure modes destroy a sports-prediction model and none of them show up in a standard train/test split:

1. **Temporal leakage.** Rolling form, Elo, and head-to-head features computed on a randomly shuffled split let the model see the future. Even an honest season split leaks across the train/test boundary because a 5-match rolling window straddles it.
2. **The draw trap.** Draws are ~27% of outcomes but the hardest class to predict. A log-loss-greedy optimizer learns to *never predict a draw* (it's "safer"), tanking real-world usefulness while looking fine on aggregate loss.
3. **Calibration collapse.** Gradient-boosted trees produce well-*ranked* but over-confident probabilities. If you bet (or just report confidence) off uncalibrated output, you're systematically wrong at the tails — and per-class calibration *before* blending silently destroys the draw probability mass.

Each of these is handled explicitly in code, not hand-waved.

---

## Architecture

```
scrapers (FBref, Football-Data, Understat, Sofascore, Open-Meteo, Transfermarkt)
        │   public HTML/CSV/JSON, with HTML-fallback breakers for banned APIs
        ▼
parser/                  raw HTML/JSON ──► canonical Parquet, cross-source match_id mapping
        ▼
features/  (~37 FeaturePlugin steps, ABC registry)  ──►  features_<league>.parquet  (~1,059 cols)
        │   form · xG trends · Elo · H2H · referee · weather · squad value · market microstructure
        ▼
ml/
  data.py           leakage-aware load: domain-imputation fit on TRAIN rows only, season-split CV
  feature_selection recency-weighted walk-forward importance  ──► 126 features (odds excluded)
  ensemble.py       XGBoost + LightGBM + CatBoost ─► scipy-optimized blend ─► single calibrator
  calibration.py    isotonic vs temperature, auto-pick by ECE
  correction_layer  static (logistic on OOF) + rolling (EMA) bias correction
        ▼
scripts/prediction · scripts/betting · web/  (Flask dashboard)   ── value-bet / Kelly signals
        ▼
scripts/worldcup/   independent Elo + Poisson-GLM model for the 2026 World Cup
```

**Data flow is one-directional and idempotent per stage:** each layer writes Parquet keyed by a deterministic `match_id`, so any stage can be recomputed without re-scraping. Coverage: 5 league configs, 21 seasons (2005-06 → 2025-26), per-league models (deliberately *not* one model across leagues — Serie A and the Premier League have different home-advantage and draw structure).

---

## Key engineering decisions & tradeoffs

**1. Walk-forward CV with a purge gap — chosen over k-fold, accepting smaller training folds.**
`ml/walk_forward.py` uses expanding-window season splits and **drops the last 2 matchweeks (~20 matches) from each training fold** before the test boundary. Reason: a 5-match rolling feature on the first test fixture would otherwise be partly computed from matches that are "in the future" relative to fold N but "in the past" relative to the rolling window. The purge gap is the cheapest correct fix; the cost is fewer training rows per fold, which is the right tradeoff when the alternative is silent leakage.

**2. Blend RAW probabilities, then calibrate ONCE — not calibrate-then-blend.**
The bug this avoids (documented in `ml/ensemble.py`): isotonic regression applied per-model *per-class* distorts the relative class proportions, and the draw mass is what gets crushed. So the ensemble optimizes blend weights on **raw** out-of-fold probabilities and fits a **single** `AutoCalibrator` on the blended output. This preserves each model's class discrimination and calibrates exactly once.

**3. Draw-aware blend objective — not pure log-loss.**
`_optimize_blend_weights` minimizes `(1 - 0.3)·log_loss − 0.3·draw_F1` via `scipy.optimize` with a **softmax parameterization** (weights are positive and sum to 1 by construction, no constrained optimizer needed). The 0.3 draw weight is justified directly from the data — draws are ~27% of outcomes, so the optimizer cannot dump all weight on the lowest-log-loss model (usually CatBoost) without paying for the draws it then misses.

**4. Odds are excluded from the feature set on purpose.**
`feature_selection.exclude_odds` strips every market-derived column before training the production model. Market odds are a near-perfect predictor of the outcome — including them produces a model that *looks* brilliant in backtest and adds **zero independent edge** in production, because at bet time you'd just be predicting the market with the market. The model has to earn its signal from football, not from the bookmaker. Odds re-enter only downstream, as the value-comparison baseline.

**5. Leakage-free fold models for backtesting — a separate, slower artifact.**
`build_fold_models()` trains and persists one CatBoost per walk-forward fold, keyed by its test season. The backtester loads *the model that never saw that season*. This duplicates training work but is the only honest way to backtest a model on data it didn't train on, and the cost is paid offline, once.

---

## Notable implementation details

- **Domain-aware, leakage-aware imputation** (`ml/data.py:_smart_impute`). NaNs are filled by *feature category*, not a global mean: H2H gets uninformative priors (`1/3` win rate, `0` counts), Elo forward-fills within team then falls back to the 1500 league mean, rolling stats get `0.0` (season start = no history), referee/market features get per-season medians. Critically, **all medians are computed on the training rows only** via a `fit_mask` — imputing with test-set statistics is itself a leak, and this is the line most pipelines get wrong.

- **Two-stage bias correction** (`ml/correction_layer.py`). A `StaticCorrector` (logistic regression on 6,500+ OOF predictions + match context) learns *persistent* biases like "underconfident above 60%"; a `RollingCorrector` (EMA of recent errors bucketed by predicted-class × confidence band) adapts week-to-week after settlement. The static layer is trained on the *pre-calibration* OOF probabilities to avoid double-calibrating.

- **Auto-selected calibration** (`ml/calibration.py`). Isotonic regression and single-parameter temperature scaling are both fit, and whichever yields lower ECE on a **chronological** held-out split wins — with a `MIN_CALIBRATION_SAMPLES = 200` guard that falls back to identity rather than calibrate on noise.

- **HTML-fallback scraping with circuit breakers.** When Sofascore's API returns Cloudflare 403s, the scraper parses the `__NEXT_DATA__` JSON embedded in the public HTML, with measured per-page-tier freshness rules (hub pages are fresh ISR renders; match pages are data-free shells) and sentinel checks (`Inter` must appear in Serie A standings, `Arsenal` in the EPL) that trip a breaker on schema drift instead of silently writing garbage.

- **Plugin feature pipeline.** `features/build.py` orchestrates ~37 feature-engineering steps as `FeaturePlugin` (ABC) subclasses over a shared `FeatureState`, producing a ~1,059-column table. The provenance of every column → step is documented in `DATA_CATALOG.md`.

- **World Cup model with real holdout discipline** (`scripts/worldcup/`). An independent Elo + Poisson-GLM model whose ratings are **leak-free by construction** (each match's expectancy uses only pre-match ratings). The variant was selected on a DEV set (WC 2018 + Euro 2020) while WC 2022 / Euro 2024 / Copa 2024 were kept **untouched as a FINAL holdout** — and it ships only because the backtest shows positive Brier skill over the base rate. Goal-quantity markets are explicitly marked "display-grade" until they clear a skill threshold; they are not dressed up as predictive.

---

## Tech stack

| Layer | Tools |
|---|---|
| Language | Python 3.11+ (strict `mypy`, `ruff` with bandit/bugbear rules) |
| Data | pandas, pyarrow (Parquet), numpy, scipy |
| ML | scikit-learn, xgboost, lightgbm, catboost, optuna |
| Calibration / opt | isotonic + temperature scaling, `scipy.optimize` (Nelder-Mead, softmax-parameterized) |
| Scraping | requests, beautifulsoup4, lxml; optional selenium/botasaurus |
| Serving | Flask dashboard; Click CLI |
| Tests | pytest — **508 test functions across 26 files** |

---

## Running it

```bash
pip install -e .                 # core
pip install -e ".[web]"          # + Flask dashboard
cp .env.example .env             # only the API keys for sources you use are required

# Full prediction pipeline (scrape/odds/predict/dashboard)
python3 scripts/pipeline/run_full_pipeline.py
python3 scripts/pipeline/run_full_pipeline.py --quick   # cached data, no API calls

# ML lifecycle via the `seriea` CLI (entry point cli:main)
seriea features --season 2025-2026
seriea ml train                  # universal models, walk-forward CV
seriea ml optimize --trials 50   # feature selection + Optuna tuning + ensemble
seriea ml backtest               # leakage-free, per-fold models

# Honest model status (reads JSON metadata, not docs)
python3 scripts/diagnostics/print_model_status.py

# Web dashboard
python3 web/app.py               # http://localhost:5001

# Quality gates
python3 -m pytest tests/
ruff check . && mypy .
```

---

## Status

Active personal project, run live on a schedule (15 launchd jobs + the Flask dashboard). **Honest about its ceiling:** the production model sits at ~51% walk-forward 1X2 accuracy — below the ~53–55% market SOTA, exactly where an honest, odds-excluded model should be, and the repo is wired to flag any number that claims otherwise. The 1X2 "who-wins" markets are the only ones treated as bet-grade; goal-quantity and corners/cards markets were backtested, found to add no skill over the base rate, and **removed from every consumer** rather than left in to inflate the feature list.

Two further sources of truth, both generated mechanically (not hand-written narrative):
- [`ARCHITECTURE_MAP.md`](ARCHITECTURE_MAP.md) — per-file map (purpose, imports, liveness, quality grade) for the whole codebase.
- [`DATA_CATALOG.md`](DATA_CATALOG.md) — authoritative per-file / per-column data reference, including NaN rates and fallback matrices.

Private — not for redistribution.
