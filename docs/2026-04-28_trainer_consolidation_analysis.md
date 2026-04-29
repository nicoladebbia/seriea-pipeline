# Trainer Consolidation — Design Analysis (2026-04-28)

## The problem

Three walk-forward implementations exist in the repo, doing overlapping work:

| # | Path | Purpose | Lines | Loaded by |
|---|---|---|---:|---|
| **A** | `scripts/models/train_walkforward.py` | Per-(league, market) per-season models for diagnostics/backtest | 720 | `scripts/diagnostics/deep_backtest_1x2*.py`; produces `data/models/walkforward/` artifacts |
| **B** | `scripts/models/retrain_no_odds_catboost.py` | Production model retrain (`catboost_no_odds.cbm`) with deployment glue | 681 | `scripts/pipeline/weekly_retrain.py`; writes `data/models/universal/catboost_no_odds.cbm` |
| **C** | `ml/walk_forward.py` | Legacy walk-forward validator with `WalkForwardFold`/`WalkForwardResult` types | ~400 | `scripts/models/train_unified.py`; some legacy paths |

Plus a separate splitter: `TimeSeriesSplitter` (used by B), and an inline season-loop in A. Three different conventions for "same train/eval split."

## What each does

### A: `train_walkforward.py`
- **Inline season-loop** in `walkforward_train_market()`: for each eval season, take prior seasons as train.
- **No calibrator fit** — saves raw probabilities only. (Calibration left to consumer.)
- **Per-market specs** via `MARKETS: dict[str, MarketSpec]`. Each market defines target builder + market type (binary/multiclass).
- **Strong leakage protection** — `LEAKY_COLUMNS` blacklist + correlation-based safety net (refuses training if any feature has |corr| > 0.5 with target).
- **Per-eval-season artifact**: `season_{YYYY-YYYY}.cbm` + metadata. Multiple per-market files.
- **No production deployment** — purely a measurement tool.

### B: `retrain_no_odds_catboost.py`
- **Uses `TimeSeriesSplitter`** from `ml/`.
- **Has all the production-deployment glue:** rejection thresholds, deployment state file updates, archive logic, calibrator fitting, variant suffix support.
- **Hard-coded for 1X2** — no per-market support.
- **Single output:** `catboost_no_odds.cbm` (the live production model).
- **Multi-seed support** added 2026-04-24 (via `--n-seeds`).
- **Calibrator fit** on held-out fold (leakage-safe), produces `lean_calibrators.pkl`.

### C: `ml/walk_forward.py`
- **Library, not a script.** Has reusable `WalkForwardFold`, `WalkForwardResult`, and `walk_forward_validate()` types.
- **Two splitters**: expanding-window (line 53) and sliding-window (line 159).
- **Used by `train_unified.py`** — but train_unified.py's xg_only mode is also a separate code path.

## The ugly part: each script reinvents the same primitives

| Concern | A reimplements as | B reimplements as | C reimplements as |
|---|---|---|---|
| Train/eval season split | inline loop in `walkforward_train_market` | `TimeSeriesSplitter.generate_splits(X["_season"])` | `expanding_window_walk_forward()` |
| Per-fold metrics | `_fit_one_fold` returns dict | `walk_forward_validate` returns list of dicts | `WalkForwardResult` dataclass |
| Calibration | absent | inline `fit_isotonic_calibrators()` | absent |
| Feature selection | `_select_features()` excludes odds + leaky | `load_current_feature_set()` reads deployment state | (uses caller's selection) |
| Leakage detection | correlation-based safety net | none | none |

**Three implementations, three slightly different behaviors, one shared bug surface.** The cache invalidation issue we hit yesterday on Sofascore is the simpler version of this same disease — when there are two ways to do the same thing, they drift apart silently.

## What each does WELL (worth preserving in the merged artifact)

| Strength | From | Why valuable |
|---|---|---|
| Leakage safety net (corr > 0.5 refuses training) | A | The single most important protection in the repo. Catches new feature mistakes automatically. |
| Per-market specs (`MarketSpec`) | A | Clean way to add new betting markets without forking the trainer. |
| Production deployment glue (rejection thresholds, archive, deployment state) | B | The production retrain pipeline depends on this. Can't simplify without breaking weekly retrain. |
| Multi-seed support + median selection | B | Phase 1's "single-seed outliers fool you" lesson is encoded here. |
| Isotonic calibrator on held-out fold | B | The 2026-04-24 honest fix. Live ROI honesty depends on this. |
| `WalkForwardFold` and `WalkForwardResult` dataclasses | C | Cleanest typing of the per-fold output. |
| `--variant-suffix` for paper-trade variants | B | Lets us ship to a side directory for A-B testing without touching prod model. |

## Proposed target architecture

### Single core library: `ml/walkforward_core.py` (NEW)

A pure library — no I/O beyond reading features from a passed-in DataFrame. Owns:

```python
@dataclass
class WalkForwardConfig:
    eval_seasons: list[str]
    min_train_seasons: int = 5
    min_prior_seasons: int = 3
    seeds: list[int] = field(default_factory=lambda: [42])
    fit_calibrator: bool = False
    leakage_corr_threshold: float = 0.5

@dataclass
class FoldResult:
    eval_season: str
    seed: int
    raw_logloss: float
    raw_accuracy: float
    cal_logloss: float | None
    cal_accuracy: float | None
    ece: float
    feature_count: int
    n_train: int
    n_eval: int
    model_bytes: bytes  # serialized .cbm
    calibrator: dict | None  # serialized isotonic

@dataclass
class WalkForwardReport:
    folds: list[FoldResult]
    config: WalkForwardConfig
    aggregate: dict  # mean log-loss, mean acc, etc.

def run_walkforward(
    df: pd.DataFrame,
    target_builder: Callable[[pd.DataFrame], pd.Series],
    feature_selector: Callable[[pd.DataFrame], list[str]],
    config: WalkForwardConfig,
) -> WalkForwardReport:
    """Pure walk-forward CV. Caller owns all I/O."""
```

**Responsibilities of the core:**
- Splitting seasons into train/eval folds.
- Per-fold model fit (CatBoost or whatever, configurable).
- Per-fold metrics (log-loss, accuracy, ECE, Brier).
- Leakage safety net (correlation check).
- Optional calibrator fit on last held-out fold (leakage-safe).
- Multi-seed iteration + median selection.

**Responsibilities the core does NOT have:**
- Reading parquet files from disk (caller's job).
- Writing models or metadata to disk (caller's job).
- Deployment state updates (caller's job).
- Telegram/notification logic.

### Three thin caller scripts

**`scripts/models/train_walkforward.py` (DIAGNOSTIC, simplified):**
- Loads features.parquet.
- Calls `run_walkforward()` with all-markets config.
- Writes per-(league, market, season) artifacts to `data/models/walkforward/`.
- Used by diagnostics/backtest.

**`scripts/models/retrain_production.py` (NEW NAME — replaces `retrain_no_odds_catboost.py`):**
- Loads features.parquet.
- Calls `run_walkforward()` with single-market 1X2 config + multi-seed + calibrator.
- Selects the last-fold model from the report.
- Applies rejection thresholds, archives old model, updates deployment state, writes `catboost_no_odds.cbm` + `lean_calibrators.pkl`.
- Used by weekly retrain.

**`ml/walk_forward.py` (DELETED).** Its types (`WalkForwardFold`, `WalkForwardResult`) become the new core's `FoldResult` / `WalkForwardReport`. `train_unified.py` migrates to the new core or is itself deleted (it has an `xg_only` mode that's only invoked weekly — could be folded into the production trainer).

## Migration plan (5 steps, reversible at each stage)

### Step 1: Build the core library (low risk)
- Write `ml/walkforward_core.py` from scratch with the API above.
- Port the leakage-safety-net logic from `train_walkforward.py`.
- Port the multi-seed selection logic from `retrain_no_odds_catboost.py`.
- Port the calibrator fit from `retrain_no_odds_catboost.py`.
- Add unit tests for each piece (fold splits, leakage check, calibrator fit, multi-seed).
- **Rollback if needed:** delete the new file, no consumer touches it yet.

### Step 2: Migrate `train_walkforward.py` to use the core (medium risk)
- Replace inline season-loop with `run_walkforward(...)` call.
- Keep the per-market spec system, per-(league, market, season) artifact writes.
- Verify diagnostic backtest output matches the prior version (run side-by-side, compare).
- **Rollback:** revert the file. Cores still exists, no other change.

### Step 3: Migrate `retrain_no_odds_catboost.py` to use the core (medium-high risk)
- Replace `walk_forward_validate(...)` call with `run_walkforward(...)`.
- Keep the deployment state, archive, rejection threshold logic outside the core.
- Verify next weekly retrain produces equivalent metrics to prior version.
- Rename to `retrain_production.py` for clarity.
- Update `weekly_retrain.py` to point at the new name.
- **Rollback:** revert + restore old name. weekly_retrain.py reverts. Pre-merge tag.

### Step 4: Delete `ml/walk_forward.py` (low risk, after step 3)
- Confirm no consumer left.
- `train_unified.py` migrates to new core OR `xg_only` mode is folded into production retrain.
- Delete `ml/walk_forward.py`.

### Step 5: Add `data_inputs` to all plugins, not just the 7 we did (low risk, optional)
- One-time annotation pass through the 30+ feature plugins.
- Cache invalidation now fully data-aware across the entire pipeline.

## Cost / benefit

**Cost:**
- Steps 1-3 are an estimated 6-10 working hours of focused work.
- Risk: weekly retrain pipeline could break for one cycle if step 3 has a regression. Mitigated by running new + old side-by-side for one cycle before flipping over.

**Benefit:**
- ~700 lines of duplicated code collapse to ~250 lines of shared core + thin scripts.
- One bug surface for walkforward semantics (currently three).
- Easier to add new markets (add to `MARKETS` dict, no trainer change).
- Multi-seed and calibration become available everywhere, not just production retrain.
- Cache invalidation, leakage safety, deployment state become composable building blocks.
- Future feature changes only need to be tested against ONE walkforward implementation.

## What NOT to consolidate

- **Do not consolidate the rejection-threshold deployment logic into the core.** That's production-specific. Keep it in the production retrain script.
- **Do not consolidate the per-market spec dict.** That's diagnostic-specific. Keep it in the diagnostic trainer.
- **Do not collapse the production retrain into the diagnostic trainer.** They produce different artifacts (single production model vs per-fold archive) and have different audiences (live betting vs measurement). Keep them as separate caller scripts that share a core.

## Recommended next session

The full migration is too much for one session. Recommended split:

- **Session 1:** Build `ml/walkforward_core.py` + unit tests. Migrate `train_walkforward.py` to use it. (Steps 1-2.)
- **Session 2 (the next weekly retrain cycle):** Migrate `retrain_no_odds_catboost.py`, run side-by-side with old version for one cycle, flip over. Delete `ml/walk_forward.py`. (Steps 3-4.)
- **Session 3 (anytime):** Annotate remaining plugins with `data_inputs` for full data-aware caching. (Step 5.)

## Verdict

**STRONG.** This is genuine duplication and is causing real bugs (the cache footgun was a symptom of "many ways to do the same thing"). The migration is reversible at each stage. The end state is meaningfully cleaner.

**Confidence:** high.

**If I'm wrong:** the migration could surface that one of the three implementations has subtle behavior the others don't replicate — e.g., `TimeSeriesSplitter` might handle missing seasons differently than the inline loop. We'd find this in step 2 (running side-by-side comparison) before any production change. Easy to roll back at that point.

**What I'm NOT saying:** I'm not saying the consolidation is urgent. The current setup works (production model trains weekly, diagnostics produce backtests). It's tech debt, not a fire. Schedule when there's a quiet week, not when other features are mid-flight.
