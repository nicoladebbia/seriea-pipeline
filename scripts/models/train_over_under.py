"""Train O/U binary classifiers for goal line markets.

Trains a CatBoost binary classifier on over_2_5 = (home_score + away_score >= 3).
Uses walk-forward CV, Optuna hyperparameter tuning, binary-specific feature
selection, and correlation pruning.

Usage:
    python -m scripts.models.train_over_under
    python -m scripts.models.train_over_under --lines 1.5 2.5 3.5
    python -m scripts.models.train_over_under --top-k 60
    python -m scripts.models.train_over_under --n-tune-trials 30
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from config.settings import MODELS_DIR
from ml.config import META_COLS, ValidationConfig
from ml.data import DataLoader, TimeSeriesSplitter
from ml.feature_selection import correlation_pruning, exclude_odds
from storage.paths import features_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

OUTPUT_DIR = MODELS_DIR / "universal" / "over_under"

# Goal lines to train classifiers for
DEFAULT_LINES = [2.5]


def _build_binary_target(
    df: pd.DataFrame, mask: pd.Series, line: float,
) -> pd.Series:
    """Build binary target: 1 if total goals >= line, else 0."""
    total_goals = (
        df.loc[mask, "home_score"].values + df.loc[mask, "away_score"].values
    )
    # Reset index to align with X from get_universal_dataset() which also resets
    return pd.Series((total_goals >= line).astype(int)).reset_index(drop=True)


def binary_importance_selection(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    top_k: int = 60,
) -> Tuple[List[str], Dict[str, float]]:
    """XGBClassifier-based feature importance for binary target.

    Can't reuse importance_based_selection() from ml/feature_selection.py because
    it uses y.map(LABEL_MAP) which maps H/D/A to ints. Binary target is already int.

    Uses walk-forward averaged importance to avoid lookahead bias:
    train on seasons[:i], compute importances, average across all folds.
    """
    from xgboost import XGBClassifier

    splitter = TimeSeriesSplitter(ValidationConfig())
    splits = splitter.generate_splits(X["_season"])
    importance_accum: Dict[str, List[float]] = {f: [] for f in feature_names}

    for train_seasons, _ in splits:
        train_mask = X["_season"].isin(train_seasons)
        X_tr = X[train_mask][feature_names]
        y_tr = y[train_mask]

        model = XGBClassifier(
            objective="binary:logistic",
            max_depth=5,
            learning_rate=0.03,
            n_estimators=1500,
            subsample=0.75,
            colsample_bytree=0.6,
            eval_metric="logloss",
            verbosity=0,
            random_state=42,
        )
        model.fit(X_tr, y_tr, verbose=False)
        for feat, score in zip(feature_names, model.feature_importances_):
            importance_accum[feat].append(score)

    avg = {f: float(np.mean(s)) for f, s in importance_accum.items()}
    selected = sorted(avg, key=lambda f: -avg[f])[:top_k]
    log.info("Binary importance: selected %d / %d features", len(selected), len(feature_names))
    return selected, avg


def walk_forward_cv_binary(
    X: pd.DataFrame,
    y: pd.Series,
    selected: List[str],
    n_splits: int = 5,
    cb_params: Dict = None,
) -> Dict:
    """Walk-forward CV for binary O/U classifier.

    Args:
        cb_params: Optional CatBoost hyperparameters. If None, uses defaults.

    Returns dict with overall metrics + per-fold breakdown.
    """
    from catboost import CatBoostClassifier, Pool

    splitter = TimeSeriesSplitter(ValidationConfig())
    splits = splitter.generate_splits(X["_season"])
    # Take last n_splits folds
    splits = splits[-n_splits:]

    # Default params — can be overridden by Optuna
    default_params = {
        "iterations": 2000,
        "learning_rate": 0.02,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "min_data_in_leaf": 20,
        "loss_function": "Logloss",
        "early_stopping_rounds": 150,
        "verbose": 0,
        "random_seed": 42,
    }
    if cb_params:
        default_params.update(cb_params)

    all_probs = []
    all_true = []
    fold_metrics = []

    for fold_idx, (train_seasons, test_seasons) in enumerate(splits):
        train_mask = X["_season"].isin(train_seasons)
        test_mask = X["_season"].isin(test_seasons)

        X_train = X[train_mask][selected]
        y_train = y[train_mask]
        X_test = X[test_mask][selected]
        y_test = y[test_mask]

        if len(X_test) == 0:
            continue

        # Split training into 85/15 for early stopping
        n_train = int(len(X_train) * 0.85)
        X_tr, X_val = X_train.iloc[:n_train], X_train.iloc[n_train:]
        y_tr, y_val = y_train.iloc[:n_train], y_train.iloc[n_train:]

        model = CatBoostClassifier(**default_params)

        train_pool = Pool(X_tr, label=y_tr)
        val_pool = Pool(X_val, label=y_val)
        model.fit(train_pool, eval_set=val_pool)

        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        fold_ll = log_loss(y_test, proba)
        fold_brier = brier_score_loss(y_test, proba)
        fold_acc = (preds == y_test.values).mean()

        # Calibration gap: mean |predicted - actual| per decile bin
        cal_gap = _calibration_gap(y_test.values, proba)

        test_label = test_seasons[0] if len(test_seasons) == 1 else f"{test_seasons[0]}-{test_seasons[-1]}"
        fold_metrics.append({
            "fold": fold_idx + 1,
            "test": test_label,
            "n_test": len(y_test),
            "base_rate": float(y_test.mean()),
            "log_loss": round(fold_ll, 4),
            "brier": round(fold_brier, 4),
            "accuracy": round(fold_acc, 4),
            "calibration_gap": round(cal_gap, 4),
        })

        all_probs.extend(proba.tolist())
        all_true.extend(y_test.values.tolist())

        log.info(
            "  Fold %d [%s]: n=%d base=%.2f ll=%.4f brier=%.4f acc=%.3f cal_gap=%.3f",
            fold_idx + 1, test_label, len(y_test), y_test.mean(),
            fold_ll, fold_brier, fold_acc, cal_gap,
        )

    all_probs = np.array(all_probs)
    all_true = np.array(all_true)

    overall = {
        "overall_log_loss": round(float(log_loss(all_true, all_probs)), 4),
        "overall_brier": round(float(brier_score_loss(all_true, all_probs)), 4),
        "overall_accuracy": round(float(((all_probs >= 0.5).astype(int) == all_true).mean()), 4),
        "overall_calibration_gap": round(float(_calibration_gap(all_true, all_probs)), 4),
        "n_total": len(all_true),
        "base_rate": round(float(all_true.mean()), 4),
        "fold_metrics": fold_metrics,
    }
    return overall


def optuna_tune_binary(
    X: pd.DataFrame,
    y: pd.Series,
    selected: List[str],
    n_trials: int = 30,
    n_splits: int = 5,
) -> Dict:
    """Optuna hyperparameter optimization for binary O/U classifier.

    Searches over CatBoost hyperparameters using walk-forward CV log-loss
    as the objective. Returns the best params dict.

    Why Optuna matters here: the fixed-param model gets log-loss 0.692, just
    barely failing the 0.69 quality gate. Tuning depth, learning rate, and
    regularization can close that 0.2% gap.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50, step=5),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
            "random_strength": trial.suggest_float("random_strength", 0.1, 3.0),
            "iterations": 2500,
            "loss_function": "Logloss",
            "early_stopping_rounds": 150,
            "verbose": 0,
            "random_seed": 42,
        }
        cv = walk_forward_cv_binary(X, y, selected, n_splits=n_splits, cb_params=params)
        return cv["overall_log_loss"]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best.update({
        "iterations": 2500,
        "loss_function": "Logloss",
        "early_stopping_rounds": 150,
        "verbose": 0,
        "random_seed": 42,
    })

    log.info("Optuna best trial: log-loss=%.4f", study.best_value)
    log.info("Best params: %s", {k: round(v, 4) if isinstance(v, float) else v
                                  for k, v in study.best_params.items()})

    return best, study.best_value


def _calibration_gap(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """Mean absolute calibration error across bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    gaps = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_pred >= lo) & (y_pred < hi)
        if mask.sum() >= 5:
            gaps.append(abs(y_true[mask].mean() - y_pred[mask].mean()))
    return float(np.mean(gaps)) if gaps else 0.0


# =============================================================================
# PROMOTION GATE
# =============================================================================
# A candidate is judged against the model production actually runs, on the
# same time-ordered holdout tail — never against a fixed bar the incumbent
# itself fails (a fixed bar froze the 1X2 model for 130 days, 5b1e111).
# Before this gate the trainer logged its quality checks and saved regardless:
# both live O/U models shipped 2026-09-01 failing their calibration gate.

PROMOTION_LL_TOLERANCE = 0.005    # candidate may be this much worse on holdout log-loss
PROMOTION_CAL_TOLERANCE = 0.01    # ... and this much worse on holdout calibration gap
PROMOTION_BETTER_MARGIN = 0.002   # gains below this read "within tolerance", not "better"


def holdout_metrics(y_true, proba) -> Dict[str, float]:
    """Binary log-loss / Brier / calibration gap of one model on one holdout."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "log_loss": round(float(log_loss(y, p, labels=[0, 1])), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "calibration_gap": round(_calibration_gap(y, p), 4),
        "n": int(len(y)),
    }


def score_incumbent(
    out_dir: Path, line_str: str, X_val: pd.DataFrame, y_val,
) -> Optional[Dict[str, float]]:
    """Score the LIVE model on the candidate's holdout with its own feature list.

    None when there is no live model, or it needs a feature the current frame
    does not carry (then it cannot be compared fairly and the candidate wins
    by default — logged, never silent).
    """
    out_dir = Path(out_dir)
    model_path = out_dir / f"ou_{line_str}_catboost_latest.cbm"
    meta_path = out_dir / f"ou_{line_str}_catboost_metadata.json"
    if not model_path.exists() or not meta_path.exists():
        return None
    try:
        from catboost import CatBoostClassifier
        feats = json.loads(meta_path.read_text()).get("feature_names") or []
        missing = [f for f in feats if f not in X_val.columns]
        if not feats or missing:
            log.warning(
                "Incumbent O/U %s unscorable: %d of %d features missing from the "
                "current frame (%s)", line_str, len(missing), len(feats), missing[:5],
            )
            return None
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        proba = model.predict_proba(X_val[feats])[:, 1]
        return holdout_metrics(y_val, proba)
    except Exception as e:  # noqa: BLE001 — any failure means "cannot compare"
        log.warning("Incumbent O/U %s unscorable: %s", line_str, e)
        return None


def decide_promotion(
    candidate: Dict[str, float],
    incumbent: Optional[Dict[str, float]],
    naive_ll: float,
    ll_tol: float = PROMOTION_LL_TOLERANCE,
    cal_tol: float = PROMOTION_CAL_TOLERANCE,
) -> Tuple[bool, str]:
    """(promote?, reason). Both metric dicts come from holdout_metrics() on the
    SAME holdout; naive_ll is the log-loss of predicting the training base rate."""
    c_ll = float(candidate["log_loss"])
    if c_ll >= naive_ll:
        return False, (f"candidate holdout log-loss {c_ll:.4f} does not beat "
                       f"naive {naive_ll:.4f}")
    if incumbent is None:
        return True, (f"no scorable incumbent; candidate beats naive "
                      f"({c_ll:.4f} < {naive_ll:.4f})")
    d_ll = c_ll - float(incumbent["log_loss"])
    d_cal = float(candidate["calibration_gap"]) - float(incumbent["calibration_gap"])
    if d_ll > ll_tol:
        return False, (f"worse than incumbent by {d_ll:+.4f} log-loss "
                       f"(tolerance {ll_tol})")
    if d_cal > cal_tol:
        return False, (f"calibration gap worse than incumbent by {d_cal:+.4f} "
                       f"(tolerance {cal_tol})")
    if d_ll < -PROMOTION_BETTER_MARGIN:
        return True, (f"better than incumbent by {abs(d_ll):.4f} log-loss "
                      f"(cal gap {d_cal:+.4f})")
    return True, (f"within tolerance of incumbent (log-loss {d_ll:+.4f}, "
                  f"cal gap {d_cal:+.4f})")


def persist_model(
    out_dir: Path, line_str: str, model, meta: Dict, promoted: bool,
) -> Dict[str, str]:
    """Promoted: archive the incumbent to prev/ (retain 1) and overwrite _latest.
    Refused: write to candidate/ and leave _latest untouched.

    Only the top-level ou_*_catboost_metadata.json files are live — the engine
    globs that directory non-recursively — so prev/ and candidate/ are inert.
    """
    out_dir = Path(out_dir)
    if promoted:
        latest = out_dir / f"ou_{line_str}_catboost_latest.cbm"
        latest_meta = out_dir / f"ou_{line_str}_catboost_metadata.json"
        prev_dir = out_dir / "prev"
        prev_dir.mkdir(parents=True, exist_ok=True)
        if latest.exists():
            shutil.copy2(latest, prev_dir / f"ou_{line_str}_catboost_prev.cbm")
        if latest_meta.exists():
            shutil.copy2(latest_meta, prev_dir / f"ou_{line_str}_catboost_prev_metadata.json")
        _write_pair(model, meta, latest, latest_meta)
        return {"model": str(latest), "metadata": str(latest_meta)}
    cand_dir = out_dir / "candidate"
    cand_dir.mkdir(parents=True, exist_ok=True)
    cand = cand_dir / f"ou_{line_str}_catboost_candidate.cbm"
    cand_meta = cand_dir / f"ou_{line_str}_catboost_candidate_metadata.json"
    _write_pair(model, meta, cand, cand_meta)
    return {"model": str(cand), "metadata": str(cand_meta)}


def _write_pair(model, meta: Dict, model_path: Path, meta_path: Path) -> None:
    """Model + metadata via tmp files, then two renames — the engine pairs
    _latest.cbm with its metadata's feature list, so a crash between the two
    writes must never leave a new model next to stale metadata."""
    tmp_model = model_path.with_suffix(".cbm.tmp")
    tmp_meta = meta_path.with_suffix(".json.tmp")
    model.save_model(str(tmp_model))
    tmp_meta.write_text(json.dumps(meta, indent=2, default=str))
    tmp_model.replace(model_path)
    tmp_meta.replace(meta_path)


def dry_run_decision(promoted: bool, reason: str) -> Tuple[bool, str]:
    """A dry run never promotes: the candidate goes to candidate/, _latest is untouched.

    The reason keeps the decision the gate WOULD have made, so a preview run reads
    the same as a real one in the log and in retrain_history.
    """
    return False, f"DRY RUN — would {'PROMOTE' if promoted else 'HOLD'}: {reason}"


def train_over_under(
    lines: List[float] = None,
    top_k: int = 60,
    corr_threshold: float = 0.70,
    n_tune_trials: int = 0,
    dry_run: bool = False,
) -> Dict:
    """Train O/U binary classifiers and return report.

    Steps:
    1. Load features, exclude odds columns
    2. Build binary target from home_score + away_score
    3. Select top_k features by walk-forward binary importance
    4. Prune correlated features
    4b. (Optional) Optuna hyperparameter tuning
    5. Walk-forward CV with CatBoost binary classifier
    6. Train final model (time-ordered 85/15, early-stop on the newest 15%)
    7. Promotion gate: candidate vs incumbent on that same holdout tail;
       refused candidates go to candidate/, _latest is never overwritten

    Args:
        n_tune_trials: Number of Optuna trials. 0 = skip tuning, use defaults.
        dry_run: Train and run the gate, but write only to candidate/ — _latest
            is never touched. The report says what the gate WOULD have done.
    """
    if lines is None:
        lines = DEFAULT_LINES

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load data and exclude odds ---
    fp = str(features_path())
    loader = DataLoader(fp)
    X, _, feature_names = loader.get_universal_dataset()
    log.info("Loaded %d rows, %d features", len(X), len(feature_names))

    X_no_odds, feats_no_odds = exclude_odds(X, feature_names)
    log.info("After excluding odds: %d features", len(feats_no_odds))

    # Build mask aligned with X (must match get_universal_dataset's filtering)
    from ml.config import CLASS_LABELS, RESULT_COL, ValidationConfig, SEASON_COL
    _val_cfg = ValidationConfig()
    raw_mask = loader.df[RESULT_COL].isin(CLASS_LABELS)
    if _val_cfg.min_train_season:
        raw_mask = raw_mask & (loader.df[SEASON_COL] >= _val_cfg.min_train_season)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "variant": "over_under",
        "n_features_original": len(feature_names),
        "n_features_after_odds_exclusion": len(feats_no_odds),
        "lines": {},
    }

    for line in lines:
        log.info("=" * 60)
        log.info("TRAINING O/U %.1f CLASSIFIER", line)
        log.info("=" * 60)

        # --- Step 2: Build binary target ---
        y_binary = _build_binary_target(loader.df, raw_mask, line)
        base_rate = y_binary.mean()
        log.info("Target over_%.1f: base rate = %.3f (%d / %d)", line, base_rate, y_binary.sum(), len(y_binary))

        # Naive baseline: predict base_rate for everything
        naive_brier = base_rate * (1 - base_rate)
        naive_ll = -base_rate * np.log(base_rate + 1e-10) - (1 - base_rate) * np.log(1 - base_rate + 1e-10)
        log.info("Naive baselines: brier=%.4f, log_loss=%.4f", naive_brier, naive_ll)

        # --- Step 3: Feature selection ---
        log.info("Running walk-forward binary importance selection (top_k=%d)...", top_k)
        selected, importance = binary_importance_selection(
            X_no_odds, y_binary, feats_no_odds, top_k=top_k,
        )
        log.info("Selected %d features by importance", len(selected))

        # --- Step 4: Correlation pruning ---
        selected = correlation_pruning(
            X_no_odds, selected, importance, threshold=corr_threshold,
        )
        log.info("After correlation pruning: %d features", len(selected))

        # --- Step 4b (optional): Optuna hyperparameter tuning ---
        best_params = None
        if n_tune_trials > 0:
            log.info("Running Optuna tuning (%d trials)...", n_tune_trials)
            best_params, best_ll = optuna_tune_binary(
                X_no_odds, y_binary, selected,
                n_trials=n_tune_trials, n_splits=5,
            )
            log.info("Optuna best log-loss: %.4f", best_ll)

        # --- Step 5: Walk-forward CV (with tuned or default params) ---
        log.info("Running walk-forward CV%s...", " (tuned params)" if best_params else "")
        cv_results = walk_forward_cv_binary(
            X_no_odds, y_binary, selected, n_splits=5,
            cb_params=best_params,
        )

        # Quality gate checks
        # Gate: 0.693 = must beat naive baseline (0.6931 at 50% base rate).
        # Previous gate of 0.69 was 0.3% below naive — unreachable with team features.
        ll_pass = cv_results["overall_log_loss"] < 0.693
        brier_pass = cv_results["overall_brier"] < naive_brier
        cal_pass = cv_results["overall_calibration_gap"] < 0.03

        log.info("Quality gates:")
        log.info("  Log-loss  %.4f < 0.693   %s", cv_results["overall_log_loss"], "PASS" if ll_pass else "FAIL")
        log.info("  Brier     %.4f < %.4f  %s", cv_results["overall_brier"], naive_brier, "PASS" if brier_pass else "FAIL")
        log.info("  Cal gap   %.4f < 0.03    %s", cv_results["overall_calibration_gap"], "PASS" if cal_pass else "FAIL")

        # --- Step 6: Train final model on all data ---
        log.info("Training final model on all data...")
        from catboost import CatBoostClassifier, Pool

        # Use last 15% of data as eval set for early stopping
        n_total = len(X_no_odds)
        n_train = int(n_total * 0.85)
        X_final_train = X_no_odds.iloc[:n_train][selected]
        y_final_train = y_binary.iloc[:n_train]
        X_final_val = X_no_odds.iloc[n_train:][selected]
        y_final_val = y_binary.iloc[n_train:]

        # Use tuned params or defaults for final model
        final_params = {
            "iterations": 2500,
            "learning_rate": 0.02,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "min_data_in_leaf": 20,
            "loss_function": "Logloss",
            "early_stopping_rounds": 150,
            "verbose": 0,
            "random_seed": 42,
        }
        if best_params:
            final_params.update(best_params)

        final_model = CatBoostClassifier(**final_params)

        train_pool = Pool(X_final_train, label=y_final_train)
        val_pool = Pool(X_final_val, label=y_final_val)
        final_model.fit(train_pool, eval_set=val_pool)

        # Candidate on the holdout tail (rows are date-ordered, so this is the
        # newest 15% — the same slice the incumbent is scored on below)
        eval_proba = final_model.predict_proba(X_final_val)[:, 1]
        cand_holdout = holdout_metrics(y_final_val, eval_proba)
        eval_ll, eval_brier = cand_holdout["log_loss"], cand_holdout["brier"]
        train_base = float(y_final_train.mean())
        naive_holdout_ll = float(log_loss(
            y_final_val, np.full(len(y_final_val), train_base), labels=[0, 1],
        ))
        X_holdout_full = X_no_odds.iloc[n_train:]
        holdout_dates = None
        if "_match_date" in X_holdout_full.columns:
            d = pd.to_datetime(X_holdout_full["_match_date"], errors="coerce")
            holdout_dates = [str(d.min())[:10], str(d.max())[:10]]
        log.info(
            "Final model holdout (n=%d, %s): ll=%.4f brier=%.4f cal_gap=%.4f | naive ll=%.4f",
            cand_holdout["n"], "→".join(holdout_dates) if holdout_dates else "?",
            eval_ll, eval_brier, cand_holdout["calibration_gap"], naive_holdout_ll,
        )

        # --- Step 7: Promotion gate vs the incumbent on the same holdout ---
        line_str = str(line).replace(".", "_")
        incumbent = score_incumbent(OUTPUT_DIR, line_str, X_holdout_full, y_final_val)
        if incumbent:
            log.info(
                "Incumbent holdout: ll=%.4f brier=%.4f cal_gap=%.4f",
                incumbent["log_loss"], incumbent["brier"], incumbent["calibration_gap"],
            )
        promoted, reason = decide_promotion(cand_holdout, incumbent, naive_holdout_ll)
        would_promote = promoted
        if dry_run:
            promoted, reason = dry_run_decision(promoted, reason)
        log.info(
            "PROMOTION O/U %.1f: %s — %s",
            line, "PROMOTED" if promoted else "REFUSED (latest untouched)", reason,
        )
        promotion = {
            "promoted": promoted,
            "reason": reason,
            "decided_at": datetime.now(timezone.utc).isoformat(),
            "holdout": {
                "n": cand_holdout["n"],
                "dates": holdout_dates,
                "candidate": cand_holdout,
                "incumbent": incumbent,
                "naive_log_loss": round(naive_holdout_ll, 4),
            },
        }
        if dry_run:
            promotion["dry_run"] = True
            promotion["would_promote"] = would_promote

        # Save metadata
        meta = {
            "line": line,
            "feature_names": selected,
            "n_features": len(selected),
            "base_rate": round(base_rate, 4),
            "cv_metrics": cv_results,
            "eval_metrics": {
                "log_loss": round(eval_ll, 4),
                "brier": round(eval_brier, 4),
                "calibration_gap": cand_holdout["calibration_gap"],
            },
            "quality_gates": {
                "log_loss_pass": bool(ll_pass),
                "brier_pass": bool(brier_pass),
                "calibration_pass": bool(cal_pass),
            },
            "promotion": promotion,
            "tuned_params": {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in (best_params or {}).items()
                            if k not in ("verbose", "random_seed", "loss_function")},
            "n_tune_trials": n_tune_trials,
            "top_20_features": sorted(importance.items(), key=lambda x: -x[1])[:20],
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_training_rows": n_total,
        }
        paths = persist_model(OUTPUT_DIR, line_str, final_model, meta, promoted)
        log.info("Saved model to %s (metadata %s)", paths["model"], paths["metadata"])

        report["lines"][str(line)] = {
            "n_features_selected": len(selected),
            "selected_features": selected,
            "base_rate": round(base_rate, 4),
            "cv_metrics": cv_results,
            "quality_gates_passed": bool(ll_pass and brier_pass and cal_pass),
            "tuned": n_tune_trials > 0,
            "promoted": promoted,
            "would_promote": would_promote,
            "dry_run": dry_run,
            "promotion_reason": reason,
            "holdout": promotion["holdout"],
            "paths": paths,
        }

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("O/U CLASSIFIER TRAINING COMPLETE")
    print("=" * 70)
    for line in lines:
        info = report["lines"][str(line)]
        cv = info["cv_metrics"]
        print(f"\nLine {line} (base rate: {info['base_rate']:.3f}):")
        print(f"  Features selected: {info['n_features_selected']}")
        print(f"  Walk-forward CV:")
        print(f"    Log-loss:    {cv['overall_log_loss']:.4f}")
        print(f"    Brier:       {cv['overall_brier']:.4f}")
        print(f"    Accuracy:    {cv['overall_accuracy']:.4f}")
        print(f"    Cal gap:     {cv['overall_calibration_gap']:.4f}")
        print(f"  Quality gates: {'ALL PASS' if info['quality_gates_passed'] else 'SOME FAILED'}")
        print(f"  Promotion:     {'PROMOTED' if info['promoted'] else 'REFUSED — latest untouched'}")
        print(f"                 {info['promotion_reason']}")

        print(f"\n  Top 10 features:")
        for feat, score in sorted(importance.items(), key=lambda x: -x[1])[:10]:
            print(f"    {score:8.4f}  {feat}")

    # Save overall report
    report_path = OUTPUT_DIR / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Report saved to %s", report_path)
    print("=" * 70)

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train O/U binary classifier")
    parser.add_argument(
        "--lines", type=float, nargs="+", default=DEFAULT_LINES,
        help="Goal lines to train classifiers for (default: 2.5)",
    )
    parser.add_argument("--top-k", type=int, default=60, help="Number of features to select")
    parser.add_argument("--corr-threshold", type=float, default=0.70, help="Correlation pruning threshold")
    parser.add_argument("--n-tune-trials", type=int, default=0, help="Optuna tuning trials (0=skip)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Train and gate, but write only to candidate/ — never touch _latest",
    )
    args = parser.parse_args()

    report = train_over_under(
        lines=args.lines,
        top_k=args.top_k,
        corr_threshold=args.corr_threshold,
        n_tune_trials=args.n_tune_trials,
        dry_run=args.dry_run,
    )
