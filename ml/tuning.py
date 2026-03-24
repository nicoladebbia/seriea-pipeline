"""Optuna-based hyperparameter optimization over walk-forward CV."""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import optuna
import pandas as pd

from ml.config import (
    CLASS_INDICES,
    LABEL_MAP,
    META_COLS,
    N_CLASSES,
    RANDOM_SEED,
    TuningConfig,
    ValidationConfig,
)
from ml.data import TimeSeriesSplitter

log = logging.getLogger(__name__)

# Silence Optuna's verbose output
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _compute_sample_weights(
    y: pd.Series,
    seasons: pd.Series | None = None,
) -> np.ndarray:
    """Compute sample weights: class balancing × draw correction × time decay.

    Three multiplicative components:
    1. Inverse-frequency class weights (sklearn standard)
    2. Draw weight correction: adapts to actual draw rate in this fold
       - "auto" mode: targets 1/3 effective draw weight (equal-class prior)
       - fixed mode: uses draw_weight_multiplier (legacy, hardcoded 2.0)
    3. Time decay: exponential decay per season gap from most recent training
       season (Dixon-Coles 1997). Newer matches influence the model more.

    Args:
        y: Target labels (H/D/A strings or 0/1/2 ints)
        seasons: Season strings aligned with y, for time-decay computation.
                 If None, time-decay is skipped.
    """
    from ml.config import ModelConfig
    mc = ModelConfig()

    # --- 1. Inverse-frequency class weights ---
    counts = y.value_counts()
    total = len(y)
    n_classes = len(counts)
    class_weights = {cls: total / (n_classes * cnt) for cls, cnt in counts.items()}

    # --- 2. Draw weight correction ---
    draw_key = "D" if "D" in class_weights else (1 if 1 in class_weights else None)
    if draw_key is not None:
        if mc.draw_weight_mode == "auto":
            # Target: draws should have effective weight = 1/3 of total.
            # Current effective draw weight = class_weights[D] * draw_count / total_weighted.
            # We compute the multiplier that achieves 1/3 target.
            draw_count = counts.get(draw_key, 0)
            if draw_count > 0:
                draw_rate = draw_count / total
                # Inverse-frequency already gives weight = total/(n_classes*count).
                # For 3-class: base_weight = 1/(3*draw_rate).
                # To target 1/3 effective share: multiplier = 1/(3 * draw_rate * base_weight)
                # Simplifies to: multiplier = n_classes * draw_rate / draw_rate = ... = 1.0
                # But that's only true if all classes have equal rate.
                # The real formula: we want draw_share = 1/3 of total effective weight.
                # total_effective = sum(class_weight[c] * count[c] for all c)
                # draw_effective = class_weight[D] * count[D]
                # Currently draw_effective/total_effective = 1/n_classes (by construction).
                # So base inverse-freq already targets 1/3. The draw boost should
                # account for the fact that draws are HARDER to predict — we want
                # the model to pay MORE attention to draws than 1/3.
                # Target: 38% effective draw weight (slightly over 1/3 = 33%).
                target_draw_share = 0.38
                current_draw_share = 1.0 / n_classes  # inverse-freq gives equal shares
                draw_multiplier = target_draw_share / current_draw_share
                class_weights[draw_key] *= draw_multiplier
        else:
            class_weights[draw_key] *= mc.draw_weight_multiplier

    w = y.map(class_weights).values.astype(np.float64)

    # --- 3. Time decay (Dixon-Coles) ---
    decay = mc.time_decay_per_season
    if seasons is not None and decay < 1.0:
        season_vals = seasons.values if hasattr(seasons, 'values') else seasons
        # Map seasons to integer order (most recent = 0)
        unique_seasons = sorted(set(season_vals))
        season_rank = {s: i for i, s in enumerate(unique_seasons)}
        max_rank = len(unique_seasons) - 1

        # Decay: weight = decay^(max_rank - rank). Most recent season = 1.0.
        season_indices = np.array([season_rank.get(s, 0) for s in season_vals])
        time_weights = np.power(decay, max_rank - season_indices)
        w *= time_weights

    return w


def _suggest_from_space(
    trial: optuna.Trial,
    space: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a param dict by calling trial.suggest_* for each entry in *space*."""
    params: Dict[str, Any] = {}
    for name, spec in space.items():
        low, high = spec["low"], spec["high"]
        if isinstance(low, int) and isinstance(high, int) and not spec.get("log"):
            step = spec.get("step", 1)
            params[name] = trial.suggest_int(name, low, high, step=step)
        elif spec.get("log"):
            params[name] = trial.suggest_float(name, low, high, log=True)
        else:
            params[name] = trial.suggest_float(name, low, high)
    return params


def _coupled_n_estimators(lr: float, cfg: TuningConfig) -> int:
    """Couple learning rate and n_estimators inversely.

    Lower LR → more trees; higher LR → fewer trees.
    """
    n = int(cfg.lr_estimator_base_n * (cfg.lr_estimator_base_lr / lr))
    return max(cfg.lr_estimator_min_n, min(n, cfg.lr_estimator_max_n))


def _xgb_search_space(trial: optuna.Trial, cfg: TuningConfig) -> Dict[str, Any]:
    params = _suggest_from_space(trial, cfg.xgb_search_space)
    n_est = _coupled_n_estimators(params["learning_rate"], cfg)
    params.update({
        "objective": "multi:softprob",
        "num_class": N_CLASSES,
        "n_estimators": n_est,
        "random_state": RANDOM_SEED,
        "tree_method": "hist",
    })
    return params


def _lgb_search_space(trial: optuna.Trial, cfg: TuningConfig) -> Dict[str, Any]:
    params = _suggest_from_space(trial, cfg.lgb_search_space)
    n_est = _coupled_n_estimators(params["learning_rate"], cfg)
    params.update({
        "objective": "multiclass",
        "num_class": N_CLASSES,
        "n_estimators": n_est,
        "random_state": RANDOM_SEED,
        "verbose": -1,
    })
    return params


def _cb_search_space(trial: optuna.Trial, cfg: TuningConfig) -> Dict[str, Any]:
    params = _suggest_from_space(trial, cfg.cb_search_space)
    n_iter = _coupled_n_estimators(params["learning_rate"], cfg)
    params.update({
        "loss_function": "MultiClass",
        "classes_count": N_CLASSES,
        "iterations": n_iter,
        "random_seed": RANDOM_SEED,
        "verbose": 0,
        "auto_class_weights": "Balanced",
    })
    return params


SEARCH_SPACES = {
    "xgboost": _xgb_search_space,
    "lightgbm": _lgb_search_space,
    "catboost": _cb_search_space,
}


def _build_model(model_type: str, params: Dict[str, Any]):
    """Build a raw sklearn-compatible model from params (no wrapper overhead)."""
    if model_type == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(**params)
    elif model_type == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(**params)
    elif model_type == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**params)
    raise ValueError(f"Unknown model type: {model_type}")


def _fit_with_early_stopping(
    model, X_tr: pd.DataFrame, y_tr, model_type: str,
    sample_weight=None,
) -> None:
    """Fit a raw sklearn model with early stopping on a chronological validation split."""
    from ml.config import ModelConfig
    mc = ModelConfig()
    n = len(X_tr)
    split = int(n * (1.0 - mc.early_stopping_val_fraction))

    # Convert to numpy arrays to avoid XGBoost/pandas compatibility issues
    X_arr = X_tr.values if hasattr(X_tr, 'values') else X_tr
    y_arr = y_tr.values if hasattr(y_tr, 'values') else y_tr

    if split < 100 or (n - split) < 30:
        fit_kw = {"sample_weight": sample_weight} if sample_weight is not None else {}
        model.fit(X_arr, y_arr, **fit_kw)
        return

    Xt, yt = X_arr[:split], y_arr[:split]
    Xv, yv = X_arr[split:], y_arr[split:]
    sw_tr = sample_weight[:split] if sample_weight is not None else None

    if model_type == "xgboost":
        fit_kw = {"sample_weight": sw_tr} if sw_tr is not None else {}
        model.set_params(early_stopping_rounds=mc.early_stopping_rounds)
        model.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False, **fit_kw)
    elif model_type == "lightgbm":
        import lightgbm as lgb
        fit_kw = {"sample_weight": sw_tr} if sw_tr is not None else {}
        callbacks = [
            lgb.early_stopping(mc.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        model.fit(Xt, yt, eval_set=[(Xv, yv)], callbacks=callbacks, **fit_kw)
    elif model_type == "catboost":
        fit_kw = {"sample_weight": sw_tr} if sw_tr is not None else {}
        model.set_params(early_stopping_rounds=mc.early_stopping_rounds)
        model.fit(Xt, yt, eval_set=(Xv, yv), verbose=False, **fit_kw)


def _walk_forward_score(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    params: Dict[str, Any],
    use_sample_weights: bool = True,
    trial: optuna.Trial | None = None,
) -> float:
    """Evaluate params via walk-forward CV. Returns mean log_loss (lower=better).

    When *trial* is provided, reports intermediate fold-level values to
    Optuna for early pruning of unpromising configurations.
    """
    config = ValidationConfig()
    splitter = TimeSeriesSplitter(config)
    splits = splitter.generate_splits(X["_season"])

    y_int = y.map(LABEL_MAP)
    fold_losses = []

    for fold_idx, (train_seasons, test_seasons) in enumerate(splits):
        train_mask = X["_season"].isin(train_seasons)
        test_mask = X["_season"].isin(test_seasons)

        X_tr = X[train_mask].drop(columns=[c for c in META_COLS if c in X.columns])
        y_tr = y_int[train_mask]
        X_te = X[test_mask].drop(columns=[c for c in META_COLS if c in X.columns])
        y_te = y_int[test_mask]

        model = _build_model(model_type, params)

        train_seasons = X[train_mask]["_season"] if "_season" in X.columns else None
        sw = _compute_sample_weights(y[train_mask], seasons=train_seasons) if use_sample_weights else None
        _fit_with_early_stopping(model, X_tr, y_tr, model_type, sample_weight=sw)

        # Convert to numpy for XGBoost compatibility
        X_te_arr = X_te.values if hasattr(X_te, 'values') else X_te
        y_proba = model.predict_proba(X_te_arr)

        from sklearn.metrics import log_loss as sk_log_loss
        ll = sk_log_loss(y_te, y_proba, labels=CLASS_INDICES)
        fold_losses.append(ll)

        # Report intermediate value for Optuna pruning
        if trial is not None:
            trial.report(np.mean(fold_losses), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return float(np.mean(fold_losses))


def tune_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    n_trials: int | None = None,
    use_sample_weights: bool = True,
    tuning_config: TuningConfig | None = None,
) -> Dict[str, Any]:
    """Run Optuna hyperparameter search for a model type.

    Returns dict with:
      best_params: the optimal parameters
      best_score: best mean CV log_loss
      study: the Optuna study object
    """
    cfg = tuning_config or TuningConfig()
    n_trials = n_trials if n_trials is not None else cfg.n_trials
    space_fn = SEARCH_SPACES[model_type]

    def objective(trial: optuna.Trial) -> float:
        params = space_fn(trial, cfg)
        try:
            score = _walk_forward_score(
                X, y, model_type, params,
                use_sample_weights=use_sample_weights,
                trial=trial,
            )
        except optuna.TrialPruned:
            raise
        except Exception as e:
            log.warning("Trial %d failed: %s", trial.number, e)
            return float("inf")
        return score

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=cfg.pruner_startup_trials,
        ),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    best_params = space_fn(best, cfg)

    log.info(
        "Tuning %s: best log_loss=%.4f after %d trials",
        model_type, best.value, len(study.trials),
    )

    return {
        "best_params": best_params,
        "best_score": best.value,
        "study": study,
    }
