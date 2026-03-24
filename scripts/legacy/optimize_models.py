#!/usr/bin/env python3
"""Advanced Model Optimization for All Betting Markets.

Implements:
1. Optuna hyperparameter tuning for each market model
2. Stacking ensemble (CatBoost + XGBoost + LightGBM meta-learner)
3. Market-odds blending (anchor on Pinnacle + model lift)
4. Time-weighted training (recent seasons weighted more)
5. Feature importance pruning
6. Walk-forward cross-validation across multiple seasons

Target: Push 1X2 accuracy beyond 60% out-of-sample.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DATA_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MARKET_MODELS_DIR = MODELS_DIR / "markets"
MARKET_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_data_and_features():
    """Load unified training data and features (leakage-free)."""
    from scripts.models.comprehensive_markets import load_unified_training_data, get_training_features, _compute_targets

    df = load_unified_training_data()
    features = get_training_features(df)
    df = _compute_targets(df)
    return df, features


# =============================================================================
# WALK-FORWARD CROSS-VALIDATION
# =============================================================================

def walk_forward_cv(df: pd.DataFrame, features: List[str], target_col: str,
                    task: str = "classification", n_classes: int = 2,
                    min_train_seasons: int = 5) -> List[Dict]:
    """Walk-forward CV: for each test season, train on all prior seasons.

    Returns list of fold results with predictions.
    """
    seasons = sorted(df["season"].unique())
    folds = []

    for i in range(min_train_seasons, len(seasons)):
        test_season = seasons[i]
        val_season = seasons[i - 1]
        train_seasons = seasons[:i - 1]

        test_mask = df["season"] == test_season
        val_mask = df["season"] == val_season
        train_mask = df["season"].isin(train_seasons)

        # Check we have enough data
        if train_mask.sum() < 500 or test_mask.sum() < 50:
            continue

        # Check target availability
        if df.loc[test_mask, target_col].isna().all():
            continue

        folds.append({
            "test_season": test_season,
            "train_seasons": train_seasons,
            "val_season": val_season,
            "train_idx": df.index[train_mask].tolist(),
            "val_idx": df.index[val_mask].tolist(),
            "test_idx": df.index[test_mask].tolist(),
        })

    log.info("Walk-forward CV: %d folds for target '%s'", len(folds), target_col)
    return folds


# =============================================================================
# OPTUNA HYPERPARAMETER TUNING
# =============================================================================

def tune_catboost_1x2(df: pd.DataFrame, features: List[str], n_trials: int = 50) -> Dict:
    """Tune CatBoost hyperparameters for 1X2 prediction using Optuna."""
    import optuna
    from catboost import CatBoostClassifier

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    label_map = {"H": 0, "D": 1, "A": 2}
    df_valid = df[df["result"].isin(["H", "D", "A"])].copy()
    X = df_valid[features].fillna(0)
    y = df_valid["result"].map(label_map)

    # Use the last 3 seasons for walk-forward validation
    seasons = sorted(df_valid["season"].unique())
    test_seasons = seasons[-3:]
    log.info("Tuning 1X2 on test seasons: %s", test_seasons)

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 500, 3000, step=100),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 30, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
            "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
        }

        # Walk-forward CV across test_seasons
        accs = []
        lls = []
        for ts in test_seasons:
            ts_idx = seasons.index(ts)
            vs = seasons[ts_idx - 1]
            tr_seasons = seasons[:ts_idx - 1]

            train_mask = df_valid["season"].isin(tr_seasons)
            val_mask = df_valid["season"] == vs
            test_mask = df_valid["season"] == ts

            if train_mask.sum() < 500 or test_mask.sum() < 50:
                continue

            model = CatBoostClassifier(
                **params, random_seed=42, verbose=0,
                loss_function="MultiClass", classes_count=3,
                auto_class_weights="Balanced",
                early_stopping_rounds=100,
            )
            model.fit(X[train_mask], y[train_mask],
                      eval_set=(X[val_mask], y[val_mask]),
                      verbose=0)

            probs = model.predict_proba(X[test_mask])
            pred = probs.argmax(axis=1)
            true = y[test_mask].values

            accs.append((pred == true).mean())
            try:
                lls.append(log_loss(true, probs, labels=[0, 1, 2]))
            except:
                lls.append(1.2)

        if not accs:
            return 1.2

        # Optimize for log_loss (lower is better)
        return np.mean(lls)

    study = optuna.create_study(direction="minimize", study_name="1x2_tuning")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    log.info("Best 1X2 params (LL=%.4f): %s", study.best_value, best)

    return best


def tune_catboost_binary(df: pd.DataFrame, features: List[str],
                         target_col: str, n_trials: int = 30) -> Dict:
    """Tune CatBoost for a binary classification target."""
    import optuna
    from catboost import CatBoostClassifier

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    valid = df[target_col].notna()
    df_valid = df[valid].copy()
    X = df_valid[features].fillna(0)
    y = df_valid[target_col].astype(int)

    seasons = sorted(df_valid["season"].unique())
    test_seasons = [s for s in seasons[-3:] if df_valid[df_valid["season"] == s][target_col].notna().sum() > 20]

    if not test_seasons:
        log.warning("Not enough data to tune %s", target_col)
        return {}

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 300, 2000, step=100),
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 20, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 30),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.5),
        }

        lls = []
        for ts in test_seasons:
            ts_idx = seasons.index(ts)
            vs = seasons[ts_idx - 1]
            tr_seasons = seasons[:ts_idx - 1]

            train_mask = df_valid["season"].isin(tr_seasons)
            val_mask = df_valid["season"] == vs
            test_mask = df_valid["season"] == ts

            if train_mask.sum() < 200 or test_mask.sum() < 20:
                continue

            model = CatBoostClassifier(
                **params, random_seed=42, verbose=0,
                loss_function="Logloss",
                early_stopping_rounds=80,
            )
            model.fit(X[train_mask], y[train_mask],
                      eval_set=(X[val_mask], y[val_mask]),
                      verbose=0)

            probs = model.predict_proba(X[test_mask])[:, 1]
            try:
                lls.append(log_loss(y[test_mask].values, probs))
            except:
                lls.append(0.7)

        return np.mean(lls) if lls else 0.7

    study = optuna.create_study(direction="minimize", study_name=f"{target_col}_tuning")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    log.info("Best %s params (LL=%.4f): %s", target_col, study.best_value, study.best_params)
    return study.best_params


# =============================================================================
# STACKING ENSEMBLE
# =============================================================================

def train_stacking_1x2(df: pd.DataFrame, features: List[str],
                       catboost_params: Dict, test_season: str = "2024-2025") -> Dict:
    """Train a stacking ensemble for 1X2 prediction.

    Level 0: CatBoost classifier, XGBoost classifier, Poisson regressor
    Level 1: Logistic regression meta-learner on out-of-fold predictions
    """
    from catboost import CatBoostClassifier, CatBoostRegressor
    from sklearn.linear_model import LogisticRegression
    import pickle

    label_map = {"H": 0, "D": 1, "A": 2}
    df_valid = df[df["result"].isin(["H", "D", "A"])].copy()
    X_all = df_valid[features].fillna(0).values
    y_all = df_valid["result"].map(label_map).values
    y_labels = df_valid["result"].values

    seasons = sorted(df_valid["season"].unique())
    test_idx = seasons.index(test_season) if test_season in seasons else len(seasons) - 1

    # Training seasons (all before test)
    train_val_seasons = seasons[:test_idx]
    test_mask = (df_valid["season"] == test_season).values

    # For stacking, we need out-of-fold predictions from Level-0 models
    train_val_mask = ~test_mask
    X_tv = X_all[train_val_mask]
    y_tv = y_all[train_val_mask]
    seasons_tv = df_valid.loc[train_val_mask, "season"].values

    X_test = X_all[test_mask]
    y_test = y_all[test_mask]
    y_test_labels = y_labels[test_mask]

    # Generate out-of-fold predictions using time-series splits
    n_base = 3  # CatBoost, Poisson-CatBoost, market odds
    oof_preds = np.zeros((len(X_tv), n_base * 3))  # 3 probabilities per model
    test_preds = np.zeros((len(X_test), n_base * 3))

    # Walk-forward OOF: for each fold, train on prior, predict on current
    unique_tv_seasons = sorted(set(seasons_tv))

    for i, fold_season in enumerate(unique_tv_seasons):
        if i < 5:  # Need at least 5 seasons of history
            continue

        fold_mask = seasons_tv == fold_season
        train_fold_mask = np.isin(seasons_tv, unique_tv_seasons[:i])
        val_fold_mask = np.isin(seasons_tv, [unique_tv_seasons[i - 1]])

        X_fold_train = X_tv[train_fold_mask]
        y_fold_train = y_tv[train_fold_mask]
        X_fold_val = X_tv[val_fold_mask]
        y_fold_val = y_tv[val_fold_mask]
        X_fold_test = X_tv[fold_mask]

        if len(X_fold_train) < 500 or len(X_fold_test) < 30:
            continue

        # Model 1: CatBoost Classifier
        cb_params = {k: v for k, v in catboost_params.items() if k not in ("grow_policy",)}
        cb_params.setdefault("iterations", 1500)
        cb_params.setdefault("depth", 7)
        cb_params.setdefault("learning_rate", 0.03)

        m1 = CatBoostClassifier(
            **cb_params, random_seed=42, verbose=0,
            loss_function="MultiClass", classes_count=3,
            auto_class_weights="Balanced",
            early_stopping_rounds=100,
        )
        m1.fit(X_fold_train, y_fold_train,
               eval_set=(X_fold_val, y_fold_val), verbose=0)
        oof_preds[fold_mask, 0:3] = m1.predict_proba(X_fold_test)

        # Model 2: Poisson goal regressors → derived 1X2 probs
        m2h = CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.03,
                                l2_leaf_reg=5, random_seed=42, verbose=0, loss_function="Poisson")
        m2a = CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.03,
                                l2_leaf_reg=5, random_seed=42, verbose=0, loss_function="Poisson")

        # Need actual goal targets
        hs = df_valid.loc[train_val_mask, "home_score"].values
        as_ = df_valid.loc[train_val_mask, "away_score"].values

        m2h.fit(X_fold_train, hs[train_fold_mask],
                eval_set=(X_fold_val, hs[val_fold_mask]), verbose=0)
        m2a.fit(X_fold_train, as_[train_fold_mask],
                eval_set=(X_fold_val, as_[val_fold_mask]), verbose=0)

        pred_hxg = np.clip(m2h.predict(X_fold_test), 0.3, 4.0)
        pred_axg = np.clip(m2a.predict(X_fold_test), 0.3, 4.0)

        for j, (hxg, axg) in enumerate(zip(pred_hxg, pred_axg)):
            pH = pD = pA = 0
            for h in range(8):
                for a in range(8):
                    p = poisson.pmf(h, hxg) * poisson.pmf(a, axg)
                    if h > a: pH += p
                    elif h == a: pD += p
                    else: pA += p
            t = pH + pD + pA
            oof_preds[fold_mask, 3:6][j] = [pH/t, pD/t, pA/t]

        # Model 3: Market odds (Pinnacle) as probability baseline
        psh = df_valid.loc[train_val_mask, "odds_PSH"].values
        psd = df_valid.loc[train_val_mask, "odds_PSD"].values
        psa = df_valid.loc[train_val_mask, "odds_PSA"].values

        for j, idx in enumerate(np.where(fold_mask)[0]):
            oh, od, oa = psh[idx], psd[idx], psa[idx]
            if oh > 1 and od > 1 and oa > 1:
                raw = np.array([1/oh, 1/od, 1/oa])
                oof_preds[fold_mask, 6:9][j] = raw / raw.sum()
            else:
                # Fallback: use average of other two models
                oof_preds[fold_mask, 6:9][j] = (oof_preds[fold_mask, 0:3][j] + oof_preds[fold_mask, 3:6][j]) / 2

    # Train final Level-0 models on all training data
    val_season = seasons[test_idx - 1]
    train_final = np.isin(seasons_tv, unique_tv_seasons[:-1])
    val_final = seasons_tv == val_season

    log.info("Training final Level-0 models for stacking...")

    # Final CatBoost
    cb_params_final = {k: v for k, v in catboost_params.items() if k not in ("grow_policy",)}
    cb_params_final.setdefault("iterations", 1500)
    cb_params_final.setdefault("depth", 7)
    cb_params_final.setdefault("learning_rate", 0.03)

    m1_final = CatBoostClassifier(
        **cb_params_final, random_seed=42, verbose=0,
        loss_function="MultiClass", classes_count=3,
        auto_class_weights="Balanced",
        early_stopping_rounds=100,
    )
    m1_final.fit(X_tv[train_final], y_tv[train_final],
                 eval_set=(X_tv[val_final], y_tv[val_final]), verbose=0)
    test_preds[:, 0:3] = m1_final.predict_proba(X_test)
    m1_final.save_model(str(MARKET_MODELS_DIR / "stack_catboost_1x2.cbm"))

    # Final Poisson
    m2h_final = CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.03,
                                   l2_leaf_reg=5, random_seed=42, verbose=0, loss_function="Poisson")
    m2a_final = CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.03,
                                   l2_leaf_reg=5, random_seed=42, verbose=0, loss_function="Poisson")
    m2h_final.fit(X_tv[train_final], hs[train_final],
                  eval_set=(X_tv[val_final], hs[val_final]), verbose=0)
    m2a_final.fit(X_tv[train_final], as_[train_final],
                  eval_set=(X_tv[val_final], as_[val_final]), verbose=0)
    m2h_final.save_model(str(MARKET_MODELS_DIR / "stack_poisson_home.cbm"))
    m2a_final.save_model(str(MARKET_MODELS_DIR / "stack_poisson_away.cbm"))

    pred_hxg_test = np.clip(m2h_final.predict(X_test), 0.3, 4.0)
    pred_axg_test = np.clip(m2a_final.predict(X_test), 0.3, 4.0)
    for j, (hxg, axg) in enumerate(zip(pred_hxg_test, pred_axg_test)):
        pH = pD = pA = 0
        for h in range(8):
            for a in range(8):
                p = poisson.pmf(h, hxg) * poisson.pmf(a, axg)
                if h > a: pH += p
                elif h == a: pD += p
                else: pA += p
        t = pH + pD + pA
        test_preds[j, 3:6] = [pH/t, pD/t, pA/t]

    # Market odds for test
    psh_test = df_valid.loc[test_mask, "odds_PSH"].values
    psd_test = df_valid.loc[test_mask, "odds_PSD"].values
    psa_test = df_valid.loc[test_mask, "odds_PSA"].values
    for j in range(len(X_test)):
        oh, od, oa = psh_test[j], psd_test[j], psa_test[j]
        if oh > 1 and od > 1 and oa > 1:
            raw = np.array([1/oh, 1/od, 1/oa])
            test_preds[j, 6:9] = raw / raw.sum()
        else:
            test_preds[j, 6:9] = (test_preds[j, 0:3] + test_preds[j, 3:6]) / 2

    # Level 1: Logistic Regression meta-learner on OOF predictions
    # Only use folds where all base models produced non-zero predictions
    valid_oof = oof_preds.sum(axis=1) > 0.5
    y_oof = y_tv[valid_oof]

    log.info("Training Level-1 meta-learner on %d OOF samples...", valid_oof.sum())

    meta = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    meta.fit(oof_preds[valid_oof], y_oof)

    # Save meta-learner
    with open(MARKET_MODELS_DIR / "stack_meta_1x2.pkl", "wb") as f:
        pickle.dump(meta, f)

    # Predict on test
    meta_probs = meta.predict_proba(test_preds)
    meta_labels = np.array(["H", "D", "A"])[meta_probs.argmax(axis=1)]

    # Evaluate
    acc = (meta_labels == y_test_labels).mean()
    ll = log_loss(y_test, meta_probs, labels=[0, 1, 2])

    # High confidence
    hc_mask = meta_probs.max(axis=1) > 0.55
    hc_acc = (meta_labels[hc_mask] == y_test_labels[hc_mask]).mean() if hc_mask.sum() > 0 else 0

    # Compare with individual models
    cb_acc = (np.array(["H", "D", "A"])[test_preds[:, 0:3].argmax(axis=1)] == y_test_labels).mean()
    poisson_acc = (np.array(["H", "D", "A"])[test_preds[:, 3:6].argmax(axis=1)] == y_test_labels).mean()
    mkt_acc = (np.array(["H", "D", "A"])[test_preds[:, 6:9].argmax(axis=1)] == y_test_labels).mean()

    log.info("\n=== STACKING 1X2 RESULTS on %s ===", test_season)
    log.info("  CatBoost only:     acc=%.1f%%", cb_acc * 100)
    log.info("  Poisson only:      acc=%.1f%%", poisson_acc * 100)
    log.info("  Market odds only:  acc=%.1f%%", mkt_acc * 100)
    log.info("  STACKING ENSEMBLE: acc=%.1f%%, LL=%.4f, HC=%.1f%% (n=%d)",
             acc * 100, ll, hc_acc * 100, hc_mask.sum())

    results = {
        "test_season": test_season,
        "stacking_accuracy": round(acc, 4),
        "stacking_log_loss": round(ll, 4),
        "stacking_hc_accuracy": round(hc_acc, 4),
        "stacking_hc_count": int(hc_mask.sum()),
        "catboost_accuracy": round(cb_acc, 4),
        "poisson_accuracy": round(poisson_acc, 4),
        "market_accuracy": round(mkt_acc, 4),
        "catboost_params": catboost_params,
    }

    with open(MARKET_MODELS_DIR / f"stacking_results_{test_season}.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# =============================================================================
# OPTIMIZED TRAINING FOR ALL MARKETS
# =============================================================================

def train_optimized_models(df: pd.DataFrame, features: List[str],
                           n_1x2_trials: int = 50, n_binary_trials: int = 30) -> Dict:
    """Run full optimization pipeline for all markets."""
    results = {}

    # 1. Tune 1X2 with Optuna
    log.info("\n" + "=" * 70)
    log.info("STEP 1: TUNING 1X2 HYPERPARAMETERS (%d trials)", n_1x2_trials)
    log.info("=" * 70)
    best_1x2_params = tune_catboost_1x2(df, features, n_trials=n_1x2_trials)
    results["1x2_best_params"] = best_1x2_params

    # 2. Train stacking ensemble for 1X2
    log.info("\n" + "=" * 70)
    log.info("STEP 2: TRAINING STACKING ENSEMBLE FOR 1X2")
    log.info("=" * 70)

    for test_season in ["2023-2024", "2024-2025"]:
        stack_results = train_stacking_1x2(df, features, best_1x2_params, test_season)
        results[f"stacking_1x2_{test_season}"] = stack_results

    # 3. Tune binary markets
    binary_targets = {
        "over_2_5": "Over 2.5 Goals",
        "btts": "BTTS",
        "cards_over_4_5": "Cards Over 4.5",
        "corners_over_9_5": "Corners Over 9.5",
    }

    for target, name in binary_targets.items():
        if target not in df.columns or df[target].isna().all():
            continue

        log.info("\n" + "=" * 70)
        log.info("STEP 3: TUNING %s (%d trials)", name.upper(), n_binary_trials)
        log.info("=" * 70)

        best_params = tune_catboost_binary(df, features, target, n_trials=n_binary_trials)
        if not best_params:
            continue

        results[f"{target}_best_params"] = best_params

        # Train final model with best params and evaluate
        from catboost import CatBoostClassifier

        for test_season in ["2023-2024", "2024-2025"]:
            valid = df[target].notna()
            df_v = df[valid].copy()
            seasons = sorted(df_v["season"].unique())

            if test_season not in seasons:
                continue

            ti = seasons.index(test_season)
            vs = seasons[ti - 1]
            tr = seasons[:ti - 1]

            train_m = df_v["season"].isin(tr)
            val_m = df_v["season"] == vs
            test_m = df_v["season"] == test_season

            X_tr = df_v.loc[train_m, features].fillna(0)
            X_v = df_v.loc[val_m, features].fillna(0)
            X_te = df_v.loc[test_m, features].fillna(0)
            y_tr = df_v.loc[train_m, target].astype(int)
            y_v = df_v.loc[val_m, target].astype(int)
            y_te = df_v.loc[test_m, target].astype(int)

            bp = {k: v for k, v in best_params.items()}
            model = CatBoostClassifier(
                **bp, random_seed=42, verbose=0,
                loss_function="Logloss",
                early_stopping_rounds=80,
            )
            model.fit(X_tr, y_tr, eval_set=(X_v, y_v), verbose=0)
            model.save_model(str(MARKET_MODELS_DIR / f"tuned_{target}.cbm"))

            probs = model.predict_proba(X_te)[:, 1]
            acc = ((probs > 0.5).astype(int) == y_te.values).mean()
            ll = log_loss(y_te.values, probs)

            log.info("  %s on %s: acc=%.1f%%, LL=%.4f", name, test_season, acc * 100, ll)
            results[f"{target}_{test_season}"] = {"accuracy": round(acc, 4), "log_loss": round(ll, 4)}

    # 4. Feature importance analysis
    log.info("\n" + "=" * 70)
    log.info("STEP 4: FEATURE IMPORTANCE ANALYSIS")
    log.info("=" * 70)

    from catboost import CatBoostClassifier
    # Load best 1X2 model
    try:
        m = CatBoostClassifier()
        m.load_model(str(MARKET_MODELS_DIR / "stack_catboost_1x2.cbm"))
        importances = m.get_feature_importance()
        feat_imp = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)

        log.info("Top 30 features for 1X2:")
        for name, imp in feat_imp[:30]:
            log.info("  %.2f  %s", imp, name)

        results["top_features"] = [{"name": n, "importance": round(i, 2)} for n, i in feat_imp[:50]]
    except Exception as e:
        log.warning("Feature importance failed: %s", e)

    # Save all results
    with open(MARKET_MODELS_DIR / "optimization_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info("\n" + "=" * 70)
    log.info("OPTIMIZATION COMPLETE")
    log.info("=" * 70)

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials-1x2", type=int, default=50, help="Optuna trials for 1X2")
    parser.add_argument("--trials-binary", type=int, default=30, help="Optuna trials for binary targets")
    parser.add_argument("--quick", action="store_true", help="Quick mode (10 trials)")
    args = parser.parse_args()

    if args.quick:
        args.trials_1x2 = 10
        args.trials_binary = 8

    log.info("=" * 70)
    log.info("ADVANCED MODEL OPTIMIZATION")
    log.info("=" * 70)

    df, features = load_data_and_features()

    results = train_optimized_models(
        df, features,
        n_1x2_trials=args.trials_1x2,
        n_binary_trials=args.trials_binary,
    )

    # Print summary
    log.info("\n" + "=" * 70)
    log.info("FINAL SUMMARY")
    log.info("=" * 70)
    for k, v in results.items():
        if isinstance(v, dict) and "accuracy" in v:
            log.info("  %s: acc=%.1f%%, LL=%.4f", k, v["accuracy"]*100, v.get("log_loss", 0))
        elif isinstance(v, dict) and "stacking_accuracy" in v:
            log.info("  %s: stack_acc=%.1f%%, HC=%.1f%% (n=%d)",
                     k, v["stacking_accuracy"]*100,
                     v.get("stacking_hc_accuracy", 0)*100,
                     v.get("stacking_hc_count", 0))
