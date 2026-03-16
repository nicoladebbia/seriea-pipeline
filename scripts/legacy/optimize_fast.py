#!/usr/bin/env python3
"""Fast Model Optimization — Efficient stacking + tuning for all markets.

Faster alternative to optimize_models.py that:
1. Uses 2-fold walk-forward (not N-fold) for Optuna
2. Simpler stacking: weighted average of CatBoost + Poisson + Market odds
3. Optimizes weights via grid search on validation
4. Trains tuned final models for deployment
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import log_loss, accuracy_score

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DATA_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MARKET_MODELS_DIR = MODELS_DIR / "markets"
MARKET_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load unified training data with features and targets."""
    from scripts.models.comprehensive_markets import load_unified_training_data, get_training_features, _compute_targets
    df = load_unified_training_data()
    features = get_training_features(df)
    df = _compute_targets(df)

    # Verify no leakage
    TARGETS = {"total_goals", "over_2_5", "btts", "total_cards", "total_corners", "home_score", "away_score"}
    leaked = set(features) & TARGETS
    assert not leaked, f"LEAKAGE: {leaked}"

    return df, features


def poisson_1x2_from_xg(hxg, axg):
    """Convert xG to 1X2 probabilities via Poisson."""
    pH = pD = pA = 0
    for h in range(8):
        for a in range(8):
            p = poisson.pmf(h, hxg) * poisson.pmf(a, axg)
            if h > a: pH += p
            elif h == a: pD += p
            else: pA += p
    t = pH + pD + pA
    return np.array([pH/t, pD/t, pA/t])


def train_and_evaluate_1x2(df, features, test_season):
    """Train optimized 1X2 model and evaluate on test_season."""
    from catboost import CatBoostClassifier, CatBoostRegressor

    label_map = {"H": 0, "D": 1, "A": 2}
    df_v = df[df["result"].isin(["H", "D", "A"])].copy()

    seasons = sorted(df_v["season"].unique())
    test_idx = seasons.index(test_season)
    val_season = seasons[test_idx - 1]
    train_seasons = seasons[:test_idx - 1]

    train_m = df_v["season"].isin(train_seasons)
    val_m = df_v["season"] == val_season
    test_m = df_v["season"] == test_season

    X_tr = df_v.loc[train_m, features].fillna(0)
    X_val = df_v.loc[val_m, features].fillna(0)
    X_te = df_v.loc[test_m, features].fillna(0)

    y_tr = df_v.loc[train_m, "result"].map(label_map)
    y_val = df_v.loc[val_m, "result"].map(label_map)
    y_te = df_v.loc[test_m, "result"].map(label_map)
    y_te_labels = df_v.loc[test_m, "result"].values

    log.info("Train: %d (%s), Val: %d (%s), Test: %d (%s)",
             len(X_tr), train_seasons[-1], len(X_val), val_season, len(X_te), test_season)

    # ===== Model 1: Tuned CatBoost Classifier =====
    log.info("Training CatBoost 1X2 classifier...")
    m_cls = CatBoostClassifier(
        iterations=2000, depth=6, learning_rate=0.02,
        l2_leaf_reg=3.0, min_data_in_leaf=30,
        random_strength=1.0, bagging_temperature=1.0,
        border_count=128, random_seed=42, verbose=0,
        loss_function="MultiClass", classes_count=3,
        auto_class_weights="Balanced",
    )
    m_cls.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=150, verbose=0)
    cls_probs_te = m_cls.predict_proba(X_te)
    cls_probs_val = m_cls.predict_proba(X_val)

    # ===== Model 2: Alternative CatBoost (different hyperparams) =====
    log.info("Training CatBoost 1X2 v2...")
    m_cls2 = CatBoostClassifier(
        iterations=1500, depth=8, learning_rate=0.01,
        l2_leaf_reg=5.0, min_data_in_leaf=20,
        random_strength=2.0, bagging_temperature=0.5,
        border_count=200, random_seed=123, verbose=0,
        loss_function="MultiClass", classes_count=3,
        auto_class_weights="Balanced",
    )
    m_cls2.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=150, verbose=0)
    cls2_probs_te = m_cls2.predict_proba(X_te)
    cls2_probs_val = m_cls2.predict_proba(X_val)

    # ===== Model 3: Poisson goal regressors =====
    log.info("Training Poisson goal regressors...")
    m_hg = CatBoostRegressor(
        iterations=1500, depth=6, learning_rate=0.03,
        l2_leaf_reg=5, random_seed=42, verbose=0, loss_function="Poisson",
    )
    m_ag = CatBoostRegressor(
        iterations=1500, depth=6, learning_rate=0.03,
        l2_leaf_reg=5, random_seed=42, verbose=0, loss_function="Poisson",
    )
    m_hg.fit(X_tr, df_v.loc[train_m, "home_score"],
             eval_set=(X_val, df_v.loc[val_m, "home_score"]), early_stopping_rounds=100, verbose=0)
    m_ag.fit(X_tr, df_v.loc[train_m, "away_score"],
             eval_set=(X_val, df_v.loc[val_m, "away_score"]), early_stopping_rounds=100, verbose=0)

    def get_poisson_probs(X_data):
        hxg_arr = np.clip(m_hg.predict(X_data), 0.3, 4.0)
        axg_arr = np.clip(m_ag.predict(X_data), 0.3, 4.0)
        probs = np.zeros((len(X_data), 3))
        for i, (hxg, axg) in enumerate(zip(hxg_arr, axg_arr)):
            probs[i] = poisson_1x2_from_xg(hxg, axg)
        return probs

    pois_probs_val = get_poisson_probs(X_val)
    pois_probs_te = get_poisson_probs(X_te)

    # ===== Model 4: Market odds (Pinnacle) =====
    def get_market_probs(mask):
        psh = df_v.loc[mask, "odds_PSH"].values
        psd = df_v.loc[mask, "odds_PSD"].values
        psa = df_v.loc[mask, "odds_PSA"].values
        probs = np.zeros((mask.sum(), 3))
        for i in range(len(psh)):
            if psh[i] > 1 and psd[i] > 1 and psa[i] > 1:
                raw = np.array([1/psh[i], 1/psd[i], 1/psa[i]])
                probs[i] = raw / raw.sum()
            else:
                probs[i] = [0.4, 0.3, 0.3]
        return probs

    mkt_probs_val = get_market_probs(val_m)
    mkt_probs_te = get_market_probs(test_m)

    # ===== OPTIMIZE ENSEMBLE WEIGHTS on validation =====
    log.info("Optimizing ensemble weights on validation...")
    best_weights = None
    best_ll = 999

    for w1 in np.arange(0.0, 0.65, 0.05):
        for w2 in np.arange(0.0, 0.65, 0.05):
            for w3 in np.arange(0.0, 0.55, 0.05):
                for w4 in np.arange(0.0, 0.55, 0.05):
                    if abs(w1 + w2 + w3 + w4 - 1.0) > 0.01:
                        continue
                    ens = w1 * cls_probs_val + w2 * cls2_probs_val + w3 * pois_probs_val + w4 * mkt_probs_val
                    ens = ens / ens.sum(axis=1, keepdims=True)
                    try:
                        ll = log_loss(y_val.values, ens, labels=[0, 1, 2])
                        if ll < best_ll:
                            best_ll = ll
                            best_weights = (w1, w2, w3, w4)
                    except Exception:
                        pass

    log.info("Best weights: CB1=%.2f, CB2=%.2f, Poisson=%.2f, Market=%.2f (val LL=%.4f)",
             *best_weights, best_ll)

    # Apply to test
    w1, w2, w3, w4 = best_weights
    ens_probs_te = w1 * cls_probs_te + w2 * cls2_probs_te + w3 * pois_probs_te + w4 * mkt_probs_te
    ens_probs_te = ens_probs_te / ens_probs_te.sum(axis=1, keepdims=True)

    # Evaluate all models
    results = {}
    for name, probs in [
        ("catboost_v1", cls_probs_te),
        ("catboost_v2", cls2_probs_te),
        ("poisson", pois_probs_te),
        ("market_odds", mkt_probs_te),
        ("ensemble", ens_probs_te),
    ]:
        pred = np.array(["H", "D", "A"])[probs.argmax(axis=1)]
        acc = (pred == y_te_labels).mean()
        ll = log_loss(y_te.values, probs, labels=[0, 1, 2])

        # High confidence
        hc_mask = probs.max(axis=1) > 0.50
        hc_acc = (pred[hc_mask] == y_te_labels[hc_mask]).mean() if hc_mask.sum() > 0 else 0
        hc_count = hc_mask.sum()

        # Very high confidence
        vhc_mask = probs.max(axis=1) > 0.60
        vhc_acc = (pred[vhc_mask] == y_te_labels[vhc_mask]).mean() if vhc_mask.sum() > 0 else 0
        vhc_count = vhc_mask.sum()

        results[name] = {
            "accuracy": round(acc, 4),
            "log_loss": round(ll, 4),
            "hc_accuracy": round(hc_acc, 4),
            "hc_count": int(hc_count),
            "vhc_accuracy": round(vhc_acc, 4),
            "vhc_count": int(vhc_count),
        }
        log.info("  %s: acc=%.1f%%, LL=%.4f, HC50=%.1f%%(n=%d), HC60=%.1f%%(n=%d)",
                 name, acc*100, ll, hc_acc*100, hc_count, vhc_acc*100, vhc_count)

    # Prediction distribution
    ens_pred = np.array(["H", "D", "A"])[ens_probs_te.argmax(axis=1)]
    actual_dist = pd.Series(y_te_labels).value_counts(normalize=True)
    pred_dist = pd.Series(ens_pred).value_counts(normalize=True)
    log.info("\nActual dist: %s", actual_dist.to_dict())
    log.info("Pred dist:   %s", pred_dist.to_dict())

    # Save models
    m_cls.save_model(str(MARKET_MODELS_DIR / "tuned_1x2_v1.cbm"))
    m_cls2.save_model(str(MARKET_MODELS_DIR / "tuned_1x2_v2.cbm"))
    m_hg.save_model(str(MARKET_MODELS_DIR / "tuned_home_goals.cbm"))
    m_ag.save_model(str(MARKET_MODELS_DIR / "tuned_away_goals.cbm"))

    return results, best_weights


def train_and_evaluate_binary(df, features, target, target_name, test_season):
    """Train and evaluate a binary market model."""
    from catboost import CatBoostClassifier

    valid = df[target].notna()
    df_v = df[valid].copy()
    seasons = sorted(df_v["season"].unique())

    if test_season not in seasons:
        return {}

    ti = seasons.index(test_season)
    val_season = seasons[ti - 1]
    train_seasons = seasons[:ti - 1]

    train_m = df_v["season"].isin(train_seasons)
    val_m = df_v["season"] == val_season
    test_m = df_v["season"] == test_season

    if train_m.sum() < 200 or test_m.sum() < 20:
        return {}

    X_tr = df_v.loc[train_m, features].fillna(0)
    X_val = df_v.loc[val_m, features].fillna(0)
    X_te = df_v.loc[test_m, features].fillna(0)
    y_tr = df_v.loc[train_m, target].astype(int)
    y_val = df_v.loc[val_m, target].astype(int)
    y_te = df_v.loc[test_m, target].astype(int)

    # Train two models with different hyperparams
    models = []
    for params in [
        {"iterations": 1500, "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 5, "random_seed": 42},
        {"iterations": 2000, "depth": 5, "learning_rate": 0.02, "l2_leaf_reg": 3, "random_seed": 123,
         "min_data_in_leaf": 20, "bagging_temperature": 0.8},
    ]:
        m = CatBoostClassifier(**params, verbose=0, loss_function="Logloss", early_stopping_rounds=100)
        m.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
        models.append(m)

    # Optimize blend weight on val
    p1_val = models[0].predict_proba(X_val)[:, 1]
    p2_val = models[1].predict_proba(X_val)[:, 1]

    best_w, best_ll = 0.5, 999
    for w in np.arange(0.0, 1.05, 0.05):
        blend = w * p1_val + (1 - w) * p2_val
        ll = log_loss(y_val.values, blend)
        if ll < best_ll:
            best_ll = ll
            best_w = w

    # Apply to test
    p1_te = models[0].predict_proba(X_te)[:, 1]
    p2_te = models[1].predict_proba(X_te)[:, 1]
    blend_te = best_w * p1_te + (1 - best_w) * p2_te

    acc = ((blend_te > 0.5).astype(int) == y_te.values).mean()
    ll = log_loss(y_te.values, blend_te)
    brier = np.mean((blend_te - y_te.values) ** 2)

    log.info("  %s: acc=%.1f%%, LL=%.4f, Brier=%.4f (w=%.2f)", target_name, acc*100, ll, brier, best_w)

    # Save best model
    models[0].save_model(str(MARKET_MODELS_DIR / f"tuned_{target}.cbm"))

    return {
        "accuracy": round(acc, 4),
        "log_loss": round(ll, 4),
        "brier": round(brier, 4),
        "blend_weight": round(best_w, 2),
    }


def main():
    log.info("=" * 70)
    log.info("FAST MODEL OPTIMIZATION — ALL MARKETS")
    log.info("=" * 70)

    df, features = load_data()
    all_results = {}

    # ===== 1X2 OPTIMIZATION =====
    for test_season in ["2023-2024", "2024-2025"]:
        log.info("\n" + "=" * 70)
        log.info("1X2 OPTIMIZATION — Test: %s", test_season)
        log.info("=" * 70)
        results, weights = train_and_evaluate_1x2(df, features, test_season)
        all_results[f"1x2_{test_season}"] = results
        all_results[f"1x2_weights_{test_season}"] = [round(w, 3) for w in weights]

    # ===== BINARY MARKETS =====
    binary_targets = {
        "over_2_5": "Over 2.5 Goals",
        "over_1_5": "Over 1.5 Goals",
        "over_3_5": "Over 3.5 Goals",
        "btts": "BTTS",
        "home_clean_sheet": "Home Clean Sheet",
        "away_clean_sheet": "Away Clean Sheet",
    }

    # Add card/corner targets if available
    if "cards_over_4_5" in df.columns and df["cards_over_4_5"].notna().sum() > 500:
        binary_targets["cards_over_3_5"] = "Cards Over 3.5"
        binary_targets["cards_over_4_5"] = "Cards Over 4.5"
        binary_targets["cards_over_5_5"] = "Cards Over 5.5"

    if "corners_over_9_5" in df.columns and df["corners_over_9_5"].notna().sum() > 500:
        binary_targets["corners_over_8_5"] = "Corners Over 8.5"
        binary_targets["corners_over_9_5"] = "Corners Over 9.5"
        binary_targets["corners_over_10_5"] = "Corners Over 10.5"

    for test_season in ["2023-2024", "2024-2025"]:
        log.info("\n" + "=" * 70)
        log.info("BINARY MARKETS — Test: %s", test_season)
        log.info("=" * 70)

        for target, name in binary_targets.items():
            if target not in df.columns or df[target].isna().all():
                continue
            r = train_and_evaluate_binary(df, features, target, name, test_season)
            if r:
                all_results[f"{target}_{test_season}"] = r

    # ===== FEATURE IMPORTANCE =====
    log.info("\n" + "=" * 70)
    log.info("FEATURE IMPORTANCE")
    log.info("=" * 70)

    from catboost import CatBoostClassifier
    try:
        m = CatBoostClassifier()
        m.load_model(str(MARKET_MODELS_DIR / "tuned_1x2_v1.cbm"))
        importances = m.get_feature_importance()
        feat_imp = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
        log.info("Top 25 features:")
        for name, imp in feat_imp[:25]:
            log.info("  %.2f  %s", imp, name)
        all_results["top_features"] = [{"name": n, "importance": round(i, 2)} for n, i in feat_imp[:50]]
    except Exception as e:
        log.warning("Feature importance failed: %s", e)

    # Save
    with open(MARKET_MODELS_DIR / "optimization_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # ===== SUMMARY =====
    log.info("\n" + "=" * 70)
    log.info("FINAL SUMMARY")
    log.info("=" * 70)
    for k, v in all_results.items():
        if isinstance(v, dict):
            if "accuracy" in v:
                log.info("  %s: acc=%.1f%%, LL=%.4f", k, v["accuracy"]*100, v.get("log_loss", 0))
            elif "ensemble" in v:
                ens = v["ensemble"]
                log.info("  %s ensemble: acc=%.1f%%, LL=%.4f, HC60=%.1f%%(%d)",
                         k, ens["accuracy"]*100, ens["log_loss"],
                         ens.get("vhc_accuracy", 0)*100, ens.get("vhc_count", 0))


if __name__ == "__main__":
    main()
