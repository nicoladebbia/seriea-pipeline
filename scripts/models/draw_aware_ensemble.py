#!/usr/bin/env python3
"""Draw-Aware Ensemble — Fix 1X2 accuracy by properly predicting draws.

Problem: Current ensemble predicts 0-3% draws vs 29% actual, destroying accuracy.
Solution: 
1. Train binary draw detector (draw vs not-draw)
2. Use draw probability to calibrate ensemble output
3. Implement threshold-based draw selection
4. Combine with H/A predictions for final 1X2

Target: Push from 54% → 60%+ overall accuracy.
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import log_loss, accuracy_score
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MARKET_MODELS_DIR = MODELS_DIR / "markets"


def load_data():
    from scripts.models.comprehensive_markets import load_unified_training_data, get_training_features, _compute_targets
    df = load_unified_training_data()
    features = get_training_features(df)
    df = _compute_targets(df)
    return df, features


def poisson_1x2(hxg, axg):
    pH = pD = pA = 0
    for h in range(8):
        for a in range(8):
            p = poisson.pmf(h, hxg) * poisson.pmf(a, axg)
            if h > a: pH += p
            elif h == a: pD += p
            else: pA += p
    t = pH + pD + pA
    return np.array([pH/t, pD/t, pA/t])


def build_draw_aware_ensemble(df, features, test_season):
    """Build a draw-aware 1X2 ensemble for a given test season."""
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

    y_tr = df_v.loc[train_m, "result"]
    y_val = df_v.loc[val_m, "result"]
    y_te = df_v.loc[test_m, "result"]

    y_tr_num = y_tr.map(label_map)
    y_val_num = y_val.map(label_map)
    y_te_num = y_te.map(label_map)

    log.info("Train: %d, Val: %d (%s), Test: %d (%s)",
             len(X_tr), len(X_val), val_season, len(X_te), test_season)

    # ===== COMPONENT 1: 1X2 Classifier =====
    log.info("Training 1X2 classifier...")
    m_cls = CatBoostClassifier(
        iterations=2000, depth=6, learning_rate=0.02,
        l2_leaf_reg=3.0, min_data_in_leaf=30,
        random_strength=1.0, bagging_temperature=1.0,
        random_seed=42, verbose=0,
        loss_function="MultiClass", classes_count=3,
        auto_class_weights="Balanced",
    )
    m_cls.fit(X_tr, y_tr_num, eval_set=(X_val, y_val_num),
              early_stopping_rounds=150, verbose=0)
    cls_probs_val = m_cls.predict_proba(X_val)
    cls_probs_te = m_cls.predict_proba(X_te)

    # ===== COMPONENT 2: Poisson goal regressors =====
    log.info("Training Poisson regressors...")
    m_hg = CatBoostRegressor(iterations=1500, depth=6, learning_rate=0.03,
                              l2_leaf_reg=5, random_seed=42, verbose=0, loss_function="Poisson")
    m_ag = CatBoostRegressor(iterations=1500, depth=6, learning_rate=0.03,
                              l2_leaf_reg=5, random_seed=42, verbose=0, loss_function="Poisson")
    m_hg.fit(X_tr, df_v.loc[train_m, "home_score"],
             eval_set=(X_val, df_v.loc[val_m, "home_score"]), early_stopping_rounds=100, verbose=0)
    m_ag.fit(X_tr, df_v.loc[train_m, "away_score"],
             eval_set=(X_val, df_v.loc[val_m, "away_score"]), early_stopping_rounds=100, verbose=0)

    def get_poisson_probs(X):
        hxg = np.clip(m_hg.predict(X), 0.3, 4.0)
        axg = np.clip(m_ag.predict(X), 0.3, 4.0)
        probs = np.zeros((len(X), 3))
        for i in range(len(X)):
            probs[i] = poisson_1x2(hxg[i], axg[i])
        return probs

    pois_probs_val = get_poisson_probs(X_val)
    pois_probs_te = get_poisson_probs(X_te)

    # ===== COMPONENT 3: Market odds =====
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

    # ===== COMPONENT 4: Binary Draw Detector =====
    log.info("Training dedicated draw detector...")
    y_draw_tr = (y_tr == "D").astype(int)
    y_draw_val = (y_val == "D").astype(int)
    y_draw_te = (y_te == "D").astype(int)

    m_draw = CatBoostClassifier(
        iterations=1500, depth=6, learning_rate=0.03,
        l2_leaf_reg=5, random_seed=42, verbose=0,
        loss_function="Logloss",
        auto_class_weights="Balanced",
    )
    m_draw.fit(X_tr, y_draw_tr, eval_set=(X_val, y_draw_val),
               early_stopping_rounds=100, verbose=0)
    draw_prob_val = m_draw.predict_proba(X_val)[:, 1]
    draw_prob_te = m_draw.predict_proba(X_te)[:, 1]

    draw_acc_val = ((draw_prob_val > 0.5).astype(int) == y_draw_val.values).mean()
    log.info("  Draw detector val accuracy: %.1f%%", draw_acc_val * 100)

    # ===== COMPONENT 5: Binary Home Win Detector =====
    log.info("Training home win detector...")
    y_home_tr = (y_tr == "H").astype(int)
    y_home_val = (y_val == "H").astype(int)

    m_home = CatBoostClassifier(
        iterations=1500, depth=6, learning_rate=0.03,
        l2_leaf_reg=5, random_seed=42, verbose=0,
        loss_function="Logloss",
        auto_class_weights="Balanced",
    )
    m_home.fit(X_tr, y_home_tr, eval_set=(X_val, y_home_val),
               early_stopping_rounds=100, verbose=0)
    home_prob_val = m_home.predict_proba(X_val)[:, 1]
    home_prob_te = m_home.predict_proba(X_te)[:, 1]

    # ===== ENSEMBLE CALIBRATION ON VALIDATION =====
    log.info("Calibrating draw-aware ensemble on validation...")

    # Strategy: Build calibrated ensemble probabilities
    # P(H), P(D), P(A) from multiple sources, then optimize decision rules

    best_acc = 0
    best_config = None

    # Grid search over:
    # - Base ensemble weights (CB, Poisson, Market)
    # - Draw boost factor (how much to boost draw probability)
    # - Draw threshold (minimum P(D) to predict draw)

    for w_cb in np.arange(0.1, 0.6, 0.1):
        for w_pois in np.arange(0.0, 0.4, 0.1):
            w_mkt = round(1.0 - w_cb - w_pois, 2)
            if w_mkt < 0 or w_mkt > 0.7:
                continue

            # Base ensemble probs
            base_val = w_cb * cls_probs_val + w_pois * pois_probs_val + w_mkt * mkt_probs_val
            base_val = base_val / base_val.sum(axis=1, keepdims=True)

            for draw_weight in np.arange(0.0, 0.6, 0.05):
                for draw_threshold in np.arange(0.24, 0.40, 0.02):
                    # Inject draw detector probability
                    adjusted = base_val.copy()
                    adjusted[:, 1] = (1 - draw_weight) * base_val[:, 1] + draw_weight * draw_prob_val

                    # Renormalize
                    adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)

                    # Apply draw threshold: if P(D) > threshold, predict draw
                    preds = np.array(["H", "D", "A"])[adjusted.argmax(axis=1)]

                    # Override with draw when draw_prob exceeds threshold
                    # AND it's not already the argmax
                    for i in range(len(preds)):
                        if adjusted[i, 1] > draw_threshold:
                            preds[i] = "D"
                        elif draw_prob_val[i] > draw_threshold + 0.05:
                            preds[i] = "D"

                    acc = (preds == y_val.values).mean()
                    if acc > best_acc:
                        best_acc = acc
                        best_config = {
                            "w_cb": w_cb, "w_pois": w_pois, "w_mkt": w_mkt,
                            "draw_weight": draw_weight, "draw_threshold": draw_threshold,
                        }

    log.info("Best val config: acc=%.1f%% | %s", best_acc * 100, best_config)

    # ===== APPLY BEST CONFIG TO TEST =====
    cfg = best_config
    base_te = cfg["w_cb"] * cls_probs_te + cfg["w_pois"] * pois_probs_te + cfg["w_mkt"] * mkt_probs_te
    base_te = base_te / base_te.sum(axis=1, keepdims=True)

    adjusted_te = base_te.copy()
    adjusted_te[:, 1] = (1 - cfg["draw_weight"]) * base_te[:, 1] + cfg["draw_weight"] * draw_prob_te
    adjusted_te = adjusted_te / adjusted_te.sum(axis=1, keepdims=True)

    preds_te = np.array(["H", "D", "A"])[adjusted_te.argmax(axis=1)]
    for i in range(len(preds_te)):
        if adjusted_te[i, 1] > cfg["draw_threshold"]:
            preds_te[i] = "D"
        elif draw_prob_te[i] > cfg["draw_threshold"] + 0.05:
            preds_te[i] = "D"

    # ===== EVALUATE =====
    acc = (preds_te == y_te.values).mean()
    
    # Also evaluate without draw fix for comparison
    no_draw_preds = np.array(["H", "D", "A"])[base_te.argmax(axis=1)]
    no_draw_acc = (no_draw_preds == y_te.values).mean()

    # Market baseline
    mkt_preds = np.array(["H", "D", "A"])[mkt_probs_te.argmax(axis=1)]
    mkt_acc = (mkt_preds == y_te.values).mean()

    # High confidence
    max_prob = adjusted_te.max(axis=1)
    hc50 = max_prob > 0.50
    hc55 = max_prob > 0.55
    hc60 = max_prob > 0.60

    hc50_acc = (preds_te[hc50] == y_te.values[hc50]).mean() if hc50.sum() > 0 else 0
    hc55_acc = (preds_te[hc55] == y_te.values[hc55]).mean() if hc55.sum() > 0 else 0
    hc60_acc = (preds_te[hc60] == y_te.values[hc60]).mean() if hc60.sum() > 0 else 0

    # Prediction distribution
    pred_dist = pd.Series(preds_te).value_counts(normalize=True)
    actual_dist = pd.Series(y_te.values).value_counts(normalize=True)

    log.info("\n=== DRAW-AWARE ENSEMBLE RESULTS: %s ===", test_season)
    log.info("  Without draw fix: acc=%.1f%%", no_draw_acc * 100)
    log.info("  Market odds:      acc=%.1f%%", mkt_acc * 100)
    log.info("  DRAW-AWARE:       acc=%.1f%%", acc * 100)
    log.info("  HC50: %.1f%% (n=%d)", hc50_acc * 100, hc50.sum())
    log.info("  HC55: %.1f%% (n=%d)", hc55_acc * 100, hc55.sum())
    log.info("  HC60: %.1f%% (n=%d)", hc60_acc * 100, hc60.sum())
    log.info("  Pred dist: %s", pred_dist.to_dict())
    log.info("  Actual dist: %s", actual_dist.to_dict())

    # Per-class accuracy
    for cls in ["H", "D", "A"]:
        mask = y_te.values == cls
        cls_acc = (preds_te[mask] == cls).mean() if mask.sum() > 0 else 0
        log.info("  Recall %s: %.1f%% (%d/%d)", cls, cls_acc * 100,
                 (preds_te[mask] == cls).sum(), mask.sum())

    # Draw precision
    draw_pred_mask = preds_te == "D"
    if draw_pred_mask.sum() > 0:
        draw_precision = (y_te.values[draw_pred_mask] == "D").mean()
        log.info("  Draw precision: %.1f%% (%d predicted, %d correct)",
                 draw_precision * 100, draw_pred_mask.sum(),
                 (y_te.values[draw_pred_mask] == "D").sum())

    # Save models
    m_cls.save_model(str(MARKET_MODELS_DIR / f"draw_aware_cls_{test_season}.cbm"))
    m_draw.save_model(str(MARKET_MODELS_DIR / f"draw_detector_{test_season}.cbm"))
    m_home.save_model(str(MARKET_MODELS_DIR / f"home_detector_{test_season}.cbm"))
    m_hg.save_model(str(MARKET_MODELS_DIR / f"poisson_home_{test_season}.cbm"))
    m_ag.save_model(str(MARKET_MODELS_DIR / f"poisson_away_{test_season}.cbm"))

    results = {
        "test_season": test_season,
        "accuracy": round(acc, 4),
        "no_draw_fix_accuracy": round(no_draw_acc, 4),
        "market_accuracy": round(mkt_acc, 4),
        "improvement_vs_no_draw": round((acc - no_draw_acc) * 100, 2),
        "improvement_vs_market": round((acc - mkt_acc) * 100, 2),
        "hc50": {"accuracy": round(hc50_acc, 4), "count": int(hc50.sum())},
        "hc55": {"accuracy": round(hc55_acc, 4), "count": int(hc55.sum())},
        "hc60": {"accuracy": round(hc60_acc, 4), "count": int(hc60.sum())},
        "config": {k: round(v, 4) if isinstance(v, float) else v for k, v in cfg.items()},
        "pred_distribution": pred_dist.to_dict(),
        "actual_distribution": actual_dist.to_dict(),
    }

    return results


def main():
    log.info("=" * 70)
    log.info("DRAW-AWARE ENSEMBLE — FIXING 1X2 ACCURACY")
    log.info("=" * 70)

    df, features = load_data()
    all_results = {}

    for test_season in ["2023-2024", "2024-2025"]:
        log.info("\n" + "=" * 70)
        log.info("TESTING: %s", test_season)
        log.info("=" * 70)
        results = build_draw_aware_ensemble(df, features, test_season)
        all_results[test_season] = results

    # Save
    with open(MARKET_MODELS_DIR / "draw_aware_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Final summary
    log.info("\n" + "=" * 70)
    log.info("FINAL SUMMARY")
    log.info("=" * 70)
    for season, r in all_results.items():
        log.info("  %s: acc=%.1f%% (was %.1f%% without draw fix, market=%.1f%%)",
                 season, r["accuracy"]*100, r["no_draw_fix_accuracy"]*100, r["market_accuracy"]*100)
        log.info("    Improvement: +%.1fpp vs no-draw, +%.1fpp vs market",
                 r["improvement_vs_no_draw"], r["improvement_vs_market"])
        log.info("    HC50=%.1f%%(%d), HC55=%.1f%%(%d), HC60=%.1f%%(%d)",
                 r["hc50"]["accuracy"]*100, r["hc50"]["count"],
                 r["hc55"]["accuracy"]*100, r["hc55"]["count"],
                 r["hc60"]["accuracy"]*100, r["hc60"]["count"])


if __name__ == "__main__":
    main()
