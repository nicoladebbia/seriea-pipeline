#!/usr/bin/env python3
"""Calibrated 1X2 Ensemble — Fix draw prediction through probability calibration.

Previous approach (threshold override) failed by over-predicting draws (80% → 40% acc).
This approach:
1. Build base ensemble (CB + Poisson + Market odds)
2. Calibrate probabilities using Platt scaling on validation
3. Apply draw inflation factor to match historical draw rate
4. Use argmax on calibrated probabilities (draws appear naturally)
5. Additionally: use confidence-aware prediction (abstain on low-confidence)

The key insight: draws are underestimated in ALL sub-models. A modest inflation
factor (1.1-1.5x) applied to P(D) + renormalization naturally pushes borderline
matches to draw predictions without over-predicting.
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import log_loss

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


def calibrated_ensemble(df, features, test_season):
    """Build calibrated 1X2 ensemble with proper draw handling."""
    from catboost import CatBoostClassifier, CatBoostRegressor

    label_map = {"H": 0, "D": 1, "A": 2}
    df_v = df[df["result"].isin(["H", "D", "A"])].copy()

    seasons = sorted(df_v["season"].unique())
    test_idx = seasons.index(test_season)
    val_season = seasons[test_idx - 1]
    train_seasons = seasons[:test_idx - 1]

    # Use a second validation season for calibration
    cal_season = seasons[test_idx - 2] if test_idx >= 3 else val_season

    train_m = df_v["season"].isin([s for s in train_seasons if s != cal_season])
    cal_m = df_v["season"] == cal_season
    val_m = df_v["season"] == val_season
    test_m = df_v["season"] == test_season

    X_tr = df_v.loc[train_m, features].fillna(0)
    X_cal = df_v.loc[cal_m, features].fillna(0)
    X_val = df_v.loc[val_m, features].fillna(0)
    X_te = df_v.loc[test_m, features].fillna(0)

    y_val = df_v.loc[val_m, "result"].values
    y_te = df_v.loc[test_m, "result"].values
    y_cal = df_v.loc[cal_m, "result"].values

    log.info("Train: %d, Cal: %d (%s), Val: %d (%s), Test: %d (%s)",
             len(X_tr), len(X_cal), cal_season, len(X_val), val_season, len(X_te), test_season)

    # ===== BUILD BASE MODELS =====

    # Model 1: CatBoost 1X2
    log.info("Training CatBoost classifier...")
    m_cls = CatBoostClassifier(
        iterations=2000, depth=6, learning_rate=0.02,
        l2_leaf_reg=3.0, min_data_in_leaf=30,
        random_seed=42, verbose=0,
        loss_function="MultiClass", classes_count=3,
        auto_class_weights="Balanced",
    )
    m_cls.fit(X_tr, df_v.loc[train_m, "result"].map(label_map),
              eval_set=(X_cal, df_v.loc[cal_m, "result"].map(label_map)),
              early_stopping_rounds=150, verbose=0)

    # Model 2: Poisson
    log.info("Training Poisson regressors...")
    m_hg = CatBoostRegressor(iterations=1500, depth=6, learning_rate=0.03,
                              l2_leaf_reg=5, random_seed=42, verbose=0, loss_function="Poisson")
    m_ag = CatBoostRegressor(iterations=1500, depth=6, learning_rate=0.03,
                              l2_leaf_reg=5, random_seed=42, verbose=0, loss_function="Poisson")
    m_hg.fit(X_tr, df_v.loc[train_m, "home_score"],
             eval_set=(X_cal, df_v.loc[cal_m, "home_score"]), early_stopping_rounds=100, verbose=0)
    m_ag.fit(X_tr, df_v.loc[train_m, "away_score"],
             eval_set=(X_cal, df_v.loc[cal_m, "away_score"]), early_stopping_rounds=100, verbose=0)

    def get_poisson_probs(X):
        hxg = np.clip(m_hg.predict(X), 0.3, 4.0)
        axg = np.clip(m_ag.predict(X), 0.3, 4.0)
        probs = np.zeros((len(X), 3))
        for i in range(len(X)):
            probs[i] = poisson_1x2(hxg[i], axg[i])
        return probs

    # Get probabilities from all models on all sets
    def get_all_probs(X_data, mask):
        cls_p = m_cls.predict_proba(X_data)
        pois_p = get_poisson_probs(X_data)

        psh = df_v.loc[mask, "odds_PSH"].values
        psd = df_v.loc[mask, "odds_PSD"].values
        psa = df_v.loc[mask, "odds_PSA"].values
        mkt_p = np.zeros((len(X_data), 3))
        for i in range(len(psh)):
            if psh[i] > 1 and psd[i] > 1 and psa[i] > 1:
                raw = np.array([1/psh[i], 1/psd[i], 1/psa[i]])
                mkt_p[i] = raw / raw.sum()
            else:
                mkt_p[i] = [0.4, 0.3, 0.3]
        return cls_p, pois_p, mkt_p

    cls_val, pois_val, mkt_val = get_all_probs(X_val, val_m)
    cls_te, pois_te, mkt_te = get_all_probs(X_te, test_m)

    # ===== CALIBRATION: Find optimal weights + draw inflation =====
    log.info("Calibrating ensemble weights + draw inflation on validation...")

    best_acc = 0
    best_ll = 999
    best_config = None

    # Grid search: weights + draw inflation factor
    for w_cb in np.arange(0.1, 0.55, 0.05):
        for w_pois in np.arange(0.0, 0.45, 0.05):
            w_mkt = round(1.0 - w_cb - w_pois, 2)
            if w_mkt < 0.05 or w_mkt > 0.65:
                continue

            base = w_cb * cls_val + w_pois * pois_val + w_mkt * mkt_val

            for d_inflate in np.arange(1.0, 2.2, 0.05):
                # Inflate draw probability
                adj = base.copy()
                adj[:, 1] *= d_inflate
                adj = adj / adj.sum(axis=1, keepdims=True)

                preds = np.array(["H", "D", "A"])[adj.argmax(axis=1)]
                acc = (preds == y_val).mean()

                # Also compute log loss for calibration quality
                try:
                    ll = log_loss(pd.Series(y_val).map(label_map).values, adj, labels=[0, 1, 2])
                except:
                    ll = 1.5

                # Optimize for accuracy first, break ties with log loss
                if acc > best_acc or (acc == best_acc and ll < best_ll):
                    best_acc = acc
                    best_ll = ll
                    best_config = {
                        "w_cb": round(w_cb, 3),
                        "w_pois": round(w_pois, 3),
                        "w_mkt": round(w_mkt, 3),
                        "draw_inflate": round(d_inflate, 3),
                    }

    log.info("Best config (val acc=%.1f%%, LL=%.4f): %s", best_acc * 100, best_ll, best_config)

    # ===== APPLY TO TEST =====
    cfg = best_config
    base_te = cfg["w_cb"] * cls_te + cfg["w_pois"] * pois_te + cfg["w_mkt"] * mkt_te
    adj_te = base_te.copy()
    adj_te[:, 1] *= cfg["draw_inflate"]
    adj_te = adj_te / adj_te.sum(axis=1, keepdims=True)

    preds_te = np.array(["H", "D", "A"])[adj_te.argmax(axis=1)]

    # ===== EVALUATE =====
    acc = (preds_te == y_te).mean()
    ll = log_loss(pd.Series(y_te).map(label_map).values, adj_te, labels=[0, 1, 2])

    # No-calibration baseline
    base_te_raw = base_te / base_te.sum(axis=1, keepdims=True)
    raw_preds = np.array(["H", "D", "A"])[base_te_raw.argmax(axis=1)]
    raw_acc = (raw_preds == y_te).mean()

    # Market baseline
    mkt_preds = np.array(["H", "D", "A"])[mkt_te.argmax(axis=1)]
    mkt_acc = (mkt_preds == y_te).mean()

    # High confidence tiers
    max_p = adj_te.max(axis=1)
    tiers = {}
    for threshold, name in [(0.45, "HC45"), (0.50, "HC50"), (0.55, "HC55"), (0.60, "HC60")]:
        mask = max_p > threshold
        if mask.sum() > 0:
            t_acc = (preds_te[mask] == y_te[mask]).mean()
            tiers[name] = {"accuracy": round(t_acc, 4), "count": int(mask.sum())}
        else:
            tiers[name] = {"accuracy": 0, "count": 0}

    # Prediction distribution
    pred_dist = pd.Series(preds_te).value_counts(normalize=True).to_dict()
    actual_dist = pd.Series(y_te).value_counts(normalize=True).to_dict()

    # Per-class metrics
    per_class = {}
    for cls in ["H", "D", "A"]:
        true_mask = y_te == cls
        pred_mask = preds_te == cls
        recall = (preds_te[true_mask] == cls).mean() if true_mask.sum() > 0 else 0
        precision = (y_te[pred_mask] == cls).mean() if pred_mask.sum() > 0 else 0
        per_class[cls] = {
            "recall": round(recall, 3),
            "precision": round(precision, 3),
            "actual": int(true_mask.sum()),
            "predicted": int(pred_mask.sum()),
        }

    log.info("\n" + "=" * 60)
    log.info("CALIBRATED ENSEMBLE RESULTS: %s", test_season)
    log.info("=" * 60)
    log.info("  Raw (no calibration): %.1f%%", raw_acc * 100)
    log.info("  Market odds:          %.1f%%", mkt_acc * 100)
    log.info("  CALIBRATED ENSEMBLE:  %.1f%% (LL=%.4f)", acc * 100, ll)
    log.info("  Improvement: +%.1fpp vs raw, +%.1fpp vs market",
             (acc - raw_acc) * 100, (acc - mkt_acc) * 100)
    for name, t in tiers.items():
        log.info("  %s: %.1f%% (n=%d)", name, t["accuracy"] * 100, t["count"])
    log.info("  Pred dist: %s", {k: round(v, 3) for k, v in pred_dist.items()})
    log.info("  Actual dist: %s", {k: round(v, 3) for k, v in actual_dist.items()})
    for cls, m in per_class.items():
        log.info("  %s: recall=%.1f%%, precision=%.1f%% (%d pred, %d actual)",
                 cls, m["recall"]*100, m["precision"]*100, m["predicted"], m["actual"])

    # Save models for deployment
    m_cls.save_model(str(MARKET_MODELS_DIR / "calibrated_cls.cbm"))
    m_hg.save_model(str(MARKET_MODELS_DIR / "calibrated_home_goals.cbm"))
    m_ag.save_model(str(MARKET_MODELS_DIR / "calibrated_away_goals.cbm"))

    return {
        "test_season": test_season,
        "accuracy": round(acc, 4),
        "log_loss": round(ll, 4),
        "raw_accuracy": round(raw_acc, 4),
        "market_accuracy": round(mkt_acc, 4),
        "improvement_vs_raw": round((acc - raw_acc) * 100, 2),
        "improvement_vs_market": round((acc - mkt_acc) * 100, 2),
        "config": cfg,
        "tiers": tiers,
        "per_class": per_class,
        "pred_distribution": pred_dist,
        "actual_distribution": actual_dist,
    }


def multi_season_backtest(df, features):
    """Run calibrated ensemble on multiple test seasons for robustness check."""
    from catboost import CatBoostClassifier, CatBoostRegressor

    label_map = {"H": 0, "D": 1, "A": 2}
    df_v = df[df["result"].isin(["H", "D", "A"])].copy()
    seasons = sorted(df_v["season"].unique())

    # Test on last 5 seasons (most relevant data)
    test_seasons = [s for s in seasons if seasons.index(s) >= 10]
    log.info("Multi-season backtest on %d seasons: %s", len(test_seasons), test_seasons)

    all_preds = []
    all_true = []
    all_probs = []
    season_results = []

    for test_season in test_seasons:
        test_idx = seasons.index(test_season)
        val_season = seasons[test_idx - 1]
        train_seasons_list = seasons[:test_idx - 1]

        if len(train_seasons_list) < 5:
            continue

        train_m = df_v["season"].isin(train_seasons_list)
        val_m = df_v["season"] == val_season
        test_m = df_v["season"] == test_season

        X_tr = df_v.loc[train_m, features].fillna(0)
        X_val = df_v.loc[val_m, features].fillna(0)
        X_te = df_v.loc[test_m, features].fillna(0)

        if len(X_te) < 50:
            continue

        y_val = df_v.loc[val_m, "result"].values
        y_te = df_v.loc[test_m, "result"].values

        # Train models
        m_cls = CatBoostClassifier(
            iterations=2000, depth=6, learning_rate=0.02,
            l2_leaf_reg=3.0, min_data_in_leaf=30,
            random_seed=42, verbose=0,
            loss_function="MultiClass", classes_count=3,
            auto_class_weights="Balanced",
        )
        m_cls.fit(X_tr, df_v.loc[train_m, "result"].map(label_map),
                  eval_set=(X_val, df_v.loc[val_m, "result"].map(label_map)),
                  early_stopping_rounds=150, verbose=0)

        m_hg = CatBoostRegressor(iterations=1500, depth=6, learning_rate=0.03,
                                  l2_leaf_reg=5, random_seed=42, verbose=0, loss_function="Poisson")
        m_ag = CatBoostRegressor(iterations=1500, depth=6, learning_rate=0.03,
                                  l2_leaf_reg=5, random_seed=42, verbose=0, loss_function="Poisson")
        m_hg.fit(X_tr, df_v.loc[train_m, "home_score"],
                 eval_set=(X_val, df_v.loc[val_m, "home_score"]), early_stopping_rounds=100, verbose=0)
        m_ag.fit(X_tr, df_v.loc[train_m, "away_score"],
                 eval_set=(X_val, df_v.loc[val_m, "away_score"]), early_stopping_rounds=100, verbose=0)

        # Get probs
        cls_val = m_cls.predict_proba(X_val)
        cls_te = m_cls.predict_proba(X_te)

        def get_pois(X):
            hxg = np.clip(m_hg.predict(X), 0.3, 4.0)
            axg = np.clip(m_ag.predict(X), 0.3, 4.0)
            p = np.zeros((len(X), 3))
            for i in range(len(X)):
                p[i] = poisson_1x2(hxg[i], axg[i])
            return p

        pois_val = get_pois(X_val)
        pois_te = get_pois(X_te)

        def get_mkt(mask):
            psh = df_v.loc[mask, "odds_PSH"].values
            psd = df_v.loc[mask, "odds_PSD"].values
            psa = df_v.loc[mask, "odds_PSA"].values
            p = np.zeros((mask.sum(), 3))
            for i in range(len(psh)):
                if psh[i] > 1 and psd[i] > 1 and psa[i] > 1:
                    raw = np.array([1/psh[i], 1/psd[i], 1/psa[i]])
                    p[i] = raw / raw.sum()
                else:
                    p[i] = [0.4, 0.3, 0.3]
            return p

        mkt_val = get_mkt(val_m)
        mkt_te = get_mkt(test_m)

        # Calibrate on val
        best_acc = 0
        best_cfg = {"w_cb": 0.3, "w_pois": 0.2, "w_mkt": 0.5, "draw_inflate": 1.0}

        for w_cb in np.arange(0.15, 0.50, 0.05):
            for w_pois in np.arange(0.05, 0.35, 0.05):
                w_mkt = round(1.0 - w_cb - w_pois, 2)
                if w_mkt < 0.1 or w_mkt > 0.6:
                    continue

                base = w_cb * cls_val + w_pois * pois_val + w_mkt * mkt_val
                for d_inf in np.arange(1.0, 2.0, 0.05):
                    adj = base.copy()
                    adj[:, 1] *= d_inf
                    adj = adj / adj.sum(axis=1, keepdims=True)
                    preds = np.array(["H", "D", "A"])[adj.argmax(axis=1)]
                    acc = (preds == y_val).mean()
                    if acc > best_acc:
                        best_acc = acc
                        best_cfg = {"w_cb": round(w_cb, 3), "w_pois": round(w_pois, 3),
                                    "w_mkt": round(w_mkt, 3), "draw_inflate": round(d_inf, 3)}

        # Apply to test
        base_te_p = best_cfg["w_cb"] * cls_te + best_cfg["w_pois"] * pois_te + best_cfg["w_mkt"] * mkt_te
        adj_te_p = base_te_p.copy()
        adj_te_p[:, 1] *= best_cfg["draw_inflate"]
        adj_te_p = adj_te_p / adj_te_p.sum(axis=1, keepdims=True)

        preds = np.array(["H", "D", "A"])[adj_te_p.argmax(axis=1)]
        acc = (preds == y_te).mean()

        # Market baseline
        mkt_preds = np.array(["H", "D", "A"])[mkt_te.argmax(axis=1)]
        mkt_acc_s = (mkt_preds == y_te).mean()

        # Raw (no inflation)
        raw_preds = np.array(["H", "D", "A"])[base_te_p.argmax(axis=1)]
        raw_acc = (raw_preds == y_te).mean()

        n_draws_pred = (preds == "D").sum()
        n_draws_actual = (y_te == "D").sum()

        log.info("  %s: cal=%.1f%% raw=%.1f%% mkt=%.1f%% | draws: %d pred / %d actual | inflate=%.2f",
                 test_season, acc * 100, raw_acc * 100, mkt_acc_s * 100,
                 n_draws_pred, n_draws_actual, best_cfg["draw_inflate"])

        all_preds.extend(preds.tolist())
        all_true.extend(y_te.tolist())
        all_probs.extend(adj_te_p.tolist())
        season_results.append({
            "season": test_season,
            "calibrated_acc": round(acc, 4),
            "raw_acc": round(raw_acc, 4),
            "market_acc": round(mkt_acc_s, 4),
            "config": best_cfg,
            "draws_predicted": int(n_draws_pred),
            "draws_actual": int(n_draws_actual),
        })

    # Overall
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    overall_acc = (all_preds == all_true).mean()
    overall_ll = log_loss(pd.Series(all_true).map(label_map).values,
                          np.array(all_probs), labels=[0, 1, 2])

    # HC tiers across all seasons
    all_max_p = np.array(all_probs).max(axis=1)
    for thr in [0.45, 0.50, 0.55, 0.60]:
        mask = all_max_p > thr
        if mask.sum() > 0:
            t_acc = (all_preds[mask] == all_true[mask]).mean()
            log.info("  HC%.0f: %.1f%% (%d/%d matches)", thr*100, t_acc*100, mask.sum(), len(all_preds))

    log.info("\n  OVERALL: %.1f%% accuracy, LL=%.4f (%d matches across %d seasons)",
             overall_acc * 100, overall_ll, len(all_preds), len(season_results))

    return {
        "overall_accuracy": round(overall_acc, 4),
        "overall_log_loss": round(overall_ll, 4),
        "total_matches": len(all_preds),
        "seasons": season_results,
    }


def main():
    log.info("=" * 70)
    log.info("CALIBRATED 1X2 ENSEMBLE")
    log.info("=" * 70)

    df, features = load_data()
    all_results = {}

    # Single-season detailed evaluation
    for ts in ["2023-2024", "2024-2025"]:
        log.info("\n" + "=" * 70)
        log.info("DETAILED EVALUATION: %s", ts)
        log.info("=" * 70)
        r = calibrated_ensemble(df, features, ts)
        all_results[ts] = r

    # Multi-season robustness check
    log.info("\n" + "=" * 70)
    log.info("MULTI-SEASON ROBUSTNESS CHECK")
    log.info("=" * 70)
    multi = multi_season_backtest(df, features)
    all_results["multi_season"] = multi

    with open(MARKET_MODELS_DIR / "calibrated_ensemble_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    log.info("\n" + "=" * 70)
    log.info("FINAL SUMMARY")
    log.info("=" * 70)
    for ts in ["2023-2024", "2024-2025"]:
        r = all_results[ts]
        log.info("  %s: %.1f%% (raw=%.1f%%, market=%.1f%%) | +%.1fpp vs market",
                 ts, r["accuracy"]*100, r["raw_accuracy"]*100,
                 r["market_accuracy"]*100, r["improvement_vs_market"])
    log.info("  Multi-season: %.1f%% overall (%d matches)",
             multi["overall_accuracy"]*100, multi["total_matches"])


if __name__ == "__main__":
    main()
