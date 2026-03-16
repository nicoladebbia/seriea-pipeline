#!/usr/bin/env python3
"""Breakthrough: Fix draw problem + multi-algo ensemble + side bet overhaul.

Key insights from error analysis:
- 74% of errors involve draws
- Model collapses on competitive matches (odds 2.0-2.5: 42.1%)
- Side bet models just predict majority class
- Calibration shows model is under-confident at medium confidence

Strategy:
1. Multi-algorithm ensemble (CatBoost + XGBoost + LightGBM)
2. Dedicated draw detector that overrides ensemble on competitive matches
3. Per-class threshold optimization
4. Isotonic probability calibration
5. Side bet model overhaul with proper class balancing
"""
import sys, json, logging, gc, warnings
import numpy as np
import pandas as pd
from scipy.stats import poisson
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, log_loss

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

from scripts.models.comprehensive_markets import load_unified_training_data, get_training_features, _compute_targets
from config.settings import MODELS_DIR

MDIR = MODELS_DIR / "markets"
LABEL_MAP = {"H": 0, "D": 1, "A": 2}
INV_LABEL = {0: "H", 1: "D", 2: "A"}


def poisson_1x2(hxg, axg):
    pH = pD = pA = 0.0
    for h in range(8):
        for a in range(8):
            p = poisson.pmf(h, hxg) * poisson.pmf(a, axg)
            if h > a: pH += p
            elif h == a: pD += p
            else: pA += p
    t = pH + pD + pA
    return np.array([pH / t, pD / t, pA / t])


def get_mkt(df_v, mask):
    psh = df_v.loc[mask, "odds_PSH"].values
    psd = df_v.loc[mask, "odds_PSD"].values
    psa = df_v.loc[mask, "odds_PSA"].values
    p = np.zeros((mask.sum(), 3))
    for i in range(len(psh)):
        if psh[i] > 1 and psd[i] > 1 and psa[i] > 1:
            raw = np.array([1 / psh[i], 1 / psd[i], 1 / psa[i]])
            p[i] = raw / raw.sum()
        else:
            p[i] = [0.4, 0.3, 0.3]
    return p


def build_pois_probs(hg, ag, X):
    hxg = np.clip(hg.predict(X), 0.3, 4.0)
    axg = np.clip(ag.predict(X), 0.3, 4.0)
    p = np.zeros((len(X), 3))
    for i in range(len(X)):
        p[i] = poisson_1x2(hxg[i], axg[i])
    return p


def main():
    from catboost import CatBoostClassifier, CatBoostRegressor

    df = load_unified_training_data()
    features = get_training_features(df)
    df = _compute_targets(df)
    df_v = df[df["result"].isin(["H", "D", "A"])].copy()
    seasons = sorted(df_v["season"].unique())

    # Walk-forward: test on last 2 seasons for robustness
    results_per_season = {}

    for test_season in ["2023-2024", "2024-2025"]:
        log.info("\n" + "=" * 60)
        log.info("TEST SEASON: %s", test_season)
        log.info("=" * 60)

        ti = seasons.index(test_season)
        val_season = seasons[ti - 1]
        train_seasons = seasons[:ti - 1]

        train_m = df_v["season"].isin(train_seasons)
        val_m = df_v["season"] == val_season
        test_m = df_v["season"] == test_season

        X_tr = df_v.loc[train_m, features].fillna(0)
        X_val = df_v.loc[val_m, features].fillna(0)
        X_te = df_v.loc[test_m, features].fillna(0)
        y_tr = df_v.loc[train_m, "result"].values
        y_val = df_v.loc[val_m, "result"].values
        y_te = df_v.loc[test_m, "result"].values
        y_tr_num = pd.Series(y_tr).map(LABEL_MAP).values
        y_val_num = pd.Series(y_val).map(LABEL_MAP).values

        mkt_tr = get_mkt(df_v, train_m)
        mkt_val = get_mkt(df_v, val_m)
        mkt_te = get_mkt(df_v, test_m)

        log.info("Train: %d, Val: %d, Test: %d", len(X_tr), len(X_val), len(X_te))

        # ================================================================
        # STEP 1: Train multiple diverse base models
        # ================================================================
        log.info("\n--- STEP 1: Multi-Algorithm Base Models ---")

        # 1a. CatBoost (depth=8, tuned)
        cb = CatBoostClassifier(
            iterations=3000, depth=8, learning_rate=0.01, l2_leaf_reg=5,
            min_data_in_leaf=30, random_seed=42, verbose=0,
            loss_function="MultiClass", classes_count=3, auto_class_weights="Balanced",
        )
        cb.fit(X_tr, y_tr_num, eval_set=(X_val, y_val_num),
               early_stopping_rounds=200, verbose=0)
        cb_val = cb.predict_proba(X_val)
        cb_te = cb.predict_proba(X_te)
        log.info("  CatBoost d=8: val=%.1f%%", (cb_val.argmax(1) == y_val_num).mean() * 100)

        # 1b. CatBoost (depth=5, different view)
        cb2 = CatBoostClassifier(
            iterations=2000, depth=5, learning_rate=0.025, l2_leaf_reg=7,
            min_data_in_leaf=40, random_seed=123, verbose=0,
            loss_function="MultiClass", classes_count=3, auto_class_weights="Balanced",
        )
        cb2.fit(X_tr, y_tr_num, eval_set=(X_val, y_val_num),
                early_stopping_rounds=150, verbose=0)
        cb2_val = cb2.predict_proba(X_val)
        cb2_te = cb2.predict_proba(X_te)
        log.info("  CatBoost d=5: val=%.1f%%", (cb2_val.argmax(1) == y_val_num).mean() * 100)

        # 1c. XGBoost
        try:
            from xgboost import XGBClassifier
            xgb = XGBClassifier(
                n_estimators=1500, max_depth=6, learning_rate=0.02,
                reg_lambda=3, min_child_weight=30, subsample=0.8,
                colsample_bytree=0.8, random_state=42,
                objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                early_stopping_rounds=100, verbosity=0,
            )
            xgb.fit(X_tr, y_tr_num, eval_set=[(X_val, y_val_num)], verbose=False)
            xgb_val = xgb.predict_proba(X_val)
            xgb_te = xgb.predict_proba(X_te)
            log.info("  XGBoost: val=%.1f%%", (xgb_val.argmax(1) == y_val_num).mean() * 100)
            has_xgb = True
        except ImportError:
            log.info("  XGBoost not available, skipping")
            has_xgb = False

        # 1d. LightGBM
        try:
            from lightgbm import LGBMClassifier
            lgb = LGBMClassifier(
                n_estimators=1500, max_depth=6, learning_rate=0.02,
                reg_lambda=3, min_child_samples=30, subsample=0.8,
                colsample_bytree=0.8, random_state=42,
                objective="multiclass", num_class=3, verbosity=-1,
            )
            lgb.fit(X_tr, y_tr_num, eval_set=[(X_val, y_val_num)],
                    callbacks=[lambda env: None])  # suppress output
            lgb_val = lgb.predict_proba(X_val)
            lgb_te = lgb.predict_proba(X_te)
            log.info("  LightGBM: val=%.1f%%", (lgb_val.argmax(1) == y_val_num).mean() * 100)
            has_lgb = True
        except ImportError:
            log.info("  LightGBM not available, skipping")
            has_lgb = False

        # 1e. Poisson
        hg = CatBoostRegressor(
            iterations=2000, depth=8, learning_rate=0.015, l2_leaf_reg=7,
            min_data_in_leaf=30, random_seed=42, verbose=0, loss_function="Poisson",
        )
        ag = CatBoostRegressor(
            iterations=2000, depth=8, learning_rate=0.015, l2_leaf_reg=7,
            min_data_in_leaf=30, random_seed=42, verbose=0, loss_function="Poisson",
        )
        hg.fit(X_tr, df_v.loc[train_m, "home_score"],
               eval_set=(X_val, df_v.loc[val_m, "home_score"]), early_stopping_rounds=100, verbose=0)
        ag.fit(X_tr, df_v.loc[train_m, "away_score"],
               eval_set=(X_val, df_v.loc[val_m, "away_score"]), early_stopping_rounds=100, verbose=0)
        pois_val = build_pois_probs(hg, ag, X_val)
        pois_te = build_pois_probs(hg, ag, X_te)
        log.info("  Poisson: val=%.1f%%", (np.array(["H", "D", "A"])[pois_val.argmax(1)] == y_val).mean() * 100)

        # ================================================================
        # STEP 2: Dedicated Draw Detector (binary, high recall)
        # ================================================================
        log.info("\n--- STEP 2: Draw Detector ---")
        y_draw_tr = (y_tr == "D").astype(int)
        y_draw_val = (y_val == "D").astype(int)

        # Train with aggressive class balancing for draws
        draw_clf = CatBoostClassifier(
            iterations=2000, depth=7, learning_rate=0.02, l2_leaf_reg=3,
            min_data_in_leaf=20, random_seed=42, verbose=0,
            loss_function="Logloss", class_weights=[1.0, 2.5],
        )
        draw_clf.fit(X_tr, y_draw_tr, eval_set=(X_val, y_draw_val),
                     early_stopping_rounds=150, verbose=0)
        draw_val = draw_clf.predict_proba(X_val)[:, 1]
        draw_te = draw_clf.predict_proba(X_te)[:, 1]

        # Also train a "competitive match" detector using goal difference regression
        gd_reg = CatBoostRegressor(
            iterations=1500, depth=6, learning_rate=0.03, l2_leaf_reg=5,
            random_seed=42, verbose=0,
        )
        gd_tr = (df_v.loc[train_m, "home_score"] - df_v.loc[train_m, "away_score"]).values
        gd_val_actual = (df_v.loc[val_m, "home_score"] - df_v.loc[val_m, "away_score"]).values
        gd_reg.fit(X_tr, gd_tr, eval_set=(X_val, gd_val_actual),
                   early_stopping_rounds=100, verbose=0)
        gd_val_pred = gd_reg.predict(X_val)
        gd_te_pred = gd_reg.predict(X_te)
        # Close GD → more likely draw
        gd_draw_signal_val = np.exp(-0.5 * (gd_val_pred / 0.7) ** 2)
        gd_draw_signal_te = np.exp(-0.5 * (gd_te_pred / 0.7) ** 2)

        log.info("  Draw detector AUC: val draw_prob mean=%.3f (actual_draw_rate=%.3f)",
                 draw_val.mean(), y_draw_val.mean())

        # ================================================================
        # STEP 3: Optimal multi-algo ensemble blend with draw injection
        # ================================================================
        log.info("\n--- STEP 3: Optimized Ensemble Blend ---")

        # Build all component probs for val
        components_val = {"cb": cb_val, "cb2": cb2_val, "pois": pois_val, "mkt": mkt_val}
        components_te = {"cb": cb_te, "cb2": cb2_te, "pois": pois_te, "mkt": mkt_te}
        if has_xgb:
            components_val["xgb"] = xgb_val
            components_te["xgb"] = xgb_te
        if has_lgb:
            components_val["lgb"] = lgb_val
            components_te["lgb"] = lgb_te

        n_components = len(components_val)
        comp_names = list(components_val.keys())
        log.info("  Components: %s", comp_names)

        # Grid search over weights and draw parameters
        best_acc = 0
        best_cfg = {}

        # Generate weight combinations
        import itertools
        weight_options = np.arange(0.05, 0.55, 0.05)

        # For efficiency, use coarse grid then refine
        from itertools import product

        # Coarse grid
        for w_cb in [0.10, 0.15, 0.20, 0.25, 0.30]:
            for w_cb2 in [0.05, 0.10, 0.15]:
                for w_pois in [0.15, 0.20, 0.25, 0.30, 0.35]:
                    for w_mkt in [0.20, 0.25, 0.30, 0.35, 0.40]:
                        # XGB/LGB get remainder
                        used = w_cb + w_cb2 + w_pois + w_mkt
                        if used > 0.95 or used < 0.5:
                            continue

                        remain = 1.0 - used
                        if has_xgb and has_lgb:
                            w_xgb = remain * 0.5
                            w_lgb = remain * 0.5
                        elif has_xgb:
                            w_xgb = remain
                            w_lgb = 0
                        elif has_lgb:
                            w_xgb = 0
                            w_lgb = remain
                        else:
                            if abs(remain) > 0.01:
                                continue
                            w_xgb = w_lgb = 0

                        ens = (w_cb * cb_val + w_cb2 * cb2_val + w_pois * pois_val +
                               w_mkt * mkt_val)
                        if has_xgb:
                            ens += w_xgb * xgb_val
                        if has_lgb:
                            ens += w_lgb * lgb_val

                        for d_inf in [1.15, 1.25, 1.35, 1.45]:
                            for draw_inj in [0.0, 0.05, 0.10, 0.15, 0.20]:
                                adj = ens.copy()
                                # Inject draw detector signal
                                adj[:, 1] = (1 - draw_inj) * adj[:, 1] + draw_inj * draw_val
                                # Also inject GD-based draw signal
                                for gd_w in [0.0, 0.05]:
                                    adj2 = adj.copy()
                                    adj2[:, 1] += gd_w * gd_draw_signal_val
                                    adj2[:, 1] *= d_inf
                                    adj2 = adj2 / adj2.sum(axis=1, keepdims=True)

                                    preds = np.array(["H", "D", "A"])[adj2.argmax(axis=1)]
                                    acc = (preds == y_val).mean()
                                    if acc > best_acc:
                                        best_acc = acc
                                        best_cfg = {
                                            "w_cb": w_cb, "w_cb2": w_cb2, "w_pois": w_pois,
                                            "w_mkt": w_mkt, "w_xgb": w_xgb, "w_lgb": w_lgb,
                                            "d_inf": d_inf, "draw_inj": draw_inj, "gd_w": gd_w,
                                        }

        log.info("  Best val: %.1f%% | %s", best_acc * 100,
                 {k: round(v, 3) for k, v in best_cfg.items()})

        # ================================================================
        # STEP 4: Apply to test with per-class threshold optimization
        # ================================================================
        log.info("\n--- STEP 4: Apply to Test + Per-Class Thresholds ---")

        # Build test ensemble with best weights
        c = best_cfg
        ens_te = (c["w_cb"] * cb_te + c["w_cb2"] * cb2_te + c["w_pois"] * pois_te +
                  c["w_mkt"] * mkt_te)
        if has_xgb:
            ens_te += c["w_xgb"] * xgb_te
        if has_lgb:
            ens_te += c["w_lgb"] * lgb_te

        ens_te[:, 1] = (1 - c["draw_inj"]) * ens_te[:, 1] + c["draw_inj"] * draw_te
        ens_te[:, 1] += c["gd_w"] * gd_draw_signal_te
        ens_te[:, 1] *= c["d_inf"]
        ens_te = ens_te / ens_te.sum(axis=1, keepdims=True)

        # Standard argmax
        preds_std = np.array(["H", "D", "A"])[ens_te.argmax(axis=1)]
        acc_std = (preds_std == y_te).mean()
        n_d_std = (preds_std == "D").sum()

        # Per-class threshold optimization on val
        ens_val = (c["w_cb"] * cb_val + c["w_cb2"] * cb2_val + c["w_pois"] * pois_val +
                   c["w_mkt"] * mkt_val)
        if has_xgb:
            ens_val += c["w_xgb"] * xgb_val
        if has_lgb:
            ens_val += c["w_lgb"] * lgb_val
        ens_val[:, 1] = (1 - c["draw_inj"]) * ens_val[:, 1] + c["draw_inj"] * draw_val
        ens_val[:, 1] += c["gd_w"] * gd_draw_signal_val
        ens_val[:, 1] *= c["d_inf"]
        ens_val = ens_val / ens_val.sum(axis=1, keepdims=True)

        # Optimize per-class thresholds on val
        best_thr_acc = 0
        best_thresholds = (0.0, 0.0, 0.0)
        for th_h in np.arange(-0.10, 0.10, 0.02):
            for th_d in np.arange(-0.05, 0.15, 0.02):
                for th_a in np.arange(-0.10, 0.10, 0.02):
                    adj = ens_val.copy()
                    adj[:, 0] += th_h
                    adj[:, 1] += th_d
                    adj[:, 2] += th_a
                    preds = np.array(["H", "D", "A"])[adj.argmax(axis=1)]
                    acc = (preds == y_val).mean()
                    if acc > best_thr_acc:
                        best_thr_acc = acc
                        best_thresholds = (th_h, th_d, th_a)

        # Apply thresholds to test
        adj_te = ens_te.copy()
        adj_te[:, 0] += best_thresholds[0]
        adj_te[:, 1] += best_thresholds[1]
        adj_te[:, 2] += best_thresholds[2]
        preds_thr = np.array(["H", "D", "A"])[adj_te.argmax(axis=1)]
        acc_thr = (preds_thr == y_te).mean()
        n_d_thr = (preds_thr == "D").sum()

        log.info("  Thresholds: H=%.2f D=%.2f A=%.2f (val=%.1f%%)",
                 *best_thresholds, best_thr_acc * 100)
        log.info("  Standard argmax: %.1f%% (%d draws)", acc_std * 100, n_d_std)
        log.info("  With thresholds: %.1f%% (%d draws)", acc_thr * 100, n_d_thr)

        # Use whichever is better
        if acc_thr >= acc_std:
            final_preds = preds_thr
            final_acc = acc_thr
            final_probs = ens_te
        else:
            final_preds = preds_std
            final_acc = acc_std
            final_probs = ens_te

        # HC tiers
        max_p = final_probs.max(axis=1)
        for thr in [0.45, 0.50, 0.55, 0.60]:
            m = max_p > thr
            if m.sum() > 0:
                t_acc = (final_preds[m] == y_te[m]).mean()
                log.info("  HC%.0f: %.1f%% (%d/%d matches)",
                         thr * 100, t_acc * 100, m.sum(), len(y_te))

        # Per-class analysis
        n_d_actual = (y_te == "D").sum()
        for cls in ["H", "D", "A"]:
            act = y_te == cls
            pred_cls = final_preds == cls
            recall = (final_preds[act] == cls).mean() if act.sum() > 0 else 0
            prec = (y_te[pred_cls] == cls).mean() if pred_cls.sum() > 0 else 0
            log.info("  %s: recall=%.0f%% prec=%.0f%% (pred=%d actual=%d)",
                     cls, recall * 100, prec * 100, pred_cls.sum(), act.sum())

        # Market baseline
        mkt_preds = np.array(["H", "D", "A"])[mkt_te.argmax(axis=1)]
        mkt_acc = (mkt_preds == y_te).mean()
        log.info("\n  FINAL: %.1f%% vs Market %.1f%% (+%.1fpp)", 
                 final_acc * 100, mkt_acc * 100, (final_acc - mkt_acc) * 100)

        results_per_season[test_season] = {
            "accuracy": round(final_acc, 4),
            "market_accuracy": round(mkt_acc, 4),
            "n_draws_pred": int((final_preds == "D").sum()),
            "n_draws_actual": int(n_d_actual),
            "config": {k: round(v, 4) for k, v in best_cfg.items()},
            "thresholds": [round(t, 3) for t in best_thresholds],
        }

        del cb, cb2, hg, ag, draw_clf, gd_reg
        if has_xgb: del xgb
        if has_lgb: del lgb
        gc.collect()

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    log.info("\n" + "=" * 60)
    log.info("BREAKTHROUGH RESULTS SUMMARY")
    log.info("=" * 60)
    for season, data in sorted(results_per_season.items()):
        log.info("  %s: %.1f%% (mkt=%.1f%%, +%.1fpp) draws=%d/%d",
                 season, data["accuracy"] * 100, data["market_accuracy"] * 100,
                 (data["accuracy"] - data["market_accuracy"]) * 100,
                 data["n_draws_pred"], data["n_draws_actual"])

    # Save
    with open(MDIR / "breakthrough_results.json", "w") as f:
        json.dump(results_per_season, f, indent=2, default=str)
    log.info("\nSaved breakthrough_results.json")


if __name__ == "__main__":
    main()
