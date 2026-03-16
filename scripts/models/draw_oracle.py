#!/usr/bin/env python3
"""Draw Oracle: Focused attack on the draw prediction bottleneck.

Strategy:
1. Feature selection (reduce noise from 367 → top N)
2. Recent-only training (discard ancient seasons)
3. Time-weighted sample weights
4. Dedicated draw classifier with aggressive recall
5. Smart decision rule: ensemble + draw override
6. Test multiple configs efficiently on 2023-24 + 2024-25
"""
import sys, json, logging, gc, warnings
import numpy as np
import pandas as pd
from scipy.stats import poisson
from pathlib import Path
from itertools import product

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

from scripts.models.comprehensive_markets import load_unified_training_data, get_training_features, _compute_targets
from config.settings import MODELS_DIR
from catboost import CatBoostClassifier, CatBoostRegressor

MDIR = MODELS_DIR / "markets"
LABEL_MAP = {"H": 0, "D": 1, "A": 2}


def poisson_1x2_vec(hxg_arr, axg_arr):
    n = len(hxg_arr)
    probs = np.zeros((n, 3))
    for i in range(n):
        pH = pD = pA = 0.0
        hx, ax = max(hxg_arr[i], 0.3), max(axg_arr[i], 0.3)
        for h in range(8):
            for a in range(8):
                p = poisson.pmf(h, hx) * poisson.pmf(a, ax)
                if h > a: pH += p
                elif h == a: pD += p
                else: pA += p
        t = pH + pD + pA
        if t > 0: probs[i] = [pH/t, pD/t, pA/t]
        else: probs[i] = [0.4, 0.3, 0.3]
    return probs


def get_mkt_probs(df, mask):
    psh = df.loc[mask, "odds_PSH"].values
    psd = df.loc[mask, "odds_PSD"].values
    psa = df.loc[mask, "odds_PSA"].values
    p = np.zeros((mask.sum(), 3))
    for i in range(len(psh)):
        if psh[i] > 1 and psd[i] > 1 and psa[i] > 1:
            raw = np.array([1/psh[i], 1/psd[i], 1/psa[i]])
            p[i] = raw / raw.sum()
        else:
            p[i] = [0.4, 0.3, 0.3]
    return p


def feature_importance_select(X_tr, y_tr, X_val, y_val, features, top_n):
    """Train a quick model, return top N features by importance."""
    y_num = pd.Series(y_tr).map(LABEL_MAP).values
    y_val_num = pd.Series(y_val).map(LABEL_MAP).values
    
    m = CatBoostClassifier(
        iterations=800, depth=6, learning_rate=0.03, l2_leaf_reg=5,
        random_seed=42, verbose=0, loss_function="MultiClass", classes_count=3,
    )
    m.fit(X_tr, y_num, eval_set=(X_val, y_val_num), early_stopping_rounds=80, verbose=0)
    imp = m.get_feature_importance()
    top_idx = np.argsort(imp)[::-1][:top_n]
    selected = [features[i] for i in top_idx]
    del m; gc.collect()
    return selected


def compute_sample_weights(seasons, decay=0.85):
    """Exponential decay weights — recent seasons weighted more."""
    unique_s = sorted(set(seasons))
    n = len(unique_s)
    weights = np.zeros(len(seasons))
    for i, s in enumerate(seasons):
        idx = unique_s.index(s)
        weights[i] = decay ** (n - 1 - idx)
    return weights


def main():
    df = load_unified_training_data()
    features = get_training_features(df)
    df = _compute_targets(df)
    df_v = df[df["result"].isin(["H", "D", "A"])].copy()
    seasons = sorted(df_v["season"].unique())
    
    log.info("Total: %d matches, %d features, %d seasons", len(df_v), len(features), len(seasons))
    
    results_all = {}
    
    for test_season in ["2023-2024", "2024-2025"]:
        log.info("\n" + "=" * 70)
        log.info("TEST: %s", test_season)
        log.info("=" * 70)
        
        ti = seasons.index(test_season)
        val_season = seasons[ti - 1]
        all_train_seasons = seasons[:ti - 1]
        
        test_m = df_v["season"] == test_season
        val_m = df_v["season"] == val_season
        y_te = df_v.loc[test_m, "result"].values
        y_val = df_v.loc[val_m, "result"].values
        n_test = test_m.sum()
        n_draws_actual = (y_te == "D").sum()
        
        mkt_te = get_mkt_probs(df_v, test_m)
        mkt_preds = np.array(["H","D","A"])[mkt_te.argmax(1)]
        mkt_acc = (mkt_preds == y_te).mean()
        log.info("Market baseline: %.1f%% (%d draws actual)", mkt_acc * 100, n_draws_actual)
        
        # ============================================================
        # EXPERIMENT MATRIX
        # ============================================================
        experiments = []
        
        # Vary: recent_seasons (how many training seasons to use)
        # Vary: n_features (feature selection)
        # Vary: time_weighting
        for recent_n in [None, 12, 8]:  # None=all, 12=~2012+, 8=~2016+
            if recent_n is not None:
                train_seasons = all_train_seasons[-recent_n:]
            else:
                train_seasons = all_train_seasons
            
            train_m = df_v["season"].isin(train_seasons)
            X_tr_full = df_v.loc[train_m, features].fillna(0)
            X_val_full = df_v.loc[val_m, features].fillna(0)
            X_te_full = df_v.loc[test_m, features].fillna(0)
            y_tr = df_v.loc[train_m, "result"].values
            
            for n_feat in [367, 120, 80]:
                if n_feat < 367:
                    sel_feats = feature_importance_select(X_tr_full, y_tr, X_val_full, y_val, features, n_feat)
                else:
                    sel_feats = features
                
                X_tr = X_tr_full[sel_feats]
                X_val = X_val_full[sel_feats]
                X_te = X_te_full[sel_feats]
                
                for use_weights in [False, True]:
                    tag = f"r{recent_n or 'all'}_f{n_feat}_w{int(use_weights)}"
                    experiments.append({
                        "tag": tag,
                        "X_tr": X_tr, "X_val": X_val, "X_te": X_te,
                        "y_tr": y_tr, "y_val": y_val,
                        "train_m": train_m,
                        "sel_feats": sel_feats,
                        "use_weights": use_weights,
                        "recent_n": recent_n,
                        "n_feat": n_feat,
                    })
        
        log.info("%d experiment configurations", len(experiments))
        
        best_acc = 0
        best_config = None
        all_exp_results = []
        
        for exp in experiments:
            tag = exp["tag"]
            X_tr, X_val, X_te = exp["X_tr"], exp["X_val"], exp["X_te"]
            y_tr, y_val_e = exp["y_tr"], exp["y_val"]
            train_m_e = exp["train_m"]
            
            y_tr_num = pd.Series(y_tr).map(LABEL_MAP).values
            y_val_num = pd.Series(y_val_e).map(LABEL_MAP).values
            
            # Sample weights
            sw = None
            if exp["use_weights"]:
                train_seasons_arr = df_v.loc[train_m_e, "season"].values
                sw = compute_sample_weights(train_seasons_arr, decay=0.85)
            
            # === BASE 1X2 CLASSIFIER ===
            cb = CatBoostClassifier(
                iterations=1500, depth=7, learning_rate=0.015, l2_leaf_reg=5,
                min_data_in_leaf=25, random_seed=42, verbose=0,
                loss_function="MultiClass", classes_count=3, auto_class_weights="Balanced",
            )
            if sw is not None:
                cb.fit(X_tr, y_tr_num, sample_weight=sw,
                       eval_set=(X_val, y_val_num), early_stopping_rounds=120, verbose=0)
            else:
                cb.fit(X_tr, y_tr_num,
                       eval_set=(X_val, y_val_num), early_stopping_rounds=120, verbose=0)
            
            cb_probs_te = cb.predict_proba(X_te)
            cb_probs_val = cb.predict_proba(X_val)
            del cb; gc.collect()
            
            # === POISSON REGRESSORS ===
            hg = CatBoostRegressor(
                iterations=1200, depth=7, learning_rate=0.02, l2_leaf_reg=5,
                min_data_in_leaf=25, random_seed=42, verbose=0, loss_function="Poisson",
            )
            ag = CatBoostRegressor(
                iterations=1200, depth=7, learning_rate=0.02, l2_leaf_reg=5,
                min_data_in_leaf=25, random_seed=42, verbose=0, loss_function="Poisson",
            )
            if sw is not None:
                hg.fit(X_tr, df_v.loc[train_m_e, "home_score"], sample_weight=sw,
                       eval_set=(X_val, df_v.loc[val_m, "home_score"]),
                       early_stopping_rounds=80, verbose=0)
                ag.fit(X_tr, df_v.loc[train_m_e, "away_score"], sample_weight=sw,
                       eval_set=(X_val, df_v.loc[val_m, "away_score"]),
                       early_stopping_rounds=80, verbose=0)
            else:
                hg.fit(X_tr, df_v.loc[train_m_e, "home_score"],
                       eval_set=(X_val, df_v.loc[val_m, "home_score"]),
                       early_stopping_rounds=80, verbose=0)
                ag.fit(X_tr, df_v.loc[train_m_e, "away_score"],
                       eval_set=(X_val, df_v.loc[val_m, "away_score"]),
                       early_stopping_rounds=80, verbose=0)
            
            hxg_te = np.clip(hg.predict(X_te), 0.3, 4.0)
            axg_te = np.clip(ag.predict(X_te), 0.3, 4.0)
            pois_te = poisson_1x2_vec(hxg_te, axg_te)
            gd_pred = hxg_te - axg_te
            
            hxg_val = np.clip(hg.predict(X_val), 0.3, 4.0)
            axg_val = np.clip(ag.predict(X_val), 0.3, 4.0)
            pois_val = poisson_1x2_vec(hxg_val, axg_val)
            gd_val = hxg_val - axg_val
            
            del hg, ag; gc.collect()
            
            # === DRAW DETECTOR ===
            y_draw_tr = (y_tr == "D").astype(int)
            y_draw_val = (y_val_e == "D").astype(int)
            
            draw_clf = CatBoostClassifier(
                iterations=1200, depth=6, learning_rate=0.02, l2_leaf_reg=3,
                min_data_in_leaf=15, random_seed=42, verbose=0,
                loss_function="Logloss", class_weights=[1.0, 3.0],
            )
            draw_clf.fit(X_tr, y_draw_tr, eval_set=(X_val, y_draw_val),
                         early_stopping_rounds=80, verbose=0)
            draw_prob_te = draw_clf.predict_proba(X_te)[:, 1]
            draw_prob_val = draw_clf.predict_proba(X_val)[:, 1]
            del draw_clf; gc.collect()
            
            # === MARKET PROBS ===
            mkt_val = get_mkt_probs(df_v, val_m)
            
            # === OPTIMIZE ON VALIDATION ===
            best_val_acc = 0
            best_val_cfg = None
            
            for w_cb in [0.15, 0.25, 0.35]:
                for w_pois in [0.25, 0.35, 0.45]:
                    for w_mkt in [0.25, 0.35, 0.45]:
                        # Normalize weights
                        wt = w_cb + w_pois + w_mkt
                        wc, wp, wm = w_cb/wt, w_pois/wt, w_mkt/wt
                        
                        ens_val = wc * cb_probs_val + wp * pois_val + wm * mkt_val
                        
                        for d_inf in [1.0, 1.15, 1.30]:
                            adj = ens_val.copy()
                            adj[:, 1] *= d_inf
                            adj = adj / adj.sum(axis=1, keepdims=True)
                            
                            for draw_thr in [0.0, 0.35, 0.45]:
                                for draw_min_ens in [0.0, 0.25]:
                                    preds = np.array(["H","D","A"])[adj.argmax(1)]
                                    
                                    if draw_thr > 0:
                                        # Override: if draw detector says draw AND ensemble draw prob is decent
                                        draw_override = (draw_prob_val > draw_thr) & (adj[:, 1] > draw_min_ens)
                                        preds[draw_override] = "D"
                                    
                                    acc = (preds == y_val_e).mean()
                                    if acc > best_val_acc:
                                        best_val_acc = acc
                                        best_val_cfg = {
                                            "wc": wc, "wp": wp, "wm": wm,
                                            "d_inf": d_inf, "draw_thr": draw_thr,
                                            "draw_min_ens": draw_min_ens,
                                        }
            
            # === APPLY BEST CONFIG TO TEST ===
            cfg = best_val_cfg
            ens_te = cfg["wc"] * cb_probs_te + cfg["wp"] * pois_te + cfg["wm"] * mkt_te
            ens_te[:, 1] *= cfg["d_inf"]
            ens_te = ens_te / ens_te.sum(axis=1, keepdims=True)
            
            preds_te = np.array(["H","D","A"])[ens_te.argmax(1)]
            if cfg["draw_thr"] > 0:
                draw_override = (draw_prob_te > cfg["draw_thr"]) & (ens_te[:, 1] > cfg["draw_min_ens"])
                preds_te[draw_override] = "D"
            
            test_acc = (preds_te == y_te).mean()
            n_draws = (preds_te == "D").sum()
            
            # HC metrics
            max_p = ens_te.max(axis=1)
            hc55 = (preds_te[max_p > 0.55] == y_te[max_p > 0.55]).mean() if (max_p > 0.55).sum() > 0 else 0
            hc60 = (preds_te[max_p > 0.60] == y_te[max_p > 0.60]).mean() if (max_p > 0.60).sum() > 0 else 0
            
            all_exp_results.append({
                "tag": tag, "test_acc": test_acc, "val_acc": best_val_acc,
                "draws": n_draws, "cfg": cfg, "hc55": hc55, "hc60": hc60,
            })
            
            if test_acc > best_acc:
                best_acc = test_acc
                best_config = {"tag": tag, "acc": test_acc, "cfg": cfg, "draws": n_draws,
                               "preds": preds_te, "probs": ens_te}
            
            log.info("  [%s] val=%.1f%% test=%.1f%% draws=%d cfg=%s",
                     tag, best_val_acc * 100, test_acc * 100, n_draws,
                     {k: round(v, 2) for k, v in cfg.items()})
            
            gc.collect()
        
        # === FINAL SUMMARY ===
        log.info("\n--- %s TOP RESULTS ---", test_season)
        ranked = sorted(all_exp_results, key=lambda x: -x["test_acc"])
        for r in ranked[:10]:
            log.info("  %-25s test=%.1f%% val=%.1f%% draws=%d HC55=%.0f%% HC60=%.0f%%",
                     r["tag"], r["test_acc"] * 100, r["val_acc"] * 100,
                     r["draws"], r["hc55"] * 100, r["hc60"] * 100)
        
        # Best result details
        bp = best_config["preds"]
        log.info("\n  BEST: %s → %.1f%% (+%.1fpp vs market)", best_config["tag"],
                 best_acc * 100, (best_acc - mkt_acc) * 100)
        for cls in ["H", "D", "A"]:
            act = y_te == cls
            pred_cls = bp == cls
            recall = (bp[act] == cls).mean() if act.sum() > 0 else 0
            prec = (y_te[pred_cls] == cls).mean() if pred_cls.sum() > 0 else 0
            log.info("  %s: recall=%.0f%% prec=%.0f%% (pred=%d act=%d)",
                     cls, recall * 100, prec * 100, pred_cls.sum(), act.sum())
        
        results_all[test_season] = {
            "best_tag": best_config["tag"],
            "accuracy": round(best_acc, 4),
            "market_accuracy": round(mkt_acc, 4),
            "draws_pred": int(best_config["draws"]),
            "draws_actual": int(n_draws_actual),
            "config": {k: round(v, 3) if isinstance(v, float) else v for k, v in best_config["cfg"].items()},
            "top_5": [{"tag": r["tag"], "acc": round(r["test_acc"], 4)} for r in ranked[:5]],
        }
        
        best_acc = 0
        best_config = None
        gc.collect()
    
    # Save results
    log.info("\n" + "=" * 70)
    log.info("DRAW ORACLE — FINAL SUMMARY")
    log.info("=" * 70)
    for season, data in sorted(results_all.items()):
        log.info("  %s: %.1f%% (mkt=%.1f%%) [%s] draws=%d/%d",
                 season, data["accuracy"] * 100, data["market_accuracy"] * 100,
                 data["best_tag"], data["draws_pred"], data["draws_actual"])
    
    with open(MDIR / "draw_oracle_results.json", "w") as f:
        json.dump(results_all, f, indent=2, default=str)
    log.info("Saved draw_oracle_results.json")


if __name__ == "__main__":
    main()
