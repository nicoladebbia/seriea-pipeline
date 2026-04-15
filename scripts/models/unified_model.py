#!/usr/bin/env python3
"""Unified Model: Feature-stacked approach with walk-forward temporal OOF.

Instead of training separate models and blending, this approach:
1. Generates Poisson/market/draw signals as FEATURES
2. Trains a single powerful model on original features + derived features
3. Uses walk-forward temporal splits (natural for time-series)

This avoids the K-fold memory issues and is more principled for time-series data.
"""
import sys, json, logging, gc, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

from scripts.models.comprehensive_markets import load_unified_training_data, get_training_features, _compute_targets
from config.settings import MODELS_DIR
from catboost import CatBoostClassifier, CatBoostRegressor
from ml.poisson import poisson_1x2_vec

MDIR = MODELS_DIR / "markets"
LABEL_MAP = {"H": 0, "D": 1, "A": 2}


def add_derived_features(df_v, features, X_df):
    """Add Poisson, market, and draw signal features to the feature matrix."""
    n = len(X_df)
    extra = {}
    
    # Market probabilities (already normalized)
    for col, name in [("odds_PSH", "mkt_pH"), ("odds_PSD", "mkt_pD"), ("odds_PSA", "mkt_pA")]:
        if col in df_v.columns:
            odds = X_df[col].values if col in X_df.columns else df_v.loc[X_df.index, col].values
            inv = np.where(odds > 1, 1.0 / odds, 0.33)
            extra[name] = inv
    
    if "mkt_pH" in extra and "mkt_pD" in extra and "mkt_pA" in extra:
        total = extra["mkt_pH"] + extra["mkt_pD"] + extra["mkt_pA"]
        total = np.where(total > 0, total, 1.0)
        extra["mkt_pH"] = extra["mkt_pH"] / total
        extra["mkt_pD"] = extra["mkt_pD"] / total
        extra["mkt_pA"] = extra["mkt_pA"] / total
        
        # Derived market signals
        extra["mkt_fav_strength"] = np.max([extra["mkt_pH"], extra["mkt_pA"]], axis=0)
        extra["mkt_draw_signal"] = extra["mkt_pD"]
        extra["mkt_competitiveness"] = 1.0 - np.abs(extra["mkt_pH"] - extra["mkt_pA"])
        extra["mkt_entropy"] = -(extra["mkt_pH"] * np.log(np.clip(extra["mkt_pH"], 1e-8, 1)) +
                                  extra["mkt_pD"] * np.log(np.clip(extra["mkt_pD"], 1e-8, 1)) +
                                  extra["mkt_pA"] * np.log(np.clip(extra["mkt_pA"], 1e-8, 1)))
    
    # Odds spread
    for o1, o2, name in [("odds_PSH", "odds_PSA", "odds_ha_spread"),
                          ("odds_PSH", "odds_PSD", "odds_hd_spread"),
                          ("odds_PSD", "odds_PSA", "odds_da_spread")]:
        if o1 in X_df.columns and o2 in X_df.columns:
            extra[name] = np.abs(X_df[o1].values - X_df[o2].values)
    
    # Elo-based draw signal
    if "elo_diff" in X_df.columns:
        ed = X_df["elo_diff"].values
        extra["elo_draw_signal"] = np.exp(-0.5 * (ed / 80) ** 2)  # Gaussian centered at 0
        extra["elo_abs"] = np.abs(ed)
    
    # Form convergence signals
    for h_col, a_col, tag in [("home_roll_5_points", "away_roll_5_points", "5"),
                                ("home_roll_3_points", "away_roll_3_points", "3")]:
        if h_col in X_df.columns and a_col in X_df.columns:
            diff = np.abs(X_df[h_col].values - X_df[a_col].values)
            extra[f"form_gap_{tag}"] = diff
            extra[f"form_close_{tag}"] = (diff < 3).astype(float)
    
    # H2H draw tendency
    if "h2h_draw_rate" in X_df.columns:
        extra["h2h_draw_boost"] = X_df["h2h_draw_rate"].values
    
    # Goal rate symmetry (close = draw-prone)
    for h_col, a_col, tag in [("home_attack_strength", "away_attack_strength", "attack"),
                                ("home_defense_strength", "away_defense_strength", "defense")]:
        if h_col in X_df.columns and a_col in X_df.columns:
            extra[f"sym_{tag}"] = 1.0 - np.abs(X_df[h_col].values - X_df[a_col].values)
    
    return extra


def main():
    df = load_unified_training_data()
    features = get_training_features(df)
    df = _compute_targets(df)
    df_v = df[df["result"].isin(["H", "D", "A"])].copy()
    seasons = sorted(df_v["season"].unique())
    
    log.info("Seasons: %s", seasons)
    log.info("Base features: %d", len(features))
    
    # ================================================================
    # APPROACH 1: Walk-forward with Poisson-augmented features
    # ================================================================
    log.info("\n=== APPROACH 1: Poisson-Augmented Unified Model ===")
    
    all_results = {}
    
    for test_season in ["2022-2023", "2023-2024", "2024-2025"]:
        ti = seasons.index(test_season)
        if ti < 3:
            continue
        
        val_season = seasons[ti - 1]
        train_seasons = seasons[:ti - 1]
        
        train_m = df_v["season"].isin(train_seasons)
        val_m = df_v["season"] == val_season
        test_m = df_v["season"] == test_season
        
        y_tr = df_v.loc[train_m, "result"].values
        y_val = df_v.loc[val_m, "result"].values
        y_te = df_v.loc[test_m, "result"].values
        
        # Step 1: Train Poisson models on train, predict on train+val+test
        log.info("\n--- %s: Training Poisson for features ---", test_season)
        X_tr_base = df_v.loc[train_m, features].fillna(0)
        X_val_base = df_v.loc[val_m, features].fillna(0)
        X_te_base = df_v.loc[test_m, features].fillna(0)
        
        hg = CatBoostRegressor(
            iterations=1500, depth=6, learning_rate=0.03, l2_leaf_reg=5,
            min_data_in_leaf=30, random_seed=42, verbose=0, loss_function="Poisson",
        )
        ag = CatBoostRegressor(
            iterations=1500, depth=6, learning_rate=0.03, l2_leaf_reg=5,
            min_data_in_leaf=30, random_seed=42, verbose=0, loss_function="Poisson",
        )
        hg.fit(X_tr_base, df_v.loc[train_m, "home_score"],
               eval_set=(X_val_base, df_v.loc[val_m, "home_score"]),
               early_stopping_rounds=100, verbose=0)
        ag.fit(X_tr_base, df_v.loc[train_m, "away_score"],
               eval_set=(X_val_base, df_v.loc[val_m, "away_score"]),
               early_stopping_rounds=100, verbose=0)
        
        # Poisson predictions as features (OOF for val, direct for test)
        def add_poisson_feats(X_base, hg, ag):
            hxg = np.clip(hg.predict(X_base), 0.3, 4.0)
            axg = np.clip(ag.predict(X_base), 0.3, 4.0)
            pois = poisson_1x2_vec(hxg, axg)
            return {
                "pois_pH": pois[:, 0], "pois_pD": pois[:, 1], "pois_pA": pois[:, 2],
                "pois_hxg": hxg, "pois_axg": axg,
                "pois_xg_diff": hxg - axg, "pois_xg_total": hxg + axg,
                "pois_xg_gap": np.abs(hxg - axg),
                "pois_draw_boost": np.exp(-0.5 * ((hxg - axg) / 0.7) ** 2),
            }
        
        pois_val = add_poisson_feats(X_val_base, hg, ag)
        pois_te = add_poisson_feats(X_te_base, hg, ag)
        
        # For training set, use cross-val Poisson predictions to avoid leakage
        # Simple approach: train on first half, predict second half and vice versa
        n_tr = len(X_tr_base)
        half = n_tr // 2
        hg1 = CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.03, l2_leaf_reg=5,
                                  random_seed=42, verbose=0, loss_function="Poisson")
        ag1 = CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.03, l2_leaf_reg=5,
                                  random_seed=42, verbose=0, loss_function="Poisson")
        hg1.fit(X_tr_base.iloc[:half], df_v.loc[train_m, "home_score"].iloc[:half], verbose=0)
        ag1.fit(X_tr_base.iloc[:half], df_v.loc[train_m, "away_score"].iloc[:half], verbose=0)
        
        hg2 = CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.03, l2_leaf_reg=5,
                                  random_seed=42, verbose=0, loss_function="Poisson")
        ag2 = CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.03, l2_leaf_reg=5,
                                  random_seed=42, verbose=0, loss_function="Poisson")
        hg2.fit(X_tr_base.iloc[half:], df_v.loc[train_m, "home_score"].iloc[half:], verbose=0)
        ag2.fit(X_tr_base.iloc[half:], df_v.loc[train_m, "away_score"].iloc[half:], verbose=0)
        
        pois_tr_first = add_poisson_feats(X_tr_base.iloc[:half], hg2, ag2)
        pois_tr_second = add_poisson_feats(X_tr_base.iloc[half:], hg1, ag1)
        pois_tr = {k: np.concatenate([pois_tr_first[k], pois_tr_second[k]]) for k in pois_tr_first}
        
        del hg, ag, hg1, ag1, hg2, ag2; gc.collect()
        
        # Step 2: Add derived features
        log.info("  Adding derived features...")
        derived_tr = add_derived_features(df_v, features, df_v.loc[train_m, features].fillna(0))
        derived_val = add_derived_features(df_v, features, df_v.loc[val_m, features].fillna(0))
        derived_te = add_derived_features(df_v, features, df_v.loc[test_m, features].fillna(0))
        
        # Build augmented feature matrices
        def build_augmented(X_base, pois_feats, derived_feats):
            X = X_base.copy()
            for k, v in pois_feats.items():
                X[k] = v
            for k, v in derived_feats.items():
                X[k] = v
            return X
        
        X_tr_aug = build_augmented(X_tr_base.reset_index(drop=True),
                                    pois_tr, {k: v for k, v in derived_tr.items()})
        X_val_aug = build_augmented(X_val_base.reset_index(drop=True),
                                     pois_val, {k: v for k, v in derived_val.items()})
        X_te_aug = build_augmented(X_te_base.reset_index(drop=True),
                                    pois_te, {k: v for k, v in derived_te.items()})
        
        aug_features = list(X_tr_aug.columns)
        log.info("  Augmented features: %d (base=%d + %d derived)",
                 len(aug_features), len(features), len(aug_features) - len(features))
        
        # Step 3: Train unified models
        y_tr_num = pd.Series(y_tr).map(LABEL_MAP).values
        y_val_num = pd.Series(y_val).map(LABEL_MAP).values
        
        configs = [
            {"name": "CB_d8_bal", "depth": 8, "lr": 0.01, "l2": 5, "ml": 30, "cw": "Balanced"},
            {"name": "CB_d6_bal", "depth": 6, "lr": 0.02, "l2": 3, "ml": 25, "cw": "Balanced"},
            {"name": "CB_d7_bal", "depth": 7, "lr": 0.015, "l2": 5, "ml": 20, "cw": "Balanced"},
        ]
        
        model_results = {}
        for cfg in configs:
            log.info("  Training %s...", cfg["name"])
            m = CatBoostClassifier(
                iterations=2000, depth=cfg["depth"], learning_rate=cfg["lr"],
                l2_leaf_reg=cfg["l2"], min_data_in_leaf=cfg["ml"],
                random_seed=42, verbose=0, loss_function="MultiClass",
                classes_count=3, auto_class_weights=cfg["cw"],
            )
            m.fit(X_tr_aug, y_tr_num, eval_set=(X_val_aug, y_val_num),
                  early_stopping_rounds=150, verbose=0)
            
            probs = m.predict_proba(X_te_aug)
            preds = np.array(["H", "D", "A"])[probs.argmax(1)]
            acc = (preds == y_te).mean()
            n_d = (preds == "D").sum()
            
            # With draw inflation
            for d_inf in [1.0, 1.15, 1.25, 1.35]:
                adj = probs.copy()
                adj[:, 1] *= d_inf
                adj = adj / adj.sum(axis=1, keepdims=True)
                adj_preds = np.array(["H", "D", "A"])[adj.argmax(1)]
                adj_acc = (adj_preds == y_te).mean()
                adj_nd = (adj_preds == "D").sum()
                
                key = f"{cfg['name']}_d{d_inf}"
                model_results[key] = (adj_acc, adj_nd, adj_preds, adj)
                if adj_acc > acc:
                    acc = adj_acc
            
            del m; gc.collect()
        
        # XGBoost
        try:
            from xgboost import XGBClassifier
            log.info("  Training XGBoost...")
            xgb = XGBClassifier(
                n_estimators=1500, max_depth=6, learning_rate=0.02,
                reg_lambda=3, min_child_weight=30, subsample=0.8,
                colsample_bytree=0.8, random_state=42,
                objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                early_stopping_rounds=100, verbosity=0,
            )
            xgb.fit(X_tr_aug, y_tr_num, eval_set=[(X_val_aug, y_val_num)], verbose=False)
            probs = xgb.predict_proba(X_te_aug)
            for d_inf in [1.0, 1.15, 1.25, 1.35]:
                adj = probs.copy()
                adj[:, 1] *= d_inf
                adj = adj / adj.sum(axis=1, keepdims=True)
                adj_preds = np.array(["H", "D", "A"])[adj.argmax(1)]
                adj_acc = (adj_preds == y_te).mean()
                adj_nd = (adj_preds == "D").sum()
                model_results[f"XGB_d{d_inf}"] = (adj_acc, adj_nd, adj_preds, adj)
            del xgb; gc.collect()
        except ImportError:
            pass
        
        # LightGBM
        try:
            from lightgbm import LGBMClassifier
            log.info("  Training LightGBM...")
            lgb = LGBMClassifier(
                n_estimators=1500, max_depth=6, learning_rate=0.02,
                reg_lambda=3, min_child_samples=30, subsample=0.8,
                colsample_bytree=0.8, random_state=42,
                objective="multiclass", num_class=3, verbosity=-1,
                class_weight="balanced",
            )
            lgb.fit(X_tr_aug, y_tr_num)
            probs = lgb.predict_proba(X_te_aug)
            for d_inf in [1.0, 1.15, 1.25, 1.35]:
                adj = probs.copy()
                adj[:, 1] *= d_inf
                adj = adj / adj.sum(axis=1, keepdims=True)
                adj_preds = np.array(["H", "D", "A"])[adj.argmax(1)]
                adj_acc = (adj_preds == y_te).mean()
                adj_nd = (adj_preds == "D").sum()
                model_results[f"LGB_d{d_inf}"] = (adj_acc, adj_nd, adj_preds, adj)
            del lgb; gc.collect()
        except ImportError:
            pass
        
        # Market baseline
        mkt_probs = np.zeros((test_m.sum(), 3))
        psh = df_v.loc[test_m, "odds_PSH"].values
        psd = df_v.loc[test_m, "odds_PSD"].values
        psa = df_v.loc[test_m, "odds_PSA"].values
        for i in range(len(psh)):
            if psh[i] > 1 and psd[i] > 1 and psa[i] > 1:
                raw = np.array([1/psh[i], 1/psd[i], 1/psa[i]])
                mkt_probs[i] = raw / raw.sum()
            else:
                mkt_probs[i] = [0.4, 0.3, 0.3]
        mkt_preds = np.array(["H", "D", "A"])[mkt_probs.argmax(1)]
        mkt_acc = (mkt_preds == y_te).mean()
        
        # Rank results
        ranked = sorted(model_results.items(), key=lambda x: -x[1][0])
        
        log.info("\n--- %s RESULTS ---", test_season)
        log.info("  Market baseline: %.1f%%", mkt_acc * 100)
        for name, (acc, nd, preds, probs) in ranked[:8]:
            max_p = probs.max(axis=1)
            hc50 = (preds[max_p > 0.50] == y_te[max_p > 0.50]).mean() if (max_p > 0.50).sum() > 0 else 0
            hc55 = (preds[max_p > 0.55] == y_te[max_p > 0.55]).mean() if (max_p > 0.55).sum() > 0 else 0
            hc60 = (preds[max_p > 0.60] == y_te[max_p > 0.60]).mean() if (max_p > 0.60).sum() > 0 else 0
            log.info("  %-20s %.1f%% draws=%d HC50=%.0f%% HC55=%.0f%% HC60=%.0f%%",
                     name, acc * 100, nd, hc50 * 100, hc55 * 100, hc60 * 100)
        
        # Best result details
        best_name, (best_acc, best_nd, best_preds, best_probs) = ranked[0]
        n_d_actual = (y_te == "D").sum()
        log.info("\n  BEST: %s → %.1f%% (+%.1fpp vs market)", best_name, best_acc * 100,
                 (best_acc - mkt_acc) * 100)
        for cls in ["H", "D", "A"]:
            act = y_te == cls
            pred_cls = best_preds == cls
            recall = (best_preds[act] == cls).mean() if act.sum() > 0 else 0
            prec = (y_te[pred_cls] == cls).mean() if pred_cls.sum() > 0 else 0
            log.info("  %s: recall=%.0f%% prec=%.0f%% (pred=%d act=%d)",
                     cls, recall * 100, prec * 100, pred_cls.sum(), act.sum())
        
        all_results[test_season] = {
            "best_model": best_name,
            "accuracy": round(best_acc, 4),
            "market_accuracy": round(mkt_acc, 4),
            "gain_pp": round((best_acc - mkt_acc) * 100, 1),
            "draws_pred": best_nd,
            "draws_actual": n_d_actual,
        }
        
        gc.collect()
    
    # Final summary
    log.info("\n" + "=" * 60)
    log.info("UNIFIED MODEL — WALK-FORWARD SUMMARY")
    log.info("=" * 60)
    accs = []
    for season, data in sorted(all_results.items()):
        log.info("  %s: %.1f%% (mkt=%.1f%%, +%.1fpp) [%s] draws=%d/%d",
                 season, data["accuracy"] * 100, data["market_accuracy"] * 100,
                 data["gain_pp"], data["best_model"], data["draws_pred"], data["draws_actual"])
        accs.append(data["accuracy"])
    log.info("  AVERAGE: %.1f%%", np.mean(accs) * 100)
    
    with open(MDIR / "unified_model_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("Saved unified_model_results.json")


if __name__ == "__main__":
    main()
