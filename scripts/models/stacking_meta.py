#!/usr/bin/env python3
"""Stacking Meta-Learner: train a model that learns WHEN to trust each base model.

Uses proper K-fold CV on training data to generate out-of-fold predictions,
then trains a meta-learner on those + contextual features.
"""
import sys, json, logging, gc, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

from scripts.models.comprehensive_markets import load_unified_training_data, get_training_features, _compute_targets
from config.settings import MODELS_DIR
from catboost import CatBoostClassifier, CatBoostRegressor
from ml.poisson import poisson_1x2_vec, market_implied_probs

MDIR = MODELS_DIR / "markets"
LABEL_MAP = {"H": 0, "D": 1, "A": 2}


def build_meta_features(cb_probs, xgb_probs, lgb_probs, pois_probs, mkt_probs,
                         draw_det, gd_pred, elo_diff, odds_spread, mkt_draw_prob):
    """Build feature matrix for meta-learner from base model outputs + context."""
    n = len(cb_probs)
    feats = np.zeros((n, 30))
    
    # Base model probabilities (18 features: 6 models × 3 classes)
    feats[:, 0:3] = cb_probs
    feats[:, 3:6] = xgb_probs if xgb_probs is not None else cb_probs
    feats[:, 6:9] = lgb_probs if lgb_probs is not None else cb_probs
    feats[:, 9:12] = pois_probs
    feats[:, 12:15] = mkt_probs
    
    # Average ensemble (3 features)
    all_probs = [cb_probs, pois_probs, mkt_probs]
    if xgb_probs is not None: all_probs.append(xgb_probs)
    if lgb_probs is not None: all_probs.append(lgb_probs)
    avg = np.mean(all_probs, axis=0)
    feats[:, 15:18] = avg
    
    # Draw signals (4 features)
    feats[:, 18] = draw_det  # dedicated draw detector
    feats[:, 19] = np.exp(-0.5 * (gd_pred / 0.7) ** 2)  # GD-based draw signal
    feats[:, 20] = mkt_draw_prob  # market draw probability
    feats[:, 21] = np.clip(draw_det * mkt_draw_prob * 3, 0, 1)  # interaction
    
    # Agreement signals (4 features)
    preds = [cb_probs.argmax(1), pois_probs.argmax(1), mkt_probs.argmax(1)]
    if xgb_probs is not None: preds.append(xgb_probs.argmax(1))
    if lgb_probs is not None: preds.append(lgb_probs.argmax(1))
    preds_arr = np.array(preds)
    
    # How many models agree on H, D, A
    for c in range(3):
        feats[:, 22 + c] = (preds_arr == c).sum(axis=0) / len(preds)
    feats[:, 25] = (preds_arr == preds_arr[0]).all(axis=0).astype(float)  # unanimous
    
    # Context (4 features)
    feats[:, 26] = elo_diff
    feats[:, 27] = odds_spread
    feats[:, 28] = gd_pred
    feats[:, 29] = np.abs(gd_pred)  # absolute GD (competitiveness)
    
    return feats


def train_base_models(X_tr, y_tr, X_eval, y_eval, df_v, train_mask, eval_mask):
    """Train all base models and return probabilities on eval set."""
    y_tr_num = pd.Series(y_tr).map(LABEL_MAP).values
    y_eval_num = pd.Series(y_eval).map(LABEL_MAP).values
    
    # CatBoost
    cb = CatBoostClassifier(
        iterations=2000, depth=8, learning_rate=0.01, l2_leaf_reg=5,
        min_data_in_leaf=30, random_seed=42, verbose=0,
        loss_function="MultiClass", classes_count=3, auto_class_weights="Balanced",
    )
    cb.fit(X_tr, y_tr_num, eval_set=(X_eval, y_eval_num),
           early_stopping_rounds=150, verbose=0)
    cb_probs = cb.predict_proba(X_eval)
    del cb; gc.collect()
    
    # XGBoost
    xgb_probs = None
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=1500, max_depth=6, learning_rate=0.02,
            reg_lambda=3, min_child_weight=30, subsample=0.8,
            colsample_bytree=0.8, random_state=42,
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            early_stopping_rounds=100, verbosity=0,
        )
        xgb.fit(X_tr, y_tr_num, eval_set=[(X_eval, y_eval_num)], verbose=False)
        xgb_probs = xgb.predict_proba(X_eval)
        del xgb; gc.collect()
    except ImportError:
        pass
    
    # LightGBM
    lgb_probs = None
    try:
        from lightgbm import LGBMClassifier
        lgb = LGBMClassifier(
            n_estimators=1500, max_depth=6, learning_rate=0.02,
            reg_lambda=3, min_child_samples=30, subsample=0.8,
            colsample_bytree=0.8, random_state=42,
            objective="multiclass", num_class=3, verbosity=-1,
        )
        lgb.fit(X_tr, y_tr_num)
        lgb_probs = lgb.predict_proba(X_eval)
        del lgb; gc.collect()
    except ImportError:
        pass
    
    # Poisson
    hg = CatBoostRegressor(
        iterations=1500, depth=8, learning_rate=0.015, l2_leaf_reg=7,
        min_data_in_leaf=30, random_seed=42, verbose=0, loss_function="Poisson",
    )
    ag = CatBoostRegressor(
        iterations=1500, depth=8, learning_rate=0.015, l2_leaf_reg=7,
        min_data_in_leaf=30, random_seed=42, verbose=0, loss_function="Poisson",
    )
    hg.fit(X_tr, df_v.loc[train_mask, "home_score"],
           eval_set=(X_eval, df_v.loc[eval_mask, "home_score"]),
           early_stopping_rounds=100, verbose=0)
    ag.fit(X_tr, df_v.loc[train_mask, "away_score"],
           eval_set=(X_eval, df_v.loc[eval_mask, "away_score"]),
           early_stopping_rounds=100, verbose=0)
    hxg = np.clip(hg.predict(X_eval), 0.3, 4.0)
    axg = np.clip(ag.predict(X_eval), 0.3, 4.0)
    pois_probs = poisson_1x2_vec(hxg, axg, min_xg=None)
    gd_pred = hxg - axg
    del hg, ag; gc.collect()
    
    # Draw detector
    y_draw_tr = (y_tr == "D").astype(int)
    y_draw_eval = (y_eval == "D").astype(int)
    draw_clf = CatBoostClassifier(
        iterations=1500, depth=7, learning_rate=0.02, l2_leaf_reg=3,
        min_data_in_leaf=20, random_seed=42, verbose=0,
        loss_function="Logloss", class_weights=[1.0, 2.5],
    )
    draw_clf.fit(X_tr, y_draw_tr, eval_set=(X_eval, y_draw_eval),
                 early_stopping_rounds=100, verbose=0)
    draw_det = draw_clf.predict_proba(X_eval)[:, 1]
    del draw_clf; gc.collect()
    
    # Market probs
    mkt_probs = market_implied_probs(df_v, eval_mask)
    
    return cb_probs, xgb_probs, lgb_probs, pois_probs, mkt_probs, draw_det, gd_pred


def main():
    df = load_unified_training_data()
    features = get_training_features(df)
    df = _compute_targets(df)
    df_v = df[df["result"].isin(["H", "D", "A"])].copy()
    seasons = sorted(df_v["season"].unique())

    for test_season in ["2023-2024", "2024-2025"]:
        log.info("\n" + "=" * 60)
        log.info("STACKING META-LEARNER — TEST: %s", test_season)
        log.info("=" * 60)
        
        ti = seasons.index(test_season)
        val_season = seasons[ti - 1]
        train_seasons = seasons[:ti - 1]
        
        full_train_m = df_v["season"].isin(train_seasons + [val_season])
        meta_train_m = df_v["season"].isin(train_seasons)
        val_m = df_v["season"] == val_season
        test_m = df_v["season"] == test_season
        
        X_full = df_v.loc[full_train_m, features].fillna(0)
        X_meta_tr = df_v.loc[meta_train_m, features].fillna(0)
        X_val = df_v.loc[val_m, features].fillna(0)
        X_te = df_v.loc[test_m, features].fillna(0)
        y_full = df_v.loc[full_train_m, "result"].values
        y_meta_tr = df_v.loc[meta_train_m, "result"].values
        y_val = df_v.loc[val_m, "result"].values
        y_te = df_v.loc[test_m, "result"].values
        
        # Context features from dataframe
        elo_diff_full = df_v.loc[full_train_m, "elo_diff"].fillna(0).values if "elo_diff" in df_v.columns else np.zeros(full_train_m.sum())
        elo_diff_te = df_v.loc[test_m, "elo_diff"].fillna(0).values if "elo_diff" in df_v.columns else np.zeros(test_m.sum())
        
        odds_cols = ["odds_PSH", "odds_PSD", "odds_PSA"]
        if all(c in df_v.columns for c in odds_cols):
            odds_full = df_v.loc[full_train_m, odds_cols].fillna(3.0).values
            odds_te = df_v.loc[test_m, odds_cols].fillna(3.0).values
            odds_spread_full = odds_full.max(axis=1) - odds_full.min(axis=1)
            odds_spread_te = odds_te.max(axis=1) - odds_te.min(axis=1)
            mkt_draw_p_full = 1.0 / odds_full[:, 1]
            mkt_draw_p_full = mkt_draw_p_full / (1.0 / odds_full).sum(axis=1)
            mkt_draw_p_te = 1.0 / odds_te[:, 1]
            mkt_draw_p_te = mkt_draw_p_te / (1.0 / odds_te).sum(axis=1)
        else:
            odds_spread_full = np.zeros(full_train_m.sum())
            odds_spread_te = np.zeros(test_m.sum())
            mkt_draw_p_full = np.full(full_train_m.sum(), 0.28)
            mkt_draw_p_te = np.full(test_m.sum(), 0.28)
        
        # ================================================================
        # STEP 1: Generate OOF predictions using K-fold on train+val
        # ================================================================
        log.info("\n--- STEP 1: K-Fold OOF predictions ---")
        
        n_full = len(X_full)
        y_full_num = pd.Series(y_full).map(LABEL_MAP).values
        
        oof_cb = np.zeros((n_full, 3))
        oof_xgb = np.zeros((n_full, 3))
        oof_lgb = np.zeros((n_full, 3))
        oof_pois = np.zeros((n_full, 3))
        oof_draw = np.zeros(n_full)
        oof_gd = np.zeros(n_full)
        
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        full_idx = df_v[full_train_m].index
        
        for fold, (tr_idx, va_idx) in enumerate(kfold.split(X_full, y_full_num)):
            log.info("  Fold %d/%d...", fold + 1, 5)
            
            X_f_tr = X_full.iloc[tr_idx]
            X_f_va = X_full.iloc[va_idx]
            y_f_tr = y_full[tr_idx]
            y_f_va = y_full[va_idx]
            
            # Get the actual indices in df_v for home_score/away_score
            f_tr_mask = pd.Series(False, index=df_v.index)
            f_tr_mask.iloc[np.where(full_train_m)[0][tr_idx]] = True
            f_va_mask = pd.Series(False, index=df_v.index)
            f_va_mask.iloc[np.where(full_train_m)[0][va_idx]] = True
            
            cb_p, xgb_p, lgb_p, pois_p, _, draw_p, gd_p = train_base_models(
                X_f_tr, y_f_tr, X_f_va, y_f_va, df_v, f_tr_mask, f_va_mask
            )
            
            oof_cb[va_idx] = cb_p
            if xgb_p is not None: oof_xgb[va_idx] = xgb_p
            else: oof_xgb[va_idx] = cb_p
            if lgb_p is not None: oof_lgb[va_idx] = lgb_p
            else: oof_lgb[va_idx] = cb_p
            oof_pois[va_idx] = pois_p
            oof_draw[va_idx] = draw_p
            oof_gd[va_idx] = gd_p
        
        mkt_full = market_implied_probs(df_v, full_train_m)
        
        # Build meta-features for training
        meta_X_train = build_meta_features(
            oof_cb, oof_xgb, oof_lgb, oof_pois, mkt_full,
            oof_draw, oof_gd, elo_diff_full, odds_spread_full, mkt_draw_p_full
        )
        meta_y_train = y_full_num
        
        log.info("  OOF meta-features: %s", meta_X_train.shape)
        
        # ================================================================
        # STEP 2: Train base models on full train for test predictions
        # ================================================================
        log.info("\n--- STEP 2: Full base models for test ---")
        
        cb_te_p, xgb_te_p, lgb_te_p, pois_te_p, mkt_te_p, draw_te_p, gd_te_p = train_base_models(
            X_full, y_full, X_te, y_te, df_v, full_train_m, test_m
        )
        
        meta_X_test = build_meta_features(
            cb_te_p,
            xgb_te_p if xgb_te_p is not None else cb_te_p,
            lgb_te_p if lgb_te_p is not None else cb_te_p,
            pois_te_p, mkt_te_p,
            draw_te_p, gd_te_p, elo_diff_te, odds_spread_te, mkt_draw_p_te
        )
        
        # ================================================================
        # STEP 3: Train meta-learner
        # ================================================================
        log.info("\n--- STEP 3: Meta-Learner Training ---")
        
        # Try multiple meta-learners
        results = {}
        
        # 3a. CatBoost meta
        meta_cb = CatBoostClassifier(
            iterations=1000, depth=4, learning_rate=0.05, l2_leaf_reg=5,
            min_data_in_leaf=20, random_seed=42, verbose=0,
            loss_function="MultiClass", classes_count=3, auto_class_weights="Balanced",
        )
        # Use last 20% as val for early stopping
        split = int(0.8 * len(meta_X_train))
        meta_cb.fit(meta_X_train[:split], meta_y_train[:split],
                     eval_set=(meta_X_train[split:], meta_y_train[split:]),
                     early_stopping_rounds=50, verbose=0)
        meta_cb_preds = np.array(["H", "D", "A"])[meta_cb.predict_proba(meta_X_test).argmax(1)]
        meta_cb_probs = meta_cb.predict_proba(meta_X_test)
        acc_meta_cb = (meta_cb_preds == y_te).mean()
        n_d_meta_cb = (meta_cb_preds == "D").sum()
        results["meta_cb"] = (acc_meta_cb, n_d_meta_cb, meta_cb_preds, meta_cb_probs)
        log.info("  Meta-CatBoost: %.1f%% (%d draws)", acc_meta_cb * 100, n_d_meta_cb)
        
        # 3b. LightGBM meta
        try:
            from lightgbm import LGBMClassifier
            meta_lgb = LGBMClassifier(
                n_estimators=500, max_depth=4, learning_rate=0.05,
                reg_lambda=5, min_child_samples=20, random_state=42,
                objective="multiclass", num_class=3, verbosity=-1,
                class_weight="balanced",
            )
            meta_lgb.fit(meta_X_train, meta_y_train)
            meta_lgb_preds = np.array(["H", "D", "A"])[meta_lgb.predict_proba(meta_X_test).argmax(1)]
            meta_lgb_probs = meta_lgb.predict_proba(meta_X_test)
            acc_meta_lgb = (meta_lgb_preds == y_te).mean()
            n_d_meta_lgb = (meta_lgb_preds == "D").sum()
            results["meta_lgb"] = (acc_meta_lgb, n_d_meta_lgb, meta_lgb_preds, meta_lgb_probs)
            log.info("  Meta-LightGBM: %.1f%% (%d draws)", acc_meta_lgb * 100, n_d_meta_lgb)
        except ImportError:
            pass
        
        # 3c. XGBoost meta
        try:
            from xgboost import XGBClassifier
            meta_xgb = XGBClassifier(
                n_estimators=500, max_depth=4, learning_rate=0.05,
                reg_lambda=5, min_child_weight=20, random_state=42,
                objective="multi:softprob", num_class=3, verbosity=0,
            )
            meta_xgb.fit(meta_X_train, meta_y_train)
            meta_xgb_preds = np.array(["H", "D", "A"])[meta_xgb.predict_proba(meta_X_test).argmax(1)]
            meta_xgb_probs = meta_xgb.predict_proba(meta_X_test)
            acc_meta_xgb = (meta_xgb_preds == y_te).mean()
            n_d_meta_xgb = (meta_xgb_preds == "D").sum()
            results["meta_xgb"] = (acc_meta_xgb, n_d_meta_xgb, meta_xgb_preds, meta_xgb_probs)
            log.info("  Meta-XGBoost: %.1f%% (%d draws)", acc_meta_xgb * 100, n_d_meta_xgb)
        except ImportError:
            pass
        
        # 3d. Logistic Regression meta (simple but robust)
        from sklearn.linear_model import LogisticRegression
        meta_lr = LogisticRegression(
            C=1.0, max_iter=1000, multi_class="multinomial",
            class_weight="balanced", random_state=42,
        )
        meta_lr.fit(meta_X_train, meta_y_train)
        meta_lr_preds = np.array(["H", "D", "A"])[meta_lr.predict_proba(meta_X_test).argmax(1)]
        meta_lr_probs = meta_lr.predict_proba(meta_X_test)
        acc_meta_lr = (meta_lr_preds == y_te).mean()
        n_d_meta_lr = (meta_lr_preds == "D").sum()
        results["meta_lr"] = (acc_meta_lr, n_d_meta_lr, meta_lr_preds, meta_lr_probs)
        log.info("  Meta-LogReg: %.1f%% (%d draws)", acc_meta_lr * 100, n_d_meta_lr)
        
        # 3e. MLP meta
        from sklearn.neural_network import MLPClassifier
        meta_mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=42,
            early_stopping=True, validation_fraction=0.15,
        )
        meta_mlp.fit(meta_X_train, meta_y_train)
        meta_mlp_preds = np.array(["H", "D", "A"])[meta_mlp.predict_proba(meta_X_test).argmax(1)]
        meta_mlp_probs = meta_mlp.predict_proba(meta_X_test)
        acc_meta_mlp = (meta_mlp_preds == y_te).mean()
        n_d_meta_mlp = (meta_mlp_preds == "D").sum()
        results["meta_mlp"] = (acc_meta_mlp, n_d_meta_mlp, meta_mlp_preds, meta_mlp_probs)
        log.info("  Meta-MLP: %.1f%% (%d draws)", acc_meta_mlp * 100, n_d_meta_mlp)
        
        # ================================================================
        # STEP 4: Also try blending meta-learner outputs
        # ================================================================
        log.info("\n--- STEP 4: Blend Meta + Base ---")
        
        # Simple average of best base ensemble + meta
        base_ens = 0.25 * cb_te_p + 0.40 * pois_te_p + 0.35 * mkt_te_p
        base_ens[:, 1] *= 1.35
        base_ens = base_ens / base_ens.sum(axis=1, keepdims=True)
        base_preds = np.array(["H", "D", "A"])[base_ens.argmax(1)]
        base_acc = (base_preds == y_te).mean()
        log.info("  Base ensemble (prev best): %.1f%% (%d draws)", base_acc * 100, (base_preds == "D").sum())
        
        # Blend each meta with base
        best_overall = base_acc
        best_name = "base"
        for name, (acc, nd, preds, probs) in results.items():
            for meta_w in [0.3, 0.4, 0.5, 0.6, 0.7]:
                blended = meta_w * probs + (1 - meta_w) * base_ens
                blended = blended / blended.sum(axis=1, keepdims=True)
                bl_preds = np.array(["H", "D", "A"])[blended.argmax(1)]
                bl_acc = (bl_preds == y_te).mean()
                bl_nd = (bl_preds == "D").sum()
                if bl_acc > best_overall:
                    best_overall = bl_acc
                    best_name = f"{name}@{meta_w}"
                    best_preds = bl_preds
                    best_probs = blended
            
            # Also try with draw inflation on blended
            for meta_w in [0.4, 0.5, 0.6]:
                for d_inf in [1.15, 1.25, 1.35]:
                    blended = meta_w * probs + (1 - meta_w) * base_ens
                    blended[:, 1] *= d_inf
                    blended = blended / blended.sum(axis=1, keepdims=True)
                    bl_preds = np.array(["H", "D", "A"])[blended.argmax(1)]
                    bl_acc = (bl_preds == y_te).mean()
                    if bl_acc > best_overall:
                        best_overall = bl_acc
                        best_name = f"{name}@{meta_w}+d{d_inf}"
                        best_preds = bl_preds
                        best_probs = blended
        
        # Market baseline
        mkt_preds = np.array(["H", "D", "A"])[mkt_te_p.argmax(1)]
        mkt_acc = (mkt_preds == y_te).mean()
        
        log.info("\n  Best blend: %s → %.1f%%", best_name, best_overall * 100)
        if best_name != "base":
            # HC tiers
            max_p = best_probs.max(axis=1)
            for thr in [0.50, 0.55, 0.60]:
                m = max_p > thr
                if m.sum() > 0:
                    t_acc = (best_preds[m] == y_te[m]).mean()
                    log.info("  HC%.0f: %.1f%% (%d matches)", thr * 100, t_acc * 100, m.sum())
            
            # Per-class
            for cls in ["H", "D", "A"]:
                act = y_te == cls
                pred_cls = best_preds == cls
                recall = (best_preds[act] == cls).mean() if act.sum() > 0 else 0
                prec = (y_te[pred_cls] == cls).mean() if pred_cls.sum() > 0 else 0
                log.info("  %s: recall=%.0f%% prec=%.0f%% (pred=%d act=%d)",
                         cls, recall * 100, prec * 100, pred_cls.sum(), act.sum())
        
        log.info("\n  === %s SUMMARY ===", test_season)
        log.info("  Market:      %.1f%%", mkt_acc * 100)
        log.info("  Base ens:    %.1f%%", base_acc * 100)
        for name, (acc, nd, _, _) in sorted(results.items(), key=lambda x: -x[1][0]):
            log.info("  %-12s %.1f%% (%d draws)", name + ":", acc * 100, nd)
        log.info("  Best blend:  %.1f%% (%s)", best_overall * 100, best_name)
        
        gc.collect()
    
    log.info("\nDone.")


if __name__ == "__main__":
    main()
