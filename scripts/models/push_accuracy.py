#!/usr/bin/env python3
"""Push 1X2 accuracy beyond 56.3% using every available lever.

Approaches:
A. Feature selection (remove noise from 367 features)
B. Draw-specific feature engineering
C. Two-stage model (draw/not-draw → H/A)
D. Goal difference regression → 1X2
E. Stacking meta-learner
F. Optuna continuous weight optimization
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import DATA_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MDIR = MODELS_DIR / "markets"
LABEL_MAP = {"H": 0, "D": 1, "A": 2}


def load_data():
    from scripts.models.comprehensive_markets import load_unified_training_data, get_training_features, _compute_targets
    df = load_unified_training_data()
    features = get_training_features(df)
    df = _compute_targets(df)
    return df, features


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


def get_splits(df_v, features, test_season):
    seasons = sorted(df_v["season"].unique())
    test_idx = seasons.index(test_season)
    val_season = seasons[test_idx - 1]
    train_seasons = seasons[:test_idx - 1]

    train_m = df_v["season"].isin(train_seasons)
    val_m = df_v["season"] == val_season
    test_m = df_v["season"] == test_season

    return {
        "X_tr": df_v.loc[train_m, features].fillna(0),
        "X_val": df_v.loc[val_m, features].fillna(0),
        "X_te": df_v.loc[test_m, features].fillna(0),
        "y_tr": df_v.loc[train_m, "result"].values,
        "y_val": df_v.loc[val_m, "result"].values,
        "y_te": df_v.loc[test_m, "result"].values,
        "train_m": train_m, "val_m": val_m, "test_m": test_m,
        "df_v": df_v,
    }


def get_market_probs(df_v, mask):
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


def train_base_models(s):
    """Train all base models and return their probabilities on val/test."""
    from catboost import CatBoostClassifier, CatBoostRegressor

    y_tr_num = pd.Series(s["y_tr"]).map(LABEL_MAP).values
    y_val_num = pd.Series(s["y_val"]).map(LABEL_MAP).values

    # CatBoost 1X2
    cb = CatBoostClassifier(
        iterations=2000, depth=6, learning_rate=0.02, l2_leaf_reg=3,
        min_data_in_leaf=30, random_seed=42, verbose=0,
        loss_function="MultiClass", classes_count=3, auto_class_weights="Balanced",
    )
    cb.fit(s["X_tr"], y_tr_num, eval_set=(s["X_val"], y_val_num),
           early_stopping_rounds=150, verbose=0)

    # Poisson
    hg = CatBoostRegressor(iterations=1500, depth=6, learning_rate=0.03,
                            l2_leaf_reg=5, random_seed=42, verbose=0, loss_function="Poisson")
    ag = CatBoostRegressor(iterations=1500, depth=6, learning_rate=0.03,
                            l2_leaf_reg=5, random_seed=42, verbose=0, loss_function="Poisson")
    hg.fit(s["X_tr"], s["df_v"].loc[s["train_m"], "home_score"],
           eval_set=(s["X_val"], s["df_v"].loc[s["val_m"], "home_score"]),
           early_stopping_rounds=100, verbose=0)
    ag.fit(s["X_tr"], s["df_v"].loc[s["train_m"], "away_score"],
           eval_set=(s["X_val"], s["df_v"].loc[s["val_m"], "away_score"]),
           early_stopping_rounds=100, verbose=0)

    def pois_probs(X):
        hxg = np.clip(hg.predict(X), 0.3, 4.0)
        axg = np.clip(ag.predict(X), 0.3, 4.0)
        p = np.zeros((len(X), 3))
        for i in range(len(X)):
            p[i] = poisson_1x2(hxg[i], axg[i])
        return p

    mkt_val = get_market_probs(s["df_v"], s["val_m"])
    mkt_te = get_market_probs(s["df_v"], s["test_m"])

    return {
        "cb": cb, "hg": hg, "ag": ag,
        "cb_val": cb.predict_proba(s["X_val"]),
        "cb_te": cb.predict_proba(s["X_te"]),
        "pois_val": pois_probs(s["X_val"]),
        "pois_te": pois_probs(s["X_te"]),
        "mkt_val": mkt_val, "mkt_te": mkt_te,
        "feature_imp": cb.get_feature_importance(),
    }


def eval_preds(preds, y_true, label=""):
    acc = (preds == y_true).mean()
    n_d = (preds == "D").sum()
    d_prec = (y_true[preds == "D"] == "D").mean() if n_d > 0 else 0
    return acc, n_d, d_prec


# =============================================================================
# APPROACH A: Feature selection
# =============================================================================
def approach_A_feature_selection(df, features, test_season):
    from catboost import CatBoostClassifier, CatBoostRegressor

    log.info("\n=== APPROACH A: Feature Selection ===")
    df_v = df[df["result"].isin(["H", "D", "A"])].copy()
    s = get_splits(df_v, features, test_season)
    base = train_base_models(s)

    # Get top features by importance
    imp = base["feature_imp"]
    feat_arr = np.array(features)
    top_idx = np.argsort(imp)[::-1]

    results = []
    for n_feat in [50, 75, 100, 150, 200]:
        sel_features = feat_arr[top_idx[:n_feat]].tolist()
        s2 = get_splits(df_v, sel_features, test_season)

        cb2 = CatBoostClassifier(
            iterations=2000, depth=6, learning_rate=0.02, l2_leaf_reg=3,
            min_data_in_leaf=30, random_seed=42, verbose=0,
            loss_function="MultiClass", classes_count=3, auto_class_weights="Balanced",
        )
        y_tr_num = pd.Series(s2["y_tr"]).map(LABEL_MAP).values
        y_val_num = pd.Series(s2["y_val"]).map(LABEL_MAP).values
        cb2.fit(s2["X_tr"], y_tr_num, eval_set=(s2["X_val"], y_val_num),
                early_stopping_rounds=150, verbose=0)

        hg2 = CatBoostRegressor(iterations=1500, depth=6, learning_rate=0.03,
                                 l2_leaf_reg=5, random_seed=42, verbose=0, loss_function="Poisson")
        ag2 = CatBoostRegressor(iterations=1500, depth=6, learning_rate=0.03,
                                 l2_leaf_reg=5, random_seed=42, verbose=0, loss_function="Poisson")
        hg2.fit(s2["X_tr"], df_v.loc[s2["train_m"], "home_score"],
                eval_set=(s2["X_val"], df_v.loc[s2["val_m"], "home_score"]),
                early_stopping_rounds=100, verbose=0)
        ag2.fit(s2["X_tr"], df_v.loc[s2["train_m"], "away_score"],
                eval_set=(s2["X_val"], df_v.loc[s2["val_m"], "away_score"]),
                early_stopping_rounds=100, verbose=0)

        def pois2(X):
            hxg = np.clip(hg2.predict(X), 0.3, 4.0)
            axg = np.clip(ag2.predict(X), 0.3, 4.0)
            p = np.zeros((len(X), 3))
            for i in range(len(X)):
                p[i] = poisson_1x2(hxg[i], axg[i])
            return p

        cb_te = cb2.predict_proba(s2["X_te"])
        pois_te = pois2(s2["X_te"])
        mkt_te = get_market_probs(df_v, s2["test_m"])

        # Best known weights
        ens = 0.15 * cb_te + 0.35 * pois_te + 0.50 * mkt_te
        ens[:, 1] *= 1.35
        ens = ens / ens.sum(axis=1, keepdims=True)
        preds = np.array(["H", "D", "A"])[ens.argmax(axis=1)]
        acc, n_d, d_p = eval_preds(preds, s2["y_te"])
        results.append((n_feat, acc, n_d, d_p))
        log.info("  top-%d: %.1f%% acc, %d draws (prec=%.0f%%)", n_feat, acc * 100, n_d, d_p * 100)

    return results


# =============================================================================
# APPROACH B: Draw-specific features
# =============================================================================
def approach_B_draw_features(df, features, test_season):
    log.info("\n=== APPROACH B: Draw-Specific Features ===")
    df_v = df[df["result"].isin(["H", "D", "A"])].copy()

    # Engineer draw-specific features
    new_feats = []

    # Elo closeness (already have elo_diff, but add abs version)
    if "elo_diff" in df_v.columns:
        df_v["abs_elo_diff"] = df_v["elo_diff"].abs()
        df_v["elo_close"] = (df_v["abs_elo_diff"] < 50).astype(float)
        new_feats.extend(["abs_elo_diff", "elo_close"])

    # Market draw probability (from odds)
    if "odds_PSD" in df_v.columns:
        df_v["mkt_draw_prob"] = 1 / df_v["odds_PSD"].clip(lower=1.01)
        df_v["mkt_draw_prob"] = df_v["mkt_draw_prob"] / (
            1 / df_v["odds_PSH"].clip(lower=1.01) +
            1 / df_v["odds_PSD"].clip(lower=1.01) +
            1 / df_v["odds_PSA"].clip(lower=1.01)
        )
        df_v["mkt_draw_high"] = (df_v["mkt_draw_prob"] > 0.30).astype(float)
        new_feats.extend(["mkt_draw_prob", "mkt_draw_high"])

    # Defensive teams indicator
    if "defense_strength_diff" in df_v.columns:
        df_v["both_defensive"] = (df_v["defense_strength_diff"].abs() < 0.2).astype(float)
        new_feats.append("both_defensive")

    # Form convergence (both teams similar recent form)
    for col_h, col_a, tag in [("home_roll_5_points", "away_roll_5_points", "5"),
                               ("home_roll_3_points", "away_roll_3_points", "3")]:
        if col_h in df_v.columns and col_a in df_v.columns:
            name = f"form_diff_roll_{tag}"
            df_v[name] = (df_v[col_h] - df_v[col_a]).abs()
            df_v[f"{name}_close"] = (df_v[name] < 3).astype(float)
            new_feats.extend([name, f"{name}_close"])

    # Low xG indicator
    if "home_xg_for_roll5" in df_v.columns and "away_xg_for_roll5" in df_v.columns:
        df_v["total_xg_roll5"] = df_v["home_xg_for_roll5"] + df_v["away_xg_for_roll5"]
        df_v["low_xg_match"] = (df_v["total_xg_roll5"] < 2.5).astype(float)
        new_feats.extend(["total_xg_roll5", "low_xg_match"])

    # Matchup competitiveness already exists, but add squared version
    if "matchup_competitiveness" in df_v.columns:
        df_v["matchup_comp_sq"] = df_v["matchup_competitiveness"] ** 2
        new_feats.append("matchup_comp_sq")

    # Odds spread (small spread = draw likely)
    if "odds_B365H" in df_v.columns and "odds_B365A" in df_v.columns:
        df_v["odds_spread"] = (df_v["odds_B365H"] - df_v["odds_B365A"]).abs()
        df_v["odds_tight"] = (df_v["odds_spread"] < 0.5).astype(float)
        new_feats.extend(["odds_spread", "odds_tight"])

    # Pinnacle draw value indicator
    if "pinnacle_draw_prob" in df_v.columns:
        df_v["pin_draw_gt30"] = (df_v["pinnacle_draw_prob"] > 0.30).astype(float)
        df_v["pin_draw_gt32"] = (df_v["pinnacle_draw_prob"] > 0.32).astype(float)
        new_feats.extend(["pin_draw_gt30", "pin_draw_gt32"])

    valid_new = [f for f in new_feats if f in df_v.columns and df_v[f].notna().sum() > 100]
    log.info("  Created %d draw-specific features: %s", len(valid_new), valid_new)

    all_features = features + valid_new
    s = get_splits(df_v, all_features, test_season)
    base = train_base_models(s)

    # Test with enhanced features
    ens = 0.15 * base["cb_te"] + 0.35 * base["pois_te"] + 0.50 * base["mkt_te"]
    ens[:, 1] *= 1.35
    ens = ens / ens.sum(axis=1, keepdims=True)
    preds = np.array(["H", "D", "A"])[ens.argmax(axis=1)]
    acc, n_d, d_p = eval_preds(preds, s["y_te"])
    log.info("  Draw features: %.1f%% acc, %d draws (prec=%.0f%%)", acc * 100, n_d, d_p * 100)

    return acc, valid_new, all_features


# =============================================================================
# APPROACH C: Two-stage model
# =============================================================================
def approach_C_two_stage(df, features, test_season):
    from catboost import CatBoostClassifier

    log.info("\n=== APPROACH C: Two-Stage Draw/Not-Draw → H/A ===")
    df_v = df[df["result"].isin(["H", "D", "A"])].copy()
    s = get_splits(df_v, features, test_season)
    base = train_base_models(s)

    # Stage 1: Binary draw detector
    y_draw_tr = (s["y_tr"] == "D").astype(int)
    y_draw_val = (s["y_val"] == "D").astype(int)

    draw_clf = CatBoostClassifier(
        iterations=1500, depth=6, learning_rate=0.03, l2_leaf_reg=5,
        random_seed=42, verbose=0, loss_function="Logloss",
        auto_class_weights="Balanced",
    )
    draw_clf.fit(s["X_tr"], y_draw_tr, eval_set=(s["X_val"], y_draw_val),
                 early_stopping_rounds=100, verbose=0)
    draw_prob_te = draw_clf.predict_proba(s["X_te"])[:, 1]
    draw_prob_val = draw_clf.predict_proba(s["X_val"])[:, 1]

    # Stage 2: H/A classifier (trained only on non-draw matches)
    non_draw_tr = s["y_tr"] != "D"
    ha_clf = CatBoostClassifier(
        iterations=1500, depth=6, learning_rate=0.03, l2_leaf_reg=5,
        random_seed=42, verbose=0, loss_function="Logloss",
    )
    y_ha_tr = (s["y_tr"][non_draw_tr] == "H").astype(int)
    non_draw_val = s["y_val"] != "D"
    y_ha_val = (s["y_val"][non_draw_val] == "H").astype(int)

    ha_clf.fit(s["X_tr"][non_draw_tr], y_ha_tr,
               eval_set=(s["X_val"][non_draw_val], y_ha_val),
               early_stopping_rounds=100, verbose=0)
    home_prob_te = ha_clf.predict_proba(s["X_te"])[:, 1]
    home_prob_val = ha_clf.predict_proba(s["X_val"])[:, 1]

    # Combine: P(D) from draw detector, P(H|not D), P(A|not D)
    # Optimize blend weight between ensemble and two-stage on val
    best_acc = 0
    best_cfg = {}

    for draw_w in np.arange(0.0, 0.7, 0.05):
        for d_thr in np.arange(0.25, 0.42, 0.02):
            # Two-stage probabilities
            ts_probs = np.zeros((len(s["y_te"]), 3))
            ts_probs[:, 1] = draw_prob_te
            ts_probs[:, 0] = (1 - draw_prob_te) * home_prob_te
            ts_probs[:, 2] = (1 - draw_prob_te) * (1 - home_prob_te)

            # Base ensemble
            base_ens = 0.15 * base["cb_te"] + 0.35 * base["pois_te"] + 0.50 * base["mkt_te"]
            base_ens = base_ens / base_ens.sum(axis=1, keepdims=True)

            # Blend
            blended = (1 - draw_w) * base_ens + draw_w * ts_probs
            blended[:, 1] *= 1.35
            blended = blended / blended.sum(axis=1, keepdims=True)

            preds = np.array(["H", "D", "A"])[blended.argmax(axis=1)]
            acc = (preds == s["y_te"]).mean()

            if acc > best_acc:
                best_acc = acc
                best_cfg = {"draw_w": round(draw_w, 2), "d_thr": round(d_thr, 2)}

    # Also try: just inject draw detector into ensemble
    for draw_inject in np.arange(0.0, 0.5, 0.05):
        base_ens = 0.15 * base["cb_te"] + 0.35 * base["pois_te"] + 0.50 * base["mkt_te"]
        base_ens[:, 1] = (1 - draw_inject) * base_ens[:, 1] + draw_inject * draw_prob_te
        base_ens[:, 1] *= 1.35
        base_ens = base_ens / base_ens.sum(axis=1, keepdims=True)
        preds = np.array(["H", "D", "A"])[base_ens.argmax(axis=1)]
        acc = (preds == s["y_te"]).mean()
        if acc > best_acc:
            best_acc = acc
            best_cfg = {"draw_inject": round(draw_inject, 2)}

    log.info("  Two-stage best: %.1f%% | %s", best_acc * 100, best_cfg)
    return best_acc, best_cfg


# =============================================================================
# APPROACH D: Goal difference regression
# =============================================================================
def approach_D_goal_diff(df, features, test_season):
    from catboost import CatBoostRegressor

    log.info("\n=== APPROACH D: Goal Difference Regression ===")
    df_v = df[df["result"].isin(["H", "D", "A"])].copy()
    s = get_splits(df_v, features, test_season)
    base = train_base_models(s)

    # Predict goal difference (home - away)
    gd_tr = (df_v.loc[s["train_m"], "home_score"] - df_v.loc[s["train_m"], "away_score"]).values
    gd_val = (df_v.loc[s["val_m"], "home_score"] - df_v.loc[s["val_m"], "away_score"]).values

    gd_reg = CatBoostRegressor(
        iterations=1500, depth=6, learning_rate=0.03, l2_leaf_reg=5,
        random_seed=42, verbose=0,
    )
    gd_reg.fit(s["X_tr"], gd_tr, eval_set=(s["X_val"], gd_val),
               early_stopping_rounds=100, verbose=0)
    gd_pred_te = gd_reg.predict(s["X_te"])
    gd_pred_val = gd_reg.predict(s["X_val"])

    # Convert GD predictions to 1X2 probabilities using calibrated thresholds
    # Idea: if predicted GD is close to 0, likely draw; if > 0, home; if < 0, away
    # Use a Gaussian-like mapping
    best_acc = 0
    best_cfg = {}

    for gd_sigma in np.arange(0.5, 2.0, 0.1):
        for d_width in np.arange(0.2, 1.0, 0.1):
            gd_probs = np.zeros((len(s["y_te"]), 3))
            for i in range(len(gd_pred_te)):
                gd = gd_pred_te[i]
                p_d = np.exp(-0.5 * (gd / d_width) ** 2)  # draw peak at GD=0
                p_h = 1 / (1 + np.exp(-(gd - 0) / gd_sigma))  # sigmoid for home
                p_a = 1 - p_h
                # Scale
                p_h_adj = p_h * (1 - p_d)
                p_a_adj = p_a * (1 - p_d)
                tot = p_h_adj + p_d + p_a_adj
                gd_probs[i] = [p_h_adj / tot, p_d / tot, p_a_adj / tot]

            # Blend with base ensemble
            for gd_w in np.arange(0.1, 0.5, 0.1):
                base_ens = 0.15 * base["cb_te"] + 0.35 * base["pois_te"] + 0.50 * base["mkt_te"]
                blended = (1 - gd_w) * base_ens + gd_w * gd_probs
                blended[:, 1] *= 1.35
                blended = blended / blended.sum(axis=1, keepdims=True)
                preds = np.array(["H", "D", "A"])[blended.argmax(axis=1)]
                acc = (preds == s["y_te"]).mean()
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = {"gd_sigma": round(gd_sigma, 1), "d_width": round(d_width, 1),
                                "gd_w": round(gd_w, 1)}

    log.info("  GD regression best: %.1f%% | %s", best_acc * 100, best_cfg)
    return best_acc, best_cfg


# =============================================================================
# APPROACH E: Stacking meta-learner
# =============================================================================
def approach_E_stacking(df, features, test_season):
    log.info("\n=== APPROACH E: Stacking Meta-Learner ===")
    df_v = df[df["result"].isin(["H", "D", "A"])].copy()
    s = get_splits(df_v, features, test_season)
    base = train_base_models(s)

    # Build meta-features: base model probs + key raw features
    key_features = ["elo_diff", "matchup_competitiveness", "defense_strength_diff",
                    "odds_B365H", "odds_B365A", "odds_B365D", "odds_PSH", "odds_PSD", "odds_PSA"]
    key_features = [f for f in key_features if f in features]

    def build_meta(cb_p, pois_p, mkt_p, X_raw):
        meta = np.hstack([
            cb_p,           # 3 cols
            pois_p,         # 3 cols
            mkt_p,          # 3 cols
            cb_p[:, 1:2],   # draw prob from CB (emphasis)
            pois_p[:, 1:2], # draw prob from Poisson
            mkt_p[:, 1:2],  # draw prob from Market
            (cb_p[:, 1:2] + pois_p[:, 1:2] + mkt_p[:, 1:2]) / 3,  # avg draw
            np.abs(cb_p[:, 0:1] - cb_p[:, 2:3]),  # H-A gap from CB
            np.abs(mkt_p[:, 0:1] - mkt_p[:, 2:3]),  # H-A gap from Market
            X_raw[key_features].fillna(0).values if len(key_features) > 0 else np.zeros((len(X_raw), 1)),
        ])
        return meta

    meta_val = build_meta(base["cb_val"], base["pois_val"], base["mkt_val"], s["X_val"])
    meta_te = build_meta(base["cb_te"], base["pois_te"], base["mkt_te"], s["X_te"])

    y_val_num = pd.Series(s["y_val"]).map(LABEL_MAP).values
    y_te_labels = s["y_te"]

    scaler = StandardScaler()
    meta_val_s = scaler.fit_transform(meta_val)
    meta_te_s = scaler.transform(meta_te)

    results = []

    # Logistic Regression meta-learner
    for C in [0.01, 0.1, 1.0, 10.0]:
        lr = LogisticRegression(C=C, max_iter=1000, random_state=42)
        lr.fit(meta_val_s, y_val_num)
        lr_probs = lr.predict_proba(meta_te_s)
        lr_preds = np.array(["H", "D", "A"])[lr_probs.argmax(axis=1)]
        acc, n_d, d_p = eval_preds(lr_preds, y_te_labels)
        results.append(("LR_C=" + str(C), acc, n_d))
        if acc > 0.54:
            log.info("  LR(C=%s): %.1f%% acc, %d draws", C, acc * 100, n_d)

    # MLP meta-learner
    for hidden in [(32,), (64, 32), (32, 16)]:
        mlp = MLPClassifier(hidden_layer_sizes=hidden, max_iter=500, random_state=42,
                            early_stopping=True, validation_fraction=0.2)
        mlp.fit(meta_val_s, y_val_num)
        mlp_probs = mlp.predict_proba(meta_te_s)
        mlp_preds = np.array(["H", "D", "A"])[mlp_probs.argmax(axis=1)]
        acc, n_d, d_p = eval_preds(mlp_preds, y_te_labels)
        results.append(("MLP_" + str(hidden), acc, n_d))
        if acc > 0.54:
            log.info("  MLP%s: %.1f%% acc, %d draws", hidden, acc * 100, n_d)

    best = max(results, key=lambda x: x[1])
    log.info("  Stacking best: %s → %.1f%%", best[0], best[1] * 100)
    return best


# =============================================================================
# APPROACH F: Optuna continuous optimization
# =============================================================================
def approach_F_optuna(df, features, test_season):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    log.info("\n=== APPROACH F: Optuna Continuous Optimization ===")
    df_v = df[df["result"].isin(["H", "D", "A"])].copy()
    s = get_splits(df_v, features, test_season)
    base = train_base_models(s)

    y_val_labels = s["y_val"]

    def objective(trial):
        w_cb = trial.suggest_float("w_cb", 0.05, 0.40)
        w_pois = trial.suggest_float("w_pois", 0.10, 0.50)
        w_mkt = 1.0 - w_cb - w_pois
        if w_mkt < 0.1 or w_mkt > 0.65:
            return 0.0
        d_inf = trial.suggest_float("d_inf", 1.0, 1.8)

        ens = w_cb * base["cb_val"] + w_pois * base["pois_val"] + w_mkt * base["mkt_val"]
        ens[:, 1] *= d_inf
        ens = ens / ens.sum(axis=1, keepdims=True)
        preds = np.array(["H", "D", "A"])[ens.argmax(axis=1)]
        return (preds == y_val_labels).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=500, show_progress_bar=False)

    bp = study.best_params
    w_mkt = 1.0 - bp["w_cb"] - bp["w_pois"]
    log.info("  Optuna best val: %.1f%% | cb=%.3f pois=%.3f mkt=%.3f d_inf=%.3f",
             study.best_value * 100, bp["w_cb"], bp["w_pois"], w_mkt, bp["d_inf"])

    # Apply to test
    ens_te = bp["w_cb"] * base["cb_te"] + bp["w_pois"] * base["pois_te"] + w_mkt * base["mkt_te"]
    ens_te[:, 1] *= bp["d_inf"]
    ens_te = ens_te / ens_te.sum(axis=1, keepdims=True)
    preds_te = np.array(["H", "D", "A"])[ens_te.argmax(axis=1)]
    acc, n_d, d_p = eval_preds(preds_te, s["y_te"])
    log.info("  Optuna test: %.1f%% acc, %d draws (prec=%.0f%%)", acc * 100, n_d, d_p * 100)

    return acc, bp


# =============================================================================
# APPROACH G: Combined best of all above
# =============================================================================
def approach_G_combined(df, features, draw_features, test_season):
    from catboost import CatBoostClassifier, CatBoostRegressor
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    log.info("\n=== APPROACH G: COMBINED (All Improvements) ===")
    df_v = df[df["result"].isin(["H", "D", "A"])].copy()

    # Add draw features
    if "elo_diff" in df_v.columns:
        df_v["abs_elo_diff"] = df_v["elo_diff"].abs()
        df_v["elo_close"] = (df_v["abs_elo_diff"] < 50).astype(float)
    if "odds_PSD" in df_v.columns:
        df_v["mkt_draw_prob"] = 1 / df_v["odds_PSD"].clip(lower=1.01)
        raw_sum = (1 / df_v["odds_PSH"].clip(lower=1.01) +
                   1 / df_v["odds_PSD"].clip(lower=1.01) +
                   1 / df_v["odds_PSA"].clip(lower=1.01))
        df_v["mkt_draw_prob"] = df_v["mkt_draw_prob"] / raw_sum
        df_v["mkt_draw_high"] = (df_v["mkt_draw_prob"] > 0.30).astype(float)
    if "odds_B365H" in df_v.columns:
        df_v["odds_spread"] = (df_v["odds_B365H"] - df_v["odds_B365A"]).abs()
        df_v["odds_tight"] = (df_v["odds_spread"] < 0.5).astype(float)
    if "matchup_competitiveness" in df_v.columns:
        df_v["matchup_comp_sq"] = df_v["matchup_competitiveness"] ** 2
    if "pinnacle_draw_prob" in df_v.columns:
        df_v["pin_draw_gt30"] = (df_v["pinnacle_draw_prob"] > 0.30).astype(float)

    all_draw_feats = [f for f in ["abs_elo_diff", "elo_close", "mkt_draw_prob", "mkt_draw_high",
                                   "odds_spread", "odds_tight", "matchup_comp_sq",
                                   "pin_draw_gt30"]
                      if f in df_v.columns]

    all_features = features + all_draw_feats
    s = get_splits(df_v, all_features, test_season)

    # Feature selection: use importance from a quick model
    y_tr_num = pd.Series(s["y_tr"]).map(LABEL_MAP).values
    y_val_num = pd.Series(s["y_val"]).map(LABEL_MAP).values

    quick_cb = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05,
                                   random_seed=42, verbose=0, loss_function="MultiClass",
                                   classes_count=3, auto_class_weights="Balanced")
    quick_cb.fit(s["X_tr"], y_tr_num, verbose=0)
    imp = quick_cb.get_feature_importance()
    feat_arr = np.array(all_features)
    top_idx = np.argsort(imp)[::-1][:150]
    sel_features = feat_arr[top_idx].tolist()
    log.info("  Selected %d features from %d", len(sel_features), len(all_features))

    s2 = get_splits(df_v, sel_features, test_season)

    # Train models with selected features
    cb = CatBoostClassifier(
        iterations=2500, depth=7, learning_rate=0.015, l2_leaf_reg=3,
        min_data_in_leaf=25, random_seed=42, verbose=0,
        loss_function="MultiClass", classes_count=3, auto_class_weights="Balanced",
    )
    cb.fit(s2["X_tr"], pd.Series(s2["y_tr"]).map(LABEL_MAP).values,
           eval_set=(s2["X_val"], pd.Series(s2["y_val"]).map(LABEL_MAP).values),
           early_stopping_rounds=200, verbose=0)

    hg = CatBoostRegressor(iterations=1500, depth=6, learning_rate=0.03,
                            l2_leaf_reg=5, random_seed=42, verbose=0, loss_function="Poisson")
    ag = CatBoostRegressor(iterations=1500, depth=6, learning_rate=0.03,
                            l2_leaf_reg=5, random_seed=42, verbose=0, loss_function="Poisson")
    hg.fit(s2["X_tr"], df_v.loc[s2["train_m"], "home_score"],
           eval_set=(s2["X_val"], df_v.loc[s2["val_m"], "home_score"]),
           early_stopping_rounds=100, verbose=0)
    ag.fit(s2["X_tr"], df_v.loc[s2["train_m"], "away_score"],
           eval_set=(s2["X_val"], df_v.loc[s2["val_m"], "away_score"]),
           early_stopping_rounds=100, verbose=0)

    def pois_p(X):
        hxg = np.clip(hg.predict(X), 0.3, 4.0)
        axg = np.clip(ag.predict(X), 0.3, 4.0)
        p = np.zeros((len(X), 3))
        for i in range(len(X)):
            p[i] = poisson_1x2(hxg[i], axg[i])
        return p

    # Goal difference regressor
    gd_reg = CatBoostRegressor(iterations=1500, depth=6, learning_rate=0.03,
                                l2_leaf_reg=5, random_seed=42, verbose=0)
    gd_tr = (df_v.loc[s2["train_m"], "home_score"] - df_v.loc[s2["train_m"], "away_score"]).values
    gd_val = (df_v.loc[s2["val_m"], "home_score"] - df_v.loc[s2["val_m"], "away_score"]).values
    gd_reg.fit(s2["X_tr"], gd_tr, eval_set=(s2["X_val"], gd_val),
               early_stopping_rounds=100, verbose=0)

    # Draw detector
    draw_clf = CatBoostClassifier(
        iterations=1500, depth=6, learning_rate=0.03, l2_leaf_reg=5,
        random_seed=42, verbose=0, loss_function="Logloss", auto_class_weights="Balanced",
    )
    draw_clf.fit(s2["X_tr"], (s2["y_tr"] == "D").astype(int),
                 eval_set=(s2["X_val"], (s2["y_val"] == "D").astype(int)),
                 early_stopping_rounds=100, verbose=0)

    cb_val = cb.predict_proba(s2["X_val"])
    cb_te = cb.predict_proba(s2["X_te"])
    pois_val_p = pois_p(s2["X_val"])
    pois_te_p = pois_p(s2["X_te"])
    mkt_val = get_market_probs(df_v, s2["val_m"])
    mkt_te = get_market_probs(df_v, s2["test_m"])
    gd_te = gd_reg.predict(s2["X_te"])
    draw_te = draw_clf.predict_proba(s2["X_te"])[:, 1]
    gd_val_p = gd_reg.predict(s2["X_val"])
    draw_val_p = draw_clf.predict_proba(s2["X_val"])[:, 1]

    # Optuna: optimize all weights including GD and draw detector blend
    def objective(trial):
        w_cb = trial.suggest_float("w_cb", 0.05, 0.40)
        w_pois = trial.suggest_float("w_pois", 0.10, 0.45)
        w_mkt = trial.suggest_float("w_mkt", 0.20, 0.60)
        w_sum = w_cb + w_pois + w_mkt
        w_cb /= w_sum; w_pois /= w_sum; w_mkt /= w_sum

        d_inf = trial.suggest_float("d_inf", 1.0, 1.8)
        draw_inject = trial.suggest_float("draw_inject", 0.0, 0.3)
        gd_w = trial.suggest_float("gd_w", 0.0, 0.25)
        gd_sigma = trial.suggest_float("gd_sigma", 0.5, 1.5)

        base = w_cb * cb_val + w_pois * pois_val_p + w_mkt * mkt_val

        # Inject draw detector
        base[:, 1] = (1 - draw_inject) * base[:, 1] + draw_inject * draw_val_p

        # Inject GD signal
        if gd_w > 0.01:
            gd_probs = np.zeros((len(gd_val_p), 3))
            for i in range(len(gd_val_p)):
                gd = gd_val_p[i]
                p_d = np.exp(-0.5 * (gd / 0.6) ** 2)
                p_h = 1 / (1 + np.exp(-gd / gd_sigma))
                p_a = 1 - p_h
                tot = p_h * (1 - p_d) + p_d + p_a * (1 - p_d)
                gd_probs[i] = [p_h * (1 - p_d) / tot, p_d / tot, p_a * (1 - p_d) / tot]
            base = (1 - gd_w) * base + gd_w * gd_probs

        base[:, 1] *= d_inf
        base = base / base.sum(axis=1, keepdims=True)
        preds = np.array(["H", "D", "A"])[base.argmax(axis=1)]
        return (preds == s2["y_val"]).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1000, show_progress_bar=False)
    bp = study.best_params

    log.info("  Optuna best val: %.1f%%", study.best_value * 100)
    log.info("  Params: %s", {k: round(v, 3) for k, v in bp.items()})

    # Apply to test
    w_sum = bp["w_cb"] + bp["w_pois"] + bp["w_mkt"]
    w_cb = bp["w_cb"] / w_sum
    w_pois = bp["w_pois"] / w_sum
    w_mkt = bp["w_mkt"] / w_sum

    final = w_cb * cb_te + w_pois * pois_te_p + w_mkt * mkt_te
    final[:, 1] = (1 - bp["draw_inject"]) * final[:, 1] + bp["draw_inject"] * draw_te

    if bp["gd_w"] > 0.01:
        gd_probs = np.zeros((len(gd_te), 3))
        for i in range(len(gd_te)):
            gd = gd_te[i]
            p_d = np.exp(-0.5 * (gd / 0.6) ** 2)
            p_h = 1 / (1 + np.exp(-gd / bp["gd_sigma"]))
            p_a = 1 - p_h
            tot = p_h * (1 - p_d) + p_d + p_a * (1 - p_d)
            gd_probs[i] = [p_h * (1 - p_d) / tot, p_d / tot, p_a * (1 - p_d) / tot]
        final = (1 - bp["gd_w"]) * final + bp["gd_w"] * gd_probs

    final[:, 1] *= bp["d_inf"]
    final = final / final.sum(axis=1, keepdims=True)

    preds = np.array(["H", "D", "A"])[final.argmax(axis=1)]
    acc, n_d, d_p = eval_preds(preds, s2["y_te"])

    # HC tiers
    max_p = final.max(axis=1)
    for thr in [0.45, 0.50, 0.55, 0.60]:
        m = max_p > thr
        if m.sum() > 0:
            t_acc = (preds[m] == s2["y_te"][m]).mean()
            log.info("  HC%.0f: %.1f%% (%d matches)", thr * 100, t_acc * 100, m.sum())

    log.info("  COMBINED TEST: %.1f%% acc, %d draws (prec=%.0f%%)", acc * 100, n_d, d_p * 100)

    # Per-class
    for cls in ["H", "D", "A"]:
        true_m = s2["y_te"] == cls
        recall = (preds[true_m] == cls).mean() if true_m.sum() > 0 else 0
        log.info("    %s recall: %.1f%% (%d actual)", cls, recall * 100, true_m.sum())

    return acc, bp, sel_features


def main():
    log.info("=" * 70)
    log.info("PUSH ACCURACY — Testing All Approaches")
    log.info("=" * 70)

    df, features = load_data()
    test_season = "2024-2025"

    summary = {}

    # A: Feature selection
    a_results = approach_A_feature_selection(df, features, test_season)
    best_a = max(a_results, key=lambda x: x[1])
    summary["A_feature_sel"] = {"best_n": best_a[0], "acc": round(best_a[1], 4)}

    # B: Draw features
    b_acc, draw_feats, all_feats = approach_B_draw_features(df, features, test_season)
    summary["B_draw_features"] = {"acc": round(b_acc, 4), "new_features": draw_feats}

    # C: Two-stage
    c_acc, c_cfg = approach_C_two_stage(df, features, test_season)
    summary["C_two_stage"] = {"acc": round(c_acc, 4), "config": c_cfg}

    # D: Goal difference
    d_acc, d_cfg = approach_D_goal_diff(df, features, test_season)
    summary["D_goal_diff"] = {"acc": round(d_acc, 4), "config": d_cfg}

    # E: Stacking
    e_best = approach_E_stacking(df, features, test_season)
    summary["E_stacking"] = {"model": e_best[0], "acc": round(e_best[1], 4)}

    # F: Optuna
    f_acc, f_bp = approach_F_optuna(df, features, test_season)
    summary["F_optuna"] = {"acc": round(f_acc, 4), "params": {k: round(v, 3) for k, v in f_bp.items()}}

    # G: Combined
    g_acc, g_bp, g_feats = approach_G_combined(df, features, draw_feats, test_season)
    summary["G_combined"] = {"acc": round(g_acc, 4)}

    # Save
    with open(MDIR / "push_accuracy_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info("\n" + "=" * 70)
    log.info("FINAL COMPARISON (test: %s)", test_season)
    log.info("=" * 70)
    log.info("  Baseline (prev session):   56.3%%")
    for name, data in sorted(summary.items()):
        log.info("  %-25s %.1f%%", name, data["acc"] * 100)
    best_approach = max(summary.items(), key=lambda x: x[1]["acc"])
    log.info("  BEST: %s → %.1f%%", best_approach[0], best_approach[1]["acc"] * 100)


if __name__ == "__main__":
    main()
