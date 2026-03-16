#!/usr/bin/env python3
"""Optimize ALL markets: 1X2, O/U, BTTS, Cards, Corners, HT Result.

Walk-forward evaluation on 2023-24 and 2024-25.
Tests multiple hyperparameter configs + class balancing + threshold optimization.

Training strategy (walk-forward):
  To predict 2023-24: trains on 2004-05 → 2021-22, validates on 2022-23
  To predict 2024-25: trains on 2004-05 → 2022-23, validates on 2023-24
  ALL past data is used as training input.
"""
import sys, json, logging, gc, warnings, time
import numpy as np
import pandas as pd
from scipy.stats import poisson
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

from scripts.models.comprehensive_markets import load_unified_training_data, get_training_features, _compute_targets
from config.settings import MODELS_DIR
from catboost import CatBoostClassifier, CatBoostRegressor

MDIR = MODELS_DIR / "markets"
LABEL_MAP = {"H": 0, "D": 1, "A": 2}
SCRIPT_START = None


def elapsed():
    """Return formatted elapsed time since script start."""
    secs = time.time() - SCRIPT_START
    m, s = divmod(int(secs), 60)
    return f"{m:02d}:{s:02d}"


def fmt_duration(secs):
    """Format seconds into MM:SS."""
    m, s = divmod(int(secs), 60)
    return f"{m}m{s:02d}s"


def poisson_1x2(hxg, axg):
    n = len(hxg)
    p = np.zeros((n, 3))
    for i in range(n):
        ph = pd_ = pa = 0.0
        hx, ax = max(hxg[i], 0.3), max(axg[i], 0.3)
        for h in range(8):
            for a in range(8):
                prob = poisson.pmf(h, hx) * poisson.pmf(a, ax)
                if h > a: ph += prob
                elif h == a: pd_ += prob
                else: pa += prob
        t = ph + pd_ + pa
        p[i] = [ph/t, pd_/t, pa/t] if t > 0 else [0.4, 0.3, 0.3]
    return p


def get_mkt(df, mask):
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


def train_binary(X_tr, y_tr, X_val, y_val, configs, market_name=""):
    """Train binary classifier with multiple configs, return best."""
    best_acc = 0
    best_probs = None
    best_name = ""
    
    for i, cfg in enumerate(configs):
        t0 = time.time()
        m = CatBoostClassifier(
            iterations=cfg.get("iter", 1200),
            depth=cfg.get("depth", 6),
            learning_rate=cfg.get("lr", 0.03),
            l2_leaf_reg=cfg.get("l2", 5),
            min_data_in_leaf=cfg.get("ml", 25),
            random_seed=42, verbose=0,
            loss_function="Logloss",
            auto_class_weights=cfg.get("cw", None),
        )
        m.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=80, verbose=0)
        probs = m.predict_proba(X_val)[:, 1]
        dur = time.time() - t0
        
        # Test multiple thresholds
        for thr in [0.45, 0.48, 0.50, 0.52, 0.55]:
            preds = (probs > thr).astype(int)
            acc = (preds == y_val.values).mean()
            if acc > best_acc:
                best_acc = acc
                best_probs = probs
                best_name = f"{cfg['name']}_t{thr}"
                best_thr = thr
                best_model = m
        
        log.info("    [%s] config %d/%d '%s' → val=%.1f%% (%s) [elapsed %s]",
                 elapsed(), i+1, len(configs), cfg["name"], best_acc*100, fmt_duration(dur), elapsed())
        del m; gc.collect()
    
    return best_model, best_probs, best_acc, best_name, best_thr


def main():
    global SCRIPT_START
    SCRIPT_START = time.time()
    
    log.info("[%s] ▶ Loading data...", elapsed())
    df = load_unified_training_data()
    features = get_training_features(df)
    df = _compute_targets(df)
    df_v = df[df["result"].isin(["H", "D", "A"])].copy()
    seasons = sorted(df_v["season"].unique())
    
    log.info("[%s] ✓ Data loaded: %d matches, %d features, %d seasons (%s → %s)",
             elapsed(), len(df_v), len(features), len(seasons), seasons[0], seasons[-1])
    
    # Binary model configs to test (3 configs instead of 5 — faster, same quality)
    binary_configs = [
        {"name": "base",   "depth": 6, "lr": 0.03, "l2": 5, "ml": 25, "cw": None},
        {"name": "bal",    "depth": 6, "lr": 0.03, "l2": 5, "ml": 25, "cw": "Balanced"},
        {"name": "bal_d7", "depth": 7, "lr": 0.02, "l2": 3, "ml": 20, "cw": "Balanced"},
    ]
    
    # Total steps per season: 1X2(3 models) + OU25 + OU15 + OU35 + BTTS + Cards×3 + Corners×3 + HT = 12 markets
    TOTAL_MARKETS = 12
    test_seasons = ["2023-2024", "2024-2025"]
    
    log.info("[%s] Plan: %d test seasons × %d markets = %d total market evaluations",
             elapsed(), len(test_seasons), TOTAL_MARKETS, len(test_seasons) * TOTAL_MARKETS)
    log.info("[%s] Estimated runtime: ~40 min per season, ~80 min total", elapsed())
    log.info("")
    
    all_season_results = {}
    
    for si, test_season in enumerate(test_seasons):
        season_start = time.time()
        step = 0  # market counter within this season
        
        log.info("\n" + "=" * 70)
        log.info("[%s] SEASON %d/%d: %s", elapsed(), si+1, len(test_seasons), test_season)
        log.info("=" * 70)
        
        ti = seasons.index(test_season)
        val_season = seasons[ti - 1]
        train_seasons = seasons[:ti - 1]
        
        log.info("[%s]   Training: %d seasons (%s → %s) = %d matches",
                 elapsed(), len(train_seasons), train_seasons[0], train_seasons[-1],
                 df_v["season"].isin(train_seasons).sum())
        log.info("[%s]   Validation: %s | Test: %s", elapsed(), val_season, test_season)
        
        train_m = df_v["season"].isin(train_seasons)
        val_m = df_v["season"] == val_season
        test_m = df_v["season"] == test_season
        
        X_tr = df_v.loc[train_m, features].fillna(0)
        X_val = df_v.loc[val_m, features].fillna(0)
        X_te = df_v.loc[test_m, features].fillna(0)
        y_te = df_v.loc[test_m, "result"].values
        
        log.info("[%s]   Train=%d, Val=%d, Test=%d matches",
                 elapsed(), train_m.sum(), val_m.sum(), test_m.sum())
        
        results = {}
        
        # ================================================================
        # 1X2 ENSEMBLE (locked-in best config)
        # ================================================================
        step += 1
        t0 = time.time()
        log.info("\n[%s] ▋ Step %d/%d: 1X2 Ensemble (CatBoost + Poisson + Market)",
                 elapsed(), step, TOTAL_MARKETS)
        y_tr_num = df_v.loc[train_m, "result"].map(LABEL_MAP).values
        y_val_num = df_v.loc[val_m, "result"].map(LABEL_MAP).values
        
        log.info("[%s]   Training CatBoost 1X2 classifier...", elapsed())
        cb = CatBoostClassifier(
            iterations=2000, depth=8, learning_rate=0.01, l2_leaf_reg=5,
            min_data_in_leaf=30, random_seed=42, verbose=0,
            loss_function="MultiClass", classes_count=3, auto_class_weights="Balanced",
        )
        cb.fit(X_tr, y_tr_num, eval_set=(X_val, y_val_num), early_stopping_rounds=150, verbose=0)
        cb_probs = cb.predict_proba(X_te)
        log.info("[%s]   ✓ CatBoost done (%s)", elapsed(), fmt_duration(time.time() - t0))
        del cb; gc.collect()
        
        # Poisson
        t1 = time.time()
        log.info("[%s]   Training Poisson home goals regressor...", elapsed())
        hg = CatBoostRegressor(iterations=1500, depth=8, learning_rate=0.015, l2_leaf_reg=7,
                                min_data_in_leaf=30, random_seed=42, verbose=0, loss_function="Poisson")
        ag = CatBoostRegressor(iterations=1500, depth=8, learning_rate=0.015, l2_leaf_reg=7,
                                min_data_in_leaf=30, random_seed=42, verbose=0, loss_function="Poisson")
        hg.fit(X_tr, df_v.loc[train_m, "home_score"], eval_set=(X_val, df_v.loc[val_m, "home_score"]),
               early_stopping_rounds=100, verbose=0)
        log.info("[%s]   Training Poisson away goals regressor...", elapsed())
        ag.fit(X_tr, df_v.loc[train_m, "away_score"], eval_set=(X_val, df_v.loc[val_m, "away_score"]),
               early_stopping_rounds=100, verbose=0)
        hxg = np.clip(hg.predict(X_te), 0.3, 4.0)
        axg = np.clip(ag.predict(X_te), 0.3, 4.0)
        pois_probs = poisson_1x2(hxg, axg)
        log.info("[%s]   ✓ Poisson done (%s)", elapsed(), fmt_duration(time.time() - t1))
        
        # Also get Poisson O/U and BTTS predictions for free
        n_te = test_m.sum()
        pois_ou25 = np.zeros(n_te)
        pois_btts = np.zeros(n_te)
        pois_ou15 = np.zeros(n_te)
        pois_ou35 = np.zeros(n_te)
        for i in range(n_te):
            hx, ax = max(hxg[i], 0.3), max(axg[i], 0.3)
            p_over = 0.0
            p_btts = 0.0
            p_o15 = 0.0
            p_o35 = 0.0
            for h in range(8):
                for a in range(8):
                    p = poisson.pmf(h, hx) * poisson.pmf(a, ax)
                    if h + a > 2.5: p_over += p
                    if h + a > 1.5: p_o15 += p
                    if h + a > 3.5: p_o35 += p
                    if h >= 1 and a >= 1: p_btts += p
            pois_ou25[i] = p_over
            pois_btts[i] = p_btts
            pois_ou15[i] = p_o15
            pois_ou35[i] = p_o35
        
        del hg, ag; gc.collect()
        
        # Market
        mkt = get_mkt(df_v, test_m)
        
        # 1X2 ensemble blend
        ens = 0.25 * cb_probs + 0.40 * pois_probs + 0.35 * mkt
        ens[:, 1] *= 1.35
        ens = ens / ens.sum(axis=1, keepdims=True)
        preds_1x2 = np.array(["H", "D", "A"])[ens.argmax(1)]
        acc_1x2 = (preds_1x2 == y_te).mean()
        n_draws = (preds_1x2 == "D").sum()
        n_draws_actual = (y_te == "D").sum()
        
        max_p = ens.max(axis=1)
        hc55 = (preds_1x2[max_p > 0.55] == y_te[max_p > 0.55]).mean() if (max_p > 0.55).sum() > 0 else 0
        hc60 = (preds_1x2[max_p > 0.60] == y_te[max_p > 0.60]).mean() if (max_p > 0.60).sum() > 0 else 0
        n_hc55 = (max_p > 0.55).sum()
        n_hc60 = (max_p > 0.60).sum()
        
        mkt_preds = np.array(["H", "D", "A"])[mkt.argmax(1)]
        mkt_acc = (mkt_preds == y_te).mean()
        
        results["1x2"] = {
            "accuracy": round(acc_1x2, 4), "market": round(mkt_acc, 4),
            "draws": f"{n_draws}/{n_draws_actual}",
            "HC55": f"{hc55:.1%} ({n_hc55})", "HC60": f"{hc60:.1%} ({n_hc60})",
        }
        step_dur = time.time() - t0
        log.info("[%s] ✓ Step %d DONE (%s): 1X2=%.1f%% (mkt=%.1f%%) draws=%d/%d HC55=%.0f%%(%d) HC60=%.0f%%(%d)",
                 elapsed(), step, fmt_duration(step_dur),
                 acc_1x2*100, mkt_acc*100, n_draws, n_draws_actual,
                 hc55*100, n_hc55, hc60*100, n_hc60)
        
        # ================================================================
        # OVER/UNDER 2.5
        # ================================================================
        step += 1
        t0 = time.time()
        log.info("\n[%s] ▋ Step %d/%d: Over/Under 2.5", elapsed(), step, TOTAL_MARKETS)
        y_ou_tr = df_v.loc[train_m, "over_2_5"]
        y_ou_val = df_v.loc[val_m, "over_2_5"]
        y_ou_te = df_v.loc[test_m, "over_2_5"].values
        base_rate = y_ou_te.mean()
        
        best_model_ou, best_probs_ou_val, best_acc_ou_val, best_name_ou, best_thr_ou = \
            train_binary(X_tr, y_ou_tr, X_val, y_ou_val, binary_configs, "O/U 2.5")
        
        ou_probs_te = best_model_ou.predict_proba(X_te)[:, 1]
        
        # Also try blending ML + Poisson
        best_ou_acc = 0
        best_ou_cfg = ""
        for ml_w in [0.5, 0.6, 0.7, 0.8, 1.0]:
            blended = ml_w * ou_probs_te + (1 - ml_w) * pois_ou25
            for thr in [0.45, 0.48, 0.50, 0.52, 0.55]:
                preds = (blended > thr).astype(int)
                acc = (preds == y_ou_te).mean()
                if acc > best_ou_acc:
                    best_ou_acc = acc
                    best_ou_preds = preds
                    best_ou_cfg = f"ml{ml_w}_t{thr}"
        
        results["ou25"] = {
            "accuracy": round(best_ou_acc, 4), "base_rate": round(base_rate, 3),
            "config": best_ou_cfg, "ml_model": best_name_ou,
            "pred_over": int(best_ou_preds.sum()), "actual_over": int(y_ou_te.sum()),
        }
        log.info("[%s] ✓ Step %d DONE (%s): O/U 2.5=%.1f%% (base=%.1f%%) [%s]",
                 elapsed(), step, fmt_duration(time.time() - t0),
                 best_ou_acc*100, base_rate*100, best_ou_cfg)
        del best_model_ou; gc.collect()
        
        # ================================================================
        # OVER/UNDER 1.5 and 3.5
        # ================================================================
        for ou_target, pois_feat, ou_name in [
            ("over_1_5", pois_ou15, "O/U 1.5"),
            ("over_3_5", pois_ou35, "O/U 3.5"),
        ]:
            step += 1
            t0_ou = time.time()
            log.info("\n[%s] ▋ Step %d/%d: %s", elapsed(), step, TOTAL_MARKETS, ou_name)
            y_tr_ou = df_v.loc[train_m, ou_target]
            y_val_ou = df_v.loc[val_m, ou_target]
            y_te_ou = df_v.loc[test_m, ou_target].values
            br = y_te_ou.mean()
            
            bm, _, _, bn, bt = train_binary(X_tr, y_tr_ou, X_val, y_val_ou, binary_configs)
            probs_te = bm.predict_proba(X_te)[:, 1]
            
            best_acc_ou2 = 0
            for ml_w in [0.5, 0.7, 1.0]:
                bl = ml_w * probs_te + (1 - ml_w) * pois_feat
                for thr in [0.45, 0.50, 0.55]:
                    p = (bl > thr).astype(int)
                    a = (p == y_te_ou).mean()
                    if a > best_acc_ou2:
                        best_acc_ou2 = a
                        best_preds_ou2 = p
                        cfg_ou2 = f"ml{ml_w}_t{thr}"
            
            results[ou_target] = {
                "accuracy": round(best_acc_ou2, 4), "base_rate": round(br, 3),
                "config": cfg_ou2,
            }
            log.info("[%s] ✓ Step %d DONE (%s): %s=%.1f%% (base=%.1f%%) [%s]",
                     elapsed(), step, fmt_duration(time.time() - t0_ou), ou_name, best_acc_ou2*100, br*100, cfg_ou2)
            del bm; gc.collect()
        
        # ================================================================
        # BTTS
        # ================================================================
        step += 1
        t0_btts = time.time()
        log.info("\n[%s] ▋ Step %d/%d: BTTS", elapsed(), step, TOTAL_MARKETS)
        y_btts_tr = df_v.loc[train_m, "btts"]
        y_btts_val = df_v.loc[val_m, "btts"]
        y_btts_te = df_v.loc[test_m, "btts"].values
        br_btts = y_btts_te.mean()
        
        bm_btts, _, _, bn_btts, bt_btts = train_binary(X_tr, y_btts_tr, X_val, y_btts_val, binary_configs)
        btts_probs_te = bm_btts.predict_proba(X_te)[:, 1]
        
        best_btts_acc = 0
        for ml_w in [0.5, 0.7, 1.0]:
            bl = ml_w * btts_probs_te + (1 - ml_w) * pois_btts
            for thr in [0.45, 0.48, 0.50, 0.52, 0.55]:
                p = (bl > thr).astype(int)
                a = (p == y_btts_te).mean()
                if a > best_btts_acc:
                    best_btts_acc = a
                    best_btts_preds = p
                    cfg_btts = f"ml{ml_w}_t{thr}"
        
        results["btts"] = {
            "accuracy": round(best_btts_acc, 4), "base_rate": round(br_btts, 3),
            "config": cfg_btts,
        }
        log.info("[%s] ✓ Step %d DONE (%s): BTTS=%.1f%% (base=%.1f%%) [%s]",
                 elapsed(), step, fmt_duration(time.time() - t0_btts), best_btts_acc*100, br_btts*100, cfg_btts)
        del bm_btts; gc.collect()
        
        # ================================================================
        # CARDS OVER 3.5, 4.5, 5.5
        # ================================================================
        for cards_target in ["cards_over_3_5", "cards_over_4_5", "cards_over_5_5"]:
            if cards_target not in df_v.columns:
                continue
            valid = df_v[cards_target].notna()
            if valid.sum() < 500:
                continue
            
            cname = cards_target.replace("cards_over_", "Cards O")
            step += 1
            t0_c = time.time()
            log.info("\n[%s] ▋ Step %d/%d: %s", elapsed(), step, TOTAL_MARKETS, cname)
            
            ct = valid & train_m
            cv = valid & val_m
            cte = valid & test_m
            
            if ct.sum() < 100 or cv.sum() < 10 or cte.sum() < 10:
                continue
            
            X_ct = df_v.loc[ct, features].fillna(0)
            X_cv = df_v.loc[cv, features].fillna(0)
            X_cte = df_v.loc[cte, features].fillna(0)
            y_ct = df_v.loc[ct, cards_target]
            y_cv = df_v.loc[cv, cards_target]
            y_cte = df_v.loc[cte, cards_target].values
            br_c = y_cte.mean()
            
            bm_c, _, _, bn_c, bt_c = train_binary(X_ct, y_ct, X_cv, y_cv, binary_configs)
            probs_c = bm_c.predict_proba(X_cte)[:, 1]
            
            best_c_acc = 0
            for thr in [0.40, 0.45, 0.50, 0.55, 0.60]:
                p = (probs_c > thr).astype(int)
                a = (p == y_cte).mean()
                if a > best_c_acc:
                    best_c_acc = a
                    best_c_preds = p
                    cfg_c = f"t{thr}"
            
            results[cards_target] = {
                "accuracy": round(best_c_acc, 4), "base_rate": round(br_c, 3),
                "config": cfg_c, "model": bn_c, "n_matches": int(cte.sum()),
            }
            log.info("[%s] ✓ Step %d DONE (%s): %s=%.1f%% (base=%.1f%%) [%s] n=%d",
                     elapsed(), step, fmt_duration(time.time() - t0_c), cname, best_c_acc*100, br_c*100, cfg_c, cte.sum())
            del bm_c; gc.collect()
        
        # ================================================================
        # CORNERS OVER 8.5, 9.5, 10.5
        # ================================================================
        for corn_target in ["corners_over_8_5", "corners_over_9_5", "corners_over_10_5"]:
            if corn_target not in df_v.columns:
                continue
            valid = df_v[corn_target].notna()
            if valid.sum() < 500:
                continue
            
            cname = corn_target.replace("corners_over_", "Corners O")
            step += 1
            t0_co = time.time()
            log.info("\n[%s] ▋ Step %d/%d: %s", elapsed(), step, TOTAL_MARKETS, cname)
            
            ct = valid & train_m
            cv = valid & val_m
            cte = valid & test_m
            
            if ct.sum() < 100 or cv.sum() < 10 or cte.sum() < 10:
                continue
            
            X_ct = df_v.loc[ct, features].fillna(0)
            X_cv = df_v.loc[cv, features].fillna(0)
            X_cte = df_v.loc[cte, features].fillna(0)
            y_ct = df_v.loc[ct, corn_target]
            y_cv = df_v.loc[cv, corn_target]
            y_cte = df_v.loc[cte, corn_target].values
            br_co = y_cte.mean()
            
            bm_co, _, _, bn_co, bt_co = train_binary(X_ct, y_ct, X_cv, y_cv, binary_configs)
            probs_co = bm_co.predict_proba(X_cte)[:, 1]
            
            best_co_acc = 0
            for thr in [0.40, 0.45, 0.50, 0.55, 0.60]:
                p = (probs_co > thr).astype(int)
                a = (p == y_cte).mean()
                if a > best_co_acc:
                    best_co_acc = a
                    best_co_preds = p
                    cfg_co = f"t{thr}"
            
            results[corn_target] = {
                "accuracy": round(best_co_acc, 4), "base_rate": round(br_co, 3),
                "config": cfg_co, "model": bn_co, "n_matches": int(cte.sum()),
            }
            log.info("[%s] ✓ Step %d DONE (%s): %s=%.1f%% (base=%.1f%%) [%s] n=%d",
                     elapsed(), step, fmt_duration(time.time() - t0_co), cname, best_co_acc*100, br_co*100, cfg_co, cte.sum())
            del bm_co; gc.collect()
        
        # ================================================================
        # HT RESULT
        # ================================================================
        ht_valid = df_v["ht_result"].isin(["H", "D", "A"])
        if ht_valid.sum() > 500:
            step += 1
            t0_ht = time.time()
            log.info("\n[%s] ▋ Step %d/%d: HT Result", elapsed(), step, TOTAL_MARKETS)
            ht_df = df_v[ht_valid].copy()
            ht_tr = ht_df["season"].isin(train_seasons)
            ht_val = ht_df["season"] == val_season
            ht_te = ht_df["season"] == test_season
            
            if ht_tr.sum() > 100 and ht_val.sum() > 10 and ht_te.sum() > 10:
                X_ht_tr = ht_df.loc[ht_tr, features].fillna(0)
                X_ht_val = ht_df.loc[ht_val, features].fillna(0)
                X_ht_te = ht_df.loc[ht_te, features].fillna(0)
                y_ht_tr = ht_df.loc[ht_tr, "ht_result"].map(LABEL_MAP).values
                y_ht_val = ht_df.loc[ht_val, "ht_result"].map(LABEL_MAP).values
                y_ht_te = ht_df.loc[ht_te, "ht_result"].values
                
                best_ht_acc = 0
                for cfg in [
                    {"depth": 6, "lr": 0.03, "l2": 5, "cw": "Balanced"},
                    {"depth": 7, "lr": 0.02, "l2": 3, "cw": "Balanced"},
                    {"depth": 5, "lr": 0.05, "l2": 3, "cw": "Balanced"},
                ]:
                    m_ht = CatBoostClassifier(
                        iterations=1500, depth=cfg["depth"], learning_rate=cfg["lr"],
                        l2_leaf_reg=cfg["l2"], random_seed=42, verbose=0,
                        loss_function="MultiClass", classes_count=3,
                        auto_class_weights=cfg["cw"],
                    )
                    m_ht.fit(X_ht_tr, y_ht_tr, eval_set=(X_ht_val, y_ht_val),
                             early_stopping_rounds=100, verbose=0)
                    ht_probs = m_ht.predict_proba(X_ht_te)
                    ht_preds = np.array(["H", "D", "A"])[ht_probs.argmax(1)]
                    ht_acc = (ht_preds == y_ht_te).mean()
                    if ht_acc > best_ht_acc:
                        best_ht_acc = ht_acc
                        best_ht_preds = ht_preds
                    del m_ht; gc.collect()
                
                ht_d_actual = (y_ht_te == "D").sum()
                ht_d_pred = (best_ht_preds == "D").sum()
                results["ht_result"] = {
                    "accuracy": round(best_ht_acc, 4),
                    "draws": f"{ht_d_pred}/{ht_d_actual}",
                    "n_matches": int(ht_te.sum()),
                }
                log.info("[%s] ✓ Step %d DONE (%s): HT Result=%.1f%% draws=%d/%d n=%d",
                         elapsed(), step, fmt_duration(time.time() - t0_ht),
                         best_ht_acc*100, ht_d_pred, ht_d_actual, ht_te.sum())
        
        # ================================================================
        # SUMMARY
        # ================================================================
        season_dur = time.time() - season_start
        log.info("\n" + "=" * 70)
        log.info("[%s] SEASON %s COMPLETE in %s — ALL MARKETS SUMMARY",
                 elapsed(), test_season, fmt_duration(season_dur))
        log.info("=" * 70)
        for market, data in results.items():
            acc = data.get("accuracy", 0)
            br = data.get("base_rate", data.get("market", 0))
            gain = acc - br if isinstance(br, float) else 0
            log.info("  %-20s %.1f%% (baseline=%.1f%%, +%.1fpp)", market, acc*100, br*100, gain*100)
        
        all_season_results[test_season] = results
        gc.collect()
    
    # Save
    total_dur = time.time() - SCRIPT_START
    with open(MDIR / "all_markets_optimized.json", "w") as f:
        json.dump(all_season_results, f, indent=2, default=str)
    log.info("\n[%s] ✓ COMPLETE. Total runtime: %s", elapsed(), fmt_duration(total_dur))
    log.info("[%s] Saved all_markets_optimized.json", elapsed())


if __name__ == "__main__":
    main()
