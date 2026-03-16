#!/usr/bin/env python3
"""Deep Betting Engine — Complete prediction + value betting system.

Features:
  1. Exact score matrix from calibrated Poisson model
  2. All derived markets: DC, Asian Handicap, O/U 0.5-4.5, BTTS, HT/FT
  3. Probability calibration (isotonic regression)
  4. Value betting: model prob vs bookmaker implied prob → edge
  5. Kelly criterion staking
  6. Full backtest on 2023-24 + 2024-25 with real odds → ROI per market

Runtime: ~20 min (1X2 ensemble) + ~5 min (side bets + calibration + backtest)
"""
import sys, json, logging, gc, warnings, time
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.isotonic import IsotonicRegression
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

from scripts.models.comprehensive_markets import load_unified_training_data, get_training_features, _compute_targets
from config.settings import MODELS_DIR
from catboost import CatBoostClassifier, CatBoostRegressor

MDIR = MODELS_DIR / "markets"
LABEL_MAP = {"H": 0, "D": 1, "A": 2}
T0 = None

def elapsed():
    m, s = divmod(int(time.time() - T0), 60)
    return f"{m:02d}:{s:02d}"

def fmt(s):
    m, s = divmod(int(s), 60)
    return f"{m}m{s:02d}s"


# =============================================================================
# POISSON SCORE MATRIX — The core of everything
# =============================================================================
def poisson_score_matrix(hxg, axg, max_goals=7):
    """Generate full NxN score probability matrix for each match."""
    n = len(hxg)
    matrices = np.zeros((n, max_goals, max_goals))
    for i in range(n):
        hx = max(hxg[i], 0.25)
        ax = max(axg[i], 0.25)
        for h in range(max_goals):
            for a in range(max_goals):
                matrices[i, h, a] = poisson.pmf(h, hx) * poisson.pmf(a, ax)
    return matrices


def markets_from_matrix(matrices):
    """Derive ALL markets from score probability matrices."""
    n = matrices.shape[0]
    mg = matrices.shape[1]
    
    results = {
        "1x2": np.zeros((n, 3)),      # H, D, A
        "dc": np.zeros((n, 3)),        # 1X, 12, X2
        "ou_0_5": np.zeros((n, 2)),    # Over, Under
        "ou_1_5": np.zeros((n, 2)),
        "ou_2_5": np.zeros((n, 2)),
        "ou_3_5": np.zeros((n, 2)),
        "ou_4_5": np.zeros((n, 2)),
        "btts": np.zeros((n, 2)),      # Yes, No
        "exact_scores": {},            # Top 10 most likely scores per match
        "home_goals_ou_0_5": np.zeros((n, 2)),
        "away_goals_ou_0_5": np.zeros((n, 2)),
        "home_goals_ou_1_5": np.zeros((n, 2)),
        "away_goals_ou_1_5": np.zeros((n, 2)),
        "ah_0": np.zeros((n, 2)),      # AH 0 (draw no bet)
        "ah_minus_1": np.zeros((n, 2)), # AH -1 (home -1)
        "ah_plus_1": np.zeros((n, 2)),  # AH +1 (home +1)
        "expected_home": np.zeros(n),
        "expected_away": np.zeros(n),
        "expected_total": np.zeros(n),
    }
    
    for i in range(n):
        m = matrices[i]
        total_p = m.sum()
        if total_p < 0.001:
            continue
        m = m / total_p  # normalize
        
        for h in range(mg):
            for a in range(mg):
                p = m[h, a]
                tg = h + a
                
                # 1X2
                if h > a: results["1x2"][i, 0] += p
                elif h == a: results["1x2"][i, 1] += p
                else: results["1x2"][i, 2] += p
                
                # O/U goals
                if tg > 0.5: results["ou_0_5"][i, 0] += p
                else: results["ou_0_5"][i, 1] += p
                if tg > 1.5: results["ou_1_5"][i, 0] += p
                else: results["ou_1_5"][i, 1] += p
                if tg > 2.5: results["ou_2_5"][i, 0] += p
                else: results["ou_2_5"][i, 1] += p
                if tg > 3.5: results["ou_3_5"][i, 0] += p
                else: results["ou_3_5"][i, 1] += p
                if tg > 4.5: results["ou_4_5"][i, 0] += p
                else: results["ou_4_5"][i, 1] += p
                
                # BTTS
                if h >= 1 and a >= 1: results["btts"][i, 0] += p
                else: results["btts"][i, 1] += p
                
                # Home/Away goals O/U
                if h > 0.5: results["home_goals_ou_0_5"][i, 0] += p
                else: results["home_goals_ou_0_5"][i, 1] += p
                if h > 1.5: results["home_goals_ou_1_5"][i, 0] += p
                else: results["home_goals_ou_1_5"][i, 1] += p
                if a > 0.5: results["away_goals_ou_0_5"][i, 0] += p
                else: results["away_goals_ou_0_5"][i, 1] += p
                if a > 1.5: results["away_goals_ou_1_5"][i, 0] += p
                else: results["away_goals_ou_1_5"][i, 1] += p
                
                # Asian Handicap
                gd = h - a
                if gd > 0: results["ah_0"][i, 0] += p       # home wins AH 0
                elif gd < 0: results["ah_0"][i, 1] += p     # away wins AH 0
                # AH 0 draw = push (refund), not counted
                
                if gd - 1 > 0: results["ah_minus_1"][i, 0] += p   # home -1 wins
                elif gd - 1 < 0: results["ah_minus_1"][i, 1] += p # home -1 loses
                
                if gd + 1 > 0: results["ah_plus_1"][i, 0] += p    # home +1 wins
                elif gd + 1 < 0: results["ah_plus_1"][i, 1] += p  # home +1 loses
                
                results["expected_home"][i] += h * p
                results["expected_away"][i] += a * p
                results["expected_total"][i] += tg * p
        
        # Double Chance (derived from 1X2)
        results["dc"][i, 0] = results["1x2"][i, 0] + results["1x2"][i, 1]  # 1X
        results["dc"][i, 1] = results["1x2"][i, 0] + results["1x2"][i, 2]  # 12
        results["dc"][i, 2] = results["1x2"][i, 1] + results["1x2"][i, 2]  # X2
        
        # Top exact scores
        flat = m.flatten()
        top_idx = np.argsort(flat)[::-1][:12]
        scores = []
        for idx in top_idx:
            h_sc, a_sc = divmod(idx, mg)
            scores.append({"score": f"{h_sc}-{a_sc}", "prob": round(float(flat[idx]), 4)})
        results["exact_scores"][i] = scores
    
    return results


# =============================================================================
# PROBABILITY CALIBRATION
# =============================================================================
def calibrate_binary(val_probs, val_true):
    """Fit isotonic calibration on validation data."""
    valid = ~np.isnan(val_probs) & ~np.isnan(val_true)
    if valid.sum() < 30:
        return None
    ir = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    ir.fit(val_probs[valid], val_true[valid])
    return ir


# =============================================================================
# VALUE BETTING + KELLY CRITERION
# =============================================================================
def find_value_bets(model_prob, implied_prob, odds, min_edge=0.03, max_kelly=0.10):
    """Find bets where model probability > implied probability + margin.
    
    Returns: list of {edge, kelly_fraction, expected_value, odds, model_prob, implied_prob}
    """
    bets = []
    n = len(model_prob)
    for i in range(n):
        mp = model_prob[i]
        ip = implied_prob[i]
        o = odds[i]
        
        if np.isnan(mp) or np.isnan(ip) or np.isnan(o) or o <= 1:
            continue
        
        edge = mp - ip
        if edge > min_edge:
            # Kelly criterion: f* = (bp - q) / b where b = odds-1, p = model_prob, q = 1-p
            b = o - 1
            kelly = (b * mp - (1 - mp)) / b
            kelly = max(0, min(kelly, max_kelly))  # cap at max_kelly
            
            ev = mp * (o - 1) - (1 - mp)  # expected value per unit
            
            bets.append({
                "idx": i,
                "edge": round(edge, 4),
                "kelly": round(kelly, 4),
                "ev_per_unit": round(ev, 4),
                "odds": round(o, 2),
                "model_prob": round(mp, 4),
                "implied_prob": round(ip, 4),
            })
    
    return bets


def backtest_bets(bets, outcomes, bankroll=1000.0, flat_stake=None):
    """Backtest a set of value bets. Returns profit, ROI, stats."""
    if not bets:
        return {"n_bets": 0, "profit": 0, "roi": 0}
    
    total_staked = 0
    total_profit = 0
    wins = 0
    
    for bet in bets:
        idx = bet["idx"]
        if flat_stake:
            stake = flat_stake
        else:
            stake = bankroll * bet["kelly"]
        
        if stake <= 0:
            continue
        
        total_staked += stake
        won = outcomes[idx]
        if won:
            total_profit += stake * (bet["odds"] - 1)
            wins += 1
        else:
            total_profit -= stake
    
    return {
        "n_bets": len(bets),
        "wins": wins,
        "win_rate": round(wins / max(len(bets), 1), 3),
        "total_staked": round(total_staked, 2),
        "profit": round(total_profit, 2),
        "roi": round(total_profit / max(total_staked, 1) * 100, 2),
        "avg_edge": round(np.mean([b["edge"] for b in bets]), 4),
        "avg_odds": round(np.mean([b["odds"] for b in bets]), 2),
    }


# =============================================================================
# MAIN ENGINE
# =============================================================================
def main():
    global T0
    T0 = time.time()
    
    log.info("[%s] ▶ Loading data...", elapsed())
    df = load_unified_training_data()
    features = get_training_features(df)
    df = _compute_targets(df)
    df_v = df[df["result"].isin(["H", "D", "A"])].copy()
    seasons = sorted(df_v["season"].unique())
    log.info("[%s] ✓ Data: %d matches, %d features, %d seasons", elapsed(), len(df_v), len(features), len(seasons))
    
    all_results = {}
    
    for test_season in ["2023-2024", "2024-2025"]:
        season_t = time.time()
        log.info("\n" + "=" * 80)
        log.info("[%s] ████ TEST SEASON: %s ████", elapsed(), test_season)
        log.info("=" * 80)
        
        ti = seasons.index(test_season)
        val_season = seasons[ti - 1]
        train_seasons = seasons[:ti - 1]
        
        train_m = df_v["season"].isin(train_seasons)
        val_m = df_v["season"] == val_season
        test_m = df_v["season"] == test_season
        
        X_tr = df_v.loc[train_m, features].fillna(0)
        X_val = df_v.loc[val_m, features].fillna(0)
        X_te = df_v.loc[test_m, features].fillna(0)
        n_tr, n_val, n_te = train_m.sum(), val_m.sum(), test_m.sum()
        
        log.info("[%s]   Train=%d (%s→%s), Val=%d (%s), Test=%d (%s)",
                 elapsed(), n_tr, train_seasons[0], train_seasons[-1],
                 n_val, val_season, n_te, test_season)
        
        # ==================================================================
        # STEP 1: Train Poisson regressors → Score matrix
        # ==================================================================
        t1 = time.time()
        log.info("\n[%s] ▋ STEP 1/6: Poisson Goal Models", elapsed())
        
        hg = CatBoostRegressor(iterations=1500, depth=8, learning_rate=0.015, l2_leaf_reg=7,
                                min_data_in_leaf=30, random_seed=42, verbose=0, loss_function="Poisson")
        ag = CatBoostRegressor(iterations=1500, depth=8, learning_rate=0.015, l2_leaf_reg=7,
                                min_data_in_leaf=30, random_seed=42, verbose=0, loss_function="Poisson")
        
        hg.fit(X_tr, df_v.loc[train_m, "home_score"],
               eval_set=(X_val, df_v.loc[val_m, "home_score"]), early_stopping_rounds=100, verbose=0)
        log.info("[%s]   ✓ Home goals model trained", elapsed())
        
        ag.fit(X_tr, df_v.loc[train_m, "away_score"],
               eval_set=(X_val, df_v.loc[val_m, "away_score"]), early_stopping_rounds=100, verbose=0)
        log.info("[%s]   ✓ Away goals model trained", elapsed())
        
        # Predict on val and test
        hxg_val = np.clip(hg.predict(X_val), 0.25, 4.5)
        axg_val = np.clip(ag.predict(X_val), 0.25, 4.5)
        hxg_te = np.clip(hg.predict(X_te), 0.25, 4.5)
        axg_te = np.clip(ag.predict(X_te), 0.25, 4.5)
        
        # Generate score matrices
        mat_val = poisson_score_matrix(hxg_val, axg_val)
        mat_te = poisson_score_matrix(hxg_te, axg_te)
        
        # Derive all markets from matrices
        mkts_val = markets_from_matrix(mat_val)
        mkts_te = markets_from_matrix(mat_te)
        
        log.info("[%s]   ✓ Score matrices generated: %dx7x7", elapsed(), n_te)
        log.info("[%s]   Expected goals test: H=%.2f A=%.2f T=%.2f",
                 elapsed(), mkts_te["expected_home"].mean(), mkts_te["expected_away"].mean(),
                 mkts_te["expected_total"].mean())
        log.info("[%s]   Actual goals test: H=%.2f A=%.2f T=%.2f",
                 elapsed(), df_v.loc[test_m, "home_score"].mean(),
                 df_v.loc[test_m, "away_score"].mean(),
                 df_v.loc[test_m, "total_goals"].mean())
        log.info("[%s]   Step 1 done (%s)", elapsed(), fmt(time.time() - t1))
        del hg, ag; gc.collect()
        
        # ==================================================================
        # STEP 2: CatBoost 1X2 + Ensemble
        # ==================================================================
        t2 = time.time()
        log.info("\n[%s] ▋ STEP 2/6: 1X2 Ensemble (CatBoost + Poisson + Market)", elapsed())
        
        y_tr_num = df_v.loc[train_m, "result"].map(LABEL_MAP).values
        y_val_num = df_v.loc[val_m, "result"].map(LABEL_MAP).values
        
        cb = CatBoostClassifier(
            iterations=2000, depth=8, learning_rate=0.01, l2_leaf_reg=5,
            min_data_in_leaf=30, random_seed=42, verbose=0,
            loss_function="MultiClass", classes_count=3, auto_class_weights="Balanced",
        )
        cb.fit(X_tr, y_tr_num, eval_set=(X_val, y_val_num), early_stopping_rounds=150, verbose=0)
        cb_probs_val = cb.predict_proba(X_val)
        cb_probs_te = cb.predict_proba(X_te)
        log.info("[%s]   ✓ CatBoost 1X2 trained", elapsed())
        del cb; gc.collect()
        
        # Market probs
        def get_mkt(mask):
            cols = ["odds_PSH", "odds_PSD", "odds_PSA"]
            p = np.zeros((mask.sum(), 3))
            for i, (_, row) in enumerate(df_v.loc[mask, cols].iterrows()):
                h, d, a = row.values
                if h > 1 and d > 1 and a > 1:
                    raw = np.array([1/h, 1/d, 1/a])
                    p[i] = raw / raw.sum()
                else:
                    p[i] = [0.4, 0.3, 0.3]
            return p
        
        mkt_val = get_mkt(val_m)
        mkt_te = get_mkt(test_m)
        
        # Poisson 1X2 from matrix
        pois_val = mkts_val["1x2"]
        pois_te = mkts_te["1x2"]
        
        # Ensemble: CB=0.25, Poisson=0.40, Market=0.35, draw inflation=1.35
        def blend_1x2(cb_p, pois_p, mkt_p):
            ens = 0.25 * cb_p + 0.40 * pois_p + 0.35 * mkt_p
            ens[:, 1] *= 1.35
            return ens / ens.sum(axis=1, keepdims=True)
        
        ens_val = blend_1x2(cb_probs_val, pois_val, mkt_val)
        ens_te = blend_1x2(cb_probs_te, pois_te, mkt_te)
        
        y_te = df_v.loc[test_m, "result"].values
        y_val = df_v.loc[val_m, "result"].values
        preds_1x2 = np.array(["H", "D", "A"])[ens_te.argmax(1)]
        acc_1x2 = (preds_1x2 == y_te).mean()
        
        max_p = ens_te.max(axis=1)
        hc60_mask = max_p > 0.60
        hc60_acc = (preds_1x2[hc60_mask] == y_te[hc60_mask]).mean() if hc60_mask.sum() > 0 else 0
        
        log.info("[%s]   ✓ 1X2: %.1f%% accuracy | HC60=%.0f%% (%d matches)",
                 elapsed(), acc_1x2*100, hc60_acc*100, hc60_mask.sum())
        log.info("[%s]   Step 2 done (%s)", elapsed(), fmt(time.time() - t2))
        
        # ==================================================================
        # STEP 3: Probability Calibration
        # ==================================================================
        t3 = time.time()
        log.info("\n[%s] ▋ STEP 3/6: Probability Calibration (isotonic regression)", elapsed())
        
        calibrators = {}
        
        # Calibrate Poisson O/U 2.5
        val_ou25_true = df_v.loc[val_m, "over_2_5"].values.astype(float)
        cal_ou25 = calibrate_binary(mkts_val["ou_2_5"][:, 0], val_ou25_true)
        if cal_ou25:
            calibrators["ou_2_5"] = cal_ou25
            raw_p = mkts_te["ou_2_5"][:, 0]
            cal_p = cal_ou25.predict(raw_p)
            actual = df_v.loc[test_m, "over_2_5"].values
            raw_acc = ((raw_p > 0.5).astype(int) == actual).mean()
            cal_acc = ((cal_p > 0.5).astype(int) == actual).mean()
            log.info("[%s]   O/U 2.5: raw=%.1f%% → calibrated=%.1f%%", elapsed(), raw_acc*100, cal_acc*100)
        
        # Calibrate BTTS
        val_btts_true = df_v.loc[val_m, "btts"].values.astype(float)
        cal_btts = calibrate_binary(mkts_val["btts"][:, 0], val_btts_true)
        if cal_btts:
            calibrators["btts"] = cal_btts
            raw_p = mkts_te["btts"][:, 0]
            cal_p = cal_btts.predict(raw_p)
            actual = df_v.loc[test_m, "btts"].values
            raw_acc = ((raw_p > 0.5).astype(int) == actual).mean()
            cal_acc = ((cal_p > 0.5).astype(int) == actual).mean()
            log.info("[%s]   BTTS: raw=%.1f%% → calibrated=%.1f%%", elapsed(), raw_acc*100, cal_acc*100)
        
        # Calibrate O/U 1.5, 3.5
        for ou_name, target in [("ou_1_5", "over_1_5"), ("ou_3_5", "over_3_5")]:
            val_true = df_v.loc[val_m, target].values.astype(float)
            cal = calibrate_binary(mkts_val[ou_name][:, 0], val_true)
            if cal:
                calibrators[ou_name] = cal
                raw_p = mkts_te[ou_name][:, 0]
                cal_p = cal.predict(raw_p)
                actual = df_v.loc[test_m, target].values
                raw_acc = ((raw_p > 0.5).astype(int) == actual).mean()
                cal_acc = ((cal_p > 0.5).astype(int) == actual).mean()
                log.info("[%s]   %s: raw=%.1f%% → calibrated=%.1f%%", elapsed(), ou_name, raw_acc*100, cal_acc*100)
        
        log.info("[%s]   Step 3 done (%s)", elapsed(), fmt(time.time() - t3))
        
        # ==================================================================
        # STEP 4: All Markets Accuracy
        # ==================================================================
        t4 = time.time()
        log.info("\n[%s] ▋ STEP 4/6: All Markets Accuracy", elapsed())
        
        market_results = {}
        
        # 1X2
        market_results["1x2"] = {"acc": round(acc_1x2, 4), "hc60_acc": round(hc60_acc, 4),
                                  "hc60_n": int(hc60_mask.sum())}
        
        # O/U markets
        for ou_name, target in [
            ("ou_0_5", "over_0_5"), ("ou_1_5", "over_1_5"), ("ou_2_5", "over_2_5"),
            ("ou_3_5", "over_3_5"), ("ou_4_5", "over_4_5"),
        ]:
            if target in df_v.columns:
                raw_p = mkts_te[ou_name][:, 0]
                if ou_name in calibrators:
                    cal_p = calibrators[ou_name].predict(raw_p)
                else:
                    cal_p = raw_p
                actual = df_v.loc[test_m, target].values
                acc = ((cal_p > 0.5).astype(int) == actual).mean()
                base = actual.mean()
                market_results[ou_name] = {"acc": round(acc, 4), "base": round(base, 3)}
                log.info("[%s]   %s: %.1f%% (base=%.1f%%)", elapsed(), ou_name, acc*100, base*100)
        
        # BTTS
        raw_p = mkts_te["btts"][:, 0]
        if "btts" in calibrators:
            cal_p = calibrators["btts"].predict(raw_p)
        else:
            cal_p = raw_p
        actual_btts = df_v.loc[test_m, "btts"].values
        acc_btts = ((cal_p > 0.5).astype(int) == actual_btts).mean()
        market_results["btts"] = {"acc": round(acc_btts, 4), "base": round(actual_btts.mean(), 3)}
        log.info("[%s]   BTTS: %.1f%% (base=%.1f%%)", elapsed(), acc_btts*100, actual_btts.mean()*100)
        
        # Double Chance
        dc_te = mkts_te["dc"]
        for dc_idx, dc_name in [(0, "1X"), (1, "12"), (2, "X2")]:
            preds = (dc_te[:, dc_idx] > 0.5).astype(int)
            if dc_name == "1X":
                actual = ((y_te == "H") | (y_te == "D")).astype(int)
            elif dc_name == "12":
                actual = ((y_te == "H") | (y_te == "A")).astype(int)
            else:
                actual = ((y_te == "D") | (y_te == "A")).astype(int)
            acc = (preds == actual).mean()
            market_results[f"dc_{dc_name}"] = {"acc": round(acc, 4)}
            log.info("[%s]   DC %s: %.1f%%", elapsed(), dc_name, acc*100)
        
        # Exact score (top-1 accuracy)
        top1_correct = 0
        top3_correct = 0
        hs_actual = df_v.loc[test_m, "home_score"].values.astype(int)
        as_actual = df_v.loc[test_m, "away_score"].values.astype(int)
        for i in range(n_te):
            actual_score = f"{hs_actual[i]}-{as_actual[i]}"
            scores = mkts_te["exact_scores"][i]
            if scores[0]["score"] == actual_score:
                top1_correct += 1
            if any(s["score"] == actual_score for s in scores[:3]):
                top3_correct += 1
        
        market_results["exact_score_top1"] = {"acc": round(top1_correct/n_te, 4)}
        market_results["exact_score_top3"] = {"acc": round(top3_correct/n_te, 4)}
        log.info("[%s]   Exact Score: top1=%.1f%% top3=%.1f%%",
                 elapsed(), top1_correct/n_te*100, top3_correct/n_te*100)
        
        # Asian Handicap 0 (DNB)
        gd_actual = hs_actual - as_actual
        ah0_preds = (mkts_te["ah_0"][:, 0] > mkts_te["ah_0"][:, 1]).astype(int)  # 1=home, 0=away
        ah0_actual = (gd_actual > 0).astype(int)
        non_draw = gd_actual != 0
        ah0_acc = (ah0_preds[non_draw] == ah0_actual[non_draw]).mean() if non_draw.sum() > 0 else 0
        market_results["ah_0_dnb"] = {"acc": round(ah0_acc, 4), "n_settled": int(non_draw.sum())}
        log.info("[%s]   AH 0 (DNB): %.1f%% (%d settled)", elapsed(), ah0_acc*100, non_draw.sum())
        
        # AH -1 (home -1)
        ah1_preds = (mkts_te["ah_minus_1"][:, 0] > mkts_te["ah_minus_1"][:, 1]).astype(int)
        ah1_actual = ((gd_actual - 1) > 0).astype(int)
        non_push = (gd_actual != 1)
        ah1_acc = (ah1_preds[non_push] == ah1_actual[non_push]).mean() if non_push.sum() > 0 else 0
        market_results["ah_minus_1"] = {"acc": round(ah1_acc, 4)}
        log.info("[%s]   AH -1: %.1f%%", elapsed(), ah1_acc*100)
        
        log.info("[%s]   Step 4 done (%s)", elapsed(), fmt(time.time() - t4))
        
        # ==================================================================
        # STEP 5: Value Betting with Real Odds
        # ==================================================================
        t5 = time.time()
        log.info("\n[%s] ▋ STEP 5/6: Value Betting (model prob vs bookmaker odds)", elapsed())
        
        betting_results = {}
        
        # 1X2 Value Bets (Pinnacle odds)
        for cls_idx, cls_name, odds_col in [(0, "Home", "odds_PSH"), (1, "Draw", "odds_PSD"), (2, "Away", "odds_PSA")]:
            model_p = ens_te[:, cls_idx]
            odds_raw = pd.to_numeric(df_v.loc[test_m, odds_col], errors="coerce").values
            implied_p = np.where(odds_raw > 1, 1/odds_raw, np.nan)
            
            # Outcomes: did this class win?
            outcomes = (y_te == ["H", "D", "A"][cls_idx]).astype(int)
            
            for min_edge in [0.03, 0.05, 0.08]:
                bets = find_value_bets(model_p, implied_p, odds_raw, min_edge=min_edge)
                bt = backtest_bets(bets, outcomes, flat_stake=10)
                
                if bt["n_bets"] > 0:
                    key = f"1x2_{cls_name}_e{int(min_edge*100)}"
                    betting_results[key] = bt
                    emoji = "💰" if bt["roi"] > 0 else "📉"
                    log.info("[%s]   %s %s: %d bets, win=%.0f%%, ROI=%+.1f%%, P/L=%+.0f (edge≥%d%%)",
                             elapsed(), emoji, cls_name, bt["n_bets"], bt["win_rate"]*100,
                             bt["roi"], bt["profit"], int(min_edge*100))
        
        # O/U 2.5 Value Bets (Pinnacle)
        for side, side_name, odds_col in [(0, "Over2.5", "odds_P>2.5"), (1, "Under2.5", "odds_P<2.5")]:
            if odds_col.replace("odds_", "") not in [c.replace("odds_", "") for c in df_v.columns if c.startswith("odds_")]:
                continue
            
            col_match = [c for c in df_v.columns if c == odds_col]
            if not col_match:
                continue
            
            raw_p = mkts_te["ou_2_5"][:, side]
            if "ou_2_5" in calibrators and side == 0:
                cal_p = calibrators["ou_2_5"].predict(raw_p)
            else:
                cal_p = raw_p
            
            odds_raw = pd.to_numeric(df_v.loc[test_m, odds_col], errors="coerce").values
            implied_p = np.where(odds_raw > 1, 1/odds_raw, np.nan)
            
            actual_ou = df_v.loc[test_m, "over_2_5"].values
            outcomes = (actual_ou == (1 if side == 0 else 0)).astype(int)
            
            for min_edge in [0.03, 0.05]:
                bets = find_value_bets(cal_p, implied_p, odds_raw, min_edge=min_edge)
                bt = backtest_bets(bets, outcomes, flat_stake=10)
                if bt["n_bets"] > 0:
                    key = f"{side_name}_e{int(min_edge*100)}"
                    betting_results[key] = bt
                    emoji = "💰" if bt["roi"] > 0 else "📉"
                    log.info("[%s]   %s %s: %d bets, win=%.0f%%, ROI=%+.1f%%, P/L=%+.0f",
                             elapsed(), emoji, side_name, bt["n_bets"], bt["win_rate"]*100,
                             bt["roi"], bt["profit"])
        
        log.info("[%s]   Step 5 done (%s)", elapsed(), fmt(time.time() - t5))
        
        # ==================================================================
        # STEP 6: Final Summary + Top Exact Scores
        # ==================================================================
        t6 = time.time()
        log.info("\n[%s] ▋ STEP 6/6: Final Summary", elapsed())
        
        # Show sample exact score predictions
        log.info("\n[%s] 📊 Sample Exact Score Predictions (first 10 matches):", elapsed())
        home_teams = df_v.loc[test_m, "home_team"].values
        away_teams = df_v.loc[test_m, "away_team"].values
        for i in range(min(10, n_te)):
            actual = f"{hs_actual[i]}-{as_actual[i]}"
            top3 = mkts_te["exact_scores"][i][:3]
            top3_str = " | ".join([f"{s['score']}({s['prob']:.0%})" for s in top3])
            match = (top3[0]["score"] == actual)
            log.info("  %s %s vs %s: actual=%s pred=[%s] %s",
                     "✓" if match else "✗", home_teams[i], away_teams[i], actual, top3_str,
                     "✓HIT" if match else "")
        
        # Comprehensive results
        season_dur = time.time() - season_t
        log.info("\n" + "=" * 80)
        log.info("[%s] ████ %s COMPLETE (%s) ████", elapsed(), test_season, fmt(season_dur))
        log.info("=" * 80)
        
        log.info("\n📈 PREDICTION ACCURACY:")
        for mkt, data in sorted(market_results.items()):
            acc = data.get("acc", 0)
            extra = ""
            if "base" in data: extra += f" (base={data['base']:.1%})"
            if "hc60_acc" in data: extra += f" HC60={data['hc60_acc']:.0%}({data['hc60_n']})"
            log.info("  %-20s %5.1f%%%s", mkt, acc*100, extra)
        
        if betting_results:
            log.info("\n💰 VALUE BETTING BACKTEST (flat €10 stakes):")
            total_profit = 0
            total_staked = 0
            total_bets = 0
            for key, bt in sorted(betting_results.items()):
                emoji = "💰" if bt["roi"] > 0 else "📉"
                log.info("  %s %-25s %3d bets | win=%4.0f%% | ROI=%+6.1f%% | P/L=%+7.1f€ | avg_edge=%.1f%%",
                         emoji, key, bt["n_bets"], bt["win_rate"]*100, bt["roi"],
                         bt["profit"], bt["avg_edge"]*100)
                total_profit += bt["profit"]
                total_staked += bt["total_staked"]
                total_bets += bt["n_bets"]
            
            log.info("  ─────────────────────────────────────────────────────")
            log.info("  TOTAL: %d bets | staked=€%.0f | P/L=%+.1f€ | ROI=%+.1f%%",
                     total_bets, total_staked, total_profit,
                     total_profit/max(total_staked, 1)*100)
        
        all_results[test_season] = {
            "markets": market_results,
            "betting": betting_results,
        }
        gc.collect()
    
    # Save everything
    total = time.time() - T0
    with open(MDIR / "deep_betting_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    log.info("\n" + "=" * 80)
    log.info("[%s] ✓ ALL DONE. Total: %s", elapsed(), fmt(total))
    log.info("Saved deep_betting_results.json")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
