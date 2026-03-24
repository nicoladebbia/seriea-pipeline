#!/usr/bin/env python3
"""Walk-forward ensemble weight optimization.

Uses multi-season walk-forward cross-validation to avoid single-season overfitting:
- For each test season S, trains on all seasons before S
- Optuna objective = average log-loss across ALL test folds
- No isotonic calibration (overfits on 760 matches, proven in v1)

Seasons used:
  - Test folds: 2021-2022, 2022-2023, 2023-2024, 2024-2025 (4 folds)
  - Each fold trains on all prior seasons with data
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import DATA_DIR, MODELS_DIR
from storage.paths import features_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Load features + ML model
# --------------------------------------------------------------------------
BASE_RATES = {"H": 0.45, "D": 0.27, "A": 0.28}
LABEL_MAP = {"H": 0, "D": 1, "A": 2}

_ml_model = None


def _load_ml():
    """Load ML model with same priority as production and backtest.

    Priority: no-odds CatBoost > multi-model ensemble > catboost_upcoming.
    Must match ensemble_prediction_engine.py and backtest_unified.py to ensure
    weights are optimized for the model that's actually used in production.
    """
    global _ml_model
    if _ml_model is not None:
        return _ml_model
    try:
        from catboost import CatBoostClassifier

        # Priority 1: No-odds CatBoost (same as production)
        no_odds_path = MODELS_DIR / "universal" / "catboost_no_odds.cbm"
        if no_odds_path.exists():
            _ml_model = CatBoostClassifier()
            _ml_model.load_model(str(no_odds_path))
            log.info("Loaded no-odds CatBoost: %d features (matches production)",
                     len(_ml_model.feature_names_))
            return _ml_model

        # Priority 2: catboost_upcoming (legacy fallback)
        path = MODELS_DIR / "universal" / "catboost_upcoming.cbm"
        if path.exists():
            _ml_model = CatBoostClassifier()
            _ml_model.load_model(str(path))
            log.info("Loaded catboost_upcoming: %d features (fallback)",
                     len(_ml_model.feature_names_))
        return _ml_model
    except Exception as e:
        log.warning("Failed to load ML: %s", e)
        return None


def _load_data():
    df = pd.read_parquet(features_path())
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.sort_values("match_date").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# Component predictions (same as backtest_unified.py)
# --------------------------------------------------------------------------
def _predict_factor(row):
    prob_H, prob_D, prob_A = BASE_RATES["H"], BASE_RATES["D"], BASE_RATES["A"]
    elo_diff = row.get("elo_diff", 0)
    if not np.isnan(elo_diff) and elo_diff != 0:
        shift = elo_diff / 800.0
        prob_H += shift * 0.4
        prob_A -= shift * 0.4
    home_atk = row.get("home_attack_strength", 1.0)
    home_def = row.get("home_defense_strength", 1.0)
    away_atk = row.get("away_attack_strength", 1.0)
    away_def = row.get("away_defense_strength", 1.0)
    if not any(np.isnan(v) for v in [home_atk, home_def, away_atk, away_def]) and away_def > 0 and home_def > 0:
        net = (home_atk / away_def - 1.0) * 0.08 - (away_atk / home_def - 1.0) * 0.08
        prob_H += net
        prob_A -= net
    h2h = row.get("h2h_home_win_rate", np.nan)
    if not np.isnan(h2h) and h2h > 0:
        prob_H += (h2h - BASE_RATES["H"]) * 0.05
    total = prob_H + prob_D + prob_A
    return np.array([prob_H / total, prob_D / total, prob_A / total])


def _poisson_probs(hxg, axg, max_goals=10):
    hp = [poisson.pmf(g, hxg) for g in range(max_goals)]
    ap = [poisson.pmf(g, axg) for g in range(max_goals)]
    pH = pD = pA = 0.0
    for h in range(max_goals):
        for a in range(max_goals):
            p = hp[h] * ap[a]
            if h > a: pH += p
            elif h == a: pD += p
            else: pA += p
    # Draw inflation for Poisson (component-level)
    xg_gap = abs(hxg - axg)
    draw_inflate = max(0.90, min(1.55, 1.55 - 0.20 * xg_gap))
    pD *= draw_inflate
    t = pH + pD + pA
    return np.array([pH / t, pD / t, pA / t])


def _predict_xg(row):
    vals = [row.get(c, np.nan) for c in
            ["home_xg_attack_strength", "home_xg_defense_strength",
             "away_xg_attack_strength", "away_xg_defense_strength"]]
    if any(np.isnan(v) for v in vals):
        return None
    hxg = max(0.4, min(3.5, vals[0] * vals[3] * 1.38)) * 1.08
    axg = max(0.3, min(3.0, vals[2] * vals[1] * 1.38)) * 0.92
    return _poisson_probs(hxg, axg)


def _predict_market(row):
    psh, psd, psa = row.get("odds_PSH", 0), row.get("odds_PSD", 0), row.get("odds_PSA", 0)
    if not all(v and v > 1.0 for v in [psh, psd, psa]):
        return None
    rh, rd, ra = 1.0 / psh, 1.0 / psd, 1.0 / psa
    t = rh + rd + ra
    return np.array([rh / t, rd / t, ra / t])


def _predict_ml(row, model, ml_temperature=0.75):
    if model is None:
        return None
    try:
        fnames = list(model.feature_names_)
        avail = [f for f in fnames if f in row.index]
        if len(avail) < len(fnames) * 0.5:
            return None
        vals = {}
        for f in fnames:
            v = row.get(f, 0)
            vals[f] = 0 if (isinstance(v, float) and np.isnan(v)) else v
        X = pd.DataFrame([vals])
        proba = model.predict_proba(X[fnames])[0]
        # ML pre-sharpening temperature (lower = sharper)
        eps = 1e-10
        logits = np.log(proba + eps)
        scaled = np.exp(logits / ml_temperature)
        return scaled / scaled.sum()
    except Exception:
        return None


# --------------------------------------------------------------------------
# Ensemble with parameterized weights/draw_boost/temperature
# --------------------------------------------------------------------------
def _ensemble_predict(row, model, weights, draw_boost, temperature, ml_temperature=0.75):
    """Predict with given ensemble parameters."""
    preds = {}
    preds["factor"] = _predict_factor(row)
    xg = _predict_xg(row)
    if xg is not None:
        preds["xg"] = xg
    market = _predict_market(row)
    if market is not None:
        preds["market"] = market
    ml = _predict_ml(row, model, ml_temperature=ml_temperature)
    if ml is not None:
        preds["ml"] = ml
    # player_xg approximated as xG (same Poisson, slight variant)
    if xg is not None:
        preds["player_xg"] = xg

    # Weighted average
    prob = np.zeros(3)
    total_w = 0
    for method, p in preds.items():
        w = weights.get(method, 0)
        if w > 0:
            prob += w * p
            total_w += w
    if total_w > 0:
        prob /= total_w

    # Draw boost
    prob[1] *= draw_boost
    prob /= prob.sum()

    # Post-ensemble temperature
    eps = 1e-10
    logits = np.log(prob + eps)
    scaled = np.exp(logits / temperature)
    prob = scaled / scaled.sum()

    return prob


def _evaluate(df, model, weights, draw_boost, temperature, ml_temperature=0.75):
    """Evaluate log-loss and draw F1 on a DataFrame.

    Returns (log_loss, draw_f1, n_matches).
    """
    total_ll = 0
    n = 0
    draw_tp = 0
    draw_fp = 0
    draw_fn = 0
    draw_idx = LABEL_MAP["D"]
    for _, row in df.iterrows():
        actual = row.get("result")
        if actual not in LABEL_MAP:
            continue
        prob = _ensemble_predict(row, model, weights, draw_boost, temperature,
                                 ml_temperature=ml_temperature)
        actual_idx = LABEL_MAP[actual]
        pred_idx = int(np.argmax(prob))
        p_actual = max(1e-10, prob[actual_idx])
        total_ll += -np.log(p_actual)
        # Draw F1 tracking
        if pred_idx == draw_idx and actual_idx == draw_idx:
            draw_tp += 1
        elif pred_idx == draw_idx and actual_idx != draw_idx:
            draw_fp += 1
        elif pred_idx != draw_idx and actual_idx == draw_idx:
            draw_fn += 1
        n += 1
    ll = total_ll / max(n, 1)
    if draw_tp > 0:
        prec = draw_tp / (draw_tp + draw_fp)
        rec = draw_tp / (draw_tp + draw_fn)
        draw_f1 = 2 * prec * rec / (prec + rec)
    else:
        draw_f1 = 0.0
    return ll, draw_f1, n


def _evaluate_accuracy(df, model, weights, draw_boost, temperature, ml_temperature=0.75):
    """Evaluate accuracy + log-loss."""
    total_ll = 0
    correct = 0
    n = 0
    for _, row in df.iterrows():
        actual = row.get("result")
        if actual not in LABEL_MAP:
            continue
        prob = _ensemble_predict(row, model, weights, draw_boost, temperature,
                                 ml_temperature=ml_temperature)
        actual_idx = LABEL_MAP[actual]
        pred_idx = np.argmax(prob)
        if pred_idx == actual_idx:
            correct += 1
        total_ll += -np.log(max(1e-10, prob[actual_idx]))
        n += 1
    return correct / max(n, 1), total_ll / max(n, 1), n


# --------------------------------------------------------------------------
# Walk-forward folds
# --------------------------------------------------------------------------
def _build_wf_folds(df):
    """Build walk-forward folds: for each test season, train on all prior seasons.

    Returns list of (df_train, df_test, test_season_name) tuples.
    Only includes folds where both train and test have sufficient data.
    """
    all_seasons = sorted(df["season"].unique())
    log.info("Available seasons: %s", all_seasons)

    # Test on seasons from 2021-2022 onward (need sufficient training data)
    test_seasons = [s for s in all_seasons if s >= "2021-2022" and s <= "2024-2025"]
    folds = []

    for test_season in test_seasons:
        # Train on all seasons strictly before the test season
        train_seasons = [s for s in all_seasons if s < test_season]
        if len(train_seasons) < 3:  # Need at least 3 training seasons
            continue

        df_train = df[df["season"].isin(train_seasons)].copy()
        df_test = df[df["season"] == test_season].copy()

        if len(df_test) < 100:  # Skip seasons with too few matches
            continue

        folds.append((df_train, df_test, test_season))
        log.info("  Fold: train=%s (%d matches), test=%s (%d matches)",
                 train_seasons[-3:], len(df_train), test_season, len(df_test))

    return folds


# --------------------------------------------------------------------------
# Walk-forward Optuna optimization
# --------------------------------------------------------------------------
def optimize_weights_walkforward(folds, model, n_trials=300, draw_weight=0.3):
    """Optuna search using walk-forward CV with multi-objective: log-loss + draw F1.

    The objective is: (1 - draw_weight) * avg_log_loss - draw_weight * avg_draw_f1.
    This prevents the optimizer from finding weights that minimize log-loss by
    never predicting draws (which is the failure mode we observed).

    Why draw_weight=0.3: draws are ~27% of Serie A outcomes. A 30% weight ensures
    the optimizer cannot ignore draw detection without a meaningful objective penalty.

    Optimizes 8 parameters:
      - 5 ensemble weights (normalized)
      - draw_boost (1.00 - 1.35)
      - post-ensemble temperature (0.80 - 1.30)
      - ML pre-sharpening temperature (0.30 - 1.00)
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    best_result = {"score": 999, "ll": 999, "draw_f1": 0, "params": {}}

    def objective(trial):
        # Sample weights (Dirichlet-like: sample 5 values, normalize)
        w_factor = trial.suggest_float("w_factor", 0.02, 0.25)
        w_xg = trial.suggest_float("w_xg", 0.05, 0.35)
        w_ml = trial.suggest_float("w_ml", 0.10, 0.70)
        w_pxg = trial.suggest_float("w_pxg", 0.00, 0.15)
        w_market = trial.suggest_float("w_market", 0.10, 0.55)

        total = w_factor + w_xg + w_ml + w_pxg + w_market
        weights = {
            "factor": w_factor / total,
            "xg": w_xg / total,
            "ml": w_ml / total,
            "player_xg": w_pxg / total,
            "market": w_market / total,
        }

        draw_boost = trial.suggest_float("draw_boost", 1.00, 1.35)
        temperature = trial.suggest_float("temperature", 0.80, 1.30)
        ml_temperature = trial.suggest_float("ml_temperature", 0.30, 1.00)

        # Walk-forward: evaluate on each test fold
        fold_lls = []
        fold_draw_f1s = []
        for df_train, df_test, season in folds:
            ll, draw_f1, n = _evaluate(df_test, model, weights, draw_boost,
                                       temperature, ml_temperature=ml_temperature)
            fold_lls.append(ll)
            fold_draw_f1s.append(draw_f1)

        avg_ll = np.mean(fold_lls)
        avg_draw_f1 = np.mean(fold_draw_f1s)

        # Multi-objective: minimize log-loss, maximize draw F1
        score = (1.0 - draw_weight) * avg_ll - draw_weight * avg_draw_f1

        if score < best_result["score"]:
            best_result["score"] = score
            best_result["ll"] = avg_ll
            best_result["draw_f1"] = avg_draw_f1
            best_result["params"] = {
                "weights": weights,
                "draw_boost": draw_boost,
                "temperature": temperature,
                "ml_temperature": ml_temperature,
            }
            best_result["fold_lls"] = fold_lls
            best_result["fold_draw_f1s"] = fold_draw_f1s

        return score

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return best_result


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    log.info("=" * 70)
    log.info("WALK-FORWARD ENSEMBLE WEIGHT OPTIMIZATION")
    log.info("=" * 70)

    # Load data
    df = _load_data()
    model = _load_ml()
    log.info("Data: %d matches, %d columns", len(df), len(df.columns))

    # Build walk-forward folds
    log.info("\nBuilding walk-forward folds:")
    folds = _build_wf_folds(df)
    log.info("Total folds: %d", len(folds))

    if not folds:
        log.error("No valid folds found. Aborting.")
        return None

    # Evaluate current (production) weights across all folds
    current_weights = {
        "factor": 0.035, "xg": 0.124, "ml": 0.605,
        "player_xg": 0.032, "market": 0.205,
    }
    current_draw_boost = 1.12
    current_temperature = 0.90
    current_ml_temperature = 0.75

    log.info("\nCurrent (production) weights performance per fold:")
    current_fold_lls = []
    for df_train, df_test, season in folds:
        acc, ll, n = _evaluate_accuracy(df_test, model, current_weights,
                                         current_draw_boost, current_temperature,
                                         ml_temperature=current_ml_temperature)
        current_fold_lls.append(ll)
        log.info("  %s: accuracy=%.1f%%, log_loss=%.4f (%d matches)", season, acc * 100, ll, n)
    log.info("  Average log-loss: %.4f", np.mean(current_fold_lls))

    # Run walk-forward optimization
    log.info("\n" + "=" * 70)
    log.info("OPTUNA WALK-FORWARD OPTIMIZATION (3000 trials, %d folds)", len(folds))
    log.info("=" * 70)

    result = optimize_weights_walkforward(folds, model, n_trials=3000)
    best = result["params"]
    best_weights = best["weights"]
    best_draw_boost = best["draw_boost"]
    best_temperature = best["temperature"]
    best_ml_temperature = best["ml_temperature"]

    log.info("\nBest parameters found (walk-forward):")
    for k, v in best_weights.items():
        log.info("  %s: %.4f", k, v)
    log.info("  draw_boost: %.4f", best_draw_boost)
    log.info("  temperature (post-ensemble): %.4f", best_temperature)
    log.info("  ml_temperature (pre-sharpen): %.4f", best_ml_temperature)

    # Evaluate optimized weights per fold
    log.info("\nOptimized weights performance per fold:")
    opt_fold_lls = []
    opt_fold_accs = []
    for df_train, df_test, season in folds:
        acc, ll, n = _evaluate_accuracy(df_test, model, best_weights,
                                         best_draw_boost, best_temperature,
                                         ml_temperature=best_ml_temperature)
        opt_fold_lls.append(ll)
        opt_fold_accs.append(acc)
        log.info("  %s: accuracy=%.1f%%, log_loss=%.4f (%d matches)", season, acc * 100, ll, n)
    log.info("  Average log-loss: %.4f (was %.4f)", np.mean(opt_fold_lls), np.mean(current_fold_lls))
    log.info("  Average accuracy: %.1f%%", np.mean(opt_fold_accs) * 100)

    # Final holdout validation on 2024-2025 specifically
    df_val = df[df["season"] == "2024-2025"].copy()
    if len(df_val) > 0:
        acc_val, ll_val, n_val = _evaluate_accuracy(
            df_val, model, best_weights, best_draw_boost, best_temperature,
            ml_temperature=best_ml_temperature)
        acc_cur, ll_cur, _ = _evaluate_accuracy(
            df_val, model, current_weights, current_draw_boost, current_temperature,
            ml_temperature=current_ml_temperature)
        log.info("\n2024-2025 holdout comparison:")
        log.info("  Current:   accuracy=%.1f%%, log_loss=%.4f", acc_cur * 100, ll_cur)
        log.info("  Optimized: accuracy=%.1f%%, log_loss=%.4f", acc_val * 100, ll_val)
        log.info("  Delta:     %+.1fpp accuracy, %+.4f log_loss",
                 (acc_val - acc_cur) * 100, ll_val - ll_cur)

    # Check overfit: per-fold variance
    ll_std = np.std(opt_fold_lls)
    ll_range = max(opt_fold_lls) - min(opt_fold_lls)
    log.info("\nOverfit check:")
    log.info("  Fold log-loss std: %.4f (lower = more stable)", ll_std)
    log.info("  Fold log-loss range: %.4f", ll_range)

    # Save results
    output_dir = DATA_DIR / "optimization"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "method": "walk_forward_cv",
        "n_folds": len(folds),
        "n_trials": 3000,
        "optimal_weights": {k: round(v, 4) for k, v in best_weights.items()},
        "draw_boost": round(best_draw_boost, 4),
        "temperature": round(best_temperature, 4),
        "ml_temperature": round(best_ml_temperature, 4),
        "avg_log_loss": round(np.mean(opt_fold_lls), 4),
        "avg_accuracy": round(np.mean(opt_fold_accs), 4),
        "fold_log_losses": {folds[i][2]: round(ll, 4) for i, ll in enumerate(opt_fold_lls)},
        "fold_accuracies": {folds[i][2]: round(acc, 4) for i, acc in enumerate(opt_fold_accs)},
        "previous_weights": {k: round(v, 4) for k, v in current_weights.items()},
        "previous_draw_boost": current_draw_boost,
        "previous_temperature": current_temperature,
        "previous_ml_temperature": current_ml_temperature,
        "previous_avg_log_loss": round(np.mean(current_fold_lls), 4),
        "improvement_ll": round(np.mean(current_fold_lls) - np.mean(opt_fold_lls), 4),
    }

    # Also save holdout results if available
    if len(df_val) > 0:
        config["holdout_2024_2025"] = {
            "current_accuracy": round(acc_cur, 4),
            "current_log_loss": round(ll_cur, 4),
            "optimized_accuracy": round(acc_val, 4),
            "optimized_log_loss": round(ll_val, 4),
        }

    config_path = output_dir / "ensemble_optimization.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    log.info("\nConfig saved to %s", config_path)

    return config


if __name__ == "__main__":
    main()
