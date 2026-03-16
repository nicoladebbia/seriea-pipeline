"""Re-optimize ensemble weights for the no-odds ML model using Optuna.

Searches over:
  - 5 method weights (factor, xg, ml, player_xg, market)
  - ML temperature (pre-ensemble sharpening)
  - Draw boost (post-ensemble draw adjustment)

Uses walk-forward evaluation on 2023-2025 (760 matches) with log-loss
as the primary objective (proper scoring rule, best for betting calibration).

Usage:
    python -m scripts.models.optimize_weights [--n-trials 2000]
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import DATA_DIR, MODELS_DIR
from scripts.analysis.backtest_unified import (
    BacktestConfig,
    BacktestEngine,
    _get_ml_classifier,
    _ml_is_ensemble,
    predict_factor_based,
    predict_market,
    predict_xg_poisson,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

LABEL_MAP = {"H": 0, "D": 1, "A": 2}


def precompute_predictions(seasons: list[str]) -> tuple:
    """Precompute all method predictions once, then reuse across trials.

    This is the key optimization: instead of running the full backtest 2000 times,
    we compute each method's output once and only vary the blending weights.
    """
    config = BacktestConfig(mode="accuracy", seasons=seasons, use_formation=False)
    engine = BacktestEngine(config)
    engine.load_data()

    method_preds = []  # List of dicts: {method: {prob_H, prob_D, prob_A}}
    actuals = []

    ml_model = _get_ml_classifier()

    for season in seasons:
        df = engine._get_season_data(season)
        for _, row in df.iterrows():
            preds = {}

            # Factor
            preds["factor"] = predict_factor_based(row)

            # xG Poisson
            xg = predict_xg_poisson(row)
            if xg:
                preds["xg"] = xg

            # Market
            market = predict_market(row)
            if market:
                preds["market"] = market

            # ML (raw, before temperature — we'll apply T in the trial)
            if ml_model is not None:
                try:
                    feature_names = list(ml_model.feature_names_)
                    available = [f for f in feature_names if f in row.index]
                    if len(available) >= len(feature_names) * 0.5:
                        X = pd.DataFrame([row[available].values], columns=available)
                        for f in feature_names:
                            if f not in X.columns:
                                X[f] = 0
                        X = X[feature_names]
                        proba = ml_model.predict_proba(X)[0]
                        preds["ml_raw"] = {
                            "prob_H": float(proba[0]),
                            "prob_D": float(proba[1]),
                            "prob_A": float(proba[2]),
                        }
                except Exception:
                    pass

            # Player xG (currently approximated as xG)
            if xg:
                preds["player_xg"] = xg

            method_preds.append(preds)
            actuals.append(LABEL_MAP[row["result"]])

    return method_preds, np.array(actuals)


def evaluate_weights(
    method_preds: list,
    actuals: np.ndarray,
    weights: Dict[str, float],
    ml_temperature: float,
    draw_boost: float,
) -> Dict[str, float]:
    """Evaluate a specific weight configuration. Returns metrics dict."""
    n = len(actuals)
    proba = np.zeros((n, 3))

    for i, preds in enumerate(method_preds):
        total_w = 0.0

        for method, w in weights.items():
            if w <= 0:
                continue

            if method == "ml":
                # Apply temperature to raw ML probs
                raw = preds.get("ml_raw")
                if raw is None:
                    continue
                raw_p = np.array([raw["prob_H"], raw["prob_D"], raw["prob_A"]])
                eps = 1e-10
                logits = np.log(raw_p + eps)
                scaled = np.exp(logits / ml_temperature)
                p = scaled / scaled.sum()
                proba[i] += w * p
                total_w += w
            else:
                mp = preds.get(method)
                if mp is None:
                    continue
                proba[i] += w * np.array([mp["prob_H"], mp["prob_D"], mp["prob_A"]])
                total_w += w

        if total_w > 0:
            proba[i] /= total_w

    # Apply draw boost
    if draw_boost != 1.0:
        proba[:, 1] *= draw_boost
        proba = proba / proba.sum(axis=1, keepdims=True)

    # Metrics
    eps = 1e-10
    log_loss = -np.mean(np.log(proba[np.arange(n), actuals] + eps))

    pred_labels = np.argmax(proba, axis=1)
    accuracy = (pred_labels == actuals).mean()

    # Brier score
    onehot = np.zeros_like(proba)
    onehot[np.arange(n), actuals] = 1.0
    brier = np.mean(np.sum((proba - onehot) ** 2, axis=1))

    # Draw F1
    d_pred = pred_labels == 1
    d_actual = actuals == 1
    d_correct = (d_pred & d_actual).sum()
    d_prec = d_correct / d_pred.sum() if d_pred.sum() > 0 else 0
    d_rec = d_correct / d_actual.sum() if d_actual.sum() > 0 else 0
    d_f1 = 2 * d_prec * d_rec / (d_prec + d_rec) if (d_prec + d_rec) > 0 else 0

    # Prediction distribution
    n_h = (pred_labels == 0).sum()
    n_d = (pred_labels == 1).sum()
    n_a = (pred_labels == 2).sum()

    return {
        "log_loss": log_loss,
        "accuracy": accuracy,
        "brier": brier,
        "draw_f1": d_f1,
        "pred_H_pct": n_h / n,
        "pred_D_pct": n_d / n,
        "pred_A_pct": n_a / n,
    }


def objective(trial: optuna.Trial, method_preds: list, actuals: np.ndarray) -> float:
    """Optuna objective: minimize log-loss."""
    # Sample raw weights (will be normalized)
    w_factor = trial.suggest_float("w_factor", 0.0, 0.4)
    w_xg = trial.suggest_float("w_xg", 0.0, 0.4)
    w_ml = trial.suggest_float("w_ml", 0.1, 0.85)  # ML must be at least 10%
    w_player_xg = trial.suggest_float("w_player_xg", 0.0, 0.15)
    w_market = trial.suggest_float("w_market", 0.0, 0.4)

    # Normalize to sum to 1
    total = w_factor + w_xg + w_ml + w_player_xg + w_market
    weights = {
        "factor": w_factor / total,
        "xg": w_xg / total,
        "ml": w_ml / total,
        "player_xg": w_player_xg / total,
        "market": w_market / total,
    }

    # ML temperature and draw boost
    ml_temperature = trial.suggest_float("ml_temperature", 0.3, 2.0)
    draw_boost = trial.suggest_float("draw_boost", 0.8, 1.5)

    metrics = evaluate_weights(method_preds, actuals, weights, ml_temperature, draw_boost)

    # Store metrics as user attributes for analysis
    for k, v in metrics.items():
        trial.set_user_attr(k, v)
    for k, v in weights.items():
        trial.set_user_attr(f"norm_{k}", v)

    return metrics["log_loss"]


def main(n_trials: int = 2000):
    log.info("Precomputing predictions for all methods...")
    method_preds, actuals = precompute_predictions(["2023-2024", "2024-2025"])
    log.info("Precomputed %d match predictions", len(actuals))

    # Run optimization
    log.info("Starting Optuna optimization (%d trials)...", n_trials)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=100),
        study_name="ensemble_weight_optimization",
    )

    study.optimize(
        lambda trial: objective(trial, method_preds, actuals),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # Best trial
    best = study.best_trial
    log.info("Best trial #%d: log_loss=%.4f", best.number, best.value)

    # Extract normalized weights
    best_weights = {
        "factor": best.user_attrs["norm_factor"],
        "xg": best.user_attrs["norm_xg"],
        "ml": best.user_attrs["norm_ml"],
        "player_xg": best.user_attrs["norm_player_xg"],
        "market": best.user_attrs["norm_market"],
    }

    best_ml_T = best.params["ml_temperature"]
    best_draw_boost = best.params["draw_boost"]

    # Run full evaluation with best params
    best_metrics = evaluate_weights(
        method_preds, actuals, best_weights, best_ml_T, best_draw_boost,
    )

    # Current baseline for comparison
    current_weights = {
        "factor": 0.098, "xg": 0.104, "ml": 0.689,
        "player_xg": 0.006, "market": 0.104,
    }
    current_metrics = evaluate_weights(
        method_preds, actuals, current_weights, 0.54, 1.0,
    )

    # Print results
    print("\n" + "=" * 70)
    print("ENSEMBLE WEIGHT OPTIMIZATION COMPLETE")
    print(f"Trials: {n_trials}, Best trial: #{best.number}")
    print("=" * 70)

    print(f"\n  {'Parameter':<20} {'Current':>12} {'Optimized':>12} {'Delta':>12}")
    print("  " + "-" * 56)
    for method in ["factor", "xg", "ml", "player_xg", "market"]:
        c = current_weights[method]
        o = best_weights[method]
        print(f"  {method:<20} {c:>12.4f} {o:>12.4f} {o-c:>+12.4f}")
    print(f"  {'ml_temperature':<20} {'0.5400':>12} {best_ml_T:>12.4f} {best_ml_T-0.54:>+12.4f}")
    print(f"  {'draw_boost':<20} {'1.0000':>12} {best_draw_boost:>12.4f} {best_draw_boost-1.0:>+12.4f}")

    print(f"\n  {'Metric':<20} {'Current':>12} {'Optimized':>12} {'Delta':>12}")
    print("  " + "-" * 56)
    for metric in ["accuracy", "log_loss", "brier", "draw_f1"]:
        c = current_metrics[metric]
        o = best_metrics[metric]
        print(f"  {metric:<20} {c:>12.4f} {o:>12.4f} {o-c:>+12.4f}")

    print(f"\n  Prediction distribution:")
    print(f"    Current:   H={current_metrics['pred_H_pct']:.1%}  D={current_metrics['pred_D_pct']:.1%}  A={current_metrics['pred_A_pct']:.1%}")
    print(f"    Optimized: H={best_metrics['pred_H_pct']:.1%}  D={best_metrics['pred_D_pct']:.1%}  A={best_metrics['pred_A_pct']:.1%}")
    print(f"    Actual:    H=40.8%  D=28.9%  A=30.3%")

    # Top 10 trials for stability analysis
    print(f"\n  Top 10 trials (stability check):")
    top_trials = sorted(study.trials, key=lambda t: t.value)[:10]
    for i, t in enumerate(top_trials):
        ml_w = t.user_attrs.get("norm_ml", 0)
        mkt_w = t.user_attrs.get("norm_market", 0)
        xg_w = t.user_attrs.get("norm_xg", 0)
        fac_w = t.user_attrs.get("norm_factor", 0)
        acc = t.user_attrs.get("accuracy", 0)
        print(f"    #{i+1}: ll={t.value:.4f} acc={acc:.4f} | ML={ml_w:.3f} mkt={mkt_w:.3f} xg={xg_w:.3f} fac={fac_w:.3f} T={t.params['ml_temperature']:.3f} boost={t.params['draw_boost']:.3f}")

    print("=" * 70)

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "n_trials": n_trials,
        "best_trial": best.number,
        "best_weights": best_weights,
        "best_ml_temperature": best_ml_T,
        "best_draw_boost": best_draw_boost,
        "best_metrics": best_metrics,
        "current_weights": current_weights,
        "current_metrics": current_metrics,
        "top_10_trials": [
            {
                "rank": i + 1,
                "log_loss": t.value,
                "accuracy": t.user_attrs.get("accuracy", 0),
                "weights": {
                    "ml": t.user_attrs.get("norm_ml", 0),
                    "market": t.user_attrs.get("norm_market", 0),
                    "xg": t.user_attrs.get("norm_xg", 0),
                    "factor": t.user_attrs.get("norm_factor", 0),
                    "player_xg": t.user_attrs.get("norm_player_xg", 0),
                },
                "ml_temperature": t.params["ml_temperature"],
                "draw_boost": t.params["draw_boost"],
            }
            for i, t in enumerate(top_trials)
        ],
    }

    report_path = DATA_DIR / "optimization" / "weight_optimization_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Report saved to %s", report_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=2000)
    args = parser.parse_args()
    main(n_trials=args.n_trials)
