"""Re-optimize ensemble weights with BETTING-AWARE objective.

The key insight: optimizing for log-loss converges to 75% market weight because
Pinnacle odds are the most accurate predictor. But that defeats the purpose —
we want to BEAT the market, not copy it.

Market caps are enforced on the NORMALIZED weight (the actual final percentage),
not on the raw sampled value. This ensures "Market ≤ 30%" truly means ≤ 30%.

Strategies:
1. Market = 0%  (pure independent signal)
2. Market ≤ 15%
3. Market ≤ 30%
4. No cap (pure log-loss, for comparison)
5. Profit-max with Market ≤ 30%

Usage:
    python -m scripts.models.optimize_weights_v2 [--n-trials 3000]
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
    predict_factor_based,
    predict_market,
    predict_xg_poisson,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
optuna.logging.set_verbosity(optuna.logging.WARNING)

LABEL_MAP = {"H": 0, "D": 1, "A": 2}


def precompute_predictions(seasons: list[str]) -> tuple:
    """Precompute all method predictions + market odds for betting simulation."""
    config = BacktestConfig(mode="accuracy", seasons=seasons, use_formation=False)
    engine = BacktestEngine(config)
    engine.load_data()

    method_preds = []
    actuals = []
    market_odds = []  # Pinnacle odds for edge calculation

    ml_model = _get_ml_classifier()

    for season in seasons:
        df = engine._get_season_data(season)
        for _, row in df.iterrows():
            preds = {}

            preds["factor"] = predict_factor_based(row)

            xg = predict_xg_poisson(row)
            if xg:
                preds["xg"] = xg

            market = predict_market(row)
            if market:
                preds["market"] = market

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

            if xg:
                preds["player_xg"] = xg

            method_preds.append(preds)
            actuals.append(LABEL_MAP[row["result"]])

            # Store raw Pinnacle odds for edge simulation
            psh = row.get("odds_PSH", 0)
            psd = row.get("odds_PSD", 0)
            psa = row.get("odds_PSA", 0)
            if all(v and v > 1.0 for v in [psh, psd, psa]):
                market_odds.append([psh, psd, psa])
            else:
                market_odds.append([0, 0, 0])

    return method_preds, np.array(actuals), np.array(market_odds)


def blend_predictions(
    method_preds: list,
    weights: Dict[str, float],
    ml_temperature: float,
    draw_boost: float,
) -> np.ndarray:
    """Blend method predictions into ensemble probabilities."""
    n = len(method_preds)
    proba = np.zeros((n, 3))

    for i, preds in enumerate(method_preds):
        total_w = 0.0
        for method, w in weights.items():
            if w <= 0:
                continue
            if method == "ml":
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

    if draw_boost != 1.0:
        proba[:, 1] *= draw_boost
        proba = proba / proba.sum(axis=1, keepdims=True)

    return proba


def compute_all_metrics(
    proba: np.ndarray,
    actuals: np.ndarray,
    market_odds: np.ndarray,
) -> Dict[str, float]:
    """Compute prediction metrics + betting simulation."""
    n = len(actuals)
    eps = 1e-10

    # Prediction metrics
    log_loss = -np.mean(np.log(proba[np.arange(n), actuals] + eps))
    pred_labels = np.argmax(proba, axis=1)
    accuracy = (pred_labels == actuals).mean()

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

    # Betting simulation (flat staking, 1X2 only)
    # Bet when our probability > implied probability + min_edge
    min_edge = 0.03  # 3% minimum edge
    bankroll = 1000.0
    stake = 20.0  # Flat €20 per bet

    bets = 0
    wins = 0
    profit = 0.0

    for i in range(n):
        odds = market_odds[i]
        if odds[0] <= 1.0:
            continue  # No odds available

        for outcome in range(3):
            implied = 1.0 / odds[outcome]
            our_prob = proba[i, outcome]
            edge = our_prob - implied

            if edge > min_edge and our_prob > 0.35:
                bets += 1
                if actuals[i] == outcome:
                    wins += 1
                    profit += stake * (odds[outcome] - 1)
                else:
                    profit -= stake

    win_rate = wins / bets if bets > 0 else 0
    roi = profit / (bets * stake) if bets > 0 else 0
    yield_pct = roi * 100

    # Prediction distribution
    n_h = (pred_labels == 0).sum()
    n_d = (pred_labels == 1).sum()
    n_a = (pred_labels == 2).sum()

    return {
        "log_loss": log_loss,
        "accuracy": accuracy,
        "brier": brier,
        "draw_f1": d_f1,
        "bets": bets,
        "wins": wins,
        "win_rate": win_rate,
        "profit": profit,
        "roi": roi,
        "yield_pct": yield_pct,
        "pred_H_pct": n_h / n,
        "pred_D_pct": n_d / n,
        "pred_A_pct": n_a / n,
    }


def make_objective(method_preds, actuals, market_odds, market_cap=None, mode="logloss"):
    """Create objective function with optional market weight cap.

    When market_cap is set (e.g. 0.30), the market's NORMALIZED weight is
    guaranteed to be ≤ market_cap. We achieve this by sampling market weight
    directly as a fraction of 1.0, then splitting the remainder among the
    other methods.
    """

    def objective(trial):
        if market_cap is not None and market_cap > 0:
            # Sample market weight directly as the final normalized percentage
            w_market_final = trial.suggest_float("w_market", 0.0, market_cap)
            remaining = 1.0 - w_market_final

            # Sample raw shares for the other methods (will be normalized to fill `remaining`)
            s_factor = trial.suggest_float("s_factor", 0.0, 1.0)
            s_xg = trial.suggest_float("s_xg", 0.0, 1.0)
            s_ml = trial.suggest_float("s_ml", 0.1, 1.0)  # ML always gets some share
            s_player_xg = trial.suggest_float("s_player_xg", 0.0, 0.3)

            s_total = s_factor + s_xg + s_ml + s_player_xg
            weights = {
                "factor": remaining * s_factor / s_total,
                "xg": remaining * s_xg / s_total,
                "ml": remaining * s_ml / s_total,
                "player_xg": remaining * s_player_xg / s_total,
                "market": w_market_final,
            }
        elif market_cap == 0:
            # No market at all — split among independent methods only
            s_factor = trial.suggest_float("s_factor", 0.0, 1.0)
            s_xg = trial.suggest_float("s_xg", 0.0, 1.0)
            s_ml = trial.suggest_float("s_ml", 0.1, 1.0)
            s_player_xg = trial.suggest_float("s_player_xg", 0.0, 0.3)

            s_total = s_factor + s_xg + s_ml + s_player_xg
            weights = {
                "factor": s_factor / s_total,
                "xg": s_xg / s_total,
                "ml": s_ml / s_total,
                "player_xg": s_player_xg / s_total,
                "market": 0.0,
            }
        else:
            # No cap — original behavior
            w_factor = trial.suggest_float("w_factor", 0.0, 0.4)
            w_xg = trial.suggest_float("w_xg", 0.0, 0.4)
            w_ml = trial.suggest_float("w_ml", 0.05, 0.85)
            w_player_xg = trial.suggest_float("w_player_xg", 0.0, 0.15)
            w_market = trial.suggest_float("w_market", 0.0, 0.85)

            total = w_factor + w_xg + w_ml + w_player_xg + w_market
            weights = {
                "factor": w_factor / total,
                "xg": w_xg / total,
                "ml": w_ml / total,
                "player_xg": w_player_xg / total,
                "market": w_market / total,
            }

        ml_temperature = trial.suggest_float("ml_temperature", 0.2, 2.0)
        draw_boost = trial.suggest_float("draw_boost", 0.8, 1.5)

        proba = blend_predictions(method_preds, weights, ml_temperature, draw_boost)
        metrics = compute_all_metrics(proba, actuals, market_odds)

        for k, v in metrics.items():
            trial.set_user_attr(k, v)
        for k, v in weights.items():
            trial.set_user_attr(f"w_{k}", v)

        if mode == "logloss":
            return metrics["log_loss"]
        elif mode == "edge_aware":
            ll = metrics["log_loss"]
            roi = metrics["roi"]
            edge_bonus = -0.05 * max(0, roi) + 0.1 * max(0, -roi)
            return ll + edge_bonus
        elif mode == "profit":
            return -metrics["profit"]

    return objective


def run_optimization(
    method_preds, actuals, market_odds,
    label: str, n_trials: int,
    market_cap=None, mode="logloss",
) -> dict:
    """Run a single optimization and return results."""
    log.info("=== %s (%d trials) ===", label, n_trials)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=min(100, n_trials // 5)),
    )

    obj = make_objective(method_preds, actuals, market_odds, market_cap=market_cap, mode=mode)
    study.optimize(obj, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    best_weights = {
        "factor": best.user_attrs["w_factor"],
        "xg": best.user_attrs["w_xg"],
        "ml": best.user_attrs["w_ml"],
        "player_xg": best.user_attrs["w_player_xg"],
        "market": best.user_attrs["w_market"],
    }

    proba = blend_predictions(
        method_preds, best_weights,
        best.params["ml_temperature"], best.params["draw_boost"],
    )
    metrics = compute_all_metrics(proba, actuals, market_odds)

    # Top 5 stability (filter out failed trials with None values)
    completed = [t for t in study.trials if t.value is not None and not np.isnan(t.value)]
    top5 = sorted(completed, key=lambda t: t.value)[:5]

    return {
        "label": label,
        "best_trial": best.number,
        "weights": best_weights,
        "ml_temperature": best.params["ml_temperature"],
        "draw_boost": best.params["draw_boost"],
        "metrics": metrics,
        "top5": [
            {
                "ll": t.user_attrs.get("log_loss", t.value),
                "acc": t.user_attrs.get("accuracy", 0),
                "profit": t.user_attrs.get("profit", 0),
                "bets": t.user_attrs.get("bets", 0),
                "roi": t.user_attrs.get("roi", 0),
                "ml": t.user_attrs.get("w_ml", 0),
                "mkt": t.user_attrs.get("w_market", 0),
                "fac": t.user_attrs.get("w_factor", 0),
                "xg": t.user_attrs.get("w_xg", 0),
            }
            for t in top5
        ],
    }


def main(n_trials: int = 3000):
    log.info("Precomputing predictions...")
    method_preds, actuals, market_odds = precompute_predictions(["2023-2024", "2024-2025"])
    log.info("Precomputed %d matches (%d with odds)", len(actuals), (market_odds[:, 0] > 1).sum())

    # Current baseline
    current_weights = {"factor": 0.098, "xg": 0.104, "ml": 0.689, "player_xg": 0.006, "market": 0.104}
    current_proba = blend_predictions(method_preds, current_weights, 0.54, 1.0)
    current_metrics = compute_all_metrics(current_proba, actuals, market_odds)

    results = {}

    # Run 1: Market = 0% (pure independent signal — no market at all)
    results["no_market"] = run_optimization(
        method_preds, actuals, market_odds,
        "Market = 0%", n_trials, market_cap=0, mode="logloss",
    )

    # Run 2: Market ≤ 15% (normalized)
    results["cap15"] = run_optimization(
        method_preds, actuals, market_odds,
        "Market ≤ 15%", n_trials, market_cap=0.15, mode="logloss",
    )

    # Run 3: Market ≤ 30% (normalized)
    results["cap30"] = run_optimization(
        method_preds, actuals, market_odds,
        "Market ≤ 30%", n_trials, market_cap=0.30, mode="logloss",
    )

    # Run 4: No cap (pure log-loss, for comparison)
    results["uncapped"] = run_optimization(
        method_preds, actuals, market_odds,
        "No cap (pure LL)", n_trials, market_cap=None, mode="logloss",
    )

    # Run 5: Profit-max with Market ≤ 30%
    results["profit_cap30"] = run_optimization(
        method_preds, actuals, market_odds,
        "Profit-max ≤30%mkt", n_trials, market_cap=0.30, mode="profit",
    )

    # Print comparison
    print("\n" + "=" * 100)
    print("WEIGHT OPTIMIZATION COMPARISON (normalized market caps)")
    print("=" * 100)

    print(f"\n  Current weights: factor={current_weights['factor']:.3f} xg={current_weights['xg']:.3f} "
          f"ml={current_weights['ml']:.3f} pxg={current_weights['player_xg']:.3f} "
          f"mkt={current_weights['market']:.3f} | T=0.54 boost=1.0")
    print(f"  Current metrics: acc={current_metrics['accuracy']:.4f} ll={current_metrics['log_loss']:.4f} "
          f"bets={current_metrics['bets']} wins={current_metrics['wins']} "
          f"profit=€{current_metrics['profit']:.0f} roi={current_metrics['yield_pct']:.1f}%")

    print(f"\n  {'Strategy':<22} {'ML%':>6} {'Mkt%':>6} {'xG%':>6} {'Fac%':>6} {'PxG%':>6} "
          f"{'T':>5} {'Bst':>5} | {'Acc':>7} {'LL':>7} {'Bets':>5} {'Wins':>5} {'Profit':>8} {'ROI':>7}")
    print("  " + "-" * 110)

    for key, r in results.items():
        w = r["weights"]
        m = r["metrics"]
        print(f"  {r['label']:<22} {w['ml']*100:>5.1f}% {w['market']*100:>5.1f}% "
              f"{w['xg']*100:>5.1f}% {w['factor']*100:>5.1f}% {w['player_xg']*100:>5.1f}% "
              f"{r['ml_temperature']:>5.2f} {r['draw_boost']:>5.2f} "
              f"| {m['accuracy']:>6.1%} {m['log_loss']:>7.4f} "
              f"{m['bets']:>5} {m['wins']:>5} €{m['profit']:>7.0f} {m['yield_pct']:>6.1f}%")

    # Detailed top-5 for each strategy
    for key, r in results.items():
        print(f"\n  {r['label']} — Top 5 trials:")
        for i, t in enumerate(r["top5"]):
            print(f"    #{i+1}: ll={t['ll']:.4f} acc={t['acc']:.4f} profit=€{t['profit']:.0f} "
                  f"bets={t['bets']} roi={t['roi']*100:.1f}% | ML={t['ml']:.3f} mkt={t['mkt']:.3f} "
                  f"xg={t['xg']:.3f} fac={t['fac']:.3f}")

    print("\n" + "=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)

    # Find best by different criteria
    best_ll = min(results.values(), key=lambda r: r["metrics"]["log_loss"])
    best_profit = max(results.values(), key=lambda r: r["metrics"]["profit"])
    best_roi = max(results.values(), key=lambda r: r["metrics"]["roi"] if r["metrics"]["bets"] > 20 else -999)

    print(f"  Best log-loss:  {best_ll['label']} (ll={best_ll['metrics']['log_loss']:.4f})")
    print(f"  Best profit:    {best_profit['label']} (€{best_profit['metrics']['profit']:.0f} from {best_profit['metrics']['bets']} bets)")
    print(f"  Best ROI:       {best_roi['label']} (ROI={best_roi['metrics']['yield_pct']:.1f}% from {best_roi['metrics']['bets']} bets)")

    # Trade-off: how much log-loss do we sacrifice for independence?
    if "cap30" in results and "uncapped" in results:
        ll_cost = results["cap30"]["metrics"]["log_loss"] - results["uncapped"]["metrics"]["log_loss"]
        print(f"\n  Market ≤ 30% costs {ll_cost:+.4f} log-loss vs uncapped")
        print(f"    but: {results['cap30']['metrics']['bets']} bets @ {results['cap30']['metrics']['yield_pct']:.1f}% ROI "
              f"vs {results['uncapped']['metrics']['bets']} bets @ {results['uncapped']['metrics']['yield_pct']:.1f}% ROI")
    print("=" * 100)

    # Save full report
    report = {
        "timestamp": datetime.now().isoformat(),
        "n_trials_per_strategy": n_trials,
        "current": {"weights": current_weights, "metrics": current_metrics},
        "strategies": {k: {
            "weights": r["weights"],
            "ml_temperature": r["ml_temperature"],
            "draw_boost": r["draw_boost"],
            "metrics": r["metrics"],
            "top5": r["top5"],
        } for k, r in results.items()},
    }

    report_path = DATA_DIR / "optimization" / "weight_optimization_v2_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Report saved to %s", report_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=3000)
    args = parser.parse_args()
    main(n_trials=args.n_trials)
