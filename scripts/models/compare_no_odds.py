"""Compare no-odds ML model vs current odds-based model in the ensemble backtest.

Runs the same walk-forward accuracy evaluation twice:
1. With the current model (odds-based CatBoost/ensemble)
2. With the no-odds CatBoost (team-performance-only features)

This tells us whether replacing the odds-dependent ML signal with an
independent team-performance signal improves ensemble diversity and accuracy.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import DATA_DIR, MODELS_DIR
from scripts.analysis import backtest_unified as bt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def load_no_odds_catboost():
    """Load the no-odds CatBoost model and its feature list."""
    from catboost import CatBoostClassifier

    model_path = MODELS_DIR / "universal" / "no_odds" / "catboost_latest.cbm"
    meta_path = MODELS_DIR / "universal" / "no_odds" / "catboost_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"No-odds model not found at {model_path}")

    model = CatBoostClassifier()
    model.load_model(str(model_path))

    with open(meta_path) as f:
        meta = json.load(f)

    return model, meta["feature_names"]


def run_backtest(label: str) -> dict:
    """Run accuracy backtest on seasons 2023-2026 and return metrics."""
    config = bt.BacktestConfig(
        mode="accuracy",
        seasons=["2023-2024", "2024-2025"],
        use_formation=False,
        compare_baseline=False,
        store_predictions=True,
    )
    engine = bt.BacktestEngine(config)
    log.info("=== Running backtest: %s ===", label)
    results = engine.run()

    accuracy_results = results.get("accuracy", results)
    return accuracy_results


def main():
    # --- Run 1: Current model (with odds) ---
    log.info("\n" + "=" * 70)
    log.info("PHASE 1: Backtest with CURRENT model (odds-based)")
    log.info("=" * 70)

    current_results = run_backtest("current (with odds)")

    # Reset the global ML classifier so we can load a different one
    bt._ml_classifier = None
    bt._ml_is_ensemble = False

    # --- Load and inject no-odds model ---
    log.info("\n" + "=" * 70)
    log.info("PHASE 2: Backtest with NO-ODDS model")
    log.info("=" * 70)

    no_odds_model, no_odds_features = load_no_odds_catboost()

    # Monkey-patch the classifier loader to return our no-odds model
    _original_get_ml = bt._get_ml_classifier

    def _get_no_odds_classifier():
        bt._ml_classifier = no_odds_model
        bt._ml_is_ensemble = False
        return no_odds_model

    bt._get_ml_classifier = _get_no_odds_classifier

    no_odds_results = run_backtest("no-odds CatBoost")

    # Restore original
    bt._get_ml_classifier = _original_get_ml
    bt._ml_classifier = None

    # --- Run 3: No ML at all (factor + xG + market only) ---
    log.info("\n" + "=" * 70)
    log.info("PHASE 3: Backtest WITHOUT ML (baseline)")
    log.info("=" * 70)

    # Set ML to return None so ensemble runs without it
    def _get_no_ml():
        return None

    bt._get_ml_classifier = _get_no_ml

    no_ml_results = run_backtest("no ML (factor+xG+market)")

    bt._get_ml_classifier = _original_get_ml
    bt._ml_classifier = None

    # --- Comparison ---
    print("\n" + "=" * 70)
    print("ENSEMBLE BACKTEST COMPARISON (2023-2026)")
    print("=" * 70)

    def extract_metrics(r):
        return {
            "accuracy": r.get("accuracy", 0),
            "log_loss": r.get("log_loss", 0),
            "brier_score": r.get("brier_score", 0),
            "rps": r.get("rps", 0),
            "ece": r.get("ece", 0),
            "n_matches": r.get("n_matches", 0),
        }

    current_m = extract_metrics(current_results)
    no_odds_m = extract_metrics(no_odds_results)
    no_ml_m = extract_metrics(no_ml_results)

    header = f"  {'Metric':<18} {'Current (odds)':>16} {'No-odds ML':>16} {'No ML':>16} {'Delta (no-odds vs current)':>28}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for metric in ["accuracy", "log_loss", "brier_score", "rps", "ece"]:
        c = current_m[metric]
        n = no_odds_m[metric]
        b = no_ml_m[metric]
        delta = n - c
        # For accuracy, positive delta = better. For loss metrics, negative = better
        if metric == "accuracy":
            arrow = "+" if delta > 0 else ""
        else:
            arrow = "" if delta > 0 else ""  # negative is better for loss
        print(f"  {metric:<18} {c:>16.4f} {n:>16.4f} {b:>16.4f} {arrow}{delta:>+27.4f}")

    print(f"\n  {'n_matches':<18} {current_m['n_matches']:>16} {no_odds_m['n_matches']:>16} {no_ml_m['n_matches']:>16}")

    # Per-season breakdown
    for label, results in [("Current (odds)", current_results), ("No-odds ML", no_odds_results), ("No ML", no_ml_results)]:
        seasons = results.get("seasons", {})
        if seasons:
            print(f"\n  {label} — per-season:")
            for season, sr in sorted(seasons.items()):
                print(f"    {season}: acc={sr.get('accuracy', 0):.4f}  logloss={sr.get('log_loss', 0):.4f}")

    # Draw prediction analysis
    print("\n  Draw prediction analysis:")
    for label, results in [("Current (odds)", current_results), ("No-odds ML", no_odds_results), ("No ML", no_ml_results)]:
        records = results.get("prediction_records", [])
        if records:
            n_draws_predicted = sum(1 for r in records if r.get("predicted") == "D")
            n_draws_actual = sum(1 for r in records if r.get("actual") == "D")
            n_draws_correct = sum(1 for r in records if r.get("predicted") == "D" and r.get("actual") == "D")
            total = len(records)
            print(f"    {label:20s}: predicted {n_draws_predicted}/{total} draws, actual {n_draws_actual}/{total}, correct {n_draws_correct}")

    # Verdict
    print("\n" + "=" * 70)
    accuracy_improved = no_odds_m["accuracy"] > current_m["accuracy"]
    logloss_improved = no_odds_m["log_loss"] < current_m["log_loss"]
    brier_improved = no_odds_m["brier_score"] < current_m["brier_score"]

    improvements = sum([accuracy_improved, logloss_improved, brier_improved])
    if improvements >= 2:
        print("VERDICT: NO-ODDS MODEL IMPROVES THE ENSEMBLE")
        print(f"  Accuracy: {'BETTER' if accuracy_improved else 'WORSE'}")
        print(f"  Log-loss: {'BETTER' if logloss_improved else 'WORSE'}")
        print(f"  Brier:    {'BETTER' if brier_improved else 'WORSE'}")
        print("  -> Recommend deploying no-odds model as ML component")
    else:
        print("VERDICT: NO-ODDS MODEL DOES NOT IMPROVE THE ENSEMBLE")
        print(f"  Accuracy: {'BETTER' if accuracy_improved else 'WORSE'}")
        print(f"  Log-loss: {'BETTER' if logloss_improved else 'WORSE'}")
        print(f"  Brier:    {'BETTER' if brier_improved else 'WORSE'}")
        if no_odds_m["accuracy"] > no_ml_m["accuracy"]:
            print("  -> No-odds ML is better than no ML; consider re-weighting")
        else:
            print("  -> No improvement; keep current model")
    print("=" * 70)

    # Save comparison report
    report = {
        "timestamp": datetime.now().isoformat(),
        "comparison": {
            "current_with_odds": current_m,
            "no_odds_ml": no_odds_m,
            "no_ml_baseline": no_ml_m,
        },
        "ensemble_weights": {
            "current": dict(bt.BACKTEST_WEIGHTS),
        },
        "no_odds_features": no_odds_features,
        "verdict": {
            "accuracy_improved": accuracy_improved,
            "logloss_improved": logloss_improved,
            "brier_improved": brier_improved,
            "recommend_deploy": improvements >= 2,
        },
    }

    report_path = MODELS_DIR / "universal" / "no_odds" / "ensemble_comparison.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Comparison report saved to %s", report_path)


if __name__ == "__main__":
    main()
