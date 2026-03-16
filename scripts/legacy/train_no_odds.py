#!/usr/bin/env python3
"""Train a prediction model WITHOUT betting odds.

This creates a model suitable for predicting future matches where
betting odds are not yet available.

Usage:
    python scripts/train_no_odds.py [--quick]

The --quick flag does fewer tuning trials for faster iteration.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.training import train_optimized

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train model without odds features")
    parser.add_argument("--quick", action="store_true", help="Use fewer tuning trials")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    args = parser.parse_args()

    n_trials = 10 if args.quick else args.trials

    log.info("=" * 70)
    log.info("TRAINING PREDICTION MODEL WITHOUT BETTING ODDS")
    log.info("=" * 70)
    log.info("")
    log.info("This model will be used to predict FUTURE matches where odds")
    log.info("are not yet available. Features used will include:")
    log.info("  - Elo ratings")
    log.info("  - Team form (goals, xG, points)")
    log.info("  - Head-to-head records")
    log.info("  - League position and points")
    log.info("  - Rolling performance metrics")
    log.info("")
    log.info("Betting odds features will be EXCLUDED:")
    log.info("  - odds_B365H, odds_B365D, odds_B365A")
    log.info("  - odds_PSCH, odds_PSCD, odds_PSCA")
    log.info("  - implied_prob_*")
    log.info("  - And all other odds-derived columns")
    log.info("")
    log.info(f"Tuning trials: {n_trials}")
    log.info("=" * 70)

    results = train_optimized(
        n_tune_trials=n_trials,
        top_k_features=60,  # Use fewer features without odds
        corr_threshold=0.85,
        exclude_odds=True,  # KEY: No betting odds
    )

    log.info("")
    log.info("=" * 70)
    log.info("TRAINING COMPLETE")
    log.info("=" * 70)

    # Print summary
    if "n_features_selected" in results:
        log.info(f"Features: {results['n_features_original']} -> {results['n_features_selected']}")

    for key in ["catboost_cv", "xgboost_cv", "lightgbm_cv"]:
        if key in results:
            cv = results[key]
            avg_acc = cv["accuracy"].mean()
            avg_ll = cv["log_loss"].mean()
            log.info(f"{key}: accuracy={avg_acc:.4f}, log_loss={avg_ll:.4f}")

    if "ensemble_cv" in results:
        ens = results["ensemble_cv"]
        log.info(f"Ensemble: accuracy={ens['ensemble_accuracy'].mean():.4f}")

    # Model paths
    log.info("")
    log.info("Saved models:")
    for key in results:
        if key.endswith("_path"):
            log.info(f"  {key}: {results[key]}")

    return results


if __name__ == "__main__":
    main()
