#!/usr/bin/env python3
"""Ensemble Weight Optimization - Phase 1.1

Tests different weight combinations to find optimal ensemble configuration.

Usage:
    python scripts/optimize_ensemble.py
    python scripts/optimize_ensemble.py --quick  # Fast testing
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from itertools import product

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR, MODELS_DIR
from scripts.prediction.ensemble_prediction_engine import (
    EnsemblePredictor, ENSEMBLE_WEIGHTS,
    XGPredictor, MLClassifier, PlayerXGPredictor,
)
from models.deep_learning import DeepPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# Weight combinations to test
WEIGHT_CONFIGS = {
    "current": {"factor": 0.30, "xg": 0.30, "ml": 0.15, "player_xg": 0.10, "deep": 0.15},
    "conservative": {"factor": 0.40, "xg": 0.30, "ml": 0.15, "player_xg": 0.10, "deep": 0.05},
    "aggressive": {"factor": 0.20, "xg": 0.30, "ml": 0.20, "player_xg": 0.15, "deep": 0.15},
    "balanced": {"factor": 0.25, "xg": 0.25, "ml": 0.20, "player_xg": 0.15, "deep": 0.15},
    "ml_heavy": {"factor": 0.20, "xg": 0.25, "ml": 0.25, "player_xg": 0.15, "deep": 0.15},
    "player_heavy": {"factor": 0.20, "xg": 0.25, "ml": 0.15, "player_xg": 0.25, "deep": 0.15},
    "xg_dominant": {"factor": 0.20, "xg": 0.40, "ml": 0.15, "player_xg": 0.10, "deep": 0.15},
    "factor_light": {"factor": 0.15, "xg": 0.35, "ml": 0.20, "player_xg": 0.15, "deep": 0.15},
    "no_deep": {"factor": 0.30, "xg": 0.35, "ml": 0.20, "player_xg": 0.15, "deep": 0.00},
    "deep_focus": {"factor": 0.20, "xg": 0.25, "ml": 0.15, "player_xg": 0.10, "deep": 0.30},
}


class WeightOptimizer:
    """Optimize ensemble weights through historical backtesting."""

    def __init__(self):
        self.df = None
        self.xg_predictor = None
        self.ml_classifier = None
        self.player_xg = None
        self.deep_predictor = None
        self.results = []

    def load_data(self) -> bool:
        """Load historical data and initialize predictors."""
        log.info("Loading data and initializing predictors...")

        # Load features
        features_path = DATA_DIR / "features" / "features.parquet"
        if not features_path.exists():
            log.error(f"Features not found: {features_path}")
            return False

        self.df = pd.read_parquet(features_path)
        self.df = self.df.sort_values("match_date").reset_index(drop=True)
        self.df = self.df.dropna(subset=["home_score", "away_score", "result"])

        log.info(f"Loaded {len(self.df)} matches")

        # Initialize predictors
        self.xg_predictor = XGPredictor()
        self.xg_predictor.load_models()

        self.ml_classifier = MLClassifier()
        self.ml_classifier.load_model()

        self.player_xg = PlayerXGPredictor()
        self.player_xg.load()

        self.deep_predictor = DeepPredictor()
        self.deep_predictor.load()

        return True

    def get_component_predictions(self, row: pd.Series) -> Dict[str, Dict[str, float]]:
        """Get predictions from each component for a single match."""
        predictions = {}

        home = row["home_team"]
        away = row["away_team"]

        # Factor-based (use base rates as fallback)
        predictions["factor"] = {"prob_H": 0.444, "prob_D": 0.265, "prob_A": 0.291}

        # XG prediction
        try:
            features_df = self.df[self.df.index == row.name]
            xg_result = self.xg_predictor.predict(features_df)
            if xg_result:
                predictions["xg"] = {
                    "prob_H": xg_result["prob_H"],
                    "prob_D": xg_result["prob_D"],
                    "prob_A": xg_result["prob_A"],
                }
        except Exception:
            pass

        # ML prediction
        try:
            features_df = self.df[self.df.index == row.name]
            ml_result = self.ml_classifier.predict(features_df)
            if ml_result:
                predictions["ml"] = ml_result
        except Exception:
            pass

        # Player XG prediction
        try:
            player_result = self.player_xg.predict(home, away)
            if player_result:
                predictions["player_xg"] = {
                    "prob_H": player_result["prob_H"],
                    "prob_D": player_result["prob_D"],
                    "prob_A": player_result["prob_A"],
                }
        except Exception:
            pass

        # Deep learning prediction
        try:
            deep_result = self.deep_predictor.predict(home, away, self.df)
            if deep_result:
                predictions["deep"] = {
                    "prob_H": deep_result["prob_H"],
                    "prob_D": deep_result["prob_D"],
                    "prob_A": deep_result["prob_A"],
                }
        except Exception:
            pass

        return predictions

    def combine_predictions(
        self,
        predictions: Dict[str, Dict[str, float]],
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Combine component predictions with given weights."""
        prob_H = 0.0
        prob_D = 0.0
        prob_A = 0.0
        total_weight = 0.0

        for method, weight in weights.items():
            if method in predictions and weight > 0:
                probs = predictions[method]
                prob_H += probs["prob_H"] * weight
                prob_D += probs["prob_D"] * weight
                prob_A += probs["prob_A"] * weight
                total_weight += weight

        if total_weight > 0:
            prob_H /= total_weight
            prob_D /= total_weight
            prob_A /= total_weight

        # Normalize
        total = prob_H + prob_D + prob_A
        if total > 0:
            prob_H /= total
            prob_D /= total
            prob_A /= total

        return {"prob_H": prob_H, "prob_D": prob_D, "prob_A": prob_A}

    def evaluate_weights(
        self,
        weights: Dict[str, float],
        test_df: pd.DataFrame,
        config_name: str = "custom"
    ) -> Dict:
        """Evaluate a weight configuration on test data."""
        correct = 0
        high_conf_correct = 0
        high_conf_total = 0
        total = 0
        log_loss_sum = 0.0

        result_map = {"H": 0, "D": 1, "A": 2}

        for idx, row in test_df.iterrows():
            actual = row["result"]
            actual_idx = result_map.get(actual)

            if actual_idx is None:
                continue

            # Get component predictions
            predictions = self.get_component_predictions(row)

            if not predictions:
                continue

            # Combine with weights
            combined = self.combine_predictions(predictions, weights)

            # Determine prediction
            probs = [combined["prob_H"], combined["prob_D"], combined["prob_A"]]
            pred_idx = np.argmax(probs)
            max_prob = max(probs)

            # Accuracy
            if pred_idx == actual_idx:
                correct += 1

            # High confidence accuracy
            if max_prob >= 0.50:
                high_conf_total += 1
                if pred_idx == actual_idx:
                    high_conf_correct += 1

            # Log loss
            eps = 1e-10
            log_loss_sum += -np.log(probs[actual_idx] + eps)

            total += 1

        if total == 0:
            return {"error": "no_predictions"}

        accuracy = correct / total
        high_conf_accuracy = high_conf_correct / high_conf_total if high_conf_total > 0 else 0
        avg_log_loss = log_loss_sum / total

        return {
            "config_name": config_name,
            "weights": weights,
            "total_matches": total,
            "accuracy": accuracy,
            "high_conf_accuracy": high_conf_accuracy,
            "high_conf_count": high_conf_total,
            "log_loss": avg_log_loss,
        }

    def run_optimization(
        self,
        test_seasons: List[str] = None,
        sample_size: int = None
    ) -> List[Dict]:
        """Run weight optimization on historical data."""
        if test_seasons is None:
            test_seasons = ["2023-2024", "2024-2025"]

        # Filter test data
        test_df = self.df[self.df["season"].isin(test_seasons)].copy()

        if sample_size and len(test_df) > sample_size:
            test_df = test_df.tail(sample_size)

        log.info(f"Testing on {len(test_df)} matches from seasons: {test_seasons}")

        self.results = []

        for config_name, weights in WEIGHT_CONFIGS.items():
            log.info(f"Testing config: {config_name}")
            result = self.evaluate_weights(weights, test_df, config_name)
            self.results.append(result)
            log.info(f"  Accuracy: {result['accuracy']:.1%}, High-Conf: {result['high_conf_accuracy']:.1%}")

        # Sort by accuracy
        self.results.sort(key=lambda x: x.get("accuracy", 0), reverse=True)

        return self.results

    def grid_search(
        self,
        test_df: pd.DataFrame,
        step: float = 0.05
    ) -> Dict:
        """Perform grid search for optimal weights."""
        log.info("Running grid search for optimal weights...")

        best_result = None
        best_accuracy = 0

        # Generate weight combinations (must sum to 1)
        weight_values = np.arange(0.0, 0.55, step)

        count = 0
        for f in weight_values:
            for x in weight_values:
                for m in weight_values:
                    for p in weight_values:
                        d = 1.0 - f - x - m - p
                        if d < 0 or d > 0.5:
                            continue

                        weights = {
                            "factor": round(f, 2),
                            "xg": round(x, 2),
                            "ml": round(m, 2),
                            "player_xg": round(p, 2),
                            "deep": round(d, 2),
                        }

                        # Quick evaluation (sample)
                        sample_df = test_df.sample(min(100, len(test_df)))
                        result = self.evaluate_weights(weights, sample_df, "grid")

                        if result.get("accuracy", 0) > best_accuracy:
                            best_accuracy = result["accuracy"]
                            best_result = result

                        count += 1
                        if count % 1000 == 0:
                            log.info(f"  Tested {count} combinations, best: {best_accuracy:.1%}")

        return best_result

    def get_recommendations(self) -> Dict:
        """Get recommendations based on optimization results."""
        if not self.results:
            return {"error": "no_results"}

        best = self.results[0]
        current = next((r for r in self.results if r["config_name"] == "current"), None)

        recommendation = {
            "best_config": best["config_name"],
            "best_weights": best["weights"],
            "best_accuracy": best["accuracy"],
            "best_high_conf_accuracy": best["high_conf_accuracy"],
            "current_accuracy": current["accuracy"] if current else None,
            "improvement": best["accuracy"] - current["accuracy"] if current else None,
            "all_results": self.results,
        }

        return recommendation


def main():
    parser = argparse.ArgumentParser(description="Optimize ensemble weights")
    parser.add_argument("--quick", action="store_true", help="Quick test (100 matches)")
    parser.add_argument("--grid", action="store_true", help="Run grid search")
    parser.add_argument("--seasons", nargs="+", default=["2023-2024", "2024-2025"], help="Test seasons")
    args = parser.parse_args()

    optimizer = WeightOptimizer()
    if not optimizer.load_data():
        return

    sample_size = 100 if args.quick else None

    # Run predefined configs
    results = optimizer.run_optimization(
        test_seasons=args.seasons,
        sample_size=sample_size,
    )

    # Print results
    print("\n" + "=" * 70)
    print("WEIGHT OPTIMIZATION RESULTS")
    print("=" * 70)

    for i, result in enumerate(results):
        print(f"\n{i+1}. {result['config_name']}")
        print(f"   Accuracy: {result['accuracy']:.1%}")
        print(f"   High-Conf Accuracy: {result['high_conf_accuracy']:.1%} ({result['high_conf_count']} matches)")
        print(f"   Log Loss: {result['log_loss']:.4f}")
        print(f"   Weights: {result['weights']}")

    # Recommendations
    recs = optimizer.get_recommendations()
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print(f"Best Config: {recs['best_config']}")
    print(f"Best Accuracy: {recs['best_accuracy']:.1%}")
    if recs.get("improvement"):
        print(f"Improvement over current: {recs['improvement']:+.1%}")

    # Save results
    results_dir = DATA_DIR / "optimization"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"weight_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(results_file, "w") as f:
        json.dump(recs, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    # Grid search if requested
    if args.grid:
        print("\n" + "=" * 70)
        print("GRID SEARCH")
        print("=" * 70)
        test_df = optimizer.df[optimizer.df["season"].isin(args.seasons)].tail(200)
        grid_result = optimizer.grid_search(test_df, step=0.10)
        if grid_result:
            print(f"Grid search best: {grid_result['accuracy']:.1%}")
            print(f"Weights: {grid_result['weights']}")


if __name__ == "__main__":
    main()
