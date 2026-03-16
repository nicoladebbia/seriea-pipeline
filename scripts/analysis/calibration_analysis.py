#!/usr/bin/env python3
"""Model Calibration Analysis - Phase 1.2

Analyzes model calibration and applies temperature/Platt scaling.

Usage:
    python scripts/calibration_analysis.py
    python scripts/calibration_analysis.py --fix  # Apply calibration fixes
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import DATA_DIR, MODELS_DIR
from scripts.prediction.ensemble_prediction_engine import (
    EnsemblePredictor, ENSEMBLE_WEIGHTS,
    XGPredictor, MLClassifier, PlayerXGPredictor,
)
from models.deep_learning import DeepPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class CalibrationAnalyzer:
    """Analyze and fix model calibration."""

    def __init__(self):
        self.df = None
        self.xg_predictor = None
        self.ml_classifier = None
        self.player_xg = None
        self.deep_predictor = None

        # Calibration parameters
        self.temperatures = {}  # Temperature scaling per model
        self.platt_scalers = {}  # Platt scaling per model

    def load_data(self) -> bool:
        """Load data and predictors."""
        log.info("Loading data and predictors...")

        features_path = DATA_DIR / "features" / "features.parquet"
        if not features_path.exists():
            return False

        self.df = pd.read_parquet(features_path)
        self.df = self.df.sort_values("match_date").reset_index(drop=True)
        self.df = self.df.dropna(subset=["home_score", "away_score", "result"])

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

    def get_model_predictions(
        self,
        model_name: str,
        test_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions and actuals for a model."""
        all_probs = []
        all_actuals = []

        result_map = {"H": 0, "D": 1, "A": 2}

        for idx, row in test_df.iterrows():
            actual = row["result"]
            actual_idx = result_map.get(actual)
            if actual_idx is None:
                continue

            probs = None

            if model_name == "xg":
                try:
                    features_df = self.df[self.df.index == row.name]
                    result = self.xg_predictor.predict(features_df)
                    if result:
                        probs = [result["prob_H"], result["prob_D"], result["prob_A"]]
                except Exception as e:
                    log.debug(f"Failed xg prediction for calibration row {row.name}: {e}")

            elif model_name == "ml":
                try:
                    features_df = self.df[self.df.index == row.name]
                    result = self.ml_classifier.predict(features_df)
                    if result:
                        probs = [result["prob_H"], result["prob_D"], result["prob_A"]]
                except Exception as e:
                    log.debug(f"Failed ml prediction for calibration row {row.name}: {e}")

            elif model_name == "player_xg":
                try:
                    result = self.player_xg.predict(row["home_team"], row["away_team"])
                    if result:
                        probs = [result["prob_H"], result["prob_D"], result["prob_A"]]
                except Exception as e:
                    log.debug(f"Failed player_xg prediction for calibration row {row.name}: {e}")

            elif model_name == "deep":
                try:
                    result = self.deep_predictor.predict(
                        row["home_team"], row["away_team"], self.df
                    )
                    if result:
                        probs = [result["prob_H"], result["prob_D"], result["prob_A"]]
                except Exception as e:
                    log.debug(f"Failed deep prediction for calibration row {row.name}: {e}")

            if probs:
                all_probs.append(probs)
                all_actuals.append(actual_idx)

        return np.array(all_probs), np.array(all_actuals)

    def calculate_calibration_error(
        self,
        probs: np.ndarray,
        actuals: np.ndarray,
        n_bins: int = 10
    ) -> Dict:
        """Calculate Expected Calibration Error and reliability diagram data."""
        if len(probs) == 0:
            return {"error": "no_data"}

        # For multiclass, calculate per-class calibration
        n_classes = probs.shape[1]

        results = {
            "ece": 0.0,
            "mce": 0.0,  # Maximum calibration error
            "per_class": {},
            "bins": [],
        }

        total_ece = 0.0

        for c in range(n_classes):
            class_name = {0: "H", 1: "D", 2: "A"}[c]
            binary_labels = (actuals == c).astype(int)
            class_probs = probs[:, c]

            class_ece = 0.0
            class_bins = []

            for bin_idx in range(n_bins):
                bin_lower = bin_idx / n_bins
                bin_upper = (bin_idx + 1) / n_bins

                in_bin = (class_probs >= bin_lower) & (class_probs < bin_upper)
                bin_size = in_bin.sum()

                if bin_size > 0:
                    avg_confidence = class_probs[in_bin].mean()
                    avg_accuracy = binary_labels[in_bin].mean()
                    gap = abs(avg_confidence - avg_accuracy)

                    class_ece += (bin_size / len(actuals)) * gap

                    class_bins.append({
                        "bin": bin_idx,
                        "confidence": float(avg_confidence),
                        "accuracy": float(avg_accuracy),
                        "gap": float(gap),
                        "count": int(bin_size),
                    })

            results["per_class"][class_name] = {
                "ece": float(class_ece),
                "bins": class_bins,
            }
            total_ece += class_ece

        results["ece"] = total_ece / n_classes

        # Overall reliability
        max_probs = probs.max(axis=1)
        predicted_class = probs.argmax(axis=1)
        correct = (predicted_class == actuals)

        overall_bins = []
        for bin_idx in range(n_bins):
            bin_lower = bin_idx / n_bins
            bin_upper = (bin_idx + 1) / n_bins

            in_bin = (max_probs >= bin_lower) & (max_probs < bin_upper)
            bin_size = in_bin.sum()

            if bin_size > 0:
                avg_confidence = max_probs[in_bin].mean()
                avg_accuracy = correct[in_bin].mean()
                gap = abs(avg_confidence - avg_accuracy)

                overall_bins.append({
                    "bin": bin_idx,
                    "confidence": float(avg_confidence),
                    "accuracy": float(avg_accuracy),
                    "gap": float(gap),
                    "count": int(bin_size),
                    "overconfident": avg_confidence > avg_accuracy,
                })

                if gap > results["mce"]:
                    results["mce"] = gap

        results["bins"] = overall_bins

        return results

    def fit_temperature_scaling(
        self,
        probs: np.ndarray,
        actuals: np.ndarray
    ) -> float:
        """Find optimal temperature for scaling."""
        best_temp = 1.0
        best_ece = float('inf')

        for temp in np.arange(0.5, 2.5, 0.1):
            # Apply temperature scaling
            scaled = self._apply_temperature(probs, temp)

            # Calculate ECE
            cal_result = self.calculate_calibration_error(scaled, actuals)
            ece = cal_result.get("ece", float('inf'))

            if ece < best_ece:
                best_ece = ece
                best_temp = temp

        return best_temp

    def _apply_temperature(self, probs: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature scaling to probabilities."""
        # Convert to logits, scale, convert back
        eps = 1e-10
        logits = np.log(probs + eps)
        scaled_logits = logits / temperature
        scaled_probs = np.exp(scaled_logits)
        scaled_probs = scaled_probs / scaled_probs.sum(axis=1, keepdims=True)
        return scaled_probs

    def fit_platt_scaling(
        self,
        probs: np.ndarray,
        actuals: np.ndarray
    ) -> Dict:
        """Fit Platt scaling for each class."""
        if not SKLEARN_AVAILABLE:
            return {}

        scalers = {}
        n_classes = probs.shape[1]

        for c in range(n_classes):
            class_name = {0: "H", 1: "D", 2: "A"}[c]
            binary_labels = (actuals == c).astype(int)

            lr = LogisticRegression(solver='lbfgs', max_iter=1000)
            lr.fit(probs[:, c].reshape(-1, 1), binary_labels)

            scalers[class_name] = {
                "coef": float(lr.coef_[0][0]),
                "intercept": float(lr.intercept_[0]),
            }

        return scalers

    def analyze_model(self, model_name: str, test_df: pd.DataFrame) -> Dict:
        """Full calibration analysis for a model."""
        log.info(f"Analyzing {model_name} model calibration...")

        probs, actuals = self.get_model_predictions(model_name, test_df)

        if len(probs) == 0:
            return {"error": f"no_predictions_for_{model_name}"}

        # Basic stats
        accuracy = (probs.argmax(axis=1) == actuals).mean()
        max_confidence = probs.max(axis=1).mean()

        # Calibration error
        cal_result = self.calculate_calibration_error(probs, actuals)

        # Find optimal temperature
        opt_temp = self.fit_temperature_scaling(probs, actuals)

        # Calibration after temperature scaling
        scaled_probs = self._apply_temperature(probs, opt_temp)
        cal_after = self.calculate_calibration_error(scaled_probs, actuals)

        # Fit Platt scaling
        platt_params = self.fit_platt_scaling(probs, actuals)

        result = {
            "model": model_name,
            "n_predictions": len(actuals),
            "accuracy": float(accuracy),
            "avg_max_confidence": float(max_confidence),
            "calibration_before": cal_result,
            "optimal_temperature": float(opt_temp),
            "calibration_after_temp": cal_after,
            "platt_scaling_params": platt_params,
            "is_overconfident": max_confidence > accuracy,
            "confidence_gap": float(max_confidence - accuracy),
        }

        self.temperatures[model_name] = opt_temp
        self.platt_scalers[model_name] = platt_params

        return result

    def run_full_analysis(self, test_seasons: List[str] = None) -> Dict:
        """Run calibration analysis on all models."""
        if test_seasons is None:
            test_seasons = ["2023-2024", "2024-2025"]

        test_df = self.df[self.df["season"].isin(test_seasons)].copy()
        log.info(f"Analyzing calibration on {len(test_df)} matches")

        results = {
            "timestamp": datetime.now().isoformat(),
            "test_seasons": test_seasons,
            "n_matches": len(test_df),
            "models": {},
        }

        for model_name in ["xg", "ml", "player_xg", "deep"]:
            model_result = self.analyze_model(model_name, test_df)
            results["models"][model_name] = model_result

        # Summary
        results["summary"] = {
            "optimal_temperatures": self.temperatures,
            "recommendations": self._get_recommendations(results),
        }

        return results

    def _get_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on analysis."""
        recs = []

        for model_name, model_result in results.get("models", {}).items():
            if "error" in model_result:
                continue

            ece_before = model_result["calibration_before"].get("ece", 0)
            ece_after = model_result["calibration_after_temp"].get("ece", 0)
            temp = model_result["optimal_temperature"]

            if model_result["is_overconfident"]:
                gap = model_result["confidence_gap"]
                recs.append(f"{model_name}: Overconfident by {gap:.1%}")

            if temp != 1.0:
                improvement = ece_before - ece_after
                if improvement > 0.01:
                    recs.append(f"{model_name}: Apply temperature scaling T={temp:.2f} (ECE improvement: {improvement:.4f})")

        return recs

    def save_calibration_params(self, output_dir: Path = None):
        """Save calibration parameters for use in ensemble."""
        if output_dir is None:
            output_dir = MODELS_DIR / "calibration"

        output_dir.mkdir(parents=True, exist_ok=True)

        params = {
            "temperatures": self.temperatures,
            "platt_scalers": self.platt_scalers,
            "timestamp": datetime.now().isoformat(),
        }

        params_file = output_dir / "calibration_params.json"
        with open(params_file, "w") as f:
            json.dump(params, f, indent=2)

        log.info(f"Saved calibration parameters to {params_file}")
        return params_file


def main():
    parser = argparse.ArgumentParser(description="Analyze model calibration")
    parser.add_argument("--fix", action="store_true", help="Save calibration fixes")
    parser.add_argument("--seasons", nargs="+", default=["2023-2024", "2024-2025"])
    args = parser.parse_args()

    analyzer = CalibrationAnalyzer()
    if not analyzer.load_data():
        log.error("Failed to load data")
        return

    results = analyzer.run_full_analysis(args.seasons)

    # Print results
    print("\n" + "=" * 70)
    print("CALIBRATION ANALYSIS RESULTS")
    print("=" * 70)

    for model_name, model_result in results["models"].items():
        if "error" in model_result:
            print(f"\n{model_name.upper()}: Error - {model_result['error']}")
            continue

        print(f"\n{model_name.upper()} MODEL")
        print("-" * 40)
        print(f"  Accuracy: {model_result['accuracy']:.1%}")
        print(f"  Avg Max Confidence: {model_result['avg_max_confidence']:.1%}")
        print(f"  Overconfident: {model_result['is_overconfident']} (gap: {model_result['confidence_gap']:+.1%})")
        print(f"  ECE (before): {model_result['calibration_before']['ece']:.4f}")
        print(f"  Optimal Temperature: {model_result['optimal_temperature']:.2f}")
        print(f"  ECE (after temp): {model_result['calibration_after_temp']['ece']:.4f}")

        # Per-class calibration
        print("  Per-class ECE:")
        for cls, cls_data in model_result['calibration_before']['per_class'].items():
            print(f"    {cls}: {cls_data['ece']:.4f}")

    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    for rec in results["summary"]["recommendations"]:
        print(f"  - {rec}")

    # Save results
    results_dir = DATA_DIR / "optimization"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"calibration_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    # Save calibration params if --fix
    if args.fix:
        params_file = analyzer.save_calibration_params()
        print(f"Calibration parameters saved to: {params_file}")


if __name__ == "__main__":
    main()
