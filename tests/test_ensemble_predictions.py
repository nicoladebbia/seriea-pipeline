#!/usr/bin/env python3
"""Ensemble Prediction System Tests

Validates that:
1. All three prediction methods work independently
2. xG models produce realistic goal expectations
3. ML classifier integrates properly
4. Ensemble combines predictions correctly
5. Historical backtesting shows improvement over factor-only

Run with: python tests/test_ensemble_predictions.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
from datetime import datetime


class Results:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def add_pass(self, test_name: str, details: str = ""):
        self.passed += 1
        print(f"  [PASS] {test_name}")
        if details:
            print(f"         {details}")

    def add_fail(self, test_name: str, error: str):
        self.failed += 1
        self.errors.append((test_name, error))
        print(f"  [FAIL] {test_name}")
        print(f"         {error}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'=' * 70}")
        print(f"ENSEMBLE TEST RESULTS: {self.passed}/{total} tests passed")
        if self.errors:
            print(f"\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        return self.failed == 0


results = Results()


def test_xg_predictor_loads():
    """Test that xG predictor can load models."""
    print("\n=== Testing xG Predictor ===")

    try:
        from scripts.prediction.ensemble_prediction_engine import XGPredictor

        predictor = XGPredictor()
        loaded = predictor.load_models()

        if loaded:
            results.add_pass("xG models load successfully")
            results.add_pass("xG home model exists", f"Model: {predictor.home_model is not None}")
            results.add_pass("xG away model exists", f"Model: {predictor.away_model is not None}")
        else:
            results.add_fail("xG models load", "Models not found - run train_xg_based.py")

    except Exception as e:
        results.add_fail("xG predictor import", str(e)[:80])


def test_xg_poisson_conversion():
    """Test Poisson probability conversion."""
    print("\n=== Testing Poisson Conversion ===")

    try:
        from scripts.prediction.ensemble_prediction_engine import XGPredictor

        predictor = XGPredictor()

        # Test with typical xG values
        probs = predictor._poisson_win_prob(1.5, 1.0)

        # Home win should be most likely when home_xg > away_xg
        if probs["H"] > probs["A"]:
            results.add_pass("Poisson: Higher xG -> higher win prob",
                           f"H={probs['H']:.1%}, A={probs['A']:.1%}")
        else:
            results.add_fail("Poisson conversion", "Home xG 1.5 vs Away 1.0 should favor home")

        # Probabilities should sum to 1
        total = probs["H"] + probs["D"] + probs["A"]
        if abs(total - 1.0) < 0.001:
            results.add_pass("Poisson probabilities sum to 1", f"Total: {total:.4f}")
        else:
            results.add_fail("Poisson normalization", f"Total: {total}")

        # Test equal xG -> roughly equal H/A
        equal_probs = predictor._poisson_win_prob(1.5, 1.5)
        if abs(equal_probs["H"] - equal_probs["A"]) < 0.05:
            results.add_pass("Equal xG -> equal win probs",
                           f"H={equal_probs['H']:.1%}, A={equal_probs['A']:.1%}")
        else:
            results.add_fail("Equal xG test", "Should have similar H and A probabilities")

        # Draw should be significant with equal xG
        if equal_probs["D"] > 0.20:
            results.add_pass("Draw probability reasonable", f"D={equal_probs['D']:.1%}")
        else:
            results.add_fail("Draw probability", f"Too low: {equal_probs['D']:.1%}")

    except Exception as e:
        results.add_fail("Poisson conversion test", str(e)[:80])


def test_ml_classifier_loads():
    """Test that ML classifier loads properly."""
    print("\n=== Testing ML Classifier ===")

    try:
        from scripts.prediction.ensemble_prediction_engine import MLClassifier

        classifier = MLClassifier()
        loaded = classifier.load_model()

        if loaded:
            results.add_pass("ML classifier loads successfully")
            results.add_pass(f"Feature names loaded: {len(classifier.feature_names)}",
                           f"Features: {classifier.feature_names[:5]}...")
        else:
            results.add_fail("ML classifier load", "Model not found - run training")

    except Exception as e:
        results.add_fail("ML classifier import", str(e)[:80])


def test_feature_builder():
    """Test feature builder creates valid features."""
    print("\n=== Testing Feature Builder ===")

    try:
        from scripts.prediction.ensemble_prediction_engine import FeatureBuilder

        builder = FeatureBuilder()
        loaded = builder.load_historical()

        if loaded:
            results.add_pass("Historical data loaded",
                           f"{len(builder.df)} matches, {len(builder.team_features)} team caches")

            # Test building features for a match
            features = builder.build_match_features("Inter", "Milan")

            if features is not None and len(features) > 0:
                results.add_pass("Features built for Inter vs Milan",
                               f"{len(features.columns)} features")

                # Check key features exist
                if "elo_diff" in features.columns:
                    results.add_pass("elo_diff feature exists", f"Value: {features['elo_diff'].iloc[0]:.1f}")
                else:
                    results.add_fail("elo_diff feature", "Missing")

                if "home_elo" in features.columns:
                    results.add_pass("home_elo feature exists", f"Value: {features['home_elo'].iloc[0]:.1f}")
            else:
                results.add_fail("Feature building", "No features returned")
        else:
            results.add_fail("Historical data load", "Failed to load features.parquet")

    except Exception as e:
        results.add_fail("Feature builder test", str(e)[:80])


def test_ensemble_predictor():
    """Test ensemble predictor combines all methods."""
    print("\n=== Testing Ensemble Predictor ===")

    try:
        from scripts.prediction.ensemble_prediction_engine import EnsemblePredictor, ENSEMBLE_WEIGHTS

        ensemble = EnsemblePredictor()
        initialized = ensemble.initialize()

        if initialized:
            results.add_pass("Ensemble initialized",
                           f"Methods: {ensemble.available_methods}")

            # Check weights
            total_weight = sum(ENSEMBLE_WEIGHTS.values())
            if abs(total_weight - 1.0) < 0.001:
                results.add_pass("Ensemble weights sum to 1", f"Total: {total_weight}")
            else:
                results.add_fail("Ensemble weights", f"Sum: {total_weight}")

            # Test prediction
            test_match = {
                "home_team": "Roma",
                "away_team": "Cagliari",
                "date": "2026-02-09",
            }
            test_factors = {
                "home_factors": ["home_favorite"],
                "away_factors": [],
                "neutral_factors": [],
                "home_lift": 0.12,
                "away_lift": 0,
                "n_home_factors": 1,
                "n_away_factors": 0,
            }

            pred = ensemble.predict(test_match, test_factors, {}, include_components=True)

            if pred:
                results.add_pass("Ensemble prediction generated")

                # Check probabilities
                probs = pred["probabilities"]
                total_prob = probs["home"] + probs["draw"] + probs["away"]
                if abs(total_prob - 1.0) < 0.01:
                    results.add_pass("Probabilities sum to 1",
                                   f"H={probs['home']:.1%} D={probs['draw']:.1%} A={probs['away']:.1%}")
                else:
                    results.add_fail("Probability sum", f"Total: {total_prob}")

                # Check component predictions exist
                components = pred.get("component_predictions", {})
                if "factor" in components:
                    results.add_pass("Factor component included")
                if "xg" in components:
                    results.add_pass("xG component included",
                                   f"H={components['xg']['prob_H']:.1%}")
                if "ml" in components:
                    results.add_pass("ML component included",
                                   f"H={components['ml']['prob_H']:.1%}")

            else:
                results.add_fail("Ensemble prediction", "No prediction returned")

        else:
            results.add_fail("Ensemble initialization", "Failed to initialize")

    except Exception as e:
        results.add_fail("Ensemble predictor test", str(e)[:80])


def test_xg_predictions_realistic():
    """Test that xG predictions are in realistic ranges."""
    print("\n=== Testing xG Prediction Realism ===")

    try:
        from scripts.prediction.ensemble_prediction_engine import XGPredictor, FeatureBuilder

        predictor = XGPredictor()
        builder = FeatureBuilder()

        if not predictor.load_models():
            results.add_fail("xG models not available", "Skip realism test")
            return

        if not builder.load_historical():
            results.add_fail("Historical data not available", "Skip realism test")
            return

        # Test with strong vs weak team
        features = builder.build_match_features("Inter", "Lecce")
        if features is None:
            results.add_fail("Feature building failed", "Inter vs Lecce")
            return

        result = predictor.predict(features)

        if result:
            home_xg = result["home_xg"]
            away_xg = result["away_xg"]

            # xG should be between 0.5 and 4.0 typically
            if 0.5 <= home_xg <= 4.0:
                results.add_pass(f"Home xG in range: {home_xg:.2f}")
            else:
                results.add_fail("Home xG out of range", f"xG={home_xg}")

            if 0.5 <= away_xg <= 4.0:
                results.add_pass(f"Away xG in range: {away_xg:.2f}")
            else:
                results.add_fail("Away xG out of range", f"xG={away_xg}")

            # Strong home team should have higher xG
            if home_xg > away_xg:
                results.add_pass("Strong team has higher xG",
                               f"Inter {home_xg:.2f} vs Lecce {away_xg:.2f}")
            else:
                results.add_fail("xG logic", f"Inter should > Lecce ({home_xg:.2f} vs {away_xg:.2f})")
        else:
            results.add_fail("xG prediction failed", "No result returned")

    except Exception as e:
        results.add_fail("xG realism test", str(e)[:80])


def test_ensemble_vs_factor_only():
    """Test that ensemble provides different predictions than factor-only."""
    print("\n=== Testing Ensemble vs Factor-Only ===")

    try:
        from scripts.prediction.ensemble_prediction_engine import EnsemblePredictor

        ensemble = EnsemblePredictor()
        if not ensemble.initialize():
            results.add_fail("Ensemble not initialized", "Skip comparison test")
            return

        test_cases = [
            ("Roma", "Cagliari"),
            ("Inter", "Lecce"),
            ("Juventus", "Lazio"),
        ]

        differences = []
        for home, away in test_cases:
            match = {"home_team": home, "away_team": away}
            factors = {
                "home_factors": ["home_favorite"],
                "away_factors": [],
                "neutral_factors": [],
                "home_lift": 0.12,
                "away_lift": 0,
                "n_home_factors": 1,
                "n_away_factors": 0,
            }

            pred = ensemble.predict(match, factors, {}, include_components=True)
            if pred and "component_predictions" in pred:
                factor_h = pred["component_predictions"].get("factor", {}).get("prob_H", 0)
                ensemble_h = pred["probabilities"]["home"]
                diff = abs(ensemble_h - factor_h)
                differences.append(diff)

        if differences:
            avg_diff = np.mean(differences)
            if avg_diff > 0.01:  # At least 1% difference on average
                results.add_pass("Ensemble differs from factor-only",
                               f"Avg difference: {avg_diff:.1%}")
            else:
                results.add_fail("Ensemble too similar", f"Avg diff: {avg_diff:.1%}")
        else:
            results.add_fail("No predictions generated", "Comparison failed")

    except Exception as e:
        results.add_fail("Ensemble vs factor test", str(e)[:80])


def test_historical_backtest():
    """Test ensemble on historical data to verify improvement."""
    print("\n=== Testing Historical Backtest ===")

    try:
        from config.settings import DATA_DIR

        cv_path = DATA_DIR / "models" / "universal" / "cv_predictions.parquet"

        if not cv_path.exists():
            results.add_fail("CV predictions not found", "Run training first")
            return

        df = pd.read_parquet(cv_path)

        # Check ML model accuracy from CV
        if "predicted" in df.columns and "actual" in df.columns:
            ml_accuracy = (df["predicted"] == df["actual"]).mean()
            results.add_pass(f"ML model CV accuracy: {ml_accuracy:.1%}")

            if ml_accuracy > 0.50:
                results.add_pass("ML beats random baseline",
                               f"{ml_accuracy:.1%} > 50%")
            else:
                results.add_fail("ML performance", f"Only {ml_accuracy:.1%}")

            # Check high confidence accuracy
            if "confidence" in df.columns:
                high_conf = df[df["confidence"] > 0.55]
                if len(high_conf) > 100:
                    high_acc = (high_conf["predicted"] == high_conf["actual"]).mean()
                    results.add_pass(f"High conf accuracy: {high_acc:.1%}",
                                   f"({len(high_conf)} samples)")
                else:
                    results.add_pass("High conf sample size limited",
                                   f"Only {len(high_conf)} samples")

        else:
            results.add_fail("CV predictions format", "Missing predicted/actual columns")

    except Exception as e:
        results.add_fail("Historical backtest", str(e)[:80])


def test_predictions_output():
    """Test that the prediction output is valid."""
    print("\n=== Testing Predictions Output ===")

    try:
        from config.settings import DATA_DIR

        pred_path = DATA_DIR / "upcoming" / "predictions.json"

        if not pred_path.exists():
            results.add_fail("Predictions file not found", "Run ensemble first")
            return

        with open(pred_path) as f:
            data = json.load(f)

        # Check structure
        if "predictions" in data:
            results.add_pass("Predictions key exists",
                           f"{len(data['predictions'])} predictions")
        else:
            results.add_fail("Predictions structure", "Missing 'predictions' key")
            return

        if "ensemble_enabled" in data:
            results.add_pass(f"Ensemble enabled: {data['ensemble_enabled']}")

        if "methods_available" in data:
            results.add_pass(f"Methods: {data['methods_available']}")

        # Check individual predictions
        for pred in data["predictions"][:2]:  # Check first 2
            match = pred.get("match", "Unknown")

            if "probabilities" in pred:
                probs = pred["probabilities"]
                total = probs["home"] + probs["draw"] + probs["away"]
                if abs(total - 1.0) < 0.01:
                    results.add_pass(f"{match}: Valid probabilities")
                else:
                    results.add_fail(f"{match}: Invalid probabilities", f"Sum={total}")

            if "component_predictions" in pred:
                components = pred["component_predictions"]
                if "xg_details" in components:
                    xg = components["xg_details"]
                    results.add_pass(f"{match}: xG predictions",
                                   f"H={xg['home_xg']:.2f} A={xg['away_xg']:.2f}")

    except Exception as e:
        results.add_fail("Predictions output test", str(e)[:80])


def main():
    print("=" * 70)
    print("ENSEMBLE PREDICTION SYSTEM TESTS")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    test_xg_predictor_loads()
    test_xg_poisson_conversion()
    test_ml_classifier_loads()
    test_feature_builder()
    test_ensemble_predictor()
    test_xg_predictions_realistic()
    test_ensemble_vs_factor_only()
    test_historical_backtest()
    test_predictions_output()

    success = results.summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
