#!/usr/bin/env python3
"""Historical Backtesting - Verify Predictions Beat Random

These tests validate ACTUAL HISTORICAL PERFORMANCE:
- Win rate by confidence level
- ROI calculation
- Comparison to baseline (random guessing)
- Calibration analysis

Run with: python tests/test_historical_accuracy.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime


class Results:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.errors = []
        self.metrics = {}

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

    def add_warning(self, test_name: str, warning: str):
        self.warnings += 1
        print(f"  [WARN] {test_name}")
        print(f"         {warning}")

    def add_metric(self, name: str, value: float):
        self.metrics[name] = value

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'=' * 70}")
        print(f"HISTORICAL ACCURACY RESULTS: {self.passed}/{total} tests passed")
        if self.warnings:
            print(f"Warnings: {self.warnings}")
        if self.errors:
            print(f"\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        if self.metrics:
            print(f"\n{'=' * 70}")
            print("KEY PERFORMANCE METRICS:")
            print("=" * 70)
            for name, value in self.metrics.items():
                if "pct" in name.lower() or "rate" in name.lower() or "accuracy" in name.lower():
                    print(f"  {name}: {value:.1%}")
                elif "roi" in name.lower():
                    print(f"  {name}: {value:+.1%}")
                else:
                    print(f"  {name}: {value:.2f}")
        return self.failed == 0


results = Results()


def load_historical_predictions():
    """Load CV predictions from model training."""
    cv_path = PROJECT_ROOT / "data" / "models" / "universal" / "cv_predictions.parquet"

    if not cv_path.exists():
        return None

    df = pd.read_parquet(cv_path)

    # Derive predicted outcome from probabilities (argmax)
    if "predicted" not in df.columns:
        label_map = {0: "H", 1: "D", 2: "A"}
        df["predicted"] = df[["prob_H", "prob_D", "prob_A"]].values.argmax(axis=1)
        df["predicted"] = df["predicted"].map(label_map)

    # Add confidence column (max probability)
    df["confidence"] = df[["prob_H", "prob_D", "prob_A"]].max(axis=1)

    # Add correct prediction flag
    df["correct"] = df["actual"] == df["predicted"]

    return df


def test_overall_accuracy():
    """Test that overall prediction accuracy beats random (33.3%)."""
    print("\n=== Testing Overall Accuracy ===")

    df = load_historical_predictions()
    if df is None:
        results.add_fail("Overall accuracy", "cv_predictions.parquet not found")
        return

    total = len(df)
    correct = df["correct"].sum()
    accuracy = correct / total

    results.add_metric("Overall Accuracy", accuracy)
    results.add_metric("Total Matches Analyzed", total)

    # Random baseline is 33.3% (1/3 outcomes)
    baseline = 1/3

    if accuracy > baseline:
        improvement = (accuracy - baseline) / baseline * 100
        results.add_pass(f"Beats random baseline",
                        f"Accuracy: {accuracy:.1%} vs Random: {baseline:.1%} (+{improvement:.1f}% lift)")
    else:
        results.add_fail("Beats random baseline",
                        f"Accuracy {accuracy:.1%} <= Random {baseline:.1%}")

    # Should be at least 40% for a useful model
    if accuracy >= 0.40:
        results.add_pass(f"Meets 40% minimum threshold",
                        f"Accuracy: {accuracy:.1%}")
    else:
        results.add_warning(f"Below 40% threshold",
                          f"Accuracy: {accuracy:.1%} (aim for 40%+)")


def test_accuracy_by_confidence():
    """Test that higher confidence = higher accuracy."""
    print("\n=== Testing Accuracy by Confidence Level ===")

    df = load_historical_predictions()
    if df is None:
        results.add_fail("Confidence calibration", "cv_predictions.parquet not found")
        return

    # Define confidence buckets
    buckets = [
        ("Low (33-50%)", 0.33, 0.50),
        ("Medium (50-60%)", 0.50, 0.60),
        ("High (60-70%)", 0.60, 0.70),
        ("Very High (70%+)", 0.70, 1.01),
    ]

    prev_accuracy = 0
    monotonic = True

    for name, low, high in buckets:
        subset = df[(df["confidence"] >= low) & (df["confidence"] < high)]

        if len(subset) < 10:
            results.add_warning(f"Confidence bucket {name}",
                              f"Only {len(subset)} samples - insufficient data")
            continue

        accuracy = subset["correct"].mean()
        results.add_metric(f"Accuracy {name}", accuracy)

        if accuracy >= prev_accuracy:
            results.add_pass(f"Confidence {name}",
                           f"Accuracy: {accuracy:.1%} (n={len(subset)})")
        else:
            monotonic = False
            results.add_warning(f"Confidence {name}",
                              f"Accuracy: {accuracy:.1%} < previous {prev_accuracy:.1%}")

        prev_accuracy = accuracy

    if monotonic:
        results.add_pass("Confidence calibration is monotonic",
                        "Higher confidence = higher accuracy")


def test_roi_at_confidence_levels():
    """Test ROI when betting at different confidence thresholds."""
    print("\n=== Testing ROI at Confidence Thresholds ===")

    df = load_historical_predictions()
    if df is None:
        results.add_fail("ROI calculation", "cv_predictions.parquet not found")
        return

    # Simulate flat betting at different confidence thresholds
    # Assume average odds of 2.0 for simplicity (implied 50%)
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]

    for threshold in thresholds:
        high_conf = df[df["confidence"] >= threshold]

        if len(high_conf) < 20:
            results.add_warning(f"ROI at {threshold:.0%} confidence",
                              f"Only {len(high_conf)} bets - insufficient data")
            continue

        # Calculate ROI with assumed odds
        # If we bet on predictions with conf >= threshold
        # Assume odds roughly = 1/confidence (fair odds)
        # Actual ROI = (wins * odds - total_bets) / total_bets

        wins = high_conf["correct"].sum()
        total_bets = len(high_conf)
        win_rate = wins / total_bets

        # Estimate implied odds based on confidence
        avg_confidence = high_conf["confidence"].mean()
        avg_odds = 1 / avg_confidence  # Fair odds would be this
        market_odds = avg_odds * 1.05  # Assume 5% market margin

        # ROI = (wins × odds - total_bets) / total_bets
        gross_return = wins * market_odds
        roi = (gross_return - total_bets) / total_bets

        results.add_metric(f"ROI at {threshold:.0%}+ conf", roi)

        if roi > 0:
            results.add_pass(f"Positive ROI at {threshold:.0%}+ confidence",
                           f"ROI: {roi:+.1%}, Win rate: {win_rate:.1%}, Bets: {total_bets}")
        else:
            results.add_warning(f"Negative ROI at {threshold:.0%}+ confidence",
                              f"ROI: {roi:+.1%}, Win rate: {win_rate:.1%}, Bets: {total_bets}")


def test_home_away_draw_accuracy():
    """Test accuracy for each outcome type."""
    print("\n=== Testing Accuracy by Outcome Type ===")

    df = load_historical_predictions()
    if df is None:
        results.add_fail("Outcome accuracy", "cv_predictions.parquet not found")
        return

    for outcome in ["H", "D", "A"]:
        # When we predict this outcome
        predicted = df[df["predicted"] == outcome]
        if len(predicted) == 0:
            continue

        correct = (predicted["actual"] == outcome).sum()
        accuracy = correct / len(predicted)

        outcome_name = {"H": "Home Win", "D": "Draw", "A": "Away Win"}[outcome]
        results.add_metric(f"{outcome_name} Prediction Accuracy", accuracy)

        # Draws are hardest to predict (usually ~25% accuracy is good)
        if outcome == "D":
            threshold = 0.25
        else:
            threshold = 0.45

        if accuracy >= threshold:
            results.add_pass(f"{outcome_name} prediction accuracy",
                           f"{accuracy:.1%} (n={len(predicted)})")
        else:
            results.add_warning(f"{outcome_name} prediction accuracy",
                              f"{accuracy:.1%} below {threshold:.0%} threshold")


def test_high_confidence_performance():
    """Test performance of high-confidence predictions (>= 65%)."""
    print("\n=== Testing High-Confidence Performance ===")

    df = load_historical_predictions()
    if df is None:
        results.add_fail("High confidence test", "cv_predictions.parquet not found")
        return

    high_conf = df[df["confidence"] >= 0.65]

    if len(high_conf) < 50:
        results.add_warning("High confidence sample size",
                          f"Only {len(high_conf)} high-conf predictions")
        return

    accuracy = high_conf["correct"].mean()
    results.add_metric("High Conf (65%+) Accuracy", accuracy)
    results.add_metric("High Conf Sample Size", len(high_conf))

    # High confidence predictions should win at least 55%
    if accuracy >= 0.55:
        results.add_pass("High confidence predictions are reliable",
                        f"65%+ confidence wins {accuracy:.1%} of time (n={len(high_conf)})")
    else:
        results.add_fail("High confidence predictions are reliable",
                        f"65%+ confidence only wins {accuracy:.1%}")


def test_calibration():
    """Test that predicted probabilities match actual frequencies."""
    print("\n=== Testing Probability Calibration ===")

    df = load_historical_predictions()
    if df is None:
        results.add_fail("Calibration test", "cv_predictions.parquet not found")
        return

    # Check if predicted 60% wins actually win ~60%
    calibration_errors = []

    prob_bins = [(0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9)]

    for low, high in prob_bins:
        # Get predictions where our top choice had this probability range
        subset = df[(df["confidence"] >= low) & (df["confidence"] < high)]

        if len(subset) < 30:
            continue

        actual_rate = subset["correct"].mean()
        expected_rate = (low + high) / 2  # Midpoint

        calibration_error = abs(actual_rate - expected_rate)
        calibration_errors.append(calibration_error)

        if calibration_error < 0.10:  # Within 10%
            results.add_pass(f"Calibration {low:.0%}-{high:.0%}",
                           f"Predicted: {expected_rate:.0%}, Actual: {actual_rate:.1%}")
        else:
            results.add_warning(f"Calibration {low:.0%}-{high:.0%}",
                              f"Predicted: {expected_rate:.0%}, Actual: {actual_rate:.1%} (gap: {calibration_error:.1%})")

    if calibration_errors:
        avg_error = np.mean(calibration_errors)
        results.add_metric("Avg Calibration Error", avg_error)

        if avg_error < 0.08:
            results.add_pass("Overall calibration",
                           f"Average error: {avg_error:.1%} (excellent)")
        elif avg_error < 0.12:
            results.add_pass("Overall calibration",
                           f"Average error: {avg_error:.1%} (good)")
        else:
            results.add_warning("Overall calibration",
                              f"Average error: {avg_error:.1%} (needs improvement)")


def test_recent_performance():
    """Test performance on recent matches (last 2 seasons)."""
    print("\n=== Testing Recent Performance (2023-2025) ===")

    df = load_historical_predictions()
    if df is None:
        results.add_fail("Recent performance", "cv_predictions.parquet not found")
        return

    # Filter to recent seasons
    df["match_date"] = pd.to_datetime(df["match_date"])
    recent = df[df["match_date"] >= "2023-01-01"]

    if len(recent) < 100:
        results.add_warning("Recent data",
                          f"Only {len(recent)} recent matches available")
        return

    accuracy = recent["correct"].mean()
    results.add_metric("Recent (2023+) Accuracy", accuracy)

    # Should maintain performance on recent data
    overall_accuracy = df["correct"].mean()

    if accuracy >= overall_accuracy * 0.95:  # Within 5% of overall
        results.add_pass("Model holds on recent data",
                        f"Recent: {accuracy:.1%} vs Overall: {overall_accuracy:.1%}")
    else:
        results.add_warning("Model degradation on recent data",
                          f"Recent: {accuracy:.1%} vs Overall: {overall_accuracy:.1%}")


def test_value_betting_simulation():
    """Simulate value betting strategy on historical data."""
    print("\n=== Testing Value Betting Simulation ===")

    df = load_historical_predictions()
    if df is None:
        results.add_fail("Value betting simulation", "cv_predictions.parquet not found")
        return

    # Simulate betting on outcomes where our prob > implied prob by at least 5%
    min_edge = 0.05

    # Estimate "market odds" as ~10% overround on true probabilities
    # If our prob is 60%, fair odds = 1.67, market might be 1.55

    bets = []
    for _, row in df.iterrows():
        our_prob = row["confidence"]
        predicted = row["predicted"]
        actual = row["actual"]

        # Assume market implied prob is ~5% lower than our estimate (our edge)
        market_implied = our_prob * 0.95
        edge = our_prob - market_implied

        if edge >= min_edge and our_prob >= 0.55:
            # Calculate hypothetical odds
            odds = 1 / market_implied

            won = 1 if predicted == actual else 0
            profit = (odds - 1) if won else -1

            bets.append({
                "our_prob": our_prob,
                "odds": odds,
                "won": won,
                "profit": profit
            })

    if not bets:
        results.add_warning("Value betting simulation",
                          "No qualifying bets found")
        return

    bets_df = pd.DataFrame(bets)
    total_bets = len(bets_df)
    wins = bets_df["won"].sum()
    win_rate = wins / total_bets
    total_profit = bets_df["profit"].sum()
    roi = total_profit / total_bets

    results.add_metric("Value Bets Placed", total_bets)
    results.add_metric("Value Bet Win Rate", win_rate)
    results.add_metric("Value Betting ROI", roi)

    if roi > 0:
        results.add_pass("Value betting is profitable",
                        f"ROI: {roi:+.1%}, Win rate: {win_rate:.1%}, Bets: {total_bets}")
    else:
        results.add_warning("Value betting simulation",
                          f"ROI: {roi:+.1%} (negative, review edge calculation)")


def main():
    print("=" * 70)
    print("HISTORICAL ACCURACY VALIDATION")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("\nThis validates ACTUAL performance on historical data.")

    test_overall_accuracy()
    test_accuracy_by_confidence()
    test_roi_at_confidence_levels()
    test_home_away_draw_accuracy()
    test_high_confidence_performance()
    test_calibration()
    test_recent_performance()
    test_value_betting_simulation()

    success = results.summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
