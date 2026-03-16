#!/usr/bin/env python3
"""Edge Case Tests - Validate Error Handling and Graceful Degradation

These tests verify the system handles:
1. Missing data gracefully
2. Invalid inputs safely
3. API failures with recovery
4. Extreme values correctly

Run with: python tests/test_edge_cases.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
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
        print(f"EDGE CASE RESULTS: {self.passed}/{total} tests passed")
        if self.errors:
            print(f"\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        return self.failed == 0


results = Results()


def test_validate_match_data():
    """Test match data validation."""
    print("\n=== Testing Match Data Validation ===")

    from scripts.utils.error_handling import validate_match_data

    # Valid match
    valid_match = {
        "home_team": "Inter",
        "away_team": "Milan",
        "date": "2026-02-10"
    }
    is_valid, errors = validate_match_data(valid_match)

    if is_valid:
        results.add_pass("Valid match accepted", f"Inter vs Milan")
    else:
        results.add_fail("Valid match accepted", f"Errors: {errors}")

    # Missing home team
    missing_home = {"away_team": "Milan"}
    is_valid, errors = validate_match_data(missing_home)

    if not is_valid and "home_team" in str(errors):
        results.add_pass("Missing home team rejected", f"Errors: {errors}")
    else:
        results.add_fail("Missing home team rejected", f"Should have errored")

    # Invalid team name
    invalid_team = {"home_team": "FakeTeam", "away_team": "Milan"}
    is_valid, errors = validate_match_data(invalid_team)

    if not is_valid and "FakeTeam" in str(errors):
        results.add_pass("Invalid team name rejected", f"Errors: {errors}")
    else:
        results.add_fail("Invalid team name rejected", f"Should have errored")

    # Same team twice
    same_team = {"home_team": "Inter", "away_team": "Inter"}
    is_valid, errors = validate_match_data(same_team)

    if not is_valid and "same" in str(errors).lower():
        results.add_pass("Same team twice rejected", f"Errors: {errors}")
    else:
        results.add_fail("Same team twice rejected", f"Should have errored")

    # Invalid date format
    bad_date = {"home_team": "Inter", "away_team": "Milan", "date": "02/10/2026"}
    is_valid, errors = validate_match_data(bad_date)

    if not is_valid and "date" in str(errors).lower():
        results.add_pass("Invalid date format rejected", f"Errors: {errors}")
    else:
        results.add_fail("Invalid date format rejected", f"Should have errored")


def test_validate_probabilities():
    """Test probability validation."""
    print("\n=== Testing Probability Validation ===")

    from scripts.utils.error_handling import validate_probability

    # Valid probability
    is_valid, error = validate_probability(0.55)
    if is_valid:
        results.add_pass("Valid probability (0.55) accepted")
    else:
        results.add_fail("Valid probability accepted", error)

    # Edge cases: 0 and 1
    is_valid, _ = validate_probability(0.0)
    if is_valid:
        results.add_pass("Probability 0.0 accepted")
    else:
        results.add_fail("Probability 0.0 accepted", "Should be valid")

    is_valid, _ = validate_probability(1.0)
    if is_valid:
        results.add_pass("Probability 1.0 accepted")
    else:
        results.add_fail("Probability 1.0 accepted", "Should be valid")

    # Invalid: negative
    is_valid, error = validate_probability(-0.1)
    if not is_valid:
        results.add_pass("Negative probability rejected", error)
    else:
        results.add_fail("Negative probability rejected", "Should have errored")

    # Invalid: > 1
    is_valid, error = validate_probability(1.5)
    if not is_valid:
        results.add_pass("Probability > 1 rejected", error)
    else:
        results.add_fail("Probability > 1 rejected", "Should have errored")

    # Invalid: non-numeric
    is_valid, error = validate_probability("fifty")
    if not is_valid:
        results.add_pass("Non-numeric probability rejected", error)
    else:
        results.add_fail("Non-numeric probability rejected", "Should have errored")


def test_validate_odds():
    """Test odds validation."""
    print("\n=== Testing Odds Validation ===")

    from scripts.utils.error_handling import validate_odds

    # Valid odds
    is_valid, _ = validate_odds(2.50)
    if is_valid:
        results.add_pass("Valid odds (2.50) accepted")
    else:
        results.add_fail("Valid odds accepted", "Should be valid")

    # Edge case: minimum valid
    is_valid, _ = validate_odds(1.01)
    if is_valid:
        results.add_pass("Minimum odds (1.01) accepted")
    else:
        results.add_fail("Minimum odds accepted", "Should be valid")

    # Invalid: below 1.01
    is_valid, error = validate_odds(1.00)
    if not is_valid:
        results.add_pass("Odds 1.00 rejected", error)
    else:
        results.add_fail("Odds 1.00 rejected", "Should have errored")

    # Invalid: way too high
    is_valid, error = validate_odds(500.0)
    if not is_valid:
        results.add_pass("Unreasonable odds (500) rejected", error)
    else:
        results.add_fail("Unreasonable odds rejected", "Should have errored")


def test_validate_predictions():
    """Test predictions list validation."""
    print("\n=== Testing Predictions Validation ===")

    from scripts.utils.error_handling import validate_predictions

    # Valid predictions
    valid_preds = [
        {
            "match": "Inter vs Milan",
            "probabilities": {"home": 0.45, "draw": 0.30, "away": 0.25}
        }
    ]
    is_valid, errors = validate_predictions(valid_preds)

    if is_valid:
        results.add_pass("Valid predictions accepted")
    else:
        results.add_fail("Valid predictions accepted", f"Errors: {errors}")

    # Empty list
    is_valid, errors = validate_predictions([])
    if not is_valid:
        results.add_pass("Empty predictions list rejected", f"Errors: {errors}")
    else:
        results.add_fail("Empty predictions list rejected", "Should have errored")

    # Probabilities don't sum to 1
    bad_probs = [
        {
            "match": "Inter vs Milan",
            "probabilities": {"home": 0.50, "draw": 0.50, "away": 0.50}  # Sum = 1.5
        }
    ]
    is_valid, errors = validate_predictions(bad_probs)

    if not is_valid or "sum" in str(errors).lower():
        results.add_pass("Invalid probability sum detected", f"Errors: {errors}")
    else:
        results.add_fail("Invalid probability sum detected", "Should have warned")


def test_health_check():
    """Test system health check."""
    print("\n=== Testing Health Check ===")

    from scripts.utils.error_handling import health_check

    health = health_check()

    # Should return a valid structure
    if "status" in health:
        results.add_pass("Health check returns status", f"Status: {health['status']}")
    else:
        results.add_fail("Health check returns status", "No status field")

    if "checks" in health:
        results.add_pass("Health check returns checks", f"Checks: {list(health['checks'].keys())}")
    else:
        results.add_fail("Health check returns checks", "No checks field")

    if "timestamp" in health:
        results.add_pass("Health check includes timestamp")
    else:
        results.add_fail("Health check includes timestamp", "No timestamp")


def test_feature_availability():
    """Test feature availability tracking."""
    print("\n=== Testing Feature Availability ===")

    from scripts.utils.error_handling import check_feature_availability, feature_status

    # Check features
    features = check_feature_availability()

    # Should return a dict
    if isinstance(features, dict):
        results.add_pass("Feature check returns dict", f"Features: {list(features.keys())}")
    else:
        results.add_fail("Feature check returns dict", f"Got {type(features)}")

    # Test feature status tracking
    feature_status.set_available("test_feature", True, "Test details")
    if feature_status.is_available("test_feature"):
        results.add_pass("Feature status tracking works")
    else:
        results.add_fail("Feature status tracking works", "Status not recorded")

    # Non-existent feature should return False
    if not feature_status.is_available("nonexistent_feature"):
        results.add_pass("Non-existent feature returns False")
    else:
        results.add_fail("Non-existent feature returns False", "Should be False")


def test_safe_load_matches():
    """Test safe match loading."""
    print("\n=== Testing Safe Match Loading ===")

    from scripts.utils.error_handling import safe_load_matches

    # Should not crash even if files are missing/corrupt
    try:
        matches = safe_load_matches()

        if isinstance(matches, list):
            results.add_pass("Safe load returns list", f"Found {len(matches)} matches")
        else:
            results.add_fail("Safe load returns list", f"Got {type(matches)}")

    except Exception as e:
        results.add_fail("Safe load doesn't crash", f"Crashed with: {e}")


def test_safe_load_predictions():
    """Test safe predictions loading."""
    print("\n=== Testing Safe Predictions Loading ===")

    from scripts.utils.error_handling import safe_load_predictions

    # Should not crash
    try:
        predictions = safe_load_predictions()

        if isinstance(predictions, list):
            results.add_pass("Safe load predictions returns list", f"Found {len(predictions)} predictions")
        else:
            results.add_fail("Safe load predictions returns list", f"Got {type(predictions)}")

    except Exception as e:
        results.add_fail("Safe load predictions doesn't crash", f"Crashed with: {e}")


def test_extreme_values():
    """Test handling of extreme values."""
    print("\n=== Testing Extreme Values ===")

    from scripts.prediction.current_form_calculator import identify_factors

    # Extreme Elo difference (1000 points)
    home_form = {"form_status": "hot"}
    away_form = {"form_status": "cold"}
    extreme_elo = 1000

    try:
        factors = identify_factors(home_form, away_form, extreme_elo, {"home_team": "Inter", "away_team": "Lecce"})

        if "big_home_favorite" in factors:
            results.add_pass("Extreme Elo (1000) handled", f"Factors: {factors}")
        else:
            results.add_fail("Extreme Elo handled", f"Expected big_home_favorite")

    except Exception as e:
        results.add_fail("Extreme Elo doesn't crash", f"Crashed with: {e}")

    # Negative Elo difference (extreme underdog)
    try:
        factors = identify_factors(home_form, away_form, -500, {"home_team": "Lecce", "away_team": "Inter"})

        if "big_away_favorite" in factors or "away_favorite" in factors:
            results.add_pass("Extreme negative Elo (-500) handled", f"Factors: {factors}")
        else:
            results.add_fail("Extreme negative Elo handled", f"Expected away_favorite")

    except Exception as e:
        results.add_fail("Extreme negative Elo doesn't crash", f"Crashed with: {e}")


def test_weather_missing_data():
    """Test weather integration with missing data."""
    print("\n=== Testing Weather Missing Data ===")

    from scripts.prediction.weather_integration import get_weather_impact

    # Empty weather dict
    empty_result = get_weather_impact({})

    if empty_result.get("goals_adj", None) == 0:
        results.add_pass("Empty weather returns 0 adjustment")
    else:
        results.add_fail("Empty weather returns 0 adjustment", f"Got: {empty_result}")

    # None weather
    none_result = get_weather_impact(None)

    if none_result.get("goals_adj", None) == 0:
        results.add_pass("None weather returns 0 adjustment")
    else:
        results.add_fail("None weather returns 0 adjustment", f"Got: {none_result}")


def test_kelly_edge_cases():
    """Test Kelly criterion with edge cases."""
    print("\n=== Testing Kelly Edge Cases ===")

    from features.bankroll_manager import calculate_kelly_stake

    bankroll = 1000.0

    # Zero probability
    stake = calculate_kelly_stake(bankroll, 0.0, 2.0)
    if stake == 0:
        results.add_pass("Zero probability returns 0 stake")
    else:
        results.add_fail("Zero probability returns 0 stake", f"Got: {stake}")

    # 100% probability (should still cap)
    stake = calculate_kelly_stake(bankroll, 1.0, 2.0)
    if stake >= 0:
        results.add_pass("100% probability handled", f"Stake: {stake}")
    else:
        results.add_fail("100% probability handled", f"Got: {stake}")

    # Odds of 1.0 (no profit possible)
    stake = calculate_kelly_stake(bankroll, 0.6, 1.0)
    if stake == 0:
        results.add_pass("Odds 1.0 returns 0 stake")
    else:
        results.add_fail("Odds 1.0 returns 0 stake", f"Got: {stake}")


def main():
    print("=" * 70)
    print("EDGE CASE VALIDATION TESTS")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("\nThese tests verify error handling and graceful degradation.")

    test_validate_match_data()
    test_validate_probabilities()
    test_validate_odds()
    test_validate_predictions()
    test_health_check()
    test_feature_availability()
    test_safe_load_matches()
    test_safe_load_predictions()
    test_extreme_values()
    test_weather_missing_data()
    test_kelly_edge_cases()

    success = results.summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
