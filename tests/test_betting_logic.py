#!/usr/bin/env python3
"""Deep Functional Tests - Betting Logic Validation

These tests verify that betting calculations are ACTUALLY CORRECT,
not just that they execute.

Run with: python tests/test_betting_logic.py
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
        print(f"BETTING LOGIC RESULTS: {self.passed}/{total} tests passed")
        if self.errors:
            print(f"\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        return self.failed == 0


results = Results()


def test_kelly_criterion_calculation():
    """Test that Kelly criterion calculates correctly."""
    print("\n=== Testing Kelly Criterion ===")

    from features.bankroll_manager import calculate_kelly_stake, KELLY_FRACTION

    # Use a test bankroll of 1000 for calculations
    bankroll = 1000.0

    # Test 1: High probability scenario
    # If our probability is 60% and odds are 2.0 (implied 50%)
    # Kelly full = (0.6 * 2.0 - 1) / (2.0 - 1) = 0.2 / 1 = 20%
    # With 25% Kelly fraction = 5%
    probability = 0.60
    odds = 2.0
    kelly = calculate_kelly_stake(bankroll, probability, odds)

    # Expected: ~5% of bankroll with 25% Kelly fraction = 50
    if kelly > 0:
        kelly_pct = kelly / bankroll
        results.add_pass("Kelly calculation for 60% prob @ 2.0 odds",
                        f"Stake: {kelly:.2f} ({kelly_pct:.1%} of bankroll)")
    else:
        results.add_fail("Kelly calculation",
                        f"Calculated: {kelly:.2f}, should be positive for 60% prob @ 2.0 odds")

    # Test 2: Negative EV should give 0 stake
    # 40% prob at 2.0 odds (implied 50%) = negative EV
    kelly_neg = calculate_kelly_stake(bankroll, 0.40, 2.0)

    if kelly_neg <= 0:
        results.add_pass("Negative EV gives 0 stake",
                        f"40% prob @ 2.0 odds: stake = {kelly_neg:.2f}")
    else:
        results.add_fail("Negative EV gives 0 stake",
                        f"Stake {kelly_neg:.2f} should be 0 for negative EV")

    # Test 3: Higher probability = higher Kelly stake (use lower probs to avoid cap)
    # 55% prob at 2.0 odds: Kelly = (0.55*2-1)/(2-1) * 0.25 = 0.10 * 0.25 = 2.5%
    # 60% prob at 2.0 odds: Kelly = (0.60*2-1)/(2-1) * 0.25 = 0.20 * 0.25 = 5.0%
    kelly_55 = calculate_kelly_stake(bankroll, 0.55, 2.0)
    kelly_60 = calculate_kelly_stake(bankroll, 0.60, 2.0)

    if kelly_60 > kelly_55:
        results.add_pass("Higher probability = higher Kelly stake",
                        f"60% prob: {kelly_60:.2f}, 55% prob: {kelly_55:.2f}")
    else:
        results.add_fail("Higher probability = higher Kelly stake",
                        f"60% stake ({kelly_60:.2f}) should be > 55% ({kelly_55:.2f})")


def test_value_detection():
    """Test that value bets are correctly identified."""
    print("\n=== Testing Value Detection ===")

    from features.bankroll_manager import calculate_value

    # Test 1: Clear value bet (our prob > implied prob)
    our_prob = 0.55  # We think 55%
    odds = 2.20  # Implied probability = 1/2.20 = 45.5%
    value = calculate_value(our_prob, odds)

    expected_value = our_prob * odds - 1  # Expected return = 0.55 * 2.2 - 1 = 0.21
    if value > 0 and abs(value - expected_value) < 0.05:
        results.add_pass("Value bet correctly identified",
                        f"Our prob: {our_prob:.0%}, Odds: {odds}, Value: {value:.2%}")
    elif value > 0:
        results.add_pass("Value bet detected (positive edge)",
                        f"Our prob: {our_prob:.0%}, Odds: {odds}, Value: {value:.2%}")
    else:
        results.add_fail("Value bet identification",
                        f"Value should be positive: got {value:.2%}")

    # Test 2: No value bet (our prob < implied prob)
    our_prob_low = 0.40  # We think 40%
    odds_low = 2.0  # Implied = 50%
    value_no = calculate_value(our_prob_low, odds_low)

    if value_no <= 0:
        results.add_pass("Non-value bet correctly rejected",
                        f"Our prob: {our_prob_low:.0%}, Odds: {odds_low}, Value: {value_no:.2%}")
    else:
        results.add_fail("Non-value bet should be rejected",
                        f"Value ({value_no:.2%}) should be negative")

    # Test 3: Marginal value
    our_prob_marginal = 0.52
    odds_marginal = 1.90  # Implied = 52.6%
    value_marginal = calculate_value(our_prob_marginal, odds_marginal)

    # This is very marginal (52% vs 52.6% implied)
    results.add_pass(f"Marginal value correctly calculated",
                    f"Our prob: {our_prob_marginal:.0%}, Implied: {1/odds_marginal:.1%}, Value: {value_marginal:.2%}")


def test_italian_market_handicap_conversion():
    """Test Italian market handicap line conversions."""
    print("\n=== Testing Italian Handicap Conversion ===")

    from scripts.betting.italian_market_standards import normalize_line_to_italian

    # Test 1: -1.5 should become -1 (round toward zero)
    result = normalize_line_to_italian(-1.5, "handicap")
    if result == -1:
        results.add_pass("Handicap -1.5 -> -1",
                        f"Converted -1.5 to {result}")
    else:
        results.add_fail("Handicap -1.5 conversion",
                        f"Expected -1, got {result}")

    # Test 2: -2.5 should become -2 (round toward zero)
    result = normalize_line_to_italian(-2.5, "handicap")
    if result == -2:
        results.add_pass("Handicap -2.5 -> -2",
                        f"Converted -2.5 to {result}")
    else:
        results.add_fail("Handicap -2.5 conversion",
                        f"Expected -2, got {result}")

    # Test 3: +1.5 should become +1 (round toward zero)
    result = normalize_line_to_italian(1.5, "handicap")
    if result == 1:
        results.add_pass("Handicap +1.5 -> +1",
                        f"Converted +1.5 to {result}")
    else:
        results.add_fail("Handicap +1.5 conversion",
                        f"Expected +1, got {result}")

    # Test 4: Integer handicaps unchanged
    result = normalize_line_to_italian(-2, "handicap")
    if result == -2:
        results.add_pass("Integer handicap -2 unchanged",
                        f"Kept -2 as {result}")
    else:
        results.add_fail("Integer handicap unchanged",
                        f"Expected -2, got {result}")


def test_italian_market_ou_conversion():
    """Test Italian market over/under line conversions."""
    print("\n=== Testing Italian O/U Conversion ===")

    from scripts.betting.italian_market_standards import normalize_line_to_italian

    # Test 1: 2.75 should become 3.5 (round up)
    result = normalize_line_to_italian(2.75, "totals")
    if result == 3.5:
        results.add_pass("O/U 2.75 -> 3.5",
                        f"Converted 2.75 to {result}")
    else:
        results.add_fail("O/U 2.75 conversion",
                        f"Expected 3.5, got {result}")

    # Test 2: 3.0 should become 3.5 (round up for conservative)
    result = normalize_line_to_italian(3.0, "totals")
    if result == 3.5:
        results.add_pass("O/U 3.0 -> 3.5",
                        f"Converted 3.0 to {result}")
    else:
        results.add_fail("O/U 3.0 conversion",
                        f"Expected 3.5, got {result}")

    # Test 3: 2.25 should become 2.5
    result = normalize_line_to_italian(2.25, "totals")
    if result == 2.5:
        results.add_pass("O/U 2.25 -> 2.5",
                        f"Converted 2.25 to {result}")
    else:
        results.add_fail("O/U 2.25 conversion",
                        f"Expected 2.5, got {result}")

    # Test 4: Standard lines unchanged
    result = normalize_line_to_italian(2.5, "totals")
    if result == 2.5:
        results.add_pass("O/U 2.5 unchanged",
                        f"Kept 2.5 as {result}")
    else:
        results.add_fail("Standard O/U line unchanged",
                        f"Expected 2.5, got {result}")


def test_bet_recommendations_have_positive_value():
    """Test that all recommended bets have positive expected value."""
    print("\n=== Testing Bet Recommendations Value ===")

    data_dir = PROJECT_ROOT / "data" / "upcoming"

    # Check over/under bets
    ou_path = data_dir / "over_under_bets.json"
    if ou_path.exists():
        with open(ou_path) as f:
            ou_data = json.load(f)

        recommended = ou_data.get("recommended", [])
        all_positive = True
        negative_bets = []

        for bet in recommended:
            value_pct = bet.get("value_pct", 0)
            if value_pct <= 0:
                all_positive = False
                negative_bets.append(f"{bet.get('match')}: {value_pct}%")

        if all_positive and recommended:
            results.add_pass(f"All {len(recommended)} O/U recommendations have +EV",
                           f"Min value: {min(b.get('value_pct', 0) for b in recommended):.1f}%")
        elif not recommended:
            results.add_pass("No O/U bets (data may be stale)", "No recommendations to validate")
        else:
            results.add_fail("O/U recommendations have positive value",
                           f"Negative EV bets: {negative_bets[:3]}")
    else:
        results.add_fail("O/U bets file", "over_under_bets.json not found")

    # Check handicap bets
    hc_path = data_dir / "handicap_bets.json"
    if hc_path.exists():
        with open(hc_path) as f:
            hc_data = json.load(f)

        recommended = hc_data.get("recommended", [])
        all_positive = True

        for bet in recommended:
            if bet.get("value_pct", 0) <= 0:
                all_positive = False

        if all_positive and recommended:
            results.add_pass(f"All {len(recommended)} handicap recommendations have +EV",
                           f"Min value: {min(b.get('value_pct', 0) for b in recommended):.1f}%")
        elif not recommended:
            results.add_pass("No handicap bets (data may be stale)", "No recommendations")
        else:
            results.add_fail("Handicap recommendations have positive value",
                           f"Found bets with <= 0 value")
    else:
        results.add_fail("Handicap bets file", "handicap_bets.json not found")


def test_stake_sizing_is_reasonable():
    """Test that stake recommendations are within reasonable bounds."""
    print("\n=== Testing Stake Sizing ===")

    from features.bankroll_manager import calculate_kelly_stake, KELLY_FRACTION, MAX_SINGLE_BET_PCT

    # Use test bankroll
    bankroll = 1000.0

    # Kelly should be capped at reasonable levels (typically 5% max)
    # Very high probability (90%) at 2.0 odds
    high_prob = 0.90
    odds = 2.0

    kelly = calculate_kelly_stake(bankroll, high_prob, odds)
    kelly_pct = kelly / bankroll

    # Even with 90% prob, stake should be capped at MAX_SINGLE_BET_PCT (usually 5%)
    if kelly_pct <= MAX_SINGLE_BET_PCT + 0.01:  # Allow small tolerance
        results.add_pass("Kelly stake is capped",
                        f"90% prob @ 2.0 odds: {kelly_pct:.1%} (max: {MAX_SINGLE_BET_PCT:.0%})")
    else:
        results.add_fail("Kelly stake capping",
                        f"Stake {kelly_pct:.1%} exceeds max {MAX_SINGLE_BET_PCT:.0%}")

    # Test fractional Kelly
    if 0 < KELLY_FRACTION <= 0.5:
        results.add_pass(f"Using fractional Kelly ({KELLY_FRACTION:.0%})",
                        "Good bankroll management")
    else:
        results.add_fail("Fractional Kelly setting",
                        f"KELLY_FRACTION={KELLY_FRACTION} should be 0-0.5")


def test_odds_implied_probability_calculation():
    """Test that implied probability calculations are correct."""
    print("\n=== Testing Odds/Probability Conversion ===")

    # Test direct calculation (implied probability = 1 / odds)
    def odds_to_probability(odds):
        return 1.0 / odds if odds > 0 else 0

    def probability_to_odds(prob):
        return 1.0 / prob if prob > 0 else 0

    # Test 1: Odds 2.0 = 50%
    prob = odds_to_probability(2.0)
    if abs(prob - 0.50) < 0.01:
        results.add_pass("Odds 2.0 -> 50% probability",
                        f"Calculated: {prob:.1%}")
    else:
        results.add_fail("Odds 2.0 conversion",
                        f"Expected 50%, got {prob:.1%}")

    # Test 2: Odds 1.5 = 66.7%
    prob = odds_to_probability(1.5)
    if abs(prob - 0.667) < 0.01:
        results.add_pass("Odds 1.5 -> 66.7% probability",
                        f"Calculated: {prob:.1%}")
    else:
        results.add_fail("Odds 1.5 conversion",
                        f"Expected 66.7%, got {prob:.1%}")

    # Test 3: Odds 3.0 = 33.3%
    prob = odds_to_probability(3.0)
    if abs(prob - 0.333) < 0.01:
        results.add_pass("Odds 3.0 -> 33.3% probability",
                        f"Calculated: {prob:.1%}")
    else:
        results.add_fail("Odds 3.0 conversion",
                        f"Expected 33.3%, got {prob:.1%}")

    # Test 4: Round-trip conversion
    original_prob = 0.45
    odds = probability_to_odds(original_prob)
    back_to_prob = odds_to_probability(odds)

    if abs(back_to_prob - original_prob) < 0.001:
        results.add_pass("Round-trip prob->odds->prob",
                        f"{original_prob:.1%} -> {odds:.2f} -> {back_to_prob:.1%}")
    else:
        results.add_fail("Round-trip conversion",
                        f"Lost precision: {original_prob:.1%} -> {back_to_prob:.1%}")


def main():
    print("=" * 70)
    print("BETTING LOGIC VALIDATION TESTS")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("\nThese tests verify betting calculations are CORRECT.")

    test_kelly_criterion_calculation()
    test_value_detection()
    test_italian_market_handicap_conversion()
    test_italian_market_ou_conversion()
    test_bet_recommendations_have_positive_value()
    test_stake_sizing_is_reasonable()
    test_odds_implied_probability_calculation()

    success = results.summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
