#!/usr/bin/env python3
"""Deep Functional Tests - Feature Impact Validation

These tests verify that features ACTUALLY IMPACT predictions,
not just that they're loaded.

Run with: python tests/test_feature_impact.py
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
        print(f"FEATURE IMPACT RESULTS: {self.passed}/{total} tests passed")
        if self.errors:
            print(f"\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        return self.failed == 0


results = Results()


def test_weather_impacts_goal_predictions():
    """Test that weather data actually affects goal predictions."""
    print("\n=== Testing Weather Impact on Goals ===")

    from scripts.prediction.weather_integration import get_weather_impact, classify_weather

    # Test 1: Rain should increase goals (per the code: rain +0.20 goals)
    rain_weather = {
        "temperature": 10,
        "precipitation": 10.0,  # Heavy rain (>5mm triggers is_rainy)
        "wind_speed": 3.0,
    }
    rain_weather["conditions"] = classify_weather(rain_weather)
    rain_impact = get_weather_impact(rain_weather)
    rain_goals_adj = rain_impact.get("goals_adj", 0)

    # Test 2: Clear weather should have no adjustment
    clear_weather = {
        "temperature": 20,
        "precipitation": 0.0,
        "wind_speed": 2.0,
    }
    clear_weather["conditions"] = classify_weather(clear_weather)
    clear_impact = get_weather_impact(clear_weather)
    clear_goals_adj = clear_impact.get("goals_adj", 0)

    # Rain should have positive goals adjustment per the code (+0.20)
    if rain_goals_adj > clear_goals_adj:
        results.add_pass("Rain affects goal prediction",
                        f"Rain: {rain_goals_adj:+.2f}, Clear: {clear_goals_adj:+.2f}")
    elif rain_goals_adj == clear_goals_adj == 0:
        results.add_fail("Rain affects goal prediction",
                        f"Weather not affecting goals: both = {rain_goals_adj}")
    else:
        results.add_fail("Rain affects goal prediction",
                        f"Rain ({rain_goals_adj:+.2f}) should differ from Clear ({clear_goals_adj:+.2f})")

    # Test 3: Wind should reduce goals (-0.22 per the code)
    windy_weather = {
        "temperature": 15,
        "precipitation": 0.0,
        "wind_speed": 35.0,  # Very windy (>30 triggers is_windy)
    }
    windy_weather["conditions"] = classify_weather(windy_weather)
    windy_impact = get_weather_impact(windy_weather)
    windy_goals_adj = windy_impact.get("goals_adj", 0)

    if windy_goals_adj < clear_goals_adj:
        results.add_pass("High wind reduces goal prediction",
                        f"Windy: {windy_goals_adj:+.2f}, Clear: {clear_goals_adj:+.2f}")
    else:
        results.add_fail("High wind reduces goal prediction",
                        f"Wind ({windy_goals_adj:+.2f}) should be < Clear ({clear_goals_adj:+.2f})")


def test_referee_impacts_card_predictions():
    """Test that referee strictness affects card predictions."""
    print("\n=== Testing Referee Impact on Cards ===")

    from scripts.prediction.referee_integration import (
        classify_referee, STRICT_REFS, LENIENT_REFS,
        HOME_FAVORING_REFS, AWAY_FAVORING_REFS
    )

    # Get a known strict referee
    strict_refs = list(STRICT_REFS.keys()) if STRICT_REFS else ["Marco Guida"]
    lenient_refs = list(LENIENT_REFS.keys()) if LENIENT_REFS else []

    if strict_refs:
        strict_ref = strict_refs[0]
        strict_class = classify_referee(strict_ref)

        # Strict referee should have high card rate
        high_card_rate = strict_class.get("high_card_rate", 0)
        if high_card_rate > 0.5:  # Above 50% high card games
            results.add_pass(f"Strict referee ({strict_ref}) classified correctly",
                           f"High card rate: {high_card_rate:.1%}")
        elif strict_class.get("bias_type") == "strict":
            results.add_pass(f"Strict referee ({strict_ref}) classified as strict",
                           f"Bias type: {strict_class.get('bias_type')}")
        else:
            results.add_fail(f"Strict referee ({strict_ref}) classification",
                           f"Expected 'strict', got: {strict_class.get('bias_type')}")
    else:
        results.add_fail("Strict referee test", "No strict referees defined")

    # Test home/away favoring refs
    if HOME_FAVORING_REFS:
        home_fav_ref = list(HOME_FAVORING_REFS.keys())[0]
        home_fav_class = classify_referee(home_fav_ref)
        home_lift = home_fav_class.get("home_win_lift", 0)

        if home_lift > 0:
            results.add_pass(f"Home-favoring ref ({home_fav_ref}) boosts home",
                           f"Home win lift: {home_lift:+.1%}")
        else:
            results.add_fail(f"Home-favoring ref ({home_fav_ref}) boosts home",
                           f"Home lift ({home_lift}) should be > 0")


def test_form_impacts_predictions():
    """Test that team form actually affects predictions."""
    print("\n=== Testing Form Impact on Predictions ===")

    from scripts.prediction.current_form_calculator import identify_factors

    # Test hot home vs cold away
    match = {"home_team": "Lecce", "away_team": "Empoli"}

    # Hot home form
    hot_home_form = {"form_status": "hot", "ppg": 2.8}
    cold_away_form = {"form_status": "cold", "ppg": 0.6}
    elo_diff = 100  # Home favorite

    factors_hot_home = identify_factors(hot_home_form, cold_away_form, elo_diff, match)

    if "hot_home" in factors_hot_home:
        results.add_pass("Hot home team gets 'hot_home' factor",
                        f"Factors: {factors_hot_home}")
    else:
        results.add_fail("Hot home team gets 'hot_home' factor",
                        f"'hot_home' not in factors: {factors_hot_home}")

    if "cold_away" in factors_hot_home:
        results.add_pass("Cold away team gets 'cold_away' factor",
                        f"Factors: {factors_hot_home}")
    else:
        results.add_fail("Cold away team gets 'cold_away' factor",
                        f"'cold_away' not in factors: {factors_hot_home}")

    # Test cold home
    cold_home_form = {"form_status": "cold", "ppg": 0.6}
    hot_away_form = {"form_status": "hot", "ppg": 2.8}
    factors_cold_home = identify_factors(cold_home_form, hot_away_form, -100, match)

    if "cold_home" in factors_cold_home:
        results.add_pass("Cold home team gets 'cold_home' factor",
                        f"Factors: {factors_cold_home}")
    else:
        results.add_fail("Cold home team gets 'cold_home' factor",
                        f"'cold_home' not in factors: {factors_cold_home}")


def test_stadium_size_impacts_predictions():
    """Test that big stadium factor is applied correctly."""
    print("\n=== Testing Stadium Size Impact ===")

    from scripts.prediction.current_form_calculator import identify_factors

    # Inter at San Siro (big stadium - in the list)
    match_big_stadium = {"home_team": "Inter", "away_team": "Lecce"}
    home_form = {"form_status": "normal", "ppg": 2.2}
    away_form = {"form_status": "normal", "ppg": 1.0}
    elo_diff = 300  # Big home favorite

    factors_inter = identify_factors(home_form, away_form, elo_diff, match_big_stadium)

    if "big_stadium" in factors_inter:
        results.add_pass("Big stadium factor applied for Inter",
                        f"Factors: {factors_inter}")
    else:
        results.add_fail("Big stadium factor applied for Inter",
                        f"'big_stadium' not in factors: {factors_inter}")

    # Test small stadium team
    match_small_stadium = {"home_team": "Lecce", "away_team": "Empoli"}
    factors_lecce = identify_factors(home_form, away_form, 50, match_small_stadium)

    if "big_stadium" not in factors_lecce:
        results.add_pass("Small stadium (Lecce) does NOT get big_stadium factor",
                        f"Factors: {factors_lecce}")
    else:
        results.add_fail("Small stadium should not get big_stadium factor",
                        f"Unexpected 'big_stadium' in: {factors_lecce}")


def test_elo_difference_impacts_favorite_status():
    """Test that Elo difference creates favorite/underdog factors."""
    print("\n=== Testing Elo Impact on Favorite Status ===")

    from scripts.prediction.current_form_calculator import identify_factors

    # Big home favorite (Elo diff >= 200)
    match = {"home_team": "Lecce", "away_team": "Empoli"}
    home_form = {"form_status": "normal", "ppg": 2.0}
    away_form = {"form_status": "normal", "ppg": 1.0}

    # Test big home favorite (200+ Elo diff)
    factors_big_fav = identify_factors(home_form, away_form, 250, match)

    if "big_home_favorite" in factors_big_fav:
        results.add_pass("Big Elo diff (250) creates 'big_home_favorite'",
                        f"Factors: {factors_big_fav}")
    else:
        results.add_fail("Big Elo diff creates 'big_home_favorite'",
                        f"No big_home_favorite in: {factors_big_fav}")

    # Test home favorite (100-199 Elo diff)
    factors_home_fav = identify_factors(home_form, away_form, 150, match)

    if "home_favorite" in factors_home_fav and "big_home_favorite" not in factors_home_fav:
        results.add_pass("Medium Elo diff (150) creates 'home_favorite' only",
                        f"Factors: {factors_home_fav}")
    else:
        results.add_fail("Medium Elo diff creates 'home_favorite'",
                        f"Factors: {factors_home_fav}")

    # Test away favorite (Elo diff <= -100)
    factors_away_fav = identify_factors(home_form, away_form, -150, match)

    if "away_favorite" in factors_away_fav:
        results.add_pass("Negative Elo diff (-150) creates 'away_favorite'",
                        f"Factors: {factors_away_fav}")
    else:
        results.add_fail("Negative Elo diff creates 'away_favorite'",
                        f"No away_favorite in: {factors_away_fav}")


def test_prediction_probabilities_sum_to_one():
    """Test that prediction probabilities always sum to ~1.0."""
    print("\n=== Testing Probability Consistency ===")

    predictions_path = PROJECT_ROOT / "data" / "upcoming" / "predictions.json"

    if not predictions_path.exists():
        results.add_fail("Probability sum test", "predictions.json not found")
        return

    with open(predictions_path) as f:
        data = json.load(f)

    predictions = data.get("predictions", data) if isinstance(data, dict) else data

    if not predictions:
        results.add_fail("Probability sum test", "No predictions found")
        return

    all_valid = True
    for pred in predictions[:5]:  # Check first 5
        probs = pred.get("probabilities", {})
        total = probs.get("home", 0) + probs.get("draw", 0) + probs.get("away", 0)

        if abs(total - 1.0) > 0.01:  # Allow 1% tolerance
            all_valid = False
            results.add_fail(f"Probability sum for {pred.get('match', '?')}",
                           f"Sum = {total:.3f}, expected ~1.0")

    if all_valid:
        results.add_pass("All probabilities sum to ~1.0",
                        f"Checked {min(5, len(predictions))} predictions")


def test_confidence_correlates_with_factors():
    """Test that more factors = higher confidence."""
    print("\n=== Testing Confidence vs Factor Count ===")

    predictions_path = PROJECT_ROOT / "data" / "upcoming" / "predictions.json"

    if not predictions_path.exists():
        results.add_fail("Confidence correlation test", "predictions.json not found")
        return

    with open(predictions_path) as f:
        data = json.load(f)

    predictions = data.get("predictions", data) if isinstance(data, dict) else data

    if not predictions:
        results.add_fail("Confidence correlation test", "No predictions found")
        return

    # Check if predictions with more factors have higher confidence
    factor_confidence_pairs = []
    for pred in predictions:
        n_factors = pred.get("n_factors", 0)
        if n_factors == 0:
            n_factors = len(pred.get("home_factors", [])) + len(pred.get("away_factors", []))

        confidence = pred.get("confidence", pred.get("expected_accuracy", 0.5))
        if isinstance(confidence, str):
            conf_map = {"VERY HIGH": 0.9, "HIGH": 0.75, "MEDIUM": 0.6, "LOW": 0.4}
            confidence = conf_map.get(confidence, 0.5)

        factor_confidence_pairs.append((n_factors, confidence))

    # Sort by factor count and check trend
    factor_confidence_pairs.sort(key=lambda x: x[0])

    if len(factor_confidence_pairs) >= 2:
        low_factor_conf = factor_confidence_pairs[0][1]
        high_factor_conf = factor_confidence_pairs[-1][1]
        low_factor_count = factor_confidence_pairs[0][0]
        high_factor_count = factor_confidence_pairs[-1][0]

        if high_factor_count > low_factor_count:
            if high_factor_conf >= low_factor_conf:
                results.add_pass("More factors = higher confidence",
                               f"{low_factor_count} factors: {low_factor_conf:.2f}, "
                               f"{high_factor_count} factors: {high_factor_conf:.2f}")
            else:
                results.add_fail("More factors = higher confidence",
                               f"Inverted: {high_factor_count} factors has lower conf "
                               f"({high_factor_conf:.2f}) than {low_factor_count} factors ({low_factor_conf:.2f})")
        else:
            results.add_pass("Factor count test", "All predictions have same factor count")
    else:
        results.add_fail("Confidence correlation test", "Not enough predictions to compare")


def test_sentiment_analysis_produces_data():
    """Test that sentiment analysis produces actual data."""
    print("\n=== Testing Sentiment Analysis ===")

    sentiment_path = PROJECT_ROOT / "data" / "upcoming" / "sentiment_analysis.json"

    if not sentiment_path.exists():
        results.add_fail("Sentiment analysis file exists", "sentiment_analysis.json not found")
        return

    with open(sentiment_path) as f:
        data = json.load(f)

    matches = data.get("matches", [])
    summary = data.get("summary", {})

    # Test 1: Sentiment analysis has data
    if matches:
        results.add_pass("Sentiment analysis produces data",
                        f"Analyzed {len(matches)} matches")
    else:
        results.add_fail("Sentiment analysis produces data",
                        "No matches analyzed - check Perplexity API")
        return

    # Test 2: Each match has required fields
    required_fields = ["match", "sentiment_edge", "home_composite", "away_composite"]
    first_match = matches[0]
    missing = [f for f in required_fields if f not in first_match]

    if not missing:
        results.add_pass("Sentiment data has required fields",
                        f"Fields: {required_fields}")
    else:
        results.add_fail("Sentiment data structure",
                        f"Missing fields: {missing}")

    # Test 3: Composite scores are reasonable (-100 to +100)
    valid_scores = True
    for m in matches:
        home = m.get("home_composite", 0)
        away = m.get("away_composite", 0)
        if abs(home) > 100 or abs(away) > 100:
            valid_scores = False
            break

    if valid_scores:
        results.add_pass("Sentiment scores in valid range",
                        "All composite scores between -100 and +100")
    else:
        results.add_fail("Sentiment scores valid",
                        "Scores outside expected range")

    # Test 4: Edges are distributed (not all same)
    edges = [m.get("sentiment_edge") for m in matches]
    unique_edges = set(edges)

    if len(unique_edges) > 1:
        results.add_pass("Sentiment produces varied edges",
                        f"Edges: {summary.get('home_edge', 0)} home, "
                        f"{summary.get('away_edge', 0)} away, "
                        f"{summary.get('neutral', 0)} neutral")
    else:
        results.add_fail("Sentiment produces varied edges",
                        f"All edges same: {unique_edges}")


def test_sentiment_analyzer_can_run():
    """Test that sentiment analyzer can be imported and run."""
    print("\n=== Testing Sentiment Analyzer Module ===")

    try:
        from scripts.prediction.sentiment_analyzer import SentimentAnalyzer, PERPLEXITY_API_KEY

        # Check API key exists
        if PERPLEXITY_API_KEY:
            results.add_pass("Perplexity API key configured",
                           f"Key length: {len(PERPLEXITY_API_KEY)}")
        else:
            results.add_fail("Perplexity API key configured",
                           "PERPLEXITY_API_KEY not set")
            return

        # Check analyzer can be instantiated
        analyzer = SentimentAnalyzer()
        results.add_pass("SentimentAnalyzer instantiated")

    except ImportError as e:
        results.add_fail("Sentiment analyzer import", str(e))


def main():
    print("=" * 70)
    print("FEATURE IMPACT VALIDATION TESTS")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("\nThese tests verify features ACTUALLY IMPACT predictions.")

    test_weather_impacts_goal_predictions()
    test_referee_impacts_card_predictions()
    test_form_impacts_predictions()
    test_stadium_size_impacts_predictions()
    test_elo_difference_impacts_favorite_status()
    test_prediction_probabilities_sum_to_one()
    test_confidence_correlates_with_factors()
    test_sentiment_analysis_produces_data()
    test_sentiment_analyzer_can_run()

    success = results.summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
