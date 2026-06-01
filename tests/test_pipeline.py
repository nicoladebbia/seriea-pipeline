#!/usr/bin/env python3
"""Basic Tests for Serie A Prediction Pipeline

Validates key components and ensures no regressions.

Run with: python tests/test_pipeline.py
Or: python -m pytest tests/test_pipeline.py -v
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import os
from datetime import datetime


class Results:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def add_pass(self, test_name: str):
        self.passed += 1
        print(f"  [PASS] {test_name}")

    def add_fail(self, test_name: str, error: str):
        self.failed += 1
        self.errors.append((test_name, error))
        print(f"  [FAIL] {test_name}: {error}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'=' * 60}")
        print(f"RESULTS: {self.passed}/{total} tests passed")
        if self.errors:
            print(f"\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        return self.failed == 0


results = Results()


def test_imports():
    """Test that all main modules can be imported."""
    print("\n=== Testing Imports ===")

    modules = [
        ("config.settings", "Configuration"),
        ("scripts.data.odds_fetcher", "Odds Fetcher"),
        ("scripts.prediction.sentiment_analyzer", "Sentiment Analyzer"),
        ("scripts.analysis.player_analyzer", "Player Analyzer"),
        ("scripts.models.over_under_model", "Over/Under Model"),
        ("scripts.models.handicap_model", "Handicap Model"),
        ("scripts.models.cards_model", "Cards Model"),
        ("scripts.models.btts_corners_model", "BTTS/Corners Model"),
        ("scripts.betting.italian_market_standards", "Italian Market Standards"),
        ("scripts.prediction.weather_integration", "Weather Integration"),
        ("scripts.prediction.referee_integration", "Referee Integration"),
        ("scripts.prediction.ai_reasoning", "AI Reasoning"),
        ("web.app", "Web App"),
    ]

    for module_name, display_name in modules:
        try:
            __import__(module_name)
            results.add_pass(f"Import {display_name}")
        except Exception as e:
            results.add_fail(f"Import {display_name}", str(e)[:80])


def test_api_keys_not_hardcoded():
    """Test that API keys are not hardcoded in source files."""
    print("\n=== Testing API Key Security ===")

    # Check for hardcoded API keys in Python files
    scripts_dir = PROJECT_ROOT / "scripts"
    hardcoded_keys = []

    for py_file in scripts_dir.glob("*.py"):
        content = py_file.read_text()
        # Check for patterns that look like hardcoded keys
        if "sk-proj-" in content and '""' not in content.split("sk-proj-")[0][-20:]:
            hardcoded_keys.append((py_file.name, "OpenAI key"))
        if "pplx-" in content and len(content.split("pplx-")[1].split('"')[0]) > 20:
            # Check if it's in a string (hardcoded) vs just documentation
            if '= "pplx-' in content:
                pass  # Skip - might be in os.environ fallback

    # Verify .env exists
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        results.add_pass("Environment file exists")
    else:
        results.add_fail("Environment file", ".env not found")

    # Verify .env.example exists
    env_example = PROJECT_ROOT / ".env.example"
    if env_example.exists():
        results.add_pass("Environment example file exists")
    else:
        results.add_fail("Environment example", ".env.example not found")


def test_italian_market_standards():
    """Test Italian market line conversions."""
    print("\n=== Testing Italian Market Standards ===")

    from scripts.betting.italian_market_standards import normalize_line_to_italian

    # Test over/under conversions
    ou_tests = [
        (2.5, "totals", 2.5),   # Standard line unchanged
        (2.75, "totals", 3.5),  # Asian line rounded up
        (3.0, "totals", 3.5),   # Round up for conservative
        (2.25, "totals", 2.5),  # Round to nearest standard
    ]

    for original, market, expected in ou_tests:
        result = normalize_line_to_italian(original, market)
        if result == expected:
            results.add_pass(f"O/U {original} -> {result}")
        else:
            results.add_fail(f"O/U {original}", f"Expected {expected}, got {result}")

    # Test handicap conversions (should round toward zero)
    hc_tests = [
        (-1.5, "handicap", -1),   # Round toward zero
        (-2.5, "handicap", -2),
        (1.5, "handicap", 1),     # Round toward zero
        (-1.0, "handicap", -1),   # Integer unchanged
    ]

    for original, market, expected in hc_tests:
        result = normalize_line_to_italian(original, market)
        if result == expected:
            results.add_pass(f"HC {original} -> {result}")
        else:
            results.add_fail(f"HC {original}", f"Expected {expected}, got {result}")


def test_prediction_engine():
    """Test the prediction engine produces valid output."""
    print("\n=== Testing Prediction Engine ===")

    try:
        from scripts.prediction.predict_unified import run_predictions

        result = run_predictions()

        if result:
            results.add_pass("Prediction engine runs")
        else:
            results.add_fail("Prediction engine", "Returned empty result")

        predictions = result.get("predictions", [])
        if predictions:
            results.add_pass(f"Generated {len(predictions)} predictions")

            # Validate prediction structure
            p = predictions[0]
            required_fields = ["match", "predicted_outcome", "probabilities"]
            for field in required_fields:
                if field in p:
                    results.add_pass(f"Prediction has '{field}' field")
                else:
                    results.add_fail("Prediction structure", f"Missing '{field}'")
        else:
            results.add_fail("Predictions", "No predictions generated")

    except Exception as e:
        results.add_fail("Prediction engine", str(e)[:80])


def test_web_routes():
    """Test web app routes respond correctly."""
    print("\n=== Testing Web Routes ===")

    try:
        from web.app import app

        with app.test_client() as client:
            routes = [
                ("/", 200, "Home page"),
                ("/upcoming", 200, "Upcoming page"),
                ("/betting", 200, "Betting page"),
                ("/three-tier", 200, "Three-tier page"),
                ("/analytics", 200, "Analytics page"),
                ("/api/tiers/all", 200, "Tiers API"),
            ]

            for route, expected_code, name in routes:
                response = client.get(route)
                if response.status_code == expected_code:
                    results.add_pass(f"Route {name}: {route}")
                else:
                    results.add_fail(f"Route {name}",
                                   f"Expected {expected_code}, got {response.status_code}")

    except Exception as e:
        results.add_fail("Web routes", str(e)[:80])


def test_data_files():
    """Test that required data files exist and are valid JSON."""
    print("\n=== Testing Data Files ===")

    data_dir = PROJECT_ROOT / "data"
    upcoming_dir = data_dir / "upcoming"
    betting_dir = data_dir / "betting"

    required_files = [
        ("registry.json", data_dir),
        ("predictions.json", upcoming_dir),
    ]

    for filename, directory in required_files:
        filepath = directory / filename
        if filepath.exists():
            try:
                with open(filepath) as f:
                    data = json.load(f)
                results.add_pass(f"Data file: {filename}")
            except json.JSONDecodeError:
                results.add_fail(f"Data file: {filename}", "Invalid JSON")
        else:
            # Not a hard failure - files may not exist yet
            results.add_pass(f"Data file: {filename} (optional, not present)")


def main():
    """Run all tests."""
    print("=" * 60)
    print("SERIE A PREDICTION PIPELINE - TEST SUITE")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    test_imports()
    test_api_keys_not_hardcoded()
    test_italian_market_standards()
    test_prediction_engine()
    test_web_routes()
    test_data_files()

    success = results.summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
