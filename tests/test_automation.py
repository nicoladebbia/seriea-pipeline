#!/usr/bin/env python3
"""Automation Verification Tests - Check Pipeline Automation

These tests verify:
1. Scheduler configuration is valid
2. Health check endpoints work
3. Data files are fresh
4. All pipeline components can run

Run with: python tests/test_automation.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
from datetime import datetime, timedelta


class Results:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
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

    def add_warning(self, test_name: str, warning: str):
        self.warnings += 1
        print(f"  [WARN] {test_name}")
        print(f"         {warning}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'=' * 70}")
        print(f"AUTOMATION RESULTS: {self.passed}/{total} tests passed")
        if self.warnings:
            print(f"Warnings: {self.warnings}")
        if self.errors:
            print(f"\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        return self.failed == 0


results = Results()


def test_scheduler_exists():
    """Test that scheduler script exists and is valid."""
    print("\n=== Testing Scheduler Setup ===")

    scheduler_path = PROJECT_ROOT / "scripts" / "scheduler.py"

    if scheduler_path.exists():
        results.add_pass("Scheduler script exists", str(scheduler_path))
    else:
        results.add_fail("Scheduler script exists", "scheduler.py not found")
        return

    # Check it can be imported
    try:
        from scripts.pipeline.scheduler import (
            SCHEDULE_CONFIG, is_match_day, get_upcoming_matches, run_pipeline
        )
        results.add_pass("Scheduler can be imported")

        # Check schedule config
        if "daily_runs" in SCHEDULE_CONFIG:
            runs = SCHEDULE_CONFIG["daily_runs"]
            # Built outside the f-string: nesting the same quote character is
            # 3.12+ syntax, and pyproject declares requires-python >= 3.11.
            times = [f"{r['hour']:02d}:{r['minute']:02d}" for r in runs]
            results.add_pass(f"Scheduler has {len(runs)} daily runs configured",
                           f"Times: {times}")
        else:
            results.add_fail("Scheduler config", "No daily_runs configured")

    except ImportError as e:
        results.add_fail("Scheduler import", str(e))


def test_scheduler_log():
    """Test that scheduler log exists and is recent."""
    print("\n=== Testing Scheduler Log ===")

    log_path = PROJECT_ROOT / "logs" / "scheduler.log"

    if log_path.exists():
        mtime = datetime.fromtimestamp(log_path.stat().st_mtime)
        age = datetime.now() - mtime

        results.add_pass("Scheduler log exists", str(log_path))

        if age < timedelta(days=7):
            results.add_pass("Scheduler log is recent",
                           f"Last modified: {mtime.strftime('%Y-%m-%d %H:%M')}")
        else:
            results.add_warning("Scheduler log is old",
                              f"Last modified: {mtime.strftime('%Y-%m-%d')} ({age.days} days ago)")
    else:
        results.add_warning("Scheduler log not found",
                          "Log will be created when scheduler runs")


def test_data_freshness():
    """Test that data files are fresh (updated within 24-48 hours)."""
    print("\n=== Testing Data Freshness ===")

    from config.settings import DATA_DIR

    data_files = {
        "predictions": DATA_DIR / "upcoming" / "predictions.json",
        "weather": DATA_DIR / "upcoming" / "weather.json",
        "current_form": DATA_DIR / "upcoming" / "current_form.json",
        "over_under_bets": DATA_DIR / "upcoming" / "over_under_bets.json",
        "handicap_bets": DATA_DIR / "upcoming" / "handicap_bets.json",
    }

    stale_files = []
    missing_files = []

    for name, path in data_files.items():
        if path.exists():
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            age = datetime.now() - mtime

            if age < timedelta(hours=48):
                results.add_pass(f"Data file {name} is fresh",
                               f"Age: {age.seconds // 3600}h")
            else:
                results.add_warning(f"Data file {name} is stale",
                                  f"Age: {age.days}d {age.seconds // 3600}h")
                stale_files.append(name)
        else:
            results.add_warning(f"Data file {name} missing",
                              "Will be created on next run")
            missing_files.append(name)

    if not stale_files and not missing_files:
        results.add_pass("All critical data files are fresh")


def test_health_check_module():
    """Test that health check module works."""
    print("\n=== Testing Health Check Module ===")

    try:
        from scripts.utils.error_handling import health_check, check_feature_availability

        health = health_check()

        if "status" in health:
            status = health["status"]
            results.add_pass(f"Health check returns status: {status}")
        else:
            results.add_fail("Health check", "No status returned")

        features = check_feature_availability()
        available = sum(1 for v in features.values() if v)
        total = len(features)

        if available == total:
            results.add_pass(f"All {total} features available")
        else:
            results.add_warning(f"Features available",
                              f"{available}/{total} features working")

    except ImportError as e:
        results.add_fail("Health check import", str(e))


def test_web_health_endpoints():
    """Test that web health endpoints work."""
    print("\n=== Testing Web Health Endpoints ===")

    try:
        from web.app import app

        with app.test_client() as client:
            # Test /api/health
            response = client.get("/api/health")
            if response.status_code in [200, 503]:
                data = response.get_json()
                if "status" in data:
                    results.add_pass("/api/health endpoint works",
                                   f"Status: {data['status']}")
                else:
                    results.add_fail("/api/health response format",
                                   "Missing status field")
            else:
                results.add_fail("/api/health endpoint",
                               f"Status code: {response.status_code}")

            # Test /api/health/features
            response = client.get("/api/health/features")
            if response.status_code == 200:
                results.add_pass("/api/health/features endpoint works")
            else:
                results.add_fail("/api/health/features endpoint",
                               f"Status code: {response.status_code}")

            # Test /api/health/data-freshness
            response = client.get("/api/health/data-freshness")
            if response.status_code == 200:
                data = response.get_json()
                summary = data.get("summary", {})
                results.add_pass("/api/health/data-freshness endpoint works",
                               f"Status: {summary.get('status', 'unknown')}")
            else:
                results.add_fail("/api/health/data-freshness endpoint",
                               f"Status code: {response.status_code}")

    except Exception as e:
        results.add_fail("Web health endpoints", str(e))


def test_pipeline_components_importable():
    """Test that all pipeline components can be imported."""
    print("\n=== Testing Pipeline Components ===")

    components = [
        ("realtime_prediction_engine", "scripts.realtime_prediction_engine"),
        ("weather_integration", "scripts.prediction.weather_integration"),
        ("referee_integration", "scripts.prediction.referee_integration"),
        ("current_form_calculator", "scripts.prediction.current_form_calculator"),
        ("over_under_model", "scripts.models.over_under_model"),
        ("handicap_model", "scripts.models.handicap_model"),
        ("betting_engine", "scripts.betting_engine"),
        ("bankroll_manager", "features.bankroll_manager"),
        ("error_handling", "scripts.utils.error_handling"),
    ]

    for name, module_path in components:
        try:
            __import__(module_path)
            results.add_pass(f"Component {name} importable")
        except ImportError as e:
            results.add_fail(f"Component {name} import", str(e)[:50])


def test_pipeline_can_run():
    """Test that prediction engine can run without errors."""
    print("\n=== Testing Pipeline Execution ===")

    try:
        from scripts.prediction.predict_unified import run_predictions

        result = run_predictions()

        if result:
            predictions = result.get("predictions", [])
            if predictions:
                results.add_pass("Pipeline produces predictions",
                               f"Generated {len(predictions)} predictions")
            else:
                results.add_warning("Pipeline ran but no predictions",
                                  "May be no upcoming matches")
        else:
            results.add_fail("Pipeline execution", "Returned empty result")

    except Exception as e:
        results.add_fail("Pipeline execution", str(e)[:80])


def test_cron_setup_output():
    """Test that cron setup command works."""
    print("\n=== Testing Cron Setup ===")

    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "scheduler.py"), "cron-setup"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            output = result.stdout
            if "crontab" in output.lower() and "* *" in output:
                results.add_pass("Cron setup command works",
                               "Run `python scheduler.py cron-setup` for details")
            else:
                results.add_warning("Cron setup output unexpected",
                                  "Check scheduler.py cron-setup manually")
        else:
            results.add_fail("Cron setup command", f"Exit code: {result.returncode}")

    except subprocess.TimeoutExpired:
        results.add_fail("Cron setup command", "Timed out")
    except Exception as e:
        results.add_fail("Cron setup command", str(e)[:50])


def test_match_day_detection():
    """Test match day detection works."""
    print("\n=== Testing Match Day Detection ===")

    try:
        from scripts.pipeline.scheduler import is_match_day, get_today_matches

        is_match = is_match_day()
        today_matches = get_today_matches()

        results.add_pass("Match day detection works",
                        f"Today is {'a match day' if is_match else 'not a match day'}")

        if today_matches:
            results.add_pass(f"Found {len(today_matches)} matches today")
        else:
            results.add_pass("No matches scheduled for today")

    except Exception as e:
        results.add_fail("Match day detection", str(e)[:50])


def main():
    print("=" * 70)
    print("AUTOMATION VERIFICATION TESTS")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("\nThis verifies the pipeline automation is set up correctly.")

    test_scheduler_exists()
    test_scheduler_log()
    test_data_freshness()
    test_health_check_module()
    test_web_health_endpoints()
    test_pipeline_components_importable()
    test_pipeline_can_run()
    test_cron_setup_output()
    test_match_day_detection()

    success = results.summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
