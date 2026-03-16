#!/usr/bin/env python3
"""MASTER ORCHESTRATOR - Complete Betting AI Pipeline

Single command to run the entire betting system:
1. Scrape fixtures (or load manual data)
2. Scrape/load referee assignments
3. Fetch weather forecasts
4. Calculate team form
5. Generate predictions
6. Calculate value and stakes
7. Generate betting slip
8. Send alerts
9. Update bankroll

This is the PRODUCTION entry point.

Usage:
    python scripts/run_betting_system.py                    # Full pipeline
    python scripts/run_betting_system.py --quick            # Skip weather
    python scripts/run_betting_system.py --predictions-only # Just predictions
    python scripts/run_betting_system.py --alerts           # Check alerts only
    python scripts/run_betting_system.py --status           # Show status

Schedule this to run daily for automated betting:
    # crontab -e
    # 0 8 * * * cd /path/to/seriea_pipeline && python scripts/run_betting_system.py
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env for API keys
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

from config.settings import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(DATA_DIR / "betting" / "system.log")
    ]
)
log = logging.getLogger(__name__)


def ensure_directories():
    """Ensure all required directories exist."""
    dirs = [
        DATA_DIR / "upcoming",
        DATA_DIR / "betting",
        DATA_DIR / "models",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def step_1_fixtures():
    """Step 1: Load/scrape fixtures."""
    log.info("=" * 60)
    log.info("STEP 1: LOADING FIXTURES")
    log.info("=" * 60)

    try:
        from scripts.data.fixture_scraper import scrape_fixtures
        result = scrape_fixtures()
        matches = result.get("matches", [])
        log.info(f"  Loaded {len(matches)} upcoming matches")
        return len(matches) > 0
    except Exception as e:
        log.error(f"  Failed to load fixtures: {e}")
        return False


def step_2_referees():
    """Step 2: Load/scrape referee assignments."""
    log.info("")
    log.info("=" * 60)
    log.info("STEP 2: LOADING REFEREES")
    log.info("=" * 60)

    try:
        from scripts.data.referee_scraper import get_matchweek_referees, update_referees_file
        assignments = get_matchweek_referees()
        all_refs = update_referees_file(assignments)
        assigned = len([v for v in all_refs.values() if v])
        log.info(f"  {assigned}/{len(all_refs)} matches have referee assignments")
        return True
    except Exception as e:
        log.error(f"  Failed to load referees: {e}")
        return False


def step_2b_odds():
    """Step 2b: Fetch real bookmaker odds."""
    log.info("")
    log.info("=" * 60)
    log.info("STEP 2b: FETCHING REAL ODDS")
    log.info("=" * 60)

    try:
        from scripts.data.odds_fetcher import fetch_and_save_odds, check_api_key

        if not check_api_key():
            log.warning("  ODDS_API_KEY not set - using manual odds if available")
            log.warning("  Get free API key at: https://the-odds-api.com/")
            return False

        odds_data = fetch_and_save_odds()
        if odds_data:
            log.info(f"  Fetched real odds for {len(odds_data)} matches")
            return True
        else:
            log.warning("  No odds data received")
            return False
    except Exception as e:
        log.warning(f"  Failed to fetch odds: {e}")
        return False


def step_3_predictions(skip_weather: bool = False):
    """Step 3: Generate predictions."""
    log.info("")
    log.info("=" * 60)
    log.info("STEP 3: GENERATING PREDICTIONS")
    log.info("=" * 60)

    try:
        from scripts.prediction.ensemble_prediction_engine import run_ensemble_predictions

        output = run_ensemble_predictions(use_ensemble=True)

        if not output:
            log.error("  Ensemble returned no output")
            return False

        predictions = output.get("predictions", [])
        methods = output.get("methods_available", ["factor"])
        high_conf = len([p for p in predictions
                        if p.get("confidence_level", p.get("confidence")) in ["VERY_HIGH", "VERY HIGH", "HIGH"]])
        log.info(f"  Generated {len(predictions)} ensemble predictions ({high_conf} high confidence)")
        log.info(f"  Methods: {', '.join(methods)}")
        return len(predictions) > 0
    except Exception as e:
        log.warning(f"  Ensemble failed ({e}), falling back to factor-only")
        try:
            from scripts.realtime_prediction_engine import run_predictions
            output = run_predictions()
            predictions = output.get("predictions", [])
            log.info(f"  Generated {len(predictions)} factor-only predictions (fallback)")
            return len(predictions) > 0
        except Exception as e2:
            log.error(f"  Failed to generate predictions: {e2}")
            import traceback
            traceback.print_exc()
            return False


def step_4_betting_slip():
    """Step 4: Generate betting slip with stakes."""
    log.info("")
    log.info("=" * 60)
    log.info("STEP 4: GENERATING BETTING SLIP")
    log.info("=" * 60)

    try:
        from scripts.betting_engine import generate_betting_slip
        slip = generate_betting_slip(include_parlays=True)

        if "error" in slip:
            log.error(f"  Betting slip error: {slip['error']}")
            return False

        summary = slip.get("summary", {})
        log.info(f"  Recommended bets: {summary.get('recommended_bets', 0)}")
        log.info(f"  Consider bets: {summary.get('consider_bets', 0)}")
        log.info(f"  Total stake: ${summary.get('total_stake', 0):.2f}")
        return True
    except Exception as e:
        log.error(f"  Failed to generate betting slip: {e}")
        import traceback
        traceback.print_exc()
        return False


def step_5_alerts():
    """Step 5: Check and send alerts."""
    log.info("")
    log.info("=" * 60)
    log.info("STEP 5: CHECKING ALERTS")
    log.info("=" * 60)

    try:
        from scripts.utils.alert_system import check_and_send_alerts
        check_and_send_alerts()
        return True
    except Exception as e:
        log.error(f"  Failed to check alerts: {e}")
        return False


def step_6_status():
    """Step 6: Show final status."""
    log.info("")
    log.info("=" * 60)
    log.info("STEP 6: SYSTEM STATUS")
    log.info("=" * 60)

    try:
        from scripts.bankroll_manager import load_bankroll, load_bets

        bankroll = load_bankroll()
        bets = load_bets()

        log.info(f"  Bankroll: ${bankroll.get('current_balance', 0):.2f}")
        log.info(f"  Pending bets: {len(bets)}")

        profit = bankroll.get('current_balance', 0) - bankroll.get('initial_balance', 0)
        roi = (profit / bankroll.get('initial_balance', 1)) * 100
        log.info(f"  Total P/L: ${profit:.2f} ({roi:+.1f}%)")

        return True
    except Exception as e:
        log.error(f"  Failed to get status: {e}")
        return False


def run_full_pipeline(skip_weather: bool = False):
    """Run the complete betting pipeline."""
    log.info("")
    log.info("*" * 60)
    log.info("* SERIE A BETTING AI - FULL PIPELINE")
    log.info(f"* Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("*" * 60)

    ensure_directories()

    results = {}

    # Step 1: Fixtures
    results["fixtures"] = step_1_fixtures()

    if not results["fixtures"]:
        log.error("Cannot continue without fixtures")
        return results

    # Step 2: Referees
    results["referees"] = step_2_referees()

    # Step 2b: Real odds (optional but recommended)
    results["odds"] = step_2b_odds()

    # Step 3: Predictions
    results["predictions"] = step_3_predictions(skip_weather=skip_weather)

    if not results["predictions"]:
        log.error("Cannot continue without predictions")
        return results

    # Step 4: Betting slip
    results["betting_slip"] = step_4_betting_slip()

    # Step 5: Alerts
    results["alerts"] = step_5_alerts()

    # Step 6: Status
    results["status"] = step_6_status()

    # Summary
    log.info("")
    log.info("*" * 60)
    log.info("* PIPELINE COMPLETE")
    log.info("*" * 60)

    success = all(results.values())
    log.info(f"  All steps successful: {success}")

    for step, status in results.items():
        icon = "" if status else ""
        log.info(f"  {icon} {step}")

    # Save run log
    run_log = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "success": success,
    }

    log_path = DATA_DIR / "betting" / "last_run.json"
    with open(log_path, "w") as f:
        json.dump(run_log, f, indent=2)

    return results


def show_quick_status():
    """Show quick system status."""
    print("\n" + "=" * 60)
    print(" SERIE A BETTING AI - STATUS")
    print("=" * 60)

    # Bankroll
    try:
        from scripts.bankroll_manager import load_bankroll, load_bets
        bankroll = load_bankroll()
        bets = load_bets()
        print(f"\n  Bankroll: ${bankroll.get('current_balance', 0):.2f}")
        print(f"  Pending bets: {len(bets)}")
    except Exception:
        print("  Bankroll: Not initialized")

    # Last run
    log_path = DATA_DIR / "betting" / "last_run.json"
    if log_path.exists():
        with open(log_path) as f:
            last_run = json.load(f)
        print(f"\n  Last run: {last_run.get('timestamp', 'Unknown')}")
        print(f"  Status: {'SUCCESS' if last_run.get('success') else 'FAILED'}")
    else:
        print("\n  No previous runs")

    # Predictions
    pred_path = DATA_DIR / "upcoming" / "predictions.json"
    if pred_path.exists():
        with open(pred_path) as f:
            preds = json.load(f)
        predictions = preds.get("predictions", [])
        high_conf = len([p for p in predictions if p.get("confidence") in ["VERY_HIGH", "VERY HIGH", "HIGH"]])
        print(f"\n  Current predictions: {len(predictions)}")
        print(f"  High confidence: {high_conf}")
    else:
        print("\n  No predictions generated")

    # Betting slip
    slip_path = DATA_DIR / "betting" / "betting_slip.json"
    if slip_path.exists():
        with open(slip_path) as f:
            slip = json.load(f)
        summary = slip.get("summary", {})
        print(f"\n  Recommended bets: {summary.get('recommended_bets', 0)}")
        print(f"  Total stake: ${summary.get('total_stake', 0):.2f}")
    else:
        print("\n  No betting slip")


def show_betting_slip():
    """Show the current betting slip."""
    try:
        from scripts.betting_engine import generate_betting_slip, print_betting_slip
        slip = generate_betting_slip(include_parlays=True)
        print_betting_slip(slip)
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Serie A Betting AI - Master Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_betting_system.py           # Run full pipeline
  python scripts/run_betting_system.py --quick   # Skip weather (faster)
  python scripts/run_betting_system.py --status  # Show status only
  python scripts/run_betting_system.py --slip    # Show betting slip

Schedule for automation:
  # crontab -e
  # 0 8 * * * cd /path/to/seriea_pipeline && python scripts/run_betting_system.py
        """
    )

    parser.add_argument("--quick", action="store_true",
                       help="Skip weather fetching (faster)")
    parser.add_argument("--predictions-only", action="store_true",
                       help="Only run predictions, skip betting logic")
    parser.add_argument("--alerts", action="store_true",
                       help="Only check and send alerts")
    parser.add_argument("--status", action="store_true",
                       help="Show system status only")
    parser.add_argument("--slip", action="store_true",
                       help="Show current betting slip")
    parser.add_argument("--backtest", action="store_true",
                       help="Run backtest")

    args = parser.parse_args()

    if args.status:
        show_quick_status()

    elif args.slip:
        show_betting_slip()

    elif args.alerts:
        step_5_alerts()

    elif args.predictions_only:
        ensure_directories()
        step_1_fixtures()
        step_2_referees()
        step_3_predictions(skip_weather=args.quick)

    elif args.backtest:
        from scripts.analysis.performance_tracker import print_backtest_results
        print_backtest_results()

    else:
        results = run_full_pipeline(skip_weather=args.quick)
        return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
