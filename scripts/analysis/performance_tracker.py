#!/usr/bin/env python3
"""PERFORMANCE TRACKER - Analytics and Optimization

Comprehensive tracking and analysis:
1. Win rate by confidence level
2. ROI by factor combination
3. Factor performance analysis
4. Optimization recommendations
5. Backtesting validation

Usage:
    python scripts/performance_tracker.py              # Full report
    python scripts/performance_tracker.py --factors    # Factor analysis
    python scripts/performance_tracker.py --optimize   # Optimization recs
    python scripts/performance_tracker.py --backtest   # Run backtest
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# =============================================================================
# DATA LOADING
# =============================================================================

def load_bet_history() -> List[Dict]:
    """Load settled bet history."""
    history_path = DATA_DIR / "betting" / "history.json"

    if history_path.exists():
        with open(history_path) as f:
            return json.load(f)
    return []


def load_validation_data() -> Optional[pd.DataFrame]:
    """Load historical match data for backtesting."""
    if not HAS_PANDAS:
        return None

    features_path = DATA_DIR / "features" / "features.parquet"
    if features_path.exists():
        return pd.read_parquet(features_path)
    return None


# =============================================================================
# PERFORMANCE ANALYSIS
# =============================================================================

def analyze_by_confidence(history: List[Dict]) -> Dict:
    """Analyze performance by confidence level."""
    by_conf = defaultdict(lambda: {"bets": 0, "wins": 0, "staked": 0, "profit": 0})

    for bet in history:
        conf = bet.get("confidence", "UNKNOWN").upper().replace("-", "_").replace(" ", "_")
        by_conf[conf]["bets"] += 1
        by_conf[conf]["staked"] += bet.get("stake", 0)
        by_conf[conf]["profit"] += bet.get("profit", 0)
        if bet.get("status") == "won":
            by_conf[conf]["wins"] += 1

    # Calculate metrics
    result = {}
    for conf, data in by_conf.items():
        if data["bets"] > 0:
            result[conf] = {
                "bets": data["bets"],
                "wins": data["wins"],
                "win_rate": round(data["wins"] / data["bets"] * 100, 1),
                "staked": round(data["staked"], 2),
                "profit": round(data["profit"], 2),
                "roi": round(data["profit"] / data["staked"] * 100, 1) if data["staked"] > 0 else 0,
            }

    return result


def analyze_by_factors(history: List[Dict]) -> Dict:
    """Analyze performance by prediction factors."""
    by_factor = defaultdict(lambda: {"bets": 0, "wins": 0, "profit": 0})

    for bet in history:
        factors = bet.get("factors", [])
        for factor in factors:
            by_factor[factor]["bets"] += 1
            by_factor[factor]["profit"] += bet.get("profit", 0)
            if bet.get("status") == "won":
                by_factor[factor]["wins"] += 1

    # Calculate metrics
    result = {}
    for factor, data in by_factor.items():
        if data["bets"] > 0:
            result[factor] = {
                "bets": data["bets"],
                "wins": data["wins"],
                "win_rate": round(data["wins"] / data["bets"] * 100, 1),
                "profit": round(data["profit"], 2),
            }

    # Sort by win rate
    result = dict(sorted(result.items(), key=lambda x: x[1]["win_rate"], reverse=True))

    return result


def analyze_by_factor_combinations(history: List[Dict]) -> Dict:
    """Analyze performance by factor combinations."""
    by_combo = defaultdict(lambda: {"bets": 0, "wins": 0, "profit": 0})

    for bet in history:
        factors = sorted(bet.get("factors", []))
        if len(factors) >= 2:
            # Create combo key
            combo = " + ".join(factors[:3])  # Max 3 factors for readability
            by_combo[combo]["bets"] += 1
            by_combo[combo]["profit"] += bet.get("profit", 0)
            if bet.get("status") == "won":
                by_combo[combo]["wins"] += 1

    # Calculate metrics and filter
    result = {}
    for combo, data in by_combo.items():
        if data["bets"] >= 3:  # At least 3 occurrences
            result[combo] = {
                "bets": data["bets"],
                "wins": data["wins"],
                "win_rate": round(data["wins"] / data["bets"] * 100, 1),
                "profit": round(data["profit"], 2),
            }

    # Sort by win rate
    result = dict(sorted(result.items(), key=lambda x: x[1]["win_rate"], reverse=True))

    return result


def analyze_by_value_range(history: List[Dict]) -> Dict:
    """Analyze performance by value (edge) ranges."""
    ranges = {
        "0-5%": {"bets": 0, "wins": 0, "profit": 0},
        "5-10%": {"bets": 0, "wins": 0, "profit": 0},
        "10-15%": {"bets": 0, "wins": 0, "profit": 0},
        "15%+": {"bets": 0, "wins": 0, "profit": 0},
    }

    for bet in history:
        value = bet.get("value", 0) * 100 if bet.get("value") else 0

        if value < 5:
            key = "0-5%"
        elif value < 10:
            key = "5-10%"
        elif value < 15:
            key = "10-15%"
        else:
            key = "15%+"

        ranges[key]["bets"] += 1
        ranges[key]["profit"] += bet.get("profit", 0)
        if bet.get("status") == "won":
            ranges[key]["wins"] += 1

    # Calculate metrics
    result = {}
    for range_key, data in ranges.items():
        if data["bets"] > 0:
            result[range_key] = {
                "bets": data["bets"],
                "wins": data["wins"],
                "win_rate": round(data["wins"] / data["bets"] * 100, 1),
                "profit": round(data["profit"], 2),
            }

    return result


def analyze_time_series(history: List[Dict]) -> Dict:
    """Analyze performance over time."""
    if not history:
        return {}

    # Group by week
    by_week = defaultdict(lambda: {"bets": 0, "wins": 0, "profit": 0})

    for bet in history:
        settled = bet.get("settled_at", "")
        if settled:
            try:
                date = datetime.fromisoformat(settled)
                week_start = date - timedelta(days=date.weekday())
                week_key = week_start.strftime("%Y-%m-%d")

                by_week[week_key]["bets"] += 1
                by_week[week_key]["profit"] += bet.get("profit", 0)
                if bet.get("status") == "won":
                    by_week[week_key]["wins"] += 1
            except Exception as e:
                log.debug(f"Failed to parse bet date for weekly stats: {e}")

    # Calculate running totals
    result = {}
    running_profit = 0
    running_bets = 0
    running_wins = 0

    for week in sorted(by_week.keys()):
        data = by_week[week]
        running_profit += data["profit"]
        running_bets += data["bets"]
        running_wins += data["wins"]

        result[week] = {
            "bets": data["bets"],
            "wins": data["wins"],
            "profit": round(data["profit"], 2),
            "running_profit": round(running_profit, 2),
            "running_win_rate": round(running_wins / running_bets * 100, 1) if running_bets > 0 else 0,
        }

    return result


# =============================================================================
# OPTIMIZATION RECOMMENDATIONS
# =============================================================================

def generate_optimization_recs(history: List[Dict]) -> List[Dict]:
    """Generate optimization recommendations based on performance."""
    recommendations = []

    if len(history) < 10:
        recommendations.append({
            "type": "info",
            "message": "Need more bet history (at least 10 settled bets) for meaningful analysis"
        })
        return recommendations

    # Analyze data
    by_conf = analyze_by_confidence(history)
    by_factor = analyze_by_factors(history)
    by_value = analyze_by_value_range(history)

    # Check confidence level performance
    for conf, data in by_conf.items():
        if data["bets"] >= 5:
            if data["roi"] > 10:
                recommendations.append({
                    "type": "positive",
                    "area": "confidence",
                    "message": f"{conf} confidence is performing well ({data['roi']}% ROI) - consider increasing stakes",
                    "data": data
                })
            elif data["roi"] < -10:
                recommendations.append({
                    "type": "warning",
                    "area": "confidence",
                    "message": f"{conf} confidence is underperforming ({data['roi']}% ROI) - consider reducing or skipping",
                    "data": data
                })

    # Check factor performance
    good_factors = []
    bad_factors = []

    for factor, data in by_factor.items():
        if data["bets"] >= 5:
            if data["win_rate"] > 60:
                good_factors.append((factor, data["win_rate"]))
            elif data["win_rate"] < 40:
                bad_factors.append((factor, data["win_rate"]))

    if good_factors:
        recommendations.append({
            "type": "positive",
            "area": "factors",
            "message": f"Strong factors: {', '.join([f'{f[0]} ({f[1]}%)' for f in good_factors[:3]])}",
        })

    if bad_factors:
        recommendations.append({
            "type": "warning",
            "area": "factors",
            "message": f"Weak factors: {', '.join([f'{f[0]} ({f[1]}%)' for f in bad_factors[:3]])}",
        })

    # Check value betting
    for range_key, data in by_value.items():
        if data["bets"] >= 5:
            if data["win_rate"] < 45 and "10" in range_key:
                recommendations.append({
                    "type": "warning",
                    "area": "value",
                    "message": f"Value range {range_key} underperforming - consider raising threshold",
                })

    # Overall recommendations
    total_bets = sum(d["bets"] for d in by_conf.values())
    total_wins = sum(d["wins"] for d in by_conf.values())
    overall_win_rate = total_wins / total_bets * 100 if total_bets > 0 else 0

    if overall_win_rate > 55:
        recommendations.append({
            "type": "positive",
            "area": "overall",
            "message": f"Overall win rate {overall_win_rate:.1f}% is excellent - system is profitable",
        })
    elif overall_win_rate < 45:
        recommendations.append({
            "type": "critical",
            "area": "overall",
            "message": f"Overall win rate {overall_win_rate:.1f}% is concerning - review predictions",
        })

    return recommendations


# =============================================================================
# BACKTESTING
# =============================================================================

def run_backtest(n_months: int = 3) -> Dict:
    """Run backtest on historical data.

    Tests prediction logic against historical results.
    """
    if not HAS_PANDAS:
        return {"error": "Pandas required for backtesting"}

    df = load_validation_data()
    if df is None or len(df) == 0:
        return {"error": "No historical data available"}

    log.info(f"Running backtest on {len(df)} matches...")

    # Filter to recent completed matches
    df = df[df["home_score"].notna()]
    df = df.sort_values("match_date", ascending=False)

    # Take last n months
    cutoff = datetime.now() - timedelta(days=n_months * 30)
    recent = df[df["match_date"] >= str(cutoff.date())]

    if len(recent) < 50:
        return {"error": f"Insufficient data: only {len(recent)} matches"}

    # Define simple prediction rules based on our factors
    results = {
        "total": 0,
        "correct": 0,
        "by_factor_count": defaultdict(lambda: {"total": 0, "correct": 0}),
    }

    for _, row in recent.iterrows():
        # Count factors
        factors = 0
        if row.get("home_elo", 0) - row.get("away_elo", 0) > 100:
            factors += 1  # Home favorite
        if row.get("home_elo", 0) - row.get("away_elo", 0) > 200:
            factors += 1  # Big home favorite
        if row.get("home_form_pts", 0) >= 10:
            factors += 1  # Hot home
        if row.get("away_form_pts", 0) <= 4:
            factors += 1  # Cold away

        # Determine actual result
        home_score = row["home_score"]
        away_score = row["away_score"]

        if home_score > away_score:
            actual = "HOME"
        elif home_score < away_score:
            actual = "AWAY"
        else:
            actual = "DRAW"

        # Our prediction: HOME if 2+ factors
        prediction = "HOME" if factors >= 2 else "OTHER"
        correct = (prediction == "HOME" and actual == "HOME")

        results["total"] += 1
        if correct or (prediction == "OTHER" and actual != "HOME"):
            results["correct"] += 1

        # Track by factor count
        if factors >= 1:
            results["by_factor_count"][factors]["total"] += 1
            if actual == "HOME":
                results["by_factor_count"][factors]["correct"] += 1

    # Calculate accuracy
    results["accuracy"] = round(results["correct"] / results["total"] * 100, 1) if results["total"] > 0 else 0

    # Factor count analysis
    factor_analysis = {}
    for count, data in results["by_factor_count"].items():
        if data["total"] > 0:
            factor_analysis[f"{count}+ factors"] = {
                "matches": data["total"],
                "home_wins": data["correct"],
                "home_win_rate": round(data["correct"] / data["total"] * 100, 1),
            }

    return {
        "period": f"Last {n_months} months",
        "matches_analyzed": results["total"],
        "overall_accuracy": results["accuracy"],
        "factor_analysis": factor_analysis,
    }


# =============================================================================
# REPORTING
# =============================================================================

def print_full_report():
    """Print comprehensive performance report."""
    history = load_bet_history()

    print("\n" + "=" * 80)
    print(" PERFORMANCE TRACKER - FULL REPORT")
    print("=" * 80)

    if not history:
        print("\n  No bet history yet. Place and settle some bets first.")
        return

    # Overall stats
    total_bets = len(history)
    total_wins = len([b for b in history if b.get("status") == "won"])
    total_profit = sum(b.get("profit", 0) for b in history)
    total_staked = sum(b.get("stake", 0) for b in history)

    print(f"\n  OVERALL PERFORMANCE")
    print(f"  {'-' * 40}")
    print(f"  Total Bets:    {total_bets}")
    print(f"  Win Rate:      {total_wins / total_bets * 100:.1f}%")
    print(f"  Total Staked:  ${total_staked:.2f}")
    print(f"  Total Profit:  ${total_profit:.2f}")
    print(f"  ROI:           {total_profit / total_staked * 100:.1f}%" if total_staked > 0 else "  ROI: N/A")

    # By confidence
    print(f"\n  BY CONFIDENCE LEVEL")
    print(f"  {'-' * 40}")
    by_conf = analyze_by_confidence(history)
    for conf, data in sorted(by_conf.items(), key=lambda x: x[1].get("roi", 0), reverse=True):
        print(f"  {conf}: {data['win_rate']}% win rate | {data['roi']}% ROI | {data['bets']} bets")

    # By factors
    print(f"\n  BY FACTOR")
    print(f"  {'-' * 40}")
    by_factor = analyze_by_factors(history)
    for factor, data in list(by_factor.items())[:10]:
        print(f"  {factor}: {data['win_rate']}% | {data['bets']} bets | ${data['profit']:.2f}")

    # Time series
    print(f"\n  WEEKLY PERFORMANCE")
    print(f"  {'-' * 40}")
    by_time = analyze_time_series(history)
    for week, data in list(by_time.items())[-5:]:  # Last 5 weeks
        print(f"  {week}: {data['bets']} bets | ${data['profit']:.2f} | Running: ${data['running_profit']:.2f}")


def print_optimization_recs():
    """Print optimization recommendations."""
    history = load_bet_history()

    print("\n" + "=" * 80)
    print(" OPTIMIZATION RECOMMENDATIONS")
    print("=" * 80)

    recs = generate_optimization_recs(history)

    if not recs:
        print("\n  No recommendations available.")
        return

    for rec in recs:
        icon = {
            "positive": "",
            "warning": "",
            "critical": "",
            "info": "",
        }.get(rec.get("type"), "")

        print(f"\n  {icon} [{rec.get('area', 'general').upper()}]")
        print(f"     {rec.get('message')}")


def print_backtest_results():
    """Print backtest results."""
    print("\n" + "=" * 80)
    print(" BACKTEST RESULTS")
    print("=" * 80)

    results = run_backtest(n_months=3)

    if "error" in results:
        print(f"\n  Error: {results['error']}")
        return

    print(f"\n  Period: {results['period']}")
    print(f"  Matches Analyzed: {results['matches_analyzed']}")
    print(f"  Overall Accuracy: {results['overall_accuracy']}%")

    print(f"\n  Factor Count Analysis:")
    for key, data in results.get("factor_analysis", {}).items():
        print(f"    {key}: {data['home_win_rate']}% home win rate ({data['matches']} matches)")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Performance Tracker")
    parser.add_argument("--factors", action="store_true", help="Show factor analysis")
    parser.add_argument("--optimize", action="store_true", help="Show optimization recommendations")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    if args.factors:
        history = load_bet_history()
        by_factor = analyze_by_factors(history)
        if args.json:
            print(json.dumps(by_factor, indent=2))
        else:
            print("\nFACTOR PERFORMANCE:")
            for factor, data in by_factor.items():
                print(f"  {factor}: {data['win_rate']}% | {data['bets']} bets")

    elif args.optimize:
        print_optimization_recs()

    elif args.backtest:
        print_backtest_results()

    else:
        print_full_report()


if __name__ == "__main__":
    main()
