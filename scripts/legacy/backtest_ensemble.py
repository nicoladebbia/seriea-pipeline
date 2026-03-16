#!/usr/bin/env python3
"""Walk-Forward Backtesting Framework - Phase 1.4

Implements proper walk-forward validation for realistic accuracy estimation.

Usage:
    python scripts/backtest_ensemble.py
    python scripts/backtest_ensemble.py --seasons 2020-2021 2021-2022 2022-2023 2023-2024 2024-2025
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR, MODELS_DIR
from scripts.prediction.ensemble_prediction_engine import EnsemblePredictor, ENSEMBLE_WEIGHTS

try:
    from features.prediction_calibration import BETTING_STRATEGIES
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class WalkForwardBacktester:
    """Walk-forward backtesting with season-by-season validation."""

    def __init__(self, strategy: str = "default"):
        self.df = None
        self.ensemble = None
        self.results = []
        self.strategy = strategy

    def load_data(self) -> bool:
        """Load feature data."""
        features_path = DATA_DIR / "features" / "features.parquet"
        if not features_path.exists():
            log.error(f"Features not found: {features_path}")
            return False

        self.df = pd.read_parquet(features_path)
        self.df = self.df.sort_values("match_date").reset_index(drop=True)
        self.df = self.df.dropna(subset=["result", "home_score", "away_score"])

        log.info(f"Loaded {len(self.df)} matches")
        log.info(f"Seasons: {sorted(self.df['season'].unique())}")
        return True

    def initialize_ensemble(self) -> bool:
        """Initialize ensemble predictor."""
        self.ensemble = EnsemblePredictor(strategy=self.strategy, live_mode=False)
        return self.ensemble.initialize()

    def backtest_season(self, test_season: str) -> Dict:
        """Backtest on a single season with Brier/RPS tracking."""
        log.info(f"Backtesting season: {test_season}")

        test_df = self.df[self.df["season"] == test_season].copy()
        if len(test_df) == 0:
            return {"error": f"No matches for {test_season}"}

        correct = 0
        high_conf_correct = 0
        high_conf_total = 0
        total = 0
        predictions_log = []
        brier_scores = []
        rps_scores = []

        result_map = {"H": "HOME", "D": "DRAW", "A": "AWAY"}
        outcome_to_idx = {"HOME": 0, "DRAW": 1, "AWAY": 2}

        for idx, row in test_df.iterrows():
            actual = row["result"]
            actual_label = result_map.get(actual, actual)

            # Create match dict
            match = {
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "match_date": str(row["match_date"]),
            }

            # Simplified factors
            factors = {}

            # Form data
            form_data = {
                "home": {},
                "away": {},
            }

            # Get prediction
            try:
                result = self.ensemble.predict(match, factors, form_data)
                if result is None:
                    continue

                predicted = result["predicted_outcome"]
                probs = result["probabilities"]
                max_prob = result["confidence"]

                # Check accuracy
                is_correct = predicted == actual_label
                if is_correct:
                    correct += 1

                # High confidence
                if max_prob >= 0.50:
                    high_conf_total += 1
                    if is_correct:
                        high_conf_correct += 1

                total += 1

                # Compute per-prediction Brier score and RPS
                actual_idx = outcome_to_idx.get(actual_label, 0)
                prob_vec = np.array([
                    probs.get("home", probs.get("HOME", 0.33)),
                    probs.get("draw", probs.get("DRAW", 0.33)),
                    probs.get("away", probs.get("AWAY", 0.34)),
                ])
                actual_vec = np.zeros(3)
                actual_vec[actual_idx] = 1.0

                brier = float(np.sum((prob_vec - actual_vec) ** 2))
                brier_scores.append(brier)

                # RPS (ordinal cumulative)
                cum_pred = np.cumsum(prob_vec)
                cum_true = np.cumsum(actual_vec)
                rps = float(np.mean((cum_pred - cum_true) ** 2))
                rps_scores.append(rps)

                predictions_log.append({
                    "match": f"{row['home_team']} vs {row['away_team']}",
                    "predicted": predicted,
                    "actual": actual_label,
                    "correct": is_correct,
                    "confidence": max_prob,
                    "probs": probs,
                    "brier": round(brier, 4),
                    "rps": round(rps, 4),
                })

            except Exception as e:
                log.debug(f"Prediction failed for {row['home_team']} vs {row['away_team']}: {e}")
                continue

        if total == 0:
            return {"error": "No successful predictions"}

        return {
            "season": test_season,
            "total_matches": total,
            "correct": correct,
            "accuracy": correct / total,
            "mean_brier": round(np.mean(brier_scores), 4) if brier_scores else None,
            "mean_rps": round(np.mean(rps_scores), 4) if rps_scores else None,
            "high_conf_total": high_conf_total,
            "high_conf_correct": high_conf_correct,
            "high_conf_accuracy": high_conf_correct / high_conf_total if high_conf_total > 0 else 0,
            "predictions": predictions_log,
        }

    def run_walk_forward(self, test_seasons: List[str] = None) -> Dict:
        """Run walk-forward validation across multiple seasons."""
        if test_seasons is None:
            # Use recent seasons for testing
            all_seasons = sorted(self.df["season"].unique())
            test_seasons = all_seasons[-3:]  # Last 3 seasons

        log.info(f"Running walk-forward backtest on: {test_seasons}")

        results = {
            "timestamp": datetime.now().isoformat(),
            "test_seasons": test_seasons,
            "season_results": [],
        }

        total_correct = 0
        total_matches = 0
        total_high_conf_correct = 0
        total_high_conf = 0

        for season in test_seasons:
            season_result = self.backtest_season(season)

            if "error" not in season_result:
                results["season_results"].append(season_result)
                total_correct += season_result["correct"]
                total_matches += season_result["total_matches"]
                total_high_conf_correct += season_result["high_conf_correct"]
                total_high_conf += season_result["high_conf_total"]

                log.info(f"  {season}: {season_result['accuracy']:.1%} "
                        f"(HC: {season_result['high_conf_accuracy']:.1%})")

        # Overall results
        results["overall"] = {
            "total_matches": total_matches,
            "correct": total_correct,
            "accuracy": total_correct / total_matches if total_matches > 0 else 0,
            "high_conf_total": total_high_conf,
            "high_conf_correct": total_high_conf_correct,
            "high_conf_accuracy": total_high_conf_correct / total_high_conf if total_high_conf > 0 else 0,
        }

        # Analyze by confidence bucket
        results["by_confidence"] = self._analyze_by_confidence(results)

        # Analyze by outcome
        results["by_outcome"] = self._analyze_by_outcome(results)

        return results

    def _analyze_by_confidence(self, results: Dict) -> Dict:
        """Analyze accuracy by confidence level."""
        buckets = {
            "low": {"min": 0.33, "max": 0.45, "correct": 0, "total": 0},
            "medium": {"min": 0.45, "max": 0.55, "correct": 0, "total": 0},
            "high": {"min": 0.55, "max": 0.65, "correct": 0, "total": 0},
            "very_high": {"min": 0.65, "max": 1.0, "correct": 0, "total": 0},
        }

        for season_result in results.get("season_results", []):
            for pred in season_result.get("predictions", []):
                conf = pred["confidence"]
                for bucket_name, bucket in buckets.items():
                    if bucket["min"] <= conf < bucket["max"]:
                        bucket["total"] += 1
                        if pred["correct"]:
                            bucket["correct"] += 1
                        break

        return {
            name: {
                "accuracy": b["correct"] / b["total"] if b["total"] > 0 else 0,
                "count": b["total"],
                "correct": b["correct"],
            }
            for name, b in buckets.items()
        }

    def _analyze_by_outcome(self, results: Dict) -> Dict:
        """Analyze accuracy by predicted outcome."""
        outcomes = {"HOME": {"correct": 0, "total": 0},
                   "DRAW": {"correct": 0, "total": 0},
                   "AWAY": {"correct": 0, "total": 0}}

        for season_result in results.get("season_results", []):
            for pred in season_result.get("predictions", []):
                predicted = pred["predicted"]
                if predicted in outcomes:
                    outcomes[predicted]["total"] += 1
                    if pred["correct"]:
                        outcomes[predicted]["correct"] += 1

        return {
            outcome: {
                "accuracy": data["correct"] / data["total"] if data["total"] > 0 else 0,
                "count": data["total"],
            }
            for outcome, data in outcomes.items()
        }

    def calculate_betting_roi(self, results: Dict) -> Dict:
        """Calculate betting ROI using actual historical odds from features data.

        Uses odds columns from the feature matrix (odds_home, odds_draw, odds_away)
        which come from real bookmaker data. Falls back to market-implied odds from
        class frequencies if historical odds are unavailable.
        """
        # Check if we have real odds in the DataFrame
        has_real_odds = (
            self.df is not None
            and "odds_home" in self.df.columns
            and "odds_draw" in self.df.columns
            and "odds_away" in self.df.columns
        )

        if has_real_odds:
            odds_source = "historical_bookmaker"
            log.info("Using real historical bookmaker odds for betting simulation")
        else:
            odds_source = "class_frequency_implied"
            log.warning("No historical odds found — using class-frequency-implied odds")

        total_stake = 0.0
        total_returns = 0.0
        bets_placed = 0
        cumulative_pnl = []

        for season_result in results.get("season_results", []):
            season = season_result.get("season", "")
            season_df = self.df[self.df["season"] == season] if self.df is not None else None

            for pred in season_result.get("predictions", []):
                # Only bet on high confidence
                if pred["confidence"] < 0.55:
                    continue

                predicted = pred["predicted"]

                # Get real odds for this match
                match_odds = None
                if has_real_odds and season_df is not None:
                    match_name = pred.get("match", "")
                    parts = match_name.split(" vs ")
                    if len(parts) == 2:
                        home, away = parts[0].strip(), parts[1].strip()
                        match_row = season_df[
                            (season_df["home_team"] == home) & (season_df["away_team"] == away)
                        ]
                        if not match_row.empty:
                            row = match_row.iloc[0]
                            match_odds = {
                                "HOME": row.get("odds_home", np.nan),
                                "DRAW": row.get("odds_draw", np.nan),
                                "AWAY": row.get("odds_away", np.nan),
                            }

                # Fallback to class-frequency-implied odds
                if match_odds is None or np.isnan(match_odds.get(predicted, np.nan)):
                    # H≈44%, D≈27%, A≈29% -> with 7% overround
                    match_odds = {"HOME": 2.10, "DRAW": 3.45, "AWAY": 3.20}

                odds = match_odds.get(predicted, 2.50)
                if odds <= 1.0:
                    continue  # invalid odds

                total_stake += 1
                bets_placed += 1

                if pred["correct"]:
                    total_returns += odds

                cumulative_pnl.append(round(total_returns - total_stake, 2))

        roi = ((total_returns - total_stake) / total_stake * 100) if total_stake > 0 else 0

        return {
            "bets_placed": bets_placed,
            "total_stake": round(total_stake, 2),
            "total_returns": round(total_returns, 2),
            "profit": round(total_returns - total_stake, 2),
            "roi_percent": round(roi, 2),
            "odds_source": odds_source,
            "cumulative_pnl_curve": cumulative_pnl[-20:] if len(cumulative_pnl) > 20 else cumulative_pnl,
        }


def main():
    parser = argparse.ArgumentParser(description="Walk-forward backtest ensemble")
    parser.add_argument("--seasons", nargs="+", help="Seasons to test")
    parser.add_argument("--strategy", default="default",
                        choices=["default", "volume", "selective", "ml_heavy", "xg_dominant"],
                        help="Betting strategy to use")
    args = parser.parse_args()

    backtester = WalkForwardBacktester(strategy=args.strategy)
    log.info(f"Using betting strategy: {args.strategy}")

    if not backtester.load_data():
        return

    if not backtester.initialize_ensemble():
        log.error("Failed to initialize ensemble")
        return

    # Run backtest
    results = backtester.run_walk_forward(args.seasons)

    # Calculate ROI
    roi_results = backtester.calculate_betting_roi(results)
    results["betting_simulation"] = roi_results

    # Add reproducibility metadata
    from ml.statistical_validation import get_reproducibility_metadata
    results["reproducibility"] = get_reproducibility_metadata()
    results["reproducibility"]["strategy"] = args.strategy

    # Print results
    print("\n" + "=" * 70)
    print("WALK-FORWARD BACKTEST RESULTS")
    print("=" * 70)

    print(f"\nTest Seasons: {results['test_seasons']}")
    print(f"Odds Source:  {roi_results.get('odds_source', 'unknown')}")

    # Season-by-season
    print("\nSEASON-BY-SEASON RESULTS")
    print("-" * 50)
    for sr in results["season_results"]:
        brier_str = f", Brier={sr['mean_brier']:.4f}" if sr.get("mean_brier") else ""
        rps_str = f", RPS={sr['mean_rps']:.4f}" if sr.get("mean_rps") else ""
        print(f"  {sr['season']}: {sr['accuracy']:.1%} accuracy "
              f"({sr['correct']}/{sr['total_matches']}{brier_str}{rps_str})")
        print(f"    High-conf: {sr['high_conf_accuracy']:.1%} "
              f"({sr['high_conf_correct']}/{sr['high_conf_total']})")

    # Overall
    overall = results["overall"]
    print("\nOVERALL RESULTS")
    print("-" * 50)
    print(f"  Total Matches: {overall['total_matches']}")
    print(f"  Overall Accuracy: {overall['accuracy']:.1%}")
    print(f"  High-Conf Accuracy: {overall['high_conf_accuracy']:.1%}")

    # Pooled accuracy with CI
    from ml.statistical_validation import accuracy_with_ci, binomial_test, BASE_RATE_HOME_WIN
    pooled = accuracy_with_ci(overall["correct"], overall["total_matches"])
    print(f"  95% CI: {pooled['ci_lo']:.1%} - {pooled['ci_hi']:.1%} [{pooled['reliability']}]")
    sig = binomial_test(overall["correct"], overall["total_matches"], BASE_RATE_HOME_WIN)
    print(f"  vs Always-Home (44.4%): p={sig['p_value']:.6f}")

    # By confidence
    print("\nACCURACY BY CONFIDENCE LEVEL")
    print("-" * 50)
    for level, data in results["by_confidence"].items():
        print(f"  {level.upper():12} {data['accuracy']:.1%} ({data['count']} predictions)")

    # By outcome
    print("\nACCURACY BY PREDICTED OUTCOME")
    print("-" * 50)
    for outcome, data in results["by_outcome"].items():
        print(f"  {outcome:12} {data['accuracy']:.1%} ({data['count']} predictions)")

    # Betting simulation
    print(f"\nBETTING SIMULATION (High-Conf Only, Flat Stakes, {roi_results.get('odds_source', 'unknown')} odds)")
    print("-" * 50)
    print(f"  Bets Placed: {roi_results['bets_placed']}")
    print(f"  ROI: {roi_results['roi_percent']:+.1f}%")
    print(f"  Profit: {roi_results['profit']:+.2f} units")

    # Reproducibility
    meta = results["reproducibility"]
    print(f"\nREPRODUCIBILITY")
    print("-" * 50)
    print(f"  Git:    {meta.get('git_hash', 'unknown')}")
    print(f"  Seed:   {meta.get('random_seed', 'N/A')}")
    print(f"  Python: {meta.get('python_version', 'N/A')}")

    # Save results
    results_dir = DATA_DIR / "optimization"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Remove detailed predictions for smaller file
    save_results = {k: v for k, v in results.items()}
    for sr in save_results.get("season_results", []):
        sr.pop("predictions", None)

    with open(results_file, "w") as f:
        json.dump(save_results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
