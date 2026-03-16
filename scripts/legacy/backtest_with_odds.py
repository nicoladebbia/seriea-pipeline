#!/usr/bin/env python3
"""Walk-Forward Backtesting with Real Odds - Phase 3.4

Uses historical bookmaker odds (Pinnacle, Bet365) for realistic ROI calculation.
Implements Kelly Criterion bet sizing and value betting detection.

Usage:
    python scripts/backtest_with_odds.py
    python scripts/backtest_with_odds.py --strategy selective --kelly 0.25
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
from scripts.prediction.ensemble_prediction_engine import EnsemblePredictor
from features.value_betting import ValueBettingPipeline, KellyCriterion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class OddsBacktester:
    """Backtester using real historical odds."""

    def __init__(self, strategy: str = "default", kelly_fraction: float = 0.25):
        self.df = None
        self.ensemble = None
        self.strategy = strategy
        self.kelly_fraction = kelly_fraction
        self.kelly = KellyCriterion(fraction=kelly_fraction)
        self.value_pipeline = ValueBettingPipeline(
            kelly_fraction=kelly_fraction,
            value_mode="standard",
        )

    def load_data(self) -> bool:
        """Load feature data with odds."""
        features_path = DATA_DIR / "features" / "features.parquet"
        if not features_path.exists():
            log.error(f"Features not found: {features_path}")
            return False

        self.df = pd.read_parquet(features_path)
        self.df = self.df.sort_values("match_date").reset_index(drop=True)
        self.df = self.df.dropna(subset=["result", "home_score", "away_score"])

        # Check for odds columns
        odds_cols = ["odds_PSH", "odds_PSD", "odds_PSA", "odds_B365H", "odds_B365D", "odds_B365A"]
        available_odds = [c for c in odds_cols if c in self.df.columns]

        log.info(f"Loaded {len(self.df)} matches")
        log.info(f"Available odds columns: {available_odds}")

        # Use Pinnacle odds if available, else Bet365
        if "odds_PSH" in self.df.columns:
            self.df["home_odds"] = self.df["odds_PSH"]
            self.df["draw_odds"] = self.df["odds_PSD"]
            self.df["away_odds"] = self.df["odds_PSA"]
            log.info("Using Pinnacle odds")
        elif "odds_B365H" in self.df.columns:
            self.df["home_odds"] = self.df["odds_B365H"]
            self.df["draw_odds"] = self.df["odds_B365D"]
            self.df["away_odds"] = self.df["odds_B365A"]
            log.info("Using Bet365 odds")
        else:
            log.warning("No odds data available, using defaults")
            self.df["home_odds"] = 1.80
            self.df["draw_odds"] = 3.40
            self.df["away_odds"] = 3.60

        # Count matches with valid odds
        has_odds = self.df[["home_odds", "draw_odds", "away_odds"]].notna().all(axis=1)
        log.info(f"Matches with valid odds: {has_odds.sum()}")

        return True

    def initialize_ensemble(self) -> bool:
        """Initialize ensemble predictor."""
        self.ensemble = EnsemblePredictor(strategy=self.strategy, live_mode=False)
        return self.ensemble.initialize()

    def backtest_season(self, test_season: str) -> Dict:
        """Backtest on a single season with real odds."""
        log.info(f"Backtesting season: {test_season}")

        test_df = self.df[self.df["season"] == test_season].copy()
        if len(test_df) == 0:
            return {"error": f"No matches for {test_season}"}

        # Filter to matches with valid odds
        test_df = test_df[test_df[["home_odds", "draw_odds", "away_odds"]].notna().all(axis=1)]

        if len(test_df) == 0:
            return {"error": f"No matches with odds for {test_season}"}

        result_map = {"H": "HOME", "D": "DRAW", "A": "AWAY"}

        # Tracking
        predictions_log = []
        correct = 0
        total = 0

        # Betting simulation
        bankroll = 1000.0
        initial_bankroll = bankroll
        bets_placed = 0
        value_bets = 0
        value_wins = 0

        # Flat betting
        flat_total_stake = 0
        flat_total_returns = 0

        # Kelly betting
        kelly_total_stake = 0
        kelly_total_returns = 0

        for idx, row in test_df.iterrows():
            actual = row["result"]
            actual_label = result_map.get(actual, actual)

            # Get odds
            odds = {
                "home": float(row["home_odds"]),
                "draw": float(row["draw_odds"]),
                "away": float(row["away_odds"]),
            }

            # Create match dict
            match = {
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "match_date": str(row["match_date"]),
            }

            # Get prediction
            try:
                result = self.ensemble.predict(match, {}, {})
                if result is None:
                    continue

                predicted = result["predicted_outcome"]
                probs = {
                    "home": result["probabilities"]["home"],
                    "draw": result["probabilities"]["draw"],
                    "away": result["probabilities"]["away"],
                }
                max_prob = result["confidence"]

                # Check accuracy
                is_correct = predicted == actual_label
                if is_correct:
                    correct += 1
                total += 1

                # Value betting analysis
                value_analysis = self.value_pipeline.analyze_bet(
                    row["home_team"], row["away_team"],
                    probs, odds, bankroll=bankroll
                )

                has_value = value_analysis["value_analysis"]["has_value"]
                best_bet = value_analysis["recommendation"]["best_bet"]

                # Flat betting: bet on prediction if confidence > threshold
                if max_prob >= 0.50:
                    stake = 10.0  # Flat 10 units
                    flat_total_stake += stake
                    bets_placed += 1

                    if predicted == "HOME" and actual == "H":
                        flat_total_returns += stake * odds["home"]
                    elif predicted == "DRAW" and actual == "D":
                        flat_total_returns += stake * odds["draw"]
                    elif predicted == "AWAY" and actual == "A":
                        flat_total_returns += stake * odds["away"]

                # Kelly betting: bet on value bets only
                if has_value and best_bet:
                    value_bets += 1
                    best_outcome = best_bet.lower()
                    kelly_calc = self.kelly.calculate_kelly(
                        probs[best_outcome], odds[best_outcome]
                    )

                    if kelly_calc["bet_size"] > 0:
                        stake = min(bankroll * kelly_calc["bet_size"], bankroll * 0.05)
                        kelly_total_stake += stake

                        # Did we win?
                        won = (
                            (best_outcome == "home" and actual == "H") or
                            (best_outcome == "draw" and actual == "D") or
                            (best_outcome == "away" and actual == "A")
                        )

                        if won:
                            kelly_total_returns += stake * odds[best_outcome]
                            value_wins += 1
                            bankroll += stake * (odds[best_outcome] - 1)
                        else:
                            bankroll -= stake

                predictions_log.append({
                    "match": f"{row['home_team']} vs {row['away_team']}",
                    "predicted": predicted,
                    "actual": actual_label,
                    "correct": is_correct,
                    "confidence": max_prob,
                    "odds": odds,
                    "has_value": has_value,
                    "value_bet": best_bet,
                })

            except Exception as e:
                log.debug(f"Prediction failed: {e}")
                continue

        if total == 0:
            return {"error": "No successful predictions"}

        # Calculate ROI
        flat_roi = ((flat_total_returns - flat_total_stake) / flat_total_stake * 100) if flat_total_stake > 0 else 0
        kelly_roi = ((kelly_total_returns - kelly_total_stake) / kelly_total_stake * 100) if kelly_total_stake > 0 else 0

        return {
            "season": test_season,
            "total_matches": total,
            "correct": correct,
            "accuracy": correct / total,
            "flat_betting": {
                "bets": bets_placed,
                "stake": flat_total_stake,
                "returns": flat_total_returns,
                "profit": flat_total_returns - flat_total_stake,
                "roi": flat_roi,
            },
            "value_betting": {
                "value_bets": value_bets,
                "value_wins": value_wins,
                "win_rate": value_wins / value_bets if value_bets > 0 else 0,
                "stake": kelly_total_stake,
                "returns": kelly_total_returns,
                "profit": kelly_total_returns - kelly_total_stake,
                "roi": kelly_roi,
                "final_bankroll": bankroll,
                "bankroll_growth": (bankroll - initial_bankroll) / initial_bankroll * 100,
            },
            "predictions": predictions_log,
        }

    def run_backtest(self, test_seasons: List[str] = None) -> Dict:
        """Run backtest across multiple seasons."""
        if test_seasons is None:
            all_seasons = sorted(self.df["season"].unique())
            test_seasons = all_seasons[-3:]

        log.info(f"Running backtest on: {test_seasons}")

        results = {
            "timestamp": datetime.now().isoformat(),
            "strategy": self.strategy,
            "kelly_fraction": self.kelly_fraction,
            "test_seasons": test_seasons,
            "season_results": [],
        }

        # Aggregate tracking
        total_correct = 0
        total_matches = 0
        total_flat_stake = 0
        total_flat_returns = 0
        total_value_bets = 0
        total_value_wins = 0
        total_kelly_stake = 0
        total_kelly_returns = 0

        for season in test_seasons:
            season_result = self.backtest_season(season)

            if "error" not in season_result:
                results["season_results"].append(season_result)
                total_correct += season_result["correct"]
                total_matches += season_result["total_matches"]
                total_flat_stake += season_result["flat_betting"]["stake"]
                total_flat_returns += season_result["flat_betting"]["returns"]
                total_value_bets += season_result["value_betting"]["value_bets"]
                total_value_wins += season_result["value_betting"]["value_wins"]
                total_kelly_stake += season_result["value_betting"]["stake"]
                total_kelly_returns += season_result["value_betting"]["returns"]

                log.info(f"  {season}: {season_result['accuracy']:.1%} "
                        f"(Flat ROI: {season_result['flat_betting']['roi']:+.1f}%, "
                        f"Kelly ROI: {season_result['value_betting']['roi']:+.1f}%)")

        # Overall results
        results["overall"] = {
            "total_matches": total_matches,
            "correct": total_correct,
            "accuracy": total_correct / total_matches if total_matches > 0 else 0,
            "flat_betting": {
                "total_stake": total_flat_stake,
                "total_returns": total_flat_returns,
                "profit": total_flat_returns - total_flat_stake,
                "roi": ((total_flat_returns - total_flat_stake) / total_flat_stake * 100) if total_flat_stake > 0 else 0,
            },
            "value_betting": {
                "total_value_bets": total_value_bets,
                "value_wins": total_value_wins,
                "win_rate": total_value_wins / total_value_bets if total_value_bets > 0 else 0,
                "total_stake": total_kelly_stake,
                "total_returns": total_kelly_returns,
                "profit": total_kelly_returns - total_kelly_stake,
                "roi": ((total_kelly_returns - total_kelly_stake) / total_kelly_stake * 100) if total_kelly_stake > 0 else 0,
            },
        }

        return results


def main():
    parser = argparse.ArgumentParser(description="Backtest with real odds")
    parser.add_argument("--seasons", nargs="+", help="Seasons to test")
    parser.add_argument("--strategy", default="default",
                        choices=["default", "volume", "selective", "ml_heavy", "xg_dominant"],
                        help="Betting strategy")
    parser.add_argument("--kelly", type=float, default=0.25, help="Kelly fraction (0.1-0.5)")
    args = parser.parse_args()

    backtester = OddsBacktester(strategy=args.strategy, kelly_fraction=args.kelly)

    if not backtester.load_data():
        return

    if not backtester.initialize_ensemble():
        log.error("Failed to initialize ensemble")
        return

    results = backtester.run_backtest(args.seasons)

    # Print results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS WITH REAL ODDS")
    print("=" * 70)
    print(f"Strategy: {args.strategy}")
    print(f"Kelly Fraction: {args.kelly}")

    print(f"\nTest Seasons: {results['test_seasons']}")

    print("\nSEASON-BY-SEASON RESULTS")
    print("-" * 60)
    for sr in results["season_results"]:
        print(f"\n  {sr['season']}:")
        print(f"    Accuracy: {sr['accuracy']:.1%} ({sr['correct']}/{sr['total_matches']})")
        print(f"    Flat Betting: ROI {sr['flat_betting']['roi']:+.1f}% "
              f"(profit: {sr['flat_betting']['profit']:+.1f})")
        print(f"    Value Betting: ROI {sr['value_betting']['roi']:+.1f}% "
              f"({sr['value_betting']['value_bets']} bets, "
              f"{sr['value_betting']['win_rate']:.1%} win rate)")

    overall = results["overall"]
    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print(f"Total Matches: {overall['total_matches']}")
    print(f"Accuracy: {overall['accuracy']:.1%}")

    print(f"\nFLAT BETTING (10 units on >50% confidence)")
    print(f"  ROI: {overall['flat_betting']['roi']:+.1f}%")
    print(f"  Profit: {overall['flat_betting']['profit']:+.1f} units")

    print(f"\nVALUE BETTING (Kelly on value bets)")
    print(f"  Value Bets: {overall['value_betting']['total_value_bets']}")
    print(f"  Win Rate: {overall['value_betting']['win_rate']:.1%}")
    print(f"  ROI: {overall['value_betting']['roi']:+.1f}%")
    print(f"  Profit: {overall['value_betting']['profit']:+.1f} units")

    # Save results
    results_dir = DATA_DIR / "optimization"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"backtest_odds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Remove predictions for smaller file
    save_results = {k: v for k, v in results.items()}
    for sr in save_results.get("season_results", []):
        sr.pop("predictions", None)

    with open(results_file, "w") as f:
        json.dump(save_results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
