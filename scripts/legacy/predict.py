#!/usr/bin/env python3
"""Unified Prediction Script - Phase 4.4

Single entry point combining all prediction phases:
- Phase 1: Ensemble prediction (factor, xG, ML, player xG, deep learning)
- Phase 2: Calibration (draw detection, home calibration, confidence filtering)
- Phase 3: Value betting (Kelly criterion, edge detection, EV calculation)
- Phase 4: Bankroll management (position sizing, risk controls)

Usage:
    python scripts/predict.py                          # Predict upcoming matches
    python scripts/predict.py --strategy selective     # Use selective strategy
    python scripts/predict.py --bankroll 500           # Set initial bankroll
    python scripts/predict.py --format json            # Output as JSON
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR

# Import ensemble predictor
from scripts.prediction.ensemble_prediction_engine import (
    EnsemblePredictor,
    run_ensemble_predictions,
    load_upcoming_matches,
)

# Import value betting
from features.value_betting import ValueBettingPipeline

# Import bankroll management
from features.bankroll_manager import BankrollManager, BettingTracker

# Import calibration
try:
    from features.prediction_calibration import BETTING_STRATEGIES, list_strategies
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False
    BETTING_STRATEGIES = {}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class UnifiedPredictor:
    """Unified prediction system combining all phases."""

    def __init__(
        self,
        strategy: str = "default",
        bankroll: float = 1000.0,
        kelly_fraction: float = 0.15,
    ):
        self.strategy = strategy
        self.initial_bankroll = bankroll
        self.kelly_fraction = kelly_fraction

        # Initialize components
        self.ensemble = None
        self.value_pipeline = None
        self.bankroll_manager = None
        self.tracker = None

    def initialize(self) -> bool:
        """Initialize all prediction components."""
        log.info("Initializing unified prediction system...")

        # Ensemble predictor
        self.ensemble = EnsemblePredictor(strategy=self.strategy)
        if not self.ensemble.initialize():
            log.error("Failed to initialize ensemble predictor")
            return False

        # Value betting pipeline
        self.value_pipeline = ValueBettingPipeline(
            kelly_fraction=self.kelly_fraction,
            value_mode="standard",
        )

        # Bankroll manager
        self.bankroll_manager = BankrollManager(initial_bankroll=self.initial_bankroll)
        self.bankroll_manager.load_state()

        # Betting tracker
        self.tracker = BettingTracker()

        log.info("Unified prediction system initialized")
        return True

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        odds: Dict[str, float] = None,
    ) -> Dict:
        """Generate full prediction for a single match.

        Args:
            home_team: Home team name
            away_team: Away team name
            odds: Optional odds dict {home, draw, away}

        Returns:
            Complete prediction with betting recommendation
        """
        # Get ensemble prediction
        match = {
            "home_team": home_team,
            "away_team": away_team,
            "match_date": datetime.now().strftime("%Y-%m-%d"),
        }

        ensemble_result = self.ensemble.predict(match, {}, {})
        if ensemble_result is None:
            return {"error": "Prediction failed"}

        # Extract probabilities
        probs = {
            "home": ensemble_result["probabilities"]["home"],
            "draw": ensemble_result["probabilities"]["draw"],
            "away": ensemble_result["probabilities"]["away"],
        }

        # Use default odds if not provided
        if odds is None:
            odds = {"home": 1.80, "draw": 3.40, "away": 4.00}

        # Value betting analysis
        value_analysis = self.value_pipeline.analyze_bet(
            home_team, away_team,
            probs, odds,
            bankroll=self.bankroll_manager.current_bankroll
        )

        # Get recommended bet size
        best_bet = value_analysis["recommendation"]["best_bet"]
        if best_bet:
            best_outcome = best_bet.lower()
            edge = value_analysis["kelly_analysis"][best_outcome]["edge"]
            confidence = "HIGH" if edge > 0.08 else "MEDIUM" if edge > 0.05 else "LOW"

            bet_size = self.bankroll_manager.calculate_bet_size(
                edge=edge,
                odds=odds[best_outcome],
                confidence=confidence
            )
        else:
            bet_size = 0
            edge = 0
            confidence = "NONE"

        # Build result
        result = {
            "match": f"{home_team} vs {away_team}",
            "timestamp": datetime.now().isoformat(),
            "strategy": self.strategy,

            # Prediction
            "prediction": {
                "outcome": ensemble_result["predicted_outcome"],
                "probabilities": probs,
                "confidence": ensemble_result["confidence"],
                "methods_used": ensemble_result["methods_used"],
            },

            # Draw analysis (if available)
            "draw_analysis": ensemble_result.get("draw_analysis"),

            # Value betting
            "value_analysis": {
                "has_value": value_analysis["value_analysis"]["has_value"],
                "best_bet": best_bet,
                "edge": edge,
                "kelly_analysis": value_analysis["kelly_analysis"],
            },

            # Betting recommendation
            "betting_recommendation": {
                "bet": best_bet,
                "stake": bet_size,
                "odds": odds.get(best_bet.lower() if best_bet else "home", 0),
                "confidence": confidence,
                "potential_return": bet_size * odds.get(best_bet.lower() if best_bet else "home", 0) if bet_size > 0 else 0,
            },

            # Bankroll status
            "bankroll_status": {
                "current": self.bankroll_manager.current_bankroll,
                "drawdown": self.bankroll_manager.get_drawdown(),
                "can_bet": self.bankroll_manager.can_bet()[0],
            },
        }

        return result

    def predict_upcoming(self) -> List[Dict]:
        """Predict all upcoming matches."""
        # Load upcoming matches
        matches = load_upcoming_matches()
        if not matches:
            log.warning("No upcoming matches found")
            return []

        # Load FULL odds data with best odds from all bookmakers
        odds_full_file = DATA_DIR / "upcoming" / "odds_full.json"
        odds_data = {}
        best_odds_data = {}

        if odds_full_file.exists():
            with open(odds_full_file) as f:
                full_data = json.load(f)
                matches_data = full_data.get("matches", {})

                for match_key, match_odds in matches_data.items():
                    h2h = match_odds.get("h2h", {})
                    # Use best available odds for maximum value
                    best_odds_data[match_key] = {
                        "home": h2h.get("best_home", h2h.get("home", 0)),
                        "draw": h2h.get("best_draw", h2h.get("draw", 0)),
                        "away": h2h.get("best_away", h2h.get("away", 0)),
                        # Also store average odds for comparison
                        "avg_home": h2h.get("home", 0),
                        "avg_draw": h2h.get("draw", 0),
                        "avg_away": h2h.get("away", 0),
                        "bookmakers_count": h2h.get("bookmakers_count", 0),
                    }
                    # Standard odds format for compatibility
                    odds_data[match_key] = {
                        "home": h2h.get("home", 0),
                        "draw": h2h.get("draw", 0),
                        "away": h2h.get("away", 0),
                    }

                log.info(f"Loaded BEST odds for {len(best_odds_data)} matches from {len(full_data.get('markets', []))} markets")
        else:
            # Fallback to simple odds file
            odds_file = DATA_DIR / "upcoming" / "odds.json"
            if odds_file.exists():
                with open(odds_file) as f:
                    odds_data = json.load(f)
                log.info(f"Loaded average odds for {len(odds_data)} matches")

        predictions = []
        for match in matches:
            home = match["home_team"]
            away = match["away_team"]
            key = f"{home} vs {away}"

            # Get BEST odds for this match (for value calculation)
            odds = best_odds_data.get(key) or odds_data.get(key)

            # Log when using real odds vs defaults
            if odds:
                log.debug(f"Using real odds for {key}: H={odds.get('home', 0):.2f} D={odds.get('draw', 0):.2f} A={odds.get('away', 0):.2f}")
            else:
                log.warning(f"No odds found for {key} - using defaults")

            # Predict
            result = self.predict_match(home, away, odds)
            result["date"] = match.get("date", "TBD")
            result["time"] = match.get("time", "TBD")
            result["venue"] = match.get("venue", f"{home} Stadium")

            # Add odds comparison if best odds available
            if key in best_odds_data and best_odds_data[key].get("bookmakers_count", 0) > 0:
                best = best_odds_data[key]
                result["odds_info"] = {
                    "best_odds": {"home": best["home"], "draw": best["draw"], "away": best["away"]},
                    "avg_odds": {"home": best["avg_home"], "draw": best["avg_draw"], "away": best["avg_away"]},
                    "bookmakers_count": best["bookmakers_count"],
                    "best_vs_avg_edge": {
                        "home": round((best["home"] - best["avg_home"]) / best["avg_home"] * 100, 1) if best["avg_home"] > 0 else 0,
                        "draw": round((best["draw"] - best["avg_draw"]) / best["avg_draw"] * 100, 1) if best["avg_draw"] > 0 else 0,
                        "away": round((best["away"] - best["avg_away"]) / best["avg_away"] * 100, 1) if best["avg_away"] > 0 else 0,
                    }
                }

            predictions.append(result)

        # Sort by value (best bets first)
        predictions.sort(
            key=lambda x: (
                -1 if x.get("value_analysis", {}).get("has_value") else 0,
                -(x.get("value_analysis", {}).get("edge", 0) or 0),
                -(x.get("prediction", {}).get("confidence", 0) or 0),
            )
        )

        return predictions

    def get_bankroll_status(self) -> Dict:
        """Get current bankroll status."""
        return self.bankroll_manager.get_status()

    def get_betting_stats(self, days: int = 30) -> Dict:
        """Get betting statistics."""
        return self.tracker.get_stats(days)


def print_predictions(predictions: List[Dict], verbose: bool = False):
    """Print predictions in a formatted way."""
    print("\n" + "=" * 70)
    print("SERIE A PREDICTIONS - UNIFIED SYSTEM")
    print("=" * 70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Value bets first
    value_bets = [p for p in predictions if p.get("value_analysis", {}).get("has_value")]
    other_bets = [p for p in predictions if not p.get("value_analysis", {}).get("has_value")]

    if value_bets:
        print("\n" + "-" * 70)
        print("VALUE BETS (Recommended)")
        print("-" * 70)

        for pred in value_bets:
            print_single_prediction(pred, verbose)

    if other_bets:
        print("\n" + "-" * 70)
        print("OTHER MATCHES")
        print("-" * 70)

        for pred in other_bets[:5]:  # Show top 5
            print_single_prediction(pred, verbose)


def print_single_prediction(pred: Dict, verbose: bool = False):
    """Print a single prediction."""
    print(f"\n{pred['match']}")
    print(f"  {pred.get('date', 'TBD')} {pred.get('time', '')}")

    prediction = pred.get("prediction", {})
    probs = prediction.get("probabilities", {})

    print(f"\n  Prediction: {prediction.get('outcome')} "
          f"({prediction.get('confidence', 0):.1%} confidence)")
    print(f"  Probabilities: H {probs.get('home', 0):.1%} | "
          f"D {probs.get('draw', 0):.1%} | A {probs.get('away', 0):.1%}")

    # Draw analysis
    draw_analysis = pred.get("draw_analysis")
    if draw_analysis and draw_analysis.get("is_draw_candidate"):
        print(f"  Draw candidate: score={draw_analysis.get('draw_score', 0):.2f}")

    # Value analysis
    value = pred.get("value_analysis", {})
    if value.get("has_value"):
        print(f"\n  VALUE BET: {value.get('best_bet', '').upper()}")
        print(f"  Edge: {value.get('edge', 0):+.1%}")

    # Betting recommendation
    rec = pred.get("betting_recommendation", {})
    if rec.get("stake", 0) > 0:
        print(f"\n  RECOMMENDATION:")
        print(f"    Bet: {rec.get('bet', 'NONE').upper()}")
        print(f"    Stake: {rec.get('stake', 0):.2f} units")
        print(f"    Odds: {rec.get('odds', 0):.2f}")
        print(f"    Potential return: {rec.get('potential_return', 0):.2f}")
    else:
        print(f"\n  RECOMMENDATION: SKIP (no edge or bankroll limit)")

    if verbose:
        print(f"\n  Methods: {', '.join(prediction.get('methods_used', []))}")


def main():
    parser = argparse.ArgumentParser(description="Unified Serie A Prediction System")
    parser.add_argument("--strategy", default="selective",
                        choices=["default", "volume", "selective", "ml_heavy", "xg_dominant"],
                        help="Betting strategy (default: selective)")
    parser.add_argument("--bankroll", type=float, default=1000.0,
                        help="Initial bankroll (default: 1000)")
    parser.add_argument("--kelly", type=float, default=0.15,
                        help="Kelly fraction (default: 0.15)")
    parser.add_argument("--format", choices=["text", "json"], default="text",
                        help="Output format (default: text)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--match", nargs=2, metavar=("HOME", "AWAY"),
                        help="Predict single match (e.g., --match Inter Milan)")
    args = parser.parse_args()

    # Initialize predictor
    predictor = UnifiedPredictor(
        strategy=args.strategy,
        bankroll=args.bankroll,
        kelly_fraction=args.kelly,
    )

    if not predictor.initialize():
        print("Failed to initialize prediction system")
        return

    # Single match or upcoming?
    if args.match:
        home, away = args.match
        result = predictor.predict_match(home, away)
        predictions = [result]
    else:
        predictions = predictor.predict_upcoming()

    # Output
    if args.format == "json":
        print(json.dumps(predictions, indent=2, default=str))
    else:
        print_predictions(predictions, verbose=args.verbose)

        # Show bankroll status
        status = predictor.get_bankroll_status()
        print("\n" + "=" * 70)
        print("BANKROLL STATUS")
        print("=" * 70)
        print(f"  Current: {status['current_bankroll']:.2f} units")
        print(f"  Drawdown: {status['drawdown']:.1%}")
        print(f"  Can bet: {'Yes' if status['can_bet'] else 'No'}")
        print(f"  Total bets: {status['total_bets']}")
        print(f"  Win rate: {status['win_rate']:.1%}")
        print(f"  ROI: {status['roi']:+.1%}")

    # Save predictions
    output_path = DATA_DIR / "upcoming" / "predictions_unified.json"
    with open(output_path, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "strategy": args.strategy,
            "predictions": predictions,
            "bankroll_status": predictor.get_bankroll_status(),
        }, f, indent=2, default=str)

    print(f"\nPredictions saved to: {output_path}")


if __name__ == "__main__":
    main()
