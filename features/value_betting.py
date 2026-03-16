#!/usr/bin/env python3
"""Value Betting Module - Phase 3

Implements:
- Kelly Criterion for optimal bet sizing
- Value betting detection (model vs market)
- Expected Value (EV) calculation
- Edge calculation and tracking

Key concepts:
- Edge = Model probability - Implied probability
- Value bet = Edge > threshold (typically 5%)
- Kelly fraction = edge / (odds - 1)
- EV = (probability * profit) - ((1-probability) * stake)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# KELLY CRITERION (Phase 3.1)
# =============================================================================

class KellyCriterion:
    """Optimal bet sizing using Kelly Criterion.

    Full Kelly is aggressive and can lead to large swings.
    Fractional Kelly (0.25-0.5) is more conservative.

    Formula: f* = (bp - q) / b
    where:
        b = decimal odds - 1 (net profit per unit)
        p = probability of winning
        q = probability of losing (1 - p)
        f* = fraction of bankroll to bet
    """

    def __init__(self, fraction: float = 0.25):
        """
        Args:
            fraction: Kelly fraction (0.25 = quarter Kelly, 0.5 = half Kelly)
        """
        self.fraction = fraction
        self.min_bet = 0.01  # Minimum 1% of bankroll
        self.max_bet = 0.10  # Maximum 10% of bankroll

    def calculate_kelly(
        self,
        probability: float,
        decimal_odds: float,
    ) -> Dict:
        """Calculate Kelly bet size.

        Args:
            probability: Model's estimated probability of winning
            decimal_odds: Decimal odds offered (e.g., 2.0 for even money)

        Returns:
            Dict with kelly_fraction, bet_size, edge, and recommendation
        """
        # Validate inputs
        if probability <= 0 or probability >= 1:
            return {"kelly_fraction": 0, "bet_size": 0, "edge": 0, "recommendation": "INVALID"}

        if decimal_odds <= 1:
            return {"kelly_fraction": 0, "bet_size": 0, "edge": 0, "recommendation": "INVALID"}

        # Calculate implied probability from odds
        implied_prob = 1 / decimal_odds

        # Calculate edge
        edge = probability - implied_prob

        # Calculate Kelly fraction
        # f* = (bp - q) / b where b = odds - 1
        b = decimal_odds - 1  # Net profit per unit
        p = probability
        q = 1 - probability

        full_kelly = (b * p - q) / b if b > 0 else 0

        # Apply fractional Kelly
        kelly_fraction = full_kelly * self.fraction

        # Apply limits
        bet_size = max(0, min(self.max_bet, kelly_fraction))

        # Generate recommendation
        if edge < 0:
            recommendation = "NO_BET"
            bet_size = 0
        elif edge < 0.03:
            recommendation = "SKIP"
            bet_size = 0
        elif edge < 0.05:
            recommendation = "SMALL"
            bet_size = min(bet_size, 0.02)
        elif edge < 0.10:
            recommendation = "MEDIUM"
            bet_size = min(bet_size, 0.05)
        else:
            recommendation = "LARGE"
            bet_size = min(bet_size, self.max_bet)

        return {
            "kelly_fraction": round(full_kelly, 4),
            "fractional_kelly": round(kelly_fraction, 4),
            "bet_size": round(bet_size, 4),
            "edge": round(edge, 4),
            "implied_prob": round(implied_prob, 4),
            "recommendation": recommendation,
        }

    def calculate_for_all_outcomes(
        self,
        probs: Dict[str, float],
        odds: Dict[str, float],
    ) -> Dict:
        """Calculate Kelly for all three outcomes.

        Args:
            probs: Model probabilities {home, draw, away}
            odds: Decimal odds {home, draw, away}

        Returns:
            Dict with Kelly calculations for each outcome and best bet
        """
        results = {}
        best_bet = None
        best_edge = -1

        for outcome in ["home", "draw", "away"]:
            prob = probs.get(outcome, 0.33)
            odd = odds.get(outcome, 2.0)

            calc = self.calculate_kelly(prob, odd)
            results[outcome] = calc

            if calc["edge"] > best_edge and calc["bet_size"] > 0:
                best_edge = calc["edge"]
                best_bet = outcome

        results["best_bet"] = best_bet
        results["best_edge"] = round(best_edge, 4) if best_edge > 0 else 0

        return results


# =============================================================================
# VALUE BETTING DETECTION (Phase 3.2)
# =============================================================================

class ValueBettingDetector:
    """Detect value bets where model edge exceeds threshold."""

    # Minimum edge thresholds by confidence level
    EDGE_THRESHOLDS = {
        "aggressive": 0.03,  # 3% edge
        "standard": 0.05,    # 5% edge
        "conservative": 0.08, # 8% edge
    }

    def __init__(self, mode: str = "standard"):
        self.mode = mode
        self.min_edge = self.EDGE_THRESHOLDS.get(mode, 0.05)
        self.odds_cache = {}
        self._load_odds_data()

    def _load_odds_data(self):
        """Load historical odds data."""
        try:
            odds_dir = DATA_DIR / "odds"
            if odds_dir.exists():
                # Load opening odds
                opening_path = odds_dir / "opening_odds.csv"
                if opening_path.exists():
                    self.opening_odds = pd.read_csv(opening_path)
                    log.info(f"Loaded {len(self.opening_odds)} historical odds records")
                else:
                    self.opening_odds = pd.DataFrame()
            else:
                self.opening_odds = pd.DataFrame()
        except Exception as e:
            log.warning(f"Failed to load odds data: {e}")
            self.opening_odds = pd.DataFrame()

    def get_odds_for_match(
        self,
        home_team: str,
        away_team: str,
        match_date: str = None,
    ) -> Optional[Dict[str, float]]:
        """Get odds for a specific match."""
        if self.opening_odds.empty:
            return None

        # Try to find match in odds data
        mask = (
            (self.opening_odds["home_team"] == home_team) &
            (self.opening_odds["away_team"] == away_team)
        )

        if match_date:
            mask &= (self.opening_odds["match_date"] == match_date)

        matches = self.opening_odds[mask]

        if len(matches) == 0:
            return None

        row = matches.iloc[0]
        return {
            "home": float(row.get("home_odds", 2.0)),
            "draw": float(row.get("draw_odds", 3.4)),
            "away": float(row.get("away_odds", 3.6)),
        }

    def detect_value(
        self,
        probs: Dict[str, float],
        odds: Dict[str, float],
    ) -> Dict:
        """Detect value betting opportunities.

        Args:
            probs: Model probabilities {home, draw, away}
            odds: Decimal odds {home, draw, away}

        Returns:
            Dict with value analysis for each outcome
        """
        results = {
            "has_value": False,
            "value_bets": [],
            "outcomes": {},
        }

        for outcome in ["home", "draw", "away"]:
            prob = probs.get(outcome, 0.33)
            odd = odds.get(outcome, 2.0)

            # Calculate implied probability
            implied_prob = 1 / odd if odd > 0 else 0

            # Calculate edge
            edge = prob - implied_prob

            # Calculate expected value per unit
            # EV = (prob * (odds - 1)) - ((1 - prob) * 1)
            ev_per_unit = (prob * (odd - 1)) - (1 - prob)

            # Is this a value bet?
            is_value = edge >= self.min_edge

            outcome_result = {
                "probability": round(prob, 4),
                "implied_prob": round(implied_prob, 4),
                "odds": odd,
                "edge": round(edge, 4),
                "ev_per_unit": round(ev_per_unit, 4),
                "is_value": is_value,
            }

            results["outcomes"][outcome] = outcome_result

            if is_value:
                results["has_value"] = True
                results["value_bets"].append({
                    "outcome": outcome,
                    "edge": round(edge, 4),
                    "ev_per_unit": round(ev_per_unit, 4),
                    "odds": odd,
                })

        # Sort value bets by edge
        results["value_bets"].sort(key=lambda x: x["edge"], reverse=True)

        return results


# =============================================================================
# EXPECTED VALUE CALCULATOR (Phase 3.3)
# =============================================================================

class EVCalculator:
    """Calculate Expected Value for betting decisions."""

    def __init__(self, min_ev: float = 0.02):
        """
        Args:
            min_ev: Minimum EV per unit to recommend bet (default 2%)
        """
        self.min_ev = min_ev

    def calculate_ev(
        self,
        probability: float,
        decimal_odds: float,
        stake: float = 1.0,
    ) -> Dict:
        """Calculate Expected Value for a bet.

        Formula: EV = (probability * profit) - ((1 - probability) * stake)

        Args:
            probability: Model's estimated win probability
            decimal_odds: Decimal odds offered
            stake: Bet stake (default 1 unit)

        Returns:
            Dict with EV metrics
        """
        if decimal_odds <= 1 or probability <= 0 or probability >= 1:
            return {
                "ev": 0,
                "ev_percent": 0,
                "profit_if_win": 0,
                "loss_if_lose": stake,
                "recommendation": "INVALID",
            }

        # Calculate potential profit
        profit_if_win = stake * (decimal_odds - 1)
        loss_if_lose = stake

        # Calculate EV
        ev = (probability * profit_if_win) - ((1 - probability) * loss_if_lose)
        ev_percent = ev / stake

        # Generate recommendation
        if ev_percent < 0:
            recommendation = "NEGATIVE_EV"
        elif ev_percent < self.min_ev:
            recommendation = "LOW_EV"
        elif ev_percent < 0.05:
            recommendation = "SMALL_BET"
        elif ev_percent < 0.10:
            recommendation = "MEDIUM_BET"
        else:
            recommendation = "STRONG_BET"

        return {
            "ev": round(ev, 4),
            "ev_percent": round(ev_percent, 4),
            "profit_if_win": round(profit_if_win, 4),
            "loss_if_lose": round(loss_if_lose, 4),
            "probability": round(probability, 4),
            "odds": decimal_odds,
            "recommendation": recommendation,
        }

    def calculate_for_all_outcomes(
        self,
        probs: Dict[str, float],
        odds: Dict[str, float],
        stake: float = 1.0,
    ) -> Dict:
        """Calculate EV for all outcomes.

        Args:
            probs: Model probabilities {home, draw, away}
            odds: Decimal odds {home, draw, away}
            stake: Bet stake

        Returns:
            Dict with EV for each outcome and best opportunity
        """
        results = {}
        best_outcome = None
        best_ev = -float("inf")

        for outcome in ["home", "draw", "away"]:
            prob = probs.get(outcome, 0.33)
            odd = odds.get(outcome, 2.0)

            calc = self.calculate_ev(prob, odd, stake)
            results[outcome] = calc

            if calc["ev"] > best_ev and calc["ev"] > 0:
                best_ev = calc["ev"]
                best_outcome = outcome

        results["best_outcome"] = best_outcome
        results["best_ev"] = round(best_ev, 4) if best_ev > 0 else 0
        results["has_positive_ev"] = best_ev > 0

        return results


# =============================================================================
# UNIFIED VALUE BETTING PIPELINE
# =============================================================================

class ValueBettingPipeline:
    """Unified pipeline for value betting analysis."""

    def __init__(
        self,
        kelly_fraction: float = 0.10,  # Synced with production (betting_unified.py)
        value_mode: str = "standard",
        min_ev: float = 0.02,
    ):
        self.kelly = KellyCriterion(fraction=kelly_fraction)
        self.value_detector = ValueBettingDetector(mode=value_mode)
        self.ev_calculator = EVCalculator(min_ev=min_ev)

    def analyze_bet(
        self,
        home_team: str,
        away_team: str,
        probs: Dict[str, float],
        odds: Dict[str, float] = None,
        match_date: str = None,
        bankroll: float = 100.0,
    ) -> Dict:
        """Complete value betting analysis for a match.

        Args:
            home_team: Home team name
            away_team: Away team name
            probs: Model probabilities {home, draw, away}
            odds: Decimal odds (if None, tries to fetch from data)
            match_date: Match date for odds lookup
            bankroll: Current bankroll for Kelly sizing

        Returns:
            Complete betting analysis
        """
        # Get odds if not provided
        if odds is None:
            odds = self.value_detector.get_odds_for_match(home_team, away_team, match_date)
            if odds is None:
                # Use default odds
                odds = {"home": 1.80, "draw": 3.40, "away": 4.00}

        # Run all analyses
        kelly_results = self.kelly.calculate_for_all_outcomes(probs, odds)
        value_results = self.value_detector.detect_value(probs, odds)
        ev_results = self.ev_calculator.calculate_for_all_outcomes(probs, odds)

        # Determine best bet
        best_bet = None
        bet_amount = 0
        confidence = "NONE"

        if value_results["has_value"] and ev_results["has_positive_ev"]:
            # Find the outcome that's best across all metrics
            best_outcome = kelly_results["best_bet"]

            if best_outcome:
                best_bet = best_outcome
                bet_amount = kelly_results[best_outcome]["bet_size"] * bankroll
                edge = kelly_results[best_outcome]["edge"]

                if edge > 0.10:
                    confidence = "HIGH"
                elif edge > 0.05:
                    confidence = "MEDIUM"
                else:
                    confidence = "LOW"

        return {
            "match": f"{home_team} vs {away_team}",
            "odds_used": odds,
            "kelly_analysis": kelly_results,
            "value_analysis": value_results,
            "ev_analysis": ev_results,
            "recommendation": {
                "best_bet": best_bet,
                "bet_amount": round(bet_amount, 2),
                "confidence": confidence,
                "has_value": value_results["has_value"],
            },
        }


def get_value_pipeline(
    kelly_fraction: float = 0.10,  # Synced with production (betting_unified.py)
    value_mode: str = "standard",
) -> ValueBettingPipeline:
    """Factory function for value betting pipeline."""
    return ValueBettingPipeline(
        kelly_fraction=kelly_fraction,
        value_mode=value_mode,
    )


if __name__ == "__main__":
    # Test value betting pipeline
    print("Testing Value Betting Pipeline")
    print("=" * 60)

    pipeline = get_value_pipeline(kelly_fraction=0.10, value_mode="standard")

    # Test case: Model predicts home win with edge
    probs = {"home": 0.55, "draw": 0.25, "away": 0.20}
    odds = {"home": 2.00, "draw": 3.40, "away": 4.00}

    result = pipeline.analyze_bet(
        "Inter", "Milan",
        probs, odds,
        bankroll=1000.0
    )

    print(f"\nMatch: {result['match']}")
    print(f"Odds: H={odds['home']:.2f} D={odds['draw']:.2f} A={odds['away']:.2f}")
    print(f"Probs: H={probs['home']:.1%} D={probs['draw']:.1%} A={probs['away']:.1%}")

    print(f"\nKelly Analysis:")
    for outcome in ["home", "draw", "away"]:
        k = result["kelly_analysis"][outcome]
        print(f"  {outcome.upper()}: edge={k['edge']:+.1%}, kelly={k['fractional_kelly']:.1%}, bet={k['bet_size']:.1%}")

    print(f"\nValue Detection:")
    print(f"  Has value: {result['value_analysis']['has_value']}")
    for vb in result["value_analysis"]["value_bets"]:
        print(f"  {vb['outcome'].upper()}: edge={vb['edge']:+.1%}, EV={vb['ev_per_unit']:+.1%}/unit")

    print(f"\nRecommendation:")
    rec = result["recommendation"]
    print(f"  Best bet: {rec['best_bet']}")
    print(f"  Amount: {rec['bet_amount']:.2f} units")
    print(f"  Confidence: {rec['confidence']}")
