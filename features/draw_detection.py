#!/usr/bin/env python3
"""DRAW Detection Module - Phase 2.1

Specialized module for detecting likely DRAW outcomes.
Draws are underrepresented in standard ensemble predictions because
they rarely exceed home/away probabilities individually.

Key insight: Draws are more likely when:
1. Teams are evenly matched (small elo_diff)
2. Both teams have strong defenses
3. Low xG predictions for both sides
4. Head-to-head history shows high draw rate
5. Derby matches
6. Relegation six-pointers
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# Draw indicator thresholds calibrated from Serie A historical data
DRAW_INDICATORS = {
    "elo_diff_tight": 35,        # Elo difference threshold for "evenly matched"
    "elo_diff_very_tight": 15,   # Very evenly matched
    "xg_low_total": 2.0,         # Low combined xG indicates defensive game
    "xg_balanced": 0.25,         # Max xG difference for balanced attack
    "h2h_draw_rate_high": 0.40,  # H2H draw rate threshold (raised from 0.35)
    "clean_sheet_both": 0.30,    # Both teams clean sheet rate threshold
    "goal_conceded_low": 0.9,    # Low goals conceded per game
    "defense_strength_high": 1.15, # Strong defense threshold
}


class DrawDetector:
    """Specialized detector for identifying likely DRAW outcomes."""

    def __init__(self):
        self.df = None
        self.derby_pairs = self._get_serie_a_derbies()
        self.draw_rate_cache = {}

    def _get_serie_a_derbies(self) -> set:
        """Get known Serie A derby matchups."""
        return {
            frozenset({"Inter", "Milan"}),          # Derby della Madonnina
            frozenset({"Roma", "Lazio"}),           # Derby della Capitale
            frozenset({"Juventus", "Torino"}),      # Derby della Mole
            frozenset({"Napoli", "Roma"}),          # Derby del Sole
            frozenset({"Genoa", "Sampdoria"}),      # Derby della Lanterna
            frozenset({"Fiorentina", "Bologna"}),   # Emilia-Romagna derby
            frozenset({"Verona", "Venezia"}),       # Veneto derby
            frozenset({"Inter", "Juventus"}),       # Derby d'Italia
            frozenset({"Milan", "Juventus"}),       # Big rivalry
            frozenset({"Roma", "Napoli"}),          # Southern rivalry
        }

    def load_historical(self) -> bool:
        """Load historical data for draw analysis."""
        try:
            features_path = DATA_DIR / "features" / "features.parquet"
            if not features_path.exists():
                return False

            self.df = pd.read_parquet(features_path)
            self.df = self.df.sort_values("match_date", ascending=False)
            self._build_draw_rate_cache()
            return True
        except Exception as e:
            log.error(f"Failed to load historical data: {e}")
            return False

    def _build_draw_rate_cache(self):
        """Build cache of team draw rates."""
        if self.df is None:
            return

        # Get recent seasons only
        recent = self.df[self.df["season"] >= "2020-2021"]

        for team in recent["home_team"].unique():
            home_matches = recent[recent["home_team"] == team]
            away_matches = recent[recent["away_team"] == team]

            home_draws = (home_matches["result"] == "D").sum()
            away_draws = (away_matches["result"] == "D").sum()
            total_matches = len(home_matches) + len(away_matches)

            if total_matches > 0:
                self.draw_rate_cache[team] = (home_draws + away_draws) / total_matches

    def is_derby(self, home_team: str, away_team: str) -> bool:
        """Check if match is a derby."""
        return frozenset({home_team, away_team}) in self.derby_pairs

    def get_h2h_draw_rate(self, home_team: str, away_team: str, n_matches: int = 10) -> float:
        """Get head-to-head draw rate."""
        if self.df is None:
            return 0.27  # Serie A average

        h2h = self.df[
            ((self.df["home_team"] == home_team) & (self.df["away_team"] == away_team)) |
            ((self.df["home_team"] == away_team) & (self.df["away_team"] == home_team))
        ].head(n_matches)

        if len(h2h) < 3:
            return 0.27  # Not enough history

        return (h2h["result"] == "D").mean()

    def analyze_draw_likelihood(
        self,
        home_team: str,
        away_team: str,
        ensemble_probs: Dict[str, float],
        features: Dict = None,
    ) -> Dict:
        """Analyze likelihood of a draw outcome.

        Returns draw score (0-1) and analysis details.
        """
        indicators = []
        draw_score = 0.0
        weights_sum = 0.0

        # 1. Check if ensemble draw probability is already significant
        base_draw_prob = ensemble_probs.get("prob_D", 0.27)
        if base_draw_prob > 0.30:
            indicators.append(("high_base_draw_prob", 0.15))
            draw_score += 0.15 * 0.8
            weights_sum += 0.15

        # 2. Check if evenly matched (home and away probs similar)
        home_prob = ensemble_probs.get("prob_H", 0.45)
        away_prob = ensemble_probs.get("prob_A", 0.28)
        prob_diff = abs(home_prob - away_prob)

        if prob_diff < 0.15:  # Very close probabilities
            indicators.append(("evenly_matched_probs", 0.20))
            match_factor = 1.0 - (prob_diff / 0.15)  # Closer = higher
            draw_score += 0.20 * match_factor
            weights_sum += 0.20

        # 3. Derby boost
        if self.is_derby(home_team, away_team):
            indicators.append(("derby_match", 0.15))
            draw_score += 0.15 * 1.0
            weights_sum += 0.15

        # 4. H2H history
        h2h_draw_rate = self.get_h2h_draw_rate(home_team, away_team)
        if h2h_draw_rate > DRAW_INDICATORS["h2h_draw_rate_high"]:
            indicators.append(("high_h2h_draw_rate", 0.20))
            h2h_factor = min(1.0, h2h_draw_rate / 0.50)
            draw_score += 0.20 * h2h_factor
            weights_sum += 0.20

        # 5. Team draw rates
        home_draw_rate = self.draw_rate_cache.get(home_team, 0.27)
        away_draw_rate = self.draw_rate_cache.get(away_team, 0.27)
        avg_draw_rate = (home_draw_rate + away_draw_rate) / 2

        if avg_draw_rate > 0.32:
            indicators.append(("high_team_draw_rates", 0.10))
            draw_score += 0.10 * min(1.0, avg_draw_rate / 0.35)
            weights_sum += 0.10

        # 6. Feature-based indicators (if available)
        if features:
            # Elo difference
            elo_diff = abs(features.get("elo_diff", 100))
            if elo_diff < DRAW_INDICATORS["elo_diff_tight"]:
                indicators.append(("tight_elo_diff", 0.15))
                elo_factor = 1.0 - (elo_diff / DRAW_INDICATORS["elo_diff_tight"])
                draw_score += 0.15 * elo_factor
                weights_sum += 0.15

            # Low xG total (defensive game)
            home_xg = features.get("home_xg", 1.4)
            away_xg = features.get("away_xg", 1.1)
            total_xg = home_xg + away_xg

            if total_xg < DRAW_INDICATORS["xg_low_total"]:
                indicators.append(("low_total_xg", 0.10))
                xg_factor = 1.0 - (total_xg / DRAW_INDICATORS["xg_low_total"])
                draw_score += 0.10 * max(0, xg_factor)
                weights_sum += 0.10

            # Balanced xG (neither team dominant)
            xg_diff = abs(home_xg - away_xg)
            if xg_diff < DRAW_INDICATORS["xg_balanced"]:
                indicators.append(("balanced_xg", 0.10))
                balance_factor = 1.0 - (xg_diff / DRAW_INDICATORS["xg_balanced"])
                draw_score += 0.10 * balance_factor
                weights_sum += 0.10

            # Both teams have strong defense
            home_def = features.get("home_defense_strength", 1.0)
            away_def = features.get("away_defense_strength", 1.0)

            if home_def > DRAW_INDICATORS["defense_strength_high"] and \
               away_def > DRAW_INDICATORS["defense_strength_high"]:
                indicators.append(("both_strong_defense", 0.10))
                draw_score += 0.10 * 0.9
                weights_sum += 0.10

            # Clean sheet rates
            home_cs = features.get("home_roll_5_clean_sheet", 0.2)
            away_cs = features.get("away_roll_5_clean_sheet", 0.2)

            if home_cs > DRAW_INDICATORS["clean_sheet_both"] and \
               away_cs > DRAW_INDICATORS["clean_sheet_both"]:
                indicators.append(("both_high_clean_sheet", 0.10))
                draw_score += 0.10 * 0.8
                weights_sum += 0.10

        # Normalize draw score against TOTAL possible weight (not just triggered).
        # All indicator weights sum to ~1.25.  Using the full denominator prevents
        # a single indicator (e.g. derby=0.15) from producing score=1.0.
        TOTAL_POSSIBLE_WEIGHT = 1.25
        draw_score = draw_score / TOTAL_POSSIBLE_WEIGHT

        # Calculate adjusted draw probability
        # Use draw score to boost the base draw probability
        draw_boost = draw_score * 0.15  # Max ~15% boost when ALL indicators fire
        adjusted_draw_prob = min(0.45, base_draw_prob + draw_boost)

        # Require breadth of evidence: at least 3 indicators must fire
        n_indicators = len(indicators)

        return {
            "draw_score": round(draw_score, 3),
            "base_draw_prob": round(base_draw_prob, 3),
            "adjusted_draw_prob": round(adjusted_draw_prob, 3),
            "draw_boost": round(draw_boost, 3),
            "indicators": indicators,
            # Tightened thresholds: old (0.30, >=3) over-triggered on 2024-25
            # (52 matches, 10 wrong flips vs 8 correct). Cross-validated on
            # 2023-24 + 2024-25: (0.35, >=4) → avg LL 0.9471→0.9452 (-0.0018)
            "is_draw_candidate": draw_score > 0.35 and n_indicators >= 4,
            "strong_draw_signal": draw_score > 0.50 and n_indicators >= 5,
        }

    def adjust_ensemble_probs(
        self,
        home_team: str,
        away_team: str,
        ensemble_probs: Dict[str, float],
        features: Dict = None,
    ) -> Tuple[Dict[str, float], Dict]:
        """Adjust ensemble probabilities based on draw analysis.

        Returns adjusted probabilities and draw analysis.
        """
        analysis = self.analyze_draw_likelihood(
            home_team, away_team, ensemble_probs, features
        )

        if not analysis["is_draw_candidate"]:
            # No adjustment needed
            return ensemble_probs, analysis

        # Get adjustment amount
        draw_boost = analysis["draw_boost"]

        # Redistribute probability
        prob_H = ensemble_probs["prob_H"]
        prob_D = ensemble_probs["prob_D"]
        prob_A = ensemble_probs["prob_A"]

        # Take proportionally from home and away
        home_share = prob_H / (prob_H + prob_A)
        away_share = prob_A / (prob_H + prob_A)

        new_prob_H = prob_H - draw_boost * home_share
        new_prob_A = prob_A - draw_boost * away_share
        new_prob_D = prob_D + draw_boost

        # Ensure valid probabilities
        new_prob_H = max(0.15, new_prob_H)
        new_prob_A = max(0.10, new_prob_A)

        # Normalize
        total = new_prob_H + new_prob_D + new_prob_A
        adjusted = {
            "prob_H": new_prob_H / total,
            "prob_D": new_prob_D / total,
            "prob_A": new_prob_A / total,
        }

        return adjusted, analysis


def get_draw_detector() -> DrawDetector:
    """Factory function for draw detector."""
    detector = DrawDetector()
    detector.load_historical()
    return detector


if __name__ == "__main__":
    # Test draw detection
    detector = get_draw_detector()

    # Test cases
    test_cases = [
        ("Inter", "Milan", {"prob_H": 0.45, "prob_D": 0.28, "prob_A": 0.27}),  # Derby
        ("Juventus", "Napoli", {"prob_H": 0.42, "prob_D": 0.29, "prob_A": 0.29}),  # Close match
        ("Inter", "Lecce", {"prob_H": 0.65, "prob_D": 0.20, "prob_A": 0.15}),  # One-sided
    ]

    for home, away, probs in test_cases:
        analysis = detector.analyze_draw_likelihood(home, away, probs)
        print(f"\n{home} vs {away}")
        print(f"  Draw score: {analysis['draw_score']:.2f}")
        print(f"  Base draw prob: {analysis['base_draw_prob']:.1%}")
        print(f"  Adjusted draw prob: {analysis['adjusted_draw_prob']:.1%}")
        print(f"  Is draw candidate: {analysis['is_draw_candidate']}")
        print(f"  Indicators: {[i[0] for i in analysis['indicators']]}")
