#!/usr/bin/env python3
"""POINT SPREAD/HANDICAP PREDICTION MODEL - Phase 3 Implementation

Predicts match margins and calculates handicap/spread probabilities.

Features:
- Expected margin calculation (Home - Away)
- Asian handicap probabilities
- European handicap (3-way) calculations
- Form and strength-based adjustments
- Value betting against bookmaker lines

Based on margin analysis from 7,829 historical matches.
"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR
from scripts.betting.italian_market_standards import normalize_line_to_italian

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Odds sanity — reject live/in-play odds before generating recommendations
ODDS_SANITY_SPREADS = {"min": 1.05, "max": 6.0}

# =============================================================================
# CONSTANTS & HISTORICAL DATA
# =============================================================================

# Historical margin statistics from Serie A
HISTORICAL_MARGINS = {
    "mean": 0.35,      # Average margin is +0.35 for home team
    "std_dev": 1.75,   # Standard deviation of margins
    "home_win_margin_avg": 1.65,  # When home wins, avg margin
    "away_win_margin_avg": -1.45,  # When away wins, avg margin
}

# Home advantage in goals
HOME_ADVANTAGE = 0.35

# Team strength ratings (from Elo/historical performance)
TEAM_RATINGS = {
    "Inter": 1.40,
    "Napoli": 1.35,
    "Juventus": 1.25,
    "Atalanta": 1.30,
    "Milan": 1.20,
    "Roma": 1.10,
    "Lazio": 1.15,
    "Fiorentina": 1.05,
    "Bologna": 1.00,
    "Torino": 0.95,
    "Genoa": 0.90,
    "Udinese": 0.90,
    "Sassuolo": 0.85,     # Promoted 2025-26
    "Como": 0.85,
    "Parma": 0.82,
    "Verona": 0.80,
    "Cagliari": 0.78,
    "Lecce": 0.75,
    "Pisa": 0.72,          # Promoted 2025-26
    "Cremonese": 0.68,     # Promoted 2025-26
}

# Margin factors
MARGIN_FACTORS = {
    "big_home_favorite": {"margin_mod": 0.60},
    "big_away_favorite": {"margin_mod": -0.70},
    "home_favorite": {"margin_mod": 0.30},
    "away_favorite": {"margin_mod": -0.35},
    "hot_home": {"margin_mod": 0.25},
    "hot_away": {"margin_mod": -0.20},
    "cold_home": {"margin_mod": -0.20},
    "cold_away": {"margin_mod": 0.15},
    "big_stadium": {"margin_mod": 0.10},
    "derby": {"margin_mod": -0.15},  # Derbies tend to be closer
    "home_fav_ref": {"margin_mod": 0.10},
    "away_fav_ref": {"margin_mod": -0.10},
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MarginPrediction:
    """Prediction for match margin."""
    match: str
    home_team: str
    away_team: str
    date: str

    # Expected margin (positive = home favored)
    expected_margin: float
    margin_std_dev: float

    # Team ratings
    home_rating: float
    away_rating: float
    rating_diff: float

    # Handicap probabilities
    handicap_probs: Dict[float, Dict[str, float]]  # {line: {"home": p, "away": p}}

    # Factors used
    factors: List[str]

    # Confidence
    confidence: str
    confidence_rank: int


@dataclass
class HandicapBet:
    """Betting recommendation for handicap market."""
    match: str
    date: str
    handicap_line: float
    bet_side: str  # "home" or "away"
    our_probability: float
    odds: float
    implied_probability: float
    value: float
    value_pct: float
    expected_margin: float
    recommendation: str
    stake_pct: float
    factors: List[str]


# =============================================================================
# PROBABILITY CALCULATIONS
# =============================================================================

def normal_cdf(x: float, mean: float, std_dev: float) -> float:
    """Calculate cumulative distribution function for normal distribution."""
    return stats.norm.cdf(x, loc=mean, scale=std_dev)


def calculate_handicap_probability(
    handicap: float,
    expected_margin: float,
    std_dev: float
) -> Dict[str, float]:
    """Calculate probabilities for Asian handicap.

    For handicap H on home team:
    - Home covers if (home_goals - away_goals) > H
    - Away covers if (home_goals - away_goals) < H

    Args:
        handicap: The handicap line (negative = home gives, positive = home receives)
        expected_margin: Expected (home - away) goals
        std_dev: Standard deviation of margin

    Returns:
        Dict with 'home' and 'away' probabilities
    """
    # P(margin > handicap) = P(home covers)
    # For home -1.5: P(margin > 1.5)
    # For home +0.5: P(margin > -0.5)

    # Adjust handicap for home perspective
    adjusted_line = -handicap  # If home gets +0.5, they need margin > -0.5

    p_home_covers = 1 - normal_cdf(adjusted_line, expected_margin, std_dev)
    p_away_covers = normal_cdf(adjusted_line, expected_margin, std_dev)

    return {
        "home": round(p_home_covers, 4),
        "away": round(p_away_covers, 4)
    }


def calculate_all_handicap_probabilities(
    expected_margin: float,
    std_dev: float,
    lines: List[float] = None
) -> Dict[float, Dict[str, float]]:
    """Calculate probabilities for multiple handicap lines."""

    if lines is None:
        lines = [-2.5, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5]

    probs = {}
    for line in lines:
        probs[line] = calculate_handicap_probability(line, expected_margin, std_dev)

    return probs


# =============================================================================
# MARGIN PREDICTION
# =============================================================================

def get_team_rating(team: str) -> float:
    """Get team strength rating."""
    return TEAM_RATINGS.get(team, 0.80)


def calculate_expected_margin(
    home_team: str,
    away_team: str,
    factors: List[str],
    home_form: Dict = None,
    away_form: Dict = None
) -> Tuple[float, float]:
    """Calculate expected goal margin for a match.

    Returns:
        Tuple of (expected_margin, std_dev)
    """
    # Get team ratings
    home_rating = get_team_rating(home_team)
    away_rating = get_team_rating(away_team)

    # Base expected margin from ratings
    rating_diff = home_rating - away_rating
    base_margin = rating_diff + HOME_ADVANTAGE

    # Apply form adjustments
    form_mod = 0
    if home_form:
        if home_form.get("status") == "hot":
            form_mod += 0.20
        elif home_form.get("status") == "cold":
            form_mod -= 0.25

    if away_form:
        if away_form.get("status") == "hot":
            form_mod -= 0.15
        elif away_form.get("status") == "cold":
            form_mod += 0.15

    # Apply contextual factors
    factor_mod = 0
    for factor in factors:
        if factor in MARGIN_FACTORS:
            factor_mod += MARGIN_FACTORS[factor]["margin_mod"]

    # Calculate final expected margin
    expected_margin = base_margin + form_mod + factor_mod

    # Calculate standard deviation (larger for closer matches)
    base_std = HISTORICAL_MARGINS["std_dev"]
    # Closer matches have higher variance
    closeness_factor = 1 + (0.3 * (1 - abs(expected_margin) / 3))
    std_dev = base_std * closeness_factor

    return round(expected_margin, 2), round(std_dev, 2)


def predict_margin(
    home_team: str,
    away_team: str,
    date: str,
    factors: List[str],
    home_form: Dict = None,
    away_form: Dict = None,
    ensemble_home_xg: float = None,
    ensemble_away_xg: float = None,
) -> MarginPrediction:
    """Generate margin prediction for a match.

    If ensemble xG is provided, uses it to improve the expected margin estimate.
    """

    # Calculate our own expected margin from team ratings
    own_margin, std_dev = calculate_expected_margin(
        home_team, away_team, factors, home_form, away_form
    )

    # If ensemble provides xG, derive margin from xG difference and blend
    if ensemble_home_xg is not None and ensemble_away_xg is not None:
        xg_margin = ensemble_home_xg - ensemble_away_xg
        expected_margin = 0.7 * xg_margin + 0.3 * own_margin
        log.info(f"  {home_team} vs {away_team}: Blended margin (xG={xg_margin:+.2f}, own={own_margin:+.2f}) -> {expected_margin:+.2f}")
    else:
        expected_margin = own_margin

    # Get team ratings
    home_rating = get_team_rating(home_team)
    away_rating = get_team_rating(away_team)
    rating_diff = home_rating - away_rating

    # Calculate handicap probabilities
    handicap_probs = calculate_all_handicap_probabilities(expected_margin, std_dev)

    # Determine confidence
    n_factors = len([f for f in factors if f in MARGIN_FACTORS])

    if abs(expected_margin) > 1.0 and n_factors >= 2:
        confidence = "HIGH"
        confidence_rank = 4
    elif abs(expected_margin) > 0.5 and n_factors >= 1:
        confidence = "MEDIUM-HIGH"
        confidence_rank = 3
    elif n_factors >= 1:
        confidence = "MEDIUM"
        confidence_rank = 2
    else:
        confidence = "LOW"
        confidence_rank = 1

    return MarginPrediction(
        match=f"{home_team} vs {away_team}",
        home_team=home_team,
        away_team=away_team,
        date=date,
        expected_margin=expected_margin,
        margin_std_dev=std_dev,
        home_rating=home_rating,
        away_rating=away_rating,
        rating_diff=round(rating_diff, 2),
        handicap_probs=handicap_probs,
        factors=factors,
        confidence=confidence,
        confidence_rank=confidence_rank
    )


# =============================================================================
# VALUE BETTING
# =============================================================================

def calculate_handicap_value(
    prediction: MarginPrediction,
    spreads_odds: List[Dict]
) -> List[HandicapBet]:
    """Calculate value bets for handicap markets."""

    bets = []

    for spread in spreads_odds:
        line = spread.get("line", 0)
        home_handicap = spread.get("home_handicap", -line)
        away_handicap = spread.get("away_handicap", line)
        home_odds = spread.get("home_odds", 0)
        away_odds = spread.get("away_odds", 0)

        if home_odds <= 1 or away_odds <= 1:
            continue

        # Reject odds outside sane range (likely live/in-play data)
        if home_odds > ODDS_SANITY_SPREADS["max"] or home_odds < ODDS_SANITY_SPREADS["min"]:
            log.warning(f"Skipping {prediction.match} HOME handicap {home_handicap}: "
                       f"odds {home_odds} outside sane range — likely live/in-play")
            continue
        if away_odds > ODDS_SANITY_SPREADS["max"] or away_odds < ODDS_SANITY_SPREADS["min"]:
            log.warning(f"Skipping {prediction.match} AWAY handicap {away_handicap}: "
                       f"odds {away_odds} outside sane range — likely live/in-play")
            continue

        # Get our probabilities for this line
        # The line represents home handicap
        if home_handicap in prediction.handicap_probs:
            probs = prediction.handicap_probs[home_handicap]
        else:
            # Calculate for non-standard line
            probs = calculate_handicap_probability(
                home_handicap,
                prediction.expected_margin,
                prediction.margin_std_dev
            )

        our_home_prob = probs["home"]
        our_away_prob = probs["away"]

        # Calculate implied probabilities
        implied_home = 1 / home_odds
        implied_away = 1 / away_odds

        # Calculate value for home handicap bet
        home_value = our_home_prob - implied_home
        home_value_pct = round(home_value * 100, 1)

        # Calculate value for away handicap bet
        away_value = our_away_prob - implied_away
        away_value_pct = round(away_value * 100, 1)

        # Determine recommendations
        def get_recommendation(value):
            if value > 0.08:  # 8%+ edge
                return "BET", 2.0
            elif value > 0.03:  # 3-8% edge
                return "CONSIDER", 1.0
            else:
                return "SKIP", 0

        home_rec, home_stake = get_recommendation(home_value)
        away_rec, away_stake = get_recommendation(away_value)

        # Add home handicap bet if there's value
        if home_value > 0:
            bets.append(HandicapBet(
                match=prediction.match,
                date=prediction.date,
                handicap_line=home_handicap,
                bet_side="home",
                our_probability=our_home_prob,
                odds=home_odds,
                implied_probability=round(implied_home, 3),
                value=round(home_value, 3),
                value_pct=home_value_pct,
                expected_margin=prediction.expected_margin,
                recommendation=home_rec,
                stake_pct=home_stake,
                factors=prediction.factors
            ))

        # Add away handicap bet if there's value
        if away_value > 0:
            bets.append(HandicapBet(
                match=prediction.match,
                date=prediction.date,
                handicap_line=away_handicap,
                bet_side="away",
                our_probability=our_away_prob,
                odds=away_odds,
                implied_probability=round(implied_away, 3),
                value=round(away_value, 3),
                value_pct=away_value_pct,
                expected_margin=prediction.expected_margin,
                recommendation=away_rec,
                stake_pct=away_stake,
                factors=prediction.factors
            ))

    # Sort by value
    bets.sort(key=lambda x: x.value, reverse=True)

    return bets


# =============================================================================
# DATA LOADING
# =============================================================================

def load_predictions() -> List[Dict]:
    """Load match predictions."""
    predictions_path = DATA_DIR / "upcoming" / "predictions.json"

    if not predictions_path.exists():
        log.warning("No predictions file found")
        return []

    with open(predictions_path) as f:
        data = json.load(f)

    return data.get("predictions", [])


def load_spreads_odds() -> Dict[str, List[Dict]]:
    """Load handicap/spread odds."""
    odds_path = DATA_DIR / "upcoming" / "odds_spreads.json"

    if not odds_path.exists():
        log.warning("No spreads odds file found")
        return {}

    with open(odds_path) as f:
        data = json.load(f)

    return data


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def generate_handicap_predictions() -> Tuple[List[MarginPrediction], List[HandicapBet]]:
    """Generate handicap predictions and betting recommendations."""

    log.info("=" * 60)
    log.info("HANDICAP/SPREAD PREDICTION MODEL")
    log.info("=" * 60)

    # Load data
    predictions = load_predictions()
    spreads_odds = load_spreads_odds()

    if not predictions:
        log.error("No match predictions available")
        return [], []

    log.info(f"Loaded {len(predictions)} match predictions")
    log.info(f"Loaded spreads odds for {len(spreads_odds)} matches")

    margin_predictions = []
    all_bets = []

    for pred in predictions:
        match_key = pred["match"]
        home_team = match_key.split(" vs ")[0]
        away_team = match_key.split(" vs ")[1]
        date = pred["date"]

        # Collect factors
        factors = []
        for f in pred.get("home_factors", []):
            factors.append(f)
        for f in pred.get("away_factors", []):
            factors.append(f)
        for f in pred.get("neutral_factors", []):
            factors.append(f)

        # Get form data
        home_form = pred.get("home_form", {})
        away_form = pred.get("away_form", {})

        # Generate margin prediction (using ensemble xG when available)
        margin_pred = predict_margin(
            home_team=home_team,
            away_team=away_team,
            date=date,
            factors=factors,
            home_form=home_form,
            away_form=away_form,
            ensemble_home_xg=pred.get("home_xg"),
            ensemble_away_xg=pred.get("away_xg"),
        )
        margin_predictions.append(margin_pred)

        # Calculate value bets if odds available
        if match_key in spreads_odds:
            match_spreads = spreads_odds[match_key].get("spreads", [])
            bets = calculate_handicap_value(margin_pred, match_spreads)
            all_bets.extend(bets)

    # Sort predictions by expected margin
    margin_predictions.sort(key=lambda x: x.expected_margin, reverse=True)

    log.info(f"Generated {len(margin_predictions)} margin predictions")
    log.info(f"Found {len(all_bets)} potential handicap bets")

    return margin_predictions, all_bets


def save_handicap_predictions(
    predictions: List[MarginPrediction],
    bets: List[HandicapBet]
):
    """Save predictions and bets to JSON files."""

    output_dir = DATA_DIR / "upcoming"

    # Save margin predictions
    predictions_data = {
        "generated_at": datetime.now().isoformat(),
        "model": "normal_margin_v1",
        "predictions": [
            {
                "match": p.match,
                "date": p.date,
                "expected_margin": p.expected_margin,
                "margin_std_dev": p.margin_std_dev,
                "home_rating": p.home_rating,
                "away_rating": p.away_rating,
                "rating_diff": p.rating_diff,
                "confidence": p.confidence,
                "factors": p.factors,
                "handicap_probs": {
                    str(k): v for k, v in list(p.handicap_probs.items())[:7]  # Main lines
                }
            }
            for p in predictions
        ]
    }

    pred_path = output_dir / "margin_predictions.json"
    with open(pred_path, "w") as f:
        json.dump(predictions_data, f, indent=2)
    log.info(f"Saved margin predictions to {pred_path}")

    # Save betting recommendations
    recommended = [b for b in bets if b.recommendation == "BET"]
    consider = [b for b in bets if b.recommendation == "CONSIDER"]

    # Helper to format bet with Italian market standards
    def format_bet_italian(bet_side: str, handicap_line: float) -> dict:
        """Convert handicap line to Italian market standard."""
        italian_line = normalize_line_to_italian(handicap_line, "handicap")
        is_converted = italian_line != handicap_line
        return {
            "bet": f"{bet_side.upper()} {italian_line:+.0f}" if italian_line == int(italian_line) else f"{bet_side.upper()} {italian_line:+.1f}",
            "original_line": handicap_line if is_converted else None,
            "converted_to_italian": is_converted,
            "italian_format": True
        }

    bets_data = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_bets_analyzed": len(bets),
            "recommended_bets": len(recommended),
            "consider_bets": len(consider),
            "italian_standard_bets": len([b for b in bets if normalize_line_to_italian(b.handicap_line, "handicap") == int(normalize_line_to_italian(b.handicap_line, "handicap"))])
        },
        "recommended": [
            {
                "match": b.match,
                "date": b.date,
                **format_bet_italian(b.bet_side, b.handicap_line),
                "our_probability": b.our_probability,
                "odds": b.odds,
                "implied_probability": b.implied_probability,
                "value_pct": b.value_pct,
                "expected_margin": b.expected_margin,
                "stake_pct": b.stake_pct,
                "factors": b.factors
            }
            for b in recommended
        ],
        "consider": [
            {
                "match": b.match,
                "date": b.date,
                **format_bet_italian(b.bet_side, b.handicap_line),
                "our_probability": b.our_probability,
                "odds": b.odds,
                "value_pct": b.value_pct,
                "expected_margin": b.expected_margin
            }
            for b in consider
        ]
    }

    bets_path = output_dir / "handicap_bets.json"
    with open(bets_path, "w") as f:
        json.dump(bets_data, f, indent=2)
    log.info(f"Saved betting recommendations to {bets_path}")


def print_summary(predictions: List[MarginPrediction], bets: List[HandicapBet]):
    """Print summary of predictions and bets."""

    print("\n" + "=" * 80)
    print(" HANDICAP/MARGIN PREDICTIONS")
    print("=" * 80)

    print(f"\n  {'Match':<30} {'Margin':>8} {'StdDev':>8} {'H-1.5':>7} {'H+0':>7} {'Conf':>10}")
    print("  " + "-" * 75)

    for p in predictions:
        h_minus_1_5 = p.handicap_probs.get(-1.5, {}).get("home", 0)
        h_0 = p.handicap_probs.get(0, {}).get("home", 0)

        print(f"  {p.match:<30} {p.expected_margin:>+8.2f} {p.margin_std_dev:>8.2f} "
              f"{h_minus_1_5:>7.1%} {h_0:>7.1%} {p.confidence:>10}")

    # Betting recommendations
    recommended = [b for b in bets if b.recommendation == "BET"]
    consider = [b for b in bets if b.recommendation == "CONSIDER"]

    if recommended:
        print("\n" + "=" * 80)
        print(" RECOMMENDED HANDICAP BETS")
        print("=" * 80)

        print(f"\n  {'Match':<25} {'Bet':>12} {'Odds':>6} {'Our%':>6} {'Imp%':>6} {'Value':>7}")
        print("  " + "-" * 70)

        for b in recommended:
            bet_str = f"{b.bet_side.upper()} {b.handicap_line:+.1f}"
            print(f"  {b.match:<25} {bet_str:>12} {b.odds:>6.2f} "
                  f"{b.our_probability:>6.1%} {b.implied_probability:>6.1%} {b.value_pct:>+7.1f}%")

    if consider:
        print("\n" + "=" * 80)
        print(" CONSIDER BETS (Small Edge)")
        print("=" * 80)

        for b in consider[:5]:
            bet_str = f"{b.bet_side.upper()} {b.handicap_line:+.1f}"
            print(f"  {b.match}: {bet_str} @ {b.odds:.2f} "
                  f"(Value: {b.value_pct:+.1f}%, Margin: {b.expected_margin:+.2f})")


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Handicap/Spread Prediction Model")
    parser.add_argument("--validate", action="store_true", help="Run model validation")
    args = parser.parse_args()

    if args.validate:
        # Run validation
        print("\n" + "=" * 60)
        print(" MODEL VALIDATION")
        print("=" * 60)

        test_cases = [
            ("Inter", "Sassuolo", ["big_away_favorite", "hot_away"]),
            ("Juventus", "Lazio", ["hot_home", "hot_away"]),
            ("Lecce", "Udinese", ["cold_away", "cold_home"]),
        ]

        for home, away, factors in test_cases:
            pred = predict_margin(home, away, "2026-02-09", factors)
            print(f"\n  {pred.match}:")
            print(f"    Expected Margin: {pred.expected_margin:+.2f}")
            print(f"    Home -1.5: {pred.handicap_probs[-1.5]['home']:.1%}")
            print(f"    Home +0: {pred.handicap_probs[0]['home']:.1%}")
            print(f"    Confidence: {pred.confidence}")
        return

    # Generate predictions and bets
    predictions, bets = generate_handicap_predictions()

    if predictions:
        # Save results
        save_handicap_predictions(predictions, bets)

        # Print summary
        print_summary(predictions, bets)


if __name__ == "__main__":
    main()
