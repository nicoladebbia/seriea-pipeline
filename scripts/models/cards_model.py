#!/usr/bin/env python3
"""CARDS PREDICTION MODEL - Phase 5 Implementation

Predicts booking points and card totals for matches.

Features:
- Referee strictness analysis (validated factor)
- Derby/intensity factor integration
- Team discipline records
- Weather impact on fouls
- Poisson distribution for card counts
- Multiple thresholds (3.5, 4.5, 5.5, 6.5 cards)

Note: The Odds API doesn't provide cards odds for Serie A.
This model generates predictions that can be used with external
bookmakers that offer cards markets (Bet365, William Hill, etc.)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR
from ml.poisson import poisson_probability, poisson_cumulative, calculate_over_probability
from scripts.models import load_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# =============================================================================
# HISTORICAL DATA & CONSTANTS
# =============================================================================

# Serie A average cards per game (from 21-season analysis)
SERIE_A_AVG_CARDS = 4.5  # Total yellow + 2*red
SERIE_A_AVG_YELLOWS = 4.2
SERIE_A_AVG_REDS = 0.15

# Historical over rates
HISTORICAL_CARD_RATES = {
    2.5: 0.92,  # 92% over 2.5 cards
    3.5: 0.78,  # 78% over 3.5 cards
    4.5: 0.52,  # 52% over 4.5 cards
    5.5: 0.28,  # 28% over 5.5 cards
    6.5: 0.12,  # 12% over 6.5 cards
}

# Referee strictness ratings (from our validated data)
# Higher = more cards
REFEREE_STRICTNESS = {
    "Marco Guida": 1.25,  # Very strict
    "Gianluca Manganiello": 1.20,
    "Daniele Orsato": 1.15,
    "Maurizio Mariani": 1.10,
    "Fabio Maresca": 1.10,
    "Marco Di Bello": 1.05,
    "Rosario Abisso": 1.05,
    "Davide Massa": 1.00,
    "Andrea Colombo": 0.95,
    "Simone Sozza": 0.95,
    "Daniele Doveri": 0.90,
    "Federico Dionisi": 0.88,
    "Giovanni Ayroldi": 0.85,
}

# Team discipline (cards per game above/below average)
TEAM_DISCIPLINE = {
    # Undisciplined teams (more cards)
    "Roma": 1.15,
    "Napoli": 1.10,
    "Lazio": 1.10,
    "Fiorentina": 1.08,
    "Juventus": 1.05,
    "Inter": 1.02,
    "Milan": 1.00,

    # Average discipline
    "Atalanta": 1.00,
    "Bologna": 0.98,
    "Torino": 0.98,
    "Genoa": 1.02,
    "Udinese": 0.95,

    # Disciplined teams (fewer cards)
    "Sassuolo": 0.92,
    "Lecce": 0.95,
    "Cagliari": 0.93,
    "Como": 0.95,
    "Parma": 0.93,
    "Verona": 0.95,
    "Cremonese": 0.90,
    "Pisa": 0.92,
}

# Cards factors
CARDS_FACTORS = {
    "derby": 1.35,         # Derby matches = more cards
    "rivalry": 1.25,       # Historic rivalries
    "high_stakes": 1.15,   # Important matches
    "strict_ref": 1.20,    # Strict referee (from our factor)
    "lenient_ref": 0.85,   # Lenient referee
    "hot_weather": 0.95,   # Hot = slower pace, fewer fouls
    "rain": 1.10,          # Rain = more slips, more fouls
    "wind": 1.05,          # Wind = more unpredictability
    "cold": 1.05,          # Cold = more physical
}

# Known derbies and rivalries
DERBIES = {
    ("Inter", "Milan"): "derby",      # Derby della Madonnina
    ("Roma", "Lazio"): "derby",       # Derby della Capitale
    ("Juventus", "Torino"): "derby",  # Derby della Mole
    ("Genoa", "Sampdoria"): "derby",  # Derby della Lanterna
    ("Inter", "Juventus"): "rivalry", # Derby d'Italia
    ("Milan", "Juventus"): "rivalry",
    ("Roma", "Napoli"): "rivalry",
    ("Napoli", "Juventus"): "rivalry",
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CardsPrediction:
    """Prediction for match cards/bookings."""
    match: str
    home_team: str
    away_team: str
    date: str

    # Expected cards
    expected_cards: float
    expected_yellows: float
    expected_reds: float

    # Probabilities
    over_3_5: float
    over_4_5: float
    over_5_5: float
    over_6_5: float

    # Booking points (y=1, r=3)
    expected_booking_points: float
    over_30_5_points: float  # Standard line
    over_40_5_points: float

    # Factors
    referee: str
    referee_strictness: float
    is_derby: bool
    factors: List[str]

    # Confidence
    confidence: str
    confidence_rank: int


@dataclass
class CardsBet:
    """Betting recommendation for cards market."""
    match: str
    date: str
    market: str  # "cards" or "booking_points"
    bet_type: str  # "over" or "under"
    line: float
    our_probability: float
    suggested_odds: float  # Fair odds based on our probability
    expected_cards: float
    recommendation: str
    factors: List[str]




# =============================================================================
# PREDICTION ENGINE
# =============================================================================

def get_referee_strictness(referee: str) -> float:
    """Get referee strictness rating."""
    return REFEREE_STRICTNESS.get(referee, 1.0)


def get_team_discipline(team: str) -> float:
    """Get team discipline factor."""
    return TEAM_DISCIPLINE.get(team, 1.0)


def check_derby(home: str, away: str) -> Tuple[bool, str]:
    """Check if match is a derby or rivalry."""
    pair = (home, away)
    reverse = (away, home)

    if pair in DERBIES:
        return True, DERBIES[pair]
    if reverse in DERBIES:
        return True, DERBIES[reverse]

    return False, ""


def calculate_expected_cards(
    home_team: str,
    away_team: str,
    referee: str,
    factors: List[str],
    home_lineup: List[str] = None,
    away_lineup: List[str] = None,
) -> Tuple[float, float, float]:
    """Calculate expected cards for a match.

    Args:
        home_team: Home team name
        away_team: Away team name
        referee: Referee name
        factors: List of match factors
        home_lineup: Optional confirmed home lineup for per-player card rates
        away_lineup: Optional confirmed away lineup for per-player card rates

    Returns:
        Tuple of (total_cards, yellows, reds)
    """
    # Base expected cards
    base_cards = SERIE_A_AVG_CARDS

    # Referee adjustment
    ref_strictness = get_referee_strictness(referee)

    # Team discipline
    home_discipline = get_team_discipline(home_team)
    away_discipline = get_team_discipline(away_team)
    team_factor = (home_discipline + away_discipline) / 2

    # Check derby
    is_derby, derby_type = check_derby(home_team, away_team)
    derby_factor = CARDS_FACTORS.get(derby_type, 1.0) if is_derby else 1.0

    # Apply other factors
    other_factor = 1.0
    for f in factors:
        if f in CARDS_FACTORS:
            other_factor *= CARDS_FACTORS[f]

    # Calculate final expected cards
    expected = base_cards * ref_strictness * team_factor * derby_factor * other_factor

    # Lineup-based card/foul adjustment
    try:
        from features.lineup_stats import get_lineup_expected_cards, get_lineup_foul_rate
        home_cards = get_lineup_expected_cards(home_team, home_lineup)
        away_cards = get_lineup_expected_cards(away_team, away_lineup)
        home_fouls = get_lineup_foul_rate(home_team, home_lineup)
        away_fouls = get_lineup_foul_rate(away_team, away_lineup)

        # Blend lineup card data with team-level estimate (30% lineup, 70% team)
        lineup_yellows = home_cards["expected_yellows"] + away_cards["expected_yellows"]
        if lineup_yellows > 0 and home_cards["confidence"] != "low":
            expected = 0.7 * expected + 0.3 * lineup_yellows

        # High foul rate teams get more cards — adjust if lineup data available
        total_fouls = home_fouls["fouls_per_90"] + away_fouls["fouls_per_90"]
        league_avg_fouls = 24.0  # ~12 fouls/team/match
        if total_fouls > 0 and home_fouls["confidence"] != "low":
            foul_ratio = total_fouls / league_avg_fouls
            # Only adjust if significantly different (±15%+)
            if abs(foul_ratio - 1.0) > 0.15:
                foul_adj = (foul_ratio - 1.0) * 0.3  # Moderate impact
                expected *= (1.0 + foul_adj)
    except Exception:
        pass  # Lineup stats not available

    # Split into yellows and reds
    # Typically yellows = 90%, reds = 3% of cards impact
    expected_yellows = expected * 0.93
    expected_reds = expected * 0.035

    return (
        round(expected, 2),
        round(expected_yellows, 2),
        round(expected_reds, 3)
    )


def calculate_booking_points(expected_yellows: float, expected_reds: float) -> float:
    """Calculate expected booking points.

    Standard: Yellow = 10 points, Red = 25 points
    """
    return expected_yellows * 10 + expected_reds * 25


def predict_cards(
    home_team: str,
    away_team: str,
    date: str,
    referee: str,
    factors: List[str],
    home_lineup: List[str] = None,
    away_lineup: List[str] = None,
) -> CardsPrediction:
    """Generate cards prediction for a match."""

    # Calculate expected cards (with lineup awareness)
    expected_cards, expected_yellows, expected_reds = calculate_expected_cards(
        home_team, away_team, referee, factors,
        home_lineup=home_lineup, away_lineup=away_lineup,
    )

    # Calculate over probabilities
    over_3_5 = calculate_over_probability(3.5, expected_cards)
    over_4_5 = calculate_over_probability(4.5, expected_cards)
    over_5_5 = calculate_over_probability(5.5, expected_cards)
    over_6_5 = calculate_over_probability(6.5, expected_cards)

    # Booking points
    expected_points = calculate_booking_points(expected_yellows, expected_reds)
    over_30_5_points = calculate_over_probability(30.5 / 10, expected_yellows) * 0.9  # Approximation
    over_40_5_points = calculate_over_probability(40.5 / 10, expected_yellows) * 0.85

    # Check derby
    is_derby, _ = check_derby(home_team, away_team)

    # Referee strictness
    ref_strictness = get_referee_strictness(referee)

    # Determine confidence
    # High confidence if strong factors present
    strong_factors = sum(1 for f in factors if f in ["strict_ref", "derby"])
    deviation = abs(expected_cards - SERIE_A_AVG_CARDS)

    if strong_factors >= 1 and deviation > 0.5:
        confidence = "HIGH"
        confidence_rank = 4
    elif deviation > 0.3:
        confidence = "MEDIUM-HIGH"
        confidence_rank = 3
    elif strong_factors >= 1:
        confidence = "MEDIUM"
        confidence_rank = 2
    else:
        confidence = "LOW"
        confidence_rank = 1

    return CardsPrediction(
        match=f"{home_team} vs {away_team}",
        home_team=home_team,
        away_team=away_team,
        date=date,
        expected_cards=expected_cards,
        expected_yellows=expected_yellows,
        expected_reds=expected_reds,
        over_3_5=round(over_3_5, 3),
        over_4_5=round(over_4_5, 3),
        over_5_5=round(over_5_5, 3),
        over_6_5=round(over_6_5, 3),
        expected_booking_points=round(expected_points, 1),
        over_30_5_points=round(over_30_5_points, 3),
        over_40_5_points=round(over_40_5_points, 3),
        referee=referee,
        referee_strictness=ref_strictness,
        is_derby=is_derby,
        factors=factors,
        confidence=confidence,
        confidence_rank=confidence_rank
    )


# =============================================================================
# BETTING RECOMMENDATIONS
# =============================================================================

def generate_cards_bets(prediction: CardsPrediction) -> List[CardsBet]:
    """Generate betting recommendations based on prediction.

    Since we don't have real cards odds, we provide fair odds
    and recommendations based on probability.
    """
    bets = []

    # Standard lines
    lines = [
        (3.5, prediction.over_3_5),
        (4.5, prediction.over_4_5),
        (5.5, prediction.over_5_5),
        (6.5, prediction.over_6_5),
    ]

    for line, over_prob in lines:
        # Calculate fair odds
        fair_over_odds = 1 / over_prob if over_prob > 0.01 else 100
        fair_under_odds = 1 / (1 - over_prob) if (1 - over_prob) > 0.01 else 100

        # Determine recommendation based on deviation from average
        if prediction.expected_cards > SERIE_A_AVG_CARDS + 0.3:
            # Expect more cards than average - lean over
            if over_prob > 0.55:  # Good edge on over
                recommendation = "BET" if over_prob > 0.65 else "CONSIDER"
                bets.append(CardsBet(
                    match=prediction.match,
                    date=prediction.date,
                    market="cards",
                    bet_type="over",
                    line=line,
                    our_probability=over_prob,
                    suggested_odds=round(fair_over_odds, 2),
                    expected_cards=prediction.expected_cards,
                    recommendation=recommendation,
                    factors=prediction.factors
                ))
        elif prediction.expected_cards < SERIE_A_AVG_CARDS - 0.3:
            # Expect fewer cards - lean under
            under_prob = 1 - over_prob
            if under_prob > 0.55:
                recommendation = "BET" if under_prob > 0.65 else "CONSIDER"
                bets.append(CardsBet(
                    match=prediction.match,
                    date=prediction.date,
                    market="cards",
                    bet_type="under",
                    line=line,
                    our_probability=under_prob,
                    suggested_odds=round(fair_under_odds, 2),
                    expected_cards=prediction.expected_cards,
                    recommendation=recommendation,
                    factors=prediction.factors
                ))

    return bets



# =============================================================================
# MAIN PIPELINE
# =============================================================================

def generate_cards_predictions() -> Tuple[List[CardsPrediction], List[CardsBet]]:
    """Generate cards predictions for all matches."""

    log.info("=" * 60)
    log.info("CARDS PREDICTION MODEL")
    log.info("=" * 60)

    # Load predictions
    predictions = load_predictions()

    if not predictions:
        log.error("No match predictions available")
        return [], []

    log.info(f"Loaded {len(predictions)} match predictions")

    # Load confirmed lineups if available
    confirmed_lineups = {}
    try:
        lineup_path = DATA_DIR / "upcoming" / "confirmed_lineups.json"
        if lineup_path.exists():
            with open(lineup_path) as f:
                lineup_data = json.load(f)
            confirmed_lineups = lineup_data.get("matches", {})
            if confirmed_lineups:
                log.info(f"Loaded confirmed lineups for {len(confirmed_lineups)} match(es)")
    except Exception as e:
        log.warning(f"Failed to load confirmed lineups for cards model: {e}")

    cards_predictions = []
    all_bets = []

    for pred in predictions:
        match_key = pred["match"]
        home_team = match_key.split(" vs ")[0]
        away_team = match_key.split(" vs ")[1]
        date = pred["date"]
        referee = pred.get("referee", "Unknown")

        # Get confirmed lineup data
        match_lineup = confirmed_lineups.get(match_key, {})
        home_lineup = match_lineup.get("home_lineup")
        away_lineup = match_lineup.get("away_lineup")

        # Collect factors that affect cards
        factors = []

        # Referee bias factor
        ref_bias = pred.get("referee_bias", "")
        if ref_bias == "strict":
            factors.append("strict_ref")

        # Weather factors
        for f in pred.get("neutral_factors", []):
            if f in CARDS_FACTORS:
                factors.append(f)

        # Generate prediction (with lineup awareness)
        cards_pred = predict_cards(
            home_team=home_team,
            away_team=away_team,
            date=date,
            referee=referee,
            factors=factors,
            home_lineup=home_lineup,
            away_lineup=away_lineup,
        )
        cards_predictions.append(cards_pred)

        # Generate betting recommendations
        bets = generate_cards_bets(cards_pred)
        all_bets.extend(bets)

    # Sort by expected cards (high to low)
    cards_predictions.sort(key=lambda x: x.expected_cards, reverse=True)

    log.info(f"Generated {len(cards_predictions)} cards predictions")
    log.info(f"Found {len(all_bets)} potential cards bets")

    return cards_predictions, all_bets


def save_cards_predictions(
    predictions: List[CardsPrediction],
    bets: List[CardsBet]
):
    """Save predictions and bets to JSON files."""

    output_dir = DATA_DIR / "upcoming"

    # Save predictions
    predictions_data = {
        "generated_at": datetime.now().isoformat(),
        "model": "poisson_cards_v1",
        "note": "Cards odds not available via API - use with external bookmakers",
        "predictions": [
            {
                "match": p.match,
                "date": p.date,
                "expected_cards": p.expected_cards,
                "expected_yellows": p.expected_yellows,
                "expected_reds": p.expected_reds,
                "over_3_5": p.over_3_5,
                "over_4_5": p.over_4_5,
                "over_5_5": p.over_5_5,
                "over_6_5": p.over_6_5,
                "booking_points": p.expected_booking_points,
                "referee": p.referee,
                "referee_strictness": p.referee_strictness,
                "is_derby": p.is_derby,
                "confidence": p.confidence,
                "factors": p.factors
            }
            for p in predictions
        ]
    }

    pred_path = output_dir / "cards_predictions.json"
    with open(pred_path, "w") as f:
        json.dump(predictions_data, f, indent=2)
    log.info(f"Saved cards predictions to {pred_path}")

    # Save betting recommendations
    recommended = [b for b in bets if b.recommendation == "BET"]
    consider = [b for b in bets if b.recommendation == "CONSIDER"]

    bets_data = {
        "generated_at": datetime.now().isoformat(),
        "note": "Compare suggested odds with bookmaker odds - bet when bookmaker offers higher",
        "summary": {
            "total_analyzed": len(bets),
            "recommended": len(recommended),
            "consider": len(consider)
        },
        "recommended": [
            {
                "match": b.match,
                "date": b.date,
                "bet": f"{b.bet_type.upper()} {b.line}",
                "our_probability": b.our_probability,
                "fair_odds": b.suggested_odds,
                "expected_cards": b.expected_cards,
                "factors": b.factors
            }
            for b in recommended
        ],
        "consider": [
            {
                "match": b.match,
                "date": b.date,
                "bet": f"{b.bet_type.upper()} {b.line}",
                "our_probability": b.our_probability,
                "fair_odds": b.suggested_odds,
                "expected_cards": b.expected_cards
            }
            for b in consider
        ]
    }

    bets_path = output_dir / "cards_bets.json"
    with open(bets_path, "w") as f:
        json.dump(bets_data, f, indent=2)
    log.info(f"Saved cards bets to {bets_path}")


def print_summary(predictions: List[CardsPrediction], bets: List[CardsBet]):
    """Print summary of predictions."""

    print("\n" + "=" * 80)
    print(" CARDS/BOOKING PREDICTIONS")
    print("=" * 80)

    print(f"\n  Note: Cards odds not available via API. Use these predictions")
    print(f"        with bookmakers that offer cards markets (Bet365, etc.)")

    print(f"\n  {'Match':<30} {'xCards':>7} {'O4.5':>6} {'O5.5':>6} {'Ref':>15} {'Conf':>10}")
    print("  " + "-" * 80)

    for p in predictions:
        ref_display = p.referee[:12] + "..." if len(p.referee) > 15 else p.referee
        print(f"  {p.match:<30} {p.expected_cards:>7.2f} {p.over_4_5:>6.1%} "
              f"{p.over_5_5:>6.1%} {ref_display:>15} {p.confidence:>10}")

    # Betting recommendations
    recommended = [b for b in bets if b.recommendation == "BET"]
    consider = [b for b in bets if b.recommendation == "CONSIDER"]

    if recommended:
        print("\n" + "=" * 80)
        print(" RECOMMENDED CARDS BETS")
        print("=" * 80)
        print("\n  Look for odds HIGHER than fair odds shown below:")

        print(f"\n  {'Match':<25} {'Bet':>10} {'Our%':>7} {'Fair':>6} {'xCards':>7}")
        print("  " + "-" * 60)

        for b in recommended:
            bet_str = f"{b.bet_type.upper()} {b.line}"
            print(f"  {b.match:<25} {bet_str:>10} {b.our_probability:>7.1%} "
                  f"{b.suggested_odds:>6.2f} {b.expected_cards:>7.2f}")

    if consider:
        print("\n" + "=" * 80)
        print(" CONSIDER BETS (if odds are good)")
        print("=" * 80)

        for b in consider[:5]:
            bet_str = f"{b.bet_type.upper()} {b.line}"
            print(f"  {b.match}: {bet_str} (Fair: {b.suggested_odds:.2f}, xCards: {b.expected_cards:.2f})")


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Cards Prediction Model")
    parser.add_argument("--validate", action="store_true", help="Run model validation")
    args = parser.parse_args()

    if args.validate:
        print("\n" + "=" * 60)
        print(" CARDS MODEL VALIDATION")
        print("=" * 60)

        # Test with different scenarios
        test_cases = [
            ("Roma", "Lazio", "Marco Guida", ["strict_ref"]),  # Derby + strict ref
            ("Inter", "Sassuolo", "Daniele Doveri", []),  # Big team vs small
            ("Lecce", "Udinese", "Andrea Colombo", []),  # Mid-table
        ]

        for home, away, ref, factors in test_cases:
            pred = predict_cards(home, away, "2026-02-09", ref, factors)
            derby_str = " (DERBY)" if pred.is_derby else ""
            print(f"\n  {pred.match}{derby_str}:")
            print(f"    Referee: {ref} (strictness: {pred.referee_strictness:.2f})")
            print(f"    Expected Cards: {pred.expected_cards:.2f}")
            print(f"    Over 4.5: {pred.over_4_5:.1%} | Over 5.5: {pred.over_5_5:.1%}")
            print(f"    Confidence: {pred.confidence}")

        return

    # Generate predictions
    predictions, bets = generate_cards_predictions()

    if predictions:
        # Save results
        save_cards_predictions(predictions, bets)

        # Print summary
        print_summary(predictions, bets)


if __name__ == "__main__":
    main()
