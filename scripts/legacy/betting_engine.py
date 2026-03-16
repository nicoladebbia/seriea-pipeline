#!/usr/bin/env python3
"""BETTING DECISION ENGINE - Intelligent Bet Selection

Combines predictions with bankroll management to generate optimal bets:
1. Value Betting Calculator
2. Single vs Parlay Logic
3. Odds Comparison (if provided)
4. Bet Recommendations with Stakes

This is the brain that decides WHAT to bet and HOW MUCH.

Usage:
    python scripts/betting_engine.py              # Generate bet recommendations
    python scripts/betting_engine.py --odds-file odds.json  # With odds
    python scripts/betting_engine.py --parlay    # Include parlay suggestions
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from itertools import combinations

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DATA_DIR

# Import our modules
try:
    from scripts.bankroll_manager import (
        load_bankroll, get_stake_recommendation, calculate_value,
        create_bet, place_bet, FLAT_STAKES
    )
except ImportError:
    from bankroll_manager import (
        load_bankroll, get_stake_recommendation, calculate_value,
        create_bet, place_bet, FLAT_STAKES
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Minimum confidence for betting
MIN_CONFIDENCE_SINGLE = "MEDIUM"  # Minimum for single bets
MIN_CONFIDENCE_PARLAY = "HIGH"    # Minimum for parlays

# Minimum value (edge) required
MIN_VALUE_SINGLE = 0.05   # 5% edge for singles
MIN_VALUE_PARLAY = 0.08   # 8% edge for parlay legs

# Parlay settings
MAX_PARLAY_LEGS = 3
MIN_PARLAY_LEGS = 2
PARLAY_STAKE_PCT = 0.01   # 1% of bankroll for parlays

# Market-specific odds limits — odds outside these are almost certainly live/in-play
# data errors, not real pre-match prices. Mirrors advanced_betting_engine.ODDS_SANITY.
ODDS_SANITY = {
    "h2h":     {"min": 1.05, "max": 25.0},
    "totals":  {"min": 1.05, "max": 6.0},   # Over 1.5 at 12.0 = live odds
    "spreads": {"min": 1.05, "max": 6.0},   # HOME +0 at 8.6 = live odds
    "handicap": {"min": 1.05, "max": 6.0},
    "btts":    {"min": 1.10, "max": 8.0},
    "corners": {"min": 1.10, "max": 10.0},
    "cards":   {"min": 1.10, "max": 10.0},
}

# Maximum credible value edge — above this, the odds are likely wrong
MAX_CREDIBLE_VALUE_PCT = 50.0

# Confidence ranking
CONFIDENCE_RANK = {
    "VERY_HIGH": 5,
    "VERY HIGH": 5,
    "HIGH": 4,
    "MEDIUM_HIGH": 3,
    "MEDIUM-HIGH": 3,
    "MEDIUM HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1,
}


def normalize_confidence(conf: str) -> str:
    """Normalize confidence string."""
    return conf.upper().replace("-", "_").replace(" ", "_")


def get_confidence_rank(conf: str) -> int:
    """Get numeric rank for confidence level."""
    norm = normalize_confidence(conf)
    return CONFIDENCE_RANK.get(norm, 1)


def load_predictions() -> List[Dict]:
    """Load predictions from file."""
    pred_path = DATA_DIR / "upcoming" / "predictions.json"

    if not pred_path.exists():
        log.warning("No predictions file found")
        return []

    with open(pred_path) as f:
        data = json.load(f)

    return data.get("predictions", [])


def load_odds(odds_file: str = None) -> Dict[str, Dict]:
    """Load odds from file.

    Expected format:
    {
        "Inter vs Milan": {"home": 1.85, "draw": 3.50, "away": 4.20},
        "Juventus vs Roma": {"home": 1.65, "draw": 3.80, "away": 5.00}
    }
    """
    if odds_file:
        odds_path = Path(odds_file)
    else:
        odds_path = DATA_DIR / "upcoming" / "odds.json"

    if not odds_path.exists():
        log.info("No odds file found - using estimated odds")
        return {}

    with open(odds_path) as f:
        return json.load(f)


def estimate_odds(probability: float) -> float:
    """Estimate decimal odds from probability.

    Adds typical bookmaker margin (~10%).
    """
    if probability <= 0 or probability >= 1:
        return 2.0

    fair_odds = 1 / probability
    # Apply 10% margin (reduce payout)
    estimated_odds = fair_odds * 0.90
    return round(estimated_odds, 2)


def analyze_single_bets(predictions: List[Dict], odds: Dict, bankroll: float) -> List[Dict]:
    """Analyze predictions for single bet opportunities.

    Args:
        predictions: List of predictions
        odds: Dict of odds by match
        bankroll: Current bankroll

    Returns:
        List of bet recommendations sorted by value
    """
    recommendations = []

    for pred in predictions:
        match = pred.get("match", "Unknown")

        # Skip matches flagged with severe market anomalies
        anomaly = pred.get("market_anomaly", {})
        if anomaly.get("severity", 0) >= 2:
            log.warning(f"Skipping {match}: market anomaly — {anomaly.get('reasons', [])}")
            continue

        outcome = pred.get("predicted_outcome", "HOME")
        confidence = pred.get("confidence_level", pred.get("confidence", "LOW"))
        if isinstance(confidence, (int, float)):
            # Ensemble outputs numeric confidence; map to label
            if confidence >= 0.55:
                confidence = "HIGH"
            elif confidence >= 0.50:
                confidence = "MEDIUM-HIGH"
            else:
                confidence = "MEDIUM"
        probs = pred.get("probabilities", {})

        # Get probability for predicted outcome
        if outcome == "HOME":
            win_prob = probs.get("home", 0.5)
            odds_key = "home"
        elif outcome == "AWAY":
            win_prob = probs.get("away", 0.3)
            odds_key = "away"
        else:
            win_prob = probs.get("draw", 0.25)
            odds_key = "draw"

        # Get actual or estimated odds
        match_odds = odds.get(match, {})
        decimal_odds = match_odds.get(odds_key, estimate_odds(win_prob))

        # Odds sanity check — this function only handles h2h (match outcome) bets
        market = "h2h"
        limits = ODDS_SANITY["h2h"]
        if decimal_odds < limits["min"] or decimal_odds > limits["max"]:
            log.warning(f"Skipping {match}: odds {decimal_odds} outside sane range "
                       f"[{limits['min']}, {limits['max']}] for {market} — likely live/in-play data")
            continue
        # Reject if value would be absurdly high (data error, not real edge)
        implied_prob = 1.0 / decimal_odds
        raw_value_pct = (win_prob - implied_prob) / implied_prob * 100 if implied_prob > 0 else 0
        if raw_value_pct > MAX_CREDIBLE_VALUE_PCT:
            log.warning(f"Skipping {match}: value {raw_value_pct:.0f}% > {MAX_CREDIBLE_VALUE_PCT}% "
                       f"— likely bad odds data")
            continue

        # Calculate value
        value = calculate_value(win_prob, decimal_odds)

        # Get stake recommendation
        stake_rec = get_stake_recommendation(pred, bankroll, decimal_odds)

        # Skip if below minimum confidence
        conf_rank = get_confidence_rank(confidence)
        min_rank = get_confidence_rank(MIN_CONFIDENCE_SINGLE)

        rec = {
            "type": "single",
            "match": match,
            "date": pred.get("date", "Unknown"),
            "prediction": outcome,
            "confidence": confidence,
            "confidence_rank": conf_rank,
            "our_probability": round(win_prob, 3),
            "odds": decimal_odds,
            "odds_source": "actual" if match_odds else "estimated",
            "implied_probability": round(1 / decimal_odds, 3),
            "value": round(value, 3),
            "value_pct": round(value * 100, 1),
            "recommended_stake": stake_rec.get("recommended_stake", 0),
            "stake_pct": stake_rec.get("stake_pct", 0),
            "potential_profit": round(stake_rec.get("recommended_stake", 0) * (decimal_odds - 1), 2),
            "factors": pred.get("home_factors", []) + pred.get("away_factors", []),
            "referee": pred.get("referee", None),
            "referee_bias": pred.get("referee_bias", None),
        }

        # Check odds movement for sharp money signals
        try:
            from scripts.data.odds_tracker import get_confidence_adjustment as get_movement_adj
            movement_adj = get_movement_adj(match, outcome)
            if movement_adj < 1.0:
                rec["odds_movement_warning"] = True
                rec["movement_adjustment"] = movement_adj
        except ImportError:
            movement_adj = 1.0

        # Determine if this is a recommended bet
        if conf_rank >= min_rank:
            if value > MIN_VALUE_SINGLE:
                if movement_adj < 0.6:
                    rec["recommendation"] = "CONSIDER"
                    rec["reason"] = f"VALUE: {value:.1%} edge but SHARP MONEY AGAINST"
                else:
                    rec["recommendation"] = "BET"
                    rec["reason"] = f"VALUE: {value:.1%} edge"
            elif value > 0:
                rec["recommendation"] = "CONSIDER"
                rec["reason"] = f"Small edge: {value:.1%}"
            else:
                rec["recommendation"] = "SKIP"
                rec["reason"] = f"No value at these odds"
                rec["recommended_stake"] = 0
        else:
            rec["recommendation"] = "SKIP"
            rec["reason"] = f"Low confidence: {confidence}"
            rec["recommended_stake"] = 0

        recommendations.append(rec)

    # Sort by value (best bets first)
    recommendations.sort(key=lambda x: (x["value"], x["confidence_rank"]), reverse=True)

    return recommendations


def analyze_parlays(predictions: List[Dict], odds: Dict, bankroll: float) -> List[Dict]:
    """Analyze predictions for parlay opportunities.

    Only combines high-confidence bets for parlays.

    Args:
        predictions: List of predictions
        odds: Dict of odds by match
        bankroll: Current bankroll

    Returns:
        List of parlay recommendations
    """
    # Filter for high-confidence predictions
    min_rank = get_confidence_rank(MIN_CONFIDENCE_PARLAY)
    def _get_conf_str(p):
        c = p.get("confidence_level", p.get("confidence", "LOW"))
        return c if isinstance(c, str) else ("HIGH" if c >= 0.55 else "MEDIUM-HIGH" if c >= 0.50 else "MEDIUM")
    high_conf = [p for p in predictions if get_confidence_rank(_get_conf_str(p)) >= min_rank]

    if len(high_conf) < MIN_PARLAY_LEGS:
        return []

    parlays = []
    parlay_stake = bankroll * PARLAY_STAKE_PCT

    # Generate 2-leg and 3-leg parlays
    for n_legs in range(MIN_PARLAY_LEGS, min(MAX_PARLAY_LEGS + 1, len(high_conf) + 1)):
        for combo in combinations(high_conf, n_legs):
            legs = []
            combined_odds = 1.0
            combined_prob = 1.0

            for pred in combo:
                match = pred.get("match", "Unknown")
                outcome = pred.get("predicted_outcome", "HOME")
                probs = pred.get("probabilities", {})

                if outcome == "HOME":
                    win_prob = probs.get("home", 0.5)
                    odds_key = "home"
                elif outcome == "AWAY":
                    win_prob = probs.get("away", 0.3)
                    odds_key = "away"
                else:
                    win_prob = probs.get("draw", 0.25)
                    odds_key = "draw"

                match_odds = odds.get(match, {})
                decimal_odds = match_odds.get(odds_key, estimate_odds(win_prob))

                legs.append({
                    "match": match,
                    "prediction": outcome,
                    "odds": decimal_odds,
                    "probability": win_prob,
                })

                combined_odds *= decimal_odds
                combined_prob *= win_prob

            # Calculate parlay value
            parlay_value = (combined_prob * combined_odds) - 1

            parlay = {
                "type": "parlay",
                "n_legs": n_legs,
                "legs": legs,
                "combined_odds": round(combined_odds, 2),
                "combined_probability": round(combined_prob, 3),
                "value": round(parlay_value, 3),
                "value_pct": round(parlay_value * 100, 1),
                "recommended_stake": round(parlay_stake, 2) if parlay_value > MIN_VALUE_PARLAY else 0,
                "potential_profit": round(parlay_stake * (combined_odds - 1), 2),
            }

            if parlay_value > MIN_VALUE_PARLAY:
                parlay["recommendation"] = "CONSIDER"
                parlay["reason"] = f"VALUE: {parlay_value:.1%} edge"
            else:
                parlay["recommendation"] = "SKIP"
                parlay["reason"] = f"Insufficient edge: {parlay_value:.1%}"

            parlays.append(parlay)

    # Sort by value
    parlays.sort(key=lambda x: x["value"], reverse=True)

    return parlays[:5]  # Return top 5 parlays


def generate_betting_slip(include_parlays: bool = True) -> Dict:
    """Generate complete betting slip with recommendations.

    Returns:
        Dict with all bet recommendations
    """
    log.info("=" * 60)
    log.info("BETTING DECISION ENGINE")
    log.info("=" * 60)

    # Load data
    predictions = load_predictions()
    if not predictions:
        return {"error": "No predictions available"}

    odds = load_odds()
    bankroll = load_bankroll()
    current_balance = bankroll.get("current_balance", 1000)

    log.info(f"Analyzing {len(predictions)} predictions")
    log.info(f"Current bankroll: ${current_balance:.2f}")

    # Analyze singles
    singles = analyze_single_bets(predictions, odds, current_balance)
    recommended_singles = [s for s in singles if s["recommendation"] == "BET"]
    consider_singles = [s for s in singles if s["recommendation"] == "CONSIDER"]

    # Analyze parlays
    parlays = []
    if include_parlays:
        parlays = analyze_parlays(predictions, odds, current_balance)
        parlays = [p for p in parlays if p["recommendation"] == "CONSIDER"]

    # Calculate totals
    total_stake = sum(s["recommended_stake"] for s in recommended_singles)
    total_potential = sum(s["potential_profit"] for s in recommended_singles)

    result = {
        "generated_at": datetime.now().isoformat(),
        "bankroll": current_balance,
        "summary": {
            "total_predictions": len(predictions),
            "recommended_bets": len(recommended_singles),
            "consider_bets": len(consider_singles),
            "parlays": len(parlays),
            "total_stake": round(total_stake, 2),
            "total_potential_profit": round(total_potential, 2),
            "stake_pct_of_bankroll": round(total_stake / current_balance * 100, 1) if current_balance > 0 else 0,
        },
        "recommended_singles": recommended_singles,
        "consider_singles": consider_singles,
        "parlays": parlays,
        "all_analysis": singles,
    }

    # Save betting slip
    output_path = DATA_DIR / "betting" / "betting_slip.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    log.info(f"Saved betting slip to {output_path}")

    return result


def print_betting_slip(slip: Dict):
    """Print betting slip in readable format."""
    print("\n" + "=" * 80)
    print(" BETTING SLIP")
    print("=" * 80)

    summary = slip.get("summary", {})
    print(f"\n  Bankroll: ${slip.get('bankroll', 0):.2f}")
    print(f"  Predictions analyzed: {summary.get('total_predictions', 0)}")

    # Recommended bets
    recommended = slip.get("recommended_singles", [])
    if recommended:
        print("\n" + "-" * 80)
        print(" RECOMMENDED BETS (Strong Value)")
        print("-" * 80)

        for bet in recommended:
            print(f"\n  {bet['match']} ({bet['date']})")
            print(f"     Prediction: {bet['prediction']} @ {bet['odds']}")
            print(f"     Our Prob: {bet['our_probability']:.1%} vs Implied: {bet['implied_probability']:.1%}")
            print(f"     VALUE: {bet['value_pct']}%")
            print(f"     Stake: ${bet['recommended_stake']:.2f} ({bet['stake_pct']}% of bankroll)")
            print(f"     Potential Profit: ${bet['potential_profit']:.2f}")
            if bet.get("referee"):
                print(f"     Referee: {bet['referee']} ({bet.get('referee_bias', 'unknown')})")
            if bet.get("factors"):
                print(f"     Factors: {', '.join(bet['factors'])}")

        total_stake = sum(b["recommended_stake"] for b in recommended)
        total_profit = sum(b["potential_profit"] for b in recommended)
        print(f"\n  Total Stake: ${total_stake:.2f} | Total Potential: ${total_profit:.2f}")
    else:
        print("\n  No high-value single bets recommended")

    # Consider bets
    consider = slip.get("consider_singles", [])[:3]  # Top 3
    if consider:
        print("\n" + "-" * 80)
        print(" CONSIDER (Smaller Edge)")
        print("-" * 80)

        for bet in consider:
            print(f"  {bet['match']}: {bet['prediction']} @ {bet['odds']} | Value: {bet['value_pct']}%")

    # Parlays
    parlays = slip.get("parlays", [])[:2]  # Top 2
    if parlays:
        print("\n" + "-" * 80)
        print(" PARLAY SUGGESTIONS")
        print("-" * 80)

        for i, parlay in enumerate(parlays, 1):
            print(f"\n  Parlay #{i} ({parlay['n_legs']} legs) @ {parlay['combined_odds']:.2f}")
            for leg in parlay["legs"]:
                print(f"     {leg['match']}: {leg['prediction']} @ {leg['odds']}")
            print(f"     Combined Prob: {parlay['combined_probability']:.1%} | Value: {parlay['value_pct']}%")
            if parlay["recommended_stake"] > 0:
                print(f"     Stake: ${parlay['recommended_stake']:.2f} | Potential: ${parlay['potential_profit']:.2f}")

    # Skip bets
    skip = [b for b in slip.get("all_analysis", []) if b["recommendation"] == "SKIP"]
    if skip:
        print("\n" + "-" * 80)
        print(f" SKIP ({len(skip)} bets)")
        print("-" * 80)
        for bet in skip[:3]:
            print(f"  {bet['match']}: {bet['reason']}")

    print("\n" + "=" * 80)
    print(f" Generated: {slip.get('generated_at', 'Unknown')}")
    print("=" * 80)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Betting Decision Engine")
    parser.add_argument("--odds-file", help="Path to odds JSON file")
    parser.add_argument("--parlay", action="store_true", help="Include parlay suggestions")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    args = parser.parse_args()

    slip = generate_betting_slip(include_parlays=args.parlay)

    if "error" in slip:
        print(f"Error: {slip['error']}")
        return 1

    if args.json:
        print(json.dumps(slip, indent=2))
    else:
        print_betting_slip(slip)

    return 0


if __name__ == "__main__":
    sys.exit(main())
