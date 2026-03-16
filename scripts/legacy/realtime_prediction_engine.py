#!/usr/bin/env python3
"""REAL-TIME PREDICTION ENGINE - The Actual Prediction System

This is the CORE prediction engine that generates predictions for FUTURE matches.
It combines all validated factors into actionable predictions.

Components:
1. Load upcoming matches (manual or scraped)
2. Calculate current form for each team
3. Fetch weather forecasts
4. Apply validated factor stacking
5. Generate confidence-weighted predictions
6. Output actionable betting recommendations

Based on 21-season validation (7,829 matches, 2005-2026):
- 3+ factors: 64.7% home win (1350 matches)
- 4+ factors: 72.7% home win (264 matches)
- 5+ factors: 95.8% home win (24 matches)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR

# Import our components
try:
    from scripts.prediction.weather_integration import fetch_all_match_weather, get_weather_impact
    from scripts.prediction.current_form_calculator import calculate_all_forms
    from scripts.prediction.referee_integration import analyze_referee_impact, classify_referee, get_perfect_storm_matches
except ImportError:
    # Fallback for direct execution
    from weather_integration import fetch_all_match_weather, get_weather_impact
    from current_form_calculator import calculate_all_forms
    from referee_integration import analyze_referee_impact, classify_referee, get_perfect_storm_matches

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# VALIDATED FACTOR LIFTS (21 Seasons)
# =============================================================================

FACTOR_LIFTS = {
    # HIGH confidence (validated across 95%+ seasons)
    "big_stadium": {"home_win": 0.15, "confidence": "HIGH", "seasons": 21},
    "home_favorite": {"home_win": 0.12, "confidence": "HIGH", "seasons": 21},
    "big_home_favorite": {"home_win": 0.20, "confidence": "HIGH", "seasons": 21},
    "hot_home": {"home_win": 0.08, "confidence": "HIGH", "seasons": 21},
    "cold_away": {"home_win": 0.06, "confidence": "HIGH", "seasons": 21},
    "derby": {"draw": 0.08, "cards": 0.10, "confidence": "HIGH", "seasons": 21},

    # MEDIUM confidence (8 seasons of data)
    "home_fav_ref": {"home_win": 0.12, "confidence": "MEDIUM", "seasons": 8},
    "away_fav_ref": {"home_win": -0.10, "confidence": "MEDIUM", "seasons": 8},
    "strict_ref": {"high_cards": 0.15, "confidence": "MEDIUM", "seasons": 8},

    # Weather (8 seasons)
    "rain": {"goals": 0.20, "home_win": 0.032, "confidence": "MEDIUM", "seasons": 8},
    "wind": {"goals": -0.22, "confidence": "MEDIUM", "seasons": 8},
    "cold": {"cards": 0.05, "confidence": "MEDIUM", "seasons": 8},

    # Away factors (for upset detection)
    "away_favorite": {"away_win": 0.10, "confidence": "HIGH", "seasons": 21},
    "big_away_favorite": {"away_win": 0.18, "confidence": "HIGH", "seasons": 21},
    "hot_away": {"away_win": 0.06, "confidence": "HIGH", "seasons": 21},
    "cold_home": {"away_win": 0.05, "confidence": "HIGH", "seasons": 21},
}

# Progressive stacking bonuses
STACKING_BONUSES = {
    1: 0.044,   # 1+ factors: +4.4%
    2: 0.106,   # 2+ factors: +10.6%
    3: 0.204,   # 3+ factors: +20.4%
    4: 0.284,   # 4+ factors: +28.4%
    5: 0.515,   # 5+ factors: +51.5%
}

# Base rates
BASE_RATES = {
    "home_win": 0.444,
    "draw": 0.265,
    "away_win": 0.291,
    "over_25_goals": 0.486,
    "btts": 0.459,
    "over_45_cards": 0.473,
}

# Referee classifications (from our 8-season analysis)
HOME_FAV_REFS = ["Federico Dionisi", "Antonio Giua", "Simone Sozza", "Valerio Marini"]
AWAY_FAV_REFS = ["Paolo Mazzoleni", "Daniele Doveri", "Marco Di Bello", "Pairetto"]
STRICT_REFS = ["Fabio Maresca", "Daniele Orsato", "Gianluca Manganiello", "Michael Fabbri"]


def _load_factor_multipliers() -> Dict[str, float]:
    """Load factor lift multipliers from feedback loop.

    Returns dict mapping factor name to multiplier (0.3-1.2).
    Missing factors default to 1.0 (no change).
    """
    adj_path = DATA_DIR / "feedback" / "factor_adjustments.json"
    if not adj_path.exists():
        return {}
    try:
        with open(adj_path) as f:
            data = json.load(f)
        multipliers = data.get("multipliers", {})
        if multipliers:
            log.info("Loaded factor multipliers for %d factors", len(multipliers))
        return multipliers
    except Exception as e:
        log.debug("Could not load factor multipliers: %s", e)
        return {}


# Load factor multipliers at module level (refreshed each pipeline run)
_FACTOR_MULTIPLIERS = _load_factor_multipliers()


def load_upcoming_matches() -> List[Dict]:
    """Load upcoming matches from all available sources."""
    matches = []

    # Try manual matches first
    manual_path = DATA_DIR / "upcoming" / "manual_matches.json"
    if manual_path.exists():
        with open(manual_path) as f:
            data = json.load(f)
            matches.extend(data.get("matches", []))
            log.info(f"Loaded {len(matches)} matches from manual file")

    # Try scraped matches
    scraped_path = DATA_DIR / "upcoming" / "matches.json"
    if scraped_path.exists():
        with open(scraped_path) as f:
            data = json.load(f)
            # Deduplicate
            existing = {f"{m['home_team']}_{m['away_team']}" for m in matches}
            for m in data.get("matches", []):
                key = f"{m['home_team']}_{m['away_team']}"
                if key not in existing:
                    matches.append(m)

    return matches


def identify_all_factors(match: Dict, form_data: Dict, weather_data: Dict, referee_data: Dict = None) -> Dict:
    """Identify all applicable factors for a match.

    Returns dict with:
    - home_factors: List of factors favoring home team
    - away_factors: List of factors favoring away team
    - neutral_factors: Derby, weather, etc.
    - total_home_lift: Combined probability lift for home win
    """
    home = match["home_team"]
    away = match["away_team"]
    key = f"{home} vs {away}"

    # Get matchup data
    matchup = form_data.get("matchups", {}).get(key, {})
    factors_from_form = matchup.get("factors", [])

    # Get weather data
    weather = weather_data.get(key, {})
    weather_impact = weather.get("impact", {})

    # Categorize factors
    home_factors = []
    away_factors = []
    neutral_factors = []

    # From form analysis
    for f in factors_from_form:
        if f in ["big_stadium", "home_favorite", "big_home_favorite", "hot_home", "cold_away"]:
            home_factors.append(f)
        elif f in ["away_favorite", "big_away_favorite", "hot_away", "cold_home"]:
            away_factors.append(f)
        elif f == "derby":
            neutral_factors.append(f)

    # Weather factors
    weather_factors = weather_impact.get("factors", [])
    for f in weather_factors:
        neutral_factors.append(f)

    # Referee factors (from referee_integration)
    if referee_data:
        ref_info = referee_data.get(key, {})
        ref_factors = ref_info.get("factors", [])
        referee = ref_info.get("referee", "")

        for f in ref_factors:
            if f == "home_fav_ref":
                home_factors.append("home_fav_ref")
            elif f == "away_fav_ref":
                away_factors.append("away_fav_ref")
            elif f == "strict_ref":
                neutral_factors.append("strict_ref")
            elif f == "lenient_ref":
                neutral_factors.append("lenient_ref")
    else:
        # Fallback to simple check
        referee = match.get("referee", "")
        if referee in HOME_FAV_REFS:
            home_factors.append("home_fav_ref")
        elif referee in AWAY_FAV_REFS:
            away_factors.append("away_fav_ref")
        if referee in STRICT_REFS:
            neutral_factors.append("strict_ref")

    # Calculate lifts (apply feedback multipliers when available)
    home_lift = sum(
        FACTOR_LIFTS.get(f, {}).get("home_win", 0) * _FACTOR_MULTIPLIERS.get(f, 1.0)
        for f in home_factors
    )
    away_lift = sum(
        FACTOR_LIFTS.get(f, {}).get("away_win", 0) * _FACTOR_MULTIPLIERS.get(f, 1.0)
        for f in away_factors
    )

    # Add stacking bonus (cumulative total for the highest matching tier)
    n_home = len(home_factors)
    n_away = len(away_factors)

    if n_home >= 5:
        home_lift += STACKING_BONUSES[5]
    elif n_home >= 4:
        home_lift += STACKING_BONUSES[4]
    elif n_home >= 3:
        home_lift += STACKING_BONUSES[3]
    elif n_home >= 2:
        home_lift += STACKING_BONUSES[2]
    elif n_home >= 1:
        home_lift += STACKING_BONUSES[1]

    if n_away >= 4:
        away_lift += STACKING_BONUSES[4]
    elif n_away >= 3:
        away_lift += STACKING_BONUSES[3]
    elif n_away >= 2:
        away_lift += STACKING_BONUSES[2]
    elif n_away >= 1:
        away_lift += STACKING_BONUSES[1]

    return {
        "home_factors": home_factors,
        "away_factors": away_factors,
        "neutral_factors": neutral_factors,
        "n_home_factors": n_home,
        "n_away_factors": n_away,
        "home_lift": home_lift,
        "away_lift": away_lift,
        "weather": weather_impact,
    }


def generate_prediction(match: Dict, factors: Dict, form_data: Dict,
                        market_probs: Dict = None) -> Dict:
    """Generate a complete prediction for a match.

    When market_probs are available, anchors on market-implied probabilities
    and applies dampened factor lifts (matching ensemble behavior).
    Falls back to BASE_RATES only when no market data is available.
    """
    home = match["home_team"]
    away = match["away_team"]
    key = f"{home} vs {away}"

    DAMPEN = 0.35  # Same as ensemble factor predictor

    if market_probs:
        home_prob = market_probs.get("prob_H", BASE_RATES["home_win"])
        draw_prob = market_probs.get("prob_D", BASE_RATES["draw"])
        away_prob = market_probs.get("prob_A", BASE_RATES["away_win"])
        # Apply dampened lifts on top of market anchor
        home_prob += factors["home_lift"] * DAMPEN
        away_prob += factors["away_lift"] * DAMPEN
    else:
        home_prob = BASE_RATES["home_win"]
        draw_prob = BASE_RATES["draw"]
        away_prob = BASE_RATES["away_win"]
        home_prob += factors["home_lift"]
        away_prob += factors["away_lift"]

    # Derby adjustment (dampened when market-anchored)
    d = DAMPEN if market_probs else 1.0
    if "derby" in factors["neutral_factors"]:
        draw_prob += 0.08 * d
        home_prob -= 0.04 * d
        away_prob -= 0.04 * d

    # Clamp and normalize
    home_prob = max(0.03, home_prob)
    draw_prob = max(0.03, draw_prob)
    away_prob = max(0.03, away_prob)
    total = home_prob + draw_prob + away_prob
    home_prob /= total
    draw_prob /= total
    away_prob /= total

    # Determine prediction (pure max-probability, no bias against draws)
    if home_prob >= draw_prob and home_prob >= away_prob:
        predicted_outcome = "HOME"
    elif away_prob >= draw_prob:
        predicted_outcome = "AWAY"
    else:
        predicted_outcome = "DRAW"

    # Confidence level based on factors
    n_factors = factors["n_home_factors"] + factors["n_away_factors"]
    if n_factors >= 5:
        confidence = "VERY HIGH"
        expected_accuracy = 0.958
    elif n_factors >= 4:
        confidence = "HIGH"
        expected_accuracy = 0.727
    elif n_factors >= 3:
        confidence = "MEDIUM-HIGH"
        expected_accuracy = 0.647
    elif n_factors >= 2:
        confidence = "MEDIUM"
        expected_accuracy = 0.55
    else:
        confidence = "LOW"
        expected_accuracy = 0.488

    # Get form data
    matchup = form_data.get("matchups", {}).get(key, {})
    home_form = matchup.get("home_form", {})
    away_form = matchup.get("away_form", {})

    # Goals prediction
    expected_goals = 2.67  # Base
    expected_goals += factors.get("weather", {}).get("goals_adj", 0)

    # Cards prediction
    expected_cards = 4.45  # Base
    if "strict_ref" in factors["neutral_factors"]:
        expected_cards += 1.0
    if "derby" in factors["neutral_factors"]:
        expected_cards += 0.5
    if "cold" in factors["neutral_factors"]:
        expected_cards += 0.2

    return {
        "match": f"{home} vs {away}",
        "date": match.get("date", "TBD"),
        "time": match.get("time", "TBD"),
        "venue": match.get("venue", f"{home} Stadium"),

        # Main prediction
        "predicted_outcome": predicted_outcome,
        "probabilities": {
            "home": round(home_prob, 3),
            "draw": round(draw_prob, 3),
            "away": round(away_prob, 3),
        },

        # Confidence
        "confidence": confidence,
        "expected_accuracy": expected_accuracy,
        "n_factors": n_factors,

        # Factors
        "home_factors": factors["home_factors"],
        "away_factors": factors["away_factors"],
        "neutral_factors": factors["neutral_factors"],

        # Form summary
        "home_form": {
            "ppg": home_form.get("ppg", "?"),
            "status": home_form.get("form_status", "?"),
            "elo": home_form.get("elo", "?"),
        },
        "away_form": {
            "ppg": away_form.get("ppg", "?"),
            "status": away_form.get("form_status", "?"),
            "elo": away_form.get("elo", "?"),
        },

        # Other markets
        "expected_goals": round(expected_goals, 2),
        "over_25": expected_goals > 2.5,
        "expected_cards": round(expected_cards, 1),

        # Betting value
        "betting_recommendation": get_betting_recommendation(
            predicted_outcome, home_prob, draw_prob, away_prob, confidence
        ),
    }


def get_betting_recommendation(outcome: str, h_prob: float, d_prob: float, a_prob: float, confidence: str) -> str:
    """Generate betting recommendation based on prediction."""
    if confidence in ["VERY HIGH", "HIGH"]:
        if outcome == "HOME" and h_prob > 0.65:
            return "STRONG BET: Home Win"
        elif outcome == "AWAY" and a_prob > 0.55:
            return "STRONG BET: Away Win"
        elif outcome == "HOME" and h_prob > 0.55:
            return "BET: Home Win or Home/Draw Double Chance"
        elif outcome == "AWAY" and a_prob > 0.45:
            return "BET: Away Win or Draw/Away Double Chance"
        else:
            return "CONSIDER: Double Chance"
    elif confidence == "MEDIUM-HIGH":
        if outcome == "HOME":
            return "LEAN: Home Win or Double Chance"
        elif outcome == "AWAY":
            return "LEAN: Away Win or Double Chance"
        else:
            return "LEAN: Draw or Under 2.5"
    else:
        return "SKIP: Low confidence, no bet recommended"


def run_predictions() -> Dict:
    """Run the full prediction pipeline."""
    log.info("=" * 70)
    log.info("REAL-TIME PREDICTION ENGINE")
    log.info("=" * 70)

    # Step 1: Load upcoming matches
    log.info("\n[1/4] Loading upcoming matches...")
    matches = load_upcoming_matches()
    if not matches:
        log.error("No upcoming matches found!")
        return {}
    log.info(f"Found {len(matches)} upcoming matches")

    # Step 2: Calculate current form
    log.info("\n[2/4] Calculating current team form...")
    form_data = calculate_all_forms()

    # Step 3: Fetch weather
    log.info("\n[3/5] Fetching weather forecasts...")
    weather_data = fetch_all_match_weather()

    # Step 4: Analyze referees
    log.info("\n[4/5] Analyzing referee assignments...")
    referee_data = analyze_referee_impact(matches)

    # Check for perfect storm matches
    perfect_storms = get_perfect_storm_matches(matches, form_data, referee_data)
    if perfect_storms:
        log.info(f"🔥 Found {len(perfect_storms)} PERFECT STORM matches!")

    # Step 5: Generate predictions
    log.info("\n[5/5] Generating predictions...")
    predictions = []

    for match in matches:
        # Identify factors (now with referee data)
        factors = identify_all_factors(match, form_data, weather_data, referee_data)

        # Generate prediction
        pred = generate_prediction(match, factors, form_data)

        # Add referee info to prediction
        key = f"{match['home_team']} vs {match['away_team']}"
        ref_info = referee_data.get(key, {})
        if ref_info.get("referee"):
            pred["referee"] = ref_info["referee"]
            pred["referee_bias"] = ref_info.get("classification", {}).get("bias_type", "unknown")

        predictions.append(pred)

    # Sort by confidence
    confidence_order = {"VERY HIGH": 0, "HIGH": 1, "MEDIUM-HIGH": 2, "MEDIUM": 3, "LOW": 4}
    predictions.sort(key=lambda x: (confidence_order.get(x["confidence"], 5), -x["probabilities"]["home"]))

    # Save predictions
    output = {
        "generated_at": datetime.now().isoformat(),
        "model_version": "v2.0-21seasons",
        "predictions": predictions,
        "summary": {
            "total_matches": len(predictions),
            "high_confidence": len([p for p in predictions if p["confidence"] in ["VERY HIGH", "HIGH"]]),
            "medium_confidence": len([p for p in predictions if p["confidence"] == "MEDIUM-HIGH"]),
        }
    }

    output_path = DATA_DIR / "upcoming" / "predictions.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    log.info(f"\nSaved predictions to {output_path}")
    return output


def print_predictions(output: Dict):
    """Print predictions in a readable format."""
    print("\n" + "=" * 80)
    print("🏆 SERIE A PREDICTIONS - MATCHWEEK")
    print("=" * 80)
    print(f"Generated: {output.get('generated_at', 'Unknown')}")
    print(f"Model: {output.get('model_version', 'Unknown')}")

    predictions = output.get("predictions", [])

    # High confidence first
    high_conf = [p for p in predictions if p["confidence"] in ["VERY HIGH", "HIGH"]]
    medium_conf = [p for p in predictions if p["confidence"] == "MEDIUM-HIGH"]
    low_conf = [p for p in predictions if p["confidence"] in ["MEDIUM", "LOW"]]

    if high_conf:
        print("\n" + "=" * 80)
        print("🔥 HIGH CONFIDENCE PREDICTIONS")
        print("=" * 80)
        for pred in high_conf:
            print_single_prediction(pred)

    if medium_conf:
        print("\n" + "=" * 80)
        print("📊 MEDIUM-HIGH CONFIDENCE")
        print("=" * 80)
        for pred in medium_conf:
            print_single_prediction(pred)

    if low_conf:
        print("\n" + "=" * 80)
        print("⚠️ LOWER CONFIDENCE (Skip or small stakes)")
        print("=" * 80)
        for pred in low_conf:
            print_single_prediction(pred, brief=True)

    # Summary
    summary = output.get("summary", {})
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total matches: {summary.get('total_matches', 0)}")
    print(f"High confidence: {summary.get('high_confidence', 0)}")
    print(f"Medium confidence: {summary.get('medium_confidence', 0)}")


def print_single_prediction(pred: Dict, brief: bool = False):
    """Print a single prediction."""
    print(f"\n{'─' * 60}")
    print(f"📍 {pred['match']}")
    venue_info = f"{pred['date']} at {pred['time']} | {pred['venue']}"
    if pred.get("referee"):
        venue_info += f" | Ref: {pred['referee']}"
    print(f"   {venue_info}")

    probs = pred["probabilities"]
    print(f"\n   PREDICTION: {pred['predicted_outcome']} ({pred['confidence']})")
    print(f"   Probabilities: H {probs['home']:.1%} | D {probs['draw']:.1%} | A {probs['away']:.1%}")

    if not brief:
        # Form
        hf = pred.get("home_form", {})
        af = pred.get("away_form", {})
        print(f"\n   Form: {pred['match'].split(' vs ')[0]} ({hf.get('ppg', '?')} PPG, {hf.get('status', '?').upper()}) vs "
              f"{pred['match'].split(' vs ')[1]} ({af.get('ppg', '?')} PPG, {af.get('status', '?').upper()})")

        # Factors
        if pred.get("home_factors"):
            print(f"   ✅ Home factors: {', '.join(pred['home_factors'])}")
        if pred.get("away_factors"):
            print(f"   ❌ Away factors: {', '.join(pred['away_factors'])}")
        if pred.get("neutral_factors"):
            print(f"   ⚖️ Neutral: {', '.join(pred['neutral_factors'])}")

        # Other markets
        print(f"\n   Expected goals: {pred.get('expected_goals', '?')} | Cards: {pred.get('expected_cards', '?')}")

    # Recommendation
    rec = pred.get("betting_recommendation", "")
    if "STRONG" in rec:
        print(f"\n   💰 {rec}")
    elif "BET" in rec or "LEAN" in rec:
        print(f"\n   📈 {rec}")
    else:
        print(f"\n   ⏸️ {rec}")


if __name__ == "__main__":
    output = run_predictions()
    if output:
        print_predictions(output)
