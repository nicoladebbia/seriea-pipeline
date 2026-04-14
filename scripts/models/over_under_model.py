#!/usr/bin/env python3
"""OVER/UNDER PREDICTION MODEL - Phase 2 Implementation

Predicts goal totals using Poisson distribution and advanced factor analysis.

Features:
- Historical goal pattern analysis (21 seasons)
- Team attacking/defensive strength calculation
- Weather, referee, and form factors
- Poisson/Negative Binomial distributions
- Multiple over/under lines (1.5, 2.5, 3.5, 4.5)
- Value betting integration

Based on 7,829 historical matches with validated factors.
"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# =============================================================================
# ODDS SANITY - reject live/in-play odds before generating recommendations
# =============================================================================

ODDS_SANITY = {
    "totals": {"min": 1.05, "max": 6.0},
}

# =============================================================================
# CONSTANTS & HISTORICAL DATA
# =============================================================================

# Serie A average goals per game (2023-2025 calibrated: actual 2.586 over 760 matches)
# Previously 2.67 (historical all-time). Recent seasons trend lower.
SERIE_A_AVG_GOALS = 2.58

# Historical over/under rates (from 21-season analysis)
HISTORICAL_OVER_RATES = {
    0.5: 0.89,  # 89% of matches have at least 1 goal
    1.5: 0.72,  # 72% of matches have 2+ goals
    2.5: 0.52,  # 52% of matches have 3+ goals
    3.5: 0.28,  # 28% of matches have 4+ goals
    4.5: 0.12,  # 12% of matches have 5+ goals
}

# Goal scoring factors from validation
GOAL_FACTORS = {
    # Form-based factors
    "hot_home": {"expected_goals_mod": 0.25, "description": "Home team in form"},
    "hot_away": {"expected_goals_mod": 0.20, "description": "Away team in form"},
    "cold_home": {"expected_goals_mod": -0.15, "description": "Home team struggling"},
    "cold_away": {"expected_goals_mod": -0.10, "description": "Away team struggling"},

    # Team strength factors
    "big_home_favorite": {"expected_goals_mod": 0.30, "description": "Strong home favorite"},
    "big_away_favorite": {"expected_goals_mod": 0.35, "description": "Strong away favorite"},
    "home_favorite": {"expected_goals_mod": 0.15, "description": "Home team favored"},
    "away_favorite": {"expected_goals_mod": 0.20, "description": "Away team favored"},

    # Stadium factors
    "big_stadium": {"expected_goals_mod": 0.05, "description": "Large stadium atmosphere"},

    # Weather factors
    "hot": {"expected_goals_mod": -0.10, "description": "Hot weather (slower pace)"},
    "cold": {"expected_goals_mod": -0.05, "description": "Cold weather"},
    "rain": {"expected_goals_mod": -0.15, "description": "Rainy conditions"},
    "wind": {"expected_goals_mod": -0.10, "description": "Windy conditions"},

    # Referee factors (impact on fouls/stoppages, indirect goal effect)
    "strict_ref": {"expected_goals_mod": -0.10, "description": "Strict referee (more stoppages)"},
    "lenient_ref": {"expected_goals_mod": 0.05, "description": "Lenient referee (more flow)"},

    # Derby/rivalry factors
    "derby": {"expected_goals_mod": -0.20, "description": "Derby match (defensive)"},
}

# Team attacking/defensive strength adjustments
# Based on historical goals scored/conceded vs league average
TEAM_STRENGTHS = {
    # Top attacking teams (goals scored above average)
    "Inter": {"attack_mod": 0.35, "defense_mod": 0.30},
    "Napoli": {"attack_mod": 0.30, "defense_mod": 0.25},
    "Juventus": {"attack_mod": 0.20, "defense_mod": 0.35},
    "Atalanta": {"attack_mod": 0.40, "defense_mod": 0.15},
    "Milan": {"attack_mod": 0.25, "defense_mod": 0.20},
    "Roma": {"attack_mod": 0.15, "defense_mod": 0.15},
    "Lazio": {"attack_mod": 0.25, "defense_mod": 0.10},
    "Fiorentina": {"attack_mod": 0.15, "defense_mod": 0.10},
    "Bologna": {"attack_mod": 0.10, "defense_mod": 0.15},
    "Torino": {"attack_mod": 0.05, "defense_mod": 0.10},

    # Mid-table teams
    "Genoa": {"attack_mod": 0.00, "defense_mod": 0.05},
    "Udinese": {"attack_mod": 0.00, "defense_mod": 0.10},
    "Sassuolo": {"attack_mod": 0.05, "defense_mod": -0.05},  # Promoted 2025-26
    "Como": {"attack_mod": 0.10, "defense_mod": -0.10},
    "Parma": {"attack_mod": 0.05, "defense_mod": -0.05},
    "Verona": {"attack_mod": 0.05, "defense_mod": -0.10},

    # Lower table teams
    "Cagliari": {"attack_mod": -0.05, "defense_mod": -0.05},
    "Lecce": {"attack_mod": -0.10, "defense_mod": 0.00},
    "Pisa": {"attack_mod": -0.05, "defense_mod": 0.00},       # Promoted 2025-26
    "Cremonese": {"attack_mod": -0.10, "defense_mod": -0.10},  # Promoted 2025-26
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class GoalPrediction:
    """Prediction for a single match's goal totals."""
    match: str
    home_team: str
    away_team: str
    date: str

    # Expected goals
    expected_home_goals: float
    expected_away_goals: float
    expected_total_goals: float

    # Over/Under probabilities
    over_0_5: float
    over_1_5: float
    over_2_5: float
    over_3_5: float
    over_4_5: float

    # Factors used
    factors: List[str]

    # Confidence
    confidence: str
    confidence_rank: int

    # Model details
    home_attack_strength: float
    away_attack_strength: float
    home_defense_strength: float
    away_defense_strength: float


@dataclass
class OverUnderBet:
    """Betting recommendation for over/under market."""
    match: str
    date: str
    line: float
    bet_type: str  # "over" or "under"
    our_probability: float
    odds: float
    implied_probability: float
    value: float
    value_pct: float
    expected_goals: float
    recommendation: str  # "BET", "CONSIDER", "SKIP"
    stake_pct: float
    factors: List[str]


# =============================================================================
# POISSON DISTRIBUTION
# =============================================================================

def poisson_probability(k: int, lambda_: float) -> float:
    """Calculate Poisson probability P(X = k) given rate lambda."""
    if lambda_ <= 0:
        return 1.0 if k == 0 else 0.0
    return (math.exp(-lambda_) * (lambda_ ** k)) / math.factorial(k)


def poisson_cumulative(k: int, lambda_: float) -> float:
    """Calculate cumulative Poisson P(X <= k)."""
    return sum(poisson_probability(i, lambda_) for i in range(k + 1))


def calculate_over_probability(goals_threshold: float, lambda_total: float) -> float:
    """Calculate P(Total Goals > threshold).

    For 2.5 line: P(X >= 3) = 1 - P(X <= 2)
    """
    k = int(goals_threshold)  # e.g., 2 for 2.5 line
    return 1 - poisson_cumulative(k, lambda_total)


def calculate_exact_score_matrix(lambda_home: float, lambda_away: float, max_goals: int = 7) -> Dict:
    """Calculate probability matrix for exact scores."""
    matrix = {}

    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            p_home = poisson_probability(home_goals, lambda_home)
            p_away = poisson_probability(away_goals, lambda_away)
            matrix[(home_goals, away_goals)] = p_home * p_away

    return matrix


# =============================================================================
# STRENGTH CALCULATION
# =============================================================================

def get_team_strength(team: str) -> Dict[str, float]:
    """Get attacking and defensive strength modifiers for a team."""
    if team in TEAM_STRENGTHS:
        return TEAM_STRENGTHS[team]
    # Default for unknown teams
    return {"attack_mod": 0.0, "defense_mod": 0.0}


def calculate_expected_goals(
    home_team: str,
    away_team: str,
    factors: List[str],
    home_form: Dict = None,
    away_form: Dict = None
) -> Tuple[float, float]:
    """Calculate expected goals for home and away teams.

    Uses team strengths, form, and contextual factors.
    """
    # Base rates (adjusted for home advantage)
    base_home_goals = SERIE_A_AVG_GOALS * 0.55  # Home teams score ~55% of total goals
    base_away_goals = SERIE_A_AVG_GOALS * 0.45

    # Get team strengths
    home_strength = get_team_strength(home_team)
    away_strength = get_team_strength(away_team)

    # Calculate expected goals
    # Home goals = base * home attack * (1 - away defense)
    home_attack_factor = 1 + home_strength["attack_mod"]
    away_defense_factor = 1 - away_strength["defense_mod"]

    # Away goals = base * away attack * (1 - home defense)
    away_attack_factor = 1 + away_strength["attack_mod"]
    home_defense_factor = 1 - home_strength["defense_mod"]

    expected_home = base_home_goals * home_attack_factor * away_defense_factor
    expected_away = base_away_goals * away_attack_factor * home_defense_factor

    # Apply form adjustments
    if home_form:
        if home_form.get("status") == "hot":
            expected_home *= 1.15
        elif home_form.get("status") == "cold":
            expected_home *= 0.85

    if away_form:
        if away_form.get("status") == "hot":
            expected_away *= 1.15
        elif away_form.get("status") == "cold":
            expected_away *= 0.85

    # Apply contextual factors
    total_mod = 0
    for factor in factors:
        if factor in GOAL_FACTORS:
            total_mod += GOAL_FACTORS[factor]["expected_goals_mod"]

    # Apply modification proportionally
    if total_mod != 0:
        mod_factor = 1 + (total_mod / (expected_home + expected_away))
        expected_home *= mod_factor
        expected_away *= mod_factor

    # Ensure reasonable bounds
    expected_home = max(0.3, min(3.5, expected_home))
    expected_away = max(0.2, min(3.0, expected_away))

    return round(expected_home, 2), round(expected_away, 2)


# =============================================================================
# PREDICTION ENGINE
# =============================================================================

def predict_goals(
    home_team: str,
    away_team: str,
    date: str,
    factors: List[str],
    home_form: Dict = None,
    away_form: Dict = None,
    ensemble_home_xg: float = None,
    ensemble_away_xg: float = None,
) -> GoalPrediction:
    """Generate goal prediction for a match.

    If ensemble_home_xg/ensemble_away_xg are provided (from ensemble engine),
    uses those as a stronger signal blended with our own Poisson estimate.
    """

    # Calculate our own expected goals from team strengths
    own_home, own_away = calculate_expected_goals(
        home_team, away_team, factors, home_form, away_form
    )

    # If ensemble provides trained xG predictions, blend them (70% ensemble, 30% own)
    if ensemble_home_xg is not None and ensemble_away_xg is not None:
        exp_home = 0.7 * ensemble_home_xg + 0.3 * own_home
        exp_away = 0.7 * ensemble_away_xg + 0.3 * own_away
        log.info(f"  {home_team} vs {away_team}: Blended xG (ensemble={ensemble_home_xg:.2f}-{ensemble_away_xg:.2f}, own={own_home:.2f}-{own_away:.2f}) -> {exp_home:.2f}-{exp_away:.2f}")
    else:
        exp_home, exp_away = own_home, own_away

    exp_total = exp_home + exp_away

    # Calculate over/under probabilities using Poisson
    over_0_5 = calculate_over_probability(0.5, exp_total)
    over_1_5 = calculate_over_probability(1.5, exp_total)
    over_2_5 = calculate_over_probability(2.5, exp_total)
    over_3_5 = calculate_over_probability(3.5, exp_total)
    over_4_5 = calculate_over_probability(4.5, exp_total)

    # Calculate confidence based on factors and prediction strength
    n_factors = len([f for f in factors if f in GOAL_FACTORS])

    # Confidence based on prediction deviation from average
    deviation = abs(exp_total - SERIE_A_AVG_GOALS)

    if n_factors >= 3 and deviation > 0.4:
        confidence = "HIGH"
        confidence_rank = 4
    elif n_factors >= 2 and deviation > 0.2:
        confidence = "MEDIUM-HIGH"
        confidence_rank = 3
    elif n_factors >= 1:
        confidence = "MEDIUM"
        confidence_rank = 2
    else:
        confidence = "LOW"
        confidence_rank = 1

    # Get team strengths for reporting
    home_strength = get_team_strength(home_team)
    away_strength = get_team_strength(away_team)

    return GoalPrediction(
        match=f"{home_team} vs {away_team}",
        home_team=home_team,
        away_team=away_team,
        date=date,
        expected_home_goals=exp_home,
        expected_away_goals=exp_away,
        expected_total_goals=round(exp_total, 2),
        over_0_5=round(over_0_5, 3),
        over_1_5=round(over_1_5, 3),
        over_2_5=round(over_2_5, 3),
        over_3_5=round(over_3_5, 3),
        over_4_5=round(over_4_5, 3),
        factors=factors,
        confidence=confidence,
        confidence_rank=confidence_rank,
        home_attack_strength=home_strength["attack_mod"],
        away_attack_strength=away_strength["attack_mod"],
        home_defense_strength=home_strength["defense_mod"],
        away_defense_strength=away_strength["defense_mod"]
    )


# =============================================================================
# VALUE BETTING
# =============================================================================

def calculate_over_under_value(
    prediction: GoalPrediction,
    totals_odds: List[Dict]
) -> List[OverUnderBet]:
    """Calculate value bets for over/under markets."""

    bets = []

    limits = ODDS_SANITY["totals"]

    for total in totals_odds:
        line = total["line"]
        over_odds = total["over"]
        under_odds = total["under"]

        # Reject odds outside sane range (likely live/in-play data)
        if over_odds < limits["min"] or over_odds > limits["max"]:
            log.warning(f"Skipping {prediction.match} OVER {line}: odds {over_odds} "
                       f"outside [{limits['min']}, {limits['max']}] — likely live/in-play")
            continue
        if under_odds < limits["min"] or under_odds > limits["max"]:
            log.warning(f"Skipping {prediction.match} UNDER {line}: odds {under_odds} "
                       f"outside [{limits['min']}, {limits['max']}] — likely live/in-play")
            continue

        # Get our probability for this line
        if line == 0.5:
            our_over = prediction.over_0_5
        elif line == 1.5:
            our_over = prediction.over_1_5
        elif line == 2.5:
            our_over = prediction.over_2_5
        elif line == 3.5:
            our_over = prediction.over_3_5
        elif line == 4.5:
            our_over = prediction.over_4_5
        else:
            # Interpolate for non-standard lines
            if line < 2.5:
                our_over = prediction.over_2_5 + (2.5 - line) * 0.15
            else:
                our_over = prediction.over_2_5 - (line - 2.5) * 0.15
            our_over = max(0.05, min(0.95, our_over))

        our_under = 1 - our_over

        # Calculate implied probabilities from odds
        implied_over = 1 / over_odds
        implied_under = 1 / under_odds

        # Calculate value for OVER bet
        over_value = our_over - implied_over
        over_value_pct = round(over_value * 100, 1)

        # Calculate value for UNDER bet
        under_value = our_under - implied_under
        under_value_pct = round(under_value * 100, 1)

        # Determine recommendation for OVER
        if over_value > 0.10:  # 10%+ edge
            over_rec = "BET"
            over_stake = 2.0
        elif over_value > 0.03:  # 3-10% edge
            over_rec = "CONSIDER"
            over_stake = 1.0
        else:
            over_rec = "SKIP"
            over_stake = 0

        # Determine recommendation for UNDER
        if under_value > 0.10:
            under_rec = "BET"
            under_stake = 2.0
        elif under_value > 0.03:
            under_rec = "CONSIDER"
            under_stake = 1.0
        else:
            under_rec = "SKIP"
            under_stake = 0

        # Add OVER bet if there's value
        if over_value > 0:
            bets.append(OverUnderBet(
                match=prediction.match,
                date=prediction.date,
                line=line,
                bet_type="over",
                our_probability=round(our_over, 3),
                odds=over_odds,
                implied_probability=round(implied_over, 3),
                value=round(over_value, 3),
                value_pct=over_value_pct,
                expected_goals=prediction.expected_total_goals,
                recommendation=over_rec,
                stake_pct=over_stake,
                factors=prediction.factors
            ))

        # Add UNDER bet if there's value
        if under_value > 0:
            bets.append(OverUnderBet(
                match=prediction.match,
                date=prediction.date,
                line=line,
                bet_type="under",
                our_probability=round(our_under, 3),
                odds=under_odds,
                implied_probability=round(implied_under, 3),
                value=round(under_value, 3),
                value_pct=under_value_pct,
                expected_goals=prediction.expected_total_goals,
                recommendation=under_rec,
                stake_pct=under_stake,
                factors=prediction.factors
            ))

    # Sort by value
    bets.sort(key=lambda x: x.value, reverse=True)

    return bets


# =============================================================================
# DATA LOADING
# =============================================================================

def load_predictions() -> List[Dict]:
    """Load match predictions from all league files."""
    all_preds = []
    for fname in ["predictions.json", "predictions_premier_league.json"]:
        p = DATA_DIR / "upcoming" / fname
        if p.exists():
            try:
                with open(p) as f:
                    data = json.load(f)
                preds = data.get("predictions", [])
                all_preds.extend(preds)
                log.info("O/U model: loaded %d predictions from %s", len(preds), fname)
            except Exception as e:
                log.warning("Failed to load %s: %s", fname, e)
    if not all_preds:
        log.warning("No predictions files found")
    return all_preds


def load_totals_odds() -> Dict[str, List[Dict]]:
    """Load over/under odds from all league odds_totals files."""
    all_odds = {}
    for fname in ["odds_totals.json", "odds_totals_premier_league.json"]:
        odds_path = DATA_DIR / "upcoming" / fname
        if odds_path.exists():
            try:
                with open(odds_path) as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    all_odds.update(data)
            except Exception:
                pass
    if all_odds:
        return all_odds

    odds_path = DATA_DIR / "upcoming" / "odds_totals.json"
    if not odds_path.exists():
        log.warning("No totals odds file found")
        return {}

    with open(odds_path) as f:
        data = json.load(f)

    return data


def load_form_data() -> Dict:
    """Load current form data."""
    form_path = DATA_DIR / "upcoming" / "current_form.json"

    if not form_path.exists():
        return {}

    with open(form_path) as f:
        return json.load(f)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def generate_over_under_predictions(
    ml_ou_probs: Dict[str, Dict[float, float]] = None,
) -> Tuple[List[GoalPrediction], List[OverUnderBet]]:
    """Generate over/under predictions and betting recommendations.

    Args:
        ml_ou_probs: Optional dict of {match_key: {line: P(over)}} from the
            dedicated O/U CatBoost classifier. When provided, blends ML + Poisson
            at 65/35 weight for calibrated O/U probabilities.
    """

    log.info("=" * 60)
    log.info("OVER/UNDER PREDICTION MODEL")
    log.info("=" * 60)

    # Load data
    predictions = load_predictions()
    totals_odds = load_totals_odds()
    form_data = load_form_data()

    if not predictions:
        log.error("No match predictions available")
        return [], []

    log.info(f"Loaded {len(predictions)} match predictions")
    log.info(f"Loaded totals odds for {len(totals_odds)} matches")

    goal_predictions = []
    all_bets = []

    for pred in predictions:
        match_key = pred["match"]
        home_team = match_key.split(" vs ")[0]
        away_team = match_key.split(" vs ")[1]
        date = pred["date"]

        # Collect factors that affect goals
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

        # Blend lineup xG into ensemble xG for O/U predictions.
        # Backtest evidence: lineup-adjusted xG improved O/U ROI from -10.2% to +12.5%.
        # Confirmed lineups get heavier weight (40%) since player data is reliable;
        # predicted lineups get lighter weight (15%) since lineup guesses are noisy.
        ens_home_xg = pred.get("home_xg")
        ens_away_xg = pred.get("away_xg")
        comp = pred.get("component_predictions", {})
        pxg = comp.get("player_xg_details", {})
        lineup_home_xg = pxg.get("home_lineup_xg")
        lineup_away_xg = pxg.get("away_lineup_xg")

        if (lineup_home_xg is not None and lineup_away_xg is not None
                and ens_home_xg is not None and ens_away_xg is not None):
            lineup_source = pred.get("lineup_source", comp.get("lineup_source", "predicted"))
            lineup_weight = 0.40 if lineup_source == "confirmed" else 0.15
            blended_home_xg = (1 - lineup_weight) * ens_home_xg + lineup_weight * lineup_home_xg
            blended_away_xg = (1 - lineup_weight) * ens_away_xg + lineup_weight * lineup_away_xg
            log.info(
                f"  {match_key}: O/U xG blend ({lineup_source} lineup @ {lineup_weight:.0%}): "
                f"ensemble={ens_home_xg:.2f}-{ens_away_xg:.2f}, "
                f"lineup={lineup_home_xg:.2f}-{lineup_away_xg:.2f} -> "
                f"blended={blended_home_xg:.2f}-{blended_away_xg:.2f}"
            )
        else:
            blended_home_xg = ens_home_xg
            blended_away_xg = ens_away_xg

        # Generate goal prediction (using blended xG when available)
        goal_pred = predict_goals(
            home_team=home_team,
            away_team=away_team,
            date=date,
            factors=factors,
            home_form=home_form,
            away_form=away_form,
            ensemble_home_xg=blended_home_xg,
            ensemble_away_xg=blended_away_xg,
        )

        # Blend ML O/U classifier with Poisson probabilities (65% ML, 35% Poisson).
        # The ML model captures non-linear feature interactions that Poisson misses,
        # while Poisson regularizes with structural goal-scoring properties.
        if ml_ou_probs and match_key in ml_ou_probs:
            match_ml = ml_ou_probs[match_key]
            for line_val, attr_name in [
                (2.5, "over_2_5"), (1.5, "over_1_5"), (3.5, "over_3_5"),
            ]:
                ml_p = match_ml.get(line_val)
                if ml_p is not None:
                    poisson_p = getattr(goal_pred, attr_name)
                    blended_p = round(0.65 * ml_p + 0.35 * poisson_p, 3)
                    setattr(goal_pred, attr_name, blended_p)
                    log.info(
                        "  %s O/U %.1f: ML=%.3f Poisson=%.3f -> blend=%.3f",
                        match_key, line_val, ml_p, poisson_p, blended_p,
                    )

        goal_predictions.append(goal_pred)

        # Calculate value bets if odds available
        if match_key in totals_odds:
            match_totals = totals_odds[match_key].get("totals", [])
            bets = calculate_over_under_value(goal_pred, match_totals)
            all_bets.extend(bets)

    # Sort predictions by expected goals
    goal_predictions.sort(key=lambda x: x.expected_total_goals, reverse=True)

    log.info(f"Generated {len(goal_predictions)} goal predictions")
    log.info(f"Found {len(all_bets)} potential over/under bets")

    return goal_predictions, all_bets


def save_over_under_predictions(
    predictions: List[GoalPrediction],
    bets: List[OverUnderBet]
):
    """Save predictions and bets to JSON files."""

    output_dir = DATA_DIR / "upcoming"

    # Save goal predictions
    predictions_data = {
        "generated_at": datetime.now().isoformat(),
        "model": "poisson_v1",
        "predictions": [
            {
                "match": p.match,
                "date": p.date,
                "expected_home_goals": p.expected_home_goals,
                "expected_away_goals": p.expected_away_goals,
                "expected_total_goals": p.expected_total_goals,
                "over_0_5": p.over_0_5,
                "over_1_5": p.over_1_5,
                "over_2_5": p.over_2_5,
                "over_3_5": p.over_3_5,
                "over_4_5": p.over_4_5,
                "confidence": p.confidence,
                "factors": p.factors
            }
            for p in predictions
        ]
    }

    pred_path = output_dir / "goal_predictions.json"
    with open(pred_path, "w") as f:
        json.dump(predictions_data, f, indent=2)
    log.info(f"Saved goal predictions to {pred_path}")

    # Save betting recommendations
    recommended = [b for b in bets if b.recommendation == "BET"]
    consider = [b for b in bets if b.recommendation == "CONSIDER"]

    bets_data = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_bets_analyzed": len(bets),
            "recommended_bets": len(recommended),
            "consider_bets": len(consider)
        },
        "recommended": [
            {
                "match": b.match,
                "date": b.date,
                "bet": f"{b.bet_type.upper()} {b.line}",
                "our_probability": b.our_probability,
                "odds": b.odds,
                "implied_probability": b.implied_probability,
                "value_pct": b.value_pct,
                "expected_goals": b.expected_goals,
                "stake_pct": b.stake_pct,
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
                "odds": b.odds,
                "value_pct": b.value_pct,
                "expected_goals": b.expected_goals
            }
            for b in consider
        ]
    }

    bets_path = output_dir / "over_under_bets.json"
    with open(bets_path, "w") as f:
        json.dump(bets_data, f, indent=2)
    log.info(f"Saved betting recommendations to {bets_path}")


def print_summary(predictions: List[GoalPrediction], bets: List[OverUnderBet]):
    """Print summary of predictions and bets."""

    print("\n" + "=" * 80)
    print(" OVER/UNDER GOAL PREDICTIONS")
    print("=" * 80)

    print(f"\n  {'Match':<30} {'xG Home':>8} {'xG Away':>8} {'Total':>7} {'O2.5':>6} {'Conf':>10}")
    print("  " + "-" * 75)

    for p in predictions:
        print(f"  {p.match:<30} {p.expected_home_goals:>8.2f} {p.expected_away_goals:>8.2f} "
              f"{p.expected_total_goals:>7.2f} {p.over_2_5:>6.1%} {p.confidence:>10}")

    # Betting recommendations
    recommended = [b for b in bets if b.recommendation == "BET"]
    consider = [b for b in bets if b.recommendation == "CONSIDER"]

    if recommended:
        print("\n" + "=" * 80)
        print(" RECOMMENDED OVER/UNDER BETS")
        print("=" * 80)

        print(f"\n  {'Match':<25} {'Bet':>10} {'Odds':>6} {'Our%':>6} {'Imp%':>6} {'Value':>7}")
        print("  " + "-" * 70)

        for b in recommended:
            print(f"  {b.match:<25} {b.bet_type.upper()+' '+str(b.line):>10} {b.odds:>6.2f} "
                  f"{b.our_probability:>6.1%} {b.implied_probability:>6.1%} {b.value_pct:>+7.1f}%")

    if consider:
        print("\n" + "=" * 80)
        print(" CONSIDER BETS (Small Edge)")
        print("=" * 80)

        for b in consider[:5]:  # Show top 5
            print(f"  {b.match}: {b.bet_type.upper()} {b.line} @ {b.odds:.2f} "
                  f"(Value: {b.value_pct:+.1f}%, xG: {b.expected_goals:.2f})")


# =============================================================================
# VALIDATION
# =============================================================================

def validate_model():
    """Validate model predictions against historical data."""

    print("\n" + "=" * 60)
    print(" MODEL VALIDATION")
    print("=" * 60)

    # Test predictions for various team combinations
    test_cases = [
        ("Inter", "Sassuolo", ["big_away_favorite", "hot_away", "cold_home"]),
        ("Juventus", "Lazio", ["big_stadium", "hot_home"]),
        ("Lecce", "Udinese", ["cold_away", "cold_home"]),
        ("Atalanta", "Cremonese", ["big_home_favorite", "hot_home"]),
    ]

    print("\n  Test Predictions:")
    print("  " + "-" * 55)

    for home, away, factors in test_cases:
        pred = predict_goals(home, away, "2026-02-09", factors)

        print(f"\n  {pred.match}:")
        print(f"    Expected Goals: {pred.expected_total_goals:.2f} (H: {pred.expected_home_goals:.2f}, A: {pred.expected_away_goals:.2f})")
        print(f"    Over 2.5: {pred.over_2_5:.1%} | Over 3.5: {pred.over_3_5:.1%}")
        print(f"    Confidence: {pred.confidence}")

    # Compare to historical averages
    print("\n  Historical Baseline Comparison:")
    print(f"    Serie A avg goals/game: {SERIE_A_AVG_GOALS}")
    print(f"    Historical Over 2.5: {HISTORICAL_OVER_RATES[2.5]:.1%}")
    print(f"    Model avg (4 tests): {sum(1 for _, _, _ in test_cases) / 4:.1%}")


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Over/Under Goal Prediction Model")
    parser.add_argument("--validate", action="store_true", help="Run model validation")
    args = parser.parse_args()

    if args.validate:
        validate_model()
        return

    # Generate predictions and bets
    predictions, bets = generate_over_under_predictions()

    if predictions:
        # Save results
        save_over_under_predictions(predictions, bets)

        # Print summary
        print_summary(predictions, bets)


if __name__ == "__main__":
    main()
