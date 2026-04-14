#!/usr/bin/env python3
"""BTTS & CORNERS PREDICTION MODEL - Phase 6 Implementation

Predicts Both Teams To Score (BTTS) and corner totals.

Features:
- BTTS probability based on team scoring/conceding rates
- Corner prediction using team styles and xG
- Integration with goal predictions
- Weather and referee factor adjustments

Now fetches real BTTS odds from The Odds API for accurate value detection.
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

# Odds sanity — reject live/in-play odds before generating recommendations
ODDS_SANITY_BTTS = {"min": 1.10, "max": 8.0}
ODDS_SANITY_CORNERS = {"min": 1.10, "max": 10.0}

# =============================================================================
# HISTORICAL DATA & CONSTANTS
# =============================================================================

# Serie A averages
SERIE_A_BTTS_RATE = 0.52   # 52% of matches have BTTS
SERIE_A_AVG_CORNERS = 10.2  # Average corners per match

# Historical corner rates
HISTORICAL_CORNER_RATES = {
    8.5: 0.72,   # 72% over 8.5 corners
    9.5: 0.58,   # 58% over 9.5 corners
    10.5: 0.45,  # 45% over 10.5 corners
    11.5: 0.32,  # 32% over 11.5 corners
    12.5: 0.20,  # 20% over 12.5 corners
}

# Team scoring rates (P(team scores) based on historical data)
TEAM_SCORING_RATES = {
    # Home scoring rates
    "home": {
        "Inter": 0.85,
        "Napoli": 0.82,
        "Atalanta": 0.83,
        "Milan": 0.78,
        "Juventus": 0.77,
        "Roma": 0.75,
        "Lazio": 0.76,
        "Fiorentina": 0.72,
        "Bologna": 0.70,
        "Torino": 0.68,
        "Genoa": 0.65,
        "Udinese": 0.62,
        "Sassuolo": 0.58,
        "Como": 0.65,
        "Parma": 0.62,
        "Verona": 0.60,
        "Cagliari": 0.58,
        "Lecce": 0.55,
        "Sassuolo": 0.58,
        "Cremonese": 0.50,
        "Pisa": 0.55,
    },
    # Away scoring rates (generally lower)
    "away": {
        "Inter": 0.78,
        "Napoli": 0.75,
        "Atalanta": 0.77,
        "Milan": 0.72,
        "Juventus": 0.70,
        "Roma": 0.68,
        "Lazio": 0.70,
        "Fiorentina": 0.65,
        "Bologna": 0.62,
        "Torino": 0.60,
        "Genoa": 0.55,
        "Udinese": 0.52,
        "Sassuolo": 0.48,
        "Como": 0.55,
        "Parma": 0.52,
        "Verona": 0.50,
        "Cagliari": 0.48,
        "Lecce": 0.45,
        "Sassuolo": 0.48,
        "Cremonese": 0.40,
        "Pisa": 0.45,
    }
}

# Team clean sheet rates (P(team keeps clean sheet))
TEAM_CLEAN_SHEET_RATES = {
    "home": {
        "Inter": 0.42,
        "Juventus": 0.40,
        "Napoli": 0.38,
        "Milan": 0.35,
        "Atalanta": 0.32,
        "Roma": 0.30,
        "Lazio": 0.28,
        "Fiorentina": 0.28,
        "Bologna": 0.30,
        "Torino": 0.28,
        "Genoa": 0.25,
        "Udinese": 0.28,
        "Sassuolo": 0.22,
        "Como": 0.22,
        "Parma": 0.22,
        "Verona": 0.22,
        "Cagliari": 0.22,
        "Lecce": 0.25,
        "Sassuolo": 0.20,
        "Cremonese": 0.18,
        "Pisa": 0.22,
    },
    "away": {
        "Inter": 0.35,
        "Juventus": 0.32,
        "Napoli": 0.30,
        "Milan": 0.28,
        "Atalanta": 0.25,
        "Roma": 0.22,
        "Lazio": 0.22,
        "Fiorentina": 0.20,
        "Bologna": 0.22,
        "Torino": 0.20,
        "Genoa": 0.18,
        "Udinese": 0.20,
        "Sassuolo": 0.15,
        "Como": 0.15,
        "Parma": 0.15,
        "Verona": 0.15,
        "Cagliari": 0.15,
        "Lecce": 0.18,
        "Sassuolo": 0.15,
        "Cremonese": 0.12,
        "Pisa": 0.15,
    }
}

# Team corner averages (corners won per match)
TEAM_CORNERS = {
    "home": {
        "Inter": 6.2,
        "Napoli": 5.8,
        "Atalanta": 6.5,
        "Milan": 5.6,
        "Juventus": 5.4,
        "Roma": 5.5,
        "Lazio": 5.4,
        "Fiorentina": 5.2,
        "Bologna": 5.0,
        "Torino": 4.8,
        "Genoa": 4.5,
        "Udinese": 4.2,
        "Sassuolo": 4.2,
        "Como": 4.5,
        "Parma": 4.2,
        "Verona": 4.0,
        "Cagliari": 4.0,
        "Lecce": 4.2,
        "Sassuolo": 4.5,
        "Cremonese": 3.8,
        "Pisa": 4.0,
    },
    "away": {
        "Inter": 5.5,
        "Napoli": 5.2,
        "Atalanta": 5.8,
        "Milan": 5.0,
        "Juventus": 4.8,
        "Roma": 4.8,
        "Lazio": 4.8,
        "Fiorentina": 4.5,
        "Bologna": 4.2,
        "Torino": 4.0,
        "Genoa": 3.8,
        "Udinese": 3.5,
        "Sassuolo": 3.5,
        "Como": 3.8,
        "Parma": 3.5,
        "Verona": 3.5,
        "Cagliari": 3.5,
        "Lecce": 3.5,
        "Sassuolo": 3.8,
        "Cremonese": 3.2,
        "Pisa": 3.5,
    }
}

# Factors affecting BTTS
BTTS_FACTORS = {
    "hot_home": 0.08,      # Hot teams more likely to score
    "hot_away": 0.08,
    "cold_home": -0.10,    # Cold teams less likely to score
    "cold_away": -0.08,
    "big_home_favorite": -0.05,  # Favorites often shut out weak teams
    "big_away_favorite": -0.05,
    "derby": 0.10,         # Derbies often have goals from both
}

# Factors affecting corners
CORNERS_FACTORS = {
    "hot_home": 0.5,       # Attacking teams generate more corners
    "hot_away": 0.5,
    "cold_home": -0.3,
    "cold_away": -0.3,
    "big_home_favorite": 0.8,  # Dominant teams win more corners
    "big_away_favorite": 0.8,
    "rain": 0.5,           # More long balls/crosses in rain
    "wind": 0.3,
    "derby": 0.5,          # Intense matches = more corners
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BTTSPrediction:
    """Both Teams To Score prediction."""
    match: str
    home_team: str
    away_team: str
    date: str

    # Probabilities
    btts_yes: float
    btts_no: float

    # Component probabilities
    home_scores_prob: float
    away_scores_prob: float
    home_clean_sheet: float
    away_clean_sheet: float

    # Factors
    factors: List[str]
    confidence: str
    confidence_rank: int


@dataclass
class CornersPrediction:
    """Corners prediction."""
    match: str
    home_team: str
    away_team: str
    date: str

    # Expected corners
    expected_home_corners: float
    expected_away_corners: float
    expected_total_corners: float

    # Probabilities
    over_8_5: float
    over_9_5: float
    over_10_5: float
    over_11_5: float

    # Factors
    factors: List[str]
    confidence: str
    confidence_rank: int


@dataclass
class BTTSCornersBet:
    """Betting recommendation."""
    match: str
    date: str
    market: str  # "btts" or "corners"
    bet_type: str
    line: Optional[float]  # For corners
    our_probability: float
    fair_odds: float
    expected_value: float
    recommendation: str
    factors: List[str]


# =============================================================================
# BTTS CALCULATIONS
# =============================================================================

def get_scoring_rate(team: str, location: str) -> float:
    """Get team's scoring rate."""
    rates = TEAM_SCORING_RATES.get(location, {})
    return rates.get(team, 0.55)


def get_clean_sheet_rate(team: str, location: str) -> float:
    """Get team's clean sheet rate."""
    rates = TEAM_CLEAN_SHEET_RATES.get(location, {})
    return rates.get(team, 0.20)


def calculate_btts_probability(
    home_team: str,
    away_team: str,
    factors: List[str],
    home_form: Dict = None,
    away_form: Dict = None
) -> Tuple[float, float, float, float, float]:
    """Calculate BTTS probability.

    BTTS Yes = P(Home scores) × P(Away scores)
    But adjusted for defensive strength

    Returns:
        Tuple of (btts_yes, home_scores, away_scores, home_cs, away_cs)
    """
    # Base scoring probabilities
    home_scores_base = get_scoring_rate(home_team, "home")
    away_scores_base = get_scoring_rate(away_team, "away")

    # Adjust for opponent's defensive strength
    away_cs = get_clean_sheet_rate(away_team, "away")
    home_cs = get_clean_sheet_rate(home_team, "home")

    # Adjusted scoring probabilities
    home_scores = home_scores_base * (1 - away_cs * 0.5)  # Partial adjustment
    away_scores = away_scores_base * (1 - home_cs * 0.5)

    # Apply form adjustments
    form_adj = 0
    for factor in factors:
        if factor in BTTS_FACTORS:
            form_adj += BTTS_FACTORS[factor]

    # Apply form to both teams
    if home_form and home_form.get("status") == "hot":
        home_scores += 0.05
    if home_form and home_form.get("status") == "cold":
        home_scores -= 0.08
    if away_form and away_form.get("status") == "hot":
        away_scores += 0.05
    if away_form and away_form.get("status") == "cold":
        away_scores -= 0.08

    # Clamp probabilities
    home_scores = max(0.20, min(0.95, home_scores))
    away_scores = max(0.15, min(0.90, away_scores))

    # Calculate BTTS
    btts_yes = home_scores * away_scores

    # Apply factor adjustments to BTTS
    btts_yes += form_adj
    btts_yes = max(0.20, min(0.85, btts_yes))

    return btts_yes, home_scores, away_scores, home_cs, away_cs


def predict_btts(
    home_team: str,
    away_team: str,
    date: str,
    factors: List[str],
    home_form: Dict = None,
    away_form: Dict = None,
    ensemble_home_xg: float = None,
    ensemble_away_xg: float = None,
) -> BTTSPrediction:
    """Generate BTTS prediction.

    If ensemble xG is provided, uses Poisson to derive scoring probabilities
    and blends with own estimate for better BTTS prediction.
    """

    btts_yes, home_scores, away_scores, home_cs, away_cs = calculate_btts_probability(
        home_team, away_team, factors, home_form, away_form
    )

    # If ensemble xG available, improve scoring probabilities via Poisson
    if ensemble_home_xg is not None and ensemble_away_xg is not None:
        from scipy.stats import poisson
        # P(team scores) = 1 - P(0 goals) using Poisson
        xg_home_scores = 1 - poisson.pmf(0, max(0.3, ensemble_home_xg))
        xg_away_scores = 1 - poisson.pmf(0, max(0.3, ensemble_away_xg))
        # Blend 70% ensemble, 30% own
        home_scores = 0.7 * xg_home_scores + 0.3 * home_scores
        away_scores = 0.7 * xg_away_scores + 0.3 * away_scores
        btts_yes = home_scores * away_scores
        btts_yes = max(0.20, min(0.85, btts_yes))
        log.info(f"  {home_team} vs {away_team}: BTTS blended with ensemble xG -> {btts_yes:.3f}")

    btts_no = 1 - btts_yes

    # Confidence based on probability strength
    deviation = abs(btts_yes - SERIE_A_BTTS_RATE)
    n_factors = len([f for f in factors if f in BTTS_FACTORS])

    if deviation > 0.15 and n_factors >= 1:
        confidence = "HIGH"
        confidence_rank = 4
    elif deviation > 0.08:
        confidence = "MEDIUM-HIGH"
        confidence_rank = 3
    elif n_factors >= 1:
        confidence = "MEDIUM"
        confidence_rank = 2
    else:
        confidence = "LOW"
        confidence_rank = 1

    return BTTSPrediction(
        match=f"{home_team} vs {away_team}",
        home_team=home_team,
        away_team=away_team,
        date=date,
        btts_yes=round(btts_yes, 3),
        btts_no=round(btts_no, 3),
        home_scores_prob=round(home_scores, 3),
        away_scores_prob=round(away_scores, 3),
        home_clean_sheet=round(home_cs, 3),
        away_clean_sheet=round(away_cs, 3),
        factors=factors,
        confidence=confidence,
        confidence_rank=confidence_rank
    )


# =============================================================================
# CORNERS CALCULATIONS
# =============================================================================

def get_team_corners(team: str, location: str) -> float:
    """Get team's average corners."""
    corners = TEAM_CORNERS.get(location, {})
    return corners.get(team, 4.0)


def poisson_probability(k: int, lambda_: float) -> float:
    """Calculate Poisson probability."""
    if lambda_ <= 0:
        return 1.0 if k == 0 else 0.0
    return (math.exp(-lambda_) * (lambda_ ** k)) / math.factorial(k)


def poisson_cumulative(k: int, lambda_: float) -> float:
    """Calculate cumulative Poisson P(X <= k)."""
    return sum(poisson_probability(i, lambda_) for i in range(k + 1))


def calculate_corners(
    home_team: str,
    away_team: str,
    factors: List[str],
    home_form: Dict = None,
    away_form: Dict = None,
    home_lineup: List[str] = None,
    away_lineup: List[str] = None,
) -> Tuple[float, float]:
    """Calculate expected corners.

    Args:
        home_team: Home team name
        away_team: Away team name
        factors: List of match factors
        home_form: Home team form data
        away_form: Away team form data
        home_lineup: Optional confirmed home lineup for corner taker check
        away_lineup: Optional confirmed away lineup for corner taker check

    Returns:
        Tuple of (home_corners, away_corners)
    """
    # Base corners
    home_corners = get_team_corners(home_team, "home")
    away_corners = get_team_corners(away_team, "away")

    # Apply factor adjustments
    factor_adj = 0
    for factor in factors:
        if factor in CORNERS_FACTORS:
            factor_adj += CORNERS_FACTORS[factor]

    # Apply to total
    home_corners += factor_adj * 0.6
    away_corners += factor_adj * 0.4

    # Form adjustments
    if home_form and home_form.get("status") == "hot":
        home_corners += 0.3
    if home_form and home_form.get("status") == "cold":
        home_corners -= 0.3
    if away_form and away_form.get("status") == "hot":
        away_corners += 0.3
    if away_form and away_form.get("status") == "cold":
        away_corners -= 0.3

    # Lineup-based corner taker adjustment
    try:
        from features.lineup_stats import get_lineup_expected_corners
        home_corner_data = get_lineup_expected_corners(home_team, home_lineup)
        away_corner_data = get_lineup_expected_corners(away_team, away_lineup)
        home_corners += home_corner_data.get("expected_corners_adj", 0)
        away_corners += away_corner_data.get("expected_corners_adj", 0)
    except Exception:
        pass  # Lineup stats not available

    # Clamp
    home_corners = max(2.5, min(8.0, home_corners))
    away_corners = max(2.0, min(7.0, away_corners))

    return round(home_corners, 2), round(away_corners, 2)


def predict_corners(
    home_team: str,
    away_team: str,
    date: str,
    factors: List[str],
    home_form: Dict = None,
    away_form: Dict = None,
    home_lineup: List[str] = None,
    away_lineup: List[str] = None,
) -> CornersPrediction:
    """Generate corners prediction."""

    home_corners, away_corners = calculate_corners(
        home_team, away_team, factors, home_form, away_form,
        home_lineup=home_lineup, away_lineup=away_lineup,
    )

    total_corners = home_corners + away_corners

    # Calculate over probabilities
    over_8_5 = 1 - poisson_cumulative(8, total_corners)
    over_9_5 = 1 - poisson_cumulative(9, total_corners)
    over_10_5 = 1 - poisson_cumulative(10, total_corners)
    over_11_5 = 1 - poisson_cumulative(11, total_corners)

    # Confidence
    deviation = abs(total_corners - SERIE_A_AVG_CORNERS)
    n_factors = len([f for f in factors if f in CORNERS_FACTORS])

    if deviation > 1.5 and n_factors >= 1:
        confidence = "HIGH"
        confidence_rank = 4
    elif deviation > 0.8:
        confidence = "MEDIUM-HIGH"
        confidence_rank = 3
    elif n_factors >= 1:
        confidence = "MEDIUM"
        confidence_rank = 2
    else:
        confidence = "LOW"
        confidence_rank = 1

    return CornersPrediction(
        match=f"{home_team} vs {away_team}",
        home_team=home_team,
        away_team=away_team,
        date=date,
        expected_home_corners=home_corners,
        expected_away_corners=away_corners,
        expected_total_corners=round(total_corners, 2),
        over_8_5=round(over_8_5, 3),
        over_9_5=round(over_9_5, 3),
        over_10_5=round(over_10_5, 3),
        over_11_5=round(over_11_5, 3),
        factors=factors,
        confidence=confidence,
        confidence_rank=confidence_rank
    )


# =============================================================================
# BETTING RECOMMENDATIONS
# =============================================================================

def _load_btts_odds() -> Dict[str, Dict]:
    """Load real BTTS odds from odds_btts.json."""
    btts_path = DATA_DIR / "upcoming" / "odds_btts.json"
    if btts_path.exists():
        try:
            with open(btts_path) as f:
                data = json.load(f)
            log.info(f"Loaded real BTTS odds for {len(data)} matches")
            return data
        except Exception as e:
            log.warning(f"Could not load BTTS odds: {e}")
    return {}


# Module-level cache
_BTTS_ODDS_CACHE = None


def _get_btts_odds() -> Dict[str, Dict]:
    """Get cached BTTS odds."""
    global _BTTS_ODDS_CACHE
    if _BTTS_ODDS_CACHE is None:
        _BTTS_ODDS_CACHE = _load_btts_odds()
    return _BTTS_ODDS_CACHE


def generate_btts_bets(prediction: BTTSPrediction) -> List[BTTSCornersBet]:
    """Generate BTTS betting recommendations using real odds when available."""
    bets = []
    real_odds = _get_btts_odds()
    match_odds = real_odds.get(prediction.match, {})

    # BTTS Yes
    if prediction.btts_yes > 0.58:  # Higher than average
        # Use real odds if available, otherwise estimate
        if match_odds.get("yes", 0) > 1.0:
            actual_odds = match_odds["yes"]
            if actual_odds > ODDS_SANITY_BTTS["max"] or actual_odds < ODDS_SANITY_BTTS["min"]:
                log.warning(f"Skipping {prediction.match} BTTS Yes: odds {actual_odds} "
                           f"outside sane range — likely live/in-play")
                actual_odds = 0  # Force fallback
            if actual_odds > 0:
                implied_prob = 1.0 / actual_odds
                edge = prediction.btts_yes - implied_prob
                fair_odds = actual_odds
            else:
                fair_odds = 1 / prediction.btts_yes
                edge = prediction.btts_yes - SERIE_A_BTTS_RATE
        else:
            fair_odds = 1 / prediction.btts_yes
            edge = prediction.btts_yes - SERIE_A_BTTS_RATE

        recommendation = "BET" if prediction.btts_yes > 0.65 and edge > 0.05 else "CONSIDER"
        bets.append(BTTSCornersBet(
            match=prediction.match,
            date=prediction.date,
            market="btts",
            bet_type="yes",
            line=None,
            our_probability=prediction.btts_yes,
            fair_odds=round(fair_odds, 2),
            expected_value=round(edge, 3),
            recommendation=recommendation,
            factors=prediction.factors
        ))

    # BTTS No
    if prediction.btts_no > 0.55:  # Higher than average
        if match_odds.get("no", 0) > 1.0:
            actual_odds = match_odds["no"]
            if actual_odds > ODDS_SANITY_BTTS["max"] or actual_odds < ODDS_SANITY_BTTS["min"]:
                log.warning(f"Skipping {prediction.match} BTTS No: odds {actual_odds} "
                           f"outside sane range — likely live/in-play")
                actual_odds = 0
            if actual_odds > 0:
                implied_prob = 1.0 / actual_odds
                edge = prediction.btts_no - implied_prob
                fair_odds = actual_odds
            else:
                fair_odds = 1 / prediction.btts_no
                edge = prediction.btts_no - (1 - SERIE_A_BTTS_RATE)
        else:
            fair_odds = 1 / prediction.btts_no
            edge = prediction.btts_no - (1 - SERIE_A_BTTS_RATE)

        recommendation = "BET" if prediction.btts_no > 0.60 and edge > 0.05 else "CONSIDER"
        bets.append(BTTSCornersBet(
            match=prediction.match,
            date=prediction.date,
            market="btts",
            bet_type="no",
            line=None,
            our_probability=prediction.btts_no,
            fair_odds=round(fair_odds, 2),
            expected_value=round(edge, 3),
            recommendation=recommendation,
            factors=prediction.factors
        ))

    return bets


def generate_corners_bets(prediction: CornersPrediction) -> List[BTTSCornersBet]:
    """Generate corners betting recommendations."""
    bets = []

    lines = [
        (8.5, prediction.over_8_5, HISTORICAL_CORNER_RATES[8.5]),
        (9.5, prediction.over_9_5, HISTORICAL_CORNER_RATES[9.5]),
        (10.5, prediction.over_10_5, HISTORICAL_CORNER_RATES[10.5]),
        (11.5, prediction.over_11_5, HISTORICAL_CORNER_RATES[11.5]),
    ]

    for line, over_prob, historical in lines:
        # Over bet
        if over_prob > historical + 0.08:  # Significant edge
            fair_odds = 1 / over_prob
            recommendation = "BET" if over_prob > historical + 0.15 else "CONSIDER"
            bets.append(BTTSCornersBet(
                match=prediction.match,
                date=prediction.date,
                market="corners",
                bet_type="over",
                line=line,
                our_probability=over_prob,
                fair_odds=round(fair_odds, 2),
                expected_value=over_prob - historical,
                recommendation=recommendation,
                factors=prediction.factors
            ))

        # Under bet
        under_prob = 1 - over_prob
        under_historical = 1 - historical
        if under_prob > under_historical + 0.08:
            fair_odds = 1 / under_prob
            recommendation = "BET" if under_prob > under_historical + 0.15 else "CONSIDER"
            bets.append(BTTSCornersBet(
                match=prediction.match,
                date=prediction.date,
                market="corners",
                bet_type="under",
                line=line,
                our_probability=under_prob,
                fair_odds=round(fair_odds, 2),
                expected_value=under_prob - under_historical,
                recommendation=recommendation,
                factors=prediction.factors
            ))

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
                all_preds.extend(data.get("predictions", []))
            except Exception:
                pass
    return all_preds


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def generate_all_predictions() -> Tuple[List, List, List]:
    """Generate BTTS and corners predictions."""

    log.info("=" * 60)
    log.info("BTTS & CORNERS PREDICTION MODEL")
    log.info("=" * 60)

    predictions = load_predictions()

    if not predictions:
        log.error("No match predictions available")
        return [], [], []

    log.info(f"Loaded {len(predictions)} match predictions")

    btts_preds = []
    corners_preds = []
    all_bets = []

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
        log.warning(f"Failed to load confirmed lineups for BTTS/corners model: {e}")

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

        home_form = pred.get("home_form", {})
        away_form = pred.get("away_form", {})

        # Get confirmed lineup data
        match_lineup = confirmed_lineups.get(match_key, {})
        home_lineup = match_lineup.get("home_lineup")
        away_lineup = match_lineup.get("away_lineup")

        # BTTS prediction (using ensemble xG when available)
        btts_pred = predict_btts(
            home_team, away_team, date, factors, home_form, away_form,
            ensemble_home_xg=pred.get("home_xg"),
            ensemble_away_xg=pred.get("away_xg"),
        )
        btts_preds.append(btts_pred)

        # Corners prediction (with lineup corner taker awareness)
        corners_pred = predict_corners(
            home_team, away_team, date, factors, home_form, away_form,
            home_lineup=home_lineup, away_lineup=away_lineup,
        )
        corners_preds.append(corners_pred)

        # Generate bets
        btts_bets = generate_btts_bets(btts_pred)
        corners_bets = generate_corners_bets(corners_pred)
        all_bets.extend(btts_bets)
        all_bets.extend(corners_bets)

    # Sort
    btts_preds.sort(key=lambda x: abs(x.btts_yes - SERIE_A_BTTS_RATE), reverse=True)
    corners_preds.sort(key=lambda x: x.expected_total_corners, reverse=True)

    log.info(f"Generated {len(btts_preds)} BTTS predictions")
    log.info(f"Generated {len(corners_preds)} corners predictions")
    log.info(f"Found {len(all_bets)} potential bets")

    return btts_preds, corners_preds, all_bets


def save_predictions(btts_preds: List, corners_preds: List, bets: List):
    """Save all predictions."""
    output_dir = DATA_DIR / "upcoming"

    # BTTS predictions
    btts_data = {
        "generated_at": datetime.now().isoformat(),
        "model": "btts_v1",
        "predictions": [
            {
                "match": p.match,
                "date": p.date,
                "btts_yes": p.btts_yes,
                "btts_no": p.btts_no,
                "home_scores": p.home_scores_prob,
                "away_scores": p.away_scores_prob,
                "confidence": p.confidence,
                "factors": p.factors
            }
            for p in btts_preds
        ]
    }
    with open(output_dir / "btts_predictions.json", "w") as f:
        json.dump(btts_data, f, indent=2)

    # Corners predictions
    corners_data = {
        "generated_at": datetime.now().isoformat(),
        "model": "corners_v1",
        "predictions": [
            {
                "match": p.match,
                "date": p.date,
                "expected_total": p.expected_total_corners,
                "home_corners": p.expected_home_corners,
                "away_corners": p.expected_away_corners,
                "over_9_5": p.over_9_5,
                "over_10_5": p.over_10_5,
                "confidence": p.confidence,
                "factors": p.factors
            }
            for p in corners_preds
        ]
    }
    with open(output_dir / "corners_predictions.json", "w") as f:
        json.dump(corners_data, f, indent=2)

    # Bets
    recommended = [b for b in bets if b.recommendation == "BET"]
    consider = [b for b in bets if b.recommendation == "CONSIDER"]

    bets_data = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "btts_bets": len([b for b in bets if b.market == "btts"]),
            "corners_bets": len([b for b in bets if b.market == "corners"]),
            "recommended": len(recommended),
            "consider": len(consider)
        },
        "recommended": [
            {
                "match": b.match,
                "date": b.date,
                "market": b.market,
                "bet": f"{b.bet_type.upper()}" + (f" {b.line}" if b.line else ""),
                "our_probability": b.our_probability,
                "fair_odds": b.fair_odds,
                "factors": b.factors
            }
            for b in recommended
        ],
        "consider": [
            {
                "match": b.match,
                "bet": f"{b.market} {b.bet_type.upper()}" + (f" {b.line}" if b.line else ""),
                "our_probability": b.our_probability,
                "fair_odds": b.fair_odds
            }
            for b in consider
        ]
    }
    with open(output_dir / "btts_corners_bets.json", "w") as f:
        json.dump(bets_data, f, indent=2)

    log.info("Saved BTTS and corners predictions")


def print_summary(btts_preds: List, corners_preds: List, bets: List):
    """Print summary."""

    print("\n" + "=" * 80)
    print(" BTTS PREDICTIONS")
    print("=" * 80)

    print(f"\n  {'Match':<30} {'BTTS Yes':>8} {'BTTS No':>8} {'H Score':>8} {'A Score':>8}")
    print("  " + "-" * 70)

    for p in btts_preds:
        print(f"  {p.match:<30} {p.btts_yes:>8.1%} {p.btts_no:>8.1%} "
              f"{p.home_scores_prob:>8.1%} {p.away_scores_prob:>8.1%}")

    print("\n" + "=" * 80)
    print(" CORNERS PREDICTIONS")
    print("=" * 80)

    print(f"\n  {'Match':<30} {'xCorners':>9} {'O9.5':>6} {'O10.5':>6} {'O11.5':>6}")
    print("  " + "-" * 65)

    for p in corners_preds:
        print(f"  {p.match:<30} {p.expected_total_corners:>9.1f} "
              f"{p.over_9_5:>6.1%} {p.over_10_5:>6.1%} {p.over_11_5:>6.1%}")

    # Bets
    recommended = [b for b in bets if b.recommendation == "BET"]

    if recommended:
        print("\n" + "=" * 80)
        print(" RECOMMENDED BETS")
        print("=" * 80)

        print(f"\n  {'Match':<25} {'Market':>8} {'Bet':>10} {'Our%':>7} {'Fair':>6}")
        print("  " + "-" * 60)

        for b in recommended:
            bet_str = b.bet_type.upper() + (f" {b.line}" if b.line else "")
            print(f"  {b.match:<25} {b.market:>8} {bet_str:>10} "
                  f"{b.our_probability:>7.1%} {b.fair_odds:>6.2f}")


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main entry point."""
    btts_preds, corners_preds, bets = generate_all_predictions()

    if btts_preds:
        save_predictions(btts_preds, corners_preds, bets)
        print_summary(btts_preds, corners_preds, bets)


if __name__ == "__main__":
    main()
