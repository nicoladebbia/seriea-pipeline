#!/usr/bin/env python3
"""UNIFIED PREDICTION SCRIPT - Consolidates All Prediction Modes

Single entry point for ALL Serie A match predictions:

Modes:
  - "ensemble" (default): Full 5-method ensemble with meta-learner + weighted averaging
  - "factor":  Factor-based system with FACTOR_LIFTS and STACKING_BONUSES
  - "xg":      xG Poisson model only
  - "ml":      ML classifier (CatBoost) only
  - "market":  Market-implied probabilities only

Usage:
    python scripts/predict_unified.py                          # Full ensemble
    python scripts/predict_unified.py --mode factor            # Factor-only (legacy)
    python scripts/predict_unified.py --mode xg                # xG Poisson only
    python scripts/predict_unified.py --match "Inter" "Milan"  # Single match
    python scripts/predict_unified.py --matchday 25            # Specific matchday
    python scripts/predict_unified.py --format json            # JSON output
    python scripts/predict_unified.py --brief                  # High confidence only
    python scripts/predict_unified.py --strategy selective     # Betting strategy
    python scripts/predict_unified.py --bankroll 500           # Set bankroll

Consolidates:
  - run_predictions.py (CLI dispatcher)
  - realtime_prediction_engine.py (factor-based core)
  - ensemble_prediction_engine.py (ensemble predictor, xG, ML, market, player_xg)
  - predict.py (value betting + bankroll management)
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import DATA_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# JSON ENCODER
# =============================================================================

class _NumpySafeEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.bool_, np.generic)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PredictionConfig:
    """All configurable parameters for the prediction pipeline."""

    # Prediction mode
    mode: str = "ensemble"  # ensemble, factor, xg, ml, market

    # Betting strategy (for calibration pipeline)
    strategy: str = "default"  # default, volume, selective, ml_heavy, xg_dominant

    # Bankroll management
    bankroll: float = 1000.0
    kelly_fraction: float = 0.10    # Synced with production (betting_unified.py)

    # Output control
    output_format: str = "text"  # text, json
    brief: bool = False
    verbose: bool = False

    # Feature toggles
    use_weather: bool = True
    use_referee: bool = True
    use_formation: bool = True
    use_injuries: bool = True
    use_value_betting: bool = False
    use_bankroll: bool = False

    # Single match / matchday filters
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    matchday: Optional[int] = None

    # Output path
    output_path: Optional[str] = None

    # Ensemble weights (override defaults if provided)
    custom_weights: Optional[Dict[str, float]] = None


# =============================================================================
# VALIDATED CONSTANTS (from realtime_prediction_engine.py - 21 seasons)
# =============================================================================

FACTOR_LIFTS = {
    "big_stadium": {"home_win": 0.15, "confidence": "HIGH", "seasons": 21},
    "home_favorite": {"home_win": 0.12, "confidence": "HIGH", "seasons": 21},
    "big_home_favorite": {"home_win": 0.20, "confidence": "HIGH", "seasons": 21},
    "hot_home": {"home_win": 0.08, "confidence": "HIGH", "seasons": 21},
    "cold_away": {"home_win": 0.06, "confidence": "HIGH", "seasons": 21},
    "derby": {"draw": 0.08, "cards": 0.10, "confidence": "HIGH", "seasons": 21},
    "home_fav_ref": {"home_win": 0.12, "confidence": "MEDIUM", "seasons": 8},
    "away_fav_ref": {"home_win": -0.10, "confidence": "MEDIUM", "seasons": 8},
    "strict_ref": {"high_cards": 0.15, "confidence": "MEDIUM", "seasons": 8},
    "rain": {"goals": 0.20, "home_win": 0.032, "confidence": "MEDIUM", "seasons": 8},
    "wind": {"goals": -0.22, "confidence": "MEDIUM", "seasons": 8},
    "cold": {"cards": 0.05, "confidence": "MEDIUM", "seasons": 8},
    "away_favorite": {"away_win": 0.10, "confidence": "HIGH", "seasons": 21},
    "big_away_favorite": {"away_win": 0.18, "confidence": "HIGH", "seasons": 21},
    "hot_away": {"away_win": 0.06, "confidence": "HIGH", "seasons": 21},
    "cold_home": {"away_win": 0.05, "confidence": "HIGH", "seasons": 21},
}

STACKING_BONUSES = {
    1: 0.044,
    2: 0.106,
    3: 0.204,
    4: 0.284,
    5: 0.515,
}

BASE_RATES = {
    "home_win": 0.444,
    "draw": 0.265,
    "away_win": 0.291,
    "over_25_goals": 0.486,
    "btts": 0.459,
    "over_45_cards": 0.473,
}

# Ensemble weights (Optuna-optimized, 3000 trials, market ≤15% normalized cap)
ENSEMBLE_WEIGHTS = {
    "factor": 0.035,
    "xg": 0.124,
    "ml": 0.605,
    "player_xg": 0.032,
    "market": 0.205,
}

ENSEMBLE_WEIGHTS_WITH_DEEP = {
    "factor": 0.035,
    "xg": 0.124,
    "ml": 0.605,
    "player_xg": 0.032,
    "deep": 0.00,
    "market": 0.205,
}

FALLBACK_WEIGHTS = {
    # ML fails: redistribute across factor+xg+pxg+market (0.396)
    "factor_xg_player_market": {"factor": 0.09, "xg": 0.31, "player_xg": 0.08, "market": 0.52},
    # ML + market fail: redistribute across factor+xg+pxg (0.191)
    "factor_xg_player": {"factor": 0.18, "xg": 0.65, "player_xg": 0.17},
    # player_xg fails: redistribute across factor+xg+ml+market (0.969)
    "factor_xg_ml_market": {"factor": 0.04, "xg": 0.13, "ml": 0.62, "market": 0.21},
    # ML+player+market fail: only factor+xg (0.159)
    "factor_xg": {"factor": 0.22, "xg": 0.78},
    # xG+player fail: factor+ml+market (0.845)
    "factor_ml_market": {"factor": 0.04, "ml": 0.72, "market": 0.24},
    # xG+player+market fail: factor+ml only (0.640)
    "factor_ml": {"factor": 0.05, "ml": 0.95},
    # xG+ML+player fail: factor+market (0.240)
    "factor_market": {"factor": 0.15, "market": 0.85},
    # Everything except factor fails
    "factor_only": {"factor": 1.0},
}

HOME_FAV_REFS = ["Federico Dionisi", "Antonio Giua", "Simone Sozza", "Valerio Marini"]
AWAY_FAV_REFS = ["Paolo Mazzoleni", "Daniele Doveri", "Marco Di Bello", "Pairetto"]
STRICT_REFS = ["Fabio Maresca", "Daniele Orsato", "Gianluca Manganiello", "Michael Fabbri"]


# =============================================================================
# DATA LOADING
# =============================================================================

def _load_factor_multipliers() -> Dict[str, float]:
    """Load factor lift multipliers from feedback loop."""
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


_FACTOR_MULTIPLIERS = _load_factor_multipliers()


def load_upcoming_matches() -> List[Dict]:
    """Load upcoming matches from all available sources."""
    matches = []

    manual_path = DATA_DIR / "upcoming" / "manual_matches.json"
    if manual_path.exists():
        with open(manual_path) as f:
            data = json.load(f)
            matches.extend(data.get("matches", []))
            log.info(f"Loaded {len(matches)} matches from manual file")

    scraped_path = DATA_DIR / "upcoming" / "matches.json"
    if scraped_path.exists():
        with open(scraped_path) as f:
            data = json.load(f)
            existing = {f"{m['home_team']}_{m['away_team']}" for m in matches}
            for m in data.get("matches", []):
                key = f"{m['home_team']}_{m['away_team']}"
                if key not in existing:
                    matches.append(m)

    return matches


def load_confirmed_lineups() -> Optional[Dict]:
    """Load confirmed lineups if available."""
    try:
        lineup_path = DATA_DIR / "upcoming" / "confirmed_lineups.json"
        if lineup_path.exists():
            with open(lineup_path) as f:
                lineup_data = json.load(f)
            confirmed = lineup_data.get("matches", {})
            if confirmed:
                log.info(f"Loaded confirmed lineups for {len(confirmed)} match(es)")
                return confirmed
    except Exception as e:
        log.debug(f"No confirmed lineups available: {e}")
    return None


def load_injury_map() -> Dict[str, list]:
    """Load injury data for xG adjustments."""
    injury_map: Dict[str, list] = {}
    try:
        from scraper.injuries import get_current_injuries
        injuries_df = get_current_injuries()
        if not injuries_df.empty and "team" in injuries_df.columns:
            for _, row in injuries_df.iterrows():
                team = row.get("team", "")
                player = row.get("player_name", "")
                if team and player:
                    injury_map.setdefault(team, []).append(player)
            log.info(f"Loaded injuries for {len(injury_map)} teams "
                     f"({sum(len(v) for v in injury_map.values())} total)")
    except Exception as e:
        log.debug(f"No injury data available: {e}")
    return injury_map


# =============================================================================
# FACTOR ANALYSIS (from realtime_prediction_engine.py)
# =============================================================================

def identify_all_factors(
    match: Dict, form_data: Dict, weather_data: Dict, referee_data: Dict = None
) -> Dict:
    """Identify all applicable factors for a match."""
    home = match["home_team"]
    away = match["away_team"]
    key = f"{home} vs {away}"

    matchup = form_data.get("matchups", {}).get(key, {})
    factors_from_form = matchup.get("factors", [])

    weather = weather_data.get(key, {})
    weather_impact = weather.get("impact", {})

    home_factors = []
    away_factors = []
    neutral_factors = []

    for f in factors_from_form:
        if f in ["big_stadium", "home_favorite", "big_home_favorite", "hot_home", "cold_away"]:
            home_factors.append(f)
        elif f in ["away_favorite", "big_away_favorite", "hot_away", "cold_home"]:
            away_factors.append(f)
        elif f == "derby":
            neutral_factors.append(f)

    weather_factors = weather_impact.get("factors", [])
    for f in weather_factors:
        neutral_factors.append(f)

    if referee_data:
        ref_info = referee_data.get(key, {})
        ref_factors = ref_info.get("factors", [])
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
        referee = match.get("referee", "")
        if referee in HOME_FAV_REFS:
            home_factors.append("home_fav_ref")
        elif referee in AWAY_FAV_REFS:
            away_factors.append("away_fav_ref")
        if referee in STRICT_REFS:
            neutral_factors.append("strict_ref")

    # Calculate lifts with feedback multipliers
    home_lift = sum(
        FACTOR_LIFTS.get(f, {}).get("home_win", 0) * _FACTOR_MULTIPLIERS.get(f, 1.0)
        for f in home_factors
    )
    away_lift = sum(
        FACTOR_LIFTS.get(f, {}).get("away_win", 0) * _FACTOR_MULTIPLIERS.get(f, 1.0)
        for f in away_factors
    )

    n_home = len(home_factors)
    n_away = len(away_factors)

    # Stacking bonuses
    for n, lift_ref in [(n_home, "home"), (n_away, "away")]:
        bonus = 0
        if n >= 5:
            bonus = STACKING_BONUSES[5]
        elif n >= 4:
            bonus = STACKING_BONUSES[4]
        elif n >= 3:
            bonus = STACKING_BONUSES[3]
        elif n >= 2:
            bonus = STACKING_BONUSES[2]
        elif n >= 1:
            bonus = STACKING_BONUSES[1]
        if lift_ref == "home":
            home_lift += bonus
        else:
            away_lift += bonus

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


# =============================================================================
# FACTOR-BASED PREDICTION (standalone, from realtime_prediction_engine.py)
# =============================================================================

def factor_predict(
    match: Dict, factors: Dict, form_data: Dict, market_probs: Dict = None
) -> Dict:
    """Generate a factor-based prediction for a match.

    When market_probs are available, anchors on market-implied probabilities
    and applies dampened factor lifts.
    """
    home = match["home_team"]
    away = match["away_team"]
    key = f"{home} vs {away}"

    DAMPEN = 0.35 if market_probs else 1.0

    if market_probs:
        home_prob = market_probs.get("prob_H", BASE_RATES["home_win"])
        draw_prob = market_probs.get("prob_D", BASE_RATES["draw"])
        away_prob = market_probs.get("prob_A", BASE_RATES["away_win"])
    else:
        home_prob = BASE_RATES["home_win"]
        draw_prob = BASE_RATES["draw"]
        away_prob = BASE_RATES["away_win"]

    home_prob += factors["home_lift"] * DAMPEN
    away_prob += factors["away_lift"] * DAMPEN

    if "derby" in factors["neutral_factors"]:
        draw_prob += 0.08 * DAMPEN
        home_prob -= 0.04 * DAMPEN
        away_prob -= 0.04 * DAMPEN

    home_prob = max(0.03, home_prob)
    draw_prob = max(0.03, draw_prob)
    away_prob = max(0.03, away_prob)
    total = home_prob + draw_prob + away_prob
    home_prob /= total
    draw_prob /= total
    away_prob /= total

    if home_prob >= draw_prob and home_prob >= away_prob:
        predicted_outcome = "HOME"
    elif away_prob >= draw_prob:
        predicted_outcome = "AWAY"
    else:
        predicted_outcome = "DRAW"

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

    matchup = form_data.get("matchups", {}).get(key, {})
    home_form = matchup.get("home_form", {})
    away_form = matchup.get("away_form", {})

    expected_goals = 2.67 + factors.get("weather", {}).get("goals_adj", 0)

    expected_cards = 4.45
    if "strict_ref" in factors["neutral_factors"]:
        expected_cards += 1.0
    if "derby" in factors["neutral_factors"]:
        expected_cards += 0.5
    if "cold" in factors["neutral_factors"]:
        expected_cards += 0.2

    return {
        "match": f"{home} vs {away}",
        "home_team": home,
        "away_team": away,
        "date": match.get("date", "TBD"),
        "time": match.get("time", "TBD"),
        "commence_time": match.get("commence_time", ""),
        "venue": match.get("venue", f"{home} Stadium"),
        "predicted_outcome": predicted_outcome,
        "probabilities": {
            "home": round(home_prob, 3),
            "draw": round(draw_prob, 3),
            "away": round(away_prob, 3),
        },
        "confidence": max(home_prob, draw_prob, away_prob),
        "confidence_level": confidence,
        "expected_accuracy": expected_accuracy,
        "n_factors": n_factors,
        "home_factors": factors["home_factors"],
        "away_factors": factors["away_factors"],
        "neutral_factors": factors["neutral_factors"],
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
        "expected_goals": round(expected_goals, 2),
        "over_25": expected_goals > 2.5,
        "expected_cards": round(expected_cards, 1),
        "methods_used": ["factor"],
        "betting_recommendation": _get_betting_recommendation(
            predicted_outcome, home_prob, draw_prob, away_prob, confidence
        ),
    }


# =============================================================================
# BETTING RECOMMENDATION
# =============================================================================

def _get_betting_recommendation(
    outcome: str, h_prob: float, d_prob: float, a_prob: float, confidence: str
) -> str:
    """Generate betting recommendation based on prediction."""
    if confidence in ["VERY HIGH", "HIGH"]:
        if outcome == "HOME" and h_prob > 0.60:
            return "STRONG BET: Home Win"
        elif outcome == "AWAY" and a_prob > 0.50:
            return "STRONG BET: Away Win"
        elif outcome == "HOME" and h_prob > 0.50:
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


# =============================================================================
# CONFIDENCE LEVELS
# =============================================================================

def _compute_confidence_level(max_prob: float, n_factors: int = 0) -> str:
    """Compute confidence level from max probability + optional factor bonus."""
    factor_bonus = max(0, min(0.04, (n_factors - 3) * 0.02))
    adjusted = max_prob + factor_bonus
    if adjusted >= 0.58:
        return "VERY HIGH"
    elif adjusted >= 0.48:
        return "HIGH"
    elif adjusted >= 0.40:
        return "MEDIUM-HIGH"
    elif adjusted >= 0.33:
        return "MEDIUM"
    else:
        return "LOW"


# =============================================================================
# UNIFIED PREDICTOR
# =============================================================================

class UnifiedPredictor:
    """Unified prediction system supporting all prediction modes.

    Modes:
        ensemble: Full 5-method ensemble (factor + xG + ML + player_xg + market)
        factor:   Factor-based system with validated lifts and stacking bonuses
        xg:       xG Poisson model only
        ml:       ML classifier (CatBoost) only
        market:   Market-implied probabilities only
    """

    def __init__(self, config: PredictionConfig = None):
        self.config = config or PredictionConfig()
        self.initialized = False

        # Sub-predictors (lazy-loaded from ensemble engine)
        self._ensemble_predictor = None  # EnsemblePredictor instance
        self._xg_predictor = None
        self._ml_classifier = None

        # Shared data
        self._form_data = None
        self._weather_data = None
        self._referee_data = None
        self._confirmed_lineups = None
        self._injury_map = None
        self._market_odds = {}

        # Available methods
        self.available_methods: List[str] = []

        # Value betting (optional)
        self._value_pipeline = None
        self._bankroll_manager = None

    def initialize(self) -> bool:
        """Initialize prediction components based on selected mode."""
        mode = self.config.mode
        log.info(f"Initializing UnifiedPredictor in '{mode}' mode...")

        if mode == "ensemble":
            return self._init_ensemble()
        elif mode == "factor":
            self.available_methods = ["factor"]
            self.initialized = True
            return True
        elif mode == "xg":
            return self._init_xg_only()
        elif mode == "ml":
            return self._init_ml_only()
        elif mode == "market":
            return self._init_market_only()
        else:
            log.error(f"Unknown mode: {mode}")
            return False

    def _init_ensemble(self) -> bool:
        """Initialize full ensemble predictor."""
        try:
            from scripts.prediction.ensemble_prediction_engine import EnsemblePredictor
            self._ensemble_predictor = EnsemblePredictor(
                weights=self.config.custom_weights,
                strategy=self.config.strategy,
            )
            if not self._ensemble_predictor.initialize():
                log.warning("Ensemble init failed, falling back to factor-only")
                self.available_methods = ["factor"]
                self.initialized = True
                return True
            self.available_methods = list(self._ensemble_predictor.available_methods)
            self.initialized = True
            log.info(f"Ensemble initialized with methods: {self.available_methods}")
            return True
        except Exception as e:
            log.error(f"Ensemble initialization failed: {e}")
            log.info("Falling back to factor-only mode")
            self.available_methods = ["factor"]
            self.initialized = True
            return True

    def _init_xg_only(self) -> bool:
        """Initialize xG predictor only."""
        try:
            from scripts.prediction.ensemble_prediction_engine import XGPredictor, FeatureBuilder
            self._xg_predictor = XGPredictor()
            if not self._xg_predictor.load_models():
                log.error("xG models not found")
                return False
            # Need feature builder for xG
            self._feature_builder = FeatureBuilder()
            self._feature_builder.load_historical()
            self.available_methods = ["xg"]
            self.initialized = True
            return True
        except Exception as e:
            log.error(f"xG initialization failed: {e}")
            return False

    def _init_ml_only(self) -> bool:
        """Initialize ML classifier only."""
        try:
            from scripts.prediction.ensemble_prediction_engine import MLClassifier, FeatureBuilder
            self._ml_classifier = MLClassifier()
            if not self._ml_classifier.load_model():
                log.error("ML classifier not found")
                return False
            self._feature_builder = FeatureBuilder()
            self._feature_builder.load_historical()
            self.available_methods = ["ml"]
            self.initialized = True
            return True
        except Exception as e:
            log.error(f"ML initialization failed: {e}")
            return False

    def _init_market_only(self) -> bool:
        """Initialize market-implied probabilities only."""
        self._market_odds = self._load_market_odds()
        if not self._market_odds:
            log.error("No market odds available")
            return False
        self.available_methods = ["market"]
        self.initialized = True
        return True

    def _init_value_betting(self):
        """Initialize value betting and bankroll management (lazy)."""
        if not self.config.use_value_betting:
            return
        try:
            from features.value_betting import ValueBettingPipeline
            self._value_pipeline = ValueBettingPipeline(
                kelly_fraction=self.config.kelly_fraction,
                value_mode="standard",
            )
        except ImportError:
            log.debug("Value betting module not available")

        if not self.config.use_bankroll:
            return
        try:
            from features.bankroll_manager import BankrollManager
            self._bankroll_manager = BankrollManager(
                initial_bankroll=self.config.bankroll
            )
            self._bankroll_manager.load_state()
        except ImportError:
            log.debug("Bankroll manager not available")

    # -----------------------------------------------------------------
    # Shared data loading
    # -----------------------------------------------------------------

    def _load_shared_data(self):
        """Load form, weather, referee data (shared across all modes)."""
        if self._form_data is not None:
            return  # already loaded

        log.info("[1/4] Calculating current team form...")
        try:
            from scripts.prediction.current_form_calculator import calculate_all_forms
            self._form_data = calculate_all_forms()
        except Exception as e:
            log.warning(f"Form calculation failed: {e}")
            self._form_data = {"matchups": {}}

        if self.config.use_weather:
            log.info("[2/4] Fetching weather forecasts...")
            try:
                from scripts.prediction.weather_integration import fetch_all_match_weather
                self._weather_data = fetch_all_match_weather()
            except Exception as e:
                log.warning(f"Weather fetch failed: {e}")
                self._weather_data = {}
        else:
            self._weather_data = {}

        if self.config.use_referee:
            log.info("[3/4] Analyzing referee assignments...")
            try:
                from scripts.prediction.referee_integration import analyze_referee_impact
                matches = load_upcoming_matches()
                self._referee_data = analyze_referee_impact(matches)
            except Exception as e:
                log.warning(f"Referee analysis failed: {e}")
                self._referee_data = {}
        else:
            self._referee_data = {}

        if self.config.use_injuries:
            self._injury_map = load_injury_map()
        else:
            self._injury_map = {}

        self._confirmed_lineups = load_confirmed_lineups()

    def _load_market_odds(self) -> Dict[str, Dict]:
        """Load bookmaker odds and convert to vig-free implied probabilities."""
        market_probs = {}

        # Sharp consensus first
        ba_path = DATA_DIR / "upcoming" / "bookmaker_analysis.json"
        if ba_path.exists():
            try:
                with open(ba_path) as f:
                    ba_data = json.load(f)
                for match_key, analysis in ba_data.get("matches", {}).items():
                    sharp = analysis.get("sharp_consensus")
                    if sharp and sharp.get("prob_H", 0) > 0:
                        market_probs[match_key] = {
                            "prob_H": sharp["prob_H"],
                            "prob_D": sharp["prob_D"],
                            "prob_A": sharp["prob_A"],
                            "overround": 2.0,
                            "source": "sharp_consensus",
                        }
            except Exception as e:
                log.debug(f"Could not load bookmaker analysis: {e}")

        # Fallback to odds.json
        odds_path = DATA_DIR / "upcoming" / "odds.json"
        if odds_path.exists():
            try:
                with open(odds_path) as f:
                    raw_odds = json.load(f)
                for match_key, odds in raw_odds.items():
                    if match_key in market_probs:
                        continue
                    h = odds.get("home", 0)
                    d = odds.get("draw", 0)
                    a = odds.get("away", 0)
                    if h <= 1.0 or d <= 1.0 or a <= 1.0:
                        continue
                    raw_h, raw_d, raw_a = 1.0 / h, 1.0 / d, 1.0 / a
                    total = raw_h + raw_d + raw_a
                    if total <= 0:
                        continue
                    market_probs[match_key] = {
                        "prob_H": raw_h / total,
                        "prob_D": raw_d / total,
                        "prob_A": raw_a / total,
                        "overround": round((total - 1.0) * 100, 1),
                        "source": "odds_average",
                    }
            except Exception as e:
                log.warning(f"Could not load odds.json: {e}")

        return market_probs

    # -----------------------------------------------------------------
    # PREDICT SINGLE MATCH
    # -----------------------------------------------------------------

    def predict_single(
        self,
        home_team: str,
        away_team: str,
        match: Dict = None,
        odds: Dict = None,
    ) -> Dict:
        """Generate prediction for a single match.

        Args:
            home_team: Home team name
            away_team: Away team name
            match: Optional match dict with date, time, venue, referee
            odds: Optional odds dict {home, draw, away} for value detection
        """
        if not self.initialized:
            if not self.initialize():
                return {"error": "Initialization failed", "match": f"{home_team} vs {away_team}"}

        self._load_shared_data()

        if match is None:
            match = {
                "home_team": home_team,
                "away_team": away_team,
                "date": datetime.now().strftime("%Y-%m-%d"),
            }

        mode = self.config.mode

        # Get factors (needed for factor and ensemble modes)
        factors = identify_all_factors(
            match, self._form_data, self._weather_data, self._referee_data
        )

        if mode == "ensemble":
            pred = self._predict_ensemble(match, factors)
        elif mode == "factor":
            market_probs = self._get_market_probs_for_match(home_team, away_team)
            pred = factor_predict(match, factors, self._form_data, market_probs)
        elif mode == "xg":
            pred = self._predict_xg_only(match, factors)
        elif mode == "ml":
            pred = self._predict_ml_only(match, factors)
        elif mode == "market":
            pred = self._predict_market_only(match, factors)
        else:
            return {"error": f"Unknown mode: {mode}"}

        # Add value betting analysis if requested
        if self.config.use_value_betting and odds and pred.get("probabilities"):
            pred["value_analysis"] = self._analyze_value(pred, odds)

        return pred

    def _get_market_probs_for_match(self, home: str, away: str) -> Optional[Dict]:
        """Get market-implied probabilities for a match."""
        if not self._market_odds:
            self._market_odds = self._load_market_odds()
        match_key = f"{home} vs {away}"
        probs = self._market_odds.get(match_key)
        if probs:
            return {"prob_H": probs["prob_H"], "prob_D": probs["prob_D"], "prob_A": probs["prob_A"]}
        return None

    def _predict_ensemble(self, match: Dict, factors: Dict) -> Dict:
        """Generate ensemble prediction using the EnsemblePredictor."""
        home = match["home_team"]
        away = match["away_team"]
        key = f"{home} vs {away}"

        pred = self._ensemble_predictor.predict(
            match, factors, self._form_data,
            confirmed_lineups=self._confirmed_lineups,
        )

        # Enrich with standard fields
        pred["date"] = match.get("date", "TBD")
        pred["time"] = match.get("time", "TBD")
        pred["commence_time"] = match.get("commence_time", "")
        pred["venue"] = match.get("venue", f"{home} Stadium")

        max_prob = pred["confidence"]
        n_factors = factors["n_home_factors"] + factors["n_away_factors"]
        pred["confidence_level"] = _compute_confidence_level(max_prob, n_factors)

        pred["home_factors"] = factors["home_factors"]
        pred["away_factors"] = factors["away_factors"]
        pred["neutral_factors"] = factors["neutral_factors"]
        pred["n_factors"] = n_factors

        matchup = self._form_data.get("matchups", {}).get(key, {})
        pred["home_form"] = matchup.get("home_form", {})
        pred["away_form"] = matchup.get("away_form", {})

        ref_info = (self._referee_data or {}).get(key, {})
        if ref_info.get("referee"):
            pred["referee"] = ref_info["referee"]
            pred["referee_bias"] = ref_info.get("classification", {}).get("bias_type", "unknown")

        # xG details
        xg_details = pred.get("component_predictions", {}).get("xg_details", {})
        if xg_details:
            pred["expected_goals"] = round(xg_details["home_xg"] + xg_details["away_xg"], 2)
            pred["home_xg"] = round(xg_details["home_xg"], 2)
            pred["away_xg"] = round(xg_details["away_xg"], 2)
        else:
            pred["expected_goals"] = 2.67

        # Injury adjustments
        self._apply_injury_adjustments(pred, home, away)

        pred["over_25"] = bool(pred.get("expected_goals", 2.67) > 2.5)

        pred["betting_recommendation"] = _get_betting_recommendation(
            pred["predicted_outcome"],
            pred["probabilities"]["home"],
            pred["probabilities"]["draw"],
            pred["probabilities"]["away"],
            pred["confidence_level"],
        )

        return pred

    def _predict_xg_only(self, match: Dict, factors: Dict) -> Dict:
        """Generate xG-only prediction."""
        home = match["home_team"]
        away = match["away_team"]

        match_features = self._feature_builder.build_match_features(
            home, away, self._form_data
        )
        if match_features is None:
            return self._error_prediction(match, "Feature building failed")

        xg_result = self._xg_predictor.predict(match_features)
        if not xg_result:
            return self._error_prediction(match, "xG prediction failed")

        home_prob = xg_result["prob_H"]
        draw_prob = xg_result["prob_D"]
        away_prob = xg_result["prob_A"]

        if home_prob >= draw_prob and home_prob >= away_prob:
            predicted = "HOME"
        elif away_prob >= draw_prob:
            predicted = "AWAY"
        else:
            predicted = "DRAW"

        max_prob = max(home_prob, draw_prob, away_prob)
        n_factors = factors["n_home_factors"] + factors["n_away_factors"]

        return {
            "match": f"{home} vs {away}",
            "home_team": home,
            "away_team": away,
            "date": match.get("date", "TBD"),
            "time": match.get("time", "TBD"),
            "venue": match.get("venue", f"{home} Stadium"),
            "predicted_outcome": predicted,
            "probabilities": {
                "home": round(home_prob, 3),
                "draw": round(draw_prob, 3),
                "away": round(away_prob, 3),
            },
            "confidence": max_prob,
            "confidence_level": _compute_confidence_level(max_prob, n_factors),
            "home_xg": round(xg_result["home_xg"], 2),
            "away_xg": round(xg_result["away_xg"], 2),
            "expected_goals": round(xg_result["home_xg"] + xg_result["away_xg"], 2),
            "over_25": xg_result["home_xg"] + xg_result["away_xg"] > 2.5,
            "methods_used": ["xg"],
            "home_factors": factors["home_factors"],
            "away_factors": factors["away_factors"],
            "neutral_factors": factors["neutral_factors"],
            "n_factors": n_factors,
            "betting_recommendation": _get_betting_recommendation(
                predicted, home_prob, draw_prob, away_prob,
                _compute_confidence_level(max_prob, n_factors),
            ),
        }

    def _predict_ml_only(self, match: Dict, factors: Dict) -> Dict:
        """Generate ML-only prediction."""
        home = match["home_team"]
        away = match["away_team"]

        match_features = self._feature_builder.build_match_features(
            home, away, self._form_data
        )
        if match_features is None:
            return self._error_prediction(match, "Feature building failed")

        ml_result = self._ml_classifier.predict(match_features)
        if not ml_result:
            return self._error_prediction(match, "ML prediction failed")

        home_prob = ml_result["prob_H"]
        draw_prob = ml_result["prob_D"]
        away_prob = ml_result["prob_A"]

        if home_prob >= draw_prob and home_prob >= away_prob:
            predicted = "HOME"
        elif away_prob >= draw_prob:
            predicted = "AWAY"
        else:
            predicted = "DRAW"

        max_prob = max(home_prob, draw_prob, away_prob)
        n_factors = factors["n_home_factors"] + factors["n_away_factors"]

        return {
            "match": f"{home} vs {away}",
            "home_team": home,
            "away_team": away,
            "date": match.get("date", "TBD"),
            "time": match.get("time", "TBD"),
            "venue": match.get("venue", f"{home} Stadium"),
            "predicted_outcome": predicted,
            "probabilities": {
                "home": round(home_prob, 3),
                "draw": round(draw_prob, 3),
                "away": round(away_prob, 3),
            },
            "confidence": max_prob,
            "confidence_level": _compute_confidence_level(max_prob, n_factors),
            "methods_used": ["ml"],
            "home_factors": factors["home_factors"],
            "away_factors": factors["away_factors"],
            "neutral_factors": factors["neutral_factors"],
            "n_factors": n_factors,
            "betting_recommendation": _get_betting_recommendation(
                predicted, home_prob, draw_prob, away_prob,
                _compute_confidence_level(max_prob, n_factors),
            ),
        }

    def _predict_market_only(self, match: Dict, factors: Dict) -> Dict:
        """Generate market-implied prediction."""
        home = match["home_team"]
        away = match["away_team"]
        match_key = f"{home} vs {away}"

        probs = self._market_odds.get(match_key)
        if not probs:
            return self._error_prediction(match, "No market odds available")

        home_prob = probs["prob_H"]
        draw_prob = probs["prob_D"]
        away_prob = probs["prob_A"]

        if home_prob >= draw_prob and home_prob >= away_prob:
            predicted = "HOME"
        elif away_prob >= draw_prob:
            predicted = "AWAY"
        else:
            predicted = "DRAW"

        max_prob = max(home_prob, draw_prob, away_prob)
        n_factors = factors["n_home_factors"] + factors["n_away_factors"]

        return {
            "match": f"{home} vs {away}",
            "home_team": home,
            "away_team": away,
            "date": match.get("date", "TBD"),
            "time": match.get("time", "TBD"),
            "venue": match.get("venue", f"{home} Stadium"),
            "predicted_outcome": predicted,
            "probabilities": {
                "home": round(home_prob, 3),
                "draw": round(draw_prob, 3),
                "away": round(away_prob, 3),
            },
            "confidence": max_prob,
            "confidence_level": _compute_confidence_level(max_prob, n_factors),
            "methods_used": ["market"],
            "market_source": probs.get("source", "unknown"),
            "overround": probs.get("overround", 0),
            "home_factors": factors["home_factors"],
            "away_factors": factors["away_factors"],
            "neutral_factors": factors["neutral_factors"],
            "n_factors": n_factors,
            "betting_recommendation": _get_betting_recommendation(
                predicted, home_prob, draw_prob, away_prob,
                _compute_confidence_level(max_prob, n_factors),
            ),
        }

    def _apply_injury_adjustments(self, pred: Dict, home: str, away: str):
        """Apply injury-based xG adjustments to a prediction."""
        if not self._injury_map:
            return

        home_injured = self._injury_map.get(home, [])
        away_injured = self._injury_map.get(away, [])
        if not home_injured and not away_injured:
            return

        try:
            from features.injury_impact import compute_injury_xg_adjustment
            if home_injured:
                h_adj = compute_injury_xg_adjustment(home, away, home_injured)
                pred["home_xg"] = round(
                    pred.get("home_xg", 1.3) + h_adj["team_xg_adj"], 2
                )
                pred["away_xg"] = round(
                    pred.get("away_xg", 1.3) + h_adj["opponent_xg_adj"], 2
                )
            if away_injured:
                a_adj = compute_injury_xg_adjustment(away, home, away_injured)
                pred["away_xg"] = round(
                    pred.get("away_xg", 1.3) + a_adj["team_xg_adj"], 2
                )
                pred["home_xg"] = round(
                    pred.get("home_xg", 1.3) + a_adj["opponent_xg_adj"], 2
                )
            pred["home_xg"] = max(0.1, pred.get("home_xg", 1.3))
            pred["away_xg"] = max(0.1, pred.get("away_xg", 1.3))
            pred["expected_goals"] = round(pred["home_xg"] + pred["away_xg"], 2)
            pred["injury_adjustments"] = {
                "home_injured": home_injured,
                "away_injured": away_injured,
            }
        except Exception as e:
            log.debug(f"Injury xG adjustment failed: {e}")

    def _analyze_value(self, pred: Dict, odds: Dict) -> Dict:
        """Run value betting analysis on a prediction."""
        if not self._value_pipeline:
            self._init_value_betting()
        if not self._value_pipeline:
            return {"has_value": False, "error": "Value betting not available"}

        probs = {
            "home": pred["probabilities"]["home"],
            "draw": pred["probabilities"]["draw"],
            "away": pred["probabilities"]["away"],
        }
        bankroll = self._bankroll_manager.current_bankroll if self._bankroll_manager else 1000.0

        try:
            return self._value_pipeline.analyze_bet(
                pred.get("home_team", ""), pred.get("away_team", ""),
                probs, odds, bankroll=bankroll,
            )
        except Exception as e:
            log.debug(f"Value analysis failed: {e}")
            return {"has_value": False, "error": str(e)}

    @staticmethod
    def _error_prediction(match: Dict, error_msg: str) -> Dict:
        """Return an error prediction dict."""
        home = match.get("home_team", "?")
        away = match.get("away_team", "?")
        return {
            "match": f"{home} vs {away}",
            "home_team": home,
            "away_team": away,
            "date": match.get("date", "TBD"),
            "predicted_outcome": "UNKNOWN",
            "probabilities": {"home": 0.333, "draw": 0.334, "away": 0.333},
            "confidence": 0.334,
            "confidence_level": "LOW",
            "methods_used": [],
            "error": error_msg,
            "betting_recommendation": "SKIP: Prediction error",
        }

    # -----------------------------------------------------------------
    # PREDICT UPCOMING (all matches)
    # -----------------------------------------------------------------

    def predict_upcoming(self) -> Dict:
        """Predict all upcoming matches.

        Returns output dict in the same format as ensemble_prediction_engine.
        """
        if not self.initialized:
            if not self.initialize():
                return {"error": "Initialization failed"}

        self._load_shared_data()

        matches = load_upcoming_matches()
        if not matches:
            log.error("No upcoming matches found!")
            return {}
        log.info(f"Found {len(matches)} upcoming matches")

        # Load lessons for ensemble mode (batch optimization)
        if self.config.mode == "ensemble" and self._ensemble_predictor:
            self._ensemble_predictor._load_lessons()

        # Load odds for value betting
        odds_data = self._load_odds_for_value()

        predictions = []
        for match in matches:
            home = match["home_team"]
            away = match["away_team"]
            key = f"{home} vs {away}"

            odds = odds_data.get(key) if odds_data else None
            pred = self.predict_single(home, away, match=match, odds=odds)
            predictions.append(pred)

        # Flush lessons
        if self.config.mode == "ensemble" and self._ensemble_predictor:
            self._ensemble_predictor._flush_lessons()

        # Sort by confidence
        confidence_order = {"VERY HIGH": 0, "HIGH": 1, "MEDIUM-HIGH": 2, "MEDIUM": 3, "LOW": 4}
        predictions.sort(key=lambda x: (
            confidence_order.get(x.get("confidence_level", "MEDIUM"), 5),
            -x["probabilities"]["home"]
        ))

        # Build output
        output = {
            "generated_at": datetime.now().isoformat(),
            "model_version": self._get_model_version(),
            "mode": self.config.mode,
            "strategy": self.config.strategy,
            "ensemble_enabled": self.config.mode == "ensemble",
            "methods_available": self.available_methods,
            "predictions": predictions,
            "summary": {
                "total_matches": len(predictions),
                "high_confidence": len([
                    p for p in predictions
                    if p.get("confidence_level") in ["VERY HIGH", "HIGH"]
                ]),
                "medium_confidence": len([
                    p for p in predictions
                    if p.get("confidence_level") == "MEDIUM-HIGH"
                ]),
            },
        }

        # Save
        output_path = self.config.output_path or str(
            DATA_DIR / "upcoming" / "predictions.json"
        )
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, cls=_NumpySafeEncoder)
        log.info(f"Saved predictions to {output_path}")

        return output

    # -----------------------------------------------------------------
    # PREDICT MATCHDAY
    # -----------------------------------------------------------------

    def predict_matchday(self, matchday: int = None) -> Dict:
        """Predict matches for a specific matchday.

        If matchday is None, predicts the next matchday from upcoming matches.
        """
        if not self.initialized:
            if not self.initialize():
                return {"error": "Initialization failed"}

        self._load_shared_data()

        matches = load_upcoming_matches()
        if not matches:
            return {}

        # Filter by matchday if specified
        if matchday is not None:
            matches = [
                m for m in matches
                if m.get("matchday") == matchday or m.get("round") == matchday
            ]
            if not matches:
                log.warning(f"No matches found for matchday {matchday}")
                # Fall back to all upcoming
                matches = load_upcoming_matches()

        # Load lessons for ensemble mode
        if self.config.mode == "ensemble" and self._ensemble_predictor:
            self._ensemble_predictor._load_lessons()

        predictions = []
        for match in matches:
            pred = self.predict_single(
                match["home_team"], match["away_team"], match=match
            )
            predictions.append(pred)

        if self.config.mode == "ensemble" and self._ensemble_predictor:
            self._ensemble_predictor._flush_lessons()

        confidence_order = {"VERY HIGH": 0, "HIGH": 1, "MEDIUM-HIGH": 2, "MEDIUM": 3, "LOW": 4}
        predictions.sort(key=lambda x: (
            confidence_order.get(x.get("confidence_level", "MEDIUM"), 5),
            -x["probabilities"]["home"]
        ))

        output = {
            "generated_at": datetime.now().isoformat(),
            "model_version": self._get_model_version(),
            "mode": self.config.mode,
            "matchday": matchday,
            "predictions": predictions,
            "summary": {
                "total_matches": len(predictions),
                "high_confidence": len([
                    p for p in predictions
                    if p.get("confidence_level") in ["VERY HIGH", "HIGH"]
                ]),
            },
        }

        return output

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _get_model_version(self) -> str:
        mode = self.config.mode
        if mode == "ensemble":
            return "v5.0-unified-ensemble"
        elif mode == "factor":
            return "v2.0-21seasons"
        elif mode == "xg":
            return "v5.0-xg-poisson"
        elif mode == "ml":
            return "v5.0-ml-catboost"
        elif mode == "market":
            return "v5.0-market-implied"
        return "v5.0-unified"

    def _load_odds_for_value(self) -> Dict:
        """Load odds data for value betting analysis."""
        if not self.config.use_value_betting:
            return {}

        odds_data = {}
        odds_full_path = DATA_DIR / "upcoming" / "odds_full.json"
        if odds_full_path.exists():
            try:
                with open(odds_full_path) as f:
                    full_data = json.load(f)
                for match_key, match_odds in full_data.get("matches", {}).items():
                    h2h = match_odds.get("h2h", {})
                    odds_data[match_key] = {
                        "home": h2h.get("best_home", h2h.get("home", 0)),
                        "draw": h2h.get("best_draw", h2h.get("draw", 0)),
                        "away": h2h.get("best_away", h2h.get("away", 0)),
                    }
            except Exception as e:
                log.warning(f"Failed to load odds_full.json: {e}")

        if not odds_data:
            odds_path = DATA_DIR / "upcoming" / "odds.json"
            if odds_path.exists():
                try:
                    with open(odds_path) as f:
                        odds_data = json.load(f)
                except Exception as e:
                    log.warning(f"Failed to load fallback odds.json: {e}")

        return odds_data


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def print_predictions(output: Dict, verbose: bool = False, brief: bool = False):
    """Print predictions in a readable format."""
    predictions = output.get("predictions", [])

    if brief:
        _print_brief(output, predictions)
        return

    print("\n" + "=" * 80)
    print("SERIE A PREDICTIONS")
    print("=" * 80)
    print(f"Generated: {output.get('generated_at', 'Unknown')}")
    print(f"Model: {output.get('model_version', 'Unknown')}")
    print(f"Mode: {output.get('mode', 'ensemble').upper()}")
    methods = output.get("methods_available", [])
    if methods:
        print(f"Methods: {', '.join(methods)}")

    # Group by confidence
    high_conf = [p for p in predictions
                 if p.get("confidence_level") in ["VERY HIGH", "HIGH"]]
    medium_conf = [p for p in predictions
                   if p.get("confidence_level") == "MEDIUM-HIGH"]
    low_conf = [p for p in predictions
                if p.get("confidence_level") in ["MEDIUM", "LOW"]]

    if high_conf:
        print("\n" + "=" * 80)
        print("HIGH CONFIDENCE PREDICTIONS")
        print("=" * 80)
        for pred in high_conf:
            _print_single(pred, verbose)

    if medium_conf:
        print("\n" + "=" * 80)
        print("MEDIUM-HIGH CONFIDENCE")
        print("=" * 80)
        for pred in medium_conf:
            _print_single(pred, verbose)

    if low_conf:
        print("\n" + "=" * 80)
        print("LOWER CONFIDENCE (Skip or small stakes)")
        print("=" * 80)
        for pred in low_conf:
            _print_single(pred, verbose=False)

    # Summary
    summary = output.get("summary", {})
    print("\n" + "=" * 80)
    print("SUMMARY")
    print(f"Total matches: {summary.get('total_matches', 0)}")
    print(f"High confidence: {summary.get('high_confidence', 0)}")
    print(f"Medium confidence: {summary.get('medium_confidence', 0)}")


def _print_brief(output: Dict, predictions: List[Dict]):
    """Print brief output -- high confidence only."""
    high_conf = [
        p for p in predictions
        if p.get("confidence_level") in ["VERY HIGH", "HIGH", "MEDIUM-HIGH"]
    ]

    print("\n" + "=" * 60)
    print("SERIE A PREDICTIONS - HIGH CONFIDENCE")
    mode = output.get("mode", "ensemble")
    methods = output.get("methods_available", [])
    print(f"Mode: {mode.upper()} ({', '.join(methods)})")
    print("=" * 60)

    for pred in high_conf:
        probs = pred["probabilities"]
        n_factors = pred.get("n_factors", 0)
        conf = pred.get("confidence_level", "MEDIUM")
        print(f"\n{pred['match']} ({pred.get('date', 'TBD')})")
        print(f"  -> {pred['predicted_outcome']} ({conf}, {n_factors} factors)")
        print(f"  H {probs['home']:.0%} | D {probs['draw']:.0%} | A {probs['away']:.0%}")

        if pred.get("home_xg") and pred.get("away_xg"):
            print(f"  xG: Home {pred['home_xg']:.2f} - Away {pred['away_xg']:.2f}")

        rec = pred.get("betting_recommendation", "")
        if rec:
            print(f"  {rec}")

    print(f"\n{len(high_conf)}/{len(predictions)} matches with MEDIUM-HIGH+ confidence")


def _print_single(pred: Dict, verbose: bool = False):
    """Print a single prediction."""
    print(f"\n{'---' * 20}")
    print(f"{pred['match']}")
    venue_info = f"  {pred.get('date', 'TBD')} {pred.get('time', '')} | {pred.get('venue', '')}"
    if pred.get("referee"):
        venue_info += f" | Ref: {pred['referee']}"
    print(venue_info)

    probs = pred["probabilities"]
    conf = pred.get("confidence_level", "MEDIUM")
    print(f"\n  PREDICTION: {pred['predicted_outcome']} ({conf})")
    print(f"  Probabilities: H {probs['home']:.1%} | D {probs['draw']:.1%} | A {probs['away']:.1%}")

    if verbose:
        # Component predictions
        components = pred.get("component_predictions", {})
        if components:
            print("\n  Component Predictions:")
            for method, comp_probs in components.items():
                if method == "xg_details":
                    print(f"    xG: Home {comp_probs['home_xg']:.2f} - Away {comp_probs['away_xg']:.2f}")
                elif method == "player_xg_details":
                    print(f"    Player xG: Home {comp_probs['home_lineup_xg']:.2f} - Away {comp_probs['away_lineup_xg']:.2f}")
                elif isinstance(comp_probs, dict) and "prob_H" in comp_probs:
                    print(f"    {method.upper()}: H {comp_probs['prob_H']:.1%} "
                          f"D {comp_probs['prob_D']:.1%} A {comp_probs['prob_A']:.1%}")

        # Methods used
        methods = pred.get("methods_used", [])
        if methods:
            print(f"\n  Methods: {', '.join(methods)}")

        # Weights applied
        weights = pred.get("weights_applied", {})
        if weights:
            w_str = ", ".join(f"{m}={w:.0%}" for m, w in weights.items())
            print(f"  Weights: {w_str}")

    # Factors
    if pred.get("home_factors"):
        print(f"\n  Home factors: {', '.join(pred['home_factors'])}")
    if pred.get("away_factors"):
        print(f"  Away factors: {', '.join(pred['away_factors'])}")
    if pred.get("neutral_factors"):
        print(f"  Neutral: {', '.join(pred['neutral_factors'])}")

    # xG
    if pred.get("home_xg") and pred.get("away_xg"):
        print(f"\n  xG: Home {pred['home_xg']:.2f} - Away {pred['away_xg']:.2f} "
              f"(Total: {pred.get('expected_goals', '?')})")

    # Market intelligence
    market = pred.get("market_intelligence", {})
    if market and verbose:
        print(f"\n  Market Intelligence:")
        print(f"    Sharp score: {market.get('sharp_score', 0):+.3f} | "
              f"Movement: {market.get('movement_magnitude', 0):.1f}%")
        if market.get("has_value"):
            best = market.get("best_bet", "home")
            edge = market.get(f"{best}_edge", 0)
            print(f"    VALUE DETECTED: {best.upper()} (edge: {edge:+.1%})")

    # Value analysis
    value = pred.get("value_analysis", {})
    if value and value.get("has_value"):
        print(f"\n  VALUE BET: {value.get('best_bet', '').upper()}")
        print(f"  Edge: {value.get('edge', 0):+.1%}")

    # Recommendation
    rec = pred.get("betting_recommendation", "")
    if "STRONG" in rec:
        print(f"\n  >>> {rec}")
    elif "BET" in rec or "LEAN" in rec:
        print(f"\n  >> {rec}")
    else:
        print(f"\n  {rec}")


# =============================================================================
# BACKWARD-COMPATIBLE WRAPPERS (for callers that used realtime_prediction_engine)
# =============================================================================

# Alias: old generate_prediction -> new factor_predict
generate_prediction = factor_predict


def run_predictions() -> Dict:
    """Backward-compatible wrapper — tries ensemble first, falls back to factor.

    IMPORTANT: This function writes to predictions.json. Using factor-only mode
    silently degrades prediction quality from 6-method blend to single method.
    Always attempt ensemble first.
    """
    # Try ensemble first (6-method blend: factor + xG + ML + player_xg + market + deep)
    try:
        from scripts.prediction.ensemble_prediction_engine import run_ensemble_predictions
        result = run_ensemble_predictions(use_ensemble=True)
        if result and result.get("predictions"):
            return result
        log.warning("Ensemble returned empty, falling back to factor-only")
    except Exception as e:
        log.warning(f"Ensemble not available ({e}), falling back to factor-only")

    # Fallback: factor-only (single method, degraded quality)
    config = PredictionConfig(mode="factor")
    predictor = UnifiedPredictor(config)
    if not predictor.initialize():
        log.error("Failed to initialize predictor for run_predictions()")
        return {}
    return predictor.predict_upcoming()


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified Serie A Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              Full ensemble prediction
  %(prog)s --mode factor                Factor-only (legacy)
  %(prog)s --mode xg                    xG Poisson model only
  %(prog)s --mode ml                    ML classifier only
  %(prog)s --mode market                Market-implied only
  %(prog)s --match Inter Milan          Single match
  %(prog)s --matchday 25                Matchday 25
  %(prog)s --brief                      High confidence only
  %(prog)s --format json                JSON output
  %(prog)s --strategy selective         Betting strategy
  %(prog)s --value --bankroll 500       Value betting with bankroll
        """
    )

    # Core options
    parser.add_argument("--mode", default="ensemble",
                        choices=["ensemble", "factor", "xg", "ml", "market"],
                        help="Prediction mode (default: ensemble)")
    parser.add_argument("--strategy", default="default",
                        choices=["default", "volume", "selective", "ml_heavy", "xg_dominant"],
                        help="Betting strategy (default: default)")

    # Match selection
    parser.add_argument("--match", nargs=2, metavar=("HOME", "AWAY"),
                        help="Predict a single match")
    parser.add_argument("--matchday", type=int, default=None,
                        help="Predict specific matchday")

    # Output
    parser.add_argument("--format", choices=["text", "json"], default="text",
                        help="Output format (default: text)")
    parser.add_argument("--brief", action="store_true",
                        help="Brief output (high confidence only)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output with component details")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: data/upcoming/predictions.json)")

    # Feature toggles
    parser.add_argument("--no-weather", action="store_true",
                        help="Skip weather fetch")
    parser.add_argument("--no-referee", action="store_true",
                        help="Skip referee analysis")
    parser.add_argument("--no-injuries", action="store_true",
                        help="Skip injury data")

    # Value betting
    parser.add_argument("--value", action="store_true",
                        help="Enable value betting analysis")
    parser.add_argument("--bankroll", type=float, default=1000.0,
                        help="Initial bankroll (default: 1000)")
    parser.add_argument("--kelly", type=float, default=0.15,
                        help="Kelly fraction (default: 0.15)")

    # Legacy compatibility
    parser.add_argument("--no-ensemble", action="store_true",
                        help="Use factor-only predictions (legacy, same as --mode factor)")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON only (legacy, same as --format json)")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Handle legacy flags
    if args.no_ensemble:
        args.mode = "factor"
    if args.json:
        args.format = "json"

    # Build config
    config = PredictionConfig(
        mode=args.mode,
        strategy=args.strategy,
        bankroll=args.bankroll,
        kelly_fraction=args.kelly,
        output_format=args.format,
        brief=args.brief,
        verbose=args.verbose,
        use_weather=not args.no_weather,
        use_referee=not args.no_referee,
        use_injuries=not args.no_injuries,
        use_value_betting=args.value,
        use_bankroll=args.value,
        home_team=args.match[0] if args.match else None,
        away_team=args.match[1] if args.match else None,
        matchday=args.matchday,
        output_path=args.output,
    )

    # Initialize predictor
    predictor = UnifiedPredictor(config)
    if not predictor.initialize():
        print("ERROR: Failed to initialize prediction system")
        return 1

    # Single match, matchday, or all upcoming
    if config.home_team and config.away_team:
        pred = predictor.predict_single(config.home_team, config.away_team)
        output = {
            "generated_at": datetime.now().isoformat(),
            "mode": config.mode,
            "predictions": [pred],
            "summary": {"total_matches": 1},
        }
    elif config.matchday is not None:
        output = predictor.predict_matchday(config.matchday)
    else:
        output = predictor.predict_upcoming()

    if not output or not output.get("predictions"):
        print("ERROR: No predictions generated!")
        return 1

    # Output
    if config.output_format == "json":
        print(json.dumps(output, indent=2, cls=_NumpySafeEncoder))
    else:
        print_predictions(output, verbose=config.verbose, brief=config.brief)
        out_path = config.output_path or str(DATA_DIR / "upcoming" / "predictions.json")
        print(f"\nPredictions saved to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
