#!/usr/bin/env python3
"""ENSEMBLE PREDICTION ENGINE - Combines All Prediction Methods

This engine combines three prediction approaches for maximum accuracy:

1. FACTOR-BASED (40% weight)
   - Validated factors from 21 seasons
   - Interpretable and robust
   - Current production system

2. XG-BASED (40% weight)
   - Predicts home_xG and away_xG using regression
   - Converts to win probabilities using Poisson distribution
   - This is how bookmakers actually work

3. ML CLASSIFIER (20% weight)
   - CatBoost trained on 51 features
   - 52.8% CV accuracy
   - Captures patterns humans miss

Expected improvement: 50.8% -> 55%+ accuracy
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class _NumpySafeEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.bool_, np.generic)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
from scipy.stats import poisson

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import DATA_DIR, MODELS_DIR

# Import formation analysis
try:
    from features.formation_analysis import FormationDatabase
    FORMATION_AVAILABLE = True
except ImportError:
    FORMATION_AVAILABLE = False

# Import Phase 4: Enhanced momentum and market intelligence
try:
    from features.enhanced_momentum import (
        compute_big_win_momentum,
        compute_comeback_momentum,
        compute_late_goal_trend,
        compute_momentum_composite,
    )
    ENHANCED_MOMENTUM_AVAILABLE = True
except ImportError:
    ENHANCED_MOMENTUM_AVAILABLE = False

try:
    from features.market_intelligence import MarketIntelligence
    MARKET_INTELLIGENCE_AVAILABLE = True
except ImportError:
    MARKET_INTELLIGENCE_AVAILABLE = False

try:
    from features.enhanced_weather import get_enhanced_weather_features
    ENHANCED_WEATHER_AVAILABLE = True
except ImportError:
    ENHANCED_WEATHER_AVAILABLE = False

try:
    from features.sentiment_analysis import get_match_sentiment_features
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

# Import Phase 5: Deep Learning Models
try:
    from models.deep_learning import DeepPredictor
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False

# Import Phase 6: Calibration Pipeline (Draw Detection, Confidence Filtering, Home Calibration)
try:
    from features.prediction_calibration import (
        CalibrationPipeline,
        BETTING_STRATEGIES,
        list_strategies,
    )
    from features.draw_detection import DrawDetector
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False

# Import Phase 7: Prediction Correction Layer (Static + Rolling bias correction)
try:
    from ml.correction_layer import (
        CorrectionLayer,
        extract_context_features,
        append_to_ledger,
    )
    CORRECTION_LAYER_AVAILABLE = True
except ImportError:
    CORRECTION_LAYER_AVAILABLE = False

# Import existing components
try:
    from scripts.prediction.predict_unified import (
        FACTOR_LIFTS, STACKING_BONUSES, BASE_RATES,
        load_upcoming_matches, identify_all_factors, generate_prediction,
    )
    from scripts.prediction.weather_integration import fetch_all_match_weather
    from scripts.prediction.current_form_calculator import calculate_all_forms
    from scripts.prediction.referee_integration import analyze_referee_impact
except ImportError:
    from scripts.prediction.predict_unified import (
        FACTOR_LIFTS, STACKING_BONUSES, BASE_RATES,
        load_upcoming_matches, identify_all_factors, generate_prediction,
    )
    from scripts.prediction.weather_integration import fetch_all_match_weather
    from scripts.prediction.current_form_calculator import calculate_all_forms
    from scripts.prediction.referee_integration import analyze_referee_impact

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# =============================================================================
# ENSEMBLE WEIGHTS
# =============================================================================

ENSEMBLE_WEIGHTS = {
    "factor": 0.035,       # Market-anchored + situational factors
    "xg": 0.124,           # xG + Poisson distribution
    "ml": 0.605,           # ML classifier (no-odds CatBoost + 3-model ensemble, 35 features)
    "player_xg": 0.032,    # Player-level xG
    "market": 0.205,       # Market-implied probabilities
}
# Optimized via Optuna (300 trials, TPE sampler) for catboost_no_odds model.
# Model: 35 features, 2017+ data, time-decay 0.85/season, auto draw weights.
# CV: acc=0.6155, ll=0.8589 (Mar 23 2026). Production acc=69.3% on 2025-2026.
# Walk-forward backtest: +12.3% ROI, €1000→€4192 (643 bets, 2023-2025).
# ML temperature T=0.75, draw_boost=1.12, post_T=0.90.

# Deep learning weights: DISABLED (0%).
# LSTM/Transformer accuracy 45-46% (near random on 3-way classification).
# Root causes: random_split ignores temporal order (data leakage), and 7,829
# matches is too small for deep learning. Keeping infrastructure for future
# use when we have 30k+ multi-league matches.
# NOTE: These weights MUST match ENSEMBLE_WEIGHTS with deep=0.00 added.
# Deep learning is loaded in production, so this dict is used for the
# 6-method case. If deep ever gets real weight, re-optimize all weights.
ENSEMBLE_WEIGHTS_WITH_DEEP = {
    "factor": 0.035,
    "xg": 0.124,
    "ml": 0.605,           # 35-feature no-odds CatBoost + ensemble (2017+ data)
    "player_xg": 0.032,
    "deep": 0.00,          # Disabled: insufficient data for deep learning
    "market": 0.205,
}

# Fallback weights if a method fails
# Derived proportionally from ENSEMBLE_WEIGHTS (factor=0.035, xg=0.124, ml=0.605, player_xg=0.032, market=0.205)
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


# =============================================================================
# XG MODEL - Poisson-based predictions
# =============================================================================

class XGPredictor:
    """Predicts win probabilities using xG regression + Poisson distribution."""

    # Extended feature list from train_extended_ensemble.py - 139 features
    # Organized by category for maintainability

    # Core ELO/Strength features (13)
    ELO_STRENGTH_FEATURES = [
        "home_elo", "away_elo", "elo_diff",
        "home_attack_strength", "home_defense_strength",
        "away_attack_strength", "away_defense_strength",
        "home_xg_attack_strength", "home_xg_defense_strength",
        "away_xg_attack_strength", "away_xg_defense_strength",
        "attack_strength_diff", "defense_strength_diff",
    ]

    # Form/Rolling features (30+)
    FORM_ROLLING_FEATURES = [
        "home_form_points_3", "home_form_points_5",
        "away_form_points_3", "away_form_points_5",
        "home_roll_3_goals_scored", "home_roll_3_goals_conceded",
        "away_roll_3_goals_scored", "away_roll_3_goals_conceded",
        "home_roll_5_goals_scored", "home_roll_5_goals_conceded",
        "away_roll_5_goals_scored", "away_roll_5_goals_conceded",
        "home_roll_10_goals_scored", "home_roll_10_goals_conceded",
        "away_roll_10_goals_scored", "away_roll_10_goals_conceded",
        "home_roll_3_clean_sheet", "away_roll_3_clean_sheet",
        "home_roll_5_clean_sheet", "away_roll_5_clean_sheet",
        "home_roll_10_clean_sheet", "away_roll_10_clean_sheet",
        "home_roll_10_win_rate", "away_roll_10_win_rate",
        "home_roll_10_shots_on_target", "away_roll_10_shots_on_target",
        "home_roll_10_yellow_cards", "away_roll_10_yellow_cards",
        "home_venue_roll_5_goals_scored", "home_venue_roll_5_goals_conceded",
        "away_venue_roll_5_goals_scored", "away_venue_roll_5_goals_conceded",
    ]

    # Understat xG features (26) - MOST IMPORTANT
    UNDERSTAT_FEATURES = [
        "home_us_team_xg", "away_us_team_xg", "us_xg_diff",
        "home_us_team_xg_per_90", "away_us_team_xg_per_90",
        "home_us_team_xg_per_shot", "away_us_team_xg_per_shot",
        "home_us_team_npxg", "away_us_team_npxg",
        "home_us_team_goals_minus_xg", "away_us_team_goals_minus_xg",
        "home_us_team_xa", "away_us_team_xa",
        "home_us_team_xa_per_90", "away_us_team_xa_per_90",
        "us_xa_diff",
        "home_us_team_xg_buildup", "away_us_team_xg_buildup",
        "home_us_team_xg_chain", "away_us_team_xg_chain",
        "home_us_top3_xg_share", "away_us_top3_xg_share",
        "home_us_team_key_passes", "away_us_team_key_passes",
        "home_us_team_shots", "away_us_team_shots",
    ]

    # Momentum/Streak features (12)
    MOMENTUM_FEATURES = [
        "home_win_streak", "away_win_streak",
        "home_unbeaten_run", "away_unbeaten_run",
        "home_winless_run", "away_winless_run",
        "home_loss_streak", "away_loss_streak",
        "home_scoring_streak", "away_scoring_streak",
        "home_clean_sheet_streak", "away_clean_sheet_streak",
    ]

    # Rest/Congestion features (11)
    REST_FEATURES = [
        "home_rest_days", "away_rest_days", "rest_advantage",
        "home_short_rest", "away_short_rest",
        "home_very_short_rest", "away_very_short_rest",
        "home_congestion_3", "away_congestion_3",
        "home_congestion_5", "away_congestion_5",
    ]

    # League position features (18)
    LEAGUE_POSITION_FEATURES = [
        "league_position_diff",
        "home_league_pos", "away_league_pos",
        "home_league_points", "away_league_points",
        "home_league_gd", "away_league_gd",
        "home_league_goals_for", "away_league_goals_for",
        "home_league_wins", "away_league_wins",
        "home_league_draws", "away_league_draws",
        "home_league_losses", "away_league_losses",
        "home_in_relegation_zone", "away_in_relegation_zone",
        "home_in_title_race", "away_in_title_race",
    ]

    # H2H features (8)
    H2H_FEATURES = [
        "h2h_matches_played", "h2h_home_wins", "h2h_draws", "h2h_away_wins",
        "h2h_home_win_rate", "h2h_home_goals_avg", "h2h_away_goals_avg",
        "h2h_last_result",
    ]

    # Venue features (4)
    VENUE_FEATURES = [
        "home_stadium_capacity", "travel_distance_km", "altitude_diff",
        "matchup_competitiveness",
    ]

    # Derived features (14)
    DERIVED_FEATURES = [
        "home_adj_attack_5", "home_adj_defense_5",
        "away_adj_attack_5", "away_adj_defense_5",
        "home_adj_attack_10", "home_adj_defense_10",
        "away_adj_attack_10", "away_adj_defense_10",
        "home_gd_roll_3", "away_gd_roll_3",
        "home_gd_roll_5", "away_gd_roll_5",
        "home_opp_difficulty_roll_5", "away_opp_difficulty_roll_5",
    ]

    # Combined extended feature list (139 total)
    XG_FEATURES = (
        ELO_STRENGTH_FEATURES +
        FORM_ROLLING_FEATURES +
        UNDERSTAT_FEATURES +
        MOMENTUM_FEATURES +
        REST_FEATURES +
        LEAGUE_POSITION_FEATURES +
        H2H_FEATURES +
        VENUE_FEATURES +
        DERIVED_FEATURES
    )

    def __init__(self):
        self.home_model = None
        self.away_model = None
        self.feature_names = self.XG_FEATURES  # fallback if metadata unavailable
        self.loaded = False

    def load_models(self) -> bool:
        """Load xG regression models and feature list from metadata."""
        try:
            from catboost import CatBoostRegressor

            home_path = MODELS_DIR / "universal" / "xg_home.cbm"
            away_path = MODELS_DIR / "universal" / "xg_away.cbm"
            meta_path = MODELS_DIR / "universal" / "xg_model_metadata.json"

            if not home_path.exists() or not away_path.exists():
                log.warning("xG models not found. Run train_extended_ensemble.py first.")
                return False

            self.home_model = CatBoostRegressor()
            self.home_model.load_model(str(home_path))

            self.away_model = CatBoostRegressor()
            self.away_model.load_model(str(away_path))

            # Load feature names from metadata (matches training feature order)
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                    saved_features = meta.get("feature_names", [])
                    if saved_features:
                        self.feature_names = saved_features
                        log.info(f"Loaded xG feature list from metadata: {len(self.feature_names)} features")
                    else:
                        log.info("Metadata has no feature_names, using fallback XG_FEATURES list")
            else:
                log.info("No extended_model_metadata.json found, using fallback XG_FEATURES list")

            self.loaded = True
            log.info("Loaded xG models successfully")
            return True

        except Exception as e:
            log.error(f"Failed to load xG models: {e}")
            return False

    def predict(self, features: pd.DataFrame) -> Dict[str, float]:
        """Predict win probabilities from features using Poisson distribution."""
        if not self.loaded:
            if not self.load_models():
                return None

        try:
            # Ensure we have the right features
            X = self._prepare_features(features)

            # Predict xG
            home_xg = float(self.home_model.predict(X)[0])
            away_xg = float(self.away_model.predict(X)[0])

            # Clip to reasonable range
            home_xg = max(0.3, min(4.0, home_xg))
            away_xg = max(0.3, min(4.0, away_xg))

            # xG bias correction: CatBoost regressors systematically over-predict
            # home goals (+0.04 avg across 2020-2025) and slightly over-predict away.
            # Cross-validated on 3 seasons (2022-2025, 1140 matches):
            #   bias_h=+0.09, bias_a=+0.04 → avg ensemble LL 0.9157→0.9151
            home_xg = max(0.3, home_xg - 0.09)
            away_xg = max(0.3, away_xg - 0.04)

            # Convert to win probabilities using Poisson
            probs = self._poisson_win_prob(home_xg, away_xg)

            return {
                "home_xg": home_xg,
                "away_xg": away_xg,
                "prob_H": probs["H"],
                "prob_D": probs["D"],
                "prob_A": probs["A"],
            }

        except Exception as e:
            log.error(f"xG prediction failed: {e}")
            return None

    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for xG model - MUST match exact training order."""
        # Build feature dict first to avoid DataFrame fragmentation
        feature_dict = {}

        for col in self.feature_names:
            if col in features.columns:
                feature_dict[col] = features[col].values[0] if len(features) == 1 else features[col].values
            else:
                # Provide sensible defaults for missing features
                if "elo" in col:
                    feature_dict[col] = 1500.0
                elif "strength" in col:
                    feature_dict[col] = 1.0
                elif "form_points" in col:
                    feature_dict[col] = 5.0
                elif "roll_" in col and "goals" in col:
                    feature_dict[col] = 1.5
                elif "roll_" in col and "clean_sheet" in col:
                    feature_dict[col] = 0.3
                elif "roll_" in col and ("win_rate" in col or "rate" in col):
                    feature_dict[col] = 0.4
                elif "roll_" in col:
                    feature_dict[col] = 0.0
                elif "h2h_" in col:
                    feature_dict[col] = 1/3 if "rate" in col else (3 if "matches" in col else 1)
                elif "in_relegation" in col or "in_title" in col:
                    feature_dict[col] = 0
                elif "stadium_capacity" in col:
                    feature_dict[col] = 30000
                elif "travel_distance" in col:
                    feature_dict[col] = 300
                elif "altitude_diff" in col or "diff" in col:
                    feature_dict[col] = 0.0
                elif "rest" in col:
                    feature_dict[col] = 7 if "days" in col else 0
                elif "competitiveness" in col:
                    feature_dict[col] = 0.5
                elif "streak" in col or "run" in col:
                    feature_dict[col] = 0
                elif "congestion" in col:
                    feature_dict[col] = 0.0
                elif "league_" in col:
                    feature_dict[col] = 10 if "pos" in col else 0
                elif "us_" in col:
                    feature_dict[col] = 1.0 if "xg" in col else 0.0
                else:
                    feature_dict[col] = 0.0

        # Create DataFrame in one go (avoids fragmentation)
        # CRITICAL: CatBoost validates column order against training feature names.
        # We must return columns in EXACTLY self.feature_names order.
        X = pd.DataFrame([feature_dict])
        X = X[self.feature_names]  # Force exact column order
        return X.fillna(0)

    def _poisson_win_prob(self, home_xg: float, away_xg: float, max_goals: int = 10) -> Dict[str, float]:
        """Calculate win probabilities from expected goals using calibrated Poisson.

        Standard Poisson under-predicts draws at typical Serie A xG (1.2-1.5)
        because P(draw) ≈ 25% while P(home) and P(away) are each ≈ 37%.
        Historical calibration (2020-2024, 1520 matches) shows:
          - Close xG (gap<0.2): actual draws 39.1% vs Poisson 27.3% → inflate 1.43x
          - Large xG gap (>0.8): actual draws 18.7% vs Poisson 21.1% → deflate 0.89x
        Apply xG-gap-dependent draw inflation, then renormalize.
        """
        home_probs = [poisson.pmf(g, home_xg) for g in range(max_goals)]
        away_probs = [poisson.pmf(g, away_xg) for g in range(max_goals)]

        prob_home = 0.0
        prob_draw = 0.0
        prob_away = 0.0

        for h in range(max_goals):
            for a in range(max_goals):
                prob = home_probs[h] * away_probs[a]
                if h > a:
                    prob_home += prob
                elif h == a:
                    prob_draw += prob
                else:
                    prob_away += prob

        # Draw calibration: inflate draws for close matches, deflate for lopsided
        # Optimized via grid search. Current: (1.55, 0.20) from LOO CV on 2017+ data.
        xg_gap = abs(home_xg - away_xg)
        draw_inflate = max(0.90, min(1.55, 1.55 - 0.20 * xg_gap))
        prob_draw *= draw_inflate

        # Normalize
        total = prob_home + prob_draw + prob_away
        return {
            "H": prob_home / total,
            "D": prob_draw / total,
            "A": prob_away / total,
        }


# =============================================================================
# DRAW DETECTOR (binary draw vs non-draw, isotonic-calibrated)
# =============================================================================

class DrawDetector:
    """Binary draw detector that blends with the 3-class model's draw probability.

    Walk-forward validated (5 seasons, 1749 matches):
    - Blend alpha=0.32: LL 0.9834→0.9816 (-0.0017)
    - Improves draw calibration without hurting H/A predictions.
    """

    BLEND_ALPHA = 0.32  # Optimal blend weight (walk-forward validated)

    def __init__(self):
        self.model = None
        self.calibrator = None
        self.feature_names = None
        self.loaded = False

    def load_model(self) -> bool:
        """Load draw detector model + isotonic calibrator."""
        try:
            from catboost import CatBoostClassifier
            import pickle

            model_path = MODELS_DIR / "universal" / "draw_detector.cbm"
            cal_path = MODELS_DIR / "universal" / "draw_detector_calibrator.pkl"
            meta_path = MODELS_DIR / "universal" / "draw_detector_metadata.json"

            if not model_path.exists():
                log.debug("Draw detector not found, skipping")
                return False

            self.model = CatBoostClassifier()
            self.model.load_model(str(model_path))
            self.feature_names = list(self.model.feature_names_)

            if cal_path.exists():
                with open(cal_path, "rb") as f:
                    self.calibrator = pickle.load(f)

            # Load blend alpha from metadata if available
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                    self.BLEND_ALPHA = meta.get("blend_alpha", 0.32)

            self.loaded = True
            log.info("Loaded draw detector (%d features, alpha=%.2f)",
                     len(self.feature_names), self.BLEND_ALPHA)
            return True

        except Exception as e:
            log.debug("Failed to load draw detector: %s", e)
            return False

    def predict_draw_prob(self, features: pd.DataFrame) -> float:
        """Predict calibrated draw probability."""
        if not self.loaded:
            return None

        try:
            # Inject draw-specific synthetic features (match training in push_accuracy.py).
            # These are computed inline during training but not in features.parquet.
            features = features.copy()
            _elo = features["elo_diff"].iloc[0] if "elo_diff" in features.columns else 0
            _elo = float(_elo) if not pd.isna(_elo) else 0
            features["abs_elo_diff"] = abs(_elo)
            features["elo_close"] = float(abs(_elo) < 50)

            # Form features
            _hf = float(features.get("home_roll_5_points", pd.Series(0)).iloc[0] or 0)
            _af = float(features.get("away_roll_5_points", pd.Series(0)).iloc[0] or 0)
            features["form_diff_abs"] = abs(_hf - _af)

            _hgs = float(features.get("home_roll_5_goals_scored", pd.Series(1.3)).iloc[0] or 1.3)
            _ags = float(features.get("away_roll_5_goals_scored", pd.Series(1.1)).iloc[0] or 1.1)
            _hgc = float(features.get("home_roll_5_goals_conceded", pd.Series(1.1)).iloc[0] or 1.1)
            _agc = float(features.get("away_roll_5_goals_conceded", pd.Series(1.3)).iloc[0] or 1.3)
            features["total_goals_form"] = _hgs + _ags + _hgc + _agc
            features["goals_form_diff"] = _hgs - _ags

            # Defense/attack draw indicators
            _hds = float(features.get("home_defense_strength", pd.Series(1.0)).iloc[0] or 1.0)
            _ads = float(features.get("away_defense_strength", pd.Series(1.0)).iloc[0] or 1.0)
            features["defense_parity"] = 1.0 / (1.0 + abs(_hds - _ads))
            _has = float(features.get("home_attack_strength", pd.Series(1.0)).iloc[0] or 1.0)
            _aas = float(features.get("away_attack_strength", pd.Series(1.0)).iloc[0] or 1.0)
            features["attack_weakness"] = max(0, 1.0 - max(_has, _aas))

            # Draw tendency
            _hdt = float(features.get("home_draw_tendency_10", pd.Series(0.28)).iloc[0] or 0.28)
            _adt = float(features.get("away_draw_tendency_10", pd.Series(0.28)).iloc[0] or 0.28)
            features["combined_draw_tendency"] = (_hdt + _adt) / 2.0
            features["draw_tendency_product"] = _hdt * _adt

            # Odds-based features
            _bh = float(features.get("odds_B365H", pd.Series(0)).iloc[0] or 0)
            _ba = float(features.get("odds_B365A", pd.Series(0)).iloc[0] or 0)
            features["odds_spread"] = abs(_bh - _ba) if _bh > 0 and _ba > 0 else 0
            _bd = float(features.get("odds_B365D", pd.Series(0)).iloc[0] or 0)
            features["market_draw_prob"] = 1.0 / max(_bd, 1.01) if _bd > 0 else 0.28
            _pd = float(features.get("odds_PSD", pd.Series(0)).iloc[0] or 0)
            features["pin_draw_prob"] = 1.0 / max(_pd, 1.01) if _pd > 0 else 0.28

            available = [f for f in self.feature_names if f in features.columns]
            missing = [f for f in self.feature_names if f not in features.columns]

            X = features[available].copy()
            for col in missing:
                X[col] = 0.0
            X = X[self.feature_names]

            raw_prob = self.model.predict_proba(X)[0][1]

            if self.calibrator is not None:
                cal_prob = float(self.calibrator.predict([raw_prob])[0])
            else:
                cal_prob = raw_prob

            return cal_prob

        except Exception as e:
            log.debug("Draw detector prediction failed: %s", e)
            return None

    def blend_draw_prob(self, ensemble_probs: Dict, features: pd.DataFrame) -> Dict:
        """Blend draw detector output with ensemble probabilities.

        Replaces ensemble draw prob with weighted average of ensemble and
        detector, then renormalizes H/A proportionally.
        """
        draw_prob = self.predict_draw_prob(features)
        if draw_prob is None:
            return ensemble_probs

        alpha = self.BLEND_ALPHA
        old_d = ensemble_probs["prob_D"]
        new_d = (1 - alpha) * old_d + alpha * draw_prob

        # Renormalize H and A proportionally
        old_ha = ensemble_probs["prob_H"] + ensemble_probs["prob_A"]
        new_ha = max(1.0 - new_d, 0.05)

        if old_ha > 0:
            ratio_h = ensemble_probs["prob_H"] / old_ha
        else:
            ratio_h = 0.5

        return {
            "prob_H": ratio_h * new_ha,
            "prob_D": new_d,
            "prob_A": (1 - ratio_h) * new_ha,
        }


# =============================================================================
# EXTRA-COMPETITION CONGESTION (Coppa Italia + UCL + UEL)
# =============================================================================

class XCompLoader:
    """Load Coppa Italia + European competition timelines for congestion features.

    Walk-forward validated (5 seasons, 1900 matches, 36 configs):
    - Best: xc_home_p4 + xc_away_p4 with l2=5 → LL=0.9694 (-0.0025 vs base)
    - Binary signal: did team play an extra-comp match in last 4 days?
    - Sources: Coppa Italia (1016 matches), UCL+UEL (5114 matches)
    """

    # FotMob → our team name mapping (only mismatches)
    _NAME_MAP = {
        'Chievo Verona': 'Chievo', 'Hellas Verona': 'Verona',
        'Union Brescia': 'Brescia', 'Alma Juventus Fano': '_skip_',
        'FC Empoli': 'Empoli', 'Parma Calcio 1913': 'Parma',
        'Como 1907': 'Como', 'Spezia Calcio': 'Spezia', 'SPAL 2013': 'SPAL',
        'AC Milan': 'Milan', 'FC Internazionale': 'Inter',
        'Internazionale': 'Inter', 'Inter Milan': 'Inter',
        'AS Roma': 'Roma', 'SS Lazio': 'Lazio', 'SSC Napoli': 'Napoli',
    }

    def __init__(self):
        self.timeline = {}  # team -> sorted list of pd.Timestamp
        self.loaded = False

    def load(self) -> bool:
        """Load cup + European match timelines from scraped JSON files."""
        try:
            parsed_dir = DATA_DIR / "parsed"
            sources = [
                (parsed_dir / "coppa_italia_matches.json", "Coppa Italia"),
                (parsed_dir / "european_matches.json", None),
            ]

            total = 0
            for path, comp_default in sources:
                if not path.exists():
                    continue
                with open(path) as f:
                    matches = json.load(f)
                for m in matches:
                    if not m.get("date") or not m.get("finished"):
                        continue
                    if m.get("home_score") is None:
                        continue
                    date = pd.Timestamp(m["date"])
                    home = self._NAME_MAP.get(m["home_team"], m["home_team"])
                    away = self._NAME_MAP.get(m["away_team"], m["away_team"])
                    if home == "_skip_" or away == "_skip_":
                        continue
                    for team in (home, away):
                        if team not in self.timeline:
                            self.timeline[team] = []
                        self.timeline[team].append(date)
                    total += 1

            for team in self.timeline:
                self.timeline[team].sort()

            if total > 0:
                self.loaded = True
                log.info("Loaded xcomp timeline: %d matches, %d teams", total, len(self.timeline))
            return self.loaded

        except Exception as e:
            log.debug("Failed to load xcomp timeline: %s", e)
            return False

    def get_p4(self, team: str, match_date) -> int:
        """Return 1 if team played an extra-comp match in the last 4 days, else 0."""
        if not self.loaded:
            return 0
        tl = self.timeline.get(team)
        if not tl:
            return 0
        if not isinstance(match_date, pd.Timestamp):
            match_date = pd.Timestamp(match_date)
        return int(any((match_date - d).days <= 4 and d < match_date for d in tl))


# =============================================================================
# META-LEARNER COMBINER (replaces fixed-weight blending)
# =============================================================================

class MetaLearnerCombiner:
    """Logistic regression meta-learner trained on sub-predictor outputs.

    Walk-forward validated (1749 matches, folds 3-5 test):
    - LL=0.9595 vs fixed-weight LL=0.9719 (-0.0123)
    - Acc=54.6% vs fixed-weight 53.5% (+1.1pp)
    - Learns optimal per-match blending from 9 predictor probability features.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.loaded = False

    def load(self) -> bool:
        """Load trained meta-learner + scaler."""
        try:
            import pickle
            path = MODELS_DIR / "universal" / "meta_learner.pkl"
            if not path.exists():
                return False

            with open(path, "rb") as f:
                data = pickle.load(f)

            self.model = data["model"]
            self.scaler = data["scaler"]
            self.feature_names = data.get("feature_names", [])
            self.loaded = True
            log.info("Loaded meta-learner combiner (%d features)", len(self.feature_names))
            return True

        except Exception as e:
            log.debug("Failed to load meta-learner: %s", e)
            return False

    def combine(self, ml_probs: Dict, market_probs: Dict, xg_probs: Dict) -> Optional[Dict]:
        """Combine sub-predictor outputs using learned meta-learner.

        Args:
            ml_probs: {"prob_H": float, "prob_D": float, "prob_A": float}
            market_probs: same
            xg_probs: same

        Returns:
            Combined {"prob_H": float, "prob_D": float, "prob_A": float} or None
        """
        if not self.loaded:
            return None

        try:
            import numpy as np

            # Build feature vector: [ml_H, ml_D, ml_A, mkt_H, mkt_D, mkt_A, xg_H, xg_D, xg_A]
            X = np.array([[
                ml_probs["prob_H"], ml_probs["prob_D"], ml_probs["prob_A"],
                market_probs["prob_H"], market_probs["prob_D"], market_probs["prob_A"],
                xg_probs["prob_H"], xg_probs["prob_D"], xg_probs["prob_A"],
            ]])

            X_scaled = self.scaler.transform(X)
            probs = self.model.predict_proba(X_scaled)[0]

            return {
                "prob_H": float(probs[0]),
                "prob_D": float(probs[1]),
                "prob_A": float(probs[2]),
            }

        except Exception as e:
            log.debug("Meta-learner prediction failed: %s", e)
            return None


# =============================================================================
# ML CLASSIFIER
# =============================================================================

class MLClassifier:
    """Multi-model ensemble classifier for match outcome prediction.

    Priority: WeightedAverageEnsemble (3-model blend: XGB + LGB + CB)
            > single CatBoost (catboost_upcoming.cbm)
    """

    def __init__(self, league: str = "serie_a"):
        self.model = None
        self.ensemble = None
        self.use_ensemble = False
        self.feature_names = None
        self.loaded = False
        self.league = league

    def _league_model_dir(self) -> Path:
        """Return the model directory for the configured league."""
        if self.league == "serie_a":
            return MODELS_DIR
        return MODELS_DIR / self.league

    def load_model(self) -> bool:
        """Load trained ML models.

        For Serie A:
          Priority: no-odds CatBoost (independent signal, no market overlap)
                  > multi-model ensemble (XGB + LGB + CB)
                  > single CatBoost (catboost_upcoming.cbm)

        For other leagues:
          Priority: league-specific ensemble (XGB + LGB + CB)
                  > league-specific CatBoost (catboost_latest.cbm)
        """
        # --- Non-Serie A: load from league-specific model directory ---
        if self.league != "serie_a":
            return self._load_league_model()

        # --- Serie A: original priority chain ---
        # Priority 1: No-odds CatBoost — provides independent signal that
        # doesn't overlap with the market predictor. Backtested at +2.6pp
        # accuracy improvement over the odds-based ensemble.
        try:
            from catboost import CatBoostClassifier

            no_odds_path = MODELS_DIR / "universal" / "catboost_no_odds.cbm"
            no_odds_meta = MODELS_DIR / "universal" / "catboost_no_odds_metadata.json"
            if no_odds_path.exists():
                self.model = CatBoostClassifier()
                self.model.load_model(str(no_odds_path))
                self.feature_names = list(self.model.feature_names_)
                if not self.feature_names and no_odds_meta.exists():
                    with open(no_odds_meta) as f:
                        meta = json.load(f)
                        self.feature_names = meta.get("feature_names", [])
                self.loaded = True
                log.info("Loaded no-odds CatBoost (%d features) — independent ML signal",
                         len(self.feature_names))
                return True
        except Exception as e:
            log.info("No-odds model not available (%s), trying ensemble", e)

        # Priority 2: Multi-model ensemble (XGB + LGB + CB with blend weights)
        try:
            from ml.ensemble import WeightedAverageEnsemble
            ens = WeightedAverageEnsemble.load("universal")
            ens.blend_calibrator = None
            self.ensemble = ens
            self.use_ensemble = True
            self.feature_names = ens.feature_names
            self.loaded = True
            log.info("Loaded multi-model ensemble (%d features, weights: %s)",
                     len(self.feature_names),
                     ", ".join(f"{mt}={w:.3f}" for mt, w in zip(ens.model_order, ens.weights)))
            return True
        except Exception as e:
            log.info("Ensemble not available (%s), falling back to single CatBoost", e)

        # Priority 3: Single CatBoost model
        try:
            from catboost import CatBoostClassifier

            v1_path = MODELS_DIR / "universal" / "catboost_upcoming.cbm"
            v2_path = MODELS_DIR / "universal" / "catboost_upcoming_v2.cbm"
            primary_meta = MODELS_DIR / "universal" / "catboost_upcoming_v2_metadata.json"
            fallback_path = MODELS_DIR / "markets" / "prod_1x2.cbm"

            if v1_path.exists():
                model_path = v1_path
            elif v2_path.exists():
                model_path = v2_path
            else:
                model_path = fallback_path

            if not model_path.exists():
                log.warning("ML classifier not found. Run training first.")
                return False

            self.model = CatBoostClassifier()
            self.model.load_model(str(model_path))
            self.feature_names = list(self.model.feature_names_)

            if not self.feature_names and primary_meta.exists():
                with open(primary_meta) as f:
                    meta = json.load(f)
                    self.feature_names = meta.get("feature_names", [])

            self.loaded = True
            log.info(f"Loaded single CatBoost from {model_path.name} "
                     f"with {len(self.feature_names)} features")
            return True

        except Exception as e:
            log.error(f"Failed to load ML classifier: {e}")
            return False

    def _load_league_model(self) -> bool:
        """Load models from a league-specific directory (non-Serie A).

        Priority: multi-model ensemble > single CatBoost (catboost_latest.cbm)
        """
        model_dir = self._league_model_dir()
        if not model_dir.exists():
            log.warning("Model directory not found for %s: %s", self.league, model_dir)
            return False

        # Priority 1: Multi-model ensemble
        try:
            from ml.ensemble import WeightedAverageEnsemble
            ens = WeightedAverageEnsemble.load(self.league)
            ens.blend_calibrator = None
            self.ensemble = ens
            self.use_ensemble = True
            self.feature_names = ens.feature_names
            self.loaded = True
            log.info("Loaded %s multi-model ensemble (%d features)",
                     self.league, len(self.feature_names))
            return True
        except Exception as e:
            log.info("%s ensemble not available (%s), trying CatBoost", self.league, e)

        # Priority 2: CatBoost (catboost_latest.cbm or any .cbm in the dir)
        try:
            from catboost import CatBoostClassifier

            latest_path = model_dir / "catboost_latest.cbm"
            meta_path = model_dir / "catboost_metadata.json"

            if not latest_path.exists():
                # Find any .cbm file
                cbm_files = sorted(model_dir.glob("catboost_*.cbm"), reverse=True)
                if not cbm_files:
                    log.warning("No CatBoost model found in %s", model_dir)
                    return False
                latest_path = cbm_files[0]
                meta_path = latest_path.with_name(
                    latest_path.name.replace(".cbm", "_metadata.json")
                )

            self.model = CatBoostClassifier()
            self.model.load_model(str(latest_path))
            self.feature_names = list(self.model.feature_names_)

            if not self.feature_names and meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                    self.feature_names = meta.get("feature_names", [])

            self.loaded = True
            log.info("Loaded %s CatBoost from %s (%d features)",
                     self.league, latest_path.name, len(self.feature_names))
            return True

        except Exception as e:
            log.error("Failed to load %s ML classifier: %s", self.league, e)
            return False

    def predict(self, features: pd.DataFrame) -> Dict[str, float]:
        """Predict win probabilities."""
        if not self.loaded:
            if not self.load_model():
                return None

        try:
            X = self._prepare_features(features)

            if self.use_ensemble:
                # Ensemble blends all 3 models (calibrator already disabled)
                proba = self.ensemble.predict_proba(X)[0]
            else:
                proba = self.model.predict_proba(X)[0]

            # Temperature scaling — sharpens CatBoost 3-class predictions.
            # T=0.40 was over-aggressive: crushed draw probability from 24% to
            # 12.4%, forcing 4 separate compensating mechanisms (draw_boost,
            # draw_detection, Poisson inflation, formation adj). T=0.75 still
            # sharpens home/away predictions but preserves draw signal.
            T = 0.75
            eps = 1e-10
            logits = np.log(proba + eps)
            scaled = np.exp(logits / T)
            proba = scaled / scaled.sum()

            return {
                "prob_H": float(proba[0]),
                "prob_D": float(proba[1]),
                "prob_A": float(proba[2]),
            }

        except Exception as e:
            log.error(f"ML prediction failed: {e}")
            return None

    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML model."""
        features = features.copy()

        # Inject _has_* metadata indicators (must match ml/data.py training behavior).
        # During training, these are set to 1 if ANY column in the group has data.
        # Without this, the model sees 0.0 at prediction time and thinks "no data".
        gk_cols = [c for c in features.columns if "_gk_" in c or "psxg" in c.lower()]
        shot_cols = [c for c in features.columns if ("_shot_" in c and "quality" not in c.lower()) or "_avg_xg_per_shot" in c]
        odds_cols = [c for c in features.columns if any(c.startswith(p) or c.endswith(p.rstrip("_")) for p in
                     ["B365", "BW", "IW", "PS", "WH", "VC", "odds_", "implied_prob_", "overround"])]

        features["_has_gk_data"] = int(bool(gk_cols) and not features[gk_cols].isna().all(axis=1).all())
        features["_has_shot_data"] = int(bool(shot_cols) and not features[shot_cols].isna().all(axis=1).all())
        features["_has_odds"] = int(bool(odds_cols) and not features[odds_cols].isna().all(axis=1).all())

        available = [f for f in self.feature_names if f in features.columns]
        missing = [f for f in self.feature_names if f not in features.columns]

        X = features[available].copy()

        # Fill missing with appropriate defaults
        for col in missing:
            if "h2h_" in col:
                if "rate" in col:
                    X[col] = 1/3  # Uninformative prior
                else:
                    X[col] = 0
            elif "elo" in col:
                X[col] = 1500
            else:
                X[col] = 0.0

        X = X[self.feature_names]
        return X.fillna(0)


# =============================================================================
# PLAYER XG PREDICTOR - Lineup-based predictions
# =============================================================================

class PlayerXGPredictor:
    """Predicts match outcomes using player-level xG data."""

    def __init__(self):
        self.player_db = None
        self.lineup_predictor = None
        self.loaded = False

    def load(self) -> bool:
        """Load player xG database from SofaScore (primary) or FBref (fallback).

        Previously loaded BOTH sources, double-counting every match and inflating
        player stats (e.g. Dennis Man: 1678 matches instead of ~47).
        Now uses SofaScore as primary source (better xG/xA data for recent seasons)
        and only falls back to FBref if SofaScore is unavailable.
        """
        try:
            from features.player_xg_model import PlayerXGDatabase, LineupXGPredictor

            self.player_db = PlayerXGDatabase()

            # Use SofaScore as primary source (better xG coverage, recent seasons)
            if not self.player_db.build_from_sofascore():
                # Fallback: try FBref match data
                if not self.player_db.load():
                    self.player_db.build_from_match_data()

            self.player_db.save()

            self.lineup_predictor = LineupXGPredictor(self.player_db)
            self.loaded = True
            log.info(f"Loaded player xG database with {len(self.player_db.profiles)} players")
            return True

        except Exception as e:
            log.warning(f"Failed to load player xG predictor: {e}")
            return False

    def predict(self, home_team: str, away_team: str, confirmed_lineups: Dict = None) -> Optional[Dict[str, float]]:
        """Predict match probabilities using player-level xG.

        Args:
            home_team: Home team name
            away_team: Away team name
            confirmed_lineups: Optional dict from confirmed_lineups.json with
                               match_key -> {home_lineup: [...], away_lineup: [...]}
        """
        if not self.loaded:
            if not self.load():
                return None

        try:
            # Check for confirmed lineups
            home_lineup = None
            away_lineup = None
            lineup_source = "predicted"

            if confirmed_lineups:
                match_key = f"{home_team} vs {away_team}"
                match_data = confirmed_lineups.get(match_key)
                if match_data and match_data.get("home_lineup") and match_data.get("away_lineup"):
                    home_lineup = match_data["home_lineup"]
                    away_lineup = match_data["away_lineup"]
                    lineup_source = "confirmed"
                    log.info(f"Using CONFIRMED lineups for {match_key}")

            # Get lineup-based predictions (with opponent strength adjustment)
            home_pred = self.lineup_predictor.predict_team_xg(
                home_team, lineup=home_lineup, use_form=True, opponent=away_team
            )
            away_pred = self.lineup_predictor.predict_team_xg(
                away_team, lineup=away_lineup, use_form=True, opponent=home_team
            )

            home_xg = home_pred["predicted_xg"]
            away_xg = away_pred["predicted_xg"]

            # Convert to probabilities using Poisson
            probs = self._poisson_win_prob(home_xg, away_xg)

            return {
                "prob_H": probs["H"],
                "prob_D": probs["D"],
                "prob_A": probs["A"],
                "home_lineup_xg": home_xg,
                "away_lineup_xg": away_xg,
                "home_confidence": home_pred["confidence"],
                "away_confidence": away_pred["confidence"],
                "lineup_source": lineup_source,
            }

        except Exception as e:
            log.error(f"Player xG prediction failed: {e}")
            return None

    def _poisson_win_prob(self, home_xg: float, away_xg: float, max_goals: int = 10) -> Dict[str, float]:
        """Calculate win probabilities from expected goals using calibrated Poisson.

        Same draw calibration as XGPredictor — see that docstring for derivation.
        """
        home_xg = max(0.3, min(4.0, home_xg))
        away_xg = max(0.3, min(4.0, away_xg))

        home_probs = [poisson.pmf(g, home_xg) for g in range(max_goals)]
        away_probs = [poisson.pmf(g, away_xg) for g in range(max_goals)]

        prob_home = 0.0
        prob_draw = 0.0
        prob_away = 0.0

        for h in range(max_goals):
            for a in range(max_goals):
                prob = home_probs[h] * away_probs[a]
                if h > a:
                    prob_home += prob
                elif h == a:
                    prob_draw += prob
                else:
                    prob_away += prob

        # Draw calibration: inflate draws for close matches, deflate for lopsided
        # Same optimized params as XGPredictor (1.55, 0.20)
        xg_gap = abs(home_xg - away_xg)
        draw_inflate = max(0.90, min(1.55, 1.55 - 0.20 * xg_gap))
        prob_draw *= draw_inflate

        total = prob_home + prob_draw + prob_away
        return {
            "H": prob_home / total,
            "D": prob_draw / total,
            "A": prob_away / total,
        }


# =============================================================================
# FEATURE BUILDER - Build features for upcoming matches
# =============================================================================

class FeatureBuilder:
    """Build features for upcoming matches from historical data."""

    def __init__(self, league: str = "serie_a"):
        self.df = None
        self.team_features = {}
        self.league = league

    def load_historical(self) -> bool:
        """Load historical features data, filtered by league."""
        try:
            features_path = DATA_DIR / "features" / "features.parquet"
            if not features_path.exists():
                log.error(f"Features file not found: {features_path}")
                return False

            self.df = pd.read_parquet(features_path)

            # Filter by league if the column exists and we're not Serie A
            # (Serie A was the only league historically, so some rows may lack the column)
            if "league" in self.df.columns and self.league:
                league_mask = self.df["league"] == self.league
                if league_mask.any():
                    self.df = self.df[league_mask]
                    log.info(f"Filtered features to {self.league}: {len(self.df)} rows")
                else:
                    log.warning(f"No rows found for league={self.league}, using all data")

            self.df = self.df.sort_values("match_date", ascending=False)

            # Build team feature cache from most recent matches
            self._build_team_cache()

            log.info(f"Loaded {len(self.df)} historical matches for features")
            return True

        except Exception as e:
            log.error(f"Failed to load historical data: {e}")
            return False

    # Non-prefixed columns that the model uses and must be cached from
    # the most recent match row (they are match-level, not team-level).
    _MATCH_LEVEL_PREFIXES = (
        "odds_", "pinnacle_", "market_", "sharp_soft_",
        "ah_line", "h2h_", "manager_h2h_",
        "home_prob_", "away_prob_", "draw_prob_",
        "weather_", "travel_distance",
        "line_vel_",
    )

    # Matchup-specific features that must NOT be cached from a random
    # previous match — they create huge distribution shifts in interaction
    # features when stale.  Fresh values come from odds injection or are
    # computed per-matchup.
    _EXCLUDE_FROM_CACHE = {
        "home_prob_best", "draw_prob_best", "away_prob_best",
        "market_home_prob", "market_draw_prob", "market_away_prob",
        "pinnacle_home_prob", "pinnacle_draw_prob", "pinnacle_away_prob",
        "market_goal_total", "market_ou_over_prob", "market_ou_under_prob",
        "pinnacle_ou_over_prob", "pinnacle_ou_under_prob",
        "sharp_soft_home_div", "sharp_soft_draw_div", "sharp_soft_away_div",
        "sharp_soft_ou_div",
    }

    def _build_team_cache(self):
        """Cache the most recent features for each team.

        Also caches the single most recent match row for non-prefixed
        (match-level) features like odds, market probs, etc.
        Excludes matchup-specific probability features that would be stale.
        """
        seen_home = set()
        seen_away = set()
        self._latest_match_features = {}  # non-prefixed features from most recent row

        for _, row in self.df.iterrows():
            home = row.get("home_team")
            away = row.get("away_team")

            if home and home not in seen_home:
                self.team_features[f"{home}_home"] = self._extract_team_features(row, "home")
                seen_home.add(home)

            if away and away not in seen_away:
                self.team_features[f"{away}_away"] = self._extract_team_features(row, "away")
                seen_away.add(away)

            # Cache non-prefixed match-level features from the very first
            # (most recent) row — used as fallback defaults at prediction time
            if not self._latest_match_features:
                for col in row.index:
                    if col in self._EXCLUDE_FROM_CACHE:
                        continue
                    if pd.notna(row[col]) and any(col.startswith(p) for p in self._MATCH_LEVEL_PREFIXES):
                        self._latest_match_features[col] = row[col]

    # Suffixes of team-prefixed columns that are matchup-specific and must
    # NOT be cached from a previous match (they depend on the opponent).
    _TEAM_EXCLUDE_SUFFIXES = (
        "prob_best", "prob_pinnacle", "prob_market",
    )

    def _extract_team_features(self, row: pd.Series, side: str) -> Dict:
        """Extract ALL prefixed features for a team from the feature row.

        Dynamically extracts any column starting with the side prefix
        (home_/away_), including ss_roll_, fb_roll_, and any future features.
        Excludes matchup-specific features (e.g. prob_best) that depend on
        the opponent and would be stale from a different match.
        """
        prefix = f"{side}_"
        features = {}

        for col in row.index:
            if col.startswith(prefix) and pd.notna(row[col]):
                key = col[len(prefix):]  # Strip prefix
                if any(key.startswith(s) or key == s for s in self._TEAM_EXCLUDE_SUFFIXES):
                    continue
                features[key] = row[col]

        return features

    def build_match_features(self, home_team: str, away_team: str, form_data: Dict = None, match_date=None) -> pd.DataFrame:
        """Build feature row for an upcoming match."""
        if self.df is None:
            if not self.load_historical():
                return None

        features = {}

        # Inject non-prefixed match-level features as fallback defaults.
        # These include odds_*, market_*, pinnacle_*, h2h_*, etc. that the
        # model was trained on but the team cache doesn't capture.
        # They'll be overridden by fresh data (odds injection, H2H computation)
        # later in the pipeline.
        if hasattr(self, '_latest_match_features'):
            features.update(self._latest_match_features)

        # Get team features from cache
        home_cache = self.team_features.get(f"{home_team}_home", {})
        away_cache = self.team_features.get(f"{away_team}_away", {})

        # Build feature row
        for key, val in home_cache.items():
            features[f"home_{key}"] = val

        for key, val in away_cache.items():
            features[f"away_{key}"] = val

        # Calculate derived features
        home_elo = features.get("home_elo", 1500)
        away_elo = features.get("away_elo", 1500)
        features["elo_diff"] = home_elo - away_elo

        home_attack = features.get("home_attack_strength", 1.0)
        away_defense = features.get("away_defense_strength", 1.0)
        home_defense = features.get("home_defense_strength", 1.0)
        away_attack = features.get("away_attack_strength", 1.0)

        features["home_attack_vs_away_def"] = home_attack - away_defense
        features["away_attack_vs_home_def"] = away_attack - home_defense

        # Additional derived features for xG model
        features["attack_strength_diff"] = home_attack - away_attack
        features["defense_strength_diff"] = home_defense - away_defense
        features["matchup_competitiveness"] = 1.0 / (1.0 + abs(home_elo - away_elo) / 400.0)

        # League position diff — use actual cached positions (matches training)
        h_pos = features.get("home_league_pos", 10)
        a_pos = features.get("away_league_pos", 10)
        features["league_position_diff"] = h_pos - a_pos

        # Use form data if available — ONLY override features that are genuinely
        # fresher than the cached features.parquet values.  Rolling goal averages
        # (roll_3_*, roll_5_*) are properly computed in features.parquet; overriding
        # them with scaled approximations degrades accuracy.
        if form_data:
            key = f"{home_team} vs {away_team}"
            matchup = form_data.get("matchups", {}).get(key, {})

            home_form = matchup.get("home_form", {})
            away_form = matchup.get("away_form", {})

            # Elo is genuinely fresh (updated after every match)
            if home_form.get("elo"):
                features["home_elo"] = home_form["elo"]
            if away_form.get("elo"):
                features["away_elo"] = away_form["elo"]
            if "elo" in home_form and "elo" in away_form:
                features["elo_diff"] = home_form["elo"] - away_form["elo"]

            # Form points over last 5 matches — direct match to training feature
            if home_form.get("total_points") is not None:
                features["home_form_points_5"] = home_form["total_points"]
            if away_form.get("total_points") is not None:
                features["away_form_points_5"] = away_form["total_points"]

            # Goals per game from fresh form data → override roll_5 averages
            # current_form_calculator provides gpg (goals per game) directly
            if home_form.get("gpg") is not None:
                features["home_roll_5_goals_scored"] = home_form["gpg"]
            if away_form.get("gpg") is not None:
                features["away_roll_5_goals_scored"] = away_form["gpg"]

        # Get H2H features (with date filtering to prevent temporal leakage)
        h2h_features = self._get_h2h_features(home_team, away_team, match_date=pd.Timestamp.now())
        features.update(h2h_features)

        # Add contextual features
        features.update(self._get_contextual_features(features))

        # Add Understat derived features
        home_xg = features.get("home_us_team_xg", 1.5)
        away_xg = features.get("away_us_team_xg", 1.2)
        home_xa = features.get("home_us_team_xa", 0.4)
        away_xa = features.get("away_us_team_xa", 0.35)
        features["us_xg_diff"] = home_xg - away_xg
        features["us_xa_diff"] = home_xa - away_xa

        # Rest advantage
        home_rest = features.get("home_rest_days", 7)
        away_rest = features.get("away_rest_days", 7)
        features["rest_advantage"] = home_rest - away_rest

        # Compute SofaScore differential features (ss_diff_*)
        # These are expected by the ML model but not stored per-side
        has_ss = False
        has_ss_xg = False
        for key in list(home_cache.keys()):
            if key.startswith("ss_roll_"):
                h_val = features.get(f"home_{key}")
                a_val = features.get(f"away_{key}")
                if h_val is not None and a_val is not None:
                    features[f"ss_diff_{key}"] = h_val - a_val
                    has_ss = True
                    if key == "ss_roll_xg":
                        has_ss_xg = True
            elif key == "top2_xg_share":
                h_val = features.get(f"home_{key}")
                a_val = features.get(f"away_{key}")
                if h_val is not None and a_val is not None:
                    features[f"ss_diff_{key}"] = h_val - a_val

        # Compute Sofascore index differential features (ss_idx_diff_*)
        for key in list(home_cache.keys()):
            if key.startswith("ss_idx_"):
                h_val = features.get(f"home_{key}")
                a_val = features.get(f"away_{key}")
                if h_val is not None and a_val is not None:
                    features[f"ss_idx_diff_{key.replace('ss_idx_', '')}"] = h_val - a_val

        # Compute FBref differential features (fb_diff_*)
        has_fb = False
        for key in list(home_cache.keys()):
            if key.startswith("fb_roll_"):
                h_val = features.get(f"home_{key}")
                a_val = features.get(f"away_{key}")
                if h_val is not None and a_val is not None:
                    diff_name = f"fb_diff_{key.replace('fb_roll_', '')}"
                    features[diff_name] = h_val - a_val
                    has_fb = True

        # Coverage flags
        features["ss_coverage"] = 1 if has_ss else 0
        features["ss_xg_coverage"] = 1 if has_ss_xg else 0
        features["fb_coverage"] = 1 if has_fb else 0

        # --- Fill remaining model-expected features with sensible defaults ---

        # Pinnacle odds: fallback from B365 or Max if not available
        if "odds_PSH" not in features or pd.isna(features.get("odds_PSH")):
            features["odds_PSH"] = features.get("odds_B365H", features.get("odds_MaxH", 0))
        if "odds_PSD" not in features or pd.isna(features.get("odds_PSD")):
            features["odds_PSD"] = features.get("odds_B365D", features.get("odds_AvgD", 0))
        if "odds_PSA" not in features or pd.isna(features.get("odds_PSA")):
            features["odds_PSA"] = features.get("odds_B365A", features.get("odds_MaxA", 0))

        # Pinnacle closing odds: derive from opening Pinnacle or B365 if missing
        if "odds_PSCD" not in features or pd.isna(features.get("odds_PSCD")):
            features["odds_PSCD"] = features.get("odds_PSD", features.get("odds_B365D", 0))
        if "odds_PSCA" not in features or pd.isna(features.get("odds_PSCA")):
            features["odds_PSCA"] = features.get("odds_PSA", features.get("odds_B365A", 0))

        # SS diff features that are computed as diffs directly (no home_/away_ versions)
        features.setdefault("ss_diff_ss_roll_mid_rating", 0.0)
        features.setdefault("ss_diff_ss_roll_mins_concentration", 0.0)

        # Calendar features — use actual match date when available, not prediction runtime
        ref_date = pd.Timestamp(match_date) if match_date is not None else pd.Timestamp.now()
        # Midweek = Tue/Wed/Thu (match congestion.py definition: dow.isin([1,2,3]))
        _is_mid = 1 if ref_date.dayofweek in (1, 2, 3) else 0
        features.setdefault("is_midweek", _is_mid)
        features.setdefault("home_is_midweek", _is_mid)
        features.setdefault("away_is_midweek", _is_mid)
        features.setdefault("is_mid_season", 1 if ref_date.month in (11, 12, 1, 2) else 0)

        # Weather — seasonal defaults (24% coverage in training, model handles 0)
        features.setdefault("weather_rain_sum", 0.0)
        month = ref_date.month
        # Approximate Italian temperature by month (°C)
        _monthly_temp = {1:7,2:9,3:13,4:16,5:21,6:26,7:29,8:29,9:24,10:18,11:12,12:8}
        features.setdefault("weather_apparent_temperature_max", _monthly_temp.get(month, 15))
        # Wind direction: 0=N, 180=S — use neutral/typical value
        features.setdefault("weather_wind_direction_10m_dominant", 200)

        # Line velocity features — 0 = no odds movement (65% coverage in training)
        for lv_col in ["line_vel_pin_home", "line_vel_pin_draw", "line_vel_pin_away",
                        "line_vel_mkt_home", "line_vel_mkt_draw", "line_vel_mkt_away",
                        "line_vel_ou_over", "line_vel_ou_under"]:
            features.setdefault(lv_col, 0.0)

        # rolling_goals_diff: home_roll_5_goals_scored - away_roll_5_goals_scored
        h_gs = features.get("home_roll_5_goals_scored", 1.3)
        a_gs = features.get("away_roll_5_goals_scored", 1.1)
        features.setdefault("rolling_goals_diff", h_gs - a_gs)

        # squad_value_ratio: approximate from Elo (Elo correlates with squad value)
        # Training mean=20.7, range ~0.3-100+. Elo 1500=avg → ratio=1.0 → scaled to ~20
        h_elo = features.get("home_elo", 1500)
        a_elo = features.get("away_elo", 1500)
        elo_ratio = max(0.3, h_elo / max(a_elo, 1000))
        features.setdefault("squad_value_ratio", elo_ratio * 20.0)

        # PPDA features — compute from team caches or use league average (12.2)
        h_ppda = features.get("home_ppda", 12.2)
        a_ppda = features.get("away_ppda", 12.2)
        h_ppda_a = features.get("home_ppda_allowed", 12.2)
        a_ppda_a = features.get("away_ppda_allowed", 12.2)
        features.setdefault("ppda_differential", h_ppda - a_ppda)
        features.setdefault("ppda_allowed_diff", h_ppda_a - a_ppda_a)
        h_xg_r5 = features.get("home_roll_5_xg_for", h_gs)
        a_xg_r5 = features.get("away_roll_5_xg_for", a_gs)
        features.setdefault("ppda_x_goals", (15 - h_ppda) * h_xg_r5 - (15 - a_ppda) * a_xg_r5)

        # formation_width_mismatch — compute from formation DB if available
        features.setdefault("formation_width_mismatch", 0.0)

        # ss_diff_ss_roll_def_rating — SofaScore defensive rating differential
        h_def_r = features.get("home_ss_roll_def_rating", 0)
        a_def_r = features.get("away_ss_roll_def_rating", 0)
        features.setdefault("ss_diff_ss_roll_def_rating", (h_def_r or 0) - (a_def_r or 0))

        # --- Additional derived features selected by the new model ---

        # Pinnacle probabilities: derive from Pinnacle odds or fallback to B365
        psh = features.get("odds_PSH", features.get("odds_B365H", 2.5))
        psd = features.get("odds_PSD", features.get("odds_B365D", 3.3))
        psa = features.get("odds_PSA", features.get("odds_B365A", 3.0))
        pin_total = (1/psh + 1/psd + 1/psa) if psh and psd and psa else 1.0
        features.setdefault("pinnacle_home_prob", round(1/psh / pin_total, 4) if psh else 0.33)
        features.setdefault("pinnacle_away_prob", round(1/psa / pin_total, 4) if psa else 0.33)
        features.setdefault("draw_prob_best", round(1/psd / pin_total, 4) if psd else 0.28)

        # goal_total_vs_xg: market O/U line minus rolling xG total (mirrors build.py Step 18)
        # market_goal_total isn't available at prediction time (not in parquet).
        # Training median=2.39, driven by O/U line (~2.5) minus rolling xG (~0.1).
        # Use training median as default since we can't recompute without O/U data.
        features.setdefault("goal_total_vs_xg", 2.39)

        # promoted_derby: 0 for most matches (rare — only 0.6% of training data)
        features.setdefault("promoted_derby", 0)

        # season_phase: 1=early, 2=mid-early, 3=mid, 4=mid-late, 5=late
        _month_to_phase = {8:1, 9:1, 10:2, 11:2, 12:3, 1:3, 2:4, 3:4, 4:5, 5:5, 6:5, 7:1}
        features.setdefault("season_phase", _month_to_phase.get(ref_date.month, 3))

        # day_of_week: 0=Mon ... 6=Sun (training mean ~4.9 = Sat/Sun heavy)
        features.setdefault("day_of_week", ref_date.dayofweek)

        # attack_defense_mismatch: ratio-based (mirrors build.py Step 10)
        # home_attack/away_defense - away_attack/home_defense
        features.setdefault("attack_defense_mismatch",
                            round(home_attack / max(away_defense, 0.1) - away_attack / max(home_defense, 0.1), 3))

        # rest_x_close_elo: rest advantage × closeness of Elo (interaction)
        # MUST use binary gate (|elo_diff| < 50) to match training in build.py:1783-1784
        rest_adv = features.get("rest_advantage", 0) or 0
        elo_close = 1.0 if abs(features.get("elo_diff", 0) or 0) < 50 else 0.0
        features.setdefault("rest_x_close_elo", round(rest_adv * elo_close, 4))

        # altitude_diff: home stadium altitude - away stadium altitude (meters)
        try:
            from features.venue import get_altitude_difference
            features.setdefault("altitude_diff", get_altitude_difference(home_team, away_team))
        except Exception as e:
            log.debug("Altitude diff unavailable: %s", e)
            features.setdefault("altitude_diff", 0.0)

        # rolling_gd_diff: goal difference differential (home GD - away GD)
        h_gd = (features.get("home_roll_5_goals_scored", 1.3) -
                features.get("home_roll_5_goals_conceded", 1.1))
        a_gd = (features.get("away_roll_5_goals_scored", 1.1) -
                features.get("away_roll_5_goals_conceded", 1.3))
        features.setdefault("rolling_gd_diff", round(h_gd - a_gd, 4))

        # formation_total_advantage: default 0 (neutral)
        features.setdefault("formation_total_advantage", 0.0)

        # defensive_form_diff: home defensive form - away defensive form
        h_def_form = features.get("home_roll_5_goals_conceded", 1.1)
        a_def_form = features.get("away_roll_5_goals_conceded", 1.3)
        features.setdefault("defensive_form_diff", round(a_def_form - h_def_form, 4))

        # combined_disruption: injury + transfer disruption + manager newness (mirrors build.py:1746-1752)
        h_inj = float(features.get("home_injury_impact", 0) or 0)
        a_inj = float(features.get("away_injury_impact", 0) or 0)
        h_dis = float(features.get("home_squad_disruption", 0) or 0)
        a_dis = float(features.get("away_squad_disruption", 0) or 0)
        h_mgr_new = 1.0 if features.get("home_manager_is_new") else 0.0
        a_mgr_new = 1.0 if features.get("away_manager_is_new") else 0.0
        features.setdefault("combined_disruption",
                            round((h_inj - a_inj) + (h_dis - a_dis) + (h_mgr_new - a_mgr_new) * 0.3, 3))

        # congestion_asymmetry: rest days differential (mirrors creative_factors.py)
        # Training uses home_rest_days - away_rest_days, NOT congestion_index
        h_rest = features.get("home_rest_days", 7) or 7
        a_rest = features.get("away_rest_days", 7) or 7
        features.setdefault("congestion_asymmetry", round(float(h_rest) - float(a_rest), 0))

        # new_mgr_x_home: matches training formula (is_new_manager * elo_diff * -0.1)
        h_mgr_new = features.get("home_manager_is_new", 0) or 0
        features.setdefault("new_mgr_x_home", round(float(h_mgr_new) * features.get("elo_diff", 0) * -0.1, 3))

        # Captain features — 0 = no data / neutral signal (matches training fillna)
        for cap_col in ["home_captain_played", "away_captain_played",
                         "home_captain_consistency", "away_captain_consistency",
                         "home_captain_effect", "away_captain_effect"]:
            features.setdefault(cap_col, 0.0)

        # Card/goal timing features — 0 = no data / neutral signal
        for ct_col in ["home_ct_early_sub_r5", "home_ct_first_sub_min_r5",
                        "home_ct_goals_first_15_rate", "home_ct_goals_last_15_rate",
                        "home_ct_total_goals_conceded_r5", "home_ct_cards_before_30_r5",
                        "away_ct_cards_before_30_r5", "away_ct_total_goals_scored_r5",
                        "away_ct_conceded_last_15_rate", "away_ct_early_sub_r5",
                        "away_ct_goals_first_15_rate"]:
            features.setdefault(ct_col, 0.0)

        # Referee features — defaults when no referee is assigned.
        # Overridden in predict() if a referee name is known.
        for ref_col in ["ref_avg_yellows", "ref_avg_reds", "ref_avg_fouls",
                         "ref_avg_penalties", "ref_matches_officiated",
                         "ref_strictness_score", "ref_home_bias",
                         "ref_home_cards_bias", "ref_avg_total_goals",
                         "ref_avg_xg", "ref_xg_vs_league",
                         "ref_strictness_trend", "ref_big_match_card_modifier",
                         "ref_home_team_cards", "ref_away_team_cards",
                         "ref_vs_home_team_bias", "ref_vs_away_team_bias",
                         # 4 features from referee_assignments (comprehensive_markets)
                         "ref_matches", "ref_avg_yellows_given",
                         "ref_avg_reds_given", "ref_total_cards_mean"]:
            features.setdefault(ref_col, 0.0)

        # --- Compute interaction features (mirrors build.py Step 36) ---
        features.update(self._compute_interaction_features(features))

        return pd.DataFrame([features])

    def _compute_interaction_features(self, f: Dict) -> Dict:
        """Compute the same interaction features as build.py _add_interaction_features.

        These are critical — 13 of the model's 103 features are interactions
        that were previously missing at prediction time.
        """
        out = {}
        elo_diff = f.get("elo_diff", 0) or 0

        # 4. Squad disruption × Elo
        h_disrupt = f.get("home_squad_disruption", 0) or 0
        a_disrupt = f.get("away_squad_disruption", 0) or 0
        out["disruption_x_elo"] = round((h_disrupt - a_disrupt) * elo_diff * -1, 3)

        # 5. Manager tenure × form
        h_tenure = min(f.get("home_manager_tenure", 1) or 1, 30)
        a_tenure = min(f.get("away_manager_tenure", 1) or 1, 30)
        h_pts = f.get("home_roll_5_points", 0) or 0
        a_pts = f.get("away_roll_5_points", 0) or 0
        out["tenure_x_form"] = round((h_tenure / 30) * h_pts - (a_tenure / 30) * a_pts, 3)

        # 9. Elo × form
        form_diff = h_pts - a_pts
        out["elo_x_form"] = round(elo_diff * form_diff / 10, 3)

        # 13. xG overperformance diff
        h_goals = f.get("home_roll_5_goals_scored", 0) or 0
        a_goals = f.get("away_roll_5_goals_scored", 0) or 0
        h_xg = f.get("home_roll_5_xg_for", 0) or 0
        a_xg = f.get("away_roll_5_xg_for", 0) or 0
        out["xg_overperformance_diff"] = round((h_goals - h_xg) - (a_goals - a_xg), 3)

        # 15. PPDA × xG trend (press_attack_signal)
        h_ppda = f.get("home_ppda", 12) or 12
        a_ppda = f.get("away_ppda", 12) or 12
        h_press_attack = (15 - h_ppda) * (h_xg if h_xg else 1)
        a_press_attack = (15 - a_ppda) * (a_xg if a_xg else 1)
        out["press_attack_signal"] = round(h_press_attack - a_press_attack, 3)

        # 16. Sharp-soft divergence × Elo
        sharp_div = f.get("sharp_soft_home_div", 0) or 0
        out["sharp_soft_x_elo"] = round(sharp_div * elo_diff, 4)

        # 17. Market prob vs Elo-expected prob (market_elo_disagreement)
        home_elo = f.get("home_elo", 1500) or 1500
        away_elo = f.get("away_elo", 1500) or 1500
        elo_expected = 1.0 / (1.0 + 10 ** (-(home_elo - away_elo) / 400))
        home_prob = f.get("home_prob_best", 0) or f.get("market_home_prob", 0) or 0
        out["market_elo_disagreement"] = round((home_prob if home_prob else elo_expected) - elo_expected, 4)

        # 19. Asian handicap line × form
        ah_line = f.get("ah_line_normalized", 0) or 0
        out["ah_x_form"] = round(ah_line * form_diff / 10, 4)

        # Understat xG trend (from cached features)
        out["home_us_xg_trend"] = f.get("home_us_xg_trend", 0) or 0

        # Manager H2H matches (from cached features)
        out["manager_h2h_matches"] = f.get("manager_h2h_matches", 0) or 0

        # --- v2 interaction features (14 new, walk-forward validated) ---
        # Odds-implied probabilities (importance: 1.77, 1.67, 1.50)
        b365h = f.get("odds_B365H", 2.5) or 2.5
        b365d = f.get("odds_B365D", 3.3) or 3.3
        b365a = f.get("odds_B365A", 3.0) or 3.0
        inv_sum = 1/max(b365h, 1.01) + 1/max(b365d, 1.01) + 1/max(b365a, 1.01)
        out.setdefault("implied_draw_prob", round((1/max(b365d, 1.01)) / inv_sum, 4))
        out.setdefault("implied_home_prob", round((1/max(b365h, 1.01)) / inv_sum, 4))
        out.setdefault("odds_overround", round(inv_sum - 1, 4))

        # Elo × attack interaction (importance: 1.13)
        atk_diff = f.get("attack_strength_diff", 0) or 0
        out.setdefault("elo_x_attack", round(elo_diff * atk_diff, 3))

        # Defense × competitiveness (importance: 0.62)
        def_diff = f.get("defense_strength_diff", 0) or 0
        comp = f.get("matchup_competitiveness", 0.5) or 0.5
        out.setdefault("defense_x_competitive", round(def_diff * comp, 3))

        # Comeback rate differential (importance: 0.85)
        h_comeback = f.get("home_comeback_rate", 0) or 0
        a_comeback = f.get("away_comeback_rate", 0) or 0
        out.setdefault("comeback_diff", round(h_comeback - a_comeback, 4))

        # Draw tendency × odds interaction (importance: 2.02)
        draw_tend = f.get("away_draw_tendency_10", 0.28) or 0.28
        out.setdefault("draw_tendency_x_odds", round(draw_tend * out.get("implied_draw_prob", 0.28), 4))

        # B365 vs Pinnacle disagreement (importance: 0.99)
        pin_away = f.get("pinnacle_away_prob", 0.33) or 0.33
        out.setdefault("b365_pin_disagreement", round(abs(1/max(b365h, 1.01) - (1 - pin_away)), 4))

        # Goals per shot × defense (importance: 0.67)
        gps = f.get("home_goals_per_shot_roll_5", 0.1) or 0.1
        out.setdefault("goals_per_shot_x_def", round(gps * def_diff, 4))

        # Injury × elo (importance: 0.75)
        inj = (f.get("home_injury_impact", 0) or 0) - (f.get("away_injury_impact", 0) or 0)
        out.setdefault("injury_x_elo", round(inj * elo_diff, 3))

        # HT lead hold differential (importance: 1.18)
        h_ht = f.get("home_ht_lead_hold", 0) or 0
        a_ht = f.get("away_ht_lead_hold", 0) or 0
        out.setdefault("ht_lead_hold_diff", round(h_ht - a_ht, 4))

        # Formation flexibility differential (importance: 1.55)
        h_flex = f.get("home_formation_flexibility", 0) or 0
        a_flex = f.get("away_formation_flexibility", 0) or 0
        out.setdefault("formation_flex_diff", round(h_flex - a_flex, 4))

        # Elo squared (non-linear elo effect) (importance: 0.28)
        out.setdefault("elo_diff_sq", round(elo_diff ** 2 * (1 if elo_diff >= 0 else -1), 1))

        # Congestion differential (importance: 0.09)
        h_cong = f.get("home_congestion_5", 0) or 0
        a_cong = f.get("away_congestion_5", 0) or 0
        out.setdefault("congestion_diff", round(h_cong - a_cong, 4))

        return out

    def _get_h2h_features(self, home_team: str, away_team: str, match_date=None) -> Dict:
        """Get head-to-head features from historical data.

        Uses date filtering to prevent temporal leakage - only considers
        matches played BEFORE match_date (mirrors features/h2h.py logic).
        """
        defaults = {
            "h2h_matches_played": 0,
            "h2h_home_wins": 0,
            "h2h_away_wins": 0,
            "h2h_draws": 0,
            "h2h_draw_rate": None,
            "h2h_goals_avg": None,
            "h2h_home_goals_avg": None,
            "h2h_away_goals_avg": None,
            "h2h_home_win_rate": None,
            "h2h_last_result": None,
            "h2h_goals_diff": None,
            "h2h_btts_rate": None,
            "h2h_over25_rate": None,
        }

        pair_mask = (
            ((self.df["home_team"] == home_team) & (self.df["away_team"] == away_team)) |
            ((self.df["home_team"] == away_team) & (self.df["away_team"] == home_team))
        )

        # Filter to only matches BEFORE the target date (prevent temporal leakage)
        if match_date is not None and "match_date" in self.df.columns:
            date_mask = self.df["match_date"] < pd.Timestamp(match_date)
            h2h = self.df[pair_mask & date_mask].sort_values("match_date", ascending=False).head(10)
        else:
            h2h = self.df[pair_mask].sort_values("match_date", ascending=False).head(10)

        if len(h2h) == 0:
            return defaults

        home_wins = 0
        away_wins = 0
        draws = 0
        home_goals_total = 0
        away_goals_total = 0
        last_result = None
        # Track per-match outcomes for recent-5 and weighted calculations
        match_outcomes = []  # 1=home_win, 0=draw, -1=away_win (most recent first)

        for _, match in h2h.iterrows():
            hs = match.get("home_score")
            as_ = match.get("away_score")
            if pd.isna(hs) or pd.isna(as_):
                continue
            hs, as_ = int(hs), int(as_)

            if match["home_team"] == home_team:
                home_goals_total += hs
                away_goals_total += as_
                if hs > as_:
                    home_wins += 1
                    match_outcomes.append(1)
                    if last_result is None:
                        last_result = 1
                elif hs < as_:
                    away_wins += 1
                    match_outcomes.append(-1)
                    if last_result is None:
                        last_result = -1
                else:
                    draws += 1
                    match_outcomes.append(0)
                    if last_result is None:
                        last_result = 0
            else:
                home_goals_total += as_
                away_goals_total += hs
                if as_ > hs:
                    home_wins += 1
                    match_outcomes.append(1)
                    if last_result is None:
                        last_result = 1
                elif as_ < hs:
                    away_wins += 1
                    match_outcomes.append(-1)
                    if last_result is None:
                        last_result = -1
                else:
                    draws += 1
                    match_outcomes.append(0)
                    if last_result is None:
                        last_result = 0

        played = home_wins + away_wins + draws

        # Recent 5: win rate from only the last 5 H2H matches
        recent_5 = match_outcomes[:5]
        recent_5_wins = sum(1 for r in recent_5 if r == 1)
        recent_5_rate = recent_5_wins / len(recent_5) if recent_5 else None

        # Weighted: exponential decay (most recent matches weighted more)
        if match_outcomes:
            weights = [0.9 ** i for i in range(len(match_outcomes))]
            weighted_wins = sum(w for w, r in zip(weights, match_outcomes) if r == 1)
            weighted_rate = weighted_wins / sum(weights)
        else:
            weighted_rate = None

        # Compute goal diffs and BTTS/Over2.5 from perspective of home_team
        goal_diffs = []
        btts_count = 0
        over25_count = 0
        for _, match in h2h.iterrows():
            hs = match.get("home_score")
            as_ = match.get("away_score")
            if pd.isna(hs) or pd.isna(as_):
                continue
            hs, as_ = int(hs), int(as_)
            if match["home_team"] == home_team:
                goal_diffs.append(hs - as_)
            else:
                goal_diffs.append(as_ - hs)
            if hs > 0 and as_ > 0:
                btts_count += 1
            if hs + as_ > 2:
                over25_count += 1

        scored_matches = len(goal_diffs)

        h2h_home_avg = home_goals_total / played if played > 0 else None
        h2h_away_avg = away_goals_total / played if played > 0 else None

        return {
            "h2h_matches_played": played,
            "h2h_home_wins": home_wins,
            "h2h_away_wins": away_wins,
            "h2h_draws": draws,
            "h2h_draw_rate": round(draws / played, 4) if played > 0 else None,
            "h2h_goals_avg": round(h2h_home_avg + h2h_away_avg, 4) if h2h_home_avg is not None else None,
            "h2h_home_goals_avg": h2h_home_avg,
            "h2h_away_goals_avg": h2h_away_avg,
            "h2h_home_win_rate": home_wins / played if played > 0 else None,
            "h2h_last_result": last_result,
            "h2h_recent_5_win_rate": recent_5_rate,
            "h2h_weighted_home_win_rate": weighted_rate,
            "h2h_goals_diff": round(np.mean(goal_diffs), 2) if goal_diffs else None,
            "h2h_btts_rate": round(btts_count / scored_matches, 3) if scored_matches else None,
            "h2h_over25_rate": round(over25_count / scored_matches, 3) if scored_matches else None,
        }

    def _get_contextual_features(self, features: Dict) -> Dict:
        """Calculate contextual features from base features."""
        contextual = {}

        # Position gap (using Elo as proxy)
        home_elo = features.get("home_elo", 1500)
        away_elo = features.get("away_elo", 1500)
        contextual["position_gap"] = abs(home_elo - away_elo) / 100

        # Mismatch score
        contextual["mismatch_score"] = (home_elo - away_elo) / 200

        # Form trends
        home_form_3 = features.get("home_form_points_3", 4.5)
        home_form_5 = features.get("home_form_points_5", 7.5)
        away_form_3 = features.get("away_form_points_3", 4.5)
        away_form_5 = features.get("away_form_points_5", 7.5)

        contextual["home_form_trend"] = home_form_3 / 3 - home_form_5 / 5 if home_form_5 > 0 else 0
        contextual["away_form_trend"] = away_form_3 / 3 - away_form_5 / 5 if away_form_5 > 0 else 0

        # Clean sheet signals
        contextual["home_clean_sheet_signal"] = features.get("home_clean_sheet_rate", 0.3) * (1 - features.get("away_scoring_rate", 0.7))
        contextual["away_clean_sheet_signal"] = features.get("away_clean_sheet_rate", 0.25) * (1 - features.get("home_scoring_rate", 0.75))

        # Recent big win (proxy from goal difference in form)
        home_gd = features.get("home_roll_3_goals_scored", 4) - features.get("home_roll_3_goals_conceded", 3)
        away_gd = features.get("away_roll_3_goals_scored", 3) - features.get("away_roll_3_goals_conceded", 4)
        contextual["home_recent_big_win"] = 1 if home_gd >= 3 else 0
        contextual["away_recent_big_win"] = 1 if away_gd >= 3 else 0

        return contextual


# =============================================================================
# OVER/UNDER ML PREDICTOR
# =============================================================================

class OverUnderPredictor:
    """Loads and runs dedicated O/U CatBoost binary classifiers.

    Produces P(over line) for each goal line (2.5 by default).
    Trained by scripts/models/train_over_under.py.
    """

    def __init__(self):
        self.models: Dict[float, Any] = {}   # line -> CatBoostClassifier
        self.feature_names: Dict[float, list] = {}  # line -> feature list
        self.loaded = False

    def load_models(self) -> bool:
        """Load O/U models from data/models/universal/over_under/."""
        try:
            from catboost import CatBoostClassifier

            ou_dir = MODELS_DIR / "universal" / "over_under"
            if not ou_dir.exists():
                log.info("O/U model directory not found: %s", ou_dir)
                return False

            for meta_path in sorted(ou_dir.glob("ou_*_catboost_metadata.json")):
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    line = float(meta["line"])
                    line_str = str(line).replace(".", "_")
                    model_path = ou_dir / f"ou_{line_str}_catboost_latest.cbm"
                    if not model_path.exists():
                        log.warning("O/U model file missing: %s", model_path)
                        continue

                    model = CatBoostClassifier()
                    model.load_model(str(model_path))
                    self.models[line] = model
                    self.feature_names[line] = meta.get("feature_names", [])
                    log.info(
                        "Loaded O/U %.1f model (%d features, CV ll=%.4f)",
                        line, len(self.feature_names[line]),
                        meta.get("cv_metrics", {}).get("overall_log_loss", 0),
                    )
                except Exception as e:
                    log.warning("Failed to load O/U model %s: %s", meta_path.name, e)

            self.loaded = len(self.models) > 0
            return self.loaded

        except ImportError:
            log.warning("CatBoost not installed — O/U ML predictor unavailable")
            return False

    def predict(self, features: pd.DataFrame, line: float = 2.5) -> Optional[float]:
        """Return P(over line) or None if model unavailable."""
        if not self.loaded or line not in self.models:
            return None

        try:
            model = self.models[line]
            feat_names = self.feature_names[line]

            available = [f for f in feat_names if f in features.columns]
            missing = [f for f in feat_names if f not in features.columns]

            X = features[available].copy()
            for col in missing:
                X[col] = 0.0

            # Ensure column order matches training
            X = X[feat_names]

            proba = model.predict_proba(X)
            return float(proba[0][1])  # P(over)

        except Exception as e:
            log.error("O/U prediction failed for line %.1f: %s", line, e)
            return None

    def predict_all_lines(self, features: pd.DataFrame) -> Dict[float, float]:
        """Return {line: P(over)} for all loaded models."""
        results = {}
        for line in self.models:
            p = self.predict(features, line)
            if p is not None:
                results[line] = round(p, 4)
        return results


# =============================================================================
# ENSEMBLE PREDICTOR
# =============================================================================

class EnsemblePredictor:
    """Combines Factor, xG, ML, and Player-level predictions."""

    def __init__(self, weights: Dict[str, float] = None, strategy: str = "default",
                 live_mode: bool = True, league: str = "serie_a"):
        self.strategy = strategy
        self.league = league
        # live_mode=True logs predictions to the ledger for rolling corrections.
        # Set to False for backtests / analysis to avoid polluting the ledger.
        self.live_mode = live_mode

        if weights is not None:
            self.weights = weights
        else:
            # Priority: feedback-optimized (if active) > hardcoded (backtest-validated)
            # NOTE: BETTING_STRATEGIES weights are NOT used — they are stale relics
            # from an earlier phase with market=0.22 instead of the validated 0.45.
            # ENSEMBLE_WEIGHTS were optimized via LOO CV on 989 matches (2023-2026).
            optimized = self._load_optimized_weights()
            if optimized:
                self.weights = optimized
            else:
                self.weights = ENSEMBLE_WEIGHTS

        self.xg_predictor = XGPredictor()
        self.ml_classifier = MLClassifier(league=league)
        self.draw_detector = DrawDetector()
        self.meta_learner = MetaLearnerCombiner()
        self.xcomp_loader = XCompLoader()
        self.player_xg_predictor = PlayerXGPredictor()
        self.feature_builder = FeatureBuilder(league=league)
        self.formation_db = None

        # Lessons system (loaded lazily via _load_lessons)
        self._lessons_data = None
        self._lessons_dirty = False

        # Phase 4: Market Intelligence
        self.market_intel = None

        # Phase 5: Deep Learning
        self.deep_predictor = None

        # Phase 6: Calibration Pipeline
        self.calibration_pipeline = None

        # Phase 7: Prediction Correction Layer
        self.correction_layer = None

        # O/U ML predictor (binary CatBoost for over/under markets)
        self.ou_predictor = OverUnderPredictor()

        # Market odds (implied probabilities from bookmakers)
        self.market_odds = {}

        # Track which methods are available
        self.available_methods = []
        self.phase4_features = []  # Track Phase 4 feature availability

    @staticmethod
    def _load_optimized_weights() -> Optional[Dict[str, float]]:
        """Load feedback-optimized weights if available and active.

        Returns optimized weights dict or None to fall back to hardcoded.
        Only applies when status == "active" (30+ settled predictions).
        """
        weights_path = DATA_DIR / "feedback" / "optimized_weights.json"
        if not weights_path.exists():
            return None
        try:
            with open(weights_path) as f:
                data = json.load(f)
            if data.get("status") != "active":
                log.info("Feedback weights status=%s, using hardcoded weights",
                         data.get("status", "unknown"))
                return None
            optimized = data.get("optimized_weights", {})
            if not optimized:
                return None
            # Validate: weights must sum to ~1.0
            total = sum(optimized.values())
            if abs(total - 1.0) > 0.05:
                log.warning("Optimized weights sum to %.3f (expected ~1.0), ignoring", total)
                return None
            log.info("Loaded feedback-optimized weights (n_settled=%d): %s",
                     data.get("n_settled", 0), optimized)
            return optimized
        except Exception as e:
            log.debug("Could not load optimized weights: %s", e)
            return None

    def initialize(self) -> bool:
        """Initialize all prediction components."""
        success = True

        # Factor-based is always available (no external dependencies)
        self.available_methods.append("factor")

        # Try to load xG models
        if self.xg_predictor.load_models():
            self.available_methods.append("xg")
        else:
            log.warning("xG models not available - ensemble will use fallback weights")

        # Try to load ML classifier
        if self.ml_classifier.load_model():
            self.available_methods.append("ml")
        else:
            log.warning("ML classifier not available - ensemble will use fallback weights")

        # Try to load draw detector (blends with ensemble draw probability)
        self.draw_detector.load_model()

        # Try to load meta-learner combiner (replaces fixed-weight blending)
        self.meta_learner.load()

        # Try to load extra-competition congestion timeline (Coppa Italia + UCL + UEL)
        self.xcomp_loader.load()

        # Try to load player xG predictor
        if self.player_xg_predictor.load():
            self.available_methods.append("player_xg")
        else:
            log.warning("Player xG predictor not available")

        # Try to load market odds for implied probabilities
        self.market_odds = self._load_market_odds()
        if self.market_odds:
            self.available_methods.append("market")
            log.info(f"Loaded market odds for {len(self.market_odds)} matches")
        else:
            log.warning("Market odds not available - ensemble will exclude market method")

        # Try to load formation database
        if FORMATION_AVAILABLE:
            try:
                self.formation_db = FormationDatabase()
                if not self.formation_db.load():
                    self.formation_db.build_from_data()
                    self.formation_db.save()
                log.info(f"Loaded formation database for {len(self.formation_db.profiles)} teams")
            except Exception as e:
                log.warning(f"Formation database not available: {e}")
                self.formation_db = None

        # Load feature data
        if not self.feature_builder.load_historical():
            log.warning("Historical features not available - ML/xG predictions limited")

        # Phase 4: Load market intelligence
        if MARKET_INTELLIGENCE_AVAILABLE:
            try:
                self.market_intel = MarketIntelligence()
                self.market_intel.load()
                self.phase4_features.append("market_intel")
                log.info("Market intelligence loaded")
            except Exception as e:
                log.warning(f"Market intelligence not available: {e}")

        # Track other Phase 4 feature availability
        if ENHANCED_MOMENTUM_AVAILABLE:
            self.phase4_features.append("enhanced_momentum")
        if ENHANCED_WEATHER_AVAILABLE:
            self.phase4_features.append("enhanced_weather")
        if SENTIMENT_AVAILABLE:
            self.phase4_features.append("sentiment")

        # Phase 5: Deep Learning (skip if weight is 0 — saves memory + init time)
        if DEEP_LEARNING_AVAILABLE and self.weights.get("deep", 0) > 0:
            try:
                self.deep_predictor = DeepPredictor()
                if self.deep_predictor.load():
                    self.available_methods.append("deep")
                    log.info("Deep learning models loaded")
                else:
                    log.warning("Deep learning models not trained yet")
            except Exception as e:
                log.warning(f"Deep learning not available: {e}")

        # Phase 6: Calibration Pipeline (Draw Detection, Confidence Filtering, Home Calibration)
        if CALIBRATION_AVAILABLE:
            try:
                self.calibration_pipeline = CalibrationPipeline(self.strategy)
                self.phase4_features.append("calibration")
                log.info(f"Calibration pipeline loaded with strategy: {self.strategy}")
            except Exception as e:
                log.warning(f"Calibration pipeline not available: {e}")

        # Phase 7: Prediction Correction Layer (Static + Rolling bias correction)
        if CORRECTION_LAYER_AVAILABLE:
            try:
                self.correction_layer = CorrectionLayer()
                if self.correction_layer.load():
                    self.phase4_features.append("correction_layer")
                    log.info("Correction layer loaded (static=%s, rolling_buckets=%d)",
                             self.correction_layer.static.is_fitted,
                             len(self.correction_layer.rolling.buckets))
                    # Disable LiveBiasCorrector when correction layer is active
                    # to avoid double correction. Setting _loaded=True prevents re-load,
                    # _corrections=None makes active→False and correct()→passthrough.
                    if (self.calibration_pipeline and
                            hasattr(self.calibration_pipeline, 'live_bias')):
                        lbc = self.calibration_pipeline.live_bias
                        lbc._loaded = True
                        lbc._corrections = None
                        lbc._sample_count = 0
                        log.info("LiveBiasCorrector disabled (superseded by correction layer)")
                else:
                    self.correction_layer = None
                    log.info("Correction layer not trained yet — passthrough mode")
            except Exception as e:
                self.correction_layer = None
                log.warning(f"Correction layer not available: {e}")

        # O/U ML predictor (binary classifier for over/under markets)
        if self.ou_predictor.load_models():
            log.info("O/U ML predictor loaded (%d lines)", len(self.ou_predictor.models))
        else:
            log.info("O/U ML predictor not available — will use Poisson-only for O/U")

        log.info(f"Ensemble initialized with methods: {self.available_methods}")
        log.info(f"Phase 4 features: {self.phase4_features}")
        return len(self.available_methods) > 0

    def predict(
        self,
        match: Dict,
        factors: Dict,
        form_data: Dict,
        include_components: bool = True,
        confirmed_lineups: Dict = None,
    ) -> Dict:
        """Generate ensemble prediction for a match."""
        # Auto-initialize if not already done
        if not self.available_methods:
            self.initialize()

        home = match["home_team"]
        away = match["away_team"]

        predictions = {}
        component_probs = {}

        # 0. FETCH MARKET PROBS FIRST (needed as anchor for factor predictor)
        market_probs = None
        if "market" in self.available_methods:
            market_probs = self._get_market_probs(home, away)
            if market_probs:
                predictions["market"] = market_probs
                component_probs["market"] = market_probs

        # 1. FACTOR-BASED PREDICTION (anchored on market when available)
        factor_probs = self._get_factor_probabilities(factors, market_probs=market_probs)
        predictions["factor"] = factor_probs
        component_probs["factor"] = factor_probs

        # Build features ONCE for both xG and ML (avoid double I/O + inject odds + referee)
        match_features = None
        if "xg" in self.available_methods or "ml" in self.available_methods:
            match_features = self.feature_builder.build_match_features(
                home, away, form_data, match_date=match.get("date")
            )
            if match_features is not None:
                match_features = self._inject_odds_into_features(match_features, home, away)
                # Inject computed referee features (same 17 features as training)
                referee_name = match.get("referee", "")
                if referee_name:
                    try:
                        from scripts.prediction.referee_integration import get_referee_features_for_prediction
                        ref_feats = get_referee_features_for_prediction(referee_name, home, away)
                        for feat_name, feat_val in ref_feats.items():
                            match_features[feat_name] = feat_val
                        # Also compute the 4 comprehensive_markets referee features
                        # from referee_assignments.parquet (ref_matches, ref_avg_yellows_given, etc.)
                        try:
                            ref_assign_path = DATA_DIR / "external" / "referee" / "referee_assignments.parquet"
                            if ref_assign_path.exists():
                                _ref_df = pd.read_parquet(ref_assign_path)
                                _ref_matches = _ref_df[_ref_df["referee"] == referee_name]
                                if len(_ref_matches) > 0:
                                    match_features["ref_matches"] = float(len(_ref_matches))
                                    match_features["ref_avg_yellows_given"] = float(_ref_matches["ref_yellows"].mean())
                                    match_features["ref_avg_reds_given"] = float(_ref_matches["ref_reds"].mean())
                                    match_features["ref_total_cards_mean"] = float(
                                        _ref_matches["ref_yellows"].mean() + _ref_matches["ref_reds"].mean()
                                    )
                        except Exception as e:
                            log.debug(f"Failed to compute referee stats: {e}")
                    except Exception as e:
                        log.debug(f"Failed to load referee assignments: {e}")

                # Inject extra-competition congestion features (xc_home_p4, xc_away_p4)
                # Walk-forward validated: LL -0.0025 vs baseline (5 seasons, 36 configs)
                if self.xcomp_loader.loaded:
                    match_date = match.get("date") or pd.Timestamp.now()
                    match_features["xc_home_p4"] = self.xcomp_loader.get_p4(home, match_date)
                    match_features["xc_away_p4"] = self.xcomp_loader.get_p4(away, match_date)

                # Inject formation matchup advantage (imp=0.47, 100% non-zero in training)
                if self.formation_db is not None:
                    try:
                        fm_result = self.formation_db.get_matchup_advantage(home, away)
                        if fm_result and "total_advantage" in fm_result:
                            match_features["formation_total_advantage"] = float(fm_result["total_advantage"])
                    except Exception as e:
                        log.debug(f"Failed to inject formation matchup advantage: {e}")

        # 2. XG-BASED PREDICTION
        if "xg" in self.available_methods and match_features is not None:
            xg_result = self.xg_predictor.predict(match_features)
            if xg_result:
                predictions["xg"] = {
                    "prob_H": xg_result["prob_H"],
                    "prob_D": xg_result["prob_D"],
                    "prob_A": xg_result["prob_A"],
                }
                component_probs["xg"] = predictions["xg"]
                component_probs["xg_details"] = {
                    "home_xg": xg_result["home_xg"],
                    "away_xg": xg_result["away_xg"],
                }

        # 3. ML CLASSIFIER PREDICTION
        if "ml" in self.available_methods and match_features is not None:
            ml_result = self.ml_classifier.predict(match_features)
            if ml_result:
                predictions["ml"] = ml_result
                component_probs["ml"] = ml_result

        # 4. PLAYER XG PREDICTION
        if "player_xg" in self.available_methods:
            player_result = self.player_xg_predictor.predict(
                home, away, confirmed_lineups=confirmed_lineups
            )
            if player_result:
                predictions["player_xg"] = {
                    "prob_H": player_result["prob_H"],
                    "prob_D": player_result["prob_D"],
                    "prob_A": player_result["prob_A"],
                }
                component_probs["player_xg"] = predictions["player_xg"]
                component_probs["player_xg_details"] = {
                    "home_lineup_xg": player_result["home_lineup_xg"],
                    "away_lineup_xg": player_result["away_lineup_xg"],
                }
                component_probs["lineup_source"] = player_result.get(
                    "lineup_source", "predicted"
                )

        # 5. DEEP LEARNING PREDICTION (Phase 5)
        if "deep" in self.available_methods and self.deep_predictor:
            try:
                deep_result = self.deep_predictor.predict(home, away, self.feature_builder.df)
                if deep_result:
                    predictions["deep"] = {
                        "prob_H": deep_result["prob_H"],
                        "prob_D": deep_result["prob_D"],
                        "prob_A": deep_result["prob_A"],
                    }
                    component_probs["deep"] = predictions["deep"]
            except Exception as e:
                log.debug(f"Deep prediction failed: {e}")

        # 6. O/U ML PREDICTION (binary CatBoost for over/under markets)
        if self.ou_predictor.loaded and match_features is not None:
            ou_ml_probs = self.ou_predictor.predict_all_lines(match_features)
            if ou_ml_probs:
                component_probs["over_under_ml"] = ou_ml_probs

        # 7. MARKET IMPLIED PROBABILITIES — already fetched at step 0

        # COMBINE PREDICTIONS
        lineup_source = component_probs.get("lineup_source", "predicted")
        ensemble_probs = self._combine_predictions(predictions, lineup_source=lineup_source)

        # 7. DRAW DETECTOR BLEND — RE-ENABLED (Mar 2026)
        # Fresh ablation (3 seasons, 1050 matches): avg LL improvement +0.0037,
        # avg accuracy +0.43pp. Consistent LL gain in all 3 test seasons.
        if self.draw_detector.loaded and match_features is not None:
            ensemble_probs = self.draw_detector.blend_draw_prob(ensemble_probs, match_features)

        # Apply formation-based probability adjustment
        formation_adjustment = None
        if self.formation_db:
            try:
                formation_data = self.formation_db.get_matchup_advantage(home, away)
                conf = formation_data.get("confidence", "low")
                if conf in ("high", "medium"):
                    fm_home = formation_data["matchup_home_win_rate"]
                    fm_draw = formation_data["matchup_draw_rate"]
                    fm_away = formation_data["matchup_away_win_rate"]

                    # Strength: high-conf (n>=20) = 0.15, medium (n>=10) = 0.08
                    strength = 0.15 if conf == "high" else 0.08

                    adj_h = (fm_home - 0.45) * strength
                    adj_d = (fm_draw - 0.27) * strength
                    adj_a = (fm_away - 0.28) * strength

                    # Cap total formation adjustment at ±5pp per outcome
                    adj_h = max(-0.05, min(0.05, adj_h))
                    adj_d = max(-0.05, min(0.05, adj_d))
                    adj_a = max(-0.05, min(0.05, adj_a))

                    old_H, old_D, old_A = ensemble_probs["prob_H"], ensemble_probs["prob_D"], ensemble_probs["prob_A"]
                    prob_H = max(0.05, old_H + adj_h)
                    prob_D = max(0.05, old_D + adj_d)
                    prob_A = max(0.05, old_A + adj_a)

                    # Normalize (preserve raw_prob_* keys through adjustment)
                    total = prob_H + prob_D + prob_A
                    ensemble_probs = {
                        "prob_H": prob_H / total,
                        "prob_D": prob_D / total,
                        "prob_A": prob_A / total,
                        "raw_prob_H": ensemble_probs.get("raw_prob_H", prob_H / total),
                        "raw_prob_D": ensemble_probs.get("raw_prob_D", prob_D / total),
                        "raw_prob_A": ensemble_probs.get("raw_prob_A", prob_A / total),
                    }

                    formation_adjustment = {
                        "adj_h": round(adj_h, 4),
                        "adj_d": round(adj_d, 4),
                        "adj_a": round(adj_a, 4),
                        "confidence": conf,
                        "strength": strength,
                    }

                    # Log if adjustment flips the predicted outcome
                    old_pred = max(("H", old_H), ("D", old_D), ("A", old_A), key=lambda x: x[1])[0]
                    new_pred = max(
                        ("H", ensemble_probs["prob_H"]),
                        ("D", ensemble_probs["prob_D"]),
                        ("A", ensemble_probs["prob_A"]),
                        key=lambda x: x[1],
                    )[0]
                    if old_pred != new_pred:
                        log.info(f"Formation adjustment flipped {home} vs {away}: {old_pred} -> {new_pred}")
            except Exception as e:
                log.debug(f"Formation adjustment failed: {e}")

        # Apply persistent lessons BEFORE calibration (lessons are raw corrections;
        # calibration should see the corrected probabilities, not override them)
        xg_details = component_probs.get("xg_details", {})
        lesson_home_xg = xg_details.get("home_xg", 1.3)
        lesson_away_xg = xg_details.get("away_xg", 1.1)

        # Determine preliminary prediction for confidence_shift matching
        if ensemble_probs["prob_H"] >= ensemble_probs["prob_D"] and ensemble_probs["prob_H"] >= ensemble_probs["prob_A"]:
            pre_predicted = "HOME"
        elif ensemble_probs["prob_A"] >= ensemble_probs["prob_D"]:
            pre_predicted = "AWAY"
        else:
            pre_predicted = "DRAW"

        lessons_applied = []
        ensemble_probs, lessons_applied = self._apply_lessons(
            ensemble_probs, home, away, pre_predicted,
            home_xg=lesson_home_xg, away_xg=lesson_away_xg,
        )

        # Build features dict for calibration (reuse match_features if available)
        features_dict = {}
        if match_features is not None:
            features_dict = match_features.iloc[0].to_dict() if len(match_features) > 0 else {}
        elif self.feature_builder.df is not None:
            feature_df = self.feature_builder.build_match_features(home, away, form_data)
            if feature_df is not None:
                features_dict = feature_df.iloc[0].to_dict() if len(feature_df) > 0 else {}

        # Phase 6: Apply calibration pipeline (Draw Detection, Home Calibration)
        draw_analysis = None
        if self.calibration_pipeline:
            calibrated_result = self.calibration_pipeline.calibrate_prediction(
                home, away, ensemble_probs, features_dict
            )
            predicted = calibrated_result["predicted_outcome"]
            ensemble_probs = {
                "prob_H": calibrated_result["probabilities"]["home"],
                "prob_D": calibrated_result["probabilities"]["draw"],
                "prob_A": calibrated_result["probabilities"]["away"],
                # Preserve raw_prob_* through calibration — betting uses these
                "raw_prob_H": ensemble_probs.get("raw_prob_H", calibrated_result["probabilities"]["home"]),
                "raw_prob_D": ensemble_probs.get("raw_prob_D", calibrated_result["probabilities"]["draw"]),
                "raw_prob_A": ensemble_probs.get("raw_prob_A", calibrated_result["probabilities"]["away"]),
            }
            max_prob = calibrated_result["confidence"]
            draw_analysis = calibrated_result.get("draw_analysis")
        else:
            if ensemble_probs["prob_H"] >= ensemble_probs["prob_D"] and ensemble_probs["prob_H"] >= ensemble_probs["prob_A"]:
                predicted = "HOME"
            elif ensemble_probs["prob_A"] >= ensemble_probs["prob_D"]:
                predicted = "AWAY"
            else:
                predicted = "DRAW"
            max_prob = max(ensemble_probs["prob_H"], ensemble_probs["prob_D"], ensemble_probs["prob_A"])

        # Phase 7: Apply prediction correction layer (Static + Rolling bias correction)
        correction_deltas = None
        if self.correction_layer and CORRECTION_LAYER_AVAILABLE:
            try:
                context = extract_context_features(
                    ensemble_probs["prob_H"],
                    ensemble_probs["prob_D"],
                    ensemble_probs["prob_A"],
                    features_dict,
                )
                corrected = self.correction_layer.correct(
                    ensemble_probs["prob_H"],
                    ensemble_probs["prob_D"],
                    ensemble_probs["prob_A"],
                    context,
                )
                ensemble_probs["prob_H"] = corrected["prob_H"]
                ensemble_probs["prob_D"] = corrected["prob_D"]
                ensemble_probs["prob_A"] = corrected["prob_A"]

                # Re-derive predicted outcome and confidence from corrected probs
                probs = [corrected["prob_H"], corrected["prob_D"], corrected["prob_A"]]
                max_prob = max(probs)
                max_idx = probs.index(max_prob)
                predicted = ["HOME", "DRAW", "AWAY"][max_idx]

                correction_deltas = {
                    "delta_H": corrected["delta_H"],
                    "delta_D": corrected["delta_D"],
                    "delta_A": corrected["delta_A"],
                }
            except Exception as e:
                log.debug("Correction layer failed for %s vs %s: %s", home, away, e)

        # Log to prediction ledger for rolling updates (live mode only —
        # backtests would pollute the ledger with thousands of non-live entries)
        if CORRECTION_LAYER_AVAILABLE and self.live_mode:
            try:
                # Get match date — try match dict first, then match_features
                match_date_str = ""
                md = match.get("date", "")
                if md:
                    match_date_str = str(md)[:10]
                elif match_features is not None and len(match_features) > 0:
                    row = match_features.iloc[0]
                    md = row["match_date"] if "match_date" in row.index else ""
                    match_date_str = str(md)[:10] if md else ""
                append_to_ledger(
                    home, away, match_date_str,
                    ensemble_probs["prob_H"], ensemble_probs["prob_D"], ensemble_probs["prob_A"],
                    predicted, max_prob, correction_deltas,
                )
            except Exception as e:
                log.debug("Prediction ledger write failed: %s", e)

        result = {
            "match": f"{home} vs {away}",
            "home_team": home,
            "away_team": away,
            "predicted_outcome": predicted,
            "probabilities": {
                "home": round(ensemble_probs["prob_H"], 3),
                "draw": round(ensemble_probs["prob_D"], 3),
                "away": round(ensemble_probs["prob_A"], 3),
            },
            # Raw probabilities without draw boost or temperature scaling.
            # Betting system uses these to avoid inflated edges from accuracy-tuned boosts.
            "betting_probabilities": {
                "home": round(ensemble_probs.get("raw_prob_H", ensemble_probs["prob_H"]), 4),
                "draw": round(ensemble_probs.get("raw_prob_D", ensemble_probs["prob_D"]), 4),
                "away": round(ensemble_probs.get("raw_prob_A", ensemble_probs["prob_A"]), 4),
            },
            "confidence": max_prob,
            "methods_used": list(predictions.keys()),
            "weights_applied": self._get_effective_weights(predictions),
            "strategy": self.strategy,
            "lineup_source": lineup_source,
        }

        # Add draw analysis if available
        if draw_analysis:
            result["draw_analysis"] = {
                "draw_score": draw_analysis.get("draw_score", 0),
                "is_draw_candidate": draw_analysis.get("is_draw_candidate", False),
                "indicators": [i[0] for i in draw_analysis.get("indicators", [])],
            }

        # Add formation adjustment details if applied
        if formation_adjustment:
            result["formation_adjustment"] = formation_adjustment

        # Add lessons applied
        if lessons_applied:
            result["lessons_applied"] = lessons_applied

        # Add correction layer deltas
        if correction_deltas:
            result["correction_deltas"] = correction_deltas

        if include_components:
            result["component_predictions"] = component_probs

        # Add market-implied probabilities for downstream value detection
        market_match_probs = self.market_odds.get(f"{home} vs {away}")
        if market_match_probs:
            result["market_implied"] = {
                "home": round(market_match_probs["prob_H"], 3),
                "draw": round(market_match_probs["prob_D"], 3),
                "away": round(market_match_probs["prob_A"], 3),
                "overround": market_match_probs.get("overround", 0),
                "source": market_match_probs.get("source", "odds_average"),
            }
            # Edge = our probability minus market probability for predicted outcome
            our_prob = result["probabilities"].get(
                {"HOME": "home", "DRAW": "draw", "AWAY": "away"}.get(result["predicted_outcome"], "home"), 0
            )
            market_prob = result["market_implied"].get(
                {"HOME": "home", "DRAW": "draw", "AWAY": "away"}.get(result["predicted_outcome"], "home"), 0
            )
            result["market_edge"] = round(our_prob - market_prob, 3)

        # Detect market anomalies (cup matches, corrupted data)
        try:
            from scripts.data.odds_fetcher import detect_market_anomaly
            if self.league == "serie_a":
                odds_path = DATA_DIR / "upcoming" / "odds_full.json"
            else:
                odds_path = DATA_DIR / "upcoming" / f"odds_full_{self.league}.json"
            if odds_path.exists():
                with open(odds_path) as f:
                    odds_full = json.load(f)
                match_odds = odds_full.get("matches", {}).get(f"{home} vs {away}", {})
                if match_odds:
                    anomaly = detect_market_anomaly(match_odds)
                    if anomaly["is_anomalous"]:
                        result["market_anomaly"] = anomaly
                        if anomaly["severity"] >= 2:
                            log.warning(f"ANOMALY: {home} vs {away}: {anomaly['reasons']}")
        except Exception as e:
            log.warning(f"Failed to detect market anomaly for {home} vs {away}: {e}")

        # Add market intelligence from aggregated signals
        try:
            from features.market_intelligence import get_match_intelligence
            intel = get_match_intelligence(f"{home} vs {away}")
            if intel and intel.get("signals_available", 0) > 0:
                result["market_intelligence"] = {
                    "sharp_direction": intel.get("sharp_direction", "neutral"),
                    "divergence": intel.get("divergence", 0),
                    "steam_detected": intel.get("steam_detected", False),
                    "offensive_dominance": intel.get("offensive_dominance"),
                    "market_confidence": intel.get("market_confidence", 0.5),
                    "composite_score": intel.get("composite_score", 0),
                }
        except Exception as e:
            log.debug(f"Market intelligence not available for {home} vs {away}: {e}")

        # Add formation analysis if available
        if self.formation_db:
            try:
                formation_data = self.formation_db.get_matchup_advantage(home, away)
                result["formation_analysis"] = {
                    "home_formation": formation_data["home_formation"],
                    "away_formation": formation_data["away_formation"],
                    "matchup_advantage": formation_data["total_advantage"],
                    "confidence": formation_data["confidence"],
                }
            except Exception as e:
                log.debug(f"Failed to add formation analysis for {home} vs {away}: {e}")

        # Situational context for downstream betting adjustments
        # (rest days, international break, congestion from features.parquet)
        if match_features is not None and len(match_features) > 0:
            mf = match_features.iloc[0]
            result["situational_context"] = {
                "home_rest_days": float(mf.get("home_rest_days", 7)) if pd.notna(mf.get("home_rest_days")) else None,
                "away_rest_days": float(mf.get("away_rest_days", 7)) if pd.notna(mf.get("away_rest_days")) else None,
                "rest_advantage": float(mf.get("rest_advantage", 0)) if pd.notna(mf.get("rest_advantage")) else None,
                "home_post_intl_break": bool(mf.get("home_post_intl_break", 0)),
                "away_post_intl_break": bool(mf.get("away_post_intl_break", 0)),
                "congestion_asymmetry": float(mf.get("congestion_asymmetry", 0)) if pd.notna(mf.get("congestion_asymmetry")) else None,
            }

        # Phase 4: Add market intelligence
        if self.market_intel and "market_intel" in self.phase4_features:
            try:
                model_probs = {
                    "home": ensemble_probs["prob_H"],
                    "draw": ensemble_probs["prob_D"],
                    "away": ensemble_probs["prob_A"],
                }
                market_data = self.market_intel.analyze_match(home, away, model_probs)
                if market_data.get("market_available"):
                    result["market_intelligence"] = {
                        "sharp_score": round(market_data.get("sharp_score", 0), 3),
                        "movement_magnitude": round(market_data.get("movement_magnitude", 0), 2),
                        "market_confidence": round(market_data.get("market_confidence", 0.5), 3),
                        "has_value": market_data.get("has_value", 0),
                        "best_bet": market_data.get("best_bet", "none"),
                        "home_edge": round(market_data.get("home_edge", 0), 3),
                        "away_edge": round(market_data.get("away_edge", 0), 3),
                    }
            except Exception as e:
                log.debug(f"Market intelligence failed: {e}")

        # Phase 4: Add enhanced momentum
        if ENHANCED_MOMENTUM_AVAILABLE and "enhanced_momentum" in self.phase4_features:
            try:
                if self.feature_builder.df is not None:
                    match_date = pd.Timestamp.now()
                    home_momentum = compute_big_win_momentum(
                        self.feature_builder.df, home, match_date
                    )
                    away_momentum = compute_big_win_momentum(
                        self.feature_builder.df, away, match_date
                    )
                    result["momentum_analysis"] = {
                        "home_big_win_recency": round(home_momentum["big_win_recency"], 2),
                        "away_big_win_recency": round(away_momentum["big_win_recency"], 2),
                        "home_dominant_ratio": round(home_momentum["dominant_win_ratio"], 3),
                        "away_dominant_ratio": round(away_momentum["dominant_win_ratio"], 3),
                    }
            except Exception as e:
                log.debug(f"Enhanced momentum failed: {e}")

        # Phase 4: Add sentiment analysis
        if SENTIMENT_AVAILABLE and "sentiment" in self.phase4_features:
            try:
                # Get league positions from form data
                key = f"{home} vs {away}"
                matchup = form_data.get("matchups", {}).get(key, {})
                home_pos = matchup.get("home_form", {}).get("league_position", 10)
                away_pos = matchup.get("away_form", {}).get("league_position", 10)

                # Get recent results
                home_results = matchup.get("home_form", {}).get("form_string", "")
                away_results = matchup.get("away_form", {}).get("form_string", "")

                sentiment = get_match_sentiment_features(
                    home, away,
                    home_position=home_pos,
                    away_position=away_pos,
                    home_results=list(home_results),
                    away_results=list(away_results),
                )
                result["sentiment_analysis"] = {
                    "home_motivation": round(sentiment.get("home_motivation_factor", 0.5), 3),
                    "away_motivation": round(sentiment.get("away_motivation_factor", 0.5), 3),
                    "sentiment_diff": round(sentiment.get("sentiment_diff", 0), 3),
                    "home_pressure": round(sentiment.get("home_pressure_factor", 0.5), 3),
                    "away_pressure": round(sentiment.get("away_pressure_factor", 0.5), 3),
                }
            except Exception as e:
                log.debug(f"Sentiment analysis failed: {e}")

        return result

    # Mapping from The Odds API bookmaker names → football-data.co.uk column prefixes.
    # Training data uses: B365, PS (Pinnacle), BW (Betway), BF (Betfair Exchange),
    # Max (best across all), Avg (market average).
    _BOOKMAKER_TO_PREFIX = {
        "Pinnacle": "PS",
        "888sport": "B365",        # Closest proxy for Bet365 (same parent company)
        "Bet365": "B365",          # Direct match if available
        "William Hill": "WH",
        "Betway": "BW",
        "Betfair Sportsbook": "BFD",
        "Betfair": "BFE",          # Betfair Exchange
        "Ladbrokes": "LB",
        "Unibet (UK)": "UN",
        "Coral": "CL",
    }

    def _inject_odds_into_features(self, features: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
        """Inject current market odds into feature DataFrame.

        Reads per-bookmaker data from odds_bookmakers.json and maps each
        bookmaker to the correct training column name (B365, PS, BW, BF, Max, Avg).
        This ensures sharp-soft divergence features work correctly at prediction time.
        """
        match_key = f"{home} vs {away}"

        # --- Step 1: Load per-bookmaker odds ---
        bookmaker_odds = []
        bk_path = DATA_DIR / "upcoming" / "odds_bookmakers.json"
        if bk_path.exists():
            try:
                with open(bk_path) as f:
                    bk_data = json.load(f)
                match_bk = bk_data.get("matches", {}).get(match_key, {})
                bookmaker_odds = match_bk.get("h2h", [])
            except Exception as e:
                log.warning(f"Failed to load bookmaker odds for {match_key}: {e}")

        # Fallback: try odds_full.json (league-aware)
        if not bookmaker_odds:
            if self.league == "serie_a":
                full_path = DATA_DIR / "upcoming" / "odds_full.json"
            else:
                full_path = DATA_DIR / "upcoming" / f"odds_full_{self.league}.json"
            if full_path.exists():
                try:
                    with open(full_path) as f:
                        full_data = json.load(f)
                    match_full = full_data.get("matches", {}).get(match_key, {})
                    h2h = match_full.get("h2h", {})
                    if isinstance(h2h, dict):
                        bookmaker_odds = h2h.get("all_bookmakers", [])
                except Exception as e:
                    log.warning(f"Failed to load fallback odds_full.json for {match_key}: {e}")

        if bookmaker_odds:
            # Map specific bookmakers to their training column names
            all_h, all_d, all_a = [], [], []
            pinnacle_set = False

            for bk in bookmaker_odds:
                bk_name = bk.get("bookmaker", "")
                h, d, a = bk.get("home", 0), bk.get("draw", 0), bk.get("away", 0)
                if not (h > 1 and d > 1 and a > 1):
                    continue

                all_h.append(h)
                all_d.append(d)
                all_a.append(a)

                prefix = self._BOOKMAKER_TO_PREFIX.get(bk_name)
                if prefix:
                    features[f"odds_{prefix}H"] = h
                    features[f"odds_{prefix}D"] = d
                    features[f"odds_{prefix}A"] = a
                    if prefix == "PS":
                        pinnacle_set = True

            # Compute Max (best odds = highest price per outcome)
            if all_h:
                features["odds_MaxH"] = round(max(all_h), 2)
                features["odds_MaxD"] = round(max(all_d), 2)
                features["odds_MaxA"] = round(max(all_a), 2)

            # Compute Avg (market average)
            if all_h:
                features["odds_AvgH"] = round(sum(all_h) / len(all_h), 2)
                features["odds_AvgD"] = round(sum(all_d) / len(all_d), 2)
                features["odds_AvgA"] = round(sum(all_a) / len(all_a), 2)

            # If no Pinnacle found, use best available sharp book as proxy
            if not pinnacle_set and all_h:
                features["odds_PSH"] = features.get("odds_MaxH", round(max(all_h), 2))
                features["odds_PSD"] = features.get("odds_MaxD", round(max(all_d), 2))
                features["odds_PSA"] = features.get("odds_MaxA", round(max(all_a), 2))

            # Pinnacle Closing odds (PSC prefix): model was trained on closing
            # odds but we only have pre-match odds. Use opening as proxy
            # (Pinnacle closing ≈ opening for efficient markets).
            for suffix in ("H", "D", "A"):
                ps_col = f"odds_PS{suffix}"
                psc_col = f"odds_PSC{suffix}"
                if ps_col in features.columns:
                    ps_val = features[ps_col].iloc[0]
                    if not pd.isna(ps_val) and float(ps_val) > 1:
                        features[psc_col] = ps_val

            # If no B365 mapped, use market average as proxy
            if "odds_B365H" not in features.columns or features["odds_B365H"].iloc[0] == 0:
                if all_h:
                    features["odds_B365H"] = round(sum(all_h) / len(all_h), 2)
                    features["odds_B365D"] = round(sum(all_d) / len(all_d), 2)
                    features["odds_B365A"] = round(sum(all_a) / len(all_a), 2)
        else:
            # Last resort: use simple odds.json (averages only)
            odds_path = DATA_DIR / "upcoming" / "odds.json"
            if odds_path.exists():
                try:
                    with open(odds_path) as f:
                        all_odds = json.load(f)
                    raw_odds = all_odds.get(match_key, {})
                    if raw_odds:
                        h = raw_odds.get("home", 0)
                        d = raw_odds.get("draw", 0)
                        a = raw_odds.get("away", 0)
                        if h > 1 and d > 1 and a > 1:
                            features["odds_B365H"] = h
                            features["odds_B365D"] = d
                            features["odds_B365A"] = a
                            features["odds_AvgH"] = h
                            features["odds_AvgD"] = d
                            features["odds_AvgA"] = a
                            features["odds_PSH"] = h
                            features["odds_PSD"] = d
                            features["odds_PSA"] = a
                            features["odds_MaxH"] = h
                            features["odds_MaxD"] = d
                            features["odds_MaxA"] = a
                except Exception as e:
                    log.warning(f"Failed to load raw odds fallback: {e}")

        # --- Step 2: Inject vig-free implied probabilities ---
        # Compute from Pinnacle odds (sharpest) and market average
        for prefix, prob_prefix in [("PS", "pinnacle"), ("Avg", "market")]:
            h_col = f"odds_{prefix}H"
            d_col = f"odds_{prefix}D"
            a_col = f"odds_{prefix}A"
            if all(c in features.columns for c in [h_col, d_col, a_col]):
                h_val = features[h_col].iloc[0] if len(features) else 0
                d_val = features[d_col].iloc[0] if len(features) else 0
                a_val = features[a_col].iloc[0] if len(features) else 0
                if h_val > 1 and d_val > 1 and a_val > 1:
                    raw_h, raw_d, raw_a = 1/h_val, 1/d_val, 1/a_val
                    total = raw_h + raw_d + raw_a
                    if total > 0:
                        features[f"{prob_prefix}_home_prob"] = round(raw_h / total, 4)
                        features[f"{prob_prefix}_draw_prob"] = round(raw_d / total, 4)
                        features[f"{prob_prefix}_away_prob"] = round(raw_a / total, 4)

        # --- Step 2b: Inject Pinnacle line velocity from odds_movement.json ---
        # line_vel_pin_* = (current - opening) / opening for each outcome
        om_path = DATA_DIR / "upcoming" / "odds_movement.json"
        if om_path.exists():
            try:
                with open(om_path) as f:
                    om_data = json.load(f)
                match_om = om_data.get("matches", {}).get(match_key, {})
                for outcome, suffix in [("home", "home"), ("draw", "draw"), ("away", "away")]:
                    vel_col = f"line_vel_pin_{suffix}"
                    if vel_col not in features.columns or features[vel_col].iloc[0] == 0:
                        opening = match_om.get(f"opening_{outcome}", 0)
                        current = match_om.get(f"current_{outcome}", 0)
                        if opening > 1 and current > 1:
                            features[vel_col] = round((current - opening) / opening, 6)
            except Exception as e:
                log.debug(f"Failed to inject Pinnacle line velocity: {e}")

        # Override with bookmaker_analysis.json if available (more precise)
        ba_path = DATA_DIR / "upcoming" / "bookmaker_analysis.json"
        if ba_path.exists():
            try:
                with open(ba_path) as f:
                    ba = json.load(f)
                match_ba = ba.get("matches", {}).get(match_key, {})
                sharp = match_ba.get("sharp_consensus", {})
                mkt = match_ba.get("market_consensus", {})

                if sharp.get("prob_H"):
                    features["pinnacle_home_prob"] = sharp["prob_H"]
                    features["pinnacle_draw_prob"] = sharp["prob_D"]
                    features["pinnacle_away_prob"] = sharp["prob_A"]
                if mkt.get("prob_H"):
                    features["market_home_prob"] = mkt["prob_H"]
                    features["market_draw_prob"] = mkt["prob_D"]
                    features["market_away_prob"] = mkt["prob_A"]
            except Exception as e:
                log.debug(f"Failed to load bookmaker analysis data: {e}")

        # --- Step 3: Compute derived odds features (mirrors odds_features.py) ---
        # These are computed during training but were missing at prediction time,
        # causing interaction features (sharp_soft_x_elo, ah_x_form, etc.) to be 0.

        # Pre-step: derive pinnacle_draw_prob from home+away if missing
        # (must happen BEFORE coalescing *_prob_best, which reads pinnacle_draw_prob)
        if "pinnacle_draw_prob" not in features.columns or pd.isna(features["pinnacle_draw_prob"].iloc[0] if "pinnacle_draw_prob" in features.columns else None):
            if "pinnacle_home_prob" in features.columns and "pinnacle_away_prob" in features.columns:
                pin_h = features["pinnacle_home_prob"].iloc[0]
                pin_a = features["pinnacle_away_prob"].iloc[0]
                if not pd.isna(pin_h) and not pd.isna(pin_a):
                    features["pinnacle_draw_prob"] = round(max(0, 1.0 - float(pin_h) - float(pin_a)), 4)

        # Best-available probabilities: coalesce Pinnacle → market
        for outcome in ("home", "draw", "away"):
            pin_col = f"pinnacle_{outcome}_prob"
            mkt_col = f"market_{outcome}_prob"
            best_col = f"{outcome}_prob_best"
            pin_val = features[pin_col].iloc[0] if pin_col in features.columns else None
            mkt_val = features[mkt_col].iloc[0] if mkt_col in features.columns else None
            if pin_val is not None and not pd.isna(pin_val):
                features[best_col] = pin_val
            elif mkt_val is not None and not pd.isna(mkt_val):
                features[best_col] = mkt_val

        # Sharp-soft divergence (Pinnacle - market average)
        for outcome in ("home", "draw", "away"):
            pin_col = f"pinnacle_{outcome}_prob"
            mkt_col = f"market_{outcome}_prob"
            div_col = f"sharp_soft_{outcome}_div"
            if pin_col in features.columns and mkt_col in features.columns:
                pin_v = features[pin_col].iloc[0]
                mkt_v = features[mkt_col].iloc[0]
                if not pd.isna(pin_v) and not pd.isna(mkt_v):
                    features[div_col] = round(pin_v - mkt_v, 4)

        # Asian handicap normalization (from spreads data)
        if "odds_AHh" in features.columns:
            ah_val = features["odds_AHh"].iloc[0]
            if not pd.isna(ah_val):
                features["ah_line_normalized"] = round(max(-3, min(3, ah_val)), 2)
                features["ah_line_abs"] = round(abs(ah_val), 2)
        else:
            # Try to derive from spreads in odds_bookmakers.json
            try:
                bk_path = DATA_DIR / "upcoming" / "odds_bookmakers.json"
                if bk_path.exists():
                    with open(bk_path) as f:
                        bk_data = json.load(f)
                    match_bk = bk_data.get("matches", {}).get(match_key, {})
                    # Look for spread data (e.g. spreads_0.75, spreads_1.0)
                    for spread_key in sorted(match_bk.keys()):
                        if spread_key.startswith("spreads_"):
                            entries = match_bk[spread_key]
                            if entries and isinstance(entries, list):
                                # Use first entry's home_point as AH line
                                ah_val = entries[0].get("home_point", 0)
                                if ah_val != 0:
                                    features["ah_line_normalized"] = round(max(-3, min(3, ah_val)), 2)
                                    features["ah_line_abs"] = round(abs(ah_val), 2)
                                    features["odds_AHh"] = ah_val
                                    break
            except Exception as e:
                log.debug(f"Failed to inject Asian handicap line: {e}")

        # Odds home favourite flag
        if "home_prob_best" in features.columns and "away_prob_best" in features.columns:
            hp = features["home_prob_best"].iloc[0]
            ap = features["away_prob_best"].iloc[0]
            if not pd.isna(hp) and not pd.isna(ap):
                features["odds_home_fav"] = float(hp > ap)

        # Odds consistency (agreement between sharp and soft books)
        if "pinnacle_home_prob" in features.columns and "market_home_prob" in features.columns:
            pin_h = features["pinnacle_home_prob"].iloc[0]
            mkt_h = features["market_home_prob"].iloc[0]
            if not pd.isna(pin_h) and not pd.isna(mkt_h):
                max_p = max(pin_h, mkt_h, 0.01)
                features["odds_consistency"] = round(1 - abs(pin_h - mkt_h) / max_p, 4)

        # Re-compute interaction features now that odds-derived inputs are available
        # _compute_interaction_features expects a dict, so convert row → dict → back
        if len(features) > 0:
            row_dict = features.iloc[0].to_dict()
            interaction_updates = self.feature_builder._compute_interaction_features(row_dict)
            for k, v in interaction_updates.items():
                features[k] = v

        # --- Step 4: Compute missing differential & derived features ---
        # Collect all new columns in a dict, then assign at once to avoid
        # DataFrame fragmentation (PerformanceWarning from repeated inserts).
        if len(features) > 0:
            row = features.iloc[0]
            new_cols = {}

            # Generic diff: scan home_ss_r5_*/home_ss_shot_r5_* → diff_ss_r5_*/diff_ss_shot_r5_*
            for prefix, diff_prefix in [("home_ss_r5_", "diff_ss_r5_"),
                                        ("home_ss_shot_r5_", "diff_ss_shot_r5_")]:
                away_prefix = prefix.replace("home_", "away_")
                for col in list(features.columns):
                    if not col.startswith(prefix):
                        continue
                    suffix = col[len(prefix):]
                    diff_col = f"{diff_prefix}{suffix}"
                    a_col = f"{away_prefix}{suffix}"
                    if diff_col in features.columns or a_col not in features.columns:
                        continue
                    h_val = row.get(col, 0) or 0
                    a_val = row.get(a_col, 0) or 0
                    try:
                        if not pd.isna(h_val) and not pd.isna(a_val):
                            new_cols[diff_col] = float(h_val) - float(a_val)
                    except (TypeError, ValueError):
                        new_cols[diff_col] = 0.0

            # Transfermarkt squad value ratios
            for metric in ("total", "avg"):
                ratio_col = f"sv_{metric}_ratio"
                if ratio_col not in features.columns:
                    h_val = row.get(f"home_sv_{metric}", 0) or 0
                    a_val = row.get(f"away_sv_{metric}", 0) or 0
                    try:
                        if not pd.isna(h_val) and not pd.isna(a_val):
                            h_f, a_f = float(h_val), float(a_val)
                            new_cols[ratio_col] = round(h_f / a_f, 4) if a_f > 0 else 1.0
                    except (TypeError, ValueError):
                        new_cols[ratio_col] = 1.0

            # Pinnacle draw prob: derive from home + away if missing
            if "pinnacle_draw_prob" not in features.columns:
                pin_h = row.get("pinnacle_home_prob", None)
                pin_a = row.get("pinnacle_away_prob", None)
                if pin_h is not None and pin_a is not None:
                    try:
                        if not pd.isna(pin_h) and not pd.isna(pin_a):
                            new_cols["pinnacle_draw_prob"] = round(
                                max(0, 1.0 - float(pin_h) - float(pin_a)), 4
                            )
                    except (TypeError, ValueError) as e:
                        log.debug(f"Failed to derive Pinnacle draw prob: {e}")

            # League-level stats — compute from latest features data if available,
            # otherwise use Serie A historical averages as fallback.
            # league_avg_goals has CatBoost importance 3.11 (5th highest) —
            # must be dynamic, not static, to match training (build.py:1909-1915).
            _league_avg_goals = 2.67  # fallback
            _league_draw_rate = 0.27
            _league_home_win_rate = 0.43
            if hasattr(self, 'df') and self.df is not None and len(self.df) > 0:
                try:
                    _latest = self.df
                    if "league_avg_goals" in _latest.columns:
                        _val = _latest["league_avg_goals"].dropna().iloc[-1] if len(_latest) > 0 else 2.67
                        _league_avg_goals = round(float(_val), 3)
                    if "league_draw_rate" in _latest.columns:
                        _val = _latest["league_draw_rate"].dropna().iloc[-1] if len(_latest) > 0 else 0.27
                        _league_draw_rate = round(float(_val), 3)
                    if "league_home_win_rate" in _latest.columns:
                        _val = _latest["league_home_win_rate"].dropna().iloc[-1] if len(_latest) > 0 else 0.43
                        _league_home_win_rate = round(float(_val), 3)
                except Exception as e:
                    log.debug("League defaults from features failed: %s", e)
            _LEAGUE_DEFAULTS = {
                "league_home_win_rate": _league_home_win_rate,
                "league_draw_rate": _league_draw_rate,
                "league_avg_goals": _league_avg_goals,
                "matchweek_avg_goals": _league_avg_goals,
            }
            for col, val in _LEAGUE_DEFAULTS.items():
                if col not in features.columns:
                    new_cols[col] = val

            # Context features
            if "form_diff" not in features.columns:
                h_f = row.get("home_roll_5_points", 0) or 0
                a_f = row.get("away_roll_5_points", 0) or 0
                try:
                    new_cols["form_diff"] = float(h_f) - float(a_f) if not pd.isna(h_f) and not pd.isna(a_f) else 0.0
                except (TypeError, ValueError):
                    new_cols["form_diff"] = 0.0
            if "momentum_diff" not in features.columns:
                h_m = row.get("home_roll_3_points", 0) or 0
                a_m = row.get("away_roll_3_points", 0) or 0
                try:
                    new_cols["momentum_diff"] = float(h_m) - float(a_m) if not pd.isna(h_m) and not pd.isna(a_m) else 0.0
                except (TypeError, ValueError):
                    new_cols["momentum_diff"] = 0.0

            # Calendar features
            now = datetime.now()
            _CAL_DEFAULTS = {
                "kickoff_hour": 15, "is_night_match": 0, "is_evening_kickoff": 0,
                "is_weekend": float(now.weekday() >= 5),
                "is_december": float(now.month == 12), "is_may": float(now.month == 5),
                "is_january": float(now.month == 1), "is_august": float(now.month == 8),
                # Training uses matchweek >= 33 (build.py:1443); MW33 ≈ mid-April
                "is_late_season": float(now.month == 5 or (now.month == 4 and now.day >= 15)),
                "is_early_kickoff": 0, "is_monday_night": 0,
            }
            for col, val in _CAL_DEFAULTS.items():
                if col not in features.columns:
                    new_cols[col] = val

            # Formation matchup features
            if self.formation_db and "formation_matchup_home_rate" not in features.columns:
                try:
                    fm = self.formation_db.get_matchup_advantage(
                        str(row.get("home_team", "")), str(row.get("away_team", ""))
                    )
                    new_cols["formation_matchup_home_rate"] = fm.get("matchup_home_win_rate", 0.45)
                    new_cols["formation_matchup_draw_rate"] = fm.get("matchup_draw_rate", 0.27)
                except Exception as e:
                    log.debug("Formation matchup lookup failed: %s", e)
                    new_cols["formation_matchup_home_rate"] = 0.45
                    new_cols["formation_matchup_draw_rate"] = 0.27

            # Batch-assign all new columns at once (avoids fragmentation)
            if new_cols:
                new_df = pd.DataFrame({k: [v] for k, v in new_cols.items()}, index=features.index)
                features = pd.concat([features, new_df], axis=1)

        return features

    def _load_market_odds(self) -> Dict[str, Dict]:
        """Load bookmaker odds and convert to vig-free implied probabilities.

        For Serie A: prefers sharp consensus from bookmaker_analysis.json,
        falls back to odds.json averages.
        For other leagues: reads odds_full_{league}.json.

        Returns dict mapping "Home vs Away" to {prob_H, prob_D, prob_A}.
        """
        market_probs = {}

        # --- Non-Serie A: read from odds_full_{league}.json ---
        if self.league != "serie_a":
            return self._load_league_market_odds()

        # --- Serie A path (unchanged) ---
        # First try sharp consensus from bookmaker analysis
        ba_path = DATA_DIR / "upcoming" / "bookmaker_analysis.json"
        sharp_loaded = 0
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
                            "overround": 2.0,  # sharp books ~2%
                            "source": "sharp_consensus",
                        }
                        sharp_loaded += 1
                if sharp_loaded:
                    log.info(f"Loaded sharp consensus for {sharp_loaded} matches")
            except Exception as e:
                log.debug(f"Could not load bookmaker analysis: {e}")

        # Fall back to odds.json for matches without sharp consensus
        odds_path = DATA_DIR / "upcoming" / "odds.json"
        if not odds_path.exists():
            return market_probs

        try:
            with open(odds_path) as f:
                raw_odds = json.load(f)
        except Exception as e:
            log.warning(f"Could not load odds.json: {e}")
            return market_probs

        fallback_count = 0
        for match_key, odds in raw_odds.items():
            if match_key in market_probs:
                continue  # Already have sharp consensus

            home_odds = odds.get("home", 0)
            draw_odds = odds.get("draw", 0)
            away_odds = odds.get("away", 0)

            if home_odds <= 1.0 or draw_odds <= 1.0 or away_odds <= 1.0:
                continue

            raw_h = 1.0 / home_odds
            raw_d = 1.0 / draw_odds
            raw_a = 1.0 / away_odds
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
            fallback_count += 1

        if fallback_count:
            log.info(f"Loaded average odds fallback for {fallback_count} matches")

        return market_probs

    def _load_league_market_odds(self) -> Dict[str, Dict]:
        """Load market odds from odds_full_{league}.json for non-Serie A leagues.

        The file format is: {matches: {match_key: {h2h: {best_home, best_draw, best_away, ...}}}}
        """
        market_probs = {}
        odds_path = DATA_DIR / "upcoming" / f"odds_full_{self.league}.json"
        if not odds_path.exists():
            log.warning("No odds file for %s at %s", self.league, odds_path)
            return market_probs

        try:
            with open(odds_path) as f:
                odds_data = json.load(f)
        except Exception as e:
            log.warning("Could not load %s odds: %s", self.league, e)
            return market_probs

        matches = odds_data.get("matches", odds_data)
        if isinstance(matches, list):
            matches = {
                m.get("match", f"{m.get('home_team', '?')} vs {m.get('away_team', '?')}"): m
                for m in matches
            }

        for match_key, match_data in matches.items():
            h2h = match_data.get("h2h", {})
            if isinstance(h2h, dict):
                best_h = h2h.get("best_home", h2h.get("home", 0))
                best_d = h2h.get("best_draw", h2h.get("draw", 0))
                best_a = h2h.get("best_away", h2h.get("away", 0))
            elif isinstance(h2h, list):
                best_h = max((bm.get("home", 0) for bm in h2h), default=0)
                best_d = max((bm.get("draw", 0) for bm in h2h), default=0)
                best_a = max((bm.get("away", 0) for bm in h2h), default=0)
            else:
                continue

            if best_h <= 1.0 or best_d <= 1.0 or best_a <= 1.0:
                continue

            raw_h = 1.0 / best_h
            raw_d = 1.0 / best_d
            raw_a = 1.0 / best_a
            total = raw_h + raw_d + raw_a
            if total <= 0:
                continue

            market_probs[match_key] = {
                "prob_H": raw_h / total,
                "prob_D": raw_d / total,
                "prob_A": raw_a / total,
                "overround": round((total - 1.0) * 100, 1),
                "source": "odds_full_best",
                "best_odds": {"home": best_h, "draw": best_d, "away": best_a},
            }

        if market_probs:
            log.info("Loaded market odds for %d %s matches", len(market_probs), self.league)
        return market_probs

    def _get_market_probs(self, home: str, away: str) -> Optional[Dict]:
        """Get market-implied probabilities for a specific match."""
        match_key = f"{home} vs {away}"
        probs = self.market_odds.get(match_key)
        if probs:
            return {
                "prob_H": probs["prob_H"],
                "prob_D": probs["prob_D"],
                "prob_A": probs["prob_A"],
            }
        return None

    def _get_factor_probabilities(self, factors: Dict, market_probs: Dict = None) -> Dict:
        """Convert factor analysis to probabilities.

        When market_probs are available, anchors on them and applies dampened
        factor lifts (since market already prices in most information).
        Without market_probs, falls back to BASE_RATES as anchor.
        """
        if market_probs:
            # Anchor on market-implied probabilities — factors add situational
            # adjustments (weather, referee, derby) that the market may underweight.
            # Dampen lifts by 0.35x since market already captures most team strength.
            DAMPEN = 0.35
            home_prob = market_probs["prob_H"]
            draw_prob = market_probs["prob_D"]
            away_prob = market_probs["prob_A"]
        else:
            # No market data — fall back to league averages (less accurate)
            DAMPEN = 1.0
            home_prob = BASE_RATES["home_win"]
            draw_prob = BASE_RATES["draw"]
            away_prob = BASE_RATES["away_win"]

        # Apply dampened lifts
        home_prob += factors.get("home_lift", 0) * DAMPEN
        away_prob += factors.get("away_lift", 0) * DAMPEN

        # Derby adjustment (dampened)
        if "derby" in factors.get("neutral_factors", []):
            draw_prob += 0.08 * DAMPEN
            home_prob -= 0.04 * DAMPEN
            away_prob -= 0.04 * DAMPEN

        # Clamp to valid range before normalizing
        home_prob = max(0.03, home_prob)
        draw_prob = max(0.03, draw_prob)
        away_prob = max(0.03, away_prob)

        # Normalize
        total = home_prob + draw_prob + away_prob
        return {
            "prob_H": home_prob / total,
            "prob_D": draw_prob / total,
            "prob_A": away_prob / total,
        }

    def _get_effective_weights(self, predictions: Dict) -> Dict:
        """Get effective weights based on available methods."""
        available = set(predictions.keys())
        has_market = "market" in available

        # Full ensemble (6 methods: all + market)
        if available == {"factor", "xg", "ml", "player_xg", "deep", "market"}:
            return ENSEMBLE_WEIGHTS_WITH_DEEP

        # 5 methods with deep but no market
        if available == {"factor", "xg", "ml", "player_xg", "deep"}:
            return {"factor": 0.30, "xg": 0.40, "ml": 0.20, "player_xg": 0.10, "deep": 0.00}

        # 5 methods: all stats + market (no deep) = standard production
        if available == {"factor", "xg", "ml", "player_xg", "market"}:
            # Filter self.weights to only available methods (strategy weights may include 'deep')
            subset = {m: self.weights[m] for m in available if m in self.weights}
            if not subset:
                subset = {m: ENSEMBLE_WEIGHTS[m] for m in available if m in ENSEMBLE_WEIGHTS}
            total = sum(subset.values())
            return {m: w / total for m, w in subset.items()} if total > 0 else ENSEMBLE_WEIGHTS

        # 4 methods without deep or market
        if available == {"factor", "xg", "ml", "player_xg"}:
            return {"factor": 0.35, "xg": 0.35, "ml": 0.20, "player_xg": 0.10}

        # 4 methods with market but no player_xg or deep
        if available == {"factor", "xg", "ml", "market"}:
            return FALLBACK_WEIGHTS["factor_xg_ml_market"]

        # 4 methods with deep but no player_xg or market
        if available == {"factor", "xg", "ml", "deep"}:
            return {"factor": 0.38, "xg": 0.38, "ml": 0.19, "deep": 0.05}

        # 3 methods with market (proportional from ENSEMBLE_WEIGHTS)
        if available == {"factor", "xg", "market"}:
            return {"factor": 0.19, "xg": 0.25, "market": 0.56}
        if available == {"factor", "xg", "player_xg"}:
            return FALLBACK_WEIGHTS["factor_xg_player"]

        # 3 methods without market
        if available == {"factor", "xg", "ml"}:
            return {"factor": 0.40, "xg": 0.40, "ml": 0.20}

        # 2 methods
        if available == {"factor", "xg"}:
            return FALLBACK_WEIGHTS["factor_xg"]
        if available == {"factor", "ml"}:
            return FALLBACK_WEIGHTS["factor_ml"]
        if available == {"factor", "market"}:
            return FALLBACK_WEIGHTS["factor_market"]
        if available == {"factor", "player_xg"}:
            return {"factor": 0.70, "player_xg": 0.30}
        if available == {"factor", "deep"}:
            return {"factor": 0.90, "deep": 0.10}

        # Dynamic fallback: proportionally redistribute from full weights
        # Handles any method subset not covered by explicit cases above
        ref = ENSEMBLE_WEIGHTS_WITH_DEEP if "deep" in available else ENSEMBLE_WEIGHTS
        subset = {m: ref.get(m, 0.1) for m in available if m in ref}
        if subset:
            total = sum(subset.values())
            if total > 0:
                return {m: w / total for m, w in subset.items()}

        # Factor only (absolute last resort)
        if "factor" in available:
            return FALLBACK_WEIGHTS["factor_only"]

        # No factor available — equal weight everything
        return {m: 1.0 / len(available) for m in available}

    @staticmethod
    def _method_confidence(probs: Dict) -> float:
        """Measure how decisive a method is (0 = uniform/guessing, 1 = certain).

        Uses max probability deviation from uniform (0.333).
        Methods outputting ~33/33/33 get penalized; those with clear
        favorites (e.g. 60/20/20) keep full weight.
        """
        max_p = max(probs["prob_H"], probs["prob_D"], probs["prob_A"])
        # Uniform = 0.333, perfect = 1.0. Scale 0-1.
        return min(1.0, max(0.0, (max_p - 0.333) / 0.333))

    def _combine_predictions(self, predictions: Dict, lineup_source: str = "predicted") -> Dict:
        """Combine predictions using weighted average with auto-downweighting.

        Methods that output near-uniform probabilities (sign of guessing)
        get their weight reduced automatically. When confirmed lineups are
        available, player_xg gets a weight boost (0.05 → 0.12).

        When MetaLearnerCombiner is loaded and ML/Market/xG are all available,
        uses learned meta-learner for the core blend (LL -0.0123 improvement),
        then mixes in factor/player_xg with a small secondary weight.
        """
        # --- META-LEARNER PATH (preferred when all 3 core predictors available) ---
        # Ablation-validated: v1+meta(C=0.5)+db=1.06+pT=1.00 → LL=0.9675
        # Beats fixed-weight (0.9694) by -0.0019 and market (0.9714) by -0.0039
        if (self.meta_learner.loaded
                and "ml" in predictions and "market" in predictions and "xg" in predictions):
            meta_result = self.meta_learner.combine(
                predictions["ml"], predictions["market"], predictions["xg"])
            if meta_result is not None:
                prob_H = meta_result["prob_H"]
                prob_D = meta_result["prob_D"]
                prob_A = meta_result["prob_A"]

                # Mix in factor/player_xg signal (10% secondary weight).
                # Previously these were entirely ignored, losing referee, derby,
                # weather, and player availability insights.
                secondary_weight = 0.0
                sec_H, sec_D, sec_A = 0.0, 0.0, 0.0
                if "factor" in predictions:
                    f = predictions["factor"]
                    sec_H += 0.75 * f["prob_H"]
                    sec_D += 0.75 * f["prob_D"]
                    sec_A += 0.75 * f["prob_A"]
                    secondary_weight += 0.75
                if "player_xg" in predictions:
                    p = predictions["player_xg"]
                    sec_H += 0.25 * p["prob_H"]
                    sec_D += 0.25 * p["prob_D"]
                    sec_A += 0.25 * p["prob_A"]
                    secondary_weight += 0.25

                if secondary_weight > 0:
                    sec_H /= secondary_weight
                    sec_D /= secondary_weight
                    sec_A /= secondary_weight
                    # Blend: 90% meta-learner + 10% secondary signals
                    mix = 0.10
                    prob_H = (1 - mix) * prob_H + mix * sec_H
                    prob_D = (1 - mix) * prob_D + mix * sec_D
                    prob_A = (1 - mix) * prob_A + mix * sec_A

                # Mild draw boost (1.06 validated optimal for meta-learner output)
                prob_D *= 1.06
                total = prob_H + prob_D + prob_A
                prob_H, prob_D, prob_A = prob_H/total, prob_D/total, prob_A/total

                # Safety clip
                prob_H = max(0.001, min(0.999, prob_H))
                prob_D = max(0.001, min(0.999, prob_D))
                prob_A = max(0.001, min(0.999, prob_A))
                total = prob_H + prob_D + prob_A
                prob_H, prob_D, prob_A = prob_H/total, prob_D/total, prob_A/total

                # Capture betting probabilities AFTER draw boost + clip (same
                # rationale as fixed-weight path — calibrated probs are best).
                raw_H, raw_D, raw_A = prob_H, prob_D, prob_A

                return {
                    "prob_H": prob_H, "prob_D": prob_D, "prob_A": prob_A,
                    "raw_prob_H": raw_H, "raw_prob_D": raw_D, "raw_prob_A": raw_A,
                }

        # --- FIXED-WEIGHT PATH (fallback) ---
        weights = self._get_effective_weights(predictions)

        # Boost player_xg weight when using confirmed lineups (re-normalize to 1.0)
        if lineup_source == "confirmed" and "player_xg" in weights:
            old_w = weights.get("player_xg", 0.05)
            target_w = 0.12
            if target_w > old_w:
                boost = target_w - old_w
                other_total = sum(v for k, v in weights.items() if k != "player_xg")
                if other_total > 0:
                    weights = {
                        k: (target_w if k == "player_xg"
                            else v - (v / other_total) * boost)
                        for k, v in weights.items()
                    }
                log.info(f"Confirmed lineups: player_xg weight {old_w:.2f} → {target_w:.2f} (re-normalized)")

        # Auto-downweight methods that look unreliable (near-uniform output)
        # Only penalize deep and player_xg — core methods keep full weight
        # Skip downweighting player_xg when using confirmed lineups
        downweight_eligible = {"deep", "player_xg"} if lineup_source != "confirmed" else {"deep"}
        effective_weights = {}
        for method, w in weights.items():
            if method in downweight_eligible and method in predictions:
                conf = self._method_confidence(predictions[method])
                # Scale: conf=0 → 30% of weight, conf=1 → 100% of weight
                penalty = 0.3 + 0.7 * conf
                effective_weights[method] = w * penalty
            else:
                effective_weights[method] = w

        # Calculate weighted average
        prob_H = 0.0
        prob_D = 0.0
        prob_A = 0.0
        total_weight = 0.0

        for method, probs in predictions.items():
            w = effective_weights.get(method, 0)
            if w > 0:
                pH, pD, pA = probs["prob_H"], probs["prob_D"], probs["prob_A"]
                # Skip methods that returned NaN (prevents poisoning entire ensemble)
                if np.isnan(pH) or np.isnan(pD) or np.isnan(pA):
                    log.warning(f"Skipping {method}: NaN probabilities")
                    continue
                prob_H += w * pH
                prob_D += w * pD
                prob_A += w * pA
                total_weight += w

        if total_weight > 0:
            prob_H /= total_weight
            prob_D /= total_weight
            prob_A /= total_weight
        else:
            # All methods returned NaN or had 0 weight — fall back to uniform
            log.error("All prediction methods failed or returned NaN, using uniform probs")
            prob_H, prob_D, prob_A = 1/3, 1/3, 1/3

        # Draw boost compensates for draw under-prediction with no-odds ML model.
        # History: 1.321 → 1.0 → 1.08 → 1.28 → 1.12 (reduced because ML T changed
        # from 0.40 to 0.75 — less draw compression means less boost needed).
        draw_boost = 1.12
        if draw_boost != 1.0:
            prob_D *= draw_boost
            total = prob_H + prob_D + prob_A
            prob_H /= total
            prob_D /= total
            prob_A /= total

        # Post-ensemble temperature scaling — sharpens underconfident predictions.
        # History: 1.08 → 1.04 → 0.90. T<1.0 sharpens (increases confidence).
        # Calibration sweep showed T=0.90 cuts ECE from 0.0587 to 0.0329.
        post_T = 0.90
        if post_T != 1.0:
            import numpy as _np
            _eps = 1e-10
            _logits = [_np.log(prob_H + _eps), _np.log(prob_D + _eps), _np.log(prob_A + _eps)]
            _scaled = [_np.exp(l / post_T) for l in _logits]
            _s_total = sum(_scaled)
            prob_H, prob_D, prob_A = _scaled[0] / _s_total, _scaled[1] / _s_total, _scaled[2] / _s_total

        # Draw probability ceiling — audit shows model calibration breaks above 30%.
        # When P(D) > 30%, actual draw rate stays flat at ~28%. Clip and redistribute.
        DRAW_CEIL = 0.30
        if prob_D > DRAW_CEIL:
            excess = prob_D - DRAW_CEIL
            prob_D = DRAW_CEIL
            # Redistribute excess to H/A proportionally
            ha_total = prob_H + prob_A
            if ha_total > 0:
                prob_H += excess * (prob_H / ha_total)
                prob_A += excess * (prob_A / ha_total)

        # Safety clip: ensure valid probabilities after all adjustments
        prob_H = max(0.001, min(0.999, prob_H))
        prob_D = max(0.001, min(0.999, prob_D))
        prob_A = max(0.001, min(0.999, prob_A))
        total = prob_H + prob_D + prob_A
        prob_H /= total
        prob_D /= total
        prob_A /= total

        # Capture betting probabilities AFTER draw boost + temperature + clip.
        # Previously captured BEFORE draw boost, causing the betting system to
        # work with systematically underestimated draw probabilities (~22% vs
        # ~26.5% base rate). The calibrated probabilities are the model's best
        # estimate — the betting system should use them.
        raw_H, raw_D, raw_A = prob_H, prob_D, prob_A

        return {
            "prob_H": prob_H,
            "prob_D": prob_D,
            "prob_A": prob_A,
            "raw_prob_H": raw_H,
            "raw_prob_D": raw_D,
            "raw_prob_A": raw_A,
        }

    def _load_lessons(self):
        """Load lessons once at prediction batch start. Call before predict loop."""
        self._lessons_data = None
        self._lessons_dirty = False
        lessons_path = DATA_DIR / "feedback" / "lessons.json"
        if not lessons_path.exists():
            return
        try:
            with open(lessons_path) as f:
                self._lessons_data = json.load(f)
        except Exception as e:
            log.warning(f"Failed to load lessons data: {e}")
            self._lessons_data = None

    def _flush_lessons(self):
        """Write accumulated applied_count changes once at batch end."""
        if not self._lessons_dirty or not self._lessons_data:
            return
        lessons_path = DATA_DIR / "feedback" / "lessons.json"
        try:
            with open(lessons_path, "w") as f:
                json.dump(self._lessons_data, f, indent=2)
        except Exception as e:
            log.error(f"Failed to flush lessons data: {e}")
        self._lessons_dirty = False

    def _apply_lessons(
        self, probs: Dict, home: str, away: str, predicted: str,
        home_xg: float = 1.3, away_xg: float = 1.1,
    ) -> tuple:
        """Apply persistent lessons to adjust probabilities.

        Uses Poisson model for xG bias corrections (not magic linear numbers).
        confidence_shift modifies the dominant outcome's probability.
        market_penalty reduces market-side probability weight.

        Returns (adjusted_probs, list_of_applied_lesson_ids).
        """
        if not self._lessons_data:
            return probs, []

        applied_ids = []
        adj_home_xg = home_xg
        adj_away_xg = away_xg
        prob_H = probs["prob_H"]
        prob_D = probs["prob_D"]
        prob_A = probs["prob_A"]

        for lesson in self._lessons_data.get("lessons", []):
            if not lesson.get("active", True):
                continue
            if lesson.get("applied_count", 0) >= lesson.get("expires_after_n", 20):
                continue

            scope = lesson.get("scope", {})
            correction = lesson.get("correction", {})
            ltype = lesson.get("type", "")
            confidence = lesson.get("confidence", 0.5)

            matched_lesson = False

            if ltype == "xg_bias":
                team = scope.get("team", "")
                venue = scope.get("venue")  # None = both, "home"/"away" = specific
                if team == home and venue in (None, "home"):
                    adj = correction.get("team_xg_adjust", 0) * confidence
                    adj_home_xg += adj
                    matched_lesson = True
                elif team == away and venue in (None, "away"):
                    adj = correction.get("team_xg_adjust", 0) * confidence
                    adj_away_xg += adj
                    matched_lesson = True

            elif ltype == "confidence_shift":
                level = scope.get("confidence_level", "")
                # Determine current confidence level from max prob
                max_p = max(prob_H, prob_D, prob_A)
                current_level = "LOW"
                if max_p >= 0.58:
                    current_level = "VERY HIGH"
                elif max_p >= 0.48:
                    current_level = "HIGH"
                elif max_p >= 0.40:
                    current_level = "MEDIUM-HIGH"
                elif max_p >= 0.33:
                    current_level = "MEDIUM"

                if level == current_level:
                    shift = correction.get("confidence_adjust", 0) * confidence
                    # Apply shift: positive = band is too optimistic, shrink dominant prob
                    # negative = band underperforms, boost dominant prob
                    if predicted == "HOME":
                        prob_H += shift
                    elif predicted == "AWAY":
                        prob_A += shift
                    else:
                        prob_D += shift
                    matched_lesson = True

            elif ltype == "market_penalty":
                # Reduce confidence of market-aligned predictions for this market type
                # Applied as a dampening toward uniform (33/33/33)
                penalty = correction.get("confidence_reduce", 0) * confidence
                if penalty > 0:
                    uniform = 1.0 / 3.0
                    prob_H = prob_H * (1 - penalty) + uniform * penalty
                    prob_D = prob_D * (1 - penalty) + uniform * penalty
                    prob_A = prob_A * (1 - penalty) + uniform * penalty
                    matched_lesson = True

            elif ltype == "method_boost":
                method = scope.get("method", "")
                matched_lesson = bool(method)

            if matched_lesson:
                applied_ids.append(lesson["id"])
                lesson["applied_count"] = lesson.get("applied_count", 0) + 1
                self._lessons_dirty = True

        # If xG was adjusted, recompute probabilities through Poisson
        if adj_home_xg != home_xg or adj_away_xg != away_xg:
            adj_home_xg = max(0.2, adj_home_xg)
            adj_away_xg = max(0.2, adj_away_xg)
            poisson_probs = self._poisson_win_prob(adj_home_xg, adj_away_xg)
            # Blend: 70% Poisson-corrected, 30% original (don't fully override ensemble)
            blend = 0.7
            prob_H = prob_H * (1 - blend) + poisson_probs["H"] * blend
            prob_D = prob_D * (1 - blend) + poisson_probs["D"] * blend
            prob_A = prob_A * (1 - blend) + poisson_probs["A"] * blend

        # Normalize
        if applied_ids:
            prob_H = max(0.05, prob_H)
            prob_D = max(0.05, prob_D)
            prob_A = max(0.05, prob_A)
            total = prob_H + prob_D + prob_A
            probs = {
                "prob_H": prob_H / total,
                "prob_D": prob_D / total,
                "prob_A": prob_A / total,
                # Preserve raw_prob_* through lessons — betting uses these
                "raw_prob_H": probs.get("raw_prob_H", prob_H / total),
                "raw_prob_D": probs.get("raw_prob_D", prob_D / total),
                "raw_prob_A": probs.get("raw_prob_A", prob_A / total),
            }

        return probs, applied_ids

    @staticmethod
    def _poisson_win_prob(home_xg: float, away_xg: float, max_goals: int = 10) -> Dict:
        """Poisson win probability from xG (lightweight version for lesson corrections).

        Applies the same draw inflation as XGPredictor._poisson_win_prob (1.55, 0.20)
        to keep lesson-corrected probabilities consistent with the main pipeline.
        """
        home_xg = max(0.2, min(4.5, home_xg))
        away_xg = max(0.2, min(4.5, away_xg))
        h_pmf = [poisson.pmf(g, home_xg) for g in range(max_goals)]
        a_pmf = [poisson.pmf(g, away_xg) for g in range(max_goals)]
        pH = pD = pA = 0.0
        for hg in range(max_goals):
            for ag in range(max_goals):
                p = h_pmf[hg] * a_pmf[ag]
                if hg > ag:
                    pH += p
                elif hg == ag:
                    pD += p
                else:
                    pA += p
        # Draw inflation: same calibrated params as XGPredictor (1.55, 0.20)
        xg_gap = abs(home_xg - away_xg)
        draw_inflate = max(0.90, min(1.55, 1.55 - 0.20 * xg_gap))
        pD *= draw_inflate
        t = pH + pD + pA
        return {"H": pH / t, "D": pD / t, "A": pA / t}


# =============================================================================
# MAIN PREDICTION PIPELINE
# =============================================================================

def _load_league_matches(league: str) -> List[Dict]:
    """Load upcoming matches for a given league.

    Serie A: delegates to existing load_upcoming_matches() (manual + scraped).
    Other leagues: reads from odds_full_{league}.json (the odds fetch already
    stores match metadata including home_team, away_team, commence_time).
    """
    if league == "serie_a":
        return load_upcoming_matches()

    odds_path = DATA_DIR / "upcoming" / f"odds_full_{league}.json"
    if not odds_path.exists():
        log.warning("No match data for %s at %s", league, odds_path)
        return []

    try:
        with open(odds_path) as f:
            odds_data = json.load(f)
    except Exception as e:
        log.error("Failed to load %s matches: %s", league, e)
        return []

    raw = odds_data.get("matches", odds_data)
    if isinstance(raw, list):
        items = [(m.get("match", f"{m.get('home_team','?')} vs {m.get('away_team','?')}"), m)
                 for m in raw]
    else:
        items = list(raw.items())

    matches = []
    for match_key, match_data in items:
        home = match_data.get("home_team", match_key.split(" vs ")[0] if " vs " in match_key else "?")
        away = match_data.get("away_team", match_key.split(" vs ")[1] if " vs " in match_key else "?")
        ct = match_data.get("commence_time", "")
        date_str = ct[:10] if ct else ""
        time_str = ct[11:16] if len(ct) >= 16 else ""

        matches.append({
            "match": f"{home} vs {away}",
            "home_team": home,
            "away_team": away,
            "date": date_str,
            "time": time_str,
            "commence_time": ct,
            "league": league,
        })

    return matches


def run_ensemble_predictions(use_ensemble: bool = True, league: str = "serie_a") -> Dict:
    """Run the full ensemble prediction pipeline.

    Args:
        use_ensemble: Whether to use the full ensemble (True) or factor-only (False).
        league: League identifier ("serie_a", "premier_league", etc.).
                Default "serie_a" preserves backward compatibility.
    """
    league_display = {"serie_a": "Serie A", "premier_league": "Premier League"}.get(league, league)
    log.info("=" * 70)
    log.info("ENSEMBLE PREDICTION ENGINE — %s", league_display)
    log.info("=" * 70)

    # Initialize ensemble with league context
    ensemble = EnsemblePredictor(league=league)
    if use_ensemble and not ensemble.initialize():
        log.warning("Failed to initialize ensemble - falling back to factor-only")
        use_ensemble = False

    # Step 1: Load upcoming matches
    log.info("\n[1/4] Loading upcoming %s matches...", league_display)
    matches = _load_league_matches(league)
    if not matches:
        log.error("No upcoming %s matches found!", league_display)
        return {}
    log.info(f"Found {len(matches)} upcoming {league_display} matches")

    # Step 2: Calculate current form (graceful for non-Serie A)
    log.info("\n[2/4] Calculating current team form...")
    try:
        form_data = calculate_all_forms()
    except Exception as e:
        log.warning("Form calculation failed for %s: %s — using empty form data", league_display, e)
        form_data = {}

    # Step 3: Fetch weather (graceful for non-Serie A)
    log.info("\n[3/4] Fetching weather forecasts...")
    try:
        weather_data = fetch_all_match_weather()
    except Exception as e:
        log.warning("Weather fetch failed for %s: %s", league_display, e)
        weather_data = {}

    # Step 4: Analyze referees (graceful for non-Serie A)
    log.info("\n[4/4] Analyzing referee assignments...")
    try:
        referee_data = analyze_referee_impact(matches)
    except Exception as e:
        log.warning("Referee analysis failed for %s: %s", league_display, e)
        referee_data = {}

    # Load confirmed lineups if available
    confirmed_lineups = None
    try:
        lineup_path = DATA_DIR / "upcoming" / "confirmed_lineups.json"
        if lineup_path.exists():
            with open(lineup_path) as f:
                lineup_data = json.load(f)
            confirmed_lineups = lineup_data.get("matches", {})
            if confirmed_lineups:
                log.info(f"Loaded confirmed lineups for {len(confirmed_lineups)} match(es)")
    except Exception as e:
        log.debug(f"No confirmed lineups available: {e}")

    # Load injury data for xG adjustments
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

    # Generate predictions
    log.info("\n[5/5] Generating ensemble predictions...")
    predictions = []

    # Load lessons once for the entire batch (fix: no per-match file I/O)
    if use_ensemble:
        ensemble._load_lessons()

    for match in matches:
        match_name = match.get("match", f"{match.get('home_team','?')} vs {match.get('away_team','?')}")

        # Get factor analysis
        try:
            factors = identify_all_factors(match, form_data, weather_data, referee_data)
        except Exception as e:
            log.warning("Factor analysis failed for %s: %s — using defaults", match_name, e)
            factors = {"n_home_factors": 0, "n_away_factors": 0, "home_factors": [], "away_factors": []}

        if use_ensemble:
            # Use ensemble — gracefully skip this match if it fails
            try:
                pred = ensemble.predict(
                    match, factors, form_data, confirmed_lineups=confirmed_lineups
                )
            except Exception as e:
                log.error("Ensemble prediction FAILED for %s: %s — skipping match", match_name, e)
                continue  # Skip this match, don't crash the entire batch

            if pred is None:
                log.warning("Ensemble returned None for %s — skipping", match_name)
                continue

            # Merge with standard format
            pred["date"] = match.get("date", "TBD")
            pred["time"] = match.get("time", "TBD")
            pred["venue"] = match.get("venue", f"{match['home_team']} Stadium")

            # Get confidence level — primarily probability-based
            # pred["confidence"] is max(prob_H, prob_D, prob_A) from ensemble
            max_prob = pred["confidence"]
            n_factors = factors["n_home_factors"] + factors["n_away_factors"]
            # Minor factor bonus: +2pp per factor above 3, capped at +4pp
            factor_bonus = max(0, min(0.04, (n_factors - 3) * 0.02))
            adjusted_conf = max_prob + factor_bonus

            if adjusted_conf >= 0.58:
                pred["confidence_level"] = "VERY HIGH"
            elif adjusted_conf >= 0.48:
                pred["confidence_level"] = "HIGH"
            elif adjusted_conf >= 0.40:
                pred["confidence_level"] = "MEDIUM-HIGH"
            elif adjusted_conf >= 0.33:
                pred["confidence_level"] = "MEDIUM"
            else:
                pred["confidence_level"] = "LOW"

            # Add factors
            pred["home_factors"] = factors["home_factors"]
            pred["away_factors"] = factors["away_factors"]
            pred["neutral_factors"] = factors["neutral_factors"]
            pred["n_factors"] = n_factors

            # Home/away form
            key = f"{match['home_team']} vs {match['away_team']}"
            matchup = form_data.get("matchups", {}).get(key, {})
            pred["home_form"] = matchup.get("home_form", {})
            pred["away_form"] = matchup.get("away_form", {})

            # Referee
            ref_info = referee_data.get(key, {})
            if ref_info.get("referee"):
                pred["referee"] = ref_info["referee"]
                pred["referee_bias"] = ref_info.get("classification", {}).get("bias_type", "unknown")

            # Expected goals from xG (if available)
            xg_details = pred.get("component_predictions", {}).get("xg_details", {})
            if xg_details:
                pred["expected_goals"] = round(xg_details["home_xg"] + xg_details["away_xg"], 2)
                pred["home_xg"] = round(xg_details["home_xg"], 2)
                pred["away_xg"] = round(xg_details["away_xg"], 2)
            else:
                pred["expected_goals"] = 2.67  # Base

            # Apply injury-based xG adjustments
            home_team = match["home_team"]
            away_team = match["away_team"]
            home_injured = injury_map.get(home_team, [])
            away_injured = injury_map.get(away_team, [])
            if home_injured or away_injured:
                try:
                    from features.injury_impact import compute_injury_xg_adjustment
                    if home_injured:
                        h_adj = compute_injury_xg_adjustment(
                            home_team, away_team, home_injured
                        )
                        pred["home_xg"] = round(
                            pred.get("home_xg", 1.3) + h_adj["team_xg_adj"], 2
                        )
                        pred["away_xg"] = round(
                            pred.get("away_xg", 1.3) + h_adj["opponent_xg_adj"], 2
                        )
                    if away_injured:
                        a_adj = compute_injury_xg_adjustment(
                            away_team, home_team, away_injured
                        )
                        pred["away_xg"] = round(
                            pred.get("away_xg", 1.3) + a_adj["team_xg_adj"], 2
                        )
                        pred["home_xg"] = round(
                            pred.get("home_xg", 1.3) + a_adj["opponent_xg_adj"], 2
                        )
                    # Ensure non-negative
                    pred["home_xg"] = max(0.1, pred.get("home_xg", 1.3))
                    pred["away_xg"] = max(0.1, pred.get("away_xg", 1.3))
                    pred["expected_goals"] = round(pred["home_xg"] + pred["away_xg"], 2)
                    pred["injury_adjustments"] = {
                        "home_injured": home_injured,
                        "away_injured": away_injured,
                    }
                except Exception as e:
                    log.debug(f"Injury xG adjustment failed: {e}")

            pred["over_25"] = bool(pred["expected_goals"] > 2.5)

            # Betting recommendation
            pred["betting_recommendation"] = _get_betting_recommendation(
                pred["predicted_outcome"],
                pred["probabilities"]["home"],
                pred["probabilities"]["draw"],
                pred["probabilities"]["away"],
                pred["confidence_level"]
            )
        else:
            # Fall back to factor-only
            pred = generate_prediction(match, factors, form_data)

        predictions.append(pred)

    # Flush accumulated lesson applied_count changes (one write, not per-match)
    if use_ensemble:
        ensemble._flush_lessons()

    # Sort by confidence
    confidence_order = {"VERY HIGH": 0, "HIGH": 1, "MEDIUM-HIGH": 2, "MEDIUM": 3, "LOW": 4}
    predictions.sort(key=lambda x: (
        confidence_order.get(x.get("confidence_level", x.get("confidence", "MEDIUM")), 5),
        -x["probabilities"]["home"]
    ))

    # Save predictions
    output = {
        "generated_at": datetime.now().isoformat(),
        "league": league,
        "model_version": "v4.0-deep-learning" if use_ensemble else "v2.0-21seasons",
        "ensemble_enabled": use_ensemble,
        "methods_available": ensemble.available_methods if use_ensemble else ["factor"],
        "phase4_features": ensemble.phase4_features if use_ensemble else [],
        "predictions": predictions,
        "summary": {
            "total_matches": len(predictions),
            "high_confidence": len([p for p in predictions if p.get("confidence_level") in ["VERY HIGH", "HIGH"]]),
            "medium_confidence": len([p for p in predictions if p.get("confidence_level") == "MEDIUM-HIGH"]),
        }
    }

    # League-aware output path: Serie A -> predictions.json, others -> predictions_{league}.json
    if league == "serie_a":
        output_path = DATA_DIR / "upcoming" / "predictions.json"
    else:
        output_path = DATA_DIR / "upcoming" / f"predictions_{league}.json"

    # Atomic write: temp file + rename to prevent corruption on crash
    tmp_path = output_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(output, f, indent=2, cls=_NumpySafeEncoder)
    tmp_path.replace(output_path)

    log.info(f"\nSaved {league_display} ensemble predictions to {output_path}")
    return output


def _get_betting_recommendation(outcome: str, h_prob: float, d_prob: float, a_prob: float, confidence: str) -> str:
    """Generate betting recommendation based on ensemble prediction."""
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


def print_ensemble_predictions(output: Dict):
    """Print ensemble predictions in a readable format."""
    league = output.get("league", "serie_a")
    league_display = {"serie_a": "Serie A", "premier_league": "Premier League"}.get(league, league.upper())
    print("\n" + "=" * 80)
    print(f"{league_display} PREDICTIONS - ENSEMBLE MODEL")
    print("=" * 80)
    print(f"Generated: {output.get('generated_at', 'Unknown')}")
    print(f"Model: {output.get('model_version', 'Unknown')}")
    print(f"Methods: {', '.join(output.get('methods_available', ['factor']))}")
    if output.get('phase4_features'):
        print(f"Phase 4: {', '.join(output.get('phase4_features', []))}")

    predictions = output.get("predictions", [])

    for pred in predictions:
        print(f"\n{'─' * 60}")
        print(f"{pred['match']}")
        print(f"  {pred.get('date', 'TBD')} {pred.get('time', '')} | {pred.get('venue', '')}")

        probs = pred["probabilities"]
        print(f"\n  PREDICTION: {pred['predicted_outcome']} ({pred.get('confidence_level', 'MEDIUM')})")
        print(f"  Probabilities: H {probs['home']:.1%} | D {probs['draw']:.1%} | A {probs['away']:.1%}")

        # Component predictions
        components = pred.get("component_predictions", {})
        if components:
            print("\n  Component Predictions:")
            for method, comp_probs in components.items():
                if method == "xg_details":
                    print(f"    xG: Home {comp_probs['home_xg']:.2f} - Away {comp_probs['away_xg']:.2f}")
                elif isinstance(comp_probs, dict) and "prob_H" in comp_probs:
                    print(f"    {method.upper()}: H {comp_probs['prob_H']:.1%} D {comp_probs['prob_D']:.1%} A {comp_probs['prob_A']:.1%}")

        # Factors
        if pred.get("home_factors"):
            print(f"\n  Home factors: {', '.join(pred['home_factors'])}")
        if pred.get("away_factors"):
            print(f"  Away factors: {', '.join(pred['away_factors'])}")

        # Phase 4: Market Intelligence
        market = pred.get("market_intelligence", {})
        if market:
            print(f"\n  Market Intelligence:")
            print(f"    Sharp score: {market.get('sharp_score', 0):+.3f} | "
                  f"Movement: {market.get('movement_magnitude', 0):.1f}%")
            if market.get("has_value"):
                best = market.get('best_bet', 'home')
                edge = market.get(f'{best}_edge', 0)
                print(f"    VALUE DETECTED: {best.upper()} (edge: {edge:+.1%})")

        # Phase 4: Momentum Analysis
        momentum = pred.get("momentum_analysis", {})
        if momentum:
            print(f"\n  Momentum:")
            print(f"    Home: Big win recency {momentum.get('home_big_win_recency', 0):.2f} | "
                  f"Away: {momentum.get('away_big_win_recency', 0):.2f}")

        # Phase 4: Sentiment
        sentiment = pred.get("sentiment_analysis", {})
        if sentiment:
            print(f"\n  Sentiment/Motivation:")
            print(f"    Home: {sentiment.get('home_motivation', 0.5):.2f} | "
                  f"Away: {sentiment.get('away_motivation', 0.5):.2f} | "
                  f"Diff: {sentiment.get('sentiment_diff', 0):+.3f}")

        # Recommendation
        rec = pred.get("betting_recommendation", "")
        print(f"\n  {rec}")

    # Summary
    summary = output.get("summary", {})
    print("\n" + "=" * 80)
    print("SUMMARY")
    print(f"Total matches: {summary.get('total_matches', 0)}")
    print(f"High confidence: {summary.get('high_confidence', 0)}")
    print(f"Medium confidence: {summary.get('medium_confidence', 0)}")


if __name__ == "__main__":
    output = run_ensemble_predictions(use_ensemble=True)
    if output:
        print_ensemble_predictions(output)
