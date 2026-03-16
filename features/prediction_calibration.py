#!/usr/bin/env python3
"""Prediction Calibration Module - Phase 2.2 & 2.3

Implements:
- Confidence filtering (reject low-quality predictions)
- Betting strategy profiles (selectable weight configurations)
- HOME advantage calibration
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# BETTING STRATEGY PROFILES (Phase 2.3)
# =============================================================================

BETTING_STRATEGIES = {
    "default": {
        "name": "Default",
        "description": "Balanced approach - current production weights",
        "weights": {
            "factor": 0.27,
            "xg": 0.30,
            "ml": 0.14,
            "player_xg": 0.05,
            "deep": 0.02,
            "market": 0.22,
        },
        "min_confidence": 0.45,
        "high_conf_threshold": 0.55,
    },
    "volume": {
        "name": "Volume",
        "description": "Lower confidence threshold, more bets",
        "weights": {
            "factor": 0.22,
            "xg": 0.25,
            "ml": 0.15,
            "player_xg": 0.08,
            "deep": 0.02,
            "market": 0.28,
        },
        "min_confidence": 0.42,
        "high_conf_threshold": 0.50,
    },
    "selective": {
        "name": "Selective (conservative)",
        "description": "Higher precision on fewer bets - strong factor/xG emphasis",
        "weights": {
            "factor": 0.32,
            "xg": 0.30,
            "ml": 0.12,
            "player_xg": 0.06,
            "deep": 0.02,
            "market": 0.18,
        },
        "min_confidence": 0.50,
        "high_conf_threshold": 0.58,
    },
    "ml_heavy": {
        "name": "ML Heavy",
        "description": "Emphasizes ML classifier - good on recent form",
        "weights": {
            "factor": 0.20,
            "xg": 0.22,
            "ml": 0.25,
            "player_xg": 0.08,
            "deep": 0.02,
            "market": 0.23,
        },
        "min_confidence": 0.45,
        "high_conf_threshold": 0.55,
    },
    "xg_dominant": {
        "name": "xG Dominant",
        "description": "Emphasizes xG predictions - bookmaker style",
        "weights": {
            "factor": 0.18,
            "xg": 0.35,
            "ml": 0.12,
            "player_xg": 0.08,
            "deep": 0.02,
            "market": 0.25,
        },
        "min_confidence": 0.45,
        "high_conf_threshold": 0.53,
    },
}


class PredictionCalibrator:
    """Calibrates and filters predictions based on strategy and confidence."""

    def __init__(self, strategy: str = "default"):
        self.strategy = strategy
        self.config = BETTING_STRATEGIES.get(strategy, BETTING_STRATEGIES["default"])

    def get_weights(self) -> Dict[str, float]:
        """Get ensemble weights for current strategy."""
        return self.config["weights"].copy()

    def get_min_confidence(self) -> float:
        """Get minimum confidence threshold."""
        return self.config["min_confidence"]

    def get_high_conf_threshold(self) -> float:
        """Get high confidence threshold."""
        return self.config["high_conf_threshold"]

    def should_bet(self, confidence: float) -> bool:
        """Check if prediction meets minimum confidence for betting."""
        return confidence >= self.config["min_confidence"]

    def is_high_confidence(self, confidence: float) -> bool:
        """Check if prediction is high confidence."""
        return confidence >= self.config["high_conf_threshold"]

    def filter_predictions(
        self,
        predictions: List[Dict],
        min_confidence: float = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """Filter predictions by confidence threshold.

        Returns (accepted, rejected) tuple.
        """
        threshold = min_confidence or self.config["min_confidence"]

        accepted = []
        rejected = []

        for pred in predictions:
            conf = pred.get("confidence", 0)
            if conf >= threshold:
                accepted.append(pred)
            else:
                pred["rejection_reason"] = f"Confidence {conf:.1%} below threshold {threshold:.1%}"
                rejected.append(pred)

        return accepted, rejected

    def classify_prediction(self, prediction: Dict) -> Dict:
        """Classify prediction by confidence level and add recommendation."""
        conf = prediction.get("confidence", 0)

        if conf < self.config["min_confidence"]:
            confidence_class = "SKIP"
            recommendation = "No bet - confidence too low"
            bet_size = 0.0
        elif conf >= self.config["high_conf_threshold"]:
            confidence_class = "HIGH"
            recommendation = "Strong bet opportunity"
            bet_size = 2.0  # 2 units
        elif conf >= self.config["min_confidence"] + 0.05:
            confidence_class = "MEDIUM"
            recommendation = "Standard bet"
            bet_size = 1.0  # 1 unit
        else:
            confidence_class = "LOW"
            recommendation = "Small bet or skip"
            bet_size = 0.5  # 0.5 units

        return {
            **prediction,
            "confidence_class": confidence_class,
            "recommendation": recommendation,
            "suggested_bet_size": bet_size,
        }


# =============================================================================
# HOME ADVANTAGE CALIBRATION (Phase 2.4)
# =============================================================================

class HomeAdvantageCalibrator:
    """Calibrates home advantage to reduce prediction bias.

    Problem: 72% of predictions are HOME but only 49% accuracy.
    Solution: Reduce home advantage inflation in factor model.
    """

    # Historical Serie A home win rates by era
    HOME_WIN_RATES = {
        "2005-2015": 0.48,  # Higher home advantage era
        "2015-2020": 0.44,  # Declining home advantage
        "2020-2024": 0.42,  # Post-COVID lower home advantage
        "2024-now": 0.43,   # Current estimate
    }

    # Calibration factors to reduce over-prediction of home wins
    CALIBRATION_FACTORS = {
        "home_elo_bonus_reduction": 0.15,   # Reduce home elo bonus by 15%
        "home_prob_scaling": 0.92,          # Scale down home probability
        "away_prob_scaling": 1.08,          # Scale up away probability
        "draw_prob_boost": 1.05,            # Slight draw boost
    }

    def __init__(self, era: str = "2024-now"):
        self.era = era
        self.target_home_rate = self.HOME_WIN_RATES.get(era, 0.43)

    def calibrate_probabilities(
        self,
        probs: Dict[str, float],
        home_elo: float = 1500,
        away_elo: float = 1500,
    ) -> Dict[str, float]:
        """Apply calibration to reduce home bias.

        Args:
            probs: Original probabilities {prob_H, prob_D, prob_A}
            home_elo: Home team Elo rating
            away_elo: Away team Elo rating

        Returns:
            Calibrated probabilities
        """
        prob_H = probs.get("prob_H", 0.45)
        prob_D = probs.get("prob_D", 0.27)
        prob_A = probs.get("prob_A", 0.28)

        # Only calibrate if home is predicted with high probability
        # and teams are not massively different in strength
        elo_diff = home_elo - away_elo

        if prob_H > 0.50 and elo_diff < 200:
            # Apply calibration
            factors = self.CALIBRATION_FACTORS

            prob_H *= factors["home_prob_scaling"]
            prob_A *= factors["away_prob_scaling"]
            prob_D *= factors["draw_prob_boost"]

        # Normalize
        total = prob_H + prob_D + prob_A
        return {
            "prob_H": prob_H / total,
            "prob_D": prob_D / total,
            "prob_A": prob_A / total,
        }

    def get_home_threshold(self, elo_diff: float) -> float:
        """Get dynamic threshold for predicting HOME based on Elo difference.

        Higher Elo difference = lower threshold needed to predict HOME.
        This prevents predicting HOME for close matches just due to home bias.
        """
        if elo_diff > 150:
            return 0.45  # Clear favorite, standard threshold
        elif elo_diff > 100:
            return 0.48  # Moderate favorite
        elif elo_diff > 50:
            return 0.52  # Slight favorite - need higher confidence
        elif elo_diff > 0:
            return 0.55  # Evenly matched - need clear signal
        else:
            return 0.58  # Away team favored - very high threshold for HOME


# =============================================================================
# LIVE BIAS CORRECTION (from fair_odds_tracker ledger)
# =============================================================================

class LiveBiasCorrector:
    """Correct systematic probability biases using settled prediction data.

    Reads the fair_odds_ledger.json (populated by fair_odds_tracker.py) and
    computes per-outcome bias adjustments. For example, if the model predicts
    home wins at 48% average but the actual home win rate is 43%, it learns
    a -5pp correction.

    WHY this works: ML models often have systematic biases that are stable
    over time (e.g., overestimating home advantage, underestimating draws).
    By measuring predicted vs actual rates on settled predictions, we can
    apply a multiplicative correction that improves calibration without
    changing the model itself.

    Activates only after MIN_SAMPLES settled predictions to avoid noise.
    """

    MIN_SAMPLES = 30  # minimum settled predictions before corrections activate
    MAX_ADJUSTMENT = 0.06  # cap per-outcome adjustment at ±6pp
    LEDGER_PATH = Path(__file__).parent.parent / "data" / "betting" / "fair_odds_ledger.json"

    def __init__(self):
        self._corrections = None  # cached: {"home": +0.02, "draw": -0.01, "away": -0.01}
        self._sample_count = 0
        self._loaded = False

    def _load_corrections(self):
        """Compute bias corrections from settled predictions."""
        if self._loaded:
            return
        self._loaded = True

        if not self.LEDGER_PATH.exists():
            return

        try:
            with open(self.LEDGER_PATH) as f:
                ledger = json.load(f)
        except (json.JSONDecodeError, IOError):
            return

        settled = [r for r in ledger if r.get("settled")]
        self._sample_count = len(settled)

        if len(settled) < self.MIN_SAMPLES:
            return

        # Compute average predicted probability vs actual outcome rate
        # for each outcome (HOME, DRAW, AWAY)
        predicted_avg = {"home": 0, "draw": 0, "away": 0}
        actual_count = {"home": 0, "draw": 0, "away": 0}

        for r in settled:
            predicted_avg["home"] += r.get("prob_home", 0)
            predicted_avg["draw"] += r.get("prob_draw", 0)
            predicted_avg["away"] += r.get("prob_away", 0)
            actual = r.get("actual_outcome", "")
            if actual == "HOME":
                actual_count["home"] += 1
            elif actual == "DRAW":
                actual_count["draw"] += 1
            elif actual == "AWAY":
                actual_count["away"] += 1

        n = len(settled)
        corrections = {}
        for outcome in ("home", "draw", "away"):
            pred_rate = predicted_avg[outcome] / n
            actual_rate = actual_count[outcome] / n
            # Correction = actual - predicted (positive means model underestimates)
            raw_correction = actual_rate - pred_rate
            # Clamp to avoid large swings
            corrections[outcome] = max(-self.MAX_ADJUSTMENT,
                                       min(self.MAX_ADJUSTMENT, raw_correction))

        self._corrections = corrections
        log.info("Live bias corrections (n=%d): H=%+.3f D=%+.3f A=%+.3f",
                 n, corrections["home"], corrections["draw"], corrections["away"])

    def correct(self, probs: Dict[str, float]) -> Dict[str, float]:
        """Apply bias corrections to raw probabilities.

        Args:
            probs: {"prob_H": 0.45, "prob_D": 0.28, "prob_A": 0.27}

        Returns:
            Corrected probabilities (renormalized to sum to 1).
        """
        self._load_corrections()

        if not self._corrections:
            return probs

        corrected = {
            "prob_H": probs["prob_H"] + self._corrections["home"],
            "prob_D": probs["prob_D"] + self._corrections["draw"],
            "prob_A": probs["prob_A"] + self._corrections["away"],
        }

        # Clamp to [0.01, 0.98] then renormalize
        for k in corrected:
            corrected[k] = max(0.01, corrected[k])
        total = sum(corrected.values())
        for k in corrected:
            corrected[k] = round(corrected[k] / total, 4)

        return corrected

    @property
    def active(self) -> bool:
        """Whether corrections are active (enough data)."""
        self._load_corrections()
        return self._corrections is not None

    @property
    def sample_count(self) -> int:
        self._load_corrections()
        return self._sample_count


# =============================================================================
# UNIFIED CALIBRATION PIPELINE
# =============================================================================

class CalibrationPipeline:
    """Unified pipeline for all calibration steps."""

    def __init__(self, strategy: str = "default"):
        self.predictor_calibrator = PredictionCalibrator(strategy)
        self.home_calibrator = HomeAdvantageCalibrator()
        self.live_bias = LiveBiasCorrector()

        # Lazy import draw detector to avoid circular imports
        self._draw_detector = None

    @property
    def draw_detector(self):
        if self._draw_detector is None:
            try:
                from features.draw_detection import get_draw_detector
                self._draw_detector = get_draw_detector()
            except ImportError:
                log.warning("Draw detector not available")
        return self._draw_detector

    def calibrate_prediction(
        self,
        home_team: str,
        away_team: str,
        raw_probs: Dict[str, float],
        features: Dict = None,
    ) -> Dict:
        """Apply full calibration pipeline to a prediction.

        Pipeline:
        1. Home advantage calibration
        2. Draw detection adjustment
        3. Confidence classification
        """
        # Get Elo values from features if available
        home_elo = features.get("home_elo", 1500) if features else 1500
        away_elo = features.get("away_elo", 1500) if features else 1500

        # Step 1: Home advantage calibration — DISABLED
        # The HomeAdvantageCalibrator was compensating for the factor predictor's
        # old home bias (BASE_RATES anchor). Now that factors anchor on market
        # probabilities, this calibration over-corrects (hurts 5/21, helps 0/21).
        calibrated = raw_probs.copy()

        # Step 2: Live bias correction (from fair_odds_tracker settled data)
        if self.live_bias.active:
            calibrated = self.live_bias.correct(calibrated)

        # Step 3: Draw detection adjustment
        draw_analysis = None
        if self.draw_detector:
            calibrated, draw_analysis = self.draw_detector.adjust_ensemble_probs(
                home_team, away_team, calibrated, features
            )

        prob_H = calibrated["prob_H"]
        prob_D = calibrated["prob_D"]
        prob_A = calibrated["prob_A"]

        # Determine prediction by highest probability (no threshold overrides).
        # The old logic had an asymmetric elif chain that systematically
        # suppressed DRAW predictions: AWAY only needed >=, DRAW needed strict >,
        # and HOME needed to beat a dynamic threshold. Now all outcomes compete
        # equally on pure probability.
        if prob_H >= prob_D and prob_H >= prob_A:
            predicted = "HOME"
        elif prob_A >= prob_D:
            predicted = "AWAY"
        else:
            predicted = "DRAW"

        # Draw candidate override: if draw detection identifies a strong draw
        # signal AND draw is competitive (within 3pp of the leader), flip to DRAW
        if (draw_analysis and draw_analysis.get("is_draw_candidate")
                and prob_D > 0.28
                and prob_D >= max(prob_H, prob_A) - 0.03):
            predicted = "DRAW"

        max_prob = max(prob_H, prob_D, prob_A)

        result = {
            "predicted_outcome": predicted,
            "probabilities": {
                "home": round(prob_H, 3),
                "draw": round(prob_D, 3),
                "away": round(prob_A, 3),
            },
            "confidence": max_prob,
            "calibration_applied": True,
            "live_bias_active": self.live_bias.active,
            "live_bias_samples": self.live_bias.sample_count,
        }

        # Add classification
        classified = self.predictor_calibrator.classify_prediction(result)
        result.update({
            "confidence_class": classified["confidence_class"],
            "recommendation": classified["recommendation"],
            "suggested_bet_size": classified["suggested_bet_size"],
        })

        if draw_analysis:
            result["draw_analysis"] = draw_analysis

        return result


def get_calibration_pipeline(strategy: str = "default") -> CalibrationPipeline:
    """Factory function for calibration pipeline."""
    return CalibrationPipeline(strategy)


def list_strategies() -> Dict[str, Dict]:
    """List available betting strategies."""
    return {
        name: {
            "name": config["name"],
            "description": config["description"],
            "min_confidence": config["min_confidence"],
            "high_conf_threshold": config["high_conf_threshold"],
        }
        for name, config in BETTING_STRATEGIES.items()
    }


if __name__ == "__main__":
    # Test calibration
    print("Available betting strategies:")
    print("-" * 60)
    for name, info in list_strategies().items():
        print(f"\n{name}: {info['name']}")
        print(f"  {info['description']}")
        print(f"  Min confidence: {info['min_confidence']:.1%}")
        print(f"  High-conf threshold: {info['high_conf_threshold']:.1%}")

    print("\n" + "=" * 60)
    print("Testing calibration pipeline...")

    pipeline = get_calibration_pipeline("selective")

    # Test case: Home-biased prediction
    raw_probs = {"prob_H": 0.52, "prob_D": 0.25, "prob_A": 0.23}
    features = {"home_elo": 1520, "away_elo": 1500, "elo_diff": 20}

    result = pipeline.calibrate_prediction(
        "Roma", "Lazio",  # Derby
        raw_probs,
        features
    )

    print(f"\nRoma vs Lazio (Derby)")
    print(f"Raw probs: H={raw_probs['prob_H']:.1%} D={raw_probs['prob_D']:.1%} A={raw_probs['prob_A']:.1%}")
    print(f"Calibrated: H={result['probabilities']['home']:.1%} D={result['probabilities']['draw']:.1%} A={result['probabilities']['away']:.1%}")
    print(f"Prediction: {result['predicted_outcome']}")
    print(f"Confidence: {result['confidence']:.1%} ({result['confidence_class']})")
    print(f"Recommendation: {result['recommendation']}")
    if result.get("draw_analysis"):
        print(f"Draw score: {result['draw_analysis']['draw_score']:.2f}")
