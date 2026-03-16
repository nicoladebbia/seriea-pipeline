#!/usr/bin/env python3
"""Unified Backtesting Framework for Serie A Predictions.

Consolidates all backtesting capabilities into a single entry point:
  - Walk-forward ensemble accuracy evaluation (from backtest.py)
  - Full betting system P&L simulation (from backtest_betting_system.py)
  - Ensemble walk-forward with strategy selection (from backtest_ensemble.py)
  - Real-odds value betting simulation (from backtest_with_odds.py)
  - Player prop backtesting (from backtest_player_props.py)
  - ML walk-forward cross-validation (from ml/walk_forward.py)

Usage:
    # Standard walk-forward backtest (accuracy + calibration)
    python scripts/backtest_unified.py --mode accuracy --season 2024-2025

    # Multi-season walk-forward
    python scripts/backtest_unified.py --mode accuracy --season 2023-2024 --season 2024-2025

    # Full betting simulation with P&L
    python scripts/backtest_unified.py --mode betting --season 2024-2025 --bankroll 1000

    # Value betting with real odds + Kelly criterion
    python scripts/backtest_unified.py --mode value --season 2024-2025 --kelly 0.25

    # ML model walk-forward cross-validation
    python scripts/backtest_unified.py --mode ml-walkforward

    # All modes at once
    python scripts/backtest_unified.py --mode all --season 2023-2024 --season 2024-2025

    # Compare against Pinnacle closing line
    python scripts/backtest_unified.py --mode accuracy --compare-baseline

    # Use specific betting strategy
    python scripts/backtest_unified.py --mode accuracy --strategy selective
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import DATA_DIR, MODELS_DIR, SEASONS
from storage.paths import features_path

# Lazy-loaded ML classifier for backtest
_ml_classifier = None
_ml_is_ensemble = False

# Lazy-loaded isotonic calibrator for ensemble post-processing.
# WARNING: The calibrator is fitted on 2023-2025 OOF predictions. Using it
# in backtests on the SAME seasons produces in-sample results (data leakage).
# For fair backtesting, set USE_ISOTONIC_CALIBRATION = False.
# For production predictions on FUTURE matches, set it to True.
USE_ISOTONIC_CALIBRATION = False  # Safe default for backtesting

_isotonic_calibrator = None
_isotonic_loaded = False

def _get_isotonic_calibrator():
    """Load isotonic calibrator for ensemble recalibration (Phase 5).

    Returns list of 3 IsotonicRegression objects [H, D, A] or None.
    Cross-validated improvement: +6pp ROI on flat bets, +3x on Kelly.
    """
    global _isotonic_calibrator, _isotonic_loaded
    if not USE_ISOTONIC_CALIBRATION:
        return None
    if _isotonic_loaded:
        return _isotonic_calibrator

    _isotonic_loaded = True
    cal_path = MODELS_DIR / "universal" / "ensemble_calibrator.pkl"
    if not cal_path.exists():
        return None

    try:
        import pickle
        with open(cal_path, "rb") as f:
            artifact = pickle.load(f)
        _isotonic_calibrator = artifact["calibrators"]
        log.info("Loaded isotonic calibrator (%d samples, seasons %s)",
                 artifact.get("n_samples", 0),
                 artifact.get("seasons_used", []))
        return _isotonic_calibrator
    except Exception as e:
        log.warning("Failed to load isotonic calibrator: %s", e)
        return None

def _get_ml_classifier():
    """Load ML model for backtest.

    Priority: no-odds CatBoost > multi-model ensemble > single CatBoost.
    The no-odds model provides independent signal that doesn't overlap with
    the market predictor, improving ensemble diversity.
    """
    global _ml_classifier, _ml_is_ensemble
    if _ml_classifier is not None:
        return _ml_classifier

    # Priority 1: No-odds CatBoost (independent signal)
    try:
        from catboost import CatBoostClassifier
        no_odds_path = MODELS_DIR / "universal" / "catboost_no_odds.cbm"
        if no_odds_path.exists():
            _ml_classifier = CatBoostClassifier()
            _ml_classifier.load_model(str(no_odds_path))
            _ml_is_ensemble = False
            log.info("Loaded no-odds CatBoost (%d features) for backtest", len(_ml_classifier.feature_names_))
            return _ml_classifier
    except Exception as e:
        log.info("No-odds model not available (%s), trying ensemble", e)

    # Priority 2: Multi-model ensemble
    try:
        from ml.ensemble import WeightedAverageEnsemble
        ens = WeightedAverageEnsemble.load("universal")
        ens.blend_calibrator = None
        _ml_classifier = ens
        _ml_is_ensemble = True
        log.info("Loaded multi-model ensemble (%d features) for backtest", len(ens.feature_names))
        return _ml_classifier
    except Exception as e:
        log.info("Ensemble not available (%s), falling back to single CatBoost", e)

    # Priority 3: Single CatBoost
    try:
        from catboost import CatBoostClassifier
        model_path = MODELS_DIR / "universal" / "catboost_upcoming.cbm"
        if not model_path.exists():
            log.warning("ML classifier not found at %s", model_path)
            return None
        _ml_classifier = CatBoostClassifier()
        _ml_classifier.load_model(str(model_path))
        _ml_is_ensemble = False
        log.info("Loaded single CatBoost (%d features) for backtest", len(_ml_classifier.feature_names_))
        return _ml_classifier
    except Exception as e:
        log.warning("Failed to load ML classifier: %s", e)
        return None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Serie A base rates (from 20-season analysis)
BASE_RATES = {"H": 0.45, "D": 0.27, "A": 0.28}
LABEL_MAP = {"H": 0, "D": 1, "A": 2}
LABEL_NAMES = {0: "H", 1: "D", 2: "A"}


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BacktestConfig:
    """Unified backtest configuration."""

    # Modes: "accuracy", "betting", "value", "ml-walkforward", "all"
    mode: str = "accuracy"

    # Seasons to backtest
    seasons: List[str] = field(default_factory=lambda: ["2023-2024", "2024-2025"])

    # Ensemble strategy
    strategy: str = "default"

    # Whether to use formation adjustments
    use_formation: bool = False

    # Compare against Pinnacle closing line baseline
    compare_baseline: bool = False

    # Betting simulation parameters
    bankroll: float = 1000.0
    kelly_fraction: float = 0.10    # Synced with production (betting_unified.py)
    min_edge: float = 0.05      # Walk-forward calibrated: +0.6% / +11.2% per season
    max_edge: float = 0.15      # Walk-forward calibrated: T=1.289 widens edge distribution
    max_bets_per_match: int = 3
    max_total_bets: int = 20
    max_daily_exposure: float = 0.35
    confidence_threshold: float = 0.50

    # ML walk-forward parameters
    min_train_seasons: int = 5
    test_season_size: int = 1
    purge_matchweeks: int = 2

    # Metrics to compute
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "log_loss", "brier_score", "rps", "ece",
        "roi", "sharpe_ratio", "max_drawdown",
    ])

    # Store per-prediction records for calibration
    store_predictions: bool = True

    # Output
    save_path: Optional[str] = None


# =============================================================================
# ENSEMBLE WEIGHTS (from backtest.py)
# =============================================================================

BACKTEST_WEIGHTS = {
    "factor": 0.035,
    "xg": 0.124,
    "ml": 0.605,
    "player_xg": 0.032,
    "market": 0.205,
    "formation": 0.0,
}

BACKTEST_WEIGHTS_WITH_FORMATION = {
    "factor": 0.26,
    "xg": 0.26,
    "market": 0.20,
    "formation": 0.08,
}


# =============================================================================
# PREDICTION METHODS (from backtest.py)
# =============================================================================

def predict_factor_based(row: pd.Series) -> Dict[str, float]:
    """Factor-based prediction using pre-computed strength/elo features."""
    prob_H = BASE_RATES["H"]
    prob_D = BASE_RATES["D"]
    prob_A = BASE_RATES["A"]

    # Elo-based adjustment
    elo_diff = row.get("elo_diff", 0)
    if not np.isnan(elo_diff) and elo_diff != 0:
        elo_shift = elo_diff / 800.0
        prob_H += elo_shift * 0.4
        prob_A -= elo_shift * 0.4

    # Attack/defense strength
    home_atk = row.get("home_attack_strength", 1.0)
    home_def = row.get("home_defense_strength", 1.0)
    away_atk = row.get("away_attack_strength", 1.0)
    away_def = row.get("away_defense_strength", 1.0)

    if (
        not any(np.isnan(v) for v in [home_atk, home_def, away_atk, away_def])
        and away_def > 0
        and home_def > 0
    ):
        home_edge = (home_atk / away_def - 1.0) * 0.08
        away_edge = (away_atk / home_def - 1.0) * 0.08
        net_edge = home_edge - away_edge
        prob_H += net_edge
        prob_A -= net_edge

    # H2H features
    h2h_home_rate = row.get("h2h_home_win_rate", np.nan)
    if not np.isnan(h2h_home_rate) and h2h_home_rate > 0:
        h2h_adj = (h2h_home_rate - BASE_RATES["H"]) * 0.05
        prob_H += h2h_adj

    # Clip before normalizing (factor adjustments can push probs outside [0,1]
    # when elo_diff or strength ratios are extreme)
    prob_H = max(0.01, prob_H)
    prob_D = max(0.01, prob_D)
    prob_A = max(0.01, prob_A)

    # Normalize
    total = prob_H + prob_D + prob_A
    prob_H /= total
    prob_D /= total
    prob_A /= total

    return {"prob_H": prob_H, "prob_D": prob_D, "prob_A": prob_A}


def predict_xg_poisson(row: pd.Series) -> Optional[Dict[str, float]]:
    """xG-based Poisson prediction using pre-computed xG features."""
    home_xg_atk = row.get("home_xg_attack_strength", np.nan)
    home_xg_def = row.get("home_xg_defense_strength", np.nan)
    away_xg_atk = row.get("away_xg_attack_strength", np.nan)
    away_xg_def = row.get("away_xg_defense_strength", np.nan)

    if any(np.isnan(v) for v in [home_xg_atk, home_xg_def, away_xg_atk, away_xg_def]):
        return None

    # Calibrated against 2023-2025 actual goal distributions (760 matches).
    # Old: league_avg=1.38, hb=1.08, ab=0.92 → total xG 2.817 (actual 2.586), +4.2% Over bias
    # New: league_avg=1.28, hb=1.06, ab=0.94 → total xG 2.614 (actual 2.586), -0.6% Over bias
    league_avg = 1.28
    home_xg = home_xg_atk * away_xg_def * league_avg
    away_xg = away_xg_atk * home_xg_def * league_avg

    home_xg = max(0.4, min(3.5, home_xg))
    away_xg = max(0.3, min(3.0, away_xg))

    home_xg *= 1.06
    away_xg *= 0.94

    probs = _poisson_probs(home_xg, away_xg)

    # Draw inflation (must match production XGPredictor.predict)
    xg_gap = abs(home_xg - away_xg)
    draw_inflate = max(0.90, min(1.55, 1.55 - 0.20 * xg_gap))
    probs["prob_D"] *= draw_inflate
    total = probs["prob_H"] + probs["prob_D"] + probs["prob_A"]
    probs["prob_H"] /= total
    probs["prob_D"] /= total
    probs["prob_A"] /= total

    return probs


def predict_formation_adjusted(
    row: pd.Series, base_probs: Dict[str, float]
) -> Dict[str, float]:
    """Apply formation matchup adjustment to base probabilities."""
    home_form = row.get("home_formation", "")
    away_form = row.get("away_formation", "")

    if not home_form or not away_form or pd.isna(home_form) or pd.isna(away_form):
        return base_probs

    fm_home_rate = row.get("formation_matchup_home_rate", np.nan)
    fm_draw_rate = row.get("formation_matchup_draw_rate", np.nan)

    if np.isnan(fm_home_rate) or np.isnan(fm_draw_rate):
        return base_probs

    fm_away_rate = 1.0 - fm_home_rate - fm_draw_rate

    conf = row.get("formation_confidence", "low")
    if conf == "high":
        strength = 0.15
    elif conf == "medium":
        strength = 0.08
    else:
        return base_probs

    adj_h = max(-0.05, min(0.05, (fm_home_rate - BASE_RATES["H"]) * strength))
    adj_d = max(-0.05, min(0.05, (fm_draw_rate - BASE_RATES["D"]) * strength))
    adj_a = max(-0.05, min(0.05, (fm_away_rate - BASE_RATES["A"]) * strength))

    prob_H = max(0.05, base_probs["prob_H"] + adj_h)
    prob_D = max(0.05, base_probs["prob_D"] + adj_d)
    prob_A = max(0.05, base_probs["prob_A"] + adj_a)

    total = prob_H + prob_D + prob_A
    return {"prob_H": prob_H / total, "prob_D": prob_D / total, "prob_A": prob_A / total}


def predict_market(row: pd.Series) -> Optional[Dict[str, float]]:
    """Market-implied probabilities from Pinnacle closing odds."""
    psh = row.get("odds_PSH", 0)
    psd = row.get("odds_PSD", 0)
    psa = row.get("odds_PSA", 0)

    if not all(v and v > 1.0 for v in [psh, psd, psa]):
        return None

    raw_h = 1.0 / psh
    raw_d = 1.0 / psd
    raw_a = 1.0 / psa
    total = raw_h + raw_d + raw_a

    if total <= 0:
        return None

    return {
        "prob_H": raw_h / total,
        "prob_D": raw_d / total,
        "prob_A": raw_a / total,
    }


def _poisson_probs(home_xg: float, away_xg: float, max_goals: int = 10) -> Dict[str, float]:
    """Convert xG to win probabilities via Poisson."""
    home_p = [poisson.pmf(g, home_xg) for g in range(max_goals)]
    away_p = [poisson.pmf(g, away_xg) for g in range(max_goals)]

    prob_H = prob_D = prob_A = 0.0
    for h in range(max_goals):
        for a in range(max_goals):
            p = home_p[h] * away_p[a]
            if h > a:
                prob_H += p
            elif h == a:
                prob_D += p
            else:
                prob_A += p

    total = prob_H + prob_D + prob_A
    return {"prob_H": prob_H / total, "prob_D": prob_D / total, "prob_A": prob_A / total}


def predict_ensemble(
    row: pd.Series,
    weights: Optional[Dict[str, float]] = None,
    use_formation: bool = False,
) -> Dict[str, float]:
    """Generate ensemble prediction for a single match row."""
    if weights is None:
        weights = BACKTEST_WEIGHTS_WITH_FORMATION if use_formation else BACKTEST_WEIGHTS

    predictions = {}

    # Factor-based
    factor_probs = predict_factor_based(row)
    predictions["factor"] = factor_probs

    # xG Poisson
    xg_probs = predict_xg_poisson(row)
    if xg_probs:
        predictions["xg"] = xg_probs

    # Market
    market_probs = predict_market(row)
    if market_probs:
        predictions["market"] = market_probs

    # ML classifier (multi-model ensemble or single CatBoost)
    ml_model = _get_ml_classifier()
    if ml_model is not None:
        try:
            if _ml_is_ensemble:
                feature_names = ml_model.feature_names
            else:
                feature_names = list(ml_model.feature_names_)
            available = [f for f in feature_names if f in row.index]
            if len(available) >= len(feature_names) * 0.5:
                X = pd.DataFrame([row[available].values], columns=available)
                for f in feature_names:
                    if f not in X.columns:
                        X[f] = 0
                X = X[feature_names]
                proba = ml_model.predict_proba(X)[0]
                # Temperature scaling T=0.75 (synced with production 2026-03-01).
                # T=0.40 was over-aggressive: crushed draw probability, requiring
                # excessive compensating mechanisms. T=0.75 preserves draw signal.
                eps = 1e-10
                logits = np.log(proba + eps)
                scaled = np.exp(logits / 0.75)
                proba = scaled / scaled.sum()
                predictions["ml"] = {
                    "prob_H": float(proba[0]),
                    "prob_D": float(proba[1]),
                    "prob_A": float(proba[2]),
                }
        except Exception:
            pass

    # Player xG: use xG Poisson with player-level weighting (simplified)
    if xg_probs and "player_xg" in weights and weights.get("player_xg", 0) > 0:
        # Approximate player_xg as xG with slight noise reduction
        predictions["player_xg"] = xg_probs

    # Formation adjustment (applied to the factor+xg average, not standalone)
    if use_formation:
        base = {}
        base_methods = [v for k, v in predictions.items() if k in ("factor", "xg")]
        if base_methods:
            for outcome in ("prob_H", "prob_D", "prob_A"):
                base[outcome] = np.mean([m[outcome] for m in base_methods])
            formation_probs = predict_formation_adjusted(row, base)
            predictions["formation"] = formation_probs

    # Weighted average
    prob_H = prob_D = prob_A = 0.0
    total_w = 0.0
    for method, probs in predictions.items():
        w = weights.get(method, 0)
        if w > 0:
            prob_H += w * probs["prob_H"]
            prob_D += w * probs["prob_D"]
            prob_A += w * probs["prob_A"]
            total_w += w

    if total_w > 0:
        prob_H /= total_w
        prob_D /= total_w
        prob_A /= total_w

    # Draw boost: 1.12 (synced with production 2026-03-01).
    # Reduced from 1.28 because T=0.75 compresses draws less than T=0.40.
    draw_boost = 1.12
    if draw_boost != 1.0:
        prob_D *= draw_boost
        total = prob_H + prob_D + prob_A
        prob_H /= total
        prob_D /= total
        prob_A /= total

    # Post-ensemble temperature: T=0.90 (synced with production 2026-03-01).
    # T<1.0 sharpens underconfident predictions. Calibration sweep: ECE 0.0587→0.0329.
    post_T = 0.90
    if post_T != 1.0:
        _eps = 1e-10
        logits = [np.log(prob_H + _eps), np.log(prob_D + _eps), np.log(prob_A + _eps)]
        scaled = [np.exp(l / post_T) for l in logits]
        s_total = sum(scaled)
        prob_H, prob_D, prob_A = scaled[0] / s_total, scaled[1] / s_total, scaled[2] / s_total

    # Isotonic recalibration (Phase 5): per-class isotonic regression fitted
    # on walk-forward OOF predictions. Improves edge quality near thresholds.
    # Backtest: +6pp ROI on flat bets, +3x on Kelly (both directions validated).
    iso_cal = _get_isotonic_calibrator()
    if iso_cal is not None:
        raw = np.array([[prob_H, prob_D, prob_A]])
        calibrated = np.zeros_like(raw)
        for ci, ir in enumerate(iso_cal):
            calibrated[:, ci] = ir.predict(raw[:, ci])
        row_sum = calibrated.sum()
        if row_sum > 0:
            calibrated /= row_sum
            prob_H, prob_D, prob_A = float(calibrated[0, 0]), float(calibrated[0, 1]), float(calibrated[0, 2])

    # Safety clip: ensure all probabilities are valid after all adjustments
    prob_H = max(0.001, min(0.999, prob_H))
    prob_D = max(0.001, min(0.999, prob_D))
    prob_A = max(0.001, min(0.999, prob_A))
    total = prob_H + prob_D + prob_A
    prob_H /= total
    prob_D /= total
    prob_A /= total

    return {"prob_H": prob_H, "prob_D": prob_D, "prob_A": prob_A}


# =============================================================================
# METRICS ENGINE
# =============================================================================

def ranked_probability_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Ranked Probability Score -- proper ordinal metric for H/D/A.

    RPS penalises predictions that are far from the true outcome in ordinal
    space (Home < Draw < Away). Lower is better. Fully vectorized.
    """
    n = len(y_true)
    y_onehot = np.zeros_like(y_proba)
    y_onehot[np.arange(n), y_true] = 1.0
    cum_pred = np.cumsum(y_proba, axis=1)
    cum_true = np.cumsum(y_onehot, axis=1)
    return float(np.mean(np.mean((cum_pred - cum_true) ** 2, axis=1)))


def expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error -- weighted average of per-bin calibration gap."""
    n = len(y_true)
    y_pred = np.argmax(y_proba, axis=1)
    confidences = np.max(y_proba, axis=1)
    correct = (y_pred == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        count = mask.sum()
        if count == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc = correct[mask].mean()
        ece += (count / n) * abs(avg_acc - avg_conf)
    return float(ece)


def _multiclass_brier(y_int: np.ndarray, y_proba: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and one-hot truth."""
    n_classes = y_proba.shape[1]
    y_onehot = np.zeros((len(y_int), n_classes))
    y_onehot[np.arange(len(y_int)), y_int] = 1
    return float(np.mean((y_proba - y_onehot) ** 2))


def compute_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """Compute annualized Sharpe ratio from per-bet returns.

    Assumes ~10 bets per matchweek, ~38 matchweeks per season.
    """
    if not returns or len(returns) < 2:
        return 0.0
    arr = np.array(returns)
    mean_ret = arr.mean() - risk_free_rate
    std_ret = arr.std(ddof=1)
    if std_ret < 1e-10:
        return 0.0
    # Annualize: assume ~380 bet periods per season
    return float(mean_ret / std_ret * np.sqrt(380))


def compute_max_drawdown(cumulative_pnl: List[float]) -> float:
    """Compute maximum drawdown from cumulative P&L series."""
    if not cumulative_pnl:
        return 0.0
    peak = cumulative_pnl[0]
    max_dd = 0.0
    for val in cumulative_pnl:
        peak = max(peak, val)
        if peak > 0:
            dd = (peak - val) / peak
            max_dd = max(max_dd, dd)
    return max_dd


# =============================================================================
# TEMPERATURE SCALING (post-hoc calibration)
# =============================================================================

def _fit_temperature(
    predictions: List[Dict[str, float]],
    actuals: List[str],
) -> float:
    """Find temperature T that minimizes NLL on 3-class predictions.

    Temperature scaling is the simplest post-hoc calibration method:
    one parameter T applied uniformly to all logits. When T > 1 the model
    becomes less confident (probabilities move toward uniform 1/3). When
    T < 1 it becomes more confident. We optimize T to minimize negative
    log-likelihood, which is a *proper* scoring rule — the optimizer can't
    cheat by just predicting 1/3 everywhere, because NLL also rewards
    sharpness when the model is correct.

    With only 1 parameter fit on 760 data points, overfitting risk is
    negligible compared to e.g. isotonic regression (non-parametric).
    """
    from scipy.optimize import minimize_scalar

    probs = np.array([
        [p["prob_H"], p["prob_D"], p["prob_A"]] for p in predictions
    ])
    labels = np.array([LABEL_MAP[a] for a in actuals])
    log_probs = np.log(np.clip(probs, 1e-10, 1.0))

    def nll(T):
        scaled = log_probs / T
        # log-sum-exp for numerical stability
        max_scaled = scaled.max(axis=1, keepdims=True)
        log_norm = np.log(np.exp(scaled - max_scaled).sum(axis=1)) + max_scaled.squeeze()
        log_calibrated = scaled[np.arange(len(labels)), labels] - log_norm
        return -log_calibrated.mean()

    result = minimize_scalar(nll, bounds=(0.5, 5.0), method="bounded")
    return float(result.x)


def _apply_temperature(
    prob_H: float, prob_D: float, prob_A: float, T: float,
) -> Tuple[float, float, float]:
    """Apply temperature scaling to a single (H, D, A) probability triple.

    Returns calibrated (prob_H, prob_D, prob_A) that sum to 1.
    """
    log_probs = np.log(np.clip([prob_H, prob_D, prob_A], 1e-10, 1.0))
    scaled = log_probs / T
    scaled -= scaled.max()  # numerical stability
    exp_scaled = np.exp(scaled)
    calibrated = exp_scaled / exp_scaled.sum()
    return float(calibrated[0]), float(calibrated[1]), float(calibrated[2])


# =============================================================================
# CONFIDENCE LEVEL ASSIGNMENT (mirrors production thresholds)
# =============================================================================

def _assign_confidence_level(max_prob: float) -> str:
    """Assign confidence level based on max predicted probability.

    Mirrors the production thresholds in ensemble_prediction_engine.py:3395-3404
    without the factor_bonus adjustment (unavailable in backtest context).
    """
    if max_prob >= 0.58:
        return "VERY HIGH"
    elif max_prob >= 0.48:
        return "HIGH"
    elif max_prob >= 0.40:
        return "MEDIUM-HIGH"
    elif max_prob >= 0.33:
        return "MEDIUM"
    else:
        return "LOW"


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

class BacktestEngine:
    """Unified backtesting engine combining all backtest modes."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.df: Optional[pd.DataFrame] = None
        self.results: Dict[str, Any] = {}

    # -----------------------------------------------------------------
    # DATA LOADING
    # -----------------------------------------------------------------

    def load_data(self) -> bool:
        """Load pre-computed features from features.parquet."""
        path = features_path()
        if not path.exists():
            log.error("features.parquet not found at %s", path)
            return False

        self.df = pd.read_parquet(path)
        self.df = self.df.sort_values("match_date").reset_index(drop=True)
        self.df = self.df.dropna(subset=["result"])

        # Ensure result is valid
        self.df = self.df[self.df["result"].isin(["H", "D", "A"])].copy()

        log.info("Loaded %d matches from features.parquet", len(self.df))
        log.info("Seasons available: %s", sorted(self.df["season"].unique()))

        # Check for odds columns
        odds_cols = ["odds_PSH", "odds_PSD", "odds_PSA", "odds_B365H", "odds_B365D", "odds_B365A"]
        available_odds = [c for c in odds_cols if c in self.df.columns]
        log.info("Available odds columns: %s", available_odds)

        return True

    def _get_season_data(self, season: str) -> pd.DataFrame:
        """Get filtered data for a specific season."""
        season_df = self.df[self.df["season"] == season].copy()
        if season_df.empty:
            log.warning("No data found for season %s", season)
        else:
            log.info("Season %s: %d matches", season, len(season_df))
        return season_df

    # -----------------------------------------------------------------
    # MODE: ACCURACY (walk-forward ensemble evaluation)
    # -----------------------------------------------------------------

    def backtest_season(self, season: str) -> Dict[str, Any]:
        """Per-season evaluation using the factor/xG/market ensemble."""
        log.info("Backtesting season: %s (accuracy mode)", season)
        df = self._get_season_data(season)
        if df.empty:
            return {"error": f"No data for {season}"}

        predictions = []
        actuals = []
        brier_scores = []
        rps_scores = []
        log_losses = []

        for _, row in df.iterrows():
            actual = row["result"]
            pred = predict_ensemble(
                row,
                use_formation=self.config.use_formation,
            )

            # Attach market probs for comparison
            market = predict_market(row)
            if market:
                pred["_market"] = market

            predictions.append(pred)
            actuals.append(actual)

            # Compute per-match metrics
            prob_vec = np.array([pred["prob_H"], pred["prob_D"], pred["prob_A"]])
            actual_idx = LABEL_MAP[actual]
            actual_vec = np.zeros(3)
            actual_vec[actual_idx] = 1.0

            # Brier
            brier = float(np.sum((prob_vec - actual_vec) ** 2))
            brier_scores.append(brier)

            # RPS
            cum_pred = np.cumsum(prob_vec)
            cum_true = np.cumsum(actual_vec)
            rps = float(np.mean((cum_pred - cum_true) ** 2))
            rps_scores.append(rps)

            # Log loss
            p_actual = max(1e-10, prob_vec[actual_idx])
            log_losses.append(-np.log(p_actual))

        # Evaluate
        eval_result = self._evaluate_predictions(predictions, actuals)
        eval_result["mean_brier"] = round(np.mean(brier_scores), 4) if brier_scores else None
        eval_result["mean_rps"] = round(np.mean(rps_scores), 4) if rps_scores else None
        eval_result["mean_log_loss"] = round(np.mean(log_losses), 4) if log_losses else None

        return eval_result

    def _evaluate_predictions(
        self,
        predictions: List[Dict[str, float]],
        actuals: List[str],
        store_records: bool = False,
    ) -> Dict[str, Any]:
        """Compute accuracy, log loss, Brier score, and calibration.

        When store_records=True, also collects per-prediction records with
        (predicted_prob, actual_outcome, confidence_level) for Beta K calibration.
        """
        n = len(predictions)
        if n == 0:
            return {}

        correct = 0
        total_log_loss = 0.0
        total_brier = 0.0

        # Calibration buckets
        buckets = {
            "25-35": {"predicted": [], "actual": []},
            "35-45": {"predicted": [], "actual": []},
            "45-55": {"predicted": [], "actual": []},
            "55-65": {"predicted": [], "actual": []},
            "65-80": {"predicted": [], "actual": []},
            "80-100": {"predicted": [], "actual": []},
        }

        outcome_stats = {
            "H": {"correct": 0, "total": 0},
            "D": {"correct": 0, "total": 0},
            "A": {"correct": 0, "total": 0},
        }

        # High confidence tracking
        high_conf_correct = 0
        high_conf_total = 0

        # Per-prediction records for calibration
        prediction_records = [] if store_records else None

        for pred, actual in zip(predictions, actuals):
            probs = [pred["prob_H"], pred["prob_D"], pred["prob_A"]]
            outcomes = ["H", "D", "A"]
            predicted = outcomes[np.argmax(probs)]
            max_prob = max(probs)

            if predicted == actual:
                correct += 1

            # High confidence
            if max_prob >= self.config.confidence_threshold:
                high_conf_total += 1
                if predicted == actual:
                    high_conf_correct += 1

            outcome_stats[predicted]["total"] += 1
            if predicted == actual:
                outcome_stats[predicted]["correct"] += 1

            # Log loss
            actual_probs = {"H": pred["prob_H"], "D": pred["prob_D"], "A": pred["prob_A"]}
            p_actual = max(1e-10, actual_probs[actual])
            total_log_loss += -np.log(p_actual)

            # Brier score (multi-class)
            for i, outcome in enumerate(outcomes):
                actual_val = 1.0 if actual == outcome else 0.0
                total_brier += (probs[i] - actual_val) ** 2

            # Calibration bucket
            if max_prob < 0.35:
                bucket = "25-35"
            elif max_prob < 0.45:
                bucket = "35-45"
            elif max_prob < 0.55:
                bucket = "45-55"
            elif max_prob < 0.65:
                bucket = "55-65"
            elif max_prob < 0.80:
                bucket = "65-80"
            else:
                bucket = "80-100"

            buckets[bucket]["predicted"].append(max_prob)
            buckets[bucket]["actual"].append(1.0 if predicted == actual else 0.0)

            # Collect per-prediction record for Beta K calibration
            if store_records:
                prediction_records.append({
                    "predicted_prob": round(max_prob, 4),
                    "actual_outcome": 1 if predicted == actual else 0,
                    "confidence_level": _assign_confidence_level(max_prob),
                    "prob_H": round(pred["prob_H"], 4),
                    "prob_D": round(pred["prob_D"], 4),
                    "prob_A": round(pred["prob_A"], 4),
                    "predicted": predicted,
                    "actual": actual,
                    "max_prob": round(max_prob, 4),
                })

        accuracy = correct / n
        log_loss_val = total_log_loss / n
        brier = total_brier / (n * 3)

        # Build calibration table
        calibration = {}
        for bucket_name, data in buckets.items():
            if data["predicted"]:
                calibration[bucket_name] = {
                    "n_matches": len(data["predicted"]),
                    "avg_predicted": round(np.mean(data["predicted"]), 4),
                    "avg_actual": round(np.mean(data["actual"]), 4),
                }

        # Per-outcome accuracy
        outcome_accuracy = {}
        for outcome in ["H", "D", "A"]:
            stats = outcome_stats[outcome]
            if stats["total"] > 0:
                outcome_accuracy[outcome] = {
                    "accuracy": stats["correct"] / stats["total"],
                    "total": stats["total"],
                    "correct": stats["correct"],
                }

        result = {
            "n_matches": n,
            "accuracy": accuracy,
            "log_loss": log_loss_val,
            "brier_score": brier,
            "calibration": calibration,
            "outcome_accuracy": outcome_accuracy,
            "correct": correct,
            "high_conf_total": high_conf_total,
            "high_conf_correct": high_conf_correct,
            "high_conf_accuracy": (
                high_conf_correct / high_conf_total if high_conf_total > 0 else 0
            ),
        }

        if store_records:
            result["prediction_records"] = prediction_records

        return result

    def _compare_with_market(
        self,
        our_probs: List[Dict[str, float]],
        actuals: List[str],
    ) -> Dict[str, Any]:
        """Head-to-head comparison with Pinnacle closing line."""
        market_probs = []
        model_probs = []
        filtered_actuals = []

        for pred, actual in zip(our_probs, actuals):
            market = pred.get("_market")
            if market is None:
                continue
            market_probs.append(market)
            model_probs.append({
                "prob_H": pred["prob_H"],
                "prob_D": pred["prob_D"],
                "prob_A": pred["prob_A"],
            })
            filtered_actuals.append(actual)

        if not filtered_actuals:
            return {}

        model_eval = self._evaluate_predictions(model_probs, filtered_actuals)
        market_eval = self._evaluate_predictions(market_probs, filtered_actuals)

        return {
            "n_matches": len(filtered_actuals),
            "model": {
                "accuracy": model_eval["accuracy"],
                "log_loss": model_eval["log_loss"],
                "brier": model_eval["brier_score"],
            },
            "market": {
                "accuracy": market_eval["accuracy"],
                "log_loss": market_eval["log_loss"],
                "brier": market_eval["brier_score"],
            },
            "accuracy_gap": model_eval["accuracy"] - market_eval["accuracy"],
            "log_loss_gap": model_eval["log_loss"] - market_eval["log_loss"],
        }

    def full_backtest_accuracy(self) -> Dict[str, Any]:
        """Run walk-forward accuracy backtest across all configured seasons."""
        all_predictions = []
        all_actuals = []
        season_results = {}
        store = self.config.store_predictions

        for season in self.config.seasons:
            log.info("\n" + "=" * 60)
            log.info("BACKTESTING SEASON: %s", season)
            log.info("=" * 60)

            df = self._get_season_data(season)
            if df.empty:
                continue

            predictions = []
            actuals = []

            for _, row in df.iterrows():
                actual = row["result"]
                pred = predict_ensemble(
                    row, use_formation=self.config.use_formation,
                )
                market = predict_market(row)
                if market:
                    pred["_market"] = market

                predictions.append(pred)
                actuals.append(actual)

            eval_result = self._evaluate_predictions(
                predictions, actuals, store_records=store,
            )
            season_results[season] = eval_result
            all_predictions.extend(predictions)
            all_actuals.extend(actuals)

            log.info(
                "Season %s: accuracy=%s, log_loss=%.4f, brier=%.4f",
                season,
                f"{eval_result['accuracy']:.1%}",
                eval_result["log_loss"],
                eval_result["brier_score"],
            )

        # Overall evaluation
        overall = self._evaluate_predictions(
            all_predictions, all_actuals, store_records=store,
        )
        overall["seasons"] = season_results

        # Strip per-prediction records from per-season results to avoid bloat
        # (overall already has the full aggregated list)
        if store:
            for season_data in season_results.values():
                season_data.pop("prediction_records", None)

        # Market comparison
        if self.config.compare_baseline:
            comparison = self._compare_with_market(all_predictions, all_actuals)
            overall["market_comparison"] = comparison

            # Also run with formation for comparison
            if not self.config.use_formation:
                log.info("Running WITH formation adjustment for comparison...")
                orig_formation = self.config.use_formation
                self.config.use_formation = True
                formation_result = self.full_backtest_accuracy()
                self.config.use_formation = orig_formation

                overall["formation_comparison"] = {
                    "without_formation": {
                        "accuracy": overall["accuracy"],
                        "log_loss": overall["log_loss"],
                    },
                    "with_formation": {
                        "accuracy": formation_result["accuracy"],
                        "log_loss": formation_result["log_loss"],
                    },
                    "accuracy_delta": formation_result["accuracy"] - overall["accuracy"],
                    "log_loss_delta": formation_result["log_loss"] - overall["log_loss"],
                }

        # RPS and ECE on pooled predictions
        if all_predictions:
            y_int = np.array([LABEL_MAP[a] for a in all_actuals])
            y_proba = np.array([
                [p["prob_H"], p["prob_D"], p["prob_A"]] for p in all_predictions
            ])
            overall["rps"] = round(ranked_probability_score(y_int, y_proba), 4)
            overall["ece"] = round(expected_calibration_error(y_int, y_proba), 4)

        # Temperature scaling: fit T as diagnostic info.
        # prediction_records keep RAW probabilities so calibrate_beta_k()
        # computes K values that match production (which uses raw probs).
        # T is saved separately for future production integration.
        if all_predictions:
            T = _fit_temperature(all_predictions, all_actuals)
            overall["temperature"] = round(T, 4)
            log.info("Fitted temperature T=%.4f (T>1 means model was overconfident)", T)

            # Save temperature artifact for future production integration
            cal_dir = DATA_DIR / "calibration"
            cal_dir.mkdir(parents=True, exist_ok=True)
            temp_path = cal_dir / "temperature.json"
            with open(temp_path, "w") as f:
                json.dump({
                    "temperature": round(T, 4),
                    "fitted_on_n": len(all_predictions),
                    "fitted_on_seasons": self.config.seasons,
                    "raw_log_loss": overall["log_loss"],
                }, f, indent=2)
            log.info("Temperature saved to %s", temp_path)

        return overall

    # -----------------------------------------------------------------
    # MODE: BETTING (full P&L simulation from backtest_betting_system.py)
    # -----------------------------------------------------------------

    def backtest_betting(self, season: Optional[str] = None) -> Dict[str, Any]:
        """Simulate betting P&L using the ensemble + Kelly criterion.

        This is a simplified value betting simulation that does not depend on
        the full betting system imports. It uses the ensemble predictions and
        Pinnacle odds directly.
        """
        target_season = season or self.config.seasons[-1]
        log.info("Running betting simulation for %s", target_season)

        df = self._get_season_data(target_season)
        if df.empty:
            return {"error": f"No data for {target_season}"}

        bankroll = self.config.bankroll
        initial_bankroll = bankroll
        all_bets = []
        cumulative_pnl = []
        market_stats = defaultdict(lambda: {"bets": 0, "wins": 0, "staked": 0.0, "profit": 0.0})
        per_bet_returns = []

        # Sort by match date for walk-forward simulation
        df = df.sort_values("match_date")

        for _, row in df.iterrows():
            actual = row["result"]
            home_score = row.get("home_score", 0)
            away_score = row.get("away_score", 0)
            if pd.isna(home_score) or pd.isna(away_score):
                continue

            home_score = int(home_score)
            away_score = int(away_score)

            # Get ensemble prediction
            pred = predict_ensemble(row, use_formation=self.config.use_formation)

            # Get odds
            psh = row.get("odds_PSH", 0)
            psd = row.get("odds_PSD", 0)
            psa = row.get("odds_PSA", 0)

            if not all(v and v > 1.0 for v in [psh, psd, psa]):
                continue

            # Remove overround for sharp probabilities
            raw = np.array([1 / psh, 1 / psd, 1 / psa])
            sharp_probs = raw / raw.sum()

            # Check value bets for each outcome
            outcomes = [
                ("H", pred["prob_H"], sharp_probs[0], psh),
                ("D", pred["prob_D"], sharp_probs[1], psd),
                ("A", pred["prob_A"], sharp_probs[2], psa),
            ]

            for outcome_label, model_prob, sharp_prob, odds in outcomes:
                edge = model_prob - sharp_prob

                if edge < self.config.min_edge or edge > self.config.max_edge:
                    continue

                # Kelly criterion
                if odds <= 1:
                    continue
                kelly_stake = self.config.kelly_fraction * (model_prob * odds - 1) / (odds - 1)
                kelly_stake = max(0, min(kelly_stake, 0.025))  # Cap at 2.5% of bankroll (synced with production)

                stake = bankroll * kelly_stake
                if stake < 1.0:
                    continue

                # Settle bet
                won = (outcome_label == actual)
                profit = stake * (odds - 1) if won else -stake
                bankroll += profit

                bet_return = profit / stake if stake > 0 else 0
                per_bet_returns.append(bet_return)

                all_bets.append({
                    "match": f"{row.get('home_team', '?')} vs {row.get('away_team', '?')}",
                    "market": "1X2",
                    "selection": {"H": "Home", "D": "Draw", "A": "Away"}[outcome_label],
                    "model_prob": round(model_prob, 4),
                    "sharp_prob": round(sharp_prob, 4),
                    "edge": round(edge, 4),
                    "odds": odds,
                    "stake": round(stake, 2),
                    "result": "WIN" if won else "LOSS",
                    "profit": round(profit, 2),
                })

                market_stats["1X2"]["bets"] += 1
                market_stats["1X2"]["wins"] += 1 if won else 0
                market_stats["1X2"]["staked"] += stake
                market_stats["1X2"]["profit"] += profit

            cumulative_pnl.append(bankroll - initial_bankroll)

        # Compute summary
        n_bets = len(all_bets)
        n_wins = sum(1 for b in all_bets if b["result"] == "WIN")
        total_staked = sum(b["stake"] for b in all_bets)
        total_pnl = bankroll - initial_bankroll

        result = {
            "season": target_season,
            "initial_bankroll": initial_bankroll,
            "final_bankroll": round(bankroll, 2),
            "total_bets": n_bets,
            "wins": n_wins,
            "win_rate": round(n_wins / max(n_bets, 1), 4),
            "total_staked": round(total_staked, 2),
            "total_pnl": round(total_pnl, 2),
            "roi": round(total_pnl / max(total_staked, 1) * 100, 2),
            "yield_pct": round(total_pnl / initial_bankroll * 100, 2),
            "sharpe_ratio": round(compute_sharpe_ratio(per_bet_returns), 4),
            "max_drawdown": round(compute_max_drawdown(
                [initial_bankroll + c for c in cumulative_pnl]
            ), 4),
            "market_breakdown": dict(market_stats),
            "bets": all_bets[:50],  # Save first 50 for inspection
        }

        # Monthly P&L
        monthly = defaultdict(lambda: {"bets": 0, "profit": 0.0, "staked": 0.0})
        for bet in all_bets:
            # Extract month from match name is not reliable; use cumulative index
            monthly["all"]["bets"] += 1
            monthly["all"]["profit"] += bet["profit"]
            monthly["all"]["staked"] += bet["stake"]
        result["monthly_summary"] = dict(monthly)

        return result

    # -----------------------------------------------------------------
    # MODE: VALUE (value betting with real odds from backtest_with_odds.py)
    # -----------------------------------------------------------------

    def backtest_value_betting(self) -> Dict[str, Any]:
        """Value betting simulation using real historical odds + Kelly.

        Combines flat betting and Kelly-sized value betting approaches.
        """
        log.info("Running value betting backtest across seasons: %s", self.config.seasons)

        # Set up odds columns
        if "odds_PSH" in self.df.columns:
            self.df["home_odds"] = self.df["odds_PSH"]
            self.df["draw_odds"] = self.df["odds_PSD"]
            self.df["away_odds"] = self.df["odds_PSA"]
            odds_source = "pinnacle"
        elif "odds_B365H" in self.df.columns:
            self.df["home_odds"] = self.df["odds_B365H"]
            self.df["draw_odds"] = self.df["odds_B365D"]
            self.df["away_odds"] = self.df["odds_B365A"]
            odds_source = "bet365"
        else:
            self.df["home_odds"] = 1.80
            self.df["draw_odds"] = 3.40
            self.df["away_odds"] = 3.60
            odds_source = "default"

        log.info("Odds source: %s", odds_source)

        season_results = []
        total_correct = 0
        total_matches = 0
        total_flat_stake = 0.0
        total_flat_returns = 0.0
        total_value_bets = 0
        total_value_wins = 0
        total_kelly_stake = 0.0
        total_kelly_returns = 0.0

        for season in self.config.seasons:
            test_df = self.df[self.df["season"] == season].copy()
            test_df = test_df[
                test_df[["home_odds", "draw_odds", "away_odds"]].notna().all(axis=1)
            ]

            if len(test_df) == 0:
                continue

            correct = 0
            total = 0
            flat_stake = 0.0
            flat_returns = 0.0
            value_bets = 0
            value_wins = 0
            kelly_stake = 0.0
            kelly_returns = 0.0
            bankroll = 1000.0
            initial_bankroll = bankroll

            for _, row in test_df.iterrows():
                actual = row["result"]
                odds = {
                    "home": float(row["home_odds"]),
                    "draw": float(row["draw_odds"]),
                    "away": float(row["away_odds"]),
                }

                pred = predict_ensemble(
                    row, use_formation=self.config.use_formation,
                )

                probs = {
                    "home": pred["prob_H"],
                    "draw": pred["prob_D"],
                    "away": pred["prob_A"],
                }

                # Check accuracy
                outcomes = ["H", "D", "A"]
                prob_list = [pred["prob_H"], pred["prob_D"], pred["prob_A"]]
                predicted_idx = np.argmax(prob_list)
                predicted_label = outcomes[predicted_idx]
                predicted_name = {"H": "home", "D": "draw", "A": "away"}[predicted_label]

                is_correct = predicted_label == actual
                if is_correct:
                    correct += 1
                total += 1

                max_prob = max(prob_list)

                # Flat betting: bet on prediction if confidence > threshold
                if max_prob >= self.config.confidence_threshold:
                    stake = 10.0
                    flat_stake += stake

                    if predicted_label == actual:
                        flat_returns += stake * odds[predicted_name]

                # Value betting: check all outcomes for value
                for out_name, out_label in [("home", "H"), ("draw", "D"), ("away", "A")]:
                    model_p = probs[out_name]
                    implied_p = 1.0 / odds[out_name] if odds[out_name] > 1 else 1.0
                    edge = model_p - implied_p

                    if edge < self.config.min_edge:
                        continue

                    value_bets += 1

                    # Kelly sizing
                    kelly_size = self.config.kelly_fraction * (model_p * odds[out_name] - 1) / (odds[out_name] - 1)
                    kelly_size = max(0, min(kelly_size, 0.025))  # 2.5% cap (synced with production)
                    stake = bankroll * kelly_size

                    if stake < 0.5:
                        continue

                    kelly_stake += stake
                    won = (out_label == actual)

                    if won:
                        kelly_returns += stake * odds[out_name]
                        value_wins += 1
                        bankroll += stake * (odds[out_name] - 1)
                    else:
                        bankroll -= stake

            if total == 0:
                continue

            flat_roi = (
                (flat_returns - flat_stake) / flat_stake * 100
                if flat_stake > 0 else 0
            )
            kelly_roi = (
                (kelly_returns - kelly_stake) / kelly_stake * 100
                if kelly_stake > 0 else 0
            )

            sr = {
                "season": season,
                "total_matches": total,
                "correct": correct,
                "accuracy": round(correct / total, 4),
                "flat_betting": {
                    "bets": int(flat_stake / 10) if flat_stake > 0 else 0,
                    "stake": round(flat_stake, 2),
                    "returns": round(flat_returns, 2),
                    "profit": round(flat_returns - flat_stake, 2),
                    "roi": round(flat_roi, 2),
                },
                "value_betting": {
                    "value_bets": value_bets,
                    "value_wins": value_wins,
                    "win_rate": round(value_wins / value_bets, 4) if value_bets > 0 else 0,
                    "stake": round(kelly_stake, 2),
                    "returns": round(kelly_returns, 2),
                    "profit": round(kelly_returns - kelly_stake, 2),
                    "roi": round(kelly_roi, 2),
                    "final_bankroll": round(bankroll, 2),
                    "bankroll_growth": round((bankroll - initial_bankroll) / initial_bankroll * 100, 2),
                },
            }

            season_results.append(sr)
            total_correct += correct
            total_matches += total
            total_flat_stake += flat_stake
            total_flat_returns += flat_returns
            total_value_bets += value_bets
            total_value_wins += value_wins
            total_kelly_stake += kelly_stake
            total_kelly_returns += kelly_returns

            log.info(
                "  %s: %s accuracy (Flat ROI: %+.1f%%, Kelly ROI: %+.1f%%)",
                season, f"{correct/total:.1%}", flat_roi, kelly_roi,
            )

        return {
            "odds_source": odds_source,
            "season_results": season_results,
            "overall": {
                "total_matches": total_matches,
                "correct": total_correct,
                "accuracy": round(total_correct / total_matches, 4) if total_matches > 0 else 0,
                "flat_betting": {
                    "total_stake": round(total_flat_stake, 2),
                    "total_returns": round(total_flat_returns, 2),
                    "profit": round(total_flat_returns - total_flat_stake, 2),
                    "roi": round(
                        (total_flat_returns - total_flat_stake) / total_flat_stake * 100
                        if total_flat_stake > 0 else 0, 2
                    ),
                },
                "value_betting": {
                    "total_value_bets": total_value_bets,
                    "value_wins": total_value_wins,
                    "win_rate": round(total_value_wins / total_value_bets, 4) if total_value_bets > 0 else 0,
                    "total_stake": round(total_kelly_stake, 2),
                    "total_returns": round(total_kelly_returns, 2),
                    "profit": round(total_kelly_returns - total_kelly_stake, 2),
                    "roi": round(
                        (total_kelly_returns - total_kelly_stake) / total_kelly_stake * 100
                        if total_kelly_stake > 0 else 0, 2
                    ),
                },
            },
        }

    # -----------------------------------------------------------------
    # MODE: ML-WALKFORWARD (from ml/walk_forward.py)
    # -----------------------------------------------------------------

    def backtest_ml_walkforward(self) -> Dict[str, Any]:
        """Run ML model walk-forward cross-validation.

        Uses the walk_forward module for proper time-series CV with purge gap.
        """
        log.info("Running ML walk-forward cross-validation")

        from ml.walk_forward import walk_forward_split, WalkForwardFold
        from ml.evaluation import ranked_probability_score as eval_rps

        # Get feature columns (exclude meta and target columns)
        exclude_cols = {
            "match_id", "match_date", "season", "result", "home_score", "away_score",
            "home_team", "away_team", "ht_result", "referee",
            "home_formation", "away_formation",
        }
        feature_cols = [
            c for c in self.df.columns
            if c not in exclude_cols
            and self.df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
            and not c.startswith("_")
        ]

        log.info("Using %d feature columns", len(feature_cols))

        fold_results = []

        try:
            from catboost import CatBoostClassifier

            for train_df, test_df, fold_info in walk_forward_split(
                self.df,
                min_train_seasons=self.config.min_train_seasons,
                test_seasons=self.config.test_season_size,
                purge_matchweeks=self.config.purge_matchweeks,
            ):
                X_train = train_df[feature_cols].fillna(0)
                y_train = train_df["result"]
                X_test = test_df[feature_cols].fillna(0)
                y_test = test_df["result"]

                # Filter valid labels
                valid_train = y_train.isin(["H", "D", "A"])
                valid_test = y_test.isin(["H", "D", "A"])
                X_train = X_train[valid_train]
                y_train = y_train[valid_train]
                X_test = X_test[valid_test]
                y_test = y_test[valid_test]

                if len(X_train) < 100 or len(X_test) < 10:
                    continue

                # Train CatBoost
                model = CatBoostClassifier(
                    iterations=1500, depth=6, learning_rate=0.03,
                    l2_leaf_reg=3.0, random_seed=42, verbose=0,
                    loss_function="MultiClass", classes_count=3,
                    auto_class_weights="Balanced",
                )
                model.fit(X_train, y_train.map(LABEL_MAP))

                # Predict
                y_pred_int = model.predict(X_test).flatten().astype(int)
                y_pred = np.array(["H", "D", "A"])[y_pred_int]
                y_proba = model.predict_proba(X_test)

                # Ensure proper class ordering (H=0, D=1, A=2)
                y_test_int = y_test.map(LABEL_MAP).values

                # Metrics
                from sklearn.metrics import accuracy_score, log_loss as sk_log_loss

                acc = accuracy_score(y_test_int, y_pred_int)
                try:
                    ll = sk_log_loss(y_test_int, y_proba, labels=[0, 1, 2])
                except ValueError:
                    ll = np.nan

                # Brier
                y_onehot = np.zeros_like(y_proba)
                y_onehot[np.arange(len(y_test_int)), y_test_int] = 1.0
                brier = float(np.mean((y_proba - y_onehot) ** 2))

                # RPS
                rps = eval_rps(y_test_int, y_proba)

                # Per-class accuracy
                home_mask = y_test == "H"
                draw_mask = y_test == "D"
                away_mask = y_test == "A"

                home_acc = (y_pred[home_mask.values] == "H").mean() if home_mask.sum() > 0 else 0
                draw_acc = (y_pred[draw_mask.values] == "D").mean() if draw_mask.sum() > 0 else 0
                away_acc = (y_pred[away_mask.values] == "A").mean() if away_mask.sum() > 0 else 0

                fold_result = {
                    "fold": fold_info.fold_number,
                    "train_period": f"{fold_info.train_start} to {fold_info.train_end}",
                    "test_period": f"{fold_info.test_start} to {fold_info.test_end}",
                    "train_matches": fold_info.train_matches,
                    "test_matches": fold_info.test_matches,
                    "accuracy": round(acc, 4),
                    "log_loss": round(ll, 4) if not np.isnan(ll) else None,
                    "brier_score": round(brier, 4),
                    "rps": round(rps, 4),
                    "home_accuracy": round(home_acc, 4),
                    "draw_accuracy": round(draw_acc, 4),
                    "away_accuracy": round(away_acc, 4),
                }

                fold_results.append(fold_result)

                log.info(
                    "Fold %d: Accuracy=%.3f, LogLoss=%.3f, RPS=%.4f, "
                    "H=%.2f, D=%.2f, A=%.2f",
                    fold_info.fold_number, acc, ll if not np.isnan(ll) else 0,
                    rps, home_acc, draw_acc, away_acc,
                )

                del model
                gc.collect()

        except ImportError as e:
            log.warning("CatBoost not available for ML walk-forward: %s", e)
            return {"error": str(e)}

        if not fold_results:
            return {"error": "No valid folds produced"}

        # Aggregate
        accs = [r["accuracy"] for r in fold_results]
        lls = [r["log_loss"] for r in fold_results if r["log_loss"] is not None]
        briers = [r["brier_score"] for r in fold_results]
        rpss = [r["rps"] for r in fold_results]

        k = len(fold_results)
        acc_se = np.std(accs, ddof=1) / np.sqrt(k) if k > 1 else 0

        return {
            "n_folds": k,
            "mean_accuracy": round(np.mean(accs), 4),
            "accuracy_se": round(acc_se, 4),
            "mean_log_loss": round(np.mean(lls), 4) if lls else None,
            "mean_brier": round(np.mean(briers), 4),
            "mean_rps": round(np.mean(rpss), 4),
            "fold_results": fold_results,
        }

    # -----------------------------------------------------------------
    # FULL BACKTEST DISPATCHER
    # -----------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Run the configured backtest mode(s) and return combined results."""
        if not self.load_data():
            return {"error": "Failed to load data"}

        results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "mode": self.config.mode,
                "seasons": self.config.seasons,
                "strategy": self.config.strategy,
                "use_formation": self.config.use_formation,
                "kelly_fraction": self.config.kelly_fraction,
            },
        }

        modes = (
            ["accuracy", "betting", "value", "ml-walkforward"]
            if self.config.mode == "all"
            else [self.config.mode]
        )

        for mode in modes:
            log.info("\n" + "#" * 70)
            log.info("RUNNING MODE: %s", mode.upper())
            log.info("#" * 70)

            if mode == "accuracy":
                results["accuracy"] = self.full_backtest_accuracy()
            elif mode == "betting":
                # Run betting for each season
                betting_results = {}
                for season in self.config.seasons:
                    betting_results[season] = self.backtest_betting(season)
                results["betting"] = betting_results
            elif mode == "value":
                results["value"] = self.backtest_value_betting()
            elif mode == "ml-walkforward":
                results["ml_walkforward"] = self.backtest_ml_walkforward()
            else:
                log.warning("Unknown mode: %s", mode)

        self.results = results
        return results


# =============================================================================
# REPORT PRINTING
# =============================================================================

def print_report(results: Dict[str, Any]):
    """Print formatted unified backtest report."""
    print("\n" + "=" * 70)
    print("UNIFIED BACKTEST REPORT")
    print("=" * 70)

    config = results.get("config", {})
    print(f"\nMode: {config.get('mode', 'unknown')}")
    print(f"Seasons: {config.get('seasons', [])}")
    print(f"Strategy: {config.get('strategy', 'default')}")
    print(f"Timestamp: {results.get('timestamp', 'N/A')}")

    # --- Accuracy Results ---
    if "accuracy" in results:
        acc = results["accuracy"]
        print("\n" + "-" * 70)
        print("ACCURACY BACKTEST")
        print("-" * 70)

        print(f"\nTotal matches: {acc.get('n_matches', 0)}")
        print(f"Accuracy:      {acc.get('accuracy', 0):.1%} ({acc.get('correct', 0)}/{acc.get('n_matches', 0)})")
        print(f"Log loss:      {acc.get('log_loss', 0):.4f}")
        print(f"Brier score:   {acc.get('brier_score', 0):.4f}")
        print(f"RPS:           {acc.get('rps', 'N/A')}")
        print(f"ECE:           {acc.get('ece', 'N/A')}")
        print(f"High-Conf:     {acc.get('high_conf_accuracy', 0):.1%} ({acc.get('high_conf_correct', 0)}/{acc.get('high_conf_total', 0)})")

        # Per-season breakdown
        if "seasons" in acc:
            print(f"\n{'Season':<15} {'Matches':>8} {'Accuracy':>10} {'Log Loss':>10} {'Brier':>10}")
            print("-" * 55)
            for season, data in sorted(acc["seasons"].items()):
                print(
                    f"{season:<15} {data['n_matches']:>8} {data['accuracy']:>9.1%} "
                    f"{data['log_loss']:>10.4f} {data['brier_score']:>10.4f}"
                )

        # Per-outcome accuracy
        if "outcome_accuracy" in acc:
            print(f"\nOutcome accuracy:")
            for outcome, data in acc["outcome_accuracy"].items():
                label = {"H": "Home", "D": "Draw", "A": "Away"}[outcome]
                print(f"  {label:>6}: {data['accuracy']:.1%} ({data['correct']}/{data['total']})")

        # Calibration
        if "calibration" in acc:
            print(f"\nCalibration:")
            print(f"  {'Bucket':<10} {'Matches':>8} {'Predicted':>10} {'Actual':>10} {'Status':>8}")
            print("  " + "-" * 48)
            for bucket, data in acc["calibration"].items():
                diff = abs(data["avg_predicted"] - data["avg_actual"])
                status = "ok" if diff < 0.05 else "DRIFT" if diff < 0.10 else "BAD"
                print(
                    f"  {bucket:<10} {data['n_matches']:>8} {data['avg_predicted']:>9.1%} "
                    f"{data['avg_actual']:>9.1%} {status:>8}"
                )

        # Market comparison
        mc = acc.get("market_comparison", {})
        if mc:
            print(f"\nVs Pinnacle closing line ({mc['n_matches']} matches):")
            print(f"  {'Metric':<12} {'Model':>10} {'Pinnacle':>10} {'Gap':>10}")
            print("  " + "-" * 42)
            print(
                f"  {'Accuracy':<12} {mc['model']['accuracy']:>9.1%} {mc['market']['accuracy']:>9.1%} "
                f"{mc['accuracy_gap']:>+9.1%}"
            )
            print(
                f"  {'Log loss':<12} {mc['model']['log_loss']:>10.4f} {mc['market']['log_loss']:>10.4f} "
                f"{mc['log_loss_gap']:>+10.4f}"
            )

        # Formation comparison
        fc = acc.get("formation_comparison", {})
        if fc:
            print(f"\nFormation adjustment impact:")
            print(
                f"  Without: accuracy={fc['without_formation']['accuracy']:.1%}, "
                f"log_loss={fc['without_formation']['log_loss']:.4f}"
            )
            print(
                f"  With:    accuracy={fc['with_formation']['accuracy']:.1%}, "
                f"log_loss={fc['with_formation']['log_loss']:.4f}"
            )
            print(
                f"  Delta:   accuracy={fc['accuracy_delta']:+.1%}, "
                f"log_loss={fc['log_loss_delta']:+.4f}"
            )

    # --- Betting Results ---
    if "betting" in results:
        print("\n" + "-" * 70)
        print("BETTING SIMULATION")
        print("-" * 70)

        for season, br in results["betting"].items():
            if "error" in br:
                print(f"\n  {season}: {br['error']}")
                continue

            print(f"\n  Season: {season}")
            print(f"  Total Bets:     {br['total_bets']}")
            print(f"  Win Rate:       {br['win_rate']:.1%}")
            print(f"  Total Staked:   {br['total_staked']:.0f}")
            print(f"  Total P&L:      {br['total_pnl']:+.0f}")
            print(f"  ROI:            {br['roi']:+.1f}%")
            print(f"  Sharpe Ratio:   {br['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown:   {br['max_drawdown']:.1%}")
            print(f"  Final Bankroll: {br['final_bankroll']:.0f}")

    # --- Value Betting Results ---
    if "value" in results:
        vr = results["value"]
        print("\n" + "-" * 70)
        print("VALUE BETTING SIMULATION")
        print("-" * 70)

        print(f"\nOdds Source: {vr.get('odds_source', 'unknown')}")

        for sr in vr.get("season_results", []):
            print(f"\n  {sr['season']}:")
            print(f"    Accuracy: {sr['accuracy']:.1%} ({sr['correct']}/{sr['total_matches']})")
            fb = sr["flat_betting"]
            print(f"    Flat Betting: ROI {fb['roi']:+.1f}% (profit: {fb['profit']:+.1f})")
            vb = sr["value_betting"]
            print(
                f"    Value Betting: ROI {vb['roi']:+.1f}% "
                f"({vb['value_bets']} bets, {vb['win_rate']:.1%} win rate)"
            )

        overall = vr.get("overall", {})
        if overall:
            print(f"\n  OVERALL:")
            print(f"    Accuracy: {overall.get('accuracy', 0):.1%}")
            ofb = overall.get("flat_betting", {})
            print(f"    Flat ROI: {ofb.get('roi', 0):+.1f}%")
            ovb = overall.get("value_betting", {})
            print(f"    Value ROI: {ovb.get('roi', 0):+.1f}%")
            print(f"    Value Bets: {ovb.get('total_value_bets', 0)}")

    # --- ML Walk-Forward Results ---
    if "ml_walkforward" in results:
        wf = results["ml_walkforward"]
        print("\n" + "-" * 70)
        print("ML WALK-FORWARD CROSS-VALIDATION")
        print("-" * 70)

        if "error" in wf:
            print(f"\n  Error: {wf['error']}")
        else:
            k = wf["n_folds"]
            print(f"\n  Folds: {k}")
            print(f"  Mean Accuracy:  {wf['mean_accuracy']:.1%} +/- {wf['accuracy_se']:.1%} (SE)")
            if wf["mean_log_loss"] is not None:
                print(f"  Mean Log Loss:  {wf['mean_log_loss']:.4f}")
            print(f"  Mean Brier:     {wf['mean_brier']:.4f}")
            print(f"  Mean RPS:       {wf['mean_rps']:.4f}")

            print(f"\n  {'Fold':>5} {'Accuracy':>10} {'LogLoss':>10} {'RPS':>8} {'H':>6} {'D':>6} {'A':>6} {'N':>6}")
            print("  " + "-" * 58)
            for fr in wf.get("fold_results", []):
                ll_str = f"{fr['log_loss']:.4f}" if fr["log_loss"] is not None else "N/A"
                print(
                    f"  {fr['fold']:>5} {fr['accuracy']:>9.1%} {ll_str:>10} "
                    f"{fr['rps']:>8.4f} {fr['home_accuracy']:>5.0%} "
                    f"{fr['draw_accuracy']:>5.0%} {fr['away_accuracy']:>5.0%} "
                    f"{fr['test_matches']:>6}"
                )

    print("\n" + "=" * 70)
    print()


# =============================================================================
# CLI INTERFACE
# =============================================================================

def parse_args() -> BacktestConfig:
    """Parse CLI arguments into BacktestConfig."""
    parser = argparse.ArgumentParser(
        description="Unified backtest framework for Serie A predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/backtest_unified.py --mode accuracy --season 2024-2025
  python scripts/backtest_unified.py --mode betting --bankroll 1000
  python scripts/backtest_unified.py --mode value --kelly 0.25
  python scripts/backtest_unified.py --mode ml-walkforward
  python scripts/backtest_unified.py --mode all --season 2023-2024 --season 2024-2025
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["accuracy", "betting", "value", "ml-walkforward", "all"],
        default="accuracy",
        help="Backtest mode (default: accuracy)",
    )
    parser.add_argument(
        "--season",
        action="append",
        dest="seasons",
        help="Season(s) to backtest (can be repeated). Default: 2023-2024, 2024-2025",
    )
    parser.add_argument(
        "--strategy",
        default="default",
        choices=["default", "volume", "selective", "ml_heavy", "xg_dominant"],
        help="Ensemble strategy (default: default)",
    )
    parser.add_argument(
        "--use-formation",
        action="store_true",
        help="Include formation-based adjustments",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare with Pinnacle closing line and formation impact",
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=1000.0,
        help="Starting bankroll for betting simulation (default: 1000)",
    )
    parser.add_argument(
        "--kelly",
        type=float,
        default=0.25,
        help="Kelly fraction for bet sizing (default: 0.25)",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.05,
        help="Minimum edge to place a bet (default: 0.05)",
    )
    parser.add_argument(
        "--max-edge",
        type=float,
        default=0.15,
        help="Maximum edge threshold (default: 0.15)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.50,
        help="Minimum confidence for flat betting (default: 0.50)",
    )
    parser.add_argument(
        "--min-train-seasons",
        type=int,
        default=5,
        help="Minimum training seasons for walk-forward CV (default: 5)",
    )
    parser.add_argument(
        "--purge-matchweeks",
        type=int,
        default=2,
        help="Matchweeks to purge at train/test boundary (default: 2)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run Beta K calibration after backtest completes",
    )
    parser.add_argument(
        "--calibrate-only",
        type=str,
        default=None,
        metavar="PATH",
        help="Run Beta K calibration on existing backtest JSON (skip backtest)",
    )
    parser.add_argument(
        "--no-store-predictions",
        action="store_true",
        help="Disable per-prediction record storage (saves memory/disk)",
    )

    args = parser.parse_args()

    config = BacktestConfig(
        mode=args.mode,
        seasons=args.seasons or ["2023-2024", "2024-2025"],
        strategy=args.strategy,
        use_formation=args.use_formation,
        compare_baseline=args.compare_baseline,
        bankroll=args.bankroll,
        kelly_fraction=args.kelly,
        min_edge=args.min_edge,
        max_edge=args.max_edge,
        confidence_threshold=args.confidence_threshold,
        min_train_seasons=args.min_train_seasons,
        purge_matchweeks=args.purge_matchweeks,
        save_path=args.save,
        store_predictions=not args.no_store_predictions,
    )

    return config, args


def main():
    config, args = parse_args()

    # --calibrate-only: run calibration on existing backtest, skip backtest
    if args.calibrate_only:
        from scripts.betting.parlay_generator import calibrate_beta_k
        calibrate_beta_k(args.calibrate_only)
        return

    log.info("Unified Backtest Framework")
    log.info("Mode: %s", config.mode)
    log.info("Seasons: %s", config.seasons)
    log.info("Formation: %s", "ON" if config.use_formation else "OFF")

    engine = BacktestEngine(config)
    results = engine.run()

    print_report(results)

    # Save results
    save_path = config.save_path
    if save_path is None:
        results_dir = DATA_DIR / "optimization"
        results_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(
            results_dir / f"backtest_unified_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

    # Clean results for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items() if not k.startswith("_")}
        elif isinstance(obj, list):
            return [clean_for_json(item) for item in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    cleaned = clean_for_json(results)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(cleaned, f, indent=2, default=str)
    log.info("Results saved to %s", save_path)

    # --calibrate: run Beta K calibration on the just-saved results
    if args.calibrate:
        from scripts.betting.parlay_generator import calibrate_beta_k
        log.info("Running Beta K calibration on %s", save_path)
        calibrate_beta_k(save_path)


if __name__ == "__main__":
    main()
