"""ML pipeline configuration: all hyperparameters, thresholds, and constants.

Every tunable number in the ML pipeline lives here. No magic numbers
should appear in any other ml/*.py file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Label encoding (derived from the data; the only "hardcoded" mapping)
# ---------------------------------------------------------------------------

LABEL_MAP: Dict[str, int] = {"H": 0, "D": 1, "A": 2}
LABEL_NAMES: Dict[int, str] = {v: k for k, v in LABEL_MAP.items()}
CLASS_LABELS: List[str] = list(LABEL_MAP.keys())       # ["H", "D", "A"]
CLASS_INDICES: List[int] = list(LABEL_MAP.values())     # [0, 1, 2]
N_CLASSES: int = len(LABEL_MAP)

# Column-name constants (single source of truth)
META_COLS = frozenset({"_season", "_match_date", "_league"})
RESULT_COL = "result"
SEASON_COL = "season"
MATCH_DATE_COL = "match_date"

# Supported model types
MODEL_TYPES: List[str] = ["xgboost", "lightgbm", "catboost"]

# File extensions per model type
MODEL_EXTENSIONS: Dict[str, str] = {
    "xgboost": ".json",
    "lightgbm": ".txt",
    "catboost": ".cbm",
}

# Patterns that identify betting-odds-derived feature columns
ODDS_COLUMN_PATTERNS: List[str] = [
    "odds_", "pinnacle_", "implied_prob_", "overround",
    "market_ou_", "sharp_soft_", "ah_line_", "market_goal_total",
    "market_home_prob", "market_away_prob", "market_draw_prob",  # Raw implied probs (leakage)
    "market_overround",                                         # Raw overround (leakage)
    "ou_consistency", "ou_over_prob_best", "ou_under_prob_best",
    "home_prob_best", "draw_prob_best", "away_prob_best",
    "market_elo_disagreement", "goal_total_vs_xg", "ah_x_form",
    "sharp_soft_x_elo",
]

# Odds-derived features to KEEP even with exclude_odds=True.
# These are disagreement/meta features — they capture WHERE the market might
# be wrong rather than echoing market levels. Available at prediction time
# (ensemble engine computes them from live odds).
#
# Why these work: the no-odds CatBoost provides independent signal. Adding
# raw odds (PSH, B365H, pinnacle_home_prob) makes it redundant with the
# market predictor (documented in baselines.md, Feb 16 2026). But disagreement
# features let the model learn NON-LINEAR conditional relationships like
# "when Elo and market disagree by >5% on away games, Elo is usually wrong" —
# which linear ensemble blending can't express.
ODDS_META_KEEP: frozenset = frozenset({
    "market_elo_disagreement",   # Elo vs market disagreement (100% coverage)
    "sharp_soft_home_div",       # Pinnacle vs soft book divergence (31% coverage)
    "sharp_soft_draw_div",       # CatBoost handles NaN natively for these
    "sharp_soft_away_div",
    "odds_consistency",          # Bookmaker agreement level (32% coverage)
    "odds_home_fav",             # Binary home favourite flag (66% coverage)
    "sharp_soft_x_elo",          # Sharp-soft divergence × Elo interaction
    # Odds velocity: direction of sharp money movement (Pinnacle line moves).
    # Captures WHERE sharp bettors are moving before kickoff — strong signal
    # for mispricings that the model can exploit. 65% coverage (Pinnacle).
    "line_vel_pin_home",         # Pinnacle implied prob change: home (open→close)
    "line_vel_pin_draw",         # Pinnacle implied prob change: draw
    "line_vel_pin_away",         # Pinnacle implied prob change: away
    "steam_move_flag",           # Binary: >3% implied prob shift (sharp steam)
})

# Reproducibility seed used everywhere
RANDOM_SEED: int = 42


# ---------------------------------------------------------------------------
# Default model hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Default hyperparameters for gradient boosting models."""

    # Early stopping: use more trees but stop when validation stops improving
    early_stopping_rounds: int = 50
    early_stopping_val_fraction: float = 0.15

    # Draw class weight: "auto" computes per-fold from actual draw rate,
    # float value is a fixed multiplier (legacy). "auto" targets 1/3 effective
    # draw weight (equal-class prior), which adapts to each fold's draw rate.
    draw_weight_mode: str = "auto"
    draw_weight_multiplier: float = 2.0  # Only used when draw_weight_mode != "auto"

    # Time-decay: exponential decay per season gap from the most recent training
    # season. 0.85 means a match 5 seasons old has weight 0.85^5 = 0.44.
    # 1.0 disables time-decay (legacy). Dixon-Coles (1997) recommends ~0.85.
    time_decay_per_season: float = 0.85

    xgb_params: dict = field(default_factory=lambda: {
        "objective": "multi:softprob",
        "num_class": N_CLASSES,
        "max_depth": 5,
        "learning_rate": 0.03,
        "n_estimators": 1500,
        "subsample": 0.75,
        "colsample_bytree": 0.6,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "min_child_weight": 5,
        "gamma": 0.1,
        "random_state": RANDOM_SEED,
        "tree_method": "hist",
    })

    lgb_params: dict = field(default_factory=lambda: {
        "objective": "multiclass",
        "num_class": N_CLASSES,
        "max_depth": 5,
        "learning_rate": 0.03,
        "n_estimators": 1500,
        "subsample": 0.75,
        "colsample_bytree": 0.6,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "min_child_samples": 20,
        "num_leaves": 31,
        "is_unbalance": True,
        "random_state": RANDOM_SEED,
        "verbose": -1,
    })

    cb_params: dict = field(default_factory=lambda: {
        "loss_function": "MultiClass",
        "classes_count": N_CLASSES,
        "depth": 6,
        "learning_rate": 0.03,
        "iterations": 1500,
        "l2_leaf_reg": 3.0,
        "bagging_temperature": 0.8,
        "random_strength": 1.0,
        "auto_class_weights": "Balanced",
        "random_seed": RANDOM_SEED,
        "verbose": 0,
    })


# ---------------------------------------------------------------------------
# Validation / cross-validation
# ---------------------------------------------------------------------------

@dataclass
class ValidationConfig:
    """Walk-forward cross-validation settings."""

    min_train_seasons: int = 5
    test_season_size: int = 1
    expanding_window: bool = True
    purge_matchweeks: int = 2  # Drop last N matchweeks from training (leakage prevention)

    # Minimum season to include in training data. Seasons before this are dropped.
    # "2017-2018" unlocks advanced features (xG, pressing, odds velocity, lineup)
    # that have near-zero coverage pre-2017. With time-decay=0.85, pre-2017 matches
    # contribute <4% weight anyway — dropping them loses almost nothing but unlocks
    # the best feature sources.
    min_train_season: str = "2017-2018"


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """Feature tier thresholds and selection parameters."""

    # A feature is "universal" if NaN fraction < this across all seasons
    # (applied AFTER smart imputation, so more features qualify).
    # Raised from 0.20 to 0.45: with min_train_season="2017-2018" and
    # time-decay weighting, pre-2017 NaN doesn't matter. This unlocks
    # xG, pressing, odds velocity, and lineup features (~90% coverage post-2017).
    universal_nan_threshold: float = 0.45

    # Correlation pruning: drop less-important feature when |r| > threshold
    correlation_threshold: float = 0.70

    # Importance-based selection: drop features with avg importance <= this
    importance_threshold: float = 0.0

    # Max features to keep after importance ranking (None = no cap)
    max_features: int = 60

    # Minimum samples required to train a rich (single-season) model
    min_rich_samples: int = 50

    # Train/val split ratio for rich model evaluation
    rich_val_ratio: float = 0.20

    # Recency weighting for feature selection importance averaging.
    # Exponential decay: weight = recency_base ^ (n_folds - 1 - fold_idx).
    # 1.0 = equal weighting (legacy), >1.0 = recent folds weighted higher.
    # 1.5 means the last fold is weighted ~1.5^15 ≈ 437× more than fold 0.
    recency_weight_base: float = 1.5

    # Number of recent-only folds for supplementary feature selection pass.
    # Features that rank in the top max_features in this recent-only pass
    # are added even if they failed the all-folds pass. 0 = disabled.
    recent_folds_for_supplement: int = 4


# ---------------------------------------------------------------------------
# Optuna hyperparameter search spaces
# ---------------------------------------------------------------------------

@dataclass
class TuningConfig:
    """Optuna tuning configuration."""

    n_trials: int = 80
    pruner_startup_trials: int = 10

    # Cap n_estimators during tuning for speed (early stopping still applies)
    tuning_n_estimators: int = 600

    # When coupling LR and n_estimators, base pair used as reference
    # n_estimators = base_n_est * (base_lr / suggested_lr)
    lr_estimator_base_lr: float = 0.03
    lr_estimator_base_n: int = 600
    lr_estimator_min_n: int = 300
    lr_estimator_max_n: int = 1200

    # XGBoost search ranges  (param_name -> (low, high, ...kwargs))
    xgb_search_space: Dict[str, Any] = field(default_factory=lambda: {
        "max_depth":        {"low": 3,    "high": 8},
        "learning_rate":    {"low": 0.01, "high": 0.15, "log": True},
        "subsample":        {"low": 0.5,  "high": 0.9},
        "colsample_bytree": {"low": 0.3,  "high": 0.8},
        "reg_alpha":        {"low": 0.01, "high": 10.0, "log": True},
        "reg_lambda":       {"low": 0.1,  "high": 10.0, "log": True},
        "min_child_weight": {"low": 3,    "high": 30},
        "gamma":            {"low": 0.0,  "high": 2.0},
    })

    # LightGBM search ranges
    lgb_search_space: Dict[str, Any] = field(default_factory=lambda: {
        "max_depth":        {"low": 3,    "high": 8},
        "learning_rate":    {"low": 0.01, "high": 0.15, "log": True},
        "subsample":        {"low": 0.5,  "high": 0.9},
        "colsample_bytree": {"low": 0.3,  "high": 0.8},
        "reg_alpha":        {"low": 0.01, "high": 10.0, "log": True},
        "reg_lambda":       {"low": 0.1,  "high": 10.0, "log": True},
        "min_child_samples": {"low": 10,  "high": 60},
        "num_leaves":       {"low": 15,   "high": 63},
    })

    # CatBoost search ranges
    cb_search_space: Dict[str, Any] = field(default_factory=lambda: {
        "depth":             {"low": 4,    "high": 8},
        "learning_rate":     {"low": 0.01, "high": 0.15, "log": True},
        "l2_leaf_reg":       {"low": 1.0,  "high": 10.0, "log": True},
        "bagging_temperature": {"low": 0.0, "high": 2.0},
        "random_strength":   {"low": 0.1,  "high": 3.0},
    })


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@dataclass
class CalibrationConfig:
    """Isotonic regression calibration settings."""

    # Clamp calibrated probabilities to [y_min, y_max]
    y_min: float = 0.01
    y_max: float = 0.99

    # Minimum row sum for renormalization (numerical stability)
    min_row_sum: float = 1e-8


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

@dataclass
class EnsembleConfig:
    """Stacking ensemble settings."""

    # Meta-learner regularization candidates (selected via inner CV)
    meta_C_candidates: List[float] = field(
        default_factory=lambda: [0.01, 0.1, 1.0, 10.0],
    )
    meta_max_iter: int = 1000
    meta_solver: str = "lbfgs"

    # Minimum training seasons required for ensemble evaluation
    min_ensemble_seasons: int = 2
