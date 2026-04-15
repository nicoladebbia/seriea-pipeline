#!/usr/bin/env python3
"""Unified Optimization Framework for Serie A Predictions.

Consolidates all optimization capabilities into a single entry point:
  - Ensemble weight optimization (from optimize_ensemble.py)
  - CatBoost/XGBoost hyperparameter tuning with Optuna (from optimize_models.py)
  - Fast stacking + weight blending (from optimize_fast.py)
  - All-markets optimization (from optimize_all_markets.py)
  - Walk-forward CV-based tuning (from ml/tuning.py)
  - Betting parameter optimization (Kelly fraction, edge thresholds)
  - Feature selection optimization

Usage:
    # Optimize ensemble weights
    python scripts/optimize_unified.py --target ensemble-weights

    # Tune 1X2 model hyperparameters with Optuna
    python scripts/optimize_unified.py --target model-hyperparams --trials 50

    # Optimize betting parameters (Kelly fraction, edge thresholds)
    python scripts/optimize_unified.py --target betting-params

    # Feature selection optimization
    python scripts/optimize_unified.py --target features

    # All-markets optimization (1X2 + binary markets)
    python scripts/optimize_unified.py --target all-markets

    # Quick mode (fewer trials, faster)
    python scripts/optimize_unified.py --target model-hyperparams --quick

    # Run all optimizations
    python scripts/optimize_unified.py --target all
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import accuracy_score, log_loss

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import DATA_DIR, MODELS_DIR
from ml.poisson import poisson_1x2, poisson_1x2_vec, market_implied_probs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

MARKET_MODELS_DIR = MODELS_DIR / "markets"
MARKET_MODELS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {"H": 0, "D": 1, "A": 2}
LABEL_NAMES = {0: "H", 1: "D", 2: "A"}


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class OptimizeConfig:
    """Unified optimization configuration."""

    # Targets: "ensemble-weights", "model-hyperparams", "betting-params",
    #          "features", "all-markets", "all"
    target: str = "ensemble-weights"

    # Optuna parameters
    n_trials: int = 50
    n_trials_binary: int = 30

    # Quick mode (fewer trials)
    quick: bool = False

    # Target metric for optimization: "log_loss", "accuracy", "rps", "brier"
    metric: str = "log_loss"

    # Seasons
    test_seasons: List[str] = field(default_factory=lambda: ["2023-2024", "2024-2025"])

    # Grid search step size for ensemble weights
    grid_step: float = 0.05

    # Walk-forward parameters
    min_train_seasons: int = 5  # Aligned with ml/config.py ValidationConfig

    # Sample size for quick evaluation (None = all)
    sample_size: Optional[int] = None

    # Output
    save_path: Optional[str] = None


# =============================================================================
# HELPER UTILITIES
# =============================================================================

def _elapsed(start_time: float) -> str:
    """Return formatted elapsed time since start."""
    secs = time.time() - start_time
    m, s = divmod(int(secs), 60)
    return f"{m:02d}:{s:02d}"


def _fmt_duration(secs: float) -> str:
    """Format seconds into MM:SS."""
    m, s = divmod(int(secs), 60)
    return f"{m}m{s:02d}s"


get_market_probs = market_implied_probs  # alias for backward compat


# =============================================================================
# UNIFIED OPTIMIZER
# =============================================================================

class UnifiedOptimizer:
    """Unified optimization engine combining all optimization modes."""

    def __init__(self, config: OptimizeConfig):
        self.config = config
        self.df: Optional[pd.DataFrame] = None
        self.features: Optional[List[str]] = None
        self.results: Dict[str, Any] = {}
        self._start_time = time.time()

    # -----------------------------------------------------------------
    # DATA LOADING
    # -----------------------------------------------------------------

    def load_data(self) -> bool:
        """Load unified training data with features."""
        log.info("[%s] Loading data...", _elapsed(self._start_time))

        features_path = DATA_DIR / "features" / "features.parquet"
        if not features_path.exists():
            log.error("Features not found: %s", features_path)
            return False

        self.df = pd.read_parquet(features_path)
        self.df = self.df.sort_values("match_date").reset_index(drop=True)
        self.df = self.df.dropna(subset=["result"])
        self.df = self.df[self.df["result"].isin(["H", "D", "A"])].copy()

        # Determine feature columns
        exclude_cols = {
            "match_id", "match_date", "season", "result", "home_score", "away_score",
            "home_team", "away_team", "ht_result", "referee",
            "home_formation", "away_formation",
            # Targets
            "total_goals", "over_2_5", "over_1_5", "over_3_5", "btts",
            "home_clean_sheet", "away_clean_sheet",
            "total_cards", "total_corners",
            "cards_over_3_5", "cards_over_4_5", "cards_over_5_5",
            "corners_over_8_5", "corners_over_9_5", "corners_over_10_5",
        }

        self.features = [
            c for c in self.df.columns
            if c not in exclude_cols
            and self.df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
            and not c.startswith("_")
        ]

        # Verify no leakage
        TARGETS = {
            "total_goals", "over_2_5", "btts", "total_cards", "total_corners",
            "home_score", "away_score",
        }
        leaked = set(self.features) & TARGETS
        if leaked:
            log.warning("POTENTIAL LEAKAGE detected in features: %s -- removing them", leaked)
            self.features = [f for f in self.features if f not in leaked]

        log.info(
            "[%s] Loaded %d matches, %d features, %d seasons",
            _elapsed(self._start_time), len(self.df), len(self.features),
            self.df["season"].nunique(),
        )

        return True

    def _load_comprehensive_data(self) -> Tuple[pd.DataFrame, List[str]]:
        """Try to load data via comprehensive_markets for richer feature/target set.

        Falls back to self.df/self.features if comprehensive_markets is not available.
        """
        try:
            from scripts.models.comprehensive_markets import (
                load_unified_training_data,
                get_training_features,
                _compute_targets,
            )
            df = load_unified_training_data()
            features = get_training_features(df)
            df = _compute_targets(df)
            log.info("Loaded comprehensive market data: %d matches, %d features", len(df), len(features))
            return df, features
        except (ImportError, Exception) as e:
            log.warning("comprehensive_markets not available (%s), using features.parquet", e)
            return self.df, self.features

    # -----------------------------------------------------------------
    # OPTIMIZE: ENSEMBLE WEIGHTS (from optimize_ensemble.py)
    # -----------------------------------------------------------------

    def optimize_ensemble_weights(self) -> Dict[str, Any]:
        """Optuna-based search over ensemble weights.

        Tests predefined weight configurations and optionally runs a grid search
        or Optuna-based search for optimal weights.
        """
        log.info("\n" + "=" * 70)
        log.info("ENSEMBLE WEIGHT OPTIMIZATION")
        log.info("=" * 70)

        # Predefined weight configs to test
        weight_configs = {
            "current":       {"factor": 0.30, "xg": 0.30, "ml": 0.15, "player_xg": 0.10, "deep": 0.15},
            "conservative":  {"factor": 0.40, "xg": 0.30, "ml": 0.15, "player_xg": 0.10, "deep": 0.05},
            "aggressive":    {"factor": 0.20, "xg": 0.30, "ml": 0.20, "player_xg": 0.15, "deep": 0.15},
            "balanced":      {"factor": 0.25, "xg": 0.25, "ml": 0.20, "player_xg": 0.15, "deep": 0.15},
            "ml_heavy":      {"factor": 0.20, "xg": 0.25, "ml": 0.25, "player_xg": 0.15, "deep": 0.15},
            "player_heavy":  {"factor": 0.20, "xg": 0.25, "ml": 0.15, "player_xg": 0.25, "deep": 0.15},
            "xg_dominant":   {"factor": 0.20, "xg": 0.40, "ml": 0.15, "player_xg": 0.10, "deep": 0.15},
            "factor_light":  {"factor": 0.15, "xg": 0.35, "ml": 0.20, "player_xg": 0.15, "deep": 0.15},
            "no_deep":       {"factor": 0.30, "xg": 0.35, "ml": 0.20, "player_xg": 0.15, "deep": 0.00},
            "deep_focus":    {"factor": 0.20, "xg": 0.25, "ml": 0.15, "player_xg": 0.10, "deep": 0.30},
        }

        # Evaluate each configuration
        test_df = self.df[self.df["season"].isin(self.config.test_seasons)].copy()
        if self.config.sample_size and len(test_df) > self.config.sample_size:
            test_df = test_df.tail(self.config.sample_size)

        log.info("Evaluating %d weight configurations on %d matches", len(weight_configs), len(test_df))

        config_results = []

        for config_name, weights in weight_configs.items():
            result = self._evaluate_ensemble_config(weights, test_df, config_name)
            config_results.append(result)
            log.info(
                "  %s: accuracy=%s, log_loss=%.4f",
                config_name, f"{result.get('accuracy', 0):.1%}", result.get("log_loss", 0),
            )

        # Sort by chosen metric
        if self.config.metric == "accuracy":
            config_results.sort(key=lambda x: x.get("accuracy", 0), reverse=True)
        else:
            config_results.sort(key=lambda x: x.get("log_loss", 999))

        # Optuna-based optimization if trials > 0
        optuna_result = None
        if self.config.n_trials > 0:
            optuna_result = self._optuna_ensemble_weights(test_df)

        # Recommendations
        best = config_results[0]
        current = next((r for r in config_results if r["config_name"] == "current"), None)

        result = {
            "predefined_configs": config_results,
            "best_config": best["config_name"],
            "best_weights": best["weights"],
            "best_accuracy": best.get("accuracy", 0),
            "best_log_loss": best.get("log_loss", 0),
            "current_accuracy": current.get("accuracy", 0) if current else None,
            "improvement_vs_current": (
                best.get("accuracy", 0) - current.get("accuracy", 0)
                if current else None
            ),
        }

        if optuna_result:
            result["optuna_best"] = optuna_result

        return result

    def _evaluate_ensemble_config(
        self,
        weights: Dict[str, float],
        test_df: pd.DataFrame,
        config_name: str = "custom",
    ) -> Dict[str, Any]:
        """Evaluate a weight configuration on test data.

        Uses simplified factor/xG/market ensemble evaluation since the
        full component predictors may not be available.
        """
        correct = 0
        high_conf_correct = 0
        high_conf_total = 0
        total = 0
        log_loss_sum = 0.0

        # Map the weight config: only factor/xg/market are available for pure eval
        # ml, player_xg, deep are evaluated as factor fallback if not available
        available_w = {
            "factor": weights.get("factor", 0) + weights.get("ml", 0) + weights.get("player_xg", 0) + weights.get("deep", 0),
            "xg": weights.get("xg", 0),
            "market": weights.get("market", 0) if "market" in weights else 0,
        }
        # Normalize
        total_w = sum(available_w.values())
        if total_w > 0:
            available_w = {k: v / total_w for k, v in available_w.items()}

        from scripts.analysis.backtest_unified import predict_ensemble

        for _, row in test_df.iterrows():
            actual = row["result"]
            actual_idx = LABEL_MAP.get(actual)
            if actual_idx is None:
                continue

            pred = predict_ensemble(row, weights=available_w)

            probs = [pred["prob_H"], pred["prob_D"], pred["prob_A"]]
            pred_idx = np.argmax(probs)
            max_prob = max(probs)

            if pred_idx == actual_idx:
                correct += 1

            if max_prob >= 0.50:
                high_conf_total += 1
                if pred_idx == actual_idx:
                    high_conf_correct += 1

            eps = 1e-10
            log_loss_sum += -np.log(probs[actual_idx] + eps)
            total += 1

        if total == 0:
            return {"config_name": config_name, "weights": weights, "error": "no_predictions"}

        return {
            "config_name": config_name,
            "weights": weights,
            "total_matches": total,
            "accuracy": correct / total,
            "high_conf_accuracy": high_conf_correct / high_conf_total if high_conf_total > 0 else 0,
            "high_conf_count": high_conf_total,
            "log_loss": log_loss_sum / total,
        }

    def _optuna_ensemble_weights(self, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Optuna search for optimal ensemble weights."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            log.warning("Optuna not installed, skipping ensemble weight optimization")
            return {}

        from scripts.analysis.backtest_unified import predict_ensemble

        def objective(trial):
            w_factor = trial.suggest_float("factor", 0.05, 0.50)
            w_xg = trial.suggest_float("xg", 0.05, 0.50)
            w_market = trial.suggest_float("market", 0.0, 0.40)
            # Normalize
            total_w = w_factor + w_xg + w_market
            weights = {
                "factor": w_factor / total_w,
                "xg": w_xg / total_w,
                "market": w_market / total_w,
            }

            correct = 0
            total = 0
            ll_sum = 0.0

            for _, row in test_df.iterrows():
                actual = row["result"]
                actual_idx = LABEL_MAP.get(actual)
                if actual_idx is None:
                    continue

                pred = predict_ensemble(row, weights=weights)
                probs = [pred["prob_H"], pred["prob_D"], pred["prob_A"]]
                pred_idx = np.argmax(probs)

                if pred_idx == actual_idx:
                    correct += 1

                ll_sum += -np.log(max(1e-10, probs[actual_idx]))
                total += 1

            if total == 0:
                return 999.0

            if self.config.metric == "accuracy":
                return -(correct / total)  # Minimize negative accuracy
            else:
                return ll_sum / total  # Minimize log loss

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=self.config.n_trials, show_progress_bar=True)

        best = study.best_params
        total_w = best["factor"] + best["xg"] + best["market"]
        best_weights = {
            "factor": round(best["factor"] / total_w, 3),
            "xg": round(best["xg"] / total_w, 3),
            "market": round(best["market"] / total_w, 3),
        }

        log.info("Optuna best ensemble weights: %s (score=%.4f)", best_weights, study.best_value)

        return {
            "weights": best_weights,
            "score": round(study.best_value, 4),
            "n_trials": len(study.trials),
        }

    # -----------------------------------------------------------------
    # OPTIMIZE: MODEL HYPERPARAMETERS (from optimize_models.py + ml/tuning.py)
    # -----------------------------------------------------------------

    def optimize_model_hyperparams(self) -> Dict[str, Any]:
        """Optuna hyperparameter tuning for CatBoost/XGBoost 1X2 + binary markets.

        Combines the Optuna tuning from optimize_models.py with the walk-forward
        validation from ml/tuning.py.
        """
        log.info("\n" + "=" * 70)
        log.info("MODEL HYPERPARAMETER OPTIMIZATION")
        log.info("=" * 70)

        df, features = self._load_comprehensive_data()
        df_valid = df[df["result"].isin(["H", "D", "A"])].copy()
        seasons = sorted(df_valid["season"].unique())

        results = {}

        # --- 1. CatBoost 1X2 Tuning ---
        log.info("\n--- Step 1: CatBoost 1X2 Tuning (%d trials) ---", self.config.n_trials)
        best_1x2_params = self._tune_catboost_1x2(df_valid, features, seasons)
        results["1x2_best_params"] = best_1x2_params

        # --- 2. XGBoost 1X2 Tuning (via ml/tuning.py) ---
        log.info("\n--- Step 2: Walk-Forward Tuning (CatBoost + XGBoost + LightGBM) ---")
        wf_results = self._tune_via_walk_forward(df_valid, features)
        results["walk_forward_tuning"] = wf_results

        # --- 3. Train optimized stacking ensemble ---
        log.info("\n--- Step 3: Stacking Ensemble Evaluation ---")
        for test_season in self.config.test_seasons:
            if test_season in seasons:
                stack_result = self._train_evaluate_stacking(
                    df_valid, features, best_1x2_params, test_season, seasons,
                )
                results[f"stacking_{test_season}"] = stack_result

        # --- 4. Binary market tuning ---
        log.info("\n--- Step 4: Binary Market Tuning ---")
        binary_targets = self._get_binary_targets(df)
        for target, name in binary_targets.items():
            if target not in df.columns or df[target].isna().all():
                continue
            log.info("Tuning %s...", name)
            bp = self._tune_catboost_binary(df_valid, features, target, seasons)
            if bp:
                results[f"{target}_best_params"] = bp

        return results

    def _tune_catboost_1x2(
        self, df_valid: pd.DataFrame, features: List[str], seasons: List[str],
    ) -> Dict[str, Any]:
        """Tune CatBoost hyperparameters for 1X2 prediction using Optuna."""
        try:
            import optuna
            from catboost import CatBoostClassifier
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError as e:
            log.warning("Required packages not available: %s", e)
            return {}

        X = df_valid[features].fillna(0)
        y = df_valid["result"].map(LABEL_MAP)

        test_seasons = [s for s in self.config.test_seasons if s in seasons]
        if not test_seasons:
            test_seasons = seasons[-3:]

        log.info("Tuning 1X2 CatBoost on test seasons: %s", test_seasons)

        def objective(trial):
            params = {
                "iterations": trial.suggest_int("iterations", 500, 3000, step=100),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 30, log=True),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
                "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
                "border_count": trial.suggest_int("border_count", 32, 255),
            }

            lls = []
            for ts in test_seasons:
                ts_idx = seasons.index(ts)
                if ts_idx < 2:
                    continue
                vs = seasons[ts_idx - 1]
                tr_seasons = seasons[:ts_idx - 1]

                train_mask = df_valid["season"].isin(tr_seasons)
                val_mask = df_valid["season"] == vs
                test_mask = df_valid["season"] == ts

                if train_mask.sum() < 500 or test_mask.sum() < 50:
                    continue

                model = CatBoostClassifier(
                    **params, random_seed=42, verbose=0,
                    loss_function="MultiClass", classes_count=3,
                    auto_class_weights="Balanced",
                    early_stopping_rounds=100,
                )
                model.fit(
                    X[train_mask], y[train_mask],
                    eval_set=(X[val_mask], y[val_mask]),
                    verbose=0,
                )

                probs = model.predict_proba(X[test_mask])
                true = y[test_mask].values

                try:
                    lls.append(log_loss(true, probs, labels=[0, 1, 2]))
                except Exception:
                    lls.append(1.2)

                del model
                gc.collect()

            return np.mean(lls) if lls else 1.2

        study = optuna.create_study(direction="minimize", study_name="1x2_tuning")
        study.optimize(objective, n_trials=self.config.n_trials, show_progress_bar=True)

        best = study.best_params
        log.info("Best 1X2 params (LL=%.4f): %s", study.best_value, best)

        return best

    def _tune_catboost_binary(
        self, df_valid: pd.DataFrame, features: List[str],
        target_col: str, seasons: List[str],
    ) -> Dict[str, Any]:
        """Tune CatBoost for a binary classification target."""
        try:
            import optuna
            from catboost import CatBoostClassifier
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            return {}

        valid = df_valid[target_col].notna() if target_col in df_valid.columns else pd.Series(False, index=df_valid.index)
        if valid.sum() < 200:
            return {}

        df_v = df_valid[valid].copy()
        X = df_v[features].fillna(0)
        y = df_v[target_col].astype(int)

        test_seasons = [
            s for s in self.config.test_seasons
            if s in df_v["season"].values
            and df_v[df_v["season"] == s][target_col].notna().sum() > 20
        ]

        if not test_seasons:
            return {}

        v_seasons = sorted(df_v["season"].unique())

        def objective(trial):
            params = {
                "iterations": trial.suggest_int("iterations", 300, 2000, step=100),
                "depth": trial.suggest_int("depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 20, log=True),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 30),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.5),
            }

            lls = []
            for ts in test_seasons:
                ts_idx = v_seasons.index(ts)
                if ts_idx < 2:
                    continue
                vs = v_seasons[ts_idx - 1]
                tr = v_seasons[:ts_idx - 1]

                train_mask = df_v["season"].isin(tr)
                val_mask = df_v["season"] == vs
                test_mask = df_v["season"] == ts

                if train_mask.sum() < 200 or test_mask.sum() < 20:
                    continue

                model = CatBoostClassifier(
                    **params, random_seed=42, verbose=0,
                    loss_function="Logloss",
                    early_stopping_rounds=80,
                )
                model.fit(
                    X[train_mask], y[train_mask],
                    eval_set=(X[val_mask], y[val_mask]),
                    verbose=0,
                )

                probs = model.predict_proba(X[test_mask])[:, 1]
                try:
                    lls.append(log_loss(y[test_mask].values, probs))
                except Exception:
                    lls.append(0.7)

                del model
                gc.collect()

            return np.mean(lls) if lls else 0.7

        study = optuna.create_study(
            direction="minimize",
            study_name=f"{target_col}_tuning",
        )
        study.optimize(
            objective,
            n_trials=self.config.n_trials_binary,
            show_progress_bar=True,
        )

        log.info(
            "Best %s params (LL=%.4f): %s",
            target_col, study.best_value, study.best_params,
        )
        return study.best_params

    def _tune_via_walk_forward(
        self, df_valid: pd.DataFrame, features: List[str],
    ) -> Dict[str, Any]:
        """Use the ml/tuning.py module for walk-forward-based tuning."""
        try:
            from ml.tuning import tune_model
            from ml.config import META_COLS, SEASON_COL

            # Prepare data with _season meta column
            X = df_valid[features].copy()
            X["_season"] = df_valid["season"].values
            y = df_valid["result"]

            n_trials = 10 if self.config.quick else min(self.config.n_trials, 40)
            results = {}

            for model_type in ["catboost", "xgboost"]:
                log.info("Walk-forward tuning: %s (%d trials)...", model_type, n_trials)
                try:
                    tune_result = tune_model(
                        X, y,
                        model_type=model_type,
                        n_trials=n_trials,
                    )
                    results[model_type] = {
                        "best_params": {
                            k: v for k, v in tune_result["best_params"].items()
                            if not callable(v)
                        },
                        "best_score": tune_result["best_score"],
                    }
                    log.info("  %s best LL: %.4f", model_type, tune_result["best_score"])
                except Exception as e:
                    log.warning("  %s tuning failed: %s", model_type, e)
                    results[model_type] = {"error": str(e)}

            return results

        except ImportError as e:
            log.warning("ml/tuning.py not available: %s", e)
            return {"error": str(e)}

    def _train_evaluate_stacking(
        self, df_valid: pd.DataFrame, features: List[str],
        catboost_params: Dict[str, Any], test_season: str, seasons: List[str],
    ) -> Dict[str, Any]:
        """Train stacking ensemble (CatBoost + Poisson + Market) for 1X2.

        From optimize_fast.py: trains multiple models and optimizes blend weights
        on validation set.
        """
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ImportError:
            return {"error": "CatBoost not installed"}

        if test_season not in seasons:
            return {"error": f"{test_season} not in available seasons"}

        test_idx = seasons.index(test_season)
        if test_idx < 2:
            return {"error": "Not enough prior seasons for training"}

        val_season = seasons[test_idx - 1]
        train_seasons = seasons[:test_idx - 1]

        train_m = df_valid["season"].isin(train_seasons)
        val_m = df_valid["season"] == val_season
        test_m = df_valid["season"] == test_season

        X_tr = df_valid.loc[train_m, features].fillna(0)
        X_val = df_valid.loc[val_m, features].fillna(0)
        X_te = df_valid.loc[test_m, features].fillna(0)

        y_tr = df_valid.loc[train_m, "result"].map(LABEL_MAP)
        y_val = df_valid.loc[val_m, "result"].map(LABEL_MAP)
        y_te = df_valid.loc[test_m, "result"].map(LABEL_MAP)
        y_te_labels = df_valid.loc[test_m, "result"].values

        log.info(
            "Stacking for %s: Train=%d, Val=%d, Test=%d",
            test_season, len(X_tr), len(X_val), len(X_te),
        )

        # Model 1: CatBoost Classifier
        cb_params = {k: v for k, v in catboost_params.items() if k not in ("grow_policy", "border_count")}
        cb_params.setdefault("iterations", 2000)
        cb_params.setdefault("depth", 6)
        cb_params.setdefault("learning_rate", 0.02)

        m_cls = CatBoostClassifier(
            **cb_params, random_seed=42, verbose=0,
            loss_function="MultiClass", classes_count=3,
            auto_class_weights="Balanced",
        )
        m_cls.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=150, verbose=0)
        cls_probs_te = m_cls.predict_proba(X_te)
        cls_probs_val = m_cls.predict_proba(X_val)
        del m_cls
        gc.collect()

        # Model 2: Poisson goal regressors
        if "home_score" in df_valid.columns and "away_score" in df_valid.columns:
            m_hg = CatBoostRegressor(
                iterations=1500, depth=6, learning_rate=0.03,
                l2_leaf_reg=5, random_seed=42, verbose=0, loss_function="Poisson",
            )
            m_ag = CatBoostRegressor(
                iterations=1500, depth=6, learning_rate=0.03,
                l2_leaf_reg=5, random_seed=42, verbose=0, loss_function="Poisson",
            )
            m_hg.fit(
                X_tr, df_valid.loc[train_m, "home_score"],
                eval_set=(X_val, df_valid.loc[val_m, "home_score"]),
                early_stopping_rounds=100, verbose=0,
            )
            m_ag.fit(
                X_tr, df_valid.loc[train_m, "away_score"],
                eval_set=(X_val, df_valid.loc[val_m, "away_score"]),
                early_stopping_rounds=100, verbose=0,
            )

            hxg_te = np.clip(m_hg.predict(X_te), 0.3, 4.0)
            axg_te = np.clip(m_ag.predict(X_te), 0.3, 4.0)
            pois_probs_te = poisson_1x2_vec(hxg_te, axg_te)

            hxg_val = np.clip(m_hg.predict(X_val), 0.3, 4.0)
            axg_val = np.clip(m_ag.predict(X_val), 0.3, 4.0)
            pois_probs_val = poisson_1x2_vec(hxg_val, axg_val)

            del m_hg, m_ag
            gc.collect()
        else:
            pois_probs_te = np.tile([0.4, 0.3, 0.3], (len(X_te), 1))
            pois_probs_val = np.tile([0.4, 0.3, 0.3], (len(X_val), 1))

        # Model 3: Market odds
        mkt_probs_te = get_market_probs(df_valid, test_m)
        mkt_probs_val = get_market_probs(df_valid, val_m)

        # Optimize blend weights on validation
        log.info("Optimizing stacking weights on validation...")
        best_weights = (0.33, 0.33, 0.34)
        best_ll = 999

        for w1 in np.arange(0.0, 0.65, 0.05):
            for w2 in np.arange(0.0, 0.65, 0.05):
                w3 = 1.0 - w1 - w2
                if w3 < 0 or w3 > 0.6:
                    continue
                ens = w1 * cls_probs_val + w2 * pois_probs_val + w3 * mkt_probs_val
                ens = ens / ens.sum(axis=1, keepdims=True)
                try:
                    ll = log_loss(y_val.values, ens, labels=[0, 1, 2])
                    if ll < best_ll:
                        best_ll = ll
                        best_weights = (w1, w2, w3)
                except Exception as e:
                    log.debug(f"Failed to compute log_loss for weight combo: {e}")

        log.info(
            "Best stacking weights: CatBoost=%.2f, Poisson=%.2f, Market=%.2f (val LL=%.4f)",
            *best_weights, best_ll,
        )

        # Apply to test
        w1, w2, w3 = best_weights
        ens_probs = w1 * cls_probs_te + w2 * pois_probs_te + w3 * mkt_probs_te
        ens_probs = ens_probs / ens_probs.sum(axis=1, keepdims=True)

        # Evaluate
        results = {}
        for name, probs in [
            ("catboost", cls_probs_te),
            ("poisson", pois_probs_te),
            ("market_odds", mkt_probs_te),
            ("ensemble", ens_probs),
        ]:
            pred = np.array(["H", "D", "A"])[probs.argmax(axis=1)]
            acc = (pred == y_te_labels).mean()
            ll = log_loss(y_te.values, probs, labels=[0, 1, 2])

            hc_mask = probs.max(axis=1) > 0.55
            hc_acc = (pred[hc_mask] == y_te_labels[hc_mask]).mean() if hc_mask.sum() > 0 else 0

            results[name] = {
                "accuracy": round(acc, 4),
                "log_loss": round(ll, 4),
                "hc_accuracy": round(hc_acc, 4),
                "hc_count": int(hc_mask.sum()),
            }

            log.info(
                "  %s: acc=%.1f%%, LL=%.4f, HC=%.1f%% (n=%d)",
                name, acc * 100, ll, hc_acc * 100, hc_mask.sum(),
            )

        results["best_weights"] = {
            "catboost": round(w1, 3),
            "poisson": round(w2, 3),
            "market": round(w3, 3),
        }

        return results

    # -----------------------------------------------------------------
    # OPTIMIZE: BETTING PARAMETERS
    # -----------------------------------------------------------------

    def optimize_betting_params(self) -> Dict[str, Any]:
        """Optimize Kelly fraction, edge thresholds, and confidence cutoff.

        Uses grid search over betting parameters to maximize ROI.
        """
        log.info("\n" + "=" * 70)
        log.info("BETTING PARAMETER OPTIMIZATION")
        log.info("=" * 70)

        from scripts.analysis.backtest_unified import BacktestConfig, BacktestEngine

        # Parameter grid
        kelly_fractions = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
        min_edges = [0.01, 0.02, 0.03, 0.04, 0.05]
        max_edges = [0.04, 0.06, 0.08, 0.10, 0.15]
        confidence_thresholds = [0.45, 0.50, 0.55, 0.60]

        if self.config.quick:
            kelly_fractions = [0.15, 0.25, 0.35]
            min_edges = [0.02, 0.03, 0.05]
            max_edges = [0.06, 0.10]
            confidence_thresholds = [0.50, 0.55]

        best_roi = -999
        best_params = {}
        all_results = []

        total_combos = len(kelly_fractions) * len(min_edges) * len(max_edges) * len(confidence_thresholds)
        log.info("Testing %d parameter combinations...", total_combos)

        combo_count = 0
        for kelly in kelly_fractions:
            for min_e in min_edges:
                for max_e in max_edges:
                    if max_e <= min_e:
                        continue
                    for conf in confidence_thresholds:
                        combo_count += 1
                        if combo_count % 20 == 0:
                            log.info(
                                "  Progress: %d/%d (best ROI: %+.1f%%)",
                                combo_count, total_combos, best_roi,
                            )

                        bt_config = BacktestConfig(
                            mode="betting",
                            seasons=self.config.test_seasons,
                            kelly_fraction=kelly,
                            min_edge=min_e,
                            max_edge=max_e,
                            confidence_threshold=conf,
                        )

                        engine = BacktestEngine(bt_config)
                        engine.df = self.df.copy()

                        # Run simplified betting sim for each season
                        total_pnl = 0
                        total_staked = 0
                        total_bets = 0

                        for season in self.config.test_seasons:
                            result = engine.backtest_betting(season)
                            if "error" not in result:
                                total_pnl += result["total_pnl"]
                                total_staked += result["total_staked"]
                                total_bets += result["total_bets"]

                        roi = (total_pnl / max(total_staked, 1)) * 100 if total_staked > 0 else 0

                        combo_result = {
                            "kelly": kelly,
                            "min_edge": min_e,
                            "max_edge": max_e,
                            "confidence": conf,
                            "total_bets": total_bets,
                            "total_staked": round(total_staked, 2),
                            "total_pnl": round(total_pnl, 2),
                            "roi": round(roi, 2),
                        }
                        all_results.append(combo_result)

                        if roi > best_roi and total_bets >= 10:
                            best_roi = roi
                            best_params = combo_result

        # Sort by ROI
        all_results.sort(key=lambda x: x["roi"], reverse=True)

        log.info("\nBest betting parameters:")
        log.info("  Kelly fraction: %.2f", best_params.get("kelly", 0))
        log.info("  Min edge: %.3f", best_params.get("min_edge", 0))
        log.info("  Max edge: %.3f", best_params.get("max_edge", 0))
        log.info("  Confidence threshold: %.2f", best_params.get("confidence", 0))
        log.info("  ROI: %+.1f%%", best_params.get("roi", 0))
        log.info("  Total bets: %d", best_params.get("total_bets", 0))

        return {
            "best_params": best_params,
            "top_10": all_results[:10],
            "total_combinations_tested": len(all_results),
        }

    # -----------------------------------------------------------------
    # OPTIMIZE: FEATURE SELECTION
    # -----------------------------------------------------------------

    def optimize_features(self) -> Dict[str, Any]:
        """Feature selection optimization using importance-based pruning.

        Trains a model, ranks features by importance, and evaluates subsets
        to find the optimal feature count.
        """
        log.info("\n" + "=" * 70)
        log.info("FEATURE SELECTION OPTIMIZATION")
        log.info("=" * 70)

        try:
            from catboost import CatBoostClassifier
        except ImportError:
            return {"error": "CatBoost not installed"}

        df_valid = self.df[self.df["result"].isin(["H", "D", "A"])].copy()
        seasons = sorted(df_valid["season"].unique())

        test_season = self.config.test_seasons[-1] if self.config.test_seasons else seasons[-1]
        if test_season not in seasons:
            test_season = seasons[-1]

        test_idx = seasons.index(test_season)
        val_season = seasons[test_idx - 1]
        train_seasons = seasons[:test_idx - 1]

        train_m = df_valid["season"].isin(train_seasons)
        val_m = df_valid["season"] == val_season
        test_m = df_valid["season"] == test_season

        X_tr = df_valid.loc[train_m, self.features].fillna(0)
        X_val = df_valid.loc[val_m, self.features].fillna(0)
        X_te = df_valid.loc[test_m, self.features].fillna(0)
        y_tr = df_valid.loc[train_m, "result"].map(LABEL_MAP)
        y_val = df_valid.loc[val_m, "result"].map(LABEL_MAP)
        y_te = df_valid.loc[test_m, "result"].map(LABEL_MAP)
        y_te_labels = df_valid.loc[test_m, "result"].values

        # Train full model to get feature importances
        log.info("Training full model with %d features...", len(self.features))
        model = CatBoostClassifier(
            iterations=1500, depth=6, learning_rate=0.03,
            l2_leaf_reg=3.0, random_seed=42, verbose=0,
            loss_function="MultiClass", classes_count=3,
            auto_class_weights="Balanced",
        )
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=100, verbose=0)

        importances = model.get_feature_importance()
        feat_imp = sorted(zip(self.features, importances), key=lambda x: x[1], reverse=True)

        # Full model baseline
        full_probs = model.predict_proba(X_te)
        full_pred = full_probs.argmax(axis=1)
        full_acc = (full_pred == y_te.values).mean()
        full_ll = log_loss(y_te.values, full_probs, labels=[0, 1, 2])
        log.info("Full model (%d features): acc=%.1f%%, LL=%.4f", len(self.features), full_acc * 100, full_ll)

        del model
        gc.collect()

        # Test feature subsets
        feature_counts = [10, 20, 30, 40, 50, 60, 80, 100, len(self.features)]
        feature_counts = sorted(set(f for f in feature_counts if f <= len(self.features)))

        subset_results = []

        for n_features in feature_counts:
            selected = [name for name, _ in feat_imp[:n_features]]

            model = CatBoostClassifier(
                iterations=1500, depth=6, learning_rate=0.03,
                l2_leaf_reg=3.0, random_seed=42, verbose=0,
                loss_function="MultiClass", classes_count=3,
                auto_class_weights="Balanced",
            )
            model.fit(
                X_tr[selected], y_tr,
                eval_set=(X_val[selected], y_val),
                early_stopping_rounds=100, verbose=0,
            )

            probs = model.predict_proba(X_te[selected])
            pred = probs.argmax(axis=1)
            acc = (pred == y_te.values).mean()
            ll = log_loss(y_te.values, probs, labels=[0, 1, 2])

            subset_results.append({
                "n_features": n_features,
                "accuracy": round(acc, 4),
                "log_loss": round(ll, 4),
                "acc_vs_full": round(acc - full_acc, 4),
                "ll_vs_full": round(ll - full_ll, 4),
            })

            log.info(
                "  %d features: acc=%.1f%% (%+.1f), LL=%.4f (%+.4f)",
                n_features, acc * 100, (acc - full_acc) * 100, ll, ll - full_ll,
            )

            del model
            gc.collect()

        # Find optimal count (best log loss)
        best_subset = min(subset_results, key=lambda x: x["log_loss"])
        optimal_n = best_subset["n_features"]
        optimal_features = [name for name, _ in feat_imp[:optimal_n]]

        log.info(
            "\nOptimal feature count: %d (LL=%.4f, acc=%.1f%%)",
            optimal_n, best_subset["log_loss"], best_subset["accuracy"] * 100,
        )

        return {
            "top_features": [
                {"name": name, "importance": round(imp, 2)}
                for name, imp in feat_imp[:50]
            ],
            "subset_results": subset_results,
            "optimal_n_features": optimal_n,
            "optimal_features": optimal_features[:50],  # Save first 50
            "full_model": {
                "n_features": len(self.features),
                "accuracy": round(full_acc, 4),
                "log_loss": round(full_ll, 4),
            },
        }

    # -----------------------------------------------------------------
    # OPTIMIZE: ALL MARKETS (from optimize_all_markets.py)
    # -----------------------------------------------------------------

    def optimize_all_markets(self) -> Dict[str, Any]:
        """Optimize all betting markets: 1X2 + O/U + BTTS + Cards + Corners + HT.

        Walk-forward evaluation with multiple hyperparameter configs and
        threshold optimization.
        """
        log.info("\n" + "=" * 70)
        log.info("ALL-MARKETS OPTIMIZATION")
        log.info("=" * 70)

        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ImportError:
            return {"error": "CatBoost not installed"}

        df, features = self._load_comprehensive_data()
        df_valid = df[df["result"].isin(["H", "D", "A"])].copy()
        seasons = sorted(df_valid["season"].unique())

        binary_configs = [
            {"name": "base", "depth": 6, "lr": 0.03, "l2": 5, "ml": 25, "cw": None},
            {"name": "bal", "depth": 6, "lr": 0.03, "l2": 5, "ml": 25, "cw": "Balanced"},
            {"name": "bal_d7", "depth": 7, "lr": 0.02, "l2": 3, "ml": 20, "cw": "Balanced"},
        ]

        all_season_results = {}

        for test_season in self.config.test_seasons:
            if test_season not in seasons:
                continue

            season_start = time.time()
            log.info("\n--- Season: %s ---", test_season)

            ti = seasons.index(test_season)
            if ti < 2:
                continue

            val_season = seasons[ti - 1]
            train_seasons_list = seasons[:ti - 1]

            train_m = df_valid["season"].isin(train_seasons_list)
            val_m = df_valid["season"] == val_season
            test_m = df_valid["season"] == test_season

            X_tr = df_valid.loc[train_m, features].fillna(0)
            X_val = df_valid.loc[val_m, features].fillna(0)
            X_te = df_valid.loc[test_m, features].fillna(0)
            y_te = df_valid.loc[test_m, "result"].values

            log.info("Train=%d, Val=%d, Test=%d", train_m.sum(), val_m.sum(), test_m.sum())

            results = {}

            # ----- 1X2 Ensemble -----
            y_tr_num = df_valid.loc[train_m, "result"].map(LABEL_MAP).values
            y_val_num = df_valid.loc[val_m, "result"].map(LABEL_MAP).values

            cb = CatBoostClassifier(
                iterations=2000, depth=8, learning_rate=0.01, l2_leaf_reg=5,
                min_data_in_leaf=30, random_seed=42, verbose=0,
                loss_function="MultiClass", classes_count=3,
                auto_class_weights="Balanced",
            )
            cb.fit(X_tr, y_tr_num, eval_set=(X_val, y_val_num), early_stopping_rounds=150, verbose=0)
            cb_probs = cb.predict_proba(X_te)
            del cb
            gc.collect()

            # Poisson
            hg_model = CatBoostRegressor(
                iterations=1500, depth=8, learning_rate=0.015, l2_leaf_reg=7,
                min_data_in_leaf=30, random_seed=42, verbose=0, loss_function="Poisson",
            )
            ag_model = CatBoostRegressor(
                iterations=1500, depth=8, learning_rate=0.015, l2_leaf_reg=7,
                min_data_in_leaf=30, random_seed=42, verbose=0, loss_function="Poisson",
            )

            if "home_score" in df_valid.columns:
                hg_model.fit(
                    X_tr, df_valid.loc[train_m, "home_score"],
                    eval_set=(X_val, df_valid.loc[val_m, "home_score"]),
                    early_stopping_rounds=100, verbose=0,
                )
                ag_model.fit(
                    X_tr, df_valid.loc[train_m, "away_score"],
                    eval_set=(X_val, df_valid.loc[val_m, "away_score"]),
                    early_stopping_rounds=100, verbose=0,
                )

                hxg = np.clip(hg_model.predict(X_te), 0.3, 4.0)
                axg = np.clip(ag_model.predict(X_te), 0.3, 4.0)
                pois_probs = poisson_1x2_vec(hxg, axg)

                # Also get O/U and BTTS from Poisson
                n_te = test_m.sum()
                pois_ou25 = np.zeros(n_te)
                pois_btts = np.zeros(n_te)
                pois_ou15 = np.zeros(n_te)
                pois_ou35 = np.zeros(n_te)
                for i in range(n_te):
                    hx, ax = max(hxg[i], 0.3), max(axg[i], 0.3)
                    for h in range(8):
                        for a in range(8):
                            p = poisson.pmf(h, hx) * poisson.pmf(a, ax)
                            if h + a > 2.5:
                                pois_ou25[i] += p
                            if h + a > 1.5:
                                pois_ou15[i] += p
                            if h + a > 3.5:
                                pois_ou35[i] += p
                            if h >= 1 and a >= 1:
                                pois_btts[i] += p
            else:
                pois_probs = np.tile([0.4, 0.3, 0.3], (test_m.sum(), 1))
                pois_ou25 = np.full(test_m.sum(), 0.5)
                pois_btts = np.full(test_m.sum(), 0.5)
                pois_ou15 = np.full(test_m.sum(), 0.7)
                pois_ou35 = np.full(test_m.sum(), 0.3)

            del hg_model, ag_model
            gc.collect()

            # Market odds
            mkt = get_market_probs(df_valid, test_m)

            # 1X2 ensemble (with draw boost from optimize_all_markets.py)
            ens = 0.25 * cb_probs + 0.40 * pois_probs + 0.35 * mkt
            ens[:, 1] *= 1.35  # Draw boost
            ens = ens / ens.sum(axis=1, keepdims=True)

            preds_1x2 = np.array(["H", "D", "A"])[ens.argmax(1)]
            acc_1x2 = (preds_1x2 == y_te).mean()
            mkt_preds = np.array(["H", "D", "A"])[mkt.argmax(1)]
            mkt_acc = (mkt_preds == y_te).mean()

            results["1x2"] = {
                "accuracy": round(acc_1x2, 4),
                "market_accuracy": round(mkt_acc, 4),
                "draws_predicted": int((preds_1x2 == "D").sum()),
                "draws_actual": int((y_te == "D").sum()),
            }
            log.info("  1X2: acc=%.1f%% (market=%.1f%%)", acc_1x2 * 100, mkt_acc * 100)

            # ----- Binary markets -----
            binary_targets = self._get_binary_targets(df_valid)

            for target, name in binary_targets.items():
                if target not in df_valid.columns or df_valid[target].isna().all():
                    continue

                y_tr_bin = df_valid.loc[train_m, target]
                y_val_bin = df_valid.loc[val_m, target]
                y_te_bin = df_valid.loc[test_m, target]

                if y_tr_bin.isna().all() or y_te_bin.isna().all():
                    continue

                # Train with best binary config
                best_acc_bin = 0
                best_probs_bin = None

                for cfg in binary_configs:
                    try:
                        m = CatBoostClassifier(
                            iterations=cfg.get("iter", 1200),
                            depth=cfg["depth"],
                            learning_rate=cfg["lr"],
                            l2_leaf_reg=cfg["l2"],
                            min_data_in_leaf=cfg["ml"],
                            random_seed=42, verbose=0,
                            loss_function="Logloss",
                            auto_class_weights=cfg.get("cw"),
                        )
                        m.fit(
                            X_tr, y_tr_bin.fillna(0).astype(int),
                            eval_set=(X_val, y_val_bin.fillna(0).astype(int)),
                            early_stopping_rounds=80, verbose=0,
                        )
                        probs = m.predict_proba(X_te)[:, 1]

                        # Test thresholds
                        for thr in [0.45, 0.48, 0.50, 0.52, 0.55]:
                            preds = (probs > thr).astype(int)
                            acc = (preds == y_te_bin.fillna(0).astype(int).values).mean()
                            if acc > best_acc_bin:
                                best_acc_bin = acc
                                best_probs_bin = probs

                        del m
                        gc.collect()
                    except Exception as e:
                        log.debug("  %s config %s failed: %s", name, cfg["name"], e)

                # Blend with Poisson features if available
                poisson_feat = {
                    "over_2_5": pois_ou25,
                    "over_1_5": pois_ou15,
                    "over_3_5": pois_ou35,
                    "btts": pois_btts,
                }.get(target)

                if best_probs_bin is not None and poisson_feat is not None:
                    for ml_w in [0.5, 0.7, 1.0]:
                        bl = ml_w * best_probs_bin + (1 - ml_w) * poisson_feat
                        for thr in [0.45, 0.50, 0.55]:
                            preds = (bl > thr).astype(int)
                            acc = (preds == y_te_bin.fillna(0).astype(int).values).mean()
                            if acc > best_acc_bin:
                                best_acc_bin = acc

                base_rate = y_te_bin.mean() if not y_te_bin.isna().all() else 0

                results[target] = {
                    "accuracy": round(best_acc_bin, 4),
                    "base_rate": round(float(base_rate), 3) if not np.isnan(base_rate) else 0,
                }
                log.info("  %s: acc=%.1f%% (base=%.1f%%)", name, best_acc_bin * 100, base_rate * 100)

            season_dur = time.time() - season_start
            log.info("  Season completed in %s", _fmt_duration(season_dur))

            all_season_results[test_season] = results

        return all_season_results

    def _get_binary_targets(self, df: pd.DataFrame) -> Dict[str, str]:
        """Get available binary market targets from the DataFrame."""
        targets = {}
        candidate_targets = {
            "over_2_5": "Over 2.5 Goals",
            "over_1_5": "Over 1.5 Goals",
            "over_3_5": "Over 3.5 Goals",
            "btts": "BTTS",
            "home_clean_sheet": "Home Clean Sheet",
            "away_clean_sheet": "Away Clean Sheet",
            "cards_over_3_5": "Cards Over 3.5",
            "cards_over_4_5": "Cards Over 4.5",
            "cards_over_5_5": "Cards Over 5.5",
            "corners_over_8_5": "Corners Over 8.5",
            "corners_over_9_5": "Corners Over 9.5",
            "corners_over_10_5": "Corners Over 10.5",
        }

        for target, name in candidate_targets.items():
            if target in df.columns and df[target].notna().sum() > 500:
                targets[target] = name

        return targets

    # -----------------------------------------------------------------
    # FULL OPTIMIZATION DISPATCHER
    # -----------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Run the configured optimization target(s) and return combined results."""
        if not self.load_data():
            return {"error": "Failed to load data"}

        results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "target": self.config.target,
                "n_trials": self.config.n_trials,
                "metric": self.config.metric,
                "test_seasons": self.config.test_seasons,
                "quick": self.config.quick,
            },
        }

        targets = (
            ["ensemble-weights", "model-hyperparams", "betting-params", "features", "all-markets"]
            if self.config.target == "all"
            else [self.config.target]
        )

        for target in targets:
            log.info("\n" + "#" * 70)
            log.info("RUNNING TARGET: %s", target.upper())
            log.info("#" * 70)

            if target == "ensemble-weights":
                results["ensemble_weights"] = self.optimize_ensemble_weights()
            elif target == "model-hyperparams":
                results["model_hyperparams"] = self.optimize_model_hyperparams()
            elif target == "betting-params":
                results["betting_params"] = self.optimize_betting_params()
            elif target == "features":
                results["features"] = self.optimize_features()
            elif target == "all-markets":
                results["all_markets"] = self.optimize_all_markets()
            else:
                log.warning("Unknown target: %s", target)

        total_dur = time.time() - self._start_time
        results["total_runtime"] = _fmt_duration(total_dur)
        log.info("\nTotal runtime: %s", _fmt_duration(total_dur))

        self.results = results
        return results


# =============================================================================
# REPORT PRINTING
# =============================================================================

def print_report(results: Dict[str, Any]):
    """Print formatted unified optimization report."""
    print("\n" + "=" * 70)
    print("UNIFIED OPTIMIZATION REPORT")
    print("=" * 70)

    config = results.get("config", {})
    print(f"\nTarget: {config.get('target', 'unknown')}")
    print(f"Metric: {config.get('metric', 'log_loss')}")
    print(f"Trials: {config.get('n_trials', 0)}")
    print(f"Runtime: {results.get('total_runtime', 'N/A')}")

    # --- Ensemble Weights ---
    if "ensemble_weights" in results:
        ew = results["ensemble_weights"]
        print("\n" + "-" * 70)
        print("ENSEMBLE WEIGHT OPTIMIZATION")
        print("-" * 70)

        print(f"\nBest config: {ew.get('best_config', 'N/A')}")
        print(f"Best accuracy: {ew.get('best_accuracy', 0):.1%}")
        print(f"Best log_loss: {ew.get('best_log_loss', 0):.4f}")
        print(f"Best weights: {ew.get('best_weights', {})}")

        if ew.get("improvement_vs_current") is not None:
            print(f"Improvement vs current: {ew['improvement_vs_current']:+.1%}")

        if ew.get("optuna_best"):
            ob = ew["optuna_best"]
            print(f"\nOptuna search ({ob.get('n_trials', 0)} trials):")
            print(f"  Best weights: {ob.get('weights', {})}")
            print(f"  Best score: {ob.get('score', 0):.4f}")

        configs = ew.get("predefined_configs", [])
        if configs:
            print(f"\nAll configurations (sorted by performance):")
            print(f"  {'Config':<16} {'Accuracy':>10} {'LogLoss':>10} {'HC Accuracy':>12}")
            print("  " + "-" * 50)
            for c in configs[:10]:
                if "error" in c:
                    continue
                print(
                    f"  {c['config_name']:<16} {c.get('accuracy', 0):>9.1%} "
                    f"{c.get('log_loss', 0):>10.4f} {c.get('high_conf_accuracy', 0):>11.1%}"
                )

    # --- Model Hyperparameters ---
    if "model_hyperparams" in results:
        mh = results["model_hyperparams"]
        print("\n" + "-" * 70)
        print("MODEL HYPERPARAMETER OPTIMIZATION")
        print("-" * 70)

        if "1x2_best_params" in mh:
            print(f"\n1X2 Best CatBoost params: {mh['1x2_best_params']}")

        wf = mh.get("walk_forward_tuning", {})
        for model_type in ["catboost", "xgboost", "lightgbm"]:
            if model_type in wf:
                mt = wf[model_type]
                if "error" in mt:
                    print(f"\n{model_type}: {mt['error']}")
                else:
                    print(f"\n{model_type} walk-forward best LL: {mt.get('best_score', 'N/A')}")

        for key, value in mh.items():
            if key.startswith("stacking_"):
                print(f"\n{key}:")
                if isinstance(value, dict):
                    for model_name, model_result in value.items():
                        if isinstance(model_result, dict) and "accuracy" in model_result:
                            print(
                                f"  {model_name}: acc={model_result['accuracy']:.1%}, "
                                f"LL={model_result.get('log_loss', 0):.4f}"
                            )

    # --- Betting Parameters ---
    if "betting_params" in results:
        bp = results["betting_params"]
        print("\n" + "-" * 70)
        print("BETTING PARAMETER OPTIMIZATION")
        print("-" * 70)

        best = bp.get("best_params", {})
        print(f"\nBest parameters:")
        print(f"  Kelly fraction: {best.get('kelly', 0):.2f}")
        print(f"  Min edge: {best.get('min_edge', 0):.3f}")
        print(f"  Max edge: {best.get('max_edge', 0):.3f}")
        print(f"  Confidence: {best.get('confidence', 0):.2f}")
        print(f"  ROI: {best.get('roi', 0):+.1f}%")
        print(f"  Total bets: {best.get('total_bets', 0)}")
        print(f"\nCombinations tested: {bp.get('total_combinations_tested', 0)}")

        top = bp.get("top_10", [])
        if top:
            print(f"\nTop 5 configurations:")
            print(f"  {'Kelly':>6} {'MinE':>6} {'MaxE':>6} {'Conf':>6} {'ROI':>8} {'Bets':>6}")
            print("  " + "-" * 42)
            for t in top[:5]:
                print(
                    f"  {t['kelly']:>6.2f} {t['min_edge']:>6.3f} {t['max_edge']:>6.3f} "
                    f"{t['confidence']:>6.2f} {t['roi']:>+7.1f}% {t['total_bets']:>6}"
                )

    # --- Feature Selection ---
    if "features" in results:
        fs = results["features"]
        print("\n" + "-" * 70)
        print("FEATURE SELECTION OPTIMIZATION")
        print("-" * 70)

        full = fs.get("full_model", {})
        print(f"\nFull model: {full.get('n_features', 0)} features, "
              f"acc={full.get('accuracy', 0):.1%}, LL={full.get('log_loss', 0):.4f}")
        print(f"Optimal features: {fs.get('optimal_n_features', 0)}")

        subsets = fs.get("subset_results", [])
        if subsets:
            print(f"\n  {'N Features':>12} {'Accuracy':>10} {'LogLoss':>10} {'vs Full':>10}")
            print("  " + "-" * 44)
            for s in subsets:
                print(
                    f"  {s['n_features']:>12} {s['accuracy']:>9.1%} "
                    f"{s['log_loss']:>10.4f} {s['acc_vs_full']:>+9.1%}"
                )

        top_feats = fs.get("top_features", [])
        if top_feats:
            print(f"\nTop 20 features:")
            for f in top_feats[:20]:
                print(f"  {f['importance']:>6.2f}  {f['name']}")

    # --- All Markets ---
    if "all_markets" in results:
        am = results["all_markets"]
        print("\n" + "-" * 70)
        print("ALL-MARKETS OPTIMIZATION")
        print("-" * 70)

        for season, markets in am.items():
            if not isinstance(markets, dict):
                continue
            print(f"\nSeason: {season}")
            for market, data in markets.items():
                if isinstance(data, dict) and "accuracy" in data:
                    br = data.get("base_rate", data.get("market_accuracy", 0))
                    acc = data["accuracy"]
                    gain = acc - br if isinstance(br, float) and br > 0 else 0
                    print(
                        f"  {market:<20} {acc:.1%} (baseline={br:.1%}, {gain:+.1%})"
                    )

    print("\n" + "=" * 70)


# =============================================================================
# CLI INTERFACE
# =============================================================================

def parse_args() -> OptimizeConfig:
    """Parse CLI arguments into OptimizeConfig."""
    parser = argparse.ArgumentParser(
        description="Unified optimization framework for Serie A predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/optimize_unified.py --target ensemble-weights
  python scripts/optimize_unified.py --target model-hyperparams --trials 50
  python scripts/optimize_unified.py --target betting-params --quick
  python scripts/optimize_unified.py --target features
  python scripts/optimize_unified.py --target all-markets
  python scripts/optimize_unified.py --target all
        """,
    )

    parser.add_argument(
        "--target",
        choices=[
            "ensemble-weights", "model-hyperparams", "betting-params",
            "features", "all-markets", "all",
        ],
        default="ensemble-weights",
        help="Optimization target (default: ensemble-weights)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of Optuna trials for hyperparameter tuning (default: 50)",
    )
    parser.add_argument(
        "--trials-binary",
        type=int,
        default=30,
        help="Number of Optuna trials for binary market tuning (default: 30)",
    )
    parser.add_argument(
        "--metric",
        choices=["log_loss", "accuracy", "rps", "brier"],
        default="log_loss",
        help="Target metric for optimization (default: log_loss)",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=["2023-2024", "2024-2025"],
        help="Test seasons for evaluation",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer trials and smaller search space",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=0.05,
        help="Grid step for ensemble weight search (default: 0.05)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Max sample size for quick evaluation (default: all)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    if args.quick:
        args.trials = min(args.trials, 10)
        args.trials_binary = min(args.trials_binary, 8)
        if args.sample_size is None:
            args.sample_size = 200

    config = OptimizeConfig(
        target=args.target,
        n_trials=args.trials,
        n_trials_binary=args.trials_binary,
        quick=args.quick,
        metric=args.metric,
        test_seasons=args.seasons,
        grid_step=args.grid_step,
        sample_size=args.sample_size,
    )

    return config


def main():
    config = parse_args()

    log.info("Unified Optimization Framework")
    log.info("Target: %s", config.target)
    log.info("Metric: %s", config.metric)
    log.info("Trials: %d (binary: %d)", config.n_trials, config.n_trials_binary)
    log.info("Quick: %s", config.quick)

    optimizer = UnifiedOptimizer(config)
    results = optimizer.run()

    print_report(results)

    # Save results
    save_path = config.save_path
    if save_path is None:
        results_dir = DATA_DIR / "optimization"
        results_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(
            results_dir / f"optimize_unified_{config.target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

    # Clean for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(item) for item in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        return obj

    cleaned = clean_for_json(results)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(cleaned, f, indent=2, default=str)
    log.info("Results saved to %s", save_path)


if __name__ == "__main__":
    main()
