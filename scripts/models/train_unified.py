#!/usr/bin/env python3
"""Unified Training Script - Consolidates ALL training modes.

Supports six training modes:
  hybrid           (default) xG regressors + 1X2 classifier with dynamic feature discovery
  classifier_only  Just the 1X2 CatBoost classifier
  xg_only          Just xG home/away CatBoost regressors (Poisson-based prediction)
  fast             Fixed params, no tuning, quick CatBoost train
  with_odds        Includes betting-odds features + walk-forward CV
  deep             LSTM + Transformer deep learning models (requires PyTorch)
  optimized        Full Optuna-tuned pipeline via ml.training (XGB + LGB + CB + ensemble)

Usage:
    python scripts/train_unified.py                         # hybrid mode (default)
    python scripts/train_unified.py --mode fast             # fast mode
    python scripts/train_unified.py --mode with_odds        # includes odds
    python scripts/train_unified.py --mode optimized --trials 50
    python scripts/train_unified.py --mode deep --epochs 50
    python scripts/train_unified.py --mode hybrid --top-k 100 --corr 0.90
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import DATA_DIR, MODELS_DIR
from features.build import get_ml_feature_columns
from ml.config import LABEL_MAP, ODDS_COLUMN_PATTERNS
from ml.data import _is_odds_feature
from ml.feature_selection import correlation_pruning, identify_odds_columns
from ml.poisson import poisson_win_prob

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ===================================================================
#  TrainConfig dataclass
# ===================================================================

@dataclass
class TrainConfig:
    """All configurable parameters for unified training."""

    # --- Mode selection ---
    mode: str = "hybrid"
    # Valid: hybrid, classifier_only, xg_only, fast, with_odds, deep, optimized

    # --- Feature selection ---
    top_k_features: int = 150           # importance-based feature cap
    corr_threshold: float = 0.85        # correlation pruning threshold
    nan_threshold: float = 0.80         # max NaN fraction for a usable feature
    exclude_odds: bool = False          # strip betting-odds columns

    # --- Cross-validation ---
    n_cv_splits: int = 5               # walk-forward CV folds
    min_train_seasons: int = 5          # minimum training seasons before first fold
    train_val_ratio: float = 0.85       # train / (train+val) within each fold
    final_train_ratio: float = 0.90     # train ratio for the final model

    # --- CatBoost regressor (xG) ---
    xg_iterations: int = 1500
    xg_learning_rate: float = 0.02
    xg_depth: int = 6
    xg_l2_leaf_reg: float = 3.0
    xg_min_data_in_leaf: int = 20
    xg_early_stopping: int = 150

    # --- CatBoost classifier (1X2) ---
    cls_iterations: int = 2000
    cls_learning_rate: float = 0.02
    cls_depth: int = 6
    cls_l2_leaf_reg: float = 3.0
    cls_min_data_in_leaf: int = 20
    cls_auto_class_weights: str = "SqrtBalanced"
    cls_early_stopping: int = 150

    # --- Fast mode overrides ---
    fast_iterations: int = 800
    fast_learning_rate: float = 0.03
    fast_depth: int = 6
    fast_l2_leaf_reg: float = 3.0
    fast_auto_class_weights: str = "Balanced"

    # --- With-odds mode ---
    odds_top_k: int = 100              # fewer features when odds dominate

    # --- Deep learning ---
    deep_epochs: int = 50
    deep_batch_size: int = 32
    deep_sequence_length: int = 10
    deep_lr_lstm: float = 0.001
    deep_lr_transformer: float = 0.0005
    deep_quick: bool = False

    # --- Optimized mode (Optuna) ---
    optuna_trials: int = 50
    optuna_quick: bool = False

    # --- Poisson ---
    poisson_max_goals: int = 10
    poisson_min_xg: float = 0.3
    poisson_max_xg: float = 4.0

    # --- High-confidence threshold ---
    high_conf_threshold: float = 0.55

    # --- Output ---
    model_dir: Optional[str] = None     # override output dir (default: MODELS_DIR/universal)
    random_seed: int = 42
    dry_run: bool = False               # train + evaluate but do NOT write any .cbm to disk

    # --- Exclude features (e.g. drift-detected) ---
    exclude_features: Optional[List[str]] = field(default_factory=list)


# ===================================================================
#  Shared constants (leakage prevention)
# ===================================================================

POST_MATCH_EXACT = {
    "home_shots_on_target", "away_shots_on_target",
    "home_shots_on_target_count", "away_shots_on_target_count",
    "home_shots_on_target_total", "away_shots_on_target_total",
    "home_total_shots", "away_total_shots",
    "home_crosses", "away_crosses",
    "home_red_cards", "away_red_cards",
    "home_saves_count", "away_saves_count",
    "home_saves", "away_saves",
    "home_corners", "away_corners",
    "home_offsides", "away_offsides",
    "home_touches", "away_touches",
    "home_dribbles", "away_dribbles",
    "home_tackles", "away_tackles",
    "home_interceptions", "away_interceptions",
    "home_fouls", "away_fouls",
    "home_clearances", "away_clearances",
    "home_blocks", "away_blocks",
    "home_possession", "away_possession",
    "home_passing_accuracy", "away_passing_accuracy",
    "attendance", "home_xg", "away_xg",
    "home_cards", "away_cards",
}

EXCLUDE_PATTERNS = [
    "match_id", "home_team", "away_team", "home_score", "away_score",
    "match_date", "season", "result", "league", "referee", "matchweek",
    "_has_", "ht_score",
]


# ===================================================================
#  Utility functions (preserved from individual scripts)
# ===================================================================

def _is_pre_match_feature(col: str) -> bool:
    """Check if a column is available before kickoff (excludes post-match stats)."""
    if col in POST_MATCH_EXACT:
        return False
    if col.startswith(("home_us_team_shots", "away_us_team_shots")):
        return True
    return True


def _is_pre_match_feature_with_odds(col: str) -> bool:
    """Check if a column is available pre-match (including odds)."""
    col_lower = col.lower()
    for pattern in EXCLUDE_PATTERNS:
        if pattern in col_lower:
            return False
    if col in POST_MATCH_EXACT:
        return False
    return True


def discover_features(df: pd.DataFrame, exclude_odds: bool = False,
                       nan_threshold: float = 0.80) -> List[str]:
    """Dynamically discover all ML-safe pre-match features from features.parquet."""
    ml_cols = get_ml_feature_columns(df)

    numeric_cols = [c for c in ml_cols
                    if df[c].dtype in ("float64", "float32", "int64", "int32", "int8", "uint8")]

    pre_match = [c for c in numeric_cols if _is_pre_match_feature(c)]

    if exclude_odds:
        pre_match = [c for c in pre_match if not _is_odds_feature(c)]

    nan_pcts = df[pre_match].isna().mean()
    usable = [c for c in pre_match if nan_pcts[c] < nan_threshold]

    log.info("Feature discovery: %d ML-safe -> %d numeric -> %d pre-match -> %d usable (<%.0f%% NaN)",
             len(ml_cols), len(numeric_cols), len(pre_match), len(usable), nan_threshold * 100)
    return usable


from ml.feature_selection import importance_based_selection as select_by_importance, correlation_pruning  # noqa: E402


def _select_by_importance_REMOVED():
    """Moved to ml.feature_selection.importance_based_selection."""
    pass


def _original_select_by_importance(
    X: pd.DataFrame, y: pd.Series, feature_names: List[str], top_k: int = 120,
) -> Tuple[List[str], Dict[str, float]]:
    """Select top_k features by XGBoost importance."""
    from xgboost import XGBRegressor

    X_sel = X[feature_names].fillna(0)

    # XGBoost forbids [, ], < and > in feature names -- sanitize temporarily
    safe_names = {f: f.replace("<", "_lt_").replace(">", "_gt_").replace("[", "_lb_").replace("]", "_rb_")
                  for f in feature_names}
    X_safe = X_sel.rename(columns=safe_names)

    selector = XGBRegressor(n_estimators=200, max_depth=6, random_state=42, verbosity=0)
    selector.fit(X_safe, y)

    reverse_map = {v: k for k, v in safe_names.items()}
    importance = {reverse_map[sn]: float(imp)
                  for sn, imp in zip(X_safe.columns, selector.feature_importances_)}
    sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    selected = [f for f, _ in sorted_feats[:top_k]]

    log.info("Importance selection: %d -> %d features", len(feature_names), len(selected))
    return selected, importance


def select_by_importance_classifier(
    X: pd.DataFrame, y: pd.Series, feature_names: List[str], top_k: int = 100,
) -> Tuple[List[str], Dict[str, float]]:
    """Select top_k features by XGBClassifier importance (used by with_odds mode)."""
    from xgboost import XGBClassifier

    X_full = X[feature_names].fillna(0)
    selector = XGBClassifier(n_estimators=200, max_depth=6, random_state=42, verbosity=0)
    selector.fit(X_full, y)

    feat_imp = sorted(zip(feature_names, selector.feature_importances_),
                      key=lambda x: x[1], reverse=True)
    selected = [f for f, _ in feat_imp[:top_k]]
    importance = {f: float(imp) for f, imp in feat_imp}

    log.info("Classifier importance selection: %d -> %d features", len(feature_names), len(selected))
    return selected, importance




def time_series_split(df: pd.DataFrame, n_splits: int = 5,
                       min_train: int = 5) -> List[Tuple[pd.Index, pd.Index]]:
    """Walk-forward time series splits by season."""
    seasons = sorted(df["season"].unique())
    splits = []

    for i in range(min_train, len(seasons)):
        train_seasons = seasons[:i]
        test_season = seasons[i]

        train_idx = df[df["season"].isin(train_seasons)].index
        test_idx = df[df["season"] == test_season].index

        if len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits[-n_splits:] if len(splits) > n_splits else splits


def evaluate_high_confidence(y_true: List[int], y_pred: List[int],
                              proba: List[List[float]],
                              threshold: float = 0.55) -> float:
    """Evaluate accuracy on high-confidence predictions only."""
    high_conf_mask = [max(p) >= threshold for p in proba]
    if not any(high_conf_mask):
        return 0.0

    y_true_hc = [y for y, m in zip(y_true, high_conf_mask) if m]
    y_pred_hc = [y for y, m in zip(y_pred, high_conf_mask) if m]

    return accuracy_score(y_true_hc, y_pred_hc)


def smart_impute(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Impute NaN values with sensible defaults based on feature name."""
    df = df.copy()
    for col in features:
        if df[col].isna().any():
            if "elo" in col:
                df[col] = df[col].ffill().fillna(1500.0)
            elif "h2h_" in col and "rate" in col:
                df[col] = df[col].fillna(1 / 3)
            elif "h2h_" in col:
                df[col] = df[col].fillna(0)
            elif "_roll_" in col or "streak" in col or "momentum" in col:
                df[col] = df[col].fillna(0.0)
            else:
                df[col] = df[col].fillna(0.0)
    return df


# ===================================================================
#  UnifiedTrainer
# ===================================================================

class UnifiedTrainer:
    """Unified trainer supporting all training modes."""

    def __init__(self, config: TrainConfig):
        self.cfg = config
        self._model_dir = self._resolve_model_dir()

    def _resolve_model_dir(self) -> Path:
        if self.cfg.model_dir:
            d = Path(self.cfg.model_dir)
        else:
            d = MODELS_DIR / "universal"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---------------------------------------------------------------
    #  Public entry point
    # ---------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Run training in the configured mode. Returns a results dict."""
        mode = self.cfg.mode.lower().strip()
        log.info("=" * 70)
        log.info("UNIFIED TRAINER  |  mode=%s", mode)
        log.info("=" * 70)

        dispatch = {
            "hybrid": self._train_hybrid,
            "classifier_only": self._train_classifier_only,
            "xg_only": self._train_xg_only,
            "fast": self._train_fast,
            "with_odds": self._train_with_odds,
            "deep": self._train_deep,
            "optimized": self._train_optimized,
        }

        handler = dispatch.get(mode)
        if handler is None:
            raise ValueError(f"Unknown training mode '{mode}'. "
                             f"Choose from: {', '.join(dispatch.keys())}")

        results = handler()
        log.info("=" * 70)
        log.info("TRAINING COMPLETE  |  mode=%s", mode)
        log.info("=" * 70)
        return results

    # ---------------------------------------------------------------
    #  Data loading (shared across most modes)
    # ---------------------------------------------------------------

    def _load_features(self) -> pd.DataFrame:
        """Load features.parquet and drop rows without scores."""
        feature_path = DATA_DIR / "features" / "features.parquet"
        df = pd.read_parquet(feature_path)
        df = df.dropna(subset=["home_score", "away_score"])
        log.info("Loaded %d matches with %d columns", len(df), len(df.columns))
        return df

    def _add_result_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a 'result' column (H/D/A) based on scores."""
        df = df.copy()
        df["result"] = df.apply(
            lambda r: "H" if r["home_score"] > r["away_score"]
            else ("A" if r["away_score"] > r["home_score"] else "D"),
            axis=1,
        )
        return df

    def _prepare_hybrid_data(self) -> Dict[str, Any]:
        """Full pipeline: load, discover features, importance select, correlation prune.

        Used by hybrid, classifier_only, and xg_only modes.
        """
        df = self._load_features()
        df = self._add_result_column(df)

        all_features = discover_features(
            df,
            exclude_odds=self.cfg.exclude_odds,
            nan_threshold=self.cfg.nan_threshold,
        )

        # Drop drift-detected / excluded features
        if self.cfg.exclude_features:
            before = len(all_features)
            exclude_set = set(self.cfg.exclude_features)
            all_features = [f for f in all_features if f not in exclude_set]
            dropped = before - len(all_features)
            if dropped:
                log.info("Excluded %d drift/NaN features (%d remaining)", dropped, len(all_features))

        df = smart_impute(df, all_features)

        # Build CV splits BEFORE feature selection to avoid leakage
        splits = time_series_split(df, n_splits=self.cfg.n_cv_splits,
                                     min_train=self.cfg.min_train_seasons)

        # Feature selection on FIRST TRAINING FOLD only (prevents look-ahead leakage).
        # Previously ran on the full dataset, which leaked test-fold information
        # into feature selection — inflating CV metrics by ~1-3%.
        first_train_idx = splits[0][0]
        df_first_train = df.loc[first_train_idx].copy()
        # Ensure _season column exists for feature_selection's TimeSeriesSplitter
        if "_season" not in df_first_train.columns and "season" in df_first_train.columns:
            df_first_train["_season"] = df_first_train["season"]
        selected, importance = select_by_importance(
            df_first_train, df_first_train["result"], all_features,
            top_k=self.cfg.top_k_features,
        )
        selected = correlation_pruning(
            df_first_train, selected, importance, threshold=self.cfg.corr_threshold,
        )
        log.info("Feature selection on first fold (%d samples): %d -> %d features",
                 len(first_train_idx), len(all_features), len(selected))

        X = df[selected].fillna(0)
        y_home_goals = df["home_score"]
        y_away_goals = df["away_score"]
        y_result = df["result"]

        log.info("Dataset: %d matches, %d features, %d CV splits",
                 len(df), len(selected), len(splits))

        return {
            "df": df,
            "X": X,
            "y_home_goals": y_home_goals,
            "y_away_goals": y_away_goals,
            "y_result": y_result,
            "available_features": selected,
            "importance": importance,
            "splits": splits,
        }

    # ---------------------------------------------------------------
    #  MODE: hybrid (xG regressors + classifier)
    # ---------------------------------------------------------------

    def _train_hybrid(self) -> Dict[str, Any]:
        """Train xG regressors + 1X2 classifier with walk-forward CV."""
        log.info("HYBRID MODE: xG regressors + 1X2 classifier")
        data = self._prepare_hybrid_data()

        # xG CV
        log.info("\n" + "=" * 60)
        log.info("PHASE 1: xG Regressor Cross-Validation")
        log.info("=" * 60)
        xg_metrics = self._evaluate_xg_cv(data)

        # Classifier CV
        log.info("\n" + "=" * 60)
        log.info("PHASE 2: 1X2 Classifier Cross-Validation")
        log.info("=" * 60)
        cls_metrics = self._evaluate_classifier_cv(data)

        # Train final models
        log.info("\n" + "=" * 60)
        log.info("PHASE 3: Training Final Production Models")
        log.info("=" * 60)
        train_result = self._train_final_models(data)

        # Save metadata
        cv_metrics = {**xg_metrics, **cls_metrics}
        fold_metrics = cv_metrics.pop("fold_metrics", [])
        metadata = self._save_hybrid_metadata(
            data["available_features"], cv_metrics, fold_metrics,
            train_result["top_xg_features"], train_result["top_classifier_features"],
        )

        # Summary
        log.info("\n" + "=" * 70)
        log.info("HYBRID TRAINING SUMMARY")
        log.info("=" * 70)
        log.info("  Features used: %d (dynamically discovered)", len(data["available_features"]))
        log.info("  xG-Poisson CV accuracy: %.1f%%", xg_metrics["xg_poisson_accuracy"] * 100)
        log.info("  xG-Poisson HC accuracy: %.1f%%", xg_metrics["xg_poisson_hc_accuracy"] * 100)
        log.info("  Classifier CV accuracy: %.1f%%", cls_metrics["classifier_accuracy"] * 100)
        log.info("  Classifier HC accuracy: %.1f%%", cls_metrics["classifier_hc_accuracy"] * 100)
        log.info("  Models saved to: %s", self._model_dir)

        return {
            "mode": "hybrid",
            "xg_metrics": xg_metrics,
            "classifier_metrics": cls_metrics,
            "features": data["available_features"],
            "metadata": metadata,
        }

    def _evaluate_xg_cv(self, data: Dict) -> Dict:
        """Walk-forward CV for xG home/away regressors."""
        from catboost import CatBoostRegressor

        X = data["X"]
        y_home = data["y_home_goals"]
        y_away = data["y_away_goals"]
        y_result = data["y_result"]
        splits = data["splits"]
        cfg = self.cfg

        all_preds, all_true, all_probs = [], [], []
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X.loc[train_idx], X.loc[test_idx]
            y_home_train = y_home.loc[train_idx]
            y_away_train = y_away.loc[train_idx]
            y_result_test = y_result.loc[test_idx].map(LABEL_MAP)
            n_train = int(len(X_train) * cfg.train_val_ratio)

            home_model = CatBoostRegressor(
                iterations=cfg.xg_iterations, learning_rate=cfg.xg_learning_rate,
                depth=cfg.xg_depth, l2_leaf_reg=cfg.xg_l2_leaf_reg,
                min_data_in_leaf=cfg.xg_min_data_in_leaf,
                early_stopping_rounds=cfg.xg_early_stopping,
                verbose=0, random_seed=cfg.random_seed,
            )
            home_model.fit(
                X_train.iloc[:n_train], y_home_train.iloc[:n_train],
                eval_set=(X_train.iloc[n_train:], y_home_train.iloc[n_train:]),
                verbose=False,
            )

            away_model = CatBoostRegressor(
                iterations=cfg.xg_iterations, learning_rate=cfg.xg_learning_rate,
                depth=cfg.xg_depth, l2_leaf_reg=cfg.xg_l2_leaf_reg,
                min_data_in_leaf=cfg.xg_min_data_in_leaf,
                early_stopping_rounds=cfg.xg_early_stopping,
                verbose=0, random_seed=cfg.random_seed,
            )
            away_model.fit(
                X_train.iloc[:n_train], y_away_train.iloc[:n_train],
                eval_set=(X_train.iloc[n_train:], y_away_train.iloc[n_train:]),
                verbose=False,
            )

            pred_home_xg = home_model.predict(X_test)
            pred_away_xg = away_model.predict(X_test)

            fold_preds, fold_probs = [], []
            for h_xg, a_xg in zip(pred_home_xg, pred_away_xg):
                probs = poisson_win_prob(h_xg, a_xg,
                                          max_goals=cfg.poisson_max_goals,
                                          min_xg=cfg.poisson_min_xg,
                                          max_xg=cfg.poisson_max_xg)
                fold_probs.append([probs["H"], probs["D"], probs["A"]])
                if probs["H"] >= probs["D"] and probs["H"] >= probs["A"]:
                    pred = LABEL_MAP["H"]
                elif probs["A"] >= probs["D"]:
                    pred = LABEL_MAP["A"]
                else:
                    pred = LABEL_MAP["D"]
                fold_preds.append(pred)

            fold_acc = accuracy_score(y_result_test, fold_preds)
            fold_hc_acc = evaluate_high_confidence(
                y_result_test.tolist(), fold_preds, fold_probs, threshold=cfg.high_conf_threshold,
            )
            home_mae = mean_absolute_error(y_home.loc[test_idx], pred_home_xg)
            away_mae = mean_absolute_error(y_away.loc[test_idx], pred_away_xg)

            fold_metrics.append({
                "fold": fold_idx + 1, "accuracy": fold_acc,
                "high_conf_accuracy": fold_hc_acc,
                "home_mae": home_mae, "away_mae": away_mae,
            })

            log.info("  xG Fold %d: Acc=%.1f%%, HC-Acc=%.1f%%, xG MAE: H=%.2f, A=%.2f",
                     fold_idx + 1, fold_acc * 100, fold_hc_acc * 100, home_mae, away_mae)

            all_preds.extend(fold_preds)
            all_true.extend(y_result_test.tolist())
            all_probs.extend(fold_probs)

        overall_acc = accuracy_score(all_true, all_preds)
        overall_hc_acc = evaluate_high_confidence(all_true, all_preds, all_probs,
                                                   threshold=cfg.high_conf_threshold)
        overall_logloss = log_loss(all_true, all_probs, labels=[0, 1, 2])

        log.info("\n  Overall xG-Poisson accuracy: %.1f%%", overall_acc * 100)
        log.info("  High-confidence (>%.0f%%) accuracy: %.1f%%",
                 cfg.high_conf_threshold * 100, overall_hc_acc * 100)

        return {
            "xg_poisson_accuracy": round(overall_acc, 4),
            "xg_poisson_hc_accuracy": round(overall_hc_acc, 4),
            "xg_poisson_logloss": round(overall_logloss, 4),
            "fold_metrics": fold_metrics,
        }

    def _evaluate_classifier_cv(self, data: Dict) -> Dict:
        """Walk-forward CV for 1X2 classifier."""
        from catboost import CatBoostClassifier

        X = data["X"]
        y_result = data["y_result"]
        splits = data["splits"]
        cfg = self.cfg

        cls_preds, cls_true, cls_probs = [], [], []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X.loc[train_idx], X.loc[test_idx]
            y_train = y_result.loc[train_idx].map(LABEL_MAP)
            y_test = y_result.loc[test_idx].map(LABEL_MAP)
            n_train = int(len(X_train) * cfg.train_val_ratio)

            model = CatBoostClassifier(
                iterations=cfg.cls_iterations, learning_rate=cfg.cls_learning_rate,
                depth=cfg.cls_depth, l2_leaf_reg=cfg.cls_l2_leaf_reg,
                min_data_in_leaf=cfg.cls_min_data_in_leaf,
                auto_class_weights=cfg.cls_auto_class_weights,
                early_stopping_rounds=cfg.cls_early_stopping,
                verbose=0, random_seed=cfg.random_seed,
            )
            model.fit(
                X_train.iloc[:n_train], y_train.iloc[:n_train],
                eval_set=(X_train.iloc[n_train:], y_train.iloc[n_train:]),
                verbose=False,
            )

            pred = model.predict(X_test)
            proba = model.predict_proba(X_test)

            fold_acc = accuracy_score(y_test, pred)
            fold_hc_acc = evaluate_high_confidence(
                y_test.tolist(), pred.tolist(), proba.tolist(),
                threshold=cfg.high_conf_threshold,
            )
            log.info("  Classifier Fold %d: Acc=%.1f%%, HC-Acc=%.1f%%",
                     fold_idx + 1, fold_acc * 100, fold_hc_acc * 100)

            cls_preds.extend(pred.tolist())
            cls_true.extend(y_test.tolist())
            cls_probs.extend(proba.tolist())

        class_acc = accuracy_score(cls_true, cls_preds)
        class_hc_acc = evaluate_high_confidence(cls_true, cls_preds, cls_probs,
                                                 threshold=cfg.high_conf_threshold)
        class_logloss = log_loss(cls_true, cls_probs, labels=[0, 1, 2])

        log.info("\n  Overall classifier accuracy: %.1f%%", class_acc * 100)
        log.info("  High-confidence accuracy: %.1f%%", class_hc_acc * 100)

        return {
            "classifier_accuracy": round(class_acc, 4),
            "classifier_hc_accuracy": round(class_hc_acc, 4),
            "classifier_logloss": round(class_logloss, 4),
        }

    def _train_final_models(self, data: Dict) -> Dict:
        """Train final xG regressors + classifier on all data and save."""
        from catboost import CatBoostRegressor, CatBoostClassifier

        cfg = self.cfg
        model_dir = self._model_dir
        X = data["X"]
        y_home = data["y_home_goals"]
        y_away = data["y_away_goals"]
        y_result = data["y_result"]
        features = data["available_features"]

        n_all = int(len(X) * cfg.final_train_ratio)

        # --- xG home ---
        home_model = CatBoostRegressor(
            iterations=cfg.xg_iterations, learning_rate=cfg.xg_learning_rate,
            depth=cfg.xg_depth, l2_leaf_reg=cfg.xg_l2_leaf_reg,
            min_data_in_leaf=cfg.xg_min_data_in_leaf,
            early_stopping_rounds=cfg.xg_early_stopping,
            verbose=0, random_seed=cfg.random_seed,
        )
        home_model.fit(
            X.iloc[:n_all], y_home.iloc[:n_all],
            eval_set=(X.iloc[n_all:], y_home.iloc[n_all:]),
            verbose=False,
        )

        # --- xG away ---
        away_model = CatBoostRegressor(
            iterations=cfg.xg_iterations, learning_rate=cfg.xg_learning_rate,
            depth=cfg.xg_depth, l2_leaf_reg=cfg.xg_l2_leaf_reg,
            min_data_in_leaf=cfg.xg_min_data_in_leaf,
            early_stopping_rounds=cfg.xg_early_stopping,
            verbose=0, random_seed=cfg.random_seed,
        )
        away_model.fit(
            X.iloc[:n_all], y_away.iloc[:n_all],
            eval_set=(X.iloc[n_all:], y_away.iloc[n_all:]),
            verbose=False,
        )

        # --- Classifier ---
        classifier = CatBoostClassifier(
            iterations=cfg.cls_iterations, learning_rate=cfg.cls_learning_rate,
            depth=cfg.cls_depth, l2_leaf_reg=cfg.cls_l2_leaf_reg,
            min_data_in_leaf=cfg.cls_min_data_in_leaf,
            auto_class_weights=cfg.cls_auto_class_weights,
            early_stopping_rounds=cfg.cls_early_stopping,
            verbose=0, random_seed=cfg.random_seed,
        )
        classifier.fit(
            X.iloc[:n_all], y_result.iloc[:n_all].map(LABEL_MAP),
            eval_set=(X.iloc[n_all:], y_result.iloc[n_all:].map(LABEL_MAP)),
            verbose=False,
        )

        # Save models (both versioned and canonical names)
        home_model.save_model(str(model_dir / "xg_home_extended.cbm"))
        away_model.save_model(str(model_dir / "xg_away_extended.cbm"))
        classifier.save_model(str(model_dir / "classifier_extended.cbm"))

        home_model.save_model(str(model_dir / "xg_home.cbm"))
        away_model.save_model(str(model_dir / "xg_away.cbm"))
        classifier.save_model(str(model_dir / "catboost_upcoming.cbm"))

        log.info("Saved xg_home.cbm, xg_away.cbm, catboost_upcoming.cbm to %s", model_dir)

        # Feature importance
        xg_importance = dict(zip(features, home_model.feature_importances_))
        top_xg = sorted(xg_importance.items(), key=lambda x: x[1], reverse=True)[:20]

        cls_importance = dict(zip(features, classifier.feature_importances_))
        top_cls = sorted(cls_importance.items(), key=lambda x: x[1], reverse=True)[:20]

        log.info("\nTop 10 xG features:")
        for feat, imp in top_xg[:10]:
            log.info("    %s: %.2f", feat, imp)
        log.info("\nTop 10 classifier features:")
        for feat, imp in top_cls[:10]:
            log.info("    %s: %.2f", feat, imp)

        return {
            "top_xg_features": top_xg,
            "top_classifier_features": top_cls,
            "model_dir": str(model_dir),
        }

    def _save_hybrid_metadata(
        self, features: List[str], cv_metrics: Dict,
        fold_metrics: List, top_xg: List, top_cls: List,
    ) -> Dict:
        """Save model metadata JSON (hybrid mode)."""
        model_dir = self._model_dir
        metadata = {
            "model_version": "unified_v1.0",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "training_mode": self.cfg.mode,
            "n_features": len(features),
            "feature_names": features,
            "cv_metrics": cv_metrics,
            "fold_metrics": fold_metrics,
            "top_xg_features": [f[0] if isinstance(f, tuple) else f for f in top_xg],
            "top_classifier_features": [f[0] if isinstance(f, tuple) else f for f in top_cls],
            "label_map": LABEL_MAP,
        }

        for name in [
            "extended_model_metadata.json",
            "catboost_upcoming_metadata.json",
            "catboost_no_odds_metadata.json",
        ]:
            with open(model_dir / name, "w") as f:
                json.dump(metadata, f, indent=2)

        log.info("Saved metadata to %s", model_dir)
        return metadata

    # ---------------------------------------------------------------
    #  MODE: classifier_only
    # ---------------------------------------------------------------

    def _train_classifier_only(self) -> Dict[str, Any]:
        """Train only the 1X2 classifier."""
        log.info("CLASSIFIER_ONLY MODE")
        data = self._prepare_hybrid_data()

        cls_metrics = self._evaluate_classifier_cv(data)

        # Train final classifier
        from catboost import CatBoostClassifier
        cfg = self.cfg
        X = data["X"]
        y_result = data["y_result"]
        n_all = int(len(X) * cfg.final_train_ratio)

        classifier = CatBoostClassifier(
            iterations=cfg.cls_iterations, learning_rate=cfg.cls_learning_rate,
            depth=cfg.cls_depth, l2_leaf_reg=cfg.cls_l2_leaf_reg,
            min_data_in_leaf=cfg.cls_min_data_in_leaf,
            auto_class_weights=cfg.cls_auto_class_weights,
            early_stopping_rounds=cfg.cls_early_stopping,
            verbose=0, random_seed=cfg.random_seed,
        )
        classifier.fit(
            X.iloc[:n_all], y_result.iloc[:n_all].map(LABEL_MAP),
            eval_set=(X.iloc[n_all:], y_result.iloc[n_all:].map(LABEL_MAP)),
            verbose=False,
        )

        model_path = self._model_dir / "catboost_upcoming.cbm"
        classifier.save_model(str(model_path))
        log.info("Saved classifier to %s", model_path)

        metadata = {
            "model_version": "unified_v1.0",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "training_mode": "classifier_only",
            "n_features": len(data["available_features"]),
            "feature_names": data["available_features"],
            "cv_metrics": cls_metrics,
            "label_map": LABEL_MAP,
        }
        with open(self._model_dir / "catboost_upcoming_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return {"mode": "classifier_only", "classifier_metrics": cls_metrics}

    # ---------------------------------------------------------------
    #  MODE: xg_only
    # ---------------------------------------------------------------

    def _train_xg_only(self) -> Dict[str, Any]:
        """Train only the xG home/away regressors."""
        log.info("XG_ONLY MODE")
        data = self._prepare_hybrid_data()

        xg_metrics = self._evaluate_xg_cv(data)

        # Train final regressors
        from catboost import CatBoostRegressor
        cfg = self.cfg
        X = data["X"]
        y_home = data["y_home_goals"]
        y_away = data["y_away_goals"]
        features = data["available_features"]
        n_all = int(len(X) * cfg.final_train_ratio)

        for target_name, y_target, out_name in [
            ("home", y_home, "xg_home.cbm"),
            ("away", y_away, "xg_away.cbm"),
        ]:
            model = CatBoostRegressor(
                iterations=cfg.xg_iterations, learning_rate=cfg.xg_learning_rate,
                depth=cfg.xg_depth, l2_leaf_reg=cfg.xg_l2_leaf_reg,
                min_data_in_leaf=cfg.xg_min_data_in_leaf,
                early_stopping_rounds=cfg.xg_early_stopping,
                verbose=0, random_seed=cfg.random_seed,
            )
            model.fit(
                X.iloc[:n_all], y_target.iloc[:n_all],
                eval_set=(X.iloc[n_all:], y_target.iloc[n_all:]),
                verbose=False,
            )
            if cfg.dry_run:
                log.info("DRY RUN — %s trained + evaluated but NOT saved", out_name)
            else:
                model.save_model(str(self._model_dir / out_name))
                log.info("Saved %s to %s", out_name, self._model_dir)

        return {"mode": "xg_only", "xg_metrics": xg_metrics, "dry_run": cfg.dry_run}

    # ---------------------------------------------------------------
    #  MODE: fast
    # ---------------------------------------------------------------

    def _train_fast(self) -> Dict[str, Any]:
        """Fast training: fixed params, no tuning, no CV."""
        from catboost import CatBoostClassifier

        log.info("FAST MODE: fixed parameters, no tuning")
        cfg = self.cfg

        df = self._load_features()
        df = self._add_result_column(df)

        # Identify feature columns (simple approach from train_no_odds_fast.py)
        exclude = {"match_id", "home_team", "away_team", "home_score", "away_score",
                    "match_date", "season", "result", "league", "referee", "matchweek"}

        feature_cols = [c for c in df.columns if c not in exclude
                        and df[c].dtype in ["float64", "int64", "float32", "int32"]
                        and not c.startswith("_")]

        # Remove odds columns
        odds_cols = identify_odds_columns(feature_cols)
        feature_cols = [f for f in feature_cols if f not in set(odds_cols)]
        log.info("Excluded %d odds columns, using %d features", len(odds_cols), len(feature_cols))

        X = df[feature_cols].fillna(0)
        y_int = df["result"].map(LABEL_MAP)

        log.info("Training on %d samples with %d features", len(X), len(feature_cols))

        model = CatBoostClassifier(
            iterations=cfg.fast_iterations,
            learning_rate=cfg.fast_learning_rate,
            depth=cfg.fast_depth,
            l2_leaf_reg=cfg.fast_l2_leaf_reg,
            auto_class_weights=cfg.fast_auto_class_weights,
            verbose=100,
            random_seed=cfg.random_seed,
        )
        model.fit(X, y_int.values)

        # Save model
        model_path = self._model_dir / "catboost_no_odds.cbm"
        model.save_model(str(model_path))
        log.info("Saved model to %s", model_path)

        # Evaluate on training data
        y_pred = model.predict(X.values)
        train_accuracy = (y_pred == y_int.values).mean()
        log.info("Training accuracy: %.1f%%", train_accuracy * 100)

        # Save metadata
        metadata = {
            "model_type": "catboost",
            "model_version": "unified_v1.0",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "training_mode": "fast",
            "n_features": len(feature_cols),
            "n_samples": len(X),
            "feature_names": feature_cols,
            "label_map": LABEL_MAP,
            "excluded_odds_features": odds_cols,
            "train_accuracy": round(float(train_accuracy), 4),
        }
        with open(self._model_dir / "catboost_no_odds_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return {"mode": "fast", "train_accuracy": float(train_accuracy), "model_path": str(model_path)}

    # ---------------------------------------------------------------
    #  MODE: with_odds
    # ---------------------------------------------------------------

    def _train_with_odds(self) -> Dict[str, Any]:
        """Train CatBoost with betting-odds features + walk-forward CV."""
        from catboost import CatBoostClassifier

        log.info("WITH_ODDS MODE: includes betting odds features")
        cfg = self.cfg

        df = self._load_features()
        df = self._add_result_column(df)

        # Get all pre-match features INCLUDING odds
        all_cols = [c for c in df.columns if df[c].dtype in ["float64", "int64", "float32", "int32"]
                    and not c.startswith("_")]
        pre_match_cols = [c for c in all_cols if _is_pre_match_feature_with_odds(c)]

        odds_cols = [c for c in pre_match_cols if "odds" in c.lower() or "pinnacle" in c.lower()]
        log.info("Odds columns available: %d", len(odds_cols))
        log.info("Total pre-match features (with odds): %d", len(pre_match_cols))

        # Build CV splits BEFORE feature selection to avoid leakage
        splits = time_series_split(df, n_splits=cfg.n_cv_splits, min_train=cfg.min_train_seasons)
        log.info("Using %d walk-forward splits", len(splits))

        # Feature selection on FIRST TRAINING FOLD only (prevents look-ahead leakage)
        first_train_idx = splits[0][0]
        df_first_train = df.loc[first_train_idx]
        X_first = df_first_train[pre_match_cols].fillna(0)
        y_first = df_first_train["result"].map(LABEL_MAP)

        selected_features, importance = select_by_importance_classifier(
            X_first, y_first, pre_match_cols, top_k=cfg.odds_top_k,
        )
        log.info("Feature selection on first fold (%d samples): %d -> %d features",
                 len(first_train_idx), len(pre_match_cols), len(selected_features))
        log.info("Top 10: %s", selected_features[:10])

        X = df[selected_features].fillna(0)
        y_int = df["result"].map(LABEL_MAP)

        all_preds, all_true, all_proba = [], [], []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            log.info("\n--- Fold %d/%d ---", fold_idx + 1, len(splits))

            X_train, X_test = X.loc[train_idx], X.loc[test_idx]
            y_train, y_test = y_int.loc[train_idx], y_int.loc[test_idx]

            n_train = int(len(X_train) * cfg.train_val_ratio)
            X_tr, y_tr = X_train.iloc[:n_train], y_train.iloc[:n_train]
            X_val, y_val = X_train.iloc[n_train:], y_train.iloc[n_train:]

            log.info("  Train: %d, Val: %d, Test: %d", len(X_tr), len(X_val), len(X_test))

            model = CatBoostClassifier(
                iterations=cfg.cls_iterations,
                learning_rate=cfg.cls_learning_rate,
                depth=cfg.cls_depth,
                l2_leaf_reg=cfg.cls_l2_leaf_reg,
                auto_class_weights=cfg.cls_auto_class_weights,
                early_stopping_rounds=cfg.cls_early_stopping,
                verbose=0, random_seed=cfg.random_seed,
            )
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)

            test_proba = model.predict_proba(X_test)
            test_pred = np.argmax(test_proba, axis=1)

            fold_acc = accuracy_score(y_test, test_pred)
            fold_loss = log_loss(y_test, test_proba)
            log.info("  Fold accuracy: %.1f%%, log_loss: %.3f", fold_acc * 100, fold_loss)

            for cls_name, cls_idx in LABEL_MAP.items():
                mask = y_test == cls_idx
                if mask.sum() > 0:
                    cls_acc = (test_pred[mask] == cls_idx).mean()
                    log.info("    %s: %.1f%%", cls_name, cls_acc * 100)

            all_preds.extend(test_pred.tolist())
            all_true.extend(y_test.tolist())
            all_proba.extend(test_proba.tolist())

        # Overall results
        overall_acc = accuracy_score(all_true, all_preds)
        overall_loss = log_loss(all_true, all_proba)

        log.info("\n" + "=" * 70)
        log.info("CROSS-VALIDATION RESULTS (WITH ODDS)")
        log.info("=" * 70)
        log.info("Overall Accuracy: %.1f%%", overall_acc * 100)
        log.info("Log Loss: %.3f", overall_loss)

        # Per-class
        log.info("\nPer-class accuracy:")
        for cls_name, cls_idx in LABEL_MAP.items():
            mask = np.array(all_true) == cls_idx
            if mask.sum() > 0:
                cls_acc = (np.array(all_preds)[mask] == cls_idx).mean()
                log.info("  %s: %.1f%% (%d samples)", cls_name, cls_acc * 100, mask.sum())

        # Train final model on all data
        log.info("\nTraining final model on all data...")
        n_train = int(len(X) * cfg.train_val_ratio)

        final_model = CatBoostClassifier(
            iterations=cfg.cls_iterations, learning_rate=cfg.cls_learning_rate,
            depth=cfg.cls_depth, l2_leaf_reg=cfg.cls_l2_leaf_reg,
            auto_class_weights=cfg.cls_auto_class_weights,
            early_stopping_rounds=cfg.cls_early_stopping,
            verbose=0, random_seed=cfg.random_seed,
        )
        final_model.fit(X.iloc[:n_train], y_int.iloc[:n_train],
                         eval_set=(X.iloc[n_train:], y_int.iloc[n_train:]),
                         verbose=False)

        model_path = self._model_dir / "catboost_latest.cbm"
        final_model.save_model(str(model_path))
        log.info("Saved model to %s", model_path)

        # Save metadata
        metadata = {
            "model_type": "catboost_with_odds",
            "model_version": "unified_v1.0",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "training_mode": "with_odds",
            "cv_accuracy": round(overall_acc, 4),
            "cv_log_loss": round(overall_loss, 4),
            "n_features": len(selected_features),
            "n_samples": len(X),
            "feature_names": selected_features,
            "label_map": LABEL_MAP,
            "includes_odds": True,
        }
        with open(self._model_dir / "catboost_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        if overall_acc >= 0.65:
            log.info("TARGET MET! Accuracy %.1f%% >= 65%%", overall_acc * 100)
        else:
            log.info("Gap to 65%% target: %.1f%%", (0.65 - overall_acc) * 100)

        return {
            "mode": "with_odds",
            "cv_accuracy": round(overall_acc, 4),
            "cv_log_loss": round(overall_loss, 4),
            "model_path": str(model_path),
        }

    # ---------------------------------------------------------------
    #  MODE: deep (LSTM + Transformer)
    # ---------------------------------------------------------------

    def _train_deep(self) -> Dict[str, Any]:
        """Train LSTM + Transformer deep learning models."""
        try:
            import torch
            from torch.utils.data import DataLoader, Subset
        except ImportError:
            log.error("PyTorch not installed. Run: pip install torch")
            return {"mode": "deep", "error": "PyTorch not installed"}

        try:
            from models.deep_learning import (
                LSTMFormModel,
                TransformerMatchModel,
                MatchSequenceDataset,
                DeepModelTrainer,
            )
        except ImportError as e:
            log.error("Deep learning modules not found: %s", e)
            return {"mode": "deep", "error": str(e)}

        cfg = self.cfg
        epochs = 10 if cfg.deep_quick else cfg.deep_epochs
        batch_size = 64 if cfg.deep_quick else cfg.deep_batch_size

        log.info("DEEP MODE: LSTM + Transformer")
        if cfg.deep_quick:
            log.info("Quick mode: %d epochs", epochs)

        # Load data
        features_path = DATA_DIR / "features" / "features.parquet"
        if not features_path.exists():
            log.error("Features file not found: %s", features_path)
            return {"mode": "deep", "error": "Features file not found"}

        df = pd.read_parquet(features_path)
        df = df.sort_values("match_date").reset_index(drop=True)
        df = df.dropna(subset=["home_score", "away_score"])
        log.info("Loaded %d matches for deep learning", len(df))

        # Create dataset
        dataset = MatchSequenceDataset(df, sequence_length=cfg.deep_sequence_length)
        log.info("Created dataset with %d samples", len(dataset))

        # Chronological split
        total = len(dataset)
        test_size = int(total * 0.15)
        val_size = int(total * 0.15)
        train_size = total - test_size - val_size

        train_dataset = Subset(dataset, range(0, train_size))
        val_dataset = Subset(dataset, range(train_size, train_size + val_size))
        test_dataset = Subset(dataset, range(train_size + val_size, total))

        log.info("Split: Train=%d, Val=%d, Test=%d", train_size, val_size, test_size)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        trainer = DeepModelTrainer()
        results = {"mode": "deep", "timestamp": datetime.now().isoformat()}

        # --- LSTM ---
        log.info("\n" + "=" * 50)
        log.info("Training LSTM Model")
        log.info("=" * 50)

        lstm_history = trainer.train_lstm(train_loader, val_loader,
                                           epochs=epochs, lr=cfg.deep_lr_lstm)
        lstm_metrics = self._evaluate_deep_model(trainer.lstm_model, test_loader, torch)

        log.info("LSTM Test Accuracy: %.1f%%", lstm_metrics["accuracy"] * 100)
        log.info("LSTM High Conf Accuracy: %.1f%% (%d samples)",
                 lstm_metrics["high_conf_accuracy"] * 100, lstm_metrics["high_conf_count"])

        results["lstm"] = {
            "final_train_loss": lstm_history["train_loss"][-1],
            "final_val_loss": lstm_history["val_loss"][-1] if lstm_history["val_loss"] else None,
            "final_val_acc": lstm_history["val_acc"][-1] if lstm_history["val_acc"] else None,
            "test_accuracy": lstm_metrics["accuracy"],
            "test_high_conf_accuracy": lstm_metrics["high_conf_accuracy"],
        }

        # --- Transformer ---
        log.info("\n" + "=" * 50)
        log.info("Training Transformer Model")
        log.info("=" * 50)

        transformer_history = trainer.train_transformer(train_loader, val_loader,
                                                         epochs=epochs, lr=cfg.deep_lr_transformer)
        transformer_metrics = self._evaluate_deep_model(trainer.transformer_model, test_loader, torch)

        log.info("Transformer Test Accuracy: %.1f%%", transformer_metrics["accuracy"] * 100)
        log.info("Transformer High Conf Accuracy: %.1f%%",
                 transformer_metrics["high_conf_accuracy"] * 100)

        results["transformer"] = {
            "final_train_loss": transformer_history["train_loss"][-1],
            "final_val_loss": transformer_history["val_loss"][-1] if transformer_history["val_loss"] else None,
            "final_val_acc": transformer_history["val_acc"][-1] if transformer_history["val_acc"] else None,
            "test_accuracy": transformer_metrics["accuracy"],
            "test_high_conf_accuracy": transformer_metrics["high_conf_accuracy"],
        }

        # Save models
        trainer.save_models()

        # Save results
        deep_model_dir = MODELS_DIR / "deep"
        deep_model_dir.mkdir(parents=True, exist_ok=True)
        results_path = deep_model_dir / "training_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Summary
        avg_acc = (lstm_metrics["accuracy"] + transformer_metrics["accuracy"]) / 2
        avg_hc = (lstm_metrics["high_conf_accuracy"] + transformer_metrics["high_conf_accuracy"]) / 2
        log.info("\nDeep learning ensemble average: %.1f%% (HC: %.1f%%)", avg_acc * 100, avg_hc * 100)

        return results

    @staticmethod
    def _evaluate_deep_model(model, test_loader, torch_module) -> Dict:
        """Evaluate a PyTorch model on a test DataLoader."""
        model.eval()
        correct = 0
        total = 0
        all_probs = []
        all_labels = []

        with torch_module.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                _, predicted = outputs.max(1)

                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()

                all_probs.extend(outputs.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        accuracy = correct / total if total > 0 else 0.0

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        max_probs = all_probs.max(axis=1)
        high_conf_mask = max_probs > 0.5
        if high_conf_mask.sum() > 0:
            high_conf_preds = all_probs[high_conf_mask].argmax(axis=1)
            high_conf_labels = all_labels[high_conf_mask]
            high_conf_acc = (high_conf_preds == high_conf_labels).mean()
        else:
            high_conf_acc = 0.0

        return {
            "accuracy": accuracy,
            "high_conf_accuracy": float(high_conf_acc),
            "high_conf_count": int(high_conf_mask.sum()),
            "total": total,
        }

    # ---------------------------------------------------------------
    #  MODE: optimized (full Optuna pipeline via ml.training)
    # ---------------------------------------------------------------

    def _train_optimized(self) -> Dict[str, Any]:
        """Full Optuna-tuned pipeline (XGB + LGB + CB + ensemble)."""
        from ml.training import train_optimized

        cfg = self.cfg
        n_trials = 10 if cfg.optuna_quick else cfg.optuna_trials

        log.info("OPTIMIZED MODE: Optuna tuning with %d trials", n_trials)
        log.info("  Exclude odds: %s", cfg.exclude_odds)

        results = train_optimized(
            n_tune_trials=n_trials,
            top_k_features=cfg.top_k_features,
            corr_threshold=cfg.corr_threshold,
            exclude_odds=cfg.exclude_odds,
        )

        # Print summary
        if "n_features_selected" in results:
            log.info("Features: %d -> %d", results["n_features_original"],
                     results["n_features_selected"])

        for key in ["catboost_cv", "xgboost_cv", "lightgbm_cv"]:
            if key in results:
                cv = results[key]
                avg_acc = cv["accuracy"].mean()
                avg_ll = cv["log_loss"].mean()
                log.info("%s: accuracy=%.4f, log_loss=%.4f", key, avg_acc, avg_ll)

        if "ensemble_cv" in results:
            ens = results["ensemble_cv"]
            log.info("Ensemble: accuracy=%.4f",
                     ens["ensemble_accuracy"].mean())

        log.info("\nSaved models:")
        for key in results:
            if key.endswith("_path"):
                log.info("  %s: %s", key, results[key])

        results["mode"] = "optimized"
        return results


# ===================================================================
#  BACKWARD-COMPATIBLE WRAPPERS
#  (for callers that used train_extended_ensemble)
# ===================================================================

def load_and_prepare_data(exclude_features=None) -> Dict[str, Any]:
    """Backward-compatible wrapper matching train_extended_ensemble.load_and_prepare_data().

    Returns dict with keys: df, X, y_home_goals, y_away_goals, y_result,
                            available_features, importance, splits.
    """
    cfg = TrainConfig(mode="hybrid")
    if exclude_features:
        cfg.exclude_features = list(exclude_features)
    trainer = UnifiedTrainer(cfg)
    data = trainer._prepare_hybrid_data()
    return data


def evaluate_xg_cv(data: Dict) -> Dict:
    """Backward-compatible wrapper matching train_extended_ensemble.evaluate_xg_cv()."""
    cfg = TrainConfig(mode="xg_only")
    trainer = UnifiedTrainer(cfg)
    return trainer._evaluate_xg_cv(data)


def evaluate_classifier_cv(data: Dict) -> Dict:
    """Backward-compatible wrapper matching train_extended_ensemble.evaluate_classifier_cv()."""
    cfg = TrainConfig(mode="classifier_only")
    trainer = UnifiedTrainer(cfg)
    return trainer._evaluate_classifier_cv(data)


def train_final_models(data: Dict, model_dir: Path = None) -> Dict:
    """Backward-compatible wrapper matching train_extended_ensemble.train_final_models()."""
    cfg = TrainConfig(mode="hybrid")
    if model_dir:
        cfg.model_dir = str(model_dir)
    trainer = UnifiedTrainer(cfg)
    return trainer._train_final_models(data)


def save_metadata(model_dir: Path, available_features: list, cv_metrics: dict,
                  fold_metrics: list, top_xg_features: list,
                  top_classifier_features: list):
    """Backward-compatible wrapper matching train_extended_ensemble.save_metadata()."""
    from datetime import datetime, timezone
    import json as _json

    metadata = {
        "model_version": "unified_v1.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "features": available_features,
        "n_features": len(available_features),
        "cv_metrics": cv_metrics,
        "fold_metrics": fold_metrics,
        "top_xg_features": top_xg_features,
        "top_classifier_features": top_classifier_features,
    }
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    meta_path = model_dir / "metadata.json"
    meta_path.write_text(_json.dumps(metadata, indent=2, default=str))
    log.info("Saved metadata to %s", meta_path)


# ===================================================================
#  CLI Interface
# ===================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the unified training CLI."""
    parser = argparse.ArgumentParser(
        description="Unified Serie A Prediction Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training modes:
  hybrid           (default) xG regressors + 1X2 classifier
  classifier_only  Just the 1X2 CatBoost classifier
  xg_only          Just xG home/away CatBoost regressors
  fast             Fixed params, no tuning, quick CatBoost
  with_odds        Includes betting-odds features + walk-forward CV
  deep             LSTM + Transformer deep learning models
  optimized        Full Optuna-tuned pipeline (XGB + LGB + CB + ensemble)

Examples:
  python scripts/train_unified.py
  python scripts/train_unified.py --mode fast
  python scripts/train_unified.py --mode optimized --trials 80 --exclude-odds
  python scripts/train_unified.py --mode deep --epochs 100
  python scripts/train_unified.py --mode hybrid --top-k 120 --corr 0.90
""",
    )

    # Mode
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="hybrid",
        choices=["hybrid", "classifier_only", "xg_only", "fast", "with_odds", "deep", "optimized"],
        help="Training mode (default: hybrid)",
    )

    # Feature selection
    parser.add_argument("--top-k", type=int, default=150,
                        help="Top-K features to keep after importance ranking (default: 150)")
    parser.add_argument("--corr", type=float, default=0.85,
                        help="Correlation pruning threshold (default: 0.85)")
    parser.add_argument("--exclude-odds", action="store_true",
                        help="Exclude betting-odds features")

    # Cross-validation
    parser.add_argument("--cv-splits", type=int, default=5,
                        help="Number of walk-forward CV splits (default: 5)")

    # Optuna (optimized mode)
    parser.add_argument("--trials", type=int, default=50,
                        help="Number of Optuna trials for optimized mode (default: 50)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer trials/epochs for fast iteration")

    # Deep learning
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs for deep mode (default: 50)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for deep mode (default: 32)")
    parser.add_argument("--sequence-length", type=int, default=10,
                        help="Sequence length for deep mode (default: 10)")

    # Output
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Override model output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Train + evaluate but do NOT write any .cbm to disk "
                             "(currently honoured by xg_only mode)")

    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    """Build a TrainConfig from parsed CLI arguments."""
    config = TrainConfig(
        mode=args.mode,
        top_k_features=args.top_k,
        corr_threshold=args.corr,
        exclude_odds=args.exclude_odds,
        n_cv_splits=args.cv_splits,
        optuna_trials=args.trials,
        optuna_quick=args.quick,
        deep_epochs=args.epochs,
        deep_batch_size=args.batch_size,
        deep_sequence_length=args.sequence_length,
        deep_quick=args.quick,
        model_dir=args.model_dir,
        random_seed=args.seed,
        dry_run=args.dry_run,
    )
    return config


def main():
    parser = build_parser()
    args = parser.parse_args()

    config = config_from_args(args)

    log.info("Configuration:")
    log.info("  Mode: %s", config.mode)
    log.info("  Top-K features: %d", config.top_k_features)
    log.info("  Correlation threshold: %.2f", config.corr_threshold)
    log.info("  Exclude odds: %s", config.exclude_odds)
    log.info("  CV splits: %d", config.n_cv_splits)
    log.info("  Random seed: %d", config.random_seed)

    trainer = UnifiedTrainer(config)
    results = trainer.run()

    return results


if __name__ == "__main__":
    main()
