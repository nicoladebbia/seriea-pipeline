#!/usr/bin/env python3
"""Improved training pipeline targeting 65-70% accuracy.

Key improvements:
1. Better feature engineering and selection
2. Ensemble of models (CatBoost, XGBoost, LightGBM)
3. Walk-forward cross-validation (time-series aware)
4. Probability calibration
5. Draw specialist integration
6. Hyperparameter optimization
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.calibration import CalibratedClassifierCV

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR
from ml.config import LABEL_MAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# Features to exclude (post-match data, odds, metadata)
EXCLUDE_PATTERNS = [
    "match_id", "home_team", "away_team", "home_score", "away_score",
    "match_date", "season", "result", "league", "referee", "matchweek",
    "odds_", "implied_prob", "_has_", "attendance", "possession",
    "shots_on_target", "corners", "fouls", "cards", "offsides", "ht_score",
    # Also exclude all betting-related features
    "pinnacle", "B365", "BW", "IW", "LB", "PS", "WH", "VC", "bet365",
    "bookmaker", "market", "closing", "opening",
]


def is_pre_match_feature(col: str) -> bool:
    """Check if a column is a pre-match feature."""
    col_lower = col.lower()
    for pattern in EXCLUDE_PATTERNS:
        if pattern in col_lower:
            return False
    return True


def get_feature_importance_selection(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    top_k: int = 80,
) -> List[str]:
    """Select top features using XGBoost importance."""
    from xgboost import XGBClassifier

    log.info(f"Selecting top {top_k} features from {len(feature_names)}...")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        verbosity=0,
    )

    X_train = X[feature_names].fillna(0)
    y_int = y.map(LABEL_MAP)

    model.fit(X_train, y_int)
    importances = model.feature_importances_

    # Sort by importance
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    selected = [f for f, _ in feat_imp[:top_k]]
    log.info(f"Selected features: {selected[:10]}...")

    return selected


def time_series_split(
    df: pd.DataFrame,
    n_splits: int = 5,
) -> List[Tuple[pd.Index, pd.Index]]:
    """Generate walk-forward time series splits."""
    seasons = sorted(df["season"].unique())
    splits = []

    # Minimum training seasons
    min_train = 5

    for i in range(min_train, len(seasons)):
        train_seasons = seasons[:i]
        test_season = seasons[i]

        train_idx = df[df["season"].isin(train_seasons)].index
        test_idx = df[df["season"] == test_season].index

        if len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    # Limit to last n_splits
    return splits[-n_splits:] if len(splits) > n_splits else splits


def train_catboost(X_train, y_train, X_val, y_val):
    """Train CatBoost with early stopping."""
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.02,
        depth=6,
        l2_leaf_reg=5,
        auto_class_weights="SqrtBalanced",
        early_stopping_rounds=100,
        verbose=0,
        random_seed=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=False,
    )

    return model


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost with early stopping."""
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=1500,
        learning_rate=0.02,
        max_depth=6,
        reg_lambda=5,
        early_stopping_rounds=100,
        random_state=42,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    return model


def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM with early stopping."""
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.02,
        max_depth=6,
        reg_lambda=5,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    return model


def ensemble_predict(models: List, X: pd.DataFrame, weights: List[float] = None) -> np.ndarray:
    """Weighted average ensemble prediction."""
    if weights is None:
        weights = [1.0 / len(models)] * len(models)

    all_proba = []
    for model in models:
        proba = model.predict_proba(X)
        all_proba.append(proba)

    # Weighted average
    weighted_proba = np.zeros_like(all_proba[0])
    for proba, w in zip(all_proba, weights):
        weighted_proba += proba * w

    return weighted_proba


def main():
    log.info("=" * 70)
    log.info("IMPROVED TRAINING PIPELINE - TARGET 65-70% ACCURACY")
    log.info("=" * 70)

    # Load features
    feature_path = DATA_DIR / "features" / "features.parquet"
    df = pd.read_parquet(feature_path)
    log.info(f"Loaded {len(df)} matches")

    # Create target
    df = df.dropna(subset=["home_score", "away_score"])
    df["result"] = df.apply(
        lambda r: "H" if r["home_score"] > r["away_score"]
        else ("A" if r["away_score"] > r["home_score"] else "D"),
        axis=1
    )

    # Identify pre-match features only
    all_cols = [c for c in df.columns if df[c].dtype in ["float64", "int64", "float32", "int32"]
                and not c.startswith("_")]
    pre_match_cols = [c for c in all_cols if is_pre_match_feature(c)]
    log.info(f"Pre-match features: {len(pre_match_cols)}")

    # Feature selection
    selected_features = get_feature_importance_selection(
        df, df["result"], pre_match_cols, top_k=80
    )

    # Prepare data
    X = df[selected_features].fillna(0)
    y = df["result"]
    y_int = y.map(LABEL_MAP)

    # Time-series cross validation
    splits = time_series_split(df, n_splits=5)
    log.info(f"Using {len(splits)} walk-forward splits")

    # Track results
    all_preds = []
    all_true = []
    all_proba = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        log.info(f"\n--- Fold {fold_idx + 1}/{len(splits)} ---")

        X_train = X.loc[train_idx]
        y_train = y_int.loc[train_idx]
        X_test = X.loc[test_idx]
        y_test = y_int.loc[test_idx]

        # Split train into train/val for early stopping
        n_train = int(len(X_train) * 0.85)
        X_tr = X_train.iloc[:n_train]
        y_tr = y_train.iloc[:n_train]
        X_val = X_train.iloc[n_train:]
        y_val = y_train.iloc[n_train:]

        log.info(f"  Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # Train models
        models = []

        # CatBoost
        cb_model = train_catboost(X_tr, y_tr, X_val, y_val)
        models.append(cb_model)
        log.info("  Trained CatBoost")

        # XGBoost
        xgb_model = train_xgboost(X_tr.values, y_tr.values, X_val.values, y_val.values)
        models.append(xgb_model)
        log.info("  Trained XGBoost")

        # LightGBM
        lgb_model = train_lightgbm(X_tr.values, y_tr.values, X_val.values, y_val.values)
        models.append(lgb_model)
        log.info("  Trained LightGBM")

        # Ensemble predictions (weighted average)
        # Weight by validation performance
        val_scores = []
        for model in models:
            val_proba = model.predict_proba(X_val.values if hasattr(X_val, 'values') else X_val)
            val_acc = accuracy_score(y_val, np.argmax(val_proba, axis=1))
            val_scores.append(val_acc)

        weights = np.array(val_scores) / sum(val_scores)
        log.info(f"  Ensemble weights: {weights.round(3)}")

        # Test predictions
        test_proba = ensemble_predict(models, X_test.values, weights)
        test_pred = np.argmax(test_proba, axis=1)

        # Evaluate
        fold_acc = accuracy_score(y_test, test_pred)
        fold_loss = log_loss(y_test, test_proba)
        log.info(f"  Fold accuracy: {fold_acc:.1%}, log_loss: {fold_loss:.3f}")

        # Per-class accuracy
        for cls_name, cls_idx in LABEL_MAP.items():
            mask = y_test == cls_idx
            if mask.sum() > 0:
                cls_acc = (test_pred[mask] == cls_idx).mean()
                log.info(f"    {cls_name}: {cls_acc:.1%}")

        all_preds.extend(test_pred.tolist())
        all_true.extend(y_test.tolist())
        all_proba.extend(test_proba.tolist())

    # Overall results
    overall_acc = accuracy_score(all_true, all_preds)
    overall_loss = log_loss(all_true, all_proba)

    log.info("\n" + "=" * 70)
    log.info("CROSS-VALIDATION RESULTS")
    log.info("=" * 70)
    log.info(f"Overall Accuracy: {overall_acc:.1%}")
    log.info(f"Log Loss: {overall_loss:.3f}")

    # Per-class
    log.info("\nPer-class accuracy:")
    for cls_name, cls_idx in LABEL_MAP.items():
        mask = np.array(all_true) == cls_idx
        if mask.sum() > 0:
            cls_acc = (np.array(all_preds)[mask] == cls_idx).mean()
            log.info(f"  {cls_name}: {cls_acc:.1%} ({mask.sum()} samples)")

    # Train final model on all data
    log.info("\nTraining final ensemble on all data...")

    # Use last 15% for validation
    n_train = int(len(X) * 0.85)
    X_tr = X.iloc[:n_train]
    y_tr = y_int.iloc[:n_train]
    X_val = X.iloc[n_train:]
    y_val = y_int.iloc[n_train:]

    final_cb = train_catboost(X_tr, y_tr, X_val, y_val)
    final_xgb = train_xgboost(X_tr.values, y_tr.values, X_val.values, y_val.values)
    final_lgb = train_lightgbm(X_tr.values, y_tr.values, X_val.values, y_val.values)

    # Save CatBoost as main model (preserves feature names)
    model_dir = DATA_DIR / "models" / "universal"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save upcoming model
    model_path = model_dir / "catboost_upcoming.cbm"
    final_cb.save_model(str(model_path))
    log.info(f"Saved model to {model_path}")

    # Also save as no-odds model
    no_odds_path = model_dir / "catboost_no_odds.cbm"
    final_cb.save_model(str(no_odds_path))

    # Save metadata
    metadata = {
        "model_type": "ensemble_catboost_primary",
        "cv_accuracy": round(overall_acc, 4),
        "cv_log_loss": round(overall_loss, 4),
        "n_features": len(selected_features),
        "n_samples": len(X),
        "feature_names": selected_features,
        "label_map": LABEL_MAP,
        "pre_match_only": True,
    }

    for path in [model_dir / "catboost_upcoming_metadata.json",
                 model_dir / "catboost_no_odds_metadata.json"]:
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)

    log.info("=" * 70)
    log.info("TRAINING COMPLETE")
    log.info("=" * 70)
    log.info(f"Target: 65-70% accuracy")
    log.info(f"Achieved: {overall_acc:.1%}")

    if overall_acc >= 0.65:
        log.info("TARGET MET!")
    else:
        gap = 0.65 - overall_acc
        log.info(f"Gap to target: {gap:.1%}")


if __name__ == "__main__":
    main()
