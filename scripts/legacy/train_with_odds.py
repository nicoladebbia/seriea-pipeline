#!/usr/bin/env python3
"""Train model WITH betting odds for maximum accuracy.

This model uses betting odds (when available) to achieve the highest possible
accuracy. This is useful for:
1. Evaluating model performance on historical data
2. Finding value bets (comparing model vs bookmaker odds)
3. Setting a performance ceiling

For predicting FUTURE matches (where odds aren't available yet),
use the catboost_upcoming.cbm model instead.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR
from ml.config import LABEL_MAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# Features to exclude (post-match data only, keep odds)
EXCLUDE_PATTERNS = [
    "match_id", "home_team", "away_team", "home_score", "away_score",
    "match_date", "season", "result", "league", "referee", "matchweek",
    "_has_", "ht_score",
]

# Post-match features (actual match data, not rolling)
POST_MATCH_FEATURES = [
    "home_shots_on_target", "away_shots_on_target",
    "home_shots_on_target_count", "away_shots_on_target_count",
    "home_shots_on_target_total", "away_shots_on_target_total",
    "home_total_shots", "away_total_shots",
    "home_us_team_shots", "away_us_team_shots",
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
    "attendance",
    "home_xg", "away_xg",  # Actual match xG
]


def is_pre_match_feature(col: str) -> bool:
    """Check if a column is available pre-match (including odds)."""
    col_lower = col.lower()
    for pattern in EXCLUDE_PATTERNS:
        if pattern in col_lower:
            return False
    # Check exact column names for post-match features
    if col in POST_MATCH_FEATURES:
        return False
    return True


def time_series_split(df: pd.DataFrame, n_splits: int = 5) -> List[Tuple[pd.Index, pd.Index]]:
    """Generate walk-forward time series splits."""
    seasons = sorted(df["season"].unique())
    splits = []
    min_train = 5

    for i in range(min_train, len(seasons)):
        train_seasons = seasons[:i]
        test_season = seasons[i]

        train_idx = df[df["season"].isin(train_seasons)].index
        test_idx = df[df["season"] == test_season].index

        if len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits[-n_splits:] if len(splits) > n_splits else splits


def main():
    log.info("=" * 70)
    log.info("TRAINING WITH BETTING ODDS - TARGET 65-70%")
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

    # Get all pre-match features INCLUDING odds
    all_cols = [c for c in df.columns if df[c].dtype in ["float64", "int64", "float32", "int32"]
                and not c.startswith("_")]
    pre_match_cols = [c for c in all_cols if is_pre_match_feature(c)]

    # Check which odds columns we have
    odds_cols = [c for c in pre_match_cols if "odds" in c.lower() or "pinnacle" in c.lower()]
    log.info(f"Odds columns available: {len(odds_cols)}")
    log.info(f"Total pre-match features (with odds): {len(pre_match_cols)}")

    # Feature selection
    log.info("Selecting features by importance...")
    from xgboost import XGBClassifier

    X_full = df[pre_match_cols].fillna(0)
    y = df["result"]
    y_int = y.map(LABEL_MAP)

    selector = XGBClassifier(n_estimators=200, max_depth=6, random_state=42, verbosity=0)
    selector.fit(X_full, y_int)

    feat_imp = sorted(zip(pre_match_cols, selector.feature_importances_),
                      key=lambda x: x[1], reverse=True)
    selected_features = [f for f, _ in feat_imp[:100]]

    log.info(f"Selected {len(selected_features)} features")
    log.info(f"Top 10: {selected_features[:10]}")

    # Prepare data
    X = df[selected_features].fillna(0)

    # Time-series cross validation
    splits = time_series_split(df, n_splits=5)
    log.info(f"Using {len(splits)} walk-forward splits")

    all_preds = []
    all_true = []
    all_proba = []

    from catboost import CatBoostClassifier

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        log.info(f"\n--- Fold {fold_idx + 1}/{len(splits)} ---")

        X_train = X.loc[train_idx]
        y_train = y_int.loc[train_idx]
        X_test = X.loc[test_idx]
        y_test = y_int.loc[test_idx]

        # Split train into train/val
        n_train = int(len(X_train) * 0.85)
        X_tr = X_train.iloc[:n_train]
        y_tr = y_train.iloc[:n_train]
        X_val = X_train.iloc[n_train:]
        y_val = y_train.iloc[n_train:]

        log.info(f"  Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # Train CatBoost
        model = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.02,
            depth=6,
            l2_leaf_reg=3,
            auto_class_weights="SqrtBalanced",
            early_stopping_rounds=150,
            verbose=0,
            random_seed=42,
        )

        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)

        # Predictions
        test_proba = model.predict_proba(X_test)
        test_pred = np.argmax(test_proba, axis=1)

        fold_acc = accuracy_score(y_test, test_pred)
        fold_loss = log_loss(y_test, test_proba)
        log.info(f"  Fold accuracy: {fold_acc:.1%}, log_loss: {fold_loss:.3f}")

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
    log.info("CROSS-VALIDATION RESULTS (WITH ODDS)")
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

    # Train final model
    log.info("\nTraining final model on all data...")

    n_train = int(len(X) * 0.85)
    X_tr = X.iloc[:n_train]
    y_tr = y_int.iloc[:n_train]
    X_val = X.iloc[n_train:]
    y_val = y_int.iloc[n_train:]

    final_model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.02,
        depth=6,
        l2_leaf_reg=3,
        auto_class_weights="SqrtBalanced",
        early_stopping_rounds=150,
        verbose=0,
        random_seed=42,
    )

    final_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)

    # Save model
    model_dir = DATA_DIR / "models" / "universal"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "catboost_latest.cbm"
    final_model.save_model(str(model_path))
    log.info(f"Saved model to {model_path}")

    # Save metadata
    metadata = {
        "model_type": "catboost_with_odds",
        "cv_accuracy": round(overall_acc, 4),
        "cv_log_loss": round(overall_loss, 4),
        "n_features": len(selected_features),
        "n_samples": len(X),
        "feature_names": selected_features,
        "label_map": LABEL_MAP,
        "includes_odds": True,
    }

    metadata_path = model_dir / "catboost_metadata.json"
    with open(metadata_path, "w") as f:
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
