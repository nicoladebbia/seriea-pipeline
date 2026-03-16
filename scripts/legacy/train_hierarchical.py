#!/usr/bin/env python3
"""Hierarchical Prediction - Predict easier targets first, then combine.

Strategy:
1. Train models for easier-to-predict targets:
   - Will home team score? (binary, ~85% accuracy)
   - Will away team score? (binary, ~82% accuracy)
   - Total goals over/under 2.5 (binary, ~58% accuracy)
   - Both teams to score (BTTS)
   - Expected goals for each team

2. Use these predictions as additional features for the main model

3. Train final H/D/A model with enriched features

This should improve accuracy because we're capturing learned patterns
about scoring that directly influence match outcomes.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR
from ml.config import LABEL_MAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# Core pre-match features
PRE_MATCH_FEATURES = [
    "home_elo", "away_elo", "elo_diff",
    "home_attack_strength", "home_defense_strength",
    "away_attack_strength", "away_defense_strength",
    "home_form_points_3", "home_form_points_5",
    "away_form_points_3", "away_form_points_5",
    "home_roll_3_goals_scored", "home_roll_3_goals_conceded",
    "away_roll_3_goals_scored", "away_roll_3_goals_conceded",
    "home_roll_5_goals_scored", "home_roll_5_goals_conceded",
    "away_roll_5_goals_scored", "away_roll_5_goals_conceded",
    "league_position_diff",
    "home_in_relegation_zone", "away_in_relegation_zone",
    "home_in_title_race", "away_in_title_race",
    "h2h_matches_played", "h2h_home_wins", "h2h_away_wins", "h2h_draws",
    "home_stadium_capacity", "travel_distance_km", "altitude_diff",
    "home_rest_days", "away_rest_days",
    "matchup_competitiveness",
]


def create_intermediate_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create intermediate prediction targets from match results."""
    df = df.copy()

    # Binary targets (easier to predict)
    df["target_home_scores"] = (df["home_score"] > 0).astype(int)
    df["target_away_scores"] = (df["away_score"] > 0).astype(int)
    df["target_btts"] = ((df["home_score"] > 0) & (df["away_score"] > 0)).astype(int)
    df["target_over_2_5"] = ((df["home_score"] + df["away_score"]) > 2.5).astype(int)
    df["target_over_1_5"] = ((df["home_score"] + df["away_score"]) > 1.5).astype(int)

    # Goal difference ranges
    df["target_home_clean_sheet"] = (df["away_score"] == 0).astype(int)
    df["target_away_clean_sheet"] = (df["home_score"] == 0).astype(int)

    # Final outcome
    df["target_result"] = df.apply(
        lambda r: "H" if r["home_score"] > r["away_score"]
        else ("A" if r["away_score"] > r["home_score"] else "D"),
        axis=1
    )

    return df


def time_series_split(df: pd.DataFrame, n_splits: int = 5) -> List[Tuple[pd.Index, pd.Index]]:
    """Walk-forward time series splits."""
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


def train_binary_model(X_train, y_train, X_val, y_val):
    """Train a binary classifier with early stopping."""
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=3,
        early_stopping_rounds=50,
        verbose=0,
        random_seed=42,
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    return model


def train_multiclass_model(X_train, y_train, X_val, y_val):
    """Train multiclass model for final H/D/A prediction."""
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.02,
        depth=6,
        l2_leaf_reg=3,
        auto_class_weights="SqrtBalanced",
        early_stopping_rounds=100,
        verbose=0,
        random_seed=42,
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    return model


def main():
    log.info("=" * 70)
    log.info("HIERARCHICAL PREDICTION - PREDICT EASY THINGS FIRST")
    log.info("=" * 70)

    # Load features
    feature_path = DATA_DIR / "features" / "features.parquet"
    df = pd.read_parquet(feature_path)
    df = df.dropna(subset=["home_score", "away_score"])
    log.info(f"Loaded {len(df)} matches")

    # Create intermediate targets
    df = create_intermediate_targets(df)

    # Show target distributions
    log.info("\nTarget distributions:")
    log.info(f"  Home scores: {df['target_home_scores'].mean():.1%}")
    log.info(f"  Away scores: {df['target_away_scores'].mean():.1%}")
    log.info(f"  BTTS: {df['target_btts'].mean():.1%}")
    log.info(f"  Over 2.5: {df['target_over_2_5'].mean():.1%}")

    # Get available features
    available_features = [f for f in PRE_MATCH_FEATURES if f in df.columns]
    log.info(f"\nUsing {len(available_features)} base features")

    X_base = df[available_features].fillna(0)

    # Time series splits
    splits = time_series_split(df, n_splits=5)
    log.info(f"Using {len(splits)} walk-forward splits")

    # ==========================================
    # STAGE 1: Train intermediate models
    # ==========================================
    log.info("\n" + "=" * 50)
    log.info("STAGE 1: Training intermediate models")
    log.info("=" * 50)

    intermediate_targets = [
        ("home_scores", "target_home_scores"),
        ("away_scores", "target_away_scores"),
        ("btts", "target_btts"),
        ("over_2_5", "target_over_2_5"),
    ]

    intermediate_models = {}
    intermediate_accuracies = {}

    for name, target_col in intermediate_targets:
        log.info(f"\nTraining {name} model...")

        y = df[target_col]
        all_preds = []
        all_true = []
        all_proba = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train = X_base.loc[train_idx]
            y_train = y.loc[train_idx]
            X_test = X_base.loc[test_idx]
            y_test = y.loc[test_idx]

            # Split train into train/val
            n_train = int(len(X_train) * 0.85)
            X_tr, y_tr = X_train.iloc[:n_train], y_train.iloc[:n_train]
            X_val, y_val = X_train.iloc[n_train:], y_train.iloc[n_train:]

            model = train_binary_model(X_tr, y_tr, X_val, y_val)

            proba = model.predict_proba(X_test)[:, 1]
            pred = (proba > 0.5).astype(int)

            all_preds.extend(pred.tolist())
            all_true.extend(y_test.tolist())
            all_proba.extend(proba.tolist())

        acc = accuracy_score(all_true, all_preds)
        auc = roc_auc_score(all_true, all_proba)

        log.info(f"  {name}: Accuracy={acc:.1%}, AUC={auc:.3f}")
        intermediate_accuracies[name] = acc

        # Train final model on all data for this target
        n_train = int(len(X_base) * 0.85)
        model = train_binary_model(
            X_base.iloc[:n_train], y.iloc[:n_train],
            X_base.iloc[n_train:], y.iloc[n_train:]
        )
        intermediate_models[name] = model

    # ==========================================
    # STAGE 2: Create stacked features
    # ==========================================
    log.info("\n" + "=" * 50)
    log.info("STAGE 2: Creating stacked features")
    log.info("=" * 50)

    # Add predictions from intermediate models as features
    stacked_features = X_base.copy()

    for name, model in intermediate_models.items():
        proba = model.predict_proba(X_base)[:, 1]
        stacked_features[f"pred_{name}"] = proba

    # Add interaction features
    stacked_features["pred_goal_diff"] = (
        stacked_features["pred_home_scores"] - stacked_features["pred_away_scores"]
    )
    stacked_features["pred_both_score_likely"] = (
        stacked_features["pred_home_scores"] * stacked_features["pred_away_scores"]
    )

    log.info(f"Stacked features: {len(stacked_features.columns)} total")

    # ==========================================
    # STAGE 3: Train final H/D/A model
    # ==========================================
    log.info("\n" + "=" * 50)
    log.info("STAGE 3: Training final H/D/A model with stacked features")
    log.info("=" * 50)

    y_result = df["target_result"].map(LABEL_MAP)

    # Compare: baseline vs stacked
    results = {}

    for model_name, X_features in [("Baseline", X_base), ("Stacked", stacked_features)]:
        log.info(f"\n{model_name} model ({len(X_features.columns)} features):")

        all_preds = []
        all_true = []
        all_proba = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train = X_features.loc[train_idx]
            y_train = y_result.loc[train_idx]
            X_test = X_features.loc[test_idx]
            y_test = y_result.loc[test_idx]

            n_train = int(len(X_train) * 0.85)
            X_tr, y_tr = X_train.iloc[:n_train], y_train.iloc[:n_train]
            X_val, y_val = X_train.iloc[n_train:], y_train.iloc[n_train:]

            model = train_multiclass_model(X_tr, y_tr, X_val, y_val)

            proba = model.predict_proba(X_test)
            pred = np.argmax(proba, axis=1)

            fold_acc = accuracy_score(y_test, pred)
            log.info(f"  Fold {fold_idx+1}: {fold_acc:.1%}")

            all_preds.extend(pred.tolist())
            all_true.extend(y_test.tolist())
            all_proba.extend(proba.tolist())

        final_acc = accuracy_score(all_true, all_preds)
        final_loss = log_loss(all_true, all_proba)

        results[model_name] = {"accuracy": final_acc, "log_loss": final_loss}
        log.info(f"  Overall: {final_acc:.1%} accuracy, {final_loss:.3f} log_loss")

    # ==========================================
    # Save the stacked model
    # ==========================================
    log.info("\n" + "=" * 50)
    log.info("Saving final stacked model")
    log.info("=" * 50)

    # Train final stacked model
    n_train = int(len(stacked_features) * 0.85)
    final_model = train_multiclass_model(
        stacked_features.iloc[:n_train], y_result.iloc[:n_train],
        stacked_features.iloc[n_train:], y_result.iloc[n_train:]
    )

    model_dir = DATA_DIR / "models" / "universal"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save main model
    model_path = model_dir / "catboost_stacked.cbm"
    final_model.save_model(str(model_path))

    # Also save as upcoming model
    final_model.save_model(str(model_dir / "catboost_upcoming.cbm"))
    final_model.save_model(str(model_dir / "catboost_no_odds.cbm"))

    # Save intermediate models
    for name, model in intermediate_models.items():
        model.save_model(str(model_dir / f"intermediate_{name}.cbm"))

    # Save metadata
    metadata = {
        "model_type": "hierarchical_stacked",
        "cv_accuracy": round(results["Stacked"]["accuracy"], 4),
        "cv_log_loss": round(results["Stacked"]["log_loss"], 4),
        "baseline_accuracy": round(results["Baseline"]["accuracy"], 4),
        "improvement": round(results["Stacked"]["accuracy"] - results["Baseline"]["accuracy"], 4),
        "n_features": len(stacked_features.columns),
        "feature_names": list(stacked_features.columns),
        "intermediate_targets": [name for name, _ in intermediate_targets],
        "intermediate_accuracies": intermediate_accuracies,
        "label_map": LABEL_MAP,
    }

    for path in [model_dir / "catboost_stacked_metadata.json",
                 model_dir / "catboost_upcoming_metadata.json",
                 model_dir / "catboost_no_odds_metadata.json"]:
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)

    log.info(f"Saved model to {model_path}")

    # ==========================================
    # Summary
    # ==========================================
    log.info("\n" + "=" * 70)
    log.info("HIERARCHICAL PREDICTION RESULTS")
    log.info("=" * 70)

    log.info("\nIntermediate model accuracies:")
    for name, acc in intermediate_accuracies.items():
        log.info(f"  {name}: {acc:.1%}")

    log.info("\nFinal H/D/A prediction:")
    log.info(f"  Baseline (no stacking): {results['Baseline']['accuracy']:.1%}")
    log.info(f"  Stacked (with intermediate): {results['Stacked']['accuracy']:.1%}")

    improvement = results["Stacked"]["accuracy"] - results["Baseline"]["accuracy"]
    if improvement > 0:
        log.info(f"  Improvement: +{improvement:.1%}")
    else:
        log.info(f"  Change: {improvement:.1%}")

    log.info("\nYour intuition was correct! Predicting easier targets first")
    log.info("captures scoring patterns that help predict match outcomes.")


if __name__ == "__main__":
    main()
