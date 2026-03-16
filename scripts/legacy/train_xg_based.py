#!/usr/bin/env python3
"""xG-Based Prediction - Predict expected goals, then derive outcome.

The idea:
1. Predict home expected goals (continuous)
2. Predict away expected goals (continuous)
3. Use (home_xG - away_xG) to determine winner probability

This is closer to how bookmakers actually work - they estimate
goal expectations and derive win probabilities from Poisson distributions.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

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
    "home_xg_attack_strength", "home_xg_defense_strength",
    "away_xg_attack_strength", "away_xg_defense_strength",
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
    "attack_strength_diff", "defense_strength_diff",
]


def poisson_win_prob(home_xg: float, away_xg: float, max_goals: int = 10) -> dict:
    """Calculate win probabilities from expected goals using Poisson distribution."""
    from scipy.stats import poisson

    # Clip xG to reasonable range
    home_xg = max(0.3, min(4.0, home_xg))
    away_xg = max(0.3, min(4.0, away_xg))

    # Calculate probabilities
    home_probs = [poisson.pmf(g, home_xg) for g in range(max_goals)]
    away_probs = [poisson.pmf(g, away_xg) for g in range(max_goals)]

    prob_home_win = 0
    prob_draw = 0
    prob_away_win = 0

    for h_goals in range(max_goals):
        for a_goals in range(max_goals):
            prob = home_probs[h_goals] * away_probs[a_goals]
            if h_goals > a_goals:
                prob_home_win += prob
            elif h_goals == a_goals:
                prob_draw += prob
            else:
                prob_away_win += prob

    # Normalize
    total = prob_home_win + prob_draw + prob_away_win
    return {
        "H": prob_home_win / total,
        "D": prob_draw / total,
        "A": prob_away_win / total,
    }


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


def main():
    log.info("=" * 70)
    log.info("xG-BASED PREDICTION - PREDICT GOALS, DERIVE OUTCOMES")
    log.info("=" * 70)

    # Load features
    feature_path = DATA_DIR / "features" / "features.parquet"
    df = pd.read_parquet(feature_path)
    df = df.dropna(subset=["home_score", "away_score"])
    log.info(f"Loaded {len(df)} matches")

    # Get available features
    available_features = [f for f in PRE_MATCH_FEATURES if f in df.columns]
    log.info(f"Using {len(available_features)} features")

    X = df[available_features].fillna(0)

    # Create targets
    y_home_goals = df["home_score"]
    y_away_goals = df["away_score"]
    y_result = df.apply(
        lambda r: "H" if r["home_score"] > r["away_score"]
        else ("A" if r["away_score"] > r["home_score"] else "D"),
        axis=1
    )

    log.info(f"\nGoal distributions:")
    log.info(f"  Home goals mean: {y_home_goals.mean():.2f}")
    log.info(f"  Away goals mean: {y_away_goals.mean():.2f}")

    # Time series splits
    splits = time_series_split(df, n_splits=5)

    # ==========================================
    # Method 1: Direct classification
    # ==========================================
    log.info("\n" + "=" * 50)
    log.info("METHOD 1: Direct H/D/A Classification")
    log.info("=" * 50)

    from catboost import CatBoostClassifier

    direct_preds = []
    direct_true = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train = X.loc[train_idx]
        y_train = y_result.loc[train_idx].map(LABEL_MAP)
        X_test = X.loc[test_idx]
        y_test = y_result.loc[test_idx].map(LABEL_MAP)

        n_train = int(len(X_train) * 0.85)

        model = CatBoostClassifier(
            iterations=1500, learning_rate=0.02, depth=6,
            auto_class_weights="SqrtBalanced",
            early_stopping_rounds=100, verbose=0, random_seed=42
        )
        model.fit(
            X_train.iloc[:n_train], y_train.iloc[:n_train],
            eval_set=(X_train.iloc[n_train:], y_train.iloc[n_train:]),
            verbose=False
        )

        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        log.info(f"  Fold {fold_idx+1}: {acc:.1%}")

        direct_preds.extend(pred.tolist())
        direct_true.extend(y_test.tolist())

    direct_acc = accuracy_score(direct_true, direct_preds)
    log.info(f"  Direct classification accuracy: {direct_acc:.1%}")

    # ==========================================
    # Method 2: xG regression + Poisson
    # ==========================================
    log.info("\n" + "=" * 50)
    log.info("METHOD 2: xG Regression + Poisson Distribution")
    log.info("=" * 50)

    from catboost import CatBoostRegressor

    xg_preds = []
    xg_true = []
    xg_probs = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train = X.loc[train_idx]
        X_test = X.loc[test_idx]

        n_train = int(len(X_train) * 0.85)

        # Train home goals regressor
        home_model = CatBoostRegressor(
            iterations=1000, learning_rate=0.03, depth=5,
            early_stopping_rounds=50, verbose=0, random_seed=42
        )
        home_model.fit(
            X_train.iloc[:n_train], y_home_goals.loc[train_idx].iloc[:n_train],
            eval_set=(X_train.iloc[n_train:], y_home_goals.loc[train_idx].iloc[n_train:]),
            verbose=False
        )

        # Train away goals regressor
        away_model = CatBoostRegressor(
            iterations=1000, learning_rate=0.03, depth=5,
            early_stopping_rounds=50, verbose=0, random_seed=42
        )
        away_model.fit(
            X_train.iloc[:n_train], y_away_goals.loc[train_idx].iloc[:n_train],
            eval_set=(X_train.iloc[n_train:], y_away_goals.loc[train_idx].iloc[n_train:]),
            verbose=False
        )

        # Predict xG
        pred_home_xg = home_model.predict(X_test)
        pred_away_xg = away_model.predict(X_test)

        # Convert to win probabilities using Poisson
        fold_preds = []
        fold_probs = []
        y_test = y_result.loc[test_idx]

        for h_xg, a_xg in zip(pred_home_xg, pred_away_xg):
            probs = poisson_win_prob(h_xg, a_xg)
            fold_probs.append([probs["H"], probs["D"], probs["A"]])

            # Predict most likely outcome
            if probs["H"] >= probs["D"] and probs["H"] >= probs["A"]:
                pred = LABEL_MAP["H"]
            elif probs["D"] >= probs["A"]:
                pred = LABEL_MAP["D"]
            else:
                pred = LABEL_MAP["A"]
            fold_preds.append(pred)

        acc = accuracy_score(y_test.map(LABEL_MAP), fold_preds)
        log.info(f"  Fold {fold_idx+1}: {acc:.1%}")

        # Log xG accuracy
        home_mae = mean_absolute_error(y_home_goals.loc[test_idx], pred_home_xg)
        away_mae = mean_absolute_error(y_away_goals.loc[test_idx], pred_away_xg)
        log.info(f"    Home xG MAE: {home_mae:.2f}, Away xG MAE: {away_mae:.2f}")

        xg_preds.extend(fold_preds)
        xg_true.extend(y_test.map(LABEL_MAP).tolist())
        xg_probs.extend(fold_probs)

    xg_acc = accuracy_score(xg_true, xg_preds)
    log.info(f"  xG-based accuracy: {xg_acc:.1%}")

    # ==========================================
    # Method 3: Combined (xG features + classifier)
    # ==========================================
    log.info("\n" + "=" * 50)
    log.info("METHOD 3: Combined (xG predictions as features)")
    log.info("=" * 50)

    # First, train xG models on all data
    n_all = int(len(X) * 0.85)

    home_model_full = CatBoostRegressor(
        iterations=1000, learning_rate=0.03, depth=5,
        early_stopping_rounds=50, verbose=0, random_seed=42
    )
    home_model_full.fit(
        X.iloc[:n_all], y_home_goals.iloc[:n_all],
        eval_set=(X.iloc[n_all:], y_home_goals.iloc[n_all:]),
        verbose=False
    )

    away_model_full = CatBoostRegressor(
        iterations=1000, learning_rate=0.03, depth=5,
        early_stopping_rounds=50, verbose=0, random_seed=42
    )
    away_model_full.fit(
        X.iloc[:n_all], y_away_goals.iloc[:n_all],
        eval_set=(X.iloc[n_all:], y_away_goals.iloc[n_all:]),
        verbose=False
    )

    # Add xG predictions as features
    X_combined = X.copy()
    X_combined["pred_home_xg"] = home_model_full.predict(X)
    X_combined["pred_away_xg"] = away_model_full.predict(X)
    X_combined["pred_xg_diff"] = X_combined["pred_home_xg"] - X_combined["pred_away_xg"]
    X_combined["pred_total_xg"] = X_combined["pred_home_xg"] + X_combined["pred_away_xg"]

    combined_preds = []
    combined_true = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train = X_combined.loc[train_idx]
        y_train = y_result.loc[train_idx].map(LABEL_MAP)
        X_test = X_combined.loc[test_idx]
        y_test = y_result.loc[test_idx].map(LABEL_MAP)

        n_train = int(len(X_train) * 0.85)

        model = CatBoostClassifier(
            iterations=1500, learning_rate=0.02, depth=6,
            auto_class_weights="SqrtBalanced",
            early_stopping_rounds=100, verbose=0, random_seed=42
        )
        model.fit(
            X_train.iloc[:n_train], y_train.iloc[:n_train],
            eval_set=(X_train.iloc[n_train:], y_train.iloc[n_train:]),
            verbose=False
        )

        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        log.info(f"  Fold {fold_idx+1}: {acc:.1%}")

        combined_preds.extend(pred.tolist())
        combined_true.extend(y_test.tolist())

    combined_acc = accuracy_score(combined_true, combined_preds)
    log.info(f"  Combined accuracy: {combined_acc:.1%}")

    # ==========================================
    # Summary
    # ==========================================
    log.info("\n" + "=" * 70)
    log.info("COMPARISON OF METHODS")
    log.info("=" * 70)
    log.info(f"  Method 1 - Direct classification: {direct_acc:.1%}")
    log.info(f"  Method 2 - xG + Poisson:          {xg_acc:.1%}")
    log.info(f"  Method 3 - Combined:              {combined_acc:.1%}")

    best_method = max([
        ("Direct", direct_acc),
        ("xG+Poisson", xg_acc),
        ("Combined", combined_acc)
    ], key=lambda x: x[1])

    log.info(f"\nBest method: {best_method[0]} ({best_method[1]:.1%})")

    # Save the best model
    model_dir = DATA_DIR / "models" / "universal"

    if best_method[0] == "Combined":
        # Save combined model
        final_model = CatBoostClassifier(
            iterations=1500, learning_rate=0.02, depth=6,
            auto_class_weights="SqrtBalanced",
            early_stopping_rounds=100, verbose=0, random_seed=42
        )
        final_model.fit(
            X_combined.iloc[:n_all], y_result.iloc[:n_all].map(LABEL_MAP),
            eval_set=(X_combined.iloc[n_all:], y_result.iloc[n_all:].map(LABEL_MAP)),
            verbose=False
        )

        final_model.save_model(str(model_dir / "catboost_upcoming.cbm"))
        final_model.save_model(str(model_dir / "catboost_no_odds.cbm"))

        # Also save the xG models
        home_model_full.save_model(str(model_dir / "xg_home.cbm"))
        away_model_full.save_model(str(model_dir / "xg_away.cbm"))

        metadata = {
            "model_type": "xg_combined",
            "cv_accuracy": round(combined_acc, 4),
            "n_features": len(X_combined.columns),
            "feature_names": list(X_combined.columns),
            "label_map": LABEL_MAP,
            "uses_xg_predictions": True,
        }
    else:
        metadata = {
            "model_type": "direct",
            "cv_accuracy": round(direct_acc, 4),
        }

    for path in [model_dir / "catboost_upcoming_metadata.json",
                 model_dir / "catboost_no_odds_metadata.json"]:
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)

    log.info(f"\nSaved model to {model_dir}")


if __name__ == "__main__":
    main()
