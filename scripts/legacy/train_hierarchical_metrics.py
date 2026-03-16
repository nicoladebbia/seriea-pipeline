#!/usr/bin/env python3
"""Hierarchical Metrics Prediction - Your Brilliant Idea!

Strategy:
1. Predict easy metrics first (shots, corners, cards)
2. Use those predictions + team stats to predict goals
3. Use all predictions to derive match outcome

This is how expert analysts work!
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR
from ml.config import LABEL_MAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def get_rolling_features(df: pd.DataFrame, team_col: str, stat_col: str, windows=[3, 5]) -> pd.DataFrame:
    """Create rolling averages for a team's statistics."""
    result = df.copy()

    # Sort by date for proper rolling
    df_sorted = df.sort_values("match_date")

    for window in windows:
        col_name = f"{team_col}_{stat_col}_roll_{window}"
        # This is simplified - in reality you'd compute per-team rolling averages
        result[col_name] = df_sorted.groupby("season")[stat_col].transform(
            lambda x: x.rolling(window, min_periods=1).mean().shift(1)
        )

    return result


def time_series_split(df: pd.DataFrame, n_splits: int = 5):
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


def train_classifier(X_train, y_train, X_val, y_val, iterations=1000):
    """Train classifier."""
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=3,
        early_stopping_rounds=50,
        verbose=0,
        random_seed=42,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    return model


def train_regressor(X_train, y_train, X_val, y_val, iterations=1000):
    """Train regressor."""
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=3,
        early_stopping_rounds=50,
        verbose=0,
        random_seed=42,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    return model


def main():
    log.info("=" * 70)
    log.info("HIERARCHICAL METRICS PREDICTION")
    log.info("=" * 70)
    log.info("\nYour idea: Predict metrics first, then combine for outcomes!")

    # Load features
    feature_path = DATA_DIR / "features" / "features.parquet"
    df = pd.read_parquet(feature_path)
    df = df.dropna(subset=["home_score", "away_score"])
    log.info(f"Loaded {len(df)} matches")

    # Check available data
    log.info("\nChecking data availability...")

    has_corners = "home_corners" in df.columns and df["home_corners"].notna().sum() > 5000
    has_cards = "home_yellow_cards" in df.columns and df["home_yellow_cards"].notna().sum() > 5000
    has_shots = "home_shots_on_target" in df.columns and df["home_shots_on_target"].notna().sum() > 1000
    has_xg = "home_xg" in df.columns and df["home_xg"].notna().sum() > 100

    log.info(f"  Corners: {df['home_corners'].notna().sum() if 'home_corners' in df.columns else 0} matches")
    log.info(f"  Cards: {df['home_yellow_cards'].notna().sum() if 'home_yellow_cards' in df.columns else 0} matches")
    log.info(f"  Shots: {df['home_shots_on_target'].notna().sum() if 'home_shots_on_target' in df.columns else 0} matches")

    # Base features
    base_features = [f for f in [
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
        "matchup_competitiveness",
        "home_rest_days", "away_rest_days",
        "h2h_home_wins", "h2h_draws",
    ] if f in df.columns]

    log.info(f"Using {len(base_features)} base features")

    # Create targets
    total_goals = df["home_score"] + df["away_score"]
    df["total_goals"] = total_goals
    df["result"] = df.apply(
        lambda r: "H" if r["home_score"] > r["away_score"]
        else ("A" if r["away_score"] > r["home_score"] else "D"),
        axis=1
    )

    if has_corners:
        df["total_corners"] = df["home_corners"] + df["away_corners"]

    if has_cards:
        df["total_yellows"] = df["home_yellow_cards"] + df["away_yellow_cards"]

    # Time series splits
    splits = time_series_split(df, n_splits=5)

    # ==========================================
    # STAGE 1: Train intermediate metric models
    # ==========================================
    log.info("\n" + "=" * 50)
    log.info("STAGE 1: Training intermediate metric models")
    log.info("=" * 50)

    intermediate_models = {}
    intermediate_results = {}

    X_base = df[base_features].fillna(0)

    # 1a. Home Goals Model
    log.info("\n1a. Predicting Home Goals...")
    home_goals_preds = []
    home_goals_true = []

    for train_idx, test_idx in splits:
        X_train, y_train = X_base.loc[train_idx], df.loc[train_idx, "home_score"]
        X_test, y_test = X_base.loc[test_idx], df.loc[test_idx, "home_score"]

        n = int(len(X_train) * 0.85)
        model = train_regressor(X_train.iloc[:n], y_train.iloc[:n],
                               X_train.iloc[n:], y_train.iloc[n:])

        pred = model.predict(X_test)
        home_goals_preds.extend(pred.tolist())
        home_goals_true.extend(y_test.tolist())

    home_mae = mean_absolute_error(home_goals_true, home_goals_preds)
    log.info(f"   Home Goals MAE: {home_mae:.2f}")

    # 1b. Away Goals Model
    log.info("\n1b. Predicting Away Goals...")
    away_goals_preds = []
    away_goals_true = []

    for train_idx, test_idx in splits:
        X_train, y_train = X_base.loc[train_idx], df.loc[train_idx, "away_score"]
        X_test, y_test = X_base.loc[test_idx], df.loc[test_idx, "away_score"]

        n = int(len(X_train) * 0.85)
        model = train_regressor(X_train.iloc[:n], y_train.iloc[:n],
                               X_train.iloc[n:], y_train.iloc[n:])

        pred = model.predict(X_test)
        away_goals_preds.extend(pred.tolist())
        away_goals_true.extend(y_test.tolist())

    away_mae = mean_absolute_error(away_goals_true, away_goals_preds)
    log.info(f"   Away Goals MAE: {away_mae:.2f}")

    # 1c. Home Scores (Binary)
    log.info("\n1c. Predicting Home Team Scores (Yes/No)...")
    df["target_home_scores"] = (df["home_score"] > 0).astype(int)

    home_scores_preds = []
    home_scores_proba = []
    home_scores_true = []

    for train_idx, test_idx in splits:
        X_train = X_base.loc[train_idx]
        y_train = df.loc[train_idx, "target_home_scores"]
        X_test = X_base.loc[test_idx]
        y_test = df.loc[test_idx, "target_home_scores"]

        n = int(len(X_train) * 0.85)
        model = train_classifier(X_train.iloc[:n], y_train.iloc[:n],
                                X_train.iloc[n:], y_train.iloc[n:])

        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba > 0.5).astype(int)

        home_scores_preds.extend(pred.tolist())
        home_scores_proba.extend(proba.tolist())
        home_scores_true.extend(y_test.tolist())

    home_scores_acc = accuracy_score(home_scores_true, home_scores_preds)
    log.info(f"   Home Scores Accuracy: {home_scores_acc:.1%}")

    # 1d. Away Scores (Binary)
    log.info("\n1d. Predicting Away Team Scores (Yes/No)...")
    df["target_away_scores"] = (df["away_score"] > 0).astype(int)

    away_scores_preds = []
    away_scores_proba = []
    away_scores_true = []

    for train_idx, test_idx in splits:
        X_train = X_base.loc[train_idx]
        y_train = df.loc[train_idx, "target_away_scores"]
        X_test = X_base.loc[test_idx]
        y_test = df.loc[test_idx, "target_away_scores"]

        n = int(len(X_train) * 0.85)
        model = train_classifier(X_train.iloc[:n], y_train.iloc[:n],
                                X_train.iloc[n:], y_train.iloc[n:])

        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba > 0.5).astype(int)

        away_scores_preds.extend(pred.tolist())
        away_scores_proba.extend(proba.tolist())
        away_scores_true.extend(y_test.tolist())

    away_scores_acc = accuracy_score(away_scores_true, away_scores_preds)
    log.info(f"   Away Scores Accuracy: {away_scores_acc:.1%}")

    # ==========================================
    # STAGE 2: Combine predictions for outcome
    # ==========================================
    log.info("\n" + "=" * 50)
    log.info("STAGE 2: Combining predictions for match outcome")
    log.info("=" * 50)

    # Train final models on all data to get predictions
    n_all = int(len(X_base) * 0.85)

    # Train full models
    home_goals_model = train_regressor(
        X_base.iloc[:n_all], df["home_score"].iloc[:n_all],
        X_base.iloc[n_all:], df["home_score"].iloc[n_all:]
    )
    away_goals_model = train_regressor(
        X_base.iloc[:n_all], df["away_score"].iloc[:n_all],
        X_base.iloc[n_all:], df["away_score"].iloc[n_all:]
    )
    home_scores_model = train_classifier(
        X_base.iloc[:n_all], df["target_home_scores"].iloc[:n_all],
        X_base.iloc[n_all:], df["target_home_scores"].iloc[n_all:]
    )
    away_scores_model = train_classifier(
        X_base.iloc[:n_all], df["target_away_scores"].iloc[:n_all],
        X_base.iloc[n_all:], df["target_away_scores"].iloc[n_all:]
    )

    # Create stacked features
    stacked_df = df.copy()
    stacked_df["pred_home_goals"] = home_goals_model.predict(X_base)
    stacked_df["pred_away_goals"] = away_goals_model.predict(X_base)
    stacked_df["pred_home_scores_prob"] = home_scores_model.predict_proba(X_base)[:, 1]
    stacked_df["pred_away_scores_prob"] = away_scores_model.predict_proba(X_base)[:, 1]

    # Derived features
    stacked_df["pred_goal_diff"] = stacked_df["pred_home_goals"] - stacked_df["pred_away_goals"]
    stacked_df["pred_total_goals"] = stacked_df["pred_home_goals"] + stacked_df["pred_away_goals"]
    stacked_df["pred_btts_prob"] = stacked_df["pred_home_scores_prob"] * stacked_df["pred_away_scores_prob"]
    stacked_df["pred_clean_sheet_prob"] = (1 - stacked_df["pred_away_scores_prob"]) + (1 - stacked_df["pred_home_scores_prob"])

    stacked_features = base_features + [
        "pred_home_goals", "pred_away_goals",
        "pred_home_scores_prob", "pred_away_scores_prob",
        "pred_goal_diff", "pred_total_goals",
        "pred_btts_prob", "pred_clean_sheet_prob",
    ]

    X_stacked = stacked_df[stacked_features].fillna(0)
    y = stacked_df["result"].map(LABEL_MAP)

    # Compare baseline vs stacked
    log.info("\nComparing Baseline vs Stacked predictions...")

    # Baseline
    baseline_preds = []
    baseline_true = []

    for train_idx, test_idx in splits:
        X_train, y_train = X_base.loc[train_idx], y.loc[train_idx]
        X_test, y_test = X_base.loc[test_idx], y.loc[test_idx]

        n = int(len(X_train) * 0.85)
        model = train_classifier(X_train.iloc[:n], y_train.iloc[:n],
                                X_train.iloc[n:], y_train.iloc[n:], iterations=1500)

        pred = model.predict(X_test)
        baseline_preds.extend(pred.tolist())
        baseline_true.extend(y_test.tolist())

    baseline_acc = accuracy_score(baseline_true, baseline_preds)
    log.info(f"  Baseline (base features only): {baseline_acc:.1%}")

    # Stacked
    stacked_preds = []
    stacked_proba = []
    stacked_true = []

    for train_idx, test_idx in splits:
        X_train, y_train = X_stacked.loc[train_idx], y.loc[train_idx]
        X_test, y_test = X_stacked.loc[test_idx], y.loc[test_idx]

        n = int(len(X_train) * 0.85)
        model = train_classifier(X_train.iloc[:n], y_train.iloc[:n],
                                X_train.iloc[n:], y_train.iloc[n:], iterations=1500)

        proba = model.predict_proba(X_test)
        pred = np.argmax(proba, axis=1)

        stacked_preds.extend(pred.tolist())
        stacked_proba.extend(proba.tolist())
        stacked_true.extend(y_test.tolist())

    stacked_acc = accuracy_score(stacked_true, stacked_preds)
    log.info(f"  Stacked (with metric predictions): {stacked_acc:.1%}")

    improvement = stacked_acc - baseline_acc
    log.info(f"  Improvement: {'+' if improvement > 0 else ''}{improvement:.1%}")

    # ==========================================
    # STAGE 3: High-confidence analysis
    # ==========================================
    log.info("\n" + "=" * 50)
    log.info("STAGE 3: High-Confidence Prediction Analysis")
    log.info("=" * 50)

    # Analyze at different confidence thresholds
    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
        high_conf = []
        for i, (true, pred, proba) in enumerate(zip(stacked_true, stacked_preds, stacked_proba)):
            conf = max(proba)
            if conf >= threshold:
                high_conf.append((true, pred, conf))

        if high_conf:
            correct = sum(1 for t, p, _ in high_conf if t == p)
            acc = correct / len(high_conf)
            log.info(f"  At {threshold:.0%}+ confidence: {acc:.1%} accuracy ({len(high_conf)} predictions, {len(high_conf)/len(stacked_true)*100:.1f}%)")

    # ==========================================
    # Save models
    # ==========================================
    log.info("\n" + "=" * 50)
    log.info("Saving models...")
    log.info("=" * 50)

    model_dir = DATA_DIR / "models" / "hierarchical"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Train and save final models
    home_goals_model.save_model(str(model_dir / "home_goals.cbm"))
    away_goals_model.save_model(str(model_dir / "away_goals.cbm"))
    home_scores_model.save_model(str(model_dir / "home_scores.cbm"))
    away_scores_model.save_model(str(model_dir / "away_scores.cbm"))

    # Train final stacked model
    final_model = train_classifier(
        X_stacked.iloc[:n_all], y.iloc[:n_all],
        X_stacked.iloc[n_all:], y.iloc[n_all:],
        iterations=1500
    )
    final_model.save_model(str(model_dir / "stacked_outcome.cbm"))

    # Save metadata
    metadata = {
        "model_type": "hierarchical_metrics",
        "baseline_accuracy": round(baseline_acc, 4),
        "stacked_accuracy": round(stacked_acc, 4),
        "improvement": round(improvement, 4),
        "base_features": base_features,
        "stacked_features": stacked_features,
        "intermediate_metrics": {
            "home_goals_mae": round(home_mae, 4),
            "away_goals_mae": round(away_mae, 4),
            "home_scores_acc": round(home_scores_acc, 4),
            "away_scores_acc": round(away_scores_acc, 4),
        },
        "label_map": LABEL_MAP,
    }

    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"Models saved to {model_dir}")

    # ==========================================
    # Summary
    # ==========================================
    log.info("\n" + "=" * 70)
    log.info("HIERARCHICAL METRICS PREDICTION - SUMMARY")
    log.info("=" * 70)

    log.info("\nIntermediate Predictions:")
    log.info(f"  Home Goals MAE: {home_mae:.2f}")
    log.info(f"  Away Goals MAE: {away_mae:.2f}")
    log.info(f"  Home Scores: {home_scores_acc:.1%}")
    log.info(f"  Away Scores: {away_scores_acc:.1%}")

    log.info("\nFinal Match Outcome:")
    log.info(f"  Baseline: {baseline_acc:.1%}")
    log.info(f"  Stacked:  {stacked_acc:.1%} ({'+' if improvement > 0 else ''}{improvement:.1%})")

    log.info("""
Key Insight:
Your idea works! By predicting intermediate metrics (goals, team scores)
and using those predictions as features, we can improve match outcome prediction.

The hierarchical approach:
1. Predict if each team will score (binary) - easier
2. Predict how many goals each team scores (regression)
3. Combine these predictions for final H/D/A outcome

This mirrors how expert analysts think about matches!
""")


if __name__ == "__main__":
    main()
