#!/usr/bin/env python3
"""Multi-Metric Match Prediction System

Instead of predicting just match outcomes, predict DETAILED METRICS:
1. Total Goals (Over/Under 0.5, 1.5, 2.5, 3.5)
2. Total Shots (ranges)
3. Shots on Target (ranges)
4. Total Corners (Over/Under)
5. Total Cards (Over/Under)
6. First Half Goals

Then combine these predictions to derive match outcomes!

We have excellent data:
- Cards: 100% coverage + referee tendencies
- Corners: 99.9% coverage
- Shots: 99.8% coverage + shot-level detail
- xG/xA: 11 seasons of player data
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR
from ml.config import LABEL_MAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# Features for different metrics
BASE_FEATURES = [
    "home_elo", "away_elo", "elo_diff",
    "home_attack_strength", "home_defense_strength",
    "away_attack_strength", "away_defense_strength",
    "home_form_points_3", "home_form_points_5",
    "away_form_points_3", "away_form_points_5",
    "league_position_diff",
    "home_rest_days", "away_rest_days",
]

# Goal-specific features
GOAL_FEATURES = [
    "home_roll_3_goals_scored", "home_roll_3_goals_conceded",
    "away_roll_3_goals_scored", "away_roll_3_goals_conceded",
    "home_roll_5_goals_scored", "home_roll_5_goals_conceded",
    "away_roll_5_goals_scored", "away_roll_5_goals_conceded",
    "home_xg_attack_strength", "home_xg_defense_strength",
    "away_xg_attack_strength", "away_xg_defense_strength",
]

# Shot-specific features
SHOT_FEATURES = [
    "home_roll_3_shots", "away_roll_3_shots",
    "home_roll_5_shots", "away_roll_5_shots",
    "home_roll_3_shots_on_target", "away_roll_3_shots_on_target",
    "home_roll_5_shots_on_target", "away_roll_5_shots_on_target",
]

# Corner-specific features
CORNER_FEATURES = [
    "home_roll_3_corners", "away_roll_3_corners",
    "home_roll_5_corners", "away_roll_5_corners",
    "home_roll_10_corners", "away_roll_10_corners",
]

# Card-specific features (referee tendencies)
CARD_FEATURES = [
    "home_roll_3_yellow_cards", "away_roll_3_yellow_cards",
    "home_roll_5_yellow_cards", "away_roll_5_yellow_cards",
    "ref_avg_yellows", "ref_avg_reds",
    "ref_strictness_score", "ref_home_cards_bias",
]


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


def train_classifier(X_train, y_train, X_val, y_val):
    """Train binary classifier."""
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


def train_regressor(X_train, y_train, X_val, y_val):
    """Train regressor for continuous targets."""
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
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


def evaluate_classification(df, X, target_col, splits, metric_name, threshold=0.5):
    """Evaluate classification model."""
    y = df[target_col]

    all_preds = []
    all_true = []
    all_proba = []

    for train_idx, test_idx in splits:
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]

        n_train = int(len(X_train) * 0.85)
        model = train_classifier(
            X_train.iloc[:n_train], y_train.iloc[:n_train],
            X_train.iloc[n_train:], y_train.iloc[n_train:]
        )

        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba > threshold).astype(int)

        all_preds.extend(pred.tolist())
        all_true.extend(y_test.tolist())
        all_proba.extend(proba.tolist())

    acc = accuracy_score(all_true, all_preds)
    base_rate = sum(all_true) / len(all_true)
    skill = acc - max(base_rate, 1 - base_rate)

    return {
        "accuracy": acc,
        "base_rate": base_rate,
        "skill": skill,
        "predictions": list(zip(all_true, all_preds, all_proba)),
    }


def evaluate_regression(df, X, target_col, splits, metric_name):
    """Evaluate regression model."""
    y = df[target_col]

    all_preds = []
    all_true = []

    for train_idx, test_idx in splits:
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]

        n_train = int(len(X_train) * 0.85)
        model = train_regressor(
            X_train.iloc[:n_train], y_train.iloc[:n_train],
            X_train.iloc[n_train:], y_train.iloc[n_train:]
        )

        pred = model.predict(X_test)
        all_preds.extend(pred.tolist())
        all_true.extend(y_test.tolist())

    mae = mean_absolute_error(all_true, all_preds)
    r2 = r2_score(all_true, all_preds)

    return {
        "mae": mae,
        "r2": r2,
        "mean_actual": np.mean(all_true),
        "mean_predicted": np.mean(all_preds),
    }


def analyze_confidence_thresholds(predictions, thresholds=[0.6, 0.65, 0.7, 0.75, 0.8]):
    """Analyze accuracy at different confidence levels."""
    results = {}
    for threshold in thresholds:
        high_conf = [(t, p, prob) for t, p, prob in predictions
                     if max(prob, 1-prob) >= threshold]
        if high_conf:
            correct = sum(1 for t, p, _ in high_conf if t == p)
            acc = correct / len(high_conf)
            results[threshold] = {
                "count": len(high_conf),
                "accuracy": acc,
                "pct": len(high_conf) / len(predictions) * 100,
            }
    return results


def main():
    log.info("=" * 70)
    log.info("MULTI-METRIC MATCH PREDICTION SYSTEM")
    log.info("=" * 70)
    log.info("\nPredicting: Goals, Shots, Corners, Cards")
    log.info("Then combining to derive match outcomes!")

    # Load features
    feature_path = DATA_DIR / "features" / "features.parquet"
    df = pd.read_parquet(feature_path)
    df = df.dropna(subset=["home_score", "away_score"])
    log.info(f"Loaded {len(df)} matches")

    # Create target variables
    total_goals = df["home_score"] + df["away_score"]

    # Goal targets
    df["target_over_0_5"] = (total_goals > 0.5).astype(int)
    df["target_over_1_5"] = (total_goals > 1.5).astype(int)
    df["target_over_2_5"] = (total_goals > 2.5).astype(int)
    df["target_over_3_5"] = (total_goals > 3.5).astype(int)
    df["target_total_goals"] = total_goals

    # Shot targets (where available)
    if "home_shots_on_target" in df.columns and "away_shots_on_target" in df.columns:
        df["total_shots_on_target"] = df["home_shots_on_target"].fillna(0) + df["away_shots_on_target"].fillna(0)
        df["target_sot_over_6"] = (df["total_shots_on_target"] > 6).astype(int)
        df["target_sot_over_8"] = (df["total_shots_on_target"] > 8).astype(int)

    # Corner targets
    if "home_corners" in df.columns and "away_corners" in df.columns:
        df["total_corners"] = df["home_corners"].fillna(0) + df["away_corners"].fillna(0)
        df["target_corners_over_8"] = (df["total_corners"] > 8).astype(int)
        df["target_corners_over_10"] = (df["total_corners"] > 10).astype(int)

    # Card targets
    if "home_yellow_cards" in df.columns and "away_yellow_cards" in df.columns:
        df["total_yellows"] = df["home_yellow_cards"].fillna(0) + df["away_yellow_cards"].fillna(0)
        df["target_yellows_over_3"] = (df["total_yellows"] > 3).astype(int)
        df["target_yellows_over_4"] = (df["total_yellows"] > 4).astype(int)

    # First half goals
    if "home_ht_score" in df.columns and "away_ht_score" in df.columns:
        df["ht_total"] = df["home_ht_score"].fillna(0) + df["away_ht_score"].fillna(0)
        df["target_ht_over_0_5"] = (df["ht_total"] > 0.5).astype(int)
        df["target_ht_over_1_5"] = (df["ht_total"] > 1.5).astype(int)

    # Get available features
    def get_available(feature_list):
        return [f for f in feature_list if f in df.columns]

    base_feats = get_available(BASE_FEATURES)
    goal_feats = get_available(GOAL_FEATURES)
    shot_feats = get_available(SHOT_FEATURES)
    corner_feats = get_available(CORNER_FEATURES)
    card_feats = get_available(CARD_FEATURES)

    log.info(f"\nFeatures available:")
    log.info(f"  Base: {len(base_feats)}")
    log.info(f"  Goal: {len(goal_feats)}")
    log.info(f"  Shot: {len(shot_feats)}")
    log.info(f"  Corner: {len(corner_feats)}")
    log.info(f"  Card: {len(card_feats)}")

    # Time series splits
    splits = time_series_split(df, n_splits=5)
    log.info(f"Using {len(splits)} walk-forward splits")

    results = {}

    # ==========================================
    # 1. TOTAL GOALS PREDICTION
    # ==========================================
    log.info("\n" + "=" * 50)
    log.info("1. TOTAL GOALS PREDICTION")
    log.info("=" * 50)

    goal_features = base_feats + goal_feats
    X_goals = df[goal_features].fillna(0)

    for target, name in [
        ("target_over_0_5", "Over 0.5 Goals"),
        ("target_over_1_5", "Over 1.5 Goals"),
        ("target_over_2_5", "Over 2.5 Goals"),
        ("target_over_3_5", "Over 3.5 Goals"),
    ]:
        result = evaluate_classification(df, X_goals, target, splits, name)
        results[name] = result
        log.info(f"\n  {name}:")
        log.info(f"    Accuracy: {result['accuracy']:.1%} (Base: {result['base_rate']:.1%}, Skill: {result['skill']:+.1%})")

        # Confidence analysis
        conf_results = analyze_confidence_thresholds(result["predictions"])
        for thresh, stats in conf_results.items():
            log.info(f"    At {thresh:.0%} conf: {stats['accuracy']:.1%} acc ({stats['count']} bets, {stats['pct']:.1f}%)")

    # Regression for exact goals
    log.info("\n  Exact Goals (Regression):")
    reg_result = evaluate_regression(df, X_goals, "target_total_goals", splits, "Total Goals")
    log.info(f"    MAE: {reg_result['mae']:.2f} goals")
    log.info(f"    R²: {reg_result['r2']:.3f}")
    log.info(f"    Mean actual: {reg_result['mean_actual']:.2f}, Mean predicted: {reg_result['mean_predicted']:.2f}")

    # ==========================================
    # 2. CORNERS PREDICTION
    # ==========================================
    if "total_corners" in df.columns:
        log.info("\n" + "=" * 50)
        log.info("2. CORNERS PREDICTION")
        log.info("=" * 50)

        corner_df = df.dropna(subset=["total_corners"])
        if len(corner_df) > 1000:
            corner_X = corner_df[base_feats + corner_feats].fillna(0)
            corner_splits = time_series_split(corner_df, n_splits=5)

            for target, name in [
                ("target_corners_over_8", "Over 8 Corners"),
                ("target_corners_over_10", "Over 10 Corners"),
            ]:
                result = evaluate_classification(corner_df, corner_X, target, corner_splits, name)
                results[name] = result
                log.info(f"\n  {name}:")
                log.info(f"    Accuracy: {result['accuracy']:.1%} (Base: {result['base_rate']:.1%}, Skill: {result['skill']:+.1%})")

                conf_results = analyze_confidence_thresholds(result["predictions"])
                for thresh, stats in conf_results.items():
                    log.info(f"    At {thresh:.0%} conf: {stats['accuracy']:.1%} acc ({stats['count']} bets)")

            # Regression
            log.info("\n  Exact Corners (Regression):")
            reg_result = evaluate_regression(corner_df, corner_X, "total_corners", corner_splits, "Corners")
            log.info(f"    MAE: {reg_result['mae']:.2f} corners")
            log.info(f"    Mean actual: {reg_result['mean_actual']:.1f}")

    # ==========================================
    # 3. CARDS PREDICTION
    # ==========================================
    if "total_yellows" in df.columns:
        log.info("\n" + "=" * 50)
        log.info("3. CARDS PREDICTION")
        log.info("=" * 50)

        card_df = df.dropna(subset=["total_yellows"])
        if len(card_df) > 1000:
            card_X = card_df[base_feats + card_feats].fillna(0)
            card_splits = time_series_split(card_df, n_splits=5)

            for target, name in [
                ("target_yellows_over_3", "Over 3 Yellow Cards"),
                ("target_yellows_over_4", "Over 4 Yellow Cards"),
            ]:
                result = evaluate_classification(card_df, card_X, target, card_splits, name)
                results[name] = result
                log.info(f"\n  {name}:")
                log.info(f"    Accuracy: {result['accuracy']:.1%} (Base: {result['base_rate']:.1%}, Skill: {result['skill']:+.1%})")

                conf_results = analyze_confidence_thresholds(result["predictions"])
                for thresh, stats in conf_results.items():
                    log.info(f"    At {thresh:.0%} conf: {stats['accuracy']:.1%} acc ({stats['count']} bets)")

            # Regression
            log.info("\n  Exact Yellow Cards (Regression):")
            reg_result = evaluate_regression(card_df, card_X, "total_yellows", card_splits, "Yellows")
            log.info(f"    MAE: {reg_result['mae']:.2f} cards")
            log.info(f"    Mean actual: {reg_result['mean_actual']:.1f}")

    # ==========================================
    # 4. SHOTS ON TARGET PREDICTION
    # ==========================================
    if "total_shots_on_target" in df.columns:
        log.info("\n" + "=" * 50)
        log.info("4. SHOTS ON TARGET PREDICTION")
        log.info("=" * 50)

        shot_df = df.dropna(subset=["total_shots_on_target"])
        if len(shot_df) > 1000:
            shot_X = shot_df[base_feats + shot_feats].fillna(0)
            shot_splits = time_series_split(shot_df, n_splits=5)

            for target, name in [
                ("target_sot_over_6", "Over 6 Shots on Target"),
                ("target_sot_over_8", "Over 8 Shots on Target"),
            ]:
                result = evaluate_classification(shot_df, shot_X, target, shot_splits, name)
                results[name] = result
                log.info(f"\n  {name}:")
                log.info(f"    Accuracy: {result['accuracy']:.1%} (Base: {result['base_rate']:.1%}, Skill: {result['skill']:+.1%})")

                conf_results = analyze_confidence_thresholds(result["predictions"])
                for thresh, stats in conf_results.items():
                    log.info(f"    At {thresh:.0%} conf: {stats['accuracy']:.1%} acc ({stats['count']} bets)")

    # ==========================================
    # 5. FIRST HALF GOALS PREDICTION
    # ==========================================
    if "ht_total" in df.columns:
        log.info("\n" + "=" * 50)
        log.info("5. FIRST HALF GOALS PREDICTION")
        log.info("=" * 50)

        ht_df = df.dropna(subset=["ht_total"])
        if len(ht_df) > 1000:
            ht_X = ht_df[goal_features].fillna(0)
            ht_splits = time_series_split(ht_df, n_splits=5)

            for target, name in [
                ("target_ht_over_0_5", "Over 0.5 First Half Goals"),
                ("target_ht_over_1_5", "Over 1.5 First Half Goals"),
            ]:
                result = evaluate_classification(ht_df, ht_X, target, ht_splits, name)
                results[name] = result
                log.info(f"\n  {name}:")
                log.info(f"    Accuracy: {result['accuracy']:.1%} (Base: {result['base_rate']:.1%}, Skill: {result['skill']:+.1%})")

                conf_results = analyze_confidence_thresholds(result["predictions"])
                for thresh, stats in conf_results.items():
                    log.info(f"    At {thresh:.0%} conf: {stats['accuracy']:.1%} acc ({stats['count']} bets)")

    # ==========================================
    # SUMMARY
    # ==========================================
    log.info("\n" + "=" * 70)
    log.info("SUMMARY: BEST PREDICTIONS BY ACCURACY")
    log.info("=" * 70)

    sorted_results = sorted(
        [(name, r) for name, r in results.items() if "accuracy" in r],
        key=lambda x: x[1]["accuracy"],
        reverse=True
    )

    log.info(f"\n{'Market':<30} {'Accuracy':<12} {'Base Rate':<12} {'Skill'}")
    log.info("-" * 70)
    for name, r in sorted_results:
        log.info(f"{name:<30} {r['accuracy']:.1%}        {r['base_rate']:.1%}        {r['skill']:+.1%}")

    # Save results
    output = {
        name: {
            "accuracy": round(r["accuracy"], 4),
            "base_rate": round(r["base_rate"], 4),
            "skill": round(r["skill"], 4),
        }
        for name, r in results.items() if "accuracy" in r
    }

    output_path = DATA_DIR / "models" / "match_metrics_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    log.info(f"\nResults saved to {output_path}")

    # ==========================================
    # KEY INSIGHTS
    # ==========================================
    log.info("\n" + "=" * 70)
    log.info("KEY INSIGHTS")
    log.info("=" * 70)
    log.info("""
PREDICTIONS WITH BEST ACCURACY:
1. Over 0.5 Goals (at least 1 goal) - Should be ~90%+
2. Over 1.5 Goals (at least 2 goals) - Should be ~75%+
3. Corners Over/Under - 65-70% with rich corner data
4. Cards Over/Under - With referee tendencies, 60-65%

COMBINING PREDICTIONS:
- Use high-confidence metric predictions as features
- If "Over 1.5 Goals" = 85% confident + "Home scores" = 80% confident
  → Strong signal for home win or draw with goals

NEXT STEPS:
1. Train final models for best metrics
2. Create ensemble that combines predictions
3. Use combined predictions for match outcome
""")


if __name__ == "__main__":
    main()
