#!/usr/bin/env python3
"""Optimized training using only truly pre-match features + odds.

This creates two models:
1. catboost_latest.cbm - Uses odds (for evaluating past matches & finding value bets)
2. catboost_upcoming.cbm - No odds (for predicting future matches)
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


# Core pre-match features that are truly available before kickoff
PRE_MATCH_FEATURES = [
    # Elo ratings
    "home_elo", "away_elo", "elo_diff",

    # Team strength ratings
    "home_attack_strength", "home_defense_strength",
    "away_attack_strength", "away_defense_strength",
    "home_xg_attack_strength", "home_xg_defense_strength",
    "away_xg_attack_strength", "away_xg_defense_strength",
    "attack_strength_diff", "defense_strength_diff",

    # Form (rolling stats)
    "home_form_points_3", "home_form_points_5",
    "away_form_points_3", "away_form_points_5",
    "home_form_overperformance", "away_form_overperformance",

    # Rolling goals
    "home_roll_3_goals_scored", "home_roll_3_goals_conceded",
    "away_roll_3_goals_scored", "away_roll_3_goals_conceded",
    "home_roll_5_goals_scored", "home_roll_5_goals_conceded",
    "away_roll_5_goals_scored", "away_roll_5_goals_conceded",

    # Rolling xG
    "home_roll_xg_5", "home_roll_xga_5",
    "away_roll_xg_5", "away_roll_xga_5",

    # Rolling points
    "home_roll_3_points", "away_roll_3_points",
    "home_roll_5_points", "away_roll_5_points",
    "home_roll_3_win_rate", "away_roll_3_win_rate",
    "home_roll_5_win_rate", "away_roll_5_win_rate",

    # League position
    "league_position_diff",

    # Motivation factors
    "home_in_relegation_zone", "away_in_relegation_zone",
    "home_in_title_race", "away_in_title_race",
    "home_in_cl_zone", "away_in_cl_zone",
    "home_in_el_zone", "away_in_el_zone",
    "home_relegation_gap", "away_relegation_gap",

    # H2H
    "h2h_matches_played", "h2h_home_wins", "h2h_away_wins", "h2h_draws",

    # Venue
    "home_stadium_capacity", "travel_distance_km", "altitude_diff",
    "long_travel", "altitude_advantage",

    # Rest days
    "home_rest_days", "away_rest_days",
    "rest_days_diff",

    # Momentum
    "momentum_home_win_streak", "momentum_away_win_streak",
    "momentum_home_unbeaten_streak", "momentum_away_unbeaten_streak",

    # Match context
    "matchup_competitiveness",

    # League averages
    "league_home_win_rate", "league_draw_rate", "league_away_win_rate",
]

# Odds features (only for odds-based model)
ODDS_FEATURES = [
    "odds_B365H", "odds_B365D", "odds_B365A",
    "odds_PSH", "odds_PSD", "odds_PSA",
    "pinnacle_home_prob", "pinnacle_draw_prob", "pinnacle_away_prob",
]


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


def train_model(X, y, use_odds: bool = True):
    """Train CatBoost model with proper validation."""
    from catboost import CatBoostClassifier

    y_int = y.map(LABEL_MAP)

    # Split for validation
    n_train = int(len(X) * 0.85)
    X_tr = X.iloc[:n_train]
    y_tr = y_int.iloc[:n_train]
    X_val = X.iloc[n_train:]
    y_val = y_int.iloc[n_train:]

    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.02,
        depth=6,
        l2_leaf_reg=3,
        auto_class_weights="SqrtBalanced",
        early_stopping_rounds=100,
        verbose=0,
        random_seed=42,
    )

    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)

    return model


def evaluate_model(model, X, y):
    """Evaluate model and return metrics."""
    y_int = y.map(LABEL_MAP)
    proba = model.predict_proba(X)
    pred = np.argmax(proba, axis=1)

    acc = accuracy_score(y_int, pred)
    loss = log_loss(y_int, proba)

    return acc, loss, pred, proba


def main():
    log.info("=" * 70)
    log.info("OPTIMIZED TRAINING - SEPARATE MODELS FOR ODDS/NO-ODDS")
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

    # Get available features
    available_pre_match = [f for f in PRE_MATCH_FEATURES if f in df.columns]
    available_odds = [f for f in ODDS_FEATURES if f in df.columns]

    log.info(f"Available pre-match features: {len(available_pre_match)}")
    log.info(f"Available odds features: {len(available_odds)}")

    # ===========================================
    # MODEL 1: With Odds (for evaluation)
    # ===========================================
    log.info("\n" + "=" * 50)
    log.info("TRAINING MODEL WITH ODDS")
    log.info("=" * 50)

    features_with_odds = available_pre_match + available_odds
    X_odds = df[features_with_odds].fillna(0)
    y = df["result"]

    # Time-series CV
    splits = time_series_split(df, n_splits=5)

    all_preds = []
    all_true = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train = X_odds.loc[train_idx]
        y_train = y.loc[train_idx]
        X_test = X_odds.loc[test_idx]
        y_test = y.loc[test_idx]

        model = train_model(X_train, y_train, use_odds=True)
        acc, loss, pred, proba = evaluate_model(model, X_test, y_test)

        log.info(f"Fold {fold_idx+1}: {acc:.1%} accuracy")

        all_preds.extend(pred.tolist())
        all_true.extend(y.loc[test_idx].map(LABEL_MAP).tolist())

    odds_acc = accuracy_score(all_true, all_preds)
    log.info(f"\nOdds Model CV Accuracy: {odds_acc:.1%}")

    # Train final odds model
    final_odds_model = train_model(X_odds, y, use_odds=True)

    # ===========================================
    # MODEL 2: Without Odds (for predictions)
    # ===========================================
    log.info("\n" + "=" * 50)
    log.info("TRAINING MODEL WITHOUT ODDS")
    log.info("=" * 50)

    X_no_odds = df[available_pre_match].fillna(0)

    all_preds = []
    all_true = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train = X_no_odds.loc[train_idx]
        y_train = y.loc[train_idx]
        X_test = X_no_odds.loc[test_idx]
        y_test = y.loc[test_idx]

        model = train_model(X_train, y_train, use_odds=False)
        acc, loss, pred, proba = evaluate_model(model, X_test, y_test)

        log.info(f"Fold {fold_idx+1}: {acc:.1%} accuracy")

        all_preds.extend(pred.tolist())
        all_true.extend(y.loc[test_idx].map(LABEL_MAP).tolist())

    no_odds_acc = accuracy_score(all_true, all_preds)
    log.info(f"\nNo-Odds Model CV Accuracy: {no_odds_acc:.1%}")

    # Train final no-odds model
    final_no_odds_model = train_model(X_no_odds, y, use_odds=False)

    # ===========================================
    # Save models
    # ===========================================
    model_dir = DATA_DIR / "models" / "universal"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save odds model
    odds_path = model_dir / "catboost_latest.cbm"
    final_odds_model.save_model(str(odds_path))
    log.info(f"Saved odds model to {odds_path}")

    odds_meta = {
        "model_type": "catboost_with_odds",
        "cv_accuracy": round(odds_acc, 4),
        "n_features": len(features_with_odds),
        "feature_names": features_with_odds,
        "label_map": LABEL_MAP,
        "includes_odds": True,
    }
    with open(model_dir / "catboost_metadata.json", "w") as f:
        json.dump(odds_meta, f, indent=2)

    # Save no-odds model
    no_odds_path = model_dir / "catboost_upcoming.cbm"
    final_no_odds_model.save_model(str(no_odds_path))
    final_no_odds_model.save_model(str(model_dir / "catboost_no_odds.cbm"))
    log.info(f"Saved no-odds model to {no_odds_path}")

    no_odds_meta = {
        "model_type": "catboost_no_odds",
        "cv_accuracy": round(no_odds_acc, 4),
        "n_features": len(available_pre_match),
        "feature_names": available_pre_match,
        "label_map": LABEL_MAP,
        "includes_odds": False,
    }
    for path in [model_dir / "catboost_upcoming_metadata.json",
                 model_dir / "catboost_no_odds_metadata.json"]:
        with open(path, "w") as f:
            json.dump(no_odds_meta, f, indent=2)

    # ===========================================
    # Summary
    # ===========================================
    log.info("\n" + "=" * 70)
    log.info("TRAINING COMPLETE - SUMMARY")
    log.info("=" * 70)
    log.info(f"Odds Model:    {odds_acc:.1%} accuracy ({len(features_with_odds)} features)")
    log.info(f"No-Odds Model: {no_odds_acc:.1%} accuracy ({len(available_pre_match)} features)")
    log.info("")
    log.info("Use cases:")
    log.info("  - catboost_latest.cbm: Evaluate past matches, find value bets")
    log.info("  - catboost_upcoming.cbm: Predict future matches")
    log.info("")

    if odds_acc >= 0.65:
        log.info("TARGET MET: Odds model achieves 65%+")
    elif no_odds_acc >= 0.53:
        log.info("No-odds model beats random (33%) by 20+ points")

    log.info("")
    log.info("NOTE: 65-70% accuracy requires betting odds or insider information.")
    log.info("Without odds, 50-55% is typical for football prediction models.")


if __name__ == "__main__":
    main()
