#!/usr/bin/env python3
"""Multi-Market Prediction System

Instead of just predicting winners, predict MULTIPLE markets and
find the ones with highest confidence:

Markets:
1. Match Result (1X2) - Hardest, ~52%
2. Over/Under 2.5 - Medium, ~55%
3. Both Teams Score (BTTS) - Medium, ~55%
4. Over/Under 1.5 - Easier, ~65%
5. Home Team Scores - Easiest, ~75%
6. Away Team Scores - Easier, ~70%
7. Draw No Bet (Home or Away, excluding draws) - Medium, ~60%

The idea: Find matches where we're MOST CONFIDENT about SOMETHING.
A 75% confident "home scores" bet is better than a 52% confident "home wins".
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR
from ml.config import LABEL_MAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# Markets to predict
MARKETS = {
    "result_H": {"name": "Home Win", "type": "binary"},
    "result_D": {"name": "Draw", "type": "binary"},
    "result_A": {"name": "Away Win", "type": "binary"},
    "over_2_5": {"name": "Over 2.5 Goals", "type": "binary"},
    "under_2_5": {"name": "Under 2.5 Goals", "type": "binary"},
    "over_1_5": {"name": "Over 1.5 Goals", "type": "binary"},
    "btts_yes": {"name": "Both Teams Score - Yes", "type": "binary"},
    "btts_no": {"name": "Both Teams Score - No", "type": "binary"},
    "home_scores": {"name": "Home Team Scores", "type": "binary"},
    "away_scores": {"name": "Away Team Scores", "type": "binary"},
    "home_clean_sheet": {"name": "Home Clean Sheet", "type": "binary"},
    "away_clean_sheet": {"name": "Away Clean Sheet", "type": "binary"},
}


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
    "h2h_matches_played", "h2h_home_wins", "h2h_draws",
    "home_stadium_capacity", "travel_distance_km", "altitude_diff",
    "home_rest_days", "away_rest_days",
    "matchup_competitiveness",
]


def create_market_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create all market targets from match data."""
    df = df.copy()

    total_goals = df["home_score"] + df["away_score"]

    # Result markets
    df["target_result_H"] = (df["home_score"] > df["away_score"]).astype(int)
    df["target_result_D"] = (df["home_score"] == df["away_score"]).astype(int)
    df["target_result_A"] = (df["home_score"] < df["away_score"]).astype(int)

    # Goals markets
    df["target_over_2_5"] = (total_goals > 2.5).astype(int)
    df["target_under_2_5"] = (total_goals < 2.5).astype(int)
    df["target_over_1_5"] = (total_goals > 1.5).astype(int)

    # BTTS markets
    df["target_btts_yes"] = ((df["home_score"] > 0) & (df["away_score"] > 0)).astype(int)
    df["target_btts_no"] = ((df["home_score"] == 0) | (df["away_score"] == 0)).astype(int)

    # Team scores
    df["target_home_scores"] = (df["home_score"] > 0).astype(int)
    df["target_away_scores"] = (df["away_score"] > 0).astype(int)

    # Clean sheets
    df["target_home_clean_sheet"] = (df["away_score"] == 0).astype(int)
    df["target_away_clean_sheet"] = (df["home_score"] == 0).astype(int)

    return df


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


def train_market_model(X_train, y_train, X_val, y_val):
    """Train a binary classifier for a market."""
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        iterations=800,
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
    log.info("MULTI-MARKET PREDICTION SYSTEM")
    log.info("=" * 70)
    log.info("\nPredicting multiple markets to find highest confidence bets")

    # Load features
    feature_path = DATA_DIR / "features" / "features.parquet"
    df = pd.read_parquet(feature_path)
    df = df.dropna(subset=["home_score", "away_score"])
    log.info(f"Loaded {len(df)} matches")

    # Create targets
    df = create_market_targets(df)

    # Get available features
    available_features = [f for f in PRE_MATCH_FEATURES if f in df.columns]
    log.info(f"Using {len(available_features)} features")

    X = df[available_features].fillna(0)

    # Time series splits
    splits = time_series_split(df, n_splits=5)

    # ==========================================
    # Train and evaluate all markets
    # ==========================================
    log.info("\n" + "=" * 50)
    log.info("TRAINING MODELS FOR ALL MARKETS")
    log.info("=" * 50)

    market_results = {}
    market_models = {}

    for market_key, market_info in MARKETS.items():
        target_col = f"target_{market_key}"
        y = df[target_col]

        all_preds = []
        all_proba = []
        all_true = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train = X.loc[train_idx]
            y_train = y.loc[train_idx]
            X_test = X.loc[test_idx]
            y_test = y.loc[test_idx]

            n_train = int(len(X_train) * 0.85)
            X_tr, y_tr = X_train.iloc[:n_train], y_train.iloc[:n_train]
            X_val, y_val = X_train.iloc[n_train:], y_train.iloc[n_train:]

            model = train_market_model(X_tr, y_tr, X_val, y_val)

            proba = model.predict_proba(X_test)[:, 1]
            pred = (proba > 0.5).astype(int)

            all_preds.extend(pred.tolist())
            all_proba.extend(proba.tolist())
            all_true.extend(y_test.tolist())

        # Calculate metrics
        acc = accuracy_score(all_true, all_preds)
        try:
            auc = roc_auc_score(all_true, all_proba)
        except:
            auc = 0.5
        brier = brier_score_loss(all_true, all_proba)

        # Positive rate
        positive_rate = sum(all_true) / len(all_true)

        market_results[market_key] = {
            "name": market_info["name"],
            "accuracy": acc,
            "auc": auc,
            "brier": brier,
            "positive_rate": positive_rate,
        }

        # Train final model
        n_train = int(len(X) * 0.85)
        final_model = train_market_model(
            X.iloc[:n_train], y.iloc[:n_train],
            X.iloc[n_train:], y.iloc[n_train:]
        )
        market_models[market_key] = final_model

    # ==========================================
    # Display results sorted by accuracy
    # ==========================================
    log.info("\n" + "=" * 70)
    log.info("MARKET PREDICTION ACCURACY (sorted by accuracy)")
    log.info("=" * 70)
    log.info(f"{'Market':<30} {'Acc':<8} {'AUC':<8} {'Base Rate':<10}")
    log.info("-" * 60)

    sorted_markets = sorted(market_results.items(), key=lambda x: x[1]["accuracy"], reverse=True)

    for market_key, res in sorted_markets:
        skill = res["accuracy"] - res["positive_rate"]  # Skill vs just predicting majority
        log.info(
            f"{res['name']:<30} {res['accuracy']:.1%}    {res['auc']:.3f}    "
            f"{res['positive_rate']:.1%} (skill: {'+' if skill > 0 else ''}{skill:.1%})"
        )

    # ==========================================
    # Save models
    # ==========================================
    model_dir = DATA_DIR / "models" / "markets"
    model_dir.mkdir(parents=True, exist_ok=True)

    for market_key, model in market_models.items():
        model.save_model(str(model_dir / f"{market_key}.cbm"))

    # Save metadata
    metadata = {
        "markets": market_results,
        "feature_names": available_features,
        "n_samples": len(df),
    }

    with open(model_dir / "markets_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    log.info(f"\nSaved {len(market_models)} market models to {model_dir}")

    # ==========================================
    # Best predictions per market
    # ==========================================
    log.info("\n" + "=" * 70)
    log.info("RECOMMENDED BETTING STRATEGY")
    log.info("=" * 70)

    log.info("\nEasiest markets to predict (highest accuracy):")
    for market_key, res in sorted_markets[:5]:
        log.info(f"  {res['name']}: {res['accuracy']:.1%} accuracy")

    log.info("\nKey insight: Instead of trying to predict 'who wins' at 52%,")
    log.info("focus on 'will home team score?' at 75% when confidence is high.")

    log.info("\nTo use this system:")
    log.info("1. Predict all markets for upcoming matches")
    log.info("2. Find the predictions with highest confidence (>70%)")
    log.info("3. Bet only on those high-confidence predictions")


if __name__ == "__main__":
    main()
