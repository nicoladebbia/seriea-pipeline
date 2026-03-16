#!/usr/bin/env python3
"""Train models with contextual features - Expert-level match analysis.

This script tests whether adding contextual features (strength-adjusted form,
matchup analysis, defensive consistency) improves prediction accuracy.

Contextual features capture how expert analysts think:
- Beating Inter (1st) is worth more than beating Verona (18th)
- Strong home attack vs weak away defense = likely home goals
- Teams with clean sheet streaks vs weak attacks = likely clean sheet
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR
from ml.config import LABEL_MAP
from features.contextual_features import add_contextual_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# Base pre-match features
BASE_FEATURES = [
    "home_elo", "away_elo", "elo_diff",
    "home_attack_strength", "home_defense_strength",
    "away_attack_strength", "away_defense_strength",
    "home_form_points_3", "home_form_points_5",
    "away_form_points_3", "away_form_points_5",
    "home_roll_3_goals_scored", "home_roll_3_goals_conceded",
    "away_roll_3_goals_scored", "away_roll_3_goals_conceded",
    "league_position_diff",
    "home_in_relegation_zone", "away_in_relegation_zone",
    "h2h_matches_played", "h2h_home_wins", "h2h_draws",
    "home_rest_days", "away_rest_days",
]

# New contextual features
CONTEXTUAL_FEATURES = [
    # Strength-adjusted form
    "home_strength_adj_points", "away_strength_adj_points",
    "home_top6_wins", "away_top6_wins",
    "home_bottom6_wins", "away_bottom6_wins",
    # Defensive
    "home_clean_sheet_streak", "away_clean_sheet_streak",
    "home_clean_sheet_rate", "away_clean_sheet_rate",
    "home_goals_vs_top_attacks", "away_goals_vs_top_attacks",
    # Attack
    "home_scoring_streak", "away_scoring_streak",
    "home_scoring_rate", "away_scoring_rate",
    "home_goals_vs_weak_def", "away_goals_vs_weak_def",
    # Matchup
    "home_attack_vs_away_def", "away_attack_vs_home_def",
    "home_clean_sheet_signal", "away_clean_sheet_signal",
    "position_gap", "mismatch_score",
    # Momentum
    "home_form_trend", "away_form_trend",
    "home_recent_big_win", "away_recent_big_win",
]


def time_series_split(df: pd.DataFrame, n_splits: int = 5) -> List[Tuple[pd.Index, pd.Index]]:
    """Walk-forward time series splits by season."""
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


def train_model(X_train, y_train, X_val, y_val):
    """Train CatBoost classifier with early stopping."""
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


def train_binary_model(X_train, y_train, X_val, y_val):
    """Train binary classifier for market predictions."""
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


def evaluate_cv(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, splits, model_name: str):
    """Run cross-validation and return metrics."""
    all_preds = []
    all_true = []
    all_proba = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx]
        X_test = X.loc[test_idx]
        y_test = y.loc[test_idx]

        # Split train into train/val
        n_train = int(len(X_train) * 0.85)
        X_tr, y_tr = X_train.iloc[:n_train], y_train.iloc[:n_train]
        X_val, y_val = X_train.iloc[n_train:], y_train.iloc[n_train:]

        model = train_model(X_tr, y_tr, X_val, y_val)

        proba = model.predict_proba(X_test)
        pred = np.argmax(proba, axis=1)

        fold_acc = accuracy_score(y_test, pred)
        log.info(f"  {model_name} Fold {fold_idx+1}: {fold_acc:.1%}")

        all_preds.extend(pred.tolist())
        all_true.extend(y_test.tolist())
        all_proba.extend(proba.tolist())

    acc = accuracy_score(all_true, all_preds)
    loss = log_loss(all_true, all_proba)

    return acc, loss, all_preds, all_true


def evaluate_binary_market(df: pd.DataFrame, X: pd.DataFrame, target_col: str, splits, market_name: str):
    """Evaluate a binary market prediction."""
    y = df[target_col]

    all_preds = []
    all_true = []
    all_proba = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx]
        X_test = X.loc[test_idx]
        y_test = y.loc[test_idx]

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
    try:
        auc = roc_auc_score(all_true, all_proba)
    except:
        auc = 0.5

    base_rate = sum(all_true) / len(all_true)
    skill = acc - max(base_rate, 1 - base_rate)

    return {"accuracy": acc, "auc": auc, "base_rate": base_rate, "skill": skill}


def main():
    log.info("=" * 70)
    log.info("CONTEXTUAL FEATURES - EXPERT-LEVEL MATCH ANALYSIS")
    log.info("=" * 70)

    # Load features
    feature_path = DATA_DIR / "features" / "features.parquet"
    df = pd.read_parquet(feature_path)
    df = df.dropna(subset=["home_score", "away_score"])
    log.info(f"Loaded {len(df)} matches")

    # Add contextual features
    log.info("\nComputing contextual features (this may take a moment)...")
    df_contextual = add_contextual_features(df, df)

    # Create targets
    df_contextual["result"] = df_contextual.apply(
        lambda r: "H" if r["home_score"] > r["away_score"]
        else ("A" if r["away_score"] > r["home_score"] else "D"),
        axis=1
    )
    y = df_contextual["result"].map(LABEL_MAP)

    # Create binary targets for markets
    df_contextual["target_home_clean_sheet"] = (df_contextual["away_score"] == 0).astype(int)
    df_contextual["target_away_clean_sheet"] = (df_contextual["home_score"] == 0).astype(int)
    df_contextual["target_home_scores"] = (df_contextual["home_score"] > 0).astype(int)
    df_contextual["target_away_scores"] = (df_contextual["away_score"] > 0).astype(int)
    df_contextual["target_btts"] = ((df_contextual["home_score"] > 0) & (df_contextual["away_score"] > 0)).astype(int)
    df_contextual["target_over_2_5"] = ((df_contextual["home_score"] + df_contextual["away_score"]) > 2.5).astype(int)

    # Get available features
    available_base = [f for f in BASE_FEATURES if f in df_contextual.columns]
    available_contextual = [f for f in CONTEXTUAL_FEATURES if f in df_contextual.columns]

    log.info(f"Base features: {len(available_base)}")
    log.info(f"Contextual features: {len(available_contextual)}")

    X_base = df_contextual[available_base].fillna(0)
    X_full = df_contextual[available_base + available_contextual].fillna(0)

    # Time series splits
    splits = time_series_split(df_contextual, n_splits=5)
    log.info(f"Using {len(splits)} walk-forward splits")

    # ==========================================
    # PART 1: H/D/A Prediction
    # ==========================================
    log.info("\n" + "=" * 50)
    log.info("PART 1: H/D/A MATCH RESULT PREDICTION")
    log.info("=" * 50)

    log.info("\nBaseline model (without contextual features):")
    base_acc, base_loss, _, _ = evaluate_cv(df_contextual, X_base, y, splits, "Baseline")
    log.info(f"  Overall: {base_acc:.1%} accuracy, {base_loss:.3f} log_loss")

    log.info("\nContextual model (with expert features):")
    ctx_acc, ctx_loss, _, _ = evaluate_cv(df_contextual, X_full, y, splits, "Contextual")
    log.info(f"  Overall: {ctx_acc:.1%} accuracy, {ctx_loss:.3f} log_loss")

    improvement = ctx_acc - base_acc
    log.info(f"\nImprovement: {'+' if improvement > 0 else ''}{improvement:.1%}")

    # ==========================================
    # PART 2: Binary Market Predictions
    # ==========================================
    log.info("\n" + "=" * 50)
    log.info("PART 2: BINARY MARKET PREDICTIONS (with contextual features)")
    log.info("=" * 50)

    markets = [
        ("Home Clean Sheet", "target_home_clean_sheet"),
        ("Away Clean Sheet", "target_away_clean_sheet"),
        ("Home Scores", "target_home_scores"),
        ("Away Scores", "target_away_scores"),
        ("BTTS", "target_btts"),
        ("Over 2.5", "target_over_2_5"),
    ]

    market_results = {}
    for market_name, target_col in markets:
        log.info(f"\n{market_name}:")
        result = evaluate_binary_market(df_contextual, X_full, target_col, splits, market_name)
        market_results[market_name] = result
        log.info(f"  Accuracy: {result['accuracy']:.1%} (Base rate: {result['base_rate']:.1%}, Skill: {'+' if result['skill'] > 0 else ''}{result['skill']:.1%})")
        log.info(f"  AUC: {result['auc']:.3f}")

    # ==========================================
    # Save best model
    # ==========================================
    log.info("\n" + "=" * 50)
    log.info("SAVING MODEL WITH CONTEXTUAL FEATURES")
    log.info("=" * 50)

    # Train final model on all data
    n_train = int(len(X_full) * 0.85)
    final_model = train_model(
        X_full.iloc[:n_train], y.iloc[:n_train],
        X_full.iloc[n_train:], y.iloc[n_train:]
    )

    model_dir = DATA_DIR / "models" / "universal"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = model_dir / "catboost_contextual.cbm"
    final_model.save_model(str(model_path))

    # Also save as upcoming model if it's better
    if ctx_acc > base_acc:
        final_model.save_model(str(model_dir / "catboost_upcoming.cbm"))
        final_model.save_model(str(model_dir / "catboost_no_odds.cbm"))
        log.info("Contextual model saved as default upcoming model")

    # Save metadata
    metadata = {
        "model_type": "contextual_expert",
        "cv_accuracy": round(ctx_acc, 4),
        "cv_log_loss": round(ctx_loss, 4),
        "baseline_accuracy": round(base_acc, 4),
        "improvement": round(improvement, 4),
        "n_features": len(X_full.columns),
        "feature_names": list(X_full.columns),
        "base_features": available_base,
        "contextual_features": available_contextual,
        "label_map": LABEL_MAP,
        "market_results": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in market_results.items()},
    }

    with open(model_dir / "catboost_contextual_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if ctx_acc > base_acc:
        for path in [model_dir / "catboost_upcoming_metadata.json",
                     model_dir / "catboost_no_odds_metadata.json"]:
            with open(path, "w") as f:
                json.dump(metadata, f, indent=2)

    log.info(f"\nSaved model to {model_path}")

    # ==========================================
    # Summary
    # ==========================================
    log.info("\n" + "=" * 70)
    log.info("CONTEXTUAL FEATURES SUMMARY")
    log.info("=" * 70)

    log.info("\nH/D/A Prediction:")
    log.info(f"  Baseline:   {base_acc:.1%}")
    log.info(f"  Contextual: {ctx_acc:.1%} ({'+' if improvement > 0 else ''}{improvement:.1%})")

    log.info("\nBest Market Predictions:")
    sorted_markets = sorted(market_results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    for name, result in sorted_markets[:5]:
        log.info(f"  {name}: {result['accuracy']:.1%} accuracy ({'+' if result['skill'] > 0 else ''}{result['skill']:.1%} skill)")

    log.info("\nKey contextual features added:")
    log.info("  - Strength-adjusted form (opponent quality weighting)")
    log.info("  - Clean sheet streaks and consistency")
    log.info("  - Attack-defense matchup scores")
    log.info("  - Form trends and momentum indicators")

    if ctx_acc >= 0.55:
        log.info("\n✓ Model achieves 55%+ accuracy!")
    elif ctx_acc > base_acc:
        log.info(f"\n✓ Contextual features improved accuracy by {improvement:.1%}")
    else:
        log.info("\n✗ Contextual features did not improve - may need more historical data")


if __name__ == "__main__":
    main()
