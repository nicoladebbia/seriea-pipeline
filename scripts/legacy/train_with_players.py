#!/usr/bin/env python3
"""Train a prediction model WITH player-level features.

This creates a model that uses individual player performance metrics
aggregated to the team level, providing richer features for prediction.

Usage:
    python scripts/train_with_players.py [--quick]
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.team_names import normalize_team
from features.player_aggregation import build_match_player_features
from storage.paths import features_path, PARSED_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def add_player_features_to_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Add player aggregation features to historical matches."""

    # Only process matches where we might have player data
    df = df.copy()
    df['match_date'] = pd.to_datetime(df['match_date'])

    # Initialize new columns
    player_feature_cols = [
        'home_player_total_xg', 'home_player_total_xga', 'home_player_avg_xg_per_90',
        'home_player_avg_touches', 'home_player_creative_output', 'home_player_defensive_output',
        'home_player_top_scorer_xg', 'home_player_data_available',
        'away_player_total_xg', 'away_player_total_xga', 'away_player_avg_xg_per_90',
        'away_player_avg_touches', 'away_player_creative_output', 'away_player_defensive_output',
        'away_player_top_scorer_xg', 'away_player_data_available',
        'player_xg_diff', 'player_creative_diff', 'player_defensive_diff',
    ]

    for col in player_feature_cols:
        df[col] = 0.0

    # Process each match
    log.info(f"Adding player features to {len(df)} matches...")

    for idx, row in df.iterrows():
        try:
            home_team = row['home_team']
            away_team = row['away_team']
            match_date = row['match_date'].date() if hasattr(row['match_date'], 'date') else row['match_date']

            player_features = build_match_player_features(
                home_team=home_team,
                away_team=away_team,
                match_date=match_date,
                n_matches=5,
            )

            for col, val in player_features.items():
                if col in df.columns:
                    df.at[idx, col] = val

        except Exception as e:
            # Keep default values for matches without player data
            pass

    # Count how many matches got player data
    with_data = (df['home_player_data_available'] > 0).sum()
    log.info(f"Added player features to {with_data} matches ({100*with_data/len(df):.1f}%)")

    return df


def train_model_with_players(df: pd.DataFrame, quick: bool = False) -> dict:
    """Train a CatBoost model with player features."""
    from catboost import CatBoostClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, log_loss

    # Prepare target variable
    def get_result(row):
        if pd.isna(row.get('home_score')) or pd.isna(row.get('away_score')):
            return None
        h = int(row['home_score'])
        a = int(row['away_score'])
        if h > a:
            return 'H'
        elif a > h:
            return 'A'
        else:
            return 'D'

    df = df.copy()
    df['result'] = df.apply(get_result, axis=1)
    df = df.dropna(subset=['result'])

    log.info(f"Training on {len(df)} matches with results")

    # Select features (exclude metadata and target)
    exclude_cols = [
        'match_id', 'home_team', 'away_team', 'home_score', 'away_score',
        'match_date', 'season', 'matchweek', 'result', 'league',
        '_has_odds', '_season', 'referee', 'venue', 'time',
    ]

    # Include betting odds columns (they help with training even if not used for upcoming)
    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols
        and not c.startswith('_')
        and df[c].dtype in ['float64', 'int64', 'float32', 'int32']
    ]

    log.info(f"Using {len(feature_cols)} features")

    # Print top features
    log.info("Features include:")
    for i, col in enumerate(sorted(feature_cols)[:20]):
        log.info(f"  {i+1}. {col}")
    log.info("  ...")

    # Prepare data
    X = df[feature_cols].fillna(0).astype(float)
    y = df['result']

    # Sort by date for time-series split
    df = df.sort_values('match_date')
    X = X.loc[df.index]
    y = y.loc[df.index]

    # Time series cross-validation
    n_splits = 3 if quick else 5
    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Class weights to help with draw prediction
        class_weights = {'H': 1.0, 'D': 1.25, 'A': 1.0}

        model = CatBoostClassifier(
            iterations=500 if quick else 1000,
            learning_rate=0.05,
            depth=6,
            class_weights=class_weights,
            verbose=False,
            random_seed=42,
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        ll = log_loss(y_test, y_proba, labels=['A', 'D', 'H'])

        # Per-class accuracy
        for cls in ['H', 'D', 'A']:
            mask = y_test == cls
            if mask.sum() > 0:
                cls_acc = (y_pred[mask] == cls).mean()
                log.info(f"  Fold {fold+1} {cls} accuracy: {cls_acc:.3f} ({mask.sum()} samples)")

        results.append({'fold': fold, 'accuracy': acc, 'log_loss': ll})
        log.info(f"Fold {fold+1}: accuracy={acc:.4f}, log_loss={ll:.4f}")

    results_df = pd.DataFrame(results)
    avg_acc = results_df['accuracy'].mean()
    avg_ll = results_df['log_loss'].mean()

    log.info(f"\nCross-validation results:")
    log.info(f"  Average accuracy: {avg_acc:.4f}")
    log.info(f"  Average log_loss: {avg_ll:.4f}")

    # Train final model on all data
    log.info("\nTraining final model on all data...")

    final_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        class_weights={'H': 1.0, 'D': 1.25, 'A': 1.0},
        verbose=False,
        random_seed=42,
    )

    final_model.fit(X, y)

    # Save model
    model_dir = Path('data/models/universal')
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / 'catboost_with_players.cbm'
    final_model.save_model(str(model_path))
    log.info(f"Saved model to {model_path}")

    # Save feature names
    import json
    metadata = {
        'feature_names': feature_cols,
        'accuracy': avg_acc,
        'log_loss': avg_ll,
        'n_features': len(feature_cols),
        'n_training_samples': len(df),
    }

    metadata_path = model_dir / 'catboost_with_players_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Saved metadata to {metadata_path}")

    # Feature importance
    importance = dict(zip(feature_cols, final_model.get_feature_importance()))
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    log.info("\nTop 20 most important features:")
    for i, (feat, imp) in enumerate(sorted_imp[:20]):
        log.info(f"  {i+1}. {feat}: {imp:.2f}")

    return {
        'accuracy': avg_acc,
        'log_loss': avg_ll,
        'model_path': str(model_path),
        'feature_names': feature_cols,
        'importance': importance,
    }


def main():
    parser = argparse.ArgumentParser(description="Train model with player features")
    parser.add_argument("--quick", action="store_true", help="Use fewer iterations")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("TRAINING PREDICTION MODEL WITH PLAYER-LEVEL FEATURES")
    log.info("=" * 70)

    # Load features
    feat_path = features_path()
    if not feat_path.exists():
        log.error(f"Features file not found: {feat_path}")
        return

    df = pd.read_parquet(feat_path)

    # Filter to Serie A and recent seasons with player data
    df = df[df['league'] == 'serie_a'].copy()
    df['match_date'] = pd.to_datetime(df['match_date'])

    # Focus on seasons where we likely have player data (2023-24, 2024-25)
    df = df[df['season'].isin(['2023-2024', '2024-2025'])]
    log.info(f"Filtered to {len(df)} Serie A matches from 2023-25")

    # Add player features
    df = add_player_features_to_dataset(df)

    # Filter to matches with player data
    df_with_players = df[df['home_player_data_available'] > 0]
    log.info(f"Found {len(df_with_players)} matches with player data")

    if len(df_with_players) < 100:
        log.warning("Not enough matches with player data, using all data")
        df_train = df
    else:
        df_train = df_with_players

    # Train model
    results = train_model_with_players(df_train, quick=args.quick)

    log.info("")
    log.info("=" * 70)
    log.info("TRAINING COMPLETE")
    log.info("=" * 70)
    log.info(f"Accuracy: {results['accuracy']:.4f}")
    log.info(f"Log loss: {results['log_loss']:.4f}")
    log.info(f"Model saved: {results['model_path']}")

    return results


if __name__ == "__main__":
    main()
