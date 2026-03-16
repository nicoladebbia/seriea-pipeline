#!/usr/bin/env python3
"""Train a model using only pre-match features available for predictions.

This model uses only features that we can compute BEFORE a match occurs:
- Elo ratings
- Rolling form (goals, points, xG)
- Head-to-head history
- League position
- Venue features (travel distance, altitude)

It EXCLUDES post-match data like:
- Actual goals scored
- Half-time scores
- Shots on target
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR
from ml.config import LABEL_MAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# Features that are AVAILABLE pre-match (rolling, historical, computed)
PRE_MATCH_PATTERNS = [
    # Elo and strength
    "home_elo", "away_elo", "elo_diff", "home_strength", "away_strength",
    # Rolling form
    "roll_", "form_", "_trend",
    # Head-to-head
    "h2h_",
    # League position and points
    "league_position", "points_", "position_diff",
    # Venue and travel
    "travel_distance", "altitude", "stadium_capacity", "long_travel",
    # Motivation and context
    "motivation", "derby", "big_game", "relegation", "title_race",
    # Manager tenure
    "manager_tenure",
    # Congestion
    "congestion", "rest_days",
    # Tactical style
    "style_", "tactical",
    # Weather
    "weather", "adverse_weather", "rain", "wind", "cold", "hot",
    # Referee tendencies (historical)
    "ref_",
    # General team metrics (per-game averages)
    "_avg", "_pct", "_rate",
]

# Features that are NOT available pre-match (actual match data)
POST_MATCH_PATTERNS = [
    "home_score", "away_score", "result", "ht_score", "ft_score",
    "shots", "shots_on_target", "corners", "fouls", "cards",
    "attendance", "possession", "passing_accuracy",
    "offsides", "touches", "dribbles", "tackles", "interceptions",
]


def is_pre_match_feature(col: str) -> bool:
    """Check if a column name represents a pre-match feature."""
    col_lower = col.lower()

    # Exclude post-match data
    for pattern in POST_MATCH_PATTERNS:
        if pattern in col_lower and "roll" not in col_lower and "avg" not in col_lower:
            return False

    # Include if matches pre-match patterns
    for pattern in PRE_MATCH_PATTERNS:
        if pattern in col_lower:
            return True

    return False


def main():
    log.info("=" * 70)
    log.info("TRAINING PRE-MATCH MODEL (for predictions)")
    log.info("=" * 70)

    # Load features
    feature_path = DATA_DIR / "features" / "features.parquet"
    df = pd.read_parquet(feature_path)
    log.info(f"Loaded {len(df)} matches with {len(df.columns)} columns")

    # Create target
    df = df.dropna(subset=["home_score", "away_score"])
    df["result"] = df.apply(
        lambda r: "H" if r["home_score"] > r["away_score"]
        else ("A" if r["away_score"] > r["home_score"] else "D"),
        axis=1
    )

    # Identify pre-match feature columns
    exclude = ["match_id", "home_team", "away_team", "home_score", "away_score",
               "match_date", "season", "result", "league", "referee", "matchweek"]

    all_numeric = [c for c in df.columns if c not in exclude
                   and df[c].dtype in ["float64", "int64", "float32", "int32"]
                   and not c.startswith("_")]

    # Filter to pre-match features only
    pre_match_cols = [c for c in all_numeric if is_pre_match_feature(c)]

    # Also exclude odds for a fair baseline
    pre_match_cols = [c for c in pre_match_cols if "odds" not in c.lower()
                      and "implied_prob" not in c.lower()]

    log.info(f"Selected {len(pre_match_cols)} pre-match features from {len(all_numeric)} total")
    log.info(f"Sample features: {pre_match_cols[:10]}")

    # Prepare data
    X = df[pre_match_cols].fillna(0)
    y = df["result"]
    y_int = y.map(LABEL_MAP)

    # Class distribution
    class_counts = y.value_counts()
    log.info(f"Class distribution: {dict(class_counts)}")

    log.info(f"Training on {len(X)} samples with {len(pre_match_cols)} features")

    # Train CatBoost model
    try:
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.02,
            depth=6,
            l2_leaf_reg=5,
            # Don't over-weight draws, let the data speak
            auto_class_weights="SqrtBalanced",
            verbose=100,
            random_seed=42,
            eval_fraction=0.15,
            early_stopping_rounds=50,
        )

        model.fit(X, y_int.values)

        # Save model
        model_dir = DATA_DIR / "models" / "universal"
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "catboost_upcoming.cbm"
        model.save_model(str(model_path))
        log.info(f"Saved model to {model_path}")

        # Also save as no_odds model (overwrite)
        no_odds_path = model_dir / "catboost_no_odds.cbm"
        model.save_model(str(no_odds_path))
        log.info(f"Also saved to {no_odds_path}")

        # Save metadata
        metadata = {
            "model_type": "catboost_upcoming",
            "n_features": len(pre_match_cols),
            "n_samples": len(X),
            "feature_names": pre_match_cols,
            "label_map": LABEL_MAP,
            "pre_match_only": True,
        }

        metadata_path = model_dir / "catboost_upcoming_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        log.info(f"Saved metadata to {metadata_path}")

        # Also save for no_odds
        no_odds_meta_path = model_dir / "catboost_no_odds_metadata.json"
        with open(no_odds_meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Evaluate on training data
        y_pred = model.predict(X)
        accuracy = (y_pred == y_int.values).mean()
        log.info(f"Training accuracy: {accuracy:.1%}")

        # Per-class accuracy
        for cls in ["H", "D", "A"]:
            mask = y == cls
            cls_int = LABEL_MAP[cls]
            cls_acc = (y_pred[mask] == cls_int).mean()
            log.info(f"  {cls}: {cls_acc:.1%}")

        log.info("=" * 70)
        log.info("PRE-MATCH MODEL TRAINING COMPLETE")
        log.info("=" * 70)

    except ImportError:
        log.error("CatBoost not installed. Install with: pip install catboost")
        return


if __name__ == "__main__":
    main()
