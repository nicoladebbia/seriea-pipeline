#!/usr/bin/env python3
"""Train a no-odds model quickly using default parameters."""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR
from ml.config import LABEL_MAP, ODDS_COLUMN_PATTERNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def identify_odds_columns(feature_names: list[str]) -> list[str]:
    """Return feature names that are derived from betting odds."""
    return [
        f for f in feature_names
        if any(f.startswith(p) or f.endswith(p.rstrip("_")) for p in ODDS_COLUMN_PATTERNS)
    ]


def main():
    log.info("=" * 70)
    log.info("TRAINING NO-ODDS MODEL (FAST)")
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

    # Identify feature columns
    exclude = ["match_id", "home_team", "away_team", "home_score", "away_score",
               "match_date", "season", "result", "league", "referee", "matchweek"]

    feature_cols = [c for c in df.columns if c not in exclude
                    and df[c].dtype in ["float64", "int64", "float32", "int32"]
                    and not c.startswith("_")]

    # Remove odds columns
    odds_cols = identify_odds_columns(feature_cols)
    feature_cols = [f for f in feature_cols if f not in set(odds_cols)]
    log.info(f"Excluded {len(odds_cols)} odds columns, using {len(feature_cols)} features")

    # Prepare data
    X = df[feature_cols].fillna(0)
    y = df["result"]
    y_int = y.map(LABEL_MAP)

    log.info(f"Training on {len(X)} samples with {len(feature_cols)} features")

    # Train CatBoost model (best default performance)
    try:
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            iterations=800,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3,
            auto_class_weights="Balanced",
            verbose=100,
            random_seed=42,
        )

        # Use DataFrame to preserve feature names
        model.fit(X, y_int.values)

        # Save model
        model_dir = DATA_DIR / "models" / "universal"
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "catboost_no_odds.cbm"
        model.save_model(str(model_path))
        log.info(f"Saved model to {model_path}")

        # Save metadata
        metadata = {
            "model_type": "catboost",
            "n_features": len(feature_cols),
            "n_samples": len(X),
            "feature_names": feature_cols,
            "label_map": LABEL_MAP,
            "excluded_odds_features": odds_cols,
        }

        metadata_path = model_dir / "catboost_no_odds_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        log.info(f"Saved metadata to {metadata_path}")

        # Evaluate on training data
        y_pred = model.predict(X.values)
        accuracy = (y_pred == y_int.values).mean()
        log.info(f"Training accuracy: {accuracy:.1%}")

        log.info("=" * 70)
        log.info("NO-ODDS MODEL TRAINING COMPLETE")
        log.info("=" * 70)

    except ImportError:
        log.error("CatBoost not installed. Install with: pip install catboost")
        return


if __name__ == "__main__":
    main()
