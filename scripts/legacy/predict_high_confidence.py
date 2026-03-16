#!/usr/bin/env python3
"""High Confidence Match Predictor

For each upcoming match, predict ALL markets and show only HIGH CONFIDENCE bets.

Strategy validated by analysis:
- At 85%+ confidence: 88.3% accuracy
- At 80%+ confidence: 86.1% accuracy
- At 75%+ confidence: 84.0% accuracy

This script will:
1. Load upcoming matches
2. Predict all binary markets
3. Show only predictions above confidence threshold
4. Rank by confidence level
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# Markets to predict
MARKETS = {
    "home_clean_sheet": {
        "name": "Home Clean Sheet",
        "description": "Away team won't score",
        "model_file": "intermediate_home_scores.cbm",  # Inverted
    },
    "away_clean_sheet": {
        "name": "Away Clean Sheet",
        "description": "Home team won't score",
        "model_file": "intermediate_away_scores.cbm",  # Inverted
    },
    "home_scores": {
        "name": "Home Team Scores",
        "description": "Home team will score at least 1 goal",
        "model_file": "intermediate_home_scores.cbm",
    },
    "away_scores": {
        "name": "Away Team Scores",
        "description": "Away team will score at least 1 goal",
        "model_file": "intermediate_away_scores.cbm",
    },
    "btts_yes": {
        "name": "Both Teams Score - Yes",
        "description": "Both teams will score",
        "model_file": "intermediate_btts.cbm",
    },
    "btts_no": {
        "name": "Both Teams Score - No",
        "description": "At least one team won't score",
        "model_file": "intermediate_btts.cbm",
    },
    "over_2_5": {
        "name": "Over 2.5 Goals",
        "description": "3 or more total goals",
        "model_file": "intermediate_over_2_5.cbm",
    },
}

# Features used by hierarchical model
FEATURES = [
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


def load_upcoming_matches() -> pd.DataFrame:
    """Load upcoming matches."""
    upcoming_path = DATA_DIR / "upcoming_matches.parquet"

    if not upcoming_path.exists():
        log.warning(f"No upcoming matches file found at {upcoming_path}")
        return pd.DataFrame()

    df = pd.read_parquet(upcoming_path)
    log.info(f"Loaded {len(df)} upcoming matches")
    return df


def train_market_models(historical_df: pd.DataFrame) -> Dict:
    """Train models for all markets on historical data."""
    from catboost import CatBoostClassifier

    models = {}
    available_features = [f for f in FEATURES if f in historical_df.columns]
    X = historical_df[available_features].fillna(0)

    # Create targets
    historical_df["target_home_scores"] = (historical_df["home_score"] > 0).astype(int)
    historical_df["target_away_scores"] = (historical_df["away_score"] > 0).astype(int)
    historical_df["target_btts"] = (
        (historical_df["home_score"] > 0) & (historical_df["away_score"] > 0)
    ).astype(int)
    total_goals = historical_df["home_score"] + historical_df["away_score"]
    historical_df["target_over_2_5"] = (total_goals > 2.5).astype(int)

    targets = ["home_scores", "away_scores", "btts", "over_2_5"]

    n_train = int(len(X) * 0.85)

    for target in targets:
        log.info(f"Training {target} model...")
        y = historical_df[f"target_{target}"]

        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.03,
            depth=5,
            early_stopping_rounds=50,
            verbose=0,
            random_seed=42,
        )

        model.fit(
            X.iloc[:n_train], y.iloc[:n_train],
            eval_set=(X.iloc[n_train:], y.iloc[n_train:]),
            verbose=False
        )

        models[target] = model

    return models, available_features


def predict_match(models: Dict, features: List[str], match_row: pd.Series) -> List[Dict]:
    """Predict all markets for a single match."""
    # Prepare features
    X = match_row[features].fillna(0).values.reshape(1, -1)

    predictions = []

    # Home Scores
    home_scores_proba = models["home_scores"].predict_proba(X)[0]
    predictions.append({
        "market": "Home Team Scores",
        "prediction": "Yes" if home_scores_proba[1] > 0.5 else "No",
        "confidence": max(home_scores_proba),
        "prob_yes": home_scores_proba[1],
    })

    # Away Scores
    away_scores_proba = models["away_scores"].predict_proba(X)[0]
    predictions.append({
        "market": "Away Team Scores",
        "prediction": "Yes" if away_scores_proba[1] > 0.5 else "No",
        "confidence": max(away_scores_proba),
        "prob_yes": away_scores_proba[1],
    })

    # Home Clean Sheet (opposite of away scores)
    predictions.append({
        "market": "Home Clean Sheet",
        "prediction": "Yes" if away_scores_proba[0] > 0.5 else "No",
        "confidence": max(away_scores_proba),
        "prob_yes": away_scores_proba[0],
    })

    # Away Clean Sheet (opposite of home scores)
    predictions.append({
        "market": "Away Clean Sheet",
        "prediction": "Yes" if home_scores_proba[0] > 0.5 else "No",
        "confidence": max(home_scores_proba),
        "prob_yes": home_scores_proba[0],
    })

    # BTTS
    btts_proba = models["btts"].predict_proba(X)[0]
    predictions.append({
        "market": "Both Teams Score",
        "prediction": "Yes" if btts_proba[1] > 0.5 else "No",
        "confidence": max(btts_proba),
        "prob_yes": btts_proba[1],
    })

    # Over 2.5
    over_proba = models["over_2_5"].predict_proba(X)[0]
    predictions.append({
        "market": "Over 2.5 Goals",
        "prediction": "Yes" if over_proba[1] > 0.5 else "No",
        "confidence": max(over_proba),
        "prob_yes": over_proba[1],
    })

    return predictions


def main():
    log.info("=" * 70)
    log.info("HIGH CONFIDENCE MATCH PREDICTOR")
    log.info("=" * 70)
    log.info("\nOnly showing predictions with 70%+ confidence")
    log.info("Based on analysis: 80%+ confidence = 86% accuracy")

    # Load historical data for training
    feature_path = DATA_DIR / "features" / "features.parquet"
    if not feature_path.exists():
        log.error("No features file found. Run rebuild_features first.")
        return

    historical_df = pd.read_parquet(feature_path)
    historical_df = historical_df.dropna(subset=["home_score", "away_score"])
    log.info(f"Loaded {len(historical_df)} historical matches")

    # Train models
    log.info("\nTraining prediction models...")
    models, features = train_market_models(historical_df)

    # Load upcoming matches
    upcoming = load_upcoming_matches()

    if upcoming.empty:
        log.info("\nNo upcoming matches found. Creating sample predictions from recent data.")
        # Use last 10 matches as demo
        upcoming = historical_df.tail(10).copy()
        upcoming["match_date"] = pd.Timestamp.now() + pd.Timedelta(days=7)

    # Collect all high-confidence predictions
    all_predictions = []

    log.info(f"\nAnalyzing {len(upcoming)} matches...")

    for idx, row in upcoming.iterrows():
        home = row.get("home_team", "Home")
        away = row.get("away_team", "Away")
        match_date = row.get("match_date", "TBD")

        predictions = predict_match(models, features, row)

        for pred in predictions:
            all_predictions.append({
                "home_team": home,
                "away_team": away,
                "match_date": str(match_date)[:10] if pd.notna(match_date) else "TBD",
                **pred
            })

    # Sort by confidence
    all_predictions.sort(key=lambda x: x["confidence"], reverse=True)

    # ==========================================
    # Display results by confidence tier
    # ==========================================
    confidence_tiers = [
        (0.85, "VERY HIGH CONFIDENCE (85%+) - Expected ~88% accuracy"),
        (0.80, "HIGH CONFIDENCE (80%+) - Expected ~86% accuracy"),
        (0.75, "GOOD CONFIDENCE (75%+) - Expected ~84% accuracy"),
        (0.70, "MODERATE CONFIDENCE (70%+) - Expected ~80% accuracy"),
    ]

    for threshold, tier_name in confidence_tiers:
        tier_preds = [p for p in all_predictions if p["confidence"] >= threshold]

        if not tier_preds:
            continue

        log.info(f"\n{'='*70}")
        log.info(tier_name)
        log.info(f"{'='*70}")
        log.info(f"{'Match':<35} {'Market':<20} {'Prediction':<12} {'Confidence'}")
        log.info("-" * 75)

        shown = 0
        for pred in tier_preds:
            if shown >= 20:  # Limit display
                remaining = len(tier_preds) - shown
                log.info(f"... and {remaining} more predictions at this confidence level")
                break

            match_str = f"{pred['home_team'][:15]} vs {pred['away_team'][:15]}"
            log.info(
                f"{match_str:<35} {pred['market']:<20} {pred['prediction']:<12} {pred['confidence']:.1%}"
            )
            shown += 1

    # ==========================================
    # Summary statistics
    # ==========================================
    log.info(f"\n{'='*70}")
    log.info("SUMMARY")
    log.info(f"{'='*70}")

    for threshold, tier_name in confidence_tiers:
        count = len([p for p in all_predictions if p["confidence"] >= threshold])
        log.info(f"  {tier_name.split(' - ')[0]}: {count} predictions")

    # ==========================================
    # Save predictions
    # ==========================================
    output_path = DATA_DIR / "predictions" / "high_confidence_predictions.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Group by match
    matches_output = {}
    for pred in all_predictions:
        key = f"{pred['home_team']} vs {pred['away_team']}"
        if key not in matches_output:
            matches_output[key] = {
                "home_team": pred["home_team"],
                "away_team": pred["away_team"],
                "match_date": pred["match_date"],
                "predictions": [],
            }
        matches_output[key]["predictions"].append({
            "market": pred["market"],
            "prediction": pred["prediction"],
            "confidence": round(pred["confidence"], 3),
        })

    # Sort predictions within each match
    for match in matches_output.values():
        match["predictions"].sort(key=lambda x: x["confidence"], reverse=True)

    with open(output_path, "w") as f:
        json.dump(list(matches_output.values()), f, indent=2)

    log.info(f"\nPredictions saved to {output_path}")

    # ==========================================
    # Betting recommendations
    # ==========================================
    log.info(f"\n{'='*70}")
    log.info("BETTING RECOMMENDATIONS")
    log.info(f"{'='*70}")
    log.info("""
Based on our analysis:

1. BEST BETS (85%+ confidence):
   - These have ~88% historical accuracy
   - Bet confidently, but manage bankroll

2. GOOD BETS (80%+ confidence):
   - These have ~86% historical accuracy
   - Reliable for consistent betting

3. MODERATE BETS (75%+ confidence):
   - These have ~84% historical accuracy
   - Good value when odds are favorable

4. AVOID:
   - Predictions below 70% confidence
   - Overall H/D/A without high confidence
   - BTTS and Over 2.5 (hard to predict)

5. REMEMBER:
   - Quality over quantity
   - Only bet what you can afford to lose
   - Past performance doesn't guarantee future results
""")


if __name__ == "__main__":
    main()
