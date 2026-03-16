#!/usr/bin/env python3
"""Generate predictions for an upcoming Serie A matchweek.

This script:
1. Fetches upcoming fixtures for the specified matchweek
2. Builds features for each match (without odds for future matches)
3. Generates predictions using the trained model
4. Outputs results in multiple formats (console, JSON, CSV)
5. Optionally saves predictions with timestamps for later validation

Usage:
    python scripts/predict_matchweek.py [--matchweek MW] [--output OUTPUT]
    python scripts/predict_matchweek.py --mw 24 --output predictions_mw24.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR
from web.predictor import MatchPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def get_matchweek_fixtures(matchweek: Optional[int] = None, days_ahead: int = 14) -> pd.DataFrame:
    """Get fixtures for a specific matchweek or upcoming matches.

    Args:
        matchweek: Specific matchweek number (1-38), or None for next upcoming
        days_ahead: Days to look ahead if matchweek not specified

    Returns:
        DataFrame with upcoming match features
    """
    try:
        from features.upcoming_features import build_all_upcoming_features

        df = build_all_upcoming_features(days_ahead=days_ahead, refresh_fixtures=True)

        if df.empty:
            log.warning("No upcoming fixtures found")
            return pd.DataFrame()

        # Filter to specific matchweek if requested
        if matchweek is not None and "matchweek" in df.columns:
            df = df[df["matchweek"] == matchweek]
            if df.empty:
                log.warning(f"No fixtures found for matchweek {matchweek}")

        return df

    except ImportError as e:
        log.error(f"Failed to import upcoming features module: {e}")
        return pd.DataFrame()


def generate_predictions(df: pd.DataFrame) -> list[dict]:
    """Generate predictions for all matches in the DataFrame.

    Args:
        df: DataFrame with match features

    Returns:
        List of prediction dictionaries
    """
    # Use no-odds model for future matches
    predictor = MatchPredictor(use_no_odds_model=True)

    predictions = []

    for _, row in df.iterrows():
        match_features = pd.DataFrame([row])

        prediction = predictor.predict(match_features)

        if "error" in prediction:
            log.warning(f"Prediction failed for {row['home_team']} vs {row['away_team']}: {prediction['error']}")
            continue

        pred_data = {
            "match_id": row.get("match_id", ""),
            "date": str(row.get("match_date", ""))[:10],
            "matchweek": int(row.get("matchweek", 0)) if pd.notna(row.get("matchweek")) else 0,
            "home_team": row.get("home_team", ""),
            "away_team": row.get("away_team", ""),

            # Prediction
            "prediction": prediction.get("prediction", "?"),
            "prob_H": round(prediction.get("probabilities", {}).get("H", 0), 3),
            "prob_D": round(prediction.get("probabilities", {}).get("D", 0), 3),
            "prob_A": round(prediction.get("probabilities", {}).get("A", 0), 3),
            "confidence": round(prediction.get("confidence", 0), 3),

            # Key features
            "home_elo": round(float(row.get("home_elo", 1500)), 0),
            "away_elo": round(float(row.get("away_elo", 1500)), 0),
            "elo_diff": round(float(row.get("home_elo", 1500) - row.get("away_elo", 1500)), 0),
            "home_form": round(float(row.get("home_roll_form_points_pct", 0)) * 100, 0),
            "away_form": round(float(row.get("away_roll_form_points_pct", 0)) * 100, 0),

            # Reasoning
            "reasons": [r["explanation"] for r in prediction.get("reasons", [])[:3]],

            # Metadata
            "generated_at": datetime.now().isoformat(),
        }

        predictions.append(pred_data)

    return predictions


def print_predictions_table(predictions: list[dict]):
    """Print predictions in a nicely formatted table."""
    if not predictions:
        print("No predictions to display")
        return

    print("\n" + "=" * 90)
    print(f"SERIE A MATCHWEEK {predictions[0].get('matchweek', '?')} PREDICTIONS")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 90)

    print(f"\n{'Match':<40} {'Pred':<5} {'H':<7} {'D':<7} {'A':<7} {'Conf':<6}")
    print("-" * 90)

    for pred in predictions:
        match = f"{pred['home_team']} vs {pred['away_team']}"
        if len(match) > 38:
            match = match[:35] + "..."

        print(f"{match:<40} {pred['prediction']:<5} "
              f"{pred['prob_H']:.2f}  {pred['prob_D']:.2f}  {pred['prob_A']:.2f}  "
              f"{pred['confidence']:.0%}")

    print("-" * 90)

    # Summary
    home_wins = sum(1 for p in predictions if p["prediction"] == "H")
    draws = sum(1 for p in predictions if p["prediction"] == "D")
    away_wins = sum(1 for p in predictions if p["prediction"] == "A")
    high_conf = sum(1 for p in predictions if p["confidence"] > 0.5)

    print(f"\nSummary: {home_wins} Home wins, {draws} Draws, {away_wins} Away wins")
    print(f"High confidence predictions (>50%): {high_conf}/{len(predictions)}")

    # Show key reasoning
    print("\n" + "=" * 90)
    print("KEY FACTORS")
    print("=" * 90)

    for pred in predictions[:5]:  # Show reasoning for first 5 matches
        print(f"\n{pred['home_team']} vs {pred['away_team']} -> {pred['prediction']}")
        print(f"  Elo: {pred['home_elo']:.0f} vs {pred['away_elo']:.0f} (diff: {pred['elo_diff']:+.0f})")
        print(f"  Form: {pred['home_form']:.0f}% vs {pred['away_form']:.0f}%")
        if pred['reasons']:
            print(f"  Reasons:")
            for reason in pred['reasons']:
                print(f"    - {reason}")


def save_predictions(predictions: list[dict], output_path: Path):
    """Save predictions to file (JSON or CSV based on extension)."""
    if output_path.suffix == ".json":
        with open(output_path, "w") as f:
            json.dump({
                "matchweek": predictions[0].get("matchweek") if predictions else None,
                "generated_at": datetime.now().isoformat(),
                "predictions": predictions,
            }, f, indent=2)
    elif output_path.suffix == ".csv":
        df = pd.DataFrame(predictions)
        df.to_csv(output_path, index=False)
    else:
        log.warning(f"Unknown file extension: {output_path.suffix}. Saving as JSON.")
        with open(output_path.with_suffix(".json"), "w") as f:
            json.dump(predictions, f, indent=2)

    log.info(f"Saved predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Serie A matchweek predictions")
    parser.add_argument("--matchweek", "--mw", type=int, help="Matchweek number (1-38)")
    parser.add_argument("--days", type=int, default=14, help="Days ahead to look for fixtures")
    parser.add_argument("--output", "-o", type=str, help="Output file (JSON or CSV)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress table output")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("SERIE A MATCHWEEK PREDICTIONS")
    log.info("=" * 70)

    # Get fixtures
    log.info(f"Fetching fixtures (matchweek={args.matchweek}, days_ahead={args.days})...")
    df = get_matchweek_fixtures(matchweek=args.matchweek, days_ahead=args.days)

    if df.empty:
        log.error("No fixtures found. Check if fixtures are available.")
        return

    log.info(f"Found {len(df)} fixtures")

    # Generate predictions
    log.info("Generating predictions...")
    predictions = generate_predictions(df)

    if not predictions:
        log.error("No predictions generated")
        return

    log.info(f"Generated {len(predictions)} predictions")

    # Print table
    if not args.quiet:
        print_predictions_table(predictions)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        save_predictions(predictions, output_path)
    else:
        # Default: save to predictions directory
        pred_dir = DATA_DIR / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)

        mw = predictions[0].get("matchweek", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        default_path = pred_dir / f"mw{mw}_{timestamp}.json"
        save_predictions(predictions, default_path)

    log.info("\n" + "=" * 70)
    log.info("PREDICTIONS COMPLETE")
    log.info("=" * 70)

    return predictions


if __name__ == "__main__":
    main()
