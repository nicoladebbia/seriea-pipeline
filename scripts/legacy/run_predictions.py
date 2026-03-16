#!/usr/bin/env python3
"""MASTER PREDICTION RUNNER

Single command to run the complete prediction pipeline:
1. Fetch/load upcoming matches
2. Calculate current form
3. Get weather forecasts
4. Generate predictions using ENSEMBLE MODEL
5. Output to console and file

The ENSEMBLE MODEL combines three approaches:
- Factor-based predictions (40%) - interpretable, validated factors
- xG-based predictions (40%) - expected goals + Poisson distribution
- ML classifier predictions (20%) - CatBoost trained on 51 features

Usage:
    python scripts/run_predictions.py              # Full ensemble output
    python scripts/run_predictions.py --brief      # Brief output (high confidence only)
    python scripts/run_predictions.py --json       # JSON only
    python scripts/run_predictions.py --no-ensemble  # Use factor-only (legacy)

This is the PRODUCTION prediction system.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="Serie A Match Prediction System")
    parser.add_argument("--brief", action="store_true", help="Brief output (high confidence only)")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    parser.add_argument("--no-weather", action="store_true", help="Skip weather fetch (faster)")
    parser.add_argument("--no-ensemble", action="store_true",
                       help="Use factor-only predictions (legacy mode)")
    args = parser.parse_args()

    # Use ensemble by default, fall back to factor-only if requested
    if args.no_ensemble:
        from scripts.realtime_prediction_engine import run_predictions, print_predictions
        output = run_predictions()
    else:
        try:
            from scripts.prediction.ensemble_prediction_engine import (
                run_ensemble_predictions,
                print_ensemble_predictions
            )
            output = run_ensemble_predictions(use_ensemble=True)
            print_predictions = print_ensemble_predictions
        except Exception as e:
            print(f"Warning: Ensemble failed ({e}), falling back to factor-only")
            from scripts.realtime_prediction_engine import run_predictions, print_predictions
            output = run_predictions()

    if not output:
        print("ERROR: No predictions generated!")
        return 1

    if args.json:
        # Output JSON only
        print(json.dumps(output, indent=2))
        return 0

    if args.brief:
        # Brief output - high confidence only
        predictions = output.get("predictions", [])
        # Handle both "confidence" (factor-only) and "confidence_level" (ensemble) fields
        high_conf = [p for p in predictions
                    if p.get("confidence_level", p.get("confidence")) in ["VERY HIGH", "HIGH", "MEDIUM-HIGH"]]

        print("\n" + "=" * 60)
        print("SERIE A PREDICTIONS - HIGH CONFIDENCE")
        if output.get("ensemble_enabled"):
            print(f"Model: ENSEMBLE ({', '.join(output.get('methods_available', ['factor']))})")
        print("=" * 60)

        for pred in high_conf:
            probs = pred["probabilities"]
            factors = len(pred.get("home_factors", [])) + len(pred.get("away_factors", []))
            conf = pred.get("confidence_level", pred.get("confidence", "MEDIUM"))
            print(f"\n{pred['match']} ({pred.get('date', 'TBD')})")
            print(f"  -> {pred['predicted_outcome']} ({conf}, {factors} factors)")
            print(f"  H {probs['home']:.0%} | D {probs['draw']:.0%} | A {probs['away']:.0%}")

            # Show xG if available
            if pred.get("home_xg") and pred.get("away_xg"):
                print(f"  xG: Home {pred['home_xg']:.2f} - Away {pred['away_xg']:.2f}")

            if pred.get("betting_recommendation"):
                print(f"  {pred['betting_recommendation']}")

        print(f"\n{len(high_conf)}/{len(predictions)} matches with MEDIUM-HIGH+ confidence")
        return 0

    # Full output
    print_predictions(output)

    # Print file location
    print(f"\nPredictions saved to: {DATA_DIR / 'upcoming' / 'predictions.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
