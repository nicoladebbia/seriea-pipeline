#!/usr/bin/env python3
"""Time-decay sweep: test different decay values and compare metrics.

Tests decay values from 0.80 to 0.92 and measures impact on walk-forward CV.
Focuses on last-2-fold performance (most recent seasons = most relevant for production).

Usage:
    python -m scripts.analysis.decay_sweep                    # Serie A
    python -m scripts.analysis.decay_sweep --league premier_league
    python -m scripts.analysis.decay_sweep --model catboost   # Specific model
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

from ml.config import ModelConfig, ValidationConfig
from ml.data import DataLoader
from ml.training import walk_forward_validate
from storage.paths import features_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DECAY_VALUES = [0.80, 0.82, 0.85, 0.88, 0.90, 0.92]
DEFAULT_MODEL = "catboost"  # CatBoost has slight edge in ensemble


def run_sweep(league: str = "serie_a", model_type: str = DEFAULT_MODEL) -> pd.DataFrame:
    """Run decay sweep and return comparison DataFrame."""
    log.info("Loading data for %s...", league)
    fp = str(features_path(league=league))
    loader = DataLoader(fp)
    X, y, feature_names = loader.get_universal_dataset(league=league)
    log.info("Loaded %d matches, %d features", len(X), len(feature_names))

    results = []
    for decay in DECAY_VALUES:
        log.info("=" * 60)
        log.info("Testing decay=%.2f", decay)

        mc = ModelConfig()
        vc = ValidationConfig()

        # Get model params
        if model_type == "xgboost":
            params = mc.xgb_params
        elif model_type == "lightgbm":
            params = mc.lgb_params
        elif model_type == "catboost":
            params = mc.cb_params
        else:
            raise ValueError(f"Unknown model: {model_type}")

        # Run walk-forward CV with decay override passed through to sample weights
        fold_df = walk_forward_validate(
            X, y, model_type, config=vc, params=params,
            use_sample_weights=True, decay_override=decay,
        )

        # Compute metrics
        n_folds = len(fold_df)
        avg_all = fold_df.mean(numeric_only=True)
        last_2 = fold_df.tail(2).mean(numeric_only=True) if n_folds >= 2 else avg_all

        results.append({
            "decay": decay,
            "n_folds": n_folds,
            "avg_log_loss": round(avg_all["log_loss"], 4),
            "avg_accuracy": round(avg_all["accuracy"] * 100, 2),
            "avg_f1_D": round(avg_all.get("f1_D", 0), 4),
            "avg_brier": round(avg_all.get("brier_score", 0), 4),
            "last2_log_loss": round(last_2["log_loss"], 4),
            "last2_accuracy": round(last_2["accuracy"] * 100, 2),
            "last2_f1_D": round(last_2.get("f1_D", 0), 4),
        })

        log.info(
            "decay=%.2f | avg: acc=%.1f%% ll=%.4f | last2: acc=%.1f%% ll=%.4f",
            decay, results[-1]["avg_accuracy"], results[-1]["avg_log_loss"],
            results[-1]["last2_accuracy"], results[-1]["last2_log_loss"],
        )

    df = pd.DataFrame(results)
    return df


def main():
    parser = argparse.ArgumentParser(description="Time-decay sweep")
    from config.leagues import ACTIVE_LEAGUES
    parser.add_argument("--league", default="serie_a", choices=ACTIVE_LEAGUES)
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=["xgboost", "lightgbm", "catboost"])
    args = parser.parse_args()

    df = run_sweep(league=args.league, model_type=args.model)

    print("\n" + "=" * 70)
    print(f"DECAY SWEEP RESULTS — {args.league} / {args.model}")
    print("=" * 70)
    print(df.to_string(index=False))

    # Find best by last-2-fold log loss (most relevant for production)
    best = df.loc[df["last2_log_loss"].idxmin()]
    current = df[df["decay"] == 0.88].iloc[0] if 0.88 in df["decay"].values else None

    print(f"\nBest decay: {best['decay']:.2f} (last2 ll={best['last2_log_loss']:.4f})")
    if current is not None:
        improvement = current["last2_log_loss"] - best["last2_log_loss"]
        print(f"Current (0.88): last2 ll={current['last2_log_loss']:.4f}")
        print(f"Improvement: {improvement:.4f} ({'better' if improvement > 0 else 'no change or worse'})")

    # Save results
    out_path = Path("data/analysis") / f"decay_sweep_{args.league}_{args.model}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
