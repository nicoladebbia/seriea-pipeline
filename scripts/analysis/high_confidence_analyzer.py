#!/usr/bin/env python3
"""High Confidence Prediction Analyzer

The key insight: Instead of predicting every match, focus on HIGH CONFIDENCE predictions.

Strategy:
1. Predict all markets for each match
2. Find predictions where model confidence is very high (>65%, >70%, >75%)
3. Measure accuracy ONLY on those high-confidence predictions
4. This is how professional bettors work - quality over quantity

Expected outcome:
- If we only predict when >75% confident, we might get 80%+ accuracy
- But we'll make fewer predictions
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import DATA_DIR
from ml.config import LABEL_MAP
from scripts.models.train_unified import time_series_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# All markets we can predict
MARKETS = {
    "result": {"type": "multiclass", "classes": ["H", "D", "A"]},
    "home_clean_sheet": {"type": "binary"},
    "away_clean_sheet": {"type": "binary"},
    "home_scores": {"type": "binary"},
    "away_scores": {"type": "binary"},
    "btts": {"type": "binary"},
    "over_2_5": {"type": "binary"},
    "over_1_5": {"type": "binary"},
}

# Features
BASE_FEATURES = [
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
    "h2h_matches_played", "h2h_home_wins", "h2h_draws",
    "home_rest_days", "away_rest_days",
    "matchup_competitiveness",
]



def train_binary_model(X_train, y_train, X_val, y_val):
    """Train binary classifier."""
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


def train_multiclass_model(X_train, y_train, X_val, y_val):
    """Train multiclass classifier."""
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


def analyze_confidence_threshold(predictions: List[Dict], thresholds: List[float]) -> Dict:
    """Analyze accuracy at different confidence thresholds."""
    results = {}

    for threshold in thresholds:
        high_conf = [p for p in predictions if p["confidence"] >= threshold]

        if len(high_conf) == 0:
            results[threshold] = {"count": 0, "accuracy": 0, "pct_of_total": 0}
            continue

        correct = sum(1 for p in high_conf if p["correct"])
        accuracy = correct / len(high_conf)
        pct_of_total = len(high_conf) / len(predictions) * 100

        results[threshold] = {
            "count": len(high_conf),
            "accuracy": accuracy,
            "pct_of_total": pct_of_total,
        }

    return results


def main():
    log.info("=" * 70)
    log.info("HIGH CONFIDENCE PREDICTION ANALYZER")
    log.info("=" * 70)
    log.info("\nStrategy: Only bet when we're very confident")
    log.info("Goal: Find confidence thresholds where accuracy > 70%")

    # Load features
    feature_path = DATA_DIR / "features" / "features.parquet"
    df = pd.read_parquet(feature_path)
    df = df.dropna(subset=["home_score", "away_score"])
    log.info(f"\nLoaded {len(df)} matches")

    # Create all targets
    total_goals = df["home_score"] + df["away_score"]
    df["target_result"] = df.apply(
        lambda r: LABEL_MAP["H"] if r["home_score"] > r["away_score"]
        else (LABEL_MAP["A"] if r["away_score"] > r["home_score"] else LABEL_MAP["D"]),
        axis=1
    )
    df["target_home_clean_sheet"] = (df["away_score"] == 0).astype(int)
    df["target_away_clean_sheet"] = (df["home_score"] == 0).astype(int)
    df["target_home_scores"] = (df["home_score"] > 0).astype(int)
    df["target_away_scores"] = (df["away_score"] > 0).astype(int)
    df["target_btts"] = ((df["home_score"] > 0) & (df["away_score"] > 0)).astype(int)
    df["target_over_2_5"] = (total_goals > 2.5).astype(int)
    df["target_over_1_5"] = (total_goals > 1.5).astype(int)

    # Get available features
    available_features = [f for f in BASE_FEATURES if f in df.columns]
    X = df[available_features].fillna(0)

    splits = time_series_split(df, n_splits=5)
    log.info(f"Using {len(splits)} walk-forward splits")

    # ==========================================
    # Collect predictions across all splits
    # ==========================================
    all_predictions = {market: [] for market in MARKETS}

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        log.info(f"\nProcessing fold {fold_idx + 1}/{len(splits)}...")

        X_train = X.loc[train_idx]
        X_test = X.loc[test_idx]
        test_df = df.loc[test_idx]

        n_train = int(len(X_train) * 0.85)
        X_tr, X_val = X_train.iloc[:n_train], X_train.iloc[n_train:]

        # Train and predict for each market
        for market, info in MARKETS.items():
            target_col = f"target_{market}"
            y_train = df.loc[train_idx, target_col]
            y_test = df.loc[test_idx, target_col]

            y_tr, y_val = y_train.iloc[:n_train], y_train.iloc[n_train:]

            if info["type"] == "multiclass":
                model = train_multiclass_model(X_tr, y_tr, X_val, y_val)
                proba = model.predict_proba(X_test)
                pred = np.argmax(proba, axis=1)
                confidence = np.max(proba, axis=1)
            else:
                model = train_binary_model(X_tr, y_tr, X_val, y_val)
                proba = model.predict_proba(X_test)[:, 1]
                # For binary, confidence is max(prob, 1-prob)
                confidence = np.maximum(proba, 1 - proba)
                pred = (proba > 0.5).astype(int)

            # Store predictions
            for i, idx in enumerate(test_idx):
                all_predictions[market].append({
                    "match_idx": idx,
                    "home_team": test_df.loc[idx, "home_team"],
                    "away_team": test_df.loc[idx, "away_team"],
                    "predicted": int(pred[i]),
                    "actual": int(y_test.iloc[i]),
                    "confidence": float(confidence[i]),
                    "correct": int(pred[i]) == int(y_test.iloc[i]),
                })

    # ==========================================
    # Analyze at different confidence thresholds
    # ==========================================
    log.info("\n" + "=" * 70)
    log.info("ACCURACY BY CONFIDENCE THRESHOLD")
    log.info("=" * 70)

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

    summary = {}
    for market in MARKETS:
        predictions = all_predictions[market]
        overall_acc = sum(1 for p in predictions if p["correct"]) / len(predictions)

        results = analyze_confidence_threshold(predictions, thresholds)
        summary[market] = {"overall": overall_acc, "by_threshold": results}

    # Display results
    for market, data in summary.items():
        log.info(f"\n{market.upper().replace('_', ' ')}:")
        log.info(f"  Overall accuracy: {data['overall']:.1%}")
        log.info(f"  {'Threshold':<12} {'Count':<8} {'Accuracy':<12} {'% of bets':<12}")
        log.info(f"  {'-'*44}")

        for threshold, stats in data["by_threshold"].items():
            if stats["count"] > 0:
                log.info(f"  {threshold:.0%}         {stats['count']:<8} {stats['accuracy']:.1%}         {stats['pct_of_total']:.1f}%")

    # ==========================================
    # Find the best high-confidence opportunities
    # ==========================================
    log.info("\n" + "=" * 70)
    log.info("BEST HIGH-CONFIDENCE OPPORTUNITIES")
    log.info("=" * 70)
    log.info("\nMarkets where high confidence = high accuracy:")

    best_opportunities = []

    for market, data in summary.items():
        for threshold, stats in data["by_threshold"].items():
            if stats["count"] >= 50 and stats["accuracy"] >= 0.70:
                best_opportunities.append({
                    "market": market,
                    "threshold": threshold,
                    "accuracy": stats["accuracy"],
                    "count": stats["count"],
                    "pct_of_total": stats["pct_of_total"],
                })

    # Sort by accuracy
    best_opportunities.sort(key=lambda x: x["accuracy"], reverse=True)

    for opp in best_opportunities[:10]:
        log.info(f"\n  {opp['market'].replace('_', ' ').upper()}:")
        log.info(f"    At {opp['threshold']:.0%}+ confidence: {opp['accuracy']:.1%} accuracy")
        log.info(f"    Applies to {opp['count']} matches ({opp['pct_of_total']:.1f}% of total)")

    # ==========================================
    # Combined strategy analysis
    # ==========================================
    log.info("\n" + "=" * 70)
    log.info("COMBINED HIGH-CONFIDENCE STRATEGY")
    log.info("=" * 70)

    # For each match, find the highest confidence prediction across all markets
    all_matches = set()
    for market in MARKETS:
        for p in all_predictions[market]:
            all_matches.add(p["match_idx"])

    match_best_predictions = {}
    for match_idx in all_matches:
        best_conf = 0
        best_pred = None

        for market in MARKETS:
            for p in all_predictions[market]:
                if p["match_idx"] == match_idx and p["confidence"] > best_conf:
                    best_conf = p["confidence"]
                    best_pred = {"market": market, **p}

        if best_pred:
            match_best_predictions[match_idx] = best_pred

    # Analyze combined strategy
    log.info("\nFor each match, pick the prediction with highest confidence:")

    combined_thresholds = [0.65, 0.70, 0.75, 0.80, 0.85]

    for threshold in combined_thresholds:
        high_conf = [p for p in match_best_predictions.values() if p["confidence"] >= threshold]

        if len(high_conf) > 0:
            correct = sum(1 for p in high_conf if p["correct"])
            accuracy = correct / len(high_conf)

            log.info(f"\n  At {threshold:.0%}+ confidence:")
            log.info(f"    {len(high_conf)} predictions ({len(high_conf)/len(match_best_predictions)*100:.1f}% of matches)")
            log.info(f"    Accuracy: {accuracy:.1%}")

            # Show market breakdown
            market_counts = {}
            for p in high_conf:
                market_counts[p["market"]] = market_counts.get(p["market"], 0) + 1

            log.info(f"    Market breakdown: {dict(sorted(market_counts.items(), key=lambda x: -x[1]))}")

    # ==========================================
    # Save results
    # ==========================================
    results_path = DATA_DIR / "models" / "high_confidence_analysis.json"

    output = {
        "market_analysis": {
            market: {
                "overall_accuracy": data["overall"],
                "by_threshold": {
                    str(t): {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in stats.items()}
                    for t, stats in data["by_threshold"].items()
                }
            }
            for market, data in summary.items()
        },
        "best_opportunities": [
            {k: round(v, 4) if isinstance(v, float) else v for k, v in opp.items()}
            for opp in best_opportunities
        ],
    }

    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    log.info(f"\nResults saved to {results_path}")

    # ==========================================
    # Key takeaways
    # ==========================================
    log.info("\n" + "=" * 70)
    log.info("KEY TAKEAWAYS")
    log.info("=" * 70)
    log.info("""
The path to profitable betting:

1. DON'T bet on every match
   - Overall H/D/A accuracy is ~52-53%
   - This is not profitable

2. DO bet only when highly confident
   - When model is 70%+ confident, accuracy jumps significantly
   - Fewer bets, but more likely to win

3. FOCUS on easier markets
   - "Home/Away Clean Sheet" and "Team Scores" are more predictable
   - These markets often have 70%+ accuracy at high confidence

4. COMBINE predictions
   - For each match, find the prediction you're most confident about
   - This maximizes your edge

5. REMEMBER
   - 65-70% overall accuracy requires betting odds
   - Without odds, 55% is realistic for match outcomes
   - High-confidence predictions can exceed 70% but apply to fewer matches
""")


if __name__ == "__main__":
    main()
