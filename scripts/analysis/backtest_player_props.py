"""
Backtest the Player Prop Value Scanner
=======================================
Simulates historical betting on anytime goalscorer markets using:
- Model predictions from CatBoost + Platt scaling
- Simulated bookmaker odds (career goal rate + margin)
- Tier-based filtering matching production rules

Results (2024-25 + 2025-26 test set):
  Tier A (forwards, 3-10% edge): +18.6% ROI on 161 bets
  Tier B (all positions, 3-20% edge): -9.3% ROI on 652 bets
  Recommendation: Only bet Tier A (forwards with small edges)
"""

import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from catboost import CatBoostClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

from scripts.betting.player_predictions import (
    load_player_data, build_player_features, get_feature_cols, MODEL_DIR
)


def run_backtest(margin: float = 0.08, bankroll: float = 1000.0):
    """Run full backtest of player prop value scanner."""
    pms = load_player_data()
    pms = build_player_features(pms)

    df = pms[(pms["is_starter"]) & (pms["minutes"] >= 45) & (pms["position"] != "G")].copy()
    test_m = df["season"].isin(["2024-2025", "2025-2026"])
    df_test = df[test_m].copy()
    log.info(f"Test set: {len(df_test)} player-matches")

    # Load model + calibrator
    model = CatBoostClassifier()
    model.load_model(str(MODEL_DIR / "player_goalscorer.cbm"))
    with open(str(MODEL_DIR / "player_goalscorer_platt.pkl"), "rb") as f:
        platt = pickle.load(f)

    model_feats = model.feature_names_
    X_test = df_test[[f for f in model_feats if f in df_test.columns]].fillna(0)
    for f in model_feats:
        if f not in X_test.columns:
            X_test[f] = 0
    X_test = X_test[model_feats]

    raw = model.predict_proba(X_test)[:, 1]
    log_odds = np.log(np.clip(raw, 1e-7, 1 - 1e-7) / (1 - np.clip(raw, 1e-7, 1 - 1e-7)))
    model_prob = platt.predict_proba(log_odds.reshape(-1, 1))[:, 1]
    actual_goal = (df_test["goals"] > 0).astype(int).values
    positions = df_test["position"].values

    # Simulate bookmaker odds
    pos_base_rates = {
        p: df_test[df_test["position"] == p]["goals"].gt(0).mean()
        for p in ["F", "M", "D"]
    }
    career_goal_rate = df_test["career_goals_mean"].fillna(0).values
    bookie_prob = np.where(
        career_goal_rate > 0.01,
        career_goal_rate,
        [pos_base_rates.get(p, 0.09) for p in positions],
    )
    bookie_prob_with_margin = bookie_prob * (1 + margin)
    bookie_odds = 1.0 / np.clip(bookie_prob_with_margin, 0.02, 0.95)
    bookie_implied = 1.0 / bookie_odds
    edge = (model_prob - bookie_implied) / bookie_implied * 100

    # Tier A: forwards only, edge 3-10%
    tier_a = (
        (edge >= 3) & (edge <= 10)
        & (np.array([p == "F" for p in positions]))
        & (bookie_odds < 20)
        & (model_prob >= 0.05)
    )
    # Tier B: edge 3-20%, any position
    tier_b = (
        (edge >= 3) & (edge <= 20)
        & (bookie_odds < 20)
        & (model_prob >= 0.05)
        & ~tier_a
    )

    stake = bankroll * 0.01  # 1% of bankroll per bet
    results = {}
    for name, mask in [("Tier A", tier_a), ("Tier B", tier_b), ("A+B", tier_a | tier_b)]:
        n = mask.sum()
        if n == 0:
            continue
        returns = np.where(
            actual_goal[mask], (bookie_odds[mask] - 1) * stake, -stake
        )
        results[name] = {
            "n_bets": n,
            "win_rate": actual_goal[mask].mean(),
            "avg_odds": bookie_odds[mask].mean(),
            "roi": returns.sum() / (n * stake) * 100,
            "profit": returns.sum(),
        }
        log.info(
            f"  {name}: {n} bets, WR={results[name]['win_rate']:.1%}, "
            f"ROI={results[name]['roi']:+.1f}%, Profit=€{results[name]['profit']:+.0f}"
        )

    return results


if __name__ == "__main__":
    run_backtest()
