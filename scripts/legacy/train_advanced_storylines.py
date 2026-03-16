#!/usr/bin/env python3
"""Advanced Storyline Predictions - Testing Football Myths!

This script predicts "story-driven" metrics that test football myths:

1. FORMATION BATTLES - Which tactical setup wins?
2. REVENGE MATCHES - Does the loser come back stronger?
3. NEW MANAGER BOUNCE - Does changing manager work?
4. FIRST 15 MINUTES - Who starts stronger?
5. FERGIE TIME - Who scores/concedes late?
6. REFEREE BIAS - Do certain refs favor certain teams?
7. TRAVEL FATIGUE - Does long travel hurt performance?
8. POST-BIG-WIN LETDOWN - Do teams drop after big wins?

Expected accuracy: 50-60% (these are complex factors!)
But even 55% on these "story" predictions is valuable insight.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def time_series_split(df: pd.DataFrame, n_splits: int = 5):
    """Walk-forward time series splits."""
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


def train_classifier(X_train, y_train, X_val, y_val):
    """Train classifier."""
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


def evaluate_model(df, X, y, splits, name):
    """Evaluate with confidence analysis."""
    all_preds = []
    all_true = []
    all_proba = []

    for train_idx, test_idx in splits:
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]

        n = int(len(X_train) * 0.85)
        model = train_classifier(
            X_train.iloc[:n], y_train.iloc[:n],
            X_train.iloc[n:], y_train.iloc[n:]
        )

        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba > 0.5).astype(int)

        all_preds.extend(pred.tolist())
        all_true.extend(y_test.tolist())
        all_proba.extend(proba.tolist())

    acc = accuracy_score(all_true, all_preds)
    base_rate = sum(all_true) / len(all_true) if all_true else 0.5
    skill = acc - max(base_rate, 1 - base_rate)

    # High confidence analysis
    high_conf = {}
    for threshold in [0.60, 0.65, 0.70]:
        hc = [(t, p, prob) for t, p, prob in zip(all_true, all_preds, all_proba)
              if max(prob, 1-prob) >= threshold]
        if hc:
            correct = sum(1 for t, p, _ in hc if t == p)
            high_conf[threshold] = {
                "count": len(hc),
                "accuracy": correct / len(hc),
                "pct": len(hc) / len(all_true) * 100,
            }

    return {
        "accuracy": acc,
        "base_rate": base_rate,
        "skill": skill,
        "n_samples": len(all_true),
        "high_conf": high_conf,
    }


def main():
    log.info("=" * 70)
    log.info("ADVANCED STORYLINE PREDICTIONS")
    log.info("=" * 70)
    log.info("\nTesting football myths with data!")

    # Load data
    feature_path = DATA_DIR / "features" / "features.parquet"
    df = pd.read_parquet(feature_path)
    df = df.dropna(subset=["home_score", "away_score"])
    log.info(f"Loaded {len(df)} matches")

    # Sort by date for proper sequence analysis
    df = df.sort_values("match_date").reset_index(drop=True)

    # Base features available
    base_features = [f for f in [
        "home_elo", "away_elo", "elo_diff",
        "home_attack_strength", "home_defense_strength",
        "away_attack_strength", "away_defense_strength",
        "home_form_points_3", "home_form_points_5",
        "away_form_points_3", "away_form_points_5",
        "league_position_diff",
        "matchup_competitiveness",
    ] if f in df.columns]

    results = {}
    splits = time_series_split(df, n_splits=5)

    # ==========================================
    # 1. REVENGE MATCH PREDICTION
    # ==========================================
    log.info("\n" + "=" * 60)
    log.info("1. REVENGE MATCH PREDICTION")
    log.info("=" * 60)
    log.info("Does the team that lost last meeting come back stronger?")

    if "h2h_last_result" in df.columns:
        # Create revenge target: did the previous loser win this time?
        df["revenge_match"] = 0

        # h2h_last_result: 1 = home won last, -1 = away won last, 0 = draw
        # Revenge = previous loser wins now
        home_win = (df["home_score"] > df["away_score"]).astype(int)
        away_win = (df["away_score"] > df["home_score"]).astype(int)

        # If away won last (h2h_last_result = -1), and home wins now = revenge
        # If home won last (h2h_last_result = 1), and away wins now = revenge
        df["revenge_success"] = (
            ((df["h2h_last_result"] == -1) & (home_win == 1)) |
            ((df["h2h_last_result"] == 1) & (away_win == 1))
        ).astype(int)

        # Only analyze matches where there was a previous winner (not draws)
        revenge_df = df[df["h2h_last_result"] != 0].copy()

        if len(revenge_df) > 1000:
            revenge_features = base_features + [f for f in [
                "h2h_matches_played", "h2h_home_wins", "h2h_away_wins",
                "h2h_last_result",
            ] if f in revenge_df.columns]

            X = revenge_df[revenge_features].fillna(0)
            y = revenge_df["revenge_success"]
            revenge_splits = time_series_split(revenge_df, n_splits=5)

            if revenge_splits:
                result = evaluate_model(revenge_df, X, y, revenge_splits, "Revenge")
                results["revenge_match"] = result

                log.info(f"  Base rate (revenge success): {result['base_rate']:.1%}")
                log.info(f"  Model accuracy: {result['accuracy']:.1%}")
                log.info(f"  Skill: {result['skill']:+.1%}")

                # Insight
                log.info(f"\n  INSIGHT: Previous losers win back {result['base_rate']:.1%} of the time")
                if result['base_rate'] > 0.35:
                    log.info("  → Revenge motivation IS a real factor!")
                else:
                    log.info("  → Revenge factor is MYTH - previous winner maintains edge")
    else:
        log.info("  No H2H data available")

    # ==========================================
    # 2. POST-BIG-WIN LETDOWN
    # ==========================================
    log.info("\n" + "=" * 60)
    log.info("2. POST-BIG-WIN LETDOWN")
    log.info("=" * 60)
    log.info("Do teams drop performance after massive victories?")

    # Create feature: did team win big (3+ goals) in previous match?
    # Then check if they underperform in next match

    # We need to track previous match results per team
    # For simplicity, use rolling goal difference
    if "home_roll_3_goals_scored" in df.columns:
        # High scoring recent = potential letdown
        df["home_high_scorer"] = (df["home_roll_3_goals_scored"] > 2.0).astype(int)
        df["away_high_scorer"] = (df["away_roll_3_goals_scored"] > 2.0).astype(int)

        # Did they fail to win?
        df["home_letdown"] = df["home_high_scorer"] & (df["home_score"] <= df["away_score"])
        df["away_letdown"] = df["away_high_scorer"] & (df["away_score"] <= df["home_score"])

        # Analyze home letdown
        letdown_df = df[df["home_high_scorer"] == 1].copy()

        if len(letdown_df) > 500:
            X = letdown_df[base_features].fillna(0)
            y = letdown_df["home_letdown"].astype(int)
            letdown_splits = time_series_split(letdown_df, n_splits=5)

            if letdown_splits:
                result = evaluate_model(letdown_df, X, y, letdown_splits, "Letdown")
                results["post_big_win_letdown"] = result

                log.info(f"  High-scoring teams that DON'T win next: {result['base_rate']:.1%}")
                log.info(f"  Model accuracy: {result['accuracy']:.1%}")

                if result['base_rate'] > 0.40:
                    log.info("\n  INSIGHT: Post-big-win letdown IS real!")
                else:
                    log.info("\n  INSIGHT: Momentum carries over - no letdown effect")

    # ==========================================
    # 3. NEW MANAGER BOUNCE
    # ==========================================
    log.info("\n" + "=" * 60)
    log.info("3. NEW MANAGER BOUNCE")
    log.info("=" * 60)
    log.info("Do teams improve immediately after manager change?")

    if "home_manager_is_new" in df.columns:
        new_manager_df = df[df["home_manager_is_new"] == 1].copy()

        if len(new_manager_df) > 100:
            # Did they win?
            new_manager_df["new_manager_win"] = (
                new_manager_df["home_score"] > new_manager_df["away_score"]
            ).astype(int)

            base_rate = new_manager_df["new_manager_win"].mean()
            overall_home_win_rate = (df["home_score"] > df["away_score"]).mean()

            log.info(f"  New manager matches: {len(new_manager_df)}")
            log.info(f"  New manager home win rate: {base_rate:.1%}")
            log.info(f"  Overall home win rate: {overall_home_win_rate:.1%}")

            bounce = base_rate - overall_home_win_rate
            results["new_manager_bounce"] = {
                "new_manager_win_rate": base_rate,
                "overall_win_rate": overall_home_win_rate,
                "bounce_effect": bounce,
            }

            if bounce > 0.05:
                log.info(f"\n  INSIGHT: New manager bounce IS real! +{bounce:.1%}")
            else:
                log.info(f"\n  INSIGHT: New manager bounce is MYTH ({bounce:+.1%})")
    else:
        log.info("  No manager data available")

    # ==========================================
    # 4. TRAVEL FATIGUE IMPACT
    # ==========================================
    log.info("\n" + "=" * 60)
    log.info("4. TRAVEL FATIGUE IMPACT")
    log.info("=" * 60)
    log.info("Does long travel hurt away team performance?")

    if "travel_distance_km" in df.columns:
        # Long travel = 500+ km
        df["long_travel"] = (df["travel_distance_km"] > 500).astype(int)

        long_travel_df = df[df["long_travel"] == 1].copy()
        short_travel_df = df[df["long_travel"] == 0].copy()

        if len(long_travel_df) > 500:
            long_away_win = (long_travel_df["away_score"] > long_travel_df["home_score"]).mean()
            short_away_win = (short_travel_df["away_score"] > short_travel_df["home_score"]).mean()

            log.info(f"  Long travel (500+ km) matches: {len(long_travel_df)}")
            log.info(f"  Away win rate with long travel: {long_away_win:.1%}")
            log.info(f"  Away win rate with short travel: {short_away_win:.1%}")

            fatigue_effect = short_away_win - long_away_win
            results["travel_fatigue"] = {
                "long_travel_away_win": long_away_win,
                "short_travel_away_win": short_away_win,
                "fatigue_effect": fatigue_effect,
            }

            if fatigue_effect > 0.03:
                log.info(f"\n  INSIGHT: Travel fatigue IS real! -{fatigue_effect:.1%} for away team")
            else:
                log.info(f"\n  INSIGHT: Travel fatigue is MINIMAL ({fatigue_effect:+.1%})")

    # ==========================================
    # 5. REFEREE STRICTNESS IMPACT
    # ==========================================
    log.info("\n" + "=" * 60)
    log.info("5. REFEREE STRICTNESS IMPACT")
    log.info("=" * 60)
    log.info("Do strict referees change match dynamics?")

    if "ref_strictness_score" in df.columns:
        # Strict ref = above 0.6 strictness
        df["strict_ref"] = (df["ref_strictness_score"] > 0.6).astype(int)

        strict_df = df[df["strict_ref"] == 1].copy()
        lenient_df = df[df["strict_ref"] == 0].copy()

        if len(strict_df) > 200:
            # Do strict refs change goal patterns?
            strict_goals = (strict_df["home_score"] + strict_df["away_score"]).mean()
            lenient_goals = (lenient_df["home_score"] + lenient_df["away_score"]).mean()

            strict_yellows = (strict_df["home_yellow_cards"].fillna(0) + strict_df["away_yellow_cards"].fillna(0)).mean() if "home_yellow_cards" in strict_df.columns else 0
            lenient_yellows = (lenient_df["home_yellow_cards"].fillna(0) + lenient_df["away_yellow_cards"].fillna(0)).mean() if "home_yellow_cards" in lenient_df.columns else 0

            log.info(f"  Strict referee matches: {len(strict_df)}")
            log.info(f"  Avg goals with strict ref: {strict_goals:.2f}")
            log.info(f"  Avg goals with lenient ref: {lenient_goals:.2f}")
            log.info(f"  Avg yellows with strict ref: {strict_yellows:.2f}")
            log.info(f"  Avg yellows with lenient ref: {lenient_yellows:.2f}")

            results["referee_strictness"] = {
                "strict_avg_goals": strict_goals,
                "lenient_avg_goals": lenient_goals,
                "strict_avg_yellows": strict_yellows,
                "lenient_avg_yellows": lenient_yellows,
            }

            if strict_yellows - lenient_yellows > 1.0:
                log.info(f"\n  INSIGHT: Strict refs give {strict_yellows - lenient_yellows:.1f} more cards!")

    # ==========================================
    # 6. HOME ADVANTAGE BY STADIUM SIZE
    # ==========================================
    log.info("\n" + "=" * 60)
    log.info("6. HOME ADVANTAGE BY STADIUM SIZE")
    log.info("=" * 60)
    log.info("Do bigger stadiums = bigger home advantage?")

    if "home_stadium_capacity" in df.columns:
        # Large stadium = 50,000+
        df["large_stadium"] = (df["home_stadium_capacity"] > 50000).astype(int)

        large_df = df[df["large_stadium"] == 1].copy()
        small_df = df[df["large_stadium"] == 0].copy()

        if len(large_df) > 500:
            large_home_win = (large_df["home_score"] > large_df["away_score"]).mean()
            small_home_win = (small_df["home_score"] > small_df["away_score"]).mean()

            log.info(f"  Large stadium (50k+) matches: {len(large_df)}")
            log.info(f"  Home win rate at large stadiums: {large_home_win:.1%}")
            log.info(f"  Home win rate at small stadiums: {small_home_win:.1%}")

            stadium_effect = large_home_win - small_home_win
            results["stadium_size_effect"] = {
                "large_stadium_home_win": large_home_win,
                "small_stadium_home_win": small_home_win,
                "effect": stadium_effect,
            }

            if stadium_effect > 0.03:
                log.info(f"\n  INSIGHT: Big stadium = bigger home advantage! +{stadium_effect:.1%}")
            else:
                log.info(f"\n  INSIGHT: Stadium size doesn't matter ({stadium_effect:+.1%})")

    # ==========================================
    # 7. FORM REVERSAL PREDICTION
    # ==========================================
    log.info("\n" + "=" * 60)
    log.info("7. FORM REVERSAL PREDICTION")
    log.info("=" * 60)
    log.info("When does good/bad form break?")

    if "home_form_points_5" in df.columns:
        # Great form = 12+ points from last 5 (avg 2.4 per game)
        df["home_great_form"] = (df["home_form_points_5"] >= 12).astype(int)
        df["away_great_form"] = (df["away_form_points_5"] >= 12).astype(int)

        # Did great form team fail to win?
        df["form_break"] = (
            (df["home_great_form"] == 1) & (df["home_score"] <= df["away_score"])
        ).astype(int)

        great_form_df = df[df["home_great_form"] == 1].copy()

        if len(great_form_df) > 300:
            form_break_rate = great_form_df["form_break"].mean()

            log.info(f"  Teams in great form (12+ pts from 5): {len(great_form_df)}")
            log.info(f"  % that fail to win: {form_break_rate:.1%}")

            results["form_reversal"] = {
                "great_form_failure_rate": form_break_rate,
            }

            if form_break_rate > 0.30:
                log.info(f"\n  INSIGHT: Form is TEMPORARY! {form_break_rate:.1%} fail despite great form")
            else:
                log.info(f"\n  INSIGHT: Form is REAL - only {form_break_rate:.1%} fail")

    # ==========================================
    # 8. DERBY INTENSITY IMPACT
    # ==========================================
    log.info("\n" + "=" * 60)
    log.info("8. DERBY/RIVALRY INTENSITY")
    log.info("=" * 60)
    log.info("Do derbies produce different patterns?")

    # Check if we have derby data
    if "matchup_competitiveness" in df.columns:
        # High competitiveness = derby/rivalry
        df["is_derby"] = (df["matchup_competitiveness"] > 0.8).astype(int)

        derby_df = df[df["is_derby"] == 1].copy()
        normal_df = df[df["is_derby"] == 0].copy()

        if len(derby_df) > 100:
            derby_goals = (derby_df["home_score"] + derby_df["away_score"]).mean()
            normal_goals = (normal_df["home_score"] + normal_df["away_score"]).mean()

            derby_draws = (derby_df["home_score"] == derby_df["away_score"]).mean()
            normal_draws = (normal_df["home_score"] == normal_df["away_score"]).mean()

            derby_yellows = 0
            normal_yellows = 0
            if "home_yellow_cards" in df.columns:
                derby_yellows = (derby_df["home_yellow_cards"].fillna(0) + derby_df["away_yellow_cards"].fillna(0)).mean()
                normal_yellows = (normal_df["home_yellow_cards"].fillna(0) + normal_df["away_yellow_cards"].fillna(0)).mean()

            log.info(f"  Derby matches: {len(derby_df)}")
            log.info(f"  Avg goals in derbies: {derby_goals:.2f} vs normal: {normal_goals:.2f}")
            log.info(f"  Draw rate in derbies: {derby_draws:.1%} vs normal: {normal_draws:.1%}")
            log.info(f"  Avg yellows in derbies: {derby_yellows:.2f} vs normal: {normal_yellows:.2f}")

            results["derby_effect"] = {
                "derby_avg_goals": derby_goals,
                "normal_avg_goals": normal_goals,
                "derby_draw_rate": derby_draws,
                "normal_draw_rate": normal_draws,
                "derby_avg_yellows": derby_yellows,
                "normal_avg_yellows": normal_yellows,
            }

            if derby_yellows > normal_yellows + 0.5:
                log.info(f"\n  INSIGHT: Derbies ARE more intense! +{derby_yellows - normal_yellows:.1f} cards")

    # ==========================================
    # 9. SCORING DROUGHT BREAKER
    # ==========================================
    log.info("\n" + "=" * 60)
    log.info("9. SCORING DROUGHT BREAKER PREDICTION")
    log.info("=" * 60)
    log.info("When does a team in a scoring drought finally score?")

    if "home_roll_3_goals_scored" in df.columns:
        # Drought = scored 0 or 1 goal in last 3 matches
        df["home_drought"] = (df["home_roll_3_goals_scored"] <= 1.0).astype(int)
        df["away_drought"] = (df["away_roll_3_goals_scored"] <= 1.0).astype(int)

        # Did they score?
        df["home_drought_broken"] = (df["home_drought"] == 1) & (df["home_score"] > 0)
        df["away_drought_broken"] = (df["away_drought"] == 1) & (df["away_score"] > 0)

        drought_df = df[df["home_drought"] == 1].copy()

        if len(drought_df) > 300:
            drought_break_rate = drought_df["home_drought_broken"].mean()

            log.info(f"  Teams in scoring drought: {len(drought_df)}")
            log.info(f"  % that score anyway: {drought_break_rate:.1%}")

            results["scoring_drought"] = {
                "n_matches": len(drought_df),
                "break_rate": drought_break_rate,
            }

            if drought_break_rate > 0.70:
                log.info(f"\n  INSIGHT: Droughts usually end - {drought_break_rate:.1%} still score!")
            else:
                log.info(f"\n  INSIGHT: Droughts persist - only {drought_break_rate:.1%} break out")

    # ==========================================
    # 10. GOAL DIFFERENCE IMPACT
    # ==========================================
    log.info("\n" + "=" * 60)
    log.info("10. HIGH-SCORING vs LOW-SCORING MATCHUPS")
    log.info("=" * 60)
    log.info("Does attack vs defense create different patterns?")

    if "home_attack_strength" in df.columns and "away_defense_strength" in df.columns:
        # High attack vs weak defense
        df["attack_mismatch"] = (
            (df["home_attack_strength"] > 1.2) & (df["away_defense_strength"] < 0.8)
        ).astype(int)

        mismatch_df = df[df["attack_mismatch"] == 1].copy()
        normal_df = df[df["attack_mismatch"] == 0].copy()

        if len(mismatch_df) > 200:
            mismatch_goals = (mismatch_df["home_score"] + mismatch_df["away_score"]).mean()
            mismatch_home_win = (mismatch_df["home_score"] > mismatch_df["away_score"]).mean()
            normal_goals = (normal_df["home_score"] + normal_df["away_score"]).mean()
            normal_home_win = (normal_df["home_score"] > normal_df["away_score"]).mean()

            log.info(f"  Attack mismatch matches: {len(mismatch_df)}")
            log.info(f"  Avg goals in mismatch: {mismatch_goals:.2f} vs normal: {normal_goals:.2f}")
            log.info(f"  Home win in mismatch: {mismatch_home_win:.1%} vs normal: {normal_home_win:.1%}")

            results["attack_mismatch"] = {
                "n_matches": len(mismatch_df),
                "mismatch_goals": mismatch_goals,
                "normal_goals": normal_goals,
                "mismatch_home_win": mismatch_home_win,
                "normal_home_win": normal_home_win,
            }

    # ==========================================
    # 11. MOMENTUM ANALYSIS
    # ==========================================
    log.info("\n" + "=" * 60)
    log.info("11. MOMENTUM CLASH")
    log.info("=" * 60)
    log.info("Hot team vs cold team - who wins?")

    if "home_form_points_3" in df.columns and "away_form_points_3" in df.columns:
        # Hot = 7+ points from 3 (avg 2.33/game), Cold = 2 or less
        df["hot_vs_cold"] = (
            (df["home_form_points_3"] >= 7) & (df["away_form_points_3"] <= 2)
        ).astype(int)

        df["cold_vs_hot"] = (
            (df["home_form_points_3"] <= 2) & (df["away_form_points_3"] >= 7)
        ).astype(int)

        hot_home_df = df[df["hot_vs_cold"] == 1].copy()
        cold_home_df = df[df["cold_vs_hot"] == 1].copy()

        if len(hot_home_df) > 100:
            hot_home_win = (hot_home_df["home_score"] > hot_home_df["away_score"]).mean()
            cold_home_win = (cold_home_df["home_score"] > cold_home_df["away_score"]).mean()

            log.info(f"  Hot home vs cold away matches: {len(hot_home_df)}")
            log.info(f"  Hot team at home win rate: {hot_home_win:.1%}")
            log.info(f"  Cold team at home win rate: {cold_home_win:.1%}")

            results["momentum_clash"] = {
                "hot_home_win_rate": hot_home_win,
                "cold_home_win_rate": cold_home_win,
                "momentum_effect": hot_home_win - cold_home_win,
            }

            log.info(f"\n  INSIGHT: Momentum effect = +{hot_home_win - cold_home_win:.1%} for hot team")

    # ==========================================
    # 12. UNDERDOG ANALYSIS
    # ==========================================
    log.info("\n" + "=" * 60)
    log.info("12. UNDERDOG UPSET PREDICTION")
    log.info("=" * 60)
    log.info("When do underdogs pull off upsets?")

    if "elo_diff" in df.columns:
        # Big underdog = 200+ Elo difference
        df["big_underdog_home"] = (df["elo_diff"] < -200).astype(int)
        df["big_underdog_away"] = (df["elo_diff"] > 200).astype(int)

        underdog_home_df = df[df["big_underdog_home"] == 1].copy()
        underdog_away_df = df[df["big_underdog_away"] == 1].copy()

        if len(underdog_home_df) > 100:
            underdog_home_upset = (underdog_home_df["home_score"] > underdog_home_df["away_score"]).mean()
            underdog_away_upset = (underdog_away_df["away_score"] > underdog_away_df["home_score"]).mean()

            log.info(f"  Big underdog at home matches: {len(underdog_home_df)}")
            log.info(f"  Underdog home upset rate: {underdog_home_upset:.1%}")
            log.info(f"  Underdog away upset rate: {underdog_away_upset:.1%}")

            results["underdog_upset"] = {
                "n_home_underdog": len(underdog_home_df),
                "n_away_underdog": len(underdog_away_df),
                "home_upset_rate": underdog_home_upset,
                "away_upset_rate": underdog_away_upset,
            }

            log.info(f"\n  INSIGHT: Underdogs win {underdog_home_upset:.1%} at home, {underdog_away_upset:.1%} away")

    # ==========================================
    # 13. LATE SEASON IMPACT
    # ==========================================
    log.info("\n" + "=" * 60)
    log.info("13. LATE SEASON DYNAMICS")
    log.info("=" * 60)
    log.info("Do matches play differently at season end?")

    if "matchweek" in df.columns:
        # Late season = matchweek 30+
        df["late_season"] = (df["matchweek"] >= 30).astype(int)

        late_df = df[df["late_season"] == 1].copy()
        early_df = df[df["late_season"] == 0].copy()

        if len(late_df) > 500:
            late_goals = (late_df["home_score"] + late_df["away_score"]).mean()
            early_goals = (early_df["home_score"] + early_df["away_score"]).mean()

            late_draws = (late_df["home_score"] == late_df["away_score"]).mean()
            early_draws = (early_df["home_score"] == early_df["away_score"]).mean()

            log.info(f"  Late season matches: {len(late_df)}")
            log.info(f"  Avg goals late season: {late_goals:.2f} vs early: {early_goals:.2f}")
            log.info(f"  Draw rate late season: {late_draws:.1%} vs early: {early_draws:.1%}")

            results["late_season"] = {
                "late_avg_goals": late_goals,
                "early_avg_goals": early_goals,
                "late_draw_rate": late_draws,
                "early_draw_rate": early_draws,
            }

            if abs(late_goals - early_goals) > 0.2:
                log.info(f"\n  INSIGHT: Late season matches ARE different!")

    # ==========================================
    # SUMMARY
    # ==========================================
    log.info("\n" + "=" * 70)
    log.info("FOOTBALL MYTHS: VALIDATED OR BUSTED?")
    log.info("=" * 70)

    myths = [
        ("Revenge Factor", "revenge_match", "accuracy"),
        ("Post-Big-Win Letdown", "post_big_win_letdown", "base_rate"),
        ("New Manager Bounce", "new_manager_bounce", "bounce_effect"),
        ("Travel Fatigue", "travel_fatigue", "fatigue_effect"),
        ("Big Stadium Advantage", "stadium_size_effect", "effect"),
        ("Form is Temporary", "form_reversal", "great_form_failure_rate"),
        ("Scoring Drought Break", "scoring_drought", "break_rate"),
        ("Momentum Clash", "momentum_clash", "momentum_effect"),
        ("Underdog Upsets", "underdog_upset", "home_upset_rate"),
        ("Late Season Different", "late_season", "late_draw_rate"),
    ]

    log.info("\n  MYTH                       STATUS")
    log.info("  " + "-" * 50)

    for name, key, metric in myths:
        if key in results:
            value = results[key].get(metric, 0)
            if metric == "accuracy":
                status = "✓ VALIDATED" if value > 0.52 else "✗ BUSTED"
            elif metric in ["bounce_effect", "fatigue_effect", "effect", "momentum_effect"]:
                status = "✓ VALIDATED" if value > 0.03 else "✗ BUSTED"
            elif metric == "base_rate":
                status = "✓ VALIDATED" if value > 0.35 else "✗ BUSTED"
            elif metric == "great_form_failure_rate":
                status = "✓ FORM IS TEMPORARY" if value > 0.30 else "✓ FORM IS REAL"
            elif metric == "break_rate":
                status = f"✓ {value:.1%} break drought" if value > 0.50 else f"✗ Only {value:.1%}"
            elif metric == "home_upset_rate":
                status = f"Home: {value:.1%}"
            elif metric == "late_draw_rate":
                status = f"Draw rate: {value:.1%}"
            else:
                status = f"Value: {value:.2f}"
            log.info(f"  {name:<25} {status}")
        else:
            log.info(f"  {name:<25} No data")

    # Save results
    output_path = DATA_DIR / "models" / "storyline_analysis.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info(f"\n\nResults saved to {output_path}")

    log.info("""

KEY TAKEAWAYS:
These "story" predictions are valuable because:
1. They reveal hidden patterns in football
2. Even 55% accuracy is useful for complex factors
3. They help understand WHY matches go certain ways
4. Combined with statistical predictions, they add context

USE THESE FOR:
- Understanding match dynamics before betting
- Identifying when "form" or "revenge" matters
- Adjusting predictions based on context
- Creating narrative around predictions
""")


if __name__ == "__main__":
    main()
