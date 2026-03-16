"""Contextual Feature Engineering - Expert-level match analysis features.

This module creates sophisticated features that capture how expert analysts think:

1. STRENGTH-ADJUSTED FORM
   - Weight results by opponent quality (beating Inter > beating Verona)
   - Adjust goals by opponent defensive/offensive strength

2. PERFORMANCE BY OPPONENT TIER
   - How does team perform vs top 6, mid-table, bottom 6?
   - Different patterns: some teams beat weak teams but struggle vs top

3. DEFENSIVE CONSISTENCY
   - Clean sheet streaks
   - Goals conceded vs quality attacks
   - Big game defensive record

4. MATCHUP ANALYSIS
   - Home attack strength vs Away defense weakness
   - Style matchups (possession vs counter-attack)

5. PRESSURE/IMPORTANCE FACTORS
   - Stadium capacity relative fill
   - Position gap (18th playing vs 4th)
   - Relegation/title pressure

6. MOMENTUM & TRENDS
   - Is team improving or declining?
   - Recent big result boost (beating top team)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def compute_opponent_strength(
    matches_df: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp,
    n_matches: int = 10,
) -> Dict[str, float]:
    """Compute strength-adjusted form by weighting results by opponent quality.

    A 3-0 win vs Inter (1st place) is worth more than 3-0 vs Verona (18th).
    """
    team_matches = matches_df[
        ((matches_df["home_team"] == team) | (matches_df["away_team"] == team)) &
        (matches_df["match_date"] < before_date)
    ].sort_values("match_date", ascending=False).head(n_matches)

    if team_matches.empty:
        return {
            "strength_adjusted_points": 0,
            "strength_adjusted_goals_for": 0,
            "strength_adjusted_goals_against": 0,
            "avg_opponent_position": 10,
            "top6_wins": 0,
            "bottom6_wins": 0,
        }

    results = []
    for _, match in team_matches.iterrows():
        is_home = match["home_team"] == team

        if is_home:
            goals_for = match.get("home_score", 0) or 0
            goals_against = match.get("away_score", 0) or 0
            opponent = match["away_team"]
            opp_position = match.get("away_league_position", 10) or 10
        else:
            goals_for = match.get("away_score", 0) or 0
            goals_against = match.get("home_score", 0) or 0
            opponent = match["home_team"]
            opp_position = match.get("home_league_position", 10) or 10

        # Calculate result
        if goals_for > goals_against:
            points = 3
            result = "W"
        elif goals_for == goals_against:
            points = 1
            result = "D"
        else:
            points = 0
            result = "L"

        # Opponent strength weight: position 1 = weight 1.5, position 20 = weight 0.5
        opp_weight = 1.5 - (opp_position - 1) * (1.0 / 19)

        results.append({
            "points": points,
            "weighted_points": points * opp_weight,
            "goals_for": goals_for,
            "goals_against": goals_against,
            "opp_position": opp_position,
            "opp_weight": opp_weight,
            "result": result,
        })

    if not results:
        return {
            "strength_adjusted_points": 0,
            "strength_adjusted_goals_for": 0,
            "strength_adjusted_goals_against": 0,
            "avg_opponent_position": 10,
            "top6_wins": 0,
            "bottom6_wins": 0,
        }

    n = len(results)

    # Strength-adjusted metrics
    total_weighted_points = sum(r["weighted_points"] for r in results)
    max_weighted_points = 3 * sum(r["opp_weight"] for r in results)

    # Wins vs different tiers
    top6_wins = sum(1 for r in results if r["result"] == "W" and r["opp_position"] <= 6)
    bottom6_wins = sum(1 for r in results if r["result"] == "W" and r["opp_position"] >= 15)

    return {
        "strength_adjusted_points": total_weighted_points / max_weighted_points if max_weighted_points > 0 else 0,
        "strength_adjusted_goals_for": sum(r["goals_for"] * r["opp_weight"] for r in results) / n,
        "strength_adjusted_goals_against": sum(r["goals_against"] / r["opp_weight"] for r in results) / n,
        "avg_opponent_position": sum(r["opp_position"] for r in results) / n,
        "top6_wins": top6_wins,
        "bottom6_wins": bottom6_wins,
        "top6_games": sum(1 for r in results if r["opp_position"] <= 6),
        "bottom6_games": sum(1 for r in results if r["opp_position"] >= 15),
    }


def compute_defensive_consistency(
    matches_df: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp,
    n_matches: int = 10,
) -> Dict[str, float]:
    """Compute defensive consistency features.

    - Clean sheet streak
    - Goals conceded vs top attacks
    - Big game defensive record
    """
    team_matches = matches_df[
        ((matches_df["home_team"] == team) | (matches_df["away_team"] == team)) &
        (matches_df["match_date"] < before_date)
    ].sort_values("match_date", ascending=False).head(n_matches)

    if team_matches.empty:
        return {
            "clean_sheet_streak": 0,
            "clean_sheets_last_n": 0,
            "clean_sheet_rate": 0.3,
            "goals_conceded_vs_top6": 1.5,
            "goals_conceded_vs_bottom6": 1.0,
            "defensive_consistency_score": 0.5,
            "avg_goals_conceded": 1.0,
        }

    clean_sheet_streak = 0
    clean_sheets = 0
    goals_vs_top6 = []
    goals_vs_bottom6 = []
    all_goals_conceded = []

    for i, (_, match) in enumerate(team_matches.iterrows()):
        is_home = match["home_team"] == team

        if is_home:
            goals_conceded = match.get("away_score", 0) or 0
            opp_position = match.get("away_league_position", 10) or 10
        else:
            goals_conceded = match.get("home_score", 0) or 0
            opp_position = match.get("home_league_position", 10) or 10

        all_goals_conceded.append(goals_conceded)

        # Clean sheet tracking
        if goals_conceded == 0:
            clean_sheets += 1
            if i == clean_sheet_streak:  # Still in streak
                clean_sheet_streak += 1

        # Goals vs opponent tiers
        if opp_position <= 6:
            goals_vs_top6.append(goals_conceded)
        elif opp_position >= 15:
            goals_vs_bottom6.append(goals_conceded)

    # Defensive consistency: low variance in goals conceded
    if len(all_goals_conceded) > 1:
        consistency = 1 / (1 + np.std(all_goals_conceded))
    else:
        consistency = 0.5

    return {
        "clean_sheet_streak": clean_sheet_streak,
        "clean_sheets_last_n": clean_sheets,
        "clean_sheet_rate": clean_sheets / len(team_matches),
        "goals_conceded_vs_top6": np.mean(goals_vs_top6) if goals_vs_top6 else 1.5,
        "goals_conceded_vs_bottom6": np.mean(goals_vs_bottom6) if goals_vs_bottom6 else 1.0,
        "defensive_consistency_score": consistency,
        "avg_goals_conceded": np.mean(all_goals_conceded) if all_goals_conceded else 1.0,
    }


def compute_attack_consistency(
    matches_df: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp,
    n_matches: int = 10,
) -> Dict[str, float]:
    """Compute attack consistency and scoring patterns.

    - Goals scored vs different defenses
    - Scoring streak
    - Big game scoring
    """
    team_matches = matches_df[
        ((matches_df["home_team"] == team) | (matches_df["away_team"] == team)) &
        (matches_df["match_date"] < before_date)
    ].sort_values("match_date", ascending=False).head(n_matches)

    if team_matches.empty:
        return {
            "scoring_streak": 0,
            "matches_scored_in": 0,
            "scoring_rate": 0.7,
            "goals_vs_top_defenses": 1.0,
            "goals_vs_weak_defenses": 1.5,
            "attack_consistency_score": 0.5,
            "avg_goals_scored": 1.0,
        }

    scoring_streak = 0
    all_goals = []
    goals_vs_top_def = []
    goals_vs_weak_def = []

    for i, (_, match) in enumerate(team_matches.iterrows()):
        is_home = match["home_team"] == team

        if is_home:
            goals = match.get("home_score", 0) or 0
            opp_position = match.get("away_league_position", 10) or 10
        else:
            goals = match.get("away_score", 0) or 0
            opp_position = match.get("home_league_position", 10) or 10

        all_goals.append(goals)

        if goals > 0:
            if i == scoring_streak:
                scoring_streak += 1

        # Top defenses = top 6 teams, weak defenses = bottom 6
        if opp_position <= 6:
            goals_vs_top_def.append(goals)
        elif opp_position >= 15:
            goals_vs_weak_def.append(goals)

    return {
        "scoring_streak": scoring_streak,
        "matches_scored_in": sum(1 for g in all_goals if g > 0),
        "scoring_rate": sum(1 for g in all_goals if g > 0) / len(all_goals) if all_goals else 0,
        "goals_vs_top_defenses": np.mean(goals_vs_top_def) if goals_vs_top_def else 1.0,
        "goals_vs_weak_defenses": np.mean(goals_vs_weak_def) if goals_vs_weak_def else 1.5,
        "attack_consistency_score": 1 / (1 + np.std(all_goals)) if len(all_goals) > 1 else 0.5,
        "avg_goals_scored": np.mean(all_goals) if all_goals else 1.0,
    }


def compute_matchup_features(
    home_attack: Dict[str, float],
    home_defense: Dict[str, float],
    away_attack: Dict[str, float],
    away_defense: Dict[str, float],
    home_position: int,
    away_position: int,
) -> Dict[str, float]:
    """Compute head-to-head matchup features.

    Key insight: It's not just about team strength, but how styles match up.
    - Strong home attack vs weak away defense = likely goals
    - Solid home defense vs weak away attack = likely clean sheet
    """
    features = {}

    # Attack vs Defense matchups
    # Home attack vs away defense
    home_attack_strength = home_attack.get("avg_goals_scored", 1.0)
    away_defense_weakness = away_defense.get("avg_goals_conceded", 1.0)
    features["home_attack_vs_away_defense"] = home_attack_strength * away_defense_weakness

    # Away attack vs home defense
    away_attack_strength = away_attack.get("avg_goals_scored", 1.0)
    home_defense_strength = home_defense.get("avg_goals_conceded", 1.0)
    features["away_attack_vs_home_defense"] = away_attack_strength * home_defense_strength

    # Clean sheet probability signals
    features["home_clean_sheet_signal"] = (
        home_defense.get("clean_sheet_rate", 0.3) *
        (1 - away_attack.get("scoring_rate", 0.7))
    )
    features["away_clean_sheet_signal"] = (
        away_defense.get("clean_sheet_rate", 0.3) *
        (1 - home_attack.get("scoring_rate", 0.7))
    )

    # Position gap (18th vs 4th is a big mismatch)
    features["position_gap"] = abs(home_position - away_position)
    features["home_position_advantage"] = away_position - home_position  # Positive = home higher

    # Mismatch score: big gap + good form = dominant result likely
    features["mismatch_score"] = (
        features["position_gap"] / 19 *
        abs(features["home_attack_vs_away_defense"] - features["away_attack_vs_home_defense"])
    )

    # Consistency matchup: consistent teams are more predictable
    features["combined_consistency"] = (
        home_defense.get("defensive_consistency_score", 0.5) *
        away_attack.get("attack_consistency_score", 0.5)
    )

    return features


def compute_momentum_features(
    matches_df: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp,
) -> Dict[str, float]:
    """Compute momentum and trend features.

    - Is team improving or declining?
    - Recent big result boost
    - Confidence level
    """
    recent = matches_df[
        ((matches_df["home_team"] == team) | (matches_df["away_team"] == team)) &
        (matches_df["match_date"] < before_date)
    ].sort_values("match_date", ascending=False).head(10)

    if len(recent) < 5:
        return {
            "form_trend": 0,
            "recent_big_win": 0,
            "confidence_boost": 0,
        }

    # Calculate points for last 5 vs previous 5
    points_list = []
    big_wins = []

    for _, match in recent.iterrows():
        is_home = match["home_team"] == team

        if is_home:
            goals_for = match.get("home_score", 0) or 0
            goals_against = match.get("away_score", 0) or 0
            opp_position = match.get("away_league_position", 10) or 10
        else:
            goals_for = match.get("away_score", 0) or 0
            goals_against = match.get("home_score", 0) or 0
            opp_position = match.get("home_league_position", 10) or 10

        if goals_for > goals_against:
            points_list.append(3)
            # Big win = beat top 6 team or win by 3+ goals
            if opp_position <= 6 or (goals_for - goals_against) >= 3:
                big_wins.append(1)
            else:
                big_wins.append(0)
        elif goals_for == goals_against:
            points_list.append(1)
            big_wins.append(0)
        else:
            points_list.append(0)
            big_wins.append(0)

    # Form trend: last 5 vs previous 5
    last_5 = sum(points_list[:5]) / 15  # Normalized to 0-1
    prev_5 = sum(points_list[5:10]) / 15 if len(points_list) > 5 else last_5

    return {
        "form_trend": last_5 - prev_5,  # Positive = improving
        "recent_big_win": 1 if any(big_wins[:3]) else 0,  # Big win in last 3
        "confidence_boost": sum(big_wins[:5]) / 5,  # Big win ratio in last 5
        "current_form": last_5,
    }


def add_contextual_features(df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
    """Add all contextual features to a match DataFrame.

    Args:
        df: DataFrame with upcoming/current matches
        matches_df: Historical matches for computing features

    Returns:
        DataFrame with contextual features added
    """
    df = df.copy()

    log.info("Computing contextual features...")

    # Ensure date columns
    df["match_date"] = pd.to_datetime(df["match_date"])
    matches_df = matches_df.copy()
    matches_df["match_date"] = pd.to_datetime(matches_df["match_date"])

    # Initialize new columns
    new_cols = [
        # Strength-adjusted
        "home_strength_adj_points", "away_strength_adj_points",
        "home_top6_wins", "away_top6_wins",
        "home_bottom6_wins", "away_bottom6_wins",
        # Defensive
        "home_clean_sheet_streak", "away_clean_sheet_streak",
        "home_clean_sheet_rate", "away_clean_sheet_rate",
        "home_goals_vs_top_attacks", "away_goals_vs_top_attacks",
        # Attack
        "home_scoring_streak", "away_scoring_streak",
        "home_scoring_rate", "away_scoring_rate",
        "home_goals_vs_weak_def", "away_goals_vs_weak_def",
        # Matchup
        "home_attack_vs_away_def", "away_attack_vs_home_def",
        "home_clean_sheet_signal", "away_clean_sheet_signal",
        "position_gap", "mismatch_score",
        # Momentum
        "home_form_trend", "away_form_trend",
        "home_recent_big_win", "away_recent_big_win",
    ]

    for col in new_cols:
        df[col] = 0.0

    for idx, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        match_date = row["match_date"]

        # Get positions if available
        home_pos = row.get("home_league_position", 10) or 10
        away_pos = row.get("away_league_position", 10) or 10

        # Compute features for home team
        home_strength = compute_opponent_strength(matches_df, home_team, match_date)
        home_defense = compute_defensive_consistency(matches_df, home_team, match_date)
        home_attack = compute_attack_consistency(matches_df, home_team, match_date)
        home_momentum = compute_momentum_features(matches_df, home_team, match_date)

        # Compute features for away team
        away_strength = compute_opponent_strength(matches_df, away_team, match_date)
        away_defense = compute_defensive_consistency(matches_df, away_team, match_date)
        away_attack = compute_attack_consistency(matches_df, away_team, match_date)
        away_momentum = compute_momentum_features(matches_df, away_team, match_date)

        # Compute matchup features
        matchup = compute_matchup_features(
            home_attack, home_defense,
            away_attack, away_defense,
            home_pos, away_pos
        )

        # Set values
        df.at[idx, "home_strength_adj_points"] = home_strength["strength_adjusted_points"]
        df.at[idx, "away_strength_adj_points"] = away_strength["strength_adjusted_points"]
        df.at[idx, "home_top6_wins"] = home_strength["top6_wins"]
        df.at[idx, "away_top6_wins"] = away_strength["top6_wins"]
        df.at[idx, "home_bottom6_wins"] = home_strength["bottom6_wins"]
        df.at[idx, "away_bottom6_wins"] = away_strength["bottom6_wins"]

        df.at[idx, "home_clean_sheet_streak"] = home_defense["clean_sheet_streak"]
        df.at[idx, "away_clean_sheet_streak"] = away_defense["clean_sheet_streak"]
        df.at[idx, "home_clean_sheet_rate"] = home_defense["clean_sheet_rate"]
        df.at[idx, "away_clean_sheet_rate"] = away_defense["clean_sheet_rate"]
        df.at[idx, "home_goals_vs_top_attacks"] = home_defense["goals_conceded_vs_top6"]
        df.at[idx, "away_goals_vs_top_attacks"] = away_defense["goals_conceded_vs_top6"]

        df.at[idx, "home_scoring_streak"] = home_attack["scoring_streak"]
        df.at[idx, "away_scoring_streak"] = away_attack["scoring_streak"]
        df.at[idx, "home_scoring_rate"] = home_attack["scoring_rate"]
        df.at[idx, "away_scoring_rate"] = away_attack["scoring_rate"]
        df.at[idx, "home_goals_vs_weak_def"] = home_attack["goals_vs_weak_defenses"]
        df.at[idx, "away_goals_vs_weak_def"] = away_attack["goals_vs_weak_defenses"]

        df.at[idx, "home_attack_vs_away_def"] = matchup["home_attack_vs_away_defense"]
        df.at[idx, "away_attack_vs_home_def"] = matchup["away_attack_vs_home_defense"]
        df.at[idx, "home_clean_sheet_signal"] = matchup["home_clean_sheet_signal"]
        df.at[idx, "away_clean_sheet_signal"] = matchup["away_clean_sheet_signal"]
        df.at[idx, "position_gap"] = matchup["position_gap"]
        df.at[idx, "mismatch_score"] = matchup["mismatch_score"]

        df.at[idx, "home_form_trend"] = home_momentum["form_trend"]
        df.at[idx, "away_form_trend"] = away_momentum["form_trend"]
        df.at[idx, "home_recent_big_win"] = home_momentum["recent_big_win"]
        df.at[idx, "away_recent_big_win"] = away_momentum["recent_big_win"]

    log.info(f"Added {len(new_cols)} contextual features")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing contextual features...")
    print("=" * 60)

    # Example match scenario
    print("\nExample: Juventus (4th) vs Verona (18th)")
    print("-" * 60)

    # Simulated data
    home_attack = {"avg_goals_scored": 2.1, "scoring_rate": 0.9, "goals_vs_weak_defenses": 2.8}
    home_defense = {"clean_sheet_rate": 0.4, "avg_goals_conceded": 0.8, "defensive_consistency_score": 0.7}
    away_attack = {"avg_goals_scored": 0.9, "scoring_rate": 0.6, "goals_vs_weak_defenses": 1.2}
    away_defense = {"clean_sheet_rate": 0.15, "avg_goals_conceded": 2.1, "defensive_consistency_score": 0.4}

    matchup = compute_matchup_features(
        home_attack, home_defense,
        away_attack, away_defense,
        home_position=4, away_position=18
    )

    print("\nMatchup Analysis:")
    print(f"  Position gap: {matchup['position_gap']} places")
    print(f"  Home attack vs Away defense: {matchup['home_attack_vs_away_defense']:.2f}")
    print(f"  Away attack vs Home defense: {matchup['away_attack_vs_home_defense']:.2f}")
    print(f"  Home clean sheet signal: {matchup['home_clean_sheet_signal']:.2f}")
    print(f"  Away clean sheet signal: {matchup['away_clean_sheet_signal']:.2f}")
    print(f"  Mismatch score: {matchup['mismatch_score']:.2f}")

    print("\nExpected outcome signals:")
    if matchup["home_clean_sheet_signal"] > 0.15:
        print("  ✓ High probability: Home clean sheet")
    if matchup["home_attack_vs_away_defense"] > 2.5:
        print("  ✓ High probability: Home scores multiple goals")
    if matchup["mismatch_score"] > 0.3:
        print("  ✓ High probability: One-sided match")
