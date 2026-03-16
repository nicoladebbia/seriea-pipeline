"""Consolidated Momentum & Streak Features.

Combines basic streak tracking (formerly in momentum.py) with advanced
psychological/momentum analysis (Phase 4) into a single module.

=== BASIC STREAKS (team-log level) ===
  - Current win/draw/loss streak length
  - Unbeaten run length, winless run length
  - Scoring/clean-sheet streaks
  - Points in last N matches (form_points_3, form_points_5)

=== ADVANCED MOMENTUM (match level) ===
  1. BIG WIN MOMENTUM
     - Scoring 3+ goals creates confidence boost
     - Tracks recent dominant victories

  2. COMEBACK MOMENTUM
     - Winning from behind shows mental strength
     - Teams that come back often have resilience factor

  3. LATE GOAL TREND
     - Goals in last 15 minutes show fitness/mentality
     - Late winners vs late concessions

  4. PRESSURE RESPONSE
     - Performance when behind in critical matches
     - Derby/big-game performance

  5. SCORING PATTERNS
     - Goals per half, blank match ratio
     - Multi-goal game ratio, scoring variance

  6. COMPOSITE SCORES
     - Offensive, psychological, physical momentum
     - Overall momentum score
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)


# =============================================================================
# BASIC STREAK FEATURES (from former momentum.py)
# =============================================================================

def add_momentum_features(team_log: pd.DataFrame) -> pd.DataFrame:
    """Add basic momentum and streak features to the team match log.

    All features use only data BEFORE the current match (shift(1)).

    Columns added:
      win_streak, unbeaten_run, winless_run, loss_streak,
      scoring_streak, clean_sheet_streak,
      form_points_3, form_points_5 (exact sum, not average)

    Parameters
    ----------
    team_log : pd.DataFrame
        Team-level match log with columns: team, match_date, result,
        goals_scored, goals_conceded, points.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with streak columns appended.
    """
    df = team_log.copy()
    df = df.sort_values(["team", "match_date"]).reset_index(drop=True)

    # Pre-compute shifted series per team
    win_streaks = []
    unbeaten_runs = []
    winless_runs = []
    loss_streaks = []
    scoring_streaks = []
    cs_streaks = []
    form_pts_3 = []
    form_pts_5 = []

    for team, group in df.groupby("team"):
        results = group["result"].tolist()
        goals = group["goals_scored"].tolist()
        conceded = group["goals_conceded"].tolist()
        points = group["points"].tolist()

        t_win = []
        t_unbeaten = []
        t_winless = []
        t_loss = []
        t_scoring = []
        t_cs = []
        t_fp3 = []
        t_fp5 = []

        # Running state
        cur_win = 0
        cur_unbeaten = 0
        cur_winless = 0
        cur_loss = 0
        cur_scoring = 0
        cur_cs = 0
        recent_pts: list[float] = []

        for i in range(len(results)):
            # Record BEFORE this match
            if i == 0:
                t_win.append(None)
                t_unbeaten.append(None)
                t_winless.append(None)
                t_loss.append(None)
                t_scoring.append(None)
                t_cs.append(None)
                t_fp3.append(None)
                t_fp5.append(None)
            else:
                t_win.append(cur_win)
                t_unbeaten.append(cur_unbeaten)
                t_winless.append(cur_winless)
                t_loss.append(cur_loss)
                t_scoring.append(cur_scoring)
                t_cs.append(cur_cs)
                t_fp3.append(sum(recent_pts[-3:]) if len(recent_pts) >= 1 else None)
                t_fp5.append(sum(recent_pts[-5:]) if len(recent_pts) >= 1 else None)

            # Update running state with this match
            r = results[i]
            gs = goals[i] if not pd.isna(goals[i]) else 0
            gc = conceded[i] if not pd.isna(conceded[i]) else 0
            pts = points[i] if not pd.isna(points[i]) else 0

            # Win streak
            if r == "W":
                cur_win += 1
            else:
                cur_win = 0

            # Unbeaten run
            if r in ("W", "D"):
                cur_unbeaten += 1
            else:
                cur_unbeaten = 0

            # Winless run
            if r in ("L", "D"):
                cur_winless += 1
            else:
                cur_winless = 0

            # Loss streak
            if r == "L":
                cur_loss += 1
            else:
                cur_loss = 0

            # Scoring streak
            if gs > 0:
                cur_scoring += 1
            else:
                cur_scoring = 0

            # Clean sheet streak
            if gc == 0:
                cur_cs += 1
            else:
                cur_cs = 0

            recent_pts.append(pts)

        win_streaks.extend(t_win)
        unbeaten_runs.extend(t_unbeaten)
        winless_runs.extend(t_winless)
        loss_streaks.extend(t_loss)
        scoring_streaks.extend(t_scoring)
        cs_streaks.extend(t_cs)
        form_pts_3.extend(t_fp3)
        form_pts_5.extend(t_fp5)

    df["win_streak"] = win_streaks
    df["unbeaten_run"] = unbeaten_runs
    df["winless_run"] = winless_runs
    df["loss_streak"] = loss_streaks
    df["scoring_streak"] = scoring_streaks
    df["clean_sheet_streak"] = cs_streaks
    df["form_points_3"] = form_pts_3
    df["form_points_5"] = form_pts_5

    return df


# =============================================================================
# ADVANCED MOMENTUM FEATURES (Phase 4)
# =============================================================================

def compute_big_win_momentum(
    matches_df: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp,
    n_matches: int = 10,
) -> Dict[str, float]:
    """Compute big win momentum features.

    A big win is:
    - Scoring 3+ goals in a match
    - Winning by 3+ goal margin
    - Beating a top-6 team

    These create psychological momentum effects.
    """
    team_matches = matches_df[
        ((matches_df["home_team"] == team) | (matches_df["away_team"] == team)) &
        (matches_df["match_date"] < before_date)
    ].sort_values("match_date", ascending=False).head(n_matches)

    if team_matches.empty:
        return {
            "big_win_count": 0,
            "big_win_recency": 0.0,
            "dominant_win_ratio": 0.0,
            "goals_explosion_streak": 0,
            "avg_goals_in_wins": 1.5,
        }

    big_wins = []
    goals_in_wins = []
    current_explosion_streak = 0

    for i, (_, match) in enumerate(team_matches.iterrows()):
        is_home = match["home_team"] == team

        if is_home:
            goals_for = match.get("home_score", 0) or 0
            goals_against = match.get("away_score", 0) or 0
            opp_position = match.get("away_league_position", 10) or 10
        else:
            goals_for = match.get("away_score", 0) or 0
            goals_against = match.get("home_score", 0) or 0
            opp_position = match.get("home_league_position", 10) or 10

        is_win = goals_for > goals_against
        margin = goals_for - goals_against

        # Track big wins
        is_big_win = False
        if is_win:
            goals_in_wins.append(goals_for)
            if goals_for >= 3 or margin >= 3 or (is_win and opp_position <= 6):
                is_big_win = True
                big_wins.append({
                    "recency": i,
                    "goals": goals_for,
                    "margin": margin,
                    "vs_top6": opp_position <= 6
                })

        # Track goals explosion streak (3+ goals in consecutive matches)
        if goals_for >= 3 and i == current_explosion_streak:
            current_explosion_streak += 1

    # Calculate recency-weighted big win score
    # Recent big wins matter more
    recency_score = 0.0
    for bw in big_wins:
        weight = 1.0 / (1 + bw["recency"])  # Decay with recency
        recency_score += weight

    return {
        "big_win_count": len(big_wins),
        "big_win_recency": recency_score,  # Higher = recent big wins
        "dominant_win_ratio": len(big_wins) / len(team_matches) if len(team_matches) > 0 else 0,
        "goals_explosion_streak": current_explosion_streak,
        "avg_goals_in_wins": np.mean(goals_in_wins) if goals_in_wins else 1.5,
    }


def compute_comeback_momentum(
    matches_df: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp,
    n_matches: int = 15,
) -> Dict[str, float]:
    """Compute comeback momentum features.

    Tracks:
    - Wins after being behind (comeback wins)
    - Points rescued from losing positions
    - Psychological resilience factor
    """
    # This requires match event data to know scoreline progression
    # For now, we'll use a proxy based on goal patterns

    team_matches = matches_df[
        ((matches_df["home_team"] == team) | (matches_df["away_team"] == team)) &
        (matches_df["match_date"] < before_date)
    ].sort_values("match_date", ascending=False).head(n_matches)

    if team_matches.empty:
        return {
            "comeback_wins": 0,
            "comeback_ratio": 0.0,
            "resilience_factor": 0.5,
            "points_from_losing_positions": 0,
        }

    # Proxy: If we can detect "first_goal_time" or "conceded_first", use that
    # Otherwise, use a heuristic based on goal difference patterns

    comeback_count = 0
    total_wins = 0
    potential_comebacks = 0

    for _, match in team_matches.iterrows():
        is_home = match["home_team"] == team

        if is_home:
            goals_for = match.get("home_score", 0) or 0
            goals_against = match.get("away_score", 0) or 0
            conceded_first = match.get("home_conceded_first", None)
        else:
            goals_for = match.get("away_score", 0) or 0
            goals_against = match.get("home_score", 0) or 0
            conceded_first = match.get("away_conceded_first", None)

        is_win = goals_for > goals_against
        is_draw = goals_for == goals_against

        if is_win:
            total_wins += 1
            # If we know they conceded first, this is a comeback
            if conceded_first is True:
                comeback_count += 1
                potential_comebacks += 1
            # Heuristic: High-scoring wins where opponent also scored
            elif goals_for >= 2 and goals_against >= 1:
                # Likely went behind at some point
                comeback_count += 0.5  # Partial credit
                potential_comebacks += 1
        elif is_draw and goals_against >= 1:
            # Draw after conceding shows some resilience
            potential_comebacks += 1
            if goals_for >= goals_against:
                comeback_count += 0.3  # Small credit for equalizing

    comeback_ratio = comeback_count / potential_comebacks if potential_comebacks > 0 else 0

    return {
        "comeback_wins": int(comeback_count),
        "comeback_ratio": comeback_ratio,
        "resilience_factor": min(1.0, 0.3 + comeback_ratio),  # 0.3 base + bonus for comebacks
        "points_from_losing_positions": int(comeback_count * 2),  # Estimate
    }


def compute_late_goal_trend(
    matches_df: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp,
    n_matches: int = 10,
) -> Dict[str, float]:
    """Compute late goal trend features.

    Goals after 75th minute indicate:
    - Physical fitness (can finish strong)
    - Mental strength (pressure situations)
    - Manager tactics (late subs, pushing for result)
    """
    team_matches = matches_df[
        ((matches_df["home_team"] == team) | (matches_df["away_team"] == team)) &
        (matches_df["match_date"] < before_date)
    ].sort_values("match_date", ascending=False).head(n_matches)

    if team_matches.empty:
        return {
            "late_goals_scored": 0,
            "late_goals_conceded": 0,
            "late_goal_diff": 0,
            "late_winner_count": 0,
            "late_collapse_count": 0,
            "fitness_indicator": 0.5,
        }

    late_scored = 0
    late_conceded = 0
    late_winners = 0
    late_collapses = 0

    for _, match in team_matches.iterrows():
        is_home = match["home_team"] == team

        # Try to get late goal data from various possible columns
        if is_home:
            late_for = match.get("home_goals_75_90", match.get("home_late_goals", 0)) or 0
            late_against = match.get("away_goals_75_90", match.get("away_late_goals", 0)) or 0
            goals_for = match.get("home_score", 0) or 0
            goals_against = match.get("away_score", 0) or 0
        else:
            late_for = match.get("away_goals_75_90", match.get("away_late_goals", 0)) or 0
            late_against = match.get("home_goals_75_90", match.get("home_late_goals", 0)) or 0
            goals_for = match.get("away_score", 0) or 0
            goals_against = match.get("home_score", 0) or 0

        late_scored += late_for
        late_conceded += late_against

        # Detect late winners (won by 1 and scored late)
        if goals_for - goals_against == 1 and late_for > 0:
            late_winners += 1

        # Detect late collapses (lost/drew after leading, conceded late)
        if goals_for <= goals_against and late_against > 0:
            late_collapses += 1

    # Fitness indicator: positive late goal diff suggests strong finish
    late_diff = late_scored - late_conceded
    fitness = 0.5 + (late_diff / (n_matches * 0.5))  # Normalized
    fitness = max(0.2, min(0.9, fitness))  # Bound

    return {
        "late_goals_scored": late_scored,
        "late_goals_conceded": late_conceded,
        "late_goal_diff": late_diff,
        "late_winner_count": late_winners,
        "late_collapse_count": late_collapses,
        "fitness_indicator": fitness,
    }


def compute_pressure_response(
    matches_df: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp,
    n_matches: int = 20,
) -> Dict[str, float]:
    """Compute pressure response features.

    Performance in:
    - Must-win situations (relegation battles, title races)
    - Derby matches (emotional pressure)
    - Big matches (vs top teams)
    """
    team_matches = matches_df[
        ((matches_df["home_team"] == team) | (matches_df["away_team"] == team)) &
        (matches_df["match_date"] < before_date)
    ].sort_values("match_date", ascending=False).head(n_matches)

    if team_matches.empty:
        return {
            "big_game_win_rate": 0.4,
            "derby_performance": 0.5,
            "pressure_success_rate": 0.5,
            "clutch_factor": 0.5,
        }

    big_game_results = []
    derby_results = []
    pressure_results = []

    for _, match in team_matches.iterrows():
        is_home = match["home_team"] == team

        if is_home:
            goals_for = match.get("home_score", 0) or 0
            goals_against = match.get("away_score", 0) or 0
            opp_position = match.get("away_league_position", 10) or 10
            in_relegation = match.get("home_in_relegation_zone", False)
        else:
            goals_for = match.get("away_score", 0) or 0
            goals_against = match.get("home_score", 0) or 0
            opp_position = match.get("home_league_position", 10) or 10
            in_relegation = match.get("away_in_relegation_zone", False)

        is_win = goals_for > goals_against
        is_loss = goals_for < goals_against
        points = 3 if is_win else (1 if goals_for == goals_against else 0)

        # Big game = vs top 6
        if opp_position <= 6:
            big_game_results.append(points / 3)  # Normalized

        # Derby detection (same city teams)
        is_derby = match.get("is_derby", False)
        if is_derby:
            derby_results.append(points / 3)

        # Pressure situation = in relegation zone or title race
        if in_relegation or match.get("is_title_decider", False):
            pressure_results.append(points / 3)

    big_game_rate = np.mean(big_game_results) if big_game_results else 0.4
    derby_rate = np.mean(derby_results) if derby_results else 0.5
    pressure_rate = np.mean(pressure_results) if pressure_results else 0.5

    # Clutch factor combines all pressure metrics
    clutch = (big_game_rate + derby_rate + pressure_rate) / 3

    return {
        "big_game_win_rate": big_game_rate,
        "derby_performance": derby_rate,
        "pressure_success_rate": pressure_rate,
        "clutch_factor": clutch,
    }


def compute_scoring_patterns(
    matches_df: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp,
    n_matches: int = 10,
) -> Dict[str, float]:
    """Compute advanced scoring pattern features.

    Tracks:
    - Goals per period (first half, second half)
    - Scoring consistency
    - Blank match probability
    """
    team_matches = matches_df[
        ((matches_df["home_team"] == team) | (matches_df["away_team"] == team)) &
        (matches_df["match_date"] < before_date)
    ].sort_values("match_date", ascending=False).head(n_matches)

    if team_matches.empty:
        return {
            "first_half_goals_avg": 0.6,
            "second_half_goals_avg": 0.8,
            "blank_match_ratio": 0.2,
            "multi_goal_ratio": 0.4,
            "scoring_variance": 1.0,
        }

    goals_list = []
    first_half_goals = []
    second_half_goals = []
    blanks = 0
    multi_goals = 0

    for _, match in team_matches.iterrows():
        is_home = match["home_team"] == team

        if is_home:
            goals = match.get("home_score", 0) or 0
            fh_goals = match.get("home_first_half_goals", goals * 0.4) or 0  # Estimate if not available
            sh_goals = match.get("home_second_half_goals", goals * 0.6) or 0
        else:
            goals = match.get("away_score", 0) or 0
            fh_goals = match.get("away_first_half_goals", goals * 0.4) or 0
            sh_goals = match.get("away_second_half_goals", goals * 0.6) or 0

        goals_list.append(goals)
        first_half_goals.append(fh_goals)
        second_half_goals.append(sh_goals)

        if goals == 0:
            blanks += 1
        if goals >= 2:
            multi_goals += 1

    return {
        "first_half_goals_avg": np.mean(first_half_goals),
        "second_half_goals_avg": np.mean(second_half_goals),
        "blank_match_ratio": blanks / len(team_matches),
        "multi_goal_ratio": multi_goals / len(team_matches),
        "scoring_variance": np.var(goals_list) if len(goals_list) > 1 else 1.0,
    }


def add_enhanced_momentum_features(
    df: pd.DataFrame,
    matches_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add all advanced momentum features to a match DataFrame.

    This adds match-level features (big win, comeback, late goals,
    pressure, scoring patterns) for both home and away teams.

    Parameters
    ----------
    df : pd.DataFrame
        Match-level DataFrame with home_team, away_team, match_date.
    matches_df : pd.DataFrame
        Historical matches for lookback context.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with advanced momentum columns appended.
    """
    df = df.copy()

    log.info("Computing enhanced momentum features...")

    # Ensure date columns
    df["match_date"] = pd.to_datetime(df["match_date"])
    matches_df = matches_df.copy()
    matches_df["match_date"] = pd.to_datetime(matches_df["match_date"])

    # Initialize new columns
    new_cols = [
        # Big win momentum
        "home_big_win_count", "away_big_win_count",
        "home_big_win_recency", "away_big_win_recency",
        "home_dominant_ratio", "away_dominant_ratio",
        # Comeback momentum
        "home_comeback_wins", "away_comeback_wins",
        "home_resilience", "away_resilience",
        # Late goals
        "home_late_goal_diff", "away_late_goal_diff",
        "home_fitness", "away_fitness",
        # Pressure
        "home_clutch", "away_clutch",
        "home_big_game_rate", "away_big_game_rate",
        # Scoring patterns
        "home_blank_ratio", "away_blank_ratio",
        "home_multi_goal_ratio", "away_multi_goal_ratio",
    ]

    for col in new_cols:
        df[col] = 0.0

    for idx, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        match_date = row["match_date"]

        # Compute features for home team
        home_big_win = compute_big_win_momentum(matches_df, home_team, match_date)
        home_comeback = compute_comeback_momentum(matches_df, home_team, match_date)
        home_late = compute_late_goal_trend(matches_df, home_team, match_date)
        home_pressure = compute_pressure_response(matches_df, home_team, match_date)
        home_scoring = compute_scoring_patterns(matches_df, home_team, match_date)

        # Compute features for away team
        away_big_win = compute_big_win_momentum(matches_df, away_team, match_date)
        away_comeback = compute_comeback_momentum(matches_df, away_team, match_date)
        away_late = compute_late_goal_trend(matches_df, away_team, match_date)
        away_pressure = compute_pressure_response(matches_df, away_team, match_date)
        away_scoring = compute_scoring_patterns(matches_df, away_team, match_date)

        # Set values
        df.at[idx, "home_big_win_count"] = home_big_win["big_win_count"]
        df.at[idx, "away_big_win_count"] = away_big_win["big_win_count"]
        df.at[idx, "home_big_win_recency"] = home_big_win["big_win_recency"]
        df.at[idx, "away_big_win_recency"] = away_big_win["big_win_recency"]
        df.at[idx, "home_dominant_ratio"] = home_big_win["dominant_win_ratio"]
        df.at[idx, "away_dominant_ratio"] = away_big_win["dominant_win_ratio"]

        df.at[idx, "home_comeback_wins"] = home_comeback["comeback_wins"]
        df.at[idx, "away_comeback_wins"] = away_comeback["comeback_wins"]
        df.at[idx, "home_resilience"] = home_comeback["resilience_factor"]
        df.at[idx, "away_resilience"] = away_comeback["resilience_factor"]

        df.at[idx, "home_late_goal_diff"] = home_late["late_goal_diff"]
        df.at[idx, "away_late_goal_diff"] = away_late["late_goal_diff"]
        df.at[idx, "home_fitness"] = home_late["fitness_indicator"]
        df.at[idx, "away_fitness"] = away_late["fitness_indicator"]

        df.at[idx, "home_clutch"] = home_pressure["clutch_factor"]
        df.at[idx, "away_clutch"] = away_pressure["clutch_factor"]
        df.at[idx, "home_big_game_rate"] = home_pressure["big_game_win_rate"]
        df.at[idx, "away_big_game_rate"] = away_pressure["big_game_win_rate"]

        df.at[idx, "home_blank_ratio"] = home_scoring["blank_match_ratio"]
        df.at[idx, "away_blank_ratio"] = away_scoring["blank_match_ratio"]
        df.at[idx, "home_multi_goal_ratio"] = home_scoring["multi_goal_ratio"]
        df.at[idx, "away_multi_goal_ratio"] = away_scoring["multi_goal_ratio"]

    log.info(f"Added {len(new_cols)} enhanced momentum features")

    return df


# =============================================================================
# MOMENTUM COMPOSITE SCORES
# =============================================================================

def compute_momentum_composite(features: Dict[str, float]) -> Dict[str, float]:
    """Compute composite momentum scores for use in predictions.

    Returns weighted combinations of momentum factors.
    """
    # Offensive momentum: Big wins + multi-goal games
    offensive_momentum = (
        0.3 * features.get("big_win_recency", 0) +
        0.3 * features.get("dominant_ratio", 0) +
        0.2 * features.get("multi_goal_ratio", 0) +
        0.2 * (1 - features.get("blank_ratio", 0.2))
    )

    # Psychological momentum: Comebacks + pressure performance
    psychological_momentum = (
        0.4 * features.get("resilience", 0.5) +
        0.3 * features.get("clutch", 0.5) +
        0.3 * features.get("big_game_rate", 0.4)
    )

    # Physical momentum: Late goals + fitness
    physical_momentum = (
        0.5 * features.get("fitness", 0.5) +
        0.5 * (0.5 + features.get("late_goal_diff", 0) / 10)
    )

    # Overall momentum score (0-1 scale)
    overall = (offensive_momentum + psychological_momentum + physical_momentum) / 3

    return {
        "offensive_momentum": offensive_momentum,
        "psychological_momentum": psychological_momentum,
        "physical_momentum": physical_momentum,
        "overall_momentum": overall,
    }


# =============================================================================
# UNIFIED ENTRY POINT
# =============================================================================

def add_all_momentum_features(
    team_log: pd.DataFrame,
    matches_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Add ALL momentum features (basic streaks + advanced) in one call.

    This is the recommended single entry point for the build pipeline.

    Parameters
    ----------
    team_log : pd.DataFrame
        Team-level match log (used for basic streak features).
    matches_df : pd.DataFrame, optional
        Match-level DataFrame for advanced momentum features.
        If None, only basic streaks are computed.

    Returns
    -------
    pd.DataFrame
        Input team_log with all momentum columns appended.

    Notes
    -----
    Basic streak features are always added (team-log level).
    Advanced momentum features (big win, comeback, late goals, pressure,
    scoring patterns) require ``matches_df`` and are added as match-level
    features via ``add_enhanced_momentum_features``.
    """
    log.info("Adding all momentum features (basic + advanced)...")

    # Step 1: Basic streak features (always)
    result = add_momentum_features(team_log)
    log.info("  Basic streaks: 8 columns added")

    # Step 2: Advanced momentum features (if matches_df provided)
    if matches_df is not None:
        result = add_enhanced_momentum_features(result, matches_df)
    else:
        log.info("  Skipping advanced momentum (no matches_df provided)")

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Consolidated Momentum Features")
    print("=" * 60)

    # Example calculations
    example_features = {
        "big_win_recency": 1.5,
        "dominant_ratio": 0.3,
        "multi_goal_ratio": 0.6,
        "blank_ratio": 0.1,
        "resilience": 0.7,
        "clutch": 0.6,
        "big_game_rate": 0.5,
        "fitness": 0.65,
        "late_goal_diff": 2,
    }

    composite = compute_momentum_composite(example_features)

    print("\nExample Team Momentum Composite:")
    print(f"  Offensive Momentum: {composite['offensive_momentum']:.2f}")
    print(f"  Psychological Momentum: {composite['psychological_momentum']:.2f}")
    print(f"  Physical Momentum: {composite['physical_momentum']:.2f}")
    print(f"  Overall Momentum: {composite['overall_momentum']:.2f}")
