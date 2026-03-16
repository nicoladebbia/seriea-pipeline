"""Compute player-level form features for match prediction.

This module calculates rolling performance metrics for individual players
and aggregates them for predicted lineups. Player form is critical for
predicting upcoming match outcomes.

Key metrics per player:
- xG + xA per 90 minutes
- Goals and assists
- Shot conversion rate
- Pass completion
- Defensive actions

Team-level aggregates:
- Average form of predicted starting XI
- Star player form (top 3 contributors)
- Form variance across squad
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from storage.paths import parsed_path

log = logging.getLogger(__name__)


@dataclass
class PlayerForm:
    """Rolling form metrics for a single player."""
    player_name: str
    team: str
    matches_played: int
    minutes_played: float
    goals: int
    assists: int
    xg: float
    xa: float  # xG assist
    xg_xa_per_90: float
    shot_conversion: float  # Goals / Shots
    pass_completion: float
    defensive_actions: float  # Tackles + Interceptions
    form_score: float  # Composite 0-1 score


def load_player_stats() -> pd.DataFrame:
    """Load player match-level statistics."""
    path = parsed_path("player_stats")
    if not path.exists():
        log.warning(f"Player stats not found at {path}")
        return pd.DataFrame()

    df = pd.read_parquet(path)

    # Convert numeric columns
    numeric_cols = [
        "minutes", "goals", "assists", "xg", "xg_assist",
        "shots", "shots_on_target", "passes", "passes_completed",
        "tackles", "interceptions", "sca", "gca",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def compute_player_form(
    player_name: str,
    team: str,
    before_date: date,
    n_matches: int = 5,
    season: Optional[str] = None,
) -> Optional[PlayerForm]:
    """Compute rolling form for a single player.

    Args:
        player_name: Player name
        team: Team name
        before_date: Only include matches before this date
        n_matches: Number of recent matches to consider
        season: Season to filter (auto-detect if not provided)

    Returns:
        PlayerForm object or None if insufficient data
    """
    df = load_player_stats()

    if df.empty:
        return None

    # Determine season if not provided
    if season is None:
        year = before_date.year
        month = before_date.month
        if month >= 8:
            season = f"{year}-{year + 1}"
        else:
            season = f"{year - 1}-{year}"

    # Filter to player matches
    player_matches = df[
        (df["player"].str.lower() == player_name.lower()) &
        (df["team"] == team) &
        (df["season"] == season)
    ].copy()

    if player_matches.empty:
        return None

    # Filter by date
    player_matches["match_date"] = pd.to_datetime(player_matches["match_date"])
    player_matches = player_matches[player_matches["match_date"].dt.date < before_date]

    # Get most recent matches
    player_matches = player_matches.sort_values("match_date", ascending=False).head(n_matches)

    if len(player_matches) < 2:  # Need at least 2 matches
        return None

    # Compute metrics
    total_minutes = player_matches["minutes"].sum()
    matches_played = len(player_matches)
    goals = int(player_matches["goals"].sum())
    assists = int(player_matches["assists"].sum())
    xg = player_matches["xg"].sum()
    xa = player_matches["xg_assist"].sum() if "xg_assist" in player_matches.columns else 0

    # Per 90 metrics
    if total_minutes > 0:
        xg_xa_per_90 = (xg + xa) / total_minutes * 90
    else:
        xg_xa_per_90 = 0

    # Shot conversion
    total_shots = player_matches["shots"].sum() if "shots" in player_matches.columns else 0
    shot_conversion = goals / total_shots if total_shots > 0 else 0

    # Pass completion
    passes = player_matches["passes"].sum() if "passes" in player_matches.columns else 0
    passes_completed = player_matches["passes_completed"].sum() if "passes_completed" in player_matches.columns else 0
    pass_completion = passes_completed / passes if passes > 0 else 0

    # Defensive actions
    tackles = player_matches["tackles"].sum() if "tackles" in player_matches.columns else 0
    interceptions = player_matches["interceptions"].sum() if "interceptions" in player_matches.columns else 0
    defensive_actions = (tackles + interceptions) / matches_played

    # Compute composite form score (0-1)
    # Weight: xG+xA most important for attackers, defensive actions for defenders
    form_score = min(1.0, (
        xg_xa_per_90 * 0.4 +  # Offensive output
        min(shot_conversion * 5, 1.0) * 0.2 +  # Finishing
        pass_completion * 0.2 +  # Ball retention
        min(defensive_actions / 5, 1.0) * 0.2  # Defensive work
    ))

    return PlayerForm(
        player_name=player_name,
        team=team,
        matches_played=matches_played,
        minutes_played=total_minutes,
        goals=goals,
        assists=assists,
        xg=round(xg, 2),
        xa=round(xa, 2),
        xg_xa_per_90=round(xg_xa_per_90, 2),
        shot_conversion=round(shot_conversion, 2),
        pass_completion=round(pass_completion, 2),
        defensive_actions=round(defensive_actions, 2),
        form_score=round(form_score, 2),
    )


def compute_squad_form(
    team: str,
    predicted_starters: list[str],
    match_date: date,
    n_matches: int = 5,
    season: Optional[str] = None,
) -> dict:
    """Compute aggregate form metrics for a predicted starting XI.

    Args:
        team: Team name
        predicted_starters: List of 11 predicted starter names
        match_date: Date of the upcoming match
        n_matches: Number of recent matches per player
        season: Season string

    Returns:
        Dictionary of aggregate form features
    """
    forms = []

    for player_name in predicted_starters:
        form = compute_player_form(player_name, team, match_date, n_matches, season)
        if form:
            forms.append(form)

    if not forms:
        return {
            "squad_avg_xg_xa_per_90": 0.0,
            "squad_avg_form_score": 0.0,
            "squad_form_variance": 0.0,
            "star_player_xg_xa": 0.0,
            "squad_total_goals": 0,
            "squad_total_assists": 0,
            "squad_coverage": 0.0,  # % of starters with form data
        }

    # Compute aggregates
    xg_xa_values = [f.xg_xa_per_90 for f in forms]
    form_scores = [f.form_score for f in forms]

    # Top 3 players by xG+xA
    top_3_xg_xa = sorted(xg_xa_values, reverse=True)[:3]
    star_player_xg_xa = sum(top_3_xg_xa) / len(top_3_xg_xa) if top_3_xg_xa else 0

    return {
        "squad_avg_xg_xa_per_90": round(np.mean(xg_xa_values), 2),
        "squad_avg_form_score": round(np.mean(form_scores), 2),
        "squad_form_variance": round(np.std(form_scores), 2),
        "star_player_xg_xa": round(star_player_xg_xa, 2),
        "squad_total_goals": sum(f.goals for f in forms),
        "squad_total_assists": sum(f.assists for f in forms),
        "squad_coverage": round(len(forms) / len(predicted_starters), 2),
    }


def compute_team_form_features(
    team: str,
    match_date: date,
    n_matches: int = 5,
    season: Optional[str] = None,
) -> dict:
    """Compute team-level form features without requiring lineup prediction.

    Uses all players who have played recently for the team.

    Args:
        team: Team name
        match_date: Date of the match
        n_matches: Number of recent matches
        season: Season string

    Returns:
        Dictionary of team form features
    """
    df = load_player_stats()

    if df.empty:
        return {
            "team_avg_xg_per_90": 0.0,
            "team_top_scorer_goals": 0,
            "team_top_assister_assists": 0,
            "team_players_in_form": 0,  # Players with form_score > 0.5
        }

    # Determine season
    if season is None:
        year = match_date.year
        month = match_date.month
        if month >= 8:
            season = f"{year}-{year + 1}"
        else:
            season = f"{year - 1}-{year}"

    # Get recent team matches
    team_stats = df[
        (df["team"] == team) &
        (df["season"] == season)
    ].copy()

    if team_stats.empty:
        return {
            "team_avg_xg_per_90": 0.0,
            "team_top_scorer_goals": 0,
            "team_top_assister_assists": 0,
            "team_players_in_form": 0,
        }

    team_stats["match_date"] = pd.to_datetime(team_stats["match_date"])
    team_stats = team_stats[team_stats["match_date"].dt.date < match_date]

    # Get most recent matches
    recent_matches = team_stats["match_date"].drop_duplicates().sort_values(ascending=False).head(n_matches)
    team_stats = team_stats[team_stats["match_date"].isin(recent_matches)]

    if team_stats.empty:
        return {
            "team_avg_xg_per_90": 0.0,
            "team_top_scorer_goals": 0,
            "team_top_assister_assists": 0,
            "team_players_in_form": 0,
        }

    # Aggregate by player
    player_agg = team_stats.groupby("player").agg({
        "goals": "sum",
        "assists": "sum",
        "xg": "sum",
        "minutes": "sum",
    }).reset_index()

    # Compute xG per 90
    player_agg["xg_per_90"] = player_agg.apply(
        lambda r: r["xg"] / r["minutes"] * 90 if r["minutes"] > 0 else 0,
        axis=1
    )

    # Team averages (for players with meaningful minutes)
    regular_players = player_agg[player_agg["minutes"] >= 45 * n_matches * 0.3]  # At least 30% playtime

    if regular_players.empty:
        regular_players = player_agg.head(11)

    avg_xg_per_90 = regular_players["xg_per_90"].mean()
    top_scorer_goals = player_agg["goals"].max()
    top_assister_assists = player_agg["assists"].max()

    # Count players "in form" (above average xG per 90)
    threshold = avg_xg_per_90 if avg_xg_per_90 > 0 else 0.2
    players_in_form = (regular_players["xg_per_90"] >= threshold).sum()

    return {
        "team_avg_xg_per_90": round(avg_xg_per_90, 2),
        "team_top_scorer_goals": int(top_scorer_goals),
        "team_top_assister_assists": int(top_assister_assists),
        "team_players_in_form": int(players_in_form),
    }


def compute_match_form_features(
    home_team: str,
    away_team: str,
    match_date: date,
    home_starters: Optional[list[str]] = None,
    away_starters: Optional[list[str]] = None,
    season: Optional[str] = None,
) -> dict:
    """Compute form features for an upcoming match.

    Args:
        home_team: Home team name
        away_team: Away team name
        match_date: Date of the match
        home_starters: Predicted home starting XI (optional)
        away_starters: Predicted away starting XI (optional)
        season: Season string

    Returns:
        Dictionary of form features for the match
    """
    features = {}

    # Team-level form (doesn't require lineup)
    home_form = compute_team_form_features(home_team, match_date, season=season)
    away_form = compute_team_form_features(away_team, match_date, season=season)

    for key, value in home_form.items():
        features[f"home_{key}"] = value
    for key, value in away_form.items():
        features[f"away_{key}"] = value

    # Squad-level form (requires predicted lineup)
    if home_starters:
        home_squad = compute_squad_form(home_team, home_starters, match_date, season=season)
        for key, value in home_squad.items():
            features[f"home_{key}"] = value

    if away_starters:
        away_squad = compute_squad_form(away_team, away_starters, match_date, season=season)
        for key, value in away_squad.items():
            features[f"away_{key}"] = value

    # Form differentials
    features["form_xg_diff"] = features.get("home_team_avg_xg_per_90", 0) - features.get("away_team_avg_xg_per_90", 0)
    features["form_goals_diff"] = features.get("home_team_top_scorer_goals", 0) - features.get("away_team_top_scorer_goals", 0)

    return features


if __name__ == "__main__":
    # Test the module
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing player form computation...")

    # Test individual player form
    form = compute_player_form(
        player_name="Lautaro Martínez",
        team="Inter",
        before_date=date(2025, 5, 1),
        n_matches=5,
    )

    if form:
        print(f"\nLautaro Martínez form (last 5 matches):")
        print(f"  Matches: {form.matches_played}")
        print(f"  Goals: {form.goals}, Assists: {form.assists}")
        print(f"  xG: {form.xg}, xA: {form.xa}")
        print(f"  xG+xA per 90: {form.xg_xa_per_90}")
        print(f"  Form score: {form.form_score}")
    else:
        print("No form data available for Lautaro Martínez")

    # Test team form
    print("\nTesting team form...")
    team_form = compute_team_form_features("Inter", date(2025, 5, 1))
    print(f"\nInter team form:")
    for key, value in team_form.items():
        print(f"  {key}: {value}")
