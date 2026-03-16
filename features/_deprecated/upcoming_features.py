"""Build features for upcoming matches that haven't been played yet.

This module creates prediction features for future matches by:
1. Computing team historical stats (Elo, form, etc.)
2. Predicting lineups using historical data
3. Adding player form metrics for predicted starters
4. Including injury and suspension data
5. Computing head-to-head statistics

Unlike features for past matches, we cannot use betting odds or
actual match statistics - everything must be predictable before kickoff.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from features.lineup_prediction import predict_lineup, compute_lineup_features
from features.player_aggregation import build_match_player_features
from features.player_form import compute_match_form_features
from features.suspensions import get_suspended_players, get_players_at_risk
from features.venue import SERIE_A_STADIUMS, get_travel_distance, get_altitude_difference, get_stadium_capacity
from scraper.injuries import get_injured_players
from scraper.upcoming import UpcomingMatch, get_upcoming_fixtures
from storage.paths import parsed_path, features_path

log = logging.getLogger(__name__)


def load_historical_data() -> dict[str, pd.DataFrame]:
    """Load all historical data needed for feature computation."""
    data = {}

    # Load matches
    matches_path = parsed_path("matches")
    if matches_path.exists():
        data["matches"] = pd.read_parquet(matches_path)
        data["matches"]["match_date"] = pd.to_datetime(data["matches"]["match_date"])
    else:
        data["matches"] = pd.DataFrame()

    # Load player stats
    player_stats_path = parsed_path("player_stats")
    if player_stats_path.exists():
        data["player_stats"] = pd.read_parquet(player_stats_path)
    else:
        data["player_stats"] = pd.DataFrame()

    # Load existing features for Elo reference
    feat_path = features_path()
    if feat_path.exists():
        data["features"] = pd.read_parquet(feat_path)
    else:
        data["features"] = pd.DataFrame()

    return data


def get_team_elo(
    team: str,
    before_date: date,
    features_df: pd.DataFrame,
) -> float:
    """Get the most recent Elo rating for a team.

    Args:
        team: Team name
        before_date: Get Elo before this date
        features_df: Historical features DataFrame

    Returns:
        Elo rating (default 1500 if not found)
    """
    if features_df.empty:
        return 1500.0

    # Look for team's most recent match
    team_matches = features_df[
        (features_df["home_team"] == team) | (features_df["away_team"] == team)
    ].copy()

    if team_matches.empty:
        return 1500.0

    team_matches["match_date"] = pd.to_datetime(team_matches["match_date"])
    team_matches = team_matches[team_matches["match_date"].dt.date < before_date]

    if team_matches.empty:
        return 1500.0

    # Get most recent
    latest = team_matches.sort_values("match_date", ascending=False).iloc[0]

    if latest["home_team"] == team:
        return float(latest.get("home_elo", 1500.0))
    else:
        return float(latest.get("away_elo", 1500.0))


def get_team_rolling_stats(
    team: str,
    before_date: date,
    matches_df: pd.DataFrame,
    n_matches: int = 5,
) -> dict:
    """Compute rolling statistics for a team.

    Args:
        team: Team name
        before_date: Calculate stats before this date
        matches_df: Historical matches DataFrame
        n_matches: Number of recent matches

    Returns:
        Dictionary of rolling stats
    """
    if matches_df.empty:
        return _empty_rolling_stats()

    # Get team's recent matches
    team_home = matches_df[matches_df["home_team"] == team].copy()
    team_away = matches_df[matches_df["away_team"] == team].copy()

    team_home["match_date"] = pd.to_datetime(team_home["match_date"])
    team_away["match_date"] = pd.to_datetime(team_away["match_date"])

    team_home = team_home[team_home["match_date"].dt.date < before_date]
    team_away = team_away[team_away["match_date"].dt.date < before_date]

    # Combine and sort
    all_matches = []

    for _, m in team_home.iterrows():
        all_matches.append({
            "date": m["match_date"],
            "goals_for": m.get("home_score", 0) or 0,
            "goals_against": m.get("away_score", 0) or 0,
            "xg": m.get("home_xg", 0) or 0,
            "xga": m.get("away_xg", 0) or 0,
            "is_home": True,
        })

    for _, m in team_away.iterrows():
        all_matches.append({
            "date": m["match_date"],
            "goals_for": m.get("away_score", 0) or 0,
            "goals_against": m.get("home_score", 0) or 0,
            "xg": m.get("away_xg", 0) or 0,
            "xga": m.get("home_xg", 0) or 0,
            "is_home": False,
        })

    if not all_matches:
        return _empty_rolling_stats()

    # Sort by date and take recent
    all_matches.sort(key=lambda x: x["date"], reverse=True)
    recent = all_matches[:n_matches]

    if not recent:
        return _empty_rolling_stats()

    # Compute aggregates
    goals_scored = sum(m["goals_for"] for m in recent)
    goals_conceded = sum(m["goals_against"] for m in recent)
    xg_total = sum(m["xg"] for m in recent)
    xga_total = sum(m["xga"] for m in recent)

    wins = sum(1 for m in recent if m["goals_for"] > m["goals_against"])
    draws = sum(1 for m in recent if m["goals_for"] == m["goals_against"])
    losses = sum(1 for m in recent if m["goals_for"] < m["goals_against"])

    n = len(recent)

    return {
        "goals_scored": goals_scored,
        "goals_conceded": goals_conceded,
        "goals_scored_avg": goals_scored / n,
        "goals_conceded_avg": goals_conceded / n,
        "goal_diff": goals_scored - goals_conceded,
        "xg_total": xg_total,
        "xga_total": xga_total,
        "xg_avg": xg_total / n,
        "xga_avg": xga_total / n,
        "xg_diff": xg_total - xga_total,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "points": wins * 3 + draws,
        "form_points_pct": (wins * 3 + draws) / (n * 3),
        "matches": n,
    }


def _empty_rolling_stats() -> dict:
    """Return empty rolling stats dict."""
    return {
        "goals_scored": 0, "goals_conceded": 0,
        "goals_scored_avg": 0, "goals_conceded_avg": 0, "goal_diff": 0,
        "xg_total": 0, "xga_total": 0, "xg_avg": 0, "xga_avg": 0, "xg_diff": 0,
        "wins": 0, "draws": 0, "losses": 0, "points": 0,
        "form_points_pct": 0, "matches": 0,
    }


def get_h2h_stats(
    home_team: str,
    away_team: str,
    before_date: date,
    matches_df: pd.DataFrame,
    n_matches: int = 10,
) -> dict:
    """Compute head-to-head statistics between two teams.

    Args:
        home_team: Home team name
        away_team: Away team name
        before_date: Only consider matches before this date
        matches_df: Historical matches DataFrame
        n_matches: Max number of H2H matches to consider

    Returns:
        Dictionary of H2H statistics
    """
    if matches_df.empty:
        return {
            "h2h_matches": 0,
            "h2h_home_wins": 0,
            "h2h_away_wins": 0,
            "h2h_draws": 0,
            "h2h_home_goals": 0,
            "h2h_away_goals": 0,
            "h2h_home_win_pct": 0.5,
        }

    matches_df = matches_df.copy()
    matches_df["match_date"] = pd.to_datetime(matches_df["match_date"])

    # Get matches between these teams
    h2h = matches_df[
        ((matches_df["home_team"] == home_team) & (matches_df["away_team"] == away_team)) |
        ((matches_df["home_team"] == away_team) & (matches_df["away_team"] == home_team))
    ].copy()

    h2h = h2h[h2h["match_date"].dt.date < before_date]
    h2h = h2h.sort_values("match_date", ascending=False).head(n_matches)

    if h2h.empty:
        return {
            "h2h_matches": 0,
            "h2h_home_wins": 0,
            "h2h_away_wins": 0,
            "h2h_draws": 0,
            "h2h_home_goals": 0,
            "h2h_away_goals": 0,
            "h2h_home_win_pct": 0.5,
        }

    home_wins = 0
    away_wins = 0
    draws = 0
    home_goals = 0
    away_goals = 0

    for _, m in h2h.iterrows():
        h_score = m.get("home_score", 0) or 0
        a_score = m.get("away_score", 0) or 0

        if m["home_team"] == home_team:
            # Match with same home/away configuration
            home_goals += h_score
            away_goals += a_score
            if h_score > a_score:
                home_wins += 1
            elif a_score > h_score:
                away_wins += 1
            else:
                draws += 1
        else:
            # Reversed fixture
            home_goals += a_score
            away_goals += h_score
            if a_score > h_score:
                home_wins += 1
            elif h_score > a_score:
                away_wins += 1
            else:
                draws += 1

    total = len(h2h)

    return {
        "h2h_matches": total,
        "h2h_home_wins": home_wins,
        "h2h_away_wins": away_wins,
        "h2h_draws": draws,
        "h2h_home_goals": home_goals,
        "h2h_away_goals": away_goals,
        "h2h_home_win_pct": (home_wins + draws * 0.5) / total if total > 0 else 0.5,
    }


def build_upcoming_match_features(
    match: UpcomingMatch,
    historical_data: dict[str, pd.DataFrame],
) -> dict:
    """Build all features for a single upcoming match.

    Args:
        match: UpcomingMatch object
        historical_data: Dict with matches, player_stats, features DataFrames

    Returns:
        Dictionary of features for this match
    """
    features = {
        "match_id": match.match_id,
        "season": match.season,
        "matchweek": match.matchweek,
        "match_date": str(match.match_date),
        "home_team": match.home_team,
        "away_team": match.away_team,
        "league": "serie_a",
    }

    matches_df = historical_data.get("matches", pd.DataFrame())
    features_df = historical_data.get("features", pd.DataFrame())

    # 1. Elo ratings
    home_elo = get_team_elo(match.home_team, match.match_date, features_df)
    away_elo = get_team_elo(match.away_team, match.match_date, features_df)

    features["home_elo"] = round(home_elo, 1)
    features["away_elo"] = round(away_elo, 1)
    features["elo_diff"] = round(home_elo - away_elo, 1)

    # 2. Rolling team stats
    home_stats = get_team_rolling_stats(match.home_team, match.match_date, matches_df)
    away_stats = get_team_rolling_stats(match.away_team, match.match_date, matches_df)

    for key, val in home_stats.items():
        features[f"home_roll_{key}"] = round(val, 2) if isinstance(val, float) else val
    for key, val in away_stats.items():
        features[f"away_roll_{key}"] = round(val, 2) if isinstance(val, float) else val

    # 3. Head-to-head
    h2h = get_h2h_stats(match.home_team, match.away_team, match.match_date, matches_df)
    features.update(h2h)

    # 4. Get injuries and suspensions
    home_injuries = get_injured_players(match.home_team, match.match_date)
    away_injuries = get_injured_players(match.away_team, match.match_date)
    home_suspensions = get_suspended_players(match.home_team, match.match_date)
    away_suspensions = get_suspended_players(match.away_team, match.match_date)

    features["home_injuries"] = len(home_injuries)
    features["away_injuries"] = len(away_injuries)
    features["home_suspensions"] = len(home_suspensions)
    features["away_suspensions"] = len(away_suspensions)

    # Players at risk
    home_at_risk = get_players_at_risk(match.home_team, match.match_date)
    away_at_risk = get_players_at_risk(match.away_team, match.match_date)
    features["home_players_at_risk"] = len(home_at_risk)
    features["away_players_at_risk"] = len(away_at_risk)

    # 5. Lineup prediction
    lineup_features = compute_lineup_features(
        match.home_team,
        match.away_team,
        match.match_date,
        home_injuries=home_injuries,
        away_injuries=away_injuries,
        home_suspensions=home_suspensions,
        away_suspensions=away_suspensions,
    )
    features.update(lineup_features)

    # 6. Player form features
    form_features = compute_match_form_features(
        match.home_team,
        match.away_team,
        match.match_date,
        season=match.season,
    )
    features.update(form_features)

    # 7. Player-level aggregation features (xG, touches, creative/defensive output)
    try:
        player_features = build_match_player_features(
            home_team=match.home_team,
            away_team=match.away_team,
            match_date=match.match_date,
            n_matches=5,
        )
        features.update(player_features)
    except Exception as e:
        log.warning(f"Failed to compute player aggregation features: {e}")

    # 8. Venue features (stadium, travel, altitude)
    try:
        # Stadium capacity
        features["home_stadium_capacity"] = get_stadium_capacity(match.home_team)

        # Travel distance for away team
        travel_dist = get_travel_distance(match.home_team, match.away_team)
        features["travel_distance_km"] = round(travel_dist, 1)
        features["travel_distance_norm"] = round(travel_dist / 1000, 3)
        features["long_travel"] = 1 if travel_dist > 500 else 0

        # Altitude difference
        alt_diff = get_altitude_difference(match.home_team, match.away_team)
        features["altitude_diff"] = alt_diff
        features["altitude_advantage"] = 1 if alt_diff > 200 else 0

        # Stadium info
        stadium_info = SERIE_A_STADIUMS.get(match.home_team, {})
        features["home_altitude_m"] = stadium_info.get("altitude_m", 0)
    except Exception as e:
        log.warning(f"Failed to compute venue features: {e}")
        features["home_stadium_capacity"] = 0
        features["travel_distance_km"] = 0
        features["travel_distance_norm"] = 0
        features["long_travel"] = 0
        features["altitude_diff"] = 0
        features["altitude_advantage"] = 0
        features["home_altitude_m"] = 0

    # 9. Derived features
    features["attack_strength_diff"] = (
        features.get("home_roll_xg_avg", 0) - features.get("away_roll_xg_avg", 0)
    )
    features["defense_strength_diff"] = (
        features.get("away_roll_xga_avg", 0) - features.get("home_roll_xga_avg", 0)
    )
    features["form_diff"] = (
        features.get("home_roll_form_points_pct", 0) - features.get("away_roll_form_points_pct", 0)
    )

    # Set flag that this has no odds (for model training)
    features["_has_odds"] = False

    return features


def build_all_upcoming_features(
    days_ahead: int = 14,
    refresh_fixtures: bool = False,
) -> pd.DataFrame:
    """Build features for all upcoming matches.

    Args:
        days_ahead: Days to look ahead for fixtures
        refresh_fixtures: Force refresh fixtures from source

    Returns:
        DataFrame with features for each upcoming match
    """
    log.info("Building features for upcoming matches...")

    # Get fixtures
    fixtures = get_upcoming_fixtures(days_ahead=days_ahead, refresh=refresh_fixtures)

    if not fixtures:
        log.warning("No upcoming fixtures found")
        return pd.DataFrame()

    log.info(f"Found {len(fixtures)} upcoming fixtures")

    # Load historical data once
    historical = load_historical_data()

    # Build features for each match
    all_features = []
    for match in fixtures:
        try:
            features = build_upcoming_match_features(match, historical)
            all_features.append(features)
            log.info(f"Built features for {match.home_team} vs {match.away_team}")
        except Exception as e:
            log.error(f"Failed to build features for {match.match_id}: {e}")

    if not all_features:
        return pd.DataFrame()

    df = pd.DataFrame(all_features)
    log.info(f"Built features for {len(df)} upcoming matches with {len(df.columns)} features")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing upcoming match features...")

    # Build features for upcoming matches
    df = build_all_upcoming_features(days_ahead=14, refresh_fixtures=True)

    if not df.empty:
        print(f"\nBuilt features for {len(df)} matches")
        print(f"Feature count: {len(df.columns)}")

        # Show summary
        print("\nUpcoming matches:")
        for _, row in df.iterrows():
            print(f"  {row['match_date']} - {row['home_team']} vs {row['away_team']}")
            print(f"    Elo: {row.get('home_elo', 'N/A')} vs {row.get('away_elo', 'N/A')}")
            print(f"    Form: {row.get('home_roll_form_points_pct', 'N/A'):.2f} vs {row.get('away_roll_form_points_pct', 'N/A'):.2f}")
            print(f"    Injuries: {row.get('home_injuries', 0)} vs {row.get('away_injuries', 0)}")
    else:
        print("No features built")
