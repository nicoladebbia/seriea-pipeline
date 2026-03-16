"""Predict likely starting lineups for upcoming matches.

This module predicts the most likely starting XI for a team based on:
1. Historical lineup data (who starts most often)
2. Player minutes from player_stats
3. Injuries (from injury scraper)
4. Suspensions (from suspension tracker)

The prediction helps estimate team strength for upcoming matches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from storage.paths import parsed_path

log = logging.getLogger(__name__)


@dataclass
class PredictedLineup:
    """Predicted starting lineup for a team."""
    team: str
    formation: str
    predicted_starters: list[str]  # 11 players
    bench: list[str]  # Likely bench players
    missing_regulars: list[str]  # Regular starters who are unavailable
    confidence: float  # 0-1, how confident we are in this prediction
    notes: list[str]  # Explanations/warnings


def load_lineups() -> pd.DataFrame:
    """Load historical lineup data."""
    path = parsed_path("lineups")
    if not path.exists():
        log.warning(f"Lineups not found at {path}")
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_player_stats() -> pd.DataFrame:
    """Load player stats for minutes played."""
    path = parsed_path("player_stats")
    if not path.exists():
        log.warning(f"Player stats not found at {path}")
        return pd.DataFrame()

    df = pd.read_parquet(path)
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0)
    return df


def get_starter_frequency(
    team: str,
    season: str,
    n_recent_matches: int = 10,
    before_date: Optional[date] = None,
) -> pd.DataFrame:
    """Calculate how often each player starts for a team.

    Args:
        team: Team name
        season: Season string
        n_recent_matches: Number of recent matches to consider
        before_date: Only consider matches before this date

    Returns:
        DataFrame with player, start_count, start_frequency, total_matches
    """
    lineups = load_lineups()

    if lineups.empty:
        return pd.DataFrame()

    # Filter to team and season (extract season from match_id)
    team_lineups = lineups[lineups["team"] == team].copy()

    if team_lineups.empty:
        return pd.DataFrame()

    # Extract match date from match_id (format: YYYY-MM-DD_Home_Away)
    team_lineups["match_date"] = pd.to_datetime(
        team_lineups["match_id"].str[:10],
        errors="coerce"
    )

    # Filter by date
    if before_date:
        team_lineups = team_lineups[team_lineups["match_date"].dt.date < before_date]

    # Get most recent matches
    match_dates = team_lineups["match_date"].drop_duplicates().sort_values(ascending=False)
    recent_matches = match_dates.head(n_recent_matches)
    team_lineups = team_lineups[team_lineups["match_date"].isin(recent_matches)]

    if team_lineups.empty:
        return pd.DataFrame()

    # Count starters
    starters = team_lineups[team_lineups["role"] == "starter"]
    total_matches = starters["match_id"].nunique()

    frequency = starters.groupby("player_name").size().reset_index(name="start_count")
    frequency["total_matches"] = total_matches
    frequency["start_frequency"] = frequency["start_count"] / total_matches

    return frequency.sort_values("start_frequency", ascending=False)


def get_recent_formation(
    team: str,
    n_recent_matches: int = 5,
    before_date: Optional[date] = None,
) -> str:
    """Get the most commonly used formation in recent matches.

    Args:
        team: Team name
        n_recent_matches: Number of recent matches to consider
        before_date: Only consider matches before this date

    Returns:
        Formation string (e.g., "4-3-3") or "Unknown"
    """
    lineups = load_lineups()

    if lineups.empty:
        return "Unknown"

    team_lineups = lineups[lineups["team"] == team].copy()

    if team_lineups.empty:
        return "Unknown"

    # Extract match date
    team_lineups["match_date"] = pd.to_datetime(
        team_lineups["match_id"].str[:10],
        errors="coerce"
    )

    if before_date:
        team_lineups = team_lineups[team_lineups["match_date"].dt.date < before_date]

    # Get most recent matches
    match_dates = team_lineups["match_date"].drop_duplicates().sort_values(ascending=False)
    recent_matches = match_dates.head(n_recent_matches)
    team_lineups = team_lineups[team_lineups["match_date"].isin(recent_matches)]

    if team_lineups.empty:
        return "Unknown"

    # Get most common formation
    formations = team_lineups[["match_id", "formation"]].drop_duplicates()
    formation_counts = formations["formation"].value_counts()

    if formation_counts.empty:
        return "Unknown"

    return formation_counts.index[0]


def get_player_minutes(
    team: str,
    season: str,
    n_recent_matches: int = 10,
    before_date: Optional[date] = None,
) -> pd.DataFrame:
    """Get total minutes played by each player in recent matches.

    Args:
        team: Team name
        season: Season string
        n_recent_matches: Number of recent matches
        before_date: Only consider matches before this date

    Returns:
        DataFrame with player, total_minutes, avg_minutes
    """
    player_stats = load_player_stats()

    if player_stats.empty:
        return pd.DataFrame()

    # Filter to team and season
    team_stats = player_stats[
        (player_stats["team"] == team) &
        (player_stats["season"] == season)
    ].copy()

    if team_stats.empty:
        return pd.DataFrame()

    # Convert date and filter
    team_stats["match_date"] = pd.to_datetime(team_stats["match_date"], errors="coerce")

    if before_date:
        team_stats = team_stats[team_stats["match_date"].dt.date < before_date]

    # Get most recent matches
    match_dates = team_stats["match_date"].drop_duplicates().sort_values(ascending=False)
    recent_matches = match_dates.head(n_recent_matches)
    team_stats = team_stats[team_stats["match_date"].isin(recent_matches)]

    if team_stats.empty:
        return pd.DataFrame()

    # Aggregate minutes per player
    minutes = team_stats.groupby("player")["minutes"].agg(["sum", "mean", "count"]).reset_index()
    minutes.columns = ["player", "total_minutes", "avg_minutes", "matches_played"]

    return minutes.sort_values("total_minutes", ascending=False)


def predict_lineup(
    team: str,
    match_date: date,
    injuries: Optional[list[str]] = None,
    suspensions: Optional[list[str]] = None,
    n_recent_matches: int = 5,
    season: Optional[str] = None,
) -> PredictedLineup:
    """Predict the starting lineup for a team.

    Algorithm:
    1. Get starter frequency from recent lineups
    2. Get minutes played from player_stats
    3. Remove injured and suspended players
    4. Score players: frequency * 0.5 + minutes_rank * 0.3 + recency * 0.2
    5. Select top 11 by score
    6. Get most common recent formation

    Args:
        team: Team name
        match_date: Date of the upcoming match
        injuries: List of injured player names
        suspensions: List of suspended player names
        n_recent_matches: Number of recent matches to analyze
        season: Season string (auto-detected if not provided)

    Returns:
        PredictedLineup object
    """
    injuries = injuries or []
    suspensions = suspensions or []

    # Determine season
    if season is None:
        year = match_date.year
        month = match_date.month
        if month >= 8:
            season = f"{year}-{year + 1}"
        else:
            season = f"{year - 1}-{year}"

    notes = []

    # Get starter frequency
    frequency = get_starter_frequency(team, season, n_recent_matches, before_date=match_date)

    if frequency.empty:
        log.warning(f"No lineup data for {team}")
        return PredictedLineup(
            team=team,
            formation="Unknown",
            predicted_starters=[],
            bench=[],
            missing_regulars=[],
            confidence=0.0,
            notes=["No historical lineup data available"],
        )

    # Get minutes data
    minutes = get_player_minutes(team, season, n_recent_matches, before_date=match_date)

    # Merge frequency and minutes
    if not minutes.empty:
        # Normalize player names for matching
        frequency["player_lower"] = frequency["player_name"].str.lower()
        minutes["player_lower"] = minutes["player"].str.lower()

        combined = frequency.merge(
            minutes[["player_lower", "total_minutes", "avg_minutes"]],
            on="player_lower",
            how="left"
        )
        combined["total_minutes"] = combined["total_minutes"].fillna(0)
    else:
        combined = frequency.copy()
        combined["total_minutes"] = 0

    # Calculate composite score
    max_freq = combined["start_frequency"].max()
    max_mins = combined["total_minutes"].max() if combined["total_minutes"].max() > 0 else 1

    combined["freq_score"] = combined["start_frequency"] / max_freq if max_freq > 0 else 0
    combined["mins_score"] = combined["total_minutes"] / max_mins

    combined["composite_score"] = (
        combined["freq_score"] * 0.6 +
        combined["mins_score"] * 0.4
    )

    # Remove injured players
    unavailable = set(p.lower() for p in injuries + suspensions)
    combined["player_lower"] = combined["player_name"].str.lower()

    available = combined[~combined["player_lower"].isin(unavailable)]
    missing = combined[combined["player_lower"].isin(unavailable)]

    # Track missing regular starters
    missing_regulars = []
    for _, row in missing.iterrows():
        if row["start_frequency"] >= 0.5:  # Started in 50%+ of matches
            missing_regulars.append(row["player_name"])
            if row["player_lower"] in [p.lower() for p in injuries]:
                notes.append(f"{row['player_name']} (injured)")
            else:
                notes.append(f"{row['player_name']} (suspended)")

    # Select top 11 starters
    available = available.sort_values("composite_score", ascending=False)

    if len(available) < 11:
        notes.append(f"Only {len(available)} players available, need 11")
        predicted_starters = available["player_name"].tolist()
        bench = []
        confidence = 0.3
    else:
        predicted_starters = available.head(11)["player_name"].tolist()
        bench = available.iloc[11:20]["player_name"].tolist()

        # Calculate confidence based on data quality
        avg_freq = available.head(11)["start_frequency"].mean()
        data_matches = frequency["total_matches"].iloc[0] if not frequency.empty else 0

        confidence = min(1.0, (avg_freq * 0.7) + (min(data_matches, 10) / 10 * 0.3))

    # Get formation
    formation = get_recent_formation(team, n_recent_matches, before_date=match_date)

    return PredictedLineup(
        team=team,
        formation=formation,
        predicted_starters=predicted_starters,
        bench=bench,
        missing_regulars=missing_regulars,
        confidence=round(confidence, 2),
        notes=notes,
    )


def compute_lineup_features(
    home_team: str,
    away_team: str,
    match_date: date,
    home_injuries: Optional[list[str]] = None,
    away_injuries: Optional[list[str]] = None,
    home_suspensions: Optional[list[str]] = None,
    away_suspensions: Optional[list[str]] = None,
) -> dict:
    """Compute lineup-related features for an upcoming match.

    Args:
        home_team: Home team name
        away_team: Away team name
        match_date: Date of the match
        home_injuries: List of injured home players
        away_injuries: List of injured away players
        home_suspensions: List of suspended home players
        away_suspensions: List of suspended away players

    Returns:
        Dictionary of features
    """
    home_lineup = predict_lineup(
        home_team, match_date,
        injuries=home_injuries,
        suspensions=home_suspensions,
    )
    away_lineup = predict_lineup(
        away_team, match_date,
        injuries=away_injuries,
        suspensions=away_suspensions,
    )

    return {
        "home_lineup_confidence": home_lineup.confidence,
        "away_lineup_confidence": away_lineup.confidence,
        "home_missing_regulars": len(home_lineup.missing_regulars),
        "away_missing_regulars": len(away_lineup.missing_regulars),
        "home_formation": home_lineup.formation,
        "away_formation": away_lineup.formation,
    }


if __name__ == "__main__":
    # Test the module
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing lineup prediction...")

    # Test for Inter
    lineup = predict_lineup(
        team="Inter",
        match_date=date(2025, 5, 1),
        injuries=["Alessandro Bastoni"],  # Simulate injury
        suspensions=[],
    )

    print(f"\nPredicted lineup for Inter:")
    print(f"Formation: {lineup.formation}")
    print(f"Confidence: {lineup.confidence}")
    print(f"\nPredicted starters ({len(lineup.predicted_starters)}):")
    for i, player in enumerate(lineup.predicted_starters, 1):
        print(f"  {i}. {player}")

    if lineup.missing_regulars:
        print(f"\nMissing regular starters: {lineup.missing_regulars}")

    if lineup.notes:
        print(f"\nNotes: {lineup.notes}")
