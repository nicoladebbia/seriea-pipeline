"""Track player suspensions from yellow and red card accumulation.

Serie A suspension rules:
- 5 yellow cards = 1 match ban (first threshold)
- 10 yellow cards = 2 match ban (second threshold)
- 15 yellow cards = 3 match ban (and so on)
- Red card = at least 1 match ban (varies by severity)
- Second yellow (in same match) = 1 match ban

This module:
1. Counts yellow cards per player per season from player_stats.parquet
2. Tracks red cards from events.parquet
3. Calculates who is suspended for upcoming matches
4. Generates suspension-related features for ML model
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd

from storage.paths import parsed_path

log = logging.getLogger(__name__)

# Serie A yellow card thresholds for suspension
YELLOW_THRESHOLDS = [5, 10, 15, 20]  # 1 match ban at each threshold


@dataclass
class SuspensionInfo:
    """Information about a player's suspension status."""
    player_name: str
    team: str
    season: str
    yellow_cards: int
    red_cards: int
    is_suspended: bool
    matches_suspended: int  # How many matches remaining on suspension
    at_risk: bool  # 1 yellow away from suspension
    reason: str  # "5 yellows", "red card", etc.


def load_player_stats() -> pd.DataFrame:
    """Load player stats with yellow/red card data."""
    path = parsed_path("player_stats")
    if not path.exists():
        log.warning(f"Player stats not found at {path}")
        return pd.DataFrame()

    df = pd.read_parquet(path)

    # Convert card columns to numeric
    for col in ["cards_yellow", "cards_red"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df


def load_events() -> pd.DataFrame:
    """Load match events (goals, red cards, etc.)."""
    path = parsed_path("events")
    if not path.exists():
        log.warning(f"Events not found at {path}")
        return pd.DataFrame()

    return pd.read_parquet(path)


def get_yellow_card_count(
    player: str,
    team: str,
    season: str,
    before_date: Optional[date] = None,
) -> int:
    """Count yellow cards for a player in a season before a given date.

    Args:
        player: Player name
        team: Team name
        season: Season string (e.g., "2024-2025")
        before_date: Only count yellows before this date (for suspension calc)

    Returns:
        Number of yellow cards accumulated
    """
    df = load_player_stats()

    if df.empty:
        return 0

    # Filter to player's matches for this team and season
    mask = (
        (df["player"].str.lower() == player.lower()) &
        (df["team"] == team) &
        (df["season"] == season)
    )

    if before_date:
        df["match_date"] = pd.to_datetime(df["match_date"])
        mask = mask & (df["match_date"].dt.date < before_date)

    player_matches = df[mask]

    if player_matches.empty:
        return 0

    return player_matches["cards_yellow"].sum()


def get_red_cards(
    player: str,
    team: str,
    season: str,
    before_date: Optional[date] = None,
) -> int:
    """Count red cards for a player in a season."""
    df = load_player_stats()

    if df.empty:
        return 0

    mask = (
        (df["player"].str.lower() == player.lower()) &
        (df["team"] == team) &
        (df["season"] == season)
    )

    if before_date:
        df["match_date"] = pd.to_datetime(df["match_date"])
        mask = mask & (df["match_date"].dt.date < before_date)

    player_matches = df[mask]

    if player_matches.empty:
        return 0

    return player_matches["cards_red"].sum()


def is_player_suspended(
    player: str,
    team: str,
    season: str,
    current_date: date,
) -> SuspensionInfo:
    """Check if a player is suspended for matches on or after current_date.

    This is an approximation - actual suspensions depend on when cards
    were received relative to match schedule.

    Args:
        player: Player name
        team: Team name
        season: Season string
        current_date: Date to check suspension status for

    Returns:
        SuspensionInfo object with suspension details
    """
    yellows = get_yellow_card_count(player, team, season, before_date=current_date)
    reds = get_red_cards(player, team, season, before_date=current_date)

    # Calculate suspension from yellows
    suspensions_from_yellows = 0
    for threshold in YELLOW_THRESHOLDS:
        if yellows >= threshold:
            suspensions_from_yellows += 1

    # Each red card = at least 1 match suspension
    suspensions_from_reds = reds

    # Total suspensions served this season
    total_suspension_matches = suspensions_from_yellows + suspensions_from_reds

    # Check if at risk (1 yellow from next threshold)
    at_risk = False
    for threshold in YELLOW_THRESHOLDS:
        if yellows == threshold - 1:
            at_risk = True
            break

    # Determine current status
    # This is simplified - in reality we'd need to track which matches
    # the player missed to know remaining suspension
    reason = ""
    is_suspended = False

    if reds > 0:
        reason = f"{reds} red card(s)"
    elif yellows >= 5:
        reason = f"{yellows} yellow cards"

    return SuspensionInfo(
        player_name=player,
        team=team,
        season=season,
        yellow_cards=yellows,
        red_cards=reds,
        is_suspended=is_suspended,  # Would need match-by-match tracking
        matches_suspended=0,
        at_risk=at_risk,
        reason=reason,
    )


def get_team_suspension_risk(
    team: str,
    season: str,
    current_date: Optional[date] = None,
) -> pd.DataFrame:
    """Get suspension status for all players on a team.

    Args:
        team: Team name
        season: Season string
        current_date: Date for calculating yellow counts

    Returns:
        DataFrame with player suspension info
    """
    df = load_player_stats()

    if df.empty:
        return pd.DataFrame()

    # Get unique players for this team/season
    team_mask = (df["team"] == team) & (df["season"] == season)
    team_df = df[team_mask]

    if team_df.empty:
        return pd.DataFrame()

    # Get latest data per player
    if current_date:
        team_df["match_date"] = pd.to_datetime(team_df["match_date"])
        team_df = team_df[team_df["match_date"].dt.date < current_date]

    # Aggregate yellow/red cards per player
    player_cards = team_df.groupby("player").agg({
        "cards_yellow": "sum",
        "cards_red": "sum",
    }).reset_index()

    # Add suspension risk flags
    player_cards["at_risk"] = player_cards["cards_yellow"].apply(
        lambda y: any(y == t - 1 for t in YELLOW_THRESHOLDS)
    )
    player_cards["suspension_level"] = player_cards["cards_yellow"].apply(
        lambda y: sum(1 for t in YELLOW_THRESHOLDS if y >= t)
    )

    player_cards["team"] = team
    player_cards["season"] = season

    return player_cards.sort_values("cards_yellow", ascending=False)


def get_suspended_players(
    team: str,
    match_date: date,
    season: Optional[str] = None,
) -> list[str]:
    """Get list of players who might be suspended for a match.

    Note: This is an approximation. For accurate suspension tracking,
    we'd need to track which matches players missed after each suspension.

    Args:
        team: Team name
        match_date: Date of the upcoming match
        season: Season string (auto-detected if not provided)

    Returns:
        List of player names who reached a suspension threshold
    """
    if season is None:
        # Determine season from match_date
        year = match_date.year
        month = match_date.month
        if month >= 8:
            season = f"{year}-{year + 1}"
        else:
            season = f"{year - 1}-{year}"

    risk_df = get_team_suspension_risk(team, season, match_date)

    if risk_df.empty:
        return []

    # Return players at exact threshold (just reached suspension)
    # This is approximate - real system would track suspension serving
    suspended = []
    for _, row in risk_df.iterrows():
        yellows = row["cards_yellow"]
        # Check if at exact threshold (just got suspended)
        if yellows in YELLOW_THRESHOLDS:
            suspended.append(row["player"])

    return suspended


def get_players_at_risk(
    team: str,
    match_date: date,
    season: Optional[str] = None,
) -> list[str]:
    """Get list of players one yellow card away from suspension.

    Args:
        team: Team name
        match_date: Date of the upcoming match
        season: Season string

    Returns:
        List of player names at risk of suspension
    """
    if season is None:
        year = match_date.year
        month = match_date.month
        if month >= 8:
            season = f"{year}-{year + 1}"
        else:
            season = f"{year - 1}-{year}"

    risk_df = get_team_suspension_risk(team, season, match_date)

    if risk_df.empty:
        return []

    at_risk = risk_df[risk_df["at_risk"] == True]["player"].tolist()
    return at_risk


def _compute_match_level_yellows(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Fallback: compute total_yellows from match-level yellow card columns.

    When player_stats is unavailable for a league (e.g. EPL), we can still
    compute cumulative season yellow totals per team from the match-level
    home_yellow_cards / away_yellow_cards columns.

    suspended_count and at_risk_count require player-level data and are left
    as 0 when only match-level data is available.
    """
    df = matches_df.copy()
    df["_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.sort_values("_date").reset_index(drop=True)

    has_yellows = (
        "home_yellow_cards" in df.columns and "away_yellow_cards" in df.columns
    )
    if not has_yellows:
        return matches_df

    # Accumulate yellows per team per season
    # Each match contributes yellows when the team plays home or away
    team_season_yellows: dict[tuple[str, str], int] = {}
    home_totals = []
    away_totals = []

    for _, row in df.iterrows():
        season = row.get("season", "")
        ht = row.get("home_team", "")
        at = row.get("away_team", "")
        h_yc = pd.to_numeric(row.get("home_yellow_cards", 0), errors="coerce")
        a_yc = pd.to_numeric(row.get("away_yellow_cards", 0), errors="coerce")
        if pd.isna(h_yc):
            h_yc = 0
        if pd.isna(a_yc):
            a_yc = 0

        # Record cumulative BEFORE this match
        home_totals.append(team_season_yellows.get((ht, season), 0))
        away_totals.append(team_season_yellows.get((at, season), 0))

        # Update cumulative after this match
        team_season_yellows[(ht, season)] = (
            team_season_yellows.get((ht, season), 0) + int(h_yc)
        )
        team_season_yellows[(at, season)] = (
            team_season_yellows.get((at, season), 0) + int(a_yc)
        )

    df["home_total_yellows"] = home_totals
    df["away_total_yellows"] = away_totals
    df["home_suspended_count"] = 0
    df["away_suspended_count"] = 0
    df["home_at_risk_count"] = 0
    df["away_at_risk_count"] = 0

    df.drop(columns=["_date"], inplace=True)
    return df


def compute_suspension_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Add suspension-related features to a matches DataFrame.

    Features added:
    - home_suspended_count: Number of suspended home players
    - away_suspended_count: Number of suspended away players
    - home_at_risk_count: Home players 1 yellow from suspension
    - away_at_risk_count: Away players 1 yellow from suspension
    - home_total_yellows: Team's total yellow card count
    - away_total_yellows: Team's total yellow card count

    When player-level stats are available (e.g. Serie A via FBref), all 6
    features are computed accurately. When only match-level yellow card
    columns exist (e.g. EPL via football-data.co.uk), total_yellows is
    computed as a cumulative season sum; suspended/at_risk default to 0.

    Args:
        matches_df: DataFrame with match_id, home_team, away_team, match_date, season

    Returns:
        DataFrame with suspension features added
    """
    log.info("Computing suspension features for %d matches", len(matches_df))

    player_stats = load_player_stats()
    susp_cols = [
        "home_suspended_count", "away_suspended_count",
        "home_at_risk_count", "away_at_risk_count",
        "home_total_yellows", "away_total_yellows",
    ]

    if player_stats.empty:
        log.warning("No player stats available — using match-level yellow fallback")
        return _compute_match_level_yellows(matches_df)

    # Determine which teams have player-level data
    player_stats["match_date"] = pd.to_datetime(player_stats["match_date"])
    teams_with_player_data = set(player_stats["team"].unique())

    # Split matches into those with player data and those without
    all_teams = set(matches_df["home_team"].unique()) | set(matches_df["away_team"].unique())
    teams_without_data = all_teams - teams_with_player_data

    if teams_without_data:
        log.info(
            "Player stats missing for %d teams — using match-level fallback for those",
            len(teams_without_data),
        )

    # Identify rows where BOTH teams have player data (full computation)
    # vs rows where at least one team lacks it (fallback)
    has_player_data = (
        matches_df["home_team"].isin(teams_with_player_data)
        & matches_df["away_team"].isin(teams_with_player_data)
    )

    df_player = matches_df[has_player_data].copy()
    df_fallback = matches_df[~has_player_data].copy()

    # --- Full player-level computation for teams with data ---
    results = []
    for _, match in df_player.iterrows():
        match_date = pd.to_datetime(match["match_date"]).date()
        season = match["season"]
        home_team = match["home_team"]
        away_team = match["away_team"]

        before_match = player_stats[
            (player_stats["season"] == season)
            & (player_stats["match_date"].dt.date < match_date)
        ]

        # Home team stats
        home_stats = before_match[before_match["team"] == home_team]
        home_player_yellows = home_stats.groupby("player")["cards_yellow"].sum()

        home_suspended = sum(1 for y in home_player_yellows if y in YELLOW_THRESHOLDS)
        home_at_risk = sum(
            1 for y in home_player_yellows if any(y == t - 1 for t in YELLOW_THRESHOLDS)
        )
        home_total_yellows = home_player_yellows.sum()

        # Away team stats
        away_stats = before_match[before_match["team"] == away_team]
        away_player_yellows = away_stats.groupby("player")["cards_yellow"].sum()

        away_suspended = sum(1 for y in away_player_yellows if y in YELLOW_THRESHOLDS)
        away_at_risk = sum(
            1 for y in away_player_yellows if any(y == t - 1 for t in YELLOW_THRESHOLDS)
        )
        away_total_yellows = away_player_yellows.sum()

        results.append({
            "match_id": match["match_id"],
            "home_suspended_count": home_suspended,
            "away_suspended_count": away_suspended,
            "home_at_risk_count": home_at_risk,
            "away_at_risk_count": away_at_risk,
            "home_total_yellows": home_total_yellows,
            "away_total_yellows": away_total_yellows,
        })

    if results:
        results_df = pd.DataFrame(results)
        df_player = df_player.merge(results_df, on="match_id", how="left")
    else:
        for col in susp_cols:
            df_player[col] = 0

    # --- Match-level fallback for teams without player data ---
    if not df_fallback.empty:
        df_fallback = _compute_match_level_yellows(df_fallback)

    # Recombine
    matches_df = pd.concat([df_player, df_fallback], ignore_index=True)

    # Fill NaN with 0
    for col in susp_cols:
        matches_df[col] = matches_df[col].fillna(0).astype(int)

    # Restore original sort order
    matches_df = matches_df.sort_values("match_date").reset_index(drop=True)

    log.info("Added suspension features to matches")
    return matches_df


if __name__ == "__main__":
    # Test the module
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing suspension tracking...")

    # Test team suspension risk
    risk = get_team_suspension_risk("Lazio", "2024-2025")
    if not risk.empty:
        print("\nLazio 2024-2025 yellow card status:")
        print(risk.head(10).to_string())

        at_risk = risk[risk["at_risk"] == True]
        print(f"\nPlayers at risk (1 yellow from ban): {len(at_risk)}")
        if not at_risk.empty:
            print(at_risk[["player", "cards_yellow"]].to_string())
    else:
        print("No data available for testing")
