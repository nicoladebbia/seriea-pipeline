"""Attack/defense strength ratings and Elo ratings.

Dixon-Coles-inspired Poisson ratings:
  attack_strength  = team_avg_goals_scored / league_avg_goals_scored
  defense_strength = team_avg_goals_conceded / league_avg_goals_conceded

Elo rating:
  K = 20, home advantage = 100 points
  expected = 1 / (1 + 10^((away_elo - home_elo - home_adv) / 400))
  update: new_elo = old_elo + K * (actual - expected)
"""

from __future__ import annotations

import pandas as pd

ELO_K = 20
ELO_HOME_ADV = 100
ELO_INITIAL = 1500
ELO_REVERT = 0.75  # Season-start regression: keep 75%, revert 25% to mean
ELO_PROMOTED = 1400  # Newly promoted teams start below average


def add_strength_ratings(team_log: pd.DataFrame) -> pd.DataFrame:
    """Add attack/defense strength ratings (season-expanding averages).

    Columns added:
      attack_strength, defense_strength,
      xg_attack_strength, xg_defense_strength
    """
    df = team_log.copy()
    df = df.sort_values(["team", "match_date"]).reset_index(drop=True)

    for stat, prefix in [
        ("goals_scored", ""),
        ("xg_for", "xg_"),
    ]:
        conceded_stat = "goals_conceded" if not prefix else "xg_against"

        # Season-expanding team average (shifted)
        team_for = df.groupby(["team", "season"])[stat].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )
        team_against = df.groupby(["team", "season"])[conceded_stat].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )

        # League average per season (shifted)
        league_for = df.groupby("season")[stat].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )
        league_against = df.groupby("season")[conceded_stat].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )

        # Avoid division by zero
        df[f"{prefix}attack_strength"] = (team_for / league_for).fillna(1.0)
        df[f"{prefix}defense_strength"] = (team_against / league_against).fillna(1.0)

    return df


def add_elo_ratings(matches: pd.DataFrame) -> pd.DataFrame:
    """Add Elo ratings to the matches table.

    Processes matches chronologically and updates ratings after each match.
    At each new season boundary:
      - Existing teams regress toward ELO_INITIAL (75% old + 25% mean)
      - Newly promoted teams start at ELO_PROMOTED (below average)

    Columns added:
      home_elo, away_elo (pre-match ratings), elo_diff
    """
    df = matches.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.sort_values("match_date").reset_index(drop=True)

    # Pre-compute which teams play in each season
    season_teams: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        s = row["season"]
        if s not in season_teams:
            season_teams[s] = set()
        season_teams[s].add(row["home_team"])
        season_teams[s].add(row["away_team"])

    elo: dict[str, float] = {}
    home_elos: list[float] = []
    away_elos: list[float] = []
    current_season: str | None = None
    previous_season_teams: set[str] = set()
    # Track per-team match count in current season for variable K-factor
    season_match_count: dict[str, int] = {}

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        hs = row.get("home_score")
        as_ = row.get("away_score")
        season = row["season"]

        # Season boundary: regress ratings toward the mean
        if season != current_season:
            if current_season is not None:
                prev_teams = season_teams.get(current_season, set())
                for team in list(elo):
                    # Regress: 75% carry-over + 25% mean
                    elo[team] = ELO_REVERT * elo[team] + (1 - ELO_REVERT) * ELO_INITIAL
                # Newly promoted teams (in this season but not previous)
                new_teams = season_teams.get(season, set()) - prev_teams
                for team in new_teams:
                    elo[team] = ELO_PROMOTED
            current_season = season
            season_match_count = {}  # Reset match counts for new season

        # Initialize if truly first appearance ever
        if home not in elo:
            elo[home] = ELO_INITIAL
        if away not in elo:
            elo[away] = ELO_INITIAL

        # Record pre-match Elo
        home_elos.append(elo[home])
        away_elos.append(elo[away])

        # Update if result is known
        if pd.notna(hs) and pd.notna(as_):
            hs, as_ = int(hs), int(as_)
            expected_home = 1 / (1 + 10 ** ((elo[away] - elo[home] - ELO_HOME_ADV) / 400))

            if hs > as_:
                actual_home = 1.0
            elif hs < as_:
                actual_home = 0.0
            else:
                actual_home = 0.5

            # Variable K-factor: higher early in season for faster adaptation
            home_matches = season_match_count.get(home, 0)
            away_matches = season_match_count.get(away, 0)
            avg_matches = (home_matches + away_matches) / 2
            if avg_matches <= 5:
                k = 40  # High volatility early season
            elif avg_matches <= 15:
                k = 30  # Moderate mid-season
            else:
                k = ELO_K  # Stable (default 20)

            delta = k * (actual_home - expected_home)
            elo[home] += delta
            elo[away] -= delta

            # Increment match counts
            season_match_count[home] = home_matches + 1
            season_match_count[away] = away_matches + 1

    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    return df
