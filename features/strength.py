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

    League-aware: when a ``league`` column is present, league averages are
    computed per (season, league) so that teams from different competitions
    are not mixed.  Without the column the behaviour is unchanged (backward
    compatible).

    Columns added:
      attack_strength, defense_strength,
      xg_attack_strength, xg_defense_strength
    """
    df = team_log.copy()
    df = df.sort_values(["team", "match_date"]).reset_index(drop=True)

    has_league = "league" in df.columns
    # Group key for league-level averages: scope by league when available
    league_group = ["season", "league"] if has_league else ["season"]

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

        # League average per season (shifted), scoped by league when present
        league_for = df.groupby(league_group)[stat].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )
        league_against = df.groupby(league_group)[conceded_stat].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )

        # Avoid division by zero
        df[f"{prefix}attack_strength"] = (team_for / league_for).fillna(1.0)
        df[f"{prefix}defense_strength"] = (team_against / league_against).fillna(1.0)

    return df


def _compute_elo_for_group(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Elo ratings for a single league group of matches.

    Expects *df* to be sorted by ``match_date`` already.  Returns the same
    DataFrame with ``home_elo``, ``away_elo``, and ``elo_diff`` columns.
    """
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
                    elo[team] = ELO_REVERT * elo[team] + (1 - ELO_REVERT) * ELO_INITIAL
                new_teams = season_teams.get(season, set()) - prev_teams
                for team in new_teams:
                    elo[team] = ELO_PROMOTED
            current_season = season
            season_match_count = {}

        if home not in elo:
            elo[home] = ELO_INITIAL
        if away not in elo:
            elo[away] = ELO_INITIAL

        home_elos.append(elo[home])
        away_elos.append(elo[away])

        if pd.notna(hs) and pd.notna(as_):
            hs, as_ = int(hs), int(as_)
            expected_home = 1 / (1 + 10 ** ((elo[away] - elo[home] - ELO_HOME_ADV) / 400))

            if hs > as_:
                actual_home = 1.0
            elif hs < as_:
                actual_home = 0.0
            else:
                actual_home = 0.5

            home_matches = season_match_count.get(home, 0)
            away_matches = season_match_count.get(away, 0)
            avg_matches = (home_matches + away_matches) / 2
            if avg_matches <= 5:
                k = 40
            elif avg_matches <= 15:
                k = 30
            else:
                k = ELO_K

            delta = k * (actual_home - expected_home)
            elo[home] += delta
            elo[away] -= delta

            season_match_count[home] = home_matches + 1
            season_match_count[away] = away_matches + 1

    df = df.copy()
    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    return df


def add_elo_ratings(matches: pd.DataFrame) -> pd.DataFrame:
    """Add Elo ratings to the matches table.

    League-aware: when a ``league`` column is present, Elo pools are
    maintained independently per league so that teams from different
    competitions never influence each other.  Without the column the
    behaviour is unchanged (backward compatible).

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

    if "league" in df.columns:
        # Process each league independently, then reassemble in original order
        parts: list[pd.DataFrame] = []
        for _league, group in df.groupby("league"):
            group = group.sort_values("match_date").reset_index(drop=False)
            group = _compute_elo_for_group(group)
            parts.append(group)
        result = pd.concat(parts).sort_values("index").reset_index(drop=True)
        result.drop(columns=["index"], inplace=True)
        return result

    return _compute_elo_for_group(df)
