"""League standing and motivation features.

Computes the current league position, points total, goal difference,
and motivation indicators (relegation zone, CL zone, title race) for
each team at each match — using only data available BEFORE that match
to prevent leakage.
"""

from __future__ import annotations

import pandas as pd


def add_league_position_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Add league standing features to the match-level table.

    For each match, computes the standings as of the PREVIOUS matchweek
    for both home and away teams.

    Columns added (for both home_ and away_ prefixes):
      {prefix}_league_pos         - current league position (1-20)
      {prefix}_league_points      - cumulative points
      {prefix}_league_gd          - goal difference
      {prefix}_league_goals_for   - total goals scored
      {prefix}_league_wins        - total wins
      {prefix}_league_draws       - total draws
      {prefix}_league_losses      - total losses
      {prefix}_in_relegation_zone - 1 if position >= 18
      {prefix}_in_cl_zone         - 1 if position <= 4
      {prefix}_in_el_zone         - 1 if position in [5, 6]
      {prefix}_in_title_race      - 1 if position <= 2
      league_position_diff        - home position - away position (negative = home higher)
    """
    df = matches.copy()
    df["_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.sort_values("_date").reset_index(drop=True)

    # Build cumulative standings season by season
    for prefix, team_col in [("home", "home_team"), ("away", "away_team")]:
        df[f"{prefix}_league_pos"] = None
        df[f"{prefix}_league_points"] = None
        df[f"{prefix}_league_gd"] = None
        df[f"{prefix}_league_goals_for"] = None
        df[f"{prefix}_league_wins"] = None
        df[f"{prefix}_league_draws"] = None
        df[f"{prefix}_league_losses"] = None

    # When a "league" column is present, standings must be computed per
    # (season, league) so that teams from different competitions are not
    # mixed into a single table.  Without the column the grouping falls
    # back to season-only (backward compatible).
    has_league = "league" in df.columns
    if has_league:
        group_iter = df.groupby(["season", "league"])
    else:
        group_iter = df.groupby("season")

    for _group_key, season_df_raw in group_iter:
        season_mask = season_df_raw.index
        season_df = df.loc[season_mask].copy()

        # Track team stats incrementally
        team_stats: dict[str, dict] = {}

        for idx, row in season_df.iterrows():
            home = row["home_team"]
            away = row["away_team"]

            # Initialize teams if first appearance
            for team in [home, away]:
                if team not in team_stats:
                    team_stats[team] = {
                        "points": 0, "gd": 0, "gf": 0,
                        "wins": 0, "draws": 0, "losses": 0,
                        "played": 0,
                    }

            # Compute standings BEFORE this match
            standings = _compute_standings(team_stats)

            # Assign standings to this row
            for prefix, team in [("home", home), ("away", away)]:
                pos_data = standings.get(team)
                if pos_data:
                    df.at[idx, f"{prefix}_league_pos"] = pos_data["position"]
                    df.at[idx, f"{prefix}_league_points"] = pos_data["points"]
                    df.at[idx, f"{prefix}_league_gd"] = pos_data["gd"]
                    df.at[idx, f"{prefix}_league_goals_for"] = pos_data["gf"]
                    df.at[idx, f"{prefix}_league_wins"] = pos_data["wins"]
                    df.at[idx, f"{prefix}_league_draws"] = pos_data["draws"]
                    df.at[idx, f"{prefix}_league_losses"] = pos_data["losses"]

            # Update stats AFTER recording (no leakage)
            hs = row.get("home_score")
            as_ = row.get("away_score")
            if pd.notna(hs) and pd.notna(as_):
                hs, as_ = int(hs), int(as_)
                team_stats[home]["gf"] += hs
                team_stats[away]["gf"] += as_
                team_stats[home]["gd"] += hs - as_
                team_stats[away]["gd"] += as_ - hs
                team_stats[home]["played"] += 1
                team_stats[away]["played"] += 1

                if hs > as_:
                    team_stats[home]["points"] += 3
                    team_stats[home]["wins"] += 1
                    team_stats[away]["losses"] += 1
                elif hs < as_:
                    team_stats[away]["points"] += 3
                    team_stats[away]["wins"] += 1
                    team_stats[home]["losses"] += 1
                else:
                    team_stats[home]["points"] += 1
                    team_stats[away]["points"] += 1
                    team_stats[home]["draws"] += 1
                    team_stats[away]["draws"] += 1

    # Derive motivation indicators
    for prefix in ("home", "away"):
        pos = pd.to_numeric(df[f"{prefix}_league_pos"], errors="coerce")
        df[f"{prefix}_in_relegation_zone"] = (pos >= 18).astype(float).where(pos.notna())
        df[f"{prefix}_in_cl_zone"] = (pos <= 4).astype(float).where(pos.notna())
        df[f"{prefix}_in_el_zone"] = pos.isin([5, 6]).astype(float).where(pos.notna())
        df[f"{prefix}_in_title_race"] = (pos <= 2).astype(float).where(pos.notna())

    # Position difference (negative = home team is higher in the table)
    h_pos = pd.to_numeric(df["home_league_pos"], errors="coerce")
    a_pos = pd.to_numeric(df["away_league_pos"], errors="coerce")
    df["league_position_diff"] = h_pos - a_pos

    df.drop(columns=["_date"], inplace=True)
    return df


def _compute_standings(team_stats: dict[str, dict]) -> dict[str, dict]:
    """Compute current standings from cumulative team stats.

    Returns dict of team -> {position, points, gd, gf, ...}
    """
    if not team_stats:
        return {}

    # Sort by points (desc), goal difference (desc), goals for (desc)
    teams = sorted(
        team_stats.keys(),
        key=lambda t: (
            team_stats[t]["points"],
            team_stats[t]["gd"],
            team_stats[t]["gf"],
        ),
        reverse=True,
    )

    result = {}
    for i, team in enumerate(teams, start=1):
        result[team] = {
            "position": i,
            **team_stats[team],
        }
    return result
