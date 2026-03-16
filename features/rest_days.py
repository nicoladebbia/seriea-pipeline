"""Rest days between consecutive matches and congestion flags."""

from __future__ import annotations

import pandas as pd

# Cap rest days at 21. Anything longer (summer break, COVID gap) is not
# a meaningful "rest" signal — every team has a similar break.
MAX_REST_DAYS = 21


def add_rest_days(team_log: pd.DataFrame) -> pd.DataFrame:
    """Add rest_days feature to the team match log.

    Columns added:
      rest_days       - days since this team's previous match (within season)
      is_congested    - True if rest_days <= 3
    """
    df = team_log.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.sort_values(["team", "match_date"]).reset_index(drop=True)

    # Compute rest days within season only (cross-season gaps are meaningless)
    df["rest_days"] = (
        df.groupby(["team", "season"])["match_date"]
        .diff()
        .dt.days
    )

    # Cap extreme values (e.g. COVID-19 lockdown gap in 2019-2020)
    df["rest_days"] = df["rest_days"].clip(upper=MAX_REST_DAYS)

    df["is_congested"] = df["rest_days"].le(3)

    return df


def pivot_rest_to_match(matches: pd.DataFrame, team_log: pd.DataFrame) -> pd.DataFrame:
    """Add home_rest_days, away_rest_days, rest_advantage to the match table."""
    df = matches.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

    # Build lookup: (match_id, team) -> rest_days
    rest_lookup = team_log.set_index(["match_id", "team"])["rest_days"].to_dict()
    congested_lookup = team_log.set_index(["match_id", "team"])["is_congested"].to_dict()

    df["home_rest_days"] = df.apply(
        lambda r: rest_lookup.get((r["match_id"], r["home_team"])), axis=1
    )
    df["away_rest_days"] = df.apply(
        lambda r: rest_lookup.get((r["match_id"], r["away_team"])), axis=1
    )
    df["rest_advantage"] = df["home_rest_days"] - df["away_rest_days"]
    df["home_is_congested"] = df.apply(
        lambda r: congested_lookup.get((r["match_id"], r["home_team"]), False), axis=1
    )
    df["away_is_congested"] = df.apply(
        lambda r: congested_lookup.get((r["match_id"], r["away_team"]), False), axis=1
    )

    return df
