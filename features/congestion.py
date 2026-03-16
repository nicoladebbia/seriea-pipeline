"""Fixture congestion features.

Flags midweek matches and periods of heavy scheduling. These features
are computed from match dates and rest_days (already computed by
features.rest_days). This module adds higher-level congestion indicators.
"""

from __future__ import annotations

import pandas as pd


def add_congestion_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Add fixture congestion features to the match-level table.

    Uses the team-level rest_days data that should already be pivoted
    to match level (home_rest_days, away_rest_days from features.rest_days).

    Columns added (for both home_ and away_ prefixes):
      {prefix}_is_midweek           - 1 if match is Tue/Wed/Thu
      {prefix}_short_rest           - 1 if rest_days < 4
      {prefix}_very_short_rest      - 1 if rest_days <= 2
      {prefix}_congestion_3         - matches played in last 15 days (3 match window)
      {prefix}_congestion_5         - matches played in last 30 days (5 match window)
      {prefix}_rest_advantage       - home_rest_days - away_rest_days
    """
    df = matches.copy()
    df["_date"] = pd.to_datetime(df["match_date"], errors="coerce")

    # Day of week: 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun
    dow = df["_date"].dt.dayofweek

    # Midweek indicator (both teams play on the same day)
    is_midweek = dow.isin([1, 2, 3]).astype(float)
    df["home_is_midweek"] = is_midweek
    df["away_is_midweek"] = is_midweek

    # Short rest indicators (require rest_days from pivot)
    for prefix in ("home", "away"):
        rest_col = f"{prefix}_rest_days"
        if rest_col in df.columns:
            rest = pd.to_numeric(df[rest_col], errors="coerce")
            df[f"{prefix}_short_rest"] = (rest < 4).astype(float).where(rest.notna())
            df[f"{prefix}_very_short_rest"] = (rest <= 2).astype(float).where(rest.notna())
        else:
            df[f"{prefix}_short_rest"] = None
            df[f"{prefix}_very_short_rest"] = None

    # Congestion: matches played in rolling windows
    # We need to build a team-level timeline for this
    df = df.sort_values("_date").reset_index(drop=True)

    # Build team match dates
    team_dates: dict[str, list[pd.Timestamp]] = {}
    for _, row in df.iterrows():
        d = row["_date"]
        if pd.isna(d):
            continue
        for team_col in ["home_team", "away_team"]:
            team = row[team_col]
            if team not in team_dates:
                team_dates[team] = []
            team_dates[team].append(d)

    # For each match, count how many matches each team played in last N days
    home_cong3 = []
    home_cong5 = []
    away_cong3 = []
    away_cong5 = []

    for _, row in df.iterrows():
        d = row["_date"]
        if pd.isna(d):
            home_cong3.append(None)
            home_cong5.append(None)
            away_cong3.append(None)
            away_cong5.append(None)
            continue

        for team_col, c3_list, c5_list in [
            ("home_team", home_cong3, home_cong5),
            ("away_team", away_cong3, away_cong5),
        ]:
            team = row[team_col]
            dates = team_dates.get(team, [])
            # Count matches in last 15 days (excluding current match)
            c3 = sum(1 for dd in dates if dd < d and (d - dd).days <= 15)
            c5 = sum(1 for dd in dates if dd < d and (d - dd).days <= 30)
            c3_list.append(c3)
            c5_list.append(c5)

    df["home_congestion_3"] = home_cong3
    df["away_congestion_3"] = away_cong3
    df["home_congestion_5"] = home_cong5
    df["away_congestion_5"] = away_cong5

    # Rest advantage
    for col in ["home_rest_days", "away_rest_days"]:
        if col not in df.columns:
            break
    else:
        h_rest = pd.to_numeric(df["home_rest_days"], errors="coerce")
        a_rest = pd.to_numeric(df["away_rest_days"], errors="coerce")
        df["rest_advantage"] = h_rest - a_rest

    df.drop(columns=["_date"], inplace=True)
    return df
