"""Build team-centric match log by unpivoting the matches table.

Transforms matches.parquet (one row per match) into a team log
(one row per team per match) with goals_scored, goals_conceded, etc.
"""

from __future__ import annotations

import pandas as pd

from storage.paths import parsed_path


# Team-stat columns that use home_*/away_* prefix in matches.parquet
STAT_COLUMNS = [
    "possession",
    "passing_accuracy",
    "shots_on_target",
    "saves",
    "yellow_cards",
    "red_cards",
    "fouls",
    "corners",
    "crosses",
    "touches",
    "tackles",
    "interceptions",
    "aerials_won",
    "clearances",
    "offsides",
    "goal_kicks",
    "throw_ins",
    "long_balls",
]


def build_team_match_log(matches: pd.DataFrame | None = None) -> pd.DataFrame:
    """Unpivot matches into team-centric rows.

    Each match produces 2 rows: one from the home team's perspective,
    one from the away team's perspective.

    Returns DataFrame sorted by (team, match_date) with columns:
      match_id, match_date, season, matchweek, team, opponent,
      is_home, goals_scored, goals_conceded, result, points,
      xg_for, xg_against, + all available team stats.
    """
    if matches is None:
        path = parsed_path("matches")
        if not path.exists():
            raise FileNotFoundError(f"matches.parquet not found at {path}")
        matches = pd.read_parquet(path)

    home_rows = _perspective(matches, is_home=True)
    away_rows = _perspective(matches, is_home=False)

    log = pd.concat([home_rows, away_rows], ignore_index=True)

    # Derived columns
    log["result"] = log.apply(_result, axis=1)
    log["points"] = log["result"].map({"W": 3, "D": 1, "L": 0})
    log["clean_sheet"] = (log["goals_conceded"] == 0).astype(int)

    # Sort chronologically per team
    log["match_date"] = pd.to_datetime(log["match_date"], errors="coerce")
    log = log.sort_values(["team", "match_date"]).reset_index(drop=True)

    return log


def _perspective(df: pd.DataFrame, is_home: bool) -> pd.DataFrame:
    """Create rows from one perspective (home or away)."""
    prefix = "home" if is_home else "away"
    opp_prefix = "away" if is_home else "home"

    rows = pd.DataFrame()
    rows["match_id"] = df["match_id"]
    rows["match_date"] = df["match_date"]
    rows["season"] = df["season"]
    rows["matchweek"] = df.get("matchweek")
    rows["team"] = df[f"{prefix}_team"]
    rows["opponent"] = df[f"{opp_prefix}_team"]
    rows["is_home"] = is_home
    rows["goals_scored"] = pd.to_numeric(df.get(f"{prefix}_score"), errors="coerce")
    rows["goals_conceded"] = pd.to_numeric(df.get(f"{opp_prefix}_score"), errors="coerce")

    # xG — column name is home_xg / away_xg in matches.parquet
    xg_for_col = f"{prefix}_xg"
    xg_against_col = f"{opp_prefix}_xg"
    rows["xg_for"] = pd.to_numeric(df[xg_for_col], errors="coerce") if xg_for_col in df.columns else float("nan")
    rows["xg_against"] = pd.to_numeric(df[xg_against_col], errors="coerce") if xg_against_col in df.columns else float("nan")

    # Team stats
    for stat in STAT_COLUMNS:
        col = f"{prefix}_{stat}"
        if col in df.columns:
            rows[stat] = pd.to_numeric(df[col], errors="coerce")

    # Use raw shots on target COUNT (not percentage) for consistency across sources
    # FBref: home_shots_on_target is a percentage; football-data.co.uk: computed %
    # Both store raw count in home_shots_on_target_count
    sot_count_col = f"{prefix}_shots_on_target_count"
    sot_total_col = f"{prefix}_shots_on_target_total"
    if sot_count_col in df.columns:
        rows["shots_on_target"] = pd.to_numeric(df[sot_count_col], errors="coerce")
    if sot_total_col in df.columns:
        rows["total_shots"] = pd.to_numeric(df[sot_total_col], errors="coerce")

    # Use raw saves COUNT instead of percentage
    saves_count_col = f"{prefix}_saves_count"
    if saves_count_col in df.columns:
        rows["saves"] = pd.to_numeric(df[saves_count_col], errors="coerce")

    # Formation
    form_col = f"{prefix}_formation"
    if form_col in df.columns:
        rows["formation"] = df[form_col]

    return rows


def _result(row: pd.Series) -> str:
    gs = row.get("goals_scored")
    gc = row.get("goals_conceded")
    if pd.isna(gs) or pd.isna(gc):
        return "U"  # unknown
    if gs > gc:
        return "W"
    if gs < gc:
        return "L"
    return "D"
