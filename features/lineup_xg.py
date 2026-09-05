"""Historical lineup xG reconstruction features.

Reconstructs per-match lineup quality by summing starter xG/90 rates.
Two data sources:
  - Sofascore (2022+): per-match starter xG, direct aggregation
  - Understat (2014-2024): season-level xG/90 per player, indirect

Outputs per match (home/away):
  - lineup_xg_sum: total xG/90 of starters
  - lineup_xa_sum: total xA/90 of starters
  - lineup_xg_vs_team: lineup xG vs team-level xG (squad utilization)
  - lineup_depth_ratio: starters xG / total squad xG (concentration)
  - lineup_rotation: how many starters changed vs previous match
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config.team_names import normalize_team

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


def _load_sofascore_starters(league: str | None = None) -> pd.DataFrame | None:
    """Load Sofascore per-match starter data with xG.

    If `league` is given, uses the league-specific file when available (EPL
    is stored in `player_match_stats_premier_league.parquet`, which is a
    separate scraper output). Without `league`, falls back to the default
    (Serie A) file for backward compatibility.
    """
    base = DATA_DIR / "external" / "sofascore"
    if league == "premier_league":
        path = base / "player_match_stats_premier_league.parquet"
    else:
        path = base / "player_match_stats.parquet"
    if not path.exists():
        log.info("No Sofascore player stats at %s", path)
        return None

    df = pd.read_parquet(path)
    # Filter to starters with xG data
    starters = df[df["is_starter"] == True].copy()
    log.info("Loaded %d Sofascore starter records for league=%s (seasons: %s)",
             len(starters), league or "default",
             sorted(starters["season"].unique()))
    return starters


def _load_understat_player_xg() -> pd.DataFrame | None:
    """Load Understat season-level player xG rates."""
    path = DATA_DIR / "parsed" / "understat_players.parquet"
    if not path.exists():
        log.info("No Understat player data at %s", path)
        return None

    df = pd.read_parquet(path)
    # Compute xG per 90 minutes
    df = df[df["minutes"] > 0].copy()
    df["xg_per_90"] = df["xg"] / df["minutes"] * 90
    df["xg_per_90"] = df["xg_per_90"].clip(upper=2.0)  # Cap outliers
    log.info("Loaded %d Understat player-season records", len(df))
    return df


def _compute_sofascore_lineup_features(
    feature_df: pd.DataFrame,
    starters: pd.DataFrame,
) -> pd.DataFrame:
    """Compute lineup features from Sofascore per-match data.

    Joins on (date, team) instead of match_id since feature_df uses
    date_home_away string IDs while Sofascore uses numeric IDs.
    """
    df = feature_df.copy()

    # Normalize date column for joining
    starters = starters.copy()
    starters["date"] = pd.to_datetime(starters["date"]).dt.date

    # Aggregate per date-team: sum xG/xA of starters
    agg = starters.groupby(["date", "team"]).agg(
        lineup_xg_sum=("xg", "sum"),
        lineup_xa_sum=("xa", "sum"),
        lineup_rating_mean=("rating", "mean"),
        n_starters=("player_name", "count"),
    ).reset_index()

    # Compute lineup rotation (starters who weren't starters last match)
    starters_sorted = starters.sort_values(["team", "date", "player_name"])
    rotation_data = []
    for team, grp in starters_sorted.groupby("team"):
        match_dates = sorted(grp["date"].unique())
        prev_starters = set()
        for date in match_dates:
            current = set(grp[grp["date"] == date]["player_name"])
            if prev_starters:
                changed = len(current - prev_starters)
                rotation_data.append({
                    "date": date,
                    "team": team,
                    "lineup_rotation": changed,
                })
            prev_starters = current

    if rotation_data:
        rot_df = pd.DataFrame(rotation_data)
        agg = agg.merge(rot_df, on=["date", "team"], how="left")
    else:
        agg["lineup_rotation"] = np.nan

    # Create date column for joining
    if "match_date" in df.columns:
        df["_join_date"] = pd.to_datetime(df["match_date"]).dt.date
    elif "date" in df.columns:
        df["_join_date"] = pd.to_datetime(df["date"]).dt.date
    else:
        log.warning("No date column in feature_df — skipping Sofascore lineup xG")
        return df

    # Join to feature_df (home and away)
    for side, team_col in [("home", "home_team"), ("away", "away_team")]:
        if team_col not in df.columns:
            continue
        merged = df[["_join_date", team_col]].reset_index().merge(
            agg, left_on=["_join_date", team_col], right_on=["date", "team"],
            how="left",
        ).set_index("index")
        for col in ["lineup_xg_sum", "lineup_xa_sum", "lineup_rating_mean",
                     "lineup_rotation"]:
            if col in merged.columns:
                df[f"{side}_{col}"] = merged[col].values

    df.drop(columns=["_join_date"], inplace=True, errors="ignore")

    # Add diffs
    for col in ["lineup_xg_sum", "lineup_xa_sum", "lineup_rating_mean"]:
        h_col = f"home_{col}"
        a_col = f"away_{col}"
        if h_col in df.columns and a_col in df.columns:
            df[f"{col}_diff"] = df[h_col] - df[a_col]

    return df


def _compute_understat_lineup_proxy(
    feature_df: pd.DataFrame,
    us_players: pd.DataFrame,
) -> pd.DataFrame:
    """Compute proxy lineup xG from Understat season-level data.

    Since we don't have per-match lineups for Understat, we use the team's
    top-N contributors' xG/90 as a proxy for lineup quality.
    """
    df = feature_df.copy()

    # For each team-season, compute top-11 xG/90 sum (proxy for best lineup)
    top_n = 11
    team_season_xg = []
    for (team, season), grp in us_players.groupby(["team", "season"]):
        # Sort by xG/90, take top 11
        top = grp.nlargest(top_n, "xg_per_90")
        team_season_xg.append({
            "team": team,
            "season": season,
            "us_top11_xg90_sum": top["xg_per_90"].sum(),
            "us_top11_xg90_mean": top["xg_per_90"].mean(),
            "us_squad_depth": len(grp[grp["minutes"] > 270]),  # Players with 3+ full matches
            "us_xg_concentration": top["xg_per_90"].iloc[0] / top["xg_per_90"].sum()
            if top["xg_per_90"].sum() > 0 else np.nan,
        })

    if not team_season_xg:
        return df

    us_df = pd.DataFrame(team_season_xg)

    # The proxy describes the LAST COMPLETE season, so each (team, season)
    # aggregate is applied to the FOLLOWING season's rows. Joining the same
    # season was wrong on both ends (found 2026-09-05): training rows read
    # the full season's totals at matchweek 2 (within-season lookahead, the
    # P1b family), and serving read season-cumulative stats that barely
    # exist yet — us_squad_depth counts players with >270 minutes, so every
    # 2026-27 row served 0.0 against a training band of [17, 31] while
    # ranking in the ML leg's top-5 SHAP drivers. A team without a prior
    # Serie A season (promoted, or back after a gap) gets NaN — honest, and
    # filled with training medians downstream.
    def _next_season(s: str) -> str:
        a, b = s.split("-")
        return f"{int(a) + 1}-{int(b) + 1}"

    us_df["season"] = us_df["season"].map(_next_season)

    # Normalize Understat team names to pipeline canonical form
    us_df["team_norm"] = us_df["team"].apply(normalize_team)

    for side, team_col in [("home", "home_team"), ("away", "away_team")]:
        if team_col not in df.columns or "season" not in df.columns:
            continue
        merged = df[[team_col, "season"]].merge(
            us_df, left_on=[team_col, "season"], right_on=["team_norm", "season"],
            how="left",
        )
        for col in ["us_top11_xg90_sum", "us_top11_xg90_mean", "us_squad_depth",
                     "us_xg_concentration"]:
            if col in merged.columns:
                df[f"{side}_{col}"] = merged[col].values

    # Diffs
    for col in ["us_top11_xg90_sum", "us_squad_depth"]:
        h, a = f"home_{col}", f"away_{col}"
        if h in df.columns and a in df.columns:
            df[f"{col}_diff"] = df[h] - df[a]

    return df


def add_lineup_xg_features(feature_df: pd.DataFrame,
                            league: str | None = None) -> pd.DataFrame:
    """Add all lineup xG features to the match-level feature DataFrame.

    `league` routes Sofascore starter data to the correct file — SA is
    `player_match_stats.parquet`, EPL is `player_match_stats_premier_league.parquet`.
    """
    df = feature_df.copy()
    cols_before = len(df.columns)

    # Source 1: Sofascore per-match (SA: 2022+, EPL: 2017+)
    starters = _load_sofascore_starters(league=league)
    if starters is not None:
        df = _compute_sofascore_lineup_features(df, starters)

    # Source 2: Understat season-level proxy (2014-2024)
    us_players = _load_understat_player_xg()
    if us_players is not None:
        df = _compute_understat_lineup_proxy(df, us_players)

    new_cols = len(df.columns) - cols_before
    log.info("Lineup xG features: added %d columns", new_cols)
    return df
