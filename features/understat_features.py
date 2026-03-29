"""Understat-based xG and player features.

Integrates data from Understat.com:
  - Player-level xG, npxG, xA per season
  - Team-level aggregated xG metrics
  - xG chain and buildup metrics

Data source: data/parsed/understat_players.parquet
Coverage: ITA-Serie A 2014-2025, ENG-Premier League 2017-2025 (multi-league)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config.team_names import normalize_team

log = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
UNDERSTAT_PLAYERS_PATH = PROJECT_ROOT / "data" / "parsed" / "understat_players.parquet"


def _normalize_team_name(name: str) -> str:
    """Normalize team name to match our pipeline.

    Understat uses varied casing; normalize_team handles all known variants
    and returns the canonical title-case name.
    """
    if pd.isna(name):
        return ""
    return normalize_team(name)


def _load_understat_players() -> pd.DataFrame | None:
    """Load understat player data (supports multiple leagues)."""
    if not UNDERSTAT_PLAYERS_PATH.exists():
        log.warning(f"Understat player data not found at {UNDERSTAT_PLAYERS_PATH}")
        return None

    df = pd.read_parquet(UNDERSTAT_PLAYERS_PATH)
    # Normalize team names
    df["team_normalized"] = df["team"].apply(_normalize_team_name)

    # Log league breakdown
    if "league" in df.columns:
        league_counts = df["league"].value_counts().to_dict()
        log.info(f"Loaded Understat players: {len(df)} rows across {len(league_counts)} league(s): {league_counts}")
    else:
        log.info(f"Loaded Understat players: {len(df)} rows (no league column)")

    return df


def _build_team_season_stats(players: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player stats to team level per season.

    Returns DataFrame with columns:
        - team, season
        - team_xg, team_npxg, team_xa
        - team_shots, team_key_passes
        - team_xg_chain, team_xg_buildup
        - avg_xg_per_90, avg_xa_per_90
        - top3_xg_share (concentration metric)
    """
    # Ensure numeric columns
    numeric_cols = ["xg", "np_xg", "xa", "shots", "key_passes", "xg_chain", "xg_buildup", "minutes", "goals", "assists"]
    for col in numeric_cols:
        if col in players.columns:
            players[col] = pd.to_numeric(players[col], errors="coerce").fillna(0)

    # Aggregate by team and season
    team_agg = players.groupby(["team_normalized", "season"]).agg({
        "xg": "sum",
        "np_xg": "sum",
        "xa": "sum",
        "shots": "sum",
        "key_passes": "sum",
        "xg_chain": "sum",
        "xg_buildup": "sum",
        "minutes": "sum",
        "goals": "sum",
        "assists": "sum",
        "player": "count",  # Number of players
    }).reset_index()

    team_agg.columns = [
        "team", "season",
        "team_xg", "team_npxg", "team_xa",
        "team_shots", "team_key_passes",
        "team_xg_chain", "team_xg_buildup",
        "team_minutes", "team_goals", "team_assists",
        "player_count"
    ]

    # Per 90 metrics (based on total team minutes / 90 / ~11 players)
    # A season is ~38 games, so ~3420 minutes per player if playing all
    team_agg["team_xg_per_90"] = team_agg["team_xg"] / (team_agg["team_minutes"] / 90).clip(lower=1)
    team_agg["team_xa_per_90"] = team_agg["team_xa"] / (team_agg["team_minutes"] / 90).clip(lower=1)

    # Efficiency metrics
    team_agg["team_xg_per_shot"] = team_agg["team_xg"] / team_agg["team_shots"].clip(lower=1)
    team_agg["team_goals_minus_xg"] = team_agg["team_goals"] - team_agg["team_xg"]

    # Compute xG concentration (top 3 players' share of team xG)
    top3_shares = []
    for (team, season), group in players.groupby(["team_normalized", "season"]):
        total_xg = group["xg"].sum()
        if total_xg > 0:
            top3_xg = group.nlargest(3, "xg")["xg"].sum()
            top3_shares.append({
                "team": team,
                "season": season,
                "top3_xg_share": top3_xg / total_xg
            })
        else:
            top3_shares.append({
                "team": team,
                "season": season,
                "top3_xg_share": 0.0
            })

    top3_df = pd.DataFrame(top3_shares)
    team_agg = team_agg.merge(top3_df, on=["team", "season"], how="left")

    return team_agg


def add_understat_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Add Understat xG features to match DataFrame.

    Uses team-level aggregated xG metrics from player data.

    Columns added:
      - home_us_xg, away_us_xg: Team total xG for the season
      - home_us_xa, away_us_xa: Team total xA for the season
      - home_us_xg_per_shot, away_us_xg_per_shot: Shot quality
      - home_us_top3_xg_share, away_us_top3_xg_share: xG concentration
      - us_coverage: Whether we have Understat data for both teams
    """
    players = _load_understat_players()
    if players is None:
        log.warning("Skipping Understat features - no data available")
        return matches

    df = matches.copy()

    # Build team-season aggregates
    team_stats = _build_team_season_stats(players)
    log.info(f"Built Understat team stats: {len(team_stats)} team-seasons")

    # Normalize team names in match data
    df["home_team_norm"] = df["home_team"].apply(_normalize_team_name)
    df["away_team_norm"] = df["away_team"].apply(_normalize_team_name)

    # Extract season from match data
    # Match data should have 'season' column in format "2023-2024"
    if "season" not in df.columns:
        # Try to derive from match_date
        if "match_date" in df.columns:
            df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
            df["match_year"] = df["match_date"].dt.year
            df["match_month"] = df["match_date"].dt.month
            # Season is Aug-May, so year is season start if month >= 8
            df["season_start"] = df.apply(
                lambda r: r["match_year"] if r["match_month"] >= 8 else r["match_year"] - 1,
                axis=1
            )
            df["season"] = df["season_start"].astype(str) + "-" + (df["season_start"] + 1).astype(str)

    # Merge home team stats
    home_stats = team_stats.copy()
    home_stats.columns = ["home_team_norm", "season"] + [f"home_us_{c}" for c in team_stats.columns[2:]]
    df = df.merge(home_stats, on=["home_team_norm", "season"], how="left")

    # Merge away team stats
    away_stats = team_stats.copy()
    away_stats.columns = ["away_team_norm", "season"] + [f"away_us_{c}" for c in team_stats.columns[2:]]
    df = df.merge(away_stats, on=["away_team_norm", "season"], how="left")

    # Compute derived features
    # xG differential
    df["us_xg_diff"] = df["home_us_team_xg"].fillna(0) - df["away_us_team_xg"].fillna(0)
    df["us_xa_diff"] = df["home_us_team_xa"].fillna(0) - df["away_us_team_xa"].fillna(0)

    # Coverage indicator
    df["us_coverage"] = (
        df["home_us_team_xg"].notna() &
        df["away_us_team_xg"].notna()
    ).astype(int)

    # Clean up temp columns
    df = df.drop(columns=["home_team_norm", "away_team_norm", "match_year", "match_month", "season_start"], errors="ignore")

    n_covered = df["us_coverage"].sum()
    log.info(f"Understat features: {n_covered}/{len(df)} matches covered ({100*n_covered/len(df):.1f}%)")

    # Log added columns
    us_cols = [c for c in df.columns if c.startswith("home_us_") or c.startswith("away_us_") or c.startswith("us_")]
    log.info(f"Added {len(us_cols)} Understat features")

    return df


def add_understat_rolling_features(matches: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add rolling xG features based on historical performance.

    This computes rolling averages over the last N seasons for each team.

    Columns added:
      - home_us_xg_trend, away_us_xg_trend: Trend in team xG over seasons
      - home_us_consistency, away_us_consistency: Std dev of xG over seasons
    """
    players = _load_understat_players()
    if players is None:
        return matches

    df = matches.copy()
    team_stats = _build_team_season_stats(players)

    # Sort by season for rolling calculations
    team_stats["season_num"] = team_stats["season"].apply(lambda x: int(x.split("-")[0]))
    team_stats = team_stats.sort_values(["team", "season_num"])

    # Compute rolling stats per team
    rolling_stats = []
    for team, group in team_stats.groupby("team"):
        group = group.sort_values("season_num")
        for i, row in group.iterrows():
            current_season = row["season"]
            prior = group[group["season_num"] < row["season_num"]].tail(window)

            if len(prior) >= 2:
                rolling_stats.append({
                    "team": team,
                    "season": current_season,
                    "xg_rolling_avg": prior["team_xg"].mean(),
                    "xg_rolling_std": prior["team_xg"].std(),
                    "xg_trend": row["team_xg"] - prior["team_xg"].mean(),
                })
            else:
                rolling_stats.append({
                    "team": team,
                    "season": current_season,
                    "xg_rolling_avg": None,
                    "xg_rolling_std": None,
                    "xg_trend": None,
                })

    rolling_df = pd.DataFrame(rolling_stats)

    # Normalize team names in match data
    df["home_team_norm"] = df["home_team"].apply(_normalize_team_name)
    df["away_team_norm"] = df["away_team"].apply(_normalize_team_name)

    # Merge home stats
    home_rolling = rolling_df.copy()
    home_rolling.columns = ["home_team_norm", "season", "home_us_xg_rolling_avg", "home_us_xg_rolling_std", "home_us_xg_trend"]
    df = df.merge(home_rolling, on=["home_team_norm", "season"], how="left")

    # Merge away stats
    away_rolling = rolling_df.copy()
    away_rolling.columns = ["away_team_norm", "season", "away_us_xg_rolling_avg", "away_us_xg_rolling_std", "away_us_xg_trend"]
    df = df.merge(away_rolling, on=["away_team_norm", "season"], how="left")

    df = df.drop(columns=["home_team_norm", "away_team_norm"], errors="ignore")

    return df
