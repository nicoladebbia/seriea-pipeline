"""Goalkeeper quality features.

Captures goalkeeper performance beyond simple save counts:
  - Post-shot xG performance (PSxG - Goals Allowed)
  - Save percentage
  - Cross stopping rate
  - Sweeper actions (defensive actions outside box)
  - Launch accuracy (long ball distribution)
"""

from __future__ import annotations

import logging

import pandas as pd

from storage.paths import parsed_path

log = logging.getLogger(__name__)

# GK stats to extract (from FBref keeper_stats table)
GK_STATS = [
    "gk_shots_on_target_against",
    "gk_goals_against",
    "gk_saves",
    "gk_save_pct",
    "gk_psxg",  # post-shot xG
    "gk_passes_completed_launched",
    "gk_passes_launched",
    "gk_passes_pct_launched",
    "gk_passes",
    "gk_passes_throws",
    "gk_pct_passes_launched",
    "gk_passes_length_avg",
    "gk_goal_kicks",
    "gk_pct_goal_kicks_launched",
    "gk_goal_kick_length_avg",
    "gk_crosses",
    "gk_crosses_stopped",
    "gk_crosses_stopped_pct",
    "gk_def_actions_outside_pen_area",
    "gk_avg_distance_def_actions",
]


def add_gk_quality_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Add goalkeeper quality features to the match-level table.

    Uses a rolling 5-match average of GK stats (shifted).

    Columns added: gk_roll5_{stat} for home and away.
    """
    gk_path = parsed_path("goalkeeper_stats")
    if not gk_path.exists():
        log.warning("goalkeeper_stats.parquet missing; skipping GK quality features")
        return matches

    gk_stats = pd.read_parquet(gk_path)
    df = matches.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.sort_values("match_date").reset_index(drop=True)

    # Get available GK stat columns
    available = [c for c in GK_STATS if c in gk_stats.columns]
    if not available:
        log.warning("No GK stat columns found; skipping GK quality features")
        return df

    for col in available:
        gk_stats[col] = pd.to_numeric(gk_stats[col], errors="coerce")

    # Aggregate per match per team (sum/mean across GKs - usually just 1)
    if "match_id" not in gk_stats.columns or "team" not in gk_stats.columns:
        return df

    gk_agg = gk_stats.groupby(["match_id", "team"])[available].mean().reset_index()

    # Add derived features
    if "gk_psxg" in gk_agg.columns and "gk_goals_against" in gk_agg.columns:
        gk_agg["gk_psxg_diff"] = gk_agg["gk_psxg"] - gk_agg["gk_goals_against"]
        available.append("gk_psxg_diff")

    # Merge match dates and compute rolling
    gk_agg = gk_agg.merge(
        df[["match_id", "match_date"]].drop_duplicates(),
        on="match_id",
        how="left",
    )
    gk_agg = gk_agg.sort_values(["team", "match_date"]).reset_index(drop=True)

    roll_cols = []
    for stat in available:
        col_name = f"gk_roll5_{stat}"
        gk_agg[col_name] = (
            gk_agg.groupby("team")[stat]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        )
        roll_cols.append(col_name)

    # Merge to match level
    for prefix, team_col in [("home", "home_team"), ("away", "away_team")]:
        team_feats = gk_agg[["match_id", "team"] + roll_cols].copy()
        rename_map = {c: f"{prefix}_{c}" for c in roll_cols}
        team_feats = team_feats.rename(columns=rename_map)
        df = df.merge(
            team_feats.rename(columns={"team": team_col}),
            on=["match_id", team_col],
            how="left",
        )

    return df
