"""FBref per-match player statistics → team-level rolling features.

Loads data/parsed/player_stats.parquet (98,140 per-match player records across
9 seasons 2017-2026) and produces ~30 rolling 5-match team features per side
(home/away) plus differentials.

Architecture mirrors sofascore_features.py:
  1. Aggregate per-player stats → per-team-per-match
  2. Compute rolling 5-match averages with shift(1) to prevent leakage
  3. merge_asof for temporal matching to feature DataFrame

Features created (fb_ prefix):
  - Goals, assists, shots, shots_on_target (basic attacking)
  - Cards (yellow, red), fouls committed/drawn
  - Squad depth (unique players used in rolling window)
  - Average age, average minutes per player
  - Pass accuracy, progressive passes (when available)
  - Tackles, interceptions, blocks, clearances (defensive)
  - xG, xA, SCA, GCA (when available — 13% of data)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config.team_names import normalize_team

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
PLAYER_STATS_PATH = PROJECT_ROOT / "data" / "parsed" / "player_stats.parquet"

ROLLING_WINDOW = 5


def _normalize_team(name: str) -> str:
    if pd.isna(name):
        return ""
    return normalize_team(name)


def _load_player_stats() -> pd.DataFrame | None:
    if not PLAYER_STATS_PATH.exists():
        log.warning("FBref player_stats.parquet not found at %s", PLAYER_STATS_PATH)
        return None
    df = pd.read_parquet(PLAYER_STATS_PATH)
    log.info("Loaded FBref player stats: %d rows, %d columns", len(df), len(df.columns))
    return df


def _build_team_match_stats(player_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-player stats to per-team-per-match."""
    df = player_df.copy()

    # Convert all stat columns to numeric
    sum_cols = [
        "goals", "assists", "shots", "shots_on_target",
        "cards_yellow", "cards_red", "fouls", "fouled",
        "interceptions", "tackles_won", "blocks",
        "pens_made", "pens_att", "crosses", "own_goals",
    ]
    # Advanced columns (only ~13% coverage but valuable where available)
    adv_sum_cols = [
        "xg", "npxg", "xg_assist", "sca", "gca",
        "passes_completed", "passes", "progressive_passes",
        "tackles", "touches", "carries", "progressive_carries",
        "take_ons", "take_ons_won",
    ]
    # Defense-specific advanced columns
    def_sum_cols = [
        "defense_tackles", "defense_tackles_won",
        "defense_interceptions", "defense_clearances",
        "defense_blocks", "defense_blocked_shots", "defense_errors",
    ]
    # Misc advanced
    misc_sum_cols = [
        "misc_ball_recoveries", "misc_aerials_won", "misc_aerials_lost",
    ]

    all_sum = sum_cols + adv_sum_cols + def_sum_cols + misc_sum_cols
    for c in all_sum:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Minutes and age
    if "minutes" in df.columns:
        df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")
    if "age" in df.columns:
        # age may be "28-123" format (years-days)
        df["age_years"] = df["age"].astype(str).str.split("-").str[0]
        df["age_years"] = pd.to_numeric(df["age_years"], errors="coerce")

    # Normalize team names
    df["team_norm"] = df["team"].apply(_normalize_team)

    # Match date
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

    # Group by match + team
    agg_map = {}
    for c in all_sum:
        if c in df.columns:
            agg_map[c] = "sum"
    if "minutes" in df.columns:
        agg_map["minutes"] = "sum"
    if "age_years" in df.columns:
        agg_map["age_years"] = "mean"

    group_cols = ["match_id", "team_norm"]
    team_match = df.groupby(group_cols).agg(agg_map).reset_index()

    # Squad depth: unique players per match
    depth = df.groupby(group_cols)["player"].nunique().reset_index(name="squad_players")
    team_match = team_match.merge(depth, on=group_cols, how="left")

    # Metadata from first row per group
    meta_cols = ["match_date", "season", "is_home"]
    meta = df.groupby(group_cols)[meta_cols].first().reset_index()
    team_match = team_match.merge(meta, on=group_cols, how="left")

    # Derived per-match
    if "shots" in team_match.columns:
        team_match["shot_accuracy"] = (
            team_match["shots_on_target"] / team_match["shots"].clip(lower=1)
        )
    if "passes" in team_match.columns and "passes_completed" in team_match.columns:
        team_match["pass_accuracy"] = (
            team_match["passes_completed"] / team_match["passes"].clip(lower=1)
        )
    if "xg" in team_match.columns and "shots" in team_match.columns:
        team_match["xg_per_shot"] = (
            team_match["xg"] / team_match["shots"].clip(lower=1)
        )
    if "misc_aerials_won" in team_match.columns and "misc_aerials_lost" in team_match.columns:
        total_aerials = (team_match["misc_aerials_won"] + team_match["misc_aerials_lost"]).clip(lower=1)
        team_match["aerial_win_rate"] = team_match["misc_aerials_won"] / total_aerials

    # Average minutes per player (proxy for rotation)
    if "minutes" in team_match.columns and "squad_players" in team_match.columns:
        team_match["avg_minutes_per_player"] = (
            team_match["minutes"] / team_match["squad_players"].clip(lower=1)
        )

    # Sort by date for rolling
    team_match = team_match.sort_values(["team_norm", "match_date"]).reset_index(drop=True)

    log.info("Built FBref team-match stats: %d rows", len(team_match))
    return team_match


def _compute_rolling(team_match: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling averages per team (shifted to avoid leakage)."""
    # Core features available for all 9 seasons
    roll_cols_core = [
        "goals", "assists", "shots", "shots_on_target", "shot_accuracy",
        "cards_yellow", "cards_red", "fouls", "fouled",
        "interceptions", "tackles_won", "blocks",
        "squad_players", "avg_minutes_per_player",
    ]
    if "age_years" in team_match.columns:
        roll_cols_core.append("age_years")

    # Advanced features (~13% coverage)
    roll_cols_adv = [
        "xg", "npxg", "xg_assist", "sca", "gca",
        "pass_accuracy", "progressive_passes", "xg_per_shot",
        "tackles", "touches", "carries", "progressive_carries",
        "defense_tackles_won", "defense_interceptions", "defense_clearances",
        "misc_ball_recoveries", "aerial_win_rate",
    ]

    all_roll = [c for c in roll_cols_core + roll_cols_adv if c in team_match.columns]

    rolling_dfs = []
    for team, grp in team_match.groupby("team_norm"):
        grp = grp.sort_values("match_date").copy()
        for col in all_roll:
            grp[f"fb_roll_{col}"] = (
                grp[col].shift(1).rolling(ROLLING_WINDOW, min_periods=2).mean()
            )

        # Rolling std for goals (form volatility)
        if "goals" in grp.columns:
            grp["fb_roll_goals_std"] = (
                grp["goals"].shift(1).rolling(ROLLING_WINDOW, min_periods=2).std()
            )

        # Squad depth rolling (unique players over window)
        if "squad_players" in grp.columns:
            grp["fb_roll_squad_depth"] = (
                grp["squad_players"].shift(1).rolling(ROLLING_WINDOW, min_periods=2).mean()
            )

        rolling_dfs.append(grp)

    return pd.concat(rolling_dfs, ignore_index=True)


def add_fbref_features(df: pd.DataFrame, parsed_dir: Path = None) -> pd.DataFrame:
    """Add FBref per-match rolling features to match DataFrame.

    Uses merge_asof for temporal matching — for each match, finds the
    team's most recent rolling stats computed BEFORE the match date.
    """
    raw = _load_player_stats()
    if raw is None:
        log.warning("Skipping FBref features — no data")
        return df

    original_cols = len(df.columns)

    # Build team-match stats and rolling averages
    team_match = _build_team_match_stats(raw)
    team_match = _compute_rolling(team_match)

    # Rolling columns to merge
    fb_cols = [c for c in team_match.columns if c.startswith("fb_roll_")]

    if not fb_cols:
        log.warning("No FBref rolling features computed")
        return df

    # Normalize match teams
    df["_home_norm"] = df["home_team"].apply(_normalize_team)
    df["_away_norm"] = df["away_team"].apply(_normalize_team)
    df["_match_date"] = pd.to_datetime(df.get("match_date", df.get("date")), errors="coerce")

    # Build lookup
    merge_cols = ["team_norm", "match_date"] + fb_cols
    lookup = team_match[merge_cols].copy()
    lookup = lookup.dropna(subset=["match_date"])
    lookup = lookup.sort_values(["team_norm", "match_date"]).reset_index(drop=True)

    def _merge_side(df_in, team_col, prefix):
        side_df = df_in[["_match_date", team_col]].copy()
        side_df = side_df.rename(columns={team_col: "team_norm"})
        side_df = side_df.dropna(subset=["_match_date"])
        side_df = side_df.sort_values("_match_date").reset_index()

        right_df = lookup.rename(columns={"match_date": "_match_date"}).copy()
        right_df = right_df.sort_values("_match_date").reset_index(drop=True)

        merged = pd.merge_asof(
            side_df,
            right_df,
            on="_match_date",
            by="team_norm",
            direction="backward",
        ).set_index("index")

        for c in fb_cols:
            if c in merged.columns:
                col_name = f"{prefix}_{c}"
                df_in.loc[merged.index, col_name] = merged[c].values

        return df_in

    df = df.sort_values("_match_date")
    df = _merge_side(df, "_home_norm", "home")
    df = _merge_side(df, "_away_norm", "away")

    # Differential features (build all at once to avoid fragmentation)
    diff_data = {}
    for col in fb_cols:
        h_col = f"home_{col}"
        a_col = f"away_{col}"
        if h_col in df.columns and a_col in df.columns:
            diff_data[f"fb_diff_{col.replace('fb_roll_', '')}"] = df[h_col] - df[a_col]

    # Coverage flag
    sample_col = f"home_{fb_cols[0]}"
    if sample_col in df.columns:
        diff_data["fb_coverage"] = df[sample_col].notna().astype(int)
    else:
        diff_data["fb_coverage"] = 0

    if diff_data:
        df = pd.concat([df, pd.DataFrame(diff_data, index=df.index)], axis=1)

    # Cleanup temp columns
    df.drop(columns=["_match_date", "_home_norm", "_away_norm"], errors="ignore", inplace=True)

    n_features = len(df.columns) - original_cols
    coverage_pct = 100 * df["fb_coverage"].mean() if "fb_coverage" in df.columns else 0
    log.info("Added %d FBref rolling features (coverage: %.0f%%)", n_features, coverage_pct)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    raw = _load_player_stats()
    if raw is not None:
        team_match = _build_team_match_stats(raw)
        team_match = _compute_rolling(team_match)
        fb_cols = [c for c in team_match.columns if c.startswith("fb_roll_")]
        print(f"\nTeam-match stats: {team_match.shape}")
        print(f"Rolling features: {len(fb_cols)}")
        print(f"Columns: {fb_cols}")
        print(f"\nTeams: {sorted(team_match['team_norm'].unique())}")
        print(f"\nSample rolling (first non-null):")
        sample = team_match[team_match[fb_cols[0]].notna()].head(3)
        print(sample[["team_norm", "match_date"] + fb_cols[:5]])
