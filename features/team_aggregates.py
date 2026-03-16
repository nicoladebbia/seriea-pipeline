"""Team-level aggregated player stat features.

Converts per-player stats (132+ columns) into team-level aggregates
that capture squad quality beyond goals and possession:
  - Progressive carries/passes (buildup quality)
  - Shot-creating and goal-creating actions
  - Pressing intensity (pressures, tackles won)
  - Aerial dominance
  - Discipline profile
"""

from __future__ import annotations

import logging

import pandas as pd

from storage.paths import parsed_path

log = logging.getLogger(__name__)

# Player stat columns to aggregate at team level (sum or mean as appropriate)
# These come from the 6 FBref tables merged per player
SUM_STATS = [
    "goals", "assists", "pens_made", "pens_att",
    "shots", "shots_on_target",
    "passes_completed", "passes", "passes_pct",
    "passes_total_distance", "passes_progressive_distance",
    "passes_completed_short", "passes_short",
    "passes_completed_medium", "passes_medium",
    "passes_completed_long", "passes_long",
    "assisted_shots", "passes_into_final_third", "passes_into_penalty_area",
    "crosses_into_penalty_area", "progressive_passes",
    "sca", "sca_passes_live", "sca_passes_dead", "sca_take_ons", "sca_shots",
    "gca", "gca_passes_live", "gca_passes_dead", "gca_take_ons", "gca_shots",
    "tackles", "tackles_won", "tackles_def_3rd", "tackles_mid_3rd", "tackles_att_3rd",
    "challenges_lost",
    "blocks", "blocked_shots", "blocked_passes", "interceptions",
    "clearances", "errors",
    "touches", "touches_def_pen_area", "touches_def_3rd",
    "touches_mid_3rd", "touches_att_3rd", "touches_att_pen_area",
    "take_ons", "take_ons_won", "take_ons_tackled",
    "carries", "carries_distance", "carries_progressive_distance",
    "progressive_carries", "carries_into_final_third", "carries_into_penalty_area",
    "miscontrols", "dispossessed",
    "passes_received", "progressive_passes_received",
    "cards_yellow", "cards_red", "cards_yellow_red",
    "fouls", "fouled",
    "offsides", "pens_won", "pens_conceded",
    "own_goals", "ball_recoveries",
    "aerials_won", "aerials_lost",
    "pressures", "pressure_regains", "pressures_def_3rd",
    "pressures_mid_3rd", "pressures_att_3rd",
]

MEAN_STATS = [
    "passes_pct",
]


def add_team_aggregate_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Add team-level aggregated player stats as rolling features.

    For each match, computes team-level sums from player_stats.parquet,
    then adds rolling 5-match averages (shifted).

    Columns added: tagg_roll5_{stat} for home and away.
    """
    ps_path = parsed_path("player_stats")
    if not ps_path.exists():
        log.warning("player_stats.parquet missing; skipping team aggregate features")
        return matches

    player_stats = pd.read_parquet(ps_path)
    df = matches.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.sort_values("match_date").reset_index(drop=True)

    # Compute team-level sums per match
    team_match_agg = _aggregate_player_stats(player_stats)

    # For each team, build a chronological log and compute rolling features
    feature_cols = [c for c in team_match_agg.columns if c not in ("match_id", "team")]

    # Build team-level rolling averages
    team_match_agg = team_match_agg.merge(
        df[["match_id", "match_date"]].drop_duplicates(),
        on="match_id",
        how="left",
    )
    team_match_agg = team_match_agg.sort_values(["team", "match_date"]).reset_index(drop=True)

    rolled: dict[str, pd.Series] = {}
    for stat in feature_cols:
        col = f"tagg_roll5_{stat}"
        team_match_agg[col] = (
            team_match_agg.groupby("team")[stat]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        )
        rolled[col] = True

    roll_cols = [c for c in team_match_agg.columns if c.startswith("tagg_roll5_")]

    # Merge back to match level
    for prefix, team_col in [("home", "home_team"), ("away", "away_team")]:
        team_feats = team_match_agg[["match_id", "team"] + roll_cols].copy()
        rename_map = {c: f"{prefix}_{c}" for c in roll_cols}
        team_feats = team_feats.rename(columns=rename_map)

        df = df.merge(
            team_feats.rename(columns={"team": team_col}),
            on=["match_id", team_col],
            how="left",
        )

    return df


def _aggregate_player_stats(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player-level stats to team level per match."""
    # Convert numeric columns
    numeric_cols = [c for c in SUM_STATS if c in player_stats.columns]
    for col in numeric_cols:
        player_stats[col] = pd.to_numeric(player_stats[col], errors="coerce")

    if "match_id" not in player_stats.columns or "team" not in player_stats.columns:
        return pd.DataFrame()

    agg_dict = {}
    for col in numeric_cols:
        if col in MEAN_STATS:
            agg_dict[col] = "mean"
        else:
            agg_dict[col] = "sum"

    result = player_stats.groupby(["match_id", "team"]).agg(agg_dict).reset_index()
    return result
