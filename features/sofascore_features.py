"""Sofascore-based comprehensive match features.

Three data layers per match (all rolled with 5-match windows):

1. Player-level (player_match_stats.parquet, 80 cols per player per match):
  - Team xG, xGOT, xA rolling averages
  - Shot quality (xG per shot), errors leading to goals/shots
  - Pass accuracy, duel win rate, progressive carries
  - Player rating (mean + std dev for consistency)
  - Position-based: attacker xG, midfielder duels, defender aerials
  - GK quality: saves, goals prevented, keeper value, save value
  - Pressing: possession_lost, touches ratio, dispossessed
  - Rotation: XI changes, minutes concentration
  - Dribble contests, offsides, unsuccessful touches

2. Team-level match stats (match_team_stats.parquet, from match.stats() endpoint):
  - Possession %, corners, free kicks, throw-ins, offsides
  - Shots inside/outside box, hit woodwork
  - Final third entries, touches in opponent box
  - Duel %, ground duels %, aerial duels %, dribbles %
  - Errors leading to shots/goals
  - GK: dive saves, high claims

3. Shotmap aggregates (shotmap_stats.parquet):
  - Avg/max shot xG, total xGoT
  - Shot location: inside box %, header %, open play %
  - Set piece %, counter attack %, conversion rate
  - Big chance shot %
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config.team_names import normalize_team

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
SOFASCORE_PATH = PROJECT_ROOT / "data" / "external" / "sofascore" / "player_match_stats.parquet"

ROLLING_WINDOW = 5


def _normalize(name: str) -> str:
    """Normalize Sofascore team names to pipeline canonical form.

    Uses the central normalize_team() which handles case-insensitive matching,
    so Sofascore's varied casing (e.g., "ac milan", "Inter") is resolved
    to canonical title-case names ("Milan", "Inter").
    """
    if pd.isna(name):
        return ""
    return normalize_team(name)


def _load_sofascore() -> pd.DataFrame | None:
    if not SOFASCORE_PATH.exists():
        log.warning("Sofascore data not found at %s", SOFASCORE_PATH)
        return None
    df = pd.read_parquet(SOFASCORE_PATH)
    log.info("Loaded Sofascore: %d rows, %d columns", len(df), len(df.columns))
    return df


def _build_team_match_stats(player_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-player stats to per-team-per-match.

    One row per (match_id, team) with summed/averaged team stats.
    """
    # Ensure numeric
    num_cols = [
        "xg", "xgot", "xa", "goals", "total_shots", "shots_on_target",
        "big_chances_created", "big_chances_missed",
        "assists", "key_passes", "accurate_passes", "total_passes",
        "accurate_long_balls", "total_long_balls",
        "accurate_crosses", "total_crosses",
        "carries", "progressive_carries", "carry_distance", "progressive_carry_distance",
        "tackles", "tackles_won", "interceptions", "clearances",
        "ball_recoveries", "blocks", "shots_blocked", "last_man_tackle",
        "duels_won", "duels_lost", "aerial_won", "aerial_lost",
        "touches", "possession_lost", "fouls", "was_fouled",
        "minutes", "rating",
        # New fields (added Feb 2026)
        "error_to_goal", "error_to_shot",
        "contest_won", "contest_total", "challenge_lost",
        "shots_off_target", "unsuccessful_touch", "offsides",
        "total_opp_half_passes", "total_own_half_passes",
    ]
    for c in num_cols:
        if c in player_df.columns:
            player_df[c] = pd.to_numeric(player_df[c], errors="coerce")

    # Normalize team names
    player_df["team_norm"] = player_df["team"].apply(_normalize)

    # Group by match + team
    agg_map = {}
    sum_cols = [
        "xg", "xgot", "xa", "goals", "total_shots", "shots_on_target",
        "big_chances_created", "big_chances_missed",
        "assists", "key_passes", "accurate_passes", "total_passes",
        "accurate_long_balls", "total_long_balls",
        "accurate_crosses", "total_crosses",
        "carries", "progressive_carries", "carry_distance", "progressive_carry_distance",
        "tackles", "tackles_won", "interceptions", "clearances",
        "ball_recoveries", "blocks", "shots_blocked", "last_man_tackle",
        "duels_won", "duels_lost", "aerial_won", "aerial_lost",
        "touches", "possession_lost", "fouls", "was_fouled",
        # New fields
        "error_to_goal", "error_to_shot",
        "contest_won", "contest_total", "challenge_lost",
        "shots_off_target", "unsuccessful_touch", "offsides",
        "total_opp_half_passes", "total_own_half_passes",
    ]
    for c in sum_cols:
        if c in player_df.columns:
            agg_map[c] = "sum"

    if "rating" in player_df.columns:
        agg_map["rating"] = "mean"
    if "minutes" in player_df.columns:
        agg_map["minutes"] = "sum"

    # Keep match metadata
    meta_cols = ["date", "season", "home_team", "away_team", "is_home"]
    group_cols = ["match_id", "team_norm"]

    team_match = player_df.groupby(group_cols).agg(agg_map).reset_index()

    # Add metadata from first row of each group
    meta = player_df.groupby(group_cols)[meta_cols].first().reset_index()
    team_match = team_match.merge(meta, on=group_cols, how="left")

    # Derived per-match metrics
    team_match["pass_accuracy"] = (
        team_match["accurate_passes"] / team_match["total_passes"].clip(lower=1)
    )
    team_match["xg_per_shot"] = (
        team_match["xg"] / team_match["total_shots"].clip(lower=1)
    )
    team_match["duel_win_rate"] = (
        team_match["duels_won"] /
        (team_match["duels_won"] + team_match["duels_lost"]).clip(lower=1)
    )
    team_match["aerial_win_rate"] = (
        team_match["aerial_won"] /
        (team_match["aerial_won"] + team_match["aerial_lost"]).clip(lower=1)
    )

    # Contest win rate (dribbles)
    if "contest_won" in team_match.columns and "contest_total" in team_match.columns:
        team_match["contest_win_rate"] = (
            team_match["contest_won"] / team_match["contest_total"].clip(lower=1)
        )
    else:
        team_match["contest_win_rate"] = 0

    # Territorial dominance: passes in opp half vs own half
    if "total_opp_half_passes" in team_match.columns and "total_own_half_passes" in team_match.columns:
        total_dir = (team_match["total_opp_half_passes"] + team_match["total_own_half_passes"]).clip(lower=1)
        team_match["opp_half_pass_ratio"] = team_match["total_opp_half_passes"] / total_dir
    else:
        team_match["opp_half_pass_ratio"] = 0

    # Carry efficiency metrics
    if "carry_distance" in team_match.columns and "carries" in team_match.columns:
        team_match["carry_distance_per_carry"] = (
            team_match["carry_distance"] / team_match["carries"].clip(lower=1)
        )
    else:
        team_match["carry_distance_per_carry"] = 0

    # Shot blocking rate
    if "shots_blocked" in team_match.columns and "total_shots" in team_match.columns:
        team_match["shot_blocked_rate"] = (
            team_match["shots_blocked"] / team_match["total_shots"].clip(lower=1)
        )
    else:
        team_match["shot_blocked_rate"] = 0

    # Crossing accuracy
    if "accurate_crosses" in team_match.columns and "total_crosses" in team_match.columns:
        team_match["cross_accuracy"] = (
            team_match["accurate_crosses"] / team_match["total_crosses"].clip(lower=1)
        )
    else:
        team_match["cross_accuracy"] = 0

    # Sort by date for rolling
    team_match["date"] = pd.to_datetime(team_match["date"], errors="coerce")
    team_match = team_match.sort_values(["team_norm", "date"]).reset_index(drop=True)

    log.info("Built team-match stats: %d rows", len(team_match))
    return team_match


def _compute_rolling(team_match: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling averages per team (shifted to avoid leakage)."""
    # Original team-level rolling features
    roll_cols = [
        "xg", "xgot", "xa", "goals", "total_shots", "shots_on_target",
        "big_chances_created", "key_passes",
        "pass_accuracy", "xg_per_shot", "duel_win_rate", "aerial_win_rate",
        "tackles_won", "interceptions", "ball_recoveries",
        "blocks", "clearances", "fouls", "was_fouled",
        "progressive_carries", "rating",
        "carry_distance_per_carry", "shot_blocked_rate", "last_man_tackle",
        "cross_accuracy",
        # New player-derived
        "error_to_goal", "error_to_shot",
        "contest_win_rate", "unsuccessful_touch", "offsides",
        "opp_half_pass_ratio",
    ]

    # Position-based features
    pos_cols = [
        "att_xg", "att_xa", "att_shots", "att_key_passes", "att_rating",
        "mid_pass_accuracy", "mid_duel_win_rate", "mid_tackles", "mid_interceptions", "mid_rating",
        "def_aerial_win_rate", "def_clearances", "def_blocks", "def_tackles_won", "def_rating",
        "gk_saves", "gk_goals_prevented", "gk_rating",
    ]

    # Pressing/territorial features
    press_cols = [
        "territory_ratio", "press_intensity", "dispossess_rate",
        "total_poss_lost",
    ]

    # Rotation features
    rotation_cols = [
        "xi_changes", "mins_concentration",
    ]

    # Shot situation features
    shot_situation_cols = [
        "counter_xg_pct", "set_piece_xg_pct", "open_play_xg_pct",
        "header_shot_pct", "first_half_xg_share", "last_15_xg_share",
        "corner_xg_share", "free_kick_xg_share", "penalty_xg_share",
        "penalties_taken", "penalties_scored",
    ]

    # Match-level team stats (from match.stats() endpoint)
    match_team_cols = [
        "possession", "corners", "offsides", "throw_ins",
        "shots_inside_box", "shots_outside_box", "shots_inside_box_pct",
        "hit_woodwork", "big_chances_scored",
        "touches_in_opp_box", "fouled_final_third",
        "final_third_entries", "final_third_phases",
        "duel_won_pct", "ground_duels_pct", "aerial_duels_pct", "dribbles_pct",
        "errors_to_shot", "errors_to_goal",
        "dive_saves", "high_claims",
        "dispossessed",
    ]

    # Shotmap aggregate features
    shotmap_cols = [
        "avg_shot_xg", "max_shot_xg", "total_xgot",
        "sm_inside_box_pct", "sm_header_pct", "sm_open_play_pct",
        "sm_set_piece_pct", "sm_counter_pct",
        "sm_conversion_rate", "sm_big_chance_pct",
        # Shot distance features — continuous measure of chance quality
        "sm_avg_shot_distance", "sm_median_shot_distance",
        "sm_shot_distance_std", "sm_close_range_pct",
    ]

    all_roll = (roll_cols + pos_cols + press_cols + rotation_cols
                + shot_situation_cols + match_team_cols + shotmap_cols)
    all_roll = [c for c in all_roll if c in team_match.columns]

    rolling_dfs = []
    for team, grp in team_match.groupby("team_norm"):
        grp = grp.sort_values("date").copy()
        for col in all_roll:
            grp[f"ss_roll_{col}"] = (
                grp[col].shift(1).rolling(ROLLING_WINDOW, min_periods=2).mean()
            )

        # Rolling std for xG (form volatility)
        if "xg" in grp.columns:
            grp["ss_roll_xg_std"] = (
                grp["xg"].shift(1).rolling(ROLLING_WINDOW, min_periods=2).std()
            )

        # Rolling std for rating (player consistency)
        if "rating" in grp.columns:
            grp["ss_roll_rating_std"] = (
                grp["rating"].shift(1).rolling(ROLLING_WINDOW, min_periods=2).std()
            )

        rolling_dfs.append(grp)

    return pd.concat(rolling_dfs, ignore_index=True)


def _compute_xg_concentration(player_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-match xG concentration (top 2 players' share).

    Returns DataFrame with (match_id, team_norm, top2_xg_share).
    """
    player_df["team_norm"] = player_df["team"].apply(_normalize)
    player_df["xg_num"] = pd.to_numeric(player_df["xg"], errors="coerce").fillna(0)

    rows = []
    for (mid, team), grp in player_df.groupby(["match_id", "team_norm"]):
        total = grp["xg_num"].sum()
        if total > 0:
            top2 = grp.nlargest(2, "xg_num")["xg_num"].sum()
            rows.append({"match_id": mid, "team_norm": team, "top2_xg_share": top2 / total})
        else:
            rows.append({"match_id": mid, "team_norm": team, "top2_xg_share": 0.0})

    return pd.DataFrame(rows)


def _build_position_stats(player_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate stats by position group per team per match.

    Position groups: F (attackers), M (midfielders), D (defenders), G (goalkeepers).
    Returns one row per (match_id, team_norm) with position-prefixed columns.
    """
    df = player_df.copy()
    df["team_norm"] = df["team"].apply(_normalize)

    # Ensure numeric
    for c in ["xg", "xgot", "xa", "total_shots", "key_passes", "accurate_passes",
              "total_passes", "tackles", "tackles_won", "interceptions",
              "duels_won", "duels_lost", "aerial_won", "aerial_lost",
              "clearances", "blocks", "saves", "goals_prevented", "rating",
              "opp_half_passes", "own_half_passes", "dispossessed",
              "touches", "possession_lost"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    group_cols = ["match_id", "team_norm"]
    results = []

    # Attackers (F): xG, xA, shots, key passes
    fwd = df[df["position"] == "F"]
    if not fwd.empty:
        fwd_agg = fwd.groupby(group_cols).agg(
            att_xg=("xg", "sum"),
            att_xa=("xa", "sum"),
            att_shots=("total_shots", "sum"),
            att_key_passes=("key_passes", "sum"),
            att_rating=("rating", "mean"),
        ).reset_index()
        results.append(fwd_agg)

    # Midfielders (M): passes, tackles, duels
    mid = df[df["position"] == "M"]
    if not mid.empty:
        mid_agg = mid.groupby(group_cols).agg(
            mid_accurate_passes=("accurate_passes", "sum"),
            mid_total_passes=("total_passes", "sum"),
            mid_tackles=("tackles", "sum"),
            mid_interceptions=("interceptions", "sum"),
            mid_duels_won=("duels_won", "sum"),
            mid_duels_lost=("duels_lost", "sum"),
            mid_rating=("rating", "mean"),
        ).reset_index()
        # Derived
        mid_agg["mid_pass_accuracy"] = (
            mid_agg["mid_accurate_passes"] / mid_agg["mid_total_passes"].clip(lower=1)
        )
        mid_agg["mid_duel_win_rate"] = (
            mid_agg["mid_duels_won"] /
            (mid_agg["mid_duels_won"] + mid_agg["mid_duels_lost"]).clip(lower=1)
        )
        results.append(mid_agg)

    # Defenders (D): aerials, clearances, blocks
    defs = df[df["position"] == "D"]
    if not defs.empty:
        def_agg = defs.groupby(group_cols).agg(
            def_aerial_won=("aerial_won", "sum"),
            def_aerial_lost=("aerial_lost", "sum"),
            def_clearances=("clearances", "sum"),
            def_blocks=("blocks", "sum"),
            def_tackles_won=("tackles_won", "sum"),
            def_rating=("rating", "mean"),
        ).reset_index()
        def_agg["def_aerial_win_rate"] = (
            def_agg["def_aerial_won"] /
            (def_agg["def_aerial_won"] + def_agg["def_aerial_lost"]).clip(lower=1)
        )
        results.append(def_agg)

    # Goalkeepers (G): saves, rating, goals prevented
    gk = df[df["position"] == "G"]
    if not gk.empty:
        gk_agg = gk.groupby(group_cols).agg(
            gk_saves=("saves", "sum"),
            gk_goals_prevented=("goals_prevented", "sum"),
            gk_rating=("rating", "mean"),
        ).reset_index()
        results.append(gk_agg)

    if not results:
        return pd.DataFrame(columns=["match_id", "team_norm"])

    # Merge all position groups
    merged = results[0]
    for r in results[1:]:
        merged = merged.merge(r, on=group_cols, how="outer")

    return merged


def _build_pressing_stats(player_df: pd.DataFrame) -> pd.DataFrame:
    """Compute pressing/territorial stats per team per match.

    - Possession lost (proxy for high press + turnovers)
    - Territory: opp_half_passes / total team passes
    - Dispossession rate
    """
    if player_df.empty:
        return pd.DataFrame(columns=["match_id", "team_norm"])

    df = player_df.copy()
    df["team_norm"] = df["team"].apply(_normalize)

    for c in ["touches", "possession_lost", "dispossessed",
              "opp_half_passes", "own_half_passes"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    group_cols = ["match_id", "team_norm"]
    agg = df.groupby(group_cols).agg(
        total_touches=("touches", "sum"),
        total_poss_lost=("possession_lost", "sum"),
        total_dispossessed=("dispossessed", "sum"),
        total_opp_half_passes=("opp_half_passes", "sum"),
        total_own_half_passes=("own_half_passes", "sum"),
    ).reset_index()

    # Territorial dominance: share of passes in opponent's half
    total_halves = (agg["total_opp_half_passes"] + agg["total_own_half_passes"]).clip(lower=1)
    agg["territory_ratio"] = agg["total_opp_half_passes"] / total_halves

    # Pressing intensity: possession_lost per touch (higher = more pressing)
    agg["press_intensity"] = agg["total_poss_lost"] / agg["total_touches"].clip(lower=1)

    # Dispossession rate per touch
    agg["dispossess_rate"] = agg["total_dispossessed"] / agg["total_touches"].clip(lower=1)

    return agg


def _build_rotation_stats(player_df: pd.DataFrame) -> pd.DataFrame:
    """Compute squad rotation and freshness indicators.

    - XI changes between consecutive matches
    - Minutes concentration (top 11 vs squad)
    - Key player (top-3 xG) availability
    """
    if player_df.empty:
        return pd.DataFrame(columns=["match_id", "team_norm"])

    df = player_df.copy()
    df["team_norm"] = df["team"].apply(_normalize)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["minutes", "xg", "is_starter"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    group_cols = ["match_id", "team_norm"]

    # Starters per match
    starters = df[df["is_starter"] == 1].copy()

    rows = []
    for team, team_df in df.groupby("team_norm"):
        match_dates = team_df.groupby("match_id")["date"].first().sort_values()
        match_order = match_dates.index.tolist()

        prev_starter_ids = None
        for mid in match_order:
            match_players = team_df[team_df["match_id"] == mid]
            starter_ids = set(
                match_players[match_players["is_starter"] == 1]["player_id"].dropna()
            )

            # XI changes from previous match
            xi_changes = np.nan
            if prev_starter_ids is not None and starter_ids:
                xi_changes = len(starter_ids - prev_starter_ids)

            # Minutes concentration: top 11 by minutes / total minutes
            total_mins = match_players["minutes"].sum()
            if total_mins > 0 and len(match_players) > 11:
                top11_mins = match_players.nlargest(11, "minutes")["minutes"].sum()
                mins_concentration = top11_mins / total_mins
            else:
                mins_concentration = 1.0

            # Key player availability: did top-3 xG players in recent history start?
            # (Approximated here; full implementation needs rolling context)
            n_starters = len(starter_ids)

            rows.append({
                "match_id": mid,
                "team_norm": team,
                "xi_changes": xi_changes,
                "mins_concentration": mins_concentration,
                "n_starters": n_starters,
            })

            prev_starter_ids = starter_ids

    return pd.DataFrame(rows)


def _build_shot_situation_stats(shots_path=None) -> pd.DataFrame | None:
    """Compute per-team shot situation breakdown from Sofascore shotmap.

    Returns DataFrame with (match_id, team_norm, counter_xg_pct, set_piece_xg_pct,
    open_play_xg_pct, header_shot_pct, first_half_xg_share, last_15_xg_share).
    """
    if shots_path is None:
        shots_path = PROJECT_ROOT / "data" / "external" / "sofascore" / "all_shots_with_xg.parquet"

    if not shots_path.exists():
        log.warning("Shot data not found at %s", shots_path)
        return None

    shots = pd.read_parquet(shots_path)
    if shots.empty:
        return None

    # Load player data to map player_id -> team
    player_df = _load_sofascore()
    if player_df is None:
        log.warning("Cannot map shots to teams without player data")
        return None

    # Build player-match-team lookup
    player_team_map = player_df[["match_id", "player_id", "team"]].drop_duplicates()
    player_team_map["team_norm"] = player_team_map["team"].apply(_normalize)

    # Ensure consistent types for merge keys (match_id can be int or str)
    for merge_col in ["match_id", "player_id"]:
        if merge_col in shots.columns and merge_col in player_team_map.columns:
            shots[merge_col] = shots[merge_col].astype(str)
            player_team_map[merge_col] = player_team_map[merge_col].astype(str)

    # Merge team into shots
    shots = shots.merge(
        player_team_map[["match_id", "player_id", "team_norm"]],
        on=["match_id", "player_id"],
        how="left"
    )

    # Drop rows without team mapping
    shots = shots.dropna(subset=["team_norm"])

    # Use xg_predicted if xg is NaN (for 2017-2022 seasons)
    shots["xg_val"] = shots["xg"].fillna(shots["xg_predicted"]).fillna(0)

    # Ensure numeric columns
    for c in ["is_fast_break", "is_set_piece", "time"]:
        if c in shots.columns:
            shots[c] = pd.to_numeric(shots[c], errors="coerce").fillna(0)

    rows = []
    for (mid, team), grp in shots.groupby(["match_id", "team_norm"]):
        total_xg = grp["xg_val"].sum()
        total_shots = len(grp)

        # Counter attack xG percentage
        counter_xg = grp[grp["is_fast_break"] == 1]["xg_val"].sum()
        counter_xg_pct = counter_xg / total_xg if total_xg > 0 else 0

        # Set piece xG percentage
        set_piece_xg = grp[grp["is_set_piece"] == 1]["xg_val"].sum()
        set_piece_xg_pct = set_piece_xg / total_xg if total_xg > 0 else 0

        # Open play xG percentage
        open_play_xg_pct = max(0, 1 - counter_xg_pct - set_piece_xg_pct)

        # Header shot percentage
        header_shots = len(grp[grp["body_part"] == "head"])
        header_shot_pct = header_shots / total_shots if total_shots > 0 else 0

        # First half xG share
        first_half_xg = grp[grp["time"] <= 45]["xg_val"].sum()
        first_half_xg_share = first_half_xg / total_xg if total_xg > 0 else 0

        # Last 15 minutes xG share
        last_15_xg = grp[grp["time"] >= 75]["xg_val"].sum()
        last_15_xg_share = last_15_xg / total_xg if total_xg > 0 else 0

        # Granular set-piece situation breakdown (using 'situation' column)
        if "situation" in grp.columns:
            corner_xg = grp[grp["situation"] == "corner"]["xg_val"].sum()
            corner_xg_share = corner_xg / total_xg if total_xg > 0 else 0

            fk_xg = grp[grp["situation"] == "free-kick"]["xg_val"].sum()
            free_kick_xg_share = fk_xg / total_xg if total_xg > 0 else 0

            pen_xg = grp[grp["situation"] == "penalty"]["xg_val"].sum()
            penalty_xg_share = pen_xg / total_xg if total_xg > 0 else 0
        else:
            corner_xg_share = 0
            free_kick_xg_share = 0
            penalty_xg_share = 0

        # Penalty patterns
        if "is_penalty" in grp.columns:
            pen_mask = grp["is_penalty"] == 1
            penalties_taken = int(pen_mask.sum())
            penalties_scored = int(
                (pen_mask & (grp["is_goal"] == 1)).sum()
            ) if "is_goal" in grp.columns else 0
        else:
            penalties_taken = 0
            penalties_scored = 0

        rows.append({
            "match_id": mid,
            "team_norm": team,
            "counter_xg_pct": counter_xg_pct,
            "set_piece_xg_pct": set_piece_xg_pct,
            "open_play_xg_pct": open_play_xg_pct,
            "header_shot_pct": header_shot_pct,
            "first_half_xg_share": first_half_xg_share,
            "last_15_xg_share": last_15_xg_share,
            "corner_xg_share": corner_xg_share,
            "free_kick_xg_share": free_kick_xg_share,
            "penalty_xg_share": penalty_xg_share,
            "penalties_taken": penalties_taken,
            "penalties_scored": penalties_scored,
        })

    df = pd.DataFrame(rows)
    log.info("Built shot situation stats: %d team-matches", len(df))
    return df


def _build_match_level_team_stats() -> pd.DataFrame | None:
    """Load match-level team statistics from match_team_stats.parquet.

    These come from Sofascore's match.stats() endpoint — team-level aggregates
    that can't be derived from player data: possession, corners, final third
    entries, touches in opponent box, errors leading to goals, etc.

    Only uses the 'ALL' period (full match). Returns DataFrame with
    (match_id, team_norm, possession, corners, ...).
    """
    path = PROJECT_ROOT / "data" / "external" / "sofascore" / "match_team_stats.parquet"
    if not path.exists():
        log.warning("match_team_stats.parquet not found at %s", path)
        return None

    df = pd.read_parquet(path)
    if df.empty:
        return None

    # Only full-match stats for rolling features
    df = df[df["period"] == "ALL"].copy()

    df["team_norm"] = df["team"].apply(_normalize)

    # Select the columns that add signal beyond what player-level aggregation gives us
    keep_cols = [
        "match_id", "team_norm",
        "possession", "corners", "offsides", "throw_ins", "free_kicks",
        "shots_inside_box", "shots_outside_box", "hit_woodwork",
        "big_chances_scored",
        "touches_in_opp_box", "fouled_final_third",
        "final_third_entries", "final_third_phases",
        "duel_won_pct", "ground_duels_pct", "aerial_duels_pct", "dribbles_pct",
        "errors_to_shot", "errors_to_goal",
        "gk_saves", "goals_prevented", "dive_saves", "high_claims",
        "dispossessed",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    result = df[keep_cols].copy()

    # Derived: shot location ratio
    if "shots_inside_box" in result.columns and "shots_outside_box" in result.columns:
        total = (result["shots_inside_box"] + result["shots_outside_box"]).clip(lower=1)
        result["shots_inside_box_pct"] = result["shots_inside_box"] / total

    log.info("Built match-level team stats: %d rows, %d columns", len(result), len(result.columns))
    return result


def _build_shotmap_aggregate_stats() -> pd.DataFrame | None:
    """Load pre-aggregated shotmap statistics from shotmap_stats.parquet.

    These come from per-shot data: shot locations, body parts, situations,
    xG distribution. Provides signal about HOW a team creates chances.

    Returns DataFrame with (match_id, team_norm, shots_inside_box_sm, ...).
    """
    path = PROJECT_ROOT / "data" / "external" / "sofascore" / "shotmap_stats.parquet"
    if not path.exists():
        log.warning("shotmap_stats.parquet not found at %s", path)
        return None

    df = pd.read_parquet(path)
    if df.empty:
        return None

    df["team_norm"] = df["team"].apply(_normalize)

    # Derived ratios
    total = df["shots_total"].clip(lower=1)
    df["sm_inside_box_pct"] = df["shots_inside_box"] / total
    df["sm_header_pct"] = df["shots_header"] / total
    df["sm_open_play_pct"] = df["shots_open_play"] / total
    df["sm_set_piece_pct"] = df["shots_set_piece"] / total
    df["sm_counter_pct"] = df["shots_counter"] / total
    df["sm_conversion_rate"] = df["goals_from_shots"] / total
    df["sm_big_chance_pct"] = df["big_chance_shots"] / total

    # Shot distance features — avg/median distance from goal, consistency,
    # and close-range shot % capture HOW a team creates chances beyond
    # the binary inside/outside box split.
    # These columns may be None for older data (pre-backfill).
    if "avg_shot_distance" in df.columns:
        df["sm_avg_shot_distance"] = df["avg_shot_distance"]
        df["sm_median_shot_distance"] = df.get("median_shot_distance")
        df["sm_shot_distance_std"] = df.get("shot_distance_std")
        df["sm_close_range_pct"] = df.get("close_range_pct", 0)

    keep_cols = [
        "match_id", "team_norm",
        "avg_shot_xg", "max_shot_xg", "total_xgot",
        "sm_inside_box_pct", "sm_header_pct", "sm_open_play_pct",
        "sm_set_piece_pct", "sm_counter_pct",
        "sm_conversion_rate", "sm_big_chance_pct",
        # Shot distance features
        "sm_avg_shot_distance", "sm_median_shot_distance",
        "sm_shot_distance_std", "sm_close_range_pct",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    result = df[keep_cols].copy()

    log.info("Built shotmap aggregate stats: %d rows, %d columns", len(result), len(result.columns))
    return result


def add_sofascore_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Add Sofascore rolling features to match DataFrame.

    Adds ~55+ rolling features per side (home/away) based on per-match
    team stats from Sofascore, including position-based, GK, pressing,
    rotation, team-level match stats, and shotmap features.
    """
    raw = _load_sofascore()
    if raw is None:
        log.warning("Skipping Sofascore features — no data")
        return matches

    df = matches.copy()

    # Build team-match stats and rolling averages
    team_match = _build_team_match_stats(raw.copy())

    # Merge position-based stats
    pos_stats = _build_position_stats(raw.copy())
    if not pos_stats.empty:
        team_match = team_match.merge(pos_stats, on=["match_id", "team_norm"], how="left")
        log.info("Merged position stats: %d columns", len(pos_stats.columns) - 2)

    # Merge pressing stats
    press_stats = _build_pressing_stats(raw.copy())
    if not press_stats.empty:
        team_match = team_match.merge(press_stats, on=["match_id", "team_norm"], how="left")
        log.info("Merged pressing stats: %d columns", len(press_stats.columns) - 2)

    # Merge rotation stats
    rotation_stats = _build_rotation_stats(raw.copy())
    if not rotation_stats.empty:
        team_match = team_match.merge(rotation_stats, on=["match_id", "team_norm"], how="left")
        log.info("Merged rotation stats: %d columns", len(rotation_stats.columns) - 2)

    # Merge shot situation stats
    shot_stats = _build_shot_situation_stats()
    if shot_stats is not None and not shot_stats.empty:
        # Align match_id dtype (shots may have str, team_match may have int)
        if shot_stats["match_id"].dtype != team_match["match_id"].dtype:
            shot_stats["match_id"] = shot_stats["match_id"].astype(team_match["match_id"].dtype)
        team_match = team_match.merge(shot_stats, on=["match_id", "team_norm"], how="left")
        log.info("Merged shot situation stats: %d columns", len(shot_stats.columns) - 2)

    # Merge match-level team stats (possession, corners, final third entries, errors, etc.)
    match_team = _build_match_level_team_stats()
    if match_team is not None and not match_team.empty:
        if match_team["match_id"].dtype != team_match["match_id"].dtype:
            match_team["match_id"] = match_team["match_id"].astype(team_match["match_id"].dtype)
        team_match = team_match.merge(match_team, on=["match_id", "team_norm"], how="left")
        log.info("Merged match-level team stats: %d new columns", len(match_team.columns) - 2)

    # Merge shotmap aggregate stats (shot location %, body part %, conversion rate, etc.)
    shotmap_agg = _build_shotmap_aggregate_stats()
    if shotmap_agg is not None and not shotmap_agg.empty:
        if shotmap_agg["match_id"].dtype != team_match["match_id"].dtype:
            shotmap_agg["match_id"] = shotmap_agg["match_id"].astype(team_match["match_id"].dtype)
        team_match = team_match.merge(shotmap_agg, on=["match_id", "team_norm"], how="left")
        log.info("Merged shotmap aggregate stats: %d new columns", len(shotmap_agg.columns) - 2)

    team_match = _compute_rolling(team_match)

    # xG concentration
    conc = _compute_xg_concentration(raw.copy())
    team_match = team_match.merge(conc, on=["match_id", "team_norm"], how="left")

    # Rolling columns to merge
    ss_cols = [c for c in team_match.columns if c.startswith("ss_roll_")]
    ss_cols.append("top2_xg_share")

    # Normalize match teams
    df["home_team_norm"] = df["home_team"].apply(_normalize)
    df["away_team_norm"] = df["away_team"].apply(_normalize)

    # Extract season info for matching
    if "season" not in df.columns and "match_date" in df.columns:
        df["match_date_dt"] = pd.to_datetime(df["match_date"], errors="coerce")
        df["_year"] = df["match_date_dt"].dt.year
        df["_month"] = df["match_date_dt"].dt.month
        df["season"] = df.apply(
            lambda r: f"{r['_year']}-{r['_year']+1}" if r["_month"] >= 8
            else f"{r['_year']-1}-{r['_year']}",
            axis=1,
        )
        df.drop(columns=["match_date_dt", "_year", "_month"], inplace=True)

    # Match date for temporal matching
    df["_match_date"] = pd.to_datetime(df.get("match_date", df.get("date")), errors="coerce")

    # Merge: for each match, find the team's most recent rolling stats BEFORE the match
    # Strategy: merge on (team_norm, match date closest before)

    # Build lookup: team_norm -> sorted list of (date, row)
    team_match["_date"] = team_match["date"]
    merge_cols = ["team_norm", "_date"] + [c for c in ss_cols if c in team_match.columns]
    lookup = team_match[merge_cols].copy()
    lookup = lookup.dropna(subset=["_date"])
    lookup = lookup.sort_values(["team_norm", "_date"]).reset_index(drop=True)

    def _merge_side(df_in, team_col, prefix):
        """Merge rolling stats for one side (home/away) using merge_asof."""
        side_df = df_in[["_match_date", team_col]].copy()
        side_df = side_df.rename(columns={team_col: "team_norm"})
        side_df = side_df.dropna(subset=["_match_date"])
        side_df = side_df.sort_values("_match_date").reset_index()

        right_df = lookup.copy()
        right_df = right_df.rename(columns={"_date": "_match_date"})
        right_df = right_df.dropna(subset=["_match_date"])
        right_df = right_df.sort_values("_match_date").reset_index(drop=True)

        merged = pd.merge_asof(
            side_df,
            right_df,
            on="_match_date",
            by="team_norm",
            direction="backward",
        ).set_index("index")

        # Rename to home_/away_ prefix
        for c in [col for col in ss_cols if col in merged.columns]:
            col_name = f"{prefix}_{c}"
            df_in.loc[merged.index, col_name] = merged[c].values

        return df_in

    df = df.sort_values("_match_date")
    df = _merge_side(df, "home_team_norm", "home")
    df = _merge_side(df, "away_team_norm", "away")

    # Differential features (build all at once to avoid fragmentation)
    diff_data = {}
    for col in ss_cols:
        if col in team_match.columns:
            h_col = f"home_{col}"
            a_col = f"away_{col}"
            if h_col in df.columns and a_col in df.columns:
                diff_data[f"ss_diff_{col}"] = df[h_col] - df[a_col]

    # Coverage flags — rating available all seasons, xG only from 2022-2023+
    rating_col = "home_ss_roll_rating"
    xg_col = "home_ss_roll_xg"
    diff_data["ss_coverage"] = df[rating_col].notna().astype(int) if rating_col in df.columns else 0
    diff_data["ss_xg_coverage"] = df[xg_col].notna().astype(int) if xg_col in df.columns else 0

    if diff_data:
        df = pd.concat([df, pd.DataFrame(diff_data, index=df.index)], axis=1)

    # Cleanup
    df.drop(columns=["_match_date", "home_team_norm", "away_team_norm"], errors="ignore", inplace=True)

    n_features = len([c for c in df.columns if c.startswith(("home_ss_", "away_ss_", "ss_diff_", "ss_coverage", "ss_xg"))])
    log.info("Added %d Sofascore features (rating coverage: %d/%.0f%%, xG coverage: %d/%.0f%%)",
             n_features,
             df["ss_coverage"].sum(), 100 * df["ss_coverage"].mean(),
             df["ss_xg_coverage"].sum(), 100 * df["ss_xg_coverage"].mean())

    return df
