"""Sofascore domain composite indices.

Reduces ~87 raw ss_roll_* features (×3 for home/away/diff = ~271) into
10 interpretable domain indices using rank-based normalization.

Each index averages the percentile-ranks of its constituent features,
producing a scale-invariant [0, 1] composite. No temporal leakage:
ranking is cross-sectional (across all rows for a given column) and the
underlying ss_roll_* features are already shift(1)-lagged.

Output columns (30 total):
    home_ss_idx_{group}   - 10 columns
    away_ss_idx_{group}   - 10 columns
    ss_idx_diff_{group}   - 10 columns (home − away)
"""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain groups — keys into ss_roll_{metric} column names
# Overlap is intentional (a metric can contribute to multiple facets).
# Missing metrics are silently skipped (nanmean over available columns).
# ---------------------------------------------------------------------------
INDEX_GROUPS: dict[str, list[str]] = {
    "attacking": [
        "xg", "xgot", "goals", "big_chances_created",
        "att_xg", "att_xa", "att_shots", "att_key_passes",
    ],
    "chance_creation": [
        "xa", "key_passes", "progressive_carries",
        "cross_accuracy", "big_chances_created",
    ],
    "shot_quality": [
        "xg_per_shot", "avg_shot_xg", "max_shot_xg", "total_xgot",
        "sm_inside_box_pct", "sm_big_chance_pct", "sm_conversion_rate",
    ],
    "defense": [
        "tackles_won", "interceptions", "blocks", "clearances",
        "def_aerial_win_rate", "def_clearances", "def_blocks", "def_tackles_won",
    ],
    "pressing": [
        "territory_ratio", "press_intensity", "dispossess_rate",
        "ball_recoveries", "opp_half_pass_ratio",
    ],
    "passing": [
        "pass_accuracy", "mid_pass_accuracy", "carry_distance_per_carry",
    ],
    "gk": [
        "gk_saves", "gk_goals_prevented", "gk_rating",
        "dive_saves", "high_claims",
    ],
    "duels": [
        "duel_win_rate", "aerial_win_rate", "mid_duel_win_rate",
        "contest_win_rate", "duel_won_pct", "ground_duels_pct", "aerial_duels_pct",
    ],
    "match_control": [
        "possession", "final_third_entries", "touches_in_opp_box",
        "shots_inside_box_pct", "rating",
    ],
    "set_pieces": [
        "corners", "corner_xg_share", "free_kick_xg_share", "set_piece_xg_pct",
    ],
}


def _compute_side_indices(df: pd.DataFrame, side: str) -> pd.DataFrame:
    """Compute domain indices for one side (home or away).

    For each index group:
      1. Collect {side}_ss_roll_{metric} columns that exist in df
      2. Rank-normalize each column: rank(pct=True) → [0, 1]
      3. Average the ranks → {side}_ss_idx_{group}

    Returns df with new index columns added.
    """
    prefix = f"{side}_ss_roll_"
    available = {c.replace(prefix, ""): c for c in df.columns if c.startswith(prefix)}

    for group_name, metrics in INDEX_GROUPS.items():
        cols = [available[m] for m in metrics if m in available]
        out_col = f"{side}_ss_idx_{group_name}"
        if not cols:
            df[out_col] = np.nan
            continue
        # FIX: Use expanding rank to avoid look-ahead bias.
        # Previous code used df[cols].rank(pct=True) which ranks across ALL rows
        # including future matches. Expanding rank only uses data up to each row.
        ranked_parts = []
        for c in cols:
            ranked_parts.append(
                df[c].expanding().rank(pct=True)
            )
        ranked = pd.concat(ranked_parts, axis=1)
        # Average across group members, ignoring NaN
        df[out_col] = ranked.mean(axis=1, skipna=True)
        # If all source columns were NaN for a row, mean returns NaN — correct

    return df


def compute_sofascore_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Compute domain composite indices from raw Sofascore rolling features.

    Adds 30 columns: 10 home, 10 away, 10 diff (home − away).
    """
    # Check if raw Sofascore features exist
    has_home = any(c.startswith("home_ss_roll_") for c in df.columns)
    has_away = any(c.startswith("away_ss_roll_") for c in df.columns)
    if not (has_home and has_away):
        log.warning("No Sofascore rolling features found — skipping indices")
        return df

    df = _compute_side_indices(df, "home")
    df = _compute_side_indices(df, "away")

    # Diff columns
    for group_name in INDEX_GROUPS:
        h_col = f"home_ss_idx_{group_name}"
        a_col = f"away_ss_idx_{group_name}"
        df[f"ss_idx_diff_{group_name}"] = df[h_col] - df[a_col]

    n_new = len(INDEX_GROUPS) * 3
    log.info("Sofascore indices: %d groups × 3 (home/away/diff) = %d columns", len(INDEX_GROUPS), n_new)
    return df
