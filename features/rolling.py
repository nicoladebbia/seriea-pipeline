"""Rolling averages over last N matches (overall, not split by venue).

All features use .shift(1) before .rolling() to prevent data leakage:
the current match is never included in its own features.
"""

from __future__ import annotations

import logging

import pandas as pd

from config.settings import ROLLING_WINDOWS

log = logging.getLogger(__name__)

# Columns to compute rolling averages for
ROLLING_STATS = [
    "goals_scored",
    "goals_conceded",
    "xg_for",
    "xg_against",
    "possession",
    "passing_accuracy",
    "shots_on_target",
    "tackles",
    "interceptions",
    "corners",
    "fouls",
    "yellow_cards",
    "red_cards",
    "points",
    "clean_sheet",
]


def add_rolling_features(team_log: pd.DataFrame) -> pd.DataFrame:
    """Add rolling average features to the team match log.

    For each window N in ROLLING_WINDOWS and each stat, adds:
      roll_{N}_{stat}

    Also adds rolling win rate.
    """
    df = team_log.copy()
    df = df.sort_values(["team", "match_date"]).reset_index(drop=True)

    # Group by team AND season to prevent rolling averages bleeding across seasons
    group_cols = ["team", "season"] if "season" in df.columns else ["team"]

    for window in ROLLING_WINDOWS:
        for stat in ROLLING_STATS:
            if stat not in df.columns:
                continue
            col_name = f"roll_{window}_{stat}"
            df[col_name] = (
                df.groupby(group_cols)[stat]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            )

        # Win rate
        df[f"roll_{window}_win_rate"] = (
            df.groupby(group_cols)["result"]
            .transform(
                lambda s: (
                    s.shift(1)
                    .map({"W": 1.0, "D": 0.0, "L": 0.0, "U": float("nan")})
                    .rolling(window, min_periods=1)
                    .mean()
                )
            )
        )

    return df


# Stats where variance (consistency) carries meaningful signal
_VARIANCE_STATS = [
    "goals_scored",
    "goals_conceded",
    "shots_on_target",
    "xg_for",
    "points",
]

# Windows for variance — only 5 and 10 (3-match std is too noisy)
_VARIANCE_WINDOWS = [w for w in ROLLING_WINDOWS if w >= 5]


def add_rolling_variance_features(team_log: pd.DataFrame) -> pd.DataFrame:
    """Add rolling standard deviation and cross-window trend features.

    Variance captures consistency: a team averaging 2.0 goals with std=0.5
    is more predictable than one with std=1.5.

    Cross-window trends capture acceleration: roll_3 - roll_10 shows whether
    recent form is better or worse than the longer-term trend.

    Columns added:
      roll_{N}_{stat}_std           - rolling std for key stats
      {stat}_trend                  - roll_3 - roll_10 (acceleration indicator)
    """
    df = team_log.copy()
    df = df.sort_values(["team", "match_date"]).reset_index(drop=True)

    group_cols = ["team", "season"] if "season" in df.columns else ["team"]
    added = 0

    # Rolling standard deviation for key stats
    for window in _VARIANCE_WINDOWS:
        for stat in _VARIANCE_STATS:
            if stat not in df.columns:
                continue
            col_name = f"roll_{window}_{stat}_std"
            df[col_name] = (
                df.groupby(group_cols)[stat]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=3).std())
            )
            added += 1

    # Cross-window trends: short-term vs long-term (acceleration signal)
    # roll_3 - roll_10: positive = improving, negative = declining
    for stat in _VARIANCE_STATS:
        short_col = f"roll_3_{stat}"
        long_col = f"roll_10_{stat}"
        if short_col in df.columns and long_col in df.columns:
            df[f"{stat}_trend"] = df[short_col] - df[long_col]
            added += 1

    log.info("Added %d rolling variance/trend features", added)
    return df
