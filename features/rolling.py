"""Rolling averages over last N matches (overall, not split by venue).

All features use .shift(1) before .rolling() to prevent data leakage:
the current match is never included in its own features.
"""

from __future__ import annotations

import pandas as pd

from config.settings import ROLLING_WINDOWS

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
