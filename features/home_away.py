"""Home/away-specific rolling features.

Same rolling logic as rolling.py, but computed only from
home matches (for the home team) or away matches (for the away team).
"""

from __future__ import annotations

import pandas as pd

from config.settings import ROLLING_WINDOWS

HOME_AWAY_STATS = [
    "goals_scored",
    "goals_conceded",
    "xg_for",
    "xg_against",
    "points",
    "clean_sheet",
]


def add_home_away_features(team_log: pd.DataFrame) -> pd.DataFrame:
    """Add venue-specific rolling features.

    For each window N and stat, adds:
      venue_roll_{N}_{stat}
    where the rolling is computed only over same-venue matches.
    """
    df = team_log.copy()
    df = df.sort_values(["team", "match_date"]).reset_index(drop=True)

    for window in ROLLING_WINDOWS:
        for stat in HOME_AWAY_STATS:
            if stat not in df.columns:
                continue

            col_name = f"venue_roll_{window}_{stat}"

            # Group by team AND venue type, then compute rolling
            df[col_name] = (
                df.groupby(["team", "is_home"])[stat]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            )

    return df
