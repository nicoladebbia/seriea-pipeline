"""xG over/under-performance trends.

Tracks whether a team is outperforming or underperforming their xG,
which is a strong regression indicator.
"""

from __future__ import annotations

import pandas as pd

from config.settings import ROLLING_WINDOWS


def add_xg_trends(team_log: pd.DataFrame) -> pd.DataFrame:
    """Add xG performance trend features.

    Columns added per window N:
      xg_diff_roll_{N}          - rolling avg of (goals_scored - xg_for)
      xg_against_diff_roll_{N}  - rolling avg of (goals_conceded - xg_against)

    Plus cumulative season totals:
      xg_for_cumulative
      xg_against_cumulative
      goals_minus_xg_cumulative
      goals_conceded_minus_xga_cumulative
    """
    df = team_log.copy()
    df = df.sort_values(["team", "match_date"]).reset_index(drop=True)

    # Raw diffs
    df["_xg_diff"] = df["goals_scored"] - df["xg_for"]
    df["_xga_diff"] = df["goals_conceded"] - df["xg_against"]

    for window in ROLLING_WINDOWS:
        df[f"xg_diff_roll_{window}"] = (
            df.groupby("team")["_xg_diff"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"xg_against_diff_roll_{window}"] = (
            df.groupby("team")["_xga_diff"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )

    # Cumulative season totals (shifted by 1 to exclude current match)
    for col, out_col in [
        ("xg_for", "xg_for_cumulative"),
        ("xg_against", "xg_against_cumulative"),
    ]:
        df[out_col] = (
            df.groupby(["team", "season"])[col]
            .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
        )

    df["goals_minus_xg_cumulative"] = (
        df.groupby(["team", "season"])["_xg_diff"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).sum())
    )
    df["goals_conceded_minus_xga_cumulative"] = (
        df.groupby(["team", "season"])["_xga_diff"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).sum())
    )

    # Cleanup temp columns
    df = df.drop(columns=["_xg_diff", "_xga_diff"])

    return df
