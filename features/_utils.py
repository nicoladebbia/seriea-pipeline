"""Shared utilities for feature engineering modules.

Provides team name normalisation, Sofascore→pipeline match-ID bridging,
and common merge helpers used across FBref, Sofascore, and other feature modules.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PMS_PATH = _PROJECT_ROOT / "data" / "external" / "sofascore" / "player_match_stats.parquet"

# Sofascore team name -> pipeline team name
SOFASCORE_TEAM_MAP: dict[str, str] = {
    "ac milan": "milan",
    "parma calcio 1913": "parma",
    "spal 2013": "spal",
    "hellas verona": "verona",
    "internazionale": "inter",
    "chievoverona": "chievo",
}


def _safe_div(a, b, default: float = 0.0):
    """Element-wise a/b, returning default where b is 0 or NaN."""
    if isinstance(a, (pd.Series, np.ndarray)):
        result = np.where((b == 0) | pd.isna(b), default, a / b)
        return pd.Series(result, index=a.index) if isinstance(a, pd.Series) else result
    if b == 0 or pd.isna(b):
        return default
    return a / b


def norm_team(name: str) -> str:
    """Normalise a Sofascore team name to the pipeline convention."""
    if pd.isna(name):
        return ""
    low = name.lower().strip()
    return SOFASCORE_TEAM_MAP.get(low, low).title()


def build_id_bridge() -> dict:
    """Build mapping: Sofascore int match_id -> pipeline string match_id.

    Pipeline string IDs have format 'YYYY-MM-DD_HomeTeam_AwayTeam'.
    Sofascore IDs are integers from player_match_stats.parquet.
    """
    if not _PMS_PATH.exists():
        return {}
    pms = pd.read_parquet(_PMS_PATH, columns=["match_id", "date", "home_team", "away_team"])
    pms = pms.drop_duplicates("match_id")
    bridge: dict = {}
    for _, row in pms.iterrows():
        home = norm_team(row["home_team"])
        away = norm_team(row["away_team"])
        bridge[row["match_id"]] = f"{row['date']}_{home}_{away}"
    log.debug("Sofascore ID bridge: %d mappings", len(bridge))
    return bridge


def merge_side(
    df_in: pd.DataFrame,
    team_col: str,
    prefix: str,
    lookup: pd.DataFrame,
    date_col: str,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Merge rolling stats for one side (home/away) using merge_asof.

    Parameters
    ----------
    df_in : DataFrame with ``_match_date`` and *team_col* columns.
    team_col : Column in *df_in* containing the team name (e.g. ``home_team_norm``).
    prefix : Prefix for newly created columns (e.g. ``"home"`` or ``"away"``).
    lookup : DataFrame of rolling stats with *date_col* and ``team_norm``.
    date_col : Name of the date column in *lookup* (renamed to ``_match_date`` internally).
    feature_cols : List of stat columns to pull from *lookup*.
    """
    side_df = df_in[["_match_date", team_col]].copy()
    side_df = side_df.rename(columns={team_col: "team_norm"})
    side_df = side_df.dropna(subset=["_match_date"])
    side_df = side_df.sort_values("_match_date").reset_index()

    right_df = lookup.copy()
    if date_col != "_match_date":
        right_df = right_df.rename(columns={date_col: "_match_date"})
    right_df = right_df.dropna(subset=["_match_date"])
    right_df = right_df.sort_values("_match_date").reset_index(drop=True)

    merged = pd.merge_asof(
        side_df,
        right_df,
        on="_match_date",
        by="team_norm",
        direction="backward",
    ).set_index("index")

    for c in feature_cols:
        if c in merged.columns:
            col_name = f"{prefix}_{c}"
            df_in.loc[merged.index, col_name] = merged[c].values

    return df_in
