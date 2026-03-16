"""Parse all 6 player stat tables per team from a FBref match report.

Tables (per team):
  - summary      : stats_{hash}_summary
  - passing      : stats_{hash}_passing
  - passing_types: stats_{hash}_passing_types
  - defense      : stats_{hash}_defense
  - possession   : stats_{hash}_possession
  - misc         : stats_{hash}_misc
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup

from parser.html_utils import find_table, table_to_dataframe

log = logging.getLogger(__name__)

TABLE_TYPES = [
    "summary",
    "passing",
    "passing_types",
    "defense",
    "possession",
    "misc",
]


def parse_player_stats(
    soup: BeautifulSoup,
    team_hash: str,
    team_name: str,
    is_home: bool,
    match_id: str,
) -> list[dict]:
    """Parse and merge all 6 player stat tables for one team.

    Returns a list of dicts (one per player), with prefixed columns for
    each non-summary table.
    """
    merged: Optional[pd.DataFrame] = None

    for table_type in TABLE_TYPES:
        table_id = f"stats_{team_hash}_{table_type}"
        table = find_table(soup, table_id)

        if table is None:
            log.debug("Table %s not found", table_id)
            continue

        df = table_to_dataframe(table)

        if df.empty:
            continue

        # Remove total/summary rows (usually have "player" == team name or empty)
        if "player" in df.columns:
            df = df[df["player"].str.strip().ne("").fillna(False)]

        if table_type == "summary":
            merged = df.copy()
        else:
            if merged is None:
                log.warning("Summary table missing for %s; skipping %s", team_name, table_type)
                continue

            # Prefix non-key columns to avoid clashes
            prefix = table_type.replace("_types", "_types")
            key_cols = {"player", "shirtnumber", "nationality", "position", "age"}
            rename_map = {
                c: f"{prefix}_{c}" for c in df.columns if c not in key_cols
            }
            df = df.rename(columns=rename_map)

            # Merge on player (and shirtnumber if available for disambiguation)
            merge_on = ["player"]
            if "shirtnumber" in df.columns and "shirtnumber" in merged.columns:
                merge_on.append("shirtnumber")

            # Drop duplicate key cols from right side before merge
            drop_cols = [c for c in key_cols if c in df.columns and c not in merge_on]
            df = df.drop(columns=drop_cols, errors="ignore")

            merged = merged.merge(df, on=merge_on, how="left", suffixes=("", f"_{prefix}_dup"))

    if merged is None or merged.empty:
        return []

    # Add context columns
    merged.insert(0, "match_id", match_id)
    merged.insert(1, "team", team_name)
    merged.insert(2, "is_home", is_home)
    merged.insert(3, "data_source", "fbref_match")

    return merged.to_dict(orient="records")
