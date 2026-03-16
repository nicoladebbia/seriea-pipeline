"""Parse goalkeeper stat tables from a FBref match report.

Table ID pattern: keeper_stats_{team_hash}
"""

from __future__ import annotations

import logging

import pandas as pd
from bs4 import BeautifulSoup

from parser.html_utils import find_table, table_to_dataframe

log = logging.getLogger(__name__)


def parse_goalkeeper_stats(
    soup: BeautifulSoup,
    team_hash: str,
    team_name: str,
    is_home: bool,
    match_id: str,
) -> list[dict]:
    """Parse the goalkeeper stats table for one team.

    Returns a list of dicts (one per GK who played).
    """
    table_id = f"keeper_stats_{team_hash}"
    table = find_table(soup, table_id)

    if table is None:
        log.debug("GK table %s not found", table_id)
        return []

    df = table_to_dataframe(table)

    if df.empty:
        return []

    # Remove total rows
    if "player" in df.columns:
        df = df[df["player"].str.strip().ne("").fillna(False)]

    df.insert(0, "match_id", match_id)
    df.insert(1, "team", team_name)
    df.insert(2, "is_home", is_home)

    return df.to_dict(orient="records")
