"""Parse the shots table from a FBref match report.

Table ID: shots_all (contains all shots from both teams).
Per-team tables also exist: shots_{team_hash}.
"""

from __future__ import annotations

import logging

from bs4 import BeautifulSoup

from config.team_names import normalize_team
from parser.html_utils import find_table, table_to_dataframe

log = logging.getLogger(__name__)


def parse_shots(
    soup: BeautifulSoup,
    match_id: str,
    home_team: str,
    away_team: str,
) -> list[dict]:
    """Parse the shots_all table.

    Returns a list of dicts with per-shot data including xG, outcome,
    body part, distance, and shot creation action chain.
    """
    table = find_table(soup, "shots_all")
    if table is None:
        log.debug("shots_all table not found for %s", match_id)
        return []

    df = table_to_dataframe(table)

    if df.empty:
        return []

    # Add match context
    df.insert(0, "match_id", match_id)

    # Normalize team names in the squad column
    if "squad" in df.columns:
        df["squad"] = df["squad"].apply(lambda x: normalize_team(x) if isinstance(x, str) else x)
        df["is_home_team_shot"] = df["squad"] == home_team

    return df.to_dict(orient="records")
