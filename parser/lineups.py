"""Parse lineup / formation data from a FBref match report."""

from __future__ import annotations

import logging
import re

from bs4 import BeautifulSoup, Tag

from config.team_names import normalize_team
from models.schemas import LineupInfo

log = logging.getLogger(__name__)


def parse_lineups(soup: BeautifulSoup) -> tuple[LineupInfo | None, LineupInfo | None]:
    """Parse both team lineups from div.lineup elements.

    Returns (home_lineup, away_lineup). Each may be None if not found.
    """
    lineup_divs = soup.find_all("div", class_="lineup")

    if len(lineup_divs) < 2:
        log.debug("Found %d lineup divs (expected 2)", len(lineup_divs))
        if len(lineup_divs) == 0:
            return None, None
        return _parse_single_lineup(lineup_divs[0]), None

    return _parse_single_lineup(lineup_divs[0]), _parse_single_lineup(lineup_divs[1])


def _parse_single_lineup(div: Tag) -> LineupInfo:
    """Parse a single team's lineup div."""
    formation = None
    starters: list[dict[str, str]] = []
    bench: list[dict[str, str]] = []

    # Formation is usually in the first <th> or header
    header = div.find("th") or div.find("div", class_="header")
    if header:
        text = header.get_text(strip=True)
        # Extract formation pattern like "(4-3-3)" or "(3-5-2)"
        m = re.search(r"\((\d[\d\-]+\d)\)", text)
        if m:
            formation = m.group(1)

    # Players are in <tr> or <li> elements
    is_bench = False
    for tr in div.find_all("tr"):
        # Check if this is a bench header
        text = tr.get_text(strip=True).lower()
        if "bench" in text or "substitutes" in text or "panchina" in text:
            is_bench = True
            continue

        # Skip header rows
        if tr.find("th") and not tr.find("td"):
            continue

        player = _extract_player(tr)
        if player:
            if is_bench:
                bench.append(player)
            else:
                starters.append(player)

    # Fallback: look for <a> tags with /players/ in href
    if not starters:
        is_bench = False
        for el in div.find_all(["a", "td", "li"]):
            text = el.get_text(strip=True).lower()
            if "bench" in text or "substitutes" in text:
                is_bench = True
                continue

            if el.name == "a" and "/players/" in el.get("href", ""):
                player = {"name": el.get_text(strip=True), "shirt_number": ""}
                if is_bench:
                    bench.append(player)
                else:
                    starters.append(player)

    return LineupInfo(formation=formation, starters=starters, bench=bench)


def _extract_player(tr: Tag) -> dict[str, str] | None:
    """Extract player name and shirt number from a lineup row."""
    tds = tr.find_all(["td", "th"])
    name = ""
    number = ""

    for td in tds:
        a_tag = td.find("a")
        if a_tag and "/players/" in a_tag.get("href", ""):
            name = a_tag.get_text(strip=True)
        elif not name:
            text = td.get_text(strip=True)
            if re.fullmatch(r"\d{1,2}", text):
                number = text
            elif text and not re.fullmatch(r"\d+", text):
                name = text

    if name:
        return {"name": name, "shirt_number": number}
    return None


def lineups_to_records(
    home_lineup: LineupInfo | None,
    away_lineup: LineupInfo | None,
    match_id: str,
    home_team: str,
    away_team: str,
) -> list[dict]:
    """Convert LineupInfo objects to flat records for storage."""
    records: list[dict] = []

    for lineup, team, is_home in [
        (home_lineup, home_team, True),
        (away_lineup, away_team, False),
    ]:
        if lineup is None:
            continue

        for player in lineup.starters:
            records.append({
                "match_id": match_id,
                "team": team,
                "is_home": is_home,
                "formation": lineup.formation or "",
                "player_name": player["name"],
                "shirt_number": player.get("shirt_number", ""),
                "role": "starter",
            })

        for player in lineup.bench:
            records.append({
                "match_id": match_id,
                "team": team,
                "is_home": is_home,
                "formation": lineup.formation or "",
                "player_name": player["name"],
                "shirt_number": player.get("shirt_number", ""),
                "role": "bench",
            })

    return records
