"""Parse match metadata from the FBref scorebox."""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from typing import Optional

from bs4 import BeautifulSoup, Tag

from config.team_names import normalize_team
from models.schemas import MatchMetadata
from parser.html_utils import safe_int

log = logging.getLogger(__name__)


def parse_scorebox(soup: BeautifulSoup, match_id: str, season: str) -> MatchMetadata:
    """Extract match metadata from div.scorebox and div.scorebox_meta."""
    scorebox = soup.find("div", class_="scorebox")

    # Teams and scores
    home_team, away_team = "", ""
    home_score, away_score = None, None
    home_xg, away_xg = None, None

    home_manager, away_manager = "", ""
    home_captain, away_captain = "", ""

    if scorebox:
        # Scorebox has 5 direct children:
        #   [0] home team div, [1] away team div, [2] scorebox_meta,
        #   [3] home events, [4] away events
        team_divs = scorebox.find_all("div", recursive=False)
        if len(team_divs) >= 2:
            home_team = _extract_team_name(team_divs[0])
            away_team = _extract_team_name(team_divs[1])

            home_score = _extract_score(team_divs[0])
            away_score = _extract_score(team_divs[1])

            home_xg = _extract_xg(team_divs[0])
            away_xg = _extract_xg(team_divs[1])

            home_manager = _extract_datapoint(team_divs[0], "Manager")
            away_manager = _extract_datapoint(team_divs[1], "Manager")
            home_captain = _extract_datapoint(team_divs[0], "Captain")
            away_captain = _extract_datapoint(team_divs[1], "Captain")

    # Metadata from scorebox_meta
    match_date = None
    kickoff_time = ""
    matchweek = None
    venue = ""
    attendance = None
    referee = ""

    # scorebox_meta can be found as a direct child or by class
    meta_div = soup.find("div", class_="scorebox_meta")

    if meta_div:
        # Date from venuetime span
        venuetime = meta_div.find("span", class_="venuetime")
        if venuetime:
            data_venue = venuetime.get("data-venue-date")
            if data_venue:
                match_date = _parse_date(data_venue)
            kickoff_time = venuetime.get("data-venue-time", "")

        # Fallback date: from strong > a tag
        if match_date is None:
            for strong in meta_div.find_all("strong"):
                a_tag = strong.find("a")
                if a_tag:
                    text = a_tag.get_text(strip=True)
                    parsed = _parse_date(text)
                    if parsed:
                        match_date = parsed
                        break

        # Parse all <div> children of scorebox_meta for structured fields.
        # FBref layout uses div elements with text like:
        #   "Serie A (Matchweek 32)"
        #   "Attendance : 23,156"
        #   "Venue : Gewiss Stadium, Bergamo"
        #   "Officials : Maurizio Mariani (Referee) · ..."
        for div in meta_div.find_all("div", recursive=False):
            text = div.get_text(separator=" ", strip=True)
            if not text:
                continue

            # Matchweek
            mw = re.search(r"(?:Matchweek|Gameweek|Round)\s*(\d+)", text, re.I)
            if mw:
                matchweek = int(mw.group(1))

            # Attendance
            if re.search(r"Attendance", text, re.I):
                m = re.search(r"[\d,]+", text.split("Attendance")[-1])
                if m:
                    attendance = safe_int(m.group(0))

            # Venue
            if re.search(r"Venue", text, re.I):
                # Extract everything after "Venue" and optional colon
                v = re.sub(r"^.*Venue\s*:?\s*", "", text, flags=re.I).strip()
                if v:
                    venue = v

            # Referee / Officials
            if re.search(r"Official|Referee", text, re.I):
                # Extract first name, usually "Name (Referee)"
                ref_match = re.search(r"([A-Z][\w\s.'-]+?)\s*\(Referee\)", text)
                if ref_match:
                    referee = ref_match.group(1).strip()
                else:
                    # Fallback: text after "Officials :" up to first · or end
                    r = re.sub(r"^.*(?:Officials?|Referee)\s*:?\s*", "", text, flags=re.I)
                    referee = r.split("·")[0].strip()
                    # Remove trailing parenthetical
                    referee = re.sub(r"\s*\(.*?\)\s*$", "", referee).strip()

    # Fallback date from match_id
    if match_date is None:
        m = re.match(r"(\d{4}-\d{2}-\d{2})", match_id)
        if m:
            match_date = _parse_date(m.group(1))

    return MatchMetadata(
        match_id=match_id,
        season=season,
        match_date=match_date or date(1900, 1, 1),
        kickoff_time=kickoff_time,
        matchweek=matchweek,
        home_team=normalize_team(home_team),
        away_team=normalize_team(away_team),
        home_score=home_score,
        away_score=away_score,
        home_xg=home_xg,
        away_xg=away_xg,
        venue=venue,
        attendance=attendance,
        referee=referee,
        home_manager=home_manager,
        away_manager=away_manager,
        home_captain=home_captain,
        away_captain=away_captain,
    )


def _extract_datapoint(div: Tag, label: str) -> str:
    """Extract a datapoint value from a scorebox team div.

    FBref uses: <div class="datapoint"><strong>Manager</strong>: Name</div>
    """
    for dp in div.find_all("div", class_="datapoint"):
        strong = dp.find("strong")
        if strong and label.lower() in strong.get_text(strip=True).lower():
            # The value is the text after the <strong> tag
            # Could be plain text or wrapped in <a>
            a_tag = dp.find("a")
            if a_tag:
                return a_tag.get_text(strip=True)
            # Fall back to full text minus the label
            full = dp.get_text(strip=True)
            # Remove "Manager:" or "Captain:" prefix
            cleaned = re.sub(rf"^{label}\s*:?\s*", "", full, flags=re.I).strip()
            return cleaned
    return ""


def _extract_team_name(div: Tag) -> str:
    """Extract team name from a scorebox team div."""
    strong = div.find("strong")
    if strong:
        a = strong.find("a")
        if a:
            return a.get_text(strip=True)
        return strong.get_text(strip=True)
    for a in div.find_all("a"):
        href = a.get("href", "")
        if "/squads/" in href or "/clubs/" in href:
            return a.get_text(strip=True)
    return div.get_text(strip=True).split("\n")[0].strip()


def _extract_score(div: Tag) -> int | None:
    """Extract score from a scorebox team div."""
    score_div = div.find("div", class_="score")
    if score_div:
        return safe_int(score_div.get_text(strip=True))
    return None


def _extract_xg(div: Tag) -> float | None:
    """Extract xG from the score_xg div."""
    xg_div = div.find("div", class_="score_xg")
    if xg_div:
        try:
            return float(xg_div.get_text(strip=True))
        except (ValueError, TypeError):
            return None
    return None


def _parse_date(text: str) -> date | None:
    text = text.strip()
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d/%m/%Y",
                "%A %B %d, %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None
