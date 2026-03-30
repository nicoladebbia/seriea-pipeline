"""Scrape the Serie A fixture / results page for a given season."""

from __future__ import annotations

import logging
import re
from datetime import date, datetime

import pandas as pd
from bs4 import BeautifulSoup, Comment

from config.settings import FBREF_BASE_URL, SERIE_A_COMP_ID
from config.team_names import normalize_team
from scraper.client import FBrefClient
from storage.paths import fixtures_csv_path

log = logging.getLogger(__name__)

# FBref competition ID mapping
FBREF_COMP_IDS: dict[str, int] = {
    "serie_a": 11,
    "premier_league": 9,
    "la_liga": 12,
    "bundesliga": 20,
    "ligue_1": 13,
}

# FBref URL-friendly league names for the schedule path
FBREF_LEAGUE_NAMES: dict[str, str] = {
    "serie_a": "Serie-A",
    "premier_league": "Premier-League",
    "la_liga": "La-Liga",
    "bundesliga": "Bundesliga",
    "ligue_1": "Ligue-1",
}

# Column order for the output CSV / DataFrame
FIXTURE_COLUMNS = [
    "match_id",
    "season",
    "matchweek",
    "day_of_week",
    "match_date",
    "kickoff_time",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "home_xg",
    "away_xg",
    "attendance",
    "venue",
    "referee",
    "match_report_url",
]


def _fixture_url(season: str, league: str = "serie_a") -> str:
    """Build the FBref fixtures URL for a league season.

    Args:
        season: Season string (e.g., "2024-2025")
        league: League identifier (default: "serie_a").
                Valid values: serie_a, premier_league, la_liga, bundesliga, ligue_1
    """
    comp_id = FBREF_COMP_IDS.get(league)
    league_name = FBREF_LEAGUE_NAMES.get(league)
    if comp_id is None or league_name is None:
        raise ValueError(
            f"Unknown league '{league}'. Valid: {', '.join(FBREF_COMP_IDS)}"
        )
    return (
        f"{FBREF_BASE_URL}/en/comps/{comp_id}"
        f"/{season}/schedule/{season}-{league_name}-Scores-and-Fixtures"
    )


def _uncomment_tables(soup: BeautifulSoup) -> None:
    """FBref sometimes hides tables inside HTML comments. Unwrap them in-place."""
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "<table" in comment:
            fragment = BeautifulSoup(comment, "lxml")
            comment.replace_with(fragment)


def _safe_int(val: str | None) -> int | None:
    if val is None:
        return None
    try:
        return int(val.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def _safe_float(val: str | None) -> float | None:
    if val is None:
        return None
    try:
        return float(val.strip())
    except (ValueError, AttributeError):
        return None


def _parse_date(text: str, season: str) -> date | None:
    """Parse a date string from FBref with fallback formats."""
    text = text.strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    # Try dateutil as last resort
    try:
        from dateutil import parser as du_parser

        return du_parser.parse(text).date()
    except Exception:
        log.warning("Unparseable date: %r", text)
        return None


def _make_match_id(match_date: date | None, home: str, away: str) -> str:
    date_str = match_date.isoformat() if match_date else "unknown"
    return f"{date_str}_{home}_{away}".replace(" ", "-")


def scrape_season_fixtures(season: str, client: FBrefClient | None = None, league: str = "serie_a") -> pd.DataFrame:
    """Scrape the fixture page for *season* and return a DataFrame.

    Also saves the raw CSV to data/raw/fixtures/{season}.csv.

    Args:
        season: Season string (e.g., "2024-2025")
        client: Optional FBrefClient instance
        league: League identifier (default: "serie_a").
                Valid values: serie_a, premier_league, la_liga, bundesliga, ligue_1
    """
    if client is None:
        client = FBrefClient()

    url = _fixture_url(season, league=league)
    log.info("Fetching fixtures from %s", url)
    resp = client.get(url)
    soup = BeautifulSoup(resp.text, "lxml")
    _uncomment_tables(soup)

    # Find the main fixture table — usually id="sched_..._1" or the first large table
    table = (
        soup.find("table", id=re.compile(r"sched.*_1"))
        or soup.find("table", {"class": "stats_table"})
    )
    if table is None:
        log.error("No fixture table found for %s", season)
        return pd.DataFrame(columns=FIXTURE_COLUMNS)

    rows: list[dict] = []
    current_week: int | None = None

    for tr in table.find("tbody").find_all("tr"):
        # Skip spacer / section-header rows
        if tr.get("class") and "spacer" in tr.get("class", []):
            continue
        if tr.find("th", {"colspan": True}):
            continue

        cells: dict[str, object] = {}
        for td in tr.find_all(["th", "td"]):
            stat = td.get("data-stat")
            if stat:
                cells[stat] = td

        if not cells:
            continue

        # Matchweek
        wk_el = cells.get("gameweek") or cells.get("week_number")
        if wk_el and wk_el.get_text(strip=True):
            current_week = _safe_int(wk_el.get_text(strip=True))

        # Date
        date_el = cells.get("date")
        date_text = date_el.get_text(strip=True) if date_el else ""
        match_date = _parse_date(date_text, season) if date_text else None

        # Day of week
        dow_el = cells.get("dayofweek")
        day_of_week = dow_el.get_text(strip=True) if dow_el else ""

        # Kickoff time
        time_el = cells.get("start_time") or cells.get("time")
        kickoff_time = time_el.get_text(strip=True) if time_el else ""

        # Teams
        home_el = cells.get("home_team") or cells.get("squad_a")
        away_el = cells.get("away_team") or cells.get("squad_b")
        home_team = normalize_team(home_el.get_text(strip=True)) if home_el else ""
        away_team = normalize_team(away_el.get_text(strip=True)) if away_el else ""

        if not home_team or not away_team:
            continue

        # Score
        score_el = cells.get("score")
        home_score, away_score = None, None
        if score_el:
            score_text = score_el.get_text(strip=True)
            m = re.match(r"(\d+)\s*[–\-:]\s*(\d+)", score_text)
            if m:
                home_score, away_score = int(m.group(1)), int(m.group(2))

        # xG
        home_xg = _safe_float(
            (cells.get("home_xg") or cells.get("xg_a", None))
            and (cells.get("home_xg") or cells.get("xg_a")).get_text(strip=True)
        )
        away_xg = _safe_float(
            (cells.get("away_xg") or cells.get("xg_b", None))
            and (cells.get("away_xg") or cells.get("xg_b")).get_text(strip=True)
        )

        # Venue, attendance, referee
        venue_el = cells.get("venue")
        venue = venue_el.get_text(strip=True) if venue_el else ""

        att_el = cells.get("attendance")
        attendance = _safe_int(att_el.get_text(strip=True)) if att_el else None

        ref_el = cells.get("referee")
        referee = ref_el.get_text(strip=True) if ref_el else ""

        # Match report link
        report_el = cells.get("match_report")
        match_report_url = ""
        if report_el:
            a_tag = report_el.find("a")
            if a_tag and a_tag.get("href"):
                href = a_tag["href"]
                match_report_url = href if href.startswith("http") else FBREF_BASE_URL + href

        match_id = _make_match_id(match_date, home_team, away_team)

        rows.append(
            {
                "match_id": match_id,
                "season": season,
                "matchweek": current_week,
                "day_of_week": day_of_week,
                "match_date": pd.Timestamp(match_date) if match_date else pd.NaT,
                "kickoff_time": kickoff_time,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "home_xg": home_xg,
                "away_xg": away_xg,
                "attendance": attendance,
                "venue": venue,
                "referee": referee,
                "match_report_url": match_report_url,
            }
        )

    df = pd.DataFrame(rows, columns=FIXTURE_COLUMNS)
    csv_path = fixtures_csv_path(season)
    df.to_csv(csv_path, index=False)
    log.info("Saved %d fixtures to %s", len(df), csv_path)
    return df
