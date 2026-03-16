"""Fetch upcoming Serie A fixtures for prediction.

This module handles fetching and managing upcoming match fixtures
that haven't been played yet.

Sources:
- football-data.co.uk (primary)
- Manual entry as fallback
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import requests

from config.team_names import normalize_team
from storage.paths import PARSED_DIR

log = logging.getLogger(__name__)

UPCOMING_FIXTURES_PATH = PARSED_DIR / "upcoming_fixtures.parquet"

# Football-data.co.uk fixture URL
FOOTBALL_DATA_FIXTURES_URL = "https://www.football-data.co.uk/new/ITA.csv"


@dataclass
class UpcomingMatch:
    """An upcoming match without results."""
    match_id: str
    season: str
    matchweek: int
    match_date: date
    kickoff_time: str
    home_team: str
    away_team: str
    venue: Optional[str] = None


def _make_match_id(match_date: date, home: str, away: str) -> str:
    """Generate unique match ID."""
    date_str = match_date.isoformat()
    return f"{date_str}_{home}_{away}".replace(" ", "-")


def fetch_fixtures_from_football_data() -> pd.DataFrame:
    """Fetch latest fixtures from football-data.co.uk."""
    try:
        log.info("Fetching fixtures from football-data.co.uk")
        df = pd.read_csv(FOOTBALL_DATA_FIXTURES_URL)

        # Filter to Serie A only (the file contains Italian football)
        # The file has: Date, Time, HomeTeam, AwayTeam, FTHG, FTAG, etc.

        if df.empty:
            log.warning("No data from football-data.co.uk")
            return pd.DataFrame()

        # Rename columns to match our schema
        df = df.rename(columns={
            "Date": "match_date",
            "Time": "kickoff_time",
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "FTHG": "home_score",
            "FTAG": "away_score",
        })

        # Parse dates - try different formats
        def parse_date(d):
            if pd.isna(d):
                return None
            for fmt in ["%d/%m/%Y", "%Y-%m-%d", "%d/%m/%y"]:
                try:
                    return pd.to_datetime(d, format=fmt)
                except:
                    continue
            try:
                return pd.to_datetime(d)
            except:
                return None

        df["match_date"] = df["match_date"].apply(parse_date)
        df = df.dropna(subset=["match_date"])

        # Normalize team names
        df["home_team"] = df["home_team"].apply(normalize_team)
        df["away_team"] = df["away_team"].apply(normalize_team)

        # Filter to future matches (no score)
        upcoming = df[df["home_score"].isna() | (df["home_score"] == "")].copy()

        if upcoming.empty:
            log.info("No upcoming matches found in football-data.co.uk")
            # Still return all recent data for reference
            return df

        return upcoming

    except Exception as e:
        log.error(f"Failed to fetch from football-data.co.uk: {e}")
        return pd.DataFrame()


def get_upcoming_fixtures(
    days_ahead: int = 14,
    refresh: bool = False,
) -> list[UpcomingMatch]:
    """Get list of upcoming fixtures for the next N days.

    Args:
        days_ahead: Number of days to look ahead
        refresh: Force refresh from source

    Returns:
        List of UpcomingMatch objects
    """
    today = date.today()
    end_date = today + timedelta(days=days_ahead)

    # Determine current season
    year = today.year
    month = today.month
    if month >= 8:
        season = f"{year}-{year + 1}"
    else:
        season = f"{year - 1}-{year}"

    # Try to load cached fixtures
    if UPCOMING_FIXTURES_PATH.exists() and not refresh:
        df = pd.read_parquet(UPCOMING_FIXTURES_PATH)
        df["match_date"] = pd.to_datetime(df["match_date"]).dt.date

        # Filter to upcoming in date range
        upcoming = df[
            (df["match_date"] >= today) &
            (df["match_date"] <= end_date)
        ]

        if not upcoming.empty:
            return _df_to_matches(upcoming, season)

    # Fetch fresh data
    df = fetch_fixtures_from_football_data()

    if df.empty:
        # Return manually curated upcoming fixtures
        return _get_known_upcoming_fixtures(season, today, end_date)

    # Filter to date range
    df["match_date"] = pd.to_datetime(df["match_date"]).dt.date
    upcoming = df[
        (df["match_date"] >= today) &
        (df["match_date"] <= end_date)
    ]

    if not upcoming.empty:
        # Save for caching
        upcoming.to_parquet(UPCOMING_FIXTURES_PATH)
        return _df_to_matches(upcoming, season)

    return _get_known_upcoming_fixtures(season, today, end_date)


def _df_to_matches(df: pd.DataFrame, season: str) -> list[UpcomingMatch]:
    """Convert DataFrame to list of UpcomingMatch."""
    matches = []
    for _, row in df.iterrows():
        match_date = row["match_date"]
        if isinstance(match_date, datetime):
            match_date = match_date.date()

        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))

        if not home or not away:
            continue

        matches.append(UpcomingMatch(
            match_id=_make_match_id(match_date, home, away),
            season=season,
            matchweek=int(row.get("matchweek", 0)) if pd.notna(row.get("matchweek")) else 0,
            match_date=match_date,
            kickoff_time=str(row.get("kickoff_time", "")) if pd.notna(row.get("kickoff_time")) else "",
            home_team=home,
            away_team=away,
            venue=str(row.get("venue", "")) if pd.notna(row.get("venue")) else None,
        ))

    return sorted(matches, key=lambda m: m.match_date)


def _get_known_upcoming_fixtures(
    season: str,
    start_date: date,
    end_date: date,
) -> list[UpcomingMatch]:
    """Return known upcoming fixtures (manually curated as fallback).

    This is used when external sources are unavailable.
    Data sourced from FBref: https://fbref.com/en/comps/11/schedule/Serie-A-Scores-and-Fixtures
    Last updated: 2026-02-04
    """
    fixtures = [
        # Matchweek 23 postponed match
        UpcomingMatch(
            match_id="",
            season=season,
            matchweek=23,
            match_date=date(2026, 2, 3),
            kickoff_time="18:30",
            home_team="Bologna",
            away_team="Milan",
        ),
        # Matchweek 24 - Feb 6-9, 2026 (from FBref)
        UpcomingMatch(
            match_id="",
            season=season,
            matchweek=24,
            match_date=date(2026, 2, 6),
            kickoff_time="20:45",
            home_team="Hellas Verona",
            away_team="Pisa",
        ),
        UpcomingMatch(
            match_id="",
            season=season,
            matchweek=24,
            match_date=date(2026, 2, 7),
            kickoff_time="18:00",
            home_team="Genoa",
            away_team="Napoli",
        ),
        UpcomingMatch(
            match_id="",
            season=season,
            matchweek=24,
            match_date=date(2026, 2, 7),
            kickoff_time="20:45",
            home_team="Fiorentina",
            away_team="Torino",
        ),
        UpcomingMatch(
            match_id="",
            season=season,
            matchweek=24,
            match_date=date(2026, 2, 8),
            kickoff_time="12:30",
            home_team="Bologna",
            away_team="Parma",
        ),
        UpcomingMatch(
            match_id="",
            season=season,
            matchweek=24,
            match_date=date(2026, 2, 8),
            kickoff_time="15:00",
            home_team="Lecce",
            away_team="Udinese",
        ),
        UpcomingMatch(
            match_id="",
            season=season,
            matchweek=24,
            match_date=date(2026, 2, 8),
            kickoff_time="18:00",
            home_team="Sassuolo",
            away_team="Inter",
        ),
        UpcomingMatch(
            match_id="",
            season=season,
            matchweek=24,
            match_date=date(2026, 2, 8),
            kickoff_time="20:45",
            home_team="Juventus",
            away_team="Lazio",
        ),
        UpcomingMatch(
            match_id="",
            season=season,
            matchweek=24,
            match_date=date(2026, 2, 9),
            kickoff_time="18:30",
            home_team="Atalanta",
            away_team="Cremonese",
        ),
        UpcomingMatch(
            match_id="",
            season=season,
            matchweek=24,
            match_date=date(2026, 2, 9),
            kickoff_time="20:45",
            home_team="Roma",
            away_team="Cagliari",
        ),
        # Matchweek 25 - Feb 14-16, 2026 (from FBref)
        UpcomingMatch(
            match_id="",
            season=season,
            matchweek=25,
            match_date=date(2026, 2, 14),
            kickoff_time="20:45",
            home_team="Napoli",
            away_team="Fiorentina",
        ),
        UpcomingMatch(
            match_id="",
            season=season,
            matchweek=25,
            match_date=date(2026, 2, 15),
            kickoff_time="15:00",
            home_team="Inter",
            away_team="Bologna",
        ),
        UpcomingMatch(
            match_id="",
            season=season,
            matchweek=25,
            match_date=date(2026, 2, 15),
            kickoff_time="18:00",
            home_team="Torino",
            away_team="Juventus",
        ),
        UpcomingMatch(
            match_id="",
            season=season,
            matchweek=25,
            match_date=date(2026, 2, 15),
            kickoff_time="20:45",
            home_team="Milan",
            away_team="Roma",
        ),
        UpcomingMatch(
            match_id="",
            season=season,
            matchweek=25,
            match_date=date(2026, 2, 16),
            kickoff_time="12:30",
            home_team="Pisa",
            away_team="Atalanta",
        ),
        UpcomingMatch(
            match_id="",
            season=season,
            matchweek=25,
            match_date=date(2026, 2, 16),
            kickoff_time="15:00",
            home_team="Cagliari",
            away_team="Sassuolo",
        ),
        UpcomingMatch(
            match_id="",
            season=season,
            matchweek=25,
            match_date=date(2026, 2, 16),
            kickoff_time="15:00",
            home_team="Parma",
            away_team="Lecce",
        ),
        UpcomingMatch(
            match_id="",
            season=season,
            matchweek=25,
            match_date=date(2026, 2, 16),
            kickoff_time="18:00",
            home_team="Udinese",
            away_team="Genoa",
        ),
        UpcomingMatch(
            match_id="",
            season=season,
            matchweek=25,
            match_date=date(2026, 2, 16),
            kickoff_time="20:45",
            home_team="Cremonese",
            away_team="Hellas Verona",
        ),
        UpcomingMatch(
            match_id="",
            season=season,
            matchweek=25,
            match_date=date(2026, 2, 17),
            kickoff_time="20:45",
            home_team="Lazio",
            away_team="Monza",
        ),
    ]

    # Generate match IDs and filter by date
    result = []
    for m in fixtures:
        if start_date <= m.match_date <= end_date:
            m.match_id = _make_match_id(m.match_date, m.home_team, m.away_team)
            result.append(m)

    return sorted(result, key=lambda m: m.match_date)


def add_upcoming_fixture(
    home_team: str,
    away_team: str,
    match_date: date,
    matchweek: int = 0,
    kickoff_time: str = "",
) -> UpcomingMatch:
    """Manually add an upcoming fixture.

    Args:
        home_team: Home team name
        away_team: Away team name
        match_date: Date of the match
        matchweek: Matchweek number
        kickoff_time: Kickoff time string

    Returns:
        Created UpcomingMatch object
    """
    year = match_date.year
    month = match_date.month
    if month >= 8:
        season = f"{year}-{year + 1}"
    else:
        season = f"{year - 1}-{year}"

    home = normalize_team(home_team)
    away = normalize_team(away_team)

    match = UpcomingMatch(
        match_id=_make_match_id(match_date, home, away),
        season=season,
        matchweek=matchweek,
        match_date=match_date,
        kickoff_time=kickoff_time,
        home_team=home,
        away_team=away,
    )

    # Add to cached file
    if UPCOMING_FIXTURES_PATH.exists():
        df = pd.read_parquet(UPCOMING_FIXTURES_PATH)
    else:
        df = pd.DataFrame()

    new_row = {
        "match_id": match.match_id,
        "season": match.season,
        "matchweek": match.matchweek,
        "match_date": match.match_date,
        "kickoff_time": match.kickoff_time,
        "home_team": match.home_team,
        "away_team": match.away_team,
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_parquet(UPCOMING_FIXTURES_PATH)

    log.info(f"Added upcoming fixture: {home} vs {away} on {match_date}")
    return match


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing upcoming fixtures...")

    # Get upcoming fixtures
    fixtures = get_upcoming_fixtures(days_ahead=14, refresh=True)

    print(f"\nFound {len(fixtures)} upcoming fixtures:")
    for m in fixtures:
        print(f"  {m.match_date} - {m.home_team} vs {m.away_team} (MW{m.matchweek})")
