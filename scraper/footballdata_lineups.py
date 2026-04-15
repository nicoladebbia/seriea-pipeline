#!/usr/bin/env python3
"""football-data.org confirmed lineup fetcher (backup source).

Free API with Serie A as Tier 1 (free forever, 10 req/min).
Requires FOOTBALLDATA_KEY environment variable.

Register for free at: https://www.football-data.org/client/register
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import DATA_DIR, UPCOMING_DIR
from config.team_names import normalize_team

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

log = logging.getLogger(__name__)

API_BASE = "https://api.football-data.org/v4"
SERIE_A_CODE = "SA"  # football-data.org competition code for Serie A

# football-data.org team names → our canonical names
_FD_NAME_MAP = {
    "AC Milan": "Milan",
    "SSC Napoli": "Napoli",
    "SS Lazio": "Lazio",
    "ACF Fiorentina": "Fiorentina",
    "Hellas Verona FC": "Verona",
    "AS Roma": "Roma",
    "FC Internazionale Milano": "Inter",
    "US Cremonese": "Cremonese",
    "Juventus FC": "Juventus",
    "Bologna FC 1909": "Bologna",
    "Torino FC": "Torino",
    "Genoa CFC": "Genoa",
    "Udinese Calcio": "Udinese",
    "US Lecce": "Lecce",
    "Cagliari Calcio": "Cagliari",
    "Parma Calcio 1913": "Parma",
    "Como 1907": "Como",
    "AC Pisa 1909": "Pisa",
    "US Sassuolo Calcio": "Sassuolo",
    "Atalanta BC": "Atalanta",
    "Empoli FC": "Empoli",
    "AC Monza": "Monza",
    "Venezia FC": "Venezia",
}


def _normalize_fd_team(name: str) -> str:
    """Normalize a football-data.org team name."""
    if name in _FD_NAME_MAP:
        return _FD_NAME_MAP[name]
    return normalize_team(name)


def _api_get(endpoint: str, params: Optional[dict] = None) -> Optional[dict]:
    """Make a football-data.org API GET request."""
    api_key = os.environ.get("FOOTBALLDATA_KEY", "")
    if not api_key:
        log.info("FOOTBALLDATA_KEY not set, skipping football-data.org")
        return None
    if not _HAS_REQUESTS:
        log.warning("requests library not available")
        return None

    url = f"{API_BASE}/{endpoint}"
    headers = {"X-Auth-Token": api_key}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code == 429:
            log.warning("football-data.org rate limit hit, waiting 60s")
            time.sleep(60)
            resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code == 403:
            log.warning("football-data.org: forbidden (check API key)")
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.warning("football-data.org request failed: %s", e)
        return None


def fetch_lineups_footballdata(odds_data: Optional[Dict] = None) -> Dict:
    """Fetch confirmed lineups from football-data.org.

    Backup source for when Sofascore fails.

    1. GET /v4/competitions/SA/matches?status=SCHEDULED to list upcoming
    2. GET /v4/matches/{id} for each match to get lineups
    3. Parse homeTeam.lineup/bench, awayTeam.lineup/bench, formation

    Args:
        odds_data: Dict from odds_full.json for filtering to imminent matches.
                   If None, loads from disk.

    Returns:
        Dict of {match_key: lineup_data} matching confirmed_lineups.json format.
    """
    # Load odds for timing filter
    if odds_data is None:
        odds_path = UPCOMING_DIR / "odds_full.json"
        if odds_path.exists():
            with open(odds_path) as f:
                full = json.load(f)
            odds_data = full.get("matches", {})
        else:
            odds_data = {}

    # Get scheduled Serie A matches
    data = _api_get(f"competitions/{SERIE_A_CODE}/matches", {"status": "SCHEDULED"})
    if not data:
        return {}

    matches = data.get("matches", [])
    if not matches:
        log.info("football-data.org: no scheduled Serie A matches")
        return {}

    log.info("football-data.org: %d scheduled matches found", len(matches))

    # Filter to imminent matches (using commence_time from odds if available)
    from scripts.utils.match_timing import classify_match_window

    # Build set of imminent match teams
    imminent_teams = set()
    for match_key, match_info in odds_data.items():
        ct = match_info.get("commence_time", "")
        window = classify_match_window(ct)
        if window in ("imminent", "approaching"):
            home = normalize_team(match_info.get("home_team", ""))
            away = normalize_team(match_info.get("away_team", ""))
            if home and away:
                imminent_teams.add((home, away))

    # If we have odds data but nothing is imminent, skip all detail fetches.
    # This avoids fetching 129 match details at 6.5s each (~14 min blocking).
    if odds_data and not imminent_teams:
        log.info("football-data.org: no imminent matches, skipping lineup detail fetch")
        return {}

    confirmed = {}
    for match in matches:
        fd_home = _normalize_fd_team(match.get("homeTeam", {}).get("name", ""))
        fd_away = _normalize_fd_team(match.get("awayTeam", {}).get("name", ""))
        match_id = match.get("id")

        if not fd_home or not fd_away or not match_id:
            continue

        # Skip if not imminent (when we have timing data)
        if imminent_teams and (fd_home, fd_away) not in imminent_teams:
            continue

        # Fetch full match detail with lineups
        time.sleep(6.5)  # 10 req/min = 6s between requests + margin
        detail = _api_get(f"matches/{match_id}")
        if not detail:
            continue

        # Parse lineups
        home_team_data = detail.get("homeTeam", {})
        away_team_data = detail.get("awayTeam", {})

        home_lineup = home_team_data.get("lineup", [])
        away_lineup = away_team_data.get("lineup", [])

        # Only proceed if lineups are populated
        if not home_lineup or not away_lineup:
            log.debug("No lineups yet for %s vs %s", fd_home, fd_away)
            continue

        match_key = f"{fd_home} vs {fd_away}"
        confirmed[match_key] = {
            "home_team": fd_home,
            "away_team": fd_away,
            "home_lineup": [p.get("name", "") for p in home_lineup],
            "away_lineup": [p.get("name", "") for p in away_lineup],
            "home_bench": [p.get("name", "") for p in (home_team_data.get("bench") or [])],
            "away_bench": [p.get("name", "") for p in (away_team_data.get("bench") or [])],
            "home_formation": home_team_data.get("formation", ""),
            "away_formation": away_team_data.get("formation", ""),
            "lineup_source": "confirmed",
            "source_api": "football-data.org",
        }
        log.info("Confirmed lineup (fd.org): %s (%s vs %s)",
                 match_key,
                 confirmed[match_key]["home_formation"],
                 confirmed[match_key]["away_formation"])

    log.info("football-data.org: %d confirmed lineups fetched", len(confirmed))
    return confirmed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        from dotenv import load_dotenv
        load_dotenv(_PROJECT_ROOT / ".env")
    except ImportError:
        pass

    result = fetch_lineups_footballdata()
    print(f"Confirmed lineups: {len(result)} matches")
    for mk, lineup in result.items():
        print(f"  {mk}: {lineup['home_formation']} vs {lineup['away_formation']}")
