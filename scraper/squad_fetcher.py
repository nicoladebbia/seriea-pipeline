#!/usr/bin/env python3
"""Live squad roster fetcher using football APIs.

Fetches current Serie A squad rosters from:
1. API-Football (primary) — uses existing APIFOOTBALL_KEY
2. Football-Data.org (fallback) — uses FOOTBALLDATA_KEY
3. Minimal fallback from player_xg_profiles.json

Caches results to data/squads/current_squads.json (7-day TTL).

Environment variables:
    APIFOOTBALL_KEY   — API-Football key (free 100 req/day)
    FOOTBALLDATA_KEY  — Football-Data.org key (free registration)
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DATA_DIR

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from config.team_names import TEAM_NAME_MAP
from scraper.lineup_fetcher import (
    normalize_player_name,
    standardize_api_team_name,
)

log = logging.getLogger(__name__)

API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"
SERIE_A_LEAGUE_ID = 135
from config.settings import get_current_season
SERIE_A_SEASON = get_current_season()

SQUADS_DIR = DATA_DIR / "squads"
SQUADS_PATH = SQUADS_DIR / "current_squads.json"
CACHE_MAX_AGE_DAYS = 7

# API-Football position codes -> pipeline codes
APIFOOTBALL_POSITION_MAP = {
    "Goalkeeper": "GK",
    "Defender": "CB",
    "Midfielder": "CM",
    "Attacker": "ST",
}

# Football-Data.org position codes -> pipeline codes
FOOTBALLDATA_POSITION_MAP = {
    "Goalkeeper": "GK",
    "Defence": "CB",
    "Midfield": "CM",
    "Offence": "ST",
}


class SquadFetcher:
    """Fetches current squad rosters from football APIs."""

    def __init__(self):
        self.apifootball_key = os.environ.get("APIFOOTBALL_KEY", "")
        self.footballdata_key = os.environ.get("FOOTBALLDATA_KEY", "")

    def _is_cache_fresh(self) -> bool:
        """Check if cached squads file is less than 7 days old."""
        if not SQUADS_PATH.exists():
            return False
        try:
            with open(SQUADS_PATH) as f:
                data = json.load(f)
            fetched_at = data.get("fetched_at", "")
            if not fetched_at:
                return False
            dt = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
            age = datetime.now(timezone.utc) - dt
            return age.days < CACHE_MAX_AGE_DAYS
        except Exception:
            return False

    def _fetch_from_api_football(self) -> Optional[Dict]:
        """Fetch squads from API-Football (21 API calls).

        1 call for team list + 20 calls for squad per team.
        """
        if not self.apifootball_key or not HAS_REQUESTS:
            return None

        headers = {"x-apisports-key": self.apifootball_key}

        # Step 1: Get all Serie A teams
        # Note: free tier only supports up to season 2024 for /teams,
        # but /players/squads returns current rosters regardless
        try:
            resp = requests.get(
                f"{API_FOOTBALL_BASE}/teams",
                headers=headers,
                params={"league": SERIE_A_LEAGUE_ID, "season": int(SERIE_A_SEASON.split("-")[0])},
                timeout=15,
            )
            if resp.status_code == 429:
                log.warning("API-Football rate limit hit during team fetch")
                return None
            resp.raise_for_status()
            teams_data = resp.json()
        except requests.RequestException as e:
            log.warning(f"API-Football teams request failed: {e}")
            return None

        teams_response = teams_data.get("response", [])
        if not teams_response:
            errors = teams_data.get("errors", {})
            if errors:
                log.warning(f"API-Football errors: {errors}")
            return None

        log.info(f"Found {len(teams_response)} Serie A teams from API-Football")

        # Step 2: Fetch squad for each team
        result_teams = {}
        for team_entry in teams_response:
            team_info = team_entry.get("team", {})
            team_id = team_info.get("id")
            api_team_name = team_info.get("name", "")
            our_name = standardize_api_team_name(api_team_name)

            if not team_id:
                continue

            time.sleep(1.5)  # Rate limiting: free tier throttles fast requests

            try:
                resp = requests.get(
                    f"{API_FOOTBALL_BASE}/players/squads",
                    headers=headers,
                    params={"team": team_id},
                    timeout=15,
                )
                if resp.status_code == 429:
                    log.warning(f"API-Football rate limit hit fetching {our_name}")
                    continue
                resp.raise_for_status()
                squad_data = resp.json()
            except requests.RequestException as e:
                log.warning(f"API-Football squad request failed for {our_name}: {e}")
                continue

            squad_response = squad_data.get("response", [])
            if not squad_response:
                continue

            players = []
            for squad_entry in squad_response:
                for player in squad_entry.get("players", []):
                    pos_raw = player.get("position", "")
                    players.append({
                        "name": player.get("name", ""),
                        "position": APIFOOTBALL_POSITION_MAP.get(pos_raw, pos_raw),
                        "number": player.get("number"),
                        "age": player.get("age"),
                        "nationality": None,  # Not in squads endpoint
                    })

            if players:
                result_teams[our_name] = {"players": players}
                log.debug(f"  {our_name}: {len(players)} players")

        if not result_teams:
            return None

        return {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": "api_football",
            "season": SERIE_A_SEASON,
            "teams": result_teams,
        }

    def _fetch_from_football_data_org(self) -> Optional[Dict]:
        """Fetch squads from Football-Data.org (1 API call)."""
        if not self.footballdata_key or not HAS_REQUESTS:
            return None

        headers = {"X-Auth-Token": self.footballdata_key}

        try:
            resp = requests.get(
                f"{FOOTBALL_DATA_BASE}/competitions/SA/teams",
                headers=headers,
                timeout=15,
            )
            if resp.status_code == 429:
                log.warning("Football-Data.org rate limit hit")
                return None
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            log.warning(f"Football-Data.org request failed: {e}")
            return None

        teams_list = data.get("teams", [])
        if not teams_list:
            return None

        log.info(f"Found {len(teams_list)} Serie A teams from Football-Data.org")

        result_teams = {}
        for team_entry in teams_list:
            api_name = team_entry.get("shortName", team_entry.get("name", ""))
            our_name = standardize_api_team_name(api_name)

            players = []
            for player in team_entry.get("squad", []):
                pos_raw = player.get("position", "")
                dob = player.get("dateOfBirth", "")
                age = None
                if dob:
                    try:
                        born = datetime.strptime(dob, "%Y-%m-%d")
                        age = (datetime.now() - born).days // 365
                    except ValueError as e:
                        log.debug(f"Failed to parse player date of birth '{dob}': {e}")

                players.append({
                    "name": player.get("name", ""),
                    "position": FOOTBALLDATA_POSITION_MAP.get(pos_raw, pos_raw),
                    "number": player.get("shirtNumber"),
                    "age": age,
                    "nationality": player.get("nationality"),
                })

            if players:
                result_teams[our_name] = {"players": players}

        if not result_teams:
            return None

        return {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": "football_data_org",
            "season": SERIE_A_SEASON,
            "teams": result_teams,
        }

    def _generate_minimal_fallback(self) -> Dict:
        """Build minimal squad data from player_xg_profiles.json."""
        profiles_path = DATA_DIR / "features" / "player_xg_profiles.json"

        result_teams: Dict[str, Dict] = {}

        if profiles_path.exists():
            try:
                with open(profiles_path) as f:
                    profiles = json.load(f)

                for key, profile in profiles.items():
                    team = profile.get("team", "")
                    name = profile.get("player_name", "")
                    if not team or not name:
                        continue

                    if team not in result_teams:
                        result_teams[team] = {"players": []}

                    result_teams[team]["players"].append({
                        "name": name,
                        "position": "CM",  # Unknown from profiles
                        "number": None,
                        "age": None,
                        "nationality": None,
                    })

                log.info(f"Built fallback squads from profiles: {len(result_teams)} teams")
            except Exception as e:
                log.warning(f"Failed to load profiles for fallback: {e}")

        if not result_teams:
            log.warning("No squad data available from any source")

        return {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": "profiles_fallback",
            "season": SERIE_A_SEASON,
            "teams": result_teams,
        }

    def fetch_and_save_squads(self, force_refresh: bool = False) -> Dict:
        """Main entry point: fetch squads and save to JSON.

        Uses cache if fresh (< 7 days). Falls through API sources.

        Args:
            force_refresh: Bypass cache and refetch from API

        Returns:
            Squad data dict
        """
        # Check cache first
        if not force_refresh and self._is_cache_fresh():
            try:
                with open(SQUADS_PATH) as f:
                    data = json.load(f)
                teams_count = len(data.get("teams", {}))
                log.info(f"Using cached squad data ({teams_count} teams, source: {data.get('source')})")
                return data
            except Exception as e:
                log.warning(f"Failed to load cached squad data: {e}")

        # Try API-Football first
        log.info("Fetching squad data from API-Football...")
        data = self._fetch_from_api_football()

        # Try Football-Data.org as fallback
        if not data:
            log.info("Trying Football-Data.org...")
            data = self._fetch_from_football_data_org()

        # Minimal fallback from profiles
        if not data:
            log.info("Using minimal fallback from player profiles...")
            data = self._generate_minimal_fallback()

        # Merge with existing data (preserve teams not returned by API)
        SQUADS_DIR.mkdir(parents=True, exist_ok=True)
        existing = {}
        if SQUADS_PATH.exists():
            try:
                with open(SQUADS_PATH) as f:
                    existing = json.load(f).get("teams", {})
            except Exception:
                existing = {}

        new_teams = data.get("teams", {})
        # Keep existing teams that API didn't return (e.g. manual audit data)
        merged_teams = {**existing, **new_teams}
        data["teams"] = merged_teams

        from config.settings import atomic_write_json
        atomic_write_json(SQUADS_PATH, data)

        api_count = len(new_teams)
        kept_count = len(merged_teams) - api_count
        total_players = sum(len(t.get("players", [])) for t in merged_teams.values())
        log.info(f"Saved squad data: {len(merged_teams)} teams ({api_count} from API, "
                 f"{kept_count} preserved from existing), {total_players} players")

        return data


def fetch_and_save_squads(force_refresh: bool = False) -> Dict:
    """Module-level convenience function."""
    fetcher = SquadFetcher()
    return fetcher.fetch_and_save_squads(force_refresh=force_refresh)
