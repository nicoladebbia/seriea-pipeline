#!/usr/bin/env python3
"""Confirmed lineup fetcher — multi-source cascade.

Fetches real confirmed starting XIs ~60-90 minutes before kickoff.

Sources (tried in order):
  1. Sofascore (primary — free, no key, uses existing infra)
  2. football-data.org (backup — free key from football-data.org/client/register)
  3. API-Football (legacy — free tier only covers 2022-2024)

Environment variables:
  FOOTBALLDATA_KEY  — football-data.org API key (optional, backup)
  APIFOOTBALL_KEY   — API-Football key (optional, legacy fallback)
"""

import difflib
import json
import logging
import os
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DATA_DIR
from config.team_names import normalize_team as _canonical_normalize

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

log = logging.getLogger(__name__)

API_BASE = "https://v3.football.api-sports.io"
from config.settings import get_current_season
SERIE_A_LEAGUE_ID = 135  # API-Football Serie A ID
PREMIER_LEAGUE_LEAGUE_ID = 39  # API-Football Premier League ID
_API_FOOTBALL_LEAGUE_IDS = {
    "serie_a": SERIE_A_LEAGUE_ID,
    "premier_league": PREMIER_LEAGUE_LEAGUE_ID,
}
SERIE_A_SEASON = int(get_current_season().split("-")[0])  # e.g., 2025 from "2025-2026"

# Team name normalization is handled by config/team_names.py via
# _canonical_normalize (imported above).

# Player name match threshold for fuzzy matching
NAME_MATCH_THRESHOLD = 0.75


def normalize_player_name(name: str) -> str:
    """Normalize a player name for matching.

    Strips accents, lowercases, removes extra whitespace.
    Handles "L. Martinez" → "l martinez" style.
    """
    if not name:
        return ""
    # Decompose unicode and remove combining characters (accents)
    nfkd = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Lowercase, strip, collapse whitespace
    cleaned = " ".join(stripped.lower().split())
    # Remove periods from initials (e.g., "L. Martinez" → "l martinez")
    cleaned = cleaned.replace(".", "")
    # Replace hyphens with spaces (e.g., "Marc-Oliver" → "marc oliver")
    cleaned = cleaned.replace("-", " ")
    # Re-collapse whitespace after replacements
    cleaned = " ".join(cleaned.split())
    return cleaned


def standardize_api_team_name(api_name: str) -> str:
    """Convert API-Football team name to our internal canonical format.

    Uses the central normalize_team() from config/team_names.py which covers
    all known name variants across every data source.
    """
    return _canonical_normalize(api_name)


class LineupFetcher:
    """Fetches confirmed lineups from API-Football."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("APIFOOTBALL_KEY", "")
        self.headers = {"x-apisports-key": self.api_key}
        self._fixture_cache: Dict[str, int] = {}  # "date:home:away" → fixture_id
        self._requests_today = 0

    @property
    def available(self) -> bool:
        """Whether the API key is configured."""
        return bool(self.api_key)

    def _api_get(self, endpoint: str, params: dict = None) -> Optional[dict]:
        """Make an API-Football GET request."""
        if not self.available:
            return None
        if not HAS_REQUESTS:
            log.warning("requests library not available for lineup fetching")
            return None

        url = f"{API_BASE}/{endpoint}"
        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=15)
            self._requests_today += 1

            if resp.status_code == 429:
                log.warning("API-Football rate limit reached")
                return None
            resp.raise_for_status()

            data = resp.json()
            errors = data.get("errors", {})
            if errors:
                log.warning(f"API-Football error: {errors}")
                return None

            return data
        except requests.RequestException as e:
            log.warning(f"API-Football request failed: {e}")
            return None

    def find_fixture_id(
        self, home_team: str, away_team: str, date: str
    ) -> Optional[int]:
        """Find the API-Football fixture ID for a match.

        Args:
            home_team: Our internal team name (e.g., "Inter")
            away_team: Our internal team name (e.g., "Milan")
            date: Match date "YYYY-MM-DD"

        Returns:
            fixture_id or None
        """
        cache_key = f"{date}:{home_team}:{away_team}"
        if cache_key in self._fixture_cache:
            return self._fixture_cache[cache_key]

        from config.leagues import infer_league
        league_key = infer_league(home_team, away_team)
        api_league_id = _API_FOOTBALL_LEAGUE_IDS.get(league_key, SERIE_A_LEAGUE_ID)

        data = self._api_get("fixtures", {
            "league": api_league_id,
            "season": SERIE_A_SEASON,
            "date": date,
        })
        if not data:
            return None

        # Cache all fixtures for this date
        for fixture in data.get("response", []):
            f_home = standardize_api_team_name(
                fixture.get("teams", {}).get("home", {}).get("name", "")
            )
            f_away = standardize_api_team_name(
                fixture.get("teams", {}).get("away", {}).get("name", "")
            )
            f_id = fixture.get("fixture", {}).get("id")
            if f_home and f_away and f_id:
                self._fixture_cache[f"{date}:{f_home}:{f_away}"] = f_id

        return self._fixture_cache.get(cache_key)

    def fetch_confirmed_lineup(
        self, fixture_id: int
    ) -> Optional[Dict]:
        """Fetch confirmed lineup for a fixture.

        Returns:
            Dict with home_lineup, away_lineup, formations, or None if not available.
        """
        data = self._api_get("fixtures/lineups", {"fixture": fixture_id})
        if not data:
            return None

        response = data.get("response", [])
        if len(response) < 2:
            log.info(f"Lineups not yet available for fixture {fixture_id}")
            return None

        result = {}
        for team_data in response:
            team_name = standardize_api_team_name(
                team_data.get("team", {}).get("name", "")
            )
            formation = team_data.get("formation", "")
            starters = []
            for player in team_data.get("startXI", []):
                p = player.get("player", {})
                name = p.get("name", "")
                if name:
                    starters.append(name)

            if not starters:
                continue

            # Determine if home or away based on order (first = home)
            if "home_team" not in result:
                result["home_team"] = team_name
                result["home_lineup"] = starters
                result["home_formation"] = formation
            else:
                result["away_team"] = team_name
                result["away_lineup"] = starters
                result["away_formation"] = formation

        if "home_lineup" in result and "away_lineup" in result:
            result["fixture_id"] = fixture_id
            result["lineup_source"] = "confirmed"
            return result

        return None

    def match_lineup_to_database(
        self, api_names: List[str], team: str, player_db_names: List[str]
    ) -> List[str]:
        """Fuzzy-match API-Football player names to internal database names.

        Args:
            api_names: Player names from API-Football
            team: Team name (for logging)
            player_db_names: Known player names in our database

        Returns:
            List of matched database names (preserving order)
        """
        normalized_db = {normalize_player_name(n): n for n in player_db_names}
        matched = []

        for api_name in api_names:
            norm_api = normalize_player_name(api_name)

            # Exact normalized match
            if norm_api in normalized_db:
                matched.append(normalized_db[norm_api])
                continue

            # Fuzzy match
            best_ratio = 0.0
            best_match = None
            for norm_db, original_db in normalized_db.items():
                # Check both full name and last name
                ratio = difflib.SequenceMatcher(None, norm_api, norm_db).ratio()
                # Also try last-name matching for "L. Martinez" vs "Lautaro Martinez"
                api_last = norm_api.split()[-1] if norm_api.split() else ""
                db_last = norm_db.split()[-1] if norm_db.split() else ""
                last_ratio = difflib.SequenceMatcher(None, api_last, db_last).ratio()

                effective_ratio = max(ratio, last_ratio * 0.9)
                if effective_ratio > best_ratio:
                    best_ratio = effective_ratio
                    best_match = original_db

            if best_match and best_ratio >= NAME_MATCH_THRESHOLD:
                matched.append(best_match)
            else:
                # Use the API name as-is (player may not be in our database)
                matched.append(api_name)
                log.debug(
                    f"No DB match for {api_name} ({team}), "
                    f"best: {best_match} ({best_ratio:.2f})"
                )

        return matched

    def fetch_lineups_for_matches(
        self, matches: Dict, player_db=None
    ) -> Dict:
        """Fetch confirmed lineups for all imminent matches.

        Args:
            matches: Dict from odds_full.json with commence_time per match
            player_db: Optional PlayerXGDatabase for name matching

        Returns:
            Dict of match_key -> lineup data
        """
        if not self.available:
            log.info("APIFOOTBALL_KEY not set, skipping lineup fetch")
            return {}

        from scripts.utils.match_timing import classify_match_window

        confirmed = {}
        for match_key, match_data in matches.items():
            commence = match_data.get("commence_time", "")
            window = classify_match_window(commence)

            # Only fetch for imminent or approaching matches
            if window not in ("imminent", "approaching"):
                continue

            home = match_data.get("home_team", "")
            away = match_data.get("away_team", "")
            if not home or not away:
                parts = match_key.split(" vs ")
                if len(parts) == 2:
                    home, away = parts[0].strip(), parts[1].strip()

            # Extract date from commence_time
            try:
                dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
                date_str = dt.strftime("%Y-%m-%d")
            except (ValueError, AttributeError):
                continue

            # Find fixture
            fixture_id = self.find_fixture_id(home, away, date_str)
            if not fixture_id:
                log.info(f"No fixture ID found for {match_key} on {date_str}")
                continue

            # Fetch lineup
            lineup = self.fetch_confirmed_lineup(fixture_id)
            if not lineup:
                log.info(f"Lineups not yet available for {match_key}")
                continue

            # Match player names to database if available
            if player_db:
                db_names = list(player_db.team_profiles.get(home, {}).keys())
                if db_names:
                    lineup["home_lineup"] = self.match_lineup_to_database(
                        lineup["home_lineup"], home, db_names
                    )
                db_names = list(player_db.team_profiles.get(away, {}).keys())
                if db_names:
                    lineup["away_lineup"] = self.match_lineup_to_database(
                        lineup["away_lineup"], away, db_names
                    )

            confirmed[match_key] = lineup
            log.info(
                f"Confirmed lineup for {match_key}: "
                f"{lineup['home_formation']} vs {lineup['away_formation']}"
            )

        log.info(f"Fetched {len(confirmed)} confirmed lineup(s)")
        return confirmed


def fetch_and_save_lineups(odds_data: Dict = None) -> Dict:
    """Main entry point: fetch lineups via multi-source cascade and save.

    Cascade order:
      1. Sofascore (primary — free, no key, uses curl_cffi infra)
      2. football-data.org (backup — needs FOOTBALLDATA_KEY)
      3. API-Football (legacy — needs APIFOOTBALL_KEY, only 2022-2024)

    Args:
        odds_data: Odds data dict (matches with commence_time).
                   If None, loads from odds_full.json.

    Returns:
        Dict of confirmed lineups (merged from all sources)
    """
    # Ensure .env is loaded (critical when called standalone or from scheduler)
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent / ".env")
    except ImportError:
        pass

    # Load odds data if not provided
    if odds_data is None:
        odds_path = DATA_DIR / "upcoming" / "odds_full.json"
        if odds_path.exists():
            with open(odds_path) as f:
                full = json.load(f)
            odds_data = full.get("matches", {})
        else:
            log.warning("No odds_full.json found, cannot determine match times")
            return {}

    confirmed = {}

    # ── Source 1: Sofascore (primary) ──────────────────────────────────
    try:
        from scraper.sofascore_lineups import fetch_all_lineups as ss_fetch
        ss_result = ss_fetch(odds_data)
        if ss_result:
            confirmed.update(ss_result)
            log.info("Sofascore: %d confirmed lineups", len(ss_result))
    except Exception as e:
        log.warning("Sofascore lineup fetch failed: %s", e)

    # ── Source 2: football-data.org (backup for missed matches) ───────
    try:
        from scraper.footballdata_lineups import fetch_lineups_footballdata as fd_fetch
        fd_result = fd_fetch(odds_data)
        if fd_result:
            # Only add matches not already confirmed by Sofascore
            added = 0
            for match_key, lineup in fd_result.items():
                if match_key not in confirmed:
                    confirmed[match_key] = lineup
                    added += 1
            if added:
                log.info("football-data.org: added %d lineups (backup)", added)
    except Exception as e:
        log.warning("football-data.org lineup fetch failed: %s", e)

    # ── Source 3: API-Football (legacy fallback) ──────────────────────
    # Only try if we still have missing matches and the key is available
    fetcher = LineupFetcher()
    if fetcher.available:
        # Find matches not yet confirmed
        remaining = {k: v for k, v in odds_data.items() if k not in confirmed}
        if remaining:
            try:
                player_db = None
                try:
                    from features.player_xg_model import PlayerXGDatabase
                    db = PlayerXGDatabase()
                    profiles_path = DATA_DIR / "features" / "player_xg_profiles.json"
                    if profiles_path.exists():
                        db.load(profiles_path)
                        player_db = db
                except Exception:
                    pass

                af_result = fetcher.fetch_lineups_for_matches(remaining, player_db)
                if af_result:
                    for match_key, lineup in af_result.items():
                        if match_key not in confirmed:
                            confirmed[match_key] = lineup
                    log.info("API-Football: added %d lineups (legacy)", len(af_result))
            except Exception as e:
                log.warning("API-Football lineup fetch failed: %s", e)

    # ── Save merged results ───────────────────────────────────────────
    if confirmed:
        from config.leagues import infer_league
        output_dir = DATA_DIR / "upcoming"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "confirmed_lineups.json"

        # Tag each lineup with league
        for lineup in confirmed.values():
            if isinstance(lineup, dict) and "league" not in lineup:
                lineup["league"] = infer_league(
                    lineup.get("home_team"), lineup.get("away_team")
                )

        # Determine primary source for metadata
        sources = set()
        for lineup in confirmed.values():
            sources.add(lineup.get("source_api", "unknown"))

        with open(output_path, "w") as f:
            json.dump({
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "sources": sorted(sources),
                "matches": confirmed,
            }, f, indent=2)
        log.info("Saved %d confirmed lineups to %s (sources: %s)",
                 len(confirmed), output_path, ", ".join(sorted(sources)))

    return confirmed
