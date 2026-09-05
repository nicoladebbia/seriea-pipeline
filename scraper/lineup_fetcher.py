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

LINEUP_CHAIN_FILE = DATA_DIR / "upcoming" / "lineup_chain_status.json"


def lineup_chain_reason(chain: Dict[str, dict]) -> str:
    """One line, in Italian, on why no source delivered a team sheet — shown on
    Telegram next to 'formazioni non disponibili', so the reader knows whether
    to wait (a transient) or act (a missing key, a dead plan)."""
    parts = []
    ss = chain.get("sofascore") or {}
    st = ss.get("last_failure_status")
    if ss.get("error"):
        parts.append(f"Sofascore: errore ({ss['error'][:60]})")
    elif st:
        parts.append(f"Sofascore: HTTP {st}" + (" (challenge/ban)" if st == 403 else ""))
    elif not ss.get("n"):
        parts.append("Sofascore: nessuna formazione pubblicata")
    es = chain.get("espn") or {}
    if es.get("error"):
        parts.append(f"ESPN: errore ({es['error'][:60]})")
    elif not es.get("n"):
        parts.append("ESPN: XI non ancora pubblicata")
    fd = chain.get("football_data") or {}
    if fd.get("no_lineup_field"):
        parts.append("football-data.org: piano free senza formazioni")
    elif not fd.get("key_set"):
        parts.append("football-data.org: FOOTBALLDATA_KEY non impostata")
    elif fd.get("error"):
        parts.append(f"football-data.org: errore ({fd['error'][:60]})")
    elif not fd.get("n"):
        parts.append("football-data.org: nessuna formazione")
    af = chain.get("api_football") or {}
    if not af.get("key_set"):
        parts.append("API-Football: APIFOOTBALL_KEY non impostata")
    elif af.get("error"):
        parts.append(f"API-Football: {af['error'][:80]}")
    elif af.get("skipped"):
        parts.append(f"API-Football: saltata ({af['skipped']})")
    elif not af.get("n"):
        parts.append("API-Football: nessuna formazione")
    return " · ".join(parts) or "nessuna fonte ha risposto"

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
        self.last_error: Optional[dict] = None

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
                self.last_error = errors  # e.g. the free plan refusing the season
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


def fetch_and_save_lineups(odds_data: Dict = None,
                           deadline_sec: float = 150.0) -> Dict:
    """Main entry point: fetch lineups via multi-source cascade and save.

    Cascade order:
      1. Sofascore (primary — free, no key, uses curl_cffi infra)
      2. ESPN (free, no key — the backup that actually carries XIs; 2026-09-05)
      3. football-data.org (needs FOOTBALLDATA_KEY; the FREE tier has NO lineup field)
      4. API-Football (legacy — needs APIFOOTBALL_KEY, free plan only 2022-2024)

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

    # Load odds data if not provided — merge across all active leagues
    if odds_data is None:
        from config.leagues import ACTIVE_LEAGUES
        odds_data = {}
        for _league in ACTIVE_LEAGUES:
            fname = "odds_full.json" if _league == "serie_a" else f"odds_full_{_league}.json"
            odds_path = DATA_DIR / "upcoming" / fname
            if odds_path.exists():
                try:
                    with open(odds_path) as f:
                        full = json.load(f)
                    odds_data.update(full.get("matches", {}))
                    log.info("Loaded %d %s matches from %s",
                             len(full.get("matches", {})), _league, fname)
                except Exception as e:
                    log.warning("Failed to load %s: %s", fname, e)
        if not odds_data:
            log.warning("No odds_full*.json found across ACTIVE_LEAGUES")
            return {}

    confirmed = {}
    import time as _time
    _t0 = _time.monotonic()
    # Why each source did or did not deliver — written to LINEUP_CHAIN_FILE every
    # run so the scheduler (which runs this in a subprocess and captures its
    # output) can say WHY a match has no team sheet. On 2026-09-05 all three
    # sources were dead (Sofascore 403 challenge, no FOOTBALLDATA_KEY, API-Football
    # free plan without the season) and the only trace was "Lineup NOT confirmed".
    chain: Dict[str, dict] = {"sofascore": {}, "espn": {}, "football_data": {}, "api_football": {}}

    def _remaining() -> float:
        return max(0.0, deadline_sec - (_time.monotonic() - _t0)) if deadline_sec else 0.0

    # ── Source 1: Sofascore (primary) ──────────────────────────────────
    try:
        from scraper.sofascore_lineups import fetch_all_lineups as ss_fetch
        # Leave ~30s of the budget for the fallback sources
        ss_budget = max(30.0, deadline_sec - 30.0) if deadline_sec else 0.0
        ss_result = ss_fetch(odds_data, deadline_sec=ss_budget)
        if ss_result:
            confirmed.update(ss_result)
            log.info("Sofascore: %d confirmed lineups", len(ss_result))
        try:
            from scraper import sofascore_events as _ss_events
            chain["sofascore"] = {"n": len(ss_result or {}),
                                  "last_failure_status": getattr(_ss_events, "_LAST_FAILURE_STATUS", None)}
        except Exception:  # noqa: BLE001 - diagnostics only
            chain["sofascore"] = {"n": len(ss_result or {})}
    except Exception as e:
        log.warning("Sofascore lineup fetch failed: %s", e)
        chain["sofascore"] = {"n": 0, "error": str(e)[:200]}

    # ── Source 2: ESPN (key-free, the only backup that actually carries XIs) ──
    try:
        from scraper.espn_lineups import fetch_lineups_espn
        remaining_for_espn = {k: v for k, v in odds_data.items() if k not in confirmed}
        es_result = fetch_lineups_espn(remaining_for_espn, deadline_sec=_remaining()) if remaining_for_espn else {}
        chain["espn"] = {"n": len(es_result or {})}
        if es_result:
            confirmed.update(es_result)
            log.info("ESPN: added %d lineups", len(es_result))
    except Exception as e:
        log.warning("ESPN lineup fetch failed: %s", e)
        chain["espn"] = {"n": 0, "error": str(e)[:200]}

    # ── Source 3: football-data.org (backup for missed matches) ───────
    try:
        if deadline_sec and _remaining() < 10:
            raise TimeoutError(f"budget spent ({deadline_sec:.0f}s) — skipping backup source")
        from scraper.footballdata_lineups import fetch_lineups_footballdata as fd_fetch
        chain["football_data"] = {"key_set": bool(os.environ.get("FOOTBALLDATA_KEY"))}
        fd_result = fd_fetch(odds_data, deadline_sec=_remaining())
        chain["football_data"]["n"] = len(fd_result or {})
        from scraper import footballdata_lineups as _fd_mod
        if getattr(_fd_mod, "NO_LINEUP_FIELD_SEEN", False):
            chain["football_data"]["no_lineup_field"] = True
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
        chain["football_data"]["error"] = str(e)[:200]

    # ── Source 4: API-Football (legacy fallback) ──────────────────────
    # Only try if we still have missing matches, budget remains and the key exists
    fetcher = LineupFetcher()
    chain["api_football"] = {"key_set": fetcher.available}
    if deadline_sec and _remaining() < 10:
        log.warning("Lineup budget spent (%.0fs) — skipping API-Football legacy source",
                    deadline_sec)
        chain["api_football"]["skipped"] = "budget spent"
    elif fetcher.available:
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
                chain["api_football"]["n"] = len(af_result or {})
                if fetcher.last_error:
                    chain["api_football"]["error"] = str(fetcher.last_error)[:200]
            except Exception as e:
                log.warning("API-Football lineup fetch failed: %s", e)
                chain["api_football"]["error"] = str(e)[:200]

    # ── Chain status: every run, so a silent chain is impossible ─────
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "n_matches": len(odds_data),
              "confirmed": sorted(confirmed), "sources": chain,
              "reason": None if confirmed else lineup_chain_reason(chain)}
    try:
        LINEUP_CHAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
        LINEUP_CHAIN_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    except OSError as e:
        log.warning("Could not write %s: %s", LINEUP_CHAIN_FILE, e)
    if not confirmed:
        log.warning("No lineup source produced a team sheet for %d match(es) — %s",
                    len(odds_data), report["reason"])

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
