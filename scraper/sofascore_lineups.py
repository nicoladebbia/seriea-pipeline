#!/usr/bin/env python3
"""Sofascore confirmed lineup fetcher.

Fetches confirmed starting XIs from Sofascore's /api/v1/event/{matchId}/lineups
endpoint, ~60 minutes before kickoff.  Uses the existing curl_cffi + Cloudflare
bypass infrastructure from sofascore_events.py.

No API key required — Sofascore's API is undocumented but freely accessible.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import DATA_DIR, UPCOMING_DIR
from config.team_names import normalize_team
from config.leagues import get_league_config, LEAGUE_REGISTRY

# Reuse HTTP infra from sofascore_events (session, retry, Cloudflare bypass)
from scraper.sofascore_events import _get_json, _jitter_delay, _BASE_URL

log = logging.getLogger(__name__)

SOFASCORE_DIR = _PROJECT_ROOT / "data" / "external" / "sofascore"

# Set of Sofascore tournament IDs we support (for filtering scheduled events)
_SUPPORTED_TOURNAMENT_IDS = {
    cfg.sofascore_tournament_id for cfg in LEAGUE_REGISTRY.values()
}

# Sofascore team name → our canonical name (handled by normalize_team, but
# a few edge cases need explicit mapping)
_SOFASCORE_NAME_MAP = {
    # Serie A
    "Hellas Verona": "Verona",
    "AC Milan": "Milan",
    "SSC Napoli": "Napoli",
    "SS Lazio": "Lazio",
    "ACF Fiorentina": "Fiorentina",
    "US Cremonese": "Cremonese",
    "AS Roma": "Roma",
    # Premier League
    "Brighton & Hove Albion": "Brighton",
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Wolverhampton": "Wolves",
    "Newcastle United": "Newcastle",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Leicester City": "Leicester",
    "Ipswich Town": "Ipswich",
}


def _normalize_sofascore_team(name: str) -> str:
    """Normalize a Sofascore team name to our canonical format."""
    if name in _SOFASCORE_NAME_MAP:
        return _SOFASCORE_NAME_MAP[name]
    return normalize_team(name)


def get_sofascore_match_ids(odds_data: Optional[Dict] = None) -> Dict[str, int]:
    """Map "Home vs Away" match keys → Sofascore event IDs.

    Strategy:
      1. Load fixture cache (data/external/sofascore/fixtures_YYYY_YYYY.json)
      2. Build normalized lookup: (home_norm, away_norm) → sofascore_id
      3. Match against odds_data keys (or predictions.json)
      4. Fallback: /api/v1/sport/football/scheduled-events/YYYY-MM-DD

    Returns:
        {"Roma vs Cremonese": 13981567, ...}
    """
    # Build match key targets from odds data
    target_matches = {}
    if odds_data:
        for match_key, match_info in odds_data.items():
            home = match_info.get("home_team", "")
            away = match_info.get("away_team", "")
            if not home or not away:
                parts = match_key.split(" vs ")
                if len(parts) == 2:
                    home, away = parts[0].strip(), parts[1].strip()
            if home and away:
                target_matches[match_key] = (normalize_team(home), normalize_team(away))

    if not target_matches:
        log.warning("No target matches provided for Sofascore ID resolution")
        return {}

    # Load fixture cache
    result = {}
    for cache_file in sorted(SOFASCORE_DIR.glob("fixtures_*.json"), reverse=True):
        try:
            with open(cache_file) as f:
                fixtures = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # Build lookup: (home_norm, away_norm) → sofascore_id
        fixture_map = {}
        for fx in fixtures:
            h = _normalize_sofascore_team(fx.get("homeTeam", {}).get("name", ""))
            a = _normalize_sofascore_team(fx.get("awayTeam", {}).get("name", ""))
            fid = fx.get("id")
            if h and a and fid:
                fixture_map[(h, a)] = fid

        # Match targets
        for match_key, (h_norm, a_norm) in target_matches.items():
            if match_key in result:
                continue
            fid = fixture_map.get((h_norm, a_norm))
            if fid:
                result[match_key] = fid

    # Fallback: API scheduled events for any unresolved matches
    unresolved = {k: v for k, v in target_matches.items() if k not in result}
    if unresolved:
        log.info("Resolving %d matches via Sofascore scheduled-events API", len(unresolved))
        # Collect unique dates from odds_data
        dates = set()
        for match_key in unresolved:
            ct = odds_data.get(match_key, {}).get("commence_time", "")
            if ct:
                try:
                    dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                    dates.add(dt.strftime("%Y-%m-%d"))
                except (ValueError, AttributeError):
                    pass

        for date_str in sorted(dates):
            _jitter_delay()
            url = f"{_BASE_URL}/sport/football/scheduled-events/{date_str}"
            data = _get_json(url)
            if not data:
                continue

            for event in data.get("events", []):
                tournament = event.get("tournament", {})
                # Filter to supported leagues
                tourn_id = tournament.get("uniqueTournament", {}).get("id")
                if tourn_id not in _SUPPORTED_TOURNAMENT_IDS:
                    continue

                h = _normalize_sofascore_team(event.get("homeTeam", {}).get("name", ""))
                a = _normalize_sofascore_team(event.get("awayTeam", {}).get("name", ""))
                eid = event.get("id")
                if not eid:
                    continue

                for match_key, (h_norm, a_norm) in unresolved.items():
                    if h == h_norm and a == a_norm and match_key not in result:
                        result[match_key] = eid
                        log.debug("Resolved %s → Sofascore ID %d via API", match_key, eid)

    resolved = len(result)
    total = len(target_matches)
    log.info("Sofascore match IDs: %d/%d resolved", resolved, total)
    if resolved < total:
        missing = [k for k in target_matches if k not in result]
        log.warning("Unresolved matches: %s", missing[:5])

    return result


def fetch_lineup(match_id: int) -> Optional[Dict]:
    """Fetch confirmed lineup from Sofascore for a specific match.

    Calls GET /api/v1/event/{match_id}/lineups

    Returns:
        Dict with home_team, away_team, home_lineup, away_lineup,
        home_formation, away_formation, lineup_source="confirmed"
        OR None if lineups not confirmed yet.
    """
    url = f"{_BASE_URL}/event/{match_id}/lineups"
    data = _get_json(url)
    if not data:
        return None

    # Check if lineups are confirmed
    if not data.get("confirmed", False):
        log.debug("Lineups for match %d not yet confirmed", match_id)
        return None

    result = {}
    for side, side_key in [("home", "home"), ("away", "away")]:
        side_data = data.get(side, {})
        formation = side_data.get("formation", "")

        starters = []
        bench = []
        for p in side_data.get("players", []):
            player_info = p.get("player", {})
            name = player_info.get("name", "")
            if not name:
                continue

            if p.get("substitute", False):
                bench.append(name)
            else:
                starters.append(name)

        result[f"{side_key}_lineup"] = starters
        result[f"{side_key}_bench"] = bench
        result[f"{side_key}_formation"] = formation

    # Need at least some starters on both sides
    if not result.get("home_lineup") or not result.get("away_lineup"):
        return None

    result["lineup_source"] = "confirmed"
    result["source_api"] = "sofascore"
    result["match_id_sofascore"] = match_id
    return result


def fetch_all_lineups(odds_data: Optional[Dict] = None) -> Dict:
    """Fetch confirmed lineups for all imminent Serie A matches.

    Main entry point for Sofascore lineup fetching.

    Args:
        odds_data: Dict from odds_full.json {match_key: {commence_time, ...}}
                   If None, loads from disk.

    Returns:
        Dict of {match_key: lineup_data} for matches with confirmed lineups.
    """
    # Load odds data if not provided
    if odds_data is None:
        odds_path = UPCOMING_DIR / "odds_full.json"
        if odds_path.exists():
            with open(odds_path) as f:
                full = json.load(f)
            odds_data = full.get("matches", {})
        else:
            log.warning("No odds_full.json found")
            return {}

    if not odds_data:
        return {}

    # Filter to imminent matches only (< 1h to kickoff)
    from scripts.utils.match_timing import classify_match_window

    imminent = {}
    for match_key, match_info in odds_data.items():
        ct = match_info.get("commence_time", "")
        window = classify_match_window(ct)
        if window in ("imminent", "approaching"):
            imminent[match_key] = match_info

    if not imminent:
        log.info("No imminent matches for Sofascore lineup fetch")
        return {}

    log.info("Fetching Sofascore lineups for %d imminent matches", len(imminent))

    # Get Sofascore match IDs
    match_ids = get_sofascore_match_ids(imminent)
    if not match_ids:
        log.warning("Could not resolve any Sofascore match IDs")
        return {}

    # Fetch lineups
    confirmed = {}
    for match_key, ss_id in match_ids.items():
        _jitter_delay()
        lineup = fetch_lineup(ss_id)
        if lineup:
            # Add team names from odds data
            match_info = imminent.get(match_key, {})
            home = match_info.get("home_team", "")
            away = match_info.get("away_team", "")
            if not home or not away:
                parts = match_key.split(" vs ")
                if len(parts) == 2:
                    home, away = parts[0].strip(), parts[1].strip()

            lineup["home_team"] = home
            lineup["away_team"] = away
            confirmed[match_key] = lineup
            log.info("Confirmed lineup: %s (%s vs %s)",
                     match_key, lineup["home_formation"], lineup["away_formation"])
        else:
            log.info("Lineups not yet available: %s", match_key)

    log.info("Sofascore: %d/%d confirmed lineups fetched",
             len(confirmed), len(match_ids))
    return confirmed


if __name__ == "__main__":
    """Quick test: resolve match IDs and try to fetch lineups."""
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Sofascore lineup fetcher")
    parser.add_argument("--match-id", type=int,
                        help="Fetch lineup for a specific Sofascore match ID")
    parser.add_argument("--resolve-ids", action="store_true",
                        help="Resolve Sofascore match IDs for upcoming matches")
    parser.add_argument("--fetch-all", action="store_true",
                        help="Fetch all imminent lineups (default)")
    args = parser.parse_args()

    if args.match_id:
        print(f"Fetching lineup for Sofascore match {args.match_id}...")
        result = fetch_lineup(args.match_id)
        if result:
            print(json.dumps(result, indent=2))
        else:
            print("Lineups not available (not confirmed or match not found)")
    elif args.resolve_ids:
        odds_path = UPCOMING_DIR / "odds_full.json"
        if odds_path.exists():
            with open(odds_path) as f:
                odds = json.load(f).get("matches", {})
            ids = get_sofascore_match_ids(odds)
            print(f"Resolved {len(ids)} match IDs:")
            for mk, mid in sorted(ids.items()):
                print(f"  {mk} → {mid}")
        else:
            print("No odds_full.json found")
    else:
        result = fetch_all_lineups()
        print(f"Confirmed lineups: {len(result)} matches")
        for mk, lineup in result.items():
            print(f"  {mk}: {lineup['home_formation']} vs {lineup['away_formation']}")
