"""Scrape per-match per-player xG data from Understat.

Each Understat match page contains:
- shotsData: every shot with xG, player, minute, result, body part, situation
- rostersData: full lineups with positions
- match_info: teams, score, date

From shotsData, we aggregate per-player per-match xG, npxG, xA, shots, key_passes.

Usage:
    # Scrape all seasons (2017-2026)
    python3 -m scripts.scrape_understat_matches

    # Single season
    python3 -m scripts.scrape_understat_matches --season 2024-2025

    # Retry failed
    python3 -m scripts.scrape_understat_matches --retry-failed
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import DATA_DIR
from scraper.understat_scraper import SEASON_MAP
from scripts.utils.scraper_state import load_failed, save_failed

log = logging.getLogger(__name__)

UNDERSTAT_BASE = "https://understat.com"
RATE_LIMIT = 3  # seconds between requests
OUTPUT_DIR = DATA_DIR / "external" / "understat"
MATCH_CACHE_DIR = OUTPUT_DIR / "matches"
FAILED_LOG = OUTPUT_DIR / "failed_matches.json"

SCRAPE_SEASONS = [
    "2017-2018", "2018-2019", "2019-2020", "2020-2021",
    "2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026",
]


def _get_http_client():
    """Get HTTP client — try curl_cffi first, fall back to requests."""
    try:
        from curl_cffi import requests as cffi_requests
        log.info("Using curl_cffi for Understat")
        return cffi_requests, "cffi"
    except ImportError:
        import requests
        log.info("Using standard requests for Understat")
        return requests, "requests"


def _fetch_page(client, client_type: str, url: str, timeout: int = 30) -> str:
    """Fetch a page with rate limiting."""
    if client_type == "cffi":
        resp = client.get(url, impersonate="chrome", timeout=timeout)
    else:
        resp = client.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} for {url}")
    return resp.text


def _extract_json_var(html: str, var_name: str):
    """Extract JSON data from inline JavaScript variable."""
    pattern = rf"var {var_name}\s*=\s*JSON\.parse\('([^']+)'\)"
    match = re.search(pattern, html)
    if not match:
        return None
    try:
        raw = match.group(1)
        decoded = raw.encode("utf-8").decode("unicode_escape")
        return json.loads(decoded)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        log.warning("Failed to parse %s: %s", var_name, e)
        return None



def get_season_match_ids(season: str) -> list[dict]:
    """Get match IDs for a season from cached JSON or scrape league page."""
    us_season = SEASON_MAP.get(season)
    if not us_season:
        log.warning("Season %s not in Understat map", season)
        return []

    # Check cached JSON
    json_file = OUTPUT_DIR / f"understat_{season.replace('-', '_')}.json"
    if json_file.exists():
        with open(json_file) as f:
            data = json.load(f)
        dates = data.get("dates", [])
        played = [d for d in dates if d.get("isResult")]
        log.info("Loaded %d played matches for %s from cache", len(played), season)
        return played

    # Scrape league page
    log.info("Fetching Understat league page for %s...", season)
    client, ctype = _get_http_client()
    url = f"{UNDERSTAT_BASE}/league/Serie_A/{us_season}"
    html = _fetch_page(client, ctype, url, timeout=60)

    dates_data = _extract_json_var(html, "datesData")
    teams_data = _extract_json_var(html, "teamsData")
    players_data = _extract_json_var(html, "playersData")

    # Cache
    cache_data = {
        "season": season,
        "dates": dates_data or [],
        "teams": teams_data or {},
        "players": players_data or [],
    }
    json_file.parent.mkdir(parents=True, exist_ok=True)
    with open(json_file, "w") as f:
        json.dump(cache_data, f, indent=2)
    log.info("Cached league data for %s", season)

    played = [d for d in (dates_data or []) if d.get("isResult")]
    log.info("Found %d played matches for %s", len(played), season)
    return played


def scrape_match(client, client_type: str, match_id: str) -> dict | None:
    """Scrape a single match page for shot-level and roster data."""
    cache_file = MATCH_CACHE_DIR / f"{match_id}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    url = f"{UNDERSTAT_BASE}/match/{match_id}"
    html = _fetch_page(client, client_type, url, timeout=45)

    shots_data = _extract_json_var(html, "shotsData")
    rosters_data = _extract_json_var(html, "rostersData")
    match_info = _extract_json_var(html, "match_info")

    if not shots_data:
        log.warning("No shotsData for match %s", match_id)
        return None

    result = {
        "match_id": match_id,
        "shots": shots_data,
        "rosters": rosters_data,
        "match_info": match_info,
    }

    # Cache
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(result, f)

    return result


def aggregate_player_xg(match_data: dict, season: str, match_meta: dict) -> list[dict]:
    """Aggregate shot-level data into per-player per-match xG stats."""
    shots = match_data.get("shots", {})
    rosters = match_data.get("rosters", {})
    match_info = match_data.get("match_info", {})
    match_id = match_data.get("match_id", "")

    # Get team names
    home_team = match_meta.get("h", {}).get("title", "")
    away_team = match_meta.get("a", {}).get("title", "")
    match_date = match_meta.get("datetime", "")[:10]
    home_goals = match_meta.get("goals", {}).get("h", "")
    away_goals = match_meta.get("goals", {}).get("a", "")

    # Aggregate shots by player
    player_stats = defaultdict(lambda: {
        "xg": 0.0, "npxg": 0.0, "xa": 0.0, "shots": 0, "goals": 0,
        "np_goals": 0, "key_passes": 0, "assists": 0,
    })

    for side in ["h", "a"]:
        team_shots = shots.get(side, [])
        for shot in team_shots:
            pid = shot.get("player_id", "")
            player_name = shot.get("player", "")
            xg = float(shot.get("xG", 0))
            result = shot.get("result", "")
            situation = shot.get("situation", "")

            key = (pid, player_name, side)
            player_stats[key]["xg"] += xg
            player_stats[key]["shots"] += 1
            player_stats[key]["player_name"] = player_name
            player_stats[key]["player_id"] = pid
            player_stats[key]["side"] = side

            # npxG = xG excluding penalties
            if situation != "Penalty":
                player_stats[key]["npxg"] += xg

            if result == "Goal":
                player_stats[key]["goals"] += 1
                if situation != "Penalty":
                    player_stats[key]["np_goals"] += 1

            # xA comes from the assist player (lastAction field)
            # Shot xG attributed to assister as xA
            assist_player = shot.get("player_assisted", "")
            if assist_player:
                # Find the assister's key
                for k, v in player_stats.items():
                    if v.get("player_name") == assist_player and k[2] == side:
                        v["xa"] += xg
                        if result == "Goal":
                            v["assists"] += 1
                        v["key_passes"] += 1
                        break
                else:
                    # Assister not yet in stats, create entry
                    assist_key = ("", assist_player, side)
                    player_stats[assist_key]["xa"] += xg
                    player_stats[assist_key]["player_name"] = assist_player
                    player_stats[assist_key]["side"] = side
                    if result == "Goal":
                        player_stats[assist_key]["assists"] += 1
                    player_stats[assist_key]["key_passes"] += 1

    # Merge with roster data for players who had no shots
    for side in ["h", "a"]:
        roster = rosters.get(side, {})
        if isinstance(roster, dict):
            for pid, pdata in roster.items():
                key = (pid, pdata.get("player_name", ""), side)
                if key not in player_stats:
                    player_stats[key]["player_name"] = pdata.get("player_name", "")
                    player_stats[key]["player_id"] = pid
                    player_stats[key]["side"] = side
                # Add roster info
                player_stats[key]["position"] = pdata.get("position", "")
                player_stats[key]["minutes"] = int(pdata.get("time", 0))

    # Build output rows
    rows = []
    for (pid, pname, side), stats in player_stats.items():
        team = home_team if side == "h" else away_team
        opponent = away_team if side == "h" else home_team
        is_home = side == "h"

        rows.append({
            "season": season,
            "match_id": match_id,
            "date": match_date,
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "team": team,
            "opponent": opponent,
            "is_home": is_home,
            "player_id": stats.get("player_id", pid),
            "player_name": stats.get("player_name", pname),
            "position": stats.get("position", ""),
            "minutes": stats.get("minutes", 0),
            "xg": round(stats["xg"], 4),
            "npxg": round(stats["npxg"], 4),
            "xa": round(stats["xa"], 4),
            "shots": stats["shots"],
            "goals": stats["goals"],
            "np_goals": stats["np_goals"],
            "assists": stats["assists"],
            "key_passes": stats["key_passes"],
        })

    return rows


def scrape_season_matches(
    season: str,
    client,
    client_type: str,
    limit: int | None = None,
) -> pd.DataFrame:
    """Scrape all matches for a season."""
    matches = get_season_match_ids(season)
    if not matches:
        return pd.DataFrame()

    all_rows = []
    scraped = 0
    skipped = 0
    failed = 0
    failed_log = load_failed(FAILED_LOG)

    for i, match_meta in enumerate(matches):
        mid = str(match_meta["id"])

        if limit and scraped >= limit:
            break

        # Check if already cached
        cache_file = MATCH_CACHE_DIR / f"{mid}.json"
        is_cached = cache_file.exists()

        try:
            if not is_cached:
                time.sleep(RATE_LIMIT)

            match_data = scrape_match(client, client_type, mid)

            if match_data:
                rows = aggregate_player_xg(match_data, season, match_meta)
                all_rows.extend(rows)

                if is_cached:
                    skipped += 1
                else:
                    scraped += 1

                # Remove from failed log if previously failed
                if mid in failed_log:
                    del failed_log[mid]

                if (scraped + skipped) % 20 == 0 or scraped <= 3:
                    h = match_meta.get("h", {}).get("title", "?")
                    a = match_meta.get("a", {}).get("title", "?")
                    log.info(
                        "  [%s] %d scraped, %d cached, %d failed | %d/%d | %s vs %s",
                        season, scraped, skipped, failed, i + 1, len(matches), h, a,
                    )
            else:
                failed += 1
                failed_log[mid] = {
                    "season": season,
                    "error": "no_shots_data",
                    "match": f"{match_meta.get('h', {}).get('title', '?')} vs {match_meta.get('a', {}).get('title', '?')}",
                }

        except Exception as e:
            failed += 1
            log.error("Failed match %s: %s", mid, e)
            failed_log[mid] = {
                "season": season,
                "error": str(e),
                "match": f"{match_meta.get('h', {}).get('title', '?')} vs {match_meta.get('a', {}).get('title', '?')}",
            }

    save_failed(FAILED_LOG, failed_log)
    log.info(
        "  [%s] Done: %d scraped, %d cached, %d failed | %d players",
        season, scraped, skipped, failed, len(all_rows),
    )
    return pd.DataFrame(all_rows)


def main():
    parser = argparse.ArgumentParser(description="Scrape Understat per-match player xG data")
    parser.add_argument("--season", help="Single season (e.g. 2024-2025)")
    parser.add_argument("--limit", type=int, help="Max matches per season")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed matches")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    client, ctype = _get_http_client()
    seasons = [args.season] if args.season else SCRAPE_SEASONS

    all_dfs = []
    start = time.time()

    for season in seasons:
        log.info("=" * 60)
        log.info("UNDERSTAT: %s", season)
        log.info("=" * 60)

        df = scrape_season_matches(season, client, ctype, limit=args.limit)
        if not df.empty:
            all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        output_path = OUTPUT_DIR / "player_match_xg.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(output_path, index=False)
        log.info("Saved %d rows to %s", len(combined), output_path)

        # Summary
        log.info("\n=== SUMMARY ===")
        log.info("Total player-match records: %d", len(combined))
        log.info("Seasons: %s", sorted(combined["season"].unique()))
        log.info("Unique players: %d", combined["player_name"].nunique())
        log.info("Unique matches: %d", combined["match_id"].nunique())
    else:
        log.warning("No data collected")

    elapsed = time.time() - start
    log.info("Completed in %.0f min", elapsed / 60)


if __name__ == "__main__":
    main()
