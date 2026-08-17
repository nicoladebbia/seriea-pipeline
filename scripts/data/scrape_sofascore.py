"""Scrape comprehensive per-match data from Sofascore via their API.

Three data layers per match:
1. **Player stats** (64→80 cols) — per-player per-match performance including xG, xA,
   passes, carries, tackles, duels, errors, GK stats, Sofascore normalized values
2. **Team stats** — match-level team aggregates: possession, corners, shots inside/outside
   box, final third entries, errors leading to goals, GK saves, duel %, etc.
   Broken down by period (ALL, 1ST, 2ND).
3. **Shotmap aggregates** — per-team shot distribution: shots inside box %, header %,
   avg shot xG, open play vs set piece, xG from counter vs buildup.

Uses the sofascore_wrapper library (Playwright-based).

Supports multiple leagues via --league flag:
    serie_a (default), premier_league, la_liga, bundesliga, ligue_1

Usage:
    # Scrape current season (Serie A default)
    python3 -m scripts.scrape_sofascore --season 2025-2026

    # Scrape EPL
    python3 -m scripts.scrape_sofascore --league premier_league --season 2024-2025

    # Scrape historical
    python3 -m scripts.scrape_sofascore --season 2023-2024

    # All seasons (slow!)
    python3 -m scripts.scrape_sofascore
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import DATA_DIR
from config.team_names import normalize_team
from config.leagues import get_league_config, LEAGUE_REGISTRY
from scripts.utils.scraper_state import load_failed, save_failed

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-league Sofascore season ID maps
# Discovered via league.get_seasons() API — only 2017+ included
# ---------------------------------------------------------------------------

LEAGUE_SEASON_MAPS: dict[str, dict[str, int]] = {
    "serie_a": {
        "2026-2027": 95836,
        "2025-2026": 76457,
        "2024-2025": 63515,
        "2023-2024": 52760,
        "2022-2023": 42415,
        "2021-2022": 37475,
        "2020-2021": 32523,
        "2019-2020": 24644,
        "2018-2019": 17932,
        "2017-2018": 13768,
    },
    "premier_league": {
        "2026-2027": 96668,
        "2025-2026": 76986,
        "2024-2025": 61627,
        "2023-2024": 52186,
        "2022-2023": 41886,
        "2021-2022": 37036,
        "2020-2021": 29415,
        "2019-2020": 23776,
        "2018-2019": 17359,
        "2017-2018": 13380,
    },
}

# Backward compat: default to Serie A
SEASON_MAP = LEAGUE_SEASON_MAPS["serie_a"]

SCRAPE_SEASONS = [
    "2017-2018", "2018-2019", "2019-2020", "2020-2021",
    "2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026",
    "2026-2027",
]

RATE_LIMIT = 2  # seconds between match requests

# ---------------------------------------------------------------------------
# League-aware output paths
# ---------------------------------------------------------------------------

def _league_output_dir(league_key: str) -> Path:
    """Return the output directory for a given league."""
    return DATA_DIR / "external" / "sofascore"

def _league_suffix(league_key: str) -> str:
    """Return a filename suffix for non-Serie A leagues (empty for serie_a)."""
    if league_key == "serie_a":
        return ""
    return f"_{league_key}"

def _get_output_paths(league_key: str) -> tuple[Path, Path, Path]:
    """Return (output_dir, match_cache_dir, failed_log) for a league."""
    base = _league_output_dir(league_key)
    suffix = _league_suffix(league_key)
    match_cache = base / f"matches{suffix}" if suffix else base / "matches"
    failed_log = base / f"failed_matches{suffix}.json" if suffix else base / "failed_matches.json"
    return base, match_cache, failed_log

# Default dirs (backward compat for module-level imports)
OUTPUT_DIR = DATA_DIR / "external" / "sofascore"
MATCH_CACHE_DIR = OUTPUT_DIR / "matches"
FAILED_LOG = OUTPUT_DIR / "failed_matches.json"

# Backward compat: Serie A league ID
SERIE_A_LEAGUE_ID = 23



async def get_season_fixtures(
    api,
    season: str,
    league_key: str = "serie_a",
    tournament_id: int | None = None,
    season_map: dict[str, int] | None = None,
    output_dir: Path | None = None,
    matchweeks: int = 38,
) -> list[dict]:
    """Get all played fixtures for a season via Sofascore API."""
    from sofascore_wrapper.league import League

    _season_map = season_map or LEAGUE_SEASON_MAPS.get(league_key, SEASON_MAP)
    _tournament_id = tournament_id or get_league_config(league_key).sofascore_tournament_id
    _output_dir = output_dir or OUTPUT_DIR
    _matchweeks = matchweeks

    ss_season_id = _season_map.get(season)
    if not ss_season_id:
        log.warning("Season %s not in Sofascore map for %s", season, league_key)
        return []

    # Check cache — league suffix in filename for non-Serie A
    suffix = _league_suffix(league_key)
    cache_file = _output_dir / f"fixtures_{season.replace('-', '_')}{suffix}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            fixtures = json.load(f)
        played = [f for f in fixtures if f.get("status", {}).get("type") == "finished"]
        log.info("Loaded %d played fixtures for %s [%s] from cache", len(played), season, league_key)
        return played

    league = League(api, _tournament_id)

    # Get all fixtures for this season by iterating rounds
    log.info("Fetching fixtures for %s [%s] (season_id=%d)...", season, league_key, ss_season_id)

    all_fixtures = []

    # Use the API directly to get fixtures per round
    for round_num in range(1, _matchweeks + 1):
        try:
            data = await api._get(
                f"/unique-tournament/{_tournament_id}/season/{ss_season_id}"
                f"/events/round/{round_num}"
            )
            events = data.get("events", [])
            all_fixtures.extend(events)
            await asyncio.sleep(0.5)  # Light rate limiting for fixture fetching
        except Exception as e:
            log.debug("Round %d: %s", round_num, e)
            break

    if not all_fixtures:
        # Fallback: try last/next fixtures
        log.warning("Round-based fetch returned 0. Trying last_fixtures...")
        try:
            last = await league.last_fixtures()
            if isinstance(last, list):
                all_fixtures.extend(last)
        except Exception as e:
            log.warning("last_fixtures failed: %s", e)

    # Guard against cache truncation: don't overwrite with fewer fixtures
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists() and all_fixtures:
        with open(cache_file) as f:
            old_fixtures = json.load(f)
        if len(all_fixtures) < len(old_fixtures) * 0.8:
            log.warning(
                "Sofascore returned %d fixtures vs %d cached — likely rate-limited. Keeping old cache.",
                len(all_fixtures), len(old_fixtures),
            )
            played = [f for f in old_fixtures if f.get("status", {}).get("type") == "finished"]
            return played

    if all_fixtures:
        tmp_path = cache_file.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(all_fixtures, f, indent=1)
        tmp_path.rename(cache_file)

    played = [f for f in all_fixtures if f.get("status", {}).get("type") == "finished"]
    log.info("Found %d played fixtures (of %d total) for %s", len(played), len(all_fixtures), season)
    return played


async def refresh_fixtures_cache(
    api,
    season: str,
    league_key: str = "serie_a",
) -> list[dict]:
    """Force-refresh the fixtures cache for a season, bypassing the cache check.

    Used by matchday_updater to ensure we have the latest fixture data
    (e.g., after a matchday finishes and status changes to "finished").

    Returns all fixtures (not just played ones).
    """
    _season_map = LEAGUE_SEASON_MAPS.get(league_key, SEASON_MAP)
    _tournament_id = get_league_config(league_key).sofascore_tournament_id
    _matchweeks = get_league_config(league_key).matchweeks_per_season

    ss_season_id = _season_map.get(season)
    if not ss_season_id:
        log.warning("Season %s not in Sofascore map for %s", season, league_key)
        return []

    log.info("Force-refreshing fixtures cache for %s [%s] (season_id=%d)...", season, league_key, ss_season_id)

    all_fixtures = []
    for round_num in range(1, _matchweeks + 1):
        try:
            data = await api._get(
                f"/unique-tournament/{_tournament_id}/season/{ss_season_id}"
                f"/events/round/{round_num}"
            )
            events = data.get("events", [])
            all_fixtures.extend(events)
            await asyncio.sleep(0.5)
        except Exception as e:
            log.debug("Round %d: %s", round_num, e)
            break

    if all_fixtures:
        suffix = _league_suffix(league_key)
        cache_file = OUTPUT_DIR / f"fixtures_{season.replace('-', '_')}{suffix}.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(all_fixtures, f, indent=1)
        log.info("Refreshed fixtures cache: %d total fixtures for %s [%s]", len(all_fixtures), season, league_key)

    return all_fixtures


async def scrape_match_stats(
    api,
    match_id: int,
    season: str,
    match_cache_dir: Path | None = None,
) -> dict | None:
    """Scrape player stats, team stats, and shotmap for a single match."""
    from sofascore_wrapper.match import Match

    _cache_dir = match_cache_dir or MATCH_CACHE_DIR
    cache_file = _cache_dir / season / f"{match_id}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            cached = json.load(f)
        # If cached file lacks team_stats, fetch just that (backfill)
        if "team_stats" not in cached:
            match = Match(api, match_id)
            try:
                team_stats = await match.stats()
                cached["team_stats"] = team_stats
                with open(cache_file, "w") as f:
                    json.dump(cached, f)
                log.debug("Backfilled team_stats for match %d", match_id)
            except Exception as e:
                log.debug("Could not backfill team_stats for %d: %s", match_id, e)
                cached["team_stats"] = {}
        return cached

    match = Match(api, match_id)

    try:
        home_lineup = await match.lineups_home()
        away_lineup = await match.lineups_away()

        result = {
            "match_id": match_id,
            "home_lineup": home_lineup,
            "away_lineup": away_lineup,
        }

        # Team-level match statistics (possession, corners, final third entries, etc.)
        try:
            team_stats = await match.stats()
            result["team_stats"] = team_stats
        except Exception:
            result["team_stats"] = {}

        # Shotmap (per-shot xG, coordinates, body part, situation)
        try:
            shotmap = await match.shotmap()
            result["shotmap"] = shotmap
        except Exception:
            result["shotmap"] = []

        # Cache
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(result, f)

        return result

    except Exception as e:
        log.debug("Failed to get lineups for match %d: %s", match_id, e)
        return None


def extract_player_rows(match_data: dict, fixture: dict, season: str) -> list[dict]:
    """Extract per-player stat rows from match data."""
    match_id = match_data.get("match_id", "")
    home_team = normalize_team(fixture.get("homeTeam", {}).get("name", ""))
    away_team = normalize_team(fixture.get("awayTeam", {}).get("name", ""))
    home_score = fixture.get("homeScore", {}).get("current", "")
    away_score = fixture.get("awayScore", {}).get("current", "")

    # Extract date from fixture
    start_ts = fixture.get("startTimestamp", 0)
    if start_ts:
        import datetime
        match_date = datetime.datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d")
    else:
        match_date = ""

    round_info = fixture.get("roundInfo", {}).get("round", "")
    round_info = int(round_info) if round_info else None

    rows = []

    for side, lineup_key, team, opponent, is_home in [
        ("home", "home_lineup", home_team, away_team, True),
        ("away", "away_lineup", away_team, home_team, False),
    ]:
        lineup = match_data.get(lineup_key, {})
        if not isinstance(lineup, dict):
            continue

        all_players = lineup.get("starters", []) + lineup.get("substitutes", [])

        for p in all_players:
            if not isinstance(p, dict):
                continue

            player_info = p.get("player", {})
            stats = p.get("statistics", {})

            if not isinstance(stats, dict):
                continue

            # Skip players with no minutes (unused substitutes)
            minutes = stats.get("minutesPlayed", 0)
            if minutes == 0:
                continue

            row = {
                "season": season,
                "match_id": match_id,
                "date": match_date,
                "round": round_info,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "team": team,
                "opponent": opponent,
                "is_home": is_home,
                "player_id": player_info.get("id", ""),
                "player_name": player_info.get("name", ""),
                "position": p.get("position", ""),
                "shirt_number": p.get("shirtNumber", ""),
                "is_starter": p in lineup.get("starters", []),
                # Core stats
                "minutes": minutes,
                "rating": stats.get("rating", None),
                # Shooting
                "xg": stats.get("expectedGoals", None),
                "xgot": stats.get("expectedGoalsOnTarget", None),
                "goals": stats.get("goals", 0),
                "total_shots": stats.get("totalShots", 0),
                "shots_on_target": stats.get("onTargetScoringAttempt", 0),
                "shots_blocked": stats.get("blockedScoringAttempt", 0),
                "big_chances_created": stats.get("bigChanceCreated", 0),
                "big_chances_missed": stats.get("bigChanceMissed", 0),
                # Passing
                "xa": stats.get("expectedAssists", None),
                "assists": stats.get("goalAssist", 0),
                "key_passes": stats.get("keyPass", 0),
                "accurate_passes": stats.get("accuratePass", 0),
                "total_passes": stats.get("totalPass", 0),
                "accurate_long_balls": stats.get("accurateLongBalls", 0),
                "total_long_balls": stats.get("totalLongBalls", 0),
                "accurate_crosses": stats.get("accurateCross", 0),
                "total_crosses": stats.get("totalCross", 0),
                "opp_half_passes": stats.get("accurateOppositionHalfPasses", 0),
                "own_half_passes": stats.get("accurateOwnHalfPasses", 0),
                # Carrying
                "carries": stats.get("ballCarriesCount", 0),
                "carry_distance": stats.get("totalBallCarriesDistance", None),
                "progressive_carries": stats.get("progressiveBallCarriesCount", 0),
                "progressive_carry_distance": stats.get("totalProgressiveBallCarriesDistance", None),
                # Defense
                "tackles": stats.get("totalTackle", 0),
                "tackles_won": stats.get("wonTackle", 0),
                "interceptions": stats.get("interceptionWon", 0),
                "clearances": stats.get("totalClearance", 0),
                "ball_recoveries": stats.get("ballRecovery", 0),
                "blocks": stats.get("outfielderBlock", 0),
                "last_man_tackle": stats.get("lastManTackle", 0),
                # Duels
                "duels_won": stats.get("duelWon", 0),
                "duels_lost": stats.get("duelLost", 0),
                "aerial_won": stats.get("aerialWon", 0),
                "aerial_lost": stats.get("aerialLost", 0),
                # Other
                "touches": stats.get("touches", 0),
                "possession_lost": stats.get("possessionLostCtrl", 0),
                "fouls": stats.get("fouls", 0),
                "was_fouled": stats.get("wasFouled", 0),
                "dispossessed": stats.get("dispossessed", 0),
                "total_progression": stats.get("totalProgression", None),
                # Errors
                "error_to_goal": stats.get("errorLeadToAGoal", 0),
                "error_to_shot": stats.get("errorLeadToAShot", 0),
                # Dribble contests
                "contest_won": stats.get("wonContest", 0),
                "contest_total": stats.get("totalContest", 0),
                "challenge_lost": stats.get("challengeLost", 0),
                # Other
                "shots_off_target": stats.get("shotOffTarget", 0),
                "unsuccessful_touch": stats.get("unsuccessfulTouch", 0),
                "offsides": stats.get("totalOffside", 0),
                # Pass breakdown (total opp/own half — we already have accurate)
                "total_opp_half_passes": stats.get("totalOppositionHalfPasses", 0),
                "total_own_half_passes": stats.get("totalOwnHalfPasses", 0),
                # GK specific (will be None for outfield)
                "goals_prevented": stats.get("goalsPrevented", None),
                "saves": stats.get("saves", None),
                "saved_shots_from_inside_box": stats.get("savedShotsFromInsideTheBox", None),
                "keeper_sweeper_accurate": stats.get("accurateKeeperSweeper", None),
                "keeper_sweeper_total": stats.get("totalKeeperSweeper", None),
                "keeper_save_value": stats.get("keeperSaveValue", None),
                "keeper_high_claim": stats.get("goodHighClaim", None),
                "keeper_value": stats.get("goalkeeperValueNormalized", None),
                # Normalized values (Sofascore's internal model)
                "pass_value": stats.get("passValueNormalized", None),
                "shot_value": stats.get("shotValueNormalized", None),
                "dribble_value": stats.get("dribbleValueNormalized", None),
                "defensive_value": stats.get("defensiveValueNormalized", None),
            }

            rows.append(row)

    return rows


def extract_team_stats_rows(match_data: dict, fixture: dict, season: str) -> list[dict]:
    """Extract team-level match statistics (possession, corners, etc.) from match.stats().

    Returns two rows per match (one home, one away) for each period (ALL, 1ST, 2ND).
    """
    match_id = match_data.get("match_id", "")
    home_team = normalize_team(fixture.get("homeTeam", {}).get("name", ""))
    away_team = normalize_team(fixture.get("awayTeam", {}).get("name", ""))
    home_score = fixture.get("homeScore", {}).get("current", "")
    away_score = fixture.get("awayScore", {}).get("current", "")

    start_ts = fixture.get("startTimestamp", 0)
    if start_ts:
        import datetime
        match_date = datetime.datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d")
    else:
        match_date = ""

    round_info = fixture.get("roundInfo", {}).get("round", "")
    round_info = int(round_info) if round_info else None

    team_stats_raw = match_data.get("team_stats", {})
    if not team_stats_raw:
        return []

    periods = team_stats_raw.get("statistics", [])
    if not periods:
        return []

    rows = []

    for period_data in periods:
        period = period_data.get("period", "ALL")

        # Flatten all stat items from all groups
        flat_home = {}
        flat_away = {}
        for group in period_data.get("groups", []):
            for item in group.get("statisticsItems", []):
                key = item.get("key", "")
                if not key:
                    continue
                flat_home[key] = item.get("homeValue", 0)
                flat_away[key] = item.get("awayValue", 0)

        if not flat_home:
            continue

        base = {
            "season": season,
            "match_id": match_id,
            "date": match_date,
            "round": round_info,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "period": period,
        }

        def _build_row(team, opponent, is_home, flat):
            return {
                **base,
                "team": team,
                "opponent": opponent,
                "is_home": is_home,
                # Match overview
                "possession": flat.get("ballPossession", None),
                "total_shots": flat.get("totalShotsOnGoal", 0),
                "shots_on_target": flat.get("shotsOnGoal", 0),
                "shots_off_target": flat.get("shotsOffGoal", 0),
                "shots_inside_box": flat.get("totalShotsInsideBox", 0),
                "shots_outside_box": flat.get("totalShotsOutsideBox", 0),
                "hit_woodwork": flat.get("hitWoodwork", 0),
                "blocked_shots": flat.get("blockedScoringAttempt", 0),
                "corners": flat.get("cornerKicks", 0),
                "free_kicks": flat.get("freeKicks", 0),
                "throw_ins": flat.get("throwIns", 0),
                "offsides": flat.get("offsides", 0),
                "fouls": flat.get("fouls", 0),
                # Attack
                "big_chances_scored": flat.get("bigChanceScored", 0),
                "big_chances_created": flat.get("bigChanceCreated", 0),
                "big_chances_missed": flat.get("bigChanceMissed", 0),
                "touches_in_opp_box": flat.get("touchesInOppBox", 0),
                "fouled_final_third": flat.get("fouledFinalThird", 0),
                "final_third_entries": flat.get("finalThirdEntries", 0),
                "final_third_phases": flat.get("finalThirdPhaseStatistic", 0),
                # Passing
                "accurate_passes": flat.get("accuratePasses", 0),
                "total_passes": flat.get("passes", 0),
                "accurate_long_balls": flat.get("accurateLongBalls", 0),
                "accurate_crosses": flat.get("accurateCross", 0),
                # Duels
                "duel_won_pct": flat.get("duelWonPercent", None),
                "ground_duels_pct": flat.get("groundDuelsPercentage", None),
                "aerial_duels_pct": flat.get("aerialDuelsPercentage", None),
                "dribbles_pct": flat.get("dribblesPercentage", None),
                # Defending
                "tackles_won_pct": flat.get("wonTacklePercent", None),
                "total_tackles": flat.get("totalTackle", 0),
                "interceptions": flat.get("interceptionWon", 0),
                "ball_recoveries": flat.get("ballRecovery", 0),
                "clearances": flat.get("totalClearance", 0),
                "errors_to_shot": flat.get("errorsLeadToShot", 0),
                "errors_to_goal": flat.get("errorsLeadToGoal", 0),
                # GK
                "gk_saves": flat.get("goalkeeperSaves", 0),
                "goals_prevented": flat.get("goalsPrevented", None),
                "dive_saves": flat.get("diveSaves", 0),
                "high_claims": flat.get("highClaims", 0),
                "goal_kicks": flat.get("goalKicks", 0),
                # xG (match level)
                "xg": flat.get("expectedGoals", None),
                "dispossessed": flat.get("dispossessed", 0),
            }

        rows.append(_build_row(home_team, away_team, True, flat_home))
        rows.append(_build_row(away_team, home_team, False, flat_away))

    return rows


def _shot_distance(shot: dict) -> float | None:
    """Euclidean distance from shot location to goal center.

    Sofascore coords: x=0 is the goal line being attacked, y=50 is center.
    Goal center is at (0, 50). Distance in pitch percentage units (~1 unit ≈ 1.05m).
    """
    coords = shot.get("playerCoordinates", {})
    x = coords.get("x")
    y = coords.get("y")
    if x is None or y is None:
        return None
    return (x ** 2 + (y - 50) ** 2) ** 0.5


def extract_shotmap_rows(match_data: dict, fixture: dict, season: str) -> list[dict]:
    """Aggregate shotmap into per-team per-match shot distribution stats.

    Returns two rows per match (home team, away team).
    """
    match_id = match_data.get("match_id", "")
    home_team = normalize_team(fixture.get("homeTeam", {}).get("name", ""))
    away_team = normalize_team(fixture.get("awayTeam", {}).get("name", ""))

    start_ts = fixture.get("startTimestamp", 0)
    if start_ts:
        import datetime
        match_date = datetime.datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d")
    else:
        match_date = ""

    round_info = fixture.get("roundInfo", {}).get("round", "")
    round_info = int(round_info) if round_info else None

    shotmap_raw = match_data.get("shotmap", {})
    if isinstance(shotmap_raw, dict):
        shots = shotmap_raw.get("shotmap", [])
    elif isinstance(shotmap_raw, list):
        shots = shotmap_raw
    else:
        shots = []

    if not shots:
        return []

    rows = []
    for is_home, team, opponent in [(True, home_team, away_team), (False, away_team, home_team)]:
        team_shots = [s for s in shots if s.get("isHome") == is_home]

        if not team_shots:
            rows.append({
                "season": season, "match_id": match_id, "date": match_date,
                "round": round_info, "home_team": home_team, "away_team": away_team,
                "team": team, "opponent": opponent, "is_home": is_home,
                "shots_total": 0, "shots_on_target": 0, "shots_inside_box": 0,
                "shots_header": 0, "shots_right_foot": 0, "shots_left_foot": 0,
                "shots_open_play": 0, "shots_set_piece": 0, "shots_counter": 0,
                "shots_penalty": 0, "total_xg": 0, "total_xgot": 0,
                "avg_shot_xg": 0, "max_shot_xg": 0,
                "goals_from_shots": 0, "big_chance_shots": 0,
                "avg_shot_distance": None, "median_shot_distance": None,
                "shot_distance_std": None, "close_range_pct": 0,
                "shots_hit_post": 0,
            })
            continue

        total = len(team_shots)
        on_target = sum(1 for s in team_shots if s.get("shotType") in ("save", "goal"))
        inside_box = sum(
            1 for s in team_shots
            if (s.get("playerCoordinates", {}).get("x", 100) or 100) <= 17
        )
        header = sum(1 for s in team_shots if s.get("bodyPart") == "head")
        right_foot = sum(1 for s in team_shots if s.get("bodyPart") == "right-foot")
        left_foot = sum(1 for s in team_shots if s.get("bodyPart") == "left-foot")
        open_play = sum(1 for s in team_shots if s.get("situation") in ("assisted", "regular"))
        set_piece = sum(
            1 for s in team_shots
            if s.get("situation") in ("set-piece", "free-kick", "corner")
        )
        counter = sum(1 for s in team_shots if s.get("situation") == "fast-break")
        penalty = sum(1 for s in team_shots if s.get("situation") == "penalty")

        xgs = [s.get("xg", 0) or 0 for s in team_shots]
        xgots = [s.get("xgot", 0) or 0 for s in team_shots]
        total_xg = sum(xgs)
        total_xgot = sum(xgots)
        avg_xg = total_xg / total if total else 0
        max_xg = max(xgs) if xgs else 0
        goals = sum(1 for s in team_shots if s.get("shotType") == "goal")
        hit_post = sum(1 for s in team_shots if s.get("shotType") == "post")

        # Shot distance features (Euclidean distance from goal center)
        distances = [d for d in (_shot_distance(s) for s in team_shots) if d is not None]
        if distances:
            avg_dist = sum(distances) / len(distances)
            sorted_d = sorted(distances)
            n = len(sorted_d)
            median_dist = (sorted_d[n // 2] + sorted_d[(n - 1) // 2]) / 2
            std_dist = (sum((d - avg_dist) ** 2 for d in distances) / len(distances)) ** 0.5
            close_range = sum(1 for d in distances if d < 8) / len(distances)
        else:
            avg_dist = None
            median_dist = None
            std_dist = None
            close_range = 0

        rows.append({
            "season": season, "match_id": match_id, "date": match_date,
            "round": round_info, "home_team": home_team, "away_team": away_team,
            "team": team, "opponent": opponent, "is_home": is_home,
            "shots_total": total,
            "shots_on_target": on_target,
            "shots_inside_box": inside_box,
            "shots_header": header,
            "shots_right_foot": right_foot,
            "shots_left_foot": left_foot,
            "shots_open_play": open_play,
            "shots_set_piece": set_piece,
            "shots_counter": counter,
            "shots_penalty": penalty,
            "total_xg": round(total_xg, 4),
            "total_xgot": round(total_xgot, 4),
            "avg_shot_xg": round(avg_xg, 4),
            "max_shot_xg": round(max_xg, 4),
            "goals_from_shots": goals,
            "big_chance_shots": sum(
                1 for s in team_shots if (s.get("xg", 0) or 0) >= 0.25
            ),
            "avg_shot_distance": round(avg_dist, 2) if avg_dist is not None else None,
            "median_shot_distance": round(median_dist, 2) if median_dist is not None else None,
            "shot_distance_std": round(std_dist, 2) if std_dist is not None else None,
            "close_range_pct": round(close_range, 4),
            "shots_hit_post": hit_post,
        })

    return rows


async def scrape_season(
    api,
    season: str,
    limit: int | None = None,
    league_key: str = "serie_a",
) -> pd.DataFrame:
    """Scrape all matches for a season."""
    _output_dir, _match_cache_dir, _failed_log = _get_output_paths(league_key)
    _matchweeks = get_league_config(league_key).matchweeks_per_season

    fixtures = await get_season_fixtures(
        api, season,
        league_key=league_key,
        output_dir=_output_dir,
        matchweeks=_matchweeks,
    )
    if not fixtures:
        return pd.DataFrame()

    all_player_rows = []
    all_team_rows = []
    all_shotmap_rows = []
    scraped = 0
    cached = 0
    failed = 0
    backfilled = 0
    failed_log = load_failed(_failed_log)

    total = min(len(fixtures), limit) if limit else len(fixtures)

    for i, fixture in enumerate(fixtures[:total]):
        match_id = fixture.get("id")
        if not match_id:
            continue

        home = normalize_team(fixture.get("homeTeam", {}).get("name", "?"))
        away = normalize_team(fixture.get("awayTeam", {}).get("name", "?"))

        # Check if already cached
        cache_file = _match_cache_dir / season / f"{match_id}.json"
        was_cached = cache_file.exists()
        needs_backfill = False
        if was_cached:
            # Check if cached file has team_stats
            try:
                with open(cache_file) as f:
                    peek = json.load(f)
                needs_backfill = "team_stats" not in peek
            except Exception:
                needs_backfill = False

        if not was_cached:
            await asyncio.sleep(RATE_LIMIT)
        elif needs_backfill:
            await asyncio.sleep(1)  # lighter rate limit for backfill

        try:
            match_data = await scrape_match_stats(api, match_id, season, match_cache_dir=_match_cache_dir)

            if match_data:
                # Player stats
                rows = extract_player_rows(match_data, fixture, season)
                all_player_rows.extend(rows)

                # Team-level stats
                team_rows = extract_team_stats_rows(match_data, fixture, season)
                all_team_rows.extend(team_rows)

                # Shotmap aggregates
                shot_rows = extract_shotmap_rows(match_data, fixture, season)
                all_shotmap_rows.extend(shot_rows)

                if was_cached and not needs_backfill:
                    cached += 1
                elif needs_backfill:
                    backfilled += 1
                else:
                    scraped += 1

                # Remove from failed
                mid_str = str(match_id)
                if mid_str in failed_log:
                    del failed_log[mid_str]

                if (scraped + cached + backfilled) % 20 == 0 or scraped <= 3:
                    log.info(
                        "  [%s] %d scraped, %d cached, %d backfilled, %d failed | %d/%d | %s vs %s",
                        season, scraped, cached, backfilled, failed, i + 1, total, home, away,
                    )
            else:
                failed += 1
                failed_log[str(match_id)] = {
                    "season": season,
                    "error": "no_lineup_data",
                    "match": f"{home} vs {away}",
                }

        except Exception as e:
            failed += 1
            log.error("Failed match %s (%s vs %s): %s", match_id, home, away, e)
            failed_log[str(match_id)] = {
                "season": season,
                "error": str(e)[:200],
                "match": f"{home} vs {away}",
            }

    save_failed(_failed_log, failed_log)
    log.info(
        "  [%s/%s] Done: %d scraped, %d cached, %d backfilled, %d failed | %d player, %d team, %d shotmap records",
        league_key, season, scraped, cached, backfilled, failed,
        len(all_player_rows), len(all_team_rows), len(all_shotmap_rows),
    )
    return (
        pd.DataFrame(all_player_rows),
        pd.DataFrame(all_team_rows),
        pd.DataFrame(all_shotmap_rows),
    )


async def async_main(args):
    """Async entry point."""
    from sofascore_wrapper.api import SofascoreAPI

    league_key = args.league
    league_cfg = get_league_config(league_key)
    tournament_id = league_cfg.sofascore_tournament_id

    api = SofascoreAPI()

    # Discover season IDs if needed
    if args.discover_seasons:
        from sofascore_wrapper.league import League
        league = League(api, tournament_id)
        seasons = await league.get_seasons()
        print(f"Available Sofascore seasons for {league_cfg.name} (tournament={tournament_id}):")
        for s in seasons:
            if isinstance(s, dict):
                print(f"  {s.get('name', '?')} — id: {s.get('id', '?')}")
            else:
                print(f"  {s}")
        await api.close()
        return

    # Validate that the league has a season map
    league_season_map = LEAGUE_SEASON_MAPS.get(league_key)
    if not league_season_map:
        log.error(
            "No season map for league '%s'. Run --discover-seasons to find IDs, "
            "then add them to LEAGUE_SEASON_MAPS in scrape_sofascore.py",
            league_key,
        )
        await api.close()
        return

    seasons = [args.season] if args.season else sorted(league_season_map.keys())

    all_player_dfs = []
    all_team_dfs = []
    all_shotmap_dfs = []
    start = time.time()

    for season in seasons:
        log.info("=" * 60)
        log.info("SOFASCORE [%s]: %s", league_cfg.name, season)
        log.info("=" * 60)

        player_df, team_df, shotmap_df = await scrape_season(
            api, season, limit=args.limit, league_key=league_key,
        )
        if not player_df.empty:
            all_player_dfs.append(player_df)
        if not team_df.empty:
            all_team_dfs.append(team_df)
        if not shotmap_df.empty:
            all_shotmap_dfs.append(shotmap_df)

    await api.close()

    # Determine output paths based on league
    output_dir = _league_output_dir(league_key)
    suffix = _league_suffix(league_key)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Helper: merge new data with existing parquet, dedup by match_id + team/player
    def _save_merged(new_dfs, filename, dedup_cols):
        if not new_dfs:
            log.warning("No data collected for %s", filename)
            return
        new_df = pd.concat(new_dfs, ignore_index=True)
        output_path = output_dir / filename
        # Merge with existing data (from other seasons)
        if output_path.exists():
            existing = pd.read_parquet(output_path)
            # Remove seasons we just scraped (to avoid duplicates)
            scraped_seasons = set(new_df["season"].unique())
            existing = existing[~existing["season"].isin(scraped_seasons)]
            # Align columns (new fields may not exist in old data)
            for c in new_df.columns:
                if c not in existing.columns:
                    existing[c] = None
            for c in existing.columns:
                if c not in new_df.columns:
                    new_df[c] = None
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined = combined.drop_duplicates(subset=dedup_cols, keep="last")
        combined.to_parquet(output_path, index=False)
        log.info("Saved %d rows (%d cols) to %s", len(combined), len(combined.columns), output_path)
        return combined

    # Output filenames: serie_a keeps original names, others get a suffix
    player_file = f"player_match_stats{suffix}.parquet"
    team_file = f"match_team_stats{suffix}.parquet"
    shotmap_file = f"shotmap_stats{suffix}.parquet"

    # Save player match stats
    saved = _save_merged(all_player_dfs, player_file,
                         ["match_id", "player_id", "season"])
    if saved is not None:
        log.info("  Seasons: %s | Players: %d | Matches: %d",
                 sorted(saved["season"].unique()),
                 saved["player_name"].nunique(),
                 saved["match_id"].nunique())

    # Save team-level match stats
    _save_merged(all_team_dfs, team_file,
                 ["match_id", "team", "period", "season"])

    # Save shotmap aggregates
    _save_merged(all_shotmap_dfs, shotmap_file,
                 ["match_id", "team", "season"])

    elapsed = time.time() - start
    log.info("Completed [%s] in %.0f min", league_cfg.name, elapsed / 60)


def main():
    valid_leagues = sorted(LEAGUE_SEASON_MAPS.keys())
    parser = argparse.ArgumentParser(description="Scrape Sofascore per-match player stats")
    parser.add_argument("--league", default="serie_a",
                        choices=list(LEAGUE_REGISTRY.keys()),
                        help=f"League to scrape (default: serie_a). "
                             f"Season maps available for: {', '.join(valid_leagues)}")
    parser.add_argument("--season", help="Single season (e.g. 2024-2025)")
    parser.add_argument("--limit", type=int, help="Max matches per season")
    parser.add_argument("--discover-seasons", action="store_true",
                        help="List available season IDs for the selected league")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
