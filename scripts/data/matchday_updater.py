"""Automated post-matchday update: detect new results, fetch Sofascore data, merge into all parquets.

Single CLI command to:
1. Detect completed matches not yet in our Sofascore parquets
2. Fetch player stats, team stats, shotmap, incidents, captains from Sofascore
3. Merge into all Sofascore parquets (player_match_stats, match_team_stats, shotmap_stats, incidents, captains)
4. Build match-level rows and append to matches.parquet
5. Optionally rebuild standings, features, and settle bets

Usage:
    # Auto-detect and fetch new matches (+ standings)
    python -m scripts.data.matchday_updater

    # Full update (features + standings + settle bets)
    python -m scripts.data.matchday_updater --full

    # Dry run — detect only
    python -m scripts.data.matchday_updater --dry-run

    # Force refresh fixtures cache
    python -m scripts.data.matchday_updater --refresh-fixtures

    # Specific season
    python -m scripts.data.matchday_updater --season 2023-2024
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import DATA_DIR, get_current_season, atomic_write_parquet
from config.team_names import normalize_team
from config.leagues import get_league_config, LEAGUE_REGISTRY
from scripts.data.scrape_sofascore import (
    OUTPUT_DIR as SOFASCORE_DIR,
    MATCH_CACHE_DIR,
    RATE_LIMIT,
    LEAGUE_SEASON_MAPS,
    SERIE_A_LEAGUE_ID,
    extract_player_rows,
    extract_shotmap_rows,
    extract_team_stats_rows,
    scrape_match_stats,
)

log = logging.getLogger(__name__)

MATCHES_PARQUET = DATA_DIR / "parsed" / "matches.parquet"
# player_match_stats / match_team_stats / shotmap_stats are league-SPLIT and so
# have no single path — resolve them per league via _sofascore_parquet(). The
# two below are genuinely shared: match ids are globally unique, so one merged
# file holds every league.
INCIDENTS_PARQUET = SOFASCORE_DIR / "match_incidents.parquet"
CAPTAINS_PARQUET = SOFASCORE_DIR / "captains.parquet"

# Fixtures cache staleness threshold (seconds)
FIXTURES_CACHE_MAX_AGE = 6 * 3600  # 6 hours


# ---------------------------------------------------------------------------
# 1. Detect new matches
# ---------------------------------------------------------------------------

def _league_suffix(league: str) -> str:
    """Filename suffix for non-Serie A leagues (empty for serie_a).

    Mirrors scripts/data/scrape_sofascore.py::_league_suffix. That module writes
    the same fixtures caches and per-league stat parquets this one reads, so the
    two MUST agree on every filename.
    """
    return "" if league == "serie_a" else f"_{league}"


def _sofascore_parquet(base: str, league: str) -> Path:
    """Path to a per-league Sofascore stat parquet.

    Serie A owns the unsuffixed name; every other league gets `_{league}`.
    Sofascore match ids are globally unique, so these files are disjoint —
    writing one league's rows into another's file is silent contamination.
    """
    return SOFASCORE_DIR / f"{base}{_league_suffix(league)}.parquet"


def _fixtures_cache_path(season: str, league: str = "serie_a") -> Path:
    """Fixtures cache path, one file per league.

    This took no `league` until 2026-08-25, so both leagues shared a single
    cache. run_matchday_update loops serie_a → premier_league, so Serie A
    refreshed the cache and EPL then found it FRESH (6h window), loaded Serie
    A's fixtures, diffed them against Serie A's ids and detected nothing. EPL
    starved silently on every run — no error, no log, just zero new matches.
    """
    return SOFASCORE_DIR / f"fixtures_{season.replace('-', '_')}{_league_suffix(league)}.json"


def _is_cache_stale(cache_path: Path) -> bool:
    """Check if fixtures cache is older than FIXTURES_CACHE_MAX_AGE."""
    if not cache_path.exists():
        return True
    mtime = cache_path.stat().st_mtime
    age = time.time() - mtime
    return age > FIXTURES_CACHE_MAX_AGE


async def _refresh_fixtures_cache(season: str, league: str = "serie_a") -> list[dict]:
    """Fetch fresh fixtures from Sofascore API and overwrite cache."""
    from sofascore_wrapper.api import SofascoreAPI

    cfg = get_league_config(league)
    tournament_id = cfg.sofascore_tournament_id
    season_map = LEAGUE_SEASON_MAPS.get(league, {})
    ss_season_id = season_map.get(season)
    if not ss_season_id:
        log.warning("Season %s not in Sofascore season map for %s", season, league)
        return []

    max_rounds = cfg.matchweeks_per_season

    api = SofascoreAPI()
    try:
        log.info("Refreshing fixtures cache for %s %s (season_id=%d)...", league, season, ss_season_id)
        all_fixtures = []
        for round_num in range(1, max_rounds + 1):
            try:
                data = await api._get(
                    f"/unique-tournament/{tournament_id}/season/{ss_season_id}"
                    f"/events/round/{round_num}"
                )
                events = data.get("events", [])
                all_fixtures.extend(events)
                await asyncio.sleep(0.5)
            except Exception as e:
                log.debug("Round %d: %s", round_num, e)
                break

        if all_fixtures:
            cache_path = _fixtures_cache_path(season, league)
            try:
                existing = _load_fixtures(season, league)
            except (OSError, ValueError):
                existing = []
            if len(all_fixtures) < len(existing):
                # A round error breaks the loop above, so a transient failure
                # yields a PARTIAL list. Writing it would truncate the season
                # cache — 2026-08-31 a round-3 error cut EPL's 380-row cache to
                # 20 played fixtures and every forward-fixture reader went
                # blind (dormancy flipped, weather/form/referee lost the EPL).
                # A partial fetch never overwrites a fuller cache.
                log.warning(
                    "[%s] partial fixtures fetch (%d < %d cached) — keeping cache",
                    league, len(all_fixtures), len(existing),
                )
                return existing
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(all_fixtures, f, indent=1)
            log.info("Refreshed fixtures cache: %d total fixtures for %s %s", len(all_fixtures), league, season)

        return all_fixtures
    finally:
        await api.close()


def _load_fixtures(season: str, league: str = "serie_a") -> list[dict]:
    """Load fixtures from cache file."""
    cache_path = _fixtures_cache_path(season, league)
    if not cache_path.exists():
        return []
    with open(cache_path) as f:
        return json.load(f)


def _get_existing_sofascore_match_ids(league: str = "serie_a") -> set[int]:
    """Match ids already held in this LEAGUE's player_match_stats parquet.

    Read the Serie A file for every league until 2026-08-25, so EPL fixtures
    were diffed against Serie A's ids. Masked in production by the shared
    fixtures cache above (EPL never got its own fixtures to diff), and only
    visible under force_refresh=True.
    """
    path = _sofascore_parquet("player_match_stats", league)
    if path.exists():
        df = pd.read_parquet(path, columns=["match_id"])
        return set(df["match_id"].unique())
    return set()


def detect_new_matches(
    season: str | None = None,
    force_refresh: bool = False,
    league: str = "serie_a",
) -> list[dict]:
    """Detect completed matches not yet in our Sofascore parquets.

    Args:
        season: Season to check.
        force_refresh: Force refresh fixtures cache even if not stale.
        league: League key (e.g. "serie_a", "premier_league").

    Returns:
        List of fixture dicts for matches that need fetching.
    """
    if season is None:
        season = get_current_season()
    cache_path = _fixtures_cache_path(season, league)

    # Refresh cache if stale or forced
    if force_refresh or _is_cache_stale(cache_path):
        log.info("Fixtures cache stale or refresh forced, fetching fresh data for %s...", league)
        fixtures = asyncio.run(_refresh_fixtures_cache(season, league=league))
        if not fixtures:
            # The refresh came back empty — Sofascore 403s the round endpoint for
            # hours-to-days at a time (see CLAUDE.md "Sofascore API blocks"), and
            # _refresh_fixtures_cache swallows that into a debug log and returns [].
            # Returning [] here means the league detects nothing for the whole ban
            # even though a usable cache is sitting on disk. A stale fixtures list
            # costs nothing: already-ingested matches are filtered out below anyway.
            fixtures = _load_fixtures(season, league)
            if fixtures:
                log.warning(
                    "[%s] fixtures refresh returned nothing — falling back to the "
                    "cache on disk (%d fixtures)", league, len(fixtures),
                )
    else:
        fixtures = _load_fixtures(season, league)

    if not fixtures:
        log.warning("No fixtures found for %s", season)
        return []

    # Filter to finished matches only
    finished = [f for f in fixtures if f.get("status", {}).get("type") == "finished"]
    log.info("Found %d finished matches in %s fixtures cache", len(finished), season)

    # Diff against existing data
    existing_ids = _get_existing_sofascore_match_ids(league)
    new_fixtures = [f for f in finished if f.get("id") not in existing_ids]

    if new_fixtures:
        log.info("Detected %d NEW matches to fetch (have %d already)", len(new_fixtures), len(existing_ids))
        for f in new_fixtures:
            home = f.get("homeTeam", {}).get("name", "?")
            away = f.get("awayTeam", {}).get("name", "?")
            rd = f.get("roundInfo", {}).get("round", "?")
            log.info("  Round %s: %s vs %s (id=%s)", rd, home, away, f.get("id"))
    else:
        log.info("No new matches detected (all %d finished matches already in parquets)", len(finished))

    return new_fixtures


# ---------------------------------------------------------------------------
# 2. Fetch match details
# ---------------------------------------------------------------------------

async def fetch_match_details(
    fixtures: list[dict],
    season: str,
) -> list[tuple[dict, dict]]:
    """Fetch full match data from Sofascore API for each new fixture.

    Args:
        fixtures: List of Sofascore fixture dicts.
        season: Season string.

    Returns:
        List of (match_data, fixture) pairs for successfully fetched matches.
    """
    from sofascore_wrapper.api import SofascoreAPI

    api = SofascoreAPI()
    results = []

    try:
        for i, fixture in enumerate(fixtures):
            match_id = fixture.get("id")
            if not match_id:
                continue

            home = normalize_team(fixture.get("homeTeam", {}).get("name", "?"))
            away = normalize_team(fixture.get("awayTeam", {}).get("name", "?"))

            log.info("Fetching match %d/%d: %s vs %s (id=%d)...",
                     i + 1, len(fixtures), home, away, match_id)

            try:
                match_data = await scrape_match_stats(api, match_id, season)
                if match_data:
                    results.append((match_data, fixture))
                    log.info("  OK — got player+team+shotmap data")
                else:
                    log.warning("  SKIP — no data returned for match %d", match_id)
            except Exception as e:
                log.error("  FAIL — match %d: %s", match_id, e)

            # Rate limit between API calls
            if i < len(fixtures) - 1:
                await asyncio.sleep(RATE_LIMIT)
    finally:
        await api.close()

    log.info("Fetched %d/%d matches successfully", len(results), len(fixtures))
    return results


# ---------------------------------------------------------------------------
# 3. Merge into Sofascore parquets
# ---------------------------------------------------------------------------

def _save_merged(new_df: pd.DataFrame, output_path: Path, dedup_cols: list[str]) -> int:
    """Merge new data with existing parquet, dedup, and save. Returns rows added."""
    if new_df.empty:
        return 0

    if output_path.exists():
        existing = pd.read_parquet(output_path)
        rows_before = len(existing)

        # Align columns
        for c in new_df.columns:
            if c not in existing.columns:
                existing[c] = None
        for c in existing.columns:
            if c not in new_df.columns:
                new_df[c] = None

        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        rows_before = 0
        combined = new_df

    combined = combined.drop_duplicates(subset=dedup_cols, keep="last")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(output_path, combined, index=False)

    rows_added = len(combined) - rows_before
    log.info("Saved %s: %d rows (+%d new)", output_path.name, len(combined), rows_added)
    return rows_added


def merge_to_sofascore_parquets(
    match_data_pairs: list[tuple[dict, dict]],
    season: str,
    league: str = "serie_a",
) -> dict[str, int]:
    """Merge fetched data into player_match_stats, match_team_stats, shotmap_stats parquets.

    These three files are league-SPLIT (Serie A unsuffixed, others `_{league}`)
    and hold disjoint match ids. This wrote the Serie A files for every league
    until 2026-08-25; it never corrupted anything only because the shared
    fixtures cache meant no EPL match was ever fetched here.

    Args:
        match_data_pairs: List of (match_data, fixture) tuples.
        season: Season string.
        league: League key (e.g. "serie_a", "premier_league").

    Returns:
        Dict mapping parquet name to rows added.
    """
    all_player_rows = []
    all_team_rows = []
    all_shotmap_rows = []

    for match_data, fixture in match_data_pairs:
        all_player_rows.extend(extract_player_rows(match_data, fixture, season))
        all_team_rows.extend(extract_team_stats_rows(match_data, fixture, season))
        all_shotmap_rows.extend(extract_shotmap_rows(match_data, fixture, season))

    result = {}

    if all_player_rows:
        df = pd.DataFrame(all_player_rows)
        result["player_match_stats"] = _save_merged(
            df,
            _sofascore_parquet("player_match_stats", league),
            ["match_id", "player_id", "season"],
        )

    if all_team_rows:
        df = pd.DataFrame(all_team_rows)
        result["match_team_stats"] = _save_merged(
            df,
            _sofascore_parquet("match_team_stats", league),
            ["match_id", "team", "period", "season"],
        )

    if all_shotmap_rows:
        df = pd.DataFrame(all_shotmap_rows)
        result["shotmap_stats"] = _save_merged(
            df,
            _sofascore_parquet("shotmap_stats", league),
            ["match_id", "team", "season"],
        )

    return result


# ---------------------------------------------------------------------------
# 4. Fetch and merge incidents + captains
# ---------------------------------------------------------------------------

def fetch_and_merge_incidents(match_ids: list[int], season: str) -> dict[str, int]:
    """Use sofascore_events.py to fetch incidents + captains for new matches.

    Args:
        match_ids: Sofascore match IDs to fetch.
        season: Season string.

    Returns:
        Dict with counts of incidents and captains added.
    """
    from scraper.sofascore_events import scrape_captains, scrape_incidents

    result = {}

    log.info("Fetching incidents for %d matches...", len(match_ids))
    try:
        incidents_df = scrape_incidents(match_ids=match_ids, force=False)
        result["incidents"] = len(incidents_df) if not incidents_df.empty else 0
    except Exception as e:
        log.error("Failed to fetch incidents: %s", e)
        result["incidents"] = 0

    log.info("Fetching captain data for %d matches...", len(match_ids))
    try:
        captains_df = scrape_captains(match_ids=match_ids, force=False)
        result["captains"] = len(captains_df) if not captains_df.empty else 0
    except Exception as e:
        log.error("Failed to fetch captains: %s", e)
        result["captains"] = 0

    return result


# ---------------------------------------------------------------------------
# 5. Update matches.parquet
# ---------------------------------------------------------------------------

def _extract_card_counts(match_id: int) -> dict[str, int]:
    """Count yellow/red cards per side from match_incidents.parquet."""
    defaults = {
        "home_yellow_cards": 0, "home_red_cards": 0,
        "away_yellow_cards": 0, "away_red_cards": 0,
    }
    if not INCIDENTS_PARQUET.exists():
        return defaults

    try:
        df = pd.read_parquet(INCIDENTS_PARQUET)
        match_inc = df[df["match_id"] == match_id]
        if match_inc.empty:
            return defaults

        cards = match_inc[match_inc["incident_type"] == "card"]

        for side, is_home in [("home", True), ("away", False)]:
            side_cards = cards[cards["is_home"] == is_home]
            # incidentClass values: yellow, red, yellowRed
            defaults[f"{side}_yellow_cards"] = int(
                side_cards["incident_class"].isin(["yellow", "yellowRed"]).sum()
            )
            defaults[f"{side}_red_cards"] = int(
                side_cards["incident_class"].isin(["red", "yellowRed"]).sum()
            )
    except Exception as e:
        log.debug("Could not extract card counts for match %d: %s", match_id, e)

    return defaults


def _get_team_stat(team_stats_raw: dict, key: str, period: str = "ALL") -> float | None:
    """Extract a single stat value from team_stats structure for a given period and side."""
    if not team_stats_raw:
        return None

    periods = team_stats_raw.get("statistics", [])
    for period_data in periods:
        if period_data.get("period") != period:
            continue
        for group in period_data.get("groups", []):
            for item in group.get("statisticsItems", []):
                if item.get("key") == key:
                    return item
    return None


def _get_team_stat_value(
    team_stats_raw: dict, key: str, side: str, period: str = "ALL"
) -> float | int | None:
    """Extract a stat value for home or away side from team_stats."""
    item = _get_team_stat(team_stats_raw, key, period)
    if item is None:
        return None
    return item.get(f"{side}Value")


def update_matches_parquet(
    match_data_pairs: list[tuple[dict, dict]],
    season: str,
    league: str = "serie_a",
) -> int:
    """Build match-level rows from Sofascore data and append to matches.parquet.

    Maps Sofascore fixture + stats → matches.parquet schema (75 cols).
    Deduplicates by (home_team, away_team, match_date, season).

    Args:
        match_data_pairs: List of (match_data, fixture) tuples.
        season: Season string.
        league: League key (e.g. "serie_a", "premier_league").

    Returns:
        Number of rows added.
    """
    rows = []

    for match_data, fixture in match_data_pairs:
        # Date from timestamp
        start_ts = fixture.get("startTimestamp", 0)
        if start_ts:
            match_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
            match_date = match_dt.strftime("%Y-%m-%d")
            kickoff_time = match_dt.strftime("%H:%M")
        else:
            match_date = ""
            kickoff_time = ""

        home_team = normalize_team(fixture.get("homeTeam", {}).get("name", ""))
        away_team = normalize_team(fixture.get("awayTeam", {}).get("name", ""))

        # matches.parquet is keyed by {date}_{home}_{away}, never by Sofascore's
        # fixture id. This wrote str(fixture["id"]) until 2026-07-17, which left
        # 94 numeric keys in the ground-truth id column beside 15,795 canonical
        # ones — two incompatible formats in one column. A dateless fixture can
        # not produce that key, and a row with no date is unusable here anyway,
        # so it is dropped rather than keyed on something that will not join.
        if not (match_date and home_team and away_team):
            continue
        match_id = f"{match_date}_{home_team}_{away_team}"

        matchday = fixture.get("roundInfo", {}).get("round", None)

        home_score = fixture.get("homeScore", {}).get("current")
        away_score = fixture.get("awayScore", {}).get("current")
        home_ht = fixture.get("homeScore", {}).get("period1")
        away_ht = fixture.get("awayScore", {}).get("period1")

        # Team stats
        ts = match_data.get("team_stats", {})
        home_xg = _get_team_stat_value(ts, "expectedGoals", "home")
        away_xg = _get_team_stat_value(ts, "expectedGoals", "away")
        home_possession = _get_team_stat_value(ts, "ballPossession", "home")
        away_possession = _get_team_stat_value(ts, "ballPossession", "away")
        home_shots_total = _get_team_stat_value(ts, "totalShotsOnGoal", "home")
        away_shots_total = _get_team_stat_value(ts, "totalShotsOnGoal", "away")
        home_sot = _get_team_stat_value(ts, "shotsOnGoal", "home")
        away_sot = _get_team_stat_value(ts, "shotsOnGoal", "away")
        home_corners = _get_team_stat_value(ts, "cornerKicks", "home")
        away_corners = _get_team_stat_value(ts, "cornerKicks", "away")
        home_fouls = _get_team_stat_value(ts, "fouls", "home")
        away_fouls = _get_team_stat_value(ts, "fouls", "away")
        home_offsides = _get_team_stat_value(ts, "offsides", "home")
        away_offsides = _get_team_stat_value(ts, "offsides", "away")
        home_tackles = _get_team_stat_value(ts, "totalTackle", "home")
        away_tackles = _get_team_stat_value(ts, "totalTackle", "away")
        home_interceptions = _get_team_stat_value(ts, "interceptionWon", "home")
        away_interceptions = _get_team_stat_value(ts, "interceptionWon", "away")
        home_clearances = _get_team_stat_value(ts, "totalClearance", "home")
        away_clearances = _get_team_stat_value(ts, "totalClearance", "away")
        home_saves = _get_team_stat_value(ts, "goalkeeperSaves", "home")
        away_saves = _get_team_stat_value(ts, "goalkeeperSaves", "away")
        home_throw_ins = _get_team_stat_value(ts, "throwIns", "home")
        away_throw_ins = _get_team_stat_value(ts, "throwIns", "away")
        home_goal_kicks = _get_team_stat_value(ts, "goalKicks", "home")
        away_goal_kicks = _get_team_stat_value(ts, "goalKicks", "away")
        home_aerials = _get_team_stat_value(ts, "aerialDuelsPercentage", "home")
        away_aerials = _get_team_stat_value(ts, "aerialDuelsPercentage", "away")
        home_long_balls = _get_team_stat_value(ts, "accurateLongBalls", "home")
        away_long_balls = _get_team_stat_value(ts, "accurateLongBalls", "away")
        home_passes_acc = _get_team_stat_value(ts, "accuratePasses", "home")
        away_passes_acc = _get_team_stat_value(ts, "accuratePasses", "away")
        home_passes_total = _get_team_stat_value(ts, "passes", "home")
        away_passes_total = _get_team_stat_value(ts, "passes", "away")
        home_crosses = _get_team_stat_value(ts, "accurateCross", "home")
        away_crosses = _get_team_stat_value(ts, "accurateCross", "away")
        home_touches = _get_team_stat_value(ts, "touchesInOppBox", "home")
        away_touches = _get_team_stat_value(ts, "touchesInOppBox", "away")

        # Formation from lineup data
        home_lineup = match_data.get("home_lineup", {})
        away_lineup = match_data.get("away_lineup", {})
        home_formation = ""
        away_formation = ""
        if isinstance(home_lineup, dict):
            home_formation = home_lineup.get("formation", "")
        if isinstance(away_lineup, dict):
            away_formation = away_lineup.get("formation", "")

        # Referee and venue
        referee_info = fixture.get("referee", {}) or {}
        referee = referee_info.get("name", "") or _referee_from_espn(league, match_date, home_team, away_team)
        # None, never "": 31 of 45 rows of 2026-27 carried "" and read as
        # "filled" in every coverage check while add_referee_features got nothing
        referee = referee or None
        venue_info = fixture.get("venue", {}) or {}
        venue = venue_info.get("stadium", {}).get("name", "") if isinstance(venue_info.get("stadium"), dict) else ""

        # Card counts from incidents
        card_counts = _extract_card_counts(fixture.get("id", 0))

        # Compute passing accuracy as percentage
        home_pass_pct = None
        away_pass_pct = None
        if home_passes_acc is not None and home_passes_total and home_passes_total > 0:
            home_pass_pct = round(home_passes_acc / home_passes_total * 100, 1)
        if away_passes_acc is not None and away_passes_total and away_passes_total > 0:
            away_pass_pct = round(away_passes_acc / away_passes_total * 100, 1)

        row = {
            "match_id": match_id,
            "season": season,
            "match_date": pd.Timestamp(match_date),
            "kickoff_time": kickoff_time,
            "matchweek": int(matchday) if matchday is not None else None,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "venue": venue,
            "attendance": None,  # Not available from Sofascore
            "referee": referee,
            "home_formation": home_formation,
            "away_formation": away_formation,
            "home_possession": home_possession,
            "away_possession": away_possession,
            "home_passing_accuracy": home_pass_pct,
            "away_passing_accuracy": away_pass_pct,
            "home_shots_on_target": home_sot,
            "away_shots_on_target": away_sot,
            "home_saves": home_saves,
            "away_saves": away_saves,
            "home_cards": (card_counts["home_yellow_cards"] or 0) + (card_counts["home_red_cards"] or 0),
            "away_cards": (card_counts["away_yellow_cards"] or 0) + (card_counts["away_red_cards"] or 0),
            "home_fouls": home_fouls,
            "away_fouls": away_fouls,
            "home_corners": home_corners,
            "away_corners": away_corners,
            "home_crosses": home_crosses,
            "away_crosses": away_crosses,
            "home_touches": home_touches,
            "away_touches": away_touches,
            "home_tackles": home_tackles,
            "away_tackles": away_tackles,
            "home_interceptions": home_interceptions,
            "away_interceptions": away_interceptions,
            "home_aerials_won": home_aerials,
            "away_aerials_won": away_aerials,
            "home_clearances": home_clearances,
            "away_clearances": away_clearances,
            "home_offsides": home_offsides,
            "away_offsides": away_offsides,
            "home_goal_kicks": home_goal_kicks,
            "away_goal_kicks": away_goal_kicks,
            "home_throw_ins": home_throw_ins,
            "away_throw_ins": away_throw_ins,
            "home_long_balls": home_long_balls,
            "away_long_balls": away_long_balls,
            "home_passing_accuracy_count": home_passes_acc,
            "home_passing_accuracy_total": home_passes_total,
            "away_passing_accuracy_count": away_passes_acc,
            "away_passing_accuracy_total": away_passes_total,
            "home_shots_on_target_count": home_sot,
            "home_shots_on_target_total": home_shots_total,
            "away_shots_on_target_count": away_sot,
            "away_shots_on_target_total": away_shots_total,
            "home_saves_count": home_saves,
            "home_saves_total": home_saves,  # Sofascore only gives saves (not total faced)
            "away_saves_count": away_saves,
            "away_saves_total": away_saves,
            "home_yellow_cards": card_counts["home_yellow_cards"],
            "home_red_cards": card_counts["home_red_cards"],
            "away_yellow_cards": card_counts["away_yellow_cards"],
            "away_red_cards": card_counts["away_red_cards"],
            "home_xg": home_xg,
            "away_xg": away_xg,
            "home_manager": "",  # Not in Sofascore fixture data
            "away_manager": "",
            "home_captain": "",
            "away_captain": "",
            "data_source": "sofascore",
            "home_ht_score": home_ht,
            "away_ht_score": away_ht,
            "league": league,
            "league_name": get_league_config(league).name,
        }

        rows.append(row)

    if not rows:
        return 0

    new_df = pd.DataFrame(rows)

    # Load existing matches.parquet and merge
    if MATCHES_PARQUET.exists():
        existing = pd.read_parquet(MATCHES_PARQUET)
        rows_before = len(existing)

        # Align columns
        for c in new_df.columns:
            if c not in existing.columns:
                existing[c] = None
        for c in existing.columns:
            if c not in new_df.columns:
                new_df[c] = None

        combined = pd.concat([existing, new_df], ignore_index=True)

        # Dedup by (home_team, away_team, match_date, season) — handles multiple data sources
        combined = combined.drop_duplicates(
            subset=["home_team", "away_team", "match_date", "season"],
            keep="first",  # Keep existing (e.g., FBref data) over Sofascore for same match
        )
    else:
        rows_before = 0
        combined = new_df

    MATCHES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(MATCHES_PARQUET, combined, index=False)

    rows_added = len(combined) - rows_before
    log.info("matches.parquet: %d rows (+%d new)", len(combined), rows_added)
    return rows_added


# ---------------------------------------------------------------------------
# 5b-pre. Backfill: ground-truth rows for matches whose stats already exist
# ---------------------------------------------------------------------------

def _referee_from_espn(league: str, match_date: str, home: str, away: str) -> str | None:
    """Sofascore's fixture list names no referee this season (0 of 21 finished
    2026-27 fixtures) and worldfootball publishes a season late; ESPN's summary
    names the referee once the match is played (live_espn.match_referee).
    Never raises: a missing referee is a NaN feature, not a failed ingest."""
    if not match_date:
        return None
    try:
        from scripts.data.live_espn import match_referee
        return match_referee(league, match_date, home, away)
    except Exception as e:  # noqa: BLE001 - network / parse trouble
        log.debug("ESPN referee lookup failed for %s vs %s (%s): %s", home, away, match_date, e)
        return None


def backfill_referees(season: str | None = None, league: str = "serie_a",
                      dry_run: bool = False) -> dict:
    """Fill the referee of played rows that have none (NaN or "") from ESPN,
    and normalise "" to None everywhere. Idempotent; a row ESPN cannot name
    stays None and is retried on the next call."""
    if season is None:
        season = get_current_season()
    summary = {"league": league, "season": season, "candidates": 0, "filled": 0, "blanked": 0}
    if not MATCHES_PARQUET.exists():
        return summary
    gt = pd.read_parquet(MATCHES_PARQUET)
    if "referee" not in gt.columns:
        return summary
    empty = gt["referee"].astype(object).map(lambda v: isinstance(v, str) and not v.strip())
    summary["blanked"] = int(empty.sum())
    gt.loc[empty, "referee"] = None
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    dates = gt["match_date"].astype(str).str[:10]
    mask = ((gt["league"] == league) & (gt["season"] == season) & gt["referee"].isna()
            & (dates <= today) & gt["home_score"].notna())
    summary["candidates"] = int(mask.sum())
    for idx in gt.index[mask]:
        row = gt.loc[idx]
        name = _referee_from_espn(league, dates[idx], row["home_team"], row["away_team"])
        if name:
            gt.at[idx, "referee"] = name
            summary["filled"] += 1
    log.info("[%s] referee backfill %s: %d candidates, %d filled, %d blanks normalised%s",
             league, season, summary["candidates"], summary["filled"], summary["blanked"],
             " (dry run)" if dry_run else "")
    if not dry_run and (summary["filled"] or summary["blanked"]):
        gt.to_parquet(MATCHES_PARQUET, index=False)
    return summary


def backfill_matches_parquet(
    season: str | None = None,
    league: str = "serie_a",
    dry_run: bool = False,
) -> dict:
    """Write matches.parquet rows for played fixtures that never got one.

    detect_new_matches() diffs the fixture list against the STAT parquets, so a
    match whose stats were scraped while the ground-truth append was broken (the
    2026-03..08 EPL starvation) is invisible to it forever: the stats exist, the
    matches.parquet row does not, and "0 new" looks exactly like health. This
    keys the diff on matches.parquet itself — the artifact actually missing
    rows — and rebuilds each missing row from the per-match JSON already on
    disk, so it needs no network. Idempotent: update_matches_parquet dedups on
    (home_team, away_team, match_date, season) with keep="first".
    """
    if season is None:
        season = get_current_season()

    fixtures = _load_fixtures(season, league)
    finished = [f for f in fixtures if f.get("status", {}).get("type") == "finished"]
    summary = {"league": league, "season": season, "finished_fixtures": len(finished),
               "missing": 0, "no_json": 0, "rows_added": 0}
    if not finished:
        log.warning("[%s] No finished fixtures in cache for %s — nothing to backfill",
                    league, season)
        return summary

    existing: set = set()
    if MATCHES_PARQUET.exists():
        gt = pd.read_parquet(
            MATCHES_PARQUET, columns=["match_date", "home_team", "away_team", "league"]
        )
        gt = gt[gt["league"] == league]
        existing = set(zip(gt["match_date"].astype(str).str[:10],
                           gt["home_team"], gt["away_team"]))

    json_dir = SOFASCORE_DIR / f"matches{_league_suffix(league)}" / season
    pairs: list[tuple[dict, dict]] = []
    for f in finished:
        ts = f.get("startTimestamp")
        home = normalize_team(f.get("homeTeam", {}).get("name", ""))
        away = normalize_team(f.get("awayTeam", {}).get("name", ""))
        if not (ts and home and away):
            continue
        # Same key derivation as update_matches_parquet — the two MUST agree.
        date = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        if (date, home, away) in existing:
            continue
        summary["missing"] += 1
        jp = json_dir / f"{f.get('id')}.json"
        if not jp.exists():
            summary["no_json"] += 1
            log.warning("[%s] %s %s vs %s missing from ground truth but no JSON on "
                        "disk (%s) — the normal API path will pick it up",
                        league, date, home, away, jp)
            continue
        try:
            with open(jp) as fh:
                match_data = json.load(fh)
        except (OSError, ValueError) as e:
            summary["no_json"] += 1
            log.warning("[%s] unreadable match JSON %s: %s", league, jp, e)
            continue
        pairs.append((match_data, f))
        log.info("[%s] backfill: %s %s vs %s (id=%s)", league, date, home, away, f.get("id"))

    if dry_run or not pairs:
        log.info("[%s] %s: %d missing, %d with JSON on disk%s", league, season,
                 summary["missing"], len(pairs),
                 " (dry run — nothing written)" if dry_run else "")
        return summary

    summary["rows_added"] = update_matches_parquet(pairs, season, league=league)
    return summary


# ---------------------------------------------------------------------------
# 5b. Fallback: ingest from results.json when Sofascore is unavailable
# ---------------------------------------------------------------------------

def _fallback_ingest_from_results(season: str | None = None) -> int:
    """Ingest completed matches from results.json into matches.parquet.

    Used when Sofascore API is down or sofascore_wrapper is unavailable.
    Only adds score + basic metadata (no xG, possession, player stats).
    Enough to keep standings and features.parquet up to date.

    Returns:
        Number of rows added.
    """
    if season is None:
        season = get_current_season()
    import numpy as np
    from config.team_names import normalize_team

    results_path = DATA_DIR / "upcoming" / "results.json"
    matches_path = DATA_DIR / "parsed" / "matches.parquet"

    if not results_path.exists():
        log.warning("Fallback: results.json not found")
        return 0
    if not matches_path.exists():
        log.warning("Fallback: matches.parquet not found")
        return 0

    with open(results_path) as f:
        results_data = json.load(f)

    results = results_data.get("results", {})
    if isinstance(results, list):
        results = {f"{r['home_team']}_v_{r['away_team']}": r for r in results}

    df = pd.read_parquet(matches_path)
    existing = set()
    df_season = df[df.season == season]
    for _, row in df_season.iterrows():
        key = f"{row.home_team}_{row.away_team}_{str(row.match_date)[:10]}"
        existing.add(key)

    new_rows = []
    for match_key, m in results.items():
        if not m.get("completed"):
            continue
        home = normalize_team(m.get("home_team", ""))
        away = normalize_team(m.get("away_team", ""))
        ct = m.get("commence_time", "")
        date = ct[:10] if ct else ""
        if not date or not home or not away:
            continue

        key = f"{home}_{away}_{date}"
        if key in existing:
            continue

        row = {col: None for col in df.columns}
        row["match_id"] = f"{date}_{home}_{away}".replace(" ", "_")
        row["season"] = season
        row["match_date"] = pd.Timestamp(date)  # Must be datetime, not string
        row["kickoff_time"] = ct[11:16] if len(ct) > 11 else ""
        row["home_team"] = home
        row["away_team"] = away
        row["home_score"] = float(m["home_score"])
        row["away_score"] = float(m["away_score"])
        # Detect league from team names
        from config.team_names import PREMIER_LEAGUE_NAMES
        epl_teams = set(PREMIER_LEAGUE_NAMES.values())
        row["league"] = "premier_league" if home in epl_teams or away in epl_teams else "serie_a"
        row["league_name"] = "Premier League" if row["league"] == "premier_league" else "Serie A"
        # Compute result from scores
        hs, as_ = row["home_score"], row["away_score"]
        row["result"] = "H" if hs > as_ else ("A" if as_ > hs else "D")
        new_rows.append(row)
        log.info("Fallback: adding %s %d-%d %s (%s)",
                 home, m["home_score"], m["away_score"], away, date)

    if not new_rows:
        return 0

    new_df = pd.DataFrame(new_rows)
    # Align dtypes with existing parquet to prevent type conversion errors
    for col in new_df.columns:
        if col in df.columns:
            try:
                new_df[col] = new_df[col].astype(df[col].dtype)
            except (ValueError, TypeError):
                pass  # leave as-is if conversion fails (NaN in int cols etc.)
    combined = pd.concat([df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["home_team", "away_team", "match_date"], keep="first"
    )
    atomic_write_parquet(matches_path, combined, index=False)
    rows_added = len(combined) - len(df)
    log.info("Fallback: matches.parquet %d → %d rows (+%d)", len(df), len(combined), rows_added)
    return rows_added


# ---------------------------------------------------------------------------
# 6. Main orchestrator
# ---------------------------------------------------------------------------

PLAYER_META_MAX_AGE_DAYS = 3.0


def _player_meta_needs_refresh(path: Path, matches_fetched: int | None) -> bool:
    """Should Step 6d rebuild ``player_metadata.json``?

    Do NOT gate this on ``matches_fetched`` alone. That counter reads 0 on every
    observed run — EPL detection is blind to matches already present in the stat
    parquets — so a fetch-only gate fires once and then never again, which is how
    this file sat frozen from 2026-02-17 with no writer at all. File age cannot go
    silently false the way a detection counter can.
    """
    if matches_fetched:
        return True
    try:
        age_days = (time.time() - path.stat().st_mtime) / 86400
    except OSError:
        return True  # missing or unreadable -> always rebuild
    return age_days > PLAYER_META_MAX_AGE_DAYS


def _should_rebuild_features(summary: dict) -> bool:
    """Rebuild features only when this run actually changed the data on disk.

    build_features(use_cache=False) is a ~23-minute full rebuild, not the
    "~30s" the old log line claimed. The settlement tick calls this every
    5 minutes on matchdays, so an unconditional rebuild saturated the box
    all afternoon on 2026-08-28 and starved the T-30 pre-kickoff run into
    its subprocess timeout. Zero rows ingested == features provably
    unchanged == rebuild is pure waste.
    """
    added = summary.get("matches_parquet_added", 0) or 0
    fetched = summary.get("matches_fetched", 0) or 0
    return (added + fetched) > 0


def run_matchday_update(
    season: str | None = None,
    rebuild_features: bool = False,
    regenerate_standings: bool = True,
    settle_bets: bool = False,
    dry_run: bool = False,
    force_refresh: bool = False,
    leagues: list[str] | None = None,
) -> dict:
    """Main orchestrator: detect → fetch → merge → rebuild.

    Runs the update for each league in `leagues` (default: serie_a + premier_league).

    Args:
        season: Season to process.
        rebuild_features: If True, rebuild features.parquet after merge.
        regenerate_standings: If True (default), regenerate standings.json.
        settle_bets: If True, run results_fetcher.fetch_and_settle().
        dry_run: If True, only detect new matches without fetching.
        force_refresh: If True, force refresh fixtures cache.
        leagues: List of league keys to update. Defaults to ["serie_a", "premier_league"].

    Returns:
        Summary dict with counts and timings.
    """
    if leagues is None:
        from config.leagues import ACTIVE_LEAGUES
        leagues = list(ACTIVE_LEAGUES)
    if season is None:
        season = get_current_season()
    t0 = time.time()
    summary = {
        "season": season,
        "leagues": leagues,
        "new_matches_detected": 0,
        "matches_fetched": 0,
        "dry_run": dry_run,
    }

    parquet_counts = {}

    for league in leagues:
        league_name = get_league_config(league).name

        # Step 1: Detect new matches
        log.info("=" * 60)
        log.info("MATCHDAY UPDATER — %s %s", league_name, season)
        log.info("=" * 60)

        sofascore_ok = True
        new_fixtures = []
        try:
            new_fixtures = detect_new_matches(season=season, force_refresh=force_refresh, league=league)
        except Exception as e:
            log.warning("[%s] Sofascore detection failed: %s — falling back to results.json", league, e)
            sofascore_ok = False

        summary["new_matches_detected"] += len(new_fixtures)

        if sofascore_ok and new_fixtures and not dry_run:
            # === PRIMARY PATH: Sofascore-based ingestion ===
            # Step 2: Fetch match details from Sofascore
            log.info("-" * 40)
            log.info("[%s] Step 2: Fetching match details from Sofascore...", league)
            match_data_pairs = asyncio.run(fetch_match_details(new_fixtures, season))
            summary["matches_fetched"] += len(match_data_pairs)

            if match_data_pairs:
                # Step 3: Merge into Sofascore parquets
                log.info("-" * 40)
                log.info("[%s] Step 3: Merging into Sofascore parquets...", league)
                league_parquet_counts = merge_to_sofascore_parquets(
                    match_data_pairs, season, league=league
                )
                for k, v in league_parquet_counts.items():
                    parquet_counts[k] = parquet_counts.get(k, 0) + v
                summary["sofascore_parquets"] = parquet_counts

                # Step 4: Fetch incidents + captains
                log.info("-" * 40)
                log.info("[%s] Step 4: Fetching incidents + captains...", league)
                match_ids = [fixture.get("id") for _, fixture in match_data_pairs if fixture.get("id")]
                incident_counts = fetch_and_merge_incidents(match_ids, season)
                summary.setdefault("incidents", {})
                for k, v in incident_counts.items():
                    summary["incidents"][k] = summary["incidents"].get(k, 0) + v

                # Step 5: Update matches.parquet
                log.info("-" * 40)
                log.info("[%s] Step 5: Updating matches.parquet...", league)
                matches_added = update_matches_parquet(match_data_pairs, season, league=league)
                summary["matches_parquet_added"] = summary.get("matches_parquet_added", 0) + matches_added
            else:
                log.warning("[%s] No match data fetched from Sofascore — falling back to results.json", league)
                sofascore_ok = False

        if not sofascore_ok or (not new_fixtures and not dry_run):
            # === FALLBACK PATH: results.json-based ingestion ===
            # When Sofascore is down or unavailable, use Odds API results to update matches.parquet
            fallback_added = _fallback_ingest_from_results(season)
            summary["fallback_used"] = True
            summary["matches_parquet_added"] = summary.get("matches_parquet_added", 0) + fallback_added
            if fallback_added > 0:
                log.info("[%s] Fallback: added %d matches from results.json", league, fallback_added)
            elif not new_fixtures:
                log.info("[%s] Nothing to update — all matches already in parquets", league)

        # Referees land AFTER the match on ESPN and never in the Sofascore
        # fixture list: fill the played rows that still have none, every run.
        if not dry_run:
            try:
                ref_fill = backfill_referees(season=season, league=league)
                summary["referees_filled"] = summary.get("referees_filled", 0) + ref_fill["filled"]
            except Exception as e:  # noqa: BLE001 - a NaN feature, not a failed ingest
                log.warning("[%s] referee backfill failed: %s", league, e)

    if dry_run:
        total_detected = summary["new_matches_detected"]
        log.info("DRY RUN — would fetch %d matches across %s. Exiting.", total_detected, ", ".join(leagues))
        summary["elapsed_seconds"] = round(time.time() - t0, 1)
        return summary

    # Step 6: Downstream triggers
    log.info("-" * 40)

    if regenerate_standings:
        log.info("Step 6a: Regenerating standings...")
        try:
            from scripts.prediction.standings_generator import generate_current_standings
            standings = generate_current_standings()
            summary["standings_regenerated"] = bool(standings)
            log.info("Standings regenerated: %d teams", len(standings))
        except Exception as e:
            log.error("Failed to regenerate standings: %s", e)
            summary["standings_regenerated"] = False

    if rebuild_features and not _should_rebuild_features(summary):
        log.info("Step 6b skipped — nothing ingested this run, features already current")
        rebuild_features = False
    if rebuild_features:
        log.info("Step 6b: Rebuilding features.parquet (full use_cache=False rebuild, ~23 min)...")
        try:
            from features.build import build_features
            features_df = build_features(use_cache=False)
            summary["features_rebuilt"] = len(features_df)
            log.info("Features rebuilt: %d rows × %d cols",
                     len(features_df), len(features_df.columns))
        except Exception as e:
            log.error("Failed to rebuild features: %s", e)
            summary["features_rebuilt"] = 0

    if settle_bets:
        log.info("Step 6c: Settling bets via results_fetcher...")
        try:
            from scripts.data.results_fetcher import fetch_and_settle
            settle_result = fetch_and_settle()
            summary["bets_settled"] = settle_result
        except Exception as e:
            log.error("Failed to settle bets: %s", e)
            summary["bets_settled"] = None

    # Step 6d: refresh the player bio metadata the player detail page reads.
    # Cheap (~6s, no network — it re-walks the match JSONs just fetched above)
    # and it must run here: this file previously had no writer at all and sat
    # frozen from 2026-02-17, leaving every bio on /player stale and every
    # player who arrived after that date with no bio at all.
    if _player_meta_needs_refresh(
        DATA_DIR / "features" / "player_metadata.json",
        summary.get("matches_fetched"),
    ):
        log.info("Step 6d: Refreshing player metadata...")
        try:
            from scripts.data.build_player_metadata import main as _write_player_meta
            summary["player_metadata_refreshed"] = _write_player_meta() == 0
        except Exception as e:  # noqa: BLE001 — cosmetic bio refresh; must never
            # fail the matchday update that already wrote the parquets above.
            log.error("Failed to refresh player metadata: %s", e)
            summary["player_metadata_refreshed"] = False

    elapsed = round(time.time() - t0, 1)
    summary["elapsed_seconds"] = elapsed

    log.info("=" * 60)
    log.info("MATCHDAY UPDATE COMPLETE in %.1fs", elapsed)
    log.info("  Matches detected: %d", summary["new_matches_detected"])
    log.info("  Matches fetched:  %d", summary.get("matches_fetched", 0))
    log.info("  matches.parquet:  +%d rows", summary.get("matches_parquet_added", 0))
    if summary.get("fallback_used"):
        log.info("  (used results.json fallback — no Sofascore stats)")
    for name, count in parquet_counts.items():
        log.info("  %s: +%d rows", name, count)
    log.info("=" * 60)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Automated post-matchday update: detect → fetch → merge → rebuild",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.data.matchday_updater                   # Auto-detect + fetch + standings
  python -m scripts.data.matchday_updater --full             # + features + settle bets
  python -m scripts.data.matchday_updater --dry-run          # Detect only
  python -m scripts.data.matchday_updater --refresh-fixtures # Force refresh fixtures cache
  python -m scripts.data.matchday_updater --features         # + rebuild features
  python -m scripts.data.matchday_updater --season 2024-2025 # Backfill season
        """,
    )
    parser.add_argument("--season", default=None,
                        help="Season to process (default: auto-detected from date)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Detect new matches without fetching")
    parser.add_argument("--refresh-fixtures", action="store_true",
                        help="Force refresh fixtures cache from Sofascore")
    parser.add_argument("--features", action="store_true",
                        help="Rebuild features.parquet after merge")
    parser.add_argument("--standings", action="store_true", default=True,
                        help="Regenerate standings (default: on)")
    parser.add_argument("--no-standings", action="store_true",
                        help="Skip standings regeneration")
    parser.add_argument("--settle", action="store_true",
                        help="Run bet settlement via results_fetcher")
    parser.add_argument("--full", action="store_true",
                        help="Full update: features + standings + settle bets")
    parser.add_argument("--league", type=str, default=None,
                        help="Comma-separated leagues to update (default: serie_a,premier_league). "
                             "E.g. --league serie_a or --league serie_a,premier_league")
    parser.add_argument("--backfill", action="store_true",
                        help="Diff finished fixtures against matches.parquet (not the stat "
                             "parquets) and rebuild missing ground-truth rows from match "
                             "JSONs already on disk. No network. Honors --season, --league, "
                             "--dry-run. Idempotent.")

    parser.add_argument("--backfill-referees", action="store_true",
                        help="Fill the referee of played rows that have none from ESPN "
                             "(summary officials) and normalise '' to null. Honors "
                             "--season, --league, --dry-run. Idempotent.")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    rebuild_features = args.features or args.full
    settle_bets = args.settle or args.full
    regenerate_standings = not args.no_standings

    # Parse leagues
    leagues = None
    if args.league:
        leagues = [l.strip() for l in args.league.split(",") if l.strip()]

    if args.backfill_referees:
        results = [
            backfill_referees(season=args.season, league=lg, dry_run=args.dry_run)
            for lg in (leagues or ["serie_a", "premier_league"])
        ]
        print(json.dumps(results, indent=2, default=str))
        return

    if args.backfill:
        results = [
            backfill_matches_parquet(season=args.season, league=lg, dry_run=args.dry_run)
            for lg in (leagues or ["serie_a", "premier_league"])
        ]
        print(json.dumps(results, indent=2, default=str))
        return

    summary = run_matchday_update(
        season=args.season,
        rebuild_features=rebuild_features,
        regenerate_standings=regenerate_standings,
        settle_bets=settle_bets,
        dry_run=args.dry_run,
        force_refresh=args.refresh_fixtures,
        leagues=leagues,
    )

    # Print summary as JSON for programmatic use
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
