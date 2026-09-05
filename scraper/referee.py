"""Scrape referee-per-match data from worldfootball.net.

Builds a mapping of (match_date, home_team, away_team) -> referee_name
for all Serie A seasons. This fills the 'referee' column that is missing
from football-data.co.uk historical imports.

Strategy:
  1. For each season, fetch the referees listing page to get all ref slugs
  2. For each referee, fetch their matches-as-referee page for that season
  3. Parse match rows (date, home-away, result, cards)
  4. Save as referee_assignments.parquet
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config.settings import DATA_DIR
from config.team_names import normalize_team as _canonical_normalize

log = logging.getLogger(__name__)

WF_BASE = "https://www.worldfootball.net"
WF_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Worldfootball.net league slug mapping
WF_LEAGUE_SLUGS: dict[str, str] = {
    "serie_a": "ita-serie-a",
    "premier_league": "eng-premier-league",
    "la_liga": "esp-primera-division",
    "bundesliga": "bundesliga",
    "ligue_1": "fra-ligue-1",
}

# Worldfootball.net competition codes
WF_COMP_CODES: dict[str, str] = {
    "serie_a": "co111",
    "premier_league": "co13",
    "la_liga": "co16",
    "bundesliga": "co14",
    "ligue_1": "co15",
}


def _wf_comp_code(league: str) -> str:
    """Get worldfootball.net competition code for a league."""
    return WF_COMP_CODES.get(league, "co111")


# worldfootball.net season IDs per league
# Key: (league, season) -> season_id
WF_SEASON_IDS_BY_LEAGUE: dict[str, dict[str, str]] = {
    "serie_a": {
        "2017-2018": "se23947",
        "2018-2019": "se28594",
        "2019-2020": "se31821",
        "2020-2021": "se36220",
        "2021-2022": "se39347",
        "2022-2023": "se45806",
        "2023-2024": "se52577",
        "2024-2025": "se74735",
        "2025-2026": "se95481",
    },
    # EPL season IDs to be backfilled. To find a season's ID:
    #   1. Fetch https://www.worldfootball.net/all_matches/eng-premier-league-{season}/
    #      where season is e.g. 2024-2025
    #   2. Grep the response for `se\d{5,}` — the first match is the season ID.
    #   3. Add the (season, id) here; the scraper will pick it up next run.
    "premier_league": {
        # Discovered via _discover_season_id (worldfootball.net). Earlier seasons
        # to be backfilled on first interactive run that has network access.
        "2025-2026": "se74714",
    },
}

# Backward-compatible alias for Serie A season IDs
WF_SEASON_IDS: dict[str, str] = WF_SEASON_IDS_BY_LEAGUE["serie_a"]


def _resolve_wf_league(league: str) -> tuple[str, dict[str, str]]:
    """Resolve league to WF slug and season ID mapping.

    Returns:
        Tuple of (league_slug, season_ids_dict)
    """
    slug = WF_LEAGUE_SLUGS.get(league)
    if not slug:
        raise ValueError(
            f"Unknown league '{league}'. Valid: {', '.join(WF_LEAGUE_SLUGS)}"
        )
    season_ids = WF_SEASON_IDS_BY_LEAGUE.get(league, {})
    return slug, season_ids


def _get_season_id(league: str, season: str) -> str | None:
    """Get the WF season ID for (league, season). Tries hardcoded map first;
    falls back to runtime discovery and caches the result in WF_SEASON_IDS_BY_LEAGUE
    so subsequent calls in the same process don't re-fetch."""
    league_slug, season_ids = _resolve_wf_league(league)
    if season in season_ids:
        return season_ids[season]
    discovered = _discover_season_id(league_slug, season)
    if discovered:
        WF_SEASON_IDS_BY_LEAGUE.setdefault(league, {})[season] = discovered
        log.info("Discovered %s %s season ID: %s — consider hardcoding for speed",
                 league, season, discovered)
        return discovered
    return None

# Team name mapping: worldfootball.net name -> our internal name
# This handles the most common variations
WF_TEAM_MAP: dict[str, str] = {
    "AC Milan": "Milan",
    "AC Monza": "Monza",
    "ACF Fiorentina": "Fiorentina",
    "AS Roma": "Roma",
    "Atalanta": "Atalanta",
    "Benevento": "Benevento",
    "Bologna": "Bologna",
    "Bologna FC": "Bologna",
    "Brescia": "Brescia",
    "Brescia Calcio": "Brescia",
    "Cagliari": "Cagliari",
    "Cagliari Calcio": "Cagliari",
    "ChievoVerona": "Chievo",
    "Chievo Verona": "Chievo",
    "Como 1907": "Como",
    "Crotone": "Crotone",
    "FC Crotone": "Crotone",
    "Empoli": "Empoli",
    "FC Empoli": "Empoli",
    "Frosinone": "Frosinone",
    "Frosinone Calcio": "Frosinone",
    "Genoa": "Genoa",
    "Genoa CFC": "Genoa",
    "Hellas Verona": "Verona",
    "Inter": "Inter",
    "FC Internazionale": "Inter",
    "Juventus": "Juventus",
    "Lazio": "Lazio",
    "Lazio Roma": "Lazio",
    "Lecce": "Lecce",
    "US Lecce": "Lecce",
    "Milan": "Milan",
    "Monza": "Monza",
    "Napoli": "Napoli",
    "SSC Napoli": "Napoli",
    "Parma": "Parma",
    "Parma Calcio": "Parma",
    "Roma": "Roma",
    "SS Lazio": "Lazio",
    "Salernitana": "Salernitana",
    "US Salernitana": "Salernitana",
    "Sampdoria": "Sampdoria",
    "UC Sampdoria": "Sampdoria",
    "Sassuolo": "Sassuolo",
    "US Sassuolo": "Sassuolo",
    "SPAL": "SPAL",
    "SPAL 2013": "SPAL",
    "Spezia": "Spezia",
    "Spezia Calcio": "Spezia",
    "Torino": "Torino",
    "Torino FC": "Torino",
    "Udinese": "Udinese",
    "Udinese Calcio": "Udinese",
    "Venezia": "Venezia",
    "Venezia FC": "Venezia",
    "Verona": "Verona",
    "US Cremonese": "Cremonese",
    "Cremonese": "Cremonese",
    "Pisa": "Pisa",
    "AC Pisa": "Pisa",
    "Pisa SC": "Pisa",
}

REFEREE_DIR = DATA_DIR / "external" / "referee"


def _normalize_team(name: str) -> str:
    """Map worldfootball.net team name to our internal name."""
    name = name.strip()
    if name in WF_TEAM_MAP:
        return WF_TEAM_MAP[name]
    # Try canonical normalize_team from config/team_names.py
    canonical = _canonical_normalize(name)
    if canonical != name:
        return canonical
    # Fallback: try partial matching
    for wf_name, internal in WF_TEAM_MAP.items():
        if wf_name.lower() in name.lower() or name.lower() in wf_name.lower():
            return internal
    log.warning("Unknown team name from worldfootball.net: '%s'", name)
    return name


try:
    from curl_cffi import requests as _cffi_requests
    _CFFI_SESSION = _cffi_requests.Session(impersonate="chrome120")
    _CFFI_SESSION.headers.update(WF_HEADERS)
    _HAS_CFFI = True
except ImportError:
    _HAS_CFFI = False


def _fetch(url: str) -> requests.Response | None:
    """Fetch with retry and delay. Uses curl_cffi Chrome impersonation to bypass 403s."""
    for attempt in range(3):
        try:
            if _HAS_CFFI:
                resp = _CFFI_SESSION.get(url, timeout=20)
            else:
                resp = requests.get(url, headers=WF_HEADERS, timeout=20)
            resp.raise_for_status()
            return resp
        except Exception as e:
            log.warning("Attempt %d failed for %s: %s", attempt + 1, url, e)
            time.sleep(3 * (attempt + 1))
    return None


def _discover_season_id(league_slug: str, season: str) -> str | None:
    """Discover a worldfootball.net season ID at runtime by parsing the
    schedule page. Slow, so callers should hardcode the result for repeated use.
    Returns the first se##### token found, or None on failure.
    """
    import re
    url = f"{WF_BASE}/all_matches/{league_slug}-{season}/"
    resp = _fetch(url)
    if not resp:
        return None
    m = re.search(r"se\d{5,}", resp.text)
    return m.group(0) if m else None


def get_season_referees(season: str, league: str = "serie_a") -> list[tuple[str, str, str]]:
    """Get list of referees for a league season.

    Args:
        season: Season string (e.g., "2024-2025")
        league: League identifier (default: "serie_a").
                Valid values: serie_a, premier_league, la_liga, bundesliga, ligue_1

    Returns list of (referee_name, person_id, slug).
    """
    league_slug, _ = _resolve_wf_league(league)
    season_id = _get_season_id(league, season)
    if not season_id:
        log.warning(
            "No season ID for %s (%s); auto-discovery failed. "
            "Add it manually under WF_SEASON_IDS_BY_LEAGUE['%s']['%s'] in scraper/referee.py.",
            season, league, league, season
        )
        return []

    url = f"{WF_BASE}/referees/{league_slug}-{season}/"
    log.info("Fetching referee list for %s: %s", season, url)

    resp = _fetch(url)
    if not resp:
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    refs = []

    # Find all person links in the first table (main referees)
    tables = soup.select("table")
    if not tables:
        return []

    for row in tables[0].select("tr"):
        for a in row.select("a[href*='/person/']"):
            href = a.get("href", "")
            name = a.get_text(strip=True)
            if not name:
                continue
            # Parse person ID and slug from href like /person/pe228195/davide-massa/
            m = re.search(r"/person/(pe\d+)/([^/]+)/", href)
            if m:
                refs.append((name, m.group(1), m.group(2)))
                break  # Only first link per row

    log.info("Found %d referees for %s", len(refs), season)
    return refs


def scrape_referee_matches(
    referee_name: str,
    person_id: str,
    slug: str,
    season: str,
    league: str = "serie_a",
) -> list[dict]:
    """Scrape all matches officiated by a referee in a season.

    Args:
        referee_name: Name of the referee
        person_id: Worldfootball.net person ID (e.g., "pe228195")
        slug: URL slug for the referee
        season: Season string (e.g., "2024-2025")
        league: League identifier (default: "serie_a")

    Returns list of dicts with: match_date, home_team, away_team, referee,
    yellows, second_yellows, reds, matchweek.
    """
    league_slug, _ = _resolve_wf_league(league)
    season_id = _get_season_id(league, season)
    if not season_id:
        return []

    # Derive the competition path segment from the league slug
    # e.g., "ita-serie-a" -> "serie-a", "eng-premier-league" -> "premier-league"
    comp_name = league_slug.split("-", 1)[1] if "-" in league_slug else league_slug

    url = (
        f"{WF_BASE}/person/{person_id}/{slug}"
        f"/{_wf_comp_code(league)}/{comp_name}/{season_id}/{season}/matches-as-referee/"
    )

    resp = _fetch(url)
    if not resp:
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    tables = soup.select("table")
    if not tables:
        return []

    rows = []
    for tr in tables[0].select("tr")[1:]:  # skip header
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) < 4:
            continue

        # cells: [Round, Date, Match, Result, Yellows, Yellow-Reds, Reds]
        round_str = cells[0]
        date_str = cells[1]
        match_str = cells[2]
        result_str = cells[3]

        # Parse matchweek
        mw_match = re.search(r"(\d+)", round_str)
        matchweek = int(mw_match.group(1)) if mw_match else None

        # Parse date (DD.MM.YYYY)
        try:
            match_date = datetime.strptime(date_str, "%d.%m.%Y").strftime("%Y-%m-%d")
        except ValueError:
            continue

        # Parse teams from "Home-Away" (note: hyphen separator)
        # Handle team names that contain hyphens by splitting on the last
        # occurrence that looks like a team separator
        teams = match_str.split("-")
        if len(teams) < 2:
            continue

        # Most team names don't have hyphens, but some might
        # Try to find the split point by matching known teams
        home_team = None
        away_team = None
        for i in range(1, len(teams)):
            candidate_home = "-".join(teams[:i]).strip()
            candidate_away = "-".join(teams[i:]).strip()
            h = _normalize_team(candidate_home)
            a = _normalize_team(candidate_away)
            if h != candidate_home or a != candidate_away:
                # At least one matched a known team
                home_team = h
                away_team = a
                break

        if not home_team:
            # Fallback: simple split on first hyphen
            home_team = _normalize_team(teams[0].strip())
            away_team = _normalize_team("-".join(teams[1:]).strip())

        # Parse cards
        yellows = _parse_card_count(cells[4]) if len(cells) > 4 else 0
        second_yellows = _parse_card_count(cells[5]) if len(cells) > 5 else 0
        reds = _parse_card_count(cells[6]) if len(cells) > 6 else 0

        rows.append({
            "match_date": match_date,
            "home_team": home_team,
            "away_team": away_team,
            "referee": referee_name,
            "matchweek": matchweek,
            "ref_yellows": yellows,
            "ref_second_yellows": second_yellows,
            "ref_reds": reds,
            "season": season,
        })

    return rows


def _parse_card_count(text: str) -> int:
    """Parse a card count cell, handling '-' for zero."""
    text = text.strip()
    if text == "-" or not text:
        return 0
    try:
        return int(text)
    except ValueError:
        return 0


def scrape_all_referee_assignments(
    seasons: list[str] | None = None,
    league: str = "serie_a",
) -> pd.DataFrame:
    """Scrape referee assignments for all specified seasons.

    Args:
        seasons: List of seasons to scrape (default: all available for the league)
        league: League identifier (default: "serie_a").
                Valid values: serie_a, premier_league, la_liga, bundesliga, ligue_1

    Returns DataFrame with columns:
        match_date, home_team, away_team, referee, matchweek,
        ref_yellows, ref_second_yellows, ref_reds, season
    """
    # League-specific cache to avoid returning Serie A data for EPL queries
    cache_path = REFEREE_DIR / f"referee_assignments_{league}.parquet"
    if cache_path.exists():
        log.info("Loading cached referee assignments from %s", cache_path)
        return pd.read_parquet(cache_path)

    _, season_ids = _resolve_wf_league(league)
    if seasons is None:
        seasons = list(season_ids.keys())

    all_rows: list[dict] = []

    for season in seasons:
        log.info("=== Scraping referees for season %s (%s) ===", season, league)
        refs = get_season_referees(season, league=league)
        time.sleep(3)

        for ref_name, person_id, slug in refs:
            log.info("  Fetching matches for %s (%s)", ref_name, season)
            matches = scrape_referee_matches(ref_name, person_id, slug, season, league=league)
            all_rows.extend(matches)
            log.info("    %d matches found", len(matches))
            time.sleep(3)  # Respectful delay

    df = pd.DataFrame(all_rows)

    # Deduplicate (same match might appear for VAR refs or duplicates)
    if not df.empty:
        dedup_cols = ["match_date", "home_team", "away_team"]
        if "season" in df.columns:
            dedup_cols.insert(0, "season")
        df = df.drop_duplicates(
            subset=dedup_cols,
            keep="first",
        )

    if df.empty:
        # Never persist an empty cache: on 2026-08-31 worldfootball had not
        # published 2026-27, this wrote 0-row files for both leagues, and the
        # exists() short-circuit above then served 0 rows on every later call.
        log.warning("No referee assignments scraped for %s — not writing %s", league, cache_path)
        return df
    REFEREE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    log.info("Saved %d referee assignments to %s", len(df), cache_path)
    return df


def merge_referee_with_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Merge scraped referee data into the matches table.

    Fills the 'referee' column where it's currently NaN.
    Also adds ref_yellows, ref_second_yellows, ref_reds from scraped data.
    """
    ref_path = REFEREE_DIR / "referee_assignments.parquet"
    if not ref_path.exists():
        log.info("No referee data found (run 'fetch-referees' first); skipping")
        return matches

    ref_df = pd.read_parquet(ref_path)
    if ref_df.empty:
        return matches

    df = matches.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    ref_df["match_date"] = pd.to_datetime(ref_df["match_date"], errors="coerce")

    # Create a lookup: (date_str, home, away) -> referee info
    ref_lookup = {}
    for _, row in ref_df.iterrows():
        key = (
            row["match_date"].strftime("%Y-%m-%d") if pd.notna(row["match_date"]) else None,
            row["home_team"],
            row["away_team"],
        )
        if key[0]:
            ref_lookup[key] = {
                "referee": row["referee"],
                "ref_yellows_scraped": row.get("ref_yellows", 0),
                "ref_second_yellows_scraped": row.get("ref_second_yellows", 0),
                "ref_reds_scraped": row.get("ref_reds", 0),
            }

    filled = 0
    for idx, row in df.iterrows():
        date_str = row["match_date"].strftime("%Y-%m-%d") if pd.notna(row["match_date"]) else None
        key = (date_str, row.get("home_team"), row.get("away_team"))

        if key in ref_lookup:
            info = ref_lookup[key]
            # Only fill referee if currently missing
            if pd.isna(row.get("referee")) or row.get("referee") == "":
                df.at[idx, "referee"] = info["referee"]
                filled += 1

    log.info("Filled %d/%d referee assignments from scraped data", filled, len(df))
    return df
