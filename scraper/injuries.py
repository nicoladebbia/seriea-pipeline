"""Scrape injury data from Transfermarkt and ESPN.

This module scrapes current injury information for Serie A players to enable
accurate predictions for upcoming matches. Knowing which key players are
injured is critical for prediction accuracy.

Sources:
  - Transfermarkt: Team squad pages with injury status (primary)
  - ESPN: Injury reports (fallback — only used when TM fails for a team)

Data stored in: data/external/injuries/

Performance: Uses concurrent fetching with per-domain rate limiting.
20 teams completes in ~30-60s (was ~4-20min sequential).
"""

from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config.settings import DATA_DIR
from scraper.transfermarkt import TM_HEADERS

log = logging.getLogger(__name__)

INJURIES_DIR = DATA_DIR / "external" / "injuries"
INJURIES_DIR.mkdir(parents=True, exist_ok=True)

# Transfermarkt base URL
TM_BASE = "https://www.transfermarkt.com"

# ESPN base URL for injuries
# ESPN changed URL format in late 2025 — try both old and new patterns
ESPN_BASE_NEW = "https://www.espn.com/soccer/team/injuries/_/id"
ESPN_BASE_OLD = "https://www.espn.com/soccer/team/injuries"
ESPN_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Referer": "https://www.espn.com/soccer/",
}

# Serie A team IDs on Transfermarkt (team_name -> (slug, verein_id))
SERIE_A_TEAMS_TM: dict[str, tuple[str, int]] = {
    "Atalanta": ("atalanta-bc", 800),
    "Bologna": ("fc-bologna", 1025),
    "Cagliari": ("cagliari-calcio", 1390),
    "Como": ("como-1907", 1047),
    "Cremonese": ("us-cremonese", 3898),
    "Empoli": ("fc-empoli", 749),
    "Fiorentina": ("acf-fiorentina", 430),
    "Genoa": ("genoa-cfc", 252),
    "Inter": ("inter-milan", 46),
    "Juventus": ("juventus-fc", 506),
    "Lazio": ("ss-lazio", 398),
    "Lecce": ("us-lecce", 1005),
    "Milan": ("ac-milan", 5),
    "Napoli": ("ssc-napoli", 6195),
    "Parma": ("parma-calcio-1913", 130),
    "Pisa": ("ac-pisa-1909", 1029),
    "Roma": ("as-roma", 12),
    "Sassuolo": ("us-sassuolo-calcio", 6574),
    "Torino": ("torino-fc", 416),
    "Udinese": ("udinese-calcio", 410),
    "Verona": ("hellas-verona", 276),
}

# Premier League team IDs on Transfermarkt (team_name -> (slug, verein_id))
EPL_TEAMS_TM: dict[str, tuple[str, int]] = {
    "Arsenal": ("arsenal-fc", 11),
    "Aston Villa": ("aston-villa", 405),
    "Bournemouth": ("afc-bournemouth", 989),
    "Brentford": ("brentford-fc", 1148),
    "Brighton": ("brighton-amp-hove-albion", 1237),
    "Chelsea": ("chelsea-fc", 631),
    "Crystal Palace": ("crystal-palace", 873),
    "Everton": ("everton-fc", 29),
    "Fulham": ("fulham-fc", 931),
    "Ipswich Town": ("ipswich-town", 677),
    "Leicester City": ("leicester-city", 1003),
    "Liverpool": ("liverpool-fc", 31),
    "Manchester City": ("manchester-city", 281),
    "Manchester United": ("manchester-united", 985),
    "Newcastle United": ("newcastle-united", 762),
    "Nottingham Forest": ("nottingham-forest", 703),
    "Southampton": ("southampton-fc", 180),
    "Sunderland": ("sunderland-afc", 289),
    "Tottenham Hotspur": ("tottenham-hotspur", 148),
    "West Ham United": ("west-ham-united", 379),
    "Wolverhampton Wanderers": ("wolverhampton-wanderers", 543),
}

# League -> TM mapping registry
LEAGUE_TEAMS_TM: dict[str, dict[str, tuple[str, int]]] = {
    "serie_a": SERIE_A_TEAMS_TM,
    "premier_league": EPL_TEAMS_TM,
}

# ESPN team IDs for Serie A (corrected — no duplicates)
SERIE_A_TEAMS_ESPN: dict[str, int] = {
    "Atalanta": 102,
    "Bologna": 107,
    "Cagliari": 104,
    "Como": 3276,
    "Cremonese": 3288,
    "Empoli": 108,
    "Fiorentina": 109,
    "Genoa": 110,
    "Inter": 114,
    "Juventus": 111,
    "Lazio": 115,
    "Lecce": 117,
    "Milan": 103,
    "Monza": 18294,
    "Napoli": 492,
    "Parma": 120,
    "Pisa": 4965,
    "Roma": 122,
    "Sassuolo": 4960,
    "Torino": 123,
    "Udinese": 124,
    "Venezia": 4706,
    "Verona": 1049,
}


# ---------------------------------------------------------------------------
# Rate limiter — ensures minimum delay between requests to same domain
# ---------------------------------------------------------------------------
class _RateLimiter:
    """Thread-safe per-domain rate limiter."""

    def __init__(self, min_interval: float = 2.5):
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._last_request: float = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request
            wait_time = max(0, self._min_interval - elapsed)
        if wait_time > 0:
            time.sleep(wait_time)
        with self._lock:
            self._last_request = time.monotonic()


# Shared rate limiters (one per domain)
_tm_limiter = _RateLimiter(min_interval=2.5)
_espn_limiter = _RateLimiter(min_interval=1.0)


@dataclass
class PlayerInjury:
    """Represents a player injury record."""
    player_name: str
    team: str
    injury_type: str  # "Muscle", "Knee", "Ankle", "Illness", etc.
    start_date: Optional[date]
    expected_return: Optional[date]
    is_currently_out: bool
    source: str  # "transfermarkt", "espn"
    scraped_at: datetime


def _parse_date(date_str: str) -> Optional[date]:
    """Parse various date formats from Transfermarkt."""
    if not date_str or date_str == "-" or date_str.lower() == "unknown":
        return None

    # Try different formats
    formats = ["%b %d, %Y", "%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue

    # Try relative dates like "2 weeks"
    match = re.search(r"(\d+)\s*(day|week|month)", date_str.lower())
    if match:
        num = int(match.group(1))
        unit = match.group(2)
        days = num if unit == "day" else num * 7 if unit == "week" else num * 30
        return date.today() + timedelta(days=days)

    return None


def _parse_tm_html(html: str, team: str) -> list[PlayerInjury]:
    """Parse Transfermarkt squad page HTML for injury data.

    Separated from HTTP logic to make testing easier.
    """
    soup = BeautifulSoup(html, "html.parser")
    injuries = []

    injury_spans = soup.select("span.verletzt-table")

    for span in injury_spans:
        title = span.get("title", "")
        if not title:
            continue

        parent_a = span.find_parent("a")
        if not parent_a:
            continue
        player_name = parent_a.get_text(strip=True).strip()

        injury_type = "Unknown"
        expected_return = None

        if " - " in title:
            parts = title.split(" - ", 1)
            injury_type = parts[0].strip()
            return_info = parts[1].strip()
            date_match = re.search(r"(\d{2}/\d{2}/\d{4})", return_info)
            if date_match:
                expected_return = _parse_date(date_match.group(1))
        else:
            injury_type = title.strip()

        injuries.append(PlayerInjury(
            player_name=player_name,
            team=team,
            injury_type=injury_type,
            start_date=None,
            expected_return=expected_return,
            is_currently_out=True,
            source="transfermarkt",
            scraped_at=datetime.now(),
        ))

    return injuries


def scrape_team_injuries_tm(team: str, max_retries: int = 1) -> list[PlayerInjury]:
    """Scrape injury data for a team from Transfermarkt.

    Uses rate limiter to space requests and retries once on failure.

    Args:
        team: Team name (must be in SERIE_A_TEAMS_TM)
        max_retries: Number of retries on failure (default 1)

    Returns:
        List of PlayerInjury objects for currently injured players
    """
    # Look up team in any league mapping
    tm_entry = None
    for league_map in LEAGUE_TEAMS_TM.values():
        if team in league_map:
            tm_entry = league_map[team]
            break
    if tm_entry is None:
        log.warning(f"Team {team} not found in any Transfermarkt mapping")
        return []

    slug, verein_id = tm_entry
    url = f"{TM_BASE}/{slug}/kader/verein/{verein_id}/plus/1"

    for attempt in range(1 + max_retries):
        try:
            _tm_limiter.wait()
            log.info(f"Scraping injuries for {team} from Transfermarkt"
                     f"{' (retry)' if attempt > 0 else ''}")
            resp = requests.get(url, headers=TM_HEADERS, timeout=15)
            resp.raise_for_status()

            injuries = _parse_tm_html(resp.text, team)
            log.info(f"Found {len(injuries)} injured players for {team}")
            return injuries

        except requests.RequestException as e:
            log.warning(f"TM request failed for {team} (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                time.sleep(3)
            continue

    log.error(f"Failed to scrape injuries for {team} after {1 + max_retries} attempts")
    return []


def scrape_team_injuries_espn(team: str) -> list[PlayerInjury]:
    """Scrape injury data for a team from ESPN.

    ESPN provides a cleaner injury table with expected return dates.

    Args:
        team: Team name (must be in SERIE_A_TEAMS_ESPN)

    Returns:
        List of PlayerInjury objects
    """
    if team not in SERIE_A_TEAMS_ESPN:
        log.warning(f"Team {team} not found in ESPN mapping")
        return []

    espn_id = SERIE_A_TEAMS_ESPN[team]

    # Try new URL format first, fall back to old
    urls_to_try = [
        f"{ESPN_BASE_NEW}/{espn_id}/league/ita.1",
        f"{ESPN_BASE_NEW}/{espn_id}",
        f"{ESPN_BASE_OLD}/_/id/{espn_id}",
    ]

    resp = None
    for url in urls_to_try:
        try:
            _espn_limiter.wait()
            log.info(f"Trying ESPN injuries for {team}: {url}")
            resp = requests.get(url, headers=ESPN_HEADERS, timeout=15)
            if resp.status_code == 200:
                break
            log.debug(f"ESPN returned {resp.status_code} for {url}")
            resp = None
        except requests.RequestException:
            continue

    if not resp or resp.status_code != 200:
        log.warning(f"ESPN injuries unavailable for {team} (all URL formats failed)")
        return []

    try:

        soup = BeautifulSoup(resp.text, "html.parser")
        injuries = []

        # Find injury table rows
        injury_rows = soup.select("table.Table tbody tr")

        for row in injury_rows:
            cells = row.select("td")
            if len(cells) < 3:
                continue

            player_name = cells[0].text.strip()
            injury_type = cells[1].text.strip() if len(cells) > 1 else "Unknown"
            status = cells[2].text.strip() if len(cells) > 2 else ""

            # Parse expected return from status
            expected_return = _parse_date(status)
            is_out = "out" in status.lower() or "day-to-day" in status.lower()

            injuries.append(PlayerInjury(
                player_name=player_name,
                team=team,
                injury_type=injury_type,
                start_date=None,
                expected_return=expected_return,
                is_currently_out=is_out,
                source="espn",
                scraped_at=datetime.now(),
            ))

        log.info(f"Found {len(injuries)} injured players for {team} (ESPN)")
        return injuries

    except requests.RequestException as e:
        log.error(f"Failed to scrape ESPN injuries for {team}: {e}")
        return []


def scrape_all_injuries(delay: float = 2.5, league: str = "serie_a") -> pd.DataFrame:
    """Scrape injury data for all teams in a given league.

    Uses concurrent fetching with per-domain rate limiting for speed.
    ESPN is only tried as fallback for teams where Transfermarkt failed.

    Args:
        delay: Minimum seconds between same-domain requests (default 2.5).
               Passed for API compat but the rate limiter handles actual timing.
        league: League key — "serie_a" or "premier_league".

    Returns:
        DataFrame with all current injuries
    """
    tm_mapping = LEAGUE_TEAMS_TM.get(league, SERIE_A_TEAMS_TM)
    current_teams = list(tm_mapping.keys())

    # Override rate limiter interval if caller passed a custom delay
    _tm_limiter._min_interval = max(delay, 2.0)

    # Phase 1: Concurrent Transfermarkt fetch (3 workers, rate-limited)
    tm_results: dict[str, list[PlayerInjury]] = {}
    t0 = time.monotonic()

    log.info(f"Phase 1: Fetching injuries from Transfermarkt for {len(current_teams)} teams (concurrent)")

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="tm_injury") as pool:
        futures = {
            pool.submit(scrape_team_injuries_tm, team): team
            for team in current_teams
        }
        for future in as_completed(futures):
            team = futures[future]
            try:
                injuries = future.result()
                tm_results[team] = injuries
            except Exception as e:
                log.error(f"Unexpected error scraping {team}: {e}")
                tm_results[team] = []

    tm_elapsed = time.monotonic() - t0
    tm_total = sum(len(v) for v in tm_results.values())
    tm_failed = [t for t, v in tm_results.items() if not v]
    log.info(f"Phase 1 complete: {tm_total} injuries from {len(current_teams)} teams in {tm_elapsed:.1f}s")

    # Phase 2: ESPN fallback — only for teams where TM returned nothing
    espn_extras: list[PlayerInjury] = []
    if tm_failed:
        log.info(f"Phase 2: ESPN fallback for {len(tm_failed)} teams: {tm_failed}")
        for team in tm_failed:
            try:
                espn_injuries = scrape_team_injuries_espn(team)
                espn_extras.extend(espn_injuries)
            except Exception as e:
                log.debug(f"ESPN fallback failed for {team}: {e}")
        log.info(f"Phase 2 complete: {len(espn_extras)} injuries from ESPN fallback")
    else:
        log.info("Phase 2: Skipped — all teams covered by Transfermarkt")

    # Merge results
    all_injuries: list[PlayerInjury] = []
    for team_injuries in tm_results.values():
        all_injuries.extend(team_injuries)
    all_injuries.extend(espn_extras)

    total_elapsed = time.monotonic() - t0
    log.info(f"Total: {len(all_injuries)} injuries in {total_elapsed:.1f}s")

    if not all_injuries:
        return pd.DataFrame()

    df = pd.DataFrame([
        {
            "player_name": inj.player_name,
            "team": inj.team,
            "injury_type": inj.injury_type,
            "start_date": inj.start_date,
            "expected_return": inj.expected_return,
            "is_currently_out": inj.is_currently_out,
            "source": inj.source,
            "scraped_at": inj.scraped_at,
        }
        for inj in all_injuries
    ])

    # Save to cache (league-specific filename for non-Serie A)
    suffix = f"_{league}" if league != "serie_a" else ""
    cache_path = INJURIES_DIR / f"injuries{suffix}_{date.today().isoformat()}.parquet"
    df.to_parquet(cache_path, index=False)
    log.info(f"Saved {len(df)} injuries to {cache_path}")

    return df


def get_current_injuries(auto_scrape: bool = False, league: str = "serie_a") -> pd.DataFrame:
    """Get the most recent injury data from cache.

    Checks for cached data from today or recent days.
    Does NOT auto-scrape to avoid blocking web requests.

    Args:
        auto_scrape: If True, scrape fresh data when no cache exists (slow)
        league: League key — "serie_a" or "premier_league".

    Returns:
        DataFrame with current injuries (may be empty if no cache)
    """
    suffix = f"_{league}" if league != "serie_a" else ""

    today_cache = INJURIES_DIR / f"injuries{suffix}_{date.today().isoformat()}.parquet"

    if today_cache.exists():
        log.debug(f"Loading cached injuries from {today_cache}")
        return pd.read_parquet(today_cache)

    # Check for recent cache — warn if stale, reject if too old
    for days_ago in range(1, 8):
        old_date = date.today() - timedelta(days=days_ago)
        old_cache = INJURIES_DIR / f"injuries{suffix}_{old_date.isoformat()}.parquet"
        if old_cache.exists():
            if days_ago > 2:
                log.warning("Injury cache is %d days old — data may be stale", days_ago)
            if days_ago >= 7:
                log.warning("Injury cache is %d days old — too stale, returning empty", days_ago)
                break  # fall through to auto_scrape or empty return
            log.debug(f"Loading cached injuries from {old_cache} ({days_ago}d old)")
            return pd.read_parquet(old_cache)

    # No recent cache - only scrape if explicitly requested
    if auto_scrape:
        log.info(f"No recent {league} injury data found, scraping fresh data")
        return scrape_all_injuries(league=league)
    else:
        log.debug(f"No {league} injury cache available, returning empty DataFrame")
        return pd.DataFrame()


def get_injured_players(team: str, match_date: Optional[date] = None) -> list[str]:
    """Get list of injured player names for a specific team.

    Args:
        team: Team name
        match_date: Optional date to filter expected returns

    Returns:
        List of player names who are injured/unavailable
    """
    df = get_current_injuries()

    if df.empty:
        return []

    team_injuries = df[df["team"] == team]

    if match_date and "expected_return" in team_injuries.columns:
        # Filter to players still expected to be out on match_date
        team_injuries = team_injuries[
            team_injuries["expected_return"].isna() |
            (pd.to_datetime(team_injuries["expected_return"]).dt.date > match_date)
        ]

    return team_injuries["player_name"].tolist()


def get_key_absences(
    team: str,
    key_players: list[str],
    match_date: Optional[date] = None
) -> list[str]:
    """Check which key players are injured for a team.

    Args:
        team: Team name
        key_players: List of key player names to check
        match_date: Optional match date

    Returns:
        List of key players who are currently injured
    """
    injured = get_injured_players(team, match_date)
    injured_lower = [p.lower() for p in injured]

    key_absences = []
    for player in key_players:
        if player.lower() in injured_lower:
            key_absences.append(player)

    return key_absences


if __name__ == "__main__":
    # Test the scraper
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("Testing injury scraper (concurrent mode)...")
    t0 = time.monotonic()
    df = scrape_all_injuries()
    elapsed = time.monotonic() - t0

    if not df.empty:
        print(f"\nFound {len(df)} total injuries across Serie A in {elapsed:.1f}s")
        print(f"\nPer-team breakdown:")
        print(df.groupby("team")["player_name"].count().sort_values(ascending=False).to_string())
        print(f"\nSample injuries:")
        print(df.head(10).to_string())
    else:
        print(f"No injuries found in {elapsed:.1f}s (check logs for errors)")
