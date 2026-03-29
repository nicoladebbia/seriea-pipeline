"""Fast FBref scraper using botasaurus with browser reuse.

This scraper reuses a single browser session across multiple requests,
making it much faster than ScraperFC's default behavior of opening
a new browser for each request.

NOTE: Requires optional dependency: pip install botasaurus
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

# Optional dependency - botasaurus for fast scraping
try:
    from botasaurus.browser import browser, Driver
    HAS_BOTASAURUS = True
except ImportError:
    HAS_BOTASAURUS = False
    browser = None
    Driver = None

from config.settings import DATA_DIR, REQUEST_DELAY_SECONDS
from config.team_names import normalize_team

log = logging.getLogger(__name__)

# Output directories
HTML_DIR = DATA_DIR / "raw" / "html"
PARSED_DIR = DATA_DIR / "parsed"

# FBref URLs
FBREF_BASE = "https://fbref.com"

# League configurations
LEAGUE_CONFIG = {
    "seriea": {"comp_id": 11, "url_slug": "Serie-A", "display_name": "Serie A"},
    "epl": {"comp_id": 9, "url_slug": "Premier-League", "display_name": "Premier League"},
}

# Rate limiting
DELAY_SECONDS = max(REQUEST_DELAY_SECONDS, 8)  # At least 8 seconds for safety during backfill


class FBrefFastScraper:
    """Fast FBref scraper with browser reuse."""

    def __init__(self, headless: bool = True, league: str = "seriea"):
        """Initialize the scraper.

        Args:
            headless: Run browser in headless mode (faster, no UI)
            league: League key from LEAGUE_CONFIG (e.g., "seriea", "epl")

        Raises:
            ImportError: If botasaurus is not installed
        """
        if not HAS_BOTASAURUS:
            raise ImportError(
                "botasaurus is required for FBrefFastScraper. "
                "Install with: pip install botasaurus"
            )
        if league not in LEAGUE_CONFIG:
            raise ValueError(f"Unknown league '{league}'. Available: {list(LEAGUE_CONFIG.keys())}")
        self.headless = headless
        self.league = league
        self._cfg = LEAGUE_CONFIG[league]
        self._driver = None
        self._urls_scraped = 0

    def scrape_season_fixtures(self, season: str) -> pd.DataFrame:
        """Scrape fixtures/results for a season.

        Args:
            season: Season string like "2023-2024"

        Returns:
            DataFrame with match info including match URLs
        """
        slug = self._cfg["url_slug"]
        comp_id = self._cfg["comp_id"]
        url = f"{FBREF_BASE}/en/comps/{comp_id}/{season}/schedule/{season}-{slug}-Scores-and-Fixtures"
        log.info(f"Scraping fixtures from {url}")

        soup = self._get_soup(url)

        # Find the fixtures table
        table = soup.find("table", {"id": "sched_2024-2025_11_1"}) or \
                soup.find("table", {"id": lambda x: x and "sched_" in x})

        if not table:
            log.warning(f"No fixtures table found for {season}")
            return pd.DataFrame()

        matches = []
        for row in table.find("tbody").find_all("tr"):
            if "thead" in row.get("class", []):
                continue

            cells = row.find_all(["td", "th"])
            if len(cells) < 10:
                continue

            # Extract match URL
            score_cell = row.find("td", {"data-stat": "score"})
            match_url = ""
            if score_cell:
                link = score_cell.find("a")
                if link and link.get("href"):
                    match_url = FBREF_BASE + link["href"]

            match = {
                "matchweek": self._get_cell_text(row, "gameweek"),
                "date": self._get_cell_text(row, "date"),
                "time": self._get_cell_text(row, "time"),
                "home_team": self._get_cell_text(row, "home_team"),
                "away_team": self._get_cell_text(row, "away_team"),
                "score": self._get_cell_text(row, "score"),
                "match_url": match_url,
                "venue": self._get_cell_text(row, "venue"),
                "attendance": self._get_cell_text(row, "attendance"),
                "referee": self._get_cell_text(row, "referee"),
            }

            if match["home_team"] and match["away_team"]:
                matches.append(match)

        df = pd.DataFrame(matches)
        df["season"] = season
        log.info(f"Found {len(df)} matches for {season}")
        return df

    def scrape_match_report(self, match_url: str) -> dict[str, Any]:
        """Scrape a single match report with all details.

        Args:
            match_url: Full FBref match URL

        Returns:
            Dict with match data, player stats, shots, etc.
        """
        soup = self._get_soup(match_url)

        result = {
            "url": match_url,
            "home_team": "",
            "away_team": "",
            "home_score": None,
            "away_score": None,
            "home_xg": None,
            "away_xg": None,
            "player_stats": [],
            "shots": [],
        }

        # Get team names and scores from scorebox
        scorebox = soup.find("div", {"class": "scorebox"})
        if scorebox:
            teams = scorebox.find_all("div", {"itemprop": "performer"})
            if len(teams) >= 2:
                home_name = teams[0].find("a", {"itemprop": "name"})
                away_name = teams[1].find("a", {"itemprop": "name"})
                result["home_team"] = normalize_team(home_name.text.strip()) if home_name else ""
                result["away_team"] = normalize_team(away_name.text.strip()) if away_name else ""

            scores = scorebox.find_all("div", {"class": "score"})
            if len(scores) >= 2:
                try:
                    result["home_score"] = int(scores[0].text.strip())
                    result["away_score"] = int(scores[1].text.strip())
                except (ValueError, AttributeError) as e:
                    log.debug(f"Failed to parse match scores: {e}")

        # Get xG values
        for div in soup.find_all("div", {"class": "score_xg"}):
            xg_text = div.text.strip()
            try:
                xg = float(xg_text)
                if result["home_xg"] is None:
                    result["home_xg"] = xg
                else:
                    result["away_xg"] = xg
            except (ValueError, AttributeError) as e:
                log.debug(f"Failed to parse xG value: {e}")

        # Get player stats tables
        result["player_stats"] = self._parse_player_stats_tables(soup)

        # Get shots table
        result["shots"] = self._parse_shots_table(soup)

        return result

    def scrape_season_player_stats(self, season: str, stat_type: str = "standard") -> pd.DataFrame:
        """Scrape season-level player statistics.

        Args:
            season: Season string like "2023-2024"
            stat_type: Type of stats (standard, shooting, passing, etc.)

        Returns:
            DataFrame with player stats
        """
        stat_url_map = {
            "standard": "stats",
            "shooting": "shooting",
            "passing": "passing",
            "passing_types": "passing_types",
            "gca": "gca",
            "defense": "defense",
            "possession": "possession",
            "misc": "misc",
        }

        url_part = stat_url_map.get(stat_type, "stats")
        slug = self._cfg["url_slug"]
        comp_id = self._cfg["comp_id"]
        url = f"{FBREF_BASE}/en/comps/{comp_id}/{season}/{url_part}/{season}-{slug}-Stats"

        log.info(f"Scraping {stat_type} stats from {url}")
        soup = self._get_soup(url)

        # Find the stats table
        table = soup.find("table", {"id": lambda x: x and "stats_" in str(x)})
        if not table:
            log.warning(f"No {stat_type} stats table found for {season}")
            return pd.DataFrame()

        return self._table_to_df(table)

    def scrape_multiple_seasons(
        self,
        seasons: list[str],
        scrape_matches: bool = True,
        scrape_player_stats: bool = True,
    ) -> dict[str, Any]:
        """Batch scrape multiple seasons efficiently.

        Args:
            seasons: List of season strings
            scrape_matches: Whether to scrape individual match reports
            scrape_player_stats: Whether to scrape season player stats

        Returns:
            Dict with all scraped data organized by season
        """
        results = {}

        for season in seasons:
            log.info(f"=== Scraping season {season} ===")
            results[season] = {"fixtures": None, "matches": [], "player_stats": {}}

            # Get fixtures first
            fixtures = self.scrape_season_fixtures(season)
            results[season]["fixtures"] = fixtures

            if scrape_matches and not fixtures.empty:
                # Scrape individual match reports
                match_urls = fixtures["match_url"].dropna().tolist()
                log.info(f"Scraping {len(match_urls)} match reports...")

                for i, url in enumerate(match_urls):
                    if not url:
                        continue
                    try:
                        match_data = self.scrape_match_report(url)
                        results[season]["matches"].append(match_data)

                        if (i + 1) % 10 == 0:
                            log.info(f"  Scraped {i + 1}/{len(match_urls)} matches")
                    except Exception as e:
                        log.error(f"Error scraping {url}: {e}")

            if scrape_player_stats:
                # Scrape season-level player stats
                for stat_type in ["standard", "shooting", "passing", "defense", "possession", "misc"]:
                    try:
                        stats_df = self.scrape_season_player_stats(season, stat_type)
                        results[season]["player_stats"][stat_type] = stats_df
                    except Exception as e:
                        log.error(f"Error scraping {stat_type} stats for {season}: {e}")

        return results

    def _get_soup(self, url: str) -> BeautifulSoup:
        """Get BeautifulSoup from URL using reused browser session.

        Includes 403/429 detection with exponential backoff (up to 3 retries).
        """
        @browser(
            headless=self.headless,
            block_images_and_css=False,  # Must load CSS/JS for dynamic stat tables
            wait_for_complete_page_load=True,  # Wait for full load including dynamic tables
            reuse_driver=True,  # KEY: Reuse browser session
            output=None,
            create_error_logs=False,
        )
        def fetch(driver: Driver, url: str):
            driver.google_get(url)
            # Wait for FBref content to load
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    driver.wait_for_element("body.fb", wait=15)
                    break
                except Exception:
                    if attempt < max_retries - 1:
                        log.warning(f"Retry {attempt + 1} for {url}")
                        driver.reload()
                        time.sleep(2)
                    else:
                        raise

            # Scroll page to trigger IntersectionObserver for lazy-loaded tables
            driver.run_js("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(4)
            driver.run_js("window.scrollTo(0, 0);")
            time.sleep(2)

            return driver.page_html

        max_attempts = 3
        for attempt in range(max_attempts):
            html = fetch(url)
            self._urls_scraped += 1

            # Detect 403 Forbidden or 429 Rate Limit responses
            is_blocked = (
                "403 Forbidden" in html
                or "Access Denied" in html
                or "Rate limit" in html
                or "429" in html
                or (len(html) < 5000 and "body.fb" not in html)
            )
            if is_blocked and attempt < max_attempts - 1:
                backoff = DELAY_SECONDS * (2 ** (attempt + 1))  # 16s, 32s
                log.warning(
                    "FBref blocked/rate-limited (attempt %d/%d) for %s — "
                    "backing off %ds",
                    attempt + 1, max_attempts, url, backoff,
                )
                time.sleep(backoff)
                continue
            elif is_blocked:
                log.error(
                    "FBref blocked after %d attempts for %s — skipping",
                    max_attempts, url,
                )

            break

        # Rate limiting
        time.sleep(DELAY_SECONDS)

        return BeautifulSoup(html, "html.parser")

    def _get_cell_text(self, row, data_stat: str) -> str:
        """Extract text from a table cell by data-stat attribute."""
        cell = row.find(["td", "th"], {"data-stat": data_stat})
        if cell:
            # Check for link text first
            link = cell.find("a")
            if link:
                return link.text.strip()
            return cell.text.strip()
        return ""

    def _table_to_df(self, table) -> pd.DataFrame:
        """Convert HTML table to DataFrame."""
        # Get headers
        headers = []
        thead = table.find("thead")
        if thead:
            header_row = thead.find_all("tr")[-1]  # Last header row
            for th in header_row.find_all(["th", "td"]):
                headers.append(th.get("data-stat", th.text.strip()))

        # Get rows
        rows = []
        tbody = table.find("tbody")
        if tbody:
            for tr in tbody.find_all("tr"):
                if "thead" in tr.get("class", []):
                    continue
                row = []
                for cell in tr.find_all(["td", "th"]):
                    link = cell.find("a")
                    text = link.text.strip() if link else cell.text.strip()
                    row.append(text)
                if row:
                    rows.append(row)

        if not rows:
            return pd.DataFrame()

        # Create DataFrame
        if len(headers) == len(rows[0]):
            return pd.DataFrame(rows, columns=headers)
        return pd.DataFrame(rows)

    def _parse_player_stats_tables(self, soup: BeautifulSoup) -> list[dict]:
        """Parse all player stats tables from a match report."""
        players = []

        # Find all player stats tables (home and away)
        stat_tables = soup.find_all("table", {"id": lambda x: x and "stats_" in str(x)})

        for table in stat_tables:
            table_id = table.get("id", "")

            # Determine team from table ID
            is_home = "_home" in table_id or table_id.endswith("_a") or "_a_" in table_id

            tbody = table.find("tbody")
            if not tbody:
                continue

            for row in tbody.find_all("tr"):
                if "thead" in row.get("class", []):
                    continue

                player_cell = row.find("th", {"data-stat": "player"})
                if not player_cell:
                    continue

                player_name = player_cell.text.strip()
                if not player_name or player_name == "":
                    continue

                player_data = {"player": player_name, "table_id": table_id}

                for cell in row.find_all("td"):
                    stat = cell.get("data-stat", "")
                    if stat:
                        player_data[stat] = cell.text.strip()

                players.append(player_data)

        return players

    def _parse_shots_table(self, soup: BeautifulSoup) -> list[dict]:
        """Parse shots table from match report."""
        shots = []

        shots_table = soup.find("table", {"id": "shots_all"})
        if not shots_table:
            return shots

        tbody = shots_table.find("tbody")
        if not tbody:
            return shots

        for row in tbody.find_all("tr"):
            if "thead" in row.get("class", []):
                continue

            shot = {}
            for cell in row.find_all(["td", "th"]):
                stat = cell.get("data-stat", "")
                if stat:
                    link = cell.find("a")
                    shot[stat] = link.text.strip() if link else cell.text.strip()

            if shot:
                shots.append(shot)

        return shots


def scrape_all_seasons(
    seasons: list[str],
    output_dir: Path | None = None,
    league: str = "seriea",
    headless: bool = True,
) -> None:
    """Scrape all data for multiple seasons and save to files.

    Args:
        seasons: List of seasons to scrape
        output_dir: Directory to save scraped data (default: DATA_DIR)
        league: League key (e.g., "seriea", "epl")
        headless: Run browser in headless mode
    """
    if output_dir is None:
        output_dir = DATA_DIR

    scraper = FBrefFastScraper(headless=headless, league=league)

    for season in seasons:
        log.info(f"\n{'='*60}")
        log.info(f"SCRAPING SEASON: {season}")
        log.info(f"{'='*60}\n")

        season_dir = output_dir / "scraped" / season.replace("-", "_")
        season_dir.mkdir(parents=True, exist_ok=True)

        # Scrape fixtures
        fixtures = scraper.scrape_season_fixtures(season)
        if not fixtures.empty:
            fixtures.to_parquet(season_dir / "fixtures.parquet", index=False)
            log.info(f"Saved {len(fixtures)} fixtures")

        # Scrape player stats for the season
        for stat_type in ["standard", "shooting", "passing", "defense", "possession", "misc"]:
            try:
                stats_df = scraper.scrape_season_player_stats(season, stat_type)
                if not stats_df.empty:
                    stats_df.to_parquet(season_dir / f"player_{stat_type}.parquet", index=False)
                    log.info(f"Saved {len(stats_df)} rows of {stat_type} stats")
            except Exception as e:
                log.error(f"Error scraping {stat_type}: {e}")

        # Scrape individual match reports
        if not fixtures.empty:
            matches_dir = season_dir / "matches"
            matches_dir.mkdir(exist_ok=True)

            match_urls = fixtures["match_url"].dropna().tolist()
            log.info(f"Scraping {len(match_urls)} match reports...")

            all_player_stats = []
            all_shots = []

            for i, url in enumerate(match_urls):
                if not url:
                    continue

                # Check if already scraped
                match_id = url.split("/")[-2] if "/" in url else f"match_{i}"
                match_file = matches_dir / f"{match_id}.json"

                if match_file.exists():
                    log.info(f"  Skipping {match_id} (already scraped)")
                    # Load existing data
                    with open(match_file) as f:
                        match_data = json.load(f)
                else:
                    try:
                        match_data = scraper.scrape_match_report(url)
                        match_data["match_id"] = match_id

                        # Save individual match
                        with open(match_file, "w") as f:
                            json.dump(match_data, f, indent=2)

                        if (i + 1) % 10 == 0:
                            log.info(f"  Scraped {i + 1}/{len(match_urls)} matches")
                    except Exception as e:
                        log.error(f"Error scraping {url}: {e}")
                        continue

                # Collect player stats and shots
                for ps in match_data.get("player_stats", []):
                    ps["match_id"] = match_id
                    ps["season"] = season
                    all_player_stats.append(ps)

                for shot in match_data.get("shots", []):
                    shot["match_id"] = match_id
                    shot["season"] = season
                    all_shots.append(shot)

            # Save aggregated data
            if all_player_stats:
                pd.DataFrame(all_player_stats).to_parquet(
                    season_dir / "match_player_stats.parquet", index=False
                )
                log.info(f"Saved {len(all_player_stats)} player stat rows")

            if all_shots:
                pd.DataFrame(all_shots).to_parquet(
                    season_dir / "match_shots.parquet", index=False
                )
                log.info(f"Saved {len(all_shots)} shots")

        log.info(f"\nCompleted {season}")


if __name__ == "__main__":
    import argparse as _ap
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = _ap.ArgumentParser(description="Fast FBref scraper with browser reuse")
    parser.add_argument(
        "--league",
        choices=list(LEAGUE_CONFIG.keys()),
        default="seriea",
        help="League to scrape (default: seriea)",
    )
    parser.add_argument(
        "seasons",
        nargs="*",
        default=[
            "2017-2018", "2018-2019", "2019-2020", "2020-2021",
            "2021-2022", "2022-2023", "2023-2024",
        ],
        help="Seasons to scrape",
    )
    args = parser.parse_args()

    scrape_all_seasons(args.seasons, league=args.league, headless=True)
