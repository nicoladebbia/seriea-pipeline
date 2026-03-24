"""Understat xG data scraper.

Understat provides free xG data for Serie A from 2014/15 onwards.
The data is embedded in inline JavaScript (JSON.parse), so a headless
browser is required to reliably extract it (Cloudflare blocks plain requests).

This scraper:
1. Loads the league page in headless Chrome via Selenium
2. Extracts team and match data with xG values from JS variables
3. Falls back to regex extraction from page source if JS extraction fails
4. Respects rate limits (3 seconds between requests)
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

UNDERSTAT_BASE = "https://understat.com"
RATE_LIMIT_SECONDS = 3  # Be nice to the server

# Try to import Selenium for headless browser support
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.support.ui import WebDriverWait
    HAS_SELENIUM = True
except ImportError:
    HAS_SELENIUM = False
    log.warning("Selenium not installed — Understat scraper will fall back to requests (may fail)")

# Season mapping: our format -> Understat format
def _build_understat_season_map() -> dict[str, str]:
    """Auto-generate season map: '2014-2015' -> '2014', etc."""
    m = {}
    for start_year in range(2014, 2040):
        key = f"{start_year}-{start_year + 1}"
        m[key] = str(start_year)
    return m

SEASON_MAP = _build_understat_season_map()


class UnderstatScraper:
    """Scraper for Understat xG data using headless Chrome (Selenium)."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })
        self._last_request = 0
        self._driver = None

    def _rate_limit(self):
        """Ensure we don't exceed rate limits."""
        elapsed = time.time() - self._last_request
        if elapsed < RATE_LIMIT_SECONDS:
            time.sleep(RATE_LIMIT_SECONDS - elapsed)
        self._last_request = time.time()

    def _get_driver(self):
        """Lazily create a headless Chrome driver."""
        if self._driver is None and HAS_SELENIUM:
            opts = ChromeOptions()
            opts.add_argument("--headless")
            opts.add_argument("--no-sandbox")
            opts.add_argument("--disable-dev-shm-usage")
            opts.add_argument("--disable-gpu")
            opts.add_argument(
                "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
            )
            self._driver = webdriver.Chrome(options=opts)
            self._driver.set_page_load_timeout(45)
        return self._driver

    def close(self):
        """Close the browser driver."""
        if self._driver:
            try:
                self._driver.quit()
            except Exception as e:
                log.debug(f"Failed to quit browser driver cleanly: {e}")
            self._driver = None

    def _get_page_browser(self, url: str) -> str:
        """Fetch page using headless Chrome — executes JS, bypasses Cloudflare."""
        driver = self._get_driver()
        if not driver:
            raise RuntimeError("Selenium not available")
        self._rate_limit()
        driver.get(url)
        # Wait for JS to finish rendering (Understat loads data via inline scripts)
        time.sleep(2)
        return driver.page_source

    def _get_page_requests(self, url: str) -> str:
        """Fetch page with plain requests (fast but may fail on Cloudflare)."""
        self._rate_limit()
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        return resp.text

    def _get_page(self, url: str) -> str:
        """Fetch page: try headless Chrome first, fall back to requests."""
        if HAS_SELENIUM:
            try:
                return self._get_page_browser(url)
            except Exception as e:
                log.warning(f"Selenium fetch failed for {url}: {e}, trying requests")
        return self._get_page_requests(url)

    def _extract_json_var(self, html: str, var_name: str) -> dict | list | None:
        """Extract JSON data from JavaScript variable in page source."""
        # Pattern: var varName = JSON.parse('...');
        pattern = rf"var {var_name}\s*=\s*JSON\.parse\('([^']+)'\)"
        match = re.search(pattern, html)
        if match:
            try:
                raw = match.group(1)
                decoded = raw.encode("utf-8").decode("unicode_escape")
                return json.loads(decoded)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                log.warning(f"Failed to parse {var_name} via regex: {e}")

        # Fallback: try extracting via Selenium JS execution
        if HAS_SELENIUM and self._driver:
            try:
                data = self._driver.execute_script(f"return typeof {var_name} !== 'undefined' ? {var_name} : null")
                if data:
                    log.info(f"Extracted {var_name} via JS execution")
                    return data
            except Exception as e:
                log.debug(f"JS extraction of {var_name} failed: {e}")

        return None

    def get_season_data(self, season: str) -> dict:
        """Get all xG data for a Serie A season.

        Args:
            season: Season in our format (e.g., "2023-2024")

        Returns:
            Dict with 'teams', 'players', and 'matches' data
        """
        understat_season = SEASON_MAP.get(season)
        if not understat_season:
            log.warning(f"Season {season} not available on Understat")
            return {}

        url = f"{UNDERSTAT_BASE}/league/Serie_A/{understat_season}"
        log.info(f"Fetching Understat data for {season} from {url}")

        try:
            html = self._get_page(url)
        except Exception as e:
            log.error(f"Failed to fetch {url}: {e}")
            return {}

        # Extract embedded JSON data
        result = {
            "season": season,
            "teams": self._extract_json_var(html, "teamsData"),
            "players": self._extract_json_var(html, "playersData"),
            "dates": self._extract_json_var(html, "datesData"),
        }

        # Log what we found
        found_any = False
        for key, data in result.items():
            if data and key != "season":
                count = len(data) if isinstance(data, (list, dict)) else "N/A"
                log.info(f"  {key}: {count} items")
                found_any = True

        if not found_any:
            log.warning(f"No data extracted from {url} — page may have changed structure")

        return result

    def get_match_details(self, match_id: int | str) -> dict:
        """Get detailed xG data for a specific match."""
        url = f"{UNDERSTAT_BASE}/match/{match_id}"
        log.debug(f"Fetching match {match_id}")

        try:
            html = self._get_page(url)
        except Exception as e:
            log.error(f"Failed to fetch match {match_id}: {e}")
            return {}

        # Extract match data via regex
        json_matches = re.findall(r"JSON\.parse\('([^']+)'\)", html)
        if json_matches:
            try:
                decoded = json_matches[0].encode("utf-8").decode("unicode_escape")
                return json.loads(decoded)
            except Exception as e:
                log.error(f"Failed to parse match {match_id}: {e}")

        # Try JS extraction
        if HAS_SELENIUM and self._driver:
            try:
                data = self._driver.execute_script("return typeof shotsData !== 'undefined' ? shotsData : null")
                if data:
                    return data
            except Exception as e:
                log.debug(f"Failed to extract shotsData via JS: {e}")

        return {}

    def scrape_all_seasons(
        self,
        seasons: list[str] | None = None,
        output_dir: Path | None = None,
    ) -> pd.DataFrame:
        """Scrape xG data for multiple seasons.

        Args:
            seasons: List of seasons to scrape (default: all available)
            output_dir: Directory to save data (optional)

        Returns:
            DataFrame with all match xG data
        """
        if seasons is None:
            seasons = list(SEASON_MAP.keys())

        all_matches = []

        try:
            for season in seasons:
                data = self.get_season_data(season)

                # Process dates/matches data
                dates = data.get("dates")
                if dates:
                    if isinstance(dates, dict):
                        # dates is {date: [matches]}
                        for date_str, matches in dates.items():
                            for match in matches:
                                match["date"] = date_str
                                match["season"] = season
                                all_matches.append(match)
                    elif isinstance(dates, list):
                        for match in dates:
                            match["season"] = season
                            all_matches.append(match)

                # Save intermediate results
                if output_dir and data:
                    season_file = output_dir / f"understat_{season.replace('-', '_')}.json"
                    season_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(season_file, "w") as f:
                        json.dump(data, f, indent=2)
                    log.info(f"Saved {season} data to {season_file}")
        finally:
            # Always close the browser when done
            self.close()

        if not all_matches:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_matches)
        log.info(f"Total matches collected: {len(df)}")

        return df


def scrape_understat_xg(
    seasons: list[str] | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Main function to scrape Understat xG data.

    Args:
        seasons: Seasons to scrape (default: 2017-2024)
        output_path: Path to save parquet file

    Returns:
        DataFrame with match xG data
    """
    if seasons is None:
        seasons = [
            "2017-2018",
            "2018-2019",
            "2019-2020",
            "2020-2021",
            "2021-2022",
            "2022-2023",
            "2023-2024",
            "2024-2025",
            "2025-2026",
        ]

    scraper = UnderstatScraper()
    df = scraper.scrape_all_seasons(seasons)

    if output_path and not df.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Flatten nested dicts — Parquet can't handle struct columns with
        # empty child fields (e.g. 'numbers.format' is an empty dict in some rows)
        save_df = df.copy()
        for col in save_df.columns:
            if save_df[col].apply(type).eq(dict).any():
                save_df[col] = save_df[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, dict) else x
                )
        save_df.to_parquet(output_path, index=False)
        log.info(f"Saved {len(df)} matches to {output_path}")

        # Also save raw JSON for the pressing feature module
        json_dir = output_path.parent
        for season in df["season"].unique() if "season" in df.columns else []:
            season_file = json_dir / f"understat_{season.replace('-', '_')}.json"
            season_data = df[df["season"] == season].to_dict(orient="records")
            with open(season_file, "w") as f:
                json.dump(season_data, f, indent=2, default=str)
            log.info(f"Saved {len(season_data)} matches to {season_file}")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from config.settings import DATA_DIR

    output_path = DATA_DIR / "external" / "understat" / "matches_xg.parquet"
    df = scrape_understat_xg(output_path=output_path)

    if not df.empty:
        print(f"\nScraped {len(df)} matches")
        print(df.head())
