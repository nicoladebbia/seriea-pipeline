"""Fallback FBref scraper using Selenium with visible Chrome.

Tier 2/3 fallback when botasaurus gets blocked. Uses the user's own Chrome
browser in visible mode, making it indistinguishable from manual browsing.

Tier 3 option: undetected-chromedriver patches ChromeDriver to evade
bot detection entirely.

Usage:
    scraper = FBrefSeleniumScraper(headless=False, delay=15)
    scraper.download_match_html(url, output_path)
    scraper.scrape_season(season, match_urls)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from config.settings import FBREF_BASE_URL, SERIE_A_COMP_ID
from storage.paths import html_match_path, html_season_dir

log = logging.getLogger(__name__)

# Try to import selenium
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    HAS_SELENIUM = True
except ImportError:
    HAS_SELENIUM = False

# Try to import undetected-chromedriver (Tier 3)
try:
    import undetected_chromedriver as uc
    HAS_UNDETECTED = True
except ImportError:
    HAS_UNDETECTED = False


class FBrefSeleniumScraper:
    """Fallback FBref scraper using Selenium with visible Chrome."""

    def __init__(
        self,
        headless: bool = False,
        delay: int = 15,
        use_undetected: bool = False,
    ):
        """Initialize the Selenium scraper.

        Args:
            headless: Run in headless mode (default False for human-like browsing)
            delay: Seconds to wait between requests
            use_undetected: Use undetected-chromedriver (Tier 3)
        """
        self.headless = headless
        self.delay = delay
        self.use_undetected = use_undetected
        self._driver: Optional[webdriver.Chrome] = None
        self._pages_fetched = 0

    def _init_driver(self):
        """Initialize the Chrome driver."""
        if self._driver is not None:
            return

        if self.use_undetected:
            if not HAS_UNDETECTED:
                raise ImportError(
                    "undetected-chromedriver required for Tier 3. "
                    "Install with: pip install undetected-chromedriver"
                )
            options = uc.ChromeOptions()
            if self.headless:
                options.add_argument("--headless=new")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--window-size=1920,1080")
            self._driver = uc.Chrome(options=options)
            log.info("Initialized undetected-chromedriver (Tier 3)")
        else:
            if not HAS_SELENIUM:
                raise ImportError(
                    "selenium required for Tier 2. "
                    "Install with: pip install selenium"
                )
            options = Options()
            if self.headless:
                options.add_argument("--headless=new")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--disable-blink-features=AutomationControlled")
            # Clean profile: no extensions (avoids ad blockers breaking Cloudflare)
            options.add_argument("--disable-extensions")
            # Remove automation indicators
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option("useAutomationExtension", False)
            self._driver = webdriver.Chrome(options=options)
            # Mask webdriver property
            self._driver.execute_cdp_cmd(
                "Page.addScriptToEvaluateOnNewDocument",
                {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
            )
            log.info("Initialized Selenium Chrome (Tier 2)")

    def get_page_html(self, url: str) -> str:
        """Fetch a page and return its HTML source.

        Handles Cloudflare "Just a moment..." challenge by waiting for it
        to auto-resolve before checking for FBref content.
        """
        self._init_driver()
        self._driver.get(url)

        # Phase 1: Wait for Cloudflare challenge to resolve (if present)
        # Cloudflare shows "Just a moment..." then auto-solves in ~5-10s
        for attempt in range(6):
            page_src = self._driver.page_source
            if "Just a moment" in page_src or "challenge-platform" in page_src:
                log.info("Cloudflare challenge detected, waiting... (attempt %d/6)", attempt + 1)
                time.sleep(5)
            else:
                break

        # Phase 2: Wait for actual FBref page to load
        try:
            WebDriverWait(self._driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "body.fb"))
            )
        except Exception:
            log.warning("Timeout waiting for body.fb on %s — using page as-is", url)

        # Phase 3: Wait for FBref to dynamically load extra stat tables
        # FBref injects passing/defense/possession/misc tables via JS after
        # initial render. We scroll + poll until they appear or timeout.
        max_poll = 12  # poll up to 12 times (~60s total)
        for attempt in range(max_poll):
            # Scroll to trigger lazy-load observers
            self._driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight * %f);" % (attempt / max_poll)
            )
            time.sleep(3)

            # Check if extra tables appeared
            passing_tables = self._driver.find_elements(By.CSS_SELECTOR, "table[id*='_passing']")
            if len(passing_tables) >= 1:
                log.info("Extra stat tables loaded (attempt %d/%d)", attempt + 1, max_poll)
                # Give remaining tables a moment to render
                time.sleep(3)
                break
        else:
            log.warning("Extra stat tables did not load for %s after %d polls", url, max_poll)

        # Scroll back to top
        self._driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(2)

        self._pages_fetched += 1
        html = self._driver.page_source

        # Rate limiting
        time.sleep(self.delay)

        return html

    def download_match_html(self, url: str, output_path: Path) -> bool:
        """Download a single match report HTML.

        Args:
            url: FBref match report URL
            output_path: Where to save the HTML file

        Returns:
            True if successful, False otherwise
        """
        try:
            html = self.get_page_html(url)

            # Basic validation — check for player stats tables
            if "stats_" not in html or len(html) < 10000:
                log.warning("Page may be incomplete or blocked: %s (%d bytes)", url, len(html))
                if "Rate limit" in html or "429" in html:
                    log.error("Rate limited! Increase delay.")
                    return False

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(html, encoding="utf-8")
            return True

        except Exception as e:
            log.error("Failed to download %s: %s", url, e)
            return False

    def scrape_fixtures_page(self, season: str) -> str:
        """Fetch the fixtures/results page for a season.

        Returns raw HTML — parsing is done by the caller.
        """
        url = (
            f"{FBREF_BASE_URL}/en/comps/{SERIE_A_COMP_ID}/{season}/"
            f"schedule/{season}-Serie-A-Scores-and-Fixtures"
        )
        log.info("Fetching fixtures page: %s", url)
        return self.get_page_html(url)

    def close(self):
        """Close the browser."""
        if self._driver:
            self._driver.quit()
            self._driver = None
            log.info("Browser closed (fetched %d pages)", self._pages_fetched)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
