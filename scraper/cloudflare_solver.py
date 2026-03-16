"""Solve Cloudflare challenges and extract cookies for reuse with requests.

Uses a visible Chrome browser to solve ONE Cloudflare interactive challenge,
then extracts the cf_clearance cookie. This cookie can be reused in plain
requests.Session() for fast bulk downloads.

Usage:
    cookies, user_agent = solve_cloudflare("https://fbref.com/en/matches/...")
    client = FBrefClient(cookies=cookies, user_agent=user_agent)
    resp = client.get(url)  # uses cookies, no browser needed
"""

from __future__ import annotations

import logging
import time
from typing import Optional

log = logging.getLogger(__name__)

# Browser User-Agent — must match between Selenium and requests
CHROME_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


def solve_cloudflare(
    url: str,
    use_undetected: bool = True,
    headless: bool = False,
    timeout: int = 90,
) -> tuple[dict, str]:
    """Solve Cloudflare challenge and return (cookies_dict, user_agent).

    Opens a visible Chrome browser, navigates to the URL, waits for the
    Cloudflare challenge to auto-resolve, then extracts cookies.

    Args:
        url: Any FBref URL to solve the challenge for.
        use_undetected: Use undetected-chromedriver (recommended).
        headless: Run headless (not recommended — Cloudflare detects it).
        timeout: Max seconds to wait for challenge resolution.

    Returns:
        Tuple of (cookies_dict, user_agent_string)
    """
    driver = _create_driver(use_undetected, headless)

    try:
        log.info("Navigating to %s to solve Cloudflare challenge...", url)
        driver.get(url)

        # Wait for Cloudflare challenge to resolve
        resolved = _wait_for_challenge(driver, timeout)
        if not resolved:
            raise RuntimeError(
                "Cloudflare challenge did not resolve within %ds. "
                "Try with --no-headless or check your network." % timeout
            )

        # Extract cookies
        cookies = {}
        for cookie in driver.get_cookies():
            cookies[cookie["name"]] = cookie["value"]

        # Get the actual User-Agent the browser reported
        user_agent = driver.execute_script("return navigator.userAgent;")

        cf_clearance = cookies.get("cf_clearance", "")
        log.info(
            "Cloudflare solved! Got %d cookies (cf_clearance: %s...)",
            len(cookies),
            cf_clearance[:20] if cf_clearance else "MISSING",
        )

        if not cf_clearance:
            log.warning(
                "No cf_clearance cookie found. Challenge may not have fully resolved. "
                "Cookies: %s", list(cookies.keys()),
            )

        return cookies, user_agent

    finally:
        driver.quit()
        log.info("Browser closed after Cloudflare solve.")


def _create_driver(use_undetected: bool, headless: bool):
    """Create Chrome driver (undetected or standard Selenium)."""
    if use_undetected:
        try:
            import undetected_chromedriver as uc
        except ImportError:
            log.warning(
                "undetected-chromedriver not installed. Falling back to selenium. "
                "Install with: pip install undetected-chromedriver"
            )
            return _create_selenium_driver(headless)

        options = uc.ChromeOptions()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-extensions")
        # Clean profile — no ad blockers or other extensions that block page content
        options.add_argument("--disable-component-extensions-with-background-pages")
        options.add_argument("--disable-default-apps")
        driver = uc.Chrome(options=options, version_main=144)
        log.info("Created undetected-chromedriver")
        return driver
    else:
        return _create_selenium_driver(headless)


def _create_selenium_driver(headless: bool):
    """Create standard Selenium Chrome driver."""
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-extensions")
    # Clean profile — no ad blockers or other extensions that block page content
    options.add_argument("--disable-component-extensions-with-background-pages")
    options.add_argument("--disable-default-apps")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(options=options)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )
    log.info("Created Selenium Chrome driver")
    return driver


def _wait_for_challenge(driver, timeout: int) -> bool:
    """Wait for Cloudflare challenge to resolve.

    Polls page source for signs that the challenge is gone and real
    FBref content has loaded.
    """
    start = time.time()

    while time.time() - start < timeout:
        try:
            page_source = driver.page_source
        except Exception:
            time.sleep(2)
            continue

        # page_source can be None if browser is in a transitional state
        if page_source is None:
            time.sleep(2)
            continue

        # Check if Cloudflare challenge is still showing
        if "Just a moment" in page_source or "challenge-platform" in page_source:
            elapsed = time.time() - start
            log.info("Cloudflare challenge active (%.0fs elapsed)...", elapsed)
            time.sleep(3)
            continue

        # Check if we got actual FBref content
        if "fbref.com" in page_source and ("body.fb" in page_source or "stats_" in page_source or "sched_" in page_source):
            log.info("FBref content detected — challenge resolved!")
            # Give cookies a moment to settle
            time.sleep(2)
            return True

        # Page loaded but might not be FBref yet (redirect, etc.)
        time.sleep(2)

    return False
