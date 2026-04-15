"""
Module: scripts/backfill_match_reports.py
Purpose: Orchestrates FBref historical match report scraping with Cloudflare bypass and resumable progress tracking
Inputs:  FBref fixture schedules (scraped from season pages), Cloudflare cookies (manual browser solve), data/raw/failed_downloads.json
Outputs: HTML match reports saved to data/raw/html/{season}/{match_id}.html, data/registry.json (download tracking)
Called by: Manual CLI invocation for backfilling historical data
Depends on: scraper.client (FBrefClient), scraper.registry (Registry), storage.raw (save_html), config.settings (DATA_DIR, FBREF_BASE_URL)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from bs4 import BeautifulSoup

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import DATA_DIR, FBREF_BASE_URL, SERIE_A_COMP_ID
from scripts.utils.scraper_state import load_failed, save_failed
from scraper.client import FBrefClient
from scraper.registry import Registry
from storage.paths import html_match_path, html_season_dir
from storage.raw import save_html

log = logging.getLogger(__name__)

# Seasons with FBref match-level player stats (xG available from 2017-2018)
BACKFILL_SEASONS = [
    "2017-2018",
    "2018-2019",
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2025-2026",
]

# Track failed downloads so we can retry them
FAILED_LOG = DATA_DIR / "raw" / "failed_downloads.json"



def _mark_failed(match_id: str, season: str, url: str, error: str) -> None:
    """Record a failed download for later retry."""
    failed = load_failed(FAILED_LOG)
    failed[match_id] = {
        "season": season,
        "match_url": url,
        "error": error,
        "failed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    save_failed(FAILED_LOG, failed)


def _remove_failed(match_id: str) -> None:
    """Remove a match from the failed log after successful download."""
    failed = load_failed(FAILED_LOG)
    if match_id in failed:
        del failed[match_id]
        save_failed(FAILED_LOG, failed)


def parse_fixtures_from_html(html: str, season: str) -> list[dict]:
    """Extract match URLs from a fixtures page HTML.

    Returns list of dicts with keys: match_id, match_url, home_team, away_team, date
    """
    soup = BeautifulSoup(html, "lxml")

    # Find fixtures table — try multiple ID patterns
    table = None
    for pattern in [
        f"sched_{season}_{SERIE_A_COMP_ID}_1",
        f"sched_{season}_{SERIE_A_COMP_ID}",
    ]:
        table = soup.find("table", id=pattern)
        if table:
            break

    if not table:
        # Fallback: find any schedule table
        table = soup.find("table", id=lambda x: x and "sched_" in str(x))

    if not table:
        log.warning("No fixtures table found for %s", season)
        return []

    tbody = table.find("tbody")
    if not tbody:
        return []

    matches = []
    for row in tbody.find_all("tr"):
        if "thead" in row.get("class", []):
            continue

        score_cell = row.find("td", {"data-stat": "score"})
        if not score_cell:
            continue

        link = score_cell.find("a")
        if not link or not link.get("href"):
            continue

        match_url = FBREF_BASE_URL + link["href"]

        # Extract match_id from URL (e.g. /matches/abc123/...)
        parts = link["href"].split("/")
        match_id = None
        for i, part in enumerate(parts):
            if part == "matches" and i + 1 < len(parts):
                match_id = parts[i + 1]
                break

        if not match_id:
            continue

        home_cell = row.find("td", {"data-stat": "home_team"})
        away_cell = row.find("td", {"data-stat": "away_team"})
        date_cell = row.find("th", {"data-stat": "date"}) or row.find("td", {"data-stat": "date"})

        home = home_cell.get_text(strip=True) if home_cell else ""
        away = away_cell.get_text(strip=True) if away_cell else ""
        date = date_cell.get_text(strip=True) if date_cell else ""

        matches.append({
            "match_id": match_id,
            "match_url": match_url,
            "home_team": home,
            "away_team": away,
            "date": date,
        })

    return matches


def _solve_cloudflare_for_cookies(
    sample_url: str,
    use_undetected: bool = True,
    headless: bool = False,
) -> tuple[dict, str]:
    """Solve Cloudflare challenge and get cookies for requests."""
    from scraper.cloudflare_solver import solve_cloudflare

    log.info("Solving Cloudflare challenge (browser will open briefly)...")
    cookies, user_agent = solve_cloudflare(
        url=sample_url,
        use_undetected=use_undetected,
        headless=headless,
    )
    return cookies, user_agent


def _scrape_with_requests(
    matches: list[dict],
    season: str,
    registry: Registry,
    delay: int,
    cookies: dict,
    user_agent: str,
    limit: int | None = None,
    use_undetected: bool = True,
    headless: bool = False,
) -> tuple[int, dict, str]:
    """Scrape using plain HTTP requests with Cloudflare cookies.

    Returns (scraped_count, current_cookies, current_user_agent).
    Cookies may be refreshed mid-run if they expire.
    """
    client = FBrefClient(delay=delay, cookies=cookies, user_agent=user_agent)

    total = len(matches)
    scraped = 0
    skipped = 0
    failed = 0
    consecutive_403 = 0

    for i, match in enumerate(matches):
        mid = match["match_id"]
        url = match["match_url"]

        if registry.is_downloaded(mid):
            skipped += 1
            continue

        if limit and scraped >= limit:
            log.info("Reached download limit (%d)", limit)
            break

        try:
            resp = client.get(url)
            html = resp.text

            # Check if we got a Cloudflare challenge page instead of content
            if "Just a moment" in html or len(html) < 10000:
                log.warning("Got Cloudflare challenge page for %s — cookies may have expired", mid)
                consecutive_403 += 1
                failed += 1
                _mark_failed(mid, season, url, "cloudflare_challenge_page")

                if consecutive_403 >= 2:
                    # Re-solve Cloudflare
                    log.info("Re-solving Cloudflare challenge (cookies expired)...")
                    cookies, user_agent = _solve_cloudflare_for_cookies(
                        url, use_undetected=use_undetected, headless=headless,
                    )
                    client = FBrefClient(delay=delay, cookies=cookies, user_agent=user_agent)
                    consecutive_403 = 0
                    # Retry this match
                    resp = client.get(url)
                    html = resp.text
                    if "Just a moment" in html or len(html) < 10000:
                        log.error("Still getting challenged after re-solve. Stopping.")
                        break
                else:
                    continue

            consecutive_403 = 0

            # Validate — check for player stats tables
            stats_count = html.count("stats_")
            if stats_count < 4:
                log.warning(
                    "Low table count for %s: %d 'stats_' references (expected ~24). "
                    "May be incomplete.",
                    mid, stats_count,
                )

            # Save via existing storage helper
            html_path = save_html(season, mid, html)

            registry.mark_downloaded(
                match_id=mid,
                season=season,
                match_url=url,
                html_path=str(html_path),
            )
            scraped += 1
            _remove_failed(mid)  # Clear from failed log if previously failed

            if scraped % 10 == 0 or scraped <= 3:
                log.info(
                    "  [%s] Progress: %d scraped, %d skipped, %d failed | "
                    "match %d/%d | %s vs %s",
                    season, scraped, skipped, failed,
                    i + 1, total,
                    match.get("home_team", "?"), match.get("away_team", "?"),
                )

        except Exception as e:
            failed += 1
            error_str = str(e)
            log.error("Failed to download %s (%s): %s", mid, url, e)
            _mark_failed(mid, season, url, error_str)

            if "403" in error_str:
                consecutive_403 += 1
                if consecutive_403 >= 3:
                    # Re-solve Cloudflare — try up to 2 times
                    solved = False
                    for solve_attempt in range(1, 3):
                        log.info("Multiple 403s — re-solving Cloudflare (attempt %d/2)...", solve_attempt)
                        try:
                            cookies, user_agent = _solve_cloudflare_for_cookies(
                                url, use_undetected=use_undetected, headless=headless,
                            )
                            client = FBrefClient(delay=delay, cookies=cookies, user_agent=user_agent)
                            consecutive_403 = 0
                            solved = True
                            break
                        except Exception as solve_err:
                            log.warning("Re-solve attempt %d failed: %s", solve_attempt, solve_err)
                            time.sleep(15)  # Wait before retry
                    if not solved:
                        log.error("Failed to re-solve Cloudflare after 2 attempts. Stopping season.")
                        break
            elif "429" in error_str:
                log.warning("Rate limited (429). Waiting 60s before continuing...")
                time.sleep(60)

    log.info(
        "  [%s] Requests method: %d scraped, %d skipped, %d failed",
        season, scraped, skipped, failed,
    )
    return scraped, cookies, user_agent


def _scrape_with_browser(
    matches: list[dict],
    season: str,
    registry: Registry,
    headless: bool,
    delay: int,
    use_undetected: bool = False,
    limit: int | None = None,
) -> int:
    """Scrape using browser for every page (Selenium/undetected-chromedriver)."""
    from scraper.fbref_selenium import FBrefSeleniumScraper

    scraped = 0
    skipped = 0
    total = len(matches)

    with FBrefSeleniumScraper(
        headless=headless, delay=delay, use_undetected=use_undetected
    ) as scraper:
        for i, match in enumerate(matches):
            mid = match["match_id"]
            url = match["match_url"]

            if registry.is_downloaded(mid):
                skipped += 1
                continue

            if limit and scraped >= limit:
                log.info("Reached download limit (%d)", limit)
                break

            out_path = html_match_path(season, mid)
            success = scraper.download_match_html(url, out_path)

            if success:
                registry.mark_downloaded(
                    match_id=mid,
                    season=season,
                    match_url=url,
                    html_path=str(out_path.relative_to(DATA_DIR)),
                )
                scraped += 1
                _remove_failed(mid)

                if scraped % 10 == 0:
                    log.info(
                        "  [%s] Progress: %d/%d scraped, %d skipped",
                        season, scraped, total - skipped, skipped,
                    )
            else:
                log.warning("Skipping %s after download failure", mid)
                _mark_failed(mid, season, url, "browser_download_failed")

    return scraped


def _get_fixtures(
    season: str,
    delay: int,
    cookies: dict | None = None,
    user_agent: str | None = None,
) -> list[dict]:
    """Get fixtures for a season (from cache or FBref)."""
    fixtures_cache = DATA_DIR / "raw" / "fixtures" / f"{season}_fixtures.json"

    if fixtures_cache.exists():
        with open(fixtures_cache) as f:
            matches = json.load(f)
        log.info("Loaded %d cached fixtures for %s", len(matches), season)
        return matches

    # Check for existing fixtures.html in HTML dir (both naming conventions)
    html = None
    for variant in [season, season.replace("-", "_")]:
        fixtures_html = DATA_DIR / "raw" / "html" / variant / "fixtures.html"
        if fixtures_html.exists():
            log.info("Found cached fixtures.html at %s", fixtures_html)
            html = fixtures_html.read_text(encoding="utf-8", errors="replace")
            break

    if html is None:
        log.info("Fetching fixtures page for %s...", season)
        client = FBrefClient(delay=delay, cookies=cookies, user_agent=user_agent)
        url = (
            f"{FBREF_BASE_URL}/en/comps/{SERIE_A_COMP_ID}/{season}/"
            f"schedule/{season}-Serie-A-Scores-and-Fixtures"
        )
        resp = client.get(url)
        html = resp.text

        # Cache the fixtures HTML
        fixtures_html_dir = html_season_dir(season)
        fixtures_html_path = fixtures_html_dir / "fixtures.html"
        fixtures_html_path.write_text(html, encoding="utf-8")
        log.info("Saved fixtures HTML to %s", fixtures_html_path)

    matches = parse_fixtures_from_html(html, season)
    log.info("Parsed %d match URLs for %s", len(matches), season)

    # Cache fixtures as JSON for fast reload
    fixtures_cache.parent.mkdir(parents=True, exist_ok=True)
    with open(fixtures_cache, "w") as f:
        json.dump(matches, f, indent=2)

    return matches


def scrape_season(
    season: str,
    method: str,
    headless: bool,
    delay: int,
    registry: Registry,
    limit: int | None = None,
    cookies: dict | None = None,
    user_agent: str | None = None,
    use_undetected: bool = True,
) -> tuple[int, dict | None, str | None]:
    """Scrape all match reports for a single season.

    Returns (scraped_count, current_cookies, current_user_agent).
    """
    log.info("=" * 60)
    log.info("SEASON: %s  |  Method: %s  |  Delay: %ds", season, method, delay)
    log.info("=" * 60)

    # Step 1: Get fixtures (match URLs)
    matches = _get_fixtures(season, delay, cookies, user_agent)

    if not matches:
        log.warning("No matches found for %s — skipping", season)
        return 0, cookies, user_agent

    # Check how many already downloaded
    already = sum(1 for m in matches if registry.is_downloaded(m["match_id"]))
    remaining = len(matches) - already
    log.info("Already downloaded: %d / %d  |  Remaining: %d", already, len(matches), remaining)

    if remaining == 0:
        log.info("Season %s fully downloaded — nothing to do", season)
        return 0, cookies, user_agent

    est_time = remaining * delay
    log.info("Estimated time: %d min (%d pages x %ds)", est_time // 60, remaining, delay)

    # Step 2: Scrape match reports
    if method == "requests":
        scraped, cookies, user_agent = _scrape_with_requests(
            matches, season, registry, delay,
            cookies=cookies or {},
            user_agent=user_agent or "",
            limit=limit,
            use_undetected=use_undetected,
            headless=headless,
        )
        return scraped, cookies, user_agent
    elif method == "browser":
        scraped = _scrape_with_browser(
            matches, season, registry, headless, delay,
            use_undetected=use_undetected,
            limit=limit,
        )
        return scraped, cookies, user_agent
    else:
        raise ValueError(f"Unknown method: {method}")


def _get_retry_matches(registry: Registry) -> list[dict]:
    """Build match list from the failed downloads log."""
    failed = load_failed(FAILED_LOG)
    if not failed:
        log.info("No failed downloads to retry.")
        return []

    # Filter out ones already in registry (successfully downloaded since)
    retry = []
    for mid, info in failed.items():
        if not registry.is_downloaded(mid):
            retry.append({
                "match_id": mid,
                "match_url": info["match_url"],
                "home_team": "",
                "away_team": "",
                "date": "",
                "season": info["season"],
            })

    log.info("Found %d failed downloads to retry (out of %d in log)", len(retry), len(failed))
    return retry


def main():
    parser = argparse.ArgumentParser(
        description="Backfill FBref match report HTML for historical seasons."
    )
    parser.add_argument(
        "--season",
        help="Single season to scrape (e.g. 2023-2024). Default: all backfill seasons.",
    )
    parser.add_argument(
        "--method",
        choices=["requests", "browser"],
        default="requests",
        help=(
            "Scraping method. 'requests' (default): solve Cloudflare once with browser, "
            "then use cookies with plain HTTP. 'browser': use browser for every page."
        ),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode (not recommended — Cloudflare detects it)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=12,
        help="Delay between requests in seconds (default: 12)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max matches to download per season (for testing)",
    )
    parser.add_argument(
        "--undetected",
        action="store_true",
        default=True,
        help="Use undetected-chromedriver (default: True)",
    )
    parser.add_argument(
        "--no-undetected",
        action="store_true",
        help="Use standard Selenium instead of undetected-chromedriver",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry previously failed downloads instead of full season scan",
    )
    args = parser.parse_args()

    use_undetected = not args.no_undetected

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    registry = Registry()

    # For requests method: solve Cloudflare once upfront
    cookies = None
    user_agent = None
    if args.method == "requests":
        sample_url = f"{FBREF_BASE_URL}/en/comps/{SERIE_A_COMP_ID}/"
        cookies, user_agent = _solve_cloudflare_for_cookies(
            sample_url,
            use_undetected=use_undetected,
            headless=args.headless,
        )

    total_scraped = 0
    start = time.time()

    if args.retry_failed:
        # Retry mode: re-download previously failed matches
        retry_matches = _get_retry_matches(registry)
        if not retry_matches:
            return

        # Group by season
        by_season: dict[str, list[dict]] = {}
        for m in retry_matches:
            s = m.pop("season")
            by_season.setdefault(s, []).append(m)

        for season, matches in sorted(by_season.items()):
            log.info("=" * 60)
            log.info("RETRY FAILED: %s  |  %d matches", season, len(matches))
            log.info("=" * 60)

            if args.method == "requests":
                scraped, cookies, user_agent = _scrape_with_requests(
                    matches, season, registry, args.delay,
                    cookies=cookies or {},
                    user_agent=user_agent or "",
                    use_undetected=use_undetected,
                    headless=args.headless,
                )
            else:
                scraped = _scrape_with_browser(
                    matches, season, registry, args.headless, args.delay,
                    use_undetected=use_undetected,
                )
            total_scraped += scraped
            log.info("Retry %s done: %d recovered", season, scraped)
    else:
        # Normal mode: scrape full seasons
        seasons = [args.season] if args.season else BACKFILL_SEASONS

        for season in seasons:
            scraped, cookies, user_agent = scrape_season(
                season=season,
                method=args.method,
                headless=args.headless,
                delay=args.delay,
                registry=registry,
                limit=args.limit,
                cookies=cookies,
                user_agent=user_agent,
                use_undetected=use_undetected,
            )
            total_scraped += scraped
            log.info("Season %s done: %d new pages", season, scraped)

    elapsed = time.time() - start
    log.info(
        "\nBACKFILL COMPLETE: %d pages scraped across %.0f min",
        total_scraped, elapsed / 60,
    )
    log.info("Registry: %d total, %d parsed", registry.count_total(), registry.count_parsed())

    # Report remaining failures
    remaining_failed = load_failed(FAILED_LOG)
    still_failed = {k: v for k, v in remaining_failed.items() if not registry.is_downloaded(k)}
    if still_failed:
        log.warning(
            "%d downloads still failed. Run with --retry-failed to retry.",
            len(still_failed),
        )


if __name__ == "__main__":
    main()
