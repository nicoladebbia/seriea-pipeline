#!/usr/bin/env python3
"""Fully automated FBref scraper using a persistent browser session.

This scraper:
1. Opens ONE browser window and keeps it open
2. Navigates to each URL sequentially
3. Waits for Cloudflare to pass (your home IP should work)
4. Saves HTML files automatically
5. Respects rate limits (4+ seconds between requests)

Supports multiple leagues via the --league flag:
    - seriea (default): Serie A (comp ID 11)
    - epl: Premier League (comp ID 9)

Usage:
    python scraper/fbref_auto_scraper.py --seasons 2023-2024 2022-2023
    python scraper/fbref_auto_scraper.py --league epl --all-seasons
    python scraper/fbref_auto_scraper.py --league seriea --seasons 2024-2025
"""

import argparse
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)

# FBref configuration
FBREF_BASE = "https://fbref.com"

# League configurations: comp_id, URL slug, display name
LEAGUE_CONFIG = {
    "seriea": {
        "comp_id": 11,
        "url_slug": "Serie-A",
        "display_name": "Serie A",
    },
    "epl": {
        "comp_id": 9,
        "url_slug": "Premier-League",
        "display_name": "Premier League",
    },
}

RATE_LIMIT_SECONDS = 4  # FBref requires max 20 requests/minute
# EPL scraping uses longer delays to reduce risk of blocking
EPL_RATE_LIMIT_SECONDS = 8


def _build_pages_to_scrape(league: str) -> list[tuple[str, str]]:
    """Build the list of (page_name, url_template) for a given league."""
    cfg = LEAGUE_CONFIG[league]
    comp_id_ph = "{comp_id}"
    season_ph = "{season}"
    slug = cfg["url_slug"]

    return [
        ("fixtures", f"/en/comps/{comp_id_ph}/{season_ph}/schedule/{season_ph}-{slug}-Scores-and-Fixtures"),
        ("stats_standard", f"/en/comps/{comp_id_ph}/{season_ph}/stats/{season_ph}-{slug}-Stats"),
        ("stats_shooting", f"/en/comps/{comp_id_ph}/{season_ph}/shooting/{season_ph}-{slug}-Stats"),
        ("stats_passing", f"/en/comps/{comp_id_ph}/{season_ph}/passing/{season_ph}-{slug}-Stats"),
        ("stats_passing_types", f"/en/comps/{comp_id_ph}/{season_ph}/passing_types/{season_ph}-{slug}-Stats"),
        ("stats_gca", f"/en/comps/{comp_id_ph}/{season_ph}/gca/{season_ph}-{slug}-Stats"),
        ("stats_defense", f"/en/comps/{comp_id_ph}/{season_ph}/defense/{season_ph}-{slug}-Stats"),
        ("stats_possession", f"/en/comps/{comp_id_ph}/{season_ph}/possession/{season_ph}-{slug}-Stats"),
        ("stats_misc", f"/en/comps/{comp_id_ph}/{season_ph}/misc/{season_ph}-{slug}-Stats"),
        ("stats_keepers", f"/en/comps/{comp_id_ph}/{season_ph}/keepers/{season_ph}-{slug}-Stats"),
        ("stats_keepers_adv", f"/en/comps/{comp_id_ph}/{season_ph}/keepersadv/{season_ph}-{slug}-Stats"),
    ]


AVAILABLE_SEASONS = [
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


def scrape_fbref_seasons(
    seasons: list[str],
    output_dir: Path,
    league: str = "seriea",
    headless: bool = False,
    skip_existing: bool = True,
):
    """Scrape FBref data for multiple seasons using a single browser session.

    Args:
        seasons: List of seasons to scrape
        output_dir: Directory to save HTML files
        league: League key from LEAGUE_CONFIG (e.g., "seriea", "epl")
        headless: Run browser invisibly (may trigger Cloudflare more)
        skip_existing: Skip files that already exist
    """
    from botasaurus.browser import browser, Driver

    if league not in LEAGUE_CONFIG:
        raise ValueError(f"Unknown league '{league}'. Available: {list(LEAGUE_CONFIG.keys())}")

    cfg = LEAGUE_CONFIG[league]
    pages_to_scrape = _build_pages_to_scrape(league)
    rate_limit = EPL_RATE_LIMIT_SECONDS if league == "epl" else RATE_LIMIT_SECONDS

    log.info(f"League: {cfg['display_name']} (comp_id={cfg['comp_id']})")
    log.info(f"Rate limit: {rate_limit}s between requests")

    # Build list of all URLs to scrape
    tasks = []
    for season in seasons:
        # Use league suffix in directory name for non-seriea leagues
        if league == "seriea":
            season_dir = output_dir / season.replace("-", "_")
        else:
            season_dir = output_dir / f"{season.replace('-', '_')}_{league}"
        season_dir.mkdir(parents=True, exist_ok=True)

        for page_name, url_template in pages_to_scrape:
            filepath = season_dir / f"{page_name}.html"

            if skip_existing and filepath.exists() and filepath.stat().st_size > 10000:
                log.info(f"Skipping {season}/{page_name}.html (already exists)")
                continue

            url = FBREF_BASE + url_template.format(
                comp_id=cfg["comp_id"],
                season=season
            )
            tasks.append({
                "season": season,
                "page_name": page_name,
                "url": url,
                "filepath": filepath,
            })

    if not tasks:
        log.info("No files to scrape - all already downloaded!")
        return

    log.info(f"Will scrape {len(tasks)} pages...")

    # Define the scraper function with persistent browser
    @browser(
        headless=headless,
        block_images_and_css=False,  # Load fully for Cloudflare
        reuse_driver=True,  # CRITICAL: Keep browser open
        output=None,
        create_error_logs=False,
    )
    def scrape_page(driver: Driver, task: dict):
        """Scrape a single page."""
        url = task["url"]
        filepath = task["filepath"]
        page_name = task["page_name"]
        season = task["season"]

        log.info(f"Scraping {season}/{page_name}...")
        log.info(f"  URL: {url}")

        # Navigate to page
        driver.get(url)

        # Wait for FBref content to load (indicates Cloudflare passed)
        max_wait = 30
        start = time.time()
        while time.time() - start < max_wait:
            html = driver.page_html

            # Check if we got past Cloudflare
            if "body.fb" in html or 'class="fb"' in html:
                log.info(f"  ✓ Page loaded successfully")
                break

            # Check if still on Cloudflare
            if "Just a moment" in html:
                log.info(f"  Waiting for Cloudflare... ({int(time.time() - start)}s)")
                time.sleep(2)
                continue

            # Some other page - might be an error
            time.sleep(1)
        else:
            log.warning(f"  ✗ Timeout waiting for page to load")
            return {"success": False, "error": "timeout"}

        # Get final HTML
        html = driver.page_html

        # Verify content
        if "Just a moment" in html:
            log.error(f"  ✗ Still on Cloudflare challenge page!")
            return {"success": False, "error": "cloudflare"}

        if len(html) < 10000:
            log.error(f"  ✗ Page too small ({len(html)} bytes)")
            return {"success": False, "error": "small_page"}

        # Save HTML
        filepath.write_text(html, encoding="utf-8")
        log.info(f"  ✓ Saved {len(html):,} bytes to {filepath.name}")

        # Rate limiting
        log.info(f"  Waiting {rate_limit}s (rate limit)...")
        time.sleep(rate_limit)

        return {"success": True, "size": len(html)}

    # Run scraper for all tasks
    results = []
    for i, task in enumerate(tasks):
        log.info(f"\n[{i+1}/{len(tasks)}] {task['season']}/{task['page_name']}")
        try:
            result = scrape_page(task)
            results.append(result)
        except Exception as e:
            log.error(f"Error scraping {task['url']}: {e}")
            results.append({"success": False, "error": str(e)})

    # Summary
    success_count = sum(1 for r in results if r and r.get("success"))
    log.info(f"\n{'='*50}")
    log.info(f"SCRAPING COMPLETE: {success_count}/{len(tasks)} pages successful")

    if success_count < len(tasks):
        failed = [t for t, r in zip(tasks, results) if not r or not r.get("success")]
        log.warning(f"Failed pages:")
        for t in failed:
            log.warning(f"  - {t['season']}/{t['page_name']}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Automated FBref scraper")
    parser.add_argument(
        "--league",
        choices=list(LEAGUE_CONFIG.keys()),
        default="seriea",
        help="League to scrape (default: seriea). Options: seriea, epl",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=["2023-2024"],
        help="Seasons to scrape (e.g., 2023-2024 2022-2023)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/html"),
        help="Output directory for HTML files",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode (may trigger Cloudflare)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files exist",
    )
    parser.add_argument(
        "--all-seasons",
        action="store_true",
        help="Scrape all available seasons (2017-2026)",
    )
    args = parser.parse_args()

    seasons = AVAILABLE_SEASONS if args.all_seasons else args.seasons
    league_name = LEAGUE_CONFIG[args.league]["display_name"]

    log.info(f"FBref Auto Scraper")
    log.info(f"League: {league_name}")
    log.info(f"Seasons: {seasons}")
    log.info(f"Output: {args.output_dir}")
    log.info(f"Headless: {args.headless}")
    log.info(f"Skip existing: {not args.force}")
    log.info("")

    scrape_fbref_seasons(
        seasons=seasons,
        output_dir=args.output_dir,
        league=args.league,
        headless=args.headless,
        skip_existing=not args.force,
    )


if __name__ == "__main__":
    main()
