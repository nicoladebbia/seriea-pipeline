#!/usr/bin/env python3
"""Fully automated FBref scraper using a persistent browser session.

This scraper:
1. Opens ONE browser window and keeps it open
2. Navigates to each URL sequentially
3. Waits for Cloudflare to pass (your home IP should work)
4. Saves HTML files automatically
5. Respects rate limits (4+ seconds between requests)

Usage:
    python scraper/fbref_auto_scraper.py --seasons 2023-2024 2022-2023
"""

import argparse
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)

# FBref configuration
FBREF_BASE = "https://fbref.com"
SERIE_A_COMP_ID = 11
RATE_LIMIT_SECONDS = 4  # FBref requires max 20 requests/minute

# Pages to scrape per season
PAGES_TO_SCRAPE = [
    ("fixtures", "/en/comps/{comp_id}/{season}/schedule/{season}-Serie-A-Scores-and-Fixtures"),
    ("stats_standard", "/en/comps/{comp_id}/{season}/stats/{season}-Serie-A-Stats"),
    ("stats_shooting", "/en/comps/{comp_id}/{season}/shooting/{season}-Serie-A-Stats"),
    ("stats_passing", "/en/comps/{comp_id}/{season}/passing/{season}-Serie-A-Stats"),
    ("stats_passing_types", "/en/comps/{comp_id}/{season}/passing_types/{season}-Serie-A-Stats"),
    ("stats_gca", "/en/comps/{comp_id}/{season}/gca/{season}-Serie-A-Stats"),
    ("stats_defense", "/en/comps/{comp_id}/{season}/defense/{season}-Serie-A-Stats"),
    ("stats_possession", "/en/comps/{comp_id}/{season}/possession/{season}-Serie-A-Stats"),
    ("stats_misc", "/en/comps/{comp_id}/{season}/misc/{season}-Serie-A-Stats"),
    ("stats_keepers", "/en/comps/{comp_id}/{season}/keepers/{season}-Serie-A-Stats"),
    ("stats_keepers_adv", "/en/comps/{comp_id}/{season}/keepersadv/{season}-Serie-A-Stats"),
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
    headless: bool = False,
    skip_existing: bool = True,
):
    """Scrape FBref data for multiple seasons using a single browser session.

    Args:
        seasons: List of seasons to scrape
        output_dir: Directory to save HTML files
        headless: Run browser invisibly (may trigger Cloudflare more)
        skip_existing: Skip files that already exist
    """
    from botasaurus.browser import browser, Driver

    # Build list of all URLs to scrape
    tasks = []
    for season in seasons:
        season_dir = output_dir / season.replace("-", "_")
        season_dir.mkdir(parents=True, exist_ok=True)

        for page_name, url_template in PAGES_TO_SCRAPE:
            filepath = season_dir / f"{page_name}.html"

            if skip_existing and filepath.exists() and filepath.stat().st_size > 10000:
                log.info(f"Skipping {season}/{page_name}.html (already exists)")
                continue

            url = FBREF_BASE + url_template.format(
                comp_id=SERIE_A_COMP_ID,
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
        log.info(f"  Waiting {RATE_LIMIT_SECONDS}s (rate limit)...")
        time.sleep(RATE_LIMIT_SECONDS)

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
        help="Scrape all available seasons (2017-2024)",
    )
    args = parser.parse_args()

    seasons = AVAILABLE_SEASONS if args.all_seasons else args.seasons

    log.info(f"FBref Auto Scraper")
    log.info(f"Seasons: {seasons}")
    log.info(f"Output: {args.output_dir}")
    log.info(f"Headless: {args.headless}")
    log.info(f"Skip existing: {not args.force}")
    log.info("")

    scrape_fbref_seasons(
        seasons=seasons,
        output_dir=args.output_dir,
        headless=args.headless,
        skip_existing=not args.force,
    )


if __name__ == "__main__":
    main()
