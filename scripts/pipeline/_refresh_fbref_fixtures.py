"""Refresh FBref fixtures.html (top-level schedule page) using botasaurus.

Called by refresh_weekly_data.py so scrape_fbref_missing sees the latest
published match list.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
HTML_DIR = PROJECT / "data" / "raw" / "html"

log = logging.getLogger(__name__)


def main(season: str) -> None:
    from botasaurus.browser import browser, Driver

    url = f"https://fbref.com/en/comps/11/{season}/schedule/{season}-Serie-A-Scores-and-Fixtures"
    out_dir = HTML_DIR / season.replace("-", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fixtures.html"

    log.info(f"Fetching: {url}")

    @browser(
        headless=True,
        block_images_and_css=False,
        wait_for_complete_page_load=True,
        reuse_driver=False,
        output=None,
        create_error_logs=False,
    )
    def fetch(driver: Driver, data: dict):
        driver.get(data["url"])
        try:
            driver.run_js("""
                document.querySelectorAll('.fc-dialog-overlay, .fc-consent-root, [class*="adblock"]').forEach(el => el.remove());
                document.querySelectorAll('button, a').forEach(btn => {
                    const text = btn.textContent.toLowerCase();
                    if (text.includes('continue without') || text.includes('dismiss') || text.includes('close') || text.includes('×')) {
                        btn.click();
                    }
                });
            """)
        except Exception:
            pass
        for attempt in range(3):
            try:
                driver.wait_for_element("body.fb", wait=20)
                break
            except Exception:
                if attempt < 2:
                    time.sleep(5)
                    driver.get(data["url"])
                    time.sleep(3)
        time.sleep(2)
        html = driver.page_html
        if len(html) < 50000:
            raise RuntimeError(f"Page too small ({len(html)} bytes)")
        return html

    html = fetch({"url": url})
    out_path.write_text(html, encoding="utf-8")
    log.info(f"Saved {len(html):,} bytes to {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    season = sys.argv[1] if len(sys.argv) > 1 else "2025-2026"
    main(season)
