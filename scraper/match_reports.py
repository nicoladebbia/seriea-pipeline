"""Download match report HTML pages from FBref (incremental)."""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

import pandas as pd

from config.settings import OLD_HTML_DIR
from scraper.client import FBrefClient
from scraper.registry import Registry
from storage.paths import fixtures_csv_path, html_match_path
from storage.raw import save_html

log = logging.getLogger(__name__)


def download_match_reports(
    season: str,
    limit: int | None = None,
    client: FBrefClient | None = None,
) -> dict[str, int]:
    """Download match report HTML for all fixtures in *season*.

    Skips matches already in the registry.
    Returns dict with keys: downloaded, skipped, failed.
    """
    if client is None:
        client = FBrefClient()

    registry = Registry()
    csv_path = fixtures_csv_path(season)

    if not csv_path.exists():
        log.error("Fixtures CSV not found at %s. Run scrape-fixtures first.", csv_path)
        return {"downloaded": 0, "skipped": 0, "failed": 0}

    df = pd.read_csv(csv_path)
    counts = {"downloaded": 0, "skipped": 0, "failed": 0}

    for _, row in df.iterrows():
        match_id = row["match_id"]
        url = row.get("match_report_url", "")

        if not url or pd.isna(url):
            continue

        if registry.is_downloaded(match_id):
            counts["skipped"] += 1
            continue

        if limit and counts["downloaded"] >= limit:
            log.info("Reached download limit (%d)", limit)
            break

        try:
            resp = client.get(url)
            html_path = save_html(season, match_id, resp.text)
            registry.mark_downloaded(
                match_id=match_id,
                season=season,
                match_url=url,
                html_path=str(html_path),
            )
            counts["downloaded"] += 1
            log.info("Downloaded %s (%d/%d)", match_id, counts["downloaded"], len(df))
        except Exception as exc:
            log.error("Failed to download %s: %s", match_id, exc)
            counts["failed"] += 1

    return counts


def import_old_html_files() -> int:
    """Import existing HTML files from the old project into the new structure.

    Looks in the old season_2024_2025/ directory and copies files into
    data/raw/html/2024-2025/ with proper naming.
    """
    registry = Registry()
    imported = 0
    season = "2024-2025"

    if not OLD_HTML_DIR.exists():
        log.warning("Old HTML directory not found: %s", OLD_HTML_DIR)
        return 0

    for html_file in sorted(OLD_HTML_DIR.glob("*.html")):
        # Parse match_id from filename like "Atalanta_VS_Bologna_2025-04-13.html"
        stem = html_file.stem
        match = re.match(r"(.+?)_VS_(.+?)_(\d{4}-\d{2}-\d{2})", stem)
        if match:
            home = match.group(1).replace("_", " ")
            away = match.group(2).replace("_", " ")
            date_str = match.group(3)
            match_id = f"{date_str}_{home}_{away}".replace(" ", "-")
        else:
            # Fallback: use filename as match_id
            match_id = stem.replace(" ", "-")

        if registry.is_downloaded(match_id):
            continue

        dest = html_match_path(season, match_id)
        shutil.copy2(html_file, dest)

        registry.mark_downloaded(
            match_id=match_id,
            season=season,
            match_url="",  # no URL for imported files
            html_path=str(dest),
        )
        imported += 1

    log.info("Imported %d HTML files from %s", imported, OLD_HTML_DIR)
    return imported
