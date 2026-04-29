"""Scrape EPL match reports from FBref and parse into player_stats_epl.parquet.

This script bridges the gap between Serie A (which has match-level player stats)
and EPL (which only has season-level aggregates). It:

  1. Reads EPL fixtures.html files already scraped by fbref_auto_scraper.py
  2. Extracts match report URLs for completed matches
  3. Downloads each match report HTML (incremental — skips existing)
  4. Parses all match HTMLs into per-player-per-match stats
  5. Saves to data/parsed/player_stats_epl.parquet (same schema as Serie A)

After running this, retrain the EPL model to pick up ~57 new FBref rolling features
(xg, sca, gca, progressive passes, carries, etc.).

Usage:
    # Full scrape (all seasons, ~5.5h at 8s rate limit)
    python3 -m scripts.data.scrape_epl_match_reports

    # Single season
    python3 -m scripts.data.scrape_epl_match_reports --season 2024-2025

    # Parse only (skip downloading, just rebuild parquet from existing HTMLs)
    python3 -m scripts.data.scrape_epl_match_reports --parse-only

    # Dry run (show what would be downloaded)
    python3 -m scripts.data.scrape_epl_match_reports --dry-run
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import DATA_DIR, RAW_HTML_DIR
from config.team_names import normalize_team
from parser.html_utils import _uncomment_tables
from parser.player_stats import parse_player_stats
from scripts.data.parse_all_player_stats import (
    extract_match_date,
    extract_team_info_from_html,
)

log = logging.getLogger(__name__)

FBREF_BASE = "https://fbref.com"
OUTPUT_PATH = DATA_DIR / "parsed" / "player_stats_epl.parquet"
RATE_LIMIT = 8  # seconds between requests (EPL needs longer delays)

SEASONS = [
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


def _epl_season_dir(season: str) -> Path:
    """Return the EPL HTML directory for a season."""
    return RAW_HTML_DIR / f"{season.replace('-', '_')}_epl"



def extract_match_urls(season: str) -> list[dict]:
    """Extract match report URLs from a season's fixtures.html.

    Returns list of dicts with keys: match_id, date, home, away, url, score
    """
    season_dir = _epl_season_dir(season)
    fixtures_html = season_dir / "fixtures.html"

    if not fixtures_html.exists():
        log.warning("No fixtures.html for EPL %s at %s", season, fixtures_html)
        return []

    html = fixtures_html.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    _uncomment_tables(soup)

    table = (
        soup.find("table", id=re.compile(r"sched.*_1"))
        or soup.find("table", {"class": "stats_table"})
    )
    if not table:
        log.warning("No fixture table found in EPL %s", season)
        return []

    tbody = table.find("tbody")
    if not tbody:
        return []

    matches = []
    for tr in tbody.find_all("tr"):
        if tr.get("class") and "spacer" in tr.get("class", []):
            continue
        if tr.find("th", {"colspan": True}):
            continue

        cells = {}
        for td in tr.find_all(["th", "td"]):
            stat = td.get("data-stat")
            if stat:
                cells[stat] = td

        if not cells:
            continue

        # Teams
        home_el = cells.get("home_team") or cells.get("squad_a")
        away_el = cells.get("away_team") or cells.get("squad_b")
        home = normalize_team(home_el.get_text(strip=True)) if home_el else ""
        away = normalize_team(away_el.get_text(strip=True)) if away_el else ""
        if not home or not away:
            continue

        # Score — only scrape completed matches
        score_el = cells.get("score")
        if not score_el:
            continue
        score_text = score_el.get_text(strip=True)
        if not re.match(r"\d+\s*[–\-:]\s*\d+", score_text):
            continue  # Match not yet played

        # Date
        date_el = cells.get("date")
        date_text = date_el.get_text(strip=True) if date_el else ""

        # Match report URL
        report_el = cells.get("match_report")
        url = ""
        if report_el:
            a_tag = report_el.find("a")
            if a_tag and a_tag.get("href"):
                href = a_tag["href"]
                url = href if href.startswith("http") else FBREF_BASE + href

        if not url:
            continue

        # Build match_id from URL (FBref hash)
        # URL format: /en/matches/{hash}/Team1-Team2-...
        url_parts = url.rstrip("/").split("/")
        match_hash = url_parts[-2] if len(url_parts) >= 2 else ""

        match_id = f"{date_text}_{home}_{away}".replace(" ", "-") if date_text else match_hash

        matches.append({
            "match_id": match_id,
            "date": date_text,
            "home": home,
            "away": away,
            "url": url,
            "score": score_text,
            "hash": match_hash,
        })

    log.info("EPL %s: found %d completed match URLs", season, len(matches))
    return matches


def _is_fbref_loaded(html: str) -> bool:
    """Check if FBref content has loaded (past Cloudflare)."""
    return (
        len(html) > 50000
        and "Just a moment" not in html
    )


def download_match_reports(
    season: str,
    matches: list[dict],
    dry_run: bool = False,
    headless: bool = False,
) -> dict[str, int]:
    """Download match report HTMLs for a season using a persistent browser.

    FBref uses Cloudflare Turnstile which requires a VISIBLE browser to solve.
    Default is headless=False. Uses botasaurus with google_get + scrolling.

    Saves to data/raw/html/{season_epl}/{match_id}.html.
    Skips files that already exist and are >5KB.
    """
    from botasaurus.browser import browser, Driver

    season_dir = _epl_season_dir(season)
    season_dir.mkdir(parents=True, exist_ok=True)

    counts = {"downloaded": 0, "skipped": 0, "failed": 0}

    # Filter to only matches that need downloading
    to_download = []
    for m in matches:
        html_path = season_dir / f"{m['match_id']}.html"
        if html_path.exists() and html_path.stat().st_size > 5000:
            counts["skipped"] += 1
        else:
            to_download.append(m)

    if dry_run:
        log.info(
            "DRY RUN — %s: %d matches total, %d already downloaded, %d to fetch",
            season, len(matches), counts["skipped"], len(to_download),
        )
        return counts

    if not to_download:
        log.info("[%s] All %d matches already downloaded", season, len(matches))
        return counts

    log.info(
        "[%s] Downloading %d match reports (%d already exist)...",
        season, len(to_download), counts["skipped"],
    )

    @browser(
        headless=headless,
        block_images_and_css=False,
        wait_for_complete_page_load=False,  # We handle waiting ourselves
        reuse_driver=True,
        output=None,
        create_error_logs=False,
        raise_exception=False,  # Don't pause browser on errors
    )
    def fetch_page(driver: Driver, url: str) -> str:
        """Fetch a single FBref match report page.

        First request solves Cloudflare Turnstile (needs visible browser,
        ~6s). Subsequent requests in the same session load instantly.
        Scrolls page to trigger lazy-loaded stat tables.
        """
        try:
            driver.get(url, timeout=40)
        except Exception:
            # Timeout or navigation error — page may still be usable
            pass

        # Wait for match report content (scorebox) or Cloudflare to pass
        for i in range(25):
            try:
                html = driver.page_html
            except Exception:
                time.sleep(2)
                continue
            if "scorebox" in html:
                break
            if "Just a moment" in html:
                time.sleep(2)
                continue
            time.sleep(2)
        else:
            # Retry with reload
            try:
                driver.reload()
            except Exception:
                return ""
            for i in range(10):
                try:
                    html = driver.page_html
                except Exception:
                    time.sleep(2)
                    continue
                if "scorebox" in html:
                    break
                time.sleep(2)
            else:
                return ""

        # Scroll incrementally to trigger IntersectionObserver for ALL
        # lazy-loaded stat tables (passing, defense, possession, misc).
        try:
            page_height = driver.run_js("return document.body.scrollHeight;")
            step = 800  # px per scroll step
            for pos in range(0, page_height + step, step):
                driver.run_js(f"window.scrollTo(0, {pos});")
                time.sleep(0.3)
            time.sleep(2)
            driver.run_js("window.scrollTo(0, 0);")
            time.sleep(1)
        except Exception:
            pass  # Scroll failure is non-fatal

        try:
            return driver.page_html
        except Exception:
            return ""

    consecutive_failures = 0

    for i, match in enumerate(to_download):
        html_path = season_dir / f"{match['match_id']}.html"

        try:
            html = fetch_page(match["url"])

            if not html or "scorebox" not in html:
                log.warning("  Failed %s (blocked or empty)", match["match_id"])
                counts["failed"] += 1
                consecutive_failures += 1
            else:
                html_path.write_text(html, encoding="utf-8")
                counts["downloaded"] += 1
                consecutive_failures = 0

            # Progress log
            total_done = counts["downloaded"] + counts["failed"]
            if total_done % 20 == 0 or counts["downloaded"] == 1:
                log.info(
                    "  [%s] Progress: %d/%d downloaded, %d failed",
                    season, counts["downloaded"], len(to_download),
                    counts["failed"],
                )

            # Rate limit between requests
            time.sleep(RATE_LIMIT)

            # Back off on repeated failures
            if consecutive_failures >= 3:
                wait = min(60 * consecutive_failures, 300)
                log.warning(
                    "  %d consecutive failures — backing off %ds",
                    consecutive_failures, wait,
                )
                time.sleep(wait)

            if consecutive_failures >= 10:
                log.error(
                    "  10 consecutive failures — aborting season %s. "
                    "Re-run to retry remaining matches.",
                    season,
                )
                break

        except Exception as e:
            log.error("  Exception for %s: %s", match["match_id"], e)
            counts["failed"] += 1
            consecutive_failures += 1

    log.info(
        "[%s] Download complete: %d downloaded, %d skipped, %d failed",
        season, counts["downloaded"], counts["skipped"], counts["failed"],
    )
    return counts


def parse_epl_match_html(html_path: Path, season: str, match_id: str) -> list[dict]:
    """Parse a single EPL match report HTML into player stat records.

    Uses the same parser as Serie A (parser/player_stats.py).
    """
    try:
        html = html_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        log.warning("Cannot read %s: %s", html_path, e)
        return []

    from parser.html_utils import get_soup
    soup = get_soup(html)

    teams = extract_team_info_from_html(soup)
    if not teams:
        log.debug("No team hashes found in %s", html_path.name)
        return []

    match_date = extract_match_date(soup)

    all_records = []
    for team in teams:
        records = parse_player_stats(
            soup=soup,
            team_hash=team["hash"],
            team_name=normalize_team(team["name"]),
            is_home=team["is_home"],
            match_id=match_id,
        )
        for r in records:
            r["season"] = season
            r["match_date"] = match_date
            r["league"] = "epl"
            r["data_source"] = "fbref_match_epl"

        all_records.extend(records)

    return all_records


def parse_all_epl_seasons(seasons: list[str]) -> pd.DataFrame:
    """Parse all downloaded EPL match HTMLs into a single DataFrame."""
    # Skip patterns — season-level stats, not match reports
    SKIP_PREFIXES = ("fixtures", "stats_")

    all_records = []
    for season in seasons:
        season_dir = _epl_season_dir(season)
        if not season_dir.exists():
            log.warning("No EPL directory for %s", season)
            continue

        html_files = sorted(
            f for f in season_dir.glob("*.html")
            if not any(f.name.startswith(p) for p in SKIP_PREFIXES)
        )

        if not html_files:
            log.info("No match HTMLs for EPL %s", season)
            continue

        log.info("Parsing %d match HTMLs for EPL %s...", len(html_files), season)
        season_records = []
        errors = 0

        for i, html_path in enumerate(html_files):
            match_id = html_path.stem
            records = parse_epl_match_html(html_path, season, match_id)

            if records:
                season_records.extend(records)
            else:
                errors += 1

            if (i + 1) % 50 == 0:
                log.info(
                    "  [EPL %s] Parsed %d/%d files (%d records)",
                    season, i + 1, len(html_files), len(season_records),
                )

        log.info(
            "  [EPL %s] Done: %d files → %d records (%d errors)",
            season, len(html_files), len(season_records), errors,
        )
        all_records.extend(season_records)

    if not all_records:
        log.warning("No EPL player stats parsed")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    log.info("Total EPL parsed: %d rows, %d columns", len(df), len(df.columns))
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Scrape EPL match reports from FBref and build player_stats_epl.parquet"
    )
    parser.add_argument(
        "--season",
        help="Process only this season (e.g., 2024-2025)",
    )
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Skip downloading, only parse existing HTML files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode (will fail if Cloudflare Turnstile is active)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    seasons = [args.season] if args.season else SEASONS

    # Step 1: Extract match URLs and download
    if not args.parse_only:
        total_to_download = 0
        total_downloaded = 0

        for season in seasons:
            matches = extract_match_urls(season)
            if not matches:
                continue

            counts = download_match_reports(
                season, matches,
                dry_run=args.dry_run,
                headless=args.headless,
            )
            total_to_download += len(matches) - counts["skipped"]
            total_downloaded += counts["downloaded"]

        if args.dry_run:
            log.info("DRY RUN complete. Use without --dry-run to download.")
            return

        log.info(
            "Download phase complete: %d new files downloaded",
            total_downloaded,
        )

    # Step 2: Parse all HTMLs into parquet
    log.info("\nParsing EPL match HTMLs into player_stats_epl.parquet...")
    df = parse_all_epl_seasons(seasons)

    if df.empty:
        log.warning("No data parsed. Run without --parse-only first to download match reports.")
        sys.exit(1)

    # Coerce numeric columns (FBref parser emits strings)
    NUMERIC_COLS = [
        "shirtnumber", "age", "minutes", "goals", "assists",
        "pens_made", "pens_att", "shots", "shots_on_target",
        "cards_yellow", "cards_red", "fouls", "fouled",
        "offsides", "crosses", "tackles_won", "interceptions",
        "own_goals", "pens_won", "pens_conceded",
    ]
    for _col in NUMERIC_COLS:
        if _col in df.columns and df[_col].dtype == object:
            df[_col] = pd.to_numeric(df[_col], errors="coerce")

    # Merge with existing parquet: replace same-season rows, preserve others.
    # Prevents `--season X` runs from wiping other seasons.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        try:
            existing = pd.read_parquet(OUTPUT_PATH)
            new_seasons = set(df["season"].unique()) if "season" in df.columns else set()
            if new_seasons and "season" in existing.columns:
                preserved = existing[~existing["season"].isin(new_seasons)]
                log.info(
                    "Merge: preserving %d rows from other seasons, replacing %d rows in %s",
                    len(preserved),
                    int(existing["season"].isin(new_seasons).sum()),
                    sorted(new_seasons),
                )
                df = pd.concat([preserved, df], ignore_index=True)
        except Exception as e:
            log.warning("Merge with existing parquet failed (will overwrite): %s", e)

    df.to_parquet(OUTPUT_PATH, index=False)

    log.info(
        "Saved player_stats_epl.parquet: %d rows, %d columns at %s",
        len(df), len(df.columns), OUTPUT_PATH,
    )

    # Summary
    log.info("\nPer-season summary:")
    for season, grp in df.groupby("season"):
        log.info(
            "  %s: %d rows, %d players, %d matches",
            season, len(grp),
            grp["player"].nunique() if "player" in grp.columns else 0,
            grp["match_id"].nunique(),
        )

    # Coverage check
    for col in ["xg", "npxg", "sca", "gca", "xg_assist", "progressive_passes"]:
        if col in df.columns:
            pct = df[col].notna().mean() * 100
            log.info("  %s coverage: %.1f%%", col, pct)
        else:
            log.info("  %s: NOT IN OUTPUT (parser may need update)", col)


if __name__ == "__main__":
    main()
