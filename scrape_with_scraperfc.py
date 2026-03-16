#!/usr/bin/env python3
"""Scrape FBref data using ScraperFC library.

ScraperFC uses botasaurus for browser automation which bypasses Cloudflare.
This script scrapes match reports and player stats for Serie A seasons.

Usage:
    python3 scrape_with_scraperfc.py --seasons 2023-2024 2022-2023
    python3 scrape_with_scraperfc.py --all  # All seasons from 2017-2018 to present
"""

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('scraperfc_log.txt'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# Output directories
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
SCRAPERFC_DIR = DATA_DIR / "scraperfc"
SCRAPERFC_DIR.mkdir(parents=True, exist_ok=True)

# Seasons to scrape (where FBref has detailed match reports)
SEASONS_TO_SCRAPE = [
    "2017-2018",
    "2018-2019",
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025",
]


def scrape_season_matches(fbref, season: str) -> pd.DataFrame:
    """Scrape all matches for a season and return as DataFrame."""
    output_file = SCRAPERFC_DIR / f"matches_{season.replace('-', '_')}.parquet"

    if output_file.exists():
        log.info(f"  {season}: Already scraped, loading from cache")
        return pd.read_parquet(output_file)

    log.info(f"  {season}: Scraping matches...")
    try:
        matches = fbref.scrape_matches(year=season, league='Italy Serie A')

        if not matches:
            log.warning(f"  {season}: No matches returned")
            return pd.DataFrame()

        df = pd.DataFrame(matches)
        df['season'] = season
        df.to_parquet(output_file, index=False)
        log.info(f"  {season}: Saved {len(df)} matches to {output_file}")
        return df

    except Exception as e:
        log.error(f"  {season}: Error scraping matches: {e}")
        return pd.DataFrame()


def scrape_match_details(fbref, match_url: str, match_id: str) -> dict:
    """Scrape detailed stats for a single match."""
    try:
        data = fbref.scrape_match(link=match_url)
        return data
    except Exception as e:
        log.warning(f"  Match {match_id}: Error: {e}")
        return {}


def scrape_season_match_details(fbref, season: str, matches_df: pd.DataFrame) -> None:
    """Scrape detailed match reports for all matches in a season."""
    details_dir = SCRAPERFC_DIR / f"match_details_{season.replace('-', '_')}"
    details_dir.mkdir(exist_ok=True)

    # Find URL column (ScraperFC returns different column names)
    url_col = None
    for col in ['Match Report', 'match_report', 'link', 'url']:
        if col in matches_df.columns:
            url_col = col
            break

    if url_col is None:
        log.warning(f"  {season}: No URL column found in matches")
        log.info(f"  Available columns: {matches_df.columns.tolist()}")
        return

    total = len(matches_df)
    scraped = 0
    skipped = 0

    for idx, row in matches_df.iterrows():
        match_url = row.get(url_col)
        if not match_url or pd.isna(match_url):
            continue

        # Create a match ID from the URL
        match_id = match_url.split('/')[-2] if '/' in str(match_url) else f"match_{idx}"
        output_file = details_dir / f"{match_id}.json"

        if output_file.exists():
            skipped += 1
            continue

        log.info(f"  {season} match {scraped+1}/{total-skipped}: {match_id}")

        try:
            data = fbref.scrape_match(link=match_url)

            # Convert DataFrames in the result to dicts for JSON serialization
            serializable_data = {}
            for key, value in data.items():
                if isinstance(value, pd.DataFrame):
                    serializable_data[key] = value.to_dict(orient='records')
                else:
                    serializable_data[key] = value

            with open(output_file, 'w') as f:
                json.dump(serializable_data, f, indent=2, default=str)

            scraped += 1

            # Rate limiting - be nice to FBref
            time.sleep(4)

        except Exception as e:
            log.error(f"  Match {match_id}: {e}")
            continue

    log.info(f"  {season}: Scraped {scraped} new matches, {skipped} already cached")


def main():
    parser = argparse.ArgumentParser(description="Scrape FBref data using ScraperFC")
    parser.add_argument("--seasons", nargs="+", help="Specific seasons to scrape")
    parser.add_argument("--all", action="store_true", help="Scrape all seasons")
    parser.add_argument("--matches-only", action="store_true", help="Only scrape match list, not details")
    args = parser.parse_args()

    from ScraperFC import FBref
    fbref = FBref()

    seasons = SEASONS_TO_SCRAPE if args.all else (args.seasons or ["2023-2024"])

    log.info(f"ScraperFC FBref Scraper")
    log.info(f"Seasons to scrape: {seasons}")
    log.info(f"Output directory: {SCRAPERFC_DIR}")
    log.info("")

    all_matches = []

    for season in seasons:
        log.info(f"=== Season {season} ===")

        # Step 1: Get match list
        matches_df = scrape_season_matches(fbref, season)
        if matches_df.empty:
            continue
        all_matches.append(matches_df)

        # Step 2: Get detailed match reports (unless --matches-only)
        if not args.matches_only:
            scrape_season_match_details(fbref, season, matches_df)

        log.info("")

    # Combine all matches
    if all_matches:
        combined = pd.concat(all_matches, ignore_index=True)
        combined_file = SCRAPERFC_DIR / "all_matches_combined.parquet"
        combined.to_parquet(combined_file, index=False)
        log.info(f"Combined {len(combined)} matches saved to {combined_file}")

    log.info("Done!")


if __name__ == "__main__":
    main()
