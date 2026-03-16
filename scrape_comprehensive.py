#!/usr/bin/env python3
"""Comprehensive data scraper for Serie A.

This script gathers data from multiple sources:
1. Understat - xG, shot events (most reliable, no Cloudflare issues)
2. Football-Data.co.uk - Betting odds, basic stats (already have this)
3. ScraperFC/FBref - Detailed player stats (optional, uses browser)

Usage:
    python3 scrape_comprehensive.py --understat  # Scrape Understat data
    python3 scrape_comprehensive.py --all        # Scrape everything
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import soccerdata as sd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_scrape.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# Output directories
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
UNDERSTAT_DIR = DATA_DIR / "understat"
UNDERSTAT_DIR.mkdir(parents=True, exist_ok=True)

# Seasons available on Understat (format: last 2 digits of year pairs, e.g., "2324" = 2023-2024)
UNDERSTAT_SEASONS = [
    "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425"
]


def scrape_understat_schedule(seasons: list[str] = None) -> pd.DataFrame:
    """Scrape match schedule with xG from Understat."""
    seasons = seasons or UNDERSTAT_SEASONS

    log.info("=== Scraping Understat Schedule ===")

    all_schedules = []
    for season in seasons:
        cache_path = UNDERSTAT_DIR / f"schedule_{season}.parquet"

        if cache_path.exists():
            log.info(f"  {season}: Loading from cache")
            df = pd.read_parquet(cache_path)
        else:
            log.info(f"  {season}: Downloading...")
            try:
                us = sd.Understat(leagues='ITA-Serie A', seasons=[season])
                df = us.read_schedule()
                df['season_code'] = season
                df.to_parquet(cache_path, index=False)
                log.info(f"  {season}: Saved {len(df)} matches")
                time.sleep(2)  # Rate limiting
            except Exception as e:
                log.error(f"  {season}: Error - {e}")
                continue

        all_schedules.append(df)

    if all_schedules:
        combined = pd.concat(all_schedules, ignore_index=True)
        combined.to_parquet(UNDERSTAT_DIR / "all_schedules.parquet", index=False)
        log.info(f"Combined schedule: {len(combined)} matches")
        return combined

    return pd.DataFrame()


def scrape_understat_player_stats(seasons: list[str] = None) -> pd.DataFrame:
    """Scrape player season stats from Understat."""
    seasons = seasons or UNDERSTAT_SEASONS

    log.info("=== Scraping Understat Player Stats ===")

    all_players = []
    for season in seasons:
        cache_path = UNDERSTAT_DIR / f"player_stats_{season}.parquet"

        if cache_path.exists():
            log.info(f"  {season}: Loading from cache")
            df = pd.read_parquet(cache_path)
        else:
            log.info(f"  {season}: Downloading...")
            try:
                us = sd.Understat(leagues='ITA-Serie A', seasons=[season])
                df = us.read_player_season_stats()
                df['season_code'] = season
                df.to_parquet(cache_path, index=False)
                log.info(f"  {season}: Saved {len(df)} player records")
                time.sleep(2)
            except Exception as e:
                log.error(f"  {season}: Error - {e}")
                continue

        all_players.append(df)

    if all_players:
        combined = pd.concat(all_players, ignore_index=True)
        combined.to_parquet(UNDERSTAT_DIR / "all_player_stats.parquet", index=False)
        log.info(f"Combined player stats: {len(combined)} records")
        return combined

    return pd.DataFrame()


def scrape_understat_shot_events(seasons: list[str] = None) -> pd.DataFrame:
    """Scrape shot-level events from Understat.

    WARNING: This is slow - each match requires a separate request.
    """
    seasons = seasons or UNDERSTAT_SEASONS

    log.info("=== Scraping Understat Shot Events ===")
    log.info("NOTE: This is slow - each match requires a separate request")

    all_shots = []
    for season in seasons:
        cache_path = UNDERSTAT_DIR / f"shots_{season}.parquet"

        if cache_path.exists():
            log.info(f"  {season}: Loading from cache ({cache_path})")
            df = pd.read_parquet(cache_path)
        else:
            log.info(f"  {season}: Downloading shots (this takes several minutes)...")
            try:
                us = sd.Understat(leagues='ITA-Serie A', seasons=[season])
                df = us.read_shot_events()
                df['season_code'] = season
                df.to_parquet(cache_path, index=False)
                log.info(f"  {season}: Saved {len(df)} shot events")
            except Exception as e:
                log.error(f"  {season}: Error - {e}")
                continue

        all_shots.append(df)

    if all_shots:
        combined = pd.concat(all_shots, ignore_index=True)
        combined.to_parquet(UNDERSTAT_DIR / "all_shots.parquet", index=False)
        log.info(f"Combined shots: {len(combined)} events")
        return combined

    return pd.DataFrame()


def show_data_summary():
    """Show what data we have."""
    log.info("=== Current Data Summary ===")

    # Check Understat data
    if UNDERSTAT_DIR.exists():
        for f in sorted(UNDERSTAT_DIR.glob("*.parquet")):
            df = pd.read_parquet(f)
            log.info(f"  {f.name}: {len(df)} rows, {len(df.columns)} cols")

    # Check odds data
    odds_dir = DATA_DIR / "external" / "odds"
    if odds_dir.exists():
        csv_files = list(odds_dir.glob("*.csv"))
        log.info(f"  Odds CSVs: {len(csv_files)} files")

    # Check existing features
    features_path = DATA_DIR / "features" / "features.parquet"
    if features_path.exists():
        df = pd.read_parquet(features_path)
        log.info(f"  Features: {len(df)} matches, {len(df.columns)} columns")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Serie A data scraper")
    parser.add_argument("--understat", action="store_true", help="Scrape Understat data")
    parser.add_argument("--shots", action="store_true", help="Scrape shot events (slow)")
    parser.add_argument("--all", action="store_true", help="Scrape all available data")
    parser.add_argument("--seasons", nargs="+", help="Specific seasons (e.g., 2324 2223)")
    parser.add_argument("--summary", action="store_true", help="Show data summary only")
    args = parser.parse_args()

    seasons = args.seasons

    if args.summary:
        show_data_summary()
        return

    if args.understat or args.all:
        scrape_understat_schedule(seasons)
        scrape_understat_player_stats(seasons)

    if args.shots or args.all:
        scrape_understat_shot_events(seasons)

    show_data_summary()
    log.info("Done!")


if __name__ == "__main__":
    main()
