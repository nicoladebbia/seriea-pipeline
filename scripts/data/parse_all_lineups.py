"""Parse cached FBref match reports into data/parsed/lineups.parquet.

Reads only HTML already on disk (scrape_fbref_missing does the fetching), so it
stays runnable while FBref is Cloudflare-blocked.

Invoked by scripts/pipeline/refresh_weekly_data.py Step 3:

    python3 -m scripts.data.parse_all_lineups --season 2025-2026 --append

lineups.parquet is keyed by more than one match_id convention: this module
writes FBref 8-hex ids, while other writers use canonical date_Home_Away ids
(2025-26 holds 346 of the former and 593 of the latter). --append therefore
replaces only the matches actually parsed — replacing by season would delete
tens of thousands of rows this module does not own.

Usage:
    python3 -m scripts.data.parse_all_lineups --season 2025-2026 --append
    python3 -m scripts.data.parse_all_lineups --dry-run --season 2025-2026
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import DATA_DIR, RAW_HTML_DIR
from config.team_names import normalize_team
from parser.html_utils import extract_team_info_from_html, get_soup
from parser.lineups import lineups_to_records, parse_lineups

log = logging.getLogger(__name__)

OUTPUT_PATH = DATA_DIR / "parsed" / "lineups.parquet"

SKIP_PREFIXES = ("fixtures", "stats_")

NUMERIC_COLS = ["shirt_number"]


def _season_dir(season: str) -> Path:
    """Match reports live under the hyphen form (2025-2026); the underscore
    sibling (2025_2026) is the season-level stats dir."""
    return RAW_HTML_DIR / season


def discover_seasons() -> list[str]:
    return sorted(
        d.name for d in RAW_HTML_DIR.iterdir()
        if d.is_dir() and "_" not in d.name and d.name[:4].isdigit()
    )


def parse_match_html(html_path: Path, season: str, match_id: str) -> list[dict]:
    try:
        html = html_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        log.warning("Cannot read %s: %s", html_path, e)
        return []

    soup = get_soup(html)
    teams = extract_team_info_from_html(soup)
    if len(teams) != 2:
        return []

    home = next((t for t in teams if t["is_home"]), None)
    away = next((t for t in teams if not t["is_home"]), None)
    if home is None or away is None:
        return []

    home_lineup, away_lineup = parse_lineups(soup)
    records = lineups_to_records(
        home_lineup=home_lineup,
        away_lineup=away_lineup,
        match_id=match_id,
        home_team=normalize_team(home["name"]),
        away_team=normalize_team(away["name"]),
    )
    for r in records:
        r["season"] = season
    return records


def parse_seasons(seasons: list[str]) -> pd.DataFrame:
    all_records: list[dict] = []

    for season in seasons:
        season_dir = _season_dir(season)
        if not season_dir.exists():
            log.warning("No directory for %s at %s", season, season_dir)
            continue

        html_files = sorted(
            f for f in season_dir.glob("*.html")
            if not f.name.startswith(SKIP_PREFIXES)
        )
        if not html_files:
            log.info("No match HTMLs for %s", season)
            continue

        log.info("Parsing %d match HTMLs for %s...", len(html_files), season)
        season_records: list[dict] = []
        errors = 0
        for i, html_path in enumerate(html_files):
            rows = parse_match_html(html_path, season, html_path.stem)
            if rows:
                season_records.extend(rows)
            else:
                errors += 1
            if (i + 1) % 100 == 0:
                log.info("  [%s] %d/%d files (%d records)",
                         season, i + 1, len(html_files), len(season_records))

        log.info("  [%s] Done: %d files -> %d records (%d without a lineup)",
                 season, len(html_files), len(season_records), errors)
        all_records.extend(season_records)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    for col in NUMERIC_COLS:
        if col in df.columns and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def merge_into_existing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace only the parsed matches; leave every other row untouched."""
    if not OUTPUT_PATH.exists():
        return df

    try:
        existing = pd.read_parquet(OUTPUT_PATH)
    except (OSError, ValueError) as e:
        log.warning("Cannot read existing parquet (will overwrite): %s", e)
        return df

    parsed_ids = set(df["match_id"].unique())
    doomed = existing["match_id"].isin(parsed_ids)
    preserved = existing[~doomed]
    log.info(
        "Merge: replacing %d rows for %d parsed matches, preserving %d rows "
        "(%d matches under other id conventions)",
        int(doomed.sum()), len(parsed_ids), len(preserved),
        preserved["match_id"].nunique(),
    )
    return pd.concat([preserved, df], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse cached FBref match reports into lineups.parquet",
    )
    parser.add_argument("--season", help="Only this season (e.g. 2025-2026)")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Merge into the existing parquet, replacing only the parsed "
             "matches (default behaviour; accepted for callers that pass it)",
    )
    parser.add_argument(
        "--replace-all",
        action="store_true",
        help="DESTRUCTIVE: rebuild the parquet from the parsed matches only",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and report, but do not write the parquet")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    seasons = [args.season] if args.season else discover_seasons()
    log.info("Seasons: %s", seasons)

    df = parse_seasons(seasons)
    if df.empty:
        log.error("No data parsed for %s — nothing written", seasons)
        sys.exit(1)

    log.info("Parsed %d rows, %d columns", len(df), len(df.columns))

    if not args.replace_all:
        df = merge_into_existing(df)
    else:
        log.warning("--replace-all: dropping every match not parsed now")

    if args.dry_run:
        log.info("DRY RUN — would write %d rows to %s", len(df), OUTPUT_PATH)
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    log.info("Saved %s: %d rows, %d columns", OUTPUT_PATH, len(df), len(df.columns))


if __name__ == "__main__":
    main()
