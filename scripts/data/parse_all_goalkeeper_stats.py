"""Parse cached FBref match reports into data/parsed/goalkeeper_stats.parquet.

Reads only HTML already on disk (scrape_fbref_missing does the fetching), so it
stays runnable while FBref is Cloudflare-blocked.

Invoked by scripts/pipeline/refresh_weekly_data.py Step 3:

    python3 -m scripts.data.parse_all_goalkeeper_stats --season 2025-2026 --append

--append replaces only the matches actually parsed, never a whole season. Rows
for matches whose HTML is no longer cached, and rows written under another
match_id convention, are left alone.

Usage:
    python3 -m scripts.data.parse_all_goalkeeper_stats --season 2025-2026 --append
    python3 -m scripts.data.parse_all_goalkeeper_stats --dry-run --season 2025-2026
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
from parser.goalkeeper_stats import parse_goalkeeper_stats
from parser.html_utils import (
    extract_match_date,
    extract_team_info_from_html,
    get_soup,
)

log = logging.getLogger(__name__)

OUTPUT_PATH = DATA_DIR / "parsed" / "goalkeeper_stats.parquet"

SKIP_PREFIXES = ("fixtures", "stats_")

# The parser returns every cell as a string. age stays a string ("27-045"),
# matching the stored parquet; everything else here is a real number.
NUMERIC_PREFIXES = ("gk_",)
NUMERIC_COLS = ["minutes"]


def _season_dir(season: str) -> Path:
    """Match reports live under the hyphen form (2025-2026); the underscore
    sibling (2025_2026) is the season-level stats dir."""
    return RAW_HTML_DIR / season


def parse_match_html(html_path: Path, season: str, fallback_id: str) -> list[dict]:
    try:
        html = html_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        log.warning("Cannot read %s: %s", html_path, e)
        return []

    soup = get_soup(html)
    teams = extract_team_info_from_html(soup)
    if not teams:
        return []

    # Canonical {date}_{home}_{away} from the report's own date+teams — NOT
    # html_path.stem. FBref 2025-26 reports are saved as {hash}.html, so keying by
    # the stem emits hash ids that never join matches.parquet; features/gk_quality.py
    # merges goalkeeper_stats on canonical match_id, so hash keys silently drop the
    # entire current season (GK features go null). fallback_id (the stem) is used
    # only if date/teams can't be read. Fixed 2026-07-17.
    home = next((t for t in teams if t["is_home"]), None)
    away = next((t for t in teams if not t["is_home"]), None)
    match_date = extract_match_date(soup)
    if match_date and home and away:
        match_id = f"{match_date}_{normalize_team(home['name'])}_{normalize_team(away['name'])}"
    else:
        match_id = fallback_id

    records: list[dict] = []
    for team in teams:
        rows = parse_goalkeeper_stats(
            soup=soup,
            team_hash=team["hash"],
            team_name=normalize_team(team["name"]),
            is_home=team["is_home"],
            match_id=match_id,
        )
        for r in rows:
            r["season"] = season
        records.extend(rows)

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

        log.info("  [%s] Done: %d files -> %d records (%d without a GK table)",
                 season, len(html_files), len(season_records), errors)
        all_records.extend(season_records)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    numeric = [
        c for c in df.columns
        if c in NUMERIC_COLS or c.startswith(NUMERIC_PREFIXES)
    ]
    for col in numeric:
        if df[col].dtype == object:
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
        "Merge: replacing %d rows for %d parsed matches, preserving %d rows",
        int(doomed.sum()), len(parsed_ids), len(preserved),
    )
    return pd.concat([preserved, df], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse cached FBref match reports into goalkeeper_stats.parquet",
    )
    # Required, deliberately: no safe all-seasons default. Only the season
    # the caller names is verified reproducible from the cached HTML; see
    # parse_all_player_stats for the 2024-25 case that proves the point.
    parser.add_argument(
        "--season", required=True, help="Season to parse (e.g. 2025-2026)",
    )
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

    seasons = [args.season]
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
