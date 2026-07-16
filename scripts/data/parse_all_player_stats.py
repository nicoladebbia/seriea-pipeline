"""Parse cached FBref match reports into data/parsed/player_stats.parquet.

Serie A counterpart of scrape_epl_match_reports.py's parse phase. This module
only ever reads HTML that is already on disk — scrape_fbref_missing is what
fetches it. That split is deliberate: parsing stays runnable while FBref is
Cloudflare-blocked.

Invoked by scripts/pipeline/refresh_weekly_data.py Step 3:

    python3 -m scripts.data.parse_all_player_stats --season 2025-2026 --append

player_stats.parquet is multi-source: this module owns the data_source=
"fbref_match" rows, and fallback_sofascore_to_fbref owns "sofascore_fallback"
rows for matches FBref never served. --append therefore replaces only this
module's own (season, fbref_match) slice and leaves every other row alone.

Usage:
    python3 -m scripts.data.parse_all_player_stats --season 2025-2026 --append
    python3 -m scripts.data.parse_all_player_stats --dry-run --season 2025-2026
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
from parser.html_utils import extract_match_date, extract_team_info_from_html, get_soup
from parser.player_stats import parse_player_stats

log = logging.getLogger(__name__)

OUTPUT_PATH = DATA_DIR / "parsed" / "player_stats.parquet"
DATA_SOURCE = "fbref_match"

# Season-level pages (fixtures.html, stats_defense.html, ...) live in the
# underscore sibling dir, but skip by name too — a stray copy must not be
# parsed as a match report.
SKIP_PREFIXES = ("fixtures", "stats_")

# Coerced to numbers by this driver; parse_player_stats returns every cell as a
# string. age ("27-045") and pens_won/pens_conceded stay strings — that is what
# the stored parquet holds, and widening it here would silently change dtypes
# under everything downstream.
NUMERIC_COLS = [
    "shirtnumber", "minutes", "goals", "assists", "pens_made", "pens_att",
    "shots", "shots_on_target", "cards_yellow", "cards_red", "fouls", "fouled",
    "offsides", "crosses", "tackles_won", "interceptions", "own_goals",
]


def _season_dir(season: str) -> Path:
    """Serie A match reports live under the hyphen form (2025-2026).

    The underscore form (2025_2026) is the season-level stats dir. Mixing them
    up yields an empty parse that still exits 0.
    """
    return RAW_HTML_DIR / season


def discover_seasons() -> list[str]:
    """Every season dir holding match reports, oldest first."""
    return sorted(
        d.name for d in RAW_HTML_DIR.iterdir()
        if d.is_dir() and "_" not in d.name and d.name[:4].isdigit()
    )


def parse_match_html(html_path: Path, season: str, match_id: str) -> list[dict]:
    """Parse one match report into per-player records (both teams)."""
    try:
        html = html_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        log.warning("Cannot read %s: %s", html_path, e)
        return []

    soup = get_soup(html)

    teams = extract_team_info_from_html(soup)
    if not teams:
        log.debug("No team hashes in %s", html_path.name)
        return []

    match_date = extract_match_date(soup)

    records: list[dict] = []
    for team in teams:
        rows = parse_player_stats(
            soup=soup,
            team_hash=team["hash"],
            team_name=normalize_team(team["name"]),
            is_home=team["is_home"],
            match_id=match_id,
        )
        for r in rows:
            r["season"] = season
            r["match_date"] = match_date
        records.extend(rows)

    return records


def parse_seasons(seasons: list[str]) -> pd.DataFrame:
    """Parse every cached match report for the given seasons."""
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
            if (i + 1) % 50 == 0:
                log.info(
                    "  [%s] %d/%d files (%d records)",
                    season, i + 1, len(html_files), len(season_records),
                )

        log.info(
            "  [%s] Done: %d files -> %d records (%d errors)",
            season, len(html_files), len(season_records), errors,
        )
        all_records.extend(season_records)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    for col in NUMERIC_COLS:
        if col in df.columns and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def merge_into_existing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace this module's own (season, fbref_match) slice; keep the rest.

    Anything written by another source — notably the sofascore_fallback rows
    for matches FBref never served — survives untouched.
    """
    if not OUTPUT_PATH.exists():
        return df

    try:
        existing = pd.read_parquet(OUTPUT_PATH)
    except (OSError, ValueError) as e:
        log.warning("Cannot read existing parquet (will overwrite): %s", e)
        return df

    seasons = set(df["season"].unique())
    if "season" not in existing.columns or "data_source" not in existing.columns:
        log.warning("Existing parquet lacks season/data_source; overwriting")
        return df

    doomed = existing["season"].isin(seasons) & (existing["data_source"] == DATA_SOURCE)
    preserved = existing[~doomed]
    log.info(
        "Merge: replacing %d %s rows in %s, preserving %d others (%s)",
        int(doomed.sum()), DATA_SOURCE, sorted(seasons), len(preserved),
        ", ".join(
            f"{s}={n}" for s, n in
            preserved[preserved["season"].isin(seasons)]["data_source"]
            .value_counts().items()
        ) or "none in these seasons",
    )
    return pd.concat([preserved, df], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse cached FBref match reports into player_stats.parquet",
    )
    parser.add_argument("--season", help="Only this season (e.g. 2025-2026)")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Merge into the existing parquet, replacing only this season's "
             "fbref_match rows (default behaviour; accepted for callers that "
             "pass it explicitly)",
    )
    parser.add_argument(
        "--replace-all",
        action="store_true",
        help="DESTRUCTIVE: rebuild the parquet from the parsed seasons only, "
             "dropping every other season and source",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and report, but do not write the parquet",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

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
        log.warning("--replace-all: dropping every season/source not parsed now")

    if args.dry_run:
        log.info("DRY RUN — would write %d rows to %s", len(df), OUTPUT_PATH)
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    log.info("Saved %s: %d rows, %d columns", OUTPUT_PATH, len(df), len(df.columns))

    for season, grp in df.groupby("season"):
        log.info(
            "  %s: %d rows, %d players, %d matches",
            season, len(grp), grp["player"].nunique(), grp["match_id"].nunique(),
        )


if __name__ == "__main__":
    main()
