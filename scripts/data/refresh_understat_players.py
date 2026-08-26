#!/usr/bin/env python3
"""Refresh season-aggregate Understat player xG into understat_players.parquet.

Understat renders its league tables client-side: the page ships the numbers in a
`playersData` JS variable, so a plain HTTP GET returns a 200 with a shell and no
data in it (measured 2026-07-16: playersData occurrences = 0). That is why this
uses a real browser rather than requests — it is not an IP block, it is JS.

Usage:
    python3 -m scripts.data.refresh_understat_players --season 2025-2026
    python3 -m scripts.data.refresh_understat_players --leagues serie_a --all-seasons
    python3 -m scripts.data.refresh_understat_players --season 2025-2026 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

import pandas as pd

from config.settings import DATA_DIR, get_current_season

log = logging.getLogger(__name__)

OUTPUT_PATH = DATA_DIR / "parsed" / "understat_players.parquet"
PAGE_TIMEOUT = 90

# league key -> (Understat URL slug, the label stored in the parquet)
LEAGUES: dict[str, tuple[str, str]] = {
    "serie_a": ("Serie_A", "ITA-Serie A"),
    "premier_league": ("EPL", "ENG-Premier League"),
    # The three remaining big-5 leagues exist for one consumer: pricing Serie A newcomers
    # at the fantacalcio auction from their real record abroad instead of a market-value
    # fallback fit. Nothing in the betting pipeline reads them.
    "la_liga": ("La_liga", "ESP-La Liga"),
    "bundesliga": ("Bundesliga", "GER-Bundesliga"),
    "ligue_1": ("Ligue_1", "FRA-Ligue 1"),
}

# playersData field -> parquet column. Understat's own names are camelCase and
# internally inconsistent (npg beside npxG); the parquet is snake_case.
_FIELD_MAP: dict[str, str] = {
    "player_name": "player",
    "team_title": "team",
    "id": "player_id",
    "position": "position",
    "games": "matches",
    "time": "minutes",
    "goals": "goals",
    "xG": "xg",
    "npg": "np_goals",
    "npxG": "np_xg",
    "assists": "assists",
    "xA": "xa",
    "shots": "shots",
    "key_passes": "key_passes",
    "yellow_cards": "yellow_cards",
    "red_cards": "red_cards",
    "xGChain": "xg_chain",
    "xGBuildup": "xg_buildup",
}

COLUMNS = [
    "league", "season", "team", "player", "league_id", "season_id", "team_id",
    "player_id", "position", "matches", "minutes", "goals", "xg", "np_goals",
    "np_xg", "assists", "xa", "shots", "key_passes", "yellow_cards",
    "red_cards", "xg_chain", "xg_buildup",
]

_INT_COLS = [
    "season_id", "team_id", "player_id", "matches", "minutes", "goals",
    "np_goals", "assists", "shots", "key_passes", "yellow_cards", "red_cards",
]
_FLOAT_COLS = ["xg", "np_xg", "xa", "xg_chain", "xg_buildup"]


def _season_start_year(season: str) -> int:
    """'2025-2026' -> 2025. Understat keys league pages by the start year."""
    return int(season.split("-")[0])


def fetch_players(league_key: str, season: str) -> list[dict[str, Any]]:
    """Return raw playersData rows for one league-season via a headless browser.

    Raises RuntimeError if the variable never renders — that means either a
    schema change or a block, and both must be loud rather than write 0 rows.
    """
    slug, _ = LEAGUES[league_key]
    url = f"https://understat.com/league/{slug}/{_season_start_year(season)}"

    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    opts = Options()
    for arg in (
        "--headless=new", "--no-sandbox", "--disable-gpu",
        "--disable-dev-shm-usage", "--window-size=1200,900",
    ):
        opts.add_argument(arg)

    driver = webdriver.Chrome(options=opts)
    try:
        driver.set_page_load_timeout(PAGE_TIMEOUT)
        driver.get(url)
        # Ask the page for the already-parsed variable instead of regexing the
        # hex-escaped JSON blob back out of the HTML.
        rows = driver.execute_script(
            "return typeof playersData !== 'undefined' ? playersData : null;"
        )
    finally:
        driver.quit()

    if not rows:
        raise RuntimeError(f"playersData did not render at {url} (schema change or block)")

    log.info("understat %s %s: %d players", league_key, season, len(rows))
    return rows


def build_frame(rows: list[dict[str, Any]], league_key: str, season: str) -> pd.DataFrame:
    """Map raw playersData rows onto the parquet's 23-column schema."""
    _, label = LEAGUES[league_key]

    records = []
    for row in rows:
        rec: dict[str, Any] = {dst: row.get(src) for src, dst in _FIELD_MAP.items()}
        rec["league"] = label
        rec["season"] = season
        rec["season_id"] = _season_start_year(season)
        # playersData carries no team or league id; both columns are placeholders
        # in the stored schema (team_id is always 0, league_id always null).
        rec["team_id"] = 0
        rec["league_id"] = None
        records.append(rec)

    df = pd.DataFrame(records)
    for col in _INT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in _FLOAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Float64")
    df["league_id"] = df["league_id"].astype("string")
    return df[COLUMNS]


def _existing_seasons(league_label: str) -> list[str]:
    """Seasons already stored for a league — what --all-seasons refreshes."""
    if not OUTPUT_PATH.exists():
        return []
    df = pd.read_parquet(OUTPUT_PATH, columns=["league", "season"])
    return sorted(df.loc[df["league"] == league_label, "season"].unique().tolist())


def refresh(league_keys: list[str], seasons: list[str] | None,
            all_seasons: bool = False, dry_run: bool = False) -> pd.DataFrame:
    """Fetch the given league-seasons and merge them into the parquet.

    Each (league, season) already present is replaced wholesale — these are
    season aggregates, so the fresh pull supersedes the stored one.
    """
    frames: list[pd.DataFrame] = []
    for league_key in league_keys:
        _, label = LEAGUES[league_key]
        targets = _existing_seasons(label) if all_seasons else list(seasons or [])
        if not targets:
            log.warning("no seasons to refresh for %s", league_key)
            continue
        for season in targets:
            frames.append(build_frame(fetch_players(league_key, season), league_key, season))

    if not frames:
        raise SystemExit("nothing to refresh — pass --season or --all-seasons")

    fresh = pd.concat(frames, ignore_index=True)

    if dry_run:
        for (lg, sn), grp in fresh.groupby(["league", "season"]):
            log.info("DRY-RUN %s %s: %d rows (not written)", lg, sn, len(grp))
        return fresh

    if OUTPUT_PATH.exists():
        existing = pd.read_parquet(OUTPUT_PATH)
        refreshed = set(zip(fresh["league"], fresh["season"], strict=False))
        keep = [
            (lg, sn) not in refreshed
            for lg, sn in zip(existing["league"], existing["season"], strict=False)
        ]
        out = pd.concat([existing[keep], fresh], ignore_index=True)
    else:
        out = fresh

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT_PATH, index=False)
    log.info("wrote %s: %d rows (%d refreshed)", OUTPUT_PATH.name, len(out), len(fresh))
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", help="Season to refresh, e.g. 2025-2026, or 'current' to resolve the calendar season at run time")
    ap.add_argument("--leagues", default="serie_a,premier_league",
                    help="Comma-separated, any of: " + ",".join(sorted(LEAGUES)))
    ap.add_argument("--all-seasons", action="store_true",
                    help="Refresh every season already stored for each league")
    ap.add_argument("--dry-run", action="store_true", help="Fetch and report; write nothing")
    args = ap.parse_args()

    league_keys = [x.strip() for x in args.leagues.split(",") if x.strip()]
    unknown = [x for x in league_keys if x not in LEAGUES]
    if unknown:
        ap.error(f"unknown league(s): {unknown}. Known: {sorted(LEAGUES)}")
    if not args.season and not args.all_seasons:
        ap.error("pass --season YYYY-YYYY, --season current, or --all-seasons")

    # Resolved at run time so a scheduled job cannot pin itself to a finished
    # season -- this plist was stuck on 2025-2026 into August 2026.
    if args.season == "current":
        args.season = get_current_season()
        log.info("--season current resolved to %s", args.season)

    refresh(league_keys, [args.season] if args.season else None,
            all_seasons=args.all_seasons, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
