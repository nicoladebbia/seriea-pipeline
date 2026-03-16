"""Audit player data coverage across all seasons.

Reports per season:
  1. player_stats.parquet — row count, unique players, date range
  2. understat_players.parquet — row count, player count
  3. Raw HTML files — count per data/raw/html/{season}/
  4. Registry — downloaded vs parsed counts
  5. Gap analysis — which seasons need scraping

Usage:
    python3 -m scripts.player_data_audit
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import DATA_DIR, RAW_HTML_DIR, SEASONS
from storage.paths import parsed_path

log = logging.getLogger(__name__)

# Seasons that should have FBref match-level player stats
PLAYER_STATS_SEASONS = [s for s in SEASONS if int(s.split("-")[0]) >= 2017]


def audit_player_stats_parquet() -> dict:
    """Check player_stats.parquet coverage."""
    path = parsed_path("player_stats")
    if not path.exists():
        return {"exists": False, "total_rows": 0, "seasons": {}}

    df = pd.read_parquet(path)
    result = {
        "exists": True,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "unique_players": df["player"].nunique() if "player" in df.columns else 0,
        "seasons": {},
    }

    if "season" in df.columns:
        for season, grp in df.groupby("season"):
            result["seasons"][season] = {
                "rows": len(grp),
                "players": grp["player"].nunique() if "player" in grp.columns else 0,
                "teams": grp["team"].nunique() if "team" in grp.columns else 0,
            }
            if "match_id" in grp.columns:
                result["seasons"][season]["matches"] = grp["match_id"].nunique()

    return result


def audit_understat_players() -> dict:
    """Check understat_players.parquet coverage."""
    path = parsed_path("understat_players")
    if not path.exists():
        return {"exists": False, "total_rows": 0, "seasons": {}}

    df = pd.read_parquet(path)
    # Column name varies: "player_name" or "player"
    player_col = "player_name" if "player_name" in df.columns else "player"
    result = {
        "exists": True,
        "total_rows": len(df),
        "unique_players": df[player_col].nunique() if player_col in df.columns else 0,
        "seasons": {},
    }

    season_col = "season" if "season" in df.columns else None
    if season_col:
        for season, grp in df.groupby(season_col):
            result["seasons"][str(season)] = {
                "rows": len(grp),
                "players": grp[player_col].nunique() if player_col in grp.columns else 0,
            }

    return result


def audit_raw_html() -> dict:
    """Count raw HTML match report files per season.

    Excludes non-match files like fixtures.html and stats_*.html.
    """
    # Known non-match filenames to skip
    SKIP_PREFIXES = ("fixtures", "stats_")

    result = {}
    if not RAW_HTML_DIR.exists():
        return result

    for season_dir in sorted(RAW_HTML_DIR.iterdir()):
        if season_dir.is_dir():
            html_files = [
                f for f in season_dir.glob("*.html")
                if not any(f.name.startswith(p) for p in SKIP_PREFIXES)
            ]
            result[season_dir.name] = len(html_files)

    return result


def audit_registry() -> dict:
    """Check registry for downloaded/parsed counts per season."""
    registry_path = DATA_DIR / "registry.json"
    if not registry_path.exists():
        return {"exists": False, "total": 0, "seasons": {}}

    with open(registry_path) as f:
        data = json.load(f)

    result = {"exists": True, "total": len(data), "parsed": 0, "seasons": {}}
    for mid, entry in data.items():
        season = entry.get("season", "unknown")
        if season not in result["seasons"]:
            result["seasons"][season] = {"downloaded": 0, "parsed": 0}
        result["seasons"][season]["downloaded"] += 1
        if entry.get("parsed"):
            result["seasons"][season]["parsed"] += 1
            result["parsed"] += 1

    return result


def gap_analysis(ps_audit: dict, html_audit: dict) -> list[dict]:
    """Identify seasons that need scraping."""
    gaps = []
    for season in PLAYER_STATS_SEASONS:
        ps_rows = ps_audit.get("seasons", {}).get(season, {}).get("rows", 0)
        # HTML dirs may use hyphens or underscores (e.g. 2017-2018 or 2017_2018)
        html_count = html_audit.get(season, 0) + html_audit.get(season.replace("-", "_"), 0)
        expected_matches = 380  # 20 teams, 38 matchdays

        status = "complete" if ps_rows >= 10000 else "partial" if ps_rows > 0 else "missing"
        gaps.append({
            "season": season,
            "player_stats_rows": ps_rows,
            "html_files": html_count,
            "expected_matches": expected_matches,
            "status": status,
            "needs_scraping": html_count < expected_matches * 0.9,
            "needs_parsing": html_count > 0 and ps_rows < html_count * 20,
        })

    return gaps


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    print("\n" + "=" * 70)
    print("  PLAYER DATA AUDIT")
    print("=" * 70)

    # 1. player_stats.parquet
    ps = audit_player_stats_parquet()
    print(f"\n{'─' * 50}")
    print(f"  player_stats.parquet: {'EXISTS' if ps['exists'] else 'NOT FOUND'}")
    print(f"  Total rows: {ps['total_rows']:,}  |  Unique players: {ps.get('unique_players', 0):,}")
    if ps["seasons"]:
        print(f"  Seasons with data: {len(ps['seasons'])}")
        for s, info in sorted(ps["seasons"].items()):
            print(f"    {s}: {info['rows']:>6,} rows, {info.get('players', '?'):>4} players, "
                  f"{info.get('teams', '?'):>2} teams, {info.get('matches', '?')} matches")

    # 2. understat_players.parquet
    us = audit_understat_players()
    print(f"\n{'─' * 50}")
    print(f"  understat_players.parquet: {'EXISTS' if us['exists'] else 'NOT FOUND'}")
    print(f"  Total rows: {us['total_rows']:,}  |  Unique players: {us.get('unique_players', 0):,}")
    if us["seasons"]:
        print(f"  Seasons: {len(us['seasons'])}")

    # 3. Raw HTML
    html = audit_raw_html()
    print(f"\n{'─' * 50}")
    print(f"  Raw HTML match reports:")
    if html:
        for season, count in sorted(html.items()):
            print(f"    {season}: {count} files")
    else:
        print("    No HTML directories found")

    # 4. Registry
    reg = audit_registry()
    print(f"\n{'─' * 50}")
    print(f"  Registry: {'EXISTS' if reg.get('exists') else 'NOT FOUND'}")
    print(f"  Total: {reg.get('total', 0)}  |  Parsed: {reg.get('parsed', 0)}")

    # 5. Gap analysis
    gaps = gap_analysis(ps, html)
    print(f"\n{'─' * 50}")
    print(f"  GAP ANALYSIS (seasons {PLAYER_STATS_SEASONS[0]} to {PLAYER_STATS_SEASONS[-1]}):")
    for g in gaps:
        icon = {"complete": "OK", "partial": "!!", "missing": "XX"}[g["status"]]
        scrape = " [NEEDS SCRAPING]" if g["needs_scraping"] else ""
        parse = " [NEEDS PARSING]" if g["needs_parsing"] else ""
        print(f"    [{icon}] {g['season']}: {g['player_stats_rows']:>6,} rows, "
              f"{g['html_files']:>3} HTMLs{scrape}{parse}")

    missing = sum(1 for g in gaps if g["status"] == "missing")
    partial = sum(1 for g in gaps if g["status"] == "partial")
    print(f"\n  Summary: {missing} missing, {partial} partial, "
          f"{len(gaps) - missing - partial} complete")

    # Save JSON report
    report = {
        "player_stats": ps,
        "understat_players": us,
        "raw_html": html,
        "registry": reg,
        "gap_analysis": gaps,
    }
    out_path = DATA_DIR / "player_data_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Full report saved to: {out_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
