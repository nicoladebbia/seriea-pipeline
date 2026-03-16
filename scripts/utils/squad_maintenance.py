#!/usr/bin/env python3
"""SQUAD MAINTENANCE - Ongoing data quality checks for squad data.

Provides utilities for:
1. Checking squad data freshness
2. Validating squad data against Understat player xG data
3. Identifying stale/missing entries
4. Quick injury status updates

Run: python3 -m scripts.squad_maintenance [--check|--sync-understat|--summary]
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR
from config.team_names import SERIE_A_2025_26, normalize_team

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SQUADS_PATH = DATA_DIR / "squads" / "current_squads.json"
PROFILES_PATH = DATA_DIR / "features" / "player_xg_profiles.json"
UNDERSTAT_PATH = DATA_DIR / "external" / "understat" / "players_xg_2025_2026.parquet"

# Canonical team name mapping (Understat -> pipeline)
UNDERSTAT_TEAM_MAP = {
    "AC Milan": "Milan",
    "Inter": "Inter",
    "Parma Calcio 1913": "Parma",
}

# Use the canonical list from config/team_names.py (single source of truth)
SERIE_A_2025_26_TEAMS = set(SERIE_A_2025_26)


def check_freshness():
    """Check how old the squad data is and report staleness."""
    if not SQUADS_PATH.exists():
        print("  ❌ No squad data file found!")
        return False

    with open(SQUADS_PATH) as f:
        data = json.load(f)

    fetched_at = data.get("fetched_at", "")
    source = data.get("source", "unknown")
    teams = data.get("teams", {})

    if fetched_at:
        try:
            dt = datetime.fromisoformat(fetched_at)
            age = datetime.now() - dt
            age_str = f"{age.days} days ago" if age.days > 0 else "today"
        except ValueError:
            age_str = "unknown"
    else:
        age_str = "unknown"

    print(f"  Source: {source}")
    print(f"  Fetched: {fetched_at} ({age_str})")
    print(f"  Teams: {len(teams)}")

    missing = SERIE_A_2025_26_TEAMS - set(teams.keys())
    extra = set(teams.keys()) - SERIE_A_2025_26_TEAMS
    if missing:
        print(f"  ❌ Missing teams: {', '.join(sorted(missing))}")
    if extra:
        print(f"  ⚠️ Extra teams (relegated?): {', '.join(sorted(extra))}")
    if not missing and not extra:
        print("  ✅ All 20 Serie A teams present")

    return len(missing) == 0


def sync_with_understat():
    """Cross-reference squad data with Understat player xG data.

    Identifies players in Understat but missing from squads (and vice versa).
    """
    try:
        import pandas as pd
    except ImportError:
        print("  ❌ pandas required for Understat sync")
        return

    if not UNDERSTAT_PATH.exists():
        print("  ❌ Understat data not found")
        return

    if not SQUADS_PATH.exists():
        print("  ❌ Squad data not found")
        return

    df = pd.read_parquet(UNDERSTAT_PATH)
    with open(SQUADS_PATH) as f:
        squads = json.load(f).get("teams", {})

    # Build Understat player-team mapping (handle multi-team entries)
    understat_players = {}
    for _, row in df.iterrows():
        team_raw = row.get("team_title", "")
        player = row.get("player_name", "")
        if not team_raw or not player:
            continue

        # Handle comma-separated teams (transferred players)
        teams = [t.strip() for t in team_raw.split(",")]
        # Use last team (most recent)
        team = teams[-1]
        team = UNDERSTAT_TEAM_MAP.get(team, team)

        if team in SERIE_A_2025_26_TEAMS:
            if team not in understat_players:
                understat_players[team] = set()
            understat_players[team].add(player)

    print(f"\n  Understat teams: {len(understat_players)}")
    print(f"  Squad teams: {len(squads)}")

    total_in_understat_only = 0
    total_in_squad_only = 0

    for team in sorted(SERIE_A_2025_26_TEAMS):
        us_players = understat_players.get(team, set())
        sq_data = squads.get(team, {})
        sq_players_list = sq_data.get("players", []) if isinstance(sq_data, dict) else []
        sq_players = {p.get("name", "") for p in sq_players_list}

        # Simple name matching (Understat uses full names, squads use abbreviated)
        # Can't do exact match, just report counts
        if team in understat_players and team in squads:
            print(f"  {team}: Understat={len(us_players)}, Squad={len(sq_players)}")
        elif team not in understat_players:
            print(f"  {team}: ⚠️ Not in Understat data")
        elif team not in squads:
            print(f"  {team}: ❌ Not in squad data")


def print_injury_summary():
    """Print current injury status from squad data."""
    if not SQUADS_PATH.exists():
        print("  ❌ No squad data found")
        return

    with open(SQUADS_PATH) as f:
        squads = json.load(f).get("teams", {})

    total_injured = 0
    for team in sorted(squads.keys()):
        players = squads[team].get("players", [])
        injured = [p for p in players if p.get("status") == "injured"]
        suspended = [p for p in players if p.get("status") == "suspended"]

        if injured or suspended:
            print(f"\n  {team}:")
            for p in injured:
                print(f"    🚑 {p['name']} ({p['position']})")
                total_injured += 1
            for p in suspended:
                print(f"    🟡 {p['name']} ({p['position']}) - suspended")

    print(f"\n  Total injured: {total_injured}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Squad data maintenance")
    parser.add_argument("--check", action="store_true", help="Check data freshness")
    parser.add_argument("--sync-understat", action="store_true", help="Sync with Understat data")
    parser.add_argument("--injuries", action="store_true", help="Show injury summary")
    parser.add_argument("--summary", action="store_true", help="Full summary")
    args = parser.parse_args()

    if args.summary or (not args.check and not args.sync_understat and not args.injuries):
        print("\n=== SQUAD DATA FRESHNESS ===")
        check_freshness()
        print("\n=== INJURY SUMMARY ===")
        print_injury_summary()
        print("\n=== UNDERSTAT SYNC ===")
        sync_with_understat()
    else:
        if args.check:
            check_freshness()
        if args.sync_understat:
            sync_with_understat()
        if args.injuries:
            print_injury_summary()


if __name__ == "__main__":
    main()
