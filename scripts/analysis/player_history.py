"""Player team history — tracks which teams each player has played for.

Builds a lookup from player name → list of former teams with seasons.
Used for:
  1. UX: "Ex-Roma player Dzeko now at Inter — knows Roma's weaknesses"
  2. Match context: "3 players facing their former team"
  3. Future: Could feed into model as a feature (ex-team motivation factor)

Data sources:
  - Understat players (2014-2025): season-level team assignments
  - Sofascore player match stats (2022+): match-level team assignments
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def build_player_history() -> Dict[str, List[Dict]]:
    """Build complete player team history from all data sources.

    Returns: {
        "player_name": [
            {"team": "Roma", "seasons": ["2018-2019", "2019-2020"], "total_matches": 65},
            {"team": "Inter", "seasons": ["2021-2022", "2022-2023"], "total_matches": 58},
        ]
    }
    """
    # Source 1: Understat (2014-2025, broader coverage)
    history = defaultdict(lambda: defaultdict(lambda: {"seasons": [], "matches": 0}))

    us_path = DATA_DIR / "parsed" / "understat_players.parquet"
    if us_path.exists():
        us = pd.read_parquet(us_path)
        for _, row in us.iterrows():
            player = row.get("player", "")
            team = row.get("team", "")
            season = row.get("season", "")
            matches = row.get("matches", 0)
            if pd.isna(matches):
                matches = 0
            if player and team:
                entry = history[player][team]
                if season and season not in entry["seasons"]:
                    entry["seasons"].append(season)
                entry["matches"] += int(matches)

    # Source 2: Sofascore (2022+, more recent)
    ss_path = DATA_DIR / "external" / "sofascore" / "player_match_stats.parquet"
    if ss_path.exists():
        ss = pd.read_parquet(ss_path)
        for player, grp in ss.groupby("player_name"):
            for team, team_grp in grp.groupby("team"):
                seasons = sorted(team_grp["season"].unique())
                n_matches = len(team_grp)
                entry = history[player][team]
                for s in seasons:
                    if s not in entry["seasons"]:
                        entry["seasons"].append(s)
                entry["matches"] = max(entry["matches"], n_matches)

    # Convert to sorted list format
    result = {}
    for player, teams in history.items():
        team_list = []
        for team, data in teams.items():
            data["seasons"].sort()
            team_list.append({
                "team": team,
                "seasons": data["seasons"],
                "total_matches": data["matches"],
                "first_season": data["seasons"][0] if data["seasons"] else "",
                "last_season": data["seasons"][-1] if data["seasons"] else "",
            })
        # Sort by most recent first
        team_list.sort(key=lambda x: x["last_season"], reverse=True)
        if len(team_list) > 1:  # Only include players with 2+ teams
            result[player] = team_list

    return result


def find_ex_players(match_key: str, home_lineup: list, away_lineup: list,
                    history: Dict = None) -> Dict:
    """Find players in a match who are facing their former team.

    Returns: {
        "home_vs_former": [
            {"player": "Dzeko", "current_team": "Inter", "former_team": "Roma",
             "seasons_at_former": ["2015-2016", ...], "matches_at_former": 65}
        ],
        "away_vs_former": [...]
    }
    """
    if history is None:
        history = build_player_history()

    parts = match_key.split(" vs ")
    if len(parts) != 2:
        return {"home_vs_former": [], "away_vs_former": []}

    home_team = parts[0].strip()
    away_team = parts[1].strip()

    # Team name aliases for matching
    ALIASES = {
        "Milan": ["AC Milan", "Milan"],
        "AC Milan": ["AC Milan", "Milan"],
        "Inter": ["Inter", "Internazionale"],
        "Roma": ["Roma", "AS Roma"],
        "AS Roma": ["Roma", "AS Roma"],
        "Verona": ["Verona", "Hellas Verona"],
        "Hellas Verona": ["Verona", "Hellas Verona"],
    }

    def _team_match(team_a: str, team_b: str) -> bool:
        aliases_a = ALIASES.get(team_a, [team_a])
        aliases_b = ALIASES.get(team_b, [team_b])
        return any(a == b for a in aliases_a for b in aliases_b)

    home_ex = []
    away_ex = []

    # Check home players who used to play for away team
    for player_name in home_lineup:
        player_history = history.get(player_name, [])
        for entry in player_history:
            if _team_match(entry["team"], away_team):
                home_ex.append({
                    "player": player_name,
                    "current_team": home_team,
                    "former_team": entry["team"],
                    "seasons_at_former": entry["seasons"],
                    "matches_at_former": entry["total_matches"],
                })
                break

    # Check away players who used to play for home team
    for player_name in away_lineup:
        player_history = history.get(player_name, [])
        for entry in player_history:
            if _team_match(entry["team"], home_team):
                away_ex.append({
                    "player": player_name,
                    "current_team": away_team,
                    "former_team": entry["team"],
                    "seasons_at_former": entry["seasons"],
                    "matches_at_former": entry["total_matches"],
                })
                break

    return {
        "home_vs_former": home_ex,
        "away_vs_former": away_ex,
        "total_ex_players": len(home_ex) + len(away_ex),
    }


def get_match_context(match_key: str) -> Dict:
    """Get full player history context for a match using confirmed lineups."""
    lineups_path = DATA_DIR / "upcoming" / "confirmed_lineups.json"
    if not lineups_path.exists():
        return {"error": "No confirmed lineups"}

    lineups = json.load(open(lineups_path))
    match_data = lineups.get("matches", {}).get(match_key, {})
    if not match_data:
        return {"error": f"No lineup for {match_key}"}

    home_lineup = match_data.get("home_lineup", [])
    away_lineup = match_data.get("away_lineup", [])

    history = build_player_history()
    ex_players = find_ex_players(match_key, home_lineup, away_lineup, history)

    return ex_players


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    history = build_player_history()
    print(f"Players with multi-team history: {len(history)}")

    # Show top movers
    for player, teams in sorted(history.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        team_str = " → ".join(f"{t['team']} ({t['last_season']})" for t in reversed(teams))
        print(f"  {player}: {team_str}")
