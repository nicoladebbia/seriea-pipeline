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
        if team_list:  # Include ALL players, even single-team ones
            result[player] = team_list

    return result


ALL_SERIE_A_SEASONS = [
    "2014-2015", "2015-2016", "2016-2017", "2017-2018", "2018-2019",
    "2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024",
    "2024-2025", "2025-2026",
]


def detect_career_gaps(career: List[Dict]) -> List[Dict]:
    """Detect seasons where the player was NOT in Serie A (likely abroad).

    Returns list of gap entries like:
      {"type": "gap", "seasons": ["2022-2023", "2023-2024"], "label": "Away from Serie A (2 seasons)"}
    """
    if not career:
        return []

    # Collect all seasons in Serie A
    all_seasons = set()
    for t in career:
        all_seasons.update(t.get("seasons", []))

    if not all_seasons:
        return []

    sorted_present = sorted(all_seasons)
    first = sorted_present[0]
    last = sorted_present[-1]

    first_idx = ALL_SERIE_A_SEASONS.index(first) if first in ALL_SERIE_A_SEASONS else -1
    last_idx = ALL_SERIE_A_SEASONS.index(last) if last in ALL_SERIE_A_SEASONS else -1

    if first_idx < 0 or last_idx < 0:
        return []

    expected = set(ALL_SERIE_A_SEASONS[first_idx:last_idx + 1])
    missing = sorted(expected - all_seasons)

    if not missing:
        return []

    # Group consecutive missing seasons into gap blocks
    gaps = []
    current_gap = [missing[0]]
    for i in range(1, len(missing)):
        prev_idx = ALL_SERIE_A_SEASONS.index(missing[i - 1])
        curr_idx = ALL_SERIE_A_SEASONS.index(missing[i])
        if curr_idx == prev_idx + 1:
            current_gap.append(missing[i])
        else:
            gaps.append(current_gap)
            current_gap = [missing[i]]
    gaps.append(current_gap)

    return [{
        "type": "gap",
        "seasons": gap,
        "n_seasons": len(gap),
        "first_season": gap[0],
        "last_season": gap[-1],
        "label": f"Abroad ({len(gap)} {'season' if len(gap) == 1 else 'seasons'})"
    } for gap in gaps]


def get_player_profile(player_name: str, history: Dict = None) -> Dict | None:
    """Get full profile for a player: career path + nationality + market value."""
    if history is None:
        history = build_player_history()

    # Get career path
    career = history.get(player_name)

    # Get nationality + market value from transfermarkt
    nationality = None
    market_value = None
    transfer_fee = None

    for season in ["2025_2026", "2024_2025", "2023_2024"]:
        mv_path = DATA_DIR / "external" / "transfermarkt" / f"market_values_{season}.parquet"
        if mv_path.exists():
            try:
                mv = pd.read_parquet(mv_path)
                match = mv[mv["player_name"].str.lower() == player_name.lower()]
                if match.empty:
                    # Fuzzy: last name match
                    last_name = player_name.split()[-1].lower()
                    match = mv[mv["player_name"].str.lower().str.contains(last_name, na=False)]
                if not match.empty:
                    row = match.iloc[0]
                    nationality = row.get("nationality")
                    market_value = row.get("market_value_eur")
                    break
            except Exception:
                pass

    # Get transfer fee (most recent arrival)
    for season in ["2025_2026", "2024_2025", "2023_2024"]:
        tf_path = DATA_DIR / "external" / "transfermarkt" / f"transfers_{season}.parquet"
        if tf_path.exists():
            try:
                tf = pd.read_parquet(tf_path)
                arrivals = tf[(tf["transfer_type"] == "in") &
                             (tf["player_name"].str.lower() == player_name.lower())]
                if arrivals.empty:
                    last_name = player_name.split()[-1].lower()
                    arrivals = tf[(tf["transfer_type"] == "in") &
                                 (tf["player_name"].str.lower().str.contains(last_name, na=False))]
                if not arrivals.empty:
                    row = arrivals.iloc[0]
                    transfer_fee = row.get("fee_eur")
                    if pd.isna(transfer_fee):
                        transfer_fee = None
                    break
            except Exception:
                pass

    if not career and not nationality:
        return None

    # Detect gaps (seasons abroad)
    gaps = detect_career_gaps(career) if career else []

    return {
        "name": player_name,
        "nationality": nationality,
        "market_value_eur": int(market_value) if market_value and not pd.isna(market_value) else None,
        "transfer_fee_eur": int(transfer_fee) if transfer_fee else None,
        "career": career or [],
        "career_gaps": gaps,
    }


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
