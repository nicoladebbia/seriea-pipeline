"""Player availability impact features.

Estimates the effect of key player availability on match outcomes by:
  - Tracking top-5 contributors per team (by cumulative minutes)
  - Checking top scorer availability
  - Computing squad rotation index vs. most common starting XI
"""

from __future__ import annotations

import logging
from collections import Counter

import pandas as pd

from storage.paths import parsed_path

log = logging.getLogger(__name__)


def add_player_impact_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Add player impact features to the match-level table.

    Columns added:
      home_key_players_available   - count of top-5 minutes players who started
      away_key_players_available
      home_top_scorer_played       - 1 if team's season top scorer started
      away_top_scorer_played
      home_squad_rotation          - number of starters different from most common XI
      away_squad_rotation
    """
    lineup_path = parsed_path("lineups")
    player_stats_path = parsed_path("player_stats")

    if not lineup_path.exists() or not player_stats_path.exists():
        log.warning("Lineup or player_stats data missing; skipping player impact features")
        return matches

    lineups = pd.read_parquet(lineup_path)
    player_stats = pd.read_parquet(player_stats_path)

    df = matches.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.sort_values("match_date").reset_index(drop=True)

    # Pre-compute cumulative data structures
    # minutes_played tracks total minutes per player per team
    minutes_by_player: dict[str, dict[str, float]] = {}  # team -> {player: total_minutes}
    goals_by_player: dict[str, dict[str, int]] = {}  # team -> {player: total_goals}
    starter_history: dict[str, list[frozenset]] = {}  # team -> [set of starters per match]

    h_key_avail = []
    a_key_avail = []
    h_scorer = []
    a_scorer = []
    h_rotation = []
    a_rotation = []

    for _, row in df.iterrows():
        match_id = row["match_id"]
        home = row["home_team"]
        away = row["away_team"]

        for team, is_home, key_list, scorer_list, rotation_list in [
            (home, True, h_key_avail, h_scorer, h_rotation),
            (away, False, a_key_avail, a_scorer, a_rotation),
        ]:
            # Get current match starters
            match_starters = set(
                lineups[
                    (lineups["match_id"] == match_id)
                    & (lineups["team"] == team)
                    & (lineups["role"] == "starter")
                ]["player_name"]
            )

            # Key players: top 5 by cumulative minutes BEFORE this match
            team_mins = minutes_by_player.get(team, {})
            top_5 = sorted(team_mins.items(), key=lambda x: x[1], reverse=True)[:5]
            top_5_names = {p for p, _ in top_5}
            key_available = len(match_starters & top_5_names) if top_5_names else None
            key_list.append(key_available)

            # Top scorer availability
            team_goals = goals_by_player.get(team, {})
            if team_goals:
                top_scorer = max(team_goals, key=team_goals.get)
                scorer_list.append(1 if top_scorer in match_starters else 0)
            else:
                scorer_list.append(None)

            # Squad rotation: compare to most common XI
            history = starter_history.get(team, [])
            if history:
                # Count how many times each player has started
                all_starters_counter: Counter = Counter()
                for prev_xi in history:
                    all_starters_counter.update(prev_xi)
                most_common_xi = {p for p, _ in all_starters_counter.most_common(11)}
                rotation = len(match_starters - most_common_xi)
                rotation_list.append(rotation)
            else:
                rotation_list.append(None)

            # --- Now update cumulative trackers with THIS match's data ---
            # Update minutes
            match_players = player_stats[
                (player_stats["match_id"] == match_id)
                & (player_stats["team"] == team)
            ]
            if team not in minutes_by_player:
                minutes_by_player[team] = {}
            if team not in goals_by_player:
                goals_by_player[team] = {}

            for _, p in match_players.iterrows():
                pname = p.get("player", "")
                mins = pd.to_numeric(p.get("minutes", 0), errors="coerce") or 0
                goals = pd.to_numeric(p.get("goals", 0), errors="coerce") or 0
                minutes_by_player[team][pname] = minutes_by_player[team].get(pname, 0) + mins
                goals_by_player[team][pname] = goals_by_player[team].get(pname, 0) + int(goals)

            # Update starter history
            if team not in starter_history:
                starter_history[team] = []
            if match_starters:
                starter_history[team].append(frozenset(match_starters))

    df["home_key_players_available"] = h_key_avail
    df["away_key_players_available"] = a_key_avail
    df["home_top_scorer_played"] = h_scorer
    df["away_top_scorer_played"] = a_scorer
    df["home_squad_rotation"] = h_rotation
    df["away_squad_rotation"] = a_rotation

    return df
