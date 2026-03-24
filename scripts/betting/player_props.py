#!/usr/bin/env python3
"""Player goalscorer prop predictions.

Uses player xG profiles + Poisson to compute per-player goal probabilities
for each upcoming match. Outputs top scorers per team with anytime and 2+
goal probabilities.
"""

import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR


def _load_current_roster() -> set:
    """Load current squad roster as a set of (last_name_lower, team) pairs.

    Used to filter out players who are no longer in Serie A.
    """
    squads_path = DATA_DIR / "squads" / "current_squads.json"
    if not squads_path.exists():
        return set()
    try:
        with open(squads_path) as f:
            squads = json.load(f)
        roster = set()
        for team, team_data in squads.get("teams", {}).items():
            for p in team_data.get("players", []):
                name = p.get("name", "")
                if name:
                    # Extract last name — handles "S. Kılıçsoy", "Zé Pedro", etc.
                    parts = name.strip().split()
                    last = parts[-1].lower() if parts else ""
                    if last:
                        roster.add((last, team))
        return roster
    except Exception:
        return set()


def _load_profiles() -> dict:
    """Load player xG profiles, deduplicate phantoms, and filter stale players.

    1. Dedup: for players with entries under multiple teams (phantom), keep
       the team with the most minutes.
    2. Roster filter: remove players whose last name + team don't appear in
       current_squads.json (they've left Serie A).
    """
    path = DATA_DIR / "features" / "player_xg_profiles.json"
    if not path.exists():
        return {}
    with open(path) as f:
        raw = json.load(f)

    # Deduplicate: for each player name, keep the team with more minutes
    best = {}  # player_name -> (key, profile)
    for key, profile in raw.items():
        name = profile.get("player_name", key.split("|")[0])
        mins = profile.get("total_minutes", 0)
        if name not in best or mins > best[name][1].get("total_minutes", 0):
            best[name] = (key, profile)

    deduped = {k: v for k, v in (best[n] for n in best)}

    # Filter against current roster
    roster = _load_current_roster()
    if not roster:
        return deduped  # No roster data, can't filter

    filtered = {}
    for key, profile in deduped.items():
        name = profile.get("player_name", key.split("|")[0])
        team = profile.get("team", "")
        parts = name.strip().split()
        last = parts[-1].lower() if parts else ""
        if (last, team) in roster:
            filtered[key] = profile
    return filtered if filtered else deduped  # Fallback to deduped if filter too aggressive


def _get_team_players(profiles: dict, team: str) -> list:
    """Get players for a team, sorted by xG/90 desc."""
    players = []
    for key, profile in profiles.items():
        if profile.get("team", "") == team:
            xg_per_90 = profile.get("xg_per_90", 0)
            minutes = profile.get("total_minutes", 0)
            # Only include players with meaningful playing time
            if minutes >= 200:
                players.append({
                    "name": profile.get("player_name", key.split("|")[0]),
                    "team": team,
                    "xg_per_90": xg_per_90,
                    "xa_per_90": profile.get("xa_per_90", 0),
                    "total_xg": profile.get("total_xg", 0),
                    "total_minutes": minutes,
                    "matches": profile.get("matches_played", 0),
                    "recent_form_xg": profile.get("recent_form_xg", 0),
                })
    players.sort(key=lambda p: p["xg_per_90"], reverse=True)
    return players[:8]  # Top 8 per team


def _load_sofascore_rolling() -> dict:
    """Load Sofascore per-player rolling stats for prop predictions.

    Returns dict: {(player_name_lower, team_lower): {avg_shots, avg_sot, avg_passes,
    avg_tackles, avg_fouls, avg_cards_yellow, avg_assists, avg_rating}}
    """
    try:
        import pandas as pd
        path = Path(DATA_DIR).parent / "data" / "external" / "sofascore" / "player_match_stats.parquet"
        if not path.exists():
            path = DATA_DIR / ".." / "data" / "external" / "sofascore" / "player_match_stats.parquet"
        if not path.exists():
            return {}
        df = pd.read_parquet(path)
        # Get latest season data
        if "season" in df.columns:
            latest = df["season"].max()
            df = df[df["season"] == latest]

        num_cols = ["total_shots", "shots_on_target", "accurate_passes", "total_passes",
                    "tackles", "fouls", "was_fouled", "key_passes", "xa", "rating",
                    "assists", "carries", "progressive_carries", "minutes"]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Rolling 5-match averages per player
        result = {}
        for (player, team), grp in df.groupby(["player_name", "team"]):
            grp = grp.sort_values("date")
            last_5 = grp.tail(5)
            if len(last_5) < 2:
                continue
            mins = last_5["minutes"].sum()
            if mins < 90:
                continue
            n = len(last_5)
            result[(str(player).lower(), str(team).lower())] = {
                "avg_shots": round(last_5["total_shots"].mean(), 2) if "total_shots" in last_5 else 0,
                "avg_sot": round(last_5["shots_on_target"].mean(), 2) if "shots_on_target" in last_5 else 0,
                "avg_passes": round(last_5["accurate_passes"].mean(), 1) if "accurate_passes" in last_5 else 0,
                "avg_tackles": round(last_5["tackles"].mean(), 2) if "tackles" in last_5 else 0,
                "avg_fouls": round(last_5["fouls"].mean(), 2) if "fouls" in last_5 else 0,
                "avg_key_passes": round(last_5["key_passes"].mean(), 2) if "key_passes" in last_5 else 0,
                "avg_assists": round(last_5["assists"].mean(), 2) if "assists" in last_5 else 0,
                "avg_rating": round(last_5["rating"].mean(), 2) if "rating" in last_5 else 0,
                "avg_minutes": round(last_5["minutes"].mean(), 0),
                "matches": n,
            }
        return result
    except Exception:
        return {}


def _poisson_over(threshold: float, expected: float) -> float:
    """P(X > threshold) using Poisson."""
    k = int(threshold)
    return 1 - sum(math.exp(-expected) * (expected ** i) / math.factorial(i)
                   for i in range(k + 1))


def compute_player_props(predictions: list, profiles: dict) -> dict:
    """Compute goalscorer + expanded prop probabilities for all matches."""
    result = {}

    # Try to load Sofascore rolling stats for shot/pass/tackle props
    ss_rolling = _load_sofascore_rolling()

    for pred in predictions:
        match = pred.get("match", "")
        home_team = pred.get("home_team", "")
        away_team = pred.get("away_team", "")

        if not home_team or not away_team:
            continue

        home_players = _get_team_players(profiles, home_team)
        away_players = _get_team_players(profiles, away_team)

        match_props = []
        for player in home_players + away_players:
            # Expected goals in ~80 effective minutes
            player_xg = player["xg_per_90"] * (80 / 90)

            # Boost by recent form if available
            recent = player.get("recent_form_xg", 0)
            if recent > 0 and player["xg_per_90"] > 0:
                form_ratio = recent / player["xg_per_90"]
                # Blend: 70% career, 30% recent form
                player_xg *= (0.7 + 0.3 * form_ratio)

            # Poisson probabilities for goals
            p_zero = math.exp(-player_xg)
            p_anytime = 1 - p_zero
            p_2plus = 1 - p_zero - player_xg * p_zero

            # First goalscorer: time-weighted probability
            # P(first scorer) ~ P(anytime) * (player_xg / team_total_xg) * time_factor
            # Approximation: first scorer = anytime * 0.35 (avg ~35% of anytime odds)
            p_first = p_anytime * 0.35
            p_last = p_anytime * 0.30

            def safe_odds(p):
                return round(1 / p, 2) if p > 0.01 else 99

            prop_entry = {
                "name": player["name"],
                "team": player["team"],
                "xg_per_90": round(player["xg_per_90"], 3),
                "xa_per_90": round(player["xa_per_90"], 3),
                "total_xg": round(player["total_xg"], 2),
                "minutes": player["total_minutes"],
                "matches": player["matches"],
                # Goal props
                "anytime_goal_prob": round(max(0, p_anytime), 4),
                "anytime_fair_odds": safe_odds(p_anytime),
                "two_plus_prob": round(max(0, p_2plus), 4),
                "two_plus_fair_odds": safe_odds(p_2plus),
                "first_goal_prob": round(max(0, p_first), 4),
                "first_goal_fair_odds": safe_odds(p_first),
                "last_goal_prob": round(max(0, p_last), 4),
                "last_goal_fair_odds": safe_odds(p_last),
                # Assist prop (from xA)
                "assist_prob": round(1 - math.exp(-player["xa_per_90"] * 80 / 90), 4),
                "assist_fair_odds": safe_odds(1 - math.exp(-player["xa_per_90"] * 80 / 90)),
            }

            # Sofascore-based props (shots, SOT, passes, tackles, fouls, cards)
            pname_lower = player["name"].lower()
            team_lower = player["team"].lower()
            ss = ss_rolling.get((pname_lower, team_lower), {})
            if not ss:
                # Try fuzzy match on just last name
                for (pn, tn), stats in ss_rolling.items():
                    if team_lower in tn and pname_lower.split()[-1] in pn:
                        ss = stats
                        break

            if ss:
                # Shots props
                avg_shots = ss.get("avg_shots", 0)
                if avg_shots > 0:
                    prop_entry["shots_over_0_5"] = round(_poisson_over(0.5, avg_shots), 4)
                    prop_entry["shots_over_1_5"] = round(_poisson_over(1.5, avg_shots), 4)
                    prop_entry["shots_over_2_5"] = round(_poisson_over(2.5, avg_shots), 4)
                    prop_entry["shots_expected"] = avg_shots

                # SOT props
                avg_sot = ss.get("avg_sot", 0)
                if avg_sot > 0:
                    prop_entry["sot_over_0_5"] = round(_poisson_over(0.5, avg_sot), 4)
                    prop_entry["sot_over_1_5"] = round(_poisson_over(1.5, avg_sot), 4)
                    prop_entry["sot_expected"] = avg_sot

                # Tackles props
                avg_tackles = ss.get("avg_tackles", 0)
                if avg_tackles > 0:
                    prop_entry["tackles_over_0_5"] = round(_poisson_over(0.5, avg_tackles), 4)
                    prop_entry["tackles_over_1_5"] = round(_poisson_over(1.5, avg_tackles), 4)
                    prop_entry["tackles_expected"] = avg_tackles

                # Fouls committed
                avg_fouls = ss.get("avg_fouls", 0)
                if avg_fouls > 0:
                    prop_entry["fouls_over_0_5"] = round(_poisson_over(0.5, avg_fouls), 4)
                    prop_entry["fouls_expected"] = avg_fouls

                # Card probability (from foul rate + referee strictness)
                if avg_fouls > 0:
                    # ~20% of fouls result in a yellow (Serie A avg)
                    card_rate = avg_fouls * 0.20
                    p_card = 1 - math.exp(-card_rate)
                    prop_entry["to_be_carded_prob"] = round(p_card, 4)
                    prop_entry["to_be_carded_fair_odds"] = safe_odds(p_card)

                # Passes props
                avg_passes = ss.get("avg_passes", 0)
                if avg_passes > 10:
                    prop_entry["passes_expected"] = avg_passes

                # Key passes / assists
                avg_kp = ss.get("avg_key_passes", 0)
                if avg_kp > 0:
                    prop_entry["key_passes_over_0_5"] = round(_poisson_over(0.5, avg_kp), 4)
                    prop_entry["key_passes_expected"] = avg_kp

                prop_entry["sofascore_rating"] = ss.get("avg_rating", 0)

            match_props.append(prop_entry)

        # Sort by anytime goal probability
        match_props.sort(key=lambda p: p["anytime_goal_prob"], reverse=True)

        result[match] = {
            "match": match,
            "home_team": home_team,
            "away_team": away_team,
            "players": match_props,
            "props_available": [
                "anytime_goal", "two_plus_goals", "first_goal", "last_goal",
                "assist", "shots", "shots_on_target", "tackles", "fouls",
                "to_be_carded", "key_passes",
            ],
        }

    return result


def generate_player_props() -> dict:
    """Main entry point: load data and generate player props."""
    pred_path = DATA_DIR / "upcoming" / "predictions.json"
    if not pred_path.exists():
        log.warning("No predictions.json found")
        return {}

    with open(pred_path) as f:
        pred_data = json.load(f)

    profiles = _load_profiles()
    if not profiles:
        log.warning("No player_xg_profiles.json found")
        return {}

    predictions = pred_data.get("predictions", [])
    props = compute_player_props(predictions, profiles)

    output = {
        "generated_at": datetime.now().isoformat(),
        "match_count": len(props),
        "total_profiles": len(profiles),
        "matches": props,
    }

    out_path = DATA_DIR / "upcoming" / "player_props.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    total_players = sum(len(m["players"]) for m in props.values())
    log.info("Generated player props for %d matches (%d players) -> %s",
             len(props), total_players, out_path)
    return output


if __name__ == "__main__":
    generate_player_props()
