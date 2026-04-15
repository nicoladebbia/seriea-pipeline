#!/usr/bin/env python3
"""CURRENT FORM CALCULATOR - Real-Time Team Form Analysis

Calculates CURRENT form metrics for teams based on their most recent matches.
This is essential for real predictions - we need TODAY's form, not historical averages.

Form factors validated across 21 seasons:
- Hot Home Form (10+ pts/5 games): +8% home win
- Cold Away Form (≤4 pts/5 games): +6% home win
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR, atomic_write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_recent_matches(n_matchweeks: int = 10, league: str = None) -> pd.DataFrame:
    """Load most recent matches from features data."""
    features_path = DATA_DIR / "features" / "features.parquet"

    if not features_path.exists():
        log.warning(f"Features file not found: {features_path}")
        return pd.DataFrame()

    df = pd.read_parquet(features_path)

    # Filter by league if specified
    if league and "league" in df.columns:
        df = df[df["league"] == league]

    # Get current season and most recent matchweeks
    df = df.sort_values("match_date", ascending=False)

    # Filter to completed matches only
    df = df[df["home_score"].notna()]

    # Get the current season's matches first, then fill from previous if needed
    if "season" in df.columns:
        seasons = sorted(df["season"].unique(), reverse=True)
        current_season = seasons[0] if seasons else None
        if current_season:
            current = df[df["season"] == current_season]
            max_mw_current = current["matchweek"].max() if len(current) > 0 else 0
            if max_mw_current > n_matchweeks:
                recent = current[current["matchweek"] > max_mw_current - n_matchweeks]
            else:
                # Current season has fewer matchweeks than requested — use all of it
                recent = current
            log.info(f"Loaded {len(recent)} recent matches (season {current_season}, MW up to {max_mw_current}){' [' + league + ']' if league else ''}")
            return recent

    # Fallback: use global matchweek (original logic)
    max_mw = df["matchweek"].max()
    recent = df[df["matchweek"] > max_mw - n_matchweeks]
    log.info(f"Loaded {len(recent)} recent matches (matchweeks {max_mw - n_matchweeks + 1}-{max_mw}){' [' + league + ']' if league else ''}")
    return recent


def calculate_team_form(df: pd.DataFrame, team: str, last_n: int = 5) -> Dict:
    """Calculate form metrics for a team based on last N matches.

    Returns:
        Dict with points, goals, form_status, etc.
    """
    # Normalize team name to match features.parquet naming
    from config.team_names import normalize_team
    team_norm = normalize_team(team)

    # Get team's matches (home or away) — try both original and normalized name
    home_matches = df[(df["home_team"] == team) | (df["home_team"] == team_norm)].copy()
    away_matches = df[(df["away_team"] == team) | (df["away_team"] == team_norm)].copy()

    # Calculate results for each
    home_matches["team_goals"] = home_matches["home_score"]
    home_matches["opp_goals"] = home_matches["away_score"]
    home_matches["is_home"] = True

    away_matches["team_goals"] = away_matches["away_score"]
    away_matches["opp_goals"] = away_matches["home_score"]
    away_matches["is_home"] = False

    # Combine and sort by date
    all_matches = pd.concat([
        home_matches[["match_date", "team_goals", "opp_goals", "is_home", "matchweek"]],
        away_matches[["match_date", "team_goals", "opp_goals", "is_home", "matchweek"]]
    ]).sort_values("match_date", ascending=False)

    if len(all_matches) == 0:
        return {"error": f"No matches found for {team}"}

    # Take last N matches
    recent = all_matches.head(last_n)

    # Calculate points
    def get_points(row):
        if row["team_goals"] > row["opp_goals"]:
            return 3
        elif row["team_goals"] == row["opp_goals"]:
            return 1
        return 0

    recent = recent.copy()
    recent["points"] = recent.apply(get_points, axis=1)

    # Separate home and away form
    home_recent = all_matches[all_matches["is_home"]].head(last_n)
    away_recent = all_matches[~all_matches["is_home"]].head(last_n)

    home_recent = home_recent.copy()
    away_recent = away_recent.copy()
    home_recent["points"] = home_recent.apply(get_points, axis=1)
    away_recent["points"] = away_recent.apply(get_points, axis=1)

    form = {
        "team": team,
        "last_n": last_n,
        "total_points": int(recent["points"].sum()),
        "total_matches": len(recent),
        "goals_scored": int(recent["team_goals"].sum()),
        "goals_conceded": int(recent["opp_goals"].sum()),
        "wins": int((recent["points"] == 3).sum()),
        "draws": int((recent["points"] == 1).sum()),
        "losses": int((recent["points"] == 0).sum()),
        "ppg": round(recent["points"].mean(), 2),  # Points per game
        "gpg": round(recent["team_goals"].mean(), 2),  # Goals per game
        "home_points": int(home_recent["points"].sum()) if len(home_recent) > 0 else 0,
        "home_matches": len(home_recent),
        "away_points": int(away_recent["points"].sum()) if len(away_recent) > 0 else 0,
        "away_matches": len(away_recent),
    }

    # Classify form status
    # Hot form: 10+ points in 5 games (2+ ppg)
    # Cold form: ≤4 points in 5 games (≤0.8 ppg)
    if form["ppg"] >= 2.0:
        form["form_status"] = "hot"
    elif form["ppg"] <= 0.8:
        form["form_status"] = "cold"
    else:
        form["form_status"] = "normal"

    # Home-specific form
    if len(home_recent) >= 3:
        home_ppg = home_recent["points"].mean()
        if home_ppg >= 2.0:
            form["home_form_status"] = "hot"
        elif home_ppg <= 0.8:
            form["home_form_status"] = "cold"
        else:
            form["home_form_status"] = "normal"
    else:
        form["home_form_status"] = "unknown"

    # Away-specific form
    if len(away_recent) >= 3:
        away_ppg = away_recent["points"].mean()
        if away_ppg >= 2.0:
            form["away_form_status"] = "hot"
        elif away_ppg <= 0.8:
            form["away_form_status"] = "cold"
        else:
            form["away_form_status"] = "normal"
    else:
        form["away_form_status"] = "unknown"

    return form


def calculate_head_to_head(df: pd.DataFrame, home_team: str, away_team: str, last_n: int = 5) -> Dict:
    """Calculate head-to-head record between two teams."""
    # Find matches between these teams
    h2h = df[
        ((df["home_team"] == home_team) & (df["away_team"] == away_team)) |
        ((df["home_team"] == away_team) & (df["away_team"] == home_team))
    ].sort_values("match_date", ascending=False).head(last_n)

    if len(h2h) == 0:
        return {"error": "No head-to-head matches found"}

    # Calculate from home_team's perspective
    home_wins = 0
    away_wins = 0
    draws = 0
    home_goals = 0
    away_goals = 0

    for _, match in h2h.iterrows():
        if match["home_team"] == home_team:
            home_goals += match["home_score"]
            away_goals += match["away_score"]
            if match["home_score"] > match["away_score"]:
                home_wins += 1
            elif match["home_score"] < match["away_score"]:
                away_wins += 1
            else:
                draws += 1
        else:
            home_goals += match["away_score"]
            away_goals += match["home_score"]
            if match["away_score"] > match["home_score"]:
                home_wins += 1
            elif match["away_score"] < match["home_score"]:
                away_wins += 1
            else:
                draws += 1

    return {
        "matches": len(h2h),
        "home_team_wins": home_wins,
        "away_team_wins": away_wins,
        "draws": draws,
        "home_team_goals": int(home_goals),
        "away_team_goals": int(away_goals),
    }


def get_elo_ratings(df: pd.DataFrame) -> Dict[str, float]:
    """Get current Elo ratings for all teams."""
    # Get most recent Elo values
    recent = df.sort_values("match_date", ascending=False)

    elo = {}
    seen = set()

    for _, row in recent.iterrows():
        if row["home_team"] not in seen and pd.notna(row.get("home_elo")):
            elo[row["home_team"]] = row["home_elo"]
            seen.add(row["home_team"])
        if row["away_team"] not in seen and pd.notna(row.get("away_elo")):
            elo[row["away_team"]] = row["away_elo"]
            seen.add(row["away_team"])

    return elo


def calculate_all_forms(matches_file: str = None, league: str = None) -> Dict:
    """Calculate current form for all teams in upcoming matches.

    Args:
        matches_file: Path to upcoming matches JSON
        league: League to filter data for (e.g. "premier_league")

    Returns:
        Dict with form data for each team
    """
    if matches_file is None:
        # Choose prediction file based on league
        if league and league != "serie_a":
            preds_path = DATA_DIR / "upcoming" / f"predictions_{league}.json"
        else:
            preds_path = DATA_DIR / "upcoming" / "predictions.json"
        manual_path = DATA_DIR / "upcoming" / "manual_matches.json"
        if preds_path.exists():
            matches_file = preds_path
        elif manual_path.exists():
            matches_file = manual_path
        else:
            log.warning("No matches file found for %s", league or "serie_a")
            return {}

    # Load recent match data (filtered by league)
    df = load_recent_matches(n_matchweeks=10, league=league)
    if len(df) == 0:
        log.error("No recent match data available")
        return {}

    # Load upcoming matches
    if not Path(matches_file).exists():
        log.warning(f"No matches file found: {matches_file}")
        return {}

    with open(matches_file) as f:
        upcoming = json.load(f)

    # Support both predictions.json format and manual_matches.json format
    matches = upcoming.get("matches", [])
    if not matches and isinstance(upcoming.get("predictions"), list):
        # predictions.json stores matches under "predictions" key
        matches = upcoming["predictions"]

    teams = set()
    for match in matches:
        home = match.get("home_team", "")
        away = match.get("away_team", "")
        if home:
            teams.add(home)
        if away:
            teams.add(away)

    # Get Elo ratings
    elo = get_elo_ratings(df)

    # Calculate form for each team
    forms = {}
    for team in teams:
        form = calculate_team_form(df, team, last_n=5)
        form["elo"] = elo.get(team, 1500)  # Default Elo if not found
        forms[team] = form

    # Calculate matchup analysis
    matchups = {}
    for match in matches:
        home = match["home_team"]
        away = match["away_team"]
        key = f"{home} vs {away}"

        home_form = forms.get(home, {})
        away_form = forms.get(away, {})

        # Head-to-head
        h2h = calculate_head_to_head(df, home, away)

        # Elo difference
        home_elo = home_form.get("elo", 1500)
        away_elo = away_form.get("elo", 1500)
        elo_diff = home_elo - away_elo

        matchups[key] = {
            "home_team": home,
            "away_team": away,
            "home_form": home_form,
            "away_form": away_form,
            "head_to_head": h2h,
            "elo_diff": elo_diff,
            "factors": identify_factors(home_form, away_form, elo_diff, match)
        }

    # Save results
    output = {
        "calculated_at": datetime.now().isoformat(),
        "teams": forms,
        "matchups": matchups
    }

    _suffix = f"_{league}" if league and league != "serie_a" else ""
    output_path = DATA_DIR / "upcoming" / f"current_form{_suffix}.json"
    atomic_write_json(output_path, output)

    log.info(f"Saved form data to {output_path}")
    return output


_SERIE_A_BIG_STADIUMS = {"Inter", "Milan", "Juventus", "Roma", "Lazio", "Napoli"}
_EPL_BIG_STADIUMS = {
    "Man United", "Manchester United", "Arsenal", "Tottenham", "Tottenham Hotspur",
    "West Ham", "West Ham United", "Liverpool", "Man City", "Manchester City",
    "Newcastle", "Newcastle United",
}

_SERIE_A_DERBIES = {
    ("Inter", "Milan"), ("Milan", "Inter"),
    ("Roma", "Lazio"), ("Lazio", "Roma"),
    ("Juventus", "Torino"), ("Torino", "Juventus"),
    ("Genoa", "Sampdoria"), ("Sampdoria", "Genoa"),
}

_EPL_DERBIES = {
    ("Arsenal", "Tottenham"), ("Tottenham", "Arsenal"),
    ("Arsenal", "Tottenham Hotspur"), ("Tottenham Hotspur", "Arsenal"),
    ("Liverpool", "Everton"), ("Everton", "Liverpool"),
    ("Man United", "Man City"), ("Man City", "Man United"),
    ("Manchester United", "Manchester City"), ("Manchester City", "Manchester United"),
    ("Chelsea", "Tottenham"), ("Tottenham", "Chelsea"),
    ("Chelsea", "Tottenham Hotspur"), ("Tottenham Hotspur", "Chelsea"),
    ("Chelsea", "Arsenal"), ("Arsenal", "Chelsea"),
    ("Chelsea", "West Ham"), ("West Ham", "Chelsea"),
    ("Chelsea", "West Ham United"), ("West Ham United", "Chelsea"),
    ("Crystal Palace", "Brighton"), ("Brighton", "Crystal Palace"),
    ("Crystal Palace", "Brighton and Hove Albion"), ("Brighton and Hove Albion", "Crystal Palace"),
}

_EPL_BIG_6 = {
    "Arsenal", "Chelsea", "Liverpool", "Man City", "Manchester City",
    "Man United", "Manchester United", "Tottenham", "Tottenham Hotspur",
}


def identify_factors(home_form: Dict, away_form: Dict, elo_diff: float, match: Dict) -> List[str]:
    """Identify which validated factors apply to this match.

    Factors from 21-season validation:
    - big_stadium: Stadium capacity 50k+ (+15% home win)
    - home_favorite: Elo diff +100 (+12% home win)
    - big_home_favorite: Elo diff +200 (+20% home win)
    - hot_home: Home team hot form (+8% home win)
    - cold_away: Away team cold form (+6% home win)
    """
    factors = []
    home_team = match.get("home_team", "")
    away_team = match.get("away_team", "")

    # Big stadium
    if home_team in _SERIE_A_BIG_STADIUMS or home_team in _EPL_BIG_STADIUMS:
        factors.append("big_stadium")

    # Elo-based factors
    if elo_diff >= 200:
        factors.append("big_home_favorite")
        factors.append("home_favorite")
    elif elo_diff >= 100:
        factors.append("home_favorite")
    elif elo_diff <= -200:
        factors.append("big_away_favorite")
        factors.append("away_favorite")
    elif elo_diff <= -100:
        factors.append("away_favorite")

    # Form factors
    if home_form.get("form_status") == "hot" or home_form.get("home_form_status") == "hot":
        factors.append("hot_home")

    if away_form.get("form_status") == "cold" or away_form.get("away_form_status") == "cold":
        factors.append("cold_away")

    if home_form.get("form_status") == "cold" or home_form.get("home_form_status") == "cold":
        factors.append("cold_home")

    if away_form.get("form_status") == "hot" or away_form.get("away_form_status") == "hot":
        factors.append("hot_away")

    # Derby detection (both leagues)
    pair = (home_team, away_team)
    if pair in _SERIE_A_DERBIES or pair in _EPL_DERBIES:
        factors.append("derby")

    # EPL Big 6 matchup
    if home_team in _EPL_BIG_6 and away_team in _EPL_BIG_6:
        factors.append("big_6_matchup")

    return factors


def print_form_summary(forms: Dict):
    """Print a summary of team forms."""
    print("\n" + "=" * 70)
    print("CURRENT TEAM FORM (Last 5 Matches)")
    print("=" * 70)

    teams = forms.get("teams", {})
    sorted_teams = sorted(teams.items(), key=lambda x: x[1].get("ppg", 0), reverse=True)

    print(f"\n{'Team':<15} {'PPG':>6} {'W-D-L':>10} {'GF-GA':>8} {'Form':>8} {'Elo':>6}")
    print("-" * 60)

    for team, form in sorted_teams:
        if "error" in form:
            continue
        record = f"{form['wins']}-{form['draws']}-{form['losses']}"
        goals = f"{form['goals_scored']}-{form['goals_conceded']}"
        status = form.get("form_status", "?").upper()
        elo = form.get("elo", 1500)
        print(f"{team:<15} {form['ppg']:>6.2f} {record:>10} {goals:>8} {status:>8} {elo:>6.0f}")


def print_matchup_analysis(forms: Dict):
    """Print matchup analysis."""
    print("\n" + "=" * 70)
    print("MATCHUP ANALYSIS")
    print("=" * 70)

    matchups = forms.get("matchups", {})

    for key, data in matchups.items():
        print(f"\n{key}")
        print("-" * 40)

        home = data["home_form"]
        away = data["away_form"]

        print(f"  {data['home_team']}: {home.get('ppg', '?')} PPG, {home.get('form_status', '?').upper()} form, Elo {home.get('elo', '?'):.0f}")
        print(f"  {data['away_team']}: {away.get('ppg', '?')} PPG, {away.get('form_status', '?').upper()} form, Elo {away.get('elo', '?'):.0f}")

        h2h = data.get("head_to_head", {})
        if "error" not in h2h:
            print(f"  H2H (last {h2h['matches']}): {data['home_team']} {h2h['home_team_wins']}-{h2h['draws']}-{h2h['away_team_wins']} {data['away_team']}")

        factors = data.get("factors", [])
        if factors:
            print(f"  FACTORS: {', '.join(factors)}")


if __name__ == "__main__":
    forms = calculate_all_forms()

    if forms:
        print_form_summary(forms)
        print_matchup_analysis(forms)
