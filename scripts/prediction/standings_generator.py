#!/usr/bin/env python3
"""Generate current Serie A standings from matches.parquet.

Computes position, points, W/D/L, GF/GA/GD, and last-5 form string
for each team in the 2025-2026 season.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR, get_current_season, atomic_write_json


def generate_current_standings(season: str | None = None) -> dict:
    """Compute current Serie A standings from matches.parquet."""
    if season is None:
        season = get_current_season()
    parquet_path = DATA_DIR / "parsed" / "matches.parquet"
    if not parquet_path.exists():
        print(f"matches.parquet not found at {parquet_path}")
        return {}

    df = pd.read_parquet(parquet_path)
    # Filter to Serie A only — matches.parquet may contain multi-league data.
    # Include league=None rows (from Sofascore scraper) as they may be Serie A.
    league_filter = df["league"].isin(["serie_a", "Serie A", "I1"]) | df["league"].isna()
    season_df = df[league_filter & (df["season"] == season)].copy()
    season_df["match_date"] = pd.to_datetime(season_df["match_date"], errors="coerce")
    season_df = season_df.dropna(subset=["home_score", "away_score"])
    # Deduplicate: multiple data sources can create duplicate rows for the same match.
    # Keep the first occurrence (football-data.co.uk rows have league='serie_a' and sort first).
    season_df = season_df.drop_duplicates(subset=["home_team", "away_team", "match_date"], keep="first")
    season_df = season_df.sort_values("match_date")

    # Accumulate stats
    teams: dict[str, dict] = {}

    for _, row in season_df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        hs = int(row["home_score"])
        as_ = int(row["away_score"])

        for team in [home, away]:
            if team not in teams:
                teams[team] = {
                    "played": 0, "wins": 0, "draws": 0, "losses": 0,
                    "gf": 0, "ga": 0, "points": 0, "last_results": [],
                    "home_wins": 0, "home_draws": 0, "home_losses": 0,
                    "home_played": 0, "home_gf": 0, "home_ga": 0,
                    "away_wins": 0, "away_draws": 0, "away_losses": 0,
                    "away_played": 0, "away_gf": 0, "away_ga": 0,
                }

        # Home team
        teams[home]["played"] += 1
        teams[home]["home_played"] += 1
        teams[home]["gf"] += hs
        teams[home]["ga"] += as_
        teams[home]["home_gf"] += hs
        teams[home]["home_ga"] += as_
        # Away team
        teams[away]["played"] += 1
        teams[away]["away_played"] += 1
        teams[away]["gf"] += as_
        teams[away]["ga"] += hs
        teams[away]["away_gf"] += as_
        teams[away]["away_ga"] += hs

        if hs > as_:
            teams[home]["wins"] += 1
            teams[home]["home_wins"] += 1
            teams[home]["points"] += 3
            teams[home]["last_results"].append("W")
            teams[away]["losses"] += 1
            teams[away]["away_losses"] += 1
            teams[away]["last_results"].append("L")
        elif hs < as_:
            teams[away]["wins"] += 1
            teams[away]["away_wins"] += 1
            teams[away]["points"] += 3
            teams[away]["last_results"].append("W")
            teams[home]["losses"] += 1
            teams[home]["home_losses"] += 1
            teams[home]["last_results"].append("L")
        else:
            teams[home]["draws"] += 1
            teams[home]["home_draws"] += 1
            teams[home]["points"] += 1
            teams[home]["last_results"].append("D")
            teams[away]["draws"] += 1
            teams[away]["away_draws"] += 1
            teams[away]["points"] += 1
            teams[away]["last_results"].append("D")

    # Sort by points, GD, GF
    sorted_teams = sorted(
        teams.keys(),
        key=lambda t: (teams[t]["points"], teams[t]["gf"] - teams[t]["ga"], teams[t]["gf"]),
        reverse=True,
    )

    standings = {}
    for pos, team in enumerate(sorted_teams, start=1):
        t = teams[team]
        form_last5 = "".join(t["last_results"][-5:])
        hp = t["home_played"] or 1
        ap = t["away_played"] or 1
        home_pts = t["home_wins"] * 3 + t["home_draws"]
        away_pts = t["away_wins"] * 3 + t["away_draws"]
        standings[team] = {
            "position": pos,
            "team": team,
            "played": t["played"],
            "wins": t["wins"],
            "draws": t["draws"],
            "losses": t["losses"],
            "gf": t["gf"],
            "ga": t["ga"],
            "gd": t["gf"] - t["ga"],
            "points": t["points"],
            "form_last5": form_last5,
            "home": {
                "played": t["home_played"],
                "wins": t["home_wins"],
                "draws": t["home_draws"],
                "losses": t["home_losses"],
                "gf": t["home_gf"],
                "ga": t["home_ga"],
                "ppg": round(home_pts / hp, 2),
            },
            "away": {
                "played": t["away_played"],
                "wins": t["away_wins"],
                "draws": t["away_draws"],
                "losses": t["away_losses"],
                "gf": t["away_gf"],
                "ga": t["away_ga"],
                "ppg": round(away_pts / ap, 2),
            },
        }

    output = {
        "generated_at": datetime.now().isoformat(),
        "season": season,
        "team_count": len(standings),
        "standings": standings,
    }

    out_path = DATA_DIR / "upcoming" / "standings.json"
    atomic_write_json(out_path, output)

    print(f"Generated standings for {len(standings)} teams -> {out_path}")
    return output


if __name__ == "__main__":
    generate_current_standings()
