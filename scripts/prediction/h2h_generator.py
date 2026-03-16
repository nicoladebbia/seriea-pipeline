#!/usr/bin/env python3
"""Generate head-to-head records for upcoming matches.

For each match in predictions.json, computes H2H history from matches.parquet:
total meetings, wins per side, draws, last result, avg goals.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR


def generate_h2h_for_upcoming() -> dict:
    """For each match in predictions.json, compute H2H from matches.parquet."""
    pred_path = DATA_DIR / "upcoming" / "predictions.json"
    parquet_path = DATA_DIR / "parsed" / "matches.parquet"

    if not pred_path.exists() or not parquet_path.exists():
        print("Missing predictions.json or matches.parquet")
        return {}

    with open(pred_path) as f:
        pred_data = json.load(f)

    df = pd.read_parquet(parquet_path)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"])
    df = df.sort_values("match_date")

    h2h_data = {}

    for pred in pred_data.get("predictions", []):
        home = pred.get("home_team", "")
        away = pred.get("away_team", "")
        match_key = pred.get("match", f"{home} vs {away}")

        if not home or not away:
            continue

        # All prior meetings (either direction)
        prior = df[
            ((df["home_team"] == home) & (df["away_team"] == away))
            | ((df["home_team"] == away) & (df["away_team"] == home))
        ]

        n = len(prior)
        if n == 0:
            h2h_data[match_key] = {
                "total_meetings": 0,
                "home_wins": 0,
                "away_wins": 0,
                "draws": 0,
                "home_goals_avg": 0,
                "away_goals_avg": 0,
                "last_result": None,
                "last_score": None,
            }
            continue

        home_wins = 0
        away_wins = 0
        draws = 0
        home_goals_total = 0
        away_goals_total = 0
        last_result = None
        last_score = None

        for _, row in prior.iterrows():
            hs = int(row["home_score"])
            as_ = int(row["away_score"])

            if row["home_team"] == home:
                home_goals_total += hs
                away_goals_total += as_
                if hs > as_:
                    home_wins += 1
                    last_result = "home_win"
                    last_score = f"{hs}-{as_}"
                elif hs < as_:
                    away_wins += 1
                    last_result = "away_win"
                    last_score = f"{hs}-{as_}"
                else:
                    draws += 1
                    last_result = "draw"
                    last_score = f"{hs}-{as_}"
            else:
                # Reversed direction
                home_goals_total += as_
                away_goals_total += hs
                if as_ > hs:
                    home_wins += 1
                    last_result = "home_win"
                    last_score = f"{as_}-{hs}"
                elif as_ < hs:
                    away_wins += 1
                    last_result = "away_win"
                    last_score = f"{as_}-{hs}"
                else:
                    draws += 1
                    last_result = "draw"
                    last_score = f"{hs}-{as_}"

        h2h_data[match_key] = {
            "total_meetings": n,
            "home_wins": home_wins,
            "away_wins": away_wins,
            "draws": draws,
            "home_goals_avg": round(home_goals_total / n, 2),
            "away_goals_avg": round(away_goals_total / n, 2),
            "last_result": last_result,
            "last_score": last_score,
        }

    output = {
        "generated_at": datetime.now().isoformat(),
        "match_count": len(h2h_data),
        "h2h": h2h_data,
    }

    out_path = DATA_DIR / "upcoming" / "h2h_upcoming.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Generated H2H for {len(h2h_data)} matches -> {out_path}")
    return output


if __name__ == "__main__":
    generate_h2h_for_upcoming()
