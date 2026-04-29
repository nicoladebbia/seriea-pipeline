"""Compute per-team scoring/clean-sheet rates from historical match data.

Output: data/cache/team_rates.json — keyed by team name, with home/away
scoring rate, clean-sheet rate, and expected goals scored/conceded. Used by
btts_corners_model.py and cards_model.py instead of hardcoded SA-only dicts.

Usage:
    python -m scripts.data.build_team_rates           # build if missing
    python -m scripts.data.build_team_rates --rebuild # force regenerate
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from config.leagues import ACTIVE_LEAGUES
from config.settings import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OUT_PATH = DATA_DIR / "cache" / "team_rates.json"
LOOKBACK_SEASONS = ("2023-2024", "2024-2025", "2025-2026")
MIN_MATCHES = 5  # team needs at least N matches at a venue to publish a rate


def _compute_venue_rates(df: pd.DataFrame, venue: str) -> Dict[str, Dict[str, float]]:
    """venue is 'home' or 'away'. Returns dict of team -> rates."""
    team_col = f"{venue}_team"
    own_score_col = f"{venue}_score"
    opp_score_col = "away_score" if venue == "home" else "home_score"

    out: Dict[str, Dict[str, float]] = {}
    for team, g in df.groupby(team_col):
        n = len(g)
        if n < MIN_MATCHES:
            continue
        scoring_rate = float((g[own_score_col] >= 1).mean())
        clean_sheet_rate = float((g[opp_score_col] == 0).mean())
        expected_goals = float(g[own_score_col].mean())
        expected_conceded = float(g[opp_score_col].mean())
        out[team] = {
            "scoring_rate": round(scoring_rate, 3),
            "clean_sheet_rate": round(clean_sheet_rate, 3),
            "expected_goals": round(expected_goals, 2),
            "expected_conceded": round(expected_conceded, 2),
            "n": int(n),
        }
    return out


def build_team_rates() -> Dict:
    matches_path = DATA_DIR / "parsed" / "matches.parquet"
    if not matches_path.exists():
        raise FileNotFoundError(f"matches.parquet not found at {matches_path}")

    df = pd.read_parquet(matches_path)
    df = df[df["league"].isin(ACTIVE_LEAGUES) & df["season"].isin(LOOKBACK_SEASONS)]
    log.info("Loaded %d matches across %d seasons", len(df), len(LOOKBACK_SEASONS))

    rates: Dict[str, Dict] = {}
    for league in ACTIVE_LEAGUES:
        league_df = df[df["league"] == league]
        home = _compute_venue_rates(league_df, "home")
        away = _compute_venue_rates(league_df, "away")
        log.info("%s: %d teams home, %d teams away", league, len(home), len(away))

        # Combine into per-team dict
        teams = set(home) | set(away)
        for team in teams:
            rates[team] = {
                "league": league,
                "home": home.get(team),
                "away": away.get(team),
            }

    return {
        "generated_at": pd.Timestamp.now().isoformat(),
        "lookback_seasons": list(LOOKBACK_SEASONS),
        "min_matches": MIN_MATCHES,
        "n_teams": len(rates),
        "teams": rates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true",
                        help="Force regeneration even if cache exists")
    args = parser.parse_args()

    if OUT_PATH.exists() and not args.rebuild:
        log.info("Cache exists at %s. Pass --rebuild to regenerate.", OUT_PATH)
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = build_team_rates()
    with open(OUT_PATH, "w") as f:
        json.dump(data, f, indent=2)
    log.info("Wrote team rates: %d teams -> %s", data["n_teams"], OUT_PATH)


if __name__ == "__main__":
    main()
