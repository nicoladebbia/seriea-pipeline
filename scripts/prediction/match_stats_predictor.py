"""Predict match statistics for upcoming matches.

Uses rolling averages from SofaScore + FBref data to predict:
  - Shots (home/away)
  - Shots on target
  - Corners (from historical corner rates)
  - Cards (from team foul/card rates)
  - Possession split (from touch/pass data)
  - Fouls

These predictions feed into over/under, cards, BTTS models.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config.team_names import normalize_team

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
SOFASCORE_PATH = PROJECT_ROOT / "data" / "external" / "sofascore" / "player_match_stats.parquet"
FBREF_PATH = PROJECT_ROOT / "data" / "parsed" / "player_stats.parquet"
MATCHES_PATH = PROJECT_ROOT / "data" / "parsed" / "matches.parquet"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "upcoming" / "predictions.json"

LOOKBACK = 5


def _normalize(name: str) -> str:
    if pd.isna(name):
        return ""
    return normalize_team(name)


def _build_team_rolling_stats() -> pd.DataFrame:
    """Build rolling match stats from SofaScore and FBref data.

    Returns DataFrame with one row per (team, match_date) containing
    rolling averages of key match stats.
    """
    rows = []

    # Primary source: SofaScore (richer stats)
    if SOFASCORE_PATH.exists():
        ss = pd.read_parquet(SOFASCORE_PATH)
        for c in ["total_shots", "shots_on_target", "fouls", "was_fouled",
                   "touches", "accurate_passes", "total_passes",
                   "opp_half_passes", "own_half_passes"]:
            if c in ss.columns:
                ss[c] = pd.to_numeric(ss[c], errors="coerce")

        ss["team_norm"] = ss["team"].apply(_normalize)
        ss["date"] = pd.to_datetime(ss["date"], errors="coerce")

        # Aggregate to team-match level
        team_match = ss.groupby(["match_id", "team_norm"]).agg(
            total_shots=("total_shots", "sum"),
            shots_on_target=("shots_on_target", "sum"),
            fouls=("fouls", "sum"),
            touches=("touches", "sum"),
            accurate_passes=("accurate_passes", "sum"),
            total_passes=("total_passes", "sum"),
            opp_half_passes=("opp_half_passes", "sum"),
            own_half_passes=("own_half_passes", "sum"),
        ).reset_index()

        # Add date
        meta = ss.groupby(["match_id", "team_norm"])["date"].first().reset_index()
        team_match = team_match.merge(meta, on=["match_id", "team_norm"])

        # Possession proxy: touch share requires opponent's touches
        # We'll compute it per match later

        rows.append(team_match)

    # Secondary: FBref (cards data more reliable)
    if FBREF_PATH.exists():
        fb = pd.read_parquet(FBREF_PATH)
        for c in ["cards_yellow", "cards_red", "shots", "shots_on_target", "fouls"]:
            if c in fb.columns:
                fb[c] = pd.to_numeric(fb[c], errors="coerce")

        fb["team_norm"] = fb["team"].apply(_normalize)
        fb["date"] = pd.to_datetime(fb["match_date"], errors="coerce")

        fb_team = fb.groupby(["match_id", "team_norm"]).agg(
            fb_cards_yellow=("cards_yellow", "sum"),
            fb_cards_red=("cards_red", "sum"),
            fb_shots=("shots", "sum"),
            fb_sot=("shots_on_target", "sum"),
            fb_fouls=("fouls", "sum"),
        ).reset_index()

        meta = fb.groupby(["match_id", "team_norm"])["date"].first().reset_index()
        fb_team = fb_team.merge(meta, on=["match_id", "team_norm"])

        rows.append(fb_team)

    if not rows:
        return pd.DataFrame()

    # Merge sources on (team_norm, date) since match_ids differ between sources
    if len(rows) == 2:
        # SofaScore and FBref have different match_id formats, merge by team+date
        rows[0]["_date_key"] = rows[0]["date"].dt.date
        rows[1]["_date_key"] = rows[1]["date"].dt.date
        combined = rows[0].merge(
            rows[1].drop(columns=["match_id", "date"], errors="ignore"),
            on=["team_norm", "_date_key"],
            how="outer",
        )
        combined.drop(columns=["_date_key"], errors="ignore", inplace=True)
        # Drop any duplicate columns from outer merge
        combined = combined.loc[:, ~combined.columns.duplicated(keep="first")]
    else:
        combined = rows[0]

    # Compute rolling averages per team
    combined = combined.sort_values(["team_norm", "date"]).reset_index(drop=True)

    stat_cols = [c for c in combined.columns
                 if c not in ("match_id", "team_norm", "date")
                 and combined[c].dtype in ("float64", "int64")]

    rolling_rows = []
    for team, grp in combined.groupby("team_norm"):
        grp = grp.sort_values("date").copy()
        for col in stat_cols:
            grp[f"roll_{col}"] = (
                grp[col].shift(1).rolling(LOOKBACK, min_periods=2).mean()
            )
        rolling_rows.append(grp)

    return pd.concat(rolling_rows, ignore_index=True)


def _compute_corner_rates() -> dict[str, float]:
    """Compute average corners per match per team from matches.parquet.

    Falls back to league average if corners not tracked.
    """
    if not MATCHES_PATH.exists():
        return {}

    m = pd.read_parquet(MATCHES_PATH)

    # Check if corner data exists
    if "home_corners" not in m.columns:
        return {}

    rates = {}
    for team in set(m["home_team"].unique()) | set(m["away_team"].unique()):
        home_corners = m[m["home_team"] == team]["home_corners"].dropna()
        away_corners = m[m["away_team"] == team]["away_corners"].dropna()
        all_corners = pd.concat([home_corners, away_corners])
        if len(all_corners) >= 5:
            rates[team] = all_corners.mean()

    return rates


def predict_match_stats(home_team: str, away_team: str,
                        rolling_data: pd.DataFrame = None) -> dict:
    """Predict match statistics for a single match.

    Returns dict with predicted stats for both teams.
    """
    if rolling_data is None:
        rolling_data = _build_team_rolling_stats()

    result = {
        "home_team": home_team,
        "away_team": away_team,
    }

    # Get most recent rolling stats for each team
    for side, team in [("home", home_team), ("away", away_team)]:
        team_norm = _normalize(team)
        team_data = rolling_data[rolling_data["team_norm"] == team_norm]

        if team_data.empty:
            # League averages as fallback
            result[f"pred_{side}_shots"] = 12.0
            result[f"pred_{side}_sot"] = 4.0
            result[f"pred_{side}_fouls"] = 13.0
            result[f"pred_{side}_cards"] = 2.0
            result[f"pred_{side}_possession"] = 50.0
            continue

        latest = team_data.sort_values("date").iloc[-1]

        # Shots
        shots = latest.get("roll_total_shots", latest.get("roll_fb_shots"))
        result[f"pred_{side}_shots"] = round(shots, 1) if pd.notna(shots) else 12.0

        # Shots on target
        sot = latest.get("roll_shots_on_target", latest.get("roll_fb_sot"))
        result[f"pred_{side}_sot"] = round(sot, 1) if pd.notna(sot) else 4.0

        # Fouls
        fouls = latest.get("roll_fouls", latest.get("roll_fb_fouls"))
        result[f"pred_{side}_fouls"] = round(fouls, 1) if pd.notna(fouls) else 13.0

        # Cards (from FBref, more reliable)
        cards = latest.get("roll_fb_cards_yellow")
        result[f"pred_{side}_cards"] = round(cards, 1) if pd.notna(cards) else 2.0

        # Possession proxy (touch-based)
        touches = latest.get("roll_touches")
        opp_passes = latest.get("roll_opp_half_passes")
        own_passes = latest.get("roll_own_half_passes")
        if pd.notna(touches):
            result[f"_raw_{side}_touches"] = round(touches, 0)
        if pd.notna(opp_passes) and pd.notna(own_passes):
            total = opp_passes + own_passes
            result[f"pred_{side}_territory"] = round(100 * opp_passes / max(total, 1), 1)

    # Normalize possession to sum to ~100%
    h_touches = result.get("_raw_home_touches", 500)
    a_touches = result.get("_raw_away_touches", 500)
    total_touches = h_touches + a_touches
    if total_touches > 0:
        result["pred_home_possession"] = round(100 * h_touches / total_touches, 1)
        result["pred_away_possession"] = round(100 * a_touches / total_touches, 1)
    else:
        result["pred_home_possession"] = 50.0
        result["pred_away_possession"] = 50.0

    # Total predicted cards
    result["pred_total_cards"] = round(
        result.get("pred_home_cards", 2.0) + result.get("pred_away_cards", 2.0), 1
    )

    # Total predicted shots
    result["pred_total_shots"] = round(
        result.get("pred_home_shots", 12.0) + result.get("pred_away_shots", 12.0), 1
    )

    # Cleanup internal keys
    for k in list(result.keys()):
        if k.startswith("_raw_"):
            del result[k]

    return result


def add_match_stats_predictions(predictions: list[dict] = None) -> list[dict]:
    """Add predicted match stats to each match in predictions list."""
    if predictions is None:
        if not PREDICTIONS_PATH.exists():
            log.warning("No predictions.json found")
            return []
        with open(PREDICTIONS_PATH) as f:
            predictions = json.load(f)

    rolling = _build_team_rolling_stats()

    for pred in predictions:
        home = pred.get("home_team", "")
        away = pred.get("away_team", "")
        if home and away:
            stats = predict_match_stats(home, away, rolling)
            pred["predicted_stats"] = stats

    log.info("Added predicted match stats for %d matches", len(predictions))
    return predictions


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with some upcoming matches
    test_matches = [
        ("Inter", "Milan"),
        ("Juventus", "Napoli"),
        ("Roma", "Lazio"),
    ]

    rolling = _build_team_rolling_stats()
    print(f"Rolling data: {len(rolling)} team-match rows")
    print()

    for home, away in test_matches:
        stats = predict_match_stats(home, away, rolling)
        print(f"\n{home} vs {away}:")
        for k, v in sorted(stats.items()):
            if k not in ("home_team", "away_team"):
                print(f"  {k}: {v}")
