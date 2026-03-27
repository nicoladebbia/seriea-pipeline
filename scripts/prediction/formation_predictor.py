"""Predict likely formations for upcoming matches.

Uses two data sources:
1. SofaScore starter position counts (G/D/M/F) → infer formation
2. matches.parquet historical formation strings (when available)

Strategy: frequency-based — most common formation in team's last N matches.

Output: adds predicted_home_formation, predicted_away_formation to predictions.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
SOFASCORE_PATH = PROJECT_ROOT / "data" / "external" / "sofascore" / "player_match_stats.parquet"
MATCHES_PATH = PROJECT_ROOT / "data" / "parsed" / "matches.parquet"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "upcoming" / "predictions.json"

LOOKBACK = 5  # last N matches

# Map (D, M, F) counts to formation string
# GK is always 1, so we only need outfield positions
POSITION_TO_FORMATION = {
    (4, 4, 2): "4-4-2",
    (4, 3, 3): "4-3-3",
    (4, 5, 1): "4-5-1",
    (4, 2, 4): "4-2-4",
    (3, 5, 2): "3-5-2",
    (3, 4, 3): "3-4-3",
    (5, 4, 1): "5-4-1",
    (5, 3, 2): "5-3-2",
    (3, 3, 4): "3-3-4",
    (4, 1, 5): "4-1-5",
    (4, 4, 1): "4-4-1",  # defensive
    (3, 4, 2): "3-4-2-1",  # approximate
    (3, 3, 3): "3-3-3-1",  # approximate
}

from config.team_names import normalize_team


def _normalize(name: str) -> str:
    if pd.isna(name):
        return ""
    return normalize_team(name)


def _infer_formation_from_positions(d_count: int, m_count: int, f_count: int) -> str:
    """Infer formation from D/M/F counts."""
    key = (d_count, m_count, f_count)
    if key in POSITION_TO_FORMATION:
        return POSITION_TO_FORMATION[key]
    # Fallback: construct from counts
    return f"{d_count}-{m_count}-{f_count}"


def get_team_recent_formations(team: str, n: int = LOOKBACK) -> list[str]:
    """Get a team's most recent N formations from available data.

    Combines matches.parquet explicit formations with SofaScore-inferred ones.
    """
    formations = []

    # Source 1: matches.parquet (explicit formations from current season)
    if MATCHES_PATH.exists():
        matches = pd.read_parquet(MATCHES_PATH)
        matches["match_date"] = pd.to_datetime(matches["match_date"], errors="coerce")
        matches = matches.sort_values("match_date", ascending=False)

        # Home matches
        home = matches[matches["home_team"] == team].head(n * 2)
        for _, row in home.iterrows():
            if pd.notna(row.get("home_formation")):
                formations.append((row["match_date"], row["home_formation"]))

        # Away matches
        away = matches[matches["away_team"] == team].head(n * 2)
        for _, row in away.iterrows():
            if pd.notna(row.get("away_formation")):
                formations.append((row["match_date"], row["away_formation"]))

    # Source 2: SofaScore starter positions
    if SOFASCORE_PATH.exists():
        ss = pd.read_parquet(SOFASCORE_PATH)
        ss["team_norm"] = ss["team"].apply(_normalize)
        ss["date"] = pd.to_datetime(ss["date"], errors="coerce")

        team_data = ss[ss["team_norm"] == team]
        starters = team_data[team_data["is_starter"] == 1]

        if not starters.empty:
            # Get position counts per match
            match_dates = starters.groupby("match_id")["date"].first()
            for mid in match_dates.sort_values(ascending=False).index[:n * 3]:
                match_starters = starters[starters["match_id"] == mid]
                pos_counts = match_starters["position"].value_counts()
                d = pos_counts.get("D", 0)
                m = pos_counts.get("M", 0)
                f = pos_counts.get("F", 0)
                if d + m + f >= 9:  # at least 9 outfield
                    formation = _infer_formation_from_positions(d, m, f)
                    dt = match_dates[mid]
                    formations.append((dt, formation))

    if not formations:
        return []

    # Sort by date descending, take most recent n
    formations.sort(key=lambda x: x[0] if pd.notna(x[0]) else pd.Timestamp.min, reverse=True)
    return [f[1] for f in formations[:n]]


def predict_formation(team: str) -> dict:
    """Predict most likely formation for a team.

    Returns dict with formation, confidence, and alternatives.
    """
    recent = get_team_recent_formations(team, LOOKBACK)

    if not recent:
        return {"formation": "4-3-3", "confidence": 0.0, "alternatives": []}

    counter = Counter(recent)
    most_common = counter.most_common()
    total = len(recent)

    predicted = most_common[0][0]
    confidence = most_common[0][1] / total

    alternatives = [
        {"formation": f, "frequency": c / total}
        for f, c in most_common[1:3]
    ]

    return {
        "formation": predicted,
        "confidence": round(confidence, 2),
        "alternatives": alternatives,
    }


def add_formation_predictions(predictions: list[dict] | None = None) -> list[dict]:
    """Add formation predictions to each match in predictions list.

    If predictions is None, loads from predictions.json.
    """
    if predictions is None:
        if not PREDICTIONS_PATH.exists():
            log.warning("No predictions.json found")
            return []
        with open(PREDICTIONS_PATH) as f:
            predictions = json.load(f)

    for pred in predictions:
        home = pred.get("home_team", "")
        away = pred.get("away_team", "")

        if home:
            home_pred = predict_formation(home)
            pred["predicted_home_formation"] = home_pred["formation"]
            pred["home_formation_confidence"] = home_pred["confidence"]

        if away:
            away_pred = predict_formation(away)
            pred["predicted_away_formation"] = away_pred["formation"]
            pred["away_formation_confidence"] = away_pred["confidence"]

    log.info("Added formation predictions for %d matches", len(predictions))
    return predictions


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with a few teams
    for team in ["Inter", "Milan", "Juventus", "Napoli", "Roma"]:
        result = predict_formation(team)
        recent = get_team_recent_formations(team, 5)
        print(f"{team}: {result['formation']} (conf={result['confidence']:.0%}, recent={recent})")
