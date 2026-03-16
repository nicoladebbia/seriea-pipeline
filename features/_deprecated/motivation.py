"""Motivation Intelligence Module - Psychological pressure features.

This module captures situational factors that affect team motivation:
- Relegation battles (fighting for survival)
- Title races (championship contention)
- European qualification (4th-7th place battles)
- Derby matches (historical rivalries)
- Revenge factor (recent heavy losses)
- Safe position (nothing to play for)

These factors significantly impact match outcomes but are missed by
pure statistical models.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

# Serie A rivalries with intensity scores (1-10)
SERIE_A_DERBIES = {
    # City derbies
    ("Inter", "Milan"): 10,  # Derby della Madonnina
    ("Milan", "Inter"): 10,
    ("Roma", "Lazio"): 10,  # Derby della Capitale
    ("Lazio", "Roma"): 10,
    ("Juventus", "Torino"): 9,  # Derby della Mole
    ("Torino", "Juventus"): 9,
    ("Genoa", "Sampdoria"): 8,  # Derby della Lanterna
    ("Sampdoria", "Genoa"): 8,

    # Historic rivalries
    ("Juventus", "Inter"): 9,  # Derby d'Italia
    ("Inter", "Juventus"): 9,
    ("Juventus", "Milan"): 8,
    ("Milan", "Juventus"): 8,
    ("Juventus", "Napoli"): 7,
    ("Napoli", "Juventus"): 7,
    ("Juventus", "Fiorentina"): 7,
    ("Fiorentina", "Juventus"): 7,
    ("Inter", "Roma"): 6,
    ("Roma", "Inter"): 6,
    ("Milan", "Napoli"): 6,
    ("Napoli", "Milan"): 6,
    ("Inter", "Napoli"): 7,
    ("Napoli", "Inter"): 7,

    # Regional rivalries
    ("Bologna", "Parma"): 6,  # Derby dell'Emilia
    ("Parma", "Bologna"): 6,
    ("Napoli", "Roma"): 7,  # Derby del Sole
    ("Roma", "Napoli"): 7,
    ("Atalanta", "Brescia"): 6,
    ("Brescia", "Atalanta"): 6,
    ("Fiorentina", "Empoli"): 5,  # Tuscan derby
    ("Empoli", "Fiorentina"): 5,
    ("Verona", "Chievo"): 5,  # Verona derby
    ("Chievo", "Verona"): 5,
    ("Hellas Verona", "Chievo"): 5,
    ("Chievo", "Hellas Verona"): 5,
}


def get_derby_intensity(home_team: str, away_team: str) -> float:
    """Get derby intensity score (0-10) for a matchup."""
    key = (home_team, away_team)
    return SERIE_A_DERBIES.get(key, 0)


def compute_league_position_context(
    df: pd.DataFrame,
    team_col: str,
    date_col: str,
    position_col: str,
    points_col: str,
) -> pd.DataFrame:
    """Compute league position context for each team at each match.

    Returns DataFrame with:
    - points_to_safety: Points gap to 18th place (relegation)
    - points_to_europe: Points gap to 6th place (Europa League)
    - points_to_title: Points gap to 1st place
    - position_tier: categorical (relegation/midtable/european/title)
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Group by season and compute standings at each matchweek
    # For now, we use a simplified approach using the position column if available

    if position_col not in df.columns or points_col not in df.columns:
        log.warning(f"Position/points columns not found. Using defaults.")
        df["points_to_safety"] = 10  # Default safe
        df["points_to_europe"] = 20
        df["points_to_title"] = 30
        df["position_tier"] = "midtable"
        return df

    # Simple thresholds for Serie A (20 teams)
    def categorize_position(pos, pts):
        if pd.isna(pos):
            return "unknown"
        pos = int(pos)
        if pos >= 18:
            return "relegation"
        elif pos <= 4:
            return "title"
        elif pos <= 7:
            return "european"
        else:
            return "midtable"

    df["position_tier"] = df.apply(
        lambda row: categorize_position(row.get(position_col), row.get(points_col)),
        axis=1
    )

    return df


def compute_motivation_score(
    position: int,
    points: int,
    matchweek: int,
    points_to_safety: Optional[float] = None,
    points_to_title: Optional[float] = None,
    is_derby: bool = False,
    derby_intensity: float = 0,
    recent_h2h_loss: bool = False,
) -> dict[str, float]:
    """Compute motivation score based on situational factors.

    Args:
        position: Current league position (1-20)
        points: Current points total
        matchweek: Current matchweek (1-38)
        points_to_safety: Gap to safety (negative if in danger)
        points_to_title: Gap to leader
        is_derby: Whether this is a derby match
        derby_intensity: Derby intensity score (0-10)
        recent_h2h_loss: Lost heavily to this opponent recently

    Returns:
        Dictionary of motivation-related features
    """
    features = {}

    # Matches remaining determines urgency
    matches_remaining = max(1, 38 - matchweek)

    # Base motivation (everyone wants to win)
    base_motivation = 0.5

    # Relegation battle motivation
    if position and position >= 15:
        # In danger zone
        relegation_urgency = min(1.0, (position - 14) / 6)  # 0 at 14th, 1 at 20th

        # Urgency increases as season progresses
        time_pressure = min(1.0, matchweek / 30)  # Max pressure after MW30

        relegation_motivation = relegation_urgency * (0.5 + 0.5 * time_pressure)
        features["relegation_battle_motivation"] = round(relegation_motivation, 3)
    else:
        features["relegation_battle_motivation"] = 0.0

    # Title race motivation
    if position and position <= 4 and points_to_title is not None:
        if points_to_title <= 9:  # Within realistic reach
            title_chance = max(0, 1 - points_to_title / 15)
            time_factor = min(1.0, matchweek / 25)
            title_motivation = title_chance * (0.5 + 0.5 * time_factor)
            features["title_race_motivation"] = round(title_motivation, 3)
        else:
            features["title_race_motivation"] = 0.0
    else:
        features["title_race_motivation"] = 0.0

    # European qualification motivation (positions 4-7)
    if position and 4 <= position <= 8:
        european_motivation = 0.6  # Moderate motivation
        if matchweek >= 30:
            european_motivation = 0.8  # Increased near season end
        features["european_race_motivation"] = round(european_motivation, 3)
    else:
        features["european_race_motivation"] = 0.0

    # Safe position (nothing to play for = lower motivation)
    if position and 10 <= position <= 14:
        if matchweek >= 32:
            # Late season, safe position
            features["safe_position_flag"] = 1.0
            features["motivation_penalty"] = 0.2  # Slight motivation drop
        else:
            features["safe_position_flag"] = 0.0
            features["motivation_penalty"] = 0.0
    else:
        features["safe_position_flag"] = 0.0
        features["motivation_penalty"] = 0.0

    # Derby premium
    if is_derby or derby_intensity > 0:
        features["is_derby"] = 1.0
        features["derby_intensity"] = derby_intensity / 10  # Normalize to 0-1
        features["derby_motivation_boost"] = min(0.3, derby_intensity / 30)
    else:
        features["is_derby"] = 0.0
        features["derby_intensity"] = 0.0
        features["derby_motivation_boost"] = 0.0

    # Revenge factor
    if recent_h2h_loss:
        features["revenge_motivation"] = 0.15
    else:
        features["revenge_motivation"] = 0.0

    # Overall motivation index (weighted combination)
    overall = (
        base_motivation +
        features["relegation_battle_motivation"] * 0.3 +
        features["title_race_motivation"] * 0.3 +
        features["european_race_motivation"] * 0.15 +
        features["derby_motivation_boost"] +
        features["revenge_motivation"] -
        features["motivation_penalty"]
    )
    features["motivation_index"] = round(min(1.0, max(0.0, overall)), 3)

    return features


def add_motivation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add motivation features to match DataFrame.

    Requires columns:
    - home_team, away_team
    - home_position, away_position (league standings)
    - home_points, away_points (current points)
    - matchweek

    Adds columns:
    - home_motivation_index, away_motivation_index
    - home_relegation_battle, away_relegation_battle
    - home_title_race, away_title_race
    - is_derby, derby_intensity
    - motivation_diff (home - away)
    """
    df = df.copy()

    log.info("Adding motivation features...")

    # Initialize columns
    motivation_cols = [
        "home_motivation_index", "away_motivation_index",
        "home_relegation_battle_motivation", "away_relegation_battle_motivation",
        "home_title_race_motivation", "away_title_race_motivation",
        "home_european_race_motivation", "away_european_race_motivation",
        "is_derby", "derby_intensity",
        "motivation_diff",
    ]

    for col in motivation_cols:
        df[col] = 0.0

    # Process each match
    for idx, row in df.iterrows():
        home_team = row.get("home_team", "")
        away_team = row.get("away_team", "")
        matchweek = row.get("matchweek", 19)  # Default mid-season

        # Get derby info
        derby_intensity = get_derby_intensity(home_team, away_team)
        is_derby = derby_intensity > 0

        # Home team motivation
        home_pos = row.get("home_position") or row.get("home_league_position")
        home_pts = row.get("home_points") or row.get("home_season_points", 0)
        home_to_title = row.get("home_points_to_leader", 30)

        if pd.notna(home_pos):
            home_motivation = compute_motivation_score(
                position=int(home_pos) if pd.notna(home_pos) else None,
                points=int(home_pts) if pd.notna(home_pts) else 0,
                matchweek=int(matchweek) if pd.notna(matchweek) else 19,
                points_to_title=home_to_title,
                is_derby=is_derby,
                derby_intensity=derby_intensity,
            )

            df.at[idx, "home_motivation_index"] = home_motivation["motivation_index"]
            df.at[idx, "home_relegation_battle_motivation"] = home_motivation["relegation_battle_motivation"]
            df.at[idx, "home_title_race_motivation"] = home_motivation["title_race_motivation"]
            df.at[idx, "home_european_race_motivation"] = home_motivation["european_race_motivation"]

        # Away team motivation
        away_pos = row.get("away_position") or row.get("away_league_position")
        away_pts = row.get("away_points") or row.get("away_season_points", 0)
        away_to_title = row.get("away_points_to_leader", 30)

        if pd.notna(away_pos):
            away_motivation = compute_motivation_score(
                position=int(away_pos) if pd.notna(away_pos) else None,
                points=int(away_pts) if pd.notna(away_pts) else 0,
                matchweek=int(matchweek) if pd.notna(matchweek) else 19,
                points_to_title=away_to_title,
                is_derby=is_derby,
                derby_intensity=derby_intensity,
            )

            df.at[idx, "away_motivation_index"] = away_motivation["motivation_index"]
            df.at[idx, "away_relegation_battle_motivation"] = away_motivation["relegation_battle_motivation"]
            df.at[idx, "away_title_race_motivation"] = away_motivation["title_race_motivation"]
            df.at[idx, "away_european_race_motivation"] = away_motivation["european_race_motivation"]

        # Derby features
        df.at[idx, "is_derby"] = float(is_derby)
        df.at[idx, "derby_intensity"] = derby_intensity / 10

    # Motivation differential
    df["motivation_diff"] = df["home_motivation_index"] - df["away_motivation_index"]

    n_derbies = (df["is_derby"] > 0).sum()
    n_relegation = ((df["home_relegation_battle_motivation"] > 0.3) |
                   (df["away_relegation_battle_motivation"] > 0.3)).sum()
    n_title = ((df["home_title_race_motivation"] > 0.3) |
              (df["away_title_race_motivation"] > 0.3)).sum()

    log.info(f"Added motivation features: {n_derbies} derbies, "
             f"{n_relegation} relegation battles, {n_title} title races")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing motivation module...")
    print("=" * 60)

    # Test derby detection
    print("\nDerby Tests:")
    test_matches = [
        ("Inter", "Milan"),
        ("Roma", "Lazio"),
        ("Juventus", "Torino"),
        ("Juventus", "Inter"),
        ("Napoli", "Bologna"),  # Not a derby
    ]

    for home, away in test_matches:
        intensity = get_derby_intensity(home, away)
        print(f"  {home} vs {away}: intensity = {intensity}")

    # Test motivation computation
    print("\nMotivation Score Tests:")

    # Relegation battle
    result = compute_motivation_score(
        position=19, points=20, matchweek=32,
        points_to_safety=-3, is_derby=False
    )
    print(f"\n  Relegation battle (19th, MW32):")
    print(f"    Relegation motivation: {result['relegation_battle_motivation']}")
    print(f"    Overall index: {result['motivation_index']}")

    # Title race
    result = compute_motivation_score(
        position=2, points=65, matchweek=30,
        points_to_title=3, is_derby=False
    )
    print(f"\n  Title race (2nd, 3pts behind, MW30):")
    print(f"    Title motivation: {result['title_race_motivation']}")
    print(f"    Overall index: {result['motivation_index']}")

    # Derby
    result = compute_motivation_score(
        position=8, points=40, matchweek=25,
        is_derby=True, derby_intensity=10
    )
    print(f"\n  Derby (midtable, intensity=10):")
    print(f"    Derby boost: {result['derby_motivation_boost']}")
    print(f"    Overall index: {result['motivation_index']}")

    # Safe position
    result = compute_motivation_score(
        position=12, points=45, matchweek=35,
        is_derby=False
    )
    print(f"\n  Safe position (12th, MW35):")
    print(f"    Safe flag: {result['safe_position_flag']}")
    print(f"    Motivation penalty: {result['motivation_penalty']}")
    print(f"    Overall index: {result['motivation_index']}")

    print("\n" + "=" * 60)
