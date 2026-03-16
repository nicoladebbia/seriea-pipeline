"""Tactical Context Module - Playing style classification and matchup analysis.

This module infers tactical styles from available statistics and computes
matchup advantages based on historical patterns.

Tactical styles detected:
- Possession-based (high possession, short passes)
- Direct/Counter-attacking (low possession, fast transitions)
- High-pressing (high tackles, interceptions in final third)
- Defensive/Low-block (compact defense, low possession)
- Wing-play (crosses, wide touches)

Matchup advantages:
- How different styles perform against each other historically
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class TacticalProfile:
    """Team's tactical profile based on recent matches."""
    team: str
    possession_style: float  # 0 = direct, 1 = possession
    pressing_intensity: float  # 0 = low block, 1 = high press
    directness: float  # 0 = patient buildup, 1 = direct
    wing_focus: float  # 0 = central, 1 = wing-heavy
    defensive_solidity: float  # 0 = open, 1 = solid


def classify_playing_style(
    possession: float,
    passes_per_90: float,
    progressive_passes: float,
    tackles: float,
    interceptions: float,
    crosses: float,
    long_balls: float,
    short_passes: float,
) -> TacticalProfile:
    """Classify a team's playing style from statistics.

    Args:
        possession: Average possession %
        passes_per_90: Total passes per 90 minutes
        progressive_passes: Progressive passes per 90
        tackles: Tackles per 90
        interceptions: Interceptions per 90
        crosses: Crosses per 90
        long_balls: Long balls per 90
        short_passes: Short passes per 90

    Returns:
        TacticalProfile with style indicators
    """
    # Possession style (0-1)
    # High possession + many passes = possession-based
    if possession > 55 and passes_per_90 > 450:
        possession_style = 0.9
    elif possession > 50 and passes_per_90 > 400:
        possession_style = 0.7
    elif possession < 45:
        possession_style = 0.3
    else:
        possession_style = 0.5

    # Pressing intensity (0-1)
    # High tackles + interceptions = high press
    defensive_actions = tackles + interceptions
    if defensive_actions > 25:
        pressing_intensity = 0.9
    elif defensive_actions > 20:
        pressing_intensity = 0.7
    elif defensive_actions > 15:
        pressing_intensity = 0.5
    else:
        pressing_intensity = 0.3

    # Directness (0-1)
    # High long balls / low short passes = direct
    if short_passes > 0:
        pass_ratio = long_balls / (short_passes + 1)
    else:
        pass_ratio = 0.1

    if pass_ratio > 0.15 or possession < 45:
        directness = 0.8
    elif pass_ratio > 0.10:
        directness = 0.6
    elif pass_ratio < 0.05:
        directness = 0.2
    else:
        directness = 0.4

    # Wing focus (0-1)
    # High crosses = wing-focused
    if crosses > 20:
        wing_focus = 0.9
    elif crosses > 15:
        wing_focus = 0.7
    elif crosses > 10:
        wing_focus = 0.5
    else:
        wing_focus = 0.3

    # Defensive solidity (inferred from progressive actions ratio)
    # Low progressive passes = more defensive
    if progressive_passes < 8:
        defensive_solidity = 0.8
    elif progressive_passes < 12:
        defensive_solidity = 0.6
    elif progressive_passes > 15:
        defensive_solidity = 0.3
    else:
        defensive_solidity = 0.5

    return TacticalProfile(
        team="",
        possession_style=round(possession_style, 2),
        pressing_intensity=round(pressing_intensity, 2),
        directness=round(directness, 2),
        wing_focus=round(wing_focus, 2),
        defensive_solidity=round(defensive_solidity, 2),
    )


# Historical matchup patterns (based on football analysis)
# Format: (style1, style2) -> home_advantage_modifier
# Positive = benefits style1, Negative = benefits style2
STYLE_MATCHUP_MATRIX = {
    # High press vs low block: Press usually wins if executed well
    ("high_press", "low_block"): 0.05,

    # Possession vs counter: Counter can exploit possession teams
    ("possession", "counter"): -0.03,

    # Direct vs possession: Direct struggles vs possession
    ("direct", "possession"): -0.02,

    # High press vs counter: Counter can exploit high line
    ("high_press", "counter"): -0.04,

    # Wing play vs narrow defense: Wings can stretch narrow teams
    ("wing_heavy", "narrow"): 0.03,

    # Similar styles: Usually competitive (draws more likely)
    ("possession", "possession"): 0.00,
    ("counter", "counter"): 0.00,
}


def compute_style_matchup(
    home_profile: TacticalProfile,
    away_profile: TacticalProfile,
) -> dict[str, float]:
    """Compute tactical matchup features.

    Args:
        home_profile: Home team's tactical profile
        away_profile: Away team's tactical profile

    Returns:
        Dictionary of matchup-related features
    """
    features = {}

    # Direct style comparisons
    features["possession_diff"] = home_profile.possession_style - away_profile.possession_style
    features["pressing_diff"] = home_profile.pressing_intensity - away_profile.pressing_intensity
    features["directness_diff"] = home_profile.directness - away_profile.directness
    features["wing_focus_diff"] = home_profile.wing_focus - away_profile.wing_focus
    features["defensive_diff"] = home_profile.defensive_solidity - away_profile.defensive_solidity

    # Style compatibility (similar styles = more competitive)
    style_similarity = 1 - (
        abs(features["possession_diff"]) +
        abs(features["pressing_diff"]) +
        abs(features["directness_diff"])
    ) / 3
    features["style_similarity"] = round(style_similarity, 3)

    # Draw likelihood based on style similarity
    # Similar styles tend to produce more draws
    features["style_draw_factor"] = round(style_similarity * 0.1, 3)

    # Specific matchup advantages
    # Press vs Low Block
    if home_profile.pressing_intensity > 0.7 and away_profile.defensive_solidity > 0.7:
        features["press_vs_block"] = 0.05  # Slight press advantage
    elif away_profile.pressing_intensity > 0.7 and home_profile.defensive_solidity > 0.7:
        features["press_vs_block"] = -0.05
    else:
        features["press_vs_block"] = 0.0

    # Counter vs Possession
    if home_profile.directness > 0.7 and away_profile.possession_style > 0.7:
        features["counter_vs_possession"] = 0.03  # Counter can exploit
    elif away_profile.directness > 0.7 and home_profile.possession_style > 0.7:
        features["counter_vs_possession"] = -0.03
    else:
        features["counter_vs_possession"] = 0.0

    # Combined tactical advantage
    tactical_edge = (
        features["press_vs_block"] +
        features["counter_vs_possession"] +
        features["possession_diff"] * 0.05 +
        features["pressing_diff"] * 0.03
    )
    features["tactical_advantage"] = round(tactical_edge, 3)

    return features


def compute_team_style_from_stats(
    df: pd.DataFrame,
    team: str,
    team_col: str = "team",
    window: int = 5,
) -> Optional[TacticalProfile]:
    """Compute team's tactical style from rolling match statistics.

    Args:
        df: DataFrame with team match statistics
        team: Team name
        team_col: Column containing team name
        window: Number of recent matches to consider

    Returns:
        TacticalProfile or None if insufficient data
    """
    team_data = df[df[team_col] == team].tail(window)

    if len(team_data) < 3:
        return None

    # Extract relevant statistics (use available columns)
    possession = team_data.get("possession", team_data.get("possession_pct", pd.Series([50]))).mean()
    passes = team_data.get("passes", team_data.get("total_passes", pd.Series([400]))).mean()
    prog_passes = team_data.get("progressive_passes", pd.Series([10])).mean()
    tackles = team_data.get("tackles", pd.Series([15])).mean()
    interceptions = team_data.get("interceptions", pd.Series([10])).mean()
    crosses = team_data.get("crosses", pd.Series([15])).mean()
    long_balls = team_data.get("long_balls", pd.Series([40])).mean()
    short_passes = passes * 0.7  # Estimate short passes

    profile = classify_playing_style(
        possession=possession,
        passes_per_90=passes,
        progressive_passes=prog_passes,
        tackles=tackles,
        interceptions=interceptions,
        crosses=crosses,
        long_balls=long_balls,
        short_passes=short_passes,
    )
    profile.team = team

    return profile


def add_tactical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add tactical style features to match DataFrame.

    Adds:
    - home_possession_style, away_possession_style
    - home_pressing_intensity, away_pressing_intensity
    - home_directness, away_directness
    - style_similarity, tactical_advantage
    - style_draw_factor
    """
    df = df.copy()

    log.info("Adding tactical features...")

    # Initialize columns
    tactical_cols = [
        "home_possession_style", "away_possession_style",
        "home_pressing_intensity", "away_pressing_intensity",
        "home_directness", "away_directness",
        "style_similarity", "tactical_advantage", "style_draw_factor",
        "possession_diff", "pressing_diff", "directness_diff",
    ]

    for col in tactical_cols:
        df[col] = 0.5 if "style" in col or "intensity" in col or "directness" in col else 0.0

    # For each match, compute styles from rolling stats
    for idx, row in df.iterrows():
        # Use available rolling statistics to infer style
        # Home team style
        home_poss = row.get("home_roll_possession_5", row.get("home_possession_pct", 50))
        home_passes = row.get("home_roll_passes_5", 400)
        home_prog = row.get("home_roll_progressive_passes_5", 10)
        home_tackles = row.get("home_roll_tackles_5", 15)
        home_ints = row.get("home_roll_interceptions_5", 10)

        # Away team style
        away_poss = row.get("away_roll_possession_5", row.get("away_possession_pct", 50))
        away_passes = row.get("away_roll_passes_5", 400)
        away_prog = row.get("away_roll_progressive_passes_5", 10)
        away_tackles = row.get("away_roll_tackles_5", 15)
        away_ints = row.get("away_roll_interceptions_5", 10)

        # Simple style inference from available data
        # Possession style
        home_poss_style = min(1.0, max(0.0, (home_poss - 40) / 20)) if pd.notna(home_poss) else 0.5
        away_poss_style = min(1.0, max(0.0, (away_poss - 40) / 20)) if pd.notna(away_poss) else 0.5

        # Pressing intensity
        home_press = (home_tackles + home_ints) if pd.notna(home_tackles) else 25
        away_press = (away_tackles + away_ints) if pd.notna(away_tackles) else 25
        home_press_style = min(1.0, max(0.0, (home_press - 15) / 15))
        away_press_style = min(1.0, max(0.0, (away_press - 15) / 15))

        # Directness (inverse of possession + progressive)
        home_direct = 1 - home_poss_style * 0.5 - min(1.0, home_prog / 20) * 0.5 if pd.notna(home_prog) else 0.5
        away_direct = 1 - away_poss_style * 0.5 - min(1.0, away_prog / 20) * 0.5 if pd.notna(away_prog) else 0.5

        # Set values
        df.at[idx, "home_possession_style"] = round(home_poss_style, 3)
        df.at[idx, "away_possession_style"] = round(away_poss_style, 3)
        df.at[idx, "home_pressing_intensity"] = round(home_press_style, 3)
        df.at[idx, "away_pressing_intensity"] = round(away_press_style, 3)
        df.at[idx, "home_directness"] = round(home_direct, 3)
        df.at[idx, "away_directness"] = round(away_direct, 3)

        # Differences
        df.at[idx, "possession_diff"] = round(home_poss_style - away_poss_style, 3)
        df.at[idx, "pressing_diff"] = round(home_press_style - away_press_style, 3)
        df.at[idx, "directness_diff"] = round(home_direct - away_direct, 3)

        # Style similarity
        similarity = 1 - (
            abs(home_poss_style - away_poss_style) +
            abs(home_press_style - away_press_style) +
            abs(home_direct - away_direct)
        ) / 3
        df.at[idx, "style_similarity"] = round(similarity, 3)

        # Draw factor (similar styles = more draws)
        df.at[idx, "style_draw_factor"] = round(similarity * 0.1, 3)

        # Tactical advantage
        tactical_edge = (
            (home_poss_style - away_poss_style) * 0.05 +
            (home_press_style - away_press_style) * 0.03
        )
        df.at[idx, "tactical_advantage"] = round(tactical_edge, 3)

    log.info(f"Added {len(tactical_cols)} tactical features")

    return df


def calculate_tactical_mismatch_score(
    home_profile: dict,
    away_profile: dict,
) -> dict:
    """Calculate expected-goals-impact from tactical mismatches.

    Key patterns:
    - High press vs possession → higher expected goals (both sides create chances)
    - Low block vs direct → lower expected goals (few clear chances)
    - Counter vs high press → counter can exploit gaps behind the press

    Args:
        home_profile: dict with pressing_intensity, possession_style, directness, defensive_solidity
        away_profile: same structure

    Returns:
        dict with xg_modifier (positive = more goals), mismatch_type, and draw_modifier
    """
    h_press = home_profile.get("pressing_intensity", 0.5)
    a_press = away_profile.get("pressing_intensity", 0.5)
    h_poss = home_profile.get("possession_style", 0.5)
    a_poss = away_profile.get("possession_style", 0.5)
    h_direct = home_profile.get("directness", 0.5)
    a_direct = away_profile.get("directness", 0.5)
    h_def = home_profile.get("defensive_solidity", 0.5)
    a_def = away_profile.get("defensive_solidity", 0.5)

    xg_modifier = 0.0
    draw_modifier = 0.0
    mismatch_type = "neutral"

    # High press vs possession → open game, more goals
    if h_press > 0.7 and a_poss > 0.7:
        xg_modifier += 0.15
        mismatch_type = "press_vs_possession"
    elif a_press > 0.7 and h_poss > 0.7:
        xg_modifier += 0.15
        mismatch_type = "press_vs_possession"

    # Low block vs direct → fewer clear chances
    if h_def > 0.7 and a_direct > 0.7:
        xg_modifier -= 0.10
        draw_modifier += 0.03
        mismatch_type = "block_vs_direct"
    elif a_def > 0.7 and h_direct > 0.7:
        xg_modifier -= 0.10
        draw_modifier += 0.03
        mismatch_type = "block_vs_direct"

    # Both high pressing → chaotic, high-scoring
    if h_press > 0.7 and a_press > 0.7:
        xg_modifier += 0.10
        mismatch_type = "dual_high_press"

    # Both defensive → low-scoring, draw-prone
    if h_def > 0.7 and a_def > 0.7:
        xg_modifier -= 0.15
        draw_modifier += 0.05
        mismatch_type = "dual_defensive"

    # Counter vs high press → counter benefits from space
    if h_direct > 0.7 and a_press > 0.7:
        xg_modifier += 0.08
        mismatch_type = "counter_vs_press"
    elif a_direct > 0.7 and h_press > 0.7:
        xg_modifier += 0.08
        mismatch_type = "counter_vs_press"

    return {
        "xg_modifier": round(xg_modifier, 3),
        "draw_modifier": round(draw_modifier, 3),
        "mismatch_type": mismatch_type,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing tactical module...")
    print("=" * 60)

    # Test style classification
    print("\nStyle Classification Tests:")

    # Possession team (e.g., Man City style)
    profile1 = classify_playing_style(
        possession=65,
        passes_per_90=600,
        progressive_passes=18,
        tackles=15,
        interceptions=10,
        crosses=12,
        long_balls=30,
        short_passes=500,
    )
    print(f"\n  Possession team (65% poss, 600 passes):")
    print(f"    Possession style: {profile1.possession_style}")
    print(f"    Pressing: {profile1.pressing_intensity}")
    print(f"    Directness: {profile1.directness}")

    # Counter-attacking team
    profile2 = classify_playing_style(
        possession=42,
        passes_per_90=350,
        progressive_passes=8,
        tackles=22,
        interceptions=15,
        crosses=18,
        long_balls=60,
        short_passes=250,
    )
    print(f"\n  Counter team (42% poss, high tackles):")
    print(f"    Possession style: {profile2.possession_style}")
    print(f"    Pressing: {profile2.pressing_intensity}")
    print(f"    Directness: {profile2.directness}")

    # Test matchup
    print("\nMatchup Analysis:")
    matchup = compute_style_matchup(profile1, profile2)
    print(f"  Possession vs Counter:")
    for k, v in matchup.items():
        print(f"    {k}: {v}")

    print("\n" + "=" * 60)
