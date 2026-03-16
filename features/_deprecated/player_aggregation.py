"""Aggregate player-level stats into team-level prediction features.

This module creates features based on individual player performance metrics,
aggregated to the team level for match prediction.

Key insight: A team's expected performance depends on WHO plays, not just
the team's historical average. If Vlahović (xG=12) is injured, Juventus's
expected goals drop significantly.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.team_names import normalize_team
from storage.paths import PARSED_DIR

log = logging.getLogger(__name__)

PLAYER_STATS_PATH = PARSED_DIR / "player_stats.parquet"


@dataclass
class PlayerForm:
    """Rolling form metrics for a single player."""
    player: str
    team: str
    position: str
    matches_played: int
    total_minutes: int

    # Offensive metrics (per 90 minutes)
    xg_per_90: float
    goals_per_90: float
    shots_per_90: float
    touches_per_90: float

    # Creative metrics (per 90)
    xg_assist_per_90: float
    assists_per_90: float
    progressive_passes_per_90: float
    progressive_carries_per_90: float
    sca_per_90: float  # shot-creating actions

    # Defensive metrics (per 90)
    tackles_per_90: float
    interceptions_per_90: float
    blocks_per_90: float

    # Raw totals for the window
    total_xg: float
    total_goals: float
    total_assists: float


def load_player_stats() -> pd.DataFrame:
    """Load and prepare player stats data."""
    if not PLAYER_STATS_PATH.exists():
        log.warning("Player stats file not found")
        return pd.DataFrame()

    df = pd.read_parquet(PLAYER_STATS_PATH)

    # Convert numeric columns
    numeric_cols = [
        'minutes', 'goals', 'assists', 'xg', 'xg_assist', 'shots',
        'shots_on_target', 'touches', 'passes', 'passes_completed',
        'progressive_passes', 'progressive_carries', 'tackles',
        'interceptions', 'blocks', 'take_ons', 'take_ons_won', 'sca', 'gca'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Normalize team names
    if 'team' in df.columns:
        df['team'] = df['team'].apply(normalize_team)

    # Extract match date from match_id if not present
    if 'match_date' not in df.columns and 'match_id' in df.columns:
        # match_id format: "2024-09-15_Team1_Team2"
        df['match_date'] = df['match_id'].str[:10]

    df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')

    return df


def get_player_rolling_form(
    player_name: str,
    team: str,
    before_date: date,
    n_matches: int = 5,
    player_stats_df: Optional[pd.DataFrame] = None,
) -> Optional[PlayerForm]:
    """Calculate rolling form metrics for a specific player.

    Args:
        player_name: Player's name
        team: Team name
        before_date: Calculate form using matches BEFORE this date
        n_matches: Number of recent matches to consider
        player_stats_df: Pre-loaded player stats (optional)

    Returns:
        PlayerForm object with rolling metrics, or None if no data
    """
    if player_stats_df is None:
        player_stats_df = load_player_stats()

    if player_stats_df.empty:
        return None

    team = normalize_team(team)
    before_dt = pd.to_datetime(before_date)

    # Filter to player's matches before the date
    mask = (
        (player_stats_df['player'].str.lower() == player_name.lower()) &
        (player_stats_df['team'] == team) &
        (player_stats_df['match_date'] < before_dt)
    )

    player_matches = player_stats_df[mask].sort_values('match_date', ascending=False)

    if player_matches.empty:
        # Try fuzzy match on player name
        mask = (
            (player_stats_df['player'].str.contains(player_name.split()[-1], case=False, na=False)) &
            (player_stats_df['team'] == team) &
            (player_stats_df['match_date'] < before_dt)
        )
        player_matches = player_stats_df[mask].sort_values('match_date', ascending=False)

    if player_matches.empty:
        return None

    # Take last N matches
    recent = player_matches.head(n_matches)

    total_minutes = recent['minutes'].sum()
    if total_minutes < 45:  # Minimum 45 minutes to calculate form
        return None

    # Calculate per-90 metrics
    ninety_mins = total_minutes / 90

    return PlayerForm(
        player=player_name,
        team=team,
        position=recent.iloc[0].get('position', 'Unknown'),
        matches_played=len(recent),
        total_minutes=int(total_minutes),

        # Offensive
        xg_per_90=recent['xg'].sum() / ninety_mins if ninety_mins > 0 else 0,
        goals_per_90=recent['goals'].sum() / ninety_mins if ninety_mins > 0 else 0,
        shots_per_90=recent['shots'].sum() / ninety_mins if ninety_mins > 0 else 0,
        touches_per_90=recent['touches'].sum() / ninety_mins if ninety_mins > 0 else 0,

        # Creative
        xg_assist_per_90=recent['xg_assist'].sum() / ninety_mins if ninety_mins > 0 else 0,
        assists_per_90=recent['assists'].sum() / ninety_mins if ninety_mins > 0 else 0,
        progressive_passes_per_90=recent['progressive_passes'].sum() / ninety_mins if ninety_mins > 0 else 0,
        progressive_carries_per_90=recent['progressive_carries'].sum() / ninety_mins if ninety_mins > 0 else 0,
        sca_per_90=recent['sca'].sum() / ninety_mins if 'sca' in recent.columns and ninety_mins > 0 else 0,

        # Defensive
        tackles_per_90=recent['tackles'].sum() / ninety_mins if ninety_mins > 0 else 0,
        interceptions_per_90=recent['interceptions'].sum() / ninety_mins if ninety_mins > 0 else 0,
        blocks_per_90=recent['blocks'].sum() / ninety_mins if ninety_mins > 0 else 0,

        # Totals
        total_xg=recent['xg'].sum(),
        total_goals=recent['goals'].sum(),
        total_assists=recent['assists'].sum(),
    )


def get_team_player_forms(
    team: str,
    before_date: date,
    n_matches: int = 5,
) -> list[PlayerForm]:
    """Get rolling form for all players in a team's recent matches.

    Returns list of PlayerForm objects for players who played recently.
    """
    player_stats_df = load_player_stats()
    if player_stats_df.empty:
        return []

    team = normalize_team(team)
    before_dt = pd.to_datetime(before_date)

    # Get unique players who played for this team recently
    mask = (
        (player_stats_df['team'] == team) &
        (player_stats_df['match_date'] < before_dt)
    )

    team_data = player_stats_df[mask]
    if team_data.empty:
        return []

    # Get recent matches to find active players
    recent_matches = team_data.sort_values('match_date', ascending=False)['match_id'].unique()[:n_matches]
    active_players = team_data[team_data['match_id'].isin(recent_matches)]['player'].unique()

    forms = []
    for player_name in active_players:
        form = get_player_rolling_form(player_name, team, before_date, n_matches, player_stats_df)
        if form is not None:
            forms.append(form)

    return forms


def aggregate_team_features(
    team: str,
    before_date: date,
    predicted_starters: Optional[list[str]] = None,
    n_matches: int = 5,
) -> dict[str, float]:
    """Aggregate player-level stats into team prediction features.

    Args:
        team: Team name
        before_date: Calculate features for match on this date
        predicted_starters: List of predicted starter names (if available)
        n_matches: Rolling window size

    Returns:
        Dictionary of aggregated features
    """
    forms = get_team_player_forms(team, before_date, n_matches)

    if not forms:
        # Return default values if no player data
        return {
            'player_total_xg': 0,
            'player_total_xga': 0,
            'player_avg_xg_per_90': 0,
            'player_avg_touches': 0,
            'player_creative_output': 0,
            'player_defensive_output': 0,
            'player_top_scorer_xg': 0,
            'player_data_available': 0,
        }

    # If we have predicted starters, filter to them
    if predicted_starters:
        starter_names_lower = [s.lower() for s in predicted_starters]
        starter_forms = [f for f in forms if f.player.lower() in starter_names_lower]

        # If we found at least half the starters, use that subset
        if len(starter_forms) >= 5:
            forms = starter_forms

    # Sort by xG (top performers)
    forms_by_xg = sorted(forms, key=lambda f: f.xg_per_90, reverse=True)

    # Aggregate features
    total_xg = sum(f.total_xg for f in forms)
    total_xga = sum(f.total_xg + f.xg_assist_per_90 * (f.total_minutes / 90) for f in forms)

    avg_xg = np.mean([f.xg_per_90 for f in forms]) if forms else 0
    avg_touches = np.mean([f.touches_per_90 for f in forms]) if forms else 0

    # Creative output (progressive actions + assists)
    creative = sum(
        f.progressive_passes_per_90 + f.progressive_carries_per_90 + f.sca_per_90
        for f in forms
    ) / len(forms) if forms else 0

    # Defensive output
    defensive = sum(
        f.tackles_per_90 + f.interceptions_per_90 + f.blocks_per_90
        for f in forms
    ) / len(forms) if forms else 0

    # Top scorer contribution (key player dependency)
    top_scorer_xg = forms_by_xg[0].xg_per_90 if forms_by_xg else 0

    return {
        'player_total_xg': round(total_xg, 2),
        'player_total_xga': round(total_xga, 2),
        'player_avg_xg_per_90': round(avg_xg, 3),
        'player_avg_touches': round(avg_touches, 1),
        'player_creative_output': round(creative, 2),
        'player_defensive_output': round(defensive, 2),
        'player_top_scorer_xg': round(top_scorer_xg, 3),
        'player_data_available': 1,
    }


def build_match_player_features(
    home_team: str,
    away_team: str,
    match_date: date,
    home_starters: Optional[list[str]] = None,
    away_starters: Optional[list[str]] = None,
    n_matches: int = 5,
) -> dict[str, float]:
    """Build player-aggregated features for a match prediction.

    This is the main entry point for the prediction pipeline.

    Returns:
        Dictionary with home_ and away_ prefixed player features
    """
    home_features = aggregate_team_features(home_team, match_date, home_starters, n_matches)
    away_features = aggregate_team_features(away_team, match_date, away_starters, n_matches)

    result = {}

    # Add home features
    for key, value in home_features.items():
        result[f'home_{key}'] = value

    # Add away features
    for key, value in away_features.items():
        result[f'away_{key}'] = value

    # Add differential features
    result['player_xg_diff'] = home_features['player_total_xg'] - away_features['player_total_xg']
    result['player_creative_diff'] = home_features['player_creative_output'] - away_features['player_creative_output']
    result['player_defensive_diff'] = home_features['player_defensive_output'] - away_features['player_defensive_output']

    return result


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing player aggregation features...")
    print("=" * 60)

    # Test for Juventus vs Lazio (MW24)
    match_date = date(2026, 2, 8)

    print(f"\nJuventus vs Lazio on {match_date}")
    print("-" * 40)

    features = build_match_player_features(
        home_team="Juventus",
        away_team="Lazio",
        match_date=match_date,
    )

    print("\nGenerated features:")
    for key, value in sorted(features.items()):
        print(f"  {key}: {value}")

    # Show individual player forms for Juventus
    print("\n" + "=" * 60)
    print("Juventus player forms:")
    print("-" * 40)

    juve_forms = get_team_player_forms("Juventus", match_date, n_matches=5)
    juve_forms = sorted(juve_forms, key=lambda f: f.xg_per_90, reverse=True)

    for f in juve_forms[:10]:
        print(f"  {f.player:25} xG/90: {f.xg_per_90:.2f}  touches: {f.touches_per_90:.0f}  prog.passes: {f.progressive_passes_per_90:.1f}")
