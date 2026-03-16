"""Injury impact model with position-aware weighting and chemistry disruption.

Calculates how injuries affect team performance beyond simple counts:
- Position-specific injury weights (GK most disruptive, FB most replaceable)
- Player importance based on xG/xA contribution share
- Chemistry disruption from partnership breakdowns
- Returning player integration penalties

Data sources:
- scraper/injuries.py -> data/external/injuries/
- data/features/player_xg_profiles.json -> player importance
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import DATA_DIR

log = logging.getLogger(__name__)

# Position-specific weights: how disruptive is losing a player in this role
POSITION_WEIGHTS = {
    "GK": 0.85,   # GK injuries most disruptive (hardest to replace)
    "CB": 0.70,   # Defensive stability at stake
    "DF": 0.65,   # Generic defender
    "FB": 0.50,   # Wing-backs more replaceable
    "LB": 0.50,
    "RB": 0.50,
    "DM": 0.65,   # Defensive midfield — shield role
    "CM": 0.65,   # Midfield control
    "MF": 0.60,   # Generic midfielder
    "AM": 0.60,   # Creative hub
    "LW": 0.55,
    "RW": 0.55,
    "FW": 0.55,   # Can be rotated more easily
    "CF": 0.55,
}

# Partnership units — if 2+ from same unit are out, extra penalty
PARTNERSHIP_UNITS = {
    "cb_pair": ["CB", "DF"],
    "cm_pair": ["CM", "DM", "MF"],
    "fw_pair": ["FW", "CF", "LW", "RW", "AM"],
}


def _load_player_profiles() -> dict:
    """Load player xG profiles from cache."""
    path = DATA_DIR / "features" / "player_xg_profiles.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _get_team_profiles(profiles: dict, team: str) -> list[dict]:
    """Get all player profiles for a team."""
    return [
        p for p in profiles.values()
        if p.get("team", "").lower() == team.lower()
    ]


def _infer_position(player_name: str, team: str, profiles: dict) -> str:
    """Infer a player's position from profile data or default to MF."""
    key = f"{player_name}|{team}"
    profile = profiles.get(key, {})
    return profile.get("position", "MF")


def get_player_importance(player_name: str, team: str, profiles: dict = None) -> float:
    """Calculate how important a player is to their team.

    Uses xG+xA per 90 relative to team total.
    Top scorer = ~1.0, bench player = ~0.1
    """
    if profiles is None:
        profiles = _load_player_profiles()

    key = f"{player_name}|{team}"
    profile = profiles.get(key)
    if not profile:
        # Try fuzzy match by player name only
        for k, p in profiles.items():
            if player_name.lower() in k.lower() and p.get("team", "").lower() == team.lower():
                profile = p
                break

    if not profile:
        return 0.2  # Unknown player — assume squad player

    player_xg_xa = profile.get("xg_per_90", 0) + profile.get("xa_per_90", 0)

    # Team total xG+xA per 90
    team_players = _get_team_profiles(profiles, team)
    team_total = sum(
        p.get("xg_per_90", 0) + p.get("xa_per_90", 0)
        for p in team_players
        if p.get("total_minutes", 0) >= 270  # At least 3 full matches
    )

    if team_total <= 0:
        return 0.2

    importance = player_xg_xa / max(team_total, 0.01)
    # Normalize: top contributor ~0.15 share → importance 1.0
    return min(1.0, importance / 0.15)


def calculate_positional_injury_impact(
    team: str,
    injured_players: list[str],
    profiles: dict = None,
) -> float:
    """Position-specific injury impact — not all absences are equal.

    For each injured player:
      1. Get position weight (GK=0.85, CB=0.70, FW=0.55, etc.)
      2. Get player importance (xG+xA share of team total)
      3. impact += position_weight * player_importance

    Returns: weighted impact score (0.0 - 1.0)
    """
    if not injured_players:
        return 0.0

    if profiles is None:
        profiles = _load_player_profiles()

    total_impact = 0.0
    for player in injured_players:
        position = _infer_position(player, team, profiles)
        pos_weight = POSITION_WEIGHTS.get(position, 0.55)
        importance = get_player_importance(player, team, profiles)
        total_impact += pos_weight * importance

    return min(total_impact, 1.0)


def model_chemistry_disruption(
    team: str,
    injured_players: list[str],
    returning_players: Optional[list[dict]] = None,
    profiles: dict = None,
) -> float:
    """Model disruption from absences AND returning players needing integration.

    - Partnership disruption: if 2+ players from same unit (CB pair, CM pair) are out
    - Returning player integration: reduced effectiveness for recently returned players

    Args:
        team: Team name
        injured_players: List of currently injured player names
        returning_players: List of dicts with {name, games_since_return}
        profiles: Player xG profiles (loaded if not provided)

    Returns:
        disruption score (0.0 - 0.5)
    """
    if profiles is None:
        profiles = _load_player_profiles()

    disruption = 0.0

    # 1. Partnership disruption
    injured_positions = []
    for player in injured_players:
        pos = _infer_position(player, team, profiles)
        injured_positions.append(pos)

    for unit_name, unit_positions in PARTNERSHIP_UNITS.items():
        injured_in_unit = sum(1 for pos in injured_positions if pos in unit_positions)
        if injured_in_unit >= 2:
            disruption += 0.15  # Significant partnership breakdown

    # 2. Returning player integration penalty
    if returning_players:
        for rp in returning_players:
            games_back = rp.get("games_since_return", 0)
            if games_back < 5:
                # Effectiveness: 0.5 at game 1, ramps to 1.0 by game 5
                effectiveness = 0.5 + 0.1 * games_back
                penalty = (1.0 - effectiveness) * 0.05  # Small per-player penalty
                disruption += penalty

    return min(disruption, 0.5)


def classify_key_absence(
    team: str,
    injured_players: list[str],
    profiles: dict = None,
) -> dict:
    """Identify if key attacker, defender, or GK is out.

    Returns dict with bool flags for key absences.
    """
    if profiles is None:
        profiles = _load_player_profiles()

    team_players = _get_team_profiles(profiles, team)
    if not team_players:
        return {
            "key_attacker_out": False,
            "key_defender_out": False,
            "key_gk_out": False,
        }

    injured_lower = [p.lower() for p in injured_players]

    # Find top attacker (highest xG per 90 with >500 minutes)
    attackers = sorted(
        [p for p in team_players if p.get("total_minutes", 0) >= 500],
        key=lambda p: p.get("xg_per_90", 0),
        reverse=True,
    )
    top_attacker = attackers[0]["player_name"].lower() if attackers else ""
    key_attacker_out = top_attacker in injured_lower if top_attacker else False

    # Find key defender (highest minutes among CB/DF/GK)
    defenders = [
        p for p in team_players
        if p.get("position", "MF") in ("CB", "DF", "GK", "FB", "LB", "RB")
        and p.get("total_minutes", 0) >= 500
    ]
    key_defender_out = any(
        d["player_name"].lower() in injured_lower for d in defenders[:2]
    )

    # GK
    gks = [p for p in team_players if p.get("position", "") == "GK"]
    key_gk_out = any(g["player_name"].lower() in injured_lower for g in gks[:1])

    return {
        "key_attacker_out": key_attacker_out,
        "key_defender_out": key_defender_out,
        "key_gk_out": key_gk_out,
    }


def compute_injury_xg_adjustment(
    team: str,
    opponent: str,
    injured_players: list[str],
    profiles: dict = None,
) -> dict:
    """Compute xG adjustments based on injuries.

    Returns adjustments to apply to ensemble xG predictions:
    - team_xg_adj: change to this team's xG (negative = fewer goals)
    - opponent_xg_adj: change to opponent's xG (positive if defense weakened)
    """
    if not injured_players:
        return {"team_xg_adj": 0.0, "opponent_xg_adj": 0.0}

    if profiles is None:
        profiles = _load_player_profiles()

    absences = classify_key_absence(team, injured_players, profiles)

    team_xg_adj = 0.0
    opponent_xg_adj = 0.0

    # Top scorer injured → reduce team xG by their contribution (cap -0.5)
    if absences["key_attacker_out"]:
        team_players = _get_team_profiles(profiles, team)
        attackers = sorted(
            [p for p in team_players if p.get("total_minutes", 0) >= 500],
            key=lambda p: p.get("xg_per_90", 0),
            reverse=True,
        )
        if attackers:
            # 50% replacement quality
            xg_loss = attackers[0].get("xg_per_90", 0) * 0.5
            team_xg_adj -= min(xg_loss, 0.5)

    # GK injured → increase opponent xG by backup GK effect
    if absences["key_gk_out"]:
        opponent_xg_adj += 0.15

    # 2+ defenders out → increase opponent xG
    injured_lower = [p.lower() for p in injured_players]
    team_players = _get_team_profiles(profiles, team)
    def_count = sum(
        1 for p in team_players
        if p.get("position", "MF") in ("CB", "DF", "FB", "LB", "RB")
        and p["player_name"].lower() in injured_lower
    )
    if def_count >= 2:
        opponent_xg_adj += 0.10

    return {
        "team_xg_adj": round(team_xg_adj, 3),
        "opponent_xg_adj": round(opponent_xg_adj, 3),
    }


def add_injury_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Add injury features to the feature DataFrame.

    Adds 10 columns:
    - home_injuries_count / away_injuries_count
    - home_key_attacker_out / away_key_attacker_out
    - home_key_defender_out / away_key_defender_out
    - home_injury_impact / away_injury_impact
    - home_chemistry_disruption / away_chemistry_disruption
    """
    df = feature_df.copy()
    profiles = _load_player_profiles()

    # Load current injuries
    try:
        from scraper.injuries import get_current_injuries
        injuries_df = get_current_injuries()
    except Exception:
        injuries_df = pd.DataFrame()

    # Initialize columns with defaults
    for prefix in ("home", "away"):
        df[f"{prefix}_injuries_count"] = 0
        df[f"{prefix}_key_attacker_out"] = 0
        df[f"{prefix}_key_defender_out"] = 0
        df[f"{prefix}_injury_impact"] = 0.0
        df[f"{prefix}_chemistry_disruption"] = 0.0

    if injuries_df.empty:
        log.info("No injury data available — injury features set to 0")
        return df

    # Build team -> injured players map
    team_injuries: dict[str, list[str]] = {}
    for _, row in injuries_df.iterrows():
        team = row.get("team", "")
        player = row.get("player_name", "")
        if team and player:
            team_injuries.setdefault(team, []).append(player)

    for idx, row in df.iterrows():
        for prefix in ("home", "away"):
            team = row.get(f"{prefix}_team", "")
            injured = team_injuries.get(team, [])

            if not injured:
                continue

            df.at[idx, f"{prefix}_injuries_count"] = len(injured)

            # Key absences
            absences = classify_key_absence(team, injured, profiles)
            df.at[idx, f"{prefix}_key_attacker_out"] = int(absences["key_attacker_out"])
            df.at[idx, f"{prefix}_key_defender_out"] = int(
                absences["key_defender_out"] or absences.get("key_gk_out", False)
            )

            # Positional impact
            impact = calculate_positional_injury_impact(team, injured, profiles)
            df.at[idx, f"{prefix}_injury_impact"] = round(impact, 3)

            # Chemistry disruption
            disruption = model_chemistry_disruption(team, injured, profiles=profiles)
            df.at[idx, f"{prefix}_chemistry_disruption"] = round(disruption, 3)

    n_with_injuries = (df["home_injuries_count"] + df["away_injuries_count"] > 0).sum()
    log.info(f"Added 10 injury features ({n_with_injuries} matches with injury data)")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing injury impact module...")
    print("=" * 60)

    profiles = _load_player_profiles()
    print(f"Loaded {len(profiles)} player profiles")

    # Test with a known team
    test_team = "Napoli"
    team_players = _get_team_profiles(profiles, test_team)
    print(f"\n{test_team}: {len(team_players)} profiled players")

    # Simulate injuries
    if team_players:
        top_players = sorted(
            team_players,
            key=lambda p: p.get("xg_per_90", 0),
            reverse=True,
        )
        test_injured = [p["player_name"] for p in top_players[:2]]
        print(f"Simulated injuries: {test_injured}")

        impact = calculate_positional_injury_impact(test_team, test_injured, profiles)
        print(f"Positional impact: {impact:.3f}")

        disruption = model_chemistry_disruption(test_team, test_injured, profiles=profiles)
        print(f"Chemistry disruption: {disruption:.3f}")

        absences = classify_key_absence(test_team, test_injured, profiles)
        print(f"Key absences: {absences}")

        xg_adj = compute_injury_xg_adjustment(test_team, "Inter", test_injured, profiles)
        print(f"xG adjustments: {xg_adj}")

    print("\n" + "=" * 60)
