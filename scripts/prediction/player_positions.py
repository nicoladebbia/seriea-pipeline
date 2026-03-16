#!/usr/bin/env python3
"""Extract detailed tactical positions from Sofascore match lineup data.

Sofascore match JSONs list starters in formation order: index 0 = GK, then
defense R→L, midfield R→L, attack R→L. Combined with the formation string
and FORMATION_TACTICAL_SLOTS, we can derive each player's exact tactical
position (CB, RB, LWB, CM, CAM, RW, ST, etc.) for every match they played.

Aggregating across all season matches gives each player their primary and
secondary positions, enabling the lineup predictor to assign players to the
correct formation slot instead of using generic DEF/MID/FWD labels.

Output: dict mapping player_id → {
    "name": str,
    "primary_position": str,      # Most frequent tactical position (e.g., "RCB")
    "position_counts": dict,      # {"RCB": 15, "CB": 3, "RB": 1}
    "matches_played": int,
    "sofascore_position": str,    # Original G/D/M/F
}
"""

import json
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.prediction.lineup_predictor import FORMATION_TACTICAL_SLOTS, _get_tactical_labels


def _formation_to_slots(formation: str) -> dict:
    """Parse formation string into position group slot counts."""
    try:
        parts = [int(x) for x in formation.split("-")]
    except (ValueError, AttributeError):
        return {"G": 1, "D": 4, "M": 3, "F": 3}

    slots = {"G": 1}
    if len(parts) == 3:
        slots["D"] = parts[0]
        slots["M"] = parts[1]
        slots["F"] = parts[2]
    elif len(parts) == 4:
        slots["D"] = parts[0]
        slots["M"] = parts[1] + parts[2]
        slots["F"] = parts[3]
    elif len(parts) == 5:
        slots["D"] = parts[0]
        slots["M"] = parts[1] + parts[2] + parts[3]
        slots["F"] = parts[4]
    else:
        slots["D"] = 4
        slots["M"] = 3
        slots["F"] = 3
    return slots


def _formation_index_to_label(formation: str) -> list:
    """Convert a formation string to an ordered list of 11 tactical labels.

    Index 0 = GK, then defense slots, midfield slots, attack slots.
    This matches Sofascore's starters list ordering.
    """
    slots = _formation_to_slots(formation)
    tactical = _get_tactical_labels(formation, slots)

    # Build ordered list: GK first, then D, M, F
    labels = []
    for group in ["G", "D", "M", "F"]:
        group_labels = tactical.get(group, [])
        # Number of slots for this group
        n = slots.get(group, 0)
        for i in range(n):
            if i < len(group_labels):
                labels.append(group_labels[i])
            else:
                from scripts.prediction.lineup_predictor import POSITION_MAP
                labels.append(POSITION_MAP.get(group, group))
    return labels


def extract_positions_from_matches(match_dir: str | Path,
                                   seasons: list = None) -> dict:
    """Parse Sofascore match JSONs and extract detailed positions for all players.

    Args:
        match_dir: Path to data/external/sofascore/matches/
        seasons: List of season dirs to process (e.g., ["2025-2026"]).
                 If None, uses current season only.

    Returns:
        Dict mapping player_id (int) → position info dict.
    """
    match_dir = Path(match_dir)
    if seasons is None:
        # Use most recent season
        season_dirs = sorted(match_dir.iterdir())
        seasons = [season_dirs[-1].name] if season_dirs else []

    # player_id → Counter of tactical positions
    player_positions = defaultdict(Counter)
    player_info = {}  # player_id → {name, sofascore_position}
    matches_processed = 0

    for season in seasons:
        pattern = str(match_dir / season / "*.json")
        files = glob.glob(pattern)
        for filepath in files:
            try:
                with open(filepath) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            for side in ["home_lineup", "away_lineup"]:
                lineup = data.get(side, {})
                formation = lineup.get("formation", "")
                starters = lineup.get("starters", [])

                if not formation or len(starters) != 11:
                    continue

                # Get ordered tactical labels for this formation
                labels = _formation_index_to_label(formation)
                if len(labels) != 11:
                    continue

                for idx, starter in enumerate(starters):
                    player = starter.get("player", {})
                    pid = player.get("id")
                    if not pid:
                        continue

                    tactical_label = labels[idx]
                    player_positions[pid][tactical_label] += 1

                    # Store latest info
                    player_info[pid] = {
                        "name": player.get("name", ""),
                        "short_name": player.get("shortName", ""),
                        "sofascore_position": player.get("position", ""),
                    }

            matches_processed += 1

    # Build output
    result = {}
    for pid, counts in player_positions.items():
        info = player_info.get(pid, {})
        primary = counts.most_common(1)[0][0] if counts else "?"
        result[pid] = {
            "name": info.get("name", ""),
            "short_name": info.get("short_name", ""),
            "primary_position": primary,
            "position_counts": dict(counts),
            "matches_played": sum(counts.values()),
            "sofascore_position": info.get("sofascore_position", ""),
        }

    print(f"Extracted positions for {len(result)} players from {matches_processed} matches")
    return result


def build_player_position_map(match_dir: str | Path = None,
                              seasons: list = None) -> dict:
    """High-level API: build and return the player position map.

    Uses current season by default. Can include historical seasons
    for players with few current-season appearances.
    """
    if match_dir is None:
        from config.settings import DATA_DIR
        match_dir = DATA_DIR / ".." / "data" / "external" / "sofascore" / "matches"

    return extract_positions_from_matches(match_dir, seasons=seasons)


if __name__ == "__main__":
    positions = build_player_position_map()

    # Show some examples
    print("\n=== Sample Players ===")
    # Sort by matches played
    by_matches = sorted(positions.values(), key=lambda x: x["matches_played"], reverse=True)
    for p in by_matches[:30]:
        counts_str = ", ".join(f"{k}:{v}" for k, v in
                               sorted(p["position_counts"].items(), key=lambda x: -x[1]))
        print(f"  {p['name']:.<30} {p['primary_position']:>5} ({p['matches_played']:>2} apps) [{counts_str}]")
