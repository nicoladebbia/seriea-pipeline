"""Pressing intensity (PPDA) features from Understat data.

PPDA = Passes Per Defensive Action = opponent passes allowed / defensive actions
Lower PPDA = more aggressive pressing (fewer passes allowed per intervention).

Data source:
- Understat team history JSON: per-match ppda.att / ppda.def
- Covers seasons 2022-2023 through 2025-2026
- Also provides ppda_allowed (pressing the team faces from opponents)

Features added (6):
- home_ppda / away_ppda — season average pressing intensity
- ppda_differential — home_ppda - away_ppda
- pressing_mismatch — 1 if extreme pressing difference
- home_ppda_allowed / away_ppda_allowed — avg pressing faced from opponents
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import DATA_DIR

log = logging.getLogger(__name__)

UNDERSTAT_DIR = DATA_DIR / "external" / "understat"

# Map Understat team names → pipeline names (lowercase)
_UNDERSTAT_TO_PIPELINE = {
    "ac milan": "milan",
    "parma calcio 1913": "parma",
    "hellas verona": "verona",
    "inter": "inter",
    "internazionale": "inter",
}


def _normalize_team(name: str) -> str:
    """Normalize Understat team name to pipeline convention."""
    low = name.lower().strip()
    return _UNDERSTAT_TO_PIPELINE.get(low, low)


def _load_understat_ppda() -> dict[tuple[str, str], dict]:
    """Load per-team per-season PPDA stats from Understat JSON files.

    Returns:
        Dict keyed by (team_normalized, season) with values:
        {ppda_mean, ppda_std, ppda_allowed_mean, n_matches}
    """
    cache: dict[tuple[str, str], dict] = {}

    season_files = sorted(UNDERSTAT_DIR.glob("understat_*.json"))
    if not season_files:
        log.warning("No Understat JSON files found in %s", UNDERSTAT_DIR)
        return cache

    for fpath in season_files:
        # Extract season from filename: understat_2024_2025.json → "2024-2025"
        stem = fpath.stem  # "understat_2024_2025"
        parts = stem.replace("understat_", "").split("_")
        if len(parts) != 2:
            continue
        season = f"{parts[0]}-{parts[1]}"

        try:
            with open(fpath) as f:
                data = json.load(f)
        except Exception as e:
            log.warning("Failed to load %s: %s", fpath, e)
            continue

        # Handle both formats:
        #   Old: {"teams": {"team_id": {"title": ..., "history": [...]}}}
        #   New: [{"h": {"title": ..., "ppda": ...}, "a": {"title": ..., "ppda": ...}}]
        if isinstance(data, dict) and "teams" in data:
            # Old format: team-grouped
            teams = data.get("teams", {})
            for team_data in teams.values():
                team_name = _normalize_team(team_data.get("title", ""))
                if not team_name:
                    continue
                history = team_data.get("history", [])
                ppda_values = []
                ppda_allowed_values = []
                for match in history:
                    ppda_raw = match.get("ppda", {})
                    ppda_allowed_raw = match.get("ppda_allowed", match.get("ppda_Allowed", {}))
                    att = ppda_raw.get("att", 0)
                    def_actions = ppda_raw.get("def", 0)
                    if def_actions > 0:
                        ppda_values.append(att / def_actions)
                    att_a = ppda_allowed_raw.get("att", 0)
                    def_a = ppda_allowed_raw.get("def", 0)
                    if def_a > 0:
                        ppda_allowed_values.append(att_a / def_a)
                if ppda_values:
                    cache[(team_name, season)] = {
                        "ppda_avg": np.mean(ppda_values),
                        "ppda_std": np.std(ppda_values) if len(ppda_values) > 1 else 0,
                        "ppda_allowed_avg": np.mean(ppda_allowed_values) if ppda_allowed_values else 12.0,
                        "n_matches": len(ppda_values),
                    }
            continue  # Skip to next file

        elif isinstance(data, list):
            # New format: list of matches with h/a team data
            team_ppda = defaultdict(list)
            team_ppda_allowed = defaultdict(list)
            for match in data:
                if not match.get("isResult"):
                    continue
                for side in ["h", "a"]:
                    side_data = match.get(side, {})
                    if isinstance(side_data, dict):
                        team_name = _normalize_team(side_data.get("title", ""))
                        if not team_name:
                            continue
                        ppda_raw = side_data.get("ppda", {})
                        if isinstance(ppda_raw, dict):
                            att = ppda_raw.get("att", 0)
                            def_actions = ppda_raw.get("def", 0)
                            if def_actions > 0:
                                team_ppda[(team_name, season)].append(att / def_actions)
                        opp_side = "a" if side == "h" else "h"
                        opp_data = match.get(opp_side, {})
                        if isinstance(opp_data, dict):
                            opp_ppda = opp_data.get("ppda", {})
                            if isinstance(opp_ppda, dict):
                                att_a = opp_ppda.get("att", 0)
                                def_a = opp_ppda.get("def", 0)
                                if def_a > 0:
                                    team_ppda_allowed[(team_name, season)].append(att_a / def_a)

            for key, values in team_ppda.items():
                if values:
                    allowed = team_ppda_allowed.get(key, [])
                    cache[key] = {
                        "ppda_avg": np.mean(values),
                        "ppda_std": np.std(values) if len(values) > 1 else 0,
                        "ppda_allowed_avg": np.mean(allowed) if allowed else 12.0,
                        "n_matches": len(values),
                    }
            continue  # Skip old format block below

        else:
            log.warning("Unknown Understat format in %s: %s", fpath.name, type(data).__name__)
            continue

        # Legacy fallback (should not reach here with new code)
        teams = {}
        ppda_values = []
        ppda_allowed_values = []

        for match in []:  # Dead code, kept for structure
            ppda_raw = match.get("ppda", {})
            ppda_allowed_raw = match.get("ppda_allowed", match.get("ppda_Allowed", {}))

            att = ppda_raw.get("att", 0)
            def_actions = ppda_raw.get("def", 0)
            if def_actions > 0:
                ppda_values.append(att / def_actions)

                att_a = ppda_allowed_raw.get("att", 0)
                def_a = ppda_allowed_raw.get("def", 0)
                if def_a > 0:
                    ppda_allowed_values.append(att_a / def_a)

            if ppda_values:
                cache[(team_name, season)] = {
                    "ppda_mean": float(np.mean(ppda_values)),
                    "ppda_std": float(np.std(ppda_values)),
                    "ppda_allowed_mean": float(np.mean(ppda_allowed_values)) if ppda_allowed_values else 12.0,
                    "n_matches": len(ppda_values),
                }

    log.info("Loaded PPDA data for %d team-season combinations from Understat", len(cache))
    return cache


# Module-level lazy cache
_ppda_cache: dict[tuple[str, str], dict] | None = None


def _get_ppda_cache() -> dict[tuple[str, str], dict]:
    """Get or build the PPDA cache (lazy singleton)."""
    global _ppda_cache
    if _ppda_cache is None:
        _ppda_cache = _load_understat_ppda()
    return _ppda_cache


def calculate_team_ppda(
    team: str,
    season: str,
    matches_df: pd.DataFrame = None,
) -> float:
    """Get PPDA for a team in a season from Understat data.

    Args:
        team: Team name (pipeline convention)
        season: Season string (e.g., "2024-2025")
        matches_df: Unused, kept for API compatibility

    Returns:
        PPDA value (5-25 range, lower = more aggressive pressing).
        Returns 12.0 (league average) if no data available.
    """
    cache = _get_ppda_cache()
    key = (team.lower().strip(), season)
    entry = cache.get(key)
    if entry:
        return entry["ppda_mean"]

    # Try common name variants
    for variant in [team.lower(), _normalize_team(team)]:
        key = (variant, season)
        entry = cache.get(key)
        if entry:
            return entry["ppda_mean"]

    return 12.0  # League average default


def add_ppda_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Add PPDA/pressing features to match-level DataFrame.

    Adds 6 columns:
    - home_ppda / away_ppda — season average pressing intensity
    - ppda_differential — home_ppda - away_ppda
    - pressing_mismatch — 1 if one team PPDA <8 and other >16
    - home_ppda_allowed / away_ppda_allowed — avg pressing faced
    """
    df = feature_df.copy()
    cache = _get_ppda_cache()

    if not cache:
        log.warning("No Understat PPDA data available — features set to defaults")
        df["home_ppda"] = 12.0
        df["away_ppda"] = 12.0
        df["ppda_differential"] = 0.0
        df["pressing_mismatch"] = 0
        df["home_ppda_allowed"] = 12.0
        df["away_ppda_allowed"] = 12.0
        return df

    # Vectorized approach: build lookup, apply via map
    home_ppda = []
    away_ppda = []
    home_ppda_allowed = []
    away_ppda_allowed = []

    for _, row in df.iterrows():
        season = row.get("season", "2024-2025")

        for prefix, ppda_list, ppda_allowed_list in [
            ("home", home_ppda, home_ppda_allowed),
            ("away", away_ppda, away_ppda_allowed),
        ]:
            team = str(row.get(f"{prefix}_team", "")).lower().strip()
            team_norm = _normalize_team(team) if team else team

            entry = cache.get((team_norm, season)) or cache.get((team, season))
            if entry:
                ppda_list.append(entry["ppda_mean"])
                ppda_allowed_list.append(entry["ppda_allowed_mean"])
            else:
                ppda_list.append(12.0)
                ppda_allowed_list.append(12.0)

    df["home_ppda"] = home_ppda
    df["away_ppda"] = away_ppda
    df["home_ppda_allowed"] = home_ppda_allowed
    df["away_ppda_allowed"] = away_ppda_allowed

    df["ppda_differential"] = (df["home_ppda"] - df["away_ppda"]).round(2)
    df["pressing_mismatch"] = (
        ((df["home_ppda"] < 8) & (df["away_ppda"] > 16)) |
        ((df["away_ppda"] < 8) & (df["home_ppda"] > 16))
    ).astype(int)

    n_with_data = ((df["home_ppda"] != 12.0) | (df["away_ppda"] != 12.0)).sum()
    log.info(
        "Added 6 PPDA features from Understat (%d/%d matches with real data, "
        "%d team-season combos cached)",
        n_with_data, len(df), len(cache),
    )

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Understat PPDA module...")
    print("=" * 60)

    cache = _get_ppda_cache()
    print(f"Loaded {len(cache)} team-season PPDA entries\n")

    # Show sample data
    for season in ["2024-2025", "2025-2026"]:
        print(f"\n{season}:")
        season_entries = {k: v for k, v in cache.items() if k[1] == season}
        for (team, _), stats in sorted(season_entries.items()):
            print(f"  {team:20s}  PPDA={stats['ppda_mean']:.1f} "
                  f"(±{stats['ppda_std']:.1f})  "
                  f"allowed={stats['ppda_allowed_mean']:.1f}  "
                  f"({stats['n_matches']} matches)")

    print("\n" + "=" * 60)
