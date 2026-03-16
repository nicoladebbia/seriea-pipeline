"""Captain features — team performance with/without captain.

Uses scraped captain data from Sofascore to compute:
  - captain_played          : 1 if team captain started this match
  - captain_consistency     : % of recent matches captain started (rolling 5)
  - captain_win_rate_diff   : team win rate with captain - without captain

All features computed per team per match with shift(1) leak prevention,
merged to match level as home_*/away_* columns.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CAPTAINS_PATH = _PROJECT_ROOT / "data" / "external" / "sofascore" / "captains.parquet"
_PMS_PATH = _PROJECT_ROOT / "data" / "external" / "sofascore" / "player_match_stats.parquet"

# Sofascore team name -> pipeline team name
_TEAM_MAP = {
    "ac milan": "milan", "parma calcio 1913": "parma", "spal 2013": "spal",
    "hellas verona": "verona", "internazionale": "inter", "chievoverona": "chievo",
}


def _norm_team(name: str) -> str:
    if pd.isna(name):
        return ""
    low = name.lower().strip()
    return _TEAM_MAP.get(low, low).title()


def _build_id_bridge() -> dict:
    """Build mapping: Sofascore int match_id -> pipeline string match_id.

    Pipeline string IDs have format 'YYYY-MM-DD_HomeTeam_AwayTeam'.
    Sofascore IDs are integers from player_match_stats.parquet.
    """
    if not _PMS_PATH.exists():
        return {}
    pms = pd.read_parquet(_PMS_PATH, columns=["match_id", "date", "home_team", "away_team"])
    pms = pms.drop_duplicates("match_id")
    bridge = {}
    for _, row in pms.iterrows():
        home = _norm_team(row["home_team"])
        away = _norm_team(row["away_team"])
        str_id = f"{row['date']}_{home}_{away}"
        bridge[row["match_id"]] = str_id
    log.debug("Captain ID bridge: %d Sofascore -> pipeline mappings", len(bridge))
    return bridge


def _load_captains() -> pd.DataFrame:
    """Load captain data from scraped parquet."""
    if not _CAPTAINS_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(_CAPTAINS_PATH)
    except Exception:
        return pd.DataFrame()


def add_captain_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Add captain-related features to match data.

    Adds ~6 columns (3 per side): captain_played, captain_consistency, captain_effect.
    """
    captains = _load_captains()
    if captains.empty:
        log.warning("No captain data available; skipping captain features")
        return matches

    df = matches.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.sort_values("match_date").reset_index(drop=True)

    log.info("Computing captain features from %d captain records...", len(captains))

    # Bridge Sofascore int IDs -> pipeline string IDs
    id_bridge = _build_id_bridge()
    pipeline_ids = set(df["match_id"].unique())

    # Build lookup: (pipeline_str_match_id, is_home) -> captain_name
    cap_lookup = {}
    mapped = 0
    for _, row in captains.iterrows():
        str_id = id_bridge.get(row["match_id"])
        if str_id and str_id in pipeline_ids:
            cap_lookup[(str_id, row["is_home"])] = row["captain_name"]
            mapped += 1
    log.info("Captain lookup: %d/%d records mapped to pipeline IDs", mapped, len(captains))

    # Per-team tracking
    team_stats: dict[str, dict] = {}

    result_cols = {k: [] for k in [
        "home_captain_played", "away_captain_played",
        "home_captain_consistency", "away_captain_consistency",
        "home_captain_effect", "away_captain_effect",
    ]}

    for _, row in df.iterrows():
        mid = row.get("match_id")  # pipeline string ID
        hs = row.get("home_score")
        as_ = row.get("away_score")

        for team_col, is_home, prefix in [
            ("home_team", True, "home"),
            ("away_team", False, "away"),
        ]:
            team = row.get(team_col, "")
            if not team or pd.isna(team):
                result_cols[f"{prefix}_captain_played"].append(None)
                result_cols[f"{prefix}_captain_consistency"].append(None)
                result_cols[f"{prefix}_captain_effect"].append(None)
                continue

            captain_name = cap_lookup.get((mid, is_home))
            stats = team_stats.get(team)

            if stats and stats["matches"] >= 5 and stats.get("current_captain"):
                # Captain played = did the known captain play this match?
                played = 1 if captain_name == stats["current_captain"] else 0
                result_cols[f"{prefix}_captain_played"].append(played)

                # Captain consistency: how often captain played in last 5
                recent = stats["captain_played_history"][-5:]
                result_cols[f"{prefix}_captain_consistency"].append(
                    round(sum(recent) / len(recent), 3) if recent else None
                )

                # Captain effect: win rate with captain - without
                with_cap = stats["wins_with_captain"]
                matches_with = stats["matches_with_captain"]
                without_cap = stats["wins_without_captain"]
                matches_without = stats["matches_without_captain"]

                if matches_with >= 3 and matches_without >= 2:
                    wr_with = with_cap / matches_with
                    wr_without = without_cap / matches_without
                    result_cols[f"{prefix}_captain_effect"].append(
                        round(wr_with - wr_without, 3)
                    )
                else:
                    result_cols[f"{prefix}_captain_effect"].append(None)
            else:
                result_cols[f"{prefix}_captain_played"].append(None)
                result_cols[f"{prefix}_captain_consistency"].append(None)
                result_cols[f"{prefix}_captain_effect"].append(None)

            # --- Update stats WITH this match ---
            if captain_name:
                if team not in team_stats:
                    team_stats[team] = {
                        "matches": 0, "current_captain": None,
                        "captain_played_history": [],
                        "wins_with_captain": 0, "matches_with_captain": 0,
                        "wins_without_captain": 0, "matches_without_captain": 0,
                    }

                s = team_stats[team]
                s["matches"] += 1

                # Determine if captain played (captain_name matches known captain)
                if s["current_captain"] is None:
                    s["current_captain"] = captain_name

                cap_played = captain_name == s["current_captain"]
                s["captain_played_history"].append(1 if cap_played else 0)

                # Track win rate with/without captain
                if not pd.isna(hs) and not pd.isna(as_):
                    gf = int(hs) if is_home else int(as_)
                    ga = int(as_) if is_home else int(hs)
                    won = gf > ga

                    if cap_played:
                        s["matches_with_captain"] += 1
                        if won:
                            s["wins_with_captain"] += 1
                    else:
                        s["matches_without_captain"] += 1
                        if won:
                            s["wins_without_captain"] += 1

                # Update current captain (most recent captain is the current one)
                s["current_captain"] = captain_name

    for col, vals in result_cols.items():
        df[col] = vals

    n = df["home_captain_played"].notna().sum()
    log.info("Added 6 captain features to %d matches", n)

    # Fill NaN with 0 for pre-Sofascore matches (no data = neutral signal)
    cap_cols = [c for c in df.columns if "captain_" in c]
    df[cap_cols] = df[cap_cols].fillna(0)

    return df
