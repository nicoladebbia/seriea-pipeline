"""Manager tenure features and head-to-head records.

Tracks how long a manager has been in charge, flags new appointments,
and computes manager vs manager head-to-head win rates.
Manager names are available from FBref-scraped data (2024-2025) but not
from football-data.co.uk historical imports. When manager data is missing,
features are left as NaN.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


def get_manager_h2h(
    home_manager: str,
    away_manager: str,
    matches_df: pd.DataFrame,
) -> dict:
    """Compute manager head-to-head record from historical matches.

    Scans all historical matches where both managers were in charge of their
    respective teams and computes win rates.

    Args:
        home_manager: Current home team manager name
        away_manager: Current away team manager name
        matches_df: DataFrame with home_manager/away_manager columns

    Returns:
        Dict with wins_home, wins_away, draws, total, home_winrate, confidence
    """
    if not home_manager or not away_manager:
        return {
            "wins_home": 0, "wins_away": 0, "draws": 0, "total": 0,
            "home_winrate": 0.5, "confidence": "none",
        }

    hm = str(home_manager).strip().lower()
    am = str(away_manager).strip().lower()

    if "home_manager" not in matches_df.columns or "away_manager" not in matches_df.columns:
        return {
            "wins_home": 0, "wins_away": 0, "draws": 0, "total": 0,
            "home_winrate": 0.5, "confidence": "none",
        }

    df = matches_df.copy()
    df["_hm"] = df["home_manager"].fillna("").str.strip().str.lower()
    df["_am"] = df["away_manager"].fillna("").str.strip().str.lower()

    # Find matches where these two managers faced each other
    # (either direction — A managed home vs B managed away, or vice versa)
    mask_direct = (df["_hm"] == hm) & (df["_am"] == am)
    mask_reverse = (df["_hm"] == am) & (df["_am"] == hm)

    direct = df[mask_direct]
    reverse = df[mask_reverse]

    wins_home = 0
    wins_away = 0
    draws = 0

    # Direct: home_manager is the current home manager
    for _, row in direct.iterrows():
        hs = pd.to_numeric(row.get("home_score"), errors="coerce")
        as_ = pd.to_numeric(row.get("away_score"), errors="coerce")
        if pd.isna(hs) or pd.isna(as_):
            continue
        if hs > as_:
            wins_home += 1
        elif hs < as_:
            wins_away += 1
        else:
            draws += 1

    # Reverse: managers swapped sides
    for _, row in reverse.iterrows():
        hs = pd.to_numeric(row.get("home_score"), errors="coerce")
        as_ = pd.to_numeric(row.get("away_score"), errors="coerce")
        if pd.isna(hs) or pd.isna(as_):
            continue
        # In reverse, home_manager = away_manager today (who won = away_manager wins)
        if hs > as_:
            wins_away += 1
        elif hs < as_:
            wins_home += 1
        else:
            draws += 1

    total = wins_home + wins_away + draws
    home_winrate = wins_home / max(total, 1)

    if total >= 5:
        confidence = "high"
    elif total >= 3:
        confidence = "medium"
    elif total >= 1:
        confidence = "low"
    else:
        confidence = "none"

    return {
        "wins_home": wins_home,
        "wins_away": wins_away,
        "draws": draws,
        "total": total,
        "home_winrate": round(home_winrate, 3),
        "confidence": confidence,
    }


def add_manager_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Add manager tenure features to the match-level table.

    Columns added (for both home_ and away_ prefixes):
      {prefix}_manager_tenure     - number of matches managed for this team
      {prefix}_manager_is_new     - 1 if this is the manager's 1st-3rd match
      {prefix}_manager_changed    - 1 if manager is different from previous match
    """
    df = matches.copy()
    df["_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.sort_values("_date").reset_index(drop=True)

    # Check if manager columns exist
    has_home_mgr = "home_manager" in df.columns and df["home_manager"].notna().any()
    has_away_mgr = "away_manager" in df.columns and df["away_manager"].notna().any()

    if not has_home_mgr and not has_away_mgr:
        # No manager data available — add NaN columns
        for prefix in ("home", "away"):
            df[f"{prefix}_manager_tenure"] = None
            df[f"{prefix}_manager_is_new"] = None
            df[f"{prefix}_manager_changed"] = None
        df.drop(columns=["_date"], inplace=True)
        return df

    # Track manager history per team
    # team -> {current_manager, tenure_count, last_manager}
    team_mgr_state: dict[str, dict] = {}

    home_tenure = []
    home_is_new = []
    home_changed = []
    away_tenure = []
    away_is_new = []
    away_changed = []

    for _, row in df.iterrows():
        for prefix, team_col, mgr_col, tenure_list, new_list, changed_list in [
            ("home", "home_team", "home_manager", home_tenure, home_is_new, home_changed),
            ("away", "away_team", "away_manager", away_tenure, away_is_new, away_changed),
        ]:
            team = row[team_col]
            manager = row.get(mgr_col)

            if not manager or pd.isna(manager) or str(manager).strip() == "":
                tenure_list.append(None)
                new_list.append(None)
                changed_list.append(None)
                continue

            manager = str(manager).strip()

            if team not in team_mgr_state:
                team_mgr_state[team] = {
                    "current": manager,
                    "tenure": 1,
                    "last": None,
                }
                # First known match — no prior reference
                tenure_list.append(1)
                new_list.append(1)
                changed_list.append(None)  # Unknown if changed
            else:
                state = team_mgr_state[team]
                if manager == state["current"]:
                    state["tenure"] += 1
                    tenure_list.append(state["tenure"])
                    new_list.append(1 if state["tenure"] <= 3 else 0)
                    changed_list.append(0)
                else:
                    # Manager changed!
                    state["last"] = state["current"]
                    state["current"] = manager
                    state["tenure"] = 1
                    tenure_list.append(1)
                    new_list.append(1)
                    changed_list.append(1)

    df["home_manager_tenure"] = home_tenure
    df["home_manager_is_new"] = home_is_new
    df["home_manager_changed"] = home_changed
    df["away_manager_tenure"] = away_tenure
    df["away_manager_is_new"] = away_is_new
    df["away_manager_changed"] = away_changed

    # Add manager head-to-head features
    df["manager_h2h_home_winrate"] = 0.5
    df["manager_h2h_matches"] = 0
    df["manager_h2h_confidence"] = 0.0  # 1.0=high, 0.5=medium, 0.25=low, 0=none

    if has_home_mgr and has_away_mgr:
        h2h_cache: dict[tuple[str, str], dict] = {}
        for idx, row in df.iterrows():
            hm = str(row.get("home_manager", "")).strip()
            am = str(row.get("away_manager", "")).strip()
            if not hm or not am or hm == "nan" or am == "nan":
                continue

            cache_key = (hm, am)
            if cache_key not in h2h_cache:
                # Only use matches BEFORE this one to avoid leakage
                prior = df.loc[:idx - 1] if idx > 0 else pd.DataFrame()
                if not prior.empty:
                    h2h_cache[cache_key] = get_manager_h2h(hm, am, prior)
                else:
                    h2h_cache[cache_key] = {
                        "home_winrate": 0.5, "total": 0, "confidence": "none",
                    }

            h2h = h2h_cache[cache_key]
            df.at[idx, "manager_h2h_home_winrate"] = h2h.get("home_winrate", 0.5)
            df.at[idx, "manager_h2h_matches"] = h2h.get("total", 0)
            conf_map = {"high": 1.0, "medium": 0.5, "low": 0.25, "none": 0.0}
            df.at[idx, "manager_h2h_confidence"] = conf_map.get(
                h2h.get("confidence", "none"), 0.0
            )

        n_h2h = (df["manager_h2h_matches"] > 0).sum()
        log.info(f"Added 3 manager H2H features ({n_h2h} matches with H2H data)")

    df.drop(columns=["_date"], inplace=True)
    return df
