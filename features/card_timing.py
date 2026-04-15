"""Card & goal timing features from Sofascore match incidents.

Uses scraped incident data to compute actual timing patterns:

  A. Card Timing (per team, rolling 5-match)
     - avg_first_card_min    : average minute of first card received
     - cards_before_30       : avg cards received before minute 30
     - cards_after_75        : avg cards received after minute 75
     - card_timing_spread    : std dev of card minutes (concentrated vs spread)

  B. Goal Timing (per team, rolling 5-match)
     - goals_first_15_rate   : % of goals scored in first 15 min
     - goals_last_15_rate    : % of goals scored in last 15 min (76-90)
     - avg_first_goal_min    : average minute of first goal scored
     - conceded_last_15_rate : % of goals conceded in last 15 min

  C. Substitution Patterns (per team, rolling 5-match)
     - avg_first_sub_min     : average minute of first substitution
     - early_sub_rate        : % of matches with sub before minute 60

All features computed per team per match with shift(1) leak prevention.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import ROLLING_WINDOW as _ROLLING_WINDOW
from features._utils import build_id_bridge

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_INCIDENTS_PATH = _PROJECT_ROOT / "data" / "external" / "sofascore" / "match_incidents.parquet"


def _load_incidents() -> pd.DataFrame:
    """Load match incidents from scraped parquet, remapping to pipeline IDs."""
    if not _INCIDENTS_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(_INCIDENTS_PATH)
        # Remap Sofascore int match_ids to pipeline string match_ids
        bridge = build_id_bridge()
        df["match_id"] = df["match_id"].map(bridge)
        df = df.dropna(subset=["match_id"])
        log.debug("Loaded %d incidents with %d mapped match IDs", len(df), df["match_id"].nunique())
        return df
    except Exception:
        return pd.DataFrame()


def _compute_per_match_patterns(incidents: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team per-match timing patterns from incidents.

    Returns DataFrame with: match_id, team, is_home, + pattern columns.
    """
    # Build match_id -> (home_team, away_team, date) lookup
    match_info = {}
    for _, row in matches.iterrows():
        mid = row.get("match_id")
        match_info[mid] = {
            "home_team": row.get("home_team", ""),
            "away_team": row.get("away_team", ""),
            "date": row.get("match_date"),
        }

    rows = []

    for match_id, grp in incidents.groupby("match_id"):
        info = match_info.get(match_id)
        if not info:
            continue

        for is_home in [True, False]:
            team = info["home_team"] if is_home else info["away_team"]
            if not team:
                continue

            side = grp[grp["is_home"] == is_home]
            opp_side = grp[grp["is_home"] != is_home]

            row = {
                "match_id": match_id,
                "team": team,
                "is_home": is_home,
                "date": info["date"],
            }

            # --- Card timing ---
            cards = side[side["incident_type"] == "card"]
            card_mins = cards["minute"].dropna().tolist()

            if card_mins:
                row["first_card_min"] = min(card_mins)
                row["cards_before_30"] = sum(1 for m in card_mins if m <= 30)
                row["cards_after_75"] = sum(1 for m in card_mins if m >= 75)
                row["card_timing_spread"] = float(np.std(card_mins)) if len(card_mins) > 1 else 0
            else:
                row["first_card_min"] = 90  # No card = treat as 90
                row["cards_before_30"] = 0
                row["cards_after_75"] = 0
                row["card_timing_spread"] = 0

            # --- Goal timing (scored) ---
            goals = side[side["incident_type"] == "goal"]
            goal_mins = goals["minute"].dropna().tolist()
            total_goals = len(goal_mins)

            if goal_mins:
                row["first_goal_min"] = min(goal_mins)
                row["goals_first_15"] = sum(1 for m in goal_mins if m <= 15)
                row["goals_last_15"] = sum(1 for m in goal_mins if m >= 76)
                row["total_goals_scored"] = total_goals
            else:
                row["first_goal_min"] = 90
                row["goals_first_15"] = 0
                row["goals_last_15"] = 0
                row["total_goals_scored"] = 0

            # --- Goals conceded timing ---
            opp_goals = opp_side[opp_side["incident_type"] == "goal"]
            opp_goal_mins = opp_goals["minute"].dropna().tolist()

            if opp_goal_mins:
                row["conceded_last_15"] = sum(1 for m in opp_goal_mins if m >= 76)
                row["total_goals_conceded"] = len(opp_goal_mins)
            else:
                row["conceded_last_15"] = 0
                row["total_goals_conceded"] = 0

            # --- Substitution timing ---
            subs = side[side["incident_type"] == "substitution"]
            sub_mins = subs["minute"].dropna().tolist()

            if sub_mins:
                row["first_sub_min"] = min(sub_mins)
                row["early_sub"] = 1 if min(sub_mins) < 60 else 0
            else:
                row["first_sub_min"] = 90
                row["early_sub"] = 0

            rows.append(row)

    return pd.DataFrame(rows)


def _apply_rolling(patterns: pd.DataFrame) -> pd.DataFrame:
    """Apply rolling averages per team with shift(1) leak prevention."""
    if patterns.empty:
        return pd.DataFrame()

    patterns["date"] = pd.to_datetime(patterns["date"], errors="coerce")
    patterns = patterns.sort_values(["team", "date"]).reset_index(drop=True)

    feature_cols = [
        "first_card_min", "cards_before_30", "cards_after_75", "card_timing_spread",
        "first_goal_min", "goals_first_15", "goals_last_15",
        "total_goals_scored", "conceded_last_15", "total_goals_conceded",
        "first_sub_min", "early_sub",
    ]

    roll_cols = []
    for col in feature_cols:
        if col in patterns.columns:
            roll_name = f"ct_{col}_r{_ROLLING_WINDOW}"
            patterns[roll_name] = (
                patterns.groupby("team")[col]
                .transform(lambda s: s.shift(1).rolling(_ROLLING_WINDOW, min_periods=3).mean())
            )
            roll_cols.append(roll_name)

    # Derived rate features
    gs_col = f"ct_total_goals_scored_r{_ROLLING_WINDOW}"
    g15_col = f"ct_goals_first_15_r{_ROLLING_WINDOW}"
    gl15_col = f"ct_goals_last_15_r{_ROLLING_WINDOW}"
    gc_col = f"ct_total_goals_conceded_r{_ROLLING_WINDOW}"
    cl15_col = f"ct_conceded_last_15_r{_ROLLING_WINDOW}"

    if gs_col in patterns.columns and g15_col in patterns.columns:
        patterns["ct_goals_first_15_rate"] = (
            patterns[g15_col] / patterns[gs_col].clip(lower=0.01)
        ).round(4)
        roll_cols.append("ct_goals_first_15_rate")

    if gs_col in patterns.columns and gl15_col in patterns.columns:
        patterns["ct_goals_last_15_rate"] = (
            patterns[gl15_col] / patterns[gs_col].clip(lower=0.01)
        ).round(4)
        roll_cols.append("ct_goals_last_15_rate")

    if gc_col in patterns.columns and cl15_col in patterns.columns:
        patterns["ct_conceded_last_15_rate"] = (
            patterns[cl15_col] / patterns[gc_col].clip(lower=0.01)
        ).round(4)
        roll_cols.append("ct_conceded_last_15_rate")

    return patterns[["match_id", "team"] + roll_cols]


def add_card_timing_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Add card/goal/sub timing features to match data.

    Adds ~30 columns (15 per side).
    """
    incidents = _load_incidents()
    if incidents.empty:
        log.warning("No incident data available; skipping card timing features")
        return matches

    df = matches.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

    log.info("Computing card/goal timing features from %d incidents...", len(incidents))

    # Step 1: Per-match patterns
    patterns = _compute_per_match_patterns(incidents, df)
    if patterns.empty:
        log.warning("No timing patterns computed")
        return df

    # Step 2: Rolling averages
    rolled = _apply_rolling(patterns)
    if rolled.empty:
        return df

    feature_cols = [c for c in rolled.columns if c.startswith("ct_")]
    log.info("Computed %d card/goal timing features per team", len(feature_cols))

    # Step 3: Merge to match level
    for prefix, team_col in [("home", "home_team"), ("away", "away_team")]:
        merge_df = rolled[["match_id", "team"] + feature_cols].copy()
        rename_map = {c: f"{prefix}_{c}" for c in feature_cols}
        rename_map["team"] = team_col
        merge_df = merge_df.rename(columns=rename_map)
        df = df.merge(merge_df, on=["match_id", team_col], how="left")

    ct_added = [c for c in df.columns if "_ct_" in c]
    log.info("Added %d card/goal timing features total", len(ct_added))

    # Fill NaN with 0 for pre-Sofascore matches (no data = neutral signal)
    df[ct_added] = df[ct_added].fillna(0)

    return df
