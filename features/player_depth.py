"""Tier 6 — Player-level feature gaps.

Adds features that capture squad-level variance, dependency, fatigue,
and lineup chemistry from Sofascore per-match player data.

Features added per side (home_/away_):
  A. Rating & xG Variance
     - rating_std_5       : std dev of team avg rating over last 5 matches
     - xg_std_5           : std dev of team total xG over last 5 matches
     - rating_consistency  : 1 / (1 + rating_std) — higher = more consistent

  B. Key Player Dependency
     - top1_xg_share      : top player's share of team xG (last 5)
     - top3_xg_share      : top-3 players' share of team xG (last 5)
     - xg_hhi             : Herfindahl index of xG concentration (0=equal, 1=monopoly)

  C. Fatigue Decay
     - avg_minutes_load   : avg minutes played by starters in last 3 matches
     - high_load_count    : starters who played 85+ min in all of last 3 matches
     - fatigue_risk        : high_load_count / starters (0-1)

  D. Lineup Chemistry
     - lineup_stability    : avg pairwise co-appearance rate of starters (last 5)
     - new_combo_count     : starter pairs that have never played together

All features are computed per team per match with shift(1) leak prevention,
then merged to match level as home_*/away_* columns.
"""

from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SOFASCORE_PATH = _PROJECT_ROOT / "data" / "external" / "sofascore" / "player_match_stats.parquet"

# Reuse team name normalization from sofascore_features
try:
    from features.sofascore_features import _normalize_team
except ImportError:
    def _normalize_team(name: str) -> str:
        return name.strip().lower()


def _load_sofascore_stats() -> pd.DataFrame:
    """Load and prepare Sofascore player match stats."""
    if not _SOFASCORE_PATH.exists():
        log.warning("Sofascore player stats not found at %s", _SOFASCORE_PATH)
        return pd.DataFrame()

    df = pd.read_parquet(_SOFASCORE_PATH)
    for col in ["xg", "rating", "minutes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "team", "match_id"])
    df["team_norm"] = df["team"].apply(_normalize_team)
    return df.sort_values("date").reset_index(drop=True)


def _compute_team_match_features(ps: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team per-match Tier 6 features.

    Returns DataFrame with: match_id, team_norm, date, + feature columns.
    """
    rows = []

    # Group by match and team
    for (match_id, team_norm), grp in ps.groupby(["match_id", "team_norm"]):
        match_date = grp["date"].iloc[0]
        starters = grp[grp["is_starter"] == True]

        row = {
            "match_id": match_id,
            "team_norm": team_norm,
            "date": match_date,
        }

        # --- A. Rating & xG per-match values (will be rolled later) ---
        row["_team_avg_rating"] = grp["rating"].mean()
        row["_team_total_xg"] = grp["xg"].sum()

        # --- B. Key Player Dependency (within this match) ---
        player_xg = grp.groupby("player_name")["xg"].sum().sort_values(ascending=False)
        total_xg = player_xg.sum()

        if total_xg > 0 and len(player_xg) > 0:
            row["_top1_xg_share"] = player_xg.iloc[0] / total_xg
            row["_top3_xg_share"] = player_xg.iloc[:3].sum() / total_xg
            # Herfindahl-Hirschman Index
            shares = player_xg / total_xg
            row["_xg_hhi"] = (shares ** 2).sum()
        else:
            row["_top1_xg_share"] = 0.0
            row["_top3_xg_share"] = 0.0
            row["_xg_hhi"] = 0.0

        # --- C. Fatigue: minutes played by starters ---
        if len(starters) > 0:
            row["_starter_avg_minutes"] = starters["minutes"].mean()
            row["_starter_count"] = len(starters)
        else:
            row["_starter_avg_minutes"] = 0.0
            row["_starter_count"] = 0

        # --- D. Lineup Chemistry: starter names for pairwise tracking ---
        starter_names = frozenset(starters["player_name"].dropna().tolist())
        row["_starter_names"] = starter_names

        rows.append(row)

    return pd.DataFrame(rows)


def _apply_rolling_features(team_match: pd.DataFrame) -> pd.DataFrame:
    """Apply rolling windows and compute final Tier 6 features.

    Processes per-team time series with shift(1) to prevent leakage.
    """
    team_match = team_match.sort_values(["team_norm", "date"]).reset_index(drop=True)

    result_rows = []

    for team, grp in team_match.groupby("team_norm"):
        grp = grp.sort_values("date").reset_index(drop=True)

        # Track pairwise co-appearances for chemistry
        pair_counts: dict[tuple, int] = {}
        match_count = 0

        for i, row in grp.iterrows():
            out = {
                "match_id": row["match_id"],
                "team_norm": row["team_norm"],
            }

            if match_count >= 3:
                # --- A. Rating & xG Variance (rolling 5, shifted) ---
                window = grp.iloc[max(0, match_count - 5):match_count]
                ratings = window["_team_avg_rating"].dropna()
                xgs = window["_team_total_xg"].dropna()

                out["t6_rating_std_5"] = ratings.std() if len(ratings) >= 2 else 0.0
                out["t6_xg_std_5"] = xgs.std() if len(xgs) >= 2 else 0.0
                out["t6_rating_consistency"] = 1.0 / (1.0 + out["t6_rating_std_5"])

                # --- B. Key Player Dependency (rolling 5 avg) ---
                dep_window = grp.iloc[max(0, match_count - 5):match_count]
                out["t6_top1_xg_share"] = dep_window["_top1_xg_share"].mean()
                out["t6_top3_xg_share"] = dep_window["_top3_xg_share"].mean()
                out["t6_xg_hhi"] = dep_window["_xg_hhi"].mean()

                # --- C. Fatigue Decay (last 3 matches) ---
                fatigue_window = grp.iloc[max(0, match_count - 3):match_count]
                avg_mins = fatigue_window["_starter_avg_minutes"].mean()
                out["t6_avg_minutes_load"] = round(avg_mins, 1)

                # High-load starters: those who played 85+ min in all last 3
                if len(fatigue_window) >= 3:
                    # Collect starter names per match in the window
                    window_starters = [r["_starter_names"] for _, r in fatigue_window.iterrows()]
                    # Find players who started all 3
                    if all(isinstance(s, frozenset) for s in window_starters):
                        common_starters = set.intersection(*[set(s) for s in window_starters])
                        # We approximate "85+ min in all 3" by checking they were starters in all 3
                        # (starters typically play 80-90 min)
                        high_load = len(common_starters)
                        starter_n = row["_starter_count"] if row["_starter_count"] > 0 else 11
                        out["t6_high_load_count"] = high_load
                        out["t6_fatigue_risk"] = round(high_load / starter_n, 3)
                    else:
                        out["t6_high_load_count"] = 0
                        out["t6_fatigue_risk"] = 0.0
                else:
                    out["t6_high_load_count"] = 0
                    out["t6_fatigue_risk"] = 0.0

                # --- D. Lineup Chemistry ---
                current_starters = row["_starter_names"]
                if isinstance(current_starters, frozenset) and len(current_starters) >= 2:
                    pairs = list(combinations(sorted(current_starters), 2))
                    if pairs:
                        # Average pairwise co-appearance rate
                        co_rates = []
                        new_combos = 0
                        for pair in pairs:
                            count = pair_counts.get(pair, 0)
                            if count == 0:
                                new_combos += 1
                            co_rates.append(count / match_count if match_count > 0 else 0)

                        out["t6_lineup_stability"] = round(np.mean(co_rates), 4)
                        out["t6_new_combo_count"] = new_combos
                    else:
                        out["t6_lineup_stability"] = 0.0
                        out["t6_new_combo_count"] = 0
                else:
                    out["t6_lineup_stability"] = 0.0
                    out["t6_new_combo_count"] = 0

            else:
                # Not enough history
                for col in [
                    "t6_rating_std_5", "t6_xg_std_5", "t6_rating_consistency",
                    "t6_top1_xg_share", "t6_top3_xg_share", "t6_xg_hhi",
                    "t6_avg_minutes_load", "t6_high_load_count", "t6_fatigue_risk",
                    "t6_lineup_stability", "t6_new_combo_count",
                ]:
                    out[col] = None

            result_rows.append(out)

            # --- Update pair tracking WITH this match ---
            current_starters = row["_starter_names"]
            if isinstance(current_starters, frozenset) and len(current_starters) >= 2:
                for pair in combinations(sorted(current_starters), 2):
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1

            match_count += 1

    return pd.DataFrame(result_rows)


def add_player_depth_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Add Tier 6 player-level features to match data.

    Loads Sofascore player stats, computes per-team per-match features,
    applies rolling windows, and merges to match level.

    Adds ~22 columns (11 per side).
    """
    ps = _load_sofascore_stats()
    if ps.empty:
        log.warning("No Sofascore data; skipping Tier 6 player depth features")
        return matches

    df = matches.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

    log.info("Computing Tier 6 player depth features...")

    # Step 1: Per-team per-match raw features
    team_match = _compute_team_match_features(ps)
    if team_match.empty:
        log.warning("No team-match features computed")
        return df

    # Step 2: Apply rolling + chemistry
    rolled = _apply_rolling_features(team_match)
    if rolled.empty:
        log.warning("No rolled features computed")
        return df

    feature_cols = [c for c in rolled.columns if c.startswith("t6_")]
    log.info("Computed %d Tier 6 feature columns per team", len(feature_cols))

    # Step 3: Normalize team names in matches for merge
    for side in ("home", "away"):
        tcol = f"{side}_team"
        if tcol in df.columns:
            df[f"_{side}_norm"] = df[tcol].apply(_normalize_team)

    # Step 4: Merge to match level via match_id + team_norm
    for prefix, norm_col in [("home", "_home_norm"), ("away", "_away_norm")]:
        if norm_col not in df.columns:
            continue

        merge_df = rolled[["match_id", "team_norm"] + feature_cols].copy()
        rename_map = {c: f"{prefix}_{c}" for c in feature_cols}
        rename_map["team_norm"] = norm_col
        merge_df = merge_df.rename(columns=rename_map)

        # Align match_id dtype to avoid merge errors
        if merge_df["match_id"].dtype != df["match_id"].dtype:
            merge_df["match_id"] = merge_df["match_id"].astype(df["match_id"].dtype)

        df = df.merge(merge_df, on=["match_id", norm_col], how="left")

    # Differential features
    diff_added = 0
    for col in feature_cols:
        h_col = f"home_{col}"
        a_col = f"away_{col}"
        if h_col in df.columns and a_col in df.columns:
            df[f"diff_{col}"] = df[h_col] - df[a_col]
            diff_added += 1

    # Cleanup
    df.drop(columns=["_home_norm", "_away_norm"], errors="ignore", inplace=True)

    total = sum(1 for c in df.columns if "t6_" in c)
    log.info("Added %d Tier 6 player depth features (%d per side + %d diffs)",
             total + diff_added, len(feature_cols), diff_added)

    return df
