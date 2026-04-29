"""Phase 0b.1 — Shot-level xG feature extraction.

Consumes:
  - data/external/sofascore/all_shots_with_xg.parquet (82k shots, 9 seasons)
  - data/parsed/match_id_mapping.parquet (canonical ↔ sofascore_id)

Produces per-team rolling (5 and 10 last matches) features:
  home/away_shot_xg_mean_{5,10}         — avg xG per shot taken
  home/away_shot_dist_mean_{5,10}       — avg shot distance
  home/away_shot_openplay_xg_{5,10}     — sum xG from non-set-piece shots
  home/away_shot_setpiece_xg_{5,10}     — sum xG from set-piece shots
  home/away_shot_counter_xg_{5,10}      — sum xG from counterattacks
  home/away_shot_penalty_xg_{5,10}      — sum xG from penalties
  home/away_shot_bigchance_rate_{5,10}  — proportion of shots with xg > 0.3
  home/away_shot_close_rate_{5,10}      — proportion of shots < 14m

All aggregations use shift(1) before rolling — no leakage.

The shotmap has been the biggest unused data gold mine in the repo
(see .plans/data-gap-research.md §A2). Historical coverage: 2017-2025
at 100% Sofascore coverage; 2025-26 partial via shotmap_stats.parquet.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHOTS_PATH = PROJECT_ROOT / "data" / "external" / "sofascore" / "all_shots_with_xg.parquet"
MAPPING_PATH = PROJECT_ROOT / "data" / "parsed" / "match_id_mapping.parquet"


# ---------------------------------------------------------------------------
# Low-level aggregation
# ---------------------------------------------------------------------------

def _load_shots() -> pd.DataFrame | None:
    if not SHOTS_PATH.exists():
        log.warning("Shot-level xG file missing: %s", SHOTS_PATH)
        return None
    return pd.read_parquet(SHOTS_PATH)


def _load_mapping() -> pd.DataFrame | None:
    if not MAPPING_PATH.exists():
        log.warning("match_id mapping missing: %s", MAPPING_PATH)
        return None
    return pd.read_parquet(MAPPING_PATH)


def _aggregate_per_match_team(shots: pd.DataFrame) -> pd.DataFrame:
    """Collapse shot events to one row per (sofascore_match_id, is_home).

    Output columns:
      sofascore_id, is_home, n_shots,
      shot_xg_total, shot_xg_mean, shot_dist_mean,
      openplay_xg, setpiece_xg, counter_xg, penalty_xg,
      bigchance_rate, close_rate
    """
    s = shots.copy()
    s["match_id"] = s["match_id"].astype(str)
    # Derive flags robustly
    s["is_set_piece"] = s.get("is_set_piece", pd.Series(False, index=s.index)).fillna(False).astype(bool)
    s["is_fast_break"] = s.get("is_fast_break", pd.Series(False, index=s.index)).fillna(False).astype(bool)
    s["is_penalty"] = s.get("is_penalty", pd.Series(False, index=s.index)).fillna(False).astype(bool)
    s["is_freekick"] = s.get("is_freekick", pd.Series(False, index=s.index)).fillna(False).astype(bool)
    s["xg"] = s["xg"].fillna(0.0).astype(float)
    s["distance"] = s.get("distance", pd.Series(np.nan, index=s.index)).astype(float)

    s["__openplay"] = ~(s["is_set_piece"] | s["is_penalty"] | s["is_freekick"])
    s["__bigchance"] = s["xg"] > 0.3
    s["__close"] = s["distance"] < 14.0

    grouped = s.groupby(["match_id", "is_home"], observed=True).agg(
        n_shots=("xg", "size"),
        shot_xg_total=("xg", "sum"),
        shot_xg_mean=("xg", "mean"),
        shot_dist_mean=("distance", "mean"),
        openplay_xg=("xg", lambda x: float(x[s.loc[x.index, "__openplay"]].sum())),
        setpiece_xg=("xg", lambda x: float(x[s.loc[x.index, "is_set_piece"]].sum())),
        counter_xg=("xg", lambda x: float(x[s.loc[x.index, "is_fast_break"]].sum())),
        penalty_xg=("xg", lambda x: float(x[s.loc[x.index, "is_penalty"]].sum())),
        bigchance_count=("__bigchance", "sum"),
        close_count=("__close", "sum"),
    ).reset_index()
    grouped["bigchance_rate"] = grouped["bigchance_count"] / grouped["n_shots"].where(grouped["n_shots"] > 0, 1)
    grouped["close_rate"] = grouped["close_count"] / grouped["n_shots"].where(grouped["n_shots"] > 0, 1)
    grouped.drop(columns=["bigchance_count", "close_count"], inplace=True)
    grouped["sofascore_id"] = grouped["match_id"].astype(str)
    return grouped.drop(columns=["match_id"])


def _map_to_canonical(match_team_shots: pd.DataFrame, mapping: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Attach canonical match_id + team name to each row.

    Uses match_id_mapping.parquet (sofascore_id ↔ match_id) and joins with
    matches.parquet to figure out which team was home vs away per match.
    """
    mp = mapping[["match_id", "sofascore_id"]].dropna().copy()
    mp["sofascore_id"] = mp["sofascore_id"].astype(str)
    joined = match_team_shots.merge(mp, on="sofascore_id", how="inner")

    m = matches[["match_id", "home_team", "away_team", "match_date", "season", "league"]].copy()
    joined = joined.merge(m, on="match_id", how="inner")
    # Attach team name per row by is_home
    joined["team"] = np.where(joined["is_home"], joined["home_team"], joined["away_team"])
    return joined


def _rolling_per_team(team_match_rows: pd.DataFrame, windows: tuple[int, ...] = (5, 10)) -> pd.DataFrame:
    """Compute shift(1).rolling(N) per team over time.

    Input columns: team, match_date, match_id, season, league, and the
    per-team stats (shot_xg_mean, shot_dist_mean, openplay_xg, ...).
    """
    df = team_match_rows.sort_values(["team", "match_date", "match_id"]).copy()
    STAT_COLS = [
        "shot_xg_mean", "shot_dist_mean",
        "openplay_xg", "setpiece_xg", "counter_xg", "penalty_xg",
        "bigchance_rate", "close_rate",
    ]
    for N in windows:
        for c in STAT_COLS:
            colname = f"shot_{c}_roll_{N}"
            df[colname] = (
                df.groupby("team", observed=True)[c]
                .transform(lambda s: s.shift(1).rolling(N, min_periods=1).mean())
            )
    return df


def _pivot_home_away(rolled: pd.DataFrame, windows: tuple[int, ...] = (5, 10)) -> pd.DataFrame:
    """Pivot team-level rolling rows into home_X / away_X match-level columns."""
    STAT_COLS = [
        "shot_xg_mean", "shot_dist_mean",
        "openplay_xg", "setpiege_xg".replace("ge_xg", "ce_xg"),  # noqa: keep consistent
        "counter_xg", "penalty_xg",
        "bigchance_rate", "close_rate",
    ]
    # Fix the typo-safe version:
    STAT_COLS = [
        "shot_xg_mean", "shot_dist_mean",
        "openplay_xg", "setpiece_xg", "counter_xg", "penalty_xg",
        "bigchance_rate", "close_rate",
    ]
    records = []
    for mk_id, group in rolled.groupby("match_id", observed=True):
        home_row = group[group["is_home"] == True]  # noqa: E712
        away_row = group[group["is_home"] == False]  # noqa: E712
        rec = {"match_id": mk_id}
        for N in windows:
            for c in STAT_COLS:
                src = f"shot_{c}_roll_{N}"
                rec[f"home_shot_{c}_roll_{N}"] = float(home_row[src].iloc[0]) if len(home_row) and pd.notna(home_row[src].iloc[0]) else np.nan
                rec[f"away_shot_{c}_roll_{N}"] = float(away_row[src].iloc[0]) if len(away_row) and pd.notna(away_row[src].iloc[0]) else np.nan
        records.append(rec)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plugin entry point — called by Step39ShotLevelXg in features/build.py
# ---------------------------------------------------------------------------

def add_shot_level_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Add shot-level rolling features to the feature table.

    Safe to call even if shot data is missing — silently skips.
    """
    shots = _load_shots()
    mapping = _load_mapping()
    if shots is None or mapping is None:
        log.warning("Shot-level features: source data unavailable, skipping")
        return feature_df

    # Require matches info (season, date, home_team, away_team) from feature_df
    req = {"match_id", "home_team", "away_team", "match_date", "season", "league"}
    if not req.issubset(feature_df.columns):
        log.warning("Shot-level features: feature_df missing required cols %s", req - set(feature_df.columns))
        return feature_df

    matches = feature_df[list(req)].drop_duplicates(subset=["match_id"])

    log.info("Shot-level: aggregating %d shot events", len(shots))
    per_match_team = _aggregate_per_match_team(shots)
    with_match = _map_to_canonical(per_match_team, mapping, matches)
    log.info("Shot-level: %d team-match rows after canonical mapping", len(with_match))

    if len(with_match) == 0:
        log.warning("Shot-level: no matches matched between shotmap and feature_df")
        return feature_df

    rolled = _rolling_per_team(with_match)
    pivoted = _pivot_home_away(rolled)
    log.info("Shot-level: %d matches pivoted to home/away form", len(pivoted))

    # Join back to features
    before_cols = len(feature_df.columns)
    feature_df = feature_df.merge(pivoted, on="match_id", how="left")
    after_cols = len(feature_df.columns)
    log.info("Shot-level: added %d new columns", after_cols - before_cols)
    return feature_df
