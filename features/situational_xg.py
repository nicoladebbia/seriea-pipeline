"""Phase 0b.2 — Situational xG share decomposition.

Builds on shot_level_xg.py. Produces per-team rolling SHARE features:

  home/away_xg_share_openplay_{5,10}   — fraction of generated xG from open play
  home/away_xg_share_setpiece_{5,10}   — fraction from set pieces
  home/away_xg_share_counter_{5,10}    — fraction from counterattacks
  home/away_xg_share_penalty_{5,10}    — fraction from penalties

Also concession equivalents (xG CONCEDED decomposition — how a defense leaks):

  home/away_xg_conceded_share_setpiece_{5,10}  — fraction of conceded xG from set pieces
  ...

A team that concedes 70% xG from set pieces is a different corners market
than 70% open play. Once in the feature table, the simulator can decompose
λ into (λ_open, λ_set, λ_counter, λ_pen) naturally.

All features use shift(1).rolling(N) over team history — no leakage.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from features._utils import load_shot_level_xg

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAPPING_PATH = PROJECT_ROOT / "data" / "parsed" / "match_id_mapping.parquet"

SITUATIONAL_WINDOWS = (5, 10)


def _load_shots() -> pd.DataFrame | None:
    return load_shot_level_xg()


def _aggregate_xg_by_situation(shots: pd.DataFrame) -> pd.DataFrame:
    """One row per (sofascore_match_id, is_home): per-situation xG sums.

    Also emits shooter-vs-conceder pair: for the DEFENSIVE shape, we need
    the xG the other team generated (hence we return rows indexed by the
    *shooter* team but will cross-join later for "conceded" features).
    """
    s = shots.copy()
    s["match_id"] = s["match_id"].astype(str)
    s["xg"] = s["xg"].fillna(0.0).astype(float)
    s["is_set_piece"] = s.get("is_set_piece", pd.Series(False, index=s.index)).fillna(False).astype(bool)
    s["is_penalty"] = s.get("is_penalty", pd.Series(False, index=s.index)).fillna(False).astype(bool)
    s["is_freekick"] = s.get("is_freekick", pd.Series(False, index=s.index)).fillna(False).astype(bool)
    s["is_fast_break"] = s.get("is_fast_break", pd.Series(False, index=s.index)).fillna(False).astype(bool)
    s["__openplay"] = ~(s["is_set_piece"] | s["is_penalty"] | s["is_freekick"])

    grouped = s.groupby(["match_id", "is_home"], observed=True).agg(
        total_xg=("xg", "sum"),
        openplay_xg=("xg", lambda x: float(x[s.loc[x.index, "__openplay"]].sum())),
        setpiece_xg=("xg", lambda x: float(x[s.loc[x.index, "is_set_piece"]].sum())),
        counter_xg=("xg", lambda x: float(x[s.loc[x.index, "is_fast_break"]].sum())),
        penalty_xg=("xg", lambda x: float(x[s.loc[x.index, "is_penalty"]].sum())),
    ).reset_index()
    grouped["sofascore_id"] = grouped["match_id"].astype(str)
    return grouped.drop(columns=["match_id"])


def _build_team_match_matrix(
    agg: pd.DataFrame,
    mapping: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """For each team, per match: xG generated + xG conceded, by situation."""
    mp = mapping[["match_id", "sofascore_id"]].dropna().copy()
    mp["sofascore_id"] = mp["sofascore_id"].astype(str)
    agg = agg.merge(mp, on="sofascore_id", how="inner")
    agg = agg.merge(matches[["match_id", "home_team", "away_team", "match_date", "season", "league"]],
                    on="match_id", how="inner")

    # Generated view: team = home if is_home else away; opponent = other
    agg["team"] = np.where(agg["is_home"], agg["home_team"], agg["away_team"])
    agg["opponent"] = np.where(agg["is_home"], agg["away_team"], agg["home_team"])

    # Create conceded view: for each row (team, match) with its *opponent's* xG stats
    # by joining same table on (match_id, opponent == team)
    opp = agg[["match_id", "team", "total_xg", "openplay_xg", "setpiece_xg", "counter_xg", "penalty_xg"]].copy()
    opp.columns = ["match_id", "opponent", "conceded_total_xg", "conceded_openplay_xg",
                   "conceded_setpiece_xg", "conceded_counter_xg", "conceded_penalty_xg"]
    merged = agg.merge(opp, on=["match_id", "opponent"], how="left")
    return merged


def _rolling_team_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Compute shift(1).rolling(N) over team history for totals, then derive shares."""
    df = df.sort_values(["team", "match_date", "match_id"]).copy()
    TOTAL_COLS = ["total_xg", "openplay_xg", "setpiece_xg", "counter_xg", "penalty_xg",
                  "conceded_total_xg", "conceded_openplay_xg",
                  "conceded_setpiece_xg", "conceded_counter_xg", "conceded_penalty_xg"]

    for N in SITUATIONAL_WINDOWS:
        for c in TOTAL_COLS:
            df[f"{c}_sum_roll_{N}"] = (
                df.groupby("team", observed=True)[c]
                .transform(lambda s: s.shift(1).rolling(N, min_periods=1).sum())
            )

    # Shares — safe division
    for N in SITUATIONAL_WINDOWS:
        tot = df[f"total_xg_sum_roll_{N}"].replace(0.0, np.nan)
        for sit in ("openplay", "setpiece", "counter", "penalty"):
            df[f"xg_share_{sit}_roll_{N}"] = df[f"{sit}_xg_sum_roll_{N}"] / tot
        tot_c = df[f"conceded_total_xg_sum_roll_{N}"].replace(0.0, np.nan)
        for sit in ("openplay", "setpiece", "counter", "penalty"):
            df[f"xg_conceded_share_{sit}_roll_{N}"] = df[f"conceded_{sit}_xg_sum_roll_{N}"] / tot_c
    return df


def _pivot_home_away(rolled: pd.DataFrame) -> pd.DataFrame:
    SHARE_COLS = []
    for N in SITUATIONAL_WINDOWS:
        for sit in ("openplay", "setpiece", "counter", "penalty"):
            SHARE_COLS.append(f"xg_share_{sit}_roll_{N}")
            SHARE_COLS.append(f"xg_conceded_share_{sit}_roll_{N}")

    records = []
    for mk_id, group in rolled.groupby("match_id", observed=True):
        home = group[group["is_home"] == True]  # noqa: E712
        away = group[group["is_home"] == False]  # noqa: E712
        rec = {"match_id": mk_id}
        for c in SHARE_COLS:
            rec[f"home_{c}"] = float(home[c].iloc[0]) if len(home) and pd.notna(home[c].iloc[0]) else np.nan
            rec[f"away_{c}"] = float(away[c].iloc[0]) if len(away) and pd.notna(away[c].iloc[0]) else np.nan
        records.append(rec)
    return pd.DataFrame(records)


def add_situational_xg_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    shots = _load_shots()
    if shots is None:
        log.warning("Situational xG: shot data unavailable, skipping")
        return feature_df
    mapping_path = MAPPING_PATH
    if not mapping_path.exists():
        log.warning("Situational xG: match_id_mapping unavailable, skipping")
        return feature_df
    mapping = pd.read_parquet(mapping_path)

    req = {"match_id", "home_team", "away_team", "match_date", "season", "league"}
    if not req.issubset(feature_df.columns):
        log.warning("Situational xG: feature_df missing %s", req - set(feature_df.columns))
        return feature_df

    matches = feature_df[list(req)].drop_duplicates(subset=["match_id"])
    log.info("Situational xG: building team-match xG matrix from %d shots", len(shots))
    agg = _aggregate_xg_by_situation(shots)
    matrix = _build_team_match_matrix(agg, mapping, matches)
    if len(matrix) == 0:
        log.warning("Situational xG: no matches joined")
        return feature_df
    rolled = _rolling_team_shares(matrix)
    pivoted = _pivot_home_away(rolled)
    before = len(feature_df.columns)
    feature_df = feature_df.merge(pivoted, on="match_id", how="left")
    log.info("Situational xG: added %d new columns", len(feature_df.columns) - before)
    return feature_df
