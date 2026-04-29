"""Phase 0b.5 — European + Coppa Italia congestion enrichment.

Current `home_congestion_3`/`_5` features count domestic league matches only.
A team in UCL quarters plays ~8 extra matches Feb-April — those matches are
in data/parsed/european_matches.json + coppa_italia_matches.json, currently
unused. That bias peaks in exactly the Feb-April window when the simulator
deploys.

This plugin produces per-team-per-match features:
  home/away_european_matches_last_7d
  home/away_european_matches_last_14d
  home/away_coppa_matches_last_7d
  home/away_days_since_european
  home/away_total_congestion_7_including_cups    (league + cups + Europe)
  home/away_total_congestion_14_including_cups

All computations use strict < match_date filter — no leakage.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EUROPEAN_PATH = PROJECT_ROOT / "data" / "parsed" / "european_matches.json"
COPPA_PATH = PROJECT_ROOT / "data" / "parsed" / "coppa_italia_matches.json"
MATCHES_PATH = PROJECT_ROOT / "data" / "parsed" / "matches.parquet"


def _load_extra_matches() -> pd.DataFrame:
    """Union of European + Coppa Italia matches as a long-form team-match frame."""
    records = []
    for path, source in [(EUROPEAN_PATH, "european"), (COPPA_PATH, "coppa")]:
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for row in data:
            if not isinstance(row, dict):
                continue
            date = row.get("date")
            if not date:
                continue
            for side in ("home_team", "away_team"):
                team = row.get(side)
                if not team:
                    continue
                records.append({
                    "team": team,
                    "match_date": pd.to_datetime(date, errors="coerce"),
                    "source": source,
                })
    df = pd.DataFrame(records).dropna(subset=["match_date"])
    return df


def _count_recent(
    team: str,
    cutoff_date: pd.Timestamp,
    window_days: int,
    extras_by_team: dict[str, pd.DataFrame],
    source_filter: str | None = None,
) -> int:
    if team not in extras_by_team:
        return 0
    sub = extras_by_team[team]
    if source_filter:
        sub = sub[sub["source"] == source_filter]
    after = cutoff_date - pd.Timedelta(days=window_days)
    mask = (sub["match_date"] >= after) & (sub["match_date"] < cutoff_date)
    return int(mask.sum())


def _days_since_last(
    team: str,
    cutoff_date: pd.Timestamp,
    extras_by_team: dict[str, pd.DataFrame],
    source_filter: str = "european",
) -> float:
    if team not in extras_by_team:
        return np.nan
    sub = extras_by_team[team]
    sub = sub[sub["source"] == source_filter]
    sub = sub[sub["match_date"] < cutoff_date]
    if len(sub) == 0:
        return np.nan
    last = sub["match_date"].max()
    return float((cutoff_date - last).days)


def add_european_congestion_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    extras = _load_extra_matches()
    if len(extras) == 0:
        log.warning("European congestion: no extra-match data, skipping")
        return feature_df

    req = {"match_id", "home_team", "away_team", "match_date"}
    if not req.issubset(feature_df.columns):
        log.warning("European congestion: feature_df missing %s", req - set(feature_df.columns))
        return feature_df

    extras_by_team: dict[str, pd.DataFrame] = {
        t: g.sort_values("match_date") for t, g in extras.groupby("team")
    }

    # Precompute existing domestic league congestion from feature_df itself
    # (each team's match dates within the same feature table).
    league_dates_by_team: dict[str, list[pd.Timestamp]] = {}
    for _, row in feature_df[["home_team", "match_date"]].dropna().drop_duplicates().iterrows():
        league_dates_by_team.setdefault(row["home_team"], []).append(row["match_date"])
    for _, row in feature_df[["away_team", "match_date"]].dropna().drop_duplicates().iterrows():
        league_dates_by_team.setdefault(row["away_team"], []).append(row["match_date"])
    league_dates_by_team = {
        t: sorted(pd.to_datetime(dates)) for t, dates in league_dates_by_team.items()
    }

    def _league_count(team: str, cutoff: pd.Timestamp, window_days: int) -> int:
        if team not in league_dates_by_team:
            return 0
        after = cutoff - pd.Timedelta(days=window_days)
        return int(sum(1 for d in league_dates_by_team[team] if after <= d < cutoff))

    new_cols = {}
    for idx, row in feature_df[["match_id", "home_team", "away_team", "match_date"]].iterrows():
        mdate = pd.to_datetime(row["match_date"])
        for prefix, team in [("home", row["home_team"]), ("away", row["away_team"])]:
            if not isinstance(team, str):
                continue
            eur_7 = _count_recent(team, mdate, 7, extras_by_team, "european")
            eur_14 = _count_recent(team, mdate, 14, extras_by_team, "european")
            coppa_7 = _count_recent(team, mdate, 7, extras_by_team, "coppa")
            coppa_14 = _count_recent(team, mdate, 14, extras_by_team, "coppa")
            days_eur = _days_since_last(team, mdate, extras_by_team, "european")
            league_7 = _league_count(team, mdate, 7)
            league_14 = _league_count(team, mdate, 14)
            new_cols.setdefault(f"{prefix}_european_matches_last_7d", {})[idx] = eur_7
            new_cols.setdefault(f"{prefix}_european_matches_last_14d", {})[idx] = eur_14
            new_cols.setdefault(f"{prefix}_coppa_matches_last_7d", {})[idx] = coppa_7
            new_cols.setdefault(f"{prefix}_coppa_matches_last_14d", {})[idx] = coppa_14
            new_cols.setdefault(f"{prefix}_days_since_european", {})[idx] = days_eur
            new_cols.setdefault(f"{prefix}_total_congestion_7_including_cups", {})[idx] = league_7 + eur_7 + coppa_7
            new_cols.setdefault(f"{prefix}_total_congestion_14_including_cups", {})[idx] = league_14 + eur_14 + coppa_14

    before = len(feature_df.columns)
    for col, vals in new_cols.items():
        feature_df[col] = pd.Series(vals)
    log.info("European congestion: added %d columns", len(feature_df.columns) - before)
    return feature_df
