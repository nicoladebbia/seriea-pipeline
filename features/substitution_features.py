"""Substitution pattern features.

Computes team-level substitution behaviour signals from the historical
ledger at data/lineup_history/substitutions.json.

Coverage note:
  The ledger is populated by scripts/data/substitution_tracker.py
  starting from live matchday JSONs.  It is empty (or sparse) until
  several matchdays of live data have accumulated, so ALL columns will
  be NaN for historical rows and for the early part of the live-tracked
  period.  This is intentional — NaN is the correct answer when we have
  no data.  The model will impute with column medians at training time.

Columns added (both home_ and away_ prefixes):
  {prefix}_avg_subs_per_game     — mean sub count per game over last N matches
  {prefix}_avg_sub_minute        — mean minute across all subs over last N matches
  {prefix}_avg_first_sub_minute  — mean minute of the FIRST sub per game
  {prefix}_sub_games_tracked     — number of games the above averages are based on
                                   (use as a confidence/weight signal in the model)

Leakage safety:
  For each match row dated D, only ledger records with date < D are
  consulted.  The sort + strict `<` comparison ensures the current match
  is never included, even if it somehow appeared in the ledger.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.team_names import normalize_team

log = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent
SUBS_FILE = BASE / "data" / "lineup_history" / "substitutions.json"

# How many past games to look back when computing rolling averages.
_LOOKBACK_N = 10


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_ledger() -> List[dict]:
    """Load the substitution ledger.  Returns [] if missing or malformed."""
    if not SUBS_FILE.exists():
        return []
    try:
        with open(SUBS_FILE) as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            log.warning("substitutions.json is not a list — returning empty ledger")
            return []
        return data
    except (json.JSONDecodeError, IOError) as exc:
        log.warning("Could not read substitutions.json: %s", exc)
        return []


def _build_team_timeline(ledger: List[dict]) -> Dict[str, List[dict]]:
    """Index ledger records by normalised team name, sorted by date ascending.

    Each entry in the per-team list is a dict with:
      date        — pd.Timestamp
      sub_count   — int (0 if no subs recorded)
      avg_minute  — float | None
      first_minute— float | None
    """
    timeline: Dict[str, List[dict]] = {}

    for rec in ledger:
        date_str = rec.get("date", "")
        try:
            ts = pd.Timestamp(date_str)
        except Exception:
            continue

        for side in ("home", "away"):
            raw_team = rec.get(f"{side}_team", "")
            team = normalize_team(raw_team)
            if not team:
                continue

            sub_count = int(rec.get(f"{side}_sub_count", 0))
            avg_minute = rec.get(f"{side}_avg_sub_minute")  # float | None

            # Compute first sub minute from raw sub list if available
            subs_list = rec.get(f"{side}_subs", [])
            first_minute: Optional[float] = None
            if subs_list:
                minutes = [s.get("minute", 0) for s in subs_list if isinstance(s, dict)]
                if minutes:
                    first_minute = float(min(minutes))

            entry = {
                "date": ts,
                "sub_count": sub_count,
                "avg_minute": float(avg_minute) if avg_minute is not None else None,
                "first_minute": first_minute,
            }
            timeline.setdefault(team, []).append(entry)

    # Sort each team's history by date ascending
    for team in timeline:
        timeline[team].sort(key=lambda e: e["date"])

    return timeline


def _rolling_sub_stats(
    team: str,
    before_date: pd.Timestamp,
    timeline: Dict[str, List[dict]],
    n: int = _LOOKBACK_N,
) -> Dict[str, Optional[float]]:
    """Return rolling sub statistics for *team* using only records with date < before_date.

    Returns a dict with keys: avg_subs_per_game, avg_sub_minute,
    avg_first_sub_minute, games_tracked.  All values are None when there
    is no data.
    """
    empty = {
        "avg_subs_per_game": None,
        "avg_sub_minute": None,
        "avg_first_sub_minute": None,
        "games_tracked": None,
    }

    records = timeline.get(team, [])
    if not records:
        return empty

    # Take the last N records strictly before before_date
    past = [r for r in records if r["date"] < before_date]
    if not past:
        return empty

    window = past[-n:]  # most recent N

    counts = [r["sub_count"] for r in window]
    avg_minutes = [r["avg_minute"] for r in window if r["avg_minute"] is not None]
    first_minutes = [r["first_minute"] for r in window if r["first_minute"] is not None]

    return {
        "avg_subs_per_game": round(float(np.mean(counts)), 2) if counts else None,
        "avg_sub_minute": round(float(np.mean(avg_minutes)), 1) if avg_minutes else None,
        "avg_first_sub_minute": round(float(np.mean(first_minutes)), 1) if first_minutes else None,
        "games_tracked": float(len(window)),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_substitution_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Add substitution pattern features to the match-level DataFrame.

    The DataFrame must have columns: match_date, home_team, away_team.
    Team names should already be normalised (normalize_team-compatible).

    Returns a new DataFrame with 8 additional columns.
    """
    df = matches.copy()

    ledger = _load_ledger()
    if not ledger:
        log.info(
            "Substitution ledger is empty — all sub pattern features will be NaN. "
            "They will populate as live matchday data accumulates."
        )
        for prefix in ("home", "away"):
            df[f"{prefix}_avg_subs_per_game"] = np.nan
            df[f"{prefix}_avg_sub_minute"] = np.nan
            df[f"{prefix}_avg_first_sub_minute"] = np.nan
            df[f"{prefix}_sub_games_tracked"] = np.nan
        return df

    log.info("Building substitution features from %d ledger records", len(ledger))
    timeline = _build_team_timeline(ledger)

    # Pre-parse match dates once
    match_dates = pd.to_datetime(df["match_date"], errors="coerce")

    # Collect results per side
    result_cols: Dict[str, List[Optional[float]]] = {
        "home_avg_subs_per_game": [],
        "home_avg_sub_minute": [],
        "home_avg_first_sub_minute": [],
        "home_sub_games_tracked": [],
        "away_avg_subs_per_game": [],
        "away_avg_sub_minute": [],
        "away_avg_first_sub_minute": [],
        "away_sub_games_tracked": [],
    }

    for i, row in df.iterrows():
        match_dt = match_dates.iloc[i] if not isinstance(match_dates, pd.Series) else match_dates.loc[i]

        if pd.isna(match_dt):
            for col in result_cols:
                result_cols[col].append(None)
            continue

        home = normalize_team(str(row.get("home_team", "")))
        away = normalize_team(str(row.get("away_team", "")))

        h_stats = _rolling_sub_stats(home, match_dt, timeline)
        a_stats = _rolling_sub_stats(away, match_dt, timeline)

        result_cols["home_avg_subs_per_game"].append(h_stats["avg_subs_per_game"])
        result_cols["home_avg_sub_minute"].append(h_stats["avg_sub_minute"])
        result_cols["home_avg_first_sub_minute"].append(h_stats["avg_first_sub_minute"])
        result_cols["home_sub_games_tracked"].append(h_stats["games_tracked"])
        result_cols["away_avg_subs_per_game"].append(a_stats["avg_subs_per_game"])
        result_cols["away_avg_sub_minute"].append(a_stats["avg_sub_minute"])
        result_cols["away_avg_first_sub_minute"].append(a_stats["avg_first_sub_minute"])
        result_cols["away_sub_games_tracked"].append(a_stats["games_tracked"])

    for col, values in result_cols.items():
        df[col] = pd.array(values, dtype=object)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Quick coverage log
    for prefix in ("home", "away"):
        col = f"{prefix}_avg_subs_per_game"
        coverage = df[col].notna().mean()
        log.info("  %s coverage: %.1f%%", col, coverage * 100)

    return df
