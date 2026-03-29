"""Tier 10-11 — Creative / unusual factors and timing features.

Features that capture non-obvious patterns from match context:

  A. Timing & Calendar
     - kickoff_hour         : hour of kickoff (12-21)
     - is_night_match       : kickoff >= 18:00
     - day_of_week          : 0=Mon..6=Sun
     - is_weekend           : Sat/Sun
     - season_phase         : early(1-10), mid(11-24), late(25-38)
     - is_post_intl_break   : first match after 12+ day gap (international break)

  B. Scheduling Patterns
     - days_since_last      : rest days for each team (already in rest_days.py but
                              we add the differential and fatigue interaction)
     - congestion_asymmetry : difference in rest days between teams
     - is_midweek           : Tue/Wed/Thu match

  C. Historical Scoring Patterns
     - team_avg_goals_at_venue : rolling avg goals when playing at this venue (H/A)
     - matchweek_goal_trend    : does this matchweek historically produce more goals?

  D. Promoted Team Effect
     - is_promoted           : team was promoted this season
     - promoted_matchup      : promoted vs established, or promoted derby

  E. Nothing-to-Play-For Detection
     - safe_from_relegation  : team mathematically safe
     - out_of_europe         : team can't reach European spots
     - nothing_to_play_for   : both safe and out of Europe in late season
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config.leagues import get_matchweeks, PROMOTED_TEAMS

log = logging.getLogger(__name__)

# Promoted teams sourced from the central league registry.
# Build a unified lookup keyed by (league_key, season) for multi-league support.
# Also keep a flat season->set fallback for backward compat.
_PROMOTED_BY_LEAGUE: dict[str, dict[str, set[str]]] = PROMOTED_TEAMS

# Map league display names / column values to registry keys
_LEAGUE_KEY_MAP: dict[str, str] = {
    "serie_a": "serie_a",
    "Serie A": "serie_a",
    "premier_league": "premier_league",
    "Premier League": "premier_league",
    "la_liga": "la_liga",
    "La Liga": "la_liga",
    "bundesliga": "bundesliga",
    "Bundesliga": "bundesliga",
    "ligue_1": "ligue_1",
    "Ligue 1": "ligue_1",
}


def _get_promoted(league: str | None, season: str) -> set[str]:
    """Return set of promoted teams for a league+season combo."""
    if league:
        key = _LEAGUE_KEY_MAP.get(league, league)
        result = _PROMOTED_BY_LEAGUE.get(key, {}).get(season, set())
        if result:
            return result
    # Fallback: search all leagues (backward compat for single-league runs)
    for lg_data in _PROMOTED_BY_LEAGUE.values():
        if season in lg_data:
            return lg_data[season]
    return set()

# International break matchweeks (approximate: after MW 3,7,10,13 in Serie A)
_INTL_BREAK_GAP_DAYS = 12  # If gap > 12 days, likely international break


def add_creative_factors(matches: pd.DataFrame) -> pd.DataFrame:
    """Add Tier 10-11 creative/unusual factor features.

    Adds ~20 columns covering timing, calendar, promoted teams,
    motivation, and historical scoring patterns.
    """
    df = matches.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.sort_values("match_date").reset_index(drop=True)

    log.info("Computing Tier 10-11 creative factor features...")
    added = 0

    # ---------------------------------------------------------------
    # A. Timing & Calendar
    # ---------------------------------------------------------------

    # Kickoff hour
    if "kickoff_time" in df.columns:
        kt = pd.to_datetime(df["kickoff_time"], format="%H:%M", errors="coerce")
        df["kickoff_hour"] = kt.dt.hour
        df["is_night_match"] = (df["kickoff_hour"] >= 18).astype(int)
        added += 2
    else:
        df["kickoff_hour"] = np.nan
        df["is_night_match"] = np.nan

    # Day of week
    df["day_of_week"] = df["match_date"].dt.dayofweek  # 0=Mon..6=Sun
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_midweek"] = df["day_of_week"].isin([1, 2, 3]).astype(int)
    added += 3

    # Season phase (dynamic bins based on league's matchweeks per season)
    if "matchweek" in df.columns:
        mw = pd.to_numeric(df["matchweek"], errors="coerce")
        _league = df["league"].iloc[0] if "league" in df.columns else None
        _max_mw = get_matchweeks(_league)
        # Three equal-ish phases: early / mid / late
        _b1 = round(_max_mw * 0.26)   # ~10 for 38-mw leagues
        _b2 = round(_max_mw * 0.63)   # ~24 for 38-mw leagues
        df["season_phase"] = pd.cut(
            mw, bins=[0, _b1, _b2, _max_mw], labels=[1, 2, 3],
        ).astype(float)
        added += 1

    # Post-international break detection
    # For each team, check if gap since last match > 12 days
    team_last_match: dict[str, pd.Timestamp] = {}
    home_post_intl = []
    away_post_intl = []

    for _, row in df.iterrows():
        md = row["match_date"]
        for team, result_list in [
            (row.get("home_team"), home_post_intl),
            (row.get("away_team"), away_post_intl),
        ]:
            if not team or pd.isna(team) or pd.isna(md):
                result_list.append(None)
                continue

            last = team_last_match.get(team)
            if last is not None and not pd.isna(last):
                gap = (md - last).days
                result_list.append(1 if gap >= _INTL_BREAK_GAP_DAYS else 0)
            else:
                result_list.append(None)

            team_last_match[team] = md

    df["home_post_intl_break"] = home_post_intl
    df["away_post_intl_break"] = away_post_intl
    added += 2

    # ---------------------------------------------------------------
    # B. Scheduling Patterns
    # ---------------------------------------------------------------

    # Congestion asymmetry (uses rest_days if available)
    if "home_rest_days" in df.columns and "away_rest_days" in df.columns:
        h_rest = pd.to_numeric(df["home_rest_days"], errors="coerce").fillna(7)
        a_rest = pd.to_numeric(df["away_rest_days"], errors="coerce").fillna(7)
        df["congestion_asymmetry"] = (h_rest - a_rest).round(0)
        # Fatigue flag: either team has < 4 days rest
        df["any_team_fatigued"] = ((h_rest < 4) | (a_rest < 4)).astype(int)
        added += 2

    # ---------------------------------------------------------------
    # C. Historical Scoring Patterns — matchweek goal trend
    # ---------------------------------------------------------------

    if "matchweek" in df.columns:
        hs = pd.to_numeric(df.get("home_score"), errors="coerce")
        as_ = pd.to_numeric(df.get("away_score"), errors="coerce")
        df["_total_goals"] = hs + as_

        # Expanding average goals per matchweek (shifted to prevent leakage)
        mw_avg = (
            df.groupby("matchweek")["_total_goals"]
            .apply(lambda s: s.expanding().mean().shift(1))
            .reset_index(level=0, drop=True)
        )
        df["matchweek_avg_goals"] = mw_avg
        df.drop(columns=["_total_goals"], inplace=True, errors="ignore")
        added += 1

    # ---------------------------------------------------------------
    # D. Promoted Team Effect
    # ---------------------------------------------------------------

    if "season" in df.columns:
        home_promoted = []
        away_promoted = []
        _league_col = df["league"].iloc[0] if "league" in df.columns else None
        for _, row in df.iterrows():
            season = row.get("season", "")
            league_val = row.get("league", _league_col)
            promoted = _get_promoted(league_val, season)
            ht = row.get("home_team", "")
            at = row.get("away_team", "")
            home_promoted.append(1 if ht in promoted else 0)
            away_promoted.append(1 if at in promoted else 0)

        df["home_is_promoted"] = home_promoted
        df["away_is_promoted"] = away_promoted
        df["promoted_derby"] = (
            (df["home_is_promoted"] == 1) & (df["away_is_promoted"] == 1)
        ).astype(int)
        df["promoted_vs_established"] = (
            (df["home_is_promoted"] != df["away_is_promoted"])
        ).astype(int)
        added += 4

    # ---------------------------------------------------------------
    # E. Nothing-to-Play-For Detection
    # ---------------------------------------------------------------

    if "matchweek" in df.columns:
        mw = pd.to_numeric(df["matchweek"], errors="coerce")
        late_season = mw >= 30

        for prefix in ("home", "away"):
            pos_col = f"{prefix}_league_pos"
            pts_col = f"{prefix}_league_points"

            if pos_col in df.columns and pts_col in df.columns:
                pos = pd.to_numeric(df[pos_col], errors="coerce")
                pts = pd.to_numeric(df[pts_col], errors="coerce")

                # Safe from relegation: position <= 14 in late season
                safe = (pos <= 14) & late_season
                # Out of Europe: position > 8 in late season
                out_of_europe = (pos > 8) & late_season

                df[f"{prefix}_safe_from_relegation"] = safe.astype(int)
                df[f"{prefix}_out_of_europe"] = out_of_europe.astype(int)
                df[f"{prefix}_nothing_to_play_for"] = (safe & out_of_europe).astype(int)
                added += 3

    log.info("Added %d Tier 10-11 creative factor features", added)
    return df
