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
                result_list.append(np.nan)
                continue

            last = team_last_match.get(team)
            if last is not None and not pd.isna(last):
                gap = (md - last).days
                result_list.append(1 if gap >= _INTL_BREAK_GAP_DAYS else 0)
            else:
                result_list.append(np.nan)

            team_last_match[team] = md

    df["home_post_intl_break"] = home_post_intl
    df["away_post_intl_break"] = away_post_intl
    added += 2

    # ---------------------------------------------------------------
    # A2. Regime Change Flags
    # ---------------------------------------------------------------
    # These capture structural changes in football that affect predictions.

    # COVID no-crowd flag (2020-21): home advantage disappeared without fans.
    # Research: Serie A home win rate dropped 46% → 36%, EPL similar.
    if "season" in df.columns:
        df["is_no_crowd_match"] = df["season"].isin(["2020-2021"]).astype(int)
        added += 1

        # Five-substitution rule era (introduced 2020-21, permanent from 2022-23).
        # Deep squads benefit more — changes fatigue dynamics.
        _five_sub_seasons = {
            "2020-2021", "2021-2022", "2022-2023", "2023-2024",
            "2024-2025", "2025-2026", "2026-2027",
        }
        df["five_sub_era"] = df["season"].isin(_five_sub_seasons).astype(int)
        added += 1

    # VAR era flag (league-specific introduction dates).
    # Serie A: from 2017-18, EPL: from 2019-20.
    # Changes penalty rates (+40%), card patterns, offside decisions.
    if "season" in df.columns:
        league_col = df.get("league")
        if league_col is not None:
            from config.leagues import LEAGUE_REGISTRY
            var_flags = []
            for _, row in df.iterrows():
                season = row["season"]
                league = row.get("league", "serie_a")
                _lcfg = LEAGUE_REGISTRY.get(league)
                _var_season = _lcfg.var_introduced if _lcfg else "2017-2018"
                var_flags.append(1 if season >= _var_season else 0)
            df["var_era"] = var_flags
        else:
            # Default: assume VAR present (training starts 2017+)
            df["var_era"] = 1
        added += 1

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

        # FIX: Group by season only (not matchweek) to prevent same-matchweek
        # peers from leaking. Matches in the same matchweek happen simultaneously,
        # so one MW10 match should not see another MW10 match's goals.
        _group_cols = ["season"] if "season" in df.columns else []
        if _group_cols:
            mw_avg = (
                df.groupby(_group_cols)["_total_goals"]
                .apply(lambda s: s.expanding().mean().shift(1))
                .reset_index(level=0, drop=True)
            )
        else:
            mw_avg = df["_total_goals"].expanding().mean().shift(1)
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

        # Enriched promoted team features: how long since this team was last
        # in the top flight?  Distinguishes returning giants (Sassuolo, 1 yr)
        # from long-absent newcomers (Pisa, many years).
        all_seasons = sorted(df["season"].unique())
        # Build team→set-of-seasons lookup from the DataFrame itself
        _team_seasons: dict[str, set[str]] = {}
        for t in set(df["home_team"].unique()) | set(df["away_team"].unique()):
            _team_seasons[t] = set(
                df[(df["home_team"] == t) | (df["away_team"] == t)]["season"].unique()
            )

        home_since = []
        away_since = []
        home_prev = []
        away_prev = []
        for _, row in df.iterrows():
            cur_season = row.get("season", "")
            for team_col, promoted_val, since_list, prev_list in [
                ("home_team", row.get("home_is_promoted", 0) if "home_is_promoted" in df.columns else 0, home_since, home_prev),
                ("away_team", row.get("away_is_promoted", 0) if "away_is_promoted" in df.columns else 0, away_since, away_prev),
            ]:
                team = row.get(team_col, "")
                if not promoted_val or not team:
                    since_list.append(0)
                    prev_list.append(0)
                    continue
                # Find prior seasons this team appeared in (before current)
                prior_szns = sorted(
                    s for s in _team_seasons.get(team, set()) if s < cur_season
                )
                if prior_szns:
                    # Gap = number of seasons between last appearance and current
                    last_idx = all_seasons.index(prior_szns[-1]) if prior_szns[-1] in all_seasons else -1
                    cur_idx = all_seasons.index(cur_season) if cur_season in all_seasons else -1
                    gap = (cur_idx - last_idx - 1) if last_idx >= 0 and cur_idx >= 0 else 0
                    since_list.append(max(gap, 1))  # at least 1 (they were promoted)
                    # Previously in league within last 5 seasons?
                    recent_prior = [s for s in prior_szns if cur_idx - all_seasons.index(s) <= 5] if cur_idx >= 0 else []
                    prev_list.append(1 if recent_prior else 0)
                else:
                    since_list.append(10)  # never seen before in this dataset
                    prev_list.append(0)

        df["home_seasons_since_topflight"] = home_since
        df["away_seasons_since_topflight"] = away_since
        df["home_previously_in_league"] = home_prev
        df["away_previously_in_league"] = away_prev
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
