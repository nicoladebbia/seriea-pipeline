"""Derived features computed from existing features.

These features require no external data — they combine existing columns
into higher-signal features for ML models.

Modules:
  1. Opponent-strength-adjusted stats
  2. Form vs schedule difficulty
  3. Half-time scoring patterns
  4. Goal difference momentum
  5. Points pace vs thresholds (relegation, CL qualification)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def add_derived_features(team_log: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to the team match log (pre-pivot).

    Called after rolling, strength, and momentum features are computed.
    """
    df = team_log.copy()
    df = df.sort_values(["team", "match_date"]).reset_index(drop=True)

    df = _add_opponent_adjusted_stats(df)
    df = _add_form_vs_difficulty(df)
    df = _add_goal_diff_momentum(df)
    df = _add_scoring_efficiency(df)

    return df


def add_match_level_derived(matches: pd.DataFrame) -> pd.DataFrame:
    """Add match-level derived features (post-pivot).

    Called after all team-level features are pivoted to match-level
    and after league position features are added.
    """
    df = matches.copy()
    df["_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.sort_values("_date").reset_index(drop=True)

    df = _add_ht_scoring_patterns(df)
    df = _add_points_pace(df)
    df = _add_relative_features(df)

    df.drop(columns=["_date"], errors="ignore", inplace=True)
    return df


# ─── Team-level derived features (pre-pivot) ─────────────────────────────


def _add_opponent_adjusted_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Adjust rolling stats by opponent strength.

    If a team scores 2 goals/game on average but only 1.5 when adjusted
    for opponent defense strength, they've been facing weaker defenses.

    Uses a self-join on (match_date, opponent) to get the OPPONENT's strength
    ratings, not the team's own.

    Columns added:
      adj_attack_{N}  - rolling goals_scored / opponent_defense_strength
      adj_defense_{N} - rolling goals_conceded / opponent_attack_strength
    """
    if "opponent" not in df.columns:
        log.warning("No 'opponent' column in team_log — skipping opponent-adjusted stats")
        return df

    # Build a lookup of each team's strength at each match_date.
    # We join the opponent's strength ratings onto each row.
    key_cols = ["team", "match_date"]
    str_cols = ["attack_strength", "defense_strength"]

    if not all(c in df.columns for c in str_cols):
        return df

    # Create opponent strength lookup: for each (team, match_date), get their strength
    opp_lookup = df[key_cols + str_cols].copy()
    opp_lookup = opp_lookup.rename(columns={
        "team": "opponent",
        "attack_strength": "opp_attack_strength",
        "defense_strength": "opp_defense_strength",
    })

    # Merge opponent strength onto each row
    df = df.merge(opp_lookup, on=["opponent", "match_date"], how="left")

    for window in [5, 10]:
        gs_col = f"roll_{window}_goals_scored"
        gc_col = f"roll_{window}_goals_conceded"

        if gs_col in df.columns and gc_col in df.columns:
            # Adjusted attack: goals scored / opponent's defense strength
            # High opp_defense_strength = opponent concedes a lot (weak defense)
            # So dividing by it normalizes: scoring against weak defense → lower adj_attack
            df[f"adj_attack_{window}"] = (
                df[gs_col] / df["opp_defense_strength"].replace(0, np.nan)
            ).fillna(df[gs_col])

            # Adjusted defense: goals conceded / opponent's attack strength
            # High opp_attack_strength = opponent scores a lot (strong attack)
            # So dividing by it normalizes: conceding against strong attack → lower adj_defense
            df[f"adj_defense_{window}"] = (
                df[gc_col] / df["opp_attack_strength"].replace(0, np.nan)
            ).fillna(df[gc_col])

    # Clean up temporary columns
    df.drop(columns=["opp_attack_strength", "opp_defense_strength"],
            inplace=True, errors="ignore")

    return df


def _add_form_vs_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """Compare recent form to schedule difficulty.

    A team winning 5/5 against bottom-table sides is less impressive than
    winning 3/5 against top-6. We compute an "overperformance" metric.

    Columns added:
      opp_strength_roll_5  - average opponent attack_strength over last 5
      form_overperformance - actual win rate minus expected win rate
    """
    if "attack_strength" not in df.columns:
        return df

    # Rolling average of opponent strength (using shifted data)
    # We need to build this from the team log where each row already has
    # an opponent's strength. We'll use the opponent's strength at the time.
    # Since we're in the team log, we can groupby team and shift.

    # First: mark opponent strength on each row
    # The opponent's attack_strength at this matchweek is their own value.
    # But we don't have it directly joined here. Use a proxy: strength ratings.
    # defense_strength > 1 means opponent concedes more than league average.

    # Use opponent's attack_strength as difficulty proxy.
    # opp_attack_strength is joined by _add_opponent_adjusted_stats (runs first).
    # Higher opp_attack_strength = harder schedule (faced stronger attacks).
    if "opp_attack_strength" in df.columns:
        df["opp_difficulty_roll_5"] = (
            df.groupby("team")["opp_attack_strength"]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        )
    elif "attack_strength" in df.columns:
        # Fallback: if opponent strength not available, use team's own (wrong but non-breaking)
        df["opp_difficulty_roll_5"] = (
            df.groupby("team")["attack_strength"]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        )

    # Form overperformance = actual points rate - expected from difficulty
    if "form_points_5" in df.columns and "opp_difficulty_roll_5" in df.columns:
        # Higher opponent attack strength → harder schedule → lower expected pts
        # Use expanding quantile to avoid future leakage (no global quantile)
        expanding_q95 = (
            df.groupby("team")["opp_difficulty_roll_5"]
            .transform(lambda s: s.expanding().quantile(0.95))
        )
        max_diff = expanding_q95.clip(lower=0.5)  # floor to avoid division instability
        norm_diff = (df["opp_difficulty_roll_5"] / max_diff).clip(0, 1)
        expected_points = (1 - norm_diff) * 3
        df["form_overperformance"] = df["form_points_5"] - expected_points

    return df


def _add_goal_diff_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Track how goal difference is trending.

    A team that keeps improving its GD is on an upswing.

    Columns added:
      gd_per_match      - expanding season goal difference / matches played
      gd_roll_3         - rolling 3-match goal difference
      gd_roll_5         - rolling 5-match goal difference
    """
    if "goals_scored" not in df.columns or "goals_conceded" not in df.columns:
        return df

    # Compute GD as a column for groupby.transform
    df["_gd"] = df["goals_scored"] - df["goals_conceded"]

    # Rolling goal difference (shifted to avoid leakage)
    for w in [3, 5]:
        df[f"gd_roll_{w}"] = (
            df.groupby("team")["_gd"]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        )

    # Season-expanding GD per match
    if "season" in df.columns:
        df["gd_per_match"] = (
            df.groupby(["team", "season"])["_gd"]
            .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
        )

    df.drop(columns=["_gd"], inplace=True)
    return df


def _add_scoring_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """Scoring efficiency metrics.

    Columns added:
      goals_per_shot_roll_5     - rolling goals / total_shots (if available)
      xg_overperformance_roll_5 - rolling (goals - xG) / matches
    """
    if "goals_scored" in df.columns and "total_shots" in df.columns:
        df["_gps"] = df["goals_scored"] / df["total_shots"].replace(0, np.nan)
        df["goals_per_shot_roll_5"] = (
            df.groupby("team")["_gps"]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=2).mean())
        )
        df.drop(columns=["_gps"], inplace=True)

    # xG overperformance
    if "goals_scored" in df.columns and "xg_for" in df.columns:
        df["_xg_diff"] = df["goals_scored"] - df["xg_for"]
        df["xg_overperformance_roll_5"] = (
            df.groupby("team")["_xg_diff"]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=2).mean())
        )
        df.drop(columns=["_xg_diff"], inplace=True)

    return df


# ─── Match-level derived features (post-pivot) ───────────────────────────


def _add_ht_scoring_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Half-time scoring pattern features.

    Columns added:
      ht_home_ratio      - home HT score / home FT score
      ht_away_ratio      - away HT score / away FT score
      ht_home_leading    - 1 if home team leads at HT
      ht_draw            - 1 if HT score is a draw
    """
    # These use CURRENT match HT data - which is leakage for prediction.
    # Instead, compute rolling HT tendencies from PREVIOUS matches.
    for prefix, score_col, ht_col in [
        ("home", "home_score", "home_ht_score"),
        ("away", "away_score", "away_ht_score"),
    ]:
        if ht_col not in df.columns or score_col not in df.columns:
            continue

        ht = pd.to_numeric(df[ht_col], errors="coerce")
        ft = pd.to_numeric(df[score_col], errors="coerce")

        # HT/FT ratio (capped at 1 for cases where all goals in first half)
        ratio = (ht / ft.replace(0, np.nan)).clip(0, 1)

        # Build team-level rolling HT tendency
        team_col = f"{prefix}_team"
        if team_col in df.columns:
            df[f"_ht_ratio_{prefix}"] = ratio
            df[f"{prefix}_ht_scoring_pct_roll_5"] = _team_rolling(
                df, team_col, f"_ht_ratio_{prefix}", window=5
            )
            df.drop(columns=[f"_ht_ratio_{prefix}"], inplace=True)

            # First half goals tendency
            df[f"_ht_{prefix}"] = ht
            df[f"{prefix}_ht_goals_roll_5"] = _team_rolling(
                df, team_col, f"_ht_{prefix}", window=5
            )
            df.drop(columns=[f"_ht_{prefix}"], inplace=True)

    return df


def _add_points_pace(df: pd.DataFrame) -> pd.DataFrame:
    """Track how each team's points pace compares to key thresholds.

    Serie A thresholds (approximate):
      - Relegation safety: ~36 points (0.95 PPG)
      - Europa League:     ~60 points (1.58 PPG)
      - Champions League:  ~70 points (1.84 PPG)
      - Title contention:  ~80 points (2.11 PPG)

    Columns added (for home_ and away_ prefixes):
      {prefix}_ppg_pace        - current points per game rate
      {prefix}_relegation_gap  - points above/below relegation pace
    """
    RELEGATION_PPG = 0.95
    MATCHES_PER_SEASON = 38

    for prefix in ("home", "away"):
        pts_col = f"{prefix}_league_points"

        if pts_col not in df.columns:
            continue

        if "matchweek" in df.columns:
            mw = pd.to_numeric(df["matchweek"], errors="coerce")
            pts = pd.to_numeric(df[pts_col], errors="coerce")

            ppg = pts / mw.replace(0, np.nan)
            df[f"{prefix}_ppg_pace"] = ppg
            # Only relegation gap — CL gap and proj_points are linear
            # transforms of ppg_pace that add no information for tree models
            df[f"{prefix}_relegation_gap"] = ppg - RELEGATION_PPG

    return df


def _add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute relative features between home and away teams.

    These are differences/ratios that capture matchup dynamics.

    Columns added:
      form_diff           - home form_points_5 - away form_points_5
      momentum_diff       - home win_streak - away win_streak
      strength_diff       - home attack - away attack strength
      rolling_gd_diff     - home gd_roll_5 - away gd_roll_5
    Note: league_position_diff is already in league_position.py
    """
    pairs = [
        ("form_diff", "home_form_points_5", "away_form_points_5"),
        ("momentum_diff", "home_win_streak", "away_win_streak"),
        ("attack_strength_diff", "home_attack_strength", "away_attack_strength"),
        ("defense_strength_diff", "home_defense_strength", "away_defense_strength"),
        ("rolling_gd_diff", "home_gd_roll_5", "away_gd_roll_5"),
        ("rolling_goals_diff", "home_roll_5_goals_scored", "away_roll_5_goals_scored"),
        ("rolling_xg_diff", "home_roll_5_xg_for", "away_roll_5_xg_for"),
    ]

    for name, h_col, a_col in pairs:
        if h_col in df.columns and a_col in df.columns:
            h = pd.to_numeric(df[h_col], errors="coerce")
            a = pd.to_numeric(df[a_col], errors="coerce")
            df[name] = h - a

    return df


# ─── Helpers ──────────────────────────────────────────────────────────────


def _team_rolling(
    df: pd.DataFrame, team_col: str, val_col: str, window: int = 5
) -> pd.Series:
    """Compute a shifted rolling mean grouped by team (match-level table).

    This works on the match-level table where we need to group by team_col
    (e.g. home_team) and compute rolling stats. It sorts by date and shifts
    to prevent leakage.
    """
    df = df.copy()
    if "_date" not in df.columns:
        df["_date"] = pd.to_datetime(df["match_date"], errors="coerce")

    vals = pd.to_numeric(df[val_col], errors="coerce")

    # Build team -> match history
    result = pd.Series(index=df.index, dtype=float)
    team_history: dict[str, list[float]] = {}

    for idx in df.sort_values("_date").index:
        team = df.loc[idx, team_col]
        val = vals.loc[idx]

        if team not in team_history:
            team_history[team] = []

        # Rolling average of PREVIOUS matches
        hist = team_history[team]
        if len(hist) >= 1:
            recent = hist[-window:]
            result.loc[idx] = np.nanmean(recent)
        else:
            result.loc[idx] = np.nan

        # Add current value to history AFTER computing feature
        if not pd.isna(val):
            team_history[team].append(val)

    return result
