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
    # Drop temporary opponent strength columns (kept through _add_form_vs_difficulty)
    df.drop(columns=["opp_attack_strength", "opp_defense_strength"],
            inplace=True, errors="ignore")
    df = _add_goal_diff_momentum(df)
    df = _add_scoring_efficiency(df)
    df = _add_momentum_sequences(df)
    df = _add_opposition_adjusted_xg(df)
    df = _add_form_volatility(df)

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
    df = _add_nonlinear_transforms(df)
    df = _add_elo_enhancements(df)
    df = _add_poisson_expectancy(df)

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

    # NOTE: opp_attack_strength / opp_defense_strength are kept for use
    # by _add_form_vs_difficulty() below, then dropped after.

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


def _add_momentum_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """Momentum sequence features: gradient, EWMA, and form acceleration.

    Captures DIRECTION of form changes, not just current level:
      momentum_gradient  — linear slope of form_points over last 5 matches
                           (positive = improving, negative = declining)
      ewma_form          — exponentially weighted form (alpha=0.3, recent = 3x older)
      last3_vs_prev3     — form_points_3 minus the 3 before that (acceleration)

    All shifted by 1 to prevent leakage.
    """
    if "points" not in df.columns:
        # Need match points (3=win, 1=draw, 0=loss) for form computations
        if "result" in df.columns:
            df["_pts"] = df["result"].map({"W": 3, "D": 1, "L": 0}).fillna(0)
        elif "goals_scored" in df.columns and "goals_conceded" in df.columns:
            gs, gc = df["goals_scored"], df["goals_conceded"]
            df["_pts"] = np.where(gs > gc, 3, np.where(gs == gc, 1, 0))
        else:
            return df
    else:
        df["_pts"] = df["points"]

    for team, grp in df.groupby("team"):
        idx = grp.index
        pts = grp["_pts"].shift(1)  # Previous match only

        # 1. Momentum gradient: slope of form over last 5 matches
        # Use simple linear regression slope: cov(x,y)/var(x) where x=0..4
        rolling_vals = pts.rolling(5, min_periods=3)
        slopes = rolling_vals.apply(
            lambda w: np.polyfit(range(len(w)), w, 1)[0] if len(w) >= 3 else np.nan,
            raw=False,
        )
        df.loc[idx, "momentum_gradient"] = slopes

        # 2. EWMA form: exponentially weighted moving average (alpha=0.3)
        # Recent matches matter ~3x more than 5 matches ago
        df.loc[idx, "ewma_form"] = pts.ewm(alpha=0.3, min_periods=3).mean()

        # 3. Last 3 vs previous 3: form acceleration
        last3 = pts.rolling(3, min_periods=2).sum()
        prev3 = pts.shift(3).rolling(3, min_periods=2).sum()
        df.loc[idx, "last3_vs_prev3"] = last3 - prev3

    df.drop(columns=["_pts"], inplace=True, errors="ignore")
    return df


def _add_opposition_adjusted_xg(df: pd.DataFrame) -> pd.DataFrame:
    """Opposition-adjusted xG: normalize team xG by opponent defensive quality.

    Like adj_attack/adj_defense but specifically for xG:
      adj_xg_attack_5  — rolling xG-for / opponent defense strength (normalized)
      adj_xg_defense_5 — rolling xG-against / opponent attack strength (normalized)

    High adj_xg_attack = scoring well even against strong defenses.
    High adj_xg_defense = conceding a lot even against weak attacks (bad).
    """
    if "xg_for" not in df.columns:
        return df

    # Need opponent strength — merge via (match_date, opponent) self-join
    if "defense_strength" not in df.columns or "attack_strength" not in df.columns:
        return df

    # Get opponent's defensive/attacking strength at match time
    opp_cols = ["team", "match_date", "defense_strength", "attack_strength"]
    if not all(c in df.columns for c in opp_cols):
        return df

    opp = df[opp_cols].copy()
    opp.columns = ["opponent", "match_date", "opp_defense_strength", "opp_attack_strength"]

    if "opponent" not in df.columns:
        return df

    merged = df.merge(opp, on=["match_date", "opponent"], how="left")

    # Adjusted xG attack: xg_for / opp_defense_strength
    # Higher opp_defense_strength = opponent concedes more (weaker defense)
    # So high adjusted xG attack means creating xG even vs solid defenses
    if "opp_defense_strength" in merged.columns:
        adj_xg = merged["xg_for"] / merged["opp_defense_strength"].replace(0, np.nan)
        merged["_adj_xg_attack"] = adj_xg

        merged["adj_xg_attack_5"] = (
            merged.groupby("team")["_adj_xg_attack"]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=2).mean())
        )
        df["adj_xg_attack_5"] = merged["adj_xg_attack_5"].values

    # Adjusted xG defense: xg_against / opp_attack_strength
    if "xg_against" in df.columns and "opp_attack_strength" in merged.columns:
        adj_xg_def = merged["xg_against"] / merged["opp_attack_strength"].replace(0, np.nan)
        merged["_adj_xg_defense"] = adj_xg_def

        merged["adj_xg_defense_5"] = (
            merged.groupby("team")["_adj_xg_defense"]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=2).mean())
        )
        df["adj_xg_defense_5"] = merged["adj_xg_defense_5"].values

    # Diff: attacking quality minus defensive vulnerability
    if "adj_xg_attack_5" in df.columns and "adj_xg_defense_5" in df.columns:
        df["adj_xg_balance_5"] = df["adj_xg_attack_5"] - df["adj_xg_defense_5"]

    return df


def _add_form_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Add form volatility: std of recent points.

    A team averaging 2.0 PPG with std=0.5 (consistent: W-W-D-W-D) is more
    predictable than one with std=1.5 (volatile: W-L-W-L-W).

    Column added:
      form_volatility - std of points over last 5 matches
    """
    if "points" not in df.columns:
        return df

    group_cols = ["team", "season"] if "season" in df.columns else ["team"]

    df["form_volatility"] = (
        df.groupby(group_cols)["points"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=3).std())
    )

    log.info("Added form_volatility feature")
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


def _add_elo_enhancements(df: pd.DataFrame) -> pd.DataFrame:
    """Add Elo-derived features that capture momentum and form alignment.

    Elo alone (importance=10.03) captures overall quality but misses:
    - Direction: is the team improving or declining?
    - Form alignment: does current form agree with long-term quality?

    Columns added:
      home/away_elo_momentum  - Elo change over last N matches (direction)
      elo_form_blend_diff     - blended Elo+form differential
      elo_form_disagreement   - divergence between Elo-implied and form-implied quality
    """
    if "home_elo" not in df.columns or "away_elo" not in df.columns:
        return df

    added = 0
    _date_col = "_date" if "_date" in df.columns else "match_date"

    # Elo momentum: compute each team's Elo N matches ago via rolling
    for side, team_col in [("home", "home_team"), ("away", "away_team")]:
        elo_col = f"{side}_elo"
        if team_col not in df.columns:
            continue

        # Build team Elo history and compute momentum (Elo now - Elo 10 matches ago)
        elo_prev = (
            df.sort_values(_date_col)
            .groupby(team_col)[elo_col]
            .transform(lambda s: s.shift(10))
        )
        df[f"{side}_elo_momentum"] = (df[elo_col] - elo_prev).round(1)
        added += 1

    # Elo momentum differential
    if "home_elo_momentum" in df.columns and "away_elo_momentum" in df.columns:
        df["elo_momentum_diff"] = (
            df["home_elo_momentum"].fillna(0) - df["away_elo_momentum"].fillna(0)
        ).round(1)
        added += 1

    # Elo-form blend: combine long-term quality (Elo) with short-term form
    # Scale form_points_5 to Elo range: 0 pts → 1400, 15 pts → 1600
    for side in ("home", "away"):
        form_col = f"{side}_form_points_5"
        elo_col = f"{side}_elo"
        if form_col in df.columns and elo_col in df.columns:
            form_scaled = 1400 + (pd.to_numeric(df[form_col], errors="coerce").fillna(7.5) / 15.0) * 200
            df[f"{side}_elo_form_blend"] = (
                0.75 * df[elo_col].fillna(1500) + 0.25 * form_scaled
            ).round(1)

    if "home_elo_form_blend" in df.columns and "away_elo_form_blend" in df.columns:
        df["elo_form_blend_diff"] = (
            df["home_elo_form_blend"] - df["away_elo_form_blend"]
        ).round(1)
        added += 1

    # Elo-form disagreement: when form ≠ Elo, regression or breakout is likely
    if "home_elo_form_blend" in df.columns:
        df["elo_form_disagreement"] = (
            (df["home_elo_form_blend"] - df["home_elo"].fillna(1500)) -
            (df["away_elo_form_blend"] - df["away_elo"].fillna(1500))
        ).round(1)
        added += 1

    # Clean up intermediate columns
    for side in ("home", "away"):
        df.drop(columns=[f"{side}_elo_form_blend"], errors="ignore", inplace=True)

    log.info("Added %d Elo enhancement features", added)
    return df


def _add_poisson_expectancy(df: pd.DataFrame) -> pd.DataFrame:
    """Add Poisson-model expected probabilities as features.

    Uses attack_strength × opponent_defense_weakness × league_avg_goals
    to compute expected goals for each team, then derives P(H), P(D), P(A)
    from the Poisson distribution.

    This gives the tree models an analytical prior — a simple statistical
    baseline they can learn to correct.

    Columns added:
      poisson_home_xg     - expected goals for home team
      poisson_away_xg     - expected goals for away team
      poisson_prob_H      - Poisson-implied P(home win)
      poisson_prob_D      - Poisson-implied P(draw)
      poisson_prob_A      - Poisson-implied P(away win)
    """
    required = ["home_attack_strength", "away_attack_strength",
                "home_defense_strength", "away_defense_strength"]
    if not all(c in df.columns for c in required):
        return df

    from scipy.stats import poisson

    LEAGUE_AVG_GOALS = 1.35  # Average goals per team per match (~2.7 total)

    home_atk = pd.to_numeric(df["home_attack_strength"], errors="coerce").fillna(1.0)
    away_atk = pd.to_numeric(df["away_attack_strength"], errors="coerce").fillna(1.0)
    home_def = pd.to_numeric(df["home_defense_strength"], errors="coerce").fillna(1.0)
    away_def = pd.to_numeric(df["away_defense_strength"], errors="coerce").fillna(1.0)

    # Expected goals: attack strength × opponent defense weakness × league avg
    # Home advantage built into attack_strength (computed on home matches)
    home_xg = (home_atk * away_def * LEAGUE_AVG_GOALS).clip(0.1, 5.0)
    away_xg = (away_atk * home_def * LEAGUE_AVG_GOALS).clip(0.1, 5.0)

    df["poisson_home_xg"] = home_xg.round(3)
    df["poisson_away_xg"] = away_xg.round(3)

    # Compute P(H), P(D), P(A) from Poisson
    max_goals = 7  # Sum probabilities for 0-7 goals (covers >99.9%)
    prob_H = np.zeros(len(df))
    prob_D = np.zeros(len(df))
    prob_A = np.zeros(len(df))

    for i in range(max_goals + 1):
        p_home_i = poisson.pmf(i, home_xg.values)
        for j in range(max_goals + 1):
            p_away_j = poisson.pmf(j, away_xg.values)
            joint = p_home_i * p_away_j
            if i > j:
                prob_H += joint
            elif i == j:
                prob_D += joint
            else:
                prob_A += joint

    df["poisson_prob_H"] = np.round(prob_H, 4)
    df["poisson_prob_D"] = np.round(prob_D, 4)
    df["poisson_prob_A"] = np.round(prob_A, 4)

    log.info("Added Poisson expectancy features (avg home_xg=%.2f, away_xg=%.2f)",
             home_xg.mean(), away_xg.mean())
    return df


def _add_nonlinear_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Add nonlinear transforms of top features to help tree models.

    Tree models can learn nonlinear splits, but explicit transforms provide
    pre-computed signals that reduce the depth needed for the same split,
    improving generalization.

    Columns added:
      elo_diff_log             - log transform for diminishing returns on Elo gaps
      attack_strength_diff_log - log transform for diminishing returns
      home_adj_attack_5_sqrt, home_adj_attack_10_sqrt  - stability dampening
      away_adj_attack_5_sqrt, away_adj_attack_10_sqrt
      home_form_points_5_sq, away_form_points_5_sq     - amplify extreme form
    """
    added = 0

    # Log transforms: diminishing returns on quality gaps
    # elo_diff 50→100 matters more than 200→250
    if "elo_diff" in df.columns:
        elo = pd.to_numeric(df["elo_diff"], errors="coerce").fillna(0)
        df["elo_diff_log"] = (np.sign(elo) * np.log1p(np.abs(elo))).round(3)
        added += 1

    if "attack_strength_diff" in df.columns:
        asd = pd.to_numeric(df["attack_strength_diff"], errors="coerce").fillna(0)
        df["attack_strength_diff_log"] = (np.sign(asd) * np.log1p(np.abs(asd))).round(4)
        added += 1

    # Sqrt transforms: stability dampening on opponent-adjusted stats
    for side in ("home", "away"):
        for window in (5, 10):
            col = f"{side}_adj_attack_{window}"
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").fillna(1.0)
                df[f"{col}_sqrt"] = np.sqrt(vals.clip(lower=0)).round(4)
                added += 1

    # Squared form: amplify extreme form (very hot or very cold)
    for side in ("home", "away"):
        col = f"{side}_form_points_5"
        if col in df.columns:
            pts = pd.to_numeric(df[col], errors="coerce").fillna(0)
            # Normalize to 0-1 range (max 15 points from 5 matches) then square
            df[f"{col}_sq"] = ((pts / 15.0) ** 2).round(4)
            added += 1

    log.info("Added %d nonlinear transform features", added)
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
