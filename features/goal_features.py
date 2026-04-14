"""Goal-specific features for Over/Under and BTTS prediction.

These features capture what drives GOAL VOLUME in a match, not who wins.
The 1X2 model uses generic features — this module adds features specifically
designed for predicting whether a match will be high-scoring or low-scoring.

Features are computed at match level (post-pivot) and feed directly into
the dedicated O/U and BTTS ML models.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import poisson

log = logging.getLogger(__name__)

LEAGUE_AVG_GOALS = 2.67  # Serie A 2017-2025 average


def add_goal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add goal-prediction features to match-level DataFrame.

    Called after all team-level features are pivoted to match-level
    and after strength/Elo/rolling features are available.

    Columns added (~20):
      Goal volume predictors:
        total_xg_expected        — sum of both teams' Poisson xG
        combined_goals_per_game  — avg of both teams' recent goals scored
        combined_goals_conceded  — avg of both teams' recent goals conceded
        attack_mismatch_score    — home attack vs away defense ratio
        defense_mismatch_score   — away attack vs home defense ratio
        scoring_variance         — avg volatility in both teams' scoring
        pressing_intensity_sum   — combined pressing intensity (high press = more goals)

      BTTS predictors:
        home_scoring_rate_5      — pct of last 5 where home scored ≥1
        away_scoring_rate_5      — pct of last 5 where away scored ≥1
        home_conceding_rate_5    — pct of last 5 where home conceded ≥1
        away_conceding_rate_5    — pct of last 5 where away conceded ≥1
        btts_probability_naive   — home_scoring × away_scoring

      Poisson analytical priors (ML model learns to correct these):
        poisson_over_1_5         — P(total > 1.5) from Poisson
        poisson_over_2_5         — P(total > 2.5)
        poisson_over_3_5         — P(total > 3.5)
        poisson_btts             — P(both score) from bivariate Poisson

      Matchup context:
        h2h_goals_vs_avg         — H2H goal avg relative to league avg
        clean_sheet_matchup      — combined clean sheet tendency
        derby_goal_suppression   — derbies tend to be lower scoring
    """
    added = 0

    # --- Goal volume predictors ---

    # Total expected goals (Poisson analytical)
    if "poisson_home_xg" in df.columns and "poisson_away_xg" in df.columns:
        home_xg = pd.to_numeric(df["poisson_home_xg"], errors="coerce").fillna(1.3)
        away_xg = pd.to_numeric(df["poisson_away_xg"], errors="coerce").fillna(1.0)
        df["total_xg_expected"] = (home_xg + away_xg).round(3)
        added += 1

    # Combined goals per game (how many goals do these teams produce?)
    for stat, name in [("goals_scored", "combined_goals_per_game"),
                       ("goals_conceded", "combined_goals_conceded")]:
        h5 = f"home_roll_5_{stat}"
        a5 = f"away_roll_5_{stat}"
        if h5 in df.columns and a5 in df.columns:
            h = pd.to_numeric(df[h5], errors="coerce").fillna(1.3)
            a = pd.to_numeric(df[a5], errors="coerce").fillna(1.3)
            df[name] = ((h + a) / 2).round(3)
            added += 1

    # Attack mismatch: how exposed is the defense?
    if "home_attack_strength" in df.columns and "away_defense_strength" in df.columns:
        h_atk = pd.to_numeric(df["home_attack_strength"], errors="coerce").fillna(1.0)
        a_def = pd.to_numeric(df["away_defense_strength"], errors="coerce").fillna(1.0).clip(lower=0.1)
        df["attack_mismatch_score"] = (h_atk / a_def).round(3)
        added += 1

    if "away_attack_strength" in df.columns and "home_defense_strength" in df.columns:
        a_atk = pd.to_numeric(df["away_attack_strength"], errors="coerce").fillna(1.0)
        h_def = pd.to_numeric(df["home_defense_strength"], errors="coerce").fillna(1.0).clip(lower=0.1)
        df["defense_mismatch_score"] = (a_atk / h_def).round(3)
        added += 1

    # Scoring variance (inconsistent teams = unpredictable totals)
    h_std = f"home_roll_5_goals_scored_std"
    a_std = f"away_roll_5_goals_scored_std"
    if h_std in df.columns and a_std in df.columns:
        h = pd.to_numeric(df[h_std], errors="coerce").fillna(1.0)
        a = pd.to_numeric(df[a_std], errors="coerce").fillna(1.0)
        df["scoring_variance"] = ((h + a) / 2).round(3)
        added += 1

    # Pressing intensity (both teams pressing = more turnovers = more goals)
    if "home_ppda" in df.columns and "away_ppda" in df.columns:
        h_ppda = pd.to_numeric(df["home_ppda"], errors="coerce").fillna(12)
        a_ppda = pd.to_numeric(df["away_ppda"], errors="coerce").fillna(12)
        # Low PPDA = aggressive press. Combined low = both pressing = high intensity
        df["pressing_intensity_sum"] = ((15 - h_ppda) + (15 - a_ppda)).round(2)
        added += 1

    # --- BTTS predictors ---

    # Scoring rates: pct of recent matches where team scored ≥1
    for side in ("home", "away"):
        gs_col = f"{side}_roll_5_goals_scored"
        gc_col = f"{side}_roll_5_goals_conceded"
        cs_col = f"{side}_roll_5_clean_sheet"

        if gs_col in df.columns:
            # Scoring rate ≈ 1 - P(blank). Use rolling goals > 0 proxy.
            # If avg goals_scored > 0.5, team almost always scores.
            gs = pd.to_numeric(df[gs_col], errors="coerce").fillna(1.0)
            # Convert avg goals to scoring probability: P(score) ≈ 1 - e^(-λ)
            df[f"{side}_scoring_rate_5"] = (1 - np.exp(-gs)).round(4)
            added += 1

        if cs_col in df.columns:
            # Conceding rate = 1 - clean_sheet_rate
            cs = pd.to_numeric(df[cs_col], errors="coerce").fillna(0.3)
            df[f"{side}_conceding_rate_5"] = (1 - cs).round(4)
            added += 1

    # Naive BTTS probability
    if "home_scoring_rate_5" in df.columns and "away_scoring_rate_5" in df.columns:
        df["btts_probability_naive"] = (
            df["home_scoring_rate_5"] * df["away_scoring_rate_5"]
        ).round(4)
        added += 1

    # --- Poisson analytical priors ---
    # These become FEATURES for the ML model to learn corrections on

    if "poisson_home_xg" in df.columns and "poisson_away_xg" in df.columns:
        home_xg = pd.to_numeric(df["poisson_home_xg"], errors="coerce").fillna(1.3)
        away_xg = pd.to_numeric(df["poisson_away_xg"], errors="coerce").fillna(1.0)
        total_xg = home_xg + away_xg

        # P(Over line) = 1 - P(X ≤ floor(line))
        for line in [1.5, 2.5, 3.5]:
            k = int(line)  # floor: 1, 2, 3
            p_under = poisson.cdf(k, total_xg.values)
            df[f"poisson_over_{str(line).replace('.', '_')}"] = (1 - p_under).round(4)
            added += 1

        # P(BTTS) from bivariate Poisson (independence assumption)
        # P(BTTS) = P(home ≥ 1) × P(away ≥ 1) = (1 - P(home=0)) × (1 - P(away=0))
        p_home_blank = poisson.pmf(0, home_xg.values)
        p_away_blank = poisson.pmf(0, away_xg.values)
        df["poisson_btts"] = ((1 - p_home_blank) * (1 - p_away_blank)).round(4)
        added += 1

    # --- Matchup context ---

    # H2H goals relative to league average
    if "h2h_goals_avg" in df.columns:
        h2h_avg = pd.to_numeric(df["h2h_goals_avg"], errors="coerce")
        df["h2h_goals_vs_avg"] = (h2h_avg / LEAGUE_AVG_GOALS).round(3)
        added += 1

    # Clean sheet matchup tendency
    h_cs = "home_roll_5_clean_sheet"
    a_cs = "away_roll_5_clean_sheet"
    if h_cs in df.columns and a_cs in df.columns:
        h = pd.to_numeric(df[h_cs], errors="coerce").fillna(0.2)
        a = pd.to_numeric(df[a_cs], errors="coerce").fillna(0.2)
        df["clean_sheet_matchup"] = ((h + a) / 2).round(3)
        added += 1

    # Derby suppression (derbies average ~0.5 fewer goals)
    if "derby_intensity" in df.columns:
        di = pd.to_numeric(df["derby_intensity"], errors="coerce").fillna(0)
        df["derby_goal_suppression"] = (di * -0.15).round(3)  # intensity 3 = -0.45 goals
        added += 1

    log.info("Added %d goal-specific features for O/U and BTTS prediction", added)
    return df
