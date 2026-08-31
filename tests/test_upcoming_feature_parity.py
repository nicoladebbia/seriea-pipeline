"""Upcoming-row derived features must match the training formulas EXACTLY.

The 2026-08-31 audit found 51/126 ML-classifier features absent from
upcoming-match rows (mean-imputed silently). The fix computes them at
prediction time with the training formulas. These tests pin:

1. Formula parity — _compute_matchup_derived reproduces the values stored in
   features.parquet for real played matches (no train/predict skew).
2. Cache coalescing — a NaN in a team's most recent row falls back to the
   previous row instead of silently dropping the feature.
"""

import math

import numpy as np
import pandas as pd
import pytest

from scripts.prediction.ensemble_prediction_engine import FeatureBuilder

# Derived features whose formulas depend only on same-row components.
# (league_avg_goals / matchweek_avg_goals / is_run_in are expanding-history
# values — they cannot be reproduced from a single row and are tested for
# plausibility separately.)
ROW_LOCAL = [
    "elo_form_blend_diff", "elo_form_disagreement",
    "momentum_diff", "position_momentum_diff",
    "us_squad_depth_diff", "us_top11_xg90_sum_diff",
    "attack_form_alignment", "form_elo_signal",
    "both_defenses_strong", "both_attacks_weak",
    "defense_mismatch_score", "btts_probability_naive",
    "poisson_home_xg", "poisson_away_xg",
    "poisson_prob_H", "poisson_prob_D", "poisson_prob_A",
    "poisson_btts",
]

INPUTS = [
    "home_elo", "away_elo", "home_form_points_5", "away_form_points_5",
    "home_elo_momentum", "away_elo_momentum",
    "home_win_streak", "away_win_streak",
    "home_us_squad_depth", "away_us_squad_depth",
    "home_us_top11_xg90_sum", "away_us_top11_xg90_sum",
    "home_position_momentum", "away_position_momentum",
    "attack_strength_diff", "elo_diff",
    "home_attack_strength", "away_attack_strength",
    "home_defense_strength", "away_defense_strength",
    "home_roll_5_goals_scored", "away_roll_5_goals_scored",
]


@pytest.fixture(scope="module")
def played_rows():
    cols = list(dict.fromkeys(INPUTS + ROW_LOCAL + ["league", "match_date", "season"]))
    df = pd.read_parquet("data/features/features.parquet", columns=cols)
    df = df[df["league"] == "serie_a"].sort_values("match_date")
    # rows where every input AND every target is present -> exact comparison
    df = df.dropna(subset=[c for c in INPUTS + ROW_LOCAL if c in df.columns])
    assert len(df) >= 20, "not enough fully-populated rows to test parity"
    return df.tail(30)


@pytest.fixture(scope="module")
def builder():
    fb = FeatureBuilder(league="serie_a")
    fb.df = pd.DataFrame()  # not needed for row-local formulas
    fb._league_ctx_cache = {}  # block history-dependent branch
    return fb


def test_row_local_formulas_match_training_values(played_rows, builder):
    mismatches = []
    for _, row in played_rows.iterrows():
        f = {k: row[k] for k in INPUTS if k in row.index and pd.notna(row[k])}
        out = builder._compute_matchup_derived(f)
        for feat in ROW_LOCAL:
            expected = row[feat]
            got = out.get(feat)
            if got is None:
                mismatches.append((feat, "not computed", expected))
                continue
            if not math.isclose(float(got), float(expected), rel_tol=1e-6, abs_tol=2e-4):
                mismatches.append((feat, got, expected))
    assert not mismatches, f"{len(mismatches)} mismatches, first 10: {mismatches[:10]}"


def test_derived_fills_only_absent_keys(builder):
    """Fresher upstream values must never be overwritten (except exact league ctx)."""
    f = {"home_elo": 1600, "away_elo": 1500, "elo_form_blend_diff": 123.4,
         "home_attack_strength": 1.2, "away_attack_strength": 0.9,
         "home_defense_strength": 1.0, "away_defense_strength": 1.1}
    out = builder._compute_matchup_derived(f)
    assert "elo_form_blend_diff" not in out  # already present -> untouched


def test_team_cache_coalesces_nan_from_previous_rows():
    """A NaN in the newest row must fall back to the previous row's value."""
    fb = FeatureBuilder(league="serie_a")
    fb.df = pd.DataFrame([
        # newest first (load_historical sorts descending)
        {"match_date": "2026-08-30", "home_team": "Inter", "away_team": "Como",
         "home_elo": 1610.0, "home_roll_5_corners": np.nan, "odds_B365H": np.nan},
        {"match_date": "2026-08-23", "home_team": "Inter", "away_team": "Pisa",
         "home_elo": 1600.0, "home_roll_5_corners": 5.4, "odds_B365H": 1.8},
    ])
    fb._build_team_cache()
    cache = fb.team_features["Inter_home"]
    assert cache["elo"] == 1610.0            # newest wins when present
    assert cache["roll_5_corners"] == 5.4    # NaN falls back to older row
    # match-level coalesce: odds_ prefix carried from the older row
    assert fb._latest_match_features.get("odds_B365H") == 1.8
