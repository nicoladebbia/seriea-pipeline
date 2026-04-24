"""Tests for Phase 0b.1 shot-level xG features."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from features.shot_level_xg import (
    _aggregate_per_match_team,
    _rolling_per_team,
)


def test_aggregate_per_match_team_basic():
    shots = pd.DataFrame({
        "match_id": ["100", "100", "100", "100", "200", "200"],
        "is_home": [True, True, False, False, True, False],
        "xg": [0.1, 0.5, 0.05, 0.35, 0.8, 0.2],
        "distance": [20.0, 8.0, 25.0, 6.0, 5.0, 18.0],
        "is_set_piece": [False, False, True, False, False, False],
        "is_penalty": [False, False, False, True, False, False],
        "is_freekick": [False, False, False, False, False, False],
        "is_fast_break": [False, True, False, False, True, False],
    })
    agg = _aggregate_per_match_team(shots)
    # 4 team-match rows (2 matches × 2 sides)
    assert len(agg) == 4
    # Home in match 100: 2 shots, xG sum 0.6, mean 0.3
    hm100 = agg[(agg["sofascore_id"] == "100") & (agg["is_home"] == True)].iloc[0]  # noqa
    assert hm100["n_shots"] == 2
    assert abs(hm100["shot_xg_total"] - 0.6) < 1e-9
    assert abs(hm100["shot_xg_mean"] - 0.3) < 1e-9
    # Big chance (xg > 0.3): only one of those two shots (0.5)
    assert abs(hm100["bigchance_rate"] - 0.5) < 1e-9
    # Close (dist < 14): one of two (8.0)
    assert abs(hm100["close_rate"] - 0.5) < 1e-9
    # Counter xG: only the 0.5 shot was fast_break
    assert abs(hm100["counter_xg"] - 0.5) < 1e-9


def test_rolling_per_team_no_leakage():
    """Rolling means should exclude the current match via shift(1)."""
    rows = pd.DataFrame({
        "team": ["TeamA"] * 4,
        "match_date": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22"]),
        "match_id": ["m1", "m2", "m3", "m4"],
        "season": ["2023-2024"] * 4,
        "league": ["serie_a"] * 4,
        "is_home": [True] * 4,
        "shot_xg_mean": [0.1, 0.2, 0.3, 0.4],
        "shot_dist_mean": [20.0, 18.0, 16.0, 14.0],
        "openplay_xg": [0.5, 0.6, 0.7, 0.8],
        "setpiece_xg": [0.0, 0.1, 0.2, 0.3],
        "counter_xg": [0.1] * 4,
        "penalty_xg": [0.0] * 4,
        "bigchance_rate": [0.1, 0.2, 0.3, 0.4],
        "close_rate": [0.5] * 4,
    })
    rolled = _rolling_per_team(rows, windows=(5,))
    # m1: rolling mean uses 0 prior matches → NaN
    assert pd.isna(rolled.iloc[0]["shot_shot_xg_mean_roll_5"])
    # m2: rolling mean uses only m1's value (0.1)
    assert abs(rolled.iloc[1]["shot_shot_xg_mean_roll_5"] - 0.1) < 1e-9
    # m4: rolling mean uses m1+m2+m3 → (0.1+0.2+0.3)/3 = 0.2
    assert abs(rolled.iloc[3]["shot_shot_xg_mean_roll_5"] - 0.2) < 1e-9


def test_rolling_respects_team_boundary():
    """Team A's match history should not bleed into Team B's rolling stats."""
    rows = pd.DataFrame({
        "team": ["TeamA", "TeamA", "TeamB", "TeamB"],
        "match_date": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-01", "2024-01-08"]),
        "match_id": ["a1", "a2", "b1", "b2"],
        "season": ["2023-2024"] * 4,
        "league": ["serie_a"] * 4,
        "is_home": [True] * 4,
        "shot_xg_mean": [10.0, 20.0, 0.1, 0.2],
        "shot_dist_mean": [5.0, 5.0, 5.0, 5.0],
        "openplay_xg": [0.0] * 4,
        "setpiece_xg": [0.0] * 4,
        "counter_xg": [0.0] * 4,
        "penalty_xg": [0.0] * 4,
        "bigchance_rate": [0.0] * 4,
        "close_rate": [0.0] * 4,
    })
    rolled = _rolling_per_team(rows, windows=(5,))
    # B's second match rolling mean should be 0.1 (B's own m1), NOT 10 (TeamA's)
    b2 = rolled[(rolled["team"] == "TeamB") & (rolled["match_id"] == "b2")].iloc[0]
    assert abs(b2["shot_shot_xg_mean_roll_5"] - 0.1) < 1e-9
