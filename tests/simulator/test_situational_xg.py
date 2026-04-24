"""Tests for Phase 0b.2 situational xG features."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from features.situational_xg import (
    _aggregate_xg_by_situation,
    _rolling_team_shares,
)


def test_aggregate_xg_by_situation_sums_correctly():
    shots = pd.DataFrame({
        "match_id": ["1"] * 5,
        "is_home": [True, True, True, False, False],
        "xg": [0.3, 0.2, 0.5, 0.1, 0.4],
        "is_set_piece": [False, True, False, False, False],
        "is_penalty": [False, False, True, False, False],
        "is_freekick": [False, False, False, False, False],
        "is_fast_break": [False, False, False, True, False],
    })
    agg = _aggregate_xg_by_situation(shots)
    home = agg[agg["is_home"] == True].iloc[0]  # noqa
    assert abs(home["total_xg"] - 1.0) < 1e-9
    assert abs(home["setpiece_xg"] - 0.2) < 1e-9
    assert abs(home["penalty_xg"] - 0.5) < 1e-9
    # Open play = total - setpiece - penalty (freekick also excluded but none here)
    assert abs(home["openplay_xg"] - 0.3) < 1e-9


def test_shares_sum_to_one_after_rolling():
    """Over rolling window, sum of shares should be 1 (within float tolerance)
    when total_xg > 0."""
    rows = pd.DataFrame({
        "team": ["A"] * 5,
        "opponent": ["B"] * 5,
        "match_date": pd.to_datetime([f"2024-01-0{i}" for i in range(1, 6)]),
        "match_id": [f"m{i}" for i in range(5)],
        "is_home": [True] * 5,
        "total_xg": [1.0, 1.0, 1.0, 1.0, 1.0],
        "openplay_xg": [0.7, 0.6, 0.5, 0.4, 0.3],
        "setpiece_xg": [0.2, 0.3, 0.4, 0.4, 0.5],
        "counter_xg": [0.1, 0.05, 0.05, 0.1, 0.1],
        "penalty_xg": [0.0, 0.05, 0.05, 0.1, 0.1],
        "conceded_total_xg": [1.0] * 5,
        "conceded_openplay_xg": [0.5] * 5,
        "conceded_setpiece_xg": [0.3] * 5,
        "conceded_counter_xg": [0.1] * 5,
        "conceded_penalty_xg": [0.1] * 5,
    })
    rolled = _rolling_team_shares(rows)
    last = rolled.iloc[-1]
    for N in (5, 10):
        op = last[f"xg_share_openplay_roll_{N}"]
        sp = last[f"xg_share_setpiece_roll_{N}"]
        co = last[f"xg_share_counter_roll_{N}"]
        pn = last[f"xg_share_penalty_roll_{N}"]
        assert abs(op + sp + co + pn - 1.0) < 1e-6
        # Conceded shares also sum to 1
        cop = last[f"xg_conceded_share_openplay_roll_{N}"]
        csp = last[f"xg_conceded_share_setpiece_roll_{N}"]
        cco = last[f"xg_conceded_share_counter_roll_{N}"]
        cpn = last[f"xg_conceded_share_penalty_roll_{N}"]
        assert abs(cop + csp + cco + cpn - 1.0) < 1e-6


def test_shares_are_nan_for_zero_xg_window():
    """Division by zero should produce NaN, not inf."""
    rows = pd.DataFrame({
        "team": ["A"] * 2,
        "opponent": ["B"] * 2,
        "match_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "match_id": ["m1", "m2"],
        "is_home": [True, True],
        "total_xg": [0.0, 0.0],
        "openplay_xg": [0.0, 0.0],
        "setpiece_xg": [0.0, 0.0],
        "counter_xg": [0.0, 0.0],
        "penalty_xg": [0.0, 0.0],
        "conceded_total_xg": [0.0, 0.0],
        "conceded_openplay_xg": [0.0, 0.0],
        "conceded_setpiece_xg": [0.0, 0.0],
        "conceded_counter_xg": [0.0, 0.0],
        "conceded_penalty_xg": [0.0, 0.0],
    })
    rolled = _rolling_team_shares(rows)
    # All shares should be NaN because total is zero
    for N in (5, 10):
        for sit in ("openplay", "setpiece", "counter", "penalty"):
            assert pd.isna(rolled.iloc[-1][f"xg_share_{sit}_roll_{N}"])
