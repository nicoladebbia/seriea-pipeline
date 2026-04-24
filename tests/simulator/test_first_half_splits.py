"""Tests for Phase 0b.4 first-half splits."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from features.first_half_splits import _extract_period_stats, _parse_match_json


def _mini_stats(home_xg_1st: float, away_xg_1st: float, home_xg_full: float, away_xg_full: float):
    """Build a minimal Sofascore team_stats structure with 1ST + ALL periods."""
    return {
        "statistics": [
            {"period": "ALL", "groups": [
                {"groupName": "Match overview", "statisticsItems": [
                    {"key": "expectedGoals", "homeValue": home_xg_full, "awayValue": away_xg_full},
                    {"key": "totalShotsOnGoal", "homeValue": 10, "awayValue": 8},
                    {"key": "cornerKicks", "homeValue": 5, "awayValue": 3},
                ]}
            ]},
            {"period": "1ST", "groups": [
                {"groupName": "Match overview", "statisticsItems": [
                    {"key": "expectedGoals", "homeValue": home_xg_1st, "awayValue": away_xg_1st},
                    {"key": "totalShotsOnGoal", "homeValue": 4, "awayValue": 2},
                    {"key": "cornerKicks", "homeValue": 2, "awayValue": 1},
                ]}
            ]},
        ]
    }


def test_extract_1st_period_stats():
    match = {"team_stats": _mini_stats(0.3, 0.7, 1.2, 1.4)}
    fh = _extract_period_stats(match, "1ST")
    assert fh is not None
    assert abs(fh["home"]["xg"] - 0.3) < 1e-9
    assert abs(fh["away"]["xg"] - 0.7) < 1e-9
    assert fh["home"]["total_shots"] == 4
    assert fh["home"]["corners"] == 2


def test_parse_match_json_computes_fh_prop(tmp_path):
    match = {"match_id": "555", "team_stats": _mini_stats(0.3, 0.7, 1.2, 1.4)}
    p = tmp_path / "555.json"
    p.write_text(json.dumps(match))
    rec = _parse_match_json(p)
    assert rec is not None
    # home_fh_xg_prop = 0.3 / 1.2 = 0.25
    assert abs(rec["home_fh_xg_prop"] - 0.25) < 1e-9
    # away_fh_xg_prop = 0.7 / 1.4 = 0.5
    assert abs(rec["away_fh_xg_prop"] - 0.5) < 1e-9


def test_parse_missing_period_returns_none(tmp_path):
    # No 1ST period
    match = {"match_id": "777",
             "team_stats": {"statistics": [
                 {"period": "ALL", "groups": [{"groupName": "x",
                  "statisticsItems": [{"key": "expectedGoals", "homeValue": 1.0, "awayValue": 1.0}]}]}
             ]}}
    p = tmp_path / "777.json"
    p.write_text(json.dumps(match))
    assert _parse_match_json(p) is None


def test_first_half_sum_approximately_matches_full():
    """1ST + 2ND ≈ ALL (sanity check, though we only parse 1ST here)."""
    match = _mini_stats(0.3, 0.7, 1.2, 1.4)
    match["statistics"].append({"period": "2ND", "groups": [
        {"groupName": "x", "statisticsItems": [
            {"key": "expectedGoals", "homeValue": 0.9, "awayValue": 0.7},
        ]}
    ]})
    fh = _extract_period_stats({"team_stats": match}, "1ST")
    sh = _extract_period_stats({"team_stats": match}, "2ND")
    full = _extract_period_stats({"team_stats": match}, "ALL")
    # 0.3 (1st) + 0.9 (2nd) = 1.2 (full)
    assert abs((fh["home"]["xg"] + sh["home"]["xg"]) - full["home"]["xg"]) < 1e-6
