"""Tests for Track 1/2 shadow prediction pipeline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.prediction.settle_shadow_log import _settle_binary, _settle_multiclass, settle_fixture


def test_settle_binary_correct_prediction():
    """Predicted 0.8 for outcome=1 — correct, low Brier."""
    r = _settle_binary(0.8, 1)
    assert r["correct"] == 1
    assert r["brier"] < 0.05


def test_settle_binary_wrong_prediction():
    r = _settle_binary(0.8, 0)
    assert r["correct"] == 0
    assert r["brier"] > 0.6


def test_settle_binary_handles_extreme_probs():
    r = _settle_binary(0.0, 1)  # Will be clipped to 1e-4
    assert r["log_loss"] > 5  # Big penalty
    r2 = _settle_binary(1.0, 0)  # Clipped
    assert r2["log_loss"] > 5


def test_settle_multiclass_1x2_correct():
    pred = {"H": 0.6, "D": 0.25, "A": 0.15}
    r = _settle_multiclass(pred, "H")
    assert r["predicted_class"] == "H"
    assert r["actual_class"] == "H"
    assert r["correct"] == 1


def test_settle_multiclass_1x2_wrong():
    pred = {"H": 0.6, "D": 0.25, "A": 0.15}
    r = _settle_multiclass(pred, "A")
    assert r["predicted_class"] == "H"
    assert r["correct"] == 0
    # Brier: (0.6-0)² + (0.25-0)² + (0.15-1)² = 0.36 + 0.0625 + 0.7225 = 1.145
    assert abs(r["brier"] - 1.145) < 0.01


def test_settle_fixture_handles_missing_optional_data():
    """If corners/cards/shots not in actuals, those markets just get skipped."""
    prediction = {
        "match_key": "2026-04-24_Napoli_Cremonese",
        "home_team": "Napoli", "away_team": "Cremonese",
        "match_date": "2026-04-24",
        "goal_markets": {
            "1X2": {"H": 0.46, "D": 0.26, "A": 0.28},
            "double_chance_1X": 0.72, "double_chance_X2": 0.54, "double_chance_12": 0.74,
            "draw_no_bet_home": 0.62, "draw_no_bet_away": 0.38,
            "over_0_5": 0.92, "over_1_5": 0.75, "over_2_5": 0.46, "over_3_5": 0.19, "over_4_5": 0.06,
            "btts": 0.50,
            "home_clean_sheet": 0.33, "away_clean_sheet": 0.25,
            "AH_home_-1.5": 0.22, "AH_home_-0.5": 0.46,
            "AH_home_+0.5": 0.72, "AH_home_+1.5": 0.89,
        },
        "corner_card_shot_markets": {},
        "top_scorers": [],
    }
    actual = {
        "home_score": 2, "away_score": 1, "result": "H",
        # No corner/card/shot fields
    }
    settled = settle_fixture(prediction, actual)
    assert settled["actual"]["total_goals"] == 3
    assert settled["actual"]["btts"] == 1
    assert settled["markets"]["1X2"]["correct"] == 1
    # Over 2.5 predicted prob = 0.46 (< 0.5 → predict "under"), actual total = 3 (over) → wrong
    assert settled["markets"]["over_2_5"]["correct"] == 0
    # BTTS predicted 0.50 (borderline), actual BTTS yes → with prob > 0.5 threshold, 0.5 is NOT > 0.5 → predicts NO
    # settle_binary uses (p > 0.5) as the predicted class marker; edge case 0.5 → predicts NO
    # Actual btts = 1, predicted 0 → wrong
    assert settled["markets"]["btts"]["correct"] == 0


def test_settle_fixture_with_full_actuals():
    """Complete settlement with corners, cards, shots."""
    prediction = {
        "match_key": "2026-04-24_Napoli_Cremonese",
        "home_team": "Napoli", "away_team": "Cremonese",
        "match_date": "2026-04-24",
        "goal_markets": {
            "1X2": {"H": 0.46, "D": 0.26, "A": 0.28},
            "double_chance_1X": 0.72, "double_chance_X2": 0.54, "double_chance_12": 0.74,
            "draw_no_bet_home": 0.62, "draw_no_bet_away": 0.38,
            "over_0_5": 0.92, "over_1_5": 0.75, "over_2_5": 0.46, "over_3_5": 0.19, "over_4_5": 0.06,
            "btts": 0.50,
            "home_clean_sheet": 0.33, "away_clean_sheet": 0.25,
            "AH_home_-1.5": 0.22, "AH_home_-0.5": 0.46,
            "AH_home_+0.5": 0.72, "AH_home_+1.5": 0.89,
        },
        "corner_card_shot_markets": {
            "corners_over_8_5": 0.60, "corners_over_9_5": 0.45,
            "cards_over_3_5": 0.30, "cards_over_4_5": 0.15,
        },
        "top_scorers": [
            {"player_id": 1, "player_name": "Rasmus Højlund", "anytime_scorer_prob": 0.40},
        ],
    }
    actual = {
        "home_score": 2, "away_score": 1, "result": "H",
        "home_corners": 7, "away_corners": 4,  # total 11, over 8.5 over 9.5 yes
        "home_yellow": 2, "away_yellow": 1, "home_red": 0, "away_red": 0,  # total 3
        "home_shots": 18, "away_shots": 9,
        "scorers": ["Rasmus Højlund"],
    }
    settled = settle_fixture(prediction, actual)
    # Corners over 8.5 predicted 0.60 (> 0.5 → predict "yes"), actual 11 > 8.5 → correct
    assert settled["markets"]["corners_over_8_5"]["correct"] == 1
    # Corners over 9.5 predicted 0.45 (< 0.5 → predict "no"), actual 11 > 9.5 → wrong
    assert settled["markets"]["corners_over_9_5"]["correct"] == 0
    # Cards over 3.5 predicted 0.30 → predict "no", actual 3 not > 3.5 → correct
    assert settled["markets"]["cards_over_3_5"]["correct"] == 1
    # Top scorer hit
    assert settled["top_scorers"][0]["scored"] == 1
    assert settled["top_scorers"][0]["brier"] < 0.4  # 0.4-1 = -0.6, brier = 0.36


def test_settle_fixture_brier_on_scorer_miss():
    prediction = {
        "match_key": "m1",
        "home_team": "A", "away_team": "B",
        "match_date": "2026-04-24",
        "goal_markets": {
            "1X2": {"H": 0.5, "D": 0.3, "A": 0.2},
            "double_chance_1X": 0.8, "double_chance_X2": 0.5, "double_chance_12": 0.7,
            "draw_no_bet_home": 0.7, "draw_no_bet_away": 0.3,
            "over_0_5": 0.9, "over_1_5": 0.7, "over_2_5": 0.5, "over_3_5": 0.2, "over_4_5": 0.05,
            "btts": 0.4,
            "home_clean_sheet": 0.4, "away_clean_sheet": 0.3,
            "AH_home_-1.5": 0.2, "AH_home_-0.5": 0.5, "AH_home_+0.5": 0.7, "AH_home_+1.5": 0.85,
        },
        "corner_card_shot_markets": {},
        "top_scorers": [
            {"player_id": 1, "player_name": "Ghost Striker", "anytime_scorer_prob": 0.9},
        ],
    }
    actual = {
        "home_score": 0, "away_score": 0, "result": "D",
        "scorers": [],  # nobody scored
    }
    settled = settle_fixture(prediction, actual)
    # Ghost Striker did NOT score but we predicted 0.9 → Brier 0.81
    assert abs(settled["top_scorers"][0]["brier"] - 0.81) < 0.01
    assert settled["top_scorers"][0]["scored"] == 0
