"""The draw detector's final fit must not validate/calibrate on the in-progress season.

2026-08-25: `val_season = seasons[-1]` was 2026-27 with 10 matches. Early stopping
stopped at 1 tree and isotonic calibration on ~2 draws collapsed to a constant
0.10; the ablation gate had scored the fold models, so a dead model shipped.
"""

import numpy as np
import pytest

from scripts.models.retrain_draw_detector import (
    _final_model_is_sane,
    _pick_final_val_season,
)

COUNTS = {"2022-2023": 380, "2023-2024": 380, "2024-2025": 380,
          "2025-2026": 380, "2026-2027": 10}


def test_val_is_newest_full_season_and_tiny_season_still_trains():
    val, train = _pick_final_val_season(COUNTS, min_n=200)
    assert val == "2025-2026"                     # bug version: "2026-2027"
    assert "2026-2027" in train and val not in train
    assert len(train) == 4


def test_no_eligible_season_raises():
    with pytest.raises(ValueError):
        _pick_final_val_season({"2026-2027": 10}, min_n=200)


def test_sanity_gate_rejects_one_tree_and_constant_calibrator():
    ok, why = _final_model_is_sane(1, np.full(300, 0.1))
    assert not ok and "trees" in why
    ok, why = _final_model_is_sane(127, np.full(300, 0.1))
    assert not ok and "constant" in why
    ok, _ = _final_model_is_sane(127, np.random.default_rng(0).uniform(0.1, 0.4, 300))
    assert ok
