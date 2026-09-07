"""The confidence on a prediction must be the PICKED outcome's probability.

Archived Bologna v Udinese 2026-02-23 carried `predicted_outcome: "DRAW"` with
`confidence: 0.4016` — the HOME probability. The draw-candidate override in
`calibrate_prediction` flips the pick without moving the confidence, which was
still `max(prob_H, prob_D, prob_A)`. `classify_prediction` reads that number to
set confidence_class / recommendation / suggested_bet_size, so an overridden
row was classified on an outcome it did not pick.

Every test here computes the OLD value alongside the new one and asserts they
differ, so the test fails against the code it was written to pin.
"""

import pytest

from features.prediction_calibration import CalibrationPipeline
from scripts.prediction.intelligence_integrator import apply_all_intelligence


class _DrawCandidateDetector:
    """Stub draw detector that always nominates the match as a draw candidate."""

    def __init__(self, is_candidate=True):
        self._is_candidate = is_candidate

    def adjust_ensemble_probs(self, home_team, away_team, probs, features):
        return probs, {"is_draw_candidate": self._is_candidate, "draw_score": 1.0}


class _NoLiveBias:
    """`active` is a read-only property, so isolate by swapping the object."""

    active = False
    sample_count = 0

    def correct(self, probs):
        return probs


def _calibrator(detector):
    c = CalibrationPipeline("default")
    c._draw_detector = detector  # draw_detector is a lazy property
    c.live_bias = _NoLiveBias()  # isolate from the settled-bet bias ledger
    return c


# The archived row, to 4dp: draw wins the override, home is the leader.
BOLOGNA = {"prob_H": 0.4016, "prob_D": 0.3980, "prob_A": 0.2004}


def test_an_overridden_draw_reports_the_draw_probability_not_the_leaders():
    c = _calibrator(_DrawCandidateDetector())
    res = c.calibrate_prediction("Bologna", "Udinese", dict(BOLOGNA))

    assert res["predicted_outcome"] == "DRAW"

    # True positive: the old code returned max(...), which is a DIFFERENT
    # number here. Without this assertion the test would pass unchanged
    # against the bug.
    old_value = max(BOLOGNA["prob_H"], BOLOGNA["prob_D"], BOLOGNA["prob_A"])
    assert old_value == pytest.approx(BOLOGNA["prob_H"])
    assert res["confidence"] != pytest.approx(old_value)

    assert res["confidence"] == pytest.approx(BOLOGNA["prob_D"])


def test_confidence_equals_the_probability_of_whatever_was_picked():
    """The invariant, stated once, for both override and no-override rows."""
    key = {"HOME": "home", "DRAW": "draw", "AWAY": "away"}
    for detector, label in ((_DrawCandidateDetector(True), "override"),
                            (_DrawCandidateDetector(False), "no override")):
        res = _calibrator(detector).calibrate_prediction(
            "Bologna", "Udinese", dict(BOLOGNA)
        )
        pick = res["predicted_outcome"]
        assert res["confidence"] == pytest.approx(
            res["probabilities"][key[pick]], abs=1e-3
        ), f"{label}: confidence does not match the pick"


def test_a_row_with_no_override_is_untouched_by_the_fix():
    """Regression guard: the fix must be a no-op on the 215-of-216 normal rows."""
    c = _calibrator(_DrawCandidateDetector(False))
    res = c.calibrate_prediction("Bologna", "Udinese", dict(BOLOGNA))

    assert res["predicted_outcome"] == "HOME"
    # Here new and old agree — that is the point.
    assert res["confidence"] == pytest.approx(
        max(BOLOGNA["prob_H"], BOLOGNA["prob_D"], BOLOGNA["prob_A"])
    )


# --- intelligence_integrator: derive from full precision, not the 3dp copy ---

# Two outcomes inside the same 0.0005 rounding bucket: draw leads by 5e-5, but
# both round to 0.400, and the old code's `p["home"] >= p["draw"]` tie-break
# then hands the match to HOME.
TIE_AFTER_ROUNDING = {"home": 0.40040, "draw": 0.40045, "away": 0.19915}


def test_a_rounding_tie_does_not_let_the_display_copy_pick_the_match():
    p = dict(TIE_AFTER_ROUNDING)

    # Precondition — assert the trap is actually armed, or the test proves
    # nothing (a rejection test needs a true positive).
    assert round(p["home"], 3) == round(p["draw"], 3), "rounding bucket assumption"
    assert p["draw"] > p["home"], "draw must genuinely lead"

    res = apply_all_intelligence({"probabilities": p})

    # The old code read the rounded copy and picked HOME on the >= tie.
    assert res["predicted_outcome"] == "DRAW"
    assert res["confidence"] == pytest.approx(p["draw"])


def test_integrator_confidence_matches_its_own_pick():
    key = {"HOME": "home", "DRAW": "draw", "AWAY": "away"}
    res = apply_all_intelligence({"probabilities": dict(TIE_AFTER_ROUNDING)})
    pick = res["predicted_outcome"]

    # Full precision, not the rounded display value it used to report.
    assert res["confidence"] == pytest.approx(TIE_AFTER_ROUNDING[key[pick]])
    assert res["confidence"] != pytest.approx(res["probabilities"][key[pick]], abs=0)
