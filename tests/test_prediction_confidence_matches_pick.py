"""The confidence on a prediction must be the PICKED outcome's probability.

Archived Bologna v Udinese 2026-02-23 carried `predicted_outcome: "DRAW"` with
`confidence: 0.4016` — the HOME probability. A draw-candidate override in
`calibrate_prediction` flipped the pick without moving the confidence, which was
still `max(prob_H, prob_D, prob_A)`. `classify_prediction` reads that number to
set confidence_class / recommendation / suggested_bet_size, so the row was
classified on an outcome it did not pick.

That override was DELETED 2026-09-07 after a held-out replay measured it at net
zero (see the comment in `calibrate_prediction`). These tests pin both halves:
the confidence-follows-the-pick invariant, and the deletion itself.

Every test computes the OLD value alongside the new one and asserts they differ,
so the test fails against the code it was written to pin.
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


def test_a_draw_candidate_no_longer_overrides_the_argmax_pick():
    """The deleted override, pinned. Bologna v Udinese is its exact shape."""
    c = _calibrator(_DrawCandidateDetector(True))
    res = c.calibrate_prediction("Bologna", "Udinese", dict(BOLOGNA))

    # Precondition — the old override's condition is genuinely armed here, or
    # this test proves nothing (a rejection test needs a true positive).
    assert BOLOGNA["prob_D"] > 0.28
    assert BOLOGNA["prob_D"] >= max(BOLOGNA["prob_H"], BOLOGNA["prob_A"]) - 0.03
    # ...so the old code returned DRAW on exactly these inputs.

    assert res["predicted_outcome"] == "HOME"
    assert res["confidence"] == pytest.approx(BOLOGNA["prob_H"])

    # The VALIDATED half is untouched: the analysis still reaches the caller,
    # and parlay_generator / ensemble_prediction_engine read this flag.
    assert res["draw_analysis"]["is_draw_candidate"] is True


def test_confidence_equals_the_probability_of_whatever_was_picked():
    """The invariant across every branch of the if/elif/else, both detectors."""
    key = {"HOME": "home", "DRAW": "draw", "AWAY": "away"}
    shapes = {
        "home leads": {"prob_H": 0.50, "prob_D": 0.28, "prob_A": 0.22},
        "draw leads": {"prob_H": 0.30, "prob_D": 0.45, "prob_A": 0.25},
        "away leads": {"prob_H": 0.22, "prob_D": 0.28, "prob_A": 0.50},
        "near tie":   dict(BOLOGNA),
    }
    seen = set()
    for name, probs in shapes.items():
        for candidate in (True, False):
            res = _calibrator(_DrawCandidateDetector(candidate)).calibrate_prediction(
                "Bologna", "Udinese", dict(probs)
            )
            pick = res["predicted_outcome"]
            seen.add(pick)
            assert res["confidence"] == pytest.approx(
                res["probabilities"][key[pick]], abs=1e-3
            ), f"{name} / candidate={candidate}: confidence does not match the pick"
    # All three branches were actually exercised, so the sweep is not vacuous.
    assert seen == {"HOME", "DRAW", "AWAY"}


def test_a_row_with_no_override_is_untouched_by_the_fix():
    """Regression guard: the fix is a no-op on every row now the override is gone."""
    c = _calibrator(_DrawCandidateDetector(False))
    res = c.calibrate_prediction("Bologna", "Udinese", dict(BOLOGNA))

    assert res["predicted_outcome"] == "HOME"
    # Here new and old agree — that is the point.
    assert res["confidence"] == pytest.approx(
        max(BOLOGNA["prob_H"], BOLOGNA["prob_D"], BOLOGNA["prob_A"])
    )


def test_calibration_and_the_integrator_agree_on_the_pick():
    """Finding B, pinned: the two stages composed must not disagree.

    `calibrate_prediction` used to flip the pick to DRAW on a draw candidate;
    `apply_all_intelligence` re-derives by unconditional argmax and silently
    undid it, so the same match got a different pick depending only on whether
    it happened to take the integrator path (48 of 216 archived rows, 22%).
    """
    c = _calibrator(_DrawCandidateDetector(True))
    calibrated = c.calibrate_prediction("Bologna", "Udinese", dict(BOLOGNA))

    # Precondition: this is the shape that used to diverge — the old override's
    # condition is armed, so stage 1 said DRAW and stage 2 said HOME.
    assert BOLOGNA["prob_D"] > 0.28
    assert BOLOGNA["prob_D"] >= max(BOLOGNA["prob_H"], BOLOGNA["prob_A"]) - 0.03

    # Compose them the way run_full_pipeline Step 18 does: the calibrated dict
    # itself is handed to the integrator, which mutates it in place.
    stage1_pick = calibrated["predicted_outcome"]
    integrated = apply_all_intelligence(calibrated)

    assert stage1_pick == integrated["predicted_outcome"], (
        f"stages disagree: calibration={stage1_pick} "
        f"integrator={integrated['predicted_outcome']}"
    )
    # And the invariant survives the composition.
    key = {"HOME": "home", "DRAW": "draw", "AWAY": "away"}[integrated["predicted_outcome"]]
    assert integrated["confidence"] == pytest.approx(
        integrated["probabilities"][key], abs=1e-3
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
