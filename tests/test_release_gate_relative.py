"""The release gate must compare candidate vs INCUMBENT on shared-OOS folds,
not against a fixed absolute bar.

The absolute thresholds (acc>=0.50, ll<=1.00) froze production for 130 days:
every candidate was rejected on numbers measured on TODAY's folds while the
incumbent's 1.004 was measured on its own older, easier folds — a
cross-condition comparison no candidate could win as seasons drift harder.
evaluate_gate goes relative whenever incumbent fold metrics exist (folds
strictly after the incumbent's trained_through, so OOS for both models),
n_test-weighted, with catastrophic floors that apply in every mode.
"""

import pandas as pd

from scripts.models.retrain_no_odds_catboost import (
    CATASTROPHIC_ACC,
    CATASTROPHIC_LL,
    REL_GATE_TOLERANCE,
    evaluate_gate,
)

TH = {"accuracy_min": 0.50, "log_loss_max": 1.00, "brier_max": 0.205}


def _last3(*folds):
    return pd.DataFrame([
        {"accuracy": acc, "log_loss": ll, "brier_score": brier,
         "n_test": n, "inc_log_loss": inc_ll, "inc_accuracy": inc_acc}
        for (ll, acc, brier, n, inc_ll, inc_acc) in folds
    ])


def test_candidate_better_than_incumbent_promotes_past_absolute_bar():
    """The freeze scenario: candidate fails the old absolute bar (ll>1.00,
    acc<0.50) but beats the incumbent on the same rows — it must ship."""
    last3 = _last3((1.005, 0.49, 0.201, 400, 1.020, 0.48))
    rejections, info = evaluate_gate(last3, TH)
    assert info["mode"] == "relative"
    assert rejections == [], rejections


def test_candidate_worse_than_incumbent_rejected():
    last3 = _last3((1.020, 0.51, 0.201, 400, 1.005, 0.51))
    rejections, info = evaluate_gate(last3, TH)
    assert info["mode"] == "relative"
    assert any("worse than incumbent" in r for r in rejections)


def test_candidate_within_tolerance_promotes():
    inc = 1.010
    cand = inc + REL_GATE_TOLERANCE - 0.001
    last3 = _last3((cand, 0.50, 0.201, 400, inc, 0.50))
    rejections, info = evaluate_gate(last3, TH)
    assert rejections == [], rejections


def test_catastrophic_floor_beats_a_broken_incumbent():
    """Candidate better than a broken incumbent but near-random on its own:
    the floors must still reject (never promote noise 'because the incumbent
    is worse')."""
    last3 = _last3((CATASTROPHIC_LL + 0.03, 0.45, 0.24, 400,
                    CATASTROPHIC_LL + 0.20, 0.40))
    rejections, info = evaluate_gate(last3, TH)
    assert info["mode"] == "relative"
    assert any("catastrophic floor" in r for r in rejections)
    # accuracy floor too
    last3 = _last3((1.00, CATASTROPHIC_ACC - 0.02, 0.20, 400, 1.05, 0.40))
    rejections, _ = evaluate_gate(last3, TH)
    assert any("catastrophic floor" in r for r in rejections)


def test_no_incumbent_falls_back_to_absolute():
    last3 = _last3((1.005, 0.51, 0.201, 400, None, None))
    rejections, info = evaluate_gate(last3, TH)
    assert info["mode"] == "absolute"
    assert any("Log-loss" in r for r in rejections)  # 1.005 > 1.00 absolute
    # and without the columns at all
    bare = pd.DataFrame([{"accuracy": 0.52, "log_loss": 0.99,
                          "brier_score": 0.20, "n_test": 400}])
    rejections, info = evaluate_gate(bare, TH)
    assert info["mode"] == "absolute"
    assert rejections == []


def test_weighting_is_by_n_test():
    """A big fold where the candidate wins must outvote a small (but still
    gate-eligible) fold where it loses — the tiny-fold lesson, applied to the
    relative comparison."""
    last3 = _last3(
        (1.000, 0.52, 0.200, 760, 1.030, 0.50),   # candidate much better, big n
        (1.020, 0.50, 0.202, 210, 1.010, 0.51),   # candidate worse, small n
    )
    rejections, info = evaluate_gate(last3, TH)
    assert info["mode"] == "relative"
    assert rejections == [], rejections
    # flipped weights: the loss now dominates
    last3 = _last3(
        (1.000, 0.52, 0.200, 210, 1.030, 0.50),
        (1.020, 0.50, 0.202, 760, 1.001, 0.51),
    )
    rejections, _ = evaluate_gate(last3, TH)
    assert any("worse than incumbent" in r for r in rejections)
