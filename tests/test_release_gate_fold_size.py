"""The release gate must not be decided by a fold too small to measure anything.

Anchored on the real 2026-08-25 retrain: `catboost_no_odds` was rejected at
accuracy 0.4565 because the walk-forward split produced a fold for the
brand-new 2026-2027 season holding ten matches, and the gate averaged it
unweighted alongside two folds of 760 and 689.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ml.evaluation import MIN_GATE_TEST_MATCHES, gate_folds


def _fold(fold: int, season: str, n_test: int, acc: float, ll: float) -> dict:
    return {
        "fold": fold,
        "test": season,
        "n_train": 5000,
        "n_test": n_test,
        "accuracy": acc,
        "log_loss": ll,
        "brier_score": 0.20,
        "f1_D": 0.25,
    }


@pytest.fixture
def august_2026_cv() -> pd.DataFrame:
    """The exact per-fold numbers the 2026-08-25 retrain produced."""
    return pd.DataFrame(
        [
            _fold(0, "2022-2023", 760, 0.493, 1.0513),
            _fold(1, "2023-2024", 760, 0.563, 0.9729),
            _fold(2, "2024-2025", 760, 0.492, 1.0110),
            _fold(3, "2025-2026", 689, 0.477, 1.0393),
            _fold(4, "2026-2027", 10, 0.400, 0.9750),
        ]
    )


def test_a_ten_match_fold_does_not_decide_the_release(august_2026_cv):
    cv = august_2026_cv

    # True positive: the naive aggregate really does reject this model, and it
    # rejects it *because* of the ten-match fold. Without both assertions the
    # test below would pass against the unfixed code.
    assert cv.tail(3)["accuracy"].mean() < 0.50
    assert cv.tail(3)["n_test"].min() == 10

    gated = gate_folds(cv)

    assert 10 not in set(gated["n_test"])
    assert gated["accuracy"].mean() >= 0.50


def test_the_gate_still_reads_the_three_most_recent_usable_folds(august_2026_cv):
    gated = gate_folds(august_2026_cv)

    assert list(gated["test"]) == ["2023-2024", "2024-2025", "2025-2026"]


def test_an_all_full_season_history_is_untouched():
    """No regression: when every fold is big enough, this is plain tail(3)."""
    cv = pd.DataFrame(
        [_fold(i, f"20{17 + i}-20{18 + i}", 760, 0.51, 0.99) for i in range(5)]
    )

    pd.testing.assert_frame_equal(gate_folds(cv), cv.tail(3))


def test_an_undersized_fold_in_the_middle_is_skipped_not_just_a_trailing_one():
    cv = pd.DataFrame(
        [
            _fold(0, "2021-2022", 760, 0.50, 1.00),
            _fold(1, "2022-2023", 760, 0.51, 0.99),
            _fold(2, "2023-2024", 12, 0.90, 0.50),  # anomaly mid-history
            _fold(3, "2024-2025", 760, 0.52, 0.98),
            _fold(4, "2025-2026", 689, 0.49, 1.01),
        ]
    )

    gated = gate_folds(cv)

    assert list(gated["test"]) == ["2022-2023", "2024-2025", "2025-2026"]


def test_a_history_with_no_usable_fold_falls_back_rather_than_gating_on_nothing(caplog):
    cv = pd.DataFrame([_fold(i, f"200{i}", 10, 0.4, 1.2) for i in range(4)])

    with caplog.at_level("ERROR"):
        gated = gate_folds(cv)

    assert len(gated) == 3
    assert "no fold" in caplog.text.lower()


def test_the_threshold_is_large_enough_that_a_fold_cannot_swing_the_decision():
    """A gate fold's accuracy standard error must stay well under the margin
    the thresholds decide on (~0.01-0.02). se = sqrt(0.25 / n)."""
    se = (0.25 / MIN_GATE_TEST_MATCHES) ** 0.5

    assert se <= 0.04


# ---------------------------------------------------------------------------
# The isotonic calibrator ships to inference — it must never be fit on a
# handful of matches. Isotonic on ten points is a step function that collapses
# most inputs onto a few outputs; that saturated curve would then be applied to
# every live prediction via lean_calibrators.pkl.
# ---------------------------------------------------------------------------


def _synthetic_seasons() -> tuple:
    """Six full seasons plus a ten-match opening season, like August 2026."""
    import numpy as np

    rng = np.random.default_rng(0)
    rows, labels = [], []
    sizes = {f"20{17 + i}-20{18 + i}": 250 for i in range(6)}
    sizes["2026-2027"] = 10  # the newest season has barely started

    for season, n in sizes.items():
        for _ in range(n):
            a, b = rng.normal(size=2)
            rows.append({"_season": season, "f_a": a, "f_b": b})
            labels.append("H" if a > b else ("D" if abs(a - b) < 0.4 else "A"))

    X = pd.DataFrame(rows)
    y_str = pd.Series(labels)
    y = y_str.map({"H": 0, "D": 1, "A": 2})
    return X, y, y_str


def test_the_shipped_calibrator_is_not_fit_on_the_new_seasons_first_ten_matches():
    from scripts.models.retrain_no_odds_catboost import walk_forward_validate

    X, y, y_str = _synthetic_seasons()

    # True positive: the final fold really is the ten-match season, so a
    # last-fold rule would genuinely calibrate on ten predictions.
    assert (X["_season"] == "2026-2027").sum() == 10
    assert sorted(X["_season"].unique())[-1] == "2026-2027"

    _cv, model, proba, y_cal = walk_forward_validate(
        X, y, y_str,
        params={"iterations": 5, "depth": 2, "verbose": False, "allow_writing_files": False},
        return_last_fold_model=True,
        return_last_fold_cal_data=True,
    )

    assert proba is not None and y_cal is not None
    assert len(proba) >= MIN_GATE_TEST_MATCHES
    assert len(proba) != 10

    # The shipped MODEL still comes from the final fold, so it has never seen
    # the season it will predict. Only the calibration source moved.
    assert model is not None


def test_calibrating_on_ten_matches_leaves_the_classes_disagreeing():
    """Pins the MEASURED harm, which is not saturation.

    On the real 2026-2027 fold, two of three classes had under five positives,
    so they fell back to the exact identity map while the third was fit on ten
    points. At inference all three are applied together and renormalised, so one
    class is pulled toward a ten-sample curve while the others pass through
    untouched. The `binary.sum() < 5` guard does not prevent that — it creates it.
    """
    import numpy as np

    from scripts.models.retrain_no_odds_catboost import fit_isotonic_calibrators

    # Ten matches: one H, one D, eight A — the class imbalance a handful of
    # opening fixtures actually produces.
    y = np.array(["H"] + ["D"] + ["A"] * 8)
    rng = np.random.default_rng(1)
    proba = rng.dirichlet([2, 2, 2], size=10)

    cal = fit_isotonic_calibrators(y, proba)
    grid = np.linspace(0.05, 0.95, 19)

    # Precondition: H and D really are below the guard, A really is above it.
    assert (y == "H").sum() < 5 and (y == "D").sum() < 5
    assert (y == "A").sum() >= 5

    passthrough = {c: np.allclose(cal[c].predict(grid), grid) for c in range(3)}

    assert passthrough[0] and passthrough[1], "sparse classes should hit the identity fallback"
    assert not passthrough[2], "the populated class is genuinely fit, so it disagrees"
