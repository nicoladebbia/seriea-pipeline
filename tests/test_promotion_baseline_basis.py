"""The promotion baseline must be measured on the same basis as the new model.

`weekly_retrain` promotes on `diff = new_ll - old_ll`. The old number is read
from a stored `cv_results.json` written by an EARLIER run. Once undersized folds
stopped being generated, a baseline that still contains one is not comparable —
the diff then reports a change in measurement basis as a change in model
quality (debugging.md #9, cross-condition comparison). Measured on the real
2026-08-25 Serie A ensemble folds, that phantom is 0.9756 -> 0.9640, a 0.0116
"improvement" nobody earned.

Since cv_results.json stores per-fold `n_test`, the baseline can simply be
recomputed on the current basis instead.
"""

from __future__ import annotations

import json

import pytest

from ml.evaluation import MIN_GATE_TEST_MATCHES


@pytest.fixture
def retrain(monkeypatch, tmp_path):
    """weekly_retrain binds MODELS_DIR at import, so patch it on the module."""
    import scripts.pipeline.weekly_retrain as wr

    monkeypatch.setattr(wr, "MODELS_DIR", tmp_path)
    (tmp_path / "universal" / "ensemble").mkdir(parents=True)
    return wr


def _write(retrain_mod, folds):
    p = retrain_mod.MODELS_DIR / "universal" / "ensemble" / "cv_results.json"
    p.write_text(json.dumps(folds))


def _fold(season, n_test, ll, acc=0.52):
    return {
        "fold": 0,
        "test": season,
        "n_test": n_test,
        "ensemble_log_loss": ll,
        "ensemble_accuracy": acc,
        "ensemble_brier": 0.19,
    }


def test_a_stored_ten_match_fold_does_not_move_the_baseline(retrain):
    """The real 2026-08-25 Serie A ensemble folds, tiny one included."""
    folds = [
        _fold("2022-2023", 760, 1.006),
        _fold("2023-2024", 760, 0.995),
        _fold("2024-2025", 760, 0.930),
        _fold("2025-2026", 689, 0.925),
        _fold("2026-2027", 10, 1.022),
    ]
    _write(retrain, folds)

    # True positive: averaging all five really does shift the baseline, so the
    # assertion below is not vacuous.
    naive = sum(f["ensemble_log_loss"] for f in folds) / 5
    honest = sum(f["ensemble_log_loss"] for f in folds[:4]) / 4
    assert abs(naive - honest) > 0.01

    metrics = retrain._load_current_cv_metrics()

    assert metrics["ensemble_log_loss"] == pytest.approx(honest, abs=1e-6)
    assert metrics["ensemble_log_loss"] != pytest.approx(naive, abs=1e-6)


def test_a_baseline_of_full_folds_is_unchanged(retrain):
    folds = [_fold(f"20{20 + i}-20{21 + i}", 760, 1.0 + i / 100) for i in range(4)]
    _write(retrain, folds)

    metrics = retrain._load_current_cv_metrics()

    assert metrics["ensemble_log_loss"] == pytest.approx(
        sum(f["ensemble_log_loss"] for f in folds) / 4, abs=1e-6
    )


def test_a_baseline_with_no_usable_fold_still_returns_something(retrain, caplog):
    """An all-tiny historical file must not silently become None and turn a
    comparison into 'no baseline, promote by default'."""
    _write(retrain, [_fold("2026-2027", 10, 1.02)])

    with caplog.at_level("WARNING"):
        metrics = retrain._load_current_cv_metrics()

    assert metrics is not None
    assert metrics["ensemble_log_loss"] == pytest.approx(1.02, abs=1e-6)


def test_a_legacy_baseline_without_n_test_is_still_readable(retrain):
    """Older files predate the n_test field; they must not become unreadable."""
    folds = [{"fold": i, "ensemble_log_loss": 1.0, "ensemble_accuracy": 0.5}
             for i in range(3)]
    _write(retrain, folds)

    metrics = retrain._load_current_cv_metrics()

    assert metrics is not None
    assert metrics["ensemble_log_loss"] == pytest.approx(1.0, abs=1e-6)


def test_the_threshold_used_here_matches_the_one_the_trainers_gate_on():
    import scripts.pipeline.weekly_retrain as wr

    assert wr.MIN_GATE_TEST_MATCHES == MIN_GATE_TEST_MATCHES
