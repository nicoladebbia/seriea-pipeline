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

import inspect
import json

import pytest

import scripts.pipeline.weekly_retrain as wr
from ml.evaluation import MIN_GATE_TEST_MATCHES


@pytest.fixture
def retrain(monkeypatch, tmp_path):
    """weekly_retrain binds MODELS_DIR at import, so patch it on the module."""
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


# ---------------------------------------------------------------------------
# The gate's own claims about itself
#
# full_retrain is shaped like a deployment gate but cannot be one: train_optimized
# writes each league's models to *_latest as it trains (ml/persistence.py:43), so
# by the time the comparison runs the new models are already serving. The `else`
# branch used to log "NOT promoting" and notify "Full retrain REJECTED", telling a
# human the bad model had been held back when it was live. These pin the wording,
# because here the wording IS the deliverable.
# ---------------------------------------------------------------------------

def _src(fn) -> str:
    """Source of one function — whole-file greps catch the sibling retrain path."""
    return inspect.getsource(fn)


def test_the_docstring_does_not_claim_to_be_a_deployment_gate():
    doc = wr.full_retrain.__doc__ or ""

    assert "Wraps train_optimized() with comparison gate and archival." not in doc, (
        "the old docstring claimed two things full_retrain does not do: it does no "
        "archival (auto_retrain does), and the comparison cannot gate"
    )
    assert "NOT a deployment gate" in doc
    assert "ALREADY SERVING" in doc


def test_full_retrain_does_not_report_a_rejection_it_did_not_make():
    """train_optimized saves every league, ensemble included, before this gate."""
    src = _src(wr.full_retrain)

    assert '"Full Retrain REJECTED"' not in src, (
        "reporting REJECTED implies the model was held back; it was not"
    )
    assert 'log.warning("NOT promoting — %s", reason)' not in src
    assert "ALREADY LIVE" in src
    assert "manual rollback required" in src
    assert "models_live_despite_failed_comparison" in src


def test_quick_retrain_says_which_half_of_the_promotion_was_withheld():
    """Its gate is PARTLY real: ens.save is withheld, *_latest is not."""
    src = _src(wr.quick_retrain)

    assert 'ens.save("universal")' in src, (
        "precondition: this test only makes sense while the ensemble save is the "
        "thing the promote branch actually gates"
    )
    assert 'log.warning("NOT promoting — %s", reason)' not in src, (
        "unqualified 'NOT promoting' hides that the individual models were "
        "already overwritten"
    )
    assert "NOT promoting the ensemble" in src
    assert "already overwritten" in src


def test_full_retrains_reported_metrics_say_which_league_they_measure():
    """One cv_results.json is shared by every league, so the last one wins."""
    src = _src(wr.full_retrain)

    assert '"New model: acc=%.4f  ll=%.4f  brier=%.4f"' not in src, (
        "an unqualified 'New model' line reads as both leagues' numbers when it "
        "is only whichever league ran last"
    )
    assert '"New model [%s only]: acc=%.4f  ll=%.4f  brier=%.4f"' in src
    assert 'result["metrics_league"] = metrics_league' in src


def test_quick_retrain_is_left_alone_because_it_trains_one_universal_model():
    """The league caveat is specific to full_retrain's per-league loop."""
    assert "for _league in ACTIVE_LEAGUES" in _src(wr.full_retrain)
    assert "for _league in ACTIVE_LEAGUES" not in _src(wr.quick_retrain)
