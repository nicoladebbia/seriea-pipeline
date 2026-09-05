"""O/U trainer promotion gate: a candidate that loses to the incumbent on the
shared holdout must NOT overwrite ou_*_catboost_latest.cbm.

Before this gate the trainer logged three quality checks and saved regardless
(both live O/U models shipped 2026-09-01 failing their calibration gate)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import scripts.models.train_over_under as tou

# ---------------------------------------------------------------------------
# decide_promotion — pure decision logic
# ---------------------------------------------------------------------------

def _m(ll, cal=0.02, brier=0.2):
    return {"log_loss": ll, "brier": brier, "calibration_gap": cal, "n": 500}


def test_no_incumbent_promotes_when_candidate_beats_naive():
    ok, reason = tou.decide_promotion(_m(0.60), None, naive_ll=0.69)
    assert ok and "no scorable incumbent" in reason


def test_no_incumbent_still_refuses_a_candidate_that_loses_to_naive():
    ok, reason = tou.decide_promotion(_m(0.70), None, naive_ll=0.69)
    assert not ok and "naive" in reason


def test_better_candidate_promotes():
    ok, reason = tou.decide_promotion(_m(0.60), _m(0.62), naive_ll=0.69)
    assert ok and "better" in reason


def test_worse_candidate_beyond_tolerance_is_refused():
    ok, reason = tou.decide_promotion(_m(0.63), _m(0.62), naive_ll=0.69)
    assert not ok and "worse than incumbent" in reason


def test_within_tolerance_promotes():
    ok, reason = tou.decide_promotion(_m(0.622), _m(0.620), naive_ll=0.69)
    assert ok and "within tolerance" in reason


def test_calibration_regression_beyond_tolerance_is_refused():
    # log-loss fine, calibration gap +0.03 worse than the incumbent
    ok, reason = tou.decide_promotion(_m(0.60, cal=0.06), _m(0.61, cal=0.03),
                                      naive_ll=0.69)
    assert not ok and "calibration" in reason


def test_gate_is_relative_not_a_fixed_bar():
    # Both fail the old fixed 0.03 calibration bar; the candidate is still
    # promotable because it is no worse than what production already runs.
    # (5b1e111: a fixed bar the incumbent itself fails froze prod 130 days.)
    ok, _ = tou.decide_promotion(_m(0.60, cal=0.055), _m(0.61, cal=0.055),
                                 naive_ll=0.69)
    assert ok


# ---------------------------------------------------------------------------
# file effects — tiny real CatBoost models on synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture
def toy():
    from catboost import CatBoostClassifier
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=["f1", "f2", "f3"])
    y = pd.Series((X["f1"] + 0.3 * rng.normal(size=120) > 0).astype(int))

    def fit(seed=0):
        m = CatBoostClassifier(iterations=8, depth=2, verbose=0, random_seed=seed)
        m.fit(X, y)
        return m
    return X, y, fit


def _write_latest(out_dir, line_str, model, feats):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_dir / f"ou_{line_str}_catboost_latest.cbm"))
    (out_dir / f"ou_{line_str}_catboost_metadata.json").write_text(
        json.dumps({"line": 2.5, "feature_names": feats}))


def test_refused_candidate_leaves_latest_untouched(tmp_path, toy):
    X, y, fit = toy
    incumbent = fit(seed=1)
    _write_latest(tmp_path, "2_5", incumbent, list(X.columns))
    before = (tmp_path / "ou_2_5_catboost_latest.cbm").read_bytes()

    paths = tou.persist_model(tmp_path, "2_5", fit(seed=2), {"line": 2.5},
                              promoted=False)

    assert (tmp_path / "ou_2_5_catboost_latest.cbm").read_bytes() == before
    assert (tmp_path / "candidate" / "ou_2_5_catboost_candidate.cbm").exists()
    assert paths["model"].endswith("candidate.cbm")
    # the live loader globs ou_*_catboost_metadata.json in the top dir only:
    # nothing new must appear there
    assert sorted(p.name for p in tmp_path.glob("ou_*_catboost_metadata.json")) == \
        ["ou_2_5_catboost_metadata.json"]


def test_promoted_candidate_replaces_latest_and_archives_incumbent(tmp_path, toy):
    X, y, fit = toy
    incumbent = fit(seed=1)
    _write_latest(tmp_path, "2_5", incumbent, list(X.columns))
    before = (tmp_path / "ou_2_5_catboost_latest.cbm").read_bytes()

    tou.persist_model(tmp_path, "2_5", fit(seed=2), {"line": 2.5, "tag": "new"},
                      promoted=True)

    after = (tmp_path / "ou_2_5_catboost_latest.cbm").read_bytes()
    assert after != before
    assert (tmp_path / "prev" / "ou_2_5_catboost_prev.cbm").read_bytes() == before
    meta = json.loads((tmp_path / "ou_2_5_catboost_metadata.json").read_text())
    assert meta["tag"] == "new"


def test_score_incumbent_returns_metrics_on_the_given_holdout(tmp_path, toy):
    X, y, fit = toy
    _write_latest(tmp_path, "2_5", fit(seed=1), list(X.columns))
    m = tou.score_incumbent(tmp_path, "2_5", X.iloc[80:], y.iloc[80:])
    assert m is not None
    assert set(m) >= {"log_loss", "brier", "calibration_gap", "n"}
    assert m["n"] == 40 and 0 < m["log_loss"] < 2


def test_score_incumbent_is_none_when_missing_or_unscorable(tmp_path, toy):
    X, y, fit = toy
    assert tou.score_incumbent(tmp_path, "2_5", X, y) is None          # no files
    _write_latest(tmp_path, "2_5", fit(seed=1), ["f1", "f2", "f3", "ghost"])
    assert tou.score_incumbent(tmp_path, "2_5", X, y) is None          # needs a feature we lack


def test_a_dry_run_never_promotes_but_says_what_it_would_have_done():
    """--dry-run previews the gate: _latest untouched, verdict kept in the reason."""
    promoted, reason = tou.dry_run_decision(True, "better than incumbent by 0.0081")
    assert promoted is False
    assert reason.startswith("DRY RUN — would PROMOTE:")
    promoted, reason = tou.dry_run_decision(False, "worse than incumbent")
    assert promoted is False
    assert "would HOLD" in reason
