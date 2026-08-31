"""The persisted OOF eval artifact must never contain in-sample-calibrated probs.

Found 2026-08-28: WeightedAverageEnsemble.fit() fitted its blend calibrator on
the full OOF set (AutoCalibrator refits the winning method on ALL rows), then
applied it back to that same set and persisted the result as prob_H/D/A in
cv_predictions.parquet. On the 2026-08-25 Serie A run this inflated OOF argmax
accuracy 0.531 -> 0.565 — above the project's own ">56% is leakage or fiction"
line — and tests/test_historical_accuracy.py graded the model from those
columns. A calibrator cannot add accuracy out-of-sample; the gain was pure
in-sample memorization (isotonic won the method selection on that run).

Contract pinned here: prob_* == raw_prob_* in the persisted parquet, byte-equal.
Calibrated output exists only for NEW matches at inference time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ml.ensemble as E


@pytest.fixture
def synthetic_matches():
    """Seven 60-match seasons (min_train_seasons=5 needs >5 to yield folds),
    weakly predictive feature, imbalanced outcomes."""
    rng = np.random.default_rng(7)
    n_per = 60
    seasons = [f"20{y}-20{y + 1}" for y in range(18, 25)]
    rows = []
    for season in seasons:
        for _ in range(n_per):
            strength = rng.normal(0, 1)
            p_h = 1 / (1 + np.exp(-strength))
            r = rng.random()
            actual = "H" if r < 0.55 * p_h + 0.2 else ("D" if r < 0.75 else "A")
            rows.append({"_season": season, "strength": strength,
                         "noise": rng.normal(0, 1), "actual": actual})
    df = pd.DataFrame(rows)
    X = df[["_season", "strength", "noise"]].copy()
    y = df["actual"].copy()
    return X, y


def test_persisted_oof_is_raw_not_insample_calibrated(synthetic_matches, tmp_path, monkeypatch):
    X, y = synthetic_matches

    monkeypatch.setattr(E, "MODELS_DIR", tmp_path)
    # Shrink the fold guard so 60-match synthetic seasons produce folds at all.
    real_cfg = E.ValidationConfig

    def small_cfg(*a, **k):
        cfg = real_cfg(*a, **k)
        cfg.min_test_matches = 10
        return cfg

    monkeypatch.setattr(E, "ValidationConfig", small_cfg)

    ens = E.WeightedAverageEnsemble(
        {"xgboost": {"n_estimators": 8, "max_depth": 2, "learning_rate": 0.3}},
        use_sample_weights=False,
        variant="synthetic",
    )
    ens.fit(X, y, feature_names=["strength", "noise"])

    out = tmp_path / "synthetic" / "cv_predictions.parquet"
    assert out.exists(), "fit() no longer persists OOF predictions"
    d = pd.read_parquet(out)

    prob = d[["prob_H", "prob_D", "prob_A"]].to_numpy()
    raw = d[["raw_prob_H", "raw_prob_D", "raw_prob_A"]].to_numpy()

    # True positive (the broken version fails exactly here): the fitted
    # calibrator is non-identity on these rows, so persisting its output as
    # prob_* would be detectably different from raw.
    calibrated = E._normalize_probs(ens.blend_calibrator.calibrate(raw))
    assert not np.allclose(calibrated, raw, atol=1e-6), (
        "calibrator degenerated to identity — this test can no longer "
        "discriminate honest from in-sample-calibrated persistence"
    )

    assert np.array_equal(prob, raw), (
        "prob_* differs from raw_prob_*: an in-sample-calibrated set is being "
        "persisted as the eval artifact again"
    )
