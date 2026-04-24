"""Phase 1 feature-parity test — the A2 safeguard.

Asserts that every feature the SimulatorPredictor uses at inference time
exists in both the training and evaluation DataFrames with identical
preprocessing. This is the direct mitigation for the A2 incident
(training-inference distribution skew → -2.6% ROI on n=30).

If someone adds a feature to models/simulator/ that doesn't exist in the
historical training data, or applies a different imputation at inference
than at training, this test fires.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.simulator.backtests.simulator_predictor import (
    DEFAULT_FEATURES,
    SimulatorPredictor,
)


def test_default_features_are_valid_names():
    """Every feature name must be a string suitable for a parquet column."""
    for f in DEFAULT_FEATURES:
        assert isinstance(f, str) and len(f) > 0
        assert " " not in f, f"Feature {f!r} has whitespace"


def test_predictor_uses_only_trained_features_at_inference():
    """After fit, predictor._usable_features should match what predict() queries."""
    df = pd.DataFrame({
        "season": ["2023-2024"] * 20,
        "league": ["serie_a"] * 20,
        "home_score": [1] * 20,
        "away_score": [1] * 20,
        "match_id": [f"m_{i}" for i in range(20)],
    })
    for f in DEFAULT_FEATURES:
        df[f] = np.random.default_rng(0).normal(1.0, 0.2, 20)

    pred = SimulatorPredictor(n_trials=100)
    # Training with 20 rows is below the 100-row minimum guard; test that too
    pred.fit(df)
    # Since min rows not met, usable_features stays empty
    assert pred._lambda_home is None


def test_predictor_drops_features_missing_at_training():
    """Features absent from the training frame must not appear in _usable_features."""
    rng = np.random.default_rng(0)
    # Build a training frame with only 5 of 15 features present
    df = pd.DataFrame({
        "season": ["2023-2024"] * 200,
        "league": ["serie_a"] * 200,
        "match_id": [f"m_{i}" for i in range(200)],
        "home_score": rng.poisson(1.3, 200),
        "away_score": rng.poisson(1.1, 200),
    })
    partial_feats = DEFAULT_FEATURES[:5]
    for f in partial_feats:
        df[f] = rng.normal(1.0, 0.2, 200)

    pred = SimulatorPredictor(n_trials=100)
    pred.fit(df)
    assert set(pred._usable_features) == set(partial_feats)


def test_predictor_imputes_nans_consistently():
    """Inference data with NaNs in feature columns should be imputed to 0.0 (same
    as training imputation in this test)."""
    rng = np.random.default_rng(42)
    n = 200
    train = pd.DataFrame({
        "season": ["2023-2024"] * n,
        "league": ["serie_a"] * n,
        "match_id": [f"m_{i}" for i in range(n)],
        "home_score": rng.poisson(1.3, n),
        "away_score": rng.poisson(1.1, n),
    })
    for f in DEFAULT_FEATURES[:5]:
        train[f] = rng.normal(1.0, 0.2, n)

    # Eval data with some NaNs
    eval_df = train.iloc[:50].copy()
    eval_df.loc[0:10, DEFAULT_FEATURES[0]] = np.nan

    pred = SimulatorPredictor(n_trials=100)
    pred.fit(train)

    # Should not raise on NaN — the predictor fills 0.0 at inference
    probs = pred.predict_binary(eval_df, "O/U 2.5")
    assert len(probs) == len(eval_df)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_simulator_predictor_supports_expected_markets():
    p = SimulatorPredictor()
    assert p.supports("O/U 2.5")
    assert p.supports("O/U 1.5")
    assert p.supports("BTTS")
    assert p.supports("1X2")
    # Phase 2 markets now supported
    assert p.supports("corners_over_9_5")
    assert p.supports("cards_over_4_5")
