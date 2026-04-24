"""Tests for Phase 2 rate estimators + simulator extensions."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.simulator.base_rates.corner_rates import (
    CornerRateEstimator,
    DEFAULT_CORNER_FEATURES,
)
from models.simulator.base_rates.card_rates import CardRateEstimator
from models.simulator.base_rates.shot_generator import ShotRateEstimator
from models.simulator.engine.simulator import simulate_match
from models.simulator.markets import all_phase2_market_probs


def _fake_train(n=500, seed=7):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "match_id": [f"m_{i}" for i in range(n)],
        "season": ["2023-2024"] * n,
        "league": ["serie_a"] * n,
        "home_team": [f"T{i%10}" for i in range(n)],
        "away_team": [f"T{(i+1)%10}" for i in range(n)],
        "match_date": pd.date_range("2024-08-01", periods=n, freq="D"),
        "home_score": rng.poisson(1.3, n),
        "away_score": rng.poisson(1.1, n),
        "home_corners": rng.poisson(5.0, n),
        "away_corners": rng.poisson(4.5, n),
        "home_yellow_cards": rng.poisson(2.0, n),
        "away_yellow_cards": rng.poisson(2.2, n),
        "home_red_cards": rng.binomial(1, 0.05, n),
        "away_red_cards": rng.binomial(1, 0.05, n),
        "home_shots_total": rng.poisson(12.0, n),
        "away_shots_total": rng.poisson(11.5, n),
        "home_shots_on_target": None,
        "away_shots_on_target": None,
        "ref_strictness_score": rng.uniform(0.3, 0.8, n),
        "is_derby": rng.binomial(1, 0.1, n),
        "league_position_diff": rng.normal(0, 5, n),
    })
    # Fill derived
    df["home_shots_on_target"] = np.clip((df["home_shots_total"] * 0.35).astype(int), 0, None)
    df["away_shots_on_target"] = np.clip((df["away_shots_total"] * 0.33).astype(int), 0, None)
    for f in DEFAULT_CORNER_FEATURES:
        if f not in df.columns:
            df[f] = rng.normal(1.0, 0.2, n)
    for f in ("home_roll_5_yellow_cards", "away_roll_5_yellow_cards",
              "home_roll_10_yellow_cards", "away_roll_10_yellow_cards",
              "home_roll_5_fouls", "away_roll_5_fouls"):
        df[f] = rng.normal(2.0, 0.5, n)
    return df


# ---------------------------------------------------------------------------
# Corner rate estimator
# ---------------------------------------------------------------------------

def test_corner_estimator_fits_and_predicts():
    df = _fake_train()
    est = CornerRateEstimator()
    est.fit(df)
    assert est.is_fit
    rate_h, rate_a = est.predict(df.iloc[:50])
    assert rate_h.shape == (50,)
    assert rate_a.shape == (50,)
    # Clamped to [0.5, 15]
    assert np.all(rate_h >= 0.5) and np.all(rate_h <= 15.0)
    assert np.all(rate_a >= 0.5) and np.all(rate_a <= 15.0)


def test_corner_estimator_mean_near_training_mean():
    df = _fake_train()
    est = CornerRateEstimator()
    est.fit(df)
    rate_h, rate_a = est.predict(df)
    # Training means are ~5 and ~4.5; predicted means should track
    assert abs(rate_h.mean() - df["home_corners"].mean()) < 1.0
    assert abs(rate_a.mean() - df["away_corners"].mean()) < 1.0


def test_corner_estimator_fallback_without_fit():
    est = CornerRateEstimator()
    df = pd.DataFrame({"home_team": ["a"]})
    rate_h, rate_a = est.predict(df)
    # Both fall back to 4.5 when unfit
    assert rate_h[0] == 4.5
    assert rate_a[0] == 4.5


# ---------------------------------------------------------------------------
# Card rate estimator
# ---------------------------------------------------------------------------

def test_card_estimator_fits_and_predicts():
    df = _fake_train()
    est = CardRateEstimator()
    est.fit(df)
    assert est.is_fit
    rate_h, rate_a = est.predict(df.iloc[:50])
    assert rate_h.shape == (50,)
    # Sane bounds
    assert np.all(rate_h >= 0.2) and np.all(rate_h <= 8.0)


def test_card_estimator_ref_scaling_amplifies_strict_refs():
    df = _fake_train()
    est = CardRateEstimator()
    est.fit(df)
    # Make a strict-ref version and a lax-ref version
    strict = df.iloc[:20].copy()
    strict["ref_strictness_score"] = 1.0
    lax = df.iloc[:20].copy()
    lax["ref_strictness_score"] = 0.0
    strict_h, _ = est.predict(strict)
    lax_h, _ = est.predict(lax)
    # Strict refs should produce more cards on the same matches
    assert strict_h.mean() > lax_h.mean()


def test_card_estimator_can_disable_ref_scaling():
    df = _fake_train()
    est = CardRateEstimator(use_ref_scaling=False)
    est.fit(df)
    strict = df.iloc[:20].copy()
    strict["ref_strictness_score"] = 1.0
    neutral = df.iloc[:20].copy()
    neutral["ref_strictness_score"] = 0.5
    strict_h, _ = est.predict(strict)
    neutral_h, _ = est.predict(neutral)
    # Without scaling, refs don't matter
    np.testing.assert_allclose(strict_h, neutral_h)


# ---------------------------------------------------------------------------
# Shot generator
# ---------------------------------------------------------------------------

def test_shot_estimator_fits_and_gives_sot_ratios():
    df = _fake_train()
    est = ShotRateEstimator()
    est.fit(df)
    assert est.is_fit
    h, a = est.sot_ratios()
    assert 0.25 <= h <= 0.55
    assert 0.25 <= a <= 0.55


def test_shot_estimator_predicts_reasonable_rates():
    df = _fake_train()
    est = ShotRateEstimator()
    est.fit(df)
    rate_h, rate_a = est.predict(df.iloc[:50])
    # Mean should be in plausible range (8-18 shots per team)
    assert 5.0 <= rate_h.mean() <= 20.0
    assert 5.0 <= rate_a.mean() <= 20.0


# ---------------------------------------------------------------------------
# simulate_match integration
# ---------------------------------------------------------------------------

def test_simulate_match_populates_phase2_arrays_when_rates_given():
    sim = simulate_match(
        lambda_home=1.3, lambda_away=1.1,
        n_trials=1000, seed=42,
        corner_rate_home=5.0, corner_rate_away=4.5,
        card_rate_home=2.0, card_rate_away=2.2,
        shot_rate_home=12.0, shot_rate_away=11.5,
        sot_ratio_home=0.35, sot_ratio_away=0.33,
    )
    assert sim.home_corners is not None and len(sim.home_corners) == 1000
    assert sim.home_cards is not None and len(sim.home_cards) == 1000
    assert sim.home_shots is not None and len(sim.home_shots) == 1000
    assert sim.home_sot is not None and len(sim.home_sot) == 1000


def test_simulate_match_leaves_phase2_empty_when_rates_none():
    sim = simulate_match(lambda_home=1.3, lambda_away=1.1, n_trials=500, seed=42)
    assert sim.home_corners is None
    assert sim.home_cards is None
    assert sim.home_shots is None
    assert sim.home_sot is None


def test_p_corners_over_raises_without_data():
    sim = simulate_match(lambda_home=1.3, lambda_away=1.1, n_trials=500, seed=42)
    with pytest.raises(AttributeError):
        sim.p_corners_over(9.5)


def test_p_corners_over_returns_sane_value():
    sim = simulate_match(
        1.3, 1.1, n_trials=10_000, seed=42,
        corner_rate_home=5.0, corner_rate_away=4.5,
    )
    # Total corner rate = 9.5 → P(over 9.5) ≈ 0.49 for Poisson(9.5)
    p = sim.p_corners_over(9.5)
    assert 0.35 < p < 0.60


def test_expected_corners_matches_poisson_mean():
    sim = simulate_match(
        1.3, 1.1, n_trials=20_000, seed=42,
        corner_rate_home=5.0, corner_rate_away=4.5,
    )
    # Sum of two Poissons has mean = sum of means
    assert abs(sim.expected_corners("both") - 9.5) < 0.2
    assert abs(sim.expected_corners("home") - 5.0) < 0.2
    assert abs(sim.expected_corners("away") - 4.5) < 0.2


def test_sot_binomial_respects_shot_and_ratio():
    sim = simulate_match(
        1.3, 1.1, n_trials=20_000, seed=42,
        shot_rate_home=20.0, shot_rate_away=15.0,
        sot_ratio_home=0.40, sot_ratio_away=0.30,
    )
    # Mean SOT should be ~8 home, ~4.5 away
    assert abs(sim.home_sot.mean() - 8.0) < 0.3
    assert abs(sim.away_sot.mean() - 4.5) < 0.3
    # Every SOT ≤ matching shot count
    assert np.all(sim.home_sot <= sim.home_shots)
    assert np.all(sim.away_sot <= sim.away_shots)


def test_all_phase2_market_probs_empty_for_goals_only():
    sim = simulate_match(1.3, 1.1, n_trials=500, seed=42)
    out = all_phase2_market_probs(sim)
    assert out == {}


def test_all_phase2_market_probs_emits_corner_keys_when_populated():
    sim = simulate_match(
        1.3, 1.1, n_trials=5000, seed=42,
        corner_rate_home=5.0, corner_rate_away=4.5,
        card_rate_home=2.0, card_rate_away=2.2,
    )
    out = all_phase2_market_probs(sim)
    assert "corners_over_9_5" in out
    assert "corners_over_8_5" in out
    assert "cards_over_3_5" in out
    assert "home_corners_over_4_5" in out
    # All probabilities must be in [0,1]
    for k, v in out.items():
        assert 0.0 <= v <= 1.0, f"{k}={v} out of range"


def test_simulator_predictor_supports_phase2_markets():
    from models.simulator.backtests.simulator_predictor import SimulatorPredictor
    p = SimulatorPredictor()
    assert p.supports("corners_over_9_5")
    assert p.supports("cards_over_3_5")
    assert p.supports("shots_over_20_5")
    assert p.supports("home_corners_over_4_5")
    # Phase 1 markets still work
    assert p.supports("O/U 2.5")
    assert p.supports("1X2")


def test_simulator_predictor_phase2_disabled_skips_corners():
    from models.simulator.backtests.simulator_predictor import SimulatorPredictor
    p = SimulatorPredictor(enable_phase2_rates=False)
    assert not p.supports("corners_over_9_5")
    assert not p.supports("cards_over_3_5")
    # Phase 1 still supported
    assert p.supports("O/U 2.5")
