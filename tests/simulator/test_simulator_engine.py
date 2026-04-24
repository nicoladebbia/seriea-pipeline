"""Tests for Phase 1 simulator engine (Dixon-Coles + simulate_match)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.simulator.engine.dixon_coles import (
    dc_correction,
    dc_joint_pmf,
    fit_tau_mle,
    sample_dc_goals,
)
from models.simulator.engine.simulator import simulate_match, MatchSimulation


# ---------------------------------------------------------------------------
# Dixon-Coles tests
# ---------------------------------------------------------------------------

def test_dc_correction_zero_tau_is_identity():
    for h in range(5):
        for a in range(5):
            assert dc_correction(h, a, 1.3, 1.1, 0.0) == 1.0


def test_dc_correction_boosts_low_scores_with_positive_tau():
    tau = 0.1
    # 1-0, 0-1 boosted
    assert dc_correction(1, 0, 1.3, 1.1, tau) > 1.0  # 1 + 1.1*0.1 = 1.11
    assert dc_correction(0, 1, 1.3, 1.1, tau) > 1.0  # 1 + 1.3*0.1 = 1.13
    # 0-0 weakened, 1-1 weakened
    assert dc_correction(0, 0, 1.3, 1.1, tau) < 1.0  # 1 - 1.3*1.1*0.1 = 0.857
    assert dc_correction(1, 1, 1.3, 1.1, tau) < 1.0  # 1 - 0.1 = 0.9
    # Other cells unchanged
    assert dc_correction(2, 1, 1.3, 1.1, tau) == 1.0
    assert dc_correction(3, 0, 1.3, 1.1, tau) == 1.0


def test_dc_joint_pmf_sums_to_one():
    for tau in [0.0, 0.05, -0.05, 0.1]:
        pmf = dc_joint_pmf(8, 8, lambda_h=1.3, lambda_a=1.1, tau=tau)
        assert abs(pmf.sum() - 1.0) < 1e-9


def test_dc_sampling_tau_zero_matches_independent_poisson():
    """With τ=0, accept-reject should match direct Poisson sampling in distribution."""
    rng = np.random.default_rng(42)
    n = 10_000
    lh = np.full(n, 1.5)
    la = np.full(n, 1.2)
    h, a = sample_dc_goals(lh, la, 0.0, rng)
    # Sample means should be near λ (within ~0.03 at n=10k)
    assert abs(h.mean() - 1.5) < 0.05
    assert abs(a.mean() - 1.2) < 0.05


def test_dc_sampling_positive_tau_boosts_low_scores():
    """With τ > 0, the 3-cell rate {0-0, 1-0, 0-1} should RISE vs independent Poisson."""
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    n = 20_000
    lh = np.full(n, 1.3)
    la = np.full(n, 1.1)
    h0, a0 = sample_dc_goals(lh, la, 0.0, rng1)
    h1, a1 = sample_dc_goals(lh, la, 0.08, rng2)
    rate_0 = float(((h0 == 0) & (a0 == 0)).mean() + ((h0 == 1) & (a0 == 0)).mean() + ((h0 == 0) & (a0 == 1)).mean())
    rate_1 = float(((h1 == 0) & (a1 == 0)).mean() + ((h1 == 1) & (a1 == 0)).mean() + ((h1 == 0) & (a1 == 1)).mean())
    assert rate_1 > rate_0


def test_fit_tau_recovers_synthetic():
    """On n=5000 matches drawn from DC with τ=0.06, MLE should recover ~0.06."""
    rng = np.random.default_rng(123)
    n = 5_000
    lh = rng.uniform(0.8, 2.2, size=n)
    la = rng.uniform(0.7, 2.0, size=n)
    h, a = sample_dc_goals(lh, la, tau=0.06, rng=rng)
    tau_hat = fit_tau_mle(h, a, lh, la)
    assert abs(tau_hat - 0.06) < 0.04, f"Got τ̂={tau_hat}, expected ~0.06"


def test_fit_tau_shrinks_to_zero_on_noise():
    """On a small sample with no DC structure, strong shrinkage should push τ̂ toward 0."""
    rng = np.random.default_rng(17)
    n = 100  # small sample
    lh = rng.uniform(1.0, 2.0, size=n)
    la = rng.uniform(1.0, 2.0, size=n)
    h = rng.poisson(lh)
    a = rng.poisson(la)
    tau_hat = fit_tau_mle(h, a, lh, la, shrinkage_prior_var=0.001)  # strong prior
    assert abs(tau_hat) < 0.10


# ---------------------------------------------------------------------------
# simulate_match tests
# ---------------------------------------------------------------------------

def test_simulate_match_reproducible_with_seed():
    s1 = simulate_match(1.3, 1.1, tau=0.05, n_trials=1000, seed=42)
    s2 = simulate_match(1.3, 1.1, tau=0.05, n_trials=1000, seed=42)
    np.testing.assert_array_equal(s1.home_goals, s2.home_goals)
    np.testing.assert_array_equal(s1.away_goals, s2.away_goals)


def test_simulate_match_different_seeds_give_different_streams():
    s1 = simulate_match(1.3, 1.1, n_trials=1000, seed=1)
    s2 = simulate_match(1.3, 1.1, n_trials=1000, seed=2)
    assert not np.array_equal(s1.home_goals, s2.home_goals)


def test_simulate_match_match_id_salts_seed():
    s1 = simulate_match(1.3, 1.1, n_trials=1000, seed=42, match_id="match_A")
    s2 = simulate_match(1.3, 1.1, n_trials=1000, seed=42, match_id="match_B")
    # Same base seed but different match_id → different streams
    assert not np.array_equal(s1.home_goals, s2.home_goals)


def test_1x2_probabilities_sum_to_one():
    sim = simulate_match(1.5, 1.2, n_trials=10_000, seed=42)
    sim.assert_consistency(tol=1e-9)


def test_over_under_complementarity():
    sim = simulate_match(1.5, 1.2, n_trials=10_000, seed=42)
    # P(over X) + P(under X) should be ≤ 1 (= 1 if X is non-integer; < 1 if X is integer — pushes)
    assert sim.p_over(2.5) + sim.p_under(2.5) == pytest.approx(1.0, abs=1e-9)


def test_expected_goals_approximately_match_lambda():
    sim = simulate_match(1.7, 1.1, tau=0.0, n_trials=50_000, seed=42)
    assert abs(sim.home_goals.mean() - 1.7) < 0.03
    assert abs(sim.away_goals.mean() - 1.1) < 0.03


def test_btts_probability_reasonable():
    sim = simulate_match(1.5, 1.5, n_trials=10_000, seed=42)
    p_btts = sim.p_btts()
    # For equal λ=1.5, P(BTTS) ≈ (1 - e^-1.5)^2 = 0.598
    assert 0.55 < p_btts < 0.65


def test_exact_score_probabilities_summable():
    sim = simulate_match(1.3, 1.1, n_trials=10_000, seed=42)
    total = sum(p for _, p in sim.top_scores(k=20))
    assert total > 0.9  # top 20 scores should cover most of the distribution


def test_handicap_zero_matches_p_home_win():
    sim = simulate_match(1.5, 1.5, n_trials=10_000, seed=42)
    # AH 0 home → home must win outright (home + 0 > away ⟺ home > away)
    assert abs(sim.p_handicap(0.0, "home") - sim.p_home_win()) < 1e-9


def test_double_chance_consistent():
    sim = simulate_match(1.3, 1.1, n_trials=10_000, seed=42)
    # 1X = P(home) + P(draw)
    assert abs(sim.p_double_chance("1X") - (sim.p_home_win() + sim.p_draw())) < 1e-9
    # 12 = P(home) + P(away) = 1 - P(draw)
    assert abs(sim.p_double_chance("12") - (1.0 - sim.p_draw())) < 1e-9


def test_clean_sheet_matches_poisson_zero():
    sim = simulate_match(1.5, 1.2, tau=0.0, n_trials=20_000, seed=42)
    # Home clean sheet = away never scores = P(away_goals = 0)
    # For λ_away=1.2, expected = e^-1.2 ≈ 0.301
    p_cs_home = sim.p_clean_sheet("home")
    assert abs(p_cs_home - np.exp(-1.2)) < 0.02


def test_tau_fit_on_simulator_output():
    """Round-trip: simulate with known τ, recover via MLE."""
    tau_true = 0.08
    rng = np.random.default_rng(0)
    lh_arr = rng.uniform(0.8, 2.0, size=3_000)
    la_arr = rng.uniform(0.7, 1.8, size=3_000)
    h, a = sample_dc_goals(lh_arr, la_arr, tau=tau_true, rng=rng)
    tau_hat = fit_tau_mle(h, a, lh_arr, la_arr)
    assert abs(tau_hat - tau_true) < 0.04


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
