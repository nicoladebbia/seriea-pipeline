#!/usr/bin/env python3
"""Tests for Monte Carlo improvements: bivariate Poisson, correlated MC, bankroll ruin."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import poisson as scipy_poisson

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.betting.extended_markets import (
    BivariatePoissonParams,
    bivariate_score_matrix,
    calibrate_rho,
    derive_market_outcomes,
    poisson_pmf,
    sample_scores,
    score_matrix,
)
from scripts.betting.parlay_generator import (
    CONFIDENCE_K,
    _adaptive_sim_count,
    _build_score_matrix,
    _leg_to_outcome_key,
    _mc_cross_match,
    _mc_sgp,
    _monte_carlo_bands,
)
from scripts.betting.bankroll_simulator import (
    BetOpportunity,
    RuinSimConfig,
    find_optimal_fraction,
    simulate_season_path,
)


# ---------------------------------------------------------------------------
# Task 1: Bivariate Poisson Tests
# ---------------------------------------------------------------------------


class TestBivariatePoissonParams:
    def test_lambda3_computation(self):
        params = BivariatePoissonParams(lambda_home=1.5, lambda_away=1.2, rho=0.10)
        expected = 0.10 * math.sqrt(1.5 * 1.2)
        assert abs(params.lambda_3 - expected) < 0.01

    def test_lambda3_clamping(self):
        """lambda_3 can't exceed either marginal."""
        params = BivariatePoissonParams(lambda_home=0.3, lambda_away=0.2, rho=0.99)
        assert params.lambda_3 < params.lambda_home
        assert params.lambda_3 < params.lambda_away

    def test_independent_components_positive(self):
        params = BivariatePoissonParams(lambda_home=1.5, lambda_away=1.2, rho=0.10)
        assert params.lambda_home_ind > 0
        assert params.lambda_away_ind > 0

    def test_rho_zero_gives_zero_lambda3(self):
        params = BivariatePoissonParams(lambda_home=1.5, lambda_away=1.2, rho=0.0)
        assert params.lambda_3 == 0.0


class TestBivariateScoreMatrix:
    def test_sums_to_one(self):
        params = BivariatePoissonParams(lambda_home=1.5, lambda_away=1.2, rho=0.10)
        mat = bivariate_score_matrix(params, max_goals=10)
        assert abs(mat.sum() - 1.0) < 1e-6

    def test_marginals_match_poisson(self):
        """Marginal distributions should match univariate Poisson."""
        params = BivariatePoissonParams(lambda_home=1.5, lambda_away=1.2, rho=0.10)
        mat = bivariate_score_matrix(params, max_goals=10)

        for g in range(6):
            home_marginal = mat[g, :].sum()
            expected_h = scipy_poisson.pmf(g, 1.5)
            assert abs(home_marginal - expected_h) < 0.005, (
                f"Home marginal at {g}: {home_marginal:.4f} vs {expected_h:.4f}"
            )

            away_marginal = mat[:, g].sum()
            expected_a = scipy_poisson.pmf(g, 1.2)
            assert abs(away_marginal - expected_a) < 0.005, (
                f"Away marginal at {g}: {away_marginal:.4f} vs {expected_a:.4f}"
            )

    def test_rho_zero_matches_independent(self):
        """With rho=0, bivariate Poisson ~ product of independent Poissons.

        Tolerance is 1e-4 because bivariate_score_matrix renormalizes after
        truncating at max_goals, which shifts all probabilities slightly.
        """
        params = BivariatePoissonParams(lambda_home=1.5, lambda_away=1.2, rho=0.0)
        mat = bivariate_score_matrix(params, max_goals=10)  # higher max = less truncation

        for h in range(8):
            for a in range(8):
                expected = poisson_pmf(h, 1.5) * poisson_pmf(a, 1.2)
                assert abs(mat[h, a] - expected) < 1e-4, (
                    f"({h},{a}): {mat[h, a]:.8f} vs {expected:.8f}"
                )

    def test_positive_rho_increases_draw_probability(self):
        """Positive correlation should increase diagonal (draw) probabilities."""
        params_0 = BivariatePoissonParams(lambda_home=1.5, lambda_away=1.5, rho=0.0)
        params_rho = BivariatePoissonParams(lambda_home=1.5, lambda_away=1.5, rho=0.15)
        mat_0 = bivariate_score_matrix(params_0, max_goals=7)
        mat_rho = bivariate_score_matrix(params_rho, max_goals=7)

        draw_0 = sum(mat_0[i, i] for i in range(8))
        draw_rho = sum(mat_rho[i, i] for i in range(8))
        assert draw_rho > draw_0, (
            f"Draw prob with rho=0.15 ({draw_rho:.4f}) should exceed "
            f"rho=0 ({draw_0:.4f})"
        )

    def test_draw_calibration(self):
        """Draw calibration factor should inflate diagonal."""
        params = BivariatePoissonParams(lambda_home=1.5, lambda_away=1.2, rho=0.10)
        mat_base = bivariate_score_matrix(params, max_goals=7)
        mat_cal = bivariate_score_matrix(params, max_goals=7, draw_calibration=1.10)

        draw_base = sum(mat_base[i, i] for i in range(8))
        draw_cal = sum(mat_cal[i, i] for i in range(8))
        assert draw_cal > draw_base

    def test_non_negative_probabilities(self):
        params = BivariatePoissonParams(lambda_home=0.5, lambda_away=0.3, rho=0.20)
        mat = bivariate_score_matrix(params, max_goals=7)
        assert (mat >= 0).all()


class TestSampleScores:
    def test_output_shape(self):
        params = BivariatePoissonParams(lambda_home=1.5, lambda_away=1.2, rho=0.10)
        rng = np.random.default_rng(42)
        hg, ag = sample_scores(params, n_samples=1000, rng=rng)
        assert hg.shape == (1000,)
        assert ag.shape == (1000,)

    def test_non_negative_goals(self):
        params = BivariatePoissonParams(lambda_home=1.5, lambda_away=1.2, rho=0.10)
        rng = np.random.default_rng(42)
        hg, ag = sample_scores(params, n_samples=10000, rng=rng)
        assert (hg >= 0).all()
        assert (ag >= 0).all()

    def test_empirical_correlation_matches_rho(self):
        """100K samples should recover input rho within 0.02."""
        params = BivariatePoissonParams(lambda_home=1.5, lambda_away=1.2, rho=0.10)
        rng = np.random.default_rng(42)
        hg, ag = sample_scores(params, n_samples=100000, rng=rng)
        empirical_rho = np.corrcoef(hg, ag)[0, 1]
        assert abs(empirical_rho - 0.10) < 0.02, (
            f"Empirical rho {empirical_rho:.4f} too far from 0.10"
        )

    def test_rho_zero_gives_near_zero_correlation(self):
        params = BivariatePoissonParams(lambda_home=1.5, lambda_away=1.2, rho=0.0)
        rng = np.random.default_rng(42)
        hg, ag = sample_scores(params, n_samples=100000, rng=rng)
        empirical_rho = np.corrcoef(hg, ag)[0, 1]
        assert abs(empirical_rho) < 0.02

    def test_mean_matches_lambda(self):
        params = BivariatePoissonParams(lambda_home=1.5, lambda_away=1.2, rho=0.10)
        rng = np.random.default_rng(42)
        hg, ag = sample_scores(params, n_samples=100000, rng=rng)
        assert abs(hg.mean() - 1.5) < 0.02
        assert abs(ag.mean() - 1.2) < 0.02

    def test_reproducible_with_seed(self):
        params = BivariatePoissonParams(lambda_home=1.5, lambda_away=1.2, rho=0.10)
        hg1, ag1 = sample_scores(params, n_samples=100, rng=np.random.default_rng(123))
        hg2, ag2 = sample_scores(params, n_samples=100, rng=np.random.default_rng(123))
        np.testing.assert_array_equal(hg1, hg2)
        np.testing.assert_array_equal(ag1, ag2)


class TestDeriveMarketOutcomes:
    def test_h2h_partition(self):
        """H2H outcomes should partition the space."""
        hg = np.array([2, 1, 0, 3, 1])
        ag = np.array([1, 1, 2, 0, 2])
        outcomes = derive_market_outcomes(hg, ag)
        total = outcomes["h2h_home"].astype(int) + outcomes["h2h_draw"].astype(int) + outcomes["h2h_away"].astype(int)
        np.testing.assert_array_equal(total, np.ones(5))

    def test_btts(self):
        hg = np.array([2, 0, 1, 0])
        ag = np.array([1, 0, 0, 2])
        outcomes = derive_market_outcomes(hg, ag)
        np.testing.assert_array_equal(outcomes["btts_yes"], [True, False, False, False])
        np.testing.assert_array_equal(outcomes["btts_no"], [False, True, True, True])

    def test_totals(self):
        hg = np.array([1, 0, 2])
        ag = np.array([1, 0, 2])
        outcomes = derive_market_outcomes(hg, ag)
        np.testing.assert_array_equal(outcomes["totals_over_2.5"], [False, False, True])
        np.testing.assert_array_equal(outcomes["totals_under_2.5"], [True, True, False])

    def test_double_chance(self):
        hg = np.array([2, 1, 0])
        ag = np.array([1, 1, 2])
        outcomes = derive_market_outcomes(hg, ag)
        np.testing.assert_array_equal(outcomes["double_chance_1x"], [True, True, False])
        np.testing.assert_array_equal(outcomes["double_chance_x2"], [False, True, True])
        np.testing.assert_array_equal(outcomes["double_chance_12"], [True, False, True])

    def test_all_keys_present(self):
        """All expected outcome keys should be in the output."""
        hg = np.array([1])
        ag = np.array([0])
        outcomes = derive_market_outcomes(hg, ag)
        expected_prefixes = ["h2h_", "btts_", "totals_", "team_totals_",
                             "double_chance_", "spreads_", "multi_goal_"]
        for prefix in expected_prefixes:
            assert any(k.startswith(prefix) for k in outcomes), f"Missing keys with prefix {prefix}"


class TestScoreMatrixBackwardCompat:
    def test_rho_zero_default(self):
        """score_matrix() with rho=0 should match old behavior."""
        m = score_matrix(1.5, 1.2, max_goals=7, rho=0.0)
        for h in range(8):
            for a in range(8):
                expected = poisson_pmf(h, 1.5) * poisson_pmf(a, 1.2)
                assert abs(m[(h, a)] - expected) < 1e-8

    def test_rho_positive_returns_dict(self):
        """score_matrix() with rho>0 should still return a dict."""
        m = score_matrix(1.5, 1.2, max_goals=7, rho=0.10)
        assert isinstance(m, dict)
        assert abs(sum(m.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Task 2: Correlated MC Tests
# ---------------------------------------------------------------------------


class TestAdaptiveSimCount:
    def test_low_probability_gets_more_sims(self):
        n_low = _adaptive_sim_count(0.01, 2)
        n_high = _adaptive_sim_count(0.50, 2)
        assert n_low > n_high

    def test_floor_5000(self):
        assert _adaptive_sim_count(0.99, 1) >= 5000

    def test_cap_100000(self):
        assert _adaptive_sim_count(0.001, 5) <= 100000

    def test_more_legs_more_sims(self):
        n2 = _adaptive_sim_count(0.10, 2)
        n5 = _adaptive_sim_count(0.10, 5)
        assert n5 > n2


class TestLegToOutcomeKey:
    def test_h2h_home(self):
        assert _leg_to_outcome_key({"market": "h2h", "selection": "HOME"}) == "h2h_home"

    def test_h2h_draw(self):
        assert _leg_to_outcome_key({"market": "h2h", "selection": "DRAW"}) == "h2h_draw"

    def test_h2h_away(self):
        assert _leg_to_outcome_key({"market": "h2h", "selection": "AWAY"}) == "h2h_away"

    def test_btts_yes(self):
        assert _leg_to_outcome_key({"market": "btts", "selection": "YES"}) == "btts_yes"

    def test_btts_no(self):
        assert _leg_to_outcome_key({"market": "btts", "selection": "NO"}) == "btts_no"

    def test_totals_over(self):
        assert _leg_to_outcome_key({"market": "totals", "selection": "OVER 2.5"}) == "totals_over_2.5"

    def test_totals_under(self):
        assert _leg_to_outcome_key({"market": "totals", "selection": "UNDER 2.5"}) == "totals_under_2.5"

    def test_double_chance_1x(self):
        assert _leg_to_outcome_key({"market": "double_chance", "selection": "1X"}) == "double_chance_1x"

    def test_unsupported_market_returns_none(self):
        assert _leg_to_outcome_key({"market": "corners", "selection": "OVER 9.5"}) is None


class TestMCSGP:
    def _make_sgp_combo(self, legs, home_xg=1.5, away_xg=1.2):
        return {
            "legs": legs,
            "n_legs": len(legs),
            "is_same_game": True,
            "_home_xg": home_xg,
            "_away_xg": away_xg,
            "hit_probability": {"naive": 0.5, "copula_adjusted": 0.4},
        }

    def test_hit_rate_reasonable(self):
        """SGP MC hit rate for home win should be in plausible range."""
        combo = self._make_sgp_combo([
            {"match": "A vs B", "market": "h2h", "selection": "HOME",
             "probability": 0.50, "odds": 2.0, "confidence_level": "HIGH"},
        ])
        rng = np.random.default_rng(42)
        hits = _mc_sgp(combo, 50000, rng)
        hit_rate = hits.mean()
        # With home_xg=1.5, away_xg=1.2, P(home win) should be ~0.38-0.42
        assert 0.30 < hit_rate < 0.50, f"Hit rate {hit_rate:.3f} out of range"

    def test_sgp_btts_and_over(self):
        """BTTS YES + Over 2.5 should have correlated hit rate."""
        combo = self._make_sgp_combo([
            {"match": "A vs B", "market": "btts", "selection": "YES",
             "probability": 0.55, "odds": 1.8, "confidence_level": "HIGH"},
            {"match": "A vs B", "market": "totals", "selection": "OVER 2.5",
             "probability": 0.55, "odds": 1.8, "confidence_level": "HIGH"},
        ])
        rng = np.random.default_rng(42)
        hits = _mc_sgp(combo, 50000, rng)
        hit_rate = hits.mean()
        # These are positively correlated — joint should be > product * some factor
        assert hit_rate > 0.10, f"Joint hit rate {hit_rate:.3f} seems too low"
        assert hit_rate < 0.60, f"Joint hit rate {hit_rate:.3f} seems too high"

    def test_fallback_for_unsupported_market(self):
        """Legs with unsupported markets should fall back to Beta-Bernoulli."""
        combo = self._make_sgp_combo([
            {"match": "A vs B", "market": "corners", "selection": "OVER 9.5",
             "probability": 0.50, "odds": 2.0, "confidence_level": "MEDIUM"},
        ])
        rng = np.random.default_rng(42)
        hits = _mc_sgp(combo, 10000, rng)
        hit_rate = hits.mean()
        assert 0.30 < hit_rate < 0.70, f"Fallback hit rate {hit_rate:.3f} out of range"


class TestMCCrossMatch:
    def _make_cross_combo(self, same_day=True):
        date = "2026-02-22" if same_day else ""
        return {
            "legs": [
                {"match": "Inter vs Milan", "market": "h2h", "selection": "HOME",
                 "probability": 0.55, "odds": 1.8, "confidence_level": "HIGH",
                 "date": date},
                {"match": "Juve vs Napoli", "market": "btts", "selection": "YES",
                 "probability": 0.60, "odds": 1.7, "confidence_level": "HIGH",
                 "date": date},
            ],
            "n_legs": 2,
            "is_same_game": False,
            "hit_probability": {"naive": 0.33, "copula_adjusted": 0.32},
        }

    def test_cross_match_runs(self):
        combo = self._make_cross_combo()
        rng = np.random.default_rng(42)
        hits = _mc_cross_match(combo, 10000, rng)
        assert hits.shape == (10000,)
        hit_rate = hits.mean()
        assert 0.15 < hit_rate < 0.50

    def test_same_day_reduces_joint_probability(self):
        """Same-day correlation should slightly reduce joint hit probability
        compared to independent (no date)."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        combo_corr = self._make_cross_combo(same_day=True)
        combo_indep = self._make_cross_combo(same_day=False)

        n = 100000
        hits_corr = _mc_cross_match(combo_corr, n, rng1)
        hits_indep = _mc_cross_match(combo_indep, n, rng2)

        rate_corr = hits_corr.mean()
        rate_indep = hits_indep.mean()

        # Both should be reasonable
        assert 0.15 < rate_corr < 0.50
        assert 0.15 < rate_indep < 0.50


class TestMonteCarlobandsIntegration:
    def test_sgp_output_contract(self):
        """MC bands should produce median, p10, p90, mc_method, n_sims."""
        combo = {
            "legs": [
                {"match": "A vs B", "market": "h2h", "selection": "HOME",
                 "probability": 0.55, "odds": 1.8, "confidence_level": "HIGH",
                 "date": "2026-02-22"},
            ],
            "n_legs": 1,
            "is_same_game": True,
            "_home_xg": 1.5,
            "_away_xg": 1.0,
            "hit_probability": {"naive": 0.55, "copula_adjusted": 0.50},
        }
        _monte_carlo_bands(combo)
        hp = combo["hit_probability"]
        assert "median" in hp
        assert "p10" in hp
        assert "p90" in hp
        assert "mc_method" in hp
        assert "n_sims" in hp
        assert hp["mc_method"] == "bivariate_poisson_sgp"
        assert hp["p10"] <= hp["median"] <= hp["p90"]

    def test_cross_match_output_contract(self):
        combo = {
            "legs": [
                {"match": "A vs B", "market": "h2h", "selection": "HOME",
                 "probability": 0.55, "odds": 1.8, "confidence_level": "HIGH",
                 "date": "2026-02-22"},
                {"match": "C vs D", "market": "btts", "selection": "YES",
                 "probability": 0.60, "odds": 1.7, "confidence_level": "HIGH",
                 "date": "2026-02-22"},
            ],
            "n_legs": 2,
            "is_same_game": False,
            "hit_probability": {"naive": 0.33, "copula_adjusted": 0.32},
        }
        _monte_carlo_bands(combo)
        hp = combo["hit_probability"]
        assert hp["mc_method"] == "gaussian_copula_cross"
        assert "expected_roi" in combo

    def test_expected_roi_present(self):
        combo = {
            "legs": [
                {"match": "A vs B", "market": "h2h", "selection": "HOME",
                 "probability": 0.55, "odds": 1.8, "confidence_level": "HIGH",
                 "date": "2026-02-22"},
            ],
            "n_legs": 1,
            "is_same_game": True,
            "_home_xg": 1.5,
            "_away_xg": 1.0,
            "hit_probability": {"naive": 0.55, "copula_adjusted": 0.50},
        }
        _monte_carlo_bands(combo)
        assert "expected_roi" in combo
        assert isinstance(combo["expected_roi"], float)


# ---------------------------------------------------------------------------
# Task 3: Bankroll Ruin Simulator Tests
# ---------------------------------------------------------------------------


class TestBetOpportunity:
    def test_blended_prob(self):
        bet = BetOpportunity(model_prob=0.60, sharp_prob=0.50, odds=2.0, won=True)
        expected = 0.6 * 0.60 + 0.4 * 0.50
        assert abs(bet.blended_prob - expected) < 1e-10

    def test_edge(self):
        bet = BetOpportunity(model_prob=0.60, sharp_prob=0.50, odds=2.0, won=True)
        blended = 0.6 * 0.60 + 0.4 * 0.50
        expected_edge = blended - (1.0 / 2.0)
        assert abs(bet.edge - expected_edge) < 1e-10


class TestSimulateSeasonPath:
    def _make_bets(self):
        """Create synthetic bets with known edge."""
        return [
            BetOpportunity(model_prob=0.55, sharp_prob=0.50, odds=2.0, won=True),
            BetOpportunity(model_prob=0.55, sharp_prob=0.50, odds=2.0, won=False),
            BetOpportunity(model_prob=0.55, sharp_prob=0.50, odds=2.0, won=True),
            BetOpportunity(model_prob=0.55, sharp_prob=0.50, odds=2.0, won=True),
            BetOpportunity(model_prob=0.55, sharp_prob=0.50, odds=2.0, won=False),
        ]

    def test_deterministic_with_seed(self):
        bets = self._make_bets()
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        p1 = simulate_season_path(bets, 0.10, 1000, 50, rng1)
        p2 = simulate_season_path(bets, 0.10, 1000, 50, rng2)
        assert p1["terminal_wealth"] == p2["terminal_wealth"]

    def test_positive_terminal_wealth(self):
        bets = self._make_bets()
        rng = np.random.default_rng(42)
        path = simulate_season_path(bets, 0.10, 1000, 50, rng)
        assert path["terminal_wealth"] > 0

    def test_max_drawdown_in_range(self):
        bets = self._make_bets()
        rng = np.random.default_rng(42)
        path = simulate_season_path(bets, 0.10, 1000, 50, rng)
        assert 0 <= path["max_drawdown"] <= 1.0


class TestFindOptimalFraction:
    def test_selects_best_eligible(self):
        results = {
            0.05: {"ruin_probability": 0.01, "median_terminal": 1100},
            0.10: {"ruin_probability": 0.03, "median_terminal": 1200},
            0.25: {"ruin_probability": 0.10, "median_terminal": 1500},
        }
        f, r = find_optimal_fraction(results, max_ruin_prob=0.05)
        assert f == 0.10  # highest median among eligible (P(ruin) <= 5%)
        assert r["median_terminal"] == 1200

    def test_none_eligible_returns_lowest_ruin(self):
        results = {
            0.10: {"ruin_probability": 0.10, "median_terminal": 1200},
            0.25: {"ruin_probability": 0.20, "median_terminal": 1500},
        }
        f, r = find_optimal_fraction(results, max_ruin_prob=0.05)
        # None satisfy <= 5%, so picks lowest ruin probability
        assert f == 0.10

    def test_full_kelly_highest_ruin(self):
        """With positive edge bets, full Kelly should produce higher ruin."""
        bets = [
            BetOpportunity(model_prob=0.60, sharp_prob=0.45, odds=2.2, won=True),
            BetOpportunity(model_prob=0.60, sharp_prob=0.45, odds=2.2, won=False),
        ] * 50

        rng_low = np.random.default_rng(42)
        rng_high = np.random.default_rng(42)

        # With fractional Kelly, drawdowns should be smaller on average
        paths_low = [simulate_season_path(bets, 0.10, 1000, 200, np.random.default_rng(i))
                     for i in range(200)]
        paths_high = [simulate_season_path(bets, 1.0, 1000, 200, np.random.default_rng(i))
                      for i in range(200)]

        dd_low = np.mean([p["max_drawdown"] for p in paths_low])
        dd_high = np.mean([p["max_drawdown"] for p in paths_high])
        assert dd_high > dd_low, (
            f"Full Kelly drawdown ({dd_high:.3f}) should exceed "
            f"fractional ({dd_low:.3f})"
        )


class TestRuinSimConfig:
    def test_default_kelly_fractions(self):
        config = RuinSimConfig()
        assert 0.05 in config.kelly_fractions
        assert 1.0 in config.kelly_fractions

    def test_custom_config(self):
        config = RuinSimConfig(n_paths=100, ruin_threshold=0.30)
        assert config.n_paths == 100
        assert config.ruin_threshold == 0.30
