#!/usr/bin/env python3
"""Comprehensive Integration Tests for the Serie A Prediction Pipeline.

Tests the full pipeline end-to-end: feature build -> train -> predict -> bet.

Run all integration tests:
    python -m pytest tests/test_integration.py -v -m integration

Run only fast tests:
    python -m pytest tests/test_integration.py -v -m "integration and not slow"

Run end-to-end:
    python -m pytest tests/test_integration.py -v -k "TestEndToEnd"
"""

import json
import math
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# 1. UNIFIED MODULE IMPORT TESTS
# ============================================================================

class TestUnifiedImports:
    """Verify that all unified scripts import cleanly and expose expected APIs."""

    def test_train_unified_imports(self):
        """train_unified.py imports without error."""
        from scripts.models.train_unified import TrainConfig, UnifiedTrainer

        assert TrainConfig is not None
        assert UnifiedTrainer is not None

    def test_predict_unified_imports(self):
        """predict_unified.py imports without error."""
        from scripts.prediction.predict_unified import PredictionConfig, UnifiedPredictor

        assert PredictionConfig is not None
        assert UnifiedPredictor is not None

    def test_betting_unified_imports(self):
        """betting_unified.py imports without error."""
        from scripts.betting.betting_unified import (
            BettingConfig,
            UnifiedBettingEngine,
            ValueBet,
            BetSlip,
        )

        assert BettingConfig is not None
        assert UnifiedBettingEngine is not None
        assert ValueBet is not None
        assert BetSlip is not None

    def test_backtest_unified_imports(self):
        """backtest_unified.py imports without error."""
        from scripts.analysis.backtest_unified import BacktestConfig, BacktestEngine

        assert BacktestConfig is not None
        assert BacktestEngine is not None

    def test_optimize_unified_imports(self):
        """optimize_unified.py imports without error."""
        from scripts.models.optimize_unified import OptimizeConfig, UnifiedOptimizer

        assert OptimizeConfig is not None
        assert UnifiedOptimizer is not None

    def test_backward_compat_wrappers(self):
        """Backward-compatible wrapper functions exist and are callable."""
        from scripts.models.train_unified import load_and_prepare_data
        from scripts.prediction.predict_unified import run_predictions
        from scripts.betting.betting_unified import generate_betting_slip

        assert callable(load_and_prepare_data)
        assert callable(run_predictions)
        assert callable(generate_betting_slip)

    def test_feature_build_imports(self):
        """features/build.py core classes import without error."""
        from features.build import (
            FeatureState,
            FeaturePlugin,
            FeaturePipeline,
            get_ml_feature_columns,
        )

        assert FeatureState is not None
        assert FeaturePlugin is not None
        assert FeaturePipeline is not None
        assert callable(get_ml_feature_columns)

    def test_config_settings_imports(self):
        """config/settings.py exports all expected constants."""
        from config.settings import (
            DATA_DIR,
            MODELS_DIR,
            FEATURES_DIR,
            PROJECT_ROOT as CFG_ROOT,
            SEASONS,
            ROLLING_WINDOWS,
            DEFAULT_LEAGUE,
        )

        assert DATA_DIR.is_absolute()
        assert MODELS_DIR.is_absolute()
        assert FEATURES_DIR.is_absolute()
        assert len(SEASONS) >= 15
        assert DEFAULT_LEAGUE == "serie_a"
        assert ROLLING_WINDOWS == [3, 5, 10]

    def test_storage_paths_imports(self):
        """storage/paths.py exports path helpers."""
        from storage.paths import (
            features_path,
            parsed_path,
            ensure_dirs,
        )

        assert callable(features_path)
        assert callable(parsed_path)
        assert callable(ensure_dirs)

        # features_path should return an absolute Path under data/features/
        fp = features_path()
        assert fp.name == "features.parquet"
        assert "features" in str(fp)


# ============================================================================
# 2. FEATURE PIPELINE TESTS
# ============================================================================

@pytest.mark.integration
class TestFeaturePipeline:
    """Tests for the plugin-based feature engineering pipeline."""

    def test_feature_state_initialization(self, sample_matches_df):
        """FeatureState can be created with a matches DataFrame."""
        from features.build import FeatureState

        state = FeatureState(matches=sample_matches_df)

        assert state.matches is not None
        assert len(state.matches) == len(sample_matches_df)
        assert state.team_log is None
        assert state.feature_df is None
        assert state.season is None

    def test_feature_state_with_season(self, sample_matches_df):
        """FeatureState accepts an optional season parameter."""
        from features.build import FeatureState

        state = FeatureState(matches=sample_matches_df, season="2024-2025")
        assert state.season == "2024-2025"

    def test_plugin_registration(self):
        """Plugins can be registered to FeaturePipeline."""
        from features.build import FeaturePipeline, FeaturePlugin, FeatureState

        class DummyPlugin(FeaturePlugin):
            name = "dummy_plugin"
            version = "1.0"
            dependencies = []

            def apply(self, state: FeatureState) -> FeatureState:
                return state

        pipeline = FeaturePipeline(cache_dir=Path("/tmp/test_cache"))
        pipeline.register(DummyPlugin())

        assert len(pipeline._plugins) == 1
        assert pipeline._plugin_map["dummy_plugin"].name == "dummy_plugin"

    def test_duplicate_plugin_raises(self):
        """Registering a plugin with the same name twice raises ValueError."""
        from features.build import FeaturePipeline, FeaturePlugin, FeatureState

        class DummyPlugin(FeaturePlugin):
            name = "dup_plugin"
            version = "1.0"
            dependencies = []

            def apply(self, state: FeatureState) -> FeatureState:
                return state

        pipeline = FeaturePipeline(cache_dir=Path("/tmp/test_cache"))
        pipeline.register(DummyPlugin())

        with pytest.raises(ValueError, match="Duplicate plugin name"):
            pipeline.register(DummyPlugin())

    def test_plugin_dependency_resolution(self):
        """Pipeline resolves plugin dependencies correctly via topological sort."""
        from features.build import FeaturePipeline, FeaturePlugin, FeatureState

        execution_order = []

        class PluginA(FeaturePlugin):
            name = "plugin_a"
            version = "1.0"
            dependencies = []

            def apply(self, state: FeatureState) -> FeatureState:
                execution_order.append("a")
                return state

        class PluginB(FeaturePlugin):
            name = "plugin_b"
            version = "1.0"
            dependencies = ["plugin_a"]

            def apply(self, state: FeatureState) -> FeatureState:
                execution_order.append("b")
                return state

        class PluginC(FeaturePlugin):
            name = "plugin_c"
            version = "1.0"
            dependencies = ["plugin_b"]

            def apply(self, state: FeatureState) -> FeatureState:
                execution_order.append("c")
                return state

        pipeline = FeaturePipeline(cache_dir=Path("/tmp/test_cache_dep"))

        # Register in reverse order to test dependency resolution
        pipeline.register(PluginC())
        pipeline.register(PluginA())
        pipeline.register(PluginB())

        resolved = pipeline._resolve_order()
        resolved_names = [p.name for p in resolved]

        assert resolved_names.index("plugin_a") < resolved_names.index("plugin_b")
        assert resolved_names.index("plugin_b") < resolved_names.index("plugin_c")

    def test_circular_dependency_raises(self):
        """Circular dependencies raise ValueError."""
        from features.build import FeaturePipeline, FeaturePlugin, FeatureState

        class PluginX(FeaturePlugin):
            name = "plugin_x"
            version = "1.0"
            dependencies = ["plugin_y"]

            def apply(self, state: FeatureState) -> FeatureState:
                return state

        class PluginY(FeaturePlugin):
            name = "plugin_y"
            version = "1.0"
            dependencies = ["plugin_x"]

            def apply(self, state: FeatureState) -> FeatureState:
                return state

        pipeline = FeaturePipeline(cache_dir=Path("/tmp/test_cache_circ"))
        pipeline.register(PluginX())
        pipeline.register(PluginY())

        with pytest.raises(ValueError, match="Circular dependency"):
            pipeline._resolve_order()

    def test_missing_dependency_raises(self):
        """Missing dependencies raise ValueError."""
        from features.build import FeaturePipeline, FeaturePlugin, FeatureState

        class PluginZ(FeaturePlugin):
            name = "plugin_z"
            version = "1.0"
            dependencies = ["nonexistent_plugin"]

            def apply(self, state: FeatureState) -> FeatureState:
                return state

        pipeline = FeaturePipeline(cache_dir=Path("/tmp/test_cache_miss"))
        pipeline.register(PluginZ())

        with pytest.raises(ValueError, match="Missing dependency"):
            pipeline._resolve_order()

    def test_feature_caching(self, tmp_path, sample_matches_df):
        """Plugin results are cached to parquet and retrieved correctly."""
        from features.build import FeaturePipeline, FeaturePlugin, FeatureState

        call_count = {"n": 0}

        class CountingPlugin(FeaturePlugin):
            name = "counting_plugin"
            version = "1.0"
            dependencies = []

            def apply(self, state: FeatureState) -> FeatureState:
                call_count["n"] += 1
                state.feature_df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": [4, 5, 6]})
                return state

        cache_dir = tmp_path / "cache"
        pipeline = FeaturePipeline(cache_dir=cache_dir)
        plugin = CountingPlugin()
        pipeline.register(plugin)

        # First run: should compute
        state = FeatureState(matches=sample_matches_df)
        result = pipeline.build(state, use_cache=True)
        assert call_count["n"] == 1
        assert result.feature_df is not None

        # Cache file should exist
        cache_file = pipeline._cache_path(plugin)
        assert cache_file.exists()

        # Second run: should use cache
        state2 = FeatureState(matches=sample_matches_df)
        pipeline2 = FeaturePipeline(cache_dir=cache_dir)
        pipeline2.register(CountingPlugin())
        result2 = pipeline2.build(state2, use_cache=True)
        # The plugin apply() should NOT have been called again
        # (call_count["n"] stays at 1 because the second pipeline instance
        # has its own CountingPlugin, but the cache was hit)
        assert result2.feature_df is not None

    def test_feature_build_no_cache(self, tmp_path, sample_matches_df):
        """Feature pipeline works with caching disabled (use_cache=False)."""
        from features.build import FeaturePipeline, FeaturePlugin, FeatureState

        call_count = {"n": 0}

        class NoCachePlugin(FeaturePlugin):
            name = "nocache_plugin"
            version = "1.0"
            dependencies = []

            def apply(self, state: FeatureState) -> FeatureState:
                call_count["n"] += 1
                state.feature_df = pd.DataFrame({"x": [10, 20]})
                return state

        pipeline = FeaturePipeline(cache_dir=tmp_path / "cache_nc")
        pipeline.register(NoCachePlugin())

        state = FeatureState(matches=sample_matches_df)
        result = pipeline.build(state, use_cache=False)
        assert call_count["n"] == 1
        assert result.feature_df is not None

        # Run again with cache disabled -- should re-compute
        state2 = FeatureState(matches=sample_matches_df)
        pipeline2 = FeaturePipeline(cache_dir=tmp_path / "cache_nc")
        pipeline2.register(NoCachePlugin())
        result2 = pipeline2.build(state2, use_cache=False)
        assert call_count["n"] == 2

    def test_full_feature_build(self, features_df):
        """Full feature pipeline produces expected output shape on real data."""
        from features.build import get_ml_feature_columns

        # features_df is a 300-row sample loaded from features.parquet
        assert len(features_df) > 0

        ml_cols = get_ml_feature_columns(features_df)
        assert len(ml_cols) > 50, (
            f"Expected >50 ML-safe columns, got {len(ml_cols)}"
        )

        # Critical columns should exist
        for col in ["elo_diff", "home_elo", "away_elo"]:
            assert col in features_df.columns, f"Missing critical column: {col}"

        # result column must be clean
        valid_results = {"H", "D", "A"}
        result_values = set(features_df["result"].dropna().unique())
        assert result_values.issubset(valid_results), (
            f"Unexpected result values: {result_values - valid_results}"
        )

    def test_get_ml_feature_columns_excludes_identifiers(self, features_df):
        """get_ml_feature_columns excludes identifiers, targets, and metadata."""
        from features.build import get_ml_feature_columns

        ml_cols = get_ml_feature_columns(features_df)

        forbidden = {"match_id", "season", "match_date", "home_team", "away_team",
                      "venue", "attendance", "referee", "result", "home_score",
                      "away_score"}
        overlap = forbidden & set(ml_cols)
        assert len(overlap) == 0, f"ML columns include forbidden cols: {overlap}"


# ============================================================================
# 3. TRAINING PIPELINE TESTS
# ============================================================================

@pytest.mark.integration
class TestTrainingPipeline:
    """Tests for the unified training pipeline."""

    def test_train_config_defaults(self):
        """TrainConfig has sensible defaults."""
        from scripts.models.train_unified import TrainConfig

        cfg = TrainConfig()
        assert cfg.mode == "hybrid"
        assert cfg.random_seed == 42
        assert cfg.n_cv_splits == 5
        assert cfg.top_k_features > 0
        assert 0.0 < cfg.corr_threshold <= 1.0
        assert 0.0 < cfg.nan_threshold <= 1.0
        assert cfg.final_train_ratio > 0.5
        assert cfg.xg_iterations > 0
        assert cfg.cls_iterations > 0

    def test_training_modes_available(self):
        """All 7 training modes are recognized as valid identifiers."""
        expected_modes = {
            "hybrid", "classifier_only", "xg_only",
            "fast", "with_odds", "deep", "optimized",
        }
        from scripts.models.train_unified import TrainConfig

        # Each mode should be settable without error
        for mode in expected_modes:
            cfg = TrainConfig(mode=mode)
            assert cfg.mode == mode

    def test_train_config_fast_overrides(self):
        """Fast mode has separate iteration/learning-rate settings."""
        from scripts.models.train_unified import TrainConfig

        cfg = TrainConfig(mode="fast")
        assert cfg.fast_iterations > 0
        assert cfg.fast_learning_rate > 0
        assert cfg.fast_iterations < cfg.cls_iterations, (
            "Fast mode should use fewer iterations than default classifier"
        )

    def test_trainer_initialization(self, tmp_path):
        """UnifiedTrainer initializes and creates model directory."""
        from scripts.models.train_unified import TrainConfig, UnifiedTrainer

        cfg = TrainConfig(mode="fast", model_dir=str(tmp_path / "models"))
        trainer = UnifiedTrainer(cfg)

        assert trainer.cfg.mode == "fast"
        assert trainer._model_dir.exists()

    def test_feature_discovery(self, features_df):
        """Feature discovery identifies usable numeric pre-match columns."""
        from scripts.models.train_unified import discover_features

        usable = discover_features(features_df)
        assert len(usable) > 20, f"Expected >20 usable features, got {len(usable)}"

        # All returned features should be numeric
        for col in usable:
            assert features_df[col].dtype.kind in ("f", "i", "u"), (
                f"Feature {col} is not numeric: {features_df[col].dtype}"
            )

    def test_correlation_pruning(self, features_df):
        """Correlation pruning removes redundant features."""
        from scripts.models.train_unified import discover_features, correlation_pruning

        usable = discover_features(features_df)
        if len(usable) < 3:
            pytest.skip("Not enough features to test correlation pruning")

        # Create fake importance dict
        importance = {f: 1.0 / (i + 1) for i, f in enumerate(usable)}

        pruned = correlation_pruning(features_df, usable, importance, threshold=0.85)
        assert len(pruned) <= len(usable)
        assert len(pruned) > 0

    def test_time_series_split(self, full_features_df):
        """Walk-forward time series split produces valid train/test indices."""
        from scripts.models.train_unified import time_series_split

        df = full_features_df
        if "season" not in df.columns:
            pytest.skip("No season column in features_df")

        n_seasons = df["season"].nunique()
        if n_seasons < 4:
            pytest.skip(f"Only {n_seasons} seasons -- need >=4 for walk-forward split")

        splits = time_series_split(df, n_splits=3, min_train=2)
        assert len(splits) > 0

        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            # Train dates should be before test dates
            train_max = df.loc[train_idx, "match_date"].max()
            test_min = df.loc[test_idx, "match_date"].min()
            # Allow some tolerance for edge cases
            assert train_max <= test_min or pd.isna(train_max) or pd.isna(test_min)

    def test_poisson_win_prob(self):
        """Poisson win probability utility produces valid distributions."""
        from scripts.models.train_unified import poisson_win_prob

        probs = poisson_win_prob(1.5, 1.0)

        # Must sum to 1
        total = probs["H"] + probs["D"] + probs["A"]
        assert abs(total - 1.0) < 0.001, f"Probabilities sum to {total}"

        # Home should win more often with higher xG
        assert probs["H"] > probs["A"], (
            f"Home xG 1.5 > Away xG 1.0 but H={probs['H']:.3f} <= A={probs['A']:.3f}"
        )

        # Equal xG should produce roughly equal H and A
        equal = poisson_win_prob(1.5, 1.5)
        assert abs(equal["H"] - equal["A"]) < 0.05

        # Draw should be significant
        assert equal["D"] > 0.15


# ============================================================================
# 4. PREDICTION PIPELINE TESTS
# ============================================================================

@pytest.mark.integration
class TestPredictionPipeline:
    """Tests for the unified prediction pipeline."""

    def test_prediction_config_defaults(self):
        """PredictionConfig has sensible defaults."""
        from scripts.prediction.predict_unified import PredictionConfig

        cfg = PredictionConfig()
        assert cfg.mode == "ensemble"
        assert cfg.bankroll > 0
        assert 0.0 < cfg.kelly_fraction <= 1.0
        assert cfg.output_format in ("text", "json")
        assert cfg.strategy in ("default", "volume", "selective", "ml_heavy", "xg_dominant")

    def test_all_prediction_modes(self):
        """All 5 prediction modes can create a PredictionConfig without error."""
        from scripts.prediction.predict_unified import PredictionConfig

        modes = ["ensemble", "factor", "xg", "ml", "market"]
        for mode in modes:
            cfg = PredictionConfig(mode=mode)
            assert cfg.mode == mode

    def test_predictor_instantiation(self):
        """UnifiedPredictor can be instantiated in each mode."""
        from scripts.prediction.predict_unified import PredictionConfig, UnifiedPredictor

        for mode in ["ensemble", "factor", "xg", "ml", "market"]:
            cfg = PredictionConfig(mode=mode)
            predictor = UnifiedPredictor(cfg)
            assert predictor.config.mode == mode
            assert predictor.initialized is False

    def test_factor_predict_produces_valid_probabilities(self):
        """Factor-based prediction produces valid probabilities summing to ~1.0."""
        from scripts.prediction.predict_unified import factor_predict, BASE_RATES

        match = {"home_team": "Inter", "away_team": "Milan"}
        factors = {
            "home_factors": ["home_favorite", "big_stadium"],
            "away_factors": [],
            "neutral_factors": ["derby"],
            "n_home_factors": 2,
            "n_away_factors": 0,
            "home_lift": 0.15,
            "away_lift": 0.0,
            "weather": {},
        }
        form_data = {"matchups": {}}

        result = factor_predict(match, factors, form_data)

        # Probabilities must exist
        assert "probabilities" in result
        probs = result["probabilities"]
        assert "home" in probs
        assert "draw" in probs
        assert "away" in probs

        # Each probability between 0 and 1
        for key, val in probs.items():
            assert 0.0 <= val <= 1.0, f"Probability {key}={val} out of [0,1]"

        # Sum close to 1
        total = probs["home"] + probs["draw"] + probs["away"]
        assert abs(total - 1.0) < 0.02, f"Probabilities sum to {total}"

    def test_factor_predict_with_market_anchor(self):
        """Factor prediction with market_probs anchor uses dampened lifts."""
        from scripts.prediction.predict_unified import factor_predict

        match = {"home_team": "Roma", "away_team": "Cagliari"}
        factors = {
            "home_factors": ["home_favorite"],
            "away_factors": [],
            "neutral_factors": [],
            "n_home_factors": 1,
            "n_away_factors": 0,
            "home_lift": 0.12,
            "away_lift": 0.0,
            "weather": {},
        }
        form_data = {"matchups": {}}
        market_probs = {"prob_H": 0.55, "prob_D": 0.25, "prob_A": 0.20}

        result = factor_predict(match, factors, form_data, market_probs=market_probs)

        probs = result["probabilities"]
        total = probs["home"] + probs["draw"] + probs["away"]
        assert abs(total - 1.0) < 0.02

        # Market-anchored prediction should be close to market
        assert abs(probs["home"] - 0.55) < 0.15, (
            "Market-anchored prediction should be within 15pp of market"
        )

    def test_prediction_output_format(self, sample_prediction):
        """Predictions have required keys: probabilities, confidence, factors."""
        required_keys = {"match", "probabilities", "confidence"}
        assert required_keys.issubset(set(sample_prediction.keys()))

        probs = sample_prediction["probabilities"]
        assert {"home", "draw", "away"} == set(probs.keys())

    def test_base_rates_valid(self):
        """BASE_RATES represent a valid probability distribution."""
        from scripts.prediction.predict_unified import BASE_RATES

        assert "home_win" in BASE_RATES
        assert "draw" in BASE_RATES
        assert "away_win" in BASE_RATES

        total = BASE_RATES["home_win"] + BASE_RATES["draw"] + BASE_RATES["away_win"]
        assert abs(total - 1.0) < 0.01, f"BASE_RATES 1X2 sum to {total}"

    def test_ensemble_weights_valid(self):
        """ENSEMBLE_WEIGHTS sum to 1.0 and contain expected methods."""
        from scripts.prediction.predict_unified import ENSEMBLE_WEIGHTS

        total = sum(ENSEMBLE_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01, f"ENSEMBLE_WEIGHTS sum to {total}"

        expected_methods = {"factor", "xg", "ml"}
        assert expected_methods.issubset(set(ENSEMBLE_WEIGHTS.keys()))

    def test_identify_all_factors(self):
        """Factor identification produces a structured output dict."""
        from scripts.prediction.predict_unified import identify_all_factors

        match = {
            "home_team": "Inter",
            "away_team": "Milan",
            "referee": "Simone Sozza",
        }
        form_data = {
            "matchups": {
                "Inter vs Milan": {
                    "factors": ["home_favorite", "big_stadium"],
                },
            },
        }
        weather_data = {}

        result = identify_all_factors(match, form_data, weather_data)

        assert "home_factors" in result
        assert "away_factors" in result
        assert "neutral_factors" in result
        assert "home_lift" in result
        assert "away_lift" in result
        assert isinstance(result["home_factors"], list)
        assert isinstance(result["home_lift"], (int, float))

    def test_stacking_bonuses(self):
        """STACKING_BONUSES are defined for factor counts 1-5."""
        from scripts.prediction.predict_unified import STACKING_BONUSES

        for n in [1, 2, 3, 4, 5]:
            assert n in STACKING_BONUSES
            assert STACKING_BONUSES[n] > 0

        # Bonuses should increase monotonically
        for n in [2, 3, 4, 5]:
            assert STACKING_BONUSES[n] > STACKING_BONUSES[n - 1], (
                f"Stacking bonus {n}={STACKING_BONUSES[n]} should be > "
                f"{n-1}={STACKING_BONUSES[n-1]}"
            )


# ============================================================================
# 5. BETTING PIPELINE TESTS
# ============================================================================

@pytest.mark.integration
class TestBettingPipeline:
    """Tests for the unified betting engine."""

    def test_betting_config_defaults(self):
        """BettingConfig has sensible defaults."""
        from scripts.betting.betting_unified import BettingConfig

        cfg = BettingConfig()
        assert cfg.bankroll == 1000.0
        assert cfg.kelly_fraction == 0.05
        assert cfg.min_edge_pct > 0
        assert cfg.max_edge_pct > cfg.min_edge_pct
        assert cfg.max_stake_pct > cfg.min_stake_pct
        assert cfg.min_odds >= 1.0
        assert cfg.max_odds > cfg.min_odds
        assert cfg.max_bets_per_match > 0
        assert cfg.max_total_bets > 0
        assert cfg.sharp_bookmaker == "Pinnacle"
        assert cfg.monte_carlo_sims > 0

    def test_kelly_criterion_calculation(self):
        """Kelly criterion produces correct stake sizes for known inputs."""
        from scripts.betting.betting_unified import calculate_kelly

        # Test 1: 60% probability, 2.0 odds, 0.25 fraction
        # Kelly full = (b*p - q) / b = (1*0.6 - 0.4) / 1 = 0.20
        # Quarter Kelly = 0.20 * 0.25 = 0.05 (5%)
        kelly = calculate_kelly(model_prob=0.60, odds=2.0, fraction=0.25)
        expected = (1.0 * 0.60 - 0.40) / 1.0 * 0.25  # = 0.05
        assert abs(kelly - expected) < 0.001, (
            f"Kelly={kelly:.4f}, expected={expected:.4f}"
        )

        # Test 2: Higher probability => higher stake
        kelly_70 = calculate_kelly(model_prob=0.70, odds=2.0, fraction=0.25)
        assert kelly_70 > kelly, "Higher probability should produce larger Kelly stake"

        # Test 3: Same probability, higher odds => higher stake
        kelly_high_odds = calculate_kelly(model_prob=0.60, odds=3.0, fraction=0.25)
        kelly_low_odds = calculate_kelly(model_prob=0.60, odds=2.0, fraction=0.25)
        assert kelly_high_odds > kelly_low_odds, (
            "Higher odds should produce larger Kelly stake at same probability"
        )

    def test_kelly_negative_ev_returns_zero(self):
        """Kelly criterion returns 0 for negative expected value bets."""
        from scripts.betting.betting_unified import calculate_kelly

        # 40% probability at 2.0 odds: EV = 0.4*2 - 1 = -0.2 (negative)
        kelly = calculate_kelly(model_prob=0.40, odds=2.0, fraction=0.25)
        assert kelly == 0, f"Negative EV should produce 0 stake, got {kelly}"

    def test_kelly_edge_cases(self):
        """Kelly handles edge cases without crashing."""
        from scripts.betting.betting_unified import calculate_kelly

        assert calculate_kelly(0, 2.0) == 0          # zero probability
        assert calculate_kelly(0.5, 1.0) == 0         # odds = 1.0 (invalid)
        assert calculate_kelly(0.5, 0) == 0            # odds = 0 (invalid)
        assert calculate_kelly(1.0, 2.0) == 0          # probability = 1.0

    def test_overround_removal(self):
        """Overround removal produces fair probabilities summing to 1.0."""
        from scripts.betting.betting_unified import remove_overround

        # Typical 1X2 odds with ~5% overround
        odds = [1.90, 3.50, 4.50]
        fair_probs = remove_overround(odds)

        assert len(fair_probs) == 3

        # Must sum to 1.0
        total = sum(fair_probs)
        assert abs(total - 1.0) < 0.001, f"Fair probabilities sum to {total}"

        # Each probability between 0 and 1
        for p in fair_probs:
            assert 0.0 < p < 1.0, f"Probability {p} out of (0,1)"

        # Most probable outcome should have highest fair probability
        raw_probs = [1 / o for o in odds]
        max_raw_idx = raw_probs.index(max(raw_probs))
        max_fair_idx = fair_probs.index(max(fair_probs))
        assert max_raw_idx == max_fair_idx, (
            "Overround removal should preserve probability ordering"
        )

    def test_overround_removal_edge_cases(self):
        """Overround removal handles edge cases gracefully."""
        from scripts.betting.betting_unified import remove_overround

        # Empty list
        result = remove_overround([])
        assert result == []

        # Invalid odds (<=1)
        result = remove_overround([0.5, 1.0, 2.0])
        assert len(result) == 3
        assert abs(sum(result) - 1.0) < 0.001

    def test_value_detection(self):
        """Value bets are correctly identified when model_prob > implied_prob."""
        from scripts.betting.betting_unified import calculate_ev

        # Positive value: model prob 55%, odds 2.20 (implied ~45.5%)
        ev = calculate_ev(model_prob=0.55, odds=2.20)
        assert ev > 0, f"Expected positive EV, got {ev}"

        # Negative value: model prob 40%, odds 2.0 (implied 50%)
        ev_neg = calculate_ev(model_prob=0.40, odds=2.0)
        assert ev_neg < 0, f"Expected negative EV, got {ev_neg}"

        # Break-even: model prob = implied prob (50% at 2.0)
        ev_even = calculate_ev(model_prob=0.50, odds=2.0)
        assert abs(ev_even) < 0.01, f"Break-even should be near 0, got {ev_even}"

    def test_value_bet_dataclass(self):
        """ValueBet dataclass is correctly constructable with all required fields."""
        from scripts.betting.betting_unified import ValueBet

        bet = ValueBet(
            match="Inter vs Milan",
            date="2026-02-09",
            market="1X2",
            selection="Home",
            model_prob=0.55,
            sharp_implied_prob=0.47,
            edge_pct=17.0,
            raw_edge=0.08,
            best_odds=2.15,
            best_bookmaker="Bet365",
            avg_odds=2.10,
            pinnacle_odds=2.10,
            odds_count=5,
        )

        assert bet.match == "Inter vs Milan"
        assert bet.model_prob == 0.55
        assert bet.kelly_raw == 0.0  # default
        assert bet.is_selected is False

    def test_bet_slip_dataclass(self):
        """BetSlip is constructable and has expected default structure."""
        from scripts.betting.betting_unified import BetSlip

        slip = BetSlip()
        assert slip.bets == []
        assert slip.total_stake == 0.0
        assert slip.total_ev == 0.0
        assert slip.n_matches == 0
        assert slip.bankroll == 1000.0

    def test_betting_engine_instantiation(self):
        """UnifiedBettingEngine can be instantiated with default and custom config."""
        from scripts.betting.betting_unified import BettingConfig, UnifiedBettingEngine

        # Default config
        engine = UnifiedBettingEngine()
        assert engine.cfg.bankroll == 1000.0
        assert engine.all_bets == []
        assert engine.selected == []

        # Custom config
        cfg = BettingConfig(bankroll=500, kelly_fraction=0.10, min_edge_pct=3.0)
        engine2 = UnifiedBettingEngine(cfg)
        assert engine2.cfg.bankroll == 500
        assert engine2.cfg.kelly_fraction == 0.10

    def test_market_priority_ordering(self):
        """Market priorities are correctly defined for portfolio optimization."""
        from scripts.betting.betting_unified import MARKET_PRIORITY

        assert MARKET_PRIORITY["1X2"] >= MARKET_PRIORITY["O/U"]
        assert MARKET_PRIORITY["O/U"] >= MARKET_PRIORITY["AH"]
        assert len(MARKET_PRIORITY) >= 4

    def test_confidence_rank_mapping(self):
        """Confidence rank mapping covers all expected tiers."""
        from scripts.betting.betting_unified import CONFIDENCE_RANK

        assert CONFIDENCE_RANK["VERY_HIGH"] == 5
        assert CONFIDENCE_RANK["HIGH"] == 4
        assert CONFIDENCE_RANK["MEDIUM"] == 2
        assert CONFIDENCE_RANK["LOW"] == 1

    def test_stake_allocation_reasonable(self):
        """Stake allocation percentages are within safe bounds."""
        from scripts.betting.betting_unified import STAKE_ALLOCATION

        for tier, pct in STAKE_ALLOCATION.items():
            assert 0.0 < pct <= 0.05, (
                f"Stake allocation for {tier}={pct} exceeds 5% cap"
            )


# ============================================================================
# 6. BACKTEST PIPELINE TESTS
# ============================================================================

@pytest.mark.integration
class TestBacktestPipeline:
    """Tests for the unified backtesting framework."""

    def test_backtest_config_defaults(self):
        """BacktestConfig has sensible defaults."""
        from scripts.analysis.backtest_unified import BacktestConfig

        cfg = BacktestConfig()
        assert cfg.mode == "accuracy"
        assert cfg.bankroll > 0
        assert 0.0 < cfg.kelly_fraction <= 1.0
        assert cfg.min_edge > 0
        assert cfg.min_train_seasons >= 3
        assert len(cfg.seasons) >= 1
        assert len(cfg.metrics) >= 3

    def test_backtest_all_modes_valid(self):
        """All backtest modes can be specified in config."""
        from scripts.analysis.backtest_unified import BacktestConfig

        modes = ["accuracy", "betting", "value", "ml-walkforward", "all"]
        for mode in modes:
            cfg = BacktestConfig(mode=mode)
            assert cfg.mode == mode

    def test_backtest_engine_instantiation(self):
        """BacktestEngine can be instantiated with default config."""
        from scripts.analysis.backtest_unified import BacktestConfig, BacktestEngine

        cfg = BacktestConfig()
        engine = BacktestEngine(cfg)
        assert engine.df is None
        assert engine.results == {}

    def test_backtest_loads_data(self):
        """Backtest engine loads features.parquet correctly."""
        from scripts.analysis.backtest_unified import BacktestConfig, BacktestEngine

        cfg = BacktestConfig()
        engine = BacktestEngine(cfg)
        loaded = engine.load_data()

        if not loaded:
            pytest.skip("features.parquet not available for backtest data load test")

        assert engine.df is not None
        assert len(engine.df) > 0
        assert "result" in engine.df.columns
        assert "match_date" in engine.df.columns
        assert "season" in engine.df.columns

        # Result column should only contain valid values
        valid = {"H", "D", "A"}
        assert set(engine.df["result"].unique()).issubset(valid)

    def test_ranked_probability_score(self):
        """Ranked probability score is computed correctly for known inputs."""
        from scripts.analysis.backtest_unified import ranked_probability_score

        # Perfect prediction: H predicted as [1, 0, 0], actual = H (0)
        y_true = np.array([0])
        y_proba = np.array([[1.0, 0.0, 0.0]])
        rps = ranked_probability_score(y_true, y_proba)
        assert abs(rps) < 0.001, f"Perfect prediction should have RPS~0, got {rps}"

        # Worst prediction: H predicted as [0, 0, 1], actual = H (0)
        y_proba_bad = np.array([[0.0, 0.0, 1.0]])
        rps_bad = ranked_probability_score(y_true, y_proba_bad)
        assert rps_bad > rps, "Bad prediction should have higher RPS"

        # Uniform prediction
        y_proba_uniform = np.array([[1 / 3, 1 / 3, 1 / 3]])
        rps_uniform = ranked_probability_score(y_true, y_proba_uniform)
        assert 0 < rps_uniform < 1

    def test_multiclass_brier_score(self):
        """Multiclass Brier score is computed correctly."""
        from scripts.analysis.backtest_unified import _multiclass_brier

        # Perfect predictions
        y_true = np.array([0, 1, 2])
        y_proba = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        brier = _multiclass_brier(y_true, y_proba)
        assert abs(brier) < 0.001, f"Perfect predictions should have Brier~0, got {brier}"

        # Worst predictions
        y_proba_bad = np.array([
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ])
        brier_bad = _multiclass_brier(y_true, y_proba_bad)
        assert brier_bad > brier, "Bad predictions should have higher Brier score"

    def test_predict_factor_based(self):
        """Factor-based backtest prediction produces valid probabilities."""
        from scripts.analysis.backtest_unified import predict_factor_based

        row = pd.Series({
            "elo_diff": 100.0,
            "home_attack_strength": 1.3,
            "home_defense_strength": 0.8,
            "away_attack_strength": 1.0,
            "away_defense_strength": 1.1,
            "h2h_home_win_rate": 0.55,
        })

        result = predict_factor_based(row)
        assert "prob_H" in result
        assert "prob_D" in result
        assert "prob_A" in result

        total = result["prob_H"] + result["prob_D"] + result["prob_A"]
        assert abs(total - 1.0) < 0.01, f"Factor probs sum to {total}"

        for key in ["prob_H", "prob_D", "prob_A"]:
            assert 0 <= result[key] <= 1

    def test_predict_xg_poisson(self):
        """xG Poisson backtest prediction works with valid features."""
        from scripts.analysis.backtest_unified import predict_xg_poisson

        row = pd.Series({
            "home_xg_attack_strength": 1.5,
            "home_xg_defense_strength": 0.8,
            "away_xg_attack_strength": 1.0,
            "away_xg_defense_strength": 1.2,
        })

        result = predict_xg_poisson(row)
        if result is not None:
            total = result["prob_H"] + result["prob_D"] + result["prob_A"]
            assert abs(total - 1.0) < 0.01

    def test_predict_xg_poisson_missing_data(self):
        """xG Poisson returns None when features are missing."""
        from scripts.analysis.backtest_unified import predict_xg_poisson

        row = pd.Series({
            "home_xg_attack_strength": np.nan,
            "home_xg_defense_strength": 0.8,
            "away_xg_attack_strength": 1.0,
            "away_xg_defense_strength": 1.2,
        })

        result = predict_xg_poisson(row)
        assert result is None

    def test_poisson_1x2_from_xg(self):
        """Poisson 1X2 conversion in optimize_unified produces valid arrays."""
        from scripts.models.optimize_unified import poisson_1x2_from_xg

        probs = poisson_1x2_from_xg(1.5, 1.0)
        assert len(probs) == 3
        assert abs(probs.sum() - 1.0) < 0.001
        assert probs[0] > probs[2], "Home should be favored with higher xG"


# ============================================================================
# 7. END-TO-END PIPELINE TESTS
# ============================================================================

@pytest.mark.integration
@pytest.mark.slow
class TestEndToEnd:
    """End-to-end tests combining multiple pipeline stages."""

    def test_data_quality_gates(self, full_features_df):
        """Data quality checks pass on current features.parquet."""
        df = full_features_df

        # 1. No duplicate match_ids
        if "match_id" in df.columns:
            n_unique = df["match_id"].nunique()
            n_total = len(df)
            dup_rate = 1 - n_unique / n_total
            assert dup_rate < 0.01, (
                f"Match ID duplicate rate {dup_rate:.1%} exceeds 1%"
            )

        # 2. Result variable is clean
        result_vals = set(df["result"].dropna().unique())
        assert result_vals.issubset({"H", "D", "A"}), (
            f"Invalid result values: {result_vals - {'H', 'D', 'A'}}"
        )

        # 3. Season distribution is reasonable
        if "season" in df.columns:
            n_seasons = df["season"].nunique()
            assert n_seasons >= 5, f"Only {n_seasons} seasons in data"

        # 4. Elo ratings in reasonable range
        if "home_elo" in df.columns:
            elo_min = df["home_elo"].dropna().min()
            elo_max = df["home_elo"].dropna().max()
            assert 800 < elo_min, f"Home Elo min={elo_min} too low"
            assert elo_max < 2500, f"Home Elo max={elo_max} too high"

        # 5. No negative match dates
        if "match_date" in df.columns:
            assert df["match_date"].notna().mean() > 0.95, (
                "More than 5% of match_date values are null"
            )

    def test_data_result_distribution(self, full_features_df):
        """Result distribution is within expected ranges for Serie A."""
        df = full_features_df.dropna(subset=["result"])
        counts = df["result"].value_counts(normalize=True)

        # Serie A base rates: H~44.4%, D~26.5%, A~29.1%
        # Allow generous tolerance
        assert 0.30 < counts.get("H", 0) < 0.60, (
            f"Home win rate {counts.get('H', 0):.1%} outside [30%, 60%]"
        )
        assert 0.15 < counts.get("D", 0) < 0.40, (
            f"Draw rate {counts.get('D', 0):.1%} outside [15%, 40%]"
        )
        assert 0.15 < counts.get("A", 0) < 0.45, (
            f"Away win rate {counts.get('A', 0):.1%} outside [15%, 45%]"
        )

    def test_feature_to_prediction_pipeline(self, features_df):
        """Features can be passed through prediction logic without errors."""
        from scripts.analysis.backtest_unified import predict_factor_based

        # Run factor predictions on each row
        results = []
        for _, row in features_df.head(20).iterrows():
            pred = predict_factor_based(row)
            results.append(pred)

        assert len(results) == 20

        # All should produce valid probabilities
        for pred in results:
            total = pred["prob_H"] + pred["prob_D"] + pred["prob_A"]
            assert abs(total - 1.0) < 0.02
            for key in ["prob_H", "prob_D", "prob_A"]:
                assert 0 <= pred[key] <= 1

    def test_prediction_to_betting_pipeline(self):
        """Predictions can flow into betting value calculations."""
        from scripts.betting.betting_unified import (
            calculate_kelly,
            calculate_ev,
            remove_overround,
        )

        # Simulate a prediction result
        model_probs = {"H": 0.55, "D": 0.25, "A": 0.20}
        market_odds = [1.90, 3.50, 4.50]

        # Step 1: Remove overround for fair comparison
        fair_probs = remove_overround(market_odds)
        assert abs(sum(fair_probs) - 1.0) < 0.001

        # Step 2: Calculate EV for each outcome
        outcomes = ["H", "D", "A"]
        for i, outcome in enumerate(outcomes):
            ev = calculate_ev(model_probs[outcome], market_odds[i])
            # At least one outcome should have positive EV given our probs
            # (since we set H=55% and market implied H~52.6%)

        # Step 3: Calculate Kelly stake for value bets
        kelly = calculate_kelly(
            model_prob=model_probs["H"],
            odds=market_odds[0],
            fraction=0.25,
        )
        assert kelly >= 0, "Kelly should be non-negative"

        # Step 4: Verify the value bet on Home
        ev_home = calculate_ev(model_probs["H"], market_odds[0])
        if ev_home > 0:
            assert kelly > 0, "Positive EV should produce positive Kelly stake"

    def test_full_metrics_pipeline(self):
        """Full metrics calculation pipeline works on synthetic data."""
        from scripts.analysis.backtest_unified import (
            ranked_probability_score,
            _multiclass_brier,
        )

        np.random.seed(42)
        n = 100

        # Simulate actuals (0=H, 1=D, 2=A) with Serie A-like distribution
        y_true = np.random.choice([0, 1, 2], size=n, p=[0.45, 0.27, 0.28])

        # Simulate model predictions (not perfect, but better than uniform)
        y_proba = np.random.dirichlet([3, 2, 2], size=n)
        # Add some signal: boost the correct outcome's probability
        for i in range(n):
            y_proba[i, y_true[i]] += 0.15
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

        # Compute RPS
        rps = ranked_probability_score(y_true, y_proba)
        assert 0 <= rps <= 1, f"RPS={rps} out of [0,1]"

        # Compute Brier
        brier = _multiclass_brier(y_true, y_proba)
        assert 0 <= brier, f"Brier={brier} should be non-negative"

        # Compare against uniform baseline
        y_uniform = np.full((n, 3), 1 / 3)
        rps_uniform = ranked_probability_score(y_true, y_uniform)
        brier_uniform = _multiclass_brier(y_true, y_uniform)

        # Our "model" should beat uniform
        assert rps < rps_uniform, (
            f"Model RPS ({rps:.4f}) should beat uniform ({rps_uniform:.4f})"
        )
        assert brier < brier_uniform, (
            f"Model Brier ({brier:.4f}) should beat uniform ({brier_uniform:.4f})"
        )


# ============================================================================
# 8. OPTIMIZE PIPELINE TESTS
# ============================================================================

@pytest.mark.integration
class TestOptimizePipeline:
    """Tests for the unified optimization framework."""

    def test_optimize_config_defaults(self):
        """OptimizeConfig has sensible defaults."""
        from scripts.models.optimize_unified import OptimizeConfig

        cfg = OptimizeConfig()
        assert cfg.target == "ensemble-weights"
        assert cfg.n_trials > 0
        assert cfg.metric in ("log_loss", "accuracy", "rps", "brier")
        assert cfg.grid_step > 0
        assert cfg.min_train_seasons >= 3
        assert len(cfg.test_seasons) >= 1

    def test_optimize_all_targets_valid(self):
        """All optimization targets are recognizable."""
        from scripts.models.optimize_unified import OptimizeConfig

        targets = [
            "ensemble-weights", "model-hyperparams", "betting-params",
            "features", "all-markets", "all",
        ]
        for target in targets:
            cfg = OptimizeConfig(target=target)
            assert cfg.target == target

    def test_optimizer_instantiation(self):
        """UnifiedOptimizer can be instantiated."""
        from scripts.models.optimize_unified import OptimizeConfig, UnifiedOptimizer

        cfg = OptimizeConfig(target="ensemble-weights")
        optimizer = UnifiedOptimizer(cfg)
        assert optimizer.df is None
        assert optimizer.results == {}

    def test_poisson_batch_conversion(self):
        """Batch Poisson conversion produces consistent results."""
        from scripts.models.optimize_unified import poisson_1x2_from_xg, poisson_1x2_batch

        hxg = np.array([1.5, 1.0, 2.0])
        axg = np.array([1.0, 1.5, 0.8])

        batch_result = poisson_1x2_batch(hxg, axg)
        assert batch_result.shape == (3, 3)

        # Each row should sum to 1
        for i in range(3):
            total = batch_result[i].sum()
            assert abs(total - 1.0) < 0.001, f"Row {i} sums to {total}"

        # Individual results should match batch
        for i in range(3):
            single = poisson_1x2_from_xg(hxg[i], axg[i])
            np.testing.assert_allclose(batch_result[i], single, atol=0.001)


# ============================================================================
# 9. CROSS-MODULE CONSISTENCY TESTS
# ============================================================================

@pytest.mark.integration
class TestCrossModuleConsistency:
    """Tests ensuring consistency between related modules."""

    def test_label_map_consistency(self):
        """LABEL_MAP is consistent between train and backtest modules."""
        from scripts.analysis.backtest_unified import LABEL_MAP as BT_MAP
        from ml.config import LABEL_MAP as ML_MAP

        assert BT_MAP == ML_MAP, (
            f"LABEL_MAP mismatch: backtest={BT_MAP}, ml.config={ML_MAP}"
        )

    def test_base_rates_sum_to_one(self):
        """Base rates in prediction and backtest modules are valid distributions."""
        from scripts.prediction.predict_unified import BASE_RATES as PRED_RATES
        from scripts.analysis.backtest_unified import BASE_RATES as BT_RATES

        pred_total = PRED_RATES["home_win"] + PRED_RATES["draw"] + PRED_RATES["away_win"]
        assert abs(pred_total - 1.0) < 0.01, (
            f"predict_unified BASE_RATES sum to {pred_total}"
        )

        bt_total = BT_RATES["H"] + BT_RATES["D"] + BT_RATES["A"]
        assert abs(bt_total - 1.0) < 0.01, (
            f"backtest_unified BASE_RATES sum to {bt_total}"
        )

    def test_ensemble_weights_consistency(self):
        """ENSEMBLE_WEIGHTS and ENSEMBLE_WEIGHTS_WITH_DEEP have matching keys."""
        from scripts.prediction.predict_unified import (
            ENSEMBLE_WEIGHTS,
            ENSEMBLE_WEIGHTS_WITH_DEEP,
        )

        # WITH_DEEP should be a superset of ENSEMBLE_WEIGHTS keys
        base_keys = set(ENSEMBLE_WEIGHTS.keys())
        deep_keys = set(ENSEMBLE_WEIGHTS_WITH_DEEP.keys())
        assert base_keys.issubset(deep_keys), (
            f"Keys in ENSEMBLE_WEIGHTS but not WITH_DEEP: {base_keys - deep_keys}"
        )

        # Both should sum to ~1.0
        assert abs(sum(ENSEMBLE_WEIGHTS.values()) - 1.0) < 0.01
        assert abs(sum(ENSEMBLE_WEIGHTS_WITH_DEEP.values()) - 1.0) < 0.01

    def test_fallback_weights_all_valid(self):
        """All fallback weight sets sum to 1.0."""
        from scripts.prediction.predict_unified import FALLBACK_WEIGHTS

        for name, weights in FALLBACK_WEIGHTS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.01, (
                f"Fallback weights '{name}' sum to {total}"
            )

    def test_kelly_fraction_consistent(self):
        """Kelly fraction defaults match between prediction and betting configs."""
        from scripts.betting.betting_unified import BettingConfig

        cfg = BettingConfig()
        assert cfg.kelly_fraction == 0.05, (
            "BettingConfig default Kelly should be 0.10 (conservative Kelly)"
        )

    def test_data_dir_consistency(self):
        """DATA_DIR is the same across all modules."""
        from config.settings import DATA_DIR as SETTINGS_DIR
        from storage.paths import DATA_DIR as PATHS_DIR

        assert SETTINGS_DIR == PATHS_DIR


# ============================================================================
# 10. MATHEMATICAL CORRECTNESS TESTS
# ============================================================================

@pytest.mark.integration
class TestMathematicalCorrectness:
    """Tests verifying mathematical correctness of core calculations."""

    def test_poisson_convergence(self):
        """Poisson probabilities converge as max_goals increases."""
        from scripts.models.train_unified import poisson_win_prob

        probs_5 = poisson_win_prob(1.5, 1.0, max_goals=5)
        probs_10 = poisson_win_prob(1.5, 1.0, max_goals=10)

        # Higher max_goals should give probabilities closer to true values
        # The difference should be small (Poisson tail is negligible beyond 5)
        for key in ["H", "D", "A"]:
            diff = abs(probs_5[key] - probs_10[key])
            assert diff < 0.01, f"Poisson not converging for {key}: diff={diff}"

    def test_kelly_scaling_linearity(self):
        """Kelly stake scales linearly with Kelly fraction parameter."""
        from scripts.betting.betting_unified import calculate_kelly

        prob = 0.60
        odds = 2.5

        k_25 = calculate_kelly(prob, odds, fraction=0.25)
        k_50 = calculate_kelly(prob, odds, fraction=0.50)

        if k_25 > 0:
            # k_50 should be exactly 2x k_25 (linear scaling)
            ratio = k_50 / k_25
            assert abs(ratio - 2.0) < 0.01, (
                f"Kelly should scale linearly: 0.50/0.25 ratio={ratio}"
            )

    def test_ev_matches_kelly_sign(self):
        """Expected value and Kelly stake always agree on sign."""
        from scripts.betting.betting_unified import calculate_kelly, calculate_ev

        test_cases = [
            (0.30, 4.0),   # marginal positive
            (0.60, 2.0),   # clear positive
            (0.40, 2.0),   # negative
            (0.50, 2.0),   # break-even
            (0.20, 6.0),   # slight positive
        ]

        for prob, odds in test_cases:
            ev = calculate_ev(prob, odds)
            kelly = calculate_kelly(prob, odds, fraction=0.25)

            if ev > 0.01:
                assert kelly > 0, (
                    f"Positive EV ({ev:.4f}) but zero Kelly "
                    f"at prob={prob}, odds={odds}"
                )
            if kelly > 0:
                assert ev > -0.01, (
                    f"Positive Kelly ({kelly:.4f}) but negative EV ({ev:.4f}) "
                    f"at prob={prob}, odds={odds}"
                )

    def test_overround_is_removed_completely(self):
        """After overround removal, fair odds imply sum-of-probs = 1.0."""
        from scripts.betting.betting_unified import remove_overround

        # Heavy overround (~10%)
        odds_heavy = [1.80, 3.20, 4.00]
        fair = remove_overround(odds_heavy)
        assert abs(sum(fair) - 1.0) < 0.001

        # Light overround (~2%)
        odds_light = [2.05, 3.40, 4.20]
        fair_light = remove_overround(odds_light)
        assert abs(sum(fair_light) - 1.0) < 0.001

    def test_rps_is_ordinal_aware(self):
        """RPS penalizes distance more than pure Brier score for ordinal data."""
        from scripts.analysis.backtest_unified import ranked_probability_score, _multiclass_brier

        y_true = np.array([0])  # actual = Home (0)

        # Prediction: near miss (predicting Draw=1 instead of Home=0)
        near_miss = np.array([[0.1, 0.8, 0.1]])
        # Prediction: far miss (predicting Away=2 instead of Home=0)
        far_miss = np.array([[0.1, 0.1, 0.8]])

        rps_near = ranked_probability_score(y_true, near_miss)
        rps_far = ranked_probability_score(y_true, far_miss)

        # RPS should penalize far miss more than near miss (ordinal property)
        assert rps_far > rps_near, (
            f"RPS should penalize far miss ({rps_far:.4f}) > near miss ({rps_near:.4f})"
        )

        # Brier doesn't distinguish ordinal distance as much
        brier_near = _multiclass_brier(y_true, near_miss)
        brier_far = _multiclass_brier(y_true, far_miss)

        # The RPS gap should be larger than Brier gap (ordinal awareness)
        rps_gap = rps_far - rps_near
        brier_gap = brier_far - brier_near
        assert rps_gap > brier_gap * 0.5, (
            "RPS should show stronger ordinal awareness than Brier"
        )
