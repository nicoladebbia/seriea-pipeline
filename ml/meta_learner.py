#!/usr/bin/env python3
"""Meta-learner for context-dependent ensemble blending.

Instead of fixed weights (ML=60.5%, Market=20.5%, xG=12.4%, Factor=3.5%),
the meta-learner trains a second-stage model that learns WHEN to trust
which predictor more, based on match context.

The key idea: fixed weights are globally optimal but locally suboptimal.
In derbies, the xG model may overpredict goals. Post-international break,
ML form features may be stale. A meta-learner captures these patterns.

Architecture:
    Stage 1: Each component generates H/D/A probabilities
    Stage 2: Meta-learner takes component probs + context → final H/D/A

Training requires out-of-sample predictions from each component to avoid
information leakage. This module generates them via walk-forward.

Usage:
    # Full evaluation
    python -m ml.meta_learner --evaluate

    # Train and save
    python -m ml.meta_learner --train --save

    # Compare fixed vs meta
    python -m ml.meta_learner --compare
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (mirroring ensemble_prediction_engine.py)
# ---------------------------------------------------------------------------

# Serie A base rates (21 seasons, 7829 matches)
BASE_RATES = {"H": 0.444, "D": 0.265, "A": 0.291}

# Production fixed weights
FIXED_WEIGHTS = {
    "factor": 0.035,
    "xg": 0.124,
    "ml": 0.605,
    "player_xg": 0.032,
    "market": 0.205,
}

# Draw inflation parameters for Poisson (from ensemble_prediction_engine.py)
DRAW_INFLATE_BASE = 1.55
DRAW_INFLATE_SLOPE = 0.20

# xG Poisson calibration (from ensemble_prediction_engine.py)
XG_LEAGUE_AVG = 1.28
XG_HOME_BOOST = 1.06
XG_AWAY_DAMPEN = 0.94

# Seasons with odds data (Pinnacle starts at 2012-2013)
MIN_SEASON_WITH_ODDS = "2012-2013"

# Promoted teams per season (for context features)
PROMOTED_TEAMS: dict[str, set[str]] = {
    "2012-2013": {"Sampdoria", "Pescara", "Torino"},
    "2013-2014": {"Sassuolo", "Verona", "Livorno"},
    "2014-2015": {"Empoli", "Palermo", "Cesena"},
    "2015-2016": {"Bologna", "Carpi", "Frosinone"},
    "2016-2017": {"Cagliari", "Crotone", "Pescara"},
    "2017-2018": {"Benevento", "Spal", "Verona"},
    "2018-2019": {"Parma", "Empoli", "Frosinone"},
    "2019-2020": {"Brescia", "Lecce", "Verona"},
    "2020-2021": {"Benevento", "Crotone", "Spezia"},
    "2021-2022": {"Empoli", "Salernitana", "Venezia"},
    "2022-2023": {"Cremonese", "Lecce", "Monza"},
    "2023-2024": {"Cagliari", "Frosinone", "Genoa"},
    "2024-2025": {"Como", "Parma", "Venezia"},
    "2025-2026": {"Sassuolo", "Pisa", "Cremonese"},
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = DATA_DIR / "models" / "universal"


# ---------------------------------------------------------------------------
# Component Prediction Generators
# ---------------------------------------------------------------------------

def poisson_win_prob(home_xg: float, away_xg: float, max_goals: int = 10) -> Dict[str, float]:
    """Calculate H/D/A probabilities from expected goals using calibrated Poisson.

    Applies draw inflation that's xG-gap-dependent (matches production code).
    """
    home_xg = max(0.3, min(4.0, home_xg))
    away_xg = max(0.3, min(4.0, away_xg))

    h_pmf = [poisson.pmf(g, home_xg) for g in range(max_goals)]
    a_pmf = [poisson.pmf(g, away_xg) for g in range(max_goals)]

    pH = pD = pA = 0.0
    for hg in range(max_goals):
        for ag in range(max_goals):
            p = h_pmf[hg] * a_pmf[ag]
            if hg > ag:
                pH += p
            elif hg == ag:
                pD += p
            else:
                pA += p

    # Draw inflation: calibrated params from production
    xg_gap = abs(home_xg - away_xg)
    draw_inflate = max(0.90, min(DRAW_INFLATE_BASE, DRAW_INFLATE_BASE - DRAW_INFLATE_SLOPE * xg_gap))
    pD *= draw_inflate
    t = pH + pD + pA
    return {"H": pH / t, "D": pD / t, "A": pA / t}


def compute_xg_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Compute xG Poisson predictions for all matches.

    Uses team-level xG attack/defense strength features from features.parquet.
    This is a simplified version — production uses CatBoost regressors for xG,
    but the strength-based Poisson is the core signal.

    Returns DataFrame with columns: xg_prob_H, xg_prob_D, xg_prob_A
    """
    results = []
    for _, row in df.iterrows():
        h_atk = row.get("home_xg_attack_strength", np.nan)
        h_def = row.get("home_xg_defense_strength", np.nan)
        a_atk = row.get("away_xg_attack_strength", np.nan)
        a_def = row.get("away_xg_defense_strength", np.nan)

        if any(pd.isna(v) for v in [h_atk, h_def, a_atk, a_def]):
            results.append({"xg_prob_H": np.nan, "xg_prob_D": np.nan, "xg_prob_A": np.nan})
            continue

        home_xg = h_atk * a_def * XG_LEAGUE_AVG * XG_HOME_BOOST
        away_xg = a_atk * h_def * XG_LEAGUE_AVG * XG_AWAY_DAMPEN

        home_xg = max(0.4, min(3.5, home_xg))
        away_xg = max(0.3, min(3.0, away_xg))

        probs = poisson_win_prob(home_xg, away_xg)
        results.append({"xg_prob_H": probs["H"], "xg_prob_D": probs["D"], "xg_prob_A": probs["A"]})

    return pd.DataFrame(results, index=df.index)


def compute_market_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Compute market-implied probabilities from Pinnacle odds.

    Removes overround using the power method (most accurate for 3-way markets).

    Returns DataFrame with columns: mkt_prob_H, mkt_prob_D, mkt_prob_A
    """
    results = []
    for _, row in df.iterrows():
        h_odds = row.get("odds_PSH", np.nan)
        d_odds = row.get("odds_PSD", np.nan)
        a_odds = row.get("odds_PSA", np.nan)

        if any(pd.isna(v) or v <= 1.0 for v in [h_odds, d_odds, a_odds]):
            results.append({"mkt_prob_H": np.nan, "mkt_prob_D": np.nan, "mkt_prob_A": np.nan})
            continue

        # Raw implied probabilities
        raw_h = 1.0 / h_odds
        raw_d = 1.0 / d_odds
        raw_a = 1.0 / a_odds
        overround = raw_h + raw_d + raw_a

        # Remove overround (normalize to sum=1)
        mkt_h = raw_h / overround
        mkt_d = raw_d / overround
        mkt_a = raw_a / overround

        results.append({"mkt_prob_H": mkt_h, "mkt_prob_D": mkt_d, "mkt_prob_A": mkt_a})

    return pd.DataFrame(results, index=df.index)


def compute_factor_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Compute factor model predictions from team strength features.

    Simplified factor model: Elo-based probabilities with home advantage.
    Production factor model uses market-anchored lifts, but for historical
    data we use the no-market fallback (BASE_RATES + strength adjustments).

    Returns DataFrame with columns: fac_prob_H, fac_prob_D, fac_prob_A
    """
    results = []
    for _, row in df.iterrows():
        elo_diff = row.get("elo_diff", 0)
        atk_diff = row.get("attack_strength_diff", 0)
        def_diff = row.get("defense_strength_diff", 0)

        if any(pd.isna(v) for v in [elo_diff, atk_diff, def_diff]):
            results.append({"fac_prob_H": np.nan, "fac_prob_D": np.nan, "fac_prob_A": np.nan})
            continue

        # Elo-based expected score (logistic function, standard chess Elo)
        # E(home) = 1 / (1 + 10^(-elo_diff/400))
        # Then split into H/D/A
        elo_expected = 1.0 / (1.0 + 10 ** (-elo_diff / 400))

        # Strength adjustment (attack + defense advantage)
        strength_adj = (atk_diff * 0.15 + def_diff * 0.10)

        # Convert expected score to H/D/A (Dixon-Coles style)
        # Draw probability peaks around 0.5 expected, falls at extremes
        draw_base = 0.25 * (1 - abs(elo_expected - 0.5) * 2.5)
        draw_base = max(0.10, min(0.35, draw_base))

        home_raw = elo_expected + strength_adj
        away_raw = 1 - elo_expected - strength_adj

        # Split remaining probability mass
        non_draw = 1 - draw_base
        prob_h = max(0.05, non_draw * home_raw / (home_raw + away_raw + 1e-10))
        prob_a = max(0.05, non_draw * away_raw / (home_raw + away_raw + 1e-10))
        prob_d = 1 - prob_h - prob_a
        prob_d = max(0.05, prob_d)

        # Normalize
        total = prob_h + prob_d + prob_a
        results.append({
            "fac_prob_H": prob_h / total,
            "fac_prob_D": prob_d / total,
            "fac_prob_A": prob_a / total,
        })

    return pd.DataFrame(results, index=df.index)


def compute_ml_predictions_walkforward(
    df: pd.DataFrame,
    min_train_seasons: int = 5,
    features: List[str] = None,
) -> pd.DataFrame:
    """Generate out-of-sample ML predictions via walk-forward.

    For each test season, trains CatBoost on all prior seasons.
    This avoids information leakage — every prediction is OOS.

    Uses the production feature list (35 features from catboost_no_odds.cbm).

    Returns DataFrame with columns: ml_prob_H, ml_prob_D, ml_prob_A
    """
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        log.error("CatBoost not installed — ML predictions unavailable")
        return pd.DataFrame(
            {"ml_prob_H": np.nan, "ml_prob_D": np.nan, "ml_prob_A": np.nan},
            index=df.index,
        )

    # Load production feature list
    if features is None:
        meta_path = MODEL_DIR / "catboost_no_odds_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            features = meta.get("features", [])
        if not features:
            # Fallback: the 35 production features
            features = [
                "elo_diff", "home_stadium_capacity", "attack_strength_diff",
                "away_losing_at_ht_rate", "matchup_competitiveness", "is_late_season",
                "away_gd_per_match", "home_losing_at_ht_rate", "away_roll_5_shots_on_target",
                "home_league_gd", "away_roll_5_yellow_cards", "away_injury_impact",
                "elo_x_form", "congestion_asymmetry", "tenure_x_form",
                "home_gd_roll_5", "home_opp_difficulty_roll_5", "home_chemistry_disruption",
                "combined_disruption", "away_roll_5_clean_sheet", "away_roll_5_red_cards",
                "injury_x_elo", "home_venue_roll_10_points", "home_adj_attack_10",
                "away_points_to_cl_zone", "away_roll_5_corners", "h2h_btts_rate",
                "home_travel_fatigue", "away_ht_lead_hold", "h2h_goals_diff",
                "rolling_gd_diff", "h2h_away_goals_avg",
                "_has_gk_data", "_has_shot_data", "_has_odds",
            ]

    # Filter to features that exist in the DataFrame
    available = [f for f in features if f in df.columns]
    if len(available) < 10:
        log.warning("Too few ML features available (%d/%d)", len(available), len(features))
        return pd.DataFrame(
            {"ml_prob_H": np.nan, "ml_prob_D": np.nan, "ml_prob_A": np.nan},
            index=df.index,
        )

    label_map = {"H": 0, "D": 1, "A": 2}
    seasons = sorted(df["season"].unique())

    # Initialize result array
    ml_preds = pd.DataFrame(
        {"ml_prob_H": np.nan, "ml_prob_D": np.nan, "ml_prob_A": np.nan},
        index=df.index,
    )

    for i, test_season in enumerate(seasons):
        if i < min_train_seasons:
            continue  # Not enough training data

        train_seasons = seasons[:i]
        train_mask = df["season"].isin(train_seasons)
        test_mask = df["season"] == test_season

        train_df = df[train_mask]
        test_df = df[test_mask]

        if len(train_df) < 100 or len(test_df) == 0:
            continue

        X_train = train_df[available]
        y_train = train_df["result"].map(label_map)
        X_test = test_df[available]

        # Drop rows with missing labels
        valid = y_train.notna()
        X_train = X_train[valid]
        y_train = y_train[valid].astype(int)

        if len(X_train) < 100:
            continue

        # Train CatBoost with moderate regularization
        model = CatBoostClassifier(
            iterations=500,
            depth=4,
            learning_rate=0.05,
            l2_leaf_reg=5.0,
            loss_function="MultiClass",
            classes_count=3,
            random_seed=42,
            verbose=0,
            allow_writing_files=False,
        )
        model.fit(X_train, y_train, verbose=0)

        # Predict probabilities
        proba = model.predict_proba(X_test)  # shape (n, 3) → [H, D, A]

        ml_preds.loc[test_mask, "ml_prob_H"] = proba[:, 0]
        ml_preds.loc[test_mask, "ml_prob_D"] = proba[:, 1]
        ml_preds.loc[test_mask, "ml_prob_A"] = proba[:, 2]

        log.info(
            "ML walk-forward: train=%s (%d matches), test=%s (%d matches)",
            f"{train_seasons[0]}..{train_seasons[-1]}", len(X_train),
            test_season, len(test_df),
        )

    n_valid = ml_preds["ml_prob_H"].notna().sum()
    log.info("ML predictions: %d/%d matches have OOS predictions", n_valid, len(df))
    return ml_preds


def compute_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract situational context features for the meta-learner.

    These are the features the meta-learner uses to learn WHEN to adjust
    component weights. They should capture situations where certain
    predictors are systematically better or worse.
    """
    ctx = pd.DataFrame(index=df.index)

    # Rest / congestion
    ctx["rest_advantage"] = df.get("rest_advantage", pd.Series(0, index=df.index))
    ctx["has_rest_advantage"] = (ctx["rest_advantage"].abs() >= 2).astype(float)
    ctx["home_short_rest"] = df.get("home_short_rest", pd.Series(0, index=df.index))
    ctx["away_short_rest"] = df.get("away_short_rest", pd.Series(0, index=df.index))

    # International break
    ctx["post_intl_break"] = (
        df.get("home_post_intl_break", pd.Series(0, index=df.index)).fillna(0).astype(bool) |
        df.get("away_post_intl_break", pd.Series(0, index=df.index)).fillna(0).astype(bool)
    ).astype(float)

    # Derby and midweek
    ctx["is_derby"] = df.get("is_derby", pd.Series(0, index=df.index)).fillna(0).astype(float)
    ctx["is_midweek"] = df.get("is_midweek", pd.Series(0, index=df.index)).fillna(0).astype(float)

    # Match competitiveness (how close the teams are)
    ctx["elo_diff_abs"] = df.get("elo_diff", pd.Series(0, index=df.index)).abs()
    ctx["competitiveness"] = df.get("matchup_competitiveness", pd.Series(0.5, index=df.index))

    # Promoted teams
    for _, row in df.iterrows():
        season = row.get("season", "")
        promoted = PROMOTED_TEAMS.get(season, set())
        ctx.loc[row.name, "home_promoted"] = float(row.get("home_team", "") in promoted)
        ctx.loc[row.name, "away_promoted"] = float(row.get("away_team", "") in promoted)

    # Season phase
    ctx["is_late_season"] = df.get("is_late_season", pd.Series(0, index=df.index)).fillna(0).astype(float)

    # Congestion asymmetry
    ctx["congestion_asymmetry"] = df.get("congestion_asymmetry", pd.Series(0, index=df.index)).fillna(0)

    return ctx


# ---------------------------------------------------------------------------
# Meta-Learner
# ---------------------------------------------------------------------------

@dataclass
class MetaLearnerConfig:
    """Configuration for the meta-learner."""
    min_train_seasons: int = 5          # Minimum seasons before generating OOS predictions
    meta_min_train_seasons: int = 3     # Minimum OOS seasons for meta-learner training
    catboost_iterations: int = 300
    catboost_depth: int = 3             # Shallow to prevent overfitting
    catboost_lr: float = 0.05
    catboost_l2: float = 10.0           # Strong regularization
    temperature: float = 0.75           # ML temperature (synced with production 2026-03-01)
    draw_boost: float = 1.12            # Draw boost (synced with production 2026-03-01)
    post_temperature: float = 0.90      # Post-ensemble temperature (synced with production 2026-03-01)


class MetaLearner:
    """Context-dependent ensemble blending via second-stage learning.

    The meta-learner replaces fixed weights with a model that adapts
    the blend based on match context (rest days, derby, promoted team, etc.).

    Training:
        1. Generate OOS predictions from each component via walk-forward
        2. Combine component predictions + context features as input
        3. Train CatBoost to predict H/D/A from these inputs
        4. Walk-forward CV for evaluation (no future leakage)

    At prediction time:
        1. Get component predictions from ensemble engine (same as now)
        2. Extract context features from match data
        3. Feed both to meta-learner → final H/D/A probabilities
    """

    def __init__(self, config: MetaLearnerConfig = None):
        self.config = config or MetaLearnerConfig()
        self.model = None
        self.feature_names: List[str] = []
        self._component_names = ["xg", "mkt", "fac", "ml"]
        self._outcomes = ["H", "D", "A"]

    def _build_meta_features(
        self,
        xg_preds: pd.DataFrame,
        mkt_preds: pd.DataFrame,
        fac_preds: pd.DataFrame,
        ml_preds: pd.DataFrame,
        ctx: pd.DataFrame,
    ) -> pd.DataFrame:
        """Combine component predictions + context into meta-learner input.

        Creates 12 component prob columns (4 components × 3 outcomes)
        plus context features and derived interaction features.
        """
        meta = pd.DataFrame(index=xg_preds.index)

        # Component probabilities (12 features)
        for out in self._outcomes:
            meta[f"xg_prob_{out}"] = xg_preds[f"xg_prob_{out}"]
            meta[f"mkt_prob_{out}"] = mkt_preds[f"mkt_prob_{out}"]
            meta[f"fac_prob_{out}"] = fac_preds[f"fac_prob_{out}"]
            meta[f"ml_prob_{out}"] = ml_preds[f"ml_prob_{out}"]

        # Component agreement features (how much do they agree?)
        for out in self._outcomes:
            probs = meta[[f"xg_prob_{out}", f"mkt_prob_{out}", f"fac_prob_{out}", f"ml_prob_{out}"]]
            meta[f"std_{out}"] = probs.std(axis=1)         # Disagreement signal
            meta[f"range_{out}"] = probs.max(axis=1) - probs.min(axis=1)

        # ML confidence (max class probability)
        ml_cols = [f"ml_prob_{o}" for o in self._outcomes]
        meta["ml_confidence"] = ml_preds[ml_cols].max(axis=1)

        # xG vs ML disagreement
        meta["xg_ml_diff_H"] = xg_preds["xg_prob_H"] - ml_preds["ml_prob_H"]
        meta["mkt_ml_diff_H"] = mkt_preds["mkt_prob_H"] - ml_preds["ml_prob_H"]

        # Context features
        for col in ctx.columns:
            meta[col] = ctx[col]

        self.feature_names = meta.columns.tolist()
        return meta

    def generate_training_data(
        self,
        df: pd.DataFrame,
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Generate complete training data for the meta-learner.

        Runs all component predictions and context extraction.

        Returns:
            meta_features: DataFrame of meta-learner input features
            components: DataFrame with all component predictions
            labels: Series with H/D/A labels
        """
        if verbose:
            log.info("Generating xG Poisson predictions...")
        xg_preds = compute_xg_predictions(df)

        if verbose:
            log.info("Generating market predictions...")
        mkt_preds = compute_market_predictions(df)

        if verbose:
            log.info("Generating factor predictions...")
        fac_preds = compute_factor_predictions(df)

        if verbose:
            log.info("Generating ML predictions (walk-forward, this takes a few minutes)...")
        ml_preds = compute_ml_predictions_walkforward(
            df, min_train_seasons=self.config.min_train_seasons,
        )

        if verbose:
            log.info("Extracting context features...")
        ctx = compute_context_features(df)

        # Combine into meta features
        meta = self._build_meta_features(xg_preds, mkt_preds, fac_preds, ml_preds, ctx)

        # Labels
        labels = df["result"]

        # Store components for baseline evaluation
        components = pd.DataFrame(index=df.index)
        for col in xg_preds.columns:
            components[col] = xg_preds[col]
        for col in mkt_preds.columns:
            components[col] = mkt_preds[col]
        for col in fac_preds.columns:
            components[col] = fac_preds[col]
        for col in ml_preds.columns:
            components[col] = ml_preds[col]

        return meta, components, labels

    def evaluate_fixed_weights(
        self,
        components: pd.DataFrame,
        labels: pd.Series,
        weights: Dict[str, float] = None,
    ) -> Dict[str, float]:
        """Evaluate fixed-weight ensemble on historical data.

        Uses the same weights as production (or custom weights).
        Only evaluates on rows where all components have predictions.
        """
        if weights is None:
            weights = FIXED_WEIGHTS

        # Map component prefixes to weight keys
        prefix_map = {"xg": "xg", "mkt": "market", "fac": "factor", "ml": "ml"}
        # Skip player_xg (not available historically)
        # Renormalize weights without player_xg
        active_weights = {k: v for k, v in weights.items() if k != "player_xg"}
        total_w = sum(active_weights.values())
        active_weights = {k: v / total_w for k, v in active_weights.items()}

        # Build ensemble predictions
        valid_mask = components.notna().all(axis=1) & labels.notna()
        comp = components[valid_mask]
        lab = labels[valid_mask]

        if len(comp) == 0:
            return {"accuracy": 0, "log_loss": 99, "n_matches": 0}

        ensemble_H = np.zeros(len(comp))
        ensemble_D = np.zeros(len(comp))
        ensemble_A = np.zeros(len(comp))

        for prefix, weight_key in prefix_map.items():
            w = active_weights.get(weight_key, 0)
            if w == 0:
                continue
            ensemble_H += w * comp[f"{prefix}_prob_H"].values
            ensemble_D += w * comp[f"{prefix}_prob_D"].values
            ensemble_A += w * comp[f"{prefix}_prob_A"].values

        # Normalize
        total = ensemble_H + ensemble_D + ensemble_A
        ensemble_H /= total
        ensemble_D /= total
        ensemble_A /= total

        return self._compute_metrics(ensemble_H, ensemble_D, ensemble_A, lab)

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        verbose: bool = True,
    ) -> Dict:
        """Full training and evaluation pipeline.

        1. Generate historical component predictions
        2. Walk-forward train/evaluate meta-learner
        3. Compare against fixed-weight baseline

        Returns dict with metrics for both approaches.
        """
        meta, components, labels = self.generate_training_data(df, verbose=verbose)

        # Evaluate fixed-weight baseline
        if verbose:
            log.info("Evaluating fixed-weight baseline...")
        baseline_metrics = self.evaluate_fixed_weights(components, labels)

        # Walk-forward meta-learner evaluation
        if verbose:
            log.info("Training meta-learner (walk-forward)...")
        meta_metrics = self._walkforward_evaluate(meta, labels, df["season"], verbose=verbose)

        # Per-situation comparison
        if verbose:
            log.info("Computing per-situation breakdown...")
        situation_analysis = self._situation_breakdown(
            meta, components, labels, df,
        )

        result = {
            "baseline": baseline_metrics,
            "meta_learner": meta_metrics,
            "improvement": {
                "accuracy_diff": meta_metrics["accuracy"] - baseline_metrics["accuracy"],
                "log_loss_diff": meta_metrics["log_loss"] - baseline_metrics["log_loss"],
                "brier_diff": meta_metrics.get("brier", 0) - baseline_metrics.get("brier", 0),
            },
            "situations": situation_analysis,
            "n_matches_baseline": baseline_metrics.get("n_matches", 0),
            "n_matches_meta": meta_metrics.get("n_matches", 0),
        }

        if verbose:
            self._print_report(result)

        return result

    def _walkforward_evaluate(
        self,
        meta: pd.DataFrame,
        labels: pd.Series,
        seasons: pd.Series,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Walk-forward evaluation of the meta-learner.

        For each test season, train on all prior seasons with valid data.
        """
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            log.error("CatBoost not installed")
            return {"accuracy": 0, "log_loss": 99, "n_matches": 0}

        label_map = {"H": 0, "D": 1, "A": 2}
        unique_seasons = sorted(seasons.unique())

        all_proba = []
        all_labels = []

        for i, test_season in enumerate(unique_seasons):
            # Need at least meta_min_train_seasons of valid meta data
            train_seasons = unique_seasons[:i]

            # Filter to rows with all features available
            train_mask = seasons.isin(train_seasons)
            test_mask = seasons == test_season

            X_train = meta[train_mask].copy()
            y_train = labels[train_mask].map(label_map)
            X_test = meta[test_mask].copy()
            y_test = labels[test_mask].map(label_map)

            # Drop rows with NaN (missing component predictions)
            train_valid = X_train.notna().all(axis=1) & y_train.notna()
            test_valid = X_test.notna().all(axis=1) & y_test.notna()

            X_train = X_train[train_valid]
            y_train = y_train[train_valid].astype(int)
            X_test = X_test[test_valid]
            y_test = y_test[test_valid].astype(int)

            if len(X_train) < 200 or len(X_test) == 0:
                continue

            cfg = self.config
            model = CatBoostClassifier(
                iterations=cfg.catboost_iterations,
                depth=cfg.catboost_depth,
                learning_rate=cfg.catboost_lr,
                l2_leaf_reg=cfg.catboost_l2,
                loss_function="MultiClass",
                classes_count=3,
                random_seed=42,
                verbose=0,
                allow_writing_files=False,
            )
            model.fit(X_train, y_train, verbose=0)

            proba = model.predict_proba(X_test)
            all_proba.append(proba)
            all_labels.extend(y_test.values)

            if verbose:
                acc = (proba.argmax(axis=1) == y_test.values).mean()
                log.info(
                    "  Meta WF fold: train=%d, test=%s (%d), acc=%.1f%%",
                    len(X_train), test_season, len(X_test), acc * 100,
                )

        if not all_proba:
            return {"accuracy": 0, "log_loss": 99, "n_matches": 0}

        all_proba = np.vstack(all_proba)
        all_labels = np.array(all_labels)

        # Convert to H/D/A format for metrics
        return self._compute_metrics(
            all_proba[:, 0], all_proba[:, 1], all_proba[:, 2],
            pd.Series(["H" if l == 0 else "D" if l == 1 else "A" for l in all_labels]),
        )

    def train_final(self, meta: pd.DataFrame, labels: pd.Series) -> None:
        """Train the final meta-learner on all available data.

        Call this after evaluation to produce the production model.
        """
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            log.error("CatBoost not installed")
            return

        label_map = {"H": 0, "D": 1, "A": 2}

        valid = meta.notna().all(axis=1) & labels.notna()
        X = meta[valid]
        y = labels[valid].map(label_map).astype(int)

        cfg = self.config
        self.model = CatBoostClassifier(
            iterations=cfg.catboost_iterations,
            depth=cfg.catboost_depth,
            learning_rate=cfg.catboost_lr,
            l2_leaf_reg=cfg.catboost_l2,
            loss_function="MultiClass",
            classes_count=3,
            random_seed=42,
            verbose=0,
            allow_writing_files=False,
        )
        self.model.fit(X, y, verbose=0)
        log.info("Final meta-learner trained on %d matches", len(X))

    def predict(
        self,
        component_probs: Dict[str, Dict[str, float]],
        context: Dict[str, float],
    ) -> Dict[str, float]:
        """Predict final H/D/A probabilities using the meta-learner.

        Args:
            component_probs: {"xg": {"H": .., "D": .., "A": ..}, "market": {...}, ...}
            context: {"rest_advantage": 3, "is_derby": 0, ...}

        Returns:
            {"prob_H": .., "prob_D": .., "prob_A": ..}
        """
        if self.model is None:
            log.warning("Meta-learner not trained, falling back to fixed weights")
            return self._fixed_weight_predict(component_probs)

        # Build feature vector
        feat = {}
        prefix_map = {"xg": "xg", "market": "mkt", "factor": "fac", "ml": "ml"}
        for comp_key, prefix in prefix_map.items():
            probs = component_probs.get(comp_key, {})
            for out in self._outcomes:
                feat[f"{prefix}_prob_{out}"] = probs.get(f"prob_{out}", probs.get(out, np.nan))

        # Derived features
        for out in self._outcomes:
            vals = [feat.get(f"{p}_prob_{out}", np.nan) for p in ["xg", "mkt", "fac", "ml"]]
            vals = [v for v in vals if not np.isnan(v)]
            feat[f"std_{out}"] = np.std(vals) if vals else 0
            feat[f"range_{out}"] = (max(vals) - min(vals)) if vals else 0

        ml_vals = [feat.get(f"ml_prob_{o}", 0) for o in self._outcomes]
        feat["ml_confidence"] = max(ml_vals) if ml_vals else 0
        feat["xg_ml_diff_H"] = feat.get("xg_prob_H", 0) - feat.get("ml_prob_H", 0)
        feat["mkt_ml_diff_H"] = feat.get("mkt_prob_H", 0) - feat.get("ml_prob_H", 0)

        # Context
        for k, v in context.items():
            feat[k] = v

        # Ensure all features are present
        X = pd.DataFrame([feat])[self.feature_names]
        proba = self.model.predict_proba(X)[0]

        return {"prob_H": float(proba[0]), "prob_D": float(proba[1]), "prob_A": float(proba[2])}

    def _fixed_weight_predict(self, component_probs: Dict) -> Dict[str, float]:
        """Fallback: fixed-weight prediction."""
        weights = {"xg": 0.124, "market": 0.205, "ml": 0.605, "factor": 0.035}
        # Renormalize to available components
        available = {k: v for k, v in weights.items() if k in component_probs}
        total_w = sum(available.values())
        if total_w == 0:
            return {"prob_H": BASE_RATES["H"], "prob_D": BASE_RATES["D"], "prob_A": BASE_RATES["A"]}

        prob_H = sum(w / total_w * component_probs[k].get("prob_H", component_probs[k].get("H", 0.33))
                     for k, w in available.items())
        prob_D = sum(w / total_w * component_probs[k].get("prob_D", component_probs[k].get("D", 0.33))
                     for k, w in available.items())
        prob_A = sum(w / total_w * component_probs[k].get("prob_A", component_probs[k].get("A", 0.33))
                     for k, w in available.items())

        total = prob_H + prob_D + prob_A
        return {"prob_H": prob_H / total, "prob_D": prob_D / total, "prob_A": prob_A / total}

    def save(self, path: Path = None) -> None:
        """Save the trained meta-learner model."""
        if self.model is None:
            log.warning("No trained model to save")
            return

        path = path or MODEL_DIR / "meta_learner.cbm"
        self.model.save_model(str(path))

        # Save feature names
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump({
                "feature_names": self.feature_names,
                "component_names": self._component_names,
                "config": {
                    "iterations": self.config.catboost_iterations,
                    "depth": self.config.catboost_depth,
                    "lr": self.config.catboost_lr,
                    "l2": self.config.catboost_l2,
                },
            }, f, indent=2)

        log.info("Meta-learner saved to %s", path)

    def load(self, path: Path = None) -> bool:
        """Load a trained meta-learner model."""
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            return False

        path = path or MODEL_DIR / "meta_learner.cbm"
        meta_path = path.with_suffix(".json")

        if not path.exists():
            log.warning("Meta-learner model not found: %s", path)
            return False

        self.model = CatBoostClassifier()
        self.model.load_model(str(path))

        if meta_path.exists():
            with open(meta_path) as f:
                meta_info = json.load(f)
            self.feature_names = meta_info.get("feature_names", [])

        log.info("Meta-learner loaded from %s", path)
        return True

    # --- Metrics and Reporting ---

    @staticmethod
    def _compute_metrics(
        prob_H: np.ndarray,
        prob_D: np.ndarray,
        prob_A: np.ndarray,
        labels: pd.Series,
    ) -> Dict[str, float]:
        """Compute accuracy, log-loss, and Brier score."""
        label_map = {"H": 0, "D": 1, "A": 2}
        y_true = labels.map(label_map).values

        # Accuracy
        preds = np.column_stack([prob_H, prob_D, prob_A])
        predicted = preds.argmax(axis=1)
        accuracy = (predicted == y_true).mean()

        # Log loss
        eps = 1e-15
        prob_H = np.clip(prob_H, eps, 1 - eps)
        prob_D = np.clip(prob_D, eps, 1 - eps)
        prob_A = np.clip(prob_A, eps, 1 - eps)

        log_loss = 0.0
        for i, true_class in enumerate(y_true):
            if true_class == 0:
                log_loss -= np.log(prob_H[i])
            elif true_class == 1:
                log_loss -= np.log(prob_D[i])
            else:
                log_loss -= np.log(prob_A[i])
        log_loss /= len(y_true)

        # Brier score (multi-class)
        brier = 0.0
        for i, true_class in enumerate(y_true):
            one_hot = np.array([1.0 if j == true_class else 0.0 for j in range(3)])
            pred_probs = np.array([prob_H[i], prob_D[i], prob_A[i]])
            brier += np.sum((pred_probs - one_hot) ** 2)
        brier /= len(y_true)

        return {
            "accuracy": round(accuracy, 4),
            "log_loss": round(log_loss, 4),
            "brier": round(brier, 4),
            "n_matches": len(y_true),
        }

    def _situation_breakdown(
        self,
        meta: pd.DataFrame,
        components: pd.DataFrame,
        labels: pd.Series,
        df: pd.DataFrame,
    ) -> Dict:
        """Compute metrics per situation (derby, promoted, rest, etc.)."""
        situations = {}
        valid = meta.notna().all(axis=1) & components.notna().all(axis=1) & labels.notna()

        for tag, condition_fn in [
            ("all", lambda df, meta: pd.Series(True, index=df.index)),
            ("derby", lambda df, meta: df.get("is_derby", pd.Series(0, index=df.index)).fillna(0).astype(bool)),
            ("midweek", lambda df, meta: df.get("is_midweek", pd.Series(0, index=df.index)).fillna(0).astype(bool)),
            ("rest_advantage", lambda df, meta: (df.get("rest_advantage", pd.Series(0, index=df.index)).abs() >= 2)),
            ("post_intl", lambda df, meta: (
                df.get("home_post_intl_break", pd.Series(0, index=df.index)).fillna(0).astype(bool) |
                df.get("away_post_intl_break", pd.Series(0, index=df.index)).fillna(0).astype(bool)
            )),
            ("high_elo_diff", lambda df, meta: (df.get("elo_diff", pd.Series(0, index=df.index)).abs() > 100)),
            ("close_match", lambda df, meta: (df.get("elo_diff", pd.Series(0, index=df.index)).abs() < 30)),
        ]:
            mask = condition_fn(df, meta) & valid
            if mask.sum() < 20:
                continue

            baseline = self.evaluate_fixed_weights(components[mask], labels[mask])
            situations[tag] = {
                "n_matches": int(mask.sum()),
                "baseline_accuracy": baseline["accuracy"],
                "baseline_log_loss": baseline["log_loss"],
            }

        return situations

    def _print_report(self, result: Dict) -> None:
        """Print a comparison report to stdout."""
        b = result["baseline"]
        m = result["meta_learner"]
        imp = result["improvement"]

        print("\n" + "=" * 70)
        print("META-LEARNER vs FIXED WEIGHTS COMPARISON")
        print("=" * 70)
        print(f"\n{'Metric':<20} {'Fixed Weights':>15} {'Meta-Learner':>15} {'Delta':>10}")
        print("-" * 60)
        print(f"{'Accuracy':<20} {b['accuracy']*100:>14.1f}% {m['accuracy']*100:>14.1f}% {imp['accuracy_diff']*100:>+9.2f}pp")
        print(f"{'Log-loss':<20} {b['log_loss']:>15.4f} {m['log_loss']:>15.4f} {imp['log_loss_diff']:>+10.4f}")
        print(f"{'Brier':<20} {b.get('brier', 0):>15.4f} {m.get('brier', 0):>15.4f} {imp['brier_diff']:>+10.4f}")
        print(f"{'Matches':<20} {b['n_matches']:>15d} {m['n_matches']:>15d}")

        situ = result.get("situations", {})
        if situ:
            print(f"\n{'Situation':<20} {'N':>6} {'Fixed Acc':>12} {'Fixed LL':>12}")
            print("-" * 50)
            for tag, data in sorted(situ.items()):
                print(f"{tag:<20} {data['n_matches']:>6d} {data['baseline_accuracy']*100:>11.1f}% {data['baseline_log_loss']:>12.4f}")

        print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """Run meta-learner evaluation from command line."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Meta-learner for ensemble blending")
    parser.add_argument("--evaluate", action="store_true", help="Full evaluation (walk-forward)")
    parser.add_argument("--train", action="store_true", help="Train final model")
    parser.add_argument("--save", action="store_true", help="Save trained model")
    parser.add_argument("--compare", action="store_true", help="Compare fixed vs meta")
    parser.add_argument("--min-season", default=MIN_SEASON_WITH_ODDS,
                        help="Earliest season to include (default: 2012-2013, first with odds)")
    args = parser.parse_args()

    # Default to evaluate if no action specified
    if not any([args.evaluate, args.train, args.compare]):
        args.evaluate = True

    # Load data
    log.info("Loading features.parquet...")
    df = pd.read_parquet(DATA_DIR / "features" / "features.parquet")

    # Filter to seasons with odds data
    df = df[df["season"] >= args.min_season].copy()
    log.info("Using %d matches from %s to %s",
             len(df), df["season"].min(), df["season"].max())

    # Ensure required columns
    if "result" not in df.columns:
        log.error("No 'result' column in features.parquet")
        return
    if "home_team" not in df.columns or "away_team" not in df.columns:
        log.error("Missing team columns in features.parquet")
        return

    meta_learner = MetaLearner()

    if args.evaluate or args.compare:
        result = meta_learner.train_and_evaluate(df, verbose=True)

        # Save results
        report_path = DATA_DIR / "models" / "universal" / "meta_learner_report.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        log.info("Report saved to %s", report_path)

    if args.train:
        log.info("Training final meta-learner on all data...")
        meta, components, labels = meta_learner.generate_training_data(df, verbose=True)
        meta_learner.train_final(meta, labels)

        if args.save:
            meta_learner.save()


if __name__ == "__main__":
    main()
