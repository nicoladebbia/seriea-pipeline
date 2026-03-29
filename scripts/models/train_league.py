#!/usr/bin/env python3
"""Generic league model trainer -- train the full optimized pipeline for any league.

This is the canonical entry point for per-league model training. It reuses the
exact same methodology as the Serie A pipeline (train_optimized):
  - min_train_season = "2017-2018"
  - time_decay = 0.85 (via ModelConfig)
  - auto draw weights
  - recency-weighted feature selection
  - Optuna hyperparameter tuning
  - Walk-forward CV with sample weights
  - 3-model ensemble (XGB + LGB + CB) with post-blend calibration

The only difference: data is filtered to a single league, and models are saved
to data/models/{league}/ instead of data/models/universal/.

Usage:
    # Train Serie A (default -- equivalent to train_optimized)
    python -m scripts.models.train_league

    # Train Premier League
    python -m scripts.models.train_league --league premier_league

    # Train any league with custom tuning budget
    python -m scripts.models.train_league --league bundesliga --trials 30

    # Dry run: validate data exists, report stats, don't train
    python -m scripts.models.train_league --league premier_league --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import MODELS_DIR
from ml.config import (
    META_COLS,
    MODEL_TYPES,
    FeatureConfig,
    TuningConfig,
    ValidationConfig,
)
from ml.data import DataLoader, TimeSeriesSplitter
from ml.evaluation import compute_metrics, print_report
from ml.models import get_model
from ml.persistence import save_model
from ml.training import _strip_meta, walk_forward_validate
from ml.tuning import _compute_sample_weights
from storage.paths import features_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train_league_model(
    league: str = "serie_a",
    n_tune_trials: int = 40,
    top_k_features: int | None = None,
    corr_threshold: float | None = None,
    exclude_odds: bool = False,
) -> Dict:
    """Train a model for any league using the standard methodology.

    This mirrors train_optimized() but operates on a single league's data
    and saves to data/models/{league}/.

    Args:
        league: League identifier (e.g. "serie_a", "premier_league").
            Must match the 'league' column in features.parquet.
        n_tune_trials: Number of Optuna trials per model type.
        top_k_features: Max features after importance selection (None = use config default).
        corr_threshold: Correlation pruning threshold (None = use config default).
        exclude_odds: If True, remove all odds-derived features before training.

    Returns:
        Dict with CV results, tuned params, model paths, and summary metrics.
    """
    from ml.ensemble import WeightedAverageEnsemble, evaluate_ensemble_cv
    from ml.feature_selection import (
        correlation_pruning,
        exclude_odds as _exclude_odds,
        importance_based_selection,
        save_importance_history,
    )
    from ml.tuning import tune_model

    feat_cfg = FeatureConfig()
    tuning_cfg = TuningConfig()
    tuning_cfg.n_trials = n_tune_trials

    if top_k_features is None:
        top_k_features = feat_cfg.max_features
    if corr_threshold is None:
        corr_threshold = feat_cfg.correlation_threshold

    output_dir = MODELS_DIR / league
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data filtered to this league ---
    fp = str(features_path())
    loader = DataLoader(fp)
    X, y, feature_names = loader.get_universal_dataset(league=league)

    n_matches = len(X)
    n_seasons = X["_season"].nunique()
    seasons = sorted(X["_season"].unique())
    log.info("=" * 60)
    log.info("LEAGUE: %s", league)
    log.info("Matches: %d across %d seasons (%s to %s)",
             n_matches, n_seasons, seasons[0], seasons[-1])
    log.info("Class distribution: %s", y.value_counts().to_dict())
    log.info("=" * 60)

    results: Dict = {
        "league": league,
        "n_matches": n_matches,
        "n_seasons": n_seasons,
        "seasons": seasons,
        "class_distribution": y.value_counts().to_dict(),
    }

    # --- Step 0 (optional): Exclude odds features ---
    if exclude_odds:
        log.info("STEP 0: Excluding odds-derived features")
        X, feature_names = _exclude_odds(X, feature_names)

    # --- Step 1: Feature selection ---
    log.info("STEP 1: Feature selection (top_k=%s, corr_threshold=%.2f)",
             top_k_features, corr_threshold)
    selected_feats, importance = importance_based_selection(
        X, y, feature_names, model_type="xgboost", top_k=top_k_features,
    )
    selected_feats = correlation_pruning(
        _strip_meta(X), selected_feats, importance, threshold=corr_threshold,
    )
    results["n_features_selected"] = len(selected_feats)
    results["n_features_original"] = len(feature_names)
    log.info("Features: %d -> %d", len(feature_names), len(selected_feats))

    # Log top 20 features
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    log.info("Top 20 features by importance:")
    for i, (feat, score) in enumerate(sorted_imp[:20]):
        log.info("  %2d. %-40s  %.4f", i + 1, feat, score)

    save_importance_history(importance, selected_feats, variant=league)

    # Filter X to selected features
    selected_set = set(selected_feats)
    meta = [c for c in X.columns if c.startswith("_") and c not in selected_set]
    X_sel = X[selected_feats + meta].copy()

    # --- Step 2: Hyperparameter tuning ---
    log.info("STEP 2: Hyperparameter tuning (%d trials per model)", tuning_cfg.n_trials)
    tuned_params: Dict[str, dict] = {}
    for mt in MODEL_TYPES:
        log.info("Tuning %s...", mt)
        result = tune_model(
            X_sel, y, model_type=mt,
            tuning_config=tuning_cfg,
        )
        tuned_params[mt] = result["best_params"]
        results[f"{mt}_tune_score"] = result["best_score"]
        log.info("%s best CV log_loss: %.4f", mt, result["best_score"])

    results["tuned_params"] = tuned_params

    # --- Step 3: Walk-forward CV with tuned params ---
    log.info("STEP 3: Walk-forward CV (tuned + weighted)")
    for mt in MODEL_TYPES:
        cv = walk_forward_validate(
            X_sel, y, model_type=mt, params=tuned_params[mt],
            use_sample_weights=True,
        )
        results[f"{mt}_cv"] = cv
        avg = cv[["accuracy", "log_loss", "brier_score"]].mean().to_dict()
        avg_f1d = cv["f1_D"].mean()
        log.info(
            "%s tuned CV: acc=%.4f  logloss=%.4f  brier=%.4f  f1_D=%.4f",
            mt, avg["accuracy"], avg["log_loss"], avg["brier_score"], avg_f1d,
        )

    # --- Step 4: Ensemble CV ---
    log.info("STEP 4: Weighted average ensemble evaluation")
    ensemble_cv = evaluate_ensemble_cv(
        X_sel, y, selected_feats, tuned_params,
        use_sample_weights=True,
    )
    results["ensemble_cv"] = ensemble_cv
    avg_ens = ensemble_cv[["ensemble_accuracy", "ensemble_log_loss", "ensemble_brier"]].mean()
    avg_ens_f1d = ensemble_cv["ensemble_f1_D"].mean()
    avg_ens_rps = ensemble_cv["ensemble_rps"].mean() if "ensemble_rps" in ensemble_cv else 0
    avg_ens_ece = ensemble_cv["ensemble_ece"].mean() if "ensemble_ece" in ensemble_cv else 0
    avg_ens_kelly = ensemble_cv["ensemble_kelly_roi"].mean() if "ensemble_kelly_roi" in ensemble_cv else 0
    log.info(
        "Ensemble CV: acc=%.4f  ll=%.4f  brier=%.4f  rps=%.4f  ece=%.4f  kelly=%.4f  f1_D=%.4f",
        avg_ens["ensemble_accuracy"], avg_ens["ensemble_log_loss"],
        avg_ens["ensemble_brier"], avg_ens_rps, avg_ens_ece, avg_ens_kelly, avg_ens_f1d,
    )

    # --- Step 5: Train final production models ---
    log.info("STEP 5: Training final production models -> %s", output_dir)
    for mt in MODEL_TYPES:
        model = get_model(mt, tuned_params[mt])
        all_seasons = X_sel["_season"] if "_season" in X_sel.columns else None
        sw = _compute_sample_weights(y, seasons=all_seasons)
        model.fit(_strip_meta(X_sel), y, sample_weight=sw)

        last_season = sorted(X_sel["_season"].unique())[-1]
        last_mask = X_sel["_season"] == last_season
        y_proba = model.predict_proba(_strip_meta(X_sel[last_mask]))
        final_metrics = compute_metrics(y[last_mask], y_proba)

        path = save_model(model, league, mt, selected_feats, final_metrics)
        results[f"{mt}_path"] = str(path)

    # Train and save ensemble
    ensemble = WeightedAverageEnsemble(tuned_params, use_sample_weights=True)
    ensemble.fit(X_sel, y, selected_feats)
    ensemble.save(league)
    results["ensemble_path"] = str(output_dir / "ensemble")

    # --- Save training report ---
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "league": league,
        "model_type": "weighted_average_ensemble",
        "n_matches": n_matches,
        "n_seasons": n_seasons,
        "seasons": seasons,
        "n_features_original": len(feature_names),
        "n_features_selected": len(selected_feats),
        "selected_features": selected_feats,
        "exclude_odds": exclude_odds,
        "tuned_params": {mt: {k: v for k, v in p.items() if not callable(v)}
                         for mt, p in tuned_params.items()},
        "per_model_cv": {mt: results[f"{mt}_cv"].to_dict() for mt in MODEL_TYPES},
        "ensemble_cv": ensemble_cv.to_dict(),
        "top_20_features": sorted_imp[:20],
    }
    report_path = output_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    log.info("Saved training report to %s", report_path)

    # --- Final summary ---
    log.info("=" * 60)
    log.info("FINAL RESULTS — %s", league)
    log.info("=" * 60)
    log.info("Features: %d -> %d (odds excluded: %s)",
             len(feature_names), len(selected_feats), exclude_odds)
    for mt in MODEL_TYPES:
        cv = results[f"{mt}_cv"]
        avg = cv[["accuracy", "log_loss", "brier_score"]].mean()
        avg_f1d = cv["f1_D"].mean()
        log.info("%s CV: acc=%.4f  logloss=%.4f  brier=%.4f  f1_D=%.4f",
                 mt, avg["accuracy"], avg["log_loss"], avg["brier_score"], avg_f1d)
    log.info("Ensemble CV: acc=%.4f  ll=%.4f  brier=%.4f  f1_D=%.4f",
             avg_ens["ensemble_accuracy"], avg_ens["ensemble_log_loss"],
             avg_ens["ensemble_brier"], avg_ens_f1d)

    return results


# ---------------------------------------------------------------------------
# Data inspection (dry run)
# ---------------------------------------------------------------------------

def dry_run(league: str) -> Dict:
    """Inspect league data without training. Returns summary stats."""
    fp = str(features_path())
    loader = DataLoader(fp)
    loader.load()

    if "league" not in loader.df.columns:
        log.error("No 'league' column in features.parquet")
        return {"error": "no_league_column"}

    available = sorted(loader.df["league"].unique())
    log.info("Available leagues in features.parquet: %s", available)

    if league not in available:
        log.warning("League '%s' NOT FOUND in features.parquet", league)
        log.info("Available leagues: %s", available)
        return {
            "league": league,
            "found": False,
            "available_leagues": available,
        }

    league_df = loader.df[loader.df["league"] == league]
    val_cfg = ValidationConfig()
    if val_cfg.min_train_season:
        league_df = league_df[league_df["season"] >= val_cfg.min_train_season]

    n_matches = len(league_df)
    n_with_result = league_df["result"].isin(["H", "D", "A"]).sum()
    seasons = sorted(league_df["season"].unique())
    class_dist = league_df[league_df["result"].isin(["H", "D", "A"])]["result"].value_counts().to_dict()

    # Feature coverage
    from features.build import get_ml_feature_columns
    ml_cols = get_ml_feature_columns(league_df)
    nan_pcts = league_df[ml_cols].isna().mean()
    n_universal = (nan_pcts < 0.45).sum()

    log.info("=" * 60)
    log.info("DRY RUN — %s", league)
    log.info("=" * 60)
    log.info("Total matches (post-2017): %d", n_matches)
    log.info("Matches with H/D/A result: %d", n_with_result)
    log.info("Seasons: %s", seasons)
    log.info("Class distribution: %s", class_dist)
    log.info("ML features total: %d, universal (<45%% NaN): %d", len(ml_cols), n_universal)

    # Walk-forward split preview
    if n_with_result > 0:
        result_df = league_df[league_df["result"].isin(["H", "D", "A"])]
        splitter = TimeSeriesSplitter(val_cfg)
        season_series = result_df["season"]
        splits = splitter.generate_splits(season_series)
        log.info("Walk-forward CV folds: %d", len(splits))
        for i, (train_s, test_s) in enumerate(splits):
            n_train = season_series.isin(train_s).sum()
            n_test = season_series.isin(test_s).sum()
            log.info("  Fold %d: train=%s (%d), test=%s (%d)",
                     i, train_s[-1], n_train, test_s[0], n_test)

    return {
        "league": league,
        "found": True,
        "n_matches": n_matches,
        "n_with_result": n_with_result,
        "seasons": seasons,
        "class_distribution": class_dist,
        "n_ml_features": len(ml_cols),
        "n_universal_features": int(n_universal),
        "n_cv_folds": len(splits) if n_with_result > 0 else 0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train ML ensemble for any league",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m scripts.models.train_league --league serie_a
    python -m scripts.models.train_league --league premier_league --trials 40
    python -m scripts.models.train_league --league premier_league --dry-run
        """,
    )
    parser.add_argument(
        "--league", default="serie_a",
        help="League identifier matching 'league' column in features.parquet "
             "(default: serie_a)",
    )
    parser.add_argument(
        "--trials", type=int, default=40,
        help="Optuna tuning trials per model (default: 40)",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Max features after importance selection (default: from config)",
    )
    parser.add_argument(
        "--corr", type=float, default=None,
        help="Correlation pruning threshold (default: from config)",
    )
    parser.add_argument(
        "--exclude-odds", action="store_true",
        help="Exclude betting-odds-derived features",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Inspect data only, don't train",
    )

    args = parser.parse_args()

    if args.dry_run:
        results = dry_run(args.league)
    else:
        results = train_league_model(
            league=args.league,
            n_tune_trials=args.trials,
            top_k_features=args.top_k,
            corr_threshold=args.corr,
            exclude_odds=args.exclude_odds,
        )

    return results


if __name__ == "__main__":
    main()
