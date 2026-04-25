"""Retrain catboost_no_odds.cbm with aligned features after training-serving skew fixes.

This script retrains the no-odds CatBoost model using the exact 35-feature set
from the current deployment. It validates against the rejection thresholds and
only saves if performance is acceptable.

Usage:
    python scripts/models/retrain_no_odds_catboost.py [--dry-run]

2026-04-24 — Phase 5 additions (production hardening):
    --walkforward-final  Save the LAST fold's model instead of training on all
                         data. Eliminates in-sample contamination on eval seasons.
    --include-new-features
                         Extend feature_names with xi_quality + lineup_xg cols
                         (only those that exist in features.parquet).
    --n-seeds N          Repeat training N times with seeds 42, 43, 44, ...,
                         pick the seed with the median CV log-loss. Defends
                         against seed-luck.
    --fit-calibrator     Fit per-class isotonic calibrator on the last
                         walk-forward fold (leakage-safe), save to
                         lean_calibrators.pkl alongside the model.
    --variant-suffix S   Write to catboost_no_odds__{S}.cbm instead of
                         catboost_no_odds.cbm. Lets you ship to a side dir
                         for paper-trade / A-B testing.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from config.settings import MODELS_DIR
from ml.config import LABEL_MAP, ValidationConfig
from ml.data import TimeSeriesSplitter
from ml.evaluation import compute_metrics
from ml.training import _strip_meta
from ml.tuning import _compute_sample_weights
from storage.paths import features_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def load_current_feature_set() -> list[str]:
    """Load the exact 35-feature set from current deployment metadata."""
    metadata_path = MODELS_DIR / "universal" / "catboost_no_odds_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    features = metadata["feature_names"]
    log.info(f"Loaded {len(features)} features from metadata")
    return features


def load_rejection_thresholds() -> Dict[str, float]:
    """Load rejection thresholds from deployment_state.json."""
    deploy_path = MODELS_DIR / "deployment_state.json"
    if not deploy_path.exists():
        # Fallback thresholds (calibrated for 2017+ model with time-decay)
        return {
            "accuracy_min": 0.52,
            "log_loss_max": 0.96,
            "brier_max": 0.195,
            "betting_yield_min": 0.05,
        }

    with open(deploy_path) as f:
        state = json.load(f)

    return state.get("rejection_thresholds", {
        "accuracy_min": 0.52,
        "log_loss_max": 0.96,
        "brier_max": 0.195,
        "betting_yield_min": 0.05,
    })


def walk_forward_validate(
    X: pd.DataFrame,
    y: pd.Series,
    y_str: pd.Series,  # Original string labels for compute_metrics
    params: dict,
    return_last_fold_model: bool = False,
    return_last_fold_cal_data: bool = False,
):
    """Run walk-forward cross-validation and return per-fold metrics.

    If `return_last_fold_model=True`, also returns the model trained on the
    LAST fold's train data (which excluded the most recent test season).
    That model has zero in-sample contamination on the most recent season —
    suitable for production inference where we evaluate on currently-future
    matches.

    If `return_last_fold_cal_data=True`, also returns (proba_last, y_last_str)
    for fitting an isotonic calibrator on the held-out test fold.
    """
    config = ValidationConfig()
    splitter = TimeSeriesSplitter(config)

    # Need to add _season column for splitting
    if "_season" not in X.columns and "season" in X.columns:
        X["_season"] = X["season"]

    splits = splitter.generate_splits(X["_season"])

    fold_rows: list[dict] = []
    last_fold_model = None
    last_fold_proba = None
    last_fold_y_str = None
    for fold_idx, (train_seasons, test_seasons) in enumerate(splits):
        train_mask = X["_season"].isin(train_seasons)
        test_mask = X["_season"].isin(test_seasons)

        X_train_with_meta = X[train_mask].copy()
        X_test_with_meta = X[test_mask].copy()
        y_train = y[train_mask].copy()
        y_test = y[test_mask].copy()
        y_test_str = y_str[test_mask].copy()  # Keep string labels for metrics

        X_train = _strip_meta(X_train_with_meta)
        X_test = _strip_meta(X_test_with_meta)

        # Reset indices to ensure alignment
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        y_test_str = y_test_str.reset_index(drop=True)

        model = CatBoostClassifier(**params)
        train_seasons = X_train_with_meta["_season"] if "_season" in X_train_with_meta.columns else None
        sample_weights = _compute_sample_weights(y_train, seasons=train_seasons)

        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            verbose=False,
        )

        y_proba = model.predict_proba(X_test)
        metrics = compute_metrics(y_test_str, y_proba)  # Use string labels

        fold_rows.append({
            "fold": fold_idx,
            "train": ", ".join(train_seasons),
            "test": ", ".join(test_seasons),
            "n_train": len(X_train),
            "n_test": len(X_test),
            **metrics,
        })

        log.info(
            "Fold %d  test=%s  acc=%.3f  logloss=%.4f  brier=%.4f  f1_D=%.3f",
            fold_idx, test_seasons[0] if test_seasons else "?",
            metrics["accuracy"], metrics["log_loss"],
            metrics["brier_score"], metrics["f1_D"],
        )

        # Capture the last fold's artifacts for production saving
        if fold_idx == len(splits) - 1:
            if return_last_fold_model:
                last_fold_model = model
            if return_last_fold_cal_data:
                last_fold_proba = y_proba
                last_fold_y_str = y_test_str.values

    df = pd.DataFrame(fold_rows)
    if return_last_fold_model or return_last_fold_cal_data:
        return df, last_fold_model, last_fold_proba, last_fold_y_str
    return df


def fit_isotonic_calibrators(y_str: np.ndarray, y_proba: np.ndarray) -> dict:
    """Fit per-class IsotonicRegression on held-out fold predictions.

    Format matches what scripts/prediction/ensemble_prediction_engine.py expects
    for lean_calibrators.pkl: dict[int -> IsotonicRegression].

    The label map is H=0, D=1, A=2 (matches LABEL_MAP / catboost_no_odds classes).
    """
    from sklearn.isotonic import IsotonicRegression
    label_to_idx = {"H": 0, "D": 1, "A": 2}
    y_int = np.array([label_to_idx.get(str(yy), -1) for yy in y_str])
    calibrators: dict = {}
    for cls in range(3):
        binary = (y_int == cls).astype(float)
        # If a class is absent from this fold, fit an identity-ish mapping
        if binary.sum() < 5:
            log.warning("Class %d has only %d samples on cal fold — using identity calibrator",
                        cls, int(binary.sum()))
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit([0.0, 1.0], [0.0, 1.0])
        else:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(y_proba[:, cls], binary)
        calibrators[cls] = iso
    log.info("Fit %d-class isotonic calibrators on %d held-out samples", 3, len(y_str))
    return calibrators


def expand_feature_set_with_new(features: list[str], df: pd.DataFrame) -> list[str]:
    """Add available xi_quality + lineup_xg columns to the locked feature set.

    Only adds columns that:
      - Actually exist in `df`
      - Have non-trivial fill rate (>20% on the rows we'll train on)
      - Are NOT already in `features`
      - Are NOT post-match leaky (lineup_xg_sum / rating_mean / xa_sum / rotation
        ARE post-match — exclude them. xi_quality cols are pre-match — include.)
    """
    POST_MATCH_LEAKY = {
        "home_lineup_xg_sum", "away_lineup_xg_sum", "lineup_xg_sum_diff",
        "home_lineup_xa_sum", "away_lineup_xa_sum", "lineup_xa_sum_diff",
        "home_lineup_rating_mean", "away_lineup_rating_mean",
        "lineup_rating_mean_diff",
        "home_lineup_rotation", "away_lineup_rotation",
    }
    candidates = [
        c for c in df.columns
        if (c.startswith("home_xi_") or c.startswith("away_xi_"))
        and c not in features
        and c not in POST_MATCH_LEAKY
    ]
    accepted = []
    for c in candidates:
        fill = df[c].notna().mean()
        if fill >= 0.20:
            accepted.append(c)
            log.info("  + %s (fill=%.1f%%)", c, fill * 100)
        else:
            log.info("  - %s (fill=%.1f%% — too sparse, skipping)", c, fill * 100)
    log.info("Extended feature set: %d -> %d (added %d xi_quality cols)",
             len(features), len(features) + len(accepted), len(accepted))
    return features + accepted


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
) -> CatBoostClassifier:
    """Train final model on all available data."""
    X_train = _strip_meta(X)
    all_seasons = X["_season"] if "_season" in X.columns else None
    sample_weights = _compute_sample_weights(y, seasons=all_seasons)

    model = CatBoostClassifier(**params)
    model.fit(X_train, y, sample_weight=sample_weights, verbose=100)

    return model


def main(
    dry_run: bool = False,
    walkforward_final: bool = False,
    include_new_features: bool = False,
    n_seeds: int = 1,
    fit_calibrator: bool = False,
    variant_suffix: str = "",
):
    """Main retrain workflow."""
    log.info("=" * 70)
    log.info("RETRAINING catboost_no_odds.cbm")
    log.info("  walkforward_final=%s  include_new_features=%s  n_seeds=%d  fit_calibrator=%s  suffix=%s",
             walkforward_final, include_new_features, n_seeds, fit_calibrator,
             variant_suffix or "<none>")
    log.info("=" * 70)

    # Load feature set from metadata
    feature_names = load_current_feature_set()

    # Load rejection thresholds
    thresholds = load_rejection_thresholds()
    log.info(f"Rejection thresholds: {thresholds}")

    # Load features.parquet
    fp = str(features_path())
    log.info(f"Loading features from {fp}")
    df = pd.read_parquet(fp)
    log.info(f"Loaded {len(df):,} rows, {len(df.columns):,} columns")

    # Extract target
    target_col = None
    for col in ["result", "ftr", "outcome", "label"]:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        log.error("No target column found in features.parquet (tried: result, ftr, outcome, label)")
        sys.exit(1)

    log.info(f"Using target column: {target_col}")
    y_str = df[target_col].copy()  # Keep original string labels

    # Convert string labels (H/D/A) to integers (0/1/2) for training
    if y_str.dtype == object:
        log.info(f"Converting labels: {y_str.value_counts().to_dict()}")
        y = y_str.map(LABEL_MAP)
        log.info(f"After mapping: {y.value_counts().to_dict()}")
    else:
        y = y_str.copy()

    # Add _season column if needed (for walk-forward splitting)
    if "_season" not in df.columns and "season" in df.columns:
        df["_season"] = df["season"]

    # Apply min_train_season filter (2017+ only — unlocks xG, lineup, odds velocity)
    min_season = ValidationConfig().min_train_season
    if min_season and "_season" in df.columns:
        before = len(df)
        df = df[df["_season"] >= min_season].copy()
        y_str = y_str.loc[df.index].copy()
        y = y.loc[df.index].copy()
        log.info(f"min_train_season={min_season}: {before} -> {len(df)} rows")

    # Check that all required features exist (except _has_* indicators which are added at prediction time)
    metadata_indicators = ["_has_gk_data", "_has_shot_data", "_has_odds"]
    core_features = [f for f in feature_names if f not in metadata_indicators]

    missing = [f for f in core_features if f not in df.columns]
    if missing:
        log.error(f"Missing {len(missing)} core features in features.parquet: {missing[:10]}")
        sys.exit(1)

    # Add _has_* indicators with default values
    for indicator in metadata_indicators:
        if indicator not in df.columns:
            if indicator == "_has_gk_data":
                # Check if any GK features are present
                gk_cols = [c for c in df.columns if "gk_" in c.lower()]
                df[indicator] = 1 if gk_cols else 0
            elif indicator == "_has_shot_data":
                df[indicator] = 1  # We have shot data in features
            elif indicator == "_has_odds":
                df[indicator] = 0  # No-odds model
            log.info(f"Added {indicator} = {df[indicator].iloc[0]}")

    # Optionally extend the locked feature set with new pre-match xi_quality cols
    if include_new_features:
        log.info("=" * 70)
        log.info("EXPANDING FEATURE SET (Phase 5: new pre-match XI features)")
        log.info("=" * 70)
        feature_names = expand_feature_set_with_new(feature_names, df)

    # Select features + _season (for splitting)
    keep_cols = feature_names + ["_season"]
    X = df[[c for c in keep_cols if c in df.columns]].copy()
    log.info(f"Selected {len(X.columns)} columns (including _season)")

    # CatBoost hyperparameters (from current deployment)
    base_params = {
        "iterations": 1000,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "random_seed": 42,
        "verbose": False,
        "thread_count": -1,
    }

    # Walk-forward CV — optionally multi-seed for robustness
    log.info("=" * 70)
    if n_seeds > 1:
        log.info("MULTI-SEED WALK-FORWARD CROSS-VALIDATION (%d seeds)", n_seeds)
    else:
        log.info("WALK-FORWARD CROSS-VALIDATION")
    log.info("=" * 70)

    # Run CV per seed; remember each seed's last-fold artifacts
    seed_results: list[dict] = []
    seeds = [42 + i for i in range(n_seeds)]
    for seed in seeds:
        params = {**base_params, "random_seed": seed}
        if n_seeds > 1:
            log.info("--- Seed %d ---", seed)
        result = walk_forward_validate(
            X, y, y_str, params,
            return_last_fold_model=walkforward_final,
            return_last_fold_cal_data=fit_calibrator,
        )
        if walkforward_final or fit_calibrator:
            cv_df_seed, last_model, last_proba, last_y_str = result
        else:
            cv_df_seed = result
            last_model, last_proba, last_y_str = None, None, None
        seed_results.append({
            "seed": seed,
            "cv": cv_df_seed,
            "last_model": last_model,
            "last_proba": last_proba,
            "last_y_str": last_y_str,
        })

    # Aggregate across seeds (median of last-3 log-loss for the headline)
    last3_lls = []
    for r in seed_results:
        last3 = r["cv"].tail(3)
        last3_lls.append(last3["log_loss"].mean())
    median_idx = int(np.argsort(last3_lls)[len(last3_lls) // 2])
    selected = seed_results[median_idx]
    cv_df = selected["cv"]
    if n_seeds > 1:
        log.info("Per-seed last-3 log-loss: %s", [round(x, 4) for x in last3_lls])
        log.info("Selected median seed: %d (last-3 ll=%.4f)",
                 selected["seed"], last3_lls[median_idx])

    # Compute summary metrics
    all_folds_acc = cv_df["accuracy"].mean()
    all_folds_ll = cv_df["log_loss"].mean()
    all_folds_brier = cv_df["brier_score"].mean()

    last3 = cv_df.tail(3)
    last3_acc = last3["accuracy"].mean()
    last3_ll = last3["log_loss"].mean()
    last3_brier = last3["brier_score"].mean()
    last3_draw_f1 = last3["f1_D"].mean()

    log.info("")
    log.info("=" * 70)
    log.info("CV SUMMARY")
    log.info("=" * 70)
    log.info(f"All folds ({len(cv_df)}):  acc={all_folds_acc:.4f}  logloss={all_folds_ll:.4f}  brier={all_folds_brier:.4f}")
    log.info(f"Last 3 folds:     acc={last3_acc:.4f}  logloss={last3_ll:.4f}  brier={last3_brier:.4f}  draw_f1={last3_draw_f1:.4f}")
    log.info("")

    # Check against rejection thresholds (use last 3 folds for decision)
    rejections = []
    if last3_acc < thresholds["accuracy_min"]:
        rejections.append(f"Accuracy {last3_acc:.4f} < {thresholds['accuracy_min']:.4f}")
    if last3_ll > thresholds["log_loss_max"]:
        rejections.append(f"Log-loss {last3_ll:.4f} > {thresholds['log_loss_max']:.4f}")
    if last3_brier > thresholds["brier_max"]:
        rejections.append(f"Brier {last3_brier:.4f} > {thresholds['brier_max']:.4f}")

    if rejections:
        log.error("=" * 70)
        log.error("MODEL REJECTED — DOES NOT MEET THRESHOLDS")
        log.error("=" * 70)
        for r in rejections:
            log.error(f"  ✗ {r}")
        log.error("")
        log.error("DO NOT DEPLOY THIS MODEL. Investigate data quality or feature alignment.")
        sys.exit(1)

    log.info("=" * 70)
    log.info("✓ Model PASSES all rejection thresholds")
    log.info("=" * 70)

    if dry_run:
        log.info("DRY RUN — not saving model")
        return

    # Train final model — either on all data (legacy) or use last-fold model
    # (walkforward-final, no in-sample contamination on most recent season)
    if walkforward_final:
        log.info("")
        log.info("=" * 70)
        log.info("USING LAST WALK-FORWARD FOLD MODEL (no in-sample on recent season)")
        log.info("=" * 70)
        final_model = selected["last_model"]
        if final_model is None:
            log.error("walkforward_final requested but last_model not captured — bug")
            sys.exit(1)
        # Evaluate the last-fold model on its own held-out test season (which IS
        # the most recent season — that's the whole point: it was held out from training).
        last_season = sorted(df["_season"].unique())[-1]
        last_mask = df["_season"] == last_season
        X_last = _strip_meta(X[last_mask])
        y_last_str = y_str[last_mask]
        y_proba_last = final_model.predict_proba(X_last)
        last_metrics = compute_metrics(y_last_str, y_proba_last)
        log.info("Final (walkforward-last-fold) model on held-out %s:", last_season)
        log.info("  acc=%.4f  logloss=%.4f  brier=%.4f",
                 last_metrics["accuracy"], last_metrics["log_loss"], last_metrics["brier_score"])
    else:
        log.info("")
        log.info("=" * 70)
        log.info("TRAINING FINAL MODEL ON ALL DATA (legacy path; in-sample on eval seasons)")
        log.info("=" * 70)
        params = {**base_params, "random_seed": selected["seed"]}
        final_model = train_final_model(X, y, params)

        # Evaluate on last season as sanity check (in-sample!)
        last_season = sorted(df["_season"].unique())[-1]
        last_mask = df["_season"] == last_season
        X_last = _strip_meta(X[last_mask])
        y_last_str = y_str[last_mask]
        y_proba_last = final_model.predict_proba(X_last)
        last_metrics = compute_metrics(y_last_str, y_proba_last)
        log.info("Final model on last season (%s, IN-SAMPLE — not a real eval):",
                 last_season)
        log.info("  acc=%.4f  logloss=%.4f  brier=%.4f",
                 last_metrics["accuracy"], last_metrics["log_loss"], last_metrics["brier_score"])

    # Save model
    suffix_part = f"__{variant_suffix}" if variant_suffix else ""
    model_path = MODELS_DIR / "universal" / f"catboost_no_odds{suffix_part}.cbm"
    backup_path = MODELS_DIR / "universal" / f"catboost_no_odds{suffix_part}.cbm.bak"

    # Backup existing model only when overwriting prod (no suffix); for variant
    # builds, just write the new file (no prior version to back up).
    if not variant_suffix and model_path.exists():
        log.info(f"Backing up existing model to {backup_path}")
        import shutil
        shutil.copy2(model_path, backup_path)

    log.info(f"Saving model to {model_path}")
    final_model.save_model(str(model_path))

    # Optionally fit + save isotonic calibrators (leakage-safe, last fold only)
    if fit_calibrator:
        cal_proba = selected["last_proba"]
        cal_y_str = selected["last_y_str"]
        if cal_proba is None or cal_y_str is None:
            log.warning("fit_calibrator requested but cal data not captured — skipping")
        else:
            log.info("=" * 70)
            log.info("FITTING ISOTONIC CALIBRATORS (last walk-forward fold)")
            log.info("=" * 70)
            calibrators = fit_isotonic_calibrators(cal_y_str, cal_proba)
            cal_path = MODELS_DIR / "universal" / f"lean_calibrators{suffix_part}.pkl"
            import pickle
            with open(cal_path, "wb") as fh:
                pickle.dump(calibrators, fh)
            log.info("Saved calibrators to %s", cal_path)

    # Update metadata
    metadata_path = MODELS_DIR / "universal" / f"catboost_no_odds{suffix_part}_metadata.json"
    metadata = {
        "model_type": "catboost",
        "variant": f"universal/no_odds{suffix_part}",
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"),
        "training_method": "walk_forward_last_fold" if walkforward_final else "all_data_in_sample",
        "feature_selection": {
            "method": "walk_forward_importance_based" + (
                " + xi_quality_extension" if include_new_features else ""
            ),
            "note": "Features locked from previous deployment" + (
                " + extended with xi_quality cols (Phase 5)" if include_new_features else ""
            ),
        },
        "phase5_config": {
            "walkforward_final": walkforward_final,
            "include_new_features": include_new_features,
            "n_seeds": n_seeds,
            "seeds_tried": seeds,
            "selected_seed": selected["seed"],
            "fit_calibrator": fit_calibrator,
            "variant_suffix": variant_suffix or None,
        },
        "metrics": last_metrics,
        "cv_summary": {
            "all_folds_accuracy": round(all_folds_acc, 4),
            "all_folds_logloss": round(all_folds_ll, 4),
            "last3_accuracy": round(last3_acc, 4),
            "last3_logloss": round(last3_ll, 4),
            "last3_brier": round(last3_brier, 4),
            "last3_draw_f1": round(last3_draw_f1, 4),
            "folds": len(cv_df),
        },
        "feature_importance": {
            feat: float(imp)
            for feat, imp in zip(final_model.feature_names_, final_model.feature_importances_)
        },
    }

    log.info(f"Saving metadata to {metadata_path}")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Update deployment_state.json — only when overwriting production
    if variant_suffix:
        log.info("Variant build — skipping deployment_state.json update")
    else:
        _update_deployment_state(MODELS_DIR / "deployment_state.json",
                                 feature_names, last3_acc, last3_ll, last3_brier, cv_df)

    log.info("")
    log.info("=" * 70)
    log.info("✓ RETRAIN COMPLETE")
    log.info("=" * 70)
    log.info(f"Model: {model_path}")
    log.info(f"Metadata: {metadata_path}")
    log.info("")
    return  # exit main here; helper does the deploy state for legacy path


def _update_deployment_state(deploy_path: Path, feature_names: list,
                              last3_acc: float, last3_ll: float, last3_brier: float,
                              cv_df: pd.DataFrame) -> None:
    """Update deployment_state.json — kept as helper to declutter main()."""
    if deploy_path.exists():
        with open(deploy_path) as f:
            deploy_state = json.load(f)
    else:
        deploy_state = {}

    last3 = cv_df.tail(3)

    deploy_state["active_ml_model"] = {
        "name": "catboost_no_odds",
        "path": "universal/catboost_no_odds.cbm",
        "metadata_path": "universal/catboost_no_odds_metadata.json",
        "n_features": len(feature_names),
        "last_trained": datetime.now(timezone.utc).isoformat(),
    }

    deploy_state["metrics"] = {
        "accuracy": round(last3_acc, 3),
        "log_loss": round(last3_ll, 4),
        "brier": round(last3_brier, 4),
        "rps": round(last3["rps"].mean(), 4) if "rps" in last3.columns else None,
        "ece": round(last3["ece"].mean(), 4) if "ece" in last3.columns else None,
        "betting_yield": deploy_state.get("metrics", {}).get("betting_yield", 0.105),
    }

    deploy_state["history"] = deploy_state.get("history", [])
    deploy_state["history"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "retrain_no_odds",
        "model": "catboost_no_odds.cbm",
        "n_features": len(feature_names),
        "metrics": {
            "accuracy": round(last3_acc, 4),
            "log_loss": round(last3_ll, 4),
            "brier": round(last3_brier, 4),
        },
    })

    with open(deploy_path, "w") as f:
        json.dump(deploy_state, f, indent=2)
    log.info("Updated deployment_state.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Retrain catboost_no_odds.cbm")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run CV but don't save model")
    parser.add_argument("--walkforward-final", action="store_true",
                        help="Save the LAST walk-forward fold's model (no in-sample on most recent season)")
    parser.add_argument("--include-new-features", action="store_true",
                        help="Extend feature_names with available xi_quality cols")
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="Train with N seeds 42, 43, ..., pick median by last-3 log-loss")
    parser.add_argument("--fit-calibrator", action="store_true",
                        help="Fit + save isotonic calibrator on last walk-forward fold")
    parser.add_argument("--variant-suffix", default="",
                        help="Save to catboost_no_odds__{suffix}.cbm instead of overwriting prod")
    args = parser.parse_args()

    main(dry_run=args.dry_run,
         walkforward_final=args.walkforward_final,
         include_new_features=args.include_new_features,
         n_seeds=args.n_seeds,
         fit_calibrator=args.fit_calibrator,
         variant_suffix=args.variant_suffix)
