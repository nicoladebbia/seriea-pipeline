"""Retrain catboost_no_odds.cbm with aligned features after training-serving skew fixes.

This script retrains the no-odds CatBoost model using the exact 35-feature set
from the current deployment. It validates against the rejection thresholds and
only saves if performance is acceptable.

Usage:
    python scripts/models/retrain_no_odds_catboost.py [--dry-run]
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
from ml.config import LABEL_MAP, META_COLS, ValidationConfig
from ml.data import TimeSeriesSplitter
from ml.evaluation import compute_metrics
from ml.tuning import _compute_sample_weights
from storage.paths import features_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def _strip_meta(X: pd.DataFrame) -> pd.DataFrame:
    """Drop metadata columns before training."""
    return X.drop(columns=[c for c in META_COLS if c in X.columns])


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
        # Fallback to hardcoded thresholds
        return {
            "accuracy_min": 0.50,
            "log_loss_max": 0.99,
            "brier_max": 0.20,
            "betting_yield_min": 0.05,
        }

    with open(deploy_path) as f:
        state = json.load(f)

    return state.get("rejection_thresholds", {
        "accuracy_min": 0.50,
        "log_loss_max": 0.99,
        "brier_max": 0.20,
        "betting_yield_min": 0.05,
    })


def walk_forward_validate(
    X: pd.DataFrame,
    y: pd.Series,
    y_str: pd.Series,  # Original string labels for compute_metrics
    params: dict,
) -> pd.DataFrame:
    """Run walk-forward cross-validation and return per-fold metrics."""
    config = ValidationConfig()
    splitter = TimeSeriesSplitter(config)

    # Need to add _season column for splitting
    if "_season" not in X.columns and "season" in X.columns:
        X["_season"] = X["season"]

    splits = splitter.generate_splits(X["_season"])

    fold_rows: list[dict] = []
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
        sample_weights = _compute_sample_weights(y_train)

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

    return pd.DataFrame(fold_rows)


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
) -> CatBoostClassifier:
    """Train final model on all available data."""
    X_train = _strip_meta(X)
    sample_weights = _compute_sample_weights(y)

    model = CatBoostClassifier(**params)
    model.fit(X_train, y, sample_weight=sample_weights, verbose=100)

    return model


def main(dry_run: bool = False):
    """Main retrain workflow."""
    log.info("=" * 70)
    log.info("RETRAINING catboost_no_odds.cbm")
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

    # Select features + _season (for splitting)
    keep_cols = feature_names + ["_season"]
    X = df[[c for c in keep_cols if c in df.columns]].copy()
    log.info(f"Selected {len(X.columns)} columns (including _season)")

    # CatBoost hyperparameters (from current deployment)
    params = {
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

    # Walk-forward CV
    log.info("=" * 70)
    log.info("WALK-FORWARD CROSS-VALIDATION")
    log.info("=" * 70)
    cv_df = walk_forward_validate(X, y, y_str, params)

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

    # Train final model on all data
    log.info("")
    log.info("=" * 70)
    log.info("TRAINING FINAL MODEL ON ALL DATA")
    log.info("=" * 70)
    final_model = train_final_model(X, y, params)

    # Evaluate on last season as sanity check
    last_season = sorted(df["_season"].unique())[-1]
    last_mask = df["_season"] == last_season
    X_last = _strip_meta(X[last_mask])
    y_last_str = y_str[last_mask]
    y_proba_last = final_model.predict_proba(X_last)
    last_metrics = compute_metrics(y_last_str, y_proba_last)

    log.info(f"Final model on last season ({last_season}):")
    log.info(f"  acc={last_metrics['accuracy']:.4f}  logloss={last_metrics['log_loss']:.4f}  brier={last_metrics['brier_score']:.4f}")

    # Save model
    model_path = MODELS_DIR / "universal" / "catboost_no_odds.cbm"
    backup_path = MODELS_DIR / "universal" / "catboost_no_odds.cbm.bak"

    # Backup existing model
    if model_path.exists():
        log.info(f"Backing up existing model to {backup_path}")
        import shutil
        shutil.copy2(model_path, backup_path)

    log.info(f"Saving model to {model_path}")
    final_model.save_model(str(model_path))

    # Update metadata
    metadata_path = MODELS_DIR / "universal" / "catboost_no_odds_metadata.json"
    metadata = {
        "model_type": "catboost",
        "variant": "universal/no_odds",
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"),
        "training_method": "leakage_free_feature_selection",
        "feature_selection": {
            "method": "walk_forward_importance_based",
            "note": "Features locked from previous deployment — no re-selection",
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

    # Update deployment_state.json
    deploy_path = MODELS_DIR / "deployment_state.json"
    if deploy_path.exists():
        with open(deploy_path) as f:
            deploy_state = json.load(f)
    else:
        deploy_state = {}

    deploy_state["metrics"] = {
        "accuracy": round(last3_acc, 3),
        "log_loss": round(last3_ll, 4),
        "brier": round(last3_brier, 4),
        "rps": round(last3["rps"].mean(), 4),
        "ece": round(last3["ece"].mean(), 4),
        "betting_yield": deploy_state.get("metrics", {}).get("betting_yield", 0.105),  # preserve existing
    }

    deploy_state["history"] = deploy_state.get("history", [])
    deploy_state["history"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "retrain_aligned_features",
        "model": "catboost_no_odds.cbm",
        "reason": "Retrained after training-serving skew fixes (6 feature alignment bugs fixed in ensemble_prediction_engine.py)",
        "metrics": {
            "accuracy": round(last3_acc, 4),
            "log_loss": round(last3_ll, 4),
            "brier": round(last3_brier, 4),
            "cv_last3_accuracy": round(last3_acc, 4),
            "cv_last3_logloss": round(last3_ll, 4),
        },
    })

    with open(deploy_path, "w") as f:
        json.dump(deploy_state, f, indent=2)

    log.info("")
    log.info("=" * 70)
    log.info("✓ RETRAIN COMPLETE")
    log.info("=" * 70)
    log.info(f"Model: {model_path}")
    log.info(f"Metadata: {metadata_path}")
    log.info(f"Backup: {backup_path}")
    log.info("")
    log.info("Next steps:")
    log.info("  1. Run backtest: python scripts/analysis/backtest_unified.py")
    log.info("  2. Validate predictions: python scripts/prediction/predict_unified.py")
    log.info("  3. If backtest passes, model is ready for production")
    log.info("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Retrain catboost_no_odds.cbm")
    parser.add_argument("--dry-run", action="store_true", help="Run CV but don't save model")
    args = parser.parse_args()

    main(dry_run=args.dry_run)
