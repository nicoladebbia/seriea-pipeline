"""Train ML ensemble WITHOUT odds features for independent signal.

The current ML classifier is dominated by bookmaker odds features, making it
redundant with the market predictor in the ensemble. This script trains a
no-odds variant that relies purely on team performance data (Elo, form, xG,
rolling stats, H2H, referee, Sofascore, FBref features).

Usage:
    python -m scripts.models.train_no_odds [--top-k 60] [--skip-tune]
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

from config.settings import MODELS_DIR
from ml.config import LABEL_MAP, META_COLS, ModelConfig, ValidationConfig
from ml.data import DataLoader
from ml.evaluation import compute_metrics, gate_folds
from ml.feature_selection import (
    correlation_pruning,
    exclude_odds,
    importance_based_selection,
    save_importance_history,
)
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

OUTPUT_DIR = MODELS_DIR / "universal" / "no_odds"


def train_no_odds(top_k: int = 60, corr_threshold: float = 0.70) -> Dict:
    """Train a no-odds ML ensemble and return metrics.

    Steps:
    1. Load features, exclude odds columns
    2. Select top_k features by walk-forward importance
    3. Prune correlated features
    4. Train CatBoost + XGBoost + LightGBM via walk-forward CV
    5. Train final ensemble with draw-aware blend
    6. Save models and report
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load data and exclude odds ---
    fp = str(features_path())
    loader = DataLoader(fp)
    X, y, feature_names = loader.get_universal_dataset()
    log.info("Loaded %d rows, %d features", len(X), len(feature_names))

    X_no_odds, feats_no_odds = exclude_odds(X, feature_names)
    log.info("After excluding odds: %d features", len(feats_no_odds))

    # --- Step 2: Feature selection by walk-forward importance ---
    log.info("Running walk-forward importance selection (top_k=%d)...", top_k)
    selected, importance = importance_based_selection(
        X_no_odds, y, feats_no_odds, model_type="xgboost", top_k=top_k,
    )
    log.info("Selected %d features by importance", len(selected))

    # --- Step 3: Correlation pruning ---
    selected = correlation_pruning(X_no_odds, selected, importance, threshold=corr_threshold)
    log.info("After correlation pruning: %d features", len(selected))

    # Save importance history
    save_importance_history(importance, selected, variant="universal/no_odds")

    # Restrict X to selected features + meta columns
    meta_cols = [c for c in X_no_odds.columns if c.startswith("_")]
    keep_cols = selected + [c for c in meta_cols if c not in selected]
    X_sel = X_no_odds[[c for c in keep_cols if c in X_no_odds.columns]].copy()

    # --- Step 4: Walk-forward CV for each model type ---
    model_types = ["xgboost", "lightgbm", "catboost"]
    cv_results = {}

    for mt in model_types:
        log.info("=== Walk-forward CV: %s (%d features) ===", mt, len(selected))
        cv_df = walk_forward_validate(
            X_sel, y, model_type=mt, use_sample_weights=True,
        )
        cv_results[mt] = cv_df

        # Print per-fold metrics
        for _, row in cv_df.iterrows():
            log.info(
                "  Fold %d [%s]: acc=%.3f logloss=%.3f f1_D=%.3f",
                row["fold"], row["test"], row["accuracy"], row["log_loss"],
                row.get("f1_D", 0),
            )

        avg = cv_df[["accuracy", "log_loss"]].mean()
        last3 = gate_folds(cv_df)[["accuracy", "log_loss"]].mean()
        log.info(
            "  %s Overall: acc=%.4f logloss=%.4f | Last 3 folds: acc=%.4f logloss=%.4f",
            mt, avg["accuracy"], avg["log_loss"], last3["accuracy"], last3["log_loss"],
        )

    # --- Step 5: Train final models on all data ---
    log.info("=== Training final models ===")
    final_models = {}
    for mt in model_types:
        model = get_model(mt)
        sw = _compute_sample_weights(y)
        X_train = _strip_meta(X_sel)
        model.fit(X_train, y, sample_weight=sw)

        # Quick sanity: evaluate on last season
        last_season = sorted(X_sel["_season"].unique())[-1]
        last_mask = X_sel["_season"] == last_season
        y_proba = model.predict_proba(_strip_meta(X_sel[last_mask]))
        metrics = compute_metrics(y[last_mask], y_proba)
        log.info("  %s final (last season %s): acc=%.3f logloss=%.3f", mt, last_season, metrics["accuracy"], metrics["log_loss"])

        path = save_model(model, "universal/no_odds", mt, selected, metrics)
        final_models[mt] = {"model": model, "path": str(path), "metrics": metrics}

    # --- Step 6: Build comparison report ---
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "variant": "no_odds",
        "n_features_original": len(feature_names),
        "n_features_after_odds_exclusion": len(feats_no_odds),
        "n_features_selected": len(selected),
        "selected_features": selected,
        "top_20_importance": sorted(importance.items(), key=lambda x: -x[1])[:20],
        "cv_summary": {},
        "final_metrics": {},
    }

    for mt in model_types:
        cv_df = cv_results[mt]
        last3 = gate_folds(cv_df)
        report["cv_summary"][mt] = {
            "all_folds_accuracy": round(cv_df["accuracy"].mean(), 4),
            "all_folds_logloss": round(cv_df["log_loss"].mean(), 4),
            "last3_accuracy": round(last3["accuracy"].mean(), 4),
            "last3_logloss": round(last3["log_loss"].mean(), 4),
            "last3_draw_f1": round(last3["f1_D"].mean(), 4) if "f1_D" in last3.columns else None,
            "per_fold": cv_df[["fold", "test", "accuracy", "log_loss"]].to_dict(orient="records"),
        }
        report["final_metrics"][mt] = final_models[mt]["metrics"]

    # Save report
    report_path = OUTPUT_DIR / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Training report saved to %s", report_path)

    # --- Print comparison with current model ---
    print("\n" + "=" * 70)
    print("NO-ODDS MODEL TRAINING COMPLETE")
    print("=" * 70)
    print(f"Features: {len(selected)} (from {len(feature_names)} total, {len(feats_no_odds)} after odds exclusion)")
    print(f"\nTop 10 features:")
    for feat, score in sorted(importance.items(), key=lambda x: -x[1])[:10]:
        marker = " *" if feat in selected else ""
        print(f"  {score:8.4f}  {feat}{marker}")

    print(f"\nWalk-forward CV (last 3 folds = 2023-2026):")
    print(f"  {'Model':<12} {'Accuracy':>10} {'Log-loss':>10} {'Draw F1':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for mt in model_types:
        s = report["cv_summary"][mt]
        draw_f1 = f"{s['last3_draw_f1']:.4f}" if s.get("last3_draw_f1") else "N/A"
        print(f"  {mt:<12} {s['last3_accuracy']:>10.4f} {s['last3_logloss']:>10.4f} {draw_f1:>10}")

    print(f"\nCurrent model baselines (with odds, last 3 folds):")
    print(f"  42-feature ensemble: acc=0.5320  logloss=0.9745")
    print(f"  106-feature CatBoost: acc=0.5340  logloss=0.9778")
    print("=" * 70)

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train no-odds ML ensemble")
    parser.add_argument("--top-k", type=int, default=60, help="Number of features to select")
    parser.add_argument("--corr-threshold", type=float, default=0.70, help="Correlation pruning threshold")
    args = parser.parse_args()

    report = train_no_odds(top_k=args.top_k, corr_threshold=args.corr_threshold)
