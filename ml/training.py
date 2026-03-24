"""Model training with walk-forward validation, tuning, and ensemble."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd

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
from ml.tuning import _compute_sample_weights
from storage.paths import features_path

log = logging.getLogger(__name__)


def _strip_meta(X: pd.DataFrame) -> pd.DataFrame:
    """Drop metadata columns before training, and deduplicate any repeated columns."""
    out = X.drop(columns=[c for c in META_COLS if c in X.columns])
    # Safety: deduplicate columns (LightGBM rejects duplicates)
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()]
    return out


# ---------------------------------------------------------------------------
# Walk-forward CV (now with sample weights + tuned params)
# ---------------------------------------------------------------------------

def walk_forward_validate(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    config: ValidationConfig | None = None,
    params: dict | None = None,
    use_sample_weights: bool = True,
) -> pd.DataFrame:
    """Run walk-forward cross-validation and return per-fold metrics."""
    config = config or ValidationConfig()
    splitter = TimeSeriesSplitter(config)
    splits = splitter.generate_splits(X["_season"])

    fold_rows: list[dict] = []
    for fold_idx, (train_seasons, test_seasons) in enumerate(splits):
        train_mask = X["_season"].isin(train_seasons)
        test_mask = X["_season"].isin(test_seasons)

        X_train = _strip_meta(X[train_mask])
        y_train = y[train_mask]
        X_test = _strip_meta(X[test_mask])
        y_test = y[test_mask]

        model = get_model(model_type, params)

        fit_kwargs = {}
        if use_sample_weights:
            train_seasons_s = X[train_mask]["_season"] if "_season" in X.columns else None
            fit_kwargs["sample_weight"] = _compute_sample_weights(y_train, seasons=train_seasons_s)

        model.fit(X_train, y_train, **fit_kwargs)
        y_proba = model.predict_proba(X_test)

        metrics = compute_metrics(y_test, y_proba)
        fold_rows.append({
            "fold": fold_idx,
            "train": ", ".join(train_seasons),
            "test": ", ".join(test_seasons),
            "n_train": len(X_train),
            "n_test": len(X_test),
            **metrics,
        })

        log.info(
            "Fold %d  test=%s  acc=%.3f  logloss=%.3f  f1_D=%.3f",
            fold_idx, test_seasons, metrics["accuracy"],
            metrics["log_loss"], metrics["f1_D"],
        )

    return pd.DataFrame(fold_rows)


# ---------------------------------------------------------------------------
# Standard training (baseline + improved)
# ---------------------------------------------------------------------------

def train_universal(
    model_types: List[str] | None = None,
    validate: bool = True,
    params: Dict[str, dict] | None = None,
    use_sample_weights: bool = True,
    feature_names_override: list[str] | None = None,
) -> Dict:
    """Train models using universal features across all seasons.

    params: optional {model_type: {param_dict}} for tuned hyperparams
    """
    model_types = model_types or list(MODEL_TYPES)
    params = params or {}
    fp = str(features_path())
    loader = DataLoader(fp)
    X, y, feature_names = loader.get_universal_dataset()

    if feature_names_override:
        keep = [c for c in feature_names_override if c in X.columns]
        meta = [c for c in X.columns if c.startswith("_")]
        X = X[keep + [c for c in meta if c not in keep]]
        feature_names = keep

    results: Dict = {}

    for mt in model_types:
        mt_params = params.get(mt)
        log.info("=== Training %s (universal, %d features) ===", mt, len(feature_names))

        if validate:
            cv = walk_forward_validate(
                X, y, model_type=mt, params=mt_params,
                use_sample_weights=use_sample_weights,
            )
            results[f"{mt}_cv"] = cv
            avg = cv[["accuracy", "log_loss", "brier_score"]].mean().to_dict()
            log.info(
                "%s CV avg: acc=%.4f  logloss=%.4f  brier=%.4f",
                mt, avg["accuracy"], avg["log_loss"], avg["brier_score"],
            )

        # Final model on all data
        model = get_model(mt, mt_params)
        fit_kwargs = {}
        if use_sample_weights:
            all_seasons = X["_season"] if "_season" in X.columns else None
            fit_kwargs["sample_weight"] = _compute_sample_weights(y, seasons=all_seasons)
        model.fit(_strip_meta(X), y, **fit_kwargs)

        # Evaluate on last season as quick sanity check
        last_season = sorted(X["_season"].unique())[-1]
        last_mask = X["_season"] == last_season
        y_proba = model.predict_proba(_strip_meta(X[last_mask]))
        final_metrics = compute_metrics(y[last_mask], y_proba)

        path = save_model(model, "universal", mt, feature_names, final_metrics)
        results[f"{mt}_path"] = str(path)

    return results


def train_rich(
    season: str | None = None,
    model_types: List[str] | None = None,
    params: Dict[str, dict] | None = None,
    use_sample_weights: bool = True,
) -> Dict:
    """Train models using all features for one FBref-rich season.

    If season is None, uses the latest season in the data.
    """
    model_types = model_types or list(MODEL_TYPES)
    params = params or {}
    feat_cfg = FeatureConfig()
    fp = str(features_path())
    loader = DataLoader(fp)
    X, y, feature_names = loader.get_rich_dataset(season)

    if len(X) < feat_cfg.min_rich_samples:
        log.warning("Only %d samples for rich model; skipping", len(X))
        return {}

    results: Dict = {}

    # Chronological split
    split = int(len(X) * (1.0 - feat_cfg.rich_val_ratio))
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    for mt in model_types:
        mt_params = params.get(mt)
        log.info(
            "=== Training %s (rich, %d features, %s) ===",
            mt, len(feature_names), season or "latest",
        )

        model = get_model(mt, mt_params)
        fit_kwargs = {}
        if use_sample_weights:
            fit_kwargs["sample_weight"] = _compute_sample_weights(y_train)
        model.fit(_strip_meta(X_train), y_train, **fit_kwargs)

        y_proba = model.predict_proba(_strip_meta(X_val))
        val_metrics = compute_metrics(y_val, y_proba)
        log.info(
            "%s validation: acc=%.4f  logloss=%.4f",
            mt, val_metrics["accuracy"], val_metrics["log_loss"],
        )

        # Retrain on all data
        model = get_model(mt, mt_params)
        fit_kwargs_all = {}
        if use_sample_weights:
            fit_kwargs_all["sample_weight"] = _compute_sample_weights(y)
        model.fit(_strip_meta(X), y, **fit_kwargs_all)

        path = save_model(model, "rich", mt, feature_names, val_metrics)
        results[f"{mt}_path"] = str(path)
        results[f"{mt}_val_metrics"] = val_metrics

    return results


# ---------------------------------------------------------------------------
# Full optimized pipeline
# ---------------------------------------------------------------------------

def train_optimized(
    n_tune_trials: int | None = None,
    top_k_features: int | None = None,
    corr_threshold: float | None = None,
    exclude_odds: bool = False,
) -> Dict:
    """Full optimized training pipeline:

    0. (Optional) Exclude odds-derived features
    1. Feature selection (importance pruning + correlation dedup)
    2. Optuna hyperparameter tuning over walk-forward CV
    3. Walk-forward CV with tuned params + sample weights
    4. Fit probability calibrators
    5. Train + evaluate weighted average ensemble
    6. Save everything

    Args:
        exclude_odds: if True, remove all odds-derived features before training.
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

    if n_tune_trials is not None:
        tuning_cfg.n_trials = n_tune_trials
    if top_k_features is None:
        top_k_features = feat_cfg.max_features
    if corr_threshold is None:
        corr_threshold = feat_cfg.correlation_threshold

    fp = str(features_path())
    loader = DataLoader(fp)
    X, y, feature_names = loader.get_universal_dataset()

    # Extract context columns from the raw DataFrame for correction layer OOF persistence.
    # These columns (matchweek, home_team, etc.) may not be in the universal feature set
    # but are needed for correction layer training (training-serving parity).
    _context_cols = ["matchweek", "home_is_promoted", "away_is_promoted",
                     "home_elo", "away_elo", "home_form_points_5", "away_form_points_5",
                     "home_rest_days", "away_rest_days", "home_team", "away_team"]
    _raw_mask = loader.df["result"].isin(["H", "D", "A"])
    _avail = [c for c in _context_cols if c in loader.df.columns]
    context_for_oof = loader.df.loc[_raw_mask, _avail].reset_index(drop=True)

    results: Dict = {}

    # --- Step 0 (optional): Exclude odds features ---
    if exclude_odds:
        log.info("=" * 60)
        log.info("STEP 0: Excluding odds-derived features")
        log.info("=" * 60)
        X, feature_names = _exclude_odds(X, feature_names)

    # --- Step 1: Feature selection ---
    log.info("=" * 60)
    log.info("STEP 1: Feature selection (top_k=%s, corr_threshold=%.2f)",
             top_k_features, corr_threshold)
    log.info("=" * 60)
    selected_feats, importance = importance_based_selection(
        X, y, feature_names, model_type="xgboost", top_k=top_k_features,
    )
    selected_feats = correlation_pruning(
        _strip_meta(X), selected_feats, importance, threshold=corr_threshold,
    )
    results["n_features_selected"] = len(selected_feats)
    results["n_features_original"] = len(feature_names)
    results["exclude_odds"] = exclude_odds
    log.info("Features: %d -> %d", len(feature_names), len(selected_feats))

    # Log top 20 features by importance
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    log.info("Top 20 features by importance:")
    for i, (feat, score) in enumerate(sorted_imp[:20]):
        marker = " [ODDS]" if any(feat.startswith(p) for p in ["odds_", "pinnacle_", "implied_prob_"]) else ""
        log.info("  %2d. %-40s  %.4f%s", i + 1, feat, score, marker)

    # Persist importance history for drift detection across runs
    save_importance_history(importance, selected_feats, variant="universal")

    # Filter X to selected features (avoid duplicating _has_* columns already selected)
    selected_set = set(selected_feats)
    meta = [c for c in X.columns if c.startswith("_") and c not in selected_set]
    X_sel = X[selected_feats + meta].copy()

    # --- Step 2: Hyperparameter tuning ---
    log.info("=" * 60)
    log.info("STEP 2: Hyperparameter tuning")
    log.info("=" * 60)
    tuned_params: Dict[str, dict] = {}
    for mt in MODEL_TYPES:
        log.info("Tuning %s (%d trials)...", mt, tuning_cfg.n_trials)
        result = tune_model(
            X_sel, y, model_type=mt,
            tuning_config=tuning_cfg,
        )
        tuned_params[mt] = result["best_params"]
        results[f"{mt}_tune_score"] = result["best_score"]
        log.info("%s best CV log_loss: %.4f", mt, result["best_score"])

    results["tuned_params"] = tuned_params

    # --- Step 3: Walk-forward CV with tuned params ---
    log.info("=" * 60)
    log.info("STEP 3: Walk-forward CV (tuned + weighted)")
    log.info("=" * 60)
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

    # --- Step 4: Ensemble CV (calibration now happens inside ensemble) ---
    log.info("=" * 60)
    log.info("STEP 4: Weighted average ensemble evaluation (with post-blend calibration)")
    log.info("=" * 60)
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

    # --- Step 5: Train final models + ensemble on all data ---
    log.info("=" * 60)
    log.info("STEP 5: Training final production models")
    log.info("=" * 60)
    for mt in MODEL_TYPES:
        model = get_model(mt, tuned_params[mt])
        all_seasons = X_sel["_season"] if "_season" in X_sel.columns else None
        sw = _compute_sample_weights(y, seasons=all_seasons)
        model.fit(_strip_meta(X_sel), y, sample_weight=sw)

        last_season = sorted(X_sel["_season"].unique())[-1]
        last_mask = X_sel["_season"] == last_season
        y_proba = model.predict_proba(_strip_meta(X_sel[last_mask]))
        final_metrics = compute_metrics(y[last_mask], y_proba)

        path = save_model(model, "universal", mt, selected_feats, final_metrics)
        results[f"{mt}_path"] = str(path)

    # Train and save weighted average ensemble
    # Pass context_for_oof so OOF persistence can source context features (matchweek,
    # is_promoted, etc.) that may not be in the feature-selected X_sel.
    ensemble = WeightedAverageEnsemble(tuned_params, use_sample_weights=True)
    ensemble.fit(X_sel, y, selected_feats, context_df=context_for_oof)
    ensemble.save("universal")
    results["ensemble_path"] = str(MODELS_DIR / "universal" / "ensemble")

    # --- Step 6: Train correction layer on OOF predictions ---
    log.info("=" * 60)
    log.info("STEP 6: Training prediction correction layer")
    log.info("=" * 60)
    try:
        from ml.correction_layer import CorrectionLayer

        cv_preds_path = MODELS_DIR / "universal" / "cv_predictions.parquet"
        if cv_preds_path.exists():
            oof_df = pd.read_parquet(cv_preds_path)

            # Load raw features for context join
            features_df = pd.read_parquet(str(features_path()))

            layer = CorrectionLayer()
            correction_metrics = layer.train_static(oof_df, features_df)
            results["correction_layer"] = correction_metrics

            if correction_metrics.get("skipped"):
                log.info("Correction layer skipped: %s (baseline ll=%.4f)",
                         correction_metrics.get("reason", "unknown"),
                         correction_metrics.get("log_loss_before", 0))
                # Delete stale model if it exists so we don't use an old one
                stale_path = MODELS_DIR / "universal" / "correction_layer.pkl"
                if stale_path.exists():
                    stale_path.unlink()
                    log.info("Removed stale correction model at %s", stale_path)
            else:
                layer.save()
                log.info("Correction layer: ll %.4f → %.4f (Δ=%.4f), ECE %.4f → %.4f",
                         correction_metrics["log_loss_before"],
                         correction_metrics["log_loss_after"],
                         correction_metrics["log_loss_improvement"],
                         correction_metrics["ece_before"],
                         correction_metrics["ece_after"])
        else:
            log.warning("No OOF predictions found at %s — skipping correction layer",
                        cv_preds_path)
    except Exception as e:
        log.warning("Correction layer training failed (non-fatal): %s", e)

    # --- Final summary ---
    log.info("=" * 60)
    log.info("FINAL RESULTS")
    log.info("=" * 60)
    log.info("Features: %d -> %d (odds excluded: %s)",
             len(feature_names), len(selected_feats), exclude_odds)
    for mt in MODEL_TYPES:
        cv = results[f"{mt}_cv"]
        avg = cv[["accuracy", "log_loss", "brier_score"]].mean()
        avg_f1d = cv["f1_D"].mean()
        log.info("%s CV: acc=%.4f  logloss=%.4f  brier=%.4f  f1_D=%.4f",
                 mt, avg["accuracy"], avg["log_loss"], avg["brier_score"], avg_f1d)
    log.info("Ensemble CV: acc=%.4f  ll=%.4f  brier=%.4f  rps=%.4f  ece=%.4f  kelly=%.4f  f1_D=%.4f",
             avg_ens["ensemble_accuracy"], avg_ens["ensemble_log_loss"],
             avg_ens["ensemble_brier"], avg_ens_rps, avg_ens_ece, avg_ens_kelly, avg_ens_f1d)

    # --- Save full training report JSON ---
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_type": "weighted_average_ensemble",
        "note": "This report covers the 3-model ensemble (XGB+LGB+CB). "
                "The no-odds CatBoost model has its own metadata at "
                "catboost_no_odds_metadata.json.",
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
    report_dir = MODELS_DIR / "universal"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    log.info("Saved training report to %s", report_path)

    return results
