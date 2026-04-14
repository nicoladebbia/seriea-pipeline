"""Ensemble: XGBoost + LightGBM + CatBoost with optimized weighted average blending.

Architecture:
  1. Base models: XGBoost, LightGBM, CatBoost trained with tuned hyperparameters
  2. Each base model produces raw 3-class probabilities
  3. Blending: optimized weighted average of RAW probabilities
     (weights found via scipy.optimize on OOF predictions)
  4. Single calibrator applied to the blended output (not per-model)

Key fix: calibrating per-model before blending destroyed draw probabilities
because isotonic regression per-class distorts relative class proportions.
Blending raw probabilities preserves the models' class discrimination,
then a single post-blend calibration adjusts the final output.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss as sk_log_loss

from config.settings import MODELS_DIR
from ml.calibration import AutoCalibrator, ProbabilityCalibrator
from ml.config import (
    CLASS_INDICES,
    LABEL_MAP,
    LABEL_NAMES,
    META_COLS,
    MODEL_EXTENSIONS,
    N_CLASSES,
    RANDOM_SEED,
    EnsembleConfig,
    ValidationConfig,
)
from ml.data import TimeSeriesSplitter
from ml.evaluation import compute_metrics
from ml.tuning import _build_model, _compute_sample_weights, _fit_with_early_stopping

log = logging.getLogger(__name__)


def _normalize_probs(probs: np.ndarray) -> np.ndarray:
    """Row-normalize probabilities to sum to 1.0 (fixes floating-point drift)."""
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return probs / row_sums


def _optimize_blend_weights(
    probas_list: list[np.ndarray],
    y_true: np.ndarray,
    draw_weight: float = 0.3,
) -> np.ndarray:
    """Find optimal blending weights by minimizing log loss + draw F1 penalty.

    Uses scipy.optimize with softmax parameterization to ensure weights sum to 1.
    The objective is: (1 - draw_weight) * log_loss - draw_weight * draw_F1.
    This prevents the optimizer from giving all weight to the best-log-loss model
    (typically CatBoost) at the expense of draw detection.

    Why draw_weight=0.3: draws are ~27% of Serie A outcomes. A 30% weight ensures
    the optimizer can't ignore draw F1 without increasing the overall objective.
    """
    n_models = len(probas_list)
    if n_models == 1:
        return np.array([1.0])

    draw_idx = LABEL_MAP["D"]

    def objective(raw_weights):
        # Softmax to ensure weights sum to 1 and are positive
        w = np.exp(raw_weights) / np.exp(raw_weights).sum()
        blended = _normalize_probs(sum(w_i * p for w_i, p in zip(w, probas_list)))
        ll = sk_log_loss(y_true, blended, labels=CLASS_INDICES)

        # Compute draw F1 from predicted classes
        y_pred = np.argmax(blended, axis=1)
        draw_mask_true = y_true == draw_idx
        draw_mask_pred = y_pred == draw_idx
        tp = (draw_mask_true & draw_mask_pred).sum()
        fp = (~draw_mask_true & draw_mask_pred).sum()
        fn = (draw_mask_true & ~draw_mask_pred).sum()
        if tp > 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            draw_f1 = 2 * precision * recall / (precision + recall)
        else:
            draw_f1 = 0.0

        # Multi-objective: minimize log-loss, maximize draw F1
        return (1.0 - draw_weight) * ll - draw_weight * draw_f1

    # Start with equal weights
    x0 = np.zeros(n_models)
    result = minimize(objective, x0, method="Nelder-Mead",
                      options={"maxiter": 500, "xatol": 1e-6})
    weights = np.exp(result.x) / np.exp(result.x).sum()
    return weights


class WeightedAverageEnsemble:
    """Weighted average ensemble with post-blend calibration."""

    def __init__(
        self,
        model_configs: Dict[str, dict],
        use_sample_weights: bool = True,
        ensemble_config: EnsembleConfig | None = None,
        variant: str = "universal",
    ):
        self.model_configs = model_configs
        self.use_sample_weights = use_sample_weights
        self.config = ensemble_config or EnsembleConfig()
        self.variant = variant
        self.base_models: Dict[str, Any] = {}
        self.blend_calibrator: AutoCalibrator | ProbabilityCalibrator | None = None
        self.weights: np.ndarray | None = None
        self.model_order: list[str] = []
        self.feature_names: list[str] | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: list[str],
        context_df: pd.DataFrame | None = None,
    ) -> "WeightedAverageEnsemble":
        """Train the weighted average ensemble.

        Args:
            context_df: Optional full features DataFrame (pre-feature-selection)
                        for sourcing context columns (matchweek, is_promoted, etc.)
                        that may not be in the feature-selected X.

        1. Generate OOF predictions via walk-forward for each base model
        2. Optimize blend weights on RAW OOF predictions
        3. Fit a single calibrator on blended OOF predictions
        4. Retrain base models on all data
        """
        self.feature_names = feature_names
        self.model_order = list(self.model_configs.keys())
        config = ValidationConfig()
        splitter = TimeSeriesSplitter(config)
        splits = splitter.generate_splits(X["_season"])

        y_int = y.map(LABEL_MAP)
        n_samples = len(X)
        n_classes = N_CLASSES

        # Collect OOF predictions (RAW, no per-model calibration)
        oof_probas = {mt: np.zeros((n_samples, n_classes)) for mt in self.model_configs}
        oof_mask = np.zeros(n_samples, dtype=bool)
        # Track which positions each model successfully predicted
        oof_model_ok = {mt: np.zeros(n_samples, dtype=bool) for mt in self.model_configs}

        for fold_idx, (train_seasons, test_seasons) in enumerate(splits):
            train_mask = X["_season"].isin(train_seasons)
            test_mask = X["_season"].isin(test_seasons)
            test_indices = X.index[test_mask]

            X_tr = X[train_mask].drop(columns=[c for c in META_COLS if c in X.columns])
            X_tr = X_tr[feature_names]
            y_tr = y_int[train_mask]
            X_te = X[test_mask].drop(columns=[c for c in META_COLS if c in X.columns])
            X_te = X_te[feature_names]

            train_seasons_s = X[train_mask]["_season"] if "_season" in X.columns else None
            sw = _compute_sample_weights(y[train_mask], seasons=train_seasons_s) if self.use_sample_weights else None

            for mt, params in self.model_configs.items():
                try:
                    model = _build_model(mt, params)
                    _fit_with_early_stopping(model, X_tr, y_tr, mt, sample_weight=sw)
                    proba = model.predict_proba(X_te)
                    for i, idx in enumerate(test_indices):
                        pos = X.index.get_loc(idx)
                        oof_probas[mt][pos] = proba[i]
                        oof_model_ok[mt][pos] = True
                except Exception as e:
                    log.warning("Model %s failed on fold %d: %s — excluded from this fold",
                                mt, fold_idx, e)

            oof_mask[test_mask.values] = True
            log.info("Ensemble fold %d: train=%s, test=%s", fold_idx, train_seasons, test_seasons)

        # Optimize blend weights on RAW OOF predictions
        # Only use positions where ALL active models have predictions
        active_models = [mt for mt in self.model_order
                         if oof_model_ok[mt][oof_mask].sum() > 0]
        if not active_models:
            raise RuntimeError("All base models failed during OOF generation")
        if len(active_models) < len(self.model_order):
            failed = set(self.model_order) - set(active_models)
            log.warning("Excluding fully-failed models from blend: %s", failed)
            self.model_order = active_models

        # Intersection mask: only positions where every active model succeeded
        joint_ok = oof_mask.copy()
        for mt in self.model_order:
            joint_ok &= oof_model_ok[mt]

        n_joint = joint_ok.sum()
        n_oof = oof_mask.sum()
        if n_joint < n_oof:
            log.warning("Partial failures reduced OOF samples from %d to %d for weight optimization",
                        n_oof, n_joint)
        if n_joint == 0:
            raise RuntimeError("No positions with complete predictions from all active models")

        oof_y = y[joint_ok]
        oof_y_int = y_int[joint_ok]
        probas_list = [oof_probas[mt][joint_ok] for mt in self.model_order]
        self.weights = _optimize_blend_weights(probas_list, oof_y_int)
        log.info("Blend weights: %s",
                 ", ".join(f"{mt}={w:.3f}" for mt, w in zip(self.model_order, self.weights)))

        # Fit a single calibrator on the BLENDED OOF predictions
        # AutoCalibrator compares isotonic vs temperature scaling, picks best
        blended_oof = _normalize_probs(sum(w * p for w, p in zip(self.weights, probas_list)))
        self.blend_calibrator = AutoCalibrator()
        self.blend_calibrator.fit(oof_y, blended_oof)

        # Persist BOTH raw and calibrated OOF predictions.
        # The correction layer trains on raw (pre-calibration) to avoid double-calibration.
        # Calibrated OOF kept for diagnostics and ensemble evaluation.
        calibrated_oof = _normalize_probs(self.blend_calibrator.calibrate(blended_oof))
        self._persist_oof_predictions(X, joint_ok, calibrated_oof, oof_y, context_df,
                                      raw_oof=blended_oof)

        # Retrain base models on all data
        X_all = X.drop(columns=[c for c in META_COLS if c in X.columns])[feature_names]
        y_all = y_int
        all_seasons_s = X["_season"] if "_season" in X.columns else None
        sw_all = _compute_sample_weights(y, seasons=all_seasons_s) if self.use_sample_weights else None

        for mt, params in self.model_configs.items():
            model = _build_model(mt, params)
            _fit_with_early_stopping(model, X_all, y_all, mt, sample_weight=sw_all)
            self.base_models[mt] = model

        log.info("Base models retrained on all %d samples", len(X_all))
        return self

    def _persist_oof_predictions(
        self,
        X: pd.DataFrame,
        joint_ok: np.ndarray,
        blended_oof: np.ndarray,
        oof_y: pd.Series,
        context_df: pd.DataFrame | None = None,
        raw_oof: np.ndarray | None = None,
    ) -> None:
        """Save OOF predictions + context features for the correction layer.

        Writes to data/models/{variant}/cv_predictions.parquet with columns:
        prob_H, prob_D, prob_A (calibrated), raw_prob_H/D/A (pre-calibration),
        actual, plus context features for correction.

        Args:
            context_df: Optional full features DataFrame (pre-feature-selection)
                        for sourcing columns like matchweek, is_promoted that may
                        have been pruned from the feature-selected X.
            raw_oof: Pre-calibration blended OOF probabilities. The correction
                     layer should train on these to avoid double-calibration.
        """
        try:
            oof_df = pd.DataFrame({
                "prob_H": blended_oof[:, 0],
                "prob_D": blended_oof[:, 1],
                "prob_A": blended_oof[:, 2],
                "actual": oof_y.values,
            })
            # Persist raw (pre-calibration) OOF for the correction layer
            if raw_oof is not None:
                oof_df["raw_prob_H"] = raw_oof[:, 0]
                oof_df["raw_prob_D"] = raw_oof[:, 1]
                oof_df["raw_prob_A"] = raw_oof[:, 2]

            # Extract context features from X (the feature-selected matrix + meta cols)
            X_oof = X[joint_ok].reset_index(drop=True)

            # Context columns — try X first, then fall back to context_df.
            # Names are mapped to match CONTEXT_FEATURE_NAMES in correction_layer.py.
            context_cols = {
                "home_elo": "home_elo",
                "away_elo": "away_elo",
                "home_is_promoted": "is_promoted_home",
                "away_is_promoted": "is_promoted_away",
                "matchweek": "matchweek",
                "home_form_points_5": "home_form_points_5",
                "away_form_points_5": "away_form_points_5",
                "home_rest_days": "home_rest_days",
                "away_rest_days": "away_rest_days",
            }

            # First pass: grab from X_oof (feature-selected)
            for src_col, tgt_col in context_cols.items():
                if src_col in X_oof.columns:
                    oof_df[tgt_col] = X_oof[src_col].values

            # Second pass: for columns NOT found in X_oof (pruned by feature selection),
            # source from the full pre-selection DataFrame. context_df has the same
            # row indexing as X (both from get_universal_dataset), so joint_ok applies.
            if context_df is not None:
                missing = {src: tgt for src, tgt in context_cols.items()
                           if tgt not in oof_df.columns and src in context_df.columns}
                if missing:
                    ctx_oof = context_df[joint_ok].reset_index(drop=True)
                    for src_col, tgt_col in missing.items():
                        if src_col in ctx_oof.columns:
                            oof_df[tgt_col] = ctx_oof[src_col].values

            # Compute derived context features used by the correction layer
            if "home_elo" in oof_df.columns and "away_elo" in oof_df.columns:
                oof_df["elo_gap"] = oof_df["home_elo"].fillna(1500) - oof_df["away_elo"].fillna(1500)
            if "home_form_points_5" in oof_df.columns and "away_form_points_5" in oof_df.columns:
                oof_df["form_gap"] = oof_df["home_form_points_5"].fillna(0) - oof_df["away_form_points_5"].fillna(0)
            if "home_rest_days" in oof_df.columns and "away_rest_days" in oof_df.columns:
                oof_df["rest_gap"] = oof_df["home_rest_days"].fillna(0) - oof_df["away_rest_days"].fillna(0)

            # Meta columns
            if "_season" in X_oof.columns:
                oof_df["season"] = X_oof["_season"].values
            if "_match_date" in X_oof.columns:
                oof_df["match_date"] = X_oof["_match_date"].values

            # Save
            save_path = MODELS_DIR / self.variant / "cv_predictions.parquet"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            oof_df.to_parquet(save_path, index=False)
            log.info("Saved %d OOF predictions to %s (columns: %s)",
                     len(oof_df), save_path, list(oof_df.columns))
        except Exception as e:
            log.warning("Failed to persist OOF predictions: %s", e)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict calibrated weighted average probabilities."""
        X_feat = X[self.feature_names] if self.feature_names else X
        probas = []
        for mt in self.model_order:
            raw = self.base_models[mt].predict_proba(X_feat)
            probas.append(raw)
        blended = _normalize_probs(sum(w * p for w, p in zip(self.weights, probas)))
        if self.blend_calibrator is not None:
            blended = _normalize_probs(self.blend_calibrator.calibrate(blended))
        return blended

    def predict_proba_breakdown(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Return per-model raw probabilities + calibrated ensemble."""
        X_feat = X[self.feature_names] if self.feature_names else X
        result = {}
        probas = []
        for mt in self.model_order:
            raw = self.base_models[mt].predict_proba(X_feat)
            result[mt] = raw
            probas.append(raw)
        blended = _normalize_probs(sum(w * p for w, p in zip(self.weights, probas)))
        if self.blend_calibrator is not None:
            blended = _normalize_probs(self.blend_calibrator.calibrate(blended))
        result["ensemble"] = blended
        return result

    def save(self, variant: str) -> Path:
        """Save the ensemble to disk."""
        save_dir = MODELS_DIR / variant / "ensemble"
        save_dir.mkdir(parents=True, exist_ok=True)

        for mt, model in self.base_models.items():
            ext = MODEL_EXTENSIONS.get(mt, ".pkl")
            if mt == "xgboost":
                model.save_model(str(save_dir / f"{mt}_base{ext}"))
            elif mt == "catboost":
                model.save_model(str(save_dir / f"{mt}_base{ext}"))
            else:
                model.booster_.save_model(str(save_dir / f"{mt}_base{ext}"))

        # Save single blend calibrator
        if self.blend_calibrator is not None:
            if isinstance(self.blend_calibrator, AutoCalibrator):
                self.blend_calibrator.save(self.variant, "blend")
            else:
                with open(save_dir / "blend_calibrator.pkl", "wb") as f:
                    pickle.dump(self.blend_calibrator.calibrators, f)

        metadata = {
            "ensemble_type": "weighted_average_v2",
            "model_configs": {mt: {k: v for k, v in p.items() if not callable(v)}
                              for mt, p in self.model_configs.items()},
            "feature_names": self.feature_names,
            "model_order": self.model_order,
            "weights": self.weights.tolist(),
            "n_base_models": len(self.base_models),
        }
        with open(save_dir / "ensemble_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        log.info("Saved ensemble to %s", save_dir)
        return save_dir

    @classmethod
    def load(cls, variant: str) -> "WeightedAverageEnsemble":
        """Load a saved ensemble from disk."""
        load_dir = MODELS_DIR / variant / "ensemble"
        if not load_dir.exists():
            raise FileNotFoundError(f"No ensemble at {load_dir}")

        with open(load_dir / "ensemble_metadata.json") as f:
            metadata = json.load(f)

        model_configs = metadata["model_configs"]
        ens = cls(model_configs=model_configs)
        ens.feature_names = metadata["feature_names"]
        ens.model_order = metadata.get("model_order", list(model_configs.keys()))
        ens.weights = np.array(metadata["weights"])

        for mt in model_configs:
            ext = MODEL_EXTENSIONS.get(mt, ".pkl")
            if mt == "xgboost":
                import xgboost as xgb
                model = xgb.XGBClassifier()
                model.load_model(str(load_dir / f"{mt}_base{ext}"))
            elif mt == "catboost":
                from catboost import CatBoostClassifier
                model = CatBoostClassifier()
                model.load_model(str(load_dir / f"{mt}_base{ext}"))
            else:
                import lightgbm as lgb
                booster = lgb.Booster(model_file=str(load_dir / f"{mt}_base{ext}"))
                # Wrap Booster so it has predict_proba() like sklearn API
                booster.predict_proba = booster.predict
                model = booster
            ens.base_models[mt] = model

        # Load blend calibrator (supports both legacy ProbabilityCalibrator and AutoCalibrator)
        cal_path = load_dir / "blend_calibrator.pkl"
        if cal_path.exists():
            with open(cal_path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict) and "method" in data:
                # New AutoCalibrator format
                cal = AutoCalibrator()
                cal.load(ens.variant, "blend")
                ens.blend_calibrator = cal
            else:
                # Legacy: list of isotonic regressors
                cal = ProbabilityCalibrator()
                cal.calibrators = data if isinstance(data, list) else []
                cal.fitted = bool(cal.calibrators)
                ens.blend_calibrator = cal

        log.info("Loaded ensemble from %s (weights: %s)", load_dir,
                 ", ".join(f"{mt}={w:.3f}" for mt, w in zip(ens.model_order, ens.weights)))
        return ens


# Keep StackingEnsemble for backward compatibility / loading old models
StackingEnsemble = WeightedAverageEnsemble


def evaluate_ensemble_cv(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    model_configs: Dict[str, dict],
    use_sample_weights: bool = True,
    ensemble_config: EnsembleConfig | None = None,
    save_fold_models: bool = True,
) -> pd.DataFrame:
    """Evaluate weighted average ensemble via walk-forward CV.

    For each outer fold:
      - Inner walk-forward generates OOF predictions for weight optimization
      - Blend weights optimized on RAW OOF predictions
      - Single calibrator fitted on blended OOF predictions
      - Models retrained on all training data, predict test
      - Test predictions blended and calibrated

    If save_fold_models=True, persists each fold's CatBoost model and metadata
    to data/models/universal/fold_models/ for leakage-free backtesting.

    Returns DataFrame of per-fold metrics.
    """
    ens_cfg = ensemble_config or EnsembleConfig()
    config = ValidationConfig()
    splitter = TimeSeriesSplitter(config)
    splits = splitter.generate_splits(X["_season"])

    y_int = y.map(LABEL_MAP)
    n_classes = N_CLASSES
    model_order = list(model_configs.keys())
    fold_rows = []

    for fold_idx, (train_seasons, test_seasons) in enumerate(splits):
        if len(train_seasons) < ens_cfg.min_ensemble_seasons:
            continue

        train_mask = X["_season"].isin(train_seasons)
        test_mask = X["_season"].isin(test_seasons)

        X_train_full = X[train_mask]
        y_train_full = y[train_mask]
        y_train_int = y_int[train_mask]
        X_test = X[test_mask].drop(columns=[c for c in META_COLS if c in X.columns])[feature_names]
        y_test = y[test_mask]

        # --- Inner walk-forward OOF within training seasons ---
        n_train = len(X_train_full)
        oof_probas = {mt: np.zeros((n_train, n_classes)) for mt in model_configs}
        oof_mask = np.zeros(n_train, dtype=bool)
        oof_model_ok = {mt: np.zeros(n_train, dtype=bool) for mt in model_configs}

        for inner_split in range(1, len(train_seasons)):
            inner_train_s = train_seasons[:inner_split]
            inner_val_s = [train_seasons[inner_split]]

            inner_train_mask = X_train_full["_season"].isin(inner_train_s)
            inner_val_mask = X_train_full["_season"].isin(inner_val_s)
            val_indices = X_train_full.index[inner_val_mask]

            X_itr = X_train_full[inner_train_mask].drop(
                columns=[c for c in META_COLS if c in X.columns]
            )[feature_names]
            y_itr = y_train_int[inner_train_mask]
            X_ival = X_train_full[inner_val_mask].drop(
                columns=[c for c in META_COLS if c in X.columns]
            )[feature_names]

            inner_seasons = X_train_full[inner_train_mask]["_season"] if "_season" in X_train_full.columns else None
            sw = _compute_sample_weights(y_train_full[inner_train_mask], seasons=inner_seasons) if use_sample_weights else None

            for mt, params in model_configs.items():
                try:
                    model = _build_model(mt, params)
                    _fit_with_early_stopping(model, X_itr, y_itr, mt, sample_weight=sw)
                    proba = model.predict_proba(X_ival)
                    for i, idx in enumerate(val_indices):
                        pos = X_train_full.index.get_loc(idx)
                        oof_probas[mt][pos] = proba[i]
                        oof_model_ok[mt][pos] = True
                except Exception as e:
                    log.warning("CV fold %d inner split %d: %s failed: %s",
                                fold_idx, inner_split, mt, e)

            oof_mask[inner_val_mask.values] = True

        # --- Optimize weights on RAW OOF (no per-model calibration) ---
        # Only use models that succeeded on at least one fold, and only
        # positions where all active models have predictions.
        active_models = [mt for mt in model_order
                         if oof_model_ok[mt][oof_mask].sum() > 0]
        if not active_models:
            log.warning("CV fold %d: all models failed inner OOF — skipping", fold_idx)
            continue

        if len(active_models) < len(model_order):
            failed = set(model_order) - set(active_models)
            log.warning("CV fold %d: excluding failed models from blend: %s", fold_idx, failed)

        joint_ok = oof_mask.copy()
        for mt in active_models:
            joint_ok &= oof_model_ok[mt]

        if joint_ok.sum() == 0:
            log.warning("CV fold %d: no positions with complete predictions — skipping", fold_idx)
            continue

        oof_y = y_train_full[joint_ok]
        oof_y_int = y_train_int[joint_ok]
        oof_list = [oof_probas[mt][joint_ok] for mt in active_models]
        weights = _optimize_blend_weights(oof_list, oof_y_int)

        # Fit single calibrator on blended OOF (auto-selects best method)
        blended_oof = _normalize_probs(sum(w * p for w, p in zip(weights, oof_list)))
        blend_cal = AutoCalibrator()
        blend_cal.fit(oof_y, blended_oof)

        # --- Retrain base models on ALL training data, predict test ---
        X_tr_all = X_train_full.drop(columns=[c for c in META_COLS if c in X.columns])[feature_names]
        outer_seasons = X_train_full["_season"] if "_season" in X_train_full.columns else None
        sw_all = _compute_sample_weights(y_train_full, seasons=outer_seasons) if use_sample_weights else None
        base_test_probas = {}
        for mt in active_models:
            params = model_configs[mt]
            try:
                model = _build_model(mt, params)
                _fit_with_early_stopping(model, X_tr_all, y_train_int, mt, sample_weight=sw_all)
                base_test_probas[mt] = model.predict_proba(X_test)
            except Exception as e:
                log.warning("CV fold %d: %s failed on full retrain: %s", fold_idx, mt, e)

        # Filter to models that succeeded on retrain
        active_retrained = [mt for mt in active_models if mt in base_test_probas]
        if not active_retrained:
            log.warning("CV fold %d: all models failed retrain — skipping", fold_idx)
            continue

        # Recompute weights if some models dropped during retrain
        if len(active_retrained) < len(active_models):
            dropped = set(active_models) - set(active_retrained)
            log.warning("CV fold %d: dropped %s after retrain failure, reoptimizing weights",
                        fold_idx, dropped)
            oof_list_r = [oof_probas[mt][joint_ok] for mt in active_retrained]
            weights = _optimize_blend_weights(oof_list_r, oof_y_int)

        # --- Weighted average blend + single calibration ---
        test_list = [base_test_probas[mt] for mt in active_retrained]
        y_proba_raw = _normalize_probs(sum(w * p for w, p in zip(weights, test_list)))
        y_proba = _normalize_probs(blend_cal.calibrate(y_proba_raw))

        metrics = compute_metrics(y_test, y_proba)

        # Also compute raw blend metrics (no calibration) for comparison
        metrics_raw = compute_metrics(y_test, y_proba_raw)

        individual = {}
        for mt in active_retrained:
            individual[mt] = compute_metrics(y_test, base_test_probas[mt])
        # Fill missing models with None so DataFrame columns are consistent
        for mt in model_configs:
            if mt not in individual:
                individual[mt] = {k: None for k in ["accuracy", "log_loss", "brier_score", "f1_D"]}

        # Build per-model weight columns (None for models excluded from this fold)
        weight_dict = {mt: None for mt in model_order}
        for mt, w in zip(active_retrained, weights):
            weight_dict[mt] = round(float(w), 4)

        fold_rows.append({
            "fold": fold_idx,
            "test": ", ".join(test_seasons),
            "n_train": sum(train_mask),
            "n_test": sum(test_mask),
            "weights_str": ", ".join(f"{mt}={w:.3f}" for mt, w in zip(active_retrained, weights)),
            **{f"weight_{mt}": weight_dict[mt] for mt in model_order},
            "ensemble_accuracy": metrics["accuracy"],
            "ensemble_log_loss": metrics["log_loss"],
            "ensemble_brier": metrics["brier_score"],
            "ensemble_rps": metrics.get("rps", None),
            "ensemble_ece": metrics.get("ece", None),
            "ensemble_kelly_roi": metrics.get("kelly_roi", None),
            "ensemble_f1_D": metrics["f1_D"],
            "ensemble_raw_accuracy": metrics_raw["accuracy"],
            "ensemble_raw_log_loss": metrics_raw["log_loss"],
            "ensemble_raw_f1_D": metrics_raw["f1_D"],
            **{f"{mt}_accuracy": individual[mt]["accuracy"] for mt in model_configs},
            **{f"{mt}_log_loss": individual[mt]["log_loss"] for mt in model_configs},
            **{f"{mt}_brier": individual[mt]["brier_score"] for mt in model_configs},
            **{f"{mt}_f1_D": individual[mt]["f1_D"] for mt in model_configs},
        })

        log.info(
            "Ensemble fold %d: test=%s  ens_acc=%.3f  ens_ll=%.3f  rps=%.3f  ece=%.3f  f1D=%.3f  (w=%s)",
            fold_idx, test_seasons,
            metrics["accuracy"], metrics["log_loss"],
            metrics.get("rps", 0), metrics.get("ece", 0), metrics["f1_D"],
            ", ".join(f"{mt}={w:.2f}" for mt, w in zip(active_retrained, weights)),
        )

    df = pd.DataFrame(fold_rows)

    # Persist CV results for comparison across training runs
    cv_dir = MODELS_DIR / "universal" / "ensemble"
    cv_dir.mkdir(parents=True, exist_ok=True)
    cv_path = cv_dir / "cv_results.json"
    df.to_json(cv_path, orient="records", indent=2)
    log.info("Saved ensemble CV results to %s", cv_path)

    return df


def build_fold_models(
    feature_names: list[str] | None = None,
) -> Path:
    """Train and save per-fold CatBoost models for leakage-free backtesting.

    For each walk-forward fold, trains CatBoost on the training seasons and
    saves the model keyed by its test season. This allows the backtest to load
    the correct model that has NEVER seen the test data.

    Returns the directory where fold models are saved.

    Saves:
        data/models/universal/fold_models/fold_{idx}_{test_season}.cbm
        data/models/universal/fold_models/fold_index.json
    """
    from ml.data import DataLoader
    from storage.paths import features_path

    fold_dir = MODELS_DIR / "universal" / "fold_models"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    fp = str(features_path())
    loader = DataLoader(fp)
    X, y, all_feature_names = loader.get_universal_dataset()

    # Use provided feature names or load from no-odds metadata
    if feature_names is None:
        meta_path = MODELS_DIR / "universal" / "catboost_no_odds_metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            feature_names = meta["feature_names"]
        else:
            log.warning("No feature names provided and no metadata found; using all features")
            feature_names = all_feature_names

    y_int = y.map(LABEL_MAP)
    config = ValidationConfig()
    splitter = TimeSeriesSplitter(config)
    splits = splitter.generate_splits(X["_season"])

    fold_index = []

    for fold_idx, (train_seasons, test_seasons) in enumerate(splits):
        train_mask = X["_season"].isin(train_seasons)
        X_tr = X[train_mask].drop(columns=[c for c in META_COLS if c in X.columns])

        # Filter to available features
        available = [f for f in feature_names if f in X_tr.columns]
        X_tr = X_tr[available]
        y_tr = y_int[train_mask]

        fold_seasons = X[train_mask]["_season"] if "_season" in X.columns else None
        sw = _compute_sample_weights(y[train_mask], seasons=fold_seasons)

        model = _build_model("catboost", {
            "loss_function": "MultiClass",
            "classes_count": N_CLASSES,
            "depth": 6,
            "learning_rate": 0.05,
            "iterations": 1000,
            "l2_leaf_reg": 3.0,
            "auto_class_weights": "Balanced",
            "random_seed": RANDOM_SEED,
            "verbose": 0,
        })
        _fit_with_early_stopping(model, X_tr, y_tr, "catboost", sample_weight=sw)

        test_season_str = test_seasons[0].replace("-", "_")
        model_path = fold_dir / f"fold_{fold_idx}_{test_season_str}.cbm"
        model.save_model(str(model_path))

        fold_index.append({
            "fold_idx": fold_idx,
            "train_seasons": train_seasons,
            "test_seasons": test_seasons,
            "model_file": model_path.name,
            "n_features": len(available),
            "feature_names": available,
            "n_train": int(train_mask.sum()),
        })

        log.info(
            "Fold %d: trained on %s, saved for test=%s (%d features, %d samples)",
            fold_idx, train_seasons[-1], test_seasons[0], len(available), train_mask.sum(),
        )

    # Save index
    index_path = fold_dir / "fold_index.json"
    index_path.write_text(json.dumps(fold_index, indent=2))
    log.info("Saved %d fold models to %s", len(fold_index), fold_dir)

    return fold_dir


def load_fold_model(test_season: str):
    """Load the CatBoost model trained WITHOUT seeing the given test season.

    Returns (model, feature_names) or (None, None) if no fold model exists.
    """
    fold_dir = MODELS_DIR / "universal" / "fold_models"
    index_path = fold_dir / "fold_index.json"

    if not index_path.exists():
        return None, None

    index = json.loads(index_path.read_text())

    # Find the fold whose test_seasons includes the requested season
    for entry in index:
        if test_season in entry["test_seasons"]:
            model_path = fold_dir / entry["model_file"]
            if model_path.exists():
                from catboost import CatBoostClassifier
                model = CatBoostClassifier()
                model.load_model(str(model_path))
                return model, entry["feature_names"]

    return None, None
