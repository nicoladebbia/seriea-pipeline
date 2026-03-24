"""Probability calibration via isotonic regression.

Raw GBM probabilities are often overconfident. This module applies
per-class isotonic regression to calibrate predicted probabilities
using held-out validation data from the last walk-forward fold.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from config.settings import MODELS_DIR
from ml.config import (
    CLASS_INDICES,
    LABEL_MAP,
    META_COLS,
    N_CLASSES,
    CalibrationConfig,
    ValidationConfig,
)

log = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """Per-class isotonic regression calibrator.

    Trains N_CLASSES isotonic regressors (one per class) mapping raw P(class)
    to calibrated P(class), then renormalizes so they sum to 1.
    """

    def __init__(self, config: CalibrationConfig | None = None):
        self.config = config or CalibrationConfig()
        self.calibrators: list[IsotonicRegression] = []
        self.fitted = False

    # Minimum samples required for calibration to be reliable
    MIN_CALIBRATION_SAMPLES = 200

    def fit(
        self,
        y_true: pd.Series,
        y_proba: np.ndarray,
        validate: bool = True,
        validation_fraction: float = 0.3,
    ) -> "ProbabilityCalibrator":
        """Fit isotonic regressors on validation predictions.

        y_true: Series with values 'H', 'D', 'A'
        y_proba: (n, N_CLASSES) array of raw probabilities
        validate: if True, use a held-out split to check that calibration
                  actually improves ECE; fall back to identity if it worsens.
        validation_fraction: fraction of data held out for validation (only
                  used when validate=True).
        """
        n_samples = len(y_true)

        # Minimum sample guard
        if n_samples < self.MIN_CALIBRATION_SAMPLES:
            log.warning(
                "Only %d OOF samples (<%d minimum) — skipping calibration (identity passthrough)",
                n_samples, self.MIN_CALIBRATION_SAMPLES,
            )
            self.calibrators = []
            self.fitted = False
            return self

        y_int = y_true.map(LABEL_MAP).values
        n_classes = y_proba.shape[1]

        if validate:
            # Chronological split: fit on first portion, validate on last portion.
            # This avoids the in-sample bias of validating on the same data we
            # fit isotonic regression on.
            split_idx = int(n_samples * (1.0 - validation_fraction))
            # Need enough in each split
            if split_idx < 50 or (n_samples - split_idx) < 50:
                validate = False  # too small to split, skip validation
            else:
                cal_y_int = y_int[:split_idx]
                cal_proba = y_proba[:split_idx]
                val_y_int = y_int[split_idx:]
                val_proba = y_proba[split_idx:]

        if validate:
            # Fit on calibration split
            self.calibrators = []
            for cls_idx in range(n_classes):
                binary_true = (cal_y_int == cls_idx).astype(float)
                raw_prob = cal_proba[:, cls_idx]
                ir = IsotonicRegression(
                    y_min=self.config.y_min,
                    y_max=self.config.y_max,
                    out_of_bounds="clip",
                )
                ir.fit(raw_prob, binary_true)
                self.calibrators.append(ir)
            self.fitted = True

            # Validate on held-out split
            from ml.evaluation import expected_calibration_error
            ece_before = expected_calibration_error(val_y_int, val_proba)
            calibrated_val = self.calibrate(val_proba)
            ece_after = expected_calibration_error(val_y_int, calibrated_val)

            if ece_after > ece_before * 1.05:  # 5% tolerance
                log.warning(
                    "Calibration worsened ECE on held-out data (%.4f -> %.4f) — using identity passthrough",
                    ece_before, ece_after,
                )
                self.calibrators = []
                self.fitted = False
                return self

            log.info(
                "Calibration validated on held-out data: ECE %.4f -> %.4f (%.1f%% reduction)",
                ece_before, ece_after,
                100 * (1 - ece_after / ece_before) if ece_before > 0 else 0,
            )

            # Validation passed — now refit on ALL data for maximum calibration quality
            self.calibrators = []
            for cls_idx in range(n_classes):
                binary_true = (y_int == cls_idx).astype(float)
                raw_prob = y_proba[:, cls_idx]
                ir = IsotonicRegression(
                    y_min=self.config.y_min,
                    y_max=self.config.y_max,
                    out_of_bounds="clip",
                )
                ir.fit(raw_prob, binary_true)
                self.calibrators.append(ir)
            self.fitted = True
        else:
            # No validation — just fit directly
            self.calibrators = []
            for cls_idx in range(n_classes):
                binary_true = (y_int == cls_idx).astype(float)
                raw_prob = y_proba[:, cls_idx]
                ir = IsotonicRegression(
                    y_min=self.config.y_min,
                    y_max=self.config.y_max,
                    out_of_bounds="clip",
                )
                ir.fit(raw_prob, binary_true)
                self.calibrators.append(ir)
            self.fitted = True

        log.info("Calibrator fitted on %d samples", n_samples)
        return self

    def calibrate(self, y_proba: np.ndarray) -> np.ndarray:
        """Apply calibration to raw probabilities.

        Returns (n, N_CLASSES) array of calibrated probabilities that sum to 1.
        If calibration was skipped or worsened metrics, returns input unchanged.
        """
        if not self.fitted or not self.calibrators:
            return y_proba

        calibrated = np.zeros_like(y_proba)
        for cls_idx, ir in enumerate(self.calibrators):
            calibrated[:, cls_idx] = ir.predict(y_proba[:, cls_idx])

        # Renormalize rows to sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, self.config.min_row_sum)
        calibrated = calibrated / row_sums

        return calibrated

    def save(self, variant: str, model_type: str) -> Path:
        """Save calibrator to disk."""
        save_dir = MODELS_DIR / variant
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"{model_type}_calibrator.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.calibrators, f)
        log.info("Saved calibrator to %s", path)
        return path

    def load(self, variant: str, model_type: str) -> "ProbabilityCalibrator":
        """Load calibrator from disk."""
        path = MODELS_DIR / variant / f"{model_type}_calibrator.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No calibrator at {path}")
        with open(path, "rb") as f:
            self.calibrators = pickle.load(f)
        self.fitted = True
        log.info("Loaded calibrator from %s", path)
        return self


def fit_calibrator_from_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    params: dict | None = None,
    use_sample_weights: bool = True,
) -> Tuple[ProbabilityCalibrator, Dict[str, float]]:
    """Fit a calibrator using the last walk-forward fold as calibration data.

    This avoids data leakage: we train on folds 0..N-1, predict on fold N,
    and use those predictions to fit the calibrator.

    Returns (calibrator, metrics_before_calibration).
    """
    from ml.evaluation import compute_metrics
    from ml.tuning import _build_model, _compute_sample_weights, _fit_with_early_stopping

    config = ValidationConfig()
    from ml.data import TimeSeriesSplitter
    splitter = TimeSeriesSplitter(config)
    splits = splitter.generate_splits(X["_season"])

    # Use the last fold for calibration
    train_seasons, test_seasons = splits[-1]
    train_mask = X["_season"].isin(train_seasons)
    test_mask = X["_season"].isin(test_seasons)

    X_tr = X[train_mask].drop(columns=[c for c in META_COLS if c in X.columns])
    y_tr = y[train_mask].map(LABEL_MAP)
    X_te = X[test_mask].drop(columns=[c for c in META_COLS if c in X.columns])
    y_te = y[test_mask]

    if params is None:
        from ml.config import ModelConfig
        cfg = ModelConfig()
        params = cfg.xgb_params if model_type == "xgboost" else cfg.lgb_params

    model = _build_model(model_type, params)
    cal_seasons = X[train_mask]["_season"] if "_season" in X.columns else None
    sw = _compute_sample_weights(y[train_mask], seasons=cal_seasons) if use_sample_weights else None
    _fit_with_early_stopping(model, X_tr, y_tr, model_type, sample_weight=sw)
    y_proba_raw = model.predict_proba(X_te)

    raw_metrics = compute_metrics(y_te, y_proba_raw)

    calibrator = ProbabilityCalibrator()
    calibrator.fit(y_te, y_proba_raw)

    # Show improvement
    y_proba_cal = calibrator.calibrate(y_proba_raw)
    cal_metrics = compute_metrics(y_te, y_proba_cal)

    log.info(
        "Calibration effect on last fold: log_loss %.4f -> %.4f, brier %.4f -> %.4f",
        raw_metrics["log_loss"], cal_metrics["log_loss"],
        raw_metrics["brier_score"], cal_metrics["brier_score"],
    )

    return calibrator, raw_metrics
