"""Probability calibration: isotonic regression + temperature scaling.

Raw GBM probabilities are often overconfident. This module provides:
  - Isotonic regression (per-class, non-parametric)
  - Temperature scaling (single parameter, more robust with small samples)
  - Auto-selection: pick whichever method produces lower ECE on validation
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

from config.settings import MODELS_DIR
from ml.config import (
    CLASS_INDICES,
    LABEL_MAP,
    META_COLS,
    N_CLASSES,
    CalibrationConfig,
    ValidationConfig,
    drop_meta,
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
            # Validation passed — refit on ALL OOF data for better isotonic fit.
            # This is valid: OOF predictions are out-of-fold by construction,
            # so refitting isotonic on all of them doesn't leak information.
            # The validation check above confirmed calibration helps.
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


class TemperatureScaler:
    """Single-parameter temperature scaling for multiclass calibration.

    Divides logits by temperature T before softmax. T < 1.0 sharpens
    predictions (increases confidence), T > 1.0 smooths them.

    More robust than isotonic with small calibration sets (~380 per class)
    because it has only 1 parameter vs isotonic's non-parametric fit.
    """

    def __init__(self, config: CalibrationConfig | None = None):
        self.config = config or CalibrationConfig()
        self.temperature: float = self.config.temperature
        self.fitted = False

    def fit(
        self,
        y_true: pd.Series,
        y_proba: np.ndarray,
    ) -> "TemperatureScaler":
        """Optimize temperature on validation data to minimize log loss.

        y_true: Series with values 'H', 'D', 'A'
        y_proba: (n, N_CLASSES) array of raw probabilities
        """
        y_int = y_true.map(LABEL_MAP).values
        n_samples = len(y_int)

        if n_samples < 50:
            log.warning("Too few samples (%d) for temperature scaling", n_samples)
            self.temperature = 1.0
            self.fitted = False
            return self

        def neg_log_loss(T: float) -> float:
            scaled = self._apply_temperature(y_proba, T)
            # Clip for numerical stability
            scaled = np.clip(scaled, 1e-7, 1 - 1e-7)
            # Compute log loss
            log_probs = np.log(scaled[np.arange(n_samples), y_int])
            return -np.mean(log_probs)

        result = minimize_scalar(neg_log_loss, bounds=(0.3, 3.0), method="bounded")
        self.temperature = round(float(result.x), 4)
        self.fitted = True

        log.info(
            "Temperature scaling: optimal T=%.4f (log_loss %.4f -> %.4f)",
            self.temperature, neg_log_loss(1.0), result.fun,
        )
        return self

    def calibrate(self, y_proba: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to raw probabilities."""
        if not self.fitted:
            return y_proba
        return self._apply_temperature(y_proba, self.temperature)

    def _apply_temperature(self, y_proba: np.ndarray, T: float) -> np.ndarray:
        """Scale logits by temperature and apply softmax."""
        # Convert probabilities to logits
        proba_clipped = np.clip(y_proba, 1e-7, 1 - 1e-7)
        logits = np.log(proba_clipped)

        # Scale by temperature
        scaled_logits = logits / T

        # Softmax
        exp_logits = np.exp(scaled_logits - scaled_logits.max(axis=1, keepdims=True))
        calibrated = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        return calibrated


class PerClassTemperatureScaler:
    """Per-class temperature scaling to fix class-specific miscalibration.

    Different classes can have different calibration issues:
    - Draws are systematically overestimated (+4-8pp)
    - Home predictions are underconfident at high probabilities

    Per-class T allows fixing each independently.
    """

    def __init__(self, config: CalibrationConfig | None = None):
        self.config = config or CalibrationConfig()
        self.temperatures: list[float] = [1.0] * N_CLASSES  # H, D, A
        self.fitted = False

    def fit(
        self,
        y_true: pd.Series,
        y_proba: np.ndarray,
    ) -> "PerClassTemperatureScaler":
        """Optimize per-class temperatures to minimize log loss."""
        from scipy.optimize import minimize

        y_int = y_true.map(LABEL_MAP).values
        n = len(y_int)

        if n < 100:
            self.fitted = False
            return self

        def neg_log_loss(temps: np.ndarray) -> float:
            scaled = self._apply(y_proba, temps)
            scaled = np.clip(scaled, 1e-7, 1 - 1e-7)
            log_probs = np.log(scaled[np.arange(n), y_int])
            return -np.mean(log_probs)

        result = minimize(neg_log_loss, x0=[1.0, 1.0, 1.0],
                          bounds=[(0.3, 3.0)] * N_CLASSES, method="L-BFGS-B")
        self.temperatures = [round(float(t), 4) for t in result.x]
        self.fitted = True

        log.info(
            "Per-class temperature scaling: T_H=%.3f, T_D=%.3f, T_A=%.3f (loss %.4f -> %.4f)",
            *self.temperatures, neg_log_loss([1.0, 1.0, 1.0]), result.fun,
        )
        return self

    def calibrate(self, y_proba: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return y_proba
        return self._apply(y_proba, self.temperatures)

    def _apply(self, y_proba: np.ndarray, temps: list | np.ndarray) -> np.ndarray:
        proba_clipped = np.clip(y_proba, 1e-7, 1 - 1e-7)
        logits = np.log(proba_clipped)
        # Scale each class by its own temperature
        for i in range(N_CLASSES):
            logits[:, i] /= temps[i]
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)


class AutoCalibrator:
    """Auto-select the best calibration method on validation data.

    Compares isotonic, temperature scaling, and per-class temperature scaling.
    Picks whichever produces lowest ECE on a held-out chronological split.
    """

    def __init__(self, config: CalibrationConfig | None = None):
        self.config = config or CalibrationConfig()
        self._isotonic = ProbabilityCalibrator(self.config)
        self._temperature = TemperatureScaler(self.config)
        self._per_class_temp = PerClassTemperatureScaler(self.config)
        self._chosen = None
        self.method_used: str = "none"
        self.fitted = False

    def fit(
        self,
        y_true: pd.Series,
        y_proba: np.ndarray,
        validation_fraction: float = 0.3,
    ) -> "AutoCalibrator":
        """Fit all methods and pick the best one based on validation ECE."""
        from ml.evaluation import expected_calibration_error

        n = len(y_true)
        split_idx = int(n * (1.0 - validation_fraction))

        if split_idx < 100 or (n - split_idx) < 50:
            self._temperature.fit(y_true, y_proba)
            self._chosen = self._temperature
            self.method_used = "temperature"
            self.fitted = True
            return self

        y_true_cal, y_true_val = y_true.iloc[:split_idx], y_true.iloc[split_idx:]
        y_proba_cal, y_proba_val = y_proba[:split_idx], y_proba[split_idx:]
        y_val_int = y_true_val.map(LABEL_MAP).values

        ece_raw = expected_calibration_error(y_val_int, y_proba_val)

        # Fit all candidates on calibration split
        candidates = {}

        # 1. Isotonic
        iso = ProbabilityCalibrator(self.config)
        iso.fit(y_true_cal, y_proba_cal, validate=False)
        candidates["isotonic"] = (iso, expected_calibration_error(y_val_int, iso.calibrate(y_proba_val)))

        # 2. Temperature scaling
        temp = TemperatureScaler(self.config)
        temp.fit(y_true_cal, y_proba_cal)
        candidates["temperature"] = (temp, expected_calibration_error(y_val_int, temp.calibrate(y_proba_val)))

        # 3. Per-class temperature scaling (fixes draw overestimation)
        pct = PerClassTemperatureScaler(self.config)
        pct.fit(y_true_cal, y_proba_cal)
        if pct.fitted:
            candidates["per_class_temp"] = (pct, expected_calibration_error(y_val_int, pct.calibrate(y_proba_val)))

        # Log comparison
        log.info("Calibration comparison — ECE: raw=%.4f %s",
                 ece_raw,
                 " | ".join(f"{name}={ece:.4f}" for name, (_, ece) in candidates.items()))

        # Pick best method that improves over raw
        best_name, (best_cal, best_ece) = min(candidates.items(), key=lambda x: x[1][1])

        if best_ece < ece_raw * 1.05:
            # Winner — refit on ALL data
            if best_name == "isotonic":
                self._isotonic.fit(y_true, y_proba, validate=False)
                self._chosen = self._isotonic
            elif best_name == "temperature":
                self._temperature.fit(y_true, y_proba)
                self._chosen = self._temperature
            elif best_name == "per_class_temp":
                self._per_class_temp.fit(y_true, y_proba)
                self._chosen = self._per_class_temp
            self.method_used = best_name
            self.fitted = True
            log.info("Auto-calibration selected: %s (ECE %.4f → %.4f)", best_name, ece_raw, best_ece)
        else:
            log.warning("All calibration methods worsened ECE — identity passthrough")
            self._chosen = None
            self.method_used = "none"
            self.fitted = False

        return self

    def calibrate(self, y_proba: np.ndarray) -> np.ndarray:
        """Apply the chosen calibration method."""
        if self._chosen is None:
            return y_proba
        return self._chosen.calibrate(y_proba)

    @property
    def calibrators(self) -> list:
        """Compatibility: return isotonic calibrators if that's the chosen method."""
        if isinstance(self._chosen, ProbabilityCalibrator):
            return self._chosen.calibrators
        return []

    def save(self, variant: str, model_type: str) -> Path:
        """Save the chosen calibrator to disk."""
        save_dir = MODELS_DIR / variant
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"{model_type}_calibrator.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "method": self.method_used,
                "isotonic": self._isotonic.calibrators if self._isotonic.fitted else [],
                "temperature": self._temperature.temperature if self._temperature.fitted else 1.0,
                "per_class_temps": self._per_class_temp.temperatures if self._per_class_temp.fitted else [1.0] * N_CLASSES,
                "config": self.config,
            }, f)
        log.info("Saved auto-calibrator (%s) to %s", self.method_used, path)
        return path

    def load(self, variant: str, model_type: str) -> "AutoCalibrator":
        """Load calibrator from disk."""
        path = MODELS_DIR / variant / f"{model_type}_calibrator.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No calibrator at {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, list):
            # Legacy format: list of isotonic regressors
            self._isotonic.calibrators = data
            self._isotonic.fitted = True
            self._chosen = self._isotonic
            self.method_used = "isotonic"
        elif isinstance(data, dict):
            method = data.get("method", "isotonic")
            if method == "isotonic":
                self._isotonic.calibrators = data.get("isotonic", [])
                self._isotonic.fitted = bool(self._isotonic.calibrators)
                self._chosen = self._isotonic
            elif method == "temperature":
                self._temperature.temperature = data.get("temperature", 1.0)
                self._temperature.fitted = True
                self._chosen = self._temperature
            elif method == "per_class_temp":
                temps = data.get("per_class_temps", [1.0] * N_CLASSES)
                self._per_class_temp.temperatures = temps
                self._per_class_temp.fitted = True
                self._chosen = self._per_class_temp
            self.method_used = method

        self.fitted = self._chosen is not None
        log.info("Loaded auto-calibrator (%s) from %s", self.method_used, path)
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

    X_tr = drop_meta(X[train_mask])
    y_tr = y[train_mask].map(LABEL_MAP)
    X_te = drop_meta(X[test_mask])
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
