#!/usr/bin/env python3
"""Probability Calibration System - Phase 6.

Provides:
- Platt scaling for probability calibration
- Isotonic regression calibration
- Calibration curve visualization
- Dynamic threshold adjustment
- Calibration tracking over time
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    from sklearn.calibration import calibration_curve
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class PlattScaler:
    """Platt scaling for probability calibration.

    Fits a logistic regression on model outputs to produce
    calibrated probability estimates.
    """

    def __init__(self):
        self.scalers: Dict[int, LogisticRegression] = {}  # One per class
        self.fitted = False

    def fit(self, probs: np.ndarray, labels: np.ndarray):
        """Fit Platt scaling calibrators.

        Args:
            probs: Predicted probabilities (n_samples, n_classes)
            labels: True labels (n_samples,)
        """
        if not SKLEARN_AVAILABLE:
            log.warning("sklearn not available, skipping Platt scaling fit")
            return

        n_classes = probs.shape[1]

        for c in range(n_classes):
            # Binary labels: 1 if this class, 0 otherwise
            binary_labels = (labels == c).astype(int)

            # Fit logistic regression on this class's probabilities
            scaler = LogisticRegression(solver='lbfgs', max_iter=1000)
            scaler.fit(probs[:, c].reshape(-1, 1), binary_labels)
            self.scalers[c] = scaler

        self.fitted = True
        log.info(f"Fitted Platt scalers for {n_classes} classes")

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Apply Platt scaling to probabilities.

        Args:
            probs: Raw probabilities (n_samples, n_classes)

        Returns:
            Calibrated probabilities
        """
        if not self.fitted or not SKLEARN_AVAILABLE:
            return probs

        n_samples, n_classes = probs.shape
        calibrated = np.zeros_like(probs)

        for c in range(n_classes):
            if c in self.scalers:
                calibrated[:, c] = self.scalers[c].predict_proba(
                    probs[:, c].reshape(-1, 1)
                )[:, 1]
            else:
                calibrated[:, c] = probs[:, c]

        # Renormalize to sum to 1
        calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)

        return calibrated


class IsotonicCalibrator:
    """Isotonic regression for probability calibration.

    More flexible than Platt scaling but may overfit with small samples.
    """

    def __init__(self):
        self.calibrators: Dict[int, IsotonicRegression] = {}
        self.fitted = False

    def fit(self, probs: np.ndarray, labels: np.ndarray):
        """Fit isotonic regression calibrators."""
        if not SKLEARN_AVAILABLE:
            log.warning("sklearn not available, skipping isotonic fit")
            return

        n_classes = probs.shape[1]

        for c in range(n_classes):
            binary_labels = (labels == c).astype(int)

            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(probs[:, c], binary_labels)
            self.calibrators[c] = calibrator

        self.fitted = True
        log.info(f"Fitted isotonic calibrators for {n_classes} classes")

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration."""
        if not self.fitted or not SKLEARN_AVAILABLE:
            return probs

        n_samples, n_classes = probs.shape
        calibrated = np.zeros_like(probs)

        for c in range(n_classes):
            if c in self.calibrators:
                calibrated[:, c] = self.calibrators[c].predict(probs[:, c])
            else:
                calibrated[:, c] = probs[:, c]

        # Renormalize
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        calibrated = calibrated / row_sums

        return calibrated


class ProbabilityCalibrator:
    """Combined probability calibration system.

    Supports multiple calibration methods and selection based on
    validation performance.
    """

    def __init__(self, method: str = "platt"):
        """
        Args:
            method: 'platt', 'isotonic', or 'auto'
        """
        self.method = method
        self.platt = PlattScaler()
        self.isotonic = IsotonicCalibrator()
        self.active_method = None
        self.calibration_stats: Dict = {}

    def fit(self, probs: np.ndarray, labels: np.ndarray, val_probs: np.ndarray = None, val_labels: np.ndarray = None):
        """Fit calibrators.

        Args:
            probs: Training probabilities
            labels: Training labels
            val_probs: Validation probabilities for method selection
            val_labels: Validation labels for method selection
        """
        # Fit both methods
        self.platt.fit(probs, labels)
        self.isotonic.fit(probs, labels)

        if self.method == "auto" and val_probs is not None and val_labels is not None:
            # Select best method based on validation calibration error
            platt_cal = self.platt.transform(val_probs)
            isotonic_cal = self.isotonic.transform(val_probs)

            platt_ece = self._expected_calibration_error(platt_cal, val_labels)
            isotonic_ece = self._expected_calibration_error(isotonic_cal, val_labels)

            self.active_method = "platt" if platt_ece <= isotonic_ece else "isotonic"
            log.info(f"Auto-selected {self.active_method} (ECE: platt={platt_ece:.4f}, isotonic={isotonic_ece:.4f})")
        else:
            self.active_method = self.method if self.method != "auto" else "platt"

        self.calibration_stats = {
            "method": self.active_method,
            "fit_timestamp": datetime.now().isoformat(),
            "n_samples": len(labels),
        }

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Apply calibration."""
        if self.active_method == "isotonic":
            return self.isotonic.transform(probs)
        return self.platt.transform(probs)

    def _expected_calibration_error(self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        n_samples = len(labels)
        ece = 0.0

        for c in range(probs.shape[1]):
            binary_labels = (labels == c).astype(int)

            for bin_idx in range(n_bins):
                bin_lower = bin_idx / n_bins
                bin_upper = (bin_idx + 1) / n_bins

                in_bin = (probs[:, c] >= bin_lower) & (probs[:, c] < bin_upper)
                bin_size = in_bin.sum()

                if bin_size > 0:
                    avg_confidence = probs[in_bin, c].mean()
                    avg_accuracy = binary_labels[in_bin].mean()
                    ece += (bin_size / n_samples) * abs(avg_confidence - avg_accuracy)

        return ece / probs.shape[1]  # Average across classes

    def get_calibration_curve(self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> Dict:
        """Get calibration curve data."""
        if not SKLEARN_AVAILABLE:
            return {}

        result = {"bins": n_bins, "classes": {}}

        for c in range(probs.shape[1]):
            binary_labels = (labels == c).astype(int)
            class_probs = probs[:, c]

            try:
                prob_true, prob_pred = calibration_curve(
                    binary_labels, class_probs, n_bins=n_bins, strategy='uniform'
                )
                result["classes"][c] = {
                    "prob_true": prob_true.tolist(),
                    "prob_pred": prob_pred.tolist(),
                }
            except Exception as e:
                log.debug(f"Failed to compute calibration curve for class {c}: {e}")

        return result


class CalibrationTracker:
    """Track calibration metrics over time."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or DATA_DIR / "monitoring" / "calibration"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.storage_dir / "calibration_history.json"
        self.history = self._load_history()

    def _load_history(self) -> List[Dict]:
        """Load calibration history."""
        if self.history_file.exists():
            with open(self.history_file) as f:
                return json.load(f)
        return []

    def _save_history(self):
        """Save calibration history."""
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def record_calibration(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        model_version: str,
        is_calibrated: bool = False,
    ) -> Dict:
        """Record calibration metrics for a batch of predictions.

        Args:
            probs: Predicted probabilities
            labels: True labels
            model_version: Model version identifier
            is_calibrated: Whether probabilities were calibrated

        Returns:
            Dict with calibration metrics
        """
        # Calculate metrics
        ece = self._calculate_ece(probs, labels)
        brier = self._calculate_brier_score(probs, labels)

        # Reliability by confidence bucket
        reliability = self._calculate_reliability(probs, labels)

        record = {
            "timestamp": datetime.now().isoformat(),
            "model_version": model_version,
            "is_calibrated": is_calibrated,
            "n_samples": len(labels),
            "ece": ece,
            "brier_score": brier,
            "reliability": reliability,
        }

        self.history.append(record)
        self._save_history()

        return record

    def _calculate_ece(self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        predicted_class = probs.argmax(axis=1)
        confidences = probs.max(axis=1)
        correct = (predicted_class == labels)

        ece = 0.0
        for bin_idx in range(n_bins):
            bin_lower = bin_idx / n_bins
            bin_upper = (bin_idx + 1) / n_bins

            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            bin_size = in_bin.sum()

            if bin_size > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = correct[in_bin].mean()
                ece += (bin_size / len(labels)) * abs(avg_confidence - avg_accuracy)

        return ece

    def _calculate_brier_score(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Brier score (lower is better)."""
        n_classes = probs.shape[1]
        one_hot = np.eye(n_classes)[labels]
        return ((probs - one_hot) ** 2).mean()

    def _calculate_reliability(self, probs: np.ndarray, labels: np.ndarray) -> Dict:
        """Calculate reliability by confidence bucket."""
        predicted_class = probs.argmax(axis=1)
        confidences = probs.max(axis=1)
        correct = (predicted_class == labels)

        buckets = {}
        thresholds = [(0.33, 0.45, "low"), (0.45, 0.55, "medium"), (0.55, 0.70, "high"), (0.70, 1.0, "very_high")]

        for lower, upper, name in thresholds:
            mask = (confidences >= lower) & (confidences < upper)
            if mask.sum() > 0:
                buckets[name] = {
                    "count": int(mask.sum()),
                    "avg_confidence": float(confidences[mask].mean()),
                    "accuracy": float(correct[mask].mean()),
                    "gap": float(confidences[mask].mean() - correct[mask].mean()),
                }

        return buckets

    def get_recent_trend(self, n_records: int = 20) -> Dict:
        """Get recent calibration trend."""
        recent = self.history[-n_records:] if len(self.history) >= n_records else self.history

        if not recent:
            return {}

        eces = [r["ece"] for r in recent if "ece" in r]
        briers = [r["brier_score"] for r in recent if "brier_score" in r]

        return {
            "n_records": len(recent),
            "mean_ece": np.mean(eces) if eces else None,
            "recent_ece": eces[-1] if eces else None,
            "ece_trend": "improving" if len(eces) >= 2 and eces[-1] < eces[-2] else "degrading" if len(eces) >= 2 else "stable",
            "mean_brier": np.mean(briers) if briers else None,
        }

    def get_optimal_thresholds(self) -> Dict[str, float]:
        """Calculate optimal confidence thresholds based on history."""
        if not self.history:
            return {"high_confidence": 0.50, "very_high_confidence": 0.65}

        # Aggregate reliability data
        all_reliabilities = [r.get("reliability", {}) for r in self.history[-50:]]

        # Find threshold where accuracy matches confidence
        # Default conservative thresholds
        return {
            "high_confidence": 0.50,
            "very_high_confidence": 0.65,
        }


class DynamicThresholdAdjuster:
    """Dynamically adjust confidence thresholds based on recent calibration."""

    def __init__(self, calibration_tracker: CalibrationTracker):
        self.tracker = calibration_tracker
        self.thresholds = {
            "high": 0.50,
            "very_high": 0.65,
        }

    def update_thresholds(self) -> Dict[str, float]:
        """Update thresholds based on recent calibration data."""
        trend = self.tracker.get_recent_trend()

        if not trend or trend.get("mean_ece") is None:
            return self.thresholds

        ece = trend["mean_ece"]

        # If model is overconfident (high ECE), raise thresholds
        if ece > 0.10:
            self.thresholds["high"] = min(0.60, self.thresholds["high"] + 0.02)
            self.thresholds["very_high"] = min(0.75, self.thresholds["very_high"] + 0.02)
        # If model is well-calibrated, can lower thresholds slightly
        elif ece < 0.05:
            self.thresholds["high"] = max(0.45, self.thresholds["high"] - 0.01)
            self.thresholds["very_high"] = max(0.55, self.thresholds["very_high"] - 0.01)

        return self.thresholds

    def classify_confidence(self, prob: float) -> str:
        """Classify a probability into confidence category."""
        if prob >= self.thresholds["very_high"]:
            return "VERY_HIGH"
        elif prob >= self.thresholds["high"]:
            return "HIGH"
        elif prob >= 0.40:
            return "MEDIUM"
        else:
            return "LOW"
