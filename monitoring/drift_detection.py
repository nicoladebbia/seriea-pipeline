#!/usr/bin/env python3
"""Drift Detection System - Phase 6.

Provides:
- Feature distribution drift detection
- Accuracy drift monitoring
- Concept drift alerts
- Statistical significance testing
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from monitoring.utils import load_json_history, save_json_history

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class FeatureDriftDetector:
    """Detect feature distribution drift over time.

    Uses statistical tests to identify when feature distributions
    have significantly changed from the training baseline.
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        min_samples: int = 50,
        storage_dir: Optional[Path] = None,
    ):
        """
        Args:
            significance_level: P-value threshold for drift detection
            min_samples: Minimum samples for statistical testing
            storage_dir: Directory for storing baselines
        """
        self.significance_level = significance_level
        self.min_samples = min_samples
        self.storage_dir = storage_dir or DATA_DIR / "monitoring" / "drift"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.baseline_file = self.storage_dir / "feature_baseline.json"
        self.baseline = self._load_baseline()
        self.drift_history_file = self.storage_dir / "drift_history.json"

    def _load_baseline(self) -> Dict:
        """Load feature baseline from file."""
        if self.baseline_file.exists():
            with open(self.baseline_file) as f:
                return json.load(f)
        return {}

    def _save_baseline(self):
        """Save feature baseline to file."""
        with open(self.baseline_file, "w") as f:
            json.dump(self.baseline, f, indent=2)

    def set_baseline(self, df: pd.DataFrame, feature_cols: List[str] = None):
        """Set baseline feature distributions.

        Args:
            df: Training data
            feature_cols: Columns to monitor (None = all numeric)
        """
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        self.baseline = {
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(df),
            "features": {},
        }

        for col in feature_cols:
            if col in df.columns:
                values = df[col].dropna().values
                if len(values) >= self.min_samples:
                    self.baseline["features"][col] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "median": float(np.median(values)),
                        "q25": float(np.percentile(values, 25)),
                        "q75": float(np.percentile(values, 75)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "n_samples": len(values),
                    }

        self._save_baseline()
        log.info(f"Set baseline for {len(self.baseline['features'])} features")

    def detect_drift(
        self,
        df: pd.DataFrame,
        feature_cols: List[str] = None,
    ) -> Dict:
        """Detect drift in new data compared to baseline.

        Args:
            df: New data to check
            feature_cols: Columns to check (None = all in baseline)

        Returns:
            Dict with drift detection results
        """
        if not self.baseline.get("features"):
            log.warning("No baseline set, cannot detect drift")
            return {"error": "no_baseline"}

        if feature_cols is None:
            feature_cols = list(self.baseline["features"].keys())

        results = {
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(df),
            "features_checked": len(feature_cols),
            "drifted_features": [],
            "drift_details": {},
        }

        for col in feature_cols:
            if col not in df.columns or col not in self.baseline["features"]:
                continue

            values = df[col].dropna().values
            if len(values) < self.min_samples:
                continue

            baseline_stats = self.baseline["features"][col]
            drift_result = self._check_feature_drift(values, baseline_stats, col)

            if drift_result["is_drifted"]:
                results["drifted_features"].append(col)

            results["drift_details"][col] = drift_result

        results["drift_detected"] = len(results["drifted_features"]) > 0
        results["drift_ratio"] = len(results["drifted_features"]) / len(feature_cols) if feature_cols else 0

        # Log drift alert
        if results["drift_detected"]:
            log.warning(f"Drift detected in {len(results['drifted_features'])} features: {results['drifted_features'][:5]}")

        return results

    def _check_feature_drift(
        self,
        values: np.ndarray,
        baseline: Dict,
        feature_name: str,
    ) -> Dict:
        """Check drift for a single feature."""
        result = {
            "feature": feature_name,
            "is_drifted": False,
            "tests": {},
        }

        # 1. Mean shift test (z-test)
        current_mean = np.mean(values)
        baseline_mean = baseline["mean"]
        baseline_std = baseline["std"]

        if baseline_std > 0:
            z_score = (current_mean - baseline_mean) / (baseline_std / np.sqrt(len(values)))
            p_value_z = 2 * (1 - stats.norm.cdf(abs(z_score))) if SCIPY_AVAILABLE else 1.0
            result["tests"]["mean_shift"] = {
                "z_score": float(z_score),
                "p_value": float(p_value_z),
                "current_mean": float(current_mean),
                "baseline_mean": float(baseline_mean),
                "significant": p_value_z < self.significance_level,
            }
            if p_value_z < self.significance_level:
                result["is_drifted"] = True

        # 2. Variance change test (Levene's test approximation)
        current_std = np.std(values)
        variance_ratio = (current_std / baseline_std) if baseline_std > 0 else 1.0
        variance_significant = variance_ratio > 1.5 or variance_ratio < 0.67
        result["tests"]["variance_change"] = {
            "variance_ratio": float(variance_ratio),
            "current_std": float(current_std),
            "baseline_std": float(baseline_std),
            "significant": variance_significant,
        }
        if variance_significant:
            result["is_drifted"] = True

        # 3. Distribution shift (KS test simulation)
        # Using IQR-based detection
        current_iqr = np.percentile(values, 75) - np.percentile(values, 25)
        baseline_iqr = baseline["q75"] - baseline["q25"]

        if baseline_iqr > 0:
            iqr_ratio = current_iqr / baseline_iqr
            iqr_significant = iqr_ratio > 1.5 or iqr_ratio < 0.67
            result["tests"]["iqr_change"] = {
                "iqr_ratio": float(iqr_ratio),
                "significant": iqr_significant,
            }
            if iqr_significant:
                result["is_drifted"] = True

        # 4. Range shift
        current_min = np.min(values)
        current_max = np.max(values)
        range_shift = (
            current_min < baseline["min"] - baseline_std or
            current_max > baseline["max"] + baseline_std
        )
        result["tests"]["range_shift"] = {
            "current_range": [float(current_min), float(current_max)],
            "baseline_range": [baseline["min"], baseline["max"]],
            "significant": range_shift,
        }
        if range_shift:
            result["is_drifted"] = True

        return result

    def get_drift_summary(self) -> Dict:
        """Get summary of recent drift detections."""
        if self.drift_history_file.exists():
            with open(self.drift_history_file) as f:
                history = json.load(f)
        else:
            history = []

        if not history:
            return {"message": "No drift history available"}

        recent = history[-10:]
        drift_counts = [len(r.get("drifted_features", [])) for r in recent]

        return {
            "total_checks": len(history),
            "recent_checks": len(recent),
            "avg_drifted_features": np.mean(drift_counts),
            "max_drifted_features": max(drift_counts),
            "drift_trend": "increasing" if len(drift_counts) >= 2 and drift_counts[-1] > drift_counts[0] else "stable",
        }


class AccuracyMonitor:
    """Monitor prediction accuracy over time.

    Detects accuracy drift and triggers alerts when performance degrades.
    """

    def __init__(
        self,
        warning_threshold: float = 0.05,
        alert_threshold: float = 0.10,
        window_size: int = 50,
        storage_dir: Optional[Path] = None,
    ):
        """
        Args:
            warning_threshold: Accuracy drop for warning (e.g., 0.05 = 5%)
            alert_threshold: Accuracy drop for alert
            window_size: Rolling window for accuracy calculation
            storage_dir: Directory for storing history
        """
        self.warning_threshold = warning_threshold
        self.alert_threshold = alert_threshold
        self.window_size = window_size
        self.storage_dir = storage_dir or DATA_DIR / "monitoring" / "accuracy"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.history_file = self.storage_dir / "accuracy_history.json"
        self.history = load_json_history(self.history_file)

        self.baseline_accuracy: Optional[float] = None
        self.recent_predictions: List[Dict] = []

    def set_baseline(self, accuracy: float):
        """Set baseline accuracy to monitor against."""
        self.baseline_accuracy = accuracy
        log.info(f"Set baseline accuracy: {accuracy:.1%}")

    def record_prediction(
        self,
        predicted: int,
        actual: int,
        confidence: float,
        match_id: str = None,
    ):
        """Record a single prediction result.

        Args:
            predicted: Predicted class
            actual: Actual class
            confidence: Prediction confidence
            match_id: Optional match identifier
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "predicted": predicted,
            "actual": actual,
            "correct": predicted == actual,
            "confidence": confidence,
            "match_id": match_id,
        }

        self.recent_predictions.append(record)
        self.history.append(record)

        # Keep recent predictions bounded
        if len(self.recent_predictions) > self.window_size * 2:
            self.recent_predictions = self.recent_predictions[-self.window_size:]

        # Periodic save
        if len(self.history) % 10 == 0:
            save_json_history(self.history_file, self.history, max_entries=1000)

    def record_batch(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        confidences: np.ndarray = None,
    ):
        """Record a batch of predictions."""
        if confidences is None:
            confidences = np.ones(len(predictions)) * 0.5

        for pred, actual, conf in zip(predictions, actuals, confidences):
            self.record_prediction(int(pred), int(actual), float(conf))

    def check_drift(self) -> Dict:
        """Check for accuracy drift.

        Returns:
            Dict with drift status and details
        """
        if len(self.recent_predictions) < self.window_size:
            return {
                "status": "insufficient_data",
                "samples": len(self.recent_predictions),
                "required": self.window_size,
            }

        recent = self.recent_predictions[-self.window_size:]
        current_accuracy = np.mean([r["correct"] for r in recent])

        result = {
            "timestamp": datetime.now().isoformat(),
            "current_accuracy": float(current_accuracy),
            "baseline_accuracy": self.baseline_accuracy,
            "window_size": len(recent),
            "status": "normal",
            "drift_detected": False,
        }

        if self.baseline_accuracy is not None:
            accuracy_drop = self.baseline_accuracy - current_accuracy

            if accuracy_drop >= self.alert_threshold:
                result["status"] = "alert"
                result["drift_detected"] = True
                result["accuracy_drop"] = float(accuracy_drop)
                log.error(f"ALERT: Accuracy dropped by {accuracy_drop:.1%} (current: {current_accuracy:.1%})")
            elif accuracy_drop >= self.warning_threshold:
                result["status"] = "warning"
                result["drift_detected"] = True
                result["accuracy_drop"] = float(accuracy_drop)
                log.warning(f"WARNING: Accuracy dropped by {accuracy_drop:.1%} (current: {current_accuracy:.1%})")

        # Check by confidence bucket
        result["by_confidence"] = self._accuracy_by_confidence(recent)

        return result

    def _accuracy_by_confidence(self, predictions: List[Dict]) -> Dict:
        """Calculate accuracy by confidence bucket."""
        buckets = {
            "low": {"range": (0, 0.45), "correct": 0, "total": 0},
            "medium": {"range": (0.45, 0.55), "correct": 0, "total": 0},
            "high": {"range": (0.55, 0.70), "correct": 0, "total": 0},
            "very_high": {"range": (0.70, 1.0), "correct": 0, "total": 0},
        }

        for pred in predictions:
            conf = pred["confidence"]
            for bucket_name, bucket in buckets.items():
                if bucket["range"][0] <= conf < bucket["range"][1]:
                    bucket["total"] += 1
                    if pred["correct"]:
                        bucket["correct"] += 1
                    break

        result = {}
        for name, bucket in buckets.items():
            if bucket["total"] > 0:
                result[name] = {
                    "accuracy": bucket["correct"] / bucket["total"],
                    "count": bucket["total"],
                }

        return result

    def get_rolling_accuracy(self, windows: List[int] = None) -> Dict:
        """Get rolling accuracy for different window sizes."""
        if windows is None:
            windows = [10, 25, 50, 100]

        result = {}
        for w in windows:
            if len(self.history) >= w:
                recent = self.history[-w:]
                acc = np.mean([r["correct"] for r in recent])
                result[f"last_{w}"] = float(acc)

        return result

    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report."""
        if not self.history:
            return {"message": "No prediction history available"}

        all_correct = [r["correct"] for r in self.history]

        # Overall stats
        report = {
            "total_predictions": len(self.history),
            "overall_accuracy": np.mean(all_correct),
            "baseline_accuracy": self.baseline_accuracy,
        }

        # Rolling accuracy
        report["rolling_accuracy"] = self.get_rolling_accuracy()

        # By confidence
        report["by_confidence"] = self._accuracy_by_confidence(self.history[-200:])

        # Trend analysis
        if len(self.history) >= 100:
            first_half = self.history[:len(self.history)//2]
            second_half = self.history[len(self.history)//2:]
            first_acc = np.mean([r["correct"] for r in first_half])
            second_acc = np.mean([r["correct"] for r in second_half])
            report["trend"] = {
                "first_half_accuracy": float(first_acc),
                "second_half_accuracy": float(second_acc),
                "direction": "improving" if second_acc > first_acc else "degrading" if second_acc < first_acc else "stable",
            }

        return report


class ConceptDriftDetector:
    """Detect concept drift using prediction error analysis.

    Concept drift occurs when the relationship between features
    and outcomes changes over time.
    """

    def __init__(
        self,
        window_size: int = 100,
        significance_level: float = 0.05,
    ):
        self.window_size = window_size
        self.significance_level = significance_level
        self.error_history: List[float] = []

    def record_error(self, predicted_prob: np.ndarray, actual_class: int):
        """Record prediction error.

        Args:
            predicted_prob: Probability distribution (3,)
            actual_class: Actual outcome (0, 1, 2)
        """
        # Cross-entropy loss
        eps = 1e-10
        error = -np.log(predicted_prob[actual_class] + eps)
        self.error_history.append(error)

        # Keep bounded
        if len(self.error_history) > self.window_size * 3:
            self.error_history = self.error_history[-self.window_size * 2:]

    def detect_drift(self) -> Dict:
        """Detect concept drift using error distribution change."""
        if len(self.error_history) < self.window_size * 2:
            return {"status": "insufficient_data"}

        old_window = self.error_history[-self.window_size * 2:-self.window_size]
        new_window = self.error_history[-self.window_size:]

        old_mean = np.mean(old_window)
        new_mean = np.mean(new_window)

        # Two-sample t-test
        if SCIPY_AVAILABLE:
            t_stat, p_value = stats.ttest_ind(old_window, new_window)
        else:
            # Approximate
            pooled_std = np.sqrt((np.var(old_window) + np.var(new_window)) / 2)
            t_stat = (new_mean - old_mean) / (pooled_std * np.sqrt(2 / self.window_size))
            p_value = 0.5 if abs(t_stat) < 2 else 0.01

        drift_detected = p_value < self.significance_level and new_mean > old_mean

        return {
            "drift_detected": drift_detected,
            "old_mean_error": float(old_mean),
            "new_mean_error": float(new_mean),
            "error_increase": float(new_mean - old_mean),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "status": "alert" if drift_detected else "normal",
        }
