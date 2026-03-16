#!/usr/bin/env python3
"""Run Monitoring System - Phase 6.

Main entry point for the continuous learning and monitoring system.

Usage:
    python scripts/run_monitoring.py --check          # Run all checks
    python scripts/run_monitoring.py --dashboard      # Print dashboard
    python scripts/run_monitoring.py --retrain        # Force retrain if needed
    python scripts/run_monitoring.py --drift          # Check for drift
    python scripts/run_monitoring.py --full           # Full monitoring cycle
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

from config.settings import DATA_DIR, MODELS_DIR
from monitoring.retraining import (
    ModelRetrainer,
    WalkForwardValidator,
    RetrainingScheduler,
    create_default_retrainer,
    create_default_validator,
)
from monitoring.calibration import (
    ProbabilityCalibrator,
    CalibrationTracker,
    DynamicThresholdAdjuster,
)
from monitoring.drift_detection import (
    FeatureDriftDetector,
    AccuracyMonitor,
    ConceptDriftDetector,
)
from monitoring.ab_testing import ABTestFramework, AutoPromoter
from monitoring.dashboard import PerformanceDashboard, MetricsCollector, AlertManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


class MonitoringSystem:
    """Unified monitoring system for Serie A predictions."""

    def __init__(self):
        # Initialize components
        self.retrainer = create_default_retrainer()
        self.validator = create_default_validator()
        self.scheduler = RetrainingScheduler(self.retrainer, self.validator)

        self.calibrator = ProbabilityCalibrator(method="auto")
        self.calibration_tracker = CalibrationTracker()
        self.threshold_adjuster = DynamicThresholdAdjuster(self.calibration_tracker)

        self.drift_detector = FeatureDriftDetector()
        self.accuracy_monitor = AccuracyMonitor()
        self.concept_detector = ConceptDriftDetector()

        self.ab_framework = ABTestFramework()
        self.auto_promoter = AutoPromoter(self.ab_framework)

        self.metrics = MetricsCollector()
        self.dashboard = PerformanceDashboard(self.metrics)
        self.alert_manager = AlertManager()

        # Add alert handlers
        self.alert_manager.add_handler(self._log_alert)

    def _log_alert(self, severity: str, alert_type: str, message: str):
        """Default alert handler - just logs."""
        log.warning(f"ALERT [{severity}] {alert_type}: {message}")

    def load_data(self) -> pd.DataFrame:
        """Load feature data for monitoring."""
        features_path = DATA_DIR / "features" / "features.parquet"
        if features_path.exists():
            df = pd.read_parquet(features_path)
            df = df.sort_values("match_date").reset_index(drop=True)
            return df
        log.error(f"Features not found: {features_path}")
        return None

    def run_drift_check(self, df: pd.DataFrame = None) -> dict:
        """Check for feature and accuracy drift."""
        log.info("=" * 60)
        log.info("DRIFT DETECTION CHECK")
        log.info("=" * 60)

        if df is None:
            df = self.load_data()
            if df is None:
                return {"error": "no_data"}

        results = {}

        # Feature drift
        if not self.drift_detector.baseline.get("features"):
            # Set baseline from first 70% of data
            split_idx = int(len(df) * 0.7)
            baseline_df = df.iloc[:split_idx]
            feature_cols = [c for c in df.columns if c.startswith(("roll5_", "adv_", "h2h_"))]
            self.drift_detector.set_baseline(baseline_df, feature_cols)
            log.info(f"Set drift baseline from {len(baseline_df)} matches")

        # Check recent data for drift
        recent_df = df.tail(200)
        drift_result = self.drift_detector.detect_drift(recent_df)
        results["feature_drift"] = drift_result

        if drift_result.get("drift_detected"):
            self.alert_manager.create_alert(
                severity="medium",
                alert_type="feature_drift",
                message=f"Drift detected in {len(drift_result['drifted_features'])} features",
                metadata={"features": drift_result["drifted_features"][:10]},
            )

        # Accuracy drift
        accuracy_drift = self.accuracy_monitor.check_drift()
        results["accuracy_drift"] = accuracy_drift

        if accuracy_drift.get("status") == "alert":
            self.alert_manager.create_alert(
                severity="high",
                alert_type="accuracy_drift",
                message=f"Accuracy dropped by {accuracy_drift.get('accuracy_drop', 0):.1%}",
            )

        # Concept drift
        concept_drift = self.concept_detector.detect_drift()
        results["concept_drift"] = concept_drift

        log.info(f"Feature drift: {'DETECTED' if drift_result.get('drift_detected') else 'none'}")
        log.info(f"Accuracy drift: {accuracy_drift.get('status', 'unknown')}")
        log.info(f"Concept drift: {concept_drift.get('status', 'unknown')}")

        return results

    def run_calibration_check(self, probs: np.ndarray = None, labels: np.ndarray = None) -> dict:
        """Check prediction calibration."""
        log.info("=" * 60)
        log.info("CALIBRATION CHECK")
        log.info("=" * 60)

        # Use stored history if no data provided
        recent = self.calibration_tracker.get_recent_trend()

        if probs is not None and labels is not None:
            record = self.calibration_tracker.record_calibration(
                probs, labels,
                model_version="v4.0-deep-learning",
                is_calibrated=False,
            )
            log.info(f"ECE: {record['ece']:.4f}, Brier: {record['brier_score']:.4f}")

        # Update dynamic thresholds
        thresholds = self.threshold_adjuster.update_thresholds()
        log.info(f"Dynamic thresholds: high={thresholds['high']:.2f}, very_high={thresholds['very_high']:.2f}")

        return {
            "recent_trend": recent,
            "thresholds": thresholds,
        }

    def run_retrain_check(self, df: pd.DataFrame = None, force: bool = False) -> dict:
        """Check if retraining is needed and execute if so."""
        log.info("=" * 60)
        log.info("RETRAINING CHECK")
        log.info("=" * 60)

        if df is None:
            df = self.load_data()
            if df is None:
                return {"error": "no_data"}

        # Check if retrain needed
        should_retrain, reason = self.retrainer.should_retrain()
        log.info(f"Should retrain: {should_retrain} ({reason})")

        result = {
            "should_retrain": should_retrain,
            "reason": reason,
            "retrained": False,
        }

        if should_retrain or force:
            log.info("Initiating retrain...")

            # Define simple train/predict functions
            # In production, these would use the actual model training
            def train_fn(train_df):
                # Placeholder - actual implementation would train models
                return {"trained": True, "samples": len(train_df)}

            def predict_fn(model, test_df):
                # Placeholder - returns random predictions
                return np.random.rand(len(test_df), 3)

            # Run validation first
            val_results = self.validator.validate(df, train_fn, predict_fn)
            log.info(f"Walk-forward accuracy: {val_results.get('mean_accuracy', 0):.1%}")

            # Execute retrain
            retrain_result = self.retrainer.retrain(
                df,
                train_fn,
                validate_fn=lambda m, d: val_results.get("mean_accuracy", 0.5),
            )

            result["retrained"] = True
            result["retrain_result"] = retrain_result
            result["validation"] = val_results

        return result

    def run_ab_check(self) -> dict:
        """Check A/B experiments and auto-promote winners."""
        log.info("=" * 60)
        log.info("A/B TESTING CHECK")
        log.info("=" * 60)

        # Evaluate all experiments
        evaluations = self.ab_framework.evaluate_all()

        # Check for promotions
        promotions = self.auto_promoter.check_promotions()

        active = self.ab_framework.get_active_experiments()
        log.info(f"Active experiments: {len(active)}")
        log.info(f"Promotions this cycle: {len(promotions)}")

        return {
            "active_experiments": active,
            "evaluations": evaluations,
            "promotions": promotions,
        }

    def run_full_cycle(self, df: pd.DataFrame = None) -> dict:
        """Run complete monitoring cycle."""
        log.info("=" * 70)
        log.info("FULL MONITORING CYCLE")
        log.info("=" * 70)

        if df is None:
            df = self.load_data()

        results = {
            "timestamp": datetime.now().isoformat(),
            "drift_check": self.run_drift_check(df),
            "calibration_check": self.run_calibration_check(),
            "retrain_check": self.run_retrain_check(df),
            "ab_check": self.run_ab_check(),
            "alerts": self.alert_manager.get_active_alerts(),
        }

        # Save results
        results_dir = DATA_DIR / "monitoring" / "reports"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        log.info(f"Results saved to {results_file}")

        return results

    def print_dashboard(self):
        """Print dashboard to console."""
        self.dashboard.print_summary()

    def get_status(self) -> dict:
        """Get overall system status."""
        return {
            "retrainer": self.retrainer.get_status(),
            "alerts": len(self.alert_manager.get_active_alerts()),
            "experiments": len(self.ab_framework.get_active_experiments()),
            "last_check": datetime.now().isoformat(),
        }


def main():
    parser = argparse.ArgumentParser(description="Run monitoring system")
    parser.add_argument("--check", action="store_true", help="Run quick health check")
    parser.add_argument("--dashboard", action="store_true", help="Print dashboard")
    parser.add_argument("--drift", action="store_true", help="Check for drift")
    parser.add_argument("--retrain", action="store_true", help="Check/run retraining")
    parser.add_argument("--force-retrain", action="store_true", help="Force retraining")
    parser.add_argument("--full", action="store_true", help="Run full monitoring cycle")
    parser.add_argument("--status", action="store_true", help="Print system status")
    args = parser.parse_args()

    system = MonitoringSystem()

    if args.dashboard:
        system.print_dashboard()
    elif args.drift:
        results = system.run_drift_check()
        print(json.dumps(results, indent=2, default=str))
    elif args.retrain or args.force_retrain:
        results = system.run_retrain_check(force=args.force_retrain)
        print(json.dumps(results, indent=2, default=str))
    elif args.full:
        results = system.run_full_cycle()
        print(json.dumps(results, indent=2, default=str))
    elif args.status:
        status = system.get_status()
        print(json.dumps(status, indent=2, default=str))
    else:
        # Default: quick check
        print("Monitoring System Status")
        print("=" * 40)
        status = system.get_status()
        print(f"Model version: {status['retrainer']['current_version']}")
        print(f"Last retrain: {status['retrainer']['last_retrain']}")
        print(f"Matches since retrain: {status['retrainer']['matches_since_retrain']}")
        print(f"Active alerts: {status['alerts']}")
        print(f"Active experiments: {status['experiments']}")
        print("\nRun with --help for more options")


if __name__ == "__main__":
    main()
