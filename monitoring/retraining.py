#!/usr/bin/env python3
"""Automated Model Retraining System - Phase 6.

Provides:
- Walk-forward validation for realistic backtesting
- Scheduled model retraining
- Accuracy threshold monitoring
- Automatic model versioning
"""

import json
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class WalkForwardValidator:
    """Walk-forward validation for time-series prediction models.

    Simulates real-world prediction by always training on past data
    and testing on future data, avoiding look-ahead bias.
    """

    def __init__(
        self,
        initial_train_size: int = 500,
        test_window: int = 20,
        step_size: int = 20,
        min_train_size: int = 200,
    ):
        """
        Args:
            initial_train_size: Minimum matches for first training
            test_window: Number of matches to test per fold
            step_size: How many matches to advance between folds
            min_train_size: Minimum training set size
        """
        self.initial_train_size = initial_train_size
        self.test_window = test_window
        self.step_size = step_size
        self.min_train_size = min_train_size
        self.results: List[Dict] = []

    def generate_splits(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate train/test splits for walk-forward validation.

        Args:
            df: DataFrame sorted by date with 'match_date' column

        Returns:
            List of (train_df, test_df) tuples
        """
        df = df.sort_values("match_date").reset_index(drop=True)
        n = len(df)
        splits = []

        start_idx = self.initial_train_size

        while start_idx + self.test_window <= n:
            train_df = df.iloc[:start_idx].copy()
            test_df = df.iloc[start_idx:start_idx + self.test_window].copy()

            if len(train_df) >= self.min_train_size:
                splits.append((train_df, test_df))

            start_idx += self.step_size

        log.info(f"Generated {len(splits)} walk-forward splits")
        return splits

    def validate(
        self,
        df: pd.DataFrame,
        train_fn,
        predict_fn,
        target_col: str = "result",
    ) -> Dict:
        """Run walk-forward validation.

        Args:
            df: Full dataset with features and target
            train_fn: Function(train_df) -> model
            predict_fn: Function(model, test_df) -> predictions
            target_col: Name of target column

        Returns:
            Dict with validation results
        """
        splits = self.generate_splits(df)
        self.results = []

        for i, (train_df, test_df) in enumerate(splits):
            fold_start = datetime.now()

            # Train model
            model = train_fn(train_df)

            # Predict
            predictions = predict_fn(model, test_df)

            # Evaluate
            if target_col in test_df.columns:
                actuals = test_df[target_col].values

                # Handle both class predictions and probabilities
                if isinstance(predictions, np.ndarray) and predictions.ndim == 2:
                    pred_classes = predictions.argmax(axis=1)
                else:
                    pred_classes = predictions

                accuracy = (pred_classes == actuals).mean()

                # Confidence-weighted accuracy
                if isinstance(predictions, np.ndarray) and predictions.ndim == 2:
                    max_probs = predictions.max(axis=1)
                    high_conf_mask = max_probs > 0.5
                    if high_conf_mask.sum() > 0:
                        high_conf_acc = (pred_classes[high_conf_mask] == actuals[high_conf_mask]).mean()
                    else:
                        high_conf_acc = 0.0
                else:
                    high_conf_acc = accuracy
            else:
                accuracy = None
                high_conf_acc = None

            fold_time = (datetime.now() - fold_start).total_seconds()

            fold_result = {
                "fold": i,
                "train_size": len(train_df),
                "test_size": len(test_df),
                "train_end_date": train_df["match_date"].max().isoformat() if hasattr(train_df["match_date"].max(), 'isoformat') else str(train_df["match_date"].max()),
                "test_start_date": test_df["match_date"].min().isoformat() if hasattr(test_df["match_date"].min(), 'isoformat') else str(test_df["match_date"].min()),
                "accuracy": accuracy,
                "high_conf_accuracy": high_conf_acc,
                "duration_seconds": fold_time,
            }

            self.results.append(fold_result)
            log.info(f"Fold {i}: Accuracy={accuracy:.1%}" if accuracy else f"Fold {i}: Complete")

        return self.get_summary()

    def get_summary(self) -> Dict:
        """Get summary statistics from validation results."""
        if not self.results:
            return {}

        accuracies = [r["accuracy"] for r in self.results if r["accuracy"] is not None]
        high_conf_accs = [r["high_conf_accuracy"] for r in self.results if r["high_conf_accuracy"] is not None]

        return {
            "num_folds": len(self.results),
            "total_test_samples": sum(r["test_size"] for r in self.results),
            "mean_accuracy": np.mean(accuracies) if accuracies else None,
            "std_accuracy": np.std(accuracies) if accuracies else None,
            "min_accuracy": np.min(accuracies) if accuracies else None,
            "max_accuracy": np.max(accuracies) if accuracies else None,
            "mean_high_conf_accuracy": np.mean(high_conf_accs) if high_conf_accs else None,
            "recent_5_fold_accuracy": np.mean(accuracies[-5:]) if len(accuracies) >= 5 else None,
            "fold_results": self.results,
        }


class ModelRetrainer:
    """Automated model retraining system.

    Monitors model performance and triggers retraining when:
    - Accuracy drops below threshold
    - New data exceeds threshold
    - Scheduled retraining interval reached
    """

    def __init__(
        self,
        accuracy_threshold: float = 0.45,
        new_data_threshold: int = 50,
        retrain_interval_days: int = 7,
        model_dir: Optional[Path] = None,
    ):
        """
        Args:
            accuracy_threshold: Minimum accuracy before triggering retrain
            new_data_threshold: Number of new matches before retrain
            retrain_interval_days: Maximum days between retrains
            model_dir: Directory to store model versions
        """
        self.accuracy_threshold = accuracy_threshold
        self.new_data_threshold = new_data_threshold
        self.retrain_interval_days = retrain_interval_days
        self.model_dir = model_dir or MODELS_DIR / "versions"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.state_file = self.model_dir / "retrainer_state.json"
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """Load retrainer state from file."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {
            "last_retrain": None,
            "last_data_hash": None,
            "matches_since_retrain": 0,
            "current_model_version": "v0.0.0",
            "accuracy_history": [],
        }

    def _save_state(self):
        """Save retrainer state to file."""
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2, default=str)

    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute hash of dataframe for change detection."""
        return hashlib.md5(
            pd.util.hash_pandas_object(df).values.tobytes()
        ).hexdigest()[:16]

    def _increment_version(self, version: str) -> str:
        """Increment model version string."""
        try:
            parts = version.lstrip("v").split(".")
            parts[-1] = str(int(parts[-1]) + 1)
            return "v" + ".".join(parts)
        except:
            return "v1.0.0"

    def should_retrain(
        self,
        current_accuracy: Optional[float] = None,
        new_matches: int = 0,
    ) -> Tuple[bool, str]:
        """Check if model should be retrained.

        Returns:
            Tuple of (should_retrain, reason)
        """
        reasons = []

        # Check accuracy threshold
        if current_accuracy is not None and current_accuracy < self.accuracy_threshold:
            reasons.append(f"accuracy_below_threshold ({current_accuracy:.1%} < {self.accuracy_threshold:.1%})")

        # Check new data threshold
        total_new = self.state.get("matches_since_retrain", 0) + new_matches
        if total_new >= self.new_data_threshold:
            reasons.append(f"new_data_threshold ({total_new} >= {self.new_data_threshold})")

        # Check time interval
        last_retrain = self.state.get("last_retrain")
        if last_retrain:
            last_dt = datetime.fromisoformat(last_retrain)
            days_since = (datetime.now() - last_dt).days
            if days_since >= self.retrain_interval_days:
                reasons.append(f"interval_exceeded ({days_since} >= {self.retrain_interval_days} days)")
        else:
            reasons.append("never_trained")

        if reasons:
            return True, "; ".join(reasons)
        return False, "no_retrain_needed"

    def retrain(
        self,
        df: pd.DataFrame,
        train_fn,
        validate_fn=None,
    ) -> Dict:
        """Execute model retraining.

        Args:
            df: Training data
            train_fn: Function(df) -> model
            validate_fn: Optional function(model, df) -> accuracy

        Returns:
            Dict with retraining results
        """
        start_time = datetime.now()
        new_version = self._increment_version(self.state["current_model_version"])

        log.info(f"Starting retrain: {self.state['current_model_version']} -> {new_version}")

        # Train model
        model = train_fn(df)

        # Validate if function provided
        accuracy = None
        if validate_fn:
            accuracy = validate_fn(model, df)
            self.state["accuracy_history"].append({
                "version": new_version,
                "accuracy": accuracy,
                "timestamp": datetime.now().isoformat(),
            })

        # Update state
        self.state["last_retrain"] = datetime.now().isoformat()
        self.state["last_data_hash"] = self._compute_data_hash(df)
        self.state["matches_since_retrain"] = 0
        self.state["current_model_version"] = new_version
        self._save_state()

        duration = (datetime.now() - start_time).total_seconds()

        result = {
            "version": new_version,
            "training_samples": len(df),
            "accuracy": accuracy,
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat(),
        }

        log.info(f"Retrain complete: {new_version}, accuracy={accuracy:.1%}" if accuracy else f"Retrain complete: {new_version}")

        return result

    def record_new_matches(self, count: int):
        """Record new matches added since last retrain."""
        self.state["matches_since_retrain"] = self.state.get("matches_since_retrain", 0) + count
        self._save_state()

    def get_status(self) -> Dict:
        """Get current retrainer status."""
        return {
            "current_version": self.state["current_model_version"],
            "last_retrain": self.state.get("last_retrain"),
            "matches_since_retrain": self.state.get("matches_since_retrain", 0),
            "accuracy_history_length": len(self.state.get("accuracy_history", [])),
            "recent_accuracy": self.state["accuracy_history"][-1]["accuracy"] if self.state.get("accuracy_history") else None,
        }


class RetrainingScheduler:
    """Scheduler for automated weekly retraining."""

    def __init__(
        self,
        retrainer: ModelRetrainer,
        validator: WalkForwardValidator,
    ):
        self.retrainer = retrainer
        self.validator = validator
        self.schedule_file = retrainer.model_dir / "schedule.json"

    def run_scheduled_check(
        self,
        df: pd.DataFrame,
        train_fn,
        predict_fn,
        force: bool = False,
    ) -> Dict:
        """Run scheduled retraining check.

        Args:
            df: Full dataset
            train_fn: Training function
            predict_fn: Prediction function
            force: Force retrain regardless of conditions

        Returns:
            Dict with check results
        """
        # Check recent accuracy via walk-forward on last 100 matches
        recent_df = df.tail(200)  # Train on 100, test on last 100

        if len(recent_df) >= 120:
            mini_validator = WalkForwardValidator(
                initial_train_size=100,
                test_window=20,
                step_size=20,
            )
            val_results = mini_validator.validate(recent_df, train_fn, predict_fn)
            current_accuracy = val_results.get("mean_accuracy")
        else:
            current_accuracy = None

        # Check if retrain needed
        should_retrain, reason = self.retrainer.should_retrain(
            current_accuracy=current_accuracy,
            new_matches=0,
        )

        result = {
            "timestamp": datetime.now().isoformat(),
            "current_accuracy": current_accuracy,
            "should_retrain": should_retrain or force,
            "reason": reason if not force else "forced",
        }

        if should_retrain or force:
            # Full walk-forward validation before retrain
            full_results = self.validator.validate(df, train_fn, predict_fn)

            # Retrain
            def validate_wrapper(model, df):
                return full_results.get("mean_accuracy", 0.5)

            retrain_result = self.retrainer.retrain(
                df,
                train_fn,
                validate_fn=validate_wrapper,
            )
            result["retrain_result"] = retrain_result
            result["validation_results"] = full_results

        return result


def create_default_retrainer() -> ModelRetrainer:
    """Create retrainer with default settings for Serie A predictions."""
    return ModelRetrainer(
        accuracy_threshold=0.42,  # Below 42% triggers retrain
        new_data_threshold=38,    # ~1 matchweek of data
        retrain_interval_days=7,  # Weekly retraining
    )


def create_default_validator() -> WalkForwardValidator:
    """Create walk-forward validator with default settings."""
    return WalkForwardValidator(
        initial_train_size=500,   # ~1.5 seasons initial
        test_window=38,           # 1 matchweek
        step_size=38,             # Move 1 matchweek
    )
