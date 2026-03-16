#!/usr/bin/env python3
"""A/B Testing Framework - Phase 6.

Provides:
- Parallel model comparison
- Statistical significance testing
- Automatic model promotion
- Experiment tracking
"""

import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple
import numpy as np

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class ModelComparator:
    """Compare two models statistically."""

    def __init__(self, significance_level: float = 0.05):
        """
        Args:
            significance_level: P-value threshold for significance
        """
        self.significance_level = significance_level

    def compare_accuracy(
        self,
        model_a_correct: np.ndarray,
        model_b_correct: np.ndarray,
    ) -> Dict:
        """Compare accuracy of two models using McNemar's test.

        Args:
            model_a_correct: Boolean array of correct predictions for model A
            model_b_correct: Boolean array of correct predictions for model B

        Returns:
            Dict with comparison results
        """
        if len(model_a_correct) != len(model_b_correct):
            raise ValueError("Arrays must have same length")

        n = len(model_a_correct)
        acc_a = model_a_correct.mean()
        acc_b = model_b_correct.mean()

        # Contingency table for McNemar's test
        # b = A correct, B wrong
        # c = A wrong, B correct
        b = ((model_a_correct == True) & (model_b_correct == False)).sum()
        c = ((model_a_correct == False) & (model_b_correct == True)).sum()

        # McNemar's test (with continuity correction)
        if b + c > 0:
            chi2 = (abs(b - c) - 1) ** 2 / (b + c)
            if SCIPY_AVAILABLE:
                p_value = 1 - stats.chi2.cdf(chi2, df=1)
            else:
                # Approximate: chi2 > 3.84 is significant at 0.05
                p_value = 0.01 if chi2 > 6.63 else 0.05 if chi2 > 3.84 else 0.5
        else:
            chi2 = 0
            p_value = 1.0

        significant = p_value < self.significance_level
        winner = None
        if significant:
            winner = "A" if acc_a > acc_b else "B"

        return {
            "model_a_accuracy": float(acc_a),
            "model_b_accuracy": float(acc_b),
            "accuracy_diff": float(acc_a - acc_b),
            "n_samples": n,
            "chi2_statistic": float(chi2),
            "p_value": float(p_value),
            "significant": significant,
            "winner": winner,
            "a_only_correct": int(b),
            "b_only_correct": int(c),
        }

    def compare_log_loss(
        self,
        model_a_probs: np.ndarray,
        model_b_probs: np.ndarray,
        actuals: np.ndarray,
    ) -> Dict:
        """Compare models using log loss (cross-entropy).

        Lower log loss is better.
        """
        eps = 1e-10

        # Calculate log loss for each prediction
        losses_a = -np.log(model_a_probs[np.arange(len(actuals)), actuals] + eps)
        losses_b = -np.log(model_b_probs[np.arange(len(actuals)), actuals] + eps)

        mean_loss_a = losses_a.mean()
        mean_loss_b = losses_b.mean()

        # Paired t-test on losses
        if SCIPY_AVAILABLE:
            t_stat, p_value = stats.ttest_rel(losses_a, losses_b)
        else:
            diff = losses_a - losses_b
            t_stat = diff.mean() / (diff.std() / np.sqrt(len(diff)))
            p_value = 0.05 if abs(t_stat) > 2 else 0.5

        significant = p_value < self.significance_level
        winner = None
        if significant:
            winner = "A" if mean_loss_a < mean_loss_b else "B"

        return {
            "model_a_log_loss": float(mean_loss_a),
            "model_b_log_loss": float(mean_loss_b),
            "log_loss_diff": float(mean_loss_a - mean_loss_b),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": significant,
            "winner": winner,
        }

    def compare_brier(
        self,
        model_a_probs: np.ndarray,
        model_b_probs: np.ndarray,
        actuals: np.ndarray,
    ) -> Dict:
        """Compare models using Brier score.

        Lower Brier score is better.
        """
        n_classes = model_a_probs.shape[1]
        one_hot = np.eye(n_classes)[actuals]

        brier_a = ((model_a_probs - one_hot) ** 2).sum(axis=1).mean()
        brier_b = ((model_b_probs - one_hot) ** 2).sum(axis=1).mean()

        # Per-sample Brier for statistical test
        brier_samples_a = ((model_a_probs - one_hot) ** 2).sum(axis=1)
        brier_samples_b = ((model_b_probs - one_hot) ** 2).sum(axis=1)

        if SCIPY_AVAILABLE:
            t_stat, p_value = stats.ttest_rel(brier_samples_a, brier_samples_b)
        else:
            diff = brier_samples_a - brier_samples_b
            t_stat = diff.mean() / (diff.std() / np.sqrt(len(diff)))
            p_value = 0.05 if abs(t_stat) > 2 else 0.5

        significant = p_value < self.significance_level
        winner = None
        if significant:
            winner = "A" if brier_a < brier_b else "B"

        return {
            "model_a_brier": float(brier_a),
            "model_b_brier": float(brier_b),
            "brier_diff": float(brier_a - brier_b),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": significant,
            "winner": winner,
        }


class ABExperiment:
    """Single A/B testing experiment."""

    def __init__(
        self,
        name: str,
        model_a_name: str,
        model_b_name: str,
        min_samples: int = 100,
        max_samples: int = 500,
    ):
        """
        Args:
            name: Experiment name
            model_a_name: Control model name
            model_b_name: Treatment model name
            min_samples: Minimum samples before evaluation
            max_samples: Maximum samples (auto-conclude)
        """
        self.name = name
        self.model_a_name = model_a_name
        self.model_b_name = model_b_name
        self.min_samples = min_samples
        self.max_samples = max_samples

        self.start_time = datetime.now()
        self.predictions_a: List[Dict] = []
        self.predictions_b: List[Dict] = []
        self.status = "running"
        self.conclusion: Optional[Dict] = None

    def record_prediction(
        self,
        model: str,
        probs: np.ndarray,
        actual: int,
        match_id: str = None,
    ):
        """Record a prediction from one of the models.

        Args:
            model: 'A' or 'B'
            probs: Predicted probabilities
            actual: Actual outcome
            match_id: Optional match identifier
        """
        if self.status != "running":
            return

        record = {
            "timestamp": datetime.now().isoformat(),
            "probs": probs.tolist() if isinstance(probs, np.ndarray) else probs,
            "predicted": int(np.argmax(probs)),
            "actual": actual,
            "correct": int(np.argmax(probs)) == actual,
            "match_id": match_id,
        }

        if model.upper() == "A":
            self.predictions_a.append(record)
        else:
            self.predictions_b.append(record)

    def record_paired_prediction(
        self,
        probs_a: np.ndarray,
        probs_b: np.ndarray,
        actual: int,
        match_id: str = None,
    ):
        """Record paired predictions from both models for same match."""
        self.record_prediction("A", probs_a, actual, match_id)
        self.record_prediction("B", probs_b, actual, match_id)

    def evaluate(self) -> Dict:
        """Evaluate experiment status.

        Returns:
            Dict with evaluation results
        """
        n_a = len(self.predictions_a)
        n_b = len(self.predictions_b)
        n_paired = min(n_a, n_b)

        result = {
            "experiment": self.name,
            "status": self.status,
            "model_a": self.model_a_name,
            "model_b": self.model_b_name,
            "samples_a": n_a,
            "samples_b": n_b,
            "samples_paired": n_paired,
        }

        if n_paired < self.min_samples:
            result["message"] = f"Need {self.min_samples - n_paired} more paired samples"
            return result

        # Extract paired data
        correct_a = np.array([p["correct"] for p in self.predictions_a[:n_paired]])
        correct_b = np.array([p["correct"] for p in self.predictions_b[:n_paired]])
        probs_a = np.array([p["probs"] for p in self.predictions_a[:n_paired]])
        probs_b = np.array([p["probs"] for p in self.predictions_b[:n_paired]])
        actuals = np.array([p["actual"] for p in self.predictions_a[:n_paired]])

        # Run comparisons
        comparator = ModelComparator()

        result["accuracy_comparison"] = comparator.compare_accuracy(correct_a, correct_b)
        result["log_loss_comparison"] = comparator.compare_log_loss(probs_a, probs_b, actuals)
        result["brier_comparison"] = comparator.compare_brier(probs_a, probs_b, actuals)

        # Determine overall winner
        votes = {"A": 0, "B": 0}
        for comp_key in ["accuracy_comparison", "log_loss_comparison", "brier_comparison"]:
            winner = result[comp_key].get("winner")
            if winner:
                votes[winner] += 1

        if votes["A"] >= 2:
            result["overall_winner"] = "A"
            result["recommendation"] = f"Keep {self.model_a_name}"
        elif votes["B"] >= 2:
            result["overall_winner"] = "B"
            result["recommendation"] = f"Promote {self.model_b_name}"
        else:
            result["overall_winner"] = None
            result["recommendation"] = "No clear winner, continue testing"

        # Auto-conclude if max samples reached
        if n_paired >= self.max_samples:
            self.conclude(result)

        return result

    def conclude(self, evaluation: Dict = None):
        """Conclude the experiment."""
        if evaluation is None:
            evaluation = self.evaluate()

        self.status = "concluded"
        self.conclusion = {
            "concluded_at": datetime.now().isoformat(),
            "duration_hours": (datetime.now() - self.start_time).total_seconds() / 3600,
            "final_evaluation": evaluation,
        }

        log.info(f"Experiment '{self.name}' concluded: {evaluation.get('recommendation', 'No recommendation')}")

    def to_dict(self) -> Dict:
        """Serialize experiment to dict."""
        return {
            "name": self.name,
            "model_a_name": self.model_a_name,
            "model_b_name": self.model_b_name,
            "start_time": self.start_time.isoformat(),
            "status": self.status,
            "samples_a": len(self.predictions_a),
            "samples_b": len(self.predictions_b),
            "conclusion": self.conclusion,
        }


class ABTestFramework:
    """Framework for managing multiple A/B experiments."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or DATA_DIR / "monitoring" / "ab_tests"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.experiments: Dict[str, ABExperiment] = {}
        self.history_file = self.storage_dir / "experiment_history.json"
        self._load_history()

    def _load_history(self):
        """Load experiment history."""
        if self.history_file.exists():
            with open(self.history_file) as f:
                self.experiment_history = json.load(f)
        else:
            self.experiment_history = []

    def _save_history(self):
        """Save experiment history."""
        with open(self.history_file, "w") as f:
            json.dump(self.experiment_history, f, indent=2)

    def create_experiment(
        self,
        name: str,
        model_a_name: str,
        model_b_name: str,
        min_samples: int = 100,
        max_samples: int = 500,
    ) -> ABExperiment:
        """Create new A/B experiment.

        Args:
            name: Unique experiment name
            model_a_name: Control model identifier
            model_b_name: Treatment model identifier
            min_samples: Minimum samples for evaluation
            max_samples: Maximum samples before auto-conclusion

        Returns:
            ABExperiment instance
        """
        if name in self.experiments:
            log.warning(f"Experiment '{name}' already exists, returning existing")
            return self.experiments[name]

        experiment = ABExperiment(
            name=name,
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            min_samples=min_samples,
            max_samples=max_samples,
        )

        self.experiments[name] = experiment
        log.info(f"Created experiment '{name}': {model_a_name} vs {model_b_name}")

        return experiment

    def get_experiment(self, name: str) -> Optional[ABExperiment]:
        """Get experiment by name."""
        return self.experiments.get(name)

    def record_prediction(
        self,
        experiment_name: str,
        probs_a: np.ndarray,
        probs_b: np.ndarray,
        actual: int,
        match_id: str = None,
    ):
        """Record paired prediction to experiment."""
        if experiment_name not in self.experiments:
            log.warning(f"Experiment '{experiment_name}' not found")
            return

        self.experiments[experiment_name].record_paired_prediction(
            probs_a, probs_b, actual, match_id
        )

    def evaluate_all(self) -> Dict[str, Dict]:
        """Evaluate all running experiments."""
        results = {}
        for name, exp in self.experiments.items():
            if exp.status == "running":
                results[name] = exp.evaluate()
        return results

    def conclude_experiment(self, name: str) -> Optional[Dict]:
        """Manually conclude an experiment."""
        if name not in self.experiments:
            return None

        exp = self.experiments[name]
        exp.conclude()

        # Archive to history
        self.experiment_history.append(exp.to_dict())
        self._save_history()

        return exp.conclusion

    def get_active_experiments(self) -> List[str]:
        """Get list of active experiment names."""
        return [name for name, exp in self.experiments.items() if exp.status == "running"]

    def get_summary(self) -> Dict:
        """Get framework summary."""
        return {
            "active_experiments": len(self.get_active_experiments()),
            "total_experiments": len(self.experiments),
            "concluded_experiments": len(self.experiment_history),
            "experiments": {name: exp.to_dict() for name, exp in self.experiments.items()},
        }


class AutoPromoter:
    """Automatically promote winning models from A/B tests."""

    def __init__(
        self,
        ab_framework: ABTestFramework,
        model_dir: Optional[Path] = None,
    ):
        self.ab_framework = ab_framework
        self.model_dir = model_dir or MODELS_DIR / "production"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.promotion_log_file = self.model_dir / "promotion_log.json"
        self.promotion_log = self._load_log()

    def _load_log(self) -> List[Dict]:
        """Load promotion log."""
        if self.promotion_log_file.exists():
            with open(self.promotion_log_file) as f:
                return json.load(f)
        return []

    def _save_log(self):
        """Save promotion log."""
        with open(self.promotion_log_file, "w") as f:
            json.dump(self.promotion_log, f, indent=2)

    def check_promotions(self) -> List[Dict]:
        """Check all experiments for promotion candidates.

        Returns:
            List of promotion actions taken
        """
        promotions = []

        evaluations = self.ab_framework.evaluate_all()

        for exp_name, eval_result in evaluations.items():
            winner = eval_result.get("overall_winner")
            if winner == "B":  # Treatment won
                promotion = self._promote_model(exp_name, eval_result)
                if promotion:
                    promotions.append(promotion)
                    # Conclude the experiment
                    self.ab_framework.conclude_experiment(exp_name)

        return promotions

    def _promote_model(self, experiment_name: str, evaluation: Dict) -> Optional[Dict]:
        """Promote winning model to production.

        In practice, this would:
        1. Copy model weights to production directory
        2. Update configuration
        3. Log the promotion

        Returns:
            Promotion record if successful
        """
        exp = self.ab_framework.get_experiment(experiment_name)
        if not exp:
            return None

        promotion_record = {
            "timestamp": datetime.now().isoformat(),
            "experiment": experiment_name,
            "promoted_model": exp.model_b_name,
            "replaced_model": exp.model_a_name,
            "accuracy_improvement": evaluation.get("accuracy_comparison", {}).get("accuracy_diff"),
            "samples_tested": evaluation.get("samples_paired"),
        }

        self.promotion_log.append(promotion_record)
        self._save_log()

        log.info(f"PROMOTED: {exp.model_b_name} replaces {exp.model_a_name}")

        return promotion_record

    def get_promotion_history(self) -> List[Dict]:
        """Get history of model promotions."""
        return self.promotion_log
