"""Evaluation metrics and reporting for match prediction."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    precision_recall_fscore_support,
)

from ml.config import CLASS_INDICES, CLASS_LABELS, LABEL_MAP, N_CLASSES


# ---------------------------------------------------------------------------
# Core new metrics
# ---------------------------------------------------------------------------

def ranked_probability_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Ranked Probability Score — proper ordinal metric for H/D/A.

    RPS penalises predictions that are far from the true outcome in ordinal
    space (Home < Draw < Away).  Lower is better.  Fully vectorized.

    y_true: int array of class indices (0=H, 1=D, 2=A)
    y_proba: (n, K) predicted probabilities
    """
    n = len(y_true)
    n_classes = y_proba.shape[1]
    # Build one-hot matrix and compute cumulative sums along class axis
    y_onehot = np.zeros_like(y_proba)
    y_onehot[np.arange(n), y_true] = 1.0
    cum_pred = np.cumsum(y_proba, axis=1)
    cum_true = np.cumsum(y_onehot, axis=1)
    return float(np.mean(np.mean((cum_pred - cum_true) ** 2, axis=1)))


def expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error — weighted average of per-bin calibration gap.

    Treats all classes together: for each predicted probability bin,
    compares mean predicted confidence to actual accuracy. Lower is better.

    y_true: int array of class indices
    y_proba: (n, K) predicted probabilities
    """
    n = len(y_true)
    y_pred = np.argmax(y_proba, axis=1)
    confidences = np.max(y_proba, axis=1)
    correct = (y_pred == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        count = mask.sum()
        if count == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc = correct[mask].mean()
        ece += (count / n) * abs(avg_acc - avg_conf)
    return float(ece)


def kelly_profit_simulation(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    odds: Optional[np.ndarray] = None,
    kelly_fraction: float = 0.10,  # Synced with production (betting_unified.py)
    market_overround: float = 1.07,
) -> dict:
    """Simulate Kelly criterion betting profit on a test set.

    If *odds* is None, simulates market odds by applying a typical bookmaker
    overround to the base-rate class frequencies in the dataset.  This gives
    a realistic edge estimate: the model profits only when it outperforms
    the market-implied probabilities.

    Returns dict with: total_return, roi, n_bets, win_rate.
    """
    n = len(y_true)
    y_pred = np.argmax(y_proba, axis=1)

    if odds is None:
        # Use fixed historical base rates to avoid leaking test-set class
        # distribution into the evaluation metric. These are the long-run
        # averages across Serie A 2015-2025.
        n_classes = y_proba.shape[1]
        base_probs = np.array([0.44, 0.27, 0.29])[:n_classes]  # H, D, A
        base_probs = np.clip(base_probs, 0.05, 0.90)
        # Apply overround: market_prob = base_prob * overround, odds = 1/market_prob
        market_probs = base_probs * market_overround
        odds = np.tile(1.0 / market_probs, (n, 1))  # same "market line" for all matches

    bankroll = 1.0
    n_bets = 0
    n_wins = 0

    for i in range(n):
        cls = y_pred[i]
        p = y_proba[i, cls]
        o = odds[i, cls]
        edge = p * o - 1.0
        if edge <= 0:
            continue
        # Kelly stake
        stake = kelly_fraction * edge / (o - 1.0) if o > 1.0 else 0.0
        stake = min(stake, 0.025)  # max 2.5% per bet (synced with production)
        n_bets += 1
        if y_true[i] == cls:
            bankroll += stake * (o - 1.0)
            n_wins += 1
        else:
            bankroll -= stake

    return {
        "total_return": round(float(bankroll), 4),
        "roi": round(float(bankroll - 1.0), 4),
        "n_bets": n_bets,
        "win_rate": round(float(n_wins / n_bets), 4) if n_bets > 0 else 0.0,
    }


def calibration_curve_data(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Per-class calibration curve data (for plotting / inspection).

    Returns dict keyed by class label with list of (predicted_avg, actual_avg, count)
    per bin.
    """
    result = {}
    for cls_idx, cls_label in enumerate(CLASS_LABELS):
        binary_true = (y_true == cls_idx).astype(float)
        predicted = y_proba[:, cls_idx]

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bins = []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (predicted > lo) & (predicted <= hi)
            count = int(mask.sum())
            if count == 0:
                bins.append({"predicted_avg": round((lo + hi) / 2, 4),
                             "actual_avg": None, "count": 0})
            else:
                bins.append({
                    "predicted_avg": round(float(predicted[mask].mean()), 4),
                    "actual_avg": round(float(binary_true[mask].mean()), 4),
                    "count": count,
                })
        result[cls_label] = bins
    return result


def compute_metrics_with_ci(
    y_true: pd.Series,
    y_proba: np.ndarray,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> dict:
    """Compute core metrics with bootstrap 95% confidence intervals.

    Returns dict of {metric_name: {"mean": x, "ci_lo": x, "ci_hi": x}}.
    """
    y_int = y_true.map(LABEL_MAP).values
    rng = np.random.RandomState(seed)
    n = len(y_int)

    boot_metrics: Dict[str, list] = {
        "accuracy": [], "log_loss": [], "brier_score": [],
        "rps": [], "ece": [],
    }

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yb = y_int[idx]
        pb = y_proba[idx]
        boot_metrics["accuracy"].append(accuracy_score(yb, np.argmax(pb, axis=1)))
        boot_metrics["log_loss"].append(log_loss(yb, pb, labels=CLASS_INDICES))
        boot_metrics["brier_score"].append(_multiclass_brier(yb, pb))
        boot_metrics["rps"].append(ranked_probability_score(yb, pb))
        boot_metrics["ece"].append(expected_calibration_error(yb, pb))

    result = {}
    for metric, values in boot_metrics.items():
        arr = np.array(values)
        result[metric] = {
            "mean": round(float(arr.mean()), 4),
            "ci_lo": round(float(np.percentile(arr, 2.5)), 4),
            "ci_hi": round(float(np.percentile(arr, 97.5)), 4),
        }
    return result


# ---------------------------------------------------------------------------
# Main compute_metrics (extended with RPS, ECE, Kelly)
# ---------------------------------------------------------------------------

def compute_baseline_metrics(y_true: pd.Series) -> Dict[str, Dict[str, float]]:
    """Compute naive baseline metrics for comparison.

    Returns dict with metrics for:
    - random: uniform random predictions
    - always_home: always predict Home
    - class_frequency: predict proportional to class frequency
    """
    y_int = y_true.map(LABEL_MAP).values
    n = len(y_int)

    # Class frequencies
    counts = np.bincount(y_int, minlength=N_CLASSES)
    freqs = counts / counts.sum()

    baselines = {}

    # 1. Random uniform (1/3 each) — true random accuracy is exactly 1/K
    uniform_proba = np.full((n, N_CLASSES), 1.0 / N_CLASSES)
    baselines["random"] = {
        "accuracy": round(1.0 / N_CLASSES, 4),
        "log_loss": round(float(log_loss(y_int, uniform_proba, labels=CLASS_INDICES)), 4),
        "brier_score": round(float(_multiclass_brier(y_int, uniform_proba)), 4),
        "rps": round(float(ranked_probability_score(y_int, uniform_proba)), 4),
    }

    # 2. Always Home
    always_home_pred = np.zeros(n, dtype=int)
    always_home_proba = np.zeros((n, N_CLASSES))
    always_home_proba[:, 0] = 1.0
    baselines["always_home"] = {
        "accuracy": round(float((y_int == 0).mean()), 4),
        "log_loss": round(float(log_loss(y_int, np.clip(always_home_proba, 1e-15, 1 - 1e-15), labels=CLASS_INDICES)), 4),
        "brier_score": round(float(_multiclass_brier(y_int, always_home_proba)), 4),
        "rps": round(float(ranked_probability_score(y_int, always_home_proba)), 4),
    }

    # 3. Class frequency (predict with historical proportions)
    freq_proba = np.tile(freqs, (n, 1))
    freq_pred = np.argmax(freq_proba, axis=1)
    baselines["class_frequency"] = {
        "accuracy": round(float((y_int == freq_pred).mean()), 4),
        "log_loss": round(float(log_loss(y_int, freq_proba, labels=CLASS_INDICES)), 4),
        "brier_score": round(float(_multiclass_brier(y_int, freq_proba)), 4),
        "rps": round(float(ranked_probability_score(y_int, freq_proba)), 4),
    }

    return baselines


def compute_metrics(y_true: pd.Series, y_proba: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive metrics for H/D/A classification.

    Returns dict with: accuracy, log_loss, brier_score, rps, ece,
    kelly_roi, macro_f1, precision_H/D/A, recall_H/D/A, f1_H/D/A,
    plus baseline comparisons.
    """
    y_int = y_true.map(LABEL_MAP).values
    y_pred = np.argmax(y_proba, axis=1)

    acc = accuracy_score(y_int, y_pred)
    ll = log_loss(y_int, y_proba, labels=CLASS_INDICES)
    brier = _multiclass_brier(y_int, y_proba)
    rps = ranked_probability_score(y_int, y_proba)
    ece = expected_calibration_error(y_int, y_proba)
    kelly = kelly_profit_simulation(y_int, y_proba)

    prec, rec, f1, sup = precision_recall_fscore_support(
        y_int, y_pred, labels=CLASS_INDICES, average=None, zero_division=0,
    )
    # Macro-averaged F1 (equal weight to all classes, penalizes poor draw recall)
    macro_f1 = float(np.mean(f1))

    metrics: Dict[str, float] = {
        "accuracy": round(acc, 4),
        "log_loss": round(ll, 4),
        "brier_score": round(brier, 4),
        "rps": round(rps, 4),
        "ece": round(ece, 4),
        "kelly_roi": kelly["roi"],
        "macro_f1": round(macro_f1, 4),
        "n_samples": len(y_int),
    }
    for idx, cls in enumerate(CLASS_LABELS):
        metrics[f"precision_{cls}"] = round(prec[idx], 4)
        metrics[f"recall_{cls}"] = round(rec[idx], 4)
        metrics[f"f1_{cls}"] = round(f1[idx], 4)
        metrics[f"support_{cls}"] = int(sup[idx])

    # Baseline comparisons
    baselines = compute_baseline_metrics(y_true)
    for baseline_name, baseline_metrics in baselines.items():
        for metric_name, value in baseline_metrics.items():
            metrics[f"baseline_{baseline_name}_{metric_name}"] = value

    return metrics


def _multiclass_brier(y_int: np.ndarray, y_proba: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and one-hot truth."""
    n_classes = y_proba.shape[1]
    y_onehot = np.zeros((len(y_int), n_classes))
    y_onehot[np.arange(len(y_int)), y_int] = 1
    return float(np.mean((y_proba - y_onehot) ** 2))


_CLASS_DISPLAY = {"H": "Home", "D": "Draw", "A": "Away"}


def print_report(metrics: Dict[str, float]) -> str:
    """Format metrics as a readable report string with baseline context."""
    n = metrics.get("n_samples", 0)
    lines = [
        "=== Evaluation Report ===",
        f"Samples:      {n}",
        "",
        "--- Model Performance ---",
        f"Accuracy:     {metrics['accuracy']:.4f}",
        f"Log Loss:     {metrics['log_loss']:.4f}",
        f"Brier Score:  {metrics['brier_score']:.4f}",
        f"RPS:          {metrics.get('rps', 0):.4f}  (lower is better, 0=perfect)",
        f"ECE:          {metrics.get('ece', 0):.4f}  (lower is better, 0=perfectly calibrated)",
        f"Macro F1:     {metrics.get('macro_f1', 0):.4f}  (equal weight to H/D/A)",
        f"Kelly ROI:    {metrics.get('kelly_roi', 0):.4f}",
        "",
    ]

    # Baseline comparison table
    baseline_keys = ["random", "always_home", "class_frequency"]
    baseline_labels = ["Random (1/3)", "Always Home", "Class Freq"]
    has_baselines = any(f"baseline_{b}_accuracy" in metrics for b in baseline_keys)
    if has_baselines:
        lines.append("--- Baseline Comparisons ---")
        lines.append(f"{'Baseline':<16} {'Acc':>6} {'LogLoss':>8} {'Brier':>7} {'RPS':>6}")
        lines.append("-" * 46)
        for bkey, blabel in zip(baseline_keys, baseline_labels):
            ba = metrics.get(f"baseline_{bkey}_accuracy", 0)
            bl = metrics.get(f"baseline_{bkey}_log_loss", 0)
            bb = metrics.get(f"baseline_{bkey}_brier_score", 0)
            br = metrics.get(f"baseline_{bkey}_rps", 0)
            lines.append(f"{blabel:<16} {ba:>6.4f} {bl:>8.4f} {bb:>7.4f} {br:>6.4f}")
        lines.append(f"{'MODEL':<16} {metrics['accuracy']:>6.4f} {metrics['log_loss']:>8.4f} "
                     f"{metrics['brier_score']:>7.4f} {metrics.get('rps', 0):>6.4f}")
        lines.append("")

    # Per-class metrics
    lines.append("--- Per-Class Metrics ---")
    for cls in CLASS_LABELS:
        label = _CLASS_DISPLAY.get(cls, cls)
        lines.append(f"{label} ({cls}):")
        lines.append(f"  Precision: {metrics[f'precision_{cls}']:.4f}")
        lines.append(f"  Recall:    {metrics[f'recall_{cls}']:.4f}")
        lines.append(f"  F1:        {metrics[f'f1_{cls}']:.4f}")
        lines.append(f"  Support:   {metrics[f'support_{cls}']}")
        lines.append("")

    return "\n".join(lines)
