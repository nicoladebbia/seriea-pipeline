"""Model comparison framework — compare two training runs on all metrics."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy import stats

from config.settings import MODELS_DIR

log = logging.getLogger(__name__)


def _load_metadata(variant: str, version: str, model_type: str = "xgboost") -> dict:
    """Load model metadata for a specific version."""
    path = MODELS_DIR / variant / f"{model_type}_{version}_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"No metadata at {path}")
    with open(path) as f:
        return json.load(f)


def _load_cv_results(variant: str) -> Optional[list]:
    """Load ensemble CV results if available."""
    path = MODELS_DIR / variant / "ensemble" / "cv_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_training_report(variant: str) -> Optional[dict]:
    """Load training report if available."""
    path = MODELS_DIR / variant / "training_report.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def compare_runs(variant: str, version_a: str, version_b: str) -> dict:
    """Compare two model versions on all metrics.

    Loads metadata for both versions, computes deltas and flags
    improvement/regression for each metric.

    Returns structured dict with per-metric comparisons.
    """
    meta_a = _load_metadata(variant, version_a)
    meta_b = _load_metadata(variant, version_b)

    metrics_a = meta_a.get("metrics", {})
    metrics_b = meta_b.get("metrics", {})

    # Metrics where lower is better
    lower_better = {"log_loss", "brier_score", "rps", "ece"}

    comparison = {
        "version_a": version_a,
        "version_b": version_b,
        "variant": variant,
        "metrics": {},
    }

    all_keys = sorted(set(list(metrics_a.keys()) + list(metrics_b.keys())))
    for key in all_keys:
        val_a = metrics_a.get(key)
        val_b = metrics_b.get(key)
        if val_a is None or val_b is None or not isinstance(val_a, (int, float)):
            continue

        delta = val_b - val_a
        if key in lower_better:
            improved = delta < 0
        else:
            improved = delta > 0

        comparison["metrics"][key] = {
            "a": val_a,
            "b": val_b,
            "delta": round(delta, 4),
            "improved": improved,
        }

    return comparison


def compare_features(variant: str, version_a: str, version_b: str) -> dict:
    """Compare feature sets and importance rankings between versions.

    Returns dict with added/removed features and Spearman rank correlation.
    """
    meta_a = _load_metadata(variant, version_a)
    meta_b = _load_metadata(variant, version_b)

    feats_a = set(meta_a.get("feature_names", []))
    feats_b = set(meta_b.get("feature_names", []))

    added = sorted(feats_b - feats_a)
    removed = sorted(feats_a - feats_b)
    shared = sorted(feats_a & feats_b)

    # Compute rank correlation on shared features' importance
    imp_a = meta_a.get("feature_importance", {})
    imp_b = meta_b.get("feature_importance", {})

    rank_corr = None
    if imp_a and imp_b and shared:
        vals_a = [imp_a.get(f, 0) for f in shared]
        vals_b = [imp_b.get(f, 0) for f in shared]
        if len(shared) >= 3:
            corr, pval = stats.spearmanr(vals_a, vals_b)
            rank_corr = {"spearman_r": round(float(corr), 4), "p_value": round(float(pval), 4)}

    return {
        "version_a": version_a,
        "version_b": version_b,
        "n_features_a": len(feats_a),
        "n_features_b": len(feats_b),
        "added": added,
        "removed": removed,
        "n_shared": len(shared),
        "rank_correlation": rank_corr,
    }


def print_comparison_report(comparison: dict) -> str:
    """Format a compare_runs() result as a readable table."""
    lines = [
        f"=== Model Comparison: {comparison['version_a']} vs {comparison['version_b']} ===",
        f"Variant: {comparison['variant']}",
        "",
        f"{'Metric':<20} {'Version A':>10} {'Version B':>10} {'Delta':>10} {'Status':>10}",
        "-" * 64,
    ]

    for key, vals in comparison["metrics"].items():
        status = "BETTER" if vals["improved"] else "WORSE"
        delta_str = f"{vals['delta']:+.4f}"
        lines.append(
            f"{key:<20} {vals['a']:>10.4f} {vals['b']:>10.4f} {delta_str:>10} {status:>10}"
        )

    return "\n".join(lines)
