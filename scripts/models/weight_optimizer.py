#!/usr/bin/env python3
"""WEIGHT OPTIMIZER - Dynamic Ensemble Weight Optimization

Reads feedback analysis, computes optimized weights via inverse-Brier weighting,
factor lift decay/boost multipliers, and calibration curves.

Graduated activation:
- <20 settled: cold_start (analyze only, keep hardcoded weights)
- 20-29 settled: advisory (show proposed weights, don't apply)
- 30+ settled: active (auto-load optimized weights)

Outputs:
- data/feedback/optimized_weights.json
- data/feedback/factor_adjustments.json
- data/feedback/calibration_curve.json

Run standalone: python3 -m scripts.weight_optimizer
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import DATA_DIR
from scripts.utils.json_utils import load_json_safe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FEEDBACK_DIR = DATA_DIR / "feedback"
ANALYSIS_PATH = FEEDBACK_DIR / "analysis.json"
WEIGHTS_PATH = FEEDBACK_DIR / "optimized_weights.json"
FACTOR_ADJ_PATH = FEEDBACK_DIR / "factor_adjustments.json"
CALIBRATION_PATH = FEEDBACK_DIR / "calibration_curve.json"

# Import from ensemble engine to stay in sync (single source of truth)
try:
    from scripts.prediction.ensemble_prediction_engine import ENSEMBLE_WEIGHTS_WITH_DEEP as CURRENT_WEIGHTS
except ImportError:
    CURRENT_WEIGHTS = {
        "factor": 0.27,
        "xg": 0.30,
        "ml": 0.14,
        "player_xg": 0.05,
        "deep": 0.02,
        "market": 0.22,
    }

# Safety rails
MAX_ABSOLUTE_SHIFT = 0.05   # Max 5pp absolute shift per cycle (symmetric for all methods)
MIN_WEIGHT = 0.02           # No method fully zeroed out
BLEND_RATIO = 0.70          # 70% data-driven, 30% current (smoothing)
FACTOR_DECAY_FLOOR = 0.3    # Minimum factor multiplier
FACTOR_BOOST_CAP = 1.2      # Maximum factor multiplier
HISTORY_MAX = 50            # Keep last 50 weight entries


class _NumpySafeEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, (np.bool_, np.generic)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)


def _determine_status(n_settled: int) -> str:
    """Determine activation status based on sample size."""
    if n_settled < 20:
        return "cold_start"
    elif n_settled < 30:
        return "advisory"
    else:
        return "active"


def optimize_weights(analysis: dict) -> dict:
    """Compute optimized weights via inverse-Brier weighting.

    Steps:
    1. Compute raw weights from inverse Brier scores
    2. Blend 70/30 with current weights (smoothing)
    3. Apply max 20% relative shift cap per method
    4. Enforce minimum 2% per method
    5. Normalize to sum to 1.0
    """
    method_brier = analysis.get("method_brier_scores", {})
    n_settled = analysis.get("n_settled", 0)
    status = _determine_status(n_settled)

    if not method_brier:
        return _build_weights_output(CURRENT_WEIGHTS, CURRENT_WEIGHTS, status, n_settled,
                                     "No method Brier scores available")

    # Step 1: Inverse-Brier raw weights (lower Brier = higher weight)
    raw_weights = {}
    for method, data in method_brier.items():
        brier = data.get("mean_brier", 1.0)
        if not isinstance(brier, (int, float)) or brier != brier:  # NaN check
            raw_weights[method] = 1.0  # Neutral weight for invalid Brier
        elif brier <= 0:
            raw_weights[method] = 10.0  # Perfect score
        else:
            raw_weights[method] = 1.0 / brier

    # Normalize raw weights
    total = sum(raw_weights.values())
    if total > 0:
        raw_weights = {k: v / total for k, v in raw_weights.items()}

    # Include only methods that exist in current weights
    data_weights = {}
    for method in CURRENT_WEIGHTS:
        if method in raw_weights:
            data_weights[method] = raw_weights[method]
        else:
            data_weights[method] = CURRENT_WEIGHTS[method]

    # Step 2: Blend with current weights (smoothing)
    blended = {}
    for method in CURRENT_WEIGHTS:
        data_w = data_weights.get(method, CURRENT_WEIGHTS[method])
        curr_w = CURRENT_WEIGHTS[method]
        blended[method] = BLEND_RATIO * data_w + (1 - BLEND_RATIO) * curr_w

    # Step 3: Apply max absolute shift cap (symmetric for all methods)
    capped = {}
    for method in CURRENT_WEIGHTS:
        curr = CURRENT_WEIGHTS[method]
        proposed = blended[method]
        delta = proposed - curr
        if abs(delta) > MAX_ABSOLUTE_SHIFT:
            capped[method] = curr + (MAX_ABSOLUTE_SHIFT if delta > 0 else -MAX_ABSOLUTE_SHIFT)
        else:
            capped[method] = proposed

    # Step 4: Enforce minimum weight
    for method in capped:
        capped[method] = max(capped[method], MIN_WEIGHT)

    # Step 5: Normalize to 1.0
    total = sum(capped.values())
    optimized = {k: round(v / total, 4) for k, v in capped.items()}

    return _build_weights_output(optimized, CURRENT_WEIGHTS, status, n_settled, "ok")


def _build_weights_output(optimized: dict, current: dict, status: str,
                          n_settled: int, reason: str) -> dict:
    """Build the output dict with history tracking."""
    # Load existing to preserve history
    existing = load_json_safe(WEIGHTS_PATH)
    history = existing.get("history", [])

    # Add current entry to history
    history.append({
        "timestamp": datetime.now().isoformat(),
        "n_settled": n_settled,
        "weights": optimized,
        "status": status,
    })
    # Trim to last N entries
    history = history[-HISTORY_MAX:]

    # Compute changes vs current
    changes = {}
    for method in current:
        curr = current.get(method, 0)
        opt = optimized.get(method, 0)
        changes[method] = {
            "current": curr,
            "optimized": opt,
            "delta": round(opt - curr, 4),
            "pct_change": round((opt - curr) / curr * 100, 1) if curr > 0 else 0,
        }

    return {
        "generated_at": datetime.now().isoformat(),
        "status": status,
        "n_settled": n_settled,
        "reason": reason,
        "optimized_weights": optimized,
        "current_weights": current,
        "changes": changes,
        "history": history,
    }


def compute_factor_decay(analysis: dict) -> dict:
    """For each factor with sufficient data, compute decay/boost multiplier.

    Rules:
    - Factor needs 20+ applications for decay consideration
    - If accuracy < (base_rate - 10%), multiply lift by 0.8 (floor 0.3x)
    - If accuracy > (base_rate + 10%), multiply lift by 1.1 (cap 1.2x)
    - Otherwise keep at 1.0
    """
    factor_eff = analysis.get("factor_effectiveness", {})

    # Load existing multipliers to apply incremental changes
    existing = load_json_safe(FACTOR_ADJ_PATH)
    existing_multipliers = existing.get("multipliers", {})

    multipliers = {}
    details = {}

    for factor, data in factor_eff.items():
        applied = data.get("applied", 0)
        accuracy = data.get("accuracy", 0)
        base_rate = data.get("base_rate", 0.444)
        current_mult = existing_multipliers.get(factor, 1.0)

        if applied < 20:
            # Not enough data, keep current multiplier
            multipliers[factor] = current_mult
            details[factor] = {
                "applied": applied,
                "accuracy": accuracy,
                "base_rate": base_rate,
                "multiplier": current_mult,
                "action": "insufficient_data",
            }
            continue

        # Determine action
        if accuracy < base_rate - 0.10:
            new_mult = current_mult * 0.8
            action = "decay"
        elif accuracy > base_rate + 0.10:
            new_mult = current_mult * 1.1
            action = "boost"
        else:
            new_mult = current_mult
            action = "maintain"

        # Apply floor/cap
        new_mult = max(FACTOR_DECAY_FLOOR, min(FACTOR_BOOST_CAP, new_mult))

        multipliers[factor] = round(new_mult, 3)
        details[factor] = {
            "applied": applied,
            "accuracy": accuracy,
            "base_rate": base_rate,
            "multiplier": round(new_mult, 3),
            "previous_multiplier": current_mult,
            "action": action,
        }

    return {
        "generated_at": datetime.now().isoformat(),
        "n_settled": analysis.get("n_settled", 0),
        "multipliers": multipliers,
        "details": details,
    }


def build_calibration_curve(analysis: dict) -> dict:
    """Build calibration mapping from predicted probability to actual win rate.

    Uses isotonic regression (sklearn) if available, otherwise bucket averaging.
    Requires 50+ settled predictions for activation.
    """
    n_settled = analysis.get("n_settled", 0)
    matched = analysis.get("matched_predictions", [])

    if n_settled < 50:
        return {
            "generated_at": datetime.now().isoformat(),
            "status": "insufficient_data",
            "n_settled": n_settled,
            "required": 50,
            "curve": {},
        }

    # Build data points: (predicted_confidence, actual_correct)
    pred_probs = []
    actual_outcomes = []

    for m in matched:
        conf = m.get("confidence", 0)
        correct = 1.0 if m.get("correct", False) else 0.0
        pred_probs.append(conf)
        actual_outcomes.append(correct)

    if not pred_probs:
        return {
            "generated_at": datetime.now().isoformat(),
            "status": "no_data",
            "n_settled": n_settled,
            "curve": {},
        }

    # Try isotonic regression
    curve = {}
    method = "bucket_average"
    try:
        from sklearn.isotonic import IsotonicRegression
        import numpy as np

        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(pred_probs, actual_outcomes)

        # Sample curve at standard points
        sample_points = [round(x * 0.05, 2) for x in range(4, 21)]  # 0.20 to 1.00
        calibrated = ir.predict(sample_points)
        curve = {str(p): round(float(c), 4) for p, c in zip(sample_points, calibrated)}
        method = "isotonic_regression"

    except ImportError:
        # Fallback: bucket averaging
        buckets = {}
        for p, a in zip(pred_probs, actual_outcomes):
            bucket = round(p * 10) / 10  # Round to nearest 0.1
            bucket = max(0.2, min(0.9, bucket))
            if bucket not in buckets:
                buckets[bucket] = {"sum": 0.0, "count": 0}
            buckets[bucket]["sum"] += a
            buckets[bucket]["count"] += 1

        for bucket, data in sorted(buckets.items()):
            if data["count"] >= 3:
                curve[str(round(bucket, 2))] = round(data["sum"] / data["count"], 4)

    # Compute calibration error
    cal_errors = []
    for p, a in zip(pred_probs, actual_outcomes):
        cal_errors.append(abs(p - a))
    mean_cal_error = sum(cal_errors) / len(cal_errors) if cal_errors else 0

    # Assess bias
    mean_pred = sum(pred_probs) / len(pred_probs)
    mean_actual = sum(actual_outcomes) / len(actual_outcomes)
    if mean_pred > mean_actual + 0.05:
        bias = "overconfident"
    elif mean_pred < mean_actual - 0.05:
        bias = "underconfident"
    else:
        bias = "well_calibrated"

    return {
        "generated_at": datetime.now().isoformat(),
        "status": "active",
        "method": method,
        "n_settled": n_settled,
        "curve": curve,
        "mean_calibration_error": round(mean_cal_error, 4),
        "mean_predicted_confidence": round(mean_pred, 4),
        "mean_actual_win_rate": round(mean_actual, 4),
        "bias": bias,
    }


def run_weight_optimization() -> dict:
    """Orchestrate all optimization steps."""
    log.info("Starting weight optimization...")

    analysis = load_json_safe(ANALYSIS_PATH)
    if not analysis:
        log.warning("No feedback analysis found at %s", ANALYSIS_PATH)
        return {"status": "no_analysis"}

    n_settled = analysis.get("n_settled", 0)
    status = _determine_status(n_settled)
    log.info("Settled predictions: %d -> status: %s", n_settled, status)

    # 1. Optimize weights
    weights_result = optimize_weights(analysis)
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    with open(WEIGHTS_PATH, "w") as f:
        json.dump(weights_result, f, indent=2, cls=_NumpySafeEncoder)
    log.info("Optimized weights saved (%s)", weights_result["status"])

    # 2. Factor decay/boost
    factor_result = compute_factor_decay(analysis)
    with open(FACTOR_ADJ_PATH, "w") as f:
        json.dump(factor_result, f, indent=2, cls=_NumpySafeEncoder)
    n_adjusted = sum(1 for d in factor_result.get("details", {}).values()
                     if d.get("action") in ("decay", "boost"))
    log.info("Factor adjustments saved (%d factors adjusted)", n_adjusted)

    # 3. Calibration curve
    cal_result = build_calibration_curve(analysis)
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(cal_result, f, indent=2, cls=_NumpySafeEncoder)
    log.info("Calibration curve saved (%s)", cal_result.get("status", "unknown"))

    return {
        "weights": weights_result,
        "factor_adjustments": factor_result,
        "calibration": cal_result,
    }


if __name__ == "__main__":
    result = run_weight_optimization()
    print(json.dumps(result, indent=2, cls=_NumpySafeEncoder))
