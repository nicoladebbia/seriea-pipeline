#!/usr/bin/env python3
"""FEEDBACK ANALYZER - Post-Settlement Analysis

Reads predictions_archive.json + results.json, computes per-method Brier scores,
factor effectiveness, intelligence adjustment effectiveness, and xG accuracy.

Outputs: data/feedback/analysis.json

Run standalone: python3 -m scripts.feedback_analyzer
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

ARCHIVE_PATH = DATA_DIR / "upcoming" / "predictions_archive.json"
RESULTS_PATH = DATA_DIR / "upcoming" / "results.json"
FEEDBACK_DIR = DATA_DIR / "feedback"
ANALYSIS_PATH = FEEDBACK_DIR / "analysis.json"


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


def match_predictions_to_results():
    """Join archive entries with settled results by match name.

    Returns list of dicts with prediction + actual outcome.
    """
    archive = load_json_safe(ARCHIVE_PATH)
    results_raw = load_json_safe(RESULTS_PATH)

    if not archive or not results_raw:
        return []

    # Build results lookup (results.json is dict keyed by match name)
    results = results_raw.get("results", results_raw)
    if not isinstance(results, dict):
        return []

    results_map = {}
    for key, val in results.items():
        if isinstance(val, dict) and val.get("completed", False):
            match_name = val.get("match", key)
            results_map[match_name] = val

    matched = []
    for _key, pred in archive.items():
        match_name = pred.get("match", "")
        result = results_map.get(match_name)
        if not result:
            continue

        home_score = result.get("home_score", 0)
        away_score = result.get("away_score", 0)
        if home_score > away_score:
            actual = "HOME"
        elif away_score > home_score:
            actual = "AWAY"
        else:
            actual = "DRAW"

        matched.append({
            "match": match_name,
            "predicted_outcome": pred.get("predicted_outcome", ""),
            "confidence": pred.get("confidence", 0),
            "confidence_level": pred.get("confidence_level", ""),
            "probabilities": pred.get("probabilities", {}),
            "actual_outcome": actual,
            "home_score": home_score,
            "away_score": away_score,
            "total_goals": result.get("total_goals", home_score + away_score),
            "home_xg": pred.get("home_xg", 0),
            "away_xg": pred.get("away_xg", 0),
            "component_predictions": pred.get("component_predictions", {}),
            "methods_used": pred.get("methods_used", []),
            "weights_applied": pred.get("weights_applied", {}),
            "intelligence_adjustments": pred.get("intelligence_adjustments", []),
            "home_factors": pred.get("home_factors", []),
            "away_factors": pred.get("away_factors", []),
            "neutral_factors": pred.get("neutral_factors", []),
            "n_factors": pred.get("n_factors", 0),
            "lineup_source": pred.get("lineup_source", ""),
            "date": pred.get("date", ""),
            "lessons_applied": pred.get("lessons_applied", []),
        })

    return matched


def brier_score(probs: dict, actual: str) -> float:
    """Compute Brier score for a single prediction.

    Lower is better. Perfect = 0.0, worst = 2.0.
    """
    actual_vec = {"HOME": 0.0, "DRAW": 0.0, "AWAY": 0.0}
    actual_vec[actual] = 1.0

    score = 0.0
    for outcome in ["HOME", "DRAW", "AWAY"]:
        pred_key = outcome.lower()
        # Handle both "home"/"draw"/"away" and "prob_H"/"prob_D"/"prob_A" formats
        p = probs.get(pred_key, probs.get(f"prob_{outcome[0]}", 0))
        score += (p - actual_vec[outcome]) ** 2

    return score


def compute_overall_metrics(matched: list) -> dict:
    """Compute mean Brier score, accuracy with CIs overall and by confidence level."""
    if not matched:
        return {"status": "no_data", "n_settled": 0}

    from ml.statistical_validation import (
        accuracy_with_ci, binomial_test, format_accuracy_claim,
        BASE_RATE_HOME_WIN, BASE_RATE_RANDOM, MIN_SAMPLES_REPORT,
    )

    total_brier = 0.0
    correct = 0
    by_confidence = {}

    for m in matched:
        probs = m.get("probabilities", {})
        actual = m["actual_outcome"]
        bs = brier_score(probs, actual)
        total_brier += bs

        is_correct = m["predicted_outcome"].upper() == actual
        if is_correct:
            correct += 1

        level = m.get("confidence_level", "UNKNOWN")
        if level not in by_confidence:
            by_confidence[level] = {"total": 0, "correct": 0, "brier_sum": 0.0}
        by_confidence[level]["total"] += 1
        if is_correct:
            by_confidence[level]["correct"] += 1
        by_confidence[level]["brier_sum"] += bs

    n = len(matched)

    # Overall accuracy with CI and significance
    overall_ci = accuracy_with_ci(correct, n)
    sig_vs_random = binomial_test(correct, n, BASE_RATE_RANDOM)
    sig_vs_home = binomial_test(correct, n, BASE_RATE_HOME_WIN)

    confidence_breakdown = {}
    for level, data in by_confidence.items():
        level_ci = accuracy_with_ci(data["correct"], data["total"])
        entry = {
            "total": data["total"],
            "correct": data["correct"],
            "accuracy": level_ci["accuracy"],
            "ci_lo": level_ci["ci_lo"],
            "ci_hi": level_ci["ci_hi"],
            "reliability": level_ci["reliability"],
            "mean_brier": round(data["brier_sum"] / data["total"], 4) if data["total"] > 0 else 0,
        }
        if data["total"] < MIN_SAMPLES_REPORT:
            entry["warning"] = f"Below minimum sample size ({data['total']}/{MIN_SAMPLES_REPORT})"
        confidence_breakdown[level] = entry

    return {
        "n_settled": n,
        "mean_brier": round(total_brier / n, 4),
        "accuracy": overall_ci["accuracy"],
        "accuracy_ci_lo": overall_ci["ci_lo"],
        "accuracy_ci_hi": overall_ci["ci_hi"],
        "reliability": overall_ci["reliability"],
        "correct": correct,
        "sig_vs_random_p": sig_vs_random["p_value"],
        "sig_vs_always_home_p": sig_vs_home["p_value"],
        "formatted_claim": format_accuracy_claim(
            "Overall", correct, n, BASE_RATE_HOME_WIN,
        ),
        "by_confidence": confidence_breakdown,
    }


def compute_method_brier_scores(matched: list) -> dict:
    """Compute per-method Brier score from component_predictions."""
    method_scores = {}

    for m in matched:
        components = m.get("component_predictions", {})
        actual = m["actual_outcome"]

        for method, probs in components.items():
            # Skip non-prediction entries (e.g. xg_details, player_xg_details)
            if not isinstance(probs, dict):
                continue
            if "prob_H" not in probs and "home" not in probs:
                continue

            if method not in method_scores:
                method_scores[method] = {"brier_sum": 0.0, "count": 0}

            # Normalize key format
            norm_probs = {}
            if "prob_H" in probs:
                norm_probs = {
                    "home": probs["prob_H"],
                    "draw": probs["prob_D"],
                    "away": probs["prob_A"],
                }
            else:
                norm_probs = probs

            bs = brier_score(norm_probs, actual)
            method_scores[method]["brier_sum"] += bs
            method_scores[method]["count"] += 1

    result = {}
    for method, data in method_scores.items():
        if data["count"] > 0:
            result[method] = {
                "mean_brier": round(data["brier_sum"] / data["count"], 4),
                "n_predictions": data["count"],
            }

    # Rank by Brier (lower is better)
    if result:
        ranked = sorted(result.items(), key=lambda x: x[1]["mean_brier"])
        for rank, (method, _) in enumerate(ranked, 1):
            result[method]["rank"] = rank

    return result


def compute_intelligence_effectiveness(matched: list) -> dict:
    """For each adjustment source, compare Brier with vs without the adjustment."""
    sources = {}

    for m in matched:
        adjustments = m.get("intelligence_adjustments", [])
        if not adjustments:
            continue

        actual = m["actual_outcome"]
        probs = m.get("probabilities", {})
        post_brier = brier_score(probs, actual)

        for adj in adjustments:
            source = adj.get("source", "unknown")
            home_shift = adj.get("home_shift", 0)
            draw_shift = adj.get("draw_shift", 0)
            away_shift = adj.get("away_shift", 0)

            # Reconstruct pre-adjustment probabilities
            pre_probs = {
                "home": probs.get("home", 0) - home_shift,
                "draw": probs.get("draw", 0) - draw_shift,
                "away": probs.get("away", 0) - away_shift,
            }
            # Re-normalize
            total = sum(pre_probs.values())
            if total > 0:
                pre_probs = {k: v / total for k, v in pre_probs.items()}

            pre_brier = brier_score(pre_probs, actual)

            if source not in sources:
                sources[source] = {"helped": 0, "hurt": 0, "neutral": 0, "total": 0,
                                   "brier_improvement_sum": 0.0}

            improvement = pre_brier - post_brier  # positive = adjustment helped
            sources[source]["total"] += 1
            sources[source]["brier_improvement_sum"] += improvement

            if improvement > 0.005:
                sources[source]["helped"] += 1
            elif improvement < -0.005:
                sources[source]["hurt"] += 1
            else:
                sources[source]["neutral"] += 1

    result = {}
    for source, data in sources.items():
        n = data["total"]
        mean_improvement = round(data["brier_improvement_sum"] / n, 4) if n > 0 else 0
        helped_pct = round(data["helped"] / n, 4) if n > 0 else 0

        if mean_improvement > 0.005:
            verdict = "beneficial"
        elif mean_improvement < -0.005:
            verdict = "detrimental"
        else:
            verdict = "neutral"

        result[source] = {
            "total_applications": n,
            "helped": data["helped"],
            "hurt": data["hurt"],
            "neutral": data["neutral"],
            "helped_pct": helped_pct,
            "mean_brier_improvement": mean_improvement,
            "verdict": verdict,
        }

    return result


def compute_factor_effectiveness(matched: list) -> dict:
    """Per-factor accuracy with CIs, significance tests, and FDR correction."""
    from ml.statistical_validation import (
        validate_multiple_claims, MIN_SAMPLES_REPORT,
        BASE_RATE_HOME_WIN, BASE_RATE_AWAY_WIN,
    )

    factor_stats = {}

    for m in matched:
        actual = m["actual_outcome"]
        home_factors = m.get("home_factors", [])
        away_factors = m.get("away_factors", [])

        for f in home_factors:
            if f not in factor_stats:
                factor_stats[f] = {"applied": 0, "correct": 0, "type": "home"}
            factor_stats[f]["applied"] += 1
            if actual == "HOME":
                factor_stats[f]["correct"] += 1

        for f in away_factors:
            if f not in factor_stats:
                factor_stats[f] = {"applied": 0, "correct": 0, "type": "away"}
            factor_stats[f]["applied"] += 1
            if actual == "AWAY":
                factor_stats[f]["correct"] += 1

    # Build claims for batch validation with FDR correction
    claims = []
    claim_keys = []
    for factor, data in factor_stats.items():
        if data["applied"] > 0:
            relevant_base = BASE_RATE_HOME_WIN if data["type"] == "home" else BASE_RATE_AWAY_WIN
            claims.append({
                "successes": data["correct"],
                "trials": data["applied"],
                "base_rate": relevant_base,
                "label": factor,
            })
            claim_keys.append(factor)

    # Validate all claims with FDR correction for multiple comparisons
    validated = validate_multiple_claims(claims) if claims else []

    result = {}
    for i, v in enumerate(validated):
        factor = claim_keys[i]
        data = factor_stats[factor]
        entry = {
            "applied": data["applied"],
            "correct": data["correct"],
            "accuracy": v["accuracy"],
            "ci_lo": v["ci_lo"],
            "ci_hi": v["ci_hi"],
            "reliability": v["reliability"],
            "base_rate": v["base_rate"],
            "lift": v["lift"],
            "lift_ci_lo": v["lift_ci_lo"],
            "lift_ci_hi": v["lift_ci_hi"],
            "p_value": v["p_value"],
            "significant": v["significant"],
            "fdr_significant": v["fdr_significant"],
            "type": data["type"],
        }
        if data["applied"] < MIN_SAMPLES_REPORT:
            entry["warning"] = f"Below minimum sample size ({data['applied']}/{MIN_SAMPLES_REPORT})"
        result[factor] = entry

    return result


def compute_xg_accuracy(matched: list) -> dict:
    """Mean xG prediction error, home/away bias detection."""
    if not matched:
        return {"status": "no_data"}

    home_errors = []
    away_errors = []

    for m in matched:
        pred_home_xg = m.get("home_xg", 0)
        pred_away_xg = m.get("away_xg", 0)
        actual_home = m.get("home_score", 0)
        actual_away = m.get("away_score", 0)

        if pred_home_xg > 0:
            home_errors.append(pred_home_xg - actual_home)
        if pred_away_xg > 0:
            away_errors.append(pred_away_xg - actual_away)

    if not home_errors:
        return {"status": "no_xg_data"}

    mean_home_err = sum(home_errors) / len(home_errors)
    mean_away_err = sum(away_errors) / len(away_errors) if away_errors else 0
    mae_home = sum(abs(e) for e in home_errors) / len(home_errors)
    mae_away = sum(abs(e) for e in away_errors) / len(away_errors) if away_errors else 0

    # Bias: positive = overestimate, negative = underestimate
    home_bias = "overestimates" if mean_home_err > 0.15 else ("underestimates" if mean_home_err < -0.15 else "neutral")
    away_bias = "overestimates" if mean_away_err > 0.15 else ("underestimates" if mean_away_err < -0.15 else "neutral")

    return {
        "n_matches": len(home_errors),
        "home_mae": round(mae_home, 3),
        "away_mae": round(mae_away, 3),
        "home_mean_error": round(mean_home_err, 3),
        "away_mean_error": round(mean_away_err, 3),
        "home_bias": home_bias,
        "away_bias": away_bias,
    }


def compute_factor_count_accuracy(matched: list) -> dict:
    """Accuracy breakdown by number of home factors, with CIs."""
    from ml.statistical_validation import accuracy_with_ci, MIN_SAMPLES_REPORT

    by_count = {}

    for m in matched:
        n = m.get("n_factors", 0)
        actual = m["actual_outcome"]
        predicted = m.get("predicted_outcome", "").upper()

        if n not in by_count:
            by_count[n] = {"total": 0, "correct": 0}
        by_count[n]["total"] += 1
        if predicted == actual:
            by_count[n]["correct"] += 1

    result = {}
    for n, data in sorted(by_count.items()):
        ci = accuracy_with_ci(data["correct"], data["total"])
        entry = {
            "total": data["total"],
            "correct": data["correct"],
            "accuracy": ci["accuracy"],
            "ci_lo": ci["ci_lo"],
            "ci_hi": ci["ci_hi"],
            "reliability": ci["reliability"],
        }
        if data["total"] < MIN_SAMPLES_REPORT:
            entry["warning"] = f"Below minimum sample size ({data['total']}/{MIN_SAMPLES_REPORT})"
        result[str(n)] = entry

    return result


def run_feedback_analysis() -> dict:
    """Orchestrate full feedback analysis. Saves to data/feedback/analysis.json."""
    log.info("Starting feedback analysis...")

    matched = match_predictions_to_results()
    log.info("Matched %d predictions to settled results", len(matched))

    overall = compute_overall_metrics(matched)
    method_brier = compute_method_brier_scores(matched)
    intelligence = compute_intelligence_effectiveness(matched)
    factors = compute_factor_effectiveness(matched)
    xg_accuracy = compute_xg_accuracy(matched)
    factor_count = compute_factor_count_accuracy(matched)

    analysis = {
        "generated_at": datetime.now().isoformat(),
        "n_settled": len(matched),
        "overall": overall,
        "method_brier_scores": method_brier,
        "intelligence_effectiveness": intelligence,
        "factor_effectiveness": factors,
        "factor_count_accuracy": factor_count,
        "xg_accuracy": xg_accuracy,
        "matched_predictions": [
            {
                "match": m["match"],
                "predicted": m["predicted_outcome"],
                "actual": m["actual_outcome"],
                "correct": m["predicted_outcome"].upper() == m["actual_outcome"],
                "confidence": m["confidence"],
                "date": m.get("date", ""),
            }
            for m in matched
        ],
    }

    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    with open(ANALYSIS_PATH, "w") as f:
        json.dump(analysis, f, indent=2, cls=_NumpySafeEncoder)

    log.info("Feedback analysis saved to %s", ANALYSIS_PATH)
    log.info("  Settled: %d, Accuracy: %.1f%%, Mean Brier: %.4f",
             overall.get("n_settled", 0),
             overall.get("accuracy", 0) * 100,
             overall.get("mean_brier", 0))

    if method_brier:
        best = min(method_brier.items(), key=lambda x: x[1]["mean_brier"])
        log.info("  Best method: %s (Brier: %.4f)", best[0], best[1]["mean_brier"])

    return analysis


if __name__ == "__main__":
    analysis = run_feedback_analysis()
    print(json.dumps(analysis, indent=2, cls=_NumpySafeEncoder))
