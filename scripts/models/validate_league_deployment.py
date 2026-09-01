"""Validate a per-league model and create/update its deployment state.

Reads training_report.json for the specified league, checks CV metrics
against rejection thresholds, and writes a deployment_state.json that
the betting engine uses to gate bet placement.

Usage:
    python -m scripts.models.validate_league_deployment --league premier_league
    python -m scripts.models.validate_league_deployment --league serie_a
    python -m scripts.models.validate_league_deployment --league premier_league --enable
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from config.settings import MODELS_DIR

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Universal rejection thresholds — same standard for ALL leagues.
# A league model must meet ALL of these to enable betting.
# ---------------------------------------------------------------------------
REJECTION_THRESHOLDS = {
    "accuracy_min": 0.50,       # Honest walk-forward ceiling for 1X2 is 51-53%; 50% is "beats coin flip on draws".
    "log_loss_max": 1.00,       # Random baseline is 1.0986; below 1.00 = real signal.
    "brier_max": 0.205,         # Holdout walk-forward typically 0.195-0.20; small headroom.
    "betting_yield_min": 0.025, # Kelly ROI > 2.5% on the held-out fold.
}


def _load_training_report(league: str) -> dict | None:
    """Load training_report.json for a league."""
    report_path = MODELS_DIR / league / "training_report.json"
    if not report_path.exists():
        log.error("No training report found at %s", report_path)
        return None
    with open(report_path) as f:
        return json.load(f)


def _load_cv_results(league: str) -> list[dict] | None:
    """Load ensemble CV results for a league."""
    cv_path = MODELS_DIR / league / "ensemble" / "cv_results.json"
    if not cv_path.exists():
        # Try alternate location
        cv_path = MODELS_DIR / league / "cv_results.json"
    if not cv_path.exists():
        log.warning("No cv_results.json found for %s", league)
        return None
    with open(cv_path) as f:
        return json.load(f)


def _compute_avg_metrics(report: dict, cv_results: list[dict] | None) -> dict:
    """Extract average CV metrics from training report or cv_results."""
    metrics = {}

    # Try cv_results first (more detailed per-fold data)
    if cv_results:
        n = len(cv_results)
        metrics["accuracy"] = sum(f.get("ensemble_accuracy", 0) for f in cv_results) / n
        metrics["log_loss"] = sum(f.get("ensemble_log_loss", 0) for f in cv_results) / n
        metrics["brier"] = sum(f.get("ensemble_brier", 0) for f in cv_results) / n
        metrics["rps"] = sum(f.get("ensemble_rps", 0) for f in cv_results) / n
        metrics["ece"] = sum(f.get("ensemble_ece", 0) for f in cv_results) / n
        metrics["kelly_roi"] = sum(f.get("ensemble_kelly_roi", 0) for f in cv_results) / n
        metrics["f1_draw"] = sum(f.get("ensemble_f1_D", 0) for f in cv_results) / n
        metrics["n_folds"] = n
        # Per-fold breakdown
        metrics["per_fold"] = [
            {
                "test_season": f.get("test", ""),
                "accuracy": f.get("ensemble_accuracy", 0),
                "log_loss": f.get("ensemble_log_loss", 0),
                "kelly_roi": f.get("ensemble_kelly_roi", 0),
                "f1_draw": f.get("ensemble_f1_D", 0),
                "n_test": f.get("n_test", 0),
            }
            for f in cv_results
        ]
        return metrics

    # Fallback: extract from training report's per_model_cv
    per_model = report.get("per_model_cv", {})
    # Use CatBoost as representative (highest weight in ensemble typically)
    for model_type in ["catboost", "lightgbm", "xgboost"]:
        if model_type in per_model:
            cv = per_model[model_type]
            acc_vals = list(cv.get("accuracy", {}).values())
            ll_vals = list(cv.get("log_loss", {}).values())
            brier_vals = list(cv.get("brier_score", {}).values())
            kelly_vals = list(cv.get("kelly_roi", {}).values())
            if acc_vals:
                metrics["accuracy"] = sum(acc_vals) / len(acc_vals)
                metrics["log_loss"] = sum(ll_vals) / len(ll_vals)
                metrics["brier"] = sum(brier_vals) / len(brier_vals)
                metrics["kelly_roi"] = sum(kelly_vals) / len(kelly_vals)
                metrics["model_used"] = model_type
                break

    return metrics


def validate_league(league: str, force_enable: bool = False) -> dict:
    """Validate a league model and return validation results.

    Returns dict with: passed (bool), metrics, checks, deployment_state
    """
    report = _load_training_report(league)
    cv_results = _load_cv_results(league)

    if report is None and cv_results is None:
        return {
            "passed": False,
            "reason": f"No training report or CV results found for '{league}'",
            "metrics": {},
            "checks": {},
        }

    metrics = _compute_avg_metrics(report or {}, cv_results)

    # Run checks against thresholds
    checks = {}
    all_passed = True

    accuracy = metrics.get("accuracy", 0)
    checks["accuracy"] = {
        "value": accuracy,
        "threshold": REJECTION_THRESHOLDS["accuracy_min"],
        "passed": accuracy >= REJECTION_THRESHOLDS["accuracy_min"],
    }
    if not checks["accuracy"]["passed"]:
        all_passed = False

    log_loss = metrics.get("log_loss", 999)
    checks["log_loss"] = {
        "value": log_loss,
        "threshold": REJECTION_THRESHOLDS["log_loss_max"],
        "passed": log_loss <= REJECTION_THRESHOLDS["log_loss_max"],
    }
    if not checks["log_loss"]["passed"]:
        all_passed = False

    brier = metrics.get("brier", 999)
    checks["brier"] = {
        "value": brier,
        "threshold": REJECTION_THRESHOLDS["brier_max"],
        "passed": brier <= REJECTION_THRESHOLDS["brier_max"],
    }
    if not checks["brier"]["passed"]:
        all_passed = False

    kelly = metrics.get("kelly_roi", 0)
    checks["kelly_roi"] = {
        "value": kelly,
        "threshold": REJECTION_THRESHOLDS["betting_yield_min"] * 100,
        "passed": kelly >= REJECTION_THRESHOLDS["betting_yield_min"] * 100,
    }
    if not checks["kelly_roi"]["passed"]:
        all_passed = False

    betting_enabled = all_passed or force_enable

    # Build deployment state
    deployment_state = {
        "league": league,
        "current_model": "per-league ensemble",
        "model_version": f"per_league_{league}",
        "deployed_at": datetime.now(timezone.utc).isoformat(),
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "accuracy": round(accuracy, 4),
            "log_loss": round(log_loss, 4),
            "brier": round(brier, 4),
            "rps": round(metrics.get("rps", 0), 4),
            "ece": round(metrics.get("ece", 0), 4),
            "f1_draw": round(metrics.get("f1_draw", 0), 4),
            "kelly_roi": round(kelly, 4),
        },
        "rejection_thresholds": REJECTION_THRESHOLDS,
        "validation_status": "passed" if all_passed else "failed",
        "betting_enabled": betting_enabled,
        "force_enabled": force_enable and not all_passed,
        "per_fold": metrics.get("per_fold", []),
    }

    return {
        "passed": all_passed,
        "betting_enabled": betting_enabled,
        "metrics": metrics,
        "checks": checks,
        "deployment_state": deployment_state,
    }


def save_deployment_state(league: str, state: dict) -> Path:
    """Save deployment state JSON for a league."""
    out_dir = MODELS_DIR / league
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "deployment_state.json"
    out_path.write_text(json.dumps(state, indent=2, default=str))
    return out_path


def print_validation_report(league: str, result: dict) -> None:
    """Print a human-readable validation report."""
    checks = result["checks"]
    metrics = result["metrics"]

    print(f"\n{'=' * 50}")
    print(f"  {league.upper()} MODEL VALIDATION")
    print(f"{'=' * 50}")

    for name, check in checks.items():
        status = "✓" if check["passed"] else "✗ FAIL"
        direction = "<=" if name in ("log_loss", "brier") else ">="
        print(f"  {name:12s}: {check['value']:7.4f}  ({direction} {check['threshold']})  {status}")

    print(f"\n  F1_Draw:      {metrics.get('f1_draw', 0):.4f}")
    if "per_fold" in metrics:
        print(f"\n  Per-fold breakdown:")
        for fold in metrics["per_fold"]:
            print(f"    {fold['test_season']}: acc={fold['accuracy']:.1%}  "
                  f"ll={fold['log_loss']:.4f}  kelly={fold['kelly_roi']:.1f}%  "
                  f"f1_D={fold['f1_draw']:.3f}  (n={fold['n_test']})")

    print(f"\n  {'=' * 48}")
    if result["passed"]:
        print(f"  RESULT: ✓ PASS — betting enabled for {league}")
    elif result.get("betting_enabled"):
        print(f"  RESULT: ✗ FAIL — but force-enabled via --enable flag")
    else:
        print(f"  RESULT: ✗ FAIL — betting DISABLED for {league}")
    print(f"  {'=' * 48}\n")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Validate league model for betting deployment")
    parser.add_argument("--league", required=True, help="League key (e.g., premier_league, serie_a)")
    parser.add_argument("--enable", action="store_true",
                        help="Force-enable betting even if validation fails (use with caution)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print results but don't write deployment_state.json")
    parser.add_argument("--paper", action="store_true",
                        help="Report the league's PAPER track against the "
                             "deployment bar (50+ settled, CLV+) and exit")
    args = parser.parse_args()

    if args.paper:
        from scripts.betting.bet_journal import get_paper_track_stats
        s = get_paper_track_stats(args.league)
        print(f"\nPAPER TRACK — {args.league}")
        print(f"  settled: {s['n_settled']}  (W {s['n_won']} / L {s['n_lost']})"
              f"  pending: {s['n_pending']}")
        print(f"  flat ROI: {s['roi_pct']}%  profit: {s['profit']}")
        print(f"  mean CLV: {s['mean_clv_pct']}%  (n={s['n_clv']})")
        bar_n = s["n_settled"] >= 50
        bar_clv = (s["mean_clv_pct"] or 0) > 0
        print(f"  deployment bar: settled>=50 [{'PASS' if bar_n else 'not yet'}]"
              f"  CLV+ [{'PASS' if bar_clv else 'not yet'}]")
        if bar_n and bar_clv:
            print("  → bar MET. Re-run validation without --paper to lift the gate.")
        sys.exit(0)

    result = validate_league(args.league, force_enable=args.enable)
    print_validation_report(args.league, result)

    if not args.dry_run:
        path = save_deployment_state(args.league, result["deployment_state"])
        print(f"  Deployment state saved to: {path}")
        gate_status = "ENABLED" if result["betting_enabled"] else "DISABLED"
        print(f"  Betting gate: {gate_status}")
    else:
        print("  (dry-run — no files written)")

    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
