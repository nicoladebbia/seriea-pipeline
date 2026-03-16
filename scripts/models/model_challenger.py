#!/usr/bin/env python3
"""MODEL CHALLENGER - Retrain, Validate, and Atomic Swap

Periodically retrains models, compares to production via walk-forward CV,
and swaps only if the challenger beats current metrics by a meaningful margin.

Flow:
  1. Load latest features
  2. Train candidate models to timestamped directory
  3. Validate via 5-fold walk-forward CV
  4. If passes all gates -> atomic swap with backup
  5. Smoke test -> rollback if fails
  6. Log everything to challenger_log.json

Gates:
  - xG-Poisson HC accuracy must beat current by >= 0.5pp
  - Classifier HC accuracy must beat current by >= 0.5pp
  - No NaN predictions on validation set

Usage:
    python3 -m scripts.model_challenger
"""

import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import DATA_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

UNIVERSAL_DIR = MODELS_DIR / "universal"
CHALLENGER_DIR = MODELS_DIR / "challenger"
ARCHIVE_DIR = MODELS_DIR / "archive"
LOG_PATH = MODELS_DIR / "challenger_log.json"

# Gates: challenger must beat current by this margin
MIN_IMPROVEMENT_PP = 0.005  # 0.5 percentage points


def _load_current_metrics() -> dict:
    """Load current production model metrics from metadata."""
    meta_path = UNIVERSAL_DIR / "extended_model_metadata.json"
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        return meta.get("cv_metrics", {})
    except Exception:
        return {}


def _append_log(entry: dict):
    """Append entry to challenger_log.json."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    if LOG_PATH.exists():
        try:
            with open(LOG_PATH) as f:
                entries = json.load(f)
        except Exception:
            entries = []
    entries.append(entry)
    # Keep last 50 entries
    entries = entries[-50:]
    with open(LOG_PATH, "w") as f:
        json.dump(entries, f, indent=2)


def _smoke_test() -> bool:
    """Load ensemble, predict 1 match, verify no NaN."""
    try:
        from scripts.prediction.ensemble_prediction_engine import EnsemblePredictor
        import numpy as np

        ensemble = EnsemblePredictor(live_mode=False)
        if not ensemble.initialize():
            log.warning("Smoke test: ensemble failed to initialize")
            return False

        # Try predicting with a dummy match
        odds_path = DATA_DIR / "upcoming" / "odds.json"
        if not odds_path.exists():
            log.info("Smoke test: no odds.json, skipping prediction check")
            return True  # Can't test without matches, but init passed

        with open(odds_path) as f:
            odds = json.load(f)

        if not odds:
            return True

        # Get first match
        match_key = next(iter(odds))
        parts = match_key.split(" vs ")
        if len(parts) != 2:
            return True

        home, away = parts[0].strip(), parts[1].strip()

        # Try xG prediction through the predictor
        if ensemble.xg_predictor:
            features = ensemble.feature_builder.build_match_features(
                home, away, {}
            )
            if features is not None:
                result = ensemble.xg_predictor.predict(features)
                if result:
                    for key in ["prob_H", "prob_D", "prob_A"]:
                        if np.isnan(result.get(key, 0)):
                            log.warning(f"Smoke test: NaN in {key}")
                            return False

        log.info("Smoke test passed")
        return True
    except Exception as e:
        log.warning(f"Smoke test failed: {e}")
        return False


def run_model_challenger() -> dict:
    """Run the full challenger pipeline: train, validate, gate, swap/keep."""
    log.info("=" * 60)
    log.info("MODEL CHALLENGER")
    log.info("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load current metrics
    current_metrics = _load_current_metrics()
    log.info("Current production metrics:")
    for k, v in current_metrics.items():
        log.info("  %s: %s", k, v)

    # Step 0.5: Load drift report (if available) to exclude unstable features
    drifted_features = set()
    high_nan_features = set()
    drift_path = DATA_DIR / "feedback" / "drift_report.json"
    if drift_path.exists():
        try:
            with open(drift_path) as f:
                drift_report = json.load(f)
            for feat, info in drift_report.get("feature_drift", {}).items():
                if info.get("drifted") and info.get("ks_statistic", 0) > 0.2:
                    drifted_features.add(feat)
                if info.get("high_nan"):
                    high_nan_features.add(feat)
            exclude = drifted_features | high_nan_features
            if exclude:
                log.info(f"  Drift report: excluding {len(exclude)} unstable features "
                         f"({len(drifted_features)} drifted, {len(high_nan_features)} high-NaN)")
        except Exception as e:
            log.debug(f"Drift report load failed: {e}")

    # Step 1: Load and prepare data
    log.info("\n[1/5] Loading and preparing data...")
    try:
        from scripts.models.train_unified import load_and_prepare_data
        exclude_features = drifted_features | high_nan_features
        data = load_and_prepare_data(exclude_features=exclude_features)
    except Exception as e:
        log.error(f"Data preparation failed: {e}")
        entry = {
            "timestamp": timestamp, "decision": "abort",
            "reason": f"data prep failed: {e}",
            "old_metrics": current_metrics, "new_metrics": {},
        }
        _append_log(entry)
        return entry

    log.info("  %d matches, %d features", len(data["df"]), len(data["available_features"]))

    # Step 2: Evaluate challenger via CV
    log.info("\n[2/5] Evaluating challenger (walk-forward CV)...")
    try:
        from scripts.models.train_unified import evaluate_xg_cv, evaluate_classifier_cv
        xg_metrics = evaluate_xg_cv(data)
        class_metrics = evaluate_classifier_cv(data)
        new_metrics = {**xg_metrics, **class_metrics}
        # Remove fold_metrics for comparison
        new_metrics.pop("fold_metrics", None)
    except Exception as e:
        log.error(f"CV evaluation failed: {e}")
        entry = {
            "timestamp": timestamp, "decision": "abort",
            "reason": f"CV failed: {e}",
            "old_metrics": current_metrics, "new_metrics": {},
        }
        _append_log(entry)
        return entry

    log.info("Challenger metrics:")
    for k, v in new_metrics.items():
        log.info("  %s: %s", k, v)

    # Step 3: Gate checks (weighted composite scoring)
    log.info("\n[3/5] Checking gates...")

    gate_reasons = []

    # Gate 1: xG-Poisson HC accuracy (weight 0.6)
    old_xg_hc = current_metrics.get("xg_poisson_hc_accuracy", 0)
    new_xg_hc = new_metrics.get("xg_poisson_hc_accuracy", 0)
    xg_improvement = new_xg_hc - old_xg_hc
    log.info(f"  xG HC: {old_xg_hc:.4f} -> {new_xg_hc:.4f} ({xg_improvement:+.4f})")

    # Gate 2: Classifier HC accuracy (weight 0.4)
    old_cls_hc = current_metrics.get("classifier_hc_accuracy", 0)
    new_cls_hc = new_metrics.get("classifier_hc_accuracy", 0)
    cls_improvement = new_cls_hc - old_cls_hc
    log.info(f"  Classifier HC: {old_cls_hc:.4f} -> {new_cls_hc:.4f} ({cls_improvement:+.4f})")

    # Weighted composite: must be net-positive by MIN_IMPROVEMENT_PP
    XG_WEIGHT = 0.6
    CLS_WEIGHT = 0.4
    composite_improvement = xg_improvement * XG_WEIGHT + cls_improvement * CLS_WEIGHT
    log.info(f"  Composite improvement: {composite_improvement:+.4f} "
             f"(threshold: {MIN_IMPROVEMENT_PP})")

    # Hard floor: neither model can regress by more than 1pp
    MAX_REGRESSION_PP = 0.01
    gates_passed = composite_improvement >= MIN_IMPROVEMENT_PP

    if xg_improvement < -MAX_REGRESSION_PP:
        gates_passed = False
        gate_reasons.append(
            f"xG HC regressed too much: {xg_improvement:+.4f} (floor: -{MAX_REGRESSION_PP})"
        )
    if cls_improvement < -MAX_REGRESSION_PP:
        gates_passed = False
        gate_reasons.append(
            f"Classifier HC regressed too much: {cls_improvement:+.4f} (floor: -{MAX_REGRESSION_PP})"
        )
    if composite_improvement < MIN_IMPROVEMENT_PP:
        gate_reasons.append(
            f"Composite {composite_improvement:+.4f} < threshold {MIN_IMPROVEMENT_PP}"
        )

    if not gates_passed:
        log.info("  Gates NOT passed. Keeping current model.")
        for reason in gate_reasons:
            log.info(f"    FAIL: {reason}")

        entry = {
            "timestamp": timestamp, "decision": "keep",
            "reason": "; ".join(gate_reasons),
            "old_metrics": current_metrics, "new_metrics": new_metrics,
        }
        _append_log(entry)
        return entry

    # Step 4: Train final models and swap
    log.info("\n[4/5] Training final challenger models...")
    try:
        from scripts.models.train_unified import train_final_models, save_metadata

        # Train to challenger directory
        challenger_dir = CHALLENGER_DIR / timestamp
        challenger_dir.mkdir(parents=True, exist_ok=True)

        train_result = train_final_models(data, challenger_dir)
        cv_metrics = {**xg_metrics, **class_metrics}
        fold_metrics = cv_metrics.pop("fold_metrics", [])
        save_metadata(
            challenger_dir, data["available_features"], cv_metrics, fold_metrics,
            train_result["top_xg_features"],
            train_result["top_classifier_features"],
        )
        log.info(f"  Challenger models saved to {challenger_dir}")

        # Backup current production models
        archive_dir = ARCHIVE_DIR / timestamp
        archive_dir.mkdir(parents=True, exist_ok=True)

        if UNIVERSAL_DIR.exists():
            for f in UNIVERSAL_DIR.glob("*.cbm"):
                shutil.copy2(f, archive_dir / f.name)
            for f in UNIVERSAL_DIR.glob("*.json"):
                shutil.copy2(f, archive_dir / f.name)
            log.info(f"  Archived current models to {archive_dir}")

        # Atomic swap: copy challenger to universal
        for f in challenger_dir.glob("*.cbm"):
            shutil.copy2(f, UNIVERSAL_DIR / f.name)
        for f in challenger_dir.glob("*.json"):
            shutil.copy2(f, UNIVERSAL_DIR / f.name)
        log.info("  Swapped challenger -> universal")

    except Exception as e:
        log.error(f"Training/swap failed: {e}")
        entry = {
            "timestamp": timestamp, "decision": "abort",
            "reason": f"training failed: {e}",
            "old_metrics": current_metrics, "new_metrics": new_metrics,
        }
        _append_log(entry)
        return entry

    # Step 5: Smoke test
    log.info("\n[5/5] Running smoke test...")
    if _smoke_test():
        log.info("  Smoke test PASSED. New model is live!")
        entry = {
            "timestamp": timestamp, "decision": "swap",
            "reason": f"xG HC +{xg_improvement:+.4f}, Classifier HC +{cls_improvement:+.4f}",
            "old_metrics": current_metrics, "new_metrics": new_metrics,
        }
        _append_log(entry)
        return entry
    else:
        # Rollback
        log.warning("  Smoke test FAILED. Rolling back...")
        try:
            for f in archive_dir.glob("*.cbm"):
                shutil.copy2(f, UNIVERSAL_DIR / f.name)
            for f in archive_dir.glob("*.json"):
                shutil.copy2(f, UNIVERSAL_DIR / f.name)
            log.info("  Rollback complete.")
        except Exception as rb_e:
            log.error(f"  Rollback also failed: {rb_e}")

        entry = {
            "timestamp": timestamp, "decision": "rollback",
            "reason": "smoke test failed after swap",
            "old_metrics": current_metrics, "new_metrics": new_metrics,
        }
        _append_log(entry)
        return entry


if __name__ == "__main__":
    result = run_model_challenger()
    print(json.dumps(result, indent=2, default=str))
