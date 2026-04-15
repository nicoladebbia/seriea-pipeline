#!/usr/bin/env python3
"""Automated model retraining triggered by matchweek completion.

Runs after the last match of each Serie A matchweek is played and settled.
Auto-detects: quick retrain normally, full Optuna retrain on last week of month.

Modes:
  (default)   Auto: quick or full based on calendar
  --quick     Force quick retrain (~15-20 min)
  --full      Force full Optuna retrain (~2-3 hours)
  --check     Just check if a matchweek completed — don't retrain
  --dry-run   Compare only, don't promote

Safety:
  - Archives current models before promoting new ones
  - Compares new vs old on walk-forward CV log_loss
  - Only promotes if new model's log_loss is within tolerance
  - Tracks metrics history for trend analysis
  - Sends macOS notification on success/failure

Usage:
    python -m scripts.pipeline.weekly_retrain              # Auto (triggered by settlement)
    python -m scripts.pipeline.weekly_retrain --check      # Check if matchweek done
    python -m scripts.pipeline.weekly_retrain --quick      # Force quick retrain
    python -m scripts.pipeline.weekly_retrain --full       # Force full retrain
    python -m scripts.pipeline.weekly_retrain --dry-run    # Compare only, don't promote
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MODELS_DIR
from storage.paths import features_path

log = logging.getLogger("weekly_retrain")

DATA_DIR = PROJECT_ROOT / "data"
PARSED_DIR = DATA_DIR / "parsed"
METRICS_HISTORY = DATA_DIR / "models" / "retrain_history.jsonl"
RETRAIN_STATE = DATA_DIR / "models" / "retrain_state.json"
ARCHIVE_DIR = MODELS_DIR / "universal" / "archive"

MATCHES_PER_MATCHWEEK = 10  # Serie A: 20 teams = 10 matches per round


# ---------------------------------------------------------------------------
# Matchweek completion detection
# ---------------------------------------------------------------------------

def _load_retrain_state() -> dict:
    """Load state tracking which matchweek we last retrained on."""
    if RETRAIN_STATE.exists():
        try:
            with open(RETRAIN_STATE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_retrain_state(state: dict):
    """Save retrain state."""
    RETRAIN_STATE.parent.mkdir(parents=True, exist_ok=True)
    with open(RETRAIN_STATE, "w") as f:
        json.dump(state, f, indent=2)


def get_matchweek_status() -> dict:
    """Check current matchweek completion status.

    Uses total match count to determine completed matchweeks, because
    Serie A rescheduled/postponed matches can get assigned to different
    matchweek numbers by the parser (e.g. MW 27 may show only 2 matches
    while its postponed games appear under MW 28/29).

    290 total matches = 29 complete matchweeks (10 matches each, 20 teams).

    Returns dict with:
        completed_matchweeks: int — how many full matchweeks are done
        total_matches: int — matches played this season
        last_match_date: str — date of most recent match
        last_retrained_matchweek: int — what we already retrained on
        needs_retrain: bool — new matchweek complete AND not yet retrained
    """
    try:
        import pandas as pd
        matches_path = PARSED_DIR / "matches.parquet"
        if not matches_path.exists():
            return {"error": "matches.parquet not found"}

        df = pd.read_parquet(matches_path)
        current_season = df[df["season"] == "2025-2026"]
        if current_season.empty:
            return {"error": "No 2025-2026 season data"}

        # Total matches with scores = reliable matchweek count
        has_score = current_season["home_score"].notna()
        total_matches = int(has_score.sum())
        completed_mws = total_matches // MATCHES_PER_MATCHWEEK
        remainder = total_matches % MATCHES_PER_MATCHWEEK

        last_date = str(current_season["match_date"].max())[:10]
        days_since_last = (datetime.now() - pd.Timestamp(last_date)).days

        # Check if we already retrained for this matchweek count
        state = _load_retrain_state()
        last_retrained_mw = state.get("last_retrained_matchweek", 0)

        needs_retrain = completed_mws > last_retrained_mw

        return {
            "completed_matchweeks": completed_mws,
            "current_matchweek": completed_mws,  # alias for display
            "total_matches": total_matches,
            "matches_in_progress": remainder,
            "last_match_date": last_date,
            "days_since_last_match": days_since_last,
            "last_retrained_matchweek": last_retrained_mw,
            "needs_retrain": needs_retrain,
        }

    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _notify(message: str, title: str = "Serie A Retrain"):
    """Send notification via all configured channels (macOS + Telegram)."""
    from scripts.pipeline.notify import notify as _unified_notify

    # Infer level from title keywords
    title_lower = title.lower()
    if "fail" in title_lower or "reject" in title_lower:
        level = "error"
    elif "success" in title_lower:
        level = "success"
    elif "skip" in title_lower:
        level = "warning"
    else:
        level = "info"

    _unified_notify(message, title=title, level=level)


def _refresh_predictions():
    """Re-run predictions for upcoming matches using the freshly trained model.

    This is critical: after retraining, the predictions for the next matchweek
    must be regenerated so they use the new model, not the old one.
    """
    log.info("Refreshing predictions with new model...")
    try:
        from scripts.prediction.ensemble_prediction_engine import run_ensemble_predictions
        result = run_ensemble_predictions(use_ensemble=True)
        n = result.get("n_predictions", 0)
        log.info("Predictions refreshed: %d matches predicted with new model", n)
        return True
    except Exception as e:
        log.error("Prediction refresh failed: %s", e)
        # Non-fatal — the morning pipeline run will catch this
        return False


def _load_current_cv_metrics() -> dict | None:
    """Load CV results from current production ensemble."""
    cv_path = MODELS_DIR / "universal" / "ensemble" / "cv_results.json"
    if not cv_path.exists():
        return None
    try:
        with open(cv_path) as f:
            folds = json.load(f)
        if not folds:
            return None
        # Compute averages across folds
        keys = ["ensemble_accuracy", "ensemble_log_loss", "ensemble_brier",
                "ensemble_rps", "ensemble_ece", "ensemble_kelly_roi"]
        avgs = {}
        for k in keys:
            vals = [f[k] for f in folds if k in f]
            avgs[k] = sum(vals) / len(vals) if vals else None
        return avgs
    except Exception as e:
        log.warning("Failed to load current CV metrics: %s", e)
        return None


def _load_tuned_params() -> dict | None:
    """Load previously tuned hyperparameters from training report."""
    report_path = MODELS_DIR / "universal" / "training_report.json"
    if not report_path.exists():
        return None
    try:
        with open(report_path) as f:
            report = json.load(f)
        return report.get("tuned_params")
    except Exception:
        return None


def _load_selected_features() -> list[str] | None:
    """Load previously selected feature names from training report."""
    report_path = MODELS_DIR / "universal" / "training_report.json"
    if not report_path.exists():
        return None
    try:
        with open(report_path) as f:
            report = json.load(f)
        return report.get("selected_features")
    except Exception:
        return None


def _archive_current_models():
    """Copy current production models to timestamped archive."""
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    archive_path = ARCHIVE_DIR / ts

    src = MODELS_DIR / "universal"
    if not src.exists():
        log.warning("No current models to archive")
        return None

    # Copy model files (not the archive dir itself)
    archive_path.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name == "archive" or item.name == "challenger":
            continue
        if item.is_file():
            shutil.copy2(item, archive_path / item.name)
        elif item.is_dir() and item.name == "ensemble":
            shutil.copytree(item, archive_path / "ensemble", dirs_exist_ok=True)

    log.info("Archived current models to %s", archive_path)

    # Keep only last 5 archives
    archives = sorted(ARCHIVE_DIR.iterdir())
    while len(archives) > 5:
        old = archives.pop(0)
        if old.is_dir():
            shutil.rmtree(old)
            log.info("Removed old archive: %s", old.name)

    return archive_path


def _append_metrics_history(entry: dict):
    """Append a retrain result to the JSONL history."""
    METRICS_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_HISTORY, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Auxiliary model retraining (catboost_no_odds + xG regressors)
# ---------------------------------------------------------------------------

def retrain_no_odds(dry_run: bool = False) -> dict:
    """Retrain catboost_no_odds.cbm using the existing feature set.

    This is the PRIMARY production model (60.5% ensemble weight).
    Uses the dedicated retrain script which validates against rejection thresholds.
    """
    result = {"model": "catboost_no_odds", "promoted": False}
    log.info("Retraining catboost_no_odds...")

    try:
        cmd = [sys.executable, "-m", "scripts.models.retrain_no_odds_catboost"]
        if dry_run:
            cmd.append("--dry-run")

        proc = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=600,
        )
        result["returncode"] = proc.returncode
        if proc.returncode == 0:
            result["promoted"] = True
            log.info("catboost_no_odds retrained successfully")
        else:
            log.warning("catboost_no_odds retrain failed: %s", proc.stderr[-500:] if proc.stderr else "unknown")
            result["error"] = proc.stderr[-300:] if proc.stderr else "exit code non-zero"

    except subprocess.TimeoutExpired:
        log.error("catboost_no_odds retrain timed out (10 min)")
        result["error"] = "timeout"
    except Exception as e:
        log.error("catboost_no_odds retrain error: %s", e)
        result["error"] = str(e)

    return result


def retrain_draw_detector(dry_run: bool = False) -> dict:
    """Retrain draw_detector.cbm + calibrator + ablation test."""
    result = {"model": "draw_detector", "promoted": False}
    log.info("Retraining draw detector...")

    try:
        cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "models" / "retrain_draw_detector.py")]
        if dry_run:
            cmd.append("--dry-run")

        proc = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=600,
        )
        result["returncode"] = proc.returncode
        if proc.returncode == 0:
            result["promoted"] = True
            log.info("draw_detector retrained successfully")
        else:
            log.warning("draw_detector retrain issue: %s", proc.stderr[-500:] if proc.stderr else "unknown")
            result["error"] = proc.stderr[-300:] if proc.stderr else "exit code non-zero"

    except subprocess.TimeoutExpired:
        log.error("draw_detector retrain timed out (10 min)")
        result["error"] = "timeout"
    except Exception as e:
        log.error("draw_detector retrain error: %s", e)
        result["error"] = str(e)

    return result


def retrain_xg_models(dry_run: bool = False) -> dict:
    """Retrain xg_home.cbm and xg_away.cbm (xG regressors for Poisson predictions).

    Uses train_unified.py in xg_only mode.
    """
    result = {"model": "xg_home/xg_away", "promoted": False}
    log.info("Retraining xG models (xg_home + xg_away)...")

    try:
        cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "models" / "train_unified.py"),
               "--mode", "xg_only"]

        proc = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=600,
        )
        result["returncode"] = proc.returncode
        if proc.returncode == 0:
            result["promoted"] = True
            log.info("xG models retrained successfully")
        else:
            log.warning("xG retrain failed: %s", proc.stderr[-500:] if proc.stderr else "unknown")
            result["error"] = proc.stderr[-300:] if proc.stderr else "exit code non-zero"

    except subprocess.TimeoutExpired:
        log.error("xG retrain timed out (10 min)")
        result["error"] = "timeout"
    except Exception as e:
        log.error("xG retrain error: %s", e)
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Feature rebuild
# ---------------------------------------------------------------------------

def rebuild_features() -> bool:
    """Rebuild features.parquet from latest parsed data."""
    log.info("Rebuilding features...")
    t0 = time.time()
    try:
        from features.build import build_features
        build_features(use_cache=False)
        elapsed = time.time() - t0
        log.info("Features rebuilt in %.1f seconds", elapsed)
        return True
    except Exception as e:
        log.error("Feature rebuild failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Quick retrain: reuse hyperparams + features, retrain + ensemble
# ---------------------------------------------------------------------------

def quick_retrain(dry_run: bool = False) -> dict:
    """Retrain models with existing hyperparams and feature set.

    Steps:
        1. Rebuild features.parquet
        2. Load tuned hyperparams and selected features from last train_optimized
        3. Train universal models with those params
        4. Rebuild ensemble with walk-forward CV
        5. Compare new vs current CV metrics
        6. Promote if not worse (log_loss within tolerance)
    """
    result = {
        "mode": "quick",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "promoted": False,
    }

    # Step 1: Rebuild features
    if not rebuild_features():
        result["error"] = "Feature rebuild failed"
        _notify("Feature rebuild failed", "Retrain FAILED")
        _append_metrics_history(result)
        return result

    # Step 2: Load existing params and features
    tuned_params = _load_tuned_params()
    selected_features = _load_selected_features()

    if not tuned_params:
        log.warning("No tuned params found — falling back to full retrain")
        return full_retrain(dry_run=dry_run)

    if not selected_features:
        log.warning("No selected features found — falling back to full retrain")
        return full_retrain(dry_run=dry_run)

    log.info("Using %d tuned features and existing hyperparams", len(selected_features))

    # Step 3: Load current metrics for comparison
    current_metrics = _load_current_cv_metrics()
    if current_metrics:
        log.info(
            "Current model: acc=%.4f  ll=%.4f  brier=%.4f",
            current_metrics.get("ensemble_accuracy", 0),
            current_metrics.get("ensemble_log_loss", 0),
            current_metrics.get("ensemble_brier", 0),
        )
    result["current_metrics"] = current_metrics

    # Step 4: Train + ensemble with existing params
    log.info("Training models with existing hyperparams...")
    t0 = time.time()

    try:
        from ml.data import DataLoader
        from ml.ensemble import WeightedAverageEnsemble, evaluate_ensemble_cv
        from ml.training import train_universal

        # Load data (respects min_train_season=2017-2018 from ValidationConfig)
        log.info("Loading data...")
        fp = str(features_path())
        loader = DataLoader(fp)
        X, y, _ = loader.get_universal_dataset()

        # Filter to selected features + meta
        keep = [c for c in selected_features if c in X.columns]
        meta = [c for c in X.columns if c.startswith("_")]
        X = X[keep + [c for c in meta if c not in keep]]
        log.info("Data: %d rows, %d features", len(X), len(keep))

        # Train individual models with tuned params + selected features
        train_results = train_universal(
            validate=True,
            params=tuned_params,
            feature_names_override=keep,
        )

        # Build ensemble on the same filtered dataset
        log.info("Building ensemble...")
        ens = WeightedAverageEnsemble(model_configs=tuned_params)
        ens.fit(X, y, feature_names=keep)

        # Evaluate ensemble
        cv_results = evaluate_ensemble_cv(
            X, y, keep, model_configs=tuned_params,
        )

        elapsed = time.time() - t0
        log.info("Training completed in %.1f seconds", elapsed)
        result["training_time_seconds"] = round(elapsed, 1)

    except Exception as e:
        log.error("Training failed: %s", e)
        result["error"] = str(e)
        _notify(f"Training failed: {e}", "Retrain FAILED")
        _append_metrics_history(result)
        return result

    # Step 5: Extract new metrics
    if isinstance(cv_results, pd.DataFrame) and not cv_results.empty:
        new_metrics = {
            "ensemble_accuracy": cv_results["ensemble_accuracy"].mean(),
            "ensemble_log_loss": cv_results["ensemble_log_loss"].mean(),
            "ensemble_brier": cv_results["ensemble_brier"].mean(),
        }
        for k in ["ensemble_rps", "ensemble_ece", "ensemble_kelly_roi"]:
            if k in cv_results.columns:
                new_metrics[k] = cv_results[k].mean()
    else:
        new_metrics = {}
        log.warning("No CV results from ensemble evaluation")

    result["new_metrics"] = new_metrics
    log.info(
        "New model: acc=%.4f  ll=%.4f  brier=%.4f",
        new_metrics.get("ensemble_accuracy", 0),
        new_metrics.get("ensemble_log_loss", 0),
        new_metrics.get("ensemble_brier", 0),
    )

    # Step 6: Compare and decide
    TOLERANCE = 0.02  # Allow up to 0.02 worse log_loss
    promote = True
    reason = "no current model to compare"

    if current_metrics and new_metrics:
        old_ll = current_metrics.get("ensemble_log_loss", 999)
        new_ll = new_metrics.get("ensemble_log_loss", 999)
        diff = new_ll - old_ll

        if diff > TOLERANCE:
            promote = False
            reason = f"new model worse by {diff:.4f} log_loss (tolerance: {TOLERANCE})"
        elif diff < -0.005:
            reason = f"new model BETTER by {abs(diff):.4f} log_loss"
        else:
            reason = f"new model within tolerance (diff: {diff:+.4f})"

    result["comparison"] = reason
    log.info("Decision: %s", reason)

    if dry_run:
        log.info("DRY RUN — skipping promotion")
        result["promoted"] = False
        result["dry_run"] = True
        _append_metrics_history(result)
        return result

    if promote:
        log.info("Promoting new models...")
        archive_path = _archive_current_models()
        result["archive_path"] = str(archive_path) if archive_path else None

        # Save ensemble (overwrites _latest files)
        try:
            ens.save("universal")

            # Save CV results
            cv_path = MODELS_DIR / "universal" / "ensemble" / "cv_results.json"
            if isinstance(cv_results, pd.DataFrame):
                cv_results.to_json(cv_path, orient="records", indent=2)

            result["promoted"] = True

            # Re-run predictions with the new model
            _refresh_predictions()

            log.info("Quick retrain complete. LL: %.4f (%s)", new_metrics.get('ensemble_log_loss', 0), reason)
            try:
                from scripts.pipeline.notify import notify_retrain
                notify_retrain(
                    mode="quick",
                    matchweek=result.get("matchweek", 0),
                    promoted=True,
                    old_ll=current_metrics.get("ensemble_log_loss", 0) if current_metrics else 0,
                    new_ll=new_metrics.get("ensemble_log_loss", 0),
                    reason=reason,
                )
            except Exception:
                _notify("Quick retrain complete. Predictions refreshed.", "Retrain SUCCESS")

        except Exception as e:
            log.error("Model promotion failed: %s", e)
            result["error"] = f"Promotion failed: {e}"
            _notify(f"Promotion failed: {e}", "Retrain FAILED")
    else:
        log.warning("NOT promoting — %s", reason)
        _notify(f"Retrain skipped: {reason}", "Retrain SKIPPED")

    _append_metrics_history(result)
    return result


# ---------------------------------------------------------------------------
# Full retrain: Optuna + feature selection + ensemble
# ---------------------------------------------------------------------------

def full_retrain(dry_run: bool = False) -> dict:
    """Full optimized retrain with Optuna tuning and feature re-selection.

    Wraps train_optimized() with comparison gate and archival.
    """
    result = {
        "mode": "full",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "promoted": False,
    }

    # Rebuild features first
    if not rebuild_features():
        result["error"] = "Feature rebuild failed"
        _notify("Feature rebuild failed", "Full Retrain FAILED")
        _append_metrics_history(result)
        return result

    # Load current metrics for comparison
    current_metrics = _load_current_cv_metrics()
    result["current_metrics"] = current_metrics

    log.info("Starting full optimized training (this will take 2-3 hours)...")
    t0 = time.time()

    try:
        from ml.training import train_optimized

        # Train both leagues with all improvements (exclude_odds, per-league)
        from config.leagues import ACTIVE_LEAGUES
        for _league in ACTIVE_LEAGUES:
            log.info("Training %s...", _league)
            try:
                train_optimized(league=_league, exclude_odds=True)
                log.info("%s training complete", _league)
            except Exception as _le:
                log.error("%s training failed: %s — continuing with other leagues", _league, _le)

        elapsed = time.time() - t0
        log.info("Full training completed in %.1f seconds (%.1f min)", elapsed, elapsed / 60)
        result["training_time_seconds"] = round(elapsed, 1)

    except Exception as e:
        log.error("Full training failed: %s", e)
        result["error"] = str(e)
        _notify(f"Full retrain failed: {e}", "Full Retrain FAILED")
        _append_metrics_history(result)
        return result

    # Extract new metrics from CV results
    new_metrics = _load_current_cv_metrics()  # train_optimized saves these
    result["new_metrics"] = new_metrics

    if new_metrics:
        log.info(
            "New model: acc=%.4f  ll=%.4f  brier=%.4f",
            new_metrics.get("ensemble_accuracy", 0),
            new_metrics.get("ensemble_log_loss", 0),
            new_metrics.get("ensemble_brier", 0),
        )

    # Compare — full retrain always promotes unless catastrophically worse
    TOLERANCE = 0.05  # More lenient for full retrain (new features/params)
    promote = True
    reason = "full retrain — promoting by default"

    if current_metrics and new_metrics:
        old_ll = current_metrics.get("ensemble_log_loss", 999)
        new_ll = new_metrics.get("ensemble_log_loss", 999)
        diff = new_ll - old_ll

        if diff > TOLERANCE:
            promote = False
            reason = f"catastrophically worse: +{diff:.4f} log_loss"
        elif diff < -0.005:
            reason = f"improved by {abs(diff):.4f} log_loss"
        else:
            reason = f"within tolerance ({diff:+.4f})"

    result["comparison"] = reason

    if dry_run:
        log.info("DRY RUN — skipping promotion")
        result["dry_run"] = True
        _append_metrics_history(result)
        return result

    if promote:
        # train_optimized already saves models to _latest
        result["promoted"] = True

        # Re-run predictions with the new model
        _refresh_predictions()

        msg = (
            f"Full retrain complete. "
            f"LL: {new_metrics.get('ensemble_log_loss', 0):.4f} "
            f"({reason}). Predictions refreshed."
        )
        log.info(msg)
        _notify(msg, "Full Retrain SUCCESS")
    else:
        log.warning("NOT promoting — %s", reason)
        _notify(f"Full retrain rejected: {reason}", "Full Retrain REJECTED")

    # Save new features and params
    result["n_features"] = train_results.get("n_features_selected")
    _append_metrics_history(result)
    return result


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------

def rollback(version: str | None = None) -> bool:
    """Restore models from archive.

    If version is None, restores the most recent archive.
    """
    if not ARCHIVE_DIR.exists():
        log.error("No archive directory found")
        return False

    archives = sorted(ARCHIVE_DIR.iterdir())
    if not archives:
        log.error("No archives available")
        return False

    if version:
        target = ARCHIVE_DIR / version
        if not target.exists():
            log.error("Archive %s not found", version)
            return False
    else:
        target = archives[-1]

    dest = MODELS_DIR / "universal"
    log.info("Rolling back to archive: %s", target.name)

    # Copy archived files back
    for item in target.iterdir():
        dest_item = dest / item.name
        if item.is_file():
            shutil.copy2(item, dest_item)
        elif item.is_dir():
            if dest_item.exists():
                shutil.rmtree(dest_item)
            shutil.copytree(item, dest_item)

    log.info("Rollback complete from %s", target.name)
    _notify(f"Rolled back to {target.name}", "Model Rollback")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _is_last_week_of_month() -> bool:
    """Check if today is in the last 7 days of the month."""
    import calendar
    today = datetime.now()
    _, last_day = calendar.monthrange(today.year, today.month)
    return today.day > last_day - 7


def auto_retrain(dry_run: bool = False, force: bool = False) -> dict:
    """Smart retrain triggered by matchweek completion.

    1. Checks if a matchweek just completed (all 10 matches played)
    2. If yes and not already retrained for this matchweek → retrain
    3. Quick retrain normally, full Optuna on last week of month

    Called by the nightly settlement job or the periodic health monitor.
    """
    status = get_matchweek_status()

    if "error" in status:
        log.error("Can't check matchweek: %s", status["error"])
        return {"error": status["error"], "mode": "auto"}

    log.info(
        "Completed matchweeks: %d (%d matches, %d in progress), last_retrained=MW%d",
        status["completed_matchweeks"],
        status["total_matches"],
        status["matches_in_progress"],
        status["last_retrained_matchweek"],
    )

    if not status["needs_retrain"] and not force:
        if status["completed_matchweeks"] <= status["last_retrained_matchweek"]:
            reason = f"already retrained for MW {status['completed_matchweeks']}"
        else:
            reason = f"MW {status['completed_matchweeks'] + 1} in progress ({status['matches_in_progress']}/{MATCHES_PER_MATCHWEEK})"
        log.info("No retrain needed — %s", reason)
        return {
            "mode": "auto",
            "action": "skipped",
            "reason": "matchweek not complete or already retrained",
            "matchweek_status": status,
        }

    mw = status["current_matchweek"]
    log.info("Matchweek %d complete — triggering retrain", mw)

    # Send matchweek summary (all bets + results + P&L)
    try:
        from scripts.pipeline.notify import notify_matchweek_summary
        notify_matchweek_summary(matchweek=mw)
        log.info("Matchweek %d summary sent", mw)
    except Exception as e:
        log.debug("Matchweek summary failed: %s", e)

    _notify(f"MW {mw} complete. Starting retrain...", "Retrain Triggered")

    # Archive current models before retraining
    _archive_current_models()

    # Choose mode: full on last week of month, quick otherwise
    if _is_last_week_of_month():
        log.info("Last week of the month — running FULL retrain with Optuna")
        result = full_retrain(dry_run=dry_run)
    else:
        log.info("Normal matchweek — running QUICK retrain")
        result = quick_retrain(dry_run=dry_run)

    # Also retrain auxiliary models (catboost_no_odds + xG regressors)
    # These are not part of the ensemble but are critical production models.
    aux_results = {}
    log.info("Retraining auxiliary models (catboost_no_odds + xG)...")
    aux_results["no_odds"] = retrain_no_odds(dry_run=dry_run)
    aux_results["xg"] = retrain_xg_models(dry_run=dry_run)
    aux_results["draw_detector"] = retrain_draw_detector(dry_run=dry_run)
    result["auxiliary_models"] = aux_results

    aux_ok = sum(1 for r in aux_results.values() if r.get("promoted"))
    aux_fail = sum(1 for r in aux_results.values() if r.get("error"))
    log.info("Auxiliary models: %d promoted, %d failed", aux_ok, aux_fail)

    # Record that we retrained for this matchweek
    if result.get("promoted") or dry_run:
        state = _load_retrain_state()
        state["last_retrained_matchweek"] = mw
        state["last_retrained_at"] = datetime.now(timezone.utc).isoformat()
        state["last_mode"] = result.get("mode", "auto")
        _save_retrain_state(state)

    result["matchweek"] = mw
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Automated model retraining triggered by matchweek completion"
    )
    parser.add_argument("--full", action="store_true",
                        help="Force full Optuna retrain (2-3 hours)")
    parser.add_argument("--quick", action="store_true",
                        help="Force quick retrain (~15-20 min)")
    parser.add_argument("--check", action="store_true",
                        help="Just check matchweek status, don't retrain")
    parser.add_argument("--force", action="store_true",
                        help="Force retrain even if matchweek not complete")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compare only, don't promote")
    parser.add_argument("--rollback", nargs="?", const="latest", metavar="VERSION",
                        help="Rollback to archived model version")
    parser.add_argument("--history", action="store_true",
                        help="Show retrain history")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(PROJECT_ROOT / "logs" / "retrain.log"),
        ],
    )

    if args.rollback:
        version = None if args.rollback == "latest" else args.rollback
        success = rollback(version)
        sys.exit(0 if success else 1)

    if args.history:
        if not METRICS_HISTORY.exists():
            print("No retrain history yet.")
            sys.exit(0)
        print(f"{'Date':<22} {'Mode':<8} {'MW':<5} {'Promoted':<10} {'LL Old':<10} {'LL New':<10} {'Reason'}")
        print("-" * 90)
        with open(METRICS_HISTORY) as f:
            for line in f:
                entry = json.loads(line)
                ts = entry.get("timestamp", "?")[:19]
                mode = entry.get("mode", "?")
                mw = str(entry.get("matchweek", "?"))
                promoted = "YES" if entry.get("promoted") else "NO"
                old_ll = entry.get("current_metrics") or {}
                new_ll = entry.get("new_metrics") or {}
                old = f"{old_ll.get('ensemble_log_loss', 0):.4f}" if old_ll else "N/A"
                new = f"{new_ll.get('ensemble_log_loss', 0):.4f}" if new_ll else "N/A"
                reason = entry.get("comparison", entry.get("error", "?"))[:30]
                print(f"{ts:<22} {mode:<8} {mw:<5} {promoted:<10} {old:<10} {new:<10} {reason}")
        sys.exit(0)

    if args.check:
        status = get_matchweek_status()
        if "error" in status:
            print(f"Error: {status['error']}")
            sys.exit(1)
        print(f"Completed matchweeks: {status['completed_matchweeks']} ({status['total_matches']} matches)")
        if status["matches_in_progress"]:
            print(f"MW {status['completed_matchweeks'] + 1} in progress: {status['matches_in_progress']}/{MATCHES_PER_MATCHWEEK} played")
        print(f"Last match: {status['last_match_date']} ({status['days_since_last_match']}d ago)")
        print(f"Last retrained: MW {status['last_retrained_matchweek']}")
        print(f"Needs retrain: {'YES' if status['needs_retrain'] else 'NO'}")
        sys.exit(0)

    # For forced quick/full, archive first then run directly
    if args.full or args.quick:
        _archive_current_models()
        if args.full:
            result = full_retrain(dry_run=args.dry_run)
        else:
            result = quick_retrain(dry_run=args.dry_run)
        # Also retrain auxiliary models
        log.info("Retraining auxiliary models (catboost_no_odds + xG)...")
        result["auxiliary_models"] = {
            "no_odds": retrain_no_odds(dry_run=args.dry_run),
            "xg": retrain_xg_models(dry_run=args.dry_run),
            "draw_detector": retrain_draw_detector(dry_run=args.dry_run),
        }
    else:
        # Auto mode: check matchweek completion, then quick or full based on calendar
        result = auto_retrain(dry_run=args.dry_run, force=args.force)
        if result.get("action") == "skipped":
            print(f"Skipped: {result.get('reason', 'matchweek not complete')}")
            sys.exit(0)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"RETRAIN RESULT: {'PROMOTED' if result.get('promoted') else 'NOT PROMOTED'}")
    print(f"Mode: {result.get('mode')}")
    print(f"Reason: {result.get('comparison', result.get('error', '?'))}")
    if result.get("training_time_seconds"):
        print(f"Training time: {result['training_time_seconds']:.0f}s ({result['training_time_seconds']/60:.1f}m)")
    if result.get("new_metrics"):
        m = result["new_metrics"]
        print(f"New metrics: acc={m.get('ensemble_accuracy', 0):.4f}  ll={m.get('ensemble_log_loss', 0):.4f}")
    print(f"{'=' * 60}")

    sys.exit(0 if result.get("promoted") or result.get("dry_run") else 1)


if __name__ == "__main__":
    # Need pandas for evaluate_ensemble_cv return type check
    import pandas as pd
    main()
