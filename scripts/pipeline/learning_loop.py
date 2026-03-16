#!/usr/bin/env python3
"""LEARNING LOOP - Post-Matchday Parallel Analysis Agents

Orchestrates 3 parallel analyzers that run after results settle:
  Agent A: Prediction Auditor — per-team calibration, lineup reconciliation, confidence bands
  Agent B: Feature Drift Detector — distribution shift, importance changes, NaN coverage
  Agent C: Betting ROI Analyzer — P&L by market, Kelly effectiveness, CLV correlation

All agents read from known file paths and write structured JSON to data/feedback/.
File-based communication: no agent-to-agent chat.

Usage:
    python3 -m scripts.learning_loop
"""

import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FEEDBACK_DIR = DATA_DIR / "feedback"


class _NumpySafeEncoder(json.JSONEncoder):
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


# =============================================================================
# Agent A: Prediction Auditor
# =============================================================================

def _run_prediction_audit() -> dict:
    """Per-team calibration, lineup reconciliation, confidence band accuracy.

    Reads: predictions_archive.json, results.json, confirmed_lineups.json
    Writes: data/feedback/prediction_audit.json
    """
    from scripts.analysis.feedback_analyzer import match_predictions_to_results
    from scripts.pipeline.lesson_writer import update_lesson_effectiveness

    matched = match_predictions_to_results()
    if not matched:
        return {"status": "ok", "summary": "no settled predictions to audit"}

    # --- Close the lesson feedback loop ---
    lessons_updated = 0
    for m in matched:
        applied = m.get("lessons_applied", [])
        if applied:
            update_lesson_effectiveness(
                m["match"], m.get("predicted_outcome", ""),
                m["actual_outcome"], applied,
            )
            lessons_updated += 1

    # --- Per-team calibration (venue-aware: home + away splits) ---
    team_stats = {}  # team -> {matches, correct, ..., home: {...}, away: {...}}
    for m in matched:
        home_team = m.get("match", "").split(" vs ")[0].strip()
        away_team = m.get("match", "").split(" vs ")[-1].strip()
        actual = m["actual_outcome"]
        predicted = m.get("predicted_outcome", "").upper()

        for team, side in [(home_team, "home"), (away_team, "away")]:
            if team not in team_stats:
                team_stats[team] = {
                    "matches": 0, "correct": 0,
                    "predicted_win": 0, "actual_win": 0,
                    "xg_errors": [],
                    "home": {"matches": 0, "correct": 0, "xg_errors": []},
                    "away": {"matches": 0, "correct": 0, "xg_errors": []},
                }
            team_stats[team]["matches"] += 1
            team_stats[team][side]["matches"] += 1

            won_actual = (side == "home" and actual == "HOME") or (side == "away" and actual == "AWAY")
            predicted_win = (side == "home" and predicted == "HOME") or (side == "away" and predicted == "AWAY")

            if won_actual:
                team_stats[team]["actual_win"] += 1
            if predicted_win:
                team_stats[team]["predicted_win"] += 1
            if predicted == actual:
                team_stats[team]["correct"] += 1
                team_stats[team][side]["correct"] += 1

            # xG error
            xg_key = f"{side}_xg"
            score_key = f"{side}_score"
            pred_xg = m.get(xg_key, 0)
            actual_goals = m.get(score_key, 0)
            if pred_xg > 0:
                xg_err = pred_xg - actual_goals
                team_stats[team]["xg_errors"].append(xg_err)
                team_stats[team][side]["xg_errors"].append(xg_err)

    per_team_calibration = {}
    for team, stats in team_stats.items():
        n = stats["matches"]
        if n < 2:
            continue
        accuracy = stats["correct"] / n if n > 0 else 0
        win_rate_predicted = stats["predicted_win"] / n if n > 0 else 0
        win_rate_actual = stats["actual_win"] / n if n > 0 else 0
        xg_errors = stats["xg_errors"]
        mae = sum(abs(e) for e in xg_errors) / len(xg_errors) if xg_errors else 0
        mean_bias = sum(xg_errors) / len(xg_errors) if xg_errors else 0

        entry = {
            "matches": n,
            "accuracy": round(accuracy, 3),
            "win_rate_predicted": round(win_rate_predicted, 3),
            "win_rate_actual": round(win_rate_actual, 3),
            "win_rate_bias": round(win_rate_predicted - win_rate_actual, 3),
            "xg_mae": round(mae, 3),
            "xg_mean_bias": round(mean_bias, 3),
        }

        # Venue splits
        for venue in ("home", "away"):
            v = stats[venue]
            vn = v["matches"]
            if vn > 0:
                v_errs = v["xg_errors"]
                entry[f"{venue}_matches"] = vn
                entry[f"{venue}_accuracy"] = round(v["correct"] / vn, 3)
                entry[f"{venue}_xg_bias"] = round(
                    sum(v_errs) / len(v_errs), 3
                ) if v_errs else 0.0

        per_team_calibration[team] = entry

    # --- Lineup reconciliation (compare xG accuracy: confirmed vs predicted) ---
    lineup_stats = {
        "predicted": {"count": 0, "xg_errors": []},
        "confirmed": {"count": 0, "xg_errors": []},
    }
    for m in matched:
        src = m.get("lineup_source", "predicted")
        bucket = "confirmed" if src == "confirmed" else "predicted"
        lineup_stats[bucket]["count"] += 1
        # Collect total xG error for this match
        for side in ("home", "away"):
            pred_xg = m.get(f"{side}_xg", 0)
            actual_goals = m.get(f"{side}_score", 0)
            if pred_xg > 0:
                lineup_stats[bucket]["xg_errors"].append(abs(pred_xg - actual_goals))

    # Compute MAE per lineup source
    for bucket in ("predicted", "confirmed"):
        errs = lineup_stats[bucket]["xg_errors"]
        lineup_stats[bucket]["xg_mae"] = round(
            sum(errs) / len(errs), 3
        ) if errs else None
        # Remove raw list for cleaner JSON
        del lineup_stats[bucket]["xg_errors"]

    # --- Confidence band accuracy ---
    bands = {}
    for m in matched:
        level = m.get("confidence_level", "UNKNOWN")
        if level not in bands:
            bands[level] = {"total": 0, "correct": 0}
        bands[level]["total"] += 1
        if m.get("predicted_outcome", "").upper() == m["actual_outcome"]:
            bands[level]["correct"] += 1

    confidence_bands = {}
    for level, data in bands.items():
        n = data["total"]
        confidence_bands[level] = {
            "total": n,
            "correct": data["correct"],
            "hit_rate": round(data["correct"] / n, 3) if n > 0 else 0,
        }

    audit = {
        "generated_at": datetime.now().isoformat(),
        "n_settled": len(matched),
        "per_team_calibration": per_team_calibration,
        "lineup_reconciliation": lineup_stats,
        "confidence_bands": confidence_bands,
    }

    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_DIR / "prediction_audit.json", "w") as f:
        json.dump(audit, f, indent=2, cls=_NumpySafeEncoder)

    # Find most biased teams
    biased = sorted(
        per_team_calibration.items(),
        key=lambda x: abs(x[1]["xg_mean_bias"]),
        reverse=True,
    )[:3]
    bias_summary = ", ".join(f"{t}({d['xg_mean_bias']:+.2f})" for t, d in biased) if biased else "none"

    return {
        "status": "ok",
        "summary": f"{len(matched)} settled, {len(per_team_calibration)} teams audited, top bias: {bias_summary}",
    }


# =============================================================================
# Agent B: Feature Drift Detector
# =============================================================================

def _run_drift_detection() -> dict:
    """Detect distribution shift in ML features, NaN coverage, importance changes.

    Reads: features.parquet, extended_model_metadata.json
    Writes: data/feedback/drift_report.json
    """
    import numpy as np
    import pandas as pd

    feature_path = DATA_DIR / "features" / "features.parquet"
    meta_path = DATA_DIR / "models" / "universal" / "extended_model_metadata.json"

    if not feature_path.exists():
        return {"status": "ok", "summary": "no features.parquet found"}

    df = pd.read_parquet(feature_path)
    if "season" not in df.columns:
        return {"status": "ok", "summary": "no season column in features"}

    # Load model metadata for feature list
    model_features = []
    top_xg_features = []
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        model_features = meta.get("feature_names", [])
        top_xg_features = meta.get("top_xg_features", [])

    # Use model features if available, else discover numeric columns
    if model_features:
        check_features = [c for c in model_features if c in df.columns]
    else:
        check_features = [c for c in df.select_dtypes(include=[np.number]).columns
                          if c not in ("home_score", "away_score")]

    if not check_features:
        return {"status": "ok", "summary": "no features to check"}

    # Split into recent (last 2 seasons) vs historical
    seasons = sorted(df["season"].unique())
    if len(seasons) < 3:
        return {"status": "ok", "summary": "not enough seasons for drift detection"}

    recent_seasons = seasons[-2:]
    historical_seasons = seasons[:-2]

    df_recent = df[df["season"].isin(recent_seasons)]
    df_hist = df[df["season"].isin(historical_seasons)]

    # KS test for each feature
    from scipy.stats import ks_2samp

    drift_results = {}
    drifted_count = 0
    high_nan_count = 0

    for feat in check_features:
        recent_vals = df_recent[feat].dropna()
        hist_vals = df_hist[feat].dropna()

        # NaN coverage in recent data
        recent_nan_pct = df_recent[feat].isna().mean()

        result = {
            "recent_nan_pct": round(float(recent_nan_pct), 3),
            "high_nan": bool(recent_nan_pct > 0.5),
        }

        if len(recent_vals) > 10 and len(hist_vals) > 10:
            stat, p_value = ks_2samp(recent_vals, hist_vals)
            result["ks_statistic"] = round(float(stat), 4)
            result["ks_p_value"] = round(float(p_value), 4)
            result["drifted"] = bool(p_value < 0.05)
            result["recent_mean"] = round(float(recent_vals.mean()), 4)
            result["hist_mean"] = round(float(hist_vals.mean()), 4)
            result["mean_shift"] = round(float(recent_vals.mean() - hist_vals.mean()), 4)

            if result["drifted"]:
                drifted_count += 1
        else:
            result["ks_statistic"] = None
            result["ks_p_value"] = None
            result["drifted"] = False

        if result["high_nan"]:
            high_nan_count += 1

        drift_results[feat] = result

    report = {
        "generated_at": datetime.now().isoformat(),
        "total_features_checked": len(check_features),
        "drifted_count": drifted_count,
        "high_nan_count": high_nan_count,
        "recent_seasons": recent_seasons,
        "historical_seasons": historical_seasons,
        "top_xg_features": top_xg_features,
        "feature_drift": drift_results,
    }

    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_DIR / "drift_report.json", "w") as f:
        json.dump(report, f, indent=2, cls=_NumpySafeEncoder)

    return {
        "status": "ok",
        "summary": f"{len(check_features)} features checked, {drifted_count} drifted, {high_nan_count} high-NaN",
    }


# =============================================================================
# Agent C: Betting ROI Analyzer
# =============================================================================

def _run_roi_analysis() -> dict:
    """P&L by market, Kelly effectiveness, CLV correlation, streak analysis.

    Reads: history.json, clv_history.json, bankroll state.json
    Writes: data/feedback/roi_report.json
    """
    history_path = DATA_DIR / "betting" / "history.json"
    clv_path = DATA_DIR / "betting" / "clv_history.json"
    bankroll_path = DATA_DIR / "bankroll" / "state.json"

    # Load betting history
    history = []
    if history_path.exists():
        try:
            with open(history_path) as f:
                raw = json.load(f)
            if isinstance(raw, list):
                history = raw
            elif isinstance(raw, dict):
                history = raw.get("settled_bets", raw.get("bets", []))
        except Exception as e:
            log.warning(f"Failed to load bet history for learning loop: {e}")

    settled = [b for b in history if isinstance(b, dict) and b.get("status") in ("won", "lost")]
    if not settled:
        return {"status": "ok", "summary": "no settled bets to analyze"}

    # --- P&L by market type ---
    by_market = {}
    total_profit = 0.0
    total_staked = 0.0

    for bet in settled:
        market = bet.get("market", "unknown")
        stake = bet.get("stake", 0)
        odds = bet.get("odds", 0)
        status = bet.get("status", "")

        profit = (odds - 1) * stake if status == "won" else -stake

        if market not in by_market:
            by_market[market] = {"bets": 0, "won": 0, "staked": 0.0, "profit": 0.0}
        by_market[market]["bets"] += 1
        by_market[market]["staked"] += stake
        by_market[market]["profit"] += profit
        if status == "won":
            by_market[market]["won"] += 1

        total_profit += profit
        total_staked += stake

    market_roi = {}
    for market, data in by_market.items():
        market_roi[market] = {
            "bets": data["bets"],
            "won": data["won"],
            "win_rate": round(data["won"] / data["bets"], 3) if data["bets"] > 0 else 0,
            "staked": round(data["staked"], 2),
            "profit": round(data["profit"], 2),
            "roi_pct": round(data["profit"] / data["staked"] * 100, 2) if data["staked"] > 0 else 0,
        }

    # --- Kelly effectiveness ---
    kelly_stats = {"total_bets": len(settled), "oversized": 0, "undersized": 0}
    for bet in settled:
        kelly_frac = bet.get("kelly_fraction", 0)
        actual_frac = bet.get("stake", 0) / 1000  # approximate
        if kelly_frac > 0 and actual_frac > 0:
            ratio = actual_frac / kelly_frac
            if ratio > 1.5:
                kelly_stats["oversized"] += 1
            elif ratio < 0.5:
                kelly_stats["undersized"] += 1

    # --- CLV correlation ---
    clv_data = {}
    if clv_path.exists():
        try:
            with open(clv_path) as f:
                clv_data = json.load(f)
        except Exception as e:
            log.warning(f"Failed to load CLV data for learning loop: {e}")

    clv_entries = clv_data.get("bets", clv_data.get("entries", []))
    if isinstance(clv_entries, dict):
        clv_entries = list(clv_entries.values())

    positive_clv_wins = 0
    positive_clv_total = 0
    negative_clv_wins = 0
    negative_clv_total = 0

    for entry in clv_entries:
        if not isinstance(entry, dict):
            continue
        clv = entry.get("clv_pct", entry.get("clv", 0))
        won = entry.get("status") == "won" or entry.get("won", False)
        if clv > 0:
            positive_clv_total += 1
            if won:
                positive_clv_wins += 1
        elif clv < 0:
            negative_clv_total += 1
            if won:
                negative_clv_wins += 1

    clv_analysis = {
        "positive_clv_bets": positive_clv_total,
        "positive_clv_win_rate": round(positive_clv_wins / positive_clv_total, 3) if positive_clv_total > 0 else 0,
        "negative_clv_bets": negative_clv_total,
        "negative_clv_win_rate": round(negative_clv_wins / negative_clv_total, 3) if negative_clv_total > 0 else 0,
    }

    # --- Streak analysis ---
    streaks = {"current": 0, "current_type": None, "max_win": 0, "max_loss": 0}
    win_streak = 0
    loss_streak = 0
    for bet in settled:
        if bet.get("status") == "won":
            win_streak += 1
            loss_streak = 0
            streaks["max_win"] = max(streaks["max_win"], win_streak)
        else:
            loss_streak += 1
            win_streak = 0
            streaks["max_loss"] = max(streaks["max_loss"], loss_streak)

    streaks["current"] = win_streak if win_streak > 0 else -loss_streak
    streaks["current_type"] = "win" if win_streak > 0 else ("loss" if loss_streak > 0 else "none")

    # --- Bankroll state ---
    bankroll_info = {}
    if bankroll_path.exists():
        try:
            with open(bankroll_path) as f:
                bankroll_info = json.load(f)
        except Exception as e:
            log.warning(f"Failed to load bankroll state for learning loop: {e}")

    roi_report = {
        "generated_at": datetime.now().isoformat(),
        "total_settled": len(settled),
        "total_staked": round(total_staked, 2),
        "total_profit": round(total_profit, 2),
        "overall_roi_pct": round(total_profit / total_staked * 100, 2) if total_staked > 0 else 0,
        "by_market": market_roi,
        "kelly_effectiveness": kelly_stats,
        "clv_analysis": clv_analysis,
        "streaks": streaks,
        "bankroll": {
            "current": bankroll_info.get("balance", bankroll_info.get("current_balance", 0)),
            "initial": bankroll_info.get("initial_balance", 1000),
            "max_drawdown": bankroll_info.get("max_drawdown", 0),
        },
    }

    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_DIR / "roi_report.json", "w") as f:
        json.dump(roi_report, f, indent=2, cls=_NumpySafeEncoder)

    # Generate per-market stake multipliers for bankroll manager
    # Markets with strong negative ROI get reduced; positive ROI get a small boost
    market_adjustments = {}
    for market, data in market_roi.items():
        n = data.get("bets", 0)
        roi = data.get("roi_pct", 0)
        if n < 3:
            market_adjustments[market] = 1.0  # not enough data
            continue
        if roi < -20:
            market_adjustments[market] = 0.5  # halve stake
        elif roi < -10:
            market_adjustments[market] = 0.75
        elif roi > 10:
            market_adjustments[market] = min(1.2, 1.0 + roi / 200)  # modest boost, capped at 1.2
        else:
            market_adjustments[market] = 1.0

    adjustments_output = {
        "generated_at": datetime.now().isoformat(),
        "market_multipliers": market_adjustments,
        "source": "roi_analysis",
        "min_bets_for_signal": 3,
    }
    with open(FEEDBACK_DIR / "market_adjustments.json", "w") as f:
        json.dump(adjustments_output, f, indent=2)

    return {
        "status": "ok",
        "summary": f"{len(settled)} bets, ROI {roi_report['overall_roi_pct']:+.1f}%, P&L ${total_profit:+.2f}",
    }


# =============================================================================
# Orchestrator
# =============================================================================

def run_learning_loop() -> dict:
    """Run all 3 analysis agents in parallel. Returns per-agent status."""
    log.info("=" * 60)
    log.info("LEARNING LOOP - 3 Parallel Analysis Agents")
    log.info("=" * 60)

    agents = {
        "prediction_audit": _run_prediction_audit,
        "drift_detection": _run_drift_detection,
        "roi_analysis": _run_roi_analysis,
    }

    results = {}

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_name = {
            executor.submit(fn): name for name, fn in agents.items()
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                results[name] = future.result()
                log.info(f"  {name}: {results[name].get('summary', 'done')}")
            except Exception as e:
                results[name] = {"status": "error", "error": str(e)}
                log.warning(f"  {name} failed: {e}")

    # Run lesson writer after audit + ROI complete (needs their outputs)
    try:
        from scripts.pipeline.lesson_writer import generate_lessons
        lesson_result = generate_lessons()
        results["lesson_writer"] = lesson_result
        log.info(f"  lesson_writer: {lesson_result.get('summary', 'done')}")
    except Exception as e:
        results["lesson_writer"] = {"status": "error", "error": str(e)}
        log.warning(f"  lesson_writer failed: {e}")

    log.info("Learning loop complete.")
    return results


if __name__ == "__main__":
    results = run_learning_loop()
    print(json.dumps(results, indent=2))
