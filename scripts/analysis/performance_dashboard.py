"""Automated performance tracking and result verification dashboard.

Tracks:
- Predictions vs actual results (accuracy by confidence level)
- Betting P&L and ROI
- Feature drift detection (PPDA, injury coverage, data freshness)
- xG correlation monitoring
- Model confidence calibration

Run: python3 -m scripts.performance_dashboard
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import DATA_DIR

log = logging.getLogger(__name__)

PREDICTIONS_PATH = DATA_DIR / "upcoming" / "predictions.json"
PREDICTIONS_ARCHIVE = DATA_DIR / "upcoming" / "predictions_archive.json"
RESULTS_PATH = DATA_DIR / "upcoming" / "results.json"
BETTING_HISTORY = DATA_DIR / "betting" / "history.json"
BETTING_LOG = DATA_DIR / "betting" / "placed_bets_log.json"
BANKROLL_PATH = DATA_DIR / "betting" / "bankroll.json"
FEATURES_PATH = DATA_DIR / "features" / "features.parquet"
DASHBOARD_OUT = DATA_DIR / "performance_dashboard.json"


def _load_json(path: Path) -> list | dict:
    """Load JSON, return empty on failure."""
    if not path.exists():
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return []


def archive_predictions():
    """Archive current predictions before they get overwritten by next pipeline run.

    Call this BEFORE running the pipeline to preserve predictions for verification.
    Stores predictions keyed by match name to avoid duplicates.
    """
    preds = _load_json(PREDICTIONS_PATH)
    if not preds:
        return 0

    predictions = preds if isinstance(preds, list) else preds.get("predictions", [])
    if not predictions:
        return 0

    # Load existing archive
    archive = _load_json(PREDICTIONS_ARCHIVE)
    if not isinstance(archive, dict):
        archive = {}

    added = 0
    for pred in predictions:
        match = pred.get("match", "")
        match_date = pred.get("date", "")
        key = f"{match}_{match_date}" if match_date else match
        if key not in archive:
            archive[key] = {
                "match": match,
                "home_team": pred.get("home_team", ""),
                "away_team": pred.get("away_team", ""),
                "predicted_outcome": pred.get("predicted_outcome", ""),
                "confidence_level": pred.get("confidence_level", ""),
                "confidence": pred.get("confidence", 0),
                "home_xg": pred.get("home_xg", 0),
                "away_xg": pred.get("away_xg", 0),
                "probabilities": pred.get("probabilities", {}),
                "date": match_date,
                "archived_at": datetime.now().isoformat(),
                # Feedback loop fields (for post-settlement analysis)
                "component_predictions": pred.get("component_predictions", {}),
                "methods_used": pred.get("methods_used", []),
                "weights_applied": pred.get("weights_applied", {}),
                "intelligence_adjustments": pred.get("intelligence_adjustments", []),
                "home_factors": pred.get("home_factors", []),
                "away_factors": pred.get("away_factors", []),
                "neutral_factors": pred.get("neutral_factors", []),
                "n_factors": pred.get("n_factors", 0),
                "lineup_source": pred.get("lineup_source", ""),
            }
            added += 1

    with open(PREDICTIONS_ARCHIVE, "w") as f:
        json.dump(archive, f, indent=2, default=str)

    if added:
        log.info("Archived %d new predictions (%d total)", added, len(archive))
    return added


def check_prediction_accuracy() -> dict:
    """Compare predictions against actual results.

    Uses both current predictions.json AND the predictions archive.
    """
    preds = _load_json(PREDICTIONS_PATH)
    archive = _load_json(PREDICTIONS_ARCHIVE)
    results = _load_json(RESULTS_PATH)

    if not preds and not archive:
        return {"status": "no_data", "predictions": 0, "results": 0}
    if not results:
        return {"status": "no_results", "predictions": 0, "results": 0}

    # Merge current predictions with archive
    predictions = preds if isinstance(preds, list) else preds.get("predictions", [])
    all_predictions = {}
    # Archive first (older predictions)
    if isinstance(archive, dict):
        for key, pred in archive.items():
            match = pred.get("match", key)
            all_predictions[match] = pred
    # Current predictions override archive for same match
    for pred in predictions:
        match = pred.get("match", "")
        if match:
            all_predictions[match] = pred
    predictions = list(all_predictions.values())
    raw_results = results if isinstance(results, list) else results.get("results", [])

    # Build results lookup — handle both list and dict formats
    results_map = {}
    if isinstance(raw_results, dict):
        for key, r in raw_results.items():
            if isinstance(r, dict):
                results_map[r.get("match", key)] = r
    else:
        for r in raw_results:
            if isinstance(r, dict):
                key = r.get("match", r.get("home_team", ""))
                results_map[key] = r

    total = 0
    correct = 0
    by_confidence = {}
    match_details = []

    for pred in predictions:
        match = pred.get("match", "")
        predicted = pred.get("predicted_outcome", "")
        confidence = pred.get("confidence_level", "UNKNOWN")

        # Find matching result
        result = results_map.get(match)
        if not result:
            continue

        hs = result.get("home_score")
        as_ = result.get("away_score")
        if hs is None or as_ is None:
            continue

        try:
            hs, as_ = int(hs), int(as_)
        except (ValueError, TypeError):
            continue

        if hs > as_:
            actual = "HOME"
        elif hs < as_:
            actual = "AWAY"
        else:
            actual = "DRAW"

        total += 1
        is_correct = predicted == actual
        if is_correct:
            correct += 1

        if confidence not in by_confidence:
            by_confidence[confidence] = {"total": 0, "correct": 0}
        by_confidence[confidence]["total"] += 1
        if is_correct:
            by_confidence[confidence]["correct"] += 1

        match_details.append({
            "match": match,
            "predicted": predicted,
            "actual": actual,
            "score": f"{hs}-{as_}",
            "correct": is_correct,
            "confidence": confidence,
        })

    accuracy = correct / max(total, 1)

    conf_accuracy = {}
    for conf, data in by_confidence.items():
        conf_accuracy[conf] = {
            "total": data["total"],
            "correct": data["correct"],
            "accuracy": round(data["correct"] / max(data["total"], 1), 4),
        }

    return {
        "total_verified": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "by_confidence": conf_accuracy,
        "matches": match_details,
    }


def check_betting_performance() -> dict:
    """Analyze betting P&L from history."""
    history = _load_json(BETTING_HISTORY)
    bankroll = _load_json(BANKROLL_PATH)
    bet_log = _load_json(BETTING_LOG)

    # history.json uses "settled_bets" or "bets" key
    if isinstance(history, dict):
        bets = history.get("settled_bets", history.get("bets", []))
        pending = history.get("pending_bets", [])
        # Use pre-computed totals if available
        totals = history.get("totals", {})
    else:
        bets = history
        pending = []
        totals = {}

    if not bets and not totals:
        return {"status": "no_settled_bets", "total_bets": 0}

    # If pre-computed totals exist, use them (most accurate)
    if totals:
        result = {
            "total_bets": totals.get("total_settled", len(bets)),
            "wins": totals.get("wins", 0),
            "losses": totals.get("losses", 0),
            "pushes": totals.get("pushes", 0),
            "win_rate": round(totals.get("win_rate", 0), 4),
            "total_staked": round(totals.get("total_staked", 0), 2),
            "total_returned": round(totals.get("total_returned", 0), 2),
            "profit": round(totals.get("net_profit", 0), 2),
            "roi": round(totals.get("roi_pct", 0) / 100, 4),
            "pending_bets": totals.get("pending_count", len(pending)),
            "pending_stake": round(totals.get("pending_stake", 0), 2),
        }
    else:
        # Compute from individual bets
        total_staked = 0
        total_returned = 0
        wins = 0
        losses = 0
        pushes = 0

        for bet in bets:
            stake = bet.get("stake", 0)
            # Handle both "outcome" (uppercase) and "status" (lowercase) formats
            outcome = bet.get("outcome", bet.get("status", "")).upper()
            odds = bet.get("odds", 0)

            if isinstance(stake, (int, float)):
                total_staked += stake

            if outcome == "WON":
                wins += 1
                total_returned += stake * odds if isinstance(odds, (int, float)) else 0
            elif outcome == "LOST":
                losses += 1
            elif outcome == "PUSH":
                pushes += 1
                total_returned += stake  # refunded

        profit = total_returned - total_staked
        roi = profit / max(total_staked, 1)

        result = {
            "total_bets": len(bets),
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_rate": round(wins / max(wins + losses, 1), 4),
            "total_staked": round(total_staked, 2),
            "total_returned": round(total_returned, 2),
            "profit": round(profit, 2),
            "roi": round(roi, 4),
            "pending_bets": len(pending),
            "pending_stake": round(sum(b.get("stake", 0) for b in pending), 2),
        }

    # Bankroll
    current_bankroll = 1000
    if isinstance(bankroll, dict):
        current_bankroll = bankroll.get("current", bankroll.get("balance", 1000))
    result["current_bankroll"] = current_bankroll

    # Pending bets from log
    if bet_log and isinstance(bet_log, list):
        pending_from_log = [b for b in bet_log if b.get("status", "").lower() == "pending"]
        result["pending_from_log"] = len(pending_from_log)

    return result


def check_data_freshness() -> dict:
    """Verify all data sources are up to date."""
    checks = {}

    # Odds freshness
    odds_path = DATA_DIR / "upcoming" / "odds.json"
    if odds_path.exists():
        mtime = datetime.fromtimestamp(odds_path.stat().st_mtime)
        age_hours = (datetime.now() - mtime).total_seconds() / 3600
        checks["odds"] = {
            "last_updated": mtime.isoformat(),
            "age_hours": round(age_hours, 1),
            "fresh": age_hours < 6,
        }

    # Injuries
    injuries_dir = DATA_DIR / "external" / "injuries"
    today_file = injuries_dir / f"injuries_{date.today().isoformat()}.parquet"
    if today_file.exists():
        df = pd.read_parquet(today_file)
        checks["injuries"] = {
            "date": date.today().isoformat(),
            "count": len(df),
            "teams": int(df["team"].nunique()) if "team" in df.columns else 0,
            "fresh": True,
        }
    else:
        # Find most recent
        recent = sorted(injuries_dir.glob("injuries_*.parquet"))
        if recent:
            last = recent[-1]
            age = (date.today() - date.fromisoformat(last.stem.replace("injuries_", ""))).days
            df = pd.read_parquet(last)
            checks["injuries"] = {
                "date": last.stem.replace("injuries_", ""),
                "count": len(df),
                "teams": int(df["team"].nunique()) if "team" in df.columns else 0,
                "age_days": age,
                "fresh": age <= 3,
            }
        else:
            checks["injuries"] = {"fresh": False, "error": "no_data"}

    # Understat PPDA
    understat_dir = DATA_DIR / "external" / "understat"
    json_files = sorted(understat_dir.glob("understat_*.json"))
    if json_files:
        latest = json_files[-1]
        mtime = datetime.fromtimestamp(latest.stat().st_mtime)
        checks["understat_ppda"] = {
            "seasons": len(json_files),
            "latest_file": latest.name,
            "last_updated": mtime.isoformat(),
            "fresh": (datetime.now() - mtime).days <= 7,
        }
    else:
        checks["understat_ppda"] = {"fresh": False, "error": "no_data"}

    # Features
    if FEATURES_PATH.exists():
        mtime = datetime.fromtimestamp(FEATURES_PATH.stat().st_mtime)
        df = pd.read_parquet(FEATURES_PATH)
        checks["features"] = {
            "rows": len(df),
            "columns": len(df.columns),
            "last_rebuilt": mtime.isoformat(),
            "fresh": (datetime.now() - mtime).total_seconds() / 3600 < 24,
        }

    return checks


def check_feature_drift() -> dict:
    """Detect if key features are maintaining expected variance."""
    if not FEATURES_PATH.exists():
        return {"status": "no_features"}

    df = pd.read_parquet(FEATURES_PATH)

    drift_checks = {}

    # PPDA should have real variance
    ppda_cols = [c for c in df.columns if "ppda" in c.lower() and "mismatch" not in c]
    for col in ppda_cols:
        vals = df[col].dropna()
        nunique = vals.nunique()
        std = vals.std()
        is_constant = nunique <= 2
        drift_checks[col] = {
            "nunique": int(nunique),
            "std": round(float(std), 4),
            "healthy": not is_constant and std > 0.1,
        }

    # Interaction features — skip structurally sparse features
    # (features where >95% of values are zero are not "drifted", they're just sparse)
    interact_cols = [c for c in df.columns if "_x_" in c or "combined_disruption" in c]
    for col in interact_cols:
        vals = df[col].dropna()
        zero_pct = (vals == 0).sum() / len(vals) if len(vals) > 0 else 1
        if zero_pct > 0.95:
            continue  # Skip structurally sparse features
        drift_checks[col] = {
            "nunique": int(vals.nunique()),
            "std": round(float(vals.std()), 4),
            "healthy": vals.nunique() > 10,
        }

    healthy = sum(1 for v in drift_checks.values() if v.get("healthy"))
    total = len(drift_checks)

    return {
        "total_checked": total,
        "healthy": healthy,
        "unhealthy": total - healthy,
        "details": drift_checks,
    }


def check_confidence_calibration() -> dict:
    """Check if confidence levels correlate with accuracy."""
    preds = _load_json(PREDICTIONS_PATH)
    archive = _load_json(PREDICTIONS_ARCHIVE)
    results = _load_json(RESULTS_PATH)

    if (not preds and not archive) or not results:
        return {"status": "insufficient_data"}

    # Merge current + archived predictions
    predictions = preds if isinstance(preds, list) else preds.get("predictions", [])
    all_predictions = {}
    if isinstance(archive, dict):
        for key, pred in archive.items():
            all_predictions[pred.get("match", key)] = pred
    for pred in predictions:
        match = pred.get("match", "")
        if match:
            all_predictions[match] = pred
    predictions = list(all_predictions.values())
    raw_results = results if isinstance(results, list) else results.get("results", [])

    results_map = {}
    if isinstance(raw_results, dict):
        for key, r in raw_results.items():
            if isinstance(r, dict):
                results_map[r.get("match", key)] = r
    else:
        for r in raw_results:
            if isinstance(r, dict):
                results_map[r.get("match", "")] = r

    confidence_levels = {
        "VERY HIGH": {"expected_min": 0.55, "actual_correct": 0, "total": 0},
        "HIGH": {"expected_min": 0.48, "actual_correct": 0, "total": 0},
        "MEDIUM-HIGH": {"expected_min": 0.40, "actual_correct": 0, "total": 0},
        "MEDIUM": {"expected_min": 0.33, "actual_correct": 0, "total": 0},
        "LOW": {"expected_min": 0.25, "actual_correct": 0, "total": 0},
    }

    for pred in predictions:
        match = pred.get("match", "")
        predicted = pred.get("predicted_outcome", "")
        conf_level = pred.get("confidence_level", "UNKNOWN")
        result = results_map.get(match)

        if not result or conf_level not in confidence_levels:
            continue

        hs = result.get("home_score")
        as_ = result.get("away_score")
        if hs is None or as_ is None:
            continue

        try:
            hs, as_ = int(hs), int(as_)
        except (ValueError, TypeError):
            continue

        actual = "HOME" if hs > as_ else "AWAY" if hs < as_ else "DRAW"

        confidence_levels[conf_level]["total"] += 1
        if predicted == actual:
            confidence_levels[conf_level]["actual_correct"] += 1

    calibration = {}
    for level, data in confidence_levels.items():
        if data["total"] > 0:
            actual_rate = data["actual_correct"] / data["total"]
            calibration[level] = {
                "total": data["total"],
                "correct": data["actual_correct"],
                "actual_accuracy": round(actual_rate, 4),
                "expected_min": data["expected_min"],
                "well_calibrated": actual_rate >= data["expected_min"],
            }

    return calibration


def check_settled_bet_feedback() -> dict:
    """Analyze settled bets to surface lessons for prediction improvement.

    Compares our predicted probability vs actual outcome, flags data errors,
    and identifies systematic biases (e.g. overconfidence on certain markets).
    """
    history = _load_json(BETTING_HISTORY)
    archive = _load_json(PREDICTIONS_ARCHIVE)

    if isinstance(history, dict):
        bets = history.get("settled_bets", history.get("bets", []))
    else:
        bets = history if isinstance(history, list) else []

    if not bets:
        return {"status": "no_settled_bets"}

    # Build prediction archive lookup
    pred_map = {}
    if isinstance(archive, dict):
        for key, pred in archive.items():
            pred_map[pred.get("match", key)] = pred

    feedback = {
        "total_settled": len(bets),
        "data_errors": [],
        "probability_review": [],
        "by_market": {},
        "lessons": [],
    }

    # Market-specific odds limits (mirrors betting_engine.ODDS_SANITY)
    ODDS_MAX = {"h2h": 25.0, "totals": 6.0, "spreads": 6.0, "handicap": 6.0,
                "btts": 8.0, "corners": 10.0, "cards": 10.0}

    for bet in bets:
        match = bet.get("match", "")
        market = bet.get("market", "h2h")
        odds = bet.get("odds", 0)
        outcome = bet.get("outcome", bet.get("status", "")).upper()
        stake = bet.get("stake", 0)
        profit = bet.get("profit_loss", bet.get("profit", 0))

        # Track by market
        if market not in feedback["by_market"]:
            feedback["by_market"][market] = {
                "bets": 0, "wins": 0, "losses": 0, "pushes": 0,
                "profit": 0, "staked": 0, "data_errors": 0,
            }
        mkt = feedback["by_market"][market]
        mkt["bets"] += 1
        mkt["staked"] += stake
        mkt["profit"] += profit
        if outcome == "WON":
            mkt["wins"] += 1
        elif outcome == "PUSH":
            mkt["pushes"] += 1
        else:
            mkt["losses"] += 1

        # Flag data errors (odds exceeded market limits)
        max_odds = ODDS_MAX.get(market, 25.0)
        if odds > max_odds:
            error = {
                "match": match,
                "market": market,
                "odds": odds,
                "max_allowed": max_odds,
                "outcome": outcome,
                "profit": profit,
                "reason": f"Odds {odds} exceeded {market} max of {max_odds} — likely live/in-play data",
            }
            feedback["data_errors"].append(error)
            mkt["data_errors"] += 1

        # Compare our probability vs implied odds
        selection = bet.get("selection", "")
        if odds > 0:
            implied_prob = 1.0 / odds

            # Use the actual edge/probability stored at bet placement time.
            # value_pct is the edge % recorded by the betting engine.
            # This is more accurate than reconstructing from 1X2 probabilities,
            # which fails for DC, O/U, AH, DNB, and BTTS markets.
            value_pct = bet.get("value_pct", bet.get("edge_pct"))
            if value_pct is not None:
                edge = value_pct / 100.0
                our_prob = round(implied_prob + edge, 3)
            else:
                # Fallback: reconstruct from prediction archive (1X2 only)
                pred = pred_map.get(match, {})
                probs = pred.get("probabilities", {})
                our_prob = 0
                sel_upper = selection.upper()

                # Check compound markets first (DC contains HOME/AWAY/DRAW)
                if ("1X" in sel_upper or "HOME OR DRAW" in sel_upper) and probs:
                    our_prob = probs.get("home", 0) + probs.get("draw", 0)
                elif ("X2" in sel_upper or "DRAW OR AWAY" in sel_upper) and probs:
                    our_prob = probs.get("draw", 0) + probs.get("away", 0)
                elif "DNB" in sel_upper and probs:
                    h, a = probs.get("home", 0), probs.get("away", 0)
                    denom = h + a
                    if "HOME" in sel_upper and denom > 0:
                        our_prob = h / denom
                    elif "AWAY" in sel_upper and denom > 0:
                        our_prob = a / denom
                elif "HOME" in sel_upper and probs:
                    our_prob = probs.get("home", 0)
                elif "AWAY" in sel_upper and probs:
                    our_prob = probs.get("away", 0)
                elif "DRAW" in sel_upper and probs:
                    our_prob = probs.get("draw", 0)
                # O/U, BTTS, AH without value_pct can't be reliably reconstructed
                edge = our_prob - implied_prob if our_prob > 0 else None

            if edge is not None and (our_prob > 0 or value_pct is not None):
                feedback["probability_review"].append({
                    "match": match,
                    "selection": selection,
                    "our_prob": round(our_prob, 3) if our_prob else None,
                    "implied_prob": round(implied_prob, 3),
                    "edge_pct": round(edge * 100, 1),
                    "outcome": outcome,
                    "correct_side": outcome == "WON",
                })

    # Compute market ROI
    for mkt_data in feedback["by_market"].values():
        staked = mkt_data["staked"]
        mkt_data["roi_pct"] = round(mkt_data["profit"] / staked * 100, 1) if staked > 0 else 0

    # Generate lessons
    if feedback["data_errors"]:
        n = len(feedback["data_errors"])
        loss = sum(e["profit"] for e in feedback["data_errors"] if e["profit"] < 0)
        feedback["lessons"].append(
            f"{n} bet(s) used live/in-play odds (data errors). "
            f"Net impact: {'+'if loss >= 0 else ''}{loss:.2f}. "
            f"Fix: market-specific odds caps are now enforced in betting_engine.py."
        )

    # Check probability calibration from settled bets
    reviews = feedback["probability_review"]
    if reviews:
        correct = sum(1 for r in reviews if r["correct_side"])
        total = len(reviews)
        avg_edge = sum(r["edge_pct"] for r in reviews) / total
        feedback["lessons"].append(
            f"Probability calibration: {correct}/{total} bets won ({correct/total*100:.0f}%). "
            f"Average predicted edge: {avg_edge:+.1f}pp. "
            f"{'Overconfident' if correct/total < 0.4 and avg_edge > 5 else 'On track'}."
        )

    # Check for push handling
    pushes = sum(1 for b in bets if b.get("outcome", b.get("status", "")).upper() == "PUSH")
    if pushes:
        feedback["lessons"].append(
            f"{pushes} push(es) recorded (Asian handicap draws). "
            f"Auto-settler now handles PUSH correctly."
        )

    return feedback


def check_bankroll_health() -> dict:
    """Check bankroll management status and risk metrics."""
    try:
        from features.bankroll_manager import BankrollManager
        bm = BankrollManager(initial_bankroll=1000)
        bm.load_state()
        status = bm.get_status()

        # Add risk assessment
        drawdown = status.get("drawdown", 0)
        roi = status.get("roi", 0)
        risk_level = "LOW"
        if drawdown > 0.15:
            risk_level = "HIGH"
        elif drawdown > 0.08:
            risk_level = "MEDIUM"

        status["risk_level"] = risk_level
        status["can_bet_status"] = "OK" if status.get("can_bet") else "BLOCKED"
        return status
    except Exception as e:
        return {"status": "unavailable", "error": str(e)}


def generate_dashboard() -> dict:
    """Generate the complete performance dashboard."""
    dashboard = {
        "generated_at": datetime.now().isoformat(),
        "prediction_accuracy": check_prediction_accuracy(),
        "betting_performance": check_betting_performance(),
        "settled_bet_feedback": check_settled_bet_feedback(),
        "bankroll_health": check_bankroll_health(),
        "data_freshness": check_data_freshness(),
        "feature_drift": check_feature_drift(),
        "confidence_calibration": check_confidence_calibration(),
    }

    # Save
    with open(DASHBOARD_OUT, "w") as f:
        json.dump(dashboard, f, indent=2, default=str)

    log.info("Dashboard saved to %s", DASHBOARD_OUT)
    return dashboard


def print_dashboard(dashboard: dict):
    """Print a human-readable dashboard summary."""
    print("=" * 70)
    print(" PERFORMANCE DASHBOARD")
    print(f" Generated: {dashboard['generated_at'][:19]}")
    print("=" * 70)

    # Prediction accuracy
    pa = dashboard.get("prediction_accuracy", {})
    print(f"\n--- Prediction Accuracy ---")
    print(f"  Verified: {pa.get('total_verified', 0)} matches")
    print(f"  Correct: {pa.get('correct', 0)}")
    print(f"  Accuracy: {pa.get('accuracy', 0):.1%}")
    for conf, data in pa.get("by_confidence", {}).items():
        print(f"    {conf:15s}: {data['correct']}/{data['total']} ({data['accuracy']:.1%})")
    for m in pa.get("matches", []):
        icon = "+" if m["correct"] else "-"
        print(f"    [{icon}] {m['match']:30s} {m['score']:5s}  pred={m['predicted']:5s} actual={m['actual']:5s} ({m['confidence']})")

    # Betting
    bp = dashboard.get("betting_performance", {})
    print(f"\n--- Betting Performance ---")
    print(f"  Settled: {bp.get('total_bets', 0)} bets")
    w, l, p = bp.get('wins', 0), bp.get('losses', 0), bp.get('pushes', 0)
    print(f"  W/L/P: {w}/{l}/{p} (win rate: {bp.get('win_rate', 0):.1%})")
    print(f"  Staked: ${bp.get('total_staked', 0):.2f} | Returned: ${bp.get('total_returned', 0):.2f}")
    profit = bp.get('profit', 0)
    roi = bp.get('roi', 0)
    sign = "+" if profit >= 0 else ""
    print(f"  P&L: {sign}${profit:.2f} (ROI: {sign}{roi:.1%})")
    print(f"  Bankroll: ${bp.get('current_bankroll', 1000):.2f}")
    if bp.get("pending_bets", 0) > 0:
        print(f"  Pending: {bp.get('pending_bets', 0)} bets (${bp.get('pending_stake', 0):.2f} at risk)")

    # Settled bet feedback
    sf = dashboard.get("settled_bet_feedback", {})
    if sf and sf.get("status") != "no_settled_bets":
        print(f"\n--- Settled Bet Feedback ---")
        print(f"  Settled: {sf.get('total_settled', 0)} bets")

        # Data errors
        errors = sf.get("data_errors", [])
        if errors:
            print(f"  DATA ERRORS ({len(errors)}):")
            for e in errors:
                print(f"    ! {e['match']}: {e['market']} odds {e['odds']} "
                      f"(max {e['max_allowed']}) — {e['reason']}")

        # By market
        for mkt, data in sf.get("by_market", {}).items():
            err_flag = f" [{data['data_errors']} error(s)]" if data.get("data_errors") else ""
            push_str = f"/{data['pushes']}P" if data.get("pushes") else ""
            print(f"  {mkt:10s}: {data['bets']} bets, "
                  f"{data['wins']}W/{data['losses']}L{push_str}, "
                  f"ROI: {data['roi_pct']:+.1f}%{err_flag}")

        # Lessons
        lessons = sf.get("lessons", [])
        if lessons:
            print(f"  LESSONS:")
            for lesson in lessons:
                print(f"    > {lesson}")

    # Bankroll health
    bh = dashboard.get("bankroll_health", {})
    if bh and bh.get("status") != "unavailable":
        print(f"\n--- Bankroll Health ---")
        current = bh.get('current_bankroll', 1000)
        initial = bh.get('initial_bankroll', 1000)
        drawdown = bh.get('drawdown', 0)
        risk = bh.get('risk_level', 'UNKNOWN')
        print(f"  Balance: ${current:.2f} (from ${initial:.2f})")
        pnl = current - initial
        sign = "+" if pnl >= 0 else ""
        print(f"  P&L: {sign}${pnl:.2f} ({sign}{pnl/max(initial,1)*100:.1f}%)")
        print(f"  Drawdown: {drawdown:.1%} | Risk: {risk}")
        print(f"  Streak: {bh.get('current_streak', 0)} | "
              f"Win rate: {bh.get('win_rate', 0):.1%} ({bh.get('total_bets', 0)} bets)")
        print(f"  Status: {bh.get('can_bet_status', 'OK')}")

    # Data freshness
    df_data = dashboard.get("data_freshness", {})
    print(f"\n--- Data Freshness ---")
    for source, info in df_data.items():
        fresh = info.get("fresh", False)
        status = "FRESH" if fresh else "STALE"
        print(f"  {source:20s}: {status}")
        if source == "injuries":
            print(f"    {info.get('count', 0)} injuries, {info.get('teams', 0)} teams")
        elif source == "understat_ppda":
            print(f"    {info.get('seasons', 0)} seasons loaded")

    # Feature drift
    fd = dashboard.get("feature_drift", {})
    print(f"\n--- Feature Health ---")
    print(f"  Healthy: {fd.get('healthy', 0)}/{fd.get('total_checked', 0)}")
    unhealthy = [k for k, v in fd.get("details", {}).items() if not v.get("healthy")]
    if unhealthy:
        print(f"  Unhealthy features: {', '.join(unhealthy)}")
    else:
        print(f"  All features healthy")

    # Confidence calibration
    cc = dashboard.get("confidence_calibration", {})
    if cc and cc != {"status": "insufficient_data"}:
        print(f"\n--- Confidence Calibration ---")
        for level, data in cc.items():
            cal = "OK" if data.get("well_calibrated") else "NEEDS REVIEW"
            print(f"  {level:15s}: {data.get('actual_accuracy', 0):.1%} "
                  f"(expected >{data.get('expected_min', 0):.0%}) [{cal}]")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dashboard = generate_dashboard()
    print_dashboard(dashboard)
