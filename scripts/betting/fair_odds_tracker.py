"""Fair Odds Historical Tracking System.

Records model fair odds (1/probability) before each match and tracks accuracy
against actual outcomes. Builds a persistent ledger for long-term calibration
and model quality monitoring.

Two main operations:
1. record_predictions() — called before matches, saves model probabilities
2. settle_predictions() — called after matches, records actual outcomes

The ledger lives at data/betting/fair_odds_ledger.json.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent.parent
BETTING_DIR = BASE / "data" / "betting"
LEDGER_PATH = BETTING_DIR / "fair_odds_ledger.json"
SUMMARY_PATH = BETTING_DIR / "fair_odds_summary.json"


def record_predictions(predictions: List[Dict]) -> Dict:
    """Record model fair odds for upcoming matches.

    Args:
        predictions: List of dicts with keys:
            match, home_team, away_team, date, commence_time,
            probabilities (home/draw/away), predicted_outcome,
            confidence_level, market_odds (h2h home/draw/away).

    Returns:
        dict with count of new predictions recorded.
    """
    ledger = _load_ledger()
    existing_keys = {
        (r["match"], r.get("prediction_date", r.get("date", "")))
        for r in ledger
    }

    new_count = 0
    for p in predictions:
        match = p.get("match") or f"{p.get('home_team', '')} vs {p.get('away_team', '')}"
        date = p.get("date") or (
            p["commence_time"][:10] if p.get("commence_time") else ""
        )
        if (match, date) in existing_keys:
            continue

        probs = p.get("probabilities", {})
        h2h = (p.get("odds", {}) or {}).get("h2h", {})

        # Fair odds = 1/probability (no margin)
        fair_home = round(1 / probs["home"], 3) if probs.get("home", 0) > 0.01 else None
        fair_draw = round(1 / probs["draw"], 3) if probs.get("draw", 0) > 0.01 else None
        fair_away = round(1 / probs["away"], 3) if probs.get("away", 0) > 0.01 else None

        record = {
            "match": match,
            "home_team": p.get("home_team", ""),
            "away_team": p.get("away_team", ""),
            "date": date,
            "prediction_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "commence_time": p.get("commence_time", ""),

            # Model probabilities
            "prob_home": round(probs.get("home", 0), 4),
            "prob_draw": round(probs.get("draw", 0), 4),
            "prob_away": round(probs.get("away", 0), 4),

            # Fair odds (1/prob)
            "fair_home": fair_home,
            "fair_draw": fair_draw,
            "fair_away": fair_away,

            # Market odds for comparison
            "market_home": h2h.get("home"),
            "market_draw": h2h.get("draw"),
            "market_away": h2h.get("away"),

            # Prediction metadata
            "predicted_outcome": p.get("predicted_outcome", ""),
            "confidence_level": p.get("confidence_level", ""),

            # Outcome (filled later by settle_predictions)
            "actual_outcome": None,
            "actual_score": None,
            "settled": False,
        }
        ledger.append(record)
        existing_keys.add((match, date))
        new_count += 1

    if new_count:
        _save_ledger(ledger)
        log.info("Recorded %d new fair odds predictions", new_count)

    return {"recorded": new_count}


def settle_predictions(results: Dict[str, Dict] = None) -> Dict:
    """Settle prediction records with actual outcomes.

    Args:
        results: Optional dict of {match_key: {score: [h, a], ...}}.
            If None, reads from the latest results file.

    Returns:
        dict with count settled and accuracy stats.
    """
    ledger = _load_ledger()
    unsettled = [r for r in ledger if not r.get("settled")]
    if not unsettled:
        return {"settled": 0}

    # Load results if not provided
    if results is None:
        results = _load_results()

    settled_count = 0
    for record in unsettled:
        match = record["match"]
        r = results.get(match)
        if not r:
            continue

        score = r.get("score") or r.get("final_score")
        if not score or len(score) < 2:
            continue

        # Determine actual outcome
        h, a = score[0], score[1]
        if h > a:
            actual = "HOME"
        elif a > h:
            actual = "AWAY"
        else:
            actual = "DRAW"

        record["actual_outcome"] = actual
        record["actual_score"] = score
        record["settled"] = True
        record["settled_at"] = datetime.now(timezone.utc).isoformat()
        record["prediction_correct"] = record["predicted_outcome"] == actual
        settled_count += 1

    if settled_count:
        _save_ledger(ledger)
        _update_summary(ledger)
        log.info("Settled %d fair odds records", settled_count)

    return {"settled": settled_count}


def get_summary() -> Dict:
    """Get the fair odds tracking summary."""
    if SUMMARY_PATH.exists():
        with open(SUMMARY_PATH) as f:
            return json.load(f)
    return {}


def _update_summary(ledger: List[Dict]):
    """Update the aggregated summary from the full ledger."""
    settled = [r for r in ledger if r.get("settled")]
    if not settled:
        return

    total = len(settled)
    correct = len([r for r in settled if r.get("prediction_correct")])

    # Calibration by probability bucket
    # For each outcome (H/D/A), bucket predictions by their probability
    calibration = {}
    for side, prob_key, outcome in [
        ("home", "prob_home", "HOME"),
        ("draw", "prob_draw", "DRAW"),
        ("away", "prob_away", "AWAY"),
    ]:
        buckets = {}
        for r in settled:
            prob = r.get(prob_key, 0)
            bucket = f"{int(prob * 10) * 10}-{int(prob * 10) * 10 + 10}%"
            if bucket not in buckets:
                buckets[bucket] = {"count": 0, "actual": 0, "avg_prob": 0}
            buckets[bucket]["count"] += 1
            buckets[bucket]["avg_prob"] += prob
            if r.get("actual_outcome") == outcome:
                buckets[bucket]["actual"] += 1

        for b, stats in buckets.items():
            stats["avg_prob"] = round(stats["avg_prob"] / stats["count"] * 100, 1)
            stats["actual_rate"] = round(stats["actual"] / stats["count"] * 100, 1)
            stats["cal_error"] = round(stats["actual_rate"] - stats["avg_prob"], 1)

        calibration[side] = buckets

    # Model vs market accuracy
    model_correct = len([r for r in settled if r.get("prediction_correct")])
    market_correct = 0
    for r in settled:
        mh, md, ma = r.get("market_home"), r.get("market_draw"), r.get("market_away")
        if mh and md and ma:
            # Market favorite = lowest odds
            min_odds = min(mh, md, ma)
            if min_odds == mh:
                market_pred = "HOME"
            elif min_odds == ma:
                market_pred = "AWAY"
            else:
                market_pred = "DRAW"
            if market_pred == r.get("actual_outcome"):
                market_correct += 1

    matches_with_market = len([
        r for r in settled
        if r.get("market_home") and r.get("market_draw") and r.get("market_away")
    ])

    # Fair odds value tracking: if model said fair odds < market odds (value),
    # did that outcome hit more often?
    value_bets = {"total": 0, "correct": 0, "avg_edge": 0}
    for r in settled:
        # Check if predicted outcome had value (fair odds < market odds)
        pred = r.get("predicted_outcome")
        if pred == "HOME":
            fair, market = r.get("fair_home"), r.get("market_home")
        elif pred == "AWAY":
            fair, market = r.get("fair_away"), r.get("market_away")
        elif pred == "DRAW":
            fair, market = r.get("fair_draw"), r.get("market_draw")
        else:
            continue
        if fair and market and fair < market:
            value_bets["total"] += 1
            edge = (market / fair - 1) * 100
            value_bets["avg_edge"] += edge
            if r.get("prediction_correct"):
                value_bets["correct"] += 1

    if value_bets["total"] > 0:
        value_bets["hit_rate"] = round(
            value_bets["correct"] / value_bets["total"] * 100, 1
        )
        value_bets["avg_edge"] = round(
            value_bets["avg_edge"] / value_bets["total"], 1
        )
    else:
        value_bets["hit_rate"] = 0
        value_bets["avg_edge"] = 0

    # Recent form (last 20 predictions)
    recent = sorted(settled, key=lambda r: r.get("settled_at", ""), reverse=True)[:20]
    recent_correct = len([r for r in recent if r.get("prediction_correct")])

    summary = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_predictions": total,
        "total_correct": correct,
        "accuracy_pct": round(correct / total * 100, 1),
        "model_accuracy": round(model_correct / total * 100, 1),
        "market_accuracy": round(
            market_correct / matches_with_market * 100, 1
        ) if matches_with_market > 0 else None,
        "matches_with_market_odds": matches_with_market,
        "recent_20_accuracy": round(recent_correct / len(recent) * 100, 1) if recent else 0,
        "value_bets": value_bets,
        "calibration": calibration,
        "unsettled_count": len([r for r in ledger if not r.get("settled")]),
    }

    BETTING_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)


def _load_results() -> Dict:
    """Load results from the bet journal or results files."""
    results = {}

    # Try bet journal first (has settled bet results)
    journal_path = BETTING_DIR / "bet_journal.json"
    if journal_path.exists():
        try:
            with open(journal_path) as f:
                journal = json.load(f)
            for entry in journal:
                if entry.get("result") and entry.get("match"):
                    mk = entry["match"]
                    if mk not in results:
                        results[mk] = {}
                    score = entry.get("actual_score") or entry.get("score")
                    if score:
                        results[mk]["score"] = score
        except (json.JSONDecodeError, IOError):
            pass

    # Also try live match data
    live_dir = BASE / "data" / "live"
    if live_dir.exists():
        for fpath in sorted(live_dir.glob("20*.json"), reverse=True)[:7]:
            try:
                with open(fpath) as f:
                    data = json.load(f)
                for mk, mdata in data.get("matches", {}).items():
                    if mdata.get("status") == "completed":
                        score = mdata.get("final_score") or mdata.get("score")
                        if score and mk not in results:
                            results[mk] = {"score": score}
            except (json.JSONDecodeError, IOError):
                pass

    return results


def _load_ledger() -> List[Dict]:
    if LEDGER_PATH.exists():
        try:
            with open(LEDGER_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def _save_ledger(ledger: List[Dict]):
    BETTING_DIR.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_PATH, "w") as f:
        json.dump(ledger, f, indent=2)
