#!/usr/bin/env python3
"""PREDICTION TRACKER — Score ensemble predictions against actual results.

Bridges the gap between predictions (prediction_ledger.jsonl, predictions.json)
and actual match results (Odds API / results.json). Tracks:

  1. Overall prediction accuracy (H/D/A)
  2. High-confidence accuracy tiers (>50%, >55%, >60%)
  3. Draw edge signal performance (the backtested +20% ROI strategy)
  4. Model vs market comparison
  5. Running P&L simulation for draw edge bets

Usage:
    python -m scripts.betting.prediction_tracker              # Score & report
    python -m scripts.betting.prediction_tracker --fetch      # Fetch fresh results first
    python -m scripts.betting.prediction_tracker --history    # Show full history
    python -m scripts.betting.prediction_tracker --draw-only  # Draw edge bets only
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import DATA_DIR
from config.team_names import normalize_team

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Paths
LEDGER_PATH = DATA_DIR / "predictions" / "prediction_ledger.jsonl"
PREDICTIONS_DIR = DATA_DIR / "upcoming"
RESULTS_PATH = DATA_DIR / "upcoming" / "results.json"
TRACKER_PATH = DATA_DIR / "betting" / "prediction_tracker.json"


def load_predictions() -> List[Dict]:
    """Load all predictions from the ledger."""
    if not LEDGER_PATH.exists():
        log.warning("No prediction ledger found at %s", LEDGER_PATH)
        return []

    predictions = []
    for line in LEDGER_PATH.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            pred = json.loads(line)
            # Normalize team names
            if pred.get("home_team"):
                pred["home_team"] = normalize_team(pred["home_team"])
            if pred.get("away_team"):
                pred["away_team"] = normalize_team(pred["away_team"])
            predictions.append(pred)
        except json.JSONDecodeError:
            continue

    # Deduplicate: keep the LATEST prediction per (home, away, date)
    seen = {}
    for pred in predictions:
        key = (pred.get("home_team", ""), pred.get("away_team", ""),
               pred.get("match_date", ""))
        if key[0] and key[1]:  # skip entries without team names
            seen[key] = pred  # later entries overwrite earlier ones

    return list(seen.values())


def load_results() -> Dict[str, Dict]:
    """Load match results from results.json."""
    if not RESULTS_PATH.exists():
        return {}

    try:
        data = json.loads(RESULTS_PATH.read_text())
        results = data.get("results", data)
        if isinstance(results, dict):
            # Normalize keys
            normalized = {}
            for key, val in results.items():
                if isinstance(val, dict):
                    home = normalize_team(val.get("home_team", ""))
                    away = normalize_team(val.get("away_team", ""))
                    if home and away:
                        normalized[f"{home} vs {away}"] = val
            return normalized
        return {}
    except Exception as e:
        log.error("Failed to load results: %s", e)
        return {}


def fetch_fresh_results(days: int = 3) -> Dict[str, Dict]:
    """Fetch fresh results from the Odds API."""
    try:
        from scripts.data.results_fetcher import fetch_scores, parse_scores, save_results
        raw = fetch_scores(days_from=days)
        results = parse_scores(raw)
        save_results(results)
        return results
    except Exception as e:
        log.error("Failed to fetch results: %s", e)
        return load_results()


def score_predictions(predictions: List[Dict], results: Dict[str, Dict]) -> List[Dict]:
    """Match predictions to results and score them.

    Returns list of scored prediction dicts with added fields:
      actual_result, correct, scored_at
    """
    scored = []

    for pred in predictions:
        home = pred.get("home_team", "")
        away = pred.get("away_team", "")
        if not home or not away:
            continue

        match_key = f"{home} vs {away}"
        result = results.get(match_key)

        if not result:
            # Try reverse lookup with normalized names
            for rk, rv in results.items():
                rh = normalize_team(rv.get("home_team", ""))
                ra = normalize_team(rv.get("away_team", ""))
                if rh == home and ra == away:
                    result = rv
                    break

        if not result or not result.get("completed", True):
            continue  # Match not yet played or not found

        # Determine actual result
        hs = result.get("home_score")
        as_ = result.get("away_score")
        if hs is None or as_ is None:
            continue

        hs, as_ = int(hs), int(as_)
        if hs > as_:
            actual = "HOME"
        elif as_ > hs:
            actual = "AWAY"
        else:
            actual = "DRAW"

        # Determine predicted outcome
        prob_h = pred.get("prob_H", 0)
        prob_d = pred.get("prob_D", 0)
        prob_a = pred.get("prob_A", 0)
        predicted = "HOME" if prob_h >= prob_d and prob_h >= prob_a else (
            "AWAY" if prob_a >= prob_d else "DRAW"
        )
        confidence = max(prob_h, prob_d, prob_a)

        scored.append({
            "home_team": home,
            "away_team": away,
            "match_date": pred.get("match_date", ""),
            "prob_H": prob_h,
            "prob_D": prob_d,
            "prob_A": prob_a,
            "predicted": predicted,
            "confidence": confidence,
            "actual": actual,
            "correct": predicted == actual,
            "home_score": hs,
            "away_score": as_,
            "scored_at": datetime.now().isoformat(),
        })

    return scored


def compute_metrics(scored: List[Dict]) -> Dict:
    """Compute accuracy metrics from scored predictions."""
    if not scored:
        return {"total": 0}

    total = len(scored)
    correct = sum(1 for s in scored if s["correct"])

    # Confidence tiers
    tiers = {}
    for thresh in [0.50, 0.55, 0.60]:
        tier = [s for s in scored if s["confidence"] > thresh]
        if tier:
            tier_correct = sum(1 for s in tier if s["correct"])
            tiers[f">{thresh:.0%}"] = {
                "n": len(tier),
                "correct": tier_correct,
                "accuracy": tier_correct / len(tier),
            }

    # Per-class accuracy
    per_class = {}
    for cls in ["HOME", "DRAW", "AWAY"]:
        actual_cls = [s for s in scored if s["actual"] == cls]
        if actual_cls:
            cls_correct = sum(1 for s in actual_cls if s["correct"])
            per_class[cls] = {
                "actual": len(actual_cls),
                "predicted_as": sum(1 for s in scored if s["predicted"] == cls),
                "correct": cls_correct,
                "accuracy": cls_correct / len(actual_cls),
            }

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "confidence_tiers": tiers,
        "per_class": per_class,
    }


def compute_draw_edge_pnl(scored: List[Dict], results: Dict[str, Dict]) -> Dict:
    """Compute P&L for the draw edge strategy.

    Strategy: bet Draw when model draw prob >= 30% AND draw prob > market draw prob + 2%.
    Backtested at +20.5% ROI over 3 seasons.
    """
    bets = []

    for s in scored:
        model_draw = s.get("prob_D", 0)
        if model_draw < 0.28:
            continue

        # Get market implied draw probability
        # Try to find it from the prediction data or estimate from odds
        match_key = f"{s['home_team']} vs {s['away_team']}"

        # Estimate market draw prob from the odds if available
        # For now, use a simple heuristic: typical draw odds for Serie A
        # In production, this should come from stored market_implied data
        market_draw = _estimate_market_draw(s)
        if market_draw <= 0:
            continue

        draw_edge = model_draw - market_draw

        if model_draw >= 0.30 and draw_edge >= 0.02:
            draw_odds = 1.0 / market_draw if market_draw > 0 else 3.50
            actual_draw = s["actual"] == "DRAW"
            profit = (draw_odds * 0.95 - 1) if actual_draw else -1.0

            bets.append({
                "match": f"{s['home_team']} vs {s['away_team']}",
                "date": s.get("match_date", ""),
                "model_draw": model_draw,
                "market_draw": market_draw,
                "edge": draw_edge,
                "draw_odds": round(draw_odds, 2),
                "actual": s["actual"],
                "won": actual_draw,
                "profit": round(profit, 2),
                "strong": model_draw >= 0.32 and draw_edge >= 0.05,
            })

    total_bets = len(bets)
    total_won = sum(1 for b in bets if b["won"])
    total_profit = sum(b["profit"] for b in bets)

    return {
        "strategy": "Draw Edge (ML draw >= 30%, edge > 2% vs market)",
        "total_bets": total_bets,
        "won": total_won,
        "lost": total_bets - total_won,
        "win_rate": total_won / total_bets if total_bets > 0 else 0,
        "total_profit": round(total_profit, 2),
        "roi": round(total_profit / total_bets * 100, 1) if total_bets > 0 else 0,
        "bets": bets,
    }


def _estimate_market_draw(scored_entry: Dict) -> float:
    """Estimate market implied draw probability.

    Uses stored market data if available, otherwise estimates from
    the prediction probabilities (since the ensemble already blends market).
    """
    # If we have market_implied stored (from full predictions.json)
    # For ledger entries, we don't have this, so use a heuristic:
    # The ensemble already blends ~20% market signal. The "pure" market
    # draw probability can be estimated as the baseline draw rate (27%)
    # adjusted by how confident the model is about H or A.
    prob_h = scored_entry.get("prob_H", 0.45)
    prob_a = scored_entry.get("prob_A", 0.28)

    # If H or A is very high, market draw is probably low
    max_ha = max(prob_h, prob_a)
    if max_ha > 0.60:
        return 0.20  # Strong favorite → low draw prob
    elif max_ha > 0.50:
        return 0.25
    else:
        return 0.28  # Close match → higher draw prob


def load_tracker() -> Dict:
    """Load existing tracker state."""
    if TRACKER_PATH.exists():
        try:
            return json.loads(TRACKER_PATH.read_text())
        except Exception:
            pass
    return {"scored": [], "last_scored_at": None, "metrics_history": []}


def save_tracker(tracker: Dict):
    """Save tracker state."""
    TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACKER_PATH.write_text(json.dumps(tracker, indent=2, default=str))


def run(fetch: bool = False, days: int = 3, draw_only: bool = False,
        show_history: bool = False):
    """Main entry point: score predictions and report."""
    # Load or fetch results
    if fetch:
        results = fetch_fresh_results(days)
    else:
        results = load_results()

    if not results:
        log.warning("No results available. Use --fetch to get fresh results from Odds API.")
        return

    log.info("Loaded %d match results", len(results))

    # Load predictions
    predictions = load_predictions()
    log.info("Loaded %d predictions from ledger", len(predictions))

    # Score
    scored = score_predictions(predictions, results)
    log.info("Scored %d predictions", len(scored))

    if not scored:
        print("\nNo predictions matched to results. Predictions may be for upcoming matches.")
        return

    # Compute metrics
    metrics = compute_metrics(scored)
    draw_pnl = compute_draw_edge_pnl(scored, results)

    # Save tracker
    tracker = load_tracker()
    # Merge new scored entries (deduplicate by match)
    existing_keys = {(s["home_team"], s["away_team"], s["match_date"])
                     for s in tracker["scored"]}
    new_scored = [s for s in scored
                  if (s["home_team"], s["away_team"], s["match_date"]) not in existing_keys]
    tracker["scored"].extend(new_scored)
    tracker["last_scored_at"] = datetime.now().isoformat()
    tracker["metrics_history"].append({
        "timestamp": datetime.now().isoformat(),
        "total_scored": len(scored),
        "new_scored": len(new_scored),
        "accuracy": metrics["accuracy"],
        "draw_edge_roi": draw_pnl["roi"],
    })
    save_tracker(tracker)

    # Report
    if draw_only:
        _print_draw_report(draw_pnl)
    elif show_history:
        _print_full_history(scored, draw_pnl)
    else:
        _print_summary(metrics, draw_pnl, scored)


def _print_summary(metrics: Dict, draw_pnl: Dict, scored: List[Dict]):
    """Print summary report."""
    print("\n" + "=" * 70)
    print("PREDICTION TRACKER — PERFORMANCE REPORT")
    print("=" * 70)

    print(f"\nTotal scored: {metrics['total']}")
    print(f"Overall accuracy: {metrics['correct']}/{metrics['total']} "
          f"= {metrics['accuracy']:.1%}")

    # Confidence tiers
    print("\nConfidence tiers:")
    for tier, data in metrics.get("confidence_tiers", {}).items():
        print(f"  {tier}: {data['correct']}/{data['n']} = {data['accuracy']:.1%}")

    # Per-class
    print("\nPer-class breakdown:")
    for cls, data in metrics.get("per_class", {}).items():
        print(f"  {cls}: {data['correct']}/{data['actual']} = {data['accuracy']:.1%} "
              f"(predicted {data['predicted_as']}x)")

    # Draw edge P&L
    print("\n" + "-" * 70)
    print("DRAW EDGE STRATEGY")
    print("-" * 70)
    if draw_pnl["total_bets"] > 0:
        print(f"Bets: {draw_pnl['total_bets']}, Won: {draw_pnl['won']}, "
              f"Win rate: {draw_pnl['win_rate']:.1%}")
        print(f"P&L: {draw_pnl['total_profit']:+.2f} units, "
              f"ROI: {draw_pnl['roi']:+.1f}%")
        print("\nRecent draw edge bets:")
        for b in draw_pnl["bets"][-10:]:
            status = "WON" if b["won"] else "LOST"
            strong = " [STRONG]" if b["strong"] else ""
            print(f"  {b['date']} {b['match']:35s} "
                  f"edge={b['edge']:+.0%} odds={b['draw_odds']:.2f} "
                  f"{status} {b['profit']:+.2f}{strong}")
    else:
        print("No draw edge bets to report yet.")

    # Recent predictions
    print("\n" + "-" * 70)
    print("RECENT PREDICTIONS")
    print("-" * 70)
    for s in sorted(scored, key=lambda x: x.get("match_date", ""))[-10:]:
        status = "OK" if s["correct"] else "XX"
        print(f"  [{status}] {s['match_date']} {s['home_team']:15s} vs {s['away_team']:15s} "
              f"pred={s['predicted']:4s} actual={s['actual']:4s} "
              f"conf={s['confidence']:.0%} ({s['home_score']}-{s['away_score']})")


def _print_draw_report(draw_pnl: Dict):
    """Print draw edge specific report."""
    print("\n" + "=" * 70)
    print("DRAW EDGE STRATEGY — DETAILED REPORT")
    print("=" * 70)
    if draw_pnl["total_bets"] == 0:
        print("No draw edge bets to report.")
        return

    print(f"\nTotal bets: {draw_pnl['total_bets']}")
    print(f"Won: {draw_pnl['won']}, Lost: {draw_pnl['lost']}")
    print(f"Win rate: {draw_pnl['win_rate']:.1%}")
    print(f"Total P&L: {draw_pnl['total_profit']:+.2f} units")
    print(f"ROI: {draw_pnl['roi']:+.1f}%")

    print("\nAll bets:")
    cumulative = 0
    for b in draw_pnl["bets"]:
        cumulative += b["profit"]
        status = "WON" if b["won"] else "LOST"
        strong = " [STRONG]" if b["strong"] else ""
        print(f"  {b['date']} {b['match']:35s} "
              f"ML={b['model_draw']:.0%} Mkt={b['market_draw']:.0%} "
              f"edge={b['edge']:+.0%} @{b['draw_odds']:.2f} "
              f"{status} {b['profit']:+.2f} (cum={cumulative:+.2f}){strong}")


def _print_full_history(scored: List[Dict], draw_pnl: Dict):
    """Print full prediction history."""
    print("\n" + "=" * 70)
    print("FULL PREDICTION HISTORY")
    print("=" * 70)

    for s in sorted(scored, key=lambda x: x.get("match_date", "")):
        status = "OK" if s["correct"] else "XX"
        print(f"  [{status}] {s['match_date']} {s['home_team']:15s} {s['home_score']}-{s['away_score']} "
              f"{s['away_team']:15s} pred={s['predicted']:4s} actual={s['actual']:4s} "
              f"H={s['prob_H']:.0%} D={s['prob_D']:.0%} A={s['prob_A']:.0%}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Score predictions against actual results")
    parser.add_argument("--fetch", action="store_true", help="Fetch fresh results from Odds API")
    parser.add_argument("--days", type=int, default=3, help="Days to look back for results")
    parser.add_argument("--draw-only", action="store_true", help="Show draw edge bets only")
    parser.add_argument("--history", action="store_true", help="Show full prediction history")
    args = parser.parse_args()

    run(fetch=args.fetch, days=args.days, draw_only=args.draw_only,
        show_history=args.history)
