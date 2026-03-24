"""Continuous Odds Edge Monitor — discovers value bets between pipeline runs.

Three capabilities:
  1. Continuous polling: Fetches odds every 15-30 min on match days
  2. Edge alerting: Notifies when a match crosses the value threshold
  3. Watchlist: Tracks matches approaching value (3-5% edge, below threshold)
  4. Live in-play value: Detects mid-match odds drift creating new edges

Designed to run as a lightweight daemon between full pipeline runs.
API cost: ~3 credits per poll (fetches h2h, totals, spreads for all Serie A).
Budget: ~2,000 credits/month at 15-min polling on match days.

Usage:
    # Single scan (for testing or manual trigger)
    python3 -m scripts.betting.odds_edge_monitor --once

    # Continuous polling (runs until killed)
    python3 -m scripts.betting.odds_edge_monitor --daemon --interval 900

    # Dry run (analyze existing data, no API calls)
    python3 -m scripts.betting.odds_edge_monitor --dry-run
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import DATA_DIR

log = logging.getLogger(__name__)

UPCOMING_DIR = DATA_DIR / "upcoming"
BETTING_DIR = DATA_DIR / "betting"
SNAPSHOTS_DIR = DATA_DIR / "odds_snapshots"

# Edge thresholds (must match betting_unified.py BettingConfig)
EDGE_THRESHOLDS = {
    "1X2_Draw": {"min": 5.0, "max": 12.0},
    "1X2_Away": {"min": 7.0, "max": 10.0},
    "O/U_Over": {"min": 6.0, "max": 8.0},
    "DC":       {"min": 5.0, "max": 8.0},
}

# Watchlist: alert when edge is within this many pp of threshold
WATCHLIST_PROXIMITY_PP = 2.0  # e.g., alert at 3% when threshold is 5%

# Minimum odds movement to consider significant
MIN_ODDS_MOVE = 0.05

# State file to track what we've already alerted on
STATE_PATH = BETTING_DIR / "edge_monitor_state.json"


def _load_predictions() -> Dict[str, Dict]:
    """Load current model predictions keyed by match."""
    pred_path = UPCOMING_DIR / "predictions.json"
    if not pred_path.exists():
        return {}

    data = json.load(open(pred_path))
    preds = data.get("predictions", data) if isinstance(data, dict) else data
    if isinstance(preds, list):
        return {p.get("match", ""): p for p in preds if p.get("match")}
    return {}


def _load_state() -> Dict:
    """Load monitor state (previously alerted edges, watchlist)."""
    if STATE_PATH.exists():
        try:
            return json.load(open(STATE_PATH))
        except Exception:
            pass
    return {"alerted": {}, "watchlist": {}, "last_poll": None}


def _save_state(state: Dict):
    """Save monitor state."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)


def _remove_overround(odds_list: List[float]) -> List[float]:
    """Remove bookmaker margin from odds to get true probabilities."""
    implied = [1.0 / o for o in odds_list if o > 1.0]
    total = sum(implied)
    if total <= 0:
        return implied
    return [p / total for p in implied]


def _get_pinnacle_odds(bookmakers: list, outcome: str) -> float:
    """Extract Pinnacle odds for a specific outcome."""
    for bm in bookmakers:
        if bm.get("bookmaker", "").lower() in ("pinnacle", "pinnaclesports"):
            return bm.get(outcome, 0)
    return 0


def scan_for_edges(predictions: Dict, odds_data: Dict) -> Dict:
    """Scan current odds against model predictions to find edges.

    Returns:
        {
            "value_bets": [{"match", "market", "selection", "edge_pct", "best_odds", "bookmaker", ...}],
            "watchlist": [{"match", "market", "selection", "edge_pct", "gap_to_threshold", ...}],
            "movements": [{"match", "selection", "old_odds", "new_odds", "direction", ...}],
        }
    """
    value_bets = []
    watchlist = []
    movements = []

    matches = odds_data.get("matches", odds_data)
    if not isinstance(matches, dict):
        return {"value_bets": [], "watchlist": [], "movements": []}

    for match_key, match_odds in matches.items():
        pred = predictions.get(match_key, {})
        if not pred:
            continue

        # Get model probabilities
        probs = pred.get("probabilities", pred.get("betting_probabilities", {}))
        if isinstance(probs, dict):
            model_h = probs.get("home", 0)
            model_d = probs.get("draw", 0)
            model_a = probs.get("away", 0)
        else:
            model_h = pred.get("prob_H", 0)
            model_d = pred.get("prob_D", 0)
            model_a = pred.get("prob_A", 0)

        if model_h + model_d + model_a < 0.5:
            continue

        # Get h2h odds — handle both dict and bookmaker-list formats
        h2h = match_odds.get("h2h", {})
        bookmakers_list = []
        if isinstance(h2h, dict):
            best_h = h2h.get("best_home", h2h.get("home", 0))
            best_d = h2h.get("best_draw", h2h.get("draw", 0))
            best_a = h2h.get("best_away", h2h.get("away", 0))
            bookmakers_list = h2h.get("all_bookmakers", [])
        elif isinstance(h2h, list):
            bookmakers_list = h2h
            best_h = max((bm.get("home", 0) for bm in h2h), default=0)
            best_d = max((bm.get("draw", 0) for bm in h2h), default=0)
            best_a = max((bm.get("away", 0) for bm in h2h), default=0)
        else:
            continue

        # Get Pinnacle for sharp benchmark
        pin_h = pin_d = pin_a = 0
        for bm in bookmakers_list:
            if bm.get("bookmaker", "").lower() in ("pinnacle", "pinnaclesports"):
                pin_h = bm.get("home", 0)
                pin_d = bm.get("draw", 0)
                pin_a = bm.get("away", 0)
                break

        # Remove overround using all 3 Pinnacle outcomes together
        pin_probs = {}
        if pin_h > 1.0 and pin_d > 1.0 and pin_a > 1.0:
            raw_probs = [1.0/pin_h, 1.0/pin_d, 1.0/pin_a]
            total = sum(raw_probs)
            pin_probs = {
                "home": raw_probs[0] / total,
                "draw": raw_probs[1] / total,
                "away": raw_probs[2] / total,
            }

        # Check each selection
        for selection, model_p, best_odds, pin_odds, market_cat in [
            ("Draw", model_d, best_d, pin_d, "1X2_Draw"),
            ("Away", model_a, best_a, pin_a, "1X2_Away"),
        ]:
            if best_odds <= 1.0 or model_p <= 0:
                continue

            # Use overround-removed Pinnacle prob if available
            sharp_implied = pin_probs.get(selection.lower(), 0)
            if sharp_implied <= 0:
                sharp_implied = 1.0 / best_odds  # Fallback

            edge_pct = ((model_p - sharp_implied) / sharp_implied * 100) if sharp_implied > 0 else 0
            thresholds = EDGE_THRESHOLDS.get(market_cat, {"min": 5.0, "max": 8.0})

            # Find best bookmaker
            best_bm = ""
            for bm in bookmakers_list:
                outcome_key = selection.lower()
                if bm.get(outcome_key, 0) == best_odds:
                    best_bm = bm.get("bookmaker", "")
                    break

            bet_info = {
                "match": match_key,
                "market": "1X2",
                "selection": selection,
                "model_prob": round(model_p, 4),
                "sharp_implied": round(sharp_implied, 4),
                "edge_pct": round(edge_pct, 2),
                "best_odds": round(best_odds, 3),
                "best_bookmaker": best_bm,
                "pinnacle_odds": round(pin_odds, 3) if pin_odds > 0 else None,
                "market_category": market_cat,
                "timestamp": datetime.now().isoformat(),
            }

            # Reject obviously wrong edges (>50% = model overconfidence or data issue)
            if edge_pct > 50 or edge_pct < -50:
                continue

            if thresholds["min"] <= edge_pct <= thresholds["max"]:
                value_bets.append(bet_info)
            elif 0 < edge_pct < thresholds["min"] and edge_pct >= thresholds["min"] - WATCHLIST_PROXIMITY_PP:
                bet_info["gap_to_threshold"] = round(thresholds["min"] - edge_pct, 2)
                watchlist.append(bet_info)

        # Check O/U totals
        totals = match_odds.get("totals", [])
        for total in (totals if isinstance(totals, list) else []):
            line = total.get("line", 0)
            if line not in (1.5, 2.5):
                continue

            over_odds = total.get("over", 0)
            if over_odds <= 1.0:
                continue

            # Get model O/U probability from predictions
            # Try multiple key formats (different parts of pipeline use different names)
            model_over = 0
            for key_attempt in [
                f"over_{str(line).replace('.', '_')}",   # over_2_5
                f"over_{str(line).replace('.', '')}",     # over_25
                f"over_{line}",                           # over_2.5
            ]:
                model_over = pred.get(key_attempt, 0)
                if model_over > 0:
                    break

            # Try goal predictions sub-dict
            if model_over <= 0:
                gp = pred.get("goal_predictions", {})
                for key_attempt in [f"over_{line}", f"over_{str(line).replace('.', '_')}"]:
                    model_over = gp.get(key_attempt, 0)
                    if model_over > 0:
                        break

            # Fallback: compute from xG using Poisson
            if model_over <= 0:
                home_xg = pred.get("home_xg", 0)
                away_xg = pred.get("away_xg", 0)
                if home_xg > 0 and away_xg > 0:
                    try:
                        from scipy.stats import poisson
                        total_xg = home_xg + away_xg
                        # P(total goals > line) = 1 - P(total goals <= floor(line))
                        prob_under = 0
                        for h in range(8):
                            for a in range(8):
                                if h + a <= int(line):
                                    prob_under += poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
                        model_over = max(0, 1.0 - prob_under)
                    except Exception:
                        pass

            if model_over <= 0:
                continue

            # Use Pinnacle for sharp benchmark if available
            pin_over = pin_under = 0
            for bm in total.get("all_bookmakers", []):
                if "pinnacle" in bm.get("bookmaker", "").lower():
                    pin_over = bm.get("over", 0)
                    pin_under = bm.get("under", 0)
                    break

            ref_over = pin_over if pin_over > 1.0 else over_odds
            ref_under = total.get("under", 0)
            if ref_under <= 1.0:
                ref_under = pin_under if pin_under > 1.0 else 2.0

            true_probs = _remove_overround([ref_over, ref_under])
            sharp_implied = true_probs[0] if true_probs else 1.0 / ref_over
            edge_pct = ((model_over - sharp_implied) / sharp_implied * 100) if sharp_implied > 0 else 0

            thresholds = EDGE_THRESHOLDS.get("O/U_Over", {"min": 6.0, "max": 8.0})

            bet_info = {
                "match": match_key,
                "market": f"O/U {line}",
                "selection": f"Over {line}",
                "model_prob": round(model_over, 4),
                "sharp_implied": round(sharp_implied, 4),
                "edge_pct": round(edge_pct, 2),
                "best_odds": round(over_odds, 3),
                "best_bookmaker": next(
                    (bm.get("bookmaker", "") for bm in total.get("all_bookmakers", [])
                     if bm.get("over", 0) == over_odds), ""),
                "market_category": "O/U_Over",
                "timestamp": datetime.now().isoformat(),
            }

            # Reject obviously wrong edges (>50% = model overconfidence or data issue)
            if edge_pct > 50 or edge_pct < -50:
                continue

            if thresholds["min"] <= edge_pct <= thresholds["max"]:
                value_bets.append(bet_info)
            elif 0 < edge_pct < thresholds["min"] and edge_pct >= thresholds["min"] - WATCHLIST_PROXIMITY_PP:
                bet_info["gap_to_threshold"] = round(thresholds["min"] - edge_pct, 2)
                watchlist.append(bet_info)

    return {
        "value_bets": sorted(value_bets, key=lambda x: x["edge_pct"], reverse=True),
        "watchlist": sorted(watchlist, key=lambda x: x.get("gap_to_threshold", 99)),
        "scan_time": datetime.now().isoformat(),
        "matches_scanned": len(matches),
    }


def scan_live_value(predictions: Dict) -> List[Dict]:
    """Scan live match odds for in-play value opportunities.

    Reads from data/live/YYYY-MM-DD.json (populated by live_monitor.py)
    and checks if current live odds create edge vs model prediction.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    live_path = DATA_DIR / "live" / f"{today}.json"
    if not live_path.exists():
        return []

    try:
        live_data = json.load(open(live_path))
    except Exception:
        return []

    live_value = []
    matches = live_data.get("matches", live_data)
    if not isinstance(matches, dict):
        return []

    for match_key, match_data in matches.items():
        status = match_data.get("status", "")
        if status not in ("first_half", "second_half", "in_play"):
            continue

        snapshots = match_data.get("snapshots", [])
        if not snapshots:
            continue

        latest = snapshots[-1]
        score = latest.get("score", [0, 0])
        minute = latest.get("min", 0)
        avg_odds = latest.get("avg_odds", {})

        pred = predictions.get(match_key, {})
        if not pred:
            continue

        # Check O/U live value: if game has 1 goal at minute 60,
        # Over 2.5 odds may have drifted to 1.80 but model says 55% over
        total_goals = sum(score)
        goals_needed_25 = max(0, 3 - total_goals)
        minutes_left = max(1, 90 - minute)

        # Simple live O/U model: goals rate from xG, adjusted for current score
        home_xg = pred.get("home_xg", 1.3)
        away_xg = pred.get("away_xg", 1.0)
        total_xg = home_xg + away_xg
        xg_per_min = total_xg / 90

        # Expected remaining goals
        expected_remaining = xg_per_min * minutes_left
        # Probability of scoring N more goals (Poisson)
        from scipy.stats import poisson
        prob_over_25 = 1.0 - poisson.cdf(goals_needed_25 - 1, expected_remaining) if goals_needed_25 > 0 else 1.0

        # Get live Over 2.5 odds
        live_over_odds = avg_odds.get("over_2_5", 0)
        if live_over_odds > 1.0 and prob_over_25 > 0.1:
            implied = 1.0 / live_over_odds
            edge = ((prob_over_25 - implied) / implied * 100) if implied > 0 else 0

            if edge >= 5.0:
                live_value.append({
                    "match": match_key,
                    "market": "O/U 2.5 LIVE",
                    "selection": f"Over 2.5 (score {score[0]}-{score[1]}, min {minute})",
                    "model_prob": round(prob_over_25, 4),
                    "live_odds": round(live_over_odds, 3),
                    "edge_pct": round(edge, 2),
                    "minutes_remaining": minutes_left,
                    "goals_needed": goals_needed_25,
                    "is_live": True,
                    "timestamp": datetime.now().isoformat(),
                })

    return live_value


def scan_arbitrage(odds_data: Dict) -> List[Dict]:
    """Detect cross-bookmaker arbitrage opportunities.

    Arb exists when sum of best implied probs across all outcomes < 1.0.
    E.g., Home @2.10 (Pinnacle) + Draw @3.80 (Betfair) + Away @4.50 (1xBet)
    = 47.6% + 26.3% + 22.2% = 96.1% → 3.9% guaranteed profit.
    """
    arbs = []
    matches = odds_data.get("matches", odds_data)
    if not isinstance(matches, dict):
        return []

    for match_key, match_odds in matches.items():
        h2h = match_odds.get("h2h", {})
        if not isinstance(h2h, dict):
            continue

        best_h = h2h.get("best_home", h2h.get("home", 0))
        best_d = h2h.get("best_draw", h2h.get("draw", 0))
        best_a = h2h.get("best_away", h2h.get("away", 0))

        if best_h <= 1.0 or best_d <= 1.0 or best_a <= 1.0:
            continue

        implied_sum = 1.0/best_h + 1.0/best_d + 1.0/best_a
        if implied_sum < 1.0:
            profit_pct = (1.0 - implied_sum) * 100

            # Find which bookmaker offers the best for each
            bm_list = h2h.get("all_bookmakers", [])
            best_h_bm = best_d_bm = best_a_bm = ""
            for bm in bm_list:
                if bm.get("home", 0) == best_h:
                    best_h_bm = bm.get("bookmaker", "")
                if bm.get("draw", 0) == best_d:
                    best_d_bm = bm.get("bookmaker", "")
                if bm.get("away", 0) == best_a:
                    best_a_bm = bm.get("bookmaker", "")

            arbs.append({
                "match": match_key,
                "profit_pct": round(profit_pct, 2),
                "implied_sum": round(implied_sum, 4),
                "home": {"odds": best_h, "bookmaker": best_h_bm},
                "draw": {"odds": best_d, "bookmaker": best_d_bm},
                "away": {"odds": best_a, "bookmaker": best_a_bm},
            })

    return sorted(arbs, key=lambda x: x["profit_pct"], reverse=True)


def scan_sharp_consensus(odds_data: Dict) -> List[Dict]:
    """Detect when sharp bookmakers disagree with each other.

    Tracks Pinnacle vs Betfair vs Matchbook. When they diverge significantly,
    it may signal an information asymmetry or a pricing error at one sharp book.
    """
    SHARP_BOOKS = {"Pinnacle", "Betfair Sportsbook", "Matchbook", "LowVig.ag"}
    alerts = []
    matches = odds_data.get("matches", odds_data)
    if not isinstance(matches, dict):
        return []

    for match_key, match_odds in matches.items():
        h2h = match_odds.get("h2h", {})
        bm_list = h2h.get("all_bookmakers", []) if isinstance(h2h, dict) else h2h if isinstance(h2h, list) else []

        sharp_odds = {}
        for bm in bm_list:
            name = bm.get("bookmaker", "")
            if name in SHARP_BOOKS:
                sharp_odds[name] = {
                    "home": bm.get("home", 0),
                    "draw": bm.get("draw", 0),
                    "away": bm.get("away", 0),
                }

        if len(sharp_odds) < 2:
            continue

        # Check each outcome for divergence between sharps
        for outcome in ["home", "draw", "away"]:
            values = [(name, data[outcome]) for name, data in sharp_odds.items() if data[outcome] > 1.0]
            if len(values) < 2:
                continue

            odds_vals = [v for _, v in values]
            spread = max(odds_vals) - min(odds_vals)
            if spread > 0.20:  # >0.20 odds difference between sharp books
                highest = max(values, key=lambda x: x[1])
                lowest = min(values, key=lambda x: x[1])
                alerts.append({
                    "match": match_key,
                    "outcome": outcome,
                    "spread": round(spread, 3),
                    "highest": {"bookmaker": highest[0], "odds": highest[1]},
                    "lowest": {"bookmaker": lowest[0], "odds": lowest[1]},
                    "all_sharps": {name: data[outcome] for name, data in sharp_odds.items()},
                })

    return sorted(alerts, key=lambda x: x["spread"], reverse=True)


def run_scan(fetch_fresh: bool = True) -> Dict:
    """Run a single edge scan cycle.

    Args:
        fetch_fresh: If True, fetch fresh odds from API (costs 3 credits).
                     If False, use cached/existing odds data (free).
    """
    # Load predictions (always from disk, free)
    predictions = _load_predictions()
    if not predictions:
        log.warning("No predictions available — skip scan")
        return {"error": "no_predictions"}

    # Fetch or load odds
    if fetch_fresh:
        try:
            from scripts.data.odds_fetcher import fetch_snapshot
            odds_data = fetch_snapshot()
            log.info("Fresh odds snapshot fetched (3 credits)")
        except Exception as e:
            log.warning("Fresh fetch failed: %s — using cached odds", e)
            odds_data = _load_cached_odds()
    else:
        odds_data = _load_cached_odds()

    if not odds_data:
        log.warning("No odds data available")
        return {"error": "no_odds"}

    # Scan for pre-match edges
    scan_result = scan_for_edges(predictions, odds_data)

    # Scan for live value
    live_value = scan_live_value(predictions)
    if live_value:
        scan_result["live_value"] = live_value

    # Scan for arbitrage opportunities
    arbs = scan_arbitrage(odds_data)
    if arbs:
        scan_result["arbitrage"] = arbs
        log.info("Found %d arbitrage opportunities", len(arbs))

    # Scan for sharp book disagreements
    sharp_alerts = scan_sharp_consensus(odds_data)
    if sharp_alerts:
        scan_result["sharp_divergence"] = sharp_alerts

    # Load state and check for NEW alerts
    state = _load_state()
    new_value = []
    new_watchlist = []
    improved_odds = []

    for bet in scan_result.get("value_bets", []):
        key = f"{bet['match']}_{bet['selection']}_{bet['market']}"
        prev = state["alerted"].get(key)
        if prev is None:
            new_value.append(bet)
            state["alerted"][key] = bet
        elif bet["best_odds"] > prev.get("best_odds", 0) + MIN_ODDS_MOVE:
            # Odds improved since last alert
            bet["prev_odds"] = prev.get("best_odds", 0)
            bet["odds_improvement"] = round(bet["best_odds"] - prev["best_odds"], 3)
            improved_odds.append(bet)
            state["alerted"][key] = bet

    for bet in scan_result.get("watchlist", []):
        key = f"{bet['match']}_{bet['selection']}_{bet['market']}"
        if key not in state["watchlist"]:
            new_watchlist.append(bet)
            state["watchlist"][key] = bet

    state["last_poll"] = datetime.now().isoformat()
    _save_state(state)

    # Send notifications for new discoveries
    _notify_discoveries(new_value, new_watchlist, improved_odds, live_value)

    scan_result["new_value_bets"] = new_value
    scan_result["new_watchlist"] = new_watchlist
    scan_result["improved_odds"] = improved_odds
    scan_result["live_value"] = live_value

    # Save scan result
    out_path = BETTING_DIR / "edge_scan_latest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(scan_result, f, indent=2, default=str)

    return scan_result


def _load_cached_odds() -> Dict:
    """Load most recent odds from cache (no API cost)."""
    odds_path = UPCOMING_DIR / "odds_full.json"
    if odds_path.exists():
        try:
            return json.load(open(odds_path))
        except Exception:
            pass
    # Try bookmakers snapshot
    bm_path = UPCOMING_DIR / "odds_bookmakers.json"
    if bm_path.exists():
        try:
            return json.load(open(bm_path))
        except Exception:
            pass
    return {}


def _notify_discoveries(new_value: list, new_watchlist: list,
                        improved: list, live_value: list):
    """Send notifications for new edge discoveries."""
    try:
        from scripts.pipeline.notify import notify, TgMsg, PRIORITY_URGENT, PRIORITY_NORMAL
    except ImportError:
        log.debug("Notification system not available")
        return

    # New value bets — highest priority
    if new_value:
        tg = TgMsg()
        tg.title(f"{len(new_value)} New Value Bet{'s' if len(new_value) > 1 else ''}", emoji="\U0001f4b0")
        tg.blank()
        for bet in new_value[:5]:
            edge = bet["edge_pct"]
            tg.raw(f"\u2705 <b>{bet['match']}</b>")
            tg.raw(f"  {bet['selection']} @{bet['best_odds']:.2f} "
                   f"({bet.get('best_bookmaker', '?')}) | "
                   f"Edge: <b>{edge:+.1f}%</b>")
            tg.blank()

        notify(
            message=f"{len(new_value)} new value bet(s) found between pipeline runs",
            title="New Value Discovered",
            level="success",
            category="betting",
            tg_html=tg.build(),
            priority=PRIORITY_URGENT,
        )

    # Improved odds
    if improved:
        tg = TgMsg()
        tg.title("Odds Improved", emoji="\U0001f4c8")
        tg.blank()
        for bet in improved[:3]:
            tg.raw(f"\u2b06\ufe0f <b>{bet['match']}</b> {bet['selection']}")
            tg.raw(f"  @{bet.get('prev_odds', 0):.2f} \u2192 @{bet['best_odds']:.2f} "
                   f"(+{bet.get('odds_improvement', 0):.2f})")
            tg.blank()

        notify(
            message=f"{len(improved)} bet(s) have improved odds",
            title="Odds Improved",
            level="info",
            category="betting",
            tg_html=tg.build(),
        )

    # Watchlist — lower priority
    if new_watchlist:
        tg = TgMsg()
        tg.title(f"{len(new_watchlist)} Approaching Value", emoji="\U0001f440")
        tg.blank()
        for bet in new_watchlist[:3]:
            gap = bet.get("gap_to_threshold", 0)
            tg.raw(f"\u23f3 {bet['match']} {bet['selection']}")
            tg.raw(f"  Edge: {bet['edge_pct']:+.1f}% ({gap:.1f}pp from threshold)")
            tg.blank()

        notify(
            message=f"{len(new_watchlist)} match(es) approaching value threshold",
            title="Watchlist Update",
            level="info",
            category="betting",
            tg_html=tg.build(),
            priority=PRIORITY_NORMAL,
        )

    # Live value — urgent
    if live_value:
        tg = TgMsg()
        tg.title(f"Live Value Found!", emoji="\u26a1")
        tg.blank()
        for bet in live_value[:3]:
            tg.raw(f"\U0001f534 LIVE: <b>{bet['match']}</b>")
            tg.raw(f"  {bet['selection']} @{bet['live_odds']:.2f} | "
                   f"Edge: <b>{bet['edge_pct']:+.1f}%</b> | "
                   f"{bet['minutes_remaining']}min left")
            tg.blank()

        notify(
            message=f"Live in-play value detected!",
            title="LIVE VALUE",
            level="success",
            category="live",
            tg_html=tg.build(),
            priority=PRIORITY_URGENT,
        )


def run_daemon(interval_seconds: int = 900, max_hours: float = 18):
    """Run continuous polling daemon.

    Args:
        interval_seconds: Seconds between polls (default 15 min)
        max_hours: Max runtime hours (default 18h, avoids overnight waste)
    """
    log.info("Starting edge monitor daemon (interval=%ds, max=%.0fh)",
             interval_seconds, max_hours)

    start = time.time()
    max_seconds = max_hours * 3600
    poll_count = 0

    while time.time() - start < max_seconds:
        poll_count += 1
        try:
            # Only fetch fresh odds every other poll to save credits
            fetch_fresh = (poll_count % 2 == 1)
            result = run_scan(fetch_fresh=fetch_fresh)

            n_value = len(result.get("new_value_bets", []))
            n_watch = len(result.get("new_watchlist", []))
            n_live = len(result.get("live_value", []))
            n_improved = len(result.get("improved_odds", []))

            log.info("Poll #%d: %d new value, %d watchlist, %d improved, %d live | "
                     "total value=%d, total watchlist=%d",
                     poll_count, n_value, n_watch, n_improved, n_live,
                     len(result.get("value_bets", [])),
                     len(result.get("watchlist", [])))

        except Exception as e:
            log.error("Poll #%d failed: %s", poll_count, e)

        time.sleep(interval_seconds)

    log.info("Edge monitor daemon stopped after %.1fh (%d polls)",
             (time.time() - start) / 3600, poll_count)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Continuous odds edge monitor")
    parser.add_argument("--once", action="store_true", help="Single scan, then exit")
    parser.add_argument("--daemon", action="store_true", help="Continuous polling mode")
    parser.add_argument("--dry-run", action="store_true", help="Use cached odds only (no API)")
    parser.add_argument("--interval", type=int, default=900, help="Poll interval in seconds (default 900)")
    parser.add_argument("--max-hours", type=float, default=18, help="Max daemon runtime hours")
    args = parser.parse_args()

    if args.daemon:
        run_daemon(interval_seconds=args.interval, max_hours=args.max_hours)
    else:
        result = run_scan(fetch_fresh=not args.dry_run)
        print(f"\nEdge scan complete:")
        print(f"  Matches scanned: {result.get('matches_scanned', 0)}")
        print(f"  Value bets: {len(result.get('value_bets', []))}")
        print(f"  Watchlist: {len(result.get('watchlist', []))}")
        print(f"  Improved odds: {len(result.get('improved_odds', []))}")
        print(f"  Live value: {len(result.get('live_value', []))}")

        for bet in result.get("value_bets", []):
            print(f"\n  \u2705 {bet['match']} | {bet['selection']} @{bet['best_odds']:.2f}")
            print(f"     Edge: {bet['edge_pct']:+.1f}% | Model: {bet['model_prob']:.1%}")

        for bet in result.get("watchlist", []):
            print(f"\n  \u23f3 {bet['match']} | {bet['selection']} @{bet['best_odds']:.2f}")
            print(f"     Edge: {bet['edge_pct']:+.1f}% (need {bet.get('gap_to_threshold', 0):.1f}pp more)")

        for bet in result.get("live_value", []):
            print(f"\n  \u26a1 LIVE: {bet['match']} | {bet['selection']} @{bet['live_odds']:.2f}")
            print(f"     Edge: {bet['edge_pct']:+.1f}% | {bet['minutes_remaining']}min left")
