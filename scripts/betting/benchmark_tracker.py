"""Benchmark Tracker — compare betting performance before vs after improvements.

Marks a cutoff date and compares metrics for bets placed before vs after.
Tracks progress toward 50-bet validation target.

Usage:
    from scripts.betting.benchmark_tracker import get_benchmark_report
    report = get_benchmark_report()
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR

log = logging.getLogger(__name__)

BENCHMARK_PATH = DATA_DIR / "betting" / "benchmark.json"

# Improvements deployed on this date
IMPROVEMENT_CUTOFF = "2026-04-10"
VALIDATION_TARGET = 50  # Bets needed to validate improvements


def _load_benchmark() -> Dict:
    if BENCHMARK_PATH.exists():
        with open(BENCHMARK_PATH) as f:
            return json.load(f)
    return {
        "cutoff_date": IMPROVEMENT_CUTOFF,
        "created_at": datetime.now().isoformat(),
        "target_bets": VALIDATION_TARGET,
        "improvements": [
            "Accuracy: 52.03% -> 53.34% (+1.31pp)",
            "Draw F1: 0.23 -> 0.34 (+48.6%)",
            "ECE: 0.046 -> 0.014 (-69%)",
            "Edge window: 4-8% -> 3-7%",
            "1X2 disabled (-20.4% ROI)",
            "Monday+Friday blocked",
            "O/U 2.5 suspended",
            "Kelly: edge-quality multiplier + real-time bankroll",
            "Per-market Kelly: O/U 8%, DC 6%",
            "Entry timing enforced (AVOID = block)",
            "Quality-first portfolio (replaces diversity pass)",
        ],
    }


def _compute_period_stats(bets: List[Dict]) -> Dict:
    """Compute stats for a list of settled bets."""
    if not bets:
        return {
            "count": 0, "wins": 0, "losses": 0, "pushes": 0,
            "win_rate": 0, "total_staked": 0, "total_profit": 0,
            "roi_pct": 0, "avg_edge": 0, "avg_clv": None,
            "avg_odds": 0, "by_market": {},
        }

    wins = sum(1 for b in bets if b.get("status") == "won")
    losses = sum(1 for b in bets if b.get("status") == "lost")
    pushes = sum(1 for b in bets if b.get("status") == "push")
    total_staked = sum(b.get("stake", 0) or 0 for b in bets)
    total_profit = sum(b.get("profit", 0) or 0 for b in bets)
    edges = [b.get("edge_pct", 0) or 0 for b in bets if b.get("edge_pct")]
    clvs = [b.get("clv_pct") for b in bets if b.get("clv_pct") is not None]
    odds_list = [b.get("odds", 0) or 0 for b in bets if b.get("odds")]

    # By market breakdown
    by_market: Dict[str, Dict] = {}
    for b in bets:
        mkt = b.get("market", "unknown")
        if mkt not in by_market:
            by_market[mkt] = {"bets": 0, "wins": 0, "staked": 0, "profit": 0}
        by_market[mkt]["bets"] += 1
        if b.get("status") == "won":
            by_market[mkt]["wins"] += 1
        by_market[mkt]["staked"] += b.get("stake", 0) or 0
        by_market[mkt]["profit"] += b.get("profit", 0) or 0

    for mkt, stats in by_market.items():
        stats["wr"] = round(stats["wins"] / stats["bets"] * 100, 1) if stats["bets"] else 0
        stats["roi"] = round(stats["profit"] / stats["staked"] * 100, 2) if stats["staked"] else 0

    settled = wins + losses + pushes
    return {
        "count": settled,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": round(wins / settled * 100, 1) if settled else 0,
        "total_staked": round(total_staked, 2),
        "total_profit": round(total_profit, 2),
        "roi_pct": round(total_profit / total_staked * 100, 2) if total_staked else 0,
        "avg_edge": round(sum(edges) / len(edges), 2) if edges else 0,
        "avg_clv": round(sum(clvs) / len(clvs), 2) if clvs else None,
        "clv_positive_pct": round(sum(1 for c in clvs if c > 0) / len(clvs) * 100, 1) if clvs else None,
        "avg_odds": round(sum(odds_list) / len(odds_list), 2) if odds_list else 0,
        "by_market": by_market,
    }


def get_benchmark_report() -> Dict:
    """Generate the full before/after comparison report."""
    benchmark = _load_benchmark()
    cutoff = benchmark["cutoff_date"]

    # Load all settled bets from journal
    from scripts.betting.bet_journal import _load_journal
    journal = _load_journal()

    before_bets = []
    after_bets = []

    for bet in journal["bets"].values():
        if bet.get("status") not in ("won", "lost", "push"):
            continue
        # Use placed_at (when bet was made) for before/after split, NOT match date.
        # A bet placed before improvements but for a match after cutoff belongs in "before".
        bet_placed = (bet.get("placed_at") or bet.get("date", ""))[:10]
        if bet_placed < cutoff:
            before_bets.append(bet)
        else:
            after_bets.append(bet)

    before_stats = _compute_period_stats(before_bets)
    after_stats = _compute_period_stats(after_bets)

    # Compute deltas
    deltas = {}
    for key in ["win_rate", "roi_pct", "avg_edge"]:
        b = before_stats.get(key, 0) or 0
        a = after_stats.get(key, 0) or 0
        deltas[key] = round(a - b, 2)

    if before_stats.get("avg_clv") is not None and after_stats.get("avg_clv") is not None:
        deltas["avg_clv"] = round(after_stats["avg_clv"] - before_stats["avg_clv"], 2)

    # Progress toward target
    progress = min(100, round(after_stats["count"] / VALIDATION_TARGET * 100, 1))

    # Generate verdict
    if after_stats["count"] < 10:
        verdict = "Too early to judge. Need at least 10 bets after improvements."
        verdict_status = "pending"
    elif after_stats["count"] < VALIDATION_TARGET:
        if after_stats["roi_pct"] > before_stats["roi_pct"]:
            verdict = f"Trending positive! ROI {after_stats['roi_pct']:+.1f}% vs {before_stats['roi_pct']:+.1f}% before. {VALIDATION_TARGET - after_stats['count']} bets to go."
            verdict_status = "positive"
        elif after_stats["roi_pct"] > 0:
            verdict = f"Profitable but below baseline. ROI {after_stats['roi_pct']:+.1f}% vs {before_stats['roi_pct']:+.1f}% before. Still early."
            verdict_status = "neutral"
        else:
            verdict = f"Negative ROI ({after_stats['roi_pct']:+.1f}%). Investigate: are edge caps too tight? Sample size: {after_stats['count']} bets."
            verdict_status = "negative"
    else:
        # Full validation
        if after_stats["roi_pct"] >= 5.0:
            verdict = f"SUCCESS: ROI {after_stats['roi_pct']:+.1f}% over {after_stats['count']} bets. Improvements validated."
            verdict_status = "success"
        elif after_stats["roi_pct"] > before_stats["roi_pct"]:
            verdict = f"Improved: ROI {after_stats['roi_pct']:+.1f}% vs {before_stats['roi_pct']:+.1f}% before. Consider further optimization."
            verdict_status = "positive"
        else:
            verdict = f"No improvement: ROI {after_stats['roi_pct']:+.1f}% vs {before_stats['roi_pct']:+.1f}% before. Review changes."
            verdict_status = "negative"

    # Recommendations based on after data
    recommendations = []
    if after_stats["count"] >= 10:
        for mkt, mstats in after_stats.get("by_market", {}).items():
            if mstats["bets"] >= 5 and mstats["roi"] < -10:
                recommendations.append(f"Consider disabling {mkt}: {mstats['roi']:+.1f}% ROI on {mstats['bets']} bets")
            elif mstats["bets"] >= 5 and mstats["roi"] > 10:
                recommendations.append(f"{mkt} performing well: {mstats['roi']:+.1f}% ROI. Consider increasing Kelly.")

        if after_stats.get("clv_positive_pct") is not None and after_stats["clv_positive_pct"] < 50:
            recommendations.append("CLV positive rate below 50% — edge may be eroding. Check model freshness.")

        if after_stats["win_rate"] < 40:
            recommendations.append("Win rate below 40%. Check if edge thresholds are calibrated correctly.")

    # --- Extended analytics (all settled bets, chronological by settlement) ---
    all_bets = sorted(before_bets + after_bets,
                      key=lambda b: b.get("settled_at") or b.get("date") or "")

    return {
        "cutoff_date": cutoff,
        "improvements": benchmark.get("improvements", []),
        "target_bets": VALIDATION_TARGET,
        "progress_pct": progress,
        "before": before_stats,
        "after": after_stats,
        "deltas": deltas,
        "verdict": verdict,
        "verdict_status": verdict_status,
        "recommendations": recommendations,
        "pnl_timeline": _compute_pnl_timeline(all_bets, cutoff),
        "edge_distribution": _compute_edge_distribution(all_bets),
        "odds_performance": _compute_odds_performance(all_bets),
        "streaks": _compute_streaks(all_bets),
        "best_worst": _compute_best_worst(all_bets),
        "weekly_trend": _compute_weekly_trend(all_bets),
        "clv_analysis": _compute_clv_analysis(all_bets),
        "generated_at": datetime.now().isoformat(),
    }


def _compute_pnl_timeline(bets: List[Dict], cutoff: str) -> List[Dict]:
    """Cumulative P&L per bet, sorted by settlement time (when profit realized)."""
    cumulative = 0.0
    balance = 1000.0  # Starting bankroll
    timeline = []
    for i, b in enumerate(bets):
        profit = b.get("profit", 0) or 0
        cumulative += profit
        balance += profit
        # Use settled_at for timeline date (when cash flow happened)
        settled_date = (b.get("settled_at") or b.get("date") or "")[:10]
        # is_after based on placed_at (when the bet decision was made)
        placed_date = (b.get("placed_at") or b.get("date") or "")[:10]
        timeline.append({
            "num": i + 1,
            "date": settled_date,
            "match": b.get("match", ""),
            "market": b.get("market", ""),
            "profit": round(profit, 2),
            "cumulative": round(cumulative, 2),
            "balance": round(balance, 2),
            "is_after": placed_date >= cutoff,
        })
    return timeline


def _compute_edge_distribution(bets: List[Dict]) -> List[Dict]:
    """ROI by edge bucket."""
    buckets = [(0, 3, "0-3%"), (3, 5, "3-5%"), (5, 7, "5-7%"), (7, 10, "7-10%"), (10, 100, "10%+")]
    result = []
    for lo, hi, label in buckets:
        group = [b for b in bets if lo <= (b.get("edge_pct") or 0) < hi]
        if not group:
            result.append({"bucket": label, "count": 0, "wins": 0, "wr": 0, "roi": 0})
            continue
        wins = sum(1 for b in group if b.get("status") == "won")
        staked = sum(b.get("stake", 0) or 0 for b in group)
        profit = sum(b.get("profit", 0) or 0 for b in group)
        result.append({
            "bucket": label, "count": len(group), "wins": wins,
            "wr": round(wins / len(group) * 100, 1),
            "roi": round(profit / staked * 100, 1) if staked else 0,
        })
    return result


def _compute_odds_performance(bets: List[Dict]) -> List[Dict]:
    """ROI by odds range."""
    ranges = [(1.0, 1.5, "1.00-1.50"), (1.5, 2.0, "1.50-2.00"),
              (2.0, 2.5, "2.00-2.50"), (2.5, 3.0, "2.50-3.00"), (3.0, 5.0, "3.00+")]
    result = []
    for lo, hi, label in ranges:
        group = [b for b in bets if lo <= (b.get("odds") or 0) < hi]
        if not group:
            result.append({"range": label, "count": 0, "wr": 0, "roi": 0})
            continue
        wins = sum(1 for b in group if b.get("status") == "won")
        staked = sum(b.get("stake", 0) or 0 for b in group)
        profit = sum(b.get("profit", 0) or 0 for b in group)
        result.append({
            "range": label, "count": len(group),
            "wr": round(wins / len(group) * 100, 1),
            "roi": round(profit / staked * 100, 1) if staked else 0,
        })
    return result


def _compute_streaks(bets: List[Dict]) -> Dict:
    """Current and historical streaks."""
    if not bets:
        return {"current": "—", "longest_win": 0, "longest_loss": 0, "recent_20": []}

    outcomes = []
    for b in bets:
        s = b.get("status", "")
        if s == "won":
            outcomes.append("W")
        elif s == "lost":
            outcomes.append("L")
        elif s == "push":
            outcomes.append("P")

    # Current streak
    current_type = outcomes[-1] if outcomes else "—"
    current_len = 0
    for o in reversed(outcomes):
        if o == current_type:
            current_len += 1
        else:
            break

    # Longest streaks
    longest_win = longest_loss = 0
    run = 0
    prev = None
    for o in outcomes:
        if o == prev:
            run += 1
        else:
            run = 1
            prev = o
        if o == "W":
            longest_win = max(longest_win, run)
        elif o == "L":
            longest_loss = max(longest_loss, run)

    return {
        "current": f"{current_len}{current_type}",
        "longest_win": longest_win,
        "longest_loss": longest_loss,
        "recent_20": outcomes[-20:],
    }


def _compute_best_worst(bets: List[Dict]) -> Dict:
    """Top 5 best and worst bets by profit."""
    valid = [b for b in bets if b.get("profit") is not None]
    by_profit = sorted(valid, key=lambda b: b.get("profit") or 0, reverse=True)

    def _fmt(b):
        return {
            "match": b.get("match", ""),
            "market": b.get("market", ""),
            "selection": b.get("selection", ""),
            "odds": b.get("odds", 0),
            "profit": round(b.get("profit", 0), 2),
            "date": (b.get("date") or "")[:10],
        }

    return {
        "best": [_fmt(b) for b in by_profit[:5]],
        "worst": [_fmt(b) for b in by_profit[-5:]],
    }


def _compute_weekly_trend(bets: List[Dict]) -> List[Dict]:
    """ROI by ISO week."""
    from collections import defaultdict
    weeks: Dict[str, Dict] = defaultdict(lambda: {"bets": 0, "staked": 0, "profit": 0})

    for b in bets:
        date_str = (b.get("date") or "")[:10]
        if not date_str:
            continue
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            week_key = dt.strftime("%Y-W%V")
        except ValueError:
            continue
        weeks[week_key]["bets"] += 1
        weeks[week_key]["staked"] += b.get("stake", 0) or 0
        weeks[week_key]["profit"] += b.get("profit", 0) or 0

    result = []
    for week in sorted(weeks.keys())[-12:]:  # Last 12 weeks
        w = weeks[week]
        result.append({
            "week": week,
            "bets": w["bets"],
            "profit": round(w["profit"], 2),
            "roi": round(w["profit"] / w["staked"] * 100, 1) if w["staked"] else 0,
        })
    return result


def _compute_clv_analysis(bets: List[Dict]) -> Dict:
    """CLV stats and CLV vs outcome correlation."""
    clv_bets = [b for b in bets if b.get("clv_pct") is not None]
    if not clv_bets:
        return {"avg": None, "median": None, "positive_pct": None,
                "clv_pos_wr": None, "clv_neg_wr": None, "by_market": {}}

    clvs = [b["clv_pct"] for b in clv_bets]
    clvs_sorted = sorted(clvs)
    median = clvs_sorted[len(clvs_sorted) // 2]

    pos_bets = [b for b in clv_bets if b["clv_pct"] > 0]
    neg_bets = [b for b in clv_bets if b["clv_pct"] <= 0]

    pos_wr = (sum(1 for b in pos_bets if b.get("status") == "won") / len(pos_bets) * 100) if pos_bets else 0
    neg_wr = (sum(1 for b in neg_bets if b.get("status") == "won") / len(neg_bets) * 100) if neg_bets else 0

    # CLV by market
    from collections import defaultdict
    mkt_clvs: Dict[str, list] = defaultdict(list)
    for b in clv_bets:
        mkt_clvs[b.get("market", "unknown")].append(b["clv_pct"])
    by_market = {m: round(sum(v) / len(v), 2) for m, v in mkt_clvs.items() if len(v) >= 3}

    return {
        "avg": round(sum(clvs) / len(clvs), 2),
        "median": round(median, 2),
        "positive_pct": round(sum(1 for c in clvs if c > 0) / len(clvs) * 100, 1),
        "clv_pos_wr": round(pos_wr, 1),
        "clv_neg_wr": round(neg_wr, 1),
        "clv_pos_count": len(pos_bets),
        "clv_neg_count": len(neg_bets),
        "by_market": by_market,
    }


if __name__ == "__main__":
    report = get_benchmark_report()
    print(json.dumps(report, indent=2, default=str))
