"""CLV (Closing Line Value) Analysis by Market.

Analyzes which bet types consistently beat the closing line and which don't.
Links CLV data with actual P&L to determine:
  1. Which markets have genuine, sustained edge (positive CLV + positive P&L)
  2. Which markets are lucky (positive P&L but negative CLV)
  3. Which markets are unlucky (positive CLV but negative P&L)
  4. Which markets to avoid (negative CLV + negative P&L)

Also computes: trends over time, bookmaker analysis, confidence correlation,
and generates actionable staking recommendations.

Usage:
    python3 -m scripts.analysis.clv_analysis              # Full report
    python3 -m scripts.analysis.clv_analysis --market DC   # Single market
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
BETTING_DIR = DATA_DIR / "betting"


def load_clv_with_pnl() -> List[Dict]:
    """Load settled bets that have both CLV and P&L data.

    Merges clv_history.json with bet_journal.json to get the full picture:
    CLV (did we beat the closing line?) + P&L (did we actually profit?).
    """
    # Load journal (has P&L per bet)
    journal_path = BETTING_DIR / "bet_journal.json"
    if not journal_path.exists():
        return []

    journal = json.load(open(journal_path))
    bets = journal.get("bets", {})

    enriched = []
    for bet_id, bet in bets.items():
        status = bet.get("status")
        if status not in ("won", "lost"):
            continue

        clv_pct = bet.get("clv_pct")
        market = bet.get("market", "unknown")
        selection = bet.get("selection", "")
        odds = bet.get("odds", 0)
        stake = bet.get("stake", 0)
        edge_pct = bet.get("edge_pct", 0)
        date = bet.get("date", "")
        match = bet.get("match", "")

        # Compute profit
        if status == "won":
            profit = bet.get("profit", stake * (odds - 1))
        else:
            profit = -(stake or 0)

        enriched.append({
            "bet_id": bet_id,
            "match": match,
            "date": date,
            "league": bet.get("league", "serie_a"),
            "market": market,
            "selection": selection,
            "odds": odds,
            "stake": stake,
            "edge_pct": edge_pct,
            "status": status,
            "profit": profit,
            "clv_pct": clv_pct,
            "has_clv": clv_pct is not None,
            "closing_odds": bet.get("closing_odds"),
            "bookmaker": bet.get("bookmaker", ""),
            "confidence": bet.get("confidence", ""),
        })

    return enriched


def analyze_by_market(min_sample: int = 5) -> Dict:
    """Per-market CLV + P&L analysis.

    For each market, computes:
      - Sample size, win rate
      - Average CLV % and positive CLV rate
      - Total P&L and ROI
      - CLV-profit correlation (does beating closing line predict profit?)
      - Edge quality quadrant (genuine/lucky/unlucky/avoid)
    """
    bets = load_clv_with_pnl()
    if not bets:
        return {"error": "No data available"}

    # Group by market
    by_market = defaultdict(list)
    for b in bets:
        by_market[b["market"]].append(b)

    results = {}
    for market, market_bets in sorted(by_market.items()):
        n = len(market_bets)
        won = sum(1 for b in market_bets if b["status"] == "won")
        total_stake = sum(b["stake"] for b in market_bets)
        total_profit = sum(b["profit"] for b in market_bets)
        roi = (total_profit / total_stake * 100) if total_stake > 0 else 0

        # CLV stats
        with_clv = [b for b in market_bets if b["has_clv"]]
        n_clv = len(with_clv)
        if n_clv > 0:
            avg_clv = np.mean([b["clv_pct"] for b in with_clv])
            median_clv = np.median([b["clv_pct"] for b in with_clv])
            positive_clv = sum(1 for b in with_clv if b["clv_pct"] > 0)
            positive_rate = positive_clv / n_clv

            # CLV standard error
            clv_values = [b["clv_pct"] for b in with_clv]
            clv_std = np.std(clv_values)
            clv_se = clv_std / math.sqrt(n_clv) if n_clv > 1 else 0
            clv_ci_low = avg_clv - 1.96 * clv_se
            clv_ci_high = avg_clv + 1.96 * clv_se

            # CLV-profit correlation
            clv_won = [b["clv_pct"] for b in with_clv if b["status"] == "won"]
            clv_lost = [b["clv_pct"] for b in with_clv if b["status"] == "lost"]
            avg_clv_won = np.mean(clv_won) if clv_won else 0
            avg_clv_lost = np.mean(clv_lost) if clv_lost else 0
        else:
            avg_clv = median_clv = 0
            positive_clv = positive_rate = 0
            clv_se = clv_ci_low = clv_ci_high = 0
            avg_clv_won = avg_clv_lost = 0

        # Edge quality quadrant
        if avg_clv > 1.0 and roi > 0:
            quadrant = "GENUINE_EDGE"
            emoji = "\U0001f7e2"  # green
        elif avg_clv <= 1.0 and roi > 0:
            quadrant = "LUCKY"
            emoji = "\U0001f7e1"  # yellow
        elif avg_clv > 1.0 and roi <= 0:
            quadrant = "UNLUCKY"
            emoji = "\U0001f7e0"  # orange
        else:
            quadrant = "NO_EDGE"
            emoji = "\U0001f534"  # red

        # Staking recommendation
        if n < min_sample:
            recommendation = "INSUFFICIENT_DATA"
            stake_adj = 1.0
        elif quadrant == "GENUINE_EDGE" and positive_rate >= 0.65:
            recommendation = "INCREASE_STAKE"
            stake_adj = min(1.5, 1.0 + avg_clv / 10)
        elif quadrant == "GENUINE_EDGE":
            recommendation = "MAINTAIN"
            stake_adj = 1.0
        elif quadrant == "LUCKY":
            recommendation = "REDUCE_STAKE"
            stake_adj = 0.8
        elif quadrant == "UNLUCKY" and n_clv >= 20:
            recommendation = "MAINTAIN_MONITOR"
            stake_adj = 0.9
        else:
            recommendation = "CONSIDER_DISABLING"
            stake_adj = 0.5

        results[market] = {
            "n_bets": n,
            "n_won": won,
            "win_rate": round(won / n * 100, 1) if n > 0 else 0,
            "total_staked": round(total_stake, 2),
            "total_profit": round(total_profit, 2),
            "roi_pct": round(roi, 1),
            "n_with_clv": n_clv,
            "avg_clv_pct": round(avg_clv, 2),
            "median_clv_pct": round(median_clv, 2),
            "positive_clv_rate": round(positive_rate * 100, 1) if n_clv > 0 else 0,
            "clv_95ci": [round(clv_ci_low, 2), round(clv_ci_high, 2)],
            "avg_clv_won": round(avg_clv_won, 2),
            "avg_clv_lost": round(avg_clv_lost, 2),
            "quadrant": quadrant,
            "quadrant_emoji": emoji,
            "recommendation": recommendation,
            "stake_adjustment": round(stake_adj, 2),
        }

    # Group by (league, market) for multi-league dimension
    by_league_market = defaultdict(list)
    for b in bets:
        key = (b.get("league", "serie_a"), b["market"])
        by_league_market[key].append(b)

    league_market_results = {}
    for (league, market), lm_bets in sorted(by_league_market.items()):
        n = len(lm_bets)
        won = sum(1 for b in lm_bets if b["status"] == "won")
        total_stake = sum(b["stake"] for b in lm_bets)
        total_profit = sum(b["profit"] for b in lm_bets)
        roi = (total_profit / total_stake * 100) if total_stake > 0 else 0

        with_clv = [b for b in lm_bets if b["has_clv"]]
        n_clv = len(with_clv)
        avg_clv = np.mean([b["clv_pct"] for b in with_clv]) if n_clv > 0 else 0
        positive_rate = (sum(1 for b in with_clv if b["clv_pct"] > 0) / n_clv) if n_clv > 0 else 0

        league_market_results[f"{league}:{market}"] = {
            "league": league,
            "market": market,
            "n_bets": n,
            "n_won": won,
            "win_rate": round(won / n * 100, 1) if n > 0 else 0,
            "total_staked": round(total_stake, 2),
            "total_profit": round(total_profit, 2),
            "roi_pct": round(roi, 1),
            "n_with_clv": n_clv,
            "avg_clv_pct": round(avg_clv, 2),
            "positive_clv_rate": round(positive_rate * 100, 1) if n_clv > 0 else 0,
        }

    return {
        "by_market": results,
        "by_league_market": league_market_results,
        "total_bets": len(bets),
        "total_with_clv": sum(1 for b in bets if b["has_clv"]),
        "overall_roi": round(sum(b["profit"] for b in bets) / max(1, sum(b["stake"] for b in bets)) * 100, 1),
        "overall_clv": round(np.mean([b["clv_pct"] for b in bets if b["has_clv"]]), 2) if any(b["has_clv"] for b in bets) else 0,
    }


def analyze_by_league(min_sample: int = 3) -> Dict:
    """Per-league CLV + P&L summary.

    Groups all settled bets by league to compare edge quality
    across competitions. Returns per-league CLV, ROI, win rate,
    and staking recommendations.
    """
    bets = load_clv_with_pnl()
    if not bets:
        return {"error": "No data available"}

    by_league = defaultdict(list)
    for b in bets:
        by_league[b.get("league", "serie_a")].append(b)

    results = {}
    for league, league_bets in sorted(by_league.items()):
        n = len(league_bets)
        if n < min_sample:
            continue

        won = sum(1 for b in league_bets if b["status"] == "won")
        total_stake = sum(b["stake"] for b in league_bets)
        total_profit = sum(b["profit"] for b in league_bets)
        roi = (total_profit / total_stake * 100) if total_stake > 0 else 0

        with_clv = [b for b in league_bets if b["has_clv"]]
        n_clv = len(with_clv)
        avg_clv = np.mean([b["clv_pct"] for b in with_clv]) if n_clv > 0 else 0
        positive_clv = sum(1 for b in with_clv if b["clv_pct"] > 0) if n_clv > 0 else 0
        positive_rate = (positive_clv / n_clv) if n_clv > 0 else 0

        # Per-market breakdown within this league
        market_breakdown = defaultdict(list)
        for b in league_bets:
            market_breakdown[b["market"]].append(b)

        markets = {}
        for mkt, mkt_bets in sorted(market_breakdown.items()):
            mn = len(mkt_bets)
            mwon = sum(1 for b in mkt_bets if b["status"] == "won")
            mstake = sum(b["stake"] for b in mkt_bets)
            mprofit = sum(b["profit"] for b in mkt_bets)
            mroi = (mprofit / mstake * 100) if mstake > 0 else 0
            mclv_bets = [b for b in mkt_bets if b["has_clv"]]
            mavg_clv = np.mean([b["clv_pct"] for b in mclv_bets]) if mclv_bets else 0
            markets[mkt] = {
                "n_bets": mn,
                "win_rate": round(mwon / mn * 100, 1) if mn > 0 else 0,
                "roi_pct": round(mroi, 1),
                "avg_clv_pct": round(mavg_clv, 2),
            }

        results[league] = {
            "n_bets": n,
            "n_won": won,
            "win_rate": round(won / n * 100, 1) if n > 0 else 0,
            "total_staked": round(total_stake, 2),
            "total_profit": round(total_profit, 2),
            "roi_pct": round(roi, 1),
            "n_with_clv": n_clv,
            "avg_clv_pct": round(avg_clv, 2),
            "positive_clv_rate": round(positive_rate * 100, 1) if n_clv > 0 else 0,
            "markets": markets,
        }

    return {
        "by_league": results,
        "total_bets": len(bets),
        "n_leagues": len(results),
    }


def analyze_trends(window_weeks: int = 2) -> Dict:
    """CLV trend analysis over time — detect edge degradation."""
    bets = load_clv_with_pnl()
    with_clv = [b for b in bets if b["has_clv"] and b["date"]]
    if len(with_clv) < 10:
        return {"error": "Insufficient data for trend analysis"}

    # Sort by date
    with_clv.sort(key=lambda b: b["date"])

    # Rolling window analysis
    window_days = window_weeks * 7
    periods = []
    dates = [b["date"] for b in with_clv]
    min_date = datetime.strptime(dates[0], "%Y-%m-%d")
    max_date = datetime.strptime(dates[-1], "%Y-%m-%d")

    current = min_date
    while current <= max_date:
        end = current + timedelta(days=window_days)
        period_bets = [b for b in with_clv
                       if current <= datetime.strptime(b["date"], "%Y-%m-%d") < end]
        if period_bets:
            avg_clv = np.mean([b["clv_pct"] for b in period_bets])
            profit = sum(b["profit"] for b in period_bets)
            staked = sum(b["stake"] for b in period_bets)
            roi = (profit / staked * 100) if staked > 0 else 0
            periods.append({
                "start": current.strftime("%Y-%m-%d"),
                "end": (end - timedelta(days=1)).strftime("%Y-%m-%d"),
                "n_bets": len(period_bets),
                "avg_clv": round(avg_clv, 2),
                "roi": round(roi, 1),
                "positive_clv_rate": round(sum(1 for b in period_bets if b["clv_pct"] > 0) / len(period_bets) * 100, 1),
            })
        current += timedelta(days=window_days)

    # Trend direction
    if len(periods) >= 2:
        clvs = [p["avg_clv"] for p in periods]
        if clvs[-1] > clvs[0] + 0.5:
            trend = "IMPROVING"
        elif clvs[-1] < clvs[0] - 0.5:
            trend = "DEGRADING"
        else:
            trend = "STABLE"
    else:
        trend = "INSUFFICIENT_DATA"

    return {
        "periods": periods,
        "trend": trend,
        "n_periods": len(periods),
        "window_weeks": window_weeks,
    }


def analyze_by_selection(min_sample: int = 3) -> Dict:
    """Per-selection CLV analysis (e.g., 'Draw' vs 'Home' vs 'Over 2.5')."""
    bets = load_clv_with_pnl()
    with_clv = [b for b in bets if b["has_clv"]]

    by_sel = defaultdict(list)
    for b in with_clv:
        key = f"{b['market']}:{b['selection']}"
        by_sel[key].append(b)

    results = {}
    for sel, sel_bets in sorted(by_sel.items()):
        n = len(sel_bets)
        if n < min_sample:
            continue
        avg_clv = np.mean([b["clv_pct"] for b in sel_bets])
        profit = sum(b["profit"] for b in sel_bets)
        staked = sum(b["stake"] for b in sel_bets)
        roi = (profit / staked * 100) if staked > 0 else 0
        won = sum(1 for b in sel_bets if b["status"] == "won")
        results[sel] = {
            "n_bets": n,
            "win_rate": round(won / n * 100, 1),
            "avg_clv_pct": round(avg_clv, 2),
            "positive_rate": round(sum(1 for b in sel_bets if b["clv_pct"] > 0) / n * 100, 1),
            "roi_pct": round(roi, 1),
            "total_profit": round(profit, 2),
        }

    return results


def analyze_by_edge_bucket() -> Dict:
    """CLV analysis by model edge size — does higher edge = higher CLV?"""
    bets = load_clv_with_pnl()
    with_clv = [b for b in bets if b["has_clv"] and b.get("edge_pct")]

    buckets = {"<4%": [], "4-6%": [], "6-8%": [], "8-10%": [], ">10%": []}
    for b in with_clv:
        e = b["edge_pct"]
        if e < 4:
            buckets["<4%"].append(b)
        elif e < 6:
            buckets["4-6%"].append(b)
        elif e < 8:
            buckets["6-8%"].append(b)
        elif e < 10:
            buckets["8-10%"].append(b)
        else:
            buckets[">10%"].append(b)

    results = {}
    for bucket, bb in buckets.items():
        if not bb:
            continue
        results[bucket] = {
            "n_bets": len(bb),
            "avg_clv": round(np.mean([b["clv_pct"] for b in bb]), 2),
            "positive_rate": round(sum(1 for b in bb if b["clv_pct"] > 0) / len(bb) * 100, 1),
            "win_rate": round(sum(1 for b in bb if b["status"] == "won") / len(bb) * 100, 1),
            "roi": round(sum(b["profit"] for b in bb) / max(1, sum(b["stake"] for b in bb)) * 100, 1),
        }

    return results


def generate_full_report() -> Dict:
    """Complete CLV analysis report with all dimensions."""
    report = {
        "generated_at": datetime.now().isoformat(),
        "market_analysis": analyze_by_market(),
        "league_analysis": analyze_by_league(),
        "selection_analysis": analyze_by_selection(),
        "edge_analysis": analyze_by_edge_bucket(),
        "trend_analysis": analyze_trends(),
    }

    # Save report
    out_path = BETTING_DIR / "clv_analysis_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("CLV analysis saved to %s", out_path)

    return report


def print_report(report: Dict = None):
    """Print formatted CLV analysis to console."""
    if report is None:
        report = generate_full_report()

    ma = report["market_analysis"]
    print("=" * 75)
    print("  CLV ANALYSIS BY MARKET")
    print(f"  {ma['total_bets']} bets | {ma['total_with_clv']} with CLV | "
          f"Overall ROI: {ma['overall_roi']:+.1f}% | Overall CLV: {ma['overall_clv']:+.2f}%")
    print("=" * 75)

    print(f"\n  {'Market':<12} {'Bets':>5} {'WR':>6} {'ROI':>7} {'CLV':>7} "
          f"{'CLV+':>6} {'Quadrant':<16} {'Action'}")
    print(f"  {'-'*12} {'-'*5} {'-'*6} {'-'*7} {'-'*7} {'-'*6} {'-'*16} {'-'*20}")

    for market, data in sorted(ma["by_market"].items(),
                                key=lambda x: x[1]["avg_clv_pct"], reverse=True):
        print(f"  {data['quadrant_emoji']} {market:<10} {data['n_bets']:>5} "
              f"{data['win_rate']:>5.0f}% {data['roi_pct']:>+6.1f}% "
              f"{data['avg_clv_pct']:>+6.2f}% {data['positive_clv_rate']:>5.0f}% "
              f"{data['quadrant']:<16} {data['recommendation']}")

    # Edge bucket analysis
    ea = report.get("edge_analysis", {})
    if ea:
        print(f"\n  {'Edge Bucket':<12} {'Bets':>5} {'CLV':>7} {'CLV+':>6} {'WR':>6} {'ROI':>7}")
        print(f"  {'-'*12} {'-'*5} {'-'*7} {'-'*6} {'-'*6} {'-'*7}")
        for bucket in ["<4%", "4-6%", "6-8%", "8-10%", ">10%"]:
            if bucket in ea:
                d = ea[bucket]
                print(f"  {bucket:<12} {d['n_bets']:>5} {d['avg_clv']:>+6.2f}% "
                      f"{d['positive_rate']:>5.0f}% {d['win_rate']:>5.0f}% {d['roi']:>+6.1f}%")

    # Trend analysis
    ta = report.get("trend_analysis", {})
    if ta and ta.get("periods"):
        print(f"\n  TREND: {ta['trend']} ({ta['n_periods']} periods of {ta['window_weeks']} weeks)")
        for p in ta["periods"]:
            print(f"    {p['start']} — {p['end']}: {p['n_bets']} bets, "
                  f"CLV {p['avg_clv']:+.2f}%, ROI {p['roi']:+.1f}%")

    # Recommendations
    print("\n  STAKING RECOMMENDATIONS:")
    for market, data in sorted(ma["by_market"].items()):
        if data["recommendation"] in ("INCREASE_STAKE", "CONSIDER_DISABLING"):
            action = data["recommendation"].replace("_", " ")
            adj = data["stake_adjustment"]
            print(f"    {data['quadrant_emoji']} {market}: {action} (x{adj:.2f})")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    report = generate_full_report()
    print_report(report)
