#!/usr/bin/env python3
"""LIVE vs BACKTEST RECONCILIATION

Compares actual live betting results against backtest expectations.
Detects training-serving skew, calibration drift, and systematic biases.

Key questions:
  1. Is the model probability accurate on unseen data?
  2. Does live ROI match backtest expectations?
  3. Are there systematic biases (e.g., overconfident on draws)?
  4. Is CLV positive (genuine edge vs luck)?

Usage:
    python -m scripts.analysis.live_reconciliation
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR


def load_live_bets() -> List[Dict]:
    """Load settled bets from journal."""
    journal_path = DATA_DIR / "betting" / "bet_journal.json"
    with open(journal_path) as f:
        journal = json.load(f)

    settled = []
    for b in journal["bets"].values():
        if b.get("status") in ("won", "lost", "push"):
            settled.append(b)
    return sorted(settled, key=lambda x: x.get("date", ""))


def analyze_calibration(bets: List[Dict]) -> Dict:
    """Check if model probabilities match actual outcomes.

    Bins bets by model_prob and checks if predicted probability
    matches the observed win rate in each bin.

    This is the key test: if model says 35% chance, does the bet
    actually win ~35% of the time?
    """
    # Only analyze bets with model_prob (not all have it)
    bets_with_prob = [b for b in bets if b.get("model_prob") is not None]
    if not bets_with_prob:
        return {"error": "No bets with model_prob"}

    # Bin by probability ranges
    bins = [
        (0.0, 0.25, "< 25%"),
        (0.25, 0.35, "25-35%"),
        (0.35, 0.45, "35-45%"),
        (0.45, 0.55, "45-55%"),
        (0.55, 1.0, "> 55%"),
    ]

    calibration = []
    for lo, hi, label in bins:
        bin_bets = [b for b in bets_with_prob if lo <= b["model_prob"] < hi]
        if not bin_bets:
            continue
        n = len(bin_bets)
        wins = sum(1 for b in bin_bets if b["status"] == "won")
        avg_prob = np.mean([b["model_prob"] for b in bin_bets])
        actual_wr = wins / n if n > 0 else 0

        calibration.append({
            "bin": label,
            "n_bets": n,
            "avg_model_prob": round(avg_prob, 3),
            "actual_win_rate": round(actual_wr, 3),
            "gap": round(actual_wr - avg_prob, 3),  # positive = model underconfident
        })

    # Overall calibration
    all_probs = [b["model_prob"] for b in bets_with_prob]
    all_wins = [1 if b["status"] == "won" else 0 for b in bets_with_prob]
    mean_pred = np.mean(all_probs)
    mean_actual = np.mean(all_wins)

    # Brier score on live data
    brier = np.mean([(p - a) ** 2 for p, a in zip(all_probs, all_wins)])

    return {
        "n_bets": len(bets_with_prob),
        "mean_predicted_prob": round(mean_pred, 4),
        "actual_win_rate": round(mean_actual, 4),
        "calibration_gap": round(mean_actual - mean_pred, 4),
        "brier_score": round(brier, 4),
        "bins": calibration,
    }


def analyze_edge_performance(bets: List[Dict]) -> Dict:
    """Check if higher model edge actually produces better returns."""
    bets_with_edge = [b for b in bets if b.get("edge_pct") is not None and b["edge_pct"] > 0]
    if not bets_with_edge:
        return {"error": "No bets with edge data"}

    # Bin by edge
    edge_bins = [
        (0, 5, "0-5%"),
        (5, 10, "5-10%"),
        (10, 15, "10-15%"),
        (15, 25, "15-25%"),
        (25, 100, "> 25%"),
    ]

    results = []
    for lo, hi, label in edge_bins:
        bin_bets = [b for b in bets_with_edge if lo <= b["edge_pct"] < hi]
        if not bin_bets:
            continue
        n = len(bin_bets)
        wins = sum(1 for b in bin_bets if b["status"] == "won")
        profit = sum(b.get("profit", 0) or 0 for b in bin_bets)
        staked = sum(b.get("stake", 0) or 0 for b in bin_bets)
        avg_edge = np.mean([b["edge_pct"] for b in bin_bets])
        roi = (profit / staked * 100) if staked > 0 else 0

        results.append({
            "bin": label,
            "n_bets": n,
            "wins": wins,
            "win_rate": round(wins / n * 100, 1),
            "avg_edge": round(avg_edge, 1),
            "profit": round(profit, 2),
            "roi": round(roi, 1),
        })

    return {
        "n_bets": len(bets_with_edge),
        "bins": results,
        # Does higher edge → better ROI? (monotonic relationship = good)
        "edge_roi_correlation": _compute_edge_roi_monotonicity(results),
    }


def _compute_edge_roi_monotonicity(bins: List[Dict]) -> str:
    """Check if ROI increases with edge (expected behavior)."""
    if len(bins) < 2:
        return "insufficient_data"
    rois = [b["roi"] for b in bins if b["n_bets"] >= 3]
    if len(rois) < 2:
        return "insufficient_data"

    # Check if generally increasing (allow some noise)
    increases = sum(1 for i in range(1, len(rois)) if rois[i] > rois[i - 1])
    if increases >= len(rois) - 1:
        return "strong_positive"
    elif increases > len(rois) / 2:
        return "weak_positive"
    else:
        return "no_relationship"


def analyze_market_comparison(bets: List[Dict]) -> Dict:
    """Compare live performance per market against backtest expectations."""
    # Backtest baselines from Phase 7 threshold sweep
    backtest_baselines = {
        "1X2": {"roi": 26.4, "win_rate": 34.9, "note": "Draw at 4% threshold"},
        "O/U 2.5": {"roi": 11.3, "win_rate": 52.0, "note": "Over at 6% threshold"},
        "DC": {"roi": None, "win_rate": None, "note": "Not backtestable (no Pinnacle odds)"},
        "AH": {"roi": None, "win_rate": None, "note": "Disabled in backtest"},
    }

    by_market = defaultdict(lambda: {"bets": 0, "won": 0, "lost": 0, "profit": 0.0, "staked": 0.0})
    for b in bets:
        m = b.get("market", "unknown")
        by_market[m]["bets"] += 1
        by_market[m]["staked"] += b.get("stake", 0) or 0
        by_market[m]["profit"] += b.get("profit", 0) or 0
        if b["status"] == "won":
            by_market[m]["won"] += 1
        elif b["status"] == "lost":
            by_market[m]["lost"] += 1

    results = {}
    for m, s in sorted(by_market.items()):
        roi = (s["profit"] / s["staked"] * 100) if s["staked"] > 0 else 0
        wr = (s["won"] / s["bets"] * 100) if s["bets"] > 0 else 0

        # Find matching baseline
        baseline = None
        for bk, bv in backtest_baselines.items():
            if m.startswith(bk) or bk in m:
                baseline = bv
                break

        entry = {
            "bets": s["bets"],
            "won": s["won"],
            "win_rate": round(wr, 1),
            "roi": round(roi, 1),
            "profit": round(s["profit"], 2),
        }
        if baseline and baseline["roi"] is not None:
            entry["backtest_roi"] = baseline["roi"]
            entry["roi_gap"] = round(roi - baseline["roi"], 1)
            entry["backtest_wr"] = baseline["win_rate"]
            entry["wr_gap"] = round(wr - baseline["win_rate"], 1)

        results[m] = entry

    return results


def analyze_clv(bets: List[Dict]) -> Dict:
    """Analyze Closing Line Value — the #1 indicator of genuine edge.

    CLV > 0 means you consistently beat the closing line,
    which is the strongest available predictor of true probability.
    """
    bets_with_clv = [b for b in bets if b.get("clv_pct") is not None]
    if not bets_with_clv:
        return {"n_bets": 0, "message": "No CLV data available"}

    clvs = [b["clv_pct"] for b in bets_with_clv]
    positive = sum(1 for c in clvs if c > 0)

    # CLV by outcome
    won_clvs = [b["clv_pct"] for b in bets_with_clv if b["status"] == "won"]
    lost_clvs = [b["clv_pct"] for b in bets_with_clv if b["status"] == "lost"]

    return {
        "n_bets": len(bets_with_clv),
        "avg_clv": round(np.mean(clvs), 2),
        "median_clv": round(np.median(clvs), 2),
        "positive_pct": round(positive / len(clvs) * 100, 1),
        "avg_clv_winners": round(np.mean(won_clvs), 2) if won_clvs else None,
        "avg_clv_losers": round(np.mean(lost_clvs), 2) if lost_clvs else None,
        "interpretation": _interpret_clv(np.mean(clvs), len(clvs)),
    }


def _interpret_clv(avg_clv: float, n: int) -> str:
    """Interpret CLV results."""
    if n < 20:
        return "Insufficient sample — need 20+ bets for reliable CLV assessment"
    if avg_clv > 3.0:
        return "Strong positive CLV — genuine edge confirmed, sustainable long-term"
    elif avg_clv > 0:
        return "Positive CLV — likely genuine edge, continue monitoring"
    elif avg_clv > -2.0:
        return "Slightly negative CLV — edge may be marginal, profits could be variance"
    else:
        return "Negative CLV — betting against the market, likely no genuine edge"


def print_reconciliation_report(
    bets: List[Dict],
    calibration: Dict,
    edge_perf: Dict,
    market_comp: Dict,
    clv: Dict,
):
    """Print the full reconciliation report."""
    print()
    print("=" * 80)
    print("  LIVE vs BACKTEST RECONCILIATION")
    print(f"  {len(bets)} settled bets | {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 80)

    # ── 1. Overall performance ──
    total_staked = sum(b.get("stake", 0) or 0 for b in bets)
    total_profit = sum(b.get("profit", 0) or 0 for b in bets)
    wins = sum(1 for b in bets if b["status"] == "won")
    losses = sum(1 for b in bets if b["status"] == "lost")
    pushes = sum(1 for b in bets if b["status"] == "push")
    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0

    print(f"\n  OVERALL: {wins}W-{losses}L-{pushes}P | "
          f"ROI={roi:+.1f}% | P&L=${total_profit:+,.2f}")
    print(f"  Backtest expectation: +20.9% ROI (Phase 7 optimal config)")

    if roi > 15:
        print(f"  [OK] Live ROI ({roi:+.1f}%) is near or above backtest ({20.9:+.1f}%)")
    elif roi > 5:
        print(f"  [OK] Live ROI ({roi:+.1f}%) is positive but below backtest — "
              f"could be small sample variance")
    elif roi > 0:
        print(f"  [!!] Live ROI ({roi:+.1f}%) is marginal — monitor closely")
    else:
        print(f"  [XX] Live ROI ({roi:+.1f}%) is negative — investigate immediately")

    # ── 2. Calibration ──
    print(f"\n  PROBABILITY CALIBRATION (model accuracy on unseen data):")
    if "error" not in calibration:
        print(f"    Mean predicted prob: {calibration['mean_predicted_prob']:.1%}")
        print(f"    Actual win rate:     {calibration['actual_win_rate']:.1%}")
        gap = calibration['calibration_gap']
        gap_word = "underconfident" if gap > 0 else "overconfident"
        print(f"    Calibration gap:     {gap:+.1%} ({gap_word})")
        print(f"    Live Brier score:    {calibration['brier_score']:.4f}")
        print()
        print(f"    {'Bin':<10} {'N':>4} {'Predicted':>10} {'Actual':>10} {'Gap':>8}")
        print(f"    {'-'*45}")
        for b in calibration["bins"]:
            print(f"    {b['bin']:<10} {b['n_bets']:>4} {b['avg_model_prob']:>9.1%} "
                  f"{b['actual_win_rate']:>9.1%} {b['gap']:>+7.1%}")
    else:
        print(f"    {calibration['error']}")

    # ── 3. Edge → ROI ──
    print(f"\n  EDGE PERFORMANCE (does higher edge → better returns?):")
    if "error" not in edge_perf:
        print(f"    {'Edge Bin':<10} {'N':>4} {'WR':>8} {'ROI':>8} {'P&L':>10}")
        print(f"    {'-'*45}")
        for b in edge_perf["bins"]:
            print(f"    {b['bin']:<10} {b['n_bets']:>4} {b['win_rate']:>7.1f}% "
                  f"{b['roi']:>+7.1f}% ${b['profit']:>+9.2f}")
        print(f"    Edge-ROI relationship: {edge_perf['edge_roi_correlation']}")
    else:
        print(f"    {edge_perf['error']}")

    # ── 4. Market comparison ──
    print(f"\n  MARKET COMPARISON (live vs backtest):")
    print(f"    {'Market':<12} {'N':>4} {'WR':>6} {'ROI':>7} {'BT ROI':>7} {'Gap':>7} {'Verdict':>12}")
    print(f"    {'-'*62}")
    for m, s in sorted(market_comp.items(), key=lambda x: -x[1]["bets"]):
        bt_roi = s.get("backtest_roi")
        gap = s.get("roi_gap")
        if bt_roi is not None:
            if gap > -10:
                verdict = "ON TRACK"
            elif gap > -20:
                verdict = "BELOW"
            else:
                verdict = "FAILING"
            print(f"    {m:<12} {s['bets']:>4} {s['win_rate']:>5.1f}% {s['roi']:>+6.1f}% "
                  f"{bt_roi:>+6.1f}% {gap:>+6.1f}% {verdict:>12}")
        else:
            print(f"    {m:<12} {s['bets']:>4} {s['win_rate']:>5.1f}% {s['roi']:>+6.1f}% "
                  f"{'N/A':>7} {'N/A':>7} {'unvalidated':>12}")

    # ── 5. CLV ──
    print(f"\n  CLOSING LINE VALUE (edge sustainability indicator):")
    if clv.get("n_bets", 0) > 0:
        print(f"    Average CLV:    {clv['avg_clv']:+.1f}%")
        print(f"    Median CLV:     {clv['median_clv']:+.1f}%")
        print(f"    Positive CLV:   {clv['positive_pct']:.0f}% of bets")
        if clv.get("avg_clv_winners") is not None:
            print(f"    CLV (winners):  {clv['avg_clv_winners']:+.1f}%")
        if clv.get("avg_clv_losers") is not None:
            print(f"    CLV (losers):   {clv['avg_clv_losers']:+.1f}%")
        print(f"    Assessment:     {clv['interpretation']}")
    else:
        print(f"    {clv.get('message', 'No CLV data')}")

    # ── 6. Verdict ──
    print(f"\n  {'='*60}")
    issues = []
    strengths = []

    if roi > 10:
        strengths.append(f"Strong live ROI ({roi:+.1f}%)")
    elif roi > 0:
        strengths.append(f"Positive live ROI ({roi:+.1f}%)")
    else:
        issues.append(f"Negative live ROI ({roi:+.1f}%)")

    if clv.get("avg_clv", 0) > 0:
        strengths.append(f"Positive CLV ({clv['avg_clv']:+.1f}%) — genuine edge")
    elif clv.get("n_bets", 0) > 10:
        issues.append(f"Negative CLV ({clv.get('avg_clv', 0):+.1f}%) — edge questionable")

    cal_gap = calibration.get("calibration_gap", 0)
    if abs(cal_gap) < 0.05:
        strengths.append(f"Well-calibrated (gap={cal_gap:+.1%})")
    elif cal_gap > 0.05:
        issues.append(f"Model underconfident (gap={cal_gap:+.1%}) — leaving money on table")
    else:
        issues.append(f"Model overconfident (gap={cal_gap:+.1%}) — inflating edge estimates")

    if "1X2" in market_comp and market_comp["1X2"]["roi"] < -20:
        issues.append(f"1X2 severely underperforming (ROI={market_comp['1X2']['roi']:+.1f}%)")

    print(f"  VERDICT:")
    if strengths:
        for s in strengths:
            print(f"    [+] {s}")
    if issues:
        for i in issues:
            print(f"    [-] {i}")
    if not issues:
        print(f"    System is performing as expected. Continue monitoring.")
    print(f"  {'='*60}")
    print()


def run():
    """Run full reconciliation analysis."""
    bets = load_live_bets()
    if not bets:
        print("No settled bets found in journal")
        return

    calibration = analyze_calibration(bets)
    edge_perf = analyze_edge_performance(bets)
    market_comp = analyze_market_comparison(bets)
    clv = analyze_clv(bets)

    print_reconciliation_report(bets, calibration, edge_perf, market_comp, clv)

    # Save to file
    output = {
        "run_date": datetime.now().isoformat(),
        "n_bets": len(bets),
        "calibration": calibration,
        "edge_performance": edge_perf,
        "market_comparison": {k: v for k, v in market_comp.items()},
        "clv": clv,
    }
    output_path = DATA_DIR / "optimization" / "live_reconciliation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    run()
