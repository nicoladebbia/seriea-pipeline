#!/usr/bin/env python3
"""PER-MARKET EDGE THRESHOLD OPTIMIZER

Sweeps edge thresholds per market (1X2 Draw, 1X2 Home, 1X2 Away, O/U Over)
to find the optimal configuration. Uses the production backtest with flat staking
for clean ROI comparison.

Key question: at what edge threshold does each market become profitable?

Usage:
    python -m scripts.analysis.threshold_sweep
    python -m scripts.analysis.threshold_sweep --staking proportional --bankroll 1000
"""

import itertools
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR

logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger(__name__)


def run_sweep():
    """Run per-market threshold sweep and find optimal configuration."""
    from scripts.analysis.backtest_multimarket import MultiMarketBacktest, print_production_report

    # ─── Step 1: Individual market sweeps ───
    # Test each market independently across thresholds
    THRESHOLDS = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    seasons = ["2023-2024", "2024-2025"]

    print("=" * 90)
    print("PER-MARKET EDGE THRESHOLD SWEEP")
    print("=" * 90)
    print(f"Seasons: {', '.join(seasons)}")
    print(f"Thresholds: {', '.join(f'{t*100:.0f}%' for t in THRESHOLDS)}")
    print(f"Staking: Flat €20 (for clean ROI comparison)")
    print()

    bt = MultiMarketBacktest(seasons=seasons, use_lineup_xg=True)
    bt.load_data()

    # ─── Sweep 1: Draw only (current production behavior) ───
    print("-" * 90)
    print("1X2 DRAW SWEEP (O/U disabled)")
    print(f"{'Threshold':>10} {'Bets':>6} {'Wins':>6} {'WinRate':>8} {'Profit':>10} {'ROI':>8} "
          f"{'AvgEdge':>8} {'AvgCLV':>8} {'S1 ROI':>8} {'S2 ROI':>8}")
    print("-" * 90)

    draw_results = {}
    for t in THRESHOLDS:
        r = bt.run_production(
            staking="flat",
            draw_min_edge=t,
            ou_min_edge=99.0,  # effectively disable O/U
        )
        s = r["combined"]
        # Per-season
        ps = r.get("per_season", {})
        s1_roi = ps.get("2023-2024", {}).get("combined", {}).get("roi", 0)
        s2_roi = ps.get("2024-2025", {}).get("combined", {}).get("roi", 0)

        draw_results[t] = {"combined": s, "s1_roi": s1_roi, "s2_roi": s2_roi}

        if s["bets"] > 0:
            print(f"{t*100:>9.0f}% {s['bets']:>6} {s['wins']:>6} {s['win_rate']:>7.1f}% "
                  f"€{s['profit']:>9.0f} {s['roi']:>+7.1f}% "
                  f"{s['avg_edge']:>+7.2f}% {s['avg_clv']:>+7.3f}% "
                  f"{s1_roi:>+7.1f}% {s2_roi:>+7.1f}%")
        else:
            print(f"{t*100:>9.0f}%      0      -        -          -        -        -        -"
                  f"        -        -")

    # ─── Sweep 2: O/U Over only (Draw disabled) ───
    print()
    print("-" * 90)
    print("O/U OVER SWEEP (Draw disabled)")
    print(f"{'Threshold':>10} {'Bets':>6} {'Wins':>6} {'WinRate':>8} {'Profit':>10} {'ROI':>8} "
          f"{'AvgEdge':>8} {'AvgCLV':>8} {'S1 ROI':>8} {'S2 ROI':>8}")
    print("-" * 90)

    ou_results = {}
    for t in THRESHOLDS:
        r = bt.run_production(
            staking="flat",
            draw_min_edge=99.0,  # effectively disable Draw
            ou_min_edge=t,
        )
        s = r["combined"]
        ps = r.get("per_season", {})
        s1_roi = ps.get("2023-2024", {}).get("combined", {}).get("roi", 0)
        s2_roi = ps.get("2024-2025", {}).get("combined", {}).get("roi", 0)

        ou_results[t] = {"combined": s, "s1_roi": s1_roi, "s2_roi": s2_roi}

        if s["bets"] > 0:
            print(f"{t*100:>9.0f}% {s['bets']:>6} {s['wins']:>6} {s['win_rate']:>7.1f}% "
                  f"€{s['profit']:>9.0f} {s['roi']:>+7.1f}% "
                  f"{s['avg_edge']:>+7.2f}% {s['avg_clv']:>+7.3f}% "
                  f"{s1_roi:>+7.1f}% {s2_roi:>+7.1f}%")
        else:
            print(f"{t*100:>9.0f}%      0      -        -          -        -        -        -"
                  f"        -        -")

    # ─── Sweep 3: 1X2 Home only (others disabled) ───
    print()
    print("-" * 90)
    print("1X2 HOME SWEEP (Draw+O/U disabled)")
    print(f"{'Threshold':>10} {'Bets':>6} {'Wins':>6} {'WinRate':>8} {'Profit':>10} {'ROI':>8} "
          f"{'AvgEdge':>8} {'AvgCLV':>8} {'S1 ROI':>8} {'S2 ROI':>8}")
    print("-" * 90)

    home_results = {}
    for t in THRESHOLDS:
        r = bt.run_production(
            staking="flat",
            draw_min_edge=99.0,
            ou_min_edge=99.0,
            enable_1x2_home=True,
            home_min_edge=t,
        )
        s = r["combined"]
        ps = r.get("per_season", {})
        s1_roi = ps.get("2023-2024", {}).get("combined", {}).get("roi", 0)
        s2_roi = ps.get("2024-2025", {}).get("combined", {}).get("roi", 0)

        home_results[t] = {"combined": s, "s1_roi": s1_roi, "s2_roi": s2_roi}

        if s["bets"] > 0:
            print(f"{t*100:>9.0f}% {s['bets']:>6} {s['wins']:>6} {s['win_rate']:>7.1f}% "
                  f"€{s['profit']:>9.0f} {s['roi']:>+7.1f}% "
                  f"{s['avg_edge']:>+7.2f}% {s['avg_clv']:>+7.3f}% "
                  f"{s1_roi:>+7.1f}% {s2_roi:>+7.1f}%")
        else:
            print(f"{t*100:>9.0f}%      0      -        -          -        -        -        -"
                  f"        -        -")

    # ─── Sweep 4: 1X2 Away only (others disabled) ───
    print()
    print("-" * 90)
    print("1X2 AWAY SWEEP (Draw+O/U disabled)")
    print(f"{'Threshold':>10} {'Bets':>6} {'Wins':>6} {'WinRate':>8} {'Profit':>10} {'ROI':>8} "
          f"{'AvgEdge':>8} {'AvgCLV':>8} {'S1 ROI':>8} {'S2 ROI':>8}")
    print("-" * 90)

    away_results = {}
    for t in THRESHOLDS:
        r = bt.run_production(
            staking="flat",
            draw_min_edge=99.0,
            ou_min_edge=99.0,
            enable_1x2_away=True,
            away_min_edge=t,
        )
        s = r["combined"]
        ps = r.get("per_season", {})
        s1_roi = ps.get("2023-2024", {}).get("combined", {}).get("roi", 0)
        s2_roi = ps.get("2024-2025", {}).get("combined", {}).get("roi", 0)

        away_results[t] = {"combined": s, "s1_roi": s1_roi, "s2_roi": s2_roi}

        if s["bets"] > 0:
            print(f"{t*100:>9.0f}% {s['bets']:>6} {s['wins']:>6} {s['win_rate']:>7.1f}% "
                  f"€{s['profit']:>9.0f} {s['roi']:>+7.1f}% "
                  f"{s['avg_edge']:>+7.2f}% {s['avg_clv']:>+7.3f}% "
                  f"{s1_roi:>+7.1f}% {s2_roi:>+7.1f}%")
        else:
            print(f"{t*100:>9.0f}%      0      -        -          -        -        -        -"
                  f"        -        -")

    # ─── Find best per-market threshold ───
    # Selection criteria: positive ROI in BOTH seasons, with ≥20 bets
    print()
    print("=" * 90)
    print("OPTIMAL THRESHOLDS (positive ROI in both seasons, ≥20 bets)")
    print("=" * 90)

    def find_best(results_dict, market_name):
        """Find the threshold with best ROI that's consistent across seasons."""
        candidates = []
        for t, r in results_dict.items():
            s = r["combined"]
            if s["bets"] >= 20 and r["s1_roi"] > 0 and r["s2_roi"] > 0:
                candidates.append((t, s["roi"], s["bets"], s["profit"], r["s1_roi"], r["s2_roi"]))

        if not candidates:
            # Relax: any positive ROI with ≥15 bets
            for t, r in results_dict.items():
                s = r["combined"]
                if s["bets"] >= 15 and s["roi"] > 0:
                    candidates.append((t, s["roi"], s["bets"], s["profit"], r["s1_roi"], r["s2_roi"]))

        if not candidates:
            return None

        # Pick highest ROI among consistent candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        best = candidates[0]
        return {
            "market": market_name,
            "threshold": best[0],
            "roi": best[1],
            "bets": best[2],
            "profit": best[3],
            "s1_roi": best[4],
            "s2_roi": best[5],
        }

    best_draw = find_best(draw_results, "1X2 Draw")
    best_ou = find_best(ou_results, "O/U Over")
    best_home = find_best(home_results, "1X2 Home")
    best_away = find_best(away_results, "1X2 Away")

    recommendations = []
    for b in [best_draw, best_ou, best_home, best_away]:
        if b:
            status = "ENABLE" if b["roi"] > 3.0 else "MARGINAL"
            print(f"  {b['market']:<12} → {b['threshold']*100:.0f}%  "
                  f"({b['bets']} bets, {b['roi']:+.1f}% ROI, "
                  f"S1={b['s1_roi']:+.1f}% S2={b['s2_roi']:+.1f}%)  [{status}]")
            recommendations.append(b)
        else:
            mkt = {"1X2 Draw": "1X2 Draw", "O/U Over": "O/U Over",
                   "1X2 Home": "1X2 Home", "1X2 Away": "1X2 Away"}.get(
                b["market"] if b else "", "?")
            print(f"  {mkt:<12} → DISABLE (no consistent profitable threshold found)")

    # ─── Combined optimal config ───
    if recommendations:
        print()
        print("-" * 90)
        print("COMBINED OPTIMAL CONFIG")
        print("-" * 90)

        # Run combined backtest with optimal thresholds
        draw_t = best_draw["threshold"] if best_draw else 99.0
        ou_t = best_ou["threshold"] if best_ou else 99.0
        home_t = best_home["threshold"] if best_home else 99.0
        away_t = best_away["threshold"] if best_away else 99.0

        r_optimal = bt.run_production(
            staking="flat",
            draw_min_edge=draw_t,
            ou_min_edge=ou_t,
            enable_1x2_home=best_home is not None and best_home["roi"] > 3.0,
            enable_1x2_away=best_away is not None and best_away["roi"] > 3.0,
            home_min_edge=home_t,
            away_min_edge=away_t,
        )
        print_production_report(r_optimal)

        # ─── Compare with proportional staking ───
        print()
        print("-" * 90)
        print("SAME CONFIG WITH PROPORTIONAL 2% STAKING (€1000 bankroll)")
        print("-" * 90)

        r_prop = bt.run_production(
            staking="proportional",
            bankroll_start=1000.0,
            proportional_pct=2.0,
            draw_min_edge=draw_t,
            ou_min_edge=ou_t,
            enable_1x2_home=best_home is not None and best_home["roi"] > 3.0,
            enable_1x2_away=best_away is not None and best_away["roi"] > 3.0,
            home_min_edge=home_t,
            away_min_edge=away_t,
        )
        print_production_report(r_prop)

        # ─── Compare vs current config ───
        print()
        print("-" * 90)
        print("CURRENT CONFIG (Draw 5%, O/U 8%, no Home/Away)")
        print("-" * 90)
        r_current = bt.run_production(
            staking="flat",
            draw_min_edge=0.05,
            ou_min_edge=0.08,
        )
        print_production_report(r_current)

        # Summary comparison
        c_curr = r_current["combined"]
        c_opt = r_optimal["combined"]
        print()
        print("=" * 90)
        print("COMPARISON: CURRENT vs OPTIMAL")
        print("=" * 90)
        print(f"{'Config':<25} {'Bets':>6} {'Profit':>10} {'ROI':>8} {'AvgCLV':>8}")
        print("-" * 60)
        print(f"{'Current (D=5%, OU=8%)':<25} {c_curr['bets']:>6} €{c_curr['profit']:>9.0f} "
              f"{c_curr['roi']:>+7.1f}% {c_curr['avg_clv']:>+7.3f}%")

        opt_desc = "Optimal ("
        parts = []
        if best_draw:
            parts.append(f"D={best_draw['threshold']*100:.0f}%")
        if best_home and best_home["roi"] > 3.0:
            parts.append(f"H={best_home['threshold']*100:.0f}%")
        if best_away and best_away["roi"] > 3.0:
            parts.append(f"A={best_away['threshold']*100:.0f}%")
        if best_ou:
            parts.append(f"OU={best_ou['threshold']*100:.0f}%")
        opt_desc += ", ".join(parts) + ")"

        print(f"{opt_desc:<25} {c_opt['bets']:>6} €{c_opt['profit']:>9.0f} "
              f"{c_opt['roi']:>+7.1f}% {c_opt['avg_clv']:>+7.3f}%")
        print("=" * 90)

        # Save results
        output = {
            "run_date": datetime.now().isoformat(),
            "recommendations": recommendations,
            "optimal_config": {
                "draw_min_edge": draw_t if best_draw else None,
                "ou_min_edge": ou_t if best_ou else None,
                "home_min_edge": home_t if best_home and best_home["roi"] > 3.0 else None,
                "away_min_edge": away_t if best_away and best_away["roi"] > 3.0 else None,
                "enable_1x2_home": best_home is not None and best_home["roi"] > 3.0,
                "enable_1x2_away": best_away is not None and best_away["roi"] > 3.0,
                "max_edge": 0.15,
            },
            "current_vs_optimal": {
                "current_bets": c_curr["bets"],
                "current_profit": c_curr["profit"],
                "current_roi": c_curr["roi"],
                "optimal_bets": c_opt["bets"],
                "optimal_profit": c_opt["profit"],
                "optimal_roi": c_opt["roi"],
            },
        }

        output_path = DATA_DIR / "optimization" / "threshold_sweep.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run_sweep()
