#!/usr/bin/env python3
"""Bankroll Ruin Simulator — Monte Carlo season-path analysis.

Simulates thousands of betting season paths at various Kelly fractions
to find the optimal stake sizing that maximizes terminal wealth while
keeping ruin probability below a configurable threshold.

Usage:
    python scripts/bankroll_simulator.py --paths 5000 --ruin-threshold 0.2
    python scripts/bankroll_simulator.py --kelly-fractions 0.05,0.10,0.15,0.20
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR

BANKROLL_OUT_DIR = DATA_DIR / "bankroll"


@dataclass
class RuinSimConfig:
    """Configuration for bankroll ruin simulation."""
    n_paths: int = 5000
    ruin_threshold: float = 0.20     # bankroll < 20% of initial = ruin
    max_ruin_prob: float = 0.05      # must satisfy P(ruin) <= 5%
    initial_bankroll: float = 1000.0
    bets_per_season: int = 300       # approximate bets per Serie A season
    blended_weight_model: float = 0.6  # model_prob weight in blend
    blended_weight_sharp: float = 0.4  # sharp_prob weight in blend
    kelly_fractions: list[float] = field(default_factory=lambda: [
        0.05, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.50, 1.0
    ])
    min_edge: float = 0.0           # minimum edge to place a bet (Kelly sizes small edges small)
    max_stake_pct: float = 0.05     # max single bet as % of bankroll
    seed: int = 42


@dataclass
class BetOpportunity:
    """A single historical bet extracted from data."""
    model_prob: float  # model's predicted probability
    sharp_prob: float  # implied probability from best odds
    odds: float        # decimal odds offered
    won: bool          # whether the bet actually won

    @property
    def blended_prob(self) -> float:
        return 0.6 * self.model_prob + 0.4 * self.sharp_prob

    @property
    def edge(self) -> float:
        return self.blended_prob - (1.0 / self.odds)


def load_historical_edge_distribution(
    config: RuinSimConfig,
    features_path: Optional[Path] = None,
) -> list[BetOpportunity]:
    """Extract historical bet distribution from features.parquet.

    Looks for matches where the model had a positive edge (model_prob > 1/odds)
    to build a realistic distribution of bet opportunities.
    """
    if features_path is None:
        features_path = DATA_DIR / "features" / "features.parquet"

    df = pd.read_parquet(features_path)

    # Need: model prob, odds, and actual result
    required_cols = {"result"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing columns: {required_cols - set(df.columns)}")

    # Find best available odds columns for each outcome
    odds_map = {}
    for outcome, prefixes in [
        ("H", ["odds_B365H", "odds_AvgH", "odds_MaxH"]),
        ("D", ["odds_B365D", "odds_AvgD", "odds_MaxD"]),
        ("A", ["odds_B365A", "odds_AvgA", "odds_MaxA"]),
    ]:
        for col in prefixes:
            if col in df.columns:
                odds_map[outcome] = col
                break

    prob_map = {}
    for outcome, col in [
        ("H", "home_prob_best"),
        ("D", "draw_prob_best"),
        ("A", "away_prob_best"),
    ]:
        if col in df.columns:
            prob_map[outcome] = col

    if not odds_map or not prob_map:
        raise ValueError(
            f"Insufficient odds/prob columns. "
            f"Found odds: {list(odds_map.keys())}, probs: {list(prob_map.keys())}"
        )

    bets = []
    for _, row in df.iterrows():
        result = row.get("result")
        if pd.isna(result):
            continue

        for outcome in ["H", "D", "A"]:
            if outcome not in odds_map or outcome not in prob_map:
                continue

            odds_val = row.get(odds_map[outcome])
            model_p = row.get(prob_map[outcome])

            if pd.isna(odds_val) or pd.isna(model_p) or odds_val <= 1.0:
                continue

            sharp_p = 1.0 / odds_val  # implied prob from odds
            blended = config.blended_weight_model * model_p + config.blended_weight_sharp * sharp_p
            edge = blended - sharp_p

            # Only include bets where model sees positive edge
            if edge >= config.min_edge:
                bets.append(BetOpportunity(
                    model_prob=float(model_p),
                    sharp_prob=float(sharp_p),
                    odds=float(odds_val),
                    won=(result == outcome),
                ))

    return bets


def simulate_season_path(
    bets: list[BetOpportunity],
    kelly_f: float,
    bankroll: float,
    n_bets: int,
    rng: np.random.Generator,
    max_stake_pct: float = 0.05,
) -> dict:
    """Simulate one season path by resampling from historical bets.

    Returns dict with terminal_wealth, max_drawdown, ruin_occurred, path.
    """
    current = bankroll
    peak = bankroll
    max_drawdown = 0.0
    ruin_bankroll = bankroll * 0.20  # ruin threshold

    # Resample bets for this season
    indices = rng.integers(0, len(bets), size=n_bets)

    for idx in indices:
        bet = bets[idx]
        if current <= ruin_bankroll:
            break

        # Kelly sizing
        b = bet.odds - 1.0
        if b <= 0:
            continue
        p = bet.blended_prob
        full_kelly = (b * p - (1 - p)) / b
        if full_kelly <= 0:
            continue

        stake = current * full_kelly * kelly_f
        stake = min(stake, current * max_stake_pct)
        stake = max(0, stake)

        # Simulate outcome using blended probability
        won = rng.random() < p
        if won:
            current += stake * b
        else:
            current -= stake

        # Track drawdown
        if current > peak:
            peak = current
        dd = (peak - current) / peak if peak > 0 else 0
        if dd > max_drawdown:
            max_drawdown = dd

    return {
        "terminal_wealth": current,
        "max_drawdown": max_drawdown,
        "ruin_occurred": current < ruin_bankroll,
        "growth_rate": current / bankroll,
    }


def run_ruin_simulation(config: RuinSimConfig) -> dict:
    """Full sweep across Kelly fractions with parallel season paths.

    Returns dict mapping kelly_fraction → aggregate results.
    """
    print(f"Loading historical edge distribution...")
    bets = load_historical_edge_distribution(config)
    print(f"Found {len(bets)} historical bet opportunities with edge >= {config.min_edge:.0%}")

    if len(bets) < 50:
        print("WARNING: Too few historical bets for reliable simulation")
        if len(bets) == 0:
            return {}

    # Show bet distribution summary
    edges = [b.edge for b in bets]
    win_rate = sum(1 for b in bets if b.won) / len(bets)
    print(f"Win rate: {win_rate:.1%}, Mean edge: {np.mean(edges):.3f}, "
          f"Median odds: {np.median([b.odds for b in bets]):.2f}")

    results = {}
    rng = np.random.default_rng(config.seed)

    for kelly_f in config.kelly_fractions:
        print(f"\n  Kelly f={kelly_f:.3f}: simulating {config.n_paths} paths...")
        paths = []
        for _ in range(config.n_paths):
            path = simulate_season_path(
                bets=bets,
                kelly_f=kelly_f,
                bankroll=config.initial_bankroll,
                n_bets=config.bets_per_season,
                rng=rng,
                max_stake_pct=config.max_stake_pct,
            )
            paths.append(path)

        terminals = np.array([p["terminal_wealth"] for p in paths])
        drawdowns = np.array([p["max_drawdown"] for p in paths])
        ruin_flags = np.array([p["ruin_occurred"] for p in paths])

        results[kelly_f] = {
            "kelly_fraction": kelly_f,
            "n_paths": config.n_paths,
            "mean_terminal": float(np.mean(terminals)),
            "median_terminal": float(np.median(terminals)),
            "p10_terminal": float(np.percentile(terminals, 10)),
            "p90_terminal": float(np.percentile(terminals, 90)),
            "mean_growth": float(np.mean(terminals) / config.initial_bankroll),
            "median_growth": float(np.median(terminals) / config.initial_bankroll),
            "ruin_probability": float(ruin_flags.mean()),
            "mean_max_drawdown": float(np.mean(drawdowns)),
            "p90_max_drawdown": float(np.percentile(drawdowns, 90)),
            "sharpe_like": float(
                (np.mean(terminals) - config.initial_bankroll)
                / max(np.std(terminals), 1)
            ),
        }

        r = results[kelly_f]
        print(f"    Median growth: {r['median_growth']:.2f}x | "
              f"P(ruin): {r['ruin_probability']:.1%} | "
              f"Max DD (p90): {r['p90_max_drawdown']:.1%}")

    return results


def find_optimal_fraction(
    results: dict,
    max_ruin_prob: float = 0.05,
) -> tuple[float, dict]:
    """Find optimal Kelly fraction: max median terminal wealth s.t. P(ruin) <= threshold.

    Returns (optimal_f, result_dict) or (0, {}) if no fraction satisfies constraint.
    """
    eligible = {
        f: r for f, r in results.items()
        if r["ruin_probability"] <= max_ruin_prob
    }

    if not eligible:
        # None satisfy — return the one with lowest ruin prob
        best_f = min(results, key=lambda f: results[f]["ruin_probability"])
        return best_f, results[best_f]

    # Among eligible, maximize median terminal wealth
    best_f = max(eligible, key=lambda f: eligible[f]["median_terminal"])
    return best_f, eligible[best_f]


def print_ruin_report(results: dict, config: RuinSimConfig):
    """Formatted terminal output of ruin simulation results."""
    print("\n" + "=" * 80)
    print("BANKROLL RUIN SIMULATION REPORT")
    print("=" * 80)
    print(f"Paths: {config.n_paths} | Bets/season: {config.bets_per_season} | "
          f"Initial: ${config.initial_bankroll:.0f}")
    print(f"Ruin threshold: {config.ruin_threshold:.0%} of initial | "
          f"Max acceptable P(ruin): {config.max_ruin_prob:.0%}")
    print("-" * 80)
    print(f"{'Kelly f':>8} {'Median$':>10} {'Growth':>8} {'P(ruin)':>8} "
          f"{'P10$':>10} {'P90$':>10} {'MDD(p90)':>9} {'Sharpe':>7}")
    print("-" * 80)

    for f in sorted(results.keys()):
        r = results[f]
        safe = r["ruin_probability"] <= config.max_ruin_prob
        marker = " *" if safe else ""
        print(f"{f:>8.3f} {r['median_terminal']:>10.0f} "
              f"{r['median_growth']:>7.2f}x "
              f"{r['ruin_probability']:>7.1%} "
              f"{r['p10_terminal']:>10.0f} {r['p90_terminal']:>10.0f} "
              f"{r['p90_max_drawdown']:>8.1%} "
              f"{r['sharpe_like']:>7.2f}{marker}")

    print("-" * 80)
    print("* = satisfies P(ruin) constraint")

    optimal_f, optimal_r = find_optimal_fraction(results, config.max_ruin_prob)
    if optimal_r:
        print(f"\nOPTIMAL Kelly fraction: {optimal_f:.3f}")
        print(f"  Median terminal wealth: ${optimal_r['median_terminal']:.0f} "
              f"({optimal_r['median_growth']:.2f}x)")
        print(f"  P(ruin): {optimal_r['ruin_probability']:.1%}")
        print(f"  P90 max drawdown: {optimal_r['p90_max_drawdown']:.1%}")
    else:
        print("\nWARNING: No Kelly fraction satisfies the ruin constraint!")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Bankroll Ruin Simulator")
    parser.add_argument("--paths", type=int, default=5000,
                        help="Number of season paths to simulate (default: 5000)")
    parser.add_argument("--ruin-threshold", type=float, default=0.20,
                        help="Ruin = bankroll below this fraction of initial (default: 0.2)")
    parser.add_argument("--max-ruin-prob", type=float, default=0.05,
                        help="Maximum acceptable ruin probability (default: 0.05)")
    parser.add_argument("--bankroll", type=float, default=1000.0,
                        help="Initial bankroll (default: 1000)")
    parser.add_argument("--bets-per-season", type=int, default=300,
                        help="Approximate bets per season (default: 300)")
    parser.add_argument("--min-edge", type=float, default=0.0,
                        help="Minimum edge to include a bet (default: 0.0)")
    parser.add_argument("--kelly-fractions", type=str, default=None,
                        help="Comma-separated Kelly fractions to test")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    config = RuinSimConfig(
        n_paths=args.paths,
        ruin_threshold=args.ruin_threshold,
        max_ruin_prob=args.max_ruin_prob,
        initial_bankroll=args.bankroll,
        bets_per_season=args.bets_per_season,
        min_edge=args.min_edge,
        seed=args.seed,
    )

    if args.kelly_fractions:
        config.kelly_fractions = [float(x) for x in args.kelly_fractions.split(",")]

    results = run_ruin_simulation(config)
    if not results:
        print("No results — check data availability.")
        sys.exit(1)

    print_ruin_report(results, config)

    # Find optimal and save
    optimal_f, optimal_r = find_optimal_fraction(results, config.max_ruin_prob)

    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_paths": config.n_paths,
            "ruin_threshold": config.ruin_threshold,
            "max_ruin_prob": config.max_ruin_prob,
            "initial_bankroll": config.initial_bankroll,
            "bets_per_season": config.bets_per_season,
            "min_edge": config.min_edge,
            "seed": config.seed,
        },
        "optimal_kelly_fraction": optimal_f,
        "optimal_result": optimal_r,
        "all_results": {str(k): v for k, v in results.items()},
    }

    BANKROLL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BANKROLL_OUT_DIR / "ruin_simulation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
