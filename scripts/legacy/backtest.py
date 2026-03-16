#!/usr/bin/env python3
"""Walk-Forward Backtest Framework for Ensemble Predictions.

Measures ensemble accuracy on historical data using features.parquet
(pre-computed rolling features) and football-data CSVs (results + Pinnacle odds).

Usage:
    python3 scripts/backtest.py --season 2024-2025
    python3 scripts/backtest.py --season 2023-2024 --season 2024-2025
    python3 scripts/backtest.py --season 2024-2025 --compare-baseline
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR
from storage.paths import features_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Serie A base rates (from 20-season analysis)
BASE_RATES = {"H": 0.45, "D": 0.27, "A": 0.28}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_features_for_season(season: str) -> pd.DataFrame:
    """Load pre-computed features for a season from features.parquet.

    Returns DataFrame with results, odds, and all ML features.
    """
    path = features_path()
    if not path.exists():
        raise FileNotFoundError(f"features.parquet not found at {path}")

    df = pd.read_parquet(path)
    season_df = df[df["season"] == season].copy()

    if season_df.empty:
        log.warning(f"No data found for season {season}")
        return pd.DataFrame()

    # Filter to completed matches only
    season_df = season_df[season_df["result"].isin(["H", "D", "A"])].copy()
    log.info(f"Loaded {len(season_df)} matches for {season}")
    return season_df


def load_football_data_season(season: str) -> pd.DataFrame:
    """Load football-data CSV for a season (backup odds source).

    Season format: '2024-2025' -> 'serie_a_2425.csv'
    """
    parts = season.split("-")
    code = parts[0][2:] + parts[1][2:]
    path = DATA_DIR / "external" / "football-data" / f"serie_a_{code}.csv"

    if not path.exists():
        log.warning(f"Football-data CSV not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    log.info(f"Loaded {len(df)} matches from football-data {path.name}")
    return df


# =============================================================================
# PREDICTION METHODS (simplified versions for backtesting)
# =============================================================================

def predict_factor_based(row: pd.Series) -> Dict[str, float]:
    """Factor-based prediction using pre-computed strength/elo features."""
    prob_H = BASE_RATES["H"]
    prob_D = BASE_RATES["D"]
    prob_A = BASE_RATES["A"]

    # Elo-based adjustment
    elo_diff = row.get("elo_diff", 0)
    if not np.isnan(elo_diff) and elo_diff != 0:
        # Elo difference -> probability shift (calibrated)
        elo_shift = elo_diff / 800.0  # ~100 Elo = 12.5pp shift
        prob_H += elo_shift * 0.4
        prob_A -= elo_shift * 0.4

    # Attack/defense strength
    home_atk = row.get("home_attack_strength", 1.0)
    home_def = row.get("home_defense_strength", 1.0)
    away_atk = row.get("away_attack_strength", 1.0)
    away_def = row.get("away_defense_strength", 1.0)

    if not any(np.isnan(v) for v in [home_atk, home_def, away_atk, away_def]) and away_def > 0 and home_def > 0:
        # Strong attack vs weak defense -> more goals
        home_edge = (home_atk / away_def - 1.0) * 0.08
        away_edge = (away_atk / home_def - 1.0) * 0.08
        net_edge = home_edge - away_edge
        prob_H += net_edge
        prob_A -= net_edge

    # H2H features
    h2h_home_rate = row.get("h2h_home_win_rate", np.nan)
    if not np.isnan(h2h_home_rate) and h2h_home_rate > 0:
        h2h_adj = (h2h_home_rate - BASE_RATES["H"]) * 0.05
        prob_H += h2h_adj

    # Normalize
    total = prob_H + prob_D + prob_A
    if total > 0:
        prob_H /= total
        prob_D /= total
        prob_A /= total

    return {"prob_H": prob_H, "prob_D": prob_D, "prob_A": prob_A}


def predict_xg_poisson(row: pd.Series) -> Optional[Dict[str, float]]:
    """xG-based Poisson prediction using pre-computed xG features."""
    # Use understat rolling xG features as the best proxy
    home_xg_atk = row.get("home_xg_attack_strength", np.nan)
    home_xg_def = row.get("home_xg_defense_strength", np.nan)
    away_xg_atk = row.get("away_xg_attack_strength", np.nan)
    away_xg_def = row.get("away_xg_defense_strength", np.nan)

    if any(np.isnan(v) for v in [home_xg_atk, home_xg_def, away_xg_atk, away_xg_def]):
        return None

    # Predicted xG: attack * opponent_defense_conceding * league_avg
    league_avg = 1.38  # Serie A average xG per team
    home_xg = home_xg_atk * away_xg_def * league_avg
    away_xg = away_xg_atk * home_xg_def * league_avg

    # Clamp
    home_xg = max(0.4, min(3.5, home_xg))
    away_xg = max(0.3, min(3.0, away_xg))

    # Home advantage adjustment
    home_xg *= 1.08
    away_xg *= 0.92

    return _poisson_probs(home_xg, away_xg)


def predict_formation_adjusted(
    row: pd.Series, base_probs: Dict[str, float]
) -> Dict[str, float]:
    """Apply formation matchup adjustment to base probabilities.

    Uses formation features from features.parquet when available.
    """
    # Check for formation data
    home_form = row.get("home_formation", "")
    away_form = row.get("away_formation", "")

    if not home_form or not away_form or pd.isna(home_form) or pd.isna(away_form):
        return base_probs

    # Formation matchup rates (from features.parquet if available)
    fm_home_rate = row.get("formation_matchup_home_rate", np.nan)
    fm_draw_rate = row.get("formation_matchup_draw_rate", np.nan)

    if np.isnan(fm_home_rate) or np.isnan(fm_draw_rate):
        return base_probs

    fm_away_rate = 1.0 - fm_home_rate - fm_draw_rate

    # Formation confidence (n_matches based)
    conf = row.get("formation_confidence", "low")
    if conf == "high":
        strength = 0.15
    elif conf == "medium":
        strength = 0.08
    else:
        return base_probs  # Low confidence = skip

    # Deviation from base rates
    adj_h = (fm_home_rate - BASE_RATES["H"]) * strength
    adj_d = (fm_draw_rate - BASE_RATES["D"]) * strength
    adj_a = (fm_away_rate - BASE_RATES["A"]) * strength

    # Cap adjustment at ±5pp
    adj_h = max(-0.05, min(0.05, adj_h))
    adj_d = max(-0.05, min(0.05, adj_d))
    adj_a = max(-0.05, min(0.05, adj_a))

    prob_H = max(0.05, base_probs["prob_H"] + adj_h)
    prob_D = max(0.05, base_probs["prob_D"] + adj_d)
    prob_A = max(0.05, base_probs["prob_A"] + adj_a)

    # Normalize
    total = prob_H + prob_D + prob_A
    return {"prob_H": prob_H / total, "prob_D": prob_D / total, "prob_A": prob_A / total}


def predict_market(row: pd.Series) -> Optional[Dict[str, float]]:
    """Market-implied probabilities from Pinnacle closing odds."""
    psh = row.get("odds_PSH", 0)
    psd = row.get("odds_PSD", 0)
    psa = row.get("odds_PSA", 0)

    if not all(v and v > 1.0 for v in [psh, psd, psa]):
        return None

    raw_h = 1.0 / psh
    raw_d = 1.0 / psd
    raw_a = 1.0 / psa
    total = raw_h + raw_d + raw_a

    if total <= 0:
        return None

    return {
        "prob_H": raw_h / total,
        "prob_D": raw_d / total,
        "prob_A": raw_a / total,
    }


def _poisson_probs(home_xg: float, away_xg: float, max_goals: int = 8) -> Dict[str, float]:
    """Convert xG to win probabilities via Poisson."""
    home_p = [poisson.pmf(g, home_xg) for g in range(max_goals)]
    away_p = [poisson.pmf(g, away_xg) for g in range(max_goals)]

    prob_H = prob_D = prob_A = 0.0
    for h in range(max_goals):
        for a in range(max_goals):
            p = home_p[h] * away_p[a]
            if h > a:
                prob_H += p
            elif h == a:
                prob_D += p
            else:
                prob_A += p

    total = prob_H + prob_D + prob_A
    return {"prob_H": prob_H / total, "prob_D": prob_D / total, "prob_A": prob_A / total}


# =============================================================================
# ENSEMBLE COMBINATION
# =============================================================================

BACKTEST_WEIGHTS = {
    "factor": 0.28,
    "xg": 0.28,
    "market": 0.20,
    "formation": 0.0,  # Off by default, toggled for comparison
}

BACKTEST_WEIGHTS_WITH_FORMATION = {
    "factor": 0.26,
    "xg": 0.26,
    "market": 0.20,
    "formation": 0.08,
}


def predict_ensemble(
    row: pd.Series,
    weights: Dict[str, float] = None,
    use_formation: bool = False,
) -> Dict[str, float]:
    """Generate ensemble prediction for a single match row."""
    if weights is None:
        weights = BACKTEST_WEIGHTS_WITH_FORMATION if use_formation else BACKTEST_WEIGHTS

    predictions = {}

    # Factor-based
    factor_probs = predict_factor_based(row)
    predictions["factor"] = factor_probs

    # xG Poisson
    xg_probs = predict_xg_poisson(row)
    if xg_probs:
        predictions["xg"] = xg_probs

    # Market
    market_probs = predict_market(row)
    if market_probs:
        predictions["market"] = market_probs

    # Formation adjustment (applied to the factor+xg average, not standalone)
    if use_formation:
        # Blend factor + xg as base for formation adjustment
        base = {}
        base_methods = [v for k, v in predictions.items() if k in ("factor", "xg")]
        if base_methods:
            for outcome in ("prob_H", "prob_D", "prob_A"):
                base[outcome] = np.mean([m[outcome] for m in base_methods])
            formation_probs = predict_formation_adjusted(row, base)
            predictions["formation"] = formation_probs

    # Weighted average
    prob_H = prob_D = prob_A = 0.0
    total_w = 0.0
    for method, probs in predictions.items():
        w = weights.get(method, 0)
        if w > 0:
            prob_H += w * probs["prob_H"]
            prob_D += w * probs["prob_D"]
            prob_A += w * probs["prob_A"]
            total_w += w

    if total_w > 0:
        prob_H /= total_w
        prob_D /= total_w
        prob_A /= total_w

    return {"prob_H": prob_H, "prob_D": prob_D, "prob_A": prob_A}


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def evaluate_predictions(
    predictions: List[Dict[str, float]],
    actuals: List[str],
) -> Dict:
    """Compute accuracy, log loss, Brier score, and calibration."""
    n = len(predictions)
    if n == 0:
        return {}

    correct = 0
    total_log_loss = 0.0
    total_brier = 0.0

    # Calibration buckets
    buckets = {
        "25-35": {"predicted": [], "actual": []},
        "35-45": {"predicted": [], "actual": []},
        "45-55": {"predicted": [], "actual": []},
        "55-65": {"predicted": [], "actual": []},
        "65-80": {"predicted": [], "actual": []},
        "80-100": {"predicted": [], "actual": []},
    }

    outcome_stats = {"H": {"correct": 0, "total": 0}, "D": {"correct": 0, "total": 0}, "A": {"correct": 0, "total": 0}}

    for pred, actual in zip(predictions, actuals):
        # Predicted outcome
        probs = [pred["prob_H"], pred["prob_D"], pred["prob_A"]]
        outcomes = ["H", "D", "A"]
        predicted = outcomes[np.argmax(probs)]
        max_prob = max(probs)

        if predicted == actual:
            correct += 1

        # Track per-outcome accuracy
        outcome_stats[predicted]["total"] += 1
        if predicted == actual:
            outcome_stats[predicted]["correct"] += 1

        # Log loss (clip to avoid log(0))
        actual_probs = {"H": pred["prob_H"], "D": pred["prob_D"], "A": pred["prob_A"]}
        p_actual = max(1e-10, actual_probs[actual])
        total_log_loss += -np.log(p_actual)

        # Brier score (multi-class)
        for i, outcome in enumerate(outcomes):
            actual_val = 1.0 if actual == outcome else 0.0
            total_brier += (probs[i] - actual_val) ** 2

        # Calibration bucket
        if max_prob < 0.35:
            bucket = "25-35"
        elif max_prob < 0.45:
            bucket = "35-45"
        elif max_prob < 0.55:
            bucket = "45-55"
        elif max_prob < 0.65:
            bucket = "55-65"
        elif max_prob < 0.80:
            bucket = "65-80"
        else:
            bucket = "80-100"

        buckets[bucket]["predicted"].append(max_prob)
        buckets[bucket]["actual"].append(1.0 if predicted == actual else 0.0)

    accuracy = correct / n
    log_loss = total_log_loss / n
    brier = total_brier / (n * 3)  # Normalized per outcome

    # Build calibration table
    calibration = {}
    for bucket_name, data in buckets.items():
        if data["predicted"]:
            calibration[bucket_name] = {
                "n_matches": len(data["predicted"]),
                "avg_predicted": np.mean(data["predicted"]),
                "avg_actual": np.mean(data["actual"]),
            }

    # Per-outcome accuracy
    outcome_accuracy = {}
    for outcome in ["H", "D", "A"]:
        stats = outcome_stats[outcome]
        if stats["total"] > 0:
            outcome_accuracy[outcome] = {
                "accuracy": stats["correct"] / stats["total"],
                "total": stats["total"],
                "correct": stats["correct"],
            }

    return {
        "n_matches": n,
        "accuracy": accuracy,
        "log_loss": log_loss,
        "brier_score": brier,
        "calibration": calibration,
        "outcome_accuracy": outcome_accuracy,
        "correct": correct,
    }


def compare_with_market(
    our_probs: List[Dict[str, float]],
    actuals: List[str],
) -> Dict:
    """Head-to-head comparison with Pinnacle closing line."""
    market_probs = []
    model_probs = []
    filtered_actuals = []

    for pred, actual in zip(our_probs, actuals):
        market = pred.get("_market")
        if market is None:
            continue
        market_probs.append(market)
        model_probs.append({"prob_H": pred["prob_H"], "prob_D": pred["prob_D"], "prob_A": pred["prob_A"]})
        filtered_actuals.append(actual)

    if not filtered_actuals:
        return {}

    model_eval = evaluate_predictions(model_probs, filtered_actuals)
    market_eval = evaluate_predictions(market_probs, filtered_actuals)

    return {
        "n_matches": len(filtered_actuals),
        "model": {
            "accuracy": model_eval["accuracy"],
            "log_loss": model_eval["log_loss"],
            "brier": model_eval["brier_score"],
        },
        "market": {
            "accuracy": market_eval["accuracy"],
            "log_loss": market_eval["log_loss"],
            "brier": market_eval["brier_score"],
        },
        "accuracy_gap": model_eval["accuracy"] - market_eval["accuracy"],
        "log_loss_gap": model_eval["log_loss"] - market_eval["log_loss"],
    }


# =============================================================================
# WALK-FORWARD BACKTEST
# =============================================================================

def run_backtest(
    seasons: List[str],
    use_formation: bool = False,
    compare_baseline: bool = False,
) -> Dict:
    """Run walk-forward backtest across seasons."""
    all_predictions = []
    all_actuals = []
    season_results = {}

    for season in seasons:
        log.info(f"\n{'='*60}")
        log.info(f"BACKTESTING SEASON: {season}")
        log.info(f"{'='*60}")

        df = load_features_for_season(season)
        if df.empty:
            continue

        predictions = []
        actuals = []

        for _, row in df.iterrows():
            actual = row["result"]
            if actual not in ("H", "D", "A"):
                continue

            pred = predict_ensemble(row, use_formation=use_formation)

            # Attach market probs for comparison
            market = predict_market(row)
            if market:
                pred["_market"] = market

            predictions.append(pred)
            actuals.append(actual)

        # Evaluate this season
        eval_result = evaluate_predictions(predictions, actuals)
        season_results[season] = eval_result

        all_predictions.extend(predictions)
        all_actuals.extend(actuals)

        log.info(f"Season {season}: accuracy={eval_result['accuracy']:.1%}, "
                 f"log_loss={eval_result['log_loss']:.4f}, "
                 f"brier={eval_result['brier_score']:.4f}")

    # Overall evaluation
    overall = evaluate_predictions(all_predictions, all_actuals)
    overall["seasons"] = season_results

    # Market comparison
    if compare_baseline:
        comparison = compare_with_market(all_predictions, all_actuals)
        overall["market_comparison"] = comparison

    # Compare with/without formation if requested
    if compare_baseline and not use_formation:
        log.info("\nRunning WITH formation adjustment for comparison...")
        formation_result = run_backtest(seasons, use_formation=True, compare_baseline=False)
        overall["formation_comparison"] = {
            "without_formation": {
                "accuracy": overall["accuracy"],
                "log_loss": overall["log_loss"],
            },
            "with_formation": {
                "accuracy": formation_result["accuracy"],
                "log_loss": formation_result["log_loss"],
            },
            "accuracy_delta": formation_result["accuracy"] - overall["accuracy"],
            "log_loss_delta": formation_result["log_loss"] - overall["log_loss"],
        }

    return overall


# =============================================================================
# REPORT PRINTING
# =============================================================================

def print_report(results: Dict):
    """Print formatted backtest report."""
    print("\n" + "=" * 70)
    print("BACKTEST REPORT")
    print("=" * 70)

    print(f"\nTotal matches: {results['n_matches']}")
    print(f"Accuracy:      {results['accuracy']:.1%} ({results['correct']}/{results['n_matches']})")
    print(f"Log loss:      {results['log_loss']:.4f}")
    print(f"Brier score:   {results['brier_score']:.4f}")

    # Per-season breakdown
    if "seasons" in results:
        print(f"\n{'Season':<15} {'Matches':>8} {'Accuracy':>10} {'Log Loss':>10} {'Brier':>10}")
        print("-" * 55)
        for season, data in sorted(results["seasons"].items()):
            print(f"{season:<15} {data['n_matches']:>8} {data['accuracy']:>9.1%} "
                  f"{data['log_loss']:>10.4f} {data['brier_score']:>10.4f}")

    # Per-outcome accuracy
    if "outcome_accuracy" in results:
        print(f"\nOutcome accuracy:")
        for outcome, data in results["outcome_accuracy"].items():
            label = {"H": "Home", "D": "Draw", "A": "Away"}[outcome]
            print(f"  {label:>6}: {data['accuracy']:.1%} ({data['correct']}/{data['total']})")

    # Calibration
    if "calibration" in results:
        print(f"\nCalibration:")
        print(f"  {'Bucket':<10} {'Matches':>8} {'Predicted':>10} {'Actual':>10} {'Status':>8}")
        print("  " + "-" * 48)
        for bucket, data in results["calibration"].items():
            diff = abs(data["avg_predicted"] - data["avg_actual"])
            status = "ok" if diff < 0.05 else "DRIFT" if diff < 0.10 else "BAD"
            print(f"  {bucket:<10} {data['n_matches']:>8} {data['avg_predicted']:>9.1%} "
                  f"{data['avg_actual']:>9.1%} {status:>8}")

    # Market comparison
    if "market_comparison" in results and results["market_comparison"]:
        mc = results["market_comparison"]
        print(f"\nVs Pinnacle closing line ({mc['n_matches']} matches):")
        print(f"  {'Metric':<12} {'Model':>10} {'Pinnacle':>10} {'Gap':>10}")
        print("  " + "-" * 42)
        print(f"  {'Accuracy':<12} {mc['model']['accuracy']:>9.1%} {mc['market']['accuracy']:>9.1%} "
              f"{mc['accuracy_gap']:>+9.1%}")
        print(f"  {'Log loss':<12} {mc['model']['log_loss']:>10.4f} {mc['market']['log_loss']:>10.4f} "
              f"{mc['log_loss_gap']:>+10.4f}")
        print(f"  {'Brier':<12} {mc['model']['brier']:>10.4f} {mc['market']['brier']:>10.4f}")

    # Formation comparison
    if "formation_comparison" in results:
        fc = results["formation_comparison"]
        print(f"\nFormation adjustment impact:")
        print(f"  Without: accuracy={fc['without_formation']['accuracy']:.1%}, "
              f"log_loss={fc['without_formation']['log_loss']:.4f}")
        print(f"  With:    accuracy={fc['with_formation']['accuracy']:.1%}, "
              f"log_loss={fc['with_formation']['log_loss']:.4f}")
        print(f"  Delta:   accuracy={fc['accuracy_delta']:+.1%}, "
              f"log_loss={fc['log_loss_delta']:+.4f}")

    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Walk-forward backtest framework")
    parser.add_argument("--season", action="append", dest="seasons",
                        help="Season(s) to backtest (e.g., 2024-2025). Can be repeated.")
    parser.add_argument("--compare-baseline", action="store_true",
                        help="Compare with Pinnacle closing line and formation impact")
    parser.add_argument("--use-formation", action="store_true",
                        help="Include formation-based adjustments")
    parser.add_argument("--save", type=str, default=None,
                        help="Save results to JSON file")

    args = parser.parse_args()

    if not args.seasons:
        args.seasons = ["2023-2024", "2024-2025"]

    log.info(f"Backtesting seasons: {args.seasons}")
    log.info(f"Formation adjustment: {'ON' if args.use_formation else 'OFF'}")
    log.info(f"Compare baseline: {args.compare_baseline}")

    results = run_backtest(
        seasons=args.seasons,
        use_formation=args.use_formation,
        compare_baseline=args.compare_baseline,
    )

    print_report(results)

    if args.save:
        save_path = Path(args.save)
        # Clean results for JSON (remove _market keys)
        clean = {k: v for k, v in results.items() if not k.startswith("_")}
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(clean, f, indent=2, default=str)
        log.info(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()
