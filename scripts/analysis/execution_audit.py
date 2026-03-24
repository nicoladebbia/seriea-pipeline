"""Execution slippage audit + match-type performance segmentation.

Two analyses that reveal WHY model edge isn't fully converting to profit:

1. EXECUTION SLIPPAGE: Compares recommended odds vs closing odds vs actual
   execution (when available). Answers: "Am I getting fair prices?"

2. MATCH-TYPE PERFORMANCE: Segments model accuracy by match category
   (derbies, promoted teams, relegation battles, Elo gaps, midweek).
   Answers: "Which match types should I bet more/less on?"

Usage:
    python3 -m scripts.analysis.execution_audit           # Full report
    python3 -m scripts.analysis.execution_audit --slippage # Slippage only
    python3 -m scripts.analysis.execution_audit --segments # Segments only
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
BETTING_DIR = DATA_DIR / "betting"


# ─── EXECUTION SLIPPAGE ANALYSIS ──────────────────────────────────────────

def analyze_slippage() -> Dict:
    """Analyze execution slippage: recommended odds vs closing line vs actual.

    For each settled bet, computes:
    - model_edge: model_prob vs sharp_implied (the theoretical edge)
    - clv: did we beat the closing line?
    - execution_gap: recommended odds vs actual odds (if tracked)
    - bookmaker_slippage: per-bookmaker average gap
    """
    journal_path = BETTING_DIR / "bet_journal.json"
    if not journal_path.exists():
        return {"error": "No bet journal"}

    journal = json.load(open(journal_path))
    bets = journal.get("bets", {})

    settled = []
    for bet_id, bet in bets.items():
        if bet.get("status") not in ("won", "lost"):
            continue

        recommended_odds = bet.get("odds", 0)
        closing_odds = bet.get("closing_odds")
        execution_odds = bet.get("execution_odds")  # New field from /api/bet/place
        clv_pct = bet.get("clv_pct")
        edge_pct = bet.get("edge_pct", 0)
        model_prob = bet.get("model_prob", 0)
        sharp_prob = bet.get("sharp_implied_prob", 0)
        bookmaker = bet.get("bookmaker", "unknown")
        market = bet.get("market", "")
        status = bet.get("status")
        stake = bet.get("stake", 0)
        profit = bet.get("profit", 0) if status == "won" else -(stake or 0)

        # Compute slippage
        rec_implied = 1.0 / recommended_odds if recommended_odds > 1 else 0
        close_implied = 1.0 / closing_odds if closing_odds and closing_odds > 1 else 0
        exec_implied = 1.0 / execution_odds if execution_odds and execution_odds > 1 else 0

        # Slippage = how much worse we got vs recommendation
        slippage_vs_close = 0
        if close_implied > 0 and rec_implied > 0:
            slippage_vs_close = (rec_implied - close_implied) / rec_implied * 100

        settled.append({
            "bet_id": bet_id,
            "match": bet.get("match", ""),
            "market": market,
            "selection": bet.get("selection", ""),
            "bookmaker": bookmaker,
            "status": status,
            "recommended_odds": recommended_odds,
            "closing_odds": closing_odds,
            "execution_odds": execution_odds,
            "clv_pct": clv_pct,
            "edge_pct": edge_pct,
            "model_prob": model_prob,
            "slippage_vs_close_pct": round(slippage_vs_close, 2),
            "profit": profit,
            "stake": stake,
        })

    if not settled:
        return {"error": "No settled bets with slippage data"}

    # Aggregate by bookmaker
    by_bookmaker = defaultdict(list)
    for b in settled:
        by_bookmaker[b["bookmaker"]].append(b)

    bookmaker_stats = {}
    for bm, bm_bets in by_bookmaker.items():
        clvs = [b["clv_pct"] for b in bm_bets if b["clv_pct"] is not None]
        profits = [b["profit"] for b in bm_bets]
        stakes = [b["stake"] for b in bm_bets]
        bookmaker_stats[bm] = {
            "n_bets": len(bm_bets),
            "avg_clv": round(np.mean(clvs), 2) if clvs else None,
            "positive_clv_rate": round(sum(1 for c in clvs if c > 0) / len(clvs) * 100, 1) if clvs else 0,
            "total_profit": round(sum(profits), 2),
            "roi_pct": round(sum(profits) / max(1, sum(stakes)) * 100, 1),
            "won": sum(1 for b in bm_bets if b["status"] == "won"),
        }

    # Aggregate by market
    by_market = defaultdict(list)
    for b in settled:
        by_market[b["market"]].append(b)

    market_slippage = {}
    for mkt, mkt_bets in by_market.items():
        clvs = [b["clv_pct"] for b in mkt_bets if b["clv_pct"] is not None]
        edges = [b["edge_pct"] for b in mkt_bets if b["edge_pct"]]
        profits = [b["profit"] for b in mkt_bets]
        stakes = [b["stake"] for b in mkt_bets]

        # Edge conversion: what % of model edge actually turned into profit?
        avg_edge = np.mean(edges) if edges else 0
        actual_roi = sum(profits) / max(1, sum(stakes)) * 100
        edge_conversion = (actual_roi / avg_edge * 100) if avg_edge > 0 else 0

        market_slippage[mkt] = {
            "n_bets": len(mkt_bets),
            "avg_edge": round(avg_edge, 2),
            "actual_roi": round(actual_roi, 1),
            "edge_conversion_pct": round(edge_conversion, 1),
            "avg_clv": round(np.mean(clvs), 2) if clvs else None,
            "won": sum(1 for b in mkt_bets if b["status"] == "won"),
            "win_rate": round(sum(1 for b in mkt_bets if b["status"] == "won") / len(mkt_bets) * 100, 1),
        }

    # Overall slippage
    all_clvs = [b["clv_pct"] for b in settled if b["clv_pct"] is not None]
    all_edges = [b["edge_pct"] for b in settled if b["edge_pct"]]
    total_profit = sum(b["profit"] for b in settled)
    total_staked = sum(b["stake"] for b in settled)

    return {
        "total_bets": len(settled),
        "overall": {
            "avg_clv": round(np.mean(all_clvs), 2) if all_clvs else 0,
            "avg_edge": round(np.mean(all_edges), 2) if all_edges else 0,
            "actual_roi": round(total_profit / max(1, total_staked) * 100, 1),
            "edge_conversion": round(
                (total_profit / max(1, total_staked) * 100) / max(0.01, np.mean(all_edges)) * 100, 1
            ) if all_edges else 0,
            "positive_clv_rate": round(
                sum(1 for c in all_clvs if c > 0) / max(1, len(all_clvs)) * 100, 1
            ) if all_clvs else 0,
        },
        "by_bookmaker": dict(sorted(bookmaker_stats.items(),
                                     key=lambda x: x[1]["n_bets"], reverse=True)),
        "by_market": dict(sorted(market_slippage.items(),
                                  key=lambda x: x[1]["n_bets"], reverse=True)),
    }


# ─── MATCH-TYPE PERFORMANCE SEGMENTATION ──────────────────────────────────

def analyze_match_types() -> Dict:
    """Segment model accuracy by match type.

    Joins OOF predictions with features.parquet to determine which
    match categories the model performs best/worst on.
    """
    oof_path = DATA_DIR / "models" / "universal" / "cv_predictions.parquet"
    feat_path = DATA_DIR / "features" / "features.parquet"

    if not oof_path.exists() or not feat_path.exists():
        return {"error": "Missing OOF predictions or features"}

    oof = pd.read_parquet(oof_path)
    features = pd.read_parquet(feat_path)

    # Derive predicted outcome
    label_map = {0: "H", 1: "D", 2: "A"}
    oof["predicted"] = oof[["prob_H", "prob_D", "prob_A"]].values.argmax(axis=1)
    oof["predicted"] = oof["predicted"].map(label_map)
    oof["correct"] = oof["actual"] == oof["predicted"]
    oof["confidence"] = oof[["prob_H", "prob_D", "prob_A"]].max(axis=1)

    # Filter to 2017+ data
    oof = oof[oof["season"] >= "2017-2018"].copy()

    # Join with features on season + match_date
    oof["match_date"] = pd.to_datetime(oof["match_date"])
    features["match_date"] = pd.to_datetime(features["match_date"])

    # Match on date (features may have more rows due to pre-2017 data)
    context_cols = [
        "match_date", "season", "home_team", "away_team",
        "is_derby", "home_is_promoted", "away_is_promoted",
        "is_midweek", "is_late_season", "is_early_season",
        "elo_diff", "matchup_competitiveness",
        "home_in_relegation_zone", "away_in_relegation_zone",
    ]
    available = [c for c in context_cols if c in features.columns]
    ctx = features[available].copy()
    ctx = ctx[ctx["season"] >= "2017-2018"]

    # Merge
    merged = oof.merge(ctx, on=["match_date", "season"], how="left")

    segments = {}

    # 1. Derby vs non-derby
    if "is_derby" in merged.columns:
        for is_derby, label in [(True, "Derby"), (False, "Non-Derby")]:
            mask = merged["is_derby"] == is_derby
            if mask.sum() < 10:
                continue
            subset = merged[mask]
            segments[label] = _compute_segment_stats(subset)

    # 2. Promoted team involved
    if "home_is_promoted" in merged.columns:
        promoted = merged[(merged["home_is_promoted"] == 1) | (merged.get("away_is_promoted", 0) == 1)]
        normal = merged[(merged["home_is_promoted"] == 0) & (merged.get("away_is_promoted", 0) == 0)]
        if len(promoted) >= 10:
            segments["Promoted Team"] = _compute_segment_stats(promoted)
        if len(normal) >= 10:
            segments["No Promoted"] = _compute_segment_stats(normal)

    # 3. Relegation battle
    if "home_in_relegation_zone" in merged.columns:
        releg = merged[(merged["home_in_relegation_zone"] == 1) | (merged.get("away_in_relegation_zone", 0) == 1)]
        stable = merged[(merged["home_in_relegation_zone"] == 0) & (merged.get("away_in_relegation_zone", 0) == 0)]
        if len(releg) >= 10:
            segments["Relegation Battle"] = _compute_segment_stats(releg)
        if len(stable) >= 10:
            segments["Mid/Top Table"] = _compute_segment_stats(stable)

    # 4. Elo gap buckets
    if "elo_diff" in merged.columns:
        elo = merged["elo_diff"].abs()
        for label, low, high in [
            ("Tight (Elo <50)", 0, 50),
            ("Moderate (Elo 50-150)", 50, 150),
            ("Lopsided (Elo >150)", 150, 999),
        ]:
            mask = (elo >= low) & (elo < high)
            if mask.sum() >= 10:
                segments[label] = _compute_segment_stats(merged[mask])

    # 5. Midweek vs weekend
    if "is_midweek" in merged.columns:
        for val, label in [(1, "Midweek"), (0, "Weekend")]:
            mask = merged["is_midweek"] == val
            if mask.sum() >= 10:
                segments[label] = _compute_segment_stats(merged[mask])

    # 6. Season phase
    if "is_late_season" in merged.columns:
        for col, label in [("is_late_season", "Late Season"), ("is_early_season", "Early Season")]:
            if col in merged.columns:
                mask = merged[col] == 1
                if mask.sum() >= 10:
                    segments[label] = _compute_segment_stats(merged[mask])

    # 7. Confidence buckets
    for label, low, high in [
        ("Low Conf (<40%)", 0, 0.40),
        ("Medium Conf (40-55%)", 0.40, 0.55),
        ("High Conf (55-70%)", 0.55, 0.70),
        ("Very High Conf (>70%)", 0.70, 1.0),
    ]:
        mask = (merged["confidence"] >= low) & (merged["confidence"] < high)
        if mask.sum() >= 10:
            segments[label] = _compute_segment_stats(merged[mask])

    # Overall
    segments["OVERALL"] = _compute_segment_stats(merged)

    # Rank by accuracy
    ranked = sorted(segments.items(), key=lambda x: x[1]["accuracy"], reverse=True)

    return {
        "segments": dict(ranked),
        "total_predictions": len(merged),
        "recommendations": _generate_recommendations(segments),
    }


def _compute_segment_stats(df: pd.DataFrame) -> Dict:
    """Compute accuracy and calibration stats for a segment."""
    n = len(df)
    correct = df["correct"].sum()
    accuracy = correct / n if n > 0 else 0

    # Draw detection
    draw_mask = df["actual"] == "D"
    draw_pred = df["predicted"] == "D"
    draw_tp = (draw_mask & draw_pred).sum()
    draw_fp = (~draw_mask & draw_pred).sum()
    draw_fn = (draw_mask & ~draw_pred).sum()
    draw_precision = draw_tp / max(1, draw_tp + draw_fp)
    draw_recall = draw_tp / max(1, draw_tp + draw_fn)
    draw_f1 = 2 * draw_precision * draw_recall / max(0.001, draw_precision + draw_recall)

    # Calibration (ECE)
    pred_class = df[["prob_H", "prob_D", "prob_A"]].values.argmax(axis=1)
    confidence = df[["prob_H", "prob_D", "prob_A"]].values.max(axis=1)
    label_map_inv = {"H": 0, "D": 1, "A": 2}
    actual_class = df["actual"].map(label_map_inv).values
    correct_arr = (pred_class == actual_class).astype(float)

    ece = 0
    bins = np.linspace(0, 1, 11)
    for i in range(10):
        mask = (confidence >= bins[i]) & (confidence < bins[i + 1])
        if mask.sum() > 0:
            ece += abs(confidence[mask].mean() - correct_arr[mask].mean()) * mask.sum()
    ece = ece / max(1, len(confidence))

    return {
        "n_matches": n,
        "accuracy": round(accuracy * 100, 1),
        "draw_f1": round(draw_f1 * 100, 1),
        "ece": round(ece, 4),
        "avg_confidence": round(confidence.mean() * 100, 1),
        "home_win_rate": round((df["actual"] == "H").mean() * 100, 1),
        "draw_rate": round((df["actual"] == "D").mean() * 100, 1),
    }


def _generate_recommendations(segments: Dict) -> List[str]:
    """Generate actionable recommendations from segment analysis."""
    recs = []
    overall_acc = segments.get("OVERALL", {}).get("accuracy", 50)

    for name, stats in segments.items():
        if name == "OVERALL":
            continue
        acc = stats["accuracy"]
        n = stats["n_matches"]

        if n < 20:
            continue

        if acc >= overall_acc + 5:
            recs.append(f"INCREASE stake on {name}: {acc:.0f}% accuracy (+{acc - overall_acc:.0f}pp vs overall, n={n})")
        elif acc <= overall_acc - 5:
            recs.append(f"REDUCE stake on {name}: {acc:.0f}% accuracy ({acc - overall_acc:+.0f}pp vs overall, n={n})")

    return recs


def generate_full_report() -> Dict:
    """Generate combined execution + match-type report."""
    report = {
        "generated_at": datetime.now().isoformat(),
        "slippage": analyze_slippage(),
        "match_types": analyze_match_types(),
    }

    out_path = BETTING_DIR / "execution_audit_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Execution audit saved to %s", out_path)

    return report


def print_report(report: Dict = None):
    """Print formatted execution audit."""
    if report is None:
        report = generate_full_report()

    # Slippage
    slip = report["slippage"]
    if "error" not in slip:
        o = slip["overall"]
        print("=" * 70)
        print("  EXECUTION SLIPPAGE ANALYSIS")
        print(f"  {slip['total_bets']} settled bets | CLV: {o['avg_clv']:+.2f}% | "
              f"ROI: {o['actual_roi']:+.1f}% | Edge conversion: {o['edge_conversion']:.0f}%")
        print("=" * 70)

        print(f"\n  {'Market':<15} {'Bets':>5} {'Edge':>7} {'ROI':>7} {'Conv':>7} {'CLV':>7} {'WR':>6}")
        print(f"  {'-'*15} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*6}")
        for mkt, data in slip["by_market"].items():
            conv = f"{data['edge_conversion_pct']:.0f}%" if data['edge_conversion_pct'] else "n/a"
            clv = f"{data['avg_clv']:+.2f}%" if data['avg_clv'] is not None else "n/a"
            print(f"  {mkt:<15} {data['n_bets']:>5} {data['avg_edge']:>+6.1f}% "
                  f"{data['actual_roi']:>+6.1f}% {conv:>7} {clv:>7} {data['win_rate']:>5.0f}%")

        print(f"\n  {'Bookmaker':<25} {'Bets':>5} {'CLV':>7} {'CLV+':>6} {'ROI':>7}")
        print(f"  {'-'*25} {'-'*5} {'-'*7} {'-'*6} {'-'*7}")
        for bm, data in slip["by_bookmaker"].items():
            if data["n_bets"] < 3 or not bm:
                continue
            clv = f"{data['avg_clv']:+.2f}%" if data['avg_clv'] is not None else "  n/a"
            clv_rate = data.get('positive_clv_rate', 0) or 0
            roi = data.get('roi_pct', 0) or 0
            print(f"  {str(bm):<25} {data['n_bets']:>5} {clv:>7} "
                  f"{clv_rate:>5.0f}% {roi:>+6.1f}%")

    # Match types
    mt = report["match_types"]
    if "error" not in mt:
        print(f"\n{'=' * 70}")
        print(f"  MATCH-TYPE PERFORMANCE ({mt['total_predictions']} predictions)")
        print("=" * 70)

        print(f"\n  {'Segment':<25} {'N':>5} {'Acc':>6} {'Draw F1':>8} {'ECE':>7} {'Conf':>6}")
        print(f"  {'-'*25} {'-'*5} {'-'*6} {'-'*8} {'-'*7} {'-'*6}")
        for name, data in mt["segments"].items():
            marker = " *" if name == "OVERALL" else ""
            print(f"  {name:<25} {data['n_matches']:>5} {data['accuracy']:>5.1f}% "
                  f"{data['draw_f1']:>7.1f}% {data['ece']:>6.4f} {data['avg_confidence']:>5.1f}%{marker}")

        if mt["recommendations"]:
            print(f"\n  STAKING RECOMMENDATIONS:")
            for rec in mt["recommendations"]:
                emoji = "\u2705" if "INCREASE" in rec else "\u26a0\ufe0f"
                print(f"    {emoji} {rec}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Execution audit + match-type analysis")
    parser.add_argument("--slippage", action="store_true", help="Slippage only")
    parser.add_argument("--segments", action="store_true", help="Match segments only")
    args = parser.parse_args()

    if args.slippage:
        report = {"slippage": analyze_slippage(), "match_types": {"error": "skipped"}}
    elif args.segments:
        report = {"slippage": {"error": "skipped"}, "match_types": analyze_match_types()}
    else:
        report = generate_full_report()

    print_report(report)
