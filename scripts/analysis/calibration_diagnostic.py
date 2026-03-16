"""
Module: scripts/calibration_diagnostic.py
Purpose: Investigates calibration drift in specific confidence buckets by decomposing predictions by outcome, calibration stage, and method
Inputs:  data/features/features.parquet (season data with features and results)
Outputs: Console report showing accuracy vs confidence breakdown by bucket, method, and calibration stage
Called by: Manual CLI invocation for debugging calibration issues
Depends on: scripts.backtest_unified (prediction methods, ensemble logic)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse backtest infrastructure
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.analysis.backtest_unified import (
    BACKTEST_WEIGHTS,
    _get_ml_classifier,
    predict_ensemble,
    predict_factor_based,
    predict_market,
    predict_xg_poisson,
)
import scripts.analysis.backtest_unified as _bt


def load_season_data(season: str) -> pd.DataFrame:
    features_path = Path(__file__).resolve().parent.parent.parent / "data" / "features" / "features.parquet"
    df = pd.read_parquet(features_path)
    return df[df["season"] == season].copy()


def analyze_calibration_bucket(season: str = "2024-2025"):
    df = load_season_data(season)
    print(f"\n{'='*70}")
    print(f"CALIBRATION DIAGNOSTIC: {season} — 55-65% BUCKET")
    print(f"{'='*70}\n")

    ml_model = _get_ml_classifier()

    records = []

    for _, row in df.iterrows():
        actual = row["result"]

        # --- Full ensemble (with draw boost + temperature) ---
        pred = predict_ensemble(row)
        probs = [pred["prob_H"], pred["prob_D"], pred["prob_A"]]
        outcomes = ["H", "D", "A"]
        max_prob = max(probs)
        predicted = outcomes[np.argmax(probs)]

        # --- Raw ensemble (NO draw boost, NO temperature) ---
        raw_pred = _predict_raw_ensemble(row, ml_model)

        # --- Market implied ---
        market = predict_market(row)

        # --- Individual methods ---
        factor = predict_factor_based(row)
        xg = predict_xg_poisson(row)
        ml_probs = _get_ml_probs(row, ml_model)

        record = {
            "match": f"{row.get('home_team', '?')} vs {row.get('away_team', '?')}",
            "actual": actual,
            "predicted": predicted,
            "correct": predicted == actual,
            "max_prob": max_prob,
            "prob_H": pred["prob_H"],
            "prob_D": pred["prob_D"],
            "prob_A": pred["prob_A"],
            # Raw (pre-calibration)
            "raw_max": max(raw_pred.values()) if raw_pred else None,
            "raw_predicted": max(raw_pred, key=raw_pred.get) if raw_pred else None,
            # Market
            "market_H": market["prob_H"] if market else None,
            "market_D": market["prob_D"] if market else None,
            "market_A": market["prob_A"] if market else None,
            # Individual methods
            "factor_H": factor["prob_H"], "factor_D": factor["prob_D"], "factor_A": factor["prob_A"],
            "xg_H": xg["prob_H"] if xg else None,
            "ml_H": ml_probs.get("prob_H") if ml_probs else None,
            "ml_D": ml_probs.get("prob_D") if ml_probs else None,
            "ml_A": ml_probs.get("prob_A") if ml_probs else None,
        }
        records.append(record)

    # Filter to the 55-65% bucket
    bucket = [r for r in records if 0.55 <= r["max_prob"] < 0.65]
    all_records = records

    print(f"Total matches: {len(all_records)}")
    print(f"Matches in 55-65% bucket: {len(bucket)}")
    print(f"Bucket accuracy: {sum(r['correct'] for r in bucket)}/{len(bucket)} = {sum(r['correct'] for r in bucket)/len(bucket)*100:.1f}%")
    print(f"Avg predicted confidence: {np.mean([r['max_prob'] for r in bucket])*100:.1f}%")
    print()

    # --- Breakdown by predicted outcome ---
    print("BY PREDICTED OUTCOME:")
    print(f"  {'Pred':<6} {'Count':>6} {'Correct':>8} {'Accuracy':>9} {'Avg Conf':>9}")
    print("  " + "-" * 40)
    for outcome in ["H", "D", "A"]:
        subset = [r for r in bucket if r["predicted"] == outcome]
        if subset:
            correct = sum(r["correct"] for r in subset)
            print(f"  {outcome:<6} {len(subset):>6} {correct:>8} {correct/len(subset)*100:>8.1f}% {np.mean([r['max_prob'] for r in subset])*100:>8.1f}%")

    # --- Breakdown by actual outcome ---
    print(f"\nBY ACTUAL OUTCOME (what really happened):")
    print(f"  {'Actual':<6} {'Count':>6} {'We predicted it':>16}")
    print("  " + "-" * 40)
    for outcome in ["H", "D", "A"]:
        subset = [r for r in bucket if r["actual"] == outcome]
        predicted_it = sum(1 for r in subset if r["predicted"] == outcome)
        if subset:
            print(f"  {outcome:<6} {len(subset):>6} {predicted_it:>16} ({predicted_it/len(subset)*100:.0f}%)")

    # --- Confusion matrix for the bucket ---
    print(f"\nCONFUSION (predicted → actual):")
    print(f"  {'Predicted':<10} → {'H':>6} {'D':>6} {'A':>6}")
    print("  " + "-" * 35)
    for pred_out in ["H", "D", "A"]:
        subset = [r for r in bucket if r["predicted"] == pred_out]
        h_count = sum(1 for r in subset if r["actual"] == "H")
        d_count = sum(1 for r in subset if r["actual"] == "D")
        a_count = sum(1 for r in subset if r["actual"] == "A")
        print(f"  {pred_out:<10}   {h_count:>6} {d_count:>6} {a_count:>6}")

    # --- Pre vs Post calibration ---
    print(f"\nPRE vs POST CALIBRATION (draw boost + temperature):")
    raw_in_bucket = [r for r in all_records if r["raw_max"] and 0.55 <= r["raw_max"] < 0.65]
    raw_bucket_correct = sum(r["correct"] for r in raw_in_bucket) if raw_in_bucket else 0
    print(f"  Raw ensemble (no boost/temp) in 55-65%: {len(raw_in_bucket)} matches, "
          f"{raw_bucket_correct/len(raw_in_bucket)*100:.1f}% accuracy" if raw_in_bucket else "  No raw matches in bucket")

    # How many matches got pushed INTO the 55-65% bucket by calibration?
    pushed_in = [r for r in bucket if r["raw_max"] and not (0.55 <= r["raw_max"] < 0.65)]
    pushed_out = [r for r in all_records if r["raw_max"] and (0.55 <= r["raw_max"] < 0.65) and not (0.55 <= r["max_prob"] < 0.65)]
    print(f"  Pushed INTO 55-65% by calibration: {len(pushed_in)} matches")
    print(f"  Pushed OUT of 55-65% by calibration: {len(pushed_out)} matches")

    if pushed_in:
        pushed_in_correct = sum(r["correct"] for r in pushed_in)
        print(f"    Pushed-in accuracy: {pushed_in_correct}/{len(pushed_in)} = {pushed_in_correct/len(pushed_in)*100:.1f}%")
        # Where did they come from?
        from_below = [r for r in pushed_in if r["raw_max"] < 0.55]
        from_above = [r for r in pushed_in if r["raw_max"] >= 0.65]
        print(f"    From <55%: {len(from_below)}, From >=65%: {len(from_above)}")

    # --- ML vs Market in the bucket ---
    print(f"\nML vs MARKET in 55-65% bucket:")
    ml_avail = [r for r in bucket if r["ml_H"] is not None]
    market_avail = [r for r in bucket if r["market_H"] is not None]

    if ml_avail:
        # For each match in the bucket, compare ML confidence to ensemble
        ml_confs = []
        for r in ml_avail:
            ml_probs = [r["ml_H"], r["ml_D"], r["ml_A"]]
            ml_confs.append(max(ml_probs))
        print(f"  ML avg max-prob for bucket matches: {np.mean(ml_confs)*100:.1f}%")
        # ML alone accuracy on these matches
        ml_correct = 0
        for r in ml_avail:
            ml_probs = {"H": r["ml_H"], "D": r["ml_D"], "A": r["ml_A"]}
            ml_pred = max(ml_probs, key=ml_probs.get)
            if ml_pred == r["actual"]:
                ml_correct += 1
        print(f"  ML alone accuracy on bucket matches: {ml_correct}/{len(ml_avail)} = {ml_correct/len(ml_avail)*100:.1f}%")

    if market_avail:
        market_confs = []
        for r in market_avail:
            m_probs = [r["market_H"], r["market_D"], r["market_A"]]
            market_confs.append(max(m_probs))
        print(f"  Market avg max-prob for bucket matches: {np.mean(market_confs)*100:.1f}%")
        market_correct = 0
        for r in market_avail:
            m_probs = {"H": r["market_H"], "D": r["market_D"], "A": r["market_A"]}
            m_pred = max(m_probs, key=m_probs.get)
            if m_pred == r["actual"]:
                market_correct += 1
        print(f"  Market accuracy on bucket matches: {market_correct}/{len(market_avail)} = {market_correct/len(market_avail)*100:.1f}%")

    # --- Disagreement analysis ---
    print(f"\nDISAGREEMENT ANALYSIS (ensemble vs market):")
    agree = [r for r in bucket if r["market_H"] is not None]
    if agree:
        ensemble_agrees_market = 0
        ensemble_disagrees_correct = 0
        ensemble_disagrees_wrong = 0
        for r in agree:
            m_probs = {"H": r["market_H"], "D": r["market_D"], "A": r["market_A"]}
            m_pred = max(m_probs, key=m_probs.get)
            if r["predicted"] == m_pred:
                ensemble_agrees_market += 1
            else:
                if r["correct"]:
                    ensemble_disagrees_correct += 1
                else:
                    ensemble_disagrees_wrong += 1
        n_disagree = len(agree) - ensemble_agrees_market
        print(f"  Ensemble agrees with market: {ensemble_agrees_market}/{len(agree)}")
        print(f"  Disagrees: {n_disagree} — correct: {ensemble_disagrees_correct}, wrong: {ensemble_disagrees_wrong}")

    # --- List worst misses (high confidence, wrong) ---
    print(f"\nWORST MISSES (sorted by confidence, wrong predictions):")
    wrong = sorted([r for r in bucket if not r["correct"]], key=lambda r: -r["max_prob"])
    print(f"  {'Match':<30} {'Pred':>4} {'Act':>4} {'Conf':>6} {'Market fav':>10}")
    print("  " + "-" * 60)
    for r in wrong[:15]:
        m_fav = ""
        if r["market_H"] is not None:
            m_probs = {"H": r["market_H"], "D": r["market_D"], "A": r["market_A"]}
            m_fav = max(m_probs, key=m_probs.get)
        print(f"  {r['match']:<30} {r['predicted']:>4} {r['actual']:>4} {r['max_prob']*100:>5.1f}% {m_fav:>10}")

    # --- Temperature sensitivity ---
    print(f"\nTEMPERATURE SENSITIVITY:")
    for T in [0.70, 0.75, 0.810, 0.90, 1.00, 1.10]:
        in_bucket = 0
        bucket_correct = 0
        for r_idx, (_, row) in enumerate(df.iterrows()):
            raw = _predict_raw_ensemble(row, ml_model)
            if not raw:
                continue
            # Apply draw boost
            rH, rD, rA = raw["prob_H"], raw["prob_D"] * 1.321, raw["prob_A"]
            total = rH + rD + rA
            rH, rD, rA = rH/total, rD/total, rA/total
            # Apply temperature
            eps = 1e-10
            logits = [np.log(rH + eps), np.log(rD + eps), np.log(rA + eps)]
            scaled = [np.exp(l / T) for l in logits]
            s_total = sum(scaled)
            pH, pD, pA = scaled[0]/s_total, scaled[1]/s_total, scaled[2]/s_total
            max_p = max(pH, pD, pA)
            if 0.55 <= max_p < 0.65:
                in_bucket += 1
                predicted = ["H", "D", "A"][[pH, pD, pA].index(max_p)]
                if predicted == records[r_idx]["actual"]:
                    bucket_correct += 1
        acc = bucket_correct / in_bucket * 100 if in_bucket else 0
        marker = " ← current" if abs(T - 0.810) < 0.01 else ""
        print(f"  T={T:.3f}: {in_bucket:>3} matches in bucket, accuracy={acc:.1f}%{marker}")

    print()


def _predict_raw_ensemble(row, ml_model):
    """Ensemble prediction WITHOUT draw boost and temperature scaling."""
    weights = BACKTEST_WEIGHTS
    predictions = {}

    factor_probs = predict_factor_based(row)
    predictions["factor"] = factor_probs

    xg_probs = predict_xg_poisson(row)
    if xg_probs:
        predictions["xg"] = xg_probs

    market_probs = predict_market(row)
    if market_probs:
        predictions["market"] = market_probs

    if ml_model is not None:
        try:
            is_ens = _bt._ml_is_ensemble
            feature_names = ml_model.feature_names if is_ens else list(ml_model.feature_names_)
            available = [f for f in feature_names if f in row.index]
            if len(available) >= len(feature_names) * 0.5:
                X = pd.DataFrame([row[available].values], columns=available)
                for f in feature_names:
                    if f not in X.columns:
                        X[f] = 0
                X = X[feature_names]
                proba = ml_model.predict_proba(X)[0]
                eps = 1e-10
                logits = np.log(proba + eps)
                scaled = np.exp(logits / 0.40)
                proba = scaled / scaled.sum()
                predictions["ml"] = {
                    "prob_H": float(proba[0]),
                    "prob_D": float(proba[1]),
                    "prob_A": float(proba[2]),
                }
        except Exception:
            pass

    if xg_probs and "player_xg" in weights and weights.get("player_xg", 0) > 0:
        predictions["player_xg"] = xg_probs

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


def _get_ml_probs(row, ml_model):
    """Get raw ML probabilities (with T=0.40 sharpening)."""
    if ml_model is None:
        return None
    try:
        is_ens = _bt._ml_is_ensemble
        feature_names = ml_model.feature_names if is_ens else list(ml_model.feature_names_)
        available = [f for f in feature_names if f in row.index]
        if len(available) < len(feature_names) * 0.5:
            return None
        X = pd.DataFrame([row[available].values], columns=available)
        for f in feature_names:
            if f not in X.columns:
                X[f] = 0
        X = X[feature_names]
        proba = ml_model.predict_proba(X)[0]
        eps = 1e-10
        logits = np.log(proba + eps)
        scaled = np.exp(logits / 0.40)
        proba = scaled / scaled.sum()
        return {"prob_H": float(proba[0]), "prob_D": float(proba[1]), "prob_A": float(proba[2])}
    except Exception:
        return None


if __name__ == "__main__":
    season = sys.argv[1] if len(sys.argv) > 1 else "2024-2025"
    analyze_calibration_bucket(season)
