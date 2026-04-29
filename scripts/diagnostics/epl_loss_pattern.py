#!/usr/bin/env python3
"""Diagnose WHICH EPL 2025-26 bets are dragging ROI to -22%.

Score every EPL 2025-26 match with the fresh model, generate paper-trade bets
at 5% edge threshold, label each as won/lost, then slice ROI by:
  - outcome class (H/D/A picked)
  - model probability range
  - opening odds range
  - matchweek
  - calibrator confidence (raw vs cal divergence)
  - home team strength tier
  - away team strength tier

Print a table of subsets sorted by total P&L impact.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    matches = pd.read_parquet(PROJECT_ROOT / "data" / "parsed" / "matches.parquet")
    matches["match_date"] = pd.to_datetime(matches["match_date"])
    matches = matches[matches["league"] == "premier_league"].copy()
    matches = matches.dropna(subset=["home_score", "away_score"]).copy()
    matches = matches[matches["season"] == "2025-2026"].copy()

    feats = pd.read_parquet(PROJECT_ROOT / "data" / "features" / "features_premier_league.parquet")
    feats["match_date"] = pd.to_datetime(feats["match_date"])
    feats = feats[feats["season"] == "2025-2026"].copy()

    from catboost import CatBoostClassifier
    import pickle as pk
    from ml.feature_selection import exclude_odds
    from scripts.models.train_walkforward import LEAKY_COLUMNS, META_COLUMNS

    model_dir = PROJECT_ROOT / "data" / "models" / "walkforward" / "premier_league" / "1x2__fresh_2025_seed42"
    model = CatBoostClassifier()
    model.load_model(str(model_dir / "season_2025-2026.cbm"))
    with open(model_dir / "season_2025-2026_calibrators.pkl", "rb") as f:
        cals = pk.load(f)

    numeric = feats.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in numeric if c not in META_COLUMNS and c not in LEAKY_COLUMNS and not c.startswith("_")]
    feat_cols = [f for f in feat_cols if not feats[feat_cols].isna().all(axis=0).get(f, False)]
    _, feat_cols = exclude_odds(feats[feat_cols].copy(), feat_cols)
    model_features = list(model.feature_names_)
    feat_cols = [f for f in model_features if f in feats.columns]

    X = feats[feat_cols]
    proba_raw = model.predict_proba(X)
    classes = list(model.classes_)
    cal_proba = np.zeros_like(proba_raw)
    for i, cls in enumerate(classes):
        if str(cls) in cals:
            cal_proba[:, i] = cals[str(cls)].transform(proba_raw[:, i])
        elif cls in cals:
            cal_proba[:, i] = cals[cls].transform(proba_raw[:, i])
        else:
            cal_proba[:, i] = proba_raw[:, i]
    rs = cal_proba.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
    cal_proba /= rs

    meta_path = model_dir / "season_2025-2026_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        spec_classes = meta.get("classes", ["A", "D", "H"])
    else:
        spec_classes = ["A", "D", "H"]
    idx_a = spec_classes.index("A")
    idx_d = spec_classes.index("D")
    idx_h = spec_classes.index("H")

    feats = feats.reset_index(drop=True)
    feats["prob_H"] = cal_proba[:, idx_h]
    feats["prob_D"] = cal_proba[:, idx_d]
    feats["prob_A"] = cal_proba[:, idx_a]
    feats["raw_H"] = proba_raw[:, idx_h]
    feats["raw_D"] = proba_raw[:, idx_d]
    feats["raw_A"] = proba_raw[:, idx_a]

    EDGE = 0.05
    bets = []
    for _, row in matches.iterrows():
        f = feats[
            (feats["home_team"] == row["home_team"])
            & (feats["away_team"] == row["away_team"])
            & (feats["match_date"] == row["match_date"])
        ]
        if f.empty:
            continue
        f = f.iloc[0]
        probs = {"H": float(f["prob_H"]), "D": float(f["prob_D"]), "A": float(f["prob_A"])}
        raws = {"H": float(f["raw_H"]), "D": float(f["raw_D"]), "A": float(f["raw_A"])}
        odds = {
            "H": float(row["odds_MaxH"]) if pd.notna(row.get("odds_MaxH")) else 0,
            "D": float(row["odds_MaxD"]) if pd.notna(row.get("odds_MaxD")) else 0,
            "A": float(row["odds_MaxA"]) if pd.notna(row.get("odds_MaxA")) else 0,
        }
        if max(odds.values()) <= 1.01:
            continue
        best = None
        for o in ["H", "D", "A"]:
            if odds[o] <= 1.01: continue
            edge = probs[o] - 1.0/odds[o]
            if edge < EDGE: continue
            if best is None or edge > best[3]:
                best = (o, probs[o], odds[o], edge)
        if best is None:
            continue
        outcome, p, op_odds, edge = best
        b = op_odds - 1.0
        full_kelly = (b * p - (1-p))/b
        if full_kelly <= 0: continue
        stake_frac = min(full_kelly * 0.25, 0.02)
        stake = stake_frac * 1000.0
        if stake < 0.01: continue
        h_score, a_score = int(row["home_score"]), int(row["away_score"])
        actual = "H" if h_score > a_score else "A" if h_score < a_score else "D"
        won = (outcome == actual)
        pnl = stake * b if won else -stake
        bets.append({
            "match_date": row["match_date"].date().isoformat(),
            "home": row["home_team"], "away": row["away_team"],
            "matchweek": int(row.get("matchweek", 0)) if pd.notna(row.get("matchweek")) else 0,
            "outcome": outcome, "actual": actual, "won": won,
            "model_p": p, "raw_p": raws[outcome], "cal_drift": p - raws[outcome],
            "odds": op_odds, "edge": edge, "stake": stake, "pnl": pnl,
            "prob_H": probs["H"], "prob_D": probs["D"], "prob_A": probs["A"],
            "odds_H": odds["H"], "odds_D": odds["D"], "odds_A": odds["A"],
            "h_score": h_score, "a_score": a_score,
        })

    if not bets:
        print("No bets generated"); return 1

    df = pd.DataFrame(bets)
    n = len(df); wins = int(df["won"].sum())
    total_stake = df["stake"].sum(); total_pnl = df["pnl"].sum()
    print(f"\n{'='*70}\nEPL 2025-26 BASELINE — n={n}, wins={wins} ({wins/n*100:.1f}%), ROI={total_pnl/total_stake*100:+.2f}%, P/L=€{total_pnl:+.2f}\n{'='*70}")

    def slice_print(label, sub):
        if len(sub) == 0:
            print(f"  {label:<40} n=0")
            return
        n_s = len(sub); w_s = int(sub["won"].sum())
        st = sub["stake"].sum(); pl = sub["pnl"].sum()
        roi = (pl/st*100) if st > 0 else 0
        print(f"  {label:<40} n={n_s:>3} win={w_s/n_s*100:>5.1f}% ROI={roi:>+7.2f}% P/L=€{pl:+8.2f}")

    print("\n— BY OUTCOME PICKED —")
    for o in ["H", "D", "A"]:
        slice_print(f"pick={o}", df[df["outcome"]==o])

    print("\n— BY MODEL PROB BAND —")
    for lo, hi in [(0.30, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 0.75), (0.75, 1.0)]:
        slice_print(f"prob {lo:.2f}-{hi:.2f}", df[(df["model_p"] >= lo) & (df["model_p"] < hi)])

    print("\n— BY OPENING ODDS BAND —")
    for lo, hi in [(1.5, 2.0), (2.0, 2.5), (2.5, 3.5), (3.5, 5.0), (5.0, 100)]:
        slice_print(f"odds {lo}-{hi}", df[(df["odds"] >= lo) & (df["odds"] < hi)])

    print("\n— BY EDGE BAND —")
    for lo, hi in [(0.05, 0.08), (0.08, 0.12), (0.12, 0.20), (0.20, 1.0)]:
        slice_print(f"edge {lo*100:.0f}-{hi*100:.0f}%", df[(df["edge"] >= lo) & (df["edge"] < hi)])

    print("\n— BY MATCHWEEK BAND —")
    for lo, hi in [(1, 12), (13, 26), (27, 38)]:
        slice_print(f"mw {lo}-{hi}", df[(df["matchweek"] >= lo) & (df["matchweek"] <= hi)])

    print("\n— BY CALIBRATION DRIFT (cal - raw) —")
    for lo, hi, name in [(-1, -0.05, "cal_lower(<-5pp)"), (-0.05, 0.05, "cal_neutral"), (0.05, 1, "cal_higher(>+5pp)")]:
        slice_print(f"{name}", df[(df["cal_drift"] >= lo) & (df["cal_drift"] < hi)])

    print("\n— OUTCOME × ODDS BAND (worst losers first) —")
    grid = []
    for o in ["H", "D", "A"]:
        for lo, hi in [(1.5, 2.5), (2.5, 4.0), (4.0, 100)]:
            sub = df[(df["outcome"]==o) & (df["odds"] >= lo) & (df["odds"] < hi)]
            if len(sub) == 0: continue
            pl = sub["pnl"].sum(); st = sub["stake"].sum()
            roi = (pl/st*100) if st > 0 else 0
            grid.append((f"pick={o} odds={lo}-{hi}", len(sub), int(sub["won"].sum()), roi, pl))
    grid.sort(key=lambda x: x[4])  # worst pnl first
    for label, n_s, w, roi, pl in grid:
        print(f"  {label:<40} n={n_s:>3} wins={w:>3} ROI={roi:>+7.2f}% P/L=€{pl:+8.2f}")

    # Save full bet log
    out = PROJECT_ROOT / "data" / "diagnostics" / "epl_2025_26_bets.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nFull log: {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
