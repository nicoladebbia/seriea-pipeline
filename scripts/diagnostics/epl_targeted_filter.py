#!/usr/bin/env python3
"""Test a TARGETED bet filter for EPL based on the 2025-26 loss pattern.

Filter (designed from the loss-pattern diagnostic):
  ACCEPT bets where:
    - model_prob in [0.30, 0.40]  (mid-confidence, profitable +16%)
    - opening odds in [3.5, 5.0]  (the sweet spot, +18% ROI)
    - outcome NOT in {H}          (home picks lose -52%)
    - if outcome=A, require odds >= 4.0  (away at <4.0 odds: -78% ROI)

Validate: tune on EPL 2022-23 + 2023-24, validate on 2024-25, holdout on 2025-26.
The filter must be POSITIVE on holdout (real out-of-sample) to be adopted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def score_season(season: str, model_dir: str) -> pd.DataFrame:
    """Score every EPL match in `season` using the model in `model_dir`,
    return a per-bet log at 5% edge threshold."""
    matches = pd.read_parquet(PROJECT_ROOT / "data" / "parsed" / "matches.parquet")
    matches["match_date"] = pd.to_datetime(matches["match_date"])
    matches = matches[matches["league"] == "premier_league"].copy()
    matches = matches.dropna(subset=["home_score", "away_score"]).copy()
    matches = matches[matches["season"] == season].copy()

    feats = pd.read_parquet(PROJECT_ROOT / "data" / "features" / "features_premier_league.parquet")
    feats["match_date"] = pd.to_datetime(feats["match_date"])
    feats = feats[feats["season"] == season].copy()

    from catboost import CatBoostClassifier
    import pickle as pk

    md = PROJECT_ROOT / "data" / "models" / "walkforward" / "premier_league" / model_dir
    model = CatBoostClassifier()
    season_file = "season_2025-2026.cbm" if "fresh_2025" in model_dir else f"season_{season}.cbm"
    cal_file = season_file.replace(".cbm", "_calibrators.pkl")
    model.load_model(str(md / season_file))
    with open(md / cal_file, "rb") as f:
        cals = pk.load(f)

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
    rs = cal_proba.sum(axis=1, keepdims=True); rs[rs == 0] = 1
    cal_proba /= rs

    meta_path = md / season_file.replace(".cbm", "_metadata.json")
    if meta_path.exists():
        with open(meta_path) as f:
            spec_classes = json.load(f).get("classes", ["A", "D", "H"])
    else:
        spec_classes = ["A", "D", "H"]
    idx = {"A": spec_classes.index("A"), "D": spec_classes.index("D"), "H": spec_classes.index("H")}
    feats = feats.reset_index(drop=True)
    for o in ["A", "D", "H"]:
        feats[f"prob_{o}"] = cal_proba[:, idx[o]]

    EDGE = 0.05
    bets = []
    for _, row in matches.iterrows():
        f = feats[
            (feats["home_team"] == row["home_team"])
            & (feats["away_team"] == row["away_team"])
            & (feats["match_date"] == row["match_date"])
        ]
        if f.empty: continue
        f = f.iloc[0]
        probs = {"H": float(f["prob_H"]), "D": float(f["prob_D"]), "A": float(f["prob_A"])}
        odds = {
            "H": float(row["odds_MaxH"]) if pd.notna(row.get("odds_MaxH")) else 0,
            "D": float(row["odds_MaxD"]) if pd.notna(row.get("odds_MaxD")) else 0,
            "A": float(row["odds_MaxA"]) if pd.notna(row.get("odds_MaxA")) else 0,
        }
        if max(odds.values()) <= 1.01: continue
        best = None
        for o in ["H", "D", "A"]:
            if odds[o] <= 1.01: continue
            edge = probs[o] - 1.0/odds[o]
            if edge < EDGE: continue
            if best is None or edge > best[3]:
                best = (o, probs[o], odds[o], edge)
        if best is None: continue
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
            "season": season, "matchweek": int(row.get("matchweek", 0)) if pd.notna(row.get("matchweek")) else 0,
            "outcome": outcome, "actual": actual, "won": won,
            "model_p": p, "odds": op_odds, "edge": edge, "stake": stake, "pnl": pnl,
        })
    return pd.DataFrame(bets)


def apply_filter(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Apply a named filter."""
    if name == "no_filter":
        return df
    if name == "v1_drop_home":
        # Reject H picks
        return df[df["outcome"] != "H"]
    if name == "v2_drop_home_and_low_odds":
        # Reject H picks AND odds < 2.5
        return df[(df["outcome"] != "H") & (df["odds"] >= 2.5)]
    if name == "v3_sweet_spot":
        # Mid-prob, mid-odds, drop home, away requires odds>=4
        keep = df[
            (df["outcome"] != "H")
            & (df["model_p"] >= 0.30) & (df["model_p"] < 0.40)
            & (df["odds"] >= 3.5) & (df["odds"] < 5.0)
        ]
        return keep
    if name == "v4_targeted":
        # Targeted: drop H, drop very low/high odds, drop low edge picks too
        keep = df[
            (df["outcome"] != "H")
            & (df["odds"] >= 2.5) & (df["odds"] < 5.0)
            & (df["model_p"] >= 0.30) & (df["model_p"] < 0.45)
        ]
        return keep
    if name == "v5_loose_targeted":
        # Slightly looser: drop H, drop tiny odds + huge odds
        keep = df[
            (df["outcome"] != "H")
            & (df["odds"] >= 2.5) & (df["odds"] < 6.0)
        ]
        return keep
    return df


def report(df: pd.DataFrame, label: str):
    if len(df) == 0:
        print(f"  {label:<35} n=0")
        return
    n = len(df); w = int(df["won"].sum())
    st = df["stake"].sum(); pl = df["pnl"].sum()
    roi = (pl/st*100) if st > 0 else 0
    print(f"  {label:<35} n={n:>4} win={w/n*100:>5.1f}% ROI={roi:>+7.2f}% P/L=€{pl:>+8.2f}")


def main():
    print("Scoring all EPL seasons with appropriate models...")
    # Score 2022-2025 with their walkforward fold models, 2025-26 with fresh
    seasons = {
        "2022-2023": "1x2__5season_seed42",
        "2023-2024": "1x2__5season_seed42",
        "2024-2025": "1x2__5season_seed42",
        "2025-2026": "1x2__fresh_2025_seed42",
    }
    all_bets = []
    for s, mdir in seasons.items():
        try:
            df = score_season(s, mdir)
            print(f"  {s}: {len(df)} bets generated")
            all_bets.append(df)
        except Exception as e:
            print(f"  {s}: ERROR {e}")
    if not all_bets:
        return 1
    bets = pd.concat(all_bets, ignore_index=True)

    discovery = bets[bets["season"].isin(["2022-2023", "2023-2024"])]
    validation = bets[bets["season"] == "2024-2025"]
    holdout = bets[bets["season"] == "2025-2026"]

    filters = ["no_filter", "v1_drop_home", "v2_drop_home_and_low_odds",
               "v3_sweet_spot", "v4_targeted", "v5_loose_targeted"]
    print()
    print("="*82)
    print("EPL TARGETED FILTER SEARCH")
    print("="*82)
    print(f"{'filter':<35}{'discovery':<24}{'validation':<24}{'HOLDOUT':<24}")
    print("-"*82)
    for fname in filters:
        d = apply_filter(discovery, fname); v = apply_filter(validation, fname); h = apply_filter(holdout, fname)
        def fmt(x):
            if len(x) == 0: return "n=0"
            roi = (x["pnl"].sum()/x["stake"].sum()*100) if x["stake"].sum() > 0 else 0
            return f"n={len(x):>3} ROI={roi:>+6.1f}%"
        print(f"  {fname:<33} {fmt(d):<22} {fmt(v):<22} {fmt(h):<22}")

    print()
    print("="*82)
    print("DETAIL: each filter on holdout 2025-26")
    print("="*82)
    for fname in filters:
        h = apply_filter(holdout, fname)
        report(fname + " (HOLDOUT)", h)

    return 0


if __name__ == "__main__":
    sys.exit(main())
