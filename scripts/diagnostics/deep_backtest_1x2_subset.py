"""Sub-strategy analysis: which bet types have the real edge?

Breaks down the Market_Max result at edge>=10% by:
  - bet class (H / D / A)
  - odds band (favorites 1.5-2.5, mid 2.5-4.0, longshots 4.0+)

Goal: identify which subset drives CLV and ROI, so we know what to bet.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoost
import pickle
from numpy.random import default_rng

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LEAGUES = {
    "serie_a": PROJECT_ROOT / "data" / "features" / "features_serie_a.parquet",
    "premier_league": PROJECT_ROOT / "data" / "features" / "features_premier_league.parquet",
}
CLASSES = ("H","D","A")
BOOT = 5000


def load_proba(league: str, eval_df: pd.DataFrame) -> np.ndarray:
    model_dir = PROJECT_ROOT / "data" / "models" / "walkforward" / league / "1x2"
    out = np.full((len(eval_df), 3), np.nan)
    for season, grp in eval_df.groupby("season", sort=False):
        mp = model_dir / f"season_{season}.cbm"
        cp = model_dir / f"season_{season}_calibrators.pkl"
        if not mp.exists():
            continue
        m = CatBoost(); m.load_model(str(mp))
        feats = list(m.feature_names_)
        X = grp.copy()
        for f in [f for f in feats if f not in X.columns]:
            X[f] = 0.0
        X = X[feats].fillna(0.0)
        p = m.predict(X, prediction_type="Probability")
        if cp.exists():
            with open(cp, "rb") as fh:
                cals = pickle.load(fh)
            cols = [cals[i].predict(p[:, i]) for i in range(3)]
            cal = np.column_stack(cols)
            cal = cal / cal.sum(axis=1, keepdims=True)
            p = np.clip(cal, 1e-6, 1-1e-6)
            p = p / p.sum(axis=1, keepdims=True)
        pos = [eval_df.index.get_loc(i) for i in grp.index]
        for k, pp in zip(pos, p):
            out[k] = pp
    return out


def bootstrap(returns: np.ndarray, stakes: np.ndarray, seed: int = 42) -> tuple:
    if stakes.sum() == 0 or len(returns) == 0:
        return (float("nan"),) * 3
    rng = default_rng(seed)
    n = len(returns)
    boots = np.empty(BOOT)
    for i in range(BOOT):
        idx = rng.integers(0, n, n)
        s = stakes[idx].sum()
        boots[i] = (returns[idx].sum() / s * 100.0) if s > 0 else 0.0
    return float(returns.sum()/stakes.sum()*100.0), float(np.percentile(boots,2.5)), float(np.percentile(boots,97.5))


for lg, path in LEAGUES.items():
    print(f"\n{'='*90}")
    print(f"{lg} — subset analysis on Market_Max, edge>=7%")
    print('='*90)
    df = pd.read_parquet(path)
    df = df[df["season"].isin(["2022-2023","2023-2024","2024-2025"])].copy()
    df = df.dropna(subset=["home_score","away_score"]).sort_values(["season","match_date"]).reset_index(drop=True)
    hs = df["home_score"].astype(int).values; asc = df["away_score"].astype(int).values
    df["_target"] = np.where(hs>asc, 0, np.where(hs==asc, 1, 2))

    proba = load_proba(lg, df)
    mask = ~np.isnan(proba).any(axis=1)
    df = df[mask].reset_index(drop=True); proba = proba[mask]

    oh = df["odds_MaxH"].values; od = df["odds_MaxD"].values; oa = df["odds_MaxA"].values
    # Also closing for CLV
    ch = df["odds_PS_close_H"].values; cd = df["odds_PS_close_D"].values; ca = df["odds_PS_close_A"].values
    o = np.column_stack([oh,od,oa]); c = np.column_stack([ch,cd,ca])
    inv = 1/o; fair = inv / inv.sum(axis=1, keepdims=True)
    edges = (proba - fair)/fair*100
    best_class = edges.argmax(axis=1); best_edge = edges[np.arange(len(edges)), best_class]
    best_odds = o[np.arange(len(o)), best_class]; close_odds = c[np.arange(len(c)), best_class]
    best_p = proba[np.arange(len(proba)), best_class]
    y = df["_target"].values
    won = (best_class == y).astype(int)

    sel = best_edge >= 7.0
    print(f"  Global n={sel.sum()}  win_rate={won[sel].mean()*100:.1f}%")

    # Break down by class
    print(f"\n{'Subset':40s} {'n':>5s} {'WR%':>5s} {'avg_odds':>9s} {'ROI%':>7s} {'CI':>22s} {'CLV%':>6s}")
    for cl in range(3):
        submask = sel & (best_class == cl)
        n = submask.sum()
        if n == 0: continue
        wr = won[submask].mean()*100
        ao = best_odds[submask].mean()
        profits = np.where(won[submask]==1, (best_odds[submask]-1)*10, -10)
        stakes = np.full(n, 10.0)
        roi, lo, hi = bootstrap(profits, stakes)
        clv = ((best_odds[submask] / close_odds[submask]) - 1).mean() * 100
        print(f"  bet class = {CLASSES[cl]:38s}".ljust(42) + f"{n:>5d} {wr:>5.1f} {ao:>9.2f} {roi:>+6.2f} [{lo:+6.2f},{hi:+6.2f}] {clv:>+5.2f}")

    # Break down by odds band
    print()
    bands = [("fav (≤2.2)", 0, 2.2), ("mid (2.2-4.0)", 2.2, 4.0), ("dog (4.0-7.0)", 4.0, 7.0), ("longshot (>7.0)", 7.0, 1000)]
    for lbl, lo_b, hi_b in bands:
        submask = sel & (best_odds >= lo_b) & (best_odds < hi_b)
        n = submask.sum()
        if n == 0: continue
        wr = won[submask].mean()*100
        ao = best_odds[submask].mean()
        profits = np.where(won[submask]==1, (best_odds[submask]-1)*10, -10)
        stakes = np.full(n, 10.0)
        roi, lo, hi = bootstrap(profits, stakes)
        clv = ((best_odds[submask] / close_odds[submask]) - 1).mean() * 100
        print(f"  odds band {lbl:38s}".ljust(42) + f"{n:>5d} {wr:>5.1f} {ao:>9.2f} {roi:>+6.2f} [{lo:+6.2f},{hi:+6.2f}] {clv:>+5.2f}")

    # Break down by our model probability band
    print()
    pbands = [("low (0-0.30)", 0, 0.30), ("mid (0.30-0.50)", 0.30, 0.50), ("high (0.50-0.70)", 0.50, 0.70), ("very_high (>0.70)", 0.70, 1.01)]
    for lbl, lo_b, hi_b in pbands:
        submask = sel & (best_p >= lo_b) & (best_p < hi_b)
        n = submask.sum()
        if n == 0: continue
        wr = won[submask].mean()*100
        ao = best_odds[submask].mean()
        profits = np.where(won[submask]==1, (best_odds[submask]-1)*10, -10)
        stakes = np.full(n, 10.0)
        roi, lo, hi = bootstrap(profits, stakes)
        clv = ((best_odds[submask] / close_odds[submask]) - 1).mean() * 100
        print(f"  our_p band {lbl:38s}".ljust(42) + f"{n:>5d} {wr:>5.1f} {ao:>9.2f} {roi:>+6.2f} [{lo:+6.2f},{hi:+6.2f}] {clv:>+5.2f}")
