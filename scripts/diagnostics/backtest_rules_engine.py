"""End-to-end backtest of the betting_rules.json strategy.

Applies the rules engine to every match in the walk-forward eval set, collects
only the bets the engine accepts, and reports ROI + CLV. This confirms that
the declarative rules (JSON + engine) reproduce the edge we found in the
deep-dive analysis.

No iteration — runs once, publishes numbers.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoost
import pickle
from numpy.random import default_rng

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Same env-var pattern as deep_backtest_1x2.py for variant comparisons.
_WALKFORWARD_SUFFIX = os.environ.get("WALKFORWARD_SUFFIX", "")
_USE_RAW_PROBA = os.environ.get("USE_RAW_PROBA", "").strip() in ("1", "true", "yes")

from config.betting_rules import BetCandidate, BettingRules  # noqa: E402

LEAGUES = {
    "serie_a": PROJECT_ROOT / "data" / "features" / "features_serie_a.parquet",
    "premier_league": PROJECT_ROOT / "data" / "features" / "features_premier_league.parquet",
}
CLASS_NAMES = ("H", "D", "A")
BANKROLL = 1000.0


def load_proba(league: str, eval_df: pd.DataFrame) -> np.ndarray:
    market_dir = f"1x2__{_WALKFORWARD_SUFFIX}" if _WALKFORWARD_SUFFIX else "1x2"
    model_dir = PROJECT_ROOT / "data" / "models" / "walkforward" / league / market_dir
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
        if cp.exists() and not _USE_RAW_PROBA:
            with open(cp, "rb") as fh:
                cals = pickle.load(fh)
            cols = [cals[i].predict(p[:, i]) for i in range(3)]
            cal = np.column_stack(cols); cal = cal / cal.sum(axis=1, keepdims=True)
            p = np.clip(cal, 1e-6, 1-1e-6); p = p / p.sum(axis=1, keepdims=True)
        pos = [eval_df.index.get_loc(i) for i in grp.index]
        for k, pp in zip(pos, p):
            out[k] = pp
    return out


def deover(odds_triplet: np.ndarray) -> np.ndarray:
    inv = 1 / odds_triplet
    return inv / inv.sum(axis=1, keepdims=True)


def bootstrap(returns: np.ndarray, stakes: np.ndarray, n: int = 5000, seed: int = 42) -> tuple:
    if stakes.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    rng = default_rng(seed)
    boots = np.empty(n)
    nn = len(returns)
    for i in range(n):
        idx = rng.integers(0, nn, nn)
        s = stakes[idx].sum()
        boots[i] = returns[idx].sum() / s * 100 if s > 0 else 0.0
    return float(returns.sum() / stakes.sum() * 100.0), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def run() -> None:
    rules = BettingRules()
    print(f"Rules version: {rules.version}")
    print(f"Preferred odds source: {rules.preferred_odds_source()}")
    print(f"Kelly fraction: {rules.kelly_fraction()}, max bet pct: {rules.max_bet_pct()}")
    print()

    for lg, path in LEAGUES.items():
        df = pd.read_parquet(path)
        df = df[df["season"].isin(["2022-2023", "2023-2024", "2024-2025"])].copy()
        df = df.dropna(subset=["home_score", "away_score"])
        df = df.sort_values(["season", "match_date"]).reset_index(drop=True)
        hs = df["home_score"].astype(int).values; asc = df["away_score"].astype(int).values
        df["_target"] = np.where(hs > asc, 0, np.where(hs == asc, 1, 2))

        proba = load_proba(lg, df)
        mask = ~np.isnan(proba).any(axis=1)
        df = df[mask].reset_index(drop=True); proba = proba[mask]

        odds_mx = df[["odds_MaxH", "odds_MaxD", "odds_MaxA"]].values.astype(float)
        odds_close_pin = df[["odds_PS_close_H", "odds_PS_close_D", "odds_PS_close_A"]].values.astype(float)
        fair_mx = deover(odds_mx)

        # Iterate every (row, class) and let rules engine decide.
        accepted: list[dict] = []
        rejected_reasons: dict[str, int] = {}
        for i in range(len(df)):
            for cls in range(3):
                if np.isnan(odds_mx[i, cls]) or np.isnan(proba[i, cls]):
                    continue
                cand = BetCandidate(
                    league=lg, market="1X2", selection=CLASS_NAMES[cls],
                    model_prob=float(proba[i, cls]),
                    odds=float(odds_mx[i, cls]),
                    odds_source="market_max",
                    fair_implied_prob=float(fair_mx[i, cls]),
                )
                dec = rules.evaluate(cand, bankroll=BANKROLL, live_bets_for_market=999)
                # live_bets=999 disables paper-trade gate for backtest purposes (we
                # want the raw, full-stake numbers here).
                if dec.accept:
                    won = int(df["_target"].iloc[i] == cls)
                    stake = dec.stake_fraction_of_bankroll * BANKROLL
                    profit = (cand.odds - 1) * stake if won else -stake
                    clv = (cand.odds / odds_close_pin[i, cls] - 1) * 100.0 if not np.isnan(odds_close_pin[i, cls]) else np.nan
                    accepted.append({
                        "class": CLASS_NAMES[cls], "odds": cand.odds, "prob": cand.model_prob,
                        "edge_pct": dec.edge_pct, "stake": stake, "profit": profit, "won": won, "clv": clv,
                    })
                else:
                    # Bucket reason by first phrase for summary
                    key = dec.reason.split(":")[0].split(" ")[0:3]
                    key = " ".join(key)
                    rejected_reasons[key] = rejected_reasons.get(key, 0) + 1

        print(f"=== {lg} ===")
        print(f"  total candidates evaluated: {len(df) * 3}")
        print(f"  accepted bets: {len(accepted)}")
        if not accepted:
            print("  no bets; reasons:")
            for k, v in sorted(rejected_reasons.items(), key=lambda x: -x[1])[:10]:
                print(f"    {v:>5d}: {k}")
            continue

        adf = pd.DataFrame(accepted)
        profits = adf["profit"].values
        stakes = adf["stake"].values
        wrate = adf["won"].mean() * 100
        roi_kelly, lo_k, hi_k = bootstrap(profits, stakes)

        # Also flat-stake ROI for comparison
        flat_profits = np.where(adf["won"].values == 1, (adf["odds"].values - 1) * 10, -10)
        flat_stakes = np.full(len(adf), 10.0)
        roi_flat, lo_f, hi_f = bootstrap(flat_profits, flat_stakes)

        mean_clv = np.nanmean(adf["clv"].values)
        clv_pos_rate = (adf["clv"] > 0).mean() * 100

        print(f"  win_rate: {wrate:.1f}%  avg_odds: {adf['odds'].mean():.2f}")
        print(f"  FLAT ROI (€10/bet):  {roi_flat:+.2f}%  CI [{lo_f:+.2f}, {hi_f:+.2f}]")
        print(f"  KELLY ROI (0.25x):   {roi_kelly:+.2f}%  CI [{lo_k:+.2f}, {hi_k:+.2f}]  mean_stake={stakes.mean():.2f}€")
        print(f"  CLV: mean {mean_clv:+.2f}%  CLV+_rate: {clv_pos_rate:.0f}%")
        # Breakdown by class
        for cl in CLASS_NAMES:
            sub = adf[adf["class"] == cl]
            if len(sub) == 0:
                continue
            wr = sub["won"].mean() * 100
            flat = np.where(sub["won"].values == 1, (sub["odds"].values - 1) * 10, -10).sum() / (len(sub) * 10) * 100
            print(f"    {cl}: n={len(sub):4d} wr={wr:.1f}% avg_odds={sub['odds'].mean():.2f} flat_ROI={flat:+.2f}%")
        print()


if __name__ == "__main__":
    run()
