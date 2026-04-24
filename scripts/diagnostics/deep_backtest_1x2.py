"""Deep backtest for 1X2 walk-forward models vs multiple odds sources.

Uses the calibrated walk-forward CatBoost models (the ones we trained today)
and backtests them against FIVE odds sources:

  1. B365 opening   — soft book, stale line, what you'd bet early
  2. B365 closing   — soft book, closer to consensus
  3. Pinnacle open  — sharp book, early liquidity
  4. Pinnacle close — sharp book, consensus sharp price (bench mark)
  5. Market MAX     — best available across all books (Odds Portal-style)

Reports per-threshold, per-source:
  - N bets, win rate, point ROI
  - 95% bootstrap CI on ROI (non-parametric, 5,000 resamples)
  - Kelly staking ROI variant (fractional 0.25 Kelly, capped 2% bankroll)
  - CLV: did the B365 open → B365 close line move in our favor?
         (Positive CLV is strong evidence of real edge, independent of ROI sample noise)

NO iteration. Backtest runs once, numbers are what they are.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.random import default_rng
from catboost import CatBoost
import pickle

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Allow temporary variant comparisons without overwriting prod models.
# WALKFORWARD_SUFFIX=2017plus   → loads from 1x2__2017plus/ instead of 1x2/
# USE_RAW_PROBA=1               → skips calibrator application (for debugging
#                                  or when calibration is known to degrade)
_WALKFORWARD_SUFFIX = os.environ.get("WALKFORWARD_SUFFIX", "")
_USE_RAW_PROBA = os.environ.get("USE_RAW_PROBA", "").strip() in ("1", "true", "yes")

LEAGUES = {
    "serie_a":        PROJECT_ROOT / "data" / "features" / "features_serie_a.parquet",
    "premier_league": PROJECT_ROOT / "data" / "features" / "features_premier_league.parquet",
}

ODDS_SOURCES = {
    "B365_open":   ("odds_B365H", "odds_B365D", "odds_B365A"),
    "B365_close":  ("odds_B365_close_H", "odds_B365_close_D", "odds_B365_close_A"),
    "Pin_open":    ("odds_PSH", "odds_PSD", "odds_PSA"),
    "Pin_close":   ("odds_PS_close_H", "odds_PS_close_D", "odds_PS_close_A"),
    "Market_Max":  ("odds_MaxH", "odds_MaxD", "odds_MaxA"),
}

EDGE_THRESHOLDS = [0.0, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0]
EVAL_SEASONS = ["2022-2023", "2023-2024", "2024-2025"]
CLASSES = ("H", "D", "A")
BANKROLL = 1000.0  # starting bankroll for Kelly simulation
FLAT_STAKE = 10.0
KELLY_FRAC = 0.25  # fractional Kelly — empirical literature favors 0.10-0.25
KELLY_MAX_PCT = 0.02  # never stake more than 2% of current bankroll on one bet
KELLY_MIN_PCT = 0.001  # ignore bets below 0.1% (noise)


def load_walkforward_proba(league: str, market: str, eval_df: pd.DataFrame) -> np.ndarray:
    """Apply the per-season walk-forward models to eval_df's rows, producing H/D/A probs.

    Dispatches each row to the model trained on strictly prior seasons, then
    applies per-class isotonic calibration saved alongside.
    """
    market_dir_name = f"{market}__{_WALKFORWARD_SUFFIX}" if _WALKFORWARD_SUFFIX else market
    model_dir = PROJECT_ROOT / "data" / "models" / "walkforward" / league / market_dir_name
    n_classes = 3
    out = np.full((len(eval_df), n_classes), np.nan, dtype=float)
    for season, grp in eval_df.groupby("season", sort=False):
        model_path = model_dir / f"season_{season}.cbm"
        cal_path = model_dir / f"season_{season}_calibrators.pkl"
        if not model_path.exists():
            continue
        m = CatBoost()
        m.load_model(str(model_path))
        feats = list(m.feature_names_)
        X = grp.copy()
        missing = [f for f in feats if f not in X.columns]
        for f in missing:
            X[f] = 0.0
        X = X[feats].fillna(0.0)
        proba = m.predict(X, prediction_type="Probability")
        # Apply calibrators if present AND not explicitly disabled.
        if cal_path.exists() and not _USE_RAW_PROBA:
            with open(cal_path, "rb") as fh:
                cals = pickle.load(fh)
            if len(cals) == n_classes:
                cols = [cals[ci].predict(proba[:, ci]) for ci in range(n_classes)]
                cal = np.column_stack(cols)
                row_sums = cal.sum(axis=1, keepdims=True)
                row_sums = np.where(row_sums > 0, row_sums, 1.0)
                cal = cal / row_sums
                cal = np.clip(cal, 1e-6, 1 - 1e-6)
                proba = cal / cal.sum(axis=1, keepdims=True)
        idx_positions = [eval_df.index.get_loc(i) for i in grp.index]
        for pos, row_probs in zip(idx_positions, proba):
            out[pos] = row_probs
    return out


def deoverround(odds_triplet: np.ndarray) -> np.ndarray:
    """Proportional overround stripping — fast, close to power method for low margins."""
    inv = 1.0 / odds_triplet
    ov = inv.sum(axis=1, keepdims=True)
    return inv / ov


def bootstrap_ci(returns: np.ndarray, stakes: np.ndarray, n_boot: int = 5000,
                 seed: int = 42) -> tuple[float, float, float]:
    """Bootstrap CI for ROI = profit / stake, in percentage points."""
    if len(returns) == 0 or stakes.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    rng = default_rng(seed)
    n = len(returns)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r = returns[idx].sum()
        s = stakes[idx].sum()
        boots[i] = (r / s * 100.0) if s > 0 else 0.0
    point = returns.sum() / stakes.sum() * 100.0
    return float(point), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def kelly_stake_fraction(p: float, odds: float) -> float:
    """Full Kelly fraction of bankroll. b = odds - 1 is net return."""
    b = odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - p
    f = (b * p - q) / b
    return max(0.0, f)


def run_league(league: str, features_path: Path) -> dict:
    df = pd.read_parquet(features_path)
    df = df[df["season"].isin(EVAL_SEASONS)].copy()
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df = df.sort_values(["season", "match_date"]).reset_index(drop=True)

    # Target
    hs = df["home_score"].astype(int).values
    a_s = df["away_score"].astype(int).values
    df["_target"] = np.where(hs > a_s, 0, np.where(hs == a_s, 1, 2))

    # Load walkforward + calibrated probabilities
    proba = load_walkforward_proba(league, "1x2", df)
    mask_valid = ~np.isnan(proba).any(axis=1)
    df = df[mask_valid].reset_index(drop=True)
    proba = proba[mask_valid]

    results: list[dict] = []

    for source_name, cols in ODDS_SOURCES.items():
        o = df[list(cols)].values.astype(float)
        have_odds = ~np.isnan(o).any(axis=1)
        if not have_odds.any():
            continue
        sub_idx = np.where(have_odds)[0]
        o_sub = o[sub_idx]
        p_sub = proba[sub_idx]
        y_sub = df["_target"].values[sub_idx]

        fair = deoverround(o_sub)
        # edge per class: (our_p - fair) / fair * 100
        edges = (p_sub - fair) / fair * 100.0

        # Pick the best class per row (highest edge).
        best_class = edges.argmax(axis=1)
        best_edge = edges[np.arange(len(edges)), best_class]
        best_odds = o_sub[np.arange(len(o_sub)), best_class]
        best_p = p_sub[np.arange(len(p_sub)), best_class]
        won = (best_class == y_sub).astype(int)

        # CLV: for non-B365 sources we don't have a comparable close-open pair.
        # For B365_open we use B365_close as the "closing line".
        # For Market_Max / Pin_open we use Pin_close as the consensus close.
        if source_name == "B365_open":
            close_cols = ODDS_SOURCES["B365_close"]
        else:
            close_cols = ODDS_SOURCES["Pin_close"]
        close_o = df[list(close_cols)].values.astype(float)[sub_idx]
        close_best = close_o[np.arange(len(close_o)), best_class]
        # CLV positive if our taken odds are BETTER than closing (odds higher = we got better price).
        # Convert to percent: (our_odds / closing_odds) - 1
        clv_pct = (best_odds / close_best - 1.0) * 100.0

        for thresh in EDGE_THRESHOLDS:
            sel = best_edge >= thresh
            n = int(sel.sum())
            if n == 0:
                continue
            odds_sel = best_odds[sel]
            won_sel = won[sel]
            p_sel = best_p[sel]

            # Flat stake ROI
            flat_returns = np.where(won_sel == 1, (odds_sel - 1) * FLAT_STAKE, -FLAT_STAKE)
            flat_stakes = np.full(n, FLAT_STAKE)
            roi_point, ci_lo, ci_hi = bootstrap_ci(flat_returns, flat_stakes)

            # Fractional Kelly stake ROI — apply to each bet independently using starting bankroll
            kelly_fracs = np.array([kelly_stake_fraction(p_sel[i], odds_sel[i]) for i in range(n)])
            kelly_fracs *= KELLY_FRAC
            kelly_fracs = np.clip(kelly_fracs, 0.0, KELLY_MAX_PCT)
            kelly_stakes = kelly_fracs * BANKROLL
            # Zero out bets below the min stake (noise filter)
            kelly_stakes = np.where(kelly_stakes >= (KELLY_MIN_PCT * BANKROLL), kelly_stakes, 0.0)
            n_kelly_active = int((kelly_stakes > 0).sum())
            kelly_returns = np.where(won_sel == 1, (odds_sel - 1) * kelly_stakes, -kelly_stakes)
            if kelly_stakes.sum() > 0:
                kelly_roi, kelly_lo, kelly_hi = bootstrap_ci(kelly_returns, kelly_stakes)
            else:
                kelly_roi, kelly_lo, kelly_hi = float("nan"), float("nan"), float("nan")

            # CLV: mean CLV of picks at this threshold
            clv_sel = clv_pct[sel]
            clv_mean = float(clv_sel.mean())
            clv_positive_rate = float((clv_sel > 0).mean())

            # Draw / Home / Away breakdown
            n_H = int(((best_class == 0) & sel).sum())
            n_D = int(((best_class == 1) & sel).sum())
            n_A = int(((best_class == 2) & sel).sum())

            results.append({
                "league": league,
                "odds_source": source_name,
                "edge_threshold_pct": thresh,
                "n_bets": n,
                "n_H": n_H, "n_D": n_D, "n_A": n_A,
                "win_rate": round(float(won_sel.mean()), 4),
                "avg_odds": round(float(odds_sel.mean()), 3),
                "avg_our_prob": round(float(p_sel.mean()), 4),
                "flat_roi_pct": round(roi_point, 2),
                "flat_ci_lower": round(ci_lo, 2),
                "flat_ci_upper": round(ci_hi, 2),
                "kelly_n_active": n_kelly_active,
                "kelly_roi_pct": round(kelly_roi, 2),
                "kelly_ci_lower": round(kelly_lo, 2),
                "kelly_ci_upper": round(kelly_hi, 2),
                "mean_clv_pct": round(clv_mean, 3),
                "clv_positive_rate": round(clv_positive_rate, 3),
            })

    return {"league": league, "rows": results}


def main() -> None:
    out: dict = {"leagues": {}}
    for lg, path in LEAGUES.items():
        print(f"\n=== {lg} ===")
        rep = run_league(lg, path)
        out["leagues"][lg] = rep

        # Pretty print
        print(f"{'source':12s} {'thresh':>6s} {'n':>5s} {'H/D/A':>11s} {'WR%':>5s} {'avg_odds':>9s} "
              f"{'flat_ROI':>9s} {'CI_lo':>7s} {'CI_hi':>7s} {'kelly_ROI':>10s} {'kelly_lo':>9s} "
              f"{'kelly_hi':>9s} {'CLV%':>6s} {'CLV+':>5s}")
        for r in rep["rows"]:
            print(f"{r['odds_source']:12s} {r['edge_threshold_pct']:>6.1f} {r['n_bets']:>5d} "
                  f"{r['n_H']:>3d}/{r['n_D']:>3d}/{r['n_A']:>3d} {r['win_rate']*100:>5.1f} "
                  f"{r['avg_odds']:>9.2f} {r['flat_roi_pct']:>+9.2f} {r['flat_ci_lower']:>+7.2f} "
                  f"{r['flat_ci_upper']:>+7.2f} {r['kelly_roi_pct']:>+10.2f} "
                  f"{r['kelly_ci_lower']:>+9.2f} {r['kelly_ci_upper']:>+9.2f} "
                  f"{r['mean_clv_pct']:>+6.2f} {r['clv_positive_rate']*100:>4.0f}%")

    out_path = PROJECT_ROOT / "data" / "diagnostics" / "deep_backtest_1x2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
