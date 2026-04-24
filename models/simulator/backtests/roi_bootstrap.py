"""Bootstrap CI, Sharpe, max drawdown, longest losing streak.

Given a list of (stake, profit) tuples from a backtest, computes point-estimate
ROI plus a 95% bootstrap CI over `n_resample` resamples. Also reports:

- Sharpe: mean(profit / stake) / std(profit / stake) × sqrt(n)  — crude, but
  comparable across markets when n matches.
- Max drawdown: worst peak-to-trough loss in cumulative bankroll curve.
- Longest losing streak: max consecutive bets with profit < 0.

Reproducibility: the RNG seed is derived from the input vector length + a
caller-supplied salt so re-running the same backtest yields identical CIs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ROIStats:
    n_bets: int
    total_stake_eur: float
    total_profit_eur: float
    roi_pct_point: float
    roi_pct_ci_lower: float | None
    roi_pct_ci_upper: float | None
    sharpe: float | None
    max_drawdown_eur: float
    max_drawdown_pct_of_stake: float | None
    longest_losing_streak: int
    max_single_bet_edge_share: float | None  # single bet's share of total profit


def compute_roi_stats(
    stakes: np.ndarray,
    profits: np.ndarray,
    n_resample: int = 1000,
    alpha: float = 0.05,
    seed_salt: int = 0,
) -> ROIStats:
    stakes = np.asarray(stakes, dtype=float)
    profits = np.asarray(profits, dtype=float)
    assert stakes.shape == profits.shape

    n = int(len(stakes))
    if n == 0:
        return ROIStats(0, 0.0, 0.0, 0.0, None, None, None, 0.0, None, 0, None)

    total_stake = float(stakes.sum())
    total_profit = float(profits.sum())
    roi_pt = (total_profit / total_stake * 100.0) if total_stake > 0 else 0.0

    # Bootstrap CI
    rng = np.random.default_rng(n * 1_000_003 + seed_salt)
    lower = upper = None
    if n >= 5 and total_stake > 0:
        idx = rng.integers(0, n, size=(n_resample, n))
        resamp_profit = profits[idx].sum(axis=1)
        resamp_stake = stakes[idx].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            resamp_roi = np.where(resamp_stake > 0, resamp_profit / resamp_stake * 100.0, 0.0)
        lower = float(np.quantile(resamp_roi, alpha / 2))
        upper = float(np.quantile(resamp_roi, 1 - alpha / 2))

    # Sharpe (crude): mean/std of per-bet return
    per_bet_return = np.where(stakes > 0, profits / stakes, 0.0)
    sharpe = None
    if n >= 2:
        std = float(per_bet_return.std(ddof=1))
        if std > 0:
            sharpe = float(per_bet_return.mean() / std * np.sqrt(n))

    # Drawdown
    cumulative = np.cumsum(profits)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_dd = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0
    max_dd_pct = (max_dd / total_stake * 100.0) if total_stake > 0 else None

    # Longest losing streak
    losing = (profits < 0).astype(int)
    streak = 0
    longest = 0
    for v in losing:
        if v == 1:
            streak += 1
            longest = max(longest, streak)
        else:
            streak = 0

    # Edge concentration: single-bet share of total profit (positive profit only)
    positive_profits = profits[profits > 0]
    max_single_share = None
    if len(positive_profits) > 0 and positive_profits.sum() > 0:
        max_single_share = float(positive_profits.max() / positive_profits.sum())

    return ROIStats(
        n_bets=n,
        total_stake_eur=round(total_stake, 2),
        total_profit_eur=round(total_profit, 2),
        roi_pct_point=round(roi_pt, 2),
        roi_pct_ci_lower=round(lower, 2) if lower is not None else None,
        roi_pct_ci_upper=round(upper, 2) if upper is not None else None,
        sharpe=round(sharpe, 3) if sharpe is not None else None,
        max_drawdown_eur=round(max_dd, 2),
        max_drawdown_pct_of_stake=round(max_dd_pct, 2) if max_dd_pct is not None else None,
        longest_losing_streak=longest,
        max_single_bet_edge_share=round(max_single_share, 3) if max_single_share is not None else None,
    )
