"""Shared Poisson math utilities for 1X2, over/under, and goal-based models."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
from scipy.stats import poisson


def poisson_probability(k: int, lambda_: float) -> float:
    """Calculate Poisson probability P(X = k) given rate lambda."""
    if lambda_ <= 0:
        return 1.0 if k == 0 else 0.0
    return (math.exp(-lambda_) * (lambda_ ** k)) / math.factorial(k)


def poisson_cumulative(k: int, lambda_: float) -> float:
    """Calculate cumulative Poisson P(X <= k)."""
    return sum(poisson_probability(i, lambda_) for i in range(k + 1))


def poisson_1x2(hxg: float, axg: float, min_xg: float | None = None) -> np.ndarray:
    """Convert xG to 1X2 probabilities via Poisson.

    Args:
        hxg: Home expected goals.
        axg: Away expected goals.
        min_xg: Optional minimum xG clamp (e.g. 0.3).

    Returns:
        np.ndarray of shape (3,): [P(Home), P(Draw), P(Away)].
    """
    if min_xg is not None:
        hxg = max(hxg, min_xg)
        axg = max(axg, min_xg)
    pH = pD = pA = 0.0
    for h in range(8):
        for a in range(8):
            p = poisson.pmf(h, hxg) * poisson.pmf(a, axg)
            if h > a:
                pH += p
            elif h == a:
                pD += p
            else:
                pA += p
    t = pH + pD + pA
    if t > 0:
        return np.array([pH / t, pD / t, pA / t])
    return np.array([0.4, 0.3, 0.3])


def poisson_1x2_vec(hxg_arr: np.ndarray, axg_arr: np.ndarray,
                     min_xg: float | None = 0.3) -> np.ndarray:
    """Batch-vectorized Poisson 1X2 probabilities.

    Args:
        hxg_arr: Array of home xG values.
        axg_arr: Array of away xG values.
        min_xg: Optional minimum xG clamp (default 0.3).

    Returns:
        np.ndarray of shape (n, 3): [P(Home), P(Draw), P(Away)] per match.
    """
    n = len(hxg_arr)
    probs = np.zeros((n, 3))
    for i in range(n):
        probs[i] = poisson_1x2(hxg_arr[i], axg_arr[i], min_xg=min_xg)
    return probs


def market_implied_probs(df: "pd.DataFrame", mask: "pd.Series") -> np.ndarray:
    """Extract market-implied 1X2 probabilities from Pinnacle odds columns.

    Converts odds_PSH / odds_PSD / odds_PSA into normalised probabilities.
    Rows where any odds value is <= 1 (missing / invalid) get a flat prior
    [0.4, 0.3, 0.3].

    Args:
        df: DataFrame containing odds_PSH, odds_PSD, odds_PSA columns.
        mask: Boolean mask selecting rows from *df*.

    Returns:
        np.ndarray of shape (mask.sum(), 3): [P(Home), P(Draw), P(Away)].
    """
    psh = df.loc[mask, "odds_PSH"].values
    psd = df.loc[mask, "odds_PSD"].values
    psa = df.loc[mask, "odds_PSA"].values
    n = mask.sum()
    probs = np.zeros((n, 3))
    for i in range(n):
        if psh[i] > 1 and psd[i] > 1 and psa[i] > 1:
            raw = np.array([1 / psh[i], 1 / psd[i], 1 / psa[i]])
            probs[i] = raw / raw.sum()
        else:
            probs[i] = [0.4, 0.3, 0.3]
    return probs


def calculate_over_probability(threshold: float, expected: float) -> float:
    """Calculate P(X > threshold) using Poisson distribution.

    For a 2.5 line: P(X >= 3) = 1 - P(X <= 2).
    """
    k = int(threshold)
    return 1 - poisson_cumulative(k, expected)


def poisson_win_prob(home_xg: float, away_xg: float,
                     max_goals: int = 10,
                     min_xg: float = 0.3,
                     max_xg: float = 4.0) -> Dict[str, float]:
    """Calculate H/D/A probabilities from expected goals.

    Clamps xG to [min_xg, max_xg] range. Returns dict with 'H', 'D', 'A' keys.
    """
    home_xg = max(min_xg, min(max_xg, home_xg))
    away_xg = max(min_xg, min(max_xg, away_xg))

    home_probs = [poisson.pmf(g, home_xg) for g in range(max_goals)]
    away_probs = [poisson.pmf(g, away_xg) for g in range(max_goals)]

    prob_home = prob_draw = prob_away = 0.0
    for hg in range(max_goals):
        for ag in range(max_goals):
            p = home_probs[hg] * away_probs[ag]
            if hg > ag:
                prob_home += p
            elif hg == ag:
                prob_draw += p
            else:
                prob_away += p

    total = prob_home + prob_draw + prob_away
    if total > 0:
        return {"H": prob_home / total, "D": prob_draw / total, "A": prob_away / total}
    return {"H": 0.4, "D": 0.3, "A": 0.3}
