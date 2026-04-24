"""Dixon-Coles τ correction for independent Poisson joint goal distributions.

Classical paper: Dixon & Coles (1997), "Modelling association football scores
and inefficiencies in the football betting market."

Independent Poisson under-predicts the 0-0, 1-0, 0-1 cells. The τ correction
multiplies those cells (and 1-1) by:

    C(0,0) = 1 - λ_h * λ_a * τ
    C(1,0) = 1 + λ_a * τ
    C(0,1) = 1 + λ_h * τ
    C(1,1) = 1 - τ
    C(h,a) = 1               otherwise

Sample via accept-reject:
    1. Draw (h, a) from independent Poisson(λ_h), Poisson(λ_a)
    2. Compute p_accept = C(h, a) / max(C)
    3. Keep if uniform(0,1) < p_accept; else redraw

v3 default: fit τ pooled across all training seasons with weak shrinkage
toward 0 (when uncertain, τ=0 is the no-correction case). Per-season fits
are noisy; pooled with shrinkage has least variance in our sample.

From the 2026-04-21 diagnostic on Serie A 2024-25:
  predicted 3-cell rate 0.289 vs actual 0.261 → +2.8pp overshoot
  → expected fitted τ ∈ [0.02, 0.08]
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import poisson


def dc_correction(h: int, a: int, lambda_h: float, lambda_a: float, tau: float) -> float:
    """Dixon-Coles cell-weight multiplier. See module docstring."""
    if h == 0 and a == 0:
        return 1.0 - lambda_h * lambda_a * tau
    if h == 1 and a == 0:
        return 1.0 + lambda_a * tau
    if h == 0 and a == 1:
        return 1.0 + lambda_h * tau
    if h == 1 and a == 1:
        return 1.0 - tau
    return 1.0


def dc_joint_pmf(
    h_max: int,
    a_max: int,
    lambda_h: float,
    lambda_a: float,
    tau: float,
) -> np.ndarray:
    """Return the (h_max+1, a_max+1) joint PMF under Dixon-Coles."""
    ph = np.array([poisson.pmf(h, lambda_h) for h in range(h_max + 1)])
    pa = np.array([poisson.pmf(a, lambda_a) for a in range(a_max + 1)])
    pmf = np.outer(ph, pa)
    # Apply DC cell corrections to the 4 affected cells only
    if h_max >= 0 and a_max >= 0:
        pmf[0, 0] *= (1.0 - lambda_h * lambda_a * tau)
    if h_max >= 1 and a_max >= 0:
        pmf[1, 0] *= (1.0 + lambda_a * tau)
    if h_max >= 0 and a_max >= 1:
        pmf[0, 1] *= (1.0 + lambda_h * tau)
    if h_max >= 1 and a_max >= 1:
        pmf[1, 1] *= (1.0 - tau)
    # Renormalize (τ leaves total prob slightly off)
    pmf = np.clip(pmf, 0.0, None)
    pmf = pmf / pmf.sum()
    return pmf


def sample_dc_goals(
    lambda_h: np.ndarray,
    lambda_a: np.ndarray,
    tau: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized accept-reject sampling for (home, away) goals with DC τ.

    For each trial index i, draws (h, a) iid from Poisson and accepts with
    probability dc_correction(h, a, λh, λa, τ) / c_max. c_max = max over
    the 4 affected cells — bounded tightly.
    """
    if tau == 0.0:
        return rng.poisson(lambda_h), rng.poisson(lambda_a)

    n = len(lambda_h)
    h_out = np.zeros(n, dtype=np.int64)
    a_out = np.zeros(n, dtype=np.int64)

    # For each trial, find local c_max for bounded rejection. c_max depends on
    # (λh[i], λa[i]) and τ. Compute once per trial.
    c_00 = 1.0 - lambda_h * lambda_a * tau
    c_10 = 1.0 + lambda_a * tau
    c_01 = 1.0 + lambda_h * tau
    c_11 = np.full(n, 1.0 - tau)  # broadcast scalar → array
    # Element-wise max over the 4 cell-correction magnitudes + 1.0 baseline
    c_max = np.maximum(
        np.maximum(np.maximum(np.ones(n), np.abs(c_00)), np.maximum(np.abs(c_10), np.abs(c_01))),
        np.abs(c_11),
    )

    # Iterative accept-reject per trial (vectorized over pending mask)
    pending = np.ones(n, dtype=bool)
    max_iter = 30
    for _ in range(max_iter):
        idx = np.where(pending)[0]
        if len(idx) == 0:
            break
        h_draw = rng.poisson(lambda_h[idx])
        a_draw = rng.poisson(lambda_a[idx])
        # Compute correction for drawn cell
        corr = np.ones(len(idx))
        m00 = (h_draw == 0) & (a_draw == 0)
        m10 = (h_draw == 1) & (a_draw == 0)
        m01 = (h_draw == 0) & (a_draw == 1)
        m11 = (h_draw == 1) & (a_draw == 1)
        corr[m00] = c_00[idx][m00]
        corr[m10] = c_10[idx][m10]
        corr[m01] = c_01[idx][m01]
        corr[m11] = c_11[idx][m11]
        p_accept = np.clip(corr / c_max[idx], 0.0, 1.0)
        u = rng.random(len(idx))
        accept = u < p_accept
        h_out[idx[accept]] = h_draw[accept]
        a_out[idx[accept]] = a_draw[accept]
        pending[idx[accept]] = False
    # Fallback: accept whatever's left as-is
    if pending.any():
        leftover = np.where(pending)[0]
        h_out[leftover] = rng.poisson(lambda_h[leftover])
        a_out[leftover] = rng.poisson(lambda_a[leftover])
    return h_out, a_out


def fit_tau_mle(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    lambda_h: np.ndarray,
    lambda_a: np.ndarray,
    bounds: tuple[float, float] = (-0.25, 0.25),
    shrinkage_prior_var: float = 0.01,
) -> float:
    """MLE τ with Gaussian-prior shrinkage toward 0.

    Minimizes -log-likelihood + τ² / (2 * shrinkage_prior_var).
    Larger prior_var = less shrinkage.
    """
    home_goals = np.asarray(home_goals, dtype=int)
    away_goals = np.asarray(away_goals, dtype=int)
    lambda_h = np.asarray(lambda_h, dtype=float)
    lambda_a = np.asarray(lambda_a, dtype=float)
    assert home_goals.shape == away_goals.shape == lambda_h.shape == lambda_a.shape

    def _neg_log_posterior(tau: float) -> float:
        # likelihood = sum log(Poisson(h)*Poisson(a) * C(h,a,λ,τ))
        # Drop constant Poisson terms since tau only affects C
        logc = np.zeros(len(home_goals))
        m00 = (home_goals == 0) & (away_goals == 0)
        m10 = (home_goals == 1) & (away_goals == 0)
        m01 = (home_goals == 0) & (away_goals == 1)
        m11 = (home_goals == 1) & (away_goals == 1)
        logc[m00] = np.log(np.clip(1.0 - lambda_h[m00] * lambda_a[m00] * tau, 1e-6, None))
        logc[m10] = np.log(np.clip(1.0 + lambda_a[m10] * tau, 1e-6, None))
        logc[m01] = np.log(np.clip(1.0 + lambda_h[m01] * tau, 1e-6, None))
        logc[m11] = np.log(np.clip(1.0 - tau, 1e-6, None))
        neg_ll = -logc.sum()
        prior_penalty = tau ** 2 / (2.0 * shrinkage_prior_var)
        return neg_ll + prior_penalty

    res = minimize_scalar(_neg_log_posterior, bounds=bounds, method="bounded",
                          options={"xatol": 1e-5})
    return float(res.x)
