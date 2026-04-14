#!/usr/bin/env python3
"""Extended betting markets generator.

Computes additional bet types for every match using existing Poisson models:
- Double Chance (1X, X2, 12)
- Team Totals (Home/Away O/U 0.5, 1.5, 2.5)
- First Half 1X2 + O/U 0.5, 1.5
- Exact Score (top 10 most likely)
- Multi-Goal Ranges (0-1, 0-2, 1-3, 2-3, 2-4, 3+, 4+)
- Half-Time/Full-Time (9 combinations)

All probabilities derived from the Poisson distribution using ensemble xG.
"""

import json
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime

log = logging.getLogger(__name__)
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR


def poisson_pmf(k: int, lam: float) -> float:
    """Poisson probability mass function P(X=k)."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def poisson_cdf(k: int, lam: float) -> float:
    """Cumulative Poisson P(X <= k)."""
    return sum(poisson_pmf(i, lam) for i in range(k + 1))


def score_matrix(home_xg: float, away_xg: float, max_goals: int = 7,
                  rho: float = 0.0) -> dict:
    """Build probability matrix for (home_goals, away_goals).

    When rho > 0, uses bivariate Poisson internally for correlated scores.
    Backward compatible: rho=0 gives the original independent Poisson.
    """
    if rho > 0:
        params = BivariatePoissonParams(
            lambda_home=home_xg, lambda_away=away_xg, rho=rho
        )
        mat = bivariate_score_matrix(params, max_goals=max_goals)
        m = {}
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                m[(h, a)] = float(mat[h, a])
        return m
    m = {}
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            m[(h, a)] = poisson_pmf(h, home_xg) * poisson_pmf(a, away_xg)
    return m


# ---------------------------------------------------------------------------
# Bivariate Poisson (Karlis-Ntzoufras decomposition)
# ---------------------------------------------------------------------------

@dataclass
class BivariatePoissonParams:
    """Parameters for bivariate Poisson model.

    Uses the Karlis-Ntzoufras decomposition:
        X = X_ind + Z,  Y = Y_ind + Z
    where X_ind ~ Poisson(lambda_home - lambda_3),
          Y_ind ~ Poisson(lambda_away - lambda_3),
          Z     ~ Poisson(lambda_3),
          lambda_3 = rho * sqrt(lambda_home * lambda_away).

    rho controls the positive correlation between home and away goals.
    Typical Serie A value: 0.05-0.15 (goals are slightly positively correlated
    because game state influences both teams — open games produce more goals
    for both sides, defensive games suppress both).
    """
    lambda_home: float
    lambda_away: float
    rho: float = 0.10

    @property
    def lambda_3(self) -> float:
        """Shared Poisson component (covariance driver)."""
        raw = self.rho * math.sqrt(max(self.lambda_home, 0.01)
                                   * max(self.lambda_away, 0.01))
        # Clamp: lambda_3 can't exceed either marginal rate
        return max(0.0, min(raw, self.lambda_home - 0.01,
                            self.lambda_away - 0.01))

    @property
    def lambda_home_ind(self) -> float:
        return max(0.01, self.lambda_home - self.lambda_3)

    @property
    def lambda_away_ind(self) -> float:
        return max(0.01, self.lambda_away - self.lambda_3)


def bivariate_score_matrix(
    params: BivariatePoissonParams,
    max_goals: int = 7,
    draw_calibration: Optional[float] = None,
) -> np.ndarray:
    """Exact NxN probability matrix via bivariate Poisson PMF.

    The bivariate Poisson PMF for (x, y) is:
        P(X=x, Y=y) = exp(-(l1+l2+l3)) * (l1^x / x!) * (l2^y / y!)
                       * sum_{k=0}^{min(x,y)} C(x,k)*C(y,k)*k! * (l3/(l1*l2))^k

    where l1 = lambda_home_ind, l2 = lambda_away_ind, l3 = lambda_3.

    draw_calibration: if set, inflate diagonal by this factor (e.g. 1.05)
                      and renormalize. None means no adjustment.
    """
    n = max_goals + 1
    l1 = params.lambda_home_ind
    l2 = params.lambda_away_ind
    l3 = params.lambda_3

    matrix = np.zeros((n, n))

    # Pre-compute Poisson PMFs for independent components
    pmf_home = np.array([poisson_pmf(k, l1) for k in range(n)])
    pmf_away = np.array([poisson_pmf(k, l2) for k in range(n)])

    for x in range(n):
        for y in range(n):
            s = 0.0
            for k in range(min(x, y) + 1):
                if l3 == 0 and k > 0:
                    break
                # C(x,k) * C(y,k) * k! * (l3 / (l1 * l2))^k
                binom_x = math.comb(x, k)
                binom_y = math.comb(y, k)
                if l1 > 0 and l2 > 0:
                    ratio = (l3 / (l1 * l2)) ** k
                elif k == 0:
                    ratio = 1.0
                else:
                    ratio = 0.0
                s += binom_x * binom_y * math.factorial(k) * ratio
            matrix[x, y] = pmf_home[x] * pmf_away[y] * s

    # Apply draw calibration if requested
    if draw_calibration is not None and draw_calibration != 1.0:
        for i in range(n):
            matrix[i, i] *= draw_calibration

    # Renormalize (small numerical drift from clamping / draw calibration)
    total = matrix.sum()
    if total > 0:
        matrix /= total

    return matrix


def sample_scores(
    params: BivariatePoissonParams,
    n_samples: int = 10000,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized MC sampling via Karlis-Ntzoufras decomposition.

    Returns (home_goals[n_samples], away_goals[n_samples]).
    """
    if rng is None:
        rng = np.random.default_rng()

    z = rng.poisson(params.lambda_3, size=n_samples)
    x_ind = rng.poisson(params.lambda_home_ind, size=n_samples)
    y_ind = rng.poisson(params.lambda_away_ind, size=n_samples)

    return x_ind + z, y_ind + z


def derive_market_outcomes(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
) -> dict[str, np.ndarray]:
    """Derive boolean arrays for all markets from score samples.

    Returns dict mapping outcome keys to boolean arrays of length n_samples.
    """
    total = home_goals + away_goals
    margin = home_goals - away_goals  # positive = home wins

    return {
        # H2H
        "h2h_home": margin > 0,
        "h2h_draw": margin == 0,
        "h2h_away": margin < 0,
        # BTTS
        "btts_yes": (home_goals > 0) & (away_goals > 0),
        "btts_no": (home_goals == 0) | (away_goals == 0),
        # Totals
        "totals_over_1.5": total > 1.5,
        "totals_under_1.5": total < 1.5,
        "totals_over_2.5": total > 2.5,
        "totals_under_2.5": total < 2.5,
        "totals_over_3.5": total > 3.5,
        "totals_under_3.5": total < 3.5,
        "totals_over_0.5": total > 0.5,
        "totals_under_0.5": total < 0.5,
        # Team totals (home)
        "team_totals_home_over_0.5": home_goals > 0.5,
        "team_totals_home_under_0.5": home_goals < 0.5,
        "team_totals_home_over_1.5": home_goals > 1.5,
        "team_totals_home_under_1.5": home_goals < 1.5,
        "team_totals_home_over_2.5": home_goals > 2.5,
        "team_totals_home_under_2.5": home_goals < 2.5,
        # Team totals (away)
        "team_totals_away_over_0.5": away_goals > 0.5,
        "team_totals_away_under_0.5": away_goals < 0.5,
        "team_totals_away_over_1.5": away_goals > 1.5,
        "team_totals_away_under_1.5": away_goals < 1.5,
        "team_totals_away_over_2.5": away_goals > 2.5,
        "team_totals_away_under_2.5": away_goals < 2.5,
        # Double chance
        "double_chance_1x": margin >= 0,
        "double_chance_x2": margin <= 0,
        "double_chance_12": margin != 0,
        # Spreads (common lines)
        "spreads_home_-0.5": margin > 0.5,
        "spreads_home_-1": margin > 1,
        "spreads_home_-1.5": margin > 1.5,
        "spreads_home_-2.5": margin > 2.5,
        "spreads_home_+0.5": margin > -0.5,
        "spreads_home_+1": margin > -1,
        "spreads_home_+1.5": margin > -1.5,
        "spreads_away_-0.5": margin < -0.5,
        "spreads_away_-1": margin < -1,
        "spreads_away_-1.5": margin < -1.5,
        # Multi-goal ranges
        "multi_goal_0-1": total <= 1,
        "multi_goal_0-2": total <= 2,
        "multi_goal_1-3": (total >= 1) & (total <= 3),
        "multi_goal_2-3": (total >= 2) & (total <= 3),
        "multi_goal_2-4": (total >= 2) & (total <= 4),
        "multi_goal_3+": total >= 3,
        "multi_goal_4+": total >= 4,
    }


def calibrate_rho(features_path: Optional[Path] = None) -> float:
    """Fit rho from historical home/away goal covariance in features.parquet.

    rho = Cov(home_goals, away_goals) / sqrt(Var(home) * Var(away))
    Clamped to [0, 0.25] — negative correlation is rare and likely noise.
    """
    import pandas as pd

    if features_path is None:
        features_path = DATA_DIR / "features" / "features.parquet"
    if not features_path.exists():
        return 0.10  # sensible default

    df = pd.read_parquet(features_path)

    # Look for goal columns
    home_col = None
    away_col = None
    for c in ["home_goals", "home_score", "FTHG"]:
        if c in df.columns:
            home_col = c
            break
    for c in ["away_goals", "away_score", "FTAG"]:
        if c in df.columns:
            away_col = c
            break

    if home_col is None or away_col is None:
        return 0.10

    hg = df[home_col].dropna()
    ag = df[away_col].dropna()

    # Align indexes
    common = hg.index.intersection(ag.index)
    hg = hg.loc[common].values.astype(float)
    ag = ag.loc[common].values.astype(float)

    if len(hg) < 100:
        return 0.10

    cov = np.cov(hg, ag)[0, 1]
    var_h = np.var(hg)
    var_a = np.var(ag)

    if var_h <= 0 or var_a <= 0:
        return 0.10

    rho = cov / math.sqrt(var_h * var_a)
    return float(np.clip(rho, 0.0, 0.25))


def compute_double_chance(probs: dict) -> dict:
    """1X, X2, 12 from existing 1X2 probabilities."""
    h = probs.get("home", 0.33)
    d = probs.get("draw", 0.33)
    a = probs.get("away", 0.33)
    return {
        "1X": {"prob": round(h + d, 4), "fair_odds": round(1 / (h + d), 2) if (h + d) > 0 else 99},
        "X2": {"prob": round(d + a, 4), "fair_odds": round(1 / (d + a), 2) if (d + a) > 0 else 99},
        "12": {"prob": round(h + a, 4), "fair_odds": round(1 / (h + a), 2) if (h + a) > 0 else 99},
    }


def compute_team_totals(home_xg: float, away_xg: float) -> dict:
    """Home/Away over/under 0.5, 1.5, 2.5."""
    result = {}
    for label, xg in [("home", home_xg), ("away", away_xg)]:
        lines = {}
        for line in [0.5, 1.5, 2.5]:
            k = int(line)
            over_p = 1 - poisson_cdf(k, xg)
            under_p = poisson_cdf(k, xg)
            lines[f"over_{line}"] = {
                "prob": round(over_p, 4),
                "fair_odds": round(1 / over_p, 2) if over_p > 0.001 else 99,
            }
            lines[f"under_{line}"] = {
                "prob": round(under_p, 4),
                "fair_odds": round(1 / under_p, 2) if under_p > 0.001 else 99,
            }
        result[label] = lines
    return result


def compute_first_half(home_xg: float, away_xg: float) -> dict:
    """First-half predictions. Scale full-match xG by 0.43 (empirical 1H share)."""
    fh_home = home_xg * 0.43
    fh_away = away_xg * 0.43

    mat = score_matrix(fh_home, fh_away, max_goals=5)

    # 1X2
    p_home = sum(v for (h, a), v in mat.items() if h > a)
    p_draw = sum(v for (h, a), v in mat.items() if h == a)
    p_away = sum(v for (h, a), v in mat.items() if h < a)

    # O/U 0.5 and 1.5
    total_xg = fh_home + fh_away
    over_05 = 1 - poisson_cdf(0, total_xg)
    over_15 = 1 - poisson_cdf(1, total_xg)

    def safe_odds(p):
        return round(1 / p, 2) if p > 0.001 else 99

    return {
        "home_xg_1h": round(fh_home, 2),
        "away_xg_1h": round(fh_away, 2),
        "result_1x2": {
            "home": {"prob": round(p_home, 4), "fair_odds": safe_odds(p_home)},
            "draw": {"prob": round(p_draw, 4), "fair_odds": safe_odds(p_draw)},
            "away": {"prob": round(p_away, 4), "fair_odds": safe_odds(p_away)},
        },
        "over_under": {
            "over_0.5": {"prob": round(over_05, 4), "fair_odds": safe_odds(over_05)},
            "under_0.5": {"prob": round(1 - over_05, 4), "fair_odds": safe_odds(1 - over_05)},
            "over_1.5": {"prob": round(over_15, 4), "fair_odds": safe_odds(over_15)},
            "under_1.5": {"prob": round(1 - over_15, 4), "fair_odds": safe_odds(1 - over_15)},
        },
    }


def compute_exact_score_top10(home_xg: float, away_xg: float) -> list:
    """Return top 10 most likely exact scores."""
    mat = score_matrix(home_xg, away_xg)
    sorted_scores = sorted(mat.items(), key=lambda x: x[1], reverse=True)[:10]
    return [
        {
            "score": f"{h}-{a}",
            "home_goals": h,
            "away_goals": a,
            "prob": round(p, 4),
            "fair_odds": round(1 / p, 2) if p > 0.001 else 99,
        }
        for (h, a), p in sorted_scores
    ]


def compute_multi_goal(home_xg: float, away_xg: float) -> list:
    """Goal range probabilities using Poisson CDF on total goals."""
    total = home_xg + away_xg
    ranges = [
        ("0-1", 0, 1),
        ("0-2", 0, 2),
        ("1-3", 1, 3),
        ("2-3", 2, 3),
        ("2-4", 2, 4),
        ("3+", 3, None),
        ("4+", 4, None),
    ]
    result = []
    for label, lo, hi in ranges:
        if hi is None:
            p = 1 - poisson_cdf(lo - 1, total)
        else:
            p = poisson_cdf(hi, total) - (poisson_cdf(lo - 1, total) if lo > 0 else 0)
        p = max(0, min(1, p))
        result.append({
            "range": label,
            "prob": round(p, 4),
            "fair_odds": round(1 / p, 2) if p > 0.001 else 99,
        })
    return result


def compute_htft(home_xg: float, away_xg: float) -> list:
    """Half-time / Full-time 9 combinations."""
    fh_home = home_xg * 0.43
    fh_away = away_xg * 0.43
    sh_home = home_xg * 0.57
    sh_away = away_xg * 0.57

    fh_mat = score_matrix(fh_home, fh_away, max_goals=5)
    sh_mat = score_matrix(sh_home, sh_away, max_goals=5)

    def result_key(h, a):
        if h > a:
            return "H"
        elif h < a:
            return "A"
        return "D"

    combos = {}
    for (fh_h, fh_a), fh_p in fh_mat.items():
        ht_res = result_key(fh_h, fh_a)
        for (sh_h, sh_a), sh_p in sh_mat.items():
            ft_res = result_key(fh_h + sh_h, fh_a + sh_a)
            key = f"{ht_res}/{ft_res}"
            combos[key] = combos.get(key, 0) + fh_p * sh_p

    # Fixed order
    order = ["H/H", "H/D", "H/A", "D/H", "D/D", "D/A", "A/H", "A/D", "A/A"]
    result = []
    for key in order:
        p = combos.get(key, 0)
        result.append({
            "combo": key,
            "prob": round(p, 4),
            "fair_odds": round(1 / p, 2) if p > 0.001 else 99,
        })
    return result


def compute_win_to_nil(home_xg: float, away_xg: float, probs: dict) -> dict:
    """Win to nil: team wins AND opponent scores 0."""
    p_home_cs = poisson_pmf(0, away_xg)  # P(away scores 0)
    p_away_cs = poisson_pmf(0, home_xg)  # P(home scores 0)
    p_home_win = probs.get("home", 0.33)
    p_away_win = probs.get("away", 0.33)

    # P(win to nil) ~ P(win) * P(opponent clean sheet), adjusted via score matrix
    mat = score_matrix(home_xg, away_xg)
    home_wtn = sum(v for (h, a), v in mat.items() if h > a and a == 0)
    away_wtn = sum(v for (h, a), v in mat.items() if a > h and h == 0)

    def safe_odds(p):
        return round(1 / p, 2) if p > 0.001 else 99

    return {
        "home_win_to_nil": {"prob": round(home_wtn, 4), "fair_odds": safe_odds(home_wtn)},
        "away_win_to_nil": {"prob": round(away_wtn, 4), "fair_odds": safe_odds(away_wtn)},
        "home_clean_sheet": {"prob": round(p_home_cs, 4), "fair_odds": safe_odds(p_home_cs)},
        "away_clean_sheet": {"prob": round(p_away_cs, 4), "fair_odds": safe_odds(p_away_cs)},
    }


def compute_goal_in_both_halves(home_xg: float, away_xg: float) -> dict:
    """Goal in both halves, highest scoring half, team to score in both halves."""
    fh_total = (home_xg + away_xg) * 0.43
    sh_total = (home_xg + away_xg) * 0.57
    fh_home = home_xg * 0.43
    fh_away = away_xg * 0.43
    sh_home = home_xg * 0.57
    sh_away = away_xg * 0.57

    # P(at least 1 goal in each half)
    p_1h_goal = 1 - poisson_pmf(0, fh_total)
    p_2h_goal = 1 - poisson_pmf(0, sh_total)
    p_goal_both_halves = p_1h_goal * p_2h_goal

    # Home scores in both halves
    p_home_1h = 1 - poisson_pmf(0, fh_home)
    p_home_2h = 1 - poisson_pmf(0, sh_home)
    p_home_both = p_home_1h * p_home_2h

    # Away scores in both halves
    p_away_1h = 1 - poisson_pmf(0, fh_away)
    p_away_2h = 1 - poisson_pmf(0, sh_away)
    p_away_both = p_away_1h * p_away_2h

    # Highest scoring half (compare expected goals)
    # Use Poisson to compute P(1H goals > 2H goals), P(equal), P(2H > 1H)
    fh_mat = {k: poisson_pmf(k, fh_total) for k in range(8)}
    sh_mat = {k: poisson_pmf(k, sh_total) for k in range(8)}
    p_1h_higher = sum(fh_mat[i] * sh_mat[j] for i in range(8) for j in range(8) if i > j)
    p_equal = sum(fh_mat[i] * sh_mat[j] for i in range(8) for j in range(8) if i == j)
    p_2h_higher = sum(fh_mat[i] * sh_mat[j] for i in range(8) for j in range(8) if j > i)

    def safe_odds(p):
        return round(1 / p, 2) if p > 0.001 else 99

    return {
        "goal_both_halves_yes": {"prob": round(p_goal_both_halves, 4), "fair_odds": safe_odds(p_goal_both_halves)},
        "goal_both_halves_no": {"prob": round(1 - p_goal_both_halves, 4), "fair_odds": safe_odds(1 - p_goal_both_halves)},
        "home_scores_both_halves": {"prob": round(p_home_both, 4), "fair_odds": safe_odds(p_home_both)},
        "away_scores_both_halves": {"prob": round(p_away_both, 4), "fair_odds": safe_odds(p_away_both)},
        "highest_scoring_half": {
            "1st_half": {"prob": round(p_1h_higher, 4), "fair_odds": safe_odds(p_1h_higher)},
            "equal": {"prob": round(p_equal, 4), "fair_odds": safe_odds(p_equal)},
            "2nd_half": {"prob": round(p_2h_higher, 4), "fair_odds": safe_odds(p_2h_higher)},
        },
    }


def compute_odd_even(home_xg: float, away_xg: float) -> dict:
    """Odd/Even total goals from score matrix."""
    mat = score_matrix(home_xg, away_xg)
    p_odd = sum(v for (h, a), v in mat.items() if (h + a) % 2 == 1)
    p_even = sum(v for (h, a), v in mat.items() if (h + a) % 2 == 0)

    def safe_odds(p):
        return round(1 / p, 2) if p > 0.001 else 99

    return {
        "odd": {"prob": round(p_odd, 4), "fair_odds": safe_odds(p_odd)},
        "even": {"prob": round(p_even, 4), "fair_odds": safe_odds(p_even)},
    }


def compute_winning_margin(home_xg: float, away_xg: float) -> dict:
    """Exact winning margin: 1 goal, 2 goals, 3+ goals, draw."""
    mat = score_matrix(home_xg, away_xg)

    margins = {}
    for (h, a), p in mat.items():
        diff = h - a
        margins[diff] = margins.get(diff, 0) + p

    p_draw = margins.get(0, 0)
    p_home_1 = margins.get(1, 0)
    p_home_2 = margins.get(2, 0)
    p_home_3plus = sum(v for k, v in margins.items() if k >= 3)
    p_away_1 = margins.get(-1, 0)
    p_away_2 = margins.get(-2, 0)
    p_away_3plus = sum(v for k, v in margins.items() if k <= -3)

    def safe_odds(p):
        return round(1 / p, 2) if p > 0.001 else 99

    return {
        "draw": {"prob": round(p_draw, 4), "fair_odds": safe_odds(p_draw)},
        "home_by_1": {"prob": round(p_home_1, 4), "fair_odds": safe_odds(p_home_1)},
        "home_by_2": {"prob": round(p_home_2, 4), "fair_odds": safe_odds(p_home_2)},
        "home_by_3_plus": {"prob": round(p_home_3plus, 4), "fair_odds": safe_odds(p_home_3plus)},
        "away_by_1": {"prob": round(p_away_1, 4), "fair_odds": safe_odds(p_away_1)},
        "away_by_2": {"prob": round(p_away_2, 4), "fair_odds": safe_odds(p_away_2)},
        "away_by_3_plus": {"prob": round(p_away_3plus, 4), "fair_odds": safe_odds(p_away_3plus)},
    }


def compute_team_to_score_first(home_xg: float, away_xg: float) -> dict:
    """Team to score first based on xG rate per minute.

    Uses exponential distribution: P(team scores first) = xG_rate / total_rate.
    P(no goal) = P(0 goals in match).
    """
    total_xg = home_xg + away_xg
    if total_xg <= 0:
        return {"home": {"prob": 0.33}, "away": {"prob": 0.33}, "no_goal": {"prob": 0.34}}

    p_no_goal = poisson_pmf(0, total_xg)
    p_at_least_one = 1 - p_no_goal
    # Given a goal is scored, team's share is proportional to xG
    p_home_first = p_at_least_one * (home_xg / total_xg)
    p_away_first = p_at_least_one * (away_xg / total_xg)

    def safe_odds(p):
        return round(1 / p, 2) if p > 0.001 else 99

    return {
        "home": {"prob": round(p_home_first, 4), "fair_odds": safe_odds(p_home_first)},
        "away": {"prob": round(p_away_first, 4), "fair_odds": safe_odds(p_away_first)},
        "no_goal": {"prob": round(p_no_goal, 4), "fair_odds": safe_odds(p_no_goal)},
    }


def compute_race_to_goals(home_xg: float, away_xg: float) -> dict:
    """Race to X goals (1, 2, 3). Uses cumulative Poisson timing."""
    mat = score_matrix(home_xg, away_xg, max_goals=8)

    def safe_odds(p):
        return round(1 / p, 2) if p > 0.001 else 99

    results = {}
    for target in [1, 2, 3]:
        # P(home reaches target first): sum all (h,a) where h >= target and the implied
        # timing has home reaching target before away. Approximation: use score matrix.
        p_home = sum(v for (h, a), v in mat.items() if h >= target and a < target)
        p_away = sum(v for (h, a), v in mat.items() if a >= target and h < target)
        p_both = sum(v for (h, a), v in mat.items() if h >= target and a >= target)
        p_neither = sum(v for (h, a), v in mat.items() if h < target and a < target)

        # When both reach target, split by xG rate
        total_xg = home_xg + away_xg
        if total_xg > 0 and p_both > 0:
            p_home += p_both * (home_xg / total_xg)
            p_away += p_both * (away_xg / total_xg)

        results[f"race_to_{target}"] = {
            "home": {"prob": round(p_home, 4), "fair_odds": safe_odds(p_home)},
            "away": {"prob": round(p_away, 4), "fair_odds": safe_odds(p_away)},
            "neither": {"prob": round(p_neither, 4), "fair_odds": safe_odds(p_neither)},
        }

    return results


def compute_second_half_result(home_xg: float, away_xg: float) -> dict:
    """Second-half result (independent of first half)."""
    sh_home = home_xg * 0.57
    sh_away = away_xg * 0.57
    mat = score_matrix(sh_home, sh_away, max_goals=5)

    p_home = sum(v for (h, a), v in mat.items() if h > a)
    p_draw = sum(v for (h, a), v in mat.items() if h == a)
    p_away = sum(v for (h, a), v in mat.items() if h < a)

    def safe_odds(p):
        return round(1 / p, 2) if p > 0.001 else 99

    return {
        "home": {"prob": round(p_home, 4), "fair_odds": safe_odds(p_home)},
        "draw": {"prob": round(p_draw, 4), "fair_odds": safe_odds(p_draw)},
        "away": {"prob": round(p_away, 4), "fair_odds": safe_odds(p_away)},
    }


def compute_penalty_probability(home_xg: float, away_xg: float) -> dict:
    """Penalty in match probability.

    Serie A average: ~0.25 penalties per match. Adjusted by total xG
    (more attacking play = more box entries = more penalties).
    """
    base_pen_rate = 0.25  # Serie A avg penalties per match
    xg_factor = (home_xg + away_xg) / 2.6  # 2.6 is avg total xG
    adj_pen_rate = base_pen_rate * max(0.5, min(1.5, xg_factor))
    p_penalty = 1 - poisson_pmf(0, adj_pen_rate)

    def safe_odds(p):
        return round(1 / p, 2) if p > 0.001 else 99

    return {
        "penalty_yes": {"prob": round(p_penalty, 4), "fair_odds": safe_odds(p_penalty)},
        "penalty_no": {"prob": round(1 - p_penalty, 4), "fair_odds": safe_odds(1 - p_penalty)},
    }


def compute_cards_by_half(expected_cards: float) -> dict:
    """Cards split by half: 1H gets ~45% of cards, 2H gets ~55%.

    Also computes O/U 1.5 and 2.5 per half.
    """
    fh_cards = expected_cards * 0.45
    sh_cards = expected_cards * 0.55

    def safe_odds(p):
        return round(1 / p, 2) if p > 0.001 else 99

    return {
        "first_half": {
            "expected": round(fh_cards, 2),
            "over_1_5": {"prob": round(1 - poisson_cdf(1, fh_cards), 4),
                         "fair_odds": safe_odds(1 - poisson_cdf(1, fh_cards))},
            "over_2_5": {"prob": round(1 - poisson_cdf(2, fh_cards), 4),
                         "fair_odds": safe_odds(1 - poisson_cdf(2, fh_cards))},
        },
        "second_half": {
            "expected": round(sh_cards, 2),
            "over_1_5": {"prob": round(1 - poisson_cdf(1, sh_cards), 4),
                         "fair_odds": safe_odds(1 - poisson_cdf(1, sh_cards))},
            "over_2_5": {"prob": round(1 - poisson_cdf(2, sh_cards), 4),
                         "fair_odds": safe_odds(1 - poisson_cdf(2, sh_cards))},
        },
        "more_cards_half": {
            "first_half": {"prob": round(sum(
                poisson_pmf(i, fh_cards) * poisson_pmf(j, sh_cards)
                for i in range(10) for j in range(10) if i > j
            ), 4)},
            "second_half": {"prob": round(sum(
                poisson_pmf(i, fh_cards) * poisson_pmf(j, sh_cards)
                for i in range(10) for j in range(10) if j > i
            ), 4)},
            "equal": {"prob": round(sum(
                poisson_pmf(i, fh_cards) * poisson_pmf(j, sh_cards)
                for i in range(10) for j in range(10) if i == j
            ), 4)},
        },
    }


def compute_first_card_timing(expected_cards: float) -> dict:
    """Probability of first card before minute X.

    Models card arrival as a Poisson process: inter-arrival time is
    exponential with rate = expected_cards / 90 per minute.
    P(first card before minute t) = 1 - exp(-rate * t).
    """
    rate = expected_cards / 90.0  # cards per minute

    def safe_odds(p):
        return round(1 / p, 2) if p > 0.001 else 99

    result = {}
    for minute in [15, 20, 25, 30, 45]:
        p = 1 - math.exp(-rate * minute)
        result[f"before_{minute}_min"] = {
            "prob": round(p, 4),
            "fair_odds": safe_odds(p),
        }

    return result


def compute_european_handicap(home_xg: float, away_xg: float) -> dict:
    """European handicap (integer handicap, 3-way result).

    Computes 1X2 probabilities with handicap applied:
    Home -1, Home -2, Home +1, Away -1, Away +1.
    """
    mat = score_matrix(home_xg, away_xg, max_goals=8)

    def safe_odds(p):
        return round(1 / p, 2) if p > 0.001 else 99

    handicaps = {}
    for hcap in [-2, -1, 1, 2]:
        # Apply handicap to home team score
        p_home = sum(v for (h, a), v in mat.items() if (h + hcap) > a)
        p_draw = sum(v for (h, a), v in mat.items() if (h + hcap) == a)
        p_away = sum(v for (h, a), v in mat.items() if (h + hcap) < a)

        label = f"home_{'+' if hcap > 0 else ''}{hcap}"
        handicaps[label] = {
            "home": {"prob": round(p_home, 4), "fair_odds": safe_odds(p_home)},
            "draw": {"prob": round(p_draw, 4), "fair_odds": safe_odds(p_draw)},
            "away": {"prob": round(p_away, 4), "fair_odds": safe_odds(p_away)},
        }

    return handicaps


def generate_extended_markets():
    """Load predictions from ALL leagues and compute extended markets for each match."""
    # Load predictions from all league files
    all_predictions = []
    for pred_file in ["predictions.json", "predictions_premier_league.json"]:
        pred_path = DATA_DIR / "upcoming" / pred_file
        if pred_path.exists():
            with open(pred_path) as f:
                pred_data = json.load(f)
            preds = pred_data.get("predictions", [])
            log.info("Loaded %d predictions from %s", len(preds), pred_file)
            all_predictions.extend(preds)

    if not all_predictions:
        log.warning("No predictions found in any league file")
        return {}

    # Create a fake pred_data structure for backward compat
    pred_data = {"predictions": all_predictions}

    # Load cards predictions for red card market
    cards_data = {}
    cards_path = DATA_DIR / "upcoming" / "cards_predictions.json"
    if cards_path.exists():
        with open(cards_path) as f:
            cards_raw = json.load(f)
        cards_list = cards_raw if isinstance(cards_raw, list) else cards_raw.get("predictions", [])
        for cp in cards_list:
            cards_data[cp["match"]] = cp

    # Load corners predictions for team corners
    corners_data = {}
    corners_path = DATA_DIR / "upcoming" / "corners_predictions.json"
    if corners_path.exists():
        with open(corners_path) as f:
            corners_raw = json.load(f)
        corners_list = corners_raw if isinstance(corners_raw, list) else corners_raw.get("predictions", [])
        for cp in corners_list:
            corners_data[cp["match"]] = cp

    predictions = pred_data.get("predictions", [])
    matches = {}

    for pred in predictions:
        match = pred.get("match", "")
        home_xg = pred.get("home_xg", 1.3)
        away_xg = pred.get("away_xg", 1.0)
        probs = pred.get("probabilities", {})

        # Red card market from cards model
        card_pred = cards_data.get(match, {})
        expected_reds = card_pred.get("expected_reds", 0.15)
        p_red_card = 1 - poisson_pmf(0, expected_reds)
        expected_cards = card_pred.get("expected_cards", 4.5)

        # Team cards split (proportional to team discipline)
        home_team = pred.get("home_team", "")
        away_team = pred.get("away_team", "")
        total_cards = expected_cards
        # Home teams get ~52% of cards on average (away slightly less disciplined but ref bias)
        home_share = 0.48
        exp_home_cards = total_cards * home_share
        exp_away_cards = total_cards * (1 - home_share)

        # Team corners from corners model
        corner_pred = corners_data.get(match, {})
        exp_home_corners = corner_pred.get("expected_home_corners", 5.0)
        exp_away_corners = corner_pred.get("expected_away_corners", 4.8)

        def safe_odds(p):
            return round(1 / p, 2) if p > 0.001 else 99

        matches[match] = {
            "match": match,
            "home_team": home_team,
            "away_team": away_team,
            "home_xg": home_xg,
            "away_xg": away_xg,
            # Existing markets
            "double_chance": compute_double_chance(probs),
            "team_totals": compute_team_totals(home_xg, away_xg),
            "first_half": compute_first_half(home_xg, away_xg),
            "exact_score": compute_exact_score_top10(home_xg, away_xg),
            "multi_goal": compute_multi_goal(home_xg, away_xg),
            "htft": compute_htft(home_xg, away_xg),
            # NEW Tier 1 markets
            "win_to_nil": compute_win_to_nil(home_xg, away_xg, probs),
            "goal_both_halves": compute_goal_in_both_halves(home_xg, away_xg),
            "odd_even": compute_odd_even(home_xg, away_xg),
            "winning_margin": compute_winning_margin(home_xg, away_xg),
            # NEW Tier 2 markets
            "team_to_score_first": compute_team_to_score_first(home_xg, away_xg),
            "race_to_goals": compute_race_to_goals(home_xg, away_xg),
            "second_half_result": compute_second_half_result(home_xg, away_xg),
            "penalty_in_match": compute_penalty_probability(home_xg, away_xg),
            # Red card market
            "red_card": {
                "yes": {"prob": round(p_red_card, 4), "fair_odds": safe_odds(p_red_card)},
                "no": {"prob": round(1 - p_red_card, 4), "fair_odds": safe_odds(1 - p_red_card)},
            },
            # Team cards
            "team_cards": {
                "home_expected": round(exp_home_cards, 2),
                "away_expected": round(exp_away_cards, 2),
                "home_over_1_5": {"prob": round(1 - poisson_cdf(1, exp_home_cards), 4),
                                  "fair_odds": safe_odds(1 - poisson_cdf(1, exp_home_cards))},
                "home_over_2_5": {"prob": round(1 - poisson_cdf(2, exp_home_cards), 4),
                                  "fair_odds": safe_odds(1 - poisson_cdf(2, exp_home_cards))},
                "away_over_1_5": {"prob": round(1 - poisson_cdf(1, exp_away_cards), 4),
                                  "fair_odds": safe_odds(1 - poisson_cdf(1, exp_away_cards))},
                "away_over_2_5": {"prob": round(1 - poisson_cdf(2, exp_away_cards), 4),
                                  "fair_odds": safe_odds(1 - poisson_cdf(2, exp_away_cards))},
                "home_more_cards": {"prob": round(sum(
                    poisson_pmf(i, exp_home_cards) * poisson_pmf(j, exp_away_cards)
                    for i in range(10) for j in range(10) if i > j
                ), 4)},
            },
            # Team corners
            "team_corners": {
                "home_expected": round(exp_home_corners, 2),
                "away_expected": round(exp_away_corners, 2),
                "home_over_4_5": {"prob": round(1 - poisson_cdf(4, exp_home_corners), 4),
                                  "fair_odds": safe_odds(1 - poisson_cdf(4, exp_home_corners))},
                "home_over_5_5": {"prob": round(1 - poisson_cdf(5, exp_home_corners), 4),
                                  "fair_odds": safe_odds(1 - poisson_cdf(5, exp_home_corners))},
                "away_over_4_5": {"prob": round(1 - poisson_cdf(4, exp_away_corners), 4),
                                  "fair_odds": safe_odds(1 - poisson_cdf(4, exp_away_corners))},
                "away_over_5_5": {"prob": round(1 - poisson_cdf(5, exp_away_corners), 4),
                                  "fair_odds": safe_odds(1 - poisson_cdf(5, exp_away_corners))},
                "corner_handicap": round(exp_home_corners - exp_away_corners, 2),
                "home_more_corners": {"prob": round(sum(
                    poisson_pmf(i, exp_home_corners) * poisson_pmf(j, exp_away_corners)
                    for i in range(15) for j in range(15) if i > j
                ), 4)},
            },
            # Booking points ranges
            "booking_points": _compute_booking_points_ranges(expected_cards),
            # NEW Tier 2 additions
            "cards_by_half": compute_cards_by_half(expected_cards),
            "first_card_timing": compute_first_card_timing(expected_cards),
            "european_handicap": compute_european_handicap(home_xg, away_xg),
        }

    output = {
        "generated_at": datetime.now().isoformat(),
        "match_count": len(matches),
        "markets_available": [
            "double_chance", "team_totals", "first_half", "exact_score",
            "multi_goal", "htft", "win_to_nil", "goal_both_halves",
            "odd_even", "winning_margin", "team_to_score_first",
            "race_to_goals", "second_half_result", "penalty_in_match",
            "red_card", "team_cards", "team_corners", "booking_points",
            "cards_by_half", "first_card_timing", "european_handicap",
        ],
        "matches": matches,
    }

    out_path = DATA_DIR / "upcoming" / "extended_markets.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    log.info("Extended markets generated for %d matches (%d market types) -> %s",
             len(matches), len(output['markets_available']), out_path)
    return output


def _compute_booking_points_ranges(expected_cards: float) -> dict:
    """Booking points ranges (10 per yellow, 25 per red).

    Approximate: expected_bp ~ expected_cards * 10.4 (avg 10 per yellow, 25 per red).
    """
    avg_bp_per_card = 10.4  # weighted by yellow/red ratio
    expected_bp = expected_cards * avg_bp_per_card

    def safe_odds(p):
        return round(1 / p, 2) if p > 0.001 else 99

    # Use normal distribution approximation for booking points
    # Std dev ~ 12 BP per match
    bp_std = 12.0
    ranges = {}
    for line in [20.5, 30.5, 40.5, 50.5, 60.5]:
        # P(BP > line) using Poisson approximation on cards then scaling
        # Simpler: use cards Poisson scaled by avg_bp_per_card
        k_cards = line / avg_bp_per_card
        p_over = 1 - poisson_cdf(int(k_cards), expected_cards)
        ranges[f"over_{line}"] = {"prob": round(p_over, 4), "fair_odds": safe_odds(p_over)}
        ranges[f"under_{line}"] = {"prob": round(1 - p_over, 4), "fair_odds": safe_odds(1 - p_over)}

    ranges["expected"] = round(expected_bp, 1)
    return ranges


if __name__ == "__main__":
    generate_extended_markets()
