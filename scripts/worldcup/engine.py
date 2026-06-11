"""World Cup 2026 prediction engine: Elo ratings + Poisson goal model.

Data: data/worldcup/international_results.csv (martj42/international_results,
full A-international history). Ratings follow the World Football Elo
convention (eloratings.net): K-factor by tournament class, goal-difference
multiplier, +100 home advantage in win expectancy for non-neutral venues.

Goal means come from a Poisson GLM on Elo difference with a venue term and a
friendly indicator, fit with time-decay weights. Elo features are leak-free
by construction (each match's pre-match ratings only see earlier matches).

This model ships only because scripts/worldcup/backtest.py shows positive
Brier skill over the neutral-competitive base rate on held-out tournaments.
Always-pick-higher-Elo accuracy is reported alongside for context: the GLM's
argmax picks nearly coincide with Elo picks — its edge is probability
calibration, not pick accuracy.
Live numbers: data/worldcup/model_metadata.json — never quote from docs.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "worldcup"
RESULTS_CSV = DATA_DIR / "international_results.csv"

INITIAL_ELO = 1500.0
ELO_HOME_ADV = 100.0

K_WORLD_CUP = 60.0
K_CONTINENTAL_FINAL = 50.0
K_QUALIFIER_MAJOR = 40.0
K_OTHER = 30.0
K_FRIENDLY = 20.0

# GLM training window and recency weighting
TRAIN_START = "2010-01-01"
HALF_LIFE_DAYS = 1460.0  # ~4 years, mirrors the Serie A time-decay convention

MAX_GOALS = 12  # score-grid truncation; P(>12 goals per team) is negligible

# Fixture feeds use FIFA display names; the results dataset (martj42) uses
# its own conventions. Map display -> canonical (results-dataset) name.
# NOTE: plain "Congo" in the dataset is Congo-Brazzaville, a different team.
FIXTURE_TO_CANON = {
    "USA": "United States",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",  # unaccented variant (some Sofascore surfaces)
    "Cabo Verde": "Cape Verde",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Czechia": "Czech Republic",
    "Congo DR": "DR Congo",
    "Curacao": "Curaçao",  # cedilla-less variant -> csv spelling
}


def canon_team(name: str) -> str:
    """Canonical results-dataset spelling for a fixture display name."""
    clean = unicodedata.normalize("NFC", name.strip())
    return FIXTURE_TO_CANON.get(clean, clean)


def atomic_write_json(path: Path, obj: object) -> None:
    """tmp + rename — a crash mid-write can never truncate the target.
    The automation rewrites several JSONs every 2h for 6 weeks; refresh.py's
    subprocess timeout SIGKILLs children, so plain write_text WILL eventually
    be interrupted."""
    import json

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(path)


def read_json_safe(path: Path, default: object, *, quarantine: bool = False) -> object:
    """Fail-soft JSON read: corruption returns the default instead of
    crash-looping every scheduled run. With quarantine=True the corrupt file
    is preserved under a timestamped name for post-mortem (used for the one
    artifact that cannot be re-derived: the pre-kickoff archive)."""
    import json
    from datetime import UTC, datetime

    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        if quarantine:
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
            path.replace(path.with_suffix(f".corrupt-{stamp}{path.suffix}"))
        return default


_CONTINENTAL_FINALS = (
    "uefa euro",
    "copa américa",
    "copa america",
    "african cup of nations",
    "africa cup of nations",
    "afc asian cup",
    "asian cup",
    "gold cup",
    "confederations cup",
    "oceania nations cup",  # OFC continental final (matters for New Zealand)
    "concacaf championship",  # pre-Gold Cup continental final, 1963-1989
)


def k_factor(tournament: str) -> float:
    """eloratings.net K-factor class for a tournament label."""
    t = tournament.strip().lower()
    if "friendly" in t:
        return K_FRIENDLY
    if "qualification" in t or "qualifier" in t:
        return K_QUALIFIER_MAJOR
    if t == "fifa world cup":
        return K_WORLD_CUP
    if any(name in t for name in _CONTINENTAL_FINALS):
        return K_CONTINENTAL_FINAL
    if "nations league" in t:
        return K_QUALIFIER_MAJOR
    return K_OTHER


def goal_multiplier(margin: int) -> float:
    """eloratings.net goal-difference multiplier."""
    margin = abs(margin)
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11.0 + margin) / 8.0


def expected_score(elo_diff: float) -> float:
    """Win expectancy for the side whose rating advantage is ``elo_diff``."""
    return 1.0 / (10.0 ** (-elo_diff / 400.0) + 1.0)


def load_results(path: Path = RESULTS_CSV) -> pd.DataFrame:
    """Load and sanitize the international results dataset, oldest first."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(bool)
    return df.sort_values("date", kind="stable").reset_index(drop=True)


def elo_history(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Single chronological Elo pass.

    Returns the input frame with ``elo_home_pre``/``elo_away_pre`` columns
    (each match's PRE-match ratings — leak-free features) and the final
    post-history ratings dict.
    """
    ratings: dict[str, float] = {}
    home_pre = np.empty(len(df), dtype=np.float64)
    away_pre = np.empty(len(df), dtype=np.float64)

    for i, row in enumerate(df.itertuples(index=False)):
        rh = ratings.get(row.home_team, INITIAL_ELO)
        ra = ratings.get(row.away_team, INITIAL_ELO)
        home_pre[i] = rh
        away_pre[i] = ra

        adv = 0.0 if row.neutral else ELO_HOME_ADV
        we = expected_score(rh + adv - ra)
        if row.home_score > row.away_score:
            w = 1.0
        elif row.home_score == row.away_score:
            w = 0.5
        else:
            w = 0.0
        k = k_factor(row.tournament) * goal_multiplier(
            row.home_score - row.away_score
        )
        delta = k * (w - we)
        ratings[row.home_team] = rh + delta
        ratings[row.away_team] = ra - delta

    out = df.copy()
    out["elo_home_pre"] = home_pre
    out["elo_away_pre"] = away_pre
    return out, ratings


# Major final tournaments have a different goal environment than qualifiers
# (no dead rubbers vs minnows); the GLM carries an explicit covariate for it.
_MAJOR_FINALS = (
    "FIFA World Cup",
    "UEFA Euro",
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "Gold Cup",
)


def is_major_final(tournament: str) -> bool:
    return tournament.strip() in _MAJOR_FINALS


@dataclass(frozen=True)
class GoalModel:
    """Poisson GLM: log(goals) = b0 + b_diff*(elo_diff/100) + b_home
    + b_friendly + b_major."""

    b0: float
    b_diff: float
    b_home: float
    b_friendly: float
    b_major: float = 0.0

    def lam(
        self,
        elo_team: float,
        elo_opp: float,
        *,
        at_home: bool,
        friendly: bool = False,
        major: bool = False,
    ) -> float:
        z = (
            self.b0
            + self.b_diff * (elo_team - elo_opp) / 100.0
            + (self.b_home if at_home else 0.0)
            + (self.b_friendly if friendly else 0.0)
            + (self.b_major if major else 0.0)
        )
        return float(math.exp(z))


def _glm_design(
    hist: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two perspective rows per match -> (X, y, sample_weight)."""
    is_friendly = (
        hist["tournament"].str.strip().str.lower().str.contains("friendly")
    ).to_numpy(dtype=np.float64)
    is_major = hist["tournament"].map(is_major_final).to_numpy(dtype=np.float64)
    not_neutral = (~hist["neutral"]).to_numpy(dtype=np.float64)

    diff = (hist["elo_home_pre"] - hist["elo_away_pre"]).to_numpy() / 100.0
    age_days = (
        (hist["date"].max() - hist["date"]).dt.days.to_numpy(dtype=np.float64)
    )
    weight = np.float_power(0.5, age_days / HALF_LIFE_DAYS)

    # Home-team perspective rows, then away-team perspective rows.
    x_home = np.column_stack([diff, not_neutral, is_friendly, is_major])
    x_away = np.column_stack([-diff, np.zeros_like(diff), is_friendly, is_major])
    x = np.vstack([x_home, x_away])
    y = np.concatenate(
        [hist["home_score"].to_numpy(np.float64), hist["away_score"].to_numpy(np.float64)]
    )
    w = np.concatenate([weight, weight])
    return x, y, w


def fit_goal_model(
    hist: pd.DataFrame,
    *,
    train_start: str = TRAIN_START,
    train_end: str | None = None,
) -> GoalModel:
    """Fit the Poisson GLM on matches in [train_start, train_end)."""
    from sklearn.linear_model import PoissonRegressor

    mask = hist["date"] >= pd.Timestamp(train_start)
    if train_end is not None:
        mask &= hist["date"] < pd.Timestamp(train_end)
    sub = hist.loc[mask]
    if len(sub) < 500:
        raise ValueError(
            f"Only {len(sub)} matches in training window — refusing to fit."
        )

    x, y, w = _glm_design(sub)
    reg = PoissonRegressor(alpha=1e-9, max_iter=1000)
    reg.fit(x, y, sample_weight=w)
    return GoalModel(
        b0=float(reg.intercept_),
        b_diff=float(reg.coef_[0]),
        b_home=float(reg.coef_[1]),
        b_friendly=float(reg.coef_[2]),
        b_major=float(reg.coef_[3]),
    )


@dataclass(frozen=True)
class DCModel:
    """Dixon-Coles-style attack/defense ratings (time-decayed Poisson MLE).

    log lam_for = c + att[team] - deff[opponent] + home*at_home
    A genuinely different estimator from the Elo-difference GLM: it sees
    asymmetric team profiles (strong attack / weak defense) that a single
    rating cannot. L2 shrinkage pulls rarely-seen teams toward average.
    """

    c: float
    home: float
    att: dict[str, float]
    deff: dict[str, float]

    def lam(self, team: str, opp: str, *, at_home: bool) -> float | None:
        if team not in self.att or opp not in self.deff:
            return None
        z = self.c + self.att[team] - self.deff[opp] + (self.home if at_home else 0.0)
        return float(math.exp(z))


def fit_dc_model(
    hist: pd.DataFrame,
    *,
    train_start: str = TRAIN_START,
    train_end: str | None = None,
    reg: float = 2.0,
    half_life_days: float = HALF_LIFE_DAYS,
    min_matches: int = 500,
) -> DCModel:
    """Fit attack/defense ratings by weighted Poisson MLE (L-BFGS, analytic
    gradient). ~2x n_teams + 2 parameters; converges in a few seconds."""
    from scipy.optimize import minimize

    mask = hist["date"] >= pd.Timestamp(train_start)
    if train_end is not None:
        mask &= hist["date"] < pd.Timestamp(train_end)
    sub = hist.loc[mask]
    if len(sub) < min_matches:
        raise ValueError(f"Only {len(sub)} matches in DC training window.")

    teams = sorted(set(sub["home_team"]) | set(sub["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = sub["home_team"].map(idx).to_numpy(dtype=np.int64)
    ai = sub["away_team"].map(idx).to_numpy(dtype=np.int64)
    yh = sub["home_score"].to_numpy(dtype=np.float64)
    ya = sub["away_score"].to_numpy(dtype=np.float64)
    home_ind = (~sub["neutral"]).to_numpy(dtype=np.float64)
    age = (sub["date"].max() - sub["date"]).dt.days.to_numpy(dtype=np.float64)
    w = np.float_power(0.5, age / half_life_days)

    def unpack(theta: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
        return theta[0], theta[1], theta[2 : 2 + n], theta[2 + n :]

    def nll_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        c, h, att, deff = unpack(theta)
        zh = c + att[hi] - deff[ai] + h * home_ind
        za = c + att[ai] - deff[hi]
        lh, la = np.exp(zh), np.exp(za)
        nll = float(
            (w * (lh - yh * zh)).sum()
            + (w * (la - ya * za)).sum()
            + reg * (att @ att + deff @ deff)
        )
        rh = w * (lh - yh)  # d nll / d zh
        ra = w * (la - ya)
        g = np.zeros_like(theta)
        g[0] = rh.sum() + ra.sum()
        g[1] = (rh * home_ind).sum()
        g_att = np.zeros(n)
        g_def = np.zeros(n)
        np.add.at(g_att, hi, rh)
        np.add.at(g_att, ai, ra)
        np.add.at(g_def, ai, -rh)
        np.add.at(g_def, hi, -ra)
        g[2 : 2 + n] = g_att + 2.0 * reg * att
        g[2 + n :] = g_def + 2.0 * reg * deff
        return nll, g

    theta0 = np.zeros(2 + 2 * n)
    theta0[0] = math.log(max(float((yh.sum() + ya.sum()) / (2 * len(sub))), 0.1))
    theta0[1] = 0.25
    res = minimize(nll_grad, theta0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 500})
    c, h, att, deff = unpack(res.x)
    return DCModel(
        c=float(c),
        home=float(h),
        att={t: float(att[idx[t]]) for t in teams},
        deff={t: float(deff[idx[t]]) for t in teams},
    )


def blend_lambdas(
    lam_glm: float, lam_dc: float | None, weight_glm: float
) -> float:
    """Geometric blend of the two estimators; falls back to the GLM when the
    DC model has never seen a team."""
    if lam_dc is None:
        return lam_glm
    return float(
        math.exp(
            weight_glm * math.log(lam_glm) + (1.0 - weight_glm) * math.log(lam_dc)
        )
    )


def score_matrix(lam_home: float, lam_away: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """Independent-Poisson scoreline grid P[h, a].

    Mirrors scripts.betting.extended_markets.score_matrix but returns a
    normalized numpy grid for vectorized model evaluation; the dict version
    there feeds the market layer and supports bivariate rho.
    """
    goals = np.arange(max_goals + 1)
    log_fact = np.cumsum(np.log(np.maximum(goals, 1)))
    ph = np.exp(goals * math.log(lam_home) - lam_home - log_fact)
    pa = np.exp(goals * math.log(lam_away) - lam_away - log_fact)
    grid = np.outer(ph, pa)
    return grid / grid.sum()


def one_x_two(lam_home: float, lam_away: float) -> tuple[float, float, float]:
    """(P_home, P_draw, P_away) from the independent-Poisson grid."""
    grid = score_matrix(lam_home, lam_away)
    p_home = float(np.tril(grid, -1).sum())
    p_draw = float(np.trace(grid))
    p_away = float(np.triu(grid, 1).sum())
    return p_home, p_draw, p_away


def tilt_lambdas(
    lam_home: float,
    lam_away: float,
    target: tuple[float, float, float],
) -> tuple[float, float]:
    """Tilt a lambda pair (lam*k, lam/k — total goals preserved) so its 1X2
    distribution best matches ``target`` (least squares over a log-k grid).
    Used to fold market-blended 1X2 probabilities back into the score grid
    so EVERY derived market stays coherent with the blended view."""
    log_ks = np.linspace(-0.9, 0.9, 61)
    best_k, best_err = 1.0, float("inf")
    for log_k in log_ks:
        k = math.exp(log_k)
        p = one_x_two(lam_home * k, lam_away / k)
        err = sum((a - b) ** 2 for a, b in zip(p, target, strict=True))
        if err < best_err:
            best_err, best_k = err, k
    return lam_home * best_k, lam_away / best_k


# Production variant: ens_w0.5_reg1, selected on DEV (WC 2018 + Euro 2020)
# by scripts/worldcup/backtest.py. On the untouched finals it beat the plain
# GLM on Brier and ECE with identical pick accuracy — numbers live in
# data/worldcup/model_metadata.json (never quote from docs).
PRODUCTION_WEIGHT_GLM = 0.5
PRODUCTION_DC_REG = 1.0
PRODUCTION_USE_ENSEMBLE = True


@dataclass
class WorldCupEngine:
    """Final Elo ratings + Elo-GLM + DC attack/defense ensemble."""

    ratings: dict[str, float]
    model: GoalModel
    fitted_through: str
    dc: DCModel | None = None
    weight_glm: float = 1.0

    @classmethod
    def build(cls, results_path: Path = RESULTS_CSV) -> WorldCupEngine:
        df = load_results(results_path)
        hist, ratings = elo_history(df)
        model = fit_goal_model(hist)
        dc = (
            fit_dc_model(hist, reg=PRODUCTION_DC_REG)
            if PRODUCTION_USE_ENSEMBLE
            else None
        )
        return cls(
            ratings=ratings,
            model=model,
            fitted_through=str(df["date"].max().date()),
            dc=dc,
            weight_glm=PRODUCTION_WEIGHT_GLM if PRODUCTION_USE_ENSEMBLE else 1.0,
        )

    def elo(self, team: str) -> float:
        if team not in self.ratings:
            raise KeyError(f"No Elo rating for team {team!r} — check name mapping.")
        return self.ratings[team]

    def lambdas(
        self,
        home: str,
        away: str,
        *,
        home_at_home: bool = False,
        away_at_home: bool = False,
    ) -> tuple[float, float]:
        """Expected goals for a World Cup match (major-tournament environment)."""
        eh, ea = self.elo(home), self.elo(away)
        lam_h = self.model.lam(eh, ea, at_home=home_at_home, major=True)
        lam_a = self.model.lam(ea, eh, at_home=away_at_home, major=True)
        if self.dc is not None:
            lam_h = blend_lambdas(
                lam_h, self.dc.lam(home, away, at_home=home_at_home), self.weight_glm
            )
            lam_a = blend_lambdas(
                lam_a, self.dc.lam(away, home, at_home=away_at_home), self.weight_glm
            )
        return lam_h, lam_a
