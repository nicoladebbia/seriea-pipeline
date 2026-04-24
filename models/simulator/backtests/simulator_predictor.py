"""Predictor adapter that wraps the simulator engine.

Provides SimulatorPredictor, which the Phase 3b backtest harness runs through
the same interface as ProdCatboostBaseline and PoissonFromFeatures — so we can
compare simulator ROI against CatBoost baselines apples-to-apples.

At training time: fit a λ estimator from features (PoissonRegressor on the
winning Phase 0 feature set) + fit Dixon-Coles τ.
At predict time: for each match, simulate with n_trials and emit market probs.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

from models.simulator.base_rates.card_rates import CardRateEstimator
from models.simulator.base_rates.corner_rates import CornerRateEstimator
from models.simulator.base_rates.shot_generator import ShotRateEstimator
from models.simulator.engine.dixon_coles import fit_tau_mle
from models.simulator.engine.simulator import simulate_match

log = logging.getLogger(__name__)


DEFAULT_FEATURES = [
    "home_attack_strength", "away_attack_strength",
    "home_defense_strength", "away_defense_strength",
    "league_avg_goals",
    "home_ss_roll_xg", "away_ss_roll_xg",
    "home_ss_roll_xgot", "away_ss_roll_xgot",
    "home_ss_roll_total_shots", "away_ss_roll_total_shots",
    "home_us_team_xg", "away_us_team_xg",
    "home_us_team_npxg", "away_us_team_npxg",
]


class SimulatorPredictor:
    """Wraps: (λ regressor) + (τ fit) + (Monte Carlo simulator) into a Predictor."""

    name = "simulator"
    version = "v1_phase1"

    def __init__(
        self,
        lambda_features: list[str] | None = None,
        use_dixon_coles: bool = True,
        n_trials: int = 10_000,
        seed: int = 42,
        enable_phase2_rates: bool = True,
    ):
        self.lambda_features = lambda_features or list(DEFAULT_FEATURES)
        self.use_dixon_coles = use_dixon_coles
        self.n_trials = n_trials
        self.seed = seed
        self.enable_phase2_rates = enable_phase2_rates
        self._lambda_home: PoissonRegressor | None = None
        self._lambda_away: PoissonRegressor | None = None
        self._tau: float = 0.0
        self._usable_features: list[str] = []
        self._corner_est: CornerRateEstimator | None = None
        self._card_est: CardRateEstimator | None = None
        self._shot_est: ShotRateEstimator | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        available = [f for f in self.lambda_features if f in train_df.columns]
        if not available:
            log.warning("Simulator: no usable features in training data; skipping fit")
            return
        train = train_df.dropna(subset=available + ["home_score", "away_score"])
        if len(train) < 100:
            log.warning("Simulator: insufficient training rows (%d)", len(train))
            return
        X = train[available].to_numpy(dtype=float)
        self._lambda_home = PoissonRegressor(alpha=0.5, max_iter=1000).fit(X, train["home_score"])
        self._lambda_away = PoissonRegressor(alpha=0.5, max_iter=1000).fit(X, train["away_score"])
        self._usable_features = available

        # Fit τ on same training data
        if self.use_dixon_coles:
            lh_pred = np.clip(self._lambda_home.predict(X), 0.1, 6.0)
            la_pred = np.clip(self._lambda_away.predict(X), 0.1, 6.0)
            self._tau = fit_tau_mle(
                train["home_score"].to_numpy(dtype=int),
                train["away_score"].to_numpy(dtype=int),
                lh_pred, la_pred,
            )
            log.info("Simulator: fit τ=%.4f on %d training rows", self._tau, len(train))
        else:
            self._tau = 0.0

        # Phase 2: fit rate estimators for corners, cards, shots
        if self.enable_phase2_rates:
            self._corner_est = CornerRateEstimator()
            self._corner_est.fit(train_df)
            self._card_est = CardRateEstimator()
            self._card_est.fit(train_df)
            self._shot_est = ShotRateEstimator()
            self._shot_est.fit(train_df)

    def supports(self, market_label: str) -> bool:
        goal_markets = {
            "O/U 0.5", "O/U 1.5", "O/U 2.0", "O/U 2.5", "O/U 3.0", "O/U 3.5", "O/U 4.5",
            "BTTS", "1X2",
            "home_clean_sheet", "away_clean_sheet",
        }
        if market_label in goal_markets:
            return True
        if not self.enable_phase2_rates:
            return False
        # Phase 2 markets
        if market_label.startswith("corners_over_") or market_label.startswith("cards_over_"):
            return True
        if market_label.startswith("home_corners_over_") or market_label.startswith("away_corners_over_"):
            return True
        if market_label.startswith("home_cards_over_") or market_label.startswith("away_cards_over_"):
            return True
        if market_label.startswith("shots_over_") or market_label.startswith("sot_over_"):
            return True
        return False

    def _predict_lambdas(self, eval_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
        if self._lambda_home is None or not self._usable_features:
            return None
        X = eval_df[self._usable_features].fillna(0.0).to_numpy(dtype=float)
        lh = np.clip(self._lambda_home.predict(X), 0.1, 6.0)
        la = np.clip(self._lambda_away.predict(X), 0.1, 6.0)
        return lh, la

    def _simulate_all(self, eval_df: pd.DataFrame) -> list | None:
        pred = self._predict_lambdas(eval_df)
        if pred is None:
            return None
        lh, la = pred

        # Phase 2 rates (optional — None for any unfit estimator)
        if self._corner_est is not None and self._corner_est.is_fit:
            rate_ch, rate_ca = self._corner_est.predict(eval_df)
        else:
            rate_ch, rate_ca = (None, None)
        if self._card_est is not None and self._card_est.is_fit:
            rate_krh, rate_kra = self._card_est.predict(eval_df)
        else:
            rate_krh, rate_kra = (None, None)
        if self._shot_est is not None and self._shot_est.is_fit:
            rate_sh, rate_sa = self._shot_est.predict(eval_df)
            sot_h, sot_a = self._shot_est.sot_ratios()
        else:
            rate_sh, rate_sa = (None, None)
            sot_h, sot_a = (0.35, 0.33)

        sims = []
        for i in range(len(eval_df)):
            match_id = str(eval_df.iloc[i].get("match_id", f"row_{i}"))
            sim = simulate_match(
                lambda_home=float(lh[i]),
                lambda_away=float(la[i]),
                tau=self._tau,
                n_trials=self.n_trials,
                seed=self.seed,
                match_id=match_id,
                corner_rate_home=float(rate_ch[i]) if rate_ch is not None else None,
                corner_rate_away=float(rate_ca[i]) if rate_ca is not None else None,
                card_rate_home=float(rate_krh[i]) if rate_krh is not None else None,
                card_rate_away=float(rate_kra[i]) if rate_kra is not None else None,
                shot_rate_home=float(rate_sh[i]) if rate_sh is not None else None,
                shot_rate_away=float(rate_sa[i]) if rate_sa is not None else None,
                sot_ratio_home=sot_h,
                sot_ratio_away=sot_a,
            )
            sims.append(sim)
        return sims

    def predict_binary(self, eval_df: pd.DataFrame, market_label: str) -> np.ndarray:
        sims = self._simulate_all(eval_df)
        if sims is None:
            return np.array([])
        if market_label.startswith("O/U"):
            line = float(market_label.split(" ")[1])
            return np.array([s.p_over(line) for s in sims])
        if market_label == "BTTS":
            return np.array([s.p_btts(yes=True) for s in sims])
        if market_label == "home_clean_sheet":
            return np.array([s.p_clean_sheet("home") for s in sims])
        if market_label == "away_clean_sheet":
            return np.array([s.p_clean_sheet("away") for s in sims])
        # Phase 2 markets: corners_over_{X}, cards_over_{X}, home_corners_over_{X}, etc.
        if "corners_over_" in market_label:
            # Parse line from trailing "N_M" -> N.M
            tail = market_label.rsplit("_over_", 1)[-1].replace("_", ".")
            try:
                line = float(tail)
            except ValueError:
                return np.array([])
            team = "home" if market_label.startswith("home_") else "away" if market_label.startswith("away_") else "both"
            try:
                return np.array([s.p_corners_over(line, team) for s in sims])
            except AttributeError:
                return np.array([])
        if "cards_over_" in market_label:
            tail = market_label.rsplit("_over_", 1)[-1].replace("_", ".")
            try:
                line = float(tail)
            except ValueError:
                return np.array([])
            team = "home" if market_label.startswith("home_") else "away" if market_label.startswith("away_") else "both"
            try:
                return np.array([s.p_cards_over(line, team) for s in sims])
            except AttributeError:
                return np.array([])
        if market_label.startswith("shots_over_"):
            line = float(market_label.rsplit("_", 1)[-1].replace("_", "."))
            try:
                return np.array([s.p_shots_over(line, "both") for s in sims])
            except AttributeError:
                return np.array([])
        if market_label.startswith("sot_over_"):
            line = float(market_label.rsplit("_", 1)[-1].replace("_", "."))
            try:
                return np.array([s.p_sot_over(line, "both") for s in sims])
            except AttributeError:
                return np.array([])
        return np.array([])

    def predict_multiclass(self, eval_df: pd.DataFrame, market_label: str) -> np.ndarray:
        sims = self._simulate_all(eval_df)
        if sims is None:
            return np.array([]).reshape(0, 3)
        if market_label == "1X2":
            return np.array([[s.p_home_win(), s.p_draw(), s.p_away_win()] for s in sims])
        return np.array([]).reshape(0, 3)

    def classes_for(self, market_label: str) -> tuple[str, ...]:
        if market_label == "1X2":
            return ("H", "D", "A")
        return ()
